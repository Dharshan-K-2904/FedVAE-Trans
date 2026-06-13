"""
src/federated/strategy.py
FedVAE-Trans — Custom Flower Aggregation Strategies

Implements two strategies for comparative analysis (Objective 3):

  FedVAEAvg   — Standard FedAvg (McMahan et al., 2017) with metric tracking.
                Weighted average of client weights proportional to dataset size.

  FedVAEProx  — FedProx (Li et al., 2020) variant. Proximal regularisation
                is enforced client-side (see client.py). This strategy simply
                sets the proximal_mu config field before each round so clients
                know to activate the proximal term.

Both strategies log per-round: global F1, AUROC, ELBO loss, and communication
payload in MB — enabling the communication efficiency analysis of Section 7.

Syllabus alignment:
  Unit 4 — FedAvg Protocol, FTL / Advanced Aggregation, Flower Architecture
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import (
    FitIns, FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.models.bilstm_trans_vae import BiLSTMTransVAE
from src.utils.metrics import evaluate_vae


# ---------------------------------------------------------------------------
# FedAvg Strategy with Metric Tracking
# ---------------------------------------------------------------------------

class FedVAEStrategy(FedAvg):
    """
    Custom Flower strategy extending FedAvg.

    Adds:
      - Per-round global evaluation (F1, AUROC) on a shared test set
      - Communication payload logging (bytes per round)
      - Optional FedProx mode (passes proximal_mu to each client via config)
      - Checkpoint saving every `save_every` rounds

    Args:
        model           : The global model instance (updated in place after each round)
        test_loader     : DataLoader for global evaluation
        device          : torch.device
        proximal_mu     : 0.0 = FedAvg, 0.01 = FedProx
        save_dir        : Directory to save round checkpoints
        save_every      : Save checkpoint every N rounds
        **kwargs        : Forwarded to FedAvg (min_fit_clients, etc.)
    """

    def __init__(
        self,
        model:        torch.nn.Module,
        test_loader:  DataLoader,
        device:       torch.device,
        proximal_mu:  float = 0.0,
        save_dir:     str   = "data/phase3_results/federated",
        save_every:   int   = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model        = model
        self.test_loader  = test_loader
        self.device       = device
        self.proximal_mu  = proximal_mu
        self.save_dir     = Path(save_dir)
        self.save_every   = save_every

        self.round_metrics: List[Dict] = []   # Populated each round
        self._threshold:    float = 0.5       # Updated after first evaluation
        self._start_time:   float = time.time()

        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "metrics").mkdir(exist_ok=True)

        mode = "FedProx" if proximal_mu > 0 else "FedAvg"
        print(f"✅ Strategy: {mode} (proximal_mu={proximal_mu})")

    # ------------------------------------------------------------------
    # Override configure_fit to inject proximal_mu into client config
    # ------------------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters:   Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Inject round number and proximal_mu into each client's config."""
        config = {
            "round":        server_round,
            "proximal_mu":  self.proximal_mu,
        }
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(client, fit_ins) for client in clients]

    # ------------------------------------------------------------------
    # Override aggregate_fit to evaluate after each round
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results:      List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Standard FedAvg weight aggregation + per-round global evaluation.
        """
        if not results:
            return None, {}

        # --- FedAvg weighted aggregation ---
        aggregated_params, _ = super().aggregate_fit(server_round, results, failures)
        if aggregated_params is None:
            return aggregated_params, {}

        # --- Load aggregated weights into global model ---
        ndarrays = parameters_to_ndarrays(aggregated_params)
        state_dict = dict(
            zip(self.model.state_dict().keys(),
                [torch.tensor(p) for p in ndarrays])
        )
        self.model.load_state_dict(state_dict, strict=True)

        # --- Global evaluation on shared test set ---
        metrics = evaluate_vae(
            model        = self.model,
            test_loader  = self.test_loader,
            train_loader = None,          # threshold already calibrated
            device       = self.device,
            threshold    = self._threshold if server_round > 1 else None,
            verbose      = False,
        )

        # Update cached threshold after first round calibration
        if server_round == 1:
            self._threshold = metrics["threshold"]

        # --- Communication payload (bytes per round, upload + download) ---
        total_params   = sum(p.numel() for p in self.model.parameters())
        payload_fp32   = total_params * 4 / 1e6   # MB (float32)
        payload_fp16   = total_params * 2 / 1e6   # MB (float16)
        num_clients    = len(results)
        round_payload  = payload_fp32 * num_clients * 2   # upload + download

        elapsed = time.time() - self._start_time

        round_data = {
            "round"           : server_round,
            "f1"              : round(metrics["f1"],        4),
            "auroc"           : round(metrics["auroc"],     4),
            "precision"       : round(metrics["precision"], 4),
            "recall"          : round(metrics["recall"],    4),
            "threshold"       : round(self._threshold,      6),
            "payload_fp32_mb" : round(round_payload,        2),
            "payload_fp16_mb" : round(payload_fp16 * num_clients * 2, 2),
            "wall_time_s"     : round(elapsed,              1),
        }
        self.round_metrics.append(round_data)

        mode = "FedProx" if self.proximal_mu > 0 else "FedAvg"
        print(
            f"  Round {server_round:>2} [{mode}] | "
            f"F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}  "
            f"Payload={round_payload:.1f} MB"
        )

        # --- Checkpoint every N rounds ---
        if server_round % self.save_every == 0 or server_round == 1:
            ckpt_path = self.save_dir / "checkpoints" / f"global_round_{server_round:03d}.pt"
            torch.save(self.model.state_dict(), ckpt_path)

        # --- Save running metrics JSON ---
        metrics_path = self.save_dir / "metrics" / "federated_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.round_metrics, f, indent=2)

        return aggregated_params, {"f1": metrics["f1"], "auroc": metrics["auroc"]}

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------

    def save_final_model(self, filename: str = "global_trans_vae.pt") -> None:
        """Save the final global model weights to disk."""
        path = self.save_dir / "checkpoints" / filename
        torch.save(self.model.state_dict(), path)
        print(f"✅ Final global model saved → {path}")

    def save_metrics(self, filename: str = "federated_metrics.json") -> None:
        """Persist all round metrics to JSON."""
        path = self.save_dir / "metrics" / filename
        with open(path, "w") as f:
            json.dump(self.round_metrics, f, indent=2)
        print(f"✅ Metrics saved → {path}")