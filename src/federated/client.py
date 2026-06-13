"""
src/federated/client.py
FedVAE-Trans — Flower Federated Client

Each client instance represents one distributed node (e.g. an enterprise
server or IoT gateway). It holds its own local DataLoader and trains the
shared BiLSTMTransVAE model for LOCAL_EPOCHS rounds per federated round.

Key features:
  - FedProx proximal term (μ‖w - w_global‖²) prevents client drift
  - Mixed-precision training (torch.amp) for memory efficiency
  - Cyclical β-annealing to prevent VAE posterior collapse

Syllabus alignment:
  Unit 4 — Flower Architecture (Hands-on), Horizontal FL, FedAvg/FedProx
"""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import NDArrays, Scalar

from src.models.bilstm_trans_vae import cyclical_beta, elbo_loss


# ---------------------------------------------------------------------------
# Flower Client
# ---------------------------------------------------------------------------

class FedVAEClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for federated Trans-VAE training.

    Protocol per round:
      1. Server sends global weights  → set_parameters()
      2. Client trains locally        → fit()
      3. Client sends updated weights → get_parameters()

    FedProx:
      When proximal_mu > 0, the client's loss is augmented with
      μ/2 · ‖w_local - w_global‖² which anchors the local model to
      the last received global model, preventing divergence under Non-IID.
    """

    def __init__(
        self,
        client_id:    int,
        model:        nn.Module,
        train_loader: DataLoader,
        test_loader:  DataLoader,
        device:       torch.device,
        local_epochs: int   = 5,
        lr:           float = 1e-4,
        proximal_mu:  float = 0.0,
        global_round: int   = 0,
    ):
        self.client_id    = client_id
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.device       = device
        self.local_epochs = local_epochs
        self.lr           = lr
        self.proximal_mu  = proximal_mu   # 0.0 = FedAvg, 0.01 = FedProx
        self.global_round = global_round  # Used for cyclical β schedule

        self.scaler = GradScaler()        # Mixed-precision gradient scaler

    # ------------------------------------------------------------------
    # Flower API: Weight serialisation
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> NDArrays:
        """Extract model weights as a list of numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Load server-sent weights into the local model."""
        state_dict = dict(
            zip(self.model.state_dict().keys(),
                [torch.tensor(p) for p in parameters])
        )
        self.model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # Flower API: Local training
    # ------------------------------------------------------------------

    def fit(
        self,
        parameters: NDArrays,
        config:     Dict,
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Receive global weights, run local training, return updated weights.

        Returns:
            (updated_weights, num_training_samples, metrics_dict)
        """
        self.set_parameters(parameters)

        # Save global weights snapshot for FedProx proximal term
        global_model = copy.deepcopy(self.model)
        for p in global_model.parameters():
            p.requires_grad_(False)

        optimiser = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.local_epochs
        )

        self.model.train()
        total_loss = 0.0

        for epoch in range(1, self.local_epochs + 1):
            # Cyclical β for this epoch (global_round * local_epochs + epoch)
            effective_epoch = self.global_round * self.local_epochs + epoch
            beta = cyclical_beta(effective_epoch, cycle_len=20, max_beta=0.1)

            for input_ids, _ in self.train_loader:
                input_ids = input_ids.to(self.device)
                optimiser.zero_grad()

                # Mixed-precision forward pass
                with autocast():
                    recon_embed, orig_embed, mu, log_var = self.model(input_ids)
                    loss, _, _ = elbo_loss(recon_embed, orig_embed, mu, log_var,
                                           beta=beta)

                    # FedProx proximal term: prevents local model from drifting
                    if self.proximal_mu > 0.0:
                        prox_term = 0.0
                        for (name, w_local), (_, w_global) in zip(
                            self.model.named_parameters(),
                            global_model.named_parameters(),
                        ):
                            prox_term += (w_local - w_global).pow(2).sum()
                        loss = loss + (self.proximal_mu / 2.0) * prox_term

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimiser)
                self.scaler.update()

                total_loss += loss.item()

            scheduler.step()

        num_samples  = len(self.train_loader.dataset)
        avg_loss     = total_loss / max(1, self.local_epochs * len(self.train_loader))

        return (
            self.get_parameters({}),
            num_samples,
            {"train_loss": avg_loss, "client_id": self.client_id},
        )

    # ------------------------------------------------------------------
    # Flower API: Local evaluation (optional — used for per-client metrics)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        parameters: NDArrays,
        config:     Dict,
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the global model on this client's local test distribution.
        Reports ELBO loss (lower = better reconstruction quality).
        """
        self.set_parameters(parameters)
        self.model.eval()

        total_loss   = 0.0
        num_batches  = 0

        with torch.no_grad():
            for input_ids, _ in self.test_loader:
                input_ids = input_ids.to(self.device)
                with autocast():
                    recon_embed, orig_embed, mu, log_var = self.model(input_ids)
                    loss, _, _ = elbo_loss(recon_embed, orig_embed, mu, log_var,
                                           beta=0.1)
                total_loss  += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss, len(self.test_loader.dataset), {"elbo": avg_loss}