import os
import json
import copy
import math
import random
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal App Setup
# ---------------------------------------------------------------------------

app   = modal.App("fedvae-trans-phase3")
vol   = modal.Volume.from_name("fedvae-trans-storage", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "numpy", "scikit-learn",
        "flwr==1.14.0",          
        "protobuf==4.25.3",      
        "ray[default]==2.10.0",  
        "tqdm",
    )
)

# ---------------------------------------------------------------------------
# Federated simulation (runs on GPU)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": vol},
    timeout=60 * 120,   # 120 minutes
    memory=16384,
)
def run_federation():
    import os
    import math, json, copy, time
    import numpy as np
    import warnings
    import logging
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, Subset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (f1_score, roc_auc_score, roc_curve,
                                 precision_score, recall_score)
    import flwr as fl
    from flwr.common import NDArrays, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedAvg
    from flwr.common import FitIns, FitRes, Parameters, Scalar

    # -----------------------------------------------------------------------
    # AGGRESSIVE TELEMETRY AND WARNING SUPPRESSION
    # -----------------------------------------------------------------------
    os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1"
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["RAY_CPP_LOG_LEVEL"] = "3"
    warnings.simplefilter("ignore")
    logging.getLogger("flwr").setLevel(logging.ERROR)
    logging.getLogger("ray").setLevel(logging.ERROR)

    SEED   = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # -----------------------------------------------------------------------
    # 1. Load artefacts
    # -----------------------------------------------------------------------
    sequences = np.load("/vol/processed/sequences.npy")
    labels    = np.load("/vol/processed/labels.npy")
    with open("/vol/processed/metadata.json") as f:
        meta = json.load(f)
    VOCAB_SIZE  = meta["vocab_size"]
    WINDOW_SIZE = meta["window_size"]
    NUM_CLIENTS = 10
    ALPHA       = 0.5

    print(f"\n✅ Data loaded: {sequences.shape}, anomaly rate {labels.mean()*100:.2f}%")

    # -----------------------------------------------------------------------
    # 2. Dataset
    # -----------------------------------------------------------------------
    class LogDS(Dataset):
        def __init__(self, s, l):
            self.s = torch.from_numpy(s).long()
            self.l = torch.from_numpy(l).long()
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.s[i], self.l[i]

    full_ds = LogDS(sequences, labels)
    train_idx, test_idx = train_test_split(
        np.arange(len(sequences)), test_size=0.2,
        stratify=labels, random_state=SEED
    )
    
    test_loader = DataLoader(Subset(full_ds, test_idx),
                             batch_size=512, shuffle=False, num_workers=0)

    def dirichlet_partition(lbls, n_clients, alpha=0.5, seed=SEED):
        rng = np.random.default_rng(seed)
        client_idx = [[] for _ in range(n_clients)]
        for cls in np.unique(lbls):
            idx = np.where(lbls == cls)[0]; rng.shuffle(idx)
            props = rng.dirichlet(np.full(n_clients, alpha))
            splits = np.clip((np.cumsum(props)*len(idx)).astype(int), 0, len(idx))
            for cid, chunk in enumerate(np.split(idx, splits[:-1])):
                client_idx[cid].extend(chunk.tolist())
        return [np.array(x, dtype=np.int64) for x in client_idx]

    train_labels_sub = labels[train_idx]
    client_splits    = dirichlet_partition(train_labels_sub, NUM_CLIENTS, ALPHA)

    client_loaders = []
    for cid, local_idx in enumerate(client_splits):
        global_idx    = train_idx[local_idx]
        normal_global = global_idx[labels[global_idx] == 0]
        if len(normal_global) == 0:
            normal_global = global_idx
        ldr = DataLoader(Subset(full_ds, normal_global),
                         batch_size=256, shuffle=True, num_workers=0, drop_last=False)
        client_loaders.append(ldr)

    # -----------------------------------------------------------------------
    # 3. Model (BiLSTMTransVAE — inline)
    # -----------------------------------------------------------------------
    class PositionalEncoding(nn.Module):
        def __init__(self, d, max_len=512, drop=0.1):
            super().__init__(); self.drop = nn.Dropout(drop)
            pe = torch.zeros(max_len, d)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float()*(-math.log(10000.)/d))
            pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x): return self.drop(x+self.pe[:,:x.size(1)])

    class BiLSTMTransVAE(nn.Module):
        def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2,
                     latent_dim=64, seq_len=128, lstm_layers=2, dropout=0.1):
            super().__init__()
            self.d_model=d_model; self.seq_len=seq_len; self.latent_dim=latent_dim
            self.cls_token = nn.Parameter(torch.randn(1,1,d_model)*0.02)
            self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.bilstm = nn.LSTM(d_model, d_model//2, lstm_layers,
                                   batch_first=True, bidirectional=True,
                                   dropout=dropout if lstm_layers>1 else 0.)
            self.lstm_norm = nn.LayerNorm(d_model)
            self.pos_enc = PositionalEncoding(d_model, seq_len+1, dropout)
            el = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout,
                                             "gelu", batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(el, num_layers)
            self.fc_mu = nn.Linear(d_model, latent_dim)
            self.fc_lv = nn.Linear(d_model, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, d_model*2), nn.GELU(),
                nn.LayerNorm(d_model*2),
                nn.Linear(d_model*2, d_model*4), nn.GELU(),
                nn.Linear(d_model*4, seq_len*d_model))
        def reparameterise(self, mu, lv):
            return mu+torch.exp(0.5*lv)*torch.randn_like(mu) if self.training else mu
        def forward(self, ids):
            B,T=ids.shape
            te=self.token_embedding(ids); oe=te.detach()
            lo,_=self.bilstm(te); lo=self.lstm_norm(lo)
            x=torch.cat([self.cls_token.expand(B,-1,-1),lo],dim=1)
            x=self.pos_enc(x)
            pm=torch.zeros(B,T+1,dtype=torch.bool,device=ids.device); pm[:,1:]=(ids==0)
            x=self.transformer(x,src_key_padding_mask=pm); c=x[:,0,:]
            mu,lv=self.fc_mu(c),self.fc_lv(c); z=self.reparameterise(mu,lv)
            return self.decoder(z).reshape(B,T,self.d_model), oe, mu, lv
        @torch.no_grad()
        def anomaly_score(self, ids):
            self.eval(); re,oe,_,_=self.forward(ids)
            return F.mse_loss(re,oe,reduction="none").mean(dim=[1,2])

    def build_model():
        return BiLSTMTransVAE(VOCAB_SIZE, d_model=128, nhead=4, num_layers=2,
                               latent_dim=64, seq_len=WINDOW_SIZE, lstm_layers=2)

    # -----------------------------------------------------------------------
    # 4. Loss helpers
    # -----------------------------------------------------------------------
    def cyclical_beta(epoch, cycle_len=20, max_beta=0.1):
        pos = (epoch-1) % cycle_len
        return max_beta*(pos/(cycle_len//2)) if pos < cycle_len//2 else max_beta

    def elbo_loss(re, oe, mu, lv, beta=0.1):
        r = F.mse_loss(re, oe, reduction="mean")
        k = -0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        k = torch.clamp(k, 0., 10.)
        return r + beta*k, r.item(), k.item()

    # -----------------------------------------------------------------------
    # 5. Evaluation with ROC-optimal threshold
    # -----------------------------------------------------------------------
    def evaluate_global(model, threshold=None):
        import warnings; warnings.simplefilter("ignore")
        model.eval()
        scores_l, labels_l = [], []
        with torch.no_grad():
            for ids, lbl in test_loader:
                with torch.amp.autocast('cuda'):
                    s = model.anomaly_score(ids.to(DEVICE))
                scores_l.append(s.cpu().numpy())
                labels_l.append(lbl.numpy())
        scores = np.concatenate(scores_l); lbls = np.concatenate(labels_l)
        auroc  = roc_auc_score(lbls, scores)
        fpr, tpr, thres = roc_curve(lbls, scores)
        if threshold is None:
            threshold = float(thres[np.argmax(tpr-fpr)])
        preds = (scores >= threshold).astype(int)
        return (f1_score(lbls,preds,zero_division=0),
                precision_score(lbls,preds,zero_division=0),
                recall_score(lbls,preds,zero_division=0),
                auroc, threshold)

    # -----------------------------------------------------------------------
    # 6. Flower Client
    # -----------------------------------------------------------------------
    LOCAL_EPOCHS = 5
    LOCAL_LR     = 1e-4

    def get_weights(model):
        return [v.cpu().numpy() for v in model.state_dict().values()]

    def set_weights(model, weights):
        sd = dict(zip(model.state_dict().keys(), [torch.tensor(w) for w in weights]))
        model.load_state_dict(sd, strict=True)

    class VAEClient(fl.client.NumPyClient):
        def __init__(self, cid, train_loader, global_round_ref):
            self.cid          = cid
            self.train_loader = train_loader
            self.model        = build_model().to(DEVICE)
            self.global_round = global_round_ref   

        def get_parameters(self, config): return get_weights(self.model)
        def set_parameters(self, params): set_weights(self.model, params)

        def fit(self, parameters, config):
            import warnings; warnings.simplefilter("ignore")
            self.set_parameters(parameters)
            proximal_mu = float(config.get("proximal_mu", 0.0))
            round_num   = int(config.get("round", 1))

            global_snap = copy.deepcopy(self.model)
            for p in global_snap.parameters(): p.requires_grad_(False)

            opt    = optim.AdamW(self.model.parameters(), lr=LOCAL_LR, weight_decay=1e-4)
            scaler = torch.amp.GradScaler('cuda')

            self.model.train()
            total_loss = 0.0

            for ep in range(1, LOCAL_EPOCHS+1):
                eff_ep = (round_num-1)*LOCAL_EPOCHS + ep
                beta   = cyclical_beta(eff_ep)

                for ids, _ in self.train_loader:
                    ids = ids.to(DEVICE); opt.zero_grad()
                    with torch.amp.autocast('cuda'):
                        re,oe,mu,lv = self.model(ids)
                        loss,_,_    = elbo_loss(re,oe,mu,lv,beta=beta)
                        if proximal_mu > 0:
                            prox=sum((wl-wg).pow(2).sum()
                                    for wl,wg in zip(self.model.parameters(),
                                                     global_snap.parameters()))
                            loss = loss + (proximal_mu/2.)*prox
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                    total_loss += loss.item()

            avg_loss = total_loss / max(1, LOCAL_EPOCHS*len(self.train_loader))
            return self.get_parameters({}), len(self.train_loader.dataset), {"loss": avg_loss}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            return 0.0, len(self.train_loader.dataset), {}

    # -----------------------------------------------------------------------
    # 7. Custom FedAvg/FedProx Strategy
    # -----------------------------------------------------------------------
    class FedVAEStrategy(FedAvg):
        def __init__(self, proximal_mu=0.0, **kwargs):
            super().__init__(**kwargs)
            self.proximal_mu  = proximal_mu
            self.round_metrics: List[Dict] = []
            self._threshold   = None
            self._global_model = build_model().to(DEVICE)
            self._save_dir    = Path("/vol/results/federated")
            mode = "FedProx" if proximal_mu > 0 else "FedAvg"
            print(f"\n🌸 Strategy: {mode} (μ={proximal_mu})")

        def configure_fit(self, server_round, parameters, client_manager):
            config  = {"round": server_round, "proximal_mu": self.proximal_mu}
            fit_ins = FitIns(parameters, config)
            clients = client_manager.sample(
                num_clients=self.min_fit_clients,
                min_num_clients=self.min_available_clients)
            return [(c, fit_ins) for c in clients]

        def aggregate_fit(self, server_round, results, failures):
            agg_params, _ = super().aggregate_fit(server_round, results, failures)
            if agg_params is None: return agg_params, {}

            set_weights(self._global_model, parameters_to_ndarrays(agg_params))
            f1, prec, rec, auroc, tau = evaluate_global(
                self._global_model,
                threshold=self._threshold
            )
            if self._threshold is None: self._threshold = tau

            total_params   = sum(p.numel() for p in self._global_model.parameters())
            payload_mb     = total_params*4/1e6 * NUM_CLIENTS * 2

            rd = {"round": server_round, "f1": round(f1,4),
                  "precision": round(prec,4), "recall": round(rec,4),
                  "auroc": round(auroc,4), "threshold": round(tau,6),
                  "payload_fp32_mb": round(payload_mb,2)}
            self.round_metrics.append(rd)

            mode = "FedProx" if self.proximal_mu>0 else "FedAvg"
            print(f"  [{mode}] Round {server_round:>2} | "
                  f"F1={f1:.4f}  AUROC={auroc:.4f}  Payload={payload_mb:.1f}MB")

            if server_round % 10 == 0 or server_round == 50:
                p = self._save_dir/f"checkpoints/round_{server_round:03d}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._global_model.state_dict(), p)

            mp = self._save_dir/"metrics/federated_metrics.json"
            mp.parent.mkdir(parents=True, exist_ok=True)
            with open(mp,"w") as fh: json.dump(self.round_metrics, fh, indent=2)

            return agg_params, {"f1": f1, "auroc": auroc}

    # -----------------------------------------------------------------------
    # 8. Run Simulation
    # -----------------------------------------------------------------------
    NUM_ROUNDS = 50
    init_model = build_model()
    init_params = ndarrays_to_parameters(get_weights(init_model))

    print(f"\n{'='*55}")
    print(f"  FedVAE-Trans Phase 3 — Federated Simulation")
    print(f"  Clients={NUM_CLIENTS}, Rounds={NUM_ROUNDS}, α={ALPHA}")
    print(f"{'='*55}")

    for strategy_name, prox_mu in [("fedavg", 0.0), ("fedprox", 0.01)]:
        print(f"\n--- Running {strategy_name.upper()} ---")

        save_dir = Path(f"/vol/results/federated/{strategy_name}")
        strategy = FedVAEStrategy(
            proximal_mu           = prox_mu,
            min_fit_clients       = NUM_CLIENTS,
            min_available_clients = NUM_CLIENTS,
            initial_parameters    = init_params,
        )
        strategy._save_dir = save_dir

        def client_fn(cid: str):
            import warnings; warnings.simplefilter("ignore")
            return VAEClient(int(cid), client_loaders[int(cid)], {}).to_client()

        fl.simulation.start_simulation(
            client_fn         = client_fn,
            num_clients       = NUM_CLIENTS,
            config            = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy          = strategy,
            client_resources  = {"num_cpus": 1.0, "num_gpus": 0.25}, 
            ray_init_args     = {
                "ignore_reinit_error": True,
                "include_dashboard"  : False,
                "num_cpus"           : 4,
                "num_gpus"           : 1,
            },
        )

        final_path = save_dir / "checkpoints/global_trans_vae.pt"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(strategy._global_model.state_dict(), final_path)
        print(f"✅ {strategy_name.upper()} complete — final model → {final_path}")

    # -----------------------------------------------------------------------
    # 9. Communication efficiency summary
    # -----------------------------------------------------------------------
    total_params = sum(p.numel() for p in build_model().parameters())
    comm_summary = {
        "total_params"           : total_params,
        "model_size_mb_fp32"     : round(total_params*4/1e6, 2),
        "model_size_mb_fp16"     : round(total_params*2/1e6, 2),
        "payload_per_round_fp32" : round(total_params*4/1e6*NUM_CLIENTS*2, 2),
        "payload_per_round_fp16" : round(total_params*2/1e6*NUM_CLIENTS*2, 2),
        "total_rounds"           : NUM_ROUNDS,
        "total_comm_fp32_mb"     : round(total_params*4/1e6*NUM_CLIENTS*2*NUM_ROUNDS, 2),
        "reduction_fp16_pct"     : 50.0,
    }
    comm_path = Path("/vol/results/federated/communication_summary.json")
    with open(comm_path, "w") as f: json.dump(comm_summary, f, indent=2)
    print(f"\n📡 Communication summary → {comm_path}")
    print(json.dumps(comm_summary, indent=2))
    vol.commit()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("🚀 Launching Phase 3 federated simulation on Modal GPU...")
    run_federation.remote()
    print("✅ Phase 3 complete.")
    print("   Download: modal volume get fedvae-trans-storage /results/federated data/phase3_results")
