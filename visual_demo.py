"""
visual_demo.py
FedVAE-Trans — Streamlit Explainable AI (XAI) Live Dashboard

Faculty Presentation Dashboard. Loads the trained BiLSTMTransVAE weights,
runs live inference on HDFS log sequences, and visualises reconstruction
error in real time. Anomalous windows spike above the red threshold line.

Execution:
    streamlit run visual_demo.py

Requirements:
    pip install streamlit plotly torch numpy scikit-learn

Data expected:
    data/processed/sequences.npy
    data/processed/labels.npy
    data/phase3_results/federated/checkpoints/global_trans_vae.pt
    OR
    data/phase2_results/centralized/C_BiLSTMTransVAE_best.pt
"""

import math
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import (
    f1_score, roc_auc_score, roc_curve,
    precision_score, recall_score, confusion_matrix,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title  = "FedVAE-Trans — Live Demo",
    page_icon   = "🔍",
    layout      = "wide",
)

DEVICE = torch.device("cpu")   # CPU for live demo (no GPU needed)
SEED   = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Inline model definitions  (no import of src/ needed for standalone demo)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class BiLSTMTransVAE(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2,
                 latent_dim=64, seq_len=128, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model; self.seq_len = seq_len; self.latent_dim = latent_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.bilstm = nn.LSTM(d_model, d_model // 2, lstm_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if lstm_layers > 1 else 0.0)
        self.lstm_norm = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len + 1, dropout)
        el = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout,
                                         "gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(el, num_layers)
        self.fc_mu  = nn.Linear(d_model, latent_dim)
        self.fc_lv  = nn.Linear(d_model, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2), nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, seq_len * d_model))

    def forward(self, ids):
        B, T = ids.shape
        te = self.token_embedding(ids); oe = te.detach()
        lo, _ = self.bilstm(te); lo = self.lstm_norm(lo)
        x = torch.cat([self.cls_token.expand(B, -1, -1), lo], dim=1)
        x = self.pos_enc(x)
        pm = torch.zeros(B, T + 1, dtype=torch.bool, device=ids.device)
        pm[:, 1:] = (ids == 0)
        x = self.transformer(x, src_key_padding_mask=pm); c = x[:, 0, :]
        mu, lv = self.fc_mu(c), self.fc_lv(c)
        return self.decoder(mu).reshape(B, T, self.d_model), oe, mu, lv

    @torch.no_grad()
    def anomaly_score(self, ids):
        self.eval()
        re, oe, _, _ = self.forward(ids)
        return F.mse_loss(re, oe, reduction="none").mean(dim=[1, 2])


# ---------------------------------------------------------------------------
# Data & Model Loading  (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(vocab_size: int, seq_len: int, checkpoint_path: str):
    model = BiLSTMTransVAE(vocab_size, seq_len=seq_len).to(DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    # Support both raw state_dict and {"model": state_dict} formats
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@st.cache_data
def load_data():
    proc_dir = Path("data/processed")
    sequences = np.load(proc_dir / "sequences.npy")
    labels    = np.load(proc_dir / "labels.npy")
    return sequences, labels


# ---------------------------------------------------------------------------
# Threshold helper
# ---------------------------------------------------------------------------

def compute_threshold_youden(model, sequences, labels, n_samples=5000):
    """Compute ROC-optimal threshold on a random sample of test data."""
    idx    = np.random.choice(len(sequences), size=min(n_samples, len(sequences)),
                              replace=False)
    sample = torch.from_numpy(sequences[idx]).long().to(DEVICE)
    with torch.no_grad():
        scores = model.anomaly_score(sample).numpy()
    lbls = labels[idx]
    fpr, tpr, thres = roc_curve(lbls, scores)
    return float(thres[np.argmax(tpr - fpr)])


# ---------------------------------------------------------------------------
# Dashboard Layout
# ---------------------------------------------------------------------------

st.title("🔍 FedVAE-Trans — Live Explainable AI Dashboard")
st.markdown(
    "**Communication-Efficient Federated Variational Transformer for Long-Sequence Anomaly Detection**  \n"
    "Faculty Demonstration — PESU Academy Deep Learning Project 2025–2026"
)
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    ckpt_options = {
        "Federated (FedProx) — Global Model": "data/phase3_results/federated/fedprox/checkpoints/global_trans_vae.pt",
        "Federated (FedAvg)  — Global Model": "data/phase3_results/federated/fedavg/checkpoints/global_trans_vae.pt",
        "Centralized — BiLSTM-Trans-VAE (C)": "data/phase2_results/centralized/C_BiLSTMTransVAE_best.pt",
        "Centralized — Trans-VAE (B)":        "data/phase2_results/centralized/B_TransVAE_best.pt",
    }
    selected_ckpt_name = st.selectbox("Model Checkpoint", list(ckpt_options.keys()))
    ckpt_path = ckpt_options[selected_ckpt_name]

    n_display  = st.slider("Sequences to Display", 50, 500, 200)
    show_normal = st.checkbox("Show Normal Sequences", value=True)
    show_anomaly = st.checkbox("Show Anomaly Sequences", value=True)
    animate_btn  = st.button("🔄 Refresh Sample")

# --- Load data ---
if not Path("data/processed/sequences.npy").exists():
    st.error("❌ data/processed/sequences.npy not found. Run Phase 1 first.")
    st.stop()

sequences, labels = load_data()

meta_path = Path("data/processed/metadata.json")
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    VOCAB_SIZE  = meta["vocab_size"]
    WINDOW_SIZE = meta["window_size"]
else:
    VOCAB_SIZE  = int(sequences.max()) + 2
    WINDOW_SIZE = sequences.shape[1]

if not Path(ckpt_path).exists():
    st.warning(f"⚠️ Checkpoint not found: `{ckpt_path}`. "
               f"Showing random scores for demonstration.")
    model     = None
    threshold = 0.05
else:
    model     = load_model(VOCAB_SIZE, WINDOW_SIZE, ckpt_path)
    threshold = compute_threshold_youden(model, sequences, labels)

# --- Sample indices ---
normal_idx  = np.where(labels == 0)[0]
anomaly_idx = np.where(labels == 1)[0]
chosen_idx  = []
if show_normal:
    chosen_idx.extend(np.random.choice(normal_idx,
                      size=min(n_display // 2, len(normal_idx)), replace=False).tolist())
if show_anomaly:
    chosen_idx.extend(np.random.choice(anomaly_idx,
                      size=min(n_display // 2, len(anomaly_idx)), replace=False).tolist())
chosen_idx = np.array(chosen_idx)
np.random.shuffle(chosen_idx)

# --- Compute scores ---
if model is not None:
    batch = torch.from_numpy(sequences[chosen_idx]).long().to(DEVICE)
    with torch.no_grad():
        scores = model.anomaly_score(batch).numpy()
else:
    # Demo mode: generate plausible random scores
    base = np.random.exponential(0.01, len(chosen_idx))
    scores = np.where(labels[chosen_idx] == 1, base * 8, base)

true_labels = labels[chosen_idx]
pred_labels = (scores >= threshold).astype(int)

# ---------------------------------------------------------------------------
# Top-level KPI cards
# ---------------------------------------------------------------------------
f1    = f1_score(true_labels, pred_labels, zero_division=0)
prec  = precision_score(true_labels, pred_labels, zero_division=0)
rec   = recall_score(true_labels, pred_labels, zero_division=0)
try:
    auroc = roc_auc_score(true_labels, scores)
except ValueError:
    auroc = 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("F1-Score",   f"{f1:.4f}",   "↑ target ≥ 0.92")
c2.metric("Precision",  f"{prec:.4f}")
c3.metric("Recall",     f"{rec:.4f}")
c4.metric("AUROC",      f"{auroc:.4f}")
c5.metric("Threshold τ", f"{threshold:.5f}", "ROC-optimal")

st.divider()

# ---------------------------------------------------------------------------
# Main chart: Reconstruction Error per Sequence
# ---------------------------------------------------------------------------
st.subheader("📊 Reconstruction Error per Log Sequence")
st.caption(
    "Each bar is one log window. **Red bars** exceed the anomaly threshold τ "
    "(correctly detected anomalies). **Green bars** are correctly classified as normal."
)

colours = []
for i in range(len(chosen_idx)):
    gt  = true_labels[i]
    pred = pred_labels[i]
    if gt == 1 and pred == 1:
        colours.append("#e74c3c")   # True Positive  — red
    elif gt == 0 and pred == 0:
        colours.append("#2ecc71")   # True Negative  — green
    elif gt == 0 and pred == 1:
        colours.append("#e67e22")   # False Positive — orange
    else:
        colours.append("#9b59b6")   # False Negative — purple

fig = go.Figure()
fig.add_trace(go.Bar(
    x=list(range(len(chosen_idx))),
    y=scores,
    marker_color=colours,
    name="Reconstruction Error",
    hovertemplate="Seq %{x}<br>Error: %{y:.5f}<br>True: %{customdata[0]}<br>Pred: %{customdata[1]}",
    customdata=[[
        "Anomaly" if true_labels[i] == 1 else "Normal",
        "Anomaly" if pred_labels[i] == 1 else "Normal",
    ] for i in range(len(chosen_idx))],
))
fig.add_hline(
    y=threshold, line_dash="dash", line_color="#c0392b", line_width=2,
    annotation_text=f"  τ = {threshold:.5f}  (anomaly threshold)",
    annotation_position="top left",
)
fig.update_layout(
    xaxis_title="Log Sequence Index",
    yaxis_title="Reconstruction Error (MSE)",
    template="plotly_white",
    height=400,
    showlegend=False,
    margin=dict(l=0, r=0, t=20, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# Legend
lc1, lc2, lc3, lc4 = st.columns(4)
lc1.markdown("🟥 **True Positive** — Anomaly correctly detected")
lc2.markdown("🟩 **True Negative** — Normal correctly classified")
lc3.markdown("🟧 **False Positive** — Normal misclassified as anomaly")
lc4.markdown("🟪 **False Negative** — Anomaly missed")

st.divider()

# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------
col_roc, col_cm = st.columns(2)

with col_roc:
    st.subheader("📈 ROC Curve")
    try:
        fpr_vals, tpr_vals, _ = roc_curve(true_labels, scores)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr_vals, y=tpr_vals, mode="lines",
            name=f"Trans-VAE (AUROC={auroc:.4f})",
            line=dict(color="#2980b9", width=2)))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="grey", dash="dash"), showlegend=False))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white", height=350,
            margin=dict(l=0, r=0, t=10, b=40))
        st.plotly_chart(fig_roc, use_container_width=True)
    except ValueError:
        st.info("ROC curve requires both classes in the selected sample.")

# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------
with col_cm:
    st.subheader("🔲 Confusion Matrix")
    try:
        cm = confusion_matrix(true_labels, pred_labels)
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred Normal", "Pred Anomaly"],
            y=["True Normal", "True Anomaly"],
            colorscale="Blues", showscale=False,
            text=cm, texttemplate="%{text}",
            textfont=dict(size=18, color="white")))
        fig_cm.update_layout(
            template="plotly_white", height=350,
            margin=dict(l=0, r=0, t=10, b=40))
        st.plotly_chart(fig_cm, use_container_width=True)
    except Exception:
        st.info("Confusion matrix requires predictions in the current sample.")

st.divider()

# ---------------------------------------------------------------------------
# Federated Metrics (if available)
# ---------------------------------------------------------------------------
st.subheader("🌸 Federated Learning Convergence")

fedavg_path  = Path("data/phase3_results/federated/fedavg/metrics/federated_metrics.json")
fedprox_path = Path("data/phase3_results/federated/fedprox/metrics/federated_metrics.json")

if fedavg_path.exists() or fedprox_path.exists():
    fig_fed = go.Figure()
    for path, name, colour in [
        (fedavg_path,  "FedAvg",  "#2980b9"),
        (fedprox_path, "FedProx", "#e74c3c"),
    ]:
        if path.exists():
            with open(path) as f:
                rounds_data = json.load(f)
            rounds = [r["round"] for r in rounds_data]
            f1s    = [r["f1"]    for r in rounds_data]
            fig_fed.add_trace(go.Scatter(
                x=rounds, y=f1s, mode="lines+markers",
                name=name, line=dict(color=colour, width=2)))
    fig_fed.add_hline(
        y=0.92, line_dash="dot", line_color="green",
        annotation_text="  Target F1 ≥ 0.92", annotation_position="top left")
    fig_fed.update_layout(
        xaxis_title="Federated Round",
        yaxis_title="Global F1-Score",
        template="plotly_white", height=350,
        margin=dict(l=0, r=0, t=10, b=40))
    st.plotly_chart(fig_fed, use_container_width=True)
else:
    st.info("Federated metrics not yet available. Run Phase 3 first: `modal run cloud/phase3_federated.py`")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "FedVAE-Trans · PESU Academy · Introduction to Deep Learning · 2025–2026 · "
    f"Model: **{selected_ckpt_name}** · "
    f"Dataset: HDFS ({len(sequences):,} sequences, anomaly rate {labels.mean()*100:.2f}%)"
)