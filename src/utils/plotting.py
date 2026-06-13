"""
src/utils/plotting.py
FedVAE-Trans — Publication-Quality Figure Generation

Generates IEEE-style figures for the project report and paper:
  - Convergence curves (FedAvg vs FedProx F1 per round)
  - ROC-AUC curves (per variant)
  - Ablation summary bar chart
  - Communication efficiency analysis
  - Confusion matrix heatmap

All figures are saved to data/paper_figures/ at 300 DPI.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay, confusion_matrix

# IEEE-style plot settings
plt.rcParams.update({
    "font.family"       : "serif",
    "font.size"         : 11,
    "axes.labelsize"    : 12,
    "axes.titlesize"    : 13,
    "legend.fontsize"   : 10,
    "xtick.labelsize"   : 10,
    "ytick.labelsize"   : 10,
    "figure.dpi"        : 150,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
})

FIG_DIR = Path("data/paper_figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "A_VAEOnly"        : "#7f8c8d",
    "B_TransVAE"       : "#2980b9",
    "C_BiLSTMTransVAE" : "#27ae60",
    "LSTM_AE_Baseline" : "#e67e22",
    "FedAvg"           : "#2980b9",
    "FedProx"          : "#c0392b",
}


# ---------------------------------------------------------------------------
# Convergence Curves
# ---------------------------------------------------------------------------

def plot_federated_convergence(
    fedavg_metrics:  List[Dict],
    fedprox_metrics: List[Dict],
    save_path:       Optional[str] = None,
) -> None:
    """
    Plot FedAvg vs FedProx F1 convergence over federated rounds.
    Highlights the target F1 ≥ 0.92 line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric_key, ylabel, title in zip(
        axes,
        ["f1", "auroc"],
        ["F1-Score", "AUROC"],
        ["F1-Score Convergence", "AUROC Convergence"],
    ):
        for data, label, colour in [
            (fedavg_metrics,  "FedAvg",  PALETTE["FedAvg"]),
            (fedprox_metrics, "FedProx", PALETTE["FedProx"]),
        ]:
            if not data: continue
            rounds = [d["round"] for d in data]
            vals   = [d[metric_key] for d in data]
            ax.plot(rounds, vals, label=label, color=colour, linewidth=2, marker="o",
                    markersize=3)

        if metric_key == "f1":
            ax.axhline(0.92, color="green", linestyle="--", linewidth=1.2,
                       label="Target F1 ≥ 0.92")

        ax.set_xlabel("Federated Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(left=1)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    plt.tight_layout()
    out = save_path or str(FIG_DIR / "federated_convergence.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ Saved: {out}")


# ---------------------------------------------------------------------------
# ROC Curves — Ablation Study
# ---------------------------------------------------------------------------

def plot_roc_curves(
    variant_results: Dict[str, Dict],
    save_path:       Optional[str] = None,
) -> None:
    """
    Overlay ROC curves for all ablation variants on a single plot.

    Args:
        variant_results: {name: {"scores": np.ndarray, "labels": np.ndarray}}
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, res in variant_results.items():
        fpr, tpr, _ = roc_curve(res["labels"], res["scores"])
        auroc        = res.get("auroc", np.trapz(tpr, fpr))
        colour       = PALETTE.get(name, "#333333")
        ax.plot(fpr, tpr, color=colour, linewidth=2,
                label=f"{name.replace('_', ' ')} (AUROC={auroc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Ablation Study (HDFS)")
    ax.legend(loc="lower right")
    plt.tight_layout()

    out = save_path or str(FIG_DIR / "roc_curves_ablation.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ Saved: {out}")


# ---------------------------------------------------------------------------
# Ablation Summary Bar Chart
# ---------------------------------------------------------------------------

def plot_ablation_summary(
    results:    Dict[str, float],
    save_path:  Optional[str] = None,
) -> None:
    """
    Horizontal bar chart comparing F1 scores across all variants and baseline.

    Args:
        results: {"variant_name": f1_score, ...}
    """
    names  = list(results.keys())
    scores = list(results.values())
    colours = [PALETTE.get(n, "#2c3e50") for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, scores, color=colours, edgecolor="white", height=0.55)

    ax.axvline(0.92, color="green", linestyle="--", linewidth=1.2,
               label="Target F1 ≥ 0.92")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    ax.set_xlabel("F1-Score (HDFS Test Set)")
    ax.set_title("FedVAE-Trans Ablation Study — F1 Comparison")
    ax.set_xlim(0.6, 1.0)
    ax.legend()
    plt.tight_layout()

    out = save_path or str(FIG_DIR / "ablation_f1_summary.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ Saved: {out}")


# ---------------------------------------------------------------------------
# Communication Efficiency
# ---------------------------------------------------------------------------

def plot_communication_efficiency(
    fedavg_metrics:  List[Dict],
    fedprox_metrics: List[Dict],
    save_path:       Optional[str] = None,
) -> None:
    """
    Dual-axis plot: F1 vs cumulative communication cost (MB).
    Illustrates the accuracy-bandwidth trade-off.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    for data, label, colour in [
        (fedavg_metrics,  "FedAvg",  PALETTE["FedAvg"]),
        (fedprox_metrics, "FedProx", PALETTE["FedProx"]),
    ]:
        if not data: continue
        cumulative_mb = np.cumsum([d.get("payload_fp32_mb", 0) for d in data])
        f1_vals       = [d["f1"] for d in data]

        ax1.plot(cumulative_mb, f1_vals, color=colour, linewidth=2,
                 label=f"{label} — F1 vs Bandwidth")
        ax2.plot(cumulative_mb, [d["round"] for d in data],
                 color=colour, linewidth=1, linestyle="--", alpha=0.4)

    ax1.axhline(0.92, color="green", linestyle=":", linewidth=1.2,
                label="Target F1 ≥ 0.92")
    ax1.set_xlabel("Cumulative Communication (MB)")
    ax1.set_ylabel("Global F1-Score")
    ax2.set_ylabel("Federated Round", color="grey")
    ax1.set_title("Communication Efficiency: F1 vs Bandwidth")
    ax1.legend(loc="lower right")
    plt.tight_layout()

    out = save_path or str(FIG_DIR / "communication_efficiency.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ Saved: {out}")


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    save_path:  Optional[str] = None,
    title:      str = "Confusion Matrix — FedVAE-Trans",
) -> None:
    """Save a styled confusion matrix heatmap."""
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Normal", "Anomaly"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()

    out = save_path or str(FIG_DIR / "confusion_matrix.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ Saved: {out}")


# ---------------------------------------------------------------------------
# Loss Curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    history:    Dict[str, List[float]],
    name:       str,
    save_path:  Optional[str] = None,
) -> None:
    """
    Plot ELBO, reconstruction, and KL loss curves over training epochs.
    """
    epochs = list(range(1, len(history.get("elbo", [])) + 1))
    if not epochs: return

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    for ax, key, label, colour in zip(
        axes,
        ["elbo", "recon", "kl"],
        ["Total ELBO Loss", "Reconstruction Loss (MSE)", "KL Divergence"],
        ["#2c3e50", "#2980b9", "#e74c3c"],
    ):
        if key in history:
            ax.plot(epochs, history[key], color=colour, linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.set_title(f"{name} — {label}")

    plt.tight_layout()
    out = save_path or str(FIG_DIR / f"loss_curve_{name}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"✅ Saved: {out}")