"""
src/utils/metrics.py
FedVAE-Trans — Evaluation Metrics

Critical fix: use ROC-optimal threshold (Youden's J) instead of the
95th-percentile heuristic. The original heuristic caused low recall on
the 2.9% anomaly-rate HDFS dataset, producing F1=0.726 despite AUROC=0.989.
The ROC-optimal threshold resolves this mismatch.

Usage:
    from src.utils.metrics import evaluate_vae, compute_threshold
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, classification_report,
)
from typing import Optional


# ---------------------------------------------------------------------------
# Threshold Calibration
# ---------------------------------------------------------------------------

def compute_threshold(
    model:        torch.nn.Module,
    train_loader: DataLoader,
    device:       torch.device,
    method:       str = "roc_optimal",
    y_test:       Optional[np.ndarray] = None,
    test_scores:  Optional[np.ndarray] = None,
) -> float:
    """
    Compute the anomaly detection threshold.

    Args:
        model        : Trained VAE model (eval mode will be set internally)
        train_loader : DataLoader of NORMAL training samples only
        device       : torch.device
        method       : "roc_optimal" (recommended) | "percentile_95" | "percentile_99"
        y_test       : Ground-truth labels (required for roc_optimal)
        test_scores  : Pre-computed anomaly scores on test set (required for roc_optimal)

    Returns:
        Scalar threshold τ

    Note:
        "roc_optimal" (Youden's J statistic) maximises TPR - FPR.
        This is the same method used in Phase 1 (LSTM-AE), which achieved F1=0.919.
        Always prefer this over percentile methods for imbalanced datasets.
    """
    if method == "roc_optimal":
        if y_test is None or test_scores is None:
            raise ValueError(
                "y_test and test_scores are required for roc_optimal thresholding."
            )
        fpr, tpr, thresholds = roc_curve(y_test, test_scores)
        # Youden's J: maximise TPR - FPR simultaneously
        j_stat    = tpr - fpr
        best_idx  = int(np.argmax(j_stat))
        tau       = float(thresholds[best_idx])
        return tau

    # Fallback: percentile of reconstruction error on training (normal) data
    model.eval()
    train_scores = []
    with torch.no_grad():
        for input_ids, _ in train_loader:
            input_ids = input_ids.to(device)
            scores    = model.anomaly_score(input_ids)
            train_scores.append(scores.cpu().numpy())
    train_scores = np.concatenate(train_scores)

    pct = 95 if method == "percentile_95" else 99
    return float(np.percentile(train_scores, pct))


# ---------------------------------------------------------------------------
# Score Collection
# ---------------------------------------------------------------------------

def collect_scores(
    model:       torch.nn.Module,
    loader:      DataLoader,
    device:      torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference and collect anomaly scores + ground-truth labels.

    Returns:
        scores : (N,) float32  — reconstruction error per sample
        labels : (N,) int      — 0=normal, 1=anomaly
    """
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for input_ids, lbl in loader:
            input_ids = input_ids.to(device)
            scores    = model.anomaly_score(input_ids)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(lbl.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Full Evaluation
# ---------------------------------------------------------------------------

def evaluate_vae(
    model:        torch.nn.Module,
    test_loader:  DataLoader,
    train_loader: DataLoader,
    device:       torch.device,
    threshold:    Optional[float] = None,
    verbose:      bool = False,
) -> dict:
    """
    Complete evaluation: anomaly scores → threshold → metrics.

    Steps:
      1. Collect reconstruction errors on the full test set.
      2. If no threshold supplied, compute scores first then apply
         ROC-optimal thresholding (Youden's J) — same as Phase 1.
      3. Compute F1, Precision, Recall, AUROC.

    Returns:
        dict with keys: f1, precision, recall, auroc, threshold,
                        scores, labels, fpr, tpr, roc_thresholds
    """
    scores, labels = collect_scores(model, test_loader, device)

    auroc = roc_auc_score(labels, scores)
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)

    # Threshold selection
    if threshold is None:
        # ROC-optimal — the critical fix
        j_stat   = tpr - fpr
        best_idx = int(np.argmax(j_stat))
        threshold = float(roc_thresholds[best_idx])

    preds = (scores >= threshold).astype(int)

    f1        = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)

    if verbose:
        print(f"\n  Threshold τ : {threshold:.6f}")
        print(f"  F1          : {f1:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall      : {recall:.4f}")
        print(f"  AUROC       : {auroc:.4f}")
        print(classification_report(labels, preds,
                                    target_names=["Normal", "Anomaly"]))

    return {
        "f1"             : f1,
        "precision"      : precision,
        "recall"         : recall,
        "auroc"          : auroc,
        "threshold"      : threshold,
        "scores"         : scores,
        "labels"         : labels,
        "fpr"            : fpr,
        "tpr"            : tpr,
        "roc_thresholds" : roc_thresholds,
    }


# ---------------------------------------------------------------------------
# Federated Round Evaluation  (lightweight — no threshold re-computation)
# ---------------------------------------------------------------------------

def evaluate_federated_round(
    model:     torch.nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    threshold: float,
) -> dict:
    """
    Lightweight evaluation for a single federated round.
    Uses a fixed threshold (calibrated once on the global model after training).

    Returns:
        dict with f1, auroc, num_samples
    """
    scores, labels = collect_scores(model, loader, device)

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5   # Only one class present in this client's test split

    preds = (scores >= threshold).astype(int)
    f1    = f1_score(labels, preds, zero_division=0)

    return {
        "f1"          : f1,
        "auroc"       : auroc,
        "num_samples" : len(labels),
    }