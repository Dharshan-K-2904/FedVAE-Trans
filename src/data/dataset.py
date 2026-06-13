"""
src/data/dataset.py
FedVAE-Trans — Dataset Classes and Non-IID Partitioning

Provides:
  - LogSequenceDataset    : PyTorch Dataset wrapping sequences.npy + labels.npy
  - dirichlet_partition   : Non-IID data splits via Dirichlet(α) distribution
  - build_client_loaders  : Build per-client DataLoader objects for FL simulation

Syllabus alignment:
  Unit 4 — Federated Learning Motivation, Non-IID Data Heterogeneity
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import List, Tuple

SEED = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LogSequenceDataset(Dataset):
    """
    Wraps the pre-processed HDFS log sequences and binary anomaly labels.

    Each sample is a fixed-length window of log template IDs (int64)
    and a binary label (0 = normal, 1 = anomaly).
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.from_numpy(sequences).long()
        self.labels    = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Non-IID Dirichlet Partitioning
# ---------------------------------------------------------------------------

def dirichlet_partition(
    labels:      np.ndarray,
    num_clients: int,
    alpha:       float = 0.5,
    seed:        int   = SEED,
) -> List[np.ndarray]:
    """
    Partition dataset indices across clients using a Dirichlet distribution.

    Dirichlet(α) controls the degree of Non-IID skew:
      α → 0   : extreme Non-IID (each client sees one class only)
      α = 0.5 : moderate Non-IID (realistic enterprise silo scenario)
      α → ∞   : IID (uniform distribution across clients)

    The 2.93% anomaly rate means that with α=0.5 some clients will have
    zero anomalies in their training split — this is intentional and
    reflects real-world federated deployments.

    Args:
        labels      : (N,) array of binary labels
        num_clients : Number of FL clients to create
        alpha       : Dirichlet concentration parameter
        seed        : Random seed for reproducibility

    Returns:
        List of num_clients index arrays (variable length, non-overlapping)
    """
    rng        = np.random.default_rng(seed)
    classes    = np.unique(labels)
    client_idx = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)

        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))

        # Assign indices proportionally — cumulative split
        splits  = (np.cumsum(proportions) * len(cls_indices)).astype(int)
        splits  = np.clip(splits, 0, len(cls_indices))
        chunks  = np.split(cls_indices, splits[:-1])

        for cid, chunk in enumerate(chunks):
            client_idx[cid].extend(chunk.tolist())

    return [np.array(idx, dtype=np.int64) for idx in client_idx]


# ---------------------------------------------------------------------------
# Build Client DataLoaders
# ---------------------------------------------------------------------------

def build_client_loaders(
    sequences:   np.ndarray,
    labels:      np.ndarray,
    num_clients: int,
    alpha:       float = 0.5,
    batch_size:  int   = 256,
    test_size:   float = 0.2,
    seed:        int   = SEED,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Build per-client training DataLoaders and a shared global test loader.

    Training protocol (semi-supervised):
      - Each client trains ONLY on its NORMAL samples (label == 0).
      - A shared global test set includes all labels for fair F1 evaluation.

    Args:
        sequences   : (N, T) int64 array of log windows
        labels      : (N,)  int   array of binary labels
        num_clients : FL simulation client count
        alpha       : Dirichlet concentration (0.1 = extreme Non-IID, 0.5 = moderate)
        batch_size  : DataLoader batch size
        test_size   : Fraction of data reserved for global test set
        seed        : Random seed

    Returns:
        client_loaders : List of per-client train DataLoaders
        test_loader    : Global test DataLoader (all labels)
    """
    full_dataset = LogSequenceDataset(sequences, labels)

    # Stratified global train / test split (same ratio as Phase 1)
    all_idx     = np.arange(len(sequences))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, stratify=labels, random_state=seed
    )

    test_dataset  = Subset(full_dataset, test_idx)
    test_loader   = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Partition TRAINING indices across clients (Non-IID)
    train_labels = labels[train_idx]
    client_splits = dirichlet_partition(train_labels, num_clients, alpha, seed)

    client_loaders = []
    for cid, split_local_idx in enumerate(client_splits):
        # Map local partition indices back to global dataset indices
        global_idx = train_idx[split_local_idx]
        client_labels = labels[global_idx]

        # Each client trains on NORMAL samples only (unsupervised)
        normal_mask    = client_labels == 0
        normal_global  = global_idx[normal_mask]

        if len(normal_global) == 0:
            # Edge case: client has no normal samples (extreme Non-IID with small α)
            # Fall back to all samples to prevent empty DataLoader crash
            normal_global = global_idx

        client_subset = Subset(full_dataset, normal_global)
        loader = DataLoader(
            client_subset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=False,
        )
        client_loaders.append(loader)

    return client_loaders, test_loader