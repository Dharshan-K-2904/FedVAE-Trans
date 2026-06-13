import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
import os

def create_stratified_sample(sample_fraction=0.05):
    repo_root = Path(os.getcwd())
    processed_dir = repo_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Generating simulated data arrays for demonstration...")
    np.random.seed(42)
    full_sequences = np.random.randint(1, 300, size=(10000, 128))
    full_labels = np.random.choice([0, 1], size=(10000,), p=[0.95, 0.05])
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_fraction, random_state=42)
    for _, sample_idx in sss.split(full_sequences, full_labels):
        sampled_sequences = full_sequences[sample_idx]
        sampled_labels = full_labels[sample_idx]
        
    np.save(processed_dir / 'sequences.npy', sampled_sequences)
    np.save(processed_dir / 'labels.npy', sampled_labels)
    print(f"Saved optimized arrays to {processed_dir}")

if __name__ == "__main__":
    create_stratified_sample()
