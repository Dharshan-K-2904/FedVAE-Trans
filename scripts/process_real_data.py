import os
import sys
import tarfile
import numpy as np
from pathlib import Path

# --- 1. Dependency Checks ---
try:
    import pandas as pd
except ImportError:
    print("❌ ERROR: Pandas is not installed. Please run: pip install pandas")
    sys.exit(1)

try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
except ImportError:
    print("❌ ERROR: Drain3 is not installed. Please run: pip install drain3")
    sys.exit(1)

def find_file(filename, search_dir):
    """Recursively searches for a file in the given directory."""
    for path in Path(search_dir).rglob(filename):
        return path
    return None

def process_real_hdfs():
    repo_root = Path(os.getcwd())
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    
    tar_path = raw_dir / "HDFS_1.tar.gz"
    
    # --- 2. Safe Extraction ---
    if tar_path.exists():
        print(f"📦 Checking extraction of {tar_path}...")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=raw_dir)
        except Exception as e:
            print(f"⚠️ Warning during extraction (might already be extracted): {e}")
    else:
        print(f"⚠️ Warning: {tar_path.name} not found. Assuming files are already extracted.")

    # --- 3. Dynamic File Discovery ---
    log_path = find_file("HDFS.log", raw_dir)
    label_path = find_file("anomaly_label.csv", raw_dir)
    
    if not log_path or not label_path:
        print("❌ ERROR: Could not locate 'HDFS.log' or 'anomaly_label.csv'.")
        print(f"Please ensure they exist anywhere inside: {raw_dir}")
        sys.exit(1)
        
    print(f"✅ Found Log File: {log_path}")
    print(f"✅ Found Label File: {label_path}")
    
    # --- 4. Load Labels ---
    print("📊 Loading Anomaly Labels...")
    labels_df = pd.read_csv(label_path)
    labels_df['Label'] = labels_df['Label'].apply(lambda x: 1 if x == 'Anomaly' else 0)
    label_dict = dict(zip(labels_df['BlockId'], labels_df['Label']))
    
    # --- 5. Drain3 Parsing ---
    config = TemplateMinerConfig()
    config.profiling = False
    miner = TemplateMiner(config=config)
    
    MAX_LINES = 500000 # Adjust this if you want more/less data
    sequences = []
    labels = []
    current_window = []
    
    print(f"🧠 Parsing the first {MAX_LINES} lines of HDFS.log using Drain3...")
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= MAX_LINES:
                break
                
            parts = line.strip().split()
            if len(parts) < 6: continue
            
            block_id = next((part for part in parts if part.startswith('blk_')), None)
            log_message = ' '.join(parts[5:])
            
            result = miner.add_log_message(log_message)
            current_window.append(result["cluster_id"])
            
            if len(current_window) == 128:
                sequences.append(current_window)
                labels.append(label_dict.get(block_id, 0) if block_id else 0)
                current_window = []

    # --- 6. Save Artifacts ---
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / 'sequences.npy', np.array(sequences, dtype=np.int32))
    np.save(processed_dir / 'labels.npy', np.array(labels, dtype=np.int32))
    
    print(f"\n🎉 Success! Extracted {len(miner.drain.clusters)} unique log templates.")
    print(f"💾 Generated {len(sequences)} sequence windows of length 128.")
    print(f"🚀 Ready to upload! Run: modal volume put fedvae-trans-storage data/processed /processed")

if __name__ == "__main__":
    process_real_hdfs()
