import modal, os, tarfile, numpy as np, pandas as pd
from pathlib import Path
from itertools import islice

volume = modal.Volume.from_name("fedvae-trans-storage", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("drain3", "numpy", "pandas")
app = modal.App("fedvae-phase1-pipeline", image=image)

@app.function(cpu=16.0, memory=65536, volumes={"/data_vol": volume}, timeout=14400)
def process_unified_datasets():
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    raw_dir, processed_dir = Path("/data_vol/raw"), Path("/data_vol/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    print("⚡ HYPER-SPEED PARSING INITIATED...")

    for archive in raw_dir.glob("*.tar.gz"):
        log_name = archive.name.replace(".tar.gz", ".log").replace("HDFS_1", "HDFS")
        if (raw_dir / log_name).exists(): continue
        try:
            with tarfile.open(archive, "r:gz") as tar: tar.extractall(path=raw_dir)
        except Exception: pass

    config = TemplateMinerConfig()
    config.profiling = False
    miner = TemplateMiner(config=config)
    sequences, labels, current_window, MAX_LINES = [], [], [], 50000 
    
    for log_file in raw_dir.rglob("*.log"):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in list(islice(f, MAX_LINES)):
                parts = line.strip().split()
                if len(parts) > 5:
                    current_window.append(miner.add_log_message(' '.join(parts[5:]))["cluster_id"])
                    if len(current_window) == 128:
                        sequences.append(current_window); labels.append(0); current_window = []

    for csv_file in raw_dir.rglob("*.csv"):
        if "anomaly_label" in csv_file.name: continue
        try:
            df = pd.read_csv(csv_file, nrows=MAX_LINES, engine='c')
            flow_strings = df.astype(str).agg(' '.join, axis=1).tolist()
            is_anomaly = 0 if "Benign" in str(csv_file) else 1
            for flow in flow_strings:
                current_window.append(miner.add_log_message(flow)["cluster_id"])
                if len(current_window) == 128:
                    sequences.append(current_window); labels.append(is_anomaly); current_window = []
        except Exception: continue

    if sequences:
        np.save(processed_dir / 'sequences.npy', np.array(sequences, dtype=np.int32))
        np.save(processed_dir / 'labels.npy', np.array(labels, dtype=np.int32))
        print(f"✅ Pipeline Concluded. Total Sequences: {len(sequences)}")

if __name__ == "__main__":
    with app.run(): process_unified_datasets.remote()
