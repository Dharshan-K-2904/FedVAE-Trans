import kagglehub
import shutil
import os
from pathlib import Path

def fetch_and_store_dataset():
    repo_root = Path(os.getcwd())
    target_dir = repo_root / "data" / "raw" / "CIC-IoT-DIAD-2024"
    os.makedirs(target_dir, exist_ok=True)
    print("Initiating dataset download via kagglehub...")
    cached_path = kagglehub.dataset_download("aymeneinformatique/cic-diad-new-2024")
    print(f"Transferring artifacts to project directory: {target_dir}")
    shutil.copytree(cached_path, target_dir, dirs_exist_ok=True)
    print("Transfer complete.")

if __name__ == "__main__":
    fetch_and_store_dataset()
