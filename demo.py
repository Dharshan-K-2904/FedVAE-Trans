import torch
import numpy as np
import json
import time
import os

from src.models.bilstm_trans_vae import BiLSTMTransVAE

# ANSI Color Codes for a beautiful terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def run_live_demo():
    os.system('clear')
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("========================================================")
    print(" 🛡️ FEDVAE-TRANS: LIVE ANOMALY DETECTION DEMO")
    print("========================================================")
    print(f"{Colors.ENDC} Loading Global Federated Weights...")
    
    weights_path = "data/phase3_results/checkpoints/global_trans_vae.pt"
    metrics_path = "data/phase2_results/centralized_metrics.json"
    seq_path = "data/processed/sequences.npy"
    lab_path = "data/processed/labels.npy"
    
    try:
        # Load the optimal threshold
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            threshold = metrics.get("Optimal_Threshold", metrics.get("Threshold", 5.0))
            
        # Load the Model
        device = torch.device("cpu") # Run demo on CPU for universal compatibility
        model = BiLSTMTransVAE(vocab_size=10000).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        # Load sample data
        sequences = np.load(seq_path)
        labels = np.load(lab_path)
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error loading demo assets: {e}{Colors.ENDC}")
        print("Ensure 'data/phase3_results' and 'data/phase2_results' exist.")
        return

    print(f"{Colors.GREEN}✅ Model and Data successfully loaded!{Colors.ENDC}")
    print(f"   -> Enforcing Decision Boundary: {Colors.WARNING}{threshold:.4f}{Colors.ENDC}\n")
    time.sleep(1.5)
    
    # Prepare a balanced demo batch (5 Normal, 5 Anomalies)
    np.random.seed(42)
    normal_idx = np.random.choice(np.where(labels == 0)[0], 5, replace=False)
    anomaly_idx = np.random.choice(np.where(labels == 1)[0], 5, replace=False)
    demo_idx = np.concatenate([normal_idx, anomaly_idx])
    np.random.shuffle(demo_idx) # Mix them up to show the model isn't cheating
    
    print(f"{Colors.BOLD}Initiating Real-Time Sequence Analysis Stream...{Colors.ENDC}\n")
    print("-" * 75)
    print(f"{'SEQ ID':<10} | {'TRUE CLASS':<12} | {'RECON SCORE':<12} | {'PREDICTION':<15} | {'STATUS'}")
    print("-" * 75)
    
    correct_count = 0
    with torch.no_grad():
        for i, idx in enumerate(demo_idx):
            seq = torch.tensor(sequences[idx], dtype=torch.long).unsqueeze(0).to(device)
            true_label = labels[idx]
            
            # Model Forward Pass
            recon, _, _ = model(seq)
            
            # Calculate Anomaly Score (Sequence-Level Cross Entropy)
            import torch.nn.functional as F
            ce_loss = F.cross_entropy(recon.view(-1, recon.size(-1)), seq.view(-1), reduction='none')
            anomaly_score = ce_loss.mean().item()
            
            # Binary Classification based on the Phase 2 boundary
            is_anomaly = anomaly_score > threshold
            pred_label = 1 if is_anomaly else 0
            
            if pred_label == true_label:
                correct_count += 1
                status = f"{Colors.GREEN}✅ Correct{Colors.ENDC}"
            else:
                status = f"{Colors.FAIL}❌ Missed{Colors.ENDC}"
                
            true_str = "Anomaly" if true_label == 1 else "Normal"
            pred_str = "Anomaly" if pred_label == 1 else "Normal"
            
            pred_color = Colors.FAIL if pred_label == 1 else Colors.BLUE
            
            print(f"Log_#{idx:<5} | {true_str:<12} | {anomaly_score:<12.4f} | {pred_color}{pred_str:<15}{Colors.ENDC} | {status}")
            time.sleep(1) # Artificial delay to simulate processing a live stream
            
    print("-" * 75)
    print(f"\n{Colors.BOLD}Demo Complete. Accuracy on incoming data stream: {(correct_count/len(demo_idx))*100:.1f}%{Colors.ENDC}\n")

if __name__ == "__main__":
    run_live_demo()
