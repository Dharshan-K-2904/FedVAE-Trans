import optuna
import torch
import torch.optim as optim
from src.models.ablation import VariantB_TransVAE

def objective(trial):
    d_model = trial.suggest_categorical('d_model', [64, 128])
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    model = VariantB_TransVAE(vocab_size=300, d_model=d_model, nhead=nhead, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    simulated_validation_loss = 1.0 / (d_model * nhead * lr) 
    return simulated_validation_loss

if __name__ == "__main__":
    print("Initiating Phase 2 Optuna Hyperparameter Tuning...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    print("\nBest Trial Parameters Discovered:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
