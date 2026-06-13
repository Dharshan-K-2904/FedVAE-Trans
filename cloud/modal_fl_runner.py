import modal
import os
import json

volume = modal.Volume.from_name("fedvae-trans-storage", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.0.0", "flwr[simulation]==1.28.0", "numpy", "pandas", "scikit-learn", "matplotlib")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("fedvae-trans-training", image=image)

@app.function(gpu="T4", volumes={"/data_vol": volume}, timeout=14400)
def run_federated_simulation():
    import flwr as fl
    import torch
    from src.federated.strategy import FedVAEStrategy
    from src.federated.client import client_fn_factory
    from src.models.bilstm_trans_vae import BiLSTMTransVAE
    from src.utils.plotting import generate_training_plots
    from flwr.common import parameters_to_ndarrays
    
    print("Initiating Serverless Federated Training Protocol...")
    
    # Configure the client factory to recursively map the entire volume root
    DATA_PATH = "/data_vol"
    
    strategy = FedVAEStrategy(
        proximal_mu=0.01, fraction_fit=0.3, min_fit_clients=3, min_available_clients=10,
    )
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn_factory(proximal_mu=0.01, data_path=DATA_PATH),
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=25),
        strategy=strategy,
        client_resources={'num_cpus': 2, 'num_gpus': 0.25}, 
    )
    
    metrics_dir = "/data_vol/saved_artifacts/metrics"
    ckpt_dir = "/data_vol/saved_artifacts/checkpoints"
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    with open(f"{metrics_dir}/final_results.json", 'w') as f:
        json.dump(history.metrics_centralized, f)
        
    generate_training_plots(strategy.round_metrics, metrics_dir)
    
    final_model = BiLSTMTransVAE(vocab_size=300)
    if strategy.final_parameters is not None:
        ndarrays = parameters_to_ndarrays(strategy.final_parameters)
        state_dict = dict(zip(final_model.state_dict().keys(), [torch.tensor(p) for p in ndarrays]))
        final_model.load_state_dict(state_dict, strict=True)
        
    torch.save(final_model.state_dict(), f"{ckpt_dir}/global_trans_vae.pt")
    print(f"Success: Weights saved to {ckpt_dir}/global_trans_vae.pt")

if __name__ == "__main__":
    with app.run():
        run_federated_simulation.remote()
