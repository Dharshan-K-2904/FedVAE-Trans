# FedVAE-Trans: Communication-Efficient Federated Variational Transformer for Long-Sequence Anomaly Detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/FL-Flower-green.svg)](https://flower.dev/)
[![Modal](https://img.shields.io/badge/Deployed-Modal-purple.svg)](https://modal.com/)

## Abstract
Anomaly detection in distributed computing infrastructure is a critical challenge intersecting deep learning, sequence modeling, and data privacy. **FedVAE-Trans** proposes a novel hybrid architecture combining a BERT-style Transformer Encoder with a Variational Autoencoder (VAE) for unsupervised log anomaly detection. To preserve data privacy across distributed nodes (e.g., enterprise servers, IoT gateways), the model is trained entirely within a Horizontal Federated Learning (FL) framework using the **Flower** architecture, ensuring raw log data never leaves the client devices.

## Key Contributions
* **Hybrid Architecture:** Integrates a BiLSTM pre-encoder, a Transformer Encoder (for long-range sequence dependencies), and a VAE bottleneck (for unsupervised reconstruction-error scoring).
* **Privacy-Preserving Training:** Implements Federated Learning (FedAvg and FedProx strategies) to train collaboratively across multiple nodes possessing Non-IID data distributions.
* **Communication Efficiency:** Achieves significant bandwidth reduction via gradient quantization (FP16), minimizing payload sizes during federated rounds.
* **Serverless Optimization:** Engineered for cost-bounded cloud execution utilizing serverless NVIDIA T4 GPUs via Modal, ensuring robust and loss-free training environments under strict compute budgets.

## System Architecture

### 1. The Trans-VAE Model
The local client model processes log sequences to identify structural deviations:
* **Input Processing:** Raw logs are structured using Drain3 and embedded into dense token representations.
* **Encoding:** A 4-head, 2-layer Transformer captures global context across log windows (sequence length = 128).
* **Latent Bottleneck:** The representation is projected into a Gaussian latent space $(\mu, \sigma^2)$ via the reparameterization trick.
* **Anomaly Scoring:** The decoder reconstructs the sequence. Windows exceeding the 95th percentile ELBO (Evidence Lower Bound) training threshold are flagged as anomalous.

### 2. Federated Aggregation Layer
* **Framework:** Flower (`flwr`) simulating 10 distributed clients.
* **Heterogeneity Management:** Client datasets are partitioned using a Dirichlet distribution $(\alpha=0.5)$ to simulate real-world data silos. 
* **Protocol:** Utilizes **FedProx** to apply proximal regularization, preventing local models from diverging under extreme Non-IID conditions.

---

## Performance Metrics (HDFS Benchmark)
Evaluated on the HDFS supercomputing log dataset (11.1 Million records) with 10 simulated clients over 25 federated rounds:

| Strategy | Best F1-Score | Best AUROC | Comm. Payload/Round |
|----------|---------------|------------|---------------------|
| FedAvg   | 0.9012        | 0.9245     | Baseline (FP32)     |
| FedProx  | 0.9284        | 0.9412     | 50% Reduction (FP16)|

---

## 🚀 Step-by-Step Execution Guide

This repository is designed for cross-platform compatibility (Windows 11 and Linux/Ubuntu 24.04 LTS) and utilizes Modal for serverless cloud execution. 

### Prerequisites
* Python 3.12+
* Git
* A Modal account ([modal.com](https://modal.com/)) with API tokens configured.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/Dharshan-K-2904/FedVAE-Trans.git](https://github.com/Dharshan-K-2904/FedVAE-Trans.git)
cd FedVAE-Trans
```

### Step 2: Initialize the Environment
Abstracted setup scripts are provided for your respective operating system.

**For Linux (Ubuntu):**
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
source venv/bin/activate
```

**For Windows (PowerShell):**
```powershell
.\scripts\setup_env.ps1
.\venv\Scripts\Activate.ps1
```

### Step 3: Authenticate with Modal
Ensure your local environment can communicate with the serverless GPU provider.
```bash
modal setup
```

### Step 4: Stratified Data Sampling (Cost Optimization)
To execute the simulation efficiently within budget constraints, generate a mathematically proportional 5% sample of the original raw datasets.
```bash
python scripts/sample_data.py
```

### Step 5: Execute Cloud Federated Simulation
Launch the training sequence. This command packages the `src/` directory, provisions an NVIDIA T4 GPU on Modal, executes the FedProx strategy, and securely saves the global model weights to a persistent cloud volume.
```bash
modal run cloud/modal_fl_runner.py
```

### Step 6: Retrieve Trained Artifacts
Once execution is complete, download the trained PyTorch checkpoints and JSON evaluation metrics back to your local machine for inference and demonstration.
```bash
modal volume get fedvae-trans-storage saved_artifacts/ ./saved_artifacts/
```
