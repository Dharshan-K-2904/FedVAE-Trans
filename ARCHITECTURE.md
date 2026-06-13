# FedVAE-Trans: Architectural Deep Dive

## System Architecture Overview

FedVAE-Trans implements a **three-layer federated learning system** combining:
1. **Client-side inference** (edge nodes)
2. **Federated aggregation** (central server)
3. **Asynchronous model updates** (communication protocol)

---

## 1. Model Architecture

### 1.1 Transformer Encoder

```
Input (seq_len × embedding_dim)
         ↓
Token Embeddings + Positional Encodings
         ↓
┌─ Multi-Head Self-Attention (4 heads) ─┐
│  Attention(Q, K, V) = softmax(QK^T/√d)V │
└─────────────────────────────────────────┘
         ↓
Feed-Forward Network (MLP)
         ↓
Layer Normalization & Residual Connection
         ↓
[Repeat N_LAYERS times]
         ↓
Encoder Output (seq_len × hidden_dim)
```

**Configuration:**
- `num_layers`: 2 (stacked transformer blocks)
- `num_heads`: 4 (multi-head attention)
- `hidden_dim`: 128 (transformer hidden dimension)
- `dropout`: 0.1 (regularization)
- `max_seq_len`: 512 (maximum sequence length)

### 1.2 VAE Bottleneck

```
Encoder Output (seq_len × hidden_dim)
         ↓
Global Average Pooling (1 × hidden_dim)
         ↓
┌─────────────────────────┐
│  μ = MLP_μ(pooled)      │  ← Mean vector (latent_dim)
│  log_σ² = MLP_σ(pooled) │  ← Log variance (latent_dim)
└─────────────────────────┘
         ↓
Reparameterization: z = μ + ε ⊙ σ
where ε ~ N(0, I)
         ↓
z (latent_dim)
```

**Configuration:**
- `latent_dim`: 64 (VAE bottleneck size)
- `hidden_dim`: 128 (encoder/decoder hidden layer)
- KL divergence weight: 1.0 (ELBO loss component)

### 1.3 Reconstruction Decoder

```
z (latent_dim)
         ↓
Expand via MLP:
    z → (hidden_dim)
         ↓
Position-wise Expansion:
    (hidden_dim) → (seq_len × hidden_dim)
         ↓
Reverse Transformer (decoder blocks)
    [Mirror of encoder with causal masking]
         ↓
Output Projection:
    (seq_len × hidden_dim) → (seq_len × embedding_dim)
         ↓
Reconstructed Sequence (seq_len × embedding_dim)
```

**Loss Function:**
```
L_ELBO = -E_q[log p(x|z)] + KL(q(z|x) || p(z))
       = Reconstruction Loss + KL Divergence
```

---

## 2. Federated Learning Strategy

### 2.1 FedProx Aggregation

```
Client 1           Client 2            Client N
   ↓                  ↓                  ↓
Train on           Train on            Train on
Local Data         Local Data          Local Data
   ↓                  ↓                  ↓
Compute ∇L_i      Compute ∇L_i       Compute ∇L_i
   ↓                  ↓                  ↓
Apply FedProx Penalty:
L_i^prox = L_i + (μ/2)||w_i - w_global||²
   ↓                  ↓                  ↓
Compute             Compute             Compute
w_i^local           w_i^local           w_i^local
   ↓                  ↓                  ↓
        Send FP16-quantized updates
              ↓
        ┌─────────────────────┐
        │  Aggregation Server │
        │ w_global = Σ p_i×w_i│
        │ where p_i = n_i/Σn_j│
        └─────────────────────┘
              ↓
        Broadcast global model
              ↓
```

**FedProx Advantages:**
- Handles Non-IID data distributions
- Provides convergence guarantees
- Reduces catastrophic forgetting
- Maintains ELBO stability

### 2.2 Communication Protocol

```
Round r:
  Server broadcasts w_global^(r)
         ↓
  Clients receive w_global^(r)
         ↓
  Clients run LOCAL_EPOCHS local updates:
    for e in 1..LOCAL_EPOCHS:
      - Sample mini-batch from local data
      - Forward pass through model
      - Compute ELBO loss + FedProx penalty
      - Backward pass (gradient computation)
      - Update: w_i ← w_i - α × ∇L_i^prox
         ↓
  Clients compute weight delta:
    Δw_i = w_i - w_global^(r)
         ↓
  FP16 Compression:
    Δw_i^compressed = compress(Δw_i)  [50% size reduction]
         ↓
  Clients send Δw_i^compressed to server
         ↓
  Server aggregates:
    w_global^(r+1) = w_global^(r) + Σ p_i × Δw_i
         ↓
  Next round (if r < NUM_ROUNDS)
```

**Communication Efficiency:**
- FP16 quantization: 50% reduction vs FP32
- Selective gradient transmission
- Per-layer compression with adaptive thresholds

---

## 3. Data Pipeline

### 3.1 Log Parsing (Drain3)

```
Raw Logs:
  "2024-01-15 10:23:45 ERROR Connection timeout at 192.168.1.1:5432"
         ↓
Drain3 Parser:
  - Extracts log templates
  - Identifies variable parameters
  - Groups similar events
         ↓
Parsed Events:
  Template: "ERROR Connection timeout at <IP>:<PORT>"
  Events: [
    {"timestamp": "2024-01-15 10:23:45", "ip": "192.168.1.1", "port": 5432},
    ...
  ]
         ↓
Tokenization:
  Unique tokens: {ERROR, Connection, timeout, <IP>, <PORT>, ...}
  Token IDs: [45, 67, 89, 102, 115, ...]
         ↓
Embedding:
  Each token_id → 128-dim embedding vector
         ↓
```

### 3.2 Sequence Windowing

```
Token Sequence: [t1, t2, t3, ..., t_N]
         ↓
Sliding Window (size=100, stride=50):
  Window 1: [t1,   t2,   ..., t100]
  Window 2: [t51,  t52,  ..., t150]
  Window 3: [t101, t102, ..., t200]
  ...
         ↓
Temporal Augmentation (for Non-IID):
  - Add random temporal shifts
  - Apply mixup to prevent overfitting
         ↓
```

### 3.3 Non-IID Distribution (Dirichlet)

```
Classes: [Normal, Anomaly, ...]
         ↓
Generate Non-IID split using Dirichlet(α):
  - α = 0.5 (severe heterogeneity)
  - Each client gets unbalanced class distribution
         ↓
Client 1:  60% Normal, 35% Anomaly, 5% Rare
Client 2:  20% Normal, 75% Anomaly, 5% Rare
Client 3:  50% Normal, 30% Anomaly, 20% Rare
         ↓
Simulates real-world edge node diversity
```

---

## 4. Anomaly Detection Inference

### 4.1 Reconstruction Error Threshold

```
Test Log Sequence
         ↓
Forward pass through VAE:
  x → Encoder → z ~ N(μ, σ²) → Decoder → x̂
         ↓
Compute per-token reconstruction error:
  e_i = ||x_i - x̂_i||²  for each token i
         ↓
Aggregate error over sequence:
  E_seq = mean(e_i) or max(e_i)
         ↓
Dynamic threshold (percentile-based):
  threshold = percentile(historical_errors, 95)
         ↓
Decision:
  if E_seq > threshold:
    ANOMALY DETECTED ✗
  else:
    NORMAL ✓
```

### 4.2 ELBO Monitoring

```
During training, track:
  ELBO = E[log p(x|z)] - KL(q(z|x)||p(z))
         ↓
Plot ELBO over rounds:
  - FedAvg: Sharp collapse (diverges under Non-IID)
  - FedProx: Stable convergence (bounded ELBO)
         ↓
ELBO stability indicates model quality
```

---

## 5. Implementation Stack

### 5.1 Dependencies

| Component | Library | Version |
|-----------|---------|---------|
| Deep Learning | PyTorch | 2.5.1+ |
| Federated Learning | Flower | 1.14.0 |
| Simulation Backend | Ray | 2.31.0+ |
| Log Parsing | Drain3 | 0.9.11+ |
| Metrics | scikit-learn | 1.4.0+ |
| Visualization | Streamlit | 1.35.0+ |
| Cloud Compute | Modal | 0.77.1 |
| Hyperparameter Tuning | Optuna | 3.6.0+ |

### 5.2 Module Organization

```
src/
├── models/
│   ├── bilstm_trans_vae.py     # Main model class
│   ├── baselines.py             # Comparison models
│   └── ablation.py              # Ablation variants
├── federated/
│   ├── client.py                # Flower client logic
│   ├── strategy.py              # FedProx strategy
│   └── server.py                # Central server
├── data/
│   ├── dataset.py               # PyTorch Dataset
│   └── parser.py                # Drain3 integration
└── utils/
    ├── metrics.py               # F1, AUC-ROC, ELBO
    └── plotting.py              # Visualization
```

---

## 6. Performance Characteristics

### 6.1 Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Transformer Encoder | O(L × seq_len² × d) | L=layers, d=dim |
| VAE Forward Pass | O(seq_len × d + z_dim) | z_dim << d |
| Decoder | O(seq_len × d × seq_len) | Reconstruct full sequence |
| FedProx Aggregation | O(N × d) | N=clients, d=params |
| FP16 Compression | O(d) | Negligible with GPU |

### 6.2 Memory Requirements

- **Per Client**: 2-4 GB (model params + batch data)
- **Central Server**: 1 GB (global model only)
- **Total System**: 30-50 GB (10 clients × 4 GB + 50 GB data)

### 6.3 Communication Rounds

```
Total Communication = NUM_ROUNDS × (clients_per_round × payload_size)

Example:
- NUM_ROUNDS = 25
- clients_per_round = 10
- payload_size = 2 MB (FP16 compressed)
- Total = 25 × 10 × 2 MB = 500 MB

FP16 Savings:
- FP32 would require: 25 × 10 × 4 MB = 1 GB
- 50% reduction achieved
```

---

## 7. Testing & Validation

### 7.1 Unit Tests

```python
def test_vae_forward():
    """Verify VAE forward pass dimensions."""
    model = BiLSTMTransVAE(...)
    x = torch.randn(32, 100, 128)
    output = model(x)
    assert output["reconstruction"].shape == x.shape
    assert output["kl_loss"].item() >= 0
```

### 7.2 Integration Tests

```python
def test_federated_round():
    """Simulate one federated round."""
    server_model = FedVAETrans(...)
    clients = [LocalClient(...) for _ in range(10)]
    
    # Client update
    for client in clients:
        client.update(server_model.state_dict())
    
    # Aggregation
    aggregated = fedprox_aggregate([c.get_updates() for c in clients])
    server_model.load_state_dict(aggregated)
```

---

## 8. Deployment Considerations

### 8.1 Edge Deployment

- Quantize model to INT8 for mobile devices
- Use ONNX Runtime for inference
- Implement local fallback if aggregation fails

### 8.2 Production Monitoring

- Track ELBO trajectory (divergence detector)
- Monitor communication round latency
- Log reconstruction error distribution
- Alert on Non-IID drift

---

**For detailed implementation, see source code in `src/`**
