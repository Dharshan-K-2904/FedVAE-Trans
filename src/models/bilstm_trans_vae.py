"""
src/models/bilstm_trans_vae.py
FedVAE-Trans — Core Neural Network Architectures

Three variants for the ablation study:
  A) VAEOnly          — MLP Encoder + VAE Bottleneck
  B) TransVAE         — Transformer Encoder + VAE Bottleneck  (core proposal)
  C) BiLSTMTransVAE   — BiLSTM Pre-Encoder + Transformer + VAE  (best performer)

Key fix applied: β-VAE formulation with β=0.1 prevents KL collapse.
The original β=1.0 caused the KL term to dominate reconstruction 39×,
collapsing the latent space and destroying the anomaly signal.

Syllabus alignment:
  Unit 1 — Transfer Learning (positional initialisation)
  Unit 2 — Transformer, BERT, Multi-Head Attention, BiLSTM
  Unit 3 — VAE, Reparameterisation Trick, ELBO Loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared Utility: Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
    Adds position information to token embeddings so the Transformer
    understands the sequential order of log events.
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# ---------------------------------------------------------------------------
# Shared Utility: ELBO Loss  (β-VAE formulation)
# ---------------------------------------------------------------------------

def elbo_loss(
    recon_embed: torch.Tensor,
    orig_embed:  torch.Tensor,
    mu:          torch.Tensor,
    log_var:     torch.Tensor,
    beta:        float = 0.1,          # FIX: was 1.0 → caused KL collapse
) -> tuple[torch.Tensor, float, float]:
    """
    Evidence Lower BOund (ELBO) loss for the VAE.

    L = MSE(reconstruction, original) + β · KL(q(z|x) ‖ p(z))

    β < 1 (here 0.1) is the β-VAE formulation (Higgins et al., 2017).
    It prevents posterior collapse by allowing the reconstruction term
    to dominate early training, which is critical for anomaly detection
    where high reconstruction error IS the anomaly signal.

    Args:
        recon_embed : Reconstructed embedding tensor  (B, T, d_model)
        orig_embed  : Original embedding tensor       (B, T, d_model)
        mu          : Latent mean                     (B, latent_dim)
        log_var     : Latent log-variance             (B, latent_dim)
        beta        : KL weight (0.1 recommended)

    Returns:
        total_loss, recon_loss_scalar, kl_loss_scalar
    """
    recon_loss = F.mse_loss(recon_embed, orig_embed, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Clamp KL to prevent numerical explosion during early warmup
    kl_loss    = torch.clamp(kl_loss, min=0.0, max=10.0)

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss.item(), kl_loss.item()


def cyclical_beta(epoch: int, cycle_len: int = 20, max_beta: float = 0.1) -> float:
    """
    Cyclical β-annealing schedule (Fu et al., 2019).
    β rises linearly from 0 to max_beta over half a cycle, then holds.
    Repeated cycles help the model escape KL vanishing.

    Args:
        epoch    : Current training epoch (1-indexed)
        cycle_len: Length of one full annealing cycle in epochs
        max_beta : Maximum β value (use 0.1 for anomaly detection)
    """
    pos_in_cycle = (epoch - 1) % cycle_len
    half         = cycle_len // 2
    if pos_in_cycle < half:
        return max_beta * (pos_in_cycle / half)
    return max_beta


# ---------------------------------------------------------------------------
# Variant A: VAE Only  (MLP Encoder — no sequential modelling)
# ---------------------------------------------------------------------------

class VAEOnly(nn.Module):
    """
    Variant A — Pure VAE with MLP encoder/decoder.
    Treats the entire log window as a flat vector; no sequence awareness.
    Establishes the lower bound for the ablation study.

    Syllabus link: Unit 3 — Variational Autoencoders, Reparameterisation
    """
    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = 64,
        latent_dim:  int = 64,
        seq_len:     int = 128,
        hidden_dim:  int = 256,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.embed_dim = embed_dim
        flat_dim = seq_len * embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.fc_mu      = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        x          = self.embedding(input_ids)          # (B, T, E)
        orig_embed = x.detach()
        x_flat     = x.reshape(B, -1)
        h          = self.encoder(x_flat)
        mu         = self.fc_mu(h)
        log_var    = self.fc_log_var(h)
        z          = self.reparameterise(mu, log_var)
        recon_flat = self.decoder(z)
        recon_embed = recon_flat.reshape(B, T, self.embed_dim)
        return recon_embed, orig_embed, mu, log_var

    @torch.no_grad()
    def anomaly_score(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-sample mean MSE reconstruction error — the anomaly signal."""
        self.eval()
        recon_embed, orig_embed, _, _ = self.forward(input_ids)
        return F.mse_loss(recon_embed, orig_embed, reduction="none").mean(dim=[1, 2])


# ---------------------------------------------------------------------------
# Variant B: TransVAE  (Transformer Encoder + VAE — core proposal)
# ---------------------------------------------------------------------------

class TransVAE(nn.Module):
    """
    Variant B — BERT-style Transformer Encoder + VAE Bottleneck.
    Core proposed architecture of FedVAE-Trans.

    The [CLS] token aggregates global context across the full log window
    before being projected into the VAE latent space.

    Syllabus links:
      Unit 2 — Transformer, BERT Architecture, Multi-Head Self-Attention
      Unit 3 — Variational Autoencoder, Reparameterisation Trick
    """
    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 128,
        nhead:       int   = 4,
        num_layers:  int   = 2,
        latent_dim:  int   = 64,
        seq_len:     int   = 128,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.seq_len    = seq_len
        self.latent_dim = latent_dim

        # Learnable [CLS] token prepended to every sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding    = PositionalEncoding(d_model, max_len=seq_len + 1,
                                                  dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,     # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)

        # VAE bottleneck
        self.fc_mu      = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

        # MLP Decoder: z → sequence of d_model vectors
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, seq_len * d_model),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for linear layers; normal init for embeddings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    def encode(self, input_ids: torch.Tensor):
        B, T       = input_ids.shape
        tok_embed  = self.token_embedding(input_ids)        # (B, T, d_model)
        orig_embed = tok_embed.detach()

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x          = torch.cat([cls_tokens, tok_embed], dim=1)   # (B, T+1, d)
        x          = self.pos_encoding(x)

        # Padding mask: True = ignore position
        pad_mask = torch.zeros(B, T + 1, dtype=torch.bool, device=input_ids.device)
        pad_mask[:, 1:] = (input_ids == 0)

        x       = self.transformer(x, src_key_padding_mask=pad_mask)
        cls_out = x[:, 0, :]                                # [CLS] pooling
        mu      = self.fc_mu(cls_out)
        log_var = self.fc_log_var(cls_out)
        return mu, log_var, orig_embed

    def decode(self, z: torch.Tensor, T: int) -> torch.Tensor:
        return self.decoder(z).reshape(-1, T, self.d_model)

    def forward(self, input_ids: torch.Tensor):
        mu, log_var, orig_embed = self.encode(input_ids)
        z           = self.reparameterise(mu, log_var)
        recon_embed = self.decode(z, input_ids.size(1))
        return recon_embed, orig_embed, mu, log_var

    @torch.no_grad()
    def anomaly_score(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.eval()
        recon_embed, orig_embed, _, _ = self.forward(input_ids)
        return F.mse_loss(recon_embed, orig_embed, reduction="none").mean(dim=[1, 2])


# ---------------------------------------------------------------------------
# Variant C: BiLSTMTransVAE  (BiLSTM + Transformer + VAE — best performer)
# ---------------------------------------------------------------------------

class BiLSTMTransVAE(nn.Module):
    """
    Variant C — BiLSTM Pre-Encoder + Transformer Encoder + VAE Bottleneck.

    BiLSTM captures local sequential dependencies (left→right + right→left)
    before the Transformer captures global long-range context. This two-stage
    feature extraction produced F1=0.889 in centralised evaluation — the
    highest of all three variants before threshold correction.

    Syllabus links:
      Unit 2 — Bidirectional RNN, LSTM Cell, Transformer, BERT
      Unit 3 — VAE, Reparameterisation Trick
    """
    def __init__(
        self,
        vocab_size:   int,
        d_model:      int   = 128,
        nhead:        int   = 4,
        num_layers:   int   = 2,
        latent_dim:   int   = 64,
        seq_len:      int   = 128,
        lstm_layers:  int   = 2,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.seq_len    = seq_len
        self.latent_dim = latent_dim

        self.cls_token       = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # BiLSTM pre-encoder: hidden_size = d_model//2 so bidirectional output = d_model
        self.bilstm = nn.LSTM(
            input_size    = d_model,
            hidden_size   = d_model // 2,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 1,
                                               dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_mu      = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, seq_len * d_model),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape

        tok_embed  = self.token_embedding(input_ids)       # (B, T, d_model)
        orig_embed = tok_embed.detach()

        # Stage 1 — BiLSTM: local context
        lstm_out, _ = self.bilstm(tok_embed)               # (B, T, d_model)
        lstm_out    = self.lstm_norm(lstm_out)

        # Stage 2 — Transformer: global context
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x          = torch.cat([cls_tokens, lstm_out], dim=1)
        x          = self.pos_encoding(x)

        pad_mask            = torch.zeros(B, T + 1, dtype=torch.bool,
                                         device=input_ids.device)
        pad_mask[:, 1:]     = (input_ids == 0)
        x                   = self.transformer(x, src_key_padding_mask=pad_mask)

        cls_out = x[:, 0, :]
        mu      = self.fc_mu(cls_out)
        log_var = self.fc_log_var(cls_out)
        z       = self.reparameterise(mu, log_var)

        recon_embed = self.decoder(z).reshape(B, T, self.d_model)
        return recon_embed, orig_embed, mu, log_var

    @torch.no_grad()
    def anomaly_score(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.eval()
        recon_embed, orig_embed, _, _ = self.forward(input_ids)
        return F.mse_loss(recon_embed, orig_embed, reduction="none").mean(dim=[1, 2])


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_model(
    variant:    str,
    vocab_size: int,
    d_model:    int = 128,
    nhead:      int = 4,
    num_layers: int = 2,
    latent_dim: int = 64,
    seq_len:    int = 128,
    dropout:    float = 0.1,
) -> nn.Module:
    """
    Construct the requested model variant.

    Args:
        variant : "A" | "B" | "C"
        ...     : architecture hyperparameters

    Returns:
        Instantiated nn.Module
    """
    kwargs = dict(vocab_size=vocab_size, latent_dim=latent_dim, seq_len=seq_len)
    if variant == "A":
        return VAEOnly(**kwargs, embed_dim=d_model)
    if variant == "B":
        return TransVAE(**kwargs, d_model=d_model, nhead=nhead,
                        num_layers=num_layers, dropout=dropout)
    if variant == "C":
        return BiLSTMTransVAE(**kwargs, d_model=d_model, nhead=nhead,
                              num_layers=num_layers, dropout=dropout)
    raise ValueError(f"Unknown variant '{variant}'. Choose from A, B, C.")