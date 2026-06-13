import torch
import torch.nn as nn
from src.models.bilstm_trans_vae import PositionalEncoding

class VariantA_VAE(nn.Module):
    def __init__(self, vocab_size, d_model=128, latent_dim=64, seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(seq_len * d_model, latent_dim)
        self.fc_log_var = nn.Linear(seq_len * d_model, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, seq_len * d_model)
        )
        self.d_model = d_model

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        B, T = x.shape
        embedded = self.embedding(x)
        flat = self.flatten(embedded)
        mu = self.fc_mu(flat)
        log_var = self.fc_log_var(flat)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z).view(B, T, self.d_model)
        return reconstruction, embedded, mu, log_var

class VariantB_TransVAE(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, latent_dim=64, seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, seq_len * d_model)
        )

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        B, T = x.shape
        embedded = self.embedding(x)
        x_pe = self.pos_encoding(embedded)
        trans_out = self.transformer(x_pe)
        cls_rep = trans_out[:, 0, :]
        mu = self.fc_mu(cls_rep)
        log_var = self.fc_log_var(cls_rep)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z).view(B, T, self.d_model)
        return reconstruction, embedded, mu, log_var
