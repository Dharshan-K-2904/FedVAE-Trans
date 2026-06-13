import torch
import torch.nn as nn

class LSTMBaselineAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, seq_len=128):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.encoder_lstm(embedded)
        repeated_hidden = hidden[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder_lstm(repeated_hidden)
        reconstruction = self.output_layer(decoded)
        return reconstruction, embedded
