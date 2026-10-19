import torch

from src.models.bilstm_trans_vae import BiLSTMTransVAE, build_model


def test_build_model_factory_works():
    model = build_model(
        variant="C",
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_layers=1,
        latent_dim=8,
        seq_len=16,
        dropout=0.0,
    )
    assert isinstance(model, BiLSTMTransVAE)


def test_bilstm_trans_vae_forward_pass():
    model = BiLSTMTransVAE(
        vocab_size=100,
        d_model=32,
        nhead=4,
        num_layers=1,
        latent_dim=8,
        seq_len=16,
        lstm_layers=1,
        dropout=0.0,
    )
    model.eval()

    batch = torch.randint(1, 100, (2, 16))
    recon, orig, mu, log_var = model(batch)

    assert recon.shape == (2, 16, 32)
    assert orig.shape == (2, 16, 32)
    assert mu.shape == (2, 8)
    assert log_var.shape == (2, 8)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(log_var).all()


def test_bilstm_trans_vae_anomaly_score_shape():
    model = BiLSTMTransVAE(
        vocab_size=100,
        d_model=24,
        nhead=4,
        num_layers=1,
        latent_dim=6,
        seq_len=12,
        lstm_layers=1,
        dropout=0.0,
    )
    model.eval()

    batch = torch.randint(1, 100, (3, 12))
    scores = model.anomaly_score(batch)

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
