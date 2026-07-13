"""Tests unitaires de l'autoencodeur convolutif."""

from __future__ import annotations

import torch

from anomaly_detection.models.autoencoder import ConvAutoencoder, build_autoencoder

SIZE = 32  # divisible par 16 (profondeur de 4 downsamplings)


def test_reconstruction_preserves_shape():
    model = ConvAutoencoder(base_channels=8, latent_dim=16)
    x = torch.randn(2, 3, SIZE, SIZE)
    out = model(x)
    assert out.shape == x.shape


def test_anomaly_map_shape():
    model = ConvAutoencoder(base_channels=8, latent_dim=16)
    x = torch.randn(2, 3, SIZE, SIZE)
    amap = model.anomaly_map(x)
    assert amap.shape == (2, 1, SIZE, SIZE)
    assert torch.all(amap >= 0)


def test_anomaly_score_shape_and_finite():
    model = ConvAutoencoder(base_channels=8, latent_dim=16)
    x = torch.randn(4, 3, SIZE, SIZE)
    scores = model.anomaly_score(x)
    assert scores.shape == (4,)
    assert torch.all(torch.isfinite(scores))


def test_build_autoencoder_from_config():
    model = build_autoencoder({"base_channels": 8, "latent_dim": 16})
    assert isinstance(model, ConvAutoencoder)
