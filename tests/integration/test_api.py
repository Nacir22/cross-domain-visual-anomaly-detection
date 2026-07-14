"""Tests d'intégration de l'API FastAPI (nécessitent torch : marqués integration)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def client_and_model(synthetic_mvtec: Path, tmp_path: Path, monkeypatch):
    """Entraîne un mini-modèle, configure l'API dessus et renvoie un TestClient."""
    import torch
    from fastapi.testclient import TestClient
    from torch.utils.data import DataLoader

    from anomaly_detection.data.datasets import MVTecDataset
    from anomaly_detection.data.splits import create_splits
    from anomaly_detection.models.autoencoder import build_autoencoder
    from anomaly_detection.training.engine import train_autoencoder

    size = 32
    splits = create_splits(synthetic_mvtec, "synthetic", seed=0)
    train_ds = MVTecDataset(synthetic_mvtec, "synthetic", "train", size, splits=splits)
    val_ds = MVTecDataset(synthetic_mvtec, "synthetic", "val", size, splits=splits)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ckpt = models_dir / "industrial_autoencoder_synthetic.pt"
    train_autoencoder(
        build_autoencoder({"base_channels": 8, "latent_dim": 16}),
        DataLoader(train_ds, batch_size=2, drop_last=True),
        DataLoader(val_ds, batch_size=2),
        epochs=1,
        learning_rate=1e-3,
        device=torch.device("cpu"),
        checkpoint_path=ckpt,
        seed=0,
        extra={
            "model": "autoencoder",
            "model_config": {"base_channels": 8, "latent_dim": 16},
            "image_size": size,
            "domain": "industrial",
        },
    )

    from app.api import dependencies
    from app.api.dependencies import Settings, get_settings
    from app.api.main import app

    # Vide les caches de pipelines et pointe l'API sur notre dossier de modèles.
    dependencies._PIPELINE_CACHE.clear()
    dependencies._META_CACHE.clear()
    app.dependency_overrides[get_settings] = lambda: Settings(
        models_dir=models_dir, default_threshold=0.0
    )
    yield TestClient(app), "industrial_autoencoder_synthetic"
    app.dependency_overrides.clear()


def _png_bytes(size: int = 32) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=(120, 120, 120)).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client_and_model):
    client, _ = client_and_model
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_models_lists_checkpoint(client_and_model):
    client, key = client_and_model
    resp = client.get("/models")
    assert resp.status_code == 200
    keys = [m["key"] for m in resp.json()["models"]]
    assert key in keys


def test_predict_valid_image(client_and_model):
    client, key = client_and_model
    files = {"file": ("x.png", _png_bytes(), "image/png")}
    resp = client.post("/predict", files=files, params={"model": key})
    assert resp.status_code == 200
    body = resp.json()
    assert "anomaly_score" in body
    assert body["metadata"]["image_width"] == 32
    assert isinstance(body["heatmap"], str) and len(body["heatmap"]) > 0


def test_predict_rejects_wrong_mime(client_and_model):
    client, key = client_and_model
    files = {"file": ("note.txt", b"hello", "text/plain")}
    resp = client.post("/predict", files=files, params={"model": key})
    assert resp.status_code == 415


def test_predict_rejects_corrupted_image(client_and_model):
    client, key = client_and_model
    files = {"file": ("bad.png", b"not a real image", "image/png")}
    resp = client.post("/predict", files=files, params={"model": key})
    assert resp.status_code == 400
