"""Test smoke : mini-entraînement de bout en bout de l'autoencodeur.

But : vérifier que le pipeline complet TOURNE (données -> entraînement ->
checkpoint -> rechargement -> inférence), sans exiger de performance. On
utilise le mini-dataset synthétique et des images 32x32 pour rester rapide.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from anomaly_detection.data.datasets import MVTecDataset
from anomaly_detection.data.splits import create_splits
from anomaly_detection.inference.pipeline import (
    AnomalyInferencePipeline,
    load_model_from_checkpoint,
)
from anomaly_detection.models.autoencoder import build_autoencoder
from anomaly_detection.training.engine import train_autoencoder

SIZE = 32
MODEL_CONFIG = {"base_channels": 8, "latent_dim": 16}


@pytest.mark.smoke
def test_train_save_reload_and_infer(synthetic_mvtec: Path, tmp_path: Path):
    splits = create_splits(synthetic_mvtec, "synthetic", val_fraction=0.25, seed=0)
    train_ds = MVTecDataset(synthetic_mvtec, "synthetic", "train", SIZE, splits=splits)
    val_ds = MVTecDataset(synthetic_mvtec, "synthetic", "val", SIZE, splits=splits)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)

    model = build_autoencoder(MODEL_CONFIG)
    ckpt = tmp_path / "ae.pt"

    history = train_autoencoder(
        model,
        train_loader,
        val_loader,
        epochs=1,
        learning_rate=1e-3,
        device=torch.device("cpu"),
        checkpoint_path=ckpt,
        seed=0,
        extra={"model": "autoencoder", "model_config": MODEL_CONFIG, "image_size": SIZE},
    )

    # 1) L'entraînement produit une perte finie.
    assert len(history.train) == 1
    assert torch.isfinite(torch.tensor(history.train[0]))

    # 2) Le checkpoint existe et se recharge.
    assert ckpt.is_file()
    reloaded, meta = load_model_from_checkpoint(ckpt, device=torch.device("cpu"))
    assert meta["image_size"] == SIZE

    # 3) L'inférence produit un score exploitable et une carte normalisée.
    pipeline = AnomalyInferencePipeline.from_checkpoint(ckpt)
    sample_image = train_ds.root / str(splits["test"][0]["image"])
    result = pipeline.predict(sample_image)

    assert isinstance(result["anomaly_score"], float)
    assert result["anomaly_map"].shape == (SIZE, SIZE)
    assert 0.0 <= float(result["anomaly_map"].min())
    assert float(result["anomaly_map"].max()) <= 1.0
