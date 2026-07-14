"""Tests smoke de PatchCore (backbone NON pré-entraîné pour rester hors ligne)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from anomaly_detection.data.datasets import MVTecDataset
from anomaly_detection.data.splits import create_splits
from anomaly_detection.inference.pipeline import load_model_from_checkpoint
from anomaly_detection.models.patchcore import PatchCore, build_patchcore

SIZE = 32
# pretrained=False évite tout téléchargement réseau (CI hors ligne).
CONFIG = {
    "backbone": "resnet18",
    "pretrained": False,
    "coreset_ratio": 0.5,
    "image_size": SIZE,
}


def _loader(root: Path):
    splits = create_splits(root, "synthetic", seed=0)
    ds = MVTecDataset(root, "synthetic", "test", SIZE, splits=splits)
    return DataLoader(ds, batch_size=2), splits


@pytest.mark.smoke
def test_anomaly_map_before_fit_raises():
    model = build_patchcore(CONFIG)
    with pytest.raises(RuntimeError):
        model.anomaly_map(torch.randn(1, 3, SIZE, SIZE))


@pytest.mark.smoke
def test_fit_then_predict_shapes(synthetic_mvtec: Path):
    loader, _ = _loader(synthetic_mvtec)
    model = build_patchcore(CONFIG)
    model.fit(loader, device=torch.device("cpu"))

    assert model.memory_bank is not None
    assert model.memory_bank.shape[0] >= 1

    x = torch.randn(2, 3, SIZE, SIZE)
    amap = model.anomaly_map(x)
    scores = model.anomaly_score(x)
    assert amap.shape == (2, 1, SIZE, SIZE)
    assert scores.shape == (2,)
    assert torch.all(torch.isfinite(scores))


@pytest.mark.smoke
def test_checkpoint_save_and_reload(synthetic_mvtec: Path, tmp_path: Path):
    loader, _ = _loader(synthetic_mvtec)
    model = build_patchcore(CONFIG)
    model.fit(loader, device=torch.device("cpu"))

    ckpt = tmp_path / "patchcore.pt"
    torch.save(
        {
            "model": "patchcore",
            "model_config": CONFIG,
            "memory_bank": model.memory_bank,
            "image_size": SIZE,
        },
        ckpt,
    )

    reloaded, meta = load_model_from_checkpoint(ckpt, device=torch.device("cpu"))
    assert isinstance(reloaded, PatchCore)
    assert reloaded.memory_bank is not None
    scores = reloaded.anomaly_score(torch.randn(1, 3, SIZE, SIZE))
    assert scores.shape == (1,)
