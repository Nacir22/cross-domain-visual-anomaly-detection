"""Tests du Dataset PyTorch MVTec (chargement, dimensions, labels)."""

from __future__ import annotations

from pathlib import Path

import pytest

from anomaly_detection.data.datasets import MVTecDataset

IMAGE_SIZE = 32


def test_train_split_is_all_normal(synthetic_mvtec: Path):
    ds = MVTecDataset(synthetic_mvtec, "synthetic", split="train", image_size=IMAGE_SIZE)
    assert len(ds) > 0
    assert all(ds[i]["label"] == 0 for i in range(len(ds)))


def test_item_has_expected_shapes(synthetic_mvtec: Path):
    ds = MVTecDataset(synthetic_mvtec, "synthetic", split="test", image_size=IMAGE_SIZE)
    item = ds[0]
    assert item["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert item["mask"].shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert item["label"] in (0, 1)
    assert isinstance(item["path"], str)


def test_loading_is_reproducible(synthetic_mvtec: Path):
    ds = MVTecDataset(synthetic_mvtec, "synthetic", split="test", image_size=IMAGE_SIZE)
    first = ds[0]["image"]
    second = ds[0]["image"]
    assert first.equal(second)


def test_invalid_split_raises(synthetic_mvtec: Path):
    with pytest.raises(ValueError):
        MVTecDataset(synthetic_mvtec, "synthetic", split="predict")


def test_missing_category_raises(synthetic_mvtec: Path):
    with pytest.raises(FileNotFoundError):
        MVTecDataset(synthetic_mvtec, "does_not_exist", split="train")
