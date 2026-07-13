"""Tests des splits train / val / test (reproductibilité et anti-fuite)."""

from __future__ import annotations

from pathlib import Path

from anomaly_detection.data.splits import (
    LABEL_ANOMALY,
    LABEL_NORMAL,
    create_splits,
    load_splits,
    save_splits,
)


def test_train_and_val_contain_only_normals(synthetic_mvtec: Path):
    splits = create_splits(synthetic_mvtec, "synthetic", val_fraction=0.25, seed=0)
    assert all(e["label"] == LABEL_NORMAL for e in splits["train"])
    assert all(e["label"] == LABEL_NORMAL for e in splits["val"])
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0


def test_test_split_contains_anomalies(synthetic_mvtec: Path):
    splits = create_splits(synthetic_mvtec, "synthetic", seed=0)
    labels = {e["label"] for e in splits["test"]}
    assert LABEL_ANOMALY in labels
    assert LABEL_NORMAL in labels


def test_no_leakage_between_train_and_val(synthetic_mvtec: Path):
    splits = create_splits(synthetic_mvtec, "synthetic", val_fraction=0.25, seed=0)
    train_imgs = {e["image"] for e in splits["train"]}
    val_imgs = {e["image"] for e in splits["val"]}
    assert train_imgs.isdisjoint(val_imgs)


def test_splits_are_reproducible(synthetic_mvtec: Path):
    a = create_splits(synthetic_mvtec, "synthetic", seed=123)
    b = create_splits(synthetic_mvtec, "synthetic", seed=123)
    assert a == b


def test_anomalies_have_masks(synthetic_mvtec: Path):
    splits = create_splits(synthetic_mvtec, "synthetic", seed=0)
    anomalies = [e for e in splits["test"] if e["label"] == LABEL_ANOMALY]
    assert anomalies and all(e["mask"] for e in anomalies)


def test_save_and_load_roundtrip(synthetic_mvtec: Path, tmp_path: Path):
    splits = create_splits(synthetic_mvtec, "synthetic", seed=0)
    out = save_splits(splits, tmp_path / "splits.json")
    assert out.is_file()
    assert load_splits(out) == splits
