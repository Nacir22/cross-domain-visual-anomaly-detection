"""Tests de transférabilité : splits par groupe (patient / zone), anti-fuite."""

from __future__ import annotations

from pathlib import Path

from anomaly_detection.data.domains import build_aerial_splits, build_medical_splits
from anomaly_detection.data.splits import (
    LABEL_ANOMALY,
    LABEL_NORMAL,
    create_grouped_splits,
)
from anomaly_detection.data.synthetic import (
    generate_synthetic_aerial,
    generate_synthetic_medical,
)


def _groups(entries):
    return {e["group"] for e in entries}


def test_medical_split_has_no_patient_leakage(tmp_path: Path):
    generate_synthetic_medical(tmp_path, seed=0)
    splits = build_medical_splits(tmp_path, seed=0)
    train_g, val_g, test_g = (_groups(splits[s]) for s in ("train", "val", "test"))
    # Aucun patient ne franchit deux splits.
    assert train_g.isdisjoint(val_g)
    assert train_g.isdisjoint(test_g)
    assert val_g.isdisjoint(test_g)


def test_medical_train_val_are_normal_only(tmp_path: Path):
    generate_synthetic_medical(tmp_path, seed=0)
    splits = build_medical_splits(tmp_path, seed=0)
    assert all(e["label"] == LABEL_NORMAL for e in splits["train"])
    assert all(e["label"] == LABEL_NORMAL for e in splits["val"])
    assert any(e["label"] == LABEL_ANOMALY for e in splits["test"])


def test_aerial_split_has_no_zone_leakage(tmp_path: Path):
    generate_synthetic_aerial(tmp_path, seed=0)
    splits = build_aerial_splits(tmp_path, seed=0)
    train_g, val_g, test_g = (_groups(splits[s]) for s in ("train", "val", "test"))
    assert train_g.isdisjoint(val_g | test_g)
    assert val_g.isdisjoint(test_g)


def test_aerial_anomalies_have_masks(tmp_path: Path):
    generate_synthetic_aerial(tmp_path, seed=0)
    splits = build_aerial_splits(tmp_path, seed=0)
    anomalies = [e for e in splits["test"] if e["label"] == LABEL_ANOMALY]
    assert anomalies and all(e["mask"] for e in anomalies)


def test_grouped_split_sends_abnormal_group_to_test():
    # Un groupe contenant une anomalie doit aller ENTIÈREMENT au test.
    records = [
        {"image": "a0.png", "label": 0, "group": "gA"},
        {"image": "a1.png", "label": 1, "group": "gA"},  # rend gA anormal
        {"image": "b0.png", "label": 0, "group": "gB"},
        {"image": "c0.png", "label": 0, "group": "gC"},
    ]
    splits = create_grouped_splits(
        records, train_fraction=0.5, val_fraction=0.25, seed=0
    )
    test_groups = {e["group"] for e in splits["test"]}
    assert "gA" in test_groups
    train_val_groups = {e["group"] for e in splits["train"] + splits["val"]}
    assert "gA" not in train_val_groups


def test_grouped_split_is_reproducible(tmp_path: Path):
    generate_synthetic_medical(tmp_path, seed=0)
    a = build_medical_splits(tmp_path, seed=7)
    b = build_medical_splits(tmp_path, seed=7)
    assert a == b
