"""Tests des stratégies de seuillage (et de l'absence de fuite de test)."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.evaluation.thresholds import (
    threshold_cost_based,
    threshold_from_normal_percentile,
    threshold_max_f1,
)


def test_percentile_matches_numpy():
    scores = np.arange(0, 101, dtype=float)
    assert threshold_from_normal_percentile(scores, 90) == 90.0


def test_percentile_uses_only_normals():
    # La fonction ne prend QUE des scores normaux : impossible d'y glisser du test.
    normal_scores = np.array([0.0, 0.1, 0.2, 0.3])
    thr = threshold_from_normal_percentile(normal_scores, 100)
    assert thr == 0.3


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        threshold_from_normal_percentile(np.array([]), 95)


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        threshold_from_normal_percentile(np.array([1.0]), 150)


def test_max_f1_separates_classes():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    thr = threshold_max_f1(labels, scores)
    preds = (scores >= thr).astype(int)
    assert list(preds) == [0, 0, 1, 1]


def test_max_f1_requires_both_classes():
    with pytest.raises(ValueError):
        threshold_max_f1(np.array([0, 0]), np.array([0.1, 0.2]))


def test_cost_based_penalizes_false_negatives():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    # Avec un coût FN très élevé, le seuil doit rester assez bas pour capter
    # l'anomalie de score 0.3 (aucun FN toléré).
    thr = threshold_cost_based(labels, scores, fp_cost=1.0, fn_cost=100.0)
    preds = (scores >= thr).astype(int)
    assert preds[2] == 1  # l'anomalie faible est bien détectée
