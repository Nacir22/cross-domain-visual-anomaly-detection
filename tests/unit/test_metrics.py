"""Tests des métriques d'évaluation sur des cas simples et contrôlés."""

from __future__ import annotations

import numpy as np

from anomaly_detection.evaluation.metrics import (
    dice_score,
    image_level_metrics,
    iou_score,
    pixel_level_metrics,
)


def test_perfect_separation_gives_auroc_one():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = image_level_metrics(labels, scores, threshold=0.5)
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0
    assert m["f1"] == 1.0
    assert (m["fp"], m["fn"]) == (0, 0)


def test_confusion_counts_are_correct():
    labels = np.array([0, 0, 1, 1])
    # Seuil 0.5 : preds = [0, 1, 0, 1] -> tn=1, fp=1, fn=1, tp=1
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    m = image_level_metrics(labels, scores, threshold=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["fpr"] == 0.5
    assert m["fnr"] == 0.5


def test_single_class_returns_none_auroc():
    labels = np.array([0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3])
    m = image_level_metrics(labels, scores, threshold=0.5)
    assert m["auroc"] is None
    assert m["auprc"] is None


def test_dice_and_iou_extremes():
    mask = np.array([[1, 1], [0, 0]])
    assert dice_score(mask, mask) > 0.99
    assert iou_score(mask, mask) > 0.99
    disjoint = np.array([[0, 0], [1, 1]])
    assert dice_score(mask, disjoint) < 0.01
    assert iou_score(mask, disjoint) < 0.01


def test_pixel_metrics_perfect_map():
    masks = np.array([[[0, 0], [1, 1]]])
    maps = np.array([[[0.0, 0.1], [0.9, 0.8]]])
    res = pixel_level_metrics(masks, maps, threshold=0.5)
    assert res["pixel_auroc"] == 1.0
    assert res["dice"] > 0.99
