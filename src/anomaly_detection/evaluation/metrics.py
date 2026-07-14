"""Métriques d'évaluation (niveau image et niveau pixel).

Rappel des concepts
--------------------
On décide « normal / anormal » en comparant un **score** à un **seuil**.

- **Faux positif (FP)** : normal classé anormal (fausse alerte).
- **Faux négatif (FN)** : anomalie manquée (le plus grave en industrie/santé).
- **Précision** : parmi les alarmes, combien sont justifiées.
- **Rappel** : parmi les vraies anomalies, combien sont détectées.
- **F1** : compromis (moyenne harmonique) entre précision et rappel.
- **AUROC** : capacité à classer une anomalie au-dessus d'un normal, *toutes
  valeurs de seuil confondues* (0.5 = hasard, 1.0 = parfait). Peut être trop
  optimiste quand les anomalies sont rares.
- **AUPRC** : aire sous la courbe précision-rappel, plus informative que
  l'AUROC en cas de fort déséquilibre (peu d'anomalies).

Au **niveau pixel** (uniquement si un masque de vérité terrain existe) :

- **Dice** et **IoU** mesurent le recouvrement entre la zone prédite et la
  vraie zone d'anomalie.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUROC en tolérant le cas dégénéré (une seule classe présente)."""
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUPRC (average precision) en tolérant le cas dégénéré."""
    if len(np.unique(labels)) < 2:
        return None
    return float(average_precision_score(labels, scores))


def image_level_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | None]:
    """Calcule les métriques niveau image.

    Args:
        labels: Vérité terrain par image, ``0`` (normal) ou ``1`` (anomalie).
        scores: Score d'anomalie par image (plus élevé = plus anormal).
        threshold: Seuil de décision (``score >= threshold`` => anomalie).

    Returns:
        Dictionnaire : ``auroc``, ``auprc``, ``precision``, ``recall``, ``f1``,
        ``tp``, ``fp``, ``tn``, ``fn``, ``fpr``, ``fnr``, ``threshold``.
        ``auroc``/``auprc`` valent ``None`` si une seule classe est présente.

    Example:
        >>> import numpy as np
        >>> m = image_level_metrics(
        ...     np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5
        ... )
        >>> m["auroc"], m["f1"]
        (1.0, 1.0)
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "auroc": _safe_auc(labels, scores),
        "auprc": _safe_ap(labels, scores),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "threshold": float(threshold),
    }


def dice_score(
    pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-8
) -> float:
    """Coefficient de Dice entre deux masques binaires.

    Dice = ``2 * |A∩B| / (|A| + |B|)``. Vaut 1 si recouvrement parfait.

    Args:
        pred_mask: Masque prédit binaire (0/1).
        true_mask: Masque de vérité terrain binaire (0/1).
        eps: Constante évitant la division par zéro (deux masques vides -> 1).

    Returns:
        Le score de Dice dans ``[0, 1]``.
    """
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    return float((2 * intersection + eps) / (pred.sum() + true.sum() + eps))


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-8) -> float:
    """Intersection-over-Union (Jaccard) entre deux masques binaires.

    IoU = ``|A∩B| / |A∪B|``. Vaut 1 si recouvrement parfait.

    Args:
        pred_mask: Masque prédit binaire (0/1).
        true_mask: Masque de vérité terrain binaire (0/1).
        eps: Constante évitant la division par zéro.

    Returns:
        Le score IoU dans ``[0, 1]``.
    """
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float((intersection + eps) / (union + eps))


def pixel_level_metrics(
    masks: np.ndarray, maps: np.ndarray, threshold: float
) -> dict[str, float | None]:
    """Calcule les métriques niveau pixel (nécessite des masques fiables).

    Args:
        masks: Masques de vérité terrain empilés, forme ``(N, H, W)``, valeurs
            ``{0, 1}``.
        maps: Cartes d'anomalie continues, forme ``(N, H, W)``.
        threshold: Seuil binarisant les cartes pour Dice/IoU.

    Returns:
        Dictionnaire : ``pixel_auroc``, ``pixel_auprc``, ``dice``, ``iou``.

    Raises:
        ValueError: Si ``masks`` et ``maps`` n'ont pas la même forme.
    """
    masks = np.asarray(masks)
    maps = np.asarray(maps, dtype=float)
    if masks.shape != maps.shape:
        raise ValueError(
            f"Formes incompatibles : masks={masks.shape}, maps={maps.shape}."
        )

    flat_labels = (masks.reshape(-1) > 0).astype(int)
    flat_scores = maps.reshape(-1)
    pred_mask = (maps >= threshold).astype(int)

    return {
        "pixel_auroc": _safe_auc(flat_labels, flat_scores),
        "pixel_auprc": _safe_ap(flat_labels, flat_scores),
        "dice": dice_score(pred_mask, (masks > 0).astype(int)),
        "iou": iou_score(pred_mask, (masks > 0).astype(int)),
    }
