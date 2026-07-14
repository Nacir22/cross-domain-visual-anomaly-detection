"""Sélection du seuil de décision (sans jamais utiliser le jeu de test).

Pourquoi c'est délicat
----------------------
Le modèle produit un score continu ; il faut un **seuil** pour trancher
normal / anormal. Choisir ce seuil sur le jeu de **test** serait tricher : on
optimiserait sur les données censées mesurer la performance finale. Le seuil
doit donc venir de la **validation** (ou de statistiques des normales).

Trois stratégies (comparées dans le rapport) :

1. **Percentile des normales** : on prend une valeur au-dessus de laquelle
   seule une petite fraction des normales tombe (ex. 95e percentile). Ne
   nécessite QUE des images normales — parfait pour notre validation one-class.
2. **F1 maximal** : on cherche le seuil qui maximise le F1 sur un jeu *étiqueté*
   (contenant normales ET anomalies). À n'utiliser que si un tel jeu de
   calibration existe, distinct du test.
3. **Coût métier** : on pondère différemment un faux positif et un faux négatif
   (en santé/industrie, manquer une anomalie coûte souvent bien plus cher).
"""

from __future__ import annotations

import numpy as np


def threshold_from_normal_percentile(
    normal_scores: np.ndarray, percentile: float = 95.0
) -> float:
    """Seuil = percentile des scores d'images NORMALES.

    Args:
        normal_scores: Scores d'anomalie d'images normales (ex. validation).
        percentile: Percentile dans ``]0, 100]`` (95 => on tolère 5 % de fausses
            alertes sur les normales de validation).

    Returns:
        La valeur de seuil.

    Raises:
        ValueError: Si ``normal_scores`` est vide ou percentile hors bornes.

    Example:
        >>> import numpy as np
        >>> threshold_from_normal_percentile(np.arange(0, 101), 90)
        90.0
    """
    scores = np.asarray(normal_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("normal_scores est vide.")
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"percentile doit être dans ]0, 100], reçu {percentile}.")
    return float(np.percentile(scores, percentile))


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Génère des seuils candidats (valeurs uniques + une borne supérieure)."""
    unique = np.unique(scores)
    high = unique[-1] + (1.0 if unique.size else 1.0)
    return np.concatenate([unique, [high]])


def threshold_max_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    """Seuil maximisant le F1 sur un jeu étiqueté (calibration).

    Args:
        labels: Étiquettes ``0``/``1`` du jeu de calibration.
        scores: Scores d'anomalie correspondants.

    Returns:
        Le seuil qui maximise le F1. En cas d'égalité, le plus petit seuil.

    Raises:
        ValueError: Si le jeu ne contient pas les deux classes.

    Example:
        >>> import numpy as np
        >>> threshold_max_f1(
        ...     np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
        ... )  # doctest: +ELLIPSIS
        0.8
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "threshold_max_f1 nécessite les deux classes (normal ET anomalie)."
        )

    best_f1, best_threshold = -1.0, float(scores.min())
    for t in _candidate_thresholds(scores):
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)
    return best_threshold


def threshold_cost_based(
    labels: np.ndarray,
    scores: np.ndarray,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> float:
    """Seuil minimisant un coût pondéré des erreurs (FP vs FN).

    En santé/industrie, un faux négatif (anomalie manquée) est souvent bien plus
    coûteux qu'un faux positif ; on le pénalise davantage (``fn_cost``).

    Args:
        labels: Étiquettes ``0``/``1`` du jeu de calibration.
        scores: Scores d'anomalie correspondants.
        fp_cost: Coût d'un faux positif.
        fn_cost: Coût d'un faux négatif (par défaut 5x un FP).

    Returns:
        Le seuil minimisant le coût total. En cas d'égalité, le plus grand seuil
        (plus conservateur : moins de fausses alertes).

    Raises:
        ValueError: Si le jeu ne contient pas les deux classes.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        raise ValueError("threshold_cost_based nécessite les deux classes.")

    best_cost, best_threshold = float("inf"), float(scores.min())
    for t in _candidate_thresholds(scores):
        preds = (scores >= t).astype(int)
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        cost = fp_cost * fp + fn_cost * fn
        if cost <= best_cost:
            best_cost, best_threshold = cost, float(t)
    return best_threshold
