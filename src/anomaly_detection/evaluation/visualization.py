"""Visualisations d'évaluation et d'interprétabilité.

Ce module produit les images qui rendent les résultats *lisibles* : courbes
ROC / précision-rappel, matrice de confusion, et surtout les **cartes de
chaleur** (heatmaps) superposées à l'image pour montrer OÙ le modèle voit une
anomalie.

Avertissement d'interprétation : une heatmap indique où l'erreur de
reconstruction est forte ; ce n'est PAS une preuve causale de la cause du
défaut. On l'utilise comme aide visuelle, pas comme explication définitive.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sans écran (serveur / CI)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    precision_recall_curve,
    roc_curve,
)


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_roc_curve(labels: np.ndarray, scores: np.ndarray, path: str | Path) -> Path:
    """Trace la courbe ROC (taux de vrais positifs vs faux positifs).

    Args:
        labels: Étiquettes ``0``/``1``.
        scores: Scores d'anomalie.
        path: Chemin du PNG de sortie.

    Returns:
        Le chemin de l'image écrite.
    """
    out = _ensure_parent(path)
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="hasard")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_pr_curve(labels: np.ndarray, scores: np.ndarray, path: str | Path) -> Path:
    """Trace la courbe précision-rappel.

    Args:
        labels: Étiquettes ``0``/``1``.
        scores: Scores d'anomalie.
        path: Chemin du PNG de sortie.

    Returns:
        Le chemin de l'image écrite.
    """
    out = _ensure_parent(path)
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title("Courbe précision-rappel")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def plot_confusion_matrix(
    tn: int, fp: int, fn: int, tp: int, path: str | Path
) -> Path:
    """Affiche la matrice de confusion sous forme de tableau coloré.

    Args:
        tn: Vrais négatifs. fp: Faux positifs. fn: Faux négatifs. tp: Vrais positifs.
        path: Chemin du PNG de sortie.

    Returns:
        Le chemin de l'image écrite.
    """
    out = _ensure_parent(path)
    matrix = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Prévu normal", "Prévu anomalie"])
    plt.yticks([0, 1], ["Réel normal", "Réel anomalie"])
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return out


def overlay_heatmap(
    image: Image.Image | str | Path,
    anomaly_map: np.ndarray,
    path: str | Path,
    alpha: float = 0.5,
) -> Path:
    """Superpose une carte de chaleur à l'image d'origine.

    Args:
        image: Image (objet PIL ou chemin).
        anomaly_map: Carte 2D normalisée dans ``[0, 1]``.
        path: Chemin du PNG de sortie.
        alpha: Opacité de la heatmap (0 = image seule, 1 = heatmap seule).

    Returns:
        Le chemin de l'image écrite.
    """
    out = _ensure_parent(path)
    pil = image if isinstance(image, Image.Image) else Image.open(image)
    pil = pil.convert("RGB")

    plt.figure(figsize=(5, 5))
    plt.imshow(pil)
    plt.imshow(
        anomaly_map,
        cmap="jet",
        alpha=alpha,
        extent=(0, pil.size[0], pil.size[1], 0),
    )
    plt.axis("off")
    plt.title("Carte de chaleur (erreur de reconstruction)")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def save_examples_gallery(
    examples: list[dict], path: str | Path, title: str = "Exemples"
) -> Path:
    """Assemble une galerie d'images annotées (ex. faux positifs / négatifs).

    Args:
        examples: Liste de dicts ``{"image": PIL/chemin, "caption": str}``.
        path: Chemin du PNG de sortie.
        title: Titre de la figure.

    Returns:
        Le chemin de l'image écrite (une figure vide légendée si liste vide).
    """
    out = _ensure_parent(path)
    n = max(len(examples), 1)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    if not examples:
        axes[0].text(0.5, 0.5, "Aucun exemple", ha="center", va="center")
    for ax, ex in zip(axes, examples):
        img = ex["image"]
        pil = img if isinstance(img, Image.Image) else Image.open(img)
        ax.imshow(pil.convert("RGB"))
        ax.set_title(ex.get("caption", ""), fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
