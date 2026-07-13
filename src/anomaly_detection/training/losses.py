"""Fonctions de perte pour l'entraînement.

Concept
-------
Une **fonction de perte** mesure à quel point une prédiction est mauvaise.
L'entraînement ajuste les poids du réseau pour la *minimiser*.

Pour un autoencodeur, on veut que la reconstruction ressemble à l'entrée : on
utilise l'**erreur quadratique moyenne** (MSE), qui pénalise le carré de la
différence pixel à pixel.
"""

from __future__ import annotations

import torch
from torch import nn


def reconstruction_loss() -> nn.Module:
    """Retourne la perte de reconstruction (MSE) à utiliser à l'entraînement.

    Returns:
        Un module :class:`torch.nn.MSELoss` (moyenne sur tous les éléments).

    Example:
        >>> criterion = reconstruction_loss()
        >>> import torch
        >>> loss = criterion(torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 8, 8))
        >>> float(loss)
        1.0
    """
    return nn.MSELoss()


def per_image_reconstruction_error(
    images: torch.Tensor, reconstructions: torch.Tensor
) -> torch.Tensor:
    """Erreur de reconstruction moyenne par image (utile pour le score).

    Args:
        images: Batch d'entrée ``(B, C, H, W)``.
        reconstructions: Reconstructions ``(B, C, H, W)``.

    Returns:
        Erreur par image de forme ``(B,)``.
    """
    squared_error = (images - reconstructions) ** 2
    return squared_error.flatten(start_dim=1).mean(dim=1)
