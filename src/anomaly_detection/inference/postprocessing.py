"""Post-traitement des cartes d'anomalie (pour affichage et heatmap).

Une carte d'anomalie brute contient des valeurs arbitraires (des erreurs de
reconstruction). Pour l'afficher ou la superposer à l'image, on la ramène dans
l'intervalle ``[0, 1]`` (normalisation min-max) et on peut la redimensionner à
la taille de l'image d'origine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


def min_max_normalize(anomaly_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Ramène une carte dans ``[0, 1]`` par normalisation min-max.

    Args:
        anomaly_map: Carte 2D de valeurs quelconques.
        eps: Petite constante évitant la division par zéro (carte constante).

    Returns:
        Carte normalisée dans ``[0, 1]``, même forme.

    Example:
        >>> import numpy as np
        >>> out = min_max_normalize(np.array([[0.0, 2.0], [4.0, 6.0]]))
        >>> float(out.min()), float(out.max())
        (0.0, 1.0)
    """
    lo = float(anomaly_map.min())
    hi = float(anomaly_map.max())
    return (anomaly_map - lo) / (hi - lo + eps)


def resize_anomaly_map(
    anomaly_map: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Redimensionne une carte ``(B, 1, h, w)`` vers ``(B, 1, height, width)``.

    Un lissage bilinéaire est utilisé car une carte d'anomalie est continue
    (contrairement à un masque binaire).

    Args:
        anomaly_map: Carte de forme ``(B, 1, h, w)``.
        height: Hauteur cible.
        width: Largeur cible.

    Returns:
        Carte redimensionnée ``(B, 1, height, width)``.
    """
    import torch.nn.functional as F

    return F.interpolate(
        anomaly_map, size=(height, width), mode="bilinear", align_corners=False
    )


def map_to_numpy(anomaly_map: torch.Tensor) -> np.ndarray:
    """Convertit une carte d'anomalie ``(1, 1, H, W)`` ou ``(1, H, W)`` en 2D NumPy.

    Args:
        anomaly_map: Tenseur de carte d'anomalie (batch de taille 1).

    Returns:
        Tableau NumPy 2D ``(H, W)``.
    """
    return anomaly_map.detach().cpu().squeeze().numpy()


def heatmap_to_base64_png(anomaly_map: np.ndarray, cmap: str = "jet") -> str:
    """Encode une carte d'anomalie ``[0, 1]`` en image PNG colorée (base64).

    Pratique pour transmettre la heatmap dans une réponse JSON d'API.

    Args:
        anomaly_map: Carte 2D de valeurs dans ``[0, 1]``.
        cmap: Nom de la colormap matplotlib.

    Returns:
        La chaîne base64 de l'image PNG (colormap appliquée).
    """
    import base64
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    arr = np.clip(np.asarray(anomaly_map, dtype=float), 0.0, 1.0)
    colored = plt.get_cmap(cmap)(arr)  # (H, W, 4) float
    rgb = (colored[..., :3] * 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
