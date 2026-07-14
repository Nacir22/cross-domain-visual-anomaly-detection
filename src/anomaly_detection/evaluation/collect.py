"""Collecte des sorties d'un modèle sur un jeu de données.

Sépare le calcul (faire tourner le modèle) de la mesure (métriques). On récupère
d'un coup, pour tout un split, les scores, cartes, étiquettes, masques et
chemins — que l'évaluation exploite ensuite sans relancer le modèle.

On utilise les cartes/scores BRUTS (non normalisés par image), car les
métriques comme l'AUROC pixel comparent les valeurs entre elles à l'échelle du
dataset entier.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_outputs(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray | list[str]]:
    """Exécute le modèle sur tout un loader et rassemble les sorties.

    Args:
        model: Modèle implémentant ``anomaly_score`` et ``anomaly_map``.
        loader: DataLoader d'un split (train, val ou test).
        device: CPU ou GPU.

    Returns:
        Dictionnaire :

        - ``scores`` : ``(N,)`` scores d'anomalie par image ;
        - ``labels`` : ``(N,)`` étiquettes 0/1 ;
        - ``maps``   : ``(N, H, W)`` cartes d'anomalie brutes ;
        - ``masks``  : ``(N, H, W)`` masques de vérité terrain (0/1) ;
        - ``paths``  : liste des chemins d'images.
    """
    model.eval()
    scores, labels, maps, masks, paths = [], [], [], [], []

    for batch in loader:
        images = batch["image"].to(device)
        batch_scores = model.anomaly_score(images).cpu().numpy()
        batch_maps = model.anomaly_map(images).squeeze(1).cpu().numpy()  # (B,H,W)

        scores.append(batch_scores)
        maps.append(batch_maps)
        labels.append(np.asarray(batch["label"]).astype(int).reshape(-1))
        masks.append(batch["mask"].squeeze(1).cpu().numpy())  # (B,H,W)
        paths.extend(list(batch["path"]))

    return {
        "scores": np.concatenate(scores) if scores else np.array([]),
        "labels": np.concatenate(labels) if labels else np.array([]),
        "maps": np.concatenate(maps) if maps else np.empty((0, 0, 0)),
        "masks": np.concatenate(masks) if masks else np.empty((0, 0, 0)),
        "paths": paths,
    }
