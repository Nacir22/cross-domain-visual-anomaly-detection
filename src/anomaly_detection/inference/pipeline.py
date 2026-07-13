"""Pipeline d'inférence : d'une image à un score et une carte d'anomalie.

Ce module fait le lien entre un checkpoint entraîné et une prédiction concrète :
il charge le modèle, applique le même preprocessing qu'à l'entraînement, puis
retourne un score d'anomalie et une carte redimensionnée à la taille d'origine.
Il sera réutilisé tel quel par l'API (Phase 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from anomaly_detection.data.preprocessing import load_image
from anomaly_detection.data.transforms import build_transforms
from anomaly_detection.inference.postprocessing import (
    map_to_numpy,
    min_max_normalize,
    resize_anomaly_map,
)
from anomaly_detection.models.autoencoder import build_autoencoder
from anomaly_detection.utils.device import resolve_device


def load_model_from_checkpoint(
    checkpoint_path: str | Path, device: torch.device | None = None
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Recharge un modèle et ses métadonnées depuis un checkpoint.

    Args:
        checkpoint_path: Chemin du fichier ``.pt`` sauvegardé à l'entraînement.
        device: Device cible (déduit automatiquement si ``None``).

    Returns:
        Un tuple ``(model, metadata)`` où ``model`` est en mode évaluation.

    Raises:
        FileNotFoundError: Si le checkpoint n'existe pas.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {path}")

    device = device or resolve_device("auto")
    payload = torch.load(path, map_location=device)
    model_config = payload.get("model_config", {})

    # Seul l'autoencodeur existe en Phase 2 ; PatchCore sera ajouté en Phase 4.
    model = build_autoencoder(model_config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


class AnomalyInferencePipeline:
    """Applique un modèle entraîné à des images individuelles.

    Args:
        model: Modèle implémentant ``anomaly_map`` / ``anomaly_score``.
        image_size: Taille de redimensionnement (doit correspondre à l'entraînement).
        device: CPU ou GPU (déduit automatiquement si ``None``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        image_size: int = 224,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or resolve_device("auto")
        self.model = model.to(self.device).eval()
        self.image_size = image_size
        self.transform = build_transforms(image_size, train=False)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, image_size: int = 224
    ) -> "AnomalyInferencePipeline":
        """Construit le pipeline directement depuis un checkpoint.

        La taille d'image est lue dans le checkpoint si disponible.
        """
        model, payload = load_model_from_checkpoint(checkpoint_path)
        size = int(payload.get("image_size", image_size))
        return cls(model, image_size=size)

    @torch.no_grad()
    def predict(self, image: str | Path | Image.Image) -> dict[str, Any]:
        """Prédit le score et la carte d'anomalie d'une image.

        Args:
            image: Chemin d'une image ou objet PIL déjà chargé.

        Returns:
            Dictionnaire :

            - ``anomaly_score`` : float (plus élevé = plus anormal) ;
            - ``anomaly_map`` : carte 2D normalisée dans ``[0, 1]`` à la taille
              du modèle ;
            - ``image_width`` / ``image_height`` : dimensions d'origine.
        """
        pil = image if isinstance(image, Image.Image) else load_image(image)
        pil = pil.convert("RGB")
        original_w, original_h = pil.size

        tensor = self.transform(pil).unsqueeze(0).to(self.device)  # (1,3,H,W)
        score = float(self.model.anomaly_score(tensor).item())

        raw_map = self.model.anomaly_map(tensor)  # (1,1,H,W)
        resized = resize_anomaly_map(raw_map, self.image_size, self.image_size)
        normalized = min_max_normalize(map_to_numpy(resized))

        return {
            "anomaly_score": score,
            "anomaly_map": normalized.astype(np.float32),
            "image_width": original_w,
            "image_height": original_h,
        }
