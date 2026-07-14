"""Configuration et registre de modèles pour l'API.

Le *registre* découvre les checkpoints présents dans le dossier ``models/`` et
fournit, à la demande, un pipeline d'inférence prêt à l'emploi. Les pipelines
sont mis en cache pour ne pas recharger le modèle à chaque requête.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from anomaly_detection.constants import SUPPORTED_IMAGE_MIME_TYPES
from anomaly_detection.inference.pipeline import (
    AnomalyInferencePipeline,
    load_model_from_checkpoint,
)

# Cache de pipelines : clé = (dossier, nom de checkpoint).
_PIPELINE_CACHE: dict[tuple[str, str], AnomalyInferencePipeline] = {}
_META_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


@dataclass(frozen=True)
class Settings:
    """Paramètres du service, surchargeable par variables d'environnement.

    Attributes:
        models_dir: Dossier contenant les checkpoints ``.pt``.
        max_upload_bytes: Taille maximale acceptée pour une image envoyée.
        default_threshold: Seuil par défaut si aucun n'est fourni.
        allowed_mime: Types MIME d'image acceptés.
        version: Version du service.
    """

    models_dir: Path = Path(os.environ.get("ANOMALY_MODELS_DIR", "models"))
    max_upload_bytes: int = (
        int(os.environ.get("ANOMALY_MAX_UPLOAD_MB", "10")) * 1024 * 1024
    )
    default_threshold: float = float(os.environ.get("ANOMALY_DEFAULT_THRESHOLD", "0.5"))
    allowed_mime: tuple[str, ...] = SUPPORTED_IMAGE_MIME_TYPES
    version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Retourne les paramètres du service (mis en cache)."""
    return Settings()


class ModelRegistry:
    """Découvre les checkpoints et fournit les pipelines d'inférence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _checkpoints(self) -> list[Path]:
        directory = self.settings.models_dir
        return sorted(directory.glob("*.pt")) if directory.is_dir() else []

    def _meta(self, path: Path) -> dict[str, Any]:
        key = (str(self.settings.models_dir), path.name)
        if key not in _META_CACHE:
            payload = torch.load(path, map_location="cpu")
            _META_CACHE[key] = {
                "domain": payload.get("domain", "unknown"),
                "model": payload.get("model", "autoencoder"),
                "image_size": int(payload.get("image_size", 224)),
            }
        return _META_CACHE[key]

    def list_models(self) -> list[dict[str, Any]]:
        """Liste les modèles disponibles avec leurs métadonnées."""
        out = []
        for path in self._checkpoints():
            meta = self._meta(path)
            out.append(
                {
                    "key": path.stem,
                    "domain": meta["domain"],
                    "model": meta["model"],
                    "image_size": meta["image_size"],
                    "threshold": self.settings.default_threshold,
                }
            )
        return out

    def default_key(self) -> str | None:
        """Clé du premier modèle disponible, ou ``None``."""
        checkpoints = self._checkpoints()
        return checkpoints[0].stem if checkpoints else None

    def entry(self, key: str) -> dict[str, Any]:
        """Métadonnées d'un modèle par sa clé.

        Raises:
            KeyError: Si la clé n'existe pas.
        """
        path = self.settings.models_dir / f"{key}.pt"
        if not path.is_file():
            raise KeyError(key)
        return self._meta(path)

    def get_pipeline(self, key: str) -> AnomalyInferencePipeline:
        """Retourne (en cache) le pipeline d'inférence d'un modèle.

        Raises:
            KeyError: Si le checkpoint n'existe pas.
        """
        cache_key = (str(self.settings.models_dir), key)
        if cache_key not in _PIPELINE_CACHE:
            path = self.settings.models_dir / f"{key}.pt"
            if not path.is_file():
                raise KeyError(key)
            model, meta = load_model_from_checkpoint(path, device=torch.device("cpu"))
            image_size = int(meta.get("image_size", 224))
            _PIPELINE_CACHE[cache_key] = AnomalyInferencePipeline(
                model, image_size=image_size, device=torch.device("cpu")
            )
        return _PIPELINE_CACHE[cache_key]


def get_registry(settings: Settings | None = None) -> ModelRegistry:
    """Fabrique un registre à partir des paramètres (dépendance FastAPI)."""
    return ModelRegistry(settings or get_settings())
