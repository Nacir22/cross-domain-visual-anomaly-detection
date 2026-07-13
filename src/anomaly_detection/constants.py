"""Constantes partagées du projet.

Centraliser les valeurs « magiques » (formats d'image acceptés, noms de
domaines, etc.) évite de les recopier — et donc de les désynchroniser — dans
plusieurs fichiers.
"""

from __future__ import annotations

from typing import Final

# Domaines pris en charge par le pipeline unique.
DOMAIN_INDUSTRIAL: Final = "industrial"
DOMAIN_MEDICAL: Final = "medical"
DOMAIN_AERIAL: Final = "aerial"
SUPPORTED_DOMAINS: Final[tuple[str, ...]] = (
    DOMAIN_INDUSTRIAL,
    DOMAIN_MEDICAL,
    DOMAIN_AERIAL,
)

# Modèles disponibles.
MODEL_AUTOENCODER: Final = "autoencoder"
MODEL_PATCHCORE: Final = "patchcore"
SUPPORTED_MODELS: Final[tuple[str, ...]] = (MODEL_AUTOENCODER, MODEL_PATCHCORE)

# Extensions d'image acceptées par l'API et les loaders.
SUPPORTED_IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)

# Types MIME acceptés par l'API (défense côté serveur).
SUPPORTED_IMAGE_MIME_TYPES: Final[tuple[str, ...]] = (
    "image/png",
    "image/jpeg",
    "image/bmp",
    "image/tiff",
)

# Graine par défaut pour la reproductibilité.
DEFAULT_SEED: Final = 42
