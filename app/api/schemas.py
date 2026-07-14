"""Schémas de validation (Pydantic) des requêtes et réponses de l'API.

Pydantic valide et documente automatiquement les données : une réponse mal
formée est impossible, et l'API expose un schéma OpenAPI clair (visible sur
``/docs``).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImageMetadata(BaseModel):
    """Dimensions de l'image d'origine."""

    image_width: int = Field(..., ge=1)
    image_height: int = Field(..., ge=1)


class PredictResponse(BaseModel):
    """Réponse d'une prédiction d'anomalie."""

    domain: str
    model_name: str
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    processing_time_ms: float
    heatmap: str = Field(..., description="Image PNG de la heatmap, encodée en base64.")
    metadata: ImageMetadata


class HealthResponse(BaseModel):
    """État de santé du service."""

    status: str
    version: str


class ModelInfo(BaseModel):
    """Description d'un modèle disponible."""

    key: str
    domain: str
    model: str
    image_size: int
    threshold: float


class ModelsResponse(BaseModel):
    """Liste des modèles disponibles."""

    models: list[ModelInfo]


class ErrorResponse(BaseModel):
    """Message d'erreur lisible."""

    detail: str
