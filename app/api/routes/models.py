"""Route listant les modèles disponibles."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import Settings, get_registry, get_settings
from app.api.schemas import ModelInfo, ModelsResponse

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def list_models(settings: Settings = Depends(get_settings)) -> ModelsResponse:
    """Liste les modèles (checkpoints) disponibles."""
    registry = get_registry(settings)
    return ModelsResponse(models=[ModelInfo(**m) for m in registry.list_models()])
