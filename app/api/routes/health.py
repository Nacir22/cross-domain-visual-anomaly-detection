"""Route de santé : vérifie que le service répond."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import Settings, get_settings
from app.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Retourne l'état du service et sa version."""
    return HealthResponse(status="ok", version=settings.version)
