"""Point d'entrée de l'API FastAPI.

Lancement local ::

    uvicorn app.api.main:app --reload

Documentation interactive : http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import health, models, predict

app = FastAPI(
    title="Cross-Domain Visual Anomaly Detection API",
    description=(
        "Détection d'anomalies visuelles normal-only sur trois domaines. "
        "AVERTISSEMENT : le module médical n'est PAS un dispositif de diagnostic."
    ),
    version="0.1.0",
)

app.include_router(health.router, tags=["health"])
app.include_router(models.router, tags=["models"])
app.include_router(predict.router, tags=["predict"])


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Message d'accueil minimal."""
    return {"service": "anomaly-detection", "docs": "/docs"}
