"""Route de prédiction : reçoit une image, renvoie score, décision et heatmap."""

from __future__ import annotations

import io
import logging
import time

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from anomaly_detection.inference.postprocessing import heatmap_to_base64_png
from app.api.dependencies import Settings, get_registry, get_settings
from app.api.schemas import ImageMetadata, PredictResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    model: str | None = Query(None, description="Clé du modèle (voir /models)."),
    threshold: float | None = Query(None, description="Seuil de décision."),
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    """Analyse une image et retourne un score d'anomalie et une heatmap.

    Sécurité et robustesse : vérification du type MIME, limite de taille,
    gestion des images corrompues, messages d'erreur clairs. L'image N'EST
    JAMAIS enregistrée (confidentialité, ex. données médicales) et n'apparaît
    pas dans les journaux.
    """
    registry = get_registry(settings)

    # 1) Type MIME.
    if file.content_type not in settings.allowed_mime:
        raise HTTPException(
            status_code=415,
            detail=f"Type non supporté : {file.content_type}. "
            f"Attendu : {settings.allowed_mime}.",
        )

    # 2) Taille.
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux.")
    if not data:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    # 3) Modèle.
    key = model or registry.default_key()
    if key is None:
        raise HTTPException(status_code=503, detail="Aucun modèle disponible.")
    try:
        pipeline = registry.get_pipeline(key)
        meta = registry.entry(key)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Modèle inconnu : {key}") from None

    # 4) Image lisible.
    try:
        pil = Image.open(io.BytesIO(data))
        pil.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400, detail="Image illisible ou corrompue."
        ) from None

    # 5) Inférence (l'image reste en mémoire, jamais écrite sur disque).
    start = time.perf_counter()
    result = pipeline.predict(pil)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    thr = threshold if threshold is not None else settings.default_threshold
    score = float(result["anomaly_score"])
    heatmap_b64 = heatmap_to_base64_png(result["anomaly_map"])

    logger.info(
        "predict model=%s domain=%s score=%.4f is_anomaly=%s (%.1f ms)",
        key,
        meta["domain"],
        score,
        score >= thr,
        elapsed_ms,
    )

    return PredictResponse(
        domain=meta["domain"],
        model_name=meta["model"],
        anomaly_score=score,
        threshold=float(thr),
        is_anomaly=bool(score >= thr),
        processing_time_ms=round(elapsed_ms, 2),
        heatmap=heatmap_b64,
        metadata=ImageMetadata(
            image_width=int(result["image_width"]),
            image_height=int(result["image_height"]),
        ),
    )
