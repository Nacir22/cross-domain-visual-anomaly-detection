#!/usr/bin/env bash
# Exemple d'appel à l'API de détection d'anomalies.
# Prérequis : API lancée (uvicorn app.api.main:app) et un modèle dans models/.
set -euo pipefail

API="${ANOMALY_API_URL:-http://localhost:8000}"

echo "== /health =="
curl -s "$API/health"; echo

echo "== /models =="
curl -s "$API/models"; echo

echo "== /predict =="
curl -s -X POST "$API/predict?model=industrial_autoencoder_bottle" \
  -F "file=@data/raw/mvtec_ad/bottle/test/broken_large/000.png"
echo
