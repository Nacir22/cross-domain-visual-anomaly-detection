"""Interface de démonstration (Streamlit) pour la détection d'anomalies.

Elle dialogue avec l'API FastAPI : l'utilisateur choisit un domaine et un
modèle, importe une image, lance l'analyse, puis visualise le score, la
décision, le seuil et la carte de chaleur.

Choix Streamlit vs Gradio (décidé en Phase 0) : Streamlit est retenu pour sa
souplesse de mise en page (colonnes, comparaison image/heatmap côte à côte),
adaptée à un rendu « vitrine » de portfolio.

Lancement ::

    streamlit run app/demo/streamlit_app.py
"""

from __future__ import annotations

import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("ANOMALY_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Détection d'anomalies visuelles", layout="wide")
st.title("Détection d'anomalies visuelles — démonstration")

st.warning(
    "Projet de recherche / portfolio. Les résultats ne constituent PAS un "
    "diagnostic. Le module médical n'est pas un dispositif médical."
)


@st.cache_data(ttl=30)
def fetch_models() -> list[dict]:
    """Récupère la liste des modèles disponibles auprès de l'API."""
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except requests.RequestException:
        return []


models = fetch_models()

with st.sidebar:
    st.header("Configuration")
    if not models:
        st.error(f"API injoignable ou aucun modèle. Vérifiez {API_URL}.")
    keys = [m["key"] for m in models]
    selected = st.selectbox("Modèle", keys) if keys else None
    entry = next((m for m in models if m["key"] == selected), None)
    if entry:
        st.caption(f"Domaine : {entry['domain']} — modèle : {entry['model']}")
    threshold = st.slider("Seuil de décision", 0.0, 5.0, 0.5, 0.01)
    uploaded = st.file_uploader("Image à analyser", type=["png", "jpg", "jpeg", "bmp"])

if entry and entry["domain"] == "medical":
    st.info(
        "Domaine médical : sortie image-level uniquement, sans localisation "
        "fiable. Ceci n'est en aucun cas un avis médical."
    )

if uploaded and selected and st.button("Analyser", type="primary"):
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    params = {"model": selected, "threshold": threshold}
    try:
        resp = requests.post(
            f"{API_URL}/predict", files=files, params=params, timeout=30
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.RequestException as exc:
        st.error(f"Erreur lors de l'analyse : {exc}")
        st.stop()

    score = result["anomaly_score"]
    is_anomaly = result["is_anomaly"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Score d'anomalie", f"{score:.3f}")
    col2.metric("Seuil", f"{result['threshold']:.3f}")
    col3.metric("Décision", "ANOMALIE" if is_anomaly else "Normal")
    st.caption(f"Temps de traitement : {result['processing_time_ms']} ms")

    left, right = st.columns(2)
    left.subheader("Image d'origine")
    left.image(Image.open(io.BytesIO(uploaded.getvalue())), use_column_width=True)
    right.subheader("Carte de chaleur")
    heatmap_bytes = base64.b64decode(result["heatmap"])
    right.image(Image.open(io.BytesIO(heatmap_bytes)), use_column_width=True)

    st.caption(
        "La carte de chaleur indique les zones de forte anomalie ; ce n'est "
        "pas une explication causale."
    )
