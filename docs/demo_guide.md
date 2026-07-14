# Guide de démonstration (5 minutes)

Objectif : présenter le projet à un recruteur en 5 minutes, de l'installation à
la démo live.

## 0. Préparation (avant l'entretien)
```bash
pip install -e ".[dev]"
python scripts/download_data.py --synthetic
python scripts/train.py --category synthetic --epochs 2      # autoencodeur
uvicorn app.api.main:app                                     # terminal 1
streamlit run app/demo/streamlit_app.py                      # terminal 2
```

## 1. Pitch (30 s)
« Un même pipeline de détection d'anomalies, entraîné uniquement sur des
images normales, appliqué à trois domaines — industrie, santé, aérien — exposé
par une API et une démo. »

## 2. Démo live (2 min)
- Ouvrir la démo Streamlit, choisir un modèle, importer une image.
- Montrer score, décision, seuil, et la **heatmap** côte à côte.
- Souligner l'**avertissement médical** (pas un diagnostic).

## 3. Sous le capot (1,5 min)
- Architecture modulaire : `src/` (code métier), configs YAML en cascade,
  socle dataset commun + adaptateurs par domaine.
- **Anti-fuite** : split par patient / par zone (montrer `tests/unit/test_domains.py`).
- Comparaison autoencodeur vs PatchCore (montrer `reports/metrics/comparison_*`).

## 4. Rigueur (1 min)
- CI verte (lint + tests), suivi MLflow, Docker Compose, rapport LaTeX.
- Honnêteté : seuil jamais choisi sur le test, pas de métrique pixel sans masque.

## Points de repli si une question surgit
Voir `docs/interviews/questions.md`.
