# Cross-Domain Visual Anomaly Detection

**One pipeline, three domains: industry, healthcare and aerial imagery.**

> Une même architecture de détection d'anomalies visuelles est adaptée à
> plusieurs contextes métiers grâce à un pipeline modulaire, configurable et
> reproductible.

Le système apprend **uniquement à partir d'images normales** (approche
*normal-only* / *one-class*), puis, pour toute nouvelle image, produit :

- un **score d'anomalie** ;
- une **décision** (normal / anormal) via un seuil documenté ;
- une **carte de chaleur** localisant la zone suspecte.

Le tout est exposé par une **API FastAPI** et une **interface Streamlit**.

---

## Statut

🚧 Projet en construction — **Phase 0 (Fondations) terminée**. Les résultats
chiffrés seront ajoutés après expérimentation ; aucune métrique n'est
inventée.

## Domaines et datasets

| Domaine    | Dataset          | Licence                     | Localisation pixel |
|------------|------------------|-----------------------------|--------------------|
| Industrie  | MVTec AD         | CC BY-NC-SA 4.0 (non comm.) | Oui (masques)      |
| Santé      | RSNA Pneumonia   | Conditions RSNA/Kaggle      | Non (image-level)  |
| Aérien     | xView2 / xBD     | Creative Commons            | Oui (polygones)    |

> ⚠️ Le module médical est un projet de recherche. **Ce n'est pas un
> dispositif de diagnostic médical.**

## Deux environnements d'exécution

| Profil        | Matériel        | Usage                                    |
|---------------|-----------------|------------------------------------------|
| `local_cpu`   | PC Windows, CPU | Développement, tests, démo, mini-runs    |
| `colab_gpu`   | Google Colab    | Entraînement réel, PatchCore, métriques  |

Aucune installation CUDA locale n'est nécessaire : le développement reste
local, les calculs lourds se font sur Colab.

## Installation rapide

```bash
python -m venv .venv
# Windows PowerShell : .venv\Scripts\Activate.ps1
# Linux/macOS       : source .venv/bin/activate
pip install -e ".[dev]"
```

Pour PyTorch en CPU (Windows) :

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Vérifier l'environnement

```bash
python scripts/check_environment.py
```

## Qualité

```bash
ruff format . && ruff check .
pytest
```

## Structure du dépôt

```text
app/        API FastAPI + démo Streamlit
configs/    YAML : base, domaines, modèles, profils (local_cpu / colab_gpu)
data/       données (jamais committées) + data/README.md
docs/       rapport LaTeX, model card, questions d'entretien
notebooks/  exploration locale + notebooks/colab pour l'entraînement GPU
scripts/    CLI : check_environment, download_data, train, evaluate, predict
src/        package Python 'anomaly_detection' (code métier)
tests/      tests unitaires et d'intégration
```

## Feuille de route

- [x] Phase 0 — Fondations
- [ ] Phase 1 — Données industrielles (MVTec AD)
- [ ] Phase 2 — Baseline autoencodeur
- [ ] Phase 3 — Évaluation et interprétabilité
- [ ] Phase 4 — PatchCore + MLflow
- [ ] Phase 5 — Transférabilité multi-domaines
- [ ] Phase 6 — API, démo, Docker
- [ ] Phase 7 — Industrialisation & portfolio

## Licence

Code sous licence **MIT** (voir `LICENSE`). Les datasets conservent leurs
licences propres (voir `data/README.md`).
