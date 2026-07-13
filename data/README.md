# Données

⚠️ **Aucune donnée volumineuse n'est versionnée.** Ces dossiers ne contiennent
que des `.gitkeep`. Les datasets se téléchargent via `scripts/download_data.py`
(implémenté à partir de la Phase 1).

## Organisation

```text
data/raw/        données brutes téléchargées (ignoré par git)
data/interim/    données intermédiaires (splits, index)
data/processed/  données prêtes pour l'entraînement
```

## Datasets et licences

| Domaine   | Dataset        | Licence                         | Accès                         |
|-----------|----------------|---------------------------------|-------------------------------|
| Industrie | MVTec AD       | CC BY-NC-SA 4.0 (non commercial)| Site MVTec (enregistrement)   |
| Santé     | RSNA Pneumonia | Conditions RSNA / Kaggle        | Kaggle (compte requis)        |
| Aérien    | xView2 / xBD   | Creative Commons                | Site xView2 (enregistrement)  |

Les licences des datasets sont **distinctes** de la licence MIT du code.
Aucune condition d'utilisation ne sera contournée ; les datasets exigeant un
compte ou l'acceptation manuelle de conditions ne sont jamais téléchargés
automatiquement sans accord explicite.

## Anti-fuite de données
- Médical : séparation **par patient** (`split_by: patient_id`).
- Aérien : séparation **par zone géographique** (`split_by: geographic_zone`).
- Général : aucune version augmentée d'une même image ne franchit deux splits.

## MVTec AD — structure attendue (Phase 1)

Après téléchargement et extraction dans `data/raw/mvtec_ad/`, chaque catégorie
`<cat>` (ex. `bottle`) doit avoir cette forme :

```text
data/raw/mvtec_ad/<cat>/
├── train/good/*.png              # images NORMALES uniquement (-> train + val)
├── test/good/*.png               # normales de test
├── test/<defect>/*.png           # images anormales (rayure, fissure, ...)
└── ground_truth/<defect>/*_mask.png   # masques pixel (vérité terrain)
```

### Préparer les données

```bash
# 1. Afficher les instructions officielles + licence
python scripts/download_data.py

# 2. (Sans téléchargement) générer une catégorie SYNTHÉTIQUE pour tester/CI
python scripts/download_data.py --synthetic

# 3. Vérifier une catégorie téléchargée
python scripts/download_data.py --verify --category bottle
```

### Découpage train / val / test

Réalisé par `anomaly_detection.data.splits.create_splits` :

- **train** et **val** proviennent exclusivement de `train/good` (normales),
  mélangées avec une **graine fixe** puis découpées selon `val_fraction`.
- **test** provient du dossier `test/` (jamais touché par le choix de seuil) et
  contient normales + anomalies, avec masques quand disponibles.

Les splits sont sauvegardés en JSON dans `data/interim/` (chemins relatifs,
donc portables) et rechargeables à l'identique.
