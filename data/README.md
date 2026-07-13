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
