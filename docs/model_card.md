# Model Card — Cross-Domain Visual Anomaly Detection

> Document vivant. Les valeurs chiffrées seront ajoutées après expérimentation.
> Aucune métrique n'est inventée.

## Usage prévu
Détection d'anomalies visuelles *normal-only* à des fins de démonstration et
de portfolio, sur trois domaines : industrie, santé, aérien.

## Usage NON prévu
- **Diagnostic médical.** Le module santé n'est pas un dispositif médical et
  n'a fait l'objet d'aucune validation clinique.
- Toute décision critique sans supervision humaine.

## Données d'entraînement
Uniquement des images **normales** (voir `data/README.md` pour licences).

## Métriques
Image-level (AUROC, AUPRC, F1...) pour les trois domaines ; pixel-level
uniquement là où un masque fiable existe (industrie, aérien). *À compléter.*

## Limites connues
Sensibilité au changement de domaine ; un autoencodeur peut reconstruire
certaines anomalies ; une heatmap n'est pas une explication causale.

## Considérations éthiques
Confidentialité des données médicales (anonymisation, pas de stockage
automatique côté API), biais possibles des datasets, RGPD.
