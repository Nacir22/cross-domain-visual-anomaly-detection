# Model Card — Cross-Domain Visual Anomaly Detection

> Document vivant. Les valeurs chiffrées seront ajoutées après expérimentation
> réelle (Colab GPU). Aucune métrique n'est inventée.

## Détails du modèle
- **Tâche** : détection d'anomalies visuelles *normal-only* (one-class).
- **Modèles** : autoencodeur convolutif (baseline) et PatchCore (backbone
  ResNet pré-entraîné + banque mémoire).
- **Sorties** : score d'anomalie (image), carte d'anomalie (pixel), décision
  binaire via seuil.
- **Domaines** : industrie (MVTec AD), santé (RSNA, image-level), aérien
  (xView2/xBD).

## Usage prévu
Démonstration et portfolio : inspection industrielle, tri d'images pour
priorisation humaine, prototypage de détection de changement aérien.

## Usage NON prévu (hors périmètre)
- **Diagnostic médical.** Le module santé n'est **pas** un dispositif médical
  et n'a fait l'objet d'**aucune validation clinique**.
- Toute décision critique automatisée sans supervision humaine.
- Usage commercial de MVTec AD (licence CC BY-NC-SA 4.0, non commercial).

## Données d'entraînement
Uniquement des images **normales**. Séparation anti-fuite : par patient
(santé), par zone géographique (aérien). Détails et licences : `data/README.md`.

## Métriques et évaluation
Image-level : AUROC, AUPRC, précision, rappel, F1, matrice de confusion,
FPR/FNR. Pixel-level (industrie, aérien uniquement) : AUROC/AUPRC pixel, Dice,
IoU. Seuil choisi sur la **validation** (jamais le test). *Valeurs : à compléter.*

## Limites connues
- Un autoencodeur peut reconstruire certaines anomalies (score sous-estimé).
- Sensibilité au **changement de domaine** (textures, résolution, variabilité
  du normal).
- Une **heatmap n'est pas une explication causale**.
- Performance non garantie hors distribution d'entraînement.

## Considérations éthiques
- **Confidentialité** : l'API ne stocke ni ne journalise les images envoyées.
- **RGPD / anonymisation** : données médicales à anonymiser en amont.
- **Biais** : les datasets publics peuvent être non représentatifs.
- **Faux négatifs** : une anomalie manquée peut être coûteuse ; le seuil « coût »
  permet d'en tenir compte, mais la supervision humaine reste requise.

## Reproductibilité
Graines fixées (Python/NumPy/PyTorch), configuration versionnée, artefacts
suivis via MLflow. Certaines opérations GPU restent non parfaitement
déterministes (documenté).
