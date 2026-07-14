# Questions d'entretien — réponses courtes

**1. Pourquoi entraîner uniquement sur des images normales ?**
Les anomalies sont rares, variées et coûteuses à annoter. Apprendre le
« normal » puis mesurer l'écart couvre des défauts jamais vus à l'entraînement.

**2. Pourquoi l'AUROC peut-elle être insuffisante ?**
Quand les anomalies sont rares, l'AUROC reste flatteuse même avec beaucoup de
fausses alertes. L'AUPRC et le F1 au seuil retenu sont plus informatifs.

**3. Différence entre score image-level et carte pixel-level ?**
Le score dit *si* une image est anormale ; la carte dit *où*. Le pixel-level
exige une vérité terrain spatiale (masques), absente en santé (RSNA).

**4. Comment avez-vous évité les fuites de données ?**
Val tirée des normales de train ; test jamais utilisé pour le seuil. Splits
**par patient** (santé) et **par zone** (aérien) : un groupe ne franchit jamais
deux jeux. Testé automatiquement.

**5. Pourquoi comparer un autoencodeur et PatchCore ?**
L'autoencodeur est une baseline simple mais peut reconstruire des anomalies.
PatchCore (features pré-entraînées + mémoire) donne des cartes plus nettes. On
compare qualité ET coût (temps, mémoire) à protocole identique.

**6. Comment choisir le seuil sans utiliser le test ?**
Percentile des scores des normales de validation (principal), ou F1-max / coût
sur une **calibration disjointe** du test.

**7. Que se passe-t-il en cas de changement de domaine ?**
Les performances chutent (textures, résolution, type d'anomalie différents).
D'où l'architecture modulaire : on réadapte config et split, pas le cœur.

**8. Pourquoi une heatmap n'est-elle pas une explication parfaite ?**
Elle montre où l'erreur de reconstruction (ou la distance mémoire) est forte,
pas la cause. Corrélation spatiale ≠ causalité.

**9. Comment déployer ce système à grande échelle ?**
API conteneurisée derrière un load-balancer, modèles versionnés, monitoring de
dérive des scores, ré-étalonnage du seuil, files d'attente pour le batch,
et supervision humaine sur les cas limites.

**10. Que faudrait-il avant une utilisation médicale réelle ?**
Validation clinique, données représentatives et anonymisées, approbation
réglementaire (dispositif médical), et un médecin dans la boucle. Le projet
actuel ne le permet PAS.
