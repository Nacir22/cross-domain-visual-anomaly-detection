"""Interface commune à tous les modèles de détection d'anomalies.

Pourquoi une interface ?
------------------------
Le projet compare plusieurs modèles (autoencodeur, puis PatchCore). Pour que le
reste du code (entraînement, évaluation, API) ne dépende PAS des détails de
chaque modèle, on définit un **contrat** commun : quel que soit le modèle, il
sait produire une *carte d'anomalie* et un *score d'anomalie*.

- **Carte d'anomalie** : une image ``(B, 1, H, W)`` où chaque pixel indique à
  quel point cet endroit semble anormal (plus c'est élevé, plus c'est suspect).
- **Score d'anomalie** : un nombre par image ``(B,)`` résumant la carte
  (typiquement le maximum ou la moyenne). Sert à la décision normal/anormal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class AnomalyModel(nn.Module, ABC):
    """Classe de base abstraite pour un détecteur d'anomalies visuelles.

    Toute sous-classe doit implémenter :meth:`anomaly_map`. Le score par défaut
    est la moyenne spatiale de la carte, mais une sous-classe peut le redéfinir.
    """

    @abstractmethod
    def anomaly_map(self, images: torch.Tensor) -> torch.Tensor:
        """Calcule la carte d'anomalie d'un batch d'images.

        Args:
            images: Batch de forme ``(B, 3, H, W)``.

        Returns:
            Carte d'anomalie de forme ``(B, 1, H, W)``, valeurs >= 0.
        """
        raise NotImplementedError

    def anomaly_score(self, images: torch.Tensor) -> torch.Tensor:
        """Résume la carte d'anomalie en un score par image.

        Par défaut : moyenne spatiale de la carte. Un score élevé = image plus
        probablement anormale.

        Args:
            images: Batch de forme ``(B, 3, H, W)``.

        Returns:
            Scores de forme ``(B,)``.
        """
        maps = self.anomaly_map(images)
        return maps.flatten(start_dim=1).mean(dim=1)
