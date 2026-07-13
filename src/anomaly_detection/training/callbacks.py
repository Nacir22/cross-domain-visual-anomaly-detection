"""Callbacks d'entraînement : historique et sauvegarde du meilleur modèle.

Un *callback* est un petit objet qu'on appelle à chaque époque pour réagir aux
résultats : ici, enregistrer les pertes et sauvegarder le meilleur checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class LossHistory:
    """Mémorise l'évolution des pertes d'entraînement et de validation.

    Attributes:
        train: Perte d'entraînement par époque.
        val: Perte de validation par époque.
    """

    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)

    def update(self, train_loss: float, val_loss: float) -> None:
        """Ajoute les pertes d'une époque."""
        self.train.append(train_loss)
        self.val.append(val_loss)

    def as_dict(self) -> dict[str, list[float]]:
        """Retourne l'historique sous forme de dictionnaire sérialisable."""
        return {"train": self.train, "val": self.val}


class CheckpointSaver:
    """Sauvegarde le modèle chaque fois que la perte de validation s'améliore.

    Args:
        path: Chemin du fichier checkpoint à écrire.
        extra: Métadonnées à enregistrer avec les poids (config, nom du modèle).
    """

    def __init__(self, path: str | Path, extra: dict | None = None) -> None:
        self.path = Path(path)
        self.extra = extra or {}
        self.best_val = float("inf")
        self.best_epoch = -1

    def step(self, model: torch.nn.Module, val_loss: float, epoch: int) -> bool:
        """Sauvegarde le modèle si ``val_loss`` bat le meilleur score connu.

        Args:
            model: Modèle à sauvegarder.
            val_loss: Perte de validation de l'époque courante.
            epoch: Numéro de l'époque (pour la traçabilité).

        Returns:
            ``True`` si un nouveau meilleur modèle a été sauvegardé.
        """
        if val_loss >= self.best_val:
            return False
        self.best_val = val_loss
        self.best_epoch = epoch
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "val_loss": val_loss,
            "epoch": epoch,
            **self.extra,
        }
        torch.save(payload, self.path)
        return True
