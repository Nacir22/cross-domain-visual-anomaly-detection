"""Datasets PyTorch pour les trois domaines (industrie, santé, aérien).

Architecture
------------
Un seul code de chargement d'image (``AnomalyImageDataset``) est partagé par
les trois domaines : c'est le cœur de la démonstration de *transférabilité*.
Ce qui change d'un domaine à l'autre n'est PAS ce code, mais :

- la manière de construire la liste des éléments (``entries``) et les splits
  (règles propres : par patient en santé, par zone en aérien) ;
- une fine classe spécialisée par domaine, qui ne fait qu'assembler les entries.

Chaque élément est un dict : ``image`` (tenseur ``(3, H, W)``), ``label``
(0/1), ``mask`` (tenseur ``(1, H, W)``, zéros si absent), ``path`` et
``defect_type``/``group``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from anomaly_detection.data.preprocessing import load_image
from anomaly_detection.data.splits import create_splits
from anomaly_detection.data.transforms import build_mask_transform, build_transforms

_VALID_SPLITS = ("train", "val", "test")


class AnomalyImageDataset(Dataset):
    """Dataset générique (indépendant du domaine) construit depuis des entries.

    Args:
        root: Racine à laquelle les chemins d'images sont relatifs.
        entries: Liste de dicts ``{image, label, mask?, defect_type?/group?}``.
        image_size: Côté des images carrées en sortie.
        train: Si ``True``, autorise une augmentation légère (déterministe sinon).
        transform: Transformation image personnalisée (sinon construite auto).
        mask_transform: Transformation masque personnalisée (sinon auto).
    """

    def __init__(
        self,
        root: str | Path,
        entries: list[dict[str, object]],
        image_size: int = 224,
        train: bool = False,
        transform: Any | None = None,
        mask_transform: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.entries = entries
        self.image_size = image_size
        self.transform = transform or build_transforms(image_size, train=train)
        self.mask_transform = mask_transform or build_mask_transform(image_size)

    def __len__(self) -> int:
        """Nombre d'éléments."""
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Charge et transforme l'élément n°``index``.

        Args:
            index: Position de l'élément.

        Returns:
            Le dictionnaire décrit dans la docstring du module.
        """
        entry = self.entries[index]
        image_path = self.root / str(entry["image"])
        image = self.transform(load_image(image_path))
        label = int(entry["label"])

        mask_rel = entry.get("mask")
        if mask_rel:
            mask = self.mask_transform(load_image(self.root / str(mask_rel)))
            mask = (mask > 0).float()  # binarise
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        group = entry.get("defect_type") or entry.get("group") or "normal"
        return {
            "image": image,
            "label": label,
            "mask": mask,
            "path": str(image_path),
            "defect_type": str(group),
            "group": str(group),
        }


def _select_split(
    splits: dict[str, list[dict[str, object]]], split: str
) -> list[dict[str, object]]:
    """Valide et retourne le sous-ensemble demandé."""
    if split not in _VALID_SPLITS:
        raise ValueError(f"split invalide : {split!r}. Attendu : {_VALID_SPLITS}.")
    return splits[split]


class MVTecDataset(AnomalyImageDataset):
    """Domaine industriel (MVTec AD) — une catégorie, un split.

    Args:
        dataset_root: Racine des catégories (ex. ``data/raw/mvtec_ad``).
        category: Nom de la catégorie (ex. ``"bottle"``).
        split: ``"train"``, ``"val"`` ou ``"test"``.
        image_size: Côté des images carrées.
        val_fraction: Fraction de ``train/good`` réservée à la validation.
        seed: Graine du découpage reproductible.
        transform: Transformation image personnalisée.
        mask_transform: Transformation masque personnalisée.
        splits: Splits pré-calculés (sinon générés). Garantit un découpage partagé.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        category: str,
        split: str = "train",
        image_size: int = 224,
        val_fraction: float = 0.2,
        seed: int = 42,
        transform: Any | None = None,
        mask_transform: Any | None = None,
        splits: dict[str, list[dict[str, object]]] | None = None,
    ) -> None:
        root = Path(dataset_root)
        if splits is None:
            splits = create_splits(root, category, val_fraction, seed)
        self.category = category
        self.split = split
        super().__init__(
            root,
            _select_split(splits, split),
            image_size=image_size,
            train=(split == "train"),
            transform=transform,
            mask_transform=mask_transform,
        )


class MedicalDataset(AnomalyImageDataset):
    """Domaine santé (ex. RSNA) — image-level uniquement (pas de masque).

    Le découpage doit être fait PAR PATIENT (voir ``data.domains``) pour éviter
    toute fuite entre images d'un même patient. Ce n'est PAS un dispositif de
    diagnostic.

    Args:
        root: Racine des images médicales.
        split: ``"train"``, ``"val"`` ou ``"test"``.
        splits: Splits pré-calculés (groupés par patient).
        image_size: Côté des images carrées.
        transform: Transformation image personnalisée.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        splits: dict[str, list[dict[str, object]]],
        image_size: int = 224,
        transform: Any | None = None,
    ) -> None:
        self.split = split
        super().__init__(
            root,
            _select_split(splits, split),
            image_size=image_size,
            train=(split == "train"),
            transform=transform,
        )


class AerialDataset(AnomalyImageDataset):
    """Domaine aérien (ex. xView2/xBD) — normal = ``no damage``.

    Le découpage doit être fait PAR ZONE GÉOGRAPHIQUE pour éviter la fuite entre
    tuiles voisines.

    Args:
        root: Racine des tuiles aériennes.
        split: ``"train"``, ``"val"`` ou ``"test"``.
        splits: Splits pré-calculés (groupés par zone).
        image_size: Côté des images carrées.
        transform: Transformation image personnalisée.
        mask_transform: Transformation masque personnalisée.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        splits: dict[str, list[dict[str, object]]],
        image_size: int = 224,
        transform: Any | None = None,
        mask_transform: Any | None = None,
    ) -> None:
        self.split = split
        super().__init__(
            root,
            _select_split(splits, split),
            image_size=image_size,
            train=(split == "train"),
            transform=transform,
            mask_transform=mask_transform,
        )
