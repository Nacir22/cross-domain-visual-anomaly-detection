"""Dataset PyTorch pour MVTec AD (détection d'anomalies *normal-only*).

Concepts
--------
Un ``Dataset`` PyTorch est un objet qui sait répondre à deux questions :
« combien d'éléments ? » (:meth:`__len__`) et « donne-moi l'élément n°i »
(:meth:`__getitem__`). PyTorch s'en sert pour construire des *batches*
(paquets d'images traités ensemble) via un ``DataLoader``.

Rappel du cadre *one-class* :

- le jeu **train** ne contient que des images **normales** (label 0) ;
- le jeu **test** contient normales (0) et anormales (1), avec parfois un
  **masque** indiquant où se situe l'anomalie (vérité terrain pixel).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from anomaly_detection.data.preprocessing import load_image
from anomaly_detection.data.splits import create_splits
from anomaly_detection.data.transforms import build_mask_transform, build_transforms


class MVTecDataset(Dataset):
    """Dataset d'une catégorie MVTec AD pour un split donné.

    Chaque élément retourné est un dictionnaire :

    - ``image`` : tenseur ``(3, H, W)`` ;
    - ``label`` : ``0`` (normal) ou ``1`` (anomalie) ;
    - ``mask``  : tenseur ``(1, H, W)`` (zéros si aucun masque) ;
    - ``path``  : chemin de l'image (chaîne) ;
    - ``defect_type`` : type de défaut (``"good"`` si normal).

    Args:
        dataset_root: Racine des catégories (ex. ``data/raw/mvtec_ad``).
        category: Nom de la catégorie (ex. ``"bottle"``).
        split: ``"train"``, ``"val"`` ou ``"test"``.
        image_size: Côté des images carrées en sortie.
        val_fraction: Fraction de ``train/good`` réservée à la validation.
        seed: Graine du découpage reproductible.
        transform: Transformation image personnalisée (sinon construite auto).
        mask_transform: Transformation masque personnalisée (sinon auto).
        splits: Splits pré-calculés (sinon générés via ``create_splits``). Sert
            à garantir que train/val/test partagent EXACTEMENT le même découpage.

    Raises:
        ValueError: Si ``split`` n'est pas reconnu.
    """

    _VALID_SPLITS = ("train", "val", "test")

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
        if split not in self._VALID_SPLITS:
            raise ValueError(
                f"split invalide : {split!r}. Attendu : {self._VALID_SPLITS}."
            )

        self.root = Path(dataset_root)
        self.category = category
        self.split = split
        self.image_size = image_size

        if splits is None:
            splits = create_splits(self.root, category, val_fraction, seed)
        self.entries: list[dict[str, object]] = splits[split]

        # En entraînement on peut augmenter légèrement ; sinon déterministe.
        self.transform = transform or build_transforms(
            image_size, train=(split == "train")
        )
        self.mask_transform = mask_transform or build_mask_transform(image_size)

    def __len__(self) -> int:
        """Nombre d'éléments du split."""
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Charge et transforme l'élément n°``index``.

        Args:
            index: Position de l'élément dans le split.

        Returns:
            Le dictionnaire décrit dans la docstring de la classe.
        """
        entry = self.entries[index]
        image_path = self.root / str(entry["image"])
        image = self.transform(load_image(image_path))

        label = int(entry["label"])

        # Masque : présent uniquement pour certaines anomalies. Sinon, un masque
        # de zéros de la bonne taille (cohérence des dimensions dans le batch).
        mask_rel = entry.get("mask")
        if mask_rel:
            mask = self.mask_transform(load_image(self.root / str(mask_rel)))
            # Binarise : tout pixel non nul devient 1.
            mask = (mask > 0).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        return {
            "image": image,
            "label": label,
            "mask": mask,
            "path": str(image_path),
            "defect_type": str(entry.get("defect_type", "good")),
        }
