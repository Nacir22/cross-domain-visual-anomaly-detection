"""Création reproductible des jeux train / validation / test (MVTec AD).

Pourquoi ce module est crucial
------------------------------
La détection d'anomalies *normal-only* impose une règle stricte :

- **train** et **validation** ne contiennent QUE des images normales ;
- **test** contient normales + anormales et sert à mesurer la performance.

Deux pièges à éviter absolument :

1. **Fuite de données** : une même image (ou sa version augmentée) présente
   dans deux jeux fausserait les scores. Ici, val est tiré de ``train/good``,
   et le dossier ``test/`` n'est jamais touché.
2. **Non-reproductibilité** : sans graine, chaque exécution produirait un
   découpage différent. On mélange donc avec un générateur *seedé*.

Structure attendue du dataset MVTec AD, pour une catégorie ``<cat>`` ::

    <root>/<cat>/train/good/*.png            (normales -> train + val)
    <root>/<cat>/test/good/*.png             (normales -> test)
    <root>/<cat>/test/<defect>/*.png         (anormales -> test)
    <root>/<cat>/ground_truth/<defect>/*_mask.png  (masques pixel)
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from anomaly_detection.constants import SUPPORTED_IMAGE_EXTENSIONS

# Labels conventionnels du projet.
LABEL_NORMAL = 0
LABEL_ANOMALY = 1

# Nom du sous-dossier des images normales dans MVTec.
_GOOD = "good"


@dataclass(frozen=True)
class SampleEntry:
    """Un élément d'un jeu de données.

    Attributes:
        image: Chemin de l'image, relatif à la racine du dataset.
        label: ``0`` (normal) ou ``1`` (anomalie).
        defect_type: Type de défaut (``"good"`` si normal).
        mask: Chemin relatif du masque pixel, ou ``None`` si absent.
    """

    image: str
    label: int
    defect_type: str
    mask: str | None = None


def _list_images(directory: Path) -> list[Path]:
    """Liste triée des images d'un dossier (tri = reproductibilité)."""
    if not directory.is_dir():
        return []
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(files)


def _find_mask(category_dir: Path, defect_type: str, image_path: Path) -> str | None:
    """Retrouve le masque associé à une image anormale, s'il existe.

    Convention MVTec : ``ground_truth/<defect>/<stem>_mask.png``.
    """
    mask_path = (
        category_dir / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
    )
    if mask_path.is_file():
        return str(mask_path.relative_to(category_dir.parent))
    return None


def create_splits(
    dataset_root: str | Path,
    category: str,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[dict[str, object]]]:
    """Construit les jeux train / val / test pour une catégorie MVTec.

    Args:
        dataset_root: Racine contenant les catégories (ex. ``data/raw/mvtec_ad``).
        category: Nom de la catégorie (ex. ``"bottle"``).
        val_fraction: Fraction des images normales de ``train/good`` réservée à
            la validation. Doit être dans ``[0, 1[``.
        seed: Graine du mélange, pour un découpage reproductible.

    Returns:
        Un dictionnaire ``{"train": [...], "val": [...], "test": [...]}`` où
        chaque élément est un dict sérialisable (voir :class:`SampleEntry`).

    Raises:
        FileNotFoundError: Si la catégorie ou son dossier ``train/good`` manque.
        ValueError: Si ``val_fraction`` n'est pas dans ``[0, 1[``.

    Example:
        >>> splits = create_splits("data/raw/mvtec_ad", "bottle")  # doctest: +SKIP
        >>> all(e["label"] == 0 for e in splits["train"])  # doctest: +SKIP
        True
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction doit être dans [0, 1[, reçu {val_fraction}.")

    root = Path(dataset_root)
    category_dir = root / category
    train_good_dir = category_dir / "train" / _GOOD
    if not train_good_dir.is_dir():
        raise FileNotFoundError(
            f"Dossier introuvable : {train_good_dir}. "
            "La catégorie MVTec est-elle bien téléchargée et extraite ?"
        )

    # --- Normales d'entraînement -> mélange seedé puis découpe train/val. ---
    train_good = _list_images(train_good_dir)
    shuffled = train_good.copy()
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    def _entry(path: Path, label: int, defect: str, mask: str | None) -> dict:
        return asdict(
            SampleEntry(
                image=str(path.relative_to(root)),
                label=label,
                defect_type=defect,
                mask=mask,
            )
        )

    splits: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    splits["train"] = [_entry(p, LABEL_NORMAL, _GOOD, None) for p in train_files]
    splits["val"] = [_entry(p, LABEL_NORMAL, _GOOD, None) for p in val_files]

    # --- Jeu de test : normales + anormales, avec masques si disponibles. ---
    test_dir = category_dir / "test"
    if test_dir.is_dir():
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            is_anomaly = defect_type != _GOOD
            label = LABEL_ANOMALY if is_anomaly else LABEL_NORMAL
            for img in _list_images(defect_dir):
                mask = (
                    _find_mask(category_dir, defect_type, img) if is_anomaly else None
                )
                splits["test"].append(_entry(img, label, defect_type, mask))

    return splits


def save_splits(splits: dict[str, list[dict[str, object]]], path: str | Path) -> Path:
    """Sauvegarde les splits dans un fichier JSON.

    Args:
        splits: Dictionnaire produit par :func:`create_splits`.
        path: Chemin du fichier JSON de sortie.

    Returns:
        Le chemin du fichier écrit.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, ensure_ascii=False, indent=2)
    return out


def load_splits(path: str | Path) -> dict[str, list[dict[str, object]]]:
    """Recharge des splits depuis un fichier JSON.

    Args:
        path: Chemin du fichier JSON.

    Returns:
        Le dictionnaire des splits.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Fichier de splits introuvable : {p}")
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)
