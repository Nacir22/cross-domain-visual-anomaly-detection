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


def records_from_manifest(
    manifest_path: str | Path,
    image_col: str = "image",
    label_col: str = "label",
    group_col: str = "group",
    mask_col: str | None = None,
) -> list[dict[str, object]]:
    """Construit une liste d'entries à partir d'un manifeste CSV.

    Le manifeste décrit, une ligne par image : le chemin (relatif à la racine du
    dataset), le label (0/1), le groupe (patient, zone...) et éventuellement un
    masque. C'est le format d'échange commun aux domaines santé et aérien.

    Args:
        manifest_path: Chemin du fichier CSV.
        image_col: Colonne du chemin d'image.
        label_col: Colonne du label (0 = normal, 1 = anomalie).
        group_col: Colonne du groupe (clé anti-fuite : patient, zone...).
        mask_col: Colonne du masque (optionnelle).

    Returns:
        Une liste de dicts ``{image, label, mask, group}``.

    Raises:
        FileNotFoundError: Si le manifeste n'existe pas.
    """
    import csv

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifeste introuvable : {path}")

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                {
                    "image": row[image_col],
                    "label": int(row[label_col]),
                    "group": row[group_col],
                    "mask": row[mask_col] if mask_col and row.get(mask_col) else None,
                }
            )
    return records


def create_grouped_splits(
    records: list[dict[str, object]],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, object]]]:
    """Découpe des entries en respectant des GROUPES (anti-fuite).

    Règles (cadre one-class avec séparation par groupe) :

    - Un groupe (patient, zone...) ne peut appartenir qu'à un seul split.
    - Un groupe contenant au moins une anomalie va ENTIÈREMENT dans le test
      (ainsi aucune anomalie ne contamine train/val).
    - Les groupes purement normaux sont répartis train / val / test-normal.

    Args:
        records: Entries ``{image, label, group, mask?}``.
        train_fraction: Part des groupes normaux pour l'entraînement.
        val_fraction: Part des groupes normaux pour la validation.
        seed: Graine du mélange (reproductibilité).

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}``.

    Raises:
        ValueError: Si ``train_fraction + val_fraction >= 1``.
    """
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction doit être < 1.")

    by_group: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        by_group.setdefault(str(rec["group"]), []).append(rec)

    normal_groups, anomaly_groups = [], []
    for group, recs in by_group.items():
        if any(int(r["label"]) == LABEL_ANOMALY for r in recs):
            anomaly_groups.append(group)
        else:
            normal_groups.append(group)

    normal_groups = sorted(normal_groups)
    random.Random(seed).shuffle(normal_groups)
    n_train = int(len(normal_groups) * train_fraction)
    n_val = int(len(normal_groups) * val_fraction)
    train_groups = set(normal_groups[:n_train])
    val_groups = set(normal_groups[n_train : n_train + n_val])

    splits: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    for group, recs in by_group.items():
        if group in anomaly_groups:
            splits["test"].extend(recs)
        elif group in train_groups:
            splits["train"].extend(recs)
        elif group in val_groups:
            splits["val"].extend(recs)
        else:
            splits["test"].extend(recs)
    return splits
     elif group in val_groups:
            splits["val"].extend(recs)
        else:
            splits["test"].extend(recs)  # normales restantes -> test
    return splits
