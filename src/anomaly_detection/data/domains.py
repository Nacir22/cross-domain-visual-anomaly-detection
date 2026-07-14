"""Constructeurs de splits par domaine (santé, aérien).

Ces fonctions ne font que combiner deux briques génériques :
``records_from_manifest`` (lire le manifeste CSV) et ``create_grouped_splits``
(découper sans fuite de groupe). C'est ici que se matérialise la règle propre à
chaque domaine :

- **santé** : groupe = ``patient_id`` -> aucune image d'un patient ne franchit
  deux splits ;
- **aérien** : groupe = ``zone`` géographique -> aucune tuile d'une zone ne
  franchit deux splits.

Le reste du pipeline (modèles, entraînement, évaluation) est strictement
identique à celui du domaine industriel : c'est la démonstration de
transférabilité.
"""

from __future__ import annotations

from pathlib import Path

from anomaly_detection.data.splits import create_grouped_splits, records_from_manifest


def build_medical_splits(
    root: str | Path,
    manifest: str = "manifest.csv",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, object]]]:
    """Construit les splits médicaux, séparés PAR PATIENT.

    Args:
        root: Racine du dataset médical (contient le manifeste et ``images/``).
        manifest: Nom du fichier manifeste CSV.
        train_fraction: Part des patients normaux pour l'entraînement.
        val_fraction: Part des patients normaux pour la validation.
        seed: Graine du découpage.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` sans fuite entre patients.
    """
    records = records_from_manifest(
        Path(root) / manifest, group_col="group", mask_col=None
    )
    return create_grouped_splits(records, train_fraction, val_fraction, seed)


def build_aerial_splits(
    root: str | Path,
    manifest: str = "manifest.csv",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, object]]]:
    """Construit les splits aériens, séparés PAR ZONE géographique.

    Args:
        root: Racine du dataset aérien (contient le manifeste, ``images/``,
            ``masks/``).
        manifest: Nom du fichier manifeste CSV.
        train_fraction: Part des zones normales pour l'entraînement.
        val_fraction: Part des zones normales pour la validation.
        seed: Graine du découpage.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` sans fuite entre zones.
    """
    records = records_from_manifest(
        Path(root) / manifest, group_col="group", mask_col="mask"
    )
    return create_grouped_splits(records, train_fraction, val_fraction, seed)
