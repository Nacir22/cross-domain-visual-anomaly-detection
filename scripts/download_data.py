"""Préparation des données MVTec AD.

Ce script NE télécharge RIEN automatiquement par défaut. MVTec AD est diffusé
sous licence **CC BY-NC-SA 4.0 (usage non commercial)** et son téléchargement
passe par le site officiel. On ne contourne jamais ces conditions.

Trois usages :

1. Afficher les instructions et la licence (par défaut) ::

       python scripts/download_data.py

2. Générer un mini-dataset SYNTHÉTIQUE (pour tests / CI, sans téléchargement) ::

       python scripts/download_data.py --synthetic --root data/raw/mvtec_ad

3. Vérifier qu'une catégorie téléchargée a la bonne structure ::

       python scripts/download_data.py --verify --root data/raw/mvtec_ad --category bottle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Rend le package importable même sans installation editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_MVTEC_URL = "https://www.mvtec.com/company/research/datasets/mvtec-ad"

_INSTRUCTIONS = f"""\
==================================================================
 MVTec AD - instructions de telechargement
==================================================================
Licence : CC BY-NC-SA 4.0 (USAGE NON COMMERCIAL). Voir data/README.md.

1. Rendez-vous sur : {_MVTEC_URL}
2. Acceptez les conditions et telechargez l'archive (mvtec_ad.tar.xz).
3. Extrayez-la dans : data/raw/mvtec_ad/
   Vous devez obtenir des dossiers comme :
       data/raw/mvtec_ad/bottle/train/good/*.png
       data/raw/mvtec_ad/bottle/test/*/*.png
       data/raw/mvtec_ad/bottle/ground_truth/*/*_mask.png
4. Verifiez avec :
       python scripts/download_data.py --verify --category bottle

Pour developper SANS le vrai dataset (tests, CI), generez une categorie
synthetique :
       python scripts/download_data.py --synthetic
==================================================================
"""


def verify_category(root: Path, category: str) -> bool:
    """Vérifie qu'une catégorie possède la structure MVTec attendue.

    Args:
        root: Racine des catégories (ex. ``data/raw/mvtec_ad``).
        category: Nom de la catégorie à vérifier.

    Returns:
        ``True`` si la structure minimale est présente, ``False`` sinon.
    """
    category_dir = root / category
    train_good = category_dir / "train" / "good"
    test_dir = category_dir / "test"

    ok = True
    if not train_good.is_dir() or not any(train_good.glob("*.png")):
        print(f"[FAIL] {train_good} manquant ou vide.")
        ok = False
    else:
        print(f"[ OK ] {train_good} ({len(list(train_good.glob('*.png')))} images)")

    if not test_dir.is_dir():
        print(f"[FAIL] {test_dir} manquant.")
        ok = False
    else:
        subdirs = [p.name for p in test_dir.iterdir() if p.is_dir()]
        print(f"[ OK ] {test_dir} (sous-dossiers : {subdirs})")

    return ok


def main() -> int:
    """Point d'entrée CLI. Retourne 0 si succès, 1 sinon."""
    parser = argparse.ArgumentParser(description="Préparation des données MVTec AD.")
    parser.add_argument(
        "--root",
        default="data/raw/mvtec_ad",
        help="Racine des catégories MVTec (défaut : data/raw/mvtec_ad).",
    )
    parser.add_argument("--category", default="bottle", help="Catégorie à vérifier.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Génère une petite catégorie synthétique (sans téléchargement).",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Vérifie la structure d'une catégorie."
    )
    parser.add_argument(
        "--make-splits",
        action="store_true",
        help="Crée et sauvegarde les splits train/val/test dans data/interim/.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2, help="Fraction de validation."
    )
    parser.add_argument("--seed", type=int, default=42, help="Graine du découpage.")
    args = parser.parse_args()

    root = Path(args.root)

    if args.synthetic:
        from anomaly_detection.data.synthetic import generate_synthetic_mvtec

        path = generate_synthetic_mvtec(root, category="synthetic")
        print(f"[ OK ] Catégorie synthétique créée : {path}")
        return 0

    if args.make_splits:
        from anomaly_detection.data.splits import create_splits, save_splits

        splits = create_splits(root, args.category, args.val_fraction, args.seed)
        out = save_splits(
            splits, Path("data/interim") / f"{args.category}_splits.json"
        )
        counts = {k: len(v) for k, v in splits.items()}
        print(f"[ OK ] Splits sauvegardés : {out}  (tailles : {counts})")
        return 0

    if args.verify:
        return 0 if verify_category(root, args.category) else 1

    print(_INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
