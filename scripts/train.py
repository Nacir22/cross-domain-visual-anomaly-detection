"""Entraîne un modèle de détection d'anomalies (Phase 2 : autoencodeur).

Le comportement est piloté par la configuration en cascade
``base -> domaine -> modèle -> profil``. On n'entraîne que sur des images
NORMALES (cadre one-class).

Exemples
--------
Profil léger CPU (vérifie le pipeline de bout en bout) ::

    python scripts/train.py --domain industrial --model autoencoder \\
        --profile local_cpu --category bottle

Sur données synthétiques (sans téléchargement) ::

    python scripts/download_data.py --synthetic
    python scripts/train.py --category synthetic --profile local_cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Rend le package importable même sans installation editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_detection.config import load_config  # noqa: E402
from anomaly_detection.constants import MODEL_AUTOENCODER  # noqa: E402
from anomaly_detection.data.datasets import MVTecDataset  # noqa: E402
from anomaly_detection.data.splits import create_splits  # noqa: E402
from anomaly_detection.models.autoencoder import build_autoencoder  # noqa: E402
from anomaly_detection.training.engine import (  # noqa: E402
    save_loss_curve,
    train_autoencoder,
)
from anomaly_detection.utils.device import resolve_device  # noqa: E402
from anomaly_detection.utils.logging import configure_logging  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement (autoencodeur).")
    parser.add_argument("--domain", default="industrial")
    parser.add_argument("--model", default="autoencoder")
    parser.add_argument(
        "--profile", default="local_cpu", choices=["local_cpu", "colab_gpu"]
    )
    parser.add_argument("--category", default=None, help="Catégorie MVTec.")
    parser.add_argument("--data-root", default="data/raw/mvtec_ad")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=None, help="Surcharge la config.")
    return parser.parse_args()


def main() -> int:
    """Point d'entrée CLI. Retourne 0 si succès, 1 sinon."""
    configure_logging()
    args = _parse_args()

    cfg = load_config(args.domain, args.model, args.profile)
    if cfg.model != MODEL_AUTOENCODER:
        print(
            f"[train] Le modèle '{cfg.model}' n'est pas encore supporté "
            "(PatchCore arrive en Phase 4)."
        )
        return 1

    epochs = args.epochs if args.epochs is not None else cfg.epochs
    category = args.category or str(cfg.extra.get("category", "bottle"))
    data_root = Path(args.data_root)
    device = resolve_device(cfg.device)

    # --- Splits partagés : train et val proviennent du MÊME découpage. ---
    splits = create_splits(data_root, category, args.val_fraction, cfg.seed)
    train_ds = MVTecDataset(
        data_root, category, "train", cfg.image_size, splits=splits
    )
    val_ds = MVTecDataset(data_root, category, "val", cfg.image_size, splits=splits)
    print(f"[train] train={len(train_ds)} images | val={len(val_ds)} images")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        # drop_last évite un dernier batch de taille 1 qui ferait échouer la
        # BatchNorm en mode entraînement ("Expected more than 1 value...").
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = build_autoencoder(cfg.extra)
    checkpoint_path = Path(args.output_dir) / f"{args.domain}_{cfg.model}_{category}.pt"

    history = train_autoencoder(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=cfg.learning_rate,
        device=device,
        checkpoint_path=checkpoint_path,
        seed=cfg.seed,
        extra={
            "model": cfg.model,
            "model_config": dict(cfg.extra),
            "image_size": cfg.image_size,
            "domain": args.domain,
            "category": category,
        },
    )

    # --- Artefacts : courbe de perte + historique JSON. ---
    curve = save_loss_curve(
        history, Path("reports/figures") / f"loss_{args.domain}_{category}.png"
    )
    metrics_path = Path("reports/metrics") / f"history_{args.domain}_{category}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(history.as_dict(), indent=2), encoding="utf-8")

    print(f"[train] Checkpoint : {checkpoint_path}")
    print(f"[train] Courbe     : {curve}")
    print(f"[train] Historique : {metrics_path}")
    if history.val:
        print(f"[train] Meilleure perte val : {min(history.val):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
