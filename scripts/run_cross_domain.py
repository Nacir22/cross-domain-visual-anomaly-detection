"""Exécute le MÊME pipeline sur les trois domaines et compare les résultats.

Démonstration de transférabilité : industrie (MVTec), santé (façon RSNA) et
aérien (façon xView2) passent par le même autoencodeur, le même entraînement et
la même évaluation. Seules changent la construction des splits (règles par
patient / par zone) et une fine classe dataset par domaine.

Par défaut, le script génère des mini-datasets SYNTHÉTIQUES (sans téléchargement)
afin d'illustrer le pipeline de bout en bout. Les valeurs obtenues sur ces
données factices ne mesurent PAS la performance réelle.

Exemple ::

    python scripts/run_cross_domain.py --epochs 2 --image-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_detection.data.datasets import (  # noqa: E402
    AerialDataset,
    MedicalDataset,
    MVTecDataset,
)
from anomaly_detection.data.domains import (  # noqa: E402
    build_aerial_splits,
    build_medical_splits,
)
from anomaly_detection.data.splits import create_splits  # noqa: E402
from anomaly_detection.data.synthetic import (  # noqa: E402
    generate_synthetic_aerial,
    generate_synthetic_medical,
    generate_synthetic_mvtec,
)
from anomaly_detection.evaluation.collect import collect_outputs  # noqa: E402
from anomaly_detection.evaluation.metrics import image_level_metrics  # noqa: E402
from anomaly_detection.evaluation.thresholds import (  # noqa: E402
    threshold_from_normal_percentile,
)
from anomaly_detection.models.autoencoder import build_autoencoder  # noqa: E402
from anomaly_detection.training.engine import train_autoencoder  # noqa: E402
from anomaly_detection.utils.device import resolve_device  # noqa: E402
from anomaly_detection.utils.logging import configure_logging  # noqa: E402


def _prepare_domain(domain: str, root: Path, image_size: int, seed: int):
    """Génère les données synthétiques et retourne (train, val, test, has_masks)."""
    if domain == "industrial":
        if not (root / "synthetic" / "train" / "good").is_dir():
            generate_synthetic_mvtec(root, category="synthetic", n_train_good=12)
        splits = create_splits(root, "synthetic", seed=seed)
        make = lambda s: MVTecDataset(  # noqa: E731
            root, "synthetic", s, image_size, splits=splits
        )
        return make("train"), make("val"), make("test"), True

    if domain == "medical":
        if not (root / "manifest.csv").is_file():
            generate_synthetic_medical(root, seed=seed)
        splits = build_medical_splits(root, seed=seed)
        make = lambda s: MedicalDataset(root, s, splits, image_size)  # noqa: E731
        return make("train"), make("val"), make("test"), False

    if domain == "aerial":
        if not (root / "manifest.csv").is_file():
            generate_synthetic_aerial(root, seed=seed)
        splits = build_aerial_splits(root, seed=seed)
        make = lambda s: AerialDataset(root, s, splits, image_size)  # noqa: E731
        return make("train"), make("val"), make("test"), True

    raise ValueError(f"Domaine inconnu : {domain}")


def _run_domain(domain, root, image_size, epochs, seed, device):
    """Entraîne et évalue l'autoencodeur sur un domaine. Retourne un dict."""
    train_ds, val_ds, test_ds, has_masks = _prepare_domain(
        domain, root, image_size, seed
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    test_loader = DataLoader(test_ds, batch_size=4)

    model = build_autoencoder({"base_channels": 16, "latent_dim": 32})
    ckpt = Path("models") / f"crossdomain_{domain}.pt"
    train_autoencoder(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=1e-3,
        device=device,
        checkpoint_path=ckpt,
        seed=seed,
        extra={
            "model": "autoencoder",
            "model_config": {"base_channels": 16, "latent_dim": 32},
        },
    )

    val = collect_outputs(model, val_loader, device)
    test = collect_outputs(model, test_loader, device)
    threshold = threshold_from_normal_percentile(val["scores"], 95.0)
    metrics = image_level_metrics(test["labels"], test["scores"], threshold)

    return {
        "domain": domain,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "has_pixel_masks": has_masks,
        "split_rule": {
            "industrial": "aléatoire (images normales)",
            "medical": "par patient",
            "aerial": "par zone géographique",
        }[domain],
        "auroc": metrics["auroc"],
        "f1": metrics["f1"],
        "fn": metrics["fn"],
        "fp": metrics["fp"],
    }


def _markdown(results: list[dict]) -> str:
    header = (
        "| Domaine | Règle de split | Masques pixel | n_test | AUROC | F1 | FN | FP |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = ""
    for r in results:
        auroc = "n/a" if r["auroc"] is None else f"{r['auroc']:.3f}"
        rows += (
            f"| {r['domain']} | {r['split_rule']} | {r['has_pixel_masks']} "
            f"| {r['n_test']} | {auroc} | {r['f1']:.3f} | {r['fn']} | {r['fp']} |\n"
        )
    return header + rows


def main() -> int:
    """Point d'entrée CLI."""
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Pipeline cross-domaines (synthétique)."
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device("cpu")

    roots = {
        "industrial": Path("data/raw/mvtec_ad"),
        "medical": Path("data/raw/rsna_synth"),
        "aerial": Path("data/raw/xview2_synth"),
    }
    results = [
        _run_domain(domain, root, args.image_size, args.epochs, args.seed, device)
        for domain, root in roots.items()
    ]

    out_dir = Path("reports/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cross_domain.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    table = _markdown(results)
    (out_dir / "cross_domain.md").write_text(
        "# Comparaison inter-domaines (données synthétiques)\n\n"
        "Même pipeline sur les trois domaines. "
        "Valeurs illustratives.\n\n" + table,
        encoding="utf-8",
    )
    print(table)
    print(f"[cross-domain] Résultats : {out_dir / 'cross_domain.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
