"""Compare honnêtement l'autoencodeur et PatchCore sous le MÊME protocole.

Les deux modèles sont évalués :
- sur les mêmes splits (val / test) et le même preprocessing ;
- avec un seuil choisi de la même façon (percentile des normales de validation) ;
- sur les mêmes métriques image-level (+ pixel si masques disponibles) ;
- avec une mesure du coût (temps d'inférence, mémoire, paramètres).

Aucune conclusion n'est tirée d'un seul exemple visuel : la comparaison repose
sur les métriques agrégées et le benchmark.

Exemple ::

    python scripts/compare_models.py --category bottle \\
        --ae-checkpoint models/industrial_autoencoder_bottle.pt \\
        --patchcore-checkpoint models/industrial_patchcore_bottle.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_detection.config import load_config  # noqa: E402
from anomaly_detection.data.datasets import MVTecDataset  # noqa: E402
from anomaly_detection.data.splits import create_splits  # noqa: E402
from anomaly_detection.evaluation.benchmark import (  # noqa: E402
    count_parameters,
    measure_inference_time,
    memory_bank_footprint_mb,
)
from anomaly_detection.evaluation.collect import collect_outputs  # noqa: E402
from anomaly_detection.evaluation.metrics import (  # noqa: E402
    image_level_metrics,
    pixel_level_metrics,
)
from anomaly_detection.evaluation.thresholds import (  # noqa: E402
    threshold_from_normal_percentile,
)
from anomaly_detection.inference.pipeline import (
    load_model_from_checkpoint,  # noqa: E402
)
from anomaly_detection.utils.device import resolve_device  # noqa: E402
from anomaly_detection.utils.logging import configure_logging  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparaison autoencodeur vs PatchCore."
    )
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--patchcore-checkpoint", required=True)
    parser.add_argument("--domain", default="industrial")
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", default="data/raw/mvtec_ad")
    parser.add_argument("--profile", default="local_cpu")
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def _evaluate_one(name, checkpoint, cfg, args, device, splits, has_masks):
    """Évalue un modèle et retourne un dict de métriques + coûts."""
    model, meta = load_model_from_checkpoint(checkpoint, device=device)
    image_size = int(meta.get("image_size", cfg.image_size))

    val_ds = MVTecDataset(
        args.data_root, args.category, "val", image_size, splits=splits
    )
    test_ds = MVTecDataset(
        args.data_root, args.category, "test", image_size, splits=splits
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    val = collect_outputs(model, val_loader, device)
    test = collect_outputs(model, test_loader, device)

    threshold = threshold_from_normal_percentile(val["scores"], args.percentile)
    img = image_level_metrics(test["labels"], test["scores"], threshold)

    pixel = None
    if has_masks and test["masks"].sum() > 0:
        pixel_threshold = threshold_from_normal_percentile(
            val["maps"].reshape(-1), args.percentile
        )
        pixel = pixel_level_metrics(test["masks"], test["maps"], pixel_threshold)

    # Benchmark : une batch de test comme charge représentative.
    sample = next(iter(test_loader))["image"]
    ms_per_image = measure_inference_time(model, sample, device)

    return {
        "image_level": img,
        "pixel_level": pixel,
        "inference_ms_per_image": ms_per_image,
        "trainable_params": count_parameters(model),
        "memory_bank_mb": memory_bank_footprint_mb(model),
    }


def _markdown_table(results: dict) -> str:
    """Construit un tableau Markdown comparatif à partir des résultats."""
    header = (
        "| Modèle | AUROC | AUPRC | F1 | FN | FP | Inférence (ms/img) "
        "| Mémoire banque (Mo) |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = ""
    for name, r in results.items():
        m = r["image_level"]
        auroc = "n/a" if m["auroc"] is None else f"{m['auroc']:.3f}"
        auprc = "n/a" if m["auprc"] is None else f"{m['auprc']:.3f}"
        rows += (
            f"| {name} | {auroc} | {auprc} | {m['f1']:.3f} | {m['fn']} | {m['fp']} "
            f"| {r['inference_ms_per_image']:.1f} | {r['memory_bank_mb']:.2f} |\n"
        )
    return header + rows


def main() -> int:
    """Point d'entrée CLI. Retourne 0 si succès, 1 sinon."""
    configure_logging()
    args = _parse_args()

    cfg = load_config(args.domain, "autoencoder", args.profile)
    device = resolve_device(cfg.device)
    has_masks = bool(cfg.extra.get("has_pixel_masks", False))
    splits = create_splits(args.data_root, args.category, seed=cfg.seed)

    results = {
        "autoencoder": _evaluate_one(
            "autoencoder", args.ae_checkpoint, cfg, args, device, splits, has_masks
        ),
        "patchcore": _evaluate_one(
            "patchcore", args.patchcore_checkpoint, cfg, args, device, splits, has_masks
        ),
    }

    metrics_dir = Path(args.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / f"comparison_{args.domain}_{args.category}.json"
    md_path = metrics_dir / f"comparison_{args.domain}_{args.category}.md"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    table = _markdown_table(results)
    md = (
        f"# Comparaison — {args.domain} / {args.category}\n\n"
        f"Protocole identique (mêmes splits, preprocessing, seuil percentile "
        f"{args.percentile:.0f}).\n\n"
        f"{table}\n"
        "> Comparaison fondée sur des métriques agrégées, pas sur un exemple isolé.\n"
    )
    md_path.write_text(md, encoding="utf-8")

    print(table)
    print(f"[compare] JSON : {json_path}")
    print(f"[compare] Tableau : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
