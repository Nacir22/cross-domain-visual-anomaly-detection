"""Évalue un modèle entraîné : métriques, seuils et interprétabilité.

Règle d'or : le SEUIL principal est choisi sur la VALIDATION (images normales),
jamais sur le test. Pour illustrer les stratégies F1 et coût (qui exigent des
anomalies étiquetées), on découpe le test en deux parts DISJOINTES : une petite
part *calibration* (choix du seuil) et une part *holdout* (mesure finale).

Exemple (données synthétiques) ::

    python scripts/download_data.py --synthetic
    python scripts/train.py --category synthetic --epochs 2
    python scripts/evaluate.py --category synthetic \\
        --checkpoint models/industrial_autoencoder_synthetic.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_detection.config import load_config  # noqa: E402
from anomaly_detection.data.datasets import MVTecDataset  # noqa: E402
from anomaly_detection.data.splits import create_splits  # noqa: E402
from anomaly_detection.evaluation.collect import collect_outputs  # noqa: E402
from anomaly_detection.evaluation.metrics import (  # noqa: E402
    image_level_metrics,
    pixel_level_metrics,
)
from anomaly_detection.evaluation.thresholds import (  # noqa: E402
    threshold_cost_based,
    threshold_from_normal_percentile,
    threshold_max_f1,
)
from anomaly_detection.evaluation.visualization import (  # noqa: E402
    overlay_heatmap,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    save_examples_gallery,
)
from anomaly_detection.inference.pipeline import load_model_from_checkpoint  # noqa: E402
from anomaly_detection.inference.postprocessing import min_max_normalize  # noqa: E402
from anomaly_detection.utils.device import resolve_device  # noqa: E402
from anomaly_detection.utils.logging import configure_logging  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évaluation d'un détecteur d'anomalies.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domain", default="industrial")
    parser.add_argument("--model", default="autoencoder")
    parser.add_argument("--profile", default="local_cpu")
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-root", default="data/raw/mvtec_ad")
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--calib-fraction", type=float, default=0.3)
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


def _stratified_calibration_split(labels: np.ndarray, calib_fraction: float, seed: int):
    """Découpe des indices en (calibration, holdout) de façon stratifiée."""
    indices = np.arange(len(labels))
    if len(np.unique(labels)) < 2:
        # Pas de stratification possible : tout en holdout, pas de calibration.
        return np.array([], dtype=int), indices
    calib_idx, hold_idx = train_test_split(
        indices, train_size=calib_fraction, stratify=labels, random_state=seed
    )
    return calib_idx, hold_idx


def main() -> int:  # noqa: PLR0915 - orchestration linéaire, volontairement plate
    """Point d'entrée CLI. Retourne 0 si succès, 1 sinon."""
    configure_logging()
    args = _parse_args()

    cfg = load_config(args.domain, args.model, args.profile)
    device = resolve_device(cfg.device)
    data_root = Path(args.data_root)
    figures_dir = Path(args.output_dir) / "figures"
    metrics_dir = Path(args.output_dir) / "metrics"

    model, meta = load_model_from_checkpoint(args.checkpoint, device=device)
    image_size = int(meta.get("image_size", cfg.image_size))

    splits = create_splits(data_root, args.category, seed=cfg.seed)
    val_ds = MVTecDataset(data_root, args.category, "val", image_size, splits=splits)
    test_ds = MVTecDataset(data_root, args.category, "test", image_size, splits=splits)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    val = collect_outputs(model, val_loader, device)
    test = collect_outputs(model, test_loader, device)
    labels, scores = test["labels"], test["scores"]

    # --- Seuil principal : percentile des NORMALES de validation (sans fuite). ---
    thr_percentile = threshold_from_normal_percentile(val["scores"], args.percentile)
    pixel_threshold = threshold_from_normal_percentile(
        val["maps"].reshape(-1), args.percentile
    )

    # --- Seuils F1 / coût : sur une part de calibration disjointe du holdout. ---
    calib_idx, hold_idx = _stratified_calibration_split(
        labels, args.calib_fraction, cfg.seed
    )
    thresholds = {"percentile_val": thr_percentile}
    if calib_idx.size and len(np.unique(labels[calib_idx])) == 2:
        thresholds["f1_calib"] = threshold_max_f1(labels[calib_idx], scores[calib_idx])
        thresholds["cost_calib"] = threshold_cost_based(
            labels[calib_idx], scores[calib_idx]
        )

    # --- Métriques image-level sur le holdout, pour chaque stratégie de seuil. ---
    image_results = {
        name: image_level_metrics(labels[hold_idx], scores[hold_idx], thr)
        for name, thr in thresholds.items()
    }

    # --- Métriques pixel-level (uniquement si masques fiables). ---
    pixel_results = None
    if bool(cfg.extra.get("has_pixel_masks", False)) and test["masks"].sum() > 0:
        pixel_results = pixel_level_metrics(
            test["masks"][hold_idx], test["maps"][hold_idx], pixel_threshold
        )

    # --- Figures : ROC, PR, matrice de confusion (seuil percentile). ---
    figures_dir.mkdir(parents=True, exist_ok=True)
    if len(np.unique(labels[hold_idx])) == 2:
        plot_roc_curve(labels[hold_idx], scores[hold_idx], figures_dir / "roc.png")
        plot_pr_curve(labels[hold_idx], scores[hold_idx], figures_dir / "pr.png")
    m = image_results["percentile_val"]
    plot_confusion_matrix(
        m["tn"], m["fp"], m["fn"], m["tp"], figures_dir / "confusion.png"
    )

    # --- Interprétabilité : heatmaps de quelques faux positifs / négatifs. ---
    preds = (scores >= thr_percentile).astype(int)
    fp_idx = np.where((preds == 1) & (labels == 0))[0][:4]
    fn_idx = np.where((preds == 0) & (labels == 1))[0][:4]
    for tag, idxs in (("fp", fp_idx), ("fn", fn_idx)):
        examples = []
        for k, i in enumerate(idxs):
            heat = min_max_normalize(test["maps"][i])
            overlay_heatmap(
                test["paths"][i], heat, figures_dir / f"{tag}_{k}_heatmap.png"
            )
            examples.append(
                {"image": test["paths"][i], "caption": f"score={scores[i]:.3f}"}
            )
        save_examples_gallery(
            examples, figures_dir / f"gallery_{tag}.png", title=tag.upper()
        )

    # --- Sauvegarde des résultats AVEC leur configuration (traçabilité). ---
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": cfg.model_dump(),
        "checkpoint": str(args.checkpoint),
        "category": args.category,
        "thresholds": thresholds,
        "image_level": image_results,
        "pixel_level": pixel_results,
        "n_val": int(val["scores"].size),
        "n_test": int(scores.size),
        "n_calibration": int(calib_idx.size),
        "n_holdout": int(hold_idx.size),
    }
    out = metrics_dir / f"eval_{args.domain}_{args.category}.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"[evaluate] Résultats : {out}")
    main_metrics = image_results["percentile_val"]
    print(
        "[evaluate] (seuil percentile) "
        f"AUROC={main_metrics['auroc']} F1={main_metrics['f1']:.3f} "
        f"FN={main_metrics['fn']} FP={main_metrics['fp']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
