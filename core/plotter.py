"""Matplotlib plot generation for training metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_plots(parsed_log: Dict[str, Any], output_dir: Path | str = Path("reports/figures")) -> List[Path]:
    """Generate metric curve images and return the created paths.

    Plot failures are swallowed per-figure so the analysis flow can still
    complete when a metric is absent or matplotlib hits an environment issue.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    series = sorted(parsed_log.get("metrics_series", []), key=lambda item: item.get("epoch", 10**9))
    if not series:
        return []

    figures: List[Path] = []
    figure_specs = [
        ("loss_curve.png", ["train_loss", "val_loss", "loss"], "Loss", None),
        ("miou_curve.png", ["miou"], "mIoU", parsed_log.get("best_miou_epoch")),
        ("f1_curve.png", ["f1"], "F1-score", parsed_log.get("best_f1_epoch")),
        ("class_iou_curve.png", ["leaf_iou", "stem_iou"], "Class-wise IoU", None),
        ("precision_recall_curve.png", ["precision", "recall"], "Precision / Recall", None),
    ]

    for filename, keys, title, best_epoch in figure_specs:
        try:
            path = _plot_metric_group(series, keys, title, out_dir / filename, best_epoch)
            if path is not None:
                figures.append(path)
        except Exception:
            continue
    return figures


def _plot_metric_group(
    series: List[Dict[str, Any]],
    keys: List[str],
    title: str,
    output_path: Path,
    best_epoch: Optional[int],
) -> Optional[Path]:
    epochs = [record.get("epoch") for record in series]
    if not any(epoch is not None for epoch in epochs):
        epochs = list(range(1, len(series) + 1))

    available_keys = [key for key in keys if any(record.get(key) is not None for record in series)]
    if not available_keys:
        return None

    plt.figure(figsize=(8, 4.8), dpi=140)
    for key in available_keys:
        xs: List[int] = []
        ys: List[float] = []
        for index, record in enumerate(series):
            value = record.get(key)
            if value is None:
                continue
            epoch = record.get("epoch") if record.get("epoch") is not None else index + 1
            xs.append(int(epoch))
            ys.append(float(value))
        if xs:
            plt.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.6, label=key)

    if best_epoch is not None:
        plt.axvline(int(best_epoch), color="#d62728", linestyle="--", linewidth=1.2, label=f"best epoch {best_epoch}")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

