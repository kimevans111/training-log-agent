"""Metric summary utilities for parsed training logs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def generate_metric_summary(parsed_log: Dict[str, Any]) -> Dict[str, Any]:
    """Generate compact best/final/trend summaries from parsed log output."""

    series = sorted(parsed_log.get("metrics_series", []), key=lambda item: item.get("epoch", 10**9))
    final_record = series[-1] if series else {}
    final_epoch = parsed_log.get("final_epoch") or final_record.get("epoch")

    best_miou, best_miou_epoch = _best_from_series_or_top(parsed_log, series, "miou")
    best_f1, best_f1_epoch = _best_from_series_or_top(parsed_log, series, "f1")

    window_size = max(1, int(np.ceil(len(series) * 0.1))) if series else 0
    recent = series[-window_size:] if window_size else []

    class_gap = _class_gap(series)
    loss_gap = _loss_gap(final_record)
    trend_metrics = {
        "last_window_size": window_size,
        "last_miou_mean": _mean(recent, "miou"),
        "last_f1_mean": _mean(recent, "f1"),
        "miou_std_recent": _std(recent, "miou"),
        "f1_std_recent": _std(recent, "f1"),
        "miou_std_all": _std(series, "miou"),
        "f1_std_all": _std(series, "f1"),
        "miou_delta_recent": _delta(recent, "miou"),
        "f1_delta_recent": _delta(recent, "f1"),
        "train_loss_delta": _delta(series, "train_loss"),
        "val_loss_delta": _delta(series, "val_loss"),
        "metrics_still_improving": _is_still_improving(recent),
    }

    final_miou = final_record.get("miou")
    final_f1 = final_record.get("f1")
    early_degradation = {
        "best_miou_far_before_final": _far_before(best_miou_epoch, final_epoch),
        "best_f1_far_before_final": _far_before(best_f1_epoch, final_epoch),
        "final_miou_drop_from_best": _drop(best_miou, final_miou),
        "final_f1_drop_from_best": _drop(best_f1, final_f1),
    }
    trend_metrics["training_sufficient"] = not trend_metrics["metrics_still_improving"]

    headline = _build_headline(parsed_log, best_miou, best_miou_epoch, best_f1, best_f1_epoch, final_epoch)
    return {
        "headline": headline,
        "best_metrics": {
            "best_miou": best_miou,
            "best_miou_epoch": best_miou_epoch,
            "best_f1": best_f1,
            "best_f1_epoch": best_f1_epoch,
        },
        "final_metrics": {
            "final_epoch": final_epoch,
            "final_miou": final_miou,
            "final_f1": final_f1,
            "final_train_loss": final_record.get("train_loss"),
            "final_val_loss": final_record.get("val_loss"),
            "final_precision": final_record.get("precision"),
            "final_recall": final_record.get("recall"),
            "final_leaf_iou": final_record.get("leaf_iou"),
            "final_stem_iou": final_record.get("stem_iou"),
        },
        "trend_metrics": trend_metrics,
        "class_gap": class_gap,
        "loss_gap": loss_gap,
        "early_degradation": early_degradation,
    }


def _best_from_series_or_top(parsed_log: Dict[str, Any], series: List[Dict[str, Any]], key: str) -> tuple[Optional[float], Optional[int]]:
    top_value = parsed_log.get(f"best_{key}")
    top_epoch = parsed_log.get(f"best_{key}_epoch")
    if top_value is not None:
        return top_value, top_epoch

    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    for record in series:
        value = record.get(key)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_epoch = record.get("epoch")
    return best_value, best_epoch


def _values(records: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(record[key]) for record in records if record.get(key) is not None]


def _mean(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = _values(records, key)
    return round(float(np.mean(values)), 6) if values else None


def _std(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = _values(records, key)
    return round(float(np.std(values)), 6) if len(values) >= 2 else None


def _delta(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = _values(records, key)
    if len(values) < 2:
        return None
    return round(values[-1] - values[0], 6)


def _class_gap(series: List[Dict[str, Any]]) -> Dict[str, Any]:
    leaf = _mean(series, "leaf_iou")
    stem = _mean(series, "stem_iou")
    gap = round(leaf - stem, 6) if leaf is not None and stem is not None else None
    status = "unknown"
    if gap is not None:
        status = "large_gap" if gap > 0.15 else "balanced"
    return {"leaf_iou_mean": leaf, "stem_iou_mean": stem, "leaf_stem_iou_gap": gap, "status": status}


def _loss_gap(final_record: Dict[str, Any]) -> Dict[str, Any]:
    train_loss = final_record.get("train_loss")
    val_loss = final_record.get("val_loss")
    if train_loss is None or val_loss is None:
        return {"final_train_loss": train_loss, "final_val_loss": val_loss, "gap": None, "relative_gap": None, "status": "unknown"}
    gap = round(float(val_loss) - float(train_loss), 6)
    relative = round(gap / max(abs(float(train_loss)), 1e-8), 6)
    status = "high_val_gap" if gap > 0.1 and relative > 0.2 else "normal"
    return {"final_train_loss": train_loss, "final_val_loss": val_loss, "gap": gap, "relative_gap": relative, "status": status}


def _is_still_improving(recent: List[Dict[str, Any]]) -> bool:
    miou_delta = _delta(recent, "miou") or 0.0
    f1_delta = _delta(recent, "f1") or 0.0
    return miou_delta > 0.02 or f1_delta > 0.02


def _far_before(best_epoch: Optional[int], final_epoch: Optional[int]) -> bool:
    if best_epoch is None or final_epoch is None or final_epoch <= 0:
        return False
    return best_epoch < final_epoch * 0.85


def _drop(best_value: Optional[float], final_value: Optional[float]) -> Optional[float]:
    if best_value is None or final_value is None:
        return None
    return round(best_value - final_value, 6)


def _fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _build_headline(
    parsed_log: Dict[str, Any],
    best_miou: Optional[float],
    best_miou_epoch: Optional[int],
    best_f1: Optional[float],
    best_f1_epoch: Optional[int],
    final_epoch: Optional[int],
) -> str:
    return (
        f"Parsed {parsed_log.get('num_epochs', 0)} epochs up to epoch {final_epoch}. "
        f"Best mIoU={_fmt(best_miou)} at epoch {best_miou_epoch}, "
        f"best F1={_fmt(best_f1)} at epoch {best_f1_epoch}."
    )

