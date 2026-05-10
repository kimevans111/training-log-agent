"""Rule-based diagnostics for training dynamics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def diagnose_training(parsed_log: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """Diagnose common deep learning training issues from metrics."""

    series = sorted(parsed_log.get("metrics_series", []), key=lambda item: item.get("epoch", 10**9))
    diagnoses: List[Dict[str, str]] = []

    _detect_overfitting(series, summary, diagnoses)
    _detect_underfitting(series, summary, diagnoses)
    _detect_oscillation(series, summary, diagnoses)
    _detect_class_imbalance(summary, diagnoses)
    _detect_learning_rate_issues(series, summary, diagnoses)
    _detect_metric_conflicts(summary, diagnoses)

    if not diagnoses:
        diagnoses.append(
            {
                "type": "healthy_training",
                "severity": "low",
                "evidence": "No strong overfitting, underfitting, oscillation, or class imbalance signal was detected.",
                "suggestion": "Use the current setup as a baseline and make one controlled change per next experiment.",
            }
        )
    return {"diagnoses": diagnoses}


def _detect_overfitting(series: List[Dict[str, Any]], summary: Dict[str, Any], diagnoses: List[Dict[str, str]]) -> None:
    trend = summary.get("trend_metrics", {})
    early = summary.get("early_degradation", {})
    loss_gap = summary.get("loss_gap", {})

    train_loss_delta = trend.get("train_loss_delta")
    val_loss_delta = trend.get("val_loss_delta")
    miou_drop = early.get("final_miou_drop_from_best") or 0.0
    f1_drop = early.get("final_f1_drop_from_best") or 0.0

    loss_signal = train_loss_delta is not None and val_loss_delta is not None and train_loss_delta < -0.05 and val_loss_delta > 0.03
    early_best_signal = (
        (early.get("best_miou_far_before_final") or early.get("best_f1_far_before_final"))
        and (miou_drop > 0.02 or f1_drop > 0.03)
    )
    val_gap_signal = loss_gap.get("status") == "high_val_gap"

    if loss_signal or early_best_signal or val_gap_signal:
        diagnoses.append(
            {
                "type": "overfitting",
                "severity": "medium" if early_best_signal or val_gap_signal else "low",
                "evidence": (
                    f"train_loss_delta={train_loss_delta}, val_loss_delta={val_loss_delta}, "
                    f"final_mIoU_drop={miou_drop}, final_F1_drop={f1_drop}, loss_gap={loss_gap.get('gap')}."
                ),
                "suggestion": "Use early stopping around the best epoch, strengthen augmentation, add weight decay/dropout, and validate the train/val split.",
            }
        )


def _detect_underfitting(series: List[Dict[str, Any]], summary: Dict[str, Any], diagnoses: List[Dict[str, str]]) -> None:
    final_metrics = summary.get("final_metrics", {})
    trend = summary.get("trend_metrics", {})
    final_miou = final_metrics.get("final_miou")
    final_f1 = final_metrics.get("final_f1")
    train_loss = final_metrics.get("final_train_loss")
    val_loss = final_metrics.get("final_val_loss")

    low_metrics = ((final_miou is not None and final_miou < 0.5) or (final_f1 is not None and final_f1 < 0.5))
    high_losses = train_loss is not None and val_loss is not None and train_loss > 0.8 and val_loss > 0.8
    still_improving = bool(trend.get("metrics_still_improving"))

    if low_metrics and high_losses:
        diagnoses.append(
            {
                "type": "underfitting",
                "severity": "medium",
                "evidence": f"Final mIoU={final_miou}, final F1={final_f1}, train_loss={train_loss}, val_loss={val_loss}.",
                "suggestion": "Increase model capacity, train longer, check point-cloud normalization, and review whether augmentation is too aggressive.",
            }
        )
    elif still_improving:
        diagnoses.append(
            {
                "type": "insufficient_training",
                "severity": "low",
                "evidence": f"Recent mIoU/F1 deltas are still positive: mIoU={trend.get('miou_delta_recent')}, F1={trend.get('f1_delta_recent')}.",
                "suggestion": "Extend training or use a scheduler that keeps a small learning rate tail for late-stage refinement.",
            }
        )


def _detect_oscillation(series: List[Dict[str, Any]], summary: Dict[str, Any], diagnoses: List[Dict[str, str]]) -> None:
    trend = summary.get("trend_metrics", {})
    miou_std = trend.get("miou_std_recent") or 0.0
    f1_std = trend.get("f1_std_recent") or 0.0

    if miou_std > 0.035 or f1_std > 0.04:
        diagnoses.append(
            {
                "type": "training_oscillation",
                "severity": "medium",
                "evidence": f"Recent metric std is high: mIoU std={miou_std}, F1 std={f1_std}.",
                "suggestion": "Lower learning rate, add warmup, consider larger batch size, gradient clipping, or EMA weights.",
            }
        )

    if f1_std > 0.03 and miou_std > 0 and f1_std > miou_std * 1.5:
        diagnoses.append(
            {
                "type": "f1_instability",
                "severity": "medium",
                "evidence": f"F1 fluctuates more than mIoU in the recent window: F1 std={f1_std}, mIoU std={miou_std}.",
                "suggestion": "Inspect instance boundaries, post-processing thresholds, clustering radius, and minority stem predictions.",
            }
        )


def _detect_class_imbalance(summary: Dict[str, Any], diagnoses: List[Dict[str, str]]) -> None:
    class_gap = summary.get("class_gap", {})
    gap = class_gap.get("leaf_stem_iou_gap")
    if gap is not None and gap > 0.15:
        diagnoses.append(
            {
                "type": "class_imbalance",
                "severity": "high" if gap > 0.25 else "medium",
                "evidence": f"Mean leaf IoU={class_gap.get('leaf_iou_mean')}, stem IoU={class_gap.get('stem_iou_mean')}, gap={gap}.",
                "suggestion": "Use class-balanced sampling, weighted/focal loss, stem-focused augmentation, and boundary-aware losses for thin structures.",
            }
        )

    final_metrics = summary.get("final_metrics", {})
    precision = final_metrics.get("final_precision")
    recall = final_metrics.get("final_recall")
    if precision is None or recall is None:
        return
    diff = precision - recall
    if diff > 0.08:
        diagnoses.append(
            {
                "type": "low_recall",
                "severity": "medium",
                "evidence": f"Precision={precision}, recall={recall}; recall is lower by {diff:.4f}.",
                "suggestion": "Reduce overly strict thresholds, increase positive/stem samples, and inspect missed thin stem or boundary points.",
            }
        )
    elif diff < -0.08:
        diagnoses.append(
            {
                "type": "low_precision",
                "severity": "medium",
                "evidence": f"Precision={precision}, recall={recall}; precision is lower by {abs(diff):.4f}.",
                "suggestion": "Tighten prediction thresholds, improve hard-negative sampling, and tune clustering/post-processing to reduce false positives.",
            }
        )


def _detect_learning_rate_issues(series: List[Dict[str, Any]], summary: Dict[str, Any], diagnoses: List[Dict[str, str]]) -> None:
    trend = summary.get("trend_metrics", {})
    miou_std = trend.get("miou_std_recent") or 0.0
    f1_std = trend.get("f1_std_recent") or 0.0
    train_loss_delta = trend.get("train_loss_delta")
    miou_delta_recent = trend.get("miou_delta_recent") or 0.0
    existing_types = {d["type"] for d in diagnoses}

    if (miou_std > 0.035 or f1_std > 0.04) and "training_oscillation" not in existing_types:
        diagnoses.append(
            {
                "type": "possible_lr_too_high",
                "severity": "medium",
                "evidence": f"Recent metric fluctuation suggests optimizer steps may be too aggressive: mIoU std={miou_std}, F1 std={f1_std}.",
                "suggestion": "Try reducing lr by 2x-5x, enabling warmup for 5-10% epochs, and adding gradient clipping.",
            }
        )
    elif train_loss_delta is not None and train_loss_delta < -0.02 and abs(miou_delta_recent) < 0.005:
        diagnoses.append(
            {
                "type": "possible_lr_too_small_or_plateau",
                "severity": "low",
                "evidence": f"Loss keeps decreasing but recent mIoU barely changes: train_loss_delta={train_loss_delta}, recent_mIoU_delta={miou_delta_recent}.",
                "suggestion": "Check scheduler plateau behavior; consider a slightly higher initial lr or cosine/OneCycle schedule for better exploration.",
            }
        )

    miou_values = [record.get("miou") for record in series if record.get("miou") is not None]
    if len(miou_values) >= 8:
        early_gain = miou_values[min(4, len(miou_values) - 1)] - miou_values[0]
        late_gain = miou_values[-1] - miou_values[len(miou_values) // 2]
        if early_gain > 0.15 and late_gain < 0.02:
            diagnoses.append(
                {
                    "type": "metric_plateau",
                    "severity": "low",
                    "evidence": f"Early mIoU gain={early_gain:.4f}, late gain={late_gain:.4f}.",
                    "suggestion": "Use scheduler milestones/cosine decay, fine-tune with lower lr, or add harder samples for late-stage improvement.",
                }
            )


def _detect_metric_conflicts(summary: Dict[str, Any], diagnoses: List[Dict[str, str]]) -> None:
    final_metrics = summary.get("final_metrics", {})
    miou = final_metrics.get("final_miou")
    f1 = final_metrics.get("final_f1")
    precision = final_metrics.get("final_precision")
    recall = final_metrics.get("final_recall")

    if miou is not None and f1 is not None and miou > 0.75 and (miou - f1) > 0.12:
        diagnoses.append(
            {
                "type": "metric_conflict_miou_high_f1_low",
                "severity": "medium",
                "evidence": f"Final mIoU={miou} but final F1={f1}.",
                "suggestion": "Semantic regions may be mostly correct while boundaries or instance structure are unstable; inspect F1 thresholding and clustering.",
            }
        )

    if precision is not None and recall is not None and abs(precision - recall) > 0.12:
        diagnoses.append(
            {
                "type": "precision_recall_gap",
                "severity": "medium",
                "evidence": f"Precision={precision}, recall={recall}, gap={abs(precision - recall):.4f}.",
                "suggestion": "Balance false-positive and false-negative costs; evaluate per-class confusion for leaf/stem and boundary points.",
            }
        )

