"""Generate concrete tuning suggestions from diagnostics."""

from __future__ import annotations

from typing import Any, Dict, List


SUGGESTION_BANK: Dict[str, List[str]] = {
    "overfitting": [
        "Enable early stopping around the best mIoU/F1 epoch and keep the checkpoint with the strongest validation metric.",
        "Increase point cloud augmentations: random rotation, jitter, scaling, elastic distortion, and plant-level crop/mix augmentation.",
        "Add regularization such as weight decay, dropout in MLP heads, stochastic depth, or reduced decoder capacity.",
    ],
    "underfitting": [
        "Train longer or increase model capacity, especially the local geometry encoder for thin stem structures.",
        "Verify coordinate/color/normal normalization and make sure labels are aligned after point sampling.",
    ],
    "insufficient_training": [
        "Extend training by 20-30% and use cosine decay or ReduceLROnPlateau to preserve late-stage gains.",
    ],
    "training_oscillation": [
        "Reduce learning rate by 2x-5x, add 5-10% warmup, and enable gradient clipping.",
        "Increase batch size or use gradient accumulation to stabilize batch statistics.",
        "Track EMA weights and compare EMA validation F1 against raw checkpoint F1.",
    ],
    "f1_instability": [
        "Tune inference thresholds, connected-component or clustering radius, and minimum instance size filters.",
        "Inspect boundary points between leaf and stem; add boundary-aware loss or local neighborhood consistency loss.",
    ],
    "class_imbalance": [
        "Use class-weighted CE/Dice/Focal loss with a higher stem weight.",
        "Apply class-balanced sampling or crop more stem-heavy regions from plant point clouds.",
        "Add stem-specific augmentation and hard example mining for thin/boundary points.",
    ],
    "low_recall": [
        "Lower positive thresholds or increase recall-oriented loss weights for stem and boundary points.",
        "Audit false negatives in thin stems and occluded junction areas.",
    ],
    "low_precision": [
        "Tighten post-processing thresholds and add hard-negative mining for leaf regions misclassified as stem.",
    ],
    "possible_lr_too_high": [
        "Try lr from 1e-4 to 5e-5, warmup 10 epochs, and gradient clipping at norm 1.0.",
    ],
    "possible_lr_too_small_or_plateau": [
        "Try a slightly higher initial lr or OneCycle/cosine schedule if metrics improve too slowly.",
    ],
    "metric_plateau": [
        "Fine-tune from the best checkpoint with a 5x lower lr and stronger rare-class sampling.",
    ],
    "metric_conflict_miou_high_f1_low": [
        "Keep mIoU checkpoint selection but add F1-aware validation, boundary F1, and per-class F1 monitoring.",
    ],
    "precision_recall_gap": [
        "Sweep decision thresholds and report class-wise precision/recall to identify false-positive or false-negative dominant classes.",
    ],
}


def generate_suggestions(diagnostics: Dict[str, Any], summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build prioritized suggestions and next experiment proposals."""

    diagnoses = diagnostics.get("diagnoses", [])
    priority_suggestions: List[str] = []
    next_experiments: List[Dict[str, str]] = []

    for diagnosis in diagnoses:
        issue_type = diagnosis.get("type", "")
        for suggestion in SUGGESTION_BANK.get(issue_type, []):
            if suggestion not in priority_suggestions:
                priority_suggestions.append(suggestion)
        next_experiment = _experiment_for_issue(issue_type)
        if next_experiment and next_experiment not in next_experiments:
            next_experiments.append(next_experiment)

    if not priority_suggestions:
        priority_suggestions.append(
            "No severe issue was detected. Use this run as a baseline and change one variable at a time in the next experiment."
        )
    if not next_experiments:
        next_experiments.append(
            {
                "name": "baseline_repeat_with_seed_sweep",
                "change": "Run 3 seeds with the same config and report mean/std of mIoU and F1.",
                "expected_effect": "Confirm whether observed metric movement is stable or seed-dependent.",
            }
        )

    return {
        "priority_suggestions": priority_suggestions,
        "next_experiments": next_experiments,
    }


def _experiment_for_issue(issue_type: str) -> Dict[str, str] | None:
    experiments = {
        "overfitting": {
            "name": "early_stop_regularized_aug",
            "change": "Enable early stopping with patience 50, add weight_decay=1e-4, dropout=0.2, and stronger point jitter/rotation.",
            "expected_effect": "Reduce validation metric drop after the best epoch.",
        },
        "underfitting": {
            "name": "longer_training_capacity_check",
            "change": "Train 30% longer and compare a wider encoder or deeper local aggregation block.",
            "expected_effect": "Improve both train and validation metrics if the current model is capacity-limited.",
        },
        "insufficient_training": {
            "name": "extend_training_with_cosine_tail",
            "change": "Extend max_epoch by 20-30% and use cosine decay to a small final lr.",
            "expected_effect": "Capture late-stage mIoU/F1 improvements without large optimizer jumps.",
        },
        "training_oscillation": {
            "name": "lower_lr_with_warmup",
            "change": "lr from 1e-4 to 5e-5, warmup 10 epochs, grad_clip_norm=1.0.",
            "expected_effect": "Reduce F1 fluctuation and stabilize validation curves.",
        },
        "f1_instability": {
            "name": "boundary_and_postprocess_sweep",
            "change": "Sweep clustering radius/min_cluster_size and add boundary-aware auxiliary loss.",
            "expected_effect": "Improve F1 consistency when mIoU is already reasonable.",
        },
        "class_imbalance": {
            "name": "stem_balanced_loss_sampling",
            "change": "Use focal loss gamma=2, stem class weight 2-4x, and stem-heavy crop sampling.",
            "expected_effect": "Raise stem IoU and reduce leaf/stem IoU gap.",
        },
        "low_recall": {
            "name": "recall_threshold_sweep",
            "change": "Lower positive threshold and evaluate per-class recall for stem and boundary regions.",
            "expected_effect": "Reduce missed thin structures.",
        },
        "low_precision": {
            "name": "hard_negative_precision_sweep",
            "change": "Increase threshold slightly and mine false-positive leaf/stem confusion samples.",
            "expected_effect": "Reduce false-positive predictions.",
        },
        "possible_lr_too_high": {
            "name": "lr_stability_ablation",
            "change": "Compare lr values [1e-4, 5e-5, 2e-5] with identical seed and warmup.",
            "expected_effect": "Find a smoother validation trajectory with less oscillation.",
        },
        "possible_lr_too_small_or_plateau": {
            "name": "scheduler_plateau_ablation",
            "change": "Compare cosine, OneCycle, and ReduceLROnPlateau schedulers.",
            "expected_effect": "Improve late-stage convergence speed.",
        },
        "metric_plateau": {
            "name": "best_checkpoint_finetune",
            "change": "Resume best checkpoint with 5x lower lr and class-balanced mini-batches.",
            "expected_effect": "Improve late validation metrics after plateau.",
        },
        "metric_conflict_miou_high_f1_low": {
            "name": "f1_aware_validation",
            "change": "Add per-class F1 and boundary F1 to checkpoint selection; sweep post-processing thresholds.",
            "expected_effect": "Align semantic IoU quality with instance/boundary stability.",
        },
        "precision_recall_gap": {
            "name": "precision_recall_threshold_sweep",
            "change": "Sweep prediction thresholds and report per-class PR curves.",
            "expected_effect": "Balance false positives and false negatives.",
        },
    }
    return experiments.get(issue_type)

