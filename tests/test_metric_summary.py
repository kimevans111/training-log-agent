from core.metric_summary import generate_metric_summary


def test_best_and_final_metrics_are_extracted() -> None:
    parsed = {
        "num_epochs": 3,
        "final_epoch": 3,
        "metrics_series": [
            {"epoch": 1, "miou": 0.5, "f1": 0.52, "train_loss": 1.0, "val_loss": 1.1, "leaf_iou": 0.7, "stem_iou": 0.4},
            {"epoch": 2, "miou": 0.7, "f1": 0.66, "train_loss": 0.7, "val_loss": 0.8, "leaf_iou": 0.8, "stem_iou": 0.5},
            {"epoch": 3, "miou": 0.65, "f1": 0.72, "train_loss": 0.5, "val_loss": 0.7, "leaf_iou": 0.82, "stem_iou": 0.55},
        ],
    }

    summary = generate_metric_summary(parsed)

    assert summary["best_metrics"]["best_miou"] == 0.7
    assert summary["best_metrics"]["best_miou_epoch"] == 2
    assert summary["best_metrics"]["best_f1"] == 0.72
    assert summary["final_metrics"]["final_miou"] == 0.65


def test_class_gap_is_computed() -> None:
    parsed = {
        "num_epochs": 2,
        "final_epoch": 2,
        "metrics_series": [
            {"epoch": 1, "leaf_iou": 0.8, "stem_iou": 0.5},
            {"epoch": 2, "leaf_iou": 0.9, "stem_iou": 0.6},
        ],
    }

    summary = generate_metric_summary(parsed)

    assert summary["class_gap"]["leaf_stem_iou_gap"] == 0.3
    assert summary["class_gap"]["status"] == "large_gap"

