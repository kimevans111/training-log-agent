from core.diagnostics import diagnose_training
from core.metric_summary import generate_metric_summary


def test_overfitting_rule_detects_early_best_and_drop() -> None:
    series = [
        {"epoch": 1, "train_loss": 0.9, "val_loss": 0.8, "miou": 0.55, "f1": 0.56},
        {"epoch": 2, "train_loss": 0.6, "val_loss": 0.7, "miou": 0.78, "f1": 0.74},
        {"epoch": 3, "train_loss": 0.4, "val_loss": 0.82, "miou": 0.70, "f1": 0.66},
        {"epoch": 4, "train_loss": 0.3, "val_loss": 0.95, "miou": 0.67, "f1": 0.62},
    ]
    parsed = {"num_epochs": 4, "final_epoch": 4, "metrics_series": series}
    summary = generate_metric_summary(parsed)

    diagnoses = diagnose_training(parsed, summary)["diagnoses"]

    assert any(item["type"] == "overfitting" for item in diagnoses)


def test_oscillation_rule_detects_unstable_f1() -> None:
    series = [
        {"epoch": i + 1, "miou": 0.8 + (0.005 if i % 2 == 0 else -0.005), "f1": 0.74 + (0.08 if i % 2 == 0 else -0.08)}
        for i in range(12)
    ]
    parsed = {"num_epochs": 12, "final_epoch": 12, "metrics_series": series}
    summary = generate_metric_summary(parsed)

    diagnoses = diagnose_training(parsed, summary)["diagnoses"]

    assert any(item["type"] in {"training_oscillation", "f1_instability"} for item in diagnoses)


def test_class_imbalance_rule_detects_stem_gap() -> None:
    series = [
        {"epoch": 1, "leaf_iou": 0.88, "stem_iou": 0.55, "precision": 0.9, "recall": 0.78},
        {"epoch": 2, "leaf_iou": 0.90, "stem_iou": 0.58, "precision": 0.91, "recall": 0.79},
    ]
    parsed = {"num_epochs": 2, "final_epoch": 2, "metrics_series": series}
    summary = generate_metric_summary(parsed)

    diagnoses = diagnose_training(parsed, summary)["diagnoses"]

    assert any(item["type"] == "class_imbalance" for item in diagnoses)

