from pathlib import Path

from core.report_generator import generate_report


def test_report_file_is_generated_with_required_titles(tmp_path: Path) -> None:
    parsed = {"file_name": "demo.log", "num_epochs": 1, "metrics_series": [{"epoch": 1, "miou": 0.8, "f1": 0.75}]}
    summary = {
        "headline": "Parsed 1 epoch.",
        "best_metrics": {"best_miou": 0.8, "best_miou_epoch": 1, "best_f1": 0.75, "best_f1_epoch": 1},
        "final_metrics": {"final_epoch": 1, "final_miou": 0.8, "final_f1": 0.75, "final_leaf_iou": None, "final_stem_iou": None},
        "trend_metrics": {"last_window_size": 1, "last_miou_mean": 0.8, "last_f1_mean": 0.75, "miou_std_recent": None, "f1_std_recent": None},
        "class_gap": {"leaf_stem_iou_gap": None},
        "loss_gap": {"gap": None},
    }
    diagnostics = {"diagnoses": [{"type": "healthy_training", "severity": "low", "evidence": "ok", "suggestion": "continue"}]}
    suggestions = {"priority_suggestions": ["Keep baseline."], "next_experiments": [{"name": "seed_sweep", "change": "3 seeds", "expected_effect": "confidence"}]}

    report_path = generate_report(parsed, summary, diagnostics, suggestions, [], tmp_path)
    content = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "# Training Log Analysis Report" in content
    assert "## 4. Diagnosed Issues" in content
    assert "## 6. Recommended Next Experiments" in content

