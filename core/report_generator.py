"""Markdown report generation for Training Log Agent."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


def generate_report(
    parsed_log: Dict[str, Any],
    summary: Dict[str, Any],
    diagnostics: Dict[str, Any],
    suggestions: Dict[str, Any],
    figures: Iterable[Path | str],
    output_dir: Path | str = Path("reports"),
) -> Path:
    """Generate a timestamped Markdown analysis report."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"training_log_report_{timestamp}.md"
    report_path.write_text(
        _build_report_markdown(parsed_log, summary, diagnostics, suggestions, list(figures)),
        encoding="utf-8",
    )
    return report_path


def _build_report_markdown(
    parsed_log: Dict[str, Any],
    summary: Dict[str, Any],
    diagnostics: Dict[str, Any],
    suggestions: Dict[str, Any],
    figures: List[Path | str],
) -> str:
    available_metrics = _available_metrics(parsed_log.get("metrics_series", []))
    best = summary.get("best_metrics", {})
    final = summary.get("final_metrics", {})
    class_gap = summary.get("class_gap", {})
    loss_gap = summary.get("loss_gap", {})
    trend = summary.get("trend_metrics", {})

    lines: List[str] = [
        "# Training Log Analysis Report",
        "",
        "## 1. Experiment Overview",
        f"- File name: {parsed_log.get('file_name')}",
        f"- Final epoch: {final.get('final_epoch')}",
        f"- Number of parsed epochs: {parsed_log.get('num_epochs')}",
        f"- Available metrics: {', '.join(available_metrics) if available_metrics else 'None'}",
        "",
        "## 2. Key Metrics",
        f"- best mIoU: {_fmt(best.get('best_miou'))} at epoch {best.get('best_miou_epoch')}",
        f"- best F1: {_fmt(best.get('best_f1'))} at epoch {best.get('best_f1_epoch')}",
        f"- final mIoU: {_fmt(final.get('final_miou'))}",
        f"- final F1: {_fmt(final.get('final_f1'))}",
        f"- leaf IoU: {_fmt(final.get('final_leaf_iou'))}",
        f"- stem IoU: {_fmt(final.get('final_stem_iou'))}",
        f"- leaf/stem IoU gap: {_fmt(class_gap.get('leaf_stem_iou_gap'))}",
        f"- train-val loss gap: {_fmt(loss_gap.get('gap'))}",
        "",
        "## 3. Training Dynamics",
        (
            f"{summary.get('headline', '')} In the last {trend.get('last_window_size')} epochs, "
            f"mean mIoU was {_fmt(trend.get('last_miou_mean'))}, mean F1 was {_fmt(trend.get('last_f1_mean'))}, "
            f"recent mIoU std was {_fmt(trend.get('miou_std_recent'))}, and recent F1 std was {_fmt(trend.get('f1_std_recent'))}."
        ),
        "",
        "## 4. Diagnosed Issues",
        "_detected issues_",
    ]

    for item in diagnostics.get("diagnoses", []):
        lines.extend(
            [
                f"- **{item.get('type')}** ({item.get('severity')}): {item.get('evidence')}",
                f"  Suggestion: {item.get('suggestion')}",
            ]
        )

    lines.extend(["", "## 5. Suggestions"])
    for suggestion in suggestions.get("priority_suggestions", []):
        lines.append(f"- {suggestion}")

    lines.extend(["", "## 6. Recommended Next Experiments", "_next experiments_"])
    for experiment in suggestions.get("next_experiments", []):
        lines.extend(
            [
                f"- **{experiment.get('name')}**",
                f"  Change: {experiment.get('change')}",
                f"  Expected effect: {experiment.get('expected_effect')}",
            ]
        )

    lines.extend(["", "## 7. Generated Figures", "_generated figures_"])
    if figures:
        for figure in figures:
            figure_path = str(figure).replace("\\", "/")
            lines.append(f"- {figure_path}")
            lines.append(f"  ![]({figure_path})")
    else:
        lines.append("- No figures were generated because no plottable metrics were found.")

    lines.append("")
    return "\n".join(lines)


def _available_metrics(series: List[Dict[str, Any]]) -> List[str]:
    excluded = {"source_line"}
    metrics = sorted({key for record in series for key in record.keys() if key not in excluded})
    return metrics


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

