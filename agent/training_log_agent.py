"""Main Training Log Agent workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from core.diagnostics import diagnose_training
from core.log_parser import parse_log_file
from core.metric_summary import generate_metric_summary
from core.plotter import generate_plots
from core.report_generator import generate_report
from core.suggestion_engine import generate_suggestions
from llm.provider import BaseLLMProvider, get_llm_provider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_FIGURE_DIR = DEFAULT_REPORT_DIR / "figures"


class TrainingLogAgent:
    """Coordinate parsing, diagnostics, plotting, reporting, and QA."""

    def __init__(
        self,
        report_dir: Path | str = DEFAULT_REPORT_DIR,
        figure_dir: Path | str = DEFAULT_FIGURE_DIR,
        llm_provider: Optional[BaseLLMProvider] = None,
    ) -> None:
        self.report_dir = Path(report_dir)
        self.figure_dir = Path(figure_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.llm_provider = llm_provider or get_llm_provider()

    def analyze(
        self,
        log_file_path: Path | str,
        user_question: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the complete Training Log Agent analysis pipeline."""

        config = config or {}
        parsed_log = parse_log_file(Path(log_file_path), normalize_percent=config.get("normalize_percent", True))
        summary = generate_metric_summary(parsed_log)
        diagnostics = diagnose_training(parsed_log, summary)
        suggestions = generate_suggestions(diagnostics, summary)
        figures = generate_plots(parsed_log, self.figure_dir)
        report_path = generate_report(parsed_log, summary, diagnostics, suggestions, figures, self.report_dir)
        answer = self.answer_question(user_question, summary, diagnostics, suggestions) if user_question else ""

        return {
            "summary": summary,
            "diagnoses": diagnostics.get("diagnoses", []),
            "suggestions": suggestions,
            "figures": [str(path) for path in figures],
            "report_path": str(report_path),
            "answer": answer,
            "parsed_log": parsed_log,
        }

    def answer_question(
        self,
        user_question: Optional[str],
        summary: Dict[str, Any],
        diagnostics: Dict[str, Any],
        suggestions: Dict[str, Any],
    ) -> str:
        """Answer a user question using the configured provider."""

        if not user_question:
            return ""
        return self.llm_provider.generate(
            user_question,
            {
                "summary": summary,
                "diagnoses": diagnostics,
                "suggestions": suggestions,
            },
        )


def analyze_log(
    log_file_path: Path | str,
    user_question: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function for one-shot analysis."""

    return TrainingLogAgent().analyze(log_file_path, user_question=user_question, config=config)

