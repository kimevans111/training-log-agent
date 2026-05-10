"""Run a local portfolio demo for Training Log Agent."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.training_log_agent import TrainingLogAgent


def main() -> None:
    """Analyze the bundled sample log and print a concise demo summary."""
    print("== Training Log Agent Demo ==")
    agent = TrainingLogAgent(
        report_dir=ROOT / "reports",
        figure_dir=ROOT / "reports" / "figures",
    )
    result = agent.analyze(
        ROOT / "examples" / "sample_pointcloud_train.log",
        user_question="What should I tune next if stem IoU is lower than leaf IoU?",
    )

    summary = result["summary"]
    print(summary["headline"])
    print("\n-- Diagnoses --")
    for item in result["diagnoses"]:
        print(f"- {item['type']} ({item['severity']}): {item['evidence']}")

    print("\n-- Priority Suggestions --")
    for item in result["suggestions"]["priority_suggestions"][:5]:
        print(f"- {item}")

    print(f"\nFigures: {len(result['figures'])}")
    print(f"Report: {result['report_path']}")


if __name__ == "__main__":
    main()
