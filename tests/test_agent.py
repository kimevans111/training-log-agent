from pathlib import Path

from agent.training_log_agent import TrainingLogAgent


def test_complete_agent_flow_runs(tmp_path: Path) -> None:
    log_path = Path("examples/sample_pointcloud_train.log")
    agent = TrainingLogAgent(report_dir=tmp_path / "reports", figure_dir=tmp_path / "figures")

    result = agent.analyze(log_path, user_question="What should I tune next?")

    assert result["summary"]["best_metrics"]["best_miou"] is not None
    assert result["diagnoses"]
    assert result["suggestions"]["priority_suggestions"]
    assert Path(result["report_path"]).exists()
    assert len(result["figures"]) >= 3
    assert result["answer"]
    for figure in result["figures"]:
        assert Path(figure).exists()

