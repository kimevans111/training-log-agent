from pathlib import Path

from fastapi.testclient import TestClient

from app.main import UPLOAD_DIR, app


ROOT = Path(__file__).resolve().parents[1]


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_analyze_log_endpoint() -> None:
    client = TestClient(app)
    sample_path = ROOT / "examples" / "sample_pointcloud_train.log"

    with sample_path.open("rb") as file_obj:
        upload = client.post(
            "/upload",
            files={"file": ("api_sample_pointcloud_train.log", file_obj, "text/plain")},
        )

    assert upload.status_code == 200
    filename = upload.json()["file_name"]

    try:
        response = client.post(
            "/analyze-log",
            json={
                "log_file_path": filename,
                "user_question": "What should I tune next?",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["summary"]["best_metrics"]["best_miou"] is not None
        assert body["diagnoses"]
        assert body["suggestions"]["priority_suggestions"]
        assert body["figures"]
        assert body["report_path"]
    finally:
        (UPLOAD_DIR / filename).unlink(missing_ok=True)


def test_analyze_examples_log_path_endpoint() -> None:
    client = TestClient(app)

    response = client.post(
        "/analyze-log",
        json={"log_file_path": "examples/sample_pointcloud_train.log"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["best_metrics"]["best_miou"] is not None
    assert body["report_path"]


def test_ask_about_log_endpoint() -> None:
    client = TestClient(app)
    sample_path = ROOT / "examples" / "sample_pointcloud_train.log"

    with sample_path.open("rb") as file_obj:
        upload = client.post(
            "/upload",
            files={"file": ("api_ask_sample.log", file_obj, "text/plain")},
        )

    filename = upload.json()["file_name"]
    try:
        response = client.post(
            "/ask-about-log",
            json={
                "log_file_path": filename,
                "question": "Why is stem IoU lower than leaf IoU?",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["summary"]["best_metrics"]["best_f1"] is not None
    finally:
        (UPLOAD_DIR / filename).unlink(missing_ok=True)
