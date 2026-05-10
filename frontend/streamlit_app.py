"""Streamlit frontend for Training Log Agent."""

from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def post_json(path: str, payload: dict) -> dict:
    response = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def upload_to_backend(file_obj) -> dict:
    files = {"file": (file_obj.name, file_obj.getvalue())}
    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Training Log Agent", layout="wide")
    st.title("Training Log Agent")
    st.write(
        "Training log analysis and tuning recommendation system for 3D plant point cloud segmentation experiments."
    )

    with st.sidebar:
        st.caption(f"Backend: {BACKEND_URL}")
        try:
            health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
            st.success(f"API {health.get('status', 'ok')}")
        except Exception:
            st.error("Backend is not reachable. Start it with: uvicorn app.main:app --reload")

    uploaded_file = st.file_uploader("Upload a training log", type=["log", "txt", "csv", "json"])
    user_question = st.text_input("Ask about this log", placeholder="Why is stem IoU much lower than leaf IoU?")

    saved_filename = None
    if uploaded_file is not None:
        try:
            upload_result = upload_to_backend(uploaded_file)
            saved_filename = upload_result["file_name"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("File", saved_filename)
            col2.metric("Size", f"{upload_result['file_size'] / 1024:.1f} KB")
            col3.metric("Type", uploaded_file.type or "unknown")
            col4.metric("Saved", "uploads/")
            st.caption(upload_result["saved_path"])
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    if st.button("Analyze Log", type="primary", disabled=saved_filename is None):
        with st.spinner("Analyzing training dynamics..."):
            try:
                result = post_json("/analyze-log", {
                    "log_file_path": f"uploads/{saved_filename}",
                    "user_question": user_question or None,
                })
                render_result(result)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


def render_result(result: dict) -> None:
    summary = result.get("summary", {})
    best = summary.get("best_metrics", {})
    final = summary.get("final_metrics", {})
    class_gap = summary.get("class_gap", {})

    st.subheader("Metric Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best mIoU", _fmt(best.get("best_miou")), f"epoch {best.get('best_miou_epoch')}")
    c2.metric("Best F1", _fmt(best.get("best_f1")), f"epoch {best.get('best_f1_epoch')}")
    c3.metric("Final mIoU", _fmt(final.get("final_miou")), f"epoch {final.get('final_epoch')}")
    c4.metric("Leaf/Stem Gap", _fmt(class_gap.get("leaf_stem_iou_gap")))

    st.subheader("Diagnosed Issues")
    for item in result.get("diagnoses", []):
        st.info(f"**{item.get('type')}** ({item.get('severity')}): {item.get('evidence')}\n\n{item.get('suggestion')}")

    suggestions = result.get("suggestions", {})
    st.subheader("Tuning Suggestions")
    for suggestion in suggestions.get("priority_suggestions", []):
        st.write(f"- {suggestion}")

    st.subheader("Recommended Next Experiments")
    for experiment in suggestions.get("next_experiments", []):
        st.write(f"**{experiment.get('name')}**")
        st.write(f"- Change: {experiment.get('change')}")
        st.write(f"- Expected effect: {experiment.get('expected_effect')}")

    if result.get("answer"):
        st.subheader("Agent Answer")
        st.write(result["answer"])

    st.subheader("Training Curves")
    figures = result.get("figures", [])
    if figures:
        for figure in figures:
            fig_path = Path(figure)
            if fig_path.exists():
                st.image(str(fig_path), caption=fig_path.name)
            else:
                fig_name = fig_path.name
                try:
                    resp = requests.get(f"{BACKEND_URL}/figures/{fig_name}", timeout=10)
                    if resp.status_code == 200:
                        st.image(resp.content, caption=fig_name)
                except Exception:
                    st.write(f"Figure not available: {fig_name}")
    else:
        st.write("No figures were generated.")

    report_path = result.get("report_path")
    if report_path:
        report_name = Path(report_path).name
        try:
            resp = requests.get(f"{BACKEND_URL}/reports/{report_name}", timeout=10)
            if resp.status_code == 200:
                st.download_button(
                    "Download Markdown Report",
                    resp.content,
                    file_name=report_name,
                    mime="text/markdown",
                )
        except Exception:
            st.info(f"Report available at: {report_path}")


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
