# AGENTS.md

## Project Role

Training Log Agent is a Python MVP for automatic deep learning training-log analysis, focused on 3D plant point cloud semantic segmentation experiments.

## Development Notes

- Keep modules small and task-oriented: parsing, summary, diagnostics, suggestions, plotting, report generation, orchestration.
- Use `pathlib.Path` for file paths.
- Keep the project runnable without an API key. LLM integration must fall back to `MockLLMProvider`.
- Avoid crashing when metrics are missing; return `None`, empty lists, or clear warnings.
- Add tests when changing parser behavior, diagnostic rules, plotting, or report output.

## Common Commands

```bash
pip install -r requirements.txt
pytest
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
streamlit run frontend/streamlit_app.py
```

