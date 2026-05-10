"""FastAPI backend for Training Log Agent."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agent.training_log_agent import TrainingLogAgent
from app.schemas import AnalyzeLogRequest, AskAboutLogRequest, UploadResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "uploads"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
ALLOWED_SUFFIXES = {".log", ".txt", ".csv", ".json"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Training Log Agent",
    description="Training log parser, diagnostic agent, and report generator for deep learning experiments.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, str]:
    """Return service metadata."""

    return {"message": "Training Log Agent API", "docs": "/docs"}


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a training log file into the uploads directory."""

    original_name = Path(file.filename or "uploaded.log").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    saved_path = UPLOAD_DIR / original_name
    counter = 1
    while saved_path.exists():
        saved_path = UPLOAD_DIR / f"{saved_path.stem}_{counter}{suffix}"
        counter += 1

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return UploadResponse(
        file_name=saved_path.name,
        file_size=saved_path.stat().st_size,
        file_type=suffix,
        saved_path=str(saved_path),
    )


@app.post("/analyze-log")
def analyze_log(request: AnalyzeLogRequest) -> Dict[str, Any]:
    """Run the complete analysis pipeline for a local log path."""

    path = _resolve_log_path(request.log_file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {path}")
    agent = TrainingLogAgent(report_dir=REPORT_DIR, figure_dir=FIGURE_DIR)
    return agent.analyze(path, user_question=request.user_question, config=request.config)


@app.post("/ask-about-log")
def ask_about_log(request: AskAboutLogRequest) -> Dict[str, Any]:
    """Analyze a log and answer a question about it."""

    path = _resolve_log_path(request.log_file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {path}")
    agent = TrainingLogAgent(report_dir=REPORT_DIR, figure_dir=FIGURE_DIR)
    result = agent.analyze(path, user_question=request.question, config=request.config)
    return {"answer": result.get("answer"), "summary": result.get("summary"), "diagnoses": result.get("diagnoses")}


@app.get("/reports/{filename}")
def get_report(filename: str) -> FileResponse:
    """Download a generated Markdown report."""

    path = _safe_file_in_dir(REPORT_DIR, filename)
    if not path.exists() or path.suffix.lower() != ".md":
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="text/markdown", filename=path.name)


@app.get("/figures/{filename}")
def get_figure(filename: str) -> FileResponse:
    """Download a generated figure."""

    path = _safe_file_in_dir(FIGURE_DIR, filename)
    if not path.exists() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=404, detail="Figure not found")
    return FileResponse(path, media_type="image/png", filename=path.name)


def _resolve_log_path(value: str) -> Path:
    """Resolve a user log path within uploads/ or examples/ only."""

    raw_path = Path(value)
    if raw_path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {raw_path.suffix.lower()}")

    if raw_path.is_absolute():
        candidates = [raw_path]
    elif raw_path.parent == Path("."):
        candidates = [UPLOAD_DIR / raw_path.name, EXAMPLES_DIR / raw_path.name]
    else:
        candidates = [PROJECT_ROOT / raw_path]

    last_path = candidates[-1]
    for candidate in candidates:
        resolved = _ensure_within_allowed_dirs(candidate, (UPLOAD_DIR, EXAMPLES_DIR))
        last_path = resolved
        if resolved.exists():
            return resolved
    return last_path


def _ensure_within_allowed_dirs(path: Path, allowed_dirs: tuple[Path, ...]) -> Path:
    resolved = path.resolve()
    allowed = [directory.resolve() for directory in allowed_dirs]
    if any(resolved == directory or directory in resolved.parents for directory in allowed):
        return resolved
    allowed_names = ", ".join(directory.name for directory in allowed_dirs)
    raise HTTPException(status_code=400, detail=f"Path must stay inside one of: {allowed_names}")


def _safe_file_in_dir(base_dir: Path, filename: str) -> Path:
    candidate = (base_dir / Path(filename).name).resolve()
    if base_dir.resolve() not in candidate.parents and candidate != base_dir.resolve():
        raise HTTPException(status_code=400, detail="Invalid path")
    return candidate
