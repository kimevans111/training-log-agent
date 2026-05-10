"""Pydantic schemas for the FastAPI app."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AnalyzeLogRequest(BaseModel):
    """Request body for log analysis."""

    log_file_path: str = Field(
        ...,
        description="Path returned from /upload, or a safe local path under uploads/ or examples/.",
    )
    user_question: Optional[str] = Field(None, description="Optional question about the log.")
    config: Dict[str, Any] = Field(default_factory=dict)


class AskAboutLogRequest(BaseModel):
    """Request body for asking a question about a log."""

    log_file_path: str
    question: str
    config: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    """Metadata returned after file upload."""

    file_name: str
    file_size: int
    file_type: str
    saved_path: str
