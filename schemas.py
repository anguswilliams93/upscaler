"""Pydantic models for API requests and responses."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "processing", "done", "error", "cancelled"]
MediaKind = Literal["image", "video"]


class UpscaleResponse(BaseModel):
    job_id: str
    status: JobStatus = "queued"


class JobStatusResponse(BaseModel):
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    output_url: str | None = None
    error: str | None = None
    input_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None
    elapsed_ms: int | None = None
    # Video-only fields (None for image jobs)
    kind: MediaKind = "image"
    frame: int | None = None
    total_frames: int | None = None
    fps: float | None = None
    duration_s: float | None = None
    status_msg: str | None = None
    # Streaming log feed
    logs: list[str] = Field(default_factory=list)
    log_seq: int = 0


class ModelEntry(BaseModel):
    key: str
    name: str
    scale: int
    best_for: str


class VideoModelEntry(BaseModel):
    key: str
    name: str
    scale: int
    best_for: str
    vram_gb: float
    max_input_height: int
