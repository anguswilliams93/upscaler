"""FastAPI app — upload, queue, serve results."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model_registry import REGISTRY, ensure_weight, list_models
from model_registry_video import (
    REGISTRY as VIDEO_REGISTRY,
    list_video_models,
    verify_weights_present,
)
from schemas import (
    JobStatusResponse,
    ModelEntry,
    UpscaleResponse,
    VideoModelEntry,
)
from upscaler import upscale
from video_upscaler import probe as video_probe, upscale_video

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MODELS_DIR = os.getenv("UPSKALE_MODELS_DIR", "./models")
UPLOADS_DIR = os.getenv("UPSKALE_UPLOADS_DIR", "./uploads")
OUTPUTS_DIR = os.getenv("UPSKALE_OUTPUTS_DIR", "./outputs")
DEFAULT_MODEL = os.getenv("UPSKALE_DEFAULT_MODEL", "ultrasharp")
MAX_UPLOAD_MB = int(os.getenv("UPSKALE_MAX_UPLOAD_MB", "50"))
MAX_VIDEO_MB = int(os.getenv("UPSKALE_MAX_VIDEO_MB", "500"))
MAX_VIDEO_SECONDS = int(os.getenv("UPSKALE_MAX_VIDEO_SECONDS", "60"))
MAX_VIDEO_HEIGHT = int(os.getenv("UPSKALE_MAX_VIDEO_HEIGHT", "720"))
OUTPUT_TTL_HOURS = int(os.getenv("UPSKALE_OUTPUT_TTL_HOURS", "1"))
LOG_LEVEL = os.getenv("UPSKALE_LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_LIME = "\033[38;2;234;249;114m"
_DIM = "\033[38;2;120;130;60m"
_BOLD = "\033[1m"
_RST = "\033[0m"

_VERSION = "v0.1.0"
_TAGLINE = "AI Upscaler"
BANNER = (
    f"\n"
    f"{_LIME} │ {_RST}  {_BOLD}Upzcaler {_VERSION}{_RST}\n"
    f"{_LIME}(│){_RST}  {_TAGLINE}\n"
    f"{_LIME} │ {_RST}  {_DIM}by {_RST}{_LIME}ZEROBI{_RST}{_DIM} · output > input{_RST}\n"
)

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_MIME = {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska"}

JOBS: dict[str, dict[str, Any]] = {}

LOG_BUFFER_MAX = 500
_current_job = threading.local()


def _push_log(job: dict[str, Any], msg: str) -> None:
    job["log_seq"] = job.get("log_seq", 0) + 1
    logs = job.setdefault("logs", [])
    logs.append(msg)
    if len(logs) > LOG_BUFFER_MAX:
        del logs[: len(logs) - LOG_BUFFER_MAX]


class _JobLogHandler(logging.Handler):
    """Routes any log record emitted while a job thread is running into that job's log buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        jid = getattr(_current_job, "job_id", None)
        if not jid:
            return
        job = JOBS.get(jid)
        if job is None:
            return
        try:
            line = self.format(record)
        except Exception:
            return
        _push_log(job, line)


_job_log_handler = _JobLogHandler()
_job_log_handler.setLevel(logging.INFO)
_job_log_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
)
logging.getLogger().addHandler(_job_log_handler)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    print(BANNER)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. RTX 5090 + CUDA 12.8 driver required."
        )
    logger.info("GPU: %s", torch.cuda.get_device_name(0))
    for d in (MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Local AI Upscaler", lifespan=lifespan)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _cleanup_file_later(path: str, delay_s: float) -> None:
    async def _delete() -> None:
        await asyncio.sleep(delay_s)
        try:
            os.remove(path)
        except OSError:
            pass

    asyncio.create_task(_delete())


def _run_job(job_id: str, input_path: str, model_key: str) -> None:
    job = JOBS[job_id]
    job["status"] = "processing"
    job["progress"] = 10
    start = time.time()
    _current_job.job_id = job_id
    try:
        logger.info("starting image upscale model=%s", model_key)
        img = Image.open(input_path)
        job["input_size"] = img.size
        logger.info("input %dx%d", img.size[0], img.size[1])
        tensor = _pil_to_tensor(img)
        job["progress"] = 30
        model_path = ensure_weight(
            model_key, MODELS_DIR,
            log=lambda m: logger.info("[weights] %s", m),
        )
        logger.info("running inference on GPU")
        out = upscale(tensor, model_key, model_path)
        job["progress"] = 85
        out_img = _tensor_to_pil(out)
        out_name = f"{job_id}.png"
        out_path = os.path.join(OUTPUTS_DIR, out_name)
        out_img.save(out_path, format="PNG")
        elapsed = int((time.time() - start) * 1000)
        job.update(
            status="done",
            progress=100,
            output_url=f"/outputs/{out_name}",
            output_size=out_img.size,
            elapsed_ms=elapsed,
        )
        logger.info(
            "upscale done model=%s input_size=%s elapsed_ms=%d",
            model_key, job["input_size"], elapsed,
        )
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        _current_job.job_id = None
        try:
            os.remove(input_path)
        except OSError:
            pass


@app.post("/api/upscale", response_model=UpscaleResponse)
async def api_upscale(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    scale: int = Form(4),
) -> UpscaleResponse:
    if model not in REGISTRY:
        raise HTTPException(400, f"Unknown model: {model}")
    if scale not in (2, 4):
        raise HTTPException(400, "scale must be 2 or 4")
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(400, f"Unsupported MIME: {file.content_type}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB")

    job_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOADS_DIR, f"{job_id}_{file.filename}")
    with open(input_path, "wb") as f:
        f.write(data)

    try:
        Image.open(input_path).verify()
    except Exception as e:
        os.remove(input_path)
        raise HTTPException(400, f"Invalid image: {e}") from e

    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "output_url": None,
        "error": None,
        "input_size": None,
        "output_size": None,
        "elapsed_ms": None,
        "logs": [],
        "log_seq": 0,
    }

    async def _dispatch() -> None:
        await asyncio.to_thread(_run_job, job_id, input_path, model)
        job = JOBS.get(job_id)
        if job and job.get("output_url"):
            out_path = os.path.join(OUTPUTS_DIR, os.path.basename(job["output_url"]))
            _cleanup_file_later(out_path, OUTPUT_TTL_HOURS * 3600)

    background.add_task(_dispatch)
    return UpscaleResponse(job_id=job_id, status="queued")


@app.get("/api/job/{job_id}", response_model=JobStatusResponse)
async def api_job(job_id: str, since: int = 0) -> JobStatusResponse:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    seq = job.get("log_seq", 0)
    all_logs = job.get("logs", [])
    # Buffer holds the last len(all_logs) entries ending at seq.
    first_seq = seq - len(all_logs) + 1
    start = max(0, since - first_seq + 1) if all_logs else 0
    new_logs = all_logs[start:]
    payload = {**job, "logs": new_logs, "log_seq": seq}
    return JobStatusResponse(**payload)


@app.post("/api/job/{job_id}/cancel")
async def api_job_cancel(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    ev = job.get("cancel_event")
    if ev is None:
        raise HTTPException(400, "job is not cancellable")
    ev.set()
    proc = job.get("proc")
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            logger.exception("Failed to terminate proc for job %s", job_id)
    logger.info("Cancel requested for job %s", job_id)
    return {"job_id": job_id, "status": "cancelling"}


@app.get("/api/models", response_model=list[ModelEntry])
async def api_models() -> list[ModelEntry]:
    return [ModelEntry(**m) for m in list_models()]


@app.get("/api/video-models", response_model=list[VideoModelEntry])
async def api_video_models() -> list[VideoModelEntry]:
    return [VideoModelEntry(**m) for m in list_video_models()]


def _run_video_job(job_id: str, input_path: str, model_key: str) -> None:
    job = JOBS[job_id]
    job["status"] = "processing"
    job["progress"] = 5
    start = time.time()
    _current_job.job_id = job_id
    cancel_event = job["cancel_event"]
    try:
        out_name = f"{job_id}.mp4"
        out_path = os.path.join(OUTPUTS_DIR, out_name)
        logger.info("starting video upscale model=%s", model_key)

        last_logged_msg: dict[str, str | None] = {"v": None}

        def on_progress(done: int, total: int, msg: str | None = None) -> None:
            job["frame"] = done
            job["total_frames"] = total
            if msg:
                job["status_msg"] = msg
                if msg != last_logged_msg["v"]:
                    last_logged_msg["v"] = msg
                    logger.info("[progress %d/%d] %s", done, total, msg)
            if total:
                # Reserve 5% for setup, 5% for mux
                job["progress"] = 5 + int(90 * done / total)

        def set_proc(proc: Any | None) -> None:
            job["proc"] = proc

        meta = upscale_video(
            input_path, out_path, model_key, MODELS_DIR,
            progress=on_progress,
            cancel_event=cancel_event,
            set_proc=set_proc,
            batch_size=job.get("batch_size", 0) or None,
        )
        if cancel_event.is_set():
            raise RuntimeError("cancelled")
        scale = VIDEO_REGISTRY[model_key].scale
        elapsed = int((time.time() - start) * 1000)
        job.update(
            status="done",
            progress=100,
            output_url=f"/outputs/{out_name}",
            input_size=(meta.width, meta.height),
            output_size=(meta.width * scale, meta.height * scale),
            fps=meta.fps,
            duration_s=meta.duration_s,
            total_frames=meta.total_frames,
            elapsed_ms=elapsed,
        )
        logger.info(
            "video upscale done model=%s input=%dx%d frames=%d elapsed_ms=%d",
            model_key, meta.width, meta.height, meta.total_frames, elapsed,
        )
    except Exception as e:
        if cancel_event.is_set():
            logger.info("Video job %s cancelled", job_id)
            job["status"] = "cancelled"
            job["error"] = "cancelled by user"
        else:
            logger.exception("Video job %s failed", job_id)
            job["status"] = "error"
            job["error"] = str(e)
    finally:
        job["proc"] = None
        _current_job.job_id = None
        try:
            os.remove(input_path)
        except OSError:
            pass


@app.post("/api/upscale-video", response_model=UpscaleResponse)
async def api_upscale_video(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(...),
    batch_size: int = Form(0),
) -> UpscaleResponse:
    if model not in VIDEO_REGISTRY:
        raise HTTPException(400, f"Unknown video model: {model}")
    if file.content_type not in ALLOWED_VIDEO_MIME:
        raise HTTPException(400, f"Unsupported video MIME: {file.content_type}")

    missing = verify_weights_present(model, MODELS_DIR)
    if missing:
        raise HTTPException(
            503,
            f"Model weights missing: {missing}. Place under {MODELS_DIR}/ per model_registry_video.py.",
        )

    data = await file.read()
    if len(data) > MAX_VIDEO_MB * 1024 * 1024:
        raise HTTPException(413, f"Video exceeds {MAX_VIDEO_MB} MB")

    job_id = uuid.uuid4().hex
    safe_name = (file.filename or "video.mp4").replace("/", "_").replace("\\", "_")
    input_path = os.path.join(UPLOADS_DIR, f"{job_id}_{safe_name}")
    with open(input_path, "wb") as f:
        f.write(data)

    try:
        meta = video_probe(input_path)
    except Exception as e:
        os.remove(input_path)
        raise HTTPException(400, f"Invalid video: {e}") from e

    if meta.duration_s > MAX_VIDEO_SECONDS:
        os.remove(input_path)
        raise HTTPException(
            400,
            f"Video {meta.duration_s:.1f}s exceeds {MAX_VIDEO_SECONDS}s limit",
        )
    cap_h = min(MAX_VIDEO_HEIGHT, VIDEO_REGISTRY[model].max_input_height)
    if meta.height > cap_h:
        os.remove(input_path)
        raise HTTPException(
            400, f"Video height {meta.height}px exceeds {cap_h}px limit",
        )

    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "output_url": None,
        "error": None,
        "input_size": (meta.width, meta.height),
        "output_size": None,
        "elapsed_ms": None,
        "kind": "video",
        "frame": 0,
        "total_frames": meta.total_frames,
        "fps": meta.fps,
        "duration_s": meta.duration_s,
        "status_msg": None,
        "logs": [],
        "log_seq": 0,
        "cancel_event": threading.Event(),
        "proc": None,
        "batch_size": batch_size,
    }

    async def _dispatch() -> None:
        await asyncio.to_thread(_run_video_job, job_id, input_path, model)
        job = JOBS.get(job_id)
        if job and job.get("output_url"):
            out_path = os.path.join(OUTPUTS_DIR, os.path.basename(job["output_url"]))
            _cleanup_file_later(out_path, OUTPUT_TTL_HOURS * 3600)

    background.add_task(_dispatch)
    return UpscaleResponse(job_id=job_id, status="queued")


@app.get("/api/gpu")
async def api_gpu() -> dict[str, Any]:
    """Live GPU stats for the topbar."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "vram_used_gb": 0.0, "vram_total_gb": 0.0, "jobs_done": 0}
    used = torch.cuda.memory_allocated(0) / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    done = sum(1 for j in JOBS.values() if j["status"] == "done")
    return {
        "name": torch.cuda.get_device_name(0),
        "vram_used_gb": round(used, 2),
        "vram_total_gb": round(total, 1),
        "jobs_done": done,
    }


@app.get("/outputs/{filename}")
async def serve_output(filename: str) -> FileResponse:
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "output not found")
    return FileResponse(path)


app.mount("/", StaticFiles(directory="static", html=True), name="static")


def _console_main() -> None:
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
