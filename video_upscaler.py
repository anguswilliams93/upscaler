"""Video upscaling pipeline: ffmpeg I/O + chunked GPU inference + audio mux."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np
import torch

import loaders
from model_registry_video import REGISTRY, get_weight_paths

logger = logging.getLogger(__name__)


def _ffmpeg_bin() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg") or "ffmpeg"


FFMPEG = _ffmpeg_bin()
FFPROBE = shutil.which("ffprobe") or "ffprobe"

CHUNK_FRAMES = 5  # temporal window per GPU pass

_pipeline_cache: dict[str, loaders.VideoUpscalerBase] = {}


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    duration_s: float
    total_frames: int
    has_audio: bool


def probe(path: str) -> VideoMeta:
    """ffprobe → VideoMeta. Raises RuntimeError on failure."""
    cmd = [
        FFPROBE, "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr.strip()}")
    data = json.loads(res.stdout)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError("no video stream")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    num, den = (v.get("r_frame_rate", "0/1") + "/1").split("/")[:2]
    fps = float(num) / float(den) if float(den) else 0.0
    duration = float(data["format"].get("duration", 0.0))
    nb = v.get("nb_frames")
    total = int(nb) if nb and nb.isdigit() else int(round(fps * duration))
    return VideoMeta(
        width=int(v["width"]),
        height=int(v["height"]),
        fps=fps,
        duration_s=duration,
        total_frames=total,
        has_audio=has_audio,
    )


def get_pipeline(model_key: str, models_dir: str) -> loaders.VideoUpscalerBase:
    if model_key not in _pipeline_cache:
        info = REGISTRY[model_key]
        wp = get_weight_paths(model_key, models_dir)
        pipe = loaders.build(info.loader, wp)
        pipe.load()
        _pipeline_cache[model_key] = pipe
    return _pipeline_cache[model_key]


def _decode_frames(path: str, w: int, h: int):
    """Yield frames as float32 HWC numpy [0,1] via ffmpeg rawvideo pipe."""
    cmd = [
        FFMPEG, "-v", "error", "-i", path,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            yield arr.astype(np.float32) / 255.0
    finally:
        proc.stdout.close()
        proc.wait()


class _FrameWriter:
    """ffmpeg encoder over rawvideo stdin → mp4 (no audio)."""

    def __init__(self, path: str, w: int, h: int, fps: float) -> None:
        self.path = path
        self.cmd = [
            FFMPEG, "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "17",
            path,
        ]
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame_u8: np.ndarray) -> None:
        self.proc.stdin.write(frame_u8.tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            err = self.proc.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"ffmpeg encode failed: {err}")


def _mux_audio(video_path: str, src_with_audio: str, out_path: str) -> None:
    """Copy audio from src into video_path → out_path."""
    cmd = [
        FFMPEG, "-v", "error", "-y",
        "-i", video_path, "-i", src_with_audio,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest", out_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"audio mux failed: {res.stderr.strip()}")


def upscale_video(
    input_path: str,
    output_path: str,
    model_key: str,
    models_dir: str,
    progress: Callable[[int, int], None] | None = None,
    cancel_event: Event | None = None,
    set_proc: Callable[[Any | None], None] | None = None,
    batch_size: int | None = None,
) -> VideoMeta:
    """Decode → chunked upscale → encode → mux audio. Returns input meta."""
    meta = probe(input_path)
    pipe = get_pipeline(model_key, models_dir)
    scale = REGISTRY[model_key].scale

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    # Whole-file backends (e.g. SeedVR2 via subprocess) handle decode+encode internally.
    try:
        pipe.upscale_video_file(
            input_path, output_path,
            progress=progress, total_frames=meta.total_frames,
            cancel_event=cancel_event, set_proc=set_proc,
            batch_size=batch_size,
        )
        return meta
    except NotImplementedError:
        pass

    out_w, out_h = meta.width * scale, meta.height * scale

    tmp_video = output_path + ".noaudio.mp4"
    writer = _FrameWriter(tmp_video, out_w, out_h, meta.fps)

    chunk: list[np.ndarray] = []
    done = 0

    def flush() -> None:
        nonlocal done
        if not chunk:
            return
        batch = np.stack(chunk, axis=0)  # (T, H, W, C)
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().cuda().half()
        try:
            out = pipe.upscale_chunk(t)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise RuntimeError(f"GPU OOM during video upscale: {e}") from e
        out_np = (
            out.clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy() * 255.0
        ).astype(np.uint8)
        for f in out_np:
            writer.write(f)
        done += len(chunk)
        chunk.clear()
        torch.cuda.empty_cache()
        if progress:
            progress(done, meta.total_frames)

    try:
        for frame in _decode_frames(input_path, meta.width, meta.height):
            if _cancelled():
                raise RuntimeError("cancelled")
            chunk.append(frame)
            if len(chunk) >= CHUNK_FRAMES:
                flush()
        flush()
    finally:
        writer.close()

    if _cancelled():
        try:
            os.remove(tmp_video)
        except OSError:
            pass
        raise RuntimeError("cancelled")

    if meta.has_audio:
        _mux_audio(tmp_video, input_path, output_path)
        try:
            os.remove(tmp_video)
        except OSError:
            pass
    else:
        os.replace(tmp_video, output_path)

    return meta
