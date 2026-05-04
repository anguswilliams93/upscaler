"""SeedVR2 7B loader — wraps vendor/seedvr2/inference_cli.py via subprocess.

Upstream module is ComfyUI-coupled (vendor/seedvr2/__init__.py imports
comfy_api). The standalone inference_cli.py has no ComfyUI deps, so we
shell out to it instead of importing.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

# Match patterns the SeedVR2 CLI emits.
# - Frame N/M / Step N/M / [N/M]
# - tqdm:  " 12/805 [00:..,"
_FRAC_RE = re.compile(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?!\d)")
# Lines that look like noisy banner / tip / emoji-only — skip frac matching there
_SKIP_FRAC = ("Tip:", "💡", "Initial CUDA", "PyTorch", "cuDNN", "Conv3d", "EulerSampler")
# Lines hidden from the UI progress msg entirely (still logged).
_HIDE_MSG = ("EulerSampler",)

import torch

from .base import VideoUpscalerBase

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "seedvr2"
CLI_SCRIPT = VENDOR_DIR / "inference_cli.py"
DIT_WEIGHT = "seedvr2_ema_7b_sharp_fp16.safetensors"


class SeedVR2Upscaler(VideoUpscalerBase):
    scale = 4

    def load(self) -> None:
        if not CLI_SCRIPT.is_file():
            raise RuntimeError(f"SeedVR2 CLI missing at {CLI_SCRIPT}")
        for name, path in self.weight_paths.items():
            if not os.path.isfile(path):
                raise RuntimeError(f"SeedVR2 weight missing: {path}")
        self._loaded = True

    def upscale_chunk(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "SeedVR2 runs whole-file via inference_cli.py — use upscale_video_file()."
        )

    def upscale_video_file(
        self,
        input_path: str,
        output_path: str,
        progress: Callable[[int, int], None] | None = None,
        total_frames: int | None = None,
        cancel_event=None,
        set_proc=None,
        batch_size: int | None = None,
    ) -> None:
        if not self._loaded:
            self.load()

        # All weights share one parent dir (models/SeedVR2/).
        model_dir = str(Path(next(iter(self.weight_paths.values()))).parent)

        # batch_size is the temporal window per diffusion pass, NOT the whole clip.
        # Keep small so tqdm emits per-batch progress and VRAM stays bounded.
        if batch_size is None or batch_size <= 0:
            batch_size = int(os.environ.get("SEEDVR2_BATCH_SIZE", "5"))
        # batch_size == 0 from API means "whole clip"; CLI uses 0/None sentinel via large value
        if batch_size == 0:
            batch_size = max(1, int(total_frames or 1))
        cmd = [
            sys.executable, "-u", str(CLI_SCRIPT),
            str(Path(input_path).resolve()),
            "--output", str(Path(output_path).resolve()),
            "--output_format", "mp4",
            "--video_backend", "ffmpeg",
            "--model_dir", model_dir,
            "--dit_model", DIT_WEIGHT,
            "--resolution", "1080",
            "--batch_size", str(batch_size),
            "--cuda_device", "0",
        ]
        logger.info("SeedVR2 CLI: %s", " ".join(cmd))

        # Use real frame count as the progress denominator until the CLI
        # emits its own N/M, then we follow that.
        total = int(total_frames) if total_frames else batch_size
        if progress:
            progress(0, total, "starting SeedVR2…")

        env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }
        proc = subprocess.Popen(
            cmd,
            cwd=str(VENDOR_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,            # unbuffered — tqdm \r updates flush immediately
            env=env,
        )
        assert proc.stdout is not None
        if set_proc:
            set_proc(proc)
        last_done = 0

        def _emit(line: str) -> None:
            nonlocal last_done, total
            line = line.rstrip()
            if not line:
                return
            logger.info("[seedvr2] %s", line)
            if not progress:
                return
            if not any(skip in line for skip in _SKIP_FRAC):
                m = _FRAC_RE.search(line)
                if m:
                    done, denom = int(m.group(1)), int(m.group(2))
                    if denom > 0:
                        last_done, total = done, denom
            if any(h in line for h in _HIDE_MSG):
                return
            msg = re.sub(r"\x1b\[[0-9;]*m", "", line)
            msg = _FRAC_RE.sub("", msg)
            msg = re.sub(r"\s{2,}", " ", msg).strip(" :-")
            if msg:
                progress(last_done, total, msg[:160])

        # Manually split on either \n or \r so tqdm progress bars are read live.
        buf = bytearray()
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                break
            if chunk in (b"\n", b"\r"):
                if buf:
                    _emit(buf.decode("utf-8", "replace"))
                    buf.clear()
            else:
                buf.extend(chunk)
        if buf:
            _emit(buf.decode("utf-8", "replace"))
        rc = proc.wait()
        if set_proc:
            set_proc(None)
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")
        if rc != 0:
            raise RuntimeError(f"SeedVR2 CLI failed (exit {rc})")

        if progress:
            progress(total, total, "done")
        torch.cuda.empty_cache()
