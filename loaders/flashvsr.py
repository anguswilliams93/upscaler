"""FlashVSR v1.1 loader — wraps vendor/flashvsr/cli_main.py via subprocess.

Upstream package's __init__.py imports ComfyUI's `folder_paths`, so we cannot
import the package. cli_main.py is the standalone entry point.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch

from .base import VideoUpscalerBase

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "flashvsr"
CLI_SCRIPT = VENDOR_DIR / "cli_main.py"

_FRAC_RE = re.compile(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?!\d)")
_SKIP_FRAC = ("Tip:", "💡", "Loading", "PyTorch", "cuDNN", "Initial", "VRAM")
# Hide noisy lines from UI msg (still logged).
_HIDE_MSG = (
    "VAE decoding", "VAE encoding",
    "Wan2.2 VAE", "LightX2V VAE",
    "it/s]", "?it/s]",
)


class FlashVSRUpscaler(VideoUpscalerBase):
    scale = 4

    def load(self) -> None:
        if not CLI_SCRIPT.is_file():
            raise RuntimeError(f"FlashVSR CLI missing at {CLI_SCRIPT}")
        for name, path in self.weight_paths.items():
            if not os.path.isfile(path):
                raise RuntimeError(f"FlashVSR weight missing: {path}")
        self._loaded = True

    def upscale_chunk(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "FlashVSR runs whole-file via cli_main.py — use upscale_video_file()."
        )

    def upscale_video_file(
        self,
        input_path: str,
        output_path: str,
        progress: Callable[[int, int, str | None], None] | None = None,
        total_frames: int | None = None,
        cancel_event=None,
        set_proc=None,
        batch_size: int | None = None,  # ignored: FlashVSR streams internally
    ) -> None:
        if not self._loaded:
            self.load()

        cmd = [
            sys.executable, "-u", str(CLI_SCRIPT),
            "--input", str(Path(input_path).resolve()),
            "--output", str(Path(output_path).resolve()),
            "--model", "FlashVSR-v1.1",
            "--mode", "tiny",
            "--vae_model", "Wan2.1",
            "--scale", "4",
        ]
        logger.info("FlashVSR CLI: %s", " ".join(cmd))

        total = max(1, int(total_frames or 1))
        if progress:
            progress(0, total, "starting FlashVSR…")

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
            bufsize=0,
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
            logger.info("[flashvsr] %s", line)
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
            msg = re.sub(r"\s{2,}", " ", msg).strip(" :-|")
            if msg:
                progress(last_done, total, msg[:160])

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
            raise RuntimeError(f"FlashVSR CLI failed (exit {rc})")

        if progress:
            progress(total, total, "done")
        torch.cuda.empty_cache()
