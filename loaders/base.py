"""Uniform interface for video super-resolution backends."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class VideoUpscalerBase(ABC):
    """All video upscalers expose load() + upscale_chunk()."""

    scale: int = 4

    def __init__(self, weight_paths: dict[str, str]) -> None:
        self.weight_paths = weight_paths
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load all weights into VRAM. Idempotent."""

    @abstractmethod
    def upscale_chunk(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (T, C, H, W) fp16 cuda, values in [0, 1].
        Returns: (T, C, H*scale, W*scale) fp16 cuda, [0, 1].
        """

    def upscale_video_file(
        self,
        input_path: str,
        output_path: str,
        progress=None,  # callable(done:int, total:int, msg:str|None=None)
        total_frames: int | None = None,
        cancel_event=None,  # threading.Event — set to abort
        set_proc=None,      # callable(proc|None) — register subprocess for kill
        batch_size=None,    # temporal window override (loader-specific)
    ) -> None:
        """Optional whole-file path. Backends that wrap external CLIs override.

        Default raises so callers fall back to chunked path.
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Free VRAM (override if backend needs explicit teardown)."""
        torch.cuda.empty_cache()
        self._loaded = False
