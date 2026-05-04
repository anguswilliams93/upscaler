"""Video upscaler loader registry."""
from __future__ import annotations

from .base import VideoUpscalerBase
from .flashvsr import FlashVSRUpscaler
from .seedvr2 import SeedVR2Upscaler

LOADERS: dict[str, type[VideoUpscalerBase]] = {
    "seedvr2": SeedVR2Upscaler,
    "flashvsr": FlashVSRUpscaler,
}


def build(loader_key: str, weight_paths: dict[str, str]) -> VideoUpscalerBase:
    if loader_key not in LOADERS:
        raise KeyError(f"Unknown video loader: {loader_key}")
    return LOADERS[loader_key](weight_paths)
