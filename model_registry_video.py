"""Registry for video super-resolution models (SeedVR2, FlashVSR).

Weights live alongside their vendor directories:
  vendor/seedvr2/models/SeedVR2/*.safetensors
  vendor/flashvsr/models/FlashVSR/*.{safetensors,ckpt,pth}
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = REPO_ROOT / "vendor"


@dataclass(frozen=True)
class VideoModelInfo:
    key: str
    name: str
    scale: int
    best_for: str
    subdir: str                       # path relative to MODELS_DIR (legacy)
    weight_files: tuple[str, ...]     # all required weights
    loader: str                       # "seedvr2" | "flashvsr"
    vram_gb: float                    # est. peak VRAM at 720p input
    max_input_height: int = 720       # hard cap
    vendor_subdir: str | None = None  # path under vendor/ where vendor CLI looks


REGISTRY: dict[str, VideoModelInfo] = {
    "seedvr2-7b": VideoModelInfo(
        key="seedvr2-7b",
        name="SeedVR2 7B (sharp)",
        scale=4,
        best_for="High quality, slow. Best for short clips.",
        subdir="SeedVR2",
        weight_files=(
            "seedvr2_ema_7b_sharp_fp16.safetensors",
            "ema_vae_fp16.safetensors",
        ),
        loader="seedvr2",
        vram_gb=25.0,
        vendor_subdir="seedvr2/models/SeedVR2",
    ),
    "flashvsr-v11": VideoModelInfo(
        key="flashvsr-v11",
        name="FlashVSR v1.1",
        scale=4,
        best_for="Fast streaming VSR. Lower VRAM.",
        subdir="FlashVSR",
        weight_files=(
            "diffusion_pytorch_model_streaming_dmd.safetensors",
            "LQ_proj_in.ckpt",
            "TCDecoder.ckpt",
            "Wan2.1_VAE.pth",
        ),
        loader="flashvsr",
        vram_gb=12.0,
        vendor_subdir="flashvsr/models/FlashVSR",
    ),
}


def _candidate_dirs(info: VideoModelInfo, models_dir: str) -> list[Path]:
    """Locations to search for weights, in priority order."""
    dirs: list[Path] = []
    if info.vendor_subdir:
        dirs.append(VENDOR_DIR / info.vendor_subdir)
    dirs.append(Path(models_dir) / info.subdir)
    return dirs


def get_weight_paths(key: str, models_dir: str) -> dict[str, str]:
    """Resolve each weight to the first existing candidate dir, else first dir."""
    info = REGISTRY[key]
    candidates = _candidate_dirs(info, models_dir)
    out: dict[str, str] = {}
    for wf in info.weight_files:
        chosen = next(
            (d / wf for d in candidates if (d / wf).is_file()),
            candidates[0] / wf,
        )
        out[wf] = str(chosen)
    return out


def list_video_models() -> list[dict[str, object]]:
    return [
        {
            "key": m.key,
            "name": m.name,
            "scale": m.scale,
            "best_for": m.best_for,
            "vram_gb": m.vram_gb,
            "max_input_height": m.max_input_height,
        }
        for m in REGISTRY.values()
    ]


def verify_weights_present(key: str, models_dir: str) -> list[str]:
    """Return list of missing weight filenames (empty if all present)."""
    paths = get_weight_paths(key, models_dir)
    return [name for name, p in paths.items() if not os.path.isfile(p)]
