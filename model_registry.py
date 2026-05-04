"""Registry mapping model keys to weight files and metadata."""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ModelInfo:
    key: str
    name: str
    scale: int
    best_for: str
    weight_file: str
    download_url: str | None = None


REGISTRY: dict[str, ModelInfo] = {
    "ultrasharp": ModelInfo(
        "ultrasharp", "4x UltraSharp", 4,
        "General photos (JPEG-tuned)", "4x-UltraSharp.pth",
        "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
    ),
    "remacri": ModelInfo(
        "remacri", "4x Remacri", 4,
        "Photos (softer sharpening)", "4x_foolhardy_Remacri.pth",
        "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
    ),
    "drct-l": ModelInfo(
        "drct-l", "4x Nomos2 DRCT-L", 4,
        "Clean photos — best quality", "4xNomos2_hq_drct-l.pth",
        "https://github.com/Phhofm/models/releases/download/4xNomos2_hq_drct-l/4xNomos2_hq_drct-l.pth",
    ),
    "nomos-atd-jpg": ModelInfo(
        "nomos-atd-jpg", "4x Nomos8k ATD JPG", 4,
        "Degraded / JPEG-compressed", "4xNomos8k_atd_jpg.pth",
        "https://github.com/Phhofm/models/releases/download/4xNomos8k_atd_jpg/4xNomos8k_atd_jpg.pth",
    ),
    "realesrgan": ModelInfo(
        "realesrgan", "Real-ESRGAN x4plus", 4,
        "General purpose", "RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    "realesrgan-anime": ModelInfo(
        "realesrgan-anime", "Real-ESRGAN Anime 6B", 4,
        "Anime / illustration", "RealESRGAN_x4plus_anime_6B.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
    "realesrgan-general": ModelInfo(
        "realesrgan-general", "Real-ESRGAN General v3", 4,
        "Mixed / unknown degradation", "realesr-general-x4v3.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    ),
}


def get_model_path(key: str, models_dir: str) -> str:
    """Resolve absolute path to weight file for a registry key."""
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key: {key}")
    return os.path.join(models_dir, REGISTRY[key].weight_file)


def list_models() -> list[dict[str, object]]:
    """Return registry as plain list for API responses."""
    return [
        {"key": m.key, "name": m.name, "scale": m.scale, "best_for": m.best_for}
        for m in REGISTRY.values()
    ]


def ensure_weight(
    key: str,
    models_dir: str,
    log: Callable[[str], None] | None = None,
) -> str:
    """Return path to the weight file, downloading from `download_url` if missing.

    Raises FileNotFoundError if no URL is registered and the file is absent.
    """
    info = REGISTRY[key]
    path = get_model_path(key, models_dir)
    if os.path.isfile(path):
        return path
    if not info.download_url:
        raise FileNotFoundError(
            f"Weight {info.weight_file} missing and no download_url registered."
        )
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    tmp = path + ".part"
    if log:
        log(f"downloading {info.weight_file}")
    with urllib.request.urlopen(info.download_url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        read = 0
        chunk = 1 << 20
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            if log and total:
                log(f"  {read / total * 100:5.1f}%  ({read >> 20}/{total >> 20} MiB)")
    os.replace(tmp, path)
    if log:
        log(f"saved {info.weight_file}")
    return path
