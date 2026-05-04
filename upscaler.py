"""GPU inference: model loading + upscale function. RTX 5090 fp16."""
from __future__ import annotations

import logging
from typing import Any

import torch
from spandrel import ModelLoader

logger = logging.getLogger(__name__)

_model_cache: dict[str, Any] = {}


def get_model(model_key: str, model_path: str) -> Any:
    """Load model into VRAM on first use, then reuse."""
    if model_key not in _model_cache:
        logger.info("Loading model %s from %s", model_key, model_path)
        descriptor = ModelLoader().load_from_file(model_path)
        model = descriptor.model.cuda().eval().half()
        _model_cache[model_key] = model
    return _model_cache[model_key]


def upscale(
    image_tensor: torch.Tensor,
    model_key: str,
    model_path: str,
) -> torch.Tensor:
    """
    Run upscale inference on GPU.

    Args:
        image_tensor: float32 CHW tensor in [0, 1] on CPU.
        model_key:    registry key.
        model_path:   absolute path to weight file.

    Returns:
        float32 CHW tensor in [0, 1] on CPU.
    """
    model = get_model(model_key, model_path)
    inp = image_tensor.unsqueeze(0).cuda().half()
    try:
        with torch.inference_mode():
            out = model(inp)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise RuntimeError(f"GPU out of memory during upscale: {e}") from e
    finally:
        torch.cuda.empty_cache()
    return out.squeeze(0).clamp(0, 1).float().cpu()
