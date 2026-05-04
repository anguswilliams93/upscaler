"""Test bootstrap: stub heavy GPU deps so the suite runs without CUDA installed.

CLAUDE.md forbids CPU PyTorch for inference and requires GPU mocking in tests.
This stubs `torch` and `spandrel` at import time so `main` / `upscaler` modules
can be imported on any machine. Real GPU runs use the actual CUDA wheels.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_torch() -> ModuleType:
    torch = ModuleType("torch")

    class _Tensor:
        def __init__(self, shape=(3, 8, 8)):
            self.shape = tuple(shape)

        def unsqueeze(self, _dim): return _Tensor((1,) + self.shape)
        def squeeze(self, _dim=0): return _Tensor(self.shape[1:] if self.shape[0] == 1 else self.shape)
        def cuda(self): return self
        def half(self): return self
        def float(self): return self
        def cpu(self): return self
        def clamp(self, *_a, **_k): return self
        def permute(self, *_a): return self
        def contiguous(self): return self
        @property
        def ndim(self): return len(self.shape)

    torch.Tensor = _Tensor
    torch.float16 = "fp16"
    torch.float32 = "fp32"

    class _OOM(RuntimeError): pass

    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda _i=0: "MOCK GPU",
        empty_cache=lambda: None,
        memory_allocated=lambda _i=0: 1_073_741_824,  # 1 GB
        get_device_properties=lambda _i=0: SimpleNamespace(total_memory=34_359_738_368),  # 32 GB
        OutOfMemoryError=_OOM,
    )
    torch.cuda = cuda
    torch.OutOfMemoryError = _OOM

    class _InferenceMode:
        def __enter__(self): return self
        def __exit__(self, *_a): return False

    torch.inference_mode = lambda: _InferenceMode()
    torch.from_numpy = lambda arr: _Tensor(arr.shape)
    return torch


def _make_spandrel() -> ModuleType:
    spandrel = ModuleType("spandrel")
    spandrel.ModelLoader = MagicMock()
    return spandrel


# Stub only when the real package isn't installed (e.g. CI without CUDA wheels).
# When real torch/spandrel are available, use them — tests still mock GPU calls.
try:
    import torch  # noqa: F401
except ImportError:
    sys.modules["torch"] = _make_torch()

try:
    import spandrel  # noqa: F401
except ImportError:
    sys.modules["spandrel"] = _make_spandrel()
