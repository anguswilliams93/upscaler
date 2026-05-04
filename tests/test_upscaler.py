"""Unit tests for upscaler.py — torch + spandrel stubbed in conftest."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch  # stubbed when CUDA absent


@pytest.fixture
def clear_cache():
    from upscaler import _model_cache
    _model_cache.clear()
    yield
    _model_cache.clear()


def test_get_model_loads_once_and_caches(clear_cache) -> None:
    from upscaler import get_model

    fake_model = MagicMock()
    fake_model.cuda.return_value.eval.return_value.half.return_value = fake_model
    descriptor = MagicMock(model=fake_model)

    with patch("upscaler.ModelLoader") as ml:
        ml.return_value.load_from_file.return_value = descriptor
        m1 = get_model("k", "/tmp/x.pth")
        m2 = get_model("k", "/tmp/x.pth")

    assert m1 is m2
    assert ml.return_value.load_from_file.call_count == 1


def test_upscale_returns_cpu_tensor(clear_cache) -> None:
    from upscaler import upscale

    inp = torch.Tensor((3, 8, 8))
    out = torch.Tensor((1, 3, 32, 32))
    fake_model = MagicMock(return_value=out)

    with patch("upscaler.get_model", return_value=fake_model):
        result = upscale(inp, "k", "/tmp/x.pth")

    fake_model.assert_called_once()
    assert hasattr(result, "shape")


def test_upscale_raises_runtimeerror_on_oom(clear_cache) -> None:
    from upscaler import upscale

    inp = torch.Tensor((3, 8, 8))
    fake_model = MagicMock(side_effect=torch.cuda.OutOfMemoryError("simulated OOM"))

    with patch("upscaler.get_model", return_value=fake_model):
        with pytest.raises(RuntimeError, match="out of memory"):
            upscale(inp, "k", "/tmp/x.pth")


def test_upscale_calls_empty_cache_even_on_success(clear_cache) -> None:
    from upscaler import upscale

    inp = torch.Tensor((3, 8, 8))
    fake_model = MagicMock(return_value=torch.Tensor((1, 3, 32, 32)))

    with patch("upscaler.get_model", return_value=fake_model), \
         patch.object(torch.cuda, "empty_cache") as ec:
        upscale(inp, "k", "/tmp/x.pth")

    assert ec.called
