"""Registry contract tests — pure Python, no GPU."""
from __future__ import annotations

import os

import pytest

from model_registry import REGISTRY, get_model_path, list_models

REQUIRED_KEYS = [
    "ultrasharp", "remacri", "drct-l", "nomos-atd-jpg",
    "realesrgan", "realesrgan-anime", "realesrgan-general",
]


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_required_key_present(key: str) -> None:
    assert key in REGISTRY


@pytest.mark.parametrize("key,info", list(REGISTRY.items()))
def test_entry_well_formed(key: str, info) -> None:
    assert info.key == key
    assert info.scale in (2, 4)
    assert info.weight_file.endswith((".pth", ".safetensors"))
    assert info.name
    assert info.best_for


def test_list_models_shape() -> None:
    items = list_models()
    assert len(items) == len(REGISTRY)
    for m in items:
        assert m.keys() == {"key", "name", "scale", "best_for"}


def test_get_model_path_resolves() -> None:
    assert get_model_path("ultrasharp", "/models") == os.path.join(
        "/models", "4x-UltraSharp.pth"
    )


def test_get_model_path_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown model key"):
        get_model_path("does-not-exist", "/models")
