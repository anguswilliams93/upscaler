"""Unit tests for the video model registry."""
from __future__ import annotations

import pytest

from model_registry_video import (
    REGISTRY,
    get_weight_paths,
    list_video_models,
    verify_weights_present,
)


@pytest.mark.parametrize("key", ["seedvr2-7b", "flashvsr-v11"])
def test_required_video_model_present(key: str) -> None:
    assert key in REGISTRY


def test_seedvr2_weights() -> None:
    info = REGISTRY["seedvr2-7b"]
    assert "seedvr2_ema_7b_sharp_fp16.safetensors" in info.weight_files
    assert "ema_vae_fp16.safetensors" in info.weight_files
    assert info.scale == 4
    assert info.max_input_height == 720


def test_flashvsr_weights() -> None:
    info = REGISTRY["flashvsr-v11"]
    assert set(info.weight_files) == {
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
        "Wan2.1_VAE.pth",
    }


def test_get_weight_paths_under_seedvr2_subdir(tmp_path) -> None:
    paths = get_weight_paths("seedvr2-7b", str(tmp_path))
    for fname, p in paths.items():
        assert p.endswith(fname)
        assert "SeedVR2" in p


def test_verify_missing_returns_all_when_absent(tmp_path, monkeypatch) -> None:
    import model_registry_video as mrv
    monkeypatch.setattr(mrv, "VENDOR_DIR", tmp_path / "novendor")
    missing = verify_weights_present("seedvr2-7b", str(tmp_path))
    assert len(missing) == len(REGISTRY["seedvr2-7b"].weight_files)


def test_verify_present_returns_empty(tmp_path) -> None:
    sub = tmp_path / "SeedVR2"
    sub.mkdir()
    for w in REGISTRY["seedvr2-7b"].weight_files:
        (sub / w).write_bytes(b"x")
    assert verify_weights_present("seedvr2-7b", str(tmp_path)) == []


@pytest.mark.parametrize("row", list_video_models())
def test_list_video_models_row_shape(row: dict) -> None:
    assert {"key", "name", "scale", "best_for", "vram_gb", "max_input_height"} <= row.keys()
