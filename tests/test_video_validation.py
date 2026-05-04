"""Validation tests for video upscale endpoint (no GPU, no real ffmpeg pass)."""
from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="MOCK"):
        from main import app
        with TestClient(app) as c:
            yield c


def _post_video(client, content=b"\x00" * 1024, mime="video/mp4", model="seedvr2-7b"):
    return client.post(
        "/api/upscale-video",
        files={"file": ("clip.mp4", io.BytesIO(content), mime)},
        data={"model": model},
    )


def test_unknown_model_400(client):
    r = _post_video(client, model="bogus")
    assert r.status_code == 400
    assert "Unknown video model" in r.json()["detail"]


def test_bad_mime_400(client):
    assert _post_video(client, mime="text/plain").status_code == 400


def test_missing_weights_503(client, monkeypatch, tmp_path):
    import model_registry_video as mrv
    monkeypatch.setattr(mrv, "VENDOR_DIR", tmp_path / "novendor")
    r = _post_video(client)
    assert r.status_code == 503
    assert "weights missing" in r.json()["detail"].lower()


def test_oversize_413(client, monkeypatch):
    monkeypatch.setattr("main.verify_weights_present", lambda *a, **k: [])
    big = b"\x00" * (501 * 1024 * 1024)  # 501 MB > 500 MB cap
    assert _post_video(client, content=big).status_code == 413


@pytest.mark.parametrize(
    "meta_kwargs,fragment",
    [
        (dict(width=640, height=360, fps=30.0, duration_s=75.0,
              total_frames=2250, has_audio=True), "exceeds"),
        (dict(width=1920, height=1080, fps=30.0, duration_s=10.0,
              total_frames=300, has_audio=False), "height"),
    ],
)
def test_metadata_caps(client, monkeypatch, meta_kwargs, fragment):
    from video_upscaler import VideoMeta

    monkeypatch.setattr("main.verify_weights_present", lambda *a, **k: [])
    monkeypatch.setattr("main.video_probe", lambda p: VideoMeta(**meta_kwargs))
    r = _post_video(client)
    assert r.status_code == 400
    assert fragment in r.json()["detail"]


def test_video_models_endpoint(client):
    r = client.get("/api/video-models")
    assert r.status_code == 200
    keys = {m["key"] for m in r.json()}
    assert {"seedvr2-7b", "flashvsr-v11"} <= keys
