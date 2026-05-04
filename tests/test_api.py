"""Integration tests via FastAPI TestClient. torch/spandrel stubbed in conftest."""
from __future__ import annotations

import io

import pytest
from PIL import Image

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main

    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def tiny_png() -> bytes:
    img = Image.new("RGB", (16, 16), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_models_endpoint(client) -> None:
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert any(m["key"] == "ultrasharp" for m in data)
    for m in data:
        assert {"key", "name", "scale", "best_for"} <= m.keys()


def test_gpu_endpoint(client) -> None:
    res = client.get("/api/gpu")
    assert res.status_code == 200
    j = res.json()
    assert {"name", "vram_used_gb", "vram_total_gb", "jobs_done"} <= j.keys()
    assert j["vram_total_gb"] > 0


def test_job_404(client) -> None:
    assert client.get("/api/job/does-not-exist").status_code == 404


@pytest.mark.parametrize(
    "filename,content,mime,model,scale,reason",
    [
        ("t.png", None, "image/png", "nonexistent", 4, "bad model"),
        ("t.png", None, "image/png", "ultrasharp", 3, "bad scale"),
        ("t.txt", b"hello", "text/plain", "ultrasharp", 4, "bad mime"),
    ],
)
def test_upscale_rejects_invalid_input(
    client, tiny_png, filename, content, mime, model, scale, reason
) -> None:
    body = content if content is not None else tiny_png
    res = client.post(
        "/api/upscale",
        files={"file": (filename, body, mime)},
        data={"model": model, "scale": scale},
    )
    assert res.status_code == 400, reason


def test_upscale_accepts_valid_image(client, tiny_png) -> None:
    res = client.post(
        "/api/upscale",
        files={"file": ("t.png", tiny_png, "image/png")},
        data={"model": "ultrasharp", "scale": 4},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "queued"
    assert body["job_id"]
