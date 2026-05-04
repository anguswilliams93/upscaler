"""Unit tests for upskale.py CLI tool.

Mocks GPU + inference so tests run without CUDA. Real PyTorch may be present
(see conftest.py for stubbing fallback) — we still mock the heavy code paths.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# --------------------------------------------------------------- import guard

@pytest.fixture(autouse=True)
def _isolate_models_dir(tmp_path, monkeypatch):
    """Point upskale at a tmp models dir so user_models.json never pollutes
    the real project."""
    fake_models = tmp_path / "models"
    fake_models.mkdir()
    monkeypatch.setenv("MODELS_DIR", str(fake_models))
    # Re-import upskale fresh so MODELS_DIR is picked up at module import time
    # for tests that exercise main(). Since module already imported once,
    # patch the constants directly.
    if "upskale" in sys.modules:
        import upskale
        monkeypatch.setattr(upskale, "MODELS_DIR", str(fake_models))
        monkeypatch.setattr(
            upskale, "USER_MODELS_FILE", str(fake_models / "_user_models.json"),
        )
    yield


# --------------------------------------------------------------- pure helpers

class TestDetectKind:
    @pytest.mark.parametrize("name", ["a.jpg", "a.JPEG", "a.png", "a.webp"])
    def test_image_extensions(self, name):
        from upskale import _detect_kind
        assert _detect_kind(Path(name)) == "image"

    @pytest.mark.parametrize("name", ["a.mp4", "a.MOV", "a.webm", "a.mkv"])
    def test_video_extensions(self, name):
        from upskale import _detect_kind
        assert _detect_kind(Path(name)) == "video"

    @pytest.mark.parametrize("name", ["a.txt", "a.gif", "noext"])
    def test_unknown_raises(self, name):
        from upskale import _detect_kind
        with pytest.raises(ValueError, match="Unsupported extension"):
            _detect_kind(Path(name))


class TestSanitizeKey:
    @pytest.mark.parametrize("raw,expected", [
        ("MyModel", "mymodel"),
        ("4x_Foo!Bar", "4x_foo-bar"),
        ("  Spaces Here  ", "spaces-here"),
        ("...weird---", "weird"),
    ])
    def test_sanitize(self, raw, expected):
        from upskale import _sanitize_key
        assert _sanitize_key(raw) == expected

    def test_empty_falls_back(self):
        from upskale import _sanitize_key
        assert _sanitize_key("---") == "model"


class TestDefaultOutput:
    def test_image_becomes_png(self, tmp_path):
        from upskale import _default_output
        p = tmp_path / "cat.jpg"
        out = _default_output(p)
        assert out.name == "cat_upscaled.png"
        assert out.parent == tmp_path

    def test_video_keeps_extension(self, tmp_path):
        from upskale import _default_output
        p = tmp_path / "clip.mp4"
        assert _default_output(p).name == "clip_upscaled.mp4"

    def test_video_mov(self, tmp_path):
        from upskale import _default_output
        p = tmp_path / "trailer.mov"
        assert _default_output(p).name == "trailer_upscaled.mov"


class TestAsciiBar:
    def test_zero(self):
        from upskale import _ascii_bar
        bar = _ascii_bar(0.0, 10)
        assert bar == "[----------]"

    def test_full(self):
        from upskale import _ascii_bar
        bar = _ascii_bar(1.0, 10)
        assert bar == "[##########]"

    def test_half(self):
        from upskale import _ascii_bar
        bar = _ascii_bar(0.5, 10)
        assert bar.count("#") == 5
        assert bar.count("-") == 5

    def test_clamp_above_one(self):
        from upskale import _ascii_bar
        assert _ascii_bar(2.0, 4) == "[####]"

    def test_clamp_negative(self):
        from upskale import _ascii_bar
        assert _ascii_bar(-1.0, 4) == "[----]"


class TestFmtEta:
    @pytest.mark.parametrize("seconds,expected", [
        (0, "--:--"),
        (-5, "--:--"),
        (1, "00:01"),
        (59, "00:59"),
        (60, "01:00"),
        (125, "02:05"),
    ])
    def test_format(self, seconds, expected):
        from upskale import _fmt_eta
        assert _fmt_eta(seconds) == expected

    def test_nan_returns_dashes(self):
        from upskale import _fmt_eta
        assert _fmt_eta(float("nan")) == "--:--"


# --------------------------------------------------------------- prompt helpers

class TestAsk:
    def test_returns_stripped_input(self, monkeypatch):
        from upskale import _ask
        monkeypatch.setattr("builtins.input", lambda _p: "  hello  ")
        assert _ask("label", "ph") == "hello"

    def test_strips_drag_drop_quotes(self, monkeypatch):
        from upskale import _ask
        monkeypatch.setattr("builtins.input", lambda _p: '"C:/path with space.png"')
        assert _ask("label", "ph") == "C:/path with space.png"

    def test_empty_returns_default(self, monkeypatch):
        from upskale import _ask
        monkeypatch.setattr("builtins.input", lambda _p: "")
        assert _ask("label", "ph", default="fallback") == "fallback"

    def test_empty_no_default_returns_empty(self, monkeypatch):
        from upskale import _ask
        monkeypatch.setattr("builtins.input", lambda _p: "")
        assert _ask("label", "ph") == ""

    def test_eof_returns_default(self, monkeypatch):
        from upskale import _ask
        def _raise(_):
            raise EOFError
        monkeypatch.setattr("builtins.input", _raise)
        assert _ask("label", "ph", default="x") == "x"


class TestInteractivePrompt:
    def test_non_tty_passthrough(self, monkeypatch):
        from upskale import interactive_prompt
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        ns = argparse.Namespace(input=None, model=None, output=None)
        assert interactive_prompt(ns) is ns
        assert ns.input is None  # untouched

    def test_fills_missing_fields(self, monkeypatch, tmp_path):
        from upskale import interactive_prompt, DEFAULT_MODEL
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")
        responses = iter([str(img), "", ""])  # input, model (default), output (default)
        monkeypatch.setattr("builtins.input", lambda _p: next(responses))
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        ns = argparse.Namespace(input=None, model=None, output=None)
        out = interactive_prompt(ns)
        assert out.input == str(img)
        assert out.model == DEFAULT_MODEL
        assert out.output.endswith("x_upscaled.png")

    def test_video_input_no_default_model(self, monkeypatch, tmp_path):
        from upskale import interactive_prompt
        vid = tmp_path / "y.mp4"
        vid.write_bytes(b"fake")
        responses = iter([str(vid), "flashvsr-v11", ""])
        monkeypatch.setattr("builtins.input", lambda _p: next(responses))
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        ns = argparse.Namespace(input=None, model=None, output=None)
        out = interactive_prompt(ns)
        assert out.model == "flashvsr-v11"
        assert out.output.endswith("y_upscaled.mp4")

    def test_no_input_aborts(self, monkeypatch):
        from upskale import interactive_prompt
        monkeypatch.setattr("builtins.input", lambda _p: "")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        ns = argparse.Namespace(input=None, model=None, output=None)
        with pytest.raises(SystemExit, match="aborted"):
            interactive_prompt(ns)


# --------------------------------------------------------------- add-model flow

class TestAddModelFromUrl:
    def test_rejects_non_http(self):
        from upskale import add_model_from_url
        with pytest.raises(ValueError, match="must be http"):
            add_model_from_url("file:///x.pth")

    def test_rejects_bad_extension(self):
        from upskale import add_model_from_url
        with pytest.raises(ValueError, match="extension"):
            add_model_from_url("https://example.com/foo.zip")

    def test_rejects_no_filename(self):
        from upskale import add_model_from_url
        with pytest.raises(ValueError, match="filename"):
            add_model_from_url("https://example.com/")

    def test_downloads_and_registers(self, monkeypatch, tmp_path):
        import upskale
        # Fake response object exposing read() and headers
        class _Resp:
            headers = {"Content-Length": "8"}
            _data = [b"abcd", b"efgh", b""]
            def read(self, _n):
                return self._data.pop(0)
            def __enter__(self): return self
            def __exit__(self, *_a): return False

        monkeypatch.setattr(
            upskale.urllib.request, "urlopen",
            lambda *_a, **_k: _Resp(),
        )
        info = upskale.add_model_from_url(
            "https://example.com/MyModel.pth",
        )
        assert info.key == "mymodel"
        assert info.weight_file == "MyModel.pth"
        assert info.scale == 4
        # File written
        assert (Path(upskale.MODELS_DIR) / "MyModel.pth").is_file()
        # Sidecar JSON written
        sidecar = json.loads(Path(upskale.USER_MODELS_FILE).read_text())
        assert any(e["key"] == "mymodel" for e in sidecar)
        # Registered in IMG_REGISTRY
        assert "mymodel" in upskale.IMG_REGISTRY
        # Cleanup
        del upskale.IMG_REGISTRY["mymodel"]

    def test_skips_download_if_present(self, monkeypatch):
        import upskale
        existing = Path(upskale.MODELS_DIR) / "Already.pth"
        existing.write_bytes(b"x")
        called = []
        monkeypatch.setattr(
            upskale.urllib.request, "urlopen",
            lambda *_a, **_k: called.append(1) or (_ for _ in ()).throw(AssertionError("must not download")),
        )
        info = upskale.add_model_from_url(
            "https://example.com/Already.pth", key="already",
        )
        assert called == []
        assert info.key == "already"
        del upskale.IMG_REGISTRY["already"]


class TestLoadUserModels:
    def test_loads_sidecar_into_registry(self, monkeypatch):
        import upskale
        sidecar = Path(upskale.USER_MODELS_FILE)
        sidecar.write_text(json.dumps([{
            "key": "user-x",
            "name": "User X",
            "scale": 2,
            "best_for": "test",
            "weight_file": "user-x.pth",
            "download_url": None,
        }]))
        upskale._load_user_models()
        assert "user-x" in upskale.IMG_REGISTRY
        assert upskale.IMG_REGISTRY["user-x"].scale == 2
        del upskale.IMG_REGISTRY["user-x"]

    def test_missing_sidecar_noop(self):
        import upskale
        # File doesn't exist by default in this fixture
        upskale._load_user_models()  # should not raise

    def test_corrupt_sidecar_warns_not_raises(self, monkeypatch, capsys):
        import upskale
        Path(upskale.USER_MODELS_FILE).write_text("not json{{")
        upskale._load_user_models()
        captured = capsys.readouterr()
        assert "warn" in captured.err.lower()


# --------------------------------------------------------------- resolve model

class TestResolveModel:
    def test_image_known_model_passes(self):
        from upskale import _resolve_model
        assert _resolve_model("ultrasharp", "image") == "ultrasharp"

    def test_video_model_on_image_input_errors(self):
        from upskale import _resolve_model
        with pytest.raises(SystemExit, match="video model"):
            _resolve_model("seedvr2-7b", "image")

    def test_image_model_on_video_input_errors(self):
        from upskale import _resolve_model
        with pytest.raises(SystemExit, match="image model"):
            _resolve_model("ultrasharp", "video")

    def test_unknown_video_errors(self):
        from upskale import _resolve_model
        with pytest.raises(SystemExit, match="unknown video"):
            _resolve_model("nope", "video")

    def test_unknown_image_no_tty_errors(self, monkeypatch):
        from upskale import _resolve_model
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        with pytest.raises(SystemExit):
            _resolve_model("nope", "image")


# --------------------------------------------------------------- listing

class TestCmdList:
    def test_runs_and_prints_both_registries(self, capsys):
        from upskale import cmd_list
        rc = cmd_list()
        out = capsys.readouterr().out
        assert rc == 0
        assert "Image models" in out
        assert "Video models" in out
        assert "ultrasharp" in out
        assert "seedvr2-7b" in out


# --------------------------------------------------------------- parser

class TestParser:
    def test_basic_parse(self):
        from upskale import build_parser
        ns = build_parser().parse_args(["foo.png", "-m", "ultrasharp"])
        assert ns.input == "foo.png"
        assert ns.model == "ultrasharp"
        assert ns.batch == 0

    def test_list_flag(self):
        from upskale import build_parser
        ns = build_parser().parse_args(["--list"])
        assert ns.list is True

    def test_add_model(self):
        from upskale import build_parser
        ns = build_parser().parse_args([
            "--add-model", "https://x/y.pth", "--key", "k", "--scale", "2",
        ])
        assert ns.add_model == "https://x/y.pth"
        assert ns.key == "k"
        assert ns.scale == 2


# --------------------------------------------------------------- gpu preflight

class TestGpuPreflight:
    def test_raises_when_cuda_unavailable(self, monkeypatch):
        import upskale
        monkeypatch.setattr(upskale.torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDA not available"):
            upskale.gpu_preflight()

    def test_invalid_device_raises(self, monkeypatch):
        import upskale
        monkeypatch.setattr(upskale.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(upskale.torch.cuda, "device_count", lambda: 1)
        with pytest.raises(RuntimeError, match="not found"):
            upskale.gpu_preflight(device_idx=5)


# --------------------------------------------------------------- run_image

class TestRunImage:
    def test_runs_with_mocked_inference(self, monkeypatch, tmp_path):
        import upskale
        from PIL import Image
        import numpy as np
        # Real image input
        src = tmp_path / "in.png"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(src)
        out_path = tmp_path / "out.png"

        # Fake upscale: return a 4x larger zero tensor (CHW float)
        import torch
        fake_out = torch.zeros(3, 32, 32, dtype=torch.float32)
        monkeypatch.setattr(upskale, "image_upscale", lambda *_a, **_k: fake_out)
        monkeypatch.setattr(
            upskale, "ensure_weight",
            lambda *_a, **_k: str(tmp_path / "weights.pth"),
        )
        rc = upskale.run_image(src, "ultrasharp", out_path)
        assert rc == 0
        assert out_path.is_file()
        assert Image.open(out_path).size == (32, 32)


# --------------------------------------------------------------- run_video

class TestRunVideo:
    def test_missing_weights_returns_2(self, monkeypatch, tmp_path):
        import upskale
        src = tmp_path / "in.mp4"
        src.write_bytes(b"fake")
        out = tmp_path / "out.mp4"
        monkeypatch.setattr(
            upskale, "verify_weights_present",
            lambda *_a, **_k: ["missing.safetensors"],
        )
        rc = upskale.run_video(src, "flashvsr-v11", out, batch_size=None)
        assert rc == 2


# --------------------------------------------------------------- main entry

class TestMain:
    def test_list_returns_zero(self, capsys):
        from upskale import main
        assert main(["--list"]) == 0
        out = capsys.readouterr().out
        assert "Image models" in out

    def test_no_input_no_tty_returns_2(self, monkeypatch, capsys):
        from upskale import main
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        assert main([]) == 2
        err = capsys.readouterr().err
        assert "input path required" in err

    def test_missing_input_file_returns_2(self, monkeypatch, capsys):
        from upskale import main
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        assert main(["does_not_exist.png"]) == 2
        err = capsys.readouterr().err
        assert "not found" in err

    def test_unsupported_extension_returns_2(
        self, monkeypatch, capsys, tmp_path,
    ):
        from upskale import main
        bad = tmp_path / "x.txt"
        bad.write_text("hi")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        assert main([str(bad)]) == 2
        err = capsys.readouterr().err
        # Either "Unsupported extension" (per-file skip) or
        # "no valid image/video" (batch summary) is acceptable.
        assert "Unsupported extension" in err or "no valid" in err

    def test_full_image_pipeline_with_mocks(self, monkeypatch, tmp_path):
        """End-to-end main() invocation, all GPU paths mocked."""
        import upskale
        from PIL import Image
        import numpy as np
        import torch
        src = tmp_path / "tiny.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(src)
        out = tmp_path / "tiny_out.png"
        fake = torch.zeros(3, 16, 16, dtype=torch.float32)
        monkeypatch.setattr(upskale, "gpu_preflight", lambda *_a, **_k: None)
        monkeypatch.setattr(upskale, "image_upscale", lambda *_a, **_k: fake)
        monkeypatch.setattr(
            upskale, "ensure_weight",
            lambda *_a, **_k: str(tmp_path / "w.pth"),
        )
        rc = upskale.main([str(src), "-m", "ultrasharp", "-o", str(out)])
        assert rc == 0
        assert out.is_file()


# --------------------------------------------------------------- _iter_paths

class TestIterPaths:
    def test_single_valid_file(self, tmp_path):
        from upskale import _iter_paths
        f = tmp_path / "a.png"
        f.write_bytes(b"x")
        valid, skipped = _iter_paths(f)
        assert valid == [f]
        assert skipped == []

    def test_single_invalid_file(self, tmp_path):
        from upskale import _iter_paths
        f = tmp_path / "a.txt"
        f.write_text("x")
        valid, skipped = _iter_paths(f)
        assert valid == []
        assert len(skipped) == 1
        assert "Unsupported extension" in skipped[0]

    def test_directory_mixed(self, tmp_path):
        from upskale import _iter_paths
        (tmp_path / "good.png").write_bytes(b"x")
        (tmp_path / "video.mp4").write_bytes(b"x")
        (tmp_path / "bad.txt").write_text("x")
        (tmp_path / "notes.md").write_text("x")
        (tmp_path / "subdir").mkdir()
        valid, skipped = _iter_paths(tmp_path)
        names = sorted(p.name for p in valid)
        assert names == ["good.png", "video.mp4"]
        assert len(skipped) == 2
        assert any("bad.txt" in s for s in skipped)
        assert any("notes.md" in s for s in skipped)

    def test_nonexistent(self, tmp_path):
        from upskale import _iter_paths
        valid, skipped = _iter_paths(tmp_path / "nope")
        assert valid == []
        assert skipped == [f"{tmp_path / 'nope'}: not a file or directory"]

    def test_empty_directory(self, tmp_path):
        from upskale import _iter_paths
        valid, skipped = _iter_paths(tmp_path)
        assert valid == []
        assert skipped == []


# --------------------------------------------------------------- batch + loop

class TestBatchAndLoop:
    def _setup(self, monkeypatch, tmp_path):
        import upskale
        from PIL import Image
        import numpy as np
        import torch
        fake = torch.zeros(3, 16, 16, dtype=torch.float32)
        monkeypatch.setattr(upskale, "gpu_preflight", lambda *_a, **_k: None)
        monkeypatch.setattr(upskale, "image_upscale", lambda *_a, **_k: fake)
        monkeypatch.setattr(
            upskale, "ensure_weight",
            lambda *_a, **_k: str(tmp_path / "w.pth"),
        )
        return Image, np

    def test_folder_processes_valid_skips_invalid(
        self, monkeypatch, tmp_path, capsys,
    ):
        Image, np = self._setup(monkeypatch, tmp_path)
        src_dir = tmp_path / "in"
        src_dir.mkdir()
        for n in ("a.png", "b.png"):
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(src_dir / n)
        (src_dir / "junk.txt").write_text("x")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        from upskale import main
        rc = main([str(src_dir), "-m", "ultrasharp"])
        assert rc == 0
        # Outputs land next to inputs by default
        assert (src_dir / "a_upscaled.png").is_file()
        assert (src_dir / "b_upscaled.png").is_file()
        captured = capsys.readouterr()
        assert "junk.txt" in captured.err
        assert "skip" in captured.err.lower()

    def test_folder_all_invalid_returns_2(self, monkeypatch, tmp_path, capsys):
        self._setup(monkeypatch, tmp_path)
        src_dir = tmp_path / "in"
        src_dir.mkdir()
        (src_dir / "x.txt").write_text("x")
        (src_dir / "y.md").write_text("x")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        from upskale import main
        assert main([str(src_dir), "-m", "ultrasharp"]) == 2
        err = capsys.readouterr().err
        assert "no valid" in err

    def test_folder_kind_mismatch_skips(self, monkeypatch, tmp_path, capsys):
        Image, np = self._setup(monkeypatch, tmp_path)
        src_dir = tmp_path / "in"
        src_dir.mkdir()
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(src_dir / "ok.png")
        (src_dir / "movie.mp4").write_bytes(b"fake")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        from upskale import main
        rc = main([str(src_dir), "-m", "ultrasharp"])
        assert rc == 0
        assert (src_dir / "ok_upscaled.png").is_file()
        err = capsys.readouterr().err
        assert "movie.mp4" in err
        assert "video-only" in err or "image-only" in err

    def test_batch_warns_when_o_given(self, monkeypatch, tmp_path, capsys):
        Image, np = self._setup(monkeypatch, tmp_path)
        src_dir = tmp_path / "in"
        src_dir.mkdir()
        for n in ("a.png", "b.png"):
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(src_dir / n)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        from upskale import main
        out = tmp_path / "ignored.png"
        main([str(src_dir), "-m", "ultrasharp", "-o", str(out)])
        err = capsys.readouterr().err
        assert "-o ignored" in err
        # -o should NOT be used; per-file defaults instead
        assert not out.is_file()
        assert (src_dir / "a_upscaled.png").is_file()

    def test_interactive_loop_processes_then_quits(
        self, monkeypatch, tmp_path, capsys,
    ):
        Image, np = self._setup(monkeypatch, tmp_path)
        img1 = tmp_path / "one.png"
        img2 = tmp_path / "two.png"
        for p in (img1, img2):
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(p)
        # Sequence of `input()` calls during the run:
        #   interactive_prompt: input, model(default), output(default)
        #   loop: "next" -> img2
        #   interactive doesn't re-prompt (model still set), output cleared so
        #     _process_input uses _default_output(p)
        #   loop: "next" -> "" to quit
        responses = iter([
            str(img1), "", "",   # first interactive prompt
            str(img2),           # next path
            "",                  # quit
        ])
        monkeypatch.setattr("builtins.input", lambda _p: next(responses))
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        from upskale import main
        rc = main([])
        assert rc == 0
        assert (tmp_path / "one_upscaled.png").is_file()
        assert (tmp_path / "two_upscaled.png").is_file()
        out = capsys.readouterr().out
        assert "bye" in out
