"""UPSKALE - standalone CLI for local image/video upscaling.

Usage:
  python upskale.py                                 # interactive prompt
  python upskale.py <input> [-m MODEL] [-o OUT] [-b BATCH] [--device N]
  python upskale.py --list
  python upskale.py --add-model URL [--key KEY] [--scale N] [--name NAME]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

from model_registry import REGISTRY as IMG_REGISTRY, ModelInfo, ensure_weight, list_models
from model_registry_video import REGISTRY as VID_REGISTRY, list_video_models, verify_weights_present
from upscaler import upscale as image_upscale
from video_upscaler import probe as video_probe, upscale_video

load_dotenv()

MODELS_DIR = os.getenv("UPSKALE_MODELS_DIR", "./models")
OUTPUTS_DIR = os.getenv("UPSKALE_OUTPUTS_DIR", "./outputs")
DEFAULT_MODEL = os.getenv("UPSKALE_DEFAULT_MODEL", "ultrasharp")
USER_MODELS_FILE = os.path.join(MODELS_DIR, "_user_models.json")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}
WEIGHT_EXTS = {".pth", ".safetensors"}

_LIME = "\033[38;2;234;249;114m"
_DIM = "\033[38;2;120;130;60m"
_YELLOW = "\033[38;2;255;215;0m"
_RED = "\033[38;2;255;90;90m"
_BOLD = "\033[1m"
_RST = "\033[0m"

_VERSION = "v0.1.0"
BANNER = (
    f"\n"
    f"{_LIME}^\\      /^{_RST}   {_BOLD}UPSKALE{_RST}  {_DIM}{_VERSION}{_RST}\n"
    f"{_LIME}  \\    /  {_RST}   {_DIM}local super-resolution . Local GPU{_RST}\n"
    f"{_LIME}  /    \\  {_RST}   {_DIM}by {_RST}{_LIME}ZEROBI{_RST}{_DIM}  output > input{_RST}\n"
    f"{_LIME}v/      \\v{_RST}\n"
)

logger = logging.getLogger("upskale")


# ---------------------------------------------------------------- helpers

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _sanitize_key(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    return s.strip("-._") or "model"


# ---------------------------------------------------------------- user models

def _load_user_models() -> None:
    """Merge user-registered models from sidecar JSON into IMG_REGISTRY."""
    if not os.path.isfile(USER_MODELS_FILE):
        return
    try:
        with open(USER_MODELS_FILE, encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        _eprint(f"{_YELLOW}warn: could not read {USER_MODELS_FILE}: {e}{_RST}")
        return
    for e in entries:
        try:
            info = ModelInfo(
                key=e["key"],
                name=e.get("name", e["key"]),
                scale=int(e.get("scale", 4)),
                best_for=e.get("best_for", "User-added model"),
                weight_file=e["weight_file"],
                download_url=e.get("download_url"),
            )
        except KeyError:
            continue
        IMG_REGISTRY[info.key] = info


def _save_user_model(info: ModelInfo) -> None:
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    if os.path.isfile(USER_MODELS_FILE):
        try:
            with open(USER_MODELS_FILE, encoding="utf-8") as f:
                entries = json.load(f)
        except (OSError, json.JSONDecodeError):
            entries = []
    entries = [e for e in entries if e.get("key") != info.key]
    entries.append(asdict(info))
    with open(USER_MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _download_weight(url: str, dest: Path, label: str) -> None:
    """Stream URL to dest atomically, with progress bar."""
    Path(dest.parent).mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"{_LIME}downloading{_RST} {label}")
    print(f"  url: {_DIM}{url}{_RST}")
    print(f"  to:  {_DIM}{dest}{_RST}")
    req = urllib.request.Request(url, headers={"User-Agent": "upzcaler/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        read = 0
        chunk = 1 << 20
        last = 0.0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            now = time.time()
            if total and now - last > 0.2:
                pct = read / total * 100
                bar = _ascii_bar(pct / 100, 30)
                sys.stdout.write(
                    f"\r  {_LIME}{bar}{_RST} {pct:5.1f}%  "
                    f"{read >> 20}/{total >> 20} MiB"
                )
                sys.stdout.flush()
                last = now
    sys.stdout.write("\r" + " " * 78 + "\r")
    os.replace(tmp, dest)
    print(f"  {_LIME}saved{_RST} {dest.name}")


def add_model_from_url(
    url: str,
    key: str | None = None,
    scale: int = 4,
    name: str | None = None,
    best_for: str = "User-added model",
) -> ModelInfo:
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme.startswith("http"):
        raise ValueError(f"URL must be http(s), got: {url}")
    fname = os.path.basename(parsed.path)
    if not fname:
        raise ValueError(f"Cannot derive filename from URL: {url}")
    ext = os.path.splitext(fname)[1].lower()
    if ext not in WEIGHT_EXTS:
        raise ValueError(
            f"Weight extension must be one of {sorted(WEIGHT_EXTS)}, got '{ext}'"
        )
    if key is None:
        key = _sanitize_key(os.path.splitext(fname)[0])
    if name is None:
        name = key
    info = ModelInfo(
        key=key, name=name, scale=scale, best_for=best_for,
        weight_file=fname, download_url=url,
    )
    dest = Path(MODELS_DIR) / fname
    if not dest.is_file():
        _download_weight(url, dest, label=fname)
    else:
        print(f"{_DIM}weight already present:{_RST} {dest}")
    IMG_REGISTRY[info.key] = info
    _save_user_model(info)
    print(f"{_LIME}registered{_RST} key={info.key} scale={info.scale}x")
    return info


# ---------------------------------------------------------------- GPU preflight

def gpu_preflight(device_idx: int | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. RTX 5090 + CUDA 12.8 driver required."
        )
    if device_idx is None:
        env = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        try:
            device_idx = int(env) if env else 0
        except ValueError:
            device_idx = 0
    if device_idx >= torch.cuda.device_count():
        raise RuntimeError(
            f"Device {device_idx} not found (have {torch.cuda.device_count()} GPU(s))."
        )
    torch.cuda.set_device(device_idx)
    name = torch.cuda.get_device_name(device_idx)
    cap = torch.cuda.get_device_capability(device_idx)
    used = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
    cuda_v = torch.version.cuda or "unknown"
    print(
        f"{_LIME}GPU{_RST} cuda:{device_idx}  {_BOLD}{name}{_RST}\n"
        f"{_DIM}  VRAM {used:.2f}/{total:.1f} GB"
        f"   CUDA {cuda_v}   cap sm_{cap[0]}{cap[1]}{_RST}"
    )
    # Live kernel test
    try:
        t = torch.zeros(1024, device=f"cuda:{device_idx}")
        t = (t + 1).cpu()
        del t
        torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError(
            f"CUDA driver present but kernel launch failed: {e}"
        ) from e
    if cap < (12, 0):
        print(
            f"{_YELLOW}note:{_RST} GPU is sm_{cap[0]}{cap[1]} "
            f"(<sm_120). RTX 5090 expected; will run but unverified."
        )


# ---------------------------------------------------------------- progress

def _ascii_bar(frac: float, width: int = 28) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class _Spinner:
    """Indeterminate spinner with elapsed timer for blocking image inference."""
    FRAMES = "|/-\\"

    def __init__(self, label: str) -> None:
        self.label = label
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        i = 0
        start = time.time()
        while not self._stop.wait(0.1):
            ch = self.FRAMES[i % len(self.FRAMES)]
            elapsed = time.time() - start
            sys.stdout.write(f"\r  {_LIME}{ch}{_RST} {self.label}  {elapsed:5.1f}s")
            sys.stdout.flush()
            i += 1

    def __enter__(self) -> "_Spinner":
        self._t.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._t.join(timeout=1)
        sys.stdout.write("\r" + " " * 78 + "\r")
        sys.stdout.flush()


_PHASE_RE = re.compile(r"Phase\s+\d+\s*[:\-]\s*([^━─\n]+?)(?:\s*[━─]|$)", re.I)
_TS_RE = re.compile(r"^\s*\[\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\]\s*")
_BARS_RE = re.compile(r"[━─]+")


def _clean_phase_label(msg: str) -> str:
    """Extract a short, terminal-safe phase label from a backend msg line."""
    s = _TS_RE.sub("", msg)
    s = _BARS_RE.sub("", s).strip()
    m = _PHASE_RE.search(s)
    if m:
        s = m.group(1)
    # Strip leading emoji/symbol runs
    s = re.sub(r"^[^A-Za-z0-9]+", "", s).strip()
    if len(s) > 60:
        s = s[:57] + "..."
    return s


class _VideoProgress:
    """ETA + bar callback for video upscale_video().

    Multi-phase backends (e.g. SeedVR2: VAE-encode -> sample -> VAE-decode ->
    mux) report progress with different totals per phase. We dedupe identical
    (done, total) pairs so the bar doesn't spam, reset the fps timer at each
    phase boundary, and emit exactly one newline when each phase completes.
    Phase labels (when the backend supplies an msg) are printed once above the
    bar at each phase boundary so the user knows which step is running.
    """

    def __init__(self, verbose: bool = False) -> None:
        now = time.time()
        self.start = now
        self.phase_start = now
        self.last_print = 0.0
        self.last_msg: str | None = None
        self.cur_total: int | None = None
        self.last_state: tuple[int, int] | None = None
        self.phase_done_emitted = False
        self.verbose = verbose
        self.phase_label: str | None = None

    def __call__(self, done: int, total: int, msg: str | None = None) -> None:
        now = time.time()
        if msg:
            cleaned = _clean_phase_label(msg)
            if cleaned:
                self.phase_label = cleaned
            if self.verbose and msg != self.last_msg:
                sys.stdout.write("\r" + " " * 78 + "\r")
                print(f"  {_DIM}>{_RST} {msg}")
                self.last_msg = msg
        if not total:
            return
        # Phase change (new total) → reset timing + state, announce label.
        if total != self.cur_total:
            if self.cur_total is not None and not self.phase_done_emitted:
                sys.stdout.write("\n")
            self.cur_total = total
            self.phase_start = now
            self.last_print = 0.0
            self.last_state = None
            self.phase_done_emitted = False
            if self.phase_label and not self.verbose:
                sys.stdout.write("\r" + " " * 78 + "\r")
                print(f"  {_DIM}>{_RST} {self.phase_label}")
        state = (done, total)
        if state == self.last_state:
            return
        # Throttle mid-phase; always print boundary states (start + completion)
        if 0 < done < total and now - self.last_print < 0.1:
            return
        self.last_state = state
        self.last_print = now
        frac = done / total
        elapsed = now - self.phase_start
        fps = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / fps if fps > 0 else 0.0
        bar = _ascii_bar(frac, 28)
        sys.stdout.write(
            f"\r  {_LIME}{bar}{_RST} {done}/{total}  "
            f"{fps:5.1f} fps  ETA {_fmt_eta(eta)}"
        )
        sys.stdout.flush()
        if done >= total and not self.phase_done_emitted:
            sys.stdout.write("\n")
            self.phase_done_emitted = True


def _fmt_eta(s: float) -> str:
    if s <= 0 or s != s:  # NaN guard
        return "--:--"
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------- listing

def cmd_list() -> int:
    print(f"\n{_BOLD}Image models{_RST}  ({_DIM}python upskale.py IMG -m KEY{_RST})\n")
    for m in list_models():
        marker = _LIME + "+" + _RST if (Path(MODELS_DIR) / IMG_REGISTRY[m["key"]].weight_file).is_file() else _DIM + "." + _RST
        print(f"  {marker}  {m['key']:<24} {m['scale']}x   {_DIM}{m['best_for']}{_RST}")
    print(f"\n{_BOLD}Video models{_RST}  ({_DIM}python upskale.py VID -m KEY{_RST})\n")
    for m in list_video_models():
        miss = verify_weights_present(m["key"], MODELS_DIR)
        marker = _DIM + "." + _RST if miss else _LIME + "+" + _RST
        print(
            f"  {marker}  {m['key']:<24} {m['scale']}x   "
            f"{_DIM}{m['best_for']} (~{m['vram_gb']:.0f}GB){_RST}"
        )
    print(
        f"\n{_DIM}+ weight on disk    . missing\n"
        f"add custom: python upskale.py --add-model URL [--key KEY]{_RST}"
    )
    return 0


# ---------------------------------------------------------------- run image

def run_image(input_path: Path, model_key: str, out_path: Path) -> int:
    info = IMG_REGISTRY[model_key]
    print(f"{_DIM}model:{_RST} {info.name}  {_DIM}({info.weight_file}){_RST}")
    weight_path = ensure_weight(model_key, MODELS_DIR, log=lambda m: print(f"  {_DIM}[weights]{_RST} {m}"))
    img = Image.open(input_path)
    in_size = img.size
    print(f"{_DIM}input:{_RST} {in_size[0]}x{in_size[1]}  {input_path}")
    tensor = _pil_to_tensor(img)
    start = time.time()
    with _Spinner("upscaling"):
        out = image_upscale(tensor, model_key, weight_path)
    elapsed_ms = int((time.time() - start) * 1000)
    out_img = _tensor_to_pil(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    print(
        f"{_LIME}done{_RST}  {in_size[0]}x{in_size[1]} -> "
        f"{out_img.size[0]}x{out_img.size[1]}  in {elapsed_ms} ms\n"
        f"{_DIM}saved:{_RST} {out_path}"
    )
    return 0


# ---------------------------------------------------------------- run video

def run_video(
    input_path: Path,
    model_key: str,
    out_path: Path,
    batch_size: int | None,
    verbose: bool = False,
) -> int:
    info = VID_REGISTRY[model_key]
    missing = verify_weights_present(model_key, MODELS_DIR)
    if missing:
        _eprint(
            f"{_RED}error:{_RST} video model weights missing: {missing}\n"
            f"  Place files under {MODELS_DIR}/{info.subdir}/ "
            f"or vendor/{info.vendor_subdir}/"
        )
        return 2
    meta = video_probe(str(input_path))
    print(
        f"{_DIM}model:{_RST} {info.name}\n"
        f"{_DIM}input:{_RST} {meta.width}x{meta.height}  "
        f"{meta.fps:.2f} fps  {meta.duration_s:.1f}s  {meta.total_frames} frames\n"
        f"{_DIM}output:{_RST} {meta.width * info.scale}x{meta.height * info.scale}  "
        f"-> {out_path}"
    )
    if meta.height > info.max_input_height:
        _eprint(
            f"{_YELLOW}warn:{_RST} input height {meta.height}px exceeds model cap "
            f"{info.max_input_height}px; may OOM."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cancel_event = threading.Event()
    progress = _VideoProgress(verbose=verbose)
    start = time.time()
    try:
        upscale_video(
            str(input_path), str(out_path), model_key, MODELS_DIR,
            progress=progress,
            cancel_event=cancel_event,
            set_proc=None,
            batch_size=batch_size,
        )
    except KeyboardInterrupt:
        cancel_event.set()
        _eprint(f"\n{_YELLOW}cancelled{_RST}")
        return 130
    elapsed_ms = int((time.time() - start) * 1000)
    print(f"\n{_LIME}done{_RST}  in {elapsed_ms / 1000:.1f}s  saved: {out_path}")
    return 0


# ---------------------------------------------------------------- model resolve

def _detect_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    raise ValueError(
        f"Unsupported extension '{ext}'. Supported: "
        f"images={sorted(IMAGE_EXTS)} videos={sorted(VIDEO_EXTS)}"
    )


def _iter_paths(path: Path) -> tuple[list[Path], list[str]]:
    """Expand a file or directory into (valid_files, skipped_messages).

    For folders, scans top-level files only (no recursion). Files with
    unsupported extensions land in `skipped` with a human-readable reason.
    """
    if path.is_file():
        try:
            _detect_kind(path)
            return [path], []
        except ValueError as e:
            return [], [f"{path.name}: {e}"]
    if not path.is_dir():
        return [], [f"{path}: not a file or directory"]
    valid: list[Path] = []
    skipped: list[str] = []
    for child in sorted(path.iterdir()):
        if not child.is_file():
            continue
        try:
            _detect_kind(child)
            valid.append(child)
        except ValueError:
            skipped.append(
                f"{child.name}: unsupported extension '{child.suffix or '<none>'}'"
            )
    return valid, skipped


def _resolve_model(model_key: str, kind: str) -> str:
    """Return validated model_key, prompting for download if unknown (image only)."""
    if kind == "image":
        if model_key in IMG_REGISTRY:
            return model_key
        if model_key in VID_REGISTRY:
            raise SystemExit(
                f"{_RED}error:{_RST} '{model_key}' is a video model, "
                f"input is an image. Use a different model or input.\n"
                f"  See: python upskale.py --list"
            )
        # Interactive add
        _eprint(f"{_YELLOW}model '{model_key}' not in registry.{_RST}")
        if not sys.stdin.isatty():
            raise SystemExit(
                "  Run with --add-model URL to register first, "
                "or python upskale.py --list to see options."
            )
        url = input("  Paste download URL (.pth/.safetensors), or empty to abort: ").strip()
        if not url:
            raise SystemExit("aborted")
        info = add_model_from_url(url, key=model_key)
        return info.key
    # video kind
    if model_key in VID_REGISTRY:
        return model_key
    if model_key in IMG_REGISTRY:
        raise SystemExit(
            f"{_RED}error:{_RST} '{model_key}' is an image model, "
            f"input is a video. Use a video model.\n"
            f"  See: python upskale.py --list"
        )
    raise SystemExit(
        f"{_RED}error:{_RST} unknown video model '{model_key}'. "
        f"Adding video models via URL not supported (multi-file vendor blobs).\n"
        f"  See: python upskale.py --list"
    )


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="upskale",
        description="UPSKALE - local AI upscaler (image or video, RTX 5090 GPU).",
    )
    p.add_argument("input", nargs="?", help="Input image or video path")
    p.add_argument(
        "-m", "--model", default=None,
        help=f"Model key (default: {DEFAULT_MODEL} for images, required for video)",
    )
    p.add_argument("-o", "--output", default=None, help="Output path (default: <stem>_upscaled.<ext>)")
    p.add_argument("-b", "--batch", type=int, default=0, help="Video batch size (0 = backend default)")
    p.add_argument("--device", type=int, default=None, help="CUDA device index (default: 0)")
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logs (show backend INFO + per-batch progress messages)",
    )
    p.add_argument("--list", action="store_true", help="List all known models and exit")
    p.add_argument("--add-model", metavar="URL", help="Download a weight from URL and register it")
    p.add_argument("--key", default=None, help="(with --add-model) registry key")
    p.add_argument("--scale", type=int, default=4, help="(with --add-model) upscale factor")
    p.add_argument("--name", default=None, help="(with --add-model) display name")
    return p


def _ask(label: str, placeholder: str, default: str | None = None) -> str:
    """Prompt with a dim placeholder hint. Returns stripped input or default."""
    hint = f"  {_DIM}({placeholder}){_RST}"
    prompt = f"{_LIME}>{_RST} {_BOLD}{label}{_RST}{hint}\n  "
    try:
        raw = input(prompt)
    except EOFError:
        return default or ""
    raw = raw.strip().strip('"').strip("'")
    if not raw and default is not None:
        return default
    return raw


def interactive_prompt(args: argparse.Namespace) -> argparse.Namespace:
    """Fill missing args via stdin prompts with placeholder hints."""
    if not sys.stdin.isatty():
        return args
    print(f"{_DIM}interactive mode . press Ctrl+C to abort{_RST}\n")
    if not args.input:
        path = _ask(
            "input",
            "drag image/video here, or paste path e.g. C:/pics/cat.jpg",
        )
        if not path:
            raise SystemExit("aborted: no input provided")
        args.input = path
    p = Path(args.input).expanduser().resolve()
    suffix = p.suffix.lower()
    kind_hint = "image" if suffix in IMAGE_EXTS else (
        "video" if suffix in VIDEO_EXTS else "unknown"
    )
    if not args.model:
        if kind_hint == "video":
            placeholder = "e.g. flashvsr-v11 or seedvr2-7b  (--list to see all)"
            default = None
        else:
            placeholder = (
                f"e.g. ultrasharp, drct-l, realesrgan-anime  "
                f"[default: {DEFAULT_MODEL}]"
            )
            default = DEFAULT_MODEL
        args.model = _ask("model", placeholder, default=default) or default
    if not args.output:
        suggested = str(_default_output(p))
        args.output = _ask(
            "output",
            f"path or empty for default: {suggested}",
            default=suggested,
        )
    print()
    return args


def _default_output(input_path: Path) -> Path:
    stem = input_path.stem
    suffix = input_path.suffix.lower()
    if suffix in IMAGE_EXTS:
        # Save image as PNG to preserve quality
        return input_path.with_name(f"{stem}_upscaled.png")
    return input_path.with_name(f"{stem}_upscaled{suffix or '.mp4'}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_level = "INFO" if args.verbose else os.getenv("UPSKALE_LOG_LEVEL", "WARNING")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        # Silence chatty backend loggers unless --verbose; the progress bar is
        # the only signal users need.
        for name in ("loaders", "loaders.seedvr2", "loaders.flashvsr"):
            logging.getLogger(name).setLevel(logging.WARNING)
    print(BANNER)
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    _load_user_models()

    if args.list:
        return cmd_list()

    if args.add_model and not args.input:
        try:
            add_model_from_url(
                args.add_model, key=args.key, scale=args.scale, name=args.name,
            )
        except (ValueError, OSError) as e:
            _eprint(f"{_RED}error:{_RST} {e}")
            return 2
        return 0

    interactive_session = args.input is None and sys.stdin.isatty()

    if not args.input:
        if sys.stdin.isatty():
            args = interactive_prompt(args)
        else:
            _eprint(
                f"{_RED}error:{_RST} input path required "
                f"(or use --list / --add-model)"
            )
            return 2

    try:
        gpu_preflight(args.device)
    except RuntimeError as e:
        _eprint(f"{_RED}error:{_RST} {e}")
        return 1

    last_rc = 0
    while True:
        rc = _process_input(args)
        if rc != 0:
            last_rc = rc
        if not interactive_session:
            return last_rc
        # Loop: prompt for the next path. Empty input quits cleanly.
        print()
        nxt = _ask(
            "next",
            "another path/folder, or empty to quit",
            default="",
        )
        if not nxt:
            print(f"{_DIM}bye{_RST}")
            return last_rc
        args.input = nxt
        # Allow user to switch model on next round; clear -o so default kicks in
        args.output = None


def _process_input(args: argparse.Namespace) -> int:
    """Resolve args.input as file-or-folder, run jobs, return aggregate rc."""
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        _eprint(f"{_RED}error:{_RST} input not found: {input_path}")
        return 2

    paths, skipped = _iter_paths(input_path)
    for s in skipped:
        _eprint(f"{_YELLOW}skip:{_RST} {s}")
    if not paths:
        _eprint(f"{_RED}error:{_RST} no valid image/video files in {input_path}")
        return 2

    is_batch = input_path.is_dir() or len(paths) > 1
    if is_batch and args.output:
        _eprint(
            f"{_YELLOW}warn:{_RST} -o ignored for batch input; "
            f"using <stem>_upscaled.<ext> per file"
        )

    # --add-model: handle once before processing, image-side only.
    if args.add_model:
        try:
            info = add_model_from_url(
                args.add_model, key=args.key, scale=args.scale, name=args.name,
            )
        except (ValueError, OSError) as e:
            _eprint(f"{_RED}error:{_RST} {e}")
            return 2
        if args.model is None:
            args.model = info.key
        args.add_model = None  # don't redo on next loop iteration

    last_rc = 0
    total = len(paths)
    for i, p in enumerate(paths, 1):
        try:
            kind = _detect_kind(p)
        except ValueError as e:
            _eprint(f"{_YELLOW}skip:{_RST} {p.name}: {e}")
            continue

        # Choose / validate model per file
        model_key = args.model or (DEFAULT_MODEL if kind == "image" else None)
        if model_key is None:
            _eprint(
                f"{_YELLOW}skip:{_RST} {p.name}: video file but no -m/--model "
                f"given (see: python upskale.py --list)"
            )
            continue
        if (kind == "image" and model_key in VID_REGISTRY) or (
            kind == "video" and model_key in IMG_REGISTRY
        ):
            _eprint(
                f"{_YELLOW}skip:{_RST} {p.name}: model '{model_key}' is "
                f"{'video' if kind == 'image' else 'image'}-only"
            )
            continue
        try:
            model_key = _resolve_model(model_key, kind)
        except SystemExit as e:
            _eprint(str(e))
            last_rc = 2
            continue

        out_path = (
            Path(args.output).expanduser().resolve()
            if (args.output and not is_batch) else _default_output(p)
        )

        if total > 1:
            print(f"\n{_LIME}[{i}/{total}]{_RST} {p.name}")
        try:
            if kind == "image":
                rc = run_image(p, model_key, out_path)
            else:
                rc = run_video(
                    p, model_key, out_path, args.batch or None,
                    verbose=args.verbose,
                )
        except torch.cuda.OutOfMemoryError as e:
            _eprint(f"\n{_RED}error:{_RST} GPU OOM on {p.name}: {e}")
            rc = 1
        except Exception as e:
            logger.exception("Job failed: %s", p)
            _eprint(f"\n{_RED}error:{_RST} {p.name}: {e}")
            rc = 1
        if rc != 0:
            last_rc = rc

    return last_rc


def _console_main() -> None:
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print(f"\n{_DIM}bye{_RST}")
        sys.exit(130)


if __name__ == "__main__":
    _console_main()
