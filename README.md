# upskale

Local AI image and video upscaler. FastAPI server + standalone CLI, designed for an
NVIDIA RTX 5090 (Blackwell, sm_120, 32 GB GDDR7) but runs on any modern CUDA GPU
with enough VRAM for the chosen model. No cloud, no per-image cost, no data leaves
the box.

```
^\      /^   UPSKALE  v0.1.0
  \    /     local super-resolution . Local GPU
  /    \     by ZEROBI  output > input
v/      \v
```

---

## Features

- **Image SR** via [`spandrel`](https://github.com/chaiNNer-org/spandrel): UltraSharp,
  DRCT-L, RealESRGAN (general / anime / x4plus), Nomos ATD-JPG, Remacri, plus any
  user-added `.pth` / `.safetensors` from [OpenModelDB](https://openmodeldb.info).
- **Video SR**: SeedVR2-7B (high quality, multi-phase) and FlashVSR-v11
  (fast streaming).
- **Two front ends**:
  - `upskale` — standalone CLI with interactive prompts, batch folder mode, and
    drag-and-drop in the terminal.
  - `upskale-server` — FastAPI app at `http://localhost:8000` with a vanilla-JS
    web UI (no bundler, no npm).
- **GPU-first**: fp16 inference, model VRAM cache, no CPU fallback,
  `torch.cuda.empty_cache()` after every job.
- **Persistent config** via `UPSKALE_*` user-environment variables.

---

## Hardware

| Component | Recommended            |
|-----------|------------------------|
| GPU       | RTX 5090 (32 GB VRAM)  |
| Driver    | NVIDIA 570+ (CUDA 12.8)|
| OS        | Windows 11 or Linux    |
| Python    | 3.11+                  |

Smaller GPUs work for the image models (8 GB+); video models need 12–25 GB.

---

## Install

```bash
git clone https://github.com/<you>/upskale.git
cd upskale
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 1. PyTorch from the CUDA 12.8 index (required for sm_120).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. The package itself (registers `upskale`, `upskale-init-env`, `upskale-server`).
pip install -e .
```

Verify:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# > NVIDIA GeForce RTX 5090
```

---

## Configure

Two options. Pick one.

### A) Persistent user-env vars (recommended)

```powershell
upskale-init-env
# Open a NEW shell so setx values load.
```

This writes the following keys to the Windows user registry (HKCU\Environment) — or
prints `export` lines on Linux/macOS:

```
UPSKALE_MODELS_DIR    = <repo>/models
UPSKALE_OUTPUTS_DIR   = <repo>/outputs
UPSKALE_UPLOADS_DIR   = <repo>/uploads
UPSKALE_DEFAULT_MODEL = ultrasharp
```

### B) Per-repo `.env` file

Copy `.env.example` to `.env` and edit. The CLI/server load it via `python-dotenv`
at startup. `.env` is git-ignored.

All recognised keys:

```ini
HOST=0.0.0.0
PORT=8000
UPSKALE_MODELS_DIR=./models
UPSKALE_UPLOADS_DIR=./uploads
UPSKALE_OUTPUTS_DIR=./outputs
UPSKALE_DEFAULT_MODEL=ultrasharp
UPSKALE_MAX_UPLOAD_MB=50
UPSKALE_MAX_VIDEO_MB=500
UPSKALE_MAX_VIDEO_SECONDS=60
UPSKALE_MAX_VIDEO_HEIGHT=720
UPSKALE_OUTPUT_TTL_HOURS=1
UPSKALE_LOG_LEVEL=INFO
```

---

## Models

Weights are not committed to the repo (large + license-bound). Drop `.pth` /
`.safetensors` files into `models/` or use the CLI:

```bash
upskale --list                                # show registry status
upskale --add-model https://.../weights.pth   # download + register
```

| Key                  | Scale | Best for                    |
|----------------------|-------|-----------------------------|
| `ultrasharp`         | 4x    | General photos (JPEG-tuned) |
| `drct-l`             | 4x    | Clean photos, best quality  |
| `realesrgan`         | 4x    | General purpose             |
| `realesrgan-anime`   | 4x    | Anime / illustration        |
| `realesrgan-general` | 4x    | Mixed degradation           |
| `nomos-atd-jpg`      | 4x    | Heavy JPEG compression      |
| `remacri`            | 4x    | Softer sharpening           |
| `seedvr2-7b`         | 4x    | Video — high quality, slow  |
| `flashvsr-v11`       | 4x    | Video — fast streaming      |

Custom: `upskale --add-model URL --key my-key --scale 4 --name "My Model"`.

---

## Usage

### CLI

```bash
upskale                                        # interactive prompt
upskale path/to/image.png                      # default model
upskale path/to/image.png -m drct-l -o out.png
upskale path/to/clip.mp4 -m flashvsr-v11
upskale path/to/folder -m ultrasharp           # batch
upskale --list
upskale -h
```

Interactive mode loops on completion: paste the next path, or empty to quit.

### Web server

```bash
upskale-server                # boots on http://localhost:8000
```

API:

```
POST /api/upscale          # multipart: file, model, scale=2|4
POST /api/upscale-video    # multipart: file, model, batch_size
GET  /api/job/{id}         # poll for status + progress + logs
POST /api/job/{id}/cancel
GET  /api/models           # image models
GET  /api/video-models     # video models
GET  /api/gpu              # live VRAM + jobs-done counter
GET  /outputs/{filename}   # download upscaled file
```

---

## Project layout

```
.
|-- main.py                  FastAPI server + job queue
|-- upskale.py               argparse CLI
|-- upskale_env.py           setx helper (`upskale-init-env`)
|-- upscaler.py              image inference path
|-- video_upscaler.py        video inference path
|-- model_registry.py        image registry
|-- model_registry_video.py  video registry
|-- schemas.py               Pydantic request/response models
|-- loaders/                 backend-specific model loaders
|-- static/                  vanilla HTML/CSS/JS web UI
|-- models/                  weights (git-ignored)
|-- uploads/, outputs/       runtime dirs (git-ignored)
|-- vendor/                  vendored backends, e.g. SeedVR2 (git-ignored)
|-- tests/                   pytest suite
|-- pyproject.toml           PEP 621 metadata + console scripts
`-- requirements.txt         legacy pin file (kept for `pip install -r ...`)
```

---

## Develop

```bash
pip install -e ".[dev]"
pytest tests/ -v
mypy . --ignore-missing-imports
```

CLAUDE.md captures the agent rules (GPU-only, fp16 default, no bundler, etc.).

---

## License

MIT (see `LICENSE`). Model weights are licensed separately by their authors;
read the OpenModelDB / Hugging Face page for each before redistributing.
