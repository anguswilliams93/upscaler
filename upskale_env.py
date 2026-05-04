"""Persist UPSKALE_* env vars for the current user.

Windows: writes via `setx` (HKCU\\Environment).
Other:   prints export lines for the user to source.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _defaults() -> dict[str, str]:
    root = Path.cwd().resolve()
    return {
        "UPSKALE_MODELS_DIR":    str(root / "models"),
        "UPSKALE_OUTPUTS_DIR":   str(root / "outputs"),
        "UPSKALE_UPLOADS_DIR":   str(root / "uploads"),
        "UPSKALE_DEFAULT_MODEL": "ultrasharp",
    }


def init() -> int:
    pairs = _defaults()
    if os.name == "nt":
        for k, v in pairs.items():
            subprocess.run(["setx", k, v], check=True)
            print(f"  set {k}={v}")
        print("\nOpen a NEW shell to pick up the variables.")
        return 0
    print("# Add to your shell rc (~/.bashrc, ~/.zshrc):")
    for k, v in pairs.items():
        print(f'export {k}="{v}"')
    return 0


if __name__ == "__main__":
    sys.exit(init())
