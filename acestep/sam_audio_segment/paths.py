"""Path helpers for SAM-Audio model and output artifacts."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUN_PREFIX = "sam_audio"
BF16_MODEL_NAME = "SAM-Audio-Large-BF16.safetensors"
FP32_MODEL_NAME = "SAM-Audio-Large.pt"


def default_model_path() -> Path:
    """Return the preferred SAM-Audio checkpoint path."""

    bf16 = DEFAULT_MODELS_DIR / BF16_MODEL_NAME
    if bf16.is_file():
        return bf16
    root_fp32 = DEFAULT_MODELS_DIR / FP32_MODEL_NAME
    if root_fp32.is_file():
        return root_fp32
    return DEFAULT_MODELS_DIR / "SAM-Audio-Large" / "checkpoint.pt"


def create_run_dir(output_folder: str | None = None) -> Path:
    """Create a numbered SAM-Audio output directory."""

    root = Path(output_folder).expanduser().resolve() if output_folder else DEFAULT_OUTPUTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{RUN_PREFIX}_(\d+)$")
    max_index = 0
    for child in root.iterdir():
        if child.is_dir():
            match = pattern.match(child.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    target = root / f"{RUN_PREFIX}_{max_index + 1:04d}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def safe_media_stem(path: str | Path) -> str:
    """Return a filesystem-safe media stem."""

    raw = Path(path).stem.strip() or "media"
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if char in invalid or ord(char) < 32 else char for char in raw)
    return cleaned.strip(" ._") or "media"
