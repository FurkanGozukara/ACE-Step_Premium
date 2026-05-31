"""Output-folder allocation for manual audio processing runs."""

from __future__ import annotations

import re
from pathlib import Path


RUN_PREFIX = "audio_processing"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs"


def create_audio_processing_run_dir(output_folder: str | None = None) -> Path:
    """Create a numbered output folder for an audio-processing run.

    Args:
        output_folder: Optional explicit output root. Blank values use ACE-Step outputs.

    Returns:
        Newly created run directory.
    """

    root = Path(output_folder).expanduser().resolve() if output_folder else DEFAULT_RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    pattern = re.compile(rf"^{RUN_PREFIX}_(\d+)$")
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    target = root / f"{RUN_PREFIX}_{max_index + 1:04d}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def safe_media_stem(path: str | Path) -> str:
    """Return a compact filesystem-safe stem for media output names."""

    raw = Path(path).stem.strip() or "media"
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if char in invalid or ord(char) < 32 else char for char in raw)
    return cleaned.strip(" ._") or "media"
