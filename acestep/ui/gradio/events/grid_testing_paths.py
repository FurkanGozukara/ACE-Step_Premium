"""Path allocation helpers for Grid Testing outputs."""

from __future__ import annotations

import re
from pathlib import Path

from acestep.training.path_inputs import normalize_user_path
from acestep.ui.gradio.events.results.output_manager import DEFAULT_RESULTS_DIR


_GRID_DIR_PATTERN = re.compile(r"^grid-(\d{4})$")


def resolve_grid_output_folder(output_folder: str | Path | None) -> Path:
    """Return the folder where final grid artifacts should be saved.

    Args:
        output_folder: Optional custom folder. Empty values allocate the next
            ``outputs/grid-0001`` style folder.

    Returns:
        Existing or newly-created output folder path.
    """

    raw_value = normalize_user_path(output_folder)
    if raw_value:
        target = Path(raw_value).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target
    return _create_next_grid_folder(DEFAULT_RESULTS_DIR)


def next_sample_index(target_folder: Path, prefix: str) -> int:
    """Return the next available numeric suffix for a prefix in the target folder."""

    pattern = re.compile(rf"^{re.escape(prefix)}-(\d{{4}})(?:$|[_.])")
    max_index = 0
    for child in target_folder.iterdir() if target_folder.exists() else []:
        match = pattern.match(child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _create_next_grid_folder(root: Path) -> Path:
    """Allocate the next ``grid-0001`` folder under the default outputs root."""

    root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _GRID_DIR_PATTERN.match(child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    target = root / f"grid-{max_index + 1:04d}"
    target.mkdir(parents=True, exist_ok=False)
    return target
