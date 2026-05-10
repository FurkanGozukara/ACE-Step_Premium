"""Input-folder scanning helpers for batch folder generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BatchFolderItem:
    """A lyrics text file with an optional companion style text file."""

    stem: str
    lyrics_path: Path
    style_path: Path | None
    lyrics: str
    style: str


def resolve_existing_input_folder(input_folder: str | Path) -> Path:
    """Return a validated input folder path.

    Raises:
        ValueError: If the supplied path is empty or not an existing directory.
    """

    raw_value = str(input_folder or "").strip()
    if not raw_value:
        raise ValueError("Enter an input folder before starting batch processing.")
    folder = Path(raw_value).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Input folder does not exist: {folder}")
    return folder


def resolve_output_folder(output_folder: str | Path) -> Path:
    """Return a writable output folder path, creating it when needed."""

    raw_value = str(output_folder or "").strip()
    if not raw_value:
        raise ValueError("Enter an output folder before starting batch processing.")
    folder = Path(raw_value).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def discover_batch_folder_items(input_folder: str | Path) -> list[BatchFolderItem]:
    """Find lyrics files and optional ``*_style.txt`` companions.

    The scan is non-recursive. Files named ``*_style.txt`` are treated only as
    style companions and are not generated directly.

    Raises:
        ValueError: If no usable lyrics files are found.
    """

    folder = resolve_existing_input_folder(input_folder)
    items: list[BatchFolderItem] = []
    for lyrics_path in sorted(folder.glob("*.txt"), key=lambda path: path.name.lower()):
        if lyrics_path.stem.lower().endswith("_style"):
            continue

        lyrics = lyrics_path.read_text(encoding="utf-8").strip()
        if not lyrics:
            continue

        style_path = lyrics_path.with_name(f"{lyrics_path.stem}_style.txt")
        style = ""
        if style_path.exists() and style_path.is_file():
            style = style_path.read_text(encoding="utf-8").strip()

        items.append(
            BatchFolderItem(
                stem=lyrics_path.stem,
                lyrics_path=lyrics_path,
                style_path=style_path if style else None,
                lyrics=lyrics,
                style=style,
            )
        )

    if not items:
        raise ValueError(
            "No lyrics .txt files found. Add files like song.txt with optional song_style.txt."
        )
    return items
