"""Path resolution helpers for user-supplied LoRA adapter paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
PEFT_CONFIG_FILENAME = "adapter_config.json"
LOKR_WEIGHTS_FILENAME = "lokr_weights.safetensors"
PEFT_WEIGHT_FILENAMES = {"adapter_model.safetensors", "adapter_model.bin"}


@dataclass(frozen=True)
class LoraPathResolution:
    """Resolved LoRA path details for status messages and loading."""

    original_path: str
    display_path: str
    resolved_path: str | None
    searched_paths: tuple[str, ...]


def _strip_wrapping_quotes(value: str) -> str:
    """Remove one or more matching quote pairs around a path."""

    result = value.strip()
    while len(result) >= 2 and result[0] == result[-1] and result[0] in {"'", '"'}:
        result = result[1:-1].strip()
    return result


def _clean_user_path(raw_path: str | os.PathLike[str]) -> str:
    """Return a normalized text path from user input."""

    value = os.fspath(raw_path)
    value = _strip_wrapping_quotes(value)
    return os.path.expanduser(os.path.expandvars(value))


def _separator_variants(path_text: str) -> list[str]:
    """Return native and cross-separator variants for common pasted paths."""

    variants = [path_text]
    if "\\" in path_text:
        variants.append(path_text.replace("\\", os.sep if os.name == "nt" else "/"))
    if os.name == "nt" and "/" in path_text:
        variants.append(path_text.replace("/", "\\"))

    deduped: list[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def _candidate_paths(path_text: str) -> list[Path]:
    """Return absolute and relative candidate paths to check."""

    candidates: list[Path] = []
    roots = [Path.cwd(), PROJECT_ROOT]
    for variant in _separator_variants(path_text):
        candidate = Path(variant)
        if candidate.is_absolute():
            candidates.append(candidate)
            continue
        for root in roots:
            candidates.append(root / candidate)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            key = str(candidate.absolute())
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _normalize_existing_path(path: Path) -> Path:
    """Return a resolved existing path when possible."""

    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return path.absolute()


def _looks_like_adapter_dir(path: Path) -> bool:
    """Return whether a directory directly contains adapter artifacts."""

    if not path.is_dir():
        return False
    if (path / PEFT_CONFIG_FILENAME).is_file():
        return True
    if (path / LOKR_WEIGHTS_FILENAME).is_file():
        return True
    try:
        return any(
            child.is_file() and child.suffix.lower() == ".safetensors"
            for child in path.iterdir()
        )
    except OSError:
        return False


def _select_adapter_path(path: Path) -> Path:
    """Return the most specific adapter path under an existing input path."""

    if path.is_file():
        if path.name in PEFT_WEIGHT_FILENAMES and (path.parent / PEFT_CONFIG_FILENAME).is_file():
            return path.parent
        return path

    if _looks_like_adapter_dir(path):
        return path

    adapter_child = path / "adapter"
    if _looks_like_adapter_dir(adapter_child):
        return adapter_child

    try:
        adapter_children = [
            child for child in path.iterdir()
            if child.is_dir() and _looks_like_adapter_dir(child)
        ]
    except OSError:
        adapter_children = []
    if len(adapter_children) == 1:
        return adapter_children[0]

    return path


def resolve_lora_input_path(raw_path: str | os.PathLike[str]) -> LoraPathResolution:
    """Resolve a user LoRA path from relative, absolute, quoted, or pasted forms."""

    original = os.fspath(raw_path)
    display_path = _clean_user_path(raw_path)
    if not display_path:
        return LoraPathResolution(original, display_path, None, ())

    searched: list[str] = []
    for candidate in _candidate_paths(display_path):
        normalized = _normalize_existing_path(candidate)
        searched.append(str(normalized))
        if normalized.exists():
            adapter_path = _select_adapter_path(normalized)
            return LoraPathResolution(
                original_path=original,
                display_path=display_path,
                resolved_path=str(adapter_path),
                searched_paths=tuple(searched),
            )

    return LoraPathResolution(
        original_path=original,
        display_path=display_path,
        resolved_path=None,
        searched_paths=tuple(searched),
    )
