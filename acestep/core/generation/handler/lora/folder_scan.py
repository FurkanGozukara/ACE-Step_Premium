"""Discover LoRA safetensors files from project-level LoRA folders."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from acestep.core.generation.handler.lora.path_resolution import (
    PROJECT_ROOT,
    resolve_lora_input_path,
)


PREFERRED_LORA_FOLDER = "Loras"
LORA_FOLDER_NAMES = ("loras", "lora")
PEFT_CONFIG_FILENAME = "adapter_config.json"
LOKR_WEIGHTS_FILENAME = "lokr_weights.safetensors"
PEFT_WEIGHT_FILENAMES = {"adapter_model.safetensors", "adapter_model.bin"}
LORA_NONE_CHOICE: tuple[str, str] = ("None", "")


@dataclass(frozen=True)
class LoraFolderItem:
    """One safetensors artifact discovered in a project LoRA folder."""

    label: str
    path: str
    root: str


def ensure_lora_folder(project_root: str | os.PathLike[str] | None = None) -> Path:
    """Create and return the preferred project ``Loras`` folder."""

    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    preferred = root / PREFERRED_LORA_FOLDER
    preferred.mkdir(parents=True, exist_ok=True)
    return preferred


def _lora_roots(project_root: Path) -> list[Path]:
    """Return case-insensitive ``Loras``/``Lora`` roots, preferring ``Loras``."""

    preferred = ensure_lora_folder(project_root).resolve()
    roots = [preferred]
    try:
        children = list(project_root.iterdir())
    except OSError:
        children = []

    for child in children:
        if not child.is_dir() or child.name.lower() not in LORA_FOLDER_NAMES:
            continue
        try:
            resolved = child.resolve()
        except OSError:
            resolved = child.absolute()
        if resolved not in roots:
            roots.append(resolved)

    def priority(path: Path) -> tuple[int, str]:
        name = path.name
        if name == PREFERRED_LORA_FOLDER:
            return (0, name.lower())
        if name.lower() == PREFERRED_LORA_FOLDER.lower():
            return (1, name.lower())
        if name == "Lora":
            return (2, name.lower())
        return (3, name.lower())

    return sorted(roots, key=priority)


def _contains_peft_weights(path: Path) -> bool:
    """Return whether a directory contains PEFT adapter weights."""

    try:
        names = {child.name.lower() for child in path.iterdir() if child.is_file()}
    except OSError:
        return False
    return bool(names.intersection(PEFT_WEIGHT_FILENAMES))


def _is_adapter_dir(path: Path) -> bool:
    """Return whether a directory directly contains loadable adapter artifacts."""

    if not path.is_dir():
        return False
    try:
        names = {child.name.lower() for child in path.iterdir() if child.is_file()}
    except OSError:
        return False
    if PEFT_CONFIG_FILENAME in names and _contains_peft_weights(path):
        return True
    return LOKR_WEIGHTS_FILENAME in names


def _is_lora_safetensors_file(path: Path) -> bool:
    """Return whether a file should be listed as a LoRA safetensors artifact."""

    if not path.is_file():
        return False
    return path.suffix.lower() == ".safetensors"


def _relative_label(path: Path, project_root: Path) -> str:
    """Return a stable, readable dropdown label."""

    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _scan_dir(
    root: Path,
    project_root: Path,
    *,
    max_depth: int,
    depth: int = 0,
) -> list[LoraFolderItem]:
    """Recursively scan a LoRA root for safetensors files."""

    items: list[LoraFolderItem] = []
    try:
        children = sorted(root.iterdir(), key=lambda child: child.name.lower())
    except OSError:
        return items

    safetensors_children = [
        child for child in children if child.is_file() and _is_lora_safetensors_file(child)
    ]
    for child in safetensors_children:
        items.append(
            LoraFolderItem(
                label=_relative_label(child, project_root),
                path=str(child),
                root=str(root),
            )
        )

    for child in children:
        if not child.is_dir() or depth >= max_depth:
            continue
        if safetensors_children and child.name.lower() == "adapter" and _is_adapter_dir(child):
            continue
        else:
            items.extend(
                _scan_dir(
                    child,
                    project_root,
                    max_depth=max_depth,
                    depth=depth + 1,
                )
            )
    return items


def discover_lora_folder_items(
    project_root: str | os.PathLike[str] | None = None,
    *,
    max_depth: int = 4,
) -> list[LoraFolderItem]:
    """Return discovered safetensors files from ``Loras`` and ``Lora`` folders."""

    root = (Path(project_root) if project_root is not None else PROJECT_ROOT).resolve()
    items: list[LoraFolderItem] = []
    seen: set[str] = set()
    for lora_root in _lora_roots(root):
        for item in _scan_dir(lora_root, root, max_depth=max_depth):
            try:
                key = str(Path(item.path).resolve())
            except OSError:
                key = item.path
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
    return items


def lora_dropdown_choices(
    project_root: str | os.PathLike[str] | None = None,
) -> list[tuple[str, str]]:
    """Return Gradio dropdown choices for discovered LoRA safetensors files."""

    return [
        LORA_NONE_CHOICE,
        *((item.label, item.path) for item in discover_lora_folder_items(project_root)),
    ]


def resolve_loadable_lora_adapter_path(raw_path: str | os.PathLike[str] | None) -> str:
    """Return a resolved loadable LoRA/LoKr adapter path, or ``""``.

    This is intentionally a filesystem-only check used by UI and generation
    orchestration before the heavyweight model loader is invoked.
    """

    if raw_path is None or not str(raw_path).strip():
        return ""

    resolution = resolve_lora_input_path(raw_path)
    if not resolution.resolved_path:
        return ""

    path = Path(resolution.resolved_path)
    if path.is_file():
        return str(path) if _is_lora_safetensors_file(path) else ""

    if not path.is_dir():
        return ""

    try:
        names = {child.name.lower() for child in path.iterdir() if child.is_file()}
    except OSError:
        return ""

    if LOKR_WEIGHTS_FILENAME in names:
        return str(path)
    if PEFT_CONFIG_FILENAME in names and bool(names.intersection(PEFT_WEIGHT_FILENAMES)):
        return str(path)
    return ""
