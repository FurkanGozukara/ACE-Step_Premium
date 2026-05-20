"""Local file and folder picker helpers for Gradio event handlers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from acestep.training.path_inputs import normalize_user_path
from acestep.training.path_safety import safe_path

try:
    from tkinter import Tk, filedialog
except ImportError:  # pragma: no cover - depends on Python build
    Tk = None
    filedialog = None


_ENV_EXCLUSION = ("COLAB_GPU", "RUNPOD_POD_ID")


def normalize_dialog_path(path: str) -> str:
    """Return a normalized filesystem path selected from a local dialog."""
    value = normalize_user_path(path)
    if not value:
        return ""
    try:
        return str(Path(value).resolve())
    except (OSError, RuntimeError):
        return os.path.abspath(os.path.normpath(value))


def is_dialog_available() -> bool:
    """Return whether this process can show a native local picker dialog."""
    if (
        Tk is None
        or filedialog is None
        or any(name in os.environ for name in _ENV_EXCLUSION)
        or sys.platform == "darwin"
    ):
        return False
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False
    return True


def select_folder_path(current_path: str = "") -> str:
    """Open a native folder picker and return the selected folder path."""
    selected = select_optional_folder_path(current_path)
    return selected if selected is not None else normalize_dialog_path(current_path)


def select_optional_folder_path(current_path: str = "") -> str | None:
    """Open a native folder picker and return ``None`` when it is canceled."""
    current = normalize_dialog_path(current_path)
    if not is_dialog_available():
        return current if current else None
    root = _create_dialog_root()
    try:
        selected = filedialog.askdirectory(initialdir=_initial_dir(current))
    finally:
        root.destroy()
    return normalize_dialog_path(selected) if selected else None


def select_json_file_path(current_path: str = "") -> str:
    """Open a native JSON file picker and return the selected file path."""
    selected = select_optional_json_file_path(current_path)
    return selected if selected is not None else normalize_dialog_path(current_path)


def select_optional_json_file_path(current_path: str = "") -> str | None:
    """Open a native JSON picker and return ``None`` when it is canceled."""
    return _select_file_path(
        current_path, ".json", "JSON files", save=False, cancel_returns_current=False
    )


def select_json_save_path(current_path: str = "") -> str:
    """Open a native JSON save picker and return the selected file path."""
    return _select_file_path(current_path, ".json", "JSON files", save=True) or ""


def select_safetensors_save_path(current_path: str = "") -> str:
    """Open a native safetensors save picker and return the selected file path."""
    return _select_file_path(current_path, ".safetensors", "SafeTensors files", save=True) or ""


def select_pt_file_path(current_path: str = "") -> str:
    """Open a native PyTorch state file picker and return the selected file path."""

    return _select_file_path(current_path, ".pt", "PyTorch state files", save=False) or ""


def open_folder_path(path: str) -> str:
    """Open a local folder in the platform file explorer.

    Args:
        path: User-selected folder path to open.

    Returns:
        User-facing status describing the opened folder or failure.
    """

    normalized = normalize_user_path(path)
    if not normalized:
        return "No folder path selected."
    try:
        target = Path(safe_path(normalized))
        target.mkdir(parents=True, exist_ok=True)
        _open_folder_in_platform_file_manager(target)
    except (OSError, RuntimeError, ValueError) as exc:
        return f"Failed to open folder: {exc}"
    return f"Opened folder: {target}"


def _select_file_path(
    current_path: str,
    default_extension: str,
    extension_name: str,
    *,
    save: bool,
    cancel_returns_current: bool = True,
) -> str | None:
    """Open a native file picker and return a normalized selected path."""
    current = normalize_dialog_path(current_path)
    if not is_dialog_available():
        return current if cancel_returns_current or current else None
    initial_dir, initial_file = _split_initial_file(current)
    root = _create_dialog_root()
    try:
        filetypes = ((extension_name, f"*{default_extension}"), ("All files", "*.*"))
        if save:
            selected = filedialog.asksaveasfilename(
                filetypes=filetypes,
                defaultextension=default_extension,
                initialdir=initial_dir,
                initialfile=initial_file,
                confirmoverwrite=False,
            )
        else:
            selected = filedialog.askopenfilename(
                filetypes=filetypes,
                defaultextension=default_extension,
                initialdir=initial_dir,
                initialfile=initial_file,
            )
    finally:
        root.destroy()
    if not selected:
        return current if cancel_returns_current else None
    selected = normalize_dialog_path(selected)
    if save and default_extension and not selected.lower().endswith(default_extension):
        selected = f"{selected}{default_extension}"
    return selected


def _create_dialog_root():
    """Create a hidden topmost Tk root for native file dialogs."""
    root = Tk()
    try:
        root.wm_attributes("-topmost", 1)
    except Exception:
        pass
    root.withdraw()
    return root


def _initial_dir(path: str) -> str:
    """Return the directory where a dialog should open."""
    if path and os.path.isdir(path):
        return path
    parent = os.path.dirname(path) if path else ""
    return parent if parent else os.getcwd()


def _split_initial_file(path: str) -> tuple[str, str]:
    """Return initial dialog directory and filename for a file path."""
    if path and os.path.isdir(path):
        return path, ""
    return _initial_dir(path), os.path.basename(path) if path else ""


def _open_folder_in_platform_file_manager(target: Path) -> None:
    """Open ``target`` with the current platform's file manager."""

    if sys.platform == "win32":
        os.startfile(str(target))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
        return

    for command in (["xdg-open"], ["gio", "open"], ["kde-open"]):
        if shutil.which(command[0]):
            subprocess.Popen([*command, str(target)])
            return
    raise RuntimeError(
        "No Linux folder opener found: install xdg-open, gio, or kde-open."
    )
