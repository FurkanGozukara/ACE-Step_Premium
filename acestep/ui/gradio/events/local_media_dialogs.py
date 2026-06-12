"""Local media file picker helpers for Gradio event handlers."""

from __future__ import annotations

from acestep.ui.gradio.events import local_path_dialogs


_MEDIA_FILE_TYPES = (
    (
        "Media files",
        "*.wav *.flac *.mp3 *.ogg *.m4a *.aac *.opus *.mp4 *.mov *.mkv *.webm *.avi",
    ),
    ("All files", "*.*"),
)


def select_media_file_path(current_path: str = "") -> str:
    """Open a native media file picker and return the selected file path."""

    current = local_path_dialogs.normalize_dialog_path(current_path)
    if not local_path_dialogs.is_dialog_available():
        return current

    initial_dir, initial_file = local_path_dialogs._split_initial_file(current)
    root = local_path_dialogs._create_dialog_root()
    try:
        selected = local_path_dialogs.filedialog.askopenfilename(
            filetypes=_MEDIA_FILE_TYPES,
            initialdir=initial_dir,
            initialfile=initial_file,
        )
    finally:
        root.destroy()
    return local_path_dialogs.normalize_dialog_path(selected) if selected else current
