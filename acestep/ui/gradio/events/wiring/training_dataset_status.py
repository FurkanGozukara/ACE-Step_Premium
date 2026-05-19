"""Status formatting helpers for training dataset wiring."""

from typing import Any


_CHECKMARK = "\u2705"
_PREVIEW_REFRESH_FAILURE_MARKERS = (
    "error",
    "failed to",
    "not initialized",
    "unavailable",
    "please scan",
    "no samples",
)


def append_preview_refresh_status(status: Any) -> str:
    """Append preview-refresh status only after successful auto-label work."""

    status_text = str(status or "")
    normalized = status_text.lower()
    if any(marker in normalized for marker in _PREVIEW_REFRESH_FAILURE_MARKERS):
        return status_text
    success_status = status_text or (_CHECKMARK + " Auto-label complete.")
    return f"{success_status}\n{_CHECKMARK} Preview refreshed."
