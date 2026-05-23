"""Console-output helpers for training subprocess streams."""

from __future__ import annotations

import sys
from typing import Any


def write_console_text(text: str, *, end: str = "", stream: Any = None) -> None:
    """Write text to a console stream with replacement fallback for Windows."""

    target = stream or sys.stdout
    payload = f"{text}{end}"
    try:
        target.write(payload)
        target.flush()
        return
    except UnicodeEncodeError:
        encoding = getattr(target, "encoding", None) or "utf-8"
        safe_payload = payload.encode(encoding, errors="replace").decode(
            encoding,
            errors="replace",
        )
        target.write(safe_payload)
        target.flush()
