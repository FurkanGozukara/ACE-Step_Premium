"""Normalize user-entered filesystem paths before validation or I/O."""

from __future__ import annotations

import os
import re
from urllib.parse import unquote, urlparse


_QUOTE_CHARS = {"'", '"'}
_WINDOWS_DRIVE_PATH_RE = re.compile(r"^/[A-Za-z]:[\\/]")


def normalize_user_path(path: object) -> str:
    """Return a clean local path string from user-entered UI text.

    Args:
        path: Raw path value from textboxes, dialogs, or CLI arguments.

    Returns:
        A normalized string with wrapping quotes removed, environment variables
        expanded, and native separators normalized while preserving spaces.
    """

    value = _coerce_path_text(path)
    value = _strip_wrapping_quotes(value)
    if not value:
        return ""

    value = _path_from_file_uri(value)
    value = os.path.expandvars(os.path.expanduser(value))
    value = _normalize_separators(value)
    return os.path.normpath(value) if value else ""


def _coerce_path_text(path: object) -> str:
    """Convert path-like input to stripped text."""

    if path is None:
        return ""
    try:
        value = os.fspath(path)
    except TypeError:
        value = str(path)
    return str(value).replace("\r", "").replace("\n", "").strip()


def _strip_wrapping_quotes(value: str) -> str:
    """Remove pasted shell quotes around a path without touching inner spaces."""

    result = value.strip()
    while len(result) >= 2 and result[0] == result[-1] and result[0] in _QUOTE_CHARS:
        result = result[1:-1].strip()
    while result and result[0] in _QUOTE_CHARS:
        result = result[1:].strip()
    while result and result[-1] in _QUOTE_CHARS:
        result = result[:-1].strip()
    return result


def _path_from_file_uri(value: str) -> str:
    """Convert local file:// URIs copied from file managers to paths."""

    parsed = urlparse(value)
    if parsed.scheme.lower() != "file":
        return value

    path = unquote(parsed.path or "")
    if os.name == "nt":
        if parsed.netloc:
            unc_path = path.replace("/", "\\")
            return f"\\\\{parsed.netloc}{unc_path}"
        if _WINDOWS_DRIVE_PATH_RE.match(path):
            path = path[1:]
    return path


def _normalize_separators(value: str) -> str:
    """Normalize common pasted separator styles for the active platform."""

    if os.name == "nt":
        return value.replace("/", "\\")
    return value
