"""Path sanitisation helpers for training modules.

Provides a single ``safe_path`` function that validates user-provided
filesystem paths against a known safe root directory.  The validation
uses ``os.path.realpath`` followed by a ``.startswith`` check — the
exact pattern that CodeQL recognises as a sanitiser for the
``py/path-injection`` query.

Symlinks are resolved on both the root and user paths so that paths
through symlinks (e.g. ``/root/data`` → ``/vepfs/.../data``) are
compared consistently.

All training modules that accept user-supplied paths should call
``safe_path`` (or ``safe_open``) before performing any filesystem I/O.
"""

import os
from typing import Iterable, Optional

from loguru import logger

from acestep.training.path_inputs import normalize_user_path


def _resolve(path: str) -> str:
    """Normalise and resolve symlinks in *path*.

    Uses ``os.path.realpath`` so that symlinked prefixes are resolved
    to their canonical form before comparison.
    """
    return os.path.normpath(os.path.realpath(normalize_user_path(path)))


# Root directories that user-provided paths must resolve under.
# Defaults to the working directory at import time. Override via
# ``set_safe_root`` / ``set_safe_roots`` if needed (e.g. in tests).
_SAFE_ROOTS: list[str] = [_resolve(os.getcwd())]


def _dedupe_roots(roots: Iterable[str]) -> list[str]:
    """Resolve and deduplicate safe roots while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root:
            continue
        resolved = _resolve(root)
        key = os.path.normcase(resolved)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)
    return deduped


def _safe_prefix(root: str) -> str:
    """Return the boundary-safe prefix for child paths under *root*."""
    normalised_root = os.path.normcase(root)
    if normalised_root.endswith(os.sep):
        return normalised_root
    return normalised_root + os.sep


def set_safe_root(root: str) -> None:
    """Override the safe root directory.

    Args:
        root: New safe root (will be normalised and symlink-resolved).
    """
    set_safe_roots([root])


def set_safe_roots(roots: Iterable[str]) -> None:
    """Override the safe root directories.

    Args:
        roots: New safe roots (will be normalised and symlink-resolved).

    Raises:
        ValueError: If no usable roots are provided.
    """
    resolved_roots = _dedupe_roots(roots)
    if not resolved_roots:
        raise ValueError("At least one safe root is required")
    global _SAFE_ROOTS  # noqa: PLW0603
    _SAFE_ROOTS = resolved_roots


def add_safe_roots(roots: Iterable[str]) -> list[str]:
    """Add extra safe root directories.

    Args:
        roots: Extra safe roots to allow.

    Returns:
        The full list of configured safe roots.
    """
    set_safe_roots([*_SAFE_ROOTS, *roots])
    return get_safe_roots()


def get_safe_root() -> str:
    """Return the current safe root directory."""
    return _SAFE_ROOTS[0]


def get_safe_roots() -> list[str]:
    """Return all current safe root directories."""
    return list(_SAFE_ROOTS)


def safe_path(user_path: str, *, base: Optional[str] = None) -> str:
    """Validate and normalise a user-provided path.

    The returned path is guaranteed to live under *base* (or one of the
    global safe roots when *base* is ``None``).  Symlinks in both
    the root and user path are resolved so that paths through symlinks
    compare correctly.

    Args:
        user_path: Untrusted path string from user input. Wrapping quotes,
                   environment variables, ``~``, and native separators are
                   normalized before validation.
        base: Optional explicit base directory.  When provided it is
              resolved (symlinks included) and used instead of
              the global safe roots.

    Returns:
        Normalised, symlink-resolved absolute path within the safe root.

    Raises:
        ValueError: If the resolved path escapes the safe root.
    """
    user_path = normalize_user_path(user_path)
    roots = [_resolve(base)] if base is not None else _SAFE_ROOTS

    # Resolve the user path.  If relative, join against *root* first.
    if os.path.isabs(user_path):
        normalised = _resolve(user_path)
    else:
        normalised = _resolve(os.path.join(roots[0], user_path))

    # ── CodeQL-recognised sanitiser barrier ──
    # ``normpath(…).startswith(safe_prefix)`` is the pattern that
    # CodeQL's ``py/path-injection`` query treats as a sanitiser.
    normalised_for_check = os.path.normcase(normalised)
    if not any(
        normalised_for_check == os.path.normcase(root)
        or normalised_for_check.startswith(_safe_prefix(root))
        for root in roots
    ):
        raise ValueError(
            f"Path escapes safe root: {user_path!r} "
            f"(resolved to {normalised!r}, roots={roots!r})"
        )

    return normalised


def safe_open(user_path: str, mode: str = "r", **kwargs):
    """Open a file after validating its path.

    Convenience wrapper around ``safe_path`` + ``open``.

    Args:
        user_path: Untrusted path string.
        mode: File open mode.
        **kwargs: Extra keyword arguments forwarded to ``open``.

    Returns:
        File object.

    Raises:
        ValueError: If the path escapes the safe root.
    """
    validated = safe_path(user_path)
    return open(validated, mode, **kwargs)  # noqa: SIM115
