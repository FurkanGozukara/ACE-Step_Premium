"""LoRA training artifact naming helpers."""

from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")
_VALID_NAME_RE = re.compile(r"^[\w][\w -]{0,79}$", re.UNICODE)
_INVALID_FILENAME_CHARS = '<>:"/\\|?*'


def normalize_lora_name(raw_name: str | None) -> str:
    """Return a display/file-safe LoRA training name.

    Args:
        raw_name: User-provided LoRA name.

    Returns:
        The stripped name with internal whitespace collapsed.
    """

    return _WHITESPACE_RE.sub(" ", str(raw_name or "").strip())


def validate_lora_name(raw_name: str | None) -> tuple[str, str | None]:
    """Validate a LoRA training name for checkpoint and file names.

    Args:
        raw_name: User-provided LoRA name.

    Returns:
        Tuple of normalized name and optional validation error.
    """

    name = normalize_lora_name(raw_name)
    if not name:
        return "", "Please enter a LoRA training name."
    if name in {".", ".."}:
        return name, "LoRA training name cannot be '.' or '..'."
    if name.endswith((" ", ".")):
        return name, "LoRA training name cannot end with a space or dot."
    if not _VALID_NAME_RE.fullmatch(name):
        return (
            name,
            "Use letters, numbers, spaces, hyphens, or underscores only "
            "(80 characters max).",
        )
    return name, None


def lora_epoch_name(lora_name: str, epoch: int) -> str:
    """Return the base artifact name for one LoRA epoch.

    Args:
        lora_name: Validated LoRA training name.
        epoch: One-based epoch number.

    Returns:
        Name in ``<lora_name>-epoch-<epoch>`` form.
    """

    return f"{lora_name}-epoch-{int(epoch)}"


def lora_training_state_filename(epoch: int, suffix: str = "") -> str:
    """Return the resume-state filename for a LoRA training epoch.

    Args:
        epoch: One-based epoch number.
        suffix: Optional filename suffix such as ``"final"``.

    Returns:
        Name in ``epoch-<epoch>-training_resume_state[-suffix].pt`` form.
    """

    suffix_text = str(suffix or "").strip()
    suffix_part = f"-{suffix_text}" if suffix_text else ""
    return f"epoch-{int(epoch)}-training_resume_state{suffix_part}.pt"


def lora_safetensors_filename(artifact_name: str) -> str:
    """Return a safe safetensors filename for a basename-only artifact."""

    name = str(artifact_name or "").strip()
    if (
        not name
        or name in {".", ".."}
        or name.endswith((".", " "))
        or any(char in name for char in _INVALID_FILENAME_CHARS)
    ):
        raise ValueError(f"Invalid LoRA artifact name: {artifact_name!r}")
    return f"{name}.safetensors"
