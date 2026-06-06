"""Settings for DiffPitcher inference inside Audio Processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DIFFPITCHER_MODE_TEMPLATE = "template"
DIFFPITCHER_MODE_SCORE = "score"
DIFFPITCHER_MODES = (DIFFPITCHER_MODE_TEMPLATE, DIFFPITCHER_MODE_SCORE)
DIFFPITCHER_DEVICE_MODES = ("auto", "cuda", "cpu")

DIFFPITCHER_UI_KEYS: tuple[str, ...] = (
    "ap_diffpitcher_enabled",
    "ap_diffpitcher_mode",
    "ap_diffpitcher_reference_audio",
    "ap_diffpitcher_midi",
    "ap_diffpitcher_steps",
    "ap_diffpitcher_shift_semitones",
    "ap_diffpitcher_mask_with_source",
    "ap_diffpitcher_device",
)


@dataclass(frozen=True)
class DiffPitcherSettings:
    """Normalized DiffPitcher inference options."""

    enabled: bool = False
    mode: str = DIFFPITCHER_MODE_TEMPLATE
    reference_audio: str = ""
    midi_path: str = ""
    steps: int = 50
    shift_semitones: float = 0.0
    mask_with_source: bool = True
    device: str = "auto"

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-safe DiffPitcher settings."""

        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "reference_audio": self.reference_audio,
            "midi_path": self.midi_path,
            "steps": self.steps,
            "shift_semitones": self.shift_semitones,
            "mask_with_source": self.mask_with_source,
            "device": self.device,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "DiffPitcherSettings":
        """Build settings from a saved payload."""

        if not isinstance(payload, dict):
            return cls()
        return cls(
            enabled=bool(payload.get("enabled", False)),
            mode=_coerce_mode(payload.get("mode")),
            reference_audio=_coerce_path(payload.get("reference_audio")),
            midi_path=_coerce_path(payload.get("midi_path")),
            steps=_coerce_steps(payload.get("steps")),
            shift_semitones=_coerce_shift(payload.get("shift_semitones")),
            mask_with_source=_coerce_bool_default_true(
                payload.get("mask_with_source")
            ),
            device=_coerce_device(payload.get("device")),
        )

    @classmethod
    def from_ui_payload(cls, payload: dict[str, Any]) -> "DiffPitcherSettings":
        """Build settings from Audio Processing UI values."""

        return cls(
            enabled=bool(payload.get("ap_diffpitcher_enabled", False)),
            mode=_coerce_mode(payload.get("ap_diffpitcher_mode")),
            reference_audio=_coerce_path(payload.get("ap_diffpitcher_reference_audio")),
            midi_path=_coerce_path(payload.get("ap_diffpitcher_midi")),
            steps=_coerce_steps(payload.get("ap_diffpitcher_steps")),
            shift_semitones=_coerce_shift(
                payload.get("ap_diffpitcher_shift_semitones")
            ),
            mask_with_source=_coerce_bool_default_true(
                payload.get("ap_diffpitcher_mask_with_source")
            ),
            device=_coerce_device(payload.get("ap_diffpitcher_device")),
        )


def _coerce_mode(value: Any) -> str:
    """Return a supported DiffPitcher inference mode."""

    normalized = str(value or "").strip().lower()
    return normalized if normalized in DIFFPITCHER_MODES else DIFFPITCHER_MODE_TEMPLATE


def _coerce_device(value: Any) -> str:
    """Return a supported DiffPitcher device selector."""

    normalized = str(value or "").strip().lower()
    return normalized if normalized in DIFFPITCHER_DEVICE_MODES else "auto"


def _coerce_steps(value: Any) -> int:
    """Return a bounded diffusion-step count."""

    try:
        steps = int(float(value))
    except (TypeError, ValueError):
        steps = 50
    return max(1, min(100, steps))


def _coerce_shift(value: Any) -> float:
    """Return a bounded pitch-shift amount in semitones."""

    try:
        shift = float(value)
    except (TypeError, ValueError):
        shift = 0.0
    return max(-24.0, min(24.0, shift))


def _coerce_bool_default_true(value: Any) -> bool:
    """Return a boolean while preserving True as the missing-value default."""

    return True if value is None else bool(value)


def _coerce_path(value: Any) -> str:
    """Return the newest path from a Gradio-like file value."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            path = _coerce_path(item)
            if path:
                return path
        return ""
    if isinstance(value, dict):
        for key in ("path", "name"):
            path = _coerce_path(value.get(key))
            if path:
                return path
        return ""
    for attr_name in ("path", "name"):
        path = _coerce_path(getattr(value, attr_name, None))
        if path:
            return path
    return ""
