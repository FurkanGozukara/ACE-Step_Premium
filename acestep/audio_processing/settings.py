"""Settings and presets for ACE-Step audio processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .presets import (
    DEFAULT_STAGE_VALUES,
    OUTPUT_FORMAT_CHOICES,
    PRESET_VALUES,
    STAGE_KEYS,
)

UI_SETTING_KEYS: tuple[str, ...] = (
    "ap_auto_postprocess",
    "ap_preserve_original",
    "ap_output_format",
    "ap_builtin_preset",
    *tuple(item for key in STAGE_KEYS for item in (f"ap_{key}_enabled", f"ap_{key}")),
)


@dataclass(frozen=True)
class AudioProcessingSettings:
    """Container for stage settings used by manual and generated-song processing.

    Args:
        enabled: Whether generated-song post-processing is active.
        preserve_original: Whether original generated files remain beside processed output.
        output_format: Output format for processed audio.
        preset: Name of the selected built-in preset.
        stages_enabled: Per-stage enabled flags keyed by stage name.
        values: Per-stage numeric values keyed by stage name.

    Returns:
        Immutable settings object with normalized stage values.
    """

    enabled: bool = False
    preserve_original: bool = True
    output_format: str = "wav"
    preset: str = "Generic AI"
    stages_enabled: dict[str, bool] = field(
        default_factory=lambda: {key: True for key in STAGE_KEYS}
    )
    values: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STAGE_VALUES))

    def stage_enabled(self, key: str) -> bool:
        """Return whether a processing stage should run."""

        return bool(self.stages_enabled.get(key, True))

    def stage_value(self, key: str) -> float:
        """Return a normalized numeric value for a processing stage."""

        return float(self.values.get(key, DEFAULT_STAGE_VALUES[key]))

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe settings payload."""

        return {
            "enabled": self.enabled,
            "preserve_original": self.preserve_original,
            "output_format": self.output_format,
            "preset": self.preset,
            "stages_enabled": dict(self.stages_enabled),
            "values": dict(self.values),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "AudioProcessingSettings":
        """Build settings from a saved JSON-like payload."""

        if not isinstance(payload, dict):
            return cls()
        values = _coerce_stage_values(payload.get("values", {}))
        enabled = _coerce_stage_enabled(payload.get("stages_enabled", {}))
        return cls(
            enabled=bool(payload.get("enabled", False)),
            preserve_original=bool(payload.get("preserve_original", True)),
            output_format=_coerce_output_format(payload.get("output_format")),
            preset=str(payload.get("preset") or "Generic AI"),
            stages_enabled=enabled,
            values=values,
        )


def preset_values(name: str | None) -> dict[str, float]:
    """Return stage values for a built-in preset name."""

    return dict(PRESET_VALUES.get(str(name or ""), DEFAULT_STAGE_VALUES))


def settings_from_ui_values(values: tuple[Any, ...] | list[Any]) -> AudioProcessingSettings:
    """Build settings from values ordered by ``UI_SETTING_KEYS``."""

    payload = dict(zip(UI_SETTING_KEYS, values))
    return AudioProcessingSettings(
        enabled=bool(payload.get("ap_auto_postprocess")),
        preserve_original=bool(payload.get("ap_preserve_original", True)),
        output_format=_coerce_output_format(payload.get("ap_output_format")),
        preset=str(payload.get("ap_builtin_preset") or "Generic AI"),
        stages_enabled={
            key: bool(payload.get(f"ap_{key}_enabled", True)) for key in STAGE_KEYS
        },
        values={
            key: _coerce_float(payload.get(f"ap_{key}"), DEFAULT_STAGE_VALUES[key])
            for key in STAGE_KEYS
        },
    )


def _coerce_output_format(value: Any) -> str:
    """Return a supported processed-audio output format."""

    normalized = str(value or "wav").strip().lower()
    return normalized if normalized in {"wav", "flac", "mp3"} else "wav"


def _coerce_stage_values(raw_values: Any) -> dict[str, float]:
    """Return numeric stage values with defaults filled in."""

    source = raw_values if isinstance(raw_values, dict) else {}
    return {
        key: _coerce_float(source.get(key), DEFAULT_STAGE_VALUES[key])
        for key in STAGE_KEYS
    }


def _coerce_stage_enabled(raw_values: Any) -> dict[str, bool]:
    """Return stage enabled flags with defaults filled in."""

    source = raw_values if isinstance(raw_values, dict) else {}
    return {key: bool(source.get(key, True)) for key in STAGE_KEYS}


def _coerce_float(value: Any, fallback: float) -> float:
    """Return a finite float or a fallback."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if result != result or result in (float("inf"), float("-inf")):
        return fallback
    return result
