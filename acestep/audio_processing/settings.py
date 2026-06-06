"""Settings and presets for ACE-Step audio processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .auto_editor_trim_settings import (
    AUTO_EDITOR_MARGIN_DEFAULT_SECONDS,
    AUTO_EDITOR_MINCLIP_DEFAULT,
    AUTO_EDITOR_MINCUT_DEFAULT,
    AUTO_EDITOR_THRESHOLD_DEFAULT_DB,
    AutoEditorTrimSettings,
    coerce_auto_editor_margin_seconds,
    coerce_auto_editor_smooth_value,
    coerce_auto_editor_threshold_db,
)
from .auto_editor_workflow import (
    AUTO_EDITOR_WORKFLOW_EXPORT_KEY,
    AUTO_EDITOR_WORKFLOW_EXPORT_NONE,
    normalize_workflow_export_mode,
)
from .diffpitcher_settings import DIFFPITCHER_UI_KEYS, DiffPitcherSettings
from .presets import DEFAULT_STAGE_VALUES, PRESET_VALUES, STAGE_KEYS
from .settings_coercion import (
    coerce_float_value,
    coerce_output_format,
    coerce_stage_enabled,
    coerce_stage_values,
)
from .video_reencode_settings import VIDEO_REENCODE_UI_KEYS, VideoReencodeSettings

UI_SETTING_KEYS: tuple[str, ...] = (
    "ap_auto_postprocess",
    "ap_preserve_original",
    "ap_output_format",
    "ap_export_audio_only",
    "ap_trim_empty_output",
    "ap_trim_threshold_db",
    "ap_trim_margin_seconds",
    "ap_trim_mincut",
    "ap_trim_minclip",
    AUTO_EDITOR_WORKFLOW_EXPORT_KEY,
    *VIDEO_REENCODE_UI_KEYS,
    *DIFFPITCHER_UI_KEYS,
    "ap_builtin_preset",
    *tuple(item for key in STAGE_KEYS for item in (f"ap_{key}_enabled", f"ap_{key}")),
)

AUDIO_PROCESSING_DEFAULT_TRIM_THRESHOLD_DB = AUTO_EDITOR_THRESHOLD_DEFAULT_DB
AUDIO_PROCESSING_DEFAULT_TRIM_MARGIN_SECONDS = AUTO_EDITOR_MARGIN_DEFAULT_SECONDS
AUDIO_PROCESSING_DEFAULT_TRIM_MINCUT = AUTO_EDITOR_MINCUT_DEFAULT
AUDIO_PROCESSING_DEFAULT_TRIM_MINCLIP = AUTO_EDITOR_MINCLIP_DEFAULT


@dataclass(frozen=True)
class AudioProcessingSettings:
    """Normalized manual and generated-song audio-processing settings."""

    enabled: bool = False
    preserve_original: bool = True
    output_format: str = "wav"
    export_audio_only: bool = False
    trim_empty_output: bool = False
    trim_threshold_db: float = AUDIO_PROCESSING_DEFAULT_TRIM_THRESHOLD_DB
    trim_margin_seconds: float = AUDIO_PROCESSING_DEFAULT_TRIM_MARGIN_SECONDS
    trim_mincut: int = AUDIO_PROCESSING_DEFAULT_TRIM_MINCUT
    trim_minclip: int = AUDIO_PROCESSING_DEFAULT_TRIM_MINCLIP
    workflow_export: str = AUTO_EDITOR_WORKFLOW_EXPORT_NONE
    video_reencode: VideoReencodeSettings = field(default_factory=VideoReencodeSettings)
    diffpitcher: DiffPitcherSettings = field(default_factory=DiffPitcherSettings)
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
            "export_audio_only": self.export_audio_only,
            "trim_empty_output": self.trim_empty_output,
            "trim_threshold_db": self.trim_threshold_db,
            "trim_margin_seconds": self.trim_margin_seconds,
            "trim_mincut": self.trim_mincut,
            "trim_minclip": self.trim_minclip,
            "workflow_export": self.workflow_export,
            "video_reencode": self.video_reencode.to_payload(),
            "diffpitcher": self.diffpitcher.to_payload(),
            "preset": self.preset,
            "stages_enabled": dict(self.stages_enabled),
            "values": dict(self.values),
        }

    def trim_settings(self) -> AutoEditorTrimSettings:
        """Return auto-editor settings for optional trimming."""

        return AutoEditorTrimSettings(
            threshold_db=self.trim_threshold_db,
            margin_seconds=self.trim_margin_seconds,
            mincut=self.trim_mincut,
            minclip=self.trim_minclip,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "AudioProcessingSettings":
        """Build settings from a saved JSON-like payload."""

        if not isinstance(payload, dict):
            return cls()
        values = coerce_stage_values(payload.get("values", {}))
        enabled = coerce_stage_enabled(payload.get("stages_enabled", {}))
        return cls(
            enabled=bool(payload.get("enabled", False)),
            preserve_original=bool(payload.get("preserve_original", True)),
            output_format=coerce_output_format(payload.get("output_format")),
            export_audio_only=bool(payload.get("export_audio_only", False)),
            trim_empty_output=bool(payload.get("trim_empty_output", False)),
            trim_threshold_db=coerce_auto_editor_threshold_db(
                payload.get("trim_threshold_db")
            ),
            trim_margin_seconds=coerce_auto_editor_margin_seconds(
                payload.get("trim_margin_seconds")
            ),
            trim_mincut=coerce_auto_editor_smooth_value(
                payload.get("trim_mincut"),
                AUDIO_PROCESSING_DEFAULT_TRIM_MINCUT,
            ),
            trim_minclip=coerce_auto_editor_smooth_value(
                payload.get("trim_minclip"),
                AUDIO_PROCESSING_DEFAULT_TRIM_MINCLIP,
            ),
            workflow_export=normalize_workflow_export_mode(payload.get("workflow_export")),
            video_reencode=VideoReencodeSettings.from_payload(payload.get("video_reencode")),
            diffpitcher=DiffPitcherSettings.from_payload(payload.get("diffpitcher")),
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
        output_format=coerce_output_format(payload.get("ap_output_format")),
        export_audio_only=bool(payload.get("ap_export_audio_only", False)),
        trim_empty_output=bool(payload.get("ap_trim_empty_output", False)),
        trim_threshold_db=coerce_auto_editor_threshold_db(
            payload.get("ap_trim_threshold_db")
        ),
        trim_margin_seconds=coerce_auto_editor_margin_seconds(
            payload.get("ap_trim_margin_seconds")
        ),
        trim_mincut=coerce_auto_editor_smooth_value(
            payload.get("ap_trim_mincut"),
            AUDIO_PROCESSING_DEFAULT_TRIM_MINCUT,
        ),
        trim_minclip=coerce_auto_editor_smooth_value(
            payload.get("ap_trim_minclip"),
            AUDIO_PROCESSING_DEFAULT_TRIM_MINCLIP,
        ),
        workflow_export=normalize_workflow_export_mode(
            payload.get(AUTO_EDITOR_WORKFLOW_EXPORT_KEY)
        ),
        video_reencode=VideoReencodeSettings.from_ui_payload(payload),
        diffpitcher=DiffPitcherSettings.from_ui_payload(payload),
        preset=str(payload.get("ap_builtin_preset") or "Generic AI"),
        stages_enabled={
            key: bool(payload.get(f"ap_{key}_enabled", True)) for key in STAGE_KEYS
        },
        values={
            key: coerce_float_value(payload.get(f"ap_{key}"), DEFAULT_STAGE_VALUES[key])
            for key in STAGE_KEYS
        },
    )
