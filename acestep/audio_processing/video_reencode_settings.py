"""Video reencode settings for Auto-Editor based video processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


VIDEO_REENCODE_UI_KEYS: tuple[str, ...] = (
    "ap_video_auto_quality",
    "ap_video_codec",
    "ap_video_bitrate",
    "ap_video_crf",
    "ap_video_preset",
    "ap_video_audio_codec",
    "ap_video_audio_bitrate",
)

VIDEO_ENCODER_PRESET_CHOICES: tuple[str, ...] = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)

DEFAULT_VIDEO_AUTO_QUALITY = True
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_VIDEO_BITRATE = ""
DEFAULT_VIDEO_CRF = 18
DEFAULT_VIDEO_PRESET = "medium"
DEFAULT_VIDEO_AUDIO_CODEC = "aac"
DEFAULT_VIDEO_AUDIO_BITRATE = "192k"

@dataclass(frozen=True)
class VideoReencodeSettings:
    """Normalized Auto-Editor video reencode settings."""

    auto_set_quality: bool = DEFAULT_VIDEO_AUTO_QUALITY
    video_codec: str = DEFAULT_VIDEO_CODEC
    video_bitrate: str = DEFAULT_VIDEO_BITRATE
    video_crf: int = DEFAULT_VIDEO_CRF
    video_preset: str = DEFAULT_VIDEO_PRESET
    audio_codec: str = DEFAULT_VIDEO_AUDIO_CODEC
    audio_bitrate: str = DEFAULT_VIDEO_AUDIO_BITRATE

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload for saved presets and metadata."""

        return {
            "auto_set_quality": self.auto_set_quality,
            "video_codec": self.video_codec,
            "video_bitrate": self.video_bitrate,
            "video_crf": self.video_crf,
            "video_preset": self.video_preset,
            "audio_codec": self.audio_codec,
            "audio_bitrate": self.audio_bitrate,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "VideoReencodeSettings":
        """Build settings from a saved JSON-like payload."""

        source = payload if isinstance(payload, Mapping) else {}
        return cls(
            auto_set_quality=_coerce_bool(
                source.get("auto_set_quality"),
                DEFAULT_VIDEO_AUTO_QUALITY,
            ),
            video_codec=_coerce_text(source.get("video_codec"), DEFAULT_VIDEO_CODEC),
            video_bitrate=_coerce_text(source.get("video_bitrate"), DEFAULT_VIDEO_BITRATE),
            video_crf=_coerce_crf(source.get("video_crf"), DEFAULT_VIDEO_CRF),
            video_preset=_coerce_preset(source.get("video_preset")),
            audio_codec=_coerce_text(source.get("audio_codec"), DEFAULT_VIDEO_AUDIO_CODEC),
            audio_bitrate=_coerce_text(
                source.get("audio_bitrate"),
                DEFAULT_VIDEO_AUDIO_BITRATE,
            ),
        )

    @classmethod
    def from_ui_payload(cls, payload: Mapping[str, Any]) -> "VideoReencodeSettings":
        """Build settings from the flattened Gradio Audio Processing payload."""

        return cls(
            auto_set_quality=_coerce_bool(
                payload.get("ap_video_auto_quality"),
                DEFAULT_VIDEO_AUTO_QUALITY,
            ),
            video_codec=_coerce_text(payload.get("ap_video_codec"), DEFAULT_VIDEO_CODEC),
            video_bitrate=_coerce_text(
                payload.get("ap_video_bitrate"),
                DEFAULT_VIDEO_BITRATE,
            ),
            video_crf=_coerce_crf(payload.get("ap_video_crf"), DEFAULT_VIDEO_CRF),
            video_preset=_coerce_preset(payload.get("ap_video_preset")),
            audio_codec=_coerce_text(
                payload.get("ap_video_audio_codec"),
                DEFAULT_VIDEO_AUDIO_CODEC,
            ),
            audio_bitrate=_coerce_text(
                payload.get("ap_video_audio_bitrate"),
                DEFAULT_VIDEO_AUDIO_BITRATE,
            ),
        )


def _coerce_bool(value: Any, fallback: bool) -> bool:
    """Return a predictable boolean from common UI and JSON values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"", "0", "false", "no", "off", "none"}:
            return False
    if value is None:
        return fallback
    return bool(value)


def _coerce_text(value: Any, fallback: str) -> str:
    """Return stripped text, preserving empty strings when they are defaults."""

    if value is None:
        return fallback
    return str(value).strip()


def _coerce_crf(value: Any, fallback: int) -> int:
    """Return an Auto-Editor compatible CRF value."""

    try:
        crf = int(round(float(value)))
    except (TypeError, ValueError):
        return fallback
    return max(0, min(63, crf))


def _coerce_preset(value: Any) -> str:
    """Return a supported encoder preset."""

    normalized = str(value or DEFAULT_VIDEO_PRESET).strip().casefold()
    return normalized if normalized in VIDEO_ENCODER_PRESET_CHOICES else DEFAULT_VIDEO_PRESET
