"""Settings for SAM-Audio segmentation in ACE-Step."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from acestep.audio_processing.auto_editor_trim_settings import (
    AUTO_EDITOR_MARGIN_DEFAULT_SECONDS,
    AUTO_EDITOR_MINCLIP_DEFAULT,
    AUTO_EDITOR_MINCUT_DEFAULT,
    AUTO_EDITOR_THRESHOLD_DEFAULT_DB,
    AUTO_EDITOR_THRESHOLD_MAX_DB,
    AUTO_EDITOR_THRESHOLD_MIN_DB,
    AutoEditorTrimSettings,
    coerce_auto_editor_margin_seconds,
    coerce_auto_editor_smooth_value,
    coerce_auto_editor_threshold_db,
)

from .attention import normalize_attention_backend
from .prompt_presets import DEFAULT_PROMPT
from .ranker_availability import normalize_ranker_mode
from .vram_presets import get_sam_vram_preset, normalize_sam_vram_preset

SAM_AUDIO_MIN_CHUNK_SECONDS = 1.0
SAM_AUDIO_MAX_CHUNK_SECONDS = 60.0
SAM_AUDIO_MAX_OVERLAP_SECONDS = 10.0
SAM_AUDIO_DEFAULT_TRIM_THRESHOLD_DB = AUTO_EDITOR_THRESHOLD_DEFAULT_DB
SAM_AUDIO_MIN_TRIM_THRESHOLD_DB = AUTO_EDITOR_THRESHOLD_MIN_DB
SAM_AUDIO_MAX_TRIM_THRESHOLD_DB = AUTO_EDITOR_THRESHOLD_MAX_DB
SAM_AUDIO_DEFAULT_TRIM_MARGIN_SECONDS = AUTO_EDITOR_MARGIN_DEFAULT_SECONDS
SAM_AUDIO_DEFAULT_TRIM_MINCUT = AUTO_EDITOR_MINCUT_DEFAULT
SAM_AUDIO_DEFAULT_TRIM_MINCLIP = AUTO_EDITOR_MINCLIP_DEFAULT
SAM_AUDIO_LONG_MODE_CHUNKED = "chunked"
SAM_AUDIO_LONG_MODE_MULTIDIFFUSION = "multidiffusion"

SAM_AUDIO_PRESET_KEYS: tuple[str, ...] = (
    "sam_auto_postprocess",
    "sam_preserve_original",
    "sam_trim_empty_output",
    "sam_trim_threshold_db",
    "sam_output_format",
    "sam_prompt_mode",
    "sam_prompt_preset",
    "sam_custom_prompt",
    "sam_batch_segment",
    "sam_use_span_anchor",
    "sam_anchor_json",
    "sam_anchor_polarity",
    "sam_anchor_start",
    "sam_anchor_end",
    "sam_predict_spans",
    "sam_reranking_candidates",
    "sam_ranker_mode",
    "sam_ode_steps",
    "sam_seed",
    "sam_random_seed",
    "sam_vram_preset",
    "sam_quantization",
    "sam_attention_backend",
    "sam_device_mode",
    "sam_low_vram_lite",
    "sam_chunked",
    "sam_long_audio_mode",
    "sam_chunk_seconds",
    "sam_chunk_overlap_seconds",
    "sam_subprocess",
    "sam_compile_model",
    "sam_unload_generation",
    "sam_include_residual",
    "sam_include_video",
    "sam_batch_input_folder",
    "sam_batch_output_folder",
    "sam_batch_recursive",
    "sam_batch_save_output_only",
)


@dataclass(frozen=True)
class SamAudioSettings:
    """Normalized SAM-Audio settings used by UI, batch, and generation paths."""

    auto_postprocess: bool = False
    preserve_original: bool = True
    trim_empty_output: bool = False
    trim_threshold_db: float = SAM_AUDIO_DEFAULT_TRIM_THRESHOLD_DB
    trim_margin_seconds: float = SAM_AUDIO_DEFAULT_TRIM_MARGIN_SECONDS
    trim_mincut: int = SAM_AUDIO_DEFAULT_TRIM_MINCUT
    trim_minclip: int = SAM_AUDIO_DEFAULT_TRIM_MINCLIP
    output_format: str = "mp3"
    prompt_mode: str = "text"
    prompt_preset: tuple[str, ...] = (DEFAULT_PROMPT,)
    custom_prompt: str = ""
    batch_segment: bool = False
    use_span_anchor: bool = False
    anchor_json: str = ""
    anchor_polarity: str = "+"
    anchor_start: float = 0.0
    anchor_end: float = 1.0
    predict_spans: bool = False
    reranking_candidates: int = 1
    ranker_mode: str = "none"
    ode_steps: int = 16
    seed: int = 99
    random_seed: bool = False
    vram_preset: str = "24gb_balanced"
    quantization: str = "none"
    attention_backend: str = "auto"
    device_mode: str = "auto"
    low_vram_lite: bool = False
    chunked: bool = True
    long_audio_mode: str = SAM_AUDIO_LONG_MODE_CHUNKED
    chunk_seconds: float = 20.0
    chunk_overlap_seconds: float = 5.0
    subprocess: bool = True
    compile_model: bool = False
    unload_generation: bool = True
    include_residual: bool = True
    include_video: bool = True
    batch_input_folder: str = ""
    batch_output_folder: str = ""
    batch_recursive: bool = False
    batch_save_output_only: bool = False

    @property
    def effective_prompt(self) -> str:
        """Return custom prompt text or the selected preset value."""

        custom = self.custom_prompt.strip()
        preset = self.prompt_presets[0] if self.prompt_presets else DEFAULT_PROMPT
        return custom or preset.strip() or DEFAULT_PROMPT

    @property
    def prompt_presets(self) -> tuple[str, ...]:
        """Return normalized selected Quick Prompt values."""

        return normalize_prompt_preset_selection(self.prompt_preset)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        payload = dict(self.__dict__)
        payload["prompt_preset"] = list(self.prompt_presets)
        return payload

    def trim_settings(self) -> AutoEditorTrimSettings:
        """Return Auto-Editor trim behavior settings for SAM-Audio output."""

        return AutoEditorTrimSettings(
            threshold_db=self.trim_threshold_db,
            margin_seconds=self.trim_margin_seconds,
            mincut=self.trim_mincut,
            minclip=self.trim_minclip,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "SamAudioSettings":
        """Build settings from a saved JSON-like payload."""

        if not isinstance(payload, dict):
            return cls()
        preset_name = normalize_sam_vram_preset(payload.get("vram_preset"))
        preset = get_sam_vram_preset(preset_name)
        return cls(
            auto_postprocess=bool(payload.get("auto_postprocess", False)),
            preserve_original=bool(payload.get("preserve_original", True)),
            trim_empty_output=bool(payload.get("trim_empty_output", False)),
            trim_threshold_db=_bounded_float(
                payload.get("trim_threshold_db"),
                SAM_AUDIO_DEFAULT_TRIM_THRESHOLD_DB,
                SAM_AUDIO_MIN_TRIM_THRESHOLD_DB,
                SAM_AUDIO_MAX_TRIM_THRESHOLD_DB,
            ),
            trim_margin_seconds=coerce_auto_editor_margin_seconds(
                payload.get("trim_margin_seconds")
            ),
            trim_mincut=coerce_auto_editor_smooth_value(
                payload.get("trim_mincut"),
                SAM_AUDIO_DEFAULT_TRIM_MINCUT,
            ),
            trim_minclip=coerce_auto_editor_smooth_value(
                payload.get("trim_minclip"),
                SAM_AUDIO_DEFAULT_TRIM_MINCLIP,
            ),
            output_format=_choice(
                payload.get("output_format"),
                {"wav", "flac", "mp3"},
                "mp3",
            ),
            prompt_mode=_choice(
                payload.get("prompt_mode"),
                {"text", "span", "visual"},
                "text",
            ),
            prompt_preset=normalize_prompt_preset_selection(
                payload.get("prompt_preset")
            ),
            custom_prompt=str(payload.get("custom_prompt") or ""),
            batch_segment=bool(payload.get("batch_segment", False)),
            use_span_anchor=bool(payload.get("use_span_anchor", False)),
            anchor_json=str(payload.get("anchor_json") or ""),
            anchor_polarity=_choice(payload.get("anchor_polarity"), {"+", "-"}, "+"),
            anchor_start=_float(payload.get("anchor_start"), 0.0),
            anchor_end=_float(payload.get("anchor_end"), 1.0),
            predict_spans=bool(payload.get("predict_spans", preset["predict_spans"])),
            reranking_candidates=max(
                1,
                min(
                    16,
                    int(
                        _float(
                            payload.get("reranking_candidates"),
                            preset["reranking_candidates"],
                        )
                    ),
                ),
            ),
            ranker_mode=normalize_ranker_mode(
                _choice(
                    payload.get("ranker_mode"),
                    {"none", "text_ensemble", "clap", "judge", "imagebind"},
                    str(preset["ranker_mode"]),
                )
            ),
            ode_steps=max(
                1,
                min(128, int(_float(payload.get("ode_steps"), preset["ode_steps"]))),
            ),
            seed=max(0, int(_float(payload.get("seed"), 99))),
            random_seed=bool(payload.get("random_seed", False)),
            vram_preset=preset_name,
            quantization=_choice(
                payload.get("quantization"),
                {"none", "fp8_scaled"},
                str(preset["quantization"]),
            ),
            attention_backend=normalize_attention_backend(
                payload.get("attention_backend", preset["attention_backend"])
            ),
            device_mode=_choice(
                payload.get("device_mode"),
                {"auto", "cuda", "cpu"},
                str(preset["device_mode"]),
            ),
            low_vram_lite=bool(payload.get("low_vram_lite", preset["low_vram_lite"])),
            chunked=bool(payload.get("chunked", preset["chunked"])),
            long_audio_mode=_choice(
                payload.get("long_audio_mode"),
                {SAM_AUDIO_LONG_MODE_CHUNKED, SAM_AUDIO_LONG_MODE_MULTIDIFFUSION},
                str(preset["long_audio_mode"]),
            ),
            chunk_seconds=_bounded_float(
                payload.get("chunk_seconds"),
                preset["chunk_seconds"],
                SAM_AUDIO_MIN_CHUNK_SECONDS,
                SAM_AUDIO_MAX_CHUNK_SECONDS,
            ),
            chunk_overlap_seconds=_bounded_float(
                payload.get("chunk_overlap_seconds"),
                preset["chunk_overlap_seconds"],
                0.0,
                SAM_AUDIO_MAX_OVERLAP_SECONDS,
            ),
            subprocess=bool(payload.get("subprocess", preset["subprocess"])),
            compile_model=bool(payload.get("compile_model", False)),
            unload_generation=bool(payload.get("unload_generation", True)),
            include_residual=bool(payload.get("include_residual", True)),
            include_video=bool(payload.get("include_video", True)),
            batch_input_folder=str(payload.get("batch_input_folder") or ""),
            batch_output_folder=str(payload.get("batch_output_folder") or ""),
            batch_recursive=bool(payload.get("batch_recursive", False)),
            batch_save_output_only=bool(payload.get("batch_save_output_only", False)),
        )


def settings_from_ui_values(values: tuple[Any, ...] | list[Any]) -> SamAudioSettings:
    """Build settings from values ordered by ``SAM_AUDIO_PRESET_KEYS``."""

    raw = dict(zip(SAM_AUDIO_PRESET_KEYS, values))
    payload = {
        key.removeprefix("sam_"): value
        for key, value in raw.items()
    }
    return SamAudioSettings.from_payload(payload)


def normalize_prompt_preset_selection(value: Any) -> tuple[str, ...]:
    """Return normalized Quick Prompt selections from old or new UI values."""

    if value is None:
        return (DEFAULT_PROMPT,)
    if isinstance(value, str):
        prompt = _normalize_prompt_text(value)
        return (prompt,) if prompt else (DEFAULT_PROMPT,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        values = (value,)
    prompts: list[str] = []
    for item in values:
        prompt = _normalize_prompt_text(item)
        if prompt:
            prompts.append(prompt)
    return tuple(prompts)


def with_auto_editor_trim_settings(
    settings: SamAudioSettings,
    trim_settings: AutoEditorTrimSettings,
) -> SamAudioSettings:
    """Return SAM-Audio settings with shared Auto-Editor trim behavior applied."""

    return replace(
        settings,
        trim_threshold_db=coerce_auto_editor_threshold_db(trim_settings.threshold_db),
        trim_margin_seconds=coerce_auto_editor_margin_seconds(
            trim_settings.margin_seconds
        ),
        trim_mincut=coerce_auto_editor_smooth_value(
            trim_settings.mincut,
            SAM_AUDIO_DEFAULT_TRIM_MINCUT,
        ),
        trim_minclip=coerce_auto_editor_smooth_value(
            trim_settings.minclip,
            SAM_AUDIO_DEFAULT_TRIM_MINCLIP,
        ),
    )


def _choice(value: Any, allowed: set[str], fallback: str) -> str:
    """Return a normalized allowed string."""

    normalized = str(value or "").strip()
    return normalized if normalized in allowed else fallback


def _normalize_prompt_text(value: Any) -> str:
    """Normalize one prompt text value for text conditioning."""

    return " ".join(str(value or "").strip().lower().split())


def _float(value: Any, fallback: float) -> float:
    """Return a finite float or fallback."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    is_finite = result == result and result not in (float("inf"), float("-inf"))
    return result if is_finite else fallback


def _bounded_float(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    """Return a finite float clamped to a closed interval."""

    return max(minimum, min(maximum, _float(value, fallback)))
