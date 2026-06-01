"""Settings for SAM-Audio segmentation in ACE-Step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .attention import normalize_attention_backend
from .prompt_presets import DEFAULT_PROMPT
from .vram_presets import get_sam_vram_preset, normalize_sam_vram_preset

SAM_AUDIO_MIN_CHUNK_SECONDS = 1.0
SAM_AUDIO_MAX_CHUNK_SECONDS = 60.0
SAM_AUDIO_MAX_OVERLAP_SECONDS = 10.0

SAM_AUDIO_PRESET_KEYS: tuple[str, ...] = (
    "sam_auto_postprocess",
    "sam_preserve_original",
    "sam_output_format",
    "sam_prompt_mode",
    "sam_prompt_preset",
    "sam_custom_prompt",
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
    "sam_chunk_seconds",
    "sam_chunk_overlap_seconds",
    "sam_subprocess",
    "sam_unload_generation",
    "sam_include_residual",
    "sam_include_video",
    "sam_batch_input_folder",
    "sam_batch_output_folder",
    "sam_batch_recursive",
)


@dataclass(frozen=True)
class SamAudioSettings:
    """Normalized SAM-Audio settings used by UI, batch, and generation paths."""

    auto_postprocess: bool = False
    preserve_original: bool = True
    output_format: str = "wav"
    prompt_mode: str = "text"
    prompt_preset: str = DEFAULT_PROMPT
    custom_prompt: str = ""
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
    chunk_seconds: float = 10.0
    chunk_overlap_seconds: float = 1.0
    subprocess: bool = True
    unload_generation: bool = True
    include_residual: bool = True
    include_video: bool = True
    batch_input_folder: str = ""
    batch_output_folder: str = ""
    batch_recursive: bool = False

    @property
    def effective_prompt(self) -> str:
        """Return custom prompt text or the selected preset value."""

        custom = self.custom_prompt.strip()
        return custom or self.prompt_preset.strip() or DEFAULT_PROMPT

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload."""

        return dict(self.__dict__)

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
            output_format=_choice(
                payload.get("output_format"),
                {"wav", "flac", "mp3"},
                "wav",
            ),
            prompt_mode=_choice(
                payload.get("prompt_mode"),
                {"text", "span", "visual"},
                "text",
            ),
            prompt_preset=str(payload.get("prompt_preset") or DEFAULT_PROMPT),
            custom_prompt=str(payload.get("custom_prompt") or ""),
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
            ranker_mode=_choice(
                payload.get("ranker_mode"),
                {"none", "text_ensemble", "clap", "judge", "imagebind"},
                str(preset["ranker_mode"]),
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
            unload_generation=bool(payload.get("unload_generation", True)),
            include_residual=bool(payload.get("include_residual", True)),
            include_video=bool(payload.get("include_video", True)),
            batch_input_folder=str(payload.get("batch_input_folder") or ""),
            batch_output_folder=str(payload.get("batch_output_folder") or ""),
            batch_recursive=bool(payload.get("batch_recursive", False)),
        )


def settings_from_ui_values(values: tuple[Any, ...] | list[Any]) -> SamAudioSettings:
    """Build settings from values ordered by ``SAM_AUDIO_PRESET_KEYS``."""

    raw = dict(zip(SAM_AUDIO_PRESET_KEYS, values))
    payload = {
        key.removeprefix("sam_"): value
        for key, value in raw.items()
    }
    return SamAudioSettings.from_payload(payload)


def _choice(value: Any, allowed: set[str], fallback: str) -> str:
    """Return a normalized allowed string."""

    normalized = str(value or "").strip()
    return normalized if normalized in allowed else fallback


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
