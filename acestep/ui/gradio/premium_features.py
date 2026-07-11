"""Premium Gradio shell helpers: presets, dashboard stats, and folder actions."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gradio as gr

from acestep.audio_processing.auto_editor_trim_settings import (
    AUTO_EDITOR_MINCLIP_DEFAULT,
    AUTO_EDITOR_MINCUT_DEFAULT,
    AUTO_EDITOR_THRESHOLD_DEFAULT_DB,
    coerce_auto_editor_margin_seconds,
    coerce_auto_editor_smooth_value,
    coerce_auto_editor_threshold_db,
)
from acestep.gpu_config import (
    GPU_TIER_LABELS,
    find_best_lm_model_on_disk,
    get_global_gpu_config,
)
from acestep.model_downloader import (
    DEFAULT_BASE_DIT_MODEL,
    DEFAULT_LM_MODEL,
    DEFAULT_PREMIUM_DIT_MODEL,
    DEFAULT_TURBO_DIT_MODEL,
    LEGACY_BF16_BASE_DIT_MODEL,
    LEGACY_BF16_PREMIUM_DIT_MODEL,
    LEGACY_BF16_TURBO_DIT_MODEL,
    SOURCE_BASE_DIT_MODEL,
    SOURCE_PREMIUM_DIT_MODEL,
    SOURCE_TURBO_DIT_MODEL,
    get_models_dir,
)
from acestep.torch_compile_workers import (
    DEFAULT_COMPILE_THREADS,
    normalize_compile_threads,
)
from acestep.core.generation.handler.lora.folder_scan import (
    lora_dropdown_choices,
    resolve_loadable_lora_adapter_path,
)
from acestep.training.dataset_vram_presets import (
    DEFAULT_DATASET_VRAM_PRESET,
    default_dataset_vram_preset_name,
)
from acestep.training.optim import (
    OPTIMIZER_CHOICES,
    optimizer_hyperparameter_defaults,
)
from acestep.constants import MODE_TO_TASK_TYPE
from acestep.ui.gradio.events.generation.model_config import (
    get_ui_control_config_for_path,
)
from acestep.ui.gradio.events.generation.remix_presets import (
    REMIX_PRESET_CHOICES,
    REMIX_PRESET_DEFAULT,
    normalize_remix_preset,
    remix_preset_elem_classes,
    remix_preset_values,
)
from acestep.ui.gradio.events.generation.audio_format_options import (
    normalize_audio_format,
    normalize_extract_audio_format,
)
from acestep.ui.gradio.events.dcw_defaults import get_dcw_defaults_for_think
from acestep.ui.gradio.events.generation.quantization import default_quantization_value
from acestep.ui.gradio.events.results.output_manager import get_results_dir
from acestep.ui.gradio.premium_preset_defaults import ADDITIONAL_DEFAULT_PRESET_VALUES
from acestep.ui.gradio.premium_preset_schema import (
    ADDITIONAL_PRESET_COMPONENT_KEYS,
    FILE_UPLOAD_PRESET_KEYS,
)
from acestep.ui.gradio.premium_preset_value_safety import coerce_preset_value


GPU_OPTIMIZATION_PRESET_NAME = "GPU Optimization Preset"
USER_PRESET_FOLDER = "premium_user_presets"
LAST_USED_PRESET_FILE = ".last_used_preset.txt"
TRIM_THRESHOLD_PRESET_KEYS = frozenset(
    {
        "extract_trim_threshold_db",
        "ap_trim_threshold_db",
        "sam_trim_threshold_db",
    }
)
_LORA_OPTIMIZER_PRESET_KEY_MAP: dict[str, str] = {
    "lora_weight_decay": "weight_decay",
    "lora_adam_beta1": "adam_beta1",
    "lora_adam_beta2": "adam_beta2",
    "lora_adam_epsilon": "adam_epsilon",
    "lora_adamw8bit_min_8bit_size": "adamw8bit_min_8bit_size",
    "lora_adamw8bit_percentile_clipping": "adamw8bit_percentile_clipping",
    "lora_adamw8bit_block_wise": "adamw8bit_block_wise",
    "lora_adamw8bit_paged": "adamw8bit_paged",
    "lora_adafactor_epsilon1": "adafactor_epsilon1",
    "lora_adafactor_epsilon2": "adafactor_epsilon2",
    "lora_adafactor_clip_threshold": "adafactor_clip_threshold",
    "lora_adafactor_decay_rate": "adafactor_decay_rate",
    "lora_adafactor_beta1": "adafactor_beta1",
    "lora_adafactor_scale_parameter": "adafactor_scale_parameter",
    "lora_adafactor_relative_step": "adafactor_relative_step",
    "lora_adafactor_warmup_init": "adafactor_warmup_init",
}
_LORA_ADAM_PRESET_KEYS = frozenset(
    {
        "lora_adam_beta1",
        "lora_adam_beta2",
        "lora_adam_epsilon",
    }
)
_LORA_ADAMW8BIT_PRESET_KEYS = frozenset(
    {
        "lora_adamw8bit_min_8bit_size",
        "lora_adamw8bit_percentile_clipping",
        "lora_adamw8bit_block_wise",
        "lora_adamw8bit_paged",
    }
)
_LORA_ADAFACTOR_PRESET_KEYS = frozenset(
    {
        "lora_adafactor_epsilon1",
        "lora_adafactor_epsilon2",
        "lora_adafactor_clip_threshold",
        "lora_adafactor_decay_rate",
        "lora_adafactor_beta1",
        "lora_adafactor_scale_parameter",
        "lora_adafactor_relative_step",
        "lora_adafactor_warmup_init",
    }
)
TRIM_MARGIN_PRESET_KEYS = frozenset({"ap_trim_margin_seconds"})
TRIM_MINCUT_PRESET_KEYS = frozenset({"ap_trim_mincut"})
TRIM_MINCLIP_PRESET_KEYS = frozenset({"ap_trim_minclip"})
GPU_TIER_PRESET_KEYS = frozenset({"tier_dropdown", "simple_create_tier_dropdown"})
DEFAULT_PRESET_CAPTION = (
    "Conscious melodic rap and modern cinematic pop-rap anthem, uplifting and humble, "
    "warm synth pads, punchy drums, deep 808 bass, subtle clean guitar textures, big "
    "melodic hook, soulful confident male vocal, polished wide stereo radio mix."
)
DEFAULT_PRESET_LYRICS = """[intro-medium]

[verse]
Woke up with the fire, put the weight on my back
No gold on my wrist, still I'm dressed in facts
Hands in the dirt, put the work on track
Built from the ground, never life on lack

I don't need the noise, I don't move for praise
I just keep it pure in a world that fades
Calm in the storm, let the heart stay brave
Planting good seeds, let the roots make waves

[chorus]
I came with peace, I came with light
Hands stay clean and my soul stays right
Work all day, still I move real kind
Head held low but I aim sky-high

No big talk, I don't need that crown
Strong stand tall, keep it humble now
Brick by brick, yeah I built this sound
Quiet on top, let the truth get loud

[verse]
I was down, I was dust, had to learn my lane
Now I move with grace, never chasing fame
If I eat, we eat, let the whole team gain
Put love in the soil, let it heal that pain

Never flex too hard, let the code stay deep
Made peace with the grind while the world can sleep
Heart stay soft but the mind don't creep
What I earn, what I keep, what I sow, I reap

[chorus]
I came with peace, I came with light
Hands stay clean and my soul stays right
Work all day, still I move real kind
Head held low but I aim sky-high

No big talk, I don't need that crown
Strong stand tall, keep it humble now
Brick by brick, yeah I built this sound
Quiet on top, let the truth get loud

[inst-short]

[bridge]
I don't wanna shine if my people can't glow
I don't want the win if it costs my soul
Real ones rise but they still stay low
Deep roots grow where the wild winds blow

I got faith in the good, got strength in the calm
Got scars in my skin but peace in my arms
I don't move in hate, I don't feed that dark
I just light one flame, let it spread through hearts

[chorus]
I came with peace, I came with light
Hands stay clean and my soul stays right
Work all day, still I move real kind
Head held low but I aim sky-high

No big talk, I don't need that crown
Strong stand tall, keep it humble now
Brick by brick, yeah I built this sound
Quiet on top, let the truth get loud

[chorus]
I came with peace, I came with light
Hands stay clean and my soul stays right
Work all day, still I move real kind
Head held low but I aim sky-high

No big talk, I don't need that crown
Strong stand tall, keep it humble now
Brick by brick, yeah I built this sound
Quiet on top, let the truth get loud

[outro-medium]"""


def _selected_lora_optimizer(optimizer_type: object) -> str:
    selected = str(optimizer_type or "adamw").strip().casefold()
    return selected if selected in OPTIMIZER_CHOICES else "adamw"


def _lora_optimizer_preset_key_visible(key: str, optimizer_type: str) -> bool:
    if key == "lora_weight_decay":
        return True
    if key in _LORA_ADAM_PRESET_KEYS:
        return True
    if key in _LORA_ADAMW8BIT_PRESET_KEYS:
        return True
    if key in _LORA_ADAFACTOR_PRESET_KEYS:
        return optimizer_type == "adafactor"
    return True

SIMPLE_MODEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("ACEStep XL 1.5 SFT", DEFAULT_PREMIUM_DIT_MODEL),
    ("ACEStep XL 1.5 Turbo", DEFAULT_TURBO_DIT_MODEL),
    ("ACEStep XL 1.5 Base", DEFAULT_BASE_DIT_MODEL),
)
SIMPLE_MODEL_VALUES = frozenset(value for _label, value in SIMPLE_MODEL_CHOICES)
SIMPLE_MODEL_ALIASES = {
    SOURCE_PREMIUM_DIT_MODEL: DEFAULT_PREMIUM_DIT_MODEL,
    SOURCE_TURBO_DIT_MODEL: DEFAULT_TURBO_DIT_MODEL,
    SOURCE_BASE_DIT_MODEL: DEFAULT_BASE_DIT_MODEL,
    LEGACY_BF16_PREMIUM_DIT_MODEL: DEFAULT_PREMIUM_DIT_MODEL,
    LEGACY_BF16_TURBO_DIT_MODEL: DEFAULT_TURBO_DIT_MODEL,
    LEGACY_BF16_BASE_DIT_MODEL: DEFAULT_BASE_DIT_MODEL,
}


def _gpu_tier_value_from_config(gpu_config: Any) -> str:
    """Return a persisted tier value supported by the current UI."""

    tier = str(getattr(gpu_config, "tier", "") or "").strip()
    return tier if tier in GPU_TIER_LABELS else "tier1"


def _current_gpu_tier_value() -> str:
    """Return the detected GPU tier, falling back to CPU when detection fails."""

    try:
        return _gpu_tier_value_from_config(get_global_gpu_config())
    except Exception:
        return "tier1"


def _coerce_gpu_tier_preset_value(value: Any) -> str:
    """Coerce saved VRAM tier values without demoting legacy values to tier1."""

    tier = str(value or "").strip()
    return tier if tier in GPU_TIER_LABELS else _current_gpu_tier_value()

PRESET_COMPONENT_KEYS: tuple[str, ...] = (
    "language_dropdown",
    "tier_dropdown",
    "checkpoint_dropdown",
    "config_path",
    "simple_model_dropdown",
    "device",
    "lm_model_path",
    "backend_dropdown",
    "init_llm_checkbox",
    "use_flash_attention_checkbox",
    "offload_to_cpu_checkbox",
    "offload_dit_to_cpu_checkbox",
    "compile_model_checkbox",
    "compile_threads_slider",
    "quantization_checkbox",
    "simple_quantization",
    "mlx_dit_checkbox",
    "lora_path",
    "lora_dropdown",
    "simple_lora_dropdown",
    "lora_scale_slider",
    "simple_lora_scale_slider",
    "generation_mode",
    "simple_query_input",
    "simple_vocal_language",
    "simple_instrumental_checkbox",
    "reference_audio",
    "captions",
    "lyrics",
    "instrumental_checkbox",
    "repaint_dont_switch_with_lyrics",
    "src_audio",
    "track_name",
    "extract_all_stems",
    "complete_track_classes",
    "text2music_audio_code_string",
    "audio_cover_strength",
    "cover_noise_strength",
    "remix_preset",
    "repainting_start",
    "repainting_end",
    "repaint_mode",
    "repaint_strength",
    "bpm",
    "key_scale",
    "time_signature",
    "vocal_language",
    "bpm_auto",
    "key_auto",
    "timesig_auto",
    "vocal_lang_auto",
    "audio_duration",
    "batch_size_input",
    "duration_auto",
    "inference_steps",
    "guidance_scale",
    "infer_method",
    "sampler_mode",
    "velocity_norm_threshold",
    "velocity_ema_factor",
    "dcw_enabled",
    "dcw_mode",
    "dcw_scaler",
    "dcw_high_scaler",
    "dcw_wavelet",
    "use_adg",
    "shift",
    "custom_timesteps",
    "cfg_interval_start",
    "cfg_interval_end",
    "seed",
    "random_seed_checkbox",
    "extract_output_format",
    "audio_format",
    "mp3_bitrate",
    "mp3_sample_rate",
    "score_scale",
    "enable_normalization",
    "normalization_db",
    "fade_in_duration",
    "fade_out_duration",
    "extract_trim_empty_output",
    "extract_trim_threshold_db",
    "latent_shift",
    "latent_rescale",
    "lm_temperature",
    "lm_cfg_scale",
    "lm_top_k",
    "lm_top_p",
    "lm_negative_prompt",
    "use_cot_metas",
    "use_cot_language",
    "constrained_decoding_debug",
    "allow_lm_batch",
    "use_cot_caption",
    "think_checkbox",
    "generate_lm_audio_codes",
    "auto_score",
    "autogen_checkbox",
    "auto_lrc",
    "lm_batch_chunk_size",
    "subprocess_mode_checkbox",
    "dataset_model_config",
    "dataset_vram_preset",
    "dataset_name",
    "all_instrumental",
    "format_lyrics",
    "transcribe_lyrics",
    "lm_lyrics_language",
    "custom_tag",
    "use_only_custom_trigger",
    "tag_position",
    "genre_ratio",
    "skip_metas",
    "only_unlabeled",
    "auto_label_output_dir",
    "auto_label_subprocess",
    "auto_label_batch_size",
    *ADDITIONAL_PRESET_COMPONENT_KEYS,
)

DEFAULT_PRESET_VALUES: dict[str, Any] = {
    "language_dropdown": "en",
    "compile_threads_slider": DEFAULT_COMPILE_THREADS,
    "config_path": DEFAULT_TURBO_DIT_MODEL,
    "simple_model_dropdown": DEFAULT_TURBO_DIT_MODEL,
    "lm_model_path": "",
    "generation_mode": "Remix",
    "captions": DEFAULT_PRESET_CAPTION,
    "lyrics": DEFAULT_PRESET_LYRICS,
    "instrumental_checkbox": False,
    "simple_vocal_language": "en",
    "simple_create_negative_prompt": "",
    "vocal_language": "en",
    "lm_negative_prompt": "",
    "use_cot_caption": False,
    "use_cot_language": False,
    "repaint_dont_switch_with_lyrics": False,
    "audio_cover_strength": remix_preset_values(REMIX_PRESET_DEFAULT)[0],
    "cover_noise_strength": remix_preset_values(REMIX_PRESET_DEFAULT)[1],
    "remix_preset": REMIX_PRESET_DEFAULT,
    "batch_size_input": 1,
    "inference_steps": 8,
    "guidance_scale": 1.0,
    "infer_method": "ode",
    "sampler_mode": "heun",
    "simple_create_sampler_mode": "heun",
    "velocity_norm_threshold": 0.0,
    "velocity_ema_factor": 0.0,
    "dcw_enabled": True,
    "dcw_mode": "double",
    "dcw_scaler": 0.02,
    "dcw_high_scaler": 0.06,
    "dcw_wavelet": "haar",
    "use_adg": False,
    "shift": 3.0,
    "custom_timesteps": "",
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "extract_all_stems": False,
    "extract_output_format": "mp3",
    "audio_format": "mp3",
    "mp3_bitrate": "256k",
    "mp3_sample_rate": 48000,
    "enable_normalization": True,
    "normalization_db": -1.0,
    "extract_trim_empty_output": False,
    "extract_trim_threshold_db": AUTO_EDITOR_THRESHOLD_DEFAULT_DB,
    "think_checkbox": True,
    "generate_lm_audio_codes": True,
    "allow_lm_batch": True,
    "quantization_checkbox": "none",
    "simple_quantization": "none",
    "lora_path": "",
    "lora_dropdown": "",
    "simple_lora_dropdown": "",
    "lora_scale_slider": 1.0,
    "simple_lora_scale_slider": 1.0,
    "auto_score": False,
    "auto_lrc": False,
    "autogen_checkbox": False,
    "subprocess_mode_checkbox": False,
    "dataset_model_config": DEFAULT_TURBO_DIT_MODEL,
    "dataset_vram_preset": DEFAULT_DATASET_VRAM_PRESET,
    "dataset_name": "my_lora_dataset",
    "all_instrumental": False,
    "format_lyrics": False,
    "transcribe_lyrics": True,
    "lm_lyrics_language": "en",
    "custom_tag": "",
    "use_only_custom_trigger": False,
    "tag_position": "prepend",
    "genre_ratio": 0,
    "skip_metas": False,
    "only_unlabeled": True,
    "auto_label_output_dir": "",
    "auto_label_subprocess": True,
    "auto_label_batch_size": 1,
    **ADDITIONAL_DEFAULT_PRESET_VALUES,
}

def normalize_simple_model_dropdown_value(value: Any) -> str:
    """Return a supported Create-tab model selector value."""

    resolved = _resolve_simple_model_reference(value)
    return resolved or DEFAULT_TURBO_DIT_MODEL


def _resolve_simple_model_reference(value: Any) -> str | None:
    """Return the BF16 Simple-tab model value for known model names."""

    model_path = str(value or "").strip()
    if model_path in SIMPLE_MODEL_VALUES:
        return model_path
    return SIMPLE_MODEL_ALIASES.get(model_path)


def model_quality_defaults(model_path: Any) -> dict[str, Any]:
    """Return generation-control defaults for a model path."""

    selected_model = str(model_path or "").strip() or DEFAULT_TURBO_DIT_MODEL
    selected_model_lower = selected_model.lower()
    is_turbo = "turbo" in selected_model_lower
    is_sft = "sft" in selected_model_lower and not is_turbo
    uses_lm_defaults = is_turbo or is_sft
    dcw_enabled = is_turbo
    dcw_defaults = (
        get_dcw_defaults_for_think(uses_lm_defaults)
        if dcw_enabled
        else {"mode": "double", "scaler": 0.0, "high_scaler": 0.0}
    )
    cfg = get_ui_control_config_for_path(selected_model)
    return {
        "inference_steps": cfg["inference_steps_value"],
        "guidance_scale": cfg["guidance_scale_value"],
        "infer_method": DEFAULT_PRESET_VALUES["infer_method"],
        "sampler_mode": DEFAULT_PRESET_VALUES["sampler_mode"],
        "velocity_norm_threshold": DEFAULT_PRESET_VALUES["velocity_norm_threshold"],
        "velocity_ema_factor": DEFAULT_PRESET_VALUES["velocity_ema_factor"],
        "use_adg": cfg["use_adg_value"],
        "shift": cfg["shift_value"],
        "custom_timesteps": DEFAULT_PRESET_VALUES["custom_timesteps"],
        "cfg_interval_start": cfg["cfg_interval_start_value"],
        "cfg_interval_end": cfg["cfg_interval_end_value"],
        "init_lm_checkbox": uses_lm_defaults,
        "think_checkbox": uses_lm_defaults,
        "generate_lm_audio_codes": True,
        "use_cot_metas": uses_lm_defaults,
        "use_cot_caption": False,
        "use_cot_language": False,
        "allow_lm_batch": is_turbo,
        "dcw_enabled": dcw_enabled,
        "dcw_mode": dcw_defaults["mode"],
        "dcw_scaler": dcw_defaults["scaler"],
        "dcw_high_scaler": dcw_defaults["high_scaler"],
        "dcw_wavelet": DEFAULT_PRESET_VALUES["dcw_wavelet"],
    }


def _runtime_default_values(base_values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return defaults that depend on the current installation path."""

    values = dict(base_values or DEFAULT_PRESET_VALUES)
    values["lm_model_path"] = _runtime_lm_model_default()
    values["dataset_vram_preset"] = default_dataset_vram_preset_name()
    values["lora_output_dir"] = str(_project_root() / "Loras")
    try:
        gpu_config = get_global_gpu_config()
        tier = _gpu_tier_value_from_config(gpu_config)
        values["tier_dropdown"] = tier
        values["simple_create_tier_dropdown"] = tier
        values["mlx_vae_chunk_size"] = gpu_config.mlx_vae_chunk_size
        if not values.get("device"):
            values["device"] = "cpu" if tier == "tier1" else "auto"
    except Exception:
        pass
    return {
        **values,
        "checkpoint_dropdown": str(get_models_dir(project_root=_project_root())),
    }


def _available_lm_models_on_disk() -> list[str]:
    """Return installed 5Hz LM model directory names."""
    models_dir = get_models_dir(project_root=_project_root())
    if not models_dir.exists():
        return []
    return sorted(
        path.name
        for path in models_dir.iterdir()
        if path.is_dir() and path.name.startswith("acestep-5Hz-lm-")
    )


def _runtime_lm_model_default() -> str:
    """Return the current GPU-tier LM recommendation when available locally."""
    try:
        gpu_config = get_global_gpu_config()
        recommended_lm = getattr(gpu_config, "recommended_lm_model", "")
        disk_models = _available_lm_models_on_disk()
        selected = find_best_lm_model_on_disk(recommended_lm, disk_models)
        return selected or recommended_lm or DEFAULT_LM_MODEL
    except Exception:
        return DEFAULT_LM_MODEL


def get_preset_component_keys() -> tuple[str, ...]:
    """Return the stable component-key order used for preset serialization."""
    return PRESET_COMPONENT_KEYS


def _project_root() -> Path:
    raw = os.environ.get("ACESTEP_PROJECT_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _user_preset_dir() -> Path:
    target = _project_root() / USER_PRESET_FOLDER
    target.mkdir(parents=True, exist_ok=True)
    return target


def _last_used_path() -> Path:
    return _user_preset_dir() / LAST_USED_PRESET_FILE


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(name or "").strip())
    cleaned = cleaned.strip(" ._")
    return cleaned or "preset"


def _preset_name_key(name: str | None) -> str:
    """Return a forgiving comparison key for preset names."""

    raw = str(name or "").strip()
    sanitized = _sanitize_name(raw)
    return re.sub(r"[\s_-]+", " ", sanitized.casefold()).strip()


def _user_preset_path(name: str | None) -> Path:
    """Return the canonical user-preset path for a preset name."""

    return _user_preset_dir() / f"{_sanitize_name(name or '')}.json"


def ensure_default_preset() -> Path:
    """Return the user preset directory; bundled immutable presets are removed."""
    return _user_preset_dir()


def _read_preset_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        values = payload.get("values", payload) if isinstance(payload, dict) else {}
        if not isinstance(values, dict):
            return None
        return values
    except Exception:
        return None


def list_preset_names() -> list[str]:
    """Return saved user preset names."""
    names: list[str] = []
    seen: set[str] = set()
    for preset_path in sorted(_user_preset_dir().glob("*.json")):
        name = preset_path.stem
        key = _preset_name_key(name)
        if key in seen:
            continue
        names.append(name)
        seen.add(key)
    return names


def get_last_used_preset_name() -> str | None:
    path = _last_used_path()
    if not path.exists():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
        return value or None
    except Exception:
        return None


def set_last_used_preset_name(name: str) -> None:
    _last_used_path().write_text(str(name or "").strip(), encoding="utf-8")


def _clear_last_used_preset_name() -> None:
    """Clear remembered user-preset selection."""
    path = _last_used_path()
    if path.exists():
        path.unlink()


def _resolve_available_preset_name(name: str | None) -> tuple[str, bool]:
    """Resolve a requested preset name to an existing selectable preset.

    Returns:
        A tuple of ``(resolved_name, fell_back_to_default)``.
    """

    requested = str(name or "").strip()
    if not requested:
        return "", False
    sanitized = _sanitize_name(requested)
    if _user_preset_path(sanitized).exists():
        return sanitized, False
    return "", True


def _apply_runtime_defaults(
    payload: dict[str, Any],
    base_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill missing preset keys from runtime-aware defaults."""

    provided_keys = set(payload)
    merged = dict(payload)
    defaults = _runtime_default_values(base_values)
    for key, value in defaults.items():
        if key not in provided_keys or merged.get(key) is None:
            merged[key] = value
    merged["compile_threads_slider"] = normalize_compile_threads(
        merged.get("compile_threads_slider")
    )
    config_path = str(merged.get("config_path") or "").strip()
    raw_simple_model = str(payload.get("simple_model_dropdown") or "").strip()
    simple_model = str(merged.get("simple_model_dropdown") or "").strip()
    config_simple_model = _resolve_simple_model_reference(config_path)
    if raw_simple_model not in SIMPLE_MODEL_VALUES and config_simple_model:
        simple_model = config_simple_model
    simple_model = normalize_simple_model_dropdown_value(simple_model)
    merged["simple_model_dropdown"] = simple_model
    if not config_path or config_simple_model:
        merged["config_path"] = simple_model
    quality_model = str(merged.get("config_path") or "").strip() or simple_model
    if (
        "dataset_model_config" not in provided_keys
        or merged.get("dataset_model_config") in (None, "")
    ):
        merged["dataset_model_config"] = quality_model
    quality_defaults = model_quality_defaults(quality_model)
    for key, value in quality_defaults.items():
        if key not in provided_keys or merged.get(key) is None:
            merged[key] = value
    generation_mode = str(merged.get("generation_mode") or "Custom").strip()
    if generation_mode in MODE_TO_TASK_TYPE:
        merged["task_type"] = MODE_TO_TASK_TYPE[generation_mode]
    if (
        "lora_optimizer_type" not in provided_keys
        and "lora_use_8bit_adam" in payload
    ):
        merged["lora_optimizer_type"] = (
            "adamw8bit" if _legacy_bool(payload.get("lora_use_8bit_adam")) else "adamw"
        )
    lora_optimizer_defaults = optimizer_hyperparameter_defaults(
        merged.get("lora_optimizer_type")
    )
    for ui_key, optimizer_key in _LORA_OPTIMIZER_PRESET_KEY_MAP.items():
        if ui_key not in provided_keys or merged.get(ui_key) in (None, ""):
            merged[ui_key] = lora_optimizer_defaults[optimizer_key]
    merged["remix_preset"] = normalize_remix_preset(merged.get("remix_preset"))
    if generation_mode == "Remix":
        remix_strength, melody_retention = remix_preset_values(
            merged["remix_preset"]
        )
        if "audio_cover_strength" not in provided_keys or merged.get(
            "audio_cover_strength"
        ) is None:
            merged["audio_cover_strength"] = remix_strength
        if "cover_noise_strength" not in provided_keys or merged.get(
            "cover_noise_strength"
        ) is None:
            merged["cover_noise_strength"] = melody_retention
    else:
        merged["cover_noise_strength"] = 0.0
    _apply_gpu_tier_preset_migration(merged, provided_keys)
    _apply_cross_tab_defaults(merged, provided_keys)
    _sync_vocal_language_preset_values(merged, provided_keys)
    _sync_negative_prompt_preset_values(merged, provided_keys)
    _sync_sampler_mode_preset_values(merged, provided_keys)
    raw_quantization = payload.get("quantization_checkbox")
    raw_simple_quantization = payload.get("simple_quantization")
    if raw_quantization in (None, "") and raw_simple_quantization not in (None, ""):
        merged["quantization_checkbox"] = raw_simple_quantization
    merged["quantization_checkbox"] = default_quantization_value(
        merged.get("quantization_checkbox")
    )
    if "simple_quantization" not in provided_keys:
        merged["simple_quantization"] = merged["quantization_checkbox"]
    else:
        merged["simple_quantization"] = default_quantization_value(
            merged.get("simple_quantization")
        )
    if not merged.get("simple_lora_dropdown") and merged.get("lora_dropdown"):
        merged["simple_lora_dropdown"] = merged.get("lora_dropdown", "")
    elif not merged.get("lora_dropdown") and merged.get("simple_lora_dropdown"):
        merged["lora_dropdown"] = merged.get("simple_lora_dropdown", "")
    raw_lora_scale = payload.get("lora_scale_slider")
    raw_simple_lora_scale = payload.get("simple_lora_scale_slider")
    if raw_simple_lora_scale in (None, "") and raw_lora_scale not in (None, ""):
        merged["simple_lora_scale_slider"] = merged.get("lora_scale_slider", 1.0)
    elif raw_lora_scale in (None, "") and raw_simple_lora_scale not in (None, ""):
        merged["lora_scale_slider"] = merged.get("simple_lora_scale_slider", 1.0)
    _clamp_trim_threshold_defaults(merged)
    return merged


def _apply_gpu_tier_preset_migration(
    merged: dict[str, Any],
    provided_keys: set[str],
) -> None:
    """Map removed or unknown saved VRAM presets to the detected GPU tier."""

    provided_tier_values = [
        str(merged.get(key) or "").strip()
        for key in GPU_TIER_PRESET_KEYS
        if key in provided_keys and merged.get(key) not in (None, "")
    ]
    if any(tier not in GPU_TIER_LABELS for tier in provided_tier_values):
        tier = _current_gpu_tier_value()
        for key in GPU_TIER_PRESET_KEYS:
            merged[key] = tier
        return
    fallback_tier = next(
        (tier for tier in provided_tier_values if tier in GPU_TIER_LABELS),
        _current_gpu_tier_value(),
    )
    for key in GPU_TIER_PRESET_KEYS:
        if key not in merged or merged.get(key) in (None, ""):
            merged[key] = fallback_tier
        else:
            merged[key] = _coerce_gpu_tier_preset_value(merged.get(key))


def _clamp_trim_threshold_defaults(payload: dict[str, Any]) -> None:
    """Clamp persisted auto-editor trim settings to supported ranges."""

    for key in TRIM_THRESHOLD_PRESET_KEYS:
        if key in payload:
            payload[key] = coerce_auto_editor_threshold_db(payload.get(key))
    for key in TRIM_MARGIN_PRESET_KEYS:
        if key in payload:
            payload[key] = coerce_auto_editor_margin_seconds(payload.get(key))
    for key in TRIM_MINCUT_PRESET_KEYS:
        if key in payload:
            payload[key] = coerce_auto_editor_smooth_value(
                payload.get(key),
                AUTO_EDITOR_MINCUT_DEFAULT,
            )
    for key in TRIM_MINCLIP_PRESET_KEYS:
        if key in payload:
            payload[key] = coerce_auto_editor_smooth_value(
                payload.get(key),
                AUTO_EDITOR_MINCLIP_DEFAULT,
            )


def _legacy_bool(value: Any) -> bool:
    """Parse legacy boolean preset fields."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)


def _apply_cross_tab_defaults(merged: dict[str, Any], provided_keys: set[str]) -> None:
    """Backfill newly tracked tab-specific fields without overwriting saved blanks."""

    _copy_missing_value(
        merged,
        provided_keys,
        "simple_create_caption",
        "captions",
        DEFAULT_PRESET_CAPTION,
    )
    _copy_missing_value(
        merged,
        provided_keys,
        "simple_create_lyrics",
        "lyrics",
        DEFAULT_PRESET_LYRICS,
    )
    _copy_missing_value(
        merged,
        provided_keys,
        "simple_create_tier_dropdown",
        "tier_dropdown",
    )
    _copy_missing_value(
        merged,
        provided_keys,
        "simple_create_vocal_language",
        "vocal_language",
        "en",
    )
    _copy_missing_value(
        merged,
        provided_keys,
        "simple_create_negative_prompt",
        "lm_negative_prompt",
        "",
    )
    _copy_missing_value(
        merged,
        provided_keys,
        "simple_create_sampler_mode",
        "sampler_mode",
        "heun",
    )
    _copy_missing_value(
        merged,
        provided_keys,
        "simple_create_instrumental",
        "instrumental_checkbox",
        False,
    )
    _copy_missing_value(
        merged,
        provided_keys,
        "simple_create_duration",
        "audio_duration",
        -1,
    )
    _copy_missing_value(
        merged,
        provided_keys,
        "simple_create_batch_size",
        "batch_size_input",
        1,
    )
    _copy_missing_value(
        merged,
        provided_keys,
        "simple_create_random_seed",
        "random_seed_checkbox",
        True,
    )
    _copy_missing_value(merged, provided_keys, "simple_create_seed", "seed", "-1")
    _copy_missing_value(
        merged,
        provided_keys,
        "lora_sample_prompt",
        "captions",
        DEFAULT_PRESET_CAPTION,
    )
    _copy_missing_value(
        merged,
        provided_keys,
        "lora_sample_lyrics",
        "lyrics",
        DEFAULT_PRESET_LYRICS,
    )


def _copy_missing_value(
    merged: dict[str, Any],
    provided_keys: set[str],
    target_key: str,
    source_key: str,
    fallback: Any = "",
) -> None:
    """Copy a source value to a missing target preset key."""

    if target_key in provided_keys and merged.get(target_key) is not None:
        return
    merged[target_key] = merged.get(source_key, fallback)


def _sync_vocal_language_preset_values(
    merged: dict[str, Any],
    provided_keys: set[str],
) -> None:
    """Keep Generate Song and advanced vocal-language preset values aligned."""

    value = _preferred_vocal_language_preset_value(merged, provided_keys)
    merged["vocal_language"] = value
    merged["simple_vocal_language"] = value
    merged["simple_create_vocal_language"] = value


def _preferred_vocal_language_preset_value(
    merged: dict[str, Any],
    provided_keys: set[str],
) -> str:
    """Choose the most explicit saved vocal-language value."""

    candidates: list[Any] = []
    if "vocal_language" in provided_keys:
        candidates.append(merged.get("vocal_language"))
    if "simple_vocal_language" in provided_keys:
        candidates.append(merged.get("simple_vocal_language"))
    if "simple_create_vocal_language" in provided_keys:
        candidates.append(merged.get("simple_create_vocal_language"))
    candidates.extend(
        [
            merged.get("vocal_language"),
            merged.get("simple_vocal_language"),
            merged.get("simple_create_vocal_language"),
        ]
    )
    saw_unknown = False
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        if text.lower() == "unknown":
            saw_unknown = True
            continue
        return text
    return "unknown" if saw_unknown else "en"


def _sync_negative_prompt_preset_values(
    merged: dict[str, Any],
    provided_keys: set[str],
) -> None:
    """Keep Generate Song and advanced negative-prompt preset values aligned."""

    value = _preferred_negative_prompt_preset_value(merged, provided_keys)
    merged["lm_negative_prompt"] = value
    merged["simple_create_negative_prompt"] = value


def _preferred_negative_prompt_preset_value(
    merged: dict[str, Any],
    provided_keys: set[str],
) -> str:
    """Choose the saved negative prompt, preserving old empty/sentinel defaults."""

    candidates: list[Any] = []
    if "simple_create_negative_prompt" in provided_keys:
        candidates.append(merged.get("simple_create_negative_prompt"))
    if "lm_negative_prompt" in provided_keys:
        candidates.append(merged.get("lm_negative_prompt"))
    candidates.extend(
        [
            merged.get("simple_create_negative_prompt"),
            merged.get("lm_negative_prompt"),
        ]
    )
    for candidate in candidates:
        text = _normalize_negative_prompt_preset_value(candidate)
        if text:
            return text
    return ""


def _normalize_negative_prompt_preset_value(value: Any) -> str:
    """Return a trimmed preset value, mapping the old sentinel to blank."""

    text = str(value or "").strip()
    return "" if text.upper() == "NO USER INPUT" else text


def _sync_sampler_mode_preset_values(
    merged: dict[str, Any],
    provided_keys: set[str],
) -> None:
    """Keep Generate Song and advanced sampler preset values aligned."""

    value = _preferred_sampler_mode_preset_value(merged, provided_keys)
    merged["sampler_mode"] = value
    merged["simple_create_sampler_mode"] = value


def _preferred_sampler_mode_preset_value(
    merged: dict[str, Any],
    provided_keys: set[str],
) -> str:
    """Choose the saved sampler value, defaulting new presets to Heun."""

    candidates: list[Any] = []
    if "simple_create_sampler_mode" in provided_keys:
        candidates.append(merged.get("simple_create_sampler_mode"))
    if "sampler_mode" in provided_keys:
        candidates.append(merged.get("sampler_mode"))
    candidates.extend(
        [
            merged.get("simple_create_sampler_mode"),
            merged.get("sampler_mode"),
        ]
    )
    for candidate in candidates:
        value = _normalize_sampler_mode_preset_value(candidate)
        if value:
            return value
    return "heun"


def _normalize_sampler_mode_preset_value(value: Any) -> str:
    """Return a supported sampler mode, or blank when no usable value exists."""

    text = str(value or "").strip().lower()
    return text if text in {"euler", "heun"} else ""


def _load_resolved_preset_payload(name: str | None) -> tuple[str, bool, dict[str, Any]]:
    """Load an existing user preset, or return an empty GPU-default fallback."""

    preset_name, fell_back = _resolve_available_preset_name(name)
    if not preset_name:
        return "", fell_back, {}

    payload = _read_preset_file(_user_preset_path(preset_name))
    if payload is None:
        return "", True, {}
    return preset_name, fell_back, _apply_runtime_defaults(payload)


def load_named_preset(name: str | None) -> dict[str, Any]:
    """Load user preset values, returning an empty payload when none is selected."""
    _preset_name, _fell_back, payload = _load_resolved_preset_payload(name)
    return payload


def _values_to_payload(values: tuple[Any, ...]) -> dict[str, Any]:
    return {
        key: value
        for key, value in zip(PRESET_COMPONENT_KEYS, values)
    }


def _payload_to_component_updates(
    payload: dict[str, Any],
    component_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Any]:
    updates: list[Any] = []
    selected_model = payload.get("simple_model_dropdown") or payload.get("config_path")
    ui_config = get_ui_control_config_for_path(selected_model or DEFAULT_TURBO_DIT_MODEL)
    selected_lora_optimizer = _selected_lora_optimizer(payload.get("lora_optimizer_type"))
    for key in PRESET_COMPONENT_KEYS:
        if key in payload:
            value = payload[key]
            if key in FILE_UPLOAD_PRESET_KEYS:
                value = _safe_file_upload_value(value)
            if key in {"quantization_checkbox", "simple_quantization"}:
                value = default_quantization_value(value)
            if key in GPU_TIER_PRESET_KEYS:
                updates.append(gr.update(value=_coerce_gpu_tier_preset_value(value)))
                continue
            if key in {"lora_dropdown", "simple_lora_dropdown"}:
                choices = lora_dropdown_choices(_project_root())
                valid_values = {choice[1] for choice in choices}
                if value and value not in valid_values:
                    resolved = resolve_loadable_lora_adapter_path(value)
                    if resolved:
                        choices.append((Path(resolved).name, resolved))
                        value = resolved
                        valid_values.add(resolved)
                updates.append(
                    gr.update(
                        choices=choices,
                        value=value if value in valid_values else "",
                    )
                )
            elif key == "inference_steps":
                value = coerce_preset_value(
                    key,
                    value,
                    component_specs,
                    minimum=ui_config["inference_steps_minimum"],
                    maximum=ui_config["inference_steps_maximum"],
                    default=ui_config["inference_steps_value"],
                )
                updates.append(
                    gr.update(
                        value=value,
                        minimum=ui_config["inference_steps_minimum"],
                        maximum=ui_config["inference_steps_maximum"],
                        step=1,
                    )
                )
            elif key == "guidance_scale":
                value = coerce_preset_value(
                    key,
                    value,
                    component_specs,
                    minimum=ui_config["guidance_scale_minimum"],
                    maximum=ui_config["guidance_scale_maximum"],
                    default=ui_config["guidance_scale_value"],
                )
                updates.append(
                    gr.update(
                        value=value,
                        minimum=ui_config["guidance_scale_minimum"],
                        maximum=ui_config["guidance_scale_maximum"],
                        step=ui_config["guidance_scale_step"],
                        visible=ui_config["guidance_scale_visible"],
                    )
                )
            elif key == "use_adg":
                updates.append(
                    gr.update(value=value, visible=ui_config["use_adg_visible"])
                )
            elif key == "shift":
                value = coerce_preset_value(
                    key,
                    value,
                    component_specs,
                    minimum=ui_config["shift_minimum"],
                    maximum=ui_config["shift_maximum"],
                    default=ui_config["shift_value"],
                )
                updates.append(
                    gr.update(
                        value=value,
                        minimum=ui_config["shift_minimum"],
                        maximum=ui_config["shift_maximum"],
                        step=ui_config["shift_step"],
                        visible=ui_config["shift_visible"],
                    )
                )
            elif key == "cfg_interval_start":
                value = coerce_preset_value(
                    key,
                    value,
                    component_specs,
                    minimum=ui_config["cfg_interval_start_minimum"],
                    maximum=ui_config["cfg_interval_start_maximum"],
                    default=ui_config["cfg_interval_start_value"],
                )
                updates.append(
                    gr.update(
                        value=value,
                        minimum=ui_config["cfg_interval_start_minimum"],
                        maximum=ui_config["cfg_interval_start_maximum"],
                        step=ui_config["cfg_interval_start_step"],
                        visible=ui_config["cfg_interval_start_visible"],
                    )
                )
            elif key == "cfg_interval_end":
                value = coerce_preset_value(
                    key,
                    value,
                    component_specs,
                    minimum=ui_config["cfg_interval_end_minimum"],
                    maximum=ui_config["cfg_interval_end_maximum"],
                    default=ui_config["cfg_interval_end_value"],
                )
                updates.append(
                    gr.update(
                        value=value,
                        minimum=ui_config["cfg_interval_end_minimum"],
                        maximum=ui_config["cfg_interval_end_maximum"],
                        step=ui_config["cfg_interval_end_step"],
                        visible=ui_config["cfg_interval_end_visible"],
                    )
                )
            elif key == "generation_mode":
                choices = ui_config["generation_mode_choices"]
                value = coerce_preset_value(
                    key,
                    value,
                    component_specs,
                    choices=choices,
                    default="Custom",
                )
                updates.append(
                    gr.update(
                        choices=choices,
                        value=value,
                    )
                )
            elif key == "remix_preset":
                updates.append(
                    gr.update(
                        choices=list(REMIX_PRESET_CHOICES),
                        value=normalize_remix_preset(value),
                        visible=True,
                        elem_classes=remix_preset_elem_classes(
                            payload.get("generation_mode") == "Remix"
                        ),
                    )
                )
            elif key in TRIM_THRESHOLD_PRESET_KEYS:
                value = coerce_auto_editor_threshold_db(value)
                updates.append(gr.update(value=value))
            elif key in TRIM_MARGIN_PRESET_KEYS:
                value = coerce_auto_editor_margin_seconds(value)
                updates.append(gr.update(value=value))
            elif key in TRIM_MINCUT_PRESET_KEYS:
                value = coerce_auto_editor_smooth_value(value, AUTO_EDITOR_MINCUT_DEFAULT)
                updates.append(gr.update(value=value))
            elif key in TRIM_MINCLIP_PRESET_KEYS:
                value = coerce_auto_editor_smooth_value(value, AUTO_EDITOR_MINCLIP_DEFAULT)
                updates.append(gr.update(value=value))
            elif key in _LORA_OPTIMIZER_PRESET_KEY_MAP:
                updates.append(
                    gr.update(
                        value=coerce_preset_value(key, value, component_specs),
                        visible=_lora_optimizer_preset_key_visible(
                            key,
                            selected_lora_optimizer,
                        ),
                    )
                )
            elif key == "audio_format":
                updates.append(gr.update(value=normalize_audio_format(value)))
            elif key == "extract_output_format":
                updates.append(gr.update(value=normalize_extract_audio_format(value)))
            else:
                value = coerce_preset_value(key, value, component_specs)
                updates.append(gr.update(value=value))
        else:
            updates.append(gr.skip())
    return updates


def _safe_file_upload_value(value: Any) -> Any:
    """Return a Gradio-safe value for file, image, and audio upload controls."""

    if value in (None, ""):
        return None
    if isinstance(value, dict):
        candidate = value.get("path") or value.get("name")
        return value if _is_existing_file(candidate) else None
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        return str(path) if _is_existing_file(path) else None
    return None


def _is_existing_file(value: Any) -> bool:
    """Return whether a value points at a readable filesystem file."""

    if value in (None, ""):
        return False
    try:
        return Path(value).expanduser().is_file()
    except (OSError, TypeError, ValueError):
        return False


def _lora_status_from_payload(payload: dict[str, Any]) -> tuple[str, Any]:
    """Return LoRA status text plus the hidden legacy checkbox update."""

    candidate = str(payload.get("lora_path") or "").strip() or str(
        payload.get("lora_dropdown")
        or payload.get("simple_lora_dropdown")
        or ""
    ).strip()
    if not candidate:
        return "No LoRA will be used.", gr.update(value=False)
    resolved = resolve_loadable_lora_adapter_path(candidate)
    if not resolved:
        return f"No LoRA will be used. Invalid LoRA path: {candidate}", gr.update(value=False)
    return f"Next run will use LoRA: {resolved}", gr.update(value=True)


def _write_user_preset(name: str, payload: dict[str, Any]) -> str:
    sanitized = _sanitize_name(name)
    target = _user_preset_path(sanitized)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "_meta": {
                    "name": sanitized,
                    "immutable": False,
                    "format": "ace_step_premium_preset",
                    "version": 1,
                },
                "values": payload,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    set_last_used_preset_name(sanitized)
    return sanitized


def _build_dashboard_markdown(
    selected_preset: str | None = None,
    subprocess_mode: bool | None = None,
) -> str:
    models_dir = get_models_dir(project_root=_project_root())
    outputs_dir = get_results_dir()
    preset_name = selected_preset or GPU_OPTIMIZATION_PRESET_NAME
    execution_mode = "Subprocess" if subprocess_mode else "In-process"
    gpu_config = get_global_gpu_config()
    tier = getattr(gpu_config, "tier", "unknown")

    model_dirs = sorted(
        child.name for child in models_dir.iterdir()
        if child.is_dir()
    ) if models_dir.exists() else []
    run_dirs = sorted(
        child.name for child in outputs_dir.iterdir()
        if child.is_dir() and child.name.isdigit()
    ) if outputs_dir.exists() else []

    try:
        free_gb = shutil.disk_usage(outputs_dir).free / (1024 ** 3)
    except Exception:
        free_gb = 0.0

    return "\n".join(
        [
            "### Studio Overview",
            f"- Preset: `{preset_name}`",
            f"- GPU optimization preset: `{tier}`",
            f"- Execution mode: `{execution_mode}`",
            f"- Models folder: `{models_dir}`",
            f"- Output folder: `{outputs_dir}`",
            f"- Available model folders: `{len(model_dirs)}`",
            f"- Saved generations: `{len(run_dirs)}`",
            f"- Last generation folder: `{run_dirs[-1] if run_dirs else 'None yet'}`",
            f"- Free disk space near outputs: `{free_gb:.1f} GB`",
        ]
    )


def refresh_dashboard(selected_preset: str | None, subprocess_mode: bool | None) -> str:
    """Refresh the Studio markdown summary."""
    return _build_dashboard_markdown(selected_preset, subprocess_mode)


def open_folder_in_system(path: str | Path) -> str:
    """Open a folder with the platform file explorer."""

    try:
        target = _prepare_folder_target(path)
    except (OSError, TypeError, ValueError) as exc:
        return f"Failed to prepare folder `{path}`: {exc}"

    try:
        _launch_folder_in_system(target)
        return f"Opened `{target}`"
    except FileNotFoundError:
        return f"Folder is available at `{target}`, but no system file explorer was found."
    except Exception as exc:
        return f"Failed to open `{target}`: {exc}"


def _prepare_folder_target(path: str | Path) -> Path:
    """Return an existing folder path, using a file parent when needed."""

    target = Path(path).expanduser()
    if target.exists() and target.is_file():
        target = target.parent
    target = target.resolve(strict=False)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _launch_folder_in_system(target: Path) -> None:
    """Launch the platform file explorer for a prepared folder path."""

    if sys.platform == "win32":
        _launch_windows_folder(target)
        return

    command = _folder_open_command()
    if command is None:
        raise FileNotFoundError("No supported file explorer command was found.")
    subprocess.Popen(
        [*command, str(target)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _launch_windows_folder(target: Path) -> None:
    """Launch Windows Explorer for a prepared folder path."""

    startfile = getattr(os, "startfile", None)
    if startfile is not None:
        startfile(str(target))
        return

    explorer = shutil.which("explorer") or shutil.which("explorer.exe")
    if explorer is None:
        raise FileNotFoundError("No Windows Explorer command was found.")
    subprocess.Popen(
        [explorer, str(target)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _folder_open_command() -> tuple[str, ...] | None:
    """Return the first available non-Windows folder opener command."""

    candidates: tuple[tuple[str, ...], ...]
    if sys.platform == "darwin":
        candidates = (("open",),)
    else:
        candidates = (
            ("xdg-open",),
            ("gio", "open"),
            ("kde-open",),
            ("gnome-open",),
            ("wslview",),
            ("explorer.exe",),
        )

    for command in candidates:
        executable = shutil.which(command[0])
        if executable:
            return (executable, *command[1:])
    return None


def open_models_folder() -> str:
    """Open the local models directory in the system file explorer."""
    return open_folder_in_system(get_models_dir(project_root=_project_root()))


def open_outputs_folder() -> str:
    """Open the local generation outputs directory in the system file explorer."""
    return open_folder_in_system(get_results_dir())


def startup_preset_updates(
    component_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Any, ...]:
    """Return startup preset updates plus dropdown, status, and dashboard."""
    _ = component_specs
    remembered = get_last_used_preset_name()
    choices = list_preset_names()
    if remembered:
        _clear_last_used_preset_name()
    return (
        *[gr.skip() for _ in PRESET_COMPONENT_KEYS],
        "No LoRA will be used.",
        gr.update(value=False),
        gr.update(choices=choices, value=None),
        f"Using {GPU_OPTIMIZATION_PRESET_NAME}.",
        _build_dashboard_markdown(None, None),
    )


def save_preset_action(
    preset_name_input: str,
    current_selection: str | None,
    *values: Any,
) -> tuple[Any, ...]:
    """Save a user preset and refresh the dropdown/status."""
    requested_name = str(preset_name_input or current_selection or "").strip()
    if not requested_name:
        return (
            gr.update(choices=list_preset_names(), value=current_selection),
            "Enter a preset name before saving.",
            _build_dashboard_markdown(current_selection, None),
        )
    payload = _values_to_payload(values)
    _sync_vocal_language_preset_values(payload, set(payload))
    _sync_negative_prompt_preset_values(payload, set(payload))
    _sync_sampler_mode_preset_values(payload, set(payload))
    saved_name = _write_user_preset(requested_name, payload)
    return (
        gr.update(choices=list_preset_names(), value=saved_name),
        f"Saved preset: {saved_name}",
        _build_dashboard_markdown(saved_name, payload.get("subprocess_mode_checkbox")),
    )


def load_preset_action(
    preset_name: str | None,
    component_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Any, ...]:
    """Load a preset and return updates for all tracked components."""
    requested = str(preset_name or "").strip()
    selected, fell_back, payload = _load_resolved_preset_payload(requested)
    if not selected:
        if fell_back:
            _clear_last_used_preset_name()
        status = (
            f"Preset `{requested}` was missing. Using {GPU_OPTIMIZATION_PRESET_NAME}."
            if requested and fell_back
            else f"Using {GPU_OPTIMIZATION_PRESET_NAME}."
        )
        return (
            *[gr.skip() for _ in PRESET_COMPONENT_KEYS],
            "No LoRA will be used.",
            gr.update(value=False),
            gr.update(choices=list_preset_names(), value=None),
            status,
            _build_dashboard_markdown(None, None),
        )

    set_last_used_preset_name(selected)
    lora_status, use_lora_update = _lora_status_from_payload(payload)
    return (
        *_payload_to_component_updates(payload, component_specs),
        lora_status,
        use_lora_update,
        gr.update(choices=list_preset_names(), value=selected),
        f"Loaded preset: {selected}",
        _build_dashboard_markdown(selected, payload.get("subprocess_mode_checkbox")),
    )


def load_lora_optimizer_hyperparameter_updates_for_preset(
    preset_name: str | None,
    component_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Any, ...]:
    """Return saved LoRA optimizer hyperparameters after a preset load event."""

    requested = str(preset_name or "").strip()
    selected, _fell_back, payload = _load_resolved_preset_payload(requested)
    if not selected:
        return tuple(gr.skip() for _key in _LORA_OPTIMIZER_PRESET_KEY_MAP)

    optimizer_type = _selected_lora_optimizer(payload.get("lora_optimizer_type"))
    defaults = optimizer_hyperparameter_defaults(optimizer_type)
    updates: list[Any] = []
    for key, optimizer_key in _LORA_OPTIMIZER_PRESET_KEY_MAP.items():
        value = payload.get(key, defaults[optimizer_key])
        updates.append(
            gr.update(
                value=coerce_preset_value(key, value, component_specs),
                visible=_lora_optimizer_preset_key_visible(key, optimizer_type),
            )
        )
    return tuple(updates)


def delete_preset_action(
    preset_name: str | None,
    default_values: list[Any] | tuple[Any, ...] | None = None,
    preset_name_input: str | None = None,
    component_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Any, ...]:
    """Delete a preset, then load the next preset or restore Gradio defaults."""

    selected = str(preset_name or preset_name_input or "").strip()
    if not selected:
        return (
            *[gr.skip() for _ in PRESET_COMPONENT_KEYS],
            "No LoRA will be used.",
            gr.update(value=False),
            gr.update(choices=list_preset_names(), value=None),
            "Select a user preset to delete.",
            _build_dashboard_markdown(None, None),
        )

    choices_before = list_preset_names()
    target = _user_preset_path(selected)
    deleted = target.exists()
    if target.exists():
        target.unlink()

    if get_last_used_preset_name() == selected:
        _clear_last_used_preset_name()

    choices_after = list_preset_names()
    next_preset = _next_preset_after_delete(selected, choices_before, choices_after)
    if next_preset:
        payload = load_named_preset(next_preset)
        set_last_used_preset_name(next_preset)
        lora_status, use_lora_update = _lora_status_from_payload(payload)
        delete_status = (
            f"Deleted preset: {selected}. Loaded next preset: {next_preset}"
            if deleted
            else f"Preset `{selected}` was missing. Loaded preset: {next_preset}"
        )
        return (
            *_payload_to_component_updates(payload, component_specs),
            lora_status,
            use_lora_update,
            gr.update(choices=choices_after, value=next_preset),
            delete_status,
            _build_dashboard_markdown(
                next_preset,
                payload.get("subprocess_mode_checkbox"),
            ),
        )

    _clear_last_used_preset_name()
    status = (
        f"Deleted preset: {selected}. Using {GPU_OPTIMIZATION_PRESET_NAME}."
        if deleted
        else f"Preset `{selected}` was missing. Using {GPU_OPTIMIZATION_PRESET_NAME}."
    )
    return (
        *_default_preset_component_updates(default_values, component_specs),
        "No LoRA will be used.",
        gr.update(value=False),
        gr.update(choices=choices_after, value=None),
        status,
        _build_dashboard_markdown(None, None),
    )


def _next_preset_after_delete(
    deleted_name: str,
    choices_before: list[str],
    choices_after: list[str],
) -> str:
    """Return the next dropdown preset after deleting a selected preset."""

    if not choices_after:
        return ""
    try:
        deleted_index = choices_before.index(deleted_name)
    except ValueError:
        return choices_after[0]
    return choices_after[min(deleted_index, len(choices_after) - 1)]


def _default_preset_component_updates(
    default_values: list[Any] | tuple[Any, ...] | None,
    component_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Any]:
    """Return component updates that restore the no-user-preset Gradio defaults."""

    if default_values and len(default_values) == len(PRESET_COMPONENT_KEYS):
        payload = _values_to_payload(tuple(default_values))
        return _payload_to_component_updates(payload, component_specs)

    fallback_payload = _apply_runtime_defaults({})
    fallback_updates = _payload_to_component_updates(fallback_payload, component_specs)
    return [
        fallback_updates[index] if key in fallback_payload else gr.skip()
        for index, key in enumerate(PRESET_COMPONENT_KEYS)
    ]
