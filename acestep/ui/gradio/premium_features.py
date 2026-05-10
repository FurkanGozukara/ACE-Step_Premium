"""Premium Gradio shell helpers: presets, dashboard stats, and folder actions."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import gradio as gr

from acestep.model_downloader import (
    DEFAULT_BASE_DIT_MODEL,
    DEFAULT_LM_MODEL,
    DEFAULT_PREMIUM_DIT_MODEL,
    DEFAULT_TURBO_DIT_MODEL,
    get_models_dir,
)
from acestep.core.generation.handler.lora.folder_scan import (
    lora_dropdown_choices,
    resolve_loadable_lora_adapter_path,
)
from acestep.ui.gradio.events.generation.quantization import default_quantization_value
from acestep.ui.gradio.events.results.output_manager import get_results_dir


DEFAULT_PRESET_NAME = "Premium Default"
DEFAULT_TURBO_PRESET_NAME = "Default_Turbo"
DEFAULT_BASE_PRESET_NAME = "Default_Base"
DEFAULT_PRESET_FOLDER = "premium_system_presets"
USER_PRESET_FOLDER = "premium_user_presets"
LAST_USED_PRESET_FILE = ".last_used_preset.txt"
SYSTEM_PRESET_FILES = {
    DEFAULT_PRESET_NAME: "default.json",
    DEFAULT_TURBO_PRESET_NAME: "default_turbo.json",
    DEFAULT_BASE_PRESET_NAME: "default_base.json",
}
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

SIMPLE_MODEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("ACEStep XL 1.5 SFT", DEFAULT_PREMIUM_DIT_MODEL),
    ("ACEStep XL 1.5 Turbo", DEFAULT_TURBO_DIT_MODEL),
    ("ACEStep XL 1.5 Base", DEFAULT_BASE_DIT_MODEL),
)
SIMPLE_MODEL_VALUES = frozenset(value for _label, value in SIMPLE_MODEL_CHOICES)

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
    "quantization_checkbox",
    "simple_quantization",
    "mlx_dit_checkbox",
    "lora_path",
    "lora_dropdown",
    "lora_scale_slider",
    "generation_mode",
    "simple_query_input",
    "simple_vocal_language",
    "simple_instrumental_checkbox",
    "reference_audio",
    "captions",
    "lyrics",
    "instrumental_checkbox",
    "src_audio",
    "track_name",
    "complete_track_classes",
    "text2music_audio_code_string",
    "audio_cover_strength",
    "cover_noise_strength",
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
    "use_adg",
    "shift",
    "custom_timesteps",
    "cfg_interval_start",
    "cfg_interval_end",
    "seed",
    "random_seed_checkbox",
    "audio_format",
    "mp3_bitrate",
    "mp3_sample_rate",
    "score_scale",
    "enable_normalization",
    "normalization_db",
    "fade_in_duration",
    "fade_out_duration",
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
    "auto_score",
    "autogen_checkbox",
    "auto_lrc",
    "lm_batch_chunk_size",
    "subprocess_mode_checkbox",
)

DEFAULT_PRESET_VALUES: dict[str, Any] = {
    "language_dropdown": "en",
    "config_path": DEFAULT_PREMIUM_DIT_MODEL,
    "simple_model_dropdown": DEFAULT_PREMIUM_DIT_MODEL,
    "lm_model_path": DEFAULT_LM_MODEL,
    "generation_mode": "Custom",
    "captions": DEFAULT_PRESET_CAPTION,
    "lyrics": DEFAULT_PRESET_LYRICS,
    "batch_size_input": 1,
    "inference_steps": 50,
    "guidance_scale": 7.0,
    "infer_method": "ode",
    "sampler_mode": "euler",
    "velocity_norm_threshold": 0.0,
    "velocity_ema_factor": 0.0,
    "use_adg": False,
    "shift": 3.0,
    "custom_timesteps": "",
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "audio_format": "flac_mp3",
    "mp3_bitrate": "320k",
    "mp3_sample_rate": 48000,
    "enable_normalization": True,
    "normalization_db": -1.0,
    "think_checkbox": True,
    "allow_lm_batch": True,
    "quantization_checkbox": "none",
    "simple_quantization": "none",
    "lora_path": "",
    "lora_dropdown": "",
    "lora_scale_slider": 1.0,
    "auto_score": False,
    "auto_lrc": False,
    "autogen_checkbox": False,
    "subprocess_mode_checkbox": False,
}

DEFAULT_TURBO_PRESET_VALUES: dict[str, Any] = {
    **DEFAULT_PRESET_VALUES,
    "config_path": DEFAULT_TURBO_DIT_MODEL,
    "simple_model_dropdown": DEFAULT_TURBO_DIT_MODEL,
    "inference_steps": 8,
    "guidance_scale": 1.0,
    "use_adg": False,
    "shift": 3.0,
    "custom_timesteps": "",
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
}

DEFAULT_BASE_PRESET_VALUES: dict[str, Any] = {
    **DEFAULT_PRESET_VALUES,
    "config_path": DEFAULT_BASE_DIT_MODEL,
    "simple_model_dropdown": DEFAULT_BASE_DIT_MODEL,
    "inference_steps": 50,
    "guidance_scale": 7.0,
    "use_adg": False,
    "shift": 3.0,
    "custom_timesteps": "",
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
}


def normalize_simple_model_dropdown_value(value: Any) -> str:
    """Return a supported Create-tab model selector value."""

    model_path = str(value or "").strip()
    return model_path if model_path in SIMPLE_MODEL_VALUES else DEFAULT_PREMIUM_DIT_MODEL

SYSTEM_PRESET_VALUES = {
    DEFAULT_PRESET_NAME: DEFAULT_PRESET_VALUES,
    DEFAULT_TURBO_PRESET_NAME: DEFAULT_TURBO_PRESET_VALUES,
    DEFAULT_BASE_PRESET_NAME: DEFAULT_BASE_PRESET_VALUES,
}


def _system_preset_payload(name: str) -> dict[str, Any]:
    """Return the canonical immutable payload for a protected system preset."""

    return {
        "_meta": {
            "name": name,
            "immutable": True,
            "format": "ace_step_premium_preset",
            "version": 2,
        },
        "values": SYSTEM_PRESET_VALUES[name],
    }


def _default_preset_payload() -> dict[str, Any]:
    """Return the canonical immutable default-preset payload."""

    return _system_preset_payload(DEFAULT_PRESET_NAME)


def _runtime_default_values(base_values: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return defaults that depend on the current installation path."""

    return {
        **(base_values or DEFAULT_PRESET_VALUES),
        "checkpoint_dropdown": str(get_models_dir(project_root=_project_root())),
    }


def get_preset_component_keys() -> tuple[str, ...]:
    """Return the stable component-key order used for preset serialization."""
    return PRESET_COMPONENT_KEYS


def _project_root() -> Path:
    raw = os.environ.get("ACESTEP_PROJECT_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _default_preset_path() -> Path:
    return _system_preset_path(DEFAULT_PRESET_NAME)


def _system_preset_path(name: str) -> Path:
    return _project_root() / DEFAULT_PRESET_FOLDER / SYSTEM_PRESET_FILES[name]


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


def _system_preset_name_for(name: str | None) -> str | None:
    """Return the canonical protected preset name for a user-provided name."""

    key = _preset_name_key(name)
    if not key:
        return None
    for system_name in SYSTEM_PRESET_FILES:
        if key == _preset_name_key(system_name):
            return system_name
    return None


def _is_default_preset_name(name: str | None) -> bool:
    """Return whether a user-provided preset name targets a protected preset."""

    return _system_preset_name_for(name) is not None


def _user_preset_path(name: str | None) -> Path:
    """Return the canonical user-preset path for a preset name."""

    return _user_preset_dir() / f"{_sanitize_name(name or '')}.json"


def ensure_default_preset() -> Path:
    """Create or refresh immutable system presets and return the default path."""
    for name in SYSTEM_PRESET_FILES:
        _ensure_system_preset(name)
    return _default_preset_path()


def _ensure_system_preset(name: str) -> Path:
    """Create or refresh a single immutable system preset file."""

    target = _system_preset_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    desired_payload = _system_preset_payload(name)
    current_payload: dict[str, Any] | None = None
    if target.exists():
        try:
            with target.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                current_payload = loaded
        except Exception:
            current_payload = None
    if current_payload != desired_payload:
        with target.open("w", encoding="utf-8") as handle:
            json.dump(desired_payload, handle, indent=2, ensure_ascii=False)
    return target


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
    """Return dropdown choices with immutable default first."""
    ensure_default_preset()
    names = list(SYSTEM_PRESET_FILES)
    seen = {_preset_name_key(name) for name in names}
    for preset_path in sorted(_user_preset_dir().glob("*.json")):
        name = preset_path.stem
        key = _preset_name_key(name)
        if _is_default_preset_name(name) or key in seen:
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


def _resolve_available_preset_name(name: str | None) -> tuple[str, bool]:
    """Resolve a requested preset name to an existing selectable preset.

    Returns:
        A tuple of ``(resolved_name, fell_back_to_default)``.
    """

    requested = str(name or "").strip()
    if not requested:
        return DEFAULT_PRESET_NAME, False
    system_name = _system_preset_name_for(requested)
    if system_name:
        return system_name, False

    sanitized = _sanitize_name(requested)
    if _user_preset_path(sanitized).exists():
        return sanitized, False
    return DEFAULT_PRESET_NAME, True


def _apply_runtime_defaults(
    payload: dict[str, Any],
    base_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill missing preset keys from runtime-aware defaults."""

    provided_keys = set(payload)
    merged = dict(payload)
    defaults = _runtime_default_values(base_values)
    for key, value in defaults.items():
        if merged.get(key) in (None, ""):
            merged[key] = value
    config_path = str(merged.get("config_path") or "").strip()
    raw_simple_model = str(payload.get("simple_model_dropdown") or "").strip()
    simple_model = str(merged.get("simple_model_dropdown") or "").strip()
    if raw_simple_model not in SIMPLE_MODEL_VALUES and config_path in SIMPLE_MODEL_VALUES:
        simple_model = config_path
    simple_model = normalize_simple_model_dropdown_value(simple_model)
    merged["simple_model_dropdown"] = simple_model
    if not config_path or config_path in SIMPLE_MODEL_VALUES:
        merged["config_path"] = simple_model
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
    return merged


def _load_resolved_preset_payload(name: str | None) -> tuple[str, bool, dict[str, Any]]:
    """Load an existing preset or the protected default fallback."""

    preset_name, fell_back = _resolve_available_preset_name(name)
    if preset_name in SYSTEM_PRESET_FILES:
        payload = _read_preset_file(_ensure_system_preset(preset_name)) or {}
        return (
            preset_name,
            fell_back,
            _apply_runtime_defaults(payload, SYSTEM_PRESET_VALUES[preset_name]),
        )

    payload = _read_preset_file(_user_preset_path(preset_name))
    if payload is None:
        default_payload = _read_preset_file(_ensure_system_preset(DEFAULT_PRESET_NAME)) or {}
        return (
            DEFAULT_PRESET_NAME,
            True,
            _apply_runtime_defaults(
                default_payload,
                SYSTEM_PRESET_VALUES[DEFAULT_PRESET_NAME],
            ),
        )
    return preset_name, fell_back, _apply_runtime_defaults(payload)


def load_named_preset(name: str | None) -> dict[str, Any]:
    """Load preset values, falling back to the protected default when needed."""
    _preset_name, _fell_back, payload = _load_resolved_preset_payload(name)
    return payload


def _values_to_payload(values: tuple[Any, ...]) -> dict[str, Any]:
    return {
        key: value
        for key, value in zip(PRESET_COMPONENT_KEYS, values)
    }


def _payload_to_component_updates(payload: dict[str, Any]) -> list[Any]:
    updates: list[Any] = []
    for key in PRESET_COMPONENT_KEYS:
        if key in payload:
            value = payload[key]
            if key in {"quantization_checkbox", "simple_quantization"}:
                value = default_quantization_value(value)
            if key == "lora_dropdown":
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
            else:
                updates.append(gr.update(value=value))
        else:
            updates.append(gr.skip())
    return updates


def _lora_status_from_payload(payload: dict[str, Any]) -> tuple[str, Any]:
    """Return LoRA status text plus the hidden legacy checkbox update."""

    candidate = str(payload.get("lora_path") or "").strip() or str(
        payload.get("lora_dropdown") or ""
    ).strip()
    if not candidate:
        return "No LoRA will be used.", gr.update(value=False)
    resolved = resolve_loadable_lora_adapter_path(candidate)
    if not resolved:
        return f"No LoRA will be used. Invalid LoRA path: {candidate}", gr.update(value=False)
    return f"Next run will use LoRA: {resolved}", gr.update(value=True)


def _write_user_preset(name: str, payload: dict[str, Any]) -> str:
    if _is_default_preset_name(name):
        raise ValueError(f"`{_system_preset_name_for(name) or name}` is immutable.")
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


def _build_dashboard_markdown(selected_preset: str | None = None, subprocess_mode: bool | None = None) -> str:
    models_dir = get_models_dir(project_root=_project_root())
    outputs_dir = get_results_dir()
    preset_name = selected_preset or get_last_used_preset_name() or DEFAULT_PRESET_NAME
    execution_mode = "Subprocess" if subprocess_mode else "In-process"

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
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return f"Opened `{target}`"
    except Exception as exc:
        return f"Failed to open `{target}`: {exc}"


def open_models_folder() -> str:
    """Open the local models directory in the system file explorer."""
    return open_folder_in_system(get_models_dir(project_root=_project_root()))


def open_outputs_folder() -> str:
    """Open the local generation outputs directory in the system file explorer."""
    return open_folder_in_system(get_results_dir())


def startup_preset_updates() -> tuple[Any, ...]:
    """Return startup preset updates plus dropdown, status, and dashboard."""
    ensure_default_preset()
    remembered = get_last_used_preset_name()
    selected, fell_back, payload = _load_resolved_preset_payload(remembered)
    if fell_back:
        set_last_used_preset_name(DEFAULT_PRESET_NAME)
    lora_status, use_lora_update = _lora_status_from_payload(payload)
    status = (
        f"Remembered preset `{remembered}` was missing. Loaded preset: {DEFAULT_PRESET_NAME}"
        if fell_back
        else f"Loaded preset: {selected}"
    )
    return (
        *_payload_to_component_updates(payload),
        lora_status,
        use_lora_update,
        gr.update(choices=list_preset_names(), value=selected),
        status,
        _build_dashboard_markdown(selected, payload.get("subprocess_mode_checkbox")),
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
    if _is_default_preset_name(requested_name):
        protected_name = _system_preset_name_for(requested_name) or requested_name
        return (
            gr.update(choices=list_preset_names(), value=current_selection),
            f"`{protected_name}` is immutable. Save under a different name.",
            _build_dashboard_markdown(current_selection, None),
        )

    payload = _values_to_payload(values)
    saved_name = _write_user_preset(requested_name, payload)
    return (
        gr.update(choices=list_preset_names(), value=saved_name),
        f"Saved preset: {saved_name}",
        _build_dashboard_markdown(saved_name, payload.get("subprocess_mode_checkbox")),
    )


def load_preset_action(preset_name: str | None) -> tuple[Any, ...]:
    """Load a preset and return updates for all tracked components."""
    requested = str(preset_name or DEFAULT_PRESET_NAME).strip() or DEFAULT_PRESET_NAME
    selected, fell_back, payload = _load_resolved_preset_payload(requested)
    set_last_used_preset_name(selected)
    lora_status, use_lora_update = _lora_status_from_payload(payload)
    status = (
        f"Preset `{requested}` was missing. Loaded preset: {DEFAULT_PRESET_NAME}"
        if fell_back
        else f"Loaded preset: {selected}"
    )
    return (
        *_payload_to_component_updates(payload),
        lora_status,
        use_lora_update,
        gr.update(choices=list_preset_names(), value=selected),
        status,
        _build_dashboard_markdown(selected, payload.get("subprocess_mode_checkbox")),
    )


def delete_preset_action(preset_name: str | None) -> tuple[Any, ...]:
    """Delete a user preset and fall back to the default selection."""
    selected = str(preset_name or "").strip()
    if not selected or _is_default_preset_name(selected):
        return (
            gr.update(choices=list_preset_names(), value=DEFAULT_PRESET_NAME),
            f"`{DEFAULT_PRESET_NAME}` cannot be deleted.",
            _build_dashboard_markdown(DEFAULT_PRESET_NAME, None),
        )

    target = _user_preset_path(selected)
    if target.exists():
        target.unlink()

    fallback = DEFAULT_PRESET_NAME
    set_last_used_preset_name(fallback)
    return (
        gr.update(choices=list_preset_names(), value=fallback),
        f"Deleted preset: {selected}",
        _build_dashboard_markdown(fallback, None),
    )
