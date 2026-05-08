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

from acestep.model_downloader import DEFAULT_LM_MODEL, DEFAULT_PREMIUM_DIT_MODEL, get_models_dir
from acestep.ui.gradio.events.generation.quantization import default_quantization_value
from acestep.ui.gradio.events.results.output_manager import get_results_dir


DEFAULT_PRESET_NAME = "Premium Default"
DEFAULT_PRESET_FOLDER = "premium_default_preset"
USER_PRESET_FOLDER = "premium_user_presets"
LAST_USED_PRESET_FILE = ".last_used_preset.txt"
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

PRESET_COMPONENT_KEYS: tuple[str, ...] = (
    "language_dropdown",
    "tier_dropdown",
    "checkpoint_dropdown",
    "config_path",
    "device",
    "lm_model_path",
    "backend_dropdown",
    "init_llm_checkbox",
    "use_flash_attention_checkbox",
    "offload_to_cpu_checkbox",
    "offload_dit_to_cpu_checkbox",
    "compile_model_checkbox",
    "quantization_checkbox",
    "mlx_dit_checkbox",
    "lora_path",
    "use_lora_checkbox",
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
    "lm_model_path": DEFAULT_LM_MODEL,
    "generation_mode": "Custom",
    "captions": DEFAULT_PRESET_CAPTION,
    "lyrics": DEFAULT_PRESET_LYRICS,
    "batch_size_input": 1,
    "audio_format": "flac_mp3",
    "mp3_bitrate": "320k",
    "mp3_sample_rate": 48000,
    "enable_normalization": True,
    "normalization_db": -1.0,
    "think_checkbox": True,
    "allow_lm_batch": True,
    "quantization_checkbox": "none",
    "auto_score": False,
    "auto_lrc": False,
    "autogen_checkbox": False,
    "subprocess_mode_checkbox": False,
}


def _default_preset_payload() -> dict[str, Any]:
    """Return the canonical immutable default-preset payload."""

    return {
        "_meta": {
            "name": DEFAULT_PRESET_NAME,
            "immutable": True,
            "format": "ace_step_premium_preset",
            "version": 2,
        },
        "values": DEFAULT_PRESET_VALUES,
    }


def _runtime_default_values() -> dict[str, Any]:
    """Return defaults that depend on the current installation path."""

    return {
        **DEFAULT_PRESET_VALUES,
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
    return _project_root() / DEFAULT_PRESET_FOLDER / "default.json"


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


def ensure_default_preset() -> Path:
    """Create or refresh the immutable default preset file."""
    target = _default_preset_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    desired_payload = _default_preset_payload()
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
    names = [DEFAULT_PRESET_NAME]
    for preset_path in sorted(_user_preset_dir().glob("*.json")):
        names.append(preset_path.stem)
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


def load_named_preset(name: str | None) -> dict[str, Any]:
    """Load default or user preset values; missing/invalid files return empty mapping."""
    preset_name = str(name or "").strip()
    if not preset_name or preset_name == DEFAULT_PRESET_NAME:
        payload = _read_preset_file(ensure_default_preset()) or {}
    else:
        payload = _read_preset_file(_user_preset_dir() / f"{preset_name}.json") or {}

    defaults = _runtime_default_values()
    for key, value in defaults.items():
        if payload.get(key) in (None, ""):
            payload[key] = value
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
            if key == "quantization_checkbox":
                value = default_quantization_value(value)
            updates.append(gr.update(value=value))
        else:
            updates.append(gr.skip())
    return updates


def _write_user_preset(name: str, payload: dict[str, Any]) -> str:
    sanitized = _sanitize_name(name)
    target = _user_preset_dir() / f"{sanitized}.json"
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
    selected = get_last_used_preset_name() or DEFAULT_PRESET_NAME
    payload = load_named_preset(selected)
    return (
        *_payload_to_component_updates(payload),
        gr.update(choices=list_preset_names(), value=selected),
        f"Loaded preset: {selected}",
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
    if requested_name == DEFAULT_PRESET_NAME:
        return (
            gr.update(choices=list_preset_names(), value=current_selection),
            f"`{DEFAULT_PRESET_NAME}` is immutable. Save under a different name.",
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
    selected = str(preset_name or DEFAULT_PRESET_NAME).strip() or DEFAULT_PRESET_NAME
    payload = load_named_preset(selected)
    set_last_used_preset_name(selected)
    return (
        *_payload_to_component_updates(payload),
        gr.update(choices=list_preset_names(), value=selected),
        f"Loaded preset: {selected}",
        _build_dashboard_markdown(selected, payload.get("subprocess_mode_checkbox")),
    )


def delete_preset_action(preset_name: str | None) -> tuple[Any, ...]:
    """Delete a user preset and fall back to the default selection."""
    selected = str(preset_name or "").strip()
    if not selected or selected == DEFAULT_PRESET_NAME:
        return (
            gr.update(choices=list_preset_names(), value=DEFAULT_PRESET_NAME),
            f"`{DEFAULT_PRESET_NAME}` cannot be deleted.",
            _build_dashboard_markdown(DEFAULT_PRESET_NAME, None),
        )

    target = _user_preset_dir() / f"{selected}.json"
    if target.exists():
        target.unlink()

    fallback = DEFAULT_PRESET_NAME
    set_last_used_preset_name(fallback)
    return (
        gr.update(choices=list_preset_names(), value=fallback),
        f"Deleted preset: {selected}",
        _build_dashboard_markdown(fallback, None),
    )
