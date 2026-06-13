"""Example sampling helpers used by generation metadata handlers."""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Optional

import gradio as gr

from acestep.inference import understand_music
from acestep.ui.gradio.i18n import t

from .validation import clamp_duration_to_gpu_limit


def _get_project_root() -> str:
    """Return the project root directory above the generation event package."""

    current_file = os.path.abspath(__file__)
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            )
        )
    )


def load_random_example(task_type: str, llm_handler=None):
    """Load a random example from the task-specific examples directory.

    Args:
        task_type: The task type, such as "text2music".
        llm_handler: Optional LLM handler used for GPU duration limit checks.

    Returns:
        Tuple of caption, lyrics, think flag, bpm, duration, key, language, and meter.
    """

    try:
        project_root = _get_project_root()
        examples_dir = os.path.join(project_root, "examples", task_type)

        if not os.path.exists(examples_dir):
            gr.Warning(f"Examples directory not found: examples/{task_type}/")
            return "", "", True, None, None, "", "", ""

        json_files = glob.glob(os.path.join(examples_dir, "*.json"))
        if not json_files:
            gr.Warning(f"No JSON files found in examples/{task_type}/")
            return "", "", True, None, None, "", "", ""

        return _load_selected_example(random.choice(json_files), llm_handler)
    except Exception as exc:
        gr.Warning(t("messages.example_error", error=str(exc)))
        return "", "", True, None, None, "", "", ""


def _load_selected_example(selected_file: str, llm_handler=None):
    """Return normalized generation example values from one JSON file."""

    try:
        with open(selected_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        caption_value = data.get("caption", data.get("prompt", ""))
        if not isinstance(caption_value, str):
            caption_value = str(caption_value) if caption_value else ""
        lyrics_value = data.get("lyrics", "")
        if not isinstance(lyrics_value, str):
            lyrics_value = str(lyrics_value) if lyrics_value else ""
        think_value = data.get("think", True)
        if not isinstance(think_value, bool):
            think_value = True
        return (
            caption_value,
            lyrics_value,
            think_value,
            _optional_int(data.get("bpm")),
            _optional_duration(data.get("duration"), llm_handler),
            _empty_if_na(data.get("keyscale", "")),
            _empty_if_na(data.get("language", "")),
            _empty_if_na(data.get("timesignature", "")),
        )
    except json.JSONDecodeError as exc:
        gr.Warning(
            t(
                "messages.example_failed",
                filename=os.path.basename(selected_file),
                error=str(exc),
            )
        )
        return "", "", True, None, None, "", "", ""
    except Exception as exc:
        gr.Warning(t("messages.example_error", error=str(exc)))
        return "", "", True, None, None, "", "", ""


def _optional_int(value) -> Optional[int]:
    """Return an optional integer parsed from example metadata."""

    if value in [None, "N/A", ""]:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _optional_duration(value, llm_handler=None):
    """Return an optional GPU-clamped duration parsed from example metadata."""

    if value in [None, "N/A", ""]:
        return None
    try:
        return clamp_duration_to_gpu_limit(float(value), llm_handler)
    except (ValueError, TypeError):
        return None


def _empty_if_na(value):
    """Return an empty string for N/A-style example metadata values."""

    return "" if value in [None, "N/A"] else value


def sample_example_smart(
    llm_handler,
    task_type: str,
    constrained_decoding_debug: bool = False,
):
    """Use LM sampling when initialized, otherwise fall back to example files.

    Args:
        llm_handler: LLM handler instance.
        task_type: The task type, such as "text2music".
        constrained_decoding_debug: Whether to enable constrained decoding logs.

    Returns:
        Tuple of caption, lyrics, think flag, bpm, duration, key, language, and meter.
    """

    if llm_handler.llm_initialized:
        try:
            result = understand_music(
                llm_handler=llm_handler,
                audio_codes="NO USER INPUT",
                temperature=0.85,
                use_constrained_decoding=True,
                constrained_decoding_debug=constrained_decoding_debug,
            )
            if result.success:
                gr.Info(t("messages.lm_generated"))
                clamped_duration = clamp_duration_to_gpu_limit(
                    result.duration,
                    llm_handler,
                )
                return (
                    result.caption,
                    result.lyrics,
                    True,
                    result.bpm,
                    clamped_duration,
                    result.keyscale,
                    result.language,
                    result.timesignature,
                )
            gr.Warning(t("messages.lm_fallback"))
            return load_random_example(task_type)
        except Exception:
            gr.Warning(t("messages.lm_fallback"))
            return load_random_example(task_type)
    return load_random_example(task_type)


def load_random_simple_description():
    """Load a random description from the simple_mode examples directory.

    Returns:
        Tuple of description, instrumental flag, and vocal language.
    """

    try:
        project_root = _get_project_root()
        examples_dir = os.path.join(project_root, "examples", "simple_mode")
        if not os.path.exists(examples_dir):
            gr.Warning(t("messages.simple_examples_not_found"))
            return gr.update(), gr.update(), gr.update()

        json_files = glob.glob(os.path.join(examples_dir, "*.json"))
        if not json_files:
            gr.Warning(t("messages.simple_examples_empty"))
            return gr.update(), gr.update(), gr.update()
        return _load_selected_simple_description(random.choice(json_files))
    except Exception as exc:
        gr.Warning(t("messages.example_error", error=str(exc)))
        return gr.update(), gr.update(), gr.update()


def _load_selected_simple_description(selected_file: str):
    """Return normalized simple-mode example values from one JSON file."""

    try:
        with open(selected_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        vocal_language = data.get("vocal_language", "unknown")
        if isinstance(vocal_language, list):
            vocal_language = vocal_language[0] if vocal_language else "unknown"
        gr.Info(
            t(
                "messages.simple_example_loaded",
                filename=os.path.basename(selected_file),
            )
        )
        return data.get("description", ""), data.get("instrumental", False), vocal_language
    except json.JSONDecodeError as exc:
        gr.Warning(
            t(
                "messages.example_failed",
                filename=os.path.basename(selected_file),
                error=str(exc),
            )
        )
        return gr.update(), gr.update(), gr.update()
    except Exception as exc:
        gr.Warning(t("messages.example_error", error=str(exc)))
        return gr.update(), gr.update(), gr.update()
