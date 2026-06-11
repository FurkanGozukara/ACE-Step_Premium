"""Shared settings controls for the SAM Audio Segment page."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.sam_audio_segment.prompt_presets import (
    ATTENTION_BACKEND_CHOICES,
    DEVICE_MODE_CHOICES,
    OUTPUT_FORMAT_CHOICES,
    PROMPT_PRESET_CHOICES,
    QUANTIZATION_CHOICES,
    RANKER_CHOICES,
    VRAM_PRESET_CHOICES,
)
from acestep.sam_audio_segment.settings import (
    SAM_AUDIO_DEFAULT_TRIM_THRESHOLD_DB,
    SAM_AUDIO_LONG_MODE_CHUNKED,
    SAM_AUDIO_LONG_MODE_MULTIDIFFUSION,
    SAM_AUDIO_MAX_CHUNK_SECONDS,
)
from acestep.sam_audio_segment.vram_presets import (
    default_sam_vram_preset_name,
    get_sam_vram_preset,
)


def add_generation_controls(controls: dict[str, Any]) -> None:
    """Add generated-song post-processing controls to the page."""

    with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
        with gr.Row():
            controls["sam_auto_postprocess"] = gr.Checkbox(
                label="Apply automatically to generated songs",
                value=False,
            )
            controls["sam_preserve_original"] = gr.Checkbox(
                label="Save original plus SAM output",
                value=True,
            )
            with gr.Column(scale=1, min_width=190):
                controls["sam_trim_empty_output"] = gr.Checkbox(
                    label="Auto-Editor trim output",
                    value=False,
                    info=(
                        "Removes quiet sections from the SAM target audio. "
                        "Threshold, margin, mincut, and minclip come from "
                        "the Audio Processing tab."
                    ),
                )
                controls["sam_trim_threshold_db"] = gr.Number(
                    value=SAM_AUDIO_DEFAULT_TRIM_THRESHOLD_DB,
                    visible=False,
                )
            controls["sam_output_format"] = gr.Dropdown(
                choices=OUTPUT_FORMAT_CHOICES,
                value="mp3",
                label="Output",
            )
            controls["sam_subprocess"] = gr.Checkbox(
                label="Subprocess mode",
                value=True,
            )
            controls["sam_compile_model"] = gr.Checkbox(
                label="Compile Model",
                value=False,
                info="Use torch.compile for repeated SAM-Audio runs. First use may be slower.",
            )
            controls["sam_unload_generation"] = gr.Checkbox(
                label="Unload generation model first",
                value=True,
            )


def add_prompt_runtime_controls(controls: dict[str, Any]) -> None:
    """Add prompt and runtime controls to the page."""

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            _add_prompt_controls(controls)
        with gr.Column(scale=1):
            _add_runtime_controls(controls)


def _add_prompt_controls(controls: dict[str, Any]) -> None:
    """Add prompt-specific SAM Audio controls."""

    preset = get_sam_vram_preset(default_sam_vram_preset_name())
    spans_interactive = not preset["low_vram_lite"]
    with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
        gr.Markdown("### Prompt")
        controls["sam_prompt_mode"] = gr.Dropdown(
            choices=[("Text", "text"), ("Span", "span")],
            value="text",
            label="Mode",
        )
        with gr.Row(equal_height=True):
            controls["sam_prompt_preset"] = gr.Dropdown(
                choices=PROMPT_PRESET_CHOICES,
                value="vocals",
                label="Quick Prompt",
                scale=3,
            )
            controls["sam_predict_spans"] = gr.Checkbox(
                label="Predict spans",
                value=preset["predict_spans"],
                info=(
                    "Uses SAM-Audio's span predictor to estimate target time ranges "
                    "from the text prompt when you did not provide anchors."
                ),
                interactive=spans_interactive,
                scale=1,
            )
        controls["sam_custom_prompt"] = gr.Textbox(label="Custom Prompt", value="")
        with gr.Row():
            controls["sam_use_span_anchor"] = gr.Checkbox(
                label="Use explicit span anchor",
                value=False,
                info="Enable manual positive or negative time ranges for span prompting.",
            )
        controls["sam_anchor_json"] = gr.Textbox(
            label="Anchor JSON",
            value="",
            placeholder='[["+", 2.0, 3.5], ["-", 0.0, 1.0]]',
            info=(
                "Optional list of anchors: '+' marks where the target exists, '-' marks "
                "where it should not. Example: [[\"+\", 2.0, 3.5], [\"-\", 0.0, 1.0]]."
            ),
        )
        with gr.Row():
            controls["sam_anchor_polarity"] = gr.Dropdown(
                choices=[("+", "+"), ("-", "-")],
                value="+",
                label="Anchor",
                info="'+' is a target-present range; '-' is a target-absent range.",
            )
            controls["sam_anchor_start"] = gr.Number(
                label="Start",
                value=0.0,
                precision=2,
                info="Anchor start time in seconds.",
            )
            controls["sam_anchor_end"] = gr.Number(
                label="End",
                value=1.0,
                precision=2,
                info="Anchor end time in seconds; must be after Start.",
            )


def _add_runtime_controls(controls: dict[str, Any]) -> None:
    """Add runtime, memory, and output controls."""

    preset_name = default_sam_vram_preset_name()
    preset = get_sam_vram_preset(preset_name)
    with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
        gr.Markdown("### Runtime")
        controls["sam_vram_preset"] = gr.Dropdown(
            choices=VRAM_PRESET_CHOICES,
            value=preset_name,
            label="VRAM Preset",
            info=(
                "32GB uses 4 candidates + Judge for highest quality. 24GB uses "
                "2 candidates + Judge for high quality. 12GB/10GB use one "
                "candidate with measured 10s GPU chunks. 8GB uses CPU-safe mode."
            ),
        )
        with gr.Row():
            controls["sam_quantization"] = gr.Dropdown(
                choices=QUANTIZATION_CHOICES,
                value=preset["quantization"],
                label="Quantization",
            )
            controls["sam_attention_backend"] = gr.Dropdown(
                choices=ATTENTION_BACKEND_CHOICES,
                value=preset["attention_backend"],
                label="Attention",
                info=(
                    "Auto SDPA matches the official SAM-Audio implementation. Memory "
                    "Efficient SDPA is available as an optional backend."
                ),
            )
        with gr.Row():
            controls["sam_reranking_candidates"] = gr.Slider(
                minimum=1,
                maximum=8,
                step=1,
                value=preset["reranking_candidates"],
                label="Candidates",
                info=(
                    "4 candidates is highest quality, 2 is high quality, and 1 is "
                    "normal quality. Higher values increase runtime and VRAM."
                ),
            )
            controls["sam_ranker_mode"] = gr.Dropdown(
                choices=RANKER_CHOICES,
                value=preset["ranker_mode"],
                label="Ranker",
                info=(
                    "Scores candidates with the selected official ranker. Disabled uses "
                    "the first generated candidate."
                ),
                interactive=not preset["low_vram_lite"],
            )
        with gr.Row():
            controls["sam_ode_steps"] = gr.Slider(
                minimum=4,
                maximum=64,
                step=1,
                value=preset["ode_steps"],
                label="ODE Steps",
                info=(
                    "Controls the flow-matching solver step size. More steps can improve "
                    "quality but are slower."
                ),
            )
            controls["sam_device_mode"] = gr.Dropdown(
                choices=DEVICE_MODE_CHOICES,
                value=preset["device_mode"],
                label="Device",
            )
        with gr.Row():
            controls["sam_low_vram_lite"] = gr.Checkbox(
                label="Text/span low VRAM lite mode",
                value=preset["low_vram_lite"],
                info=(
                    "Drops unused visual, ranker, and span modules for text/span runs to "
                    "reduce memory."
                ),
            )
            controls["sam_chunked"] = gr.Checkbox(
                label="Chunk long audio",
                value=preset["chunked"],
                info=(
                    "Splits long audio into overlapping model windows. Lower values "
                    "may separate vocals and other text-prompt targets better."
                ),
            )
            controls["sam_long_audio_mode"] = gr.Dropdown(
                choices=[
                    ("Chunk-wise crossfade", SAM_AUDIO_LONG_MODE_CHUNKED),
                    (
                        "Multi-diffusion text-only",
                        SAM_AUDIO_LONG_MODE_MULTIDIFFUSION,
                    ),
                ],
                value=preset["long_audio_mode"],
                label="Long Audio Mode",
                info=(
                    "Multi-diffusion fuses latent windows at each ODE step. It is "
                    "text-only, high-VRAM, and best matches the paper at 20s window "
                    "/ 5s overlap."
                ),
            )
        with gr.Row():
            controls["sam_chunk_seconds"] = gr.Number(
                label="Chunk Seconds",
                value=preset["chunk_seconds"],
                precision=2,
                minimum=1.0,
                maximum=SAM_AUDIO_MAX_CHUNK_SECONDS,
                info=(
                    f"Allowed range: 1-{SAM_AUDIO_MAX_CHUNK_SECONDS:.0f}s. Lower "
                    "values may improve quality; larger windows can miss sources."
                ),
            )
            controls["sam_chunk_overlap_seconds"] = gr.Number(
                label="Overlap Seconds",
                value=preset["chunk_overlap_seconds"],
                precision=2,
                minimum=0.0,
                maximum=10.0,
                info="Crossfades neighboring chunks; 1-2s is usually enough.",
            )
        with gr.Row():
            controls["sam_seed"] = gr.Number(label="Seed", value=99, precision=0)
            controls["sam_random_seed"] = gr.Checkbox(
                label="Random seed",
                value=False,
            )
        with gr.Row():
            controls["sam_include_residual"] = gr.Checkbox(
                label="Save remaining audio",
                value=True,
            )
            controls["sam_include_video"] = gr.Checkbox(
                label="Save extracted video for video input",
                value=True,
            )
