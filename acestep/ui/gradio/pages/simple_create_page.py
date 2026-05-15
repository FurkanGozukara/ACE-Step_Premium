"""Simple first-tab creation page for the premium Gradio shell."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.gpu_config import GPU_TIER_LABELS
from acestep.ui.gradio.events.results.video_export import VIDEO_RESOLUTION_CHOICES
from acestep.ui.gradio.events.generation.quantization import (
    QUANTIZATION_CHOICES,
    default_quantization_value,
)
from acestep.ui.gradio.events.generation.generation_count import (
    generation_count_info,
    normalize_generation_count,
)
from acestep.ui.gradio.language_choices import language_dropdown_choices
from acestep.ui.gradio.premium_features import (
    DEFAULT_PRESET_CAPTION,
    DEFAULT_PRESET_LYRICS,
    SIMPLE_MODEL_CHOICES,
    normalize_simple_model_dropdown_value,
)


def create_simple_create_page(init_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the simplified Create tab."""

    params = init_params or {}
    gpu_config = params.get("gpu_config")
    tier_value = getattr(gpu_config, "tier", None)
    max_duration = getattr(gpu_config, "max_duration_without_lm", 240)
    default_batch = normalize_generation_count(params.get("default_batch_size") or 1)
    default_quant = default_quantization_value(
        params.get(
            "simple_quantization",
            params.get(
                "quantization_checkbox",
                getattr(gpu_config, "quantization_default", False),
            ),
        )
    )
    default_model = normalize_simple_model_dropdown_value(
        params.get("simple_model_dropdown") or params.get("config_path")
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=5, min_width=430):
            simple_caption = gr.Textbox(
                label="Style",
                placeholder="Modern pop, warm piano, clean female vocal, emotional chorus",
                value=params.get("captions", DEFAULT_PRESET_CAPTION),
                lines=4,
                max_lines=8,
                buttons=["copy"],
            )
            with gr.Row(equal_height=True):
                simple_generate_btn = gr.Button(
                    "Generate Song",
                    variant="primary",
                    size="lg",
                    scale=1,
                    elem_classes=[
                        "action-btn",
                        "action-btn-generate",
                        "action-btn-generate-song",
                    ],
                )
                simple_random_btn = gr.Button(
                    "Random Style",
                    variant="secondary",
                    size="lg",
                    scale=1,
                    elem_classes=["action-btn", "action-btn-preview"],
                )
                simple_enhance_caption_btn = gr.Button(
                    "Enhance Style",
                    variant="secondary",
                    size="lg",
                    scale=1,
                    elem_classes=["action-btn", "action-btn-open"],
                )
                simple_enhance_lyrics_btn = gr.Button(
                    "Enhance Lyrics",
                    variant="secondary",
                    size="lg",
                    scale=1,
                    elem_classes=["action-btn", "action-btn-clear"],
                )
            simple_lyrics = gr.Textbox(
                label="Lyrics",
                placeholder="Write lyrics here, or leave empty for instrumental.",
                value=params.get("lyrics", DEFAULT_PRESET_LYRICS),
                lines=12,
                max_lines=24,
                buttons=["copy"],
            )

        with gr.Column(scale=3, min_width=320):
            simple_latest_audio = gr.Audio(
                label="Latest Song",
                type="filepath",
                interactive=False,
                buttons=["download"],
            )
            simple_latest_video = gr.Video(
                label="Latest Song Video",
                interactive=False,
                visible=False,
            )
            simple_generated_files = gr.File(
                label="Generated Files (All Songs)",
                file_count="multiple",
                interactive=False,
                visible=False,
            )
            simple_model_dropdown = gr.Dropdown(
                choices=SIMPLE_MODEL_CHOICES,
                value=default_model,
                label="Model",
                info=(
                    "SFT uses 50-step CFG with Thinking metadata and shift 1.0. Base uses 64-step "
                    "APG/CFG with shift 1.0. Turbo uses 8-step fast defaults with shift 3.0. "
                    "All are XL 4B models; >=12GB VRAM is the practical floor."
                ),
            )
            with gr.Row():
                simple_tier_dropdown = gr.Dropdown(
                    choices=[(label, key) for key, label in GPU_TIER_LABELS.items()],
                    value=tier_value,
                    label="GPU Optimization Preset",
                    info="Quickly applies VRAM-safe defaults for generation.",
                    scale=1,
                )
                simple_quantization = gr.Dropdown(
                    choices=QUANTIZATION_CHOICES,
                    value=default_quant,
                    label="DiT Quantization",
                    info="Use FP8 scaled cache to build and reuse scaled FP8 DiT weights.",
                    scale=1,
                )
            with gr.Row():
                simple_vocal_language = gr.Dropdown(
                    choices=language_dropdown_choices(),
                    value=params.get("vocal_language", "en"),
                    label="Vocal Language",
                    allow_custom_value=True,
                )
                simple_vocal_gender = gr.Radio(
                    choices=[("Male", "male"), ("Female", "female")],
                    value=params.get("vocal_gender", "male"),
                    label="Vocal",
                )
                simple_instrumental = gr.Checkbox(label="Instrumental", value=False)
            with gr.Row():
                simple_duration = gr.Number(
                    label="Song Duration",
                    value=-1,
                    minimum=-1,
                    maximum=float(max_duration),
                    step=0.1,
                    info="-1 = auto. Turbo uses 5Hz LM; Base/SFT estimate duration directly from lyrics.",
                    scale=1,
                )
                simple_batch_size = gr.Number(
                    label="Songs",
                    value=default_batch,
                    minimum=1,
                    step=1,
                    info=generation_count_info(),
                    scale=1,
                )
                simple_random_seed = gr.Checkbox(
                    label="Random Seed",
                    value=True,
                    scale=1,
                )
                simple_seed = gr.Textbox(
                    label="Seed",
                    value="-1",
                    info="Used only when Random Seed is off. Comma-separated seeds are supported.",
                    scale=1,
                )
            with gr.Row():
                simple_cover_image = gr.Image(
                    label="Optional Image for MP4",
                    type="filepath",
                    height=170,
                )
                simple_video_resolution = gr.Dropdown(
                    choices=VIDEO_RESOLUTION_CHOICES,
                    value="1080p",
                    label="Video Resolution",
                    info="Used only when an image is uploaded.",
                )
            simple_open_outputs_btn = gr.Button(
                "Open Outputs Folder",
                variant="secondary",
                size="lg",
                elem_classes=["action-btn", "action-btn-open"],
            )
            simple_status = gr.Textbox(
                label="Status",
                value="Ready",
                interactive=False,
                lines=3,
                max_lines=4,
            )
            simple_bpm_state = gr.State(value=None)
            simple_key_scale_state = gr.State(value="")
            simple_time_signature_state = gr.State(value="")
            simple_is_format_caption_state = gr.State(value=False)

    return {
        "simple_caption": simple_caption,
        "simple_enhance_caption_btn": simple_enhance_caption_btn,
        "simple_enhance_lyrics_btn": simple_enhance_lyrics_btn,
        "simple_lyrics": simple_lyrics,
        "simple_generate_btn": simple_generate_btn,
        "simple_random_btn": simple_random_btn,
        "simple_model_dropdown": simple_model_dropdown,
        "simple_tier_dropdown": simple_tier_dropdown,
        "simple_quantization": simple_quantization,
        "simple_vocal_language": simple_vocal_language,
        "simple_vocal_gender": simple_vocal_gender,
        "simple_instrumental": simple_instrumental,
        "simple_duration": simple_duration,
        "simple_batch_size": simple_batch_size,
        "simple_random_seed": simple_random_seed,
        "simple_seed": simple_seed,
        "simple_cover_image": simple_cover_image,
        "simple_video_resolution": simple_video_resolution,
        "simple_open_outputs_btn": simple_open_outputs_btn,
        "simple_status": simple_status,
        "simple_latest_audio": simple_latest_audio,
        "simple_latest_video": simple_latest_video,
        "simple_generated_files": simple_generated_files,
        "simple_bpm_state": simple_bpm_state,
        "simple_key_scale_state": simple_key_scale_state,
        "simple_time_signature_state": simple_time_signature_state,
        "simple_is_format_caption_state": simple_is_format_caption_state,
    }
