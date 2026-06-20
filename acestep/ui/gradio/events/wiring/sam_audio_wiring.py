"""Event wiring for the SAM Audio Segment tab."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.trim_ui_settings import AUTO_EDITOR_TRIM_UI_KEYS
from acestep.sam_audio_segment.settings import SAM_AUDIO_PRESET_KEYS
from acestep.ui.gradio.events.local_path_dialogs import select_folder_path
from acestep.ui.gradio.premium_features import open_outputs_folder

from .sam_audio_action_helpers import (
    SAM_BATCH_CANCEL_CONFIRM_JS,
    SAM_CANCEL_CONFIRM_JS,
    apply_vram_preset,
    preview_upload,
    request_sam_audio_cancel_from_ui,
)
from .sam_audio_compatibility import apply_sam_audio_compatibility
from .media_upload_preview import preview_video_upload
from .sam_audio_processing import process_batch_folder, process_single_file


def sam_audio_generation_inputs(component_map: dict[str, Any]) -> list[Any]:
    """Return ordered SAM-Audio controls for generation callbacks."""

    return [component_map[key] for key in SAM_AUDIO_PRESET_KEYS]


def register_sam_audio_handlers(
    sam_page: dict[str, Any],
    *,
    dit_handler: Any,
    llm_handler: Any,
    audio_processing_page: dict[str, Any] | None = None,
) -> None:
    """Register SAM Audio Segment handlers."""

    settings_inputs = [
        *sam_audio_generation_inputs(sam_page),
        *_audio_processing_trim_inputs(audio_processing_page),
    ]
    compatibility_inputs = [
        sam_page["sam_prompt_mode"],
        sam_page["sam_low_vram_lite"],
        sam_page["sam_long_audio_mode"],
    ]
    compatibility_outputs = [
        sam_page["sam_prompt_mode"],
        sam_page["sam_ranker_mode"],
        sam_page["sam_predict_spans"],
        sam_page["sam_reranking_candidates"],
        sam_page["sam_use_span_anchor"],
        sam_page["sam_anchor_json"],
        sam_page["sam_anchor_polarity"],
        sam_page["sam_anchor_start"],
        sam_page["sam_anchor_end"],
        sam_page["sam_visual_mask_file"],
    ]

    def single_wrapper(*args: Any, progress: gr.Progress = gr.Progress(track_tqdm=True)):
        """Run one SAM-Audio request with Gradio progress updates."""

        yield from process_single_file(
            dit_handler,
            llm_handler,
            *args,
            progress=progress,
        )

    def batch_wrapper(*args: Any, progress: gr.Progress = gr.Progress(track_tqdm=True)):
        """Stream SAM-Audio batch updates from the generator helper."""

        yield from process_batch_folder(
            dit_handler,
            llm_handler,
            *args,
            progress=progress,
        )

    sam_page["sam_single_file"].change(
        fn=preview_upload,
        inputs=[sam_page["sam_single_file"]],
        outputs=[
            sam_page["sam_upload_audio_preview"],
            sam_page["sam_upload_video_preview"],
            sam_page["sam_single_status"],
        ],
        queue=False,
    )

    sam_page["sam_visual_mask_file"].change(
        fn=preview_video_upload,
        inputs=[sam_page["sam_visual_mask_file"]],
        outputs=[sam_page["sam_visual_mask_video_preview"]],
        queue=False,
    )
    preset_event = sam_page["sam_vram_preset"].change(
        fn=apply_vram_preset,
        inputs=[sam_page["sam_vram_preset"]],
        outputs=[
            sam_page["sam_quantization"],
            sam_page["sam_attention_backend"],
            sam_page["sam_reranking_candidates"],
            sam_page["sam_ranker_mode"],
            sam_page["sam_predict_spans"],
            sam_page["sam_subprocess"],
            sam_page["sam_ode_steps"],
            sam_page["sam_device_mode"],
            sam_page["sam_low_vram_lite"],
            sam_page["sam_chunked"],
            sam_page["sam_long_audio_mode"],
            sam_page["sam_chunk_seconds"],
            sam_page["sam_chunk_overlap_seconds"],
        ],
    )
    preset_event.then(
        fn=apply_sam_audio_compatibility,
        inputs=compatibility_inputs,
        outputs=compatibility_outputs,
    )
    for key in ("sam_prompt_mode", "sam_low_vram_lite", "sam_long_audio_mode"):
        sam_page[key].change(
            fn=apply_sam_audio_compatibility,
            inputs=compatibility_inputs,
            outputs=compatibility_outputs,
        )
    single_event = sam_page["sam_process_btn"].click(
        fn=single_wrapper,
        inputs=[
            sam_page["sam_single_file"],
            sam_page["sam_upload_audio_preview"],
            sam_page["sam_visual_mask_file"],
            *settings_inputs,
        ],
        outputs=[
            sam_page["sam_target_audio"],
            sam_page["sam_residual_audio"],
            sam_page["sam_target_video"],
            sam_page["sam_single_files"],
            sam_page["sam_single_status"],
        ],
        api_name="sam_audio_process",
    )
    sam_page["sam_cancel_btn"].click(
        fn=None,
        inputs=None,
        outputs=[sam_page["sam_cancel_confirmed_state"]],
        js=SAM_CANCEL_CONFIRM_JS,
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
        cancels=[single_event],
    ).then(
        fn=request_sam_audio_cancel_from_ui,
        inputs=[
            sam_page["sam_cancel_confirmed_state"],
            sam_page["sam_subprocess"],
        ],
        outputs=[sam_page["sam_single_status"]],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )
    sam_page["sam_open_outputs_btn"].click(
        fn=open_outputs_folder,
        outputs=[sam_page["sam_single_status"]],
    )
    sam_page["sam_batch_input_browse_btn"].click(
        fn=select_folder_path,
        inputs=[sam_page["sam_batch_input_folder"]],
        outputs=[sam_page["sam_batch_input_folder"]],
    )
    sam_page["sam_batch_output_browse_btn"].click(
        fn=select_folder_path,
        inputs=[sam_page["sam_batch_output_folder"]],
        outputs=[sam_page["sam_batch_output_folder"]],
    )
    batch_event = sam_page["sam_batch_process_btn"].click(
        fn=batch_wrapper,
        inputs=[
            sam_page["sam_batch_input_folder"],
            sam_page["sam_batch_output_folder"],
            sam_page["sam_batch_recursive"],
            *settings_inputs,
        ],
        outputs=[sam_page["sam_batch_status"], sam_page["sam_batch_files"]],
        api_name="sam_audio_batch",
    )
    sam_page["sam_batch_cancel_btn"].click(
        fn=None,
        inputs=None,
        outputs=[sam_page["sam_batch_cancel_confirmed_state"]],
        js=SAM_BATCH_CANCEL_CONFIRM_JS,
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
        cancels=[batch_event],
    ).then(
        fn=request_sam_audio_cancel_from_ui,
        inputs=[
            sam_page["sam_batch_cancel_confirmed_state"],
            sam_page["sam_subprocess"],
        ],
        outputs=[sam_page["sam_batch_status"]],
        queue=False,
        show_progress="hidden",
        show_progress_on=[],
    )


def _audio_processing_trim_inputs(component_map: dict[str, Any] | None) -> list[Any]:
    """Return shared Audio Processing trim controls for standalone SAM runs."""

    if component_map is None:
        return []
    return [component_map[key] for key in AUTO_EDITOR_TRIM_UI_KEYS]
