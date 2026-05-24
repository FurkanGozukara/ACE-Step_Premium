"""Training dataset-builder event wiring helpers."""

from typing import Any, Mapping

import gradio as gr

from acestep.ui.gradio.events.local_path_dialogs import select_folder_path

from .. import training_handlers as train_h
from ..training.dataset_ops import select_sample_from_table
from ..training.subprocess_cancel import (
    AUTO_LABEL_CANCEL_CONFIRM_JS,
    request_auto_label_cancel_from_ui,
)
from ..training.subprocess_dataset import run_auto_label_subprocess
from .context import TrainingWiringContext
from .training_dataset_status import append_preview_refresh_status
from .training_dataset_save_wiring import register_training_dataset_save_handlers
from .training_dataset_vram_payloads import (
    build_auto_label_init_payloads,
    should_run_dataset_action_in_subprocess,
)


_SAMPLE_PREVIEW_OUTPUT_KEYS = (
    "preview_audio",
    "preview_filename",
    "edit_caption",
    "edit_genre",
    "prompt_override",
    "edit_lyrics",
    "edit_bpm",
    "edit_keyscale",
    "edit_timesig",
    "edit_duration",
    "edit_language",
    "edit_instrumental",
    "raw_lyrics_display",
    "has_raw_lyrics_state",
)

_SETTINGS_TRIGGER_KEYS = (
    "custom_tag",
    "tag_position",
    "all_instrumental",
    "genre_ratio",
)


def _build_sample_preview_outputs(training_section: Mapping[str, Any]) -> list[Any]:
    """Return ordered sample-preview outputs shared by preview refresh handlers."""

    return [training_section[key] for key in _SAMPLE_PREVIEW_OUTPUT_KEYS]


def register_training_dataset_builder_handlers(context: TrainingWiringContext) -> None:
    """Register dataset-builder handlers while preserving existing IO ordering."""

    training_section = context.training_section
    dit_handler = context.dit_handler
    llm_handler = context.llm_handler
    sample_preview_outputs = _build_sample_preview_outputs(training_section)

    def run_auto_label(
        state,
        skip,
        fmt_lyrics,
        trans_lyrics,
        lm_lyrics_language,
        only_unlab,
        model,
        vram_preset,
        save_path,
        dataset_name,
        label_output_dir,
        batch_size,
        subprocess_mode,
        custom_tag,
        tag_position,
        all_instrumental,
        genre_ratio,
        progress=gr.Progress(track_tqdm=True),
    ):
        """Run automatic dataset labeling with the selected model and save path."""

        state = train_h.update_settings(
            custom_tag,
            tag_position,
            all_instrumental,
            genre_ratio,
            state,
        )

        if should_run_dataset_action_in_subprocess(vram_preset, subprocess_mode):
            dit_init_params, llm_init_params = build_auto_label_init_payloads(
                dit_handler,
                llm_handler,
                model,
                vram_preset,
            )
            return run_auto_label_subprocess(
                builder_state=state,
                settings={
                    "skip_metas": skip,
                    "format_lyrics": fmt_lyrics,
                    "transcribe_lyrics": trans_lyrics,
                    "lm_lyrics_language": lm_lyrics_language,
                    "only_unlabeled": only_unlab,
                    "model_config": model,
                    "vram_preset": vram_preset,
                    "save_path": save_path,
                    "dataset_name": dataset_name,
                    "label_output_dir": label_output_dir,
                    "batch_size": batch_size,
                },
                dit_init_params=dit_init_params,
                llm_init_params=llm_init_params,
                progress=progress,
            )

        return train_h.auto_label_all(
            dit_handler,
            llm_handler,
            state,
            skip,
            fmt_lyrics,
            trans_lyrics,
            lm_lyrics_language,
            only_unlab,
            progress=progress,
            model_config=model,
            save_path=save_path,
            dataset_name=dataset_name,
            label_output_dir=label_output_dir,
            label_source_root=getattr(state, "_current_dir", None),
            batch_size=batch_size,
        )

    training_section["scan_directory_browse_btn"].click(
        fn=select_folder_path,
        inputs=[training_section["audio_directory"]],
        outputs=[training_section["audio_directory"]],
    )

    training_section["scan_btn"].click(
        fn=lambda directory, name, tag, pos, instr, state: train_h.scan_directory(
            directory, name, tag, pos, instr, state
        ),
        inputs=[
            training_section["audio_directory"],
            training_section["dataset_name"],
            training_section["custom_tag"],
            training_section["tag_position"],
            training_section["all_instrumental"],
            training_section["dataset_builder_state"],
        ],
        outputs=[
            training_section["audio_files_table"],
            training_section["scan_status"],
            training_section["sample_selector"],
            training_section["dataset_builder_state"],
        ],
    )

    auto_label_event = training_section["auto_label_btn"].click(
        fn=run_auto_label,
        inputs=[
            training_section["dataset_builder_state"],
            training_section["skip_metas"],
            training_section["format_lyrics"],
            training_section["transcribe_lyrics"],
            training_section["lm_lyrics_language"],
            training_section["only_unlabeled"],
            training_section["dataset_model_config"],
            training_section["dataset_vram_preset"],
            training_section["save_path"],
            training_section["dataset_name"],
            training_section["auto_label_output_dir"],
            training_section["auto_label_batch_size"],
            training_section["auto_label_subprocess"],
            training_section["custom_tag"],
            training_section["tag_position"],
            training_section["all_instrumental"],
            training_section["genre_ratio"],
        ],
        outputs=[
            training_section["audio_files_table"],
            training_section["label_progress"],
            training_section["dataset_builder_state"],
        ],
    )
    auto_label_preview_event = auto_label_event.then(
        fn=train_h.get_sample_preview,
        inputs=[
            training_section["sample_selector"],
            training_section["dataset_builder_state"],
        ],
        outputs=sample_preview_outputs,
    )
    auto_label_preview_event.then(
        fn=append_preview_refresh_status,
        inputs=[training_section["label_progress"]],
        outputs=[training_section["label_progress"]],
    )
    training_section["cancel_auto_label_btn"].click(
        fn=request_auto_label_cancel_from_ui,
        inputs=None,
        outputs=[training_section["label_progress"]],
        js=AUTO_LABEL_CANCEL_CONFIRM_JS,
        queue=False,
        concurrency_limit=None,
        show_progress="hidden",
    )

    training_section["auto_label_output_dir_browse_btn"].click(
        fn=select_folder_path,
        inputs=[training_section["auto_label_output_dir"]],
        outputs=[training_section["auto_label_output_dir"]],
    )

    training_section["sample_selector"].change(
        fn=train_h.get_sample_preview,
        inputs=[
            training_section["sample_selector"],
            training_section["dataset_builder_state"],
        ],
        outputs=sample_preview_outputs,
    )

    training_section["audio_files_table"].select(
        fn=select_sample_from_table,
        inputs=[training_section["dataset_builder_state"]],
        outputs=[
            training_section["sample_selector"],
            *sample_preview_outputs,
        ],
    )

    training_section["save_edit_btn"].click(
        fn=train_h.save_sample_edit,
        inputs=[
            training_section["sample_selector"],
            training_section["edit_caption"],
            training_section["edit_genre"],
            training_section["prompt_override"],
            training_section["edit_lyrics"],
            training_section["edit_bpm"],
            training_section["edit_keyscale"],
            training_section["edit_timesig"],
            training_section["edit_language"],
            training_section["edit_instrumental"],
            training_section["dataset_builder_state"],
        ],
        outputs=[
            training_section["audio_files_table"],
            training_section["edit_status"],
            training_section["dataset_builder_state"],
        ],
    )

    for trigger_key in _SETTINGS_TRIGGER_KEYS:
        training_section[trigger_key].change(
            fn=train_h.update_settings,
            inputs=[
                training_section["custom_tag"],
                training_section["tag_position"],
                training_section["all_instrumental"],
                training_section["genre_ratio"],
                training_section["dataset_builder_state"],
            ],
            outputs=[training_section["dataset_builder_state"]],
        )

    register_training_dataset_save_handlers(context)
