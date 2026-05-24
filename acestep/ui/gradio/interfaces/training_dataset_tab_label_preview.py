"""Dataset labeling and sample-preview controls for the training dataset tab."""

from __future__ import annotations

import gradio as gr

from acestep.ui.gradio.i18n import t


def build_dataset_label_and_preview_controls() -> dict[str, object]:
    """Render auto-label and sample-preview editors for dataset-builder workflows."""

    gr.HTML(f"<hr><h3>🤖 {t('training.step2_title')}</h3>")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown(t("training.step2_instruction"))
            skip_metas = gr.Checkbox(
                label=t("training.skip_metas"),
                value=False,
                info=t("training.skip_metas_info"),
                elem_classes=["has-info-container"],
            )
            only_unlabeled = gr.Checkbox(
                label=t("training.only_unlabeled"),
                value=True,
                info=t("training.only_unlabeled_info"),
                elem_classes=["has-info-container"],
            )
        with gr.Column(scale=1):
            auto_label_output_dir = gr.Textbox(
                label=t("training.auto_label_output_dir"),
                placeholder="./datasets/processed_labels",
                info=t("training.auto_label_output_dir_info"),
                elem_classes=["has-info-container"],
            )
            with gr.Row(equal_height=True):
                auto_label_subprocess = gr.Checkbox(
                    label="Run in isolated subprocess",
                    value=True,
                    info="Frees VRAM/RAM when the auto-label worker exits.",
                    elem_classes=["has-info-container"],
                )
                auto_label_batch_size = gr.Slider(
                    minimum=1,
                    maximum=99,
                    step=1,
                    value=1,
                    label="Auto-label batch size",
                    info=(
                        "Higher values group LM metadata requests for faster "
                        "labeling on high-VRAM GPUs."
                    ),
                    elem_classes=["has-info-container"],
                )
            with gr.Row(equal_height=True):
                auto_label_output_dir_browse_btn = gr.Button(
                    t("training.browse_label_folder_btn"),
                    variant="secondary",
                    size="lg",
                )
                auto_label_btn = gr.Button(
                    t("training.auto_label_btn"),
                    variant="primary",
                    size="lg",
                )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=220):
            cancel_auto_label_btn = gr.Button("Cancel Auto-Label", variant="stop", size="lg")
        with gr.Column(scale=3):
            label_progress = gr.Textbox(
                label=t("training.label_progress"),
                interactive=False,
                lines=2,
            )

    gr.HTML(f"<hr><h3>👀 {t('training.step3_title')}</h3>")

    with gr.Row():
        with gr.Column(scale=1):
            sample_selector = gr.Slider(
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                label=t("training.select_sample"),
                info=t("training.select_sample_info"),
                elem_classes=["has-info-container"],
            )

            preview_audio = gr.Audio(
                label=t("training.audio_preview"),
                type="filepath",
                interactive=False,
            )

            preview_filename = gr.Textbox(
                label=t("training.filename"),
                interactive=False,
            )

        with gr.Column(scale=2):
            with gr.Row():
                edit_caption = gr.Textbox(
                    label=t("training.caption"),
                    lines=3,
                    placeholder="Music description...",
                )

            with gr.Row():
                edit_genre = gr.Textbox(
                    label=t("training.genre"),
                    lines=1,
                    placeholder="pop, electronic, dance...",
                )
                prompt_override = gr.Dropdown(
                    choices=["Use Global Ratio", "Caption", "Genre"],
                    value="Use Global Ratio",
                    label=t("training.prompt_override_label"),
                    info=t("training.prompt_override_info"),
                    elem_classes=["has-info-container"],
                )

            with gr.Row():
                edit_lyrics = gr.Textbox(
                    label=t("training.lyrics_editable_label"),
                    lines=6,
                    placeholder="[Verse 1]\nLyrics here...\n\n[Chorus]\n...",
                )
                raw_lyrics_display = gr.Textbox(
                    label=t("training.raw_lyrics_label"),
                    lines=6,
                    placeholder=t("training.no_lyrics_placeholder"),
                    interactive=False,
                    visible=False,
                )
                has_raw_lyrics_state = gr.State(False)

            with gr.Row():
                edit_bpm = gr.Number(
                    label=t("training.bpm"),
                    precision=0,
                )
                edit_keyscale = gr.Textbox(
                    label=t("training.key_label"),
                    placeholder=t("training.key_placeholder"),
                )
                edit_timesig = gr.Dropdown(
                    choices=["", "2", "3", "4", "6", "N/A"],
                    label=t("training.time_sig"),
                )
                edit_duration = gr.Number(
                    label=t("training.duration_s"),
                    precision=1,
                    interactive=False,
                )

            with gr.Row():
                edit_language = gr.Dropdown(
                    choices=[
                        "instrumental",
                        "en",
                        "zh",
                        "ja",
                        "ko",
                        "es",
                        "fr",
                        "de",
                        "pt",
                        "ru",
                        "unknown",
                    ],
                    value="instrumental",
                    label=t("training.language"),
                )
                edit_instrumental = gr.Checkbox(
                    label=t("training.instrumental"),
                    value=True,
                )
                save_edit_btn = gr.Button(t("training.save_changes_btn"), variant="secondary")

            edit_status = gr.Textbox(
                label=t("training.edit_status"),
                interactive=False,
            )

    return {
        "skip_metas": skip_metas,
        "only_unlabeled": only_unlabeled,
        "auto_label_output_dir": auto_label_output_dir,
        "auto_label_output_dir_browse_btn": auto_label_output_dir_browse_btn,
        "auto_label_subprocess": auto_label_subprocess,
        "auto_label_batch_size": auto_label_batch_size,
        "auto_label_btn": auto_label_btn,
        "cancel_auto_label_btn": cancel_auto_label_btn,
        "label_progress": label_progress,
        "sample_selector": sample_selector,
        "preview_audio": preview_audio,
        "preview_filename": preview_filename,
        "edit_caption": edit_caption,
        "edit_genre": edit_genre,
        "prompt_override": prompt_override,
        "edit_lyrics": edit_lyrics,
        "raw_lyrics_display": raw_lyrics_display,
        "has_raw_lyrics_state": has_raw_lyrics_state,
        "edit_bpm": edit_bpm,
        "edit_keyscale": edit_keyscale,
        "edit_timesig": edit_timesig,
        "edit_duration": edit_duration,
        "edit_language": edit_language,
        "edit_instrumental": edit_instrumental,
        "save_edit_btn": save_edit_btn,
        "edit_status": edit_status,
    }
