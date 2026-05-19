"""Dataset scan/load and settings controls for the training dataset tab."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.ui.gradio.i18n import t
from acestep.ui.gradio.interfaces.training_dataset_model_selector import (
    build_dataset_model_selector,
)
from acestep.ui.gradio.interfaces.training_dataset_vram_presets import (
    build_dataset_vram_preset_dropdown,
)
from acestep.ui.gradio.language_choices import language_dropdown_choices


def build_dataset_scan_and_settings_controls(
    dit_handler: Any = None,
    init_params: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Render scan/load controls and dataset settings for the dataset-builder tab.

    Args:
        dit_handler: DiT handler used to discover selectable dataset models.
        init_params: Optional startup params used to seed the selected model.
    """

    gr.Markdown(
        f"### {t('training.quick_start_title')}\n"
        "Choose one: **Load existing dataset** OR **Scan new directory**"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h4>📂 Load Existing Dataset</h4>")
            load_json_path = gr.Textbox(
                label=t("training.load_dataset_label"),
                placeholder="./datasets/my_lora_dataset.json",
                info=t("training.load_dataset_info"),
                elem_classes=["has-info-container"],
            )
            load_json_browse_btn = gr.Button(
                "Browse Dataset JSON",
                variant="secondary",
            )
            load_json_btn = gr.Button(t("training.load_btn"), variant="primary")
            load_json_status = gr.Textbox(
                label=t("training.load_status"),
                interactive=False,
            )

        with gr.Column(scale=1):
            gr.HTML("<h4>🔍 Scan New Directory</h4>")
            audio_directory = gr.Textbox(
                label=t("training.scan_label"),
                placeholder="/path/to/your/audio/folder",
                info=t("training.scan_info"),
                elem_classes=["has-info-container"],
            )
            scan_directory_browse_btn = gr.Button(
                "Browse Audio Folder",
                variant="secondary",
            )
            scan_btn = gr.Button(t("training.scan_btn"), variant="secondary")
            scan_status = gr.Textbox(
                label=t("training.scan_status"),
                interactive=False,
            )

    gr.HTML("<hr>")

    with gr.Row():
        with gr.Column(scale=2):
            audio_files_table = gr.Dataframe(
                headers=["#", "Filename", "Duration", "Lyrics", "Labeled", "BPM", "Key", "Caption"],
                datatype=["number", "str", "str", "str", "str", "str", "str", "str"],
                label=t("training.found_audio_files"),
                interactive=False,
                wrap=True,
            )

        with gr.Column(scale=1):
            gr.HTML(f"<h3>⚙️ {t('training.dataset_settings_header')}</h3>")

            with gr.Row():
                with gr.Column(scale=2):
                    model_controls = build_dataset_model_selector(dit_handler, init_params)
                with gr.Column(scale=1):
                    dataset_vram_preset = build_dataset_vram_preset_dropdown()

            dataset_name = gr.Textbox(
                label=t("training.dataset_name"),
                value="my_lora_dataset",
                placeholder=t("training.dataset_name_placeholder"),
            )

            all_instrumental = gr.Checkbox(
                label=t("training.all_instrumental"),
                value=False,
                info=t("training.all_instrumental_info"),
                elem_classes=["has-info-container"],
            )

            format_lyrics = gr.Checkbox(
                label="Format Lyrics (LM)",
                value=False,
                info=(
                    "Optional: use LM to format/structure user-provided lyrics "
                    "from .txt files during auto-labeling. Leave off for safest "
                    "dataset quality unless the lyric text is already reliable."
                ),
                elem_classes=["has-info-container"],
                interactive=True,
            )

            transcribe_lyrics = gr.Checkbox(
                label="Transcribe Lyrics (LM)",
                value=True,
                info=(
                    "Enabled by default: generate approximate lyrics from audio "
                    "during auto-labeling. Review transcribed lyrics before training."
                ),
                elem_classes=["has-info-container"],
                interactive=True,
            )

            lm_lyrics_language = gr.Dropdown(
                choices=language_dropdown_choices(),
                value="en",
                label="LM Lyrics Language",
                info=(
                    "Optional language hint for LM lyric formatting/transcription. "
                    "Use Instrumental / auto when the language is unknown."
                ),
                elem_classes=["has-info-container"],
            )

            custom_tag = gr.Textbox(
                label=t("training.custom_tag"),
                placeholder="e.g., 8bit_retro, my_style",
                info=t("training.custom_tag_info"),
                elem_classes=["has-info-container"],
            )

            tag_position = gr.Radio(
                choices=[
                    (t("training.tag_prepend"), "prepend"),
                    (t("training.tag_append"), "append"),
                    (t("training.tag_replace"), "replace"),
                ],
                value="prepend",
                label=t("training.tag_position"),
                info=t("training.tag_position_info"),
                elem_classes=["has-info-container"],
            )

            genre_ratio = gr.Slider(
                minimum=0,
                maximum=100,
                step=10,
                value=0,
                label=t("training.genre_ratio"),
                info=t("training.genre_ratio_info"),
                elem_classes=["has-info-container"],
            )

    return {
        "load_json_path": load_json_path,
        "load_json_browse_btn": load_json_browse_btn,
        "load_json_btn": load_json_btn,
        "load_json_status": load_json_status,
        "audio_directory": audio_directory,
        "scan_directory_browse_btn": scan_directory_browse_btn,
        "scan_btn": scan_btn,
        "scan_status": scan_status,
        "audio_files_table": audio_files_table,
        "dataset_model_config": model_controls["dataset_model_config"],
        "dataset_vram_preset": dataset_vram_preset,
        "dataset_name": dataset_name,
        "all_instrumental": all_instrumental,
        "format_lyrics": format_lyrics,
        "transcribe_lyrics": transcribe_lyrics,
        "lm_lyrics_language": lm_lyrics_language,
        "custom_tag": custom_tag,
        "tag_position": tag_position,
        "genre_ratio": genre_ratio,
    }
