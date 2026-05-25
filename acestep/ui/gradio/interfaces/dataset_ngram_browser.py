"""Dataset import and n-gram browser controls for the Dataset tab."""

from __future__ import annotations

import gradio as gr


def build_dataset_import_controls() -> dict[str, object]:
    """Render dataset JSON import controls and status output."""

    with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
        gr.HTML(
            """
            <section class="ace-page-intro">
              <span class="ace-page-eyebrow">Dataset N-Gram Explorer</span>
              <h2>Import a dataset JSON, inspect caption/style word grams, and open matching songs.</h2>
            </section>
            """
        )
        dataset_type = gr.State("train")
        with gr.Row(equal_height=True):
            dataset_json_file = gr.File(
                label="Select Dataset JSON",
                file_count="single",
                file_types=[".json"],
                type="filepath",
                scale=3,
            )
            dataset_import_path = gr.Textbox(
                label="Dataset JSON Path",
                placeholder="./datasets/my_lora_dataset.json",
                info="Paste a local dataset JSON path, or use the file selector.",
                elem_classes=["has-info-container"],
                scale=4,
            )
            import_json_browse_btn = gr.Button(
                "Browse JSON",
                variant="secondary",
                scale=1,
                visible=False,
            )
            import_folder_browse_btn = gr.Button(
                "Browse Folder",
                variant="secondary",
                scale=1,
                visible=False,
            )
            import_dataset_btn = gr.Button(
                "Import Dataset",
                variant="primary",
                scale=1,
                elem_classes=["action-btn", "action-btn-preview"],
            )

        data_status = gr.Textbox(
            label="Dataset Status",
            interactive=False,
            value="No dataset imported",
            lines=3,
        )

    return {
        "dataset_type": dataset_type,
        "dataset_json_file": dataset_json_file,
        "dataset_import_path": dataset_import_path,
        "import_json_browse_btn": import_json_browse_btn,
        "import_folder_browse_btn": import_folder_browse_btn,
        "import_dataset_btn": import_dataset_btn,
        "data_status": data_status,
    }


def build_ngram_browser_controls() -> dict[str, object]:
    """Render six selectable top-n word gram tables and match results."""

    selected_ngram = gr.State("")
    selected_ngram_size = gr.State(0)

    with gr.Group(visible=False, elem_classes=["ace-ngram-browser", "ace-stack"]) as group:
        gr.HTML(
            """
            <section class="ace-page-intro">
              <span class="ace-page-eyebrow">Top Word Grams</span>
              <h2>Select a gram, then select a matching song below.</h2>
            </section>
            """
        )
        with gr.Row(equal_height=True, elem_classes=["ace-ngram-grid"]):
            ngram_1_table = _create_ngram_selector("1grams")
            ngram_2_table = _create_ngram_selector("2grams")
            ngram_3_table = _create_ngram_selector("3grams")
            ngram_4_table = _create_ngram_selector("4grams")
            ngram_5_table = _create_ngram_selector("5grams")
            ngram_6_table = _create_ngram_selector("6grams")

        selected_ngram_display = gr.Textbox(
            label="Selected Gram",
            interactive=False,
            value="Select a gram from the columns above.",
            lines=1,
        )
        ngram_song_table = gr.Radio(
            choices=[],
            label="Songs Containing Selected Gram",
            interactive=False,
        )

    return {
        "ngram_browser_group": group,
        "ngram_1_table": ngram_1_table,
        "ngram_2_table": ngram_2_table,
        "ngram_3_table": ngram_3_table,
        "ngram_4_table": ngram_4_table,
        "ngram_5_table": ngram_5_table,
        "ngram_6_table": ngram_6_table,
        "selected_ngram_display": selected_ngram_display,
        "ngram_song_table": ngram_song_table,
        "selected_ngram": selected_ngram,
        "selected_ngram_size": selected_ngram_size,
    }


def _create_ngram_selector(label: str) -> gr.Radio:
    """Create one top-25 n-gram selector."""

    return gr.Radio(
        choices=[],
        label=label,
        interactive=True,
    )
