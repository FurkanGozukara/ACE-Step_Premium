"""Dataset explorer section component definitions for the Gradio UI."""

from __future__ import annotations

import gradio as gr


def create_dataset_section(
    dataset_handler,
    *,
    title: str = "Dataset Explorer",
    open: bool = True,
    visible: bool = False,
) -> dict:
    """Create the dataset explorer section."""

    del dataset_handler

    with gr.Accordion(title, open=open, visible=visible):
        with gr.Row(equal_height=True):
            dataset_type = gr.Dropdown(
                choices=["train", "test"],
                value="train",
                label="Dataset",
                info="Choose dataset to explore",
                elem_classes=["has-info-container"],
                scale=2,
            )
            dataset_import_path = gr.Textbox(
                label="Dataset JSON or Audio Folder",
                placeholder="./datasets/my_lora_dataset.json",
                info="Select a saved dataset JSON or an audio folder to scan.",
                elem_classes=["has-info-container"],
                scale=4,
            )
            import_json_browse_btn = gr.Button(
                "Browse JSON",
                variant="secondary",
                scale=1,
            )
            import_folder_browse_btn = gr.Button(
                "Browse Folder",
                variant="secondary",
                scale=1,
            )
            import_dataset_btn = gr.Button(
                "Import Dataset",
                variant="primary",
                scale=1,
                elem_classes=["action-btn", "action-btn-preview"],
            )

            search_type = gr.Dropdown(
                choices=["keys", "idx", "random"],
                value="random",
                label="Search Type",
                info="How to find items",
                elem_classes=["has-info-container"],
                scale=1,
            )
            search_value = gr.Textbox(
                label="Search Value",
                placeholder="Enter keys or index (leave empty for random)",
                info="Keys: exact match, Index: 0 to dataset size-1",
                elem_classes=["has-info-container"],
                scale=2,
            )

        instruction_display = gr.Textbox(
            label="Instruction",
            interactive=False,
            placeholder="No instruction available",
            lines=1,
        )

        repaint_viz_plot = gr.Plot()

        with gr.Accordion("Item Metadata (JSON)", open=True):
            item_info_json = gr.Code(
                label="Complete Item Information",
                language="json",
                interactive=False,
                lines=15,
            )

        with gr.Row(equal_height=True):
            item_src_audio = gr.Audio(
                label="Source Audio",
                type="filepath",
                interactive=False,
                scale=8,
            )
            get_item_btn = gr.Button(
                "Get Item",
                variant="secondary",
                interactive=False,
                scale=2,
                elem_classes=["action-btn", "action-btn-open"],
            )

        with gr.Row(equal_height=True):
            item_target_audio = gr.Audio(
                label="Target Audio",
                type="filepath",
                interactive=False,
                scale=8,
            )
            item_refer_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
                interactive=False,
                scale=2,
            )

        with gr.Row():
            use_src_checkbox = gr.Checkbox(
                label="Use Source Audio from Dataset",
                value=True,
                info="Check to use the source audio from dataset",
            )

        data_status = gr.Textbox(
            label="Data Status",
            interactive=False,
            value="No dataset imported",
        )
        auto_fill_btn = gr.Button(
            "Auto-fill Generation Form",
            variant="primary",
            elem_classes=["action-btn", "action-btn-upscale"],
        )

    return {
        "dataset_type": dataset_type,
        "dataset_import_path": dataset_import_path,
        "import_json_browse_btn": import_json_browse_btn,
        "import_folder_browse_btn": import_folder_browse_btn,
        "import_dataset_btn": import_dataset_btn,
        "search_type": search_type,
        "search_value": search_value,
        "instruction_display": instruction_display,
        "repaint_viz_plot": repaint_viz_plot,
        "item_info_json": item_info_json,
        "item_src_audio": item_src_audio,
        "get_item_btn": get_item_btn,
        "item_target_audio": item_target_audio,
        "item_refer_audio": item_refer_audio,
        "use_src_checkbox": use_src_checkbox,
        "data_status": data_status,
        "auto_fill_btn": auto_fill_btn,
    }
