"""Song preview controls for the Dataset tab."""

from __future__ import annotations

import gradio as gr


def build_dataset_preview_controls() -> dict[str, object]:
    """Render playable song preview and complete metadata controls."""

    with gr.Group(elem_classes=["ace-song-preview", "ace-stack"]):
        gr.HTML(
            """
            <section class="ace-page-intro">
              <span class="ace-page-eyebrow">Song Preview</span>
              <h2>Selected songs load here with playable audio and full metadata.</h2>
            </section>
            """
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                item_src_audio = gr.Audio(
                    label="Song Audio",
                    type="filepath",
                    interactive=False,
                )
                use_src_checkbox = gr.Checkbox(
                    label="Use Source Audio from Dataset",
                    value=True,
                    info="Kept for preset compatibility with dataset workflows.",
                    elem_classes=["has-info-container"],
                )
            with gr.Column(scale=6):
                instruction_display = gr.Textbox(
                    label="Song Summary",
                    interactive=False,
                    placeholder="Select a song to show its summary.",
                    lines=4,
                )
                item_info_json = gr.Code(
                    label="All Song Information",
                    language="json",
                    interactive=False,
                    lines=18,
                )

        with gr.Accordion("Direct Item Lookup", open=False):
            with gr.Row(equal_height=True):
                search_type = gr.Dropdown(
                    choices=["keys", "idx", "random"],
                    value="random",
                    label="Search Type",
                    info="Load by exact key, numeric index, or random item.",
                    elem_classes=["has-info-container"],
                    scale=1,
                )
                search_value = gr.Textbox(
                    label="Search Value",
                    placeholder="Enter a filename/id or index.",
                    info="Keys match id, filename, or audio path exactly.",
                    elem_classes=["has-info-container"],
                    scale=3,
                )
                get_item_btn = gr.Button(
                    "Get Item",
                    variant="secondary",
                    interactive=False,
                    scale=1,
                    elem_classes=["action-btn", "action-btn-open"],
                )

    item_target_audio = gr.Audio(
        label="Target Audio",
        type="filepath",
        interactive=False,
        visible=False,
    )
    item_refer_audio = gr.Audio(
        label="Reference Audio",
        type="filepath",
        interactive=False,
        visible=False,
    )
    repaint_viz_plot = gr.Plot(visible=False)
    auto_fill_btn = gr.Button(
        "Auto-fill Generation Form",
        variant="primary",
        visible=False,
        elem_classes=["action-btn", "action-btn-upscale"],
    )

    return {
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
        "auto_fill_btn": auto_fill_btn,
    }
