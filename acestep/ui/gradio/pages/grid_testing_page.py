"""Grid Testing page for comparing LoRAs with the current quick settings."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.core.generation.handler.lora.folder_scan import lora_dropdown_choices


def create_grid_testing_page() -> dict[str, Any]:
    """Build the Grid Testing tab controls.

    Returns:
        Component map for the grid LoRA selector, output controls, and status.
    """

    with gr.Column(elem_classes=["ace-page", "ace-stack"]):
        gr.HTML(
            """
            <section class="ace-page-intro">
              <span class="ace-page-eyebrow">LoRA Comparison</span>
              <h2>Grid Testing</h2>
              <p>
                Runs the current Generate Song settings once per selected LoRA.
                Select <code>None</code> to include the base model in the same run.
              </p>
            </section>
            """
        )
        with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
            with gr.Row():
                with gr.Column(scale=4):
                    grid_lora_filter = gr.Textbox(
                        label="Filter LoRAs",
                        placeholder="Type part of a LoRA name or path",
                        value="",
                        interactive=True,
                    )
                    grid_lora_dropdown = gr.Dropdown(
                        label="LoRAs",
                        choices=lora_dropdown_choices(),
                        value=[""],
                        multiselect=True,
                        interactive=True,
                        info="Each selected LoRA gets its own generated file prefix.",
                    )
                grid_refresh_loras_btn = gr.Button(
                    "Refresh LoRAs",
                    variant="secondary",
                    size="sm",
                    scale=1,
                    min_width=110,
                )
            with gr.Row():
                grid_output_folder = gr.Textbox(
                    label="Custom Output Folder",
                    placeholder=r"G:\songs\grid_outputs",
                    info="Optional. Empty creates outputs/grid-0001, grid-0002, and so on.",
                    scale=4,
                )
                grid_output_folder_browse_btn = gr.Button(
                    "Browse Output Folder",
                    variant="secondary",
                    scale=1,
                    min_width=130,
                )
            grid_mp3_only = gr.Checkbox(
                label="Save only MP3 files",
                value=True,
                info="When checked, grid output is limited to MP3 files.",
            )
            with gr.Row(equal_height=True):
                grid_generate_btn = gr.Button(
                    "Generate Grid",
                    variant="primary",
                    size="lg",
                    scale=2,
                    elem_classes=["action-btn", "action-btn-generate"],
                )
                grid_cancel_btn = gr.Button(
                    "Cancel Grid",
                    variant="stop",
                    size="lg",
                    scale=1,
                    elem_classes=[
                        "action-btn",
                        "action-btn-cancel",
                        "action-btn-cancel-batch",
                    ],
                )
            grid_generated_files = gr.File(
                label="Grid Files",
                file_count="multiple",
                interactive=False,
                visible=False,
            )
            grid_status = gr.Textbox(
                label="Grid Status",
                value="Select LoRAs, adjust Generate Song settings, then generate a grid.",
                lines=18,
                interactive=False,
            )
            grid_cancel_confirmed_state = gr.State(value=False)

    return {
        "grid_lora_filter": grid_lora_filter,
        "grid_lora_dropdown": grid_lora_dropdown,
        "grid_refresh_loras_btn": grid_refresh_loras_btn,
        "grid_output_folder": grid_output_folder,
        "grid_output_folder_browse_btn": grid_output_folder_browse_btn,
        "grid_mp3_only": grid_mp3_only,
        "grid_generate_btn": grid_generate_btn,
        "grid_cancel_btn": grid_cancel_btn,
        "grid_generated_files": grid_generated_files,
        "grid_status": grid_status,
        "grid_cancel_confirmed_state": grid_cancel_confirmed_state,
    }
