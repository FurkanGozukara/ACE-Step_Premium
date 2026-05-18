"""Batch folder processing page for the premium Gradio shell."""

from __future__ import annotations

from typing import Any

import gradio as gr


def create_batch_folder_page() -> dict[str, Any]:
    """Build the Batch Folder Processing page."""

    with gr.Column(elem_classes=["ace-page", "ace-stack"]):
        gr.HTML(
            """
            <section class="ace-page-intro">
              <span class="ace-page-eyebrow">Batch Generation</span>
              <h2>Batch Folder Processing</h2>
              <p>
                Prepare one lyrics file per song in the input folder. A file named
                <code>my_song.txt</code> is treated as lyrics, and an optional
                <code>my_song_style.txt</code> beside it is used as that song's style prompt.
                The scan is non-recursive and ignores files ending in <code>_style.txt</code>.
              </p>
              <p>
                This batch uses the current generation settings from the Create/Advanced controls:
                model, LoRA, audio format, batch size, seed behavior, Think mode, scoring, LRC,
                and diffusion settings. Only lyrics and per-file style are changed per song.
                Generated run folders are named after each lyrics file. Leave the output folder
                empty to save in the default outputs folder.
              </p>
            </section>
            """
        )
        with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
            with gr.Row():
                input_folder = gr.Textbox(
                    label="Input Folder",
                    placeholder=r"G:\songs\batch_inputs",
                    info="Folder containing song.txt files and optional song_style.txt companions.",
                )
                output_folder = gr.Textbox(
                    label="Output Folder",
                    placeholder=r"G:\songs\batch_outputs",
                    info="Optional. Generated audio is saved here, or in the default outputs folder.",
                )
            with gr.Row():
                auto_improve_lyrics = gr.Checkbox(
                    label="Auto improve lyrics",
                    value=False,
                    info="Use the 5Hz LM formatter before generation for each lyrics file.",
                )
                auto_improve_style = gr.Checkbox(
                    label="Auto improve style",
                    value=False,
                    info="Use the 5Hz LM formatter before generation for each style prompt.",
                )
            with gr.Row(equal_height=True):
                process_btn = gr.Button(
                    "Process Folder",
                    variant="primary",
                    size="lg",
                    scale=2,
                    elem_classes=["action-btn", "action-btn-generate"],
                )
                cancel_btn = gr.Button(
                    "Cancel Batch",
                    variant="stop",
                    size="lg",
                    scale=1,
                    elem_classes=[
                        "action-btn",
                        "action-btn-cancel",
                        "action-btn-cancel-batch",
                    ],
                )
            cancel_confirmed_state = gr.State(value=False)
            status = gr.Textbox(
                label="Batch Status",
                value=(
                    "Select folders, adjust the main generation settings, "
                    "then process the folder."
                ),
                lines=18,
                interactive=False,
            )

    return {
        "batch_input_folder": input_folder,
        "batch_output_folder": output_folder,
        "batch_auto_improve_lyrics": auto_improve_lyrics,
        "batch_auto_improve_style": auto_improve_style,
        "batch_process_btn": process_btn,
        "batch_cancel_btn": cancel_btn,
        "batch_cancel_confirmed_state": cancel_confirmed_state,
        "batch_status": status,
    }
