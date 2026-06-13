"""Load Metadata page for restoring generation settings from manifests."""

from __future__ import annotations

from typing import Any

import gradio as gr


def create_load_metadata_page() -> dict[str, Any]:
    """Build controls for loading ACE-Step generation metadata JSON files."""

    with gr.Column(elem_classes=["ace-page", "ace-stack"]):
        with gr.Group(elem_classes=["ace-panel", "ace-stack"]):
            metadata_manifest_file = gr.File(
                label="generation_manifest.json",
                type="filepath",
                file_types=[".json"],
                elem_id="acestep-load-metadata-file",
            )
            with gr.Row(equal_height=True):
                metadata_manifest_path = gr.Textbox(
                    label="File Path",
                    placeholder=r"G:\ACE_Step_v1\ACE-Step_Premium\outputs\...\generation_manifest.json",
                    scale=6,
                )
                metadata_browse_btn = gr.Button(
                    "Browse",
                    variant="secondary",
                    scale=1,
                    elem_classes=["action-btn", "action-btn-open"],
                )
                metadata_load_btn = gr.Button(
                    "Load Metadata",
                    variant="primary",
                    scale=2,
                    elem_classes=["action-btn", "action-btn-preview"],
                )
            metadata_load_status = gr.Textbox(
                label="Status",
                value="",
                interactive=False,
                lines=2,
            )
    return {
        "metadata_manifest_file": metadata_manifest_file,
        "metadata_manifest_path": metadata_manifest_path,
        "metadata_browse_btn": metadata_browse_btn,
        "metadata_load_btn": metadata_load_btn,
        "metadata_load_status": metadata_load_status,
    }
