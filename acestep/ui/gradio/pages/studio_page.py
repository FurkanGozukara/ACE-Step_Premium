"""Studio dashboard page for the premium ACE-Step Gradio shell."""

from __future__ import annotations

import gradio as gr


def create_studio_page() -> dict[str, gr.components.Component]:
    """Build the Studio page with presets, workspace stats, and quick actions."""

    with gr.Column(elem_classes=["ace-page", "ace-stack"]):
        gr.HTML(
            """
            <section class="ace-page-intro">
              <span class="ace-page-eyebrow">Premium Studio</span>
              <h2>Save custom presets, keep everything local, and monitor the workspace.</h2>
              <p>
                Startup uses the current GPU Optimization Preset. Saved user presets can override
                those runtime defaults whenever you load one.
              </p>
            </section>
            """
        )
        with gr.Row(elem_classes=["ace-workspace-row"]):
            with gr.Column(scale=4, elem_classes=["ace-panel", "ace-stack"]):
                gr.Markdown("### Presets")
                preset_dropdown = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Preset",
                    info=(
                        "Load one of your saved presets. With no preset selected, "
                        "GPU optimization defaults apply."
                    ),
                )
                preset_name_input = gr.Textbox(
                    label="Preset Name",
                    placeholder="Enter a preset name to save",
                )
                with gr.Row():
                    load_preset_btn = gr.Button(
                        "Load Preset",
                        variant="secondary",
                        size="lg",
                        elem_classes=["action-btn", "action-btn-preview"],
                    )
                    save_preset_btn = gr.Button(
                        "Save Preset",
                        variant="primary",
                        size="lg",
                        elem_classes=["action-btn", "action-btn-upscale"],
                    )
                preset_status = gr.Markdown(
                    "Preset system ready.",
                    elem_classes=["ace-status-markdown"],
                )
            with gr.Column(scale=5, elem_classes=["ace-panel", "ace-stack"]):
                gr.Markdown("### Workspace Overview")
                studio_overview = gr.Markdown(
                    "Loading studio overview...",
                    elem_classes=["ace-overview-markdown"],
                )
                studio_status = gr.Markdown(
                    "",
                    elem_classes=["ace-status-markdown"],
                )
            with gr.Column(scale=3, elem_classes=["ace-panel", "ace-stack"]):
                gr.Markdown("### Quick Actions")
                open_outputs_btn = gr.Button(
                    "Open Outputs Folder",
                    variant="secondary",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-open"],
                )
                open_models_btn = gr.Button(
                    "Open Models Folder",
                    variant="secondary",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-open"],
                )
                refresh_dashboard_btn = gr.Button(
                    "Refresh Studio",
                    variant="primary",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-clear"],
                )
                gr.HTML(
                    """
                    <div class="ace-inline-note">
                      <strong>Storage policy</strong>
                      <p>All ACE-Step assets stay under the local <code>models</code> folder instead of the Hugging Face cache.</p>
                    </div>
                    """
                )
                delete_preset_btn = gr.Button(
                    "Delete Preset",
                    variant="stop",
                    size="lg",
                    elem_classes=["action-btn", "action-btn-delete-preset"],
                )
        with gr.Row(elem_classes=["ace-workspace-row"]):
            with gr.Column(scale=6, elem_classes=["ace-panel"]):
                gr.HTML(
                    """
                    <div class="ace-detail-card">
                      <h3>Recommended flow</h3>
                      <ol class="ace-flow-list">
                        <li>Start from the auto-detected GPU Optimization Preset.</li>
                        <li>Choose XL Turbo, SFT, or Base in the Create page.</li>
                        <li>Use subprocess mode for cleaner isolation on long runs.</li>
                        <li>Save a user preset when you want to override runtime defaults.</li>
                      </ol>
                    </div>
                    """
                )
            with gr.Column(scale=6, elem_classes=["ace-panel"]):
                gr.HTML(
                    """
                    <div class="ace-detail-card">
                      <h3>What this shell keeps visible</h3>
                      <ul class="ace-flow-list">
                        <li>Preset state, local model storage, and output history.</li>
                        <li>Generation workspace on a full-width page instead of under a giant top accordion.</li>
                        <li>Training and dataset tools on dedicated pages so the main composer stays clean.</li>
                      </ul>
                    </div>
                    """
                )

    return {
        "preset_dropdown": preset_dropdown,
        "preset_name_input": preset_name_input,
        "load_preset_btn": load_preset_btn,
        "save_preset_btn": save_preset_btn,
        "delete_preset_btn": delete_preset_btn,
        "preset_status": preset_status,
        "studio_status": studio_status,
        "studio_overview": studio_overview,
        "open_outputs_btn": open_outputs_btn,
        "open_models_btn": open_models_btn,
        "refresh_dashboard_btn": refresh_dashboard_btn,
    }
