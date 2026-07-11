"""Create page for the premium ACE-Step Gradio shell."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.ui.gradio.interfaces.generation import (
    create_advanced_settings_section,
    create_generation_body_section,
    create_generation_mode_section,
)
from acestep.ui.gradio.interfaces.generation_defaults import compute_init_defaults


def create_generation_workspace_page(
    dit_handler: Any,
    llm_handler: Any,
    init_params: dict[str, Any] | None = None,
    language: str = "en",
    include_results: bool = True,
) -> dict[str, Any]:
    """Build the Create page in a single-column workflow layout."""

    defaults = compute_init_defaults(init_params=init_params, language=language)
    params = init_params or {}
    if defaults["service_pre_initialized"]:
        compile_default = params.get("compile_model", defaults["default_compile"])
    else:
        compile_default = defaults["default_compile"]

    with gr.Column():
        with gr.Row(equal_height=True):
            with gr.Column(scale=7, min_width=520):
                with gr.Group():
                    gr.Markdown("### Generation Mode")
                    mode_section = create_generation_mode_section(
                        dit_handler=dit_handler,
                        llm_handler=llm_handler,
                        init_params=init_params,
                        language=language,
                    )

            with gr.Column(scale=5, min_width=360):
                with gr.Group():
                    gr.Markdown("### Runtime")
                    with gr.Row():
                        subprocess_mode_checkbox = gr.Checkbox(
                            label="Use isolated subprocess generation",
                            value=False,
                            info=(
                                "Safer memory isolation with a separate worker process. "
                                "Slightly slower because models initialize inside that worker."
                            ),
                            elem_id="acestep-subprocess-mode-checkbox",
                        )
                        compile_model_checkbox = gr.Checkbox(
                            label="Compile 5Hz LM (torch.compile)",
                            value=bool(compile_default),
                            info=(
                                "Compiles only the measured-beneficial PyTorch 5Hz LM. "
                                "DiT and VAE stay eager because compilation was slower "
                                "for variable-length generation. First LM use is slower."
                            ),
                            elem_id="acestep-runtime-compile-model-checkbox",
                        )
                    gr.Markdown(
                        "Execution note: The Turbo model is automatically downloaded with "
                        "installation. To download the SFT and Base models, please use the "
                        "Windows_Download_All_Models.bat file. RunPod and Massed Compute "
                        "installers auto downloads all models."
                    )

        with gr.Group():
            gr.Markdown("### Composition")
            body_section = create_generation_body_section(
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                init_params=init_params,
                language=language,
            )

        results_section: dict[str, Any] = {}
        results_wrapper = None
        if include_results:
            from acestep.ui.gradio.interfaces.result import create_results_section

            with gr.Group():
                with gr.Column(visible=True) as results_wrapper:
                    results_section = create_results_section(dit_handler)

        with gr.Group():
            settings_section = create_advanced_settings_section(
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                init_params=init_params,
                language=language,
                show_compile_toggle=False,
            )

    generation_section: dict[str, Any] = {}
    generation_section.update(mode_section)
    generation_section.update(body_section)
    generation_section["compile_model_checkbox"] = compile_model_checkbox
    if results_wrapper is not None:
        generation_section["results_wrapper"] = results_wrapper

    return {
        "generation_section": generation_section,
        "settings_section": settings_section,
        "results_section": results_section,
        "subprocess_mode_checkbox": subprocess_mode_checkbox,
        "compile_model_checkbox": compile_model_checkbox,
    }
