"""Generation metadata file-load wiring helpers."""

from typing import Any, Sequence

from acestep.ui.gradio.events.generation.metadata_fields import (
    LOAD_METADATA_GENERATION_OUTPUT_KEYS,
)
from acestep.ui.gradio.events.local_path_dialogs import select_json_file_path

from .. import generation_handlers as gen_h
from .context import GenerationWiringContext

_METADATA_TAB_KEYS = (
    "metadata_manifest_file",
    "metadata_manifest_path",
    "metadata_browse_btn",
    "metadata_load_btn",
    "metadata_load_status",
)


def _build_load_metadata_outputs(context: GenerationWiringContext) -> list[Any]:
    """Return ordered outputs for the metadata file-load upload handler."""

    generation_section = context.generation_section
    results_section = context.results_section
    outputs = [
        generation_section[key] for key in LOAD_METADATA_GENERATION_OUTPUT_KEYS
    ]
    outputs.append(results_section["is_format_caption_state"])
    return outputs


def _build_load_metadata_tab_outputs(context: GenerationWiringContext) -> list[Any]:
    """Return metadata-tab outputs with a final status textbox."""

    outputs = _build_load_metadata_outputs(context)
    outputs.append(context.generation_section["metadata_load_status"])
    return outputs


def _metadata_tab_components(generation_section: dict[str, Any]) -> dict[str, Any] | None:
    """Return optional Load Metadata tab components when present."""

    components = {key: generation_section.get(key) for key in _METADATA_TAB_KEYS}
    if any(component is None for component in components.values()):
        return None
    return components


def register_generation_metadata_file_handlers(
    context: GenerationWiringContext,
    *,
    auto_checkbox_inputs: Sequence[Any],
    auto_checkbox_outputs: Sequence[Any],
) -> None:
    """Register metadata load-file upload and auto-checkbox sync handlers."""

    generation_section = context.generation_section
    llm_handler = context.llm_handler
    metadata_outputs = _build_load_metadata_outputs(context)

    generation_section["load_file"].upload(
        fn=lambda file_obj: gen_h.load_metadata(file_obj, llm_handler),
        inputs=[generation_section["load_file"]],
        outputs=metadata_outputs,
    ).then(
        fn=gen_h.uncheck_auto_for_populated_fields,
        inputs=list(auto_checkbox_inputs),
        outputs=list(auto_checkbox_outputs),
    )

    metadata_tab = _metadata_tab_components(dict(generation_section))
    if metadata_tab is None:
        return

    metadata_tab_outputs = _build_load_metadata_tab_outputs(context)
    metadata_tab["metadata_browse_btn"].click(
        fn=select_json_file_path,
        inputs=[metadata_tab["metadata_manifest_path"]],
        outputs=[metadata_tab["metadata_manifest_path"]],
    )
    metadata_tab["metadata_manifest_file"].upload(
        fn=lambda file_obj, path: gen_h.load_metadata_with_status(
            file_obj,
            path,
            llm_handler,
        ),
        inputs=[
            metadata_tab["metadata_manifest_file"],
            metadata_tab["metadata_manifest_path"],
        ],
        outputs=metadata_tab_outputs,
    ).then(
        fn=gen_h.uncheck_auto_for_populated_fields,
        inputs=list(auto_checkbox_inputs),
        outputs=list(auto_checkbox_outputs),
    )
    metadata_tab["metadata_load_btn"].click(
        fn=lambda file_obj, path: gen_h.load_metadata_with_status(
            file_obj,
            path,
            llm_handler,
        ),
        inputs=[
            metadata_tab["metadata_manifest_file"],
            metadata_tab["metadata_manifest_path"],
        ],
        outputs=metadata_tab_outputs,
    ).then(
        fn=gen_h.uncheck_auto_for_populated_fields,
        inputs=list(auto_checkbox_inputs),
        outputs=list(auto_checkbox_outputs),
    )
