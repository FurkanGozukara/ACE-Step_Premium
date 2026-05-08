"""Event wiring for the generated-song Library tab."""

from __future__ import annotations

from typing import Any

from acestep.ui.gradio.generated_library import refresh_library, select_library_item


def register_library_handlers(library_page: dict[str, Any], demo: Any | None = None) -> None:
    """Wire refresh and selection events for the generated-song library."""

    outputs = [
        library_page["library_state"],
        library_page["library_selector"],
        library_page["library_table"],
        library_page["library_audio"],
        library_page["library_details"],
        library_page["library_lyrics"],
        library_page["library_metadata"],
    ]
    if demo is not None:
        demo.load(fn=refresh_library, outputs=outputs)
    library_page["refresh_library_btn"].click(fn=refresh_library, outputs=outputs)
    library_page["library_selector"].change(
        fn=select_library_item,
        inputs=[library_page["library_selector"], library_page["library_state"]],
        outputs=[
            library_page["library_audio"],
            library_page["library_details"],
            library_page["library_lyrics"],
            library_page["library_metadata"],
        ],
    )
