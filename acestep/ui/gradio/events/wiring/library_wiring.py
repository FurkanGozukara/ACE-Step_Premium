"""Event wiring for the generated-song Library tab."""

from __future__ import annotations

from typing import Any

from acestep.ui.gradio.generated_library import (
    filter_library_by_date,
    refresh_library,
    select_library_table_item,
)


_BROWSER_TIMEZONE_JS = """() => {
    try {
        return [Intl.DateTimeFormat().resolvedOptions().timeZone || null];
    } catch (_e) {
        return [null];
    }
}"""
_LIBRARY_EVENT_OPTIONS = {
    "queue": False,
    "show_progress": "hidden",
    "show_progress_on": [],
}


def register_library_handlers(library_page: dict[str, Any], demo: Any | None = None) -> None:
    """Wire refresh and selection events for the generated-song library."""

    outputs = [
        library_page["library_state"],
        library_page["library_filtered_state"],
        library_page["library_selector"],
        library_page["library_table"],
        library_page["library_audio"],
        library_page["library_details"],
        library_page["library_lyrics"],
        library_page["library_metadata"],
    ]
    if demo is not None:
        demo.load(
            fn=refresh_library,
            inputs=None,
            outputs=outputs,
            js=_BROWSER_TIMEZONE_JS,
            **_LIBRARY_EVENT_OPTIONS,
        )
    library_page["refresh_library_btn"].click(
        fn=refresh_library,
        inputs=None,
        outputs=outputs,
        js=_BROWSER_TIMEZONE_JS,
        **_LIBRARY_EVENT_OPTIONS,
    )
    library_page["library_selector"].change(
        fn=filter_library_by_date,
        inputs=[library_page["library_selector"], library_page["library_state"]],
        outputs=[
            library_page["library_filtered_state"],
            library_page["library_table"],
            library_page["library_audio"],
            library_page["library_details"],
            library_page["library_lyrics"],
            library_page["library_metadata"],
        ],
        **_LIBRARY_EVENT_OPTIONS,
    )
    library_page["library_table"].select(
        fn=select_library_table_item,
        inputs=[library_page["library_filtered_state"]],
        outputs=[
            library_page["library_audio"],
            library_page["library_details"],
            library_page["library_lyrics"],
            library_page["library_metadata"],
        ],
        **_LIBRARY_EVENT_OPTIONS,
    )
