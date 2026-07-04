"""Event wiring for the generated-song Library tab."""

from __future__ import annotations

from typing import Any

from acestep.ui.gradio.generated_library import (
    filter_library_view,
    next_library_page,
    previous_library_page,
    refresh_library,
    select_library_table_item,
)


_REFRESH_LIBRARY_JS = """(timezoneState, searchQuery) => {
    try {
        return [
            Intl.DateTimeFormat().resolvedOptions().timeZone || timezoneState || null,
            searchQuery || ""
        ];
    } catch (_e) {
        return [timezoneState || null, searchQuery || ""];
    }
}"""
_LIBRARY_EVENT_OPTIONS = {
    "queue": False,
    "show_progress": "hidden",
    "show_progress_on": [],
}


def register_library_handlers(library_page: dict[str, Any], demo: Any | None = None) -> None:
    """Wire refresh and selection events for the generated-song library."""

    refresh_outputs = [
        library_page["library_state"],
        library_page["library_filtered_state"],
        library_page["library_page_state"],
        library_page["library_selector"],
        library_page["library_table"],
        library_page["library_page_status"],
        library_page["library_prev_page_btn"],
        library_page["library_next_page_btn"],
        library_page["library_audio"],
        library_page["library_details"],
        library_page["library_lyrics"],
        library_page["library_metadata"],
    ]
    filtered_outputs = [
        library_page["library_filtered_state"],
        library_page["library_page_state"],
        library_page["library_table"],
        library_page["library_page_status"],
        library_page["library_prev_page_btn"],
        library_page["library_next_page_btn"],
        library_page["library_audio"],
        library_page["library_details"],
        library_page["library_lyrics"],
        library_page["library_metadata"],
    ]
    filter_inputs = [
        library_page["library_selector"],
        library_page["library_search_query"],
        library_page["library_state"],
    ]
    page_inputs = [*filter_inputs, library_page["library_page_state"]]
    if demo is not None:
        demo.load(
            fn=refresh_library,
            inputs=[
                library_page["library_timezone_state"],
                library_page["library_search_query"],
            ],
            outputs=refresh_outputs,
            js=_REFRESH_LIBRARY_JS,
            **_LIBRARY_EVENT_OPTIONS,
        )
    library_page["refresh_library_btn"].click(
        fn=refresh_library,
        inputs=[
            library_page["library_timezone_state"],
            library_page["library_search_query"],
        ],
        outputs=refresh_outputs,
        js=_REFRESH_LIBRARY_JS,
        **_LIBRARY_EVENT_OPTIONS,
    )
    library_page["library_selector"].change(
        fn=filter_library_view,
        inputs=filter_inputs,
        outputs=filtered_outputs,
        **_LIBRARY_EVENT_OPTIONS,
    )
    library_page["library_search_query"].input(
        fn=filter_library_view,
        inputs=filter_inputs,
        outputs=filtered_outputs,
        **_LIBRARY_EVENT_OPTIONS,
    )
    library_page["library_search_query"].submit(
        fn=filter_library_view,
        inputs=filter_inputs,
        outputs=filtered_outputs,
        **_LIBRARY_EVENT_OPTIONS,
    )
    library_page["library_search_query"].change(
        fn=filter_library_view,
        inputs=filter_inputs,
        outputs=filtered_outputs,
        **_LIBRARY_EVENT_OPTIONS,
    )
    library_page["library_prev_page_btn"].click(
        fn=previous_library_page,
        inputs=page_inputs,
        outputs=filtered_outputs,
        **_LIBRARY_EVENT_OPTIONS,
    )
    library_page["library_next_page_btn"].click(
        fn=next_library_page,
        inputs=page_inputs,
        outputs=filtered_outputs,
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
