"""Tests for premium app browser-tab branding."""

from __future__ import annotations

import unittest
from pathlib import Path

from acestep.ui.gradio import premium_app
from acestep.ui.gradio.events.generation.cancel_actions import (
    BATCH_CANCEL_CONFIRM_JS,
    CANCEL_CONFIRM_JS,
)
from acestep.ui.gradio.interfaces.source_audio_preview import (
    TRIM_AUDIO_PREVIEW_CLASS,
)
from acestep.ui.gradio.pages import sam_audio_page_io, studio_page


class PremiumAppTests(unittest.TestCase):
    """Verify browser title and favicon branding stay bundled."""

    def test_favicon_data_uri_is_svg(self):
        """Bundled favicon should be exposed as an SVG data URI."""

        favicon_href = premium_app._load_favicon_data_uri()
        self.assertTrue(favicon_href.startswith("data:image/svg+xml,"))
        self.assertIn("ace_step_premium_favicon", str(premium_app._FAVICON_PATH))

    def test_head_contains_exact_browser_title_and_favicon_links(self):
        """Head snippet should force the requested browser-tab branding."""

        head = premium_app._build_head(service_mode=False)
        self.assertEqual(premium_app.APP_BROWSER_TITLE, "ACE-Step 1.5 XL Premium v5.0")
        self.assertIn(premium_app.APP_BROWSER_TITLE, head)
        self.assertIn('rel="icon"', head)
        self.assertIn("data:image/svg+xml,", head)
        self.assertIn("/ace-step/cancel-generation", head)
        self.assertIn("subprocessModeEnabled", head)

    def test_unavailable_generation_modes_are_browser_disabled(self):
        """Unavailable generation modes should be visible but not selectable."""

        head = premium_app._build_head(service_mode=False)
        self.assertIn("acestep-generation-mode", head)
        self.assertIn("aceModeDisabled", head)
        self.assertIn("ace-mode-unavailable", premium_app._PREMIUM_CSS)

    def test_button_personalization_does_not_rewrite_labels(self):
        """Button decoration must not replace server-provided labels."""

        head = premium_app._build_head(service_mode=False)
        self.assertIn("setStyleProperty(button", head)
        self.assertIn("showTemporaryButtonText", head)
        self.assertNotIn("desiredLabel", head)
        self.assertNotIn("aceOriginalLabel", head)

    def test_generate_action_row_has_fixed_button_height(self):
        """Generate/Cancel row should not stretch into a full-page loading panel."""

        self.assertIn("ace-generate-action-row", premium_app._PREMIUM_CSS)
        self.assertIn("max-height: 46px", premium_app._PREMIUM_CSS)
        self.assertIn("button[data-ace-command-button", premium_app._PREMIUM_CSS)
        self.assertIn(".gradio-container .action-btn", premium_app._PREMIUM_CSS)

    def test_source_audio_preview_has_scoped_trim_css(self):
        """Source preview trim controls should be more visible without global audio CSS."""

        css = premium_app._PREMIUM_CSS
        self.assertIn(f".{TRIM_AUDIO_PREVIEW_CLASS}", css)
        self.assertIn('button[aria-label="Trim audio to selection"]', css)
        self.assertIn('content: "Trim"', css)
        self.assertIn("::part(region)", css)
        self.assertIn("::part(region-handle)", css)
        self.assertIn("::part(region-handle-left)", css)
        self.assertIn(".timestamps time", css)
        self.assertIn(".timestamps #trim-duration", css)

    def test_audio_processing_preview_has_scoped_trim_css(self):
        """Audio Processing previews should use the same scoped player styling."""

        css = premium_app._PREMIUM_CSS
        self.assertIn(TRIM_AUDIO_PREVIEW_CLASS, css)
        page_path = Path(premium_app.__file__).with_name("pages") / "audio_processing_page.py"
        self.assertIn("AUDIO_PROCESSING_UPLOAD_PREVIEW_ELEM_ID", page_path.read_text())
        self.assertIn('button[aria-label="Trim audio to selection"]', css)
        self.assertIn("::part(region-handle-right)", css)

    def test_audio_processing_process_cancel_buttons_use_large_lower_row(self):
        """Audio Processing Process File and Cancel buttons should be large and scoped."""

        css = premium_app._PREMIUM_CSS
        page_path = (
            Path(premium_app.__file__).with_name("pages")
            / "audio_processing_single_file_controls.py"
        )
        source = page_path.read_text(encoding="utf-8")
        row_start = source.index('elem_classes=["ace-audio-processing-primary-row"]')

        self.assertLess(source.index('"Open Outputs Folder"'), row_start)
        self.assertGreater(source.index('"Process File"'), row_start)
        self.assertGreater(source.index('"Cancel"'), row_start)
        self.assertIn("action-btn-audio-processing-main", source)
        self.assertIn("ace-audio-processing-primary-row", css)
        self.assertIn("action-btn-audio-processing-main", css)
        self.assertIn("height: 92px", css)

    def test_sam_audio_preview_has_trim_css(self):
        """SAM upload previews should opt into the shared trim presentation."""

        self.assertIn(TRIM_AUDIO_PREVIEW_CLASS, premium_app._PREMIUM_CSS)
        source = Path(sam_audio_page_io.__file__).read_text()
        self.assertIn("SAM_UPLOAD_AUDIO_PREVIEW_ELEM_ID", source)

    def test_sam_audio_action_rows_use_fixed_button_layout(self):
        """SAM Audio buttons should use the fixed-height action row."""

        source = Path(sam_audio_page_io.__file__).read_text(encoding="utf-8")
        self.assertIn('elem_classes=["ace-generate-action-row"]', source)
        self.assertNotIn("with gr.Row(equal_height=True):", source)

    def test_header_shows_plain_title_and_release_link(self):
        """Visible app header should show a plain title and clickable release URL."""

        self.assertIn(premium_app.APP_BROWSER_TITLE, premium_app.APP_HEADER_MARKDOWN)
        self.assertIn(premium_app.APP_RELEASE_URL, premium_app.APP_HEADER_MARKDOWN)
        self.assertNotIn(
            f"[{premium_app.APP_BROWSER_TITLE}]",
            premium_app.APP_HEADER_MARKDOWN,
        )

    def test_batch_folder_tab_is_registered(self):
        """Premium shell should expose and wire the Batch Folder Processing tab."""

        source = Path(premium_app.__file__).read_text(encoding="utf-8")
        self.assertIn("Batch Folder Processing", source)
        self.assertIn("register_batch_folder_handlers", source)

    def test_grid_testing_tab_is_registered(self):
        """Premium shell should expose and wire the Grid Testing tab."""

        source = Path(premium_app.__file__).read_text(encoding="utf-8")
        self.assertIn("Grid Testing", source)
        self.assertIn("register_grid_testing_handlers", source)

    def test_generate_song_tab_and_preset_dropdown_autoload_are_registered(self):
        """The simple tab name and preset dropdown auto-load wiring should be present."""

        source = Path(premium_app.__file__).read_text(encoding="utf-8")
        self.assertIn('gr.Tab("Generate Song"', source)
        self.assertIn('gr.Tab("Custom Preset System"', source)
        self.assertIn('studio_page["preset_dropdown"].change', source)
        self.assertIn("preset_load_outputs", source)
        self.assertIn("preset_default_values", source)

    def test_cancel_buttons_have_confirmation_and_distinct_colors(self):
        """Cancel controls should confirm and use per-screen color classes."""

        source = Path(premium_app.__file__).read_text(encoding="utf-8")
        self.assertIn("Are you sure", CANCEL_CONFIRM_JS)
        self.assertIn("Are you sure", BATCH_CANCEL_CONFIRM_JS)
        self.assertIn("action-btn-cancel-simple", source)
        self.assertIn("action-btn-cancel-advanced", source)
        self.assertIn("action-btn-cancel-batch", source)
        self.assertIn('button.closest(".action-btn-cancel-simple")', source)
        self.assertNotIn('Boolean(button.closest(".action-btn-cancel"))', source)
        self.assertIn("acestep-subprocess-mode-checkbox", source)

    def test_delete_preset_button_does_not_use_generation_cancel_handler(self):
        """Preset deletion should not be intercepted by generation cancel JavaScript."""

        source = Path(studio_page.__file__).read_text(encoding="utf-8")
        self.assertLess(source.index("Storage policy"), source.index("delete_preset_btn"))
        delete_start = source.index("delete_preset_btn = gr.Button")
        delete_block = source[
            delete_start : source.index(
                "with gr.Row(elem_classes=[\"ace-workspace-row\"]):",
                delete_start,
            )
        ]
        self.assertIn("action-btn-delete-preset", delete_block)
        self.assertNotIn("action-btn-cancel", delete_block)


if __name__ == "__main__":
    unittest.main()
