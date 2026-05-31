"""Tests for premium app browser-tab branding."""

from __future__ import annotations

import unittest
from pathlib import Path

from acestep.ui.gradio import premium_app
from acestep.ui.gradio.events.generation.cancel_actions import (
    BATCH_CANCEL_CONFIRM_JS,
    CANCEL_CONFIRM_JS,
)
from acestep.ui.gradio.pages import studio_page


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
        self.assertEqual(premium_app.APP_BROWSER_TITLE, "ACE-Step 1.5 XL Premium v3.9.1")
        self.assertIn(premium_app.APP_BROWSER_TITLE, head)
        self.assertIn('rel="icon"', head)
        self.assertIn("data:image/svg+xml,", head)
        self.assertIn("/ace-step/cancel-generation", head)
        self.assertIn("subprocessModeEnabled", head)

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
