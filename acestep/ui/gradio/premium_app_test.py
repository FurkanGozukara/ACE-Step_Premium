"""Tests for premium app browser-tab branding."""

from __future__ import annotations

import unittest

from acestep.ui.gradio import premium_app


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
        self.assertIn(premium_app.APP_BROWSER_TITLE, head)
        self.assertIn('rel="icon"', head)
        self.assertIn("data:image/svg+xml,", head)

    def test_header_shows_plain_title_and_release_link(self):
        """Visible app header should show a plain title and clickable release URL."""

        self.assertIn(premium_app.APP_BROWSER_TITLE, premium_app.APP_HEADER_MARKDOWN)
        self.assertIn(premium_app.APP_RELEASE_URL, premium_app.APP_HEADER_MARKDOWN)
        self.assertNotIn(
            f"[{premium_app.APP_BROWSER_TITLE}]",
            premium_app.APP_HEADER_MARKDOWN,
        )


if __name__ == "__main__":
    unittest.main()
