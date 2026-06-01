"""Unit tests for Gradio generation cancel handlers."""

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.generation.cancel_actions import (
    BATCH_CANCEL_REQUESTED_STATUS,
    CANCEL_REQUESTED_STATUS,
    NO_ACTIVE_BATCH_STATUS,
    SUBPROCESS_MODE_DISABLED_STATUS,
    request_full_generation_cancel_from_ui,
    request_generation_cancel_from_ui,
)


class CancelActionsTests(unittest.TestCase):
    """Verify cancel controls request the intended generation scope."""

    def test_cancel_noops_when_subprocess_mode_is_off(self) -> None:
        """Cancel should not request cancellation when subprocess mode is off."""

        with patch(
            "acestep.ui.gradio.events.generation.cancel_actions.request_generation_cancel"
        ) as request_cancel:
            status = request_generation_cancel_from_ui(
                confirmed=True,
                subprocess_mode_enabled=False,
            )

        request_cancel.assert_not_called()
        self.assertEqual(status, SUBPROCESS_MODE_DISABLED_STATUS)

    def test_cancel_requests_subprocess_only_when_enabled(self) -> None:
        """Cancel should request subprocess-only cancellation when enabled."""

        with patch(
            "acestep.ui.gradio.events.generation.cancel_actions.request_generation_cancel",
            return_value=True,
        ) as request_cancel:
            status = request_generation_cancel_from_ui(
                confirmed=True,
                subprocess_mode_enabled=True,
            )

        request_cancel.assert_called_once_with(subprocess_only=True)
        self.assertEqual(status, CANCEL_REQUESTED_STATUS)

    def test_full_cancel_requests_foreground_and_subprocess_work(self) -> None:
        """Batch cancel should request full cooperative cancellation."""

        with patch(
            "acestep.ui.gradio.events.generation.cancel_actions.request_generation_cancel",
            return_value=True,
        ) as request_cancel:
            status = request_full_generation_cancel_from_ui(confirmed=True)

        request_cancel.assert_called_once_with()
        self.assertEqual(status, BATCH_CANCEL_REQUESTED_STATUS)

    def test_full_cancel_reports_no_active_work(self) -> None:
        """Batch cancel should tell the UI when nothing was active."""

        with patch(
            "acestep.ui.gradio.events.generation.cancel_actions.request_generation_cancel",
            return_value=False,
        ):
            status = request_full_generation_cancel_from_ui(confirmed=True)

        self.assertEqual(status, NO_ACTIVE_BATCH_STATUS)


if __name__ == "__main__":
    unittest.main()
