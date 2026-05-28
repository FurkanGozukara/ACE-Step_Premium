"""Unit tests for generation input validation helpers."""

import unittest
from unittest.mock import patch

from acestep.ui.gradio.events.generation.validation import parse_and_validate_timesteps


class ParseAndValidateTimestepsTests(unittest.TestCase):
    """Verify custom timestep parsing rejects numerically unsafe values."""

    @patch("acestep.ui.gradio.events.generation.validation.gr.Warning")
    def test_rejects_non_finite_timesteps(self, warning_mock):
        """NaN and Inf timesteps should not reach the diffusion sampler."""

        parsed, has_warning, message = parse_and_validate_timesteps("nan,0", 1)

        self.assertIsNone(parsed)
        self.assertTrue(has_warning)
        self.assertEqual("Out of range", message)
        warning_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
