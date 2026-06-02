"""Unit tests for flow-edit parameter normalization."""

import unittest

from acestep.core.generation.handler.flow_edit_params import normalize_flow_edit_n_avg


class NormalizeFlowEditNAvgTest(unittest.TestCase):
    """Verify ``n_avg`` values cannot reach the sampler below one."""

    def test_fractional_values_round_up_and_clamp_to_one(self) -> None:
        """Fractional UI/API values should not truncate to zero."""

        self.assertEqual(normalize_flow_edit_n_avg(0.5), 1)
        self.assertEqual(normalize_flow_edit_n_avg("0.5"), 1)
        self.assertEqual(normalize_flow_edit_n_avg(1.2), 2)

    def test_invalid_and_non_positive_values_fall_back_to_one(self) -> None:
        """Bad saved preset values should become safe defaults."""

        self.assertEqual(normalize_flow_edit_n_avg(None), 1)
        self.assertEqual(normalize_flow_edit_n_avg(""), 1)
        self.assertEqual(normalize_flow_edit_n_avg("bad"), 1)
        self.assertEqual(normalize_flow_edit_n_avg(0), 1)
        self.assertEqual(normalize_flow_edit_n_avg(-3), 1)

    def test_integer_values_are_preserved(self) -> None:
        """Valid integer sample counts should pass through unchanged."""

        self.assertEqual(normalize_flow_edit_n_avg(1), 1)
        self.assertEqual(normalize_flow_edit_n_avg("4"), 4)


if __name__ == "__main__":
    unittest.main()
