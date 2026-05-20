"""Tests for adapter training timestep schedules."""

from __future__ import annotations

import unittest

from acestep.training.timestep_schedule import build_shifted_timestep_schedule


class TimestepScheduleTests(unittest.TestCase):
    """Verify schedule values are derived from submitted training settings."""

    def test_shift_three_eight_steps_matches_previous_default(self) -> None:
        """The old shift=3, 8-step schedule remains reproducible."""

        schedule = build_shifted_timestep_schedule(8, 3.0)

        self.assertEqual(
            [
                1.0,
                0.9545454545454546,
                0.9,
                0.8333333333333334,
                0.75,
                0.6428571428571429,
                0.5,
                0.3,
            ],
            schedule,
        )

    def test_shift_one_uses_linear_schedule(self) -> None:
        """Shift 1 should leave the schedule linear."""

        self.assertEqual(
            [1.0, 0.8, 0.6, 0.4, 0.19999999999999996],
            build_shifted_timestep_schedule(5, 1.0),
        )

    def test_invalid_values_fall_back_to_valid_schedule(self) -> None:
        """Invalid submitted values should not create NaN schedules."""

        self.assertEqual([1.0], build_shifted_timestep_schedule(0, 0.0))
        self.assertEqual([1.0], build_shifted_timestep_schedule(1, float("inf")))


if __name__ == "__main__":
    unittest.main()
