"""Tests for generated-song library timestamp formatting."""

from __future__ import annotations

import unittest
from datetime import timedelta, timezone
from unittest.mock import patch

from acestep.ui.gradio.generated_library_time import (
    format_library_created_date,
    format_library_created_time,
    resolve_library_timezone,
)


class GeneratedLibraryTimeTests(unittest.TestCase):
    """Verify manifest timestamps are formatted for library display."""

    def test_formats_utc_timestamp_in_requested_local_timezone(self) -> None:
        """UTC manifest times are converted before display formatting."""

        local_timezone = timezone(timedelta(hours=-4))

        result = format_library_created_time(
            "2026-04-15T19:12:00+00:00",
            local_timezone=local_timezone,
        )

        self.assertEqual("15 April 2026, 3:12 PM", result)

    def test_formats_local_date_for_filtering(self) -> None:
        """UTC manifest times are converted before date filter formatting."""

        local_timezone = timezone(timedelta(hours=3))

        result = format_library_created_date(
            "2026-04-15T22:12:00+00:00",
            local_timezone=local_timezone,
        )

        self.assertEqual("16 April 2026", result)

    def test_formats_z_suffix_timestamp(self) -> None:
        """UTC ``Z`` suffixes are accepted as manifest timestamps."""

        local_timezone = timezone.utc

        result = format_library_created_time(
            "2026-04-15T03:05:00Z",
            local_timezone=local_timezone,
        )

        self.assertEqual("15 April 2026, 3:05 AM", result)

    def test_returns_unparseable_values_unchanged(self) -> None:
        """Unexpected manifest values should stay visible in the library."""

        result = format_library_created_time("not a timestamp")

        self.assertEqual("not a timestamp", result)

    def test_resolves_browser_timezone_names(self) -> None:
        """Browser IANA timezone names should convert to ``tzinfo`` values."""

        with patch(
            "acestep.ui.gradio.generated_library_time.ZoneInfo",
            return_value=timezone.utc,
        ) as zone_info:
            result = resolve_library_timezone("America/New_York")

        zone_info.assert_called_once_with("America/New_York")
        self.assertIs(timezone.utc, result)

    def test_ignores_invalid_browser_timezone_names(self) -> None:
        """Invalid browser timezone names should use the local fallback."""

        result = resolve_library_timezone("not/a-zone")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
