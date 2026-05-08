"""Timestamp formatting helpers for the generated-song library."""

from __future__ import annotations

from datetime import datetime, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_MONTH_NAMES = (
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def format_library_created_time(value: Any, local_timezone: tzinfo | None = None) -> str:
    """Format a manifest timestamp for display in the user's local timezone.

    Args:
        value: ISO-like timestamp from a generation manifest.
        local_timezone: Optional timezone override for deterministic tests.

    Returns:
        A display timestamp like ``15 April 2026, 3:12 PM``. Unparseable
        values are returned unchanged so opaque manifest data remains visible.
    """

    raw_value, local_created_at = _local_manifest_time(value, local_timezone)
    if local_created_at is None:
        return raw_value

    hour = local_created_at.hour % 12 or 12
    meridiem = "AM" if local_created_at.hour < 12 else "PM"
    month = _MONTH_NAMES[local_created_at.month]
    return (
        f"{local_created_at.day} {month} {local_created_at.year}, "
        f"{hour}:{local_created_at.minute:02d} {meridiem}"
    )


def format_library_created_date(value: Any, local_timezone: tzinfo | None = None) -> str:
    """Format a manifest timestamp as a local day label for filtering.

    Args:
        value: ISO-like timestamp from a generation manifest.
        local_timezone: Optional timezone override for deterministic tests.

    Returns:
        A display date like ``15 April 2026``. Unparseable values are returned
        unchanged so they can still be selected in the library.
    """

    raw_value, local_created_at = _local_manifest_time(value, local_timezone)
    if local_created_at is None:
        return raw_value

    month = _MONTH_NAMES[local_created_at.month]
    return f"{local_created_at.day} {month} {local_created_at.year}"


def resolve_library_timezone(value: Any) -> tzinfo | None:
    """Return a timezone from a browser-provided IANA timezone name.

    Args:
        value: Browser timezone name, such as ``America/New_York``.

    Returns:
        A ``tzinfo`` when the name is valid, otherwise ``None`` so callers can
        fall back to the server's local timezone.
    """

    timezone_name = str(value or "").strip()
    if not timezone_name:
        return None

    try:
        return ZoneInfo(timezone_name)
    except (ValueError, ZoneInfoNotFoundError):
        return None


def _local_manifest_time(value: Any, local_timezone: tzinfo | None) -> tuple[str, datetime | None]:
    """Return the raw timestamp and parsed local datetime when possible."""

    raw_value = str(value or "").strip()
    if not raw_value:
        return "", None

    created_at = _parse_manifest_time(raw_value)
    if created_at is None:
        return raw_value, None

    if local_timezone is not None:
        return raw_value, created_at.astimezone(local_timezone)
    return raw_value, created_at.astimezone()


def _parse_manifest_time(value: str) -> datetime | None:
    """Parse a manifest timestamp, treating timezone-free values as UTC."""

    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        created_at = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if created_at.tzinfo is None:
        return created_at.replace(tzinfo=timezone.utc)
    return created_at
