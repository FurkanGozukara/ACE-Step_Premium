"""Shared Auto-Editor trim UI values used outside the Audio Processing tab."""

from __future__ import annotations

from typing import Any

from .auto_editor_trim_settings import (
    AUTO_EDITOR_MINCLIP_DEFAULT,
    AUTO_EDITOR_MINCUT_DEFAULT,
    AutoEditorTrimSettings,
    coerce_auto_editor_margin_seconds,
    coerce_auto_editor_smooth_value,
    coerce_auto_editor_threshold_db,
)

AUTO_EDITOR_TRIM_UI_KEYS: tuple[str, ...] = (
    "ap_trim_threshold_db",
    "ap_trim_margin_seconds",
    "ap_trim_mincut",
    "ap_trim_minclip",
)


def trim_settings_from_ui_values(values: tuple[Any, ...] | list[Any]) -> AutoEditorTrimSettings:
    """Build Auto-Editor trim settings from ordered Audio Processing UI values."""

    payload = dict(zip(AUTO_EDITOR_TRIM_UI_KEYS, values))
    return AutoEditorTrimSettings(
        threshold_db=coerce_auto_editor_threshold_db(payload.get("ap_trim_threshold_db")),
        margin_seconds=coerce_auto_editor_margin_seconds(
            payload.get("ap_trim_margin_seconds")
        ),
        mincut=coerce_auto_editor_smooth_value(
            payload.get("ap_trim_mincut"),
            AUTO_EDITOR_MINCUT_DEFAULT,
        ),
        minclip=coerce_auto_editor_smooth_value(
            payload.get("ap_trim_minclip"),
            AUTO_EDITOR_MINCLIP_DEFAULT,
        ),
    )
