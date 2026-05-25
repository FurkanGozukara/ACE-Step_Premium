"""Sanitize custom preset values before they are applied to Gradio controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any


ComponentSpecs = dict[str, dict[str, Any]]


def component_specs_from_components(
    keys: Sequence[str],
    components: Sequence[Any],
) -> ComponentSpecs:
    """Return Gradio value constraints keyed by premium preset field name."""

    specs: ComponentSpecs = {}
    for key, component in zip(keys, components):
        specs[key] = _component_spec(component)
    return specs


def coerce_preset_value(
    key: str,
    value: Any,
    specs: Mapping[str, Mapping[str, Any]] | None,
    *,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    choices: Sequence[Any] | None = None,
    allow_custom_value: bool | None = None,
    default: Any = None,
) -> Any:
    """Return a value that Gradio can preprocess for the target component."""

    spec = dict((specs or {}).get(key, {}))
    fallback = default if default is not None else spec.get("value")
    choice_values = _choice_values(choices if choices is not None else spec.get("choices"))
    allow_custom = (
        bool(allow_custom_value)
        if allow_custom_value is not None
        else bool(spec.get("allow_custom_value", False))
    )
    if choice_values:
        return _coerce_choice(value, choice_values, allow_custom, fallback)

    lower = minimum if minimum is not None else spec.get("minimum")
    upper = maximum if maximum is not None else spec.get("maximum")
    if lower is not None or upper is not None:
        return _coerce_number(value, lower, upper, fallback)

    if spec.get("component_type") == "Checkbox":
        return _coerce_bool(value, fallback)

    return value


def _component_spec(component: Any) -> dict[str, Any]:
    """Extract the preset-relevant constraints from one Gradio component."""

    spec: dict[str, Any] = {"component_type": type(component).__name__}
    for attribute in ("value", "minimum", "maximum", "allow_custom_value"):
        if hasattr(component, attribute):
            spec[attribute] = getattr(component, attribute)
    if hasattr(component, "choices"):
        spec["choices"] = _choice_values(getattr(component, "choices"))
    return spec


def _choice_values(choices: Any) -> list[Any]:
    """Return concrete submitted values from Gradio choice definitions."""

    if not choices:
        return []
    return [_choice_value(choice) for choice in choices]


def _choice_value(choice: Any) -> Any:
    """Return the submitted value for one Gradio choice definition."""

    if isinstance(choice, (list, tuple)) and len(choice) >= 2:
        return choice[1]
    return choice


def _coerce_choice(
    value: Any,
    choices: Sequence[Any],
    allow_custom_value: bool,
    default: Any,
) -> Any:
    """Normalize a saved choice value without rejecting valid custom dropdowns."""

    matched = _match_choice(value, choices)
    if matched is not None:
        return matched
    if allow_custom_value:
        return value

    matched_default = _match_choice(default, choices)
    if matched_default is not None:
        return matched_default
    return choices[0]


def _match_choice(value: Any, choices: Sequence[Any]) -> Any:
    """Return the existing choice that matches a saved value, ignoring case."""

    exact = _first_matching_choice(value, choices, ignore_case=False)
    if exact is not None:
        return exact
    return _first_matching_choice(value, choices, ignore_case=True)


def _first_matching_choice(
    value: Any,
    choices: Sequence[Any],
    *,
    ignore_case: bool,
) -> Any:
    """Return the first choice equal to a saved value."""

    value_text = str(value).casefold()
    for choice in choices:
        if ignore_case and value_text == str(choice).casefold():
            return choice
        if not ignore_case and value == choice:
            return choice
    return None


def _coerce_number(
    value: Any,
    minimum: Any,
    maximum: Any,
    default: Any,
) -> int | float:
    """Clamp a saved numeric value to a component's allowed range."""

    number = _first_number(value, default, minimum, maximum, 0.0)
    lower = _to_float(minimum)
    upper = _to_float(maximum)
    if lower is not None:
        number = max(number, lower)
    if upper is not None:
        number = min(number, upper)
    return _preserve_numeric_type(number, value, default, minimum, maximum)


def _to_float(value: Any) -> float | None:
    """Return a finite float for scalar numeric values."""

    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _first_number(*values: Any) -> float:
    """Return the first finite numeric value from a candidate list."""

    for value in values:
        number = _to_float(value)
        if number is not None:
            return number
    return 0.0


def _preserve_numeric_type(number: float, *templates: Any) -> int | float:
    """Return ints for integer-like controls and floats otherwise."""

    if any(isinstance(template, int) and not isinstance(template, bool) for template in templates):
        return int(round(number))
    return number


def _coerce_bool(value: Any, default: Any) -> bool:
    """Parse common serialized checkbox values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"", "0", "false", "no", "off", "none"}:
            return False
    return bool(default)
