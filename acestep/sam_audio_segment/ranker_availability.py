"""Availability checks for optional SAM-Audio rankers."""

from __future__ import annotations

from importlib.util import find_spec

from .judge_assets import local_judge_assets_available

RANKER_CHOICES_ALL: tuple[tuple[str, str], ...] = (
    ("Disabled", "none"),
    ("Official text ensemble (CLAP + Judge)", "text_ensemble"),
    ("CLAP text similarity", "clap"),
    ("SAM-Audio Judge", "judge"),
    ("ImageBind visual similarity", "imagebind"),
)


def available_ranker_choices() -> tuple[tuple[str, str], ...]:
    """Return ranker choices whose optional dependencies are importable."""

    return tuple(choice for choice in RANKER_CHOICES_ALL if is_ranker_available(choice[1]))


def normalize_ranker_mode(value: object) -> str:
    """Return a supported ranker mode, falling back to disabled."""

    mode = str(value or "none").strip()
    return mode if is_ranker_available(mode) else "none"


def is_ranker_available(mode: str) -> bool:
    """Return whether a ranker can be initialized in the current environment."""

    if mode == "none":
        return True
    if mode == "judge":
        return (
            find_spec("safetensors") is not None
            and find_spec("transformers") is not None
            and local_judge_assets_available()
        )
    if mode == "clap":
        return find_spec("laion_clap") is not None
    if mode == "text_ensemble":
        return find_spec("laion_clap") is not None and is_ranker_available("judge")
    if mode == "imagebind":
        return find_spec("imagebind") is not None
    return False
