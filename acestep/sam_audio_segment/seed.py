"""Seed resolution helpers for SAM-Audio runs."""

from __future__ import annotations

import secrets
from dataclasses import replace

from .settings import SamAudioSettings

MAX_RANDOM_SEED = 2**31 - 1


def resolve_runtime_seed(settings: SamAudioSettings) -> SamAudioSettings:
    """Return settings with an actual seed chosen for this run."""

    if not settings.random_seed:
        return settings
    return replace(settings, seed=secrets.randbelow(MAX_RANDOM_SEED) + 1)
