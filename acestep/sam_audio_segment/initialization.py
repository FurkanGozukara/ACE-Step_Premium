"""Initialization helpers for SAM-Audio checkpoint-backed model construction."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from threading import RLock
from typing import Callable, Iterator

import torch

from .settings import SamAudioSettings

_RESET_PARAMETER_CLASSES: tuple[type[torch.nn.Module], ...] = (
    torch.nn.Linear,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.Embedding,
    torch.nn.LayerNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.GroupNorm,
)
_INIT_LOCK = RLock()
_SKIPPED_VISUAL_PREFIXES = ("vision_encoder.",)


class _TextOnlyVisionEncoder(torch.nn.Module):
    """Minimal visual encoder used when text/span inference cannot use video masks."""

    def __init__(self, config) -> None:
        super().__init__()
        self.dim = int(getattr(config, "dim", 1024))
        self.batch_size = int(getattr(config, "batch_size", 0))

    def forward(self, videos):
        """Reject visual-mask inference when the visual encoder was skipped."""

        raise ValueError("Text/span SAM-Audio fast load does not support visual masks.")


def should_skip_visual_encoder(settings: SamAudioSettings) -> bool:
    """Return whether the current settings can use the text/span-only model."""

    return settings.prompt_mode != "visual"


def checkpoint_skip_prefixes_for_settings(
    settings: SamAudioSettings,
) -> tuple[str, ...]:
    """Return checkpoint prefixes that are unused for the selected SAM mode."""

    if should_skip_visual_encoder(settings):
        return _SKIPPED_VISUAL_PREFIXES
    return ()


@contextmanager
def fast_checkpoint_model_initialization(
    *,
    skip_visual_encoder: bool = False,
) -> Iterator[None]:
    """Skip default init and optional visual construction before checkpoint load."""

    with _INIT_LOCK:
        reset_originals = _replace_reset_parameters()
        vision_original = _replace_visual_encoder(skip_visual_encoder)
        try:
            yield
        finally:
            _restore_visual_encoder(vision_original)
            _restore_reset_parameters(reset_originals)


@contextmanager
def skip_default_parameter_initialization() -> Iterator[None]:
    """Temporarily skip PyTorch layer resets before strict checkpoint loading."""

    with _INIT_LOCK:
        originals = _replace_reset_parameters()
        try:
            yield
        finally:
            _restore_reset_parameters(originals)


def _replace_reset_parameters() -> list[tuple[type[torch.nn.Module], Callable]]:
    """Replace default PyTorch layer reset methods and return originals."""

    originals: list[tuple[type[torch.nn.Module], Callable]] = []

    def _noop_reset(_module: torch.nn.Module) -> None:
        return None

    for module_class in _RESET_PARAMETER_CLASSES:
        if not hasattr(module_class, "reset_parameters"):
            continue
        originals.append((module_class, module_class.reset_parameters))
        module_class.reset_parameters = _noop_reset
    return originals


def _restore_reset_parameters(
    originals: list[tuple[type[torch.nn.Module], Callable]],
) -> None:
    """Restore PyTorch layer reset methods captured earlier."""

    for module_class, original in reversed(originals):
        module_class.reset_parameters = original


def _replace_visual_encoder(skip_visual_encoder: bool) -> tuple[object, object] | None:
    """Replace vendor visual encoder construction when visual prompts are disabled."""

    if not skip_visual_encoder:
        return None
    model_module = import_module("sam_audio.model.model")
    original = model_module.PerceptionEncoder
    model_module.PerceptionEncoder = _TextOnlyVisionEncoder
    return model_module, original


def _restore_visual_encoder(original: tuple[object, object] | None) -> None:
    """Restore the vendor visual encoder constructor when it was patched."""

    if original is None:
        return
    model_module, encoder_class = original
    model_module.PerceptionEncoder = encoder_class
