"""SAM-Audio separator integration for text-only multi-diffusion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from acestep.core.generation.cancellation import check_generation_cancelled

from .attention import attention_backend_context
from .cancel import check_sam_audio_cancelled
from .chunking import should_process_chunked
from .multidiffusion import separate_text_multidiffusion
from .progress import report_progress
from .settings import (
    SAM_AUDIO_LONG_MODE_MULTIDIFFUSION,
    SamAudioSettings,
)

if TYPE_CHECKING:
    from .separation import SamAudioSeparator


def should_use_multidiffusion_long_audio(
    settings: SamAudioSettings,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    masked_videos: list[torch.Tensor] | None,
    anchors: Any,
) -> bool:
    """Return whether a long-audio request should use text-only multi-diffusion."""

    if not settings.chunked or settings.long_audio_mode != SAM_AUDIO_LONG_MODE_MULTIDIFFUSION:
        return False
    if not should_process_chunked(audio_tensor, sample_rate, settings.chunk_seconds):
        return False
    _validate_text_only_request(settings, masked_videos, anchors)
    return True


def separate_multidiffusion_with_separator(
    separator: "SamAudioSeparator",
    audio_tensor: torch.Tensor,
    *,
    description: str,
    anchors: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Run text-only multi-diffusion with a configured separator instance."""

    report_progress(
        separator.progress_callback,
        separator.progress_start,
        "Preparing SAM-Audio multi-diffusion",
    )
    batch = separator.processor(
        descriptions=[description],
        audios=[audio_tensor],
        anchors=anchors,
        masked_videos=None,
    ).to(separator.device)
    if separator.dtype is not torch.float32:
        batch.audios = batch.audios.to(dtype=separator.dtype)
    ode_opt = {
        "method": "midpoint",
        "options": {"step_size": 1.0 / max(1, int(separator.settings.ode_steps))},
    }
    with (
        torch.inference_mode(),
        separator._autocast_context(),
        attention_backend_context(separator.settings.attention_backend),
    ):
        result = separate_text_multidiffusion(
            separator.model,
            batch,
            ode_opt=ode_opt,
            window_seconds=separator.settings.chunk_seconds,
            overlap_seconds=separator.settings.chunk_overlap_seconds,
            sample_rate=separator.sample_rate,
            progress_callback=lambda done, total: _report_progress(
                separator,
                done,
                total,
            ),
            cancel_callback=_check_cancelled,
        )
    target = result.target[0]
    residual = result.residual[0] if result.residual else None
    report_progress(
        separator.progress_callback,
        separator.progress_end,
        "Audio separation complete",
    )
    return target, residual, result.window_count


def _validate_text_only_request(
    settings: SamAudioSettings,
    masked_videos: list[torch.Tensor] | None,
    anchors: Any,
) -> None:
    """Raise if a multi-diffusion request is outside the implemented subset."""

    if settings.prompt_mode != "text" or masked_videos is not None or anchors is not None:
        raise ValueError(
            "Multi-diffusion long audio currently supports text-only prompts "
            "without visual masks or span anchors."
        )
    if settings.predict_spans:
        raise ValueError("Multi-diffusion long audio does not support predicted spans yet.")
    if int(settings.reranking_candidates) != 1:
        raise ValueError("Multi-diffusion long audio requires Candidates = 1.")


def _report_progress(separator: "SamAudioSeparator", completed: int, total: int) -> None:
    """Report midpoint progress within the separator progress range."""

    if total <= 0:
        return
    fraction = separator.progress_start + (
        (separator.progress_end - separator.progress_start) * completed / total
    )
    report_progress(
        separator.progress_callback,
        fraction,
        f"SAM-Audio multi-diffusion step {completed}/{total}",
    )


def _check_cancelled() -> None:
    """Raise if the user has cancelled the active request."""

    check_generation_cancelled()
    check_sam_audio_cancelled()
