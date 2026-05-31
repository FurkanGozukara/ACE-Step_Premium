"""SAM-Audio tensor separation helpers."""

from __future__ import annotations

import torch
from loguru import logger

from acestep.core.generation.cancellation import check_generation_cancelled

from .attention import attention_backend_context
from .chunking import (
    iter_audio_chunks,
    overlap_add_chunks,
    seconds_to_samples,
    should_process_chunked,
)
from .progress import ProgressCallback, report_progress
from .settings import SamAudioSettings


class SamAudioSeparator:
    """Run SAM-Audio on full or chunked audio tensors."""

    def __init__(
        self,
        *,
        model,
        processor,
        settings: SamAudioSettings,
        device: torch.device,
        dtype: torch.dtype,
        sample_rate: int,
        progress_callback: ProgressCallback | None = None,
        progress_start: float = 0.0,
        progress_end: float = 1.0,
    ) -> None:
        self.model = model
        self.processor = processor
        self.settings = settings
        self.device = device
        self.dtype = dtype
        self.sample_rate = sample_rate
        self.progress_callback = progress_callback
        self.progress_start = progress_start
        self.progress_end = progress_end

    def use_chunking(
        self,
        audio_tensor: torch.Tensor,
        masked_videos: list[torch.Tensor] | None,
    ) -> bool:
        """Return whether the current request should use chunked separation."""

        if not self.settings.chunked or masked_videos is not None:
            return False
        return should_process_chunked(
            audio_tensor,
            self.sample_rate,
            self.settings.chunk_seconds,
        )

    def seed_run(self) -> None:
        """Seed torch once for a full SAM-Audio request."""

        if not self.settings.seed:
            return
        torch.manual_seed(int(self.settings.seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(self.settings.seed))

    def separate_audio(
        self,
        audio_tensor: torch.Tensor,
        *,
        description: str,
        anchors,
        masked_videos: list[torch.Tensor] | None,
    ):
        """Run SAM-Audio separation on one tensor."""

        batch = self.processor(
            descriptions=[description],
            audios=[audio_tensor],
            anchors=anchors,
            masked_videos=masked_videos,
        ).to(self.device)
        if self.dtype is not torch.float32:
            batch.audios = batch.audios.to(dtype=self.dtype)
        ode_opt = {
            "method": "midpoint",
            "options": {"step_size": 1.0 / max(1, int(self.settings.ode_steps))},
        }
        with (
            torch.inference_mode(),
            self._autocast_context(),
            attention_backend_context(self.settings.attention_backend),
        ):
            return self.model.separate(
                batch,
                ode_opt=ode_opt,
                reranking_candidates=int(self.settings.reranking_candidates),
                predict_spans=bool(self.settings.predict_spans),
            )

    def separate_chunked(
        self,
        audio_tensor: torch.Tensor,
        *,
        description: str,
        anchors,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Separate long audio in overlapping chunks and stitch the outputs."""

        target_chunks: list[tuple[int, int, torch.Tensor]] = []
        residual_chunks: list[tuple[int, int, torch.Tensor]] = []
        total_samples = int(audio_tensor.shape[-1])
        overlap = seconds_to_samples(self.settings.chunk_overlap_seconds, self.sample_rate)
        chunks = list(
            iter_audio_chunks(
                audio_tensor,
                self.sample_rate,
                self.settings.chunk_seconds,
                self.settings.chunk_overlap_seconds,
            )
        )
        logger.info("[sam_audio] Processing {} chunks for long audio", len(chunks))
        for index, chunk in enumerate(chunks, start=1):
            check_generation_cancelled()
            self._report_chunk_progress(index - 1, len(chunks), index, "Separating")
            logger.info(
                "[sam_audio] Chunk {}/{} samples {}:{}",
                index,
                len(chunks),
                chunk.start,
                chunk.end,
            )
            result = self.separate_audio(
                chunk.audio,
                description=description,
                anchors=anchors,
                masked_videos=None,
            )
            target_chunks.append((chunk.start, chunk.end, result.target[0]))
            if result.residual:
                residual_chunks.append((chunk.start, chunk.end, result.residual[0]))
            del result
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._report_chunk_progress(index, len(chunks), index, "Finished")
        report_progress(self.progress_callback, self.progress_end, "Stitching audio chunks")
        target = overlap_add_chunks(target_chunks, total_samples, overlap)
        residual = (
            overlap_add_chunks(residual_chunks, total_samples, overlap)
            if residual_chunks
            else None
        )
        return target, residual, len(chunks)

    def _report_chunk_progress(
        self,
        completed: int,
        total: int,
        chunk_index: int,
        action: str,
    ) -> None:
        """Report chunk-level progress within the configured progress range."""

        if total <= 0:
            return
        fraction = self.progress_start + (
            (self.progress_end - self.progress_start) * completed / total
        )
        report_progress(
            self.progress_callback,
            fraction,
            f"{action} SAM-Audio chunk {min(chunk_index, total)}/{total}",
        )

    def _autocast_context(self):
        """Return a CUDA autocast context for BF16/FP16 SAM inference."""

        enabled = self.device.type == "cuda" and self.dtype is not torch.float32
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=enabled,
        )
