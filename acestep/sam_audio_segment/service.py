"""SAM-Audio model loading and file processing service."""

from __future__ import annotations

import gc
import importlib
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from loguru import logger

from acestep.sam_audio_vendor import ensure_vendor_path
from acestep.torch_compile_runtime import TorchCompileResult, compile_module_forward

from .attention import attention_stats, reset_attention_stats
from .anchors import anchors_for_settings
from .fp8_scaled import apply_sam_fp8_scaled
from .initialization import (
    checkpoint_skip_prefixes_for_settings,
    fast_checkpoint_model_initialization,
    should_skip_visual_encoder,
)
from .lite_mode import apply_text_lite_mode, validate_text_lite_settings
from .media_output import SamAudioArtifacts, save_sam_audio_outputs
from .memory import peak_memory_metrics, reset_peak_memory
from .model_config import config_for_settings
from .paths import default_model_path, safe_media_stem
from .progress import ProgressCallback, report_progress
from .runtime import load_checkpoint, read_audio_tensor, resolve_device, resolve_dtype
from .seed import resolve_runtime_seed
from .separation import SamAudioSeparator
from .separation_multidiffusion import (
    separate_multidiffusion_with_separator,
    should_use_multidiffusion_long_audio,
)
from .settings import SamAudioSettings
from .video_mask import load_masked_video_tensor


class SamAudioService:
    """Load SAM-Audio Large and process media files."""

    def __init__(
        self,
        settings: SamAudioSettings,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.settings = resolve_runtime_seed(settings)
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path
            else default_model_path()
        )
        requested_device = settings.device_mode if device == "auto" else device
        self.device = resolve_device(requested_device)
        self.dtype = resolve_dtype(self.device)
        self.model = None
        self.processor = None
        self.compile_result: TorchCompileResult | None = None
        self.progress_callback = progress_callback
        self.sample_rate = 48000

    def load(self) -> None:
        """Load the SAM-Audio model and processor."""

        if self.model is not None and self.processor is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"SAM-Audio checkpoint not found: {self.model_path}")
        validate_text_lite_settings(self.settings)

        model_name = self.model_path.name
        report_progress(
            self.progress_callback,
            0.03,
            f"Initializing SAM-Audio with {model_name}",
        )
        ensure_vendor_path()
        from sam_audio.model.config import SAMAudioConfig
        from sam_audio.processor import SAMAudioProcessor

        started = time.time()
        report_progress(
            self.progress_callback,
            0.06,
            f"Building SAM-Audio model for {model_name}",
        )
        model_config = config_for_settings(self.settings)
        _set_local_ranker_map_location(model_config, self.device)
        logger.info(
            "[sam_audio] Runtime settings: prompt_mode={} candidates={} "
            "ranker={} text_ranker={} visual_ranker={} span_predictor={} "
            "long_mode={} chunk_seconds={} overlap_seconds={} lite={}",
            self.settings.prompt_mode,
            self.settings.reranking_candidates,
            self.settings.ranker_mode,
            _ranker_kind(model_config.get("text_ranker")),
            _ranker_kind(model_config.get("visual_ranker")),
            model_config.get("span_predictor") or "disabled",
            self.settings.long_audio_mode,
            self.settings.chunk_seconds,
            self.settings.chunk_overlap_seconds,
            self.settings.low_vram_lite,
        )
        config = SAMAudioConfig(**model_config)
        skip_visual_encoder = should_skip_visual_encoder(self.settings)
        skip_prefixes = checkpoint_skip_prefixes_for_settings(self.settings)
        try:
            model, meta_direct_init = _build_sam_audio_model(
                config=config,
                settings=self.settings,
                model_path=self.model_path,
                device=self.device,
                skip_visual_encoder=skip_visual_encoder,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize SAM-Audio optional components: {exc}"
            ) from exc
        report_progress(
            self.progress_callback,
            0.12,
            f"Loading SAM-Audio checkpoint: {model_name}",
        )
        if self.settings.low_vram_lite:
            _load_checkpoint_into_model(
                model,
                self.model_path,
                self.device,
                direct_device_load=False,
                skip_prefixes=skip_prefixes,
            )
            apply_text_lite_mode(model)
        else:
            _load_checkpoint_into_model(
                model,
                self.model_path,
                self.device,
                skip_prefixes=skip_prefixes,
            )
        if meta_direct_init:
            _materialize_deferred_sam_text_encoder(model, config)
        report_progress(
            self.progress_callback,
            0.18,
            f"Moving {model_name} to {self.device} as {self.dtype}",
        )
        if self.dtype is not torch.float32:
            model = model.to(dtype=self.dtype)
        model = model.to(self.device)
        if self.settings.quantization == "fp8_scaled":
            report_progress(
                self.progress_callback,
                0.21,
                f"Applying SAM-Audio FP8 cache for {model_name}",
            )
            apply_sam_fp8_scaled(model, checkpoint_path=self.model_path, device=self.device)
        if self.settings.compile_model:
            report_progress(
                self.progress_callback,
                0.23,
                f"Compiling SAM-Audio forward for {model_name}",
            )
        self.compile_result = compile_module_forward(
            model,
            label="SAM-Audio",
            enabled=bool(self.settings.compile_model),
        )
        if self.settings.compile_model:
            if not self.compile_result.compiled:
                logger.warning(
                    "[sam_audio] torch.compile disabled: {}",
                    self.compile_result.detail,
                )

        self.model = model
        self.processor = SAMAudioProcessor(
            audio_hop_length=config.audio_codec.hop_length,
            audio_sampling_rate=config.audio_codec.sample_rate,
        )
        self.sample_rate = int(config.audio_codec.sample_rate)
        logger.info(
            "[sam_audio] Loaded {} on {} dtype={} in {:.2f}s",
            self.model_path.name,
            self.device,
            self.dtype,
            time.time() - started,
        )
        report_progress(
            self.progress_callback,
            0.25,
            f"SAM-Audio model ready: {model_name}",
        )

    def process_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        output_stem: str | None = None,
        mask_video_path: str | Path | None = None,
        output_only: bool = False,
    ) -> SamAudioArtifacts:
        """Run SAM-Audio segmentation for one media file."""

        report_progress(
            self.progress_callback,
            0.0,
            f"Starting SAM-Audio with {self.model_path.name}",
        )
        self.load()
        assert self.model is not None and self.processor is not None
        source = Path(input_path).expanduser().resolve()
        stem = output_stem or f"{safe_media_stem(source)}_sam"
        report_progress(self.progress_callback, 0.28, "Reading input media")
        audio_tensor = read_audio_tensor(source, self.sample_rate)
        report_progress(self.progress_callback, 0.32, "Preparing SAM-Audio prompt")
        masked_videos = self._masked_videos(source, mask_video_path)
        anchors = anchors_for_settings(self.settings)
        description = self._description()
        separator = SamAudioSeparator(
            model=self.model,
            processor=self.processor,
            settings=self.settings,
            device=self.device,
            dtype=self.dtype,
            sample_rate=self.sample_rate,
            progress_callback=self.progress_callback,
            progress_start=0.35,
            progress_end=0.90,
        )
        separator.seed_run()
        reset_peak_memory(self.device)
        reset_attention_stats(self.settings.attention_backend)
        started = time.time()
        if should_use_multidiffusion_long_audio(
            self.settings,
            audio_tensor,
            self.sample_rate,
            masked_videos,
            anchors,
        ):
            target, residual, chunk_count = separate_multidiffusion_with_separator(
                separator,
                audio_tensor,
                description=description,
                anchors=anchors,
            )
        elif separator.use_chunking(audio_tensor, masked_videos):
            target, residual, chunk_count = separator.separate_chunked(
                audio_tensor,
                description=description,
                anchors=anchors,
            )
        else:
            report_progress(self.progress_callback, 0.45, "Separating audio")
            result = separator.separate_audio(
                audio_tensor,
                description=description,
                anchors=anchors,
                masked_videos=masked_videos,
            )
            target = result.target[0]
            residual = result.residual[0] if result.residual else None
            chunk_count = 1
            report_progress(self.progress_callback, 0.90, "Audio separation complete")
        metadata = self._metadata(started, chunk_count, description, anchors)
        report_progress(self.progress_callback, 0.94, "Saving SAM-Audio outputs")
        artifacts = save_sam_audio_outputs(
            source_path=source,
            output_dir=output_dir,
            output_stem=stem,
            target=target,
            residual=residual,
            sample_rate=self.sample_rate,
            output_format=self.settings.output_format,
            include_residual=self.settings.include_residual and not output_only,
            include_video=self.settings.include_video and not output_only,
            metadata=metadata,
            trim_empty_output=self.settings.trim_empty_output,
            trim_settings=self.settings.trim_settings(),
            trim_threshold_db=self.settings.trim_threshold_db,
            save_metadata=not output_only,
        )
        report_progress(self.progress_callback, 1.0, "SAM-Audio complete")
        return artifacts

    def unload(self) -> None:
        """Release model memory held by the service."""

        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _metadata(
        self,
        started: float,
        chunk_count: int,
        description: str,
        anchors,
    ) -> dict:
        """Return SAM-Audio run metadata."""

        return {
            "settings": self.settings.to_payload(),
            "model": {
                "path": str(self.model_path).replace("\\", "/"),
                "device": str(self.device),
                "dtype": str(self.dtype),
                "torch_compile": self._compile_metadata(),
            },
            "prompt": {
                "mode": self.settings.prompt_mode,
                "description": description,
                "anchors": anchors,
            },
            "chunking": {
                "enabled": chunk_count > 1,
                "mode": self.settings.long_audio_mode,
                "chunks": chunk_count,
                "chunk_seconds": self.settings.chunk_seconds,
                "overlap_seconds": self.settings.chunk_overlap_seconds,
            },
            "timing": {"separation_seconds": round(time.time() - started, 3)},
            "memory": peak_memory_metrics(self.device),
            "attention": attention_stats(),
        }

    def _description(self) -> str:
        """Return the model description for the selected prompt mode."""

        if self.settings.prompt_mode == "visual":
            return self.settings.custom_prompt.strip().lower()
        return self.settings.effective_prompt.lower()

    def _masked_videos(
        self,
        source: Path,
        mask_video_path: str | Path | None,
    ) -> list[torch.Tensor] | None:
        """Return masked video tensors for visual prompting."""

        if self.settings.prompt_mode != "visual":
            return None
        if not mask_video_path:
            raise ValueError("Visual prompting requires a mask video/image file.")
        mask = Path(mask_video_path).expanduser().resolve()
        return [load_masked_video_tensor(source, mask)]

    def _compile_metadata(self) -> dict[str, object]:
        """Return torch.compile metadata for the loaded model."""

        result = self.compile_result
        return {
            "requested": bool(self.settings.compile_model),
            "compiled": bool(getattr(self.model, "_acestep_torch_compiled", False)),
            "attempts": int(
                getattr(self.model, "_acestep_torch_compile_attempts", 0)
            ),
            "detail": getattr(
                self.model,
                "_acestep_torch_compile_detail",
                result.detail if result is not None else "",
            ),
        }


def _ranker_kind(config: object) -> str:
    """Return a concise optional-ranker name for logs."""

    if not isinstance(config, dict):
        return "disabled"
    return str(config.get("kind") or "unknown")


def _build_sam_audio_model(
    *,
    config,
    settings: SamAudioSettings,
    model_path: Path,
    device: torch.device,
    skip_visual_encoder: bool,
) -> tuple[torch.nn.Module, bool]:
    """Build SAM-Audio, using meta init for direct CUDA safetensors when safe."""

    ensure_vendor_path()
    from sam_audio.model.model import SAMAudio

    use_meta_direct_init = _should_use_meta_direct_sam_load(
        settings,
        model_path=model_path,
        device=device,
        skip_visual_encoder=skip_visual_encoder,
    )
    with fast_checkpoint_model_initialization(
        skip_visual_encoder=skip_visual_encoder,
    ):
        if use_meta_direct_init:
            with _sam_audio_meta_initialization_context():
                with torch.device("meta"):
                    model = SAMAudio(config).eval()
            logger.info(
                "[sam_audio] Built checkpoint-backed modules on meta for direct CUDA assignment"
            )
            return model, True
        model = SAMAudio(config).eval()
        return model, False


def _should_use_meta_direct_sam_load(
    settings: SamAudioSettings,
    *,
    model_path: Path,
    device: torch.device,
    skip_visual_encoder: bool,
) -> bool:
    """Return whether SAM-Audio can skip real CPU construction for large modules."""

    return (
        device.type == "cuda"
        and model_path.suffix.lower() == ".safetensors"
        and not settings.low_vram_lite
        and skip_visual_encoder
        and not settings.predict_spans
        and str(settings.ranker_mode).lower() == "none"
    )


@contextmanager
def _sam_audio_meta_initialization_context():
    """Defer external text encoder loading while SAM checkpoint modules use meta."""

    model_module = importlib.import_module("sam_audio.model.model")
    original_text_encoder = model_module.T5TextEncoder
    original_arange = torch.arange
    original_tensor = torch.tensor

    class _DeferredTextEncoder(torch.nn.Module):
        """Placeholder replaced by the real T5 encoder after checkpoint assignment."""

        _acestep_deferred_text_encoder = True

        def __init__(self, config) -> None:
            super().__init__()
            self.config = config

        def forward(self, *_args, **_kwargs):
            raise RuntimeError("Deferred SAM-Audio text encoder was not materialized.")

    def _cpu_arange_without_explicit_device(*args, **kwargs):
        if kwargs.get("device") is None:
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
        return original_arange(*args, **kwargs)

    def _cpu_tensor_without_explicit_device(*args, **kwargs):
        if kwargs.get("device") is None:
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    model_module.T5TextEncoder = _DeferredTextEncoder
    torch.arange = _cpu_arange_without_explicit_device
    torch.tensor = _cpu_tensor_without_explicit_device
    try:
        yield
    finally:
        model_module.T5TextEncoder = original_text_encoder
        torch.arange = original_arange
        torch.tensor = original_tensor


def _materialize_deferred_sam_text_encoder(model: torch.nn.Module, config) -> None:
    """Replace the meta-init placeholder with the real SAM text encoder."""

    if not getattr(
        getattr(model, "text_encoder", None),
        "_acestep_deferred_text_encoder",
        False,
    ):
        return
    from sam_audio.model.text_encoder import T5TextEncoder

    model.text_encoder = T5TextEncoder(config.text_encoder)
    logger.info("[sam_audio] Materialized deferred T5 text encoder")


def _load_checkpoint_into_model(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    *,
    direct_device_load: bool = True,
    skip_prefixes: tuple[str, ...] = (),
) -> None:
    """Load a checkpoint into ``model``, optionally skipping unused prefixes."""

    load_device = _checkpoint_load_device(path, device, direct_device_load)
    state_dict: dict[str, torch.Tensor] | None = None
    try:
        state_dict = load_checkpoint(
            path,
            device=load_device,
            skip_prefixes=skip_prefixes,
        )
        assign = load_device != "cpu"
        try:
            model.load_state_dict(state_dict, strict=True, assign=assign)
        except TypeError:
            if not assign:
                raise
            del state_dict
            state_dict = None
            state_dict = load_checkpoint(
                path,
                device="cpu",
                skip_prefixes=skip_prefixes,
            )
            model.load_state_dict(state_dict, strict=True)
    finally:
        state_dict = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    logger.info(
        "[sam_audio] Loaded checkpoint tensors from {} via {}",
        path.name,
        load_device,
    )


def _checkpoint_load_device(
    path: Path,
    device: torch.device,
    direct_device_load: bool,
) -> str:
    """Return direct checkpoint load device for safetensors on CUDA."""

    if (
        direct_device_load
        and device.type == "cuda"
        and path.suffix.lower() == ".safetensors"
    ):
        return str(device)
    return "cpu"


def _set_local_ranker_map_location(config: dict, device: torch.device) -> None:
    """Set local Judge ranker checkpoint placement in a nested SAM-Audio config."""

    map_location = str(device) if device.type == "cuda" else "cpu"
    for key in ("text_ranker", "visual_ranker"):
        _set_ranker_map_location(config.get(key), map_location)


def _set_ranker_map_location(config: object, map_location: str) -> None:
    """Apply ``map_location`` to Judge rankers without touching other ranker types."""

    if not isinstance(config, dict):
        return
    if config.get("kind") == "judge":
        config["map_location"] = map_location
        return
    if config.get("kind") != "ensemble":
        return
    for ranker_pair in config.get("rankers", {}).values():
        if isinstance(ranker_pair, (list, tuple)) and ranker_pair:
            _set_ranker_map_location(ranker_pair[0], map_location)
