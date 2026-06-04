"""Process-local SAM-Audio service cache for same-process inference."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Iterator

from .initialization import should_skip_visual_encoder
from .paths import default_model_path
from .progress import ProgressCallback
from .seed import resolve_runtime_seed
from .service import SamAudioService
from .settings import SamAudioSettings

_MODEL_SETTING_FIELDS: tuple[str, ...] = (
    "predict_spans",
    "ranker_mode",
    "quantization",
    "low_vram_lite",
)


class _SamAudioServiceCache:
    """Keep one compatible SAM-Audio service loaded inside the current process."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._key: tuple[object, ...] | None = None
        self._service: SamAudioService | None = None

    @contextmanager
    def session(
        self,
        settings: SamAudioSettings,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        progress_callback: ProgressCallback | None = None,
    ) -> Iterator[SamAudioService]:
        """Yield a cached compatible service, rebuilding only when required."""

        with self._lock:
            key = _cache_key(settings, model_path=model_path, device=device)
            if self._service is None or self._key != key:
                self._unload_locked()
                self._service = SamAudioService(
                    settings,
                    model_path=model_path,
                    device=device,
                    progress_callback=progress_callback,
                )
                self._key = key
            else:
                self._service.settings = resolve_runtime_seed(settings)
                self._service.progress_callback = progress_callback
            yield self._service

    def clear(self) -> None:
        """Unload and forget the cached SAM-Audio service."""

        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        """Unload the current service while the cache lock is held."""

        if self._service is not None:
            self._service.unload()
        self._service = None
        self._key = None


_CACHE = _SamAudioServiceCache()


@contextmanager
def cached_sam_audio_service(
    settings: SamAudioSettings,
    *,
    model_path: str | Path | None = None,
    device: str = "auto",
    progress_callback: ProgressCallback | None = None,
) -> Iterator[SamAudioService]:
    """Yield the process-local SAM-Audio service for same-process requests."""

    with _CACHE.session(
        settings,
        model_path=model_path,
        device=device,
        progress_callback=progress_callback,
    ) as service:
        yield service


def clear_cached_sam_audio_service() -> None:
    """Unload the process-local SAM-Audio service cache."""

    _CACHE.clear()


def _cache_key(
    settings: SamAudioSettings,
    *,
    model_path: str | Path | None,
    device: str,
) -> tuple[object, ...]:
    """Return the values that affect SAM-Audio model construction/loading."""

    checkpoint = Path(model_path).expanduser().resolve() if model_path else default_model_path()
    requested_device = settings.device_mode if device == "auto" else device
    setting_values = tuple(getattr(settings, field) for field in _MODEL_SETTING_FIELDS)
    return (
        str(checkpoint),
        str(requested_device),
        should_skip_visual_encoder(settings),
        *setting_values,
    )
