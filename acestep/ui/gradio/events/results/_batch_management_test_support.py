"""Shared test utilities for ``batch_management`` unit tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from typing import Any, Dict, Tuple
from unittest.mock import patch


_MISSING = object()


def build_progress_result(*, length: int = 56, all_audio_paths: Any = _MISSING) -> tuple:
    """Build a minimally valid ``generate_with_progress`` result tuple."""
    result = [None] * length
    for idx in range(min(8, length)):
        result[idx] = {"value": f"audio_{idx}.flac", "playback_position": 0}
    for idx in range(8, min(16, length)):
        result[idx] = {"value": None, "visible": False, "playback_position": 0}
    if length > 16:
        result[16] = ["audio_0.flac", "audio_0.json"] if all_audio_paths is _MISSING else all_audio_paths
    if length > 17:
        result[17] = "generation info"
    if length > 18:
        result[18] = "Generation Complete"
    if length > 19:
        result[19] = "42"
    if length > 52:
        result[52] = {"bpm": 120}
    if length > 53:
        result[53] = False
    if length > 54:
        result[54] = {"lrcs": ["lrc"] * 8, "subtitles": ["sub"] * 8}
    if length > 55:
        result[55] = ["codes"] * 8
    if length > 56:
        result[56] = {"future_tail_field": True}
    return tuple(result)


def load_batch_management_module(
    *, is_windows: bool = False, mps_available: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """Load ``batch_management.py`` with dependency stubs and trackers.

    Args:
        is_windows: Simulate Windows platform flag in ``generation_info`` stub.
        mps_available: When True, the fake ``torch`` stub will expose
            ``torch.mps.empty_cache`` so MPS code-paths can be exercised.
    """
    state: Dict[str, Any] = {
        "store_calls": [],
        "info_messages": [],
        "warning_messages": [],
        "log_info": [],
        "log_warning": [],
        "cuda_empty_cache_calls": 0,
        "mps_empty_cache_calls": 0,
    }

    def _gr_update(**kwargs):
        """Return a deterministic Gradio-like update payload."""
        return {"kind": "update", **kwargs}

    def _gr_skip():
        """Return a deterministic Gradio-like skip payload."""
        return {"kind": "skip"}

    def _gr_info(message):
        """Capture ``gr.Info`` messages for assertions."""
        state["info_messages"].append(message)

    def _gr_warning(message):
        """Capture ``gr.Warning`` messages for assertions."""
        state["warning_messages"].append(message)

    def _logger_info(*message):
        """Capture logger info messages for assertions."""
        if message:
            state["log_info"].append(" ".join(str(part) for part in message))

    def _logger_warning(*message):
        """Capture logger warning messages for assertions."""
        if message:
            state["log_warning"].append(" ".join(str(part) for part in message))

    def _default_generate_with_progress(*_args, **_kwargs):
        """Default empty generator placeholder patched per test."""
        if False:
            yield None

    def _store_batch_in_queue(batch_queue, batch_idx, all_audio_paths, generation_info, seed_value_for_ui, **kwargs):
        """Store synthetic batch data and keep call history for assertions."""
        call = {
            "batch_queue": dict(batch_queue),
            "batch_idx": batch_idx,
            "all_audio_paths": all_audio_paths,
            "generation_info": generation_info,
            "seed_value_for_ui": seed_value_for_ui,
            **kwargs,
        }
        state["store_calls"].append(call)
        next_queue = dict(batch_queue)
        next_queue[batch_idx] = {"status": "completed", **call}
        return next_queue

    def _translate(key, **kwargs):
        """Return predictable translation output with formatted kwargs."""
        return f"{key}|{kwargs}" if kwargs else key

    # --- Fake torch stub --------------------------------------------------
    # Provides just enough of the torch API used by batch-management modules
    # (torch.cuda.is_available, torch.cuda.empty_cache, torch.mps.empty_cache).
    # Call counts are recorded in ``state`` for assertion in tests.
    def _cuda_empty_cache():
        state["cuda_empty_cache_calls"] += 1

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=_cuda_empty_cache,
    )

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = fake_cuda
    fake_torch.Tensor = object
    fake_torch.nn = types.SimpleNamespace(Module=object)

    fake_backends_mps = types.SimpleNamespace(is_available=lambda: mps_available)
    fake_torch.backends = types.SimpleNamespace(mps=fake_backends_mps)

    if mps_available:
        def _mps_empty_cache():
            state["mps_empty_cache_calls"] += 1

        fake_torch.mps = types.SimpleNamespace(empty_cache=_mps_empty_cache)
    # When mps_available is False, ``torch.mps`` is absent, so
    # ``hasattr(torch, "mps")`` returns False and the cache call is skipped.
    # -----------------------------------------------------------------------

    acestep_pkg = types.ModuleType("acestep")
    ui_pkg = types.ModuleType("acestep.ui")
    gradio_pkg = types.ModuleType("acestep.ui.gradio")
    events_pkg = types.ModuleType("acestep.ui.gradio.events")
    results_pkg = types.ModuleType("acestep.ui.gradio.events.results")
    core_pkg = types.ModuleType("acestep.core")
    generation_pkg = types.ModuleType("acestep.core.generation")
    handler_pkg = types.ModuleType("acestep.core.generation.handler")
    lora_pkg = types.ModuleType("acestep.core.generation.handler.lora")
    sam_audio_pkg = types.ModuleType("acestep.sam_audio_segment")

    results_dir = Path(__file__).resolve().parent
    events_dir = results_dir.parent
    gradio_dir = events_dir.parent
    ui_dir = gradio_dir.parent
    acestep_dir = ui_dir.parent

    acestep_pkg.__path__ = [str(acestep_dir)]
    ui_pkg.__path__ = [str(ui_dir)]
    gradio_pkg.__path__ = [str(gradio_dir)]
    events_pkg.__path__ = [str(events_dir)]
    results_pkg.__path__ = [str(results_dir)]
    core_pkg.__path__ = [str(acestep_dir / "core")]
    generation_pkg.__path__ = [str(acestep_dir / "core" / "generation")]
    handler_pkg.__path__ = [str(acestep_dir / "core" / "generation" / "handler")]
    lora_pkg.__path__ = [str(acestep_dir / "core" / "generation" / "handler" / "lora")]
    sam_audio_pkg.__path__ = [str(acestep_dir / "sam_audio_segment")]

    fake_gradio = types.ModuleType("gradio")
    fake_gradio.update = _gr_update
    fake_gradio.skip = _gr_skip
    fake_gradio.Progress = lambda track_tqdm=True: None
    fake_gradio.Info = _gr_info
    fake_gradio.Warning = _gr_warning

    fake_logger = types.SimpleNamespace(
        info=_logger_info,
        warning=_logger_warning,
        error=lambda _msg: None,
        exception=_logger_warning,
    )

    modules = {
        "gradio": fake_gradio,
        "torch": fake_torch,
        "loguru": types.SimpleNamespace(logger=fake_logger),
        "acestep": acestep_pkg,
        "acestep.core": core_pkg,
        "acestep.core.generation": generation_pkg,
        "acestep.core.generation.handler": handler_pkg,
        "acestep.core.generation.handler.lora": lora_pkg,
        "acestep.core.generation.handler.lora.folder_scan": types.SimpleNamespace(
            resolve_loadable_lora_adapter_path=lambda candidate: str(candidate or "").strip()
        ),
        "acestep.sam_audio_segment": sam_audio_pkg,
        "acestep.sam_audio_segment.settings": types.SimpleNamespace(
            SAM_AUDIO_PRESET_KEYS=(),
            settings_from_ui_values=lambda _values: types.SimpleNamespace(to_payload=lambda: {}),
        ),
        "acestep.ui": ui_pkg,
        "acestep.ui.gradio": gradio_pkg,
        "acestep.ui.gradio.i18n": types.SimpleNamespace(t=_translate),
        "acestep.ui.gradio.events": events_pkg,
        "acestep.ui.gradio.events.results": results_pkg,
        "acestep.ui.gradio.events.results.generation_info": types.SimpleNamespace(IS_WINDOWS=is_windows),
        "acestep.ui.gradio.events.results.generation_progress": types.SimpleNamespace(
            generate_with_progress=_default_generate_with_progress
        ),
        "acestep.ui.gradio.events.results.batch_queue": types.SimpleNamespace(
            store_batch_in_queue=_store_batch_in_queue,
            update_batch_indicator=lambda current, total: f"Batch {current + 1}/{total}",
            update_navigation_buttons=lambda current, total: (current > 0, current < total - 1),
        ),
    }

    acestep_pkg.ui = ui_pkg
    acestep_pkg.core = core_pkg
    core_pkg.generation = generation_pkg
    generation_pkg.handler = handler_pkg
    handler_pkg.lora = lora_pkg
    acestep_pkg.sam_audio_segment = sam_audio_pkg
    ui_pkg.gradio = gradio_pkg
    gradio_pkg.events = events_pkg
    events_pkg.results = results_pkg

    for module_name in list(sys.modules):
        if module_name.startswith("acestep.ui.gradio.events.results.batch_management"):
            sys.modules.pop(module_name, None)

    module_path = Path(__file__).with_name("batch_management.py")
    spec = importlib.util.spec_from_file_location("batch_management", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", modules):
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module, state
