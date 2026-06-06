"""DiffPitcher inference service for vocal pitch correction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .diffpitcher_audio_io import load_mono_24k, restore_output_shape, to_mono_24k
from .diffpitcher_constants import DIFFPITCHER_F0_CLAMP_MAX, DIFFPITCHER_F0_MIN
from .diffpitcher_inference_loop import fit_frames, run_diffusion, vocode
from .diffpitcher_mel_features import world_mel_from_wav
from .diffpitcher_pitch_features import (
    estimate_f0_world,
    log_f0_bins,
    matched_reference_f0,
    midi_f0_to_frames,
)
from .diffpitcher_quality import build_pitch_guide_report, emit_pitch_guide_report
from .diffpitcher_runtime import (
    DiffPitcherRuntime,
    get_diffpitcher_runtime,
    select_diffpitcher_device,
)
from .diffpitcher_settings import (
    DIFFPITCHER_MODE_SCORE,
    DIFFPITCHER_MODE_TEMPLATE,
    DiffPitcherSettings,
)
from .process_logging import emit_process_message


def apply_diffpitcher(
    audio: np.ndarray,
    sample_rate: int,
    settings: DiffPitcherSettings,
    *,
    progress_callback=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply optional DiffPitcher inference to channel-last audio."""

    if not settings.enabled:
        return audio, {"enabled": False, "applied": False}

    _validate_required_guide(settings)
    original = np.asarray(audio, dtype=np.float32)
    original_channels = 1 if original.ndim == 1 else int(original.shape[1])
    original_samples = int(original.shape[0])
    source_wav = to_mono_24k(original, sample_rate)
    device = select_diffpitcher_device(settings.device)
    emit_process_message(progress_callback, "DiffPitcher pitch fix: preparing features")
    source_mel = world_mel_from_wav(source_wav)
    runtime = None
    if settings.mode == DIFFPITCHER_MODE_SCORE:
        emit_process_message(progress_callback, "DiffPitcher pitch fix: loading models")
        runtime = get_diffpitcher_runtime(device, needs_pitchformer=True)
    target_f0 = _target_f0(
        settings,
        runtime,
        source_wav,
        source_mel,
        progress_callback=progress_callback,
    )
    source_f0 = fit_frames(estimate_f0_world(source_wav), len(target_f0))
    pitch_guide = build_pitch_guide_report(target_f0, source_f0=source_f0)
    emit_pitch_guide_report(pitch_guide, progress_callback)
    if settings.mode == DIFFPITCHER_MODE_TEMPLATE and pitch_guide.get("unsafe"):
        emit_process_message(
            progress_callback,
            "DiffPitcher pitch fix: skipped unsafe template guide",
        )
        return original, _metadata(settings, device, False, pitch_guide)
    shifted_f0 = target_f0 * float(2 ** (settings.shift_semitones / 12.0))
    f0_bins = log_f0_bins(shifted_f0)
    if runtime is None:
        emit_process_message(progress_callback, "DiffPitcher pitch fix: loading models")
        runtime = get_diffpitcher_runtime(device, needs_pitchformer=False)
    emit_process_message(progress_callback, "DiffPitcher pitch fix: running diffusion")
    predicted = run_diffusion(
        runtime,
        source_mel,
        f0_bins,
        settings.steps,
        progress_callback=progress_callback,
    )
    emit_process_message(progress_callback, "DiffPitcher pitch fix: rendering vocal audio")
    output = vocode(runtime, predicted)
    restored = restore_output_shape(output, original_samples, original_channels, sample_rate)
    return restored, _metadata(settings, device, True, pitch_guide)


def _metadata(
    settings: DiffPitcherSettings,
    device: torch.device,
    applied: bool,
    pitch_guide: dict[str, Any],
) -> dict[str, Any]:
    """Return JSON-safe DiffPitcher processing metadata."""

    metadata = {
        "enabled": True,
        "applied": bool(applied),
        "mode": settings.mode,
        "steps": settings.steps,
        "shift_semitones": settings.shift_semitones,
        "mask_with_source": settings.mask_with_source,
        "device": str(device),
        "reference_audio": _basename(settings.reference_audio),
        "midi_path": _basename(settings.midi_path),
        "pitch_guide": pitch_guide,
    }
    if not applied:
        metadata["reason"] = "unsafe_template_guide"
    return metadata


def _target_f0(
    settings: DiffPitcherSettings,
    runtime: DiffPitcherRuntime | None,
    source_wav: np.ndarray,
    source_mel: np.ndarray,
    *,
    progress_callback=None,
) -> np.ndarray:
    """Return frame-aligned target F0 values for the selected mode."""

    if settings.mode == DIFFPITCHER_MODE_TEMPLATE:
        reference_path = _require_user_file(settings.reference_audio, "reference vocal")
        emit_process_message(progress_callback, "DiffPitcher pitch fix: aligning reference vocal")
        return fit_frames(
            matched_reference_f0(
                source_wav,
                load_mono_24k(reference_path),
                progress_callback=progress_callback,
            ),
            source_mel.shape[-1],
        )
    if settings.mode == DIFFPITCHER_MODE_SCORE:
        midi_path = _require_user_file(settings.midi_path, "MIDI score")
        if runtime is None:
            raise RuntimeError("DiffPitcher score mode requires loaded models.")
        return _score_target_f0(settings, runtime, source_wav, source_mel, midi_path)
    raise ValueError(f"Unsupported DiffPitcher mode: {settings.mode}")


def _score_target_f0(
    settings: DiffPitcherSettings,
    runtime: DiffPitcherRuntime,
    source_wav: np.ndarray,
    source_mel: np.ndarray,
    midi_path: Path,
) -> np.ndarray:
    """Predict score-guided F0 from MIDI and source spectral features."""

    if runtime.pitchformer is None:
        raise RuntimeError("DiffPitcher PitchFormer model was not loaded.")
    midi = midi_f0_to_frames(midi_path, source_mel.shape[-1])
    with torch.inference_mode():
        source = torch.from_numpy(source_mel).float().unsqueeze(0).to(runtime.device)
        midi_tensor = torch.from_numpy(midi).float().unsqueeze(0).to(runtime.device)
        f0_pred = runtime.pitchformer(midi=midi_tensor, sp=source).cpu().numpy()[0]
    if settings.mask_with_source:
        source_f0 = fit_frames(estimate_f0_world(source_wav), len(f0_pred))
        f0_pred[source_f0 == 0] = 0
    f0_pred[f0_pred < DIFFPITCHER_F0_MIN] = 0
    f0_pred[f0_pred > DIFFPITCHER_F0_CLAMP_MAX] = DIFFPITCHER_F0_CLAMP_MAX
    return np.asarray(f0_pred, dtype=np.float32)


def _validate_required_guide(settings: DiffPitcherSettings) -> None:
    """Validate the mode-specific guide file before model loading."""

    if settings.mode == DIFFPITCHER_MODE_TEMPLATE:
        _require_user_file(settings.reference_audio, "reference vocal")
    elif settings.mode == DIFFPITCHER_MODE_SCORE:
        _require_user_file(settings.midi_path, "MIDI score")


def _require_user_file(path_value: str, label: str) -> Path:
    """Validate a user-supplied DiffPitcher guide file."""

    path = Path(str(path_value or "")).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Select a DiffPitcher {label} file before processing.")
    return path.resolve()


def _basename(path_value: str) -> str:
    """Return a JSON-safe basename for optional user guide files."""

    return Path(path_value).name if path_value else ""
