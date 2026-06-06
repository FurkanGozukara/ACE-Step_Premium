"""Pitch-guide quality diagnostics for DiffPitcher inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from .process_logging import ProcessCallback, emit_process_message

MIN_VOICED_PERCENT = 20.0
MIN_COMMON_VOICED_PERCENT = 35.0
LARGE_SHIFT_CENTS = 1200.0
MAX_LARGE_SHIFT_PERCENT = 35.0
LOW_MEDIAN_F0_HZ = 95.0


def build_pitch_guide_report(
    target_f0: np.ndarray,
    *,
    source_f0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return JSON-safe diagnostics for a DiffPitcher pitch guide."""

    target = np.asarray(target_f0, dtype=np.float32)
    report = _target_report(target)
    warnings: list[str] = list(report["warnings"])
    unsafe = bool(report["unsafe"])
    if source_f0 is not None and len(target) > 0:
        source_report = _source_alignment_report(np.asarray(source_f0, dtype=np.float32), target)
        report.update(source_report)
        warnings.extend(source_report["warnings"])
        unsafe = unsafe or bool(source_report["unsafe"])
    report["warnings"] = warnings
    report["unsafe"] = unsafe
    return report


def emit_pitch_guide_report(
    report: dict[str, Any],
    progress_callback: ProcessCallback | None,
) -> None:
    """Emit pitch-guide diagnostics to terminal and optional UI callback."""

    if int(report.get("frames") or 0) <= 0:
        return
    if float(report.get("voiced_percent") or 0.0) <= 0.0:
        emit_process_message(
            progress_callback,
            "DiffPitcher pitch guide: no voiced pitch frames detected",
        )
    else:
        emit_process_message(
            progress_callback,
            (
                "DiffPitcher pitch guide: "
                f"{float(report['voiced_percent']):.1f}% voiced, "
                f"median {float(report['median_f0_hz']):.1f} Hz"
            ),
        )
    common = report.get("common_voiced_percent")
    if common is not None:
        emit_process_message(
            progress_callback,
            (
                "DiffPitcher pitch guide/source overlap: "
                f"{float(common):.1f}% common voiced frames"
            ),
        )
    for warning in report.get("warnings") or []:
        emit_process_message(progress_callback, f"DiffPitcher warning: {warning}")


def _target_report(target: np.ndarray) -> dict[str, Any]:
    """Return voiced-frame diagnostics for the target guide alone."""

    voiced = target[target > 0]
    report: dict[str, Any] = {
        "frames": int(len(target)),
        "voiced_percent": 0.0,
        "median_f0_hz": None,
        "warnings": [],
        "unsafe": False,
    }
    if len(target) == 0:
        return report
    if len(voiced) == 0:
        report["warnings"].append("pitch guide has no voiced frames")
        report["unsafe"] = True
        return report
    voiced_percent = 100.0 * len(voiced) / len(target)
    median_f0 = float(np.median(voiced))
    report["voiced_percent"] = float(voiced_percent)
    report["median_f0_hz"] = median_f0
    if voiced_percent < MIN_VOICED_PERCENT:
        report["warnings"].append(
            "pitch guide has very few voiced frames. Use a clean isolated vocal or "
            "a matching MIDI score."
        )
        report["unsafe"] = True
    if median_f0 < LOW_MEDIAN_F0_HZ:
        report["warnings"].append(
            "pitch guide is very low. Use an isolated reference vocal, not a full "
            "mixed song or instrumental."
        )
    return report


def _source_alignment_report(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Return source-vs-guide compatibility diagnostics."""

    frame_count = min(len(source), len(target))
    if frame_count <= 0:
        return {"warnings": [], "unsafe": False}
    source = source[:frame_count]
    target = target[:frame_count]
    common = (source > 0) & (target > 0)
    common_percent = 100.0 * int(np.count_nonzero(common)) / frame_count
    report: dict[str, Any] = {
        "common_voiced_percent": float(common_percent),
        "warnings": [],
        "unsafe": False,
    }
    if not np.any(common):
        report["warnings"].append(
            "template guide has no voiced overlap with the source. Use the same "
            "song/timing or MIDI score mode."
        )
        report["unsafe"] = True
        return report
    cents = 1200.0 * np.log2(target[common] / source[common])
    large_shift_percent = 100.0 * float(np.mean(np.abs(cents) > LARGE_SHIFT_CENTS))
    report["median_shift_cents"] = float(np.median(cents))
    report["p10_shift_cents"] = float(np.percentile(cents, 10))
    report["p90_shift_cents"] = float(np.percentile(cents, 90))
    report["large_shift_percent"] = float(large_shift_percent)
    if common_percent < MIN_COMMON_VOICED_PERCENT:
        report["warnings"].append(
            "template guide overlaps the source in too few voiced frames. Use a "
            "reference vocal from the same song/phrasing, or use MIDI score mode."
        )
        report["unsafe"] = True
    if large_shift_percent > MAX_LARGE_SHIFT_PERCENT:
        report["warnings"].append(
            "template guide differs from the source by octave-sized jumps across "
            "many frames. This usually means the reference is a different song."
        )
        report["unsafe"] = True
    return report
