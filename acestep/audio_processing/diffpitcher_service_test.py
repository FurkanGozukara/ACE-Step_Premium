"""Tests for DiffPitcher service edge behavior."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from acestep.audio_processing.diffpitcher_audio_io import load_mono_24k
from acestep.audio_processing.diffpitcher_quality import build_pitch_guide_report
from acestep.audio_processing.diffpitcher_service import apply_diffpitcher
from acestep.audio_processing.diffpitcher_settings import DiffPitcherSettings


class DiffPitcherServiceTests(unittest.TestCase):
    """Verify DiffPitcher service validation and disabled behavior."""

    def test_disabled_diffpitcher_returns_input_without_loading_models(self) -> None:
        """Disabled DiffPitcher should be a cheap no-op."""

        audio = np.zeros((128, 2), dtype=np.float32)
        with patch(
            "acestep.audio_processing.diffpitcher_service.get_diffpitcher_runtime"
        ) as runtime:
            output, metadata = apply_diffpitcher(
                audio,
                24000,
                DiffPitcherSettings(enabled=False),
            )

        runtime.assert_not_called()
        self.assertIs(output, audio)
        self.assertFalse(metadata["applied"])

    def test_missing_template_reference_fails_before_loading_models(self) -> None:
        """Template mode should require a reference vocal before loading checkpoints."""

        audio = np.zeros((128, 2), dtype=np.float32)
        with patch(
            "acestep.audio_processing.diffpitcher_service.get_diffpitcher_runtime"
        ) as runtime:
            with self.assertRaisesRegex(FileNotFoundError, "reference vocal"):
                apply_diffpitcher(
                    audio,
                    24000,
                    DiffPitcherSettings(enabled=True, mode="template"),
                )

        runtime.assert_not_called()

    def test_video_reference_loader_extracts_audio_track(self) -> None:
        """Reference videos should load through media audio extraction."""

        stereo = np.stack(
            [
                np.linspace(0.0, 0.2, 4800, dtype=np.float32),
                np.linspace(0.2, 0.0, 4800, dtype=np.float32),
            ],
            axis=1,
        )
        with patch(
            "acestep.audio_processing.diffpitcher_audio_io.read_media_audio",
            return_value=(stereo, 48000),
        ) as read_media:
            audio = load_mono_24k("C:/music/reference.mp4")

        read_media.assert_called_once()
        self.assertEqual(2400, len(audio))
        self.assertTrue(np.isfinite(audio).all())

    def test_enabled_diffpitcher_passes_progress_callback_to_diffusion(self) -> None:
        """Service should forward UI progress callbacks into the diffusion loop."""

        audio = np.zeros((128, 1), dtype=np.float32)
        callback_calls: list[tuple[object, object]] = []

        def callback(progress_value=None, text=None) -> None:
            callback_calls.append((progress_value, text))

        runtime = types.SimpleNamespace(device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.wav"
            reference_path.write_bytes(b"fake")
            settings = DiffPitcherSettings(
                enabled=True,
                mode="template",
                reference_audio=str(reference_path),
            )

            with patch(
                "acestep.audio_processing.diffpitcher_service.select_diffpitcher_device",
                return_value=torch.device("cpu"),
            ), patch(
                "acestep.audio_processing.diffpitcher_service.get_diffpitcher_runtime",
                return_value=runtime,
            ), patch(
                "acestep.audio_processing.diffpitcher_service.to_mono_24k",
                return_value=np.zeros(128, dtype=np.float32),
            ), patch(
                "acestep.audio_processing.diffpitcher_service.world_mel_from_wav",
                return_value=np.zeros((100, 4), dtype=np.float32),
            ), patch(
                "acestep.audio_processing.diffpitcher_service._target_f0",
                return_value=np.full(4, 220.0, dtype=np.float32),
            ), patch(
                "acestep.audio_processing.diffpitcher_service.estimate_f0_world",
                return_value=np.full(4, 210.0, dtype=np.float32),
            ), patch(
                "acestep.audio_processing.diffpitcher_service.log_f0_bins",
                return_value=np.zeros(4, dtype=np.float32),
            ), patch(
                "acestep.audio_processing.diffpitcher_service.run_diffusion",
                return_value=torch.zeros((1, 100, 4)),
            ) as run_diffusion, patch(
                "acestep.audio_processing.diffpitcher_service.vocode",
                return_value=np.zeros(128, dtype=np.float32),
            ), patch(
                "acestep.audio_processing.diffpitcher_service.restore_output_shape",
                return_value=audio,
            ):
                output, metadata = apply_diffpitcher(
                    audio,
                    24000,
                    settings,
                    progress_callback=callback,
                )

        self.assertIs(output, audio)
        self.assertTrue(metadata["applied"])
        self.assertIs(callback, run_diffusion.call_args.kwargs["progress_callback"])

    def test_enabled_diffpitcher_skips_unsafe_template_guide(self) -> None:
        """Unsafe template guides should not run diffusion or replace the source."""

        audio = np.zeros((128, 1), dtype=np.float32)
        settings = DiffPitcherSettings(
            enabled=True,
            mode="template",
            reference_audio=__file__,
        )
        runtime = types.SimpleNamespace(device=torch.device("cpu"))
        messages: list[str] = []

        def callback(progress_value=None, text=None) -> None:
            messages.append(str(text))

        with patch(
            "acestep.audio_processing.diffpitcher_service.select_diffpitcher_device",
            return_value=torch.device("cpu"),
        ), patch(
            "acestep.audio_processing.diffpitcher_service.get_diffpitcher_runtime",
            return_value=runtime,
        ), patch(
            "acestep.audio_processing.diffpitcher_service.to_mono_24k",
            return_value=np.zeros(128, dtype=np.float32),
        ), patch(
            "acestep.audio_processing.diffpitcher_service.world_mel_from_wav",
            return_value=np.zeros((100, 6), dtype=np.float32),
        ), patch(
            "acestep.audio_processing.diffpitcher_service._target_f0",
            return_value=np.array([0.0, 0.0, 220.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ), patch(
            "acestep.audio_processing.diffpitcher_service.estimate_f0_world",
            return_value=np.full(6, 220.0, dtype=np.float32),
        ), patch(
            "acestep.audio_processing.diffpitcher_service.run_diffusion",
        ) as run_diffusion:
            output, metadata = apply_diffpitcher(
                audio,
                24000,
                settings,
                progress_callback=callback,
            )

        self.assertIs(output, audio)
        self.assertFalse(metadata["applied"])
        self.assertEqual("unsafe_template_guide", metadata["reason"])
        self.assertTrue(metadata["pitch_guide"]["unsafe"])
        self.assertTrue(any("skipped unsafe" in message for message in messages))
        run_diffusion.assert_not_called()

    def test_pitch_guide_report_marks_sparse_guide_unsafe(self) -> None:
        """Sparse voiced guide frames should be marked unsafe."""

        report = build_pitch_guide_report(
            np.array([0.0, 0.0, 220.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

        self.assertTrue(report["unsafe"])
        self.assertTrue(any("few voiced" in warning for warning in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
