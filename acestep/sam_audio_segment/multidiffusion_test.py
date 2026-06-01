"""Tests for SAM-Audio text-only multi-diffusion helpers."""

import unittest

import torch

from acestep.sam_audio_segment.multidiffusion_solver import (
    midpoint_schedule,
    solve_midpoint_multidiffusion,
)
from acestep.sam_audio_segment.multidiffusion_windows import (
    iter_latent_windows,
    latent_window_frames,
    slice_forward_args,
    soft_window_mask,
)
from acestep.sam_audio_segment.separation_multidiffusion import (
    should_use_multidiffusion_long_audio,
)
from acestep.sam_audio_segment.settings import (
    SAM_AUDIO_LONG_MODE_MULTIDIFFUSION,
    SamAudioSettings,
)


class ConstantVelocityModel:
    """Tiny model returning a constant vector field for solver tests."""

    def forward(self, noisy_audio, **_kwargs):
        """Return a constant velocity with the same shape as the noisy audio."""

        return torch.ones_like(noisy_audio)


class TestSamAudioMultiDiffusion(unittest.TestCase):
    """Verify multi-diffusion windowing, solver, and request gating."""

    def test_paper_window_seconds_map_to_latent_frames(self):
        """A 20s/5s paper setup maps to 500/125 latent frames."""

        window, overlap = latent_window_frames(
            sample_rate=48000,
            hop_length=1920,
            window_seconds=20.0,
            overlap_seconds=5.0,
        )

        self.assertEqual(500, window)
        self.assertEqual(125, overlap)

    def test_latent_windows_cover_full_sequence_with_tail_shift(self):
        """Windows should cover the full latent sequence without gaps."""

        windows = iter_latent_windows(total_frames=40, window_frames=10, overlap_frames=2)

        self.assertEqual((0, 10), (windows[0].start, windows[0].end))
        self.assertEqual((30, 40), (windows[-1].start, windows[-1].end))
        self.assertTrue(
            all(left.end > right.start for left, right in zip(windows, windows[1:]))
        )

    def test_soft_masks_keep_every_frame_weighted(self):
        """Merged triangular masks should leave no uncovered latent frames."""

        windows = iter_latent_windows(total_frames=40, window_frames=10, overlap_frames=2)
        weights = torch.zeros(40)
        for window in windows:
            weights[window.start : window.end] += soft_window_mask(
                window.length,
                2,
                start=window.start,
                end=window.end,
                total_frames=40,
            )

        self.assertTrue(torch.all(weights > 0.0))

    def test_slice_forward_args_crops_only_time_conditioning(self):
        """Only per-audio time axes should be cropped for a latent window."""

        forward_args = {
            "audio_features": torch.arange(2 * 8 * 3).reshape(2, 8, 3),
            "text_features": torch.zeros(2, 4, 5),
            "text_mask": torch.ones(2, 4, dtype=torch.bool),
            "masked_video_features": torch.zeros(2, 6, 8),
            "anchor_ids": torch.zeros(2, 2, dtype=torch.long),
            "anchor_alignment": torch.arange(16).reshape(2, 8),
            "audio_pad_mask": torch.ones(2, 8, dtype=torch.bool),
        }

        sliced = slice_forward_args(forward_args, 2, 6)

        self.assertEqual((2, 4, 3), tuple(sliced["audio_features"].shape))
        self.assertEqual((2, 6, 4), tuple(sliced["masked_video_features"].shape))
        self.assertIs(forward_args["text_features"], sliced["text_features"])
        self.assertTrue(
            torch.equal(
                forward_args["anchor_alignment"][:, 2:6],
                sliced["anchor_alignment"],
            )
        )

    def test_midpoint_solver_fuses_constant_vector_field(self):
        """The explicit midpoint solver should integrate a constant velocity."""

        state = torch.zeros(1, 6, 2)
        forward_args = {
            "audio_features": torch.zeros(1, 6, 2),
            "text_features": torch.zeros(1, 1, 2),
            "text_mask": torch.ones(1, 1, dtype=torch.bool),
            "masked_video_features": torch.zeros(1, 1, 6),
            "anchor_ids": torch.zeros(1, 2, dtype=torch.long),
            "anchor_alignment": torch.zeros(1, 6, dtype=torch.long),
            "audio_pad_mask": torch.ones(1, 6, dtype=torch.bool),
        }
        progress: list[tuple[int, int]] = []

        result = solve_midpoint_multidiffusion(
            ConstantVelocityModel(),
            state,
            forward_args,
            iter_latent_windows(6, 4, 2),
            overlap_frames=2,
            step_size=0.5,
            num_steps=2,
            progress_callback=lambda done, total: progress.append((done, total)),
            cancel_callback=None,
        )

        self.assertTrue(torch.allclose(torch.ones_like(result), result))
        self.assertEqual([(1, 2), (2, 2)], progress)

    def test_midpoint_schedule_rejects_non_divisible_steps(self):
        """Multi-diffusion should stay on fixed midpoint schedules."""

        with self.assertRaises(ValueError):
            midpoint_schedule({"method": "midpoint", "options": {"step_size": 0.3}})

    def test_long_audio_gate_accepts_text_only_request(self):
        """Text-only long audio with one candidate can use multi-diffusion."""

        settings = SamAudioSettings(
            long_audio_mode=SAM_AUDIO_LONG_MODE_MULTIDIFFUSION,
            chunk_seconds=2.0,
            reranking_candidates=1,
            predict_spans=False,
        )

        self.assertTrue(
            should_use_multidiffusion_long_audio(
                settings,
                torch.zeros(1, 30),
                sample_rate=10,
                masked_videos=None,
                anchors=None,
            )
        )

    def test_long_audio_gate_rejects_non_text_only_request(self):
        """Visual/span/reranked requests should not enter the text-only path."""

        settings = SamAudioSettings(
            prompt_mode="visual",
            long_audio_mode=SAM_AUDIO_LONG_MODE_MULTIDIFFUSION,
            chunk_seconds=2.0,
        )

        with self.assertRaises(ValueError):
            should_use_multidiffusion_long_audio(
                settings,
                torch.zeros(1, 30),
                sample_rate=10,
                masked_videos=[torch.zeros(1)],
                anchors=None,
            )


if __name__ == "__main__":
    unittest.main()
