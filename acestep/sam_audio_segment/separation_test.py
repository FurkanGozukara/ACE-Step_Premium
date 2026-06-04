"""Tests for SAM-Audio separation call arguments."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from acestep.sam_audio_segment.separation import SamAudioSeparator
from acestep.sam_audio_segment.settings import SamAudioSettings


class _FakeBatch:
    """Minimal processor batch used by separator tests."""

    def __init__(self) -> None:
        self.audios = torch.zeros(1, 8)

    def to(self, _device: torch.device) -> "_FakeBatch":
        """Return the batch after the requested device move."""

        return self


class _FakeProcessor:
    """Record processor inputs and return a fake SAM batch."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        """Return a fake batch while recording preprocessing arguments."""

        self.calls.append(kwargs)
        return _FakeBatch()


class _FakeModel:
    """Record the final separate call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def separate(self, batch, **kwargs):
        """Return a minimal separation result."""

        self.calls.append({"batch": batch, **kwargs})
        return SimpleNamespace(target=[torch.zeros(8)], residual=[torch.zeros(8)])


class TestSamAudioSeparator(unittest.TestCase):
    """Verify separator options forwarded to the official SAM-Audio model."""

    def test_predict_spans_and_reranking_candidates_are_forwarded(self) -> None:
        """The UI settings should become official ``separate`` keyword arguments."""

        model = _FakeModel()
        processor = _FakeProcessor()
        separator = SamAudioSeparator(
            model=model,
            processor=processor,
            settings=SamAudioSettings(predict_spans=True, reranking_candidates=8),
            device=torch.device("cpu"),
            dtype=torch.float32,
            sample_rate=48000,
        )

        separator.separate_audio(
            torch.zeros(1, 48000),
            description="vocals",
            anchors=None,
            masked_videos=None,
        )

        self.assertEqual("vocals", processor.calls[0]["descriptions"][0])
        self.assertIsNone(processor.calls[0]["anchors"])
        self.assertTrue(model.calls[0]["predict_spans"])
        self.assertEqual(8, model.calls[0]["reranking_candidates"])


if __name__ == "__main__":
    unittest.main()
