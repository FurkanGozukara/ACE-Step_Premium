"""Tests for tensor preprocessing cancellation."""

from __future__ import annotations

import tempfile
import unittest

from acestep.training.dataset_builder_modules.models import AudioSample
from acestep.training.dataset_builder_modules.preprocess import PreprocessMixin


class _Metadata:
    """Minimal dataset metadata for preprocess tests."""

    genre_ratio = 0

    def to_dict(self) -> dict[str, object]:
        """Return serializable metadata."""

        return {"name": "cancel-test"}


class _Builder(PreprocessMixin):
    """Minimal builder exposing samples and metadata."""

    def __init__(self) -> None:
        """Build one labeled sample."""

        self.metadata = _Metadata()
        self.samples = [
            AudioSample(
                audio_path="sample.wav",
                filename="sample.wav",
                caption="caption",
                labeled=True,
            )
        ]


class _DitHandler:
    """Minimal initialized DiT handler for early-cancel checks."""

    model = object()
    vae = object()
    text_encoder = object()
    text_tokenizer = object()
    silence_latent = object()
    device = "cpu"
    dtype = "float32"


class PreprocessCancelTests(unittest.TestCase):
    """Verify cancel callbacks stop preprocessing before model work."""

    def test_cancel_callback_stops_before_first_sample(self) -> None:
        """A pending cancel request should return a cancelled status."""

        with tempfile.TemporaryDirectory() as tmpdir:
            output_paths, status = _Builder().preprocess_to_tensors(
                dit_handler=_DitHandler(),
                output_dir=tmpdir,
                cancel_callback=lambda: True,
            )

        self.assertEqual([], output_paths)
        self.assertIn("Tensor preprocess cancelled after 0/1 samples; left 1", status)


if __name__ == "__main__":
    unittest.main()
