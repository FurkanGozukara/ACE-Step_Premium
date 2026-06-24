"""Tests for Remix full-song area splicing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from acestep.ui.gradio.events.results.remix_area_splice import save_remix_area_splice


class RemixAreaSpliceTests(unittest.TestCase):
    """Verify Remix range replacement semantics."""

    def test_bounded_range_hard_overwrites_source_with_generated_audio(self) -> None:
        """The selected range should be 100% Remix output, not a blend."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = _touch(root / "source.wav")
            generated_path = _touch(root / "generated.wav")
            source_audio = np.zeros((10, 2), dtype=np.float32)
            generated_audio = np.ones((10, 2), dtype=np.float32)
            saved = {}

            def fake_read(path):
                if Path(path).name == "source.wav":
                    return source_audio, 10
                return generated_audio, 10

            def fake_save_audio(**kwargs):
                saved.update(kwargs)
                return kwargs["output_path"]

            with patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_media_audio_duration_seconds",
                return_value=1.0,
            ), patch(
                "acestep.ui.gradio.events.results.remix_area_splice.read_media_audio",
                side_effect=fake_read,
            ):
                result = save_remix_area_splice(
                    task_type="cover",
                    generated_audio_path=str(generated_path),
                    source_audio_path=str(source_path),
                    run_dir=root,
                    key="sample",
                    repainting_start=0.2,
                    repainting_end=0.5,
                    output_format="wav",
                    save_audio_fn=fake_save_audio,
                )

        self.assertTrue(result["applied"])
        output = saved["audio_data"]
        np.testing.assert_array_equal(output[:, :2], np.zeros((2, 2), dtype=np.float32))
        np.testing.assert_array_equal(output[:, 2:5], np.ones((2, 3), dtype=np.float32))
        np.testing.assert_array_equal(output[:, 5:], np.zeros((2, 5), dtype=np.float32))

    def test_full_source_range_does_not_create_splice(self) -> None:
        """Default whole-source Remix should keep the normal generated output."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = _touch(root / "source.wav")
            generated_path = _touch(root / "generated.wav")

            with patch(
                "acestep.ui.gradio.events.results.remix_source_range."
                "_media_audio_duration_seconds",
                return_value=1.0,
            ):
                result = save_remix_area_splice(
                    task_type="cover",
                    generated_audio_path=str(generated_path),
                    source_audio_path=str(source_path),
                    run_dir=root,
                    key="sample",
                    repainting_start=0.0,
                    repainting_end=-1,
                )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "not_bounded_remix_range")


def _touch(path: Path) -> Path:
    """Create a tiny placeholder file for path-existence checks."""

    path.write_bytes(b"fake")
    return path


if __name__ == "__main__":
    unittest.main()
