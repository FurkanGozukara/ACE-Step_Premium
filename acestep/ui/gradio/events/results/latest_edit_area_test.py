"""Tests for latest edited-area comparison clip creation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.ui.gradio.events.results.latest_edit_area import create_latest_edit_area_clips


class LatestEditAreaTests(unittest.TestCase):
    """Verify edited-area clip metadata and trimming decisions."""

    def test_repaint_uses_selected_range(self) -> None:
        """Repaint clips should use the requested source start/end range."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = _touch(root / "generated.flac")
            source = _touch(root / "source.wav")
            seen_segments = []

            def fake_extract(_source, target, segment, **_kwargs):
                seen_segments.append(segment)
                return str(target).replace("\\", "/")

            with patch(
                "acestep.ui.gradio.events.results.latest_edit_area."
                "_extract_audio_segment",
                side_effect=fake_extract,
            ):
                result = create_latest_edit_area_clips(
                    task_type="repaint",
                    generated_audio_path=str(generated),
                    source_audio_path=str(source),
                    run_dir=root,
                    key="sample",
                    repainting_start=10.0,
                    repainting_end=15.0,
                )

        self.assertTrue(result["applied"])
        self.assertEqual(seen_segments, [(10.0, 5.0), (10.0, 5.0)])
        self.assertTrue(result["generated_area_path"].endswith("_latest_repainted_area.wav"))
        self.assertTrue(
            result["original_area_path"].endswith("_latest_repainted_area_original.wav")
        )

    def test_mp3_output_uses_mp3_clip_extension(self) -> None:
        """Selected MP3 output should create MP3 edited-area clips."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = _touch(root / "generated.mp3")
            source = _touch(root / "source.mp3")

            def fake_extract(_source, target, segment, **_kwargs):
                return str(target).replace("\\", "/")

            with patch(
                "acestep.ui.gradio.events.results.latest_edit_area."
                "_extract_audio_segment",
                side_effect=fake_extract,
            ):
                result = create_latest_edit_area_clips(
                    task_type="repaint",
                    generated_audio_path=str(generated),
                    source_audio_path=str(source),
                    run_dir=root,
                    key="sample",
                    repainting_start=10.0,
                    repainting_end=15.0,
                    output_format="mp3",
                )

        self.assertTrue(result["applied"])
        self.assertTrue(result["generated_area_path"].endswith("_latest_repainted_area.mp3"))
        self.assertTrue(
            result["original_area_path"].endswith("_latest_repainted_area_original.mp3")
        )

    def test_remix_uses_resolved_whole_source_segment(self) -> None:
        """Remix clips should compare the whole resolved Remix source segment."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = _touch(root / "generated.flac")
            source = _touch(root / "source.wav")
            seen_segments = []

            def fake_extract(_source, target, segment, **_kwargs):
                seen_segments.append(segment)
                return str(target).replace("\\", "/")

            with patch(
                "acestep.ui.gradio.events.results.latest_edit_area."
                "_extract_audio_segment",
                side_effect=fake_extract,
            ):
                result = create_latest_edit_area_clips(
                    task_type="cover",
                    generated_audio_path=str(generated),
                    source_audio_path=str(source),
                    run_dir=root,
                    key="sample",
                    repainting_start=40.0,
                    repainting_end=50.0,
                )

        self.assertTrue(result["applied"])
        self.assertEqual(seen_segments, [(0.0, None), (0.0, None)])
        self.assertTrue(result["generated_area_path"].endswith("_latest_remixed_area.wav"))
        self.assertTrue(
            result["original_area_path"].endswith("_latest_remixed_area_original.wav")
        )

    def test_lego_uses_lego_area_filename(self) -> None:
        """Lego clips should use Lego-specific names for inline preview labels."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = _touch(root / "generated.flac")
            source = _touch(root / "source.wav")

            def fake_extract(_source, target, segment, **_kwargs):
                return str(target).replace("\\", "/")

            with patch(
                "acestep.ui.gradio.events.results.latest_edit_area."
                "_extract_audio_segment",
                side_effect=fake_extract,
            ):
                result = create_latest_edit_area_clips(
                    task_type="lego",
                    generated_audio_path=str(generated),
                    source_audio_path=str(source),
                    run_dir=root,
                    key="sample",
                    repainting_start=10.0,
                    repainting_end=15.0,
                )

        self.assertTrue(result["applied"])
        self.assertTrue(result["generated_area_path"].endswith("_latest_lego_area.wav"))
        self.assertTrue(
            result["original_area_path"].endswith("_latest_lego_area_original.wav")
        )

    def test_invalid_bounded_range_hides_clips(self) -> None:
        """Invalid Repaint/Lego/Complete ranges should not create clips."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = _touch(root / "generated.flac")
            source = _touch(root / "source.wav")

            result = create_latest_edit_area_clips(
                task_type="lego",
                generated_audio_path=str(generated),
                source_audio_path=str(source),
                run_dir=root,
                key="sample",
                repainting_start=15.0,
                repainting_end=10.0,
            )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "invalid_range")

    def test_custom_mode_does_not_create_clips(self) -> None:
        """Non-source edit tasks should leave the edited-area row hidden."""

        result = create_latest_edit_area_clips(
            task_type="text2music",
            generated_audio_path="generated.flac",
            source_audio_path="source.wav",
            run_dir=".",
            key="sample",
            repainting_start=0.0,
            repainting_end=-1,
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "unsupported_task")


def _touch(path: Path) -> Path:
    """Create a tiny placeholder file for path-existence checks."""

    path.write_bytes(b"fake")
    return path


if __name__ == "__main__":
    unittest.main()
