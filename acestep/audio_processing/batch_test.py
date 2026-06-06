"""Tests for Audio Processing batch folder behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from acestep.audio_processing.batch import run_batch_audio_processing
from acestep.audio_processing.settings import AudioProcessingSettings


class TestAudioProcessingBatch(unittest.TestCase):
    """Verify filename behavior for batch audio processing."""

    def test_batch_uses_input_filename_as_output_stem(self):
        """Batch processing saves with the source filename stem."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "Song Name.wav"
            source.write_bytes(b"placeholder")
            artifact = MagicMock()
            artifact.file_list.return_value = [str(output_dir / "Song Name.wav")]
            artifact.processed_audio.lufs_before = -20.0
            artifact.processed_audio.lufs_after = -14.0

            with patch(
                "acestep.audio_processing.batch.process_media_file",
                return_value=artifact,
            ) as process_media_file:
                list(
                    run_batch_audio_processing(
                        str(input_dir),
                        str(output_dir),
                        False,
                        AudioProcessingSettings(),
                    )
                )

            _, _, kwargs = process_media_file.mock_calls[0]
            self.assertEqual("Song Name", kwargs["output_stem"])

    def test_batch_workflow_export_skips_media_processing(self):
        """Workflow export mode should only export Auto-Editor workflow files."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "Clip.mp4"
            source.write_bytes(b"placeholder")
            workflow_path = str(output_dir / "Clip.fcpxml")

            with (
                patch(
                    "acestep.audio_processing.batch.export_auto_editor_workflow",
                    return_value=workflow_path,
                ) as export_workflow,
                patch("acestep.audio_processing.batch.process_media_file") as process_media_file,
            ):
                updates = list(
                    run_batch_audio_processing(
                        str(input_dir),
                        str(output_dir),
                        False,
                        AudioProcessingSettings(workflow_export="resolve"),
                    )
                )

            export_workflow.assert_called_once()
            process_media_file.assert_not_called()
            self.assertIn(workflow_path, updates[-1][1])

    def test_batch_export_audio_only_status_and_settings(self):
        """Batch processing should pass export-audio-only settings to each file."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "Clip.mp4"
            source.write_bytes(b"placeholder")
            artifact = MagicMock()
            artifact.file_list.return_value = [str(output_dir / "Clip.wav")]
            artifact.processed_audio.lufs_before = -20.0
            artifact.processed_audio.lufs_after = -14.0

            with patch(
                "acestep.audio_processing.batch.process_media_file",
                return_value=artifact,
            ) as process_media_file:
                updates = list(
                    run_batch_audio_processing(
                        str(input_dir),
                        str(output_dir),
                        False,
                        AudioProcessingSettings(export_audio_only=True),
                    )
                )

            args, _kwargs = process_media_file.call_args
            self.assertTrue(args[2].export_audio_only)
            self.assertIn("Saving processed audio under", updates[1][0])


if __name__ == "__main__":
    unittest.main()
