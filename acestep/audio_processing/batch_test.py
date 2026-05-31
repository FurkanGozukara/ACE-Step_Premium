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


if __name__ == "__main__":
    unittest.main()
