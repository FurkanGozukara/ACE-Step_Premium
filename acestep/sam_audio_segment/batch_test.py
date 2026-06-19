"""Tests for SAM-Audio batch folder processing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from acestep.sam_audio_segment.batch import run_batch_sam_audio
from acestep.sam_audio_segment.settings import SamAudioSettings


class TestSamAudioBatch(unittest.TestCase):
    """Verify batch processing behavior without loading the model."""

    def test_batch_uses_input_filename_as_output_stem(self):
        """Batch outputs preserve the source filename stem."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "My Song.wav"
            source.write_bytes(b"placeholder")
            service = MagicMock()
            artifact = MagicMock()
            artifact.file_list.return_value = [str(output_dir / "My Song.wav")]
            service.process_file.return_value = artifact

            with patch("acestep.sam_audio_segment.batch.SamAudioService", return_value=service):
                list(
                    run_batch_sam_audio(
                        str(input_dir),
                        str(output_dir),
                        False,
                        SamAudioSettings(),
                    )
                )

            _, _, kwargs = service.process_file.mock_calls[0]
            self.assertEqual("My Song", kwargs["output_stem"])
            service.unload.assert_called_once()

    def test_batch_segment_processes_each_quick_prompt_with_suffix(self):
        """Batch Segment runs each selected Quick Prompt and suffixes output stems."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "My Song.wav"
            source.write_bytes(b"placeholder")
            service = MagicMock()
            first_artifact = MagicMock()
            first_artifact.file_list.return_value = [str(output_dir / "My Song_vocals.wav")]
            second_artifact = MagicMock()
            second_artifact.file_list.return_value = [str(output_dir / "My Song_guitar.wav")]
            service.process_file.side_effect = [first_artifact, second_artifact]

            with patch("acestep.sam_audio_segment.batch.SamAudioService", return_value=service):
                list(
                    run_batch_sam_audio(
                        str(input_dir),
                        str(output_dir),
                        False,
                        SamAudioSettings(
                            batch_segment=False,
                            prompt_preset=("vocals", "guitar"),
                        ),
                    )
                )

            stems = [call.kwargs["output_stem"] for call in service.process_file.mock_calls]
            self.assertEqual(["My Song_vocals", "My Song_guitar"], stems)
            self.assertEqual("guitar", service.settings.custom_prompt)
            service.unload.assert_called_once()

    def test_batch_save_output_only_writes_direct_files_without_manifest(self):
        """Output-only batch mode should not create sidecar or manifest outputs."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "My Song.wav"
            source.write_bytes(b"placeholder")
            service = MagicMock()
            artifact = MagicMock()
            artifact.file_list.return_value = [str(output_dir / "My Song.wav")]
            service.process_file.return_value = artifact

            with patch("acestep.sam_audio_segment.batch.SamAudioService", return_value=service):
                updates = list(
                    run_batch_sam_audio(
                        str(input_dir),
                        str(output_dir),
                        False,
                        SamAudioSettings(
                            batch_save_output_only=True,
                            output_format="wav",
                        ),
                    )
                )

            call = service.process_file.mock_calls[0]
            self.assertEqual(output_dir.resolve(), Path(call.args[1]))
            self.assertEqual("My Song", call.kwargs["output_stem"])
            self.assertTrue(call.kwargs["output_only"])
            self.assertNotIn("sam_audio_batch_manifest.json", "\n".join(updates[-1][1]))
            self.assertFalse(any(output_dir.glob("sam_audio_*")))

    def test_batch_save_output_only_renames_existing_file_by_default(self):
        """Output-only batch mode should keep existing target files unless overwrite is on."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "My Song.wav"
            source.write_bytes(b"placeholder")
            (output_dir / "My Song.wav").write_bytes(b"existing")
            service = MagicMock()
            artifact = MagicMock()
            artifact.file_list.return_value = [str(output_dir / "My Song_extract1.wav")]
            service.process_file.return_value = artifact

            with patch("acestep.sam_audio_segment.batch.SamAudioService", return_value=service):
                list(
                    run_batch_sam_audio(
                        str(input_dir),
                        str(output_dir),
                        False,
                        SamAudioSettings(
                            batch_save_output_only=True,
                            output_format="wav",
                        ),
                    )
                )

            call = service.process_file.mock_calls[0]
            self.assertEqual("My Song_extract1", call.kwargs["output_stem"])
            self.assertTrue(call.kwargs["output_only"])

    def test_batch_save_output_only_overwrites_when_enabled(self):
        """Output-only batch mode should reuse the target stem when overwrite is on."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            source = input_dir / "My Song.wav"
            source.write_bytes(b"placeholder")
            (output_dir / "My Song.wav").write_bytes(b"existing")
            service = MagicMock()
            artifact = MagicMock()
            artifact.file_list.return_value = [str(output_dir / "My Song.wav")]
            service.process_file.return_value = artifact

            with patch("acestep.sam_audio_segment.batch.SamAudioService", return_value=service):
                list(
                    run_batch_sam_audio(
                        str(input_dir),
                        str(output_dir),
                        False,
                        SamAudioSettings(
                            batch_save_output_only=True,
                            batch_overwrite_existing=True,
                            output_format="wav",
                        ),
                    )
                )

            call = service.process_file.mock_calls[0]
            self.assertEqual("My Song", call.kwargs["output_stem"])
            self.assertTrue(call.kwargs["output_only"])


if __name__ == "__main__":
    unittest.main()
