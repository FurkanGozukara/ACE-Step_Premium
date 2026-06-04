"""Tests for generated-song SAM-Audio post-processing."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from acestep.sam_audio_segment.generated_postprocess import postprocess_generated_sample
from acestep.sam_audio_segment.settings import SamAudioSettings


class _Artifact:
    """Small SAM-Audio artifact object for post-processing tests."""

    def __init__(self) -> None:
        self.target_audio_path = "target.wav"
        self.residual_audio_path = "residual.wav"
        self.target_video_path = None
        self.metadata_path = "metadata.json"

    def file_list(self) -> list[str]:
        """Return generated artifact file paths."""

        return [self.target_audio_path, self.residual_audio_path, self.metadata_path]


class GeneratedSamAudioPostprocessTests(unittest.TestCase):
    """Verify generated-song SAM-Audio same-process routing."""

    def test_same_process_postprocess_uses_cached_service(self) -> None:
        """Auto post-processing should reuse the in-process model cache."""

        service = MagicMock()
        service.process_file.return_value = _Artifact()
        settings = SamAudioSettings(auto_postprocess=True, subprocess=False)

        @contextmanager
        def _cached_service(*args, **kwargs):
            """Yield a fake cached SAM-Audio service."""

            self.assertEqual((settings,), args)
            self.assertEqual({}, kwargs)
            yield service

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "acestep.sam_audio_segment.generated_postprocess.cached_sam_audio_service",
            new=_cached_service,
        ):
            result = postprocess_generated_sample(
                source_audio_path="C:/music/generated.wav",
                run_dir=Path(temp_dir),
                key="sample_001",
                settings=settings,
            )

        self.assertTrue(result["applied"])
        self.assertEqual("target.wav", result["target_audio_path"])
        service.unload.assert_not_called()
        _, args, kwargs = service.process_file.mock_calls[0]
        self.assertEqual("C:/music/generated.wav", args[0])
        self.assertEqual("sample_001_sam_audio", kwargs["output_stem"])


if __name__ == "__main__":
    unittest.main()
