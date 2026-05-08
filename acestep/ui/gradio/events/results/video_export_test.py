"""Tests for still-image MP4 export helpers."""

import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from acestep.ui.gradio.events.results.video_export import (
    create_still_image_video,
    resolve_video_dimensions,
)


class VideoExportTests(unittest.TestCase):
    """Verify image-ratio sizing and ffmpeg MP4 export."""

    def test_resolve_video_dimensions_preserves_aspect_ratio(self):
        """A square image should export as square inside the selected preset."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "cover.png"
            Image.new("RGB", (1000, 1000), "navy").save(image_path)

            self.assertEqual(resolve_video_dimensions(str(image_path), "1080p"), (1080, 1080))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for MP4 export")
    def test_create_still_image_video_writes_mp4(self):
        """ffmpeg should combine a still image and WAV audio into MP4."""

        import wave

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "cover.png"
            audio_path = root / "audio.wav"
            video_path = root / "video.mp4"
            Image.new("RGB", (320, 180), "navy").save(image_path)
            with wave.open(str(audio_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(48000)
                handle.writeframes(b"\x00\x00" * 4800)

            result = create_still_image_video(
                image_path=str(image_path),
                audio_path=str(audio_path),
                output_path=str(video_path),
                resolution="720p",
            )

            self.assertTrue(Path(result).is_file())
            self.assertGreater(Path(result).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
