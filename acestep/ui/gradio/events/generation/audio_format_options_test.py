"""Tests for Gradio audio-format option helpers."""

import unittest

from acestep.ui.gradio.events.generation.audio_format_options import (
    audio_file_extension,
    audio_format_label,
    mp3_controls_visible,
    normalize_extract_audio_format,
    output_audio_formats,
    primary_audio_format,
)


class AudioFormatOptionsTests(unittest.TestCase):
    """Verify generation output-format helper behavior."""

    def test_legacy_dual_format_defaults_to_mp3(self):
        """Old FLAC+MP3 values should normalize to the new MP3 default."""

        self.assertEqual(output_audio_formats("flac_mp3"), ["mp3"])
        self.assertEqual(primary_audio_format("flac_mp3"), "mp3")
        self.assertTrue(mp3_controls_visible("flac_mp3"))
        self.assertEqual(audio_format_label("flac_mp3"), "MP3")

    def test_wav_uses_wav_extension(self):
        """WAV output should use a .wav file extension."""

        self.assertEqual(audio_file_extension("wav"), "wav")

    def test_extract_audio_format_defaults_to_mp3(self):
        """Extract-specific output format should support wav/mp3 and default to MP3."""

        self.assertEqual(normalize_extract_audio_format(" WAV "), "wav")
        self.assertEqual(normalize_extract_audio_format("invalid"), "mp3")


if __name__ == "__main__":
    unittest.main()
