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
    """Verify dual-format output helper behavior."""

    def test_dual_format_saves_flac_and_mp3(self):
        """Default dual output should write FLAC first and MP3 second."""

        self.assertEqual(output_audio_formats("flac_mp3"), ["flac", "mp3"])
        self.assertEqual(primary_audio_format("flac_mp3"), "flac")
        self.assertTrue(mp3_controls_visible("flac_mp3"))
        self.assertEqual(audio_format_label("flac_mp3"), "FLAC + MP3")

    def test_wav32_uses_wav_extension(self):
        """WAV 32-bit should still use a .wav file extension."""

        self.assertEqual(audio_file_extension("wav32"), "wav")

    def test_extract_audio_format_defaults_to_mp3(self):
        """Extract-specific output format should support wav/mp3/flac and default to MP3."""

        self.assertEqual(normalize_extract_audio_format(" FLAC "), "flac")
        self.assertEqual(normalize_extract_audio_format("invalid"), "mp3")


if __name__ == "__main__":
    unittest.main()
