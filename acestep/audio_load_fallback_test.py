"""Tests for TorchCodec-safe audio loading fallback."""

import unittest
from unittest.mock import patch

import numpy as np
from loguru import logger

from acestep.audio_load_fallback import load_audio_with_torchaudio_fallback


class AudioLoadFallbackTests(unittest.TestCase):
    def test_falls_back_to_media_decoder_and_logs(self):
        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}", level="WARNING")
        decoded = np.zeros((8, 2), dtype=np.float32)
        try:
            with (
                patch("torchaudio.load", side_effect=RuntimeError("codec missing")),
                patch(
                    "acestep.audio_processing.media_io.read_media_audio",
                    return_value=(decoded, 44100),
                ),
            ):
                waveform, sample_rate = load_audio_with_torchaudio_fallback(
                    "song.m4a",
                    context="test loader",
                )
        finally:
            logger.remove(sink_id)

        self.assertEqual(sample_rate, 44100)
        self.assertEqual(tuple(waveform.shape), (2, 8))
        joined = "\n".join(messages)
        self.assertIn("torchaudio.load failed", joined)
        self.assertIn("FFmpeg/soundfile fallback succeeded", joined)

    def test_logs_both_errors_when_all_loaders_fail(self):
        messages: list[str] = []
        sink_id = logger.add(messages.append, format="{message}", level="WARNING")
        try:
            with (
                patch("torchaudio.load", side_effect=RuntimeError("codec missing")),
                patch(
                    "acestep.audio_processing.media_io.read_media_audio",
                    side_effect=RuntimeError("ffmpeg missing"),
                ),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    load_audio_with_torchaudio_fallback(
                        "bad.m4a",
                        context="test loader",
                    )
        finally:
            logger.remove(sink_id)

        self.assertIn("codec missing", str(ctx.exception))
        self.assertIn("ffmpeg missing", str(ctx.exception))
        joined = "\n".join(messages)
        self.assertIn("torchaudio.load failed", joined)
        self.assertIn("Audio load failed", joined)
        self.assertIn("codec missing", joined)
        self.assertIn("ffmpeg missing", joined)


if __name__ == "__main__":
    unittest.main()
