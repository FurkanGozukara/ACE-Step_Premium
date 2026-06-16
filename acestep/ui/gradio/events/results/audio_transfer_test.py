"""Tests for result-audio transfer helpers."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from acestep.ui.gradio.events.results.audio_transfer import (
    convert_result_audio_to_codes,
    send_audio_to_repaint,
)


class SendAudioToRepaintTests(unittest.TestCase):
    """Verify generated-source repaint transfer behavior."""

    def test_send_to_repaint_keeps_random_enabled_and_shows_generated_seed(self):
        """Generated-source repaint should show the source seed without fixing it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "generated.wav"
            audio_path.with_suffix(".json").write_text(
                json.dumps({"seed": 12345}),
                encoding="utf-8",
            )

            updates = send_audio_to_repaint(
                audio_file=str(audio_path),
                lm_metadata={"lyrics": "new words", "caption": "new caption"},
                current_lyrics="old words",
                current_caption="old caption",
                current_mode="Custom",
                current_seed="-1",
                llm_handler=None,
            )

        self.assertEqual(str(audio_path), updates[0])
        self.assertEqual(True, updates[6]["value"])
        self.assertEqual("12345", updates[7]["value"])

    def test_send_to_repaint_falls_back_to_current_seed(self):
        """Generated-source repaint should keep the visible seed if no sidecar exists."""
        updates = send_audio_to_repaint(
            audio_file="/tmp/generated.wav",
            lm_metadata={"lyrics": "new words", "caption": "new caption"},
            current_lyrics="old words",
            current_caption="old caption",
            current_mode="Custom",
            current_seed="67890",
            llm_handler=None,
        )

        self.assertEqual(True, updates[6]["value"])
        self.assertEqual("67890", updates[7]["value"])


class ConvertResultAudioToCodesTests(unittest.TestCase):
    """Verify generated audio code conversion can reuse sidecar metadata."""

    def test_convert_uses_generated_sidecar_codes_without_model(self):
        """A generated JSON sidecar avoids re-encoding audio through the VAE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "generated.wav"
            audio_path.with_suffix(".json").write_text(
                json.dumps({"audio_codes": "cached-codes"}),
                encoding="utf-8",
            )

            with patch("gradio.Info"):
                codes_update, accordion_update = convert_result_audio_to_codes(None, str(audio_path))

            self.assertEqual("cached-codes", codes_update["value"])
            self.assertTrue(accordion_update["open"])

    def test_convert_falls_back_to_handler_without_sidecar_codes(self):
        """Uploaded or uncached audio still uses the existing conversion path."""
        handler = MagicMock()
        handler.model = object()
        handler.convert_src_audio_to_codes.return_value = "fresh-codes"

        with tempfile.TemporaryDirectory() as tmpdir, patch("gradio.Info"):
            audio_path = Path(tmpdir) / "uploaded.wav"
            codes_update, _accordion_update = convert_result_audio_to_codes(handler, str(audio_path))

        self.assertEqual("fresh-codes", codes_update["value"])
        handler.convert_src_audio_to_codes.assert_called_once_with(str(audio_path))


if __name__ == "__main__":
    unittest.main()
