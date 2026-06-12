"""Tests for OpenRouter request conversion helpers."""

import sys
import types
import unittest
from unittest.mock import patch

from acestep.openrouter_adapter import _to_generate_music_request
from acestep.openrouter_models import ChatCompletionRequest


class _FakeGenerateMusicRequest:
    """Small request container used to avoid importing the full API server."""

    def __init__(self, **kwargs):
        """Store generated request fields as attributes."""

        self.__dict__.update(kwargs)


class OpenRouterAdapterTests(unittest.TestCase):
    """Verify OpenRouter chat requests preserve ACE-Step task fields."""

    def _convert(self, req: ChatCompletionRequest):
        """Convert a request while faking the heavy API server import."""

        fake_api_server = types.ModuleType("acestep.api_server")
        fake_api_server.GenerateMusicRequest = _FakeGenerateMusicRequest
        with patch.dict(sys.modules, {"acestep.api_server": fake_api_server}):
            return _to_generate_music_request(
                req,
                prompt="prompt",
                lyrics="lyrics",
                sample_query=None,
                reference_audio_path=None,
                src_audio_path="source.wav",
            )

    def test_to_generate_music_request_forwards_track_name(self):
        """Extract/Lego track names should reach the shared API setup layer."""

        req = ChatCompletionRequest(
            model="acemusic/acestep-v15-xl-base",
            messages=[],
            task_type="extract",
            track_name="vocals",
        )

        out = self._convert(req)

        self.assertEqual("extract", out.task_type)
        self.assertEqual("vocals", out.track_name)

    def test_to_generate_music_request_forwards_track_classes(self):
        """Complete track classes should reach the shared API setup layer."""

        req = ChatCompletionRequest(
            model="acemusic/acestep-v15-xl-base",
            messages=[],
            task_type="complete",
            track_classes=["drums", "bass"],
        )

        out = self._convert(req)

        self.assertEqual("complete", out.task_type)
        self.assertEqual(["drums", "bass"], out.track_classes)


if __name__ == "__main__":
    unittest.main()
