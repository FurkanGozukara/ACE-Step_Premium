"""Tests for Audio Processing subprocess UI wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.audio_processing.settings import UI_SETTING_KEYS
from acestep.ui.gradio.events.wiring.audio_processing_cancel_actions import (
    AUDIO_PROCESSING_CANCEL_REQUESTED_STATUS,
    AUDIO_PROCESSING_IN_PROCESS_STATUS,
    request_audio_processing_cancel_from_ui,
)
from acestep.ui.gradio.events.wiring.audio_processing_single_file_subprocess import (
    process_single_file_subprocess,
)
from acestep.ui.gradio.events.wiring.audio_processing_wiring import (
    _process_single_file_event,
)


class AudioProcessingSubprocessWiringTests(unittest.TestCase):
    """Verify Audio Processing subprocess UI helpers."""

    def test_process_file_event_routes_to_subprocess_when_enabled(self) -> None:
        """The Process File event should use the subprocess path when checked."""

        expected = ("audio.wav", {"visible": False}, "figure", {"visible": True}, "done")
        with patch(
            "acestep.ui.gradio.events.wiring.audio_processing_wiring."
            "process_single_file_subprocess",
            return_value=expected,
        ) as subprocess_process:
            result = _process_single_file_event(
                "clip.mp4",
                None,
                True,
                *_settings_values(),
            )

        self.assertEqual(expected, result)
        subprocess_process.assert_called_once()

    def test_subprocess_process_file_maps_media_result_outputs(self) -> None:
        """Subprocess media results should map to the standard Gradio outputs."""

        with (
            patch(
                "acestep.ui.gradio.events.wiring.audio_processing_single_file_subprocess."
                "create_audio_processing_run_dir",
                return_value="run",
            ),
            patch(
                "acestep.ui.gradio.events.wiring.audio_processing_single_file_subprocess."
                "run_audio_processing_subprocess",
                return_value={
                    "kind": "media",
                    "audio_path": "processed.wav",
                    "video_path": "processed.mp4",
                    "files": ["processed.wav", "processed.mp4", "metadata.json"],
                    "figure": "figure",
                    "status_markdown": "metrics",
                },
            ) as run_subprocess,
        ):
            audio, video, figure, files, status = process_single_file_subprocess(
                "clip.mp4",
                None,
                *_settings_values(),
            )

        self.assertEqual("processed.wav", audio)
        self.assertEqual("processed.mp4", video["value"])
        self.assertTrue(video["visible"])
        self.assertEqual("figure", figure)
        self.assertEqual(["processed.wav", "processed.mp4", "metadata.json"], files["value"])
        self.assertIn("metrics", status)
        run_subprocess.assert_called_once()

    def test_cancel_processing_requires_subprocess_mode(self) -> None:
        """Cancel should report why it cannot stop in-process work."""

        status = request_audio_processing_cancel_from_ui(True, False)

        self.assertEqual(AUDIO_PROCESSING_IN_PROCESS_STATUS, status)

    def test_cancel_processing_requests_active_subprocess_stop(self) -> None:
        """Cancel should request Audio Processing subprocess termination."""

        with patch(
            "acestep.ui.gradio.events.wiring.audio_processing_cancel_actions."
            "request_audio_processing_cancel",
            return_value=True,
        ) as request_cancel:
            status = request_audio_processing_cancel_from_ui(True, True)

        self.assertEqual(AUDIO_PROCESSING_CANCEL_REQUESTED_STATUS, status)
        request_cancel.assert_called_once()


def _settings_values() -> list[object]:
    """Return Audio Processing UI values for subprocess wiring tests."""

    values: list[object] = []
    for key in UI_SETTING_KEYS:
        if key.endswith("_enabled"):
            values.append(True)
        elif key == "ap_output_format":
            values.append("wav")
        else:
            values.append(None)
    return values


if __name__ == "__main__":
    unittest.main()
