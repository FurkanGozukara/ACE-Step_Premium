"""Live Gradio integration tests for media upload previews."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gradio_client import Client

from acestep.ui.gradio.events.wiring import media_upload_preview
from acestep.ui.gradio.events.wiring.media_upload_preview_gradio_support import (
    assert_all_videos_are_capped,
    assert_audio_purpose_video_preview,
    assert_media_upload_video_preview,
    assert_preview_routes_bypass_queue,
    assert_source_audio_purpose_video_preview,
    assert_video_preview,
    build_preview_test_app,
    free_port,
    write_wav,
)


class LiveGradioMediaPreviewTests(unittest.TestCase):
    """Verify media preview behavior through an actual Gradio server."""

    def test_live_gradio_preview_routes_cover_video_cap_and_upload_outputs(self) -> None:
        """All video-capable uploads should expose expected preview outputs."""

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_path = write_wav(root / "preview.wav")
            old_video_path = root / "old_clip.mp4"
            old_video_path.write_bytes(b"old fake mp4 payload")
            video_path = root / "clip.mp4"
            video_path.write_bytes(b"fake mp4 payload")

            with patch.object(
                media_upload_preview,
                "extract_audio_preview",
                return_value=str(audio_path),
            ):
                demo = build_preview_test_app()
                assert_all_videos_are_capped(demo)
                assert_preview_routes_bypass_queue(demo)
                source_preview = _component_props(demo, "audio", "Source Audio Preview")
                self.assertTrue(source_preview.get("interactive"))
                self.assertTrue(source_preview.get("editable"))
                self.assertEqual(source_preview.get("sources"), ["upload"])
                demo.queue(default_concurrency_limit=1)
                port = free_port()
                demo.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    prevent_thread_lock=True,
                    quiet=True,
                )
                try:
                    client = Client(f"http://127.0.0.1:{port}")
                    assert_source_audio_purpose_video_preview(
                        client,
                        video_path,
                        "/generation_source_preview",
                    )
                    assert_audio_purpose_video_preview(
                        client,
                        video_path,
                        "/generation_reference_preview",
                    )
                    assert_audio_purpose_video_preview(
                        client,
                        video_path,
                        "/generation_lm_codes_preview",
                    )
                    assert_video_preview(client, video_path, "/sam_mask_video_preview")
                    assert_media_upload_video_preview(client, video_path, "/sam_single_preview")
                    assert_media_upload_video_preview(client, video_path, "/ap_single_preview")
                    assert_source_audio_purpose_video_preview(
                        client,
                        [audio_path, video_path],
                        "/generation_source_preview",
                    )
                    assert_video_preview(
                        client,
                        [old_video_path, video_path],
                        "/sam_mask_video_preview",
                    )
                    assert_media_upload_video_preview(
                        client,
                        [audio_path, video_path],
                        "/ap_single_preview",
                    )
                finally:
                    demo.close()


def _component_props(demo, component_type: str, label: str) -> dict:
    """Return component props from the live Gradio config."""

    for component in demo.config["components"]:
        props = component.get("props") or {}
        if component.get("type") == component_type and props.get("label") == label:
            return props
    raise AssertionError(f"Could not find {component_type} component {label!r}")


if __name__ == "__main__":
    unittest.main()
