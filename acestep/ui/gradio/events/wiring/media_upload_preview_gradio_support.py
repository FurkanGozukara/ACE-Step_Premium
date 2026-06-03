"""Support helpers for live Gradio media upload preview tests."""

from __future__ import annotations

import socket
import wave
from pathlib import Path

import gradio as gr
from gradio_client import Client, handle_file

from acestep.ui.gradio.events.wiring import media_upload_preview
from acestep.ui.gradio.events.wiring.generation_upload_handlers import (
    finalize_src_audio_upload,
    handle_src_audio_upload,
)
from acestep.ui.gradio.events.wiring.audio_processing_wiring import (
    _preview_upload as preview_audio_processing_upload,
)
from acestep.ui.gradio.events.wiring.sam_audio_action_helpers import preview_upload
from acestep.ui.gradio.interfaces.generation_tab_secondary_controls import (
    build_custom_mode_controls,
)
from acestep.ui.gradio.interfaces.generation_tab_source_controls import (
    build_source_track_and_code_controls,
)
from acestep.ui.gradio.pages.audio_processing_page import create_audio_processing_page
from acestep.ui.gradio.pages.sam_audio_page import create_sam_audio_page
from acestep.ui.gradio.pages.simple_create_page import create_simple_create_page


def build_preview_test_app() -> gr.Blocks:
    """Build a small live app from the real modified Gradio UI builders."""

    with gr.Blocks() as demo:
        generation = build_source_track_and_code_controls()
        custom = build_custom_mode_controls()
        source_mode = gr.State(value="Lego")
        source_audio_duration = gr.Number(value=-1, visible=False)
        create_simple_create_page()
        sam_page = create_sam_audio_page()
        ap_page = create_audio_processing_page()

        source_upload_event = generation["src_audio"].change(
            handle_src_audio_upload,
            inputs=[generation["src_audio"], source_mode],
            outputs=[
                generation["src_audio_preview"],
                generation["src_video_preview"],
                source_audio_duration,
                generation["src_audio_preview_original"],
            ],
            api_name="generation_source_preview",
            queue=False,
        )
        source_upload_event.then(
            finalize_src_audio_upload,
            inputs=[generation["src_audio"], source_mode],
            outputs=[
                generation["src_audio_preview"],
                generation["src_video_preview"],
                source_audio_duration,
                generation["src_audio_preview_original"],
            ],
            queue=False,
            show_progress="full",
            show_progress_on=[
                generation["src_audio_preview"],
                generation["src_video_preview"],
            ],
        )
        custom["reference_audio"].change(
            media_upload_preview.preview_audio_purpose_upload,
            inputs=[custom["reference_audio"]],
            outputs=[custom["reference_audio_preview"], custom["reference_video_preview"]],
            api_name="generation_reference_preview",
            queue=False,
        )
        generation["lm_codes_audio_upload"].change(
            media_upload_preview.preview_audio_purpose_upload,
            inputs=[generation["lm_codes_audio_upload"]],
            outputs=[
                generation["lm_codes_audio_preview"],
                generation["lm_codes_video_preview"],
            ],
            api_name="generation_lm_codes_preview",
            queue=False,
        )
        sam_page["sam_visual_mask_file"].change(
            media_upload_preview.preview_video_upload,
            inputs=[sam_page["sam_visual_mask_file"]],
            outputs=[sam_page["sam_visual_mask_video_preview"]],
            api_name="sam_mask_video_preview",
            queue=False,
        )
        sam_page["sam_single_file"].change(
            preview_upload,
            inputs=[sam_page["sam_single_file"]],
            outputs=[
                sam_page["sam_upload_audio_preview"],
                sam_page["sam_upload_video_preview"],
                sam_page["sam_single_status"],
            ],
            api_name="sam_single_preview",
            queue=False,
        )
        ap_page["ap_single_file"].change(
            preview_audio_processing_upload,
            inputs=[ap_page["ap_single_file"]],
            outputs=[
                ap_page["ap_upload_audio_preview"],
                ap_page["ap_upload_video_preview"],
                ap_page["ap_single_status"],
            ],
            api_name="ap_single_preview",
            queue=False,
        )
    return demo


def assert_all_videos_are_capped(demo: gr.Blocks) -> None:
    """Assert every Gradio video component has the shared cap class."""

    videos = [
        component
        for component in demo.config["components"]
        if component.get("type") == "video"
    ]
    labels = [component["props"].get("label") for component in videos]
    assert "Source Video Preview" in labels
    assert "Reference Video Preview" in labels
    assert "Visual Mask Video Preview" in labels
    for component in videos:
        elem_classes = component["props"].get("elem_classes") or []
        assert "ace-video-preview" in elem_classes, component["props"].get("label")


def assert_preview_routes_bypass_queue(demo: gr.Blocks) -> None:
    """Assert preview API routes are explicitly non-queued in Gradio config."""

    route_names = {
        "generation_source_preview",
        "generation_reference_preview",
        "generation_lm_codes_preview",
        "sam_mask_video_preview",
        "sam_single_preview",
        "ap_single_preview",
    }
    dependencies = [
        dep
        for dep in demo.config["dependencies"]
        if dep.get("api_name") in route_names
    ]
    assert {dep.get("api_name") for dep in dependencies} == route_names
    for dependency in dependencies:
        assert dependency.get("queue") is False, dependency.get("api_name")


def assert_audio_purpose_video_preview(
    client: Client,
    video_path: Path | list[Path],
    api_name: str,
) -> None:
    """Assert a video upload returns both extracted audio and video previews."""

    expected_video_path = _expected_latest_path(video_path)
    audio_update, video_update = client.predict(_upload_payload(video_path), api_name=api_name)
    assert audio_update["visible"] is True
    assert str(audio_update["value"]).endswith(".wav")
    assert video_update["visible"] is True
    assert str(video_update["value"]).endswith(expected_video_path.suffix)


def assert_source_audio_purpose_video_preview(
    client: Client,
    video_path: Path | list[Path],
    api_name: str,
) -> None:
    """Assert a Source Audio upload returns the immediate direct preview."""

    expected_video_path = _expected_latest_path(video_path)
    result = client.predict(
        _upload_payload(video_path),
        api_name=api_name,
    )
    audio_update, video_update = result[:2]
    assert audio_update["visible"] is False
    assert video_update["visible"] is True
    assert str(video_update["value"]).endswith(expected_video_path.suffix)
    if len(result) >= 4:
        assert result[3] is None


def assert_video_preview(client: Client, video_path: Path | list[Path], api_name: str) -> None:
    """Assert a video-only upload returns a visible video update."""

    expected_video_path = _expected_latest_path(video_path)
    video_update = client.predict(_upload_payload(video_path), api_name=api_name)
    assert video_update["visible"] is True
    assert str(video_update["value"]).endswith(expected_video_path.suffix)


def assert_media_upload_video_preview(
    client: Client,
    video_path: Path | list[Path],
    api_name: str,
) -> None:
    """Assert a mixed media upload returns hidden audio and visible video."""

    expected_video_path = _expected_latest_path(video_path)
    audio_update, video_update, status = client.predict(
        _upload_payload(video_path),
        api_name=api_name,
    )
    assert audio_update["visible"] is False
    assert video_update["visible"] is True
    assert str(video_update["value"]).endswith(expected_video_path.suffix)
    assert "Loaded video" in status


def free_port() -> int:
    """Return an available localhost TCP port."""

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def write_wav(path: Path) -> Path:
    """Write a tiny WAV fixture for Gradio audio outputs."""

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\0\0" * 4_800)
    return path


def _upload_payload(paths: Path | list[Path]) -> list[dict]:
    """Return a Gradio-client payload for upload fields using multiple mode."""

    return [handle_file(str(path)) for path in _as_path_list(paths)]


def _expected_latest_path(paths: Path | list[Path]) -> Path:
    """Return the newest path from a single or stale upload-path list."""

    return _as_path_list(paths)[-1]


def _as_path_list(paths: Path | list[Path]) -> list[Path]:
    """Return paths as a non-empty list."""

    if isinstance(paths, Path):
        return [paths]
    return list(paths)
