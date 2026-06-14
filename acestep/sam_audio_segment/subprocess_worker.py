"""SAM-Audio subprocess worker entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .batch_segment import batch_segment_prompts, settings_for_batch_segment_prompt
from .batch import run_batch_sam_audio
from .paths import safe_media_stem
from .progress import ProgressCallback, encode_progress_line, report_progress
from .service import SamAudioService
from .settings import SamAudioSettings


def main(argv: list[str] | None = None) -> int:
    """Run a SAM-Audio request from JSON and write a JSON result."""

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: python -m acestep.sam_audio_segment.subprocess_worker request result")
        return 2
    request_path = Path(args[0])
    result_path = Path(args[1])
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    settings = SamAudioSettings.from_payload(payload.get("settings"))
    try:
        if payload.get("mode") == "batch":
            result = _run_batch(payload, settings, _stdout_progress)
        else:
            result = _run_single(payload, settings, _stdout_progress)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result.get("ok") else 1


def _run_single(
    payload: dict,
    settings: SamAudioSettings,
    progress_callback: ProgressCallback,
) -> dict:
    """Run one SAM-Audio file request."""

    prompts = batch_segment_prompts(settings)
    if prompts:
        return _run_single_batch_segments(payload, settings, prompts, progress_callback)

    service = SamAudioService(
        settings,
        model_path=payload.get("model_path"),
        progress_callback=progress_callback,
    )
    try:
        artifacts = service.process_file(
            payload["input_path"],
            payload["output_dir"],
            output_stem=payload.get("output_stem"),
            mask_video_path=payload.get("mask_video_path"),
        )
    finally:
        service.unload()
    return {"ok": True, "artifacts": artifacts.__dict__, "files": artifacts.file_list()}


def _run_single_batch_segments(
    payload: dict,
    settings: SamAudioSettings,
    prompts,
    progress_callback: ProgressCallback,
) -> dict:
    """Run one file once per Batch Segment prompt."""

    service = SamAudioService(
        settings_for_batch_segment_prompt(settings, prompts[0]),
        model_path=payload.get("model_path"),
        progress_callback=_segment_progress_callback(
            progress_callback,
            1,
            len(prompts),
            prompts[0].text,
        ),
    )
    artifacts = []
    input_path = payload["input_path"]
    base_stem = payload.get("output_stem") or safe_media_stem(input_path)
    try:
        for index, prompt in enumerate(prompts, start=1):
            service.settings = settings_for_batch_segment_prompt(settings, prompt)
            service.progress_callback = _segment_progress_callback(
                progress_callback,
                index,
                len(prompts),
                prompt.text,
            )
            artifacts.append(
                service.process_file(
                    input_path,
                    payload["output_dir"],
                    output_stem=f"{base_stem}_{prompt.suffix}",
                    mask_video_path=payload.get("mask_video_path"),
                )
            )
    finally:
        service.unload()
    files = [path for artifact in artifacts for path in artifact.file_list()]
    batch_artifacts = []
    for artifact, prompt in zip(artifacts, prompts):
        item = dict(artifact.__dict__)
        item["_batch_segment_prompt"] = prompt.text
        batch_artifacts.append(item)
    return {
        "ok": True,
        "artifacts": dict(batch_artifacts[0]),
        "batch_artifacts": batch_artifacts,
        "files": files,
    }


def _run_batch(
    payload: dict,
    settings: SamAudioSettings,
    progress_callback: ProgressCallback,
) -> dict:
    """Run one SAM-Audio batch request."""

    last_status = ""
    files: list[str] = []
    for status, current_files in run_batch_sam_audio(
        payload.get("input_folder") or "",
        payload.get("output_folder") or "",
        bool(payload.get("recursive")),
        settings,
        progress_callback=progress_callback,
    ):
        last_status = status
        files = current_files
    return {"ok": True, "status": last_status, "files": files}


def _stdout_progress(fraction: float, message: str) -> None:
    """Write one progress event for the parent process to consume."""

    sys.stdout.write(encode_progress_line(fraction, message) + "\n")
    sys.stdout.flush()


def _segment_progress_callback(
    callback: ProgressCallback,
    index: int,
    total: int,
    prompt: str,
) -> ProgressCallback:
    """Map one prompt's progress into the whole Batch Segment run."""

    def _report(fraction: float, message: str) -> None:
        overall = ((index - 1) + float(fraction)) / max(1, total)
        report_progress(callback, overall, f"[{index}/{total}] {prompt}: {message}")

    return _report


if __name__ == "__main__":
    raise SystemExit(main())
