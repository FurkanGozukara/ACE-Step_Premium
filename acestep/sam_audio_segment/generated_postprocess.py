"""Generated-song SAM-Audio post-processing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from .service_cache import cached_sam_audio_service
from .settings import SamAudioSettings
from .subprocess_runner import run_sam_audio_subprocess


def postprocess_generated_sample(
    *,
    source_audio_path: str,
    run_dir: str | Path,
    key: str,
    settings: SamAudioSettings,
) -> dict[str, Any]:
    """Apply SAM-Audio to one generated audio file."""

    if not settings.auto_postprocess or not source_audio_path:
        return {"applied": False}
    output_stem = f"{key}_sam_audio"
    try:
        if settings.subprocess:
            result = run_sam_audio_subprocess(
                {
                    "mode": "single",
                    "input_path": source_audio_path,
                    "output_dir": str(run_dir),
                    "output_stem": output_stem,
                    "settings": settings.to_payload(),
                }
            )
            artifacts = result["artifacts"]
            files = result.get("files", [])
        else:
            with cached_sam_audio_service(settings) as service:
                artifact_obj = service.process_file(
                    source_audio_path,
                    run_dir,
                    output_stem=output_stem,
                )
            artifacts = artifact_obj.__dict__
            files = artifact_obj.file_list()
    except Exception as exc:
        logger.exception("[sam_audio] Generated-song post-processing failed")
        return {"applied": False, "error": str(exc)}

    return {
        "applied": True,
        "target_audio_path": artifacts.get("target_audio_path"),
        "residual_audio_path": artifacts.get("residual_audio_path"),
        "target_video_path": artifacts.get("target_video_path"),
        "metadata_path": artifacts.get("metadata_path"),
        "settings": settings.to_payload(),
        "files": files,
    }
