"""Audio metadata file and duration helpers for dataset scanning."""

import json
import os
import subprocess
from typing import Any, Dict, Tuple

from loguru import logger

from acestep.training.path_safety import safe_path
from .lyrics_file_parser import ParsedLyricsFile, parse_lyrics_text_file


_LYRICS_SUBDIRS = (
    "codex_formatted_lyrics",
    "formatted_lyrics",
    "lyrics",
    "fixed_org_lyrics",
    "raw_lyrics",
    "org_lyrics",
    "lyrics_by_Whisper_App",
)


def _read_text_file(path: str) -> Tuple[str, bool]:
    """Read a text file; return (content.strip(), True) if present and non-empty.

    Args:
        path: Already-validated file path.
    """
    validated = safe_path(path)
    if not os.path.exists(validated):
        return "", False
    try:
        with open(validated, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            return content, True
        return "", False
    except Exception as e:
        logger.warning(f"Failed to read {validated}: {e}")
        return "", False


def load_caption_file(audio_path: str) -> Tuple[str, bool]:
    """Load caption from <basename>.caption.txt (explicit convention)."""
    validated = safe_path(audio_path)
    base_path = os.path.splitext(validated)[0]
    caption_path = base_path + ".caption.txt"
    content, ok = _read_text_file(caption_path)
    if ok:
        logger.debug(f"Loaded caption from {caption_path}")
    return content, ok


def load_json_metadata(audio_path: str) -> Tuple[Dict[str, Any], bool]:
    """Load metadata from <basename>.json.

    Expected JSON structure:
        {
            "caption": "",
            "bpm": 120,
            "keyscale": "C major",
            "timesignature": "4",
            "language": "ja"
        }

    All fields are optional.
    """
    validated = safe_path(audio_path)
    base_path = os.path.splitext(validated)[0]
    json_path = base_path + ".json"
    if not os.path.exists(json_path):
        return {}, False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            logger.debug(f"Loaded JSON metadata from {json_path}")
            return data, True
        return {}, False
    except Exception as e:
        logger.warning(f"Failed to read {json_path}: {e}")
        return {}, False


def load_lyrics_file(audio_path: str) -> Tuple[str, bool]:
    """Load parsed lyrics from explicit and legacy sidecar text files."""
    parsed, ok = load_lyrics_file_data(audio_path)
    return parsed.lyrics, ok


def load_lyrics_file_data(audio_path: str) -> Tuple[ParsedLyricsFile, bool]:
    """Load and parse a lyric sidecar for an audio file."""

    validated = safe_path(audio_path)
    for path in _lyrics_file_candidates(validated):
        content, ok = _read_text_file(path)
        if ok:
            parsed = parse_lyrics_text_file(content)
            if not parsed.lyrics:
                continue
            if path.endswith(".lyrics.txt"):
                logger.debug(f"Loaded lyrics from {path}")
            else:
                logger.debug(f"Loaded lyrics from {path} (legacy .txt)")
            return parsed, True
    return ParsedLyricsFile(), False


def _lyrics_file_candidates(audio_path: str) -> list[str]:
    """Return lyric sidecar candidates in compatibility order."""

    base_path = os.path.splitext(audio_path)[0]
    audio_dir = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    candidates = [base_path + ".lyrics.txt", base_path + ".txt"]

    for subdir in _LYRICS_SUBDIRS:
        folder = os.path.join(audio_dir, subdir)
        candidates.append(os.path.join(folder, stem + ".lyrics.txt"))
        candidates.append(os.path.join(folder, stem + ".txt"))

    deduped = []
    seen = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.normpath(candidate))
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def get_audio_duration(audio_path: str) -> int:
    """Get the duration of an audio file in seconds."""
    validated = safe_path(audio_path)
    duration = _duration_from_soundfile(validated)
    if duration is not None:
        return duration

    duration = _duration_from_ffprobe(validated)
    if duration is not None:
        return duration

    logger.warning(f"Failed to get duration for {validated}")
    return 0


def _duration_from_soundfile(audio_path: str) -> int | None:
    """Return duration using libsndfile when it supports the format."""

    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return int(info.duration)
    except Exception as exc:
        logger.debug(f"soundfile duration probe failed for {audio_path}: {exc}")
        return None


def _duration_from_ffprobe(audio_path: str) -> int | None:
    """Return duration using the standard FFmpeg CLI tools."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        logger.debug(f"ffprobe duration probe failed for {audio_path}: {exc}")
        return None

    if result.returncode != 0:
        stderr = result.stderr.strip().splitlines()
        detail = stderr[0] if stderr else f"exit code {result.returncode}"
        logger.debug(f"ffprobe duration probe failed for {audio_path}: {detail}")
        return None

    try:
        return int(float(result.stdout.strip()))
    except ValueError:
        logger.debug(f"ffprobe returned invalid duration for {audio_path}: {result.stdout!r}")
        return None
