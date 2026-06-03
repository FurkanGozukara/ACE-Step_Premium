"""Shared media upload extension lists for Gradio UI controls."""

AUDIO_FILE_TYPES = [
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".aif",
    ".aiff",
]
VIDEO_FILE_TYPES = [".mp4", ".mov", ".mkv", ".webm", ".avi"]
MEDIA_FILE_TYPES = [*AUDIO_FILE_TYPES, *VIDEO_FILE_TYPES]
