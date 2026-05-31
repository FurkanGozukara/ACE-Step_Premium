"""Prompt presets for open-vocabulary SAM-Audio segmentation."""

from .attention import SAM_ATTENTION_CHOICES
from .vram_presets import SAM_VRAM_PRESET_CHOICES

PROMPT_PRESET_CHOICES: tuple[tuple[str, str], ...] = (
    ("Vocals", "vocals"),
    ("Lead vocal", "lead vocal"),
    ("Backing vocals", "backing vocals"),
    ("Singing voice", "singing voice"),
    ("Speech", "speech"),
    ("Man speaking", "man speaking"),
    ("Woman speaking", "woman speaking"),
    ("Speaker", "speaker"),
    ("Music", "music"),
    ("Instrumental", "instrumental"),
    ("Drums", "drums"),
    ("Kick drum", "kick drum"),
    ("Snare drum", "snare drum"),
    ("Hi-hat", "hi-hat"),
    ("Bass", "bass"),
    ("Electric guitar", "electric guitar"),
    ("Acoustic guitar", "acoustic guitar"),
    ("Piano", "piano"),
    ("Synthesizer", "synthesizer"),
    ("Strings", "strings"),
    ("Violin", "violin"),
    ("Trumpet", "trumpet"),
    ("Saxophone", "saxophone"),
    ("Percussion", "percussion"),
    ("Applause", "applause"),
    ("Crowd", "crowd"),
    ("Laughter", "laughter"),
    ("Footsteps", "footsteps"),
    ("Rain", "rain"),
    ("Thunder", "thunder"),
    ("Car engine", "car engine"),
    ("Horn honking", "horn honking"),
)

DEFAULT_PROMPT = "vocals"

RANKER_CHOICES: tuple[tuple[str, str], ...] = (
    ("Disabled", "none"),
    ("Official text ensemble (CLAP + Judge)", "text_ensemble"),
    ("CLAP text similarity", "clap"),
    ("SAM-Audio Judge", "judge"),
    ("ImageBind visual similarity", "imagebind"),
)

VRAM_PRESET_CHOICES = SAM_VRAM_PRESET_CHOICES

QUANTIZATION_CHOICES: tuple[tuple[str, str], ...] = (
    ("Disabled", "none"),
    ("FP8 scaled cache", "fp8_scaled"),
)

ATTENTION_BACKEND_CHOICES = SAM_ATTENTION_CHOICES

DEVICE_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Auto", "auto"),
    ("CUDA", "cuda"),
    ("CPU", "cpu"),
)

OUTPUT_FORMAT_CHOICES: tuple[tuple[str, str], ...] = (
    ("WAV", "wav"),
    ("FLAC", "flac"),
    ("MP3", "mp3"),
)
