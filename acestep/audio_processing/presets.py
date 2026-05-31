"""Built-in audio-processing presets and stage metadata."""

STAGE_KEYS: tuple[str, ...] = (
    "phase",
    "stereo",
    "hf",
    "harmonic",
    "jitter",
    "noise",
    "multiband",
    "tape",
    "glue",
    "mseq",
    "clip",
    "lufs",
)

STAGE_LABELS: dict[str, str] = {
    "phase": "Stereo Depth",
    "stereo": "Stereo Width",
    "hf": "HF Refinement",
    "harmonic": "Harmonic Enrichment",
    "jitter": "Timing Humanizer",
    "noise": "Ambience Shaping",
    "multiband": "Multiband Compressor",
    "tape": "Tape Saturation",
    "glue": "Glue Compressor",
    "mseq": "Mid/Side EQ",
    "clip": "Soft Clipper",
    "lufs": "LUFS Normalization",
}

DEFAULT_STAGE_VALUES: dict[str, float] = {
    "phase": 0.6,
    "stereo": 1.3,
    "hf": 0.5,
    "harmonic": 0.25,
    "jitter": 0.3,
    "noise": 0.2,
    "multiband": 0.3,
    "tape": 0.25,
    "glue": 0.3,
    "mseq": 0.25,
    "clip": 0.2,
    "lufs": -14.0,
}

PRESET_VALUES: dict[str, dict[str, float]] = {
    "Custom": DEFAULT_STAGE_VALUES,
    "Suno": {
        "phase": 0.7,
        "stereo": 1.4,
        "hf": 0.7,
        "harmonic": 0.3,
        "jitter": 0.4,
        "noise": 0.3,
        "multiband": 0.5,
        "tape": 0.4,
        "glue": 0.4,
        "mseq": 0.4,
        "clip": 0.3,
        "lufs": -14.0,
    },
    "Udio": {
        "phase": 0.6,
        "stereo": 1.3,
        "hf": 0.8,
        "harmonic": 0.2,
        "jitter": 0.3,
        "noise": 0.25,
        "multiband": 0.4,
        "tape": 0.35,
        "glue": 0.35,
        "mseq": 0.3,
        "clip": 0.25,
        "lufs": -14.0,
    },
    "Generic AI": DEFAULT_STAGE_VALUES,
    "Light Touch": {
        "phase": 0.3,
        "stereo": 1.1,
        "hf": 0.3,
        "harmonic": 0.1,
        "jitter": 0.2,
        "noise": 0.1,
        "multiband": 0.15,
        "tape": 0.1,
        "glue": 0.15,
        "mseq": 0.1,
        "clip": 0.1,
        "lufs": -14.0,
    },
}

OUTPUT_FORMAT_CHOICES: tuple[tuple[str, str], ...] = (
    ("WAV", "wav"),
    ("FLAC", "flac"),
    ("MP3", "mp3"),
)
