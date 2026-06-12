"""Stage controls for the Audio Processing page."""

from __future__ import annotations

from typing import Any

import gradio as gr

from acestep.audio_processing.presets import (
    DEFAULT_STAGE_VALUES,
    STAGE_KEYS,
    STAGE_LABELS,
)

STAGE_INFOS: dict[str, str] = {
    "phase": (
        "Adds subtle phase movement between the left and right channels for depth while "
        "keeping the low end stable. 0 = off; 1 = strongest. Example: helps a flat AI mix "
        "feel less stuck in the center; use less if mono playback sounds hollow."
    ),
    "stereo": (
        "Expands the side signal with mid/side processing. 1.0 = original width; "
        "1.3 = natural widening; 2.0 = very wide. Example: spreads pads and guitars "
        "outward while the vocal, kick, and bass still feel centered."
    ),
    "hf": (
        "Smooths very high frequencies above about 12 kHz to reduce brittle fizz, hiss, "
        "or splashy cymbals. 0 = off; 1 = strongest smoothing. Example: tames sharp AI "
        "high end without muting the whole mix."
    ),
    "harmonic": (
        "Blends gentle saturation that adds extra harmonics and warmth. 0 = clean; "
        "1 = strongest color. Example: gives thin vocals, bass, or synths more body; "
        "too much can turn warmth into audible grit."
    ),
    "jitter": (
        "Finds transient peaks and nudges tiny regions by sub-millisecond amounts. 0 = "
        "locked timing; 1 = strongest looseness. Example: makes rigid drums or piano feel "
        "less machine-like; too much can soften attacks."
    ),
    "noise": (
        "Adds a very quiet gated pink-noise ambience that follows the music envelope. "
        "0 = no added floor; 1 = most ambience. Example: fills sterile digital silence "
        "between phrases; too much can sound like hiss."
    ),
    "multiband": (
        "Splits lows, mids, and highs and compresses each band gently. 0 = off; "
        "1 = tightest dynamics. Example: controls boomy bass and spiky hats without "
        "squashing the entire mix."
    ),
    "tape": (
        "Adds tape-like saturation plus a light high-frequency rolloff. 0 = clean; "
        "1 = warmest and densest. Example: rounds a sharp digital mix and adds weight; "
        "too much can make the track dull."
    ),
    "glue": (
        "Applies linked stereo bus compression so loud peaks pull both channels together. "
        "0 = off; 1 = most squeeze. Example: helps drums, vocals, and instruments feel "
        "like one record; too much can pump or flatten the groove."
    ),
    "mseq": (
        "Shapes center and side tone separately: focuses low-end center, adds center "
        "presence, and gives the sides a little air. 0 = off; 1 = strongest shaping. "
        "Example: keeps kick/vocal solid while widening sparkle."
    ),
    "clip": (
        "Rounds loud peaks near the ceiling for controlled loudness. 0 = off; "
        "1 = strongest peak rounding. Example: catches kick or snare spikes before "
        "normalization; too much can sound crunchy."
    ),
    "lufs": (
        "Sets the final integrated loudness target. More negative is quieter with more "
        "headroom; less negative is louder. Example: -14 LUFS suits common streaming "
        "playback, -24 is conservative, and -8 is very loud. Peaks are capped near -1 dBFS."
    ),
}


def add_stage_controls(controls: dict[str, Any]) -> None:
    """Add Audio Enhancement and Pre-Mastering stage controls."""

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### Audio Enhancement")
            controls["ap_toggle_audio_enhancement_btn"] = gr.Button(
                "Check / Uncheck All Stages",
                variant="secondary",
                size="sm",
            )
            for key in STAGE_KEYS[:6]:
                _add_stage_control(controls, key)
        with gr.Column(scale=1):
            gr.Markdown("### Pre-Mastering")
            for key in STAGE_KEYS[6:]:
                _add_stage_control(controls, key)


def _add_stage_control(controls: dict[str, Any], key: str) -> None:
    """Add one enabled checkbox and value slider to the page controls."""

    minimum, maximum, step = _slider_bounds(key)
    with gr.Row(equal_height=True):
        controls[f"ap_{key}_enabled"] = gr.Checkbox(
            label=STAGE_LABELS[key],
            value=True,
            scale=1,
        )
        controls[f"ap_{key}"] = gr.Slider(
            minimum=minimum,
            maximum=maximum,
            step=step,
            value=DEFAULT_STAGE_VALUES[key],
            label=STAGE_LABELS[key],
            info=STAGE_INFOS[key],
            scale=4,
        )


def _slider_bounds(key: str) -> tuple[float, float, float]:
    """Return slider bounds for an audio-processing stage."""

    if key == "stereo":
        return 1.0, 2.0, 0.05
    if key == "lufs":
        return -24.0, -8.0, 0.5
    return 0.0, 1.0, 0.05
