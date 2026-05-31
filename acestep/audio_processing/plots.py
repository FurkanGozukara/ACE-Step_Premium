"""Plot helpers for audio-processing A/B comparisons."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def make_spectrogram_figure(
    before: np.ndarray,
    after: np.ndarray,
    sample_rate: int,
) -> plt.Figure:
    """Create a before/after spectrogram figure."""

    mono_before = _mono(before)
    mono_after = _mono(after)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), dpi=110)
    for axis, data, title in zip(axes, (mono_before, mono_after), ("Before", "After")):
        axis.specgram(
            data,
            NFFT=2048,
            Fs=sample_rate,
            noverlap=1024,
            cmap="magma",
            vmin=-100,
            vmax=0,
        )
        axis.set_title(title)
        axis.set_xlabel("Time")
        axis.set_ylabel("Hz")
        axis.set_ylim(0, min(sample_rate / 2, 20000))
    fig.tight_layout()
    return fig


def _mono(audio: np.ndarray) -> np.ndarray:
    """Return mono audio for plotting."""

    return np.mean(audio, axis=1) if audio.ndim > 1 else audio
