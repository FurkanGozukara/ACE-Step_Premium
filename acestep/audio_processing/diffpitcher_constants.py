"""Constants shared by DiffPitcher inference helpers."""

from __future__ import annotations

import librosa
import numpy as np
from librosa.filters import mel as librosa_mel_fn

DIFFPITCHER_SAMPLE_RATE = 24000
DIFFPITCHER_N_FFT = 1024
DIFFPITCHER_HOP = 256
DIFFPITCHER_N_MELS = 100
DIFFPITCHER_FMAX = 12000
DIFFPITCHER_STFT_PAD = 384
DIFFPITCHER_MEL_MIN = float(np.log(1e-5))
DIFFPITCHER_MEL_MAX = 2.5
DIFFPITCHER_F0_MIN = float(librosa.note_to_hz("C2"))
DIFFPITCHER_F0_MAX = float(librosa.note_to_hz("C#6"))
DIFFPITCHER_F0_CLAMP_MAX = float(librosa.note_to_hz("C6"))
DIFFPITCHER_F0_BIN = 345

DIFFPITCHER_MEL_BASIS = librosa_mel_fn(
    sr=DIFFPITCHER_SAMPLE_RATE,
    n_fft=DIFFPITCHER_N_FFT,
    n_mels=DIFFPITCHER_N_MELS,
    fmin=0,
    fmax=DIFFPITCHER_FMAX,
)
