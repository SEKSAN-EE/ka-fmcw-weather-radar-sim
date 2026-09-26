"""Block 11 — range FFT."""
from __future__ import annotations

import numpy as np

from ..config import C, RadarConfig, get_window


def range_fft(cfg: RadarConfig, beat_lp):
    """Windowed FFT, normalised so a beat tone of power P gives |X|^2 = P at its peak.

    Returns (profiles[n_chirps, n_bins], ranges[n_bins]) limited to r <= r_max.
    """
    d = cfg.d
    pr = cfg.processing
    n = beat_lp.shape[-1]
    w = get_window(pr.range_window, n)
    nfft = n * max(pr.range_zero_pad, 1)
    X = np.fft.fft(beat_lp * w, nfft, axis=-1) / np.sum(w)
    f = np.arange(nfft) * d.fs_beat / nfft
    r = f * C / (2 * d.slope) - pr.range_offset_m
    keep = (f < d.fs_beat / 2) & (r <= pr.r_max) & (r > -d.delta_r)
    return X[..., keep], r[keep]
