"""Blocks 9–10 — digital dechirp and beat low-pass / second decimation."""
from __future__ import annotations

import numpy as np
from scipy.signal import firwin

from ..adc import fir_decimate
from ..config import RadarConfig
from ..waveform import reference_chirp


def dechirp(cfg: RadarConfig, data, fs: float, freq_offset: float):
    """beat = ref * conj(rx) over the chirp (idle samples dropped). Beat frequency = +S*tau.

    The reference is the TX envelope shifted by ``freq_offset`` (= f_IF - f_NCO) so that it
    matches what the DDC produced.
    """
    ref = reference_chirp(cfg, fs, freq_offset)
    return ref * np.conj(data[..., :ref.size])          # works for (chirps, n) and (chirps, rx, n)


def beat_filter(cfg: RadarConfig):
    d = cfg.d
    D = d.beat_decimation
    cutoff = min(1.25 * d.fb_max, 0.45 * d.fs_beat)
    ntaps = max(16 * D, 64) + 1
    return firwin(ntaps, cutoff, window=("kaiser", 8.0), fs=d.fs_ddc), cutoff


def beat_decimate(cfg: RadarConfig, beat):
    h, _ = beat_filter(cfg)
    return fir_decimate(beat, h, cfg.d.beat_decimation, axis=-1)
