"""Blocks 13–14 — Doppler processing (FFT + pulse-pair) and noise estimation."""
from __future__ import annotations

import numpy as np

from ..config import RadarConfig, get_window


def doppler_spectrum(cfg: RadarConfig, profiles):
    """Doppler power spectrum per range bin, shape (n_vel, n_bins), sum over velocity = mean power.

    Velocity axis uses the convention v > 0 toward the radar (beat-domain Doppler = -2v/lambda).
    """
    d = cfg.d
    M = profiles.shape[0]
    w = get_window(cfg.processing.doppler_window, M)
    X = np.fft.fftshift(np.fft.fft(profiles * w[:, None], axis=0), axes=0)
    spec = np.abs(X) ** 2 / (M * np.sum(w ** 2))
    f = np.fft.fftshift(np.fft.fftfreq(M, d.t_rep))
    v = -d.lam * f / 2
    order = np.argsort(v)
    return spec[order], v[order]


def pulse_pair(profiles):
    R0 = np.mean(np.abs(profiles) ** 2, axis=0)
    R1 = np.mean(profiles[1:] * np.conj(profiles[:-1]), axis=0)
    return R0, R1


def hildebrand_sekhon(spec_col, n_avg: int = 1):
    """Noise level (per spectral bin) of one Doppler spectrum (Hildebrand & Sekhon 1974)."""
    s = np.sort(spec_col)
    csum = np.cumsum(s)
    csum2 = np.cumsum(s ** 2)
    n = np.arange(1, s.size + 1)
    mean = csum / n
    var = csum2 / n - mean ** 2
    ok = (var * n_avg <= mean ** 2) | (n <= 2)
    k = np.max(np.nonzero(ok)[0])
    return mean[k]


def moments(cfg: RadarConfig, R0, R1, noise):
    d = cfg.d
    S = R0 - noise
    v = -d.lam / (4 * np.pi * d.t_rep) * np.angle(R1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(S > 0, S / np.maximum(np.abs(R1), 1e-300), np.nan)
        sw = d.lam / (2 * np.sqrt(2) * np.pi * d.t_rep) * np.sqrt(np.abs(np.log(np.maximum(ratio, 1.0))))
        snr = S / noise
    return S, v, sw, snr
