"""Blocks 4–6 — down-converter (ADMV1014 + 90° hybrid), receiver noise, anti-alias BPF.

The "analog" signal is simulated at ``fs_analog = fs_adc * analog_oversample``.

Model
-----
1. Complex IF w.r.t. the LO:  a(t) = sqrt(G) * env(t) * exp(j 2 pi f_IF t)  (wanted = +f_IF)
   plus thermal noise over BOTH sidebands (the -f_IF side is the image band).
2. I/Q imbalance of the mixer:  a' = mu a + nu conj(a)   (image rejection = |mu|^2/|nu|^2)
3. 90° hybrid (image-reject combiner):  y = sqrt(2) Re{P+ a'}  (P+ = positive-frequency part)
   -> image-band content leaks into the wanted band attenuated by the IRR.
4. LO feed-through appears at DC of the real IF.
5. BPF selects one Nyquist zone of the ADC.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from .config import RadarConfig, dbm_to_w


def envelope_to_analog(cfg: RadarConfig, env_stream, fs_env: float):
    """Resample a channel envelope (1-D stream) to the analog simulation rate."""
    up = cfg.d.fs_analog / fs_env
    if abs(up - round(up)) > 1e-9:
        raise ValueError("fs_analog / fs_env must be an integer")
    return resample_poly(env_stream, int(round(up)), 1, window=("kaiser", 10.0))


def analytic_part(x):
    """Keep positive frequencies of a complex signal (P+)."""
    X = np.fft.fft(x)
    f = np.fft.fftfreq(x.size)
    X[f < 0] = 0
    X[f == 0] *= 0.5
    return np.fft.ifft(X)


def bpf_mask(cfg: RadarConfig, freqs):
    """Anti-alias bandpass (raised-cosine edges) selecting Nyquist zone ``d.zone``."""
    d = cfg.d
    g = cfg.frontend.bpf_guard_hz
    lo, hi = (d.zone - 1) * d.zone_width, d.zone * d.zone_width
    m = np.zeros_like(freqs, dtype=float)
    pb = (freqs >= lo + g) & (freqs <= hi - g)
    m[pb] = 1.0
    lo_edge = (freqs > lo) & (freqs < lo + g)
    hi_edge = (freqs > hi - g) & (freqs < hi)
    m[lo_edge] = 0.5 - 0.5 * np.cos(np.pi * (freqs[lo_edge] - lo) / g)
    m[hi_edge] = 0.5 - 0.5 * np.cos(np.pi * (hi - freqs[hi_edge]) / g)
    return m


def downconvert(cfg: RadarConfig, env_an, rng, add_noise: bool = True, t0: float = 0.0,
                apply_bpf: bool = True) -> dict:
    """Blocks 4–6 on an envelope sampled at fs_analog. Returns intermediate signals."""
    d = cfg.d
    fs = d.fs_analog
    n = env_an.size
    t = t0 + np.arange(n) / fs
    a = np.sqrt(d.g_chain) * env_an * np.exp(2j * np.pi * d.f_if_center * t)
    a_signal = a.copy()
    if add_noise:
        sig = np.sqrt(d.n0_thermal * fs / 2)
        a = a + sig * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    a_imb = d.iq_mu * a + d.iq_nu * np.conj(a)
    y = np.sqrt(2) * np.real(analytic_part(a_imb))
    y = y + np.sqrt(dbm_to_w(cfg.frontend.lo_leakage_dbm))
    out = dict(fs=fs, t=t, complex_if=a_imb, complex_if_clean=a_signal, real_if=y)
    if apply_bpf:
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(n, 1 / fs)
        out["bpf_freqs"] = f
        out["bpf_mask"] = bpf_mask(cfg, f)
        out["filtered"] = np.fft.irfft(Y * out["bpf_mask"], n)
    else:
        out["filtered"] = y
    return out
