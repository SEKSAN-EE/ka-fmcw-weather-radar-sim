"""Blocks 7–8 — RF-ADC (sampling, jitter, noise, 14-bit quantisation, clipping) and DDC.

The DDC here is the reference implementation also used for real raw-IF captures
(:mod:`radar_sim.io.capture`), so simulated and measured data go through the same code.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import firwin, upfirdn

from .config import RadarConfig


def adc_sample(cfg: RadarConfig, y_analog, rng, add_noise: bool = True) -> dict:
    """Sample the analog IF (at fs_analog) at fs_adc and quantise. Returns volts-like sqrt(W) and codes."""
    d, adc = cfg.d, cfg.adc
    osr = adc.analog_oversample
    y = y_analog
    if adc.jitter_rms > 0:
        # first-order jitter model: y(t + dt) ~ y(t) + dt * y'(t)
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(y.size, 1 / d.fs_analog)
        dy = np.fft.irfft(2j * np.pi * f * Y, y.size)
        y = y + adc.jitter_rms * rng.standard_normal(y.size) * dy
    x = y[::osr].copy()
    if add_noise:
        x += np.sqrt(d.n0_adc * adc.fs / 2) * rng.standard_normal(x.size)
    q = 2 * d.a_fs / 2 ** adc.bits
    lo, hi = -(2 ** (adc.bits - 1)), 2 ** (adc.bits - 1) - 1
    raw = np.round(x / q)
    clipped = np.count_nonzero((raw < lo) | (raw > hi))
    codes = np.clip(raw, lo, hi).astype(np.int16)
    return dict(fs=adc.fs, codes=codes, x=codes * q, lsb=q, clipped=clipped,
                peak_dbfs=20 * np.log10(np.max(np.abs(codes)) / 2 ** (adc.bits - 1) + 1e-30))


def codes_to_signal(cfg: RadarConfig, codes):
    """ADC codes -> sqrt(W) at the ADC input (inverse of the quantiser scale)."""
    return np.asarray(codes, float) * 2 * cfg.d.a_fs / 2 ** cfg.adc.bits


def fir_decimate(x, h, D: int, axis: int = -1):
    """Linear-phase FIR + decimate by D with the group delay removed.

    Output sample k is aligned with input sample k*D.
    """
    x = np.moveaxis(np.asarray(x), axis, -1)
    delay = (len(h) - 1) // 2
    pad = (-delay) % D
    k0 = (pad + delay) // D
    n_out = x.shape[-1] // D
    xp = np.concatenate([np.zeros(x.shape[:-1] + (pad,), x.dtype), x,
                         np.zeros(x.shape[:-1] + (len(h),), x.dtype)], axis=-1)
    y = upfirdn(h, xp, up=1, down=D, axis=-1)[..., k0:k0 + n_out]
    return np.moveaxis(y, -1, axis)


def ddc_filter(cfg: RadarConfig):
    D = cfg.adc.ddc_decimation
    fs_out = cfg.adc.fs / D
    if D == 1:
        return np.array([1.0])
    ntaps = max(24 * D, 192) + 1
    return firwin(ntaps, 0.91 * fs_out / 2, window=("kaiser", 8.0), fs=cfg.adc.fs)


def ddc(cfg: RadarConfig, x_real, n0: int = 0, f_nco: float | None = None) -> np.ndarray:
    """NCO mix to complex baseband, low-pass, decimate. Scaled so complex power == real power.

    ``n0`` is the ADC sample index of ``x_real[0]`` (keeps NCO phase continuous across blocks).
    """
    d = cfg.d
    f_nco = d.f_nco if f_nco is None else f_nco
    n = n0 + np.arange(x_real.size)
    lo = np.exp(-2j * np.pi * ((f_nco / cfg.adc.fs) % 1.0) * n)
    return np.sqrt(2) * fir_decimate(x_real * lo, ddc_filter(cfg), cfg.adc.ddc_decimation)
