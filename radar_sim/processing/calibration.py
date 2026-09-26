"""Blocks 15–17 — calibration to reflectivity, attenuation correction, rain rate."""
from __future__ import annotations

import numpy as np

from ..config import C, RadarConfig, db, get_window, window_enbw
from ..scene import radar_constant


def delta_r_eff(cfg: RadarConfig) -> float:
    """Effective range-bin length for distributed targets = ENBW(window) * c/2B."""
    return window_enbw(cfg.processing.range_window, cfg.d.n_beat) * cfg.d.delta_r


def overlap_loss(cfg: RadarConfig, ranges):
    """Power loss because the beat tone only exists for t in [tau, T_chirp] (windowed)."""
    d = cfg.d
    w = get_window(cfg.processing.range_window, d.n_beat)
    cw = np.concatenate([[0.0], np.cumsum(w)])
    n_tau = np.clip(np.round(2 * np.asarray(ranges) / C * d.fs_beat).astype(int), 0, d.n_beat)
    return ((cw[-1] - cw[n_tau]) / cw[-1]) ** 2


def reflectivity(cfg: RadarConfig, power_adc, ranges):
    """Noise-corrected bin power at the ADC input (W) -> Z [mm^6/m^3]."""
    d = cfg.d
    p_ant = power_adc / d.g_chain * 10 ** (cfg.processing.cal_offset_db / 10)
    c = radar_constant(cfg, delta_r_eff(cfg))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.maximum(np.asarray(ranges), 1e-3)
        return c * p_ant * r ** 2 / overlap_loss(cfg, r)


def hitschfeld_bordan(cfg: RadarConfig, z_meas, ranges, valid, max_pia_db: float = 10.0):
    """Hitschfeld–Bordan correction with k = alpha Z^beta derived from k-R and Z-R.

    Returns (z_corrected, pia_two_way_db). Unstable when the path attenuation is large, so the
    correction is capped at ``max_pia_db``.
    """
    sc, pr = cfg.scene, cfg.processing
    beta = sc.k_b / pr.zr_b
    alpha = sc.k_a * pr.zr_a ** (-beta)
    z = np.where(valid & (z_meas > 0), z_meas, 0.0)
    dr_km = np.gradient(np.asarray(ranges)) / 1000.0
    integ = np.cumsum(z ** beta * dr_km) - 0.5 * z ** beta * dr_km
    denom = 1 - 0.2 * np.log(10) * beta * alpha * integ
    denom = np.maximum(denom, 10 ** (-beta * max_pia_db / 10))
    corr = denom ** (-1 / beta)
    return z_meas * corr, db(corr)


def rain_rate(cfg: RadarConfig, z):
    pr = cfg.processing
    return np.where(z > 0, (np.maximum(z, 0) / pr.zr_a) ** (1 / pr.zr_b), np.nan)


def min_detectable_dbz(cfg: RadarConfig, noise_adc, ranges, snr_db: float = 0.0):
    z = reflectivity(cfg, noise_adc * 10 ** (snr_db / 10), ranges)
    with np.errstate(divide="ignore", invalid="ignore"):
        return db(z)
