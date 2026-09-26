"""Helpers shared by tests and scripts: IF-vs-baseband equivalence, multi-dwell simulation."""
from __future__ import annotations

import numpy as np

from .adc import adc_sample, ddc
from .baseband import baseband_path
from .channel import simulate_channel
from .config import RadarConfig
from .frontend import downconvert, envelope_to_analog
from .scene import make_rain_cells
from .simulate import _has_rain, simulate_baseband


def if_vs_baseband(cfg: RadarConfig, cells=None, n_chirps: int = 1, seed: int = 0):
    """Noise-free comparison of the DDC output of both paths. Returns (x_if, x_bb, nmse_db)."""
    d = cfg.d
    if _has_rain(cfg):
        ch_bb = simulate_channel(cfg, fs=d.fs_ddc, n_chirps=n_chirps, rng=np.random.default_rng(seed),
                                 cells=cells)
        env_an = envelope_to_analog(cfg, ch_bb.total.reshape(-1), d.fs_ddc)
    else:
        ch_bb = simulate_channel(cfg, fs=d.fs_ddc, n_chirps=n_chirps, rng=np.random.default_rng(seed))
        env_an = simulate_channel(cfg, fs=d.fs_analog, n_chirps=n_chirps,
                                  rng=np.random.default_rng(seed)).total.reshape(-1)
    rng = np.random.default_rng(seed + 1)
    fe = downconvert(cfg, env_an, rng, add_noise=False)
    ad = adc_sample(cfg, fe["filtered"], rng, add_noise=False)
    x_if = ddc(cfg, ad["x"])
    x_bb = baseband_path(cfg, ch_bb.total, d.fs_ddc, rng, add_noise=False).stream()
    n = min(x_if.size, x_bb.size)
    x_if, x_bb = x_if[:n], x_bb[:n]
    core = slice(int(0.05 * n), int(0.95 * n))       # skip filter transients at the record edges
    nmse = np.sum(np.abs(x_if[core] - x_bb[core]) ** 2) / np.sum(np.abs(x_bb[core]) ** 2)
    return x_if, x_bb, float(10 * np.log10(nmse))


def simulate_dwells(cfg: RadarConfig, n_dwells: int | None = None, first_cells=None):
    """Independent dwells (new rain realisation each) through the baseband path."""
    n_dwells = n_dwells or cfg.sim.n_dwells
    frames, all_cells = [], []
    for k in range(n_dwells):
        cells = first_cells if (k == 0 and first_cells is not None) else (
            make_rain_cells(cfg, np.random.default_rng(cfg.scene.seed + 1000 * k)) if _has_rain(cfg) else None)
        out = simulate_baseband(cfg, rng=np.random.default_rng(cfg.scene.seed + 7 + k), cells=cells)
        frames.append(out["frame"])
        all_cells.append(cells)
        if k == 0:
            first_channel = out["channel"]
    return frames, first_channel, all_cells


def simulate_mimo_dwells(cfg: RadarConfig, n_dwells: int | None = None, add_noise: bool | None = None):
    """TDM-MIMO dwells through the baseband path (new rain realisation per dwell)."""
    from .mimo import make_rain_cells_2d, simulate_channel_mimo
    n_dwells = n_dwells or cfg.sim.n_dwells
    add_noise = cfg.sim.add_noise if add_noise is None else add_noise
    frames, first = [], None
    for k in range(n_dwells):
        rng = np.random.default_rng(cfg.scene.seed + 500 + k)
        cells = make_rain_cells_2d(cfg, rng) if _has_rain(cfg) else None
        ch = simulate_channel_mimo(cfg, rng=rng, cells=cells)
        frames.append(baseband_path(cfg, ch.total, cfg.d.fs_ddc, rng, add_noise))
        if first is None:
            first = ch
    return frames, first


def mimo_calibration(cfg: RadarConfig, range_m: float, angle_deg: float = 0.0):
    """Simulated calibration capture: one strong reflector at a known angle, same channel errors."""
    from .processing.mimo import estimate_calibration, mimo_profiles, to_virtual
    cal_cfg = cfg.replace(scene={"kind": "point", "point_targets": [
        {"name": "cal", "range_m": range_m, "rcs_dbsm": 30.0, "velocity": 0.0, "angle_deg": angle_deg}]},
        processing={"clutter": "none"})
    frames, _ = simulate_mimo_dwells(cal_cfg, 1)
    X, r = mimo_profiles(cal_cfg, frames[0])
    return estimate_calibration(cal_cfg, to_virtual(cal_cfg, X), int(np.argmin(np.abs(r - range_m))), angle_deg)
