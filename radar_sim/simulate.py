"""Simulation orchestration: full IF path (few chirps) and complex-baseband path (all chirps)."""
from __future__ import annotations

import numpy as np

from .adc import adc_sample, ddc
from .baseband import baseband_path
from .channel import simulate_channel
from .config import RadarConfig
from .frontend import downconvert, envelope_to_analog
from .iq import IQFrame
from .scene import make_rain_cells


def _has_rain(cfg):
    return cfg.scene.kind in ("rain", "both") and bool(cfg.scene.rain_layers)


def simulate_if_path(cfg: RadarConfig, rng=None, n_chirps: int | None = None, cells=None,
                     add_noise: bool | None = None, apply_bpf: bool = True) -> dict:
    """Blocks 3–8 at full rate: envelope -> analog IF -> BPF -> ADC -> DDC."""
    d = cfg.d
    rng = rng if rng is not None else np.random.default_rng(cfg.scene.seed + 1)
    add_noise = cfg.sim.add_noise if add_noise is None else add_noise
    n_chirps = n_chirps or cfg.sim.if_chirps
    if _has_rain(cfg):
        ch = simulate_channel(cfg, fs=d.fs_ddc, n_chirps=n_chirps, rng=rng, cells=cells)
        env_an = envelope_to_analog(cfg, ch.total.reshape(-1), d.fs_ddc)
    else:
        ch = simulate_channel(cfg, fs=d.fs_analog, n_chirps=n_chirps, rng=rng)
        env_an = ch.total.reshape(-1)
    fe = downconvert(cfg, env_an, rng, add_noise, apply_bpf=apply_bpf)
    ad = adc_sample(cfg, fe["filtered"], rng, add_noise)
    x = ddc(cfg, ad["x"])
    n_rep = d.n_rep_ddc
    frame = IQFrame(data=x[: n_chirps * n_rep].reshape(n_chirps, n_rep), fs=d.fs_ddc,
                    freq_offset=d.f_if_center - d.f_nco, source="sim-if")
    return dict(channel=ch, frontend=fe, adc=ad, frame=frame)


def simulate_baseband(cfg: RadarConfig, rng=None, n_chirps: int | None = None, cells=None,
                      add_noise: bool | None = None, channel=None) -> dict:
    d = cfg.d
    rng = rng if rng is not None else np.random.default_rng(cfg.scene.seed + 2)
    add_noise = cfg.sim.add_noise if add_noise is None else add_noise
    ch = channel or simulate_channel(cfg, fs=d.fs_ddc, n_chirps=n_chirps, rng=rng, cells=cells)
    frame = baseband_path(cfg, ch.total, d.fs_ddc, rng, add_noise)
    return dict(channel=ch, frame=frame)


def make_cells(cfg: RadarConfig):
    if not _has_rain(cfg):
        return None
    return make_rain_cells(cfg, np.random.default_rng(cfg.scene.seed))
