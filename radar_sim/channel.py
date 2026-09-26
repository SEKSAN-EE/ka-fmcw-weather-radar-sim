"""Block 3 — channel: delay, Doppler and TX->RX leakage.

Output is the RF-equivalent complex envelope at the RX antenna port, referenced to the RF
centre frequency: an echo with delay tau is ``A * s(t - tau) * exp(-j 2 pi f_c tau)``.
Units: sqrt(W). Shape: (n_chirps, samples_per_repetition), sample 0 = start of chirp m.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import C, RadarConfig, undb
from .scene import RainCells, make_rain_cells, point_target_power
from .waveform import tx_envelope


@dataclass
class ChannelOutput:
    fs: float
    total: np.ndarray                      # (n_chirps, n_rep)
    parts: dict = field(default_factory=dict)
    cells: RainCells | None = None
    info: dict = field(default_factory=dict)


def _time_grid(cfg: RadarConfig, fs: float, n_chirps: int):
    n_rep = int(round(cfg.d.t_rep * fs))
    t = (np.arange(n_chirps)[:, None] * n_rep + np.arange(n_rep)[None, :]) / fs
    return t, n_rep


def point_echo(cfg: RadarConfig, t, range_m, velocity, amplitude):
    """Echo of a point target; tau varies within the chirp so intra-chirp Doppler is exact."""
    tau = 2 * (range_m - velocity * t) / C
    return amplitude * tx_envelope(t - tau, cfg) * np.exp(-2j * np.pi * cfg.d.f_rf_center * tau)


def leakage(cfg: RadarConfig, t, rng):
    ch = cfg.channel
    d = cfg.d
    amp = np.sqrt(d.pt * undb(-ch.tx_rx_isolation_db))
    n_chirps = t.shape[0]
    fl = np.sqrt(undb(ch.leakage_fluct_dbc) / 2) * (
        rng.standard_normal(n_chirps) + 1j * rng.standard_normal(n_chirps))
    tau = ch.leakage_distance_m / C
    return (amp * (1 + fl))[:, None] * tx_envelope(t - tau, cfg) * np.exp(-2j * np.pi * d.f_rf_center * tau)


def rain_echo(cfg: RadarConfig, cells: RainCells, fs: float, n_chirps: int, chunk: int = 2048):
    """Sum of all rain cells: env[m, n] = sum_c E[n, c] a_c[m] (matrix product, chunked in n)."""
    d = cfg.d
    n_rep = int(round(d.t_rep * fs))
    t = np.arange(n_rep) / fs
    tau = 2 * cells.r / C
    fd = 2 * cells.v / d.lam
    carrier = np.exp(-2j * np.pi * d.f_rf_center * tau)
    A = cells.series[:, :n_chirps]
    out = np.empty((n_chirps, n_rep), complex)
    for i0 in range(0, n_rep, chunk):
        tt = t[i0:i0 + chunk, None]
        E = tx_envelope(tt - tau[None, :], cfg) * carrier[None, :] * np.exp(2j * np.pi * fd[None, :] * tt)
        out[:, i0:i0 + chunk] = (E @ A).T
    return out


def simulate_channel(cfg: RadarConfig, fs: float | None = None, n_chirps: int | None = None,
                     rng=None, cells: RainCells | None = None, with_leakage: bool = True) -> ChannelOutput:
    d = cfg.d
    fs = fs or d.fs_ddc
    n_chirps = n_chirps or cfg.waveform.n_chirps
    rng = rng if rng is not None else np.random.default_rng(cfg.scene.seed)
    t, n_rep = _time_grid(cfg, fs, n_chirps)
    parts, info = {}, {}
    sc = cfg.scene

    if sc.kind in ("point", "both") and sc.point_targets:
        echo = np.zeros_like(t, dtype=complex)
        for tg in sc.point_targets:
            p = point_target_power(cfg, tg.range_m, tg.rcs_dbsm)
            info[f"P_rx {tg.name}"] = p
            echo += point_echo(cfg, t, tg.range_m, tg.velocity, np.sqrt(p))
        parts["targets"] = echo

    if sc.kind in ("rain", "both") and sc.rain_layers:
        if cells is None:
            cells = make_rain_cells(cfg, rng)
        parts["rain"] = rain_echo(cfg, cells, fs, n_chirps)

    if with_leakage:
        parts["leakage"] = leakage(cfg, t, rng)

    total = np.zeros_like(t, dtype=complex)
    for p in parts.values():
        total += p
    return ChannelOutput(fs=fs, total=total, parts=parts, cells=cells, info=info)
