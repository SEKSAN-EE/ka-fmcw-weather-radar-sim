"""Block 18 (simulation side) — 4TX x 4RX TDM-MIMO channel.

Geometry (positions in wavelengths along the array axis x):
    TX t at x_t = t * tx_spacing,  RX r at x_r = r * rx_spacing
    virtual element v = t * n_rx + r at x_v = x_t + x_r   (tx_spacing = n_rx * rx_spacing -> filled ULA)

A scatterer at angle theta from boresight (in the array plane) reaches the TX->RX pair (t, r) with
delay tau = (2R - (x_t + x_r) * lambda * sin(theta)) / c, i.e. an extra envelope phase
    exp(+j 2 pi (x_t + x_r) sin(theta)).
TDM: chirp m is sent by TX (m mod n_tx); all RX channels record every chirp.

Scene geometry for weather: the radar points up (elevation 90 deg), the array axis is horizontal (x).
A cell at slant range R and angle theta sits at height h = R cos(theta), x = R sin(theta).
Radial velocity (+ toward the radar) = V_fall(h) cos(theta) - u(h) sin(theta), u = wind along +x.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import ChannelOutput, _time_grid, point_echo
from .config import C, RadarConfig, undb
from .scene import point_target_power, rain_profile, weather_power_density, zrnic_series
from .waveform import tx_envelope


# --------------------------------------------------------------------------- geometry
def array_geometry(cfg: RadarConfig):
    mi = cfg.mimo
    x_tx = np.arange(mi.n_tx) * mi.tx_spacing
    x_rx = np.arange(mi.n_rx) * mi.rx_spacing
    x_v = (x_tx[:, None] + x_rx[None, :]).reshape(-1)          # v = t * n_rx + r
    return x_tx, x_rx, x_v


def tx_schedule(cfg: RadarConfig, n_chirps: int):
    if cfg.mimo.scheme != "tdm":
        raise NotImplementedError("only TDM-MIMO is modelled")
    return np.arange(n_chirps) % cfg.mimo.n_tx


def element_pattern_2way(cfg: RadarConfig, theta):
    """Two-way amplitude pattern of one TX/RX sub-array pair (Gaussian, beamwidth in config)."""
    tb = np.radians(cfg.frontend.beamwidth_az_deg)
    return np.exp(-4 * np.log(2) * np.asarray(theta) ** 2 / tb ** 2)   # = f_tx * f_rx (amplitude)


def channel_errors(cfg: RadarConfig, rng=None):
    """Complex gain of each TX and RX channel (simulated hardware mismatch)."""
    mi = cfg.mimo
    rng = rng if rng is not None else np.random.default_rng(cfg.scene.seed + 99)

    def draw(n, g_db, p_deg):
        g = 10 ** (rng.standard_normal(n) * g_db / 20)
        return g * np.exp(1j * np.radians(rng.standard_normal(n) * p_deg))
    gt = draw(mi.n_tx, mi.tx_gain_err_db, mi.tx_phase_err_deg)
    gr = draw(mi.n_rx, mi.rx_gain_err_db, mi.rx_phase_err_deg)
    gt[0] = gr[0] = 1.0
    return gt, gr


def wind_profile(cfg: RadarConfig, h):
    w = cfg.scene.wind
    if not w:
        return np.zeros_like(np.asarray(h, float))
    w = np.asarray(w, float)
    return np.interp(h, w[:, 0], w[:, 1])


# --------------------------------------------------------------------------- rain on a range x angle grid
@dataclass
class RainCells2D:
    r: np.ndarray          # (Nr,) slant range of each range cell
    theta: np.ndarray      # (Nr, Na) angle of each cell (rad)
    power: np.ndarray      # (Nr, Na) mean power at the RX port of one element pair (W)
    v: np.ndarray          # (Nr, Na) radial velocity
    series: np.ndarray     # (Nr, Na, n_chirps)
    v_range: np.ndarray    # (Nr,) power-weighted mean velocity (for intra-chirp Doppler)


def make_rain_cells_2d(cfg: RadarConfig, rng) -> RainCells2D:
    sc, mi, d = cfg.scene, cfg.mimo, cfg.d
    dr = d.delta_r / max(sc.cells_per_bin, 1)
    r0 = max(min(l.r_bottom for l in sc.rain_layers), dr)
    r1 = cfg.processing.r_max
    r = np.arange(r0, r1, dr)
    r = r + rng.random(r.size) * dr
    na = mi.angle_cells
    span = np.radians(mi.angle_span_deg)
    dth = 2 * span / na
    theta = -span + (np.arange(na)[None, :] + rng.random((r.size, na))) * dth
    h = r[:, None] * np.cos(theta)
    # vertical profile on a fine grid (PIA integrates from the ground), slant path factor 1/cos
    hg = np.linspace(0, r1, 4000)
    prof = rain_profile(cfg, hg)
    z = np.interp(h, hg, prof["z"])
    pia = np.interp(h, hg, prof["pia"]) / np.cos(theta)
    vfall = np.interp(h, hg, np.nan_to_num(prof["v"]))
    sw = np.interp(h, hg, np.nan_to_num(prof["sw"], nan=0.3))
    inside = np.interp(h, hg, (prof["layer"] >= 0).astype(float)) > 0.5
    z = np.where(inside, z, 0.0)
    ang_w = element_pattern_2way(cfg, theta) ** 2 * dth / (
        np.radians(cfg.frontend.beamwidth_az_deg) * np.sqrt(np.pi / (8 * np.log(2))))
    power = weather_power_density(cfg, r[:, None], z, pia) * dr * ang_w
    v = vfall * np.cos(theta) - wind_profile(cfg, h) * np.sin(theta)
    series = zrnic_series(power.reshape(-1), v.reshape(-1), sw.reshape(-1), d.lam, d.t_rep,
                          cfg.waveform.n_chirps, rng, oversample=2).reshape(r.size, na, -1)
    wsum = power.sum(axis=1)
    v_range = np.where(wsum > 0, (power * v).sum(axis=1) / np.maximum(wsum, 1e-300), 0.0)
    return RainCells2D(r=r, theta=theta, power=power, v=v, series=series, v_range=v_range)


# --------------------------------------------------------------------------- channel
def simulate_channel_mimo(cfg: RadarConfig, fs: float | None = None, n_chirps: int | None = None,
                          rng=None, cells: RainCells2D | None = None, with_leakage: bool = True,
                          chunk: int = 1024) -> ChannelOutput:
    """Complex envelope at each RX antenna port: shape (n_chirps, n_rx, n_rep)."""
    d, mi, sc = cfg.d, cfg.mimo, cfg.scene
    fs = fs or d.fs_ddc
    n_chirps = n_chirps or cfg.waveform.n_chirps
    rng = rng if rng is not None else np.random.default_rng(sc.seed)
    t, n_rep = _time_grid(cfg, fs, n_chirps)
    sched = tx_schedule(cfg, n_chirps)
    x_tx, x_rx, _ = array_geometry(cfg)
    gt, gr = channel_errors(cfg)
    out = np.zeros((n_chirps, mi.n_rx, n_rep), complex)
    parts = {}

    if sc.kind in ("point", "both") and sc.point_targets:
        acc = np.zeros_like(out)
        for tg in sc.point_targets:
            th = np.radians(tg.angle_deg)
            p = point_target_power(cfg, tg.range_m, tg.rcs_dbsm)
            base = point_echo(cfg, t, tg.range_m, tg.velocity, np.sqrt(p) * element_pattern_2way(cfg, th))
            steer = np.exp(2j * np.pi * (x_tx[sched][:, None] + x_rx[None, :]) * np.sin(th))
            acc += base[:, None, :] * steer[:, :, None]
        parts["targets"] = acc

    if sc.kind in ("rain", "both") and sc.rain_layers:
        if cells is None:
            cells = make_rain_cells_2d(cfg, rng)
        tr = np.arange(n_rep) / fs
        tau = 2 * cells.r / C
        carrier = np.exp(-2j * np.pi * d.f_rf_center * tau)
        fd = 2 * cells.v_range / d.lam
        # B[rho, m, r] = sum over angle cells of a[rho, a, m] * steering(t_m, r, theta)
        sin_t = np.sin(cells.theta)                                        # (Nr, Na)
        ph_tx = np.exp(2j * np.pi * x_tx[sched][None, None, :] * sin_t[:, :, None])   # (Nr, Na, M)
        a = cells.series[:, :, :n_chirps] * ph_tx
        ph_rx = np.exp(2j * np.pi * x_rx[None, None, :] * sin_t[:, :, None])         # (Nr, Na, R)
        B = np.einsum("pam,par->pmr", a, ph_rx).reshape(cells.r.size, -1)          # (Nr, M*R)
        rain = np.empty((n_rep, n_chirps * mi.n_rx), complex)
        for i0 in range(0, n_rep, chunk):
            tt = tr[i0:i0 + chunk, None]
            E = tx_envelope(tt - tau[None, :], cfg) * carrier[None, :] * np.exp(2j * np.pi * fd[None, :] * tt)
            rain[i0:i0 + chunk] = E @ B
        parts["rain"] = rain.T.reshape(n_chirps, mi.n_rx, n_rep)

    if with_leakage:
        amp = np.sqrt(d.pt * undb(-cfg.channel.tx_rx_isolation_db))
        pair_phase = np.exp(2j * np.pi * rng.random((mi.n_tx, mi.n_rx)))     # each TX->RX coupling differs
        fl = 1 + np.sqrt(undb(cfg.channel.leakage_fluct_dbc) / 2) * (
            rng.standard_normal((n_chirps, mi.n_rx)) + 1j * rng.standard_normal((n_chirps, mi.n_rx)))
        tau_l = cfg.channel.leakage_distance_m / C
        env = tx_envelope(t - tau_l, cfg) * np.exp(-2j * np.pi * d.f_rf_center * tau_l)
        parts["leakage"] = amp * (pair_phase[sched] * fl)[:, :, None] * env[:, None, :]

    for p in parts.values():
        out += p
    # hardware mismatch of the active TX and of each RX channel
    out *= (gt[sched][:, None] * gr[None, :])[:, :, None]
    for k in parts:
        parts[k] = parts[k] * (gt[sched][:, None] * gr[None, :])[:, :, None]
    return ChannelOutput(fs=fs, total=out, parts=parts, cells=cells,
                         info={"tx_schedule": sched, "gt": gt, "gr": gr})
