"""Block 2 — scene / target model.

* Point targets (corner reflector): constant RCS, point-target radar equation.
* Rain / cloud layers: profiles of Z(r), v(r), sigma_v(r). The volume is split into
  cells (``cells_per_bin`` per range bin). Each cell gets a slow-time complex series with a
  Gaussian Doppler spectrum using the spectral method of Zrnic (1975), so Z, v and sigma_v
  are reproduced statistically without tracking individual drops.
* Two-way attenuation from the k-R relation (k in dB/km).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import C, RadarConfig, db, undb


# --------------------------------------------------------------------------- point targets
def point_target_power(cfg: RadarConfig, range_m: float, rcs_dbsm: float) -> float:
    """Received power at the RX antenna port (W): Pt Gt Gr lambda^2 sigma / ((4pi)^3 R^4 L)."""
    d = cfg.d
    sigma = float(undb(rcs_dbsm))
    return d.pt * d.g_tx * d.g_rx * d.lam ** 2 * sigma / ((4 * np.pi) ** 3 * range_m ** 4 * d.loss)


# --------------------------------------------------------------------------- weather profile
def _interp_layer(layer, r):
    x = (r - layer.r_bottom) / (layer.r_top - layer.r_bottom)
    if layer.dbz_peak is None:
        dbz = layer.dbz_bottom + x * (layer.dbz_top - layer.dbz_bottom)
    else:
        dbz = np.where(x < 0.5,
                       layer.dbz_bottom + 2 * x * (layer.dbz_peak - layer.dbz_bottom),
                       layer.dbz_peak + 2 * (x - 0.5) * (layer.dbz_top - layer.dbz_peak))
    v = layer.v_bottom + x * (layer.v_top - layer.v_bottom)
    sw = layer.sw_bottom + x * (layer.sw_top - layer.sw_bottom)
    return dbz, v, sw


def rain_profile(cfg: RadarConfig, r) -> dict:
    """Ground-truth profiles on range grid ``r`` (m).

    Returns dBZ (NaN outside layers), linear Z [mm^6/m^3], v, sigma_v, rain rate R [mm/h],
    specific attenuation k [dB/km] and two-way path-integrated attenuation PIA [dB].
    """
    r = np.asarray(r, dtype=float)
    z = np.zeros_like(r)
    v = np.full_like(r, np.nan)
    sw = np.full_like(r, np.nan)
    k = np.zeros_like(r)
    layer_idx = np.full(r.shape, -1)
    pr, sc = cfg.processing, cfg.scene
    for i, layer in enumerate(sc.rain_layers):
        m = (r >= layer.r_bottom) & (r < layer.r_top)
        if not np.any(m):
            continue
        dbz_l, v_l, sw_l = _interp_layer(layer, r[m])
        z[m] = undb(dbz_l)
        v[m], sw[m] = v_l, sw_l
        rr = (z[m] / pr.zr_a) ** (1 / pr.zr_b)
        k[m] = layer.att_scale * sc.k_a * rr ** sc.k_b
        layer_idx[m] = i
    rain_rate = np.where(z > 0, (z / pr.zr_a) ** (1 / pr.zr_b), 0.0)
    if sc.attenuation:
        dr_km = np.diff(r, prepend=r[0]) / 1000.0
        pia = 2 * np.cumsum(k * dr_km)
    else:
        pia = np.zeros_like(r)
    with np.errstate(divide="ignore"):
        dbz = np.where(z > 0, db(np.maximum(z, 1e-30)), np.nan)
    return dict(r=r, dbz=dbz, z=z, v=v, sw=sw, rain_rate=rain_rate, k=k, pia=pia,
                dbz_att=dbz - pia, layer=layer_idx)


def weather_power_density(cfg: RadarConfig, r, z_lin, pia_db=0.0):
    """Received power per metre of range (W/m) from reflectivity Z [mm^6/m^3] at range r.

    Probert-Jones: P = Pt G^2 theta phi dr pi^3 |K|^2 Z / (512 ln2 lambda^2 r^2 L)
    """
    d = cfg.d
    const = d.pt * d.g_tx * d.g_rx * d.theta * d.phi * np.pi ** 3 * d.k2 / (
        512 * np.log(2) * d.lam ** 2 * d.loss)
    return const * np.asarray(z_lin) * 1e-18 / np.asarray(r) ** 2 * undb(-np.asarray(pia_db))


def radar_constant(cfg: RadarConfig, delta_r_eff: float) -> float:
    """C such that Z[mm^6/m^3] = C * P_ant * r^2 (P_ant: antenna-port power in a range bin)."""
    d = cfg.d
    return (512 * np.log(2) * d.lam ** 2 * d.loss) / (
        d.pt * d.g_tx * d.g_rx * d.theta * d.phi * np.pi ** 3 * d.k2 * delta_r_eff) / 1e-18


# --------------------------------------------------------------------------- rain cells
@dataclass
class RainCells:
    r: np.ndarray          # cell ranges (m)
    power: np.ndarray      # mean power per cell at RX antenna port (W)
    v: np.ndarray          # mean radial velocity (m/s, + toward radar)
    sw: np.ndarray         # spectrum width (m/s)
    series: np.ndarray     # (n_cells, n_chirps) complex slow-time amplitudes, E|a|^2 = power


def zrnic_series(power, v, sw, lam, t_rep, n, rng, oversample=4):
    """Slow-time series with Gaussian Doppler spectra (Zrnic 1975), one row per cell.

    The received envelope of a scatterer moving toward the radar rotates at +2v/lambda.
    """
    power = np.asarray(power, float)[:, None]
    K = int(n * oversample)
    prf = 1.0 / t_rep
    f = np.fft.fftfreq(K, t_rep)[None, :]
    fd = (2 * np.asarray(v, float) / lam)[:, None]
    sf = np.maximum(2 * np.asarray(sw, float) / lam, 0.5 * prf / K)[:, None]
    spec = np.zeros((power.shape[0], K))
    for wrap in (-2, -1, 0, 1, 2):          # aliased Gaussian
        spec += np.exp(-0.5 * ((f - fd + wrap * prf) / sf) ** 2)
    spec *= power / spec.sum(axis=1, keepdims=True)
    amp = np.sqrt(spec * rng.exponential(1.0, spec.shape))
    phase = np.exp(2j * np.pi * rng.random(spec.shape))
    series = np.fft.ifft(amp * phase, axis=1) * K
    start = rng.integers(0, K - n + 1)
    return series[:, start:start + n]


def make_rain_cells(cfg: RadarConfig, rng) -> RainCells | None:
    sc = cfg.scene
    if not sc.rain_layers:
        return None
    d = cfg.d
    dr = d.delta_r / max(sc.cells_per_bin, 1)
    r0 = max(min(l.r_bottom for l in sc.rain_layers), dr)
    r1 = min(max(l.r_top for l in sc.rain_layers), cfg.processing.r_max)
    grid = np.arange(r0, r1, dr)
    r = grid + rng.random(grid.size) * dr
    prof = rain_profile(cfg, np.concatenate([[0.0], r]))      # PIA integrates from the radar
    keep = prof["z"][1:] > 0
    z, v, sw, pia = (prof[k][1:][keep] for k in ("z", "v", "sw", "pia"))
    r = r[keep]
    power = weather_power_density(cfg, r, z, pia) * dr
    series = zrnic_series(power, v, sw, d.lam, d.t_rep, cfg.waveform.n_chirps, rng)
    return RainCells(r=r, power=power, v=v, sw=sw, series=series)


def range_response_kernel(cfg: RadarConfig, window: str, span_bins: float = 8, n: int = 801):
    """|H(dr)|^2 of the range FFT (peak normalised to 1) versus range offset dr (m)."""
    from .config import get_window
    d = cfg.d
    w = get_window(window, d.n_beat)
    dr = np.linspace(-span_bins, span_bins, n) * d.delta_r
    fb = d.slope * 2 * dr / C
    t = np.arange(d.n_beat) / d.fs_beat
    h = np.abs(np.exp(-2j * np.pi * fb[:, None] * t[None, :]) @ w) ** 2 / np.sum(w) ** 2
    return dr, h


def expected_measured_dbz(cfg: RadarConfig, r_bins, window: str):
    """Truth smoothed by the range response and attenuated: what an ideal estimator returns."""
    dr_k, h = range_response_kernel(cfg, window)
    step = dr_k[1] - dr_k[0]
    r_fine = np.arange(max(r_bins[0] - dr_k[-1], step), r_bins[-1] + dr_k[-1], step)
    prof = rain_profile(cfg, r_fine)
    p = prof["z"] * undb(-prof["pia"]) / r_fine ** 2
    num = np.convolve(p, h[::-1], mode="same")
    den = np.convolve(np.ones_like(p), h[::-1], mode="same")
    z_exp = np.interp(r_bins, r_fine, num / den) * np.asarray(r_bins) ** 2
    with np.errstate(divide="ignore"):
        return np.where(z_exp > 0, db(np.maximum(z_exp, 1e-30)), np.nan)
