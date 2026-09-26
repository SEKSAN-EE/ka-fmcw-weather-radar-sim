"""Figures for block 18 (MIMO) and the steps after the angle FFT."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from .config import RadarConfig
from .mimo import array_geometry, element_pattern_2way, tx_schedule, wind_profile
from .plots import CRIT, INK, INK2, MUTED, SEQ, SERIES, _fig, _note

DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a6a0", "#e34948", "#9c1f1f"])


def fig_array(cfg: RadarConfig):
    d, mi = cfg.d, cfg.mimo
    x_tx, x_rx, x_v = array_geometry(cfg)
    fig, ax = _fig(f"Block 18 — {mi.n_tx}TX × {mi.n_rx}RX TDM-MIMO: virtual array and timing", 1, 2,
                   size=(14, 4.2), gridspec_kw=dict(width_ratios=[1.2, 1]))
    a = ax[0]
    a.scatter(x_tx, np.full_like(x_tx, 2), s=120, marker="s", color=SERIES[6], label="TX (spacing 2λ)", zorder=3)
    a.scatter(x_rx, np.full_like(x_rx, 1), s=120, marker="o", color=SERIES[0], label="RX (spacing λ/2)", zorder=3)
    for t in range(mi.n_tx):
        a.scatter(x_v[t * mi.n_rx:(t + 1) * mi.n_rx], np.zeros(mi.n_rx), s=90, marker="D",
                  color=SERIES[[6, 1, 2, 3][t % 4]], zorder=3, label=f"virtual from TX{t+1}")
    for i, x in enumerate(x_v):
        a.text(x, -0.38, f"{i}", ha="center", fontsize=7.5, color=INK2)
    a.set(ylim=(-0.8, 2.6), yticks=[0, 1, 2], yticklabels=["virtual\n(TX + RX)", "RX", "TX"],
          xlabel="position along the array (wavelengths)",
          title=f"x_v = x_tx + x_rx → {x_v.size} elements at λ/2, aperture {x_v[-1]+mi.rx_spacing:g}λ")
    a.legend(loc="upper right", fontsize=7, ncol=2)
    a.grid(axis="y", visible=False)
    b = ax[1]
    sched = tx_schedule(cfg, 12)
    for m, t in enumerate(sched):
        b.add_patch(Rectangle(
            (m * d.t_rep * 1e6, t - 0.35), cfg.waveform.t_chirp * 1e6, 0.7, color=SERIES[[6, 1, 2, 3][t % 4]]))
    b.set(xlim=(0, 12 * d.t_rep * 1e6), ylim=(-0.6, mi.n_tx - 0.4), yticks=range(mi.n_tx),
          yticklabels=[f"TX{t+1}" for t in range(mi.n_tx)], xlabel="time (µs)",
          title=f"TDM: one TX per chirp → each TX every {d.t_tx*1e6:g} µs")
    _note(b, f"v_max = λ/(4·{mi.n_tx}·T) = ±{d.v_max_mimo:.2f} m/s\n(with 200 µs chirps: ±{d.lam/(4*mi.n_tx*200e-6):.2f} m/s)",
          loc="lower right")
    return fig


def _angle_power(res):
    return res["P"].sum(axis=0)          # (A, K) power summed over Doppler


def fig_lab_range_angle(cfg: RadarConfig, res_raw, res_cal, cal, errors):
    th = res_raw["theta"]
    ok = res_raw["angle_ok"]
    r = res_raw["ranges"]
    fig, ax = _fig("Block 18 — range–angle map of two corner reflectors: before / after channel calibration",
                   1, 3, size=(15, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for a, res, title in ((ax[0], res_raw, "uncalibrated"), (ax[1], res_cal, "calibrated")):
        P = _angle_power(res)[ok]
        Pd = 10 * np.log10(P / P.max() + 1e-12)
        im = a.pcolormesh(th[ok], r, Pd.T, cmap=SEQ, vmin=-40, vmax=0, shading="auto")
        for tg in cfg.scene.point_targets:
            a.plot(tg.angle_deg, tg.range_m, "o", mfc="none", mec=CRIT, ms=14, mew=1.2)
        a.set(xlabel="angle (deg)", ylabel="range (m)", title=title, xlim=(-60, 60), ylim=(0.5, 2.2))
    fig.colorbar(im, ax=ax[1], label="dB rel. peak")
    c = ax[2]
    tg = cfg.scene.point_targets[-1]
    k = np.argmin(np.abs(r - tg.range_m))
    for res, col, lab in ((res_raw, MUTED, "uncalibrated"), (res_cal, SERIES[0], "calibrated")):
        P = _angle_power(res)[ok, k]
        c.plot(th[ok], 10 * np.log10(P / P.max() + 1e-12), color=col, label=lab)
    c.axvline(tg.angle_deg, color=CRIT, ls=":", lw=1)
    c.set(xlabel="angle (deg)", ylabel="dB rel. peak", ylim=(-45, 3), xlim=(-60, 60), title=f"angle cut at {tg.range_m} m ({tg.name})")
    c.legend(loc="lower left")
    gt, gr = errors
    true = np.conj((gt[:, None] * gr[None, :]).reshape(-1))     # profiles carry conj(channel) after dechirp
    err = np.degrees(np.angle(cal / true))
    _note(c, f"calibration phase error after estimate:\nmax {np.max(np.abs(err)):.1f}° over 16 channels", loc="upper right")
    return fig


def _edges(c):
    c = np.asarray(c, float)
    mid = (c[1:] + c[:-1]) / 2
    return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])


def fig_sector(cfg: RadarConfig, res):
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    m = res["in_beam"]
    th_e = np.radians(_edges(res["theta"][m]))
    r_e = _edges(res["ranges"])
    X = r_e[None, :] * np.sin(th_e)[:, None] / 1000
    H = r_e[None, :] * np.cos(th_e)[:, None] / 1000
    fig, ax = _fig(f"After the angle FFT — range × angle cells in the vertical plane "
                   f"({res['n_dwells']} dwells, |θ| ≤ {cfg.mimo.max_angle_deg:g}°)", 1, 3, size=(15, 5.2))
    snr = 10 * np.log10(np.maximum(res["snr"][m], 1e-3))
    im0 = ax[0].pcolormesh(X, H, snr, cmap=SEQ, vmin=-5, vmax=30, shading="flat")
    fig.colorbar(im0, ax=ax[0], label="SNR (dB)")
    ax[0].set(title="SNR", xlabel="horizontal distance x (km)", ylabel="height (km)")
    im1 = ax[1].pcolormesh(X, H, res["dbz"][m], cmap=SEQ, vmin=20, vmax=42, shading="flat")
    fig.colorbar(im1, ax=ax[1], label="dBZ")
    ax[1].set(title="Reflectivity (element pattern corrected)", xlabel="x (km)")
    v = res["v"][m]
    anom = v - np.nanmean(v, axis=0, keepdims=True)
    lim = np.nanpercentile(np.abs(anom), 98)
    im2 = ax[2].pcolormesh(X, H, anom, cmap=DIV, vmin=-lim, vmax=lim, shading="flat")
    fig.colorbar(im2, ax=ax[2], label="v_r − mean across the beam (m/s)")
    ax[2].set(title="v_r anomaly across the beam (wind)", xlabel="x (km)")
    for a in ax:
        a.set_aspect("equal")
        a.set_ylim(0, cfg.processing.r_max / 1000)
    _note(ax[2], "wind blows toward +x:\nleft side approaches (+),\nright side recedes (−)", loc="upper left")
    return fig


def fig_angle_cut(cfg: RadarConfig, res, range_m=1000.0):
    k = int(np.argmin(np.abs(res["ranges"] - range_m)))
    m = res["in_beam"]
    th, the = res["theta"][m], res["theta_eff"][m]
    fig, ax = _fig(f"One range ({res['ranges'][k]/1000:.2f} km): angle spectrum and Doppler beam swinging", 1, 3,
                   size=(15, 4.3))
    P = res["P"][:, m, k]
    Pd = 10 * np.log10(P / (res["noise"] / res["P"].shape[0]) + 1e-12)
    im = ax[0].pcolormesh(th, res["velocity_axis"], Pd, cmap=SEQ, vmin=-3, vmax=np.nanpercentile(Pd, 99.5), shading="auto")
    fig.colorbar(im, ax=ax[0], label="dB above noise")
    ax[0].set(xlabel="angle bin (deg)", ylabel="radial velocity (m/s)", title="Doppler spectrum per angle bin")
    snr = 10 * np.log10(np.maximum(res["snr"][m, k], 1e-3))
    ax[1].plot(th, snr, color=SERIES[0], marker="o", ms=3, label="measured SNR")
    f4 = 20 * np.log10(element_pattern_2way(cfg, np.radians(th)))
    ax[1].plot(th, np.nanmax(snr) + f4, color=MUTED, ls="--", label="two-way element pattern")
    ax[1].axhline(cfg.processing.snr_threshold_db, color=CRIT, ls=":", lw=1)
    ax[1].set(xlabel="angle bin (deg)", ylabel="SNR (dB)", title="Power vs angle", ylim=(-15, None))
    ax[1].legend(loc="lower center", fontsize=7.5)
    v = res["v"][m, k]
    ok = np.isfinite(v)
    ax[2].plot(th[ok], v[ok], "o", color=MUTED, ms=4, label="v_r at bin angle θ_i")
    ax[2].plot(the[ok], v[ok], "o", color=SERIES[0], ms=5, label="v_r at effective angle θ_eff")
    tt = np.radians(np.linspace(-20, 20, 50))
    V, u = res["wind_V"][k], res["wind_u"][k]
    if np.isfinite(V):
        ax[2].plot(np.degrees(tt), V * np.cos(tt) - u * np.sin(tt), color=SERIES[1],
                   label=f"fit: V = {V:.2f}, u = {u:.2f} m/s")
    ut = wind_profile(cfg, res["ranges"][k])
    ax[2].set(xlabel="angle (deg)", ylabel="v_r (m/s, + toward radar)", title=f"v_r = V cosθ − u sinθ  (true u = {ut:.1f})")
    ax[2].legend(loc="lower left", fontsize=7.5)
    return fig


def fig_wind(cfg: RadarConfig, runs: dict):
    from .scene import rain_profile
    first = next(iter(runs.values()))
    r = first["ranges"]
    hg = np.linspace(0, cfg.processing.r_max, 400)
    fig, ax = _fig("Wind and fall speed retrieved from the MIMO angle bins (DBS)", 1, 2, size=(12, 5.2), sharey=True)
    ax[0].plot(wind_profile(cfg, hg), hg / 1000, color=INK, lw=1.2, label="truth")
    ax[1].plot(rain_profile(cfg, hg)["v"], hg / 1000, color=INK, lw=1.2, label="truth")
    for (lab, res), col in zip(runs.items(), (SERIES[0], SERIES[1])):
        if col == SERIES[0]:
            ax[0].plot(res["wind_u_gate"], r / 1000, ".", ms=3, color=MUTED, label="single 15 m gate")
            ax[1].plot(res["wind_V_gate"], r / 1000, ".", ms=3, color=MUTED, label="single 15 m gate")
        ax[0].plot(res["wind_u"], r / 1000, "-", lw=1.8, color=col, label=f"{lab}, {cfg.mimo.wind_gates}-gate fit")
        ax[1].plot(res["wind_V"], r / 1000, "-", lw=1.8, color=col, label=f"{lab}, {cfg.mimo.wind_gates}-gate fit")
    ax[0].set(xlabel="horizontal wind u (m/s)", ylabel="height (km)", title="u(h)", xlim=(-5, 15))
    ax[1].set(xlabel="fall speed + vertical air motion V (m/s)", title="V(h)", xlim=(0, 10))
    ax[0].legend(loc="upper left", fontsize=8)
    return fig
