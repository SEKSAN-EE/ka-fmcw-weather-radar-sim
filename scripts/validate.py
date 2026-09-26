#!/usr/bin/env python
"""Validation plots of brief §5 (items 2–5). Output: out/validation/*.png"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import load_config  # noqa: E402
from radar_sim.linkbudget import brief_link_budget_snr_db, weather_snr_db  # noqa: E402
from radar_sim.plots import CRIT, INK, INK2, MUTED, SERIES, _fig, _note, _psd  # noqa: E402
from radar_sim.processing import process  # noqa: E402
from radar_sim.scene import expected_measured_dbz, rain_profile  # noqa: E402
from radar_sim.simulate import simulate_if_path  # noqa: E402
from radar_sim.validation import simulate_dwells  # noqa: E402

OUT = Path("out/validation")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=130)
    plt.close(fig)
    print(f"  saved {OUT/name}.png")


def uniform_rain(base, dbz_near, dbz_far, r0=100.0, r1=5900.0, v=5.0, sw=1.5):
    return base.replace(scene={"attenuation": False, "rain_layers": [dict(
        name="uniform", kind="rain", r_bottom=r0, r_top=r1, dbz_bottom=dbz_near, dbz_top=dbz_far,
        v_bottom=v, v_top=v, sw_bottom=sw, sw_top=sw)]})


# ------------------------------------------------------------------ §5.3 SNR vs range
def snr_vs_range(base):
    r = np.linspace(200, 6000, 300)
    fig, ax = _fig("§5.3 — SNR vs range for rain of 0/20/35/45 dBZ", 1, 2, size=(13, 4.8), sharey=True)
    for i, dbz in enumerate([0, 20, 35, 45]):
        for pt, ls in [(20, "-"), (0, "--")]:
            ax[0].plot(r / 1000, brief_link_budget_snr_db(base, r, dbz, pt), color=SERIES[i], ls=ls,
                       label=f"{dbz} dBZ" if pt == 20 else None)
    ax[0].axhline(0, color=INK2, lw=1)
    ax[0].axvspan(1, 3, color=MUTED, alpha=0.08, lw=0)
    ax[0].set(xlabel="range (km)", ylabel="SNR (dB)", ylim=(-40, 60),
              title="Brief link budget: NF 10 dB, ΔR 30 m, T_coh 1 ms")
    ax[0].legend(loc="upper right", title="solid Pt 20 dBm · dashed 0 dBm", title_fontsize=8)
    # simulation chain: per-chirp SNR predicted vs measured
    for i, dbz in enumerate([20, 35]):
        cfg = uniform_rain(base, dbz, dbz)
        frames, _, _ = simulate_dwells(cfg, 4)
        res = process(frames, cfg)
        pred = weather_snr_db(cfg, res["ranges"], dbz)
        ax[1].plot(res["ranges"] / 1000, pred, color=SERIES[i + 1], label=f"{dbz} dBZ theory")
        ax[1].plot(res["ranges"] / 1000, 10 * np.log10(np.maximum(res["snr"], 1e-4)), ".",
                   color=SERIES[i + 1], ms=2.5, alpha=0.7, label=f"{dbz} dBZ simulated")
    ax[1].axhline(0, color=INK2, lw=1)
    ax[1].set(xlabel="range (km)", title="This chain, per-chirp SNR (ΔR 15 m, T 200 µs, NF_eff 11.2 dB)")
    ax[1].legend(loc="upper right")
    _note(ax[1], "per-chirp noise BW = 1.5/T = 7.5 kHz vs 1 kHz in the brief\n"
                 "→ ~8.8 dB lower SNR; averaging chirps lowers the\ndetection threshold instead",
          loc="lower left")
    save(fig, "v1_snr_vs_range")


# ------------------------------------------------------------------ §5.4 Nyquist aliasing
def nyquist_aliasing():
    lab = load_config("config/lab.yaml")
    lab = lab.replace(processing={"range_zero_pad": 8})
    cases = [("(a) PLL 8.594 GHz: IF inside zone 3", lab, True),
             ("(b) PLL 8.540 GHz + zone-3 BPF: chirp top cut", lab.replace(
                 lo={"n_pll": 1112}, frontend={"bpf_zone": 3}), True),
             ("(c) PLL 8.540 GHz, no BPF: zone 4 folds in", lab.replace(
                 lo={"n_pll": 1112}, frontend={"bpf_zone": 3}), False)]
    fig, ax = _fig("§5.4 — Nyquist-zone placement of the 1 GHz lab chirp", 2, 3, size=(15, 7))
    for j, (title, cfg, bpf) in enumerate(cases):
        d = cfg.d
        out = simulate_if_path(cfg, n_chirps=2, apply_bpf=bpf)
        f, p = _psd(out["adc"]["x"], d.fs)
        ax[0, j].plot(f / 1e9, p, color=SERIES[0], lw=1)
        ax[0, j].set(title=title, xlabel="frequency after sampling (GHz)", ylabel="PSD (dBm/Hz)")
        extra = "\nnoise of all zones folds in" if not bpf else ""
        _note(ax[0, j], f"IF {d.f_if_lo/1e9:.3f}–{d.f_if_hi/1e9:.3f} GHz\nzone edge 3.75 GHz{extra}",
              loc="lower left")
        res = process(out["frame"], cfg)
        P = np.mean(np.abs(res["profiles_raw"]) ** 2, 0)
        ax[1, j].plot(res["ranges"], 10 * np.log10(P / P.max()), color=SERIES[0])
        ax[1, j].axvline(1.0, color=CRIT, lw=1, ls=":")
        ax[1, j].set(xlabel="range (m)", ylabel="dB rel. peak", ylim=(-70, 3), xlim=(0, 4),
                     title="Range profile (corner at 1 m)")
        k = np.argmax(P * (res["ranges"] > 0.5))
        m3 = P > P[k] / 2
        step = res["ranges"][1] - res["ranges"][0]
        _note(ax[1, j], f"peak {res['ranges'][k]:.3f} m\n−3 dB width ≈ {m3.sum()*step*100:.1f} cm")
    save(fig, "v2_nyquist_aliasing")


# ------------------------------------------------------------------ §5.5 velocity aliasing
def velocity_aliasing(base):
    cfg = base.replace(waveform={"t_idle": 200e-6}, processing={"unfold": True}, sim={"n_dwells": 8})
    d = cfg.d
    frames, _, _ = simulate_dwells(cfg)
    res = process(frames, cfg)
    truth = rain_profile(cfg, np.linspace(1, cfg.processing.r_max, 3000))
    fig, ax = _fig(f"§5.5 — velocity aliasing: T_rep = {d.t_rep*1e6:.0f} µs → v_max = ±{d.v_max:.1f} m/s",
                   1, 2, size=(12, 5.5), sharey=True, gridspec_kw=dict(width_ratios=[1.6, 1]))
    per_bin = res["noise"] / res["spectrum"].shape[0]
    sd = 10 * np.log10(res["spectrum"] / per_bin + 1e-12)
    from radar_sim.plots import SEQ
    im = ax[0].pcolormesh(res["velocity_axis"], res["ranges"] / 1000, sd.T, cmap=SEQ, vmin=-2, vmax=30,
                          shading="auto")
    fig.colorbar(im, ax=ax[0], label="dB above noise")
    ax[0].set(xlabel="Doppler velocity (m/s)", ylabel="height (km)", title="Doppler spectrum (rain wraps around)")
    ax[1].plot(truth["v"], truth["r"] / 1000, color=INK, lw=1, label="truth")
    ax[1].plot(res["v"], res["ranges"] / 1000, ".", color=SERIES[1], ms=3, label="measured (aliased)")
    ax[1].plot(res["v_unfolded"], res["ranges"] / 1000, ".", color=SERIES[0], ms=3,
               label="unfolded (continuity, top→down)")
    for s in (-1, 1):
        ax[1].axvline(s * d.v_max, color=CRIT, ls=":", lw=1)
    ax[1].set(xlabel="v (m/s)", title="Mean velocity", xlim=(-12, 12))
    ax[1].legend(loc="upper left", fontsize=7)
    _note(ax[1], "the melting-layer top (~1.6 m/s)\nis the unaliased reference", loc="lower left")
    save(fig, "v3_velocity_aliasing")


# ------------------------------------------------------------------ §5.2 error vs SNR
def error_vs_snr(base):
    cfg = uniform_rain(base, 42, 12)
    frames, _, _ = simulate_dwells(cfg, 8)
    res = process(frames, cfg)
    exp = expected_measured_dbz(cfg, res["ranges"], cfg.processing.range_window)
    truth = rain_profile(cfg, res["ranges"])
    snr = 10 * np.log10(np.maximum(res["snr"], 1e-4))
    ok = res["valid"] & (res["ranges"] > 300)
    fig, ax = _fig("§5.2 — estimation error vs SNR (8 dwells × 64 chirps)", 1, 3, size=(14, 4.2), sharex=True)
    for a, err, name, lim in [(ax[0], res["dbz"] - exp, "Z error (dB)", 6),
                              (ax[1], res["v"] - truth["v"], "v error (m/s)", 2),
                              (ax[2], res["sw"] - truth["sw"], "σv error (m/s)", 2)]:
        a.plot(snr[ok], err[ok], ".", color=SERIES[0], ms=3)
        bins = np.arange(-8, 32, 4)
        idx = np.digitize(snr[ok], bins)
        centres = [bins[i - 1] + 2 for i in range(1, len(bins)) if np.sum(idx == i) > 3]
        stds = [np.nanstd(err[ok][idx == i]) for i in range(1, len(bins)) if np.sum(idx == i) > 3]
        a.plot(centres, stds, color=SERIES[1], marker="o", ms=4, label="std per 4-dB bin")
        a.plot(centres, -np.array(stds), color=SERIES[1], marker="o", ms=4)
        a.axhline(0, color=INK2, lw=1)
        a.set(xlabel="SNR per chirp (dB)", ylabel=name, ylim=(-lim, lim))
        a.legend(loc="upper right")
    save(fig, "v4_error_vs_snr")


if __name__ == "__main__":
    base = load_config("config/weather.yaml")
    snr_vs_range(base)
    nyquist_aliasing()
    velocity_aliasing(base)
    error_vs_snr(base)
