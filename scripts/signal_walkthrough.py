#!/usr/bin/env python
"""Figures + numbers that explain the signal at the ADC input and what every processing step does.

    python scripts/signal_walkthrough.py            # -> out/walkthrough/*.png, numbers.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, resample, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radar_sim import load_config  # noqa: E402
from radar_sim.adc import adc_sample, ddc, ddc_filter  # noqa: E402
from radar_sim.channel import simulate_channel  # noqa: E402
from radar_sim.config import C, db, w_to_dbm  # noqa: E402
from radar_sim.frontend import downconvert, envelope_to_analog  # noqa: E402
from radar_sim.plots import CRIT, INK, INK2, MUTED, SERIES, _fig, _note  # noqa: E402
from radar_sim.processing import process, theoretical_noise  # noqa: E402
from radar_sim.processing.calibration import delta_r_eff, overlap_loss  # noqa: E402
from radar_sim.processing.dechirp import beat_decimate, beat_filter, dechirp  # noqa: E402
from radar_sim.processing.range_fft import range_fft  # noqa: E402
from radar_sim.scene import expected_measured_dbz, radar_constant, rain_profile, weather_power_density  # noqa: E402
from radar_sim.simulate import make_cells, simulate_baseband  # noqa: E402
from radar_sim.validation import simulate_dwells  # noqa: E402

OUT = ROOT / "out/walkthrough"
R_OHM = 50.0
R_BIN = 1000.0          # the range bin followed through the chain


def mv(x):
    """sqrt(W) amplitude -> mV across 50 ohm."""
    return np.asarray(x) * np.sqrt(R_OHM) * 1e3


def dbm_to_mv_peak(p_dbm):
    return np.sqrt(2 * 10 ** ((p_dbm - 30) / 10) * R_OHM) * 1e3


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=130)
    plt.close(fig)
    print(f"  saved {name}.png")


def adc_input_components(cfg, cells):
    """Leakage, rain and noise at the ADC input, each separately, for one chirp."""
    d = cfg.d
    ch = simulate_channel(cfg, fs=d.fs_ddc, n_chirps=1, rng=np.random.default_rng(5), cells=cells)
    rng = np.random.default_rng(6)
    comp = {}
    for name in ("leakage", "rain"):
        env = envelope_to_analog(cfg, ch.parts[name].reshape(-1), d.fs_ddc)
        fe = downconvert(cfg, env, rng, add_noise=False)
        comp[name] = fe["filtered"]
    fe = downconvert(cfg, np.zeros_like(env), rng, add_noise=True)
    lo_dc = np.sqrt(10 ** ((cfg.frontend.lo_leakage_dbm - 30) / 10))
    for name, x in (("noise", fe["filtered"]), ("rain", comp["rain"]), ("leakage", comp["leakage"])):
        comp[name] = x - np.mean(x)          # LO feed-through (DC) is removed by the BPF anyway
    total = comp["leakage"] + comp["rain"] + comp["noise"]
    ad = adc_sample(cfg, total, np.random.default_rng(7), add_noise=True)
    return comp, total, ad, lo_dc


def main():
    cfg = load_config(ROOT / "config/weather.yaml")
    d = cfg.d
    print(d.summary())
    cells = make_cells(cfg)
    comp, total, ad, _ = adc_input_components(cfg, cells)
    fs_an = d.fs_analog
    t_an = np.arange(total.size) / fs_an
    nums: dict = {}

    # ------------------------------------------------------------------ A1: time domain at ADC input
    fig, ax = _fig("What the ADC sees — one 200 µs chirp at the ADC input (weather scene)", 1, 3,
                   size=(15, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    step = 400
    for name, col in (("leakage", SERIES[1]), ("rain", SERIES[0])):
        env = np.abs(hilbert(comp[name]))[::step]
        ax[0].plot(t_an[::step] * 1e6, w_to_dbm(np.maximum(env ** 2 / 2, 1e-30)), color=col, lw=1,
                   label=f"{'TX leakage' if name == 'leakage' else 'rain echo (all heights)'}")
    p_noise = np.mean(comp["noise"] ** 2)
    ax[0].axhline(w_to_dbm(p_noise), color=MUTED, ls="--", lw=1.2, label="noise rms (whole Nyquist zone)")
    ax[0].axhline(cfg.adc.full_scale_dbm, color=CRIT, ls=":", lw=1.2, label="ADC full scale")
    ax[0].set(xlabel="time in chirp (µs)", ylabel="power at ADC input (dBm)", ylim=(-110, 10),
              title="Envelope of each component")
    ax[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=7.5)
    for p, lab in ((-3, "0 dBm ≈ 316 mV pk"), (-49, "−50 dBm ≈ 1 mV pk"), (-104, "−90 dBm ≈ 10 µV pk")):
        ax[0].text(3, p, lab, fontsize=7.5, color=INK2, va="top")

    t0 = 100e-6
    i0 = int(t0 * fs_an)
    n_seg = int(4e-9 * fs_an)
    seg = total[i0:i0 + n_seg + 1]
    up = 32
    fine = resample(total[i0 - 64:i0 + n_seg + 64], (n_seg + 128) * up)[64 * up:(64 + n_seg) * up + 1]
    tf = (np.arange(fine.size) / (fs_an * up)) * 1e9
    ax[1].plot(tf, mv(fine), color=SERIES[1], lw=1.4, label=f"analog IF ({d.f_if_center/1e9:.3f} GHz)")
    osr = cfg.adc.analog_oversample
    idx = np.arange(0, n_seg + 1)
    idx = idx[(i0 + idx) % osr == 0]
    ax[1].plot(idx / fs_an * 1e9, mv(seg[idx]), "o", color=INK, ms=6, label="ADC samples (2.5 GS/s)")
    ax[1].set(xlabel=f"time from t = {t0*1e6:.0f} µs (ns)", ylabel="voltage at ADC input (mV, 50 Ω)",
              title="Zoom 4 ns: 1.25 carrier cycles per sample")
    ax[1].legend(loc="lower right", fontsize=7.5)

    fs = d.fs
    n_a = 20
    ia = (i0 + (-i0) % osr) // osr
    xs = ad["x"][ia:ia + n_a]
    ts = np.arange(n_a) / fs
    fa = abs(d.f_if_center - round(d.f_if_center / fs) * fs)
    M = np.column_stack([np.cos(2 * np.pi * fa * ts), np.sin(2 * np.pi * fa * ts)])
    coef, *_ = np.linalg.lstsq(M, xs, rcond=None)
    tt = np.linspace(0, ts[-1], 2000)
    ax[2].plot(tt * 1e9, mv(coef[0] * np.cos(2 * np.pi * fa * tt) + coef[1] * np.sin(2 * np.pi * fa * tt)),
               color=SERIES[0], lw=1.2, label=f"alias at {fa/1e6:.1f} MHz")
    ax[2].plot(ts * 1e9, mv(xs), "o", color=INK, ms=4, label="same ADC samples")
    ax[2].set(xlabel="time (ns)", ylabel="mV", title="The samples trace the zone-1 alias")
    ax[2].legend(loc="lower right", fontsize=7.5)
    save(fig, "a1_adc_input_time")
    nums["adc_input"] = {
        "leak_dbm": float(w_to_dbm(np.mean(comp["leakage"] ** 2))),
        "rain_dbm": float(w_to_dbm(np.mean(comp["rain"] ** 2))),
        "noise_zone_dbm": float(w_to_dbm(p_noise)),
        "leak_mv_pk": float(dbm_to_mv_peak(w_to_dbm(np.mean(comp["leakage"] ** 2)))),
        "rain_mv_rms": float(mv(np.sqrt(np.mean(comp["rain"] ** 2)))),
        "noise_mv_rms": float(mv(np.sqrt(p_noise))),
        "fs_mv_pk": float(dbm_to_mv_peak(cfg.adc.full_scale_dbm)),
        "alias_mhz": fa / 1e6,
        "cycles_per_sample": d.f_if_center / fs,
    }

    # ------------------------------------------------------------------ A2: spectrum of each component
    fig, ax = _fig("What the ADC sees — spectrum of each component at the ADC input", 1, 2, size=(14, 4.4))
    for name, col, lab in (("leakage", SERIES[1], "TX leakage"), ("rain", SERIES[0], "rain echo"),
                           ("noise", MUTED, "thermal noise (kT₀·F·G)")):
        f, p = welch(comp[name], fs_an, nperseg=1 << 14, window="blackmanharris")
        ax[0].plot(f / 1e9, w_to_dbm(np.maximum(p, 1e-40)), color=col, lw=1, label=lab)
        f2, p2 = welch(comp[name], fs_an, nperseg=1 << 19, window="blackmanharris")
        m = (f2 > d.f_if_center - 12e6) & (f2 < d.f_if_center + 12e6)
        ax[1].plot((f2[m] - d.f_if_center) / 1e6, w_to_dbm(np.maximum(p2[m], 1e-40)), color=col, lw=1, label=lab)
    for a in ax:
        a.axhline(w_to_dbm(d.n0_adc), color=SERIES[3], ls="--", lw=1.2, label="ADC noise (NSD)")
    ax[0].set(xlabel="frequency (GHz)", ylabel="PSD (dBm/Hz)", ylim=(-200, -60), title="0–5 GHz (BPF passes zone 3)")
    ax[1].set(xlabel=f"offset from {d.f_if_center/1e9:.5f} GHz (MHz)", ylabel="PSD (dBm/Hz)", ylim=(-200, -60),
              title="Zoom on the 10 MHz chirp band")
    ax[0].legend(loc="upper right", fontsize=7.5)
    _note(ax[1], "in the chirp band the rain is only a few dB above noise\nand ~70 dB below the leakage; echoes from all heights\nshare the same 10 MHz, so only dechirp + FFT separates them",
          loc="lower left")
    save(fig, "a2_adc_input_spectrum")

    # ------------------------------------------------------------------ A3: codes and LSB budget
    lsb = 2 * d.a_fs / 2 ** cfg.adc.bits
    frames, _, _ = simulate_dwells(cfg, first_cells=cells)
    res = process(frames, cfg)
    k1 = int(np.argmin(np.abs(res["ranges"] - R_BIN)))
    r1 = float(res["ranges"][k1])
    p_bin_theory = float(weather_power_density(cfg, [r1], rain_profile(cfg, [r1])["z"],
                                               rain_profile(cfg, np.linspace(0, r1, 400))["pia"][-1])[0]
                         * delta_r_eff(cfg) * d.g_chain * overlap_loss(cfg, [r1])[0])
    fig, ax = _fig("ADC codes — how many LSB each part of the signal occupies", 1, 2, size=(14, 4.2),
                   gridspec_kw=dict(width_ratios=[1.4, 1]))
    n_show = 80
    ax[0].stem(np.arange(n_show), ad["codes"][ia:ia + n_show], linefmt=SERIES[0], markerfmt="o", basefmt=" ")
    ax[0].set(xlabel="sample index n (1 sample = 0.4 ns)", ylabel="ADC code (14-bit, ±8192)",
              title="First 80 codes from t = 100 µs")
    items = [("ADC full scale (peak)", 2 ** (cfg.adc.bits - 1)),
             ("TX leakage (peak)", np.sqrt(2 * np.mean(comp["leakage"] ** 2)) / lsb),
             ("noise, whole zone (rms)", np.sqrt(p_noise) / lsb),
             ("rain, all heights (rms)", np.sqrt(np.mean(comp["rain"] ** 2)) / lsb),
             (f"rain, one 15 m bin at {r1/1000:.1f} km (rms)", np.sqrt(p_bin_theory) / lsb),
             ("1 LSB", 1.0)]
    y = np.arange(len(items))[::-1]
    vals = [v for _, v in items]
    ax[1].barh(y, vals, color=[CRIT, SERIES[1], MUTED, SERIES[0], SERIES[2], INK2], height=0.55)
    ax[1].set_xscale("log")
    ax[1].set_xlim(0.01, 3e4)
    for yi, (lab, v) in zip(y, items):
        ax[1].text(0.012, yi + 0.34, lab, fontsize=8, color=INK2, va="bottom")
        ax[1].text(v * 1.15, yi, f"{v:,.2f}" if v < 10 else f"{v:,.0f}", fontsize=8, va="center")
    ax[1].set(yticks=[], xlabel="amplitude in LSB (log scale)", title="LSB budget")
    save(fig, "a3_adc_codes")
    nums["lsb"] = {"lsb_uV": float(mv(lsb) * 1e3), **{lab: float(v) for lab, v in items}}

    # ------------------------------------------------------------------ A4: SNR at each stage for one bin
    n0 = d.n0_total
    h_ddc = ddc_filter(cfg)
    h_beat, cutoff = beat_filter(cfg)
    noise_stage = [
        ("ADC input\n(Nyquist zone, 1.25 GHz)", n0 * d.zone_width),
        (f"after DDC\n({d.fs_ddc/1e6:g} MS/s)", n0 * d.fs * np.sum(h_ddc ** 2)),
        (f"after dechirp + LPF\n({d.fs_beat/1e6:g} MS/s)", n0 * d.fs_ddc * np.sum(h_beat ** 2)),
        (f"range FFT bin\n(ENBW {1.5/cfg.waveform.t_chirp/1e3:.1f} kHz)", theoretical_noise(cfg)),
    ]
    spec_k = res["spectrum"][:, k1]
    per_bin = res["noise"] / spec_k.size
    spec_snr = float(db((spec_k.max() - per_bin) / per_bin))
    snr = [float(db(p_bin_theory / n)) for _, n in noise_stage]
    labels = [s for s, _ in noise_stage] + [f"Doppler spectrum peak\n({res['n_dwells']}×{cfg.waveform.n_chirps} chirps)"]
    snr.append(spec_snr)
    fig, ax = _fig(f"SNR of the rain echo from one range bin ({r1/1000:.2f} km, 15 m) along the chain", size=(13, 4.6))
    x = np.arange(len(snr))
    ax.bar(x, np.array(snr) + 60, bottom=-60, color=[SERIES[0] if s < 0 else SERIES[2] for s in snr], width=0.55)
    ax.axhline(0, color=INK, lw=1)
    for xi, s in zip(x, snr):
        ax.text(xi, s + (1.5 if s >= 0 else -4), f"{s:+.1f} dB", ha="center", fontsize=9, color=INK)
    for xi in range(len(snr) - 1):
        ax.annotate(f"gain +{snr[xi+1]-snr[xi]:.1f} dB", (xi + 0.5, 33), ha="center", fontsize=8, color=INK2)
        ax.annotate("", (xi + 0.8, 29), (xi + 0.2, 29), arrowprops=dict(arrowstyle="->", color=MUTED))
    meas = float(db(res["snr"][k1]))
    ax.plot([3], [meas], "D", color=CRIT, ms=7)
    ax.text(3.32, meas - 6, f"measured in the\nsimulation: {meas:+.1f} dB", fontsize=8, color=CRIT, va="center")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set(ylabel="SNR (dB)", ylim=(-60, 40))
    save(fig, "a4_snr_budget")
    nums["snr_budget"] = {"range_m": r1, "p_bin_dbm": float(w_to_dbm(p_bin_theory)),
                          "stages": [{"stage": l.replace("\n", " "), "snr_db": s} for l, s in zip(labels, snr)],
                          "measured_bin_snr_db": meas}

    # ------------------------------------------------------------------ B1: DDC output I/Q
    x_ddc = ddc(cfg, ad["x"])
    fig, ax = _fig(f"DDC output — complex baseband I/Q at {d.fs_ddc/1e6:g} MS/s", 1, 2, size=(14, 4))
    n2 = int(2e-6 * d.fs_ddc)
    tb = np.arange(n2) / d.fs_ddc * 1e6
    ax[0].plot(tb, mv(x_ddc[:n2].real), color=SERIES[0], marker="o", ms=2.5, lw=1, label="I")
    ax[0].plot(tb, mv(x_ddc[:n2].imag), color=SERIES[1], marker="o", ms=2.5, lw=1, label="Q")
    ax[0].set(xlabel="time from chirp start (µs)", ylabel="mV (50 Ω equivalent)",
              title="First 2 µs: I and Q are 90° apart")
    ax[0].legend(loc="upper right")
    ph = np.unwrap(np.angle(x_ddc[:d.n_chirp_ddc]))
    fi = np.convolve(np.diff(ph), np.ones(25) / 25, mode="same") * d.fs_ddc / (2 * np.pi)
    ax[1].plot(np.arange(fi.size) / d.fs_ddc * 1e6, fi / 1e6, color=SERIES[0])
    ax[1].set(xlabel="time (µs)", ylabel="instantaneous frequency (MHz)", ylim=(-7, 7),
              title="Phase slope: the chirp now sweeps −5 → +5 MHz")
    _note(ax[1], "dominated by the TX leakage;\nthe rain echoes are the same chirp,\ndelayed and 70 dB weaker",
          loc="upper left")
    save(fig, "b1_ddc_iq")

    # ------------------------------------------------------------------ B2: FMCW time–frequency picture
    fig, ax = _fig("Dechirp idea — echo = delayed copy of the TX chirp; the delay becomes a frequency", 1, 2,
                   size=(14, 4.4))
    T, B, S = cfg.waveform.t_chirp, cfg.waveform.bandwidth, d.slope
    t = np.linspace(0, 2 * T, 4000)
    ftx = (np.mod(t, T)) * S - B / 2
    ax[0].plot(t * 1e6, ftx / 1e6, color=INK, lw=2, label="TX / reference")
    for i, R in enumerate((1000, 3000, 5000)):
        tau = 2 * R / C
        frx = np.mod(t - tau, T) * S - B / 2
        ax[0].plot(t * 1e6, frx / 1e6, color=SERIES[i], lw=1.3, ls="--", label=f"echo {R/1000:g} km (τ = {tau*1e6:.1f} µs)")
    ax[0].set(xlabel="time (µs)", ylabel="baseband frequency (MHz)", title="Frequency vs time")
    ax[0].legend(loc="lower right", fontsize=7.5)
    ax[0].annotate("", (150, (150e-6 * S - B / 2) / 1e6), (150, ((150e-6 - 2 * 5000 / C) * S - B / 2) / 1e6),
                   arrowprops=dict(arrowstyle="<->", color=SERIES[2]))
    ax[0].text(152, 1.8, "f_b = S·τ", color=SERIES[2], fontsize=9)
    for i, R in enumerate((1000, 3000, 5000)):
        tau = 2 * R / C
        tb_ = np.linspace(tau, T, 50)
        ax[1].plot(tb_ * 1e6, np.full_like(tb_, S * tau / 1e6), color=SERIES[i], lw=2.2,
                   label=f"{R/1000:g} km → {S*tau/1e3:.0f} kHz")
        ax[1].plot([0, tau * 1e6], [(S * tau - B) / 1e6 * 0 + S * tau / 1e6] * 2, color=SERIES[i], lw=0.8, ls=":")
    ax[1].set(xlabel="time (µs)", ylabel="beat frequency (MHz)", ylim=(0, 2.2),
              title="After dechirp: one constant tone per range")
    ax[1].legend(loc="upper right", fontsize=8)
    _note(ax[1], "dotted: first τ seconds carry the previous\nchirp's tail → out-of-band, filtered out",
          loc="lower right")
    save(fig, "b2_fmcw_tf")

    # ------------------------------------------------------------------ B3: dechirp demo with 3 targets
    demo = cfg.replace(scene={"kind": "point", "point_targets": [
        {"name": "A", "range_m": 1000.0, "rcs_dbsm": 10.0, "velocity": 0.0},
        {"name": "B", "range_m": 2500.0, "rcs_dbsm": 20.0, "velocity": 0.0},
        {"name": "C", "range_m": 4000.0, "rcs_dbsm": 25.0, "velocity": 0.0}]})
    out = simulate_baseband(demo, n_chirps=1, add_noise=True, rng=np.random.default_rng(1))
    fr = out["frame"]
    beat = dechirp(demo, fr.data, fr.fs, fr.freq_offset)
    blp = beat_decimate(demo, beat)
    prof, rr = range_fft(demo, blp)
    fig, ax = _fig("Dechirp + range FFT on three point targets (1, 2.5, 4 km) + leakage + noise", 1, 2, size=(14, 4.2))
    tb = np.arange(blp.shape[1]) / d.fs_beat * 1e6
    leak_dc = np.mean(blp[0])
    ax[0].plot(tb, mv((blp[0] - leak_dc).real) * 1e3, color=SERIES[0], lw=0.8)
    ax[0].set(xlabel="time (µs)", ylabel="Re{beat} − leakage DC (µV)",
              title=f"Beat signal after LPF ({d.fs_beat/1e6:g} MS/s, {blp.shape[1]} samples)")
    ax[1].plot(rr / 1000, w_to_dbm(np.abs(prof[0]) ** 2), color=SERIES[0])
    for tg in demo.scene.point_targets:
        kk = np.argmin(np.abs(rr - tg.range_m))
        ax[1].annotate(f"{tg.name}: {tg.range_m/1000:g} km\nf_b = {d.slope*2*tg.range_m/C/1e3:.0f} kHz",
                       (rr[kk] / 1000, w_to_dbm(np.abs(prof[0, kk]) ** 2)), xytext=(8, 4), textcoords="offset points",
                       fontsize=8, color=INK2)
    ax[1].axhline(w_to_dbm(theoretical_noise(demo)), color=MUTED, ls=":", lw=1)
    ax[1].set(xlabel="range (km)", ylabel="bin power (dBm)", title="|FFT|²: each tone becomes a peak at R = c·f_b/(2S)")
    save(fig, "b3_dechirp_demo")

    # ------------------------------------------------------------------ B4: slow time of one bin → pulse pair
    xk = res["profiles"][:, k1]
    R0, R1 = res["R0"][k1], res["R1"][k1]
    fig, ax = _fig(f"Slow time: the same range bin ({r1/1000:.2f} km) over {xk.size} chirps → Doppler moments",
                   1, 3, size=(15, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1, 1.3]))
    m = np.arange(xk.size)
    ax[0].plot(m, mv(xk.real) * 1e3, color=SERIES[0], marker="o", ms=3, lw=1, label="I")
    ax[0].plot(m, mv(xk.imag) * 1e3, color=SERIES[1], marker="o", ms=3, lw=1, label="Q")
    ax[0].set(xlabel=f"chirp index m (T = {d.t_rep*1e6:g} µs)", ylabel="X_m[k] (µV)", title="Complex value of the bin, chirp by chirp")
    ax[0].legend(loc="upper right")
    z = mv(xk) * 1e3
    ax[1].plot(z.real, z.imag, color=MUTED, lw=0.6)
    ax[1].scatter(z.real, z.imag, c=m, cmap="Blues", s=16, zorder=3)
    r1v = mv(np.sqrt(np.abs(R1))) * 1e3 * np.exp(1j * np.angle(R1))
    ax[1].annotate("", (r1v.real * 1.6, r1v.imag * 1.6), (0, 0), arrowprops=dict(arrowstyle="->", color=CRIT, lw=2))
    ax[1].text(r1v.real * 1.7, r1v.imag * 1.7, f"R(1)\narg = {np.degrees(np.angle(R1)):.0f}°", color=CRIT, fontsize=8)
    ax[1].set(xlabel="I (µV)", ylabel="Q (µV)", title="Phasor rotates chirp to chirp", aspect="equal")
    vax = res["velocity_axis"]
    ax[2].plot(vax, w_to_dbm(np.maximum(spec_k, 1e-30)), color=SERIES[0], marker="o", ms=3, lw=1,
               label=f"Doppler spectrum (avg of {res['n_dwells']} dwells)")
    ax[2].axhline(w_to_dbm(per_bin), color=MUTED, ls=":", lw=1, label="noise per Doppler bin")
    vt = rain_profile(cfg, [r1])["v"][0]
    ax[2].axvline(res["v"][k1], color=CRIT, lw=1.2, label=f"pulse-pair v = {res['v'][k1]:.2f} m/s")
    ax[2].axvline(vt, color=INK, lw=1, ls="--", label=f"truth v = {vt:.2f} m/s")
    ax[2].set(xlabel="radial velocity (m/s, + toward radar)", ylabel="power per bin (dBm)", title="Doppler spectrum of this bin")
    ax[2].legend(loc="lower left", fontsize=7)
    save(fig, "b4_slow_time")

    # ------------------------------------------------------------------ numbers for the calibration table
    S = float(res["power"][k1])
    G = d.g_chain
    ov = float(overlap_loss(cfg, [r1])[0])
    Cr = radar_constant(cfg, delta_r_eff(cfg))
    z_lin = float(res["z"][k1])
    exp_dbz = float(expected_measured_dbz(cfg, res["ranges"], cfg.processing.range_window)[k1])
    nums["calibration"] = {
        "range_m": r1, "R0_dbm": float(w_to_dbm(R0)), "noise_dbm": float(w_to_dbm(res["noise"])),
        "S_dbm": float(w_to_dbm(S)), "snr_db": float(db(res["snr"][k1])), "g_chain_db": float(db(G)),
        "P_ant_dbm": float(w_to_dbm(S / G)), "overlap_loss_db": float(db(ov)),
        "delta_r_eff_m": float(delta_r_eff(cfg)), "radar_const_db": float(db(Cr)),
        "r2_db": float(db(r1 ** 2)), "dbz": float(db(z_lin)), "dbz_expected": exp_dbz,
        "pia_est_db": float(res["pia_est"][k1]), "dbz_corr": float(res["dbz_corr"][k1]),
        "dbz_truth": float(rain_profile(cfg, [r1])["dbz"][0]),
        "rain_rate": float(res["rain_rate"][k1]), "v": float(res["v"][k1]), "sw": float(res["sw"][k1]),
        "v_truth": float(vt), "sw_truth": float(rain_profile(cfg, [r1])["sw"][0]),
        "arg_R1_deg": float(np.degrees(np.angle(R1))), "abs_R1_over_S": float(np.abs(R1) / S),
    }
    nums["rates"] = {
        "adc_samples_per_chirp": int(round(cfg.waveform.t_chirp * d.fs)), "ddc_samples_per_chirp": d.n_chirp_ddc,
        "beat_samples_per_chirp": d.n_beat, "range_bins": int(res["ranges"].size),
        "n_chirps": cfg.waveform.n_chirps, "n_dwells": res["n_dwells"],
    }
    (OUT / "numbers.json").write_text(json.dumps(nums, indent=2))
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    main()
