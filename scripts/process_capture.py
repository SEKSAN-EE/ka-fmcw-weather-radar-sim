#!/usr/bin/env python
"""Process a real (or simulated) ZCU216 capture with the same blocks 8–17 as the simulation.

    python scripts/process_capture.py data/run01.bin --config config/weather.yaml
    python scripts/process_capture.py data/test_lab.bin --config config/lab.yaml --out out/capture_lab

The config must describe the hardware settings used for the capture (LO, chirp, ADC, DDC,
gains). Simulation-only sections (scene) are ignored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import load_config, plots  # noqa: E402
from radar_sim.interpret import fig_interpretation  # noqa: E402
from radar_sim.io import capture_to_frame, load_capture, split_dwells  # noqa: E402
from radar_sim.plots import CRIT, SERIES, _fig, _note, _psd  # noqa: E402
from radar_sim.processing import process  # noqa: E402


def fig_capture_overview(cfg, info, frame):
    d = cfg.d
    fig, ax = _fig("Capture overview — raw ADC, DDC output, chirp alignment", 1, 3, size=(15, 4))
    if "adc_signal" in info:
        f, p = _psd(info["adc_signal"], d.fs)
        ax[0].plot(f / 1e6, p, color=SERIES[0], lw=1)
        ax[0].set(xlabel="frequency after sampling (MHz)", ylabel="PSD (dBm/Hz)", title="Raw ADC spectrum")
        _note(ax[0], f"peak {info['peak_dbfs']:.1f} dBFS\nclipped samples: {info['clipped']}")
    else:
        ax[0].axis("off")
    f, p = _psd(frame.stream(), frame.fs, nper=4096, onesided=False)
    ax[1].plot(f / 1e6, p, color=SERIES[0], lw=1)
    ax[1].set(xlabel="baseband frequency (MHz)", ylabel="PSD (dBm/Hz)", title="DDC output")
    if "correlation" in info:
        c = info["correlation"]
        t = np.arange(c.size) / frame.fs * 1e6
        ax[2].plot(t, 20 * np.log10(c / c.max() + 1e-9), color=SERIES[0], lw=1)
        n_rep = int(round(d.t_rep * frame.fs))
        for k in range(info["start_sample_ddc"], c.size, n_rep):
            ax[2].axvline(k / frame.fs * 1e6, color=CRIT, ls=":", lw=1)
        ax[2].set(xlabel="lag (µs)", ylabel="dB", ylim=(-60, 3), title="Correlation with reference chirp")
        _note(ax[2], f"chirp start at DDC sample {info['start_sample_ddc']}")
    else:
        ax[2].axis("off")
    return fig


def process_mimo_capture(cfg, frame, info, out, args):
    """MIMO branch: (n_chirps, n_rx, n_rep) frame -> range x angle moments and wind."""
    from radar_sim import plots_mimo
    from radar_sim.processing.mimo import process_mimo
    per = min(args.chirps_per_dwell or cfg.waveform.n_chirps, frame.n_chirps)
    per -= per % cfg.mimo.n_tx
    cfg = cfg.replace(waveform={"n_chirps": per})
    dwells = split_dwells(frame, per)
    cal = None
    if args.cal:
        cal = np.load(args.cal)
        print(f"applying channel calibration from {args.cal}")
    elif cfg.mimo.calibration:
        print("applying channel calibration from config mimo.calibration")
    else:
        print("WARNING: no channel calibration (see scripts/calibrate_mimo.py)")
    print(f"{frame.n_chirps} chirps x {frame.data.shape[1]} RX -> {len(dwells)} dwell(s); {info['alignment']}")
    res = process_mimo(dwells, cfg, cal=cal)

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=130)
        plt.close(fig)
        print(f"  saved {name}.png")

    save(plots_mimo.fig_array(cfg), "m1_array_tdm")
    save(plots_mimo.fig_sector(cfg, res), "m3_range_angle_sector")
    k = int(np.nanargmax(np.where(np.isfinite(res["wind_u"]), res["valid"].sum(0), -1))) if np.any(res["valid"]) else 0
    save(plots_mimo.fig_angle_cut(cfg, res, float(res["ranges"][k])), "m4_angle_cut_dbs")
    np.savez_compressed(out / "mimo_moments.npz", range_m=res["ranges"], theta_deg=res["theta"],
                        theta_eff_deg=res["theta_eff"], dbz=res["dbz"], v=res["v"], sw=res["sw"], snr=res["snr"],
                        valid=res["valid"], wind_u=res["wind_u"], wind_V=res["wind_V"])
    print(f"saved {out}/mimo_moments.npz")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="path to .bin (with .yaml sidecar)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chirps-per-dwell", type=int, default=None, help="default: waveform.n_chirps")
    ap.add_argument("--cal", default=None, help="MIMO: .npy of complex channel gains from calibrate_mimo.py")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, {"scene": {"kind": "none", "rain_layers": [], "point_targets": []}})
    out = Path(args.out or f"out/capture_{Path(args.capture).stem}")
    out.mkdir(parents=True, exist_ok=True)
    print(cfg.d.summary())
    samples, meta = load_capture(args.capture)
    print(f"loaded {samples.size} samples, format {meta.format}; notes: {meta.notes}")
    frame, info = capture_to_frame(samples, meta, cfg)
    if cfg.mimo.enabled:
        return process_mimo_capture(cfg, frame, info, out, args)
    per = args.chirps_per_dwell or cfg.waveform.n_chirps
    per = min(per, frame.n_chirps)
    cfg = cfg.replace(waveform={"n_chirps": per})
    dwells = split_dwells(frame, per)
    print(f"{frame.n_chirps} chirps -> {len(dwells)} dwell(s) of {per}; alignment: {info['alignment']} "
          f"(DDC sample {info['start_sample_ddc']})")
    res = process(dwells, cfg)

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=130)
        plt.close(fig)
        print(f"  saved {name}.png")

    save(fig_capture_overview(cfg, info, frame), "00_capture_overview")
    save(plots.fig_dechirp(cfg, res), "09_dechirp")
    save(plots.fig_beat_lpf(cfg, res), "10_beat_lpf")
    save(plots.fig_range(cfg, res), "11_range_fft")
    save(plots.fig_clutter(cfg, res), "12_clutter_removal")
    if per >= 4:
        save(plots.fig_doppler(cfg, res), "13_doppler")
        save(plots.fig_detection(cfg, res), "14_detection")
        save(plots.fig_reflectivity(cfg, res), "15_reflectivity")
        save(fig_interpretation(cfg, res), "18_spectrogram_interpretation")
    P = np.mean(np.abs(res["profiles_raw"]) ** 2, 0)
    r = res["ranges"]
    k = int(np.argmax(np.where(r > cfg.processing.blank_m, P, 0)))
    summary = {"n_chirps": frame.n_chirps, "n_dwells": len(dwells), "start_sample_ddc": info["start_sample_ddc"],
               "strongest_echo_range_m": float(r[k]), "strongest_echo_dbm": float(10 * np.log10(P[k]) + 30),
               "noise_dbm": float(10 * np.log10(res["noise"]) + 30),
               "noise_theory_dbm": float(10 * np.log10(res["noise_theory"]) + 30),
               "valid_bins": int(res["valid"].sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    np.savez_compressed(out / "moments.npz", range_m=r, dbz=res["dbz"], dbz_corr=res["dbz_corr"], v=res["v"],
                        sw=res["sw"], snr=res["snr"], valid=res["valid"], spectrum=res["spectrum"],
                        velocity_axis=res["velocity_axis"])
    return res


if __name__ == "__main__":
    main()
