#!/usr/bin/env python
"""Run the whole chain (blocks 1–17) and save one figure per block.

    python scripts/run_pipeline.py --config config/weather.yaml
    python scripts/run_pipeline.py --config config/lab.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import load_config, plots  # noqa: E402
from radar_sim.interpret import fig_interpretation  # noqa: E402
from radar_sim.linkbudget import point_target_snr_db  # noqa: E402
from radar_sim.processing import process  # noqa: E402
from radar_sim.scene import expected_measured_dbz, rain_profile  # noqa: E402
from radar_sim.simulate import _has_rain, make_cells, simulate_if_path  # noqa: E402
from radar_sim.validation import if_vs_baseband, simulate_dwells  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/weather.yaml")
    ap.add_argument("--out", default=None, help="output folder (default out/<config name>)")
    ap.add_argument("--dwells", type=int, default=None, help="override sim.n_dwells")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out or f"out/{cfg.name}")
    out.mkdir(parents=True, exist_ok=True)
    d = cfg.d
    print(f"== {cfg.name} ==\n{d.summary()}")
    (out / "summary.txt").write_text(d.summary() + "\n")
    t0 = time.time()

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=130)
        plt.close(fig)
        print(f"  saved {name}.png")

    cells = make_cells(cfg)
    ifp = simulate_if_path(cfg, cells=cells)
    frames, ch, _ = simulate_dwells(cfg, args.dwells, first_cells=cells)
    res = process(frames, cfg)
    x_if, x_bb, nmse = if_vs_baseband(cfg, cells)
    rain = _has_rain(cfg)
    truth = rain_profile(cfg, np.linspace(1, cfg.processing.r_max, 3000)) if rain else None
    expected = expected_measured_dbz(cfg, res["ranges"], cfg.processing.range_window) if rain else None

    save(plots.fig_waveform(cfg), "01_waveform")
    save(plots.fig_scene(cfg, cells), "02_scene")
    save(plots.fig_channel(cfg, ifp["channel"]), "03_channel")
    save(plots.fig_downconverter(cfg, ifp), "04_downconverter")
    save(plots.fig_noise(cfg, ch), "05_noise")
    save(plots.fig_bpf(cfg, ifp), "06_bpf")
    save(plots.fig_adc(cfg, ifp), "07_adc")
    save(plots.fig_ddc(cfg, ifp), "08_ddc")
    save(plots.fig_equivalence(cfg, x_if, x_bb, nmse), "08b_if_vs_baseband")
    save(plots.fig_dechirp(cfg, res), "09_dechirp")
    save(plots.fig_beat_lpf(cfg, res), "10_beat_lpf")
    save(plots.fig_range(cfg, res, None if rain else cfg.scene.point_targets), "11_range_fft")
    save(plots.fig_clutter(cfg, res), "12_clutter_removal")
    metrics = {"config": cfg.name, "if_vs_baseband_nmse_db": nmse, "noise_est_minus_theory_db":
               float(10 * np.log10(res["noise"] / res["noise_theory"])),
               "adc_clipped_samples": int(ifp["adc"]["clipped"]), "warnings": d.warnings}
    if rain:
        save(plots.fig_doppler(cfg, res, truth), "13_doppler")
        save(plots.fig_detection(cfg, res), "14_detection")
        save(plots.fig_reflectivity(cfg, res, truth, expected), "15_reflectivity")
        save(plots.fig_attenuation(cfg, res, truth), "16_attenuation")
        save(plots.fig_rainrate(cfg, res, truth), "17_rain_rate")
        save(fig_interpretation(cfg, res, truth), "18_spectrogram_interpretation")
        err = res["dbz"] - expected
        tv = rain_profile(cfg, res["ranges"])["v"]            # NaN outside layers -> ignored
        metrics.update(valid_bins=int(res["valid"].sum()), z_bias_db=float(np.nanmean(err)),
                       z_std_db=float(np.nanstd(err)),
                       v_rmse=float(np.sqrt(np.nanmean((res["v"] - tv) ** 2))))
    else:
        P = np.mean(np.abs(res["profiles_raw"]) ** 2, 0)
        r = res["ranges"]
        for tg in cfg.scene.point_targets:
            win = np.abs(r - tg.range_m) < 3 * d.delta_r
            k = np.nonzero(win)[0][np.argmax(P[win])]
            metrics[tg.name] = dict(
                peak_range_m=float(r[k]), error_bins=float((r[k] - tg.range_m) / d.delta_r),
                snr_measured_db=float(10 * np.log10(P[k] / res["noise_theory"])),
                snr_radar_eq_db=point_target_snr_db(cfg, tg.range_m, tg.rcs_dbsm))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"done in {time.time()-t0:.1f} s -> {out}/")
    return res


if __name__ == "__main__":
    main()
