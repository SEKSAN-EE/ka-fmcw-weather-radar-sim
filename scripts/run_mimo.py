#!/usr/bin/env python
"""Block 18 — MIMO angle processing and the steps after it.

    python scripts/run_mimo.py            # -> out/mimo/*.png, metrics.json

1. lab_mimo:     two corner reflectors (0 deg, +20 deg); estimate channel calibration from corner A
2. weather_mimo: calibration capture (strong reflector at boresight) -> rain with wind ->
                 range x angle moments -> wind / fall-speed retrieval (Doppler beam swinging)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radar_sim import load_config, plots_mimo  # noqa: E402
from radar_sim.mimo import channel_errors, wind_profile  # noqa: E402
from radar_sim.processing.mimo import estimate_calibration, mimo_profiles, process_mimo, to_virtual  # noqa: E402
from radar_sim.scene import rain_profile  # noqa: E402
from radar_sim.validation import mimo_calibration, simulate_mimo_dwells  # noqa: E402

OUT = ROOT / "out/mimo"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=130)
    plt.close(fig)
    print(f"  saved {name}.png")


def main():
    metrics = {}
    # ------------------------------------------------------------------ lab: two corners
    lab = load_config(ROOT / "config/lab_mimo.yaml")
    print(lab.d.summary())
    save(plots_mimo.fig_array(lab), "m1_array_tdm")
    frames, ch = simulate_mimo_dwells(lab, 1)
    res_raw = process_mimo(frames, lab)
    X, r = mimo_profiles(lab, frames[0])
    cal = estimate_calibration(lab, to_virtual(lab, X), int(np.argmin(np.abs(r - 1.0))), 0.0)
    res_cal = process_mimo(frames, lab, cal=cal)
    save(plots_mimo.fig_lab_range_angle(lab, res_raw, res_cal, cal, (ch.info["gt"], ch.info["gr"])), "m2_lab_range_angle")
    for tg in lab.scene.point_targets:
        k = int(np.argmin(np.abs(res_cal["ranges"] - tg.range_m)))
        a = int(np.argmax(res_cal["P"].sum(0)[:, k]))
        metrics[f"lab {tg.name}"] = {"true_deg": tg.angle_deg, "est_deg": float(res_cal["theta"][a])}

    # ------------------------------------------------------------------ weather with wind
    wx = load_config(ROOT / "config/weather_mimo.yaml")
    print(wx.d.summary())
    save(plots_mimo.fig_array(wx), "m1_array_tdm_weather")
    cal_w = mimo_calibration(wx, 800.0)
    frames, _ = simulate_mimo_dwells(wx)
    res_w = process_mimo(frames, wx, cal=cal_w)
    res_w_raw = process_mimo(frames, wx)
    save(plots_mimo.fig_sector(wx, res_w), "m3_range_angle_sector")
    save(plots_mimo.fig_angle_cut(wx, res_w, 1000.0), "m4_angle_cut_dbs")
    save(plots_mimo.fig_wind(wx, {"calibrated": res_w, "without calibration": res_w_raw}), "m5_wind_profile")
    r = res_w["ranges"]
    ut = wind_profile(wx, r)
    vt = rain_profile(wx, r)["v"]
    for lab_, res in (("calibrated", res_w), ("uncalibrated", res_w_raw)):
        ok = np.isfinite(res["wind_u"]) & (r < 2400)
        metrics[f"wind {lab_}"] = {
            "gates": int(ok.sum()),
            "u_bias": float(np.mean(res["wind_u"][ok] - ut[ok])),
            "u_rmse": float(np.sqrt(np.mean((res["wind_u"][ok] - ut[ok]) ** 2))),
            "V_rmse": float(np.sqrt(np.nanmean((res["wind_V"][ok] - vt[ok]) ** 2))),
            "u_rmse_single_gate": float(np.sqrt(np.nanmean((res["wind_u_gate"][ok] - ut[ok]) ** 2)))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
