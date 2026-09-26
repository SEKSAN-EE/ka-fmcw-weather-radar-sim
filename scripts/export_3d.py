#!/usr/bin/env python
"""Export simulation results for the interactive 3D view and build viz/radar_3d.html.

    python scripts/export_3d.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radar_sim import load_config  # noqa: E402
from radar_sim.interpret import KIND_TEXT  # noqa: E402
from radar_sim.processing import process  # noqa: E402
from radar_sim.scene import rain_profile  # noqa: E402
from radar_sim.mimo import wind_profile  # noqa: E402
from radar_sim.processing.mimo import process_mimo  # noqa: E402
from radar_sim.validation import mimo_calibration, simulate_dwells, simulate_mimo_dwells  # noqa: E402


def _clean(a, nd=2):
    return [None if not np.isfinite(x) else round(float(x), nd) for x in np.asarray(a, float)]


def scenario(path, label, hint):
    cfg = load_config(ROOT / path)
    d = cfg.d
    frames, _, _ = simulate_dwells(cfg)
    res = process(frames, cfg)
    per_bin = res["noise"] / res["spectrum"].shape[0]
    sd = 10 * np.log10(res["spectrum"] / per_bin + 1e-12)
    sd = np.clip(np.round(sd), -3, 45).astype(int)
    print(f"  {label}: {res['valid'].sum()} valid bins")
    return cfg, {
        "label": label, "hint": hint,
        "pt_dbm": cfg.frontend.pt_dbm, "nf_eff_db": round(d.nf_eff_db, 2),
        "lna": cfg.frontend.lna.enabled, "n_dwells": res["n_dwells"], "n_chirps": cfg.waveform.n_chirps,
        "dwell_ms": round(d.t_dwell * res["n_dwells"] * 1e3, 1),
        "snr_threshold_db": cfg.processing.snr_threshold_db,
        "meas": {"r": _clean(res["ranges"], 1), "dbz": _clean(res["dbz_corr"]), "mdz": _clean(res["mdz"]),
                 "v": _clean(res["v"]), "valid": [int(x) for x in res["valid"]]},
        "spec": {"v": _clean(res["velocity_axis"], 3), "db": sd.T.reshape(-1).tolist(),
                 "n_r": int(sd.shape[1]), "n_v": int(sd.shape[0])},
    }


def mimo_block():
    """Block 18 results for the 3D view: angle-bin fan (v_r anomaly, dBZ) and the wind profile."""
    cfg = load_config(ROOT / "config/weather_mimo.yaml")
    d, mi = cfg.d, cfg.mimo
    cal = mimo_calibration(cfg, 800.0)
    frames, _ = simulate_mimo_dwells(cfg)
    res = process_mimo(frames, cfg, cal=cal)
    m = res["in_beam"]
    step = 2
    r = res["ranges"][::step]
    v = res["v"][m][:, ::step]
    with np.errstate(all="ignore"):
        anom = v - np.nanmean(v, axis=0, keepdims=True)
    hs = slice(0, None, 5)
    print(f"  MIMO: {int(m.sum())} angle bins x {r.size} ranges")
    return {
        "n_tx": mi.n_tx, "n_rx": mi.n_rx, "tx_spacing": mi.tx_spacing, "rx_spacing": mi.rx_spacing,
        "t_chirp_us": cfg.waveform.t_chirp * 1e6, "v_max": round(d.v_max_mimo, 2),
        "v_max_200us": round(d.lam / (4 * mi.n_tx * 200e-6), 2), "res_deg": 11, "r_max": cfg.processing.r_max,
        "wind_gates": mi.wind_gates, "n_dwells": res["n_dwells"],
        "theta": _clean(res["theta"][m], 2), "theta_eff": _clean(res["theta_eff"][m], 2),
        "r": _clean(r, 1), "anom": [_clean(row, 2) for row in anom], "dbz": [_clean(row, 1) for row in res["dbz"][m][:, ::step]],
        "wind": {"h": _clean(res["ranges"][hs], 0), "u": _clean(res["wind_u"][hs]), "V": _clean(res["wind_V"][hs]),
                 "u_true": _clean(wind_profile(cfg, res["ranges"][hs]))},
    }


def main():
    print("simulating scenarios for the 3D view ...")
    cfg, base = scenario("config/weather.yaml", "Baseline",
                         "PA 20 dBm, ADMV1014 without LNA, 8 dwells (0.1 s)")
    _, lna = scenario("config/weather_lna.yaml", "+ LNA, 0.4 s",
                      "adds a 3 dB-NF LNA and averages 32 dwells")
    d = cfg.d
    r = np.arange(0, 6001, 25.0)
    prof = rain_profile(cfg, r)
    data = {
        "meta": {"f_rf_ghz": d.f_rf_center / 1e9, "f_lo_ghz": round(d.f_lo / 1e9, 5),
                 "f_if_ghz": round(d.f_if_center / 1e9, 5), "lam_mm": round(d.lam * 1e3, 3),
                 "bandwidth_mhz": cfg.waveform.bandwidth / 1e6, "t_chirp_us": cfg.waveform.t_chirp * 1e6,
                 "delta_r": round(d.delta_r, 2), "v_max": round(d.v_max, 2),
                 "beam_deg": cfg.frontend.beamwidth_az_deg, "g_dbi": cfg.frontend.g_tx_dbi,
                 "r_max": cfg.processing.r_max},
        "layers": [{"name": L.name, "kind": L.kind, "r_bottom": L.r_bottom, "r_top": L.r_top,
                    "dbz_bottom": L.dbz_bottom, "dbz_top": L.dbz_top, "dbz_peak": L.dbz_peak,
                    "v_bottom": L.v_bottom, "v_top": L.v_top, "sw": L.sw_bottom,
                    "text": KIND_TEXT.get(L.kind, "").replace("\n", " ")} for L in cfg.scene.rain_layers],
        "profile": {"h": _clean(r, 0), "dbz": _clean(prof["dbz"]), "pia": _clean(prof["pia"]),
                    "v": _clean(prof["v"])},
        "scenarios": [base, lna],
        "mimo": mimo_block(),
    }
    js = json.dumps(data, separators=(",", ":"))
    (ROOT / "viz/scene_data.json").write_text(js)
    tpl = (ROOT / "viz/radar_3d.template.html").read_text()
    (ROOT / "viz/radar_3d.html").write_text(tpl.replace("__SCENE_DATA__", js))
    print(f"wrote viz/radar_3d.html ({len(js)/1024:.0f} kB of data)")


if __name__ == "__main__":
    main()
