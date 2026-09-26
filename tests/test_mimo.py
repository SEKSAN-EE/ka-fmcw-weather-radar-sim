"""Block 18: TDM-MIMO virtual array, angle FFT, TDM compensation, calibration, wind retrieval."""
from pathlib import Path

import numpy as np
import pytest

from radar_sim import load_config
from radar_sim.mimo import array_geometry, channel_errors, wind_profile
from radar_sim.processing.mimo import estimate_calibration, mimo_profiles, process_mimo, to_virtual
from radar_sim.validation import simulate_mimo_dwells

ROOT = Path(__file__).resolve().parents[1]
NO_ERR = {"tx_gain_err_db": 0, "tx_phase_err_deg": 0, "rx_gain_err_db": 0, "rx_phase_err_deg": 0}


def _peak_angle(res, range_m):
    k = int(np.argmin(np.abs(res["ranges"] - range_m)))
    return res["theta"][int(np.argmax(res["P"].sum(0)[:, k]))]


def test_virtual_array_is_filled_ula():
    cfg = load_config(ROOT / "config/lab_mimo.yaml")
    _, _, x_v = array_geometry(cfg)
    assert np.allclose(np.diff(x_v), 0.5)
    assert x_v.size == 16


@pytest.mark.parametrize("angle", [-25.0, 0.0, 12.0, 30.0])
def test_point_target_angle(angle):
    cfg = load_config(ROOT / "config/lab_mimo.yaml", {"mimo": NO_ERR, "scene": {"point_targets": [
        {"name": "c", "range_m": 1.2, "rcs_dbsm": 10.0, "angle_deg": angle}]}})
    frames, _ = simulate_mimo_dwells(cfg, 1)
    assert _peak_angle(process_mimo(frames, cfg), 1.2) == pytest.approx(angle, abs=1.5)


def test_tdm_compensation_keeps_angle_of_moving_target():
    cfg = load_config(ROOT / "config/weather_mimo.yaml", {"mimo": NO_ERR, "scene": {"kind": "point", "point_targets": [
        {"name": "t", "range_m": 1200.0, "rcs_dbsm": 40.0, "velocity": 7.0, "angle_deg": 10.0}]}})
    frames, _ = simulate_mimo_dwells(cfg, 1)
    on = _peak_angle(process_mimo(frames, cfg), 1200.0)
    off = _peak_angle(process_mimo(frames, cfg.replace(mimo={"doppler_compensation": False})), 1200.0)
    assert on == pytest.approx(10.0, abs=1.5)
    assert abs(off - 10.0) > abs(on - 10.0)


def test_calibration_recovers_channel_errors():
    cfg = load_config(ROOT / "config/lab_mimo.yaml")
    frames, ch = simulate_mimo_dwells(cfg, 1)
    X, r = mimo_profiles(cfg, frames[0])
    cal = estimate_calibration(cfg, to_virtual(cfg, X), int(np.argmin(np.abs(r - 1.0))), 0.0)
    gt, gr = channel_errors(cfg)
    true = np.conj((gt[:, None] * gr[None, :]).reshape(-1))
    assert np.max(np.abs(np.degrees(np.angle(cal / true)))) < 3.0
    assert _peak_angle(process_mimo(frames, cfg, cal=cal), 1.6) == pytest.approx(20.0, abs=1.5)


def test_wind_retrieval():
    cfg = load_config(ROOT / "config/weather_mimo.yaml", {"mimo": NO_ERR, "scene": {"attenuation": False}})
    frames, _ = simulate_mimo_dwells(cfg, 4)
    res = process_mimo(frames, cfg)
    r = res["ranges"]
    ok = np.isfinite(res["wind_u"]) & (r > 200) & (r < 1500)
    assert ok.sum() > 50
    err = res["wind_u"][ok] - wind_profile(cfg, r[ok])
    assert abs(np.median(err)) < 0.8
    assert np.sqrt(np.mean(err ** 2)) < 1.5
