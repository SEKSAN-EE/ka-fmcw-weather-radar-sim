"""Brief §5.2: estimated Z, v, sigma_v close to ground truth at good SNR."""
import numpy as np
import pytest

from radar_sim.processing import process
from radar_sim.processing.doppler import moments, pulse_pair
from radar_sim.processing.unfold import unfold_continuity
from radar_sim.scene import expected_measured_dbz, rain_profile, zrnic_series
from radar_sim.validation import simulate_dwells


@pytest.fixture(scope="module")
def weather_result():
    from radar_sim import load_config
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "config/weather.yaml")
    frames, _, _ = simulate_dwells(cfg, 4)
    return cfg, process(frames, cfg)


def test_reflectivity_unbiased_at_high_snr(weather_result):
    cfg, res = weather_result
    exp = expected_measured_dbz(cfg, res["ranges"], cfg.processing.range_window)
    good = res["valid"] & (res["snr"] > 10 ** 0.5) & (res["ranges"] > 300) & np.isfinite(exp)
    err = res["dbz"][good] - exp[good]
    assert good.sum() > 30
    assert abs(np.mean(err)) < 0.7
    assert np.std(err) < 1.5


def test_velocity_and_width_in_rain(weather_result):
    cfg, res = weather_result
    truth = rain_profile(cfg, res["ranges"])
    good = res["valid"] & (res["snr"] > 10) & (res["ranges"] > 300) & (res["ranges"] < 2200)
    assert np.sqrt(np.mean((res["v"][good] - truth["v"][good]) ** 2)) < 0.3
    assert abs(np.median(res["sw"][good] - truth["sw"][good])) < 0.3


def test_pulse_pair_on_synthetic_series():
    rng = np.random.default_rng(3)
    lam, t = 8e-3, 200e-6
    # zrnic_series is the received envelope (+2v/lambda); the beat domain is its conjugate
    s = np.conj(zrnic_series(np.full(200, 1.0), np.full(200, 4.0), np.full(200, 1.0), lam, t, 64, rng)).T
    R0, R1 = pulse_pair(s)

    class Cfg:  # minimal stand-in for moments()
        class d:
            pass
    Cfg.d.lam, Cfg.d.t_rep = lam, t
    _, v, sw, _ = moments(Cfg, R0, R1, 1e-12)
    assert np.median(v) == pytest.approx(4.0, abs=0.2)
    assert np.median(sw) == pytest.approx(1.0, abs=0.25)


def test_unfolding_recovers_aliased_rain():
    v_true = np.linspace(12, 1, 50)            # far range slow, near range fast (> v_nyq = 10)
    v_alias = (v_true + 10) % 20 - 10
    # index 0 = nearest gate; scanning "down" starts in the slow, unaliased far gates
    out = unfold_continuity(v_alias, np.ones(50, bool), 10.0, "down")
    assert np.allclose(out, v_true)
