import numpy as np
import pytest

from radar_sim.adc import adc_sample, ddc
from radar_sim.channel import simulate_channel
from radar_sim.processing import process
from radar_sim.simulate import simulate_baseband


def test_tone_aliases_to_predicted_frequency(lab_cfg):
    d = lab_cfg.d
    n = 1 << 16
    t = np.arange(n) / d.fs_analog
    f_in = 3.3e9                                           # zone 3 -> 0.8 GHz, not inverted
    y = 0.1 * np.cos(2 * np.pi * f_in * t)
    ad = adc_sample(lab_cfg, y, np.random.default_rng(0), add_noise=False)
    X = np.abs(np.fft.rfft(ad["x"] * np.hanning(ad["x"].size)))
    f = np.fft.rfftfreq(ad["x"].size, 1 / d.fs)
    assert f[np.argmax(X)] == pytest.approx(f_in - d.fs, abs=2 * d.fs / ad["x"].size)


def test_zone4_is_inverted(lab_cfg):
    d = lab_cfg.d
    n = 1 << 16
    t = np.arange(n) / d.fs_analog
    f_in = 3.9e9                                           # zone 4 -> 5.0 - 3.9 = 1.1 GHz
    ad = adc_sample(lab_cfg, 0.1 * np.cos(2 * np.pi * f_in * t), np.random.default_rng(0), add_noise=False)
    X = np.abs(np.fft.rfft(ad["x"] * np.hanning(ad["x"].size)))
    f = np.fft.rfftfreq(ad["x"].size, 1 / d.fs)
    assert f[np.argmax(X)] == pytest.approx(2 * d.fs - f_in, abs=2 * d.fs / ad["x"].size)


def test_ddc_power_scaling(weather_cfg):
    d = weather_cfg.d
    n = 400_000
    p = 1e-6
    t = np.arange(n) / d.fs
    x = np.sqrt(2 * p) * np.cos(2 * np.pi * (d.f_nco + 1e6) * t)
    z = ddc(weather_cfg, x)
    assert np.mean(np.abs(z[100:-100]) ** 2) == pytest.approx(p, rel=0.01)


def test_noise_floor_matches_theory(weather_cfg):
    cfg = weather_cfg.replace(scene={"kind": "none"})
    ch = simulate_channel(cfg, with_leakage=False)
    res = process(simulate_baseband(cfg, channel=ch)["frame"], cfg)
    measured = np.mean(np.abs(res["profiles"][:, 5:]) ** 2)
    assert 10 * np.log10(measured / res["noise_theory"]) == pytest.approx(0, abs=0.3)
    assert 10 * np.log10(res["noise"] / res["noise_theory"]) == pytest.approx(0, abs=0.6)
