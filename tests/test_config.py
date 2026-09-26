import numpy as np
import pytest

from radar_sim import load_config


def test_frequency_plan(weather_cfg):
    d = weather_cfg.d
    assert d.f_pll == pytest.approx(1119 * 7.68e6)
    assert d.f_lo == pytest.approx(34.37568e9)
    assert d.zone_lo == d.zone_hi == 3
    assert not d.zone_inverted


def test_v_max_matches_brief(weather_cfg):
    # brief: lambda ~ 8 mm, T_rep = 200 us -> v_max ~ 10 m/s
    assert weather_cfg.d.v_max == pytest.approx(10.0, abs=0.05)


def test_old_lo_plan_straddles_zone(lab_cfg):
    d = lab_cfg.replace(lo={"n_pll": 1112}).d          # 8.54 GHz plan, 1 GHz chirp
    assert d.zone_lo != d.zone_hi
    assert any("straddles" in w for w in d.warnings)


def test_integer_sample_counts(lab_cfg, weather_cfg):
    for cfg in (lab_cfg, weather_cfg):
        d = cfg.d
        assert d.n_chirp_ddc % d.beat_decimation == 0
        assert d.fs_beat > 2 * d.fb_max


def test_yaml_base_inheritance(weather_cfg):
    assert weather_cfg.adc.fs == 2.5e9                 # from default.yaml
    assert len(weather_cfg.scene.rain_layers) == 4     # from weather.yaml
