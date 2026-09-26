"""Brief §7.3: the complex-baseband model must match the full IF -> ADC -> DDC chain."""
from radar_sim.simulate import make_cells
from radar_sim.validation import if_vs_baseband


def test_equivalence_lab(lab_cfg):
    _, _, nmse = if_vs_baseband(lab_cfg, n_chirps=1)
    assert nmse < -40


def test_equivalence_weather_rain(weather_cfg):
    _, _, nmse = if_vs_baseband(weather_cfg, make_cells(weather_cfg), n_chirps=1)
    assert nmse < -40


def test_equivalence_with_nco_offset(weather_cfg):
    cfg = weather_cfg.replace(adc={"nco_freq": 3.13e9}, scene={"kind": "point", "point_targets": [
        {"name": "t", "range_m": 800.0, "rcs_dbsm": 10.0, "velocity": 3.0}]})
    _, _, nmse = if_vs_baseband(cfg, n_chirps=1)
    assert nmse < -40
