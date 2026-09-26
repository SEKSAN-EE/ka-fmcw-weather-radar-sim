"""Brief §5.1: corner reflector at 1.00 m +/- 1 bin, SNR consistent with the radar equation."""
import numpy as np

from radar_sim.linkbudget import point_target_snr_db
from radar_sim.processing import process
from radar_sim.simulate import simulate_baseband, simulate_if_path


def _peak(res, cfg, r0):
    P = np.mean(np.abs(res["profiles_raw"]) ** 2, 0)
    r = res["ranges"]
    win = np.abs(r - r0) < 3 * cfg.d.delta_r
    k = np.nonzero(win)[0][np.argmax(P[win])]
    return r[k], 10 * np.log10(P[k] / res["noise_theory"])


def test_corner_reflector_range_and_snr_baseband(lab_cfg):
    res = process(simulate_baseband(lab_cfg)["frame"], lab_cfg)
    r_peak, snr = _peak(res, lab_cfg, 1.0)
    assert abs(r_peak - 1.0) <= lab_cfg.d.delta_r
    # Hann scalloping loss is at most 1.42 dB
    assert abs(snr - point_target_snr_db(lab_cfg, 1.0, 5.0)) < 1.6


def test_corner_reflector_through_full_if_chain(lab_cfg):
    out = simulate_if_path(lab_cfg, n_chirps=2)
    res = process(out["frame"], lab_cfg)
    r_peak, _ = _peak(res, lab_cfg, 1.0)
    assert abs(r_peak - 1.0) <= lab_cfg.d.delta_r
    assert out["adc"]["clipped"] == 0


def test_zero_padding_refines_peak(lab_cfg):
    cfg = lab_cfg.replace(processing={"range_zero_pad": 8})
    res = process(simulate_baseband(cfg)["frame"], cfg)
    r_peak, _ = _peak(res, cfg, 1.0)
    assert abs(r_peak - 1.0) < 0.03
