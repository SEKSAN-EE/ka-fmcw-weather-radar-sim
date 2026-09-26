"""Real-data path: a simulated capture written in the ZCU216 format must come back aligned."""
import numpy as np
import pytest

from radar_sim.io import CaptureMeta, capture_to_frame, load_capture, write_capture
from radar_sim.processing import process
from radar_sim.simulate import simulate_if_path


@pytest.mark.parametrize("offset", [1234, 20001])
def test_raw_adc_capture_roundtrip(lab_cfg, tmp_path, offset):
    out = simulate_if_path(lab_cfg, n_chirps=5)
    write_capture(tmp_path / "cap", out["adc"]["codes"][offset:], CaptureMeta(format="adc_real", fs=2.5e9))
    samples, meta = load_capture(tmp_path / "cap.bin")
    frame, info = capture_to_frame(samples, meta, lab_cfg)
    n_rep_adc = int(round(lab_cfg.d.t_rep * lab_cfg.adc.fs))
    true_start = ((n_rep_adc - offset) % n_rep_adc) / lab_cfg.adc.ddc_decimation
    assert info["start_fractional"] == pytest.approx(true_start, abs=0.3)
    res = process(frame, lab_cfg)
    P = np.mean(np.abs(res["profiles_raw"]) ** 2, 0)
    k = np.argmax(np.where(res["ranges"] > 0.5, P, 0))
    assert abs(res["ranges"][k] - 1.0) <= lab_cfg.d.delta_r


def test_msb_aligned_codes(lab_cfg, tmp_path):
    codes = np.array([-8192, -1, 0, 1, 8191], dtype=np.int16)
    write_capture(tmp_path / "m", codes, CaptureMeta(format="adc_real", msb_aligned=True))
    back, _ = load_capture(tmp_path / "m.bin")
    assert np.array_equal(back, codes)


def test_mimo_ddc_iq_capture_roundtrip(tmp_path):
    from pathlib import Path
    from radar_sim import load_config
    from radar_sim.processing.mimo import process_mimo
    from radar_sim.validation import simulate_mimo_dwells
    cfg = load_config(Path(__file__).resolve().parents[1] / "config/lab_mimo.yaml",
                      {"mimo": {"tx_gain_err_db": 0, "tx_phase_err_deg": 0, "rx_gain_err_db": 0,
                                "rx_phase_err_deg": 0}})
    frames, _ = simulate_mimo_dwells(cfg, 1)
    x = frames[0].data.transpose(1, 0, 2).reshape(cfg.mimo.n_rx, -1)     # (n_rx, N) continuous streams
    n_rep = frames[0].data.shape[-1]
    off = 3 * n_rep + 777                                                # starts inside chirp 3 (TX3)
    codes = x[:, off:] / (cfg.d.a_fs / np.sqrt(2)) * 32768
    first_tx = int(np.ceil(off / n_rep)) % cfg.mimo.n_tx
    write_capture(tmp_path / "m", codes, CaptureMeta(format="ddc_iq", fs=cfg.d.fs_ddc, f_nco=cfg.d.f_nco,
                                                     first_chirp_tx=first_tx))
    samples, meta = load_capture(tmp_path / "m.bin")
    assert samples.shape[0] == 4
    frame, info = capture_to_frame(samples, meta, cfg)
    assert frame.data.shape[1] == 4 and frame.n_chirps % 4 == 0
    res = process_mimo(frame, cfg)
    for tg in cfg.scene.point_targets:
        k = int(np.argmin(np.abs(res["ranges"] - tg.range_m)))
        assert res["theta"][int(np.argmax(res["P"].sum(0)[:, k]))] == pytest.approx(tg.angle_deg, abs=1.5)
