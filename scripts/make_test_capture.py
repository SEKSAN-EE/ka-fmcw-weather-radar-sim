#!/usr/bin/env python
"""Write a *simulated* capture in exactly the format expected from the ZCU216.

Use it to test scripts/process_capture.py before real data exists, and hand the generated
.yaml to the FPGA side as the format reference.

    python scripts/make_test_capture.py --config config/lab.yaml --out data/test_lab
    python scripts/make_test_capture.py --config config/weather.yaml --chirps 4 --out data/test_weather
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import load_config  # noqa: E402
from radar_sim.io import CaptureMeta, write_capture  # noqa: E402
from radar_sim.simulate import make_cells, simulate_if_path  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/lab.yaml")
    ap.add_argument("--out", default="data/test_capture")
    ap.add_argument("--chirps", type=int, default=None, help="chirps to simulate (+1 for the offset)")
    ap.add_argument("--format", choices=["adc_real", "ddc_iq"], default="adc_real")
    ap.add_argument("--offset", type=int, default=None, help="drop this many ADC samples at the start "
                    "(random if omitted) so the record does not begin at a chirp boundary")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if cfg.mimo.enabled:
        return make_mimo_capture(cfg, args)
    n = (args.chirps or cfg.waveform.n_chirps) + 1
    out = simulate_if_path(cfg, n_chirps=n, cells=make_cells(cfg))
    rng = np.random.default_rng(123)
    n_rep_adc = int(round(cfg.d.t_rep * cfg.adc.fs))
    off = args.offset if args.offset is not None else int(rng.integers(1, n_rep_adc))
    off -= off % cfg.adc.ddc_decimation if args.format == "ddc_iq" else 0
    if args.format == "adc_real":
        samples = out["adc"]["codes"][off:]
        meta = CaptureMeta(format="adc_real", fs=cfg.adc.fs, adc_bits=cfg.adc.bits,
                           notes=f"SIMULATED from {args.config}; true first chirp at ADC sample "
                                 f"{(n_rep_adc - off) % n_rep_adc}")
    else:
        x = out["frame"].stream()[off // cfg.adc.ddc_decimation:]
        full = 32768.0
        codes = x / (cfg.d.a_fs / np.sqrt(2)) * full
        samples = codes
        meta = CaptureMeta(format="ddc_iq", fs=cfg.d.fs_ddc, f_nco=cfg.d.f_nco, iq_full_scale=full,
                           notes=f"SIMULATED from {args.config}")
    path = write_capture(args.out, samples, meta)
    print(f"wrote {path} ({path.stat().st_size/1e6:.1f} MB) + {path.with_suffix('.yaml').name}")
    print(f"  {meta.notes}")


def make_mimo_capture(cfg, args):
    """MIMO: 4 RX channels of DDC I/Q (baseband model), interleaved sample by sample."""
    from radar_sim.validation import simulate_mimo_dwells
    frames, _ = simulate_mimo_dwells(cfg, 1)
    x = frames[0].data.transpose(1, 0, 2).reshape(cfg.mimo.n_rx, -1)
    n_rep = frames[0].data.shape[-1]
    off = args.offset if args.offset is not None else int(np.random.default_rng(123).integers(1, 2 * n_rep))
    codes = x[:, off:] / (cfg.d.a_fs / np.sqrt(2)) * 32768.0
    first_tx = int(np.ceil(off / n_rep)) % cfg.mimo.n_tx
    meta = CaptureMeta(format="ddc_iq", fs=cfg.d.fs_ddc, f_nco=cfg.d.f_nco, iq_full_scale=32768.0,
                       first_chirp_tx=first_tx,
                       notes=f"SIMULATED MIMO from {args.config}; {cfg.mimo.n_rx} RX interleaved; "
                             f"first complete chirp is TX{first_tx}")
    path = write_capture(args.out, codes, meta)
    print(f"wrote {path} ({path.stat().st_size/1e6:.1f} MB) + {path.with_suffix('.yaml').name}")
    print(f"  {meta.notes}")


if __name__ == "__main__":
    main()
