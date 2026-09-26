#!/usr/bin/env python
"""Corner-reflector calibration: range offset and power offset of the whole RX chain.

Point the radar at a trihedral of known edge length at a known range (brief: ~1 m), capture,
then:

    python scripts/calibrate_corner.py data/corner.bin --config config/lab.yaml --range 1.0 --edge 0.10

Prints ``processing.range_offset_m`` and ``processing.cal_offset_db`` to paste into the config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import load_config  # noqa: E402
from radar_sim.config import C  # noqa: E402
from radar_sim.io import capture_to_frame, load_capture  # noqa: E402
from radar_sim.processing import process  # noqa: E402
from radar_sim.processing.calibration import overlap_loss  # noqa: E402
from radar_sim.scene import point_target_power  # noqa: E402


def trihedral_rcs_dbsm(edge_m: float, lam: float) -> float:
    return 10 * np.log10(4 * np.pi * edge_m ** 4 / (3 * lam ** 2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--config", default="config/lab.yaml")
    ap.add_argument("--range", type=float, required=True, help="true range of the reflector (m)")
    ap.add_argument("--edge", type=float, default=None, help="trihedral edge length (m)")
    ap.add_argument("--rcs-dbsm", type=float, default=None, help="or give the RCS directly")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"processing": {"range_zero_pad": 16, "clutter": "none",
                                                   "range_offset_m": 0.0, "cal_offset_db": 0.0}})
    d = cfg.d
    rcs = args.rcs_dbsm if args.rcs_dbsm is not None else trihedral_rcs_dbsm(args.edge, d.lam)
    samples, meta = load_capture(args.capture)
    frame, _ = capture_to_frame(samples, meta, cfg)
    res = process(frame, cfg)
    P = np.mean(np.abs(res["profiles_raw"]) ** 2, 0)
    r = res["ranges"]
    win = np.abs(r - args.range) < max(0.5, 5 * d.delta_r)
    k = np.nonzero(win)[0][np.argmax(P[win])]
    expected = point_target_power(cfg, args.range, rcs) * d.g_chain * overlap_loss(cfg, [args.range])[0]
    range_offset = r[k] - args.range
    cal = 10 * np.log10(expected / P[k])
    print(f"RCS used: {rcs:.2f} dBsm; peak at {r[k]:.4f} m, power {10*np.log10(P[k])+30:.2f} dBm "
          f"(expected {10*np.log10(expected)+30:.2f} dBm)")
    print("\nprocessing:")
    print(f"  range_offset_m: {range_offset:.4f}")
    print(f"  cal_offset_db: {cal:.2f}")


if __name__ == "__main__":
    main()
