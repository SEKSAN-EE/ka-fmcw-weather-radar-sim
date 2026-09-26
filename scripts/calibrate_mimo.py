#!/usr/bin/env python
"""MIMO channel calibration from a corner reflector at a known angle (usually boresight).

    python scripts/calibrate_mimo.py data/corner_mimo.bin --config config/hw_lab_mimo.yaml --range 1.0 --angle 0

Writes <capture>_cal.npy (16 complex gains, virtual order v = t*n_rx + r) and prints a YAML
snippet for ``mimo.calibration``. Re-measure after any cable / temperature / LO change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import load_config  # noqa: E402
from radar_sim.io import capture_to_frame, load_capture  # noqa: E402
from radar_sim.processing.mimo import estimate_calibration, mimo_profiles, to_virtual  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--config", required=True)
    ap.add_argument("--range", type=float, required=True)
    ap.add_argument("--angle", type=float, default=0.0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"processing": {"clutter": "none"}, "mimo": {"calibration": None}})
    samples, meta = load_capture(args.capture)
    frame, _ = capture_to_frame(samples, meta, cfg)
    X, r = mimo_profiles(cfg, frame)
    k = int(np.argmin(np.abs(r - args.range)))
    Pv = np.mean(np.abs(to_virtual(cfg, X)) ** 2, axis=(0, 1))
    win = np.abs(r - args.range) < 5 * cfg.d.delta_r
    k = np.nonzero(win)[0][np.argmax(Pv[win])]
    cal = estimate_calibration(cfg, to_virtual(cfg, X), int(k), args.angle)
    out = Path(args.capture).with_name(Path(args.capture).stem + "_cal.npy")
    np.save(out, cal)
    print(f"reflector found at {r[k]:.3f} m; saved {out}")
    print("gain (dB):  ", np.round(20 * np.log10(np.abs(cal)), 2))
    print("phase (deg):", np.round(np.degrees(np.angle(cal)), 1))
    print("\nmimo:\n  calibration:")
    for c in cal:
        print(f"    - [{c.real:.6f}, {c.imag:.6f}]")


if __name__ == "__main__":
    main()
