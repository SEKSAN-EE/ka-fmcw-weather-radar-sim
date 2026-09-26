"""Block 12 — TX leakage / static clutter removal."""
from __future__ import annotations

import numpy as np

from ..config import RadarConfig


def remove_static(cfg: RadarConfig, profiles, ranges):
    """Mean subtraction along slow time (zero-Doppler removal) + near-range blanking."""
    pr = cfg.processing
    out = profiles.copy()
    if pr.clutter == "mean":
        out = out - out.mean(axis=0, keepdims=True)
    elif pr.clutter != "none":
        raise ValueError(f"unknown clutter method {pr.clutter!r}")
    if pr.blank_m > 0:
        out[:, ranges < pr.blank_m] = 0
    return out
