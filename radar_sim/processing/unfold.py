"""Velocity de-aliasing by range continuity."""
from __future__ import annotations

import numpy as np


def unfold_continuity(v, valid, v_nyq, direction: str = "down", v_ref: float | None = None):
    """Add multiples of 2*v_nyq so each valid gate is closest to the previous valid gate.

    ``direction='down'`` scans from far range toward the radar (vertically pointing: from the
    slowly falling ice at the top down into the fast rain), which is usually the safe choice.
    """
    v = np.array(v, float)
    out = np.full_like(v, np.nan)
    idx = np.nonzero(valid & np.isfinite(v))[0]
    if direction == "down":
        idx = idx[::-1]
    ref = v_ref
    for i in idx:
        if ref is None:
            out[i] = v[i]
        else:
            n = np.round((ref - v[i]) / (2 * v_nyq))
            out[i] = v[i] + 2 * v_nyq * n
        ref = out[i]
    return out
