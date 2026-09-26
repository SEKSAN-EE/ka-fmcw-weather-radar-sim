"""IQFrame: the hand-off point between "simulation / hardware" and "processing".

Everything from block 9 on consumes an IQFrame, whether it came from the full IF simulation,
the complex-baseband equivalent model, or a real ZCU216 capture.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class IQFrame:
    data: np.ndarray          # (n_chirps, n_rep) complex, sqrt(W) at the ADC input
    fs: float                 # sample rate after DDC
    freq_offset: float        # f_IF_centre - f_NCO: where the chirp centre sits in the DDC output
    source: str = "sim"
    meta: dict = field(default_factory=dict)

    @property
    def n_chirps(self) -> int:
        return self.data.shape[0]

    def stream(self) -> np.ndarray:
        return self.data.reshape(-1)
