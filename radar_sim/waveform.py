"""Block 1 — waveform generator (TX).

The chirp is described by its complex envelope around the RF/IF centre frequency:
``s(t) = exp(j phi(t))``, ``phi(t) = pi S t^2 - pi B t`` for ``0 <= t < T_chirp``
(instantaneous frequency -B/2 .. +B/2), and zero during the idle time.
The DAC output is the real IF signal ``cos(2 pi f_IF t + phi(t))``.
"""
from __future__ import annotations

import numpy as np

from .config import RadarConfig


def chirp_phase(t, bandwidth: float, t_chirp: float):
    slope = bandwidth / t_chirp
    return np.pi * slope * t ** 2 - np.pi * bandwidth * t


def tx_envelope(t, cfg: RadarConfig):
    """Periodic TX complex envelope (unit amplitude) at absolute times ``t`` (t=0: chirp 0 start)."""
    wf = cfg.waveform
    t_rep = wf.t_chirp + wf.t_idle
    tm = np.mod(t, t_rep)
    on = tm < wf.t_chirp
    return np.where(on, np.exp(1j * chirp_phase(tm, wf.bandwidth, wf.t_chirp)), 0.0)


def reference_chirp(cfg: RadarConfig, fs: float, freq_offset: float = 0.0):
    """One chirp of the TX envelope sampled at ``fs``, shifted by ``freq_offset`` Hz.

    ``freq_offset = f_IF_centre - f_NCO`` makes it match the DDC output.
    """
    n = int(round(cfg.waveform.t_chirp * fs))
    t = np.arange(n) / fs
    return np.exp(1j * (chirp_phase(t, cfg.waveform.bandwidth, cfg.waveform.t_chirp)
                        + 2 * np.pi * freq_offset * t))


def dac_if_waveform(cfg: RadarConfig, fs: float, n_chirps: int = 1, amplitude: float = 1.0):
    """Real IF waveform produced by the DAC (for display / block 1)."""
    d = cfg.d
    n = int(round(n_chirps * d.t_rep * fs))
    t = np.arange(n) / fs
    env = tx_envelope(t, cfg)
    return t, amplitude * np.real(env * np.exp(2j * np.pi * d.f_if_center * t))


def instantaneous_frequency(cfg: RadarConfig, t):
    """Instantaneous RF frequency of the TX chirp (for plots)."""
    d = cfg.d
    wf = cfg.waveform
    tm = np.mod(t, d.t_rep)
    f = d.f_rf_center - wf.bandwidth / 2 + d.slope * tm
    return np.where(tm < wf.t_chirp, f, np.nan)
