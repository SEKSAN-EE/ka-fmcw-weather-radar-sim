"""Complex-baseband equivalent of blocks 4–8.

Maps the channel envelope directly to what the DDC would output:
    x = mu * sqrt(G) * env(t) * exp(j 2 pi (f_IF - f_NCO) t) + complex noise,
with noise PSD = thermal (incl. image leakage) + ADC NSD. Quantisation is represented by the
ADC NSD; clipping is not modelled (use the IF path / headroom check for that).
``tests/test_equivalence.py`` checks this against the full IF simulation.
"""
from __future__ import annotations

import numpy as np

from .config import RadarConfig
from .iq import IQFrame


def baseband_path(cfg: RadarConfig, env, fs_env: float, rng, add_noise: bool = True) -> IQFrame:
    d = cfg.d
    if abs(fs_env - d.fs_ddc) > 1e-6:
        raise ValueError("baseband path expects the channel envelope at the DDC output rate")
    n_chirps, n_rep = env.shape[0], env.shape[-1]
    t = (np.arange(n_chirps)[:, None] * n_rep + np.arange(n_rep)[None, :]) / fs_env
    if env.ndim == 3:                      # MIMO: (n_chirps, n_rx, n_rep), independent noise per RX
        t = t[:, None, :]
    df = d.f_if_center - d.f_nco
    x = d.iq_mu * np.sqrt(d.g_chain) * env * np.exp(2j * np.pi * df * t)
    if add_noise:
        sig = np.sqrt(d.n0_total * fs_env / 2)
        x = x + sig * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return IQFrame(data=x, fs=fs_env, freq_offset=df, source="sim-baseband")
