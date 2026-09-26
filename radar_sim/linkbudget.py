"""Analytic link budgets (validation references for the simulation)."""
from __future__ import annotations

import numpy as np

from .config import K_B, RadarConfig, db, undb
from .processing.calibration import delta_r_eff, overlap_loss
from .processing.pipeline import theoretical_noise
from .scene import point_target_power, weather_power_density


def point_target_snr_db(cfg: RadarConfig, range_m: float, rcs_dbsm: float) -> float:
    """Expected per-chirp range-bin SNR of a point target (bin-centred, no scalloping)."""
    p = point_target_power(cfg, range_m, rcs_dbsm) * cfg.d.g_chain * overlap_loss(cfg, [range_m])[0]
    return float(db(p / theoretical_noise(cfg)))


def weather_snr_db(cfg: RadarConfig, ranges, dbz, pia_db=0.0):
    """Expected per-chirp range-bin SNR of uniform rain of reflectivity ``dbz`` (processing chain)."""
    p = weather_power_density(cfg, ranges, undb(dbz), pia_db) * delta_r_eff(cfg)
    p = p * cfg.d.g_chain * overlap_loss(cfg, ranges)
    return db(p / theoretical_noise(cfg))


def brief_link_budget_snr_db(cfg: RadarConfig, ranges, dbz, pt_dbm, nf_db=10.0,
                             delta_r=30.0, t_coh=1e-3):
    """The simple link budget of the project brief: SNR = P_r / (k T0 F / T_coh), bin length delta_r."""
    c2 = cfg.replace(frontend={"pt_dbm": pt_dbm})
    d = c2.d
    const = d.pt * d.g_tx * d.g_rx * d.theta * d.phi * np.pi ** 3 * d.k2 / (512 * np.log(2) * d.lam ** 2 * d.loss)
    p = const * undb(dbz) * 1e-18 * delta_r / np.asarray(ranges) ** 2
    n = K_B * cfg.physics.t0 * undb(nf_db) / t_coh
    return db(p / n)
