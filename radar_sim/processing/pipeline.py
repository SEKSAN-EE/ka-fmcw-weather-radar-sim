"""Blocks 9–17 chained together. Input: an IQFrame (simulated or measured)."""
from __future__ import annotations

import numpy as np

from ..config import RadarConfig, db, get_window
from ..iq import IQFrame
from .calibration import (hitschfeld_bordan, min_detectable_dbz, overlap_loss, rain_rate,
                          reflectivity)
from .clutter import remove_static
from .dechirp import beat_decimate, dechirp
from .doppler import doppler_spectrum, hildebrand_sekhon, moments, pulse_pair
from .range_fft import range_fft
from .unfold import unfold_continuity


def theoretical_noise(cfg: RadarConfig) -> float:
    """Expected noise power per range bin (W at ADC input) after blocks 9–12."""
    d = cfg.d
    w = get_window(cfg.processing.range_window, d.n_beat)
    # in-band noise PSD (N0) x bin noise bandwidth (fs_beat/N * ENBW), beat filter flat in-band
    n = d.n0_total * d.fs_beat * np.sum(w ** 2) / np.sum(w) ** 2
    if cfg.processing.clutter == "mean":
        n *= 1 - 1 / cfg.waveform.n_chirps
    return n


def _front_blocks(frame: IQFrame, cfg: RadarConfig) -> dict:
    """Blocks 9–13 for one dwell."""
    beat = dechirp(cfg, frame.data, frame.fs, frame.freq_offset)            # 9
    beat_lp = beat_decimate(cfg, beat)                                      # 10
    profiles, ranges = range_fft(cfg, beat_lp)                              # 11
    prof = remove_static(cfg, profiles, ranges)                             # 12
    spec, vel = doppler_spectrum(cfg, prof)                                 # 13
    R0, R1 = pulse_pair(prof)
    return dict(beat_raw_chirp0=beat[0], beat_lp=beat_lp, profiles_raw=profiles, ranges=ranges,
                profiles=prof, spectrum=spec, velocity_axis=vel, R0=R0, R1=R1)


def process(frames, cfg: RadarConfig) -> dict:
    """Run blocks 9–17. ``frames`` is one IQFrame or a list of dwells (averaged incoherently:
    Doppler spectra, R(0) and R(1) are averaged over dwells before moment estimation)."""
    d, pr = cfg.d, cfg.processing
    frames = [frames] if isinstance(frames, IQFrame) else list(frames)
    per = [_front_blocks(f, cfg) for f in frames]
    res: dict = {"cfg": cfg, "frame": frames[0], "n_dwells": len(frames)}
    res.update(per[0])                       # intermediates of the first dwell for plotting
    res["spectrum_single"] = per[0]["spectrum"]
    spec = np.mean([p["spectrum"] for p in per], axis=0)
    R0 = np.mean([p["R0"] for p in per], axis=0)
    R1 = np.mean([p["R1"] for p in per], axis=0)
    res["spectrum"] = spec
    ranges = res["ranges"]
    # 14. noise estimate + threshold
    n_theory = theoretical_noise(cfg)
    hs = np.array([hildebrand_sekhon(spec[:, k], len(frames)) for k in range(spec.shape[1])]) * spec.shape[0]
    res["noise_hs_bins"] = hs
    res["noise_theory"] = n_theory
    usable = ranges >= max(pr.blank_m, d.delta_r)
    if pr.noise_method == "theory":
        noise = n_theory
    else:
        noise = float(np.median(hs[usable])) if np.any(usable) else n_theory
    res["noise"] = noise
    S, v, sw, snr = moments(cfg, R0, R1, noise)
    valid = usable & (snr >= 10 ** (pr.snr_threshold_db / 10))
    res.update(R0=R0, R1=R1, power=S, snr=snr, valid=valid,
               v=np.where(valid, v, np.nan), sw=np.where(valid, sw, np.nan), v_all=v)
    # velocity unfolding
    if pr.unfold:
        res["v_unfolded"] = unfold_continuity(v, valid, d.v_max, pr.unfold_direction)
    # 15. reflectivity
    z = reflectivity(cfg, np.maximum(S, 0), ranges)
    with np.errstate(divide="ignore", invalid="ignore"):
        res["dbz"] = np.where(valid, db(z), np.nan)
        res["dbz_all"] = db(np.maximum(z, 1e-30))
    res["z"] = z
    res["mdz"] = min_detectable_dbz(cfg, noise, ranges, pr.snr_threshold_db)
    res["overlap_loss_db"] = db(overlap_loss(cfg, ranges))
    # 16. attenuation correction
    if pr.attenuation_correction:
        zc, pia = hitschfeld_bordan(cfg, z, ranges, valid)
        with np.errstate(divide="ignore", invalid="ignore"):
            res["dbz_corr"] = np.where(valid, db(zc), np.nan)
        res["pia_est"] = pia
        z_for_rain = zc
    else:
        res["dbz_corr"] = res["dbz"]
        z_for_rain = z
    # 17. rain rate
    res["rain_rate"] = np.where(valid, rain_rate(cfg, z_for_rain), np.nan)
    return res
