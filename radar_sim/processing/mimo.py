"""Block 18 — MIMO processing and what comes after the angle FFT.

    per RX channel: dechirp -> LPF -> range FFT                      X[m, r, k]
    TDM de-interleave -> virtual array                                Xv[q, v, k]   v = t*n_rx + r
    channel calibration (corner reflector)                            Xv / c_v
    static-clutter removal, Doppler FFT over q                        D[f, v, k]
    TDM motion compensation  exp(-j 2 pi f t_v T)                     (TX t fires t*T later)
    angle FFT over v                                                  A[f, theta, k]
    -> power spectrum, noise, moments per (range, angle)              S, v_r, sigma_v, Z
    -> Doppler beam swinging: v_r(theta) = V cos(theta) - u sin(theta)  ->  u(h), V(h)
"""
from __future__ import annotations

import numpy as np

from ..config import RadarConfig, db, get_window
from ..iq import IQFrame
from ..mimo import array_geometry, element_pattern_2way
from .calibration import reflectivity
from .dechirp import beat_decimate, dechirp
from .doppler import hildebrand_sekhon
from .range_fft import range_fft


def mimo_profiles(cfg: RadarConfig, frame: IQFrame):
    """Blocks 9–11 on every RX channel: (n_chirps, n_rx, n_rep) -> (n_chirps, n_rx, n_bins)."""
    beat = dechirp(cfg, frame.data, frame.fs, frame.freq_offset)
    return range_fft(cfg, beat_decimate(cfg, beat))


def to_virtual(cfg: RadarConfig, X):
    """De-interleave TDM chirps: X[m, r, k] (m = q*n_tx + t) -> Xv[q, v, k] with v = t*n_rx + r."""
    mi = cfg.mimo
    q = X.shape[0] // mi.n_tx
    X4 = X[: q * mi.n_tx].reshape(q, mi.n_tx, mi.n_rx, X.shape[-1])
    return X4.reshape(q, mi.n_tx * mi.n_rx, X.shape[-1])


def estimate_calibration(cfg: RadarConfig, Xv, k_target: int, theta0_deg: float = 0.0):
    """Complex gain of each virtual channel from a point target (corner reflector) at a known angle.

    c_v = Xv[v, k] / (Xv[0, k] * a_v(theta0)),  a_v = exp(-j 2 pi x_v sin(theta0)) (profile domain)
    """
    _, _, x_v = array_geometry(cfg)
    x = Xv[:, :, k_target].mean(axis=0)
    a = np.exp(-2j * np.pi * x_v * np.sin(np.radians(theta0_deg)))
    c = x / (x[0] * a)
    return c


def angle_axis(cfg: RadarConfig):
    mi = cfg.mimo
    u = np.fft.fftshift(np.fft.fftfreq(mi.angle_fft))
    s = u / mi.rx_spacing                          # sin(theta); virtual spacing = rx_spacing
    ok = np.abs(s) <= 1
    return np.degrees(np.arcsin(np.clip(s, -1, 1))), ok


def range_angle_doppler(cfg: RadarConfig, Xv, cal=None):
    """Doppler-angle power spectrum per range bin: P[f, a, k] (sum over f = mean power)."""
    mi, d = cfg.mimo, cfg.d
    if cal is not None:
        Xv = Xv / np.asarray(cal)[None, :, None]
    if cfg.processing.clutter == "mean":
        Xv = Xv - Xv.mean(axis=0, keepdims=True)
    Q, V, _ = Xv.shape
    wq = get_window(mi.doppler_window, Q)
    D = np.fft.fft(Xv * wq[:, None, None], axis=0)
    f = np.fft.fftfreq(Q, d.t_tx)
    if mi.doppler_compensation:
        t_v = np.arange(V) // mi.n_rx
        D = D * np.exp(-2j * np.pi * f[:, None] * t_v[None, :] * d.t_rep)[:, :, None]
    wv = get_window(mi.angle_window, V)
    A = np.fft.ifft(D * wv[None, :, None], n=mi.angle_fft, axis=1) * mi.angle_fft / np.sum(wv)
    P = np.abs(A) ** 2 / (Q * np.sum(wq ** 2))
    P = np.fft.fftshift(P, axes=(0, 1))
    return P, np.fft.fftshift(f)


def angle_correction(cfg: RadarConfig, theta_deg):
    """Factor turning the power in one angle bin into the full-beam power the radar constant expects.

    bin power = rho(theta) * dtheta_eff,  rho = P_beam * f^4(theta) / (theta_b sqrt(pi / 8 ln2)),
    dtheta_eff = ENBW_window / (N_v * d * cos(theta))   [rad, d in wavelengths]
    """
    mi = cfg.mimo
    th = np.radians(theta_deg)
    wv = get_window(mi.angle_window, cfg.d.n_virtual)
    enbw = wv.size * np.sum(wv ** 2) / np.sum(wv) ** 2
    dtheta = enbw / (wv.size * mi.rx_spacing * np.cos(th))
    tb = np.radians(cfg.frontend.beamwidth_az_deg)
    f4 = element_pattern_2way(cfg, th) ** 2
    return tb * np.sqrt(np.pi / (8 * np.log(2))) / (f4 * dtheta)


def effective_angles(cfg: RadarConfig, theta_deg):
    """Power-weighted angle actually seen by each angle bin.

    A bin's response |H(theta - theta_i)|^2 (window + finite aperture) is multiplied by the two-way
    element pattern f^4(theta); near the edge of the 20 deg beam most of the power comes from
    closer to boresight, so v_r in that bin belongs to a smaller angle than theta_i:
        theta_eff,i = sum theta * f^4 |H_i|^2 / sum f^4 |H_i|^2
    (assumes reflectivity roughly uniform across the beam at one range).
    """
    mi = cfg.mimo
    _, _, x_v = array_geometry(cfg)
    wv = get_window(mi.angle_window, x_v.size)
    th = np.radians(np.linspace(-89, 89, 3561))
    f4 = element_pattern_2way(cfg, th) ** 2
    out = np.full(np.shape(theta_deg), np.nan)
    for i, t0 in enumerate(np.radians(np.asarray(theta_deg))):
        H = np.exp(2j * np.pi * x_v[None, :] * (np.sin(th)[:, None] - np.sin(t0))) @ wv / np.sum(wv)
        g = f4 * np.abs(H) ** 2
        out[i] = np.degrees(np.sum(th * g) / np.sum(g))
    return out


def retrieve_wind(theta_deg, v, weight, valid, min_angles: int = 4, gates: int = 1):
    """Weighted least squares  v_r(theta) = V cos(theta) - u sin(theta)  per range bin.

    ``gates`` > 1 pools the angle bins of neighbouring range gates into one fit (height averaging,
    as wind profilers do); u is sensitive to noise because sin(theta_eff) is small.
    """
    th = np.radians(theta_deg)
    nk = v.shape[1]
    half = gates // 2
    V = np.full(nk, np.nan)
    u = np.full(nk, np.nan)
    rms = np.full(nk, np.nan)
    for k in range(nk):
        ks = slice(max(0, k - half), min(nk, k + half + 1))
        m = valid[:, ks] & np.isfinite(v[:, ks])
        if m[:, min(k, half) if k >= half else k].sum() < min_angles:
            continue
        tt = np.broadcast_to(th[:, None], m.shape)[m]
        vv, ww = v[:, ks][m], np.sqrt(weight[:, ks][m])
        G = np.column_stack([np.cos(tt), -np.sin(tt)])
        sol, *_ = np.linalg.lstsq(G * ww[:, None], vv * ww, rcond=None)
        V[k], u[k] = sol
        rms[k] = np.sqrt(np.mean((G @ sol - vv) ** 2))
    return V, u, rms


def process_mimo(frames, cfg: RadarConfig, cal=None) -> dict:
    """Blocks 9–18 for TDM-MIMO frames (one IQFrame or a list of dwells)."""
    d, mi, pr = cfg.d, cfg.mimo, cfg.processing
    frames = [frames] if isinstance(frames, IQFrame) else list(frames)
    if cal is None and mi.calibration:
        cal = np.array([complex(a, b) for a, b in mi.calibration])
    P_sum, Xv0 = None, None
    for fr in frames:
        X, ranges = mimo_profiles(cfg, fr)
        Xv = to_virtual(cfg, X)
        if Xv0 is None:
            Xv0 = Xv
        P, f = range_angle_doppler(cfg, Xv, cal)
        P_sum = P if P_sum is None else P_sum + P
    P = P_sum / len(frames)
    theta, ang_ok = angle_axis(cfg)
    vel = -d.lam * f / 2
    usable_r = ranges >= max(pr.blank_m, 2 * d.delta_r)
    in_beam = ang_ok & (np.abs(theta) <= mi.max_angle_deg)

    # noise: Hildebrand–Sekhon on every (angle, range) Doppler spectrum, robust median
    F = P.shape[0]
    cols = [(a, k) for a in np.nonzero(in_beam)[0] for k in np.nonzero(usable_r)[0][::4]]
    hs = np.array([hildebrand_sekhon(P[:, a, k], len(frames)) for a, k in cols]) * F
    noise = float(np.median(hs))

    R0 = P.sum(axis=0)
    R1 = (P * np.exp(2j * np.pi * f * d.t_tx)[:, None, None]).sum(axis=0)
    S = R0 - noise
    snr = S / noise
    v = -d.lam / (4 * np.pi * d.t_tx) * np.angle(R1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(S > 0, S / np.maximum(np.abs(R1), 1e-300), np.nan)
        sw = d.lam / (2 * np.sqrt(2) * np.pi * d.t_tx) * np.sqrt(np.abs(np.log(np.maximum(ratio, 1.0))))
    valid = (snr >= 10 ** (pr.snr_threshold_db / 10)) & in_beam[:, None] & usable_r[None, :]
    z = reflectivity(cfg, np.maximum(S, 0), ranges[None, :]) * angle_correction(cfg, theta)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        dbz = np.where(valid, db(z), np.nan)
    weight = np.clip(snr, 0, None) / (1 + np.clip(snr, 0, None))
    theta_eff = np.where(in_beam, effective_angles(cfg, np.where(ang_ok, theta, 0.0)), np.nan)
    V1, u1, _ = retrieve_wind(theta_eff, np.where(valid, v, np.nan), weight, valid)
    V, u, rms = retrieve_wind(theta_eff, np.where(valid, v, np.nan), weight, valid, gates=mi.wind_gates)
    th = np.radians(theta)
    return dict(cfg=cfg, n_dwells=len(frames), P=P, doppler_f=f, velocity_axis=vel, theta=theta, theta_eff=theta_eff,
                angle_ok=ang_ok, in_beam=in_beam, ranges=ranges, noise=noise, S=S, snr=snr, valid=valid,
                v=np.where(valid, v, np.nan), sw=np.where(valid, sw, np.nan), dbz=dbz,
                x=ranges[None, :] * np.sin(th)[:, None], h=ranges[None, :] * np.cos(th)[:, None],
                wind_V=V, wind_u=u, wind_rms=rms, wind_u_gate=u1, wind_V_gate=V1, Xv=Xv0)
