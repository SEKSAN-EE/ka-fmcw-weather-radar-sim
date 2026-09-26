"""One plotting function per block. Each returns a matplotlib Figure."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.signal import freqz, spectrogram, welch  # noqa: E402

from .config import C, RadarConfig, db, w_to_dbm  # noqa: E402

# ------------------------------------------------------------------ style (reference palette)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CRIT, GOOD = "#d03b3b", "#0ca30c"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    "legend.frameon": False, "legend.fontsize": 8, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "figure.dpi": 110,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
})


def _fig(title: str, nrows=1, ncols=1, size=(11, 4.2), **kw):
    fig, ax = plt.subplots(nrows, ncols, figsize=size, constrained_layout=True, **kw)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    return fig, ax


def _note(ax, text, loc="upper right", **kw):
    xy = {"upper right": (0.98, 0.95), "upper left": (0.02, 0.95), "lower right": (0.98, 0.05),
          "lower left": (0.02, 0.05)}[loc]
    ax.text(*xy, text, transform=ax.transAxes, ha=loc.split()[1], va="top" if "upper" in loc else "bottom",
            fontsize=8, color=INK2, bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=GRID), **kw)


def _psd(x, fs, nper=8192, onesided=None):
    onesided = np.isrealobj(x) if onesided is None else onesided
    nper = min(nper, x.size)
    f, p = welch(x, fs, nperseg=nper, return_onesided=onesided, detrend=False, window="blackmanharris")
    if not onesided:
        f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    return f, w_to_dbm(np.maximum(p, 1e-40))


def _dbm(p):
    return w_to_dbm(np.maximum(p, 1e-40))


# ------------------------------------------------------------------ block 1
def fig_waveform(cfg: RadarConfig):
    from .waveform import dac_if_waveform, instantaneous_frequency, reference_chirp
    d, wf = cfg.d, cfg.waveform
    fig, ax = _fig("Block 1 — TX waveform (DAC IF chirp)", 1, 3, size=(13, 3.8))
    fs = d.fs_analog
    t, y = dac_if_waveform(cfg, fs, 1)
    n = int(3e-9 * fs)
    ax[0].plot(t[:n] * 1e9, y[:n], color=SERIES[0])
    ax[0].set(xlabel="time (ns)", ylabel="amplitude (norm.)", title="DAC output, first 3 ns")
    _note(ax[0], f"IF centre {d.f_if_center/1e9:.4f} GHz")
    tt = np.linspace(0, 3 * d.t_rep, 3000)
    ax[1].plot(tt * 1e6, instantaneous_frequency(cfg, tt) / 1e9, color=SERIES[0])
    ax[1].set(xlabel="time (µs)", ylabel="RF frequency (GHz)", title="Instantaneous frequency, 3 chirps")
    sec = ax[1].secondary_yaxis("right", functions=(lambda f: f - d.f_lo / 1e9, lambda f: f + d.f_lo / 1e9))
    sec.set_ylabel("IF (GHz)", color=INK2)
    _note(ax[1], f"B = {wf.bandwidth/1e6:g} MHz\nT = {wf.t_chirp*1e6:g} µs\nS = {d.slope:.2e} Hz/s",
          loc="lower right")
    s = reference_chirp(cfg, d.fs_ddc)
    nper = max(64, s.size // 64)
    f, tseg, S = spectrogram(s, d.fs_ddc, nperseg=nper, noverlap=nper * 3 // 4, return_onesided=False,
                             mode="psd")
    f, S = np.fft.fftshift(f), np.fft.fftshift(S, axes=0)
    im = ax[2].pcolormesh(tseg * 1e6, (f + d.f_if_center) / 1e9, 10 * np.log10(S / S.max() + 1e-12),
                          cmap=SEQ, vmin=-50, vmax=0, shading="auto")
    fig.colorbar(im, ax=ax[2], label="dB")
    ax[2].set(xlabel="time (µs)", ylabel="IF frequency (GHz)", title="Spectrogram of one chirp")
    ax[2].set_ylim((d.f_if_center - 0.7 * wf.bandwidth) / 1e9, (d.f_if_center + 0.7 * wf.bandwidth) / 1e9)
    return fig


# ------------------------------------------------------------------ block 2
def fig_scene(cfg: RadarConfig, cells=None):
    from .scene import point_target_power, rain_profile
    d = cfg.d
    if not (cfg.scene.kind in ("rain", "both") and cfg.scene.rain_layers):
        fig, ax = _fig("Block 2 — scene: point targets", size=(7, 3.5))
        for i, tg in enumerate(cfg.scene.point_targets):
            p = point_target_power(cfg, tg.range_m, tg.rcs_dbsm)
            ax.stem([tg.range_m], [w_to_dbm(p)], linefmt=SERIES[i], markerfmt="o", basefmt=" ",
                    bottom=w_to_dbm(p) - 30)
            ax.annotate(f"{tg.name}\nRCS {tg.rcs_dbsm:g} dBsm\nP_rx {w_to_dbm(p):.1f} dBm",
                        (tg.range_m, w_to_dbm(p)), xytext=(10, -10), textcoords="offset points",
                        fontsize=8, color=INK2)
        ax.set(xlabel="range (m)", ylabel="received power at antenna (dBm)", xlim=(0, cfg.processing.r_max))
        return fig
    r = np.linspace(1, cfg.processing.r_max, 2000)
    pr = rain_profile(cfg, r)
    fig, ax = _fig("Block 2 — scene: ground-truth weather profile (range = height, vertical beam)",
                   1, 4, size=(13, 5), sharey=True)
    h = r / 1000
    ax[0].plot(pr["dbz"], h, color=SERIES[0], label="Z (true)")
    ax[0].plot(pr["dbz_att"], h, color=SERIES[1], ls="--", label="Z attenuated (2-way)")
    ax[0].set(xlabel="reflectivity (dBZ)", ylabel="range / height (km)", title="Z(r)")
    ax[0].legend(loc="lower left")
    ax[1].plot(pr["v"], h, color=SERIES[0])
    ax[1].axvline(d.v_max, color=CRIT, lw=1, ls=":")
    ax[1].text(d.v_max, h[-1], " v_max", color=CRIT, fontsize=8, va="top")
    ax[1].set(xlabel="radial velocity (m/s, + toward radar)", title="v(r)")
    ax[2].plot(pr["sw"], h, color=SERIES[0])
    ax[2].set(xlabel="spectrum width σv (m/s)", title="σv(r)")
    ax[3].plot(pr["pia"], h, color=SERIES[1])
    ax[3].set(xlabel="two-way PIA (dB)", title="path attenuation")
    for a in ax:
        _shade_layers(a, cfg)
    for i, L in enumerate(cfg.scene.rain_layers):
        ax[3].text(ax[3].get_xlim()[1], (L.r_bottom + L.r_top) / 2000, L.name + " ", ha="right",
                   va="center", fontsize=8, color=INK2)
    if cells is not None:
        _note(ax[1], f"{cells.r.size} scattering cells\n(Zrnić spectral method)", loc="lower right")
    return fig


def _shade_layers(ax, cfg, horizontal=True, alpha=0.08):
    for i, L in enumerate(cfg.scene.rain_layers):
        if horizontal:
            ax.axhspan(L.r_bottom / 1000, L.r_top / 1000, color=SERIES[i % 8], alpha=alpha, lw=0)
        else:
            ax.axvspan(L.r_bottom / 1000, L.r_top / 1000, color=SERIES[i % 8], alpha=alpha, lw=0)


# ------------------------------------------------------------------ block 3
def fig_channel(cfg: RadarConfig, ch):
    d = cfg.d
    fig, ax = _fig("Block 3 — channel output at the RX antenna (complex envelope)", 1, 2, size=(12, 4))
    t = np.arange(ch.total.shape[1]) / ch.fs
    for i, (name, x) in enumerate(ch.parts.items()):
        ax[0].plot(t * 1e6, _dbm(np.abs(x[0]) ** 2), color=SERIES[i], label=name, lw=1)
    ax[0].set(xlabel="time within chirp 0 (µs)", ylabel="instantaneous power (dBm)",
              title="Leakage vs echo power (chirp 0)")
    ax[0].legend(loc="lower right")
    for i, (name, x) in enumerate(ch.parts.items()):
        f, p = _psd(x.reshape(-1), ch.fs, nper=4096, onesided=False)
        ax[1].plot((f + d.f_rf_center) / 1e9, p, color=SERIES[i], label=name, lw=1)
    ax[1].set(xlabel="RF frequency (GHz)", ylabel="PSD (dBm/Hz)", title="Spectrum around RF centre")
    ax[1].legend(loc="lower center")
    leak = np.mean(np.abs(ch.parts.get("leakage", np.zeros(1))) ** 2)
    echo = sum(np.mean(np.abs(v) ** 2) for k, v in ch.parts.items() if k != "leakage")
    _note(ax[0], f"leakage {_dbm(leak):.1f} dBm\necho total {_dbm(echo):.1f} dBm\n"
                 f"ratio {db(leak / max(echo, 1e-40)):.1f} dB", loc="upper right")
    return fig


# ------------------------------------------------------------------ block 4
def _irr_demo(cfg: RadarConfig, offset=20e6, n=1 << 16):
    from .frontend import analytic_part
    d = cfg.d
    fs = d.fs_analog
    t = np.arange(n) / fs
    fw = d.f_if_center
    a = np.exp(2j * np.pi * fw * t) + np.exp(-2j * np.pi * (fw + offset) * t)
    a2 = d.iq_mu * a + d.iq_nu * np.conj(a)
    y = np.sqrt(2) * np.real(analytic_part(a2))
    Y = np.abs(np.fft.rfft(y * np.blackman(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    return f, 10 * np.log10(Y / Y.max() + 1e-16)


def fig_downconverter(cfg: RadarConfig, ifp):
    d = cfg.d
    fe = ifp["frontend"]
    fig, ax = _fig("Block 4 — down-converter ADMV1014 (LO×4) + 90° hybrid", 1, 3, size=(14, 4))
    f, p = _psd(fe["complex_if"], fe["fs"], onesided=False)
    ax[0].plot(f / 1e9, p, color=SERIES[0], lw=1)
    ax[0].axvspan(d.f_if_lo / 1e9, d.f_if_hi / 1e9, color=SERIES[2], alpha=0.15, lw=0)
    ax[0].axvspan(-d.f_if_hi / 1e9, -d.f_if_lo / 1e9, color=SERIES[1], alpha=0.15, lw=0)
    ax[0].text(d.f_if_center / 1e9, ax[0].get_ylim()[1], "wanted\n(USB)", ha="center", va="top", fontsize=8)
    ax[0].text(-d.f_if_center / 1e9, ax[0].get_ylim()[1], "image\n(LSB)", ha="center", va="top", fontsize=8)
    ax[0].set(xlabel="frequency relative to LO (GHz)", ylabel="PSD (dBm/Hz)",
              title="Complex I/Q output (before hybrid)")
    f2, p2 = _irr_demo(cfg)
    m = (f2 > d.f_if_center - 80e6) & (f2 < d.f_if_center + 100e6)
    ax[1].plot(f2[m] / 1e9, p2[m], color=SERIES[0])
    ax[1].set(xlabel="real IF (GHz)", ylabel="dB rel. wanted", title="Test tones: wanted vs image",
              ylim=(-80, 5))
    ax[1].annotate(f"image tone −{d.irr_db:.1f} dB\n(gain {cfg.frontend.iq_gain_imbalance_db} dB, "
                   f"phase {cfg.frontend.iq_phase_imbalance_deg}°)", (d.f_if_center / 1e9 + 0.02, -d.irr_db),
                   xytext=(10, 20), textcoords="offset points", fontsize=8, color=INK2,
                   arrowprops=dict(arrowstyle="-", color=MUTED))
    f3, p3 = _psd(fe["real_if"], fe["fs"])
    ax[2].plot(f3 / 1e9, p3, color=SERIES[0], lw=1)
    ax[2].set(xlabel="frequency (GHz)", ylabel="PSD (dBm/Hz)", title="Real IF after hybrid")
    _note(ax[2], f"LO leakage {cfg.frontend.lo_leakage_dbm:g} dBm at DC\nchirp at "
                 f"{d.f_if_lo/1e9:.3f}–{d.f_if_hi/1e9:.3f} GHz")
    return fig


# ------------------------------------------------------------------ block 5
def fig_noise(cfg: RadarConfig, ch):
    from .processing.pipeline import theoretical_noise
    d = cfg.d
    kt0 = 1.380649e-23 * cfg.physics.t0
    fig, ax = _fig("Block 5 — receiver noise and levels at the ADC input", 1, 2, size=(13, 4.2))
    items = [("kT0 (−174)", kt0), (f"thermal kT0·F·G  (NF {d.nf_rx_db:.1f} dB, G {db(d.g_chain):.0f} dB)",
                                   d.n0_thermal),
             (f"image-band leak (IRR {d.irr_db:.0f} dB)", d.n0_thermal / 10 ** (d.irr_db / 10)),
             (f"ADC NSD ({cfg.adc.nsd_dbfs_hz:g} dBFS/Hz)", d.n0_adc),
             (f"total  → effective NF {d.nf_eff_db:.2f} dB", d.n0_total)]
    y = np.arange(len(items))[::-1]
    vals = [w_to_dbm(v) for _, v in items]
    ax[0].barh(y, np.array(vals) + 200, left=-200, height=0.55,
               color=[MUTED, SERIES[0], SERIES[1], SERIES[3], INK2])
    for yi, v, (lab, _) in zip(y, vals, items):
        ax[0].text(-199, yi + 0.36, lab, fontsize=8, color=INK2, va="bottom")
        ax[0].text(v + 0.5, yi, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax[0].set(xlim=(-200, -130), xlabel="noise PSD at ADC input (dBm/Hz)", yticks=[],
              title="Noise cascade (Friis + ADC)")
    leak = np.mean(np.abs(ch.parts.get("leakage", np.zeros(1))) ** 2) * d.g_chain
    echo = sum(np.mean(np.abs(v) ** 2) for k, v in ch.parts.items() if k != "leakage") * d.g_chain
    n_b = d.n0_total * cfg.waveform.bandwidth
    n_bin = theoretical_noise(cfg)
    lv = [("ADC full scale", d.p_fs), ("TX leakage", leak), ("echo (all targets)", echo),
          (f"noise in chirp band B", n_b), ("noise per range bin (after FFT)", n_bin)]
    y = np.arange(len(lv))[::-1]
    vals = [w_to_dbm(max(v, 1e-40)) for _, v in lv]
    lo = min(vals) - 15
    ax[1].barh(y, np.array(vals) - lo, left=lo, height=0.55,
               color=[CRIT, SERIES[1], SERIES[0], MUTED, INK2])
    for yi, v, (lab, _) in zip(y, vals, lv):
        ax[1].text(lo + 1, yi + 0.36, lab, fontsize=8, color=INK2, va="bottom")
        ax[1].text(v + 0.8, yi, f"{v:.1f} dBm", va="center", fontsize=8, color=INK)
    ax[1].set(xlim=(lo, max(vals) + 18), yticks=[], xlabel="power at ADC input (dBm)",
              title="Level diagram")
    _note(ax[1], f"SNR at ADC input (B): {db(echo / n_b):.1f} dB\nADC headroom vs leakage: "
                 f"{d.adc_headroom_db:.1f} dB", loc="lower right")
    return fig


# ------------------------------------------------------------------ block 6
def fig_bpf(cfg: RadarConfig, ifp):
    d = cfg.d
    fe = ifp["frontend"]
    fig, ax = _fig(f"Block 6 — anti-alias BPF (Nyquist zone {d.zone})", size=(12, 4))
    f, p = _psd(fe["real_if"], fe["fs"])
    f2, p2 = _psd(fe["filtered"], fe["fs"])
    ax.plot(f / 1e9, p, color=MUTED, lw=1, label="before BPF")
    ax.plot(f2 / 1e9, p2, color=SERIES[0], lw=1.2, label="after BPF")
    for k in range(1, int(d.fs_analog / 2 / d.zone_width) + 1):
        ax.axvline(k * d.zone_width / 1e9, color=AXIS, lw=0.8, ls="--")
        ax.text((k - 0.5) * d.zone_width / 1e9, ax.get_ylim()[1], f"zone {k}", ha="center", va="top",
                fontsize=8, color=MUTED)
    ax2 = ax.twinx()
    ax2.plot(fe["bpf_freqs"] / 1e9, 20 * np.log10(fe["bpf_mask"] + 1e-6), color=SERIES[1], lw=1, ls=":")
    ax2.set_ylim(-120, 5)
    ax2.set_ylabel("BPF response (dB)", color=SERIES[1])
    ax2.grid(False)
    ax.set(xlabel="frequency (GHz)", ylabel="PSD (dBm/Hz)")
    ax.legend(loc="lower left")
    return fig


# ------------------------------------------------------------------ block 7
def fig_adc(cfg: RadarConfig, ifp):
    d = cfg.d
    ad = ifp["adc"]
    fig, ax = _fig(f"Block 7 — RF-ADC: fs = {d.fs/1e9:g} GSPS, {cfg.adc.bits}-bit", 1, 3, size=(14, 4))
    f, p = _psd(ad["x"], d.fs)
    ax[0].plot(f / 1e6, p, color=SERIES[0], lw=1)
    fa = _alias(d.f_if_center, d.fs)
    ax[0].axvspan(_alias(d.f_if_lo, d.fs) / 1e6, _alias(d.f_if_hi, d.fs) / 1e6, color=SERIES[2], alpha=0.15)
    ax[0].set(xlabel="frequency after sampling (MHz)", ylabel="PSD (dBm/Hz)",
              title="Spectrum folded into zone 1")
    _note(ax[0], f"IF {d.f_if_center/1e9:.4f} GHz → {fa/1e6:.1f} MHz\nzone {d.zone}: "
                 f"{'inverted' if d.zone_inverted else 'not inverted'}")
    # zone folding diagram
    zones = int(np.ceil(max(d.f_if_hi, 4 * d.zone_width) / d.zone_width))
    for k in range(zones):
        ax[1].add_patch(plt.Rectangle((k * d.zone_width / 1e9, 0), d.zone_width / 1e9, 1,
                                      color=SERIES[0] if k % 2 == 0 else SERIES[1], alpha=0.10, lw=0))
        ax[1].text((k + 0.5) * d.zone_width / 1e9, 0.93, f"zone {k+1}", ha="center", fontsize=8, color=INK2)
    ax[1].plot([d.f_if_lo / 1e9, d.f_if_hi / 1e9], [0.6, 0.6], color=SERIES[2], lw=6, solid_capstyle="butt")
    ax[1].plot([_alias(d.f_if_lo, d.fs) / 1e9, _alias(d.f_if_hi, d.fs) / 1e9], [0.3, 0.3], color=SERIES[2],
               lw=6, solid_capstyle="butt")
    ax[1].annotate("", (_alias(d.f_if_center, d.fs) / 1e9, 0.34), (d.f_if_center / 1e9, 0.56),
                   arrowprops=dict(arrowstyle="->", color=INK2))
    ax[1].text(d.f_if_center / 1e9, 0.66, "IF band", ha="center", fontsize=8)
    ax[1].text(_alias(d.f_if_center, d.fs) / 1e9, 0.2, "after sampling", ha="center", fontsize=8)
    ax[1].set(xlim=(0, zones * d.zone_width / 1e9), ylim=(0, 1), yticks=[], xlabel="frequency (GHz)",
              title="Nyquist-zone map")
    ax[1].grid(False)
    codes = ad["codes"]
    ax[2].hist(codes, bins=200, color=SERIES[0])
    fsc = 2 ** (cfg.adc.bits - 1)
    for s in (-fsc, fsc):
        ax[2].axvline(s, color=CRIT, lw=1, ls="--")
    ax[2].set(xlabel="ADC code", ylabel="count", title="Code histogram")
    ax[2].set_yscale("log")
    _note(ax[2], f"peak {ad['peak_dbfs']:.1f} dBFS\nclipped samples: {ad['clipped']}", loc="upper left")
    return fig


def _alias(f, fs):
    f = np.mod(f, fs)
    return np.where(f > fs / 2, fs - f, f)


# ------------------------------------------------------------------ block 8
def fig_ddc(cfg: RadarConfig, ifp):
    from .adc import ddc_filter
    d = cfg.d
    x = ifp["frame"].stream()
    fig, ax = _fig(f"Block 8 — DDC: NCO {d.f_nco/1e9:.5f} GHz, decimate ×{cfg.adc.ddc_decimation}",
                   1, 2, size=(12, 4))
    f, p = _psd(x, d.fs_ddc, nper=min(4096, x.size), onesided=False)
    ax[0].plot(f / 1e6, p, color=SERIES[0], lw=1)
    ax[0].axvspan(-cfg.waveform.bandwidth / 2e6 + (d.f_if_center - d.f_nco) / 1e6,
                  cfg.waveform.bandwidth / 2e6 + (d.f_if_center - d.f_nco) / 1e6, color=SERIES[2], alpha=0.15)
    ax[0].set(xlabel="baseband frequency (MHz)", ylabel="PSD (dBm/Hz)", title="Complex I/Q output")
    h = ddc_filter(cfg)
    w, H = freqz(h, worN=8192, fs=d.fs)
    ax[1].plot(w / 1e6, 20 * np.log10(np.abs(H) + 1e-12), color=SERIES[0])
    ax[1].axvline(d.fs_ddc / 2e6, color=CRIT, ls="--", lw=1)
    ax[1].text(d.fs_ddc / 2e6, -10, " fs_out/2", color=CRIT, fontsize=8)
    ax[1].set(xlim=(0, 2 * d.fs_ddc / 1e6), ylim=(-120, 5), xlabel="frequency (MHz)", ylabel="dB",
              title=f"Decimation filter ({h.size} taps)")
    rate_in = d.fs * cfg.adc.bits / 1e9
    rate_out = d.fs_ddc * 2 * 16 / 1e9
    _note(ax[0], f"ADC: {d.fs/1e9:g} GSPS × {cfg.adc.bits} b = {rate_in:.1f} Gb/s\n"
                 f"DDC out: {d.fs_ddc/1e6:g} MSPS I/Q × 16 b = {rate_out:.2f} Gb/s", loc="lower right")
    return fig


def fig_equivalence(cfg: RadarConfig, x_if, x_bb, nmse_db):
    d = cfg.d
    fig, ax = _fig("IF path vs complex-baseband model (after DDC, noise-free)", 1, 2, size=(12, 4))
    n = x_if.size
    seg = slice(n // 2, n // 2 + min(400, n // 4))
    t = np.arange(n)[seg] / d.fs_ddc * 1e6
    ax[0].plot(t, x_if.real[seg] * 1e3, color=SERIES[0], label="IF path (real part)")
    ax[0].plot(t, x_bb.real[seg] * 1e3, color=SERIES[1], ls="--", label="baseband model")
    ax[0].set(xlabel="time (µs)", ylabel="amplitude (√mW)", title="Waveform overlay")
    ax[0].legend(loc="upper right")
    err = x_if - x_bb
    f, p1 = _psd(x_bb, d.fs_ddc, nper=2048, onesided=False)
    f, p2 = _psd(err, d.fs_ddc, nper=2048, onesided=False)
    ax[1].plot(f / 1e6, p1, color=SERIES[1], label="baseband model", lw=1)
    ax[1].plot(f / 1e6, p2, color=CRIT, label="difference", lw=1)
    ax[1].set(xlabel="frequency (MHz)", ylabel="PSD (dBm/Hz)", title=f"Spectrum — NMSE {nmse_db:.1f} dB")
    ax[1].legend(loc="lower right")
    return fig


# ------------------------------------------------------------------ blocks 9–10
def fig_dechirp(cfg: RadarConfig, res):
    d = cfg.d
    b = res["beat_raw_chirp0"]
    fig, ax = _fig("Block 9 — digital dechirp: beat = ref · conj(rx)", 1, 2, size=(12, 4))
    nper = 256 if b.size > 4096 else 64
    f, t, S = spectrogram(b, d.fs_ddc, nperseg=nper, noverlap=nper // 2, return_onesided=False)
    f, S = np.fft.fftshift(f), np.fft.fftshift(S, axes=0)
    Sd = 10 * np.log10(S + 1e-30)
    im = ax[0].pcolormesh(t * 1e6, f / 1e6, Sd, cmap=SEQ, vmin=np.percentile(Sd, 50), vmax=Sd.max(),
                          shading="auto")
    fig.colorbar(im, ax=ax[0], label="dB")
    ax[0].set(xlabel="time (µs)", ylabel="beat frequency (MHz)", title="Beat spectrogram, chirp 0")
    ax[0].set_ylim(-3 * d.fb_max / 1e6, 3 * d.fb_max / 1e6)
    ax[0].axhline(d.fb_max / 1e6, color=CRIT, ls="--", lw=1)
    ax[0].text(t[0] * 1e6, d.fb_max / 1e6, f" f_b at r_max = {d.fb_max/1e6:.2f} MHz", color=CRIT,
               fontsize=8, va="bottom")
    ranges = np.linspace(0, cfg.processing.r_max, 200)
    ax[1].plot(ranges, d.slope * 2 * ranges / C / 1e6, color=SERIES[0])
    ax[1].set(xlabel="range (m)", ylabel="beat frequency (MHz)", title="f_b = S·τ = 2SR/c")
    _note(ax[1], f"S = {d.slope:.3e} Hz/s\n1 range bin = {d.delta_r:.3f} m = {1/cfg.waveform.t_chirp/1e3:.1f} kHz",
          loc="upper left")
    return fig


def fig_beat_lpf(cfg: RadarConfig, res):
    from .processing.dechirp import beat_filter
    d = cfg.d
    fig, ax = _fig(f"Block 10 — beat low-pass + decimate ×{d.beat_decimation} → {d.fs_beat/1e6:g} MSPS",
                   size=(12, 4))
    b = res["beat_raw_chirp0"]
    blp = res["beat_lp"][0]
    f, p = _psd(b, d.fs_ddc, nper=min(4096, b.size), onesided=False)
    f2, p2 = _psd(blp, d.fs_beat, nper=min(1024, blp.size), onesided=False)
    ax.plot(f / 1e6, p, color=MUTED, lw=1, label=f"before ({d.fs_ddc/1e6:g} MSPS)")
    ax.plot(f2 / 1e6, p2, color=SERIES[0], lw=1.2, label=f"after ({d.fs_beat/1e6:g} MSPS)")
    h, cutoff = beat_filter(cfg)
    w, H = freqz(h, worN=4096, fs=d.fs_ddc, whole=True)
    w = np.where(w > d.fs_ddc / 2, w - d.fs_ddc, w)
    o = np.argsort(w)
    ax2 = ax.twinx()
    ax2.plot(w[o] / 1e6, 20 * np.log10(np.abs(H[o]) + 1e-12), color=SERIES[1], ls=":", lw=1)
    ax2.set_ylim(-120, 5)
    ax2.set_ylabel("filter (dB)", color=SERIES[1])
    ax2.grid(False)
    ax.set(xlabel="beat frequency (MHz)", ylabel="PSD (dBm/Hz)")
    ax.set_xlim(-min(d.fs_ddc / 2, 6 * d.fb_max) / 1e6, min(d.fs_ddc / 2, 6 * d.fb_max) / 1e6)
    ax.legend(loc="lower left")
    return fig


# ------------------------------------------------------------------ blocks 11–12
def fig_range(cfg: RadarConfig, res, point_targets=None):
    d = cfg.d
    r = res["ranges"]
    P = np.abs(res["profiles_raw"]) ** 2
    fig, ax = _fig(f"Block 11 — range FFT ({cfg.processing.range_window} window)", 1, 2, size=(13, 4.2))
    ax[0].plot(r, _dbm(P[0]), color=MUTED, lw=0.8, label="chirp 0")
    ax[0].plot(r, _dbm(P.mean(0)), color=SERIES[0], label="mean over chirps")
    ax[0].axhline(_dbm(res["noise_theory"]), color=INK2, ls=":", lw=1)
    ax[0].text(r[-1], _dbm(res["noise_theory"]), "noise (theory) ", ha="right", va="bottom", fontsize=8,
               color=INK2)
    ax[0].set(xlabel="range (m)", ylabel="bin power at ADC (dBm)", title="Range profile")
    ax[0].legend(loc="upper right")
    if point_targets:
        for tg in point_targets:
            k = np.argmax(np.where(r > cfg.processing.blank_m, P.mean(0), 0))
            ax[0].annotate(f"{tg.name}\npeak at {r[k]:.3f} m\n(truth {tg.range_m:.2f} m, bin {d.delta_r*100:.0f} cm)",
                           (r[k], _dbm(P.mean(0)[k])), xytext=(25, -30), textcoords="offset points",
                           fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    im = ax[1].pcolormesh(r, np.arange(P.shape[0]), _dbm(P), cmap=SEQ, shading="auto",
                          vmin=_dbm(res["noise_theory"]) - 5, vmax=_dbm(P).max())
    fig.colorbar(im, ax=ax[1], label="dBm")
    ax[1].set(xlabel="range (m)", ylabel="chirp index", title="Range–chirp matrix")
    return fig


def fig_clutter(cfg: RadarConfig, res):
    r = res["ranges"]
    fig, ax = _fig(f"Block 12 — leakage / static-clutter removal ({cfg.processing.clutter})", size=(12, 4))
    ax.plot(r, _dbm(np.mean(np.abs(res["profiles_raw"]) ** 2, 0)), color=MUTED, label="before")
    ax.plot(r, _dbm(np.mean(np.abs(res["profiles"]) ** 2, 0)), color=SERIES[0], label="after")
    ax.axhline(_dbm(res["noise_theory"]), color=INK2, ls=":", lw=1)
    ax.set(xlabel="range (m)", ylabel="mean bin power (dBm)")
    ax.legend(loc="upper right")
    _note(ax, "TX leakage sits at ~0 m; its window sidelobes\nand chirp-to-chirp jitter set the near-range floor",
          loc="lower left")
    return fig


# ------------------------------------------------------------------ block 13
def fig_doppler(cfg: RadarConfig, res, truth=None):
    d = cfg.d
    r = res["ranges"] / 1000
    spec = res["spectrum"]
    fig, ax = _fig(f"Block 13 — Doppler processing ({res['n_dwells']} dwell(s) × "
                   f"{cfg.waveform.n_chirps} chirps)", 1, 3, size=(14, 5), sharey=True,
                   gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
    per_bin = res["noise"] / spec.shape[0]
    sd = 10 * np.log10(spec / per_bin + 1e-12)
    im = ax[0].pcolormesh(res["velocity_axis"], r, sd.T, cmap=SEQ, vmin=-3, vmax=max(20, np.nanpercentile(sd, 99.5)),
                          shading="auto")
    fig.colorbar(im, ax=ax[0], label="dB above noise")
    ax[0].set(xlabel="radial velocity (m/s, + toward radar)", ylabel="range / height (km)",
              title="Range–Doppler map")
    ax[1].plot(res["v"], r, ".", color=SERIES[0], ms=3, label="pulse-pair")
    if "v_unfolded" in res:
        ax[1].plot(res["v_unfolded"], r, ".", color=SERIES[2], ms=3, label="unfolded")
    if truth is not None:
        ax[1].plot(truth["v"], truth["r"] / 1000, color=INK, lw=1, label="truth")
    for s in (-1, 1):
        ax[1].axvline(s * d.v_max, color=CRIT, lw=1, ls=":")
    ax[1].set(xlabel="v (m/s)", title="Mean velocity")
    ax[1].legend(loc="upper left")
    ax[2].plot(res["sw"], r, ".", color=SERIES[0], ms=3, label="pulse-pair")
    if truth is not None:
        ax[2].plot(truth["sw"], truth["r"] / 1000, color=INK, lw=1, label="truth")
    ax[2].set(xlabel="σv (m/s)", title="Spectrum width", xlim=(0, None))
    ax[2].legend(loc="upper right")
    return fig


# ------------------------------------------------------------------ block 14
def fig_detection(cfg: RadarConfig, res):
    r = res["ranges"] / 1000
    fig, ax = _fig("Block 14 — noise estimate and SNR threshold", 1, 2, size=(12, 4.5))
    snr = 10 * np.log10(np.maximum(res["snr"], 1e-6))
    ax[0].plot(snr, r, color=SERIES[0], lw=1)
    ax[0].axvline(cfg.processing.snr_threshold_db, color=CRIT, ls="--", lw=1)
    ax[0].fill_betweenx(r, -60, 80, where=res["valid"], color=GOOD, alpha=0.08, lw=0)
    ax[0].set(xlabel="SNR per chirp (dB)", ylabel="range (km)", xlim=(-30, max(30, np.nanmax(snr) + 3)),
              title="SNR profile and valid mask")
    ax[0].text(cfg.processing.snr_threshold_db, r[-1], " threshold", color=CRIT, fontsize=8, va="top")
    ax[1].plot(_dbm(res["noise_hs_bins"]), r, color=MUTED, lw=0.8, label="HS per range bin")
    ax[1].axvline(_dbm(res["noise"]), color=SERIES[0], lw=1.5, label="noise used (median HS)")
    ax[1].axvline(_dbm(res["noise_theory"]), color=SERIES[1], lw=1.5, ls="--", label="theory")
    ax[1].set(xlabel="noise power per bin (dBm)", title="Noise estimation")
    ax[1].legend(loc="lower right")
    _note(ax[1], f"estimate − theory = {db(res['noise']/res['noise_theory']):+.2f} dB", loc="upper left")
    return fig


# ------------------------------------------------------------------ blocks 15–17
def fig_reflectivity(cfg: RadarConfig, res, truth=None, expected=None):
    r = res["ranges"] / 1000
    fig, ax = _fig("Block 15 — calibration to reflectivity", 1, 2, size=(12, 5),
                   gridspec_kw=dict(width_ratios=[1.6, 1]))
    if truth is not None:
        ax[0].plot(truth["dbz"], truth["r"] / 1000, color=INK, lw=1, label="truth (unattenuated)")
    if expected is not None:
        ax[0].plot(expected, r, color=SERIES[1], lw=1.2, ls="--", label="expected (attenuated, range-weighted)")
    ax[0].plot(res["dbz"], r, ".", color=SERIES[0], ms=3.5, label="estimated")
    ax[0].plot(res["mdz"], r, color=MUTED, lw=1, ls=":", label="minimum detectable Z")
    ax[0].set(xlabel="dBZ", ylabel="range / height (km)", title="Z(r)", xlim=(-30, 50))
    ax[0].legend(loc="upper right")
    _shade_layers(ax[0], cfg)
    if expected is not None:
        err = res["dbz"] - expected
        snr = 10 * np.log10(np.maximum(res["snr"], 1e-6))
        ax[1].plot(snr, err, ".", color=SERIES[0], ms=3.5)
        ax[1].axhline(0, color=INK2, lw=1)
        ax[1].set(xlabel="SNR (dB)", ylabel="error (dB)", title="Z error vs SNR", ylim=(-8, 8))
        ok = np.isfinite(err)
        if ok.any():
            _note(ax[1], f"bias {np.nanmean(err):+.2f} dB\nstd {np.nanstd(err):.2f} dB\n"
                         f"{ok.sum()} valid bins", loc="upper right")
    return fig


def fig_attenuation(cfg: RadarConfig, res, truth=None):
    r = res["ranges"] / 1000
    fig, ax = _fig("Block 16 — attenuation correction (Hitschfeld–Bordan)", 1, 2, size=(12, 5), sharey=True)
    if truth is not None:
        ax[0].plot(truth["dbz"], truth["r"] / 1000, color=INK, lw=1, label="truth")
    ax[0].plot(res["dbz"], r, ".", color=MUTED, ms=3, label="measured")
    ax[0].plot(res["dbz_corr"], r, ".", color=SERIES[0], ms=3, label="corrected")
    ax[0].set(xlabel="dBZ", ylabel="range / height (km)", title="Z before / after")
    ax[0].legend(loc="upper right")
    if truth is not None:
        ax[1].plot(truth["pia"], truth["r"] / 1000, color=INK, lw=1, label="true PIA")
    if "pia_est" in res:
        ax[1].plot(np.where(res["valid"], res["pia_est"], np.nan), r, color=SERIES[0], label="HB estimate")
    ax[1].set(xlabel="two-way PIA (dB)", title="Path-integrated attenuation")
    ax[1].legend(loc="lower right")
    return fig


def fig_rainrate(cfg: RadarConfig, res, truth=None):
    r = res["ranges"] / 1000
    pr = cfg.processing
    fig, ax = _fig(f"Block 17 — rain rate  Z = {pr.zr_a:g} R^{pr.zr_b:g}", size=(7, 5))
    if truth is not None:
        rr = np.where(truth["layer"] >= 0, truth["rain_rate"], np.nan)
        ax.plot(rr, truth["r"] / 1000, color=INK, lw=1, label="truth (Z–R of true Z)")
    ax.plot(res["rain_rate"], r, ".", color=SERIES[0], ms=3.5, label="estimated")
    ax.set(xlabel="R (mm/h)", ylabel="range / height (km)")
    ax.legend(loc="upper right")
    _shade_layers(ax, cfg)
    _note(ax, "Z–R is only meaningful in the liquid layer;\nabove the melting layer it is shown for reference.",
          loc="lower right")
    return fig
