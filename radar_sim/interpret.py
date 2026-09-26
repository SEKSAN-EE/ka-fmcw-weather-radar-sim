"""Annotated Doppler spectrogram (height vs velocity) that explains what each echo is."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle

from .plots import CRIT, GOOD, INK, INK2, MUTED, SEQ, SERIES, SURFACE, _fig, _shade_layers

KIND_TEXT = {
    "rain": "drops fall 5–9 m/s; wide spectrum from the\nspread of drop sizes + turbulence",
    "melting": "bright band: wet snowflakes look like huge\nwater drops → Z peak; fall speed jumps as\nflakes collapse into raindrops",
    "snow": "snow / ice crystals fall slowly (~1 m/s);\nnarrow spectrum, weaker Z",
    "cloud": "cloud droplets barely fall; Z < 0 dBZ,\nusually below the sensitivity of a\nlow-power radar",
}


GUIDE = [
    ("Rain", SERIES[0], "4–9 m/s toward the radar, broad spectrum;\nZ 20–50 dBZ, attenuates the signal above"),
    ("Melting layer / bright band", SERIES[1], "thin Z maximum; fall speed jumps from\n~1–2 m/s (snow) to ~6 m/s (rain) within ~300 m"),
    ("Snow / ice", SERIES[2], "~1 m/s, narrow spectrum, Z decreasing\nwith height"),
    ("Cloud", SERIES[3], "~0 m/s, very weak (< 0 dBZ)"),
    ("Artefacts", MUTED, "leakage at ~0 m spread over all velocities;\nspectra past ±v_max wrap around (aliasing)"),
]


def fig_interpretation(cfg, res, truth=None):
    d = cfg.d
    r_km = res["ranges"] / 1000
    spec = res["spectrum"]
    per_bin = res["noise"] / spec.shape[0]
    sd = 10 * np.log10(spec / per_bin + 1e-12)
    fig, ax = _fig("How to read the Doppler spectrogram of a vertically pointing Ka-band radar",
                   1, 3, size=(16, 7.5), gridspec_kw=dict(width_ratios=[2.3, 1.1, 1.5]))
    a = ax[0]
    im = a.pcolormesh(res["velocity_axis"], r_km, sd.T, cmap=SEQ, vmin=-2,
                      vmax=max(20, np.nanpercentile(sd, 99.5)), shading="auto")
    cb = fig.colorbar(im, ax=a, pad=0.01)
    cb.set_label("spectral power (dB above noise)")
    if truth is not None:
        a.plot(truth["v"], truth["r"] / 1000, color=INK, lw=0.9, ls="--", label="true mean fall speed")
    else:
        a.plot(res["v"], r_km, color=INK, lw=0.9, ls="--", label="measured mean velocity")
    for s in (-1, 1):
        a.axvline(s * d.v_max, color=CRIT, lw=1, ls=":")
    a.text(d.v_max, r_km[-1], f"+v_max\n{d.v_max:.1f} m/s ", color=CRIT, fontsize=8, ha="right", va="top")
    a.text(-d.v_max, r_km[-1], f" −v_max", color=CRIT, fontsize=8, ha="left", va="top")
    a.set(xlabel="Doppler velocity (m/s)   ← moving up  |  falling toward radar →",
          ylabel="height above radar (km)", title="Doppler spectrum vs height")
    a.legend(loc="upper center")
    detected = {}
    for i, L in enumerate(cfg.scene.rain_layers):
        m = (res["ranges"] >= L.r_bottom) & (res["ranges"] < L.r_top)
        frac = float(np.mean(res["valid"][m])) if m.any() else 0.0
        detected[i] = frac
        hmid = (L.r_bottom + L.r_top) / 2000
        vmid = (L.v_bottom + L.v_top) / 2
        x_text = vmid + 2.5 if vmid < d.v_max - 4 else vmid - 2.5
        a.annotate(L.name, (vmid, hmid), xytext=(x_text, hmid), fontsize=9, fontweight="bold",
                   color=INK, ha="left" if x_text > vmid else "right", va="center",
                   bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=SERIES[i % 8], lw=1.2),
                   arrowprops=dict(arrowstyle="-", color=SERIES[i % 8], lw=1.2))
    a.annotate("TX-leakage residual: chirp-to-chirp jitter\nspreads it over all velocities at ~0 m",
               (-d.v_max * 0.3, r_km[0]), xytext=(-d.v_max * 0.62, 0.3), fontsize=8, color=INK2,
               arrowprops=dict(arrowstyle="-", color=MUTED))
    # velocity aliasing: power of a layer that spills past +v_max appears at -v_max
    edge = (res["velocity_axis"] < -0.8 * d.v_max)
    alias_rows = np.nonzero((np.nanmean(sd[edge], axis=0) > 6) & (res["ranges"] > 2 * d.delta_r))[0]
    if alias_rows.size:
        h = r_km[alias_rows[len(alias_rows) // 2]]
        a.annotate("velocity aliasing: the fast tail of the rain\nspectrum (> +v_max) wraps to −v_max",
                   (-d.v_max * 0.92, h), xytext=(-d.v_max * 0.95, h + 1.3), fontsize=8, color=INK2,
                   arrowprops=dict(arrowstyle="-", color=MUTED))

    b = ax[1]
    _shade_layers(b, cfg, alpha=0.10)
    if truth is not None:
        b.plot(truth["dbz"], truth["r"] / 1000, color=INK, lw=1, label="true Z")
    b.plot(res["dbz_corr"], r_km, ".", color=SERIES[0], ms=3, label="measured (att.-corrected)")
    b.plot(res["mdz"], r_km, color=CRIT, lw=1, ls=":", label="min. detectable Z")
    b.set(xlabel="reflectivity (dBZ)", title="Reflectivity", ylim=a.get_ylim(), xlim=(-30, 45))
    b.legend(loc="upper right", fontsize=7)

    c = ax[2]
    c.axis("off")
    c.set_title("What each layer is", loc="left")
    layers = list(enumerate(cfg.scene.rain_layers))[::-1]      # top of the atmosphere first
    if not layers:                                            # real data: generic reading guide
        c.set_title("Reading guide (typical signatures)", loc="left")
        y = 0.97
        for name, col, txt in GUIDE:
            c.add_patch(Rectangle((0.0, y - 0.035), 0.03, 0.035, transform=c.transAxes, color=col))
            c.text(0.05, y, name, transform=c.transAxes, fontsize=10, fontweight="bold", color=INK, va="top")
            c.text(0.05, y - 0.045, txt, transform=c.transAxes, fontsize=8.5, color=INK2, va="top",
                   linespacing=1.3)
            y -= 0.18
    y = 0.97
    for i, L in layers:
        mdz_mid = np.interp((L.r_bottom + L.r_top) / 2, res["ranges"], res["mdz"])
        zmax = max(L.dbz_bottom, L.dbz_top, L.dbz_peak or -99)
        frac = detected[i]
        if frac > 0.5:
            status, col = f"detected ({frac:.0%} of gates)", GOOD
        elif frac > 0.05:
            status, col = f"partly detected ({frac:.0%})", "#c98500"
        else:
            status, col = f"not detected: Z ≤ {zmax:.0f} dBZ < MDZ ≈ {mdz_mid:.0f} dBZ", CRIT
        c.add_patch(Rectangle(
            (0.0, y - 0.035), 0.03, 0.035, transform=c.transAxes, color=SERIES[i % 8]))
        c.text(0.05, y, f"{L.name}  ({L.r_bottom/1000:.1f}–{L.r_top/1000:.1f} km)", transform=c.transAxes,
               fontsize=10, fontweight="bold", color=INK, va="top")
        c.text(0.05, y - 0.045, KIND_TEXT.get(L.kind, ""), transform=c.transAxes, fontsize=8.5,
               color=INK2, va="top", linespacing=1.3)
        c.text(0.05, y - 0.045 - 0.028 * (KIND_TEXT.get(L.kind, "").count("\n") + 1) - 0.01,
               f"● {status}", transform=c.transAxes, fontsize=8.5, color=col, va="top")
        y -= 0.235
    c.text(0.0, 0.02, f"Pt {cfg.frontend.pt_dbm:g} dBm · G {cfg.frontend.g_tx_dbi:g} dBi · NF_eff "
                      f"{d.nf_eff_db:.1f} dB · {res['n_dwells']}×{cfg.waveform.n_chirps} chirps\n"
                      f"ΔR {d.delta_r:.1f} m · v_max ±{d.v_max:.1f} m/s · λ {d.lam*1e3:.2f} mm",
           transform=c.transAxes, fontsize=8, color=MUTED, va="bottom")
    return fig
