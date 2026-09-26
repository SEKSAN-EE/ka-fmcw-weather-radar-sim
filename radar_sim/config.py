"""Configuration: dataclasses loaded from YAML, plus derived parameters.

All physical quantities are SI (Hz, s, m, W) unless the field name says otherwise
(e.g. ``_db``, ``_dbm``, ``_deg``).

Sign / unit conventions used throughout the package
---------------------------------------------------
* Signals are amplitudes in sqrt(W) referred to a 50-ohm port: a real tone
  ``sqrt(2P) cos(wt)`` and a complex envelope ``sqrt(P) exp(jwt)`` both carry power P.
* Radial velocity ``v`` is positive for targets moving TOWARD the radar
  (for a vertically pointing radar: positive = falling).
* The received complex envelope of an approaching target rotates at +2v/lambda.
  The dechirped (beat) signal is ``ref * conj(rx)`` so the beat frequency is +S*tau and
  the slow-time Doppler is -2v/lambda, i.e. ``v = -(lambda / 4 pi T) arg R(1)``.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path

import numpy as np
import yaml

C = 299_792_458.0
K_B = 1.380649e-23


def db(x):
    return 10.0 * np.log10(x)


def undb(x):
    return 10.0 ** (np.asarray(x, dtype=float) / 10.0)


def dbm_to_w(p_dbm):
    return 10.0 ** ((np.asarray(p_dbm, dtype=float) - 30.0) / 10.0)


def w_to_dbm(p_w):
    return 10.0 * np.log10(np.asarray(p_w, dtype=float)) + 30.0


# --------------------------------------------------------------------------- dataclasses
@dataclass
class LOConfig:
    f_ref: float = 7.68e6          # reference clock from FPGA
    n_pll: int = 1119              # integer-N divider: 1119 -> 8.59392 GHz, 1112 -> 8.54016 GHz
    f_pll: float | None = None     # set to override n_pll (e.g. fractional-N)
    multiplier: int = 4            # ADMV1013/1014 internal LO x4
    sideband: str = "usb"          # RF = LO + IF


@dataclass
class WaveformConfig:
    f_rf_center: float = 37.5e9
    bandwidth: float = 10e6
    t_chirp: float = 200e-6
    t_idle: float = 0.0            # TX is off during idle
    n_chirps: int = 64


@dataclass
class ADCConfig:
    fs: float = 2.5e9
    bits: int = 14
    full_scale_dbm: float = 1.0    # power of a full-scale sine at the ADC input
    nsd_dbfs_hz: float = -150.0    # ADC noise spectral density (assumption, check DS926)
    jitter_rms: float = 0.0        # s
    ddc_decimation: int = 40
    nco_freq: float | None = None  # None -> IF centre
    analog_oversample: int = 4     # "analog" simulation rate = fs * analog_oversample


@dataclass
class LNAConfig:
    enabled: bool = False
    nf_db: float = 3.0
    gain_db: float = 20.0


@dataclass
class FrontendConfig:
    pt_dbm: float = 20.0
    g_tx_dbi: float = 20.0
    g_rx_dbi: float = 20.0
    beamwidth_az_deg: float = 20.0
    beamwidth_el_deg: float = 20.0
    system_loss_db: float = 2.0
    mixer_nf_db: float = 10.0      # down-converter NF (no LNA)
    rx_gain_db: float = 20.0       # antenna port -> ADC input, excluding optional LNA
    lna: LNAConfig = field(default_factory=LNAConfig)
    iq_gain_imbalance_db: float = 0.5
    iq_phase_imbalance_deg: float = 4.0
    lo_leakage_dbm: float = -40.0  # LO feed-through at the IF output (appears at DC)
    bpf_zone: int | None = None    # None -> zone containing the IF band
    bpf_guard_hz: float = 50e6     # distance of BPF passband edges from zone edges


@dataclass
class ChannelConfig:
    tx_rx_isolation_db: float = 40.0
    leakage_distance_m: float = 0.05
    leakage_fluct_dbc: float = -60.0   # chirp-to-chirp random fluctuation of the leakage
    elevation_deg: float = 90.0        # 90 = vertically pointing


@dataclass
class PointTarget:
    range_m: float = 1.0
    rcs_dbsm: float = 5.0
    velocity: float = 0.0          # m/s, + toward radar
    name: str = "target"
    angle_deg: float = 0.0         # angle from boresight in the array plane (MIMO only)


@dataclass
class RainLayer:
    name: str = "rain"
    kind: str = "rain"             # rain | melting | snow | cloud (only used for labels)
    r_bottom: float = 100.0
    r_top: float = 2400.0
    dbz_bottom: float = 30.0
    dbz_top: float = 30.0
    dbz_peak: float | None = None  # triangular peak in the middle (bright band)
    v_bottom: float = 6.0
    v_top: float = 6.0
    sw_bottom: float = 1.5
    sw_top: float = 1.5
    att_scale: float = 1.0         # multiplies the rain k-R attenuation


@dataclass
class SceneConfig:
    kind: str = "rain"             # point | rain | both
    point_targets: list = field(default_factory=list)
    rain_layers: list = field(default_factory=list)
    cells_per_bin: int = 4
    attenuation: bool = True
    k_a: float = 0.263             # k[dB/km] = k_a * R^k_b  (approx. ITU at ~37 GHz)
    k_b: float = 0.979
    seed: int = 1
    wind: list = field(default_factory=list)   # [[height_m, u_m_s], ...] horizontal wind along the array axis


@dataclass
class ProcessingConfig:
    r_max: float = 6000.0
    beat_decimation: int | None = None   # None -> automatic
    range_window: str = "hann"
    range_zero_pad: int = 1
    clutter: str = "mean"          # none | mean
    blank_m: float = 0.0           # blank range bins closer than this
    doppler_window: str = "hann"
    snr_threshold_db: float = 0.0
    noise_method: str = "hs"       # hs (Hildebrand-Sekhon) | theory
    attenuation_correction: bool = True
    zr_a: float = 200.0
    zr_b: float = 1.6
    unfold: bool = False
    unfold_direction: str = "down"  # scan from far range toward the radar
    cal_offset_db: float = 0.0     # added to measured power (from corner-reflector calibration)
    range_offset_m: float = 0.0    # subtracted from ranges (cables / alignment, from calibration)


@dataclass
class MIMOConfig:
    enabled: bool = False
    n_tx: int = 4
    n_rx: int = 4
    rx_spacing: float = 0.5        # wavelengths
    tx_spacing: float = 2.0        # wavelengths; = n_rx * rx_spacing gives a filled 16-element virtual ULA
    scheme: str = "tdm"            # tdm: TX0, TX1, TX2, TX3, TX0, ... one TX per chirp
    angle_fft: int = 64
    angle_window: str = "hann"
    doppler_window: str = "hann"
    doppler_compensation: bool = True
    tx_gain_err_db: float = 0.0    # simulated per-channel gain / phase errors (to exercise calibration)
    tx_phase_err_deg: float = 0.0
    rx_gain_err_db: float = 0.0
    rx_phase_err_deg: float = 0.0
    calibration: list | None = None   # measured complex gain per virtual channel, [[re, im], ...]
    angle_cells: int = 16          # rain cells across angle per range cell (simulation)
    angle_span_deg: float = 30.0   # rain simulated within +/- this angle
    max_angle_deg: float = 20.0    # angle bins used for moments and wind retrieval
    wind_gates: int = 7            # range gates pooled in one wind fit (7 x 15 m ~ 105 m)


@dataclass
class PhysicsConfig:
    k2: float = 0.88
    t0: float = 290.0


@dataclass
class SimConfig:
    if_chirps: int = 2             # chirps simulated through the full IF/ADC/DDC chain
    add_noise: bool = True
    n_dwells: int = 1              # independent dwells averaged by the processing (spectral averaging)


@dataclass
class RadarConfig:
    name: str = "default"
    lo: LOConfig = field(default_factory=LOConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    adc: ADCConfig = field(default_factory=ADCConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    mimo: MIMOConfig = field(default_factory=MIMOConfig)

    @property
    def d(self) -> "Derived":
        return Derived(self)

    def to_dict(self) -> dict:
        return asdict(self)

    def replace(self, **patch) -> "RadarConfig":
        """Return a copy with nested overrides, e.g. ``cfg.replace(lo={'n_pll': 1112})``."""
        return config_from_dict(_deep_merge(self.to_dict(), patch))


_LIST_ITEM_TYPES = {"point_targets": PointTarget, "rain_layers": RainLayer}


def _build(cls, data: dict):
    obj = cls()
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        cur = getattr(obj, f.name)
        if f.name in _LIST_ITEM_TYPES:
            item_cls = _LIST_ITEM_TYPES[f.name]
            val = [v if isinstance(v, item_cls) else _build(item_cls, v) for v in (val or [])]
        elif is_dataclass(cur):
            val = _build(type(cur), val or {})
        elif isinstance(val, str) and isinstance(cur, float):
            val = float(val)          # YAML reads "1e9" as a string
        setattr(obj, f.name, val)
    return obj


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    base = data.pop("base", None)
    if base:
        data = _deep_merge(_read_yaml((path.parent / base).resolve()), data)
    return data


def config_from_dict(data: dict) -> RadarConfig:
    return _build(RadarConfig, data)


def load_config(path, overrides: dict | None = None) -> RadarConfig:
    """Load a YAML config. A ``base:`` key inherits from another YAML file."""
    data = _read_yaml(Path(path))
    if overrides:
        data = _deep_merge(data, overrides)
    return config_from_dict(data)


# --------------------------------------------------------------------------- derived
def _largest_divisor_at_most(n: int, limit: float) -> int:
    best = 1
    for k in range(1, int(limit) + 1):
        if n % k == 0:
            best = k
    return best


def window_enbw(name: str, n: int = 4096) -> float:
    """Equivalent noise bandwidth of a window in bins."""
    w = get_window(name, n)
    return n * np.sum(w ** 2) / np.sum(w) ** 2


def get_window(name: str, n: int) -> np.ndarray:
    import scipy.signal as sps
    if name in (None, "none", "rect", "boxcar"):
        return np.ones(n)
    return sps.get_window(name, n, fftbins=True)


class Derived:
    """Quantities computed from a RadarConfig (one place for all the physics bookkeeping)."""

    def __init__(self, cfg: RadarConfig):
        self.cfg = cfg
        lo, wf, adc, fe, pr = cfg.lo, cfg.waveform, cfg.adc, cfg.frontend, cfg.processing
        self.warnings: list[str] = []

        # --- frequency plan
        self.f_pll = lo.f_pll if lo.f_pll else lo.f_ref * lo.n_pll
        self.f_lo = self.f_pll * lo.multiplier
        if lo.sideband != "usb":
            raise NotImplementedError("simulation currently models USB only (RF = LO + IF)")
        self.f_rf_center = wf.f_rf_center
        self.f_if_center = wf.f_rf_center - self.f_lo
        self.f_if_lo = self.f_if_center - wf.bandwidth / 2
        self.f_if_hi = self.f_if_center + wf.bandwidth / 2
        self.lam = C / wf.f_rf_center

        # --- waveform
        self.slope = wf.bandwidth / wf.t_chirp
        self.t_rep = wf.t_chirp + wf.t_idle
        self.prf = 1.0 / self.t_rep
        self.delta_r = C / (2 * wf.bandwidth)
        self.v_max = self.lam / (4 * self.t_rep)
        self.t_dwell = self.t_rep * wf.n_chirps
        self.tau_max = 2 * pr.r_max / C
        self.fb_max = self.slope * self.tau_max

        # --- ADC / Nyquist zones
        self.fs = adc.fs
        self.zone_width = adc.fs / 2
        self.zone_lo = int(math.floor(self.f_if_lo / self.zone_width)) + 1
        self.zone_hi = int(math.floor(self.f_if_hi / self.zone_width)) + 1
        self.zone = fe.bpf_zone or self.zone_lo
        self.zone_inverted = self.zone % 2 == 0
        self.fs_analog = adc.fs * adc.analog_oversample
        self.f_nco = adc.nco_freq if adc.nco_freq is not None else self.f_if_center
        self.fs_ddc = adc.fs / adc.ddc_decimation
        self.n_chirp_ddc = int(round(wf.t_chirp * self.fs_ddc))
        self.n_rep_ddc = int(round(self.t_rep * self.fs_ddc))

        if self.zone_lo != self.zone_hi:
            self.warnings.append(
                f"IF band {self.f_if_lo/1e9:.3f}-{self.f_if_hi/1e9:.3f} GHz straddles Nyquist zones "
                f"{self.zone_lo}/{self.zone_hi} (edge {self.zone_lo*self.zone_width/1e9:.3f} GHz)")
        if abs(wf.t_chirp * self.fs_ddc - self.n_chirp_ddc) > 1e-6 or \
                abs(self.t_rep * self.fs_ddc - self.n_rep_ddc) > 1e-6:
            self.warnings.append("t_chirp / t_rep is not an integer number of DDC samples")
        if wf.bandwidth > 0.8 * self.fs_ddc:
            self.warnings.append(
                f"chirp bandwidth {wf.bandwidth/1e6:.0f} MHz exceeds 80% of DDC output rate "
                f"{self.fs_ddc/1e6:.1f} MSPS")

        # --- MIMO (TDM): each TX repeats every n_tx chirps
        mi = cfg.mimo
        self.mimo = mi.enabled
        self.n_tx = mi.n_tx if mi.enabled else 1
        self.t_tx = self.t_rep * self.n_tx
        self.v_max_mimo = self.lam / (4 * self.t_tx)
        self.n_virtual = mi.n_tx * mi.n_rx
        if mi.enabled:
            if wf.n_chirps % mi.n_tx:
                self.warnings.append("n_chirps must be a multiple of mimo.n_tx")
            if abs(mi.tx_spacing - mi.n_rx * mi.rx_spacing) > 1e-9:
                self.warnings.append("tx_spacing != n_rx * rx_spacing: virtual array is not a filled ULA")
            if self.v_max_mimo < 5:
                self.warnings.append(f"TDM-MIMO v_max is only {self.v_max_mimo:.2f} m/s (T per TX "
                                     f"{self.t_tx*1e6:.0f} us): rain will alias; shorten the chirp")

        # --- beat decimation
        if pr.beat_decimation:
            self.beat_decimation = pr.beat_decimation
        else:
            self.beat_decimation = _largest_divisor_at_most(
                math.gcd(self.n_chirp_ddc, self.n_rep_ddc), self.fs_ddc / (2.5 * self.fb_max))
        self.fs_beat = self.fs_ddc / self.beat_decimation
        self.n_beat = self.n_chirp_ddc // self.beat_decimation
        if self.fb_max > 0.45 * self.fs_beat:
            self.warnings.append("beat bandwidth too large for chosen beat decimation")
        if self.tau_max > 0.5 * wf.t_chirp:
            self.warnings.append("max delay is more than half the chirp: large SNR loss at far range")

        # --- gains and noise (Friis)
        self.k2 = cfg.physics.k2
        self.pt = float(dbm_to_w(fe.pt_dbm))
        self.g_tx = float(undb(fe.g_tx_dbi))
        self.g_rx = float(undb(fe.g_rx_dbi))
        self.loss = float(undb(fe.system_loss_db))
        self.theta = math.radians(fe.beamwidth_az_deg)
        self.phi = math.radians(fe.beamwidth_el_deg)
        f_mix = float(undb(fe.mixer_nf_db))
        if fe.lna.enabled:
            f_lna, g_lna = float(undb(fe.lna.nf_db)), float(undb(fe.lna.gain_db))
            self.f_rx = f_lna + (f_mix - 1) / g_lna
            self.g_chain = float(undb(fe.rx_gain_db + fe.lna.gain_db))
        else:
            self.f_rx = f_mix
            self.g_chain = float(undb(fe.rx_gain_db))
        self.nf_rx_db = float(db(self.f_rx))
        kt0 = K_B * cfg.physics.t0
        self.n0_thermal = kt0 * self.f_rx * self.g_chain           # W/Hz at ADC input
        self.p_fs = float(dbm_to_w(adc.full_scale_dbm))
        self.a_fs = math.sqrt(2 * self.p_fs)                       # full-scale peak amplitude
        self.n0_adc = self.p_fs * float(undb(adc.nsd_dbfs_hz))     # W/Hz
        g, ph = float(undb(fe.iq_gain_imbalance_db / 2)), math.radians(fe.iq_phase_imbalance_deg)
        # complex IQ-imbalance model  z' = mu z + nu conj(z)
        self.iq_mu = 0.5 * (1 + g * np.exp(-1j * ph))
        self.iq_nu = 0.5 * (1 - g * np.exp(1j * ph))
        self.irr_db = float(db(abs(self.iq_mu) ** 2 / abs(self.iq_nu) ** 2))
        # image band noise leaks through with 1/IRR
        self.n0_total = self.n0_thermal * (1 + 1 / undb(self.irr_db)) + self.n0_adc
        self.nf_eff_db = float(db(self.n0_total / (kt0 * self.g_chain)))

        # --- leakage level at ADC input
        self.p_leak_adc = self.pt * undb(-cfg.channel.tx_rx_isolation_db) * self.g_chain
        self.adc_headroom_db = float(db(self.p_fs / self.p_leak_adc))
        if self.adc_headroom_db < 3:
            self.warnings.append(
                f"TX leakage at ADC is {w_to_dbm(self.p_leak_adc):.1f} dBm, only "
                f"{self.adc_headroom_db:.1f} dB below full scale")

    # convenience -------------------------------------------------------------------
    def summary(self) -> str:
        c = self.cfg
        rows = [
            ("PLL", f"{self.f_pll/1e9:.5f} GHz  (N={c.lo.n_pll}, ref {c.lo.f_ref/1e6:.2f} MHz)"),
            ("LO at mixer", f"{self.f_lo/1e9:.5f} GHz"),
            ("RF centre / lambda", f"{self.f_rf_center/1e9:.3f} GHz / {self.lam*1e3:.3f} mm"),
            ("IF band", f"{self.f_if_lo/1e9:.4f} - {self.f_if_hi/1e9:.4f} GHz"),
            ("Nyquist zone", f"{self.zone_lo}..{self.zone_hi}  (zone width {self.zone_width/1e9:.3f} GHz)"),
            ("NCO", f"{self.f_nco/1e9:.5f} GHz"),
            ("Chirp", f"B={c.waveform.bandwidth/1e6:g} MHz, T={c.waveform.t_chirp*1e6:g} us, "
                      f"S={self.slope:.3e} Hz/s"),
            ("Range res.", f"{self.delta_r:.3f} m"),
            ("v_max (Nyquist)", f"+/-{self.v_max:.2f} m/s  (T_rep {self.t_rep*1e6:g} us)"),
            ("Dwell", f"{c.waveform.n_chirps} chirps = {self.t_dwell*1e3:.2f} ms"),
            *([("MIMO (TDM)", f"{c.mimo.n_tx} TX x {c.mimo.n_rx} RX = {self.n_virtual} virtual, "
                              f"T per TX {self.t_tx*1e6:g} us, v_max +/-{self.v_max_mimo:.2f} m/s")]
              if self.mimo else []),
            ("fb_max @ r_max", f"{self.fb_max/1e6:.3f} MHz  (r_max {c.processing.r_max:g} m)"),
            ("DDC", f"/{c.adc.ddc_decimation} -> {self.fs_ddc/1e6:.3f} MSPS, "
                    f"{self.n_chirp_ddc} samples/chirp"),
            ("Beat decimation", f"/{self.beat_decimation} -> {self.fs_beat/1e6:.4f} MSPS, "
                                f"{self.n_beat} samples/chirp"),
            ("RX NF (chain)", f"{self.nf_rx_db:.2f} dB, incl. ADC {self.nf_eff_db:.2f} dB"),
            ("IQ image rejection", f"{self.irr_db:.1f} dB"),
            ("Leakage @ ADC", f"{w_to_dbm(self.p_leak_adc):.1f} dBm (headroom {self.adc_headroom_db:.1f} dB)"),
        ]
        w = max(len(k) for k, _ in rows)
        text = "\n".join(f"  {k:<{w}} : {v}" for k, v in rows)
        if self.warnings:
            text += "\n  WARNINGS:\n" + "\n".join(f"   ! {m}" for m in self.warnings)
        return text
