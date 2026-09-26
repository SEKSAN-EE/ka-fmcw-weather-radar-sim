"""Real ZCU216 captures -> IQFrame.

A capture is a binary sample file plus a YAML sidecar with the same stem:

    myrun.bin   raw samples
    myrun.yaml  CaptureMeta fields (see docs/real_data.md)

Two formats are supported:

* ``adc_real`` — raw RF-ADC samples (real, int16, one channel), i.e. the IF signal *before*
  any DDC. Processed with the same software DDC as the simulation (:func:`radar_sim.adc.ddc`).
* ``ddc_iq``   — complex samples after the hardware DDC, interleaved I,Q int16.

The chirp start inside the record is found by correlating with the reference chirp: the TX->RX
leakage is by far the strongest echo, so the correlation peak marks the chirp boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from ..adc import codes_to_signal, ddc, ddc_filter
from ..config import RadarConfig
from ..iq import IQFrame
from ..waveform import reference_chirp


@dataclass
class CaptureMeta:
    format: str = "adc_real"          # adc_real | ddc_iq
    fs: float = 2.5e9                 # sample rate of the samples in the file
    dtype: str = "int16"
    byte_order: str = "<"
    msb_aligned: bool = False         # True if 14-bit codes sit in the top bits of 16-bit words
    adc_bits: int = 14
    f_nco: float | None = None        # ddc_iq: NCO used by the hardware DDC
    iq_gain_db: float = 0.0           # ddc_iq: code-to-ADC-input scaling correction (calibrate!)
    iq_full_scale: float = 32768.0    # ddc_iq: |I+jQ| code produced by a full-scale ADC sine (measure!)
    first_chirp_sample: int | None = None   # None -> find it by correlation
    n_chirps: int | None = None       # None -> as many complete chirps as the file holds
    channel: int = 0
    n_channels: int = 1               # MIMO: RX channels interleaved sample by sample (ch0, ch1, ..., ch0, ...)
    first_chirp_tx: int = 0           # MIMO/TDM: TX index of the first complete chirp in the record
    notes: str = ""
    extra: dict = field(default_factory=dict)


def read_meta(path) -> CaptureMeta:
    path = Path(path)
    with open(path.with_suffix(".yaml")) as fh:
        data = yaml.safe_load(fh) or {}
    known = {f.name for f in fields(CaptureMeta)}
    meta = CaptureMeta(**{k: v for k, v in data.items() if k in known})
    meta.extra.update({k: v for k, v in data.items() if k not in known})
    for k in ("fs", "f_nco"):
        if isinstance(getattr(meta, k), str):
            setattr(meta, k, float(getattr(meta, k)))
    return meta


def load_capture(path):
    """Return (samples, meta). adc_real -> int codes; ddc_iq -> complex codes."""
    path = Path(path)
    meta = read_meta(path)
    if path.suffix == ".npy":
        raw = np.load(path)
    else:
        raw = np.fromfile(path.with_suffix(".bin") if path.suffix != ".bin" else path,
                          dtype=np.dtype(meta.dtype).newbyteorder(meta.byte_order))
    nc = max(int(meta.n_channels), 1)
    if meta.format == "adc_real":
        codes = raw.astype(np.int32)
        if meta.msb_aligned:
            codes >>= 16 - meta.adc_bits
        if nc > 1:
            codes = codes[: codes.size - codes.size % nc].reshape(-1, nc).T      # (n_channels, N)
        return codes, meta
    if meta.format == "ddc_iq":
        if np.iscomplexobj(raw):
            z = raw
        else:
            z = raw[0::2].astype(float) + 1j * raw[1::2].astype(float)            # I,Q per sample
        if nc > 1:
            z = z[: z.size - z.size % nc].reshape(-1, nc).T
        return z, meta
    raise ValueError(f"unknown capture format {meta.format!r}")


def write_capture(path, samples, meta: CaptureMeta):
    """Write samples + YAML sidecar (used for test captures and as a format reference)."""
    path = Path(path).with_suffix(".bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = np.dtype(meta.dtype).newbyteorder(meta.byte_order)
    samples = np.asarray(samples)
    if samples.ndim == 2:                          # (n_channels, N) -> interleave sample by sample
        meta.n_channels = samples.shape[0]
        samples = samples.T.reshape(-1)
    if meta.format == "ddc_iq" and np.iscomplexobj(samples):
        inter = np.empty(2 * samples.size, dt)
        inter[0::2], inter[1::2] = np.round(samples.real), np.round(samples.imag)
        inter.tofile(path)
    else:
        s = np.asarray(samples)
        if meta.msb_aligned:
            s = s.astype(np.int32) << (16 - meta.adc_bits)
        s.astype(dt).tofile(path)
    d = asdict(meta)
    d.update(d.pop("extra") or {})
    with open(path.with_suffix(".yaml"), "w") as fh:
        yaml.safe_dump(d, fh, sort_keys=False)
    return path


# --------------------------------------------------------------------------- DDC for long records
def ddc_stream(cfg: RadarConfig, x_real, block: int = 1 << 22, f_nco: float | None = None):
    """Software DDC of a long real record in blocks (bounded memory, identical to ``ddc``)."""
    D = cfg.adc.ddc_decimation
    L = ddc_filter(cfg).size
    ov = int(np.ceil(L / D)) * D
    block = max(block // D, 1) * D
    out = []
    n = x_real.size - x_real.size % D
    for start in range(0, n, block):
        s0 = max(0, start - ov)
        seg = np.asarray(x_real[s0:min(n, start + block + ov)], float)
        y = ddc(cfg, seg, n0=s0, f_nco=f_nco)
        k0 = (start - s0) // D
        out.append(y[k0:k0 + min(block, n - start) // D])
    return np.concatenate(out)


# --------------------------------------------------------------------------- alignment
def find_chirp_start(cfg: RadarConfig, x, fs: float, freq_offset: float, n_periods: int = 3,
                     rel_threshold_db: float = 20.0) -> tuple[float, np.ndarray]:
    """First chirp start in ``x`` (complex, DDC rate), in fractional samples, and |correlation|.

    The direct TX->RX leakage is the *earliest* echo, but not always the strongest (a corner
    reflector at 1 m can be stronger). So: fold the correlation onto one chirp period, find the
    strongest lag, then take the earliest local peak within r_max before it that is no more than
    ``rel_threshold_db`` below it. Parabolic interpolation gives the sub-sample position.
    """
    d = cfg.d
    n_rep = int(round(d.t_rep * fs))
    # Hann taper on the reference = spectral weighting of the chirp -> correlation sidelobes < -31 dB,
    # so the sidelobes of a strong echo are not mistaken for an earlier (leakage) peak
    ref = reference_chirp(cfg, fs, freq_offset) * np.hanning(int(round(cfg.waveform.t_chirp * fs)))
    seg = x[: min(x.size, (n_periods + 1) * n_rep)]
    nfft = 1 << int(np.ceil(np.log2(seg.size + ref.size)))
    corr = np.fft.ifft(np.fft.fft(seg, nfft) * np.conj(np.fft.fft(ref, nfft)))[: seg.size - ref.size + 1]
    mag = np.abs(corr)
    folded = np.zeros(n_rep)
    for k in range(0, mag.size, n_rep):
        part = mag[k:k + n_rep]
        folded[: part.size] += part
    i_max = int(np.argmax(folded))
    thr = folded[i_max] * 10 ** (-rel_threshold_db / 20)
    search = int(np.ceil(2 * cfg.processing.r_max / 299_792_458.0 * fs)) + 2
    best = i_max
    for lag in range(search, 0, -1):                       # earliest first
        i = (i_max - lag) % n_rep
        a, b, c = folded[(i - 1) % n_rep], folded[i], folded[(i + 1) % n_rep]
        if b >= thr and b >= a and b >= c:
            best = i
            break
    a, b, c = folded[(best - 1) % n_rep], folded[best], folded[(best + 1) % n_rep]
    den = a - 2 * b + c
    frac = 0.5 * (a - c) / den if den != 0 else 0.0
    return best + float(np.clip(frac, -0.5, 0.5)), mag


def _fractional_shift(x, shift: float):
    """Advance ``x`` by ``shift`` samples (0 <= shift < 1) with an FFT phase ramp."""
    if abs(shift) < 1e-6:
        return x
    f = np.fft.fftfreq(x.size)
    return np.fft.ifft(np.fft.fft(x) * np.exp(2j * np.pi * f * shift))


def capture_to_frame(samples, meta: CaptureMeta, cfg: RadarConfig) -> tuple[IQFrame, dict]:
    """Convert raw capture samples into an IQFrame (chirp-aligned, sqrt(W) at the ADC input).

    One channel -> data (n_chirps, n_rep). Several RX channels (MIMO) -> data (n_chirps, n_rx, n_rep);
    all channels share the chirp alignment found on channel 0.
    """
    d = cfg.d
    info: dict = {}
    multi = np.ndim(samples) == 2
    chans = samples if multi else np.asarray(samples)[None, :]
    if meta.format == "adc_real":
        if abs(meta.fs - cfg.adc.fs) > 1:
            raise ValueError(f"capture fs {meta.fs} != config adc.fs {cfg.adc.fs}")
        c0 = chans[0]
        info["adc_codes"] = c0[: min(c0.size, 1 << 20)]
        info["peak_dbfs"] = float(20 * np.log10(np.max(np.abs(chans)) / 2 ** (cfg.adc.bits - 1) + 1e-30))
        info["clipped"] = int(np.sum(np.abs(chans) >= 2 ** (cfg.adc.bits - 1) - 1))
        xs = []
        for ch in chans:
            x_adc = codes_to_signal(cfg, ch)
            if meta.first_chirp_sample is not None:      # known from MTS / trigger: cut before the DDC
                x_adc = x_adc[int(meta.first_chirp_sample):]
            if not xs:
                info["adc_signal"] = x_adc[: min(x_adc.size, 1 << 20)]
            xs.append(ddc_stream(cfg, x_adc))
        x = np.array(xs)
        fs, f_nco = d.fs_ddc, d.f_nco
    else:
        if abs(meta.fs - d.fs_ddc) > 1:
            raise ValueError(f"capture fs {meta.fs} != config DDC rate {d.fs_ddc}")
        scale = d.a_fs / meta.iq_full_scale * 10 ** (meta.iq_gain_db / 20) / np.sqrt(2)
        x = chans * scale
        fs, f_nco = meta.fs, (meta.f_nco if meta.f_nco is not None else d.f_nco)
    freq_offset = d.f_if_center - f_nco
    n_rep = int(round(d.t_rep * fs))
    if meta.first_chirp_sample is not None:
        start = 0 if meta.format == "adc_real" else int(meta.first_chirp_sample)
        info["alignment"] = "from metadata"
    else:
        start_f, corr = find_chirp_start(cfg, x[0], fs, freq_offset)
        start = int(np.floor(start_f))
        x = np.array([_fractional_shift(xi, start_f - start) for xi in x])
        info["alignment"] = "correlation with reference chirp (earliest strong echo = leakage)"
        info["correlation"] = corr
        info["start_fractional"] = start_f
    if multi and cfg.mimo.enabled:                 # start the record at a TX0 chirp
        drop = (cfg.mimo.n_tx - meta.first_chirp_tx) % cfg.mimo.n_tx
        start += drop * n_rep
        info["dropped_chirps_for_tx0"] = drop
    info["start_sample_ddc"] = start
    n_avail = (x.shape[1] - start) // n_rep
    n_chirps = min(meta.n_chirps or n_avail, n_avail)
    if multi and cfg.mimo.enabled:
        n_chirps -= n_chirps % cfg.mimo.n_tx
    data = x[:, start:start + n_chirps * n_rep].reshape(x.shape[0], n_chirps, n_rep).transpose(1, 0, 2)
    if not multi:
        data = data[:, 0, :]
    frame = IQFrame(data=data, fs=fs, freq_offset=freq_offset, source=f"capture:{meta.format}",
                    meta={"notes": meta.notes, **meta.extra})
    return frame, info


def split_dwells(frame: IQFrame, chirps_per_dwell: int) -> list[IQFrame]:
    n = frame.n_chirps // chirps_per_dwell
    return [IQFrame(frame.data[k * chirps_per_dwell:(k + 1) * chirps_per_dwell], frame.fs, frame.freq_offset,
                    frame.source, frame.meta) for k in range(max(n, 1))]
