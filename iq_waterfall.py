#!/usr/bin/env python3
"""Waterfall plot and signal statistics for a complex IQ WAV written by airspy_live.py's IQ recorder.

Input: stereo WAV, left = I, right = Q. 32-bit float (as written by the recorder) or 16-bit PCM.
The sidecar JSON (<name>_info.json, found automatically) supplies the tuning frequency, the
sample rate if the header disagrees, and the displayed window. Frequencies are absolute RF.

The recording is split into a carrier-on interval, then into unmodulated and modulated segments.
For each segment the program prints duration, carrier and sideband power, energy shares, three
bandwidth measures, SNR in a 2.5 kHz noise bandwidth, and for modulated segments the tone frequency
and deviation (from the instantaneous frequency). Boundaries are drawn on the waterfall.

All power levels are relative (the recorder does not calibrate them). Differences and SNR are valid.

Requires: numpy, matplotlib.

Examples:
    ./iq_waterfall.py iq_20261126_015909_145600kHz.wav
    ./iq_waterfall.py rec.wav --nfft 8192 --span-khz 10 --save rec.png
    ./iq_waterfall.py rec.wav --start 1.0 --stop 6.0 --no-plot
"""
import argparse
import json
import os
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np

CARRIER_ON_DB = 6.0       # total in-span power this far above the noise: carrier present
CARRIER_OFF_DB = 3.0      # carrier counts as gone below this (hysteresis)
MOD_ON_DB = 6.0           # power outside the carrier this far above the noise: modulated
MOD_OFF_DB = 3.0          # modulation ends below this (hysteresis)
CARRIER_HALF_BINS = 2     # carrier power is summed this many bins either side of the carrier bin
CHANNEL_HZ = 2500.0       # SNR reference noise bandwidth
MARGIN_DB = 6.0           # "bandwidth above noise": bins this far above the noise per bin
MIN_SEG_S = 0.25          # shorter runs (key-up transients, blips) are merged into a neighbour
EDGE_SKIP_S = 0.05        # samples this close to a segment edge are left out of the tone analysis
NOISE_QUIET_PCT = 30      # noise is taken from the quietest this percentage of frames
SETTLE_STEADY_S = 0.5     # steady-state reference: median width over the last this much of the lead-in
SETTLE_TOL_FRAC = 0.2     # settled when the smoothed BW+6 is within this fraction of its steady value (min half a bin)
SETTLE_HOLD_S = 0.03      # ...and stays there for this long
SETTLE_MIN_S = 0.2        # the settled part must last at least this long for a transient to be split off
TRANSIENT_MIN_S = 0.04    # a start transient shorter than this is not split off (onset frames straddle the edge)
AUDIO_FS = 48000.0        # audio rate for the sequence search (IQ rate / 4)
SEQ_CHIP_RATE = 2400.0    # 1023-chip sequence: chip rate, chips/s (as in syncgen)
SEQ_CHIPS = 1023
SEQ_ALPHA = 0.10          # raised-cosine rolloff
SEQ_FC = 1650.0           # audio carrier of the sequence, Hz
SEQ_SPAN = 12             # pulse half-length in chips
SEQ_SPC = int(round(AUDIO_FS / SEQ_CHIP_RATE))   # samples per chip at AUDIO_FS (20)
SEQ_DET_SIGMA = 12.0      # detected when the correlation peak is this many robust sigmas above the noise
SEQ_PSL_EXCL_CHIPS = 2    # peak-to-sidelobe is measured beyond this many chips from the peak
SEQ_MIN_PSL_DB = 8.0      # and the peak must stand this far above the largest sidelobe (noise peaks do not)
TONE_MIN_SHARE = 0.5      # a modulated segment is a tone only if its strongest audio line carries this share of the power


def read_iq_wav(path):
    """Return (iq as complex64, sample rate in Hz, notes). Walks the RIFF chunks, so extra chunks are
    tolerated. Bytes after the data chunk, or a partial last sample, are ignored and reported in notes."""
    notes = []
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            sys.exit(f"{path}: not a RIFF/WAVE file")
        fmt = None
        data_off = data_len = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, csize = struct.unpack("<4sI", hdr)
            if cid == b"fmt ":
                body = f.read(csize)
                tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
                if tag == 0xFFFE and len(body) >= 26:          # WAVE_FORMAT_EXTENSIBLE: sub-format
                    tag = struct.unpack("<H", body[24:26])[0]
                fmt = (tag, ch, rate, bits)
            elif cid == b"data":
                data_off, data_len = f.tell(), csize
                break
            else:
                f.seek(csize + (csize & 1), 1)
            if csize & 1:
                f.seek(1, 1)
    if fmt is None or data_off is None:
        sys.exit(f"{path}: missing fmt or data chunk")
    tag, ch, rate, bits = fmt
    if ch != 2:
        sys.exit(f"{path}: expected 2 channels (I, Q), found {ch}")
    if tag == 3 and bits == 32:
        dtype, scale = "<f4", 1.0
    elif tag == 1 and bits == 16:
        dtype, scale = "<i2", 1.0 / 32768.0
    else:
        sys.exit(f"{path}: unsupported format tag {tag}, {bits} bits")

    frame_bytes = 2 * np.dtype(dtype).itemsize
    avail = size - data_off
    if data_len > avail:
        notes.append(f"data chunk says {data_len} bytes but file has {avail}; using what is there")
        data_len = avail
    elif avail > data_len:
        notes.append(f"{avail - data_len} bytes follow the data chunk and were ignored")
    n = data_len // frame_bytes
    if data_len % frame_bytes:
        notes.append(f"{data_len % frame_bytes} bytes at the end are a partial sample and were ignored")
    raw = np.memmap(path, dtype=dtype, mode="r", offset=data_off, shape=(n, 2))
    iq = (raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)) * np.float32(scale)
    return iq.astype(np.complex64), rate, notes


def load_sidecar(wav_path, explicit):
    """The sidecar JSON as a dict, or {} when there is none."""
    path = explicit or os.path.splitext(wav_path)[0] + "_info.json"
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def stft_power(iq, rate, nfft, hop, keep):
    """Linear power per frame and kept bin, normalised by the window energy. Returns (P [frames, bins],
    frame centre times in s from the start of iq)."""
    win = np.hanning(nfft).astype(np.float32)
    wnorm = float(np.sum(win.astype(np.float64) ** 2))
    n_frames = 1 + (len(iq) - nfft) // hop
    if n_frames < 1:
        sys.exit("recording is shorter than one FFT frame")
    frames = np.lib.stride_tricks.sliding_window_view(iq, nfft)[::hop]
    P = np.empty((n_frames, int(keep.sum())), dtype=np.float32)
    for i0 in range(0, n_frames, 512):
        i1 = min(n_frames, i0 + 512)
        X = np.fft.fftshift(np.fft.fft(frames[i0:i1] * win, axis=1), axes=1)[:, keep]
        P[i0:i1] = (X.real * X.real + X.imag * X.imag) / wnorm
    times = (np.arange(n_frames) * hop + nfft / 2.0) / rate
    return P, times


def hysteresis(x, on, off):
    """Boolean state that turns on above `on` and off below `off`."""
    out = np.zeros(len(x), dtype=bool)
    state = False
    for i, v in enumerate(x):
        state = (v > off) if state else (v > on)
        out[i] = state
    return out


def runs_of(flags):
    """Runs of equal values as [first, last, value] with inclusive indices."""
    out, start = [], 0
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != flags[start]:
            out.append([start, i - 1, bool(flags[start])])
            start = i
    return out


def clean_runs(flags, min_frames):
    """Runs after merging any run shorter than min_frames into the run before it (or after, if first)."""
    f = flags.copy()
    while True:
        rs = runs_of(f)
        short = [r for r in rs if r[1] - r[0] + 1 < min_frames]
        if len(rs) <= 1 or not short:
            return rs
        a, b, _ = short[0]
        f[a:b + 1] = f[a - 1] if a > 0 else f[b + 1]


def bandwidths(mean_p, noise, df):
    """Three bandwidths of a time-averaged spectrum, in kHz: 99% of the noise-subtracted power, the
    outermost extent within 26 dB of the peak, and the outermost extent above the noise by MARGIN_DB."""
    sig = mean_p - noise
    pos = np.clip(sig, 0.0, None)
    tot = pos.sum()
    bw99 = np.nan
    if tot > 0:
        cum = np.cumsum(pos)
        lo = int(np.searchsorted(cum, 0.005 * tot))
        hi = int(np.searchsorted(cum, 0.995 * tot))
        bw99 = (hi - lo + 1) * df / 1e3
    bw26 = np.nan
    pk = sig.max()
    if pk > 0:
        idx = np.flatnonzero(sig >= pk * 10 ** (-2.6))
        bw26 = (idx[-1] - idx[0] + 1) * df / 1e3
    idx = np.flatnonzero(mean_p >= noise * 10 ** (MARGIN_DB / 10.0))
    bwm = (idx[-1] - idx[0] + 1) * df / 1e3 if idx.size else np.nan
    return bw99, bw26, bwm


def parabolic_peak(y, k):
    """Fractional offset of a peak at index k from log values y[k-1], y[k], y[k+1]."""
    if not 0 < k < len(y) - 1:
        return 0.0
    y0, y1, y2 = np.log(y[k - 1:k + 2] + 1e-30)
    den = y0 - 2 * y1 + y2
    return 0.5 * (y0 - y2) / den if den < 0 else 0.0


def tone_frequency(x, rate):
    """Frequency in Hz of the strongest component of a real sequence x (50 Hz to 20 kHz)."""
    n = len(x)
    nz = 1 << int(np.ceil(np.log2(4 * n)))
    X = np.abs(np.fft.rfft(x * np.hanning(n), nz))
    f = np.fft.rfftfreq(nz, 1.0 / rate)
    band = np.flatnonzero((f >= 50.0) & (f <= 20000.0))
    k = int(band[np.argmax(X[band])])
    return float(f[k] + parabolic_peak(X, k) * (f[1] - f[0]))


def deviation(x, fm, rate):
    """Peak deviation in Hz of x at tone fm, x already free of its mean. Projecting onto a sinusoid at fm
    gives its amplitude; wideband phase noise in the per-sample frequency averages out of that projection,
    which is why a plain standard deviation is not used (it is dominated by the noise)."""
    t = np.arange(len(x)) / rate
    c = np.mean(x * np.exp(-2j * np.pi * fm * t))
    return 2.0 * abs(c)


def running_mean_rows(P, w):
    """Centred running mean over w frames (w odd) for a [frames, bins] array, edges replicated."""
    h = w // 2
    Pp = np.pad(P, ((h, h), (0, 0)), mode="edge")
    c = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), Pp]), axis=0)
    return (c[w:] - c[:-w]) / w


def frame_occupied_khz(Pa, N, df):
    """Per-frame occupied width in kHz: the number of bins MARGIN_DB or more above the noise N (per bin), times
    the bin width. A count rather than an outermost span, so a stray noise bin far out cannot widen it. Applied
    to frames averaged in time, which keeps the chance of a noise bin over the threshold low."""
    return (Pa >= N * 10 ** (MARGIN_DB / 10.0)).sum(axis=1) * df / 1e3


def running_median(x, w):
    """Centred running median over w values (w odd), edges replicated."""
    h = w // 2
    xp = np.pad(x, h, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, w), axis=1)


def settle_index(bw, steady, tol, hold):
    """Index of the first entry that starts a run of `hold` consecutive entries within tol of steady. A brief
    excursion after the spectrum has settled does not move this point. Returns len(bw) if there is no such run."""
    good = np.abs(bw - steady) <= tol
    if len(good) < hold:
        return len(good)
    runs = np.lib.stride_tricks.sliding_window_view(good, hold).all(axis=1)
    idx = np.flatnonzero(runs)
    return int(idx[0]) if idx.size else len(good)


def sequence_reference():
    """Audio waveform of the 1023-chip sequence, generated as syncgen does: same LFSR (x^10 + x^7 + 1), chip
    rate, raised-cosine pulse and audio carrier, with the first chip centred on sample 0."""
    state = [1] * 10
    bits = []
    for _ in range(SEQ_CHIPS):
        bits.append(state[-1])
        fb = state[9] ^ state[6]
        state = [fb] + state[:-1]
    chips = 1.0 - 2.0 * np.array(bits, dtype=float)
    up = np.zeros(len(chips) * SEQ_SPC)
    up[::SEQ_SPC] = chips
    T = 1.0 / SEQ_CHIP_RATE
    k = np.arange(-SEQ_SPAN * SEQ_SPC, SEQ_SPAN * SEQ_SPC + 1)
    t = k / AUDIO_FS / T
    denom = 1.0 - (2 * SEQ_ALPHA * t) ** 2
    sing = np.isclose(denom, 0.0, atol=1e-9)
    h = np.sinc(t) * np.cos(np.pi * SEQ_ALPHA * t) / np.where(sing, 1.0, denom)
    h[sing] = (np.pi / 4) * np.sinc(1 / (2 * SEQ_ALPHA))
    h /= np.max(np.abs(h))
    base = np.convolve(up, h, mode="same")
    tt = np.arange(len(base)) / AUDIO_FS
    return base * np.cos(2 * np.pi * SEQ_FC * tt)


def matched_filter(x, r):
    """Normalised cross-correlation of x with reference r at every lag where r lies wholly inside x. Returns
    (ncc in -1..1, energy of x under r at each lag, energy of r)."""
    L = len(r)
    nfft = 1 << int(np.ceil(np.log2(len(x) + L)))
    c = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(r, nfft)), nfft)[:len(x) - L + 1]
    e = np.concatenate([[0.0], np.cumsum(x * x)])
    energy = e[L:] - e[:-L]
    er = float(np.dot(r, r))
    return c / np.sqrt(np.maximum(energy * er, 1e-30)), energy, er


def tone_test(x, fs, f_lo=300.0, f_hi=3000.0):
    """Strongest line of audio x within f_lo..f_hi, refined to a fraction of a bin, and the share of the in-band
    power it carries: the power of a sinusoid at that frequency over the mean power of x within f_lo..f_hi.
    Only the band counts, because FM demodulator noise above the voice band would otherwise dominate."""
    n = len(x)
    xc = x - np.mean(x)
    nz = 1 << int(np.ceil(np.log2(4 * n)))
    f = np.fft.rfftfreq(nz, 1.0 / fs)
    inband = (f >= f_lo) & (f <= f_hi)
    xb = np.fft.irfft(np.fft.rfft(xc, nz) * inband, nz)[:n]
    X = np.abs(np.fft.rfft(xb * np.hanning(n), nz)) ** 2
    band = np.flatnonzero(inband)
    k = int(band[np.argmax(X[band])])
    fm = float(f[k] + parabolic_peak(X, k) * (f[1] - f[0]))
    c = np.mean(xb * np.exp(-2j * np.pi * fm * np.arange(n) / fs))
    return fm, float(2 * abs(c) ** 2 / np.mean(xb ** 2))


def fmt(v, width, prec):
    return f"{v:>{width}.{prec}f}" if np.isfinite(v) else f"{'--':>{width}}"


def main():
    ap = argparse.ArgumentParser(description="Waterfall and signal statistics of an IQ WAV file")
    ap.add_argument("wav", help="IQ WAV file written by the recorder")
    ap.add_argument("--sidecar", help="JSON sidecar (default: <wav name>_info.json if present)")
    ap.add_argument("--nfft", type=int, default=4096, help="FFT length (default 4096, 47 Hz bins at 192 kS/s)")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="fraction of each FFT frame shared with the next (default 0.5)")
    ap.add_argument("--start", type=float, default=0.0, help="first second to analyse and plot")
    ap.add_argument("--stop", type=float, default=None, help="last second to analyse and plot")
    ap.add_argument("--center-khz", type=float, default=None,
                    help="centre of the RF span, kHz from the tuning frequency (default: sidecar window)")
    ap.add_argument("--span-khz", type=float, default=None,
                    help="width of the RF span in kHz (default: sidecar window, else full bandwidth)")
    ap.add_argument("--vmin", type=float, default=None, help="colour scale minimum, dB")
    ap.add_argument("--vmax", type=float, default=None, help="colour scale maximum, dB")
    ap.add_argument("--save", metavar="PNG", help="save the figure instead of showing it")
    ap.add_argument("--no-plot", action="store_true", help="print the statistics only")
    ap.add_argument("--debug-transient", action="store_true",
                    help="print the per-frame BW+6 trace used to find the start transient")
    args = ap.parse_args()

    if not 0.0 <= args.overlap < 1.0:
        ap.error("--overlap must be 0 <= x < 1")
    if args.nfft < 64 or args.nfft & (args.nfft - 1):
        ap.error("--nfft must be a power of 2, at least 64")

    iq, rate, notes = read_iq_wav(args.wav)
    side = load_sidecar(args.wav, args.sidecar)
    if side.get("sample_rate_hz") and int(side["sample_rate_hz"]) != rate:
        notes.append(f"sidecar sample rate {side['sample_rate_hz']} differs from WAV header {rate}; using {rate}")
    tune_hz = float(side.get("tune_hz", 0.0))
    if "tune_hz" not in side:
        notes.append("no sidecar: frequencies are relative to the tuning frequency")

    i0 = max(0, int(round(args.start * rate)))
    i1 = len(iq) if args.stop is None else min(len(iq), int(round(args.stop * rate)))
    if i1 - i0 < args.nfft:
        sys.exit("selected time range is shorter than one FFT frame")
    iq = iq[i0:i1]
    t_offset = i0 / rate

    nfft = args.nfft
    hop = max(1, int(round(nfft * (1.0 - args.overlap))))
    df = rate / nfft
    freqs_full = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / rate))

    # RF span: sidecar window unless overridden
    win_span = side["window_hi_hz"] - side["window_lo_hz"] if "window_lo_hz" in side else rate
    win_c = ((side["window_lo_hz"] + side["window_hi_hz"]) / 2 - tune_hz) if "window_lo_hz" in side else 0.0
    span_hz = args.span_khz * 1e3 if args.span_khz is not None else win_span
    c_hz = args.center_khz * 1e3 if args.center_khz is not None else win_c
    keep = (freqs_full >= c_hz - span_hz / 2) & (freqs_full <= c_hz + span_hz / 2)
    if not keep.any():
        sys.exit("requested RF span has no bins; check --center-khz / --span-khz")
    freqs = freqs_full[keep]

    P, times = stft_power(iq, rate, nfft, hop, keep)
    nf = len(times)
    frame_s = hop / rate

    # noise per bin: median of the quietest frames, corrected to a mean (exponential statistics)
    T = P.sum(axis=1)
    quiet = T <= np.percentile(T, NOISE_QUIET_PCT)
    N = np.median(P[quiet], axis=0) / np.log(2.0)
    N_tot = float(N.sum())

    carrier = hysteresis(10.0 * np.log10(T / N_tot + 1e-30), CARRIER_ON_DB, CARRIER_OFF_DB)
    if not carrier.any():
        print("no carrier found above the noise in the displayed span")
        return
    on_idx = np.flatnonzero(carrier)
    i_on, i_off = int(on_idx[0]), int(on_idx[-1])

    # carrier bin: peak of the mean spectrum over carrier-on frames, refined to a fraction of a bin
    Pm = P[carrier].mean(axis=0)
    k0 = int(np.argmax(Pm))
    cmask = np.abs(np.arange(len(freqs)) - k0) <= CARRIER_HALF_BINS
    omask = ~cmask

    c_pow = (P[:, cmask] - N[cmask]).sum(axis=1)       # carrier line power above noise, per frame
    s_pow = (P[:, omask] - N[omask]).sum(axis=1)       # power outside the carrier, above noise
    P_out = P[:, omask].sum(axis=1)
    N_out = float(N[omask].sum())
    mod = hysteresis(10.0 * np.log10(np.maximum(P_out, 1e-30) / N_out), MOD_ON_DB, MOD_OFF_DB) & carrier

    # segments inside the carrier-on interval
    rs = clean_runs(mod[i_on:i_off + 1], max(1, int(round(MIN_SEG_S / frame_s))))
    n_mod = sum(1 for r in rs if r[2])
    mod_js = [j for j, r in enumerate(rs) if r[2]]
    segs, k_mod = [], 0
    for j, (a, b, is_mod) in enumerate(rs):
        if is_mod:
            k_mod += 1
            label = "modulated" if n_mod == 1 else f"mod {k_mod}"
        elif not mod_js:
            label = "carrier only"
        elif j < mod_js[0]:
            label = "lead-in"
        elif j > mod_js[-1]:
            label = "tail"
        else:
            label = "gap"
        a, b = a + i_on, b + i_on
        segs.append({"a": a, "b": b, "mod": is_mod, "label": label,
                     "t0": times[a] - frame_s / 2, "t1": times[b] + frame_s / 2})

    # start transient: the carrier-on frames before the carrier settles to a steady, narrow spectrum
    transient_msg = "start transient: none separated"
    lead_seg = next((sg for sg in segs if not sg["mod"]), None)
    if lead_seg is not None:
        a, b = lead_seg["a"], lead_seg["b"] - 3     # last frames overlap the start of modulation: not used
        bw = running_median(frame_occupied_khz(running_mean_rows(P[a:b + 1], 3), N, df), 5)
        n_steady = min(len(bw), max(1, int(round(SETTLE_STEADY_S / frame_s))))
        steady = float(np.median(bw[-n_steady:]))
        tol = max(0.5 * df / 1e3, SETTLE_TOL_FRAC * steady)
        s_rel = settle_index(bw, steady, tol, max(1, int(round(SETTLE_HOLD_S / frame_s))))
        if args.debug_transient:
            for i in range(min(len(bw), 80)):
                print(f"dbg {times[a + i] + t_offset:7.3f} s  BW+6 {bw[i]:5.2f} kHz")
            print(f"dbg steady BW+6 {steady:.2f} kHz, settled from index {s_rel} of {len(bw)}")
        s_f = a + s_rel
        if s_rel * frame_s >= TRANSIENT_MIN_S and (b - s_f + 1) * frame_s >= SETTLE_MIN_S:
            t_onset, t_settle = lead_seg["t0"], times[s_f] - frame_s / 2
            k = next(i for i, sg in enumerate(segs) if sg is lead_seg)
            trans = {"a": a, "b": s_f - 1, "mod": False, "label": "transient",
                     "t0": t_onset, "t1": t_settle}
            steady_seg = dict(lead_seg, a=s_f, t0=t_settle)
            segs[k:k + 1] = [trans, steady_seg]
            transient_msg = (f"start transient: above noise at {t_onset + t_offset:.3f} s, settled at "
                             f"{t_settle + t_offset:.3f} s, duration {t_settle - t_onset:.3f} s "
                             f"(steady BW+6 {steady:.2f} kHz)")

    # instantaneous frequency of the carrier: the modulating signal, and the carrier offset
    fi = np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float64) * (rate / (2 * np.pi))
    edge = int(round(EDGE_SKIP_S * rate))

    def sample_range(sg):
        s0 = max(0, int(round(sg["t0"] * rate)) + edge)
        s1 = min(len(fi), int(round(sg["t1"] * rate)) - edge)
        return s0, s1

    # carrier offset: mean instantaneous frequency over the steady unmodulated part (not the start transient)
    ref = next((sg for sg in segs if not sg["mod"] and sg["label"] != "transient"), segs[0])
    s0, s1 = sample_range(ref)
    if s1 <= s0:
        s0, s1 = sample_range({"t0": segs[0]["t0"], "t1": segs[-1]["t1"]})
    fc_off = float(np.mean(fi[s0:s1]))

    # matched-filter search for the 1023-chip sequence in the demodulated audio
    dec = int(round(rate / AUDIO_FS))
    audio = (fi - fc_off)[:(len(fi) // dec) * dec].reshape(-1, dec).mean(axis=1)
    ref = sequence_reference()
    ncc, energy, er = matched_filter(audio, ref)
    # search only lags whose whole window lies inside the carrier-on interval: outside it the audio is noise
    t_on_a = times[i_on] - frame_s / 2
    t_off_a = times[i_off] + frame_s / 2
    j_lo = max(0, int(np.ceil(t_on_a * AUDIO_FS - 0.5)))
    j_hi = min(len(ncc) - 1, int(np.floor(t_off_a * AUDIO_FS - 0.5 - len(ref))))
    sub = ncc[j_lo:j_hi + 1]
    seq = None
    det_snr = 0.0
    psl_db = 0.0
    if sub.size:
        k = j_lo + int(np.argmax(np.abs(sub)))
        pk = float(ncc[k])
        sigma = float(np.median(np.abs(sub)) / 0.6745)
        det_snr = abs(pk) / sigma if sigma > 0 else 0.0
        ex = SEQ_PSL_EXCL_CHIPS * SEQ_SPC
        lobes = np.abs(ncc)
        side_max = float(np.max(np.concatenate([lobes[:max(0, k - ex)], lobes[k + ex + 1:]])))
        psl_db = 20 * np.log10(abs(pk) / side_max) if side_max > 0 else 0.0
    if det_snr >= SEQ_DET_SIGMA and psl_db >= SEQ_MIN_PSL_DB:
        L = len(ref)
        win = audio[k:k + L]
        gain = pk * np.sqrt(energy[k] / er)            # least-squares scale of the reference in the audio
        fit = gain * ref
        resid = win - fit
        t_s = (k + 0.5) / AUDIO_FS
        t_e = t_s + L / AUDIO_FS
        f0 = max(0, int(np.searchsorted(times, t_s)))
        f1 = min(nf - 1, int(np.searchsorted(times, t_e)) - 1)
        seq = {"t_s": t_s, "t_e": t_e, "pk": pk, "psl_db": psl_db,
               "dev_rms": abs(gain) * float(np.sqrt(np.mean(ref ** 2))),
               "res_rms": float(np.sqrt(np.mean(resid ** 2))),
               "audio_snr": 10 * np.log10(np.sum(fit ** 2) / max(np.sum(resid ** 2), 1e-30))}
        new = []
        for sg in segs:
            if not sg["mod"] or sg["b"] < f0 or sg["a"] > f1:
                new.append(sg)
                continue
            parts = []
            if sg["a"] < f0 and (f0 - sg["a"]) * frame_s >= MIN_SEG_S:
                parts.append((sg["a"], f0 - 1, False))
            parts.append((max(sg["a"], f0), min(sg["b"], f1), True))
            if sg["b"] > f1 and (sg["b"] - f1) * frame_s >= MIN_SEG_S:
                parts.append((f1 + 1, sg["b"], False))
            for a, b, is_seq in parts:
                new.append({"a": a, "b": b, "mod": True, "seq": is_seq,
                            "label": "sequence" if is_seq else sg["label"],
                            "t0": times[a] - frame_s / 2, "t1": times[b] + frame_s / 2})
        segs = new

    # per-segment statistics
    E_all = float((c_pow + s_pow)[i_on:i_off + 1].sum())
    chan = np.abs(np.arange(len(freqs)) - k0) * df <= CHANNEL_HZ / 2
    N_chan = float(N[chan].mean() * CHANNEL_HZ / df)       # noise power in the 2.5 kHz reference bandwidth
    N_bin_db = 10 * np.log10(N.mean() + 1e-30)
    tones = {}
    datas = {}
    for sg in segs:
        a, b = sg["a"], sg["b"]
        sl = slice(a, b + 1)
        mean_p = P[sl].mean(axis=0)
        cp, sp = c_pow[sl], s_pow[sl]
        Ec, Es = float(cp.sum()), float(sp.sum())
        sg["dur"] = sg["t1"] - sg["t0"]
        cp_m, sp_m = float(cp.mean()), float(sp.mean())
        sg["carrier_db"] = 10 * np.log10(cp_m) if cp_m > 0 else np.nan
        sg["side_db"] = 10 * np.log10(sp_m) if sp_m > 0 else np.nan     # nan: below the noise, shown as --
        sg["side_pct"] = 100.0 * Es / (Ec + Es) if (Ec + Es) != 0 else np.nan
        sg["energy_pct"] = 100.0 * (Ec + Es) / E_all if E_all > 0 else np.nan
        sg["bw99"], sg["bw26"], sg["bwm"] = bandwidths(mean_p, N, df)
        S_ch = float((mean_p - N)[chan].sum())
        sg["snr"] = 10 * np.log10(S_ch / N_chan) if S_ch > 0 else np.nan
        if sg["mod"] and not sg.get("seq"):
            s0, s1 = sample_range(sg)
            j0 = max(0, int(np.ceil((sg["t0"] + EDGE_SKIP_S) * AUDIO_FS)))
            j1 = min(len(audio), int((sg["t1"] - EDGE_SKIP_S) * AUDIO_FS))
            if s1 - s0 > rate * 0.05 and j1 - j0 > AUDIO_FS * 0.05:
                xa = audio[j0:j1]
                fm, share = tone_test(xa, AUDIO_FS)
                if share >= TONE_MIN_SHARE:
                    x = fi[s0:s1] - np.mean(fi[s0:s1])
                    tones[id(sg)] = (fm, deviation(x, fm, rate), share)
                else:
                    sg["label"] = "data"
                    datas[id(sg)] = (share, float(np.sqrt(np.mean(xa ** 2))))

    # console report
    carrier_hz = tune_hz + fc_off
    t_on = times[i_on] - frame_s / 2 + t_offset
    t_off = times[i_off] + frame_s / 2 + t_offset
    print(f"file      {os.path.basename(args.wav)}")
    print(f"rate      {rate} S/s   analysed {len(iq) / rate:.3f} s from {t_offset:.3f} s   "
          f"FFT {nfft} ({df:.2f} Hz bins), frame {frame_s * 1e3:.1f} ms")
    print(f"tuned     {tune_hz / 1e6:.6f} MHz   carrier {carrier_hz / 1e6:.6f} MHz "
          f"({fc_off:+.1f} Hz from tuning)")
    print(f"carrier   on {t_on:.3f} s  to {t_off:.3f} s   duration {t_off - t_on:.3f} s")
    print(f"noise     {10 * np.log10(N_chan + 1e-30):.1f} dB in 2.5 kHz, {N_bin_db:.1f} dB per bin "
          f"(relative; levels are uncalibrated, ratios are valid)")
    print(transient_msg)
    print()
    hdr = (f"{'seg':>3}  {'label':<12} {'start':>7} {'end':>7} {'dur':>6}  {'carrier':>8} {'sideband':>9} "
           f"{'side%':>6} {'energy%':>8} {'BW99':>7} {'BW-26':>7} {'BW+6':>7} {'SNR2.5':>7}")
    units = (f"{'':>3}  {'':<12} {'(s)':>7} {'(s)':>7} {'(s)':>6}  {'(dB)':>8} {'(dB)':>9} "
             f"{'(%)':>6} {'(%)':>8} {'(kHz)':>7} {'(kHz)':>7} {'(kHz)':>7} {'(dB)':>7}")
    print(hdr)
    print(units)
    print("-" * len(hdr))
    for i, sg in enumerate(segs, 1):
        print(f"{i:>3}  {sg['label']:<12} {sg['t0'] + t_offset:>7.3f} {sg['t1'] + t_offset:>7.3f} "
              f"{sg['dur']:>6.3f}  {fmt(sg['carrier_db'], 8, 1)} {fmt(sg['side_db'], 9, 1)} "
              f"{fmt(sg['side_pct'], 6, 1)} {fmt(sg['energy_pct'], 8, 1)} {fmt(sg['bw99'], 7, 2)} "
              f"{fmt(sg['bw26'], 7, 2)} {fmt(sg['bwm'], 7, 2)} {fmt(sg['snr'], 7, 1)}")
    if tones or datas:
        print()
        print(f"{'seg':>3}  {'label':<12} {'tone (Hz)':>10} {'share (%)':>10} {'deviation (kHz)':>16} {'index':>6}")
        print("-" * 64)
        for i, sg in enumerate(segs, 1):
            if id(sg) in tones:
                fm, pk, share = tones[id(sg)]
                print(f"{i:>3}  {sg['label']:<12} {fm:>10.2f} {100 * share:>10.1f} {pk / 1e3:>16.3f} {pk / fm:>6.2f}")
            elif id(sg) in datas:
                share, rms = datas[id(sg)]
                print(f"{i:>3}  {sg['label']:<12} {'--':>10} {100 * share:>10.1f} {rms / 1e3:>16.3f}   rms, no tone")
        print("share = part of the segment's audio power in its strongest line (300-3000 Hz); a tone needs "
              f"{100 * TONE_MIN_SHARE:.0f}% or more")
    print()
    print()
    if seq is None:
        print(f"sequence  not detected (peak {det_snr:.1f} sigma, need {SEQ_DET_SIGMA:.0f}; "
              f"peak-to-sidelobe {psl_db:.1f} dB, need {SEQ_MIN_PSL_DB:.0f})")
    else:
        print(f"sequence  1023 chips at {SEQ_CHIP_RATE:.0f} cps, matched filter: start {seq['t_s'] + t_offset:.3f} s, "
              f"end {seq['t_e'] + t_offset:.3f} s, duration {seq['t_e'] - seq['t_s']:.3f} s")
        print(f"          correlation {seq['pk']:+.3f} ({'normal' if seq['pk'] > 0 else 'inverted'} polarity), "
              f"output SNR {20 * np.log10(det_snr):.1f} dB, peak-to-sidelobe {seq['psl_db']:.1f} dB")
        print(f"          fitted deviation rms {seq['dev_rms'] / 1e3:.3f} kHz, residual rms {seq['res_rms'] / 1e3:.3f} kHz, "
              f"audio SNR {seq['audio_snr']:.1f} dB")
        print("          residual = received minus scaled reference: FM noise, mic-path filtering and distortion")
    print()
    print("bandwidths: BW99 = width holding 99% of the segment's noise-subtracted power; BW-26 = outermost bins within")
    print(f"26 dB of the peak; BW+6 = outermost bins {MARGIN_DB:.0f} dB above the noise per bin. Time-averaged over each segment.")
    print("sideband = power outside the carrier bins (+/-2 bins); side% and energy% are shares of the carrier-on total.")
    print(f"SNR2.5 = segment mean power within +/-1.25 kHz of the carrier over the noise in a 2.5 kHz bandwidth.")
    print("deviation = peak frequency deviation of the carrier, from the instantaneous frequency at the tone;")
    print("index = deviation / tone (modulation index).")
    for n in notes:
        print("note:", n, file=sys.stderr)

    if args.no_plot:
        return

    # waterfall with boundary markers
    S_db = 10.0 * np.log10(P + 1e-20)
    lo = args.vmin if args.vmin is not None else float(np.percentile(S_db, 2))
    hi = args.vmax if args.vmax is not None else float(np.percentile(S_db, 99.9))
    if hi <= lo:
        hi = lo + 1.0
    mean_db = 10.0 * np.log10(P.mean(axis=0) + 1e-20)
    f_rf_mhz = (tune_hz + freqs) / 1e6
    f0, f1 = f_rf_mhz[0], f_rf_mhz[-1]

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(4, 1, hspace=0.08)
    ax_s = fig.add_subplot(gs[0])
    ax_w = fig.add_subplot(gs[1:], sharex=ax_s)
    ax_s.plot(f_rf_mhz, mean_db, lw=0.8, color="tab:blue")
    ax_s.set_ylabel("Mean (dB)")
    ax_s.grid(alpha=0.3)
    ax_s.tick_params(labelbottom=False)
    title = os.path.basename(args.wav)
    if side.get("start_utc"):
        title += f"   start {side['start_utc']}"
    ax_s.set_title(title, fontsize=10)

    times_plot = times + t_offset
    t_lo = times_plot[0] - frame_s / 2
    t_hi = times_plot[-1] + frame_s / 2
    im = ax_w.imshow(S_db, origin="lower", aspect="auto", cmap="viridis", vmin=lo, vmax=hi,
                     extent=[f0, f1, t_lo, t_hi], interpolation="nearest")
    for sg in segs:
        y0, y1 = sg["t0"] + t_offset, sg["t1"] + t_offset
        if sg["mod"]:
            ax_w.axhspan(y0, y1, color="orange" if sg.get("seq") else "red", alpha=0.10, lw=0)
        for y in (y0, y1):
            ax_w.axhline(y, color="white", ls="--", lw=0.8, alpha=0.9)
        ax_w.text(f0 + 0.01 * (f1 - f0), (y0 + y1) / 2,
                  f"{sg['label']}  {y0:.2f}-{y1:.2f} s", color="white", fontsize=8, va="center",
                  bbox=dict(facecolor="black", alpha=0.5, lw=0))
    ax_w.set_xlabel("RF frequency (MHz)")
    ax_w.set_ylabel("Time from start of file (s)")
    ax_w.xaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    cb = fig.colorbar(im, ax=[ax_s, ax_w], pad=0.02)
    cb.set_label("Power (dB, relative)")

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print("saved", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
