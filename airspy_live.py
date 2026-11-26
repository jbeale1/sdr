#!/usr/bin/env python3

"""Live spectrum + waterfall for the Airspy HF+ with live FT8 decoding via WSJT-X's jt9.

Requires: numpy, scipy, PySide6, pyqtgraph, libairspyhf, and jt9 on PATH.
Decoder WAV files are written to /dev/shm and deleted after each decode.
Decodes are appended to ALL.TXT in the current directory and shown in a pane at the
bottom of the window, or in a separate window after "Pop out". 

A WWV timing mode logs the clock offset of the receiver and the carrier frequency,
and reads the WWV 100 Hz timecode to get date/time (if reception quality allows).
There is also a morse decode mode (not above 22 wpm though).

FM mode decodes Morse sent as a 400 Hz FM tone keyed on a carrier, from the sidebands of the
carrier. Output goes to the decodes pane and to FM.TXT in the current directory.

Track mode follows every signal on screen and logs each transmission: start, length,
gap since the same frequency last went quiet (turnaround), level and width. Output goes
to the decodes pane and to track_YYYYmmdd_HHMMSS.csv in the current directory.

The Settings menu holds raw/SNR display, time average and the FFT size. Changing the FFT
size takes effect at once; the waterfall and background average start again.

-J.Beale N8JUF  25-Nov-2026

Examples:
    ./airspy_live.py                              # 14.074 MHz, 14074-14077 kHz, 192 kS/s
    ./airspy_live.py -f 14.074 --span 3 --rate 192000
    ./airspy_live.py --no-decode                  # display only
"""
import argparse
import bisect
import csv
import ctypes
import ctypes.util
import io
import itertools
import json
import os
import math
import re
import shutil
import struct
import subprocess
import tempfile
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
import scipy.fft as sfft
import scipy.signal as ssig
import scipy.special as sspec
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

VERSION = "0.43.0"
DEFAULT_FREQ_MHZ = 14.074
DEFAULT_LO_KHZ = 14074.0        # default display window when no display option is given
DEFAULT_HI_KHZ = 14077.0
DEFAULT_RATE = 192_000
DEFAULT_NFFT = 131072
NFFT_CHOICES = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]   # Settings > FFT size
MAX_DISPLAY_COLS = 2048
RING_SECONDS = 17               # ring must hold a full 15 s FT8 slot plus margin
HEADROOM_DB = 40.0              # colour range above the calibrated top
SETTLE_S = 0.25                 # discard samples after a retune
BG_TAU_S = 60.0                 # background tracker time constant
BG_GATE_DB = 6.0                # bins this far above the background are treated as signal
BG_GATE_MAX_S = 120.0           # a bin held above the gate this long is taken as new background
SPEC_AVG_S = 15.0                # time constant of the averaged spectrum trace
LEVELS_MIN_ROWS = 10            # waterfall rows needed before colour levels are fixed
DECODE_PANE_HEIGHT = 170        # initial height of the docked decodes pane, px
DECODE_HISTORY_LINES = 10000    # scrollback kept in the decodes pane
RATE_WINDOW_S = 15 * 60         # decode-rate window: most recent 15 minutes
RATE_MIN_SPAN_S = 60.0          # below this much elapsed time the rate is shown as "--"
MY_GRID = "CN85"                # own Maidenhead grid square, for distances in the hover popup

SPEC_SNAP_PX = 10               # hover readout snaps to the highest blue-trace point this close
COUNT_STRIP_W = 34              # width of the decode-count strip beside the waterfall, px
COUNT_AXIS_H = 50               # height of the top and bottom axes of the waterfall column, px

SPAN_PRESETS = [
    ("1 kHz", 1.0),
    ("3 kHz", 3.0),
    ("6 kHz", 6.0),
    ("12 kHz", 12.0),
    ("20 kHz", 20.0),
    ("100 kHz", 100.0),
    ("350 kHz", 350.0),
]

BANDS = [
    # (name, standard FT8 dial MHz, standard FT4 dial MHz or None); 6 m is left out
    # because it is outside the tuning range check
    ("160 m", 1.840, None), ("80 m", 3.573, 3.575), ("60 m", 5.357, None),
    ("40 m", 7.074, 7.0475), ("30 m", 10.136, 10.140), ("20 m", 14.074, 14.080),
    ("17 m", 18.100, 18.104), ("15 m", 21.074, 21.140), ("12 m", 24.915, 24.919),
    ("10 m", 28.074, 28.180), ("2 m", 144.174, 144.170),
]
BAND_SPAN_KHZ = 3.0             # display width when a band is selected, as at startup
BAND_SPAN_OVERRIDE_KHZ = {"2 m": 100.0}   # bands that need a wider display when selected

# Decoding: one mode at a time, via WSJT-X's jt9
FT8_AUDIO_RATE = 12_000
FT8_AUDIO_CENTER = 2100.0                       # centre of the 200-4000 Hz decode band
FT8_AUDIO_MAX_SPAN = 3800.0                     # widest displayed range that can be decoded
MODES = {
    # slot_s: transmit cycle; tone_bw: occupied bandwidth of one signal; jt9: decoder arguments;
    # dial_mhz: normal calling frequency; lo/hi: display window (kHz) selected with the mode
    "FT8": {"slot_s": 15.0, "tone_bw": 50.0, "jt9": ["-8", "-d", "3", "-f", "2100"],
            "dial_mhz": 14.074, "lo": 14074.0, "hi": 14077.0},
    "FT4": {"slot_s": 7.5, "tone_bw": 90.0, "jt9": ["-5", "-d", "3", "-f", "2100", "-p", "7.5"],
            "dial_mhz": 14.080, "lo": 14080.0, "hi": 14083.0},
}
DECODE_RE = re.compile(r"^\s*(?:\d{6}\s+)?(-?\d+)\s+(-?\d+\.\d+)\s+(\d+(?:\.\d+)?)\s+~\s+(.*\S)\s*$")

ANN_RE = re.compile(r"^(.*?)((?:\s+\?)?(?:\s+a\d+)?)\s*$")   # message, then optional jt9 tag

MARK_RGBA_CQ = (183, 65, 14, 255)               # rust red: not a viridis colour
MARK_RGBA_OTHER = (130, 130, 130, 255)
SLOT_LINE_RGBA = (90, 200, 255, 255)           # slot boundary lines, drawn with Mark on

# ---------------------------------------------------------------- libairspyhf

class Transfer(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_void_p),
        ("ctx", ctypes.c_void_p),
        ("samples", ctypes.POINTER(ctypes.c_float)),  # interleaved re, im
        ("sample_count", ctypes.c_int),               # complex samples
        ("dropped_samples", ctypes.c_uint64),
    ]

SAMPLE_CB_T = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(Transfer))


def load_lib():
    if sys.platform == "win32":
        # Python 3.8+ does not search the script folder for a DLL's own dependencies, so add it
        here = os.path.dirname(os.path.abspath(__file__))
        os.add_dll_directory(here)
        name = os.path.join(here, "airspyhf.dll")
    else:
        name = ctypes.util.find_library("airspyhf") or "libairspyhf.so.0"
    lib = ctypes.CDLL(name)
    dev_t = ctypes.c_void_p
    sigs = {
        "airspyhf_open": ([ctypes.POINTER(dev_t)], ctypes.c_int),
        "airspyhf_close": ([dev_t], ctypes.c_int),
        "airspyhf_start": ([dev_t, SAMPLE_CB_T, ctypes.c_void_p], ctypes.c_int),
        "airspyhf_stop": ([dev_t], ctypes.c_int),
        "airspyhf_is_low_if": ([dev_t], ctypes.c_int),
        "airspyhf_get_samplerates": ([dev_t, ctypes.POINTER(ctypes.c_uint32),
                                      ctypes.c_uint32], ctypes.c_int),
        "airspyhf_set_freq": ([dev_t, ctypes.c_uint32], ctypes.c_int),
        "airspyhf_set_samplerate": ([dev_t, ctypes.c_uint32], ctypes.c_int),
        "airspyhf_set_hf_agc": ([dev_t, ctypes.c_uint8], ctypes.c_int),
        "airspyhf_set_hf_agc_threshold": ([dev_t, ctypes.c_uint8], ctypes.c_int),
        "airspyhf_set_hf_att": ([dev_t, ctypes.c_uint8], ctypes.c_int),
        "airspyhf_set_hf_lna": ([dev_t, ctypes.c_uint8], ctypes.c_int),
    }
    for fname, (argtypes, restype) in sigs.items():
        fn = getattr(lib, fname)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


def get_sample_rates(lib, dev):
    """Query the device's supported sample rates (two-call pattern, as in airspyhf_rx.c)."""
    n = ctypes.c_uint32(0)
    if lib.airspyhf_get_samplerates(dev, ctypes.byref(n), 0) != 0 or n.value == 0:
        sys.exit("airspyhf_get_samplerates() failed")
    buf = (ctypes.c_uint32 * n.value)()
    if lib.airspyhf_get_samplerates(dev, buf, n.value) != 0:
        sys.exit("airspyhf_get_samplerates() failed")
    return [int(v) for v in buf]


class Radio:
    """Thin wrapper so the GUI can retune the device."""

    def __init__(self, lib, dev):
        self.lib, self.dev = lib, dev

    def stop(self):
        self.lib.airspyhf_stop(self.dev)

    def start(self):
        return self.lib.airspyhf_start(self.dev, self.cb, None) == 0

    def set_freq(self, hz):
        return self.lib.airspyhf_set_freq(self.dev, int(hz)) == 0

# ------------------------------------------------------------ ring buffer

class RingBuffer:
    """Single-producer (USB callback) / single-consumer (GUI) complex buffer."""

    def __init__(self, size):
        self.size = size
        self.buf = np.zeros(size, dtype=np.complex64)
        self.total = 0            # absolute number of samples written
        self.dropped = 0          # from the driver
        self.pushes = deque(maxlen=50000)   # (total after the push, time.time() at the push)
        self.lock = threading.Lock()

    def push(self, x):
        n = len(x)
        if n == 0:
            return
        with self.lock:
            if n > self.size:
                x = x[-self.size:]
                n = self.size
            w = self.total % self.size
            first = min(n, self.size - w)
            self.buf[w:w + first] = x[:first]
            if n > first:
                self.buf[:n - first] = x[first:]
            self.total += n
            self.pushes.append((self.total, time.time()))

    def read(self, start, count):
        """Copy absolute range [start, start+count). Returns None if overwritten."""
        with self.lock:
            if self.total - start > self.size:
                return None
            s = start % self.size
            first = min(count, self.size - s)
            out = np.empty(count, dtype=np.complex64)
            out[:first] = self.buf[s:s + first]
            if count > first:
                out[first:] = self.buf[:count - first]
            return out

    def wall_at(self, sample, rate):
        """Wall-clock time (time.time()) at which absolute sample index `sample` reached the ring. The
        last sample of each USB block arrived at that block's push time, and earlier samples are spaced at
        the sample rate. None if the sample is newer than every push record."""
        with self.lock:
            recs = list(self.pushes)
        i = bisect.bisect_left([r[0] for r in recs], sample)
        if i >= len(recs):
            return None
        total_after, t_push = recs[i]
        return t_push - (total_after - sample) / rate

# ------------------------------------------------------------ USB callback

class Receiver:
    def __init__(self, ring):
        self.ring = ring
        # Keep a reference to the ctypes callback object for the whole run.
        self.cb = SAMPLE_CB_T(self._on_samples)

    def _on_samples(self, tp):
        try:
            t = tp.contents
            n = t.sample_count
            if n > 0:
                floats = np.ctypeslib.as_array(t.samples, shape=(2 * n,))
                self.ring.push(floats.view(np.complex64))   # copies the data
            self.ring.dropped = t.dropped_samples
        except Exception:
            return -1       # stop streaming on error
        return 0

# ------------------------------------------------------------ DSP

# Tx Test: a transmission is a block of spectrum this far above the background reference
TX_THRESHOLD_DB = 6.0           # on the TX_STAT_KHZ-average excess above band median, central 50%
TX_MAX_BW_KHZ = 25.0            # wider than this above threshold is not a voice transmission
TX_STAT_KHZ = 2.0               # detection statistic: power averaged over this width
TX_MIN_S = 1.5                  # continuous time above threshold needed to confirm a transmission
TX_GAP_S = 0.3                  # a dropout shorter than this does not end a transmission
TX_SKIP_S = 0.5                 # left out of the peak-hold trace at the start and at the end
TX_MAX_S = 60.0                 # longer than this is taken as a constant carrier, not a transmission
TX_CHANNEL_HALF_KHZ = 7.5       # the transmission's channel: within this of its strongest column
TX_REF_TAU_S = 60.0             # background reference follows slow changes while nothing is detected
TX_WARMUP_S = 5.0               # initial background reference = average over this long
TX_SMOOTH_KHZ = 1.0             # bandwidth is measured on the average spectrum, smoothed over this
TX_BW_DB = 6.0                  # occupied bandwidth: contiguous columns within this of the peak
TX_DELAY_S = TX_GAP_S + TX_SKIP_S   # rows are held this long, so the last TX_SKIP_S can be dropped
NOISE_REPORT_S = 600            # noise stats are logged on UTC multiples of this

# WWV timing mode
WWV_FREQS_MHZ = [("2.5 MHz", 2.5), ("5 MHz", 5.0), ("10 MHz", 10.0), ("15 MHz", 15.0),
                 ("20 MHz", 20.0)]           # menu entries; the manual frequency entry still works
WWV_DEFAULT_MHZ = 10.0
WWV_SPAN_KHZ = 3.0                           # display width centred on the carrier
WWV_TONES_HZ = (1000.0, 1200.0, 1500.0)      # WWV seconds and minute, WWVH, WWV hour marker
WWV_DEC_RATE = 8000                          # envelope rate after decimation, S/s
WWV_LP_HZ = 3500.0                           # envelope lowpass before decimation
WWV_SEG_S = 3.0                              # analysis window, re-run every second
WWV_EDGE_S = 0.75                            # a second is decided only when all its peaks lie this far
                                             # inside the window, so a tick's sidelobes are never split across passes
WWV_MF_DETECT_DB = 14.0                      # matched-filter peak: this far above the median output
WWV_TICK_LEN_S = 0.005                       # tick template length
WWV_MARKER_LEN_S = 0.8                       # marker template length
WWV_TICK_MIN_FRAC = 0.3                      # tick amplitude, relative to the strongest tick in the window
WWV_NO_TICK_SEC = (29, 59)                  # seconds with no tick: any tick detected there is noise
WWV_DECIDE_MARGIN_S = 1.0                   # a second is decided once the trusted window passes it by this much
WWV_MARKER_COVER = 0.9                       # share of the 800 ms marker that must carry the 1 kHz tone
WWV_QUIET_COVER = 0.3                       # share of the quiet span before a marker allowed to carry the tone
WWV_QUIET_SPAN_S = (0.6, 0.1)               # quiet span before a marker onset (WWV sends no tick at second 59)
WWV_MISS_SAVES_PER_HOUR = 6                 # cap on automatic envelope WAVs for minutes with no accepted marker
WWV_MISS_SPAN_S = (3.0, 2.0)                # saved span around a missed minute: from 3 s before to 2 s after
WWV_SHORT_S = 0.02                          # short tone window for the coverage test
WWV_CARRIER_S = 10.0                         # carrier frequency and strength measured over this span
WWV_CAPTURE_S = 10.0               # length of the raw IQ and envelope capture
WWV_CARRIER_SEARCH_HZ = 3000.0
WWV_SETTLE_S = 4.0                           # a minute is reported this long after it ends
WWV_MAP_LAG_S = 1.0                          # ring samples map to wall time one second later: markers then fall on second 0
WWV_HIST_S = 130.0                           # envelope history kept for the per-minute 100 Hz analysis
WWV_FRAME_BAND_HZ = (90.0, 110.0)           # band of the 100 Hz time-code subcarrier in the envelope
WWV_FRAME_EDGE_MS = 30.0                    # time-code pulses start 30 ms into the second (NIST SP 250-67)

# WWV 100 Hz BCD time code. Layout follows NTP's WWV/H encoder (util/tg.c): one pulse per second,
# a 1-minute frame, and no subcarrier in second 0. Each pulse starts WWV_FRAME_EDGE_MS into the
# second and lasts 200 ms (binary 0), 500 ms (binary 1) or 800 ms (position identifier).
TC_P_SECONDS = (9, 19, 29, 39, 49, 59)      # position identifiers
TC_FIELDS = {                               # (second, weight), least significant bit first
    "year_units": ((4, 1), (5, 2), (6, 4), (7, 8)),
    "minute_units": ((10, 1), (11, 2), (12, 4), (13, 8)),
    "minute_tens": ((15, 10), (16, 20), (17, 40)),
    "hour_units": ((20, 1), (21, 2), (22, 4), (23, 8)),
    "hour_tens": ((25, 10), (26, 20)),
    "day_units": ((30, 1), (31, 2), (32, 4), (33, 8)),
    "day_tens": ((35, 10), (36, 20), (37, 40), (38, 80)),
    "day_hundreds": ((40, 100), (41, 200)),
    "year_tens": ((51, 10), (52, 20), (53, 40), (54, 80)),
}
TC_MAX_INT_S = 2            # larger whole-second disagreement with the PC clock: frame rejected
TC_NB = 100                 # 10 ms bins across the first second after each pulse-edge reference
TC_TAIL_S = 0.85             # bins from here on carry no pulse: used for the noise level
TC_BIN_S = 0.01
TC_ON_S = {"0": 0.2, "1": 0.5, "P": 0.8}   # subcarrier on-time of each pulse, from the edge
TC_SEARCH_MIN = 5          # candidate offset of the PC clock from true UTC, +/- whole minutes
TC_HIST_MIN = 15           # minutes of time-code frames kept for the joint decode
TC_MIN_MARGIN = 5.0       # log-likelihood margin (nats) the best offset needs over the next best


def tc_window(hist, t_last, t0, t1, fs):
    """Band-limited (90-110 Hz) envelope of the envelope history between wall times t0 and t1.
    Returns (t_abs, env): wall time and envelope of each sample, or (None, None)."""
    n = len(hist)
    j0 = n - 1 - int(round((t_last - t0) * fs))
    j1 = n - 1 - int(round((t_last - t1) * fs))
    if n < 10 or j0 < 0 or j1 > n - 1 or j1 <= j0:
        return None, None
    seg = hist[j0:j1 + 1]
    t_abs = t_last - (n - 1 - np.arange(j0, j1 + 1)) / fs
    sos = ssig.butter(4, list(WWV_FRAME_BAND_HZ), btype="bandpass", fs=fs, output="sos")
    bp = ssig.sosfiltfilt(sos, seg - float(np.mean(seg)))
    return t_abs, np.abs(ssig.hilbert(bp))


def tc_profiles(t_abs, env, delta, k0, count):
    """Mean envelope in TC_NB bins of TC_BIN_S across the first second after each whole PC-clock
    second k0..k0+count-1 (second boundaries at k + delta). Returns (count, TC_NB), or None if the
    window does not cover every second."""
    base = np.arange(k0, k0 + count, dtype=float) + delta
    if base[0] < t_abs[0] or base[-1] + 1.0 > t_abs[-1]:
        return None
    cs = np.concatenate([[0.0], np.cumsum(env)])
    edges = base[:, None] + np.arange(TC_NB + 1)[None, :] * TC_BIN_S
    idx = np.searchsorted(t_abs, edges)
    n = np.maximum(idx[:, 1:] - idx[:, :-1], 1)
    return (cs[idx[:, 1:]] - cs[idx[:, :-1]]) / n


def tc_template(on_s):
    """Expected subcarrier shape across the bins for one pulse: 1 while the subcarrier is on."""
    centre = (np.arange(TC_NB) + 0.5) * TC_BIN_S
    edge = WWV_FRAME_EDGE_MS / 1000.0
    return ((centre >= edge) & (centre < edge + on_s)).astype(float)


TC_TEMPLATES = {c: tc_template(s) for c, s in TC_ON_S.items()}


def tc_amp(prof, tmpl):
    """Least-squares amplitude of a template in each profile row (level offset free)."""
    g = tmpl - tmpl.mean()
    p = prof - prof.mean(axis=1, keepdims=True)
    return (p @ g) / float(g @ g)


def tc_sse(prof, tmpl, amp):
    """Residual sum of squares of each row for a template at a fixed amplitude (level offset free)."""
    r = prof - amp * tmpl[None, :]
    r = r - r.mean(axis=1, keepdims=True)
    return (r * r).sum(axis=1)


def tc_encode(epoch):
    """The 60-character frame for the UTC minute starting at epoch: '-' second 0, 'P' position
    identifiers, '0'/'1' elsewhere."""
    t = datetime.fromtimestamp(epoch, timezone.utc)
    yd = t.timetuple().tm_yday
    word = ["-"] + ["0"] * 59
    for p in TC_P_SECONDS:
        word[p] = "P"
    vals = {"minute_units": t.minute % 10, "minute_tens": (t.minute // 10) * 10,
            "hour_units": t.hour % 10, "hour_tens": (t.hour // 10) * 10,
            "day_units": yd % 10, "day_tens": ((yd // 10) % 10) * 10,
            "day_hundreds": (yd // 100) * 100,
            "year_units": t.year % 10, "year_tens": ((t.year // 10) % 10) * 10}
    for name, v in vals.items():
        for pos, w in sorted(TC_FIELDS[name], key=lambda x: -x[1]):
            if v >= w:
                word[pos] = "1"
                v -= w
    return "".join(word)


def tc_costs(rows, amp):
    """Cost of each symbol for each second's profile, columns '-', '0', '1', 'P' (scaled by
    2 x noise variance). A pulse-type symbol costs no more than leaving the second with no pulse,
    so a dropped pulse does not count against any candidate frame."""
    flat = tc_sse(rows, np.zeros(TC_NB), amp)
    cols = [flat] + [np.minimum(tc_sse(rows, TC_TEMPLATES[c], amp), flat) for c in "01P"]
    return np.stack(cols, axis=1)


def tc_solve(entries, search_min):
    """Joint time-code decode over several minutes. entries: (P, k0, delta, prof) per frame, where P
    is the PC-clock minute index of the frame's window, k0 its first PC second and prof its profiles.
    The true minute of entry P is P + d for one integer d, and the frame start within each window
    is one shared alignment s, because the PC clock's whole-second error is the same in every
    minute. Each (d, s) is scored by the summed likelihood of all entries. Returns the result dict
    for the newest entry."""
    SYM = {"-": 0, "0": 1, "1": 2, "P": 3}
    tail = int(round(TC_TAIL_S / TC_BIN_S))
    opts = []
    for P, k0, delta, prof in entries:
        # one noise variance per window, from bins after the last possible pulse, so that
        # alignments and candidate words are compared on the same scale
        sig2 = float(np.median(prof[:, tail:].var(axis=1)))
        per = {}
        if sig2 > 0:
            for s in range(prof.shape[0] - 59):
                rows = prof[s:s + 60]
                # keep every alignment: a weak or negative P median in one window must not
                # veto that alignment for all minutes; a floored amplitude makes it erased
                amp = max(float(np.median(tc_amp(rows[list(TC_P_SECONDS)], TC_TEMPLATES["P"]))), 1e-9)
                per[s] = (tc_costs(rows, amp), 2.0 * sig2)
        opts.append(per)
    if not all(opts):
        return {"time": None, "ok": False, "clock_ms": None, "bits": "", "conf": "",
                "reason": "no usable frame in the joint window"}
    common = set.intersection(*(set(o) for o in opts))
    if not common:
        return {"time": None, "ok": False, "clock_ms": None, "bits": "", "conf": "",
                "reason": "no alignment common to all minutes"}

    totals = {}
    for d in range(-search_min, search_min + 1):
        words = [np.array([SYM[c] for c in tc_encode((P + d) * 60)])
                 for P, _, _, _ in entries]
        for s in common:
            total = 0.0
            for idx, per in zip(words, opts):
                cost, two_nv = per[s]
                total += -float(cost[np.arange(60), idx].sum()) / two_nv
            totals[(d, s)] = total
    d_best, s_best = max(totals, key=totals.get)
    margin = totals[(d_best, s_best)] - max(v for (d, _), v in totals.items() if d != d_best)

    P, k0, delta, prof = entries[-1]
    E_last = (P + d_best) * 60
    word = tc_encode(E_last)
    idx = np.array([SYM[c] for c in word])
    cost, two_nv = opts[-1][s_best]
    conf = []
    for j in range(60):
        e = idx[j]
        other = min(cost[j, k] for k in range(4) if k != e)
        conf.append(str(max(0, min(9, int((other - cost[j, e]) / two_nv)))))
    whole = (k0 + s_best) - E_last
    ok = abs(whole) <= TC_MAX_INT_S and margin >= TC_MIN_MARGIN
    if ok:
        reason = f"ok, {len(entries)} minutes, offset {d_best:+d} min, margin {margin:.1f} nats"
    elif margin < TC_MIN_MARGIN:
        reason = f"not yet unique, margin {margin:.1f} nats over {len(entries)} minutes"
    else:
        reason = f"whole second differs from PC clock by {whole} s"
    start_t = datetime.fromtimestamp(E_last, timezone.utc)
    return {"time": start_t.strftime("%Y-%m-%d %H:%M"), "ok": ok,
            "clock_ms": (whole + delta) * 1000.0 if ok else None,
            "bits": word, "conf": "".join(conf), "reason": reason}


def tx_line(r, raw):
    """One log line for a finished transmission."""
    unit = "dB (raw)" if raw else "dB SNR"
    stamp = time.strftime("%H:%M:%S", time.gmtime(r["wall"]))
    bw = "--" if r["bw"] is None else f"{r['bw']:.1f} kHz"
    pk = "--" if r["peak"] is None else f"{r['peak']:+.1f} {unit}"
    av = "--" if r["avg"] is None else f"{r['avg']:+.1f} {unit}"
    sn = "--" if r.get("snr") is None else f"{r['snr']:+.1f} dB"
    return (f"{stamp} UTC  {r['mhz']:.4f} MHz  BW {bw}  peak {pk}  avg {av}  "
            f"avg SNR {sn}  {r['dur']:.1f} s")


def tx_noise_line(rep):
    """One log line of background noise stats."""
    stamp = time.strftime("%H:%M", time.gmtime(rep["wall"]))
    return (f"{stamp} UTC  NOISE  floor {rep['floor']:.1f} dB (raw, central 50%)  "
            f"2 kHz-avg excess above median, max {rep['max']:+.1f} dB, mean {rep['mean']:+.2f} dB, "
            f"threshold {TX_THRESHOLD_DB:.1f} dB  over {rep['rows']} rows")


class TxDetector:
    """Tx Test: finds transmissions and their peak, bandwidth, SNR and duration, and tracks the
    background. Peak hold uses the rows feed() returns."""

    def __init__(self, freq_khz, row_s):
        self.freq_khz = freq_khz
        self.row_s = row_s
        n = len(freq_khz)
        dk = float(np.median(np.diff(freq_khz)))
        self.n_stat = max(1, int(round(TX_STAT_KHZ / dk)))
        idx = np.arange(n)
        self.center_mask = (idx >= n // 4) & (idx < 3 * n // 4)
        self.ref_lin = None     # background reference, linear power per column
        self.ref_s = None       # ref_lin averaged over TX_STAT_KHZ
        self.warm_sum = None
        self.warm_n = 0
        self.new_result = None  # finished transmission waiting to be logged
        self._reset_noise()
        self.reset_events()

    def _smooth(self, arr, n):
        n = min(n, len(arr))         # np.convolve 'same' returns the longer input's length
        return np.convolve(arr, np.ones(n) / n, mode="same")

    def _reset_noise(self):
        self.noise_max = -np.inf
        self.noise_sum = 0.0
        self.noise_n = 0

    def _noise_sample(self, top, mean):
        self.noise_max = max(self.noise_max, top)
        self.noise_sum += mean
        self.noise_n += 1

    def take_noise_report(self, reset=True):
        """Noise stats since the last reset, or None if there is nothing to report.
        reset=False reads them without clearing the accumulators (for Check Noise)."""
        if self.ref_lin is None or self.noise_n == 0:
            if reset:
                self._reset_noise()
            return None
        rep = {"wall": time.time(),
               "floor": float(np.median(10 * np.log10(self.ref_lin[self.center_mask] + 1e-30))),
               "max": float(self.noise_max),
               "mean": self.noise_sum / self.noise_n,
               "rows": self.noise_n}
        if reset:
            self._reset_noise()
        return rep

    def reset_events(self):
        """Forget any candidate or transmission in progress (keeps the background reference)."""
        self._drop_candidate()
        self.new_result = None

    def reset_reference(self):
        """Discard the background reference and re-measure it from the next warm-up period."""
        self._drop_candidate()
        self.ref_lin = None
        self.ref_s = None
        self.warm_sum = None
        self.warm_n = 0

    def _drop_candidate(self):
        self.cand = None        # row time at which the current candidate began
        self.last_above = None  # row time of the latest row above threshold
        self.center = None      # column of the signal's strongest statistic
        self.mask = None        # columns within the channel
        self.active = False     # True once the candidate has lasted TX_MIN_S
        self.queue = deque()    # (t, display row, linear row, channel peak), held back
        self.tx_start = None
        self.tx_peak = None
        self.tx_sum = 0.0
        self.tx_n = 0
        self.tx_abs = None      # per-column sum of linear power over the counted rows
        self.tx_lin = None      # per-column sum of power relative to the peak's level
        self.last = None

    def _width_khz(self, S):
        """Width of the contiguous columns around the candidate centre at or above threshold."""
        lo = hi = self.center
        while lo > 0 and S[lo - 1] >= TX_THRESHOLD_DB:
            lo -= 1
        while hi < len(S) - 1 and S[hi + 1] >= TX_THRESHOLD_DB:
            hi += 1
        return float(self.freq_khz[hi] - self.freq_khz[lo])

    def _track_ref(self, lin):
        w = self.row_s / TX_REF_TAU_S
        self.ref_lin = self.ref_lin + w * (lin - self.ref_lin)
        self.ref_s = self._smooth(self.ref_lin, self.n_stat)

    def feed(self, t, lin, disp):
        """One row at time t (s since configure). lin: averaged linear power per column;
        disp: displayed values. Returns the display rows to count toward peak hold now."""
        if self.ref_lin is None:
            self.warm_sum = lin.copy() if self.warm_sum is None else self.warm_sum + lin
            self.warm_n += 1
            if t >= TX_WARMUP_S:
                self.ref_lin = self.warm_sum / self.warm_n
                self.ref_s = self._smooth(self.ref_lin, self.n_stat)
            return [disp]
        S = 10 * np.log10(self._smooth(lin, self.n_stat) / self.ref_s + 1e-30)
        S = S - float(np.median(S[self.center_mask]))     # remove band-wide floor changes
        if self.cand is None:
            masked = np.where(self.center_mask, S, -np.inf)
            idx = int(np.argmax(masked))
            top = float(masked[idx])
            if top < TX_THRESHOLD_DB:
                self._noise_sample(top, float(S[self.center_mask].mean()))
                self._track_ref(lin)
                return [disp]
            self.cand = t
            self.center = idx
            self.mask = np.abs(self.freq_khz - self.freq_khz[idx]) <= TX_CHANNEL_HALF_KHZ
            self.last_above = t
            self.queue.clear()
        if float(S[self.mask].max()) >= TX_THRESHOLD_DB:
            self.last_above = t
        out = []
        if t - self.last_above > TX_GAP_S or t - self.cand > TX_MAX_S:
            if self.active:
                self._finish()          # the queue holds only the last TX_SKIP_S: dropped
            else:
                out.extend(d for _, d, _, _ in self.queue)     # not a transmission after all
            self._drop_candidate()
            out.append(disp)
            return out
        if t >= self.cand + TX_SKIP_S:
            self.queue.append((t, disp, lin, float(disp[self.mask].max())))
        if not self.active and self.last_above - self.cand >= TX_MIN_S:
            if self._width_khz(S) > TX_MAX_BW_KHZ:
                out.extend(d for _, d, _, _ in self.queue)     # too wide: not voice
                self._drop_candidate()
                out.append(disp)
                return out
            self.active = True
            self.tx_start = time.time() - (t - self.cand)
        if self.active:
            while self.queue and self.queue[0][0] <= t - TX_DELAY_S:
                _, d, l, pk = self.queue.popleft()
                out.append(d)
                self._count(l, d, pk)
        return out

    def _count(self, lin, disp, pk):
        self.tx_peak = pk if self.tx_peak is None else max(self.tx_peak, pk)
        self.tx_sum += pk
        self.tx_n += 1
        self.tx_abs = lin.copy() if self.tx_abs is None else self.tx_abs + lin
        rel = 10 ** (disp / 10.0)
        self.tx_lin = rel.copy() if self.tx_lin is None else self.tx_lin + rel

    def _bandwidth(self):
        """Width in kHz of the contiguous columns within TX_BW_DB of the peak, the peak column,
        and the first and last of those columns."""
        if self.tx_lin is None:
            return None, self.center, None, None
        mean = self.tx_lin / self.tx_n
        dk = float(np.median(np.diff(self.freq_khz)))
        n = max(1, min(int(round(TX_SMOOTH_KHZ / dk)), len(mean)))
        spec = 10 * np.log10(np.convolve(mean, np.ones(n) / n, mode="same") + 1e-30)
        p = int(np.argmax(np.where(self.mask, spec, -np.inf)))
        level = spec[p] - TX_BW_DB
        lo = p
        while lo > 0 and spec[lo - 1] >= level:
            lo -= 1
        hi = p
        while hi < len(spec) - 1 and spec[hi + 1] >= level:
            hi += 1
        return float(self.freq_khz[hi] - self.freq_khz[lo]), p, lo, hi

    def _finish(self):
        bw, p, lo, hi = self._bandwidth()
        snr = None
        if lo is not None and self.tx_abs is not None:
            sig = float(np.mean(self.tx_abs[lo:hi + 1] / self.tx_n))
            bg = float(np.mean(self.ref_lin[lo:hi + 1]))
            snr = 10 * np.log10(sig / bg + 1e-30)
        self.last = {"wall": self.tx_start,
                     "mhz": float(self.freq_khz[p]) / 1e3,
                     "bw": bw,
                     "peak": self.tx_peak,
                     "avg": self.tx_sum / self.tx_n if self.tx_n else None,
                     "snr": snr,
                     "dur": self.last_above - self.cand}
        self.new_result = self.last


# Track mode: follows each signal in the span across display rows and reports every transmission.
# A new transmission must rise well above the band floor; an existing one continues while it stays
# above a lower level (hysteresis), so the skirts of a strong signal do not start tracks of their own.
# Runs are matched to tracks by frequency extent, so a transmission and its reply on the same
# frequency are split apart by the gap between them.
TRACK_START_DB = 12.0           # a new transmission must rise this far above the band floor
TRACK_KEEP_DB = 6.0             # an existing one continues while above this
TRACK_SMOOTH_KHZ = 0.5          # spectrum is smoothed over this width before thresholding
TRACK_MIN_WIDTH_KHZ = 0.5       # narrower runs are ignored
TRACK_MARGIN_KHZ = 0.5          # a run this close to a track's extent belongs to that track
TRACK_HANG_S = 0.03             # a dropout longer than this ends a transmission (keep it near one row)
TRACK_MIN_S = 0.10              # shorter tracks are not logged
TRACK_RECENT = 50               # finished transmissions remembered for the turnaround (gap) time
TRACK_CSV_FIELDS = ["track", "utc_start", "epoch_start", "dur_ms", "gap_ms",
                    "center_khz", "peak_db", "width_khz", "raw"]


class TrackDetector:
    """Track mode. feed() takes one display row at a time and returns the transmissions that ended
    on it. gap_ms is the time since the same frequency last went quiet, i.e. the turnaround."""

    def __init__(self):
        self.active = []        # transmissions in progress: frequency extent, level, timing
        self.recent = []        # (lo, hi, t_end) of finished transmissions, for the gap time
        self.next_id = 1

    def flush(self):
        """End every transmission in progress (retune, pause, leaving the mode). Returns their records."""
        out = [r for r in (self._close(tr) for tr in self.active) if r is not None]
        self.active = []
        self.recent = []
        return out

    def _close(self, tr):
        dur = tr["t_end"] - tr["t_start"]
        if dur < TRACK_MIN_S:
            return None                 # a blip: not a transmission, and not a gap reference either
        self.recent = (self.recent + [(tr["lo"], tr["hi"], tr["t_end"])])[-TRACK_RECENT:]
        return {"id": tr["id"], "t_start": tr["t_start"], "t_end": tr["t_end"], "dur_s": dur,
                "gap_s": tr["gap_s"], "center_khz": tr["csum"] / tr["n"],
                "peak_db": tr["peak"], "width_khz": tr["hi"] - tr["lo"]}

    def feed(self, row_db, freq_khz, t):
        """One display row at wall-clock time t (s). Returns the transmissions that ended on this row."""
        f = np.asarray(freq_khz, dtype=float)
        if f.size < 4:
            return []
        df = abs(f[1] - f[0]) or 1e-9
        n_sm = min(f.size, max(1, int(round(TRACK_SMOOTH_KHZ / df))))
        lin = 10.0 ** (np.asarray(row_db, dtype=float) / 10.0)
        sm = 10.0 * np.log10(np.convolve(lin, np.ones(n_sm) / n_sm, mode="same") + 1e-30)
        floor = float(np.percentile(sm, 25))
        keep = sm > floor + TRACK_KEEP_DB
        min_bins = max(2, int(round(TRACK_MIN_WIDTH_KHZ / df)))
        edges = np.flatnonzero(np.diff(np.concatenate(([0], keep.astype(np.int8), [0]))))
        updated = set()
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a < min_bins:
                continue
            lo, hi = sorted((float(f[a]), float(f[b - 1])))
            c = float(f[a:b].mean())
            pk = float(sm[a:b].max())
            match = next((tr for tr in self.active
                          if tr["lo"] - TRACK_MARGIN_KHZ <= hi and lo <= tr["hi"] + TRACK_MARGIN_KHZ),
                         None)
            if match is None:
                if pk - floor < TRACK_START_DB:
                    continue            # too faint to start a transmission
                prev = [te for rlo, rhi, te in self.recent
                        if rlo - TRACK_MARGIN_KHZ <= hi and lo <= rhi + TRACK_MARGIN_KHZ]
                match = {"id": self.next_id, "t_start": t, "t_end": t,
                         "gap_s": (t - max(prev)) if prev else None,
                         "lo": lo, "hi": hi, "csum": 0.0, "n": 0, "peak": pk}
                self.next_id += 1
                self.active.append(match)
            if id(match) in updated:
                match["lo"], match["hi"] = min(match["lo"], lo), max(match["hi"], hi)
            else:
                match["lo"], match["hi"] = lo, hi
                updated.add(id(match))
            match["t_end"] = t
            match["csum"] += c
            match["n"] += 1
            match["peak"] = max(match["peak"], pk)
        done, still = [], []
        for tr in self.active:
            if t - tr["t_end"] <= TRACK_HANG_S:
                still.append(tr)
            else:
                rec = self._close(tr)
                if rec is not None:
                    done.append(rec)
        self.active = still
        return done


def track_line(r, raw):
    """One decodes-pane line for a finished transmission."""
    unit = "dB (raw)" if raw else "dB SNR"
    stamp = datetime.fromtimestamp(r["t_start"], timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    gap ="--" if r["gap_s"] is None else f"{r['gap_s'] * 1e3:.0f} ms"
    return (f"{stamp} UTC  track {r['id']:<4d} {r['center_khz'] / 1e3:.4f} MHz  "
            f"dur {r['dur_s'] * 1e3:.0f} ms  gap {gap}  peak {r['peak_db']:+.1f} {unit}  "
            f"width {r['width_khz']:.1f} kHz")


def track_csv_row(r, raw):
    t0 = datetime.fromtimestamp(r["t_start"], timezone.utc)
    return [r["id"], t0.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], f"{r['t_start']:.3f}",
            f"{r['dur_s'] * 1e3:.0f}", "" if r["gap_s"] is None else f"{r['gap_s'] * 1e3:.0f}",
            f"{r['center_khz']:.3f}", f"{r['peak_db']:.2f}", f"{r['width_khz']:.3f}", int(raw)]


# ---- CW decoder: Morse on/off keying of the strongest carrier in the display window ----

CW_SPAN_KHZ = 3.0               # display width when CW mode is entered, centred on the tuning
CW_N = 2048                     # FFT size for the carrier search and the band power
CW_HOP_S = 0.005                # one band-power sample every 5 ms
CW_AVG_S = 0.5                  # averaging time of the spectrum used to find the carrier (tracks drift)
CW_BAND_HALF_HZ = 150.0         # carrier power is summed over this half-width (signal is about 100 Hz wide)
CW_SMOOTH_FRAMES = 3            # frames averaged in linear power before the level is measured
CW_RESEL_FRAMES = 5             # how often the carrier bin is re-chosen (5 frames = 25 ms)
CW_MIN_SEP_DB = 6.0            # threshold is never closer than half this above the key-up level
CW_HYST_DB = 1.5                # hysteresis around the key threshold
CW_HIST_S = 30.0                # key-up level history (noise floor)
CW_LEVEL_S = 3.0               # recent level history (key-down level), so each transmission sets its own threshold
CW_QUEUE_S = 5.0                # frames kept until a threshold exists
CW_TX_END_S = 2.5               # this much key-up ends a transmission
CW_MIN_ON_S = 0.035             # shorter key-down blips are treated as noise (dit at 22 wpm is 55 ms)
CW_TRACE_S = 10.0               # comb level history kept for the FM level plot
CW_WPM_MIN = 8.0
CW_WPM_MAX = 22.0
CW_MIN_CHARS = 3                # a transmission needs this many recognised characters to be logged
CW_CODE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.",
    "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.",
    "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-",
    "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....",
    "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.", "=": "-...-",
}
CW_TEXT = {code: ch for ch, code in CW_CODE.items()}


def cw_estimate_unit(ons, offs):
    """Dit length in seconds that best fits the key-down durations (1 or 3 dits) and the key-up
    durations (1, 3 or 7 dits), searched over CW_WPM_MIN to CW_WPM_MAX."""
    on = np.asarray(ons, dtype=np.float64)
    off = np.asarray(offs, dtype=np.float64)
    best_e, best_u = np.inf, None
    for wpm in np.arange(CW_WPM_MIN, CW_WPM_MAX + 1e-9, 0.1):
        u = 1.2 / wpm
        e = float(np.min([np.log(on / u) ** 2, np.log(on / (3 * u)) ** 2], axis=0).sum())
        if off.size:
            e += float(np.min([np.log(off / u) ** 2, np.log(off / (3 * u)) ** 2,
                               np.log(off / (7 * u)) ** 2], axis=0).sum())
        if e < best_e:
            best_e, best_u = e, u
    return best_u


def cw_decode(segs, unit):
    """segs: [is_key_down, seconds] in time order. Returns (text, recognised character count).
    A dit is under 2 units and a dah is over. Key-up under 2 units stays within a character,
    2 to 5 units ends a character, 5 units or more is a word space. Unknown codes show as #."""
    out, sym = [], ""
    for on, d in segs:
        if on:
            sym += "." if d < 2 * unit else "-"
            continue
        if d < 2 * unit:
            continue
        if sym:
            out.append(CW_TEXT.get(sym, "#"))
            sym = ""
        if d >= 5 * unit:
            out.append(" ")
    if sym:
        out.append(CW_TEXT.get(sym, "#"))
    text = " ".join("".join(out).split())
    return text, sum(1 for c in text if c not in " #")


def cw_line(rec):
    stamp = time.strftime("%H:%M:%S", time.gmtime(rec["wall"]))
    return (f"{stamp} UTC  {rec['mhz']:.4f} MHz  CW {rec['wpm']:.1f} wpm  {rec['text']}  "
            f"raw {rec['raw']:+.1f} dB  SNR {rec['snr']:+.1f} dB  {rec['dur']:.1f} s")


_TX_IDS = itertools.count(1)                     # transmission numbers, unique across decoders
STREAM_UNIT_GUESS_S = 0.1                        # dit length used before a transmission has an estimate (12 wpm)


class CwDecoder:
    """CW decoder for the strongest on/off-keyed carrier in a display window. Band power around
    the carrier is sampled every CW_HOP_S. Key-down runs are grouped into transmissions, and each
    transmission is decoded when it ends. Raw level is the mean key-down band power in dB
    (uncalibrated); SNR is that above the key-up level."""

    STREAM_CHARS = False                             # True: characters are emitted as they complete
    REJECT_FRAC = None                               # key-down elements further than this from ideal are dropped
    TURNON_MASK_S = None                             # s after a carrier turns on in which no element is kept (strong signals)
    FFT_N = CW_N                                     # FFT length (a subclass may use another)

    def __init__(self, ring, rate):
        self.ring = ring
        self.rate = rate
        self.nfft = self.FFT_N
        self.hop = max(1, int(round(CW_HOP_S * rate)))
        self.win = np.hanning(self.nfft).astype(np.float32)
        self.wnorm = float(np.sum(self.win.astype(np.float64) ** 2))
        self.freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, 1.0 / rate))
        self.a_avg = self.hop / (CW_AVG_S * rate)
        self.hist_len = int(CW_HIST_S * rate / self.hop)
        self.recent_len = int(CW_LEVEL_S * rate / self.hop)
        self.queue_len = int(CW_QUEUE_S * rate / self.hop)
        self.trace_len = int(CW_TRACE_S * rate / self.hop)
        self.win_mask = np.ones(self.nfft, dtype=bool)
        self.tune_hz = 0
        self.reset()

    def set_window(self, tune_hz, lo_off, hi_off):
        """Search window in Hz offsets from tuning. Any change restarts the decoder."""
        self.tune_hz = int(tune_hz)
        self.win_mask = (self.freqs >= lo_off) & (self.freqs <= hi_off)
        self.reset()

    def reset(self):
        self.pos = None                                  # next ring sample to read
        self.avg = None                                  # time-averaged power, for the carrier search
        self.frame_n = 0
        self.fc = None                                   # carrier offset from tuning, Hz
        self.band = None                                 # bins summed for the carrier power
        self.hist = deque(maxlen=self.hist_len)          # band level, dB, for the noise floor
        self.aux_hist = deque(maxlen=self.hist_len)      # second feature in dB over 30 s
        self.aux_recent = deque(maxlen=self.recent_len)  # second feature in dB (FM carrier), for the fit
        self.recent = deque(maxlen=self.recent_len)      # band level, dB, for the key-down level
        self.sm = deque(maxlen=CW_SMOOTH_FRAMES)         # recent band powers, for smoothing
        self.trace = deque(maxlen=self.trace_len)        # (centre, level dB, threshold dB, key-up dB, key down)
        self.queue = deque(maxlen=self.queue_len)        # (centre sample, level) awaiting decisions
        self.thr = None                                  # key-down threshold, dB (None until keying is seen)
        self.noise_db = None                             # key-up level, dB
        self.run_on = False                              # state of the run being accumulated
        self.run_n = 0
        self.run_t0 = 0
        self.run_pow = 0.0
        self.closed = []                                 # finished runs since the last transmission
        self.events = []                                 # char and drop events for process() to return
        self.tx_id = None                                # number of the transmission being accumulated
        self.tx_chars = 0                                # characters already emitted for it
        self.stream_unit = None                          # dit length from the last streamed estimate, s
        self.gap_fired = False                           # the open key-up gap has been checked for a character
        self.tx_spec = None                              # key-down spectrum summed over the transmission (FM)
        self.tx_spec_n = 0
        self.aux_floor_db = None                         # 10th percentile of the carrier level, dB (turn-on mask)
        self.carrier_on = None                           # centre sample of the carrier's turn-on, or None
        self.carrier_peak = -math.inf                    # peak comb level in the turn-on window, dB
        self.carrier_mask = None                         # mask length for this turn-on, s (0: none); None until decided
        self.carrier_below = 0                           # frames in a row with the carrier below its level
        self.pre_levels = []                             # comb levels in the carrier-only preamble (strong signals)
        self.pre_done = False                            # the preamble has ended
        self.last_turnon = None                          # (turn-on centre, mask s) of the last decided turn-on

    def process(self, now=None):
        """Measure new frames and return the transmissions that ended since the last call."""
        ring, hop = self.ring, self.hop
        total = ring.total
        now = time.time() if now is None else now
        n = self.nfft
        if total < n:
            return []
        if self.pos is None or total - self.pos > ring.size - n:
            self.pos = total - n
        avail = total - self.pos
        if avail < n:
            return []
        k = min((avail - n) // hop + 1, 200)
        seg = ring.read(self.pos, (k - 1) * hop + n)
        if seg is None:
            self.pos = None
            return []
        base = self.pos
        self.pos += k * hop
        frames = np.lib.stride_tricks.sliding_window_view(seg, n)[::hop] * self.win
        X = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
        P = X.real * X.real + X.imag * X.imag
        for i in range(k):
            self._frame(P[i], base + i * hop + n // 2)
        out = self._drain(total, now)
        if self.events:
            out = self.events + out
            self.events = []
        return out

    def _frame(self, Pi, center):
        self.frame_n += 1
        if self.avg is None:
            self.avg = Pi.astype(np.float64)
        else:
            self.avg += self.a_avg * (Pi - self.avg)
        if self.fc is None or self.frame_n % CW_RESEL_FRAMES == 0:   # a carrier that appears is picked up within a few frames
            idx = int(np.argmax(np.where(self.win_mask, self.avg, -np.inf)))
            self.fc = self._refine_peak(idx)
            self._set_bands()
        self.sm.append(self._power(Pi))
        env = sum(self.sm) / len(self.sm)
        lev = 10.0 * math.log10(env + 1e-30)
        self.hist.append(lev)
        self.recent.append(lev)
        aux = self._aux(Pi)                              # appended before the fit so the histories match
        aux_db = 10.0 * math.log10(aux + 1e-30)
        self.aux_recent.append(aux_db)
        self.aux_hist.append(aux_db)
        if self.TURNON_MASK_S is not None:
            self._track_carrier(center, lev, aux_db)
        if self.frame_n % 20 == 0 and len(self.hist) >= 200:
            self._fit_levels()
        self.queue.append((center, env, aux))

    def _track_carrier(self, center, lev, aux_db):
        """Turn-on mask (FM only). The turn-on starts at the first frame whose carrier level is FM_CARRIER_ON_DB
        above its floor. Once the window has passed, its peak comb level decides the mask for that turn-on."""
        if self.aux_floor_db is None:
            return
        above = aux_db > self.aux_floor_db + FM_CARRIER_ON_DB
        if self.carrier_on is None:
            if above:
                self.carrier_on = center
                self.carrier_peak = lev
                self.carrier_mask = None
                self.carrier_below = 0
                self.pre_levels, self.pre_done = [], False
            return
        if not above:
            self.carrier_below += 1
            if self.carrier_below * self.hop / self.rate >= FM_CARRIER_OFF_S:
                self.carrier_on, self.carrier_mask, self.carrier_below = None, None, 0
                self.pre_levels, self.pre_done = [], False
            return
        self.carrier_below = 0
        self._track_preamble(center, lev)
        if self.carrier_mask is not None:
            return
        if center < self.carrier_on + self.TURNON_MASK_S * self.rate:
            self.carrier_peak = max(self.carrier_peak, lev)
        elif self.noise_db is not None:
            self._decide_carrier_mask()

    def _track_preamble(self, center, lev):
        """Carrier-only comb level after the turn-on window and before the first key down. For a strong
        signal this is the key-up level, which the noise-only history cannot give."""
        if self.pre_done or center < self.carrier_on + self.TURNON_MASK_S * self.rate:
            return
        if len(self.pre_levels) >= FM_PRE_MIN_FRAMES:
            if lev > float(np.median(self.pre_levels)) + FM_PRE_JUMP_DB:
                self.pre_done = True
                return
        self.pre_levels.append(lev)
        if len(self.pre_levels) >= FM_PRE_MAX_FRAMES:
            self.pre_done = True

    def _preamble(self):
        """(median, sd) in dB of the carrier-only preamble, if the signal is strong, else None."""
        if self.carrier_on is None or self.noise_db is None or len(self.pre_levels) < FM_PRE_MIN_FRAMES:
            return None
        p = np.asarray(self.pre_levels, dtype=np.float64)
        p0 = float(np.median(p))
        if p0 - self.noise_db < FM_TURNON_MASK_DB:
            return None
        return p0, max(float(p.std()), 0.5)

    def _decide_carrier_mask(self):
        strong = self.noise_db is not None and self.carrier_peak - self.noise_db >= FM_TURNON_MASK_DB
        self.carrier_mask = self.TURNON_MASK_S if strong else 0.0
        self.last_turnon = (self.carrier_on, self.carrier_mask)

    def _fit_levels(self):
        """Key-up level (10th percentile of 30 s) and key-down threshold (halfway from there to the
        90th percentile of the last 3 s)."""
        lo = float(np.percentile(np.asarray(self.hist), 10))
        hi = float(np.percentile(np.asarray(self.recent), 90))
        self.noise_db = lo
        self.thr = lo + 0.5 * max(hi - lo, 2.0 * CW_MIN_SEP_DB)

    def _carrier_fine(self):
        """Carrier offset from tuning used for the transmission's frequency. The base decoder has
        only the peak bin, refined in _refine_peak."""
        return self.fc

    def _refine_peak(self, k):
        """Carrier offset from tuning, Hz: the peak bin of the averaged spectrum, refined to a fraction
        of a bin by a parabola through the dB values of that bin and its two neighbours."""
        if 0 < k < self.nfft - 1:
            y = 10.0 * np.log10(self.avg[k - 1:k + 2] + 1e-30)
            den = float(y[0] - 2.0 * y[1] + y[2])
            if den < 0.0:
                df = float(self.freqs[k + 1] - self.freqs[k])
                return float(self.freqs[k]) + 0.5 * float(y[0] - y[2]) / den * df
        return float(self.freqs[k])

    def _set_bands(self):
        """Bins that make up the level, for the current carrier."""
        self.band = np.flatnonzero(np.abs(self.freqs - self.fc) <= CW_BAND_HALF_HZ)

    def _power(self, Pi):
        """Linear power of one frame in the bins from _set_bands, normalised by the window."""
        return float(np.sum(Pi[self.band])) / self.wnorm

    def _aux(self, Pi):
        """Second feature of one frame, linear power. The base decoder has none."""
        return 0.0

    def _closed_segs(self, closed, open_s=0.0):
        """Key runs as [key down, seconds] segments. Runs too short to be elements become key up,
        and neighbours of the same state merge. open_s adds a key-up run still in progress, and the
        leading key-up is dropped. Returns (segs, t_first): t_first is the centre sample of the first
        element."""
        segs, t_first = [], None
        for on, n, t0, _ in closed:
            d = n * self.hop / self.rate
            if on and d < CW_MIN_ON_S:
                on = False
            if on and t_first is None:
                t_first = t0
            if segs and segs[-1][0] == on:
                segs[-1][1] += d
            else:
                segs.append([on, d])
        if open_s > 0.0:
            if segs and not segs[-1][0]:
                segs[-1][1] += open_s
            else:
                segs.append([False, open_s])
        while segs and not segs[0][0]:
            segs.pop(0)
        return segs, t_first

    def _turnon(self, segs, t_first, final):
        """Apply the turn-on mask (TURNON_MASK_S, FM only). The window runs from the carrier's turn-on; elements
        that start in it are removed when the turn-on was strong. The anchor is kept after the carrier drops, so
        the final decode of a transmission uses the same mask as its streamed characters. While the mask is
        undecided, returns None and the caller waits (streaming). Returns (segs, t_first, masked)."""
        if self.TURNON_MASK_S is None:
            return segs, t_first, []
        if self.carrier_on is not None:
            if self.carrier_mask is None:
                if not final:
                    return None
                self._decide_carrier_mask()
            anchor = (self.carrier_on, self.carrier_mask)
        else:
            anchor = self.last_turnon
        if anchor is None or anchor[1] <= 0.0:
            return segs, t_first, []
        on, mask = anchor
        first_s = max(0.0, (t_first - on) / self.rate)
        segs, lead, masked = cw_mask_turnon(segs, mask, first_s)
        return segs, t_first + lead * self.rate, masked

    def _stream(self, open_s=0.0):
        """Emit the characters of the current transmission that are complete (STREAM_CHARS only). A
        character is complete once a key-up gap of two dits or more follows it. The dit length is
        re-estimated from the transmission so far, so an early character can still change at the end."""
        segs, t_first = self._closed_segs(self.closed, open_s)
        if t_first is None:
            return
        held = self._turnon(segs, t_first, final=False)
        if held is None:
            return
        segs, t_first, _ = held
        ons = [d for on, d in segs if on]
        if not ons:
            return
        offs = [d for on, d in segs if not on]
        unit = cw_estimate_unit(ons, offs)
        self.stream_unit = unit
        chars = cw_char_times(segs, unit, final=False)
        for ch, t0, t1 in chars[self.tx_chars:]:
            self.events.append({"kind": "char", "tx": self.tx_id, "ch": ch,
                                "sample": t_first + 0.5 * (t0 + t1) * self.rate})
        self.tx_chars = max(self.tx_chars, len(chars))

    def _drain(self, total, now):
        """Decide key-down or key-up for queued frames once a threshold exists. Returns finished
        transmissions. Frames queued before the threshold are decided retroactively."""
        out = []
        while self.thr is not None and self.queue:
            center, env, _ = self.queue.popleft()
            env_db = 10.0 * math.log10(env + 1e-30)
            on = self.run_on
            if on and env_db < self.thr - CW_HYST_DB:
                on = False
            elif not on and env_db > self.thr + CW_HYST_DB:
                on = True
            self.trace.append((center, env_db, self.thr, self.noise_db, on,
                               self.thr - CW_HYST_DB, self.thr + CW_HYST_DB))
            rec = self._step(center, env, on, total, now)
            if rec is not None:
                out.append(rec)
        return out

    def _step(self, center, env, on, total, now):
        """Add one decided frame to the current run. Returns a finished transmission or None."""
        if on != self.run_on:
            if self.run_n:
                self.closed.append((self.run_on, self.run_n, self.run_t0, self.run_pow))
                if self.run_on:
                    if self.tx_id is None:
                        self.tx_id = next(_TX_IDS)
                elif self.STREAM_CHARS and self.tx_id is not None:
                    self._stream()                   # a key-up run has closed: a character may be complete
            self.run_on, self.run_n, self.run_pow = on, 0, 0.0
            self.gap_fired = False
        if self.run_n == 0:
            self.run_t0 = center
        self.run_n += 1
        self.run_pow += env
        if self.run_on:
            return None
        gap_s = self.run_n * self.hop / self.rate
        if (self.STREAM_CHARS and not self.gap_fired and self.tx_id is not None
                and gap_s >= 2.0 * (self.stream_unit or STREAM_UNIT_GUESS_S)):
            self.gap_fired = True                    # the key-up gap is already two dits: the character is complete
            self._stream(open_s=gap_s)
        if gap_s < CW_TX_END_S:
            return None
        rec = None
        if any(r[0] for r in self.closed):
            rec = self._finish(total, now)
            if rec is None and self.tx_chars:
                self.events.append({"kind": "drop", "tx": self.tx_id})
        self.closed = []
        self.tx_id, self.tx_chars, self.stream_unit = None, 0, None
        self.tx_spec, self.tx_spec_n = None, 0
        return rec

    def _wall_of(self, t_sample, total, now):
        """Wall-clock time of an absolute sample: from the USB push that delivered it, when the ring has
        push records, otherwise from the time of this call."""
        w = self.ring.wall_at(t_sample, self.rate) if hasattr(self.ring, "wall_at") else None
        return w if w is not None else now - (total - t_sample) / self.rate

    def _finish(self, total, now):
        segs, t_first = self._closed_segs(self.closed)
        while segs and not segs[-1][0]:
            segs.pop()
        if t_first is None:
            return None
        segs, t_first, masked = self._turnon(segs, t_first, final=True)
        on_pow, on_n = 0.0, 0
        for on, n, _, pw in self.closed:
            if on and n * self.hop / self.rate >= CW_MIN_ON_S:
                on_pow += pw
                on_n += n
        ons = [d for on, d in segs if on]
        if not ons or t_first is None:
            return None
        offs = [d for on, d in segs if not on]
        unit = cw_estimate_unit(ons, offs)
        c = 0.0
        for _ in range(3):                           # classes depend on the unit, and the unit on the classes
            c, unit = cw_fit_timing(segs, unit)
        dropped = []
        if self.REJECT_FRAC is not None:
            segs, lead, dropped = cw_drop_outliers(segs, unit, self.REJECT_FRAC)
            if dropped:
                t_first += lead * self.rate          # the first element is now the first one kept
                for _ in range(3):
                    c, unit = cw_fit_timing(segs, unit)
        text, n_chars = cw_decode(segs, unit)
        if n_chars < CW_MIN_CHARS:
            return None
        raw = 10.0 * math.log10(on_pow / on_n + 1e-30)
        return {"wall": self._wall_of(t_first, total, now) + c,
                "mhz": (self.tune_hz + self._carrier_fine()) / 1e6,
                "wpm": 1.2 / unit,
                "text": text,
                "raw": raw,
                "snr": raw - self.noise_db,
                "dur": sum(d for _, d in segs),
                "unit": unit,
                "segs": segs,
                "t_first": t_first,
                "char_t": cw_char_times(segs, unit),
                "kind": "tx", "tx": self.tx_id,
                "dropped": dropped, "masked": masked}


# ---- FM keyed-tone decoder: Morse sent as a 400 Hz FM tone, read from the carrier's sidebands

FM_TONE_HZ = 400.0                               # modulating tone: sidebands sit at carrier +/- 400 Hz x k
FM_BETA = 6.25                                   # modulation index: 2.5 kHz deviation / 400 Hz tone
FM_KS = [k for k in range(-7, 8) if k != 0]      # lines used; lines 9 to 11 are staircase images, so stop at 7
FM_W = sspec.jv(np.asarray(FM_KS, dtype=np.float64), FM_BETA) ** 2   # Bessel line powers
FM_W = FM_W / FM_W.sum()                         # matched weights, summing to 1
FM_N = 4096                                      # FFT length: 47 Hz bins, enough to separate lines 400 Hz apart
FM_SIDE_HALF_HZ = 60.0                           # each sideband line is summed over this half-width
FM_SEGMENTER = "viterbi"                         # "viterbi" (two-state HMM) or "hysteresis" (threshold)
FM_SWITCH_P = 0.05                               # per-frame chance of a key change (about 100 ms runs at 5 ms frames)
FM_LAG_S = 0.3                                   # frames this old are committed; newer frames can still change
FM_LAG_FRAMES = int(round(FM_LAG_S / CW_HOP_S))
FM_FEATURE = "carrier3"                           # "carrier3": OFF/UP/DOWN states from comb and carrier; "comb"; "diff"
FM_MIN_SEP_DB = 4.0                             # least key-down/key-up separation accepted by the level fit
FM_HOLD_FRAMES = int(10.0 / CW_HOP_S)           # frames held (up to 10 s) until the levels exist
FM_SMOOTH_FRAMES = 1                             # frames averaged before the level; 1 keeps the edges sharp
FM_REJECT_FRAC = 0.2                             # a key-down element this far from its ideal length (dit 1, dah 3 units) is noise
FM_TURNON_MASK_S = 0.25                         # no element is kept this long after a carrier turns on (turn-on chirp)
FM_TURNON_MASK_DB = 20.0                        # the mask applies only if the peak comb level in that window is this far above key up
FM_CARRIER_ON_DB = 10.0                         # carrier is on when its band level is this far above its 10th-percentile floor
FM_CARRIER_OFF_S = 0.2                          # carrier is off after this long below that level
FM_PRE_FLOOR_DB = 10.0                          # strong signal: fit only levels this far above the noise (drops pre-carrier noise)
FM_PRE_JUMP_DB = 10.0                           # the carrier-only preamble ends when the comb rises this far above its median
FM_PRE_MIN_FRAMES = 20                          # preamble frames needed before it is used (0.1 s)
FM_PRE_MAX_FRAMES = 400                         # preamble frames kept at most (2 s)
FM_PRE_BOOT_DB = 10.0                           # before any key down: threshold this far above the preamble


def cw_char_times(segs, unit, final=True):
    """Start and end of each decoded character, in seconds from the start of segs, as (char, t0, t1).
    With final=False a character still open at the end of segs is left out: it is not yet complete."""
    out, sym, t0, t1 = [], "", 0.0, 0.0
    t = 0.0
    for on, d in segs:
        r = d / unit
        if on:
            if not sym:
                t0 = t
            sym += "." if r < 2 else "-"
            t1 = t + d
        elif r >= 2 and sym:                          # letter or word gap ends the character
            out.append((CW_TEXT.get(sym, "#"), t0, t1))
            sym = ""
        t += d
    if final and sym:
        out.append((CW_TEXT.get(sym, "#"), t0, t1))
    return out


def cw_mask_turnon(segs, mask_s, first_s=0.0):
    """Remove the key-down elements that start within mask_s of the carrier's turn-on: the turn-on chirp of
    a strong carrier. first_s is the time from the turn-on to the first element in segs. Each removed element
    becomes key up, and neighbouring key-up runs merge. Leading key up is stripped. Returns (segs, lead,
    masked): lead is the time removed from the start, and masked lists the removed lengths in seconds."""
    start, flags = first_s, []
    for on, d in segs:
        flags.append(bool(on) and start < mask_s)
        start += d
    masked = [d for (_, d), m in zip(segs, flags) if m]
    out = []
    for (on, d), m in zip(segs, flags):
        on = bool(on) and not m
        if out and out[-1][0] == on:
            out[-1][1] += d
        else:
            out.append([on, d])
    lead = 0.0
    while out and not out[0][0]:
        lead += out[0][1]
        out.pop(0)
    return out, lead, masked


def cw_drop_outliers(segs, unit, frac):
    """Remove key-down elements whose length is more than frac away from ideal (dit 1 unit, dah 3 units).
    The key-up runs either side of a removed element become one run. Leading key-up is stripped, and its
    time, with the removed leading elements, is returned as lead. Returns (segs, lead, dropped), where
    dropped lists the lengths in seconds of the removed elements."""
    segs = [[on, d] for on, d in segs]
    lead, dropped = 0.0, []
    while True:
        bad = None
        for i, (on, d) in enumerate(segs):
            if on:
                ideal = 1.0 if d / unit < 2 else 3.0
                if abs(d / unit - ideal) > frac * ideal:
                    bad = i
                    break
        if bad is None:
            break
        dropped.append(segs[bad][1])
        if bad == 0:
            lead += segs[0][1]
            del segs[0]
        elif bad + 1 < len(segs):
            segs[bad - 1][1] += segs[bad + 1][1]
            del segs[bad:bad + 2]
        else:
            del segs[bad]
        while segs and not segs[0][0]:
            lead += segs[0][1]
            del segs[0]
    return segs, lead, dropped


def cw_fit_timing(segs, unit):
    """Least-squares fit of a transmission's start time and dit length to a perfect keyer. Each run is
    classed at the given unit (key down: dit or dah; key up: within a character, letter gap or word
    gap), and its ideal length is 1, 3, 1, 3 or 7 units. The start of each run, measured from the start
    of segs, is fitted as c + unit * S, where S is its ideal start in units. Returns (c, unit): c is the
    offset in seconds of the fitted start from the start of segs."""
    if len(segs) < 3:
        return 0.0, unit
    measured = np.concatenate([[0.0], np.cumsum([d for _, d in segs])[:-1]])
    ideal = []
    for on, d in segs:
        r = d / unit
        if on:
            ideal.append(1.0 if r < 2 else 3.0)
        elif r < 2:
            ideal.append(1.0)
        elif r < 5:
            ideal.append(3.0)
        else:
            ideal.append(7.0)
    S = np.concatenate([[0.0], np.cumsum(ideal)[:-1]])
    A = np.column_stack([np.ones(len(S)), S])
    (c, u), _, _, _ = np.linalg.lstsq(A, measured, rcond=None)
    return float(c), float(u)


def cw_timing(segs, unit):
    """Timing of a transmission against ideal Morse at the fitted speed. Returns (chars, on_dev, off_dev).
    chars: (character, ms from its nearest decision boundary), smallest over its elements and the gap
    after it; a word space is ("/", None). The boundaries are 2 units (dit/dah, intra/letter gap) and
    5 units (letter/word gap). on_dev and off_dev: relative deviation of each key-down and key-up
    duration from its ideal (dit 1, dah 3, gaps 1, 3, 7 units)."""
    chars, sym, cmarg = [], "", []
    on_dev, off_dev = [], []
    for on, d in segs:
        r = d / unit
        if on:
            ideal = 1.0 if r < 2 else 3.0
            on_dev.append((r - ideal) / ideal)
            cmarg.append(abs(r - 2.0))
            sym += "." if r < 2 else "-"
            continue
        if r < 2:
            off_dev.append(r - 1.0)
            cmarg.append(abs(r - 2.0))
            continue
        if r < 5:
            off_dev.append((r - 3.0) / 3.0)
            cmarg.append(min(abs(r - 2.0), abs(r - 5.0)))
            word = False
        else:
            off_dev.append((r - 7.0) / 7.0)
            cmarg.append(abs(r - 5.0))
            word = True
        if sym:
            chars.append((CW_TEXT.get(sym, "#"), min(cmarg) * unit * 1000.0))
        if word:
            chars.append(("/", None))
        sym, cmarg = "", []
    if sym:
        chars.append((CW_TEXT.get(sym, "#"), min(cmarg) * unit * 1000.0))
    return chars, on_dev, off_dev


def fm_lines(rec):
    """One log line for an FM transmission: date and time to the millisecond, the decode, the timing
    summary, then the closest boundary margin of each character as data (ms)."""
    stamp = datetime.fromtimestamp(rec["wall"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    chars, on_dev, off_dev = cw_timing(rec["segs"], rec["unit"])

    def rms(v):
        return 100.0 * math.sqrt(sum(x * x for x in v) / len(v)) if v else float("nan")

    ms = [m for _, m in chars if m is not None]
    closest = min(ms) if ms else float("nan")
    per_char = "  ".join(ch if m is None else f"{ch}:{m:.0f}" for ch, m in chars)
    if rec.get("masked"):
        per_char += "  masked " + ", ".join(f"{d * 1e3:.0f} ms" for d in rec["masked"])
    if rec.get("dropped"):
        per_char += "  dropped " + ", ".join(f"{d * 1e3:.0f} ms" for d in rec["dropped"])
    line = (f"{stamp} UTC  {rec['mhz']:.6f} MHz  FM {rec['wpm']:.1f} wpm  {rec['text']}  "
            f"raw {rec['raw']:+.1f} dB  SNR {rec['snr']:+.1f} dB  {rec['dur']:.1f} s  "
            f"timing on +/-{rms(on_dev):.0f}% gap +/-{rms(off_dev):.0f}%  closest {closest:.0f} ms  "
            f"{per_char}")
    return [line]


def fm_gauss(x, mu, s):
    """Gaussian log-likelihood (up to a constant) of each value."""
    return -0.5 * ((np.asarray(x, dtype=np.float64) - mu) / s) ** 2 - math.log(s)


def fm_loglik(x_db, mu0, s0, mu1, s1):
    """Log-likelihood of each level under key up (column 0) and key down (column 1)."""
    e = np.empty((len(x_db), 2))
    e[:, 0] = fm_gauss(x_db, mu0, s0)
    e[:, 1] = fm_gauss(x_db, mu1, s1)
    return e


def fm_trans2(p_sw):
    """Log transition matrix for key up (0) and key down (1)."""
    lstay, lsw = math.log(1.0 - p_sw), math.log(p_sw)
    return np.array([[lstay, lsw], [lsw, lstay]])


def fm_trans3(p_sw):
    """Log transition matrix for OFF (0, no carrier), UP (1, carrier, tone off) and DOWN (2, carrier,
    tone on). The carrier is on in UP and DOWN, so OFF connects only to UP."""
    lstay, lsw = math.log(1.0 - p_sw), math.log(p_sw)
    bad = -1e9
    return np.array([[lstay, lsw, bad],
                     [0.5 * lsw, lstay, 0.5 * lsw],
                     [bad, lsw, lstay]])


def fm_viterbi(e, trans, prev):
    """Most likely state for each frame, given log-emissions e (frames by states) and a log transition
    matrix. prev is the state before the first frame."""
    T, K = e.shape
    delta = trans[prev] + e[0]
    back = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        cand = delta[:, None] + trans
        back[t] = np.argmax(cand, axis=0)
        delta = np.max(cand, axis=0) + e[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


class FmDecoder(CwDecoder):
    """Keyed-tone decoder for FM transmissions. The carrier is located as in CwDecoder. The level is
    the Bessel-weighted power of the sideband lines at carrier +/- 400 Hz x k: tone on gives the lines,
    tone off gives the carrier alone. Key state comes from a two-state Viterbi pass over the frame
    levels (FM_SEGMENTER = "viterbi"), or from the threshold of CwDecoder ("hysteresis"). Runs, timing
    and Morse decoding are shared with CwDecoder."""

    STREAM_CHARS = True
    REJECT_FRAC = FM_REJECT_FRAC
    TURNON_MASK_S = FM_TURNON_MASK_S
    FFT_N = FM_N

    def reset(self):
        super().reset()
        self.sm = deque(maxlen=FM_SMOOTH_FRAMES)
        self.mu = None           # (key-up mean, key-down mean, key-up sd, key-down sd), dB; None until seen
        self.vit = []            # (centre, linear power) frames not yet committed
        self.prev = 0            # state of the last committed frame
        self.aux_mu = None       # carrier dB: (OFF mean, OFF sd, UP mean, UP sd, DOWN mean, DOWN sd) or None
        self.spec_q = {}         # frame centre -> (first bin, spectrum slice), until the frame is committed
        self.spec_half = int(FM_SPEC_HALF_HZ / float(self.freqs[1] - self.freqs[0]))

    def _fit_levels(self):
        if len(self.aux_hist) >= 200:
            self.aux_floor_db = float(np.percentile(np.asarray(self.aux_hist), 10))
        if FM_SEGMENTER == "hysteresis":
            super()._fit_levels()
            self.mu = None
            return
        self.noise_db = float(np.percentile(np.asarray(self.hist), 10))
        r = np.asarray(self.recent, dtype=np.float64)
        if FM_FEATURE == "diff":                     # key down is high: comb minus carrier
            r = r - np.asarray(self.aux_recent, dtype=np.float64)
        pre = self._preamble()                       # strong signal: carrier-only level, sd, or None
        rf = r if pre is None else r[r > self.noise_db + FM_PRE_FLOOR_DB]
        if rf.size < 20:
            rf = r
        m0, m1 = np.percentile(rf, [10.0, 90.0])
        for _ in range(10):                          # 1-D two-means on the last 3 s of levels above the noise
            low_f = np.abs(rf - m0) <= np.abs(rf - m1)
            if not low_f.any() or low_f.all():
                break
            n0, n1 = float(rf[low_f].mean()), float(rf[~low_f].mean())
            if n0 == m0 and n1 == m1:
                break
            m0, m1 = n0, n1
        low_f = np.abs(rf - m0) <= np.abs(rf - m1)
        if not low_f.any() or low_f.all() or m1 - m0 < FM_MIN_SEP_DB or (~low_f).mean() < 0.05:
            if pre is None:
                return                               # no keying in this window: keep the last valid levels
            p0, p_sd = pre                           # strong, no key down yet: key up at the preamble level
            self.mu = (p0, p0 + 2.0 * FM_PRE_BOOT_DB, p_sd, max(p_sd, 3.0))
            self.thr = p0 + FM_PRE_BOOT_DB
            self.aux_mu = None
            return
        low = np.abs(r - m0) <= np.abs(r - m1)       # full window, for the carrier classes below
        s0 = max(float(rf[low_f].std()), 0.5)
        s1 = max(float(rf[~low_f].std()), 0.5)
        self.mu = (m0, m1, s0, s1)
        self.thr = 0.5 * (m0 + m1)
        self.aux_mu = None
        if FM_FEATURE == "carrier3" and len(self.aux_hist) >= 200:
            hc = np.asarray(self.hist, dtype=np.float64)
            hq = np.asarray(self.aux_hist, dtype=np.float64)
            cl = hq[hc < 0.5 * (m0 + m1)]                # carrier where the comb is key up: OFF and UP
            if cl.size >= 50:
                c0, c1 = np.percentile(cl, [10.0, 90.0])
                for _ in range(10):
                    lo_c = np.abs(cl - c0) <= np.abs(cl - c1)
                    if not lo_c.any() or lo_c.all():
                        break
                    n0, n1 = float(cl[lo_c].mean()), float(cl[~lo_c].mean())
                    if n0 == c0 and n1 == c1:
                        break
                    c0, c1 = n0, n1
                lo_c = np.abs(cl - c0) <= np.abs(cl - c1)
                if lo_c.any() and (~lo_c).any() and float(cl[~lo_c].mean() - cl[lo_c].mean()) >= 6.0:
                    off_mu = float(cl[lo_c].mean())
                    off_sd = max(float(cl[lo_c].std()), 0.5)
                    q = np.asarray(self.aux_recent, dtype=np.float64)
                    up = low & (q > off_mu + 8.0)
                    dn = ~low
                    if up.sum() >= 5 and dn.sum() >= 5:
                        self.aux_mu = (off_mu, off_sd,
                                       float(q[up].mean()), max(float(q[up].std()), 0.5),
                                       float(q[dn].mean()), max(float(q[dn].std()), 0.5))

    def _drain(self, total, now):
        if FM_SEGMENTER == "hysteresis":
            return super()._drain(total, now)
        out = []
        while self.queue:
            self.vit.append(self.queue.popleft())
        if self.mu is None and len(self.vit) < FM_HOLD_FRAMES:
            return out                               # no levels yet: hold the frames, decide them later
        n_commit = len(self.vit) - FM_LAG_FRAMES
        if n_commit <= 0:
            return out
        comb = [10.0 * math.log10(env + 1e-30) for _, env, _ in self.vit]
        nd = float("nan") if self.noise_db is None else self.noise_db
        three = FM_FEATURE == "carrier3" and self.aux_mu is not None
        if self.mu is None:                          # held too long without levels: call it key up
            x = comb
            path = np.zeros(len(x), dtype=np.int64)
            mu0 = mu1 = mid = float("nan")
            down = 1
        elif three:
            mu0, mu1, s0, s1 = self.mu
            x = comb
            off_mu, off_sd, up_mu, up_sd, dn_mu, dn_sd = self.aux_mu
            c = [10.0 * math.log10(aux + 1e-30) for _, _, aux in self.vit]
            lk0 = fm_gauss(comb, mu0, s0)            # OFF and UP both key up in the comb
            lk1 = fm_gauss(comb, mu1, s1)
            e = np.empty((len(comb), 3))
            e[:, 0] = lk0 + fm_gauss(c, off_mu, off_sd)
            e[:, 1] = lk0 + fm_gauss(c, up_mu, up_sd)
            e[:, 2] = lk1 + fm_gauss(c, dn_mu, dn_sd)
            path = fm_viterbi(e, fm_trans3(FM_SWITCH_P), self.prev)
            mid = 0.5 * (mu0 + mu1)
            down = 2
        else:
            mu0, mu1, s0, s1 = self.mu
            x = self._features()
            path = fm_viterbi(fm_loglik(x, mu0, s0, mu1, s1), fm_trans2(FM_SWITCH_P), self.prev)
            mid = 0.5 * (mu0 + mu1)
            down = 1
        for i in range(n_commit):
            center, env, _ = self.vit[i]
            state = int(path[i])
            on = state == down
            if on:
                self._add_spec(center)
            self.trace.append((center, x[i], mid, nd, on, mu0, mu1))
            self.prev = state
            rec = self._step(center, env, on, total, now)
            if rec is not None:
                out.append(rec)
        del self.vit[:n_commit]
        return out

    def _features(self):
        """Level of each queued frame in dB, key down high: the comb level, or comb minus carrier."""
        if FM_FEATURE == "diff":
            return [10.0 * math.log10(env + 1e-30) - 10.0 * math.log10(aux + 1e-30)
                    for _, env, aux in self.vit]
        return [10.0 * math.log10(env + 1e-30) for _, env, _ in self.vit]

    def _frame(self, Pi, center):
        super()._frame(Pi, center)
        k0 = int(np.argmin(np.abs(self.freqs - self.fc)))
        lo, hi = max(0, k0 - self.spec_half), min(self.nfft, k0 + self.spec_half + 1)
        self.spec_q[center] = (lo, Pi[lo:hi].astype(np.float64))

    def _add_spec(self, center):
        """Add a key-down frame's spectrum slice to the transmission's key-down average."""
        e = self.spec_q.pop(center, None)
        if e is None:
            return
        lo, sl = e
        if self.tx_spec is None:
            self.tx_spec = np.zeros(self.nfft)
        self.tx_spec[lo:lo + sl.size] += sl
        self.tx_spec_n += 1

    def _carrier_fine(self):
        if self.tx_spec is None or self.tx_spec_n < 20:
            return self.fc
        fine = fm_fine_carrier(self.tx_spec / self.tx_spec_n, self.fc, self.freqs)
        return self.fc if fine is None else fine

    def _set_bands(self):
        idx, wts = [], []
        for k, w in zip(FM_KS, FM_W):
            b = np.flatnonzero(np.abs(self.freqs - (self.fc + FM_TONE_HZ * k)) <= FM_SIDE_HALF_HZ)
            idx.append(b)
            wts.append(np.full(b.size, w))
        self.band = np.concatenate(idx)
        self.band_w = np.concatenate(wts)
        self.cband = np.flatnonzero(np.abs(self.freqs - self.fc) <= FM_SIDE_HALF_HZ)

    def _power(self, Pi):
        return float(np.dot(Pi[self.band], self.band_w)) / self.wnorm

    def _aux(self, Pi):
        return float(np.sum(Pi[self.cband])) / self.wnorm


FM_PEAK_SEARCH_HZ = 120.0                        # a sideband peak is searched this far either side of fc +/- 400 Hz x k
FM_SPEC_HALF_HZ = 3300.0                         # spectrum kept per frame, this far either side of the carrier


def fm_fine_carrier(P, fc, freqs):
    """Carrier offset from tuning, Hz, from the sideband lines of a key-down average spectrum P. The
    lines sit at fc +/- 400 Hz x k, so the midpoint of each symmetric pair estimates fc whatever the tone
    frequency. Each peak is refined by a parabola in dB, and the pair midpoints are averaged with weight
    of the weaker line's power. Returns None if no pair was found."""
    df = float(freqs[1] - freqs[0])
    est, wts = [], []
    for k in FM_KS:
        if k <= 0:
            continue
        found = []
        for sgn in (1, -1):
            c = fc + sgn * FM_TONE_HZ * k
            idx = np.flatnonzero(np.abs(freqs - c) <= FM_PEAK_SEARCH_HZ)
            if idx.size < 3:
                break
            j = int(idx[int(np.argmax(P[idx]))])
            if j <= 0 or j >= P.size - 1:
                break
            y0, y1, y2 = 10.0 * np.log10(P[j - 1:j + 2] + 1e-30)
            den = float(y0 - 2.0 * y1 + y2)
            if den >= 0.0:
                break
            found.append((float(freqs[j]) + 0.5 * float(y0 - y2) / den * df, float(P[j])))
        if len(found) == 2:
            est.append(0.5 * (found[0][0] + found[1][0]))
            wts.append(min(found[0][1], found[1][1]))
    if not est:
        return None
    return float(np.average(est, weights=wts))


FM_PLOT_PERIOD_S = 0.2                           # redraw interval of the FM level plot


class FmLevelPlot(QtWidgets.QWidget):
    """Window with the FM comb level against time for the last CW_TRACE_S seconds, the key threshold
    with its hysteresis band, the key-up level, and a green mark at the bottom where key is down."""

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.setWindowTitle("FM level")
        self.t_last = 0.0
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time relative to newest sample", units="s")
        self.plot.setLabel("left", "Comb level", units="dB (uncalibrated)")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.legend = self.plot.addLegend(brush=pg.mkBrush(0, 0, 0, 191))   # 75% black behind the legend
        self.legend.setZValue(10)                  # drawn over the trace lines
        self.lev = self.plot.plot(pen=pg.mkPen("c", width=1), name="comb level")
        self.key = self.plot.plot(pen=pg.mkPen("g", width=3), name="key down")
        self.thr = self.plot.plot(pen=pg.mkPen("y", width=1), name="midpoint")
        self.thr_hi = self.plot.plot(pen=pg.mkPen("g", width=1, style=QtCore.Qt.PenStyle.DashLine),
                                     name="key-down mean")
        self.thr_lo = self.plot.plot(pen=pg.mkPen((160, 160, 160), width=1,
                                                  style=QtCore.Qt.PenStyle.DashLine), name="key-up mean")
        self.noise = self.plot.plot(pen=pg.mkPen((160, 160, 160), width=1,
                                                 style=QtCore.Qt.PenStyle.DotLine), name="noise floor")
        self.marks = []                               # (absolute sample, TextItem) for decoded characters
        self.btn_save = QtWidgets.QPushButton("Save PNG")
        self.btn_save.clicked.connect(self.save_png)
        self.status = QtWidgets.QLabel("")
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.status, 1)
        bar.addWidget(self.btn_save)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.plot, 1)
        lay.addLayout(bar)
        self.resize(900, 380)

    def update_from(self, dec, total):
        """Redraw from the decoder's trace, at most every FM_PLOT_PERIOD_S."""
        now = time.monotonic()
        if now - self.t_last < FM_PLOT_PERIOD_S or len(dec.trace) < 2:
            return
        self.t_last = now
        a = np.asarray(dec.trace, dtype=np.float64)
        t = (a[:, 0] - total) / dec.rate
        lev, mid, noise, on = a[:, 1], a[:, 2], a[:, 3], a[:, 4] > 0.5
        mu0, mu1 = a[:, 5], a[:, 6]
        hi, lo = float(np.max(lev)), float(np.min(lev))
        self.lev.setData(t, lev)
        # key-down is a short horizontal mark just above the bottom edge; nothing is drawn when key is up
        self.key.setData(t, np.where(on, lo - 1.9, np.nan), connect="finite")
        self.thr.setData(t, mid)
        self.thr_hi.setData(t, mu1)
        self.thr_lo.setData(t, mu0)
        self.noise.setData(t, noise)
        self.plot.setXRange(-CW_TRACE_S, 0.0, padding=0.0)
        self.plot.setYRange(lo - 2.0, hi + 2.0, padding=0.0)
        y_txt = lo - 1.5                              # just above the key-down marks
        kept = []
        for ms, item, tx in self.marks:
            x = (ms - total) / dec.rate
            if x < -CW_TRACE_S - 0.5:
                self.plot.removeItem(item)
            else:
                item.setPos(x, y_txt)
                kept.append((ms, item, tx))
        self.marks = kept
        self.status.setText(f"key down {100.0 * np.mean(on):.0f}% of last {-t[0]:.0f} s, "
                            f"midpoint {mid[-1]:.1f} dB, key-up {mu0[-1]:.1f} dB, "
                            f"key-down {mu1[-1]:.1f} dB, noise floor {noise[-1]:.1f} dB")

    def _add_mark(self, ms, ch, tx):
        import html as _html
        item = pg.TextItem(html=f'<span style="color:#7f7;font-size:12pt;font-weight:bold">'
                                f'{_html.escape(ch)}</span>', anchor=(0.5, 1.0))
        self.plot.addItem(item)
        self.marks.append((ms, item, tx))

    def add_char(self, ev):
        """A character of a transmission still in progress, at its sample position."""
        self._add_mark(ev["sample"], ev["ch"], ev["tx"])

    def remove_tx(self, tx):
        """Remove the characters of one transmission, when it is rejected or replaced."""
        keep = []
        for ms, item, t in self.marks:
            if t == tx:
                self.plot.removeItem(item)
            else:
                keep.append((ms, item, t))
        self.marks = keep

    def add_decode(self, rec, rate):
        """Show the final characters of a decoded transmission in place of its streamed ones."""
        self.remove_tx(rec["tx"])
        for ch, t0, t1 in rec.get("char_t", []):
            self._add_mark(rec["t_first"] + 0.5 * (t0 + t1) * rate, ch, rec["tx"])

    def save_png(self):
        import pyqtgraph.exporters
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.abspath(f"fm_level_{stamp}.png")
        exp = pyqtgraph.exporters.ImageExporter(self.plot.plotItem)
        exp.parameters()["width"] = 1400
        exp.export(path)
        self.status.setText(f"Saved {os.path.basename(path)}")


# ------------------------------------------------------------ WWV timing

def wwv_ms(v):
    return "--" if v is None else f"{v:+.1f} ms"


def wwv_tones(rec):
    return "+".join(str(t) for t in rec["tones"]) or "--"


def wwv_off_ms(w):
    """Offset of a time from the nearest whole UTC second, in ms. Positive: this clock is ahead."""
    return (w - round(w)) * 1000.0


def wwv_line(rec, mhz):
    """One decode-pane line for a completed UTC minute."""
    stamp = time.strftime("%H:%M", time.gmtime(rec["wall"]))
    car = rec["carrier"]
    cs = ("carrier --" if car is None else
          f"carrier {car['hz']:+.2f} Hz ({car['ppm']:+.3f} ppm), SNR {car['snr']:+.1f} dB")
    tone = "--" if rec["tone_db"] is None else f"{rec['tone_db']:.1f} dB"
    return (f"{stamp} UTC  WWV {mhz:.3f} MHz  clock {wwv_ms(rec['clock_ms'])} ({rec['n']} marks)  "
            f"marker {wwv_ms(rec['marker_ms'])}  {cs}  tone peak/median {tone}  "
            f"tones {wwv_tones(rec)}  time code {rec.get('tc_time') or '--'}  "
            f"clock offset {wwv_ms(rec.get('clock_full_ms'))}  "
            f"tc {rec.get('tc_bits') or '--'} conf {rec.get('tc_conf') or '--'} "
            f"[{rec.get('tc_reason') or ''}]")


class WwvMonitor:
    """WWV timing. Finds the time-signal tones in the AM envelope of the tuned carrier, gives each
    one's offset from the nearest whole UTC second, and once a minute reports the carrier's offset
    from the nominal frequency and its strength. Reads the ring buffer directly, so the display
    span does not matter."""

    def __init__(self, ring, rate, nominal_hz, log_path=None, minute_log_path=None):
        self.ring = ring
        self.rate = rate
        self.dec = int(round(rate / WWV_DEC_RATE))
        self.fs = rate / self.dec
        self.sos_lp = ssig.butter(4, WWV_LP_HZ, btype="low", fs=rate, output="sos")
        self.templates = {(f, kind): self._template(f, dur)
                          for f in WWV_TONES_HZ
                          for kind, dur in (("tick", WWV_TICK_LEN_S), ("marker", WWV_MARKER_LEN_S))}
        self.short_templates = {f: self._template(f, WWV_SHORT_S) for f in WWV_TONES_HZ}
        self.nominal_hz = nominal_hz
        self.file = None
        self.log = None
        if log_path:
            self.start_onset_log(log_path)
        self.minute_file = open(minute_log_path, "w", newline="") if minute_log_path else None
        self.minute_log = csv.writer(self.minute_file) if self.minute_file else None
        if self.minute_log is not None:
            self.minute_log.writerow(["utc_minute", "wall_s", "marks", "markers", "clock_ms",
                                      "tick_ms", "marker_ms", "frame100_ms", "frame100_contrast",
                                      "carrier_hz", "carrier_ppm", "carrier_snr_db", "tone_db",
                                      "tones", "nominal_hz", "marker_1k_n", "marker_cov",
                                      "marker_quiet", "marker_db",
                                      "tc_time", "tc_ok", "clock_full_ms", "tc_bits", "tc_conf", "tc_reason"])
        self.reset()

    def start_onset_log(self, path):
        """Write every accepted and rejected onset to a CSV file (debugging aid)."""
        self.stop_onset_log()
        self.file = open(path, "w", newline="")
        self.log = csv.writer(self.file)
        self.log.writerow(["wall_s", "tone_hz", "kind", "ratio_db", "phase_ms",
                           "sec_of_minute", "nominal_hz"])

    def stop_onset_log(self):
        if self.file is not None:
            self.file.close()
        self.file = None
        self.log = None

    def close(self):
        self.stop_onset_log()
        if self.minute_file is not None:
            self.minute_file.close()
            self.minute_file = None
            self.minute_log = None

    def reset(self):
        """Forget streaming state and pending events (on start and after any retune)."""
        self.pos = None                          # next ring sample to read
        self.primed = False                      # filter state set from the first samples
        self.zi_re = np.zeros((self.sos_lp.shape[0], 2))
        self.zi_im = np.zeros((self.sos_lp.shape[0], 2))
        self.env = np.zeros(0)                   # recent envelope, decimated to self.fs
        self.hist = np.zeros(0)                  # longer envelope history for the 100 Hz analysis
        self.env_end = 0                         # absolute decimated index just after env[-1]
        self.next_seg = 0                        # analyse when env_end reaches this
        self.done_keys = set()                   # UTC seconds already decided
        self.events = []                         # {"w", "tone", "kind"} not yet reported
        self.recent = deque(maxlen=600)          # same events, kept for WWV captures
        self.minute = None                       # UTC minute being collected
        self.tone_db = None                      # strongest tone-to-median ratio this minute, dB
        self.pending = {}                        # UTC second -> candidates, decided once the window passes it
        self.recent_markers = deque(maxlen=20)   # recently accepted markers, for plateau removal
        self.marker_diag = []                    # every 1 kHz marker candidate seen, before the gates
        self.miss_saves = deque(maxlen=200)      # wall times of automatic miss captures
        self.tc_hist = deque(maxlen=TC_HIST_MIN) # time-code frames for the joint decode

    def set_nominal(self, hz):
        self.nominal_hz = hz
        self.reset()

    def _wall(self, d, total, now):
        """Wall-clock time of decimated index d, from the ring position at time now."""
        return now - (total - d * self.dec) / self.rate + WWV_MAP_LAG_S

    def process(self, now=None):
        """Read new samples, analyse once a second, and return the minutes completed since the last call.
        now: wall-clock time of the newest sample (default: the system clock; the self-test passes its own)."""
        ring, dec = self.ring, self.dec
        total = ring.total
        now = time.time() if now is None else now
        if total < self.rate // 2:
            return []
        if self.minute is None:
            self.minute = int(now // 60)
        if self.pos is None or total - self.pos > ring.size:
            self.pos = total
        start = self.pos
        if total > start:
            x = ring.read(start, total - start)
            if x is None:
                self.pos = None
                return []
            self.pos = total
            if not self.primed:                 # start from steady state, not from zero
                self.zi_re = ssig.sosfilt_zi(self.sos_lp) * float(x.real[0])
                self.zi_im = ssig.sosfilt_zi(self.sos_lp) * float(x.imag[0])
                self.primed = True
            yr, self.zi_re = ssig.sosfilt(self.sos_lp, x.real.astype(np.float64), zi=self.zi_re)
            yi, self.zi_im = ssig.sosfilt(self.sos_lp, x.imag.astype(np.float64), zi=self.zi_im)
            off = (-start) % dec
            new = np.hypot(yr, yi)[off::dec]
            self.env = np.concatenate([self.env, new])[-int((WWV_SEG_S + 1.0) * self.fs):]
            self.hist = np.concatenate([self.hist, new])[-int(WWV_HIST_S * self.fs):]
            self.env_end = (start + off) // dec + len(new)
        seg_n = int(WWV_SEG_S * self.fs)
        if len(self.env) >= seg_n and self.env_end >= self.next_seg:
            self.next_seg = self.env_end + int(self.fs)
            self._analyse(total, now)
        # newest UTC minute that has ended WWV_SETTLE_S ago; if the loop stalled and fell behind,
        # jump to it so the row label always names the minute that just ended
        newest_done = int((now - WWV_SETTLE_S) // 60) - 1
        if self.minute < newest_done:
            self.minute = newest_done
        if now >= (self.minute + 1) * 60 + WWV_SETTLE_S:
            rec = self._finish_minute(self.minute, total, now)
            self._log_minute(rec)
            if rec["marker_n"] == 0:
                self._save_marker_miss(rec, total, now)
            self.minute += 1
            return [rec]
        return []

    def _template(self, f, dur):
        """Complex tone burst of length dur at f: the matched filter for a tick or a marker."""
        L = int(round(dur * self.fs)) | 1               # odd length, so the kernel has a centre sample
        t = (np.arange(L) - (L - 1) / 2) / self.fs
        return np.exp(-2j * np.pi * f * t)

    def _analyse(self, total, now):
        """Matched-filter search for each tone's 5 ms tick and 800 ms marker template. A candidate
        must reach a fraction of the strongest tick in the window, a tick inside a marker is that
        marker's plateau, and each UTC second keeps only its strongest candidate."""
        fs = self.fs
        seg_n = int(WWV_SEG_S * fs)
        seg = self.env[-seg_n:]
        seg_start = self.env_end - seg_n          # absolute decimated index of seg[0]
        x = seg - float(np.mean(seg))             # the carrier's steady envelope is not a signal
        edge = int(WWV_EDGE_S * fs)
        trusted = seg_n - edge                    # peaks must lie inside this range

        # pass 1: matched outputs, ticks first so the reference amplitude is known
        found, ref = [], 0.0
        for kind in ("tick", "marker"):
            for (f, k), h in self.templates.items():
                if k != kind:
                    continue
                m = np.abs(ssig.fftconvolve(x, h, mode="same"))
                med = float(np.median(m)) + 1e-12
                db = 20.0 * math.log10((float(m.max()) + 1e-12) / med)
                self.tone_db = db if self.tone_db is None else max(self.tone_db, db)
                if kind == "tick":
                    ref = max(ref, float(m.max()) / len(h))
                pk, _ = ssig.find_peaks(m, height=med * 10.0 ** (WWV_MF_DETECT_DB / 20.0),
                                        distance=int(0.5 * fs))
                found.append((f, kind, h, m, pk))

        # local 1 kHz level every 20 ms, against the window's noise: a true marker carries the tone
        # over all 800 ms and is quiet just before its onset
        short, thr_s = {}, {}
        for f, hs in self.short_templates.items():
            s = np.abs(ssig.fftconvolve(x, hs, mode="same")) / len(hs)
            short[f] = s
            thr_s[f] = float(np.median(s)) * 10.0 ** (WWV_MF_DETECT_DB / 20.0) + 1e-12

        # pass 2: candidates
        cands = []
        for f, kind, h, m, pk in found:
            half = (len(h) - 1) // 2
            for p in pk:
                p = int(p)
                if p >= trusted or p - half < edge or p + half >= seg_n:
                    continue                      # not fully inside the trusted part of the window
                amp = float(m[p]) / len(h)        # same units for every template
                if kind == "marker":
                    s, t = short[f], thr_s[f]
                    cov = float(np.mean(s[p - half:p + half + 1] > t))
                    onset = p - half                  # marker onset: the quiet span lies before it
                    a = onset - int(WWV_QUIET_SPAN_S[0] * fs)
                    b = onset - int(WWV_QUIET_SPAN_S[1] * fs)
                    quiet = float(np.mean(s[a:b] > t)) if a >= 0 else 0.0
                    if f == 1000.0:                   # diagnostics: every candidate, accepted or not
                        self.marker_diag.append({
                            "w": self._wall(seg_start + p - half, total, now),
                            "cov": cov, "quiet": quiet,
                            "db": 20.0 * math.log10(float(m[p]) / np.median(m) + 1e-12)})
                    if cov < WWV_MARKER_COVER or quiet > WWV_QUIET_COVER:
                        continue                  # partial overlap with a tick train, or tone before onset
                elif amp < WWV_TICK_MIN_FRAC * ref:
                    continue                      # leakage from a stronger tick
                w = self._wall(seg_start + p - half, total, now)
                if kind == "tick" and int(round(w)) % 60 in WWV_NO_TICK_SEC:
                    continue                      # WWV sends no tick at seconds 29 and 59
                cands.append({"w": w, "key": int(round(w)), "tone": f, "kind": kind,
                              "db": 20.0 * math.log10(float(m[p]) / np.median(m) + 1e-12),
                              "amp": amp})

        # candidates wait in self.pending until their second is well inside the trusted window:
        # a marker that completes later in the window must still compete with that second's ticks
        for c in cands:
            lst = self.pending.setdefault(c["key"], [])
            if not any(o["tone"] == c["tone"] and o["kind"] == c["kind"] and abs(o["w"] - c["w"]) < 0.05
                       for o in lst):
                lst.append(c)
        t_ok = self._wall(seg_start + trusted, total, now)
        newest = None
        for key in sorted(k for k in self.pending if k + WWV_DECIDE_MARGIN_S <= t_ok):
            group = self.pending.pop(key)
            if key in self.done_keys:
                continue
            self.done_keys.add(key)
            newest = key
            # a tick inside an accepted marker's span (same tone) is that marker's plateau
            marks = [c for c in group if c["kind"] == "marker"] + list(self.recent_markers)
            group = [c for c in group if c["kind"] == "marker" or not any(
                mk["tone"] == c["tone"] and mk["w"] - 0.02 <= c["w"] <= mk["w"] + WWV_MARKER_LEN_S + 0.02
                for mk in marks)]
            # an accepted marker (800 cycles) outranks any tick (5 cycles); among equals, strongest wins
            group.sort(key=lambda c: (c["kind"] == "marker", c["amp"]), reverse=True)
            for i, c in enumerate(group):
                if self.log is not None:
                    self.log.writerow([f"{c['w']:.4f}", int(c["tone"]),
                                       c["kind"] if i == 0 else "rejected",
                                       f"{c['db']:.2f}", f"{wwv_off_ms(c['w']):+.2f}",
                                       key % 60, int(self.nominal_hz)])
                if i == 0:
                    ev = {"w": c["w"], "tone": c["tone"], "kind": c["kind"]}
                    self.events.append(ev)
                    self.recent.append(ev)
                    if c["kind"] == "marker":
                        self.recent_markers.append({"w": c["w"], "tone": c["tone"]})
        if newest is not None:
            self.done_keys = {k for k in self.done_keys if k >= newest - 60}
        for k in [k for k in self.pending if k < t_ok - 120]:
            del self.pending[k]
        if self.file is not None:
            self.file.flush()

    def _frame100(self, M, total, now):
        """Where the 100 Hz time-code edge falls within the second (ms, positive: this clock ahead),
        folded over UTC minute M. Zero-phase filtering, so the edge is not delayed."""
        n = len(self.hist)
        if n < 10:
            return None, None
        t_last = self._wall(self.env_end - 1, total, now)     # wall time of hist[-1]
        t0, t1 = M * 60 - 1.0, M * 60 + 61.0                  # one second of margin each side
        j0 = n - 1 - int(round((t_last - t0) * self.fs))
        j1 = n - 1 - int(round((t_last - t1) * self.fs))
        if j0 < 0 or j1 > n - 1 or j1 - j0 < int(60 * self.fs):
            return None, None
        seg = self.hist[j0:j1 + 1]
        tt = (t_last - M * 60) - (n - 1 - np.arange(j0, j1 + 1)) / self.fs
        sos = ssig.butter(4, list(WWV_FRAME_BAND_HZ), btype="bandpass", fs=self.fs, output="sos")
        bp = ssig.sosfiltfilt(sos, seg - float(np.mean(seg)))
        e = np.abs(ssig.hilbert(bp))
        b = np.floor(np.mod(tt, 1.0) * 1000.0).astype(int) % 1000
        cnt = np.bincount(b, minlength=1000)
        F = np.bincount(b, weights=e, minlength=1000) / np.maximum(cnt, 1)
        # rising edge: mean over the next 100 ms minus mean over the previous 100 ms
        cs = np.concatenate([F, F, F]).cumsum()
        phi = np.arange(1000)
        nxt = (cs[phi + 1000 + 100 - 1] - cs[phi + 1000 - 1]) / 100.0
        prv = (cs[phi + 1000 - 1] - cs[phi + 1000 - 101]) / 100.0
        score = nxt - prv
        k = int(np.argmax(score))
        contrast = float(score[k] / (F.std() + 1e-12))
        ms = (k - 1000 if k > 500 else k) - WWV_FRAME_EDGE_MS
        ms = ms + 1000 if ms <= -500 else (ms - 1000 if ms > 500 else ms)
        return float(ms), contrast

    def _log_minute(self, rec):
        if self.minute_log is None:
            return
        def f(v, p=1):
            return "" if v is None else f"{v:+.{p}f}"
        car = rec["carrier"]
        self.minute_log.writerow([
            time.strftime("%Y-%m-%d %H:%M", time.gmtime(rec["wall"])), rec["wall"], rec["n"],
            rec["marker_n"], f(rec["clock_ms"]), f(rec["tick_ms"]), f(rec["marker_ms"]),
            f(rec["frame100_ms"]), "" if rec["frame100_contrast"] is None else f"{rec['frame100_contrast']:.2f}",
            "" if car is None else f"{car['hz']:+.2f}", "" if car is None else f"{car['ppm']:+.3f}",
            "" if car is None else f"{car['snr']:+.1f}",
            "" if rec["tone_db"] is None else f"{rec['tone_db']:.1f}",
            wwv_tones(rec), int(self.nominal_hz), rec["marker_1k_n"],
            _opt(rec["marker_cov"], 2), _opt(rec["marker_quiet"], 2), _opt(rec["marker_db"], 1),
            rec.get("tc_time") or "", "" if rec.get("tc_ok") is None else int(rec["tc_ok"]),
            f(rec.get("clock_full_ms")), rec.get("tc_bits") or "", rec.get("tc_conf") or "",
            rec.get("tc_reason") or ""])
        self.minute_file.flush()

    def _save_marker_miss(self, rec, total, now):
        """Save a 5 s envelope WAV and JSON sidecar around a minute with no accepted 1 kHz marker,
        at most WWV_MISS_SAVES_PER_HOUR per hour, so the detector can be checked against it."""
        while self.miss_saves and self.miss_saves[0] < now - 3600.0:
            self.miss_saves.popleft()
        if len(self.miss_saves) >= WWV_MISS_SAVES_PER_HOUR:
            return None
        M = int(rec["wall"] // 60)
        n = len(self.hist)
        t_last = self._wall(self.env_end - 1, total, now)     # wall time of hist[-1]
        t0, t1 = M * 60 - WWV_MISS_SPAN_S[0], M * 60 + WWV_MISS_SPAN_S[1]
        j0 = n - 1 - int(round((t_last - t0) * self.fs))
        j1 = n - 1 - int(round((t_last - t1) * self.fs))
        if n < 10 or j0 < 0 or j1 > n - 1 or j1 <= j0:
            return None
        seg = self.hist[j0:j1 + 1]
        peak = float(np.max(np.abs(seg))) or 1.0
        pcm = np.clip(seg / peak * 0.9 * 32767, -32767, 32767).astype("<i2")
        base = os.path.join(os.getcwd(), f"wwv_miss_{time.strftime('%Y%m%d_%H%M', time.gmtime(M * 60))}")
        wav_path, json_path = base + "_env8k.wav", base + "_info.json"
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.fs)
            w.writeframes(pcm.tobytes())
        info = {
            "minute_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(M * 60)),
            "window_start_utc": datetime.fromtimestamp(t0, timezone.utc).isoformat(timespec="milliseconds"),
            "window_span_s": [-WWV_MISS_SPAN_S[0], WWV_MISS_SPAN_S[1]],
            "sample_rate_hz": self.fs,
            "clock_offset_ms_minute": rec["clock_ms"],
            "marker_1k_candidates": rec["marker_1k_n"],
            "best_candidate": {"cov": rec["marker_cov"], "quiet": rec["marker_quiet"],
                               "db": rec["marker_db"]},
            "candidates": [{"wall_s": round(d["w"], 4), "cov": round(d["cov"], 3),
                            "quiet": round(d["quiet"], 3), "db": round(d["db"], 2)}
                           for d in rec["diag"]],
            "note": ("Envelope (3.5 kHz lowpass, 8 kHz rate, peak-normalised), not raw audio. "
                     "Saved because no 1 kHz marker was accepted in this minute."),
        }
        with open(json_path, "w") as f:
            json.dump(info, f, indent=2)
        self.miss_saves.append(now)
        return wav_path, json_path

    def _timecode(self, M, total, now, frame100_ms):
        """Decode the 100 Hz time-code frame that starts in UTC minute M and solve jointly with the
        frames of the previous minutes. The frame ends one second into minute M+1, so it is complete
        when minute M is reported. Always returns a dict as tc_solve does."""
        def fail(reason):
            return {"time": None, "ok": False, "clock_ms": None, "bits": "", "conf": "",
                    "reason": reason}

        if frame100_ms is None:
            return fail("no 100 Hz edge measured")
        T0 = M * 60 - 1.0                        # frame start considered (one second before the minute)
        # _wall() adds WWV_MAP_LAG_S for the tone markers; the time code must use true PC time,
        # or every frame is read one second late and the full clock gains a whole second
        t_last = self._wall(self.env_end - 1, total, now) - WWV_MAP_LAG_S
        t_abs, env = tc_window(self.hist, t_last, T0 - 1.0, T0 + 63.0, self.fs)
        if t_abs is None:
            return fail("envelope history too short")
        delta = frame100_ms / 1000.0
        k0 = int(math.ceil(T0))
        prof = tc_profiles(t_abs, env, delta, k0, 62)
        if prof is None:
            return fail("window does not cover the frame")
        self.tc_hist.append((M, k0, delta, prof))
        return tc_solve(list(self.tc_hist), TC_SEARCH_MIN)

    def _carrier(self, total):
        """Carrier offset from the nominal frequency (Hz, ppm) and its SNR in a 4 Hz band."""
        n = int(WWV_CARRIER_S * self.rate)
        if total < n:
            return None
        x = self.ring.read(total - n, n)
        if x is None:
            return None
        P = np.abs(sfft.fft(x * np.hanning(n).astype(np.float32), workers=-1)) ** 2
        f = sfft.fftfreq(n, 1.0 / self.rate)
        idx = np.flatnonzero((np.abs(f) <= WWV_CARRIER_SEARCH_HZ) & (np.abs(f) >= 0.5))
        k = idx[int(np.argmax(P[idx]))]
        fpk = float(f[k])
        dif = np.abs(f - fpk)
        sig = dif <= 2.0
        noise = (dif >= 2000.0) & (dif <= 2800.0)
        p_sig = float(P[sig].sum())
        p_noise = float(np.median(P[noise])) / math.log(2.0)   # mean power per bin of noise
        snr = 10.0 * math.log10(p_sig / (int(sig.sum()) * p_noise + 1e-30) + 1e-30)
        return {"hz": fpk, "ppm": fpk / self.nominal_hz * 1e6, "snr": snr}

    def _finish_minute(self, M, total, now):
        """Summarise the onsets of UTC minute M (events are assigned to the nearest second)."""
        mine = [e for e in self.events if int(round(e["w"])) // 60 == M]
        self.events = [e for e in self.events if int(round(e["w"])) // 60 > M]
        ms = [wwv_off_ms(e["w"]) for e in mine]
        ticks = [wwv_off_ms(e["w"]) for e in mine if e["kind"] == "tick"]
        marks = [wwv_off_ms(e["w"]) for e in mine if e["kind"] == "marker"]
        rec = {
            "wall": M * 60,
            "n": len(mine),
            "clock_ms": float(np.median(ms)) if ms else None,
            "tick_ms": float(np.median(ticks)) if ticks else None,
            "marker_ms": float(np.median(marks)) if marks else None,
            "tones": sorted({int(round(e["tone"])) for e in mine}),
            "tone_db": self.tone_db,
            "carrier": self._carrier(total),
            "marker_n": len(marks),
        }
        diag = [d for d in self.marker_diag if int(round(d["w"])) // 60 == M]
        self.marker_diag = [d for d in self.marker_diag if int(round(d["w"])) // 60 > M]
        best = max(diag, key=lambda d: (d["cov"], d["db"])) if diag else None
        rec["diag"] = diag
        rec["marker_1k_n"] = len({int(round(d["w"])) for d in diag})   # distinct seconds with a candidate
        rec["marker_cov"] = best["cov"] if best else None
        rec["marker_quiet"] = best["quiet"] if best else None
        rec["marker_db"] = best["db"] if best else None
        rec["frame100_ms"], rec["frame100_contrast"] = self._frame100(M, total, now)
        tc = self._timecode(M, total, now, rec["frame100_ms"])
        rec["tc_time"] = tc["time"]
        rec["tc_ok"] = tc["ok"]
        rec["clock_full_ms"] = tc["clock_ms"]
        rec["tc_bits"] = tc["bits"]
        rec["tc_reason"] = tc["reason"]
        rec["tc_conf"] = tc["conf"]
        self.tone_db = None
        return rec


class Analyzer:
    def __init__(self, ring, rate, nfft, avg_s=1.0):
        self.ring = ring
        self.rate = rate
        self.nfft = nfft
        self.hop = nfft // 2
        self.win = np.hanning(nfft).astype(np.float32)
        self.max_frames = max(1, 2_000_000 // nfft)
        self.freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / rate))
        self.row_s = self.hop / rate                                 # seconds per waterfall row
        self.avg_s = avg_s
        self.alpha = min(1.0, self.row_s / max(avg_s, self.row_s))   # time-average weight
        self.avg_on = True      # time averaging requested (see set_avg)
        self.next_pos = None
        self.ema = None         # time-averaged power, per bin
        self.bg = None          # background level, dB, per bin
        self.gated_s = None     # seconds each bin has spent above the gate
        self.n_rows = 0         # rows since configure()
        self.raw = True        # True: absolute power; False: power above background (SNR)
        self.ema_long = None    # longer time average of power, for the averaged spectrum trace
        self.long_alpha = min(1.0, self.row_s / max(SPEC_AVG_S, self.row_s))
        self.spec_avg_db = None
        self.keep = None
        self.group = 1
        self.freq_khz = None
        self.peak_db = None     # peak hold of the displayed trace, per bin; cleared by configure()

        self.skip_tx = False      # Tx Test mode
        self.tx = None          # TxDetector, made by configure()

    def set_avg(self, on):
        """Time-average the waterfall over avg_s seconds, or show each row's own power."""
        self.avg_on = on
        self.alpha = min(1.0, self.row_s / max(self.avg_s, self.row_s)) if on else 1.0

    def set_nfft(self, nfft):
        """Change the FFT size in place. The bin grid and row timing change, so configure() must
        follow to rebuild the displayed bins and reset the averages."""
        self.nfft = nfft
        self.hop = nfft // 2
        self.win = np.hanning(nfft).astype(np.float32)
        self.max_frames = max(1, 2_000_000 // nfft)
        self.freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / self.rate))
        self.row_s = self.hop / self.rate
        self.long_alpha = min(1.0, self.row_s / max(SPEC_AVG_S, self.row_s))
        self.set_avg(self.avg_on)
        self.next_pos = None

    def restart_average(self):
        """Make the 15 s average equal the current live trace and restart averaging from it."""
        if self.ema is None:
            return
        self.ema_long = self.ema.copy()
        self.spec_avg_db = 10 * np.log10(self.ema_long + 1e-20) - (0.0 if self.raw else self.bg)

    def configure(self, tune_hz, lo_off, hi_off, settle_s=0.0):
        """Set the displayed bins (offsets from tuning, Hz) and reset averaging."""
        self.keep = (self.freqs >= lo_off) & (self.freqs <= hi_off)
        nb = int(self.keep.sum())
        self.group = max(1, int(np.ceil(nb / MAX_DISPLAY_COLS)))
        self.freq_khz = self.reduce((tune_hz + self.freqs[self.keep]) / 1e3)
        self.tx = TxDetector(self.freq_khz, self.row_s)
        self.ema = None
        self.bg = None
        self.gated_s = None
        self.n_rows = 0
        self.ema_long = None
        self.spec_avg_db = None
        self.peak_db = None
        if settle_s > 0:
            self.next_pos = self.ring.total + int(settle_s * self.rate)
        else:
            self.next_pos = None

    def reduce(self, p):
        """Average groups of bins along the last axis to limit display width."""
        g = self.group
        pad = (-p.shape[-1]) % g
        if pad:
            p = np.pad(p, [(0, 0)] * (p.ndim - 1) + [(0, pad)], mode="edge")
        return p.reshape(p.shape[:-1] + (-1, g)).mean(axis=-1)

    def process(self):
        """Return (rows_snr_db, spectrum_snr_db) for new data, or (None, None)."""
        ring, nfft, hop = self.ring, self.nfft, self.hop
        total = ring.total
        if total < nfft:
            return None, None
        if self.next_pos is None or total - self.next_pos > ring.size - nfft:
            self.next_pos = total - nfft
        avail = total - self.next_pos
        if avail < nfft:
            return None, None
        k = min((avail - nfft) // hop + 1, self.max_frames)
        seg = ring.read(self.next_pos, (k - 1) * hop + nfft)
        if seg is None:
            self.next_pos = None
            return None, None
        self.next_pos += k * hop

        frames = np.lib.stride_tricks.sliding_window_view(seg, nfft)[::hop]
        X = np.fft.fftshift(np.fft.fft(frames * self.win, axis=1), axes=1)[:, self.keep]
        p = (X.real * X.real + X.imag * X.imag).astype(np.float32)
        pr = self.reduce(p)                                  # (k, ncols) linear power

        if self.ema is None:
            self.ema = pr[0].copy()
        if self.ema_long is None:
            self.ema_long = self.ema.copy()
        out = np.empty_like(pr)
        bgs = np.empty_like(pr)
        for i in range(pr.shape[0]):
            self.ema += self.alpha * (pr[i] - self.ema)      # time average
            self.ema_long += self.long_alpha * (pr[i] - self.ema_long)
            x = 10 * np.log10(self.ema + 1e-20)
            if self.bg is None:
                self.bg = x.copy()
                self.gated_s = np.zeros_like(x)
            self.n_rows += 1
            # Bins well above the background are treated as signal and do not move it, so the
            # background is unchanged when a signal ends. A bin held above the gate long enough
            # is accepted as a genuine new background.
            above = x > self.bg + BG_GATE_DB
            self.gated_s = np.where(above, self.gated_s + self.row_s, 0.0)
            move = ~above | (self.gated_s > BG_GATE_MAX_S)
            w = max(self.row_s / BG_TAU_S, 1.0 / self.n_rows)   # running mean until BG_TAU_S is reached
            self.bg = np.where(move, self.bg + w * (x - self.bg), self.bg)
            out[i] = self.ema
            bgs[i] = self.bg

        if self.raw:
            rows_db = 10 * np.log10(out + 1e-20)
            spec_db = 10 * np.log10(self.ema + 1e-20)
        else:
            rows_db = 10 * np.log10(out + 1e-20) - bgs
            spec_db = 10 * np.log10(self.ema + 1e-20) - self.bg
        self.spec_avg_db = 10 * np.log10(self.ema_long + 1e-20) - (0.0 if self.raw else self.bg)
        n0 = self.n_rows - pr.shape[0]
        for i in range(pr.shape[0]):
            counted = self.tx.feed((n0 + i + 1) * self.row_s, out[i], rows_db[i])
            if not self.skip_tx:
                counted = [rows_db[i]]
            for row in counted:
                self.peak_db = row.copy() if self.peak_db is None else np.maximum(self.peak_db, row)
        return rows_db.astype(np.float32), spec_db

# ------------------------------------------------------------ FT8 decoding

def make_wav(z, rate, disp_off, span_hz, path, audio_len):
    """Demodulate the displayed band to 12 kHz USB-style audio and write a mono WAV of audio_len samples.

    z must hold exactly one slot of complex samples. The displayed band is centred
    on 2100 Hz audio, clipped to 3.8 kHz, so audio freq = 2100 + (RF offset - display centre).
    """
    n_in = z.size
    X = sfft.fft(z, workers=-1)
    m = min(span_hz, FT8_AUDIO_MAX_SPAN)
    a_lo = FT8_AUDIO_CENTER - m / 2
    a_hi = FT8_AUDIO_CENTER + m / 2
    M = audio_len
    b = np.arange(int(np.ceil(a_lo * M / FT8_AUDIO_RATE)),
                  int(np.floor(a_hi * M / FT8_AUDIO_RATE)) + 1)
    a = b * FT8_AUDIO_RATE / M
    off = disp_off + (a - FT8_AUDIO_CENTER)                 # RF offset from tuning, Hz
    k = np.round(off * n_in / rate).astype(np.int64) % n_in
    Y = np.zeros(M, dtype=np.complex64)
    Y[b] = X[k] * np.complex64(M / n_in)
    r = sfft.ifft(Y, workers=-1).real.astype(np.float64)
    peak = float(np.max(np.abs(r)))
    if peak <= 0:
        raise ValueError("no signal energy in decode band")
    pcm = (r / peak * 0.5 * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FT8_AUDIO_RATE)
        w.writeframes(pcm.tobytes())


def format_all_line(slot_time, dial_hz, snr, dt, audio_hz, msg, mode):
    """One ALL.TXT line in WSJT-X's layout (verify column widths against a real ALL.TXT)."""
    ts = time.strftime("%y%m%d_%H%M%S", time.gmtime(slot_time))
    return (f"{ts}  {dial_hz / 1e6:9.3f} Rx {mode}  {snr:4d}  {dt:4.1f} "
            f"{int(round(audio_hz)):4d} ~  {msg}")


GRID_RE = re.compile(r"^[A-R]{2}[0-9]{2}(?:[A-X]{2})?$")
CALL_TOKEN_RE = re.compile(r"^[A-Z0-9/]{3,}$")


def is_callsign(tok):
    return (CALL_TOKEN_RE.match(tok) is not None
            and any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)
            and GRID_RE.match(tok) is None)


def sending_call(msg):
    """Callsign of the station that sent a decoded message, or None (e.g. hashed calls)."""
    toks = msg.split()
    if len(toks) < 2:
        return None
    if toks[0] == "CQ":                                  # CQ [modifier] CALL [GRID]
        cands = [t for t in toks[1:] if is_callsign(t)]
        tok = cands[0] if cands else None
    else:                                                # TO FROM [REPORT|GRID|RR73|73]
        tok = toks[1]
    return tok if tok is not None and is_callsign(tok) else None


def sending_grid(msg):
    """Grid square sent by the station that sent a decoded message, or None."""
    toks = msg.split()
    if len(toks) < 3 or toks[-1] == "RR73" or GRID_RE.match(toks[-1]) is None:
        return None
    return toks[-1]


REPORT_RE = re.compile(r"^R?[+-]\d{1,2}$")


def report_to(msg):
    """Callsign a 'TO FROM REPORT' message is addressed to, if it carries a signal report, else None."""
    toks = msg.split()
    if len(toks) != 3 or toks[0] == "CQ" or REPORT_RE.match(toks[2]) is None:
        return None
    if not (is_callsign(toks[0]) and is_callsign(toks[1])):
        return None
    return toks[0]


EARTH_RADIUS_MI = 3958.8


def grid_center(grid):
    """(lat, lon) in degrees of the centre of a 4- or 6-character Maidenhead grid."""
    g = grid.upper()
    lon = (ord(g[0]) - 65) * 20 - 180 + int(g[2]) * 2
    lat = (ord(g[1]) - 65) * 10 - 90 + int(g[3])
    if len(g) == 6:
        lon += (ord(g[4]) - 65) * 5 / 60 + 2.5 / 60
        lat += (ord(g[5]) - 65) * 2.5 / 60 + 1.25 / 60
    else:
        lon += 1
        lat += 0.5
    return lat, lon


def grid_distance_miles(grid_a, grid_b):
    """Great-circle distance in miles between the centres of two grid squares."""
    lat1, lon1 = (math.radians(v) for v in grid_center(grid_a))
    lat2, lon2 = (math.radians(v) for v in grid_center(grid_b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_MI * math.asin(math.sqrt(h))


def run_decode(z, rate, slot_time, tune_hz, disp_off, span_hz, jt9_path, shm_dir, mode):
    """Write the slot WAV to shared memory, run jt9 in the given mode, and return decode records."""
    m = MODES[mode]
    stamp = time.strftime("%y%m%d_%H%M%S", time.gmtime(slot_time))
    path = os.path.join(shm_dir, stamp + ".wav")
    make_wav(z, rate, disp_off, span_hz, path, int(round(m["slot_s"] * FT8_AUDIO_RATE)))
    try:
        proc = subprocess.run([jt9_path] + m["jt9"] + [path],
                              capture_output=True, text=True, timeout=120)
        text = proc.stdout
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if proc.returncode != 0:
        first = (proc.stderr.strip().splitlines() or ["no error text"])[0]
        raise RuntimeError(f"jt9 exit {proc.returncode}: {first}")
    recs = []
    for ln in text.splitlines():
        m = DECODE_RE.match(ln)
        if not m:
            continue
        snr, dt, a_hz, raw = m.groups()
        # jt9 pads the message and may add a decoder tag ("a1", "? a1"); keep the tag on the
        # displayed line only, so the message itself is clean for parsing and highlighting
        ann_m = ANN_RE.match(raw.strip())
        msg = " ".join(ann_m.group(1).split())
        full = " ".join(filter(None, [msg, " ".join(ann_m.group(2).split())]))
        audio = disp_off + (float(a_hz) - FT8_AUDIO_CENTER)   # Hz relative to tuned frequency
        recs.append({
            "slot": slot_time,
            "mode": mode,
            "line": format_all_line(slot_time, tune_hz, int(snr), float(dt), audio, full, mode),
            "msg": msg,
            "call": sending_call(msg),
            "grid": sending_grid(msg),
            "snr": int(snr),
            "offset": int(round(audio)),                      # same audio offset as the waterfall axis
            "rf_hz": tune_hz + audio,                         # absolute RF of the tone start
        })
    return recs


class DecodeBridge(QtCore.QObject):
    """Carries decoder-thread results to the GUI thread."""
    decode = QtCore.Signal(object)
    status = QtCore.Signal(str)
    done = QtCore.Signal()
    count = QtCore.Signal(object)   # valid decodes in one slot: {"s0", "mode", "n"}

# ---- IQ recorder: triggered raw capture of the displayed span, written as a WAV file

IQ_NFFT = 4096                  # trigger statistic FFT: 47 Hz bins at 192 kS/s
IQ_AVG_S = 0.03                 # power is averaged over this before the statistic; short, so the stop is not delayed
IQ_WARM_S = 1.0                 # no triggering until the averages have settled
IQ_FLOOR_TAU_S = 10.0           # background level follows slow changes over about this time
IQ_ON_DB = 10.0                 # a peak this far above the background starts a recording
IQ_OFF_DB = 4.0                 # below this the span counts as background again
IQ_ON_MIN_S = 0.05              # the peak must stay above IQ_ON_DB this long
IQ_PRE_S = 0.5                  # written before the carrier first rises above the background
IQ_POST_S = 1.0                 # written after the span returns to background
IQ_MAX_S = 120.0                # longest single recording


def iq_utc(t, iso=False):
    d = datetime.fromtimestamp(t, timezone.utc)
    return d.isoformat(timespec="milliseconds") if iso else d.strftime("%H:%M:%S")


def wav_float_header(rate, data_bytes):
    """44-byte WAV header: 2 channels (I, Q), 32-bit IEEE float, interleaved."""
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + data_bytes, b"WAVE", b"fmt ",
                       16, 3, 2, rate, rate * 8, 8, 32, b"data", data_bytes)


class IqRecorder:
    """Triggered IQ capture of the displayed span, wherever the receiver is tuned. Every new block of
    samples goes into a pre-roll ring. Per FFT frame, the statistic is the strongest bin in the span,
    relative to the span's median, after averaging. A recording starts when the statistic rises
    IQ_ON_DB above a slowly tracked background, beginning IQ_PRE_S before the rise. It ends IQ_POST_S
    after the statistic falls back within IQ_OFF_DB of the background, or at IQ_MAX_S.
    Output: WAV (left = I, right = Q, receiver sample rate) and a JSON sidecar."""

    def __init__(self, ring, rate, outdir):
        self.ring = ring
        self.rate = rate
        self.outdir = outdir
        self.nfft = IQ_NFFT
        self.hop = IQ_NFFT // 2
        self.win = np.hanning(self.nfft).astype(np.float32)
        self.freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, 1.0 / rate))
        self.alpha = min(1.0, (self.hop / rate) / IQ_AVG_S)
        self.pre_n = int(IQ_PRE_S * rate)
        self.pre_size = self.pre_n + int(3.0 * rate) + self.nfft
        self.mask = None
        self.tune_hz = 0
        self.lo_off = self.hi_off = 0.0
        self.rec = None
        self.restart("recorder started")

    def retarget(self, tune_hz, lo_off, hi_off):
        """New tuning or displayed span: the statistic is taken over the new span from scratch."""
        self.tune_hz = int(tune_hz)
        self.lo_off, self.hi_off = lo_off, hi_off
        self.mask = (self.freqs >= lo_off) & (self.freqs <= hi_off)
        return self.restart("tuning or span changed")

    def close(self):
        return self.restart("recorder off")

    def restart(self, reason):
        """Forget the background and the pre-roll. A recording in progress is closed; returns its log lines."""
        lines = self._finish(reason) if self.rec is not None else []
        self.pos = None
        self.base = 0
        self.pre = RingBuffer(self.pre_size)
        self.ema = None
        self.s_floor = None
        self.n_floor = 0
        self.n_frames = 0
        self.on_start = None
        self.on_run = 0
        self.stop_abs = None
        self.rearm = False
        return lines

    def process(self, now=None):
        """Read new samples, update the statistic and the recording. Returns log lines."""
        ring, n, hop = self.ring, self.nfft, self.hop
        out = []
        if self.mask is None or not self.mask.any() or ring.total < n:
            return out
        now = time.time() if now is None else now
        total = ring.total
        if self.pos is None or total - self.pos > ring.size - n:
            out += self.restart("samples lost")
            self.pos = total - n
            self.base = self.pos
        avail = total - self.pos
        if avail < n:
            return out
        k = min((avail - n) // hop + 1, 200)
        seg = ring.read(self.pos, (k - 1) * hop + n)
        if seg is None:
            return out + self.restart("samples overwritten")
        base_abs = self.pos
        self.pos += k * hop
        self.pre.push(seg[:k * hop])
        frames = np.lib.stride_tricks.sliding_window_view(seg, n)[::hop] * self.win
        X = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)[:, self.mask]
        P = X.real * X.real + X.imag * X.imag
        for i in range(k):
            self._frame(P[i], base_abs + i * hop + n // 2, now, out)
        out += self._writes()
        return out

    def _frame(self, Pi, fc, now, out):
        """One FFT frame: update the averaged spectrum, the statistic and the start and stop decisions.
        fc is the frame centre as a ring sample index."""
        if self.ema is None:
            self.ema = Pi.astype(np.float64)
        else:
            self.ema += self.alpha * (Pi - self.ema)
        self.n_frames += 1
        e = 10.0 * np.log10(self.ema + 1e-30)
        S = float(e.max() - np.median(e))
        if self.s_floor is None:
            self.s_floor = S
        excess = S - self.s_floor
        if self.rec is not None:
            if excess < IQ_OFF_DB:
                if self.stop_abs is None:
                    self.stop_abs = fc + int(IQ_POST_S * self.rate)
            else:
                self.stop_abs = None
            return
        if self.n_frames * self.hop < IQ_WARM_S * self.rate:
            self._track_floor(S)
            return
        if self.rearm and excess < IQ_OFF_DB:
            self.rearm = False
        if self.rearm or excess <= IQ_ON_DB:
            self.on_start, self.on_run = None, 0
            if excess < IQ_OFF_DB:
                self._track_floor(S)
            return
        if self.on_start is None:
            self.on_start = fc
        self.on_run += self.hop
        if self.on_run >= IQ_ON_MIN_S * self.rate:
            self._open(self.on_start, now, out)
            self.on_start, self.on_run = None, 0

    def _track_floor(self, S):
        self.n_floor += 1
        a = max((self.hop / self.rate) / IQ_FLOOR_TAU_S, 1.0 / self.n_floor)
        self.s_floor += a * (S - self.s_floor)

    def _wall(self, idx, now):
        w = self.ring.wall_at(idx, self.rate)
        return w if w is not None else now - (self.pos - idx) / self.rate

    def _open(self, onset, now, out):
        """Start a recording with its pre-roll: IQ_PRE_S before the onset, as far as the buffer holds it."""
        oldest = self.base + max(0, self.pre.total - self.pre.size)
        start = max(onset - self.pre_n, oldest)
        t_start = self._wall(start, now)
        t_on = self._wall(onset, now)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(t_start))
        base = os.path.join(self.outdir, f"iq_{stamp}_{self.tune_hz / 1e3:.0f}kHz")
        path = base + ".wav"
        f = open(path, "wb")
        f.write(wav_float_header(self.rate, 0))
        self.rec = {"f": f, "path": path, "base": base, "wpos": start, "n": 0, "lost": 0,
                    "t_start": t_start, "t_onset": t_on}
        out.append(f"{iq_utc(t_start)} UTC  IQ recording started  {self.tune_hz / 1e6:.4f} MHz  "
                   f"carrier at {iq_utc(t_on)}  -> {os.path.basename(path)}")

    def _writes(self):
        """Write the samples due so far: up to the newest sample, or to the stop point. Returns log lines."""
        rec = self.rec
        if rec is None:
            return []
        max_n = int(IQ_MAX_S * self.rate)
        end = self.pos
        if self.stop_abs is not None:
            end = min(end, self.stop_abs)
        end = min(end, rec["wpos"] + (max_n - rec["n"]))
        lo = max(rec["wpos"] - self.base, self.pre.total - self.pre.size, 0)
        hi = end - self.base
        if hi > lo:
            rec["lost"] += lo - (rec["wpos"] - self.base)
            data = self.pre.read(lo, hi - lo)
            buf = np.empty((hi - lo, 2), dtype="<f4")
            buf[:, 0] = data.real
            buf[:, 1] = data.imag
            rec["f"].write(buf.tobytes())
            rec["n"] += hi - lo
            rec["wpos"] = self.base + hi
            db = rec["n"] * 8
            rec["f"].seek(4)
            rec["f"].write(struct.pack("<I", 36 + db))
            rec["f"].seek(40)
            rec["f"].write(struct.pack("<I", db))
            rec["f"].seek(0, 2)
        if rec["n"] >= max_n:
            self.rearm = True
            return self._finish("reached maximum length")
        if self.stop_abs is not None and rec["wpos"] >= self.stop_abs:
            return self._finish("span back at background")
        return []

    def _finish(self, reason):
        rec = self.rec
        self.rec = None
        self.stop_abs = None
        rec["f"].close()
        t_end = self._wall(rec["wpos"], time.time())
        info = {
            "file": os.path.basename(rec["path"]),
            "format": "WAV, 2 channels, 32-bit IEEE float; left = I, right = Q (complex baseband)",
            "sample_rate_hz": self.rate,
            "tune_hz": self.tune_hz,
            "window_lo_hz": self.tune_hz + self.lo_off,
            "window_hi_hz": self.tune_hz + self.hi_off,
            "start_utc": iq_utc(rec["t_start"], iso=True),
            "carrier_onset_utc": iq_utc(rec["t_onset"], iso=True),
            "end_utc": iq_utc(t_end, iso=True),
            "samples": rec["n"],
            "seconds": rec["n"] / self.rate,
            "samples_lost": rec["lost"],
            "ended_by": reason,
            "trigger": {"on_db": IQ_ON_DB, "off_db": IQ_OFF_DB, "on_min_s": IQ_ON_MIN_S,
                        "pre_roll_s": IQ_PRE_S, "post_s": IQ_POST_S, "max_s": IQ_MAX_S},
        }
        try:
            with open(rec["base"] + "_info.json", "w") as f:
                json.dump(info, f, indent=2)
        except OSError:
            pass
        lost = f", {rec['lost']} samples lost" if rec["lost"] else ""
        return [f"{iq_utc(t_end)} UTC  IQ recording stopped ({reason})  {rec['n'] / self.rate:.2f} s  "
                f"{os.path.basename(rec['path'])}{lost}"]


# ------------------------------------------------------------ GUI

class DecodeWindow(QtWidgets.QWidget):
    """Scrollable FT8 decodes: one pane, or two (CQ calls above other messages). CQ lines are highlighted."""

    CQ_BG = QtGui.QColor(31, 92, 46)
    CQ_FG = QtGui.QColor(240, 255, 240)

    def __init__(self, parent=None):
        super().__init__(parent)
        # The single pane holds everything; the two split panes get their own subsets.
        # Every decode is appended to the single pane and to one split pane, so switching
        # modes shows the full history.
        self.view = self._make_view()
        self.view.setPlaceholderText("FT8 decodes appear here and in ALL.TXT")
        self.view_cq = self._make_view()
        self.view_other = self._make_view()

        self.split_box = QtWidgets.QWidget()
        split_lay = QtWidgets.QVBoxLayout(self.split_box)
        split_lay.setContentsMargins(0, 0, 0, 0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        for title, view in (("CQ calls", self.view_cq), ("Other messages", self.view_other)):
            pane = QtWidgets.QWidget()
            pl = QtWidgets.QVBoxLayout(pane)
            pl.setContentsMargins(0, 0, 0, 0)
            pl.setSpacing(0)
            pl.addWidget(QtWidgets.QLabel(title))
            pl.addWidget(view)
            splitter.addWidget(pane)
        split_lay.addWidget(splitter)
        self.split_box.hide()

        self.btn_split = QtWidgets.QPushButton("Two panes")
        self.btn_split.setCheckable(True)
        self.btn_split.toggled.connect(self._set_split)
        self.btn_stats = QtWidgets.QPushButton("Stats")  # opens the per-callsign stats window (MainWindow)
        self.top_bar = QtWidgets.QWidget()           # hidden while popped out; the buttons move to that window
        top = QtWidgets.QHBoxLayout(self.top_bar)
        top.setContentsMargins(0, 0, 0, 0)
        top.addStretch(1)
        top.addWidget(self.btn_stats)
        top.addWidget(self.btn_split)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.top_bar)
        lay.addWidget(self.view)
        lay.addWidget(self.split_box)

    @staticmethod
    def _make_view():
        view = QtWidgets.QTextEdit()
        view.setReadOnly(True)
        view.document().setMaximumBlockCount(DECODE_HISTORY_LINES)
        view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        font = QtGui.QFont("monospace")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        view.setFont(font)
        return view

    def _set_split(self, on):
        self.view.setVisible(not on)
        self.split_box.setVisible(on)
        self.btn_split.setText("Single pane" if on else "Two panes")

    def append(self, line, is_cq):
        self._append_to(self.view, line, is_cq)
        self._append_to(self.view_cq if is_cq else self.view_other, line, is_cq)

    def _append_to(self, view, line, is_cq):
        sb = view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        cur = view.textCursor()
        cur.movePosition(QtGui.QTextCursor.MoveOperation.End)
        if cur.position() > 0:
            cur.setCharFormat(QtGui.QTextCharFormat())   # plain formatting for the new block
            cur.insertBlock()
        fmt = QtGui.QTextCharFormat()
        if is_cq:
            fmt.setBackground(self.CQ_BG)
            fmt.setForeground(self.CQ_FG)
            fmt.setFontWeight(QtGui.QFont.Weight.Bold)
        cur.insertText(line, fmt)
        if at_bottom:
            sb.setValue(sb.maximum())


class _SortItem(QtWidgets.QTableWidgetItem):
    """Table cell whose sort order comes from a separate key (number or text), not its display text."""

    def __init__(self, text, key):
        super().__init__(text)
        self.key = key

    def __lt__(self, other):
        return self.key < other.key


class StatsWindow(QtWidgets.QWidget):
    """Per-callsign FT8 packet statistics. Sortable by any column; refreshed while visible."""

    COLUMNS = ["Call", "First (UTC)", "Last (UTC)", "Total", "CQ", "2way",
               "Min dB", "Max dB", "Avg dB", "Grid", "Distance (mi)"]
    SUMMARY = ["Decodes", "Calls", "CQs", "2-way"]
    TITLE_FG = QtGui.QColor(190, 190, 190)     # header titles: bold, but not full white

    def __init__(self, source, clear_fn=None):
        super().__init__(None, QtCore.Qt.Window)
        self.source = source            # callable returning (stats dict, mode)
        self.clear_fn = clear_fn        # empties the owner's stats; the table is rebuilt after it
        self.setWindowTitle("FT8 stats")
        self.resize(860, 480)
        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        bold = self.table.horizontalHeader().font()
        bold.setBold(True)                          # bold titles set the column widths
        for i in range(len(self.COLUMNS)):
            self.table.horizontalHeaderItem(i).setFont(bold)
            self.table.horizontalHeaderItem(i).setForeground(QtGui.QBrush(self.TITLE_FG))
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setSortingEnabled(True)
        self.table.sortItems(3, QtCore.Qt.SortOrder.DescendingOrder)     # busiest senders first
        header = self.table.horizontalHeader()
        # The Windows native style ignores per-section background brushes, so the sorted column's
        # highlight disappeared (white title on a light header). Fusion paints it on every platform.
        self._header_style = QtWidgets.QStyleFactory.create("Fusion")
        header.setStyle(self._header_style)
        header.sortIndicatorChanged.connect(self._mark_sort_column)
        self._mark_sort_column(3, QtCore.Qt.SortOrder.DescendingOrder)
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.summary_vals = {}
        summary = QtWidgets.QHBoxLayout()
        for name in self.SUMMARY:
            val = QtWidgets.QLabel("0")
            summary.addWidget(QtWidgets.QLabel(name + ":"))
            summary.addWidget(val)
            summary.addSpacing(14)
            self.summary_vals[name] = val
        summary.addStretch(1)
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(self.clear)
        summary.addWidget(btn_clear)
        summary_box = QtWidgets.QWidget()
        summary_box.setLayout(summary)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(summary_box)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(1, 1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(splitter)
        self.dirty = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh_if_due)
        self.timer.start(1000)          # new decodes are batched into at most one rebuild per second

    def changeEvent(self, ev):
        # Losing focus (clicking into another window) drops the orange row highlight.
        if ev.type() == QtCore.QEvent.Type.WindowDeactivate:
            self.table.clearSelection()
        super().changeEvent(ev)

    def _context_menu(self, pos):
        item = self.table.itemAt(pos)
        if item is None or item.column() != 0:          # Call column only
            return
        call = item.text()
        menu = QtWidgets.QMenu(self)
        menu.addAction("Look up on QRZ.com",
                       lambda: QtGui.QDesktopServices.openUrl(
                           QtCore.QUrl(f"https://www.qrz.com/db/{call}")))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def mark_dirty(self):
        self.dirty = True

    def clear(self):
        if self.clear_fn is not None:
            self.clear_fn()
        self.rebuild()

    def _refresh_if_due(self):
        if self.dirty and self.isVisible():
            self.rebuild()

    def _mark_sort_column(self, col, order):
        """Highlight the sorted column's title instead of drawing a sort arrow, which would not fit."""
        pal = self.table.palette()
        for i in range(self.table.columnCount()):
            item = self.table.horizontalHeaderItem(i)
            if i == col:
                item.setBackground(pal.highlight())
                item.setForeground(pal.highlightedText())
            else:
                item.setData(QtCore.Qt.ItemDataRole.BackgroundRole, None)
                item.setForeground(QtGui.QBrush(self.TITLE_FG))

    def rebuild(self):
        stats, mode = self.source()
        self.dirty = False
        self.setWindowTitle(f"{mode} stats" if mode else "FT8 stats")
        self.table.setSortingEnabled(False)     # rows must not move while they are being filled
        self.table.setRowCount(len(stats))
        two_total = 0                   # summed over rows: each 2-way contact is counted twice
        for row, (call, s) in enumerate(stats.items()):
            total = s["total"]
            avg = s["snr_sum"] / total
            grid = s["grid"] or ""
            miles = grid_distance_miles(MY_GRID, grid) if grid else None
            # 2way: partners this call sent a report to that also sent one back to it
            two_way = sum(1 for p in s["rpt_to"] if call in stats.get(p, {}).get("rpt_to", ()))
            two_total += two_way
            cells = [
                _SortItem(call, call),
                _SortItem(time.strftime("%H:%M:%S", time.gmtime(s["first"])), s["first"]),
                _SortItem(time.strftime("%H:%M:%S", time.gmtime(s["last"])), s["last"]),
                _SortItem(str(total), total),
                _SortItem(str(s["cq"]), s["cq"]),
                _SortItem(str(two_way), two_way),
                _SortItem(f"{s['snr_min']:+d}", s["snr_min"]),
                _SortItem(f"{s['snr_max']:+d}", s["snr_max"]),
                _SortItem(f"{avg:+.1f}", avg),
                _SortItem(grid, grid),
                _SortItem(f"{miles:.0f}" if miles is not None else "",
                          miles if miles is not None else -1.0),
            ]
            for col, item in enumerate(cells):
                self.table.setItem(row, col, item)
        totals = {
            "Decodes": sum(s["total"] for s in stats.values()),
            "Calls": len(stats),
            "CQs": sum(s["cq"] for s in stats.values()),
            "2-way": two_total // 2,
        }
        for name, value in totals.items():
            self.summary_vals[name].setText(f"{value:,}")
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSortIndicatorShown(False)   # re-enabling sorting turns it on


def _opt(v, p):
    """Fixed-point text for a CSV cell, empty when there is no value."""
    return "" if v is None else f"{v:.{p}f}"


def write_wwv_capture(ring, rate, tune_hz, seconds=WWV_CAPTURE_S, outdir=None,
                      events=(), clock_ms=None):
    """Save the newest `seconds` as IQ (.npz), an 8 kHz envelope WAV and a JSON sidecar.

    Wall times use the detector's mapping (newest sample = save time), so the sidecar onsets
    and the window start share one reference. The filename uses the window start, which
    carries the PC clock offset. The envelope WAV keeps the carrier level as DC.
    Returns (iq_path, wav_path, json_path).
    """
    n = int(round(seconds * rate))
    end = ring.total
    if end < n:
        raise RuntimeError("not enough samples yet")
    x = ring.read(end - n, n)
    if x is None:
        raise RuntimeError("samples overwritten before they could be read")
    now = time.time()
    start_wall = now - n / rate + WWV_MAP_LAG_S
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(start_wall))
    base = os.path.join(outdir or os.getcwd(), f"wwv_{stamp}")
    iq_path = base + "_iq.npz"
    wav_path = base + "_env8k.wav"
    json_path = base + "_info.json"

    np.savez(iq_path, iq=x, rate=float(rate), tune_hz=float(tune_hz), wall_start=start_wall)

    env = np.abs(x).astype(np.float64)
    e8 = ssig.resample_poly(env, 1, int(rate // WWV_DEC_RATE))
    peak = float(np.max(np.abs(e8))) or 1.0
    pcm = np.clip(e8 / peak * 0.9 * 32767, -32767, 32767).astype("<i2")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(WWV_DEC_RATE)
        w.writeframes(pcm.tobytes())

    onsets = [{"wall_s": round(e["w"], 4),
               "offset_in_window_s": round(e["w"] - start_wall, 4),
               "tone_hz": float(e["tone"]), "kind": e["kind"]}
              for e in events if start_wall <= e["w"] <= now]
    info = {
        "window_start_utc": datetime.fromtimestamp(start_wall, timezone.utc).isoformat(timespec="milliseconds"),
        "window_start_wall_s": round(start_wall, 4),
        "window_seconds": seconds,
        "sample_rate_hz": rate,
        "tune_hz": tune_hz,
        "save_wall_s": round(now, 4),
        "ring_total_at_save": end,
        "clock_offset_ms_last_minute": clock_ms,
        "onsets": onsets,
        "note": ("Wall times are the PC clock as the detector maps it (newest sample = save time). "
                 "The window start in the filename carries the PC clock offset. Onsets are accepted "
                 "detector events; the last ~1 s of the window may be missing because detection "
                 "waits for later samples."),
    }
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)
    return iq_path, wav_path, json_path


class WwvStatsWindow(QtWidgets.QWidget):
    """WWV timing: one row per UTC minute, newest first. Updated as each minute completes."""

    COLUMNS = ["UTC", "Clock offset (ms)", "Marks", "Tick med (ms)", "Marker (ms)", "100 Hz edge (ms)",
               "100 Hz contrast", "Carrier (Hz)",
               "Carrier (ppm)", "Carrier SNR (dB)", "Tone (dB)", "Tones",
               "Time code (UTC)"]
    MAX_ROWS = 1440

    def __init__(self, capture_fn=None, onset_log_fn=None):
        super().__init__(None, QtCore.Qt.Window)
        self.capture_fn = capture_fn
        self.onset_log_fn = onset_log_fn
        self.setWindowTitle("WWV timing")
        self.resize(1040, 480)
        self.recent = deque(maxlen=10)           # clock offsets (ms) of the latest minutes
        self.summary = QtWidgets.QLabel("No minutes yet")
        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        note = QtWidgets.QLabel("Clock: positive means this computer's clock is ahead of UTC. "
                                "Includes USB and buffer latency and propagation delay.")
        btn_copy = QtWidgets.QPushButton("Copy CSV")
        btn_copy.clicked.connect(self.copy_csv)
        btn_save = QtWidgets.QPushButton("Save CSV...")
        btn_save.clicked.connect(self.save_csv)
        btn_cap = QtWidgets.QPushButton("Save 10 s...")
        btn_cap.clicked.connect(self.save_capture)
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(self.clear)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.summary, 1)
        top.addWidget(btn_copy)
        top.addWidget(btn_save)
        top.addWidget(btn_cap)
        top.addWidget(btn_clear)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(top)
        lay.addWidget(self.table)
        lay.addWidget(note)
        self.cap_msg = QtWidgets.QLabel("")
        lay.addWidget(self.cap_msg)
        chk = QtWidgets.QCheckBox("Log every onset to CSV (debugging)")
        chk.setChecked(False)
        chk.toggled.connect(lambda on: self.onset_log_fn and self.onset_log_fn(on))
        lay.addWidget(chk)

    def clear(self):
        """Empty the minute table and the clock-offset summary."""
        self.recent.clear()
        self.table.setRowCount(0)
        self.summary.setText("No minutes yet")

    def add(self, rec, mhz):
        car = rec["carrier"]
        vals = [
            time.strftime("%H:%M", time.gmtime(rec["wall"])),
            wwv_ms(rec.get("clock_full_ms")),
            str(rec["n"]),
            wwv_ms(rec["tick_ms"]),
            wwv_ms(rec["marker_ms"]),
            wwv_ms(rec.get("frame100_ms")),
            "--" if rec.get("frame100_contrast") is None else f"{rec['frame100_contrast']:.2f}",
            "--" if car is None else f"{car['hz']:+.2f}",
            "--" if car is None else f"{car['ppm']:+.3f}",
            "--" if car is None else f"{car['snr']:+.1f}",
            "--" if rec["tone_db"] is None else f"{rec['tone_db']:.1f}",
            wwv_tones(rec),
            rec.get("tc_time") or "--",
        ]
        self.table.insertRow(0)
        for col, v in enumerate(vals):
            self.table.setItem(0, col, QtWidgets.QTableWidgetItem(v))
        while self.table.rowCount() > self.MAX_ROWS:
            self.table.removeRow(self.table.rowCount() - 1)
        if rec.get("clock_full_ms") is not None:
            self.recent.append(rec["clock_full_ms"])
        if self.recent:
            r = list(self.recent)
            cs = ("" if car is None else
                  f"   Carrier {car['hz']:+.2f} Hz ({car['ppm']:+.3f} ppm), SNR {car['snr']:+.1f} dB")
            self.summary.setText(
                f"{mhz:.3f} MHz   Clock offset last {r[-1]:+.1f} ms, median of last {len(r)} min "
                f"{float(np.median(r)):+.1f} ms, range {min(r):+.1f} to {max(r):+.1f} ms{cs}")


    def _table_csv(self):
        """The whole table as CSV text, oldest minute first."""
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(self.COLUMNS)
        for row in reversed(range(self.table.rowCount())):
            w.writerow([self.table.item(row, c).text() if self.table.item(row, c) else ""
                        for c in range(len(self.COLUMNS))])
        return buf.getvalue()

    def copy_csv(self):
        QtWidgets.QApplication.clipboard().setText(self._table_csv())

    def save_csv(self):
        default = os.path.join(os.getcwd(), datetime.now().strftime("wwv_%Y%m%d_%H%M%S.csv"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save WWV timing table",
                                                        default, "CSV files (*.csv)")
        if path:
            with open(path, "w", newline="") as f:
                f.write(self._table_csv())


    def save_capture(self):
        if self.capture_fn is None:
            self.cap_msg.setText("Capture not available")
            return
        try:
            paths = self.capture_fn()
            self.cap_msg.setText("Saved " + "   ".join(os.path.basename(p) for p in paths))
        except Exception as e:
            self.cap_msg.setText(f"Capture failed: {e}")


class DecodePopout(QtWidgets.QWidget):
    """Free-floating decodes window. Closing it reports back so the pane can re-dock."""

    closed = QtCore.Signal()

    def __init__(self):
        super().__init__(None, QtCore.Qt.Window)
        self.setWindowTitle("FT8 decodes")
        self.btn_dock = QtWidgets.QPushButton("Dock")
        self.top = QtWidgets.QHBoxLayout()
        self.top.addStretch(1)
        self.top.addWidget(self.btn_dock)
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.addLayout(self.top)
        self.resize(700, 400)

    def closeEvent(self, ev):
        self.closed.emit()
        ev.accept()


class FreqEntry(QtWidgets.QWidget):
    """Tuned frequency in MHz as six digits; each digit has its own up and down arrow."""

    requested = QtCore.Signal(int)                      # wanted tuning frequency, Hz
    PLACES_HZ = [100_000_000, 10_000_000, 1_000_000, 100_000, 10_000, 1_000]

    def __init__(self):
        super().__init__()
        self._hz = 0
        font = QtGui.QFont("monospace")
        font.setPointSize(14)
        font.setBold(True)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.digits = []
        for i, place in enumerate(self.PLACES_HZ):
            if i == 3:                                  # decimal point after the MHz digits
                dot = QtWidgets.QLabel(".")
                dot.setFont(font)
                lay.addWidget(dot)
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(0)
            up = QtWidgets.QToolButton()
            up.setArrowType(QtCore.Qt.ArrowType.UpArrow)
            lab = QtWidgets.QLabel("0")
            lab.setFont(font)
            lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            dn = QtWidgets.QToolButton()
            dn.setArrowType(QtCore.Qt.ArrowType.DownArrow)
            col.addWidget(up)
            col.addWidget(lab)
            col.addWidget(dn)
            up.clicked.connect(lambda _=False, p=place: self._step(p))
            dn.clicked.connect(lambda _=False, p=place: self._step(-p))
            lay.addLayout(col)
            self.digits.append(lab)
        lay.addWidget(QtWidgets.QLabel("MHz"))
        lay.addStretch(1)

    def set_hz(self, hz):
        """Show the tuned frequency on the digits."""
        self._hz = int(hz)
        for lab, ch in zip(self.digits, f"{round(hz / 1000):06d}"):
            lab.setText(ch)

    def _step(self, delta_hz):
        self.requested.emit(self._hz + delta_hz)


class AudioOffsetAxis(pg.AxisItem):
    """Top axis of the waterfall: audio offset in Hz from the tuned (dial) frequency."""

    def __init__(self):
        super().__init__("top")
        self.tune_hz = 0.0
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        return [f"{v * 1e3 - self.tune_hz:.0f}" for v in values]     # values are in kHz


def utc_tick_values(ref, lo, hi):
    """Ticks for a waterfall y range lo..hi (seconds, 0 = the newest row at wall time ref), placed on
    whole seconds past the UTC minute. Returns (step_s, y values)."""
    span = abs(hi - lo)
    step = next((s for s in (1, 2, 5, 10, 15, 30, 60) if span / s <= 12), 60)
    w = math.ceil((ref + min(lo, hi)) / step) * step
    vals = []
    while w <= ref + max(lo, hi):
        vals.append(w - ref)
        w += step
    return step, vals


class UtcSecondsAxis(pg.AxisItem):
    """Left axis of the waterfall: seconds past the UTC minute. The axis is anchored to the newest
    row's wall-clock time, so the ticks move down with the rows."""

    def __init__(self, ref_fn):
        super().__init__("left")
        self.ref_fn = ref_fn
        self._ref = 0.0
        self.enableAutoSIPrefix(False)

    def tickValues(self, minVal, maxVal, size):
        self._ref = float(self.ref_fn())
        step, vals = utc_tick_values(self._ref, minVal, maxVal)
        return [(step, vals)]

    def tickStrings(self, values, scale, spacing):
        return [f"{int(round(self._ref + v)) % 60:02d}" for v in values]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, an, radio, ring, low_if, title, tune_hz, disp_off, span_hz,
                 jt9_path, shm_dir, all_txt, decode_ok, decode_start,
                 vmin=None, vmax=None, compress=6.0):
        super().__init__()
        self.an, self.radio, self.ring = an, radio, ring
        self.rate = an.rate
        self.compress = compress
        self.low_if = low_if
        self.tune_hz = tune_hz          # hardware tuning frequency, Hz
        self.disp_off = disp_off        # display centre relative to tuning, Hz
        self.span_hz = span_hz          # display width, Hz
        self.jt9_path = jt9_path
        self.shm_dir = shm_dir
        self.all_txt = all_txt
        self.cfg_version = 0            # bumped on retune/span change
        self.cur_slot = None            # (boundary_time, sample_index, cfg_version)
        self.paused = False             # True: receiver stopped, display frozen
        self.newest_wall = time.time()  # wall-clock time of the newest waterfall row (UTC axis origin)
        self.clock_sec = -1             # whole second last shown in the UTC clock
        self.clock_min = -1             # UTC minute last shown on the waterfall
        self.noise_blk = 0
        self.blue_mode = "avg"          # blue trace: "avg" (15 s average) or "peak" (peak hold)
        self.spec_y = None              # raw-mode spectrum y range currently set, (lo, hi) dB
        self.decoding = False
        self.decode_times = deque()     # slot start times of decodes in the rate window
        self.decode_t0 = None           # when decoding was switched on; None while off
        self.mode = None                # "FT8", "FT4", or None for decoding off
        self.manual_tuned = False       # True once the user has retuned or changed the span
        self.wwv_on = False             # WWV timing mode: tuned to a WWV frequency, decoding off
        self.wwv = None                 # WwvMonitor while wwv_on
        self.cw_on = False              # CW decode mode: 3 kHz window, Morse decoding on
        self.cw = None                  # CwDecoder or FmDecoder while cw_on
        self.fm_plot = None             # FmLevelPlot window, shown while in FM mode
        self.cw_kind = "CW"             # "CW" or "FM": which decoder self.cw is
        self.cw_txt = os.path.join(os.path.dirname(os.path.abspath(all_txt)), "CW.TXT")
        self.fm_txt = os.path.join(os.path.dirname(os.path.abspath(all_txt)), "FM.TXT")
        self.onset_log_on = False       # write the onset CSV (debugging); set from the WWV window
        self.wwv_mhz = WWV_DEFAULT_MHZ  # WWV frequency last chosen from the menu
        self.track_on = False           # Track mode: one line per transmission, logged to CSV
        self.trk = None                 # TrackDetector while track_on
        self.trk_n = 0                  # transmissions logged in this Track session
        self.trk_path = None            # CSV file of this Track session
        self.trk_file = None
        self.trk_writer = None
        self.iq = None                  # IqRecorder while Settings > Record IQ on signal is on

        # Decode marks are drawn into the waterfall's own pixel buffer. Each mark records the
        # sample range of its slot, so it scrolls on the same clock as the waterfall rows.
        self.marks = []                 # [{"s0", "s1", "cq", "x0", "w"}]
        self.show_marks = True
        self.grids = {}                 # session: callsign -> last grid square heard from it
        self.stats = {}                 # callsign -> running packet totals, for the Stats window
        self.stats_mode = None          # mode the stats were gathered in: "FT8", "FT4" or None
        self.ov = None                  # overlay codes, same shape as wf: 0 none, 1 other, 2 CQ
        self.newest_s = None            # sample index of the newest waterfall row's start
        self.dirty = False              # set when the image needs rebuilding

        self.levels = (vmin, vmax) if (vmin is not None and vmax is not None) else None
        self.fixed_levels = self.levels     # user-given colour range, in SNR units
        self.rows = 0
        self.valid_rows = 0
        self.wf = None

        self.slot_counts = []           # [{"s0", "s1", "n", "item"}] decode count per slot, for the strip
        self.bridge = DecodeBridge()
        self.bridge.decode.connect(self._on_decode)
        self.bridge.status.connect(self._on_decode_status)
        self.bridge.done.connect(self._on_decode_done)
        self.bridge.count.connect(self._on_slot_count)

        self.spec = pg.PlotWidget(title="Spectrum")
        self.spec.setLabel("left", "SNR", units="dB")
        self.spec.setLabel("bottom", "Frequency", units="kHz")
        self.spec.getAxis("bottom").enableAutoSIPrefix(False)
        self.spec.setYRange(-5, 40, padding=0)
        self.spec.addLegend(offset=(-10, 10))
        self.spec_curve = self.spec.plot(pen="y", name="Live")
        self.spec_avg_curve = self.spec.plot(pen=pg.mkPen((0, 200, 255), width=2),
                                             name="15 s average")

        self.wf_plot = pg.PlotWidget(axisItems={"top": AudioOffsetAxis(),
                                                "left": UtcSecondsAxis(lambda: self.newest_wall)})
        self.wf_plot.setLabel("bottom", "Frequency", units="kHz")
        self.row_s = an.hop / self.rate             # seconds per waterfall row
        self.wf_plot.setLabel("left", "Seconds past UTC minute")
        self.wf_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.off_axis = self.wf_plot.getAxis("top")
        self.wf_plot.setLabel("top", "Audio offset", units="Hz")
        self.wf_plot.showAxis("top")
        self.wf_plot.setXLink(self.spec)
        self.wf_plot.setMouseEnabled(x=False, y=False)
        self.img = pg.ImageItem()
        self.wf_plot.addItem(self.img)
        # UTC HH:MM sits in the empty top-left corner above the waterfall, level with the top axis labels
        self.utc_min_lbl = QtWidgets.QLabel("----------\n--:-- UTC", self.wf_plot)
        self.utc_min_lbl.setStyleSheet("background-color: rgba(0, 0, 0, 170); color: #e6e6e6; "
                                       "font-family: monospace; font-weight: bold; padding: 1px 4px;")
        self.utc_min_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.utc_min_lbl.adjustSize()
        self.utc_min_lbl.move(6, (COUNT_AXIS_H - self.utc_min_lbl.height()) // 2)
        self.wf_plot.getAxis("top").setHeight(COUNT_AXIS_H)
        self.wf_plot.getAxis("bottom").setHeight(COUNT_AXIS_H)
        # narrow strip right of the waterfall: one decode count per slot, scrolls with the rows
        self.count_plot = pg.PlotWidget()
        self.count_plot.setFixedWidth(COUNT_STRIP_W)
        self.count_plot.hideAxis("left")
        self.count_plot.hideAxis("right")
        self.count_plot.showAxis("top")
        self.count_plot.showAxis("bottom")
        for name in ("top", "bottom"):
            self.count_plot.getAxis(name).setStyle(showValues=False)
            self.count_plot.getAxis(name).setHeight(COUNT_AXIS_H)
        self.count_plot.setMouseEnabled(x=False, y=False)
        self.count_plot.setMenuEnabled(False)
        self.count_plot.hideButtons()
        self.count_plot.setYLink(self.wf_plot)
        self.count_plot.disableAutoRange()
        self.count_plot.setXRange(0.0, 1.0, padding=0)
        # spacer under the spectrum, so its frequency axis still lines up with the waterfall's
        self.count_spacer = QtWidgets.QWidget()
        self.count_spacer.setFixedWidth(COUNT_STRIP_W)
        self.lut_rgba = pg.colormap.get("viridis").getLookupTable(
            0.0, 1.0, 256, alpha=True, mode="byte")          # (256, 4) uint8

        # --- controls row
        self.btn_save = QtWidgets.QPushButton("Save  ▾")
        self.btn_popout = QtWidgets.QPushButton("Pop out")
        self.btn_marks = QtWidgets.QPushButton("Mark")
        self.btn_marks.setCheckable(True)
        self.btn_marks.setChecked(True)                 # on by default
        self.show_marks = True
        self.btn_calls = QtWidgets.QPushButton("Call")
        self.btn_calls.setCheckable(True)
        self.btn_calls.setChecked(True)                 # on by default, as Mark
        self.show_call = True
        # mode menu: FT8, FT4, CW, FM, Track, a WWV submenu (frequencies, timing window, capture), Off
        self.mode_menu = QtWidgets.QMenu(self)
        self.mode_actions = {}
        for name in ("FT8", "FT4", "CW", "FM", "Track"):
            a = self.mode_menu.addAction(name)
            a.setCheckable(True)
            a.triggered.connect(lambda _=False, n=name: self._on_mode_action(n))
            self.mode_actions[name] = a
        if not decode_ok:
            for name in ("FT8", "FT4"):                 # both need jt9
                self.mode_actions[name].setEnabled(False)
        self.wwv_menu = self.mode_menu.addMenu("WWV")
        self.wwv_actions = []
        for name, mhz in WWV_FREQS_MHZ:
            a = self.wwv_menu.addAction(f"WWV {name}")
            a.setCheckable(True)
            a.triggered.connect(lambda _=False, m=mhz: self._enter_wwv(m))
            self.wwv_actions.append((mhz, a))
        self.wwv_menu.addSeparator()
        a_win = self.wwv_menu.addAction("Timing window...")
        a_win.triggered.connect(self.show_wwv_stats)
        a_cap = self.wwv_menu.addAction("Save 10 s capture")
        a_cap.triggered.connect(self._menu_save_capture)
        a_off = self.mode_menu.addAction("Off")
        a_off.setCheckable(True)
        a_off.triggered.connect(lambda _=False: self._on_mode_action("Off"))
        self.mode_actions["Off"] = a_off
        self.mode_btn = QtWidgets.QPushButton()
        self.mode_btn.setMenu(self.mode_menu)
        start = "FT8" if (decode_start and decode_ok) else "Off"
        self._select_mode_box(start)
        if start != "Off":
            self.mode = start
            self.decode_t0 = time.time()
        self.stats_mode = self.mode
        self.band_box = QtWidgets.QComboBox()
        self.band_box.addItem("Band...", None)
        for name, ft8_mhz, ft4_mhz in BANDS:
            self.band_box.addItem(name, (ft8_mhz, ft4_mhz))
        self.band_box.setMaxVisibleItems(self.band_box.count())     # every band shows without scrolling
        self.band_box.currentIndexChanged.connect(self.on_band)
        self.span_box = QtWidgets.QComboBox()
        for name, khz in SPAN_PRESETS:
            self.span_box.addItem(name, khz)
        self.span_box.addItem("Max", None)
        self._select_span(span_hz / 1e3)

        controls = QtWidgets.QHBoxLayout()
        self.freq_entry = FreqEntry()
        self.freq_entry.requested.connect(self.tune_to)
        controls.addWidget(self.freq_entry)
        controls.addWidget(self.band_box)
        controls.addWidget(QtWidgets.QLabel("Span:"))
        controls.addWidget(self.span_box)
        self.blue_box = QtWidgets.QComboBox()
        self.blue_box.addItem("15 s average", "avg")
        self.blue_box.addItem("Peak hold", "peak")
        self.blue_box.addItem("Tx Test", "txskip")
        self.blue_box.currentIndexChanged.connect(self._on_blue_mode)
        controls.addWidget(self.blue_box)
        self.btn_peak_reset = QtWidgets.QPushButton("Reset peak")
        self.btn_peak_reset.clicked.connect(self._reset_peak)
        controls.addWidget(self.btn_peak_reset)
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._set_paused)
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_save)
        controls.addWidget(self.mode_btn)
        controls.addWidget(self.btn_marks)
        controls.addWidget(self.btn_calls)
        # Settings menu: display options, and the FFT size (applied in place)
        self.settings_menu = QtWidgets.QMenu(self)
        self.act_raw = self.settings_menu.addAction("Raw power")
        self.act_raw.setCheckable(True)
        self.act_raw.setChecked(True)
        self.act_avg = self.settings_menu.addAction("Time average")
        self.act_avg.setCheckable(True)
        self.act_avg.setChecked(False)
        self.an.set_avg(False)          # startup default: no time average
        self.act_avg.toggled.connect(self.an.set_avg)
        self.settings_menu.addSeparator()
        self.act_iq = self.settings_menu.addAction("Record IQ on signal")
        self.act_iq.setCheckable(True)
        self.act_iq.toggled.connect(self._set_iq)
        self.settings_menu.addSeparator()
        self.nfft_menu = self.settings_menu.addMenu("FFT size")
        self.nfft_group = QtGui.QActionGroup(self)
        self.nfft_actions = {}
        for n in NFFT_CHOICES:
            a = self.nfft_menu.addAction(f"{n}   bin {self.rate / n:.2f} Hz")
            a.setCheckable(True)
            a.setChecked(n == self.an.nfft)
            self.nfft_group.addAction(a)
            a.triggered.connect(lambda _=False, n=n: self._on_nfft_action(n))
            self.nfft_actions[n] = a
        self.btn_settings = QtWidgets.QPushButton("Settings  ▾")
        self.btn_settings.setMenu(self.settings_menu)
        controls.addWidget(self.btn_settings)
        controls.addStretch(1)
        self.btn_exit = QtWidgets.QPushButton("Exit")
        controls.addWidget(self.btn_exit)

        self.span_box.currentIndexChanged.connect(self.on_span)
        self.save_menu = QtWidgets.QMenu(self)
        self.save_menu.addAction("Save PNG image").triggered.connect(self.save_png)
        self.save_menu.addAction("Save blue trace as CSV").triggered.connect(self.save_trace_csv)
        self.btn_save.setMenu(self.save_menu)
        self.btn_save.setToolTip("Save PNG image (Ctrl+S) or the blue trace as CSV")
        self.save_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.save_png)
        self.btn_popout.clicked.connect(self.pop_out)
        self.btn_exit.clicked.connect(self.exit_app)
        self.btn_marks.toggled.connect(self._set_marks_visible)
        self.btn_calls.toggled.connect(self._set_calls_visible)
        self.act_raw.toggled.connect(self._on_raw_toggled)
        self.spec.setLabel("left", "Power" if self.act_raw.isChecked() else "SNR", units="dB")

        # --- one status line: radio/FFT info, transient messages, version
        self.status = QtWidgets.QLabel()
        self.msg = QtWidgets.QLabel()
        self.version_label = QtWidgets.QLabel(f"v{VERSION}")
        self.version_label.setStyleSheet("color: #808080;")
        status_row = QtWidgets.QHBoxLayout()
        self.rate_label = QtWidgets.QLabel()
        self.utc_label = QtWidgets.QLabel("UTC --:--:--")
        self.utc_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        status_row.addWidget(self.utc_label)
        status_row.addWidget(self.status)
        self.tx_label = QtWidgets.QLabel()
        status_row.addWidget(self.rate_label)
        status_row.addWidget(self.tx_label)
        status_row.addWidget(self.msg, 1)
        status_row.addWidget(self.version_label)

        for pw in (self.spec, self.wf_plot):
            pw.getAxis("left").setWidth(60)             # fixed left-axis width
        self.spec.disableAutoRange(axis="y")            # stop y-autoranging from changing labels
        self.wf_plot.disableAutoRange(axis="x")

        spec_row = QtWidgets.QHBoxLayout()
        spec_row.setContentsMargins(0, 0, 0, 0)
        spec_row.setSpacing(0)
        spec_row.addWidget(self.spec, 1)
        spec_row.addWidget(self.count_spacer)
        wf_row = QtWidgets.QHBoxLayout()
        wf_row.setContentsMargins(0, 0, 0, 0)
        wf_row.setSpacing(0)
        wf_row.addWidget(self.wf_plot, 1)
        wf_row.addWidget(self.count_plot)
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addLayout(controls)
        lay.addLayout(spec_row, 1)
        lay.addLayout(wf_row, 2)
        lay.addLayout(status_row)
        self.setCentralWidget(central)
        self.resize(1200 + COUNT_STRIP_W, 900)     # widened so the waterfall keeps its width
        self._set_count_strip(self.mode is not None)

        # --- decodes: docked along the bottom, or popped out into its own window
        self.decode_pane = DecodeWindow()
        self.decode_dock = QtWidgets.QDockWidget("FT8 decodes", self)
        self.decode_dock.setObjectName("decodes_dock")
        self.decode_dock.setWidget(self.decode_pane)
        self.decode_pane.top_bar.layout().addWidget(self.btn_popout)
        self.btn_clear_log = QtWidgets.QPushButton("Clear log")     # clears the decodes window in every mode
        self.btn_clear_log.clicked.connect(self._clear_log)
        self.decode_pane.top_bar.layout().addWidget(self.btn_clear_log)
        self.chk_noise = QtWidgets.QCheckBox("Noise stats")
        self.chk_noise.setChecked(True)
        self.chk_noise.setVisible(False)
        self.decode_pane.top_bar.layout().addWidget(self.chk_noise)
        self.btn_ref_reset = QtWidgets.QPushButton("Ref reset")
        self.btn_ref_reset.setVisible(False)
        self.btn_ref_reset.clicked.connect(self._ref_reset)
        self.decode_pane.top_bar.layout().addWidget(self.btn_ref_reset)
        self.btn_check_noise = QtWidgets.QPushButton("Check Noise")
        self.btn_check_noise.setVisible(False)
        self.btn_check_noise.clicked.connect(self._check_noise)
        self.decode_pane.top_bar.layout().addWidget(self.btn_check_noise)
        self.chk_noise.toggled.connect(self._sync_check_btn)
        self.decode_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.decode_dock)
        self.resizeDocks([self.decode_dock], [DECODE_PANE_HEIGHT],
                         QtCore.Qt.Orientation.Vertical)

        self.popout = DecodePopout()
        self.popout.btn_dock.clicked.connect(self.dock_back)
        self.popout.closed.connect(self.dock_back)
        self.popped = False             # True while the history lives in the popout window

        self.stats_win = StatsWindow(lambda: (self.stats, self.stats_mode), clear_fn=self._clear_stats)
        self.decode_pane.btn_stats.clicked.connect(self.show_stats)
        self.wwv_win = WwvStatsWindow(capture_fn=self.save_wwv_capture,
                                      onset_log_fn=self.set_onset_log)

        self._configure(settle=False)
        self._refresh()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)

        # --- hover popup: sender's callsign while the pointer is inside a decoded signal's box
        # Parented to the main window so the window manager treats it as this window's tooltip,
        # not one belonging to whichever window (such as the Stats window) last had focus.
        self.pop = QtWidgets.QLabel(self, QtCore.Qt.WindowType.ToolTip)
        self.pop.setFont(QtGui.QFont("sans-serif", 11))
        self.pop.setMargin(6)
        self.pop_mark = None
        self.pop_html = None
        self.pop_hide = QtCore.QTimer(self)
        self.pop_hide.setSingleShot(True)
        self.pop_hide.setInterval(1000)                 # stays at least 1 s after the pointer leaves
        self.pop_hide.timeout.connect(self._hide_popup)
        self.wf_plot.scene().sigMouseMoved.connect(self._on_hover)
        self.wf_plot.viewport().installEventFilter(self)

        # --- spectrum hover readout: frequency and level of the highest blue-trace point nearby
        self.spec_pop = QtWidgets.QLabel(self, QtCore.Qt.WindowType.ToolTip)
        self.spec_pop.setFont(QtGui.QFont("sans-serif", 11))
        self.spec_pop.setMargin(6)
        self.spec_pop.setStyleSheet("background-color: rgba(0, 0, 0, 220); "
                                    "border: 1px solid #00c8ff; color: #00c8ff;")
        self.spec.scene().sigMouseMoved.connect(self._on_spec_hover)
        self.spec.viewport().installEventFilter(self)
        self.spec_marker = pg.ScatterPlotItem(symbol="o", size=12, pen=pg.mkPen("w", width=2),
                                              brush=pg.mkBrush(0, 0, 0, 0))
        self.spec_marker.setVisible(False)
        self.spec.addItem(self.spec_marker)

    # ---- exit: close every window, then leave app.exec() so main() releases the radio
    def _detach_filters(self):
        self.timer.stop()
        self.spec.viewport().removeEventFilter(self)
        self.wf_plot.viewport().removeEventFilter(self)

    def closeEvent(self, ev):
        self.save_settings()
        if self.iq is not None:
            self._iq_log(self.iq.close())
        self._detach_filters()
        super().closeEvent(ev)

    # ---- settings file: written on exit, read on start (load_settings and main)
    def settings_dict(self):
        if self.cw_on:
            mode = self.cw_kind
        elif self.wwv_on:
            mode = "WWV"
        elif self.track_on:
            mode = "Track"
        else:
            mode = self.mode or "Off"
        return {
            "frequency_mhz": round(self.tune_hz / 1e6, 6),
            "center_mhz": round((self.tune_hz + self.disp_off) / 1e6, 6),
            "span_khz": round(self.span_hz / 1e3, 6),
            "band": self.band_box.currentText(),
            "mode": mode,
            "wwv_mhz": self.wwv_mhz,
            "peak_avg": self.blue_box.currentData(),
            "raw_power": self.act_raw.isChecked(),
            "time_average": self.act_avg.isChecked(),
            "fft_size": self.an.nfft,
        }

    def save_settings(self):
        path = os.path.join(os.getcwd(), SETTINGS_NAME)
        try:
            with open(path, "w") as f:
                json.dump(self.settings_dict(), f, indent=2)
        except Exception as e:
            print(f"warning: cannot save {path} ({e})", file=sys.stderr)

    def apply_settings(self, saved, decode_allowed):
        """Restore the mode, peak/average and display settings. The saved window is kept: entering
        a mode does not move the tuning."""
        def get(key, typ):
            v = saved.get(key)
            return v if isinstance(v, typ) else None
        self.manual_tuned = True
        mode = get("mode", str)
        if mode in ("FT8", "FT4") and not decode_allowed:
            mode = "Off"
        wwv = get("wwv_mhz", (int, float))
        if mode == "WWV" and wwv is not None:
            self._enter_wwv(float(wwv))
        elif mode in ("CW", "FM", "Track", "FT8", "FT4", "Off"):
            self._on_mode_action(mode)
        peak = get("peak_avg", str)
        if peak is not None and self.blue_box.findData(peak) >= 0:
            self.blue_box.setCurrentIndex(self.blue_box.findData(peak))
        raw = get("raw_power", bool)
        if raw is not None:
            self.act_raw.setChecked(raw)
        avg = get("time_average", bool)
        if avg is not None:
            self.act_avg.setChecked(avg)
        # Entering CW, FM or WWV sets its own narrow window, so the saved window is put back last
        center, span = get("center_mhz", (int, float)), get("span_khz", (int, float))
        if center is not None and span is not None and span > 0:
            c_khz, half = float(center) * 1e3, float(span) / 2
            self._tune_window(self.tune_hz / 1e6, c_khz - half, c_khz + half)

    def exit_app(self):
        self._detach_filters()
        self.popout.close()             # re-docks the pane if it was popped out; harmless otherwise
        self.stats_win.close()
        self.wwv_win.close()
        self.close()
        QtWidgets.QApplication.quit()

    # ---- decodes pane: pop out / dock back
    def _bar_controls(self):
        """Buttons of the decodes top bar other than Pop out, in display order."""
        pane = self.decode_pane
        return [pane.btn_stats, pane.btn_split, self.btn_clear_log,
                self.chk_noise, self.btn_ref_reset, self.btn_check_noise]

    def pop_out(self):
        if not self.popped:
            self.popout.lay.addWidget(self.decode_pane)        # moves the whole decodes window
            # the top bar is hidden below, so every button in it moves to the popout's bar
            for i, w in enumerate(self._bar_controls()):
                self.popout.top.insertWidget(1 + i, w)         # after the stretch, before Dock
            self.decode_pane.top_bar.hide()
            self.popped = True
            self.decode_dock.hide()
        self.popout.show()
        self.popout.raise_()
        self.popout.activateWindow()

    def dock_back(self):
        if not self.popped:
            return
        self.decode_dock.setWidget(self.decode_pane)                 # moves the decodes window back
        # restore the bar: Stats, Two panes, Pop out (it never left the bar), then the rest
        bar = self.decode_pane.top_bar.layout()
        bar.insertWidget(1, self.decode_pane.btn_stats)
        bar.insertWidget(2, self.decode_pane.btn_split)
        for i, w in enumerate(self._bar_controls()[2:]):
            bar.insertWidget(4 + i, w)
        self.decode_pane.top_bar.show()
        self.popped = False
        self.popout.hide()
        self.decode_dock.show()
        self.decode_dock.raise_()

    # ---- span selection helpers
    def _select_span(self, khz):
        """Select the matching preset, or add a custom entry, without firing signals."""
        self.span_box.blockSignals(True)
        idx = None
        for i in range(self.span_box.count()):
            data = self.span_box.itemData(i)
            if data is not None and abs(data - khz) < 1e-6:
                idx = i
                break
        if idx is None:
            self.span_box.addItem(f"Custom {khz:g} kHz", khz)
            idx = self.span_box.count() - 1
        self.span_box.setCurrentIndex(idx)
        self.span_box.blockSignals(False)

    # ---- configuration
    def _configure(self, settle):
        self._flush_tracks()            # a new frequency axis ends any transmission in progress
        self.spec_y = None              # refit the spectrum axis to the new span
        half = self.rate / 2
        lo = max(self.disp_off - self.span_hz / 2, -half)
        hi = min(self.disp_off + self.span_hz / 2, half)
        self.an.configure(self.tune_hz, lo, hi, settle_s=SETTLE_S if settle else 0.0)
        self.f0 = float(self.an.freq_khz[0])
        self.f1 = float(self.an.freq_khz[-1])
        self.spec_y = None
        if self.cw is not None:
            self.cw.set_window(self.tune_hz, lo, hi)
        self._iq_window()

    def _refresh(self):
        self.off_axis.tune_hz = float(self.tune_hz)     # audio offset is measured from the dial
        self.off_axis.picture = None
        self.spec.setXRange(self.f0, self.f1, padding=0)
        self.wf_plot.setXRange(self.f0, self.f1, padding=0)
        self.freq_entry.set_hz(self.tune_hz)
        self.setWindowTitle(f"Airspy HF+  {self.tune_hz / 1e6:.6f} MHz")
        clipped = self.span_hz > FT8_AUDIO_MAX_SPAN
        self.status.setText(
            f"{self.rate / 1e3:.0f} kS/s   FFT {self.an.nfft} "
            f"(bin {self.rate / self.an.nfft:.2f} Hz, row {self.row_s * 1e3:.1f} ms)   "
            f"{'LOW-IF' if self.low_if else 'zero-IF'}   "
            f"Decode {self.f0:.3f}-{self.f1:.3f} kHz"
            + (" (3.8 kHz max)" if clipped else ""))
        if self.rows:
            self._realloc(self.rows)
        if self.wwv_on:
            if self.wwv is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mlog = os.path.join(os.getcwd(), f"wwv_minutes_{stamp}.csv")
                self.wwv = WwvMonitor(self.ring, self.rate, self.tune_hz, minute_log_path=mlog)
                if self.onset_log_on:
                    self.wwv.start_onset_log(self._onset_log_path())
            elif self.wwv.nominal_hz != self.tune_hz:
                self.wwv.set_nominal(self.tune_hz)      # any retune re-references the carrier
        self._sync_band()
        self._sync_wwv_box()

    # ---- display mode: power above background (SNR) or absolute power
    def _on_raw_toggled(self, on):
        self.an.raw = on
        self.spec.setLabel("left", "Power" if on else "SNR", units="dB")
        self.spec.setYRange(-10 if on else -5, 40, padding=0)   # fixed scale in both modes
        self.spec_y = None
        self.an.peak_db = None
        self.levels = None if on else self.fixed_levels    # colour range is re-set for the new units
        if self.rows:
            self._realloc(self.rows)                        # history is in the other units: clear it

    # ---- decode mode: FT8, FT4, WWV or off (one at a time)
    def _on_mode_action(self, name):
        if name == "Track":
            self._enter_track()
            self._select_mode_box("Track")
            return
        if self.track_on:
            self._leave_track()
        if self.wwv_on:
            self._leave_wwv()
        if name in ("CW", "FM"):
            self._enter_cw(name)
            return
        mode = None if name == "Off" else name
        if self.cw_on:
            self._leave_cw()
        if mode is not None and self.an.skip_tx:
            self.blue_box.setCurrentIndex(self.blue_box.findData("peak"))
        if mode != self.mode:
            self._set_mode(mode)
            if mode is not None and not self.manual_tuned:
                self._shift_to(mode)
        self._select_mode_box(name)

    def _select_mode_box(self, name):
        """Show a mode (FT8, FT4, WWV or Off) in the mode menu without firing its handler."""
        for n, a in self.mode_actions.items():
            a.setChecked(n == name)
        for mhz, a in self.wwv_actions:
            a.setChecked(name == "WWV" and abs(mhz - self.wwv_mhz) < 1e-6)
        label = f"WWV {self.wwv_mhz:g} MHz" if name == "WWV" else name
        self.mode_btn.setText(label + "  \u25BE")

    def _menu_save_capture(self):
        try:
            paths = self.save_wwv_capture()
            self.tx_label.setText("Saved " + "  ".join(os.path.basename(p) for p in paths))
        except Exception as e:
            self.tx_label.setText(f"Capture failed: {e}")

    def _set_mode(self, mode):
        self.mode = mode
        self._clear_slot_counts()
        self._set_count_strip(mode is not None)
        if mode is not None and mode != self.stats_mode:    # Stats cover one mode at a time
            self.stats = {}
            self.stats_mode = mode
            self.stats_win.mark_dirty()
        self.cfg_version += 1           # a slot in progress under the old mode is skipped
        self.cur_slot = None            # slot boundaries follow the new cycle length
        self.decode_times.clear()
        self.decode_t0 = time.time() if mode is not None else None
        title = f"{mode} decodes" if mode else "Decodes"
        self.decode_dock.setWindowTitle(title)
        self.popout.setWindowTitle(title)
        self._update_rate()

    def _shift_to(self, mode):
        """Move the tuning and display window to the normal calling frequency for this mode."""
        m = MODES[mode]
        self._tune_window(m["dial_mhz"], m["lo"], m["hi"])

    def _tune_window(self, dial_mhz, lo_khz, hi_khz):
        """Retune to dial_mhz and show lo_khz-hi_khz (absolute kHz). Returns False if the radio refuses."""
        tune = round(dial_mhz * 1e6)
        if not self.radio.set_freq(tune):
            self.msg.setText("airspyhf_set_freq() failed")
            return False
        self.tune_hz = tune
        self.levels = self.fixed_levels     # new band: re-learn the colour range from its first rows
        self.disp_off = round((lo_khz + hi_khz) / 2 * 1e3) - tune
        self.span_hz = (hi_khz - lo_khz) * 1e3
        self.cfg_version += 1
        self._select_span(self.span_hz / 1e3)
        self._configure(settle=True)
        self._refresh()
        self.msg.setText("")
        return True

    # ---- band menu
    def on_band(self, _idx):
        dials = self.band_box.currentData()
        if dials is None:
            return
        # FT4 selected: use the FT4 segment of the band; FT8 or off: the FT8 segment
        dial = dials[1] if self.mode == "FT4" else dials[0]
        if dial is None:
            self.msg.setText("No standard FT4 frequency on this band")
            return
        self.manual_tuned = True
        name = self.band_box.currentText()
        if name in BAND_SPAN_OVERRIDE_KHZ:
            half = BAND_SPAN_OVERRIDE_KHZ[name] / 2
            self._tune_window(dial, dial * 1e3 - half, dial * 1e3 + half)
        else:
            self._tune_window(dial, dial * 1e3, dial * 1e3 + BAND_SPAN_KHZ)

    def _sync_band(self):
        """Show the band-menu entry whose FT8 or FT4 dial matches the current frequency, or 'Band...'."""
        f = self.tune_hz / 1e6
        idx = 0
        for i in range(1, self.band_box.count()):
            for d in self.band_box.itemData(i):
                if d is not None and abs(d - f) < 1e-6:
                    idx = i
                    break
            if idx:
                break
        self.band_box.blockSignals(True)
        self.band_box.setCurrentIndex(idx)
        self.band_box.blockSignals(False)

    # ---- decode rate

    def _update_rate(self):
        """Refresh the decodes-per-hour readout over the last 15 minutes, or since decoding started."""
        name = self.mode or "Decode"
        if self.decode_t0 is None:
            self.rate_label.setText(f"   {name}: off")
            return
        now = time.time()
        start = max(now - RATE_WINDOW_S, self.decode_t0)
        while self.decode_times and self.decode_times[0] < start:
            self.decode_times.popleft()
        span = now - start
        if span < RATE_MIN_SPAN_S:
            self.rate_label.setText(f"   {name} rate: --")
            return
        n = len(self.decode_times)
        basis = (f"last {RATE_WINDOW_S // 60} min" if span >= RATE_WINDOW_S
                 else f"{span / 60:.1f} min so far")
        self.rate_label.setText(f"   {name} rate: {n * 3600 / span:.0f}/h ({basis})")

    def tune_to(self, hz):
        self.retune(hz - self.tune_hz)

    def _on_blue_mode(self, _idx):
        self.blue_mode = self.blue_box.currentData()
        self.an.skip_tx = self.blue_mode == "txskip"
        self.spec.plotItem.legend.getLabel(self.spec_avg_curve).setText(self.blue_box.currentText())
        if self.an.skip_tx:
            self._enter_tx_mode()
        else:
            self._leave_tx_mode()

    def _enter_tx_mode(self):
        self.an.tx.reset_events()
        if self.track_on:
            self._leave_track()
        if self.wwv_on:
            self._leave_wwv()
        if self.cw_on:
            self._leave_cw()
        self._set_mode(None)                    # Tx Test, WWV and FT8/FT4 are mutually exclusive
        self._select_mode_box("Off")
        self.decode_dock.setWindowTitle("Tx log")
        self.chk_noise.setVisible(True)
        self.btn_ref_reset.setVisible(True)
        self.btn_check_noise.setVisible(True)
        self._sync_check_btn(self.chk_noise.isChecked())
        self.noise_blk = int(time.time() // NOISE_REPORT_S)
        self.tx_label.setText("Tx Test: listening")

    def _leave_tx_mode(self):
        self.tx_label.setText("")
        self.chk_noise.setVisible(False)
        self.btn_ref_reset.setVisible(False)
        self.btn_check_noise.setVisible(False)
        self.decode_dock.setWindowTitle("Decodes")

    # ---- Track mode: one decodes-pane line and one CSV row per transmission
    def _enter_track(self):
        if self.track_on:
            return
        if self.cw_on:
            self._leave_cw()
        if self.wwv_on:
            self._leave_wwv()
        if self.an.skip_tx:
            self.blue_box.setCurrentIndex(self.blue_box.findData("peak"))
        self._set_mode(None)
        self.track_on = True
        self.trk = TrackDetector()
        self.trk_n = 0
        self.trk_path = os.path.join(os.getcwd(), datetime.now().strftime("track_%Y%m%d_%H%M%S.csv"))
        self.trk_file, self.trk_writer = None, None
        try:
            self.trk_file = open(self.trk_path, "w", newline="")
            self.trk_writer = csv.writer(self.trk_file)
            self.trk_writer.writerow(TRACK_CSV_FIELDS)
        except OSError as e:
            self.msg.setText(f"Track CSV not written: {e}")
        self.decode_dock.setWindowTitle("Track log")
        self.popout.setWindowTitle("Track log")
        self._select_mode_box("Track")
        self.rate_label.setText("")
        self.tx_label.setText("Track: listening")

    def _leave_track(self):
        self._flush_tracks()
        self.track_on = False
        self.trk = None
        if self.trk_file is not None:
            self.trk_file.close()
        self.trk_file, self.trk_writer = None, None
        self.tx_label.setText("")
        self.decode_dock.setWindowTitle("Decodes")
        self.popout.setWindowTitle("Decodes")
        self._update_rate()

    def _flush_tracks(self):
        if self.trk is not None:
            self._log_tracks(self.trk.flush())

    def _log_tracks(self, recs):
        for r in recs:
            self.decode_pane.append(track_line(r, self.an.raw), False)
            if self.trk_writer is not None:
                self.trk_writer.writerow(track_csv_row(r, self.an.raw))
            self.trk_n += 1
        if recs and self.trk_file is not None:
            self.trk_file.flush()

    def _update_wf_clock(self):
        """Waterfall label: UTC date and minute of the newest row, the same reference as the seconds
        axis. Only changes when rows arrive, so it stays fixed while the display is paused."""
        minute = int(self.newest_wall // 60)
        if minute != self.clock_min:
            self.clock_min = minute
            self.utc_min_lbl.setText(time.strftime("%Y-%m-%d\n%H:%M UTC", time.gmtime(minute * 60)))
            self.utc_min_lbl.adjustSize()

    def _feed_track(self, rows_db):
        """Feed the newest display rows to the track detector. Row times are wall-clock: the newest
        row is the centre of the newest FFT frame, and each earlier row is one row earlier."""
        k = rows_db.shape[0]
        newest = self.newest_wall
        closed = []
        for i in range(k):
            closed.extend(self.trk.feed(rows_db[i], self.an.freq_khz,
                                        newest - (k - 1 - i) * self.row_s))
        self._log_tracks(closed)
        self.tx_label.setText(f"Track: {self.trk_n} logged, {len(self.trk.active)} in progress")

    # ---- CW and FM modes: Morse decoder on a 3 kHz window. CW reads on/off keying of the carrier;
    # FM reads keying of a 400 Hz FM tone from the sidebands of the carrier.
    def _enter_cw(self, kind="CW"):
        if self.track_on:
            self._leave_track()
        if self.an.skip_tx:
            self.blue_box.setCurrentIndex(self.blue_box.findData("peak"))
        self._set_mode(None)
        self.cw_on = True
        self.cw_kind = kind
        if kind == "FM":
            if self.fm_plot is None:
                self.fm_plot = FmLevelPlot()
            self.fm_plot.show()
        self.cw = (FmDecoder if kind == "FM" else CwDecoder)(self.ring, self.rate)
        self.decode_dock.setWindowTitle(f"{kind} decodes")
        self.popout.setWindowTitle(f"{kind} decodes")
        self._select_mode_box(kind)
        khz = self.tune_hz / 1e3
        self._tune_window(self.tune_hz / 1e6, khz - CW_SPAN_KHZ / 2, khz + CW_SPAN_KHZ / 2)
        self.cw.set_window(self.tune_hz, self.disp_off - self.span_hz / 2,
                           self.disp_off + self.span_hz / 2)
        self.rate_label.setText("")
        self.tx_label.setText(f"{kind}: listening")

    def _leave_cw(self):
        self.cw_on = False
        self.cw = None
        if self.fm_plot is not None:
            self.fm_plot.hide()
        self.tx_label.setText("")
        self.decode_dock.setWindowTitle("Decodes")
        self.popout.setWindowTitle("Decodes")
        self._update_rate()

    def _on_cw(self, rec):
        kind = rec.get("kind")
        if kind == "char":                           # FM: a character complete while the transmission goes on
            if self.cw_kind == "FM" and self.fm_plot is not None:
                self.fm_plot.add_char(rec)
            return
        if kind == "drop":                           # the transmission was rejected: remove its characters
            if self.fm_plot is not None:
                self.fm_plot.remove_tx(rec["tx"])
            return
        if self.cw_kind == "FM":
            lines, path, tag = fm_lines(rec), self.fm_txt, "FM"
        else:
            lines, path, tag = [cw_line(rec)], self.cw_txt, "CW"
        for line in lines:
            self.decode_pane.append(line, False)
        if self.cw_kind == "FM" and self.fm_plot is not None:
            self.fm_plot.add_decode(rec, self.cw.rate)
        self.tx_label.setText(f"{tag} {rec['mhz']:.4f} MHz: {rec['text']}")
        try:
            with open(path, "a") as f:
                for line in lines:
                    f.write(line + "\n")
        except OSError as e:
            self.tx_label.setText(f"{tag}.TXT write failed: {e}")

    # ---- WWV timing mode: tuned to a WWV frequency with decoding off; separate from Tx Test
    def _enter_wwv(self, mhz):
        if self.track_on:
            self._leave_track()
        if self.cw_on:
            self._leave_cw()
        if self.an.skip_tx:
            self.blue_box.setCurrentIndex(self.blue_box.findData("peak"))
        self._set_mode(None)
        self.wwv_mhz = mhz
        self.wwv_on = True
        self.decode_dock.setWindowTitle("WWV timing")
        self.popout.setWindowTitle("WWV timing")
        self._select_mode_box("WWV")
        self._tune_window(mhz, mhz * 1e3 - WWV_SPAN_KHZ / 2, mhz * 1e3 + WWV_SPAN_KHZ / 2)
        self.rate_label.setText("")
        self.tx_label.setText(f"WWV {mhz:.3f} MHz: listening")

    def _leave_wwv(self):
        self.wwv_on = False
        if self.wwv is not None:
            self.wwv.close()
        self.wwv = None
        self.tx_label.setText("")
        self.decode_dock.setWindowTitle("Decodes")
        self.popout.setWindowTitle("Decodes")

    def _sync_wwv_box(self):
        """Tick the WWV submenu entry for the tuned frequency while WWV timing is on."""
        f = self.tune_hz / 1e6
        for mhz, a in self.wwv_actions:
            a.setChecked(self.wwv_on and abs(mhz - f) < 1e-6)

    def _on_wwv_minute(self, rec):
        mhz = self.tune_hz / 1e6
        self.decode_pane.append(wwv_line(rec, mhz), False)
        self.wwv_win.add(rec, mhz)
        car = rec["carrier"]
        cs = "" if car is None else f", carrier {car['hz']:+.2f} Hz, SNR {car['snr']:+.1f} dB"
        self.tx_label.setText(f"WWV {mhz:.3f} MHz: clock {wwv_ms(rec['clock_ms'])}{cs}")

    def _onset_log_path(self):
        return os.path.join(os.getcwd(), datetime.now().strftime("wwv_onsets_%Y%m%d_%H%M%S.csv"))

    def set_onset_log(self, on):
        self.onset_log_on = bool(on)
        if self.wwv is not None:
            if self.onset_log_on:
                self.wwv.start_onset_log(self._onset_log_path())
            else:
                self.wwv.stop_onset_log()

    def save_wwv_capture(self):
        mon = self.wwv
        events = list(mon.recent) if mon is not None else []
        clock = self.wwv_win.recent[-1] if self.wwv_win.recent else None
        return write_wwv_capture(self.ring, self.rate, self.tune_hz, events=events, clock_ms=clock)

    def show_wwv_stats(self):
        self.wwv_win.show()
        self.wwv_win.raise_()
        self.wwv_win.activateWindow()

    def _log_tx(self, r):
        self.decode_pane.append(tx_line(r, self.an.raw), False)

    def _log_noise(self, rep):
        self.decode_pane.append(tx_noise_line(rep), False)

    def _sync_check_btn(self, on):
        self.btn_check_noise.setEnabled(bool(on))

    def _ref_reset(self):
        self.an.tx.reset_reference()
        stamp = time.strftime("%H:%M:%S", time.gmtime())
        self.decode_pane.append(f"{stamp} UTC  reference reset, re-measuring background", False)

    def _check_noise(self):
        rep = self.an.tx.take_noise_report(reset=False)
        if rep is None:
            self.decode_pane.append("noise check: reference not ready yet", False)
            return
        self._log_noise(rep)

    def _clear_log(self):
        """Clear button: empties the lines shown in the decodes window (the CW, Track, Tx or FT8 log).
        Files on disk (ALL.TXT, CW.TXT, track CSV) are not touched."""
        for view in (self.decode_pane.view, self.decode_pane.view_cq, self.decode_pane.view_other):
            view.clear()

    def _clear_stats(self):
        """Clear button on the FT8 stats window: start the per-callsign totals again."""
        self.stats = {}

    # ---- IQ recording: armed from Settings; watches the displayed span, wherever the receiver is tuned
    def _set_iq(self, on):
        if on:
            self.iq = IqRecorder(self.ring, self.rate, os.getcwd())
            self._iq_window()
            self.msg.setText("IQ recorder armed: waiting for a signal above the background")
        else:
            lines = self.iq.close() if self.iq is not None else []
            self.iq = None
            self._iq_log(lines)
            self.msg.setText("IQ recorder off")

    def _iq_window(self):
        """Pass the tuning and displayed span to the recorder. A change restarts it."""
        if self.iq is not None:
            half = self.span_hz / 2
            self._iq_log(self.iq.retarget(self.tune_hz, self.disp_off - half, self.disp_off + half))

    def _iq_log(self, lines):
        for line in lines:
            self.decode_pane.append(line, False)
            self.msg.setText(line)

    # ---- Settings: FFT size, applied in place
    def _on_nfft_action(self, n):
        """The waterfall and the background average start again. Frequency, span, band and mode are kept."""
        if n == self.an.nfft:
            return
        self.an.set_nfft(n)
        self.row_s = self.an.row_s
        self.cfg_version += 1          # decodes still in flight were built on the old rows
        self._configure(settle=False)  # rebuilds the displayed bins, averages and the rate detector
        self._refresh()                # re-sizes the waterfall buffer and updates the status line

    def _update_tx_label(self):
        tx = self.an.tx
        unit = "(raw)" if self.an.raw else "(SNR)"
        if tx.active and tx.center is not None:
            pk = "--" if tx.tx_peak is None else f"{tx.tx_peak:+.1f}"
            text = (f"Tx in progress: {self.an.freq_khz[tx.center] / 1e3:.4f} MHz, "
                    f"peak {pk} dB {unit}")
        elif tx.last is not None:
            text = "Last Tx: " + tx_line(tx.last, self.an.raw)
        else:
            text = "Tx Test: listening"
        self.tx_label.setText(text)

    def _set_paused(self, on):
        self._flush_tracks()            # pausing or resuming ends any transmission in progress
        if self.iq is not None:
            self._iq_log(self.iq.restart("receiver stopped" if on else "receiver restarted"))
        self.paused = on
        self.btn_pause.setText("Resume" if on else "Pause")
        if on:
            self.radio.stop()
            self.msg.setText("Paused: receiver stopped")
            return
        self.cur_slot = None                    # no decode of the slot that straddles the pause
        self.an.next_pos = None                 # resync the analyzer with the ring
        self.an.tx.reset_events()
        if self.radio.start():
            self.msg.setText("")
        else:
            self.msg.setText("airspyhf_start() failed")

    def _reset_peak(self):
        if self.blue_mode == "avg":
            # restart the 15 s average from the live trace, shown at once even if paused
            self.an.restart_average()
            if self.an.spec_avg_db is not None:
                self.spec_avg_curve.setData(self.an.freq_khz, self.an.spec_avg_db)
        else:
            self.an.peak_db = None

    def _fit_spec_y(self, spec_db, blue):
        """Raw power: fit the spectrum axis to the traces. The floor is 5 dB below the 5th percentile
        of the live trace in the current span; the top sits 5 dB above the highest trace."""
        lo = math.floor(float(np.percentile(spec_db, 5))) - 5
        hi = math.ceil(float(np.max(spec_db)))
        if blue is not None:
            hi = max(hi, math.ceil(float(np.max(blue))))
        hi += 5
        if self.spec_y is None or abs(lo - self.spec_y[0]) > 3 or abs(hi - self.spec_y[1]) > 3:
            self.spec_y = (lo, hi)                      # small changes are ignored: no label jitter
            self.spec.setYRange(lo, hi, padding=0)

    # ---- control callbacks
    def retune(self, delta_hz):
        new = self.tune_hz + delta_hz
        if not (9e3 <= new <= 31e6 or 60e6 <= new <= 260e6):
            self.msg.setText("Out of tuning range (9 kHz-31 MHz or 60-260 MHz)")
            return
        if not self.radio.set_freq(new):
            self.msg.setText("airspyhf_set_freq() failed")
            return
        self.tune_hz = new
        self.manual_tuned = True
        self.cfg_version += 1
        self._configure(settle=True)
        self._refresh()
        self.msg.setText("")

    def on_span(self, _idx):
        self.manual_tuned = True
        data = self.span_box.currentData()
        khz = self.rate / 1e3 if data is None else float(data)
        khz = min(khz, self.rate / 1e3)
        self.span_hz = khz * 1e3
        self.cfg_version += 1
        self._configure(settle=False)
        self._refresh()
        self.msg.setText("")

    def save_png(self):
        path = os.path.join(os.getcwd(), datetime.now().strftime("airspy_%Y%m%d_%H%M%S.png"))
        pix = self.grab()
        if self.spec_pop.isVisible():
            # the spectrum readout is a separate top-level window, which grab() leaves out
            p = QtGui.QPainter(pix)
            self.spec_pop.render(p, self.spec_pop.pos() - self.mapToGlobal(QtCore.QPoint(0, 0)))
            p.end()
        if pix.save(path, "PNG"):
            self.msg.setText(f"Saved {path}")
        else:
            self.msg.setText("Save failed")

    def save_trace_csv(self):
        """Save the blue trace as currently displayed (15 s average or peak hold) as a CSV file."""
        x, y = self.spec_avg_curve.getData()
        if x is None or y is None or len(x) == 0:
            self.msg.setText("No trace to save yet")
            return
        kind = "avg" if self.blue_mode == "avg" else "peak"
        unit = "power" if self.an.raw else "snr"
        path = os.path.join(os.getcwd(), datetime.now().strftime(f"airspy_{kind}_%Y%m%d_%H%M%S.csv"))
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frequency_kHz", f"{kind}_{unit}_dB"])
                for xv, yv in zip(x, y):
                    w.writerow([f"{xv:.4f}", f"{yv:.2f}"])
        except OSError as e:
            self.msg.setText(f"CSV save failed: {e}")
            return
        self.msg.setText(f"Saved {path}")

    def _set_marks_visible(self, on):
        self.show_marks = on
        if not on and self.ov is not None:
            self.ov.fill(0)
        self.dirty = True

    def _set_calls_visible(self, on):
        self.show_call = on
        self.dirty = True

    # ---- hover popup
    def eventFilter(self, obj, ev):
        if obj is self.wf_plot.viewport():
            if ev.type() == QtCore.QEvent.Type.Leave and self.pop.isVisible():
                self.pop_hide.start()
            elif (ev.type() == QtCore.QEvent.Type.MouseButtonPress
                  and ev.button() == QtCore.Qt.MouseButton.LeftButton
                  and self.pop.isVisible() and self.pop_mark is not None):
                sp = self.wf_plot.mapToScene(ev.position().toPoint())
                p = self.wf_plot.getViewBox().mapSceneToView(sp)
                if self._mark_at(p.x(), p.y()) is self.pop_mark:
                    QtGui.QDesktopServices.openUrl(
                        QtCore.QUrl(f"https://www.qrz.com/db/{self.pop_mark['call']}"))
        if obj is self.spec.viewport():
            if ev.type() == QtCore.QEvent.Type.Leave:
                self._hide_spec_readout()
        return super().eventFilter(obj, ev)

    def _hide_spec_readout(self):
        self.spec_pop.hide()
        self.spec_marker.setVisible(False)

    def _on_spec_hover(self, scene_pos):
        vb = self.spec.getViewBox()
        x_data, y_data = self.spec_avg_curve.getData()
        if (x_data is None or len(x_data) == 0 or y_data is None
                or not vb.sceneBoundingRect().contains(scene_pos)):
            self._hide_spec_readout()
            return
        p = vb.mapSceneToView(scene_pos)
        x0, x1 = vb.viewRange()[0]
        khz_per_px = (x1 - x0) / max(vb.width(), 1)
        idx = np.flatnonzero(np.abs(x_data - p.x()) <= SPEC_SNAP_PX * khz_per_px)
        if idx.size == 0:
            self._hide_spec_readout()
            return
        i = idx[int(np.argmax(y_data[idx]))]
        unit = "(raw)" if self.an.raw else "(SNR)"
        self.spec_pop.setText(f"{x_data[i] / 1000.0:.4f} MHz   {y_data[i]:+.1f} dB {unit}")
        self.spec_marker.setData([x_data[i]], [y_data[i]])
        self.spec_marker.setVisible(True)
        self.spec_pop.adjustSize()
        self.spec_pop.move(QtGui.QCursor.pos() + QtCore.QPoint(18, 18))
        self.spec_pop.show()

    def _mark_at(self, x_khz, y_s):
        """Most recent decoded box with a callsign under the given plot coordinates, or None."""
        if self.newest_s is None or not self.rows:
            return None
        j = y_s / self.row_s + self.rows                # fractional buffer row, as in _row_index
        for m in reversed(self.marks):
            if not m.get("call") or not m["x0"] <= x_khz <= m["x0"] + m["w"]:
                continue
            if self._row_index(m["s0"]) <= j < self._row_index(m["s1"]) + 1:
                return m
        return None

    def _on_hover(self, scene_pos):
        vb = self.wf_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return
        p = vb.mapSceneToView(scene_pos)
        m = self._mark_at(p.x(), p.y())
        if m is None:
            if self.pop.isVisible() and not self.pop_hide.isActive():
                self.pop_hide.start()
            return
        self.pop_hide.stop()
        if m is not self.pop_mark:
            self.pop_mark = m
            self.pop_html = None
            colour = "#b7410e" if m["cq"] else "#ffffff"
            self.pop.setStyleSheet(f"background-color: rgba(0, 0, 0, 220); "
                                   f"border: 1px solid {colour}; color: {colour};")
        html = self._popup_html(m)
        if html != self.pop_html:                       # refreshed in case a grid arrived since
            self.pop_html = html
            self.pop.setText(html)
            self.pop.adjustSize()
        self.pop.move(QtGui.QCursor.pos() + QtCore.QPoint(18, 18))
        self.pop.show()

    def _popup_html(self, m):
        grid = self.grids.get(m["call"])
        lines = [f"<div style='font-size:18pt; font-weight:bold'>{m['call']}</div>"]
        if grid:
            miles = grid_distance_miles(MY_GRID, grid)
            lines.append(f"<div>Grid {grid}, {miles:.0f} mi</div>")
        lines.append(f"<div>SNR {m['snr']:+d} dB</div>")
        lines.append(f"<div>Offset {m['offset']} Hz</div>")
        return "".join(lines)

    def _hide_popup(self):
        self.pop.hide()
        self.pop_mark = None
        self.pop_html = None

    def _drop_label(self, m):
        item = m.pop("label", None)
        if item is not None:
            self.wf_plot.removeItem(item)

    def _sync_call_labels(self):
        """Place each sender's callsign in the middle of its signal box; hidden with Mark or Call off.
        Where two labels would overlap, one moves up and the other down by half a label height."""
        show = self.show_marks and self.show_call
        fm = QtGui.QFontMetrics(QtGui.QFont("sans-serif", 8))
        lab_h = fm.height()                                 # rows: the waterfall has one row per pixel
        px_per_khz = self.wf_plot.getViewBox().width() / max(self.f1 - self.f0, 1e-9)
        entries = []
        for m in self.marks:
            if not show or not m.get("call") or self.newest_s is None:
                self._drop_label(m)
                continue
            r0 = self._row_index(m["s0"])
            r1 = self._row_index(m["s1"])
            lo, hi = max(r0, 0), min(r1, self.rows - 1)     # visible part of the box
            if lo > hi:
                self._drop_label(m)
                continue
            item = m.get("label")
            if item is None:
                item = pg.TextItem(m["call"], color=MARK_RGBA_CQ[:3] if m["cq"] else (200, 200, 200),
                                   anchor=(0.5, 0.5), fill=pg.mkBrush(0, 0, 0, 170))
                item.setFont(QtGui.QFont("sans-serif", 8))
                self.wf_plot.addItem(item)
                m["label"] = item
            cx_khz = m["x0"] + m["w"] / 2
            entries.append({"item": item, "cx_khz": cx_khz,
                            "x_px": (cx_khz - self.f0) * px_per_khz,
                            "w_px": fm.horizontalAdvance(m["call"]) + 6,
                            "row": (lo + hi) / 2 + 0.5, "shift": 0.0})
        for i, b in enumerate(entries):
            for a in entries[:i]:
                if a["shift"] or b["shift"]:               # each label moves at most once
                    continue
                if (abs(a["x_px"] - b["x_px"]) < (a["w_px"] + b["w_px"]) / 2
                        and abs(a["row"] - b["row"]) < lab_h):
                    half = lab_h / 2
                    up, down = (a, b) if a["row"] >= b["row"] else (b, a)
                    up["shift"], down["shift"] = half, -half
        for e in entries:
            e["item"].setPos(e["cx_khz"], (e["row"] + e["shift"] - self.rows) * self.row_s)

    # ---- FT8 slot handling
    def _index_at(self, now, boundary):
        return self.ring.total - int((now - boundary) * self.rate)

    def _start_decode(self, slot_time, start_idx, cfg_ver):
        stamp = time.strftime("%H:%M:%S", time.gmtime(slot_time))
        if cfg_ver != self.cfg_version:
            self.msg.setText(f"Slot {stamp} UTC skipped: window changed during slot")
            return
        if self.decoding:
            self.msg.setText(f"Slot {stamp} UTC skipped: previous decode still running")
            return
        z = self.ring.read(start_idx, int(round(MODES[self.mode]["slot_s"] * self.rate)))
        if z is None:
            self.msg.setText(f"Slot {stamp} UTC skipped: samples overwritten")
            return
        self.decoding = True
        cfg = (self.tune_hz, self.disp_off, self.span_hz, self.mode)
        threading.Thread(target=self._decode_thread, args=(z, slot_time, cfg, start_idx),
                         daemon=True).start()

    def _decode_thread(self, z, slot_time, cfg, start_idx):
        tune_hz, disp_off, span_hz, mode = cfg
        try:
            recs = run_decode(z, self.rate, slot_time, tune_hz, disp_off, span_hz,
                              self.jt9_path, self.shm_dir, mode)
            for rec in recs:
                rec["s0"] = start_idx              # sample index of the slot start
            stamp = time.strftime("%H:%M:%S", time.gmtime(slot_time))
            self.bridge.count.emit({"s0": start_idx, "mode": mode, "n": len(recs)})
            self.bridge.status.emit(f"Slot {stamp} UTC: {len(recs)} decodes")
            for rec in recs:
                self.bridge.decode.emit(rec)
        except Exception as e:
            self.bridge.status.emit(f"Decode error: {e}")
        finally:
            self.bridge.done.emit()

    def _on_decode(self, rec):
        is_cq = rec["msg"].startswith("CQ ")
        self.decode_pane.append(rec["line"], is_cq)
        if rec["call"] and rec["grid"]:
            self.grids[rec["call"]] = rec["grid"]          # last grid heard from each call, this session
        self._update_stats(rec, is_cq)
        if self.decode_t0 is not None and rec["mode"] == self.mode:
            self.decode_times.append(rec["slot"])
        try:
            with open(self.all_txt, "a") as f:
                f.write(rec["line"] + "\n")
        except OSError as e:
            self.msg.setText(f"ALL.TXT write failed: {e}")
        self.marks.append({
            "s0": rec["s0"],
            "s1": rec["s0"] + int(round(MODES[rec["mode"]]["slot_s"] * self.rate)),
            "cq": is_cq,
            "call": rec["call"],
            "snr": rec["snr"],
            "offset": rec["offset"],
            "x0": rec["rf_hz"] / 1e3,
            "w": MODES[rec["mode"]]["tone_bw"] / 1e3,
        })
        if self.newest_s is not None and self.rows:      # forget boxes scrolled off, even when hidden
            keep = []
            for old in self.marks:
                if self._row_index(old["s1"]) >= 0:
                    keep.append(old)
                else:
                    self._drop_label(old)
            self.marks = keep
        self.dirty = True

    def _update_stats(self, rec, is_cq):
        """Add one decode to the per-callsign totals shown in the Stats window."""
        call = rec["call"]
        if not call or rec["mode"] != self.stats_mode:   # unidentified sender, or a leftover from the old mode
            return
        snr, slot = rec["snr"], rec["slot"]
        s = self.stats.get(call)
        if s is None:
            s = self.stats[call] = {"first": slot, "last": slot, "total": 0, "cq": 0,
                                    "snr_min": snr, "snr_max": snr, "snr_sum": 0, "grid": None,
                                    "rpt_to": set()}
        s["first"] = min(s["first"], slot)
        s["last"] = max(s["last"], slot)
        s["total"] += 1
        s["cq"] += int(is_cq)
        s["snr_min"] = min(s["snr_min"], snr)
        s["snr_max"] = max(s["snr_max"], snr)
        s["snr_sum"] += snr
        if rec["grid"]:
            s["grid"] = rec["grid"]
        addressee = report_to(rec["msg"])
        if addressee:
            s["rpt_to"].add(addressee)
        self.stats_win.mark_dirty()

    def show_stats(self):
        self.stats_win.rebuild()
        self.stats_win.show()
        self.stats_win.raise_()
        self.stats_win.activateWindow()

    def _on_decode_status(self, text):
        self.msg.setText(text)

    def _on_decode_done(self):
        self.decoding = False
        self._update_rate()     # once per FT8 slot, after that slot's decodes have been counted

    # ---- decode count strip beside the waterfall
    def _set_count_strip(self, on):
        self.count_plot.setVisible(on)
        self.count_spacer.setVisible(on)

    def _clear_slot_counts(self):
        for e in self.slot_counts:
            if e["item"] is not None:
                self.count_plot.removeItem(e["item"])
        self.slot_counts = []

    def _on_slot_count(self, info):
        if info["mode"] != self.mode:           # slot decoded under the previous mode
            return
        span = int(round(MODES[info["mode"]]["slot_s"] * self.rate))
        self.slot_counts.append({"s0": info["s0"], "s1": info["s0"] + span,
                                 "n": info["n"], "item": None})
        self.dirty = True

    def _sync_count_labels(self):
        """Centre each slot's decode count on that slot's rows in the strip."""
        if self.newest_s is None or not self.rows:
            return
        keep = []
        for e in self.slot_counts:
            r0 = self._row_index(e["s0"])
            r1 = self._row_index(e["s1"])
            if r1 < 0:                          # scrolled off the top: forget it
                if e["item"] is not None:
                    self.count_plot.removeItem(e["item"])
                continue
            keep.append(e)
            if e["item"] is None:
                item = pg.TextItem(str(e["n"]), color=(200, 200, 200), anchor=(0.5, 0.5),
                                   fill=pg.mkBrush(0, 0, 0, 170))
                item.setFont(QtGui.QFont("sans-serif", 9))
                self.count_plot.addItem(item)
                e["item"] = item
            lo, hi = max(r0, 0), min(r1, self.rows - 1)
            e["item"].setPos(0.5, ((lo + hi) / 2 + 0.5 - self.rows) * self.row_s)
        self.slot_counts = keep

    # ---- waterfall
    def _realloc(self, h):
        """Make the waterfall exactly h rows, one row per screen pixel."""
        self.rows = h
        ncols = len(self.an.freq_khz)
        self.wf = np.full((h, ncols), -1e9, dtype=np.float32)
        self.ov = np.zeros((h, ncols), dtype=np.uint8)
        self.valid_rows = 0
        self.wf_plot.setYRange(-h * self.row_s, 0, padding=0)
        self.dirty = True

    def _compress(self, x, knee):
        """Linear up to the calibrated top, then logarithmic so strong signals keep their gradient."""
        K = self.compress
        if K <= 0:
            return x
        over = np.maximum(x - knee, 0.0)
        return np.where(x <= knee, x, knee + K * np.log1p(over / K))

    def _row_index(self, s):
        """Buffer row holding the waterfall row that starts at sample s (newest row = bottom)."""
        return self.rows - 1 - int(round((self.newest_s - s) / self.an.hop))

    def _paint_marks(self):
        """Write corner brackets for each decode into the overlay codes, in row/column space."""
        h = self.rows
        freq = self.an.freq_khz
        ncols = len(freq)
        keep = []
        for m in self.marks:
            r0 = self._row_index(m["s0"])
            r1 = self._row_index(m["s1"])
            if r1 < 0:                              # scrolled off the top: forget it
                self._drop_label(m)
                continue
            keep.append(m)
            c0 = int(np.searchsorted(freq, m["x0"], side="left"))
            c1 = int(np.searchsorted(freq, m["x0"] + m["w"], side="right"))
            c1 = max(c1, c0 + 1)
            c0 = min(c0, ncols - 1)
            c1 = min(c1, ncols)
            if c0 >= c1:
                continue
            flag = 2 if m["cq"] else 1
            arm_c = max(2, int(0.4 * (c1 - c0)))   # horizontal arm, columns
            arm_r = max(2, int(0.2 * (r1 - r0 + 1)))   # vertical arm, rows

            # CQ outlines are 2 px: each stroke also paints the pixel just inside it
            t = 1 if m["cq"] else 0
            for k in range(t + 1):
                # vertical arms along the left and right edges, clipped to the buffer
                for col in (c0 + k, c1 - 1 - k):
                    if not 0 <= col < ncols:
                        continue
                    ta, tb = max(r0, 0), min(r0 + arm_r, h)
                    ba, bb = max(r1 - arm_r + 1, 0), min(r1 + 1, h)
                    if ta < tb:
                        self.ov[ta:tb, col] = flag
                    if ba < bb:
                        self.ov[ba:bb, col] = flag
                # horizontal arms at the top and bottom corners
                for row in (r0 + k, r1 - k):
                    if 0 <= row < h:
                        self.ov[row, c0:c0 + arm_c] = flag
                        self.ov[row, c1 - arm_c:c1] = flag
        self.marks = keep

    def _compose(self):
        """Build one RGBA image of waterfall + overlay and hand it to the image item."""
        lo, knee = self.levels
        top = float(self._compress(knee + HEADROOM_DB, knee))
        comp = self._compress(self.wf, knee)
        scale = max(top - lo, 1e-9)
        idx = np.clip((comp - lo) / scale * 255.0, 0, 255).astype(np.uint8)
        rgba = self.lut_rgba[idx]                   # (rows, cols, 4) uint8
        if self.show_marks and self.ov is not None:
            rgba[self.ov == 1] = MARK_RGBA_OTHER
            rgba[self.ov == 2] = MARK_RGBA_CQ
        if self.show_marks and self.mode is not None and self.newest_s is not None:
            self._paint_slot_lines(rgba)
        self.img.setImage(np.ascontiguousarray(rgba.transpose(1, 0, 2)), autoLevels=False)
        self.img.setRect(QtCore.QRectF(self.f0, -self.rows * self.row_s,
                                       self.f1 - self.f0, self.rows * self.row_s))

    def _paint_slot_lines(self, rgba):
        """Draw one row across the waterfall at each decode slot boundary (FT8 15 s, FT4 7.5 s)."""
        slot_s = MODES[self.mode]["slot_s"]
        now = time.time()
        b = math.ceil((now - self.rows * self.row_s) / slot_s) * slot_s
        while b <= now:
            r = self._row_index(self._index_at(now, b))
            if 0 <= r < self.rows:
                rgba[r, :] = SLOT_LINE_RGBA
            b += slot_s

    def tick(self):
        sec = int(time.time())
        if sec != self.clock_sec:
            self.clock_sec = sec
            self.utc_label.setText("UTC " + time.strftime("%H:%M:%S", time.gmtime(sec)))
        if self.paused:
            return
        h = int(self.wf_plot.getViewBox().height())
        if h > 10 and h != self.rows:
            self._realloc(h)

        # decode slot boundaries: UTC multiples of the mode's cycle (15 s FT8, 7.5 s FT4)
        now = time.time()
        slot_s = MODES[self.mode]["slot_s"] if self.mode is not None else 15.0
        slot_b = int(now // slot_s) * slot_s
        if self.cur_slot is None:
            self.cur_slot = (slot_b, self._index_at(now, slot_b), self.cfg_version)
        elif slot_b != self.cur_slot[0]:
            prev_b, prev_idx, prev_ver = self.cur_slot
            if self.mode is not None:
                self._start_decode(prev_b, prev_idx, prev_ver)
            self.cur_slot = (slot_b, self._index_at(now, slot_b), self.cfg_version)

        rows_db, spec_db = self.an.process()
        if rows_db is not None:
            self.newest_wall = time.time() - self.an.nfft / (2.0 * self.rate)
            self._update_wf_clock()
        if self.track_on and self.trk is not None and rows_db is not None:
            self._feed_track(rows_db)
        if self.wwv_on and self.wwv is not None:
            for rec in self.wwv.process():
                self._on_wwv_minute(rec)
        if self.cw_on and self.cw is not None:
            for rec in self.cw.process():
                self._on_cw(rec)
            if self.cw_kind == "FM" and self.fm_plot is not None and self.fm_plot.isVisible():
                self.fm_plot.update_from(self.cw, self.ring.total)
        if self.iq is not None:
            self._iq_log(self.iq.process())
        if self.an.skip_tx:
            self._update_tx_label()
            blk = int(time.time() // NOISE_REPORT_S)
            if blk != self.noise_blk:
                self.noise_blk = blk
                rep = self.an.tx.take_noise_report()
                if rep is not None and self.chk_noise.isChecked():
                    self._log_noise(rep)
            res = self.an.tx.new_result
            if res is not None:
                self.an.tx.new_result = None
                self._log_tx(res)
        if rows_db is not None and self.wf is not None:
            k = min(rows_db.shape[0], self.rows)
            if k < self.rows:
                self.wf[:-k] = self.wf[k:]
                self.ov[:-k] = self.ov[k:]          # overlay scrolls with the pixels
            self.wf[-k:] = rows_db[-k:]
            self.ov[-k:] = 0
            self.newest_s = self.an.next_pos - self.an.hop
            self.valid_rows = min(self.valid_rows + k, self.rows)

            if self.levels is None and self.valid_rows >= LEVELS_MIN_ROWS:
                v = self.wf[-self.valid_rows:][::2, ::4]     # valid data only
                lo, hi = np.percentile(v, [5, 99.5])
                hi = max(float(hi), float(lo) + 1.0)
                self.levels = (float(lo), hi)                # frozen after this

            self.spec_curve.setData(self.an.freq_khz, spec_db)
            blue = self.an.peak_db if self.blue_mode in ("peak", "txskip") else self.an.spec_avg_db
            if blue is not None:
                self.spec_avg_curve.setData(self.an.freq_khz, blue)
            if self.an.raw:
                self._fit_spec_y(spec_db, blue)
            self.dirty = True

        # One image update per change: waterfall pixels and marks together
        if self.dirty and self.wf is not None and self.levels is not None and self.valid_rows > 0:
            if self.show_marks and self.newest_s is not None and self.marks:
                self._paint_marks()
            self._compose()
            self._sync_call_labels()
            self._sync_count_labels()
            self.wf_plot.getAxis("left").picture = None     # UTC ticks move with the rows
            self.wf_plot.getAxis("left").update()
            self.dirty = False

# ------------------------------------------------------------ main

SETTINGS_NAME = "airspy_live_settings.json"      # saved on exit and read on start, in the folder PNG images go to


def load_settings():
    """Settings saved by the last run, read from the working directory. {} if there is no usable file."""
    path = os.path.join(os.getcwd(), SETTINGS_NAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"warning: ignoring {path} ({e})", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"warning: ignoring {path} (not a settings object)", file=sys.stderr)
        return {}
    return data


def main():
    ap = argparse.ArgumentParser(description="Airspy HF+ live spectrum/waterfall with FT8 decoding")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    ap.add_argument("-f", "--freq", type=float, default=None,
                    help=f"tuning frequency in MHz (default: --center, or {DEFAULT_FREQ_MHZ})")
    ap.add_argument("-c", "--center", type=float, default=None,
                    help="display center frequency in MHz (default: tuning frequency)")
    ap.add_argument("--span", type=float, default=None,
                    help="display width in kHz, centered on --center "
                         "(default: full bandwidth at the chosen rate)")
    ap.add_argument("--lo", type=float,
                    help="display lower edge, absolute kHz (alternative to --center/--span)")
    ap.add_argument("--hi", type=float, help="display upper edge, absolute kHz")
    ap.add_argument("--rate", type=int, default=None,
                    help=f"sample rate in S/s; must be one the device reports (default {DEFAULT_RATE})")
    ap.add_argument("--nfft", type=int, default=None,
                    help=f"FFT size, power of 2 (default {DEFAULT_NFFT} resolves FT8 tones at 192 kS/s)")
    ap.add_argument("-g", "--agc", choices=["on", "off"], default="on")
    ap.add_argument("-l", "--agc-threshold", choices=["low", "high"], default="low")
    ap.add_argument("-t", "--att", type=int, default=0,
                    help="HF attenuator 0..8, 6 dB steps (AGC off only)")
    ap.add_argument("-m", "--lna", choices=["on", "off"], default="off")
    ap.add_argument("--vmin", type=float,
                    help="colour scale minimum in dB (give with --vmax to fix the scale)")
    ap.add_argument("--vmax", type=float, help="colour scale maximum in dB")
    ap.add_argument("--compress", type=float, default=6.0,
                    help="compression scale in dB above the colour top "
                         "(0 = linear; larger = less compression)")
    ap.add_argument("--jt9", default="jt9", help="path to the jt9 decoder (default: jt9 on PATH)")
    shm_default = ("/dev/shm/airspy_ft8" if os.path.isdir("/dev/shm")
                   else os.path.join(tempfile.gettempdir(), "airspy_ft8"))
    ap.add_argument("--shm-dir", default=shm_default,
                    help=f"directory for temporary decoder WAVs (default {shm_default})")
    ap.add_argument("--no-decode", action="store_true", help="start with FT8 decoding off")
    args = ap.parse_args()

    # Saved frequency and window are used unless one was given on the command line; --nfft falls back
    # to the saved size. Mode and display settings are applied once the window exists.
    saved = load_settings()

    def saved_num(key):
        v = saved.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    window_given = any(v is not None for v in (args.freq, args.center, args.span, args.lo, args.hi))
    if not window_given:
        args.freq = saved_num("frequency_mhz")
        args.center = saved_num("center_mhz")
        args.span = saved_num("span_khz")
    if args.nfft is None:
        n = int(saved_num("fft_size") or 0)
        args.nfft = n if n >= 256 and n & (n - 1) == 0 else DEFAULT_NFFT

    freq_given = args.freq is not None
    if args.freq is None:
        args.freq = args.center if args.center is not None else DEFAULT_FREQ_MHZ
    center_mhz = args.center if args.center is not None else args.freq

    # With no tuning or display options at all, use the default 14074-14077 kHz window.
    if not freq_given and all(v is None for v in (args.center, args.span, args.lo, args.hi)):
        args.lo, args.hi = DEFAULT_LO_KHZ, DEFAULT_HI_KHZ

    if not (9e3 <= args.freq * 1e6 <= 31e6 or 60e6 <= args.freq * 1e6 <= 260e6):
        ap.error("frequency must be 9 kHz-31 MHz or 60-260 MHz")
    if args.nfft < 256 or args.nfft & (args.nfft - 1):
        ap.error("--nfft must be a power of 2, at least 256")
    if not 0 <= args.att <= 8:
        ap.error("--att must be 0..8")
    if (args.vmin is None) != (args.vmax is None):
        ap.error("--vmin and --vmax must be given together")
    if (args.lo is None) != (args.hi is None):
        ap.error("--lo and --hi must be given together")
    if args.lo is not None and (args.center is not None or args.span is not None):
        ap.error("use either --lo/--hi or --center/--span, not both")
    if args.lo is not None and args.hi <= args.lo:
        ap.error("--hi must be greater than --lo")

    # Decoder prerequisites
    decode_ok = True
    if shutil.which(args.jt9) is None:
        print(f"warning: decoder '{args.jt9}' not found; FT8 decoding disabled", file=sys.stderr)
        decode_ok = False
    try:
        os.makedirs(args.shm_dir, exist_ok=True)
    except OSError as e:
        print(f"warning: cannot use {args.shm_dir} ({e}); FT8 decoding disabled", file=sys.stderr)
        decode_ok = False
    all_txt = os.path.join(os.getcwd(), "ALL.TXT")

    lib = load_lib()
    dev = ctypes.c_void_p()
    if lib.airspyhf_open(ctypes.byref(dev)) != 0:
        sys.exit("airspyhf_open() failed")

    try:
        rates = get_sample_rates(lib, dev)
        rate = args.rate if args.rate is not None else (
            DEFAULT_RATE if DEFAULT_RATE in rates else max(rates))
        if rate not in rates:
            ap.error(f"--rate {rate} not supported; device reports {sorted(rates)}")

        tune_hz = round(args.freq * 1e6)
        if args.lo is not None:
            span_hz = (args.hi - args.lo) * 1e3
            disp_off = (args.lo + args.hi) / 2 * 1e3 - tune_hz
        else:
            span_khz = args.span if args.span is not None else rate / 1e3
            span_hz = span_khz * 1e3
            disp_off = round(center_mhz * 1e6) - tune_hz
        if not (0 < span_hz <= rate):
            ap.error(f"span must be between 0 and {rate / 1e3:g} kHz at this rate")
        if not (disp_off - span_hz / 2 < rate / 2 and disp_off + span_hz / 2 > -rate / 2):
            ap.error("display window does not overlap the receiver bandwidth")

        if lib.airspyhf_set_samplerate(dev, rate) != 0:
            sys.exit(f"airspyhf_set_samplerate({rate}) failed")
        agc_on = args.agc == "on"
        lib.airspyhf_set_hf_agc(dev, 1 if agc_on else 0)
        if agc_on:
            lib.airspyhf_set_hf_agc_threshold(dev, 1 if args.agc_threshold == "high" else 0)
        else:
            lib.airspyhf_set_hf_att(dev, args.att)
        lib.airspyhf_set_hf_lna(dev, 1 if args.lna == "on" else 0)

        ring = RingBuffer(int(RING_SECONDS * rate))
        rx = Receiver(ring)
        if lib.airspyhf_start(dev, rx.cb, None) != 0:
            sys.exit("airspyhf_start() failed")
        if lib.airspyhf_set_freq(dev, tune_hz) != 0:
            sys.exit("airspyhf_set_freq() failed")
        low_if = lib.airspyhf_is_low_if(dev) == 1

        an = Analyzer(ring, rate, args.nfft)
        radio = Radio(lib, dev)
        radio.cb = rx.cb       # kept so the receiver can be restarted after a pause
        app = QtWidgets.QApplication(sys.argv)
        win = MainWindow(an, radio, ring, low_if,
                         title=f"Airspy HF+  {args.freq:.6f} MHz",
                         tune_hz=tune_hz, disp_off=disp_off, span_hz=span_hz,
                         jt9_path=args.jt9, shm_dir=args.shm_dir, all_txt=all_txt,
                         decode_ok=decode_ok, decode_start=not args.no_decode,
                         vmin=args.vmin, vmax=args.vmax, compress=args.compress)
        win.show()
        if saved:
            win.apply_settings(saved, decode_ok and not args.no_decode)
        app.exec()
    finally:
        lib.airspyhf_stop(dev)
        lib.airspyhf_close(dev)


if __name__ == "__main__":
    main()
    
