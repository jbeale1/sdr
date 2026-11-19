#!/usr/bin/env python3

"""Live spectrum + waterfall for the Airspy HF+ with live FT8 decoding via WSJT-X's jt9.

Requires: numpy, scipy, PySide6, pyqtgraph, libairspyhf, and jt9 on PATH.
Decoder WAV files are written to /dev/shm and deleted after each decode.
Decodes are appended to ALL.TXT in the current directory and shown in a pane at the
bottom of the window, or in a separate window after "Pop out". CQ decodes are
highlighted there and can be marked on the waterfall.

A WWV timing mode logs the clock offset of the receiver and the carrier frequency.

-J.Beale N8JUF  19-Nov-2026

Examples:
    ./airspy_live.py                              # 14.074 MHz, 14074-14077 kHz, 192 kS/s
    ./airspy_live.py -f 14.074 --span 3 --rate 192000
    ./airspy_live.py --no-decode                  # display only
"""
import argparse
import csv
import ctypes
import ctypes.util
import io
import json
import os
import math
import re
import shutil
import subprocess
import tempfile
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime, timezone

import numpy as np
import scipy.fft as sfft
import scipy.signal as ssig
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

VERSION = "0.18.7"
DEFAULT_FREQ_MHZ = 14.074
DEFAULT_LO_KHZ = 14074.0        # default display window when no display option is given
DEFAULT_HI_KHZ = 14077.0
DEFAULT_RATE = 192_000
DEFAULT_NFFT = 131072
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
    ("500 Hz", 0.5),
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
WWV_TONE_HALF_BW_HZ = 80.0
WWV_DEC_RATE = 8000                          # envelope rate after decimation, S/s
WWV_LP_HZ = 3500.0                           # envelope lowpass before decimation
WWV_DETECT_DB = 15.0                         # tone onset: this far above the median envelope
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
WWV_TICK_S = (0.002, 0.05)                   # 5 ms seconds ticks
WWV_MARKER_S = (0.5, 1.2)                    # 800 ms minute and hour markers
WWV_CARRIER_S = 10.0                         # carrier frequency and strength measured over this span
WWV_CAPTURE_S = 10.0               # length of the raw IQ and envelope capture
WWV_CARRIER_SEARCH_HZ = 3000.0
WWV_SETTLE_S = 4.0                           # a minute is reported this long after it ends
WWV_MAP_LAG_S = 1.0                          # ring samples map to wall time one second later: markers then fall on second 0
WWV_HIST_S = 130.0                           # envelope history kept for the per-minute 100 Hz analysis
WWV_FRAME_BAND_HZ = (90.0, 110.0)           # band of the 100 Hz time-code subcarrier in the envelope
WWV_FRAME_EDGE_MS = 30.0                    # time-code pulses start 30 ms into the second (NIST SP 250-67)


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
        n = max(1, int(round(TX_SMOOTH_KHZ / dk)))
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
            f"tones {wwv_tones(rec)}")


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
                                      "marker_quiet", "marker_db"])
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
            _opt(rec["marker_cov"], 2), _opt(rec["marker_quiet"], 2), _opt(rec["marker_db"], 1)])
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
        self.alpha = min(1.0, self.row_s / max(self.avg_s, self.row_s)) if on else 1.0

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

    def __init__(self, source):
        super().__init__(None, QtCore.Qt.Window)
        self.source = source            # callable returning (stats dict, mode)
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

    COLUMNS = ["UTC", "Clock (ms)", "Marks", "Tick med (ms)", "Marker (ms)", "100 Hz edge (ms)",
               "100 Hz contrast", "Carrier (Hz)",
               "Carrier (ppm)", "Carrier SNR (dB)", "Tone (dB)", "Tones"]
    MAX_ROWS = 1440

    def __init__(self, capture_fn=None, onset_log_fn=None):
        super().__init__(None, QtCore.Qt.Window)
        self.capture_fn = capture_fn
        self.onset_log_fn = onset_log_fn
        self.setWindowTitle("WWV timing")
        self.resize(1040, 480)
        self.recent = deque(maxlen=10)           # clock offsets of the latest minutes
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
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.summary, 1)
        top.addWidget(btn_copy)
        top.addWidget(btn_save)
        top.addWidget(btn_cap)
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

    def add(self, rec, mhz):
        car = rec["carrier"]
        vals = [
            time.strftime("%H:%M", time.gmtime(rec["wall"])),
            wwv_ms(rec["clock_ms"]),
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
        ]
        self.table.insertRow(0)
        for col, v in enumerate(vals):
            self.table.setItem(0, col, QtWidgets.QTableWidgetItem(v))
        while self.table.rowCount() > self.MAX_ROWS:
            self.table.removeRow(self.table.rowCount() - 1)
        if rec["clock_ms"] is not None:
            self.recent.append(rec["clock_ms"])
        if self.recent:
            r = list(self.recent)
            cs = ("" if car is None else
                  f"   Carrier {car['hz']:+.2f} Hz ({car['ppm']:+.3f} ppm), SNR {car['snr']:+.1f} dB")
            self.summary.setText(
                f"{mhz:.3f} MHz   Clock last {r[-1]:+.1f} ms, median of last {len(r)} min "
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
        self.onset_log_on = False       # write the onset CSV (debugging); set from the WWV window
        self.wwv_mhz = WWV_DEFAULT_MHZ  # WWV frequency last chosen from the menu

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

        self.wf_plot = pg.PlotWidget(axisItems={"top": AudioOffsetAxis()})
        self.wf_plot.setLabel("bottom", "Frequency", units="kHz")
        self.row_s = an.hop / self.rate             # seconds per waterfall row
        self.wf_plot.setLabel("left", "Time relative to now", units="s")
        self.wf_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.off_axis = self.wf_plot.getAxis("top")
        self.wf_plot.setLabel("top", "Audio offset", units="Hz")
        self.wf_plot.showAxis("top")
        self.wf_plot.setXLink(self.spec)
        self.wf_plot.setMouseEnabled(x=False, y=False)
        self.img = pg.ImageItem()
        self.wf_plot.addItem(self.img)
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
        self.btn_save = QtWidgets.QPushButton("Save PNG")
        self.btn_popout = QtWidgets.QPushButton("Pop out")
        self.btn_marks = QtWidgets.QPushButton("Mark")
        self.btn_marks.setCheckable(True)
        self.btn_marks.setChecked(True)                 # on by default
        self.show_marks = True
        self.btn_calls = QtWidgets.QPushButton("Call")
        self.btn_calls.setCheckable(True)
        self.btn_calls.setChecked(True)                 # on by default, as Mark
        self.show_call = True
        # mode menu: FT8, FT4, Off, and a WWV submenu (frequencies, timing window, capture)
        self.mode_menu = QtWidgets.QMenu(self)
        self.mode_actions = {}
        for name in ("FT8", "FT4", "Off"):
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
        self.chk_raw = QtWidgets.QCheckBox("Raw power")
        self.chk_raw.setChecked(True)
        controls.addWidget(self.chk_raw)
        self.chk_avg = QtWidgets.QCheckBox("Time avg")
        self.chk_avg.setChecked(False)
        self.an.set_avg(False)          # startup default: no time average
        self.chk_avg.toggled.connect(self.an.set_avg)
        controls.addWidget(self.chk_avg)
        controls.addStretch(1)
        self.btn_exit = QtWidgets.QPushButton("Exit")
        controls.addWidget(self.btn_exit)

        self.span_box.currentIndexChanged.connect(self.on_span)
        self.btn_save.clicked.connect(self.save_png)
        self.btn_popout.clicked.connect(self.pop_out)
        self.btn_exit.clicked.connect(self.exit_app)
        self.btn_marks.toggled.connect(self._set_marks_visible)
        self.btn_calls.toggled.connect(self._set_calls_visible)
        self.chk_raw.toggled.connect(self._on_raw_toggled)
        self.spec.setLabel("left", "Power" if self.chk_raw.isChecked() else "SNR", units="dB")

        # --- one status line: radio/FFT info, transient messages, version
        self.status = QtWidgets.QLabel()
        self.msg = QtWidgets.QLabel()
        self.version_label = QtWidgets.QLabel(f"v{VERSION}")
        self.version_label.setStyleSheet("color: #808080;")
        status_row = QtWidgets.QHBoxLayout()
        self.rate_label = QtWidgets.QLabel()
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
        self.btn_clear_log = QtWidgets.QPushButton("Clear log")
        self.btn_clear_log.setVisible(False)
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

        self.stats_win = StatsWindow(lambda: (self.stats, self.stats_mode))
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
        self._detach_filters()
        super().closeEvent(ev)

    def exit_app(self):
        self._detach_filters()
        self.popout.close()             # re-docks the pane if it was popped out; harmless otherwise
        self.stats_win.close()
        self.wwv_win.close()
        self.close()
        QtWidgets.QApplication.quit()

    # ---- decodes pane: pop out / dock back
    def pop_out(self):
        if not self.popped:
            self.popout.lay.addWidget(self.decode_pane)        # moves the whole decodes window
            self.popout.top.insertWidget(1, self.decode_pane.btn_stats)   # left of Single pane / Two panes
            self.popout.top.insertWidget(2, self.decode_pane.btn_split)   # left of Dock
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
        # keep the order: Stats, Two panes, Pop out (the Pop out button stayed in the bar)
        self.decode_pane.top_bar.layout().insertWidget(1, self.decode_pane.btn_stats)
        self.decode_pane.top_bar.layout().insertWidget(2, self.decode_pane.btn_split)
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
        self.spec_y = None              # refit the spectrum axis to the new span
        half = self.rate / 2
        lo = max(self.disp_off - self.span_hz / 2, -half)
        hi = min(self.disp_off + self.span_hz / 2, half)
        self.an.configure(self.tune_hz, lo, hi, settle_s=SETTLE_S if settle else 0.0)
        self.f0 = float(self.an.freq_khz[0])
        self.f1 = float(self.an.freq_khz[-1])
        self.spec_y = None

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
        mode = None if name == "Off" else name
        if self.wwv_on:
            self._leave_wwv()
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
        if self.wwv_on:
            self._leave_wwv()
        self._set_mode(None)                    # Tx Test, WWV and FT8/FT4 are mutually exclusive
        self._select_mode_box("Off")
        self.decode_dock.setWindowTitle("Tx log")
        self.btn_clear_log.setVisible(True)
        self.chk_noise.setVisible(True)
        self.btn_ref_reset.setVisible(True)
        self.btn_check_noise.setVisible(True)
        self._sync_check_btn(self.chk_noise.isChecked())
        self.noise_blk = int(time.time() // NOISE_REPORT_S)
        self.tx_label.setText("Tx Test: listening")

    def _leave_tx_mode(self):
        self.tx_label.setText("")
        self.btn_clear_log.setVisible(False)
        self.chk_noise.setVisible(False)
        self.btn_ref_reset.setVisible(False)
        self.btn_check_noise.setVisible(False)
        self.decode_dock.setWindowTitle("Decodes")

    # ---- WWV timing mode: tuned to a WWV frequency with decoding off; separate from Tx Test
    def _enter_wwv(self, mhz):
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
        for view in (self.decode_pane.view, self.decode_pane.view_cq, self.decode_pane.view_other):
            view.clear()

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
        if self.grab().save(path, "PNG"):
            self.msg.setText(f"Saved {path}")
        else:
            self.msg.setText("Save failed")

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
        if self.wwv_on and self.wwv is not None:
            for rec in self.wwv.process():
                self._on_wwv_minute(rec)
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
            self.dirty = False

# ------------------------------------------------------------ main

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
    ap.add_argument("--nfft", type=int, default=DEFAULT_NFFT,
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
        app.exec()
    finally:
        lib.airspyhf_stop(dev)
        lib.airspyhf_close(dev)


if __name__ == "__main__":
    main()
