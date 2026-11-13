#!/usr/bin/env python3

"""Live spectrum + waterfall for the Airspy HF+ with live FT8 decoding via WSJT-X's jt9.

Requires: numpy, scipy, PySide6, pyqtgraph, libairspyhf, and jt9 on PATH.
Decoder WAV files are written to /dev/shm and deleted after each decode.
Decodes are appended to ALL.TXT in the current directory and shown in a pane at the
bottom of the window, or in a separate window after "Pop out". CQ decodes are
highlighted there and can be marked on the waterfall.

Examples:
    ./airspy_live.py                              # 14.074 MHz, 14074-14077 kHz, 192 kS/s
    ./airspy_live.py -f 14.074 --span 3 --rate 192000
    ./airspy_live.py --no-decode                  # display only
"""
import argparse
import ctypes
import ctypes.util
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime

import numpy as np
import scipy.fft as sfft
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

VERSION = "0.6.1"
DEFAULT_FREQ_MHZ = 14.074
DEFAULT_LO_KHZ = 14074.0        # default display window when no display option is given
DEFAULT_HI_KHZ = 14077.0
DEFAULT_RATE = 192_000
DEFAULT_NFFT = 131072
MAX_DISPLAY_COLS = 2048
RING_SECONDS = 17               # ring must hold a full 15 s FT8 slot plus margin
HEADROOM_DB = 40.0              # colour range above the calibrated top
SETTLE_S = 0.25                 # discard samples after a retune
BG_HISTORY_S = 10.0             # noise-floor estimate window
LEVELS_MIN_ROWS = 10            # waterfall rows needed before colour levels are fixed
DECODE_PANE_HEIGHT = 170        # initial height of the docked decodes pane, px
DECODE_HISTORY_LINES = 10000    # scrollback kept in the decodes pane

STEPS = [("1 Hz", 1), ("10 Hz", 10), ("50 Hz", 50), ("100 Hz", 100),
         ("500 Hz", 500), ("1 kHz", 1000), ("2 kHz", 2000),
         ("5 kHz", 5000), ("10 kHz", 10000)]
DEFAULT_STEP_INDEX = 5          # 1 kHz

SPAN_PRESETS = [
    ("CW 500 Hz", 0.5),
    ("SSB 2.7 kHz", 2.7),
    ("SSB 3 kHz", 3.0),
    ("SSB 6 kHz", 6.0),
    ("20 kHz", 20.0),
    ("100 kHz", 100.0),
    ("350 kHz", 350.0),
]

# FT8 decoding
FT8_SLOT_S = 15
FT8_TONE_BW_HZ = 50.0                           # occupied bandwidth of one FT8 signal
FT8_AUDIO_RATE = 12_000
FT8_AUDIO_LEN = FT8_SLOT_S * FT8_AUDIO_RATE     # 180000 samples
FT8_AUDIO_CENTER = 2100.0                       # centre of the 200-4000 Hz decode band
FT8_AUDIO_MAX_SPAN = 3800.0                     # widest displayed range that can be decoded
JT9_ARGS = ["-8", "-d", "3", "-f", "2100"]     # receive offset: centre of the decode band
DECODE_RE = re.compile(r"^\s*(?:\d{6}\s+)?(-?\d+)\s+(-?\d+\.\d+)\s+(\d+(?:\.\d+)?)\s+~\s+(.*\S)\s*$")

MARK_RGBA_CQ = (200, 200, 200, 255)
MARK_RGBA_OTHER = (130, 130, 130, 255)

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

class Analyzer:
    def __init__(self, ring, rate, nfft, avg_s=1.0, bg_s=BG_HISTORY_S):
        self.ring = ring
        self.rate = rate
        self.nfft = nfft
        self.hop = nfft // 2
        self.win = np.hanning(nfft).astype(np.float32)
        self.max_frames = max(1, 2_000_000 // nfft)
        self.freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / rate))
        row_s = self.hop / rate
        self.rows_per_s = rate / self.hop
        self.alpha = min(1.0, row_s / max(avg_s, row_s))   # time-average weight
        self.bg_on = bg_s > 0
        self.bg_s = bg_s
        self.dec = max(1, int(round(self.rows_per_s / 10)))  # keep ~10 history rows/s, at least 1 per row
        self.next_pos = None
        self.ema = None
        self.bg_db = 0.0
        self.hist = None
        self.hist_n = 0
        self.hist_pos = 0
        self.row_count = 0
        self.keep = None
        self.group = 1
        self.freq_khz = None

    def configure(self, tune_hz, lo_off, hi_off, settle_s=0.0):
        """Set the displayed bins (offsets from tuning, Hz) and reset averaging."""
        self.keep = (self.freqs >= lo_off) & (self.freqs <= hi_off)
        nb = int(self.keep.sum())
        self.group = max(1, int(np.ceil(nb / MAX_DISPLAY_COLS)))
        self.freq_khz = self.reduce((tune_hz + self.freqs[self.keep]) / 1e3)
        self.ema = None
        self.bg_db = 0.0
        self.hist_n = 0
        self.hist_pos = 0
        self.row_count = 0
        if self.bg_on:
            # history length in entries = bg_s seconds of entries stored every `dec` rows
            hist_len = max(1, int(round(self.bg_s * self.rows_per_s / self.dec)))
            self.hist = np.zeros((hist_len, len(self.freq_khz)), np.float32)
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
        out = np.empty_like(pr)
        for i in range(pr.shape[0]):
            self.ema += self.alpha * (pr[i] - self.ema)      # time average
            out[i] = self.ema
            if self.bg_on:
                self.row_count += 1
                if self.row_count % self.dec == 0:
                    self.hist[self.hist_pos] = self.ema
                    self.hist_pos = (self.hist_pos + 1) % self.hist.shape[0]
                    self.hist_n = min(self.hist_n + 1, self.hist.shape[0])

        if self.bg_on and self.hist_n:
            bg = np.percentile(self.hist[:self.hist_n], 20, axis=0)   # noise-floor estimate
            self.bg_db = 10 * np.log10(bg + 1e-20)

        rows_db = 10 * np.log10(out + 1e-20) - self.bg_db
        spec_db = 10 * np.log10(self.ema + 1e-20) - self.bg_db
        return rows_db.astype(np.float32), spec_db

# ------------------------------------------------------------ FT8 decoding

def make_ft8_wav(z, rate, disp_off, span_hz, path):
    """Demodulate the displayed band to 12 kHz USB-style audio and write a 15 s mono WAV.

    z must hold exactly FT8_SLOT_S * rate complex samples. The displayed band is centred
    on 2100 Hz audio, clipped to 3.8 kHz, so audio freq = 2100 + (RF offset - display centre).
    """
    n_in = z.size
    X = sfft.fft(z, workers=-1)
    m = min(span_hz, FT8_AUDIO_MAX_SPAN)
    a_lo = FT8_AUDIO_CENTER - m / 2
    a_hi = FT8_AUDIO_CENTER + m / 2
    M = FT8_AUDIO_LEN
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


def format_all_line(slot_time, dial_hz, snr, dt, audio_hz, msg):
    """One ALL.TXT line in WSJT-X's layout (verify column widths against a real ALL.TXT)."""
    ts = time.strftime("%y%m%d_%H%M%S", time.gmtime(slot_time))
    return (f"{ts}  {dial_hz / 1e6:9.3f} Rx FT8  {snr:4d}  {dt:4.1f} "
            f"{int(round(audio_hz)):4d} ~  {msg}")


def run_decode(z, rate, slot_time, tune_hz, disp_off, span_hz, jt9_path, shm_dir):
    """Write the slot WAV to shared memory, run jt9, and return decode records."""
    stamp = time.strftime("%y%m%d_%H%M%S", time.gmtime(slot_time))
    path = os.path.join(shm_dir, stamp + ".wav")
    make_ft8_wav(z, rate, disp_off, span_hz, path)
    try:
        proc = subprocess.run([jt9_path] + JT9_ARGS + [path],
                              capture_output=True, text=True, timeout=120)
        text = proc.stdout
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    recs = []
    for ln in text.splitlines():
        m = DECODE_RE.match(ln)
        if not m:
            continue
        snr, dt, a_hz, msg = m.groups()
        msg = msg.strip()
        audio = disp_off + (float(a_hz) - FT8_AUDIO_CENTER)   # Hz relative to tuned frequency
        recs.append({
            "slot": slot_time,
            "line": format_all_line(slot_time, tune_hz, int(snr), float(dt), audio, msg),
            "msg": msg,
            "rf_hz": tune_hz + audio,                         # absolute RF of the tone start
        })
    return recs


class DecodeBridge(QtCore.QObject):
    """Carries decoder-thread results to the GUI thread."""
    decode = QtCore.Signal(object)
    status = QtCore.Signal(str)
    done = QtCore.Signal()

# ------------------------------------------------------------ GUI

class DecodeWindow(QtWidgets.QWidget):
    """Scrollable list of recent FT8 decodes. CQ lines are highlighted."""

    CQ_BG = QtGui.QColor(31, 92, 46)
    CQ_FG = QtGui.QColor(240, 255, 240)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QtWidgets.QTextEdit()
        self.view.setReadOnly(True)
        self.view.document().setMaximumBlockCount(DECODE_HISTORY_LINES)
        self.view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        font = QtGui.QFont("monospace")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.view.setFont(font)
        self.view.setPlaceholderText("FT8 decodes appear here and in ALL.TXT")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(self.view)

    def append(self, line, is_cq):
        sb = self.view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        cur = self.view.textCursor()
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


class DecodePopout(QtWidgets.QWidget):
    """Free-floating decodes window. Closing it reports back so the pane can re-dock."""

    closed = QtCore.Signal()

    def __init__(self):
        super().__init__(None, QtCore.Qt.Window)
        self.setWindowTitle("FT8 decodes")
        self.btn_dock = QtWidgets.QPushButton("Dock")
        top = QtWidgets.QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self.btn_dock)
        self.lay = QtWidgets.QVBoxLayout(self)
        self.lay.addLayout(top)
        self.resize(700, 400)

    def closeEvent(self, ev):
        self.closed.emit()
        ev.accept()


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
        self.step_hz = STEPS[DEFAULT_STEP_INDEX][1]
        self.jt9_path = jt9_path
        self.shm_dir = shm_dir
        self.all_txt = all_txt
        self.cfg_version = 0            # bumped on retune/span change
        self.cur_slot = None            # (boundary_time, sample_index, cfg_version)
        self.decoding = False

        # Decode marks are drawn into the waterfall's own pixel buffer. Each mark records the
        # sample range of its slot, so it scrolls on the same clock as the waterfall rows.
        self.marks = []                 # [{"s0", "s1", "cq", "x0", "w"}]
        self.show_marks = True
        self.ov = None                  # overlay codes, same shape as wf: 0 none, 1 other, 2 CQ
        self.newest_s = None            # sample index of the newest waterfall row's start
        self.dirty = False              # set when the image needs rebuilding

        self.levels = (vmin, vmax) if (vmin is not None and vmax is not None) else None
        self.rows = 0
        self.valid_rows = 0
        self.wf = None

        self.bridge = DecodeBridge()
        self.bridge.decode.connect(self._on_decode)
        self.bridge.status.connect(self._on_decode_status)
        self.bridge.done.connect(self._on_decode_done)

        self.spec = pg.PlotWidget(title="Spectrum")
        self.spec.setLabel("left", "SNR", units="dB")
        self.spec.setLabel("bottom", "Frequency", units="kHz")
        self.spec.getAxis("bottom").enableAutoSIPrefix(False)
        self.spec.setYRange(-5, 40, padding=0)
        self.spec_curve = self.spec.plot(pen="y")

        self.wf_plot = pg.PlotWidget(title="Waterfall (newest at top)")
        self.wf_plot.setLabel("bottom", "Frequency", units="kHz")
        self.row_s = an.hop / self.rate             # seconds per waterfall row
        self.wf_plot.setLabel("left", "Time relative to now", units="s")
        self.wf_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.wf_plot.setXLink(self.spec)
        self.wf_plot.setMouseEnabled(x=False, y=False)
        self.img = pg.ImageItem()
        self.wf_plot.addItem(self.img)
        self.lut_rgba = pg.colormap.get("viridis").getLookupTable(
            0.0, 1.0, 256, alpha=True, mode="byte")          # (256, 4) uint8

        # --- controls row
        self.btn_up = QtWidgets.QPushButton("▲ Up")
        self.btn_down = QtWidgets.QPushButton("▼ Down")
        self.btn_save = QtWidgets.QPushButton("Save PNG")
        self.btn_popout = QtWidgets.QPushButton("Pop out")
        self.btn_marks = QtWidgets.QPushButton("Mark")
        self.btn_marks.setCheckable(True)
        self.btn_marks.setChecked(True)                 # on by default
        self.show_marks = True
        self.chk_decode = QtWidgets.QCheckBox("Decode FT8")
        self.chk_decode.setChecked(decode_start and decode_ok)
        self.chk_decode.setEnabled(decode_ok)
        self.step_box = QtWidgets.QComboBox()
        for name, hz in STEPS:
            self.step_box.addItem(name, hz)
        self.step_box.setCurrentIndex(DEFAULT_STEP_INDEX)
        self.span_box = QtWidgets.QComboBox()
        for name, khz in SPAN_PRESETS:
            self.span_box.addItem(name, khz)
        self.span_box.addItem("Max (sample rate)", None)
        self._select_span(span_hz / 1e3)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.btn_down)
        controls.addWidget(self.btn_up)
        controls.addWidget(QtWidgets.QLabel("Step:"))
        controls.addWidget(self.step_box)
        controls.addWidget(QtWidgets.QLabel("Span:"))
        controls.addWidget(self.span_box)
        controls.addWidget(self.btn_save)
        controls.addWidget(self.chk_decode)
        controls.addWidget(self.btn_popout)
        controls.addWidget(self.btn_marks)
        controls.addStretch(1)
        self.tune_label = QtWidgets.QLabel()
        controls.addWidget(self.tune_label)

        self.btn_up.clicked.connect(lambda: self.retune(+self.step_hz))
        self.btn_down.clicked.connect(lambda: self.retune(-self.step_hz))
        self.step_box.currentIndexChanged.connect(self.on_step)
        self.span_box.currentIndexChanged.connect(self.on_span)
        self.btn_save.clicked.connect(self.save_png)
        self.btn_popout.clicked.connect(self.pop_out)
        self.btn_marks.toggled.connect(self._set_marks_visible)

        # --- one status line: radio/FFT info, transient messages, version
        self.status = QtWidgets.QLabel()
        self.msg = QtWidgets.QLabel()
        self.version_label = QtWidgets.QLabel(f"v{VERSION}")
        self.version_label.setStyleSheet("color: #808080;")
        status_row = QtWidgets.QHBoxLayout()
        status_row.addWidget(self.status)
        status_row.addWidget(self.msg, 1)
        status_row.addWidget(self.version_label)

        for pw in (self.spec, self.wf_plot):
            pw.getAxis("left").setWidth(60)             # fixed left-axis width
        self.spec.disableAutoRange(axis="y")            # stop y-autoranging from changing labels
        self.wf_plot.disableAutoRange(axis="x")

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addLayout(controls)
        lay.addWidget(self.spec, 1)
        lay.addWidget(self.wf_plot, 2)
        lay.addLayout(status_row)
        self.setCentralWidget(central)
        self.resize(1200, 900)

        # --- decodes: docked along the bottom, or popped out into its own window
        self.decode_pane = DecodeWindow()
        self.decode_dock = QtWidgets.QDockWidget("FT8 decodes", self)
        self.decode_dock.setObjectName("decodes_dock")
        self.decode_dock.setWidget(self.decode_pane)
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

        self._configure(settle=False)
        self._refresh()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)

    # ---- decodes pane: pop out / dock back
    def pop_out(self):
        if not self.popped:
            self.popout.lay.addWidget(self.decode_pane.view)   # moves the history widget
            self.popped = True
            self.decode_dock.hide()
        self.popout.show()
        self.popout.raise_()
        self.popout.activateWindow()

    def dock_back(self):
        if not self.popped:
            return
        self.decode_pane.layout().addWidget(self.decode_pane.view)   # moves the history back
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
        half = self.rate / 2
        lo = max(self.disp_off - self.span_hz / 2, -half)
        hi = min(self.disp_off + self.span_hz / 2, half)
        self.an.configure(self.tune_hz, lo, hi, settle_s=SETTLE_S if settle else 0.0)
        self.f0 = float(self.an.freq_khz[0])
        self.f1 = float(self.an.freq_khz[-1])

    def _refresh(self):
        self.spec.setXRange(self.f0, self.f1, padding=0)
        self.wf_plot.setXRange(self.f0, self.f1, padding=0)
        self.tune_label.setText(
            f"Tune {self.tune_hz / 1e6:.6f} MHz   Span {self.span_hz / 1e3:g} kHz")
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
        self.cfg_version += 1
        self._configure(settle=True)
        self._refresh()
        self.msg.setText("")

    def on_step(self, _idx):
        self.step_hz = self.step_box.currentData()

    def on_span(self, _idx):
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
        z = self.ring.read(start_idx, FT8_SLOT_S * self.rate)
        if z is None:
            self.msg.setText(f"Slot {stamp} UTC skipped: samples overwritten")
            return
        self.decoding = True
        cfg = (self.tune_hz, self.disp_off, self.span_hz)
        threading.Thread(target=self._decode_thread, args=(z, slot_time, cfg, start_idx),
                         daemon=True).start()

    def _decode_thread(self, z, slot_time, cfg, start_idx):
        tune_hz, disp_off, span_hz = cfg
        try:
            recs = run_decode(z, self.rate, slot_time, tune_hz, disp_off, span_hz,
                              self.jt9_path, self.shm_dir)
            for rec in recs:
                rec["s0"] = start_idx              # sample index of the slot start
            stamp = time.strftime("%H:%M:%S", time.gmtime(slot_time))
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
        try:
            with open(self.all_txt, "a") as f:
                f.write(rec["line"] + "\n")
        except OSError as e:
            self.msg.setText(f"ALL.TXT write failed: {e}")
        self.marks.append({
            "s0": rec["s0"],
            "s1": rec["s0"] + FT8_SLOT_S * self.rate,
            "cq": is_cq,
            "x0": rec["rf_hz"] / 1e3,
            "w": FT8_TONE_BW_HZ / 1e3,
        })
        self.dirty = True

    def _on_decode_status(self, text):
        self.msg.setText(text)

    def _on_decode_done(self):
        self.decoding = False

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

            # vertical arms along the left and right edges, clipped to the buffer
            for col in (c0, c1 - 1):
                ta, tb = max(r0, 0), min(r0 + arm_r, h)
                ba, bb = max(r1 - arm_r + 1, 0), min(r1 + 1, h)
                if ta < tb:
                    self.ov[ta:tb, col] = flag
                if ba < bb:
                    self.ov[ba:bb, col] = flag
            # horizontal arms at the top and bottom corners
            for row in (r0, r1):
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
        self.img.setImage(np.ascontiguousarray(rgba.transpose(1, 0, 2)), autoLevels=False)
        self.img.setRect(QtCore.QRectF(self.f0, -self.rows * self.row_s,
                                       self.f1 - self.f0, self.rows * self.row_s))

    def tick(self):
        h = int(self.wf_plot.getViewBox().height())
        if h > 10 and h != self.rows:
            self._realloc(h)

        # FT8 slot boundaries (UTC multiples of 15 s)
        now = time.time()
        slot_b = int(now // FT8_SLOT_S) * FT8_SLOT_S
        if self.cur_slot is None:
            self.cur_slot = (slot_b, self._index_at(now, slot_b), self.cfg_version)
        elif slot_b != self.cur_slot[0]:
            prev_b, prev_idx, prev_ver = self.cur_slot
            if self.chk_decode.isChecked():
                self._start_decode(prev_b, prev_idx, prev_ver)
            self.cur_slot = (slot_b, self._index_at(now, slot_b), self.cfg_version)

        rows_db, spec_db = self.an.process()
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
            self.dirty = True

        # One image update per change: waterfall pixels and marks together
        if self.dirty and self.wf is not None and self.levels is not None and self.valid_rows > 0:
            if self.show_marks and self.newest_s is not None and self.marks:
                self._paint_marks()
            self._compose()
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
    ap.add_argument("--shm-dir", default="/dev/shm/airspy_ft8",
                    help="directory for temporary decoder WAVs (default /dev/shm/airspy_ft8)")
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
