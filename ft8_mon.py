#!/usr/bin/env python3
"""
Live spectrum and waterfall display of the 20 m FT8 window (14.074-14.076 MHz)
from an RTL-SDR in direct sampling mode. Display only, no controls.
Close the window to stop.

Requires: pip install pyrtlsdr numpy matplotlib setuptools
"""
import os
import queue
import threading
import time
import warnings

# Let Python find librtlsdr.dll in this folder (Python 3.8+ on Windows)
os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from rtlsdr import RtlSdr
from rtlsdr.librtlsdr import librtlsdr

# ---- Frequencies ----
FT8_START_HZ  = 14.074e6     # FT8 band start
# FT8_START_HZ  = 7.074e6     # FT8 band start
FT8_LO_HZ     = FT8_START_HZ # FT8 window edges, shown as dashed lines
FT8_HI_HZ     = FT8_START_HZ + 2e3  # 2 kHz wide
DISPLAY_LO_HZ = FT8_START_HZ - 1.25e3  # displayed span, slightly wider than the FT8 window
DISPLAY_HI_HZ = FT8_START_HZ + 1.25e3
OFFSET_HZ     = 100e3        # tuner center this far above the display keeps the DC spike out
TUNE_HZ       = FT8_START_HZ + 1.0e3 + OFFSET_HZ

# ---- Receiver ----
SAMPLE_RATE     = 250e3      # covers the tuned span with low USB load
DIRECT_SAMPLING = 2          # 2 = Q branch (SDRangel's DS mode); try 1 if no signal appears
PPM             = 0
FFT_SIZE        = 4096       # ~61 Hz per bin at 250 kS/s
SEGS_PER_FRAME  = 8          # 32768 samples per frame
FRAME_S         = FFT_SIZE * SEGS_PER_FRAME / SAMPLE_RATE   # ~0.13 s per frame

# ---- Display ----
WATERFALL_ROWS  = 300        # about 40 s of history
MAXHOLD_S       = 60         # max-hold trace resets this often
WF_MIN_DB       = -3         # waterfall color range, dB above the noise floor
WF_MAX_DB       = 15
SPEC_MIN_DB     = -5         # spectrum plot range, dB above the noise floor
SPEC_MAX_DB     = 25
UPDATE_MS       = 200        # screen refresh interval

FREQS  = TUNE_HZ + np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1 / SAMPLE_RATE))
MASK   = (FREQS >= DISPLAY_LO_HZ) & (FREQS <= DISPLAY_HI_HZ)
FX_MHZ = FREQS[MASK] / 1e6
NBINS  = int(MASK.sum())
WINDOW = np.hanning(FFT_SIZE)


def frame_rel_db(samples):
    """One display row: power in dB relative to the median across the whole span."""
    n = len(samples) // FFT_SIZE
    segs = samples[:n * FFT_SIZE].reshape(n, FFT_SIZE) * WINDOW
    spec = np.fft.fftshift(np.fft.fft(segs, axis=1), axes=1)
    ps = np.mean(np.abs(spec) ** 2, axis=0)
    db = 10 * np.log10(ps + 1e-12)
    return db[MASK] - np.median(db)


def reader(sdr, q, stop):
    """Read samples in the background and queue one display row per frame."""
    while not stop.is_set():
        samples = sdr.read_samples(FFT_SIZE * SEGS_PER_FRAME)
        q.put(frame_rel_db(samples))


def main():
    sdr = RtlSdr()
    stop = threading.Event()
    t = None
    try:
        sdr.sample_rate = SAMPLE_RATE
        librtlsdr.rtlsdr_set_direct_sampling(sdr.dev_p, DIRECT_SAMPLING)
        sdr.center_freq = TUNE_HZ          # set after direct sampling is enabled
        if PPM:
            sdr.freq_correction = PPM

        fig, (ax_s, ax_w) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True,
            gridspec_kw={"height_ratios": [1, 2]})

        # Spectrum panel
        ax_s.set_xlim(FX_MHZ[0], FX_MHZ[-1])
        ax_s.set_ylim(SPEC_MIN_DB, SPEC_MAX_DB)
        ax_s.set_ylabel("dB above noise floor")
        ax_s.grid(True, alpha=0.3)
        line_now, = ax_s.plot(FX_MHZ, np.full(NBINS, np.nan),
                              color="tab:blue", lw=1.2, label="now")
        line_hold, = ax_s.plot(FX_MHZ, np.full(NBINS, np.nan),
                               color="tab:orange", lw=1,
                               label=f"max hold {MAXHOLD_S} s")
        ax_s.legend(loc="upper right")

        # Waterfall panel: newest row at the bottom
        cmap = plt.get_cmap("inferno").copy()
        cmap.set_bad("black")
        hist = np.full((WATERFALL_ROWS, NBINS), np.nan)
        im = ax_w.imshow(hist, aspect="auto", origin="upper", cmap=cmap,
                         vmin=WF_MIN_DB, vmax=WF_MAX_DB, interpolation="nearest",
                         extent=[FX_MHZ[0], FX_MHZ[-1], 0, WATERFALL_ROWS * FRAME_S])
        ax_w.set_ylabel("seconds ago")
        ax_w.set_xlabel("frequency (MHz)")

        for ax in (ax_s, ax_w):
            ax.axvline(FT8_LO_HZ / 1e6, color="gray", ls="--", lw=0.8)
            ax.axvline(FT8_HI_HZ / 1e6, color="gray", ls="--", lw=0.8)

        state = {"hist": hist, "row": None, "hold": None, "hold_t": 0.0}

        def update(_frame):
            changed = False
            while True:
                try:
                    row = q.get_nowait()
                except queue.Empty:
                    break
                now = time.time()
                state["hist"] = np.roll(state["hist"], -1, axis=0)
                state["hist"][-1] = row
                if state["hold"] is None or now - state["hold_t"] > MAXHOLD_S:
                    state["hold"] = row.copy()
                    state["hold_t"] = now
                else:
                    state["hold"] = np.maximum(state["hold"], row)
                state["row"] = row
                changed = True
            if changed:
                line_now.set_ydata(state["row"])
                line_hold.set_ydata(state["hold"])
                im.set_data(state["hist"])
                title_string = f"{FT8_START_HZ / 1e6:.3f} MHz FT8 window   {time.strftime('%H:%M:%S')}"
                fig.suptitle(title_string, fontsize=14)
            return line_now, line_hold, im

        q = queue.Queue()
        t = threading.Thread(target=reader, args=(sdr, q, stop), daemon=True)
        t.start()

        anim = FuncAnimation(fig, update, interval=UPDATE_MS, cache_frame_data=False)
        plt.show()

    finally:
        stop.set()
        if t is not None:
            t.join(timeout=2)
        sdr.close()


if __name__ == "__main__":
    main()
