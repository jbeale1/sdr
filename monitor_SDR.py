#!/usr/bin/env python3
"""
Detect transmissions on a fixed frequency with an RTL-SDR and log
start time, duration, signal strength, occupied bandwidth, and center-of-mass
frequency to a CSV file. Shows a live meter with a rolling noise-only statistic
to help choose a threshold. Logs noise-only statistics every NOISE_LOG_PERIOD_S
seconds. After each detected event, uploads an HTML table of recent events by FTP.

Requires: pip install pyrtlsdr numpy setuptools
Credentials file: ftp_credentials.txt with
SITE=ftp.mysite.com
USER=myloginname
PASS=mypassword
"""
import collections
import ftplib
import html
import io
import os
import sys

# Let Python find librtlsdr.dll in this folder (Python 3.8+ on Windows)
os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))
os.system("")  # enables ANSI escape codes in the Windows console

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import csv
import datetime as dt
import threading
import time

import numpy as np
from rtlsdr import RtlSdr

TARGET_HZ   = 146.440e6
OFFSET_HZ   = 100e3          # tune this far from target to keep the DC spike out of the channel
CENTER_HZ   = TARGET_HZ + OFFSET_HZ
SAMPLE_RATE = 2.048e6
GAIN        = 20.0           # dB, or "auto"
PPM         = 0              # frequency correction for your dongle
FFT_SIZE    = 4096           # ~500 Hz per bin at 2.048 MS/s
SEGS_PER_READ = 64           # ~0.13 s of data per measurement
BW_HZ       = 12.5e3         # channel width to integrate (FM 2m is ~ +/-5 kHz)
CENTROID_SPAN_HZ = 50e3      # center-of-mass searched within +/- this of target
THRESH_DB   = 3.0            # dB above noise floor to declare a signal
HANG_S      = 1.0            # keep an event open this long after the signal drops
MIN_EVENT_S = 0.5            # drop events shorter than this (after transient skip)
SKIP_S      = 0.2            # ignore this much of the start of each event (keying transient)
BW_ABOVE_NOISE_DB = 6.0      # bandwidth threshold above noise floor
PEAK_DROP_DB = 20            # bandwidth limited to bins within this many dB of the peak
NOISE_WINDOW_S = 600         # rolling window for noise-only statistics (10 minutes)
NOISE_LOG_PERIOD_S = 600     # how often to write a noise statistics row (10 minutes)
LOG_FILE    = "signal_log.csv"
NOISE_LOG_FILE = "noise_stats.csv"
CREDENTIALS_FILE = "ftp_credentials.txt"

FTP_DIR     = "raw"          # remote directory
FTP_NAME    = "signal_log.html"      # remote file name
TAIL_LINES  = 5                      # most recent events shown on the page

METER_PERIOD = 0.25          # seconds between meter updates (4 per second)
METER_MIN_DB = -5            # SNR shown as empty bar at or below this
METER_MAX_DB = 30            # SNR shown as full bar at or above this
METER_WIDTH  = 40            # characters in the bar

FREQS = CENTER_HZ + np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1 / SAMPLE_RATE))
BAND  = np.abs(FREQS - TARGET_HZ) <= BW_HZ / 2
BAND_IDX = np.flatnonzero(BAND)
CENTROID_IDX = np.flatnonzero(np.abs(FREQS - TARGET_HZ) <= CENTROID_SPAN_HZ)
BIN_HZ = SAMPLE_RATE / FFT_SIZE
WINDOW = np.hanning(FFT_SIZE)
SMOOTH = np.ones(3) / 3      # 3-bin smoothing to tame per-bin noise

# (time, snr) for measurements made while no signal was present
noise_hist = collections.deque()


def load_credentials(path):
    """Read HOST, USER and PASS from a simple KEY=VALUE file."""
    creds = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            creds[key.strip().upper()] = value.strip()
    if "HOST" not in creds or "USER" not in creds or "PASS" not in creds:
        raise ValueError(f"{path} must contain HOST=, USER= and PASS= lines")
    return creds["HOST"], creds["USER"], creds["PASS"]


def build_html(rows):
    """Build a phone-friendly HTML page from the event rows (newest first)."""
    updated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    if rows:
        body = []
        for r in rows:
            start = html.escape(r["start_local"].replace("T", " ")[5:])  # MM-DD HH:MM:SS
            body.append(
                f"<tr><td>{start}</td>"
                f"<td>{float(r['duration_s']):.1f}</td>"
                f"<td>{float(r['mean_snr_db']):+.1f}</td>"
                f"<td>{float(r['mean_bw_khz']):.1f}</td></tr>")
        rows_html = "\n".join(body)
    else:
        rows_html = '<tr><td colspan="4">No events yet</td></tr>'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>146.433 MHz signal log</title>
<style>
  body {{ font-family: sans-serif; font-size: 18px; margin: 12px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px 0; }}
  p.updated {{ color: #555; font-size: 14px; margin: 0 0 12px 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 10px 6px; text-align: right; }}
  th {{ border-bottom: 2px solid #333; font-size: 16px; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
</style>
</head>
<body>
<h1>146.433 MHz events</h1>
<p class="updated">Updated {updated}</p>
<table>
<tr><th>Start</th><th>Dur (s)</th><th>SNR (dB)</th><th>BW (kHz)</th></tr>
{rows_html}
</table>
</body>
</html>
"""


def upload_tail(ftp_site, user, password):
    """Upload the most recent TAIL_LINES events as an HTML page."""
    if not os.path.exists(LOG_FILE):
        return
    with open(LOG_FILE, newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = rows[-TAIL_LINES:][::-1]          # newest first
    data = io.BytesIO(build_html(rows).encode("utf-8"))

    ftp = ftplib.FTP(ftp_site, timeout=30)
    try:
        ftp.login(user, password)
        ftp.cwd(FTP_DIR)
        ftp.storbinary(f"STOR {FTP_NAME}", data)
    finally:
        try:
            ftp.quit()
        except ftplib.all_errors:
            ftp.close()

def upload_async(ftp_site, user, password):
    """Start the upload in a background thread so sampling isn't interrupted."""
    def worker():
        try:
            upload_tail(ftp_site, user, password)
        except Exception as e:
            clear_meter()
            print(f"FTP upload failed: {type(e).__name__}: {e!r}")
    # Non-daemon, so an upload started just before Ctrl+C still finishes
    threading.Thread(target=worker).start()


def power_spectrum(samples):
    n = len(samples) // FFT_SIZE
    segs = samples[:n * FFT_SIZE].reshape(n, FFT_SIZE) * WINDOW
    spec = np.fft.fftshift(np.fft.fft(segs, axis=1), axes=1)
    return np.mean(np.abs(spec) ** 2, axis=0)


def occupied_bw_khz(ps, noise_db):
    """Contiguous width around the strongest in-band bin, limited to bins
    within PEAK_DROP_DB of that peak and at least BW_ABOVE_NOISE_DB above noise."""
    smoothed = np.convolve(ps, SMOOTH, mode="same")
    db = 10 * np.log10(smoothed + 1e-12)
    p = BAND_IDX[np.argmax(db[BAND_IDX])]
    if db[p] < noise_db + BW_ABOVE_NOISE_DB:
        return 0.0
    thr = max(noise_db + BW_ABOVE_NOISE_DB, db[p] - PEAK_DROP_DB)

    lo = p
    while lo > 0 and db[lo - 1] >= thr:
        lo -= 1
    hi = p
    while hi < FFT_SIZE - 1 and db[hi + 1] >= thr:
        hi += 1
    return (hi - lo + 1) * BIN_HZ / 1e3


def center_of_mass_mhz(ps, noise_db):
    """Power-weighted mean frequency (MHz) of bins within CENTROID_SPAN_HZ of
    the target that are at least THRESH_DB above the noise floor. NaN if none."""
    smoothed = np.convolve(ps, SMOOTH, mode="same")
    db = 10 * np.log10(smoothed + 1e-12)
    sel = CENTROID_IDX[db[CENTROID_IDX] >= noise_db + THRESH_DB]
    if sel.size == 0:
        return float("nan")
    w = smoothed[sel]
    return float(np.sum(FREQS[sel] * w) / np.sum(w)) / 1e6


def record_noise(now, snr):
    noise_hist.append((now, snr))
    cutoff = now - NOISE_WINDOW_S
    while noise_hist and noise_hist[0][0] < cutoff:
        noise_hist.popleft()


def noise_stats(now):
    """Summary of noise-only SNR over the rolling window, or None if too little data."""
    span = now - noise_hist[0][0] if noise_hist else 0.0
    if len(noise_hist) < 20 or span < 60:
        return None
    vals = np.array([s for _, s in noise_hist])
    return {
        "n": len(vals),
        "span_min": span / 60,
        "mean": vals.mean(),
        "sd": vals.std(),
        "max": vals.max(),
        "p999": np.percentile(vals, 99.9),
        "pct_over": 100.0 * np.mean(vals >= THRESH_DB),
    }


def draw_meter(level_db, snr, bw_khz, active, now):
    frac = (snr - METER_MIN_DB) / (METER_MAX_DB - METER_MIN_DB)
    frac = min(max(frac, 0.0), 1.0)
    filled = int(round(frac * METER_WIDTH))
    bar = "#" * filled + "-" * (METER_WIDTH - filled)
    tag = f"SIGNAL  BW {bw_khz:6.1f} kHz" if active else " " * 20
    line1 = f"Level {level_db:7.1f} dB  SNR {snr:+6.1f} dB |{bar}| {tag}"

    s = noise_stats(now)
    if s is None:
        line2 = "Noise (rolling): collecting data..."
    else:
        line2 = (f"Noise {s['span_min']:4.1f} min (n={s['n']}): "
                 f"mean {s['mean']:+5.1f}  sd {s['sd']:4.2f}  "
                 f"p99.9 {s['p999']:+5.1f}  max {s['max']:+5.1f}  "
                 f"above {THRESH_DB:+.1f} dB: {s['pct_over']:.2f}%")

    sys.stdout.write(f"\r\x1b[2K{line1}\n\x1b[2K{line2}\x1b[1A\r")
    sys.stdout.flush()


def clear_meter():
    sys.stdout.write("\r\x1b[2K\n\x1b[2K\x1b[1A\r")
    sys.stdout.flush()


def open_log():
    new = not os.path.exists(LOG_FILE)
    f = open(LOG_FILE, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["start_local", "duration_s",
                    "mean_level_db", "peak_level_db", "mean_snr_db", "peak_snr_db",
                    "mean_bw_khz", "peak_bw_khz", "mean_center_mhz"])
        f.flush()
    return f, w


def open_noise_log():
    new = not os.path.exists(NOISE_LOG_FILE)
    f = open(NOISE_LOG_FILE, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp_local", "window_min", "n", "mean_db", "sd_db",
                    "p99_9_db", "max_db", "pct_above_threshold", "threshold_db"])
        f.flush()
    return f, w


def main():
    print("Starting SDR monitor...")
    try:
        ftp_site, ftp_user, ftp_pass = load_credentials(CREDENTIALS_FILE)
    except (OSError, ValueError) as e:
        sys.exit(f"Cannot read FTP credentials: {e}")

    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_HZ
    sdr.gain = GAIN
    if PPM:
        sdr.freq_correction = PPM

    f, w = open_log()
    nf, nw = open_noise_log()
    print(f"Monitoring {TARGET_HZ/1e6:.3f} MHz, threshold {THRESH_DB} dB. Ctrl+C to stop.")

    in_event = False
    start_t = last_t = 0.0
    samples_in_event = []        # (time, sig_db, snr, bw_khz, center_mhz) per measurement
    last_meter = 0.0
    last_noise_log = time.time()

    def close_event():
        nonlocal in_event
        in_event = False
        steady = [m for m in samples_in_event if m[0] - start_t >= SKIP_S]
        duration = steady[-1][0] - steady[0][0] if steady else 0.0

        if duration >= MIN_EVENT_S and steady:
            levels = np.array([m[1] for m in steady])
            snrs = np.array([m[2] for m in steady])
            bws = np.array([m[3] for m in steady])
            centers = np.array([m[4] for m in steady])
            valid = centers[~np.isnan(centers)]
            center_mean = valid.mean() if valid.size else float("nan")
            start = dt.datetime.fromtimestamp(steady[0][0])
            w.writerow([start.isoformat(timespec="seconds"),
                        f"{duration:.2f}",
                        f"{levels.mean():.1f}", f"{levels.max():.1f}",
                        f"{snrs.mean():.1f}", f"{snrs.max():.1f}",
                        f"{bws.mean():.1f}", f"{bws.max():.1f}",
                        f"{center_mean:.3f}"])
            f.flush()
            clear_meter()
            print(f"{start:%H:%M:%S}  dur {duration:6.2f}s  "
                  f"mean SNR {snrs.mean():5.1f} dB  level {levels.mean():6.1f} dB  "
                  f"BW mean {bws.mean():5.1f} kHz  peak {bws.max():5.1f} kHz  "
                  f"center {center_mean:.3f} MHz")
            upload_async(ftp_site, ftp_user, ftp_pass)
        samples_in_event.clear()

    def log_noise(now):
        s = noise_stats(now)
        if s is None:
            return
        nw.writerow([dt.datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                     f"{s['span_min']:.1f}", s["n"],
                     f"{s['mean']:.2f}", f"{s['sd']:.2f}", f"{s['p999']:.2f}",
                     f"{s['max']:.2f}", f"{s['pct_over']:.3f}", f"{THRESH_DB:.1f}"])
        nf.flush()

    try:
        while True:
            samples = sdr.read_samples(FFT_SIZE * SEGS_PER_READ)
            ps = power_spectrum(samples)
            now = time.time()

            sig_db = 10 * np.log10(ps[BAND].mean())      # power in channel
            noise_db = 10 * np.log10(np.median(ps))      # noise floor across span
            snr = sig_db - noise_db
            above = snr >= THRESH_DB
            bw_khz = occupied_bw_khz(ps, noise_db) if above else 0.0

            if above:
                center_mhz = center_of_mass_mhz(ps, noise_db)
                if not in_event:
                    in_event = True
                    start_t = now
                    samples_in_event.clear()
                last_t = now
                samples_in_event.append((now, sig_db, snr, bw_khz, center_mhz))
            else:
                record_noise(now, snr)
                if in_event and now - last_t >= HANG_S:
                    close_event()

            if now - last_noise_log >= NOISE_LOG_PERIOD_S:
                log_noise(now)
                last_noise_log = now

            if now - last_meter >= METER_PERIOD:
                draw_meter(sig_db, snr, bw_khz, above, now)
                last_meter = now

    except KeyboardInterrupt:
        if in_event:
            close_event()
        clear_meter()
    finally:
        sdr.close()
        f.close()
        nf.close()


if __name__ == "__main__":
    main()