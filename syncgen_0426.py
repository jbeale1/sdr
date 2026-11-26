#!/usr/bin/env python3
# generate a 2400 cps sync signal using a 10-stage m-sequence
# modulated onto a 1650 Hz audio carrier

import wave
import numpy as np
from scipy.signal import fftconvolve

FS = 48000                 # audio sample rate, Hz
RC = 2400.0                # chip rate, chips/s
SPS = 20                   # samples per chip (48000 / 2400)
ALPHA = 0.10               # raised-cosine rolloff
FC = 1650.0                # audio carrier, Hz (occupies about 330-2970 Hz)
SPAN = 12                  # pulse half-length in chips
AMP = 10 ** (-6 / 20)      # -6 dBFS peak

# 10-stage m-sequence, x^10 + x^7 + 1 (primitive), period 1023
state = [1] * 10
bits = []
for _ in range(1023):
    bits.append(state[-1])
    fb = state[9] ^ state[6]   # x^10 + x^7 + 1: bits from 10 and 7 steps ago
    state = [fb] + state[:-1]
bits = np.array(bits)
chips = 1.0 - 2.0 * bits   # 0 -> +1, 1 -> -1
assert len(chips) == 1023 and bits.sum() == 512

# upsample chips to the sample rate
up = np.zeros(len(chips) * SPS)
up[::SPS] = chips

# raised-cosine pulse, sampled at FS
T = 1.0 / RC
k = np.arange(-SPAN * SPS, SPAN * SPS + 1)
t = k / FS / T
denom = 1.0 - (2 * ALPHA * t) ** 2
sing = np.isclose(denom, 0.0, atol=1e-9)
h = np.sinc(t) * np.cos(np.pi * ALPHA * t) / np.where(sing, 1.0, denom)
h[sing] = (np.pi / 4) * np.sinc(1 / (2 * ALPHA))
h /= np.max(np.abs(h))

baseband = fftconvolve(up, h, mode="same")

# modulate onto the audio carrier
tt = np.arange(len(baseband)) / FS
sig = baseband * np.cos(2 * np.pi * FC * tt)
sig *= AMP / np.max(np.abs(sig))

# 100 ms silence before, 50 ms after
lead = np.zeros(int(0.100 * FS))
trail = np.zeros(int(0.050 * FS))
out = np.concatenate([lead, sig, trail])

pcm = np.round(out * 32767).astype("<i2")
with wave.open("sync_m1023_2400cps.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(FS)
    w.writeframes(pcm.tobytes())