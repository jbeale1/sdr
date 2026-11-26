#!/usr/bin/env python3
# play a WAV file through the audio device feeding the radio, keying PTT over serial
# Requires sounddevice, pyserial, Python 3.6 ... 3.12 with the standard library wave module.
# J.Beale, 2026-11-25

import argparse
import sys
import time
import warnings
import wave
import threading

warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop  # stdlib in Python 3.12 and earlier

import serial
import sounddevice as sd

PTT_PORT = "COM10"                       # serial port whose RTS line keys the radio
AUDIO_NAME = "Speakers (AIOC Audio)"     # audio output device feeding the radio
API_PREFERENCE = ["Windows WASAPI", "Windows DirectSound", "MME", "Windows WDM-KS"]
SAMPLE_DTYPES = {1: "uint8", 2: "int16", 3: "int24", 4: "int32"}
CHUNK_FRAMES = 4096
BAUD = 9600
DEFAULT_PRE_DELAY = 0.5                  # seconds PTT is held before audio starts
DEFAULT_POST_DELAY = 0.5                 # seconds PTT is held after audio finishes


def find_target_devices():
    hostapis = sd.query_hostapis()
    found = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["name"] == AUDIO_NAME and dev["max_output_channels"] > 0:
            api = hostapis[dev["hostapi"]]["name"]
            rank = API_PREFERENCE.index(api) if api in API_PREFERENCE else len(API_PREFERENCE)
            found.append((rank, idx, api, dev["max_output_channels"]))
    found.sort()
    return [(idx, api, ch) for _, idx, api, ch in found]


def match_channels(frames, width, wav_ch, dev_ch):
    if wav_ch == dev_ch:
        return frames
    unsigned8 = width == 1  # WAV 8-bit is unsigned; audioop expects signed
    if unsigned8:
        frames = audioop.bias(frames, 1, -128)
    if wav_ch == 2 and dev_ch == 1:
        frames = audioop.tomono(frames, width, 0.5, 0.5)
    elif wav_ch == 1 and dev_ch == 2:
        frames = audioop.tostereo(frames, width, 1, 1)
    else:
        raise ValueError(f"Cannot map {wav_ch}-channel WAV to {dev_ch}-channel device")
    if unsigned8:
        frames = audioop.bias(frames, 1, 128)
    return frames


def load_wav(path):
    try:
        with wave.open(path, "rb") as wf:
            wav_ch = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
    except (wave.Error, OSError) as e:
        sys.exit(f"Cannot read {path}: {e}")

    if width not in SAMPLE_DTYPES:
        sys.exit(f"Unsupported sample width: {width} bytes")
    print(f"WAV length: {nframes / rate:.3f} s "
          f"({nframes} frames, {rate} Hz, {wav_ch} ch, {width * 8}-bit)")
    return wav_ch, width, rate, raw

def open_audio_stream(wav_ch, width, rate, raw):
    """Create (but do not start) a callback output stream on the first working device.
    Runs before PTT is keyed, so a device failure never transmits."""
    targets = find_target_devices()
    if not targets:
        sys.exit(f"No output device named {AUDIO_NAME!r} was found.")

    errors = []
    for dev_idx, api, dev_ch in targets:
        try:
            frames = match_channels(raw, width, wav_ch, dev_ch)
            bytes_per_frame = width * dev_ch
            silence = b"\x80" if width == 1 else b"\x00"  # 8-bit WAV silence is 0x80
            finished = threading.Event()
            pos = 0

            def callback(outdata, frame_count, time_info, status):
                nonlocal pos
                n = frame_count * bytes_per_frame
                chunk = frames[pos:pos + n]
                pos += len(chunk)
                outdata[:len(chunk)] = chunk
                if len(chunk) < n:
                    outdata[len(chunk):n] = silence * (n - len(chunk))
                    raise sd.CallbackStop  # play out what is buffered, then finish

            stream = sd.RawOutputStream(samplerate=rate, channels=dev_ch,
                                        latency="low", blocksize=1024, dtype=SAMPLE_DTYPES[width], device=dev_idx,
                                        callback=callback, finished_callback=finished.set)  
            print(f"blocksize: {stream.blocksize}")
            
        except sd.PortAudioError as e:
            errors.append(f"{api}: {e}")
            continue
        return stream, finished, dev_ch, api

    sys.exit(f"Could not open {AUDIO_NAME!r}. Errors: " + "; ".join(errors))


def play_with_ptt(path, pre_delay, post_delay, port):
    wav_ch, width, rate, raw = load_wav(path)
    stream, finished, dev_ch, api = open_audio_stream(wav_ch, width, rate, raw)


    ptt = serial.Serial()
    ptt.port = port
    ptt.baudrate = BAUD
    ptt.rts = False

    try:
        ptt.open()
    except serial.SerialException as exc:
        stream.close()
        sys.exit(f"Could not open {port}: {exc}")

    t_key_down = None
    t_key_up = None
    wav_seconds = len(raw) / (rate * wav_ch * width)
    try:
        ptt.rts = True  # assert RTS (keys the radio)
        t_key_down = time.perf_counter()
        time.sleep(pre_delay)
        print(f"PTT keyed on {port}; playing {path} ({rate} Hz) via {api}")
        t_start = time.perf_counter()
        stream.start()            # audio starts only after PTT has settled
        print(f"stream latency: {stream.latency:.3f} s")
        finished.wait()           # set by PortAudio once playback has really ended
        t_done = time.perf_counter()
        print(f"keyed -> stream.start(): {t_start - t_key_down:.3f} s")
        print(f"stream.start() -> playback done: {t_done - t_start:.3f} s")
        time.sleep(post_delay)
    finally:
        ptt.rts = False  # unkey even if playback fails or Ctrl+C is pressed
        t_key_up = time.perf_counter()
        ptt.close()
        stream.close()
        print("PTT released")
        if t_key_down is not None:
            print(f"PTT measured: key-down to key-up = {t_key_up - t_key_down:.3f} s")

def main():
    parser = argparse.ArgumentParser(
        description="Key the radio's PTT over serial, play a WAV file, then unkey.")
    parser.add_argument("wav", help="path to the WAV file to play")
    parser.add_argument("--port", default=PTT_PORT,
                        help=f"serial port for PTT (default {PTT_PORT})")
    parser.add_argument("--pre-delay", type=float, default=DEFAULT_PRE_DELAY,
                        help="seconds to hold PTT before audio starts")
    parser.add_argument("--post-delay", type=float, default=DEFAULT_POST_DELAY,
                        help="seconds to hold PTT after audio ends")
    args = parser.parse_args()
    play_with_ptt(args.wav, args.pre_delay, args.post_delay, args.port)


if __name__ == "__main__":
    main()
