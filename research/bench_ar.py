"""Benchmark the AR model alone and in fusion with the NMF / PnP / SPADE models.

Usage: python research/bench_ar.py OUT.jsonl CLEAN_TRACK [CLEAN_TRACK ...] [--device mps]
Each clean (unclipped) track gives an 8 s excerpt (its loudest), degraded three ways: hard clipping of 3 %
and 8 % of the samples at full scale, and 3 % clipping followed by a 128 kbps MP3 round trip (a smeared
ceiling, like clipped masters distributed as MP3). Every model runs once in auto mode on one shared
analysis; the fusions are formed from those estimates. Scores are SDR (dB) against the unclipped excerpt.
"""
import argparse
import json
import os
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf

from betterdeclipper.engine import declip, fuse
from betterdeclipper.metrics import sdr

EXCERPT_S = 8.0
MODELS = {
    "nmf150": ("nmf", 93, dict(n_iter=150)),
    "nmf800": ("nmf", 93, dict(n_iter=800)),
    "pnp800": ("pnp", 93, dict(n_iter=800)),
    "spade": ("spade", 93, dict()),
    "ar": ("ar", 60, dict()),
}
# name: [(model, weight below 4 kHz, weight above 4 kHz)]
FUSIONS = {
    "fast": [("nmf150", 1, 1)],
    "best": [("nmf800", 1, 1), ("pnp800", 1, 0), ("spade", 1, 0)],
    "ar": [("ar", 1, 1)],
    "fast+ar 0.75/0.25": [("nmf150", 0.75, 0.75), ("ar", 0.25, 0.25)],
    "fast+ar 0.6/0.4": [("nmf150", 0.6, 0.6), ("ar", 0.4, 0.4)],
    "fast+ar lo 0.5": [("nmf150", 0.5, 1), ("ar", 0.5, 0)],
    "best+ar": [("nmf800", 1, 1), ("pnp800", 1, 0), ("spade", 1, 0), ("ar", 1, 0)],
    "best+ar x2": [("nmf800", 1, 1), ("pnp800", 1, 0), ("spade", 1, 0), ("ar", 2, 0)],
    "best+ar all": [("nmf800", 1, 1), ("pnp800", 1, 0), ("spade", 1, 0), ("ar", 1, 0.5)],
}


def loudest_excerpt(path):
    info = sf.info(path)
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    n = int(EXCERPT_S * sr)
    hop = sr // 2
    energy = np.convolve((x ** 2).sum(1), np.ones(n), mode="valid")[::hop]
    start = int(np.argmax(energy)) * hop
    return x[start:start + n], info.samplerate


def mp3_round_trip(y, sr):
    with tempfile.TemporaryDirectory() as d:
        wav, mp3, out = (os.path.join(d, f) for f in ("in.wav", "in.mp3", "out.wav"))
        sf.write(wav, y.astype(np.float32), sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-c:a", "libmp3lame", "-b:a", "128k", mp3], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp3, "-c:a", "pcm_f32le", out], check=True)
        z, _ = sf.read(out, dtype="float64", always_2d=True)
    # align (the decoder trims the encoder delay; check it)
    a, b = y[:, 0], z[: len(y), 0]
    lags = np.arange(-2000, 2001)
    corr = [np.dot(a[2000:-2000], np.roll(b, -k)[2000:-2000]) for k in lags]
    lag = int(lags[int(np.argmax(corr))])
    z = np.roll(z, -lag, axis=0)[: len(y)]
    return z, lag


def run_case(name, gt, y, sr, device):
    out = {"case": name}
    t = time.time()
    _, info = declip(y, sr, preset="fast", device=device, verbose=False)
    analysis = info["analysis"]["data"] if info.get("analysis") else None
    out["analysis_s"] = round(time.time() - t, 1)
    out["flagged_pct"] = round(100 * info["clipped_frac"], 2)
    out["mode"] = info["mode"]
    ests = {}
    for key, model in MODELS.items():
        t = time.time()
        ests[key], _ = declip(y, sr, models=[model], device=device, verbose=False, analysis=analysis)
        out[f"time_{key}"] = round(time.time() - t, 1)
    out["sdr_in"] = round(sdr(gt, y), 2)
    for fname, parts in FUSIONS.items():
        est = fuse(y, [ests[k] for k, _, _ in parts], [w for _, w, _ in parts], [w for _, _, w in parts], sr)
        out[fname] = round(sdr(gt, est), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("tracks", nargs="+")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    with open(args.out, "a") as f:
        for path in args.tracks:
            x, sr = loudest_excerpt(path)
            label = os.path.splitext(os.path.basename(path))[0][:24]
            for frac, codec in ((0.03, False), (0.08, False), (0.03, True)):
                gt = x / np.quantile(np.abs(x), 1 - frac)
                y = np.clip(gt, -1, 1)
                case = f"{label} clip {frac * 100:.0f}%"
                if codec:
                    y, lag = mp3_round_trip(y, sr)
                    case += f" +mp3 (lag {lag})"
                row = run_case(case, gt, y, sr, args.device)
                print(json.dumps(row), flush=True)
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
