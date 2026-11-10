import numpy as np

from betterdeclipper.auto import analyze_ceilings
from betterdeclipper.detect import estimate_lsb

SR = 48000
LSB_16 = 1 / 32768


def _quantize_16(y):
    return np.round(y / LSB_16) * LSB_16


def _quiet_stem(level, clip=None, silence_s=5.0, seconds=30.0, seed=0):
    """Mono 16-bit stem: digital silence, then noise around -26 dBFS (a dry DI part)."""
    y = np.zeros(int(seconds * SR))
    body = level * np.random.default_rng(seed).standard_normal(y.size - int(silence_s * SR))
    if clip is not None:
        body = np.clip(body, -clip, clip)
    y[int(silence_s * SR):] = body
    return _quantize_16(y)[:, None]


def test_silent_intro_keeps_the_source_bit_depth_and_analysis_runs():
    y = _quiet_stem(level=0.01)

    lsb = estimate_lsb(y)
    ceilings = analyze_ceilings(y, lsb, SR)

    assert lsb == LSB_16
    assert [c["kind"] for c in ceilings[0]] == ["none", "none"]


def test_clipping_after_a_silent_intro_is_still_detected():
    y = _quiet_stem(level=0.01, clip=0.02)

    ceilings = analyze_ceilings(y, estimate_lsb(y), SR)

    assert all(c["kind"] != "none" and abs(c["theta"] - 0.02) < 2 * LSB_16 for c in ceilings[0])


def test_near_silent_channel_has_no_ceiling():
    rng = np.random.default_rng(1)
    hiss = 1.6 * LSB_16 * rng.standard_normal(10 * SR)  # peaks 7 LSBs: too coarse for a ceiling fit
    loud = 0.1 * rng.standard_normal(10 * SR)
    y = _quantize_16(np.column_stack((loud, hiss)))

    ceilings = analyze_ceilings(y, estimate_lsb(y), SR)

    assert [c["kind"] for c in ceilings[1]] == ["none", "none"]
