"""True-peak limiter for restored masters.

Declipping restores peaks above the clip level, often above full scale. Normalizing the whole file to its
new peak costs as much loudness as the peaks gained; this limiter only turns down the restored peaks
(like RX De-clip's post-limiter), so the output keeps the input's loudness under a true-peak ceiling.

Offline, zero-latency design: the gain reduction needed at each sample (from the 4x oversampled peak,
linked over channels) is held for `lookahead` on both sides and averaged over the same span, which
reaches every peak's full reduction smoothly. A decaying hold then adds a short attack before and a
release after it. Gains are in dB; the passes repeat until the ceiling holds.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import resample_poly

OVERSAMPLE = 4


def true_peak_envelope(x, oversample=OVERSAMPLE):
    """Per-sample true peak (T,): max |x| over channels and over each sample's oversampled interval."""
    up = np.abs(resample_poly(x, oversample, 1, axis=0)).max(axis=1)
    T = x.shape[0]
    return np.maximum(up[: T * oversample].reshape(T, oversample).max(axis=1), np.abs(x).max(axis=1))


def _decaying_hold(r, k):
    """r_out[n] = max(r[n], k * r_out[n - 1]): a hold that decays by the factor k per sample, vectorized in
    blocks (within a block, the scaled running maximum of r[m] / k^m)."""
    out = np.empty_like(r)
    block = max(1, int(np.log(1e6) / -np.log(k))) if k > 0 else 1
    carry = 0.0
    for b in range(0, len(r), block):
        seg = r[b:b + block]
        powk = k ** np.arange(len(seg))
        held = np.maximum.accumulate(np.maximum(seg / powk, carry * k))
        out[b:b + block] = held * powk
        carry = out[b + len(seg) - 1]
    return out


def limit_true_peak(x, sr, ceiling_db=-0.1, lookahead_ms=1.5, attack_ms=2.0, release_ms=30.0, max_passes=4):
    """Limit x (T, C) or (T,) to a true-peak ceiling in dBTP. Returns (limited x, max gain reduction in dB)."""
    x = np.asarray(x, dtype=np.float64)
    mono = x.ndim == 1
    if mono:
        x = x[:, None]
    L = max(1, round(lookahead_ms * 1e-3 * sr))
    k_att = np.exp(-1.0 / (attack_ms * 1e-3 * sr))
    k_rel = np.exp(-1.0 / (release_ms * 1e-3 * sr))
    total = np.zeros(x.shape[0])
    for _ in range(max_passes):
        # the 1e-3 dB margin keeps the interpolated peaks of the next pass under the ceiling
        need = np.maximum(20 * np.log10(np.maximum(true_peak_envelope(x), 1e-12)) - ceiling_db + 1e-3, 0.0)
        if need.max() <= 1e-3:
            break
        r = uniform_filter1d(maximum_filter1d(need, size=2 * L + 1), size=2 * L + 1)
        r = np.maximum(_decaying_hold(r, k_rel), _decaying_hold(r[::-1], k_att)[::-1])
        x = x * (10 ** (-r / 20))[:, None]
        total += r
    return (x[:, 0] if mono else x), float(total.max())
