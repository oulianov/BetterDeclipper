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
import torch
import torch.nn.functional as Fnn
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import firwin

OVERSAMPLE = 4


def true_peak_envelope(x, oversample=OVERSAMPLE, device="cpu"):
    """Per-sample true peak (T,): max |x| over channels and over each sample's oversampled interval.

    The interpolation is scipy's resample_poly(x, oversample, 1) (its Kaiser-windowed sinc filter), run as
    one polyphase convolution on `device`: oversampled sample oversample * n + p is
    sum_i h[oversample * i + p] x[n + 10 - i]."""
    device = torch.device(device)
    dtype = torch.float32 if device.type == "mps" else torch.float64  # MPS has no float64
    h = np.zeros(21 * oversample)
    h[: 20 * oversample + 1] = oversample * firwin(20 * oversample + 1, 1.0 / oversample, window=("kaiser", 5.0))
    phases = torch.as_tensor(h.reshape(21, oversample).T[:, ::-1].copy(), dtype=dtype, device=device)
    xt = torch.as_tensor(np.asarray(x).T, dtype=dtype, device=device)
    up = Fnn.conv1d(xt[:, None], phases[:, None], padding=10).abs_().amax((0, 1))
    return torch.maximum(up, xt.abs().amax(0)).cpu().numpy().astype(np.float64)


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


def limit_true_peak(x, sr, ceiling_db=-0.1, lookahead_ms=1.5, attack_ms=2.0, release_ms=30.0, max_passes=4,
                    device="cpu"):
    """Limit x (T, C) or (T,) to a true-peak ceiling in dBTP. Returns (limited x, max gain reduction in dB).
    device: where the oversampled peaks are computed (the slow part), e.g. 'cuda'."""
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
        need = np.maximum(20 * np.log10(np.maximum(true_peak_envelope(x, device=device), 1e-12)) - ceiling_db + 1e-3, 0.0)
        if need.max() <= 1e-3:
            break
        r = uniform_filter1d(maximum_filter1d(need, size=2 * L + 1), size=2 * L + 1)
        r = np.maximum(_decaying_hold(r, k_rel), _decaying_hold(r[::-1], k_att)[::-1])
        x = x * (10 ** (-r / 20))[:, None]
        total += r
    return (x[:, 0] if mono else x), float(total.max())
