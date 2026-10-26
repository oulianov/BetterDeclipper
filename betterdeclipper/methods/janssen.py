"""AR (Janssen / LSAR) restoration of clipped samples.

The flagged samples are re-estimated from an autoregressive (AR) model of their surroundings: they are
chosen to minimize the AR prediction-error energy (least-squares AR interpolation, LSAR), subject to the
clipping constraints (a sample clipped high is at least its clip level, ...). Like iZotope RX's De-clip,
several passes refit the AR model on the previous estimate with a growing order; unlike it, every pass
stays consistent with the clipping constraints.

Flagged runs closer than the AR order interact and are solved together as one segment. For a segment with
unknown samples u and known samples k, the normal equations are (R_a)_uu x_u = -(R_a)_uk x_k, where R_a is
the (banded Toeplitz) autocorrelation matrix of the AR polynomial. Segments are solved exactly in batches
of similar size; the bounds are enforced with an active-set loop.
References: A. Janssen, R. Veldhuis, L. Vries, "Adaptive interpolation of discrete-time signals that can be
modeled as autoregressive processes", IEEE TASSP 1986; S. Godsill, P. Rayner, "Digital Audio Restoration", 1998.
"""
import numpy as np
from scipy.signal import fftconvolve

from .common import make_bounds

ORDERS = (64, 128, 256)  # one pass per order
FIT_MS = 60.0            # AR fit window around each segment (RX: 60 ms)
LAG_HZ = 5.0             # Gaussian lag window (bandwidth expansion) for numerical robustness
WNC = 1e-6               # white-noise correction of the autocorrelation
MAX_UNKNOWNS = 512       # segment size limit; longer clusters are split between runs
MAX_RUN = 2048           # longer runs are left to the pilot estimate (AR cannot bridge them)
_BATCH_ELEMS = 1 << 23   # dense-solve batch size limit (B * M * M)


def _next_pow2(n):
    return np.left_shift(1, np.ceil(np.log2(np.maximum(n, 2))).astype(np.int64))


def _segments(miss, order):
    """Clusters of flagged runs in one channel: (start, end) spans holding at most MAX_UNKNOWNS flagged
    samples, merging runs separated by at most `order` reliable samples."""
    d = np.diff(np.concatenate([[0], miss.view(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    segs = []
    i = 0
    while i < len(st):
        n = en[i] - st[i]
        j = i
        while j + 1 < len(st) and st[j + 1] - en[j] <= order and n + en[j + 1] - st[j + 1] <= MAX_UNKNOWNS:
            j += 1
            n += en[j] - st[j]
        if n <= MAX_RUN:
            segs.append((int(st[i]), int(en[j])))
        i = j + 1
    return segs


def _windows(x, centers, length):
    """Rows of x (1-D) of the given length centered on `centers`, zero outside the signal."""
    idx = centers[:, None] - length // 2 + np.arange(length)[None, :]
    ok = (idx >= 0) & (idx < len(x))
    return np.where(ok, x[np.clip(idx, 0, len(x) - 1)], 0.0)


def _levinson(r, order):
    """Batched Levinson-Durbin: autocorrelations r (B, order+1) -> AR polynomials [1, a1..ap] (B, order+1)."""
    B = r.shape[0]
    a = np.zeros((B, order + 1))
    a[:, 0] = 1.0
    err = r[:, 0].copy()
    for i in range(1, order + 1):
        acc = r[:, i] + np.einsum("bj,bj->b", a[:, 1:i], r[:, i - 1:0:-1])
        k = -acc / np.maximum(err, 1e-30)
        a[:, 1:i] += k[:, None] * a[:, i - 1:0:-1]
        a[:, i] = k
        err *= 1.0 - k * k
    return a


def _ar_fit(frames, order, sr):
    """AR polynomials of Hann-windowed frames (B, W), with lag window and white-noise correction.
    Returns the autocorrelation of each polynomial, ra (B, order+1): (R_a)_ij = ra[|i - j|]."""
    W = frames.shape[1]
    nfft = _next_pow2(2 * W)
    F = np.fft.rfft(frames * np.hanning(W + 2)[1:-1], n=nfft, axis=1)
    r = np.fft.irfft(np.abs(F) ** 2, n=nfft, axis=1)[:, : order + 1]
    k = np.arange(order + 1)
    r = r * np.exp(-0.5 * (2 * np.pi * LAG_HZ * k / sr) ** 2)[None, :]
    r[:, 0] = r[:, 0] * (1.0 + WNC) + 1e-20
    a = _levinson(r, order)
    na = _next_pow2(2 * (order + 1))
    return np.fft.irfft(np.abs(np.fft.rfft(a, n=na, axis=1)) ** 2, n=na, axis=1)[:, : order + 1]


def _solve_box(G, b, lb, ub, valid, max_iter=30):
    """Batched min 1/2 x'Gx - b'x subject to lb <= x <= ub (active set). Padded entries (valid False)
    are pinned at zero."""
    pinned = ~valid
    at_lb = np.zeros_like(valid)
    at_ub = np.zeros_like(valid)
    xf = np.zeros_like(b)                                   # values of fixed entries
    eye = np.eye(G.shape[1], dtype=bool)[None]
    for _ in range(max_iter):
        fixed = pinned | at_lb | at_ub
        rhs = np.where(fixed, xf, b - np.einsum("bij,bj->bi", G, np.where(fixed, xf, 0.0)))
        Gm = np.where(fixed[:, :, None] | fixed[:, None, :], eye, G)
        x = np.linalg.solve(Gm, rhs[..., None])[..., 0]
        lo = valid & ~fixed & (x < lb)
        hi = valid & ~fixed & (x > ub)
        if lo.any() or hi.any():
            at_lb |= lo
            at_ub |= hi
            xf = np.where(lo, lb, np.where(hi, ub, xf))
            continue
        # feasible: release bounds whose multiplier has the wrong sign (the cost decreases inward)
        g = np.einsum("bij,bj->bi", G, x) - b
        rel_lb = at_lb & (g < 0)
        rel_ub = at_ub & (g > 0)
        if not (rel_lb.any() or rel_ub.any()):
            break
        at_lb &= ~rel_lb
        at_ub &= ~rel_ub
    return np.clip(x, lb, ub)


def _pass(x, miss, lb, ub, order, sr, fit_len):
    """One LSAR pass over one channel: refit the AR model around each segment on the current estimate x,
    then re-solve its flagged samples. Updates x in place."""
    segs = _segments(miss, order)
    if not segs:
        return
    T = len(x)
    spans = np.array(segs)
    centers = (spans[:, 0] + spans[:, 1]) // 2
    seg_len = spans[:, 1] - spans[:, 0]
    # AR fit windows: fit_len, or longer when the segment and the order on both sides need it
    ra = np.zeros((len(segs), order + 1))
    need = seg_len + 4 * order
    lengths = np.where(need > fit_len, _next_pow2(need), fit_len)
    for L in np.unique(lengths):
        sel = np.flatnonzero(lengths == L)
        ra[sel] = _ar_fit(_windows(x, centers[sel], int(L)), order, sr)
    # unknowns of each segment, bucketed by count for the dense solves
    unknowns = [np.flatnonzero(miss[s:e]) + s for s, e in segs]
    counts = np.array([len(u) for u in unknowns])
    for M in np.unique(_next_pow2(counts)):
        sel_all = np.flatnonzero(_next_pow2(counts) == M)
        step = max(1, _BATCH_ELEMS // (M * M))
        for c0 in range(0, len(sel_all), step):
            sel = sel_all[c0:c0 + step]
            B = len(sel)
            U = np.zeros((B, M), dtype=np.int64)
            valid = np.zeros((B, M), dtype=bool)
            for r, i in enumerate(sel):
                U[r, : counts[i]] = unknowns[i]
                valid[r, : counts[i]] = True
            U = np.where(valid, U, U[:, :1])                 # padded entries are pinned by _solve_box
            # G = (R_a)_uu
            D = np.minimum(np.abs(U[:, :, None] - U[:, None, :]), order + 1)
            ra_ext = np.concatenate([ra[sel], np.zeros((B, 1))], axis=1)
            G = ra_ext[np.arange(B)[:, None, None], D]
            # b = -(R_a x~)_u with the unknowns of x~ zeroed
            S0 = spans[sel, 0] - order
            S = int((spans[sel, 1] - spans[sel, 0]).max() + 2 * order)
            seg_idx = S0[:, None] + np.arange(S)[None, :]
            inside = (seg_idx >= 0) & (seg_idx < T)
            xt = np.where(inside, x[np.clip(seg_idx, 0, T - 1)], 0.0)
            rr, cc = np.nonzero(valid)
            xt[rr, U[rr, cc] - S0[rr]] = 0.0
            ra_sym = np.concatenate([ra[sel][:, :0:-1], ra[sel]], axis=1)       # lags -p..p
            conv = fftconvolve(xt, ra_sym, mode="full", axes=1)
            rel = np.clip(U - S0[:, None] + order, 0, conv.shape[1] - 1)
            b = -np.take_along_axis(conv, rel, axis=1)
            Uc = np.clip(U, 0, T - 1)
            lo = np.where(valid, lb[Uc], -np.inf)
            hi = np.where(valid, ub[Uc], np.inf)
            sol = _solve_box(G, np.where(valid, b, 0.0), lo, hi, valid)
            x[U[valid]] = sol[valid]


def ar_refine(y, x0, m_hi, m_lo, th_hi, th_lo, sr, orders=ORDERS, fit_ms=FIT_MS, max_gain=None, callback=None):
    """Consistent multi-pass LSAR restoration.

    y: clipped input (T, C); x0: starting estimate (T, C), e.g. y itself or another model's output.
    m_hi, m_lo: flagged samples (T, C); th_hi, th_lo: clip levels, per channel (C,) or per sample (T, C).
    orders: one pass per AR order (growing orders refine the model on the previous estimate).
    max_gain: optional cap on |x| / threshold for flagged samples (scalar or (T, C)).
    Returns the restored (T, C) estimate: reliable samples of y unchanged, flagged samples within bounds."""
    y = np.asarray(y, dtype=np.float64)
    lb, ub = make_bounds(y, m_hi, m_lo, th_hi, th_lo, max_gain)
    miss = (m_hi | m_lo).T
    x = np.where(miss.T, np.clip(np.asarray(x0, dtype=np.float64), lb.T, ub.T), y).T.copy()
    fit_len = round(fit_ms * 1e-3 * sr)
    for n, order in enumerate(orders):
        for c in range(x.shape[0]):
            _pass(x[c], miss[c], lb[c], ub[c], int(order), sr, fit_len)
        if callback is not None:
            callback(n, x.T)
    return x.T.copy()
