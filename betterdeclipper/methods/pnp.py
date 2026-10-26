"""Signal-domain plug-and-play declipping: x <- Denoise_lambda(P_Gamma(x)) with FISTA momentum.

The denoiser is a time-frequency shrinkage (PEW) in a Parseval STFT; averaging it over several
time shifts of the STFT grid ("cycle spinning") makes it translation invariant. lambda is
annealed geometrically (continuation).

Options:
 - chan_gain: lambda multiplier per (PCA-rotated) channel, e.g. [1, 2] shrinks the minor
   (side-like) component harder.
 - fweight: frequency-dependent lambda ~ (f/f0)^fweight.
 - pilot: a previous estimate whose smoothed TF energy is used (mixed with the current energy
   by `pilot_mix`) in the Wiener gain -> BM3D-like second stage.

Signals are solved as a batch (B, C, T): one call can solve several equally long signals at once.
"""
import math
import os
import warnings
import numpy as np
import torch

torch.set_flush_denormal(True)  # denormals are very slow on older x86 CPUs
import torch.nn.functional as Fnn

from ..stft import TightSTFT
from .common import make_bounds, pad_bounds, Box, threshold_scale, ChannelMix, DipProj
from .social import _neigh_kernel, channel_mixing


def _tf_smooth(a2, kernel):
    """(..., F, K) energies smoothed over a time-frequency neighborhood kernel (1, 1, kt, kf)."""
    kt, kf = kernel.shape[-2:]
    e = Fnn.conv2d(a2.reshape(-1, 1, *a2.shape[-2:]), kernel, padding=(kt // 2, kf // 2))
    return e.reshape(a2.shape)


class PEWDenoiser:
    def __init__(self, stft, neighs, device, dtype, shifts=1, combine="max", chan_gain=None, fw=None,
                 pilot_e=None, pilot_mix=0.0, gain_mode="pew"):
        self.stft = stft
        self.kernels = [_neigh_kernel(nb[0], nb[1], device, dtype) for nb in neighs]
        self.shifts = [int(round(i * stft.hop / shifts)) for i in range(shifts)]
        self.combine = combine
        self.chan_gain = None if chan_gain is None else torch.as_tensor(chan_gain, dtype=dtype, device=device)[:, None, None]
        self.fw = fw
        self.pilot_e = pilot_e  # list per shift of (C, F, K) energies
        self.pilot_mix = pilot_mix
        self.gain_mode = gain_mode

    def fresh(self):
        return self  # stateless between calls

    def energy(self, a2):
        es = [_tf_smooth(a2, k) for k in self.kernels]
        if len(es) == 1:
            return es[0]
        e = torch.stack(es, 0)
        return e.max(0).values if self.combine == "max" else e.mean(0)

    def __call__(self, x, lam2):
        """x: (B, C, Tp). lam2: squared threshold per signal ((B, 1, 1, 1) tensor)."""
        Tp = x.shape[-1]
        out = torch.zeros_like(x)
        if self.chan_gain is not None:
            lam2 = lam2 * self.chan_gain ** 2
        if self.fw is not None:
            lam2 = lam2 * self.fw ** 2
        for si, s in enumerate(self.shifts):
            xs = torch.roll(x, -s, dims=-1) if s else x
            z = self.stft.analysis(xs)
            a2 = z.real ** 2 + z.imag ** 2
            e = self.energy(a2)
            if self.pilot_e is not None and self.pilot_mix > 0:
                e = (1 - self.pilot_mix) * e + self.pilot_mix * self.pilot_e[si]
            if self.gain_mode == "wiener":
                g = e / (e + lam2)
            else:
                g = torch.clamp(1.0 - lam2 / (e + 1e-30), min=0.0)
            xd = self.stft.synthesis(z.mul_(g), Tp)
            out += torch.roll(xd, s, dims=-1) if s else xd
        return out / len(self.shifts)


class NMFDenoiser:
    """Wiener-type shrinkage whose signal power comes from a low-rank NMF model of the current
    iterate's power spectrogram (templates shared by all channels of a signal, warm-started across calls).
    gain_mode 'wiener': V/(V+lam^2); 'pew': max(0, 1 - lam^2/V)."""

    def __init__(self, stft, rank=32, nmf_iter=2, beta=1.0, gain_mode="wiener", smooth_mix=0.0,
                 pew_kernel=None, device="cpu", dtype=torch.float32, chan_gain=None):
        self.stft, self.rank, self.nmf_iter, self.beta = stft, rank, nmf_iter, beta
        self.gain_mode, self.smooth_mix, self.pew_kernel = gain_mode, smooth_mix, pew_kernel
        self.W = None
        self.H = None
        self.static = False  # True: update W, H, cached V in place (CUDA graph replay)
        self.device, self.dtype = device, dtype
        self.shifts = [0]
        self.chan_gain = None if chan_gain is None else torch.as_tensor(chan_gain, dtype=dtype, device=device)[:, None, None]

    def energy(self, a2):  # used for pilot energies (same interface as PEWDenoiser)
        return a2

    def fresh(self):
        """Reset the NMF state (used when a CUDA-graph solve has to be redone eagerly)."""
        self.W = self.H = None
        self._V = None
        self.static = False
        return self

    def _fit(self, P, n_iter):
        """P: (B, N, K) power spectrogram rows of each signal; templates W (B, K, rank), activations H (B, N, rank)."""
        eps = 1e-12
        if self.W is None:
            g = torch.Generator().manual_seed(0)
            B, N, K = P.shape
            m = torch.sqrt(P.mean((1, 2), keepdim=True))
            self.W = (torch.rand(K, self.rank, generator=g, dtype=self.dtype) + 0.1).to(self.device) * m
            self.H = (torch.rand(N, self.rank, generator=g, dtype=self.dtype) + 0.1).to(self.device) * m
            self._V = None
        W, H, b = self.W, self.H, self.beta
        for i in range(n_iter):
            # H, W are unchanged since the previous call returned H @ W.T, so reuse it (exact)
            V = self._V if (i == 0 and self._V is not None) else H @ W.mT + eps
            if b == 1.0:  # KL: denominators are column sums (ones @ W == W.sum over K)
                H = H * ((P / V) @ W) / (W.sum(1, keepdim=True) + eps)
                V = H @ W.mT + eps
                W = W * ((P / V).mT @ H) / (H.sum(1, keepdim=True) + eps)
            else:
                H = H * ((V ** (b - 2) * P) @ W) / ((V ** (b - 1)) @ W + eps)
                V = H @ W.mT + eps
                W = W * ((V ** (b - 2) * P).mT @ H) / ((V ** (b - 1)).mT @ H + eps)
            # normalize templates (scale into activations)
            s = W.sum(1, keepdim=True) + eps
            W = W / s
            H = H * s
        V = H @ W.mT
        if self.static:
            self.W.copy_(W)
            self.H.copy_(H)
            self._V.copy_(V + eps)
        else:
            self.W, self.H = W, H
            self._V = V + eps
        return V

    def __call__(self, x, lam2):
        """x: (B, C, Tp). lam2: squared threshold per signal ((B, 1, 1, 1) tensor)."""
        Tp = x.shape[-1]
        z = self.stft.analysis(x)
        B, C, F, K = z.shape
        a2 = z.real ** 2 + z.imag ** 2
        n_it = self.nmf_iter if self.W is not None else 50
        V = self._fit(a2.reshape(B, C * F, K), n_it).reshape(B, C, F, K)
        if self.smooth_mix > 0 and self.pew_kernel is not None:
            V = (1 - self.smooth_mix) * V + self.smooth_mix * _tf_smooth(a2, self.pew_kernel)
        if self.chan_gain is not None:
            lam2 = lam2 * self.chan_gain ** 2
        if self.gain_mode == "wiener":
            g = V + lam2
            torch.div(V, g, out=g)
        else:
            g = torch.clamp(1.0 - lam2 / (V + 1e-30), min=0.0)
        return self.stft.synthesis(z.mul_(g), Tp)


def declip_pnp(y, m_hi, m_lo, th_hi, th_lo, sr=44100, win_len=4096, hop=1024, neigh=(3, 7), neighs=None,
               combine="max", shifts=1, n_iter=400, lam0=0.1, lam1=1e-4, stereo="pca", momentum=True,
               chan_gain=None, fweight=None, pilot=None, pilot_mix=0.0, gain_mode="pew", relax=1.0,
               device="cpu", dtype=torch.float32, callback=None, x_init=None, lam_ref=None,
               den_type="pew", nmf_rank=32, nmf_iter=2, nmf_beta=1.0, nmf_smooth=0.0, nmf_rank_ratio=None,
               max_gain=None, dips=None):
    """Declip y (T, C). Batch: lists of equally long signals for y, m_hi, m_lo, th_hi, th_lo (and lam_ref,
    max_gain when given) are solved together, each with its own scale, channel rotation, lambda schedule
    and NMF factors: the same estimates as one call per signal (returned as a list), in one GPU solve."""
    batch = isinstance(y, (list, tuple))
    if not batch:
        y, m_hi, m_lo, th_hi, th_lo, lam_ref, max_gain = [y], [m_hi], [m_lo], [th_hi], [th_lo], [lam_ref], [max_gain]
    elif pilot is not None or x_init is not None or dips is not None or callback is not None:
        raise ValueError("pilot, x_init, dips and callback need a single signal")
    B = len(y)
    T, C = y[0].shape
    lam_ref = lam_ref if lam_ref is not None else [None] * B
    max_gain = max_gain if max_gain is not None else [None] * B
    scale = np.array([threshold_scale(a, b) for a, b in zip(th_hi, th_lo)])
    stft = TightSTFT(win_len, hop, None, device, dtype)
    left, right = stft.pad_len(T)
    # extra right padding so that circular shifts only wrap free (padded) samples, then to whole hops
    right += win_len
    right += (hop - (T + left + right - win_len) % hop) % hop
    Tp = T + left + right
    lb, ub = zip(*(pad_bounds(*make_bounds(yb / s, mh, ml, th / s, tl / s, mg), left, right)
                   for yb, mh, ml, th, tl, mg, s in zip(y, m_hi, m_lo, th_hi, th_lo, max_gain, scale)))
    proj = Box(np.stack(lb), np.stack(ub), device, dtype)
    if dips is not None:  # limiter restoration: (idx (T, C) event ids or -1, v (T, C) dip shape, n_events, bmax)
        didx, dv, n_ev, bmax = dips
        ip = np.full((1, C, Tp), -1, dtype=np.int64); ip[0, :, left:left + T] = didx.T
        vp = np.zeros((1, C, Tp)); vp[0, :, left:left + T] = dv.T
        yp = np.zeros((1, C, Tp)); yp[0, :, left:left + T] = y[0].T / scale[0]
        proj = DipProj(proj, torch.as_tensor(yp, dtype=dtype, device=device), torch.as_tensor(ip, device=device),
                       torch.as_tensor(vp, dtype=dtype, device=device), n_ev, bmax)
    cm = ChannelMix(np.stack([channel_mixing(yb, stereo) for yb in y]), device, dtype)
    mix, unmix = cm.mix, cm.unmix

    def to_pad(a):  # list of (T, C) signals -> (B, C, Tp) normalized and padded
        v = torch.zeros(len(a), C, Tp, dtype=dtype, device=device)
        v[:, :, left:left + T] = torch.as_tensor(np.stack([ab.T / s for ab, s in zip(a, scale)]), dtype=dtype, device=device)
        return v

    fw = None
    if fweight:
        K = (stft.nfft // 2 + 1)
        f = torch.arange(K, dtype=dtype, device=device).clamp(min=1.0) / (K * 0.05)
        fw = f ** float(fweight)
        fw = fw / fw.mean()
    if den_type == "nmf":
        if nmf_rank_ratio is not None:
            # rank proportional to the number of spectrogram rows (channels x frames) of this chunk
            n_rows = C * ((Tp - win_len) // hop + 1)
            nmf_rank = int(max(16, min(nmf_rank, round(nmf_rank_ratio * n_rows))))
        den = NMFDenoiser(stft, nmf_rank, nmf_iter, nmf_beta, gain_mode, nmf_smooth,
                          _neigh_kernel(neigh[0], neigh[1], device, dtype), device, dtype, chan_gain)
    else:
        den = PEWDenoiser(stft, neighs or [neigh], device, dtype, shifts, combine, chan_gain, fw, None, pilot_mix, gain_mode)
    if pilot is not None:
        pu = unmix(to_pad([pilot]))
        pe = []
        for s in den.shifts:
            zp = stft.analysis(torch.roll(pu, -s, dims=-1) if s else pu)
            pe.append(den.energy(zp.real ** 2 + zp.imag ** 2))
        den.pilot_e = pe

    x0 = to_pad(y if x_init is None else [x_init])
    # lambda reference: max |coefficient| of the (normalized) clipped signal; pass lam_ref to use a
    # file-global value so that chunks of a long file share the same schedule
    if any(r is None for r in lam_ref):
        zy = torch.abs(stft.analysis(unmix(to_pad(y)))).amax((1, 2, 3)).tolist()
        lam_ref = [z if r is None else r for r, z in zip(lam_ref, zy)]
    lam2s = np.stack([np.geomspace(lam0 * float(z), lam1 * float(z), n_iter) ** 2 for z in lam_ref], 1)
    lam2s = torch.as_tensor(lam2s, dtype=dtype, device=device).reshape(n_iter, B, 1, 1, 1)
    coefs = []  # FISTA momentum coefficients (t_k - 1) / t_{k+1}
    t = 1.0
    for _ in range(n_iter):
        tn = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        coefs.append((t - 1) / tn if momentum else 0.0)
        t = tn

    use_graph = (torch.device(device).type == "cuda" and callback is None and relax == 1.0 and n_iter > 4
                 and os.environ.get("BD_CUDA_GRAPHS", "1") != "0")
    x = None
    if use_graph:
        try:
            x = _solve_graph(x0, lam2s, coefs, proj, mix, unmix, den)
        except Exception as e:  # never fail because of graph capture: redo the solve eagerly
            warnings.warn(f"CUDA graph path failed ({e!r}); falling back to eager mode")
            den = den.fresh()
            x = None
    if x is None:
        x = _solve_eager(x0, lam2s, coefs, proj, mix, unmix, den, relax,
                         None if callback is None else (lambda it, xx: callback(
                             it, proj(xx)[0, :, left:left + T].T.cpu().numpy().astype(np.float64) * scale[0])))
    x = proj(x)[:, :, left:left + T].transpose(1, 2).cpu().numpy().astype(np.float64) * scale[:, None, None]
    return list(x) if batch else x[0]


def _solve_eager(x, lam2s, coefs, proj, mix, unmix, den, relax=1.0, callback=None):
    xbar = x.clone()
    for it, (lam2, c) in enumerate(zip(lam2s, coefs)):
        px = proj(xbar)
        if relax != 1.0:
            px = xbar + relax * (px - xbar)
        xn = mix(den(unmix(px), lam2))
        xbar = xn + c * (xn - x) if c else xn
        x = xn
        if callback is not None and (it % 50 == 49 or it == len(lam2s) - 1):
            callback(it, x)
    return x


def _solve_graph(x, lam2s, coefs, proj, mix, unmix, den):
    """Same iterations as _solve_eager, but one iteration is captured as a CUDA graph and replayed:
    the ~60 small kernel launches per iteration otherwise dominate on slower host CPUs.
    State lives in static buffers updated in place; lambda^2 (per signal) and the momentum coefficient
    are device tensors refreshed before each replay."""
    x_s = x.clone()
    xbar_s = x.clone()
    lam2_t = lam2s[0].clone()
    c_t = torch.zeros((), dtype=x.dtype, device=x.device)

    def step():
        xn = mix(den(unmix(proj(xbar_s)), lam2_t))
        xbar_new = xn + c_t * (xn - x_s)
        x_s.copy_(xn)
        xbar_s.copy_(xbar_new)

    n = len(lam2s)
    # iteration 0 eagerly: the NMF denoiser initializes its factors on the first call
    c_t.fill_(coefs[0])
    step()
    if hasattr(den, "static"):
        den.static = True
    # two more real iterations on a side stream (creates cuFFT plans / cuBLAS handles before capture)
    side = torch.cuda.Stream(device=x.device)
    side.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(side):
        for it in (1, 2):
            lam2_t.copy_(lam2s[it]); c_t.fill_(coefs[it])
            step()
    torch.cuda.current_stream(x.device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()  # recorded, not executed
    for it in range(3, n):
        lam2_t.copy_(lam2s[it]); c_t.fill_(coefs[it])
        graph.replay()
    return x_s
