"""BetterDeclipper: offline, high-accuracy audio declipping.

Python API (torch is imported on first use):
    import soundfile as sf
    from betterdeclipper import declip
    y, sr = sf.read("in.wav", always_2d=True)
    x, info = declip(y, sr, preset="normal", mode="auto", progress=lambda step, n, seconds: None)
info["analysis"] holds the automatic analysis; betterdeclipper.cli.analysis_lines(info, channels) formats it.
    x, reduction_db = limit_true_peak(x, sr, ceiling_db=-0.1)   # keep the loudness under a true-peak ceiling
"""
__version__ = "0.3.0"
__all__ = ["declip", "limit_true_peak", "PRESETS", "__version__"]


def __getattr__(name):  # lazy, so that `import betterdeclipper` stays light
    if name in ("declip", "PRESETS"):
        from . import engine
        return getattr(engine, name)
    if name == "limit_true_peak":
        from .limiter import limit_true_peak
        return limit_true_peak
    raise AttributeError(f"module 'betterdeclipper' has no attribute {name!r}")
