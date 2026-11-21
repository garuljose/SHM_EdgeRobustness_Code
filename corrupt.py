"""Missingness / corruption injection: MCAR, MAR (burst), MNAR (amplitude-dependent)."""
import numpy as np


def mcar(x, rate, rng):
    """Missing completely at random: independent random sample dropout, per channel."""
    mask = rng.random(x.shape) < rate
    xc = x.copy()
    xc[mask] = 0.0
    return xc, mask


def mar_burst(x, rate, rng, burst_len=10):
    """Missing at random, simulated as contiguous burst dropout (sensor/link outage)."""
    T, C = x.shape
    xc = x.copy()
    mask = np.zeros_like(x, dtype=bool)
    target = int(rate * T)
    for c in range(C):
        missing = 0
        guard = 0
        while missing < target and guard < 10000:
            start = rng.integers(0, max(T - burst_len, 1))
            end = min(start + burst_len, T)
            mask[start:end, c] = True
            missing = mask[:, c].sum()
            guard += 1
    xc[mask] = 0.0
    return xc, mask


def mnar_clip(x, rate, rng):
    """Missing not at random: dropout correlated with signal amplitude (clipping
    of the largest-amplitude samples, e.g. sensor saturation during high-amplitude
    events -- often exactly when damage-relevant transients occur)."""
    xc = x.copy()
    mask = np.zeros_like(x, dtype=bool)
    T, C = x.shape
    for c in range(C):
        thresh = np.quantile(np.abs(x[:, c]), 1 - rate)
        idx = np.abs(x[:, c]) >= thresh
        mask[idx, c] = True
    xc[mask] = 0.0
    return xc, mask


CORRUPTION_FNS = {"MCAR": mcar, "MAR": mar_burst, "MNAR": mnar_clip}


def linear_interpolate(x, mask):
    """Per-channel linear interpolation to fill masked (zeroed) samples."""
    xc = x.copy()
    T, C = x.shape
    t = np.arange(T)
    for c in range(C):
        m = mask[:, c]
        if m.all():
            xc[:, c] = 0.0
            continue
        if m.any():
            xc[m, c] = np.interp(t[m], t[~m], xc[~m, c])
    return xc
