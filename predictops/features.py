"""Window features. Training and live scoring both call this, so they cannot drift apart."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from predictops.sim import COLS

W = 60  # ticks of history behind each prediction
LIMITS = {COLS.index("mem"): 98.0, COLS.index("disk"): 99.0}
NAMES = ([f"{c}_last" for c in COLS] + [f"{c}_mean" for c in COLS] + [f"{c}_slope" for c in COLS]
         + [f"{c}_slope10" for c in COLS] + ["mem_tte", "disk_tte"])


def _slope(win):
    x = np.arange(win.shape[-1]) - (win.shape[-1] - 1) / 2
    return (win * x).sum(-1) / (x**2).sum()  # least squares; sum(x) == 0 so the intercept drops out


def window_features(series):
    """series: (ticks, len(COLS)) with NaN where the host was down.

    Returns (ticks - W + 1, len(NAMES)); row i describes the window ending at tick i + W - 1.
    Rows touching a NaN are NaN.
    """
    win = sliding_window_view(series, W, axis=0)  # (n, cols, W)
    last = win[..., -5:].mean(-1)
    slope = _slope(win)
    with np.errstate(divide="ignore", invalid="ignore"):
        tte = [np.where(slope[:, c] > 1e-3, (lim - last[:, c]) / slope[:, c], 1e4).clip(0, 1e4)
               for c, lim in LIMITS.items()]
    feats = np.column_stack([last, win.mean(-1), slope, _slope(win[..., -10:]), *tte]).astype(np.float32)
    feats[np.isnan(win).any((1, 2))] = np.nan
    return feats
