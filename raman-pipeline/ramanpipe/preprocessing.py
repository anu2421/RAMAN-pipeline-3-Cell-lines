import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .config import (
    ALS_LAMBDA,
    ALS_NITER,
    ALS_P,
    QUARTZ_CENTER_RANGE_CM1,
    QUARTZ_FIT_WINDOW_CM1,
    QUARTZ_SIGMA_RANGE_CM1,
    ROI_MAX_CM1,
    ROI_MIN_CM1,
    SG_POLYORDER,
    SG_WINDOW,
    SPIKE_THRESHOLD,
    SPIKE_WINDOW,
)
from .utils import in_artifact, trapz


def remove_cosmic_spikes(y, window=5, threshold=7.0):
    if window % 2 == 0:
        window += 1
    y = y.copy()
    med = medfilt(y, kernel_size=window)
    residual = y - med
    mad = np.median(np.abs(residual - np.median(residual))) + 1e-9
    spike_mask = np.abs(residual) > threshold * mad * 1.4826
    y[spike_mask] = med[spike_mask]
    return y


def apply_roi(x, y, roi_min, roi_max):
    mask = (x >= roi_min) & (x <= roi_max)
    return x[mask], y[mask]


def smooth_savgol(y, window=11, polyorder=3):
    window = min(window, len(y) - (1 - len(y) % 2))
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        window = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    if window >= len(y):
        return y
    return savgol_filter(y, window_length=window, polyorder=polyorder)


def als_baseline(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    D = D.tocsc()
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    z = np.zeros(L)
    for _ in range(niter):
        W.setdiag(w)
        z = spsolve((W + D).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def normalize_spectrum(x, y, method):
    keep = ~in_artifact(x)
    if not keep.any():
        keep = np.ones_like(keep)
    if method == "area":
        edges = np.flatnonzero(np.diff(np.concatenate(([0], keep.astype(int), [0]))))
        area = sum(trapz(np.abs(y[a:b])) for a, b in zip(edges[::2], edges[1::2]))
        return y / area if area != 0 else y
    if method == "l2":
        norm = np.linalg.norm(y[keep])
        return y / norm if norm != 0 else y
    raise ValueError(f"Unknown normalization method: {method}")


def remove_quartz_800(x, y):
    from scipy.optimize import curve_fit
    lo, hi = QUARTZ_FIT_WINDOW_CM1
    m = (x >= lo) & (x <= hi)
    if m.sum() < 30:
        return y, np.nan
    xs, ys = x[m], y[m]
    g = lambda xx, a, c, w: a * np.exp(-(xx - c) ** 2 / (2 * w * w))

    def model(xx, a1, c1, w1, a2, c2, w2, aq, cq, wq, b0, b1):
        return g(xx, a1, c1, w1) + g(xx, a2, c2, w2) + g(xx, aq, cq, wq) + b0 + b1 * (xx - 800.0)

    base = float(np.percentile(ys, 5))
    plateau = max(float(np.median(ys[(xs > 800) & (xs < 840)]) - base), 0.0) if np.any((xs > 800) & (xs < 840)) else 0.0
    peak783 = max(float(ys[(xs > 775) & (xs < 792)].max() - base - plateau), 1e-9) if np.any((xs > 775) & (xs < 792)) else 1e-9
    p0 = [peak783, 783, 5, max(plateau * 0.3, 1e-9), 852, 6, max(plateau, 1e-9), 805, 22, base, 0.0]
    lb = [0, 776, 2, 0, 845, 2, 0, QUARTZ_CENTER_RANGE_CM1[0], QUARTZ_SIGMA_RANGE_CM1[0], -np.inf, -np.inf]
    ub = [np.inf, 790, 10, np.inf, 860, 12, np.inf, QUARTZ_CENTER_RANGE_CM1[1], QUARTZ_SIGMA_RANGE_CM1[1], np.inf, np.inf]
    p0 = [min(max(v, l + 1e-9), u - 1e-9) if np.isfinite(u) else max(v, l) for v, l, u in zip(p0, lb, ub)]
    try:
        p, _ = curve_fit(model, xs, ys, p0=p0, bounds=(lb, ub), maxfev=20000)
    except Exception:
        return y, np.nan
    return y - g(x, p[6], p[7], p[8]), float(p[6])


def preprocess_until_baseline(x, y, label=""):
    y = remove_cosmic_spikes(y, window=SPIKE_WINDOW, threshold=SPIKE_THRESHOLD)
    x, y = apply_roi(x, y, ROI_MIN_CM1, ROI_MAX_CM1)
    if len(x) < max(SG_POLYORDER + 2, 5):
        print(f"    [WARN] {label}: too few points after ROI cut, skipping.")
        return None
    y = smooth_savgol(y, window=SG_WINDOW, polyorder=SG_POLYORDER)
    y = y - als_baseline(y, lam=ALS_LAMBDA, p=ALS_P, niter=ALS_NITER)
    return x, y
