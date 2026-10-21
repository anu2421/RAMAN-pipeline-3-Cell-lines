import numpy as np
from scipy.signal import find_peaks

from .config import (
    ANCHOR_BANDS_CM1,
    ANCHOR_SEARCH_HALF_WIDTH_CM1,
    GLOBAL_SHIFT_ITERATIONS,
    GLOBAL_SHIFT_LEVEL,
    GLOBAL_SHIFT_MATCH_MAX_CM1,
    GLOBAL_SHIFT_MATCH_MIN_CM1,
    GLOBAL_SHIFT_MAX_CM1,
    GLOBAL_SHIFT_REFERENCE,
    GLOBAL_SHIFT_STEP_CM1,
    MAX_ALLOWED_SHIFT_CM1,
    PHE_ANCHOR,
)
from .utils import in_artifact


def to_grid(cell, common_x, shift=0.0):
    return np.interp(common_x, cell["x"] + shift, cell["y"])


def locate_peak(x, y, lo, hi):
    m = (x >= lo) & (x <= hi)
    if m.sum() < 5:
        return None
    xs, ys = x[m], y[m]
    idx, _ = find_peaks(ys)
    if len(idx) == 0:
        return None
    i = int(idx[np.argmax(ys[idx])])
    a, b = max(0, i - 2), min(len(xs), i + 3)
    if b - a >= 3:
        c = np.polyfit(xs[a:b] - xs[i], ys[a:b], 2)
        if c[0] < 0:
            v = -c[1] / (2 * c[0])
            if xs[a] - xs[i] <= v <= xs[b - 1] - xs[i]:
                return float(xs[i] + v)
    return float(xs[i])


def _best_global_shift(x, y, reference, common_x, match_mask):
    candidates = np.arange(-GLOBAL_SHIFT_MAX_CM1, GLOBAL_SHIFT_MAX_CM1 + 1e-9, GLOBAL_SHIFT_STEP_CM1)
    ref = reference[match_mask]
    scores = []
    for s in candidates:
        moved = np.interp(common_x, x + s, y)[match_mask]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.corrcoef(moved, ref)[0, 1]
        scores.append(r if np.isfinite(r) else -np.inf)
    return float(candidates[int(np.argmax(scores))])


def shifts_global(data, common_x):
    match_mask = ((common_x >= GLOBAL_SHIFT_MATCH_MIN_CM1) & (common_x <= GLOBAL_SHIFT_MATCH_MAX_CM1)
                  & ~in_artifact(common_x))
    shifts = {k: {r: np.zeros(len(c)) for r, c in reps.items()} for k, reps in data.items()}

    def reference_for(dataset):
        if GLOBAL_SHIFT_REFERENCE == "own_control":
            ref_keys = [(dataset, "control")]
        else:
            ref_keys = [k for k in data if k[1] == "control"]
        ref_keys = [k for k in ref_keys if k in data] or list(data.keys())

        return np.mean([np.mean([to_grid(c, common_x, shifts[k][r][i])
                                 for r, cells in data[k].items() for i, c in enumerate(cells)], axis=0)
                        for k in ref_keys], axis=0)

    for _ in range(max(1, GLOBAL_SHIFT_ITERATIONS)):
        refs = {ds: reference_for(ds) for ds in {k[0] for k in data}}
        for key, reps in data.items():
            for rep, cells in reps.items():
                if GLOBAL_SHIFT_LEVEL == "replicate":
                    mean = np.mean([to_grid(c, common_x) for c in cells], axis=0)
                    shifts[key][rep][:] = _best_global_shift(common_x, mean, refs[key[0]], common_x, match_mask)
                else:
                    for i, c in enumerate(cells):
                        shifts[key][rep][i] = _best_global_shift(c["x"], c["y"], refs[key[0]], common_x, match_mask)

    if PHE_ANCHOR:
        ctrl = [k for k in data if k[1] == "control"] or list(data.keys())
        ref = np.mean([np.mean([to_grid(c, common_x, shifts[k][r][i])
                                for r, cells in data[k].items() for i, c in enumerate(cells)], axis=0)
                       for k in ctrl], axis=0)
        offs = []
        for t in ANCHOR_BANDS_CM1:
            pos = locate_peak(common_x, ref, t - ANCHOR_SEARCH_HALF_WIDTH_CM1, t + ANCHOR_SEARCH_HALF_WIDTH_CM1)
            if pos is not None:
                offs.append(t - pos)
                print(f"  anchor band {t:.0f}: found at {pos:.2f} (offset {t - pos:+.2f})")
        if offs and abs(float(np.median(offs))) <= MAX_ALLOWED_SHIFT_CM1:
            delta = float(np.median(offs))
            for k in shifts:
                for r in shifts[k]:
                    shifts[k][r] += delta
            print(f"  Multi-band anchor: median offset {delta:+.2f} cm-1 applied to every spectrum "
                  f"(spread of band offsets {min(offs):+.2f} to {max(offs):+.2f})")
        else:
            print("  [WARN] anchor bands not found -> no absolute anchor applied")
    return shifts
