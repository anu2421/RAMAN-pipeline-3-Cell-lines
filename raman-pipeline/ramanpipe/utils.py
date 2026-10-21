import os
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import find_peaks

from .config import (
    ARTIFACT_REGIONS_CM1,
    EXPOSURE,
    PASTEL_FILL,
)
from . import state


def bar_fill(c):
    return PASTEL_FILL.get(c, c)


def safe_name(s):
    return re.sub(r'[^\w\-]+', '_', s)


def in_artifact(x):
    m = np.zeros(np.shape(x), dtype=bool)
    for lo, hi in ARTIFACT_REGIONS_CM1:
        m |= (x >= lo) & (x <= hi)
    return m


def _segments(n):
    b = [0] + [int(i) for i in state.GRID_BREAKS if 0 < i < n] + [n]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] > b[i]]


def segment_of(idx, n):
    for s, e in _segments(n):
        if s <= idx < e:
            return s, e
    return 0, n


def find_peaks_segmented(y, **kw):
    out = []
    for s, e in _segments(len(y)):
        if e - s < 3:
            continue
        idx, _ = find_peaks(y[s:e], **kw)
        out.extend(idx + s)
    return np.array(sorted(out), dtype=int)


def gapped(x, y):
    return x, y


def shade_color(base_color, factor):
    r, g, b = mcolors.to_rgb(base_color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def rescale01(y):
    rng = y.max() - y.min()
    return (y - y.min()) / rng if rng != 0 else y


def trapz(y):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y)


def run_title(ds):
    return f"{ds} {EXPOSURE}" if EXPOSURE else ds


def wrapped_title(label, *args, **kwargs):
    if isinstance(label, str):
        label = "\n".join(textwrap.fill(line, 50) if "$" not in line else line for line in label.split("\n"))
    return plt.gca().set_title(label, *args, **kwargs)


def read_accuracy(path):
    try:
        df = pd.read_csv(path, index_col=0)
        return float(df.loc["accuracy", "f1-score"]), float(df.loc["macro avg", "f1-score"])
    except Exception:
        return np.nan, np.nan


def install_safe_savefig():
    import time
    from matplotlib.figure import Figure
    original = Figure.savefig
    if getattr(original, "_safe", False):
        return

    def safe(self, fname, *args, **kwargs):
        try:
            return original(self, fname, *args, **kwargs)
        except OSError as e:
            if not isinstance(fname, (str, os.PathLike)):
                raise
            root, ext = os.path.splitext(str(fname))
            alt = f"{root}_{time.strftime('%H%M%S')}{ext}"
            print(f"  [WARN] could not write {os.path.basename(str(fname))} ({e.strerror}); "
                  f"is it open in another program? Saved as {os.path.basename(alt)} instead.")
            return original(self, alt, *args, **kwargs)
    safe._safe = True
    Figure.savefig = safe
