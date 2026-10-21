import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from .config import (
    CLUSTER_ALPHA,
    CONSENSUS_MATCH_TOL_CM1,
    CONSENSUS_MAX_SPAN_CM1,
    LOCAL_BASELINE_NEAR_CUT_CM1,
    LOCAL_BASELINE_SPAN_CM1,
    MIN_ETA_SQUARED,
    PEAK_DISTANCE,
    PEAK_INTENSITY_HALF_WINDOW_CM1,
    PEAK_PROMINENCE,
    PEAK_SEARCH_WINDOW,
    POSTHOC_ALPHA,
    REFERENCE_BANDS_CM1,
    SNAP_TOL_CM1,
    TAIL_FIT_REGIONS_CM1,
)
from .utils import find_peaks_segmented, in_artifact, rescale01, segment_of
from .stats import anova_F_vectorized, games_howell_pairwise


def _refined_local_max(x, y, idx, half_pts):
    s, e = segment_of(idx, len(y))
    lo, hi = max(s, idx - half_pts), min(e, idx + half_pts + 1)
    i = lo + int(np.argmax(y[lo:hi]))
    if lo < i < hi - 1:
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(x[i] + 0.5 * (y0 - y2) / den * (x[i + 1] - x[i]))
    return float(x[i])


def build_consensus_peaks(common_x, group_means):
    hits = []
    for lab, m in group_means.items():
        for i in find_peaks_segmented(rescale01(m), prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE):
            hits.append((float(common_x[i]), lab))
    hits.sort()
    clusters = []
    for wn, lab in hits:
        if clusters and wn - clusters[-1][-1][0] <= CONSENSUS_MATCH_TOL_CM1\
                and wn - clusters[-1][0][0] <= CONSENSUS_MAX_SPAN_CM1:
            clusters[-1].append((wn, lab))
        else:
            clusters.append([(wn, lab)])

    grand = rescale01(np.mean(np.vstack(list(group_means.values())), axis=0))
    rows = []
    for c in clusters:
        pos = float(np.median([w for w, _ in c]))
        ref = min(REFERENCE_BANDS_CM1, key=lambda r: abs(r - pos))
        if abs(ref - pos) <= SNAP_TOL_CM1:
            pos = float(ref)
        idx = int(np.argmin(np.abs(common_x - pos)))
        labs = sorted({l for _, l in c})
        rows.append({"consensus_cm-1": int(round(common_x[idx])), "idx": idx,
                     "n_groups_detected": len(labs), "n_groups_total": len(group_means),
                     "observed_min_cm-1": min(w for w, _ in c), "observed_max_cm-1": max(w for w, _ in c),
                     "grand_mean_height": round(float(grand[idx]), 3),
                     "detected_in": "; ".join(labs)})
    df = pd.DataFrame(rows).drop_duplicates("consensus_cm-1").reset_index(drop=True)
    df = df[~in_artifact(df["consensus_cm-1"].to_numpy(dtype=float))].reset_index(drop=True)
    return df


def _band_height_on_tail(x, y, idx, s, e):
    from scipy.optimize import curve_fit
    x0 = float(x[idx])
    hi = min(e, idx + int(round(LOCAL_BASELINE_SPAN_CM1 / max(x[1] - x[0], 1e-9))) + 6)
    lo = max(s, idx - 25)
    xs, ys = x[lo:hi], y[lo:hi]

    def model(xx, A, c, w, b0, b1, b2):
        return A * np.exp(-(xx - c) ** 2 / (2 * w * w)) + b0 + b1 * (xx - x0) + b2 * (xx - x0) ** 2

    try:
        amp0 = max(float(y[idx] - ys.min()), 1e-12)
        p, _ = curve_fit(model, xs, ys, p0=[amp0, x0, 4.0, float(ys.min()), 0.0, 0.0],
                         bounds=([0, x0 - 6, 1.5, -np.inf, -np.inf, -np.inf],
                                 [np.inf, x0 + 6, 12.0, np.inf, np.inf, np.inf]), maxfev=20000)
        return float(p[0])
    except Exception:
        li = lo + int(np.argmin(y[lo:max(lo + 1, idx - 2)]))
        ri = idx + 3 + int(np.argmin(y[idx + 3:hi])) if hi > idx + 3 else hi - 1
        base = y[li] + (y[ri] - y[li]) * (x0 - x[li]) / ((x[ri] - x[li]) or 1)
        return float(y[idx] - base)


def analyze_consensus_peaks(common_x, group_arrays, keys, display_names, sig_mask, consensus_df):
    n = len(common_x)
    step = float(np.median(np.diff(common_x)))
    half = max(0, int(round(PEAK_INTENSITY_HALF_WINDOW_CM1 / step)))
    tol_pts = max(1, int(round(CONSENSUS_MATCH_TOL_CM1 / step)))
    means_by_key = {k: group_arrays[k].mean(axis=0) for k in keys}
    records = []
    for _, cp in consensus_df.iterrows():
        idx = int(cp["idx"])
        s, e = segment_of(idx, n)
        lo, hi = max(s, idx - half), min(e, idx + half + 1)
        samples = [group_arrays[k][:, lo:hi].mean(axis=1) for k in keys]
        near_cut = any(min(abs(common_x[idx] - a), abs(common_x[idx] - b)) <= LOCAL_BASELINE_NEAR_CUT_CM1
                       for a, b in TAIL_FIT_REGIONS_CM1)
        if near_cut:
            samples = [np.array([_band_height_on_tail(common_x, row, idx, s, e) for row in group_arrays[k]])
                       for k in keys]
        X = np.concatenate(samples)[:, None]
        gi = np.concatenate([np.full(len(sm), i) for i, sm in enumerate(samples)])
        F, eta, dfn, dfd = anova_F_vectorized(X, gi, len(keys))
        F, eta = float(F[0]), float(eta[0])
        p_anova = float(f_dist.sf(F, dfn, dfd))
        wlo, whi = max(s, idx - PEAK_SEARCH_WINDOW), min(e, idx + PEAK_SEARCH_WINDOW + 1)
        in_cluster = bool(sig_mask[wlo:whi].any())
        passes = in_cluster and eta >= MIN_ETA_SQUARED

        means = {k: float(np.mean(sm)) for k, sm in zip(keys, samples)}
        sems = {k: float(np.std(sm, ddof=1) / np.sqrt(len(sm))) for k, sm in zip(keys, samples)}
        gh = games_howell_pairwise(samples, keys, POSTHOC_ALPHA)
        sig_pairs = [f"{display_names[a]}-{display_names[b]}" for pair, p in gh.items()
                     if p < POSTHOC_ALPHA for a, b in [pair.split('-', 1)]]

        rec = {"wavenumber_cm-1": int(cp["consensus_cm-1"]), "eta_squared": round(eta, 3),
               "F": round(F, 2), "p_ANOVA_at_band": p_anova, "in_significant_cluster": in_cluster,
               "passes_gate": passes,
               "intensity_measure": "fitted band height above curved local background (next to artifact cut)" if near_cut
               else f"mean intensity +-{PEAK_INTENSITY_HALF_WINDOW_CM1:g} cm-1"}
        for k in keys:
            rec[f"mean_{display_names[k]}"] = means[k]
            rec[f"sem_{display_names[k]}"] = sems[k]
        for k in keys:
            rec[f"observed_peak_{display_names[k]}"] = round(
                _refined_local_max(common_x, means_by_key[k], idx, tol_pts), 1)
        rec["dominant_condition"] = display_names[max(means, key=means.get)]
        rec["significant_pairs_GamesHowell"] = ", ".join(sig_pairs) if sig_pairs else "(none)"
        records.append(rec)
    return pd.DataFrame(records)


def describe_significant_regions(common_x, eta_sq, cluster_records, all_df):
    rows = []
    for run, mass, p in cluster_records:
        if p >= CLUSTER_ALPHA:
            continue
        lo_x, hi_x = float(common_x[run[0]]), float(common_x[run[-1]])
        inside = all_df[(all_df["wavenumber_cm-1"] >= lo_x - 1) & (all_df["wavenumber_cm-1"] <= hi_x + 1)]\
            if all_df is not None and len(all_df) else pd.DataFrame()
        best = run[int(np.argmax(eta_sq[run]))]
        rows.append({
            "start_cm-1": round(lo_x, 1), "end_cm-1": round(hi_x, 1), "cluster_p": p,
            "max_eta2_at_cm-1": round(float(common_x[best]), 1), "max_eta2": round(float(eta_sq[best]), 3),
            "consensus_bands_inside": ", ".join(str(int(w)) for w in inside["wavenumber_cm-1"]) if len(inside) else "",
            "bands_passing_eta2_gate": ", ".join(str(int(w)) for w in inside.loc[inside["passes_gate"], "wavenumber_cm-1"])
            if len(inside) else "",
            "label_on_plot": (", ".join(str(int(w)) for w in inside["wavenumber_cm-1"]) if len(inside)
                              else f"~{common_x[best]:.0f}"),
        })
    return pd.DataFrame(rows)
