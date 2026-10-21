import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .config import (
    PEAK_DISTANCE,
    PEAK_PROMINENCE,
    PEAK_SHIFT_BOOTSTRAP_N,
    PEAK_SHIFT_BOOTSTRAP_SEED,
    PEAK_SHIFT_MATCH_TOLERANCE_CM1,
    PEAK_SHIFT_REFINE_WINDOW,
)
from .utils import find_peaks_segmented, in_artifact, rescale01, segment_of
from .plotting import plot_blank_placeholder


def _find_peaks_in_mean(y_mean, prominence, distance):
    return find_peaks_segmented(rescale01(y_mean), prominence=prominence, distance=distance)


def _refine_peak_position(y, approx_idx, window):
    s, e = segment_of(approx_idx, len(y))
    lo, hi = max(s, approx_idx - window), min(e, approx_idx + window + 1)
    return lo + int(np.argmax(y[lo:hi]))


def _bootstrap_peak_positions(cell_matrix, common_x, approx_idx, window, n_boot, seed):
    rng = np.random.default_rng(seed)
    n_cells = cell_matrix.shape[0]
    positions = np.empty(n_boot)
    for b in range(n_boot):
        boot_mean = cell_matrix[rng.integers(0, n_cells, size=n_cells)].mean(axis=0)
        positions[b] = common_x[_refine_peak_position(boot_mean, approx_idx, window)]
    return positions


def _match_peaks_across_groups(peaks_per_group, common_x, tolerance_cm1):
    labels = list(peaks_per_group.keys())
    anchor = max(labels, key=lambda l: len(peaks_per_group[l]))
    remaining = {l: list(peaks_per_group[l]) for l in labels}
    groups = []
    for anchor_idx in list(remaining[anchor]):
        group = {anchor: anchor_idx}
        for l in labels:
            if l == anchor or not remaining[l]:
                continue
            diffs = np.abs(common_x[np.array(remaining[l])] - common_x[anchor_idx])
            best = int(np.argmin(diffs))
            if diffs[best] <= tolerance_cm1:
                group[l] = remaining[l][best]
        groups.append(group)
    remaining[anchor] = []
    return groups, {l: idxs for l, idxs in remaining.items() if idxs}


def detect_group_peak_shifts(common_x, group_arrays, display_names, prominence, distance,
                             match_tol_cm1, refine_window, n_boot, seed):
    peaks_per_group = {l: _find_peaks_in_mean(group_arrays[l].mean(axis=0), prominence, distance)
                       for l in group_arrays}
    groups, leftover = _match_peaks_across_groups(peaks_per_group, common_x, match_tol_cm1)
    records = []
    for gi, group in enumerate(groups):
        if len(group) < 2:
            continue
        boot = {l: _bootstrap_peak_positions(group_arrays[l], common_x, idx, refine_window, n_boot, seed + gi)
                for l, idx in group.items()}
        ref = next((l for l in group if l.endswith("control")), sorted(group)[0])
        for l in sorted(group):
            row = {"peak_group": gi, "condition": display_names[l], "reference_condition": display_names[ref],
                   "position_mean_cm-1": round(float(boot[l].mean()), 2),
                   "position_sd_cm-1": round(float(boot[l].std(ddof=1)), 2)}
            if l != ref:
                diff = boot[l] - boot[ref]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                row.update({"shift_vs_reference_cm-1": round(float(diff.mean()), 2),
                            "shift_ci_low": round(float(lo), 2), "shift_ci_high": round(float(hi), 2),
                            "significant_shift": bool(lo > 0 or hi < 0)})
            records.append(row)
    cols = ["peak_group", "condition", "reference_condition", "position_mean_cm-1", "position_sd_cm-1",
            "shift_vs_reference_cm-1", "shift_ci_low", "shift_ci_high", "significant_shift"]
    shift_df = pd.DataFrame(records, columns=cols)
    leftover_df = pd.DataFrame([{"condition": display_names[l], "position_cm-1": round(float(common_x[i]), 1),
                                 "note": "peak found only in this condition"}
                                for l, idxs in leftover.items() for i in idxs])
    return shift_df, leftover_df


def plot_group_shift_chart(shift_df, title, path, color_map):
    if shift_df.empty:
        plot_blank_placeholder(title, path, reason="No matched peaks across groups to chart")
        return
    order = shift_df.groupby("peak_group")["position_mean_cm-1"].mean().sort_values().index.tolist()
    plt.figure(figsize=(9, max(3, 0.6 * len(order) + 1.5)))
    y_labels = []
    for row_i, gi in enumerate(order):
        g = shift_df[shift_df["peak_group"] == gi].sort_values("position_mean_cm-1")
        xs = g["position_mean_cm-1"].to_numpy()
        plt.plot([xs.min(), xs.max()], [row_i, row_i], color="0.75", linewidth=1, zorder=1)
        for x_val, err, cond, sig in zip(xs, g["position_sd_cm-1"], g["condition"], g["significant_shift"]):
            is_sig = (sig == True)
            plt.errorbar(x_val, row_i, xerr=err, fmt="o", color=color_map.get(cond, "gray"),
                         markeredgecolor="red" if is_sig else "none", markeredgewidth=2 if is_sig else 0,
                         markersize=9, capsize=3, zorder=3)
        y_labels.append(f"~{xs.mean():.0f} cm$^{{-1}}$")
    plt.yticks(range(len(order)), y_labels)
    plt.xlabel("Raman shift, peak position (cm$^{-1}$)")
    plt.ylabel("Matched peak")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.figtext(0.99, 0.01, "red outline: 95% CI of shift excludes 0", ha="right", fontsize=7, color="0.4")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=n)
               for n, c in color_map.items() if n in shift_df["condition"].unique()]
    if handles:
        plt.legend(handles=handles, loc="best", fontsize=9)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> Peak position shift chart saved: {path}")


def run_peak_shift_for_comparison(common_x, group_info, comp, output_dir):
    present_keys = [k for k in comp["groups"] if group_info[k]["pooled"] is not None]
    chart_path = os.path.join(output_dir, f"{comp['suffix']}_peak_position_shift_chart.png")
    if len(present_keys) < 2:
        plot_blank_placeholder(f"{comp['title']}: peak position shifts", chart_path,
                               reason="Need >=2 groups with data")
        return
    group_arrays = {k: group_info[k]["pooled"] for k in present_keys}
    display_names = {k: group_info[k]["display"] for k in present_keys}
    color_map = {group_info[k]["display"]: group_info[k]["color"] for k in present_keys}
    shift_df, leftover_df = detect_group_peak_shifts(
        common_x, group_arrays, display_names, PEAK_PROMINENCE, PEAK_DISTANCE,
        PEAK_SHIFT_MATCH_TOLERANCE_CM1, PEAK_SHIFT_REFINE_WINDOW, PEAK_SHIFT_BOOTSTRAP_N, PEAK_SHIFT_BOOTSTRAP_SEED)
    if len(shift_df):
        gpos = shift_df.groupby("peak_group")["position_mean_cm-1"].transform("mean").to_numpy(dtype=float)
        shift_df = shift_df[~in_artifact(gpos)].reset_index(drop=True)
    if len(leftover_df):
        leftover_df = leftover_df[~in_artifact(leftover_df["position_cm-1"].to_numpy(dtype=float))].reset_index(drop=True)
    shift_df.to_csv(os.path.join(output_dir, f"{comp['suffix']}_peak_position_shifts.csv"), index=False)
    n_sig = int(shift_df["significant_shift"].fillna(False).astype(bool).sum()) if len(shift_df) else 0
    print(f"  -> Peak position shift table saved ({n_sig} significant condition-pair shift(s))")
    plot_group_shift_chart(shift_df, f"{comp['title']}: peak positions", chart_path, color_map)
    if not leftover_df.empty:
        leftover_df.to_csv(os.path.join(output_dir, f"{comp['suffix']}_condition_specific_peaks.csv"), index=False)
