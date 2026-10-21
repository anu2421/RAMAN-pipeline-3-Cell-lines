import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from .config import (
    COLORS,
    COMMON_GRID_POINTS,
    DATASETS,
    GROUP_DISPLAY,
    GROUP_KEYWORDS,
    LAYOUT,
    METHODS,
    NORM_LABELS,
    N_MARKERS,
    OUTPUT_DIR,
    PHE_SEARCH_MAX_CM1,
    PHE_SEARCH_MIN_CM1,
    ROI_MAX_CM1,
    ROI_MIN_CM1,
    RUN_TAG,
    SHIFT_LABELS,
)
from .optional import HAS_GAMES_HOWELL
from . import state
from .utils import (
    in_artifact,
    install_safe_savefig,
    read_accuracy,
    run_title,
    safe_name,
    trapz,
    wrapped_title,
)
from .loading import discover_set_groups, find_group_folders, load_group, rep_sort
from .preprocessing import normalize_spectrum
from .alignment import locate_peak, shifts_global, to_grid
from .bands import build_consensus_peaks
from .peak_shifts import run_peak_shift_for_comparison
from .plotting import (
    plot_cross_cellline_bars,
    plot_group_comparison_overlapped,
    plot_group_comparison_stacked,
    plot_overlap,
    plot_stacked,
)
from .anova import run_anova_for_comparison


def run_all(common_x_full):
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    raw = {}
    for ds, root in DATASETS.items():
        print(f"\n{'='*70}\nFOLDER DISCOVERY: {ds}  ({root})\n{'='*70}")
        if not os.path.isdir(root):
            print(f"[ERROR] {ds} folder not found: {root}")
            continue
        if LAYOUT == "sets":
            sources = [(g, None, reps) for g, reps in discover_set_groups(ds, root).items()]
        else:
            sources = [(g, gdir, None) for g, gdir in find_group_folders(ds, root).items()]
        for g, gdir, reps in sources:
            print(f"  Loading {ds} {GROUP_DISPLAY[g]}" + (f" from {gdir}" if gdir else ""))
            cells = load_group(ds, g, gdir, paths_by_rep=reps)
            if cells:
                raw[(ds, g)] = cells
    if not raw:
        print("[ERROR] No usable spectra found.")
        sys.exit(1)
    dss = [ds for ds in DATASETS if any(k[0] == ds for k in raw)]


    qc = []
    for (ds, g), reps in raw.items():
        for r, cells in reps.items():
            for c in cells:
                a = in_artifact(c["x"])
                tot = trapz(np.abs(c["y"]))
                qc.append({"cell_line": ds, "group": GROUP_DISPLAY[g], "replicate": r,
                           "file": os.path.basename(c["file"]),
                           "quartz_800_fitted_amplitude": c.get("quartz_amp", np.nan),
                           "excluded_region_area_fraction": float(trapz(np.abs(c["y"][a])) / tot) if tot else np.nan})
    qc_df = pd.DataFrame(qc)
    qc_df.to_csv(os.path.join(OUTPUT_DIR, "QC_artifact_region_strength_per_cell.csv"), index=False)
    print("\nQuartz/artifact-region share of total signal (before removal), mean per group:")
    print(qc_df.groupby(["cell_line", "group"])["excluded_region_area_fraction"].mean().round(3).to_string())

    keep = np.ones_like(common_x_full, dtype=bool)
    state.PLOT_XLIM = (float(common_x_full.min()), float(common_x_full.max()))
    common_x = common_x_full[keep]
    state.ANALYSIS_MASK = ~in_artifact(common_x)
    step = float(np.median(np.diff(common_x_full)))
    state.GRID_BREAKS = np.where(np.diff(common_x) > 1.5 * step)[0] + 1

    summary, cross_results = [], {}
    for norm, shift_method in METHODS:
        tag = f"{NORM_LABELS[norm]}_{SHIFT_LABELS[shift_method]}"
        mlabel = f"{NORM_LABELS[norm]} + {SHIFT_LABELS[shift_method]}"
        out = os.path.join(OUTPUT_DIR, tag)
        os.makedirs(out, exist_ok=True)
        print(f"\n{'#'*70}\nMETHOD: {mlabel}  ->  {out}\n{'#'*70}")

        data = {k: {r: [{"x": c["x"], "y": normalize_spectrum(c["x"], c["y"], norm)} for c in cells]
                    for r, cells in reps.items()} for k, reps in raw.items()}
        shifts = shifts_global(data, common_x_full)

        log_rows = []
        for k in data:
            for r in rep_sort(data[k]):
                for s in shifts[k][r]:
                    log_rows.append({"cell_line": k[0], "group": GROUP_DISPLAY[k[1]], "replicate": r,
                                     "applied_shift_cm-1": float(s)})
        pd.DataFrame(log_rows).to_csv(os.path.join(out, "applied_shift_log.csv"), index=False)

        rep_arrays = {k: {r: np.vstack([to_grid(c, common_x_full, shifts[k][r][i])
                                        for i, c in enumerate(reps[r])])[:, keep]
                          for r in rep_sort(reps)} for k, reps in data.items()}
        pooled = {k: np.vstack([rep_arrays[k][r] for r in rep_sort(rep_arrays[k])]) for k in rep_arrays}


        phe = []
        for k in rep_arrays:
            for r in rep_sort(rep_arrays[k]):
                pos = locate_peak(common_x, rep_arrays[k][r].mean(axis=0), PHE_SEARCH_MIN_CM1, PHE_SEARCH_MAX_CM1)
                raw_mean = np.mean([to_grid(c, common_x_full) for c in data[k][r]], axis=0)
                raw_pos = locate_peak(common_x_full, raw_mean, PHE_SEARCH_MIN_CM1 - 8, PHE_SEARCH_MAX_CM1 + 8)
                phe.append({"cell_line": k[0], "group": GROUP_DISPLAY[k[1]], "replicate": r,
                            "raw_axis_start_cm-1": round(float(np.median([c["x_start"] for c in raw[k][r]])), 2),
                            "Phe_peak_raw_cm-1": round(raw_pos, 2) if raw_pos else np.nan,
                            "applied_shift_cm-1": float(shifts[k][r][0]),
                            "Phe_peak_after_correction_cm-1": round(pos, 2) if pos else np.nan})
        phe_df = pd.DataFrame(phe)
        phe_df.to_csv(os.path.join(out, "QC_phenylalanine_position_per_replicate.csv"), index=False)
        print("  Axis check per cell line (raw start of recorded range, raw Phe position, shift applied):")
        for cl, g in phe_df.groupby("cell_line"):
            print(f"    {cl:<5} raw start {g['raw_axis_start_cm-1'].median():7.2f} | raw Phe "
                  f"{g['Phe_peak_raw_cm-1'].mean():7.2f} | applied shift {g['applied_shift_cm-1'].mean():+.2f} cm-1 "
                  f"(range {g['applied_shift_cm-1'].min():+.1f} to {g['applied_shift_cm-1'].max():+.1f})")
        print(f"  Phe band after correction: mean {phe_df['Phe_peak_after_correction_cm-1'].mean():.2f}, "
              f"range {phe_df['Phe_peak_after_correction_cm-1'].min():.2f}"
              f"-{phe_df['Phe_peak_after_correction_cm-1'].max():.2f} cm-1 across all replicates")


        group_means = {f"{GROUP_DISPLAY[k[1]]} {k[0]}": pooled[k].mean(axis=0) for k in pooled}
        consensus_df = build_consensus_peaks(common_x, group_means)
        consensus_df.drop(columns="idx").to_csv(os.path.join(out, "consensus_band_positions.csv"), index=False)
        majority = consensus_df[consensus_df["n_groups_detected"] >= max(1, len(group_means) // 2)]
        state.CONSENSUS_MARKERS = sorted(majority.nlargest(N_MARKERS, "grand_mean_height")["consensus_cm-1"].tolist())
        print(f"  {len(consensus_df)} consensus bands; plot markers: {state.CONSENSUS_MARKERS}")


        stk = os.path.join(out, "stacked_plots")
        os.makedirs(stk, exist_ok=True)
        print("\n-- stacked / overlap plots --")

        def trace(k, label=None):
            arr = pooled[k]
            return {"label": label or f"{GROUP_DISPLAY[k[1]]} {k[0]}", "color": COLORS[k],
                    "mean_y": arr.mean(axis=0), "n_cells": arr.shape[0],
                    "sem_y": arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else 0 * arr[0]}

        for ds in dss:
            for g in GROUP_KEYWORDS:
                k = (ds, g)
                if k not in rep_arrays:
                    continue
                traces = [{"label": r, "color": COLORS[k], "mean_y": rep_arrays[k][r].mean(axis=0),
                           "n_cells": rep_arrays[k][r].shape[0]} for r in rep_sort(rep_arrays[k])]
                plot_stacked(common_x, traces, f"{ds} {GROUP_DISPLAY[g]}: replicates",
                             os.path.join(stk, f"{ds}_{safe_name(GROUP_DISPLAY[g])}_replicates_stacked.png"),
                             shade_within=True)
            allg = [trace((ds, g)) for g in GROUP_KEYWORDS if (ds, g) in pooled]
            plot_stacked(common_x, allg, f"{run_title(ds)}: mean spectra",
                         os.path.join(stk, f"{ds}_Control_Ecoli_Saureus_stacked.png"))
            plot_overlap(common_x, allg, f"{run_title(ds)}: mean spectra ± SEM",
                         os.path.join(stk, f"{ds}_Control_Ecoli_Saureus_overlap.png"),
                         NORM_LABELS[norm])


        cross = os.path.join(out, "cross_cell_line")
        os.makedirs(cross, exist_ok=True)
        ctrl_traces = [dict(trace((ds, "control"), f"Control {ds}"), color=c)
                       for ds, c in zip(dss, ["#4D4D4D", "#8FB9A8", "#C9A26B"]) if (ds, "control") in pooled]
        plot_stacked(common_x, ctrl_traces, "Controls (shared axis)",
                     os.path.join(cross, "AllCellLines_Controls_stacked.png"))
        plot_stacked(common_x, [trace(k) for k in pooled],
                     "All groups (alignment check)",
                     os.path.join(cross, "AllCellLines_all_groups_stacked.png"))


        group_info = {}
        for ds in dss:
            for g in GROUP_KEYWORDS:
                k = (ds, g)
                present = k in pooled
                group_info[f"{ds}_{g}"] = {"display": f"{GROUP_DISPLAY[g]} {ds}", "color": COLORS[k],
                                           "replicate_data": rep_arrays[k] if present else {},
                                           "pooled": pooled[k] if present else None}

        for ds in dss:
            comp = {"suffix": "_".join(p for p in (ds, RUN_TAG, "Control_vs_Ecoli_vs_Saureus") if p),
                    "title": run_title(ds),
                    "groups": [f"{ds}_{g}" for g in GROUP_KEYWORDS]}
            cdir = os.path.join(out, comp["suffix"])
            os.makedirs(cdir, exist_ok=True)
            print(f"\n{'='*70}\n{comp['title']}\n{'='*70}")
            plot_group_comparison_stacked(common_x, group_info, comp, cdir)
            plot_group_comparison_overlapped(common_x, group_info, comp, cdir)
            all_df = run_anova_for_comparison(common_x, group_info, comp, cdir, consensus_df)
            cross_results[ds] = all_df
            run_peak_shift_for_comparison(common_x, group_info, comp, cdir)

            s = comp["suffix"]
            cv_acc, cv_f1 = read_accuracy(os.path.join(cdir, f"{s}_classification_report.csv"))
            lo_acc, lo_f1 = read_accuracy(os.path.join(cdir, f"{s}_classification_report_LORO.csv"))
            try:
                cl = pd.read_csv(os.path.join(cdir, f"{s}_significant_clusters.csv"))
                n_cl = int(cl["significant"].astype(str).str.lower().eq("true").sum())
            except Exception:
                n_cl = 0
            n_groups = len([g for g in comp["groups"] if group_info[g]["pooled"] is not None])
            gated = all_df[all_df["passes_gate"]] if all_df is not None else pd.DataFrame()
            summary.append({"cell_line": ds, "method": mlabel, "n_groups": n_groups,
                            "chance_accuracy": round(1 / n_groups, 3) if n_groups else np.nan,
                            "lda_accuracy_cell_level_cv": cv_acc, "lda_macro_f1_cell_level_cv": cv_f1,
                            "lda_accuracy_LORO": lo_acc, "lda_macro_f1_LORO": lo_f1,
                            "n_significant_clusters": n_cl,
                            "significant_bands_cm-1": ", ".join(str(w) for w in gated.get("wavenumber_cm-1", []))})


        rows = []
        for ds, df in cross_results.items():
            if df is None:
                continue
            for _, r in df.iterrows():
                rows.append({"wavenumber_cm-1": r["wavenumber_cm-1"], "cell_line": ds,
                             "eta_squared": r["eta_squared"], "significant": r["passes_gate"],
                             "dominant_condition": r["dominant_condition"],
                             "significant_pairs": r["significant_pairs_GamesHowell"]})
        if rows:
            long = pd.DataFrame(rows)
            wide = long.pivot_table(index="wavenumber_cm-1", columns="cell_line",
                                    values=["eta_squared", "significant"], aggfunc="first")
            wide.columns = [f"{a}_{b}" for a, b in wide.columns]
            wide.reset_index().merge(consensus_df.drop(columns="idx").rename(
                columns={"consensus_cm-1": "wavenumber_cm-1"}), on="wavenumber_cm-1", how="left"
            ).to_csv(os.path.join(cross, "cross_cell_line_band_table.csv"), index=False)
            long.to_csv(os.path.join(cross, "cross_cell_line_band_table_long.csv"), index=False)
        plot_cross_cellline_bars(cross_results, os.path.join(cross, "AllCellLines_peak_bar_comparison.png"))

    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(OUTPUT_DIR, "summary_all_cell_lines.csv"), index=False)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print("\n", sdf.to_string(index=False))
    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


def main():
    install_safe_savefig()
    common_x = np.linspace(ROI_MIN_CM1, ROI_MAX_CM1, COMMON_GRID_POINTS)
    if not HAS_GAMES_HOWELL:
        print("[NOTE] statsmodels not found -> Games-Howell falls back to Bonferroni-corrected Welch t-tests.")
    previous_title = plt.title
    plt.title = wrapped_title
    try:
        run_all(common_x)
    finally:
        plt.title = previous_title
