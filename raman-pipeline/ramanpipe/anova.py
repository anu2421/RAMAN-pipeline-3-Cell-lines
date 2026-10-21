import os
import numpy as np
import pandas as pd

from .config import (
    CLASSIFICATION_CV_FOLDS,
    CLUSTER_ALPHA,
    CLUSTER_FORMING_ALPHA,
    MIN_ETA_SQUARED,
    N_PERMUTATIONS,
    RANDOM_SEED,
)
from . import state
from .stats import cluster_permutation_test
from .bands import analyze_consensus_peaks, describe_significant_regions
from .classification import (
    evaluate_classification,
    evaluate_classification_loro,
    plot_lda_scores,
    plot_pca_scores,
)
from .plotting import (
    plot_anova_spectra,
    plot_blank_placeholder,
    plot_peak_bar_chart,
    plot_permutation_null_distribution,
)


def run_anova_for_comparison(common_x, group_info, comp, output_dir, consensus_df):
    present_keys = [k for k in comp["groups"] if group_info[k]["pooled"] is not None]
    suffix = comp["suffix"]
    anova_plot_path = os.path.join(output_dir, f"{suffix}_anova_spectra.png")
    if len(present_keys) < 2:
        plot_blank_placeholder(f"{comp['title']}: ANOVA", anova_plot_path,
                               reason=f"Need >=2 groups with data (have {len(present_keys)})")
        return None

    display_names = {k: group_info[k]["display"] for k in present_keys}
    colors = {k: group_info[k]["color"] for k in present_keys}
    group_arrays = {k: group_info[k]["pooled"] for k in present_keys}
    key_to_idx = {k: i for i, k in enumerate(present_keys)}
    X = np.vstack([group_arrays[k] for k in present_keys])
    group_idx = np.concatenate([np.full(group_arrays[k].shape[0], key_to_idx[k]) for k in present_keys])

    print(f"  Running cluster-permutation ANOVA ({N_PERMUTATIONS} permutations)...")
    F_obs, eta_sq, sig_mask, cluster_records, F_thresh, null_max_mass = cluster_permutation_test(
        X, group_idx, len(present_keys), N_PERMUTATIONS, CLUSTER_ALPHA, CLUSTER_FORMING_ALPHA, RANDOM_SEED)
    n_sig = sum(1 for _, _, p in cluster_records if p < CLUSTER_ALPHA)
    print(f"  -> {len(cluster_records)} candidate cluster(s), {n_sig} significant; "
          f"{int(sig_mask.sum())}/{len(sig_mask)} points covered.")

    plot_permutation_null_distribution(null_max_mass, cluster_records, CLUSTER_ALPHA,
                                       f"{comp['title']}: permutation test",
                                       os.path.join(output_dir, f"{suffix}_permutation_null_distribution.png"))

    pd.DataFrame({"RamanShift_cm-1": common_x, "F_stat": F_obs, "eta_squared": eta_sq,
                  "in_significant_cluster": sig_mask, "passes_effect_size_gate": eta_sq >= MIN_ETA_SQUARED}
                 ).to_csv(os.path.join(output_dir, f"{suffix}_anova_by_wavenumber.csv"), index=False)
    pd.DataFrame([{"start_cm-1": round(float(common_x[r[0]]), 1), "end_cm-1": round(float(common_x[r[-1]]), 1),
                   "cluster_mass": round(m, 2), "p_value": p, "significant": p < CLUSTER_ALPHA}
                  for r, m, p in cluster_records]
                 ).to_csv(os.path.join(output_dir, f"{suffix}_significant_clusters.csv"), index=False)

    all_df = analyze_consensus_peaks(common_x, group_arrays, present_keys, display_names, sig_mask, consensus_df)
    all_df.to_csv(os.path.join(output_dir, f"{suffix}_all_consensus_bands.csv"), index=False)
    peaks_df = all_df[all_df["passes_gate"]].reset_index(drop=True)
    peaks_csv = os.path.join(output_dir, f"{suffix}_significant_peaks.csv")
    peaks_df.to_csv(peaks_csv, index=False)
    print(f"  -> {len(peaks_df)} significant, effect-size-gated band(s) at consensus positions: "
          f"{', '.join(str(w) for w in peaks_df['wavenumber_cm-1'])}")

    plot_peak_bar_chart(peaks_df, present_keys, display_names, colors, comp["title"], suffix, output_dir)
    regions_df = describe_significant_regions(common_x, eta_sq, cluster_records, all_df)
    regions_df.to_csv(os.path.join(output_dir, f"{suffix}_significant_regions_labelled.csv"), index=False)
    plot_anova_spectra(common_x, group_arrays, present_keys, display_names, colors, peaks_df,
                       sig_mask, CLUSTER_ALPHA, MIN_ETA_SQUARED, f"{comp['title']}: ANOVA",
                       anova_plot_path, regions_df)
    X = X[:, state.ANALYSIS_MASK] if state.ANALYSIS_MASK is not None else X
    plot_pca_scores(X, group_idx, present_keys, key_to_idx, display_names, colors,
                    f"{comp['title']}: PCA",
                    os.path.join(output_dir, f"{suffix}_pca_scores.png"))
    plot_lda_scores(X, group_idx, present_keys, key_to_idx, display_names, colors,
                    f"{comp['title']}: LDA",
                    os.path.join(output_dir, f"{suffix}_lda_scores.png"))
    evaluate_classification(X, group_idx, present_keys, key_to_idx, display_names, comp["title"],
                            suffix, output_dir, CLASSIFICATION_CV_FOLDS, RANDOM_SEED)
    replicate_groups = np.concatenate([
        np.array([f"{k}::{r}" for r, mat in group_info[k]["replicate_data"].items() for _ in range(mat.shape[0])])
        for k in present_keys])
    print(f"  Running leave-one-replicate-out CV ({len(np.unique(replicate_groups))} folds)...")
    evaluate_classification_loro(X, group_idx, replicate_groups, present_keys, key_to_idx, display_names,
                                 comp["title"], suffix, output_dir)
    return all_df
