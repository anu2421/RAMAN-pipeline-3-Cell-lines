import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .config import (
    BAR_EDGE,
    COLORS,
    EXPOSURE,
    GROUP_DISPLAY,
    GROUP_KEYWORDS,
    SIG_SHADE,
)
from . import state
from .utils import bar_fill, find_peaks_segmented, gapped, rescale01, shade_color


def plot_blank_placeholder(title, path, reason):
    plt.figure(figsize=(9, 4))
    plt.axis("off")
    plt.text(0.5, 0.55, "No data yet", ha="center", va="center", fontsize=15, color="0.3")
    plt.text(0.5, 0.4, reason, ha="center", va="center", fontsize=10, color="0.5", wrap=True)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> Blank placeholder saved ({reason}): {path}")


def _detect_shared_peaks(display_traces, common_x, prominence=0.04, distance=30, max_peaks=6):
    if state.CONSENSUS_MARKERS:
        return np.array(state.CONSENSUS_MARKERS)
    grand = np.mean(np.vstack(display_traces), axis=0)
    peak_idx = find_peaks_segmented(grand, prominence=prominence, distance=distance)
    return common_x[peak_idx][:max_peaks]


def _label_levels(peaks, min_gap=45):
    out, last, lvl = [], None, 0
    for wn in sorted(peaks):
        lvl = (1 - lvl) if (last is not None and wn - last < min_gap) else 0
        out.append((wn, lvl)); last = wn
    return out


def _draw_peak_markers(peak_positions, top_y):
    for wn, lvl in _label_levels(peak_positions):
        plt.axvline(wn, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.text(wn, top_y * (1 + 0.05 * lvl) if top_y > 0 else top_y, f"{wn:.0f}", ha="center", va="bottom", fontsize=8)


def _style_stacked_axes(common_x):
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim(*state.PLOT_XLIM)
    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")


def plot_group_comparison_stacked(common_x, group_info, comp, output_dir):
    fname = os.path.join(output_dir, f"{comp['suffix']}_stacked.png")
    present_keys = [k for k in comp["groups"] if group_info[k]["pooled"] is not None]
    if not present_keys:
        plot_blank_placeholder(f"{comp['title']}: stacked mean spectra", fname,
                               reason="None of this comparison's groups have data")
        return
    means_disp = {k: rescale01(group_info[k]["pooled"].mean(axis=0)) for k in present_keys}
    offset_step = 1.05
    peaks = _detect_shared_peaks(list(means_disp.values()), common_x)
    n = len(present_keys)
    plt.figure(figsize=(9, 2.5 + 1.5 * n))
    for i, k in enumerate(present_keys):
        info = group_info[k]
        plt.plot(*gapped(common_x, means_disp[k] + i * offset_step), color=info["color"], linewidth=1.6,
                 label=f"{info['display']} (n={info['pooled'].shape[0]} cells)")
    _draw_peak_markers(peaks, top_y=(n - 1) * offset_step + 1.15)
    _style_stacked_axes(common_x)
    plt.yticks([])
    plt.ylabel("Normalized intensity (offset)")
    plt.title(comp["title"], fontsize=13, fontweight="bold")
    plt.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"  -> Comparison stacked plot saved: {fname}")

    table = pd.DataFrame({"RamanShift_cm-1": common_x})
    for k in present_keys:
        table[group_info[k]["display"]] = group_info[k]["pooled"].mean(axis=0)
    table.to_csv(os.path.join(output_dir, f"{comp['suffix']}_mean_spectra.csv"), index=False)


def plot_group_comparison_overlapped(common_x, group_info, comp, output_dir):
    fname = os.path.join(output_dir, f"{comp['suffix']}_overlapped.png")
    present_keys = [k for k in comp["groups"] if group_info[k]["pooled"] is not None]
    if not present_keys:
        plot_blank_placeholder(f"{comp['title']}: overlapped mean spectra", fname,
                               reason="None of this comparison's groups have data")
        return
    means_disp = {k: rescale01(group_info[k]["pooled"].mean(axis=0)) for k in present_keys}
    peaks = _detect_shared_peaks(list(means_disp.values()), common_x)
    plt.figure(figsize=(9, 5))
    for k in present_keys:
        info = group_info[k]
        plt.plot(*gapped(common_x, means_disp[k]), color=info["color"], linewidth=1.6, alpha=0.85,
                 label=f"{info['display']} (n={info['pooled'].shape[0]} cells)")
    _draw_peak_markers(peaks, top_y=1.08)
    _style_stacked_axes(common_x)
    plt.ylim(-0.05, 1.2)
    plt.title(f"{comp['title']} (overlay)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"  -> Comparison overlapped plot saved: {fname}")


def plot_permutation_null_distribution(null_max_mass, cluster_records, cluster_alpha, title, path):
    if len(cluster_records) == 0:
        print("  [INFO] No candidate clusters to show a null distribution for, skipping.")
        return None
    plt.figure(figsize=(9, 5))
    plt.hist(null_max_mass, bins=40, color="0.75", edgecolor="white")
    for run, mass, p in cluster_records:
        plt.axvline(mass, color="#C8645F" if p < cluster_alpha else "0.4", linestyle="--", linewidth=1.5, alpha=0.85)
    handles = [
        Line2D([0], [0], color="0.75", lw=8, label=f"Null distribution ({len(null_max_mass)} shuffles)"),
        Line2D([0], [0], color="#C8645F", linestyle="--", lw=1.5, label=f"Significant cluster (p<{cluster_alpha})"),
        Line2D([0], [0], color="0.4", linestyle="--", lw=1.5, label="Non-significant candidate"),
    ]
    plt.legend(handles=handles, fontsize=9)
    plt.xlabel("Cluster mass (summed F-statistic within a contiguous run)")
    plt.ylabel("Count (out of permutations)")
    plt.title(title, fontsize=13, fontweight="bold")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> Permutation null-distribution plot saved: {path}")
    return path


def plot_peak_bar_chart(peaks_df, keys, display_names, colors, title, safe_suffix, output_dir):
    if peaks_df.empty:
        print(f"  [INFO] No significant peaks to chart for {title}, skipping bar chart.")
        return None
    n_peaks = len(peaks_df)
    n_cond = len(keys)
    x = np.arange(n_peaks)
    width = 0.8 / n_cond
    plt.figure(figsize=(max(6, n_peaks * 1.4), 5))
    for i, k in enumerate(keys):
        lab = display_names[k]
        offset = (i - (n_cond - 1) / 2) * width
        plt.bar(x + offset, peaks_df[f"mean_{lab}"].to_numpy(), width=width,
                yerr=peaks_df[f"sem_{lab}"].to_numpy(), capsize=3, label=lab, color=bar_fill(colors[k]),
                edgecolor=BAR_EDGE, linewidth=0.8, error_kw=dict(ecolor="#2B2B2B", elinewidth=1, capthick=1))
    plt.xticks(x, [f"{wn:.0f}" for wn in peaks_df["wavenumber_cm-1"]])
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Mean normalized intensity ± SEM")
    plt.title(f"{title}: significant bands", fontsize=13, fontweight="bold")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{safe_suffix}_peak_bar_comparison.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> Peak bar chart saved: {path}")
    return path


def plot_anova_spectra(common_x, group_arrays, keys, display_names, colors, peaks_df,
                       sig_mask, cluster_alpha, min_eta_sq, title, path, regions_df=None):
    plt.figure(figsize=(10, 6))
    all_means = []
    for k in keys:
        mean_y = group_arrays[k].mean(axis=0)
        sem_y = group_arrays[k].std(axis=0, ddof=1) / np.sqrt(group_arrays[k].shape[0])
        all_means.append(mean_y)
        gx, gm = gapped(common_x, mean_y)
        _, gs = gapped(common_x, sem_y)
        plt.plot(gx, gm, label=f"{display_names[k]} (n={group_arrays[k].shape[0]})", color=colors[k], linewidth=1.5)
        plt.fill_between(gx, gm - gs, gm + gs, color=colors[k], alpha=0.2)
    all_means = np.vstack(all_means)

    sig_idx = np.where(sig_mask)[0]
    if len(sig_idx) > 0:
        split_here = (np.diff(sig_idx) > 1) | np.isin(sig_idx[1:], state.GRID_BREAKS)
        for run in np.split(sig_idx, np.where(split_here)[0] + 1):
            plt.axvspan(common_x[run[0]], common_x[run[-1]], color=SIG_SHADE, alpha=0.6, lw=0)

    data_max = float(all_means.max())
    data_min = float(min(0.0, all_means.min()))
    headroom = (data_max - data_min) * 0.25
    plt.ylim(data_min - 0.02 * (data_max - data_min), data_max + headroom)
    label_y = data_max + headroom * 0.3
    strong = set(int(w) for w in peaks_df["wavenumber_cm-1"])
    for wn in sorted(strong):
        plt.axvline(wn, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.text(wn, label_y, f"{wn:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)

    if regions_df is not None and len(regions_df):
        for _, r in regions_df.iterrows():
            if r["consensus_bands_inside"]:
                marks = [(float(w), str(w).strip()) for w in str(r["consensus_bands_inside"]).split(",")
                         if int(float(w)) not in strong]
            else:
                marks = [(float(r["max_eta2_at_cm-1"]), r["label_on_plot"])]
            for wn, lab in marks:
                plt.axvline(wn, color="0.45", linestyle=":", linewidth=0.9, alpha=0.8)
                plt.text(wn, label_y, lab, ha="center", va="bottom", fontsize=7, rotation=90,
                         color="0.35", style="italic")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim(*state.PLOT_XLIM)
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Normalized intensity (a.u.)")
    plt.title(title, fontsize=13, fontweight="bold", pad=22)
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    h += [Patch(color=SIG_SHADE, alpha=0.6, label=f"significant cluster (p<{cluster_alpha})"),
          Line2D([0], [0], color="black", ls="--", lw=0.8, label=f"band with $\\eta^2$ ≥ {min_eta_sq}")]
    plt.legend(handles=h, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> ANOVA spectra plot saved: {path}")


def _finish(title, path, ylabel):
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim(*state.PLOT_XLIM)
    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.ylabel(ylabel)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> saved: {path}")


def plot_stacked(common_x, traces, title, path, shade_within=False):
    if not traces:
        print(f"  [WARN] nothing to plot for '{title}', skipped.")
        return
    n = len(traces)
    disp = [rescale01(t["mean_y"]) for t in traces]
    offset = 1.05
    peaks = _detect_shared_peaks(disp, common_x)
    plt.figure(figsize=(9, 2.5 + 1.5 * n))
    for i, (t, d) in enumerate(zip(traces, disp)):
        color = shade_color(traces[0]["color"], (i / max(n - 1, 1)) * 0.55) if shade_within else t["color"]
        plt.plot(*gapped(common_x, d + i * offset), color=color, linewidth=1.5,
                 label=f"{t['label']} (n={t['n_cells']} cells)")
    for wn, lvl in _label_levels(peaks):
        plt.axvline(wn, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.text(wn, (n - 1) * offset + 1.15 + 0.12 * lvl, f"{wn:.0f}", ha="center", va="bottom", fontsize=8)
    plt.ylim(-0.15, (n - 1) * offset + 1.55)
    plt.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2 if n > 2 else n, frameon=False)
    _finish(title, path, "Intensity (a.u.)")


def plot_overlap(common_x, traces, title, path, norm_label):
    if not traces:
        return
    plt.figure(figsize=(9, 5))
    for t in traces:
        gx, gm = gapped(common_x, t["mean_y"])
        _, gs = gapped(common_x, t["sem_y"])
        plt.plot(gx, gm, color=t["color"], linewidth=1.5, alpha=0.9, label=f"{t['label']} (n={t['n_cells']} cells)")
        plt.fill_between(gx, gm - gs, gm + gs, color=t["color"], alpha=0.2)
    peaks = _detect_shared_peaks([rescale01(t["mean_y"]) for t in traces], common_x)
    top = max(float(t["mean_y"].max()) for t in traces)
    for wn, lvl in _label_levels(peaks):
        plt.axvline(wn, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.text(wn, top * (1.06 + 0.06 * lvl), f"{wn:.0f}", ha="center", va="bottom", fontsize=8)
    plt.ylim(top=top * 1.22)
    plt.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    _finish(title, path, f"Mean {norm_label}-normalized intensity (a.u.)")


def plot_cross_cellline_bars(results, path):
    union = sorted({int(w) for df in results.values() if df is not None
                    for w in df.loc[df["passes_gate"], "wavenumber_cm-1"]})
    if not union:
        print("  [INFO] no significant bands in any cell line -> cross-cell-line bar chart skipped")
        return
    dss = [d for d in results if results[d] is not None]
    fig, axes = plt.subplots(len(dss), 1, figsize=(max(9, 1.15 * len(union) + 2), 3.3 * len(dss)), squeeze=False)
    x = np.arange(len(union))
    width = 0.8 / len(GROUP_KEYWORDS)
    for ax, ds in zip(axes[:, 0], dss):
        sub = results[ds].set_index("wavenumber_cm-1").reindex(union)
        gate = sub["passes_gate"].fillna(False).astype(bool).to_numpy()
        top = 0
        for i, g in enumerate(GROUP_KEYWORDS):
            disp = f"{GROUP_DISPLAY[g]} {ds}"
            if f"mean_{disp}" not in sub:
                continue
            m = sub[f"mean_{disp}"].to_numpy(float)
            s = sub[f"sem_{disp}"].to_numpy(float)
            top = max(top, np.nanmax(m + s))
            off = (i - (len(GROUP_KEYWORDS) - 1) / 2) * width
            for sel, alpha, lab in [(gate, 0.95, disp), (~gate, 0.25, None)]:
                if sel.any():
                    ax.bar(x[sel] + off, m[sel], width=width, yerr=s[sel], capsize=2,
                           color=bar_fill(COLORS[(ds, g)]), edgecolor=BAR_EDGE, linewidth=0.6,
                           alpha=alpha, label=lab)
        for j in np.where(gate)[0]:
            ax.text(x[j], top * 1.04, "*", ha="center", fontsize=13)
        ax.set_ylim(0, top * 1.18 if top > 0 else 1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in union])
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean norm. intensity $\\pm$ SEM", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    axes[-1, 0].set_xlabel("Raman Shift (cm$^{-1}$), consensus band position (identical for all cell lines)")
    fig.suptitle(f"{EXPOSURE.capitalize()}: band intensities" if EXPOSURE else "Band intensities", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> saved: {path}")
