import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from .optional import HAS_GAMES_HOWELL, psturng
from . import state


def anova_F_vectorized(X, group_idx, n_groups):
    N, P = X.shape
    grand_mean = X.mean(axis=0)
    ss_between = np.zeros(P)
    ss_within = np.zeros(P)
    for g in range(n_groups):
        mask = group_idx == g
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        Xg = X[mask]
        group_mean = Xg.mean(axis=0)
        ss_between += n_g * (group_mean - grand_mean) ** 2
        ss_within += ((Xg - group_mean) ** 2).sum(axis=0)
    dfn = n_groups - 1
    dfd = N - n_groups
    ms_between = ss_between / dfn
    ms_within = ss_within / dfd
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(ms_within > 0, ms_between / ms_within, 0.0)
        ss_total = ss_between + ss_within
        eta_sq = np.where(ss_total > 0, ss_between / ss_total, 0.0)
    return F, eta_sq, dfn, dfd


def find_clusters_above_threshold(values, threshold):
    idx = np.where(values > threshold)[0]
    if len(idx) == 0:
        return []

    split_here = (np.diff(idx) > 1) | np.isin(idx[1:], state.GRID_BREAKS)
    runs = np.split(idx, np.where(split_here)[0] + 1)
    return [(run, float(values[run].sum())) for run in runs]


def cluster_permutation_test(X, group_idx, n_groups, n_perms, cluster_alpha, form_alpha, seed):
    F_obs, eta_sq_obs, dfn, dfd = anova_F_vectorized(X, group_idx, n_groups)
    m = state.ANALYSIS_MASK if state.ANALYSIS_MASK is not None and len(state.ANALYSIS_MASK) == X.shape[1] else np.ones(X.shape[1], bool)
    F_obs = np.where(m, F_obs, 0.0)
    eta_sq_obs = np.where(m, eta_sq_obs, np.nan)
    F_thresh = f_dist.ppf(1 - form_alpha, dfn, dfd)
    obs_clusters = find_clusters_above_threshold(F_obs, F_thresh)
    rng = np.random.default_rng(seed)
    null_max_mass = np.zeros(n_perms)
    for i in range(n_perms):
        F_perm, _, _, _ = anova_F_vectorized(X, rng.permutation(group_idx), n_groups)
        F_perm = np.where(m, F_perm, 0.0)
        null_max_mass[i] = max((m for _, m in find_clusters_above_threshold(F_perm, F_thresh)), default=0.0)
    sig_mask = np.zeros(X.shape[1], dtype=bool)
    cluster_records = []
    for run, mass in obs_clusters:
        p_val = (1 + np.sum(null_max_mass >= mass)) / (n_perms + 1)
        if p_val < cluster_alpha:
            sig_mask[run] = True
        cluster_records.append((run, mass, p_val))
    return F_obs, eta_sq_obs, sig_mask, cluster_records, F_thresh, null_max_mass


def games_howell_pairwise(samples, labels, alpha):
    k = len(samples)
    n_pairs = k * (k - 1) // 2
    pvals = {}
    for i in range(k):
        for j in range(i + 1, k):
            ni, nj = len(samples[i]), len(samples[j])
            mi, mj = np.mean(samples[i]), np.mean(samples[j])
            vi, vj = np.var(samples[i], ddof=1), np.var(samples[j], ddof=1)
            se_sq = vi / ni + vj / nj
            if se_sq <= 0:
                continue
            t_stat = abs(mi - mj) / np.sqrt(se_sq)
            df = se_sq ** 2 / ((vi / ni) ** 2 / (ni - 1) + (vj / nj) ** 2 / (nj - 1))
            if HAS_GAMES_HOWELL:
                p = psturng(t_stat * np.sqrt(2), k, df)
                p = float(p[0]) if isinstance(p, np.ndarray) else float(p)
            else:
                p = min(2 * t_dist.sf(t_stat, df) * n_pairs, 1.0)
            pvals[f"{labels[i]}-{labels[j]}"] = p
    return pvals
