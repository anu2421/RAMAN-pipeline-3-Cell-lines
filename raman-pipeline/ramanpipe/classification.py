import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .optional import HAS_SKLEARN_DA, HAS_SKLEARN_METRICS, LeaveOneGroupOut, LinearDiscriminantAnalysis, PCA, StratifiedKFold, classification_report, confusion_matrix, cross_val_predict


def plot_pca_scores(X, group_idx, keys, key_to_idx, display_names, colors, title, path):
    if not HAS_SKLEARN_DA:
        print("  [NOTE] scikit-learn not found -> skipping PCA plot.")
        return None
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(7, 6))
    for k in keys:
        mask = group_idx == key_to_idx[k]
        plt.scatter(scores[mask, 0], scores[mask, 1], s=18, alpha=0.7, color=colors[k], label=display_names[k])
    plt.xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
    plt.ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(fontsize=9)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> PCA scores plot saved: {path}")
    return path


def plot_lda_scores(X, group_idx, keys, key_to_idx, display_names, colors, title, path):
    if not HAS_SKLEARN_DA:
        print("  [NOTE] scikit-learn not found -> skipping LDA plot.")
        return None
    n_comp = min(2, len(keys) - 1)
    if n_comp < 1:
        print("  [WARN] Need >=2 conditions for LDA, skipping.")
        return None
    scores = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(X, group_idx)
    n_comp = min(n_comp, scores.shape[1])
    plt.figure(figsize=(7, 6))
    rng = np.random.default_rng(0)
    for k in keys:
        mask = group_idx == key_to_idx[k]
        if n_comp == 1:
            plt.scatter(scores[mask, 0], rng.normal(0, 0.02, size=int(mask.sum())), s=18, alpha=0.7,
                        color=colors[k], label=display_names[k])
        else:
            plt.scatter(scores[mask, 0], scores[mask, 1], s=18, alpha=0.7, color=colors[k], label=display_names[k])
    plt.xlabel("LD1")
    plt.ylabel("LD2" if n_comp == 2 else "(vertical jitter for display only)")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(fontsize=9)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> LDA scores plot saved: {path}")
    return path


def plot_confusion_matrix(cm, display_names_list, title, path, accuracy, macro_f1):
    plt.figure(figsize=(6, 5.5))
    pastel = mcolors.LinearSegmentedColormap.from_list("pastel_blue", ["#FFFFFF", "#C9D9EC", "#6F9BCF", "#3F6FA8"])
    im = plt.imshow(cm, cmap=pastel)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="cells (count)")
    n = len(display_names_list)
    plt.xticks(range(n), display_names_list, rotation=30, ha="right")
    plt.yticks(range(n), display_names_list)
    plt.xlabel("Predicted condition")
    plt.ylabel("True condition")
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=11)
    plt.title(f"{title}\naccuracy {accuracy:.0%} · macro F1 {macro_f1:.2f}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> Confusion matrix plot saved: {path}")
    return path


def _report(group_idx, y_pred, keys, key_to_idx, display_names):
    label_order = [key_to_idx[k] for k in keys]
    display_list = [display_names[k] for k in keys]
    cm = confusion_matrix(group_idx, y_pred, labels=label_order)
    rd = classification_report(group_idx, y_pred, labels=label_order, target_names=display_list,
                               output_dict=True, zero_division=0)
    acc, f1 = rd["accuracy"], rd["macro avg"]["f1-score"]
    df = pd.DataFrame({k: v for k, v in rd.items() if k != "accuracy"}).transpose()
    df.loc["accuracy"] = {"precision": np.nan, "recall": np.nan, "f1-score": acc, "support": cm.sum()}
    return df, cm, acc, f1, display_list


def evaluate_classification(X, group_idx, keys, key_to_idx, display_names, title_prefix,
                            safe_prefix, output_dir, n_splits, seed):
    if not HAS_SKLEARN_METRICS or not HAS_SKLEARN_DA:
        print("  [NOTE] scikit-learn not fully available -> skipping classification report.")
        return None, None
    class_counts = {k: int((group_idx == key_to_idx[k]).sum()) for k in keys}
    n_splits_eff = max(2, min(n_splits, min(class_counts.values())))
    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X, group_idx, cv=cv)
    df, cm, acc, f1, display_list = _report(group_idx, y_pred, keys, key_to_idx, display_names)
    csv = os.path.join(output_dir, f"{safe_prefix}_classification_report.csv")
    df.to_csv(csv)
    print(f"  -> Classification report saved ({n_splits_eff}-fold stratified CV): {csv} "
          f"(accuracy={acc:.2f}, macro F1={f1:.2f})")
    plot_confusion_matrix(cm, display_list, f"{title_prefix}: confusion matrix (5-fold CV)",
                          os.path.join(output_dir, f"{safe_prefix}_confusion_matrix.png"), acc, f1)
    return df, cm


def evaluate_classification_loro(X, group_idx, replicate_groups, keys, key_to_idx, display_names,
                                 title_prefix, safe_prefix, output_dir):
    if not HAS_SKLEARN_METRICS or not HAS_SKLEARN_DA:
        print("  [NOTE] scikit-learn not fully available -> skipping LORO classification report.")
        return None, None
    if len(np.unique(replicate_groups)) < 2:
        print("  [WARN] Need >=2 replicate groups for leave-one-replicate-out CV, skipping.")
        return None, None
    reps_per_class = [len(np.unique(np.asarray(replicate_groups)[np.asarray(group_idx) == c])) for c in np.unique(group_idx)]
    if min(reps_per_class) < 2:
        print("  [NOTE] A group has only one replicate -> leave-one-replicate-out CV is not possible. "
              "Skipped; use the 5-fold CV result.")
        return None, None
    y_pred = cross_val_predict(LinearDiscriminantAnalysis(), X, group_idx, cv=LeaveOneGroupOut(),
                               groups=replicate_groups)
    df, cm, acc, f1, display_list = _report(group_idx, y_pred, keys, key_to_idx, display_names)
    csv = os.path.join(output_dir, f"{safe_prefix}_classification_report_LORO.csv")
    df.to_csv(csv)
    print(f"  -> LORO classification report saved: {csv} (accuracy={acc:.2f}, macro F1={f1:.2f})")
    plot_confusion_matrix(cm, display_list,
                          f"{title_prefix}: confusion matrix (leave-one-replicate-out)",
                          os.path.join(output_dir, f"{safe_prefix}_confusion_matrix_LORO.png"), acc, f1)
    return df, cm
