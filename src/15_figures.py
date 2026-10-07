"""All figures for the write-up and slides, drawn from saved results (no retraining).

Inputs : reports/experiments/* (from 14_experiments.py), reports/svm_cv.csv, data/interim/clean.parquet
Outputs: reports/figures/*.png
"""
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from config import BIN_LABELS, INTERIM, ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
EXP = ROOT / "reports" / "experiments"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = ["Ordinal LightGBM", "LightGBM classifier", "SVM (linear)", "Logistic Regression", "Naive Bayes"]
# Validated categorical palette (fixed order, colorblind-checked); dash styles as secondary encoding
COLOR = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
DASH = dict(zip(MODELS, ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]))
PROBABILISTIC = ["LightGBM classifier", "Logistic Regression", "Naive Bayes"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STANFORD_F1, YEAR_ONLY_F1 = 0.7026, 0.6998

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 200,
    "savefig.bbox": "tight", "lines.linewidth": 2,
})


def key(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def load():
    out = np.load(EXP / "test_outputs.npz")
    return out, pd.read_csv(EXP / "metrics.csv").set_index("model")


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"  {name}")


def fig_dataset():
    d = pd.read_parquet(INTERIM / "clean.parquet", columns=["year", "label"])
    share = pd.crosstab(d.year, d.label, normalize="index")
    blues = ["#cfe0f6", "#6fa5e8", "#1f5fae"]  # ordered bins -> one hue, light to dark
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bottom = np.zeros(len(share))
    for k in (0, 1, 2):
        ax.bar(share.index, share[k], bottom=bottom, color=blues[k], width=0.72, edgecolor="white",
               linewidth=2, label=BIN_LABELS[k])
        for x, b, v in zip(share.index, bottom, share[k]):
            if v > 0.06:
                ax.text(x, b + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if k == 2 else INK)
        bottom += share[k].to_numpy()
    ax.set_ylabel("Share of that year's papers")
    ax.set_title("Citation bin by publication year (top 20,000 cited papers per year)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)
    ax.legend(title="Citation bin", loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    save(fig, "01_bins_by_year.png")


def fig_confusion(out):
    y = out["y_test"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    for ax, name in zip(axes, MODELS):
        cm = confusion_matrix(y, out[f"{key(name)}__pred"], labels=[0, 1, 2])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{share[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center", fontsize=7.5,
                        color="white" if share[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), BIN_LABELS)
        ax.set_yticks(range(3), BIN_LABELS)
        ax.grid(False)
        ax.set_xlabel("Predicted bin")
        ax.set_title(name)
    axes[0].set_ylabel("Actual bin")
    fig.suptitle("Confusion matrices on the test set (share of each actual bin; counts in brackets)", y=1.03)
    save(fig, "02_confusion_matrices.png")


def class_curves(out, kind):
    """ROC or precision-recall: one panel per bin (one-vs-rest) plus a macro-average panel."""
    y = out["y_test"]
    yb = label_binarize(y, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0), sharey=True)
    grid = np.linspace(0, 1, 501)
    rows = []
    for name in MODELS:
        S = out[f"{key(name)}__scores"]
        curves, vals = [], []
        for k in range(3):
            if kind == "roc":
                x, yv, _ = roc_curve(yb[:, k], S[:, k])
                v = roc_auc_score(yb[:, k], S[:, k])
            else:
                yv, x, _ = precision_recall_curve(yb[:, k], S[:, k])
                v = average_precision_score(yb[:, k], S[:, k])
            axes[k].plot(x, yv, color=COLOR[name], ls=DASH[name], lw=1.8, label=f"{name} ({v:.3f})")
            order = np.argsort(x)
            curves.append(np.interp(grid, np.asarray(x)[order], np.asarray(yv)[order]))
            vals.append(v)
            rows.append({"model": name, "class": BIN_LABELS[k], "value": round(v, 4)})
        macro = np.mean(curves, axis=0)
        if kind == "roc":
            macro[0] = 0.0
        axes[3].plot(grid, macro, color=COLOR[name], ls=DASH[name], lw=1.8,
                     label=f"{name} ({np.mean(vals):.3f})")
        rows.append({"model": name, "class": "macro", "value": round(float(np.mean(vals)), 4)})
    metric = "AUC" if kind == "roc" else "average precision"
    for ax, title in zip(axes, [f"{b} vs rest" for b in BIN_LABELS] + ["Macro-average"]):
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
            ax.set_xlabel("False positive rate (FPR)")
        else:
            ax.set_xlabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.legend(loc="lower right" if kind == "roc" else "lower left", frameon=False, title=metric,
                  title_fontsize=7.5)
    axes[0].set_ylabel("True positive rate (TPR)" if kind == "roc" else "Precision")
    name = "ROC curves (TPR vs FPR)" if kind == "roc" else "Precision-recall curves"
    fig.suptitle(f"{name} on the test set, one-vs-rest per citation bin", y=1.02)
    save(fig, "03_roc_curves.png" if kind == "roc" else "04_precision_recall_curves.png")
    pd.DataFrame(rows).to_csv(EXP / ("roc_auc.csv" if kind == "roc" else "average_precision.csv"), index=False)


def fig_calibration(out, metrics):
    y = out["y_test"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.2, 1]})
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":", label="Perfectly calibrated")
    for name in PROBABILISTIC:
        P = out[f"{key(name)}__proba"]
        conf, correct = P.max(1), (P.argmax(1) == y).astype(int)
        frac, mean_conf = calibration_curve(correct, conf, n_bins=10, strategy="quantile")
        brier = np.mean(np.sum((P - label_binarize(y, classes=[0, 1, 2])) ** 2, axis=1))
        ax.plot(mean_conf, frac, color=COLOR[name], ls=DASH[name], marker="o", ms=4,
                label=f"{name} (log loss {metrics.loc[name, 'test_log_loss']:.3f}, Brier {brier:.3f})")
        ax2.hist(conf, bins=30, range=(1 / 3, 1), histtype="step", lw=1.8, color=COLOR[name], ls=DASH[name],
                 label=name)
    ax.set_xlabel("Predicted probability of the chosen bin")
    ax.set_ylabel("Actual share correct")
    ax.set_title("Calibration (reliability) curve")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax2.set_xlabel("Predicted probability of the chosen bin")
    ax2.set_ylabel("Number of test papers")
    ax2.set_title("How confident each model is")
    ax2.legend(loc="upper right", frameon=False)
    fig.suptitle("Probability quality of the probabilistic models (test set)", y=1.02)
    save(fig, "05_calibration.png")


def fig_per_class(out):
    y = out["y_test"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    width = 0.16
    x = np.arange(3)
    for i, name in enumerate(MODELS):
        p, r, f, _ = precision_recall_fscore_support(y, out[f"{key(name)}__pred"], labels=[0, 1, 2],
                                                     zero_division=0)
        for ax, vals in zip(axes, (p, r, f)):
            ax.bar(x + (i - 2) * width, vals, width * 0.88, color=COLOR[name], label=name)
    for ax, title in zip(axes, ("Precision", "Recall (= TPR)", "F1 score")):
        ax.set_xticks(x, BIN_LABELS)
        ax.set_xlabel("Citation bin")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=5, frameon=False)
    fig.suptitle("Per-bin precision, recall and F1 on the test set", y=1.02)
    save(fig, "06_per_class_metrics.png")


def fig_model_comparison(metrics):
    m = metrics.loc[MODELS]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(MODELS))
    w = 0.26
    shades = {"train_macro_f1": ("Train", "#b7cff0"), "cv_macro_f1": ("5-fold CV", "#6fa5e8"),
              "test_macro_f1": ("Test", "#1f5fae")}
    for i, (col, (label, color)) in enumerate(shades.items()):
        ax.bar(x + (i - 1) * w, m[col], w * 0.9, color=color, label=label)
        for xi, v in zip(x, m[col]):
            ax.text(xi + (i - 1) * w, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.8, color=INK)
    ax.axhline(STANFORD_F1, color="#e34948", ls="--", lw=1.2)
    ax.text(len(MODELS) - 0.45, STANFORD_F1 + 0.004, "Stanford NN 0.7026", color=INK, fontsize=7.5, ha="right")
    ax.axhline(YEAR_ONLY_F1, color=MUTED, ls=":", lw=1.2)
    ax.text(len(MODELS) - 0.45, YEAR_ONLY_F1 - 0.012, "Year-only rule 0.6998", color=MUTED, fontsize=7.5,
            ha="right")
    ax.set_xticks(x, MODELS)
    ax.set_ylim(0.55, 0.95)
    ax.set_ylabel("Macro F1")
    ax.set_title("Train vs cross-validation vs test macro F1 (a large train-test gap = overfitting)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=3, frameon=False)
    save(fig, "07_model_comparison.png")


def fig_timing(metrics):
    m = metrics.loc[MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, col, title in ((axes[0], "fit_seconds", "Training time on 118,971 papers"),
                           (axes[1], "predict_seconds", "Prediction time on 20,997 test papers")):
        order = m[col].sort_values()
        ax.barh(order.index, order.values, color=[COLOR[n] for n in order.index], height=0.6)
        for i, v in enumerate(order.values):
            ax.text(v * 1.15, i, f"{v:,.2f} s" if v < 10 else f"{v:,.0f} s", va="center", fontsize=7.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel("Seconds (log scale)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(right=order.max() * 6)
    fig.suptitle("Computation time per model (Apple M4 Pro; LightGBM 6 threads, SVM single-threaded)", y=1.04)
    save(fig, "08_training_time.png")


def fig_scaling():
    s = pd.read_csv(EXP / "scaling.csv")
    names = MODELS + ["SVM (exact RBF kernel)"]
    colors = {**COLOR, "SVM (exact RBF kernel)": "#4a3aa7"}
    dashes = {**DASH, "SVM (exact RBF kernel)": "-"}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name in names:
        d = s[s.model == name]
        if d.empty:
            continue
        ax.plot(d.n_train, d.fit_seconds, color=colors[name], ls=dashes[name], marker="o", ms=4.5, label=name)
        ax.annotate(name, (d.n_train.iloc[-1], d.fit_seconds.iloc[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color=INK, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training papers (log scale)")
    ax.set_ylabel("Training time, seconds (log scale)")
    ax.set_title("How training time grows with data size (exact RBF SVM stopped at 30,000 papers)")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "09_time_scaling.png")


def fig_learning():
    lc = pd.read_csv(EXP / "learning_curves.csv")
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    for ax, name in zip(axes, MODELS):
        d = lc[lc.model == name]
        ax.plot(d.n_train, d.train_macro_f1, color=COLOR[name], ls="--", marker="o", ms=4, label="Train")
        ax.plot(d.n_train, d.val_macro_f1, color=COLOR[name], ls="-", marker="s", ms=4, label="Validation")
        ax.fill_between(d.n_train, d.val_macro_f1, d.train_macro_f1, color=COLOR[name], alpha=0.12)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("Training papers (log)")
        ax.legend(loc="lower right", frameon=False)
    axes[0].set_ylabel("Macro F1")
    fig.suptitle("Learning curves: the shaded gap between train and validation is overfitting", y=1.03)
    save(fig, "10_learning_curves.png")


def feature_group(f):
    if "qwen" in f:
        return "Title embedding"
    if "specter2_abstract" in f:
        return "Abstract embedding"
    if f.startswith("journal") and "count" not in f:
        return "Journal (target-encoded)"
    if f.startswith("year"):
        return "Year"
    if f.startswith("title"):
        return "Title shape"
    if any(t in f for t in ("country", "author", "intl", "inst_share", "n_institutions", "n_countries")):
        return "Authors & countries"
    if any(t in f for t in ("abstract", "has_abstract")):
        return "Abstract stats"
    if any(t in f for t in ("subfield", "topic", "field", "domain")):
        return "Field / topic"
    return "Other paper metadata"


def fig_importance():
    imp = pd.read_csv(EXP / "lgbm_feature_importance.csv")
    imp["share"] = imp.gain / imp.gain.sum()
    imp["group"] = imp.feature.map(feature_group)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    top = imp.head(15).iloc[::-1]
    ax.barh(top.feature, top.share, color="#2a78d6", height=0.65)
    ax.set_xlabel("Share of total split gain")
    ax.set_title("Top 15 features (LightGBM classifier)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    g = imp.groupby("group").share.sum().sort_values()
    ax2.barh(g.index, g.values, color="#1f5fae", height=0.6)
    for i, v in enumerate(g.values):
        ax2.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=7.5, color=INK)
    ax2.set_xlabel("Share of total split gain")
    ax2.set_title("Importance by feature group")
    ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.grid(axis="y", visible=False)
    fig.suptitle("What the LightGBM model relies on", y=1.02)
    save(fig, "11_feature_importance.png")


def fig_svm_kernels():
    import csv
    rows = []
    with open(ROOT / "reports" / "svm_cv.csv") as f:
        for rec in csv.reader(f):
            if rec[0] == "setup" or not rec[0].startswith("full|") or not rec[0].endswith("|orig"):
                continue
            if rec[-6] != "standard" or "balanced" in rec:
                continue
            rows.append({"variant": rec[1], "train": float(rec[-5]), "cv": float(rec[-4]), "minutes": float(rec[-1])})
    d = pd.DataFrame(rows).sort_values("cv").groupby("variant").tail(1)
    names = {"linear": "Linear", "rbf_exact": "Exact RBF\n(30k subsample)", "rbf_nystroem": "Nyström RBF",
             "rbf_fourier": "Random Fourier\n(RBF approx.)", "poly_nystroem": "Nyström\npolynomial",
             "poly_sketch": "Polynomial\nsketch"}
    d = d.set_index("variant").loc[list(names)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(d))
    ax.bar(x - 0.18, d.train, 0.34, color="#b7cff0", label="Train")
    ax.bar(x + 0.18, d.cv, 0.34, color="#1f5fae", label="5-fold CV")
    for xi, (tr_, cv_) in enumerate(zip(d.train, d.cv)):
        ax.text(xi - 0.18, tr_ + 0.01, f"{tr_:.3f}", ha="center", fontsize=7, color=INK)
        ax.text(xi + 0.18, cv_ + 0.01, f"{cv_:.3f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, [names[v] for v in d.index])
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Macro F1")
    ax.set_title("SVM kernels / feature transformations (full features): curved boundaries did not beat linear")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False)
    save(fig, "12_svm_kernels.png")


def fig_ordinal(out):
    y, s = out["y_test"], out["Ordinal_LightGBM__raw"]
    th = json.loads((EXP / "settings.json").read_text())["ordinal_thresholds"]
    blues = ["#9ec2ee", "#2a78d6", "#123e78"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 70)
    for k in (0, 1, 2):
        ax.hist(s[y == k], bins=bins, histtype="step", lw=2, color=blues[k], label=f"Actual {BIN_LABELS[k]}")
    for t, lbl in zip(th, ("<60 | 60-120", "60-120 | >120")):
        ax.axvline(t, color=INK, ls="--", lw=1)
        ax.text(t, ax.get_ylim()[1] * 0.96, f" threshold\n {lbl}", fontsize=7, color=INK, va="top")
    ax.set_xlabel("Predicted score = predicted log(1 + citations)")
    ax.set_ylabel("Number of test papers")
    ax.set_title("Ordinal LightGBM: one predicted score, cut into bins by two tuned thresholds")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "13_ordinal_scores.png")


def main():
    out, metrics = load()
    print("Saving figures to reports/figures/:")
    fig_dataset()
    fig_confusion(out)
    class_curves(out, "roc")
    class_curves(out, "pr")
    fig_calibration(out, metrics)
    fig_per_class(out)
    fig_model_comparison(metrics)
    fig_timing(metrics)
    if (EXP / "scaling.csv").exists():
        fig_scaling()
    if (EXP / "learning_curves.csv").exists():
        fig_learning()
    fig_importance()
    fig_svm_kernels()
    fig_ordinal(out)


if __name__ == "__main__":
    main()
