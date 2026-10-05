"""ROC curves on the test set (original task: <60 / 60-120 / >120).

One-vs-rest ROC for each bin, plus the macro-average, for:
  - year-only rule   (score = share of training papers of that year that fall in the bin)
  - Stanford-style neural network (rebuilt as in 07_baseline.py / 12_confusion.py)
  - our LightGBM classifier, full features (class probabilities)
  - our stacking ensemble, full features (class probabilities)
  - our best model: ordinal LightGBM, full features (continuous citation score)

All fitted models are loaded from models/full_orig.joblib; nothing is retrained except the
small Stanford-style NN. The test set is only scored, never used for fitting.

Ordinal model: it outputs one continuous score (predicted log citations), not 3 probabilities, so
per-bin scores are derived from it:
  <60      -> -score             (lower predicted citations = more likely)
  >120     -> +score             (higher predicted citations = more likely)
  60-120   -> -|score - centre|  (closer to the middle of the two bin edges = more likely; heuristic)

Outputs: reports/roc_curves.png, reports/roc_auc.csv
"""
import importlib.util
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

from common import build_dataset, load_frame
from config import BIN_LABELS, PROCESSED, ROOT, SEED

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
CLASSES = [0, 1, 2]


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ordinal_to_class_scores(score, thresholds):
    centre = (thresholds[0] + thresholds[1]) / 2
    return np.column_stack([-score, -np.abs(score - centre), score])


def macro_curve(y_bin, S, grid=np.linspace(0, 1, 501)):
    tprs = []
    for k in CLASSES:
        fpr, tpr, _ = roc_curve(y_bin[:, k], S[:, k])
        tprs.append(np.interp(grid, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    macro_auc = np.mean([roc_auc_score(y_bin[:, k], S[:, k]) for k in CLASSES])
    return grid, mean_tpr, macro_auc


def main():
    df = load_frame()
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    y = df.label.to_numpy()
    y_te = y[te]
    y_bin = label_binarize(y_te, classes=CLASSES)

    all_scores = {}  # model name -> (n_test, 3) array, higher = more likely that bin

    # 1. Year-only rule: P(bin | year) estimated on training rows
    prob_by_year = pd.crosstab(df.year[tr], y[tr], normalize="index").reindex(columns=CLASSES, fill_value=0)
    all_scores["Year-only rule"] = prob_by_year.reindex(df.year[te]).fillna(1 / 3).to_numpy()

    # 2. Stanford-style NN, same inputs and settings as 12_confusion.py
    emb = np.load(PROCESSED / "emb_minilm_title.npy")
    journal_code = df.journal.astype("category").cat.codes.to_numpy()
    numeric = np.column_stack([df.year, df.title_chars, journal_code]).astype(float)
    X = np.hstack([emb, StandardScaler().fit(numeric[tr]).transform(numeric)])
    nn = MLPClassifier((256, 128), early_stopping=True, random_state=SEED, max_iter=200).fit(X[tr], y[tr])
    all_scores["Stanford-style NN (rebuilt)"] = nn.predict_proba(X[te])

    # 3. Our models, from the saved bundle
    models = load_module("models", "10_models.py")
    bundle = joblib.load(ROOT / "models" / "full_orig.joblib")
    X_all, frame, _ = build_dataset(bundle["feature_set"], bundle["emb"].split(","), bundle["n_pca"])
    assert (frame.paper_id.to_numpy() == df.paper_id.to_numpy()).all()
    outs = models.predict_all(bundle["fitted"], X_all[te].reset_index(drop=True))

    all_scores["Ours: LightGBM classifier"] = outs["lgbm"]
    all_scores["Ours: stacking ensemble"] = bundle["stacker"].predict_proba(models.stack_features(outs))
    all_scores["Ours: ordinal LightGBM (best)"] = ordinal_to_class_scores(outs["ordinal"][:, 0],
                                                                          bundle["thresholds"])

    styles = {
        "Year-only rule": dict(color="#999999", ls=":", lw=1.6),
        "Stanford-style NN (rebuilt)": dict(color="#d95f02", ls="--", lw=1.6),
        "Ours: LightGBM classifier": dict(color="#7570b3", ls="-", lw=1.4),
        "Ours: stacking ensemble": dict(color="#1b9e77", ls="-", lw=1.4),
        "Ours: ordinal LightGBM (best)": dict(color="#1f4e9c", ls="-", lw=2.4),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    rows = []
    panels = [(f"{BIN_LABELS[k]} vs rest", k) for k in CLASSES] + [("Macro-average", None)]
    for ax, (title, k) in zip(axes.ravel(), panels):
        for name, S in all_scores.items():
            if k is None:
                fpr, tpr, a = macro_curve(y_bin, S)
                label_cls = "macro-average"
            else:
                fpr, tpr, _ = roc_curve(y_bin[:, k], S[:, k])
                a = roc_auc_score(y_bin[:, k], S[:, k])
                label_cls = BIN_LABELS[k]
            ax.plot(fpr, tpr, label=f"{name} (AUC {a:.3f})", **styles[name])
            rows.append({"model": name, "class": label_cls, "auc": round(a, 4)})
        ax.plot([0, 1], [0, 1], color="black", lw=0.8, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("ROC curves on the held-out test set (one-vs-rest)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "roc_curves.png", dpi=200)

    auc_table = pd.DataFrame(rows)
    auc_table.to_csv(ROOT / "reports" / "roc_auc.csv", index=False)
    print(auc_table.pivot(index="model", columns="class", values="auc").round(3).to_string())
    print("Saved reports/roc_curves.png and reports/roc_auc.csv")


if __name__ == "__main__":
    main()
