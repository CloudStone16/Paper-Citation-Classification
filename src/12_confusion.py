"""Confusion matrices on the test set for the write-up and slides.

Models compared (original task: <60 / 60-120 / >120):
  - year-only rule (most common bin of the paper's year, learned on training rows)
  - Stanford-style neural network (rebuilt as in 07_baseline.py)
  - our best model: ordinal LightGBM on full features (loaded from models/full_orig.joblib)

Outputs: reports/confusion_matrices.png, reports/confusion_matrices.csv
Rows = actual bin, columns = predicted bin; cells show the share of each actual bin (row-normalized).
"""
import importlib.util
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common import build_dataset, load_frame
from config import BIN_LABELS, PROCESSED, ROOT, SEED

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    df = load_frame()
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    y = df.label.to_numpy()

    # 1. Year-only rule
    rule = df[tr].groupby("year").label.agg(lambda s: s.mode()[0])
    pred_year = df.year[te].map(rule).to_numpy()

    # 2. Stanford-style NN, same inputs and settings as 07_baseline.py
    emb = np.load(PROCESSED / "emb_minilm_title.npy")
    journal_code = df.journal.astype("category").cat.codes.to_numpy()
    numeric = np.column_stack([df.year, df.title_chars, journal_code]).astype(float)
    X = np.hstack([emb, StandardScaler().fit(numeric[tr]).transform(numeric)])
    nn = MLPClassifier((256, 128), early_stopping=True, random_state=SEED, max_iter=200).fit(X[tr], y[tr])
    pred_nn = nn.predict(X[te])

    # 3. Our best model, from the saved bundle
    models = load_module("models", "10_models.py")
    bundle = joblib.load(ROOT / "models" / "full_orig.joblib")
    X_all, frame, _ = build_dataset(bundle["feature_set"], bundle["emb"].split(","), bundle["n_pca"])
    assert (frame.paper_id.to_numpy() == df.paper_id.to_numpy()).all()
    outs = models.predict_all(bundle["fitted"], X_all[te].reset_index(drop=True))
    pred_ours = models.cut(outs["ordinal"][:, 0], bundle["thresholds"])

    panels = [("Year-only rule", pred_year), ("Stanford-style NN (rebuilt)", pred_nn),
              ("Ours: ordinal LightGBM, full features", pred_ours)]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, (name, pred) in zip(axes, panels):
        cm = confusion_matrix(y[te], pred, labels=[0, 1, 2])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{share[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8,
                        color="white" if share[i, j] > 0.55 else "black")
                rows.append({"model": name, "actual": BIN_LABELS[i], "predicted": BIN_LABELS[j],
                             "count": int(cm[i, j]), "share_of_actual": round(share[i, j], 4)})
        ax.set_xticks(range(3), BIN_LABELS, fontsize=8)
        ax.set_yticks(range(3), BIN_LABELS, fontsize=8)
        ax.set_xlabel("Predicted bin", fontsize=8)
        ax.set_title(name, fontsize=9)
    axes[0].set_ylabel("Actual bin", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "confusion_matrices.png", dpi=200)
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "confusion_matrices.csv", index=False)
    for name, pred in panels:
        cm = confusion_matrix(y[te], pred, labels=[0, 1, 2])
        print(name, "\n", (cm / cm.sum(axis=1, keepdims=True)).round(3))
    print("Saved reports/confusion_matrices.png and .csv")


if __name__ == "__main__":
    main()
