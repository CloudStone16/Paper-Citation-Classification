"""Helpers shared by the modelling scripts: data loading, labels, metrics, result logging."""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import INTERIM, PROCESSED, ROOT

RESULTS = ROOT / "reports" / "results.csv"


def load_frame():
    """Cleaned papers joined with their fixed split, in clean.parquet row order."""
    df = pd.read_parquet(INTERIM / "clean.parquet")
    split = pd.read_parquet(PROCESSED / "split.parquet")
    df = df.merge(split, on="paper_id", how="left", validate="1:1")
    df["label_yn"] = year_normalized_labels(df)
    return df


def year_normalized_labels(df):
    """Secondary task: bottom / middle / top third of citations *within the paper's year*.

    Cut points come from the training rows only, so the test set does not influence them.
    Year alone carries no information here, so it measures skill beyond 'older papers have more citations'.
    """
    labels = np.zeros(len(df), dtype=int)
    for year, idx in df.groupby("year").groups.items():
        rows = df.loc[idx]
        train_cites = rows.loc[rows.split == "train", "citations"]
        lo, hi = np.quantile(train_cites, [1 / 3, 2 / 3])
        labels[df.index.get_indexer(idx)] = (rows.citations > lo).astype(int) + (rows.citations > hi).astype(int)
    return labels


def scores(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def log_result(model, task, split, metrics, notes=""):
    row = {"time": datetime.now().isoformat(timespec="seconds"), "model": model, "task": task,
           "split": split, **{k: round(v, 4) for k, v in metrics.items()}, "notes": notes}
    out = pd.DataFrame([row])
    out.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
    print(f"{model:<32} {task:<10} {split:<6} macroF1={metrics['macro_f1']:.4f} "
          f"weightedF1={metrics['weighted_f1']:.4f} acc={metrics['accuracy']:.4f}")
