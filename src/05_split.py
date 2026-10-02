"""Create one fixed train/test split and 5 cross-validation folds.

- Stratified by label, so every split has the same bin proportions.
- Grouped by dup_group, so copies of the same title (e.g. a guideline co-published in several
  journals) never end up on both sides of a split.
- The test set (15%) is only used for the final evaluation.

Output: data/processed/split.parquet with columns paper_id, split ('train'/'test'), cv_fold (0-4, train only)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import INTERIM, PROCESSED, SEED

N_CV_FOLDS = 5


def main():
    df = pd.read_parquet(INTERIM / "clean.parquet", columns=["paper_id", "label", "dup_group", "year"])

    # 20 stratified, grouped chunks of ~5% each; 3 chunks (~15%) form the test set.
    outer = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    chunk = np.empty(len(df), dtype=int)
    for i, (_, idx) in enumerate(outer.split(df, df.label, df.dup_group)):
        chunk[idx] = i
    df["split"] = np.where(chunk < 3, "test", "train")

    df["cv_fold"] = -1
    train = df[df.split == "train"]
    inner = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, idx) in enumerate(inner.split(train, train.label, train.dup_group)):
        df.loc[train.index[idx], "cv_fold"] = k

    # Sanity checks: no group on both sides, proportions preserved
    groups = df.groupby("dup_group").split.nunique()
    assert (groups == 1).all(), "a duplicate group crosses the train/test boundary"
    print(df.split.value_counts().to_string())
    print("\nLabel share by split:\n", pd.crosstab(df.split, df.label, normalize="index").round(3).to_string())
    print("\nTrain rows per CV fold:", df[df.split == "train"].cv_fold.value_counts().sort_index().to_dict())

    out = PROCESSED / "split.parquet"
    df[["paper_id", "split", "cv_fold"]].to_parquet(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
