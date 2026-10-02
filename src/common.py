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


# ---------------------------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------------------------
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, PowerTransformer,  # noqa: E402
                                   QuantileTransformer, StandardScaler, TargetEncoder)

TASK_TARGETS = {"orig": "label", "yn": "label_yn"}

BASE_NUMERIC = ["year", "title_chars", "title_words", "title_has_colon", "title_has_question",
                "title_has_digit", "title_has_acronym", "journal_train_count"]
BASE_ONEHOT = ["year_cat"]
BASE_TARGET_ENC = ["journal"]

FULL_NUMERIC = BASE_NUMERIC + [
    "is_enriched", "has_doi", "n_authors", "n_references", "n_institutions", "n_countries",
    "intl_collab", "n_funders", "has_funder", "n_keywords", "n_pages", "is_oa", "has_abstract",
    "abstract_words", "abstract_flesch", "abstract_sentence_len",
    "inst_share_education", "inst_share_healthcare", "inst_share_company", "inst_share_government",
    "inst_share_facility", "inst_share_nonprofit", "title_abstract_cos"]
FULL_ONEHOT = BASE_ONEHOT + ["domain", "field", "oa_status", "oa_type", "country_top"]
FULL_TARGET_ENC = BASE_TARGET_ENC + ["primary_country", "first_author_country", "subfield", "topic"]

# Heavy-tailed counts get log1p before scaling
LOG_COLS = {"journal_train_count", "n_authors", "n_references", "n_institutions", "n_countries",
            "n_funders", "n_pages", "abstract_words"}

FEATURE_SETS = {
    "base": (BASE_NUMERIC, BASE_ONEHOT, BASE_TARGET_ENC),
    "full": (FULL_NUMERIC, FULL_ONEHOT, FULL_TARGET_ENC),
}


def pca_embedding(name, train_mask, n_components, mask_name=None):
    """PCA-compress a cached embedding. PCA is unsupervised and fitted on training rows only."""
    emb = np.load(PROCESSED / f"emb_{name}.npy")
    fit_rows = train_mask
    if mask_name:  # abstracts: fit only on rows that really have one
        fit_rows = train_mask & np.load(PROCESSED / f"emb_{mask_name}.npy")
    pca = PCA(n_components=n_components, random_state=0).fit(emb[fit_rows])
    out = pca.transform(emb).astype(np.float32)
    if mask_name:
        out[~np.load(PROCESSED / f"emb_{mask_name}.npy")] = 0.0
    return out, pca.explained_variance_ratio_.sum()


def build_dataset(feature_set, embeddings, n_pca=128):
    """Return (X DataFrame, frame with labels/split, column spec) for a feature set.

    embeddings: list of embedding names, e.g. ["specter2_title", "specter2_abstract"]
    """
    feats = pd.read_parquet(PROCESSED / "features.parquet") if (PROCESSED / "features.parquet").exists() \
        else None
    if feats is None:  # enrichment not built yet: derive base features directly
        raise FileNotFoundError("Run 08_features.py first")
    feats["year_cat"] = feats["year"].astype(str)
    if "primary_country" in feats:
        top = feats.loc[feats.split == "train", "primary_country"].value_counts().head(30).index
        feats["country_top"] = feats["primary_country"].where(feats["primary_country"].isin(top), "OTHER")
        feats.loc[feats["primary_country"].isna(), "country_top"] = "UNKNOWN"

    numeric, onehot, target_enc = FEATURE_SETS[feature_set]
    numeric = [c for c in numeric if c in feats]
    onehot = [c for c in onehot if c in feats]
    target_enc = [c for c in target_enc if c in feats]
    for c in onehot + target_enc:
        feats[c] = feats[c].fillna("UNKNOWN").astype(str)

    train_mask = (feats.split == "train").to_numpy()
    blocks, emb_cols = [feats[numeric + onehot + target_enc].reset_index(drop=True)], []
    for name in embeddings:
        mask_name = f"{name}_mask" if name.endswith("abstract") else None
        arr, var = pca_embedding(name, train_mask, n_pca, mask_name)
        cols = [f"{name}_pc{i}" for i in range(arr.shape[1])]
        blocks.append(pd.DataFrame(arr, columns=cols))
        emb_cols += cols
        print(f"  {name}: PCA {n_pca} keeps {var:.1%} of variance")
    X = pd.concat(blocks, axis=1)
    spec = {"numeric": numeric, "onehot": onehot, "target_enc": target_enc, "emb": emb_cols}
    return X, feats.reset_index(drop=True), spec


def make_preprocessor(spec, scaler="standard"):
    """Column-wise preprocessing; fitted inside each CV fold, so it never sees validation labels."""
    scalers = {
        "standard": StandardScaler,
        "quantile": lambda: QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=0),
        "power": PowerTransformer,
    }
    log_idx = [i for i, c in enumerate(spec["numeric"]) if c in LOG_COLS]

    def log1p_some(a):
        a = np.array(a, dtype=float, copy=True)
        a[:, log_idx] = np.log1p(np.clip(a[:, log_idx], 0, None))
        return a

    numeric_pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                 FunctionTransformer(log1p_some), scalers[scaler]())
    parts = [("num", numeric_pipe, spec["numeric"]),
             ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False),
              spec["onehot"]),
             ("te", make_pipeline(TargetEncoder(target_type="multiclass", cv=KFold(5, shuffle=True, random_state=0)), StandardScaler()),
              spec["target_enc"])]
    if spec["emb"]:
        parts.append(("emb", StandardScaler(), spec["emb"]))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def cv_splits(frame):
    """The fixed 5 CV folds, as (train_idx, val_idx) positions within the training rows."""
    folds = frame.loc[frame.split == "train", "cv_fold"].to_numpy()
    return [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(5)]
