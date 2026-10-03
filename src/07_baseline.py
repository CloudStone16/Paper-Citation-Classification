"""Reproduce the Stanford baselines on our split, plus a year-only floor.

Stanford inputs: title embedding + year + title length + journal as an integer code.
They used Google's Universal Sentence Encoder; we use all-MiniLM-L6-v2 as a comparable small
general-purpose encoder (their report found embedding choice made little difference).

Task: the paper's bins (<60, 60-120, >120), comparable with their reported F1.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from common import load_frame, log_result, scores
from config import PROCESSED, SEED


def year_only(df, target):
    train = df[df.split == "train"]
    rule = train.groupby("year")[target].agg(lambda s: s.mode()[0])
    for part in ("train", "test"):
        d = df[df.split == part]
        log_result("year-only majority rule", TASKS[target], part, scores(d[target], d.year.map(rule)))


def stanford_features(df):
    emb = np.load(PROCESSED / "emb_minilm_title.npy")
    journal_code = df.journal.astype("category").cat.codes.to_numpy()  # integer code, as in the report
    numeric = np.column_stack([df.year, df.title_chars, journal_code]).astype(float)
    return emb, numeric


def run_models(df, target, emb, numeric):
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    y = df[target].to_numpy()

    # Standardize numeric features with training statistics only.
    sc = StandardScaler().fit(numeric[tr])
    X = np.hstack([emb, sc.transform(numeric)])

    # Naive Bayes needs non-negative inputs, so it gets a min-max scaled copy (as in the report).
    mm = MinMaxScaler().fit(X[tr])
    X_pos = np.clip(mm.transform(X), 0, None)

    models = {
        "Stanford NB (MultinomialNB)": (MultinomialNB(), X_pos),
        "Stanford RF (default, deep trees)": (RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED), X),
        "Stanford NN (MLP 256-128)": (MLPClassifier((256, 128), early_stopping=True, random_state=SEED, max_iter=200), X),
    }
    for name, (model, feats) in models.items():
        model.fit(feats[tr], y[tr])
        for part, mask in (("train", tr), ("test", te)):
            log_result(name, TASKS[target], part, scores(y[mask], model.predict(feats[mask])),
                       notes="MiniLM title emb + year + title length + journal code")


TASKS = {"label": "orig"}

if __name__ == "__main__":
    df = load_frame()
    emb, numeric = stanford_features(df)
    for target in TASKS:
        year_only(df, target)
        run_models(df, target, emb, numeric)
