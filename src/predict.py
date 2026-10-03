"""Load the saved models and use them, without retraining.

Usage (from the project folder):
  .venv/bin/python src/predict.py --task orig --n 10        # predictions for 10 random test papers
  .venv/bin/python src/predict.py --task yn --evaluate      # recompute test scores from the saved models
  .venv/bin/python src/predict.py --task orig --title "graphene"   # test papers whose title contains a word

--task orig : Stanford's bins (<60 / 60-120 / >120)
--task yn   : within-year thirds (bottom / middle / top third of the paper's year)
"""
import argparse
import importlib.util
import warnings

import joblib
import numpy as np

from common import build_dataset, scores
from config import BIN_LABELS, ROOT

warnings.filterwarnings("ignore")
LABELS = {"orig": BIN_LABELS, "yn": ["bottom third", "middle third", "top third"]}
TARGET = {"orig": "label", "yn": "label_yn"}

spec = importlib.util.spec_from_file_location("models", ROOT / "src" / "10_models.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def predict(bundle, X):
    outs = M.predict_all(bundle["fitted"], X)
    preds = {
        "ordinal (best)": M.cut(outs["ordinal"][:, 0], bundle["thresholds"]),
        "stack": bundle["stacker"].predict(M.stack_features(outs)),
        "lgbm": outs["lgbm"].argmax(1),
        "mlp": outs["mlp"].argmax(1),
    }
    if "svm" in outs:
        preds["svm"] = outs["svm"].argmax(1)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="orig", choices=["orig", "yn"])
    ap.add_argument("--features", default="full", choices=["full", "base"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--title", help="only test papers whose title contains this text")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    bundle = joblib.load(ROOT / "models" / f"{args.features}_{args.task}.joblib")
    print("Rebuilding feature matrix from cached features/embeddings (about a minute)...")
    X_all, frame, _ = build_dataset(bundle["feature_set"], bundle["emb"].split(","), bundle["n_pca"])
    test = (frame.split == "test").to_numpy()
    X, fr = X_all[test].reset_index(drop=True), frame[test].reset_index(drop=True)
    y = fr[TARGET[args.task]].to_numpy()

    if args.evaluate:
        for name, p in predict(bundle, X).items():
            s = scores(y, p)
            print(f"{name:<15} test macro F1 {s['macro_f1']:.4f}  weighted F1 {s['weighted_f1']:.4f}  acc {s['accuracy']:.4f}")
        return

    titles = (ROOT / "data" / "interim" / "clean.parquet")
    import pandas as pd
    meta = pd.read_parquet(titles, columns=["paper_id", "title", "journal_raw"]).set_index("paper_id")
    rows = np.arange(len(fr))
    if args.title:
        rows = rows[meta.loc[fr.paper_id, "title"].str.contains(args.title, case=False).to_numpy()]
    rows = np.random.default_rng().choice(rows, min(args.n, len(rows)), replace=False)
    p = predict(bundle, X.iloc[rows])["ordinal (best)"]
    names = LABELS[args.task]
    for i, r in enumerate(rows):
        m = meta.loc[fr.paper_id[r]]
        ok = "correct" if p[i] == y[r] else "wrong"
        print(f"\n{m.title[:110]}\n  {m.journal_raw[:60]}, {fr.year[r]}, {fr.citations[r]} citations\n"
              f"  actual: {names[y[r]]:<13} predicted: {names[p[i]]:<13} ({ok})")


if __name__ == "__main__":
    main()
