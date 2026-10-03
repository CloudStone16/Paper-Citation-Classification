"""Higher-capacity models + stacking, evaluated on the same CV folds as the SVM track.

  lgbm        LightGBM classifier (regularized; early stopping on an inner split, never the CV fold)
  ordinal     LightGBM regression on a continuous citation score, cut into bins with thresholds
              tuned on out-of-fold predictions (uses the order <60 < 60-120 < >120)
  mlp         neural network (2 hidden layers, L2 penalty, early stopping)
  svm         best SVM configuration from 09_svm.py (decision scores used for stacking)
  stack       logistic regression over the out-of-fold outputs of all models above

Usage:
  python 10_models.py --feature-set full --emb specter2_title,specter2_abstract --task orig [--final]
"""
import argparse
import os
import importlib.util
import json
from itertools import product

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from common import (TASK_TARGETS, build_dataset, cv_splits, log_result, make_preprocessor, scores)
from config import ROOT, SEED

LGB_PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0,
                  n_jobs=int(os.getenv("LGBM_THREADS", "-1")), verbose=-1, random_state=SEED)


def continuous_target(frame, task):
    """Ordinal target: log citations (orig) or within-year percentile rank (yn)."""
    if task == "orig":
        return np.log1p(frame.citations.to_numpy())
    return frame.groupby("year").citations.rank(pct=True).to_numpy()


def cut(pred, thresholds):
    return (pred > thresholds[0]).astype(int) + (pred > thresholds[1]).astype(int)


def tune_thresholds(pred, y):
    """Grid search the two bin edges that maximize macro F1 on out-of-fold predictions."""
    qs = np.quantile(pred, np.linspace(0.05, 0.95, 91))
    best = (-1, None)
    for a, b in product(qs, qs):
        if a < b:
            f1 = scores(y, cut(pred, (a, b)))["macro_f1"]
            if f1 > best[0]:
                best = (f1, (a, b))
    return best[1]


def fit_lgbm(Xtr, ytr, regression=False):
    Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.1, random_state=SEED,
                                      stratify=None if regression else ytr)
    model = (lgb.LGBMRegressor if regression else lgb.LGBMClassifier)(**LGB_PARAMS)
    model.fit(Xa, ya, eval_set=[(Xb, yb)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def load_svm(feature_set, task, n_feat):
    path = ROOT / "reports" / f"svm_best_{feature_set}_{task}.json"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("svm", ROOT / "src" / "09_svm.py")
    svm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(svm)
    choice = json.loads(path.read_text())
    cfg = {k: choice[k] for k in ("C", "gamma_mult", "class_weight") if k in choice}
    if choice["variant"] == "rbf_exact":
        return None
    return svm.variant(choice["variant"], n_feat, **cfg), choice["scaler"]


def fit_all(pre_by_scaler, Xtr, ytr, rtr, svm_cfg):
    """Fit every base model on one training set; return fitted objects."""
    pre = clone(pre_by_scaler["standard"]).fit(Xtr, ytr)
    Ztr = pre.transform(Xtr)
    fitted = {"pre": pre,
              "lgbm": fit_lgbm(Ztr, ytr),
              "ordinal": fit_lgbm(Ztr, rtr, regression=True),
              "mlp": MLPClassifier((512, 256), alpha=1e-3, batch_size=512, learning_rate_init=1e-3,
                                   early_stopping=True, n_iter_no_change=5, max_iter=100,
                                   random_state=SEED).fit(Ztr, ytr)}
    if svm_cfg:
        model, scaler = svm_cfg
        spre = clone(pre_by_scaler[scaler]).fit(Xtr, ytr)
        fitted["svm"] = (spre, clone(model).fit(spre.transform(Xtr), ytr))
    return fitted


def predict_all(fitted, X):
    Z = fitted["pre"].transform(X)
    out = {"lgbm": fitted["lgbm"].predict_proba(Z), "ordinal": fitted["ordinal"].predict(Z)[:, None],
           "mlp": fitted["mlp"].predict_proba(Z)}
    if "svm" in fitted:
        spre, m = fitted["svm"]
        out["svm"] = m.decision_function(spre.transform(X))
    return out


def stack_features(outs):
    return np.hstack([outs[k] for k in sorted(outs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="full", choices=["base", "full"])
    ap.add_argument("--emb", default="specter2_title")
    ap.add_argument("--task", default="orig", choices=list(TASK_TARGETS))
    ap.add_argument("--n-pca", type=int, default=128)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    X_all, frame, spec = build_dataset(args.feature_set, [e for e in args.emb.split(",") if e], args.n_pca)
    y_all = frame[TASK_TARGETS[args.task]].to_numpy()
    r_all = continuous_target(frame, args.task)
    tr = (frame.split == "train").to_numpy()
    X, y, r = X_all[tr].reset_index(drop=True), y_all[tr], r_all[tr]
    pres = {s: make_preprocessor(spec, s) for s in ("standard", "quantile", "power")}
    n_feat = clone(pres["standard"]).fit(X.head(5000), y[:5000]).transform(X.head(5)).shape[1]
    svm_cfg = load_svm(args.feature_set, args.task, n_feat)
    tag = f"{args.feature_set} | {args.emb}"
    print(f"{tag} | {args.task}: {len(X):,} train rows, {n_feat} features, svm={'yes' if svm_cfg else 'no'}")

    # Out-of-fold predictions for every model (used for CV scores, thresholds and the stacker)
    oof, train_scores = {}, {}
    for k, (a, b) in enumerate(cv_splits(frame)):
        fitted = fit_all(pres, X.iloc[a], y[a], r[a], svm_cfg)
        for name, pred in predict_all(fitted, X.iloc[b]).items():
            oof.setdefault(name, np.zeros((len(X), pred.shape[1])))[b] = pred
        tr_pred = predict_all(fitted, X.iloc[a])
        for name in ("lgbm", "mlp"):
            train_scores.setdefault(name, []).append(scores(y[a], tr_pred[name].argmax(1))["macro_f1"])
        print(f"  fold {k} done", flush=True)

    thresholds = tune_thresholds(oof["ordinal"][:, 0], y)
    cv_preds = {"lgbm": oof["lgbm"].argmax(1), "mlp": oof["mlp"].argmax(1),
                "ordinal": cut(oof["ordinal"][:, 0], thresholds)}
    if "svm" in oof:
        cv_preds["svm"] = oof["svm"].argmax(1)

    # Stacker: trained on out-of-fold outputs; its own CV score uses a nested split of those outputs
    S = stack_features(oof)
    stacker = LogisticRegression(max_iter=2000, C=1.0)
    stack_oof = np.zeros(len(X), dtype=int)
    for a, b in cv_splits(frame):
        stack_oof[b] = clone(stacker).fit(S[a], y[a]).predict(S[b])
    cv_preds["stack"] = stack_oof

    for name, pred in cv_preds.items():
        note = f"train(fold avg) macroF1={np.mean(train_scores[name]):.4f}" if name in train_scores else ""
        log_result(f"{name} ({args.feature_set})", args.task, "cv", scores(y, pred), notes=f"{args.emb} {note}")

    if not args.final:
        return

    # Final: refit everything on all training rows; the test set is touched only here.
    fitted = fit_all(pres, X, y, r, svm_cfg)
    stacker.fit(S, y)
    te = ~tr
    test_out = predict_all(fitted, X_all[te])
    train_out = predict_all(fitted, X)
    finals = {
        "lgbm": (train_out["lgbm"].argmax(1), test_out["lgbm"].argmax(1)),
        "mlp": (train_out["mlp"].argmax(1), test_out["mlp"].argmax(1)),
        "ordinal": (cut(train_out["ordinal"][:, 0], thresholds), cut(test_out["ordinal"][:, 0], thresholds)),
        "stack": (stacker.predict(stack_features(train_out)), stacker.predict(stack_features(test_out))),
    }
    if "svm" in test_out:
        finals["svm"] = (train_out["svm"].argmax(1), test_out["svm"].argmax(1))
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    bundle_path = out_dir / f"{args.feature_set}_{args.task}.joblib"
    joblib.dump({"fitted": fitted, "stacker": stacker, "thresholds": thresholds, "spec": spec,
                 "feature_set": args.feature_set, "emb": args.emb, "task": args.task, "n_pca": args.n_pca},
                bundle_path, compress=3)
    print(f"Saved models to {bundle_path}", flush=True)
    for name, (p_train, p_test) in finals.items():
        log_result(f"{name} ({args.feature_set})", args.task, "train", scores(y, p_train), notes=args.emb)
        log_result(f"{name} ({args.feature_set})", args.task, "test", scores(y_all[te], p_test), notes=args.emb)


if __name__ == "__main__":
    main()
