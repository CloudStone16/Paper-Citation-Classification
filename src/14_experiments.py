"""Final model comparison: probabilistic models vs SVM vs LightGBM (original task).

Models (all on the full feature set, same preprocessing, same split and CV folds):
  - Gaussian Naive Bayes         probabilistic, generative (Bayes' theorem, independent features)
  - Logistic regression          probabilistic, discriminative (softmax over the 3 bins)
  - SVM (linear)                 best SVM variant from 09_svm.py; margin-based, decision scores
  - LightGBM classifier          gradient-boosted trees, class probabilities
  - Ordinal LightGBM             LightGBM regression on log(1 + citations), cut into bins

Stages (run in order; each saves its results so figures can be redrawn without retraining):
  python 14_experiments.py main      # CV + final fit, timing, test scores/probabilities  (~15 min)
  python 14_experiments.py learning  # learning curves: train/validation F1 vs training size (~15 min)
  python 14_experiments.py scaling   # fit time vs training size, incl. exact RBF SVM (~25 min)

Outputs in reports/experiments/.
"""
import argparse
import importlib.util
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from common import build_dataset, cv_splits, make_preprocessor
from config import ROOT, SEED

warnings.filterwarnings("ignore")
OUT = ROOT / "reports" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
EMB = ["qwen3-0.6b_title", "specter2_abstract"]
MODELS = ["Naive Bayes", "Logistic Regression", "SVM (linear)", "LightGBM classifier", "Ordinal LightGBM"]


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M10 = load_module("models10", "10_models.py")
M09 = load_module("svm09", "09_svm.py")
THRESHOLDS = tuple(joblib.load(ROOT / "models" / "full_orig.joblib")["thresholds"])
SVM_CFG = json.loads((ROOT / "reports" / "svm_best_full_orig.json").read_text())


def ordinal_class_scores(score):
    """Per-bin scores from the ordinal model's single score (higher = more likely that bin)."""
    centre = (THRESHOLDS[0] + THRESHOLDS[1]) / 2
    return np.column_stack([-score, -np.abs(score - centre), score])


def make_model(name, n_feat, lr_C=1.0):
    if name == "Naive Bayes":
        return GaussianNB()
    if name == "Logistic Regression":
        return LogisticRegression(C=lr_C, max_iter=3000)
    if name == "SVM (linear)":
        cfg = {k: SVM_CFG[k] for k in ("C", "gamma_mult", "class_weight") if k in SVM_CFG}
        return M09.variant(SVM_CFG["variant"], n_feat, **cfg)
    return None  # LightGBM models are built by M10.fit_lgbm (inner early-stopping split)


def fit(name, Z, y, r, n_feat, lr_C=1.0):
    if name == "LightGBM classifier":
        return M10.fit_lgbm(Z, y)
    if name == "Ordinal LightGBM":
        return M10.fit_lgbm(Z, r, regression=True)
    return make_model(name, n_feat, lr_C).fit(Z, y)


def scores_of(name, model, Z):
    """Return (predicted bin, per-bin score matrix, probabilities or None)."""
    if name == "Ordinal LightGBM":
        s = model.predict(Z)
        return M10.cut(s, THRESHOLDS), ordinal_class_scores(s), None
    if name == "SVM (linear)":
        d = model.decision_function(Z)
        return d.argmax(1), d, None
    p = model.predict_proba(Z)
    return p.argmax(1), p, p


def data():
    X_all, frame, spec = build_dataset("full", EMB, 128)
    tr = (frame.split == "train").to_numpy()
    y_all = frame.label.to_numpy()
    r_all = np.log1p(frame.citations.to_numpy())
    return X_all, frame, spec, tr, y_all, r_all


def stage_main():
    X_all, frame, spec, tr, y_all, r_all = data()
    X, y, r = X_all[tr].reset_index(drop=True), y_all[tr], r_all[tr]
    Xte, yte = X_all[~tr].reset_index(drop=True), y_all[~tr]
    folds = cv_splits(frame)

    cache = OUT / "cv_cache.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}

    # Logistic regression: choose C by CV (the only new model with a hyperparameter to tune)
    lr_cv = {float(k): v for k, v in cached.get("lr_cv", {}).items()}
    for C in (0.01, 0.1, 1.0):
        if C in lr_cv:
            print(f"  logistic regression C={C}: CV macro F1 {lr_cv[C]:.4f} (cached)", flush=True)
            continue
        f1s = []
        for a, b in folds:
            pre = clone(make_preprocessor(spec)).fit(X.iloc[a], y[a])
            m = make_model("Logistic Regression", 0, C).fit(pre.transform(X.iloc[a]), y[a])
            f1s.append(f1_score(y[b], m.predict(pre.transform(X.iloc[b])), average="macro"))
        lr_cv[C] = float(np.mean(f1s))
        print(f"  logistic regression C={C}: CV macro F1 {lr_cv[C]:.4f}", flush=True)
        cache.write_text(json.dumps({**cached, "lr_cv": lr_cv}))
    lr_C = max(lr_cv, key=lr_cv.get)

    # 5-fold CV for the two probabilistic models (CV scores of the others come from 09/10)
    cv_f1 = dict(cached.get("cv_f1", {}))
    for name in ("Naive Bayes", "Logistic Regression"):
        if name in cv_f1:
            print(f"  {name}: CV macro F1 {cv_f1[name]:.4f} (cached)", flush=True)
            continue
        f1s = []
        for a, b in folds:
            pre = clone(make_preprocessor(spec)).fit(X.iloc[a], y[a])
            m = fit(name, pre.transform(X.iloc[a]), y[a], r[a], 0, lr_C)
            f1s.append(f1_score(y[b], scores_of(name, m, pre.transform(X.iloc[b]))[0], average="macro"))
        cv_f1[name] = float(np.mean(f1s))
        print(f"  {name}: CV macro F1 {cv_f1[name]:.4f}", flush=True)
        cached = {**cached, "lr_cv": lr_cv, "cv_f1": cv_f1}
        cache.write_text(json.dumps(cached))
    prev = pd.read_csv(ROOT / "reports" / "results.csv")
    prev = prev[(prev.task == "orig") & (prev.split == "cv") & prev.notes.fillna("").str.startswith(",".join(EMB))]
    for name, key in (("SVM (linear)", "svm (full)"), ("LightGBM classifier", "lgbm (full)"),
                      ("Ordinal LightGBM", "ordinal (full)")):
        cv_f1[name] = float(prev[prev.model == key].macro_f1.iloc[-1])

    # Final fit on all training rows, timed; test set scored once
    t = time.perf_counter()
    pre = clone(make_preprocessor(spec)).fit(X, y)
    Z, Zte = pre.transform(X), pre.transform(Xte)
    pre_sec = time.perf_counter() - t
    n_feat = Z.shape[1]
    feature_names = list(pre.get_feature_names_out())

    rows, saved = [], {"y_test": yte, "year_test": frame.year[~tr].to_numpy()}
    for name in MODELS:
        t = time.perf_counter()
        model = fit(name, Z, y, r, n_feat, lr_C)
        fit_sec = time.perf_counter() - t
        t = time.perf_counter()
        pred, S, P = scores_of(name, model, Zte)
        pred_sec = time.perf_counter() - t
        train_pred = scores_of(name, model, Z)[0]
        key = name.replace(" ", "_").replace("(", "").replace(")", "")
        saved[f"{key}__pred"], saved[f"{key}__scores"] = pred, S
        if P is not None:
            saved[f"{key}__proba"] = P
        if name == "Ordinal LightGBM":
            saved[f"{key}__raw"] = model.predict(Zte)
        if name == "LightGBM classifier":
            imp = pd.DataFrame({"feature": feature_names,
                                "gain": model.booster_.feature_importance(importance_type="gain")})
            imp.sort_values("gain", ascending=False).to_csv(OUT / "lgbm_feature_importance.csv", index=False)
        row = {"model": name, "fit_seconds": round(fit_sec, 2), "predict_seconds": round(pred_sec, 3),
               "train_macro_f1": round(f1_score(y, train_pred, average="macro"), 4),
               "cv_macro_f1": round(cv_f1[name], 4),
               "test_macro_f1": round(f1_score(yte, pred, average="macro"), 4),
               "test_weighted_f1": round(f1_score(yte, pred, average="weighted"), 4),
               "test_accuracy": round(accuracy_score(yte, pred), 4),
               "test_log_loss": round(log_loss(yte, P), 4) if P is not None else None}
        rows.append(row)
        print(f"  {name:<22} fit {fit_sec:7.1f}s  predict {pred_sec:6.2f}s  test macro F1 {row['test_macro_f1']:.4f}",
              flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metrics.csv", index=False)
    np.savez_compressed(OUT / "test_outputs.npz", **saved)
    (OUT / "settings.json").write_text(json.dumps({
        "logistic_regression_C": lr_C, "logistic_regression_cv": lr_cv, "svm": SVM_CFG,
        "ordinal_thresholds": list(map(float, THRESHOLDS)), "preprocessing_seconds": round(pre_sec, 2),
        "n_features": n_feat, "n_train": int(len(y)), "n_test": int(len(yte)),
        "lgbm_threads": int(M10.LGB_PARAMS["n_jobs"])}, indent=2))
    print(res.to_string(index=False))


def stage_learning():
    """Train on growing subsets of CV fold 0's training part, score on fold 0's validation part."""
    X_all, frame, spec, tr, y_all, r_all = data()
    X, y, r = X_all[tr].reset_index(drop=True), y_all[tr], r_all[tr]
    a, b = cv_splits(frame)[0]
    lr_C = json.loads((OUT / "settings.json").read_text())["logistic_regression_C"]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(a)
    rows = []
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        sub = order[: int(len(order) * frac)]
        pre = clone(make_preprocessor(spec)).fit(X.iloc[sub], y[sub])
        Zs, Zv = pre.transform(X.iloc[sub]), pre.transform(X.iloc[b])
        for name in MODELS:
            m = fit(name, Zs, y[sub], r[sub], Zs.shape[1], lr_C)
            rows.append({"model": name, "n_train": len(sub),
                         "train_macro_f1": f1_score(y[sub], scores_of(name, m, Zs)[0], average="macro"),
                         "val_macro_f1": f1_score(y[b], scores_of(name, m, Zv)[0], average="macro")})
            print(f"  n={len(sub):>6} {name:<22} train {rows[-1]['train_macro_f1']:.4f} "
                  f"val {rows[-1]['val_macro_f1']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "learning_curves.csv", index=False)


def stage_scaling():
    """Fit time vs number of training rows. Exact RBF SVM is capped at 30k rows (cost grows ~n^2)."""
    X_all, frame, spec, tr, y_all, r_all = data()
    X, y, r = X_all[tr].reset_index(drop=True), y_all[tr], r_all[tr]
    lr_C = json.loads((OUT / "settings.json").read_text())["logistic_regression_C"]
    pre = clone(make_preprocessor(spec)).fit(X, y)
    Z = pre.transform(X)
    n_feat = Z.shape[1]
    order = np.random.default_rng(SEED).permutation(len(y))
    rows = []
    for n in (2500, 5000, 10000, 20000, 30000, 60000, len(y)):
        idx = order[:n]
        for name in MODELS + ["SVM (exact RBF kernel)"]:
            if name == "SVM (exact RBF kernel)" and n > 30000:
                continue
            t = time.perf_counter()
            if name == "SVM (exact RBF kernel)":
                SVC(C=1.0, kernel="rbf", gamma=1.0 / n_feat, cache_size=2000).fit(Z[idx], y[idx])
            else:
                fit(name, Z[idx], y[idx], r[idx], n_feat, lr_C)
            sec = time.perf_counter() - t
            rows.append({"model": name, "n_train": n, "fit_seconds": round(sec, 3)})
            print(f"  n={n:>6} {name:<24} {sec:8.2f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "scaling.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["main", "learning", "scaling"])
    {"main": stage_main, "learning": stage_learning, "scaling": stage_scaling}[ap.parse_args().stage]()
