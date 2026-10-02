"""SVM track: compare kernels / feature transformations on the same CV folds.

Variants (all after the same column preprocessing):
  S1  linear       LinearSVC on the features as they are
  S2  rbf_exact    exact RBF-kernel SVC (cost grows with n^2, so tuned on a 30k subsample)
  S3  rbf_nystroem Nystroem approximation of the RBF kernel + LinearSVC (uses all rows)
  S4  rbf_fourier  random Fourier features (RBFSampler) + LinearSVC
  S5  poly_nystroem Nystroem polynomial kernel (degree 2) + LinearSVC
  S6  poly_sketch  PolynomialCountSketch (explicit degree-2 interactions) + LinearSVC
Then, for the best variant: scaler ablation (standard / quantile / power) and class weighting.

Usage:
  python 09_svm.py --feature-set base --emb specter2_title --task orig
  python 09_svm.py ... --final     # fit the chosen config on all training rows, evaluate on test once
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.kernel_approximation import Nystroem, PolynomialCountSketch, RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC

from common import (TASK_TARGETS, build_dataset, cv_splits, log_result, make_preprocessor,
                    scores)
from config import ROOT, SEED

N_COMP = 1000  # kept modest: each worker holds an n_rows x N_COMP matrix in memory
SUBSAMPLE = 30_000


def variant(name, n_features, C, gamma_mult=1.0, class_weight=None):
    gamma = gamma_mult / n_features  # RBF width relative to the number of (standardized) features
    svc = LinearSVC(C=C, class_weight=class_weight, dual="auto", max_iter=3000, random_state=SEED)
    if name == "linear":
        return svc
    if name == "rbf_exact":
        return SVC(C=C, kernel="rbf", gamma=gamma, class_weight=class_weight, cache_size=4000)
    if name == "rbf_nystroem":
        return make_pipeline(Nystroem(kernel="rbf", gamma=gamma, n_components=N_COMP, random_state=SEED), svc)
    if name == "rbf_fourier":
        return make_pipeline(RBFSampler(gamma=gamma, n_components=N_COMP, random_state=SEED), svc)
    if name == "poly_nystroem":
        return make_pipeline(Nystroem(kernel="poly", degree=2, gamma=gamma, coef0=1,
                                      n_components=N_COMP, random_state=SEED), svc)
    if name == "poly_sketch":
        return make_pipeline(PolynomialCountSketch(degree=2, n_components=N_COMP, random_state=SEED), svc)
    raise ValueError(name)


def run_fold(pre, model, X, y, tr, va, subsample):
    if subsample and len(tr) > subsample:
        tr = np.random.default_rng(SEED).choice(tr, subsample, replace=False)
    p = clone(pre).fit(X.iloc[tr], y[tr])
    Xtr, Xva = p.transform(X.iloc[tr]), p.transform(X.iloc[va])
    m = clone(model).fit(Xtr, y[tr])
    return scores(y[tr], m.predict(Xtr))["macro_f1"], scores(y[va], m.predict(Xva))


def cross_validate(X, y, folds, pre, model, subsample=None, n_jobs=2):
    res = Parallel(n_jobs=n_jobs)(delayed(run_fold)(pre, model, X, y, tr, va, subsample) for tr, va in folds)
    train_f1 = np.mean([r[0] for r in res])
    val = {k: np.mean([r[1][k] for r in res]) for k in res[0][1]}
    return train_f1, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="base", choices=["base", "full"])
    ap.add_argument("--emb", default="specter2_title", help="comma-separated embedding names")
    ap.add_argument("--task", default="orig", choices=list(TASK_TARGETS))
    ap.add_argument("--n-pca", type=int, default=128)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    tag = f"{args.feature_set}|{args.emb}|{args.task}"
    X_all, frame, spec = build_dataset(args.feature_set, [e for e in args.emb.split(",") if e], args.n_pca)
    y_all = frame[TASK_TARGETS[args.task]].to_numpy()
    tr_mask = (frame.split == "train").to_numpy()
    X, y = X_all[tr_mask].reset_index(drop=True), y_all[tr_mask]
    folds = cv_splits(frame)
    n_feat = make_preprocessor(spec).fit(X.head(5000), y[:5000]).transform(X.head(5)).shape[1]
    print(f"{tag}: {len(X):,} training rows, {n_feat} features after preprocessing")

    out_csv = ROOT / "reports" / "svm_cv.csv"
    best_path = ROOT / "reports" / f"svm_best_{args.feature_set}_{args.task}.json"

    if not args.final:
        rows = []

        def record(name, cfg, scaler, train_f1, val, secs):
            row = {"setup": tag, "variant": name, **cfg, "scaler": scaler, "train_f1": round(train_f1, 4),
                   **{f"cv_{k}": round(v, 4) for k, v in val.items()}, "minutes": round(secs / 60, 1)}
            rows.append(row)
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            print(f"  {name:<14} {cfg} {scaler:<8} train={train_f1:.4f} cv_macroF1={val['macro_f1']:.4f} "
                  f"({secs / 60:.1f} min)", flush=True)

        grid = [("linear", {"C": c}) for c in (0.01, 0.1, 1.0)]
        grid += [("rbf_exact", {"C": c, "gamma_mult": 1.0}) for c in (1.0, 10.0)]
        grid += [(v, {"C": c, "gamma_mult": g}) for v in ("rbf_nystroem", "rbf_fourier")
                 for c in (0.1, 1.0) for g in (0.5, 1.0)]
        grid += [("poly_nystroem", {"C": c, "gamma_mult": 1.0}) for c in (0.1, 1.0)]
        grid += [("poly_sketch", {"C": c}) for c in (0.1, 1.0)]

        pre = make_preprocessor(spec)
        for name, cfg in grid:
            t = time.time()
            sub = SUBSAMPLE if name == "rbf_exact" else None
            # PolynomialCountSketch builds large complex FFT arrays (~6 GB per worker): run it alone
            jobs = 1 if name == "poly_sketch" else 2
            train_f1, val = cross_validate(X, y, folds, pre, variant(name, n_feat, **cfg), subsample=sub,
                                           n_jobs=jobs)
            record(name, cfg, "standard", train_f1, val, time.time() - t)

        # Best full-data variant (exact RBF is compared separately because it saw only 30k rows)
        df = pd.DataFrame(rows)
        best = df[df.variant != "rbf_exact"].sort_values("cv_macro_f1", ascending=False).iloc[0]
        cfg = {k: best[k] for k in ("C", "gamma_mult") if k in best and pd.notna(best[k])}
        print(f"Best: {best.variant} {cfg}; ablating scaler and class weights", flush=True)
        for scaler in ("quantile", "power"):
            t = time.time()
            train_f1, val = cross_validate(X, y, folds, make_preprocessor(spec, scaler),
                                           variant(best.variant, n_feat, **cfg))
            record(best.variant, cfg, scaler, train_f1, val, time.time() - t)
        t = time.time()
        train_f1, val = cross_validate(X, y, folds, pre,
                                       variant(best.variant, n_feat, class_weight="balanced", **cfg))
        record(best.variant, {**cfg, "class_weight": "balanced"}, "standard", train_f1, val, time.time() - t)

        df = pd.DataFrame(rows)
        top = df[df.variant != "rbf_exact"].sort_values("cv_macro_f1", ascending=False).iloc[0]
        choice = {"variant": top.variant, "scaler": top.scaler,
                  **{k: top[k] for k in ("C", "gamma_mult", "class_weight") if k in top and pd.notna(top[k])}}
        best_path.write_text(json.dumps(choice, default=float, indent=2))
        print(f"Chosen: {choice} (cv macro F1 {top.cv_macro_f1:.4f}) -> {best_path.name}")
        return

    # Final: fit the chosen configuration on all training rows, evaluate on the test set once.
    choice = json.loads(best_path.read_text())
    cfg = {k: choice[k] for k in ("C", "gamma_mult", "class_weight") if k in choice}
    pre = make_preprocessor(spec, choice["scaler"]).fit(X, y)
    model = variant(choice["variant"], n_feat, **cfg).fit(pre.transform(X), y)
    te = ~tr_mask
    name = f"SVM {choice['variant']} ({args.feature_set})"
    log_result(name, args.task, "train", scores(y, model.predict(pre.transform(X))), notes=f"{args.emb} {choice}")
    log_result(name, args.task, "test", scores(y_all[te], model.predict(pre.transform(X_all[te]))),
               notes=f"{args.emb} {choice}")


if __name__ == "__main__":
    main()
