"""Predict the citation bin (<60 / 60-120 / >120) for one new paper that is not in the dataset.

Usage (from the project folder):
  .venv/bin/python src/predict_one.py                      # the default example paper below
  .venv/bin/python src/predict_one.py --title "..." --journal "..." --year 2021 [--citations 55]

What it does:
  1. Looks the paper up the same way the pipeline did: Crossref title search -> DOI (03_crossref_match),
     then OpenAlex by DOI (04_openalex_enrich), keeping only publication-time fields.
  2. Builds the feature row with the pipeline's own functions (02_clean, 08_features).
  3. Age normalisation: LMP2015 holds papers from 2007-2013 with citations counted in 2015, so what
     the model really learned from "year" is the paper's age. A paper that is N years old today is
     given the dataset year 2015 - N (a 2022 paper in 2026 is 4 years old -> 2011).
  4. Predicts with an ordinal LightGBM (the best model type) on the full tabular features. The saved
     full_orig.joblib cannot be used here: it needs the cached title/abstract embeddings to rebuild
     its PCA, and those files are no longer in data/. This model skips the embeddings; it is trained
     once (a few minutes) and saved to models/full_orig_noemb.joblib.
"""
import argparse
import importlib.util
import warnings
from datetime import date
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import requests
import textstat
from sklearn.base import clone

from common import build_dataset, cv_splits, make_preprocessor, scores
from config import BIN_LABELS, ROOT

warnings.filterwarnings("ignore")

SNAPSHOT_YEAR = 2015          # year LMP2015 counted its citations
DATA_YEARS = (2007, 2013)     # publication years in the dataset
MODEL_PATH = ROOT / "models" / "full_orig_noemb.joblib"

EXAMPLE = dict(
    title="Efficient Hybrid Blind Watermarking in DWT-DCT-SVD with Dual Biometric Features for Images",
    journal="Contrast Media & Molecular Imaging", year=2022, citations=40)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CLEAN, CROSSREF, OPENALEX, FEATS, MODELS = (load(n) for n in (
    "02_clean", "03_crossref_match", "04_openalex_enrich", "08_features", "10_models"))


# ---------------------------------------------------------------------------------------------
# Model (ordinal LightGBM, no embeddings)
# ---------------------------------------------------------------------------------------------
def train_model():
    print("Training ordinal LightGBM on the full tabular features (one-off, a few minutes)...")
    X_all, frame, spec = build_dataset("full", [])
    tr = (frame.split == "train").to_numpy()
    X, y = X_all[tr].reset_index(drop=True), frame.label.to_numpy()[tr]
    r = np.log1p(frame.citations.to_numpy()[tr])
    base = make_preprocessor(spec)

    # Out-of-fold scores on the fixed CV folds, used to tune the two bin thresholds
    oof = np.zeros(len(X))
    for k, (a, b) in enumerate(cv_splits(frame)):
        pre = clone(base).fit(X.iloc[a], y[a])
        oof[b] = MODELS.fit_lgbm(pre.transform(X.iloc[a]), r[a], regression=True).predict(pre.transform(X.iloc[b]))
        print(f"  fold {k} done", flush=True)
    thresholds = MODELS.tune_thresholds(oof, y)

    pre = clone(base).fit(X, y)
    model = MODELS.fit_lgbm(pre.transform(X), r, regression=True)
    test_pred = MODELS.cut(model.predict(pre.transform(X_all[~tr])), thresholds)
    bundle = {"pre": pre, "ordinal": model, "thresholds": thresholds, "spec": spec,
              "cv_macro_f1": scores(y, MODELS.cut(oof, thresholds))["macro_f1"],
              "test_macro_f1": scores(frame.label.to_numpy()[~tr], test_pred)["macro_f1"]}
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Saved {MODEL_PATH.name}  CV macro F1 {bundle['cv_macro_f1']:.4f}  "
          f"test macro F1 {bundle['test_macro_f1']:.4f}")
    return bundle


# ---------------------------------------------------------------------------------------------
# Metadata lookup (same sources and matching rule as the pipeline)
# ---------------------------------------------------------------------------------------------
def crossref_lookup(title, journal, year):
    """Crossref title search with 03_crossref_match's rule; returns the cr_* fields and the DOI."""
    row = SimpleNamespace(title=title, year=year, paper_id=0)
    res = CROSSREF.query(CROSSREF.make_session(), CROSSREF.RateLimiter(2), row)
    best = CROSSREF.best_candidate(title, journal, (res or {}).get("items", []))
    if best is None or not (best["title_score"] >= CROSSREF.TITLE_MIN and (
            best["journal_score"] >= CROSSREF.JOURNAL_MIN or best["title_score"] >= CROSSREF.TITLE_STRICT)):
        return {"doi": None}
    it = best["item"]
    return {"doi": it.get("DOI", "").lower(), "cr_reference_count": it.get("references-count"),
            "cr_n_authors": len(it.get("author") or []), "cr_page": it.get("page"),
            "cr_n_funders": len(it.get("funder") or []), "cr_abstract": it.get("abstract")}


def openalex_lookup(doi):
    if not doi:
        return {}
    found = OPENALEX.fetch(requests.Session(), [doi])
    return OPENALEX.extract(found[0]) if found else {}


# ---------------------------------------------------------------------------------------------
# Feature row (mirrors 08_features.py for a single paper)
# ---------------------------------------------------------------------------------------------
def build_row(title, journal, data_year, cr, oa, spec):
    feats = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet",
                            columns=["split", "journal", "primary_country"])
    train = feats[feats.split == "train"]

    row = {"year": data_year, "year_cat": str(data_year), "journal": journal, "title_chars": len(title)}
    row.update(FEATS.title_features(pd.Series([title])).iloc[0].to_dict())
    row["journal_train_count"] = int((train.journal == journal).sum())
    row.update({k: v for k, v in oa.items() if k in spec["numeric"] + spec["onehot"] + spec["target_enc"]})

    row["is_enriched"] = int(bool(oa.get("oa_id")))
    row["has_doi"] = int(bool(cr.get("doi")))
    use_cr = oa.get("authors_truncated") or oa.get("oa_n_authors") is None
    row["n_authors"] = cr.get("cr_n_authors") if use_cr else oa.get("oa_n_authors")
    row["n_references"] = oa.get("oa_reference_count", cr.get("cr_reference_count"))
    row["n_funders"] = np.fmax(float(oa.get("n_funders", np.nan)), float(cr.get("cr_n_funders", np.nan)))
    row["has_funder"] = float(row["n_funders"] > 0) if not np.isnan(row["n_funders"]) else np.nan
    n_countries = oa.get("n_countries")
    row["intl_collab"] = float(n_countries > 1) if n_countries is not None else np.nan
    row["n_pages"] = FEATS.page_count(cr.get("cr_page"))
    row["is_oa"] = float(oa["is_oa"]) if oa.get("is_oa") is not None else np.nan

    abstract = FEATS.strip_markup(oa.get("oa_abstract")) or FEATS.strip_markup(cr.get("cr_abstract"))
    row["has_abstract"] = int(abstract is not None)
    row["abstract_words"] = len(abstract.split()) if abstract else np.nan
    row["abstract_flesch"] = textstat.flesch_reading_ease(abstract) if abstract else np.nan
    row["abstract_sentence_len"] = textstat.words_per_sentence(abstract) if abstract else np.nan

    top = train.primary_country.value_counts().head(30).index
    country = oa.get("primary_country")
    row["country_top"] = "UNKNOWN" if country is None else (country if country in top else "OTHER")

    X = pd.DataFrame([row]).reindex(columns=spec["numeric"] + spec["onehot"] + spec["target_enc"])
    for c in spec["onehot"] + spec["target_enc"]:
        X[c] = X[c].fillna("UNKNOWN").astype(str)
    X[spec["numeric"]] = X[spec["numeric"]].astype(float)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default=EXAMPLE["title"])
    ap.add_argument("--journal", default=EXAMPLE["journal"])
    ap.add_argument("--year", type=int, default=EXAMPLE["year"], help="real publication year")
    ap.add_argument("--citations", type=int, default=EXAMPLE["citations"],
                    help="current citation count, only to compare with the prediction")
    ap.add_argument("--today", type=int, default=date.today().year)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    if args.title != EXAMPLE["title"] and args.citations == EXAMPLE["citations"]:
        args.citations = None

    bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() and not args.retrain else train_model()

    title, journal = CLEAN.clean_text(args.title), CLEAN.normalize_journal(args.journal)
    age = args.today - args.year
    data_year = SNAPSHOT_YEAR - age
    clipped = int(np.clip(data_year, *DATA_YEARS))

    print("\nLooking the paper up on Crossref and OpenAlex...")
    cr = crossref_lookup(title, journal, args.year)
    oa = openalex_lookup(cr["doi"])
    X = build_row(title, journal, clipped, cr, oa, bundle["spec"])

    score = bundle["ordinal"].predict(bundle["pre"].transform(X))[0]
    lo, hi = bundle["thresholds"]
    pred = int(MODELS.cut(np.array([score]), bundle["thresholds"])[0])

    print(f"\n{title}\n  {args.journal}, {args.year}  DOI: {cr['doi'] or 'not found'}")
    print(f"\nAge normalisation: {args.year} -> {age} years old in {args.today} -> "
          f"dataset year {SNAPSHOT_YEAR} - {age} = {data_year}"
          + (f" (outside {DATA_YEARS[0]}-{DATA_YEARS[1]}, clipped to {clipped})" if clipped != data_year else ""))
    print("\nFeatures given to the model:")
    shown = X.iloc[0]
    print(shown[shown.notna()].to_string())
    print(f"\nModel score (predicted log(1 + citations)): {score:.3f}  ->  about {np.expm1(score):.0f} citations")
    print(f"Bin edges on that score: {lo:.3f} (~{np.expm1(lo):.0f} cites) and {hi:.3f} (~{np.expm1(hi):.0f} cites)")
    print(f"\nPREDICTED BIN: {BIN_LABELS[pred]}")
    if args.citations is not None:
        actual = int(args.citations >= 60) + int(args.citations > 120)
        print(f"Actual: {args.citations} citations -> {BIN_LABELS[actual]} "
              f"({'correct' if actual == pred else 'wrong'})")
    print(f"\n(model: ordinal LightGBM, no embeddings; CV macro F1 {bundle['cv_macro_f1']:.4f}, "
          f"test macro F1 {bundle['test_macro_f1']:.4f})")


if __name__ == "__main__":
    main()
