"""Build the tabular feature table and the abstract text file.

Inputs : clean.parquet, split.parquet, crossref_matches.parquet, openalex.parquet (enrichment is
         optional: papers without it get missing values plus an is_enriched = 0 flag)
Outputs: data/processed/features.parquet  - one row per paper, raw (unencoded) features
         data/interim/abstracts.parquet   - paper_id, abstract (input for 06_embed.py --text abstract)

Encoding (scaling, one-hot, target encoding) is NOT done here: it must be fitted on training rows
only, so it lives inside each model's sklearn pipeline (see common.make_preprocessor).
"""
import re

import numpy as np
import pandas as pd
import textstat

from common import load_frame
from config import INTERIM, PROCESSED


def title_features(t):
    return pd.DataFrame({
        "title_words": t.str.split().str.len(),
        "title_has_colon": t.str.contains(":").astype(int),
        "title_has_question": t.str.contains(r"\?").astype(int),
        "title_has_digit": t.str.contains(r"\d").astype(int),
        "title_has_acronym": t.str.contains(r"\b[A-Z]{2,}\b").astype(int),
    })


def strip_markup(s):
    if not isinstance(s, str) or not s.strip():
        return None
    s = re.sub(r"<[^>]+>", " ", s)                       # JATS / HTML tags in Crossref abstracts
    s = re.sub(r"^\s*(abstract|summary)[\s:.]*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s.split()) >= 20 else None           # very short strings are not real abstracts


def page_count(p):
    m = re.fullmatch(r"\D*(\d+)\D*[-–]\D*(\d+)\D*", str(p or ""))
    if not m:
        return np.nan
    a, b = int(m.group(1)), int(m.group(2))
    return b - a + 1 if 0 < b - a < 500 else np.nan


def main():
    df = load_frame()
    feats = df[["paper_id", "year", "journal", "title_chars", "split", "cv_fold",
                "label", "label_yn", "citations"]].copy()
    feats = pd.concat([feats, title_features(df.title)], axis=1)

    # Journal size, counted on training rows only
    journal_n = df[df.split == "train"].journal.value_counts()
    feats["journal_train_count"] = df.journal.map(journal_n).fillna(0).to_numpy()

    cr_path, oa_path = INTERIM / "crossref_matches.parquet", INTERIM / "openalex.parquet"
    cr = pd.read_parquet(cr_path) if cr_path.exists() else pd.DataFrame(columns=["paper_id", "matched", "doi"])
    cr = cr[cr.matched]
    feats = feats.merge(cr.drop(columns=["matched", "cr_title", "cr_journal"], errors="ignore"),
                        on="paper_id", how="left")
    if oa_path.exists():
        oa = pd.read_parquet(oa_path)
        oa = oa[oa.oa_id.notna()]
        feats = feats.merge(oa, on="doi", how="left")
    else:
        feats["oa_id"] = None

    feats["is_enriched"] = feats.oa_id.notna().astype(int)
    feats["has_doi"] = feats.doi.notna().astype(int)

    # Combine Crossref and OpenAlex where both describe the same thing
    oa_auth = feats.get("oa_n_authors")
    use_cr = feats.get("authors_truncated", pd.Series(False, index=feats.index)).fillna(False).astype(bool) | oa_auth.isna()
    feats["n_authors"] = np.where(use_cr, feats.cr_n_authors, oa_auth)
    feats["n_references"] = feats.oa_reference_count.fillna(feats.cr_reference_count)
    feats["n_funders"] = np.fmax(feats.n_funders.astype(float), feats.cr_n_funders.astype(float))
    feats["has_funder"] = (feats.n_funders > 0).astype(float).where(feats.n_funders.notna())
    feats["intl_collab"] = (feats.n_countries > 1).astype(float).where(feats.n_countries.notna())
    feats["n_pages"] = feats.cr_page.map(page_count)
    feats["is_oa"] = feats.is_oa.astype(float)

    # Abstract: OpenAlex first, Crossref as fallback
    abstract = feats.oa_abstract.map(strip_markup)
    abstract = abstract.fillna(feats.cr_abstract.map(strip_markup))
    feats["has_abstract"] = abstract.notna().astype(int)
    feats["abstract_words"] = abstract.str.split().str.len()
    feats["abstract_flesch"] = abstract.map(lambda a: textstat.flesch_reading_ease(a) if a else np.nan)
    feats["abstract_sentence_len"] = abstract.map(
        lambda a: textstat.avg_sentence_length(a) if a else np.nan)

    pd.DataFrame({"paper_id": feats.paper_id, "abstract": abstract}).dropna().to_parquet(
        INTERIM / "abstracts.parquet", index=False)

    drop = ["oa_abstract", "cr_abstract", "oa_reference_count", "cr_reference_count", "cr_n_authors",
            "oa_n_authors", "cr_n_funders", "cr_page", "cr_publisher", "cr_year", "cr_type", "oa_id",
            "title_score", "journal_score", "keywords", "authors_truncated", "is_paratext"]
    feats = feats.drop(columns=[c for c in drop if c in feats.columns])
    assert feats.paper_id.is_unique and len(feats) == len(df)
    feats.to_parquet(PROCESSED / "features.parquet", index=False)

    print(f"Rows: {len(feats):,}  Columns: {feats.shape[1]}")
    print(f"Enriched: {feats.is_enriched.mean():.1%}  Abstract: {feats.has_abstract.mean():.1%}  "
          f"Country: {feats.primary_country.notna().mean():.1%}")
    print(feats.drop(columns=["paper_id"]).isna().mean().round(3).sort_values().to_string())


if __name__ == "__main__":
    main()
