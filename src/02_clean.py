"""Clean the raw dataset and attach citation-bin labels.

Output: data/interim/clean.parquet and reports/cleaning_report.md
"""
import html
import re

import ftfy
import pandas as pd

from config import BIN_EDGES, BIN_LABELS, INTERIM, RAW_CSV, ROOT

# Titles that are placeholders rather than real paper titles
PLACEHOLDER_TITLES = {"article", "editorial", "untitled", "introduction", "preface",
                      "foreword", "erratum", "corrigendum", "index", "contents"}


def clean_text(s):
    s = ftfy.fix_text(str(s))
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)          # HTML tags such as <i>, <sup>
    s = re.sub(r"\$([^$]*)\$", r"\1", s)    # inline LaTeX $...$
    return re.sub(r"\s+", " ", s).strip()


def normalize_key(s):
    """Lowercase alphanumeric form, used for duplicate detection and grouping."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def normalize_journal(s):
    s = clean_text(s).lower().replace("&", " and ")
    s = re.sub(r"^the\s+", "", s)
    s = re.sub(r"\s*:.*$", "", s)           # drop subtitles, e.g. "heart rhythm : the official journal ..."
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main():
    raw = pd.read_csv(RAW_CSV).drop(columns="Unnamed: 0")
    log = [("Raw rows", len(raw))]

    df = raw.rename(columns={"cites": "citations", "journal": "journal_raw"})
    df.insert(0, "paper_id", df.index)
    df["title"] = df["title"].map(clean_text)
    df["journal"] = df["journal_raw"].map(normalize_journal)
    df["title_key"] = df["title"].map(normalize_key)

    mask = (df["title"].str.len() > 0) & df["citations"].ge(0)
    log.append(("Dropped: empty title or invalid citations", int((~mask).sum())))
    df = df[mask]

    mask = ~df["title_key"].isin(PLACEHOLDER_TITLES)
    log.append(("Dropped: placeholder titles (e.g. 'Article')", int((~mask).sum())))
    df = df[mask]

    before = len(df)
    df = df.drop_duplicates(["title_key", "year", "journal"])
    log.append(("Dropped: exact duplicates (title, year, journal)", before - len(df)))

    # Same title in the same year (e.g. guidelines co-published in several journals) stays,
    # but shares a group id so the train/test split can keep the copies together.
    df["dup_group"] = df.groupby(["title_key", "year"]).ngroup()
    n_shared = int(df.duplicated("dup_group", keep=False).sum())
    log.append(("Kept, grouped: same title + year in different journals", n_shared))

    # The original title_length disagrees on non-ASCII characters (e.g. Greek letters); recompute.
    df["title_chars"] = df["title"].str.len()
    df = df.drop(columns="title_length")

    low, high = BIN_EDGES
    df["label"] = (df["citations"] >= low).astype(int) + (df["citations"] > high).astype(int)  # <60 / 60-120 / >120
    df["label_name"] = df["label"].map(dict(enumerate(BIN_LABELS)))
    log.append(("Final rows", len(df)))

    out = INTERIM / "clean.parquet"
    df.reset_index(drop=True).to_parquet(out, index=False)

    by_year = pd.crosstab(df["year"], df["label_name"], normalize="index")[BIN_LABELS].round(3)
    lines = ["# Cleaning report", "", "| Step | Rows |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in log]
    lines += ["", "## Label balance", "",
              df["label_name"].value_counts(normalize=True).reindex(BIN_LABELS).round(3).to_markdown(),
              "", "## Label balance by year", "", by_year.to_markdown(),
              "", f"Unique journals after normalization: {df['journal'].nunique():,} "
              f"(raw: {raw['journal'].nunique():,})"]
    (ROOT / "reports" / "cleaning_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
