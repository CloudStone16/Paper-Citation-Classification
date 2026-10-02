"""Fetch publication-time metadata from OpenAlex for every matched DOI.

Looks up 100 DOIs per request, extracts features straight away (so the cache stays small) and
appends them to a JSONL cache, so the script can stop and resume (e.g. when the daily free
budget runs out). Writes data/interim/openalex.parquet.

Leakage rule: only fields that were known when the paper was published are kept. Citation counts,
percentiles, FWCI and retraction status are never requested.
"""
import json
import os
import time
from collections import Counter

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from config import INTERIM, ROOT

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()
EMAIL = os.getenv("CONTACT_EMAIL", "").strip()

API = "https://api.openalex.org/works"
SELECT = ("id,doi,type,language,open_access,authorships,is_authors_truncated,primary_topic,keywords,"
          "referenced_works_count,abstract_inverted_index,funders,countries_distinct_count,"
          "institutions_distinct_count,is_paratext")
BATCH = 100
CACHE = INTERIM / "openalex_cache.jsonl"
OUT = INTERIM / "openalex.parquet"
INSTITUTION_TYPES = ["education", "healthcare", "company", "government", "facility", "nonprofit"]


def abstract_from_index(inv):
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inv:
        return None
    words = sorted((pos, w) for w, positions in inv.items() for pos in positions)
    return " ".join(w for _, w in words)


def primary_country(authorships):
    """Most common country among authors; ties go to the first author's country."""
    counts = Counter(c for a in authorships for c in (a.get("countries") or []))
    if not counts:
        return None, None
    first = next((a["countries"][0] for a in authorships if a.get("countries")), None)
    top = max(counts.values())
    tied = [c for c, n in counts.items() if n == top]
    return (first if first in tied else sorted(tied)[0]), first


def extract(w):
    auths = w.get("authorships") or []
    country, first_country = primary_country(auths)
    inst_types = Counter(
        inst.get("type") for a in auths for inst in (a.get("institutions") or []) if inst.get("type"))
    n_inst = sum(inst_types.values())
    topic = w.get("primary_topic") or {}
    oa = w.get("open_access") or {}
    row = {
        "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
        "oa_id": w.get("id", "").rsplit("/", 1)[-1],
        "oa_type": w.get("type"),
        "language": w.get("language"),
        "is_paratext": w.get("is_paratext"),
        "is_oa": oa.get("is_oa"),
        "oa_status": oa.get("oa_status"),
        "oa_n_authors": len(auths),
        "authors_truncated": bool(w.get("is_authors_truncated")),
        "n_institutions": w.get("institutions_distinct_count"),
        "n_countries": w.get("countries_distinct_count"),
        "primary_country": country,
        "first_author_country": first_country,
        "topic": (topic.get("display_name")),
        "subfield": (topic.get("subfield") or {}).get("display_name"),
        "field": (topic.get("field") or {}).get("display_name"),
        "domain": (topic.get("domain") or {}).get("display_name"),
        "keywords": "; ".join(k["display_name"] for k in (w.get("keywords") or [])),
        "n_keywords": len(w.get("keywords") or []),
        "oa_reference_count": w.get("referenced_works_count"),
        "n_funders": len(w.get("funders") or []),
        "oa_abstract": abstract_from_index(w.get("abstract_inverted_index")),
    }
    for t in INSTITUTION_TYPES:
        row[f"inst_share_{t}"] = inst_types[t] / n_inst if n_inst else None
    return row


def fetch(session, dois):
    params = {"filter": "doi:" + "|".join("https://doi.org/" + d for d in dois),
              "per-page": BATCH, "select": SELECT}
    if API_KEY:
        params["api_key"] = API_KEY
    if EMAIL:
        params["mailto"] = EMAIL
    for attempt in range(6):
        try:
            r = session.get(API, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()["results"]
        if r.status_code == 429:
            # Either too fast, or the daily free budget is spent; the message says which.
            if "budget" in r.text.lower() or "credit" in r.text.lower() or attempt == 5:
                raise SystemExit(f"OpenAlex daily budget reached; rerun later to resume.\n{r.text[:300]}")
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        raise SystemExit(f"OpenAlex error {r.status_code}: {r.text[:300]}")
    raise SystemExit("OpenAlex unreachable; rerun later to resume.")


def main():
    matches = pd.read_parquet(INTERIM / "crossref_matches.parquet")
    dois = sorted(set(matches.loc[matches.matched, "doi"]))

    done = set()
    if CACHE.exists():
        with open(CACHE) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["queried"])
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [d for d in dois if d not in done]
    print(f"DOIs: {len(dois):,}  Cached: {len(done):,}  To fetch: {len(todo):,}  "
          f"Key: {'yes' if API_KEY else 'no (free keyless budget is small)'}")

    session = requests.Session()
    with open(CACHE, "a") as f:
        for i in tqdm(range(0, len(todo), BATCH), mininterval=5):
            chunk = todo[i:i + BATCH]
            found = {}
            for w in fetch(session, chunk):
                row = extract(w)
                found[row["doi"]] = row
            for d in chunk:  # record misses too, so they are not re-queried
                f.write(json.dumps({"queried": d, **found.get(d, {"doi": d, "oa_id": None})}) + "\n")
            f.flush()
            time.sleep(0.1)

    rows = []
    with open(CACHE) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows).drop_duplicates("queried").drop(columns="doi").rename(columns={"queried": "doi"})
    df.to_parquet(OUT, index=False)
    found = df.oa_id.notna()
    print(f"Found in OpenAlex: {found.sum():,} / {len(df):,}")
    print(f"Abstract coverage: {df.loc[found, 'oa_abstract'].notna().mean():.1%}")
    print(f"Country coverage:  {df.loc[found, 'primary_country'].notna().mean():.1%}")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
