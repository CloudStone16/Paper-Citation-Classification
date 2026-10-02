"""Find each paper's DOI on Crossref by searching its title.

Two phases, so the matching rule can be tuned without re-downloading:
  1. crawl  - query Crossref for every paper, append raw candidates to a JSONL cache (resumable)
  2. match  - pick the best candidate per paper and write data/interim/crossref_matches.parquet

Usage:
  python 03_crossref_match.py crawl [--limit N]
  python 03_crossref_match.py match
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import re
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz
from tqdm import tqdm

from config import INTERIM, ROOT

load_dotenv(ROOT / ".env")
EMAIL = os.getenv("CONTACT_EMAIL", "").strip()

API = "https://api.crossref.org/works"
FIELDS = "DOI,title,subtitle,abstract,container-title,short-container-title,issued,references-count,author,page,type,publisher,funder,score"
CACHE = INTERIM / "crossref_cache.jsonl"
OUT = INTERIM / "crossref_matches.parquet"

# Crossref etiquette: identified ("polite") clients get a higher rate limit than anonymous ones.
RATE_PER_SEC = 9 if EMAIL else 4
WORKERS = 3 if EMAIL else 1

TITLE_MIN = 90      # title similarity needed to accept a match
JOURNAL_MIN = 70    # journal similarity needed unless the title match is near exact
TITLE_STRICT = 97


class RateLimiter:
    def __init__(self, per_sec):
        self.interval = 1.0 / per_sec
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                time.sleep(self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval


def make_session():
    s = requests.Session()
    agent = "citation-bin-research/0.1"
    s.headers["User-Agent"] = f"{agent} (mailto:{EMAIL})" if EMAIL else agent
    return s


def query(session, limiter, row):
    params = {
        "query.bibliographic": row.title,
        "filter": f"from-pub-date:{row.year - 1},until-pub-date:{row.year + 1}",
        "rows": 5,
        "select": FIELDS,
    }
    if EMAIL:
        params["mailto"] = EMAIL
    for attempt in range(6):
        limiter.wait()
        try:
            r = session.get(API, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return {"paper_id": int(row.paper_id), "items": r.json()["message"]["items"]}
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
            continue
        return {"paper_id": int(row.paper_id), "items": [], "error": r.status_code}
    return None  # gave up; will be retried on the next run


def crawl(limit=None):
    df = pd.read_parquet(INTERIM / "clean.parquet", columns=["paper_id", "title", "year"])
    done = set()
    if CACHE.exists():
        with open(CACHE) as f:
            recs = [json.loads(line) for line in f]
        done = {r["paper_id"] for r in recs if "error" not in r}
    todo = df[~df.paper_id.isin(done)]
    # Random order: if the crawl is stopped early, the cached part is still a fair sample of all years.
    todo = todo.sample(frac=1, random_state=0)
    if limit:
        todo = todo.head(limit)
    print(f"Cached: {len(done):,}  To fetch: {len(todo):,}  "
          f"Mode: {'polite' if EMAIL else 'anonymous'} ({RATE_PER_SEC}/s, {WORKERS} workers)")

    limiter, lock = RateLimiter(RATE_PER_SEC), threading.Lock()
    sessions = threading.local()

    def task(row):
        if not hasattr(sessions, "s"):
            sessions.s = make_session()
        return query(sessions.s, limiter, row)

    with open(CACHE, "a") as f, ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(task, row) for row in todo.itertuples()]
        for fut in tqdm(as_completed(futures), total=len(futures), mininterval=5):
            res = fut.result()
            if res is not None:
                with lock:
                    f.write(json.dumps(res) + "\n")
                    f.flush()


def norm(s):
    s = re.sub(r"<[^>]+>", " ", s)  # Crossref titles may contain HTML / MathML markup
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in s).split())


def title_similarity(ours, cand_title, cand_subtitle):
    """Best of full-title match and match with the subtitle appended.

    Crossref often stores 'Main title' and ': subtitle' separately. If one title is a prefix of the
    other (a long one, at least 6 words), it is treated as a near match too.
    """
    a, b = norm(ours), norm(cand_title)
    full = norm(f"{cand_title} {cand_subtitle}")
    score = max(fuzz.token_sort_ratio(a, b), fuzz.token_sort_ratio(a, full))
    shorter = min(a, b, key=len)
    if len(shorter.split()) >= 6 and (a.startswith(b) or b.startswith(a)):
        score = max(score, 95.0)
    return score


def best_candidate(title, journal, items):
    best = None
    for it in items:
        cand_title = " ".join(it.get("title") or [])
        if not cand_title or cand_title.startswith("Faculty Opinions recommendation"):
            continue
        t_score = title_similarity(title, cand_title, " ".join(it.get("subtitle") or []))
        journals = (it.get("container-title") or []) + (it.get("short-container-title") or [])
        j_score = max((fuzz.token_set_ratio(journal, norm(j)) for j in journals), default=0)
        if best is None or (t_score, j_score) > (best["title_score"], best["journal_score"]):
            best = {"item": it, "title_score": t_score, "journal_score": j_score}
    return best


def match():
    df = pd.read_parquet(INTERIM / "clean.parquet", columns=["paper_id", "title", "journal", "year"])
    cache = {}
    with open(CACHE) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # last line may be half-written while the crawl runs
                continue
            if "error" not in rec:
                cache[rec["paper_id"]] = rec["items"]

    rows = []
    for r in df[df.paper_id.isin(cache)].itertuples():
        best = best_candidate(r.title, r.journal, cache[r.paper_id])
        if best is None:
            continue
        ok = best["title_score"] >= TITLE_MIN and (
            best["journal_score"] >= JOURNAL_MIN or best["title_score"] >= TITLE_STRICT)
        it = best["item"]
        issued = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        rows.append({
            "paper_id": r.paper_id,
            "matched": ok,
            "title_score": best["title_score"],
            "journal_score": best["journal_score"],
            "doi": it.get("DOI", "").lower() if ok else None,
            "cr_title": " ".join(it.get("title") or []),
            "cr_journal": " ".join(it.get("container-title") or []),
            "cr_year": issued,
            "cr_type": it.get("type"),
            "cr_reference_count": it.get("references-count"),
            "cr_n_authors": len(it.get("author") or []),
            "cr_page": it.get("page"),
            "cr_publisher": it.get("publisher"),
            "cr_n_funders": len(it.get("funder") or []),
            "cr_abstract": it.get("abstract"),  # JATS XML, cleaned later
        })
    out = pd.DataFrame(rows)
    # Two papers must not claim the same DOI; keep the stronger match.
    out = out.sort_values(["title_score", "journal_score"], ascending=False)
    dup = out.matched & out.doi.duplicated()
    out.loc[dup, ["matched", "doi"]] = [False, None]
    out.sort_values("paper_id").to_parquet(OUT, index=False)

    print(f"Queried: {len(cache):,} / {len(df):,}")
    print(f"Matched: {out.matched.sum():,} ({out.matched.mean():.1%} of queried)")
    print(f"Dropped as duplicate DOI claims: {int(dup.sum())}")
    print("Title score of best candidate (quantiles):",
          out.title_score.quantile([.05, .1, .25, .5]).round(1).to_dict())
    print(f"Saved {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["crawl", "match"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    crawl(args.limit) if args.phase == "crawl" else match()
