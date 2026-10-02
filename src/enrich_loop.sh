#!/bin/sh
# Every 30 minutes: match newly crawled Crossref results, then enrich new DOIs via OpenAlex.
# Stops once the Crossref crawl has finished and everything is enriched.
cd "$(dirname "$0")"
PY=../.venv/bin/python
while true; do
  echo "=== $(date '+%Y-%m-%d %H:%M')"
  $PY 03_crossref_match.py match | head -2
  $PY 04_openalex_enrich.py 2>&1 | grep -v "it/s\]" | tail -3
  if ! pgrep -f "03_crossref_match.py crawl" >/dev/null; then
    $PY 03_crossref_match.py match | head -2
    $PY 04_openalex_enrich.py 2>&1 | grep -v "it/s\]" | tail -3 && echo "crawl finished, final enrichment pass done" && break
  fi
  sleep 1800
done
