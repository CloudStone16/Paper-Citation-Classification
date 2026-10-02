#!/bin/sh
# Every 30 minutes: match newly crawled Crossref results, then enrich new DOIs via OpenAlex.
# Exits only when the Crossref crawl has finished AND the OpenAlex pass completed successfully
# (if the daily OpenAlex budget runs out, it keeps retrying until the budget resets).
cd "$(dirname "$0")"
PY=../.venv/bin/python
while true; do
  echo "=== $(date '+%Y-%m-%d %H:%M')"
  crawl_running=0
  pgrep -f "03_crossref_match.py crawl" >/dev/null && crawl_running=1
  $PY 03_crossref_match.py match | head -2
  if $PY 04_openalex_enrich.py > ../logs/openalex_last.log 2>&1; then status=ok; else status=failed; fi
  grep -v "it/s\]" ../logs/openalex_last.log | tail -3
  echo "openalex pass: $status"
  if [ $crawl_running -eq 0 ] && [ $status = ok ]; then
    echo "crawl finished and enrichment complete"
    break
  fi
  sleep 1800
done
