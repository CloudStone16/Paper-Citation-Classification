#!/bin/sh
# Quick status of the background pipeline. Usage: ./status.sh
cd "$(dirname "$0")"
echo "Time: $(date '+%H:%M')"
echo
echo "Running jobs:"
for p in "03_crossref_match.py crawl" "enrich_loop.sh" "run_final.sh" "notify_when_done.sh" "08_features" "06_embed" "09_svm" "10_models"; do
  pgrep -f "$p" >/dev/null && echo "  [running] $p"
done
echo
echo "Crossref crawl:"
tail -c 200 logs/crossref_crawl.log | tr '\r' '\n' | grep . | tail -1
echo
echo "Enrichment loop (last pass):"
tail -6 logs/enrich_loop.log
echo
echo "Final stage:"
[ -s logs/run_final.log ] && cat logs/run_final.log || echo "  waiting for enrichment to finish"
echo
memory_pressure | tail -1
