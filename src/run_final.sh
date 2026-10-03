#!/bin/sh
# Final stage, run once enrichment is complete. One heavy job at a time (24 GB machine).
# 1. wait for the crawl + enrichment loop to finish
# 2. features -> abstract embeddings -> SVM comparison (full) -> final test evaluation -> report -> push
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
LOG=../logs
EMB=qwen3-0.6b_title,specter2_abstract

while pgrep -f "enrich_loop.sh" >/dev/null || pgrep -f "03_crossref_match.py crawl" >/dev/null; do
  sleep 300
done
echo "=== enrichment finished $(date '+%H:%M')"

$PY 08_features.py > $LOG/features_final.log 2>&1
$PY 06_embed.py --model specter2 --text abstract --batch-size 32 > $LOG/embed_abstract_final.log 2>&1
echo "=== features + abstract embeddings done $(date '+%H:%M')"

for task in orig; do
  $PY 09_svm.py --feature-set full --emb $EMB --task $task > $LOG/svm_full_$task.log 2>&1
  echo "=== svm full $task done $(date '+%H:%M')"
done

# Final test-set evaluation (the only place the test set is used)
for task in orig; do
  $PY 10_models.py --feature-set base --emb qwen3-0.6b_title --task $task --final > $LOG/final_base_$task.log 2>&1
  echo "=== final base $task done $(date '+%H:%M')"
  $PY 10_models.py --feature-set full --emb $EMB --task $task --final > $LOG/final_full_$task.log 2>&1
  echo "=== final full $task done $(date '+%H:%M')"
done

$PY 11_report.py > $LOG/report.log 2>&1
cd ..
git add -A
git commit -q -m "Final evaluation: full-feature SVM comparison and held-out test results for base and full features"
git push -q
echo "=== all done $(date '+%H:%M')"
