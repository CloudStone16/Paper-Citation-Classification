#!/bin/sh
# Waits for the final stage to finish (or stop early) and shows a macOS notification.
cd "$(dirname "$0")"
notify() { osascript -e "display notification \"$1\" with title \"Citation pipeline\" sound name \"Glass\""; }
started=0
while true; do
  if grep -q "all done" logs/run_final.log 2>/dev/null; then
    notify "All done: results pushed to GitHub (see reports/results.md)"; exit 0
  fi
  if grep -q "=== enrichment finished" logs/run_final.log 2>/dev/null && [ $started -eq 0 ]; then
    started=1; notify "Enrichment finished, final stage started"
  fi
  if ! pgrep -f run_final.sh >/dev/null; then
    notify "Final stage STOPPED before finishing: check logs/run_final.log"; exit 1
  fi
  sleep 60
done
