#!/usr/bin/env bash
# Retry the recipe search until every seed-2 ensemble file exists (the process has been
# killed by the OS under memory pressure), then run the final-model queue.
cd "$(dirname "$0")/.."
for i in 1 2 3 4 5 6 7 8 9 10; do
  n=$(ls results/_tune | grep -c "calm_l_ls10_ema_tta_s2_m")
  [ "$n" -ge 15 ] && break
  echo "RETRY tune attempt $i ($n/15 seed-2 files)" >> results/tune.log
  python -I scripts/tune.py --base calm_l --members 3 >> results/tune.log 2>&1
done
n=$(ls results/_tune | grep -c "calm_l_ls10_ema_tta_s2_m")
[ "$n" -ge 15 ] || { echo "STAGE tune FAILED after retries" >> results/tune.log; exit 1; }
bash scripts/run_chain4.sh
