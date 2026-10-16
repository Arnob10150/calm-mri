#!/usr/bin/env bash
# Remaining AJSE stages after the final model. Resumable: every fold is cached on disk.
cd "$(dirname "$0")/.."
step() { for t in 1 2 3 4; do echo "STAGE $1 started $(date +%H:%M) (try $t)" >> results/tune.log
  python -I scripts/run_experiments.py "$1" >> results/run_bench.log 2>&1 && return 0
  echo "STAGE $1 FAILED (try $t)" >> results/tune.log; sleep 60; done; return 1; }
for st in abl bench ood contam xai dataeff eff; do step $st; [ "$st" = bench ] && python -I scripts/analyze.py >> results/tune.log 2>&1; done
python -I scripts/analyze.py >> results/tune.log 2>&1
echo "CHAIN COMPLETE $(date +%H:%M)" >> results/tune.log
