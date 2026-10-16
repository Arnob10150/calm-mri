#!/usr/bin/env bash
# Concatenation-head CALM-MRI: train ensemble members 1-2, materialise the final model,
# then every downstream experiment. A failed step is logged and retried once.
cd "$(dirname "$0")/.."
step() { echo "STAGE $1 started $(date +%H:%M)" >> results/tune.log
  for t in 1 2; do "${@:2}" >> results/run_bench.log 2>&1 && return 0; echo "STAGE $1 FAILED (try $t)" >> results/tune.log; done; }
step concat python -I scripts/concat_members.py
step finalize python -I scripts/finalize.py
python -I scripts/finalize.py >> results/tune.log 2>&1
python -I scripts/analyze.py >> results/tune.log 2>&1
for st in abl bench ood contam xai dataeff eff; do
  step $st python -I scripts/run_experiments.py $st
  [ "$st" = bench ] && python -I scripts/analyze.py >> results/tune.log 2>&1
done
echo "CHAIN COMPLETE $(date +%H:%M)" >> results/tune.log
