#!/usr/bin/env bash
# Concat-head CALM-MRI queue. Per the user's instruction, any YOLO run (train_det.py) is
# stopped before each step so the GPU is not shared. The concat ensemble must be complete
# before the final model is materialised.
cd "$(dirname "$0")/.."
clear_gpu() { powershell -c "Get-CimInstance Win32_Process | ? { \$_.CommandLine -match 'train_det' -and \$_.CommandLine -notmatch 'Get-CimInstance' } | % { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue; 'stopped YOLO ' + \$_.ProcessId }" >> results/tune.log; }
step() { for t in 1 2 3; do clear_gpu; echo "STAGE $1 started $(date +%H:%M) (try $t)" >> results/tune.log
  "${@:2}" >> results/run_bench.log 2>&1 && return 0; echo "STAGE $1 FAILED (try $t)" >> results/tune.log; sleep 20; done; return 1; }
for t in $(seq 1 8); do
  [ "$(ls results/_tune | grep -c calm_l_ls10cat_)" -ge 30 ] && break
  step concat python -I scripts/concat_members.py
done
[ "$(ls results/_tune | grep -c calm_l_ls10cat_)" -ge 30 ] || { echo "STAGE concat GAVE UP" >> results/tune.log; exit 1; }
python -I scripts/finalize.py >> results/tune.log 2>&1
python -I scripts/analyze.py >> results/tune.log 2>&1
for st in abl bench ood contam xai dataeff eff; do
  step $st python -I scripts/run_experiments.py $st
  [ "$st" = bench ] && python -I scripts/analyze.py >> results/tune.log 2>&1
done
echo "CHAIN COMPLETE $(date +%H:%M)" >> results/tune.log
