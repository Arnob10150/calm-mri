#!/usr/bin/env bash
# Finish the seed-2 ensemble without sharing the GPU: before every attempt, wait until no
# other GPU training job (train_det.py) is running; retry until all files exist.
cd "$(dirname "$0")/.."
busy() { powershell -c "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ? { \$_.CommandLine -match 'train_det' }" | grep -q .; }
for i in $(seq 1 40); do
  n=$(ls results/_tune | grep -c "calm_l_ls10_ema_tta_s2_m")
  [ "$n" -ge 15 ] && break
  if busy; then echo "WAITING for GPU (other job running) $(date +%H:%M)" >> results/tune.log; while busy; do sleep 60; done; fi
  echo "RETRY tune attempt $i ($n/15 seed-2 files)" >> results/tune.log
  python -I scripts/tune.py --base calm_l --members 3 >> results/tune.log 2>&1
done
[ "$(ls results/_tune | grep -c calm_l_ls10_ema_tta_s2_m)" -ge 15 ] || { echo "STAGE tune FAILED" >> results/tune.log; exit 1; }
bash scripts/run_chain4.sh
