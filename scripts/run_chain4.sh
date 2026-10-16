#!/usr/bin/env bash
# Final-model queue. Waits for the recipe search, materialises CALM-MRI (finalize.py) and
# its significance tests, then runs the remaining experiments in order of importance to
# the paper; a failed stage is logged and the queue moves on.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while powershell -c "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ? { \$_.CommandLine -match 'tune.py' }" | grep -q .; do sleep 30; done
python -I scripts/finalize.py >> results/tune.log 2>&1 || echo "STAGE finalize FAILED" >> results/tune.log
python -I scripts/analyze.py >> results/tune.log 2>&1 || echo "STAGE analyze FAILED" >> results/tune.log
for st in abl ood contam bench xai dataeff eff; do
  echo "STAGE $st started $(date +%H:%M)" >> results/tune.log
  python -I scripts/run_experiments.py "$st" >> results/run_bench.log 2>&1 || echo "STAGE $st FAILED" >> results/tune.log
  if [ "$st" = bench ]; then python -I scripts/analyze.py >> results/tune.log 2>&1; fi
done
echo "CHAIN COMPLETE $(date +%H:%M)" >> results/tune.log
