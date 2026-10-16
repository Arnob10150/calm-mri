#!/usr/bin/env bash
# Waits for the running CALM-L search, then: AJSE-paper baselines (patient-grouped),
# the paper's remaining stages, efficiency, and finally the lightweight-CALM search.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while powershell -c "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ? { \$_.CommandLine -match 'tune.py' }" | grep -q .; do sleep 30; done
python -I scripts/run_experiments.py bench >> results/run_bench.log 2>&1 || exit 1
python -I scripts/run_experiments.py ood contam abl dataeff xai eff >> results/run_bench.log 2>&1 || exit 2
python -I scripts/tune.py --base calm --members 3 >> results/tune.log 2>&1 || exit 3
echo "CHAIN COMPLETE" >> results/tune.log
