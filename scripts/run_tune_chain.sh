#!/usr/bin/env bash
# Benchmark -> recipe search (CALM-L, then lightweight CALM) -> remaining stages.
# Every step resumes from its own cache, so re-running after a crash is safe.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -I scripts/run_experiments.py bench >> results/run_bench.log 2>&1 || exit 1
python -I scripts/tune.py --base calm_l --members 3 >> results/tune.log 2>&1 || exit 2
python -I scripts/tune.py --base calm --members 3 >> results/tune.log 2>&1 || exit 3
python -I scripts/run_experiments.py ood contam abl dataeff xai >> results/run_bench.log 2>&1 || exit 4
echo "CHAIN COMPLETE" >> results/tune.log
