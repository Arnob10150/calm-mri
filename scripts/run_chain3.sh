#!/usr/bin/env bash
# One proposed model: CALM-MRI on the ConvNeXt-T trunk with the searched recipe.
# Waits for the running recipe search, then adds the AJSE-paper baselines and the
# stages that use the final configuration.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while powershell -c "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ? { \$_.CommandLine -match 'tune.py' }" | grep -q .; do sleep 30; done
python -I scripts/run_experiments.py bench >> results/run_bench.log 2>&1 || exit 1
python -I scripts/run_experiments.py ood contam dataeff xai eff abl >> results/run_bench.log 2>&1 || exit 2
echo "CHAIN COMPLETE" >> results/tune.log
