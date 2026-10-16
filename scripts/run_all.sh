#!/usr/bin/env bash
# Full CALM-MRI pipeline. Every stage resumes from its own cache, so re-running after
# an interruption picks up where it stopped.
#   bash scripts/run_all.sh            # prepare data, then every experiment
#   bash scripts/run_all.sh bench ood  # only the named stages
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE

if [ ! -f data/processed/manifest_all.csv ]; then
  echo "== preparing data"
  python -I scripts/prepare_cheng.py    || exit 1
  python -I scripts/prepare_external.py || exit 1
fi

STAGES="${*:-audit eff bench ood abl dataeff xai}"
echo "== stages: $STAGES"
python -I scripts/run_experiments.py $STAGES || exit 2

echo "== building paper assets"
python -I paper/make_assets.py || exit 3
echo "ALL DONE"
