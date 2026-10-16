"""Members 1 and 2 of the concatenation-head CALM-MRI ensemble.

Member 0 is the ablation run results/_folds/abl2_fusion_concat_s{s}_f{k}.npz, trained by
run_experiments.run_fold("final", fusion="concat") with the selected recipe and the same
patient-disjoint calibration split. Members 1 and 2 use the same folds, split, recipe, and
flip TTA, with initial seeds 100*s + m. The head was chosen over the gate on inner
validation (calibration-split) accuracy, never on test folds.

Run: python -I scripts/concat_members.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import data as D, train as T  # noqa: E402
import tune as TU  # noqa: E402

RES = ROOT / "results"
OUT = RES / "_tune"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REC = TU.RECIPES["ls10_ema_tta"]


def main():
    ch = D.cohort(D.load_manifest(), "cheng")
    T.load_cache(ch["path"].tolist())
    kw = {x: REC[x] for x in ("epochs", "lr", "label_smooth", "mixup", "ema")}
    for s in range(3):
        for k, (tr, te) in enumerate(D.grouped_folds(ch, 5, s)):
            for m in (1, 2):
                f = OUT / f"calm_l_ls10cat_s{s}_m{m}_f{k}.npz"
                if f.exists():
                    continue
                tr2, cal = D.split_calibration(ch, tr, frac=0.2, seed=s)
                model, _ = T.train_one("calm_l", ch.iloc[tr2], None, len(D.CLASSES),
                                       seed=100 * s + m, device=DEV, fusion="concat", **kw)
                np.savez_compressed(f, logits_cal=T.infer(model, ch.iloc[cal], DEV, tta=True),
                                    y_cal=ch.iloc[cal]["y"].to_numpy(),
                                    logits_te=T.infer(model, ch.iloc[te], DEV, tta=True),
                                    idx_te=np.asarray(te))
                del model
                torch.cuda.empty_cache()
                print(f"concat member s{s} m{m} f{k} done", flush=True)


if __name__ == "__main__":
    main()
