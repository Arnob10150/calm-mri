"""Per-metric leaderboard under the patient-grouped protocol, from saved OOF predictions.

Run: python -I scripts/standings.py            (benchmark models)
     python -I scripts/standings.py --tune     (also include tuned CALM candidates)
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import metrics as MT  # noqa: E402

KEYS = ["acc", "f1", "auroc", "ece", "brier", "nll", "aurc", "acc90"]
HIGH = {"acc", "f1", "auroc", "acc90"}


def score(p, y):
    return dict(acc=MT.accuracy(p, y), f1=MT.macro_f1(p, y), auroc=MT.auroc_ovr(p, y, 3),
                ece=MT.ece(p, y), brier=MT.brier(p, y, 3), nll=MT.nll(p, y), aurc=MT.aurc(p, y),
                acc90=MT.accuracy_at_coverage(p, y, 0.9))


def collect(pattern, rx):
    rows = {}
    for f in glob.glob(str(ROOT / pattern)):
        mo = re.search(rx, Path(f).name)
        if not mo:
            continue
        z = np.load(f)
        rows.setdefault(mo.group(1), []).append(score(z["p"], z["y"]))
    return {m: {k: float(np.mean([r[k] for r in v])) for k in KEYS} | {"n": len(v)}
            for m, v in rows.items()}


def main():
    mean = collect("results/oof_grouped_*_s*.npz", r"oof_grouped_(.+)_s\d\.npz$")
    if "--tune" in sys.argv:
        mean |= collect("results/_tune/oof_*.npz", r"oof_(.+)_s\d\.npz$")
    print("%-26s" % "model" + "".join("%9s" % k for k in KEYS) + "  seeds")
    for m in sorted(mean, key=lambda m: -mean[m]["acc"]):
        print("%-26s" % m + "".join("%9.4f" % mean[m][k] for k in KEYS) + "  %d" % mean[m]["n"])
    print("\nleader per metric:")
    for k in KEYS:
        b = (max if k in HIGH else min)(mean, key=lambda m: mean[m][k])
        print(f"  {k:6s} {b:26s} {mean[b][k]:.4f}")
    return mean


if __name__ == "__main__":
    main()
