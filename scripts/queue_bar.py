"""One-line progress bar for the final-model queue (Stage B ensemble + run_chain4 stages)."""
import glob
import re
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
# (name, done-units function, total units, est. hours)
def n(p):
    return len(glob.glob(str(R / p)))

log = (R / "tune.log").read_text(encoding="utf-8", errors="ignore") if (R / "tune.log").exists() else ""
started = set(re.findall(r"STAGE (\w+) started", log))
fin = lambda s: 1 if s in started else 0
STEPS = [
    ("concat-ensemble", n("_tune/calm_l_ls10cat_*_f*.npz"), 30, 1.7),
    ("ablation", n("_folds/abl2_fusion_last_*.npz") + n("_folds/abl3_two_scales_*.npz"), 30, 1.6),
    ("external", 1 if fin("contam") else 0, 1, 1.2),
    ("contam", 1 if fin("xai") else 0, 1, 0.6),
    ("baselines", sum(n(f"_folds/bench_grouped_{m}_s*_f*.npz") for m in
                      ("resnet50", "densenet121", "vgg16", "xception", "inception_v3", "mobilenetv2")), 30, 1.5),
    ("xai", 1 if (R / "E9_xai.json").exists() else 0, 1, 0.3),
    ("learning", n("_folds/de_*.npz"), 24, 0.7),
    ("speed", 1 if "CHAIN COMPLETE" in log else 0, 1, 0.1),
]
tot = sum(h for *_, h in STEPS)
done = sum(h * min(1, d / t) for _, d, t, h in STEPS)
W = 30
f = int(W * done / tot)
cur = next((s for s, d, t, _ in STEPS if d < t), "done")
print(f"[{'#' * f}{'-' * (W - f)}] {100 * done / tot:5.1f}% | ~{tot - done:.1f} h left | now: {cur} | "
      + " ".join(f"{s} {min(d, t)}/{t}" for s, d, t, _ in STEPS if t > 1))
