"""Live progress of the CALM-MRI experiment queue.

Work is counted in folds, weighted by each architecture's measured cost relative to
CALM-MRI, so the bar tracks compute rather than fold count.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CK = RES / "_folds"

# cost per fold relative to CALM, from the measured throughput in E6_efficiency.json
COST = {"calm": 1.0, "calm_l": 2.6, "efficientnet_b0": 1.0, "resnet18": 0.85,
        "mobilenetv3_small": 0.7, "convnext_tiny": 2.4, "deit_small": 2.2, "swin_tiny": 2.9,
        "resnet50": 1.5, "densenet121": 1.8, "vgg16": 4.0, "xception": 2.0,
        "inception_v3": 1.6, "mobilenetv2": 0.8}
GROUPED_ONLY = {"calm_l", "resnet50", "densenet121", "vgg16", "xception", "inception_v3",
                "mobilenetv2"}
MODELS = list(COST)
SEEDS, FOLDS = 3, 5


def planned():
    """(tag-prefix, weight) for every fold the queue will run."""
    out = {}
    for proto in ("grouped", "slice"):
        for m in MODELS:
            if proto == "slice" and m in GROUPED_ONLY:
                continue
            for s in range(SEEDS):
                for f in range(FOLDS):
                    out[f"bench_{proto}_{m}_s{s}_f{f}"] = COST[m]
    for v in ("full", "fusion_concat", "fusion_last", "no_pretrain",
              "trunk_mobilenet", "two_scales"):
        for s in range(SEEDS):
            for f in range(FOLDS):
                out[f"abl_{v}_s{s}_f{f}"] = COST["calm"]
    for m in ("calm", "efficientnet_b0"):
        for fr in (10, 25, 50, 100):
            for s in range(SEEDS):
                for k in range(3):
                    out[f"de_{m}_f{fr}_s{s}_k{k}"] = COST[m] * (0.3 + 0.7 * fr / 100)
    return out


def bar(frac, width=28):
    f = int(round(frac * width))
    return "[" + "#" * f + "-" * (width - f) + f"] {100 * frac:5.1f}%"


def main():
    plan = planned()
    done = {p.stem for p in CK.glob("*.npz")}
    tot = sum(plan.values())
    got = sum(w for k, w in plan.items() if k in done)
    # stages that do not use the fold cache
    extras = {"audit": (RES / "E0_audit.json").exists(),
              "eff": (RES / "E6_efficiency.json").exists(),
              "ood": (RES / "E4_ood.json").exists(),
              "contam": (RES / "E4b_contamination.json").exists(),
              "xai": (RES / "E9_xai.json").exists()}
    parts = []
    for name, pre in (("bench", "bench_"), ("abl", "abl_"), ("dataeff", "de_")):
        sub = {k: w for k, w in plan.items() if k.startswith(pre)}
        d = sum(w for k, w in sub.items() if k in done)
        parts.append(f"{name} {100 * d / sum(sub.values()):.0f}%")
    extra_txt = " ".join(f"{k}{'OK' if v else '..'}" for k, v in extras.items())
    print(f"OVERALL {bar(got / tot)} | " + " | ".join(parts) + " | " + extra_txt, flush=True)


if __name__ == "__main__":
    main()
