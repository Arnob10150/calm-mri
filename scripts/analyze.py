"""Paired significance tests for the proposed model and the leakage contrast.

Model comparisons pair the fifteen (seed, fold) accuracies: every model sees the same
patient-grouped folds for a given seed, so fold k of seed s is the same set of patients
for all of them. Wilcoxon signed-rank, Holm-corrected across baselines.

The leakage contrast pairs slice-level against patient-grouped accuracy of the same
architecture and seed, over every architecture run under both protocols.

Run: python -I scripts/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OURS_KEYS = {"calm", "calm_l"}


def holm(pvals, labels):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return {labels[i]: {"p_raw": float(pvals[i]), "p_holm": float(adj[i]),
                        "significant_at_05": bool(adj[i] < 0.05)} for i in range(m)}


def fold_acc(model, proto="grouped"):
    vals = []
    for sd in range(3):
        for k in range(5):
            f = RES / "_folds" / f"bench_{proto}_{model}_s{sd}_f{k}.npz"
            if not f.exists():
                return None
            z = np.load(f)
            vals.append(float((z["logits_te"].argmax(1) == z["y_te"]).mean()))
    return np.array(vals)


def main():
    B = json.loads((RES / "E1_benchmark.json").read_text(encoding="utf-8"))
    G, S = B["protocols"]["grouped"], B["protocols"].get("slice", {})
    out = {}

    ff = RES / "final" / "folds.json"
    if ff.exists():
        ref = np.array(json.loads(ff.read_text(encoding="utf-8")))
        labels, ps, eff = [], [], {}
        for m in G:
            if m in OURS_KEYS:
                continue
            v = fold_acc(m)
            if v is None or len(v) != len(ref):
                continue
            d = ref - v
            p = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(ref, v).pvalue)
            labels.append(m)
            ps.append(p)
            eff[m] = {"mean_diff": float(d.mean()), "n_pairs": int(len(d)),
                      "wins_folds": int((d > 0).sum())}
        out["final_vs_baselines"] = {k: v | eff[k] for k, v in holm(ps, labels).items()}

    dif, labs = [], []
    for m in G:
        if m in OURS_KEYS or m not in S:
            continue
        a = np.array([s["accuracy"] for s in S[m]["per_seed"]])
        b = np.array([s["accuracy"] for s in G[m]["per_seed"]])
        n = min(len(a), len(b))
        dif.append(a[:n] - b[:n])
        labs.append(m)
    fs = RES / "E1_final_slice.json"
    if fs.exists():  # the proposed model under both protocols, when available
        FS = json.loads(fs.read_text(encoding="utf-8"))
        FG = json.loads((RES / "E1_final.json").read_text(encoding="utf-8"))
        a = np.array([s["accuracy"] for s in FS["per_seed"]])
        b = np.array([s["accuracy"] for s in FG["per_seed"]])
        dif.append(a - b)
        labs.append("final")
    if dif:
        allp = np.concatenate(dif)
        res = stats.wilcoxon(allp)
        out["leakage"] = {"per_model_mean_delta": {labs[i]: float(d.mean()) for i, d in enumerate(dif)},
                          "pooled_mean_delta": float(allp.mean()),
                          "pooled_sd": float(allp.std(ddof=1)),
                          "pooled_p_wilcoxon": float(res.pvalue),
                          "n_pairs": int(len(allp)), "n_architectures": len(dif),
                          "all_models_inflated": bool(all(d.mean() > 0 for d in dif))}

    (RES / "E8_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote results/E8_stats.json")
    for k, v in out.get("final_vs_baselines", {}).items():
        print(f"  CALM-MRI vs {k:18s} {100 * v['mean_diff']:+.2f} pts  p_holm {v['p_holm']:.4f}"
              f"  {'*' if v['significant_at_05'] else ''}")
    if "leakage" in out:
        L = out["leakage"]
        print(f"  leakage: {100 * L['pooled_mean_delta']:+.2f} pts over {L['n_architectures']} "
              f"architectures, p={L['pooled_p_wilcoxon']:.1e}")


if __name__ == "__main__":
    sys.exit(main())
