"""Materialise the single proposed model ("CALM-MRI") from whichever configuration is
named in results/final_model.json, in the same form as the benchmark models.

final_model.json:
  {"source": "bench", "key": "calm_l"}                                   or
  {"source": "tune", "base": "calm_l", "recipe": "<name>", "members": 3}

Writes
  results/final/oof_s{seed}.npz   p (temperature-scaled), p_raw, y
  results/final/folds.json        fold accuracies paired with the benchmark folds
  results/E1_final.json           per-seed metrics, conformal gate, mean and SD

Run: python -I scripts/finalize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import data as D, metrics as MT  # noqa: E402
import run_experiments as RE  # noqa: E402  (score() and constants)

RES = ROOT / "results"
OUT = RES / "final"
OUT.mkdir(exist_ok=True)


def fold_files(cfg, seed, k):
    """Per-fold records (one per ensemble member) for the chosen configuration."""
    if cfg["source"] == "bench":
        return [RES / "_folds" / f"bench_grouped_{cfg['key']}_s{seed}_f{k}.npz"]
    if cfg.get("fusion") == "concat":   # member 0 from the ablation, 1..M-1 from concat_members.py
        return ([RES / "_folds" / f"abl2_fusion_concat_s{seed}_f{k}.npz"] +
                [RES / "_tune" / f"{cfg['base']}_ls10cat_s{seed}_m{m}_f{k}.npz"
                 for m in range(1, cfg.get("members", 1))])
    return [RES / "_tune" / f"{cfg['base']}_{cfg['recipe']}_s{seed}_m{m}_f{k}.npz"
            for m in range(cfg.get("members", 1))]


def main():
    cfg = json.loads((RES / "final_model.json").read_text(encoding="utf-8"))
    ch = D.cohort(D.load_manifest(), "cheng")
    y_all = ch["y"].to_numpy()
    per_seed, folds_acc = [], []
    for seed in range(3):
        oof = np.zeros((len(ch), RE.N_CLS), dtype=np.float32)
        cl, cy = [], []
        for k in range(5):
            fs = fold_files(cfg, seed, k)
            if not all(f.exists() for f in fs):
                print(f"  missing fold files for seed {seed} fold {k}; stopping")
                return
            zs = [np.load(f) for f in fs]
            lc = np.mean([z["logits_cal"] for z in zs], axis=0)
            lt = np.mean([z["logits_te"] for z in zs], axis=0)
            idx = zs[0]["idx_te"]
            oof[idx] = lt
            cl.append(lc)
            cy.append(zs[0]["y_cal"])
            folds_acc.append(float((lt.argmax(1) == y_all[idx]).mean()))
        s, p = RE.score(np.concatenate(cl), np.concatenate(cy), oof, y_all)
        per_seed.append(s)
        np.savez_compressed(OUT / f"oof_s{seed}.npz", p=p, p_raw=MT.softmax(oof), y=y_all)
    floats = [k for k in per_seed[0] if isinstance(per_seed[0][k], float)]
    rec = {"config": cfg, "per_seed": per_seed,
           "mean": {k: float(np.mean([d[k] for d in per_seed])) for k in floats},
           "sd": {k: float(np.std([d[k] for d in per_seed], ddof=1)) for k in floats}}
    (RES / "E1_final.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    (OUT / "folds.json").write_text(json.dumps(folds_acc), encoding="utf-8")
    m = rec["mean"]
    print(f"CALM-MRI final ({cfg}): acc {m['accuracy']:.4f} f1 {m['macro_f1']:.4f} "
          f"auroc {m['auroc']:.4f} ece_cal {m['ece_cal']:.4f} ece_raw {m['ece_raw']:.4f} "
          f"aurc {m['aurc']:.4f}")


if __name__ == "__main__":
    main()
