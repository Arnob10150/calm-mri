"""Recipe search for CALM-MRI without touching the test folds.

Selection protocol (nested in spirit, cheap in practice): every candidate recipe is run
on the five patient-grouped outer folds, and each fold's 20% patient-disjoint
calibration split doubles as the inner validation set. Candidates are ranked ONLY on
inner-validation accuracy (ties: inner-validation NLL of the uncalibrated softmax).
Outer-test metrics are written to the log for transparency but never read by the
selection code. The winner is then retrained as a small seed ensemble across all three
split seeds and scored once on the outer folds.

Stage A  single model, split seed 0, one recipe per row of RECIPES
Stage B  winning recipe, MEMBERS members per split seed, split seeds 0-2

Run: python -I scripts/tune.py [--base calm_l] [--members 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import data as D, metrics as MT, train as T  # noqa: E402
import standings as ST  # noqa: E402

RES = ROOT / "results"
TUNE = RES / "_tune"
TUNE.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_CLS = len(D.CLASSES)

RECIPES = {
    "base":        dict(epochs=18, lr=3e-4, label_smooth=0.05, mixup=0.0, ema=0.0, tta=False),
    "ema_tta":     dict(epochs=18, lr=3e-4, label_smooth=0.05, mixup=0.0, ema=0.998, tta=True),
    "mix_ema_tta": dict(epochs=24, lr=3e-4, label_smooth=0.05, mixup=0.2, ema=0.998, tta=True),
    "ls10_ema_tta": dict(epochs=24, lr=3e-4, label_smooth=0.10, mixup=0.0, ema=0.998, tta=True),
    "lr2_ema_tta": dict(epochs=24, lr=2e-4, label_smooth=0.05, mixup=0.0, ema=0.998, tta=True),
}


def log(msg):
    print(msg, flush=True)


def run_member(base, rname, rec, ch, split_seed, member, k, tr, te):
    """Train one member on one outer fold; cached by tag."""
    tag = f"{base}_{rname}_s{split_seed}_m{member}_f{k}"
    f = TUNE / f"{tag}.npz"
    if f.exists():
        z = np.load(f)
        return {x: z[x] for x in z.files}
    tr2, cal = D.split_calibration(ch, tr, frac=0.2, seed=split_seed)
    init = 100 * split_seed + member
    kw = {x: rec[x] for x in ("epochs", "lr", "label_smooth", "mixup", "ema")}
    model, _ = T.train_one(base, ch.iloc[tr2], None, N_CLS, seed=init, device=DEV, **kw)
    out = {"logits_cal": T.infer(model, ch.iloc[cal], DEV, tta=rec["tta"]),
           "y_cal": ch.iloc[cal]["y"].to_numpy(),
           "logits_te": T.infer(model, ch.iloc[te], DEV, tta=rec["tta"]),
           "idx_te": np.asarray(te)}
    np.savez_compressed(f, **out)
    del model
    torch.cuda.empty_cache()
    return out


def evaluate(base, rname, rec, ch, split_seed, members):
    """OOF logits (members averaged) plus inner-validation and outer-test metrics."""
    folds = D.grouped_folds(ch, 5, split_seed)
    y = ch["y"].to_numpy()
    oof = np.zeros((len(ch), N_CLS), dtype=np.float32)
    cal_l, cal_y, inner_acc, inner_nll = [], [], [], []
    for k, (tr, te) in enumerate(folds):
        rs = [run_member(base, rname, rec, ch, split_seed, m, k, tr, te) for m in range(members)]
        lc = np.mean([r["logits_cal"] for r in rs], axis=0)
        lt = np.mean([r["logits_te"] for r in rs], axis=0)
        yc = rs[0]["y_cal"]
        pc = MT.softmax(lc)
        inner_acc.append(MT.accuracy(pc, yc))
        inner_nll.append(MT.nll(pc, yc))
        oof[rs[0]["idx_te"]] = lt
        cal_l.append(lc)
        cal_y.append(yc)
    lc_all, yc_all = np.concatenate(cal_l), np.concatenate(cal_y)
    t = MT.fit_temperature(lc_all, yc_all)
    p = MT.softmax(oof, t)
    test = ST.score(p, y)
    return {"inner_acc": float(np.mean(inner_acc)), "inner_nll": float(np.mean(inner_nll)),
            "test": test, "temperature": t}, p, y


def leaders():
    mean = ST.collect("results/oof_grouped_*_s*.npz", r"oof_grouped_(.+)_s\d\.npz$")
    best = {}
    for k in ST.KEYS:
        b = (max if k in ST.HIGH else min)(mean, key=lambda m: mean[m][k])
        best[k] = (b, mean[b][k])
    return best


def compare(name, res):
    lead = leaders()
    wins = []
    for k in ST.KEYS:
        b, v = lead[k]
        ours = res[k]
        ok = ours > v if k in ST.HIGH else ours < v
        wins.append(ok)
        log(f"    {k:6s} ours {ours:.4f} vs best baseline {b:18s} {v:.4f}  {'WIN' if ok else 'behind'}")
    log(f"BEATS {name}: {sum(wins)}/{len(wins)} metrics")
    return sum(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="calm_l")
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--recipes", default=",".join(RECIPES))
    a = ap.parse_args()

    df = D.load_manifest()
    ch = D.cohort(df, "cheng")
    T.load_cache(ch["path"].tolist())
    report = {"base": a.base, "stage_a": {}, "selection_rule":
              "max inner-validation accuracy, ties by inner-validation NLL; outer test unused"}

    # ---------------- Stage A: one model per recipe on split seed 0
    log(f"Stage A: {a.base}, split seed 0, single model")
    for rname in a.recipes.split(","):
        t0 = time.time()
        r, _, _ = evaluate(a.base, rname, RECIPES[rname], ch, 0, 1)
        report["stage_a"][rname] = r
        log(f"CANDIDATE {a.base}/{rname}: inner acc {r['inner_acc']:.4f} nll {r['inner_nll']:.4f}"
            f" | (test, not used) acc {r['test']['acc']:.4f} ece {r['test']['ece']:.4f}"
            f"  [{(time.time() - t0) / 60:.1f} min]")
        (RES / f"tune_{a.base}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    win = max(report["stage_a"], key=lambda n: (report["stage_a"][n]["inner_acc"],
                                                -report["stage_a"][n]["inner_nll"]))
    report["selected"] = win
    log(f"SELECT {a.base}/{win} (by inner validation only)")

    # ---------------- Stage B: ensemble across split seeds
    log(f"Stage B: {a.base}/{win}, {a.members} members x 3 split seeds")
    per_seed_single, per_seed_ens = [], []
    for s in range(3):
        rs, p1, y = evaluate(a.base, win, RECIPES[win], ch, s, 1)
        np.savez_compressed(TUNE / f"oof_{a.base}-{win}_s{s}.npz", p=p1, y=y)
        per_seed_single.append(rs["test"])
        re_, pe, y = evaluate(a.base, win, RECIPES[win], ch, s, a.members)
        np.savez_compressed(TUNE / f"oof_{a.base}-{win}-ens{a.members}_s{s}.npz", p=pe, y=y)
        per_seed_ens.append(re_["test"])
        log(f"  split seed {s}: single acc {rs['test']['acc']:.4f} | "
            f"ensemble acc {re_['test']['acc']:.4f} ece {re_['test']['ece']:.4f}")
    agg = lambda L: {k: float(np.mean([d[k] for d in L])) for k in ST.KEYS}  # noqa: E731
    report["final_single"] = agg(per_seed_single)
    report["final_ensemble"] = agg(per_seed_ens)
    log(f"FINAL single {a.base}/{win}: " + " ".join(f"{k} {v:.4f}" for k, v in report['final_single'].items()))
    n1 = compare(f"{a.base}/{win} (single)", report["final_single"])
    log(f"FINAL ensemble x{a.members}: " + " ".join(f"{k} {v:.4f}" for k, v in report['final_ensemble'].items()))
    n2 = compare(f"{a.base}/{win} (ensemble x{a.members})", report["final_ensemble"])
    report["wins_single"], report["wins_ensemble"] = n1, n2
    (RES / f"tune_{a.base}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
