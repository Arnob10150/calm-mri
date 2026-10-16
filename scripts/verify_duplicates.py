"""Pixel-level check that hash-matched pairs are real duplicates.

A 64-bit difference hash on a 9x8 thumbnail could in principle match two different but
similar-looking slices. For a sample of matched pairs we therefore compare the images
themselves (Pearson correlation and mean absolute difference at 96x96) and compare
against randomly paired images from the same two cohorts as a control. Real duplicates
should sit near correlation 1; unrelated slices should not.

Run: python -I scripts/verify_duplicates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as D  # noqa: E402

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
PX = 96


def vec(rel):
    a = np.asarray(Image.open(PROC / rel).convert("L"), dtype=np.uint8)
    h, w = a.shape
    if h != w:
        s = max(h, w)
        p = np.zeros((s, s), dtype=np.uint8)
        p[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
        a = p
    return np.asarray(Image.fromarray(a).resize((PX, PX), Image.LANCZOS), dtype=np.float32)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def check(df, A, B, n=300, seed=0):
    a, b = D.cohort(df, A), D.cohort(df, B)
    ha = [D.dhash(PROC / p) for p in a["path"]]
    hb = [D.dhash(PROC / p) for p in b["path"]]
    pairs = D.near_duplicate_pairs(ha, hb, max_dist=3)
    rng = np.random.default_rng(seed)
    pick = [pairs[i] for i in rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)]
    matched = []
    for i, j, d in pick:
        x, y = vec(a.iloc[i]["path"]), vec(b.iloc[j]["path"])
        matched.append((corr(x, y), float(np.abs(x - y).mean()) / 255.0))
    ctrl = []
    for _ in range(len(pick)):
        x = vec(a.iloc[int(rng.integers(len(a)))]["path"])
        y = vec(b.iloc[int(rng.integers(len(b)))]["path"])
        ctrl.append((corr(x, y), float(np.abs(x - y).mean()) / 255.0))
    m, c = np.array(matched), np.array(ctrl)
    out = {"pair": f"{A}|{B}", "n_pairs_total": len(pairs), "n_checked": len(pick),
           "matched_corr_median": float(np.median(m[:, 0])),
           "matched_corr_p10": float(np.percentile(m[:, 0], 10)),
           "matched_share_corr_ge_0.95": float((m[:, 0] >= 0.95).mean()),
           "matched_mad_median": float(np.median(m[:, 1])),
           "control_corr_median": float(np.median(c[:, 0])),
           "control_corr_p90": float(np.percentile(c[:, 0], 90)),
           "control_share_corr_ge_0.95": float((c[:, 0] >= 0.95).mean())}
    print(f"  {A} vs {B}: {len(pairs)} pairs | matched corr median {out['matched_corr_median']:.3f} "
          f"(p10 {out['matched_corr_p10']:.3f}, {100 * out['matched_share_corr_ge_0.95']:.0f}% >=0.95) "
          f"| random-pair control median {out['control_corr_median']:.3f} "
          f"({100 * out['control_share_corr_ge_0.95']:.1f}% >=0.95)", flush=True)
    return out


def main():
    df = D.load_manifest()
    res = [check(df, "hemg", "pmram"), check(df, "hemg", "sartaj"), check(df, "pmram", "sartaj")]
    (RES / "E0b_duplicate_verification.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote results/E0b_duplicate_verification.json")


if __name__ == "__main__":
    main()
