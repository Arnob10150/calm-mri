"""Cohorts, splits, and the duplicate audit.

Three cohorts, all reduced to the label space they share (glioma, meningioma,
pituitary) so that cross-cohort transfer measures domain shift and not a change in
the label set:

  cheng   figshare 10.6084/m9.figshare.1512427 (CC BY 4.0) -- the only one of the
          three that ships a patient identifier, so it is the primary cohort and the
          only one on which patient-grouped evaluation is possible.
  sartaj  HuggingFace sartajbhuvaji/Brain-Tumor-Classification (MIT).
  hemg    HuggingFace Hemg/Brain-Tumor-MRI-Dataset.

Splits: GroupKFold over patient_id (grouped) versus StratifiedKFold over slices
(slice-level). The gap between the two is the leakage estimate reported in the paper.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CLASSES = ["glioma", "meningioma", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Folder spellings differ between cohorts; everything outside this map is dropped
# (notably the "no tumour" class, which cheng does not contain).
NAME_MAP = {
    "glioma": "glioma", "glioma_tumor": "glioma", "gliomatumor": "glioma",
    "meningioma": "meningioma", "meningioma_tumor": "meningioma", "meningiomatumor": "meningioma",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary", "pituitarytumor": "pituitary",
}


def norm_label(s: str):
    k = str(s).strip().lower().replace(" ", "_").replace("-", "_")
    k = k.lstrip("0123456789")  # PMRAM folders are named "512Glioma", "512Normal", ...
    return NAME_MAP.get(k)


# ---------------------------------------------------------------- manifests
def load_manifest() -> pd.DataFrame:
    """Unified manifest across cohorts, restricted to the shared 3-class label space."""
    df = pd.read_csv(PROC / "manifest_all.csv")
    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)
    df["y"] = df["label"].map(CLASS_TO_IDX).astype(int)
    return df


def cohort(df, name):
    return df[df["cohort"] == name].reset_index(drop=True)


# ---------------------------------------------------------------- splits
def grouped_folds(df, n_splits=5, seed=0):
    """Patient-grouped folds: every slice of a patient lands in exactly one fold."""
    g = df["patient_id"].astype(str).values
    # GroupKFold is deterministic; shuffle group order by seed for multi-seed runs.
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(g)))
    perm = {p: i for i, p in enumerate(rng.permutation(uniq))}
    order = np.array([perm[p] for p in g])
    gk = GroupKFold(n_splits=n_splits)
    return list(gk.split(df, df["y"].values, groups=order))


def slice_folds(df, n_splits=5, seed=0):
    """Slice-level stratified folds: the standard, leaky protocol we compare against."""
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sk.split(df, df["y"].values))


def split_calibration(df, tr_idx, frac=0.2, seed=0):
    """Carve a calibration set out of a training fold, disjoint by patient where
    patient identifiers exist, so temperature and the conformal quantile are never
    fitted on a patient the model trained on."""
    sub = df.iloc[tr_idx]
    rng = np.random.default_rng(seed)
    pids = sub["patient_id"].astype(str)
    if pids.nunique() > 1 and not pids.eq("nan").all():
        uniq = np.array(sorted(pids.unique()))
        rng.shuffle(uniq)
        n_cal = max(1, int(round(frac * len(uniq))))
        cal_p = set(uniq[:n_cal])
        m = pids.isin(cal_p).values
    else:
        m = rng.random(len(sub)) < frac
    return tr_idx[~m], tr_idx[m]


# ---------------------------------------------------------------- duplicate audit
def dhash(path, size=8):
    """64-bit difference hash. Near-duplicate slices survive resizing and requantising,
    which exact file hashing misses, and those are exactly the ones that leak."""
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def hamming(a, b):
    return bin(a ^ b).count("1")


def hashes_to_array(hs):
    """Unpack 64-bit hashes into a (n, 64) uint8 matrix for vectorised distances."""
    arr = np.zeros((len(hs), 64), dtype=np.uint8)
    for i, h in enumerate(hs):
        for b in range(64):
            arr[i, 63 - b] = (h >> b) & 1
    return arr


def near_duplicate_pairs(hA, hB, max_dist=3, block=512):
    """All cross-set pairs within `max_dist` bits. Blocked so memory stays bounded."""
    A, B = hashes_to_array(hA), hashes_to_array(hB)
    pairs = []
    for s in range(0, len(A), block):
        d = (A[s:s + block, None, :] != B[None, :, :]).sum(-1)
        ii, jj = np.where(d <= max_dist)
        pairs += [(int(s + i), int(j), int(d[i, j])) for i, j in zip(ii, jj)]
    return pairs


# ---------------------------------------------------------------- pixel verification
_PIX: dict = {}
PIX_PX = 96


def pixvec(rel):
    """96x96 grayscale vector of a slice (square-padded), cached per path."""
    if rel not in _PIX:
        a = np.asarray(Image.open(PROC / rel).convert("L"), dtype=np.uint8)
        h, w = a.shape
        if h != w:
            s = max(h, w)
            p = np.zeros((s, s), dtype=np.uint8)
            p[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
            a = p
        _PIX[rel] = np.asarray(Image.fromarray(a).resize((PIX_PX, PIX_PX), Image.LANCZOS),
                               dtype=np.float32)
    return _PIX[rel]


def pix_corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def verify_pairs(pairs, paths_a, paths_b, thr=0.95):
    """Keep the hash candidates whose pixels also agree (Pearson r >= thr)."""
    keep = []
    for i, j, dist in pairs:
        if pix_corr(pixvec(paths_a[i]), pixvec(paths_b[j])) >= thr:
            keep.append((i, j, dist))
    return keep
