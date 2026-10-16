"""Extract the Cheng brain-tumor dataset (figshare 10.6084/m9.figshare.1512427, CC BY 4.0).

Each .mat is MATLAB v7.3 (HDF5) holding cjdata = {label, PID, image, tumorBorder, tumorMask}.
label: 1 = meningioma, 2 = glioma, 3 = pituitary. PID is the patient identifier that makes
patient-grouped evaluation possible; it is the reason this cohort is our primary one.

Writes data/processed/cheng/<class>/<slice_id>.png and data/processed/cheng_manifest.csv.
Run:  python -I scripts/prepare_cheng.py
"""
import csv
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cheng"
MATS = RAW / "mats"
OUT = ROOT / "data" / "processed" / "cheng"
LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def unzip_all():
    MATS.mkdir(parents=True, exist_ok=True)
    for z in sorted(RAW.glob("p*.zip")):
        with zipfile.ZipFile(z) as f:
            names = [n for n in f.namelist() if n.lower().endswith(".mat")]
            todo = [n for n in names if not (MATS / Path(n).name).exists()]
            for n in todo:
                with f.open(n) as src, open(MATS / Path(n).name, "wb") as dst:
                    dst.write(src.read())
        print(f"  {z.name}: {len(names)} mats", flush=True)


def read_mat(p: Path):
    with h5py.File(p, "r") as f:
        c = f["cjdata"]
        label = int(np.array(c["label"]).ravel()[0])
        pid = "".join(chr(int(v)) for v in np.array(c["PID"]).ravel()).strip()
        img = np.array(c["image"], dtype=np.float64)
    return label, pid, img


def to_uint8(img):
    """Per-slice min-max to 8-bit. Source slices are 16-bit with scanner-dependent range,
    so a fixed window would blank out whole acquisitions."""
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def main():
    print("unzipping...", flush=True)
    unzip_all()
    mats = sorted(MATS.glob("*.mat"), key=lambda p: int(p.stem))
    print(f"{len(mats)} slices", flush=True)
    for c in LABELS.values():
        (OUT / c).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(mats):
        label, pid, img = read_mat(p)
        cls = LABELS[label]
        arr = to_uint8(img)
        name = f"{p.stem}.png"
        Image.fromarray(arr).save(OUT / cls / name)
        rows.append({"slice_id": p.stem, "patient_id": pid, "label": cls,
                     "height": arr.shape[0], "width": arr.shape[1],
                     "path": f"cheng/{cls}/{name}"})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(mats)}", flush=True)
    man = ROOT / "data" / "processed" / "cheng_manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pids = {r["patient_id"] for r in rows}
    print(f"wrote {man}: {len(rows)} slices, {len(pids)} patients")
    for c in LABELS.values():
        sub = [r for r in rows if r["label"] == c]
        print(f"  {c}: {len(sub)} slices, {len({r['patient_id'] for r in sub})} patients")


if __name__ == "__main__":
    sys.exit(main())
