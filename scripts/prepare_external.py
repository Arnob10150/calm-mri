"""Unpack the two external cohorts and build the unified manifest.

sartaj: Training.zip / Testing.zip with one folder per class.
hemg:   a single parquet of {image: bytes, label: int}.

Neither ships patient identifiers, so both are used only as external test cohorts
(never for patient-grouped training claims). Run: python -I scripts/prepare_external.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT))
from src.data import norm_label  # noqa: E402


def prep_sartaj():
    out = PROC / "sartaj"
    rows = []
    for zname in ("Training.zip", "Testing.zip"):
        zp = RAW / "sartaj" / zname
        if not zp.exists():
            print(f"  missing {zp}")
            continue
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.endswith("/") or not n.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                parts = [p for p in Path(n).parts if p not in (".", "")]
                lab = None
                for p in parts[:-1]:
                    lab = norm_label(p) or lab
                if lab is None:
                    continue
                dst = out / lab / f"{zname[:-4].lower()}_{Path(n).stem}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    with z.open(n) as f:
                        Image.open(io.BytesIO(f.read())).convert("L").save(dst)
                rows.append({"cohort": "sartaj", "slice_id": dst.stem, "patient_id": "nan",
                             "label": lab, "path": f"sartaj/{lab}/{dst.name}"})
    print(f"  sartaj: {len(rows)} images")
    return rows


def prep_hemg():
    p = RAW / "hemg" / "train.parquet"
    if not p.exists():
        print("  missing hemg parquet")
        return []
    try:
        df = pd.read_parquet(p)
    except Exception as e:  # truncated or still downloading: skip, do not lose the manifest
        print(f"  hemg parquet unreadable ({type(e).__name__}); skipping this cohort")
        return []
    print(f"  hemg parquet columns: {list(df.columns)}, rows {len(df)}")
    icol = next((c for c in df.columns if "image" in c.lower() or "img" in c.lower()), df.columns[0])
    lcol = next((c for c in df.columns if "label" in c.lower() or "class" in c.lower()), df.columns[-1])
    # Integer class ids, per the dataset card's class_label feature. "notumor" has no
    # counterpart in the Cheng cohort and is dropped by norm_label().
    names = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}
    cand = p.parent / "label_names.json"
    if cand.exists():
        names = {int(k): v for k, v in json.loads(cand.read_text()).items()}
    rows = []
    out = PROC / "hemg"
    for i, r in df.iterrows():
        raw = r[icol]
        if isinstance(raw, dict):
            raw = raw.get("bytes", raw.get("path"))
        if not isinstance(raw, (bytes, bytearray)):
            continue
        lv = r[lcol]
        lab = norm_label(names.get(int(lv), lv) if str(lv).lstrip("-").isdigit() else lv)
        if lab is None:
            continue
        dst = out / lab / f"hemg_{i:06d}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            Image.open(io.BytesIO(raw)).convert("L").save(dst)
        rows.append({"cohort": "hemg", "slice_id": dst.stem, "patient_id": "nan",
                     "label": lab, "path": f"hemg/{lab}/{dst.name}"})
    print(f"  hemg: {len(rows)} images")
    return rows


def prep_pmram():
    """PMRAM Bangladeshi Brain Cancer MRI (Mendeley 10.17632/m7w55sw88b.1, CC BY 4.0).

    Only the *raw* archive is used. The augmented archive holds rotated and sheared
    copies of those same images, which would put near-duplicates inside the cohort.
    The "Normal" class has no counterpart in Cheng and is dropped by norm_label().
    """
    zp = RAW / "pmram" / "raw.zip"
    if not zp.exists():
        print("  missing pmram raw.zip")
        return []
    out = PROC / "pmram"
    rows = []
    with zipfile.ZipFile(zp) as z:
        for n in sorted(z.namelist()):
            if n.endswith("/") or not n.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            lab = None
            for part in Path(n).parts[:-1]:
                lab = norm_label(part) or lab
            if lab is None:
                continue
            dst = out / lab / f"pmram_{Path(n).stem.replace(' ', '').replace('(', '_').replace(')', '')}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                with z.open(n) as f:
                    Image.open(io.BytesIO(f.read())).convert("L").save(dst)
            rows.append({"cohort": "pmram", "slice_id": dst.stem, "patient_id": "nan",
                         "label": lab, "path": f"pmram/{lab}/{dst.name}"})
    print(f"  pmram: {len(rows)} images (normal class excluded)")
    return rows


def main():
    rows = []
    cm = PROC / "cheng_manifest.csv"
    if cm.exists():
        c = pd.read_csv(cm)
        c["cohort"] = "cheng"
        rows += c[["cohort", "slice_id", "patient_id", "label", "path"]].to_dict("records")
        print(f"  cheng: {len(c)} images")
    print("sartaj...", flush=True)
    rows += prep_sartaj()
    print("hemg...", flush=True)
    rows += prep_hemg()
    print("pmram...", flush=True)
    rows += prep_pmram()
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "manifest_all.csv", index=False)
    print(f"\nwrote manifest_all.csv: {len(df)} rows")
    print(df.groupby(["cohort", "label"]).size().to_string())


if __name__ == "__main__":
    sys.exit(main())
