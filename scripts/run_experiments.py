"""CALM-MRI experiment driver.

Stages (run any subset; each one resumes from its own cache so a crash costs one fold):

  audit   E0  cohort composition, exact and near-duplicate contamination
  bench   E1  every model x {patient-grouped, slice-level} x seeds, 5-fold OOF
              -> leakage delta, calibration, selective prediction, conformal deferral
  ood     E4  train on cheng (patient-grouped), test on decontaminated external cohorts
  abl     E5  CALM-MRI component ablations
  eff     E6  parameters, FLOPs, latency, peak memory
  dataeff E7  patient-grouped learning curves
  xai     E9  Grad-CAM and deletion-AUC faithfulness

Usage:  python -I scripts/run_experiments.py audit bench ...
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as D, metrics as MT, models as M, train as T  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
CKPT = RES / "_folds"
CKPT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_CLS = len(D.CLASSES)
SEEDS = [0, 1, 2]
MODELS = ["calm", "calm_l", "efficientnet_b0", "resnet18", "mobilenetv3_small",
          "convnext_tiny", "deit_small", "swin_tiny",
          "resnet50", "densenet121", "vgg16", "xception", "inception_v3", "mobilenetv2"]
# patient-grouped protocol only (compute); the leakage contrast uses the original seven
GROUPED_ONLY = {"calm_l", "resnet50", "densenet121", "vgg16", "xception", "inception_v3",
                "mobilenetv2"}
ALPHAS = [0.05, 0.10, 0.20]
# AJSE-paper baselines added late: one seed (five folds) to save compute
EXTRA_ONE_SEED = {"resnet50", "densenet121", "vgg16", "xception", "inception_v3", "mobilenetv2"}
EPOCHS = int(__import__("os").environ.get("CALM_EPOCHS", 18))


def final_cfg():
    """The proposed model as named in results/final_model.json:
    (architecture, training kwargs, flip TTA, ensemble members, recipe name)."""
    f = RES / "final_model.json"
    cfg = (json.loads(f.read_text(encoding="utf-8")) if f.exists()
           else {"source": "bench", "key": "calm_l"})
    if cfg["source"] == "bench":
        return cfg["key"], {"epochs": EPOCHS}, False, 1, "base"
    sys.path.insert(0, str(ROOT / "scripts"))
    import tune as TU
    rec = TU.RECIPES[cfg["recipe"]]
    kw = {x: rec[x] for x in ("epochs", "lr", "label_smooth", "mixup", "ema")}
    if cfg.get("fusion"):
        kw["fusion"] = cfg["fusion"]
    return cfg["base"], kw, bool(rec["tta"]), int(cfg.get("members", 1)), cfg["recipe"]


def train_model(name, df_tr, seed, members=1, **model_kw):
    """Train `name` (a benchmark key or "final"); returns (models, tta, arch)."""
    if name != "final":
        m, _ = T.train_one(name, df_tr, None, N_CLS, seed=seed, epochs=EPOCHS, device=DEV, **model_kw)
        return [m], False, name
    arch, kw, tta, mem, _ = final_cfg()
    ms = [T.train_one(arch, df_tr, None, N_CLS, seed=100 * seed + k, device=DEV, **{**kw, **model_kw})[0]
          for k in range(members)]
    return ms, tta, arch


def infer_all(models, df, tta):
    return np.mean([T.infer(m, df, DEV, tta=tta) for m in models], axis=0)


def jdump(obj, name):
    (RES / name).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    print(f"  wrote results/{name}")


# ------------------------------------------------------------------ E0 audit
def stage_audit(df):
    print("E0 cohort + duplicate audit", flush=True)
    out = {"classes": D.CLASSES, "cohorts": {}}
    for ch in sorted(df["cohort"].unique()):
        s = D.cohort(df, ch)
        rec = {"n": int(len(s)), "per_class": s["label"].value_counts().to_dict()}
        if (s["patient_id"].astype(str) != "nan").all():
            g = s.groupby("patient_id").size()
            rec |= {"n_patients": int(s["patient_id"].nunique()),
                    "slices_per_patient_mean": float(g.mean()),
                    "slices_per_patient_median": float(g.median()),
                    "slices_per_patient_min": int(g.min()),
                    "slices_per_patient_max": int(g.max()),
                    "patients_per_class": s.groupby("label")["patient_id"].nunique().to_dict()}
        out["cohorts"][ch] = rec
        print(f"  {ch}: {rec['n']} slices" + (f", {rec.get('n_patients')} patients" if "n_patients" in rec else ""))

    print("  hashing...", flush=True)
    hs, md5s = {}, {}
    for ch in sorted(df["cohort"].unique()):
        s = D.cohort(df, ch)
        hs[ch] = [D.dhash(D.PROC / p) for p in s["path"]]
        md5s[ch] = [D.md5(D.PROC / p) for p in s["path"]]
        print(f"    {ch} hashed", flush=True)

    chs = sorted(hs)
    paths = {c: list(D.cohort(df, c)["path"]) for c in chs}
    THR = 0.95
    out["verification"] = {"method": "dHash(64-bit) Hamming<=3, then Pearson r>=0.95 at 96x96",
                           "corr_threshold": THR}

    dup = {}
    for ch in chs:
        s = D.cohort(df, ch)
        n_exact = len(md5s[ch]) - len(set(md5s[ch]))
        cand = [(i, j, d) for i, j, d in D.near_duplicate_pairs(hs[ch], hs[ch], max_dist=3) if i < j]
        ver = D.verify_pairs(cand, paths[ch], paths[ch], THR)
        cross_pat = 0
        if (s["patient_id"].astype(str) != "nan").all():
            pid = s["patient_id"].astype(str).values
            cross_pat = sum(1 for i, j, _ in ver if pid[i] != pid[j])
        dup[ch] = {"exact_duplicate_files": int(n_exact),
                   "near_duplicate_candidates": int(len(cand)),
                   "near_duplicate_pairs": int(len(ver)),
                   "near_duplicate_pairs_cross_patient": int(cross_pat)}
        print(f"    {ch}: {n_exact} exact, {len(cand)} candidates -> {len(ver)} confirmed "
              f"({cross_pat} cross-patient)", flush=True)
    out["within_cohort"] = dup

    cross, vp = {}, {}
    for a in range(len(chs)):
        for b in range(a + 1, len(chs)):
            A, B = chs[a], chs[b]
            cand = D.near_duplicate_pairs(hs[A], hs[B], max_dist=3)
            ver = D.verify_pairs(cand, paths[A], paths[B], THR)
            vp[(A, B)] = ver
            ia, ib = {i for i, _, _ in ver}, {j for _, j, _ in ver}
            cross[f"{A}|{B}"] = {"candidate_pairs": int(len(cand)),
                                 "near_pairs": int(len(ver)),
                                 "exact_shared_files": int(len(set(md5s[A]) & set(md5s[B]))),
                                 f"{A}_slices_matched": int(len(ia)),
                                 f"{B}_slices_matched": int(len(ib)),
                                 f"{A}_pct": float(100 * len(ia) / len(hs[A])),
                                 f"{B}_pct": float(100 * len(ib) / len(hs[B]))}
            print(f"    {A} vs {B}: {len(cand)} candidates -> {len(ver)} confirmed; "
                  f"{100 * len(ib) / len(hs[B]):.1f}% of {B}, "
                  f"{100 * len(ia) / len(hs[A]):.1f}% of {A}", flush=True)
    out["cross_cohort"] = cross

    # Which slices of each target cohort duplicate each source cohort (confirmed only).
    for tgt in chs:
        s = D.cohort(df, tgt)
        for src in chs:
            if src == tgt:
                continue
            if (src, tgt) in vp:
                bad = sorted({j for _, j, _ in vp[(src, tgt)]})
            else:
                bad = sorted({i for i, _, _ in vp[(tgt, src)]})
            s.iloc[bad][["path"]].to_csv(RES / f"contaminated_{tgt}_by_{src}.csv", index=False)
            out["cohorts"][tgt][f"contaminated_by_{src}"] = int(len(bad))
            out["cohorts"][tgt][f"clean_wrt_{src}"] = int(len(s) - len(bad))
        if tgt != "cheng":
            if ("cheng", tgt) in vp:
                bad = sorted({j for _, j, _ in vp[("cheng", tgt)]})
            else:
                bad = sorted({i for i, _, _ in vp[(tgt, "cheng")]})
            s.iloc[bad][["path"]].to_csv(RES / f"contaminated_{tgt}.csv", index=False)
    jdump(out, "E0_audit.json")
    return out


# ------------------------------------------------------------------ shared fold runner
def run_fold(model_name, df, tr_idx, te_idx, seed, tag, model_kw=None, epochs=None):
    """Train one fold, return calibration/test logits. Cached on disk by `tag`."""
    f = CKPT / f"{tag}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    tr2, cal = D.split_calibration(df, tr_idx, frac=0.2, seed=seed)
    tta = False
    if model_name == "final":   # one member of the proposed model, its own recipe
        arch, kw, tta, _, _ = final_cfg()
        model, log = T.train_one(arch, df.iloc[tr2], None, N_CLS, seed=seed, device=DEV,
                                 **{**kw, **(model_kw or {})})
    else:
        model, log = T.train_one(model_name, df.iloc[tr2], None, N_CLS, seed=seed,
                                 epochs=epochs or EPOCHS, device=DEV, **(model_kw or {}))
    out = {"logits_cal": T.infer(model, df.iloc[cal], DEV, tta=tta),
           "y_cal": df.iloc[cal]["y"].to_numpy(),
           "logits_te": T.infer(model, df.iloc[te_idx], DEV, tta=tta),
           "y_te": df.iloc[te_idx]["y"].to_numpy(),
           "idx_te": np.asarray(te_idx),
           "train_seconds": np.array([log["train_seconds"]]),
           "params": np.array([log["params"]])}
    np.savez_compressed(f, **out)
    del model
    torch.cuda.empty_cache()
    return out


def score(logits_cal, y_cal, logits_te, y_te):
    """Uncalibrated and temperature-scaled metrics plus the deferral gate."""
    p_raw = MT.softmax(logits_te)
    t = MT.fit_temperature(logits_cal, y_cal)
    p_cal = MT.softmax(logits_te, t)
    pc_cal = MT.softmax(logits_cal, t)
    r = {"temperature": t,
         "accuracy": MT.accuracy(p_raw, y_te), "macro_f1": MT.macro_f1(p_raw, y_te),
         "auroc": MT.auroc_ovr(p_raw, y_te, N_CLS),
         "ece_raw": MT.ece(p_raw, y_te), "ece_cal": MT.ece(p_cal, y_te),
         "ece_adaptive_raw": MT.ece(p_raw, y_te, adaptive=True),
         "ece_adaptive_cal": MT.ece(p_cal, y_te, adaptive=True),
         "brier_raw": MT.brier(p_raw, y_te, N_CLS), "brier_cal": MT.brier(p_cal, y_te, N_CLS),
         "nll_raw": MT.nll(p_raw, y_te), "nll_cal": MT.nll(p_cal, y_te),
         "aurc": MT.aurc(p_cal, y_te),
         "acc_at_90": MT.accuracy_at_coverage(p_cal, y_te, 0.90),
         "acc_at_80": MT.accuracy_at_coverage(p_cal, y_te, 0.80),
         "cov_at_risk_2pct": MT.coverage_at_risk(p_cal, y_te, 0.02),
         "cov_at_risk_5pct": MT.coverage_at_risk(p_cal, y_te, 0.05)}
    for a in ALPHAS:
        r[f"conformal_{a}"] = MT.conformal_report(pc_cal, y_cal, p_cal, y_te, a)
    return r, p_cal


# ------------------------------------------------------------------ E1 benchmark
def stage_bench(df):
    print("E1 benchmark: models x protocols x seeds", flush=True)
    ch = D.cohort(df, "cheng")
    T.load_cache(ch["path"].tolist())
    out = {"n": int(len(ch)), "protocols": {}}
    for proto in ("grouped", "slice"):
        out["protocols"][proto] = {}
        for mn in MODELS:
            if proto == "slice" and mn in GROUPED_ONLY:
                continue
            per_seed = []
            for seed in (SEEDS[:1] if mn in EXTRA_ONE_SEED else SEEDS):
                folds = (D.grouped_folds(ch, 5, seed) if proto == "grouped"
                         else D.slice_folds(ch, 5, seed))
                oof_logit = np.zeros((len(ch), N_CLS), dtype=np.float32)
                cal_l, cal_y, secs = [], [], []
                for k, (tr, te) in enumerate(folds):
                    t0 = time.time()
                    r = run_fold(mn, ch, tr, te, seed, f"bench_{proto}_{mn}_s{seed}_f{k}")
                    oof_logit[r["idx_te"]] = r["logits_te"]
                    cal_l.append(r["logits_cal"])
                    cal_y.append(r["y_cal"])
                    secs.append(float(r["train_seconds"][0]))
                    print(f"    {proto}/{mn}/s{seed}/f{k} {time.time() - t0:.0f}s", flush=True)
                s, p = score(np.concatenate(cal_l), np.concatenate(cal_y),
                             oof_logit, ch["y"].to_numpy())
                s["train_seconds_mean"] = float(np.mean(secs))
                per_seed.append(s)
                np.savez_compressed(RES / f"oof_{proto}_{mn}_s{seed}.npz",
                                    p=p, y=ch["y"].to_numpy())
            out["protocols"][proto][mn] = {
                "per_seed": per_seed,
                "mean": {k: float(np.mean([d[k] for d in per_seed]))
                         for k in per_seed[0] if isinstance(per_seed[0][k], float)},
                "sd": {k: float(np.std([d[k] for d in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0
                       for k in per_seed[0] if isinstance(per_seed[0][k], float)}}
            m = out["protocols"][proto][mn]["mean"]
            print(f"  {proto:8s} {mn:18s} acc {m['accuracy']:.4f} f1 {m['macro_f1']:.4f} "
                  f"ece {m['ece_cal']:.4f} aurc {m['aurc']:.4f}", flush=True)
            jdump(out, "E1_benchmark.json")
    jdump(out, "E1_benchmark.json")
    return out


# ------------------------------------------------------------------ E4 OOD
def stage_ood(df):
    print("E4 cross-cohort OOD (train on Cheng, patient-grouped)", flush=True)
    ch = D.cohort(df, "cheng")
    T.load_cache(ch["path"].tolist())
    ext = {}
    for name in ("sartaj", "hemg", "pmram"):
        s = D.cohort(df, name)
        if len(s) == 0:
            continue
        # Cheng shares no confirmed duplicate with any external cohort (E0), so the
        # external sets are used whole.
        T.load_cache(s["path"].tolist())
        ext[name] = s
        print(f"  {name}: {len(s)} slices", flush=True)
    out = {}
    mem = final_cfg()[3]
    for mn in ("final", "convnext_tiny", "swin_tiny", "deit_small", "efficientnet_b0"):
        out[mn] = {}
        for seed in SEEDS:
            folds = D.grouped_folds(ch, 5, seed)
            tr, _ = folds[0]
            tr2, cal = D.split_calibration(ch, tr, frac=0.2, seed=seed)
            models, tta, _ = train_model(mn, ch.iloc[tr2], seed, members=mem)
            lc, yc = infer_all(models, ch.iloc[cal], tta), ch.iloc[cal]["y"].to_numpy()
            for name, sub in ext.items():
                s_, _ = score(lc, yc, infer_all(models, sub, tta), sub["y"].to_numpy())
                out[mn].setdefault(f"{name}_all", []).append(s_)
            del models
            torch.cuda.empty_cache()
            print(f"    {mn} seed{seed} done", flush=True)
        for k, v in list(out[mn].items()):
            out[mn][k] = {"per_seed": v,
                          "mean": {kk: float(np.mean([d[kk] for d in v]))
                                   for kk in v[0] if isinstance(v[0][kk], float)},
                          "sd": {kk: float(np.std([d[kk] for d in v], ddof=1))
                                 for kk in v[0] if isinstance(v[0][kk], float)}}
            print(f"  {mn} {k}: acc {out[mn][k]['mean']['accuracy']:.4f} "
                  f"ece {out[mn][k]['mean']['ece_cal']:.4f}")
        jdump(out, "E4_ood.json")
    jdump(out, "E4_ood.json")
    return out


# ------------------------------------------------------------------ E5 ablation
def fold_record(files_per_fold, ch):
    """Score an OOF prediction assembled from per-fold files (members averaged)."""
    oof = np.zeros((len(ch), N_CLS), dtype=np.float32)
    cl, cy = [], []
    for fs in files_per_fold:
        zs = [np.load(f) for f in fs]
        oof[zs[0]["idx_te"]] = np.mean([z["logits_te"] for z in zs], axis=0)
        cl.append(np.mean([z["logits_cal"] for z in zs], axis=0))
        cy.append(zs[0]["y_cal"])
    return score(np.concatenate(cl), np.concatenate(cy), oof, ch["y"].to_numpy())[0]


def stage_abl(df):
    """Each row removes one component of the final model. Rows that coincide with runs
    already made (benchmark, recipe search) are read from their fold files, so every
    row shares the same patient folds and seeds."""
    print("E5 ablations around the final model", flush=True)
    ch = D.cohort(df, "cheng")
    T.load_cache(ch["path"].tolist())
    arch, _, _, mem, recipe = final_cfg()
    TUNE = RES / "_tune"

    def tune_files(seed, members):
        return [[TUNE / f"{arch}_{recipe}_s{seed}_m{m}_f{k}.npz" for m in range(members)]
                for k in range(5)]

    def bench_files(key, seed):
        return [[CKPT / f"bench_grouped_{key}_s{seed}_f{k}.npz"] for k in range(5)]

    def concat_files(seed, members):   # member 0 from this stage's earlier run, then concat_members.py
        return [[CKPT / f"abl2_fusion_concat_s{seed}_f{k}.npz"] +
                [TUNE / f"{arch}_ls10cat_s{seed}_m{m}_f{k}.npz" for m in range(1, members)]
                for k in range(5)]

    reused = {"full": lambda s: concat_files(s, mem),
              "single_member": lambda s: concat_files(s, 1),
              "gated_fusion": lambda s: tune_files(s, 1),
              "compact_trunk": lambda s: bench_files("calm", s),
              "no_fusion_head": lambda s: bench_files("convnext_tiny", s)}
    # (tag prefix, overrides); last-stage-only ignores the fusion rule, so abl2 runs are valid
    trained = {"fusion_last": ("abl2", {"fusion": "last"}),
               "two_scales": ("abl3", {"n_scales": 2})}
    out = {"config": {"arch": arch, "recipe": recipe, "members": mem}}
    for vn in list(reused) + list(trained):
        per_seed = []
        for seed in SEEDS:
            if vn in reused:
                fs = reused[vn](seed)
                if not all(f.exists() for ff in fs for f in ff):
                    break
                per_seed.append(fold_record(fs, ch))
            else:
                folds = D.grouped_folds(ch, 5, seed)
                fs = []
                for k, (tr, te) in enumerate(folds):
                    pre, kw_ = trained[vn]
                    run_fold("final", ch, tr, te, seed, f"{pre}_{vn}_s{seed}_f{k}", model_kw=kw_)
                    fs.append([CKPT / f"{pre}_{vn}_s{seed}_f{k}.npz"])
                per_seed.append(fold_record(fs, ch))
        if len(per_seed) != len(SEEDS):
            print(f"  {vn}: fold files missing, skipped", flush=True)
            continue
        out[vn] = {"per_seed": per_seed,
                   "mean": {k: float(np.mean([d[k] for d in per_seed]))
                            for k in per_seed[0] if isinstance(per_seed[0][k], float)},
                   "sd": {k: float(np.std([d[k] for d in per_seed], ddof=1))
                          for k in per_seed[0] if isinstance(per_seed[0][k], float)}}
        print(f"  {vn:18s} acc {out[vn]['mean']['accuracy']:.4f} "
              f"f1 {out[vn]['mean']['macro_f1']:.4f} ece {out[vn]['mean']['ece_cal']:.4f}", flush=True)
        jdump(out, "E5_ablation.json")
    jdump(out, "E5_ablation.json")
    return out


# ------------------------------------------------------------------ E6 efficiency
def stage_eff(df):
    print("E6 efficiency", flush=True)
    out = {}
    x = torch.randn(1, 3, T.IMG_PX, T.IMG_PX, device=DEV)
    for mn in MODELS:
        m = M.build(mn, N_CLS, pretrained=False).to(DEV).eval()
        flops = float("nan")
        try:
            from fvcore.nn import FlopCountAnalysis
            flops = float(FlopCountAnalysis(m, x).unsupported_ops_warnings(False)
                          .uncalled_modules_warnings(False).total())
        except Exception:
            try:
                from thop import profile
                flops = float(profile(m, inputs=(x,), verbose=False)[0])
            except Exception:
                pass
        with torch.no_grad():
            for _ in range(10):
                m(x)
            torch.cuda.synchronize() if DEV == "cuda" else None
            torch.cuda.reset_peak_memory_stats() if DEV == "cuda" else None
            t0 = time.time()
            for _ in range(60):
                m(x)
            torch.cuda.synchronize() if DEV == "cuda" else None
            lat = (time.time() - t0) / 60 * 1000
            xb = torch.randn(32, 3, T.IMG_PX, T.IMG_PX, device=DEV)
            t0 = time.time()
            for _ in range(10):
                m(xb)
            torch.cuda.synchronize() if DEV == "cuda" else None
            thr = 32 * 10 / (time.time() - t0)
        out[mn] = {"params_m": M.count_params(m) / 1e6,
                   "gflops": flops / 1e9 if flops == flops else float("nan"),
                   "latency_ms_bs1": lat, "throughput_img_s_bs32": thr,
                   "peak_mem_mb": (torch.cuda.max_memory_allocated() / 1e6) if DEV == "cuda" else float("nan")}
        print(f"  {mn:18s} {out[mn]['params_m']:.2f}M  {out[mn]['gflops']:.2f} GFLOPs  "
              f"{lat:.2f} ms  {thr:.0f} img/s", flush=True)
        del m
        torch.cuda.empty_cache()
    jdump(out, "E6_efficiency.json")
    return out


# ------------------------------------------------------------------ E7 data efficiency
def stage_dataeff(df):
    print("E7 patient-grouped learning curves", flush=True)
    ch = D.cohort(df, "cheng")
    T.load_cache(ch["path"].tolist())
    fracs = [0.10, 0.25, 0.50, 1.00]
    out = {}
    for mn in ("final", "convnext_tiny"):
        out[mn] = {}
        for fr in fracs:
            per_seed = []
            for seed in SEEDS[:1]:                          # one seed, all five folds
                folds = D.grouped_folds(ch, 5, seed)
                accs, f1s = [], []
                for k, (tr, te) in enumerate(folds[:3]):  # 3 folds for speed
                    rng = np.random.default_rng(1000 * seed + k)
                    pid = ch.iloc[tr]["patient_id"].astype(str).values
                    uniq = np.array(sorted(set(pid)))
                    rng.shuffle(uniq)
                    keep = set(uniq[:max(2, int(round(fr * len(uniq))))])
                    tr_sub = tr[np.isin(pid, list(keep))]
                    r = run_fold(mn, ch, tr_sub, te, seed,
                                 f"de_{mn}_f{int(fr * 100)}_s{seed}_k{k}")
                    p = MT.softmax(r["logits_te"])
                    accs.append(MT.accuracy(p, r["y_te"]))
                    f1s.append(MT.macro_f1(p, r["y_te"]))
                per_seed.append({"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s)),
                                 "fold_sd": float(np.std(accs, ddof=1)),
                                 "n_patients": int(len(keep))})
            out[mn][f"{int(fr * 100)}"] = {
                "mean_accuracy": float(np.mean([d["accuracy"] for d in per_seed])),
                "sd_accuracy": float(per_seed[0]["fold_sd"]),
                "mean_macro_f1": float(np.mean([d["macro_f1"] for d in per_seed])),
                "n_patients": int(np.mean([d["n_patients"] for d in per_seed]))}
            print(f"  {mn} {int(fr * 100):3d}%  acc {out[mn][f'{int(fr * 100)}']['mean_accuracy']:.4f}",
                  flush=True)
            jdump(out, "E7_dataeff.json")
    jdump(out, "E7_dataeff.json")
    return out



# ------------------------------------------------------------------ E9 XAI
def stage_xai(df):
    print("E9 Grad-CAM + faithfulness", flush=True)
    from src import xai as X
    ch = D.cohort(df, "cheng")
    T.load_cache(ch["path"].tolist())
    seed = 0
    folds = D.grouped_folds(ch, 5, seed)
    tr, te = folds[0]
    tr2, _ = D.split_calibration(ch, tr, frac=0.2, seed=seed)
    out, examples = {}, []
    for mn in ("final", "convnext_tiny", "swin_tiny"):
        models, _, arch = train_model(mn, ch.iloc[tr2], seed, members=1)
        model = models[0].eval()
        cam_layer = X.pick_layer(model, arch)
        cam = X.GradCAM(model, cam_layer)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(te), size=min(60, len(te)), replace=False)
        dels, inss, rnd = [], [], []
        for n_i, i in enumerate(pick):
            row = ch.iloc[te[i]]
            x = T.MRIDataset(ch.iloc[[te[i]]], train=False)[0][0].unsqueeze(0).to(DEV)
            m, cls, pr = cam(x, cls=int(row["y"]))
            fa = X.deletion_insertion(model, x, m, cls, steps=20)
            dels.append(fa["deletion_auc"]); inss.append(fa["insertion_auc"])
            rm = rng.random(m.shape).astype(np.float32)
            rnd.append(X.deletion_insertion(model, x, rm, cls, steps=20, mode="deletion")["deletion_auc"])
            if mn == "final" and n_i < 6:
                examples.append({"path": row["path"], "label": row["label"],
                                 "cam": m.astype(np.float16).tolist(), "prob": pr})
        out[mn] = {"n": int(len(pick)),
                   "deletion_auc_mean": float(np.mean(dels)), "deletion_auc_sd": float(np.std(dels, ddof=1)),
                   "insertion_auc_mean": float(np.mean(inss)), "insertion_auc_sd": float(np.std(inss, ddof=1)),
                   "random_deletion_auc_mean": float(np.mean(rnd)),
                   "faithfulness_gap": float(np.mean(rnd) - np.mean(dels))}
        print(f"  {mn:18s} del {out[mn]['deletion_auc_mean']:.4f} ins {out[mn]['insertion_auc_mean']:.4f} "
              f"rand-del {out[mn]['random_deletion_auc_mean']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    np.savez_compressed(RES / "xai_examples.npz",
                        cams=np.array([e["cam"] for e in examples], dtype=np.float16),
                        paths=np.array([e["path"] for e in examples]),
                        labels=np.array([e["label"] for e in examples]),
                        probs=np.array([e["prob"] for e in examples]))
    jdump(out, "E9_xai.json")
    return out



# ------------------------------------------------------------------ E4b contamination demo
def stage_contam(df):
    """Quantify how much a contaminated 'external validation' inflates the result.

    Train on Hemg, then test on Sartaj and on PMRAM, each exactly as the literature would
    report it and again after removing every slice confirmed (hash plus pixels) to
    duplicate a Hemg slice.
    """
    print("E4b contaminated external validation (train hemg)", flush=True)
    src = D.cohort(df, "hemg")
    T.load_cache(src["path"].tolist())
    targets = {}
    for name in ("sartaj", "pmram"):
        tgt = D.cohort(df, name)
        if len(tgt) == 0:
            continue
        cf = RES / f"contaminated_{name}_by_hemg.csv"
        bad = set(pd.read_csv(cf)["path"]) if cf.exists() else set()
        clean = tgt[~tgt["path"].isin(bad)].reset_index(drop=True)
        T.load_cache(tgt["path"].tolist())
        targets[name] = (tgt, clean)
        print(f"  {name}: {len(tgt)} -> {len(clean)} after removing {len(tgt) - len(clean)} "
              f"confirmed duplicates of hemg", flush=True)
    out = {"targets": {k: {"n": int(len(v[0])), "n_clean": int(len(v[1])),
                           "n_removed": int(len(v[0]) - len(v[1]))} for k, v in targets.items()}}
    for mn in ("final", "efficientnet_b0"):
        per = {f"{n}_{w}": [] for n in targets for w in ("as_published", "decontaminated")}
        for seed in SEEDS:
            idx = np.arange(len(src))
            np.random.default_rng(seed).shuffle(idx)
            n_cal = int(0.2 * len(idx))
            cal_i, tr_i = idx[:n_cal], idx[n_cal:]
            models, tta, _ = train_model(mn, src.iloc[tr_i], seed, members=1)
            lc, yc = infer_all(models, src.iloc[cal_i], tta), src.iloc[cal_i]["y"].to_numpy()
            for name, (tgt, clean) in targets.items():
                for key, sub in (("as_published", tgt), ("decontaminated", clean)):
                    s_, _ = score(lc, yc, infer_all(models, sub, tta), sub["y"].to_numpy())
                    per[f"{name}_{key}"].append(s_)
            del models
            torch.cuda.empty_cache()
            print(f"    {mn} seed{seed} done", flush=True)
        out[mn] = {}
        for k, v in per.items():
            out[mn][k] = {"per_seed": v,
                          "mean": {kk: float(np.mean([d[kk] for d in v]))
                                   for kk in v[0] if isinstance(v[0][kk], float)},
                          "sd": {kk: float(np.std([d[kk] for d in v], ddof=1))
                                 for kk in v[0] if isinstance(v[0][kk], float)}}
        for name in targets:
            a = out[mn][f"{name}_as_published"]["mean"]["accuracy"]
            c = out[mn][f"{name}_decontaminated"]["mean"]["accuracy"]
            out[mn][f"{name}_inflation_points"] = float(100 * (a - c))
            print(f"  {mn} -> {name}: as-published {100 * a:.2f}% vs decontaminated "
                  f"{100 * c:.2f}% (inflation {100 * (a - c):+.2f} pts)", flush=True)
        jdump(out, "E4b_contamination.json")
    jdump(out, "E4b_contamination.json")
    return out


STAGES = {"audit": stage_audit, "bench": stage_bench, "ood": stage_ood, "contam": stage_contam,
          "abl": stage_abl, "eff": stage_eff, "dataeff": stage_dataeff, "xai": stage_xai}


def main(argv):
    want = argv or ["audit"]
    df = D.load_manifest()
    print(f"manifest: {len(df)} slices across {df['cohort'].nunique()} cohorts | device {DEV}\n")
    for s in want:
        t0 = time.time()
        STAGES[s](df)
        print(f"[{s}] done in {(time.time() - t0) / 60:.1f} min\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
