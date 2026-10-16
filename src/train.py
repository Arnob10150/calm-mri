"""Training and inference for one fold.

All slices are decoded once into a uint8 cache at CACHE_PX so the many short runs the
study needs (models x folds x seeds x protocols) are GPU-bound rather than disk-bound.
Every run uses the identical schedule, so protocol and architecture are the only
things that differ between the numbers we compare.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import models as M

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CACHE_PX = 256
IMG_PX = 224
MEAN, STD = 0.449, 0.226  # ImageNet grayscale equivalent

_CACHE: dict[str, np.ndarray] = {}


def _clahe(a):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(a)


def load_cache(paths, use_clahe=True):
    """Decode, CLAHE, and square-pad every slice once; keep it in RAM as uint8."""
    todo = [p for p in paths if p not in _CACHE]
    for i, rel in enumerate(todo):
        im = Image.open(PROC / rel).convert("L")
        a = np.asarray(im, dtype=np.uint8)
        h, w = a.shape
        if h != w:  # pad to square before resize so aspect ratio is preserved
            s = max(h, w)
            pad = np.zeros((s, s), dtype=np.uint8)
            pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
            a = pad
        a = cv2.resize(a, (CACHE_PX, CACHE_PX), interpolation=cv2.INTER_AREA)
        if use_clahe:
            a = _clahe(a)
        _CACHE[rel] = a
        if (i + 1) % 2000 == 0:
            print(f"    cached {i + 1}/{len(todo)}", flush=True)
    return len(todo)


class MRIDataset(Dataset):
    def __init__(self, df, train: bool, rng_seed=0):
        self.paths = df["path"].tolist()
        self.y = df["y"].to_numpy()
        self.train = train
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        a = _CACHE[self.paths[i]]
        if self.train:
            # scale/shift/rotate: MRI slices vary in head size and in-plane rotation
            ang = self.rng.uniform(-12, 12)
            sc = self.rng.uniform(0.9, 1.1)
            tx, ty = self.rng.uniform(-0.05, 0.05, 2) * CACHE_PX
            Mx = cv2.getRotationMatrix2D((CACHE_PX / 2, CACHE_PX / 2), ang, sc)
            Mx[:, 2] += (tx, ty)
            a = cv2.warpAffine(a, Mx, (CACHE_PX, CACHE_PX), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            o = (CACHE_PX - IMG_PX) // 2
            j = self.rng.integers(0, 2 * o + 1, 2) if o > 0 else (0, 0)
            a = a[j[0]:j[0] + IMG_PX, j[1]:j[1] + IMG_PX]
            if self.rng.random() < 0.5:
                a = a[:, ::-1]
            if self.rng.random() < 0.5:  # brightness / contrast jitter
                a = np.clip(a.astype(np.float32) * self.rng.uniform(0.9, 1.1)
                            + self.rng.uniform(-12, 12), 0, 255).astype(np.uint8)
        else:
            o = (CACHE_PX - IMG_PX) // 2
            a = a[o:o + IMG_PX, o:o + IMG_PX]
        x = (np.ascontiguousarray(a).astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1), int(self.y[i])


@torch.no_grad()
def infer(model, df, device, bs=64, workers=0, tta=False):
    """Logits for every row of df. tta=True averages the logits of the slice and its
    horizontal mirror, which is label-preserving for axial brain MRI."""
    model.eval()
    dl = DataLoader(MRIDataset(df, train=False), batch_size=bs, shuffle=False, num_workers=workers)
    out = []
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            z = model(x).float()
            if tta:
                z = 0.5 * (z + model(torch.flip(x, dims=[3])).float())
        out.append(z.cpu())
    return torch.cat(out).numpy()


def train_one(model_name, df_tr, df_va, n_classes, seed=0, epochs=18, bs=32, lr=3e-4,
              head_epochs=3, device="cuda", label_smooth=0.05, workers=0, verbose=False,
              mixup=0.0, ema=0.0, **model_kw):
    """Two-stage transfer: a short frozen-trunk warmup, then full fine-tuning with
    cosine decay. Returns the trained model and a small training log.

    mixup > 0 mixes pairs of training slices with Beta(mixup, mixup) weights during
    fine-tuning; ema > 0 keeps an exponential moving average of the weights over the
    fine-tuning steps and returns those averaged weights. Both default to off, so the
    benchmark recipe is unchanged unless a caller asks for them."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = M.build(model_name, n_classes, pretrained=True, **model_kw).to(device)
    is_calm = model_name.startswith("calm")
    # Warm-up freezes the pretrained trunk but must leave the new head trainable. For
    # CALM the head is a separate module; for a timm baseline it is get_classifier().
    if is_calm:
        head_params = {id(p) for p in model.head.parameters()}
    else:
        head_params = {id(p) for p in model.get_classifier().parameters()}

    dl = DataLoader(MRIDataset(df_tr, train=True, rng_seed=seed), batch_size=bs, shuffle=True,
                    num_workers=workers, drop_last=len(df_tr) > bs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    def set_trunk_grad(flag):
        for p in model.parameters():
            p.requires_grad = flag or (id(p) in head_params)

    ema_state = None
    beta = torch.distributions.Beta(mixup, mixup) if mixup > 0 else None

    hist, t0 = [], time.time()
    for ep in range(epochs):
        if ep == 0:
            set_trunk_grad(False)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3,
                                    weight_decay=1e-4)
            sch = None
        if ep == head_epochs:
            set_trunk_grad(True)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - head_epochs))
        model.train()
        tot, n, corr = 0.0, 0, 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            mix = beta is not None and ep >= head_epochs
            if mix:
                lam = float(beta.sample())
                perm = torch.randperm(x.size(0), device=x.device)
                x = lam * x + (1.0 - lam) * x[perm]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                z = model(x)
                loss = (lam * crit(z, y) + (1.0 - lam) * crit(z, y[perm])) if mix else crit(z, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema > 0 and ep >= head_epochs:
                with torch.no_grad():
                    sd = model.state_dict()
                    if ema_state is None:
                        ema_state = {k: v.detach().clone() for k, v in sd.items()}
                    else:
                        for k, v in sd.items():
                            if v.dtype.is_floating_point:
                                ema_state[k].mul_(ema).add_(v.detach(), alpha=1.0 - ema)
                            else:
                                ema_state[k].copy_(v)
            tot += loss.item() * len(y)
            corr += (z.argmax(1) == y).sum().item()
            n += len(y)
        if ep >= head_epochs and sch is not None:
            sch.step()
        rec = {"epoch": ep, "train_loss": tot / n, "train_acc": corr / n}
        if df_va is not None and (ep == epochs - 1 or verbose):
            pv = infer(model, df_va, device, workers=workers)
            rec["val_acc"] = float((pv.argmax(1) == df_va["y"].to_numpy()).mean())
        hist.append(rec)
        if verbose:
            print(f"    ep{ep} loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f}"
                  + (f" val {rec.get('val_acc', float('nan')):.3f}" if "val_acc" in rec else ""),
                  flush=True)
    if ema_state is not None:
        model.load_state_dict(ema_state)
    return model, {"history": hist, "train_seconds": time.time() - t0,
                   "params": M.count_params(model)}
