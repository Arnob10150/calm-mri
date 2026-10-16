"""Grad-CAM with a quantitative faithfulness check.

A heat map that looks plausible is not evidence that the model used that region, which
is the usual weakness of Grad-CAM figures in this literature. We therefore pair every
map with deletion and insertion AUC: pixels are removed (or restored) in attribution
order and the predicted-class probability is tracked. A faithful map gives a steep
deletion curve (low AUC) and a steep insertion curve (high AUC).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM on the last spatial feature map of the trunk."""

    def __init__(self, model, layer):
        self.model = model.eval()
        self.a = None
        self.g = None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, o):
        self.a = o.detach() if isinstance(o, torch.Tensor) else o[0].detach()

    def _bwd(self, _m, _gi, go):
        self.g = go[0].detach()

    def __call__(self, x, cls=None):
        self.model.zero_grad(set_to_none=True)
        z = self.model(x)
        if cls is None:
            cls = int(z.argmax(1)[0])
        z[0, cls].backward()
        a, g = self.a, self.g
        if a.dim() == 3:  # transformer tokens (B, N, C) -> square grid
            b, n, c = a.shape
            s = int(round(n ** 0.5))
            if s * s != n:          # drop a class token if present
                a, g, n = a[:, 1:], g[:, 1:], n - 1
                s = int(round(n ** 0.5))
            a = a.transpose(1, 2).reshape(b, c, s, s)
            g = g.transpose(1, 2).reshape(b, c, s, s)
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).cpu().numpy(), cls, float(torch.softmax(z, 1)[0, cls])


def pick_layer(model, name):
    """Last spatial module of the trunk, per architecture family."""
    trunk = getattr(model, "trunk", model)
    if name.startswith("calm"):
        return trunk.stages_3 if hasattr(trunk, "stages_3") else list(trunk.children())[-1]
    mods = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if mods:
        return mods[-1]
    norms = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
    return norms[-2] if len(norms) > 1 else norms[-1]


@torch.no_grad()
def _prob(model, x, cls):
    return float(torch.softmax(model(x), 1)[0, cls])


def deletion_insertion(model, x, cam, cls, steps=25, mode="both"):
    """Remove (or restore) pixels in attribution order; return normalised AUCs."""
    n_px = cam.size
    order = np.argsort(-cam.flatten())
    base = x.clone()
    blur = F.avg_pool2d(x, 17, stride=1, padding=8)  # blurred image as the neutral state
    out = {}
    if mode in ("both", "deletion"):
        xs, cur = [], base.clone()
        for s in range(steps + 1):
            k = int(round(s / steps * n_px))
            cur = base.clone()
            if k:
                idx = order[:k]
                m = torch.zeros(n_px, dtype=torch.bool, device=x.device)
                m[torch.as_tensor(idx.copy(), device=x.device)] = True
                m = m.reshape(cam.shape)
                cur[:, :, m] = blur[:, :, m]
            xs.append(_prob(model, cur, cls))
        out["deletion_curve"] = xs
        out["deletion_auc"] = float(np.trapezoid(xs, dx=1.0 / steps))
    if mode in ("both", "insertion"):
        xs = []
        for s in range(steps + 1):
            k = int(round(s / steps * n_px))
            cur = blur.clone()
            if k:
                idx = order[:k]
                m = torch.zeros(n_px, dtype=torch.bool, device=x.device)
                m[torch.as_tensor(idx.copy(), device=x.device)] = True
                m = m.reshape(cam.shape)
                cur[:, :, m] = base[:, :, m]
            xs.append(_prob(model, cur, cls))
        out["insertion_curve"] = xs
        out["insertion_auc"] = float(np.trapezoid(xs, dx=1.0 / steps))
    return out
