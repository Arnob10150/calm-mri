"""CALM-MRI and the baseline backbones.

CALM-MRI keeps a lightweight ImageNet-pretrained CNN trunk and adds a multi-scale
gated fusion head: the last three stages are pooled, projected to a common width, and
combined with input-dependent softmax gates. Deep stages carry tumour-class evidence
while shallower stages carry texture/edge evidence, and which one matters varies by
slice; a fixed concatenation forces one compromise weighting for every image, so we
let the network choose per image. The head costs well under 1M parameters, which keeps
the model deployable on the kind of hardware a resource-limited radiology unit has.

Calibration (temperature) and the conformal deferral gate are post-hoc and live in
src/metrics.py, so they apply identically to every baseline.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Baselines span lightweight CNNs, a modern CNN, and two transformers, so the
# benchmark is not CNN-only (a 2026 reviewer expectation).
BASELINES = {
    "mobilenetv3_small": "mobilenetv3_small_100.lamb_in1k",
    "resnet18": "resnet18.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    # architectures reported by the journal's accepted brain-tumor papers, re-run here
    # under the patient-grouped protocol so their numbers become comparable
    "resnet50": "resnet50.a1_in1k",
    "densenet121": "densenet121.ra_in1k",
    "vgg16": "vgg16_bn.tv_in1k",
    "xception": "legacy_xception.tf_in1k",
    "inception_v3": "inception_v3.tv_in1k",
    "mobilenetv2": "mobilenetv2_100.ra_in1k",
}


class GatedMultiScaleHead(nn.Module):
    """Pool each scale, project to `dim`, and mix with input-dependent softmax gates."""

    def __init__(self, in_chs, n_classes, dim=256, p_drop=0.3):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(c, dim), nn.BatchNorm1d(dim), nn.SiLU())
                                   for c in in_chs])
        self.gate = nn.Sequential(nn.Linear(sum(in_chs), 64), nn.SiLU(), nn.Linear(64, len(in_chs)))
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, feats):
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        w = torch.softmax(self.gate(torch.cat(pooled, dim=1)), dim=1)      # (B, S)
        z = torch.stack([p(x) for p, x in zip(self.proj, pooled)], dim=1)  # (B, S, dim)
        fused = (w.unsqueeze(-1) * z).sum(1)
        return self.fc(self.drop(fused)), w


class CALMNet(nn.Module):
    """Lightweight trunk + gated multi-scale head. `fusion='concat'|'last'` are ablations."""

    def __init__(self, n_classes, trunk="efficientnet_b0.ra_in1k", fusion="gated",
                 dim=256, p_drop=0.3, pretrained=True, n_scales=3):
        super().__init__()
        self.fusion = fusion
        self.trunk = timm.create_model(trunk, pretrained=pretrained, features_only=True)
        chs = self.trunk.feature_info.channels()[-n_scales:]
        self.n_scales = n_scales
        if fusion == "gated":
            self.head = GatedMultiScaleHead(chs, n_classes, dim, p_drop)
        elif fusion == "concat":
            self.head = nn.Sequential(nn.Linear(sum(chs), dim), nn.BatchNorm1d(dim), nn.SiLU(),
                                      nn.Dropout(p_drop), nn.Linear(dim, n_classes))
        elif fusion == "last":
            self.head = nn.Sequential(nn.Linear(chs[-1], dim), nn.BatchNorm1d(dim), nn.SiLU(),
                                      nn.Dropout(p_drop), nn.Linear(dim, n_classes))
        else:
            raise ValueError(fusion)
        self.last_gates = None

    def forward(self, x):
        feats = self.trunk(x)[-self.n_scales:]
        if self.fusion == "gated":
            logits, w = self.head(feats)
            self.last_gates = w.detach()
            return logits
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        v = torch.cat(pooled, dim=1) if self.fusion == "concat" else pooled[-1]
        return self.head(v)


def build_baseline(name, n_classes, pretrained=True):
    return timm.create_model(BASELINES[name], pretrained=pretrained, num_classes=n_classes)


# CALM-L keeps the gated multi-scale head and swaps in the strongest trunk in the
# benchmark, which separates the head's contribution from the trunk's capacity.
CALM_TRUNKS = {"calm": "efficientnet_b0.ra_in1k", "calm_l": "convnext_tiny.fb_in1k"}


def build(name, n_classes, pretrained=True, **kw):
    if name.startswith("calm"):
        kw.setdefault("trunk", CALM_TRUNKS.get(name, CALM_TRUNKS["calm"]))
        return CALMNet(n_classes, pretrained=pretrained, **kw)
    return build_baseline(name, n_classes, pretrained=pretrained)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
