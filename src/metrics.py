"""Reliability metrics: calibration, selective prediction, conformal deferral.

Everything here takes plain numpy probabilities (n, C) and integer labels (n,), so the
same code scores every backbone and every evaluation regime.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


# ---------------------------------------------------------------- basic
def accuracy(p, y):
    return float((p.argmax(1) == y).mean())


def macro_f1(p, y):
    return float(f1_score(y, p.argmax(1), average="macro"))


def auroc_ovr(p, y, n_cls):
    """Macro one-vs-rest AUROC; returns nan if a class is absent from y."""
    if len(np.unique(y)) < n_cls:
        return float("nan")
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


def brier(p, y, n_cls):
    oh = np.eye(n_cls)[y]
    return float(((p - oh) ** 2).sum(1).mean())


def nll(p, y, eps=1e-12):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0)).mean())


# ---------------------------------------------------------------- calibration
def ece(p, y, bins=15, adaptive=False):
    """Expected calibration error of the top-1 confidence.

    adaptive=True uses equal-mass bins, which avoids the near-empty high-confidence
    bins that make fixed-width ECE unstable on near-ceiling medical classifiers.
    """
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    n = len(y)
    if adaptive:
        order = np.argsort(conf)
        chunks = np.array_split(order, bins)
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
        chunks = [np.where(idx == b)[0] for b in range(bins)]
    e = 0.0
    for c in chunks:
        if len(c) == 0:
            continue
        e += len(c) / n * abs(correct[c].mean() - conf[c].mean())
    return float(e)


def reliability_bins(p, y, bins=15):
    """Per-bin (confidence, accuracy, count) for reliability diagrams."""
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"center": float((edges[b] + edges[b + 1]) / 2),
                    "conf": float(conf[m].mean()), "acc": float(correct[m].mean()),
                    "count": int(m.sum())})
    return out


def fit_temperature(logits, y, lo=0.05, hi=10.0, iters=200):
    """Temperature scaling by golden-section search on NLL (no autograd needed)."""
    def loss(t):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        return float((np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]).mean())

    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if loss(c) < loss(d):
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
        if abs(b - a) < 1e-6:
            break
    return float((a + b) / 2)


def softmax(z, t=1.0):
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------- selective prediction
def risk_coverage(p, y, n_points=101):
    """Sort by descending confidence; error rate of the most confident fraction."""
    conf, pred = p.max(1), p.argmax(1)
    err = (pred != y).astype(float)
    order = np.argsort(-conf)
    e = err[order]
    cum = np.cumsum(e) / np.arange(1, len(e) + 1)
    cov = np.arange(1, len(e) + 1) / len(e)
    ks = np.unique(np.linspace(0, len(e) - 1, n_points).astype(int))
    return [{"coverage": float(cov[k]), "risk": float(cum[k])} for k in ks]


def aurc(p, y):
    """Area under the risk-coverage curve (lower is better)."""
    conf, pred = p.max(1), p.argmax(1)
    err = (pred != y).astype(float)[np.argsort(-p.max(1))]
    cum = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(cum.mean())


def accuracy_at_coverage(p, y, coverage):
    """Accuracy over the most confident `coverage` fraction."""
    conf, pred = p.max(1), p.argmax(1)
    k = max(1, int(round(coverage * len(y))))
    keep = np.argsort(-conf)[:k]
    return float((pred[keep] == y[keep]).mean())


def coverage_at_risk(p, y, target_risk):
    """Largest coverage whose selective risk stays at or below target_risk."""
    rc = risk_coverage(p, y, n_points=len(y))
    ok = [r["coverage"] for r in rc if r["risk"] <= target_risk]
    return float(max(ok)) if ok else 0.0


# ---------------------------------------------------------------- conformal deferral
def conformal_threshold(p_cal, y_cal, alpha):
    """Split-conformal (LAC) threshold on the calibration scores s = 1 - p[true].

    Returns qhat such that {c : p(c) >= 1 - qhat} covers the truth with
    probability at least 1 - alpha on exchangeable data.
    """
    s = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, q, method="higher"))


def conformal_sets(p, qhat):
    """Boolean (n, C) membership matrix of the LAC prediction sets."""
    return p >= (1.0 - qhat)


def conformal_report(p_cal, y_cal, p_te, y_te, alpha):
    """Deferral behaviour of the gate: a singleton set is auto-reported, anything
    else (empty or ambiguous) is deferred to a radiologist."""
    qhat = conformal_threshold(p_cal, y_cal, alpha)
    S = conformal_sets(p_te, qhat)
    sizes = S.sum(1)
    covered = S[np.arange(len(y_te)), y_te]
    singleton = sizes == 1
    auto = singleton
    acc_auto = float((p_te[auto].argmax(1) == y_te[auto]).mean()) if auto.any() else float("nan")
    return {"alpha": float(alpha), "qhat": qhat,
            "coverage": float(covered.mean()),
            "mean_set_size": float(sizes.mean()),
            "automated_share": float(auto.mean()),
            "accuracy_on_automated": acc_auto,
            "deferred_share": float(1.0 - auto.mean()),
            "error_on_deferred": float((p_te[~auto].argmax(1) != y_te[~auto]).mean()) if (~auto).any() else float("nan")}


# ---------------------------------------------------------------- resampling
def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def bootstrap_metric_ci(p, y, fn, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CI for any metric fn(p, y) by resampling test items."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            vals.append(fn(p[i], y[i]))
        except ValueError:
            continue
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))
