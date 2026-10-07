"""Reference ternary rounding (numpy) and the GDN value-head regrouping.

Used by tools/forensics.py and the tests; tbr/quant.py (torch, S4) must match `rtn` bit for bit.
"""
from __future__ import annotations

import numpy as np

from .pack import GROUP

RULES = ("absmean", "twn", "mse")
MSE_GRID = np.arange(0.3, 1.21, 0.1, dtype=np.float32)      # threshold as a fraction of mean|w|


def rtn(w: np.ndarray, rule: str = "mse", group: int = GROUP):
    """Round-to-nearest ternary of w [rows, in], one scale per `group` consecutive inputs.

    absmean  s = mean|w|,            t = clip(round(w / s), -1, 1)          (BitNet b1.58, per group)
    twn      Δ = 0.7·mean|w|,        t = sign(w)·[|w| > Δ], s = mean|w| over the support  (Li & Liu 2016)
    mse      Δ ∈ MSE_GRID·mean|w| minimising the group's squared error, s = mean|w| over the support
    Returns trits int8 [rows, in] and scales float32 [rows, in/group].
    """
    rows = w.shape[0]
    g = np.asarray(w, dtype=np.float32).reshape(rows, -1, group)
    a = np.abs(g)
    m = a.mean(-1, keepdims=True)
    if rule == "absmean":
        s = m
        t = np.clip(np.round(g / np.maximum(s, 1e-12)), -1, 1)
    elif rule in ("twn", "mse"):
        if rule == "twn":
            d = 0.7 * m
        else:
            best = d = None
            for f in MSE_GRID:
                dd = f * m
                mask = a > dd
                ss = (a * mask).sum(-1, keepdims=True) / np.maximum(mask.sum(-1, keepdims=True), 1)
                err = (((a - ss) ** 2) * mask).sum(-1, keepdims=True) + ((a ** 2) * ~mask).sum(-1, keepdims=True)
                if best is None:
                    best, d = err, dd
                else:
                    better = err < best
                    best, d = np.where(better, err, best), np.where(better, dd, d)
        mask = a > d
        t = np.sign(g) * mask
        s = (a * mask).sum(-1, keepdims=True) / np.maximum(mask.sum(-1, keepdims=True), 1)
    else:
        raise ValueError(rule)
    return t.astype(np.int8).reshape(rows, -1), s[..., 0].astype(np.float32)


def vperm(nv: int, nk: int, unit: int) -> np.ndarray:
    """The pack runtime's value-head regrouping (runtime.py `vperm`): interleaved -> grouped by key head."""
    return np.arange(nv * unit).reshape(nv // nk, nk, unit).transpose(1, 0, 2).reshape(-1)


def inverse(p: np.ndarray) -> np.ndarray:
    q = np.empty_like(p)
    q[p] = np.arange(len(p))
    return q
