"""R3: Hessian-aware ternary PTQ (GPTQ-style) in Bonsai's rotated basis.

For every TernaryLinear the calibration inputs are rotated the way the runtime rotates them, x_rot = FWHT(x ⊙ signs),
and H = 2/n Σ x_rot x_rotᵀ is accumulated (float32, [in, in]). Columns of W_rot are then quantized left to right with
OBS error feedback (Frantar et al. 2022): q_j = Q(w_j) with the group's ternary scale fixed when its first column is
reached, err = (w_j − q_j) / [H⁻¹]_jj, W[:, j+1:] −= err · [H⁻¹]_{j, j+1:}, in lazy blocks of `blocksize` columns.
Q is `tbr.quant.ternary` with the chosen rule, so the result is still a plain ternary pack (trits + f16 scales);
the module's latent weight is set to fold(dequant(Q)), which re-quantizes to exactly the same trits (the rule is
idempotent on {−s, 0, s} groups), so materialize / export_pack / training can proceed unchanged.

Non-sequential by design: Hessians come from one pass of the *full-precision* model (quant off), not from
already-quantized earlier layers — simpler, architecture-agnostic (hooks only), and the usual first GPTQ variant;
"true-sequential" and act-order are R3 variants for later. Embeddings have no input Hessian and stay RTN.
"""
from __future__ import annotations

import math

import torch

from .quant import TernaryLinear, dequant, fold, fwht, ternary, ternary_modules


class HessianHook:
    """Accumulates H = 2/n Σ x_rot x_rotᵀ for one TernaryLinear from its forward inputs."""

    def __init__(self, module: TernaryLinear):
        self.m = module
        self.H = torch.zeros(module.in_features, module.in_features, dtype=torch.float32, device=module.weight.device)
        self.n = 0
        self.handle = module.register_forward_pre_hook(self)

    def __call__(self, module, args):
        x = args[0].detach().reshape(-1, module.in_features).float()
        if module.signs is not None:
            x = fwht(x * module.signs, module.block)
        n = x.shape[0]
        self.H.mul_(self.n / (self.n + n))
        self.H.add_(x.T @ x, alpha=2.0 / (self.n + n))
        self.n += n

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def gptq_ternary(w_rot: torch.Tensor, H: torch.Tensor, rule: str = "bonsai", group: int = 128, blocksize: int = 128,
                 percdamp: float = 0.01):
    """GPTQ column loop on w_rot [out, in] with Hessian H [in, in] → (trits, scales) like tbr.quant.ternary.
    Columns are processed in order; each group's scale is fixed by `rule` from the (error-updated) weights when its
    first column is reached, then every column is rounded to the nearest level of {−s, 0, s} — the grid rounding OBS
    assumes (the rule's own threshold only chooses the scale). Rules `bonsai` and `twn` re-quantize a {−s, 0, s} group
    to the same trits and scale (idempotent), which apply_gptq relies on; `absmean` / `mse` do not and are refused."""
    if rule not in ("bonsai", "twn"):
        raise ValueError(f"gptq needs an idempotent scale rule (bonsai, twn), not {rule}")
    blocksize = max(blocksize, group) // group * group       # every group lies inside one lazy block
    W = w_rot.float().clone()
    out, cols = W.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(H))
    H += torch.eye(cols, device=H.device) * damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)        # upper Cholesky of H⁻¹: the rows used by the OBS update
    T = torch.zeros_like(W)
    S = torch.zeros(out, cols // group, dtype=torch.float32, device=W.device)
    s_cur = None
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1, Err1 = W[:, i1:i2].clone(), torch.zeros(out, i2 - i1, device=W.device)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            j = i1 + i
            if j % group == 0:                              # new group: scale from the current (updated) weights
                _, s_cur = ternary(W1[:, i:i + group], rule, group)
                S[:, j // group] = s_cur[:, 0]
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / s_cur[:, 0].clamp_min(1e-12)), -1, 1)
            T[:, j] = q
            err = (w - q * s_cur[:, 0]) / d
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return T, S


@torch.no_grad()
def apply_gptq(model, calib: list[torch.Tensor], rule: str = "bonsai", blocksize: int = 128, percdamp: float = 0.01,
               log=print) -> dict:
    """Calibrate every TernaryLinear of `model` (already converted) on `calib` (list of [1, T] token-id tensors on the
    model's device) and overwrite its latent weight with the GPTQ solution. Returns {name: ‖ΔW‖/‖W‖ in the rotated basis}."""
    mods = [(n, m) for n, m in ternary_modules(model) if isinstance(m, TernaryLinear)]
    was = {n: m.enabled for n, m in mods}
    for _, m in mods:
        m.enabled = False                                   # Hessians from the full-precision model
    hooks = {n: HessianHook(m) for n, m in mods}
    model.eval()
    for ids in calib:
        model(input_ids=ids, use_cache=False)
    for h in hooks.values():
        h.remove()
    rel = {}
    for k, (n, m) in enumerate(mods):
        h = hooks[n]
        w_rot = m.rotated()
        t, s = gptq_ternary(w_rot, h.H, rule, m.group, blocksize, percdamp)
        q = dequant(t, s, m.group)
        rel[n] = float((q - w_rot).norm() / w_rot.norm().clamp_min(1e-12))
        w_new = q if m.signs is None else fold(q, m.signs, m.block)
        m.weight.data.copy_(w_new.to(m.weight.dtype))
        m.enabled = was[n]
        del h.H
        if log and (k % 20 == 0 or k == len(mods) - 1):
            log(f"  gptq {k + 1}/{len(mods)} {n}: rel err {rel[n]:.3f} (n={h.n:,} rows)")
    return rel
