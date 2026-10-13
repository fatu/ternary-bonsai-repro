"""S4: ternary quantization in Bonsai's format, as torch modules.

A `TernaryLinear` / `TernaryEmbedding` keeps the model's own weight (bf16, original basis) as the latent parameter and
computes, every forward,

    w_eff = fold( STE( Q( rotate(W, signs) ) ), signs )           (= tbr.pack's fold / rotate, in torch)

where Q is round-to-nearest ternary per group of 128 inputs in the rotated basis (rule `bonsai` = what S1 found in the
pack: support |w| > 0.5·mean|w|, scale = mean|w| over the support, f16-rounded) and STE passes the gradient straight
through Q. The surrounding model never changes — `F.linear(x, w_eff)` — so HF, FSDP2 and `save_pretrained` work as-is;
`materialize()` turns the modules back into stock Linear / Embedding holding w_eff (the dense checkpoint the S3 harness
scores), and `export_pack()` writes our trits / scales / signs in the pack's own layout, so `tools/forensics.py` can
compare them with Bonsai's trit for trit.

Coverage `v2` (Bonsai's list): every Linear of the language model incl. GDN in_proj_qkv / in_proj_z / out_proj,
attention q/k/v/o, MLP gate/up/down, lm_head, and embed_tokens — keeping in_proj_a/b, conv1d, A_log, dt_bias and the
norms in full precision. `v1` also ternarizes in_proj_a/b (the ablation arm of R2). Vision tower and MTP are never touched.
One ±1 sign vector per input width, shared by every module of that width (seeded, or Bonsai's own via `load_signs`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pack import BLOCK, GROUP, pack_codes
from .ternary import MSE_GRID

RULES = ("absmean", "twn", "mse", "bonsai")
FP_V2 = ("in_proj_a", "in_proj_b")                 # the GDN state path Bonsai keeps in full precision
NEVER = ("visual", "vision_tower", "vision_model", "mtp")


# ── the arithmetic (float32, matches tbr.pack / tbr.ternary) ───────────────────────────────────────────────────────

def fwht(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Normalised Sylvester Walsh-Hadamard transform over blocks of the last axis."""
    shape = x.shape
    if shape[-1] % block:
        raise ValueError(f"width {shape[-1]} not divisible by block {block}")
    y = x.float().reshape(-1, block)
    h = 1
    while h < block:
        y = y.reshape(-1, block // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack([a + b, a - b], dim=2)
        h *= 2
    return y.reshape(shape) / math.sqrt(block)


def rotate(w: torch.Tensor, signs: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Original basis → Bonsai's rotated basis: FWHT(w ⊙ signs)."""
    return fwht(w.float() * signs.float(), block)


def fold(w_rot: torch.Tensor, signs: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Rotated basis → original basis: FWHT(w_rot) ⊙ signs (H and diag(signs) are involutions)."""
    return fwht(w_rot, block) * signs.float()


def ternary(w_rot: torch.Tensor, rule: str = "bonsai", group: int = GROUP, f16_scales: bool = True):
    """Round-to-nearest ternary of w_rot [rows, in]: trits (float ∈ {-1, 0, 1}) [rows, in], scales [rows, in/group].

    absmean  s = mean|w|,            t = clip(round(w / s), -1, 1)                      (BitNet b1.58)
    twn      Δ = 0.7·mean|w|,        t = sign(w)·[|w| > Δ], s = mean|w| over the support  (Li & Liu 2016)
    mse      Δ ∈ MSE_GRID·mean|w| minimising the group's squared error, s = support mean
    bonsai   Δ = 0.5·mean|w| (absmean's support), s = support mean — the pack's own quantizer (S1 verdict)
    Scales are rounded to f16 when f16_scales (the pack stores f16), so w_eff here == the exported pack dequantised.
    """
    rows = w_rot.shape[0]
    if w_rot.shape[1] % group:
        raise ValueError(f"width {w_rot.shape[1]} not divisible by group {group}")
    g = w_rot.float().reshape(rows, -1, group)
    a = g.abs()
    m = a.mean(-1, keepdim=True)
    if rule == "absmean":
        s = m
        t = torch.clamp(torch.round(g / s.clamp_min(1e-12)), -1, 1)
    elif rule in ("twn", "mse", "bonsai"):
        if rule == "twn":
            d = 0.7 * m
        elif rule == "bonsai":
            d = 0.5 * m
        else:
            best = d = None
            for f in MSE_GRID:
                dd = float(f) * m
                mask = a > dd
                ss = (a * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
                err = (((a - ss) ** 2) * mask).sum(-1, keepdim=True) + ((a ** 2) * ~mask).sum(-1, keepdim=True)
                if best is None:
                    best, d = err, dd
                else:
                    better = err < best
                    best, d = torch.where(better, err, best), torch.where(better, dd, d)
        mask = a > d
        t = torch.sign(g) * mask
        s = (a * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
    else:
        raise ValueError(rule)
    s = s[..., 0]
    if f16_scales:
        s = s.half().float()
    return t.reshape(rows, -1), s


def dequant(trits: torch.Tensor, scales: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    rows = trits.shape[0]
    return (trits.float().reshape(rows, -1, group) * scales.float()[..., None]).reshape(rows, -1)


# ── the modules ────────────────────────────────────────────────────────────────────────────────────────────────────

class _Ternary(nn.Module):
    """Shared weight path. `weight` is the model's own Parameter (latent, original basis); `signs` a ±1 buffer or None
    (no rotation — R0 of the ladder, or a width that is not a multiple of the block)."""

    def _setup(self, weight: nn.Parameter, signs, rule: str, group: int, block: int):
        self.weight = weight
        self.register_buffer("signs", None if signs is None else signs.to(weight.device).float(), persistent=False)
        self.rule, self.group, self.block, self.enabled = rule, group, block, True

    def rotated(self) -> torch.Tensor:
        w = self.weight.float()
        return w if self.signs is None else rotate(w, self.signs, self.block)

    @torch.no_grad()
    def quantize(self):
        """(trits, scales) of the current latent weight in the rotated basis — what the pack would store."""
        return ternary(self.rotated(), self.rule, self.group)

    def effective_weight(self) -> torch.Tensor:
        if not self.enabled:
            return self.weight
        w_rot = self.rotated()
        with torch.no_grad():
            q = dequant(*ternary(w_rot, self.rule, self.group), self.group)
        w_q = w_rot + (q - w_rot).detach()                        # value q, gradient of the identity (STE)
        w_eff = w_q if self.signs is None else fold(w_q, self.signs, self.block)
        return w_eff.to(self.weight.dtype)

    def extra_repr(self) -> str:
        o, i = self.weight.shape
        return (f"{o}×{i}, rule={self.rule}, group={self.group}, "
                f"{'rotated block ' + str(self.block) if self.signs is not None else 'unrotated'}, "
                f"{'on' if self.enabled else 'off'}")


class TernaryLinear(_Ternary):
    def __init__(self, linear: nn.Linear, signs, rule="bonsai", group=GROUP, block=BLOCK):
        super().__init__()
        self._setup(linear.weight, signs, rule, group, block)
        self.bias = linear.bias
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)


class TernaryEmbedding(_Ternary):
    def __init__(self, emb: nn.Embedding, signs, rule="bonsai", group=GROUP, block=BLOCK):
        super().__init__()
        self._setup(emb.weight, signs, rule, group, block)
        self.num_embeddings, self.embedding_dim, self.padding_idx = emb.num_embeddings, emb.embedding_dim, emb.padding_idx

    def forward(self, ids):
        return F.embedding(ids, self.effective_weight(), self.padding_idx)


# ── model surgery ──────────────────────────────────────────────────────────────────────────────────────────────────

def covered(name: str, coverage: str) -> bool:
    parts = name.split(".")
    if any(p in NEVER for p in parts):
        return False
    if coverage == "v2" and parts[-1] in FP_V2:
        return False
    if coverage not in ("v1", "v2"):
        raise ValueError(coverage)
    return True


def seeded_signs(width: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed * 1_000_003 + width)
    return (torch.randint(0, 2, (width,), generator=g) * 2 - 1).float()


def convert(model: nn.Module, coverage: str = "v2", rule: str = "bonsai", group: int = GROUP, block: int = BLOCK,
            signs: dict | None = None, seed: int = 0, rotate_weights: bool = True) -> dict:
    """Replace the covered nn.Linear / nn.Embedding modules in place. Returns {"signs": {width: tensor},
    "converted": [names], "unrotated": [names], "skipped": [names]} — share `signs` across runs for comparable trits."""
    signs = dict(signs or {})
    rep = {"signs": signs, "converted": [], "unrotated": [], "skipped": []}
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))]
    for name, mod in targets:
        if not covered(name, coverage):
            rep["skipped"].append(name)
            continue
        width = mod.weight.shape[1]
        if width % group:
            rep["skipped"].append(name)
            continue
        if rotate_weights and width % block == 0:
            if width not in signs:
                signs[width] = seeded_signs(width, seed)
            s = signs[width]
        else:
            s = None
            rep["unrotated"].append(name)
        new = (TernaryLinear if isinstance(mod, nn.Linear) else TernaryEmbedding)(mod, s, rule, group, block)
        parent, child = (name.rsplit(".", 1) if "." in name else ("", name))
        setattr(model.get_submodule(parent) if parent else model, child, new)
        rep["converted"].append(name)
    return rep


def ternary_modules(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, _Ternary)]


def set_quant(model: nn.Module, enabled: bool):
    for _, m in ternary_modules(model):
        m.enabled = enabled


@torch.no_grad()
def materialize(model: nn.Module) -> int:
    """Ternary modules → stock Linear / Embedding holding w_eff (the dense checkpoint for vLLM). Returns the count."""
    n = 0
    for name, m in ternary_modules(model):
        w = m.effective_weight().detach().clone()
        if isinstance(m, TernaryLinear):
            new = nn.Linear(m.in_features, m.out_features, bias=m.bias is not None, device=w.device, dtype=w.dtype)
            new.weight = nn.Parameter(w)
            if m.bias is not None:
                new.bias = m.bias
        else:
            new = nn.Embedding(m.num_embeddings, m.embedding_dim, padding_idx=m.padding_idx, device=w.device, dtype=w.dtype)
            new.weight = nn.Parameter(w)
        parent, child = (name.rsplit(".", 1) if "." in name else ("", name))
        setattr(model.get_submodule(parent) if parent else model, child, new)
        n += 1
    return n


def pack_name(name: str) -> str:
    """HF base tensor name → the pack's name (canon() of both is the same key)."""
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    if name.startswith("model."):
        return "language_model.model." + name[len("model."):]
    if name.startswith("lm_head"):
        return "language_model." + name
    return name


@torch.no_grad()
def export_pack(model: nn.Module, out_dir, block: int = BLOCK, metadata: dict | None = None) -> dict:
    """Our trits / scales / signs in the pack's layout (schema: uint32 codes, f16 scales, biases = −scales, f16 signs,
    hadamard.json) + every other language-model tensor as F32 under its pack name. Head order = HF order (S1: `none`).
    Readable by tools/inspect_ckpt.py and tools/forensics.py; the deployable MLX pack is S12's job."""
    from .st import write_safetensors

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors, widths = {}, {}
    seen = set()
    for name, m in ternary_modules(model):
        if id(m.weight) in seen and isinstance(m, TernaryEmbedding):
            pass                                              # tied weights: both names get their own copy below
        seen.add(id(m.weight))
        t, s = m.quantize()
        codes = (t.cpu().numpy().astype(np.int8) + 1).astype(np.uint8)
        pn = pack_name(name)
        tensors[pn + ".weight"] = pack_codes(codes)
        tensors[pn + ".scales"] = s.cpu().numpy().astype(np.float16)
        tensors[pn + ".biases"] = (-s).cpu().numpy().astype(np.float16)
        if m.signs is not None:
            sv = m.signs.cpu().numpy().astype(np.float16)
            tensors[pn + ".signs"] = sv
            widths[int(sv.shape[0])] = sv
    tern = {n for n, _ in ternary_modules(model)}
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        owner = name.rsplit(".", 1)[0] if "." in name else ""
        if owner in tern or any(seg in NEVER for seg in name.split(".")):
            continue
        tensors[pack_name(name)] = p.detach().float().cpu().numpy()
    write_safetensors(out / "model.safetensors", tensors, metadata=metadata)
    ws = sorted(widths)
    (out / "hadamard.json").write_text(json.dumps({
        "prism.hadamard.block_size": block, "prism.hadamard.sign_widths": ws,
        "prism.hadamard.sign_values": np.concatenate([widths[w] for w in ws]).astype(float).tolist() if ws else []}))
    return {"packed": len(tern), "tensors": len(tensors), "widths": ws}
