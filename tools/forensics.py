#!/usr/bin/env python3
"""S1 forensics: how far are Bonsai's ternary weights from a round-to-nearest of the base model?

    python tools/forensics.py checkpoints/bonsai2-27b-mlx checkpoints/qwen3.8-27b              # 8 layers + embed/head
    python tools/forensics.py checkpoints/bonsai2-27b-mlx checkpoints/qwen3.8-27b --skip-embed  # faster first look
    python tools/forensics.py ... --all                                                        # every layer
    python tools/forensics.py ... --regroup none|vperm|inverse                                 # skip the auto test

For every packed module: W_rot = rotate(W_base, signs) — the pack's own basis — rounded to ternary under
three scale rules, compared with the pack's trits t_b and scales s_b.
  flip%      trits that differ from Bonsai's, per rule      low → PTQ-dominant recipe; high → long training
  zero%      share of zeros, Bonsai vs the MSE rule          which threshold rule they used
  agree%     sign agreement where both are non-zero
  s_b/s*     Bonsai's scale over the least-squares scale for Bonsai's own trits on the base weights
             ≈ 1 with a tight spread → scales fit to the base; drifting → weights moved / scales learned
  rel_err    ‖W_rot − s_b·t_b‖ / ‖W_rot‖ (b) vs the MSE rule's own error (q); b/q > 1 means Bonsai's
             weights are NOT the nearest ternary point to the base — training moved them; by how much
             is the training-intensity clue.
GDN regrouping: the pack stores value heads grouped (`gdn_activation_layout: grouped`). Before the main pass
the script tries none / vperm / inverse on four linear-attention layers (V rows of in_proj_qkv, rows of
in_proj_z, input columns of out_proj) and keeps the order with the lowest flip rate.
Three tables on screen (photograph them); per-module numbers in runs/forensics/<pack>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tbr.pack import GROUP, canon, layer_pattern, load_signs, rotate, trits as pack_trits  # noqa: E402
from tbr.st import Checkpoint  # noqa: E402
from tbr.ternary import RULES, inverse, rtn, vperm  # noqa: E402

CHUNK = 2048            # rows per pass (rotation and rounding are row-independent)
SUB = 20000             # scale ratios kept per module for the medians
TYPES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "lm_head", "embed_tokens"]
DEFAULT_LAYERS = "0,3,16,19,32,35,48,51"      # four linear-attention + four full-attention layers across depth
HEAD = ("                           n   flip% abs   twn   mse   zero% b   mse  agree%   s_b/s*  med [p10,p90]"
        "   rel_err  b     q   b/q")

_G: dict = {}


def _init(pack_dir, base_dir):
    _G["pack_dir"], _G["base_dir"] = pack_dir, base_dir


def _state():
    if "pack" not in _G:
        pack, base = Checkpoint(_G["pack_dir"]), Checkpoint(_G["base_dir"])
        hj = json.loads((Path(_G["pack_dir"]) / "hadamard.json").read_text())
        tc = json.loads((Path(_G["base_dir"]) / "config.json").read_text())
        tc = tc.get("text_config", tc)
        _G.update(pack=pack, base=base, signs=load_signs(_G["pack_dir"]),
                  block=int(hj.get("prism.hadamard.block_size", 1024)),
                  base_keys={canon(k): k for k in base.keys() if canon(k)},
                  nv=int(tc["linear_num_value_heads"]), nk=int(tc["linear_num_key_heads"]),
                  hd=int(tc["linear_value_head_dim"]), hk=int(tc["linear_key_head_dim"]),
                  n_layers=int(tc.get("num_hidden_layers", 0)))
    return _G


def gdn_role(key: str):
    """Which axis of this module carries the value heads: V rows of qkv, rows of z, input columns of out_proj."""
    if key.endswith("linear_attn.in_proj_qkv.weight"):
        return "rows_v"
    if key.endswith("linear_attn.in_proj_z.weight"):
        return "rows"
    if key.endswith("linear_attn.out_proj.weight"):
        return "cols"
    return None


def _sub(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    return x[::max(1, x.size // SUB)][:SUB].tolist()


def analyze(module: str, key: str, regroup: str, v_only: bool = False) -> dict:
    """Statistics for one packed module under one regrouping hypothesis (chunked over rows)."""
    G = _state()
    pack, base = G["pack"], G["base"]
    bk = G["base_keys"][key]
    rows_total, words = pack.info(module + ".weight")["shape"]
    signs = G["signs"][words * 16]
    role = gdn_role(key)
    qk = 2 * G["nk"] * G["hk"]
    perm = None
    if role and regroup != "none" and G["nv"] != G["nk"]:
        p = vperm(G["nv"], G["nk"], G["hd"])
        perm = p if regroup == "vperm" else inverse(p)
    off = qk if role == "rows_v" else 0
    acc = dict(key=key, regroup=regroup, n=0, bad=0, zeros_b=0, both=0, agree=0, err_b=0.0, err_q=0.0, norm=0.0,
               flips={r: 0 for r in RULES}, zeros={r: 0 for r in RULES}, ratio={r: [] for r in RULES}, refit=[])
    for r0 in range(off if v_only else 0, rows_total, CHUNK):
        r1 = min(r0 + CHUNK, rows_total)
        t_b = pack_trits(np.asarray(pack.get(module + ".weight", rows=slice(r0, r1))))
        s_b = pack.get(module + ".scales", float32=True, rows=slice(r0, r1))
        if perm is not None and role != "cols":
            idx = np.arange(r0, r1)
            src, v = idx.copy(), idx >= off              # pack row off+i holds base row off+perm[i]
            src[v] = off + perm[idx[v] - off]
            w = base.get(bk, rows=src)
        else:
            w = base.get(bk, rows=slice(r0, r1))
            if perm is not None:
                w = w[:, perm]                           # pack column j holds base column perm[j]
        w_rot = rotate(w, signs, G["block"])
        g = w_rot.reshape(w_rot.shape[0], -1, GROUP)
        tb = t_b.reshape(g.shape)
        acc["n"] += t_b.size
        acc["bad"] += int((t_b == 2).sum())              # code 3 — must stay 0
        acc["zeros_b"] += int((t_b == 0).sum())
        for rule in RULES:
            t, s = rtn(w_rot, rule)
            acc["flips"][rule] += int((t != t_b).sum())
            acc["zeros"][rule] += int((t == 0).sum())
            acc["ratio"][rule] += _sub(s_b / np.maximum(s, 1e-12))
            if rule == "mse":
                both = (t != 0) & (t_b != 0)
                acc["both"] += int(both.sum())
                acc["agree"] += int((both & (t == t_b)).sum())
                acc["err_q"] += float(((g - s[..., None] * t.reshape(g.shape)) ** 2).sum())
        s_star = (g * tb).sum(-1) / np.maximum((tb.astype(np.int32) ** 2).sum(-1), 1)
        ok = s_star > 0
        acc["refit"] += _sub(s_b[ok] / s_star[ok])
        acc["err_b"] += float(((g - s_b[..., None] * tb) ** 2).sum())
        acc["norm"] += float((g ** 2).sum())
    return acc


def merge(accs):
    out = dict(n=0, bad=0, zeros_b=0, both=0, agree=0, err_b=0.0, err_q=0.0, norm=0.0,
               flips={r: 0 for r in RULES}, zeros={r: 0 for r in RULES}, ratio={r: [] for r in RULES}, refit=[])
    for a in accs:
        for k in ("n", "bad", "zeros_b", "both", "agree", "err_b", "err_q", "norm"):
            out[k] += a[k]
        for r in RULES:
            out["flips"][r] += a["flips"][r]
            out["zeros"][r] += a["zeros"][r]
            out["ratio"][r] += a["ratio"][r]
        out["refit"] += a["refit"]
    return out


def stats(a):
    n = max(a["n"], 1)
    refit = np.asarray(a["refit"]) if a["refit"] else np.array([np.nan])
    rel_b = float(np.sqrt(a["err_b"] / max(a["norm"], 1e-30)))
    rel_q = float(np.sqrt(a["err_q"] / max(a["norm"], 1e-30)))
    return dict(n=a["n"], flip={r: 100 * a["flips"][r] / n for r in RULES}, zero_b=100 * a["zeros_b"] / n,
                zero_q=100 * a["zeros"]["mse"] / n, agree=100 * a["agree"] / max(a["both"], 1),
                refit_med=float(np.median(refit)), refit_p10=float(np.percentile(refit, 10)),
                refit_p90=float(np.percentile(refit, 90)),
                ratio_med={r: float(np.median(a["ratio"][r])) if a["ratio"][r] else float("nan") for r in RULES},
                rel_b=rel_b, rel_q=rel_q, bad=a["bad"])


def fmt(label, st):
    f = st["flip"]
    return (f"  {label:<22} {st['n'] / 1e6:7.1f}M   {f['absmean']:5.1f} {f['twn']:5.1f} {f['mse']:5.1f}"
            f"   {st['zero_b']:5.1f} {st['zero_q']:5.1f}  {st['agree']:5.1f}"
            f"   {st['refit_med']:5.2f} [{st['refit_p10']:4.2f},{st['refit_p90']:4.2f}]"
            f"   {st['rel_b']:5.3f} {st['rel_q']:5.3f} {st['rel_b'] / max(st['rel_q'], 1e-9):5.2f}")


def packed_modules(pack: Checkpoint):
    out = {}
    for k in pack.keys():
        if k.endswith(".weight") and pack.info(k)["dtype"] == "U32":
            m = k[:-len(".weight")]
            if m + ".scales" in pack.where and m + ".biases" in pack.where and canon(k):
                out[m] = canon(k)
    return out


def layer_of(key: str):
    return int(key.split(".")[1]) if key.startswith("layers.") else None


def type_of(key: str):
    p = layer_pattern(key)[:-len(".weight")]
    return p[len("layers.N."):] if p.startswith("layers.N.") else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("base")
    ap.add_argument("--layers", default=DEFAULT_LAYERS, help="comma-separated layer indices")
    ap.add_argument("--all", action="store_true", help="every layer")
    ap.add_argument("--skip-embed", action="store_true", help="skip embed_tokens and lm_head (the two biggest)")
    ap.add_argument("--regroup", choices=["auto", "none", "vperm", "inverse"], default="auto")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="json path (default runs/forensics/<pack>.json)")
    a = ap.parse_args()
    t0 = time.time()
    _init(a.pack, a.base)
    G = _state()
    pack = G["pack"]
    mods = packed_modules(pack)
    n_layers = G["n_layers"] or (max(layer_of(k) for k in mods.values() if layer_of(k) is not None) + 1)
    layers = set(range(n_layers)) if a.all else {int(x) for x in a.layers.split(",")}
    chosen = {m: k for m, k in mods.items()
              if (layer_of(k) in layers) or (layer_of(k) is None and not a.skip_embed)}
    missing = [k for k in chosen.values() if k not in G["base_keys"]]
    if missing:
        sys.exit(f"packed modules without a base tensor: {missing[:5]}")
    print(f"{Path(a.pack).name} vs {Path(a.base).name}: {len(chosen)}/{len(mods)} packed modules, layers "
          f"{sorted(layers) if not a.all else 'all'}, block {G['block']}, GDN v/k heads {G['nv']}/{G['nk']} × {G['hd']}")

    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.pack, a.base)) as ex:
        regroup = a.regroup
        if regroup == "auto" and G["nv"] != G["nk"]:
            lin = sorted({layer_of(k) for k in chosen.values() if gdn_role(k)}, key=lambda x: x)[:4]
            gdn = [(m, k) for m, k in chosen.items() if gdn_role(k) and layer_of(k) in lin]
            futs = {(v, m): ex.submit(analyze, m, k, v, True) for v in ("none", "vperm", "inverse") for m, k in gdn}
            print(f"\n== C. GDN value-head order (layers {lin}; flip% under the best-fitting rule; V rows / rows / input columns)")
            print("  order      in_proj_qkv[V]   in_proj_z   out_proj   mean")
            best = None
            for v in ("none", "vperm", "inverse"):
                by = {}
                for m, k in gdn:
                    by.setdefault(gdn_role(k), []).append(futs[(v, m)].result())
                f = {r: min(stats(merge(by[r]))["flip"].values()) for r in ("rows_v", "rows", "cols") if r in by}
                mean = float(np.mean(list(f.values())))
                print(f"  {v:<9} {f.get('rows_v', float('nan')):13.1f} {f.get('rows', float('nan')):11.1f}"
                      f" {f.get('cols', float('nan')):10.1f} {mean:7.1f}")
                if best is None or mean < best[1]:
                    best = (v, mean)
            regroup = best[0]
            print(f"  chosen regroup = {regroup}")
        elif regroup == "auto":
            regroup = "none"
            print("\n== C. GDN heads not grouped (v == k heads): no regrouping question")

        futs = {m: ex.submit(analyze, m, k, regroup) for m, k in chosen.items()}
        results = {m: futs[m].result() for m in chosen}

    by_type = {}
    by_depth = {"early": [], "mid": [], "late": []}
    for m, acc in results.items():
        by_type.setdefault(type_of(acc["key"]), []).append(acc)
        L = layer_of(acc["key"])
        if L is not None:
            by_depth["early" if L < n_layers / 3 else "mid" if L < 2 * n_layers / 3 else "late"].append(acc)
    print(f"\n== A. by module type (regroup = {regroup})")
    print(HEAD)
    for t in TYPES:
        if t in by_type:
            print(fmt(t, stats(merge(by_type[t]))))
    print(fmt("ALL", stats(merge(list(results.values())))))
    print(f"\n== B. by depth (layers only; n_layers = {n_layers})")
    print(HEAD)
    for d, accs in by_depth.items():
        if accs:
            print(fmt(d, stats(merge(accs))))
    bad = sum(acc["bad"] for acc in results.values())
    if bad:
        print(f"\n  ⚠ {bad} codes equal to 3 (unused value) — the pack is not pure ternary where that happens")
    out = Path(a.out) if a.out else Path("runs/forensics") / f"{Path(a.pack).name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pack": a.pack, "base": a.base, "regroup": regroup, "layers": sorted(layers),
                               "modules": {acc["key"]: stats(acc) for acc in results.values()}}, indent=1))
    print(f"\n{len(results)} modules · {time.time() - t0:.0f} s · {out}")


if __name__ == "__main__":
    main()
