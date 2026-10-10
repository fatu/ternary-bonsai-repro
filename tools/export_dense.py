#!/usr/bin/env python3
"""S2: export the Bonsai pack as a dense bf16 checkpoint in the base model's own layout.

    python tools/export_dense.py checkpoints/bonsai2-27b-mlx checkpoints/qwen3.8-27b checkpoints/bonsai2-27b-dense

Every packed (ternary) module → W_eff = fold(dequant(codes, scales, biases), signs)  (= the pack's runtime
function, see tbr/pack.py), then undo the GDN value-head regrouping, then bf16 into the base's tensor name.
Full-precision tensors the pack keeps (GDN state path, norms) come from the pack, aligned to the base's
layout; everything the pack lacks (vision tower, MTP) is copied from the base. Same shard files, same index,
same configs/tokenizer as the base, so stock vLLM / transformers / lm-eval load it unchanged.

--regroup: none | vperm | inverse | auto (default: the order forensics.py chose, read from
runs/forensics/<pack>.json; `none` with a warning if that file is missing).

The report doubles as a forensic: for every kept-FP group it prints how far the pack's values are from the
base's (rel ≈ 0 → Bonsai kept the base values; rel > 0 → they were trained), and which layout transform was
needed (transpose for conv1d, −1 for zero-centred norm weights, none).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tbr.pack import canon, dequant, fold, gdn_role, layer_pattern, load_signs  # noqa: E402
from tbr.st import Checkpoint, SafeTensors, bf16_bits, write_safetensors  # noqa: E402
from tbr.ternary import inverse, vperm  # noqa: E402

CHUNK = 4096
SMALL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json",
               "special_tokens_map.json", "model.safetensors.index.json")
_G: dict = {}


def _init(pack_dir, base_dir, regroup):
    _G.update(pack_dir=pack_dir, base_dir=base_dir, regroup=regroup)


def _state():
    if "pack" not in _G:
        pack = Checkpoint(_G["pack_dir"])
        hj = json.loads((Path(_G["pack_dir"]) / "hadamard.json").read_text())
        tc = json.loads((Path(_G["base_dir"]) / "config.json").read_text())
        tc = tc.get("text_config", tc)
        packed, fp = {}, {}
        for k in pack.keys():
            c = canon(k)
            if c is None:
                continue
            if k.endswith(".weight") and pack.info(k)["dtype"] == "U32":
                packed[c] = k[:-len(".weight")]
            elif not k.endswith((".scales", ".biases", ".signs")):
                fp[c] = k
        _G.update(pack=pack, signs=load_signs(_G["pack_dir"]), block=int(hj.get("prism.hadamard.block_size", 1024)),
                  packed=packed, fp=fp, nv=int(tc["linear_num_value_heads"]), nk=int(tc["linear_num_key_heads"]),
                  hd=int(tc["linear_value_head_dim"]), hk=int(tc["linear_key_head_dim"]))
    return _G


def _perm(role):
    """Index map with 'pack position i holds base position perm[i]' for this role, or None."""
    G = _state()
    if G["regroup"] == "none" or G["nv"] == G["nk"] or role is None:
        return None
    p = vperm(G["nv"], G["nk"], 1 if role == "rows_head" else G["hd"])
    return p if G["regroup"] == "vperm" else inverse(p)


def to_base_layout(a: np.ndarray, key: str) -> np.ndarray:
    """Undo the value-head regrouping: pack layout → base layout."""
    role = gdn_role(key)
    perm = _perm(role)
    if perm is None:
        return a
    out = a.copy()
    if role == "cols":
        out[:, perm] = a                                   # pack column j holds base column perm[j]
    else:
        off = 2 * _G["nk"] * _G["hk"] if role == "rows_v" else 0
        out[off:][perm] = a[off:]                          # pack row off+i holds base row off+perm[i]
    return out


def dense(module: str, key: str) -> np.ndarray:
    """float32 W_eff in the base layout for one packed module (chunked over rows)."""
    G = _state()
    pack = G["pack"]
    rows, words = pack.info(module + ".weight")["shape"]
    signs = G["signs"][words * 16]
    out = np.empty((rows, words * 16), np.float32)
    for r0 in range(0, rows, CHUNK):
        sl = slice(r0, min(r0 + CHUNK, rows))
        d = dequant(np.asarray(pack.get(module + ".weight", rows=sl)), pack.get(module + ".scales", rows=sl),
                    pack.get(module + ".biases", rows=sl))
        out[sl] = fold(d, signs, G["block"])
    return to_base_layout(out, key)


def align_fp(p: np.ndarray, b: np.ndarray, key: str):
    """The pack's FP tensor in the base layout: pick the layout transform that brings it closest to the base.
    Returns (array, transform name, relative distance to the base)."""
    p = to_base_layout(np.asarray(p, dtype=np.float32), key)
    cands = [("as-is", p)]
    if p.ndim == 3 and p.shape != b.shape and p.shape[1:] == b.shape[1:][::-1]:
        cands.append(("transpose", p.transpose(0, 2, 1)))
    if p.shape == b.shape:
        cands.append(("minus-1", p - 1.0))
    cands = [(n, c) for n, c in cands if c.shape == b.shape]
    if not cands:
        raise ValueError(f"{key}: pack shape {p.shape} cannot be aligned to base shape {b.shape}")
    nb = float(np.linalg.norm(b)) or 1.0
    name, arr = min(cands, key=lambda nc: float(np.linalg.norm(nc[1] - b)))
    return arr, name, float(np.linalg.norm(arr - b)) / nb


def _encode(a: np.ndarray, dtype: str):
    if dtype == "BF16":
        return bf16_bits(a)
    return a.astype({"F32": np.float32, "F16": np.float16}[dtype])


def export_shard(shard: str, out_dir: str, fp_source: str) -> dict:
    G = _state()
    base = SafeTensors(Path(G["base_dir"]) / shard)
    tensors, bf16, rep = {}, set(), {"packed": defaultdict(int), "fp": defaultdict(list), "base": 0}
    for name in base.keys():
        dt = base.info(name)["dtype"]
        key = canon(name)
        if key in G["packed"]:
            arr = _encode(dense(G["packed"][key], key), dt)
            rep["packed"][layer_pattern(key)] += 1
        elif key in G["fp"] and fp_source != "base":
            b = base.get(name, float32=True)
            arr, how, rel = align_fp(G["pack"].get(G["fp"][key], float32=True), b, key)
            if rel > 0.5 and fp_source == "auto":            # misaligned beyond any training drift → keep the base's
                arr, how = b, f"BASE (pack rel {rel:.2f})"
            arr = _encode(arr, dt)
            rep["fp"][layer_pattern(key)].append((how, rel))
        else:
            arr = np.asarray(base.get(name))               # unchanged (vision, MTP, or --fp-source base)
            if dt == "BF16":
                arr = bf16_bits(arr)                       # the reader widened bf16 to fp32 — back to the exact bits
            rep["base"] += 1
        if dt == "BF16":
            bf16.add(name)
        tensors[name] = arr
    write_safetensors(Path(out_dir) / shard, tensors, bf16=bf16, metadata={"format": "pt"})
    return {"packed": dict(rep["packed"]), "fp": {k: v for k, v in rep["fp"].items()}, "base": rep["base"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--regroup", choices=["auto", "none", "vperm", "inverse"], default="auto")
    ap.add_argument("--fp-source", choices=["auto", "pack", "base"], default="auto",
                    help="kept-FP tensors from the pack (aligned), the base, or auto (pack unless misaligned)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    regroup = a.regroup
    if regroup == "auto":
        f = Path("runs/forensics") / f"{Path(a.pack).name}.json"
        regroup = json.loads(f.read_text())["regroup"] if f.exists() else "none"
        print(f"regroup = {regroup}" + ("  (from " + str(f) + ")" if f.exists() else "  (⚠ no forensics json: run S1 first)"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = json.loads((Path(a.base) / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx["weight_map"].values()))
    for f in SMALL_FILES:
        if (Path(a.base) / f).exists():
            shutil.copy2(Path(a.base) / f, out / f)
    _init(a.pack, a.base, regroup)
    G = _state()
    print(f"{len(G['packed'])} packed modules · {len(G['fp'])} kept-FP tensors in the pack · {len(shards)} base shards · "
          f"{a.workers} workers")
    packed, fp, from_base = defaultdict(int), defaultdict(list), 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.pack, a.base, regroup)) as ex:
        for i, rep in enumerate(ex.map(export_shard, shards, [str(out)] * len(shards), [a.fp_source] * len(shards))):
            for k, n in rep["packed"].items():
                packed[k] += n
            for k, v in rep["fp"].items():
                fp[k] += v
            from_base += rep["base"]
            print(f"  shard {i + 1}/{len(shards)} done · {time.time() - t0:.0f} s", flush=True)
    print(f"\n== packed → dense ({sum(packed.values())} modules)")
    for k in sorted(packed):
        print(f"  {packed[k]:3d} × {k}")
    print("\n== kept-FP tensors from the pack   (rel = ‖pack − base‖/‖base‖ after alignment: 0 → base values kept)")
    for k in sorted(fp):
        hows = sorted({h for h, _ in fp[k]})
        rels = np.array([r for _, r in fp[k]])
        print(f"  {len(fp[k]):3d} × {k:<44} {'/'.join(hows):<12} rel median {np.median(rels):.2e}  max {rels.max():.2e}")
    print(f"\n== copied from the base: {from_base} tensors (vision tower, MTP, or anything the pack lacks)")
    n_pack = sum(packed.values())
    if n_pack != len(G["packed"]):
        print(f"  ⚠ {len(G['packed']) - n_pack} packed modules had no base tensor")
    gb = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
    print(f"\n{out}: {gb:.1f} GB · {time.time() - t0:.0f} s · next: python tools/verify_download.py {out}  then  tools/ppl.py")


if __name__ == "__main__":
    main()
