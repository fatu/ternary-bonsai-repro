"""B1 step 1: list what the Bonsai pack stores, in one photographable screen.

    python tools/inspect_ckpt.py checkpoints/bonsai2-27b-mlx
    python tools/inspect_ckpt.py checkpoints/bonsai2-27b-mlx --base checkpoints/qwen3.8-27b

Sections:
  1. tensors by module pattern (layers.N collapsed): packed (ternary) vs kept-FP, shape, dtype
  2. sign vectors: how many `.signs` tensors, distinct vectors per width, equal to hadamard.json?
  3. sampled packed modules (first and last layer of each type, + embed/lm_head, first --rows rows):
     share of codes 0/1/2/3 (= trit -1/0/+1/unused), max |bias + scale|, scale median [min, max]
  4. (--base) does every packed module have a base tensor of the same [out, in] shape?
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tbr.pack import canon, layer_pattern, load_signs, unpack_codes  # noqa: E402
from tbr.st import Checkpoint  # noqa: E402


def modules(ck: Checkpoint):
    """{packed module name: canonical key} for every U32 weight with scales + biases."""
    out = {}
    for k in ck.keys():
        if k.endswith(".weight") and ck.info(k)["dtype"] == "U32":
            m = k[:-len(".weight")]
            if m + ".scales" in ck.where and m + ".biases" in ck.where:
                out[m] = canon(k)
    return out


def section_patterns(ck, packed):
    rows = defaultdict(lambda: [0, ""])
    other = [0, 0]
    packed_parts = {m + s for m in packed for s in (".weight", ".scales", ".biases", ".signs")}
    for k in ck.keys():
        c = canon(k)
        h = ck.info(k)
        if c is None:
            other[0] += 1
            other[1] += h["data_offsets"][1] - h["data_offsets"][0]
            continue
        if k in packed_parts:
            if not k.endswith(".weight"):
                continue
            r, w = h["shape"]
            sig = f".signs[{ck.info(k[:-7] + '.signs')['shape'][0]}]" if k[:-7] + ".signs" in ck.where else "no signs"
            desc = f"packed  [{r} x {w * 16}]  {sig}"
        else:
            desc = f"FP {h['dtype']:<4} {h['shape']}"
        row = rows[layer_pattern(c)]
        row[0] += 1
        row[1] = desc
    print("== 1. language-model tensors by pattern")
    for p in sorted(rows, key=lambda p: (not rows[p][1].startswith("packed"), p)):
        print(f"  {rows[p][0]:3d} × {p:<44} {rows[p][1]}")
    print(f"  (+ {other[0]} non-LM tensors: vision / other, {other[1] / 1e9:.2f} GB)")


def section_signs(ck, pack_dir):
    by_width = defaultdict(set)
    for k in ck.keys():
        if k.endswith(".signs"):
            v = ck.get(k).astype(np.float32)
            by_width[v.shape[0]].add(v.tobytes())
    n = sum(1 for k in ck.keys() if k.endswith(".signs"))
    print(f"== 2. sign vectors: {n} `.signs` tensors")
    try:
        ref = load_signs(pack_dir)
    except FileNotFoundError:
        ref = {}
    for w in sorted(by_width):
        vals = [np.frombuffer(b, np.float32) for b in by_width[w]]
        pm1 = all(np.isin(v, (-1, 1)).all() for v in vals)
        same = w in ref and len(vals) == 1 and np.array_equal(vals[0], ref[w])
        plus = float((vals[0] > 0).mean())
        print(f"  width {w:6d}: {len(vals)} distinct · ±1 only {pm1} · == hadamard.json {same} · share +1 {plus:.3f}")


def section_samples(ck, packed, n_rows):
    by_pattern = defaultdict(list)
    for m, c in packed.items():
        by_pattern[layer_pattern(c)].append((m, c))
    picks = []
    for p, ms in sorted(by_pattern.items()):
        ms.sort(key=lambda mc: int(mc[1].split(".")[1]) if mc[1].startswith("layers.") else 0)
        picks += [ms[0]] + ([ms[-1]] if len(ms) > 1 else [])
    print(f"== 3. sampled packed modules (first {n_rows} rows)   codes 0/1/2/3 = trit -1/0/+1/unused")
    for m, c in picks:
        w = np.asarray(ck.get(m + ".weight")[:n_rows])
        s = ck.get(m + ".scales", float32=True)[:n_rows]
        b = ck.get(m + ".biases", float32=True)[:n_rows]
        frac = np.bincount(unpack_codes(w).ravel(), minlength=4) / (w.size * 16)
        dev = float(np.abs(b + s).max())
        print(f"  {c[:-len('.weight')]:<34} {frac[0]:.3f} {frac[1]:.3f} {frac[2]:.3f} {frac[3]:.3f}"
              f"  |b+s|max {dev:.1e}  s {np.median(s):.2e} [{s.min():.1e}, {s.max():.1e}]")


def section_base(packed, ck, base_dir):
    base = Checkpoint(base_dir)
    by_key = {canon(k): k for k in base.keys() if canon(k)}
    ok, missing, mismatch = 0, [], []
    for m, c in packed.items():
        bk = by_key.get(c)
        if bk is None:
            missing.append(c); continue
        r, w = ck.info(m + ".weight")["shape"]
        if list(base.info(bk)["shape"]) != [r, w * 16]:
            mismatch.append(f"{c} {base.info(bk)['shape']} vs {[r, w * 16]}")
        else:
            ok += 1
    fp_pack = {canon(k) for k in ck.keys() if canon(k) and not any(k.startswith(m + ".") for m in packed)}
    fp_missing = sorted(fp_pack - set(by_key))
    lm_base = sum(math.prod(base.info(k)["shape"]) for k in by_key.values())
    print(f"== 4. vs base {Path(base_dir).name}: {ok}/{len(packed)} packed modules match [out, in] · "
          f"{len(missing)} missing · {len(mismatch)} shape mismatch · base LM params {lm_base / 1e9:.2f}B")
    for x in (missing + mismatch + [f"FP tensor not in base: {k}" for k in fp_missing])[:8]:
        print(f"  ✗ {x}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--base", help="Qwen3.8-27B directory, for the shape match")
    ap.add_argument("--rows", type=int, default=2048, help="rows sampled per module in section 3")
    a = ap.parse_args()
    ck = Checkpoint(a.pack)
    packed = modules(ck)
    meta = {k: v for f in ck.files for k, v in f.metadata.items()}
    print(f"{Path(a.pack).name}: {len(ck.where)} tensors · {len(packed)} packed modules · metadata {meta}")
    section_patterns(ck, packed)
    section_signs(ck, a.pack)
    section_samples(ck, packed, a.rows)
    if a.base:
        section_base(packed, ck, a.base)


if __name__ == "__main__":
    main()
