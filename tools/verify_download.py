"""B0 check: both checkpoints are on disk and complete.

    python tools/verify_download.py checkpoints/bonsai2-27b-mlx checkpoints/qwen3.8-27b
    python tools/verify_download.py checkpoints/bonsai2-27b-mlx --sha     # + sha256 (about a minute)

Pack (has files.json): every listed file present with the listed size; --sha checks sha256 too.
Base (has model.safetensors.index.json): every shard present, each file exactly the size its own
header implies (catches truncation), tensor bytes vs the index's total_size, parameter counts.
One line per checkpoint, meant to be photographed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tbr.pack import canon  # noqa: E402
from tbr.st import expected_size, read_header  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def check_pack(d: Path, sha: bool) -> tuple[bool, str]:
    files = json.loads((d / "files.json").read_text())
    bad = []
    for name, want in files.items():
        p = d / name
        if not p.exists() or p.stat().st_size != want["size"]:
            bad.append(f"{name}: {'missing' if not p.exists() else 'size ' + str(p.stat().st_size)}")
        elif sha and sha256(p) != want["sha256"]:
            bad.append(f"{name}: sha256 mismatch")
    header, _, _ = read_header(d / "model.safetensors")
    packed = sum(1 for h in header.values() if h["dtype"] == "U32")
    signs = sum(1 for k in header if k.endswith(".signs"))
    gb = (d / "model.safetensors").stat().st_size / 1e9 if (d / "model.safetensors").exists() else 0
    msg = (f"pack  {d.name}: {len(files) - len(bad)}/{len(files)} files ok"
           f"{' · sha256 ok' if sha and not bad else ''} · {gb:.2f} GB · {packed} packed weights · {signs} sign tensors")
    return not bad, msg + "".join(f"\n      ✗ {b}" for b in bad)


def check_base(d: Path) -> tuple[bool, str]:
    idx = json.loads((d / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx["weight_map"].values()))
    bad, data_bytes, lm, total = [], 0, 0, 0
    for s in shards:
        p = d / s
        if not p.exists():
            bad.append(f"{s}: missing"); continue
        if p.stat().st_size != expected_size(p):
            bad.append(f"{s}: truncated ({p.stat().st_size} < {expected_size(p)})"); continue
        header, _, _ = read_header(p)
        for name, h in header.items():
            n = math.prod(h["shape"])
            total += n
            lm += n if canon(name) else 0
            data_bytes += h["data_offsets"][1] - h["data_offsets"][0]
    want = int(idx.get("metadata", {}).get("total_size", data_bytes))
    if not bad and data_bytes != want:
        bad.append(f"tensor bytes {data_bytes} != index total_size {want}")
    msg = (f"base  {d.name}: {len(shards) - len(bad)}/{len(shards)} shards ok · {data_bytes / 1e9:.2f} GB"
           f" · language-model params {lm / 1e9:.2f}B · all params {total / 1e9:.2f}B")
    return not bad, msg + "".join(f"\n      ✗ {b}" for b in bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--sha", action="store_true", help="also sha256 the pack's files")
    a = ap.parse_args()
    ok = True
    for d in map(Path, a.dirs):
        if (d / "files.json").exists():
            good, msg = check_pack(d, a.sha)
        elif (d / "model.safetensors.index.json").exists():
            good, msg = check_base(d)
        else:
            good, msg = False, f"????  {d}: neither files.json nor model.safetensors.index.json"
        ok &= good
        print(("✓ " if good else "✗ ") + msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
