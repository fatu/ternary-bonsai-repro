"""verify_download.py + inspect_ckpt.py on a tiny fake pack and base (same names, dtypes, layout)."""
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from tbr.pack import pack_codes

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(1)
_DT = {np.dtype(np.uint32): "U32", np.dtype(np.float16): "F16", np.dtype(np.float32): "F32"}


def write_st(path, tensors, bf16=()):
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        raw = a.tobytes()
        header[name] = {"dtype": "BF16" if name in bf16 else _DT[a.dtype], "shape": list(a.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def make_fixture(tmp):
    hid, ff = 256, 512
    signs = {w: rng.choice([-1.0, 1.0], size=w).astype(np.float16) for w in (hid, ff)}
    pack, base = {}, {}
    for name, (r, c) in {"mlp.up_proj": (ff, hid), "mlp.down_proj": (hid, ff)}.items():
        for layer in (0, 1):
            m = f"language_model.model.layers.{layer}.{name}"
            s = rng.uniform(0.01, 0.02, size=(r, c // 128)).astype(np.float16)
            pack[m + ".weight"] = pack_codes(rng.integers(0, 3, size=(r, c), dtype=np.uint8))
            pack[m + ".scales"], pack[m + ".biases"], pack[m + ".signs"] = s, -s, signs[c]
            base[f"model.language_model.layers.{layer}.{name}.weight"] = rng.integers(0, 2**16, (r, c), dtype=np.uint16)
    pack["language_model.model.layers.0.input_layernorm.weight"] = np.ones(hid, np.float16)
    pack["vision_tower.blocks.0.attn.qkv.weight"] = np.zeros((8, 8), np.float16)

    pd, bd = tmp / "pack", tmp / "base"
    pd.mkdir(); bd.mkdir()
    write_st(pd / "model.safetensors", pack)
    vals = np.concatenate([signs[hid], signs[ff]]).astype(float).tolist()
    (pd / "hadamard.json").write_text(json.dumps({"prism.hadamard.sign_widths": [hid, ff], "prism.hadamard.sign_values": vals}))
    files = {n: {"size": (pd / n).stat().st_size, "sha256": hashlib.sha256((pd / n).read_bytes()).hexdigest()}
             for n in ("model.safetensors", "hadamard.json")}
    (pd / "files.json").write_text(json.dumps(files))

    names = sorted(base)
    shards = {"model-00001-of-00002.safetensors": names[:2], "model-00002-of-00002.safetensors": names[2:]}
    for f, ns in shards.items():
        write_st(bd / f, {n: base[n] for n in ns}, bf16=set(ns))
    total = sum(base[n].nbytes for n in names)
    (bd / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": {n: f for f, ns in shards.items() for n in ns}}))
    return pd, bd


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def test_verify_download(tmp_path):
    pd, bd = make_fixture(tmp_path)
    r = run("tools/verify_download.py", str(pd), str(bd), "--sha")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2/2 files ok · sha256 ok" in r.stdout and "2/2 shards ok" in r.stdout
    shard = bd / "model-00002-of-00002.safetensors"
    shard.write_bytes(shard.read_bytes()[:-10])               # simulate an interrupted download
    r = run("tools/verify_download.py", str(bd))
    assert r.returncode == 1 and "truncated" in r.stdout


def test_inspect(tmp_path):
    pd, bd = make_fixture(tmp_path)
    r = run("tools/inspect_ckpt.py", str(pd), "--base", str(bd), "--rows", "64")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "4 packed modules" in out
    assert "layers.N.mlp.up_proj" in out and "packed  [512 x 256]" in out
    assert "width    256: 1 distinct · ±1 only True · == hadamard.json True" in out
    assert "0.000  |b+s|max 0.0e+00" in out                    # code 3 unused, biases == -scales
    assert "4/4 packed modules match" in out
    assert "1 non-LM tensors" in out
