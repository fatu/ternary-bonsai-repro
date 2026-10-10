"""export_dense.py end to end on a synthetic pair whose base is EXACTLY representable by the pack.

Base weights = (fold of a ternary pack) in the base layout, so a correct export must return the base bit
for bit — under each regrouping, with the pack's conv1d in MLX layout [C, 4, 1], its layer norms stored with
a +1 offset, and vision/MTP tensors only in the base.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tbr.pack import fold, pack_codes
from tbr.st import Checkpoint, bf16_bits, write_safetensors
from tbr.ternary import inverse, vperm

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(3)
BLOCK, HID, NV, NK, HD = 16, 256, 6, 2, 64
QK, VW = 2 * NK * HD, NV * HD
SIGNS = {w: rng.choice([-1.0, 1.0], w).astype(np.float32) for w in (HID, VW)}


def bits(x):
    return bf16_bits(np.asarray(x, np.float32))


def make_pair(tmp, regroup):
    p = vperm(NV, NK, HD)
    perm = {"none": None, "vperm": p, "inverse": inverse(p)}[regroup]
    ph = None if perm is None else {"vperm": vperm(NV, NK, 1), "inverse": inverse(vperm(NV, NK, 1))}[regroup]
    base, pack = {}, {}

    def to_pack_rows(a, prm, off=0):                 # pack row off+i holds base row off+prm[i]
        if prm is None:
            return a
        b = a.copy()
        b[off:] = a[off:][prm]
        return b

    def packed(kb, kp, rows, width, role=None):
        t = rng.integers(-1, 2, size=(rows, width)).astype(np.int8)
        s = rng.uniform(0.01, 0.03, size=(rows, width // 128)).astype(np.float16)
        w_pack = fold(t * np.repeat(s.astype(np.float32), 128, axis=1), SIGNS[width], BLOCK)
        w_base = w_pack.copy()
        if perm is not None and role == "cols":
            w_base[:, perm] = w_pack
        elif perm is not None and role in ("rows", "rows_v"):
            off = QK if role == "rows_v" else 0
            w_base[off:][perm] = w_pack[off:]
        base[kb] = bits(w_base)
        pack[kp + ".weight"] = pack_codes((t + 1).astype(np.uint8))
        pack[kp + ".scales"], pack[kp + ".biases"], pack[kp + ".signs"] = s, -s, SIGNS[width].astype(np.float16)

    def fp(kb, kp, a_base, pack_value):
        base[kb] = bits(a_base)
        pack[kp] = pack_value.astype(np.float32)

    for L in (0, 1):
        b, q = f"model.language_model.layers.{L}.", f"language_model.model.layers.{L}."
        la, lq = b + "linear_attn.", q + "linear_attn."
        packed(la + "in_proj_qkv.weight", lq + "in_proj_qkv", QK + VW, HID, "rows_v")
        packed(la + "in_proj_z.weight", lq + "in_proj_z", VW, HID, "rows")
        packed(la + "out_proj.weight", lq + "out_proj", HID, VW, "cols")
        packed(b + "mlp.up_proj.weight", q + "mlp.up_proj", 384, HID)
        a_w = rng.standard_normal((NV, HID)).astype(np.float32) * 0.02
        fp(la + "in_proj_a.weight", lq + "in_proj_a.weight", a_w, to_pack_rows(bits_f(a_w), ph))
        A_log = rng.standard_normal(NV).astype(np.float32)
        fp(la + "A_log", lq + "A_log", A_log, to_pack_rows(bits_f(A_log), ph))
        conv = rng.standard_normal((QK + VW, 1, 4)).astype(np.float32)
        conv_pack = to_pack_rows(bits_f(conv), None if perm is None else perm, QK).transpose(0, 2, 1)   # MLX [C, 4, 1]
        fp(la + "conv1d.weight", lq + "conv1d.weight", conv, conv_pack)
        ln = rng.standard_normal(HID).astype(np.float32) * 0.1
        fp(b + "input_layernorm.weight", q + "input_layernorm.weight", ln, bits_f(ln) + 1.0)      # zero-centred +1
    packed("model.language_model.embed_tokens.weight", "language_model.model.embed_tokens", 300, HID)
    packed("lm_head.weight", "language_model.lm_head", 300, HID)
    base["model.visual.blocks.0.attn.qkv.weight"] = bits(rng.standard_normal((32, 32)))
    base["mtp.fc.weight"] = bits(rng.standard_normal((16, 32)))

    pd, bd = tmp / "pack", tmp / "base"
    pd.mkdir(); bd.mkdir()
    write_safetensors(pd / "model.safetensors", pack)
    (pd / "hadamard.json").write_text(json.dumps({
        "prism.hadamard.block_size": BLOCK, "prism.hadamard.sign_widths": [HID, VW],
        "prism.hadamard.sign_values": np.concatenate([SIGNS[HID], SIGNS[VW]]).astype(float).tolist()}))
    names = sorted(base)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    for f, ns in shards.items():
        write_safetensors(bd / f, {n: base[n] for n in ns}, bf16=set(ns))
    (bd / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(base[n].nbytes for n in names)},
         "weight_map": {n: f for f, ns in shards.items() for n in ns}}))
    (bd / "config.json").write_text(json.dumps({"text_config": {
        "linear_num_value_heads": NV, "linear_num_key_heads": NK, "linear_value_head_dim": HD,
        "linear_key_head_dim": HD, "num_hidden_layers": 2}}))
    (bd / "tokenizer.json").write_text("{}")
    return pd, bd


def bits_f(x):
    """float32 value of the bf16 rounding of x (what the base stores)."""
    return (bits(x).astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize("regroup", ["none", "vperm", "inverse"])
def test_export_reproduces_the_base(tmp_path, regroup):
    pd, bd = make_pair(tmp_path, regroup)
    od = tmp_path / "dense"
    r = subprocess.run([sys.executable, "tools/export_dense.py", str(pd), str(bd), str(od), "--regroup", regroup,
                        "--workers", "2"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    base, out = Checkpoint(bd), Checkpoint(od)
    assert set(base.keys()) == set(out.keys())
    for k in base.keys():
        assert base.info(k)["shape"] == out.info(k)["shape"] and base.info(k)["dtype"] == out.info(k)["dtype"], k
        assert np.array_equal(np.asarray(base.get(k)), np.asarray(out.get(k))), k     # bit-exact
    assert "transpose" in r.stdout and "minus-1" in r.stdout and "rel median 0.00e+00" in r.stdout
    assert (od / "tokenizer.json").exists() and (od / "model.safetensors.index.json").exists()
    v = subprocess.run([sys.executable, "tools/verify_download.py", str(od)], cwd=ROOT, capture_output=True, text=True)
    assert v.returncode == 0, v.stdout


def test_wrong_regroup_is_visible(tmp_path):
    """Export with the wrong order: the packed GDN tensors no longer match, and the FP alignment report flags it."""
    pd, bd = make_pair(tmp_path, "vperm")
    od = tmp_path / "dense"
    r = subprocess.run([sys.executable, "tools/export_dense.py", str(pd), str(bd), str(od), "--regroup", "none",
                        "--workers", "1"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    base, out = Checkpoint(bd), Checkpoint(od)
    k = "model.language_model.layers.0.linear_attn.out_proj.weight"
    assert not np.array_equal(np.asarray(base.get(k)), np.asarray(out.get(k)))
    assert "BASE (pack rel" in r.stdout or "max 1." in r.stdout
