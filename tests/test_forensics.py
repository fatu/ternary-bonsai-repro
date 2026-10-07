"""forensics.py on a synthetic pair: pack = absmean RTN of rotate(base), GDN value heads regrouped on purpose.

The tool must pick the regrouping that was used, report 0.0 % flips under the absmean rule, matching zero
shares, and rel_err b/q ≥ 1 (the MSE rule can only do as well or better than absmean on the base weights).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tbr.pack import pack_codes, rotate
from tbr.st import bf16_bits, write_safetensors
from tbr.ternary import inverse, rtn, vperm

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(2)


def bf16_round(x):
    return (bf16_bits(x).astype(np.uint32) << 16).view(np.float32)


def make_pair(tmp, regroup):
    block, hid, nv, nk, hd = 16, 256, 6, 2, 64          # 3 value heads per key head, as in the 27B (vperm ≠ its inverse)
    qk, vw = 2 * nk * hd, nv * hd                        # out_proj input width 384 gets its own sign vector
    signs = {w: rng.choice([-1.0, 1.0], w).astype(np.float32) for w in (hid, vw)}
    p = vperm(nv, nk, hd)
    perm = {"none": None, "vperm": p, "inverse": inverse(p)}[regroup]
    base, pack = {}, {}

    def add(kb, kp, rows, perm_rows=None, perm_cols=None, offset=0, width=hid):
        w = bf16_round(rng.standard_normal((rows, width)).astype(np.float32) * 0.02)
        base[kb] = bf16_bits(w)
        wp = w.copy()
        if perm_rows is not None:
            wp[offset:] = w[offset:][perm_rows]            # pack row offset+i holds base row offset+perm[i]
        if perm_cols is not None:
            wp = wp[:, perm_cols]                          # pack column j holds base column perm[j]
        t, s = rtn(rotate(wp, signs[width], block), "absmean")
        pack[kp + ".weight"] = pack_codes((t + 1).astype(np.uint8))
        pack[kp + ".scales"], pack[kp + ".biases"] = s.astype(np.float16), (-s).astype(np.float16)
        pack[kp + ".signs"] = signs[width].astype(np.float16)

    for L in (0, 1):
        b, q = f"model.language_model.layers.{L}.", f"language_model.model.layers.{L}."
        add(b + "linear_attn.in_proj_qkv.weight", q + "linear_attn.in_proj_qkv", qk + nv * hd, perm_rows=perm, offset=qk)
        add(b + "linear_attn.in_proj_z.weight", q + "linear_attn.in_proj_z", nv * hd, perm_rows=perm)
        add(b + "linear_attn.out_proj.weight", q + "linear_attn.out_proj", hid, perm_cols=perm, width=vw)
        add(b + "mlp.up_proj.weight", q + "mlp.up_proj", 384)
    add("model.language_model.embed_tokens.weight", "language_model.model.embed_tokens", 300)
    add("lm_head.weight", "language_model.lm_head", 300)

    pd, bd = tmp / "pack", tmp / "base"
    pd.mkdir(); bd.mkdir()
    write_safetensors(pd / "model.safetensors", pack)
    (pd / "hadamard.json").write_text(json.dumps({"prism.hadamard.block_size": block, "prism.hadamard.sign_widths": [hid, vw],
                                                  "prism.hadamard.sign_values": np.concatenate([signs[hid], signs[vw]]).astype(float).tolist()}))
    write_safetensors(bd / "model-00001-of-00001.safetensors", base, bf16=set(base))
    (bd / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {n: "model-00001-of-00001.safetensors" for n in base}}))
    (bd / "config.json").write_text(json.dumps({"text_config": {
        "linear_num_value_heads": nv, "linear_num_key_heads": nk, "linear_value_head_dim": hd,
        "linear_key_head_dim": hd, "num_hidden_layers": 2}}))
    return pd, bd


@pytest.mark.parametrize("regroup", ["none", "vperm", "inverse"])
def test_forensics_recovers_the_pack(tmp_path, regroup):
    pd, bd = make_pair(tmp_path, regroup)
    out = tmp_path / "f.json"
    r = subprocess.run([sys.executable, "tools/forensics.py", str(pd), str(bd), "--layers", "0,1", "--workers", "2",
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"chosen regroup = {regroup}" in r.stdout, r.stdout
    res = json.loads(out.read_text())
    assert res["regroup"] == regroup and len(res["modules"]) == 10
    for key, st in res["modules"].items():
        assert st["flip"]["absmean"] == 0.0, (key, st["flip"])
        assert abs(st["zero_b"] - st["zero_q"]) < 25                 # same weights, different threshold rule
        assert st["rel_b"] >= st["rel_q"] * 0.999, key              # MSE rule is at least as good as absmean
        assert st["bad"] == 0
    # the wrong orders show large flip rates in table C
    rows = {m.group(1): float(m.group(2)) for m in re.finditer(r"^  (none|vperm|inverse) +[\d.]+ +[\d.]+ +[\d.]+ +([\d.]+)$",
                                                               r.stdout, re.M)}
    assert rows[regroup] == 0.0 and all(v > 20 for k, v in rows.items() if k != regroup), rows
