"""tbr/gptq.py: GPTQ ternary == nearest-level RTN under a white Hessian, beats RTN under correlated inputs, is idempotent
for the module's re-quantization, and runs end to end through apply_gptq and tools/ptq.py --method gptq."""
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tbr import quant as Q                                      # noqa: E402
from tbr.gptq import apply_gptq, gptq_ternary                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
G = 16


def nearest_level(w, rule, group):
    _, s = Q.ternary(w, rule, group)
    rows = w.shape[0]
    t = torch.clamp(torch.round(w.reshape(rows, -1, group) / s[..., None].clamp_min(1e-12)), -1, 1).reshape(rows, -1)
    return t, s


def test_white_hessian_is_nearest_level_rtn():
    torch.manual_seed(0)
    w = torch.randn(6, 64) * 0.02
    t, s = gptq_ternary(w, torch.eye(64), "bonsai", G, blocksize=G)
    t0, s0 = nearest_level(w, "bonsai", G)
    assert torch.equal(t, t0) and torch.equal(s, s0)


@pytest.mark.parametrize("rule", ["bonsai", "twn"])
def test_correlated_inputs_beat_rtn_and_result_is_idempotent(rule):
    torch.manual_seed(1)
    n, d = 4096, 64
    basis = torch.randn(d, 8)
    x = torch.randn(n, 8) @ basis.T + 0.1 * torch.randn(n, d)            # strongly correlated inputs
    H = 2 * x.T @ x / n
    w = torch.randn(16, d) * 0.02
    t, s = gptq_ternary(w, H, rule, G, blocksize=32)
    q = Q.dequant(t, s, G)
    t_rtn, s_rtn = Q.ternary(w, rule, G)
    q_rtn = Q.dequant(t_rtn, s_rtn, G)
    err = lambda qq: float(((w - qq) @ x.T).norm())
    assert err(q) < err(q_rtn), (err(q), err(q_rtn))
    t2, s2 = Q.ternary(q, rule, G)                                      # the module re-quantizes the latent: same trits
    assert torch.equal(t2, t) and torch.equal(s2, s)
    assert set(t.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_apply_gptq_on_the_toy_model():
    from test_quant import BLOCK, Toy

    torch.manual_seed(2)
    model = Toy()
    Q.convert(model, "v2", "bonsai", G, BLOCK, seed=3)
    ids = torch.randint(0, 50, (2, 5))
    with torch.no_grad():
        rtn_out = model(ids)
    before = {n: m.quantize()[0].clone() for n, m in Q.ternary_modules(model) if isinstance(m, Q.TernaryLinear)}
    calib = [torch.randint(0, 50, (1, 12)) for _ in range(8)]
    rel = apply_gptq(model, calib, "bonsai", blocksize=G, log=None)
    assert set(rel) == set(before) and all(0 < v < 1.5 for v in rel.values())
    changed = sum(not torch.equal(before[n], m.quantize()[0]) for n, m in Q.ternary_modules(model) if n in before)
    assert changed > 0                                                   # error feedback moved some trits
    assert all(m.enabled for _, m in Q.ternary_modules(model))
    with torch.no_grad():
        out = model(ids)
    assert torch.isfinite(out).all() and not torch.equal(out, rtn_out)
    for n, m in Q.ternary_modules(model):                                # latent == its own re-quantization (idempotent)
        if n in before:
            t, s = m.quantize()
            assert torch.allclose(m.rotated(), Q.dequant(t, s, G), atol=1e-6)
    with pytest.raises(ValueError):
        apply_gptq(model, calib, "absmean", log=None)


def test_ptq_cli_gptq(tmp_path):
    pytest.importorskip("transformers")
    from test_ptq import tiny_qwen

    tiny_qwen(tmp_path)
    out = tmp_path / "r3"
    r = subprocess.run([sys.executable, "tools/ptq.py", str(tmp_path / "base"), str(out), "--group", "32", "--block", "64",
                        "--method", "gptq", "--calib", "random", "--n-calib", "4", "--calib-len", "16"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "gptq on" in r.stdout and "materialized" in r.stdout
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.isfinite(m(torch.randint(0, 96, (1, 8))).logits).all()
