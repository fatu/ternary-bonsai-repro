"""tbr/quant.py: torch arithmetic == the numpy reference, STE, convert / materialize / export_pack on a toy model."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from tbr import pack as P                                                   # noqa: E402
from tbr import quant as Q                                                  # noqa: E402
from tbr.pack import canon, load_signs, trits as unpack_trits               # noqa: E402
from tbr.st import Checkpoint                                               # noqa: E402
from tbr.ternary import rtn                                                 # noqa: E402

BLOCK, G = 16, 8
rng = np.random.default_rng(0)


def signs_for(width):
    return torch.as_tensor(rng.choice([-1.0, 1.0], width).astype(np.float32))


def test_fwht_rotate_fold_match_numpy():
    w = rng.standard_normal((5, 64)).astype(np.float32)
    s = signs_for(64)
    assert np.allclose(Q.fwht(torch.as_tensor(w), BLOCK).numpy(), P.fwht(w, BLOCK), atol=1e-6)
    r = Q.rotate(torch.as_tensor(w), s, BLOCK)
    assert np.allclose(r.numpy(), P.rotate(w, s.numpy(), BLOCK), atol=1e-6)
    assert np.allclose(Q.fold(r, s, BLOCK).numpy(), w, atol=1e-5)          # involution


@pytest.mark.parametrize("rule", Q.RULES)
def test_ternary_matches_numpy(rule):
    w = rng.standard_normal((7, 64)).astype(np.float32) * 0.02
    t_np, s_np = rtn(w, rule, G)
    t, s = Q.ternary(torch.as_tensor(w), rule, G, f16_scales=False)
    assert np.array_equal(t.numpy().astype(np.int8), t_np)
    assert np.allclose(s.numpy(), s_np, rtol=1e-6)
    t16, s16 = Q.ternary(torch.as_tensor(w), rule, G)
    assert np.array_equal(t16.numpy(), t.numpy()) and np.array_equal(s16.numpy(), s_np.astype(np.float16).astype(np.float32))
    assert set(np.unique(t.numpy())) <= {-1.0, 0.0, 1.0}


def test_bonsai_rule_is_absmean_support_with_ls_scale():
    w = rng.standard_normal((3, 64)).astype(np.float32)
    t_b, s_b = Q.ternary(torch.as_tensor(w), "bonsai", G, f16_scales=False)
    t_a, _ = Q.ternary(torch.as_tensor(w), "absmean", G, f16_scales=False)
    assert torch.equal(t_b, t_a)                                            # same support as round(w / mean|w|)
    g, tg = torch.as_tensor(w).reshape(3, -1, G), t_b.reshape(3, -1, G)
    ls = (g * tg).sum(-1) / (tg * tg).sum(-1).clamp_min(1)
    assert torch.allclose(s_b, ls, rtol=1e-5)                               # support mean == least-squares scale


def test_linear_off_is_stock_on_matches_reference_and_ste_gradient():
    lin = nn.Linear(64, 10)
    w0 = lin.weight.detach().clone()
    s = signs_for(64)
    x = torch.randn(4, 64)
    tl = Q.TernaryLinear(lin, s, "bonsai", G, BLOCK)
    assert tl.weight is lin.weight and tl.bias is lin.bias
    tl.enabled = False
    assert torch.equal(tl(x), lin(x))
    tl.enabled = True
    t_np, s_np = rtn(P.rotate(w0.numpy(), s.numpy(), BLOCK), "bonsai", G)
    ref = P.fold(np.repeat(s_np.astype(np.float16).astype(np.float32), G, axis=1) * t_np, s.numpy(), BLOCK)
    assert np.allclose(tl.effective_weight().detach().numpy(), ref, atol=1e-6)
    assert torch.allclose(tl(x), x @ torch.as_tensor(ref).T + lin.bias, atol=1e-5)
    tl(x).square().sum().backward()                                        # STE: dL/dW == dL/dw_eff (H, signs involutions)
    frozen = nn.Linear(64, 10)
    frozen.weight = nn.Parameter(torch.as_tensor(ref)); frozen.bias = nn.Parameter(lin.bias.detach().clone())
    frozen(x).square().sum().backward()
    assert torch.allclose(lin.weight.grad, frozen.weight.grad, atol=1e-5)
    assert torch.allclose(lin.bias.grad, frozen.bias.grad, atol=1e-5)


def test_embedding_uses_the_same_fold():
    emb = nn.Embedding(20, 64)
    s = signs_for(64)
    te = Q.TernaryEmbedding(emb, s, "bonsai", G, BLOCK)
    ids = torch.tensor([[1, 5, 19]])
    assert torch.allclose(te(ids), F_embed(te.effective_weight(), ids))


def F_embed(w, ids):
    return torch.nn.functional.embedding(ids, w)


class Toy(nn.Module):
    """Qwen3.5-shaped names: GDN layer 0, attention layer 1, MLPs, tied embeddings / lm_head, a vision tower."""

    def __init__(self):
        super().__init__()
        H, FF, V = 64, 96, 50
        self.model = nn.Module(); self.model.language_model = nn.Module()
        lm = self.model.language_model
        lm.embed_tokens = nn.Embedding(V, H)
        lm.layers = nn.ModuleList([nn.Module(), nn.Module()])
        la = lm.layers[0].linear_attn = nn.Module()
        la.in_proj_qkv = nn.Linear(H, 3 * H, bias=False); la.in_proj_z = nn.Linear(H, H, bias=False)
        la.in_proj_a = nn.Linear(H, 4, bias=False); la.in_proj_b = nn.Linear(H, 4, bias=False)
        la.out_proj = nn.Linear(H, H, bias=False); la.conv1d = nn.Conv1d(3 * H, 3 * H, 4, groups=3 * H)
        la.A_log = nn.Parameter(torch.zeros(4)); la.norm = nn.LayerNorm(H)
        at = lm.layers[1].self_attn = nn.Module()
        at.q_proj = nn.Linear(H, H, bias=False); at.o_proj = nn.Linear(H, H, bias=False)
        for L in lm.layers:
            L.mlp = nn.Module(); L.mlp.up_proj = nn.Linear(H, FF, bias=False); L.mlp.down_proj = nn.Linear(FF, H, bias=False)
            L.input_layernorm = nn.LayerNorm(H)
        self.model.visual = nn.Module(); self.model.visual.proj = nn.Linear(H, H)
        self.lm_head = nn.Linear(H, V, bias=False)
        self.lm_head.weight = lm.embed_tokens.weight                        # tied

    def forward(self, ids):
        lm = self.model.language_model
        h = lm.embed_tokens(ids)
        la = lm.layers[0].linear_attn
        h = h + la.out_proj(la.in_proj_z(h) * la.in_proj_qkv(h)[..., :64]) + la.in_proj_a(h).sum(-1, keepdim=True)
        h = h + lm.layers[0].mlp.down_proj(lm.layers[0].mlp.up_proj(h))
        at = lm.layers[1].self_attn
        h = h + at.o_proj(at.q_proj(h)) + lm.layers[1].mlp.down_proj(lm.layers[1].mlp.up_proj(h))
        return self.lm_head(lm.layers[1].input_layernorm(h))


def test_convert_coverage_signs_tying_materialize_and_export(tmp_path):
    torch.manual_seed(0)
    model = Toy()
    ids = torch.randint(0, 50, (2, 5))
    with torch.no_grad():
        stock = model(ids)
    rep = Q.convert(model, "v2", "bonsai", G, BLOCK, seed=1)
    names = set(rep["converted"])
    assert "model.language_model.layers.0.linear_attn.in_proj_qkv" in names and "lm_head" in names
    assert "model.language_model.embed_tokens" in names and "model.language_model.layers.1.self_attn.q_proj" in names
    assert "model.language_model.layers.0.linear_attn.in_proj_a" in rep["skipped"]
    assert "model.visual.proj" in rep["skipped"] and rep["unrotated"] == []
    assert sorted(rep["signs"]) == [64, 96]
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight      # still tied
    assert all(m.signs is rep["signs"][m.weight.shape[1]].float() or torch.equal(m.signs, rep["signs"][m.weight.shape[1]])
               for _, m in Q.ternary_modules(model))
    Q.set_quant(model, False)
    with torch.no_grad():
        assert torch.equal(model(ids), stock)                                           # quant off == stock, bit for bit
    Q.set_quant(model, True)
    with torch.no_grad():
        quant = model(ids)
    assert not torch.allclose(quant, stock, atol=1e-3)
    # v1 also takes in_proj_a / b
    m1 = Toy(); r1 = Q.convert(m1, "v1", seed=1, group=G, block=BLOCK)
    assert "model.language_model.layers.0.linear_attn.in_proj_a" in r1["converted"]
    # export in the pack layout, then read it back with the pack tools
    r = Q.export_pack(model, tmp_path / "pack", BLOCK)
    assert r["packed"] == len(names) and r["widths"] == [64, 96]
    ck = Checkpoint(tmp_path / "pack")
    assert load_signs(tmp_path / "pack")[64].tolist() == rep["signs"][64].tolist()
    for name, m in Q.ternary_modules(model):
        pn = Q.pack_name(name)
        assert canon(pn + ".weight") == canon(name + ".weight")
        t, s = m.quantize()
        assert np.array_equal(unpack_trits(np.asarray(ck.get(pn + ".weight"))), t.numpy().astype(np.int8))
        assert np.array_equal(np.asarray(ck.get(pn + ".scales")), s.numpy().astype(np.float16))
        assert np.array_equal(np.asarray(ck.get(pn + ".biases")), -np.asarray(ck.get(pn + ".scales")))
        assert ck.info(pn + ".signs")["dtype"] == "F16"
    assert ck.info("language_model.model.layers.0.linear_attn.in_proj_a.weight")["dtype"] == "F32"
    assert ck.info("language_model.model.layers.0.linear_attn.A_log")["dtype"] == "F32"
    assert not any(k.startswith("language_model.model.visual") for k in ck.keys())
    hj = json.loads((tmp_path / "pack" / "hadamard.json").read_text())
    assert hj["prism.hadamard.block_size"] == BLOCK and hj["prism.hadamard.sign_widths"] == [64, 96]
    # materialize: stock modules, same outputs, no ternary modules left
    n = Q.materialize(model)
    assert n == len(names) and Q.ternary_modules(model) == []
    assert isinstance(model.lm_head, nn.Linear) and isinstance(model.model.language_model.embed_tokens, nn.Embedding)
    with torch.no_grad():
        assert torch.allclose(model(ids), quant, atol=1e-5)
