"""Format tests: packing layout, FWHT, and that the dense fold equals the pack's runtime path.

    python -m pytest tests -q
The MLX cross-checks run only where `mlx` imports (a Mac); the box skips them.
"""
import numpy as np
import pytest

from tbr.pack import canon, dequant, fold, fwht, pack_codes, rotate, trits, unpack_codes

rng = np.random.default_rng(0)


def ternary_layer(rows, width):
    codes = rng.integers(0, 3, size=(rows, width), dtype=np.uint8)
    scales = rng.uniform(0.005, 0.05, size=(rows, width // 128)).astype(np.float16)
    return pack_codes(codes), scales, -scales, codes


def sylvester(n):
    i = np.arange(n)
    return np.where(np.vectorize(lambda v: bin(v).count("1") % 2)(i[:, None] & i[None, :]), -1.0, 1.0)


def test_pack_roundtrip_and_layout():
    words, _, _, codes = ternary_layer(4, 256)
    assert np.array_equal(unpack_codes(words), codes)
    one = np.zeros((1, 16), np.uint8)
    one[0, 1], one[0, 15] = 2, 1                      # code k sits in bits 2k..2k+1, low bits first
    assert pack_codes(one)[0, 0] == (2 << 2) | (1 << 30)


def test_trits_and_dequant():
    words, s, b, codes = ternary_layer(3, 256)
    t = trits(words)
    assert set(np.unique(t)) <= {-1, 0, 1}
    w = dequant(words, s, b)
    assert np.allclose(w, t * np.repeat(s.astype(np.float32), 128, axis=1))


def test_fwht_is_normalised_sylvester_per_block():
    n = 16
    x = rng.standard_normal((5, 3 * n)).astype(np.float32)
    h = sylvester(n) / np.sqrt(n)
    ref = np.concatenate([x[:, k * n:(k + 1) * n] @ h.T for k in range(3)], axis=1)
    assert np.allclose(fwht(x, n), ref, atol=1e-5)
    assert np.allclose(fwht(fwht(x, n), n), x, atol=1e-5)


@pytest.mark.parametrize("block", [16, 1024])
def test_fold_equals_runtime_linear(block):
    width = max(2 * block, 256)                       # multiple of both the block and the 128 group
    words, s, b, _ = ternary_layer(8, width)
    signs = rng.choice([-1.0, 1.0], size=width).astype(np.float32)
    x = rng.standard_normal((4, width)).astype(np.float32)
    w_q = dequant(words, s, b)
    y_runtime = fwht(x * signs, block) @ w_q.T        # what Packed.__call__ computes
    y_dense = x @ fold(w_q, signs, block).T
    assert np.allclose(y_runtime, y_dense, atol=1e-4)
    assert np.allclose(rotate(fold(w_q, signs, block), signs, block), w_q, atol=1e-5)


def test_fold_equals_runtime_embedding():
    block, width = 16, 256
    words, s, b, _ = ternary_layer(10, width)
    signs = rng.choice([-1.0, 1.0], size=width).astype(np.float32)
    e_q = dequant(words, s, b)
    idx = np.array([3, 7, 3])
    runtime = fwht(e_q[idx], block) * signs           # Packed embedding path, inverse=True
    assert np.allclose(runtime, fold(e_q, signs, block)[idx], atol=1e-5)


def test_canon_maps_both_namespaces():
    assert canon("model.language_model.layers.3.mlp.up_proj.weight") == "layers.3.mlp.up_proj.weight"
    assert canon("language_model.model.layers.3.mlp.up_proj.scales") == "layers.3.mlp.up_proj.scales"
    assert canon("lm_head.weight") == canon("language_model.lm_head.weight") == "lm_head.weight"
    assert canon("model.language_model.norm.weight") == "norm.weight"
    assert canon("model.visual.blocks.0.norm1.weight") is None
    assert canon("mtp.layers.0.mlp.up_proj.weight") is None


def test_against_mlx():
    mx = pytest.importorskip("mlx.core")
    words, s, b, _ = ternary_layer(4, 1024)
    ref = mx.dequantize(mx.array(words), mx.array(s), mx.array(b), group_size=128, bits=2)
    assert np.allclose(np.array(ref.astype(mx.float32)), dequant(words, s, b), atol=1e-3)
    x = rng.standard_normal((3, 2048)).astype(np.float32)
    h = mx.hadamard_transform(mx.array(x).reshape(-1, 1024), scale=1 / np.sqrt(1024)).reshape(3, 2048)
    assert np.allclose(np.array(h), fwht(x), atol=1e-4)
