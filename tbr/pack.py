"""The Ternary Bonsai 2 MLX pack format, in numpy.

Restated from the pack's own loader (`runtime/runtime.py`, `runtime/codec.py`, `hadamard.json`):

* Packed linear: `weight` uint32 [out, in/16] holds 16 two-bit codes per word, code k in bits
  2k..2k+1 (low bits first); `scales`, `biases` f16 [out, in/128] with biases == -scales.
  Dequant w = s*q + b = s*(q - 1), so the trit is t = q - 1 in {-1, 0, +1} and code 3 is unused.
* Rotation: y = W_q @ FWHT(x * signs). FWHT = normalised Sylvester Walsh-Hadamard over blocks of
  1024 along the input axis. One +-1 sign vector per input width (5120 / 6144 / 17408), shared by
  every layer with that width. Dense equivalent in the original basis:
      W_eff = W_q @ H @ diag(signs)        H = blockdiag(H_1024) / sqrt(1024)
* Embedding (the one "inverse" tensor): row = FWHT(dequant(E_q[i])) * signs, which is the same
  fold, E_eff = E_q @ H @ diag(signs).
* GDN value heads are stored grouped (`gdn_activation_layout: grouped`): rows of the V part of
  in_proj_qkv, in_proj_z, in_proj_a/b, A_log, dt_bias, conv1d and the input columns of out_proj
  may be permuted relative to the HF base. Check before comparing trits position by position.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

GROUP = 128
BLOCK = 1024


def unpack_codes(words: np.ndarray) -> np.ndarray:
    """uint32 [rows, in/16] -> uint8 codes [rows, in]."""
    shifts = 2 * np.arange(16, dtype=np.uint32)
    return ((words[..., None] >> shifts) & 3).astype(np.uint8).reshape(words.shape[0], -1)


def pack_codes(codes: np.ndarray) -> np.ndarray:
    """Inverse of unpack_codes (tests and a later export step)."""
    rows, width = codes.shape
    c = codes.astype(np.uint32).reshape(rows, width // 16, 16) << (2 * np.arange(16, dtype=np.uint32))
    return np.bitwise_or.reduce(c, axis=-1).astype(np.uint32)


def trits(words: np.ndarray) -> np.ndarray:
    """int8 trits in {-1, 0, +1} (valid when biases == -scales; code 3 maps to 2 and is flagged)."""
    return unpack_codes(words).astype(np.int8) - 1


def dequant(words, scales, biases) -> np.ndarray:
    """float32 [rows, in] = s*q + b in the rotated basis."""
    q = unpack_codes(words).astype(np.float32).reshape(words.shape[0], -1, GROUP)
    s = scales.astype(np.float32)[..., None]
    b = biases.astype(np.float32)[..., None]
    return (q * s + b).reshape(words.shape[0], -1)


def fwht(x: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Normalised Sylvester-order Walsh-Hadamard transform over blocks of the last axis (float32)."""
    shape = x.shape
    if shape[-1] % block:
        raise ValueError(f"width {shape[-1]} not divisible by block {block}")
    y = np.array(x, dtype=np.float32).reshape(-1, block)
    h = 1
    while h < block:
        y = y.reshape(-1, block // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = np.stack([a + b, a - b], axis=2)
        h *= 2
    return (y.reshape(shape) / np.sqrt(block)).astype(np.float32)


def fold(w_rot: np.ndarray, signs: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Rotated-basis weight -> dense weight in the original basis: W @ H @ diag(signs)."""
    return fwht(w_rot, block) * signs.astype(np.float32)


def rotate(w: np.ndarray, signs: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Original-basis weight -> Bonsai's rotated basis (inverse of fold; H and diag(signs) are involutions)."""
    return fwht(w * signs.astype(np.float32), block)


def load_signs(pack_dir) -> dict[int, np.ndarray]:
    """{input width: +-1 float32 vector} from hadamard.json."""
    h = json.loads((Path(pack_dir) / "hadamard.json").read_text())
    vals = np.asarray(h["prism.hadamard.sign_values"], dtype=np.float32)
    out, off = {}, 0
    for w in h["prism.hadamard.sign_widths"]:
        out[int(w)] = vals[off:off + w]
        off += w
    if off != len(vals) or not np.isin(vals, (-1.0, 1.0)).all():
        raise ValueError("malformed sign vectors in hadamard.json")
    return out


_KEY = re.compile(r"(layers\.\d+\..*|embed_tokens\..*|lm_head\..*|norm\.weight)$")


def canon(name: str) -> str | None:
    """Shared key for a language-model tensor in either checkpoint ('layers.3.mlp.up_proj.weight',
    'embed_tokens.weight', 'lm_head.weight', 'norm.weight'); None for vision / MTP tensors.

    Base: model.language_model.layers.3.mlp.up_proj.weight · Pack: language_model.model.layers.3...
    """
    if not ("language_model" in name or name.startswith("lm_head")):
        return None
    m = _KEY.search(name)
    return m.group(1) if m else None


def layer_pattern(key: str) -> str:
    return re.sub(r"layers\.\d+\.", "layers.N.", key)
