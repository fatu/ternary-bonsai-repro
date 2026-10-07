# ternary-bonsai-repro

Reproducing PrismML's **Ternary Bonsai 2 27B**: Qwen3.8-27B with ternary {-1, 0, +1} weights,
groups of 128 sharing one FP16 scale, in a fixed blockwise-Hadamard basis. The format is
disclosed; the recovery training is not. This repo reads the format exactly, scores any ternary
checkpoint with stock tools, and rebuilds the training recipe by ablation (R0–R7).

Weights, data and run outputs never enter the repo (`.gitignore`).

## The format, as the pack's own loader defines it

Read from `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (`runtime/runtime.py`, `runtime/codec.py`,
`hadamard.json`, `config.json`). `tbr/pack.py` restates it in numpy, and `tests/test_pack.py` checks
it against MLX's own `dequantize` and `hadamard_transform`.

| | |
|---|---|
| Packed linear | `weight` uint32 `[out, in/16]` (16 two-bit codes per word, low bits first) · `scales`, `biases` f16 `[out, in/128]` · `biases == -scales`, so `w = s·(q-1)`, trit `t = q-1`, code 3 unused |
| Rotation | `y = W_q · FWHT(x ⊙ signs)`, normalised Sylvester Walsh-Hadamard, block 1024, along the input axis |
| Signs | explicit ±1 vectors, **one per input width** (5120 / 6144 / 17408), shared by every layer; also stored as `<module>.signs` |
| Dense fold | `W_eff = W_q · H · diag(signs)` = `fold(W_q, signs)`; the inverse is `rotate(W, signs)` |
| Embedding | ternary too; the one "inverse" module: `row = FWHT(dequant(E_q[i])) ⊙ signs`, which is the same fold |
| Packed (402) | q/k/v/o, GDN `in_proj_qkv` / `in_proj_z` / `out_proj`, MLP gate/up/down, `lm_head`, `embed_tokens` |
| Kept FP | GDN `in_proj_a/b`, `conv1d`, `A_log`, `dt_bias`, all norms; vision tower FP16, unrotated |
| GDN layout | value heads stored *grouped* (`gdn_activation_layout: grouped`); see the B1 caveat below |

## Layout

```
tbr/st.py          safetensors header + memmap reader (no torch / MLX; handles U32, F16, BF16)
tbr/pack.py        unpack, dequant, trits, fwht, fold / rotate, load_signs, canon (shared tensor names)
tools/verify_download.py   B0: both checkpoints complete (sizes, sha256, truncation, param counts)
tools/inspect_ckpt.py      B1 step 1: tensor patterns, sign vectors, code shares, shape match vs base
scripts/setup_env.sh       uv venv + vLLM + this package + eval, GPU listing, tests
scripts/download.sh        both checkpoints (~64 GB, resumable) + verify
tests/                     format tests (MLX cross-checks skip on Linux) + tool tests on a fake pack
```

## On the box

```bash
bash scripts/setup_env.sh                                    # 📷 the torch / GPU lines + "N passed"
CKPT_DIR=/path/with/70GB bash scripts/download.sh            # 📷 the two ✓ lines
source .venv/bin/activate
python tools/inspect_ckpt.py checkpoints/bonsai2-27b-mlx --base checkpoints/qwen3.8-27b   # 📷 all four sections
```

## Steps

- **B0** repo, env, downloads, verify. ◀ now
- **B1** forensics: per layer type and depth, compare Bonsai's trits with `RTN_ternary(rotate(W_base, signs))`
  — flip rate, zero share, scale ratio (absmean vs TWN 0.7·mean|w| vs learned). Low flip rate: mostly PTQ;
  high: long QAT. **Caveat:** GDN value-head order may differ between the pack (grouped) and the HF base.
  Rows of `in_proj_qkv` (V part) / `in_proj_z`, and the *input columns* of `out_proj` (permute before
  rotating). Test both orders on one layer; the right one has the far lower flip rate.
- **B2** dense-dequant export: `fold` every packed module back to the HF layout, so stock vLLM + lm-eval score it.
- **B3** baselines on the inner suite: Qwen3.8-27B bf16 and the dequantised Bonsai 2.
- **B4** Phase-1 model: Qwen3.8 has no small sibling on HF (27B, Flash-Next, 2.4T-A95B only). Need a small
  Gated-DeltaNet hybrid with the same layer layout; PrismML's v1 packs `Ternary-Bonsai-{1.7B,4B,8B}-unpacked`
  are small references, but v1 is unrotated.
