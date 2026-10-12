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
tools/forensics.py         S1: flip rate / zero share / scale rule / reconstruction error vs the base, GDN order test
tools/export_dense.py      S2: the pack as a dense bf16 checkpoint in the base layout (+ were the FP tensors retrained?)
tools/ppl.py               WikiText-2 perplexity (32 × 2,048 seeded windows) + one greedy generation, via vLLM
tools/eval_suite.sh        S3: the inner task suite for one model (vLLM server + lm-eval + EvalPlus), thinking off
tools/suite_summary.py     S3: one table row from a suite run (+ ratios to the base, Table-8 groups)
tools/lm_tasks/            lm-eval task configs the harness lacks (math500: boxed answer, math_verify grading)
tools/kl.py  tbr/kl.py     S3: KL(base ‖ model) on 1 M reference tokens, two models side by side in transformers
tbr/ternary.py             reference ternary rounding (absmean · TWN · MSE) + the GDN value-head permutation
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
python tools/forensics.py checkpoints/bonsai2-27b-mlx checkpoints/qwen3.8-27b               # S1, ~3 min · 📷 tables C, A, B
python tools/forensics.py checkpoints/bonsai2-27b-mlx checkpoints/qwen3.8-27b --all         # every layer, ~5 min
# S2 (uses the head order forensics chose)
python tools/export_dense.py checkpoints/bonsai2-27b-mlx checkpoints/qwen3.8-27b checkpoints/bonsai2-27b-dense   # ~55 GB · 📷 report
python tools/verify_download.py checkpoints/bonsai2-27b-dense
python tools/ppl.py checkpoints/qwen3.8-27b          # 📷  bf16 base
python tools/ppl.py checkpoints/bonsai2-27b-dense    # 📷  dense Bonsai — within ~20 % of the base, sane generation
# S3 — task suite + KL (one-time: uv pip install -e ".[eval]"; GPQA is gated: accept its terms + huggingface-cli login)
bash tools/eval_suite.sh checkpoints/qwen3.8-27b base                        # ~15 min · 📷 the summary row
VS=runs/suite/base bash tools/eval_suite.sh checkpoints/bonsai2-27b-dense bonsai   # 📷 both rows + the ratio line
python tools/kl.py checkpoints/qwen3.8-27b checkpoints/bonsai2-27b-dense    # ~20 min · 📷 the KL line
```

The suite is greedy, chat-templated, thinking **off** (`THINK=true GEN=16384 MAXLEN=32768` for the final 27B run):
GSM8K 8-shot CoT · MATH-500 (boxed, `math_verify`) · HumanEval+ (EvalPlus) · IFEval (prompt-level strict) ·
GPQA-Diamond CoT zero-shot · MMLU-Pro 5-shot CoT, 8 per category (112 q). Table-8 groups: Math = GSM8K + MATH-500,
Coding = HumanEval+, Knowledge = GPQA-D + MMLU-Pro, IF = IFEval. The KL is forward KL(base ‖ model) per token on 245
windows × 4,096 tokens of FineWeb-Edu (cached in `runs/kl/`, so every later row is scored on the same positions) plus
top-1 agreement and both perplexities on that text.

## Steps

- **B0** ✅ repo, env, downloads, verify.
- **B1** ◀ now — `tools/forensics.py`: per layer type and depth, Bonsai's trits vs `RTN_ternary(rotate(W_base, signs))`
  under absmean / TWN / MSE scales — flip rate, zero share, sign agreement, `s_b/s*`, reconstruction error of the
  base (Bonsai's vs the best RTN). Low flip rate → mostly PTQ; high → long QAT. The GDN value-head order
  (pack stores them grouped) is tested automatically: none / vperm / inverse on four layers, lowest flip wins.
- **B2** `tools/export_dense.py` + `tools/ppl.py` — tested bit-exact on synthetic packs under every head order; the
  vLLM part of `ppl.py` runs only on the box.
- **B3** baselines on the inner suite: Qwen3.8-27B bf16 and the dequantised Bonsai 2 — WikiText-2 PPL done
  (7.169 → 9.740); `tools/eval_suite.sh` + `tools/kl.py` give the task rows and the KL target.
- **B4** Phase-1 models: **Qwen3.5-2B** for iteration (same `Qwen3_5ForConditionalGeneration` layout as
  Qwen3.8-27B: 18 GDN + 6 full-attention layers, 2048 / 6144 both multiples of 1024, 16 v / 16 k heads, tied
  embeddings) and **Qwen3.5-9B** as the scaling point (32 v / 16 k heads = the grouped-GDN case). Qwen3.8 itself
  has no small sibling on HF.
- **B5–B11** quantizer (`tbr/quant.py`), data + teacher, the R0–R7 ladder, 27B, pack + verify, write-up.
