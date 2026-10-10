#!/usr/bin/env python3
"""WikiText-2 perplexity + one greedy generation, through vLLM (S2's sanity check, S3's first metric).

    python tools/ppl.py checkpoints/qwen3.8-27b
    python tools/ppl.py checkpoints/bonsai2-27b-dense

Fixed protocol: WikiText-2 raw test split joined with blank lines, tokenized once, 32 windows of 2,048 tokens
at seeded non-overlapping offsets (seed 0) — every model is scored on the same 65,504 predicted tokens.
Perplexity = exp(mean NLL over tokens) from vLLM prompt logprobs. A wrong fold or head order shows up as a
perplexity in the hundreds or thousands; dense Bonsai should land within ~20 % of the base.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tbr.evalutil import make_windows, perplexity  # noqa: E402

PROMPT = "The three primary colors of light are"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-seqs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp", type=int, default=4, help="tensor-parallel GPUs")
    a = ap.parse_args()
    t0 = time.time()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    tok = AutoTokenizer.from_pretrained(a.model)
    windows = make_windows(tok.encode(text, add_special_tokens=False), a.seq_len, a.n_seqs, a.seed)

    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", max_model_len=a.seq_len + 64,
              gpu_memory_utilization=0.85, seed=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in windows],
                        SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0))
    nll, n = 0.0, 0
    for w, o in zip(windows, outs):
        for pos, lp in enumerate(o.prompt_logprobs):
            if lp is None:                         # position 0 has no prediction
                continue
            nll -= lp[w[pos]].logprob
            n += 1
    gen = llm.generate([PROMPT], SamplingParams(max_tokens=24, temperature=0.0))[0].outputs[0].text
    print(f"{Path(a.model).name}: WikiText-2 PPL {perplexity(nll, n):.3f} over {n:,} tokens "
          f"({a.n_seqs}×{a.seq_len}, seed {a.seed}) · {time.time() - t0:.0f} s")
    print(f"  greedy: {PROMPT!r} → {gen.strip()!r}")


if __name__ == "__main__":
    main()
