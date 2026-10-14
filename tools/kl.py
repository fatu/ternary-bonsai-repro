#!/usr/bin/env python3
"""S3: KL(base ‖ model) on 1 M reference tokens — the number every recovery run must beat.

    python tools/kl.py checkpoints/qwen3.8-27b checkpoints/bonsai2-27b-dense
    python tools/kl.py checkpoints/qwen3.8-27b checkpoints/bonsai2-27b-dense --gpus 0,1:2,3 --n-windows 245

Both models in bf16 through transformers, side by side (base on the first GPU set, model on the second), fed the
same 245 windows of 4,096 tokens = 1,003,520 predicted positions from FineWeb-Edu (sample-10BT, streamed in file
order, documents joined by EOS). The tokens are cached in runs/kl/ on first use, so every later row is scored on
exactly the same positions. Reports forward KL per token (nats), top-1 agreement, and both models' perplexity on
this text. "Reference", not "held-out": public web text may well be in either model's training data — it is a
fixed yardstick, not a generalization claim.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tbr.evalutil import perplexity  # noqa: E402
from tbr.kl import fineweb_windows, kl_stats  # noqa: E402

def load_model(path: str, gpus: list[int]):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    n = torch.cuda.device_count()
    max_memory = {i: ("88GiB" if i in gpus else "0GiB") for i in range(n)}
    kw = dict(dtype=torch.bfloat16, device_map="auto", max_memory=max_memory)
    try:
        m = AutoModelForCausalLM.from_pretrained(path, **kw)
    except (ValueError, KeyError):                      # Qwen3_5ForConditionalGeneration is a vision-text class
        m = AutoModelForImageTextToText.from_pretrained(path, **kw)
    return m.eval()


def logits_of(model, ids):
    import torch

    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model(input_ids=torch.as_tensor(ids, device=dev)[None], use_cache=False)
    return out.logits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--n-windows", type=int, default=245, help="245 × 4096 = 1,003,520 positions")
    ap.add_argument("--gpus", default="0,1:2,3", help="GPU sets for base:model")
    ap.add_argument("--out-dir", default="runs/kl")
    a = ap.parse_args()
    t0 = time.time()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base)
    windows = fineweb_windows(tok, a.seq_len, a.n_windows, a.out_dir)
    g_base, g_model = ([int(x) for x in s.split(",")] for s in a.gpus.split(":"))
    base, model = load_model(a.base, g_base), load_model(a.model, g_model)
    print(f"loaded both models · {time.time() - t0:.0f} s")

    tot = {"kl": 0.0, "nll_p": 0.0, "nll_q": 0.0, "top1": 0, "n": 0}
    per_window = []
    for i, w in enumerate(windows):
        s = kl_stats(logits_of(base, w[:-1]), logits_of(model, w[:-1]), w[1:])
        per_window.append(s["kl"] / s["n"])
        for k in tot:
            tot[k] += s[k]
        if i % 20 == 0 or i == len(windows) - 1:
            print(f"  window {i + 1}/{len(windows)} · KL so far {tot['kl'] / tot['n']:.4f} nats/token · "
                  f"{time.time() - t0:.0f} s", flush=True)
    n = tot["n"]
    res = {"base": a.base, "model": a.model, "tokens": n, "seq_len": a.seq_len, "n_windows": len(windows),
           "kl_per_token": tot["kl"] / n, "kl_window_std": float(np.std(per_window)),
           "top1_agreement": tot["top1"] / n, "ppl_base": perplexity(tot["nll_p"], n),
           "ppl_model": perplexity(tot["nll_q"], n), "seconds": time.time() - t0}
    out = Path(a.out_dir) / f"{Path(a.model).name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"\n{Path(a.model).name} vs {Path(a.base).name} on {n:,} tokens: "
          f"KL(base‖model) {res['kl_per_token']:.4f} nats/token (window std {res['kl_window_std']:.4f}) · "
          f"top-1 agreement {100 * res['top1_agreement']:.1f} % · PPL base {res['ppl_base']:.3f} → model "
          f"{res['ppl_model']:.3f} · {res['seconds']:.0f} s → {out}")


if __name__ == "__main__":
    main()
