#!/usr/bin/env python3
"""Training-free ternary rows of the ladder (R0 / R1 / R2): convert → materialize → dense bf16 checkpoint for S3.

    python tools/ptq.py checkpoints/qwen3.5-2b runs/ladder/r0 --no-rotate --coverage v1     # R0: RTN, no rotation, all ternary
    python tools/ptq.py checkpoints/qwen3.5-2b runs/ladder/r1 --coverage v1                 # R1: + Hadamard (seeded signs)
    python tools/ptq.py checkpoints/qwen3.5-2b runs/ladder/r2                               # R2: + Bonsai's FP list (v2)
    python tools/ptq.py checkpoints/qwen3.5-2b runs/ladder/r3 --method gptq --device cuda     # R3: Hessian-aware (GPTQ) ternary
    python tools/ptq.py checkpoints/qwen3.8-27b runs/ladder/r2_27b --signs checkpoints/bonsai2-27b-mlx --export-pack runs/ladder/r2_27b_pack

Then `bash tools/eval_suite.sh runs/ladder/r2 r2` and `python tools/kl.py checkpoints/qwen3.5-2b runs/ladder/r2`.
--signs <pack dir> uses Bonsai's own sign vectors (27B only) so `tools/forensics.py <pack> <base>` can compare trits.
--export-pack also writes our trits in the pack layout (tools/inspect_ckpt.py / forensics.py read it).
--method gptq (tbr/gptq.py) calibrates on 128 × 2,048 FineWeb-Edu tokens taken *after* the KL reference windows
(cached in runs/kl/); Hessians sit on the GPU, so the 2B wants --device cuda; the 27B needs the per-layer variant (S9).
CPU-only is fine for the 2B RTN rows (a few minutes); the 27B wants one 96 GB card (--device cuda).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tbr.pack import load_signs  # noqa: E402
from tbr.quant import RULES, convert, export_pack, materialize  # noqa: E402

SMALL_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
               "special_tokens_map.json", "generation_config.json", "preprocessor_config.json",
               "video_preprocessor_config.json")


def load_model(path: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    except (ValueError, KeyError):                      # Qwen3_5ForConditionalGeneration is a vision-text class
        m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16)
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--rule", choices=RULES, default="bonsai")
    ap.add_argument("--coverage", choices=["v1", "v2"], default="v2")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--no-rotate", action="store_true", help="R0: ternary in the original basis")
    ap.add_argument("--signs", help="Bonsai pack dir: use its sign vectors (default: seeded ±1 per width)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--export-pack", help="also write our trits / scales / signs in the pack layout here")
    ap.add_argument("--method", choices=["rtn", "gptq"], default="rtn")
    ap.add_argument("--calib", default="fineweb", help="gptq calibration: fineweb | random (tests)")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--calib-len", type=int, default=2048)
    ap.add_argument("--calib-skip", type=int, default=245, help="FineWeb windows to skip (= the KL reference)")
    ap.add_argument("--percdamp", type=float, default=0.01)
    a = ap.parse_args()
    t0 = time.time()
    import torch

    model = load_model(a.model, a.device)
    signs = None
    if a.signs:
        signs = {w: torch.as_tensor(v) for w, v in load_signs(a.signs).items()}
    rep = convert(model, a.coverage, a.rule, a.group, a.block, signs, a.seed, rotate_weights=not a.no_rotate)
    print(f"{len(rep['converted'])} modules ternary ({a.rule}, {a.coverage}, g{a.group}, "
          f"{'no rotation' if a.no_rotate else 'block ' + str(a.block) + ' · widths ' + str(sorted(rep['signs']))}) · "
          f"{len(rep['unrotated'])} unrotated · {len(rep['skipped'])} kept FP/skipped · {time.time() - t0:.0f} s")
    if a.method == "gptq":
        from tbr.gptq import apply_gptq
        from tbr.kl import fineweb_windows

        dev = next(model.parameters()).device
        if a.calib == "random":
            g = torch.Generator().manual_seed(a.seed)
            vocab = model.get_input_embeddings().num_embeddings
            calib = [torch.randint(0, vocab, (1, a.calib_len), generator=g).to(dev) for _ in range(a.n_calib)]
        else:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(a.model)
            w = fineweb_windows(tok, a.calib_len - 1, a.n_calib, "runs/kl", skip=a.calib_skip)
            calib = [torch.as_tensor(x[None], device=dev) for x in w]
        rel = apply_gptq(model, calib, a.rule, percdamp=a.percdamp)
        import numpy as np

        v = np.array(list(rel.values()))
        print(f"gptq on {len(rel)} modules · {a.n_calib} × {a.calib_len} {a.calib} tokens · rel err median {np.median(v):.3f} "
              f"[{v.min():.3f}, {v.max():.3f}] · {time.time() - t0:.0f} s")
    if a.export_pack:
        r = export_pack(model, a.export_pack, a.block, metadata={"rule": a.rule, "coverage": a.coverage})
        print(f"pack layout → {a.export_pack}: {r['packed']} packed modules, {r['tensors']} tensors")
    n = materialize(model)
    out = Path(a.out)
    model.save_pretrained(out, safe_serialization=True)
    for f in SMALL_FILES:
        if (Path(a.model) / f).exists():
            shutil.copy2(Path(a.model) / f, out / f)
    (out / "ptq.json").write_text(json.dumps({k: v for k, v in vars(a).items()} | {
        "converted": rep["converted"], "unrotated": rep["unrotated"], "skipped": rep["skipped"]}, indent=1))
    print(f"{n} modules materialized → {out} ({sum(f.stat().st_size for f in out.glob('*.safetensors')) / 1e9:.1f} GB) · "
          f"{time.time() - t0:.0f} s · next: bash tools/eval_suite.sh {out} <tag> · python tools/kl.py {a.model} {out}")


if __name__ == "__main__":
    main()
