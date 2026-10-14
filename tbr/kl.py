"""KL-to-base arithmetic (torch imported lazily; the model loading lives in tools/kl.py)."""
from __future__ import annotations


def kl_stats(logits_p, logits_q, targets, chunk: int = 512) -> dict:
    """Per-token statistics between a reference P (the base) and a model Q over one window.

    logits_p, logits_q: [T, V] on any device / dtype (Q's chunks are moved to P's device, both to float32);
    targets: [T] next-token ids.  Returns sums over the T positions:
      kl       Σ KL(P_t ‖ Q_t)         forward KL in nats — the number every ladder row must beat
      nll_p    Σ −log P_t(target)      (→ reference-text perplexity of the base)
      nll_q    Σ −log Q_t(target)
      top1     # positions where argmax P == argmax Q
      n        T
    """
    import torch

    T = logits_p.shape[0]
    tgt = torch.as_tensor(targets, device=logits_p.device).long()
    kl = nll_p = nll_q = 0.0
    top1 = 0
    for i in range(0, T, chunk):
        a = logits_p[i:i + chunk].float()
        b = logits_q[i:i + chunk].to(a.device).float()
        la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        t = tgt[i:i + chunk, None]
        kl += float((la.exp() * (la - lb)).sum(-1).sum())
        nll_p -= float(la.gather(1, t).sum())
        nll_q -= float(lb.gather(1, t).sum())
        top1 += int((a.argmax(-1) == b.argmax(-1)).sum())
    return {"kl": kl, "nll_p": nll_p, "nll_q": nll_q, "top1": top1, "n": T}


def windows_from_stream(docs, tokenize, seq_len: int, n_windows: int, sep: list[int], skip: int = 0) -> list[list[int]]:
    """Windows `skip` … `skip + n_windows` of seq_len + 1 tokens from a deterministic document stream, documents joined
    by `sep` — the same tokens for every model scored later (tools/kl.py caches them). `skip` keeps calibration
    windows (tools/ptq.py --method gptq) disjoint from the KL reference windows."""
    need, ids = (seq_len + 1) * (n_windows + skip), []
    for d in docs:
        ids.extend(tokenize(d))
        ids.extend(sep)
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise ValueError(f"stream too short: {len(ids)} tokens for {n_windows + skip} windows of {seq_len + 1}")
    return [ids[i * (seq_len + 1):(i + 1) * (seq_len + 1)] for i in range(skip, skip + n_windows)]


DATASET, CONFIG = "HuggingFaceFW/fineweb-edu", "sample-10BT"


def fineweb_windows(tok, seq_len: int, n_windows: int, cache_dir, skip: int = 0):
    """Cached int64 array [n_windows, seq_len + 1] of FineWeb-Edu (sample-10BT, streamed in file order, EOS-joined)."""
    from pathlib import Path

    import numpy as np

    cache_dir = Path(cache_dir)
    f = cache_dir / f"tokens_fineweb-edu_{seq_len}x{n_windows}{'_skip' + str(skip) if skip else ''}.npy"
    if f.exists():
        return np.load(f)
    from datasets import load_dataset

    ds = load_dataset(DATASET, name=CONFIG, split="train", streaming=True)
    sep = [tok.eos_token_id] if tok.eos_token_id is not None else tok.encode("\n\n", add_special_tokens=False)
    w = windows_from_stream((d["text"] for d in ds), lambda t: tok.encode(t, add_special_tokens=False),
                            seq_len, n_windows, sep, skip)
    arr = np.asarray(w, dtype=np.int64)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(f, arr)
    print(f"cached {arr.shape[0]} × {arr.shape[1]} FineWeb-Edu tokens → {f}")
    return arr
