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


def windows_from_stream(docs, tokenize, seq_len: int, n_windows: int, sep: list[int]) -> list[list[int]]:
    """The first n_windows windows of seq_len + 1 tokens from a deterministic document stream, documents joined
    by `sep` — the same tokens for every model scored later (tools/kl.py caches them)."""
    need, ids = (seq_len + 1) * n_windows, []
    for d in docs:
        ids.extend(tokenize(d))
        ids.extend(sep)
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise ValueError(f"stream too short: {len(ids)} tokens for {n_windows} windows of {seq_len + 1}")
    return [ids[i * (seq_len + 1):(i + 1) * (seq_len + 1)] for i in range(n_windows)]
