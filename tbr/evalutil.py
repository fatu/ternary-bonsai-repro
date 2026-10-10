"""Shared evaluation helpers (pure Python; the GPU parts live in tools/)."""
from __future__ import annotations

import math
import random


def make_windows(ids: list[int], seq_len: int, n_seqs: int, seed: int = 0) -> list[list[int]]:
    """n_seqs windows of seq_len + 1 tokens at seeded, non-overlapping offsets over one token stream.
    Same protocol as projects/QAD/experiments/bitnet_1bit/ppl.py (seeded offsets, fixed text), so every row of
    the ladder is scored on exactly the same tokens."""
    span = seq_len + 1
    if len(ids) < span * n_seqs:
        raise ValueError(f"eval text too short: {len(ids)} tokens for {n_seqs} windows of {span}")
    slots = len(ids) // span                       # non-overlapping by construction
    picks = sorted(random.Random(seed).sample(range(slots), n_seqs))
    return [ids[s * span:(s + 1) * span] for s in picks]


def perplexity(nll_sum: float, n_tokens: int) -> float:
    return math.exp(nll_sum / max(n_tokens, 1))
