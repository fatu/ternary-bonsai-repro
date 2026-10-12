import numpy as np
import pytest

from tbr.kl import kl_stats, windows_from_stream

torch = pytest.importorskip("torch")


def naive(lp, lq, tgt):
    p, q = torch.softmax(lp.float(), -1), torch.softmax(lq.float(), -1)
    kl = (p * (p.log() - q.log())).sum(-1).sum()
    nll_p = -p.log()[torch.arange(len(tgt)), tgt].sum()
    nll_q = -q.log()[torch.arange(len(tgt)), tgt].sum()
    return float(kl), float(nll_p), float(nll_q), int((lp.argmax(-1) == lq.argmax(-1)).sum())


def test_kl_stats_matches_naive_and_is_chunk_invariant():
    g = torch.Generator().manual_seed(0)
    lp = torch.randn(37, 50, generator=g)
    lq = lp + 0.3 * torch.randn(37, 50, generator=g)
    tgt = torch.randint(0, 50, (37,), generator=g)
    kl, nll_p, nll_q, top1 = naive(lp, lq, tgt)
    for chunk in (5, 37, 1000):
        s = kl_stats(lp.bfloat16(), lq.bfloat16(), tgt.numpy(), chunk=chunk)
        assert s["n"] == 37 and s["top1"] == int((lp.bfloat16().argmax(-1) == lq.bfloat16().argmax(-1)).sum())
        assert abs(s["kl"] - kl) < 0.05 * kl + 1e-3            # bf16 inputs, float32 arithmetic
        assert abs(s["nll_p"] - nll_p) < 0.02 * nll_p and abs(s["nll_q"] - nll_q) < 0.02 * nll_q
    same = kl_stats(lp, lp, tgt, chunk=8)
    assert same["kl"] < 1e-5 and same["top1"] == 37 and same["nll_p"] == same["nll_q"]
    assert kl_stats(lp, lq, tgt)["kl"] >= 0


def test_windows_from_stream_is_deterministic_and_joined():
    docs = [f"doc {i} " * (i % 7 + 1) for i in range(200)]
    tokenize = lambda t: [ord(c) for c in t]
    w = windows_from_stream(iter(docs), tokenize, 31, 5, sep=[0])
    assert len(w) == 5 and all(len(x) == 32 for x in w)
    flat = [t for x in w for t in x]
    stream = []
    for d in docs:
        stream += tokenize(d) + [0]
    assert flat == stream[:160]                                  # first tokens of the stream, in order
    assert w == windows_from_stream(iter(docs), tokenize, 31, 5, sep=[0])
    with pytest.raises(ValueError):
        windows_from_stream(iter(docs[:2]), tokenize, 31, 5, sep=[0])
