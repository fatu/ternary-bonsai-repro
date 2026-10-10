from tbr.evalutil import make_windows, perplexity


def test_windows_are_seeded_disjoint_and_sized():
    ids = list(range(10_000))
    w1, w2 = make_windows(ids, 99, 20, seed=0), make_windows(ids, 99, 20, seed=0)
    assert w1 == w2 and len(w1) == 20 and all(len(w) == 100 for w in w1)
    starts = [w[0] for w in w1]
    assert starts == sorted(starts) and all(b - a >= 100 for a, b in zip(starts, starts[1:]))
    assert make_windows(ids, 99, 20, seed=1) != w1


def test_perplexity():
    assert abs(perplexity(0.0, 10) - 1.0) < 1e-12
