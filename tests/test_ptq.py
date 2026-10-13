"""tools/ptq.py end to end on a tiny Qwen2 checkpoint (CPU): the saved dense model == convert + materialize in memory."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tbr import quant as Q                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tiny_qwen(tmp):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    m.save_pretrained(tmp / "base", safe_serialization=True)
    return m


@pytest.mark.parametrize("flags, coverage, rotate", [([], "v2", True), (["--no-rotate", "--coverage", "v1"], "v1", False)])
def test_ptq_cli_matches_in_memory(tmp_path, flags, coverage, rotate):
    base = tiny_qwen(tmp_path)
    out = tmp_path / "r"
    r = subprocess.run([sys.executable, "tools/ptq.py", str(tmp_path / "base"), str(out), "--group", "32", "--block", "64",
                        "--export-pack", str(tmp_path / "pack"), *flags], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "modules ternary" in r.stdout and "materialized" in r.stdout
    rep = json.loads((out / "ptq.json").read_text())
    assert "lm_head" in rep["converted"] or "model.embed_tokens" in rep["converted"]
    assert (tmp_path / "pack" / "hadamard.json").exists()
    # the same recipe in memory
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(tmp_path / "base", dtype=torch.bfloat16)
    Q.convert(ref, coverage, "bonsai", 32, 64, seed=0, rotate_weights=rotate)
    Q.materialize(ref)
    got = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    ids = torch.randint(0, 96, (1, 8))
    with torch.no_grad():
        a, b = ref(ids).logits, got(ids).logits
    assert torch.equal(a, b), (a - b).abs().max()
    with torch.no_grad():
        assert not torch.allclose(base(ids).logits.float(), b.float(), atol=1e-2)      # it did quantize something
