"""suite_summary.py on a fake run directory (lm-eval results files + an EvalPlus log)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from suite_summary import evalplus_scores, summarize  # noqa: E402

EVALPLUS_LOG = """Computing expected output...
humaneval (base tests)
pass@1:\t0.823
humaneval+ (base + extra tests)
pass@1:\t0.768
"""


def make_run(d: Path, scale=1.0, drop=()):
    res = {"gsm8k_cot_llama": {"exact_match,strict-match": 0.90 * scale, "exact_match,flexible-extract": 0.92 * scale},
           "math500": {"exact_match,none": 0.80 * scale},
           "ifeval": {"prompt_level_strict_acc,none": 0.70 * scale, "inst_level_strict_acc,none": 0.8},
           "gpqa_diamond_cot_zeroshot": {"exact_match,strict-match": 0.40 * scale, "exact_match,flexible-extract": 0.50 * scale},
           "mmlu_pro": {"exact_match,custom-extract": 0.60 * scale}, "mmlu_pro_math": {"exact_match,custom-extract": 0.1}}
    for i, (task, r) in enumerate(res.items()):
        if task in drop:
            continue
        f = d / "lm_eval" / "model" / f"results_2026-10-12T00-00-0{i}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"results": {task: r}}))
    if "HumanEval+" not in drop:
        (d / "humaneval_plus.log").write_text(EVALPLUS_LOG)


def test_scores_groups_and_ratio(tmp_path):
    base, bonsai = tmp_path / "base", tmp_path / "bonsai"
    make_run(base)
    make_run(bonsai, scale=0.9, drop=("ifeval",))
    s = summarize(base)
    assert s["scores"] == {"GSM8K": 0.90, "MATH-500": 0.80, "IFEval": 0.70, "GPQA-D": 0.50, "MMLU-Pro": 0.60,
                           "HumanEval+": 0.768, "HumanEval": 0.823}
    assert abs(s["mean"] - (0.9 + 0.8 + 0.768 + 0.7 + 0.5 + 0.6) / 6) < 1e-9 and s["missing"] == []
    assert s["groups"] == pytest.approx({"Math": 0.85, "Coding": 0.768, "Knowledge": 0.55, "IF": 0.70})
    b = summarize(bonsai)
    assert b["missing"] == ["IFEval"] and "IF" not in b["groups"] and abs(b["groups"]["Math"] - 0.765) < 1e-9
    r = subprocess.run([sys.executable, "tools/suite_summary.py", str(bonsai), "--vs", str(base)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ratio" in r.stdout and "missing: IFEval" in r.stdout and "98.2" in r.stdout
    j = json.loads((bonsai / "suite.json").read_text())
    assert abs(j["ratio"]["GSM8K"] - 0.9) < 1e-9 and abs(j["ratio_groups"]["Math"] - 0.9) < 1e-9
    assert abs(j["ratio"]["HumanEval+"] - 1.0) < 1e-9 and "IFEval" not in j["ratio"]


def test_evalplus_log_parsing(tmp_path):
    f = tmp_path / "humaneval_plus.log"
    f.write_text(EVALPLUS_LOG)
    assert evalplus_scores(f) == {"HumanEval": 0.823, "HumanEval+": 0.768}
    assert evalplus_scores(tmp_path / "missing.log") == {}
