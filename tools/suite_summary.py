#!/usr/bin/env python3
"""One row of the S3 table from a tools/eval_suite.sh run directory (+ the ratio to another run).

    python tools/suite_summary.py runs/suite/bonsai --vs runs/suite/base

Reads lm-eval's results_*.json files and the EvalPlus log, writes <run>/suite.json, prints the six task scores,
their mean, and the whitepaper Table-8 groups (Math = GSM8K + MATH-500, Coding = HumanEval+, Knowledge = GPQA-D +
MMLU-Pro slice, IF = IFEval). With --vs, a second line with model/base ratios — Bonsai's published retention is
98.2 % overall; a group far below its Table-8 ratio points at the harness or the export, not the model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# task → (metric keys in order of preference, column name)
TASKS = {
    "gsm8k_cot_llama": (["exact_match,strict-match", "exact_match,flexible-extract"], "GSM8K"),
    "math500": (["exact_match,none"], "MATH-500"),
    "ifeval": (["prompt_level_strict_acc,none"], "IFEval"),
    "gpqa_diamond_cot_zeroshot": (["exact_match,flexible-extract", "exact_match,strict-match"], "GPQA-D"),
    "mmlu_pro": (["exact_match,custom-extract", "exact_match,none"], "MMLU-Pro"),
}
GROUPS = {"Math": ["GSM8K", "MATH-500"], "Coding": ["HumanEval+"], "Knowledge": ["GPQA-D", "MMLU-Pro"], "IF": ["IFEval"]}
COLS = ["GSM8K", "MATH-500", "HumanEval+", "IFEval", "GPQA-D", "MMLU-Pro"]


def lm_eval_scores(run: Path) -> dict[str, float]:
    """{column: score} from every results_*.json under <run>/lm_eval (the latest file wins per task)."""
    out, seen = {}, {}
    for f in sorted(run.glob("lm_eval/**/results_*.json")):
        res = json.loads(f.read_text()).get("results", {})
        for task, (keys, col) in TASKS.items():
            if task in res and (col not in seen or f.stat().st_mtime >= seen[col]):
                for k in keys:
                    if k in res[task]:
                        out[col], seen[col] = float(res[task][k]), f.stat().st_mtime
                        break
    return out


def evalplus_scores(log: Path) -> dict[str, float]:
    """{'HumanEval': base pass@1, 'HumanEval+': plus pass@1} from evalplus.evaluate's stdout."""
    if not log.exists():
        return {}
    out, section = {}, None
    for line in log.read_text().splitlines():
        s = line.strip().lower()
        if s.startswith("humaneval+"):
            section = "HumanEval+"
        elif s.startswith("humaneval"):
            section = "HumanEval"
        m = re.match(r"pass@1:\s*([0-9.]+)", s)
        if m and section:
            out[section] = float(m.group(1))
    return out


def summarize(run: Path) -> dict:
    scores = {**lm_eval_scores(run), **evalplus_scores(run / "humaneval_plus.log")}
    have = [c for c in COLS if c in scores]
    groups = {g: sum(scores[c] for c in cs) / len(cs) for g, cs in GROUPS.items() if all(c in scores for c in cs)}
    return {"run": str(run), "scores": scores, "mean": sum(scores[c] for c in have) / len(have) if have else None,
            "missing": [c for c in COLS if c not in scores], "groups": groups}


def fmt(x):
    return f"{100 * x:5.1f}" if isinstance(x, (int, float)) else "   — "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--vs", help="reference run dir (the bf16 base) for the ratio line")
    a = ap.parse_args()
    run = Path(a.run)
    s = summarize(run)
    (run / "suite.json").write_text(json.dumps(s, indent=1))
    head = f"{'':10}" + "".join(f"{c:>11}" for c in COLS) + f"{'mean':>8} │ " + "  ".join(f"{g:>9}" for g in GROUPS)
    row = lambda name, d, m, g: (f"{name:10}" + "".join(f"{fmt(d.get(c)):>11}" for c in COLS) + f"{fmt(m):>8} │ "
                                 + "  ".join(f"{fmt(g.get(k)):>9}" for k in GROUPS))
    print(head)
    print(row(run.name, s["scores"], s["mean"], s["groups"]))
    if a.vs:
        r = summarize(Path(a.vs))
        print(row(Path(a.vs).name, r["scores"], r["mean"], r["groups"]))
        ratio = {c: s["scores"][c] / r["scores"][c] for c in COLS if c in s["scores"] and r["scores"].get(c)}
        gratio = {g: s["groups"][g] / r["groups"][g] for g in GROUPS if g in s["groups"] and r["groups"].get(g)}
        mratio = s["mean"] / r["mean"] if s["mean"] and r["mean"] else None
        print(row("ratio", ratio, mratio, gratio) + "   (whitepaper: 98.2 % overall)")
        s["ratio_vs"], s["ratio"], s["ratio_groups"], s["ratio_mean"] = a.vs, ratio, gratio, mratio
        (run / "suite.json").write_text(json.dumps(s, indent=1))
    if s["missing"]:
        print(f"missing: {', '.join(s['missing'])}  (see the task logs in {run})")


if __name__ == "__main__":
    main()
