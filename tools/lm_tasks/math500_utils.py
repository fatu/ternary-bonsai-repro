"""Grader for the math500 task: last \\boxed{} of the answer vs the gold, equivalent under math_verify."""
from __future__ import annotations

from math_verify import parse, verify


def process_results(doc: dict, results: list[str]) -> dict:
    gold = parse("\\boxed{" + doc["answer"] + "}")
    pred = parse(results[0])
    return {"exact_match": float(bool(gold) and bool(pred) and verify(gold, pred))}
