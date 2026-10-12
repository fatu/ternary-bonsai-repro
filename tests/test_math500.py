import sys
from pathlib import Path

import pytest

pytest.importorskip("math_verify")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "lm_tasks"))
from math500_utils import process_results  # noqa: E402


@pytest.mark.parametrize("answer, generation, ok", [
    ("\\frac{1}{2}", "So the answer is $\\boxed{\\frac{1}{2}}$.", 1.0),
    ("\\frac{1}{2}", "Therefore \\boxed{0.5}", 1.0),                    # numerically equal
    ("12", "First guess \\boxed{11}. Correcting: \\boxed{12}", 1.0),    # last box wins
    ("12", "The answer is 12", 1.0),                                   # no box: still parsed
    ("12", "\\boxed{13}", 0.0),
    ("(2, 3)", "\\boxed{(2,3)}", 1.0),
    ("12", "", 0.0),
])
def test_grader(answer, generation, ok):
    assert process_results({"answer": answer}, [generation]) == {"exact_match": ok}
