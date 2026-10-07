"""The shared ``prove`` helper (tests/_formal_verify_common.py) must never hang and never crash.

Regression for a CI incident: a pure-Z3 proof ran the formal-verification job to GitHub's
6-hour limit (no solver timeout), and on an ``unknown`` result the old failure message called
``solver.model()`` and crashed with "model is not available".
"""

import pytest
from _formal_verify_common import prove, z3


def test_a_valid_claim_is_proved():
    x, y = z3.Reals("x y")
    prove(z3.Implies(z3.And(x >= 0, y >= 0), x * y >= 0))


def test_an_invalid_claim_fails_with_a_counterexample():
    x = z3.Real("x")
    with pytest.raises(AssertionError, match=r"nope: counterexample"):
        prove(x > 0, msg="nope")


def test_a_python_bool_claim_is_accepted():
    prove(True)
    with pytest.raises(AssertionError, match="counterexample"):
        prove(False)


def test_a_claim_the_solver_cannot_decide_fails_clearly_instead_of_hanging_or_crashing():
    # Fermat for n = 3 over the integers: true, but nothing here can prove it, so every strategy
    # runs into its timeout and the helper must say so (not call model() on an unknown result).
    a, b, c = z3.Ints("a b c")
    claim = z3.Implies(z3.And(a > 0, b > 0, c > 0), a * a * a + b * b * b != c * c * c)
    with pytest.raises(
        AssertionError,
        match=r"solver gave up on every strategy.*neither proved nor refuted",
    ):
        prove(claim, msg="fermat", timeout=0.2)


def test_the_proof_does_not_depend_on_what_was_built_in_the_global_context_before():
    # Terms created in the main context before the call (as earlier tests do) must not matter:
    # the claim is copied into a fresh context. Build lots of unrelated terms first.
    for i in range(200):
        z3.Real(f"junk{i}") * z3.Int(f"j{i}")
    x, y = z3.Reals("x y")
    prove(z3.Implies(z3.And(x >= 0, y >= 0), x + y >= x))
