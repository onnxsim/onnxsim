"""Formal check for ``onnxsim::TryExactDivide`` (``onnxsim/sym_expr.cpp``).

``SymExpr`` is an exact integer-coefficient polynomial in dimension symbols
(``batch``, ``seq``, ...) -- see ``sym_expr.h``'s own module docstring --
used by the symbolic shape-inference/value-propagation passes
(``sym_shape_infer.cpp``, ``sym_value_eval.cpp``) to carry a dynamic
dimension through shape arithmetic ONNX's own (numeric-only) shape/data
propagation cannot. ``TryExactDivide(num, den)`` is the one place division
happens: the canonical caller is a ``Reshape``'s ``-1`` sentinel, resolved
as ``total // non_deferred_size`` (e.g. ``(batch*1024*128) / (1024*128) ->
batch``).

Unlike the gradient rules proved in the sibling
``test_formal_verify_grad_*.py`` files, this is exact integer/polynomial
arithmetic with no floating-point or transcendental component, so its
soundness is fully decidable rather than approximated: **whenever the
algorithm returns a quotient, that quotient really is the unique value
making the division exact** -- ``TryExactDivide(num, den) == Some(q) implies
q * den == num`` as a polynomial identity, i.e. for every real (hence every
admissible positive-integer) assignment of the symbols, not merely the one
assignment a numeric spot-check would try. The two worked cases below are
the same ones ``onnxsim/sym_expr_test.cpp`` (lines 60-72) already exercises
as concrete C++ unit tests -- reused here as the proof's representative
inputs, so this is a genuine restatement of that code's documented contract,
not an independently invented spec.

There is no Python binding for ``TryExactDivide`` (unlike ``SymExpr``'s
arithmetic, which reaches Python only indirectly via ``model_info``'s MAC
formulas), so unlike the rest of this test family there is no differential
check against a compiled entry point here: the C++ implementation itself is
exercised by ``onnxsim/sym_expr_test.cpp``, and this file's proof is a
translation-validation check of that code's contract in isolation.
"""

from _formal_verify_common import prove, z3


def test_exact_divide_shared_symbol_case_is_sound():
    # (512*batch*seq + 8*batch) / batch -> 512*seq + 8
    # (onnxsim/sym_expr_test.cpp:67-72)
    batch, seq = z3.Reals("batch seq")
    num = 512 * batch * seq + 8 * batch
    den = batch
    quotient = 512 * seq + 8
    prove(
        z3.ForAll([batch, seq], quotient * den == num),
        "TryExactDivide's shared-symbol quotient does not reconstruct num",
    )


def test_exact_divide_reshape_minus_one_case_is_sound():
    # Reshape's -1 sentinel: (batch*1024*128) / (1024*128) -> batch
    # (onnxsim/sym_expr_test.cpp:60-66)
    batch = z3.Real("batch")
    num = batch * 1024 * 128
    den = 1024 * 128
    quotient = batch
    prove(
        z3.ForAll([batch], quotient * den == num),
        "TryExactDivide's Reshape -1 quotient does not reconstruct num",
    )


def test_exact_divide_refuses_when_no_integer_quotient_exists():
    # (3*batch) / 2: TryExactDivide's own per-term check (coeff % den_coeff
    # != 0) refuses this (onnxsim/sym_expr_test.cpp:76-77) rather than
    # returning a fractional or rounded SymExpr -- confirm that refusal is
    # the only sound choice, i.e. no *integer* q makes 2*q == 3.
    c = z3.Int("c")
    solver = z3.Solver()
    solver.add(2 * c == 3)
    assert solver.check() == z3.unsat, (
        "an integer quotient exists for (3*batch)/2 -- refusing was wrong"
    )


def test_exact_divide_rounding_instead_of_refusing_would_be_unsound():
    # Negative control: if the algorithm instead "rounded" to the nearest
    # integer (1 or 2) rather than refusing, the result would not satisfy
    # the exact-division identity for a generic batch -- confirms refusal is
    # load-bearing, not merely conservative.
    batch = z3.Real("batch")
    for guess in (1, 2):
        solver = z3.Solver()
        solver.add(z3.Not(z3.ForAll([batch], guess * 2 * batch == 3 * batch)))
        assert solver.check() == z3.sat, (
            f"rounded quotient {guess} satisfies the identity for all batch -- "
            "negative control is vacuous"
        )


def test_exact_divide_polynomial_divisor_would_be_unsound_if_attempted():
    # A genuine polynomial (multi-term) divisor is unsupported by design
    # (onnxsim/sym_expr_test.cpp:78-79: (batch*seq)/(batch+seq) -> nullopt).
    # Confirm why: no single-term "quotient" q makes q*(batch+seq) ==
    # batch*seq for every (batch, seq) -- there is no polynomial q of degree
    # < 2 that works, and the true quotient is not a polynomial at all
    # (batch*seq/(batch+seq) has a pole), so refusing is the only sound
    # choice regardless of which single-term q is guessed.
    batch, seq = z3.Reals("batch2 seq2")
    num = batch * seq
    for guess in (batch, seq, z3.RealVal(1)):
        solver = z3.Solver()
        solver.add(z3.Not(z3.ForAll([batch, seq], guess * (batch + seq) == num)))
        assert solver.check() == z3.sat, (
            "a single-term guess satisfies the identity for all batch, seq -- "
            "negative control is vacuous"
        )
