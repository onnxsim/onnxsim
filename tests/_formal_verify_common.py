"""Shared helpers for the formal-verification tests (tests/test_formal_verify_*.py).

Each targeted optimizer rewrite gets two independent checks:

1. A hand-written Z3 proof (``prove`` below) that the rewrite's documented
   algebra is a sound equivalence -- e.g. that composing two Transpose
   permutations the way fuse_consecutive_transposes.h does is equivalent to
   applying both in sequence, for every index and every tensor value, not
   just a finite sample.
2. A differential check (``simplify_isolated`` below) that the actual
   compiled pass, run alone via ``skipped_optimizers``, produces output
   consistent with that same algebra on a concrete model.

(1) alone only proves the hand-written *spec* is sound; there is no way to
symbolically execute onnxsim's C++ pass code itself from Python (no per-pass
hook is exposed across the nanobind boundary -- only whole-model
``optimize()`` is). (2) narrows that gap by validating the compiled pass
against the same spec on concrete numbers. Neither replaces onnxsim's own
random-sampling ``--check`` (see onnxsim/model_checking.py), which still runs
alongside via ``simplify_isolated``'s own ``check_n``.

z3-solver backs this and is an optional dependency (the ``verify`` extra):
these tests are skipped, not failed, when it isn't installed.
"""

import collections

import onnxsim.onnxsim_cpp2py_export as C
import pytest

import onnxsim

z3 = pytest.importorskip(
    "z3", reason="formal verification tests need the 'verify' extra (z3-solver)"
)


def isolate(*pass_names):
    """``skipped_optimizers`` value that leaves only the named default passes active."""
    names = set(pass_names)
    all_default = set(C._list_optimizers())
    unknown = names - all_default
    assert not unknown, f"not a default onnxsim optimizer pass: {sorted(unknown)}"
    return sorted(all_default - names)


def simplify_isolated(model, *pass_names, check_n=3):
    sim_model, check_ok = onnxsim.simplify(
        model, check_n=check_n, skipped_optimizers=isolate(*pass_names)
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def simplify_isolated_extra(model, *pass_names, check_n=3):
    """Like ``simplify_isolated``, but for opt-in ("other") passes -- ones not
    part of the default set, which must be named via ``extra_optimizers`` to
    run at all (see ``onnxsim --list-other-optimizers``). Every default pass
    is skipped, so only the named opt-in pass(es) run.
    """
    names = set(pass_names)
    all_other = set(C._list_other_optimizers())
    unknown = names - all_other
    assert not unknown, f"not an opt-in onnxsim optimizer pass: {sorted(unknown)}"
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        extra_optimizers=sorted(names),
        skipped_optimizers=sorted(C._list_optimizers()),
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def producer(model, output_name):
    """The node that produces ``output_name`` in ``model``'s graph.

    Isolating one opt-in pass via ``simplify_isolated_extra`` runs it without
    its usual companion dead-code pass (``eliminate_deadend`` is a default
    pass, skipped here like every other one) -- so a rewrite that leaves its
    old input dangling (rather than deleting it outright) can leave a second,
    dead copy of the rewrite's own output type sitting unused elsewhere in
    the graph. A raw ``Counter`` of op types then overcounts; walking
    backward from a real graph output instead finds the live computation
    regardless of what dead code is also lying around.
    """
    return next(n for n in model.graph.node if output_name in n.output)


#: Solver budget for the last-resort strategy of one proof, in seconds. Real proofs here finish in
#: well under a minute; the cap exists so a proof that Z3 happens to find hard fails (with a clear
#: message) instead of hanging the whole CI job until GitHub's 6-hour limit.
PROOF_TIMEOUT_S = 120


def _then(*names):
    return lambda ctx: z3.Then(*names, ctx=ctx).solver()


def _solver_with(**params):
    def make(ctx):
        solver = z3.Solver(ctx=ctx)
        for key, value in params.items():
            solver.set(key.replace("__", "."), value)
        return solver

    return make


# Strategies tried in order, each in the proof's own fresh Z3 context: (name, factory, cap in
# seconds -- None means the full ``timeout``). A strategy that proves the claim (unsat) or refutes it
# (sat) is conclusive and sound, so "first conclusive wins" cannot weaken what is proved; it only
# decides how long finding the proof takes. The first four have SHORT caps: no single strategy is
# fast on every proof (measured, fresh context, docs/formal-verify-robustness.md), so a short first
# round bounds the time wasted on a strategy that is wrong for a proof, and only the last, plain
# default solver gets the full ``timeout``.
#   quantized MAC bounds (nonlinear reals):   qfnra 0.2-0.8 s;  default 0.2-40 s
#   four-corner formula (uninterpreted fn):   arith.solver=2 0.0 s;  smt tactic 0.1 s;  default: gave up
#   flat-BEV-index bijection / scatter witness: default 0.0 s;  qfnra: burns its whole cap
_STRATEGIES = (
    ("default solver, short", lambda ctx: z3.Solver(ctx=ctx), 3),
    ("qfnra", lambda ctx: z3.Tactic("qfnra", ctx=ctx).solver(), 6),
    (
        "simplify;propagate-values;ctx-simplify;smt",
        _then("simplify", "propagate-values", "ctx-simplify", "smt"),
        3,
    ),
    ("default + smt.arith.solver=2", _solver_with(smt__arith__solver=2), 3),
    ("default solver", lambda ctx: z3.Solver(ctx=ctx), None),
)


def prove(claim, msg="rewrite is not a sound equivalence", timeout=PROOF_TIMEOUT_S):
    """Prove ``claim`` valid.

    Mirrors z3's own ``prove()`` helper (free variables in ``claim`` are
    implicitly universally quantified: ``claim`` is valid iff its negation is
    unsatisfiable), but raises with the counterexample instead of printing it,
    so a broken proof fails the test with a useful message.

    Robustness (see docs/formal-verify-robustness.md): Z3's search depends on the AST ids and
    other state earlier work leaves in the *global* context, so the same proof could finish in
    0.2 s after some tests and not at all (measured: > 250 s) after others -- one such proof
    hung CI's formal-verification job for the full 6-hour limit. Each proof is therefore
    copied (``translate``) into a fresh ``z3.Context`` -- deterministic numbering, independent of
    earlier tests -- and tried with a short portfolio of strategies (see ``_STRATEGIES``), the
    last of which gets ``timeout`` seconds. If none is conclusive the test fails saying so (never
    calls ``model()`` on an unknown result, never silently passes).
    """
    if isinstance(claim, bool):
        claim = z3.BoolVal(claim)
    ctx = z3.Context()
    negated = z3.Not(claim.translate(ctx), ctx=ctx)
    gave_up = []
    for name, make_solver, cap in _STRATEGIES:
        budget = timeout if cap is None else min(cap, timeout)
        solver = make_solver(ctx)
        solver.set("timeout", int(budget * 1000))
        solver.add(negated)
        result = solver.check()
        if result == z3.unsat:
            return
        if result == z3.sat:
            raise AssertionError(f"{msg}: counterexample {solver.model()}")
        gave_up.append(f"{name} ({budget:g} s): {solver.reason_unknown()}")
    raise AssertionError(
        f"{msg}: solver gave up on every strategy ({'; '.join(gave_up)}) "
        "-- the claim is neither proved nor refuted"
    )
