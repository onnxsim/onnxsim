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


def prove(claim, msg="rewrite is not a sound equivalence"):
    """Prove ``claim`` valid.

    Mirrors z3's own ``prove()`` helper (free variables in ``claim`` are
    implicitly universally quantified: ``claim`` is valid iff its negation is
    unsatisfiable), but raises with the counterexample instead of printing it,
    so a broken proof fails the test with a useful message.
    """
    solver = z3.Solver()
    solver.add(z3.Not(claim))
    result = solver.check()
    assert result == z3.unsat, f"{msg}: counterexample {solver.model()}"
