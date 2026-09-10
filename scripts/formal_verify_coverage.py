#!/usr/bin/env python3
"""Report how many of onnxsim's registered optimizer passes have a Z3
soundness proof under ``tests/test_formal_verify_*.py``.

This is a distinct metric from the C++ line/branch coverage the
``coverage.yml`` workflow already produces (via gcovr): that measures how
much of a pass's *code* executes when the test suite runs, regardless of
whether any test proves the pass's rewrite is actually sound. This script
instead measures how many of the *passes themselves* -- named rewrite
rules registered with onnxoptimizer -- have a hand-written Z3 proof at all,
against the full universe of passes ``onnxsim --list-default-optimizers``
and ``--list-other-optimizers`` report.

Usage: ``python scripts/formal_verify_coverage.py`` (needs the built
``onnxsim_cpp2py_export`` extension importable, same as running the tests).
"""

import glob
import os

import onnxsim.onnxsim_cpp2py_export as C

_TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")

# test_formal_verify_*.py files in this family that verify a property shared
# across ONNX ops onnxsim's quantization relies on, or a derived lemma
# composed from other such files, rather than one of onnxsim's own
# optimizer passes (see each such file's own module docstring) -- these
# don't correspond to a pass name and aren't counted against the pass
# universe below.
_NOT_A_PASS = {
    "quantize_round_trip",
    "quantized_mac_bound",
    # graph_grad's VJP rules (onnxsim/graph_grad.py / graph_grad.cpp) are not
    # onnxoptimizer passes at all -- they're proved here for the same reason,
    # but have no C._list_optimizers()/_list_other_optimizers() name to match
    # against.
    "grad_transpose",
    "grad_add",
    "grad_mul",
    "grad_relu",
    "grad_matmul",
    "grad_gemm",
    "grad_conv",
    # onnxsim::TryExactDivide (onnxsim/sym_expr.cpp) is a shape-arithmetic
    # primitive, not an onnxoptimizer pass either.
    "sym_expr_exact_divide",
}


def _proved_pass_names():
    proved = set()
    for path in glob.glob(os.path.join(_TESTS_DIR, "test_formal_verify_*.py")):
        name = os.path.basename(path)[len("test_formal_verify_") : -len(".py")]
        if name not in _NOT_A_PASS:
            proved.add(name)
    return proved


def main():
    default_passes = set(C._list_optimizers())
    other_passes = set(C._list_other_optimizers())
    all_passes = default_passes | other_passes
    proved = _proved_pass_names()

    unknown = proved - all_passes
    if unknown:
        raise SystemExit(
            "test_formal_verify_*.py names a pass onnxsim doesn't register "
            f"(stale after a rename?): {sorted(unknown)}"
        )

    proved_default = proved & default_passes
    proved_other = proved & other_passes
    unproved_default = sorted(default_passes - proved)

    print(
        f"Default (on-by-default) passes: {len(proved_default)}/{len(default_passes)} proved"
    )
    print(
        f"Other (opt-in) passes:          {len(proved_other)}/{len(other_passes)} proved"
    )
    print(
        f"All registered passes:          {len(proved)}/{len(all_passes)} proved "
        f"({100 * len(proved) / len(all_passes):.1f}%)"
    )
    print()
    print("Proved:")
    for name in sorted(proved):
        print(f"  {name}")
    print()
    print(f"Not yet proved, on by default ({len(unproved_default)}):")
    for name in unproved_default:
        print(f"  {name}")


if __name__ == "__main__":
    main()
