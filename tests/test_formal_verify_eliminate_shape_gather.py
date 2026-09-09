"""Formal check for EliminateShapeGather (eliminate_shape_gather.h).

``patternMatchPredicate`` matches ``Gather(Shape(X), indices)`` -- a
``Gather`` node whose data input (input 0) is the output of a ``Shape`` node
-- ONLY when ``indices`` is a compile-time constant (``IsConstantTensor``)
AND ``X`` (the ``Shape`` node's own input) has a statically known rank
(``HasDimsOfInputOfNode``; individual dims of ``X`` may still be symbolic
``dim_param``s, only the *rank* need be known -- same predicate shape as
``eliminate_slice_after_shape``).

``runTransform`` fetches ``indices``' single scalar value
(``FetchSoleIntValueOfTensor`` -- so, like ``eliminate_slice_after_shape``,
this pass only ever handles a single-element ``indices``, not a general
multi-index ``Gather``), then reproduces ONNX ``Gather``'s own negative-index
wraparound BY HAND: ``AddYIfNegative(indices_val, end - start)`` normalizes a
negative index relative to the LENGTH OF THE SUB-RANGE ``Shape``'s own
(rarely used) ``start``/``end`` attributes select -- NOT ``X``'s full rank --
then adds ``start`` to land on ``X``'s actual axis. It asserts the result is
in bounds (``ONNX_ASSERT``, not a graceful decline -- only reachable with a
genuinely out-of-range ``indices`` on a real graph, which no valid ONNX
model would have, so out-of-range inputs are out of this file's scope, same
stance ``eliminate_slice_after_shape``'s file takes for its own asserted
invariants) and declines (leaving both ``Shape`` and ``Gather`` in place) if
the selected dim isn't statically known as a concrete int (``!dims[
indices_val].is_int || dims[indices_val].dim == -1``). Otherwise it builds a
fresh INT64 constant holding just that ONE dim's value -- shaped ``[1]`` if
``indices`` itself was rank-1, or a bare scalar if ``indices`` was rank-0 --
and rewires ``Gather``'s consumers onto it directly.

Formal content: as the task framing for this file spells out, this is the
same "trust static shape metadata equals the actual runtime shape" premise
every shape-family pass in this suite relies on and cannot itself establish
(a graph-level property no single-pass proof can prove -- taken as given,
the same honesty ``eliminate_slice_after_shape``'s file applies), COMBINED
with ``Gather``'s own single-index-extraction semantics: for a scalar/
rank-1 index, ``Gather(seq, indices)[0]`` selects exactly ``seq`` at the
negative-aware normalized index -- a standard, simple indexing fact, not
something specific to shapes. Modeled below as: an uninterpreted per-axis
"declared shape" function (``declared_dim``, standing for ``X``'s static
shape-inference metadata -- what ``dims[...]`` in ``runTransform`` reads) and
"runtime shape" function (``runtime_dim``, standing for what a real
``Shape(X)`` op would produce at runtime), related by the premise that they
agree pointwise; the index arithmetic itself
(``AddYIfNegative(indices_val, end - start) + start``, a direct
transliteration of ``runTransform``'s own two-line computation) is encoded
directly as it is definitional ONNX ``Shape``/``Gather`` spec bookkeeping,
not a hypothesis needing its own premise -- mirroring how
``eliminate_slice_after_shape``'s ``_walk_slice`` transliterates its walk
rather than hypothesizing it; and an arbitrary uninterpreted ``consumer`` for
substitution safety.

A structural finding shared verbatim with
``test_formal_verify_extract_constant_to_initializer.py`` (read that file's
own docstring for the fuller story -- this paragraph only summarizes it):
unlike every OTHER pass targeted by this suite, ``eliminate_shape_gather`` is
UNREACHABLE through onnxsim's own Python API, by deliberate design.
``onnxsim.cpp``'s ``SimplifyImpl`` builds ``config.optimizer_passes`` from
``onnx::optimization::GetFuseAndEliminationPass()`` filtered through a fixed
``always_disabled_passes = {"eliminate_shape_gather",
"extract_constant_to_initializer"}`` list -- applied not only to the default
pass set but also to ``extra_optimizers`` (the general opt-in mechanism
every other non-default pass in this suite is exercised through), so there
is no combination of ``skipped_optimizers``/``isolate()`` and/or
``extra_optimizers`` that makes it run. This was confirmed empirically while
writing this file (not just read off the source): reading
``always_disabled_passes``' definition directly in ``onnxsim/onnxsim.cpp``,
and separately building a ``Gather(Shape(X), indices)`` model that satisfies
``eliminate_shape_gather``'s own predicate (constant scalar ``indices``, ``X``
with fully known static shape) and running it through
``onnxsim.simplify(model, skipped_optimizers=isolate("eliminate_shape_gather"),
skip_constant_folding=True)`` -- the ``Gather``/``Shape`` chain survives
completely untouched, confirming the pass never fires. The comment above
``always_disabled_passes`` in ``onnxsim.cpp`` explains why: onnxsim's own
constant folder (see that comment, and ``extract_constant_to_initializer``'s
docstring) already performs the equivalent shape/gather-constant-folding
work itself via a dedicated partial-shape-evaluation step ("Shape/Gather-on-
shape into constants", per the ``FoldConstant`` comment in ``onnxsim.cpp``)
BEFORE the optimizer passes run, and folds the result into onnxsim's own
constant-node-preserving representation rather than
``eliminate_shape_gather``'s bare initializer -- so this onnx-optimizer pass
is permanently redundant with, and would fight, onnxsim's own folding
strategy, and is dropped unconditionally.

Consequently, exactly as for ``extract_constant_to_initializer``, there is no
way to run "the real compiled pass, in isolation" against a concrete model
*through onnxsim's own public Python surface* -- the surface this suite
otherwise insists on testing against (see CLAUDE.md and every other
``test_formal_verify_*.py`` file). An earlier version of the sibling
``extract_constant_to_initializer`` file drove onnxoptimizer's C++ pass class
directly via a small C++ harness compiled at test time against this
checkout's ``.setuptools-cmake-build`` artifacts; that approach was dropped
because it only works when a source build's ``compile_commands.json`` and
static libraries happen to be present in the CWD -- true in a local dev
checkout, but never true in CI's actual test job (``CIBW_TEST_COMMAND`` in
``.github/workflows/build-and-test.yml`` installs a prebuilt wheel and runs
``pytest`` against it, with no leftover CMake build tree or C++ toolchain
available), so those tests would silently skip on every real CI run and
provide no ongoing verification value there, while adding real fragility
(compiler flags, link order, static-lib layout) for a local-only benefit.
The same reasoning applies here unchanged, so this file follows suit and
does not attempt one either.

Instead, ``test_eliminate_shape_gather_unreachable_via_onnxsim_simplify``
below tests the property that actually matters for onnxsim's own users and
behavior -- that this pass never fires through the public API -- directly
and unconditionally (no special build machinery, so it always runs in CI):
if onnxsim's own ``always_disabled_passes`` filter (``onnxsim.cpp``) ever
stopped excluding this pass, this is the test that would catch it, by
observing the ``Gather``/``Shape`` chain unexpectedly collapsing into a fresh
initializer.
"""

from _formal_verify_common import isolate, prove, z3
from onnx import parser

import onnxsim


def _normalize_gather_index(indices_val, start, end):
    """Transliterates runTransform's own index arithmetic (eliminate_shape_gather.h,
    lines 45-46): ``AddYIfNegative(indices_val, end - start)`` wraps a negative
    index relative to the length of the sub-range Shape's own start/end
    attributes select, then ``+= start`` lands on X's actual axis.
    """
    sub_len = end - start
    idx = indices_val + sub_len if indices_val < 0 else indices_val
    return start + idx


def test_eliminate_shape_gather_index_normalization_matches_python_indexing():
    # Exhaustively confirms the transliterated index arithmetic always lands
    # on the same element of X's full declared-shape list that plain
    # two-step Python indexing -- first Shape's own start:end sub-range
    # (list slicing), then Gather's negative-aware element index into that
    # sub-range (list indexing) -- would select, for every valid
    # (start, end, indices_val) triple. This is pure index bookkeeping (no
    # value algebra), so a concrete exhaustive sweep is the right tool, the
    # same choice eliminate_slice_after_shape's file makes for its own
    # (more elaborate) index walk.
    dims = [10, 11, 12, 13, 14, 15]
    n = len(dims)
    for start in range(0, n + 1):
        for end in range(start, n + 1):
            sub = dims[start:end]
            for indices_val in range(-len(sub), len(sub)):
                axis = _normalize_gather_index(indices_val, start, end)
                assert dims[axis] == sub[indices_val]


def test_eliminate_shape_gather_is_sound():
    declared_dim = z3.Function("declared_dim", z3.IntSort(), z3.RealSort())
    runtime_dim = z3.Function("runtime_dim", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    a, idx, start, end = z3.Ints("a idx start end")

    # "X's declared (static shape-inference) per-axis dims equal its actual
    # runtime per-axis dims" -- the premise this suite's shape-family passes
    # all rely on and cannot themselves establish; taken as given here, same
    # as eliminate_slice_after_shape's file does for its own comparable
    # claim.
    shape_matches_runtime = z3.ForAll([a], declared_dim(a) == runtime_dim(a))

    # runTransform's own index normalization (eliminate_shape_gather.h:45-46),
    # encoded directly -- definitional ONNX Shape/Gather bookkeeping, not a
    # hypothesis (see _normalize_gather_index above and the module
    # docstring).
    sub_len = end - start
    normalized = z3.If(idx < 0, idx + sub_len, idx)
    axis = start + normalized

    # Only in-bounds indices are in scope: the pass ONNX_ASSERTs this rather
    # than declining gracefully, so any real graph reaching runTransform
    # already satisfies it (see module docstring).
    valid_range = z3.And(0 <= start, start <= end, -sub_len <= idx, idx < sub_len)

    # Substitution safety: whatever the fresh constant this pass bakes in
    # (declared_dim(axis), since axis is exactly where runTransform reads
    # dims[indices_val].dim) is consumed by, it is indistinguishable from
    # what the ORIGINAL Gather(Shape(X), indices) would have produced at
    # runtime (runtime_dim(axis) -- Shape(X)'s runtime output at that axis,
    # by Shape's own spec, then Gather's own single-index selection, by
    # Gather's own spec), given the shape-trust premise.
    prove(
        z3.Implies(
            z3.And(shape_matches_runtime, valid_range),
            consumer(declared_dim(axis)) == consumer(runtime_dim(axis)),
        )
    )


def test_eliminate_shape_gather_negative_control_needs_shape_trust_premise():
    # Without shape_matches_runtime, declared_dim and runtime_dim are two
    # independent, fully unconstrained uninterpreted functions: the claim
    # must NOT be valid then (even restricted to the same in-bounds
    # indices), or the proof above would be vacuously true regardless of
    # what the premise says.
    declared_dim = z3.Function("declared_dim", z3.IntSort(), z3.RealSort())
    runtime_dim = z3.Function("runtime_dim", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    idx, start, end = z3.Ints("idx start end")

    sub_len = end - start
    normalized = z3.If(idx < 0, idx + sub_len, idx)
    axis = start + normalized
    valid_range = z3.And(0 <= start, start <= end, -sub_len <= idx, idx < sub_len)

    solver = z3.Solver()
    solver.add(valid_range)
    solver.add(z3.Not(consumer(declared_dim(axis)) == consumer(runtime_dim(axis))))
    assert solver.check() == z3.sat


def test_eliminate_shape_gather_unreachable_via_onnxsim_simplify():
    # Regression guard for the module docstring's central claim: unlike
    # every other pass in this suite, no combination of onnxsim.simplify's
    # own knobs runs this pass. This model satisfies eliminate_shape_gather's
    # own predicate exactly (a Gather node whose input 0 is Shape(X), a
    # constant scalar-shaped indices, and X with a fully known static rank
    # AND fully known static dims) -- it WOULD trigger the pass if it were
    # reachable. isolate("eliminate_shape_gather") skips every OTHER default
    # pass, nominally leaving only this one active, and
    # skip_constant_folding=True keeps onnxsim's separate (and, per the
    # module docstring, functionally equivalent) shape/gather constant
    # folder from folding this away on its own first, which would otherwise
    # obscure whether eliminate_shape_gather itself ever touched anything.
    # If onnxsim's own always_disabled_passes filter (onnxsim.cpp) ever
    # stopped excluding this pass, this test would start failing here (the
    # Gather/Shape chain would collapse into a fresh initializer) rather
    # than silently changing onnxsim's constant-node-preservation guarantee.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,3,4] X) => (int64[1] Y)
        <int64[1] indices = {1}>
        {
            s = Shape(X)
            Y = Gather(s, indices)
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("eliminate_shape_gather"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = {n.op_type for n in sim_model.graph.node}
    assert ops == {"Shape", "Gather"}
    initializer_names = {init.name for init in sim_model.graph.initializer}
    assert initializer_names == {"indices"}
