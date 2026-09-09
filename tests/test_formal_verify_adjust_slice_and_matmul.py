"""Formal check for AdjustSliceAndMatmul (opt-in; onnx-optimizer's own
``onnxoptimizer/passes/adjust_slice_and_matmul.h``): rewrites
``Y = MatMul(Slice(data, start, end, axes), rhs)`` into
``Y = Slice(MatMul(data, rhs), start, end, axes)`` -- i.e. it swaps the ORDER
of ``Slice`` and ``MatMul``, moving the slice from before the matmul to
after it.

``patternMatchPredicate`` requires, per ``CheckKind(node, kMatMul, 0,
kSlice)`` and the checks alongside it: ``node`` is a ``MatMul`` whose operand
0 is produced by a ``Slice``; the matmul's operand 1 (``rhs``) is a
compile-time constant (``IsConstantTensor(node, 1)``); the Slice's own data
input (``data``) is a compile-time constant too (``IsConstantTensor(node, 0,
0)``); the Slice has an explicit, constant ``axes`` input (4th input,
``GetInputsOfPreNode(node, 0).size() >= 4`` and ``IsConstantTensor(node, 0,
3)``); the Slice's output has exactly one use (``node->inputs()[0]->uses()
.size() == 1``, i.e. nothing else already reads the sliced value); and,
crucially, none of the Slice's ``axes`` (normalized via ``AddYIfNegative``)
equal ``rank - 1`` where ``rank`` is ``data``'s own static rank -- the slice
must never touch ``data``'s LAST axis, which for a 2-D ``MatMul`` operand is
exactly the contraction axis. ``runTransform`` builds a new ``MatMul(data,
rhs)`` and a new ``Slice`` reading that MatMul's output with the OLD Slice's
own starts/ends/axes(/steps) inputs verbatim, rewires ``n``'s uses onto the
new Slice, and destroys only the original ``MatMul`` node (``DestroyOne``) --
the original ``Slice`` node is left dangling (its output now unused), not
cleaned up by this pass itself (``eliminate_deadend``'s job, a separate
default pass, mirroring the same dangling-producer subtlety documented in
e.g. ``test_formal_verify_fuse_consecutive_slices.py``). Per the header
comment, the purpose is to expose ``MatMul(data, rhs)`` -- a matmul of two
FULL constant tensors, trivially foldable by onnxsim's separate constant
folder -- where before the rewrite the two constant operands were `data`'s
own *sliced view* and `rhs`, a shape that onnxsim's constant folding may not
recognize/fold as eagerly (e.g. once `data` is large enough to sit above
onnxsim's constant-folding size threshold) as a single self-contained
``MatMul`` of two already-whole constants.

Formal content: this is a genuine claim about ``MatMul``/``Slice``
commuting, valid only when the ``Slice`` restricts a NON-contracted axis of
the sliced operand. With ``data`` 2-D (``[M, K]``) and ``rhs`` 2-D
(``[K, N]``), ``MatMul(data, rhs)[i, n] = sum_k data(i, k) * rhs(k, n)``. If
the Slice restricts axis 0 (rows, ``i``) of ``data`` to a sub-range
``[start, end)``, then reading ``Slice(MatMul(data, rhs))`` at local row
``i'`` is ``MatMul(data, rhs)[start + i', n]``, i.e.
``sum_k data(start + i', k) * rhs(k, n)`` -- and this is EXACTLY
``MatMul(Slice(data, start, end, axes=[0]), rhs)[i', n]``, since
``Slice(data)``'s own row ``i'`` reads ``data``'s row ``start + i'`` and the
contraction sum is otherwise untouched. The two sides don't merely turn out
numerically equal, they are literally the same expression term-by-term --
restricting a non-contracted axis of ``data`` and restricting the
corresponding axis of the MatMul *output* are the same coordinate, only
relabeled by a fixed offset, so which order the Slice happens in cannot
matter.

That identity breaks down completely once the Slice touches the CONTRACTED
axis (``K``) instead -- exactly why the predicate excludes ``rank - 1``. If
``axes=[1]`` restricts ``data``'s own ``K`` axis to a genuine sub-range
``[start, end)`` (``L = end - start < K``), the ORIGINAL graph computes, for
this to even be a well-typed ``MatMul``, ``rhs`` itself having only ``L``
rows: ``sum_{l=0}^{L-1} data(i, start + l) * rhs(l, n)`` -- a sum over ONLY
the sliced ``K`` range. Blindly moving that same ``axes=[1]`` past the
``MatMul`` -- as ``runTransform`` would do if the predicate didn't exclude
this case -- makes it slice the MatMul's OUTPUT along axis 1 instead, which
is ``N`` (the free/output axis), not ``K`` at all: an entirely different
operation ("restrict which output column is read") that happens to share
the same numeral axis index purely by coincidence of both tensors being
2-D. The negative-control test below proves these two computations
genuinely disagree in general. (There is also a second, structural reason
the rewrite can't even be attempted here: for the original graph to be
well-typed, ``rhs``'s own row count must already equal the slice length
``L``, not ``data``'s full ``K`` -- so the "rewritten" ``MatMul(data, rhs)``
the pass would build, with ``data``'s full, un-sliced ``K`` rows against
``rhs``'s ``L`` rows, is shape-mismatched outright whenever ``L < K``. The
value-level disagreement proved below is the deeper reason; this shape
mismatch is a second, independent tripwire that would catch the same
mistake even without appealing to it.)

Both proofs model ``data`` as an uninterpreted ``Int, Int -> Real`` function
(``data(i, k)``) and ``rhs`` as ``Int, Int -> Real`` (``rhs(k, n)``), with a
small fixed contraction width ``K = 3`` so the summation is a finite,
concrete Z3 expression rather than a symbolic-length one, and are composed
with an arbitrary uninterpreted ``consumer`` per this suite's
substitution-safety idiom (see ``test_formal_verify_rewrite_where.py``).

The differential checks below confirm, against the real compiled pass: (1) a
basic firing case -- ``Slice`` on axis 0 (a non-contracted axis) of constant
``data``, matmul'd with constant ``rhs`` -- rewrites to
``MatMul(data, rhs)`` followed by ``Slice(..., axes=[0])`` with the exact
node wiring ``runTransform`` describes, and (bonus) that onnxsim's normal,
full ``simplify()`` pipeline reduces the whole thing to the correct folded
constant; (2) the pass declines when the Slice touches ``data``'s last axis
(the contraction axis); (3) the pass declines when ``rhs`` is not a
compile-time constant (a graph input); and (4) the pass declines when the
Slice's own ``axes`` input isn't a compile-time constant. Cases (1) and (2)
both need ``skip_constant_folding=True`` to observe anything at all: every
input the predicate requires to be constant (``data``, ``rhs``, ``axes``,
and implicitly ``starts``/``ends``) is *always* constant whenever the
pattern can match in the first place, so with constant folding left on,
onnxsim's separate, unconditional constant-folding step (see
``test_formal_verify_adjust_add.py`` for the same subtlety) folds the
``Slice`` -- and, in case (1), the whole graph -- away before this pass ever
runs, regardless of whether the pass would have fired or declined. Case (3)
needs it too: with ``rhs`` non-constant the ``MatMul`` itself can't fold,
but the ``Slice`` (whose own inputs, ``data``/``starts``/``ends``/``axes``,
are still all constant) would otherwise still be pre-folded into a plain
initializer, erasing the very ``Slice`` node the predicate is supposed to
inspect and decline on for a DIFFERENT reason (``rhs`` non-constant) than
the one that would produce. Case (4) is the one exception: a genuinely
non-constant ``axes`` input means the ``Slice`` node itself is not
foldable, so it survives intact even with constant folding left on.
"""

import collections

import numpy as np
import onnx
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 3  # fixed, concrete contraction width used by both proofs below

# Concrete axis-0 (row) slice window used in the soundness proof and in the
# firing differential test -- a non-contracted axis, so the window's exact
# bounds are immaterial to the identity (see docstring): both sides reduce
# to the literal same term-by-term expression regardless of what it is.
_ROW_START, _ROW_END = 1, 3

# Concrete axis-1 (contraction, K) slice window used in the negative
# control -- a genuine sub-range (`L = 1 < K = 3`) so the original,
# restricted-sum meaning and the wrongly-swapped, full-sum-then-read-column
# meaning are actually different computations.
_K_START, _K_END = 1, 2


def test_adjust_slice_and_matmul_is_sound_for_a_non_contracted_axis():
    data = z3.Function("data", z3.IntSort(), z3.IntSort(), z3.RealSort())
    rhs = z3.Function("rhs", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i, n = z3.Ints("i n")

    def original(i, n):
        # Y = MatMul(Slice(data, start, end, axes=[0]), rhs)[i, n]: Slice's
        # own local row i reads data's row (start + i); the contraction sum
        # over k (data's *last* axis, untouched by this slice) is exactly
        # MatMul's usual sum.
        return sum(data(_ROW_START + i, k) * rhs(k, n) for k in range(_K))

    def rewritten(i, n):
        # Y = Slice(MatMul(data, rhs), start, end, axes=[0])[i, n]: MatMul
        # is computed first, in full, over the *original*, unsliced data;
        # the result is then sliced along axis 0 at local row i, i.e. read
        # at global row (start + i).
        def full_matmul(row, n):
            return sum(data(row, k) * rhs(k, n) for k in range(_K))

        return full_matmul(_ROW_START + i, n)

    prove(original(i, n) == rewritten(i, n))
    prove(consumer(original(i, n)) == consumer(rewritten(i, n)))


def test_adjust_slice_and_matmul_negative_control_contraction_axis_does_not_commute():
    # Sanity check that the predicate's `rank - 1` exclusion is load-bearing,
    # not overcautious: build the analogous claim for a Slice that restricts
    # axis 1 (K, the contraction axis) instead, and confirm Z3 finds a
    # genuine counterexample where the two orderings disagree.
    data = z3.Function("data", z3.IntSort(), z3.IntSort(), z3.RealSort())
    rhs = z3.Function("rhs", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i, j = z3.Ints("i j")
    length = _K_END - _K_START  # L = 1: a genuine sub-range of K = 3

    def original(i, j):
        # Y = MatMul(Slice(data, start=1, end=2, axes=[1]), rhs)[i, j]: only
        # the sliced K-range [start, end) is summed over -- a strictly
        # smaller sum than the full contraction (rhs here has only `length`
        # rows, matching the slice, for this to be a well-typed MatMul).
        return sum(
            data(i, _K_START + step) * rhs(_K_START + step, j) for step in range(length)
        )

    def wrongly_swapped(i, j):
        # What runTransform would build if the predicate didn't exclude this
        # axis: MatMul(data, rhs) computed over data's FULL, un-sliced K
        # range (the slice no longer restricts the sum at all), then sliced
        # along axis 1 of the *output* at position j -- i.e. axis 1 of a
        # [M, N]-shaped result is N, not K, so this reads one output column
        # rather than restricting which k terms get summed.
        return sum(data(i, k) * rhs(k, j) for k in range(_K))

    # For j ranging over the slice's own kept output positions
    # ([start, end)), the two computations should (per the excluded case)
    # generally disagree.
    wrong_claim = z3.Implies(
        z3.And(j >= _K_START, j < _K_END),
        consumer(original(i, j)) == consumer(wrongly_swapped(i, j)),
    )
    solver = z3.Solver()
    solver.add(z3.Not(wrong_claim))
    assert solver.check() == z3.sat, (
        "slicing the contraction axis should not commute with MatMul -- "
        "negative control is vacuous"
    )


def _model(body, initializer=(), opset=13, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _simplify_no_fold(model, *pass_names, check_n=3):
    # Like `simplify_isolated_extra` (see _formal_verify_common.py), but
    # with constant folding itself turned off -- needed here because every
    # input the predicate requires to be constant is, unavoidably, always
    # constant whenever the pattern can match at all, so onnxsim's separate,
    # always-on constant folding would otherwise erase the very Slice/MatMul
    # structure this pass is supposed to rewrite (or decline to rewrite)
    # before it ever runs. See this file's module docstring.
    names = set(pass_names)
    all_other = set(C._list_other_optimizers())
    unknown = names - all_other
    assert not unknown, f"not an opt-in onnxsim optimizer pass: {sorted(unknown)}"
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        extra_optimizers=sorted(names),
        skipped_optimizers=sorted(C._list_optimizers()),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def test_adjust_slice_and_matmul_pass_fires_on_non_contraction_axis_slice():
    # data [5,4] sliced on axis 0 (rows, non-contracted) to rows [1,3); rhs
    # [4,3]. Both data and rhs are constant, so the predicate should fire
    # and swap Slice/MatMul order. `skip_constant_folding=True` is required
    # -- see `_simplify_no_fold`'s docstring -- since without it the entire,
    # fully-constant graph collapses straight to a single initializer before
    # the pass ever sees a Slice or MatMul node (confirmed by the "bonus"
    # full-pipeline check at the end of this test, which folds to the same
    # result with or without this pass involved at all).
    rng = np.random.default_rng(0)
    data = rng.standard_normal((5, 4))
    rhs = rng.standard_normal((4, 3))
    model = _model(
        """
        g (float[1] Dummy) => (float[2,3] Y)
        <int64[1] starts = {1}, int64[1] ends = {3}, int64[1] axes = {0}>
        {
            sliced = Slice(data, starts, ends, axes)
            Y = MatMul(sliced, rhs)
        }
        """,
        initializer=[_f32(data, "data"), _f32(rhs, "rhs")],
    )

    sim_model, ops = _simplify_no_fold(model, "adjust_slice_and_matmul")
    assert ops["MatMul"] == 1
    assert ops["Slice"] == 2  # the new Slice + the dangling old one

    # The graph output Y is now produced by the new Slice, reading directly
    # from a new MatMul(data, rhs) -- exactly runTransform's wiring.
    new_slice = producer(sim_model, "Y")
    assert new_slice.op_type == "Slice"
    new_matmul = producer(sim_model, new_slice.input[0])
    assert new_matmul.op_type == "MatMul"
    assert list(new_matmul.input) == ["data", "rhs"]
    # The new Slice reuses the old Slice's own starts/ends/axes verbatim.
    assert list(new_slice.input[1:]) == ["starts", "ends", "axes"]

    # The old Slice node is left dangling (its own output, "sliced", is no
    # longer consumed by anything) -- eliminate_deadend is skipped here.
    dangling = [n for n in sim_model.graph.node if n.output[0] == "sliced"]
    assert len(dangling) == 1
    assert dangling[0].op_type == "Slice"
    assert list(dangling[0].input) == ["data", "starts", "ends", "axes"]

    # Bonus: running onnxsim's normal, full pipeline (constant folding back
    # on) reduces the whole graph to the correct folded constant -- the
    # pass's actual purpose. Note this holds regardless of whether the pass
    # is even enabled here, since (as above) the fully-constant graph is
    # already foldable on its own at this small size; the pass's real value
    # is exposing this same fold for `data` too large for onnxsim's
    # constant-folding size threshold to fold eagerly on its own, which is
    # out of scope for this differential test.
    sim_model_full, ok_full = onnxsim.simplify(
        model, check_n=1, extra_optimizers=["adjust_slice_and_matmul"]
    )
    assert ok_full
    assert len(sim_model_full.graph.node) == 0
    (y_init,) = [i for i in sim_model_full.graph.initializer if i.name == "Y"]
    expected = data[_ROW_START:_ROW_END] @ rhs
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(y_init), expected, rtol=1e-5, atol=1e-6
    )


def test_adjust_slice_and_matmul_declines_on_contraction_axis_slice():
    # data [5,4] sliced on axis 1 (K, the contraction axis, i.e. rank-1 for
    # this rank-2 tensor) to columns [1,3); rhs is [2,3] to match the
    # sliced width, so the (unrewritten) graph is well-typed. The predicate
    # must decline: MatMul keeps consuming Slice's output directly.
    rng = np.random.default_rng(1)
    data = rng.standard_normal((5, 4))
    rhs = rng.standard_normal((2, 3))
    model = _model(
        """
        g (float[1] Dummy) => (float[5,3] Y)
        <int64[1] starts = {1}, int64[1] ends = {3}, int64[1] axes = {1}>
        {
            sliced = Slice(data, starts, ends, axes)
            Y = MatMul(sliced, rhs)
        }
        """,
        initializer=[_f32(data, "data"), _f32(rhs, "rhs")],
    )

    sim_model, ops = _simplify_no_fold(model, "adjust_slice_and_matmul")
    assert ops["Slice"] == 1
    assert ops["MatMul"] == 1
    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    assert matmul_node.input[0] == "sliced"
    slice_node = producer(sim_model, "sliced")
    assert slice_node.op_type == "Slice"
    assert list(slice_node.input) == ["data", "starts", "ends", "axes"]


def test_adjust_slice_and_matmul_declines_when_rhs_not_constant():
    # data [5,4] sliced on axis 0 (as in the firing case), but rhs is now a
    # graph input rather than a constant -- IsConstantTensor(node, 1) is
    # false, so the predicate declines regardless of the Slice's axis.
    # `skip_constant_folding=True` is still needed: without it, the Slice
    # (whose own inputs are all constant) gets pre-folded into a plain
    # initializer before the pass runs, for the wrong reason.
    rng = np.random.default_rng(2)
    data = rng.standard_normal((5, 4))
    model = _model(
        """
        g (float[4,3] rhs) => (float[2,3] Y)
        <int64[1] starts = {1}, int64[1] ends = {3}, int64[1] axes = {0}>
        {
            sliced = Slice(data, starts, ends, axes)
            Y = MatMul(sliced, rhs)
        }
        """,
        initializer=[_f32(data, "data")],
    )

    sim_model, ops = _simplify_no_fold(model, "adjust_slice_and_matmul")
    assert ops["Slice"] == 1
    assert ops["MatMul"] == 1
    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    assert list(matmul_node.input) == ["sliced", "rhs"]
    slice_node = producer(sim_model, "sliced")
    assert slice_node.op_type == "Slice"
    assert list(slice_node.input) == ["data", "starts", "ends", "axes"]


def test_adjust_slice_and_matmul_declines_when_slice_axes_not_constant():
    # data [5,4] and rhs [4,3] both constant (as in the firing case), but
    # the Slice's own `axes` input (its 4th input) is a plain graph input
    # rather than a compile-time constant -- IsConstantTensor(node, 0, 3) is
    # false, so the predicate declines. Unlike the other two differential
    # tests above, no `skip_constant_folding` trick is needed here: a
    # genuinely non-constant `axes` makes the Slice node itself unfoldable,
    # so it survives onnxsim's normal pipeline intact regardless.
    # `check_n=0` skips onnxsim's own random-sample correctness check, which
    # would otherwise feed `axes` an out-of-range value for this rank-2
    # `data` (only -2..1 are valid).
    rng = np.random.default_rng(3)
    data = rng.standard_normal((5, 4))
    rhs = rng.standard_normal((4, 3))
    model = _model(
        """
        g (float[1] Dummy, int64[1] axes) => (float[2,3] Y)
        <int64[1] starts = {1}, int64[1] ends = {3}>
        {
            sliced = Slice(data, starts, ends, axes)
            Y = MatMul(sliced, rhs)
        }
        """,
        initializer=[_f32(data, "data"), _f32(rhs, "rhs")],
    )

    sim_model, ops = simplify_isolated_extra(
        model, "adjust_slice_and_matmul", check_n=0
    )
    assert ops["Slice"] == 1
    assert ops["MatMul"] == 1
    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    assert matmul_node.input[0] == "sliced"
    slice_node = producer(sim_model, "sliced")
    assert slice_node.op_type == "Slice"
    assert list(slice_node.input) == ["data", "starts", "ends", "axes"]
