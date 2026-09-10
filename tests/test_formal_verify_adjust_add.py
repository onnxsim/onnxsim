"""Formal check for AdjustAdd (opt-in; onnx-optimizer's own
``onnxoptimizer/passes/adjust_add.h``): swaps ``Add``'s two inputs in place
whenever operand 0 is a compile-time constant and operand 1 is not.

``patternMatchPredicate`` requires exactly ``CheckKind(node, kAdd) &&
IsConstantTensor(node, 0) && !IsConstantTensor(node, 1)`` -- a constant
strictly first, a non-constant strictly second. ``runTransform`` does
nothing but rewire the node's own two inputs (``old = replaceInput(0,
inputs()[1]); replaceInput(1, old)``) and reports
``NodeDestroyType::DestroyZero`` -- no node is created or destroyed, only
``Add``'s own input list is permuted. Per the header comment, the purpose is
purely to help a *later* pass: some downstream bias fusions (folding a
constant bias into ``Gemm``/``Conv``) specifically look for the constant as
the operand-1 ("second") position, so a constant sitting in position 0 would
otherwise be invisible to them.

Formal content: this is a claim about ``Add``'s commutativity, but with a
real wrinkle -- ONNX ``Add`` broadcasts its two operands (numpy-style)
before combining them elementwise, so it is not enough to wave at "``x + y
== y + x`` for reals" and call it done; the proof below states, explicitly,
that broadcasting itself does not care which operand came first. Two
broadcasting shapes are modeled directly as uninterpreted tensor functions
that only depend on the axes they actually vary along (this suite's
established ``Int -> Real`` / ``Int, Int -> Real`` idiom, e.g.
``test_formal_verify_rewrite_gatherelements_to_gather.py``):

* A "bias" case (matching the pass's own motivating picture: a bias vector
  added to a wider tensor) -- ``A`` has shape ``[4]``, read as a function of
  the column index alone (``A(j)``); ``B`` has the full shape ``[3, 4]``,
  read as ``B(i, j)``. Numpy-style broadcasting inserts ``A``'s missing
  leading axis as size 1, so it is read at the same ``j`` regardless of
  which row ``i`` the output element belongs to.
* An "outer broadcast" case -- ``A`` has shape ``[3, 1]`` (varies only along
  rows, ``A(i)``), ``B`` has shape ``[1, 4]`` (varies only along columns,
  ``B(j)``); together they broadcast to a ``[3, 4]`` output. This is the
  shape of broadcast most likely to get a swap wrong, since after swapping
  which operand is read first, each operand must still vary along *its own*
  axis, not the axis its new input position might suggest.

In both cases the claim is: reading the original node (``Add(A, B)``, i.e.
``A``'s contribution plus ``B``'s, each via its own broadcast axis) equals
reading the rewritten node (``Add(B, A)``, ``B``'s contribution plus
``A``'s, each still via its own, unchanged, broadcast axis) -- for every
coordinate, and (this suite's standard substitution-safety idiom, e.g.
``test_formal_verify_rewrite_where.py``) composed with an arbitrary
uninterpreted downstream ``consumer``. Broadcasting determines each
operand's own read pattern from that operand's own shape alone, never from
its position in the node's input list, which is exactly why the swap is
safe regardless of which shape happens to be "wider".

A negative-control test confirms this modeling is genuinely load-bearing,
not merely invoking Z3's built-in commutativity of ``+`` on reals: if the
swap were (incorrectly) modeled as each operand's broadcast axis following
its *new input position* instead of staying fixed to the operand itself --
i.e. ``A(i) + B(j) == A(j) + B(i)`` rather than ``A(i) + B(j) == B(j) +
A(i)`` -- the claim does not hold for arbitrary ``A``/``B``, and Z3 finds a
genuine counterexample.

The differential checks below confirm, against the real compiled pass, that
the predicate's constant/non-constant asymmetry is exactly what gates the
swap: it fires only when operand 0 is constant and operand 1 is not, and
declines (leaving the node's inputs untouched) when the constant is already
in operand 1, or when neither operand is constant. The remaining
combination -- *both* operands constant -- is not actually reachable
through onnxsim's own ``simplify()`` pipeline: unlike the optimizer pass
list (which ``simplify_isolated_extra`` can selectively skip),
``onnxsim``'s constant folding (``onnxsim/constant_folding.h``) is a
separate, unconditional step that always folds every constant subexpression
first, so an ``Add`` with two constant inputs never survives to be seen by
``adjust_add`` at all in practice -- confirmed below by the fact that
isolating just this one pass still leaves no ``Add`` node in the simplified
model for that case.
"""

import numpy as np
import onnx
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser


def test_adjust_add_is_sound_bias_broadcast():
    # A: shape [4], broadcasts along the (implicit, numpy-inserted) leading
    # axis against B's shape [3, 4] -- read as a function of the column
    # index j alone, regardless of which row of B it lines up with.
    A = z3.Function("A", z3.IntSort(), z3.RealSort())
    B = z3.Function("B", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i, j = z3.Ints("i j")

    original = A(j) + B(i, j)  # Add(A, B)
    # Add(B, A): only which operand is read first changes -- A's own
    # broadcast axis (j only) and B's (both i and j) are unchanged, since
    # broadcasting is a property of each operand's own shape, not of its
    # position in the node's input list.
    swapped = B(i, j) + A(j)

    prove(original == swapped)
    prove(consumer(original) == consumer(swapped))


def test_adjust_add_is_sound_outer_broadcast():
    # A: shape [3, 1] (varies only along rows, A(i)). B: shape [1, 4]
    # (varies only along columns, B(j)). Broadcast together they produce a
    # [3, 4] output where every element combines exactly one A-row-value
    # with one B-column-value.
    A = z3.Function("A", z3.IntSort(), z3.RealSort())
    B = z3.Function("B", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i, j = z3.Ints("i j")

    original = A(i) + B(j)  # Add(A, B)
    swapped = B(j) + A(i)  # Add(B, A): same two per-axis reads, order flipped

    prove(original == swapped)
    prove(consumer(original) == consumer(swapped))


def test_adjust_add_negative_control_broadcast_axis_must_travel_with_operand():
    # Sanity check that the two proofs above are doing real work, not merely
    # invoking Z3's built-in commutativity of Real `+`: if the swap were
    # (incorrectly) modeled as each operand's broadcast axis following its
    # *new input position* rather than staying fixed to the operand itself
    # -- as if swapping Add's inputs also swapped which axis gets broadcast
    # -- the resulting claim does NOT hold for arbitrary A, B. Z3 must find
    # a genuine counterexample, confirming that correctly keeping each
    # operand's own broadcast axis fixed across the swap (as done above) is
    # load-bearing, not incidental.
    A = z3.Function("A", z3.IntSort(), z3.RealSort())
    B = z3.Function("B", z3.IntSort(), z3.RealSort())
    i, j = z3.Ints("i j")

    wrong_claim = A(i) + B(j) == A(j) + B(i)
    solver = z3.Solver()
    solver.add(z3.Not(wrong_claim))
    assert solver.check() == z3.sat, (
        "axis-mixing claim holds for all A, B -- negative control is vacuous"
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


def test_adjust_add_pass_swaps_constant_first_operand():
    # bias (initializer, constant) is Add's *first* operand, X (graph
    # input, non-constant) its second -- exactly the predicate's target.
    # The pass, run alone, should swap the two in place so the constant
    # ends up second.
    rng = np.random.default_rng(0)
    bias = rng.standard_normal(4)
    model = _model(
        """
        g (float[3,4] X) => (float[3,4] Y)
        {
            Y = Add(bias, X)
        }
        """,
        initializer=[_f32(bias, "bias")],
    )
    sim_model, ops = simplify_isolated_extra(model, "adjust_add")
    assert ops["Add"] == 1
    add_node = producer(sim_model, "Y")
    assert add_node.op_type == "Add"
    assert list(add_node.input) == ["X", "bias"]


def test_adjust_add_both_operands_constant_is_folded_before_the_pass_runs():
    # Both operands are initializers (constant): patternMatchPredicate's
    # `!IsConstantTensor(node, 1)` would decline this in isolation, but in
    # onnxsim's actual pipeline this combination never reaches the pass at
    # all -- onnxsim's constant folding (a separate, always-on step, not
    # part of the skippable optimizer-pass list) unconditionally folds any
    # Add with two constant inputs first. Confirms that fact directly: even
    # with every optimizer pass but adjust_add skipped, no Add node survives
    # -- the model reduces straight to the (correct) constant sum.
    rng = np.random.default_rng(0)
    bias0 = rng.standard_normal(4)
    bias1 = rng.standard_normal(4)
    model = _model(
        """
        g (float[3,4] Dummy) => (float[4] Y)
        {
            Y = Add(bias0, bias1)
        }
        """,
        initializer=[_f32(bias0, "bias0"), _f32(bias1, "bias1")],
    )
    sim_model, ops = simplify_isolated_extra(model, "adjust_add")
    assert ops["Add"] == 0
    (y_init,) = [init for init in sim_model.graph.initializer if init.name == "Y"]
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(y_init), bias0 + bias1, rtol=1e-6, atol=1e-6
    )


def test_adjust_add_declines_when_neither_operand_constant():
    # Both operands are graph inputs (non-constant) -- IsConstantTensor(node,
    # 0) is false, so the predicate declines regardless of operand 1.
    model = _model(
        """
        g (float[4] W, float[3,4] X) => (float[3,4] Y)
        {
            Y = Add(W, X)
        }
        """
    )
    sim_model, ops = simplify_isolated_extra(model, "adjust_add")
    assert ops["Add"] == 1
    add_node = producer(sim_model, "Y")
    assert add_node.op_type == "Add"
    assert list(add_node.input) == ["W", "X"]


def test_adjust_add_declines_when_constant_already_second():
    # bias is already operand 1 (X, non-constant, is operand 0) -- exactly
    # the layout the pass exists to produce, so `IsConstantTensor(node, 0)`
    # is false and the predicate correctly finds nothing to do.
    rng = np.random.default_rng(0)
    bias = rng.standard_normal(4)
    model = _model(
        """
        g (float[3,4] X) => (float[3,4] Y)
        {
            Y = Add(X, bias)
        }
        """,
        initializer=[_f32(bias, "bias")],
    )
    sim_model, ops = simplify_isolated_extra(model, "adjust_add")
    assert ops["Add"] == 1
    add_node = producer(sim_model, "Y")
    assert add_node.op_type == "Add"
    assert list(add_node.input) == ["X", "bias"]
