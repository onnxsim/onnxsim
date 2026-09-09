"""Formal check for RewriteGatherElementsToGather (opt-in; onnxsim's own
``onnxsim/passes/rewrite_gatherelements_to_gather.h``): rewrites
``GatherElements(data, indices, axis)`` into plain ``Gather(data,
representative, axis)`` in the case where a full-rank, elementwise
``indices`` tensor turns out not to actually vary along any axis other than
``axis`` -- i.e. it is really a broadcast of a 1-D index vector along that
axis, just spelled out to full rank.

``GatherElements`` is the fundamentally more expressive op:
``output[i0,...,i_{r-1}] = data[i0,...,i_{axis-1}, indices[i0,...,i_{r-1}],
i_{axis+1},...,i_{r-1}]`` -- the index used can depend on *every* coordinate
of ``indices``. Plain ``Gather`` can only apply the *same* index set
uniformly across every combination of the other axes (an outer-product
gather). The two are equivalent only when ``indices``, despite having full
rank, happens not to vary along axes other than ``axis`` -- exactly the
invariance property the pass's own constant-folding check verifies before
firing (see the header comment's flat-index-arithmetic derivation, `i /
stride[axis] % shape[axis]` isolating the axis coordinate and `axis_coord *
stride[axis]` being the representative position with every other coordinate
forced to 0 -- easy to get backwards by zeroing the wrong coordinate, per
that file's own warning).

This proof models a concrete rank-2 case (``data``/``indices`` shape
``[3,4]``, ``axis=1``): ``indices`` is an uninterpreted ``Int,Int -> Int``
function constrained by the invariance hypothesis
(``ForAll([i0,i1,i0p], indices(i0,i1) == indices(i0p,i1))`` -- it does not
depend on the non-``axis`` coordinate ``i0``), ``data`` an uninterpreted
``Int,Int -> Real`` function, ``GatherElements`` modeled directly per its
definitional formula (``output(i0,i1) = data(i0, indices(i0,i1))``), and
``Gather`` modeled as broadcasting a 1-D ``representative`` vector
(``output(i0,i1) = data(i0, representative(i1))``) where ``representative``
is defined as ``indices`` at the arbitrary-but-fixed other coordinate 0
(``representative(i1) = indices(0, i1)``, matching the header comment's "take
the slice at all-other-axes-coordinate 0" description). The invariance
hypothesis is exactly what lets ``indices(i0, i1)`` be swapped for
``representative(i1) = indices(0, i1)`` inside ``data``'s first argument --
without it, the two computations can disagree (confirmed below by a
negative-control test in which Z3 finds a genuine counterexample).

Since negative index values in ``indices`` use the same convention as
``Gather``'s own ``indices`` input (per the header comment), the extracted
representative vector is handed to the replacement ``Gather`` node
unchanged -- unlike ``rewrite_gathernd_to_gather``, there is no
per-column normalization step here to get wrong, so the differential check
below only needs to confirm a negative value survives verbatim, not that
some combination arithmetic was done correctly on it.
"""

import numpy as np
import onnx
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser


def test_rewrite_gatherelements_to_gather_is_sound():
    # data: uninterpreted rank-2 buffer indexed (axis0, axis1). indices:
    # uninterpreted, but constrained by the invariance hypothesis to not
    # depend on axis0 (the non-`axis` coordinate, for axis=1) -- exactly the
    # property the pass's own O(size) scan verifies before firing.
    data = z3.Function("data", z3.IntSort(), z3.IntSort(), z3.RealSort())
    indices = z3.Function("indices", z3.IntSort(), z3.IntSort(), z3.IntSort())
    representative = z3.Function("representative", z3.IntSort(), z3.IntSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    i0, i1, i0p = z3.Ints("i0 i1 i0p")

    # Invariance hypothesis: indices doesn't actually vary along axis 0 (the
    # non-`axis` axis, for axis=1) -- mirrors runTransform's own check.
    invariance = z3.ForAll([i0, i1, i0p], indices(i0, i1) == indices(i0p, i1))
    # representative(i1) is indices' value at the arbitrary-but-fixed other
    # coordinate 0 -- the header comment's "slice at all-other-axes-coordinate
    # 0" (here just coordinate 0, since axis0 is the only non-axis axis).
    representative_def = z3.ForAll([i1], representative(i1) == indices(0, i1))

    # GatherElements(data, indices, axis=1), per its own definitional
    # formula: output[i0,i1] = data[i0, indices[i0,i1]].
    gather_elements_out = data(i0, indices(i0, i1))
    # Gather(data, representative, axis=1): output[i0,i1] =
    # data[i0, representative[i1]] -- the same 1-D index vector applied
    # uniformly across every i0 (the outer-product semantics plain Gather
    # is restricted to).
    gather_out = data(i0, representative(i1))

    hypotheses = z3.And(invariance, representative_def)

    # The core equivalence, for every (i0, i1).
    prove(z3.Implies(hypotheses, gather_elements_out == gather_out))

    # Full substitution-safety claim, composed with an arbitrary downstream
    # consumer (matching this suite's established style, e.g.
    # test_formal_verify_rewrite_where.py) -- the rewrite is safe for any
    # consumer of the output, not just an equality check on the raw value.
    prove(z3.Implies(hypotheses, consumer(gather_elements_out) == consumer(gather_out)))


def test_rewrite_gatherelements_to_gather_negative_control_requires_invariance():
    # Sanity check that the proof above is genuine, not vacuous: without the
    # invariance hypothesis, GatherElements(data, indices, axis=1) ==
    # Gather(data, representative, axis=1) (representative still defined as
    # indices(0, i1)) must NOT hold for every data/indices -- Z3 should find
    # a real counterexample where indices genuinely varies elementwise
    # (differs between i0 and some other row) so the two computations
    # disagree.
    data = z3.Function("data", z3.IntSort(), z3.IntSort(), z3.RealSort())
    indices = z3.Function("indices", z3.IntSort(), z3.IntSort(), z3.IntSort())
    representative = z3.Function("representative", z3.IntSort(), z3.IntSort())

    i0, i1 = z3.Ints("i0 i1")
    representative_def = z3.ForAll([i1], representative(i1) == indices(0, i1))

    gather_elements_out = data(i0, indices(i0, i1))
    gather_out = data(i0, representative(i1))

    claim = z3.ForAll([i0, i1], gather_elements_out == gather_out)

    solver = z3.Solver()
    solver.add(representative_def)
    solver.add(z3.Not(claim))
    assert solver.check() == z3.sat


def _i64(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)


def test_rewrite_gatherelements_to_gather_pass_fires_when_invariant():
    # indices genuinely has shape [3,4] (full rank, matching data) but every
    # row is the same [0, 2, 1, 3] pattern -- i.e. broadcast down axis 0, so
    # it doesn't actually vary along the non-axis (axis 0) dimension. The
    # pass, run alone, should rewrite to Gather(data, [0,2,1,3], axis=1).
    indices = np.array([[0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3]], dtype=np.int64)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[3,4] data) => (float[3,4] Y)
        {
            Y = GatherElements<axis = 1>(data, indices)
        }
        """
    )
    model.graph.initializer.append(_i64(indices, "indices"))
    sim_model, ops = simplify_isolated_extra(model, "rewrite_gatherelements_to_gather")
    assert ops["GatherElements"] == 0
    gather_node = producer(sim_model, "Y")
    assert gather_node.op_type == "Gather"
    assert gather_node.input[0] == "data"
    new_indices_init = next(
        init
        for init in sim_model.graph.initializer
        if init.name == gather_node.input[1]
    )
    assert list(onnx.numpy_helper.to_array(new_indices_init)) == [0, 2, 1, 3]
    axis_attr = next(a for a in gather_node.attribute if a.name == "axis")
    assert axis_attr.i == 1


def test_rewrite_gatherelements_to_gather_declines_when_genuinely_elementwise():
    # indices varies per row (a real elementwise GatherElements use) --
    # the invariance check must fail, so the pass, run alone, must leave
    # GatherElements untouched.
    indices = np.array([[0, 2, 1, 3], [1, 0, 3, 2], [2, 3, 0, 1]], dtype=np.int64)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[3,4] data) => (float[3,4] Y)
        {
            Y = GatherElements<axis = 1>(data, indices)
        }
        """
    )
    model.graph.initializer.append(_i64(indices, "indices"))
    sim_model, ops = simplify_isolated_extra(model, "rewrite_gatherelements_to_gather")
    assert ops["GatherElements"] == 1
    node = producer(sim_model, "Y")
    assert node.op_type == "GatherElements"


def test_rewrite_gatherelements_to_gather_declines_when_indices_not_constant():
    # indices is a runtime graph input, not a compile-time constant -- the
    # predicate's FetchConstantTensor check fails, so the pass must decline
    # regardless of what values indices would happen to take at runtime.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[3,4] data, int64[3,4] indices) => (float[3,4] Y)
        {
            Y = GatherElements<axis = 1>(data, indices)
        }
        """
    )
    sim_model, ops = simplify_isolated_extra(model, "rewrite_gatherelements_to_gather")
    assert ops["GatherElements"] == 1
    node = producer(sim_model, "Y")
    assert node.op_type == "GatherElements"


def test_rewrite_gatherelements_to_gather_negative_index_carries_through_unchanged():
    # indices includes a negative value (-2, -1); it is broadcast down
    # axis 0 the same way as the basic firing case, so the invariance check
    # still passes. Per the header comment, GatherElements' negative-index
    # convention already matches Gather's own, so the extracted
    # representative vector must carry the negative values through to the
    # new Gather node completely unchanged -- no renormalization step (in
    # contrast to rewrite_gathernd_to_gather, which does normalize).
    indices = np.array([[0, -2, 1, -1], [0, -2, 1, -1], [0, -2, 1, -1]], dtype=np.int64)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[3,4] data) => (float[3,4] Y)
        {
            Y = GatherElements<axis = 1>(data, indices)
        }
        """
    )
    model.graph.initializer.append(_i64(indices, "indices"))
    sim_model, ops = simplify_isolated_extra(model, "rewrite_gatherelements_to_gather")
    assert ops["GatherElements"] == 0
    gather_node = producer(sim_model, "Y")
    assert gather_node.op_type == "Gather"
    new_indices_init = next(
        init
        for init in sim_model.graph.initializer
        if init.name == gather_node.input[1]
    )
    assert list(onnx.numpy_helper.to_array(new_indices_init)) == [0, -2, 1, -1]
