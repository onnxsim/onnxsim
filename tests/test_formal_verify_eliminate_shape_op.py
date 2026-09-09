"""Formal check for EliminateShapeOp (eliminate_shape_op.h).

``patternMatchPredicate`` matches a ``Shape`` node ONLY when its input ``X``
has a fully statically known RANK (``HasDimsOfInputOfNode``) AND every dim in
the sub-range ``[start, end)`` that ``Shape``'s own (rarely-set) ``start``/
``end`` attributes select (``FetchStartAndEndAttrOfShape`` -- defaults to the
WHOLE rank when absent) is BOTH statically known as a concrete int and
non-negative (``dim.is_int && dim.dim >= 0``).

``runTransform`` builds a fresh INT64 constant tensor holding exactly ``X``'s
declared dims in ``[start, end)``, adds it as a graph initializer, and
rewires the ``Shape`` node's consumers onto it directly, destroying the
``Shape`` node outright (unlike ``eliminate_slice_after_shape``, this pass
deletes the ``Shape`` node itself rather than leaving it dangling).

Formal content, and why the proof here is intentionally thin: this is the
simplest and most foundational member of this suite's "trust static shape
metadata equals actual runtime shape" family (see also
``eliminate_slice_after_shape``, which shares the same premise but layers
genuinely nontrivial Slice index arithmetic -- forward/reverse walks,
negative-index normalization, asymmetric clamping -- on top of it). ONNX's
``Shape`` operator is, by definition, "return the runtime shape of the input
tensor" -- and ``X``'s own DECLARED static shape metadata (whatever shape
inference / the model's ``value_info`` claims ``X``'s shape statically is)
is, by the very meaning of "statically known", a claim that the runtime shape
WILL equal those declared values whenever the graph actually runs. That
premise is a graph-level property no single pass's proof can establish in
isolation -- it is simply taken as given here, the same honesty this suite's
other shape-family files apply (see ``eliminate_slice_after_shape``'s module
docstring for the same framing). Given that premise, replacing
``Shape(X)[start:end]``'s runtime computation with a literal constant holding
the SAME declared values is trivially sound: there is no interesting index
arithmetic at all here (no reversal, no clamping, no negative-index
normalization -- ``start``/``end`` are already resolved to a plain forward
``[start, end)`` window over ``X``'s own dims by the time this pass runs) --
this is the base case ``eliminate_slice_after_shape`` and (elsewhere in this
suite) ``eliminate_shape_gather`` build on. Modeled below the same way
``eliminate_duplicate_initializer``'s own thin proof is: an uninterpreted
per-axis "declared shape" function and an uninterpreted "runtime shape"
function, constrained equal by the premise, composed with an arbitrary
uninterpreted ``consumer`` reading a value at an axis selected by an
arbitrary ``[start, end)`` window -- the claim reduces to the premise doing
essentially all the work, stated plainly rather than dressed up as deeper
than it is. The negative-control test below confirms this isn't vacuous:
without the premise, the claim does not hold.

A finding worth flagging honestly: unlike several sibling shape-family tests
in this suite (e.g. ``eliminate_slice_after_shape``), the differential tests
below do NOT strictly need ``skip_constant_folding=True`` to see *a* Shape
node disappear -- confirmed empirically that onnxsim's own separate
constant-folding step, entirely on its own (with ``eliminate_shape_op``
itself also skipped via ``skipped_optimizers``), already collapses a
``Shape(X)`` with fully static ``X`` into an equivalent constant, both with
and without ``start``/``end`` attributes. So without
``skip_constant_folding=True``, a passing differential test here would not
actually prove anything about THIS pass -- it would just be re-observing
constant folding under a different pass's name. ``skip_constant_folding=True``
is used throughout below anyway, precisely to rule that out and attribute the
rewrite to the real compiled ``eliminate_shape_op`` pass alone.
"""

import collections

from _formal_verify_common import isolate, prove, z3
from onnx import helper, numpy_helper, parser

import onnxsim

# X's shape for every differential test below: 5 distinct dims, so which ones
# got selected (and in what order) is unambiguous from the values alone.
_X_SHAPE = [2, 3, 4, 5, 6]


def _model(body, opset=13, ir_version=10):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def test_eliminate_shape_op_is_sound():
    # Uninterpreted per-axis "declared shape" and "runtime shape" functions,
    # to prove substitution safety: given the graph-level premise that X's
    # declared static dims equal its actual runtime shape (axis by axis),
    # any downstream consumer sees the exact same value at any axis selected
    # by Shape's [start, end) window whether it reads the ACTUAL runtime
    # Shape(X)[start:end] computation, or the FRESH CONSTANT this pass
    # builds from X's declared dims [start:end) instead. No index arithmetic
    # is modeled here (unlike eliminate_slice_after_shape's reversed-slice
    # walk) because this pass does none: the window is a plain forward
    # sub-range copy.
    declared = z3.Function("declared", z3.IntSort(), z3.RealSort())
    runtime = z3.Function("runtime", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    a, i, start, end = z3.Ints("a i start end")

    declared_matches_runtime = z3.ForAll([a], declared(a) == runtime(a))

    prove(
        z3.Implies(
            declared_matches_runtime,
            z3.Implies(
                z3.And(start <= i, i < end),
                consumer(runtime(i)) == consumer(declared(i)),
            ),
        ),
        msg="rewrite is not a sound equivalence",
    )


def test_eliminate_shape_op_negative_control_needs_hypothesis():
    # Without the declared-equals-runtime-shape premise, "declared" and
    # "runtime" are two independent, fully unconstrained uninterpreted
    # functions: the claim must NOT be valid then, or the proof above would
    # be vacuously true regardless of what the premise says. Z3 should find
    # a sat counterexample negating the claim, not unsat.
    declared = z3.Function("declared", z3.IntSort(), z3.RealSort())
    runtime = z3.Function("runtime", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    i = z3.Int("i")

    solver = z3.Solver()
    solver.add(z3.Not(consumer(runtime(i)) == consumer(declared(i))))
    assert solver.check() == z3.sat


def test_eliminate_shape_op_pass_matches_full_shape():
    # The common case: no start/end attributes, so the whole rank is
    # selected. skip_constant_folding=True is required -- see the module
    # docstring -- since onnxsim's own constant-folding step, run alone,
    # collapses a fully-static Shape(X) into the same constant by itself,
    # which would make this test pass without ever exercising the real
    # compiled eliminate_shape_op pass.
    model = _model(
        f"""
        g (float[{",".join(map(str, _X_SHAPE))}] X) => (int64[5] Z)
        {{
          Z = Shape(X)
        }}
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("eliminate_shape_op"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    # Unlike eliminate_slice_after_shape, this pass destroys the Shape node
    # outright -- nothing but the fresh constant should remain.
    assert ops == {}
    out_name = sim_model.graph.output[0].name
    z_init = next(i for i in sim_model.graph.initializer if i.name == out_name)
    assert list(numpy_helper.to_array(z_init)) == _X_SHAPE


def test_eliminate_shape_op_pass_matches_start_end_subrange():
    # start=1, end=3 selects a genuine sub-range, not the whole shape --
    # confirms the resulting constant holds exactly that sub-range's dims
    # ([3, 4], X's axes 1 and 2), not the full [2, 3, 4, 5, 6]. Shape's
    # start/end attributes were added in opset 15.
    model = _model(
        f"""
        g (float[{",".join(map(str, _X_SHAPE))}] X) => (int64[2] Z)
        {{
          Z = Shape<start=1, end=3>(X)
        }}
        """,
        opset=15,
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("eliminate_shape_op"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops == {}
    out_name = sim_model.graph.output[0].name
    z_init = next(i for i in sim_model.graph.initializer if i.name == out_name)
    assert list(numpy_helper.to_array(z_init)) == _X_SHAPE[1:3] == [3, 4]


def test_eliminate_shape_op_declines_when_x_rank_is_unresolvable():
    # X is itself Squeeze(Y, axes) where axes comes from an Add of two
    # initializers rather than being a constant (or Constant node) itself --
    # deterministic at runtime ([0, 3] either way), but FetchConstantTensor
    # can't see through the Add, so X's rank is genuinely unresolvable by
    # onnxsim's shape inference, and HasDimsOfInputOfNode(shape_node, 0) is
    # false: the predicate declines outright, before ever looking at
    # start/end. Same construction eliminate_slice_after_shape's own
    # formal-verify test uses for the same purpose.
    # skip_constant_folding=True: constant folding would otherwise fold the
    # Add/Squeeze away first and restore a statically-known rank before this
    # pass ever saw the graph.
    model = _model(
        """
        g (float[1,1,2,3,1,5,1] Y) => (int64[?] Z)
        <int64[2] axes_a = {0, 3}, int64[2] axes_b = {0, 1}>
        {
          axes = Add(axes_a, axes_b)
          X = Squeeze(Y, axes)
          Z = Shape(X)
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("eliminate_shape_op"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops == {"Add": 1, "Squeeze": 1, "Shape": 1}
    (shape_node,) = [n for n in sim_model.graph.node if n.op_type == "Shape"]
    assert shape_node.input[0] == "X"


def test_eliminate_shape_op_declines_when_selected_dim_is_symbolic():
    # X's rank is statically known (3), so the predicate's own
    # HasDimsOfInputOfNode check passes -- but X's axis 0 is a dim_param
    # ("N"), and with no start/end attributes the whole rank (including axis
    # 0) is selected. runTransform's own `dim.is_int` check fails for that
    # axis, so the predicate declines and Shape is left untouched.
    model = _model(
        """
        g (float[N,3,4] X) => (int64[3] Z)
        {
          Z = Shape(X)
        }
        """
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("eliminate_shape_op"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops == {"Shape": 1}


def test_eliminate_shape_op_fires_when_symbolic_dim_is_outside_selected_range():
    # Same symbolic-axis-0 X as above, but start=1, end=3 excludes axis 0
    # from the selected window entirely -- every dim actually selected (axes
    # 1, 2: concrete ints 3 and 4) is known and non-negative, so the
    # predicate fires despite X's rank containing an unresolved dim_param
    # elsewhere. Confirms the dim check is scoped to [start, end), not the
    # whole rank.
    model = _model(
        """
        g (float[N,3,4] X) => (int64[2] Z)
        {
          Z = Shape<start=1, end=3>(X)
        }
        """,
        opset=15,
    )
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("eliminate_shape_op"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops == {}
    out_name = sim_model.graph.output[0].name
    z_init = next(i for i in sim_model.graph.initializer if i.name == out_name)
    assert list(numpy_helper.to_array(z_init)) == [3, 4]


def test_eliminate_shape_op_declines_when_declared_dim_is_negative():
    # A genuinely negative declared dim (dim.dim >= 0 fails). Not
    # constructible through the onnx.parser text format or onnx.helper's
    # usual shape-inference path (a negative *static* dim_value isn't
    # something a real exporter or ONNX's own shape inference would ever
    # produce), so this is built by hand: a value_info whose TensorShapeProto
    # carries an explicit negative dim_value, which onnx.checker.check_model
    # accepts (nothing in the ONNX spec's checker actually forbids it, even
    # though it's semantically nonsensical as a real shape). check_n=0
    # because onnxsim's own random-input equivalence check can't construct a
    # tensor with a negative shape dimension to run the model against.
    X_vi = helper.make_tensor_value_info("X", 1, [2, 3, 4])  # 1 == FLOAT
    X_vi.type.tensor_type.shape.dim[1].dim_value = -5
    shape_node = helper.make_node("Shape", ["X"], ["Z"])
    Z_vi = helper.make_tensor_value_info("Z", 7, [3])  # 7 == INT64
    graph = helper.make_graph([shape_node], "g", [X_vi], [Z_vi])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10

    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=0,
        skipped_optimizers=isolate("eliminate_shape_op"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    ops = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert ops == {"Shape": 1}
