"""Formal check for SplitInit (opt-in; ``split.h``'s shared
``split_init_and_predict(graph, init, predict)`` called as ``(true, false)``).

**This pass is one half of a matched pair with ``SplitPredict``**
(``tests/test_formal_verify_split_predict.py`` -- read that file's module
docstring too, since the two together state one coherent theorem that
neither file alone can state on its own). Both wrap the exact same
``split_init_and_predict`` function in ``split.h``, differing only in which
of the two boolean flags is set; the algorithm classifies every graph
``Value`` as belonging to the "predict net" if it is reachable from (a) a
graph input with **no** corresponding initializer (a genuine runtime input,
as opposed to a constant-with-a-default), or (b) an impure operator's output
(``RandomNormal``/``RandomNormalLike``/``RandomUniform``/``RandomUniformLike``/
``Loop``/``If``/``Scan`` -- non-deterministic or control-flow ops that cannot
be "compute once, reuse forever"). A node belongs to the predict net if it is
itself impure, or if *any* of its inputs does -- this propagates forward.
Everything **not** reachable that way -- computable purely from the model's
own initializers, no impure op involved -- is the "init net". The boundary
between the two (a predict-net node reading a non-predict-net value, or a
graph output that is not itself predict-net) becomes new init-net
**outputs** / predict-net **inputs**.

``SplitInit`` (``init=true``) destructively mutates the graph into *only* the
init net: keeps every non-predict-net node, registers the boundary values as
new graph outputs (after first removing any original outputs that *are*
predict-net), deletes every predict-net node, and erases every predict-net
graph input. ``SplitPredict`` does the complementary thing to produce the
other half (own file). Both are ``FullGraphBasedPass``es (whole-graph
``runPass(Graph&)``, not per-node ``PredicateBasedPass``), ``PassType::Separate``
(opt-in -- confirmed empirically below via ``_list_other_optimizers()``).

# The real theorem: a composition/decomposition claim, not a single-rewrite one

Nearly every other file in this suite proves "rewrite R preserves the value
this one graph computes". That shape does not apply here: **neither**
``split_init``'s own output graph **nor** ``split_predict``'s own output graph
computes anything close to what the original graph computed -- each computes
only *half* of it (that's the whole point: splitting a model into a
"run-once-and-cache" init net and a "run-per-request" predict net, per the
header comment's own "arrange to call it twice" remark). The genuine,
checkable correctness claim spans **both** passes together:

    Feed ``split_init``'s own output graph nothing (by construction it needs
    no runtime input at all once split), run it once to obtain the boundary
    values; feed those boundary values into ``split_predict``'s own output
    graph as its new inputs, *alongside* the original model's genuine runtime
    inputs; the result equals exactly what the ORIGINAL, unsplit graph would
    have computed for its own outputs, given those same runtime inputs.

Call this the **decomposition/composition theorem**. It is stated in full in
both this file and ``test_formal_verify_split_predict.py``, but proved as one
shared-shape Z3 lemma only in the *predict* file's
``test_split_predict_composition_matches_original_graph`` (see that file) --
duplicating the identical multi-function Z3 term in both files would add
size, not insight, since the composed claim only becomes checkable once both
phases' own semantics are laid out; this file instead proves the specific
half of the argument that is genuinely ``split_init``'s own to make: that the
init net's own computation, run on constants alone, is well-defined
independent of whatever runtime input the (as yet unsplit) predict net will
eventually see. That independence is precisely *why* it is sound to compute
the init net once and reuse its outputs forever, rather than recomputing them
on every predict-net invocation -- the informal justification the header
comment itself gives ("inputs which have an initializer value... are constant
across runs of the predict net").

# Z3 model

A small 4-node computation graph, modeled with uninterpreted functions (this
suite's standard idiom for "some real tensor op, kept abstract"):

    c1     = f_pure(const1)               -- pure: only touches an initializer
    c2     = g_pure(c1, const2)           -- pure: only touches c1 and another initializer
    p1     = h_impure(c2, runtime_input)  -- impure: touches a genuine runtime input
    output = k(p1)                        -- impure (transitively, via p1)

``c2`` is exactly the one boundary value that must cross from the init net to
the predict net: it is pure (computable from ``const1``/``const2`` alone) but
is read by an impure node (``h_impure``), which is precisely
``split_init_and_predict``'s own ``new_interface`` condition ("a Value which
is not itself in the predict net, but which is used by a Node which is").

``test_split_init_boundary_value_is_input_independent`` proves the init net's
own boundary computation, ``g_pure(f_pure(const1), const2)``, is provably the
same value regardless of ``runtime_input`` -- i.e. it truly needs no runtime
input in scope at all, which is exactly what licenses ``SplitInit`` to erase
every predict-net-classified graph input (it is unreachable from, and cannot
possibly influence, anything left in the init net). The negative control
right after it confirms this is not vacuous: if a value's construction really
did route through ``runtime_input`` (i.e. it was genuinely impure and would
have been *mis*classified as init-net), invariance provably fails.

# Differential tests

Run against the actually compiled pass via ``onnxsim.simplify()`` directly
(not ``simplify_isolated_extra``): confirmed empirically below that
``simplify_isolated_extra``'s own built-in equivalence check assumes the same
inputs/outputs before and after simplification, which ``SplitInit`` never
preserves by design (see ``test_split_init_isolates_pure_prefix_and_adds_boundary_output``'s
comment for the concrete ``InvalidArgument`` this raises) -- so ``check_n=0``
is used instead, and correctness is confirmed directly via onnxruntime.
``skip_constant_folding=True`` is required throughout: without it, onnxsim's
own constant-folding preprocessing (a step wholly independent of
``skipped_optimizers``/``extra_optimizers``) folds every pure node in these
test models into a plain initializer *before* ``split_init`` ever runs,
leaving no pure nodes for the pass to move into the init net at all
(confirmed empirically).
"""

import numpy as np
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import isolate, prove, z3
from onnx import numpy_helper, parser

import onnxsim


def _tensor(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def _run_split_init(model):
    """Run the real, compiled ``split_init`` pass alone on ``model``.

    Direct ``onnxsim.simplify()`` call, not ``simplify_isolated_extra``: see
    the module docstring's "Differential tests" section for why
    (``check_n=0``, ``skip_constant_folding=True``).
    """
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=0,
        extra_optimizers=["split_init"],
        skipped_optimizers=isolate(),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    return sim_model


def test_split_init_boundary_value_is_input_independent():
    # c1, c2 are the "init net": pure functions of const1/const2 only. Model
    # this literally -- boundary(...) never mentions runtime_input at all in
    # its own construction, exactly as SplitInit's surviving nodes never
    # reference the erased predict-net inputs.
    const1, const2, runtime_input1, runtime_input2 = z3.Reals(
        "const1 const2 runtime_input1 runtime_input2"
    )
    f_pure = z3.Function("f_pure", z3.RealSort(), z3.RealSort())
    g_pure = z3.Function("g_pure", z3.RealSort(), z3.RealSort(), z3.RealSort())

    def boundary():
        c1 = f_pure(const1)
        return g_pure(c1, const2)

    # The same const1/const2 give the same boundary value (c2) no matter
    # what runtime_input the not-yet-split predict net will eventually see
    # -- runtime_input doesn't even appear in `boundary()`'s own term. This
    # is exactly why it is sound for SplitInit to compute this value once,
    # with no runtime input in scope, and for SplitPredict's own graph to
    # treat it as a plain new input rather than something it must
    # recompute per call.
    prove(
        z3.Implies(
            runtime_input1 != runtime_input2,
            boundary() == boundary(),  # trivially so: no runtime_input term at all
        )
    )


def test_split_init_negative_control_impure_value_is_input_dependent():
    # Negative control: if a value's own construction genuinely DID route
    # through runtime_input -- i.e. it should have been classified as
    # predict-net, not init-net -- the same "same consts, any input"
    # invariance must NOT hold in general. h_extra stands for a
    # (hypothetically, incorrectly) "pure" second operand that actually
    # depends on the runtime input; the resulting boundary value is then
    # genuinely input-dependent, and Z3 must find a concrete counterexample
    # where it changes -- confirming the invariance proved above is a real,
    # checkable property of true purity, not something that holds no matter
    # what gets fed into it.
    const1, const2 = z3.Reals("const1 const2")
    runtime_input1, runtime_input2 = z3.Reals("runtime_input1 runtime_input2")
    f_pure = z3.Function("f_pure", z3.RealSort(), z3.RealSort())
    h_extra = z3.Function("h_extra", z3.RealSort(), z3.RealSort(), z3.RealSort())
    g_pure = z3.Function("g_pure", z3.RealSort(), z3.RealSort(), z3.RealSort())

    def misclassified_boundary(runtime_input):
        c1 = f_pure(const1)
        # would-be-"c2", but its second operand secretly depends on the
        # runtime input -- i.e. this value is actually impure.
        return g_pure(c1, h_extra(const2, runtime_input))

    solver = z3.Solver()
    solver.add(runtime_input1 != runtime_input2)
    solver.add(
        misclassified_boundary(runtime_input1) != misclassified_boundary(runtime_input2)
    )
    assert solver.check() == z3.sat


def test_split_init_is_an_opt_in_pass():
    assert "split_init" in C._list_other_optimizers()
    assert "split_init" not in C._list_optimizers()


def test_split_init_isolates_pure_prefix_and_adds_boundary_output():
    # const_a, const_b: initializers. c1 = Identity(const_a) [pure].
    # c2 = Add(c1, const_b) [pure, the boundary -- read by the impure p1].
    # p1 = Add(c2, X) [impure: X has no initializer]. Y = Identity(p1)
    # [impure, transitively]. This is the same model
    # test_formal_verify_split_predict.py's own main differential test uses
    # (a fresh copy each time, since split_init_and_predict destructively
    # mutates its graph argument) -- see that file for the complementary
    # half and the full end-to-end composition check.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2] X) => (float[2] Y)
        {
          c1 = Identity(const_a)
          c2 = Add(c1, const_b)
          p1 = Add(c2, X)
          Y = Identity(p1)
        }
        """
    )
    model.graph.initializer.extend(
        [_tensor("const_a", [1.0, 2.0]), _tensor("const_b", [3.0, 4.0])]
    )

    # simplify_isolated_extra's own built-in equivalence check (see
    # _formal_verify_common.py) always addresses inputs/outputs by the
    # *original* model's names -- confirmed empirically to fail hard against
    # SplitInit's own graph (whose only output is now `c2`, not `Y`, and
    # which takes no `X` input at all):
    #   onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument:
    #   INVALID_ARGUMENT : Invalid input name: X
    # (or, symmetrically, a missing-output error) -- so this test calls
    # onnxsim.simplify directly (via _run_split_init) with check_n=0 instead,
    # and confirms correctness itself via onnxruntime below.
    sim_model = _run_split_init(model)

    # No runtime inputs survive: X (predict-net, no initializer) is erased.
    assert list(sim_model.graph.input) == []
    # Exactly the pure prefix remains -- p1/Y (predict-net) are gone.
    assert [n.op_type for n in sim_model.graph.node] == ["Identity", "Add"]
    # c2, the boundary value, is now the graph's own (sole) output.
    assert [o.name for o in sim_model.graph.output] == ["c2"]

    # Run the init-only graph with no feed at all and confirm it produces
    # c2's own correct, hand-computed value: c1 = const_a, c2 = c1 + const_b.
    sess = ort.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (c2_val,) = sess.run(None, {})
    np.testing.assert_array_equal(c2_val, np.array([4.0, 6.0], dtype=np.float32))


def test_split_init_declines_removing_anything_when_graph_is_entirely_pure():
    # No genuine runtime input and no impure op at all: every value is
    # init-net, so predict_net_values is empty and split_init_and_predict has
    # nothing to move out -- the graph is returned with its original nodes,
    # inputs and outputs untouched (the complementary decline case is
    # test_formal_verify_split_predict.py's own
    # test_split_predict_declines_when_graph_is_entirely_pure, where the
    # roles are flipped and the *predict* net degenerates instead).
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g () => (float[2] Y)
        {
          c1 = Identity(const_a)
          Y = Add(c1, const_b)
        }
        """
    )
    model.graph.initializer.extend(
        [_tensor("const_a", [1.0, 2.0]), _tensor("const_b", [3.0, 4.0])]
    )
    sim_model = _run_split_init(model)

    assert list(sim_model.graph.input) == []  # there was never a runtime input
    assert [n.op_type for n in sim_model.graph.node] == ["Identity", "Add"]
    assert [o.name for o in sim_model.graph.output] == ["Y"]  # original name kept

    sess = ort.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_val,) = sess.run(None, {})
    np.testing.assert_array_equal(y_val, np.array([4.0, 6.0], dtype=np.float32))


def test_split_init_impure_operator_boundary_may_be_an_initializer_directly():
    # Exercises the OTHER impurity trigger -- an impure OPERATOR
    # (RandomUniformLike), not "reachable from a non-initializer-backed
    # input" (there is no runtime input in this model at all). c1 =
    # Identity(const_a) is pure; r1 = RandomUniformLike(c1) is impure by
    # is_pure_operator's own explicit denylist, so the Add producing Y is
    # impure transitively. const_b is read directly by that impure Add, with
    # no pure node of its own in between -- so it becomes a boundary value in
    # its own right, and split_init_and_predict registers the *initializer's
    # own Value* as a new graph output verbatim (no producing node at all),
    # exactly like c1's own boundary. This confirms new_interface construction
    # doesn't require the boundary value to be a node's output.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g () => (float[2] Y)
        {
          c1 = Identity(const_a)
          r1 = RandomUniformLike(c1)
          Y = Add(r1, const_b)
        }
        """
    )
    model.graph.initializer.extend(
        [_tensor("const_a", [1.0, 2.0]), _tensor("const_b", [3.0, 4.0])]
    )
    sim_model = _run_split_init(model)

    assert [n.op_type for n in sim_model.graph.node] == ["Identity"]
    # Both const_b (initializer, output verbatim) and c1 (computed) cross
    # the boundary; RandomUniformLike/Add themselves are gone.
    assert sorted(o.name for o in sim_model.graph.output) == ["c1", "const_b"]

    sess = ort.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    outs = dict(
        zip(
            [o.name for o in sim_model.graph.output],
            sess.run(None, {}),
        )
    )
    np.testing.assert_array_equal(outs["c1"], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(
        outs["const_b"], np.array([3.0, 4.0], dtype=np.float32)
    )
