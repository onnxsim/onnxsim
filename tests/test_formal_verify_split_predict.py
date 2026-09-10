"""Formal check for SplitPredict (opt-in; ``split.h``'s shared
``split_init_and_predict(graph, init, predict)`` called as ``(false, true)``).

**This pass is one half of a matched pair with ``SplitInit``**
(``tests/test_formal_verify_split_init.py`` -- read that file's module
docstring first; it lays out the shared algorithm in full, and this file
follows the same conventions rather than repeating them). The two wrap the
exact same ``split_init_and_predict`` function, differing only in which flag
is set. Recap of the classification (full detail in the ``split_init`` file):
every Value reachable from a genuine runtime input (no corresponding
initializer) or from an impure operator's output
(``RandomNormal(Like)``/``RandomUniform(Like)``/``Loop``/``If``/``Scan``)
belongs to the "predict net"; a node belongs to it if it is itself impure or
any input does; everything else is the "init net". The boundary -- a
predict-net node reading a non-predict-net value, or a graph output that
isn't itself predict-net -- becomes new init-net outputs / predict-net
inputs.

``SplitPredict`` (``predict=true``) destructively mutates the graph into
*only* the predict net: keeps every predict-net node; adds the boundary
values as new graph **inputs**, cutting the old producer via
``replaceAllUsesWith`` (an ``optionalInputDummyNode``/``kUndefined``
placeholder stands in for anything that still has uses but isn't a "real"
value, e.g. an originally-optional input); deletes every node that both (a)
isn't predict-net and (b) has no remaining uses, in reverse topological
order; erases every now-unused graph input; and **clears every initializer**
-- the predict net is meant to run driven entirely by its own new inputs
(the caller's genuine runtime values, plus the init net's boundary outputs
fed back in), never by re-deriving anything from constants baked into its
own copy of the graph.

# The real theorem (full statement -- see ``test_formal_verify_split_init.py`` too)

As that file's docstring explains at length: this is a genuine
**composition** theorem, not a single-rewrite value-preservation claim.
Neither ``split_init``'s own output graph nor ``split_predict``'s own output
graph, run alone, computes anything resembling the original graph's own
outputs -- each computes only half. The actual claim:

    Run ``split_init``'s own output graph once, with no runtime input (by
    construction it needs none), to obtain the boundary values. Feed those
    boundary values into ``split_predict``'s own output graph as its new
    inputs, alongside the original model's genuine runtime inputs. The
    result equals exactly what the ORIGINAL, unsplit graph would have
    computed for its own outputs, given those same runtime inputs.

This file owns the full Z3 proof of that composed claim (the ``split_init``
file proves only its own half -- that the init net's boundary computation is
provably independent of the runtime input the predict net will see, which is
what licenses computing it once and reusing it). It also carries this pass's
own half of the informal claim: that the predict net's own computation,
*given* the boundary values as extra inputs alongside the genuine runtime
inputs, reproduces the original graph's own outputs -- plus this pass's own
specific structural behavior (new inputs added at the boundary,
non-predict-net nodes removed, every initializer cleared).

# Z3 model and the composition proof

Same 4-node graph as the ``split_init`` file, modeled with uninterpreted
functions:

    c1     = f_pure(const1)               -- init net
    c2     = g_pure(c1, const2)           -- init net; c2 is the boundary
    p1     = h_impure(c2, runtime_input)  -- predict net
    output = k(p1)                        -- predict net

Define, matching each pass's own surviving computation exactly:

  - ``init_net_boundary(const1, const2) = g_pure(f_pure(const1), const2)``
    -- what ``split_init``'s own surviving nodes compute, with no
    ``runtime_input`` in scope at all.
  - ``predict_net_output(boundary_in, runtime_input) = k(h_impure(boundary_in, runtime_input))``
    -- what ``split_predict``'s own surviving nodes compute, treating the
    boundary as a plain new input, exactly as ``graph.addInput()`` +
    ``replaceAllUsesWith`` wires it in.
  - ``original_output(const1, const2, runtime_input) = k(h_impure(g_pure(f_pure(const1), const2), runtime_input))``
    -- the whole, unsplit graph, evaluated in one pass.

``test_split_predict_composition_matches_original_graph`` proves
``predict_net_output(init_net_boundary(c1, c2), r) == original_output(c1, c2, r)``
for all ``c1, c2, r``. Z3 discharges this close to immediately, by congruence
-- both sides unfold to the identical term. That is deliberate, and mirrors
this suite's own honest precedent (``eliminate_unused_initializer``'s
docstring makes the same point about its own invariance proof): the
interesting content was never "is this a deep algebraic identity" -- it is
"is the WIRING correct", i.e. does the predict net's own new input get fed
*exactly* the value the init net actually produces, with nothing dropped and
nothing re-derived from scratch. The negative control right after shows this
is not a foregone conclusion: a predict net that (incorrectly) tries to
recompute ``p1`` from ``runtime_input`` alone, via some other function that
never sees the boundary value at all, does **not** in general match the
original graph -- Z3 finds a concrete counterexample, confirming that
correctly threading the boundary value through is load-bearing, not
incidental busywork.

# Differential tests

Same rationale as the ``split_init`` file for calling ``onnxsim.simplify()``
directly with ``check_n=0`` (``simplify_isolated_extra``'s own equivalence
check assumes unchanged inputs/outputs, which ``SplitPredict`` also
violates -- confirmed empirically, same ``InvalidArgument`` shape) and for
``skip_constant_folding=True`` (constant folding would otherwise fold every
pure node away before this pass ever sees it, leaving nothing to isolate).
The centerpiece,
``test_split_predict_end_to_end_composition_with_real_init_net``, is the
single most important differential check across *both* files: it runs the
real, compiled ``split_init`` **and** ``split_predict`` passes on fresh
copies of the same model, chains their onnxruntime outputs exactly per the
composition theorem above, and confirms the result matches the original,
unsplit model's own onnxruntime output for the same runtime input --
concrete-number confirmation of the Z3 claim proved above.
"""

import numpy as np
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import isolate, prove, z3
from onnx import numpy_helper, parser

import onnxsim


def _tensor(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def _model():
    # Same model as test_formal_verify_split_init.py's own main differential
    # test (a fresh parse each call, since split_init_and_predict
    # destructively mutates its graph argument, and SplitInit/SplitPredict
    # must each get their own untouched copy of the original).
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
    return model


def _run_split(model, pass_name):
    """Run the real, compiled ``split_init``/``split_predict`` pass alone.

    Direct ``onnxsim.simplify()`` call, not ``simplify_isolated_extra`` --
    see the module docstring's "Differential tests" section for why
    (``check_n=0``, ``skip_constant_folding=True``).
    """
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=0,
        extra_optimizers=[pass_name],
        skipped_optimizers=isolate(),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    return sim_model


def test_split_predict_composition_matches_original_graph():
    # The full decomposition/composition theorem (see module docstring),
    # proved once here for both files. const1/const2 stand for whatever the
    # model's initializers hold; runtime_input for X.
    const1, const2, runtime_input = z3.Reals("const1 const2 runtime_input")
    f_pure = z3.Function("f_pure", z3.RealSort(), z3.RealSort())
    g_pure = z3.Function("g_pure", z3.RealSort(), z3.RealSort(), z3.RealSort())
    h_impure = z3.Function("h_impure", z3.RealSort(), z3.RealSort(), z3.RealSort())
    k = z3.Function("k", z3.RealSort(), z3.RealSort())

    def init_net_boundary(c1v, c2v):
        return g_pure(f_pure(c1v), c2v)

    def predict_net_output(boundary_in, input_v):
        return k(h_impure(boundary_in, input_v))

    def original_output(c1v, c2v, input_v):
        return k(h_impure(g_pure(f_pure(c1v), c2v), input_v))

    prove(
        predict_net_output(init_net_boundary(const1, const2), runtime_input)
        == original_output(const1, const2, runtime_input)
    )


def test_split_predict_negative_control_boundary_omission_breaks_composition():
    # If the predict net instead tried to recompute p1 from runtime_input
    # ALONE -- omitting the boundary value entirely, e.g. because the split
    # dropped or mis-threaded it -- via some other function h_wrong that
    # never even takes the boundary as an argument, the result must NOT in
    # general match the original graph. h_wrong is a fresh, independent
    # uninterpreted function: nothing ties it to h_impure/g_pure/f_pure, so
    # Z3 must find a concrete assignment where they disagree, or the
    # composition theorem above would have been proved by an argument loose
    # enough to "prove" this wrong variant too.
    const1, const2, runtime_input = z3.Reals("const1 const2 runtime_input")
    f_pure = z3.Function("f_pure", z3.RealSort(), z3.RealSort())
    g_pure = z3.Function("g_pure", z3.RealSort(), z3.RealSort(), z3.RealSort())
    h_impure = z3.Function("h_impure", z3.RealSort(), z3.RealSort(), z3.RealSort())
    h_wrong = z3.Function("h_wrong", z3.RealSort(), z3.RealSort())
    k = z3.Function("k", z3.RealSort(), z3.RealSort())

    def original_output(c1v, c2v, input_v):
        return k(h_impure(g_pure(f_pure(c1v), c2v), input_v))

    def predict_net_output_wrong(input_v):
        return k(h_wrong(input_v))  # never sees const1/const2's boundary at all

    solver = z3.Solver()
    solver.add(
        predict_net_output_wrong(runtime_input)
        != original_output(const1, const2, runtime_input)
    )
    assert solver.check() == z3.sat


def test_split_predict_is_an_opt_in_pass():
    assert "split_predict" in C._list_other_optimizers()
    assert "split_predict" not in C._list_optimizers()


def test_split_predict_isolates_impure_suffix_and_adds_boundary_input():
    sim_model = _run_split(_model(), "split_predict")

    # No initializers at all: the predict net must be driven entirely by its
    # own new inputs, per split_init_and_predict's own clearInitializers().
    assert list(sim_model.graph.initializer) == []
    # c2 (the boundary) joins X as a new graph input; const_a/const_b's own
    # pure computation (Identity/first Add) is gone.
    assert sorted(i.name for i in sim_model.graph.input) == ["X", "c2"]
    assert [n.op_type for n in sim_model.graph.node] == ["Add", "Identity"]
    assert [o.name for o in sim_model.graph.output] == ["Y"]

    x = np.array([10.0, 20.0], dtype=np.float32)
    c2 = np.array([4.0, 6.0], dtype=np.float32)  # hand-computed: const_a + const_b
    sess = ort.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_val,) = sess.run(None, {"X": x, "c2": c2})
    # p1 = c2 + X, Y = p1 -- matches the original, unsplit model's own Y for
    # the same X (see the end-to-end test below for the compiled-pass version
    # of this same claim, chained through a real split_init run instead of a
    # hand-computed c2).
    np.testing.assert_array_equal(y_val, np.array([14.0, 26.0], dtype=np.float32))


def test_split_predict_end_to_end_composition_with_real_init_net():
    # THE central differential check spanning both files: run the real,
    # compiled split_init and split_predict passes on independent fresh
    # copies of the same original model, chain their onnxruntime outputs
    # exactly per the composition theorem proved above (init net's own c2
    # output feeds predict net's own c2 input), and confirm the result
    # matches the ORIGINAL, unsplit model's own output for the same X.
    original = _model()
    init_model = _run_split(_model(), "split_init")
    predict_model = _run_split(_model(), "split_predict")

    init_sess = ort.InferenceSession(
        init_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (c2_val,) = init_sess.run(None, {})  # init net needs no runtime input at all

    x = np.array([10.0, 20.0], dtype=np.float32)
    predict_sess = ort.InferenceSession(
        predict_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_composed,) = predict_sess.run(None, {"X": x, "c2": c2_val})

    orig_sess = ort.InferenceSession(
        original.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_original,) = orig_sess.run(None, {"X": x})

    np.testing.assert_array_equal(y_composed, y_original)


def test_split_predict_declines_when_graph_is_entirely_pure():
    # Complementary edge case to test_formal_verify_split_init.py's own
    # test_split_init_declines_removing_anything_when_graph_is_entirely_pure:
    # here it's the PREDICT net that degenerates. No genuine runtime input
    # and no impure op at all means every value is init-net; the entire
    # original output Y itself becomes the one boundary value (a graph
    # output that isn't predict-net), so the predict net shrinks to a bare
    # passthrough: one new input (also named Y, replacing the old producer),
    # no nodes, feeding straight to the one original output.
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
    sim_model = _run_split(model, "split_predict")

    assert list(sim_model.graph.initializer) == []
    assert [i.name for i in sim_model.graph.input] == ["Y"]
    assert list(sim_model.graph.node) == []
    assert [o.name for o in sim_model.graph.output] == ["Y"]

    sess = ort.InferenceSession(
        sim_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    fed = np.array([4.0, 6.0], dtype=np.float32)  # the init net's own Y value
    (y_val,) = sess.run(None, {"Y": fed})
    np.testing.assert_array_equal(y_val, fed)
