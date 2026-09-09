"""Formal check for EliminateIfWithConstCond
(eliminate_if_with_const_cond.h): given an ``If`` node whose ``cond`` input is
a compile-time constant (a ``Constant`` node's output, or a constant
initializer -- ``patternMatchPredicate``), inline the taken branch's subgraph
(``then_branch`` if ``cond`` is true, else ``else_branch`` -- the ``If`` op's
own two ``GraphProto`` attributes) directly into the parent graph and destroy
the ``If`` node (``runTransform``).

The DEEP correctness content of this pass is *not* algebraic. ONNX's own
``If`` operator is DEFINED by its spec as "evaluate ``then_branch`` and use
its outputs if ``cond`` is true, else evaluate ``else_branch`` and use its
outputs" -- so modeling ``if_result = If(cond, then_val, else_val)`` as Z3's
native ``z3.If(cond, then_val, else_val)`` (exactly mirroring
test_formal_verify_rewrite_where.py's use of ``z3.If`` for a similarly
definitional boolean-selection identity) makes "when ``cond`` is the concrete
boolean ``True``, ``if_result == then_val``" nothing more than Z3's own
``If``-simplification at a known condition value -- there is no derivation to
do. That *is* however the whole point of this pass: because ``cond`` is known
at COMPILE TIME (not merely constant-but-opaque-to-the-compiler at runtime),
"run ``If(True, T, E)``" and "run ``T`` directly" are the same computation,
so replacing the former with the latter -- i.e. beta-reduction / substitution
transparency for a pure, side-effect-free computation graph -- is sound.

What actually has teeth here is getting the SUBGRAPH-INLINING MECHANICS
right: copying each taken-branch node's kind/attributes into the parent
graph, remapping each of its inputs to either an already-copied sibling
node's output, a value captured from the outer (parent) scope, or a copied
subgraph-local initializer (the ``value_dict``/``kCaptured``/``kParam``
handling in ``runTransform``), and rewiring the original ``If`` node's own
outputs to the corresponding inlined values. None of that is expressible as
a clean algebraic claim -- it is exactly what the differential tests below
exist to check, by running the real compiled pass on concrete graphs and
inspecting the resulting node/edge structure.

``onnx.parser.parse_model()`` turns out to handle ``If`` nodes with
``then_branch``/``else_branch`` ``GraphProto`` attributes cleanly -- the
text-format grammar treats a bare (non-numeric, non-typed) attribute value as
a nested graph literal (``name (ins) => (outs) { nodes }``), so e.g.
``Z = If <then_branch = g1 () => (float[4] Y) { Y = Relu(X) }, ...> (cond)``
parses and passes ``onnx.checker`` directly, with no ``onnx.helper`` fallback
needed anywhere in this file.
"""

from _formal_verify_common import producer, prove, simplify_isolated, z3
from onnx import parser


def test_eliminate_if_with_const_cond_is_sound():
    # `captured` stands for one arbitrary element of a value captured from
    # the outer (parent) scope; `then_val`/`else_val` stand for whatever a
    # branch's own internal computation produces from it (an arbitrary
    # uninterpreted function -- the branch's actual node graph is opaque to
    # this proof, only the definitional If-selects-a-branch fact matters).
    cond = z3.Bool("cond")
    captured = z3.Real("captured")
    then_fn = z3.Function("then_fn", z3.RealSort(), z3.RealSort())
    else_fn = z3.Function("else_fn", z3.RealSort(), z3.RealSort())
    then_val = then_fn(captured)
    else_val = else_fn(captured)
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    # If's own definition: select then_val when cond holds, else_val
    # otherwise. This is z3.If verbatim -- there is no algebra to derive.
    if_result = z3.If(cond, then_val, else_val)

    # The pass's soundness claim, split by which constant cond takes:
    # inlining then_branch (dropping the If and else_branch entirely) is
    # correct exactly when cond is statically True, and symmetrically for
    # False/else_branch.
    prove(z3.Implies(cond, if_result == then_val))
    prove(z3.Implies(z3.Not(cond), if_result == else_val))

    # Full substitution-safety claim: composed with an arbitrary downstream
    # consumer of the If's output, inlining the taken branch produces the
    # exact same value for *any* consumer -- not just the one-hop checks in
    # test_eliminate_if_with_const_cond_pass_matches_true/false below.
    prove(z3.Implies(cond, consumer(if_result) == consumer(then_val)))
    prove(z3.Implies(z3.Not(cond), consumer(if_result) == consumer(else_val)))


def test_eliminate_if_with_const_cond_negative_control_wrong_branch_is_unsound():
    # Sanity check that the proof is genuine, not vacuous: selecting the
    # WRONG branch when cond is (statically) true -- If(cond, T, E) == E,
    # rather than == T -- must NOT hold for every cond/T/E. Confirm Z3 finds
    # a real counterexample rather than reporting the (wrong) claim valid.
    cond = z3.Bool("cond")
    then_val = z3.Real("then_val")
    else_val = z3.Real("else_val")
    if_result = z3.If(cond, then_val, else_val)
    wrong_claim = z3.Implies(cond, if_result == else_val)
    solver = z3.Solver()
    solver.add(z3.Not(wrong_claim))
    assert solver.check() == z3.sat


def test_eliminate_if_with_const_cond_pass_matches_true():
    # Differential check: with cond statically True, the real compiled pass,
    # run alone, inlines then_branch's Relu node directly into the parent
    # graph, rewires Z (the If's own output) to read from it, and destroys
    # both the If node and the untaken else_branch's Sigmoid entirely --
    # confirmed by inspecting the surviving graph's actual node structure,
    # not merely counting op types.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4] X) => (float[4] Z)
        {
            cond = Constant <value = bool[1] {1}> ()
            Z = If <
                then_branch = then_g () => (float[4] Y) { Y = Relu(X) },
                else_branch = else_g () => (float[4] Y) { Y = Sigmoid(X) }
            > (cond)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_if_with_const_cond")
    assert ops["If"] == 0
    assert ops["Sigmoid"] == 0
    z_node = producer(sim_model, "Z")
    assert z_node.op_type == "Relu"
    assert list(z_node.input) == ["X"]


def test_eliminate_if_with_const_cond_pass_matches_false():
    # Symmetric case: cond statically False inlines else_branch's Sigmoid
    # instead, and then_branch's Relu does not appear anywhere in the result.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4] X) => (float[4] Z)
        {
            cond = Constant <value = bool[1] {0}> ()
            Z = If <
                then_branch = then_g () => (float[4] Y) { Y = Relu(X) },
                else_branch = else_g () => (float[4] Y) { Y = Sigmoid(X) }
            > (cond)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_if_with_const_cond")
    assert ops["If"] == 0
    assert ops["Relu"] == 0
    z_node = producer(sim_model, "Z")
    assert z_node.op_type == "Sigmoid"
    assert list(z_node.input) == ["X"]


def test_eliminate_if_with_const_cond_captured_value_used_twice():
    # The taken branch (then_branch here) reads the outer-scope value X both
    # directly (as one of Add's two inputs) and indirectly (through its own
    # internal Relu node feeding Add's other input) -- exercising the
    # unique_name_to_value_in_parent/kCaptured remapping machinery more
    # thoroughly than a single-input branch does: X must be correctly
    # resolved to the SAME parent-graph Value both times, not duplicated or
    # left dangling.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4] X) => (float[4] Z)
        {
            cond = Constant <value = bool[1] {1}> ()
            Z = If <
                then_branch = then_g () => (float[4] Y) {
                    internal = Relu(X)
                    Y = Add(X, internal)
                },
                else_branch = else_g () => (float[4] Y) { Y = Sigmoid(X) }
            > (cond)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_if_with_const_cond")
    assert ops["If"] == 0
    assert ops["Sigmoid"] == 0
    z_node = producer(sim_model, "Z")
    assert z_node.op_type == "Add"
    relu_node = producer(sim_model, z_node.input[1])
    assert relu_node.op_type == "Relu"
    # Both of Add's paths back to a graph input resolve to the very same
    # captured X -- not two independently-captured copies of it.
    assert z_node.input[0] == "X"
    assert list(relu_node.input) == ["X"]


def test_eliminate_if_with_const_cond_constant_initializer_cond():
    # patternMatchPredicate also matches a cond that is a constant
    # INITIALIZER rather than a Constant node's output
    # (is_constant_initializer) -- exercise that half of the predicate too.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4] X) => (float[4] Z)
        <bool[1] cond = {1}>
        {
            Z = If <
                then_branch = then_g () => (float[4] Y) { Y = Relu(X) },
                else_branch = else_g () => (float[4] Y) { Y = Sigmoid(X) }
            > (cond)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_if_with_const_cond")
    assert ops["If"] == 0
    assert ops["Sigmoid"] == 0
    z_node = producer(sim_model, "Z")
    assert z_node.op_type == "Relu"
    assert list(z_node.input) == ["X"]


def test_eliminate_if_with_const_cond_declines_when_cond_is_not_constant():
    # cond is a genuine graph input (a runtime-computed boolean, from the
    # pass's point of view) rather than a Constant node's output or a
    # constant initializer -- patternMatchPredicate declines outright, so
    # the real compiled pass, run alone, must leave the If node (and both of
    # its branches, unexamined) completely untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4] X, bool[1] Cond) => (float[4] Z)
        {
            Z = If <
                then_branch = then_g () => (float[4] Y) { Y = Relu(X) },
                else_branch = else_g () => (float[4] Y) { Y = Sigmoid(X) }
            > (Cond)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_if_with_const_cond")
    assert ops["If"] == 1
    z_node = producer(sim_model, "Z")
    assert z_node.op_type == "If"
    assert list(z_node.input) == ["Cond"]


def test_eliminate_if_with_const_cond_forwarded_initializer_output():
    # A related special case to the one above: else_branch has no nodes at
    # all -- its sole output C is directly one of the subgraph's own
    # constant initializers (as if constant folding had already reduced
    # that branch down to a bare constant before this pass runs). This hits
    # the `output_in_subgraph->node()->kind() == kParam` branch at the end
    # of runTransform: C is absent from value_dict (nothing produces it),
    # so the initializer itself must be copied into the parent graph and
    # used as If's output directly.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4] X) => (float[4] Z)
        {
            cond = Constant <value = bool[1] {0}> ()
            Z = If <
                then_branch = then_g () => (float[4] Y) { Y = Relu(X) },
                else_branch = else_g () => (float[4] C)
                    <float[4] C = {1.0, 2.0, 3.0, 4.0}> { }
            > (cond)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_if_with_const_cond")
    assert ops["If"] == 0
    assert ops["Relu"] == 0
    assert "Z" in [i.name for i in sim_model.graph.initializer]


# Not covered: a branch that forwards a CAPTURED outer-scope value straight
# through as ITS OWN OUTPUT with no intervening node at all (the
# `output_in_subgraph->node()->kind() == kCaptured` special case, as
# distinct from the kParam one just above). This is not just awkward to
# write -- it is not constructible as a model onnx.checker accepts at all: a
# subgraph output name that is neither produced by one of the subgraph's own
# nodes nor a formal graph input is rejected ("Graph output '...' is not an
# output of any node in graph"), and If's branch subgraphs never declare
# their captured names as formal inputs (that is precisely what makes them
# "captured" rather than ordinary parameters) -- confirmed by hand against
# both onnx.checker directly and the real compiled pass (which raises the
# same checker error before the pass itself ever runs, since onnxsim
# validates its input model up front).
