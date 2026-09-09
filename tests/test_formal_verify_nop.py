"""Formal check for NopEmptyPass (nop.h).

This is the simplest pass in the whole suite. Straight from the header:
``NopEmptyPass::runPass(Graph&)`` takes the graph by reference and does
absolutely nothing to it -- no node is added, removed, or modified -- and
just returns an empty ``PostPassAnalysis``. There is no comment in the
header explaining its purpose beyond the code itself, but its shape (a
registered pass named ``"nop"`` that provably changes nothing) is the
standard meaning of such a name in an optimizer pass registry: a documented
"do nothing" option/placeholder pass name, useful e.g. for testing the pass
harness itself or as an explicit no-op entry in a pass list.

Formal content, and why the proof here is not dressed up as more than it
is: there is no rewrite here to prove sound -- no algebra, no operator
semantics, not even the thin "substitution under equality"/"unreachable
code" arguments the other structurally-simple passes in this suite
(``eliminate_duplicate_initializer``, ``eliminate_deadend``) still needed.
A pass that is the identity function on graphs is trivially
semantics-preserving, for the same reason any identity function trivially
preserves whatever its argument denotes. This is modeled below as one
uninterpreted ``graph_semantics`` function -- standing for "whatever a
graph abstractly computes", with no attempt to model operators, tensors, or
graph structure at all -- applied to a single arbitrary/uninterpreted graph
constant ``G``, and the "proof" is exactly reflexivity of equality:
``graph_semantics(G) == graph_semantics(G)``. That is the entire content;
it is not narrowed down to this by argument, it simply *is* reflexivity,
stated honestly rather than padded out with machinery this pass has no use
for.

Given that the algebra is empty, all of the real verification value here is
in the differential tests below: confirming, against the actually compiled
pass, that running ``nop`` in isolation via ``simplify_isolated`` leaves a
concrete model's node list (kinds, order, inputs, outputs, attributes) and
initializers genuinely unchanged -- not merely "still computes the same
thing" (which the trivial proof above already covers by construction) but
structurally byte-for-byte identical.
"""

import numpy as np
from _formal_verify_common import isolate, prove, simplify_isolated, z3
from onnx import numpy_helper, parser

import onnxsim


def test_nop_is_sound():
    # No graph/operator structure is modeled at all -- graph_semantics is a
    # totally uninterpreted stand-in for "whatever a graph computes", and G
    # is a single arbitrary graph. The claim is exactly reflexivity: nop
    # changes nothing, so a graph's semantics before and after are the same
    # expression, syntactically.
    graph_semantics = z3.Function("graph_semantics", z3.IntSort(), z3.RealSort())
    G = z3.Int("G")
    prove(graph_semantics(G) == graph_semantics(G))


def test_nop_pass_matches_identity_on_multi_node_model():
    # A small chain of a few different op kinds, plus an initializer, run
    # through nop in isolation: the resulting graph's nodes (kind, inputs,
    # outputs, attributes, order) and initializers must be identical to the
    # input's -- not just equivalent, but the very same structure untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] Y)
        {
          v1 = Relu(X)
          v2 = Sigmoid(v1)
          v3 = Add(v2, W)
          Y = Transpose<perm = [1, 0]>(v3)
        }
        """
    )
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(2, 2), name="W"
        )
    )

    sim_model, ops = simplify_isolated(model, "nop")

    assert ops == {"Relu": 1, "Sigmoid": 1, "Add": 1, "Transpose": 1}
    assert [
        (n.op_type, list(n.input), list(n.output), list(n.attribute))
        for n in sim_model.graph.node
    ] == [
        (n.op_type, list(n.input), list(n.output), list(n.attribute))
        for n in model.graph.node
    ]
    assert [i.name for i in sim_model.graph.initializer] == ["W"]
    assert list(numpy_helper.to_array(sim_model.graph.initializer[0]).flatten()) == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_nop_is_a_real_default_pass():
    # Sanity check mirroring eliminate_deadend's own equivalent test: nop is
    # a genuine default onnxsim pass, and isolate() skips it whenever a
    # *different* single pass is isolated.
    assert "nop" in onnxsim.onnxsim_cpp2py_export._list_optimizers()
    assert "nop" in isolate("eliminate_identity")
