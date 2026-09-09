"""Formal check for EliminateCommonSubexpression (eliminate_common_subexpression.h).

Like ``eliminate_duplicate_initializer``, this pass is a ``FullGraphBasedPass``
(whole-graph analysis via ``runPass(Graph&)``), not a ``PredicateBasedPass``.
It walks the graph's nodes in order; for each node it considers (skipping any
node with no uses at all, and any node ``IsSupportedByCSE`` excludes -- see
below), it hashes the node (``CSENodeHash``) and looks it up in a hash map
keyed by structural equality (``CSEEqual``, both in ``cse_util.h``). The first
time a given structural shape is seen, the node becomes that shape's
canonical representative in the map. Every later node found structurally
equal to an already-seen one has every one of its outputs rewired
(``tryReplacingAllUsesWith``) onto the corresponding output of that earlier,
canonical node, and is left dead for a later dead-code pass to remove.

``IsSupportedByCSE`` -- read directly from ``cse_util.h`` rather than assumed
-- is *not* an op-kind allowlist/denylist at all. It excludes a node purely
by the *kind of its attributes*: any node carrying a graph-valued attribute
(``AttributeKind::g``/``gs``, e.g. ``If``'s ``then_branch``/``else_branch`` or
``Loop``'s ``body``) or a sparse-tensor-valued one (``tp``/``ts``... here
``tp``/``tps``) is excluded, and every other node -- regardless of op type --
is eligible. So there is no fixed list of "CSE-supported ops" to look up;
ordinary compute ops like ``Relu``, ``Add``, and ``Cast`` are all eligible
simply because none of them ever carries a graph- or sparse-tensor-valued
attribute.

``CSEEqual``'s structural-identity notion -- the real engineering content of
this pass, and what the differential tests below focus on -- is, precisely:
same op kind, same number of inputs *and outputs*, the same set of attribute
names, matching attribute values for each (tensor-valued attributes compared
by ``CSETensorCompare``, i.e. shape+dtype+bytes, same as
``eliminate_duplicate_initializer``), and, crucially, same inputs *by
identity*: ``inputs_l[i]->uniqueName() != inputs_r[i]->uniqueName()`` compares
the ordered list of input **graph values (edges)** by name, not by any
looser notion of "equal shape" or "equal type" or "equal content". Two
``Add`` nodes reading ``(X, Y)`` and ``(X, Z)`` are therefore never merged no
matter how alike ``Y`` and ``Z`` might otherwise be -- they are literally
different edges in the graph -- which is the main negative control below.

Formal content, and why the proof is thin (matching
``eliminate_duplicate_initializer``'s own precedent for this style of pass):
this is not a nontrivial rewrite of some operator's algebra -- it is a
determinism/purity argument. If two node instances are structurally
identical -- same op kind, same attributes, and the exact same input
*values* (not merely equal-valued-but-distinct tensors, since ``CSEEqual``
compares graph edges directly) -- then, because an ONNX operator is a
(deterministic, side-effect-free) *function* of its attributes and inputs,
applying that same function to the same arguments must produce the same
result. This is modeled below as a single uninterpreted function
``op(attr, x, y)``: the hypothesis "two node instances share the exact same
``attr``, ``x``, ``y`` arguments" makes ``op(attr, x, y) == op(attr, x, y)``
hold by reflexivity -- essentially the definition of a function, not a
derivation -- and is stated as such rather than dressed up as deeper. The
negative control confirms this is not vacuous: two *independent*,
unconstrained applications of ``op`` are not provably equal.

One more empirically-confirmed subtlety, in the same spirit as
``eliminate_duplicate_initializer``'s ``producer()`` helper note in
``_formal_verify_common.py``: this pass only rewires the duplicate node's
*uses* onto the canonical node -- it does not itself delete the now-unused
duplicate. Deleting dead nodes is ``eliminate_deadend``'s job, a separate
default pass that ``isolate()`` disables here along with every other pass.
So a merge test run through ``simplify_isolated`` still shows *two* nodes of
the merged op type in the result (one live, one now-dead-but-still-present)
-- the positive tests below locate the live one by walking back from a real
graph output (``producer``) rather than by counting op types.
"""

from _formal_verify_common import isolate, producer, prove, simplify_isolated, z3
from onnx import parser


def test_eliminate_common_subexpression_is_sound():
    # op(attr, x, y) models an arbitrary ONNX operator as an uninterpreted,
    # deterministic function of its attribute(s) and (here, two) inputs.
    # Two node instances that share the exact same attr/x/y arguments -- the
    # formal counterpart of CSEEqual's "same kind, same attributes, same
    # input values" -- trivially compute the same output.
    op = z3.Function("op", z3.RealSort(), z3.RealSort(), z3.RealSort(), z3.RealSort())
    attr, x, y = z3.Reals("attr x y")
    prove(op(attr, x, y) == op(attr, x, y))


def test_eliminate_common_subexpression_negative_control_needs_hypothesis():
    # Without the "same arguments" hypothesis, two structurally-unrelated
    # applications of op (independent attr/x/y on each side) are NOT
    # provably equal -- confirming the claim above isn't vacuously true
    # regardless of which arguments are fed in. Z3 must find a sat
    # counterexample negating equality, not unsat.
    op = z3.Function("op", z3.RealSort(), z3.RealSort(), z3.RealSort(), z3.RealSort())
    attr1, x1, y1, attr2, x2, y2 = z3.Reals("attr1 x1 y1 attr2 x2 y2")

    solver = z3.Solver()
    solver.add(z3.Not(op(attr1, x1, y1) == op(attr2, x2, y2)))
    assert solver.check() == z3.sat


def test_eliminate_common_subexpression_pass_matches():
    # R1 and R2 are structurally identical Relu(X) nodes, each feeding a
    # separate downstream Add -- the compiled pass merges them: both
    # downstream Adds get redirected to read from a single Relu's output.
    #
    # This pass only rewires uses; it doesn't itself delete the
    # now-unused duplicate node (that is eliminate_deadend's job, a
    # separate default pass skipped here like every other one by
    # isolate() -- see simplify_isolated/producer's own docstring in
    # _formal_verify_common.py) -- so both `Relu` nodes are still present
    # in sim_model.graph.node, and only the *live* one (found by walking
    # back from a real graph output via `producer`) is meaningful to
    # check.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] G)
        {
          R1 = Relu(X)
          R2 = Relu(X)
          A1 = Add(R1, X)
          A2 = Add(R2, X)
          G = Add(A1, A2)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_common_subexpression")
    assert ops["Add"] == 3
    a1_node = producer(sim_model, "A1")
    a2_node = producer(sim_model, "A2")
    # Both surviving consumers read the same Relu output -- whichever of
    # R1/R2 the pass kept as canonical.
    assert a1_node.input[0] == a2_node.input[0]
    assert producer(sim_model, a1_node.input[0]).op_type == "Relu"


def test_eliminate_common_subexpression_declines_different_input_value():
    # Same op kind (Add), same first operand (X), but a DIFFERENT second
    # operand (Y vs Z) -- CSEEqual compares inputs by graph VALUE (edge
    # name), not by op-kind-and-shape, so these are not structurally equal
    # even though both are "an Add of X with some float[2,2]". Both nodes
    # must survive untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X, float[2,2] Y, float[2,2] Z) => (float[2,2] A1, float[2,2] A2)
        {
          A1 = Add(X, Y)
          A2 = Add(X, Z)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_common_subexpression")
    assert ops["Add"] == 2
    (a1_node,) = [n for n in sim_model.graph.node if list(n.output) == ["A1"]]
    (a2_node,) = [n for n in sim_model.graph.node if list(n.output) == ["A2"]]
    assert list(a1_node.input) == ["X", "Y"]
    assert list(a2_node.input) == ["X", "Z"]


def test_eliminate_common_subexpression_declines_different_kind():
    # Relu(X) and Sigmoid(X): same (single) input value, but a different op
    # kind -- a basic sanity check that CSEEqual's kind check is load-bearing.
    # Both must survive untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] R1, float[2,2] R2)
        {
          R1 = Relu(X)
          R2 = Sigmoid(X)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_common_subexpression")
    assert ops["Relu"] == 1
    assert ops["Sigmoid"] == 1


def test_eliminate_common_subexpression_declines_different_attribute():
    # Two Cast nodes on the same input X, but with different `to` dtypes
    # (FLOAT vs DOUBLE): CSEEqual compares attribute values too, so these are
    # not structurally equal despite sharing op kind and input. Both must
    # survive untouched. (Outputs are typed per-`to`, and left un-combined by
    # any arithmetic op, so onnxruntime's own type checking during
    # onnxsim's --check step doesn't itself get in the way of this test.)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] C1, double[2,2] C2)
        {
          C1 = Cast <to = 1> (X)
          C2 = Cast <to = 11> (X)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_common_subexpression")
    assert ops["Cast"] == 2


def test_eliminate_common_subexpression_merges_same_attribute():
    # Companion positive case to the test above: two Cast nodes on the same
    # input with the SAME `to` dtype are structurally identical and must
    # merge -- confirming the attribute check is comparing values (and
    # merges when they match), not just detecting mismatches. As in
    # test_eliminate_common_subexpression_pass_matches above, the
    # now-unused duplicate Cast node is left behind (eliminate_deadend is
    # skipped too), so the live one is found via `producer`.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] G)
        {
          C1 = Cast <to = 1> (X)
          C2 = Cast <to = 1> (X)
          G = Add(C1, C2)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_common_subexpression")
    assert ops["Add"] == 1
    g_node = producer(sim_model, "G")
    assert g_node.input[0] == g_node.input[1]
    assert producer(sim_model, g_node.input[0]).op_type == "Cast"


def test_isolate_accepts_eliminate_common_subexpression():
    # Sanity check that the pass name is a recognized default optimizer, so
    # isolate()/simplify_isolated actually exercise it above rather than
    # silently running the full default pipeline.
    assert "eliminate_common_subexpression" not in isolate(
        "eliminate_common_subexpression"
    )
