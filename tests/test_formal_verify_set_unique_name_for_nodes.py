"""Formal check for SetUniqueNameForNodes (opt-in;
``set_unique_name_for_nodes.h``): a ``PredicateBasedPass`` whose predicate
fires on every node that lacks a ``name`` -- ONNX's own ``NodeProto.name``
field, which is purely cosmetic/for-debugging and entirely distinct from a
node's *input*/*output* value names (the strings that actually carry
data-flow identity). Its transform assigns each such node a fresh,
graph-unique string via ``node->setName(nextReservedName(graph))``;
``NodeDestroyType::DestroyZero`` -- no node is created or destroyed, no
input/output is rewired, and no other node is touched at all. The batching
scheme behind ``nextReservedName`` (draw ``kNameBatchSize`` reserved names at
once via ``Graph::reserveUniqueNames``, rather than paying a full graph scan
per node) is a performance detail of how fresh names are minted, not part of
the correctness claim, so it isn't modeled below beyond a token check that it
doesn't produce a collision or an off-by-one over several unnamed nodes in
one graph.

Formal content, and why the proof here is intentionally thin: this pass
**does not change any computed value in the graph at all**. A node's own
``name`` field is never consulted by any ONNX operator's execution
semantics -- onnxruntime and onnx's own reference evaluator both identify
data flow purely through the *input*/*output value* names listed on each
node, never through the node's own optional ``name``, which exists solely
for human-readable debugging/error messages and tools like Netron. So, as
with ``lift_lexical_references``'s ``__control_inputs`` attribute and
``rename_input_output``'s boundary-``Value`` renaming, the claim to
formalize is not an operator-semantics algebra (there is none to get right
or wrong); it is: assigning an inert, execution-irrelevant per-node label
cannot change what that node -- or, by extension, the graph composed with
anything downstream -- computes, because nothing in ONNX's execution
semantics ever reads a node's ``name`` field. This is modeled below as an
uninterpreted ``annotate(inputs, node_name)`` function standing for "the
node's real output, augmented with an irrelevant name slot", constrained by
an *explicit* axiom that it is name-invariant (``annotate`` never actually
varies with its second argument) -- mirroring
``lift_lexical_references``'s own "metadata-invariance axiom" idiom, adapted
here to a node's own name rather than a ``__control_inputs`` list. As in
that file, a negative control confirms the claim is not vacuously true
without the invariance axiom.

Given how thin the algebra is, essentially all of the real verification
value is in the differential tests below, run against the actually compiled
pass via ``simplify_isolated_extra``:

1. A model whose nodes were all left unnamed by the parser gets every node
   assigned a non-empty, pairwise-distinct name.
2. The graph's actual computed outputs are numerically unchanged (covered by
   ``simplify_isolated_extra``'s own default ``check_n=3`` random-input
   equivalence check, which applies with no countermeasures needed here --
   unlike ``rename_input_output``, this pass never touches a graph-level
   input/output name).
3. A node that already has an explicit, non-empty name is left with that
   exact name -- confirms ``!node->has_name()`` really does gate on
   "already named" rather than clobbering it.
4. Several (eight) originally-unnamed nodes in one graph all come out
   pairwise distinct, a token exercise of the batched
   ``reserveUniqueNames``/``nextReservedName`` mechanism.

On node-name presence via ``onnx.parser``: confirmed empirically below
(``test_set_unique_name_for_nodes_parser_leaves_nodes_unnamed``) that
``onnx.parser.parse_model()`` leaves every node's ``name`` field entirely
unset (``NodeProto.HasField("name")`` is ``False``, matching
``ir_pb_converter.cc``'s own ``np.has_name()`` check used to populate
``Node::has_name_`` on import) when the text form never mentions a node
name -- there is no text-format syntax for a node's own name (only its
output value names), so no post-parse ``node.name = ""`` clearing is needed
to get genuinely unnamed nodes; an explicit name is instead set the same way
for the "already named" test, via ``model.graph.node[i].name = "..."``
directly on the parsed ``NodeProto``.
"""

from _formal_verify_common import prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim


def test_set_unique_name_for_nodes_is_sound():
    # annotate(inputs, node_name) stands for "the node's real output value,
    # with an extra inert name slot alongside its genuine inputs". The axiom
    # below is the actual claim this pass's soundness rests on: annotate
    # never actually varies with its node_name argument, for any inputs and
    # any two node names -- i.e. nothing that computes the graph's values
    # ever reads a node's own `name` field. Given that axiom, a downstream
    # consumer composed with annotate cannot tell which name was assigned.
    inputs, name1, name2 = z3.Ints("inputs name1 name2")
    annotate = z3.Function("annotate", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    name_invariant = z3.ForAll(
        [inputs, name1, name2], annotate(inputs, name1) == annotate(inputs, name2)
    )
    prove(
        z3.Implies(
            name_invariant,
            consumer(annotate(inputs, name1)) == consumer(annotate(inputs, name2)),
        )
    )


def test_set_unique_name_for_nodes_negative_control_needs_name_invariance_axiom():
    # Without the name-invariance axiom, annotate is a fully unconstrained
    # uninterpreted function of two arguments: the claim must NOT be valid
    # then (Z3 must find a counterexample where two different node names
    # really do produce different consumer results), or the proof above
    # would be vacuously true regardless of what the axiom says -- the same
    # shape of negative control lift_lexical_references's own test file uses
    # for its metadata-invariance axiom.
    inputs, name1, name2 = z3.Ints("inputs name1 name2")
    annotate = z3.Function("annotate", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    solver = z3.Solver()
    solver.add(name1 != name2)
    solver.add(
        z3.Not(consumer(annotate(inputs, name1)) == consumer(annotate(inputs, name2)))
    )
    assert solver.check() == z3.sat


def test_set_unique_name_for_nodes_is_an_opt_in_pass():
    C = onnxsim.onnxsim_cpp2py_export
    assert "set_unique_name_for_nodes" in C._list_other_optimizers()
    assert "set_unique_name_for_nodes" not in C._list_optimizers()


def test_set_unique_name_for_nodes_parser_leaves_nodes_unnamed():
    # There is no onnx text-format syntax for a node's own `name` field
    # (only its output value names) -- confirms parse_model() alone already
    # produces genuinely unnamed nodes (HasField("name") is False, matching
    # Node::has_name_ as populated by ir_pb_converter.cc on import), so the
    # differential tests below need no post-parse `node.name = ""` clearing.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] Z)
        {
          T = Relu(X)
          Z = Relu(T)
        }
        """
    )
    for node in model.graph.node:
        assert node.name == ""
        assert not node.HasField("name")


def test_set_unique_name_for_nodes_names_unnamed_nodes_and_preserves_numerics():
    # Two originally-unnamed nodes; simplify_isolated_extra's own default
    # check_n=3 random-input equivalence check (against onnxruntime/onnx's
    # reference evaluator) already confirms the graph's computed outputs are
    # numerically unchanged -- this pass never touches a graph-level
    # input/output name, so unlike rename_input_output's test file, no
    # countermeasure (check_n=0 + a manual onnxruntime comparison) is needed.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] Z)
        {
          T = Relu(X)
          Z = Relu(T)
        }
        """
    )
    for node in model.graph.node:
        assert not node.HasField("name")

    sim_model, ops = simplify_isolated_extra(
        model, "set_unique_name_for_nodes", check_n=3
    )
    assert ops == {"Relu": 2}

    names = [node.name for node in sim_model.graph.node]
    assert len(names) == 2
    assert all(name != "" for name in names)
    assert len(set(names)) == len(names)


def test_set_unique_name_for_nodes_leaves_existing_name_untouched():
    # T already has an explicit, non-empty name; Z is left unnamed by the
    # parser. Confirms the predicate (!node->has_name()) really does gate on
    # "already named" -- T's exact original name survives, it is not
    # overwritten -- while Z still gets a fresh, distinct name.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[2,2] X) => (float[2,2] Z)
        {
          T = Relu(X)
          Z = Relu(T)
        }
        """
    )
    model.graph.node[0].name = "MyExplicitNodeName"
    assert model.graph.node[0].HasField("name")
    assert not model.graph.node[1].HasField("name")

    sim_model, ops = simplify_isolated_extra(
        model, "set_unique_name_for_nodes", check_n=3
    )
    assert ops == {"Relu": 2}

    relu_names = [node.name for node in sim_model.graph.node]
    assert "MyExplicitNodeName" in relu_names
    assert all(name != "" for name in relu_names)
    assert len(set(relu_names)) == len(relu_names)


def test_set_unique_name_for_nodes_many_unnamed_nodes_stay_pairwise_distinct():
    # Eight unnamed nodes chained together -- a token exercise of the
    # batched reserveUniqueNames()/nextReservedName() mechanism (the header
    # comment's own kNameBatchSize=256 draws names in batches to avoid a
    # per-node full-graph scan): not deep coverage of that implementation
    # detail, just confirmation that batching doesn't accidentally produce a
    # collision or an off-by-one across several matches in one pass run.
    n_nodes = 8
    lines = [f"g (float[2,2] X0) => (float[2,2] X{n_nodes})", "{"]
    for i in range(n_nodes):
        lines.append(f"  X{i + 1} = Relu(X{i})")
    lines.append("}")
    body = "\n".join(lines)

    model = parser.parse_model(f'<ir_version: 10, opset_import: ["": 13]>\n{body}')
    assert len(model.graph.node) == n_nodes
    for node in model.graph.node:
        assert not node.HasField("name")

    sim_model, ops = simplify_isolated_extra(
        model, "set_unique_name_for_nodes", check_n=3
    )
    assert ops == {"Relu": n_nodes}

    names = [node.name for node in sim_model.graph.node]
    assert len(names) == n_nodes
    assert all(name != "" for name in names)
    assert len(set(names)) == n_nodes
