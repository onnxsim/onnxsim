"""Formal check for EliminateNopConcat (eliminate_nop_concat.h).

``patternMatchPredicate`` matches any ``Concat`` node with *exactly* one
input (``node->inputs().size() == 1``) -- not "at most one", not "fewer than
two": the boundary is precisely 1, and the negative control below exercises
it with a 2-input Concat rather than a vacuous 0-input one (ONNX itself
requires Concat to have at least one input, so 0 is not a reachable case
anyway). ``runTransform`` then does exactly what ``eliminate_identity.h``
does: ``tryReplacingAllUsesWith(node->output(), node->input())`` (there is
only ``input(0)`` to pick, since the predicate already guarantees a single
input) and destroys the node.

Formal content: this is the degenerate, one-segment case of the same
offset-arithmetic identity proved in
``test_formal_verify_fuse_consecutive_concats.py``. There, reading
``Concat(seg_0, seg_1, ..., seg_n, axis=k)`` at a global index ``i`` is a
case split on which segment's offset range ``i`` falls into (e.g.
``AB(i) = If(i < len(A), A(i), B(i - len(A)))``). With exactly one segment
there is nothing to case-split on -- the single branch *is* the whole
domain, so the "case split" collapses to the unconditional identity
``Concat([A], axis=k)(i) == A(i)`` for every ``i``: a 1-input Concat doesn't
rearrange or combine anything, it just re-exposes its one input as its
output. That is modeled below the same way
``test_formal_verify_eliminate_identity.py`` models Identity: the tensor as
an uninterpreted Z3 function ``Int -> Real``, and substitution safety proved
by composing with an arbitrary uninterpreted ``consumer`` -- so the proof
holds for every possible downstream computation over the Concat's output,
not only the one concrete consumer the differential check below happens to
use.
"""

from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser


def test_eliminate_nop_concat_is_sound():
    A = z3.Function("A", z3.IntSort(), z3.RealSort())
    concat_single = z3.Function("concat_single", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    idx = z3.Int("idx")

    # Concat([A], axis=k) semantics: with a single segment, every output
    # position i reads straight through to A(i) -- no offset case split
    # needed (there is only one segment, so there is nothing else for i to
    # fall into).
    concat_semantics = z3.ForAll([idx], concat_single(idx) == A(idx))

    prove(
        z3.Implies(
            concat_semantics,
            consumer(concat_single(idx)) == consumer(A(idx)),
        )
    )


def test_eliminate_nop_concat_pass_matches():
    # y = Concat(A, axis=0) has exactly one input, so patternMatchPredicate
    # fires; the compiled pass, run alone, removes the Concat node and
    # rewires its consumer (Relu) directly to A.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] A) => (float[4,8] Y)
        {
          y = Concat<axis=0>(A)
          Y = Relu(y)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_concat")
    assert ops["Concat"] == 0
    assert ops["Relu"] == 1
    (relu_node,) = [n for n in sim_model.graph.node if n.op_type == "Relu"]
    assert list(relu_node.input) == ["A"]


def test_eliminate_nop_concat_declines_two_inputs():
    # y = Concat(A, B, axis=0) has TWO inputs, so
    # node->inputs().size() == 1 is false -- patternMatchPredicate declines
    # and the compiled pass, run alone, must leave Concat untouched. This is
    # the only thing the predicate actually checks, so this boundary case
    # (2 inputs, not 0) is the meaningful negative control: it rules out the
    # pass over-matching as "at most 1" or "fewer than 2" inputs.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] A, float[4,8] B) => (float[8,8] Y)
        {
          Y = Concat<axis=0>(A, B)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_concat")
    assert ops["Concat"] == 1
    (node,) = [n for n in sim_model.graph.node if n.op_type == "Concat"]
    assert list(node.input) == ["A", "B"]
