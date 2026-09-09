"""Formal check for EliminateIdentity (eliminate_identity.h).

The rewrite deletes ``a = Identity(x)`` and rewires every use of ``a``
directly to ``x`` (``patternMatchPredicate`` matches an Identity node;
``runTransform`` calls ``tryReplacingAllUsesWith(node->output(), node->input())``).
Soundness needs two facts: Identity's own semantics (``a == x``), and that
ONNX's dataflow graph model has no notion of node identity beyond the values
it produces/consumes -- so *any* downstream computation over ``a`` gets the
same result over ``x``. Modeling the downstream computation as an arbitrary
uninterpreted function (rather than the one concrete consumer used in the
differential check below) is what makes this a real soundness argument
instead of just restating ``x == x``: the proof holds for every possible
consumer, not only the one this file happens to test against.
"""

from onnx import parser

from _formal_verify_common import prove, simplify_isolated, z3


def test_eliminate_identity_is_sound():
    x = z3.Real("x")
    a = z3.Real("a")
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    identity_semantics = a == x
    prove(z3.Implies(identity_semantics, consumer(a) == consumer(x)))


def test_eliminate_identity_pass_matches():
    # Differential check: the actual compiled pass, run alone, removes the
    # Identity node and rewires its consumer directly to X.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Identity(X)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_identity")
    assert ops["Identity"] == 0
    assert ops["Relu"] == 1
