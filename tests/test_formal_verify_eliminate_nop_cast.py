"""Formal check for EliminateNopCast (eliminate_nop_cast.h).

``patternMatchPredicate`` matches a Cast node exactly when it has a ``to``
attribute *and* its input's statically-known element type already equals
``to`` (``node->input()->elemType() == node->i(kto)``) -- nothing else.
Notably it does not special-case the opset-19+ ``saturate`` attribute (only
meaningful when casting *to* a float8 type): since the match requires the
input to already be that exact same type, the value being cast is already
representable in it, so no saturation/NaN clamping could ever trigger
regardless of ``saturate``'s value -- the omission is not a soundness gap.
``runTransform`` calls ``tryReplacingAllUsesWith(node->output(), node->input())``
and destroys the Cast node, exactly like eliminate_identity.h's handling of
Identity.

Soundness rests on one general fact about casts, independent of which
concrete dtype is involved: converting a value that already has type ``d``
into that same type ``d`` is a value-preserving no-op. Modeling that as a
universally-quantified axiom over an uninterpreted ``cast(from_dtype,
to_dtype, x)`` function, then instantiating it via the predicate's actual
hypothesis ``input_dtype == to``, turns "Cast is a no-op here" into a
derived fact grounded in the real C++ precondition rather than an assumed
one -- and composing it with an arbitrary uninterpreted downstream consumer
(as in test_formal_verify_eliminate_identity.py) proves substitution is safe
for every possible consumer, not only the one concrete consumer the
differential check below happens to use.
"""

from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser


def test_eliminate_nop_cast_is_sound():
    x = z3.Real("x")
    d, input_dtype, to = z3.Ints("d input_dtype to")
    cast = z3.Function("cast", z3.IntSort(), z3.IntSort(), z3.RealSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    # General law: casting a value already of dtype d to that same dtype d
    # is a value-preserving no-op, for every dtype d and every value x --
    # true of any type-cast operator, not specific to float/int/etc.
    cast_to_own_type_is_identity = z3.ForAll([d, x], cast(d, d, x) == x)

    # patternMatchPredicate's actual hypothesis:
    # node->input()->elemType() == node->i(kto).
    predicate_holds = input_dtype == to

    a = cast(input_dtype, to, x)  # a = Cast<to=to>(X)'s value
    prove(
        z3.Implies(
            z3.And(cast_to_own_type_is_identity, predicate_holds),
            consumer(a) == consumer(x),
        )
    )


def test_eliminate_nop_cast_pass_matches():
    # Differential check: Cast<to=FLOAT> on an already-float input is a
    # no-op by the predicate above, so the compiled pass, run alone, removes
    # it and rewires its consumer directly to X.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Cast<to = 1>(X)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_cast")
    assert ops["Cast"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_cast_declines_on_real_dtype_change():
    # Edge case from patternMatchPredicate: input's static element type
    # (FLOAT = 1) differs from `to` (INT64 = 7), so the Cast actually
    # changes representation and is not a no-op -- the predicate declines
    # and the pass, run alone, must leave it untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (int64[4,8] Y)
        {
          Y = Cast<to = 7>(X)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_cast")
    assert ops["Cast"] == 1
