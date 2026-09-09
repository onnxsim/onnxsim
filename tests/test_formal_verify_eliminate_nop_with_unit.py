"""Formal check for EliminateOpWithUnit (eliminate_nop_with_unit.h).

``patternMatchPredicate`` always returns ``true`` -- every bit of actual
matching lives in ``runTransform``, which loops over ``node->inputs()`` for
the first input ``i`` that is BOTH a compile-time-fetchable constant tensor
(``FetchConstantTensor``) AND an "identity/unit" value for that node's op
kind at that operand position (``isUnit(tensor, kind, i)``):

* ``And``/``Mul``: the constant is all-ones (``isAllOne``), at *either*
  operand position -- ``isUnit`` doesn't look at ``index`` for these kinds.
* ``Or``/``Add``: the constant is all-zeros, again at either position.
* ``Sub``: only when the constant is at index **1** (the subtrahend) and
  all-zeros -- ``index == 1 && isAllOf(tensor, 0)``. Index 0 (the minuend)
  is never accepted, since ``0 - X == -X != X`` in general.
* ``Div``/``Pow``: only when the constant is at index **1** (the
  divisor/exponent) and all-ones -- same shape of restriction as ``Sub``.
* ``Concat``: the constant has zero elements (``ElemCntOfTensor(tensor) ==
  0``) -- no index restriction (any empty input, at any position, qualifies).

For the six binary-op kinds above (``isBroadcastBinaryOp``: And, Or, Mul,
Add, Sub, Div, Pow), a match additionally requires
``isABroadcastToB(tensor->sizes(), other_input->sizes())`` (the exact same
``pass_util.h`` helper modeled in ``test_formal_verify_eliminate_nop_expand.py``)
to hold *before* firing -- read there for the full per-axis definition; in
short, it holds only when broadcasting the constant's own shape against the
other operand's shape reproduces that other operand's shape unchanged, which
is exactly the condition under which "just return the other operand" doesn't
silently change the node's output shape. When it holds, ``runTransform``
calls ``tryReplacingAllUsesWith(node->output(), other_input)`` -- the matched
node itself is left dangling (0 uses) rather than destroyed outright, same
as e.g. ``fuse_matmul_add_bias_into_gemm``'s treatment of the original
``MatMul``. ``Concat`` is different machinery entirely: instead of replacing
the whole node's output, ``node->removeInput(i)`` just drops that one empty
input from the Concat's own input list in place -- the Concat node survives
with one fewer input, and there is no broadcast-shape guard for it (an empty
tensor trivially contributes nothing to any axis's output size). The loop
attempts (and returns after) exactly one rewrite per ``runTransform`` call --
either a successful ``tryReplacingAllUsesWith``/``removeInput``, or (for the
binary-op case) continuing to the next input if ``isUnit`` held but the
broadcast guard failed for that particular input.

Given how many op kinds and index-specific branches ``isUnit`` packs into one
function, this file does not re-derive every op's identity element from
scratch. It proves two representative pieces of algebra in Z3:

1. The ``Mul``-by-all-ones / ``Add``-by-all-zeros pair, modeled so the
   constant unit operand can be at *either* input position (matching
   ``isUnit``'s lack of an ``index`` check for And/Or/Mul/Add) -- the most
   common cases, and the ones where operand order genuinely doesn't matter.
2. The ``Sub``/``Div``/``Pow`` family's order-*sensitivity* -- the one place
   this pass's own C++ hard-codes ``index == 1``, i.e. a genuine
   operand-order-matters argument in the spirit of
   ``test_formal_verify_fuse_matmul_add_bias_into_gemm.py``'s own
   swapped-operand negative control.

``Concat``'s empty-input case gets no separate Z3 proof: removing a
zero-length segment from a concatenation is a degenerate instance of the
same offset-arithmetic content already proven in
``test_formal_verify_fuse_consecutive_concats.py`` (there, splicing a
Concat's inputs into another Concat via index-offset remapping; here, the
"spliced-in" segment simply has length 0, so removing it changes no other
segment's offset). It is covered only by a differential test below, plus
this note.

Every differential test below was run against the real compiled pass first
(via a throwaway debug script, since deleted) to confirm the claims in this
docstring rather than assuming a paraphrase of the C++ generalizes --
notably: (a) the ``Add(zeros, X)`` commutative case does fire with the
constant at index 0; (b) ``Sub``/``Div`` genuinely decline when the constant
is at index 0 instead of 1; (c) ``Concat``'s empty-input removal really does
leave the Concat node alive with a shorter input list, matching
``removeInput`` rather than a whole-node replacement; and (d) the
broadcast-shape guard is not merely defense-in-depth here -- it is possible
to construct a concrete case where an operand is genuinely all-ones/all-zeros
yet the rewrite is correctly declined because returning the other operand
outright would silently change the output's shape (see
``test_eliminate_nop_with_unit_declines_on_broadcast_shape_mismatch`` below).
"""

from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser


def test_eliminate_nop_with_unit_mul_add_is_sound():
    # `ones`/`zeros` model isAllOne/isAllOf(tensor, 0) as arbitrary tensor
    # functions constrained (via the ForAll hypotheses below) to actually be
    # uniformly 1 / 0 everywhere -- i.e. the exact property isUnit checks --
    # rather than modeling them as the literal scalar constants 1 and 0,
    # which would trivialize the claim via Z3's built-in arithmetic
    # simplification instead of exercising the "elementwise-uniform" shape
    # of the real predicate.
    tensor = z3.Function("tensor", z3.IntSort(), z3.RealSort())
    ones = z3.Function("ones", z3.IntSort(), z3.RealSort())
    zeros = z3.Function("zeros", z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    idx, k = z3.Ints("idx k")

    all_ones = z3.ForAll(k, ones(k) == 1)
    all_zeros = z3.ForAll(k, zeros(k) == 0)

    # Both operand orders for each op are modeled explicitly (tensor*unit and
    # unit*tensor, tensor+unit and unit+tensor) -- matching isUnit's lack of
    # an `index` check for And/Or/Mul/Add -- and composed with an arbitrary
    # uninterpreted `consumer` to prove substitution safety for any
    # downstream consumer of the node's output, matching this repo's
    # established style.
    claim = z3.Implies(
        z3.And(all_ones, all_zeros),
        z3.And(
            consumer(tensor(idx) * ones(idx)) == consumer(tensor(idx)),
            consumer(ones(idx) * tensor(idx)) == consumer(tensor(idx)),
            consumer(tensor(idx) + zeros(idx)) == consumer(tensor(idx)),
            consumer(zeros(idx) + tensor(idx)) == consumer(tensor(idx)),
        ),
    )
    prove(claim)


def test_eliminate_nop_with_unit_sub_div_pow_order_sensitive():
    # Sub's "x - 0 == x" and Div's "x / 1 == x" are ordinary real-arithmetic
    # facts. Pow is modeled abstractly via an uninterpreted `power` function
    # with only the one axiom isUnit actually relies on (exponent-1 is an
    # identity) -- there's no need to teach Z3 general real exponentiation
    # (which is ill-defined/multivalued for negative bases) just to capture
    # the one algebraic fact this pass's rewrite depends on.
    x = z3.Real("x")
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    power = z3.Function("power", z3.RealSort(), z3.RealSort(), z3.RealSort())
    pow_exponent_one_is_identity = z3.ForAll(x, power(x, 1) == x)

    positive_claim = z3.Implies(
        pow_exponent_one_is_identity,
        z3.And(
            consumer(x - 0) == consumer(x),
            consumer(x / 1) == consumer(x),
            consumer(power(x, 1)) == consumer(x),
        ),
    )
    prove(positive_claim)

    # The genuinely order-*sensitive* half: isUnit's `index == 1` check for
    # Sub means the minuend-is-zero case (index 0) is deliberately never
    # accepted, because `0 - x == x` is NOT a general identity -- unlike the
    # Mul/Add pair above, swapping operand order here changes the result.
    # Demonstrated with a concrete Z3-found counterexample rather than only
    # asserted in a comment (the real-pass declining to fire on this case is
    # separately confirmed empirically by
    # test_eliminate_nop_with_unit_declines_sub_index0 below).
    solver = z3.Solver()
    solver.add(0 - x != x)
    result = solver.check()
    assert result == z3.sat, "0 - x == x should NOT be a general identity"
    counterexample = solver.model()[x].as_fraction()
    assert counterexample != 0, (
        f"expected a genuine nonzero counterexample, got {counterexample}"
    )


def test_eliminate_nop_with_unit_pass_matches_mul_ones_index1():
    # Mul(X, ones) -- constant at index 1, isAllOne holds, isABroadcastToB
    # trivially holds (ones' shape [8] broadcasts into X's own shape [4,8]
    # unchanged). Fires: the Relu consumer is rewired directly to X, and Mul
    # is left dangling (0 uses) rather than destroyed, per the module
    # docstring.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <float[8] ones = {1, 1, 1, 1, 1, 1, 1, 1}>
        {
          m = Mul(X, ones)
          Y = Relu(m)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_with_unit")
    assert ops["Relu"] == 1
    (relu_node,) = [n for n in sim_model.graph.node if n.op_type == "Relu"]
    assert list(relu_node.input) == ["X"]


def test_eliminate_nop_with_unit_pass_matches_add_zeros_index0():
    # Add(zeros, X) -- the constant this time at index 0, exercising
    # isUnit's lack of an index check for Add/Mul/And/Or (unlike
    # Sub/Div/Pow). Fires just as readily as the index-1 case.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <float[8] zeros = {0, 0, 0, 0, 0, 0, 0, 0}>
        {
          a = Add(zeros, X)
          Y = Relu(a)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_with_unit")
    assert ops["Relu"] == 1
    (relu_node,) = [n for n in sim_model.graph.node if n.op_type == "Relu"]
    assert list(relu_node.input) == ["X"]


def test_eliminate_nop_with_unit_pass_matches_sub_index1():
    # Sub(X, zeros) -- constant at index 1 (the subtrahend): isUnit's
    # `index == 1` check is satisfied, so this fires.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <float[8] zeros = {0, 0, 0, 0, 0, 0, 0, 0}>
        {
          s = Sub(X, zeros)
          Y = Relu(s)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_with_unit")
    assert ops["Relu"] == 1
    (relu_node,) = [n for n in sim_model.graph.node if n.op_type == "Relu"]
    assert list(relu_node.input) == ["X"]


def test_eliminate_nop_with_unit_declines_sub_index0():
    # Sub(zeros, X) -- the constant at index 0 (the MINUEND). `0 - X` is
    # `-X`, not `X`, so isUnit's `index == 1` guard must (and empirically
    # does) block this: a real negative control, not merely a hypothetical
    # one, and the pass-level analogue of
    # test_eliminate_nop_with_unit_sub_div_pow_order_sensitive's Z3
    # counterexample above.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <float[8] zeros = {0, 0, 0, 0, 0, 0, 0, 0}>
        {
          s = Sub(zeros, X)
          Y = Relu(s)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_with_unit")
    assert ops["Sub"] == 1
    (relu_node,) = [n for n in sim_model.graph.node if n.op_type == "Relu"]
    assert list(relu_node.input) == ["s"]


def test_eliminate_nop_with_unit_pass_matches_div_index1():
    # Div(X, ones) -- divisor (index 1) all-ones: fires, same shape of
    # restriction as Sub above.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <float[8] ones = {1, 1, 1, 1, 1, 1, 1, 1}>
        {
          d = Div(X, ones)
          Y = Relu(d)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_with_unit")
    assert ops["Relu"] == 1
    (relu_node,) = [n for n in sim_model.graph.node if n.op_type == "Relu"]
    assert list(relu_node.input) == ["X"]


def test_eliminate_nop_with_unit_declines_div_index0():
    # Div(ones, X) -- constant at index 0 (the DIVIDEND). `1 / X` is a
    # reciprocal, not `X`, so this must not fire -- the Div/Pow analogue of
    # the Sub negative control above, confirming isUnit's `index == 1` check
    # is load-bearing for this whole family, not just Sub specifically.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <float[8] ones = {1, 1, 1, 1, 1, 1, 1, 1}>
        {
          d = Div(ones, X)
          Y = Relu(d)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_with_unit")
    assert ops["Div"] == 1
    (relu_node,) = [n for n in sim_model.graph.node if n.op_type == "Relu"]
    assert list(relu_node.input) == ["d"]


def test_eliminate_nop_with_unit_pass_matches_concat_empty_input():
    # Concat(X, empty) where `empty` is a zero-element (shape [0, 8])
    # constant: ElemCntOfTensor(tensor) == 0 holds, so this fires -- but via
    # the Concat-specific `removeInput` branch, not
    # `tryReplacingAllUsesWith`. Confirmed empirically: the Concat node
    # itself survives (same output name `c`), just with `empty` dropped from
    # its input list, unlike every binary-op case above where the matched
    # node is left dangling and a *different* node (Relu) gets rewired.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <float[0,8] empty = {}>
        {
          c = Concat<axis=0>(X, empty)
          Y = Relu(c)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_with_unit")
    assert ops["Concat"] == 1
    (concat_node,) = [n for n in sim_model.graph.node if n.op_type == "Concat"]
    assert list(concat_node.input) == ["X"]
    assert concat_node.output[0] == "c"
    (relu_node,) = [n for n in sim_model.graph.node if n.op_type == "Relu"]
    assert list(relu_node.input) == ["c"]


def test_eliminate_nop_with_unit_declines_on_broadcast_shape_mismatch():
    # A constructed, empirically-confirmed case where the broadcast-shape
    # guard actually matters (not just defense-in-depth): X has shape [8]
    # and `ones` is a genuinely all-ones [4, 8] constant. `Mul(X, ones)`'s
    # *output* shape is [4, 8] (X gets broadcast up to it). If runTransform
    # fired here anyway and returned X directly, the node's output would
    # silently shrink to shape [8] -- wrong. isABroadcastToB(ones.sizes()=
    # [4, 8], X.sizes()=[8]) is false immediately (its own ndim_a > ndim_b
    # check, see pass_util.h and test_formal_verify_eliminate_nop_expand.py),
    # so the guard correctly blocks this even though `ones` genuinely
    # satisfies isAllOne. Same reasoning as
    # test_eliminate_nop_expand_declines_on_real_broadcast, adapted to a
    # rewrite whose "other operand" isn't the one being resized.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[8] X) => (float[4,8] Y)
        <float[4,8] ones = {1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1}>
        {
          m = Mul(X, ones)
          Y = Relu(m)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_with_unit")
    assert ops["Mul"] == 1
    (relu_node,) = [n for n in sim_model.graph.node if n.op_type == "Relu"]
    assert list(relu_node.input) == ["m"]
