"""Formal check for EliminateNopFlatten (eliminate_nop_flatten.h).

``patternMatchPredicate`` matches a ``Flatten`` node only when its input has
a statically-known shape (``input->has_sizes()``) of rank exactly 2, and
either:

  (a) ``axis == 1`` or ``axis == -1`` -- ``axis`` is read via
      ``GetValueFromAttrWithDefault(node, kaxis, 1)``, so an absent ``axis``
      attribute defaults to 1, the same branch as an explicit ``axis=1``; or
  (b) ``axis == 0`` AND the input's dim 0 is statically known to equal 1
      (``input_shape[0].is_int && input_shape[0].dim == 1``).

Any other rank, any unresolved input shape, or ``axis == 0`` with a dim-0
that isn't provably 1 all decline. ``runTransform`` then does exactly what
``eliminate_identity.h`` does: ``tryReplacingAllUsesWith(node->output(),
node->input())`` and destroys the node -- a pure identity substitution, not
a "shape-compatible but value-shuffling" reshape.

Why both admitted cases really are the identity, not just shape-compatible:
Flatten with axis ``a`` on shape ``[d_0, ..., d_{r-1}]`` produces
``[prod(d_0..d_{a-1}), prod(d_a..d_{r-1})]``. Restricted to rank-2 input
``[d0, d1]`` (the predicate's own hard requirement -- this file does not
model, and does not need, the general N-dimensional formula):

  - ``axis in {1, -1}``: groups ``[d0]`` and ``[d1]`` -- output shape is
    exactly ``[d0, d1]``, identical to the input's, and since Flatten never
    reorders elements (it's a row-major reshape), value ``(i, j)`` of the
    output is value ``(i, j)`` of the input, verbatim.
  - ``axis == 0`` with ``d0 == 1``: groups ``[]`` and ``[d0, d1]`` -- output
    shape is ``[1, d0*d1]``, which (since ``d0 == 1``) is ``[1, d1]``, again
    identical to the input's shape. The only valid output row is row 0, and
    output position ``(0, j)`` reads input position ``(0, j)``.

The proof below encodes exactly those two closed-form cases (row 0 for the
``axis == 0`` branch, matching row/col for the other) rather than a generic
row-major linearization formula -- Z3's nonlinear integer arithmetic chokes
on a symbolic ``d0 * d1`` divisor, and the task doesn't require modeling
ranks/shapes this file never proves anything about. Composing the pointwise
identity with an arbitrary uninterpreted downstream ``consumer`` (as in
test_formal_verify_eliminate_identity.py) proves substitution safety for
every possible consumer, not only the ones this file's differential checks
happen to use. The ``domain`` hypothesis (``0 <= i < d0``, ``0 <= j < d1``)
is what turns ``axis == 0 and d0 == 1`` into ``i == 0`` -- there is no other
valid row to read.
"""

from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser


def _model(body, opset=13, ir_version=10):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def test_eliminate_nop_flatten_is_sound():
    x = z3.Function("x", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    axis, d0, d1, i, j = z3.Ints("axis d0 d1 i j")

    is_axis_1_or_neg1 = z3.Or(axis == 1, axis == -1)
    # axis in {1, -1}: output(i, j) == input(i, j) (same shape, no grouping
    # change). axis == 0: only row 0 is valid output; output(0, j) reads
    # input(0, j) -- i.e. x(0, j), independent of i.
    flatten_val = z3.If(is_axis_1_or_neg1, x(i, j), x(0, j))

    # i, j range over the rank-2 input/output's valid positions.
    domain = z3.And(d0 > 0, d1 > 0, 0 <= i, i < d0, 0 <= j, j < d1)
    # patternMatchPredicate's own admitted set, restated for rank 2.
    predicate_hypothesis = z3.Or(is_axis_1_or_neg1, z3.And(axis == 0, d0 == 1))

    prove(
        z3.Implies(
            z3.And(domain, predicate_hypothesis),
            consumer(flatten_val) == consumer(x(i, j)),
        )
    )


def test_eliminate_nop_flatten_pass_matches_explicit_axis_1():
    # Differential check: axis=1 on a rank-2 input matches branch (a), so
    # the compiled pass, run alone, removes Flatten and rewires Relu to X.
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Flatten<axis = 1>(X)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_flatten")
    assert ops["Flatten"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_flatten_pass_matches_default_axis():
    # Same, but with no `axis` attribute at all -- GetValueFromAttrWithDefault
    # defaults it to 1, so this should fire identically to the explicit case.
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Flatten(X)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_flatten")
    assert ops["Flatten"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_flatten_pass_matches_axis_neg1():
    # axis=-1 matches branch (a) too (`axis == 1 || axis == -1`).
    model = _model(
        """
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Flatten<axis = -1>(X)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_flatten")
    assert ops["Flatten"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_flatten_pass_matches_axis_0_with_static_unit_dim0():
    # axis=0 matches branch (b) only when dim 0 is statically known to be 1;
    # here X's shape is [1, 8], so it does.
    model = _model(
        """
        g (float[1,8] X) => (float[1,8] Y)
        {
          a = Flatten<axis = 0>(X)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_flatten")
    assert ops["Flatten"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_flatten_declines_on_axis_0_non_unit_dim0():
    # Edge case from branch (b): axis=0 but dim 0 is statically 4, not 1, so
    # `input_shape[0].dim == 1` is false and the predicate declines. This is
    # a genuinely different result too -- Flatten<axis=0> on [4, 8] produces
    # [1, 32], not [4, 8] -- a good negative control.
    model = _model(
        """
        g (float[4,8] X) => (float[1,32] Y)
        {
          Y = Flatten<axis = 0>(X)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_flatten")
    assert ops["Flatten"] == 1


def test_eliminate_nop_flatten_declines_on_rank_3_input():
    # Edge case from the predicate's own rank check: `input_shape.size() ==
    # 2` is required outright, so a rank-3 input declines regardless of
    # axis, even one ([1, 4, 8] with axis=1) that would otherwise look like
    # a shape-preserving no-op-ish grouping.
    model = _model(
        """
        g (float[1,4,8] X) => (float[1,32] Y)
        {
          Y = Flatten<axis = 1>(X)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_flatten")
    assert ops["Flatten"] == 1
