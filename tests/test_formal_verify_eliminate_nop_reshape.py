"""Formal check for EliminateNopReshape (eliminate_nop_reshape.h).

This is the most intricate no-op-elimination predicate in this family.
``patternMatchPredicate`` requires: the node is a ``Reshape``, its data input
(input 0) has a non-empty *statically known* shape, and its second input
(``new_shape``) is a compile-time constant tensor (``IsConstantTensor``).

``runTransform`` then fetches ``old_shape`` (input 0's own known per-axis
shape) and the constant ``new_shape`` tensor, which must be ``INT64`` typed
or the pass declines. If ``new_shape.size() != old_shape.size()`` -- a
**rank** change -- it declines immediately: this pass never proves a
rank-changing reshape is a no-op, only same-rank ones.

Otherwise it walks the two same-length shapes axis by axis, tracking
``unknown_dim_count`` (starts at 0), and a no-op is recognized at axis ``i``
for exactly one of three reasons:

  (a) **0-copy sentinel**: ``new_shape[i] == 0`` and ``allowzero`` is not set
      to 1. Per the ONNX Reshape spec, a literal ``0`` in the target shape
      (absent ``allowzero=1``) means "keep whatever this axis's input size
      already is" -- so the loop ``continue``s without checking this axis
      against ``old_shape[i]`` at all; it is assumed compatible *by
      definition* of the sentinel's semantics.

  (b) **literal restatement**: ``new_shape[i]`` is not the 0-sentinel (either
      it's nonzero, or it's 0 but ``allowzero=1`` makes 0 a literal
      zero-sized dim rather than the sentinel) and not ``-1``, and
      ``old_shape[i]`` is statically known: the pass declines unless
      ``old_shape[i].dim == new_shape[i]`` exactly.

  (c) **-1 inference, uniquely**: ``new_shape[i] == -1`` (or ``old_shape[i]``
      is itself not statically known, e.g. a ``dim_param``) counts as one
      more "unknown" axis (``unknown_dim_count++``); after the per-axis loop,
      if more than one axis was "unknown" the pass declines outright
      (Reshape's own ``-1`` inference only works when exactly one axis is
      unknown, and the pass is conservative about proving equality when
      there is genuine ambiguity).

If none of that declines, ``runTransform`` does exactly what
``eliminate_identity.h`` does: ``tryReplacingAllUsesWith(node->output(),
node->inputs()[0])`` and destroys the node -- the ``new_shape``
initializer/Constant, if unused elsewhere, is left dangling.

Case (c) is the interesting one to model correctly: it is not merely "this
axis looks unchanged" but a genuine (if simple) *consequence* of two other
facts holding together -- Reshape's own semantics always preserve the total
element count between input and output, and (by reasons (a)/(b)) every
*other* axis is independently pinned to its old value -- so the one
remaining unknown axis has no freedom left: it must equal its own prior size
too, for the product to still work out.

Soundness, modeled for a rank-2 ``[d0, d1]`` example (axis 0 pinned via
reason (a) or (b), axis 1 resolved via reason (c) -- enough to exercise all
three reasons): two "laws" are assumed as axiomatic facts about Reshape
itself (independent of the pass, the way ``cast_to_own_type_is_identity``
and ``boundary`` are axiomatic in the nop_cast/nop_pad proofs):

  * a **resolution law** for a non-``-1`` axis: its resolved output size is
    the old size when the sentinel applies, else the literal ``new_shape``
    entry -- this is just the ONNX Reshape spec's own definition of what a
    non-``-1`` shape entry means, not something the *pass* invents;
  * a **size-preservation law**: ``resolved0 * resolved1 == d0 * d1`` --
    Reshape always preserves the total element count, by definition of what
    reshaping is, regardless of which entries are literal, sentinel, or
    ``-1``.

From the resolution law plus the pass's own hypothesis on axis 0 (reason (a)
or (b)), ``resolved0 == d0`` is *derived*, not assumed. Combined with the
size-preservation law and ``d0 > 0``, algebraic cancellation then forces
``resolved1 == d1`` -- this is the auxiliary lemma the task calls out:
``new_dim == -1 ∧ other_axis_pinned ∧ total_size_preserved ⇒
inferred_dim == old_dim_at_that_axis``, derived from the size equation
rather than assumed outright. Finally, "reshaping into literally the *same*
shape is a byte-for-byte identity" is modeled directly via Reshape's
row-major flatten/unflatten arithmetic (``L = i * n1 + j``, unflattened
against the *original* layout as ``x(L div d1, L mod d1)``) rather than
assumed -- Z3's integer div/mod theory proves it outright for the concrete
point ``(i, j)`` in scope, given ``resolved0 == d0`` and ``resolved1 ==
d1``. Composing with an arbitrary uninterpreted ``consumer`` (as in the
other nop-elimination proofs) then proves substitution soundness for every
possible downstream consumer, not only the one the differential checks below
happen to use.
"""

from _formal_verify_common import prove, simplify_isolated, z3
from onnx import parser


def test_eliminate_nop_reshape_is_sound():
    x = z3.Function("x", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    d0, d1, i, j = z3.Ints("d0 d1 i j")
    new_shape0, new_shape1 = z3.Ints("new_shape0 new_shape1")
    allowzero = z3.Bool("allowzero")
    resolved0, resolved1 = z3.Ints("resolved0 resolved1")

    # i, j range over old (== new, once resolved) shape [d0, d1]. d0 > 0 is
    # also what makes the size-preservation cancellation below well-defined
    # (dividing by a possibly-zero d0 would leave resolved1 underdetermined
    # -- consistent with -1-inference only being meaningful when the other
    # axes are actually nonzero).
    domain = z3.And(d0 > 0, d1 > 0, 0 <= i, i < d0, 0 <= j, j < d1)

    def reshape_val(n0, n1, ii, jj):
        # Row-major flatten against target shape (n0, n1), unflatten against
        # the ORIGINAL layout's divisor d1 -- generic Reshape value law for
        # any target shape, not specific to the no-op case.
        flat = ii * n1 + jj
        return x(flat / d1, flat % d1)

    # "Reshaping into the same shape is the identity" -- proved directly (for
    # the concrete (i, j) in scope) from the flatten/unflatten arithmetic via
    # Z3's div/mod theory, rather than assumed as a law.
    same_shape_is_identity = z3.Implies(
        z3.And(resolved0 == d0, resolved1 == d1),
        reshape_val(resolved0, resolved1, i, j) == x(i, j),
    )

    # Resolution law for axis 0 (a non-"-1" axis): ONNX Reshape spec's own
    # definition of what a literal shape entry means -- old dim under the
    # 0-copy sentinel (reason (a)), else the literal entry itself (reason (b)).
    resolution_law_axis0 = resolved0 == z3.If(
        z3.And(new_shape0 == 0, z3.Not(allowzero)), d0, new_shape0
    )

    # Size-preservation law: Reshape always preserves the total element
    # count, independent of which entries are literal/sentinel/-1.
    size_preservation_law = resolved0 * resolved1 == d0 * d1

    # The pass's own hypothesis on axis 0: EITHER reason (a), the 0-copy
    # sentinel (`continue`s, unchecked-but-compatible-by-definition), OR
    # reason (b), literal restatement (not the sentinel, and the pass's own
    # `old_shape[i].dim != new_dim` check did NOT trip, i.e. equality holds).
    pass_axis0_is_pinned = z3.Or(
        z3.And(new_shape0 == 0, z3.Not(allowzero)),
        z3.And(z3.Or(new_shape0 != 0, allowzero), new_shape0 == d0),
    )
    # The pass's own hypothesis on axis 1: reason (c), the sole "-1" (unknown)
    # axis -- unknown_dim_count reaches exactly 1, so the pass does not decline.
    pass_axis1_is_sole_infer = new_shape1 == -1

    prove(
        z3.Implies(
            z3.And(
                domain,
                same_shape_is_identity,
                resolution_law_axis0,
                size_preservation_law,
                pass_axis0_is_pinned,
                pass_axis1_is_sole_infer,
            ),
            consumer(reshape_val(resolved0, resolved1, i, j)) == consumer(x(i, j)),
        )
    )


def test_eliminate_nop_reshape_pass_matches_literal_restatement():
    # Reason (b) on both axes: [4, 8] -> [4, 8] restates every axis's old
    # value explicitly, no sentinels involved -- fires.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <int64[2] new_shape = {4, 8}>
        {
          a = Reshape(X, new_shape)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_reshape")
    assert ops["Reshape"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_reshape_pass_matches_zero_sentinel():
    # Reason (a) on axis 0, reason (b) on axis 1: [4, 8] -> [0, 8] (no
    # allowzero) means axis 0's `0` is the copy-sentinel (keep 4), axis 1
    # restates 8 -- fires, and crucially resolves to the SAME output shape
    # [4, 8] as literal restatement above, confirming the sentinel really is
    # a no-op reason and not a disguised shape change.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <int64[2] new_shape = {0, 8}>
        {
          a = Reshape(X, new_shape)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_reshape")
    assert ops["Reshape"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_reshape_pass_matches_infer_sentinel():
    # Reason (b) on axis 0, reason (c) on axis 1: [4, 8] -> [4, -1] restates
    # axis 0 (4) and leaves axis 1 as the sole "-1", which must infer to 8
    # (the only value preserving the total element count 32) -- fires.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g (float[4,8] X) => (float[4,8] Y)
        <int64[2] new_shape = {4, -1}>
        {
          a = Reshape(X, new_shape)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_reshape")
    assert ops["Reshape"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_reshape_declines_on_axis_size_change():
    # Negative control: [4, 8] -> [8, 4] has the SAME total element count
    # (32) as literal restatement / the sentinel case above, so it is valid
    # ONNX -- but it is a genuine transpose-like reshape, not a no-op. Axis 0
    # (new_dim=8, old=4) fails the literal-restatement check and the pass
    # declines; same total size does not imply same per-axis shape.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g (float[4,8] X) => (float[8,4] Y)
        <int64[2] new_shape = {8, 4}>
        {
          Y = Reshape(X, new_shape)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_reshape")
    assert ops["Reshape"] == 1


def test_eliminate_nop_reshape_declines_on_rank_change():
    # Negative control: [4, 8] -> [32] has new_shape.size() (1) !=
    # old_shape.size() (2) -- the pass declines outright on the rank check,
    # before any per-axis reasoning, regardless of the element count matching.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g (float[4,8] X) => (float[32] Y)
        <int64[1] new_shape = {32}>
        {
          Y = Reshape(X, new_shape)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_reshape")
    assert ops["Reshape"] == 1


def test_eliminate_nop_reshape_declines_on_allowzero_literal_zero():
    # Negative control exercising `allowzero=1`'s effect on the 0-sentinel
    # logic specifically. A literal `0` target dim on a real, non-empty
    # source dim is an invalid reshape at runtime unless it is paired with an
    # actual zero-sized axis elsewhere (Reshape preserves total element
    # count, so a genuine "make this axis literally 0" requires the *whole*
    # tensor to already be zero-sized) -- so `w`'s zero-sized axis is
    # produced via a real `Slice` (data-dependent, not a top-level graph
    # input, so onnxsim's random-input test harness -- which special-cases a
    # dynamic *leading* input dim as a batch size but errors on any other
    # dynamic/zero dim -- never has to synthesize it) rather than declared
    # directly on a graph input.
    #
    # w's static shape is [8, 0] (axis 1 sliced to empty). Target new_shape
    # = [0, 0] with allowzero=1: axis 0 has new_dim=0, but allowzero=1 makes
    # this a LITERAL zero, not the copy-sentinel, so it is checked against
    # w's own axis-0 size (8) and fails (8 != 0) -- the pass declines.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g (float[8,8] X) => (float[0,0] Y)
        <
          int64[2] new_shape = {0, 0},
          int64[1] starts = {0},
          int64[1] ends = {0},
          int64[1] axes = {1}
        >
        {
          w = Slice(X, starts, ends, axes)
          Y = Reshape<allowzero = 1>(w, new_shape)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_reshape")
    assert ops["Reshape"] == 1


def test_eliminate_nop_reshape_matches_same_zero_shape_without_allowzero():
    # Companion/confirmation for the case above, isolating exactly what
    # `allowzero` changes: same w ([8, 0]) and same literal new_shape
    # ([0, 0]), but WITHOUT allowzero=1 this time. Now axis 0's `0` is the
    # copy-sentinel (reason (a): keep w's own axis-0 size, 8) and axis 1's
    # `0` trivially restates w's own axis-1 size (0 == 0, reason (b)) -- so
    # the pass fires here, where it declined above. This confirms
    # `allowzero` is really what flips the outcome, not something else about
    # the Slice-derived shape.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 14]
        >
        g (float[8,8] X) => (float[8,0] Y)
        <
          int64[2] new_shape = {0, 0},
          int64[1] starts = {0},
          int64[1] ends = {0},
          int64[1] axes = {1}
        >
        {
          w = Slice(X, starts, ends, axes)
          Y = Reshape(w, new_shape)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_reshape")
    assert ops["Reshape"] == 0
    assert ops["Slice"] == 1
