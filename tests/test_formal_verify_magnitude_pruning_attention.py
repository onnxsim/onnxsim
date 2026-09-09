"""Formal check for MagnitudePruningAttention (opt-in;
``onnxsim/passes/magnitude_pruning.h``): the data-free unstructured pruning
baseline (Han et al., 2015), restricted here to its ``com.microsoft``
Attention-family matcher (``Attention``/``DecoderMaskedSelfAttention``/
``PackedAttention``, merged QKV weight only) -- N:M (semi-structured) mode
and ``global_sparsity`` mode are OUT OF SCOPE for this file (see
``test_formal_verify_magnitude_pruning_matmul.py`` for those boundaries,
which apply identically here).

This file is deliberately a close mirror of
``test_formal_verify_magnitude_pruning_matmul.py`` and
``test_formal_verify_magnitude_pruning_conv.py`` -- read the MatMul file's
own module docstring first for the full rationale, which carries over
unchanged: magnitude pruning is a **selection-correctness** claim (the pass
correctly implements a specified top-k-by-magnitude masking per output
unit), not a value-preservation or bounded-error claim, since pruning
deliberately changes the output. ``MagnitudePruningAttention`` shares its
ENTIRE masking algorithm with ``MagnitudePruningMatMul``/
``MagnitudePruningConv`` -- all three dispatch through the exact same
``MagnitudePruningMaskRowMajor``/``SparsityMaskRowMajor``/
``MagnitudePruningWouldChange``/``ApplyMaskRowMajor`` machinery over a
row-major ``[rows, cols]`` view -- so this file reuses the MatMul proof
file's ``_spec`` predicate and both of its Z3 proofs (uniqueness given no
ties, kept-value preservation) verbatim in spirit: there is no new masking
algebra to formalize here.

**What genuinely differs for Attention is which axis of the merged QKV
weight becomes ``SparsityMaskRowMajor``'s "row", not the masking itself** --
this is the one Attention-specific wrinkle this file's differential tests
exist to pin down, and it is the OPPOSITE orientation from what a hasty
skim of ``ReadWeightNKF64``'s call site might suggest:

* The merged QKV weight ``w_t`` is constant, 2-D, shape ``[K, Nq+Nk+Nv]``
  (``dim0 = K``, ``dim1 = N = Nq+Nk+Nv``) -- "already ``[K, N]``-shaped by
  construction" per ``MagnitudePruningAttention``'s own comment, i.e. there
  is no ``weight_transposed`` concept here at all (unlike
  ``MagnitudePruningMatMul``, which must handle both layouts).
* Both ``patternMatchPredicate`` and ``runTransform`` call
  ``ReadWeightNKF64(*w_t, /*transposed=*/false)``. Re-reading
  ``ReadWeightNKF64`` itself (not just this call site) with
  ``transposed=false``: ``rows = dim1`` (``= N``), ``cols = dim0`` (``=
  K``), and the returned buffer is genuinely laid out row-major as
  ``[N, K]`` (output-unit-first) -- element ``(i, j)`` of the on-disk
  ``[K, N]`` tensor (``i`` in ``[0,K)``, ``j`` in ``[0,N)``) lands at
  ``w_nk[j * K + i]``, i.e. row ``j`` (one of the ``N`` merged QKV output
  units) collects that unit's own ``K`` input-side weights.
* Confirming against the actual call:
  ``MagnitudePruningWouldChange(w_nk, dim1, dim0)`` -- and identically in
  ``runTransform``, ``MagnitudePruningMaskRowMajor(w_nk, dim1, dim0)`` --
  passes ``rows=dim1=N``, ``cols=dim0=K``, which matches ``w_nk``'s own
  ``[N, K]`` layout exactly (no accidental transpose between how the buffer
  is laid out and how it is then interpreted).
* **So masking happens PER OUTPUT UNIT of the merged QKV weight** (one of
  the ``N = Nq+Nk+Nv`` merged Q/K/V output columns -- a "row" in
  ``SparsityMaskRowMajor``'s own sense, despite being a *column* of the
  on-disk ``[K, N]`` tensor), competing the ``K`` reduction-dimension
  entries feeding that unit against each other. This is EXACTLY the same
  orientation as ``MagnitudePruningMatMul``'s own default (non-transposed,
  ``[K, N]``) weight case -- Attention's merged QKV weight is read "exactly
  like a non-transposed MatMul weight" (the header's own words), and
  ``ReadWeightNKF64``'s ``rows=dim1, cols=dim0`` when ``transposed=false``
  is identical in both call sites. There is no row/column swap or surprise
  here despite this file's own construction warning to re-verify it
  independently -- confirmed empirically below against the real compiled
  pass's own output, not just by re-reading the C++.

The differential tests below build a small, hand-verifiable ``[4, 6]``
merged QKV weight (``K=4``, ``Nq=Nk=Nv=2``) by horizontally tiling
``test_formal_verify_magnitude_pruning_matmul.py``'s own ``[4, 2]`` ``_W``
three times (once per Q/K/V share) -- since the masking orientation is
confirmed identical to that file's own non-transposed MatMul case, the two
per-column masks it already independently verified (no ties at the cutoff)
carry over here unchanged, column-for-column, with no need to hand-pick a
fresh row.

Since ``MagnitudePruningSparsity()`` is a function-local C++ static, there
is no way to set it from Python except through the one real, documented
entry point that sets it as a side effect: ``PruneMagnitude``
(``pruning_entry.cpp``), exposed to Python as
:func:`onnxsim.prune_magnitude_cpp` -- see ``_set_sparsity``'s own doc
comment, copied verbatim (module name aside) from the MatMul/Conv proof
files.

**Model construction: ``onnx.parser``, not ``onnx.helper``.** Tested
empirically first (per this repo's own CLAUDE.md instructions): the ONNX
text format parser cleanly accepts a custom-domain node written as
``com.microsoft.Attention<attrs>(...)`` given an ``opset_import`` entry for
``"com.microsoft"`` -- this is exactly the pattern already used by
``tests/test_attention_head_pruning_cpp.py``'s own ``_attention_model`` and
``tests/test_dynamic_quantize_attention.py``'s own ``_attention_model``, so
there is no need to fall back to ``onnx.helper`` here.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

# --- The specification: SparsityMaskRowMajor's own two defining properties -
#
# Identical to test_formal_verify_magnitude_pruning_matmul.py's own _spec --
# MagnitudePruningAttention dispatches through the exact same
# MagnitudePruningMaskRowMajor/SparsityMaskRowMajor as
# MagnitudePruningMatMul/MagnitudePruningConv, so there is no
# Attention-specific masking algebra to state here. See that file's own
# module docstring for why this is deliberately looser than the
# std::stable_sort-based implementation (silent on tie-breaking).


def _spec(mask, values, keep):
    """Z3 formula for ``SparsityMaskRowMajor``'s own SPECIFICATION, not its
    stable-sort-based implementation: exactly ``keep`` of ``mask``'s entries
    are ``True`` (kept), and every kept entry's magnitude is ``>=`` every
    dropped entry's magnitude. ``values`` are ``|w|`` magnitudes (already
    ``fabs``'d, matching ``MagnitudePruningMaskRowMajor``'s own
    ``importance`` array, not raw signed weights) and ``mask`` is a list of
    Z3 Bools, one per column, both the same length. Deliberately silent on
    which of several EQUAL-magnitude entries gets dropped -- see this file's
    own module docstring for why that's not part of the correctness claim.
    """
    cols = len(values)
    count_kept = z3.Sum([z3.If(mask[i], 1, 0) for i in range(cols)])
    ordering = z3.And(
        *[
            z3.Implies(z3.And(mask[i], z3.Not(mask[j])), values[i] >= values[j])
            for i in range(cols)
            for j in range(cols)
            if i != j
        ]
    )
    return z3.And(count_kept == keep, ordering)


def test_magnitude_pruning_attention_selection_is_unique_given_no_ties():
    # The genuinely general claim: for a row of PAIRWISE-DISTINCT magnitudes
    # (no ties), _spec pins down a UNIQUE mask -- any two masks both
    # satisfying _spec for the same row/keep count are elementwise
    # identical. This is what makes "keep the highest-magnitude entries" a
    # well-defined target at all once ties are excluded (which the
    # differential tests below always ensure), rather than merely a
    # necessary-but-not-sufficient shape. cols=4 keeps this small enough
    # that Z3 (no quantifier alternation needed: keep/cols are fixed
    # concrete integers here, so this is an ordinary free-variable-universal
    # claim, this suite's usual prove() idiom) solves it in milliseconds.
    cols, keep = 4, 2
    values = z3.Reals(" ".join(f"v{i}" for i in range(cols)))
    mask_a = z3.Bools(" ".join(f"ka{i}" for i in range(cols)))
    mask_b = z3.Bools(" ".join(f"kb{i}" for i in range(cols)))

    no_ties = z3.And(
        *[values[i] != values[j] for i in range(cols) for j in range(i + 1, cols)]
    )
    both_satisfy_spec = z3.And(_spec(mask_a, values, keep), _spec(mask_b, values, keep))
    same_mask = z3.And(*[mask_a[i] == mask_b[i] for i in range(cols)])

    prove(z3.Implies(z3.And(no_ties, both_satisfy_spec), same_mask))


def test_magnitude_pruning_attention_masking_preserves_kept_values_exactly():
    # The clean, separate equality claim: ApplyMaskRowMajor's literal rule
    # is pruned[i] = mask[i] ? value[i] : 0 -- a kept entry (mask[i] ==
    # True) is untouched, not recomputed or rounded. An ordinary
    # unconditional tautology over a free (universally quantified via
    # prove()) mask bit and value -- unlike every claim above, this one
    # isn't about SELECTION correctness at all, just that masking doesn't
    # corrupt what it decides to keep.
    mask_i = z3.Bool("mask_i")
    value_i = z3.Real("value_i")
    pruned_i = z3.If(mask_i, value_i, z3.RealVal(0))
    prove(z3.Implies(mask_i, pruned_i == value_i))


# --- Positive/negative controls on one concrete row -------------------------
#
# This row is output unit 0's own K=4 input-side weights in the differential
# tests below -- column 0 of the merged QKV weight's on-disk [K, N] layout,
# which ReadWeightNKF64(..., transposed=false) turns into row 0 of the
# [N, K] w_nk view -- W[:, 0] == [3.0, -1.0, 0.5, -2.0], so the mask asserted
# here is independently cross-checked against the real compiled pass's own
# output in
# test_magnitude_pruning_attention_pass_fires_and_prunes_exact_entries. Taken
# verbatim from test_formal_verify_magnitude_pruning_matmul.py's own
# _ROW0_MAGNITUDES -- the underlying selection algorithm and orientation are
# identical (see this file's own module docstring), so there is no need for
# a fresh row here.

_ROW0_MAGNITUDES = [3.0, 1.0, 0.5, 2.0]  # |3.0|, |-1.0|, |0.5|, |-2.0|
_ROW0_KEEP = 2
# SparsityMaskRowMajor's own output for this row at keep=2: the two smallest
# magnitudes (0.5 at index 2, 1.0 at index 1) are dropped, no ties.
_ROW0_ALGORITHM_MASK = [True, False, False, True]


def test_magnitude_pruning_attention_algorithm_output_satisfies_spec():
    # Positive control: the mask SparsityMaskRowMajor actually computes for
    # this concrete row is a valid highest-magnitude selection -- Z3
    # confirms both of _spec's defining properties hold for this exact
    # concrete assignment (a ground/closed formula: prove() here amounts to
    # confirming it evaluates to True, done via Z3 rather than a bare
    # Python bool expression for consistency with the rest of this file).
    mask = [z3.BoolVal(v) for v in _ROW0_ALGORITHM_MASK]
    values = [z3.RealVal(v) for v in _ROW0_MAGNITUDES]
    prove(_spec(mask, values, _ROW0_KEEP))


def test_magnitude_pruning_attention_negative_control_wrong_selection_violates_spec():
    # Negative control: a mask that drops the row's SINGLE HIGHEST-magnitude
    # entry (index 0, |3.0|) while keeping a strictly lower-magnitude one
    # (index 2, |0.5|) is a genuinely wrong selection. _spec must not hold
    # for it -- confirming _spec has real teeth, not being vacuously true
    # for any mask with the right popcount.
    wrong_mask = [False, False, True, True]  # keeps |0.5| and |2.0|, drops |3.0|
    mask = [z3.BoolVal(v) for v in wrong_mask]
    values = [z3.RealVal(v) for v in _ROW0_MAGNITUDES]
    prove(z3.Not(_spec(mask, values, _ROW0_KEEP)))


# --- Differential tests against the real compiled pass -----------------------


def _model(body, initializer=(), opset=17, ir_version=10):
    # opset_import carries a "com.microsoft" entry alongside the usual ""
    # domain -- required for the parser to accept a
    # `com.microsoft.Attention<...>(...)` node (confirmed empirically per
    # this file's own module docstring), mirroring
    # tests/test_dynamic_quantize_attention.py's own _model and
    # tests/test_attention_head_pruning_cpp.py's own _attention_model.
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}, "com.microsoft": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


_THROWAWAY_SPARSITY_MODEL = _model(
    """
    g (float[b,2] X) => (float[b,2] Y)
    { Y = MatMul(X, W) }
    """,
    [_f32(np.array([[1.0, 2.0], [3.0, 4.0]]), "W")],
)


def _set_sparsity(sparsity):
    """Sets ``MagnitudePruningSparsity()``'s process-wide C++ static (see
    ``magnitude_pruning.h``'s own doc comment) to ``sparsity``, via the one
    real, supported way to do so from Python: :func:`onnxsim.prune_magnitude_cpp`
    (``PruneMagnitude``/``pruning_entry.cpp``), which sets it immediately
    before running ``OptimizeFixed`` -- there is no Python-visible way to
    poke this file-local static directly. Run here on a throwaway one-node
    MatMul model, distinct from whatever Attention model a test actually
    cares about, purely for this side effect: ``simplify_isolated_extra``'s
    own ``onnxsim.simplify()`` call has no parameter-passing path of its own
    for an opt-in pass's file-local static, so every test below calls this
    immediately before its own
    ``simplify_isolated_extra(..., "magnitude_pruning_attention", ...)``
    call, which is what actually exercises the real compiled pass under
    test.
    """
    onnxsim.prune_magnitude_cpp(_THROWAWAY_SPARSITY_MODEL, sparsity=sparsity)


def _pruned_weight(sim_model):
    """The pruned weight actually written by the pass, found via the real
    Attention node's own CURRENT weight input name -- mirrors
    ``tests/test_pruning_cpp.py``'s own ``_weight`` helper's doc comment:
    the pass leaves the original initializer dangling and rewires the node
    to a brand-new one (matching every other onnxsim rewrite's "replace,
    don't mutate" convention for constants), so the pruned tensor must be
    looked up by the node's current input name, not assumed to still be
    named "W".
    """
    node = producer(sim_model, "Y")
    w_name = node.input[1]
    init = next(t for t in sim_model.graph.initializer if t.name == w_name)
    return numpy_helper.to_array(init), node


# K=4 (attention input hidden size / reduction dim), Nq=Nk=Nv=2 (merged QKV
# width N=6, evenly 3-way split -- the schema-default split applies, so no
# explicit qkv_hidden_sizes attribute is needed), num_heads=2 (each Q/K/V
# hidden size of 2 divides evenly by 2, satisfying
# MatchAttentionQkvWeightOnly's own num_heads-divisibility guard). This is
# test_formal_verify_magnitude_pruning_matmul.py's own [4, 2] _W, tiled
# horizontally three times -- since this file's own module docstring
# confirms the masking orientation (per output unit, i.e. per on-disk
# COLUMN) is identical to that file's non-transposed MatMul case, each of
# the three Q/K/V shares gets an independent copy of the exact same two
# already-verified (no ties at the cutoff) columns, rather than a
# hand-picked fresh row.
_W_MATMUL_2COL = np.array(
    [
        [3.0, 0.2],
        [-1.0, 4.0],
        [0.5, -0.1],
        [-2.0, 1.0],
    ]
)
_W = np.tile(_W_MATMUL_2COL, (1, 3))
assert _W.shape == (4, 6)
# sparsity=0.5 -> keep = round(4 * 0.5) = 2 per output unit (column). Column
# pattern 0 (Q/K/V's own first column each) keeps rows {0, 3} (|3.0|,
# |-2.0|), drops {1, 2} (|-1.0|, |0.5|) -- matching _ROW0_ALGORITHM_MASK
# above exactly. Column pattern 1 (Q/K/V's own second column each) keeps
# {1, 3} (|4.0|, |1.0|), drops {0, 2} (|0.2|, |-0.1|).
_EXPECTED_PRUNED_W_2COL = np.array(
    [
        [3.0, 0.0],
        [0.0, 4.0],
        [0.0, 0.0],
        [-2.0, 1.0],
    ]
)
_EXPECTED_PRUNED_W = np.tile(_EXPECTED_PRUNED_W_2COL, (1, 3))


def _attention_model(w_array, w_name="W", op_type="Attention", num_heads=2):
    # X is rank-3 (batch, seq, K) per Attention's own schema; Y is rank-3
    # (batch, seq, Nv) -- Nv=2 here. Only the batch dim is left dynamic
    # ("batch", matching every other proof file's own dynamic-first-dim-only
    # convention, e.g. test_formal_verify_magnitude_pruning_matmul.py's own
    # `float[b,4] X`) -- onnxsim's own model_checking.generate_rand_input
    # requires every OTHER dim to be statically known to synthesize a random
    # input, so `seq` is fixed at 3 rather than left symbolic. Bias/
    # mask_index/past/attention_bias are all omitted (trailing optional
    # inputs may simply not be spelled out, per
    # tests/test_attention_head_pruning_cpp.py's own _attention_model
    # comment) -- MatchAttentionQkvWeightOnly only requires 2 inputs.
    return _model(
        """
        g (float[batch,3,4] X) => (float[batch,3,2] Y)
        { Y = com.microsoft.%s <num_heads=%d> (X, %s) }
        """
        % (op_type, num_heads, w_name),
        [_f32(w_array, w_name)],
    )


def test_magnitude_pruning_attention_pass_fires_and_prunes_exact_entries():
    _set_sparsity(0.5)
    model = _attention_model(_W)

    # check_n=0: this pass deliberately changes the computed output (that's
    # the whole point of pruning), so onnxsim's own random-input equivalence
    # check (which expects the rewrite to PRESERVE the function) does not
    # apply here -- mirrors every other genuinely lossy rewrite this suite
    # tests (e.g. test_formal_verify_magnitude_pruning_matmul.py's own
    # check_n=0 firing test).
    sim_model, ops = simplify_isolated_extra(
        model, "magnitude_pruning_attention", check_n=0
    )
    assert ops["Attention"] == 1

    w_pruned, _node = _pruned_weight(sim_model)
    np.testing.assert_array_equal(w_pruned, _EXPECTED_PRUNED_W)

    # Every dropped entry is EXACTLY zero, and every KEPT entry is
    # byte-identical to its ORIGINAL value (not merely numerically close) --
    # the "masking never modifies a surviving entry" property, checked here
    # directly against the real compiled pass's own output, separately from
    # the clean Z3 lemma proved above.
    kept = _EXPECTED_PRUNED_W != 0
    np.testing.assert_array_equal(w_pruned[kept], _W[kept])
    assert np.count_nonzero(w_pruned) == 12  # 2 kept per column * 6 columns


def test_magnitude_pruning_attention_declines_when_already_at_target_sparsity():
    # MagnitudePruningWouldChange's own idempotency guard: using this file's
    # own ALREADY-pruned _EXPECTED_PRUNED_W as the INPUT weight, at
    # sparsity=0.5 the very same two positions per output unit would be
    # "dropped" again -- but both are already exactly zero, so the pass
    # must be a strict no-op. This is the real, testable behavior the
    # guard exists for (see magnitude_pruning.h's own top comment): without
    # it, OptimizeFixed's fixed-point re-application would loop forever
    # re-"pruning" an already-pruned weight. Confirmed via the
    # initializer's own NAME staying "W" (mirrors
    # test_formal_verify_magnitude_pruning_matmul.py's own "already at
    # target sparsity" test), not just its value -- a byte-identical
    # REPLACEMENT initializer would still pass a value-only check but would
    # NOT be the "no change at all" the guard promises.
    # check_n=0 on every differential test below (not just the "would
    # change the output" firing test, unlike the MatMul/Conv proof files'
    # own "declines" tests, which keep the default check_n and let it pass
    # trivially on an unchanged model): onnxsim's own equivalence check
    # would actually execute this degenerate (num_heads=2, head_size=1,
    # no bias/mask) Attention node through onnxruntime, which is observed
    # to crash the process outright (a real onnxruntime-side issue with
    # this shape, unrelated to the pass under test) rather than merely
    # fail -- so every test in this file sidesteps onnxruntime execution
    # entirely and relies solely on inspecting the pass's own structural
    # output, which is what these tests actually care about anyway.
    _set_sparsity(0.5)
    model = _attention_model(_EXPECTED_PRUNED_W)

    sim_model, ops = simplify_isolated_extra(
        model, "magnitude_pruning_attention", check_n=0
    )
    assert ops["Attention"] == 1
    node = producer(sim_model, "Y")
    assert node.input[1] == "W"
    w = next(t for t in sim_model.graph.initializer if t.name == "W")
    np.testing.assert_array_equal(numpy_helper.to_array(w), _EXPECTED_PRUNED_W)


def test_magnitude_pruning_attention_declines_non_constant_weight():
    # A weight that is a graph INPUT rather than a constant initializer can
    # never be matched: FetchConstantTensor returns null, so
    # MatchAttentionQkvWeightOnly declines before SparsityMaskRowMajor ever
    # runs -- the simplest of MatchAttentionQkvWeightOnly's several guards
    # to exercise (see test_formal_verify_magnitude_pruning_matmul.py's own
    # module docstring for why this suite doesn't exhaustively cover every
    # one).
    _set_sparsity(0.5)
    model = _model(
        """
        g (float[batch,3,4] X, float[4,6] W) => (float[batch,3,2] Y)
        { Y = com.microsoft.Attention <num_heads=2> (X, W) }
        """
    )

    sim_model, ops = simplify_isolated_extra(
        model, "magnitude_pruning_attention", check_n=0
    )
    assert ops["Attention"] == 1
    node = producer(sim_model, "Y")
    assert list(node.input) == ["X", "W"]


def test_magnitude_pruning_attention_declines_when_num_heads_does_not_divide_qkv():
    # MatchAttentionQkvWeightOnly's own num_heads-divisibility guard: with
    # the schema-default even 3-way split (no qkv_hidden_sizes attribute),
    # Nq=Nk=Nv=2 here -- num_heads=4 fails `nq % num_heads == 0` (2 % 4 !=
    # 0), so the matcher declines outright regardless of the weight's own
    # values. This is a genuinely different guard than the "missing
    # num_heads entirely" case, which the ONNX Runtime contrib schema
    # itself already rejects as a malformed node before onnxsim ever sees
    # it (num_heads is a REQUIRED attribute on Attention's own schema) --
    # so this is the smallest num_heads-related decline this suite can
    # actually construct a loadable model for.
    _set_sparsity(0.5)
    model = _attention_model(_W, num_heads=4)

    sim_model, ops = simplify_isolated_extra(
        model, "magnitude_pruning_attention", check_n=0
    )
    assert ops["Attention"] == 1
    node = producer(sim_model, "Y")
    assert node.input[1] == "W"
    w = next(t for t in sim_model.graph.initializer if t.name == "W")
    np.testing.assert_array_equal(numpy_helper.to_array(w), _W.astype(np.float32))


def test_magnitude_pruning_attention_declines_when_keep_covers_all_columns():
    # SparsityMaskRowMajor's own "keep >= cols" branch: at a low enough
    # sparsity, round(cols * (1 - sparsity)) reaches (or exceeds) cols
    # itself, so the mask keeps every entry -- combined with
    # MagnitudePruningWouldChange's own "no entry would actually be zeroed"
    # check, the pass must decline outright regardless of the weight's own
    # values. cols=K=4, sparsity=0.1: round(4 * 0.9) == round(3.6) == 4 ==
    # cols.
    _set_sparsity(0.1)
    model = _attention_model(_W)

    sim_model, ops = simplify_isolated_extra(
        model, "magnitude_pruning_attention", check_n=0
    )
    assert ops["Attention"] == 1
    node = producer(sim_model, "Y")
    assert node.input[1] == "W"
    w = next(t for t in sim_model.graph.initializer if t.name == "W")
    np.testing.assert_array_equal(numpy_helper.to_array(w), _W.astype(np.float32))


def test_magnitude_pruning_attention_fires_on_packed_attention_op_type():
    # IsMicrosoftAttentionOp's own multi-op_type matching: PackedAttention
    # is one of the three op_types MatchAttentionQkvWeightOnly accepts (see
    # this file's own module docstring and magnitude_pruning.h's own
    # IsMicrosoftAttentionOp), sharing the same weight/bias input positions
    # as plain Attention with no extra guards of its own (unlike
    # DecoderMaskedSelfAttention's do_rotary/qkv_hidden_sizes/past checks) --
    # a minimal 2-input PackedAttention node exercises this with the exact
    # same weight/expected-output pair as the plain-Attention firing test
    # above, confirming the op_type dispatch itself, not a new masking
    # orientation.
    _set_sparsity(0.5)
    model = _attention_model(_W, op_type="PackedAttention")

    sim_model, ops = simplify_isolated_extra(
        model, "magnitude_pruning_attention", check_n=0
    )
    assert ops["PackedAttention"] == 1

    w_pruned, _node = _pruned_weight(sim_model)
    np.testing.assert_array_equal(w_pruned, _EXPECTED_PRUNED_W)
