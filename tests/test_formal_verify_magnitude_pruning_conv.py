"""Formal check for MagnitudePruningConv (opt-in;
``onnxsim/passes/magnitude_pruning.h``): the data-free unstructured pruning
baseline (Han et al., 2015), restricted here to its Conv matcher -- N:M
(semi-structured) mode, the Attention variant, and ``global_sparsity`` mode
are OUT OF SCOPE for this file (see ``test_formal_verify_magnitude_pruning_matmul.py``
for those boundaries, which apply identically here).

This file is deliberately a close mirror of
``test_formal_verify_magnitude_pruning_matmul.py`` -- read that file's own
module docstring first for the full rationale, which carries over unchanged:
magnitude pruning is a **selection-correctness** claim (the pass correctly
implements a specified top-k-by-magnitude masking per output channel), not a
value-preservation or bounded-error claim, since pruning deliberately changes
the output. ``MagnitudePruningConv`` shares its ENTIRE masking algorithm with
``MagnitudePruningMatMul`` -- both dispatch through the exact same
``MagnitudePruningMaskRowMajor``/``SparsityMaskRowMajor``/
``MagnitudePruningWouldChange``/``ApplyMaskRowMajor`` machinery over a
row-major ``[rows, cols]`` view -- so this file reuses the MatMul proof
file's ``_spec`` predicate and both of its Z3 proofs (uniqueness given no
ties, kept-value preservation) verbatim in spirit: there is no new masking
algebra to formalize here.

**What genuinely differs for Conv is the RESHAPE into that ``[rows, cols]``
view, not the masking itself** -- this is the one Conv-specific wrinkle this
file's differential tests exist to pin down:

* Conv's weight layout is ``[out_channels, in_channels/groups, kH, kW]``
  (rank exactly 4, checked by ``MagnitudePruningConv::patternMatchPredicate``/
  ``runTransform`` alike), which ALWAYS puts the output channel on axis 0 --
  unlike MatMul/Gemm, whose weight can be stored transposed
  (``info.weight_transposed``) and therefore needs ``ReadWeightNKF64`` to
  relayout it to an output-channel-first view. Conv's own matcher does no
  such relayout: it reads the tensor flat, as-is, via ``ReadTensorAsF64Flat``
  (NOT ``ReadWeightNKF64``, which is MatMul-specific and carries transpose
  logic Conv has no use for), then treats ``rows = out_channels`` (axis 0)
  and ``cols = in_channels/groups * kH * kW`` (every other axis flattened
  together) purely by arithmetic on the sizes -- no data movement, since a
  standard row-major tensor with out_channels as its leading axis is ALREADY
  exactly the ``[rows, cols]`` view ``SparsityMaskRowMajor`` wants, once the
  trailing axes are flattened. So a ``[2, 3, 2, 2]`` Conv weight reshapes to
  ``rows=2, cols=3*2*2=12`` for masking purposes with no permutation at all
  -- confirmed by re-reading ``MagnitudePruningConv`` in
  ``magnitude_pruning.h`` directly (both ``patternMatchPredicate`` and
  ``runTransform`` compute ``cols`` via ``for (size_t i = 1; ...) cols *=
  w_t->sizes()[i];`` over the SAME flat buffer ``ReadTensorAsF64Flat``
  returns).
* This is a genuine simplification over the MatMul version: there is no
  "weight_transposed" case to handle, so Conv's matcher/writer pair is
  correspondingly shorter (no ``ReadWeightNKF64``/``WeightNkToOriginalF64``
  round-trip -- ``WriteF64FlatAsTensor`` is called directly on the masked
  flat buffer).

The differential tests below are what actually exercise this reshape:
a small, hand-verifiable ``[2, 1, 2, 2]`` Conv weight (2 output channels, 1
input channel, a 2x2 kernel -- so each output channel's own flattened weight
has exactly 4 entries, ``cols=4``) with per-output-channel magnitudes chosen
so sparsity=0.5 (keep 2 of 4 per output channel) has no ties at the cutoff;
getting the reshape wrong (e.g. accidentally interleaving the two output
channels' entries, or transposing kH/kW) would make the hand-computed
expected pruned values below wrong in a way that could still confusingly
pass or fail, so channel 0's magnitudes are deliberately taken from
``test_formal_verify_magnitude_pruning_matmul.py``'s own ``_ROW0_MAGNITUDES``
row (independently cross-checked there against the same algorithm) and
channel 1's from that file's own column-1 row, rather than picked fresh.

Since ``MagnitudePruningSparsity()`` is a function-local C++ static, there is
no way to set it from Python except through the one real, documented entry
point that sets it as a side effect: ``PruneMagnitude`` (``pruning_entry.cpp``),
exposed to Python as :func:`onnxsim.prune_magnitude_cpp` -- see
``_set_sparsity``'s own doc comment, copied verbatim (module name aside) from
the MatMul proof file.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

# --- The specification: SparsityMaskRowMajor's own two defining properties -
#
# Identical to test_formal_verify_magnitude_pruning_matmul.py's own _spec --
# MagnitudePruningConv dispatches through the exact same
# MagnitudePruningMaskRowMajor/SparsityMaskRowMajor as MagnitudePruningMatMul,
# so there is no Conv-specific masking algebra to state here. See that file's
# own module docstring for why this is deliberately looser than the
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


def test_magnitude_pruning_conv_selection_is_unique_given_no_ties():
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


def test_magnitude_pruning_conv_masking_preserves_kept_values_exactly():
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
# This row is output channel 0's own flattened weight in the differential
# tests below: W[0].flatten() == [3.0, -1.0, 0.5, -2.0] (Cin/groups * kH * kW
# == 1*2*2 == 4 flattened entries feeding output channel 0), so the mask
# asserted here is independently cross-checked against the real compiled
# pass's own output in
# test_magnitude_pruning_conv_pass_fires_and_prunes_exact_entries. Taken
# verbatim from test_formal_verify_magnitude_pruning_matmul.py's own
# _ROW0_MAGNITUDES -- the underlying selection algorithm is identical, so
# there is no need for a fresh row here.

_ROW0_MAGNITUDES = [3.0, 1.0, 0.5, 2.0]  # |3.0|, |-1.0|, |0.5|, |-2.0|
_ROW0_KEEP = 2
# SparsityMaskRowMajor's own output for this row at keep=2: the two smallest
# magnitudes (0.5 at index 2, 1.0 at index 1) are dropped, no ties.
_ROW0_ALGORITHM_MASK = [True, False, False, True]


def test_magnitude_pruning_conv_algorithm_output_satisfies_spec():
    # Positive control: the mask SparsityMaskRowMajor actually computes for
    # this concrete row is a valid highest-magnitude selection -- Z3
    # confirms both of _spec's defining properties hold for this exact
    # concrete assignment (a ground/closed formula: prove() here amounts to
    # confirming it evaluates to True, done via Z3 rather than a bare
    # Python bool expression for consistency with the rest of this file).
    mask = [z3.BoolVal(v) for v in _ROW0_ALGORITHM_MASK]
    values = [z3.RealVal(v) for v in _ROW0_MAGNITUDES]
    prove(_spec(mask, values, _ROW0_KEEP))


def test_magnitude_pruning_conv_negative_control_wrong_selection_violates_spec():
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


def _model(body, initializer=(), opset=13, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
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
    MatMul model, distinct from whatever Conv model a test actually cares
    about, purely for this side effect: ``simplify_isolated_extra``'s own
    ``onnxsim.simplify()`` call has no parameter-passing path of its own for
    an opt-in pass's file-local static, so every test below calls this
    immediately before its own
    ``simplify_isolated_extra(..., "magnitude_pruning_conv", ...)`` call,
    which is what actually exercises the real compiled pass under test.
    """
    onnxsim.prune_magnitude_cpp(_THROWAWAY_SPARSITY_MODEL, sparsity=sparsity)


def _pruned_weight(sim_model):
    """The pruned weight actually written by the pass, found via the real
    Conv node's own CURRENT weight input name -- mirrors
    ``tests/test_pruning_cpp.py``'s own ``_weight`` helper's doc comment:
    the pass leaves the original initializer dangling and rewires the node
    to a brand-new one (matching every other onnxsim rewrite's "replace,
    don't mutate" convention for constants), so the pruned tensor must be
    looked up by the node's current input name, not assumed to still be
    named "W1".
    """
    node = producer(sim_model, "Y")
    w_name = node.input[1]
    init = next(t for t in sim_model.graph.initializer if t.name == w_name)
    return numpy_helper.to_array(init), node


# Cout=2, Cin/groups=1, kH=2, kW=2 -- rows="output channels"=2,
# cols="flattened rest"=1*2*2=4 in SparsityMaskRowMajor's own [rows, cols]
# view. Output channel 0's flattened weight is exactly _ROW0_MAGNITUDES's
# signed source above; output channel 1's is a second, independently-
# unambiguous (no ties at the cutoff) row -- both taken verbatim from
# test_formal_verify_magnitude_pruning_matmul.py's own _W (its column 0 and
# column 1 respectively), since Conv's own reshape (see this file's own
# module docstring) makes each output channel's flattened row exactly what
# MatMul's own column was there. No entry is zero to begin with, so this
# also exercises a genuine, non-trivial prune.
_W = np.array(
    [
        [[[3.0, -1.0], [0.5, -2.0]]],  # output channel 0, flattened: row0
        [[[0.2, 4.0], [-0.1, 1.0]]],  # output channel 1
    ]
)
assert _W.shape == (2, 1, 2, 2)
# sparsity=0.5 -> keep = round(4 * 0.5) = 2 per output channel. Channel 0
# keeps flattened indices {0, 3} (|3.0|, |-2.0|), drops {1, 2} (|-1.0|,
# |0.5|) -- matching _ROW0_ALGORITHM_MASK above exactly. Channel 1 keeps
# {1, 3} (|4.0|, |1.0|), drops {0, 2} (|0.2|, |-0.1|).
_EXPECTED_PRUNED_W = np.array(
    [
        [[[3.0, 0.0], [0.0, -2.0]]],
        [[[0.0, 4.0], [0.0, 1.0]]],
    ]
)


def _conv_model(w_array, w_name="W1"):
    # spatial=3, kernel_shape=[2,2] -> out_spatial = 3-2+1 = 2. group=1,
    # Cin=Cin_per_group=1.
    return _model(
        """
        g (float[N,1,3,3] X) => (float[N,2,2,2] Y)
        { Y = Conv<kernel_shape=[2,2]>(X, %s) }
        """
        % w_name,
        [_f32(w_array, w_name)],
    )


def test_magnitude_pruning_conv_pass_fires_and_prunes_exact_entries():
    _set_sparsity(0.5)
    model = _conv_model(_W)

    # check_n=0: this pass deliberately changes the computed output (that's
    # the whole point of pruning), so onnxsim's own random-input equivalence
    # check (which expects the rewrite to PRESERVE the function) does not
    # apply here -- mirrors every other genuinely lossy rewrite this suite
    # tests (e.g. test_formal_verify_magnitude_pruning_matmul.py's own
    # check_n=0 firing test).
    sim_model, ops = simplify_isolated_extra(model, "magnitude_pruning_conv", check_n=0)
    assert ops["Conv"] == 1

    w_pruned, _node = _pruned_weight(sim_model)
    np.testing.assert_array_equal(w_pruned, _EXPECTED_PRUNED_W)

    # Every dropped entry is EXACTLY zero, and every KEPT entry is
    # byte-identical to its ORIGINAL value (not merely numerically close) --
    # the "masking never modifies a surviving entry" property, checked here
    # directly against the real compiled pass's own output, separately from
    # the clean Z3 lemma proved above.
    kept = _EXPECTED_PRUNED_W != 0
    np.testing.assert_array_equal(w_pruned[kept], _W[kept])
    assert np.count_nonzero(w_pruned) == 4  # 2 kept per channel * 2 channels


def test_magnitude_pruning_conv_declines_when_already_at_target_sparsity():
    # MagnitudePruningWouldChange's own idempotency guard: using this file's
    # own ALREADY-pruned _EXPECTED_PRUNED_W as the INPUT weight, at
    # sparsity=0.5 the very same two positions per output channel would be
    # "dropped" again -- but both are already exactly zero, so the pass
    # must be a strict no-op. This is the real, testable behavior the
    # guard exists for (see magnitude_pruning.h's own top comment): without
    # it, OptimizeFixed's fixed-point re-application would loop forever
    # re-"pruning" an already-pruned weight. Confirmed via the
    # initializer's own NAME staying "W1" (mirrors
    # test_formal_verify_magnitude_pruning_matmul.py's own "already at
    # target sparsity" test, itself mirroring
    # test_formal_verify_cross_layer_equalization.py's own "already
    # balanced reports no change" test), not just its value -- a
    # byte-identical REPLACEMENT initializer would still pass a
    # value-only check but would NOT be the "no change at all" the guard
    # promises.
    _set_sparsity(0.5)
    model = _conv_model(_EXPECTED_PRUNED_W)

    sim_model, ops = simplify_isolated_extra(model, "magnitude_pruning_conv")
    assert ops["Conv"] == 1
    node = producer(sim_model, "Y")
    assert node.input[1] == "W1"
    w = next(t for t in sim_model.graph.initializer if t.name == "W1")
    np.testing.assert_array_equal(numpy_helper.to_array(w), _EXPECTED_PRUNED_W)


def test_magnitude_pruning_conv_declines_non_constant_weight():
    # A weight that is a graph INPUT rather than a constant initializer can
    # never be matched: FetchConstantTensor returns null, so
    # MagnitudePruningConv::patternMatchPredicate declines before
    # SparsityMaskRowMajor ever runs -- the simplest of MatchConv's own
    # guards to exercise (see this file's own module docstring, and
    # test_formal_verify_magnitude_pruning_matmul.py's own module docstring,
    # for why this suite doesn't exhaustively cover every one).
    _set_sparsity(0.5)
    model = _model(
        """
        g (float[N,1,3,3] X, float[2,1,2,2] W1) => (float[N,2,2,2] Y)
        { Y = Conv<kernel_shape=[2,2]>(X, W1) }
        """
    )

    sim_model, ops = simplify_isolated_extra(model, "magnitude_pruning_conv")
    assert ops["Conv"] == 1
    node = producer(sim_model, "Y")
    assert list(node.input) == ["X", "W1"]


def test_magnitude_pruning_conv_declines_when_keep_covers_all_columns():
    # SparsityMaskRowMajor's own "keep >= cols" branch: at a low enough
    # sparsity, round(cols * (1 - sparsity)) reaches (or exceeds) cols
    # itself, so the mask keeps every entry -- combined with
    # MagnitudePruningWouldChange's own "no entry would actually be zeroed"
    # check, the pass must decline outright regardless of the weight's own
    # values. cols=4, sparsity=0.1: round(4 * 0.9) == round(3.6) == 4 ==
    # cols.
    _set_sparsity(0.1)
    model = _conv_model(_W)

    sim_model, ops = simplify_isolated_extra(model, "magnitude_pruning_conv")
    assert ops["Conv"] == 1
    node = producer(sim_model, "Y")
    assert node.input[1] == "W1"
    w = next(t for t in sim_model.graph.initializer if t.name == "W1")
    np.testing.assert_array_equal(numpy_helper.to_array(w), _W.astype(np.float32))
