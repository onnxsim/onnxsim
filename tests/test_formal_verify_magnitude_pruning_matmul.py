"""Formal check for MagnitudePruningMatMul (opt-in;
``onnxsim/passes/magnitude_pruning.h``): the data-free unstructured pruning
baseline (Han et al., 2015), restricted here to its MatMul/vanilla-Gemm
matcher -- N:M (semi-structured) mode, the Attention variant, and
``global_sparsity`` mode are OUT OF SCOPE for this file.

**This pass is a fundamentally different KIND of thing to formally verify
than every other pass in this suite.** Every other ``test_formal_verify_*.py``
file proves that a rewrite either preserves the computed function exactly (an
equivalence claim, e.g. ``test_formal_verify_cross_layer_equalization.py``)
or bounds how much it can change it (a numerical-error claim, e.g.
``test_formal_verify_quantized_mac_bound.py``). Magnitude pruning does
neither: it *deliberately* zeros out weight entries, which is not a
value-preserving rewrite at all, and the pass makes no numeric-closeness
claim about its own output either (a 50%-sparsified layer can change its
output arbitrarily much). So "soundness" here cannot mean "the output is
unchanged" or "the output error is boundable" -- there is no such property to
prove, and pretending otherwise would misrepresent what this pass actually
promises.

What the pass DOES promise, and what this file formalizes instead, is a
**selection/masking correctness claim**: given a target sparsity ratio, the
pass correctly implements its OWN specified per-row selection algorithm --
it zeros EXACTLY the lowest-magnitude entries the specification calls for,
and leaves every other entry's value completely untouched. This is much
closer in spirit to verifying a small sorting/top-k procedure than to
verifying a numerical rewrite, so it gets a correspondingly different Z3
treatment:

* ``_spec`` states ``SparsityMaskRowMajor``'s own specification as a
  predicate over an abstract Bool mask and Real magnitude array -- "exactly
  ``keep`` entries are kept" and "every kept entry's magnitude is >= every
  dropped entry's magnitude" -- WITHOUT reference to stable-sort or any
  particular tie-breaking rule. This is deliberately a *looser* spec than
  the algorithm's own implementation (which additionally pins down exactly
  which of several equal-magnitude entries get dropped, via
  ``std::stable_sort``'s stability): the algorithm's actual tie-breaking
  choice is not part of what makes a masking "correct" magnitude pruning
  (the header's own comment concurs: "both are equally valid magnitude-
  pruning outcomes"), so a formal spec that hard-codes one particular
  tie-break would be proving a stronger, less faithful claim than the pass
  actually needs to satisfy. Concrete test rows are deliberately chosen with
  NO ties at the cutoff, so the algorithm's actual output is checked against
  ``_spec`` unambiguously.
* Because this is fundamentally a finite selection problem, not an algebraic
  identity, the two most important checks below are Z3 ``Solver`` checks on
  small CONCRETE numeric instances (one row of 4 magnitudes) rather than a
  single symbolic ``ForAll``-style proof over an entire row: (1) a positive
  control confirming the mask ``SparsityMaskRowMajor`` actually computes for
  that row satisfies ``_spec``, and (2) a negative control confirming a
  genuinely wrong selection (dropping the row's highest-magnitude entry
  while keeping a lower one) does NOT satisfy ``_spec`` -- i.e. ``_spec``
  has real teeth, not vacuous ones.
* One genuinely general (symbolic, free-variable-universal via ``prove()``)
  claim IS proved: given a row of pairwise-DISTINCT magnitudes (no ties),
  ``_spec`` pins down a UNIQUE mask -- any two masks both satisfying
  ``_spec`` for the same row and ``keep`` count are elementwise identical.
  This is the sense in which the algorithm's selection is well-defined at
  all (there is nothing left for ``std::stable_sort``'s own tie-breaking to
  decide, once ties are excluded) -- a real combinatorial fact about
  ``_spec``, provable outright by Z3 for a small concrete column count
  (no quantifier alternation needed: the claim is already
  quantifier-free once ``keep``/``cols`` are fixed integers, so ordinary
  free-variable universals suffice, this suite's usual ``prove()`` idiom).
* Separately, and much more in this suite's usual style, one CLEAN equality
  claim is proved: a kept entry's value is completely untouched by masking
  (``ApplyMaskRowMajor``'s literal rule, ``pruned[i] = mask[i] ? value[i] :
  0``) -- an ordinary, unconditional tautology, not a selection-correctness
  claim at all. This is the one part of this pass's correctness that DOES
  look like the rest of this suite's "some specific value is provably
  unchanged" flavor, so it is proved on its own, distinct from the harder
  selection-correctness content above.

Differential tests build a small, hand-verifiable 4x2 ``MatMul`` weight via
``onnx.parser.parse_model()`` (per this repo's own CLAUDE.md) with per-column
magnitudes chosen so the sparsity=0.5 (keep 2 of 4 per output channel)
top-2 selection has no ties at the cutoff, and confirm: the real compiled
pass zeros exactly the two lowest-magnitude entries of each column, leaving
every other entry byte-identical to its original value; a weight that
ALREADY matches its own target sparsity pattern is left completely alone,
including its initializer's own NAME (``MagnitudePruningWouldChange``'s
idempotency guard -- the same "already balanced reports no change" style
check ``test_formal_verify_cross_layer_equalization.py`` uses); a
non-constant (graph-input) weight is declined outright; and the
``keep >= cols`` edge case (sparsity low enough that nothing would be
dropped) is declined too.

Since ``MagnitudePruningSparsity()`` is a function-local C++ static (the
same ``QuantizeFp16KeepIoTypes()``-style pattern ``quantize_fp16.h`` uses --
see ``magnitude_pruning.h``'s own doc comment), there is no way to set it
from Python except through the one real, documented entry point that sets
it as a side effect: ``PruneMagnitude`` (``pruning_entry.cpp``), exposed to
Python as :func:`onnxsim.prune_magnitude_cpp`. Every differential test below
calls :func:`onnxsim.prune_magnitude_cpp` on a throwaway model (purely for
that side effect) immediately before its own
``simplify_isolated_extra(..., "magnitude_pruning_matmul", ...)`` call, which
is what then actually runs the isolated pass under test and reads back its
own compiled behavior via ``producer`` -- see ``_set_sparsity``'s own doc
comment.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

# --- The specification: SparsityMaskRowMajor's own two defining properties -


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


def test_magnitude_pruning_matmul_selection_is_unique_given_no_ties():
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


def test_magnitude_pruning_matmul_masking_preserves_kept_values_exactly():
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
# This row is column 0 of _W in the differential tests below: W[:, 0] ==
# [3.0, -1.0, 0.5, -2.0] (K=4 input rows feeding output channel 0), so the
# mask asserted here is independently cross-checked against the real
# compiled pass's own output in
# test_magnitude_pruning_matmul_pass_fires_and_prunes_exact_entries.

_ROW0_MAGNITUDES = [3.0, 1.0, 0.5, 2.0]  # |3.0|, |-1.0|, |0.5|, |-2.0|
_ROW0_KEEP = 2
# SparsityMaskRowMajor's own output for this row at keep=2: the two smallest
# magnitudes (0.5 at index 2, 1.0 at index 1) are dropped, no ties.
_ROW0_ALGORITHM_MASK = [True, False, False, True]


def test_magnitude_pruning_matmul_algorithm_output_satisfies_spec():
    # Positive control: the mask SparsityMaskRowMajor actually computes for
    # this concrete row is a valid highest-magnitude selection -- Z3
    # confirms both of _spec's defining properties hold for this exact
    # concrete assignment (a ground/closed formula: prove() here amounts to
    # confirming it evaluates to True, done via Z3 rather than a bare
    # Python bool expression for consistency with the rest of this file).
    mask = [z3.BoolVal(v) for v in _ROW0_ALGORITHM_MASK]
    values = [z3.RealVal(v) for v in _ROW0_MAGNITUDES]
    prove(_spec(mask, values, _ROW0_KEEP))


def test_magnitude_pruning_matmul_negative_control_wrong_selection_violates_spec():
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
    model, distinct from whatever model a test actually cares about, purely
    for this side effect: ``simplify_isolated_extra``'s own
    ``onnxsim.simplify()`` call has no parameter-passing path of its own for
    an opt-in pass's file-local static, so every test below calls this
    immediately before its own
    ``simplify_isolated_extra(..., "magnitude_pruning_matmul", ...)`` call,
    which is what actually exercises the real compiled pass under test.
    """
    onnxsim.prune_magnitude_cpp(_THROWAWAY_SPARSITY_MODEL, sparsity=sparsity)


def _pruned_weight(sim_model):
    """The pruned weight actually written by the pass, found via the real
    MatMul node's own CURRENT weight input name -- mirrors
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


# K=4 (input channels), N=2 (output channels) -- rows="output channels"=2,
# cols="input channels"=4 in SparsityMaskRowMajor's own [rows, cols] view.
# Column 0 is exactly _ROW0_MAGNITUDES's signed source above; column 1 is a
# second, independently-unambiguous (no ties at the cutoff) row. No entry is
# zero to begin with, so this also exercises a genuine, non-trivial prune.
_W = np.array(
    [
        [3.0, 0.2],
        [-1.0, 4.0],
        [0.5, -0.1],
        [-2.0, 1.0],
    ]
)
# sparsity=0.5 -> keep = round(4 * 0.5) = 2 per column. Column 0 keeps
# indices {0, 3} (|3.0|, |-2.0|), drops {1, 2} (|-1.0|, |0.5|) -- matching
# _ROW0_ALGORITHM_MASK above exactly. Column 1 keeps {1, 3} (|4.0|, |1.0|),
# drops {0, 2} (|0.2|, |-0.1|).
_EXPECTED_PRUNED_W = np.array(
    [
        [3.0, 0.0],
        [0.0, 4.0],
        [0.0, 0.0],
        [-2.0, 1.0],
    ]
)


def test_magnitude_pruning_matmul_pass_fires_and_prunes_exact_entries():
    _set_sparsity(0.5)
    model = _model(
        """
        g (float[b,4] X) => (float[b,2] Y)
        { Y = MatMul(X, W) }
        """,
        [_f32(_W, "W")],
    )

    # check_n=0: this pass deliberately changes the computed output (that's
    # the whole point of pruning), so onnxsim's own random-input equivalence
    # check (which expects the rewrite to PRESERVE the function) does not
    # apply here -- mirrors every other genuinely lossy rewrite this suite
    # tests (e.g. test_formal_verify_dynamic_quantize_matmul.py's own
    # check_n=0 firing tests).
    sim_model, ops = simplify_isolated_extra(
        model, "magnitude_pruning_matmul", check_n=0
    )
    assert ops["MatMul"] == 1

    w_pruned, _node = _pruned_weight(sim_model)
    np.testing.assert_array_equal(w_pruned, _EXPECTED_PRUNED_W)

    # Every dropped entry is EXACTLY zero, and every KEPT entry is
    # byte-identical to its ORIGINAL value (not merely numerically close) --
    # the "masking never modifies a surviving entry" property, checked here
    # directly against the real compiled pass's own output, separately from
    # the clean Z3 lemma proved above.
    kept = _EXPECTED_PRUNED_W != 0
    np.testing.assert_array_equal(w_pruned[kept], _W[kept])
    assert np.count_nonzero(w_pruned) == 4  # 2 kept per column * 2 columns


def test_magnitude_pruning_matmul_declines_when_already_at_target_sparsity():
    # MagnitudePruningWouldChange's own idempotency guard: using this file's
    # own ALREADY-pruned _EXPECTED_PRUNED_W as the INPUT weight, at
    # sparsity=0.5 the very same two positions per column would be
    # "dropped" again -- but both are already exactly zero, so the pass
    # must be a strict no-op. This is the real, testable behavior the
    # guard exists for (see magnitude_pruning.h's own top comment): without
    # it, OptimizeFixed's fixed-point re-application would loop forever
    # re-"pruning" an already-pruned weight. Confirmed via the
    # initializer's own NAME staying "W" (mirrors
    # test_formal_verify_cross_layer_equalization.py's own "already
    # balanced reports no change" test), not just its value -- a
    # byte-identical REPLACEMENT initializer would still pass a
    # value-only check but would NOT be the "no change at all" the guard
    # promises.
    _set_sparsity(0.5)
    model = _model(
        """
        g (float[b,4] X) => (float[b,2] Y)
        { Y = MatMul(X, W) }
        """,
        [_f32(_EXPECTED_PRUNED_W, "W")],
    )

    sim_model, ops = simplify_isolated_extra(model, "magnitude_pruning_matmul")
    assert ops["MatMul"] == 1
    node = producer(sim_model, "Y")
    assert node.input[1] == "W"
    w = next(t for t in sim_model.graph.initializer if t.name == "W")
    np.testing.assert_array_equal(numpy_helper.to_array(w), _EXPECTED_PRUNED_W)


def test_magnitude_pruning_matmul_declines_non_constant_weight():
    # A weight that is a graph INPUT rather than a constant initializer can
    # never be matched: FetchConstantTensor returns null, so
    # patternMatchPredicate (MatchMatMulLike's own constant-weight
    # requirement) declines before SparsityMaskRowMajor ever runs -- the
    # simplest of MatchMatMulLike's several guards to exercise (see this
    # file's own module docstring for why this suite doesn't exhaustively
    # cover every one).
    _set_sparsity(0.5)
    model = _model(
        """
        g (float[b,4] X, float[4,2] W) => (float[b,2] Y)
        { Y = MatMul(X, W) }
        """
    )

    sim_model, ops = simplify_isolated_extra(model, "magnitude_pruning_matmul")
    assert ops["MatMul"] == 1
    node = producer(sim_model, "Y")
    assert list(node.input) == ["X", "W"]


def test_magnitude_pruning_matmul_declines_when_keep_covers_all_columns():
    # SparsityMaskRowMajor's own "keep >= cols" branch: at a low enough
    # sparsity, round(cols * (1 - sparsity)) reaches (or exceeds) cols
    # itself, so the mask keeps every entry -- combined with
    # MagnitudePruningWouldChange's own "no entry would actually be zeroed"
    # check, the pass must decline outright regardless of the weight's own
    # values. cols=4, sparsity=0.1: round(4 * 0.9) == round(3.6) == 4 ==
    # cols.
    _set_sparsity(0.1)
    model = _model(
        """
        g (float[b,4] X) => (float[b,2] Y)
        { Y = MatMul(X, W) }
        """,
        [_f32(_W, "W")],
    )

    sim_model, ops = simplify_isolated_extra(model, "magnitude_pruning_matmul")
    assert ops["MatMul"] == 1
    node = producer(sim_model, "Y")
    assert node.input[1] == "W"
    w = next(t for t in sim_model.graph.initializer if t.name == "W")
    np.testing.assert_array_equal(numpy_helper.to_array(w), _W.astype(np.float32))
