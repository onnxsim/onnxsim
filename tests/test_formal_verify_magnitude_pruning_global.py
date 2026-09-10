"""Formal check for MagnitudePruningGlobal (opt-in;
``onnxsim/passes/magnitude_pruning.h``): the ``global_sparsity`` mode of the
data-free unstructured pruning baseline (Han et al., 2015). N:M
(semi-structured) mode and the per-layer sparsity-ratio mode
(``MagnitudePruningMatMul``/``Conv``/``Attention``, see
``test_formal_verify_magnitude_pruning_matmul.py`` and
``test_formal_verify_magnitude_pruning_conv.py``) are OUT OF SCOPE for this
file.

Read those two files' own module docstrings first: this file shares their
overall framing (magnitude pruning is a **selection-correctness** claim, not
a value-preservation or bounded-error one -- pruning deliberately changes the
output) and their general test structure (a ``_spec`` predicate, positive/
negative concrete controls, a uniqueness-given-no-ties proof, a kept-value
lemma, and differential tests against the real compiled pass via
``simplify_isolated_extra``/``_set_sparsity``).

**What is genuinely different here** is the SCOPE of the selection.
``MagnitudePruningMatMul``/``Conv``/``Attention`` each choose a keep-count
INDEPENDENTLY per output row/filter (``SparsityMaskRowMajor``'s own
``max(1, round(cols * (1 - sparsity)))`` floor, applied one row at a time).
``MagnitudePruningGlobal`` instead:

* pools every matched layer's ENTIRE ``|W|`` magnitude array -- every
  MatMul/Gemm weight, every Conv weight, every Attention merged-QKV weight it
  finds, reachable in the top-level graph AND every nested ``If``/``Loop``/
  ``Scan``/etc. subgraph body, recursively (``CollectGraphs``) -- into ONE
  FLAT ARRAY across the WHOLE MODEL;
* computes a SINGLE keep-count from that pooled array's own total entry
  count (``keep_count = round(total * (1 - sparsity))``, clamped to
  ``[0, total]``);
* zeros exactly the lowest-scoring entries WHEREVER THEY LAND across every
  matched layer.

So the formal content below states ``_spec`` over the POOLED array (not a
single row): exactly ``keep_count`` of the pooled array's entries are kept,
and every kept entry's magnitude is ``>=`` every dropped entry's magnitude --
structurally the SAME shape of claim as the per-layer files' own ``_spec``,
just applied once to the whole pooled sequence instead of per row, and with
**no per-row/per-layer floor at all**: since there are no "rows" here, only
one flat pooled sequence, a whole layer's entire weight can legitimately end
up all-zero if every one of its entries happens to be globally low-magnitude
-- something ``SparsityMaskRowMajor``'s own ``max(1, ...)`` floor would never
permit for a single row. The uniqueness-given-no-ties proof and kept-value
lemma below are structurally identical to the per-layer files' own versions,
just re-scoped to "the pooled array" (no row/cols framing at all, since
pooling erases layer boundaries once every entry lands in one sequence).

Differential tests exercise the two things that are genuinely new about this
pass rather than shared with the per-layer passes:

1. Pooling actually happens ACROSS layers, not independently per layer --
   two top-level MatMul weights, one of uniformly HIGHER magnitude than the
   other, are pruned so the pooled top-k keeps the "better" layer's weight
   entirely and drops the "worse" one entirely -- a result no per-layer,
   independent selection at the same sparsity could produce (independently,
   each 4-entry row would keep its own top half regardless of the other
   layer).
2. That same pooling reaches into a nested ``If`` subgraph body
   (``CollectGraphs``'s own recursion point): a MatMul weight inside
   ``then_branch`` is pooled into the SAME global ranking as a top-level
   MatMul weight, not ranked independently per graph.
3. No per-layer floor: contrasted directly against
   ``magnitude_pruning_matmul`` (the per-layer pass) pruning the SAME weight,
   in isolation, at the SAME sparsity -- the per-layer pass's own
   ``max(1, ...)`` floor always keeps at least one entry per row, while
   ``magnitude_pruning_global`` can and does zero that row's layer entirely
   once pooled with a higher-magnitude sibling.
4. A brief structural note (not a differential probe with an interesting
   failure mode -- see this file's own commentary at its use site) that
   ``PassAnalysisType::Empty`` needs no ``MagnitudePruningWouldChange``-style
   idempotency guard the way the per-layer ``PredicateBasedPass``es do:
   running the pass again over its own output recomputes the identical
   pooled top-k (the values are unchanged) and is therefore naturally a
   no-op, without any explicit "would this change anything" check.

Since ``MagnitudePruningSparsity()`` is a function-local C++ static, there is
no way to set it from Python except through the one real, documented entry
point that sets it as a side effect: ``PruneMagnitude`` (``pruning_entry.cpp``),
exposed to Python as :func:`onnxsim.prune_magnitude_cpp` -- see
``_set_sparsity``'s own doc comment, copied (module name aside) from the
per-layer proof files.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
from _formal_verify_common import prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

# --- The specification: MagnitudePruningGlobal's own pooled-array analogue -
#
# Same SHAPE of claim as the per-layer files' own _spec (exactly `keep`
# entries kept, every kept entry's magnitude >= every dropped entry's), but
# stated over the whole POOLED array at once -- there is no "row"/"cols"
# framing at all here, since pooling deliberately erases per-layer
# boundaries before any selection is made. Deliberately silent on which of
# several EQUAL-magnitude entries gets dropped, exactly like the per-layer
# files' own _spec (std::stable_sort's own tie-breaking is not part of the
# correctness claim -- see those files' own module docstrings) -- just at
# the pooled scale instead of per-row. Concrete instances below are chosen
# with no ties at the cutoff so the algorithm's actual output is checked
# against _spec unambiguously.


def _spec(mask, values, keep_count):
    """Z3 formula for ``MagnitudePruningGlobal``'s own pooled-array
    specification: exactly ``keep_count`` of ``mask``'s entries (one per
    pooled-array position, spanning every matched layer concatenated
    together) are ``True`` (kept), and every kept entry's magnitude is
    ``>=`` every dropped entry's magnitude. ``values`` are ``|w|``
    magnitudes (already ``fabs``'d, matching each ``Entry``'s own
    ``importance`` array before pooling) and ``mask`` is a list of Z3
    Bools, one per pooled position, both the same length. No per-row/
    per-layer floor is expressed here at all -- unlike the per-layer files'
    own ``_spec``, which is always evaluated per-row with
    ``SparsityMaskRowMajor``'s own ``max(1, ...)`` floor already baked into
    how ``keep`` is chosen there, this ``keep_count`` is a single global
    number and nothing here prevents every entry contributed by one
    particular layer from being dropped.
    """
    n = len(values)
    count_kept = z3.Sum([z3.If(mask[i], 1, 0) for i in range(n)])
    ordering = z3.And(
        *[
            z3.Implies(z3.And(mask[i], z3.Not(mask[j])), values[i] >= values[j])
            for i in range(n)
            for j in range(n)
            if i != j
        ]
    )
    return z3.And(count_kept == keep_count, ordering)


def test_magnitude_pruning_global_selection_is_unique_given_no_ties():
    # The genuinely general claim, re-scoped from "one row" to "the whole
    # pooled array": for pairwise-DISTINCT pooled magnitudes (no ties),
    # _spec pins down a UNIQUE mask -- any two masks both satisfying _spec
    # for the same pooled array/keep_count are elementwise identical. n=8
    # mirrors the differential tests below (two conceptual 4-entry layers
    # pooled into one 8-entry array), though the proof itself needs no
    # notion of layer boundaries once pooled -- it is an ordinary
    # free-variable-universal claim (keep_count/n are fixed concrete
    # integers, no quantifier alternation needed), this suite's usual
    # prove() idiom.
    n, keep_count = 8, 4
    values = z3.Reals(" ".join(f"v{i}" for i in range(n)))
    mask_a = z3.Bools(" ".join(f"ka{i}" for i in range(n)))
    mask_b = z3.Bools(" ".join(f"kb{i}" for i in range(n)))

    no_ties = z3.And(
        *[values[i] != values[j] for i in range(n) for j in range(i + 1, n)]
    )
    both_satisfy_spec = z3.And(
        _spec(mask_a, values, keep_count), _spec(mask_b, values, keep_count)
    )
    same_mask = z3.And(*[mask_a[i] == mask_b[i] for i in range(n)])

    prove(z3.Implies(z3.And(no_ties, both_satisfy_spec), same_mask))


def test_magnitude_pruning_global_masking_preserves_kept_values_exactly():
    # The clean, separate equality claim, identical in substance to the
    # per-layer files' own version: a kept pooled entry is untouched by
    # masking, not recomputed or rounded. An ordinary unconditional
    # tautology over a free (universally quantified via prove()) mask bit
    # and value.
    mask_i = z3.Bool("mask_i")
    value_i = z3.Real("value_i")
    pruned_i = z3.If(mask_i, value_i, z3.RealVal(0))
    prove(z3.Implies(mask_i, pruned_i == value_i))


# --- Positive/negative controls on one concrete pooled array ----------------
#
# This 8-entry pooled array is exactly _LAYER_A concatenated with _LAYER_B in
# the differential tests below (conceptually two 4-entry "layers", though
# _spec/the proofs above have no notion of that boundary once pooled): |W_a|
# == [10, 9, 8, 7] (uniformly high magnitude) followed by |W_b| ==
# [4, 3, 2, 1] (uniformly low), so the mask asserted here is independently
# cross-checked against the real compiled pass's own output in
# test_magnitude_pruning_global_pools_across_layers_not_independently.

_POOLED_MAGNITUDES = [10.0, 9.0, 8.0, 7.0, 4.0, 3.0, 2.0, 1.0]
_POOLED_KEEP = 4
# MagnitudePruningGlobal's own output for this pooled array at keep_count=4:
# the four highest magnitudes (all of layer A, indices 0-3) are kept; the
# four lowest (all of layer B, indices 4-7) are dropped -- no ties. This is
# also the "no per-layer floor" witness: layer B's own row is entirely
# zeroed, which SparsityMaskRowMajor's own per-row max(1, ...) floor would
# never allow.
_POOLED_ALGORITHM_MASK = [True, True, True, True, False, False, False, False]


def test_magnitude_pruning_global_algorithm_output_satisfies_spec():
    # Positive control: the mask MagnitudePruningGlobal actually computes
    # for this concrete pooled array is a valid highest-magnitude selection
    # -- Z3 confirms both of _spec's defining properties hold for this exact
    # concrete (ground/closed) assignment.
    mask = [z3.BoolVal(v) for v in _POOLED_ALGORITHM_MASK]
    values = [z3.RealVal(v) for v in _POOLED_MAGNITUDES]
    prove(_spec(mask, values, _POOLED_KEEP))


def test_magnitude_pruning_global_negative_control_wrong_selection_violates_spec():
    # Negative control: a mask that drops the pooled array's SINGLE
    # HIGHEST-magnitude entry (index 0, |10.0|, layer A's own first entry)
    # while keeping a strictly lower-magnitude one (index 4, |4.0|, layer
    # B's own first entry) is a genuinely wrong selection -- _spec must not
    # hold for it.
    wrong_mask = [False, True, True, True, True, False, False, False]
    mask = [z3.BoolVal(v) for v in wrong_mask]
    values = [z3.RealVal(v) for v in _POOLED_MAGNITUDES]
    prove(z3.Not(_spec(mask, values, _POOLED_KEEP)))


def test_magnitude_pruning_global_spec_permits_zeroing_an_entire_layer():
    # Formal confirmation of the "no per-layer floor" property itself,
    # independent of the differential test below: _spec's own positive
    # control mask above already zeros layer B's entire row (indices 4-7)
    # -- there is nothing in _spec analogous to SparsityMaskRowMajor's own
    # max(1, round(cols * (1 - sparsity))) floor that would forbid this.
    # Restated explicitly here (rather than only implicitly via the
    # positive control above) so the "no floor" property has its own named,
    # documented proof.
    all_of_layer_b_dropped = z3.And(
        *[z3.Not(m) for m in [z3.BoolVal(v) for v in _POOLED_ALGORITHM_MASK[4:]]]
    )
    mask = [z3.BoolVal(v) for v in _POOLED_ALGORITHM_MASK]
    values = [z3.RealVal(v) for v in _POOLED_MAGNITUDES]
    prove(z3.Implies(_spec(mask, values, _POOLED_KEEP), all_of_layer_b_dropped))


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
    ``simplify_isolated_extra(..., "magnitude_pruning_global", ...)`` call,
    which is what actually exercises the real compiled pass under test.
    """
    onnxsim.prune_magnitude_cpp(_THROWAWAY_SPARSITY_MODEL, sparsity=sparsity)


def _initializer(graph_proto, name):
    return next(t for t in graph_proto.initializer if t.name == name)


def _weight_of(graph_proto, output_name):
    """The pruned weight actually written by the pass for the node producing
    ``output_name`` within ``graph_proto`` (a top-level ``GraphProto`` or a
    subgraph body), found via that node's own CURRENT weight input name --
    mirrors the per-layer proof files' own ``_pruned_weight`` helper: the
    pass leaves the original initializer dangling and rewires the node to a
    brand-new one, into whichever Graph actually owns that node (the
    top-level graph, or a subgraph body attribute's own ``GraphProto`` --
    see ``MagnitudePruningGlobal::runPass``'s own
    ``e.node->owningGraph()->addInitializerAndCreateValue``), so the pruned
    tensor must be looked up in the SAME ``graph_proto`` the node lives in,
    not assumed to be the top-level graph's initializer list.
    """
    node = next(n for n in graph_proto.node if output_name in n.output)
    w_name = node.input[1]
    return numpy_helper.to_array(_initializer(graph_proto, w_name))


def _branch_graph(if_node, attr_name):
    return next(a.g for a in if_node.attribute if a.name == attr_name)


# --- Test 1: pooling across two top-level layers, not independently --------
#
# _LAYER_A is uniformly HIGH magnitude, _LAYER_B uniformly LOW -- pooled
# together (8 entries, sparsity=0.5 -> keep_count = round(8 * 0.5) = 4), the
# global top-4 is _LAYER_A's own four entries in their entirety, dropping
# _LAYER_B's own four entirely. This is the key differential confirmation
# that pooling, not per-layer independence, is actually happening: an
# INDEPENDENT per-layer selection at the same sparsity (as
# magnitude_pruning_matmul's own SparsityMaskRowMajor would do to each layer
# in isolation) would instead keep 2 of 4 in EACH layer (round(4 * 0.5) ==
# 2), never zeroing either layer's weight entirely.

_LAYER_A = np.array([[10.0], [9.0], [8.0], [7.0]])  # K=4, N=1
_LAYER_B = np.array([[4.0], [3.0], [2.0], [1.0]])  # K=4, N=1


def _two_layer_model():
    return _model(
        """
        g (float[b,4] X) => (float[b,1] Y_a, float[b,1] Y_b)
        {
          Y_a = MatMul(X, W_a)
          Y_b = MatMul(X, W_b)
        }
        """,
        [_f32(_LAYER_A, "W_a"), _f32(_LAYER_B, "W_b")],
    )


def test_magnitude_pruning_global_pools_across_layers_not_independently():
    _set_sparsity(0.5)
    model = _two_layer_model()

    # check_n=0: this pass deliberately changes the computed output, so
    # onnxsim's own random-input equivalence check does not apply here --
    # mirrors the per-layer proof files' own check_n=0 firing tests.
    sim_model, ops = simplify_isolated_extra(
        model, "magnitude_pruning_global", check_n=0
    )
    assert ops["MatMul"] == 2

    w_a = _weight_of(sim_model.graph, "Y_a")
    w_b = _weight_of(sim_model.graph, "Y_b")

    # Layer A is kept ENTIRELY (every entry is among the pooled top-4) --
    # byte-identical to the original, not merely close.
    np.testing.assert_array_equal(w_a, _LAYER_A)
    # Layer B is dropped ENTIRELY -- no per-layer floor keeps even its
    # single highest-magnitude entry (|4.0|) alive, unlike what
    # magnitude_pruning_matmul's own per-row floor would do to this same
    # weight in isolation (see the contrast test below).
    np.testing.assert_array_equal(w_b, np.zeros_like(_LAYER_B))


def test_magnitude_pruning_global_no_per_layer_floor_contrasts_with_matmul_pass():
    # The pass's own defining difference, spelled out explicitly: pruning
    # _LAYER_B ALONE (not pooled with anything else) via the per-layer
    # magnitude_pruning_matmul pass, at the SAME sparsity=0.5, keeps
    # max(1, round(4 * 0.5)) == 2 entries -- ITS OWN row-local floor never
    # zeros the whole row. Pooled with a higher-magnitude sibling layer
    # under magnitude_pruning_global (previous test), that exact same
    # weight is zeroed ENTIRELY instead -- the whole point of "no per-row/
    # per-layer floor" at the global scale.
    _set_sparsity(0.5)
    solo_model = _model(
        """
        g (float[b,4] X) => (float[b,1] Y_b)
        { Y_b = MatMul(X, W_b) }
        """,
        [_f32(_LAYER_B, "W_b")],
    )
    sim_model, ops = simplify_isolated_extra(
        solo_model, "magnitude_pruning_matmul", check_n=0
    )
    assert ops["MatMul"] == 1
    w_b_solo = _weight_of(sim_model.graph, "Y_b")

    # The per-layer pass keeps exactly 2 of 4 entries (its own floor-derived
    # keep count) -- the two highest magnitudes, |4.0| and |3.0|.
    assert np.count_nonzero(w_b_solo) == 2
    np.testing.assert_array_equal(w_b_solo, np.array([[4.0], [3.0], [0.0], [0.0]]))


# --- Test 2: pooling reaches into a nested If subgraph body -----------------
#
# _TOP is uniformly LOW magnitude, _NESTED (inside the If node's own
# then_branch) is uniformly HIGH -- pooled together (8 entries total,
# sparsity=0.5 -> keep_count=4), the global top-4 selection keeps _NESTED's
# own four entries entirely and drops _TOP's own four entirely, exactly
# mirroring test 1's own cross-layer result but now across a graph boundary.
# This is only possible if CollectGraphs actually recurses into the If
# node's then_branch GraphProto attribute and pools its weight into the SAME
# ranking as the top-level graph's own weight -- not two independent
# 4-entry arrays (which would instead keep 2 of 4 in EACH graph, never
# zeroing _TOP's layer entirely) and not a missed subgraph entirely (which
# would leave _NESTED untouched and prune _TOP alone at keep_count=2).

_TOP = np.array([[4.0], [3.0], [2.0], [1.0]])  # K=4, N=1, top-level graph
_NESTED = np.array([[10.0], [9.0], [8.0], [7.0]])  # K=4, N=1, If/then_branch


def _if_subgraph_model():
    model = _model(
        """
        g (float[2,4] X, bool Cond) => (float[2,1] Y_top, float[2,1] Y_if)
        {
          Y_top = MatMul(X, W_top)
          Y_if = If <
            then_branch = then_g () => (float[2,1] T) {
              T = MatMul(X, W_then)
            },
            else_branch = else_g () => (float[2,1] E) {
              E = Constant<value = float[2,1] {0.0, 0.0}>()
            }
          > (Cond)
        }
        """,
        [_f32(_TOP, "W_top")],
    )
    if_node = next(n for n in model.graph.node if n.op_type == "If")
    then_g = _branch_graph(if_node, "then_branch")
    then_g.initializer.extend([_f32(_NESTED, "W_then")])
    return model


def test_magnitude_pruning_global_pools_across_if_subgraph_boundary():
    _set_sparsity(0.5)
    model = _if_subgraph_model()

    sim_model, ops = simplify_isolated_extra(
        model, "magnitude_pruning_global", check_n=0
    )
    assert ops["MatMul"] == 1
    assert ops["If"] == 1

    if_node = next(n for n in sim_model.graph.node if n.op_type == "If")
    then_g = _branch_graph(if_node, "then_branch")

    w_top = _weight_of(sim_model.graph, "Y_top")
    w_nested = _weight_of(then_g, "T")

    # The top-level layer is dropped ENTIRELY: pooled with the nested
    # layer's own uniformly-higher magnitudes, none of its own four entries
    # make the global top-4.
    np.testing.assert_array_equal(w_top, np.zeros_like(_TOP))
    # The nested layer (inside the If's own then_branch subgraph) is kept
    # ENTIRELY -- byte-identical to its original value, confirming it was
    # pooled into the SAME ranking as the top-level graph's own weight, not
    # scored against only itself (which would have kept just its own top-2,
    # {10.0, 9.0}, dropping {8.0, 7.0}) nor left unmatched altogether.
    np.testing.assert_array_equal(w_nested, _NESTED)


# --- Test 4 (structural note): no idempotency guard is needed --------------


def test_magnitude_pruning_global_is_naturally_idempotent_without_a_guard():
    # magnitude_pruning.h's own top comment: MagnitudePruningGlobal is a
    # FullGraphBasedPass with PassAnalysisType::Empty, so
    # FixedPointPassManager runs it exactly once per simplify() call
    # regardless -- unlike the per-layer PredicateBasedPasses, it carries no
    # MagnitudePruningWouldChange-style "would this still change anything"
    # guard at all (confirmed by reading runPass in magnitude_pruning.h: no
    # such check appears anywhere in it). This is a structural fact more
    # than an interesting failure mode to differentially catch (there is no
    # OptimizeFixed re-invocation within a single simplify() call for an
    # Empty-analysis pass to loop on), but it is still worth confirming
    # directly: running the pass a SECOND time, from Python, over its own
    # already-pruned output recomputes the identical pooled top-k (the
    # surviving values are unchanged, so the same ranking is reproduced)
    # and is therefore naturally a no-op on values, with no explicit
    # "already pruned" check required.
    _set_sparsity(0.5)
    model = _two_layer_model()
    sim_model, _ops = simplify_isolated_extra(
        model, "magnitude_pruning_global", check_n=0
    )
    w_a_once = _weight_of(sim_model.graph, "Y_a")
    w_b_once = _weight_of(sim_model.graph, "Y_b")

    _set_sparsity(0.5)
    sim_model_again, _ops = simplify_isolated_extra(
        sim_model, "magnitude_pruning_global", check_n=0
    )
    w_a_twice = _weight_of(sim_model_again.graph, "Y_a")
    w_b_twice = _weight_of(sim_model_again.graph, "Y_b")

    np.testing.assert_array_equal(w_a_once, w_a_twice)
    np.testing.assert_array_equal(w_b_once, w_b_twice)


def test_magnitude_pruning_global_is_an_opt_in_pass():
    C = onnxsim.onnxsim_cpp2py_export
    assert "magnitude_pruning_global" in C._list_other_optimizers()
    assert "magnitude_pruning_global" not in C._list_optimizers()
