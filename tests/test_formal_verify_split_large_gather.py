"""Formal check for SplitLargeGather (opt-in; onnxsim's own
``onnxsim/passes/split_large_gather.h``): the mirror image of
``fuse_split_gather_concat`` (a default-on pass that collapses exactly this
shape back into one ``Gather``, out of scope here -- see this pass's own
header comment). Rewrites ``Gather(x, idx, axis=ga)``, when ``idx``'s total
element count exceeds the constant ``kMaxIndicesPerGather`` (``2**16``), into
``Concat(Gather(x, s_0, axis=ga), ..., Gather(x, s_{k-1}, axis=ga), axis=ga)``
where ``s_0, ..., s_{k-1} = Split(idx, axis=0)`` -- i.e. the (large) indices
tensor is split along *its own* axis 0 into ``k`` contiguous chunks, ``x`` is
gathered separately with each chunk (still along the *original* gather axis
``ga`` -- a different axis of a different tensor than indices' axis 0, easy
to conflate but never the same axis here), and the ``k`` partial results are
re-concatenated along ``ga``.

Formal content: this is the same concatenation-offset arithmetic as
``test_formal_verify_fuse_consecutive_concats.py`` and
``test_formal_verify_rewrite_gather_over_concat.py`` (each segment/chunk an
uninterpreted function, a global position resolved by a nested ``z3.If``
case-split into "which chunk, and at what local offset within it"), but
applied in the mirror direction from ``rewrite_gather_over_concat``: there,
one Concat'd *input* was read at a single resolved offset; here, the *whole*
indices tensor drives one Gather each per chunk, and it is the *outputs* of
those per-chunk Gathers that get concatenated back together. Concretely: for
an indices tensor conceptually split into 3 contiguous chunks of concrete
lengths 2, 3, 2 (as concrete as ``fuse_consecutive_concats``'s own
``_LEN_*`` constants, for the same tractability reason) along its own axis
0, reading ``Gather(x, idx, axis=ga)`` at any given output row position ``p``
(corresponding to one index, i.e. one row of ``idx``) is proved equal to
reading the correct one of the three per-chunk ``Gather(x, s_j, axis=ga)``
results at ``p``'s local offset within its own chunk.

``x`` (the data being gathered from) is modeled as an uninterpreted
``Int -> Real`` function, 1-D for simplicity -- the real op is more
general-rank (``Gather`` can act along any axis of an N-D ``x``), but the
axis-0-of-*indices* splitting logic this pass adds is entirely independent
of ``x``'s own rank or of which axis ``ga`` happens to be, so a 1-D ``x`` is
enough to capture the essential content, exactly as
``fuse_consecutive_concats``'s own docstring makes the same 1-D-suffices
argument for Concat's axis. ``idx`` is modeled as an uninterpreted
``Int -> Int`` function (full generality -- real indices must be
compile-time constants for the predicate to fire at all, per the header
comment, but nothing in the offset arithmetic below depends on the specific
constant values, so leaving ``idx`` uninterpreted proves the identity for
every possible constant indices tensor at once rather than one concrete
example). A negative control confirms the chunk/offset selection is load-
bearing, not vacuously true: reading the *wrong* chunk at the same local
offset does not generally match.

The differential checks below additionally confirm, against the real
compiled pass: (1) a basic firing case, checking both the resulting node
counts (Split/Gather/Concat) *and* ``ComputeChunkSizes``'s exact greedy
chunking arithmetic, hand-computed and compared against the actual chunk
sizes the pass produces -- read off the ``Split`` node's own split-sizes
initializer when ``indices`` is left non-constant (a *structure* test), and
off the folded per-chunk indices tensors' shapes and values when ``indices``
is a concrete constant fed through onnxsim's own random-sample correctness
check instead (a *values* test -- see that test's docstring for why the
``Split`` node itself doesn't survive there); (2) the pass declines when
``idx``'s total element count is at or under the limit; and (3) the pass
declines when the per-"row" element count (product of every axis but 0)
alone already exceeds the limit, so no split along axis 0 could help --
distinct from the (untested here) ``dim0 <= 1`` decline branch. All three
also exercise ``ga`` != axis 0 (gathering along axis 1 of a 2-D ``x`` while
splitting axis 0 of ``idx``), confirming the two axes are not conflated by
the implementation either.

Note (not this test's job): ``runTransform`` tags the ``Split`` node's doc
string with ``kSizeLimitedGatherSplitMarker`` (see
``gather_split_concat_markers.h``) specifically so ``fuse_split_gather_concat``
-- a *default-on* pass that would otherwise recognize this exact
Split/Gather*/Concat shape and fuse it straight back into one oversized
Gather -- leaves it alone. Isolating ``split_large_gather`` on its own via
``simplify_isolated_extra`` already skips every default pass including
``fuse_split_gather_concat``, so the marker plays no role in the differential
tests below; it is ``fuse_split_gather_concat``'s own test's job to verify.
"""

import numpy as np
import onnx
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

_LEN_A = 2
_LEN_B = 3
_LEN_C = 2

_K_MAX_INDICES_PER_GATHER = 1 << 16  # kMaxIndicesPerGather


def _compute_chunk_sizes(dim0, per_row):
    """Pure-Python mirror of ``SplitLargeGather::ComputeChunkSizes`` in
    split_large_gather.h, used below to predict the pass's exact Split
    sizes by hand rather than just trusting whatever it produces."""
    if dim0 <= 1 or per_row > _K_MAX_INDICES_PER_GATHER:
        return []
    chunk_rows = max(1, _K_MAX_INDICES_PER_GATHER // per_row)
    sizes = []
    remaining = dim0
    while remaining > 0:
        c = min(chunk_rows, remaining)
        sizes.append(c)
        remaining -= c
    return sizes


def test_split_large_gather_is_sound():
    x = z3.Function("x", z3.IntSort(), z3.RealSort())
    idx = z3.Function("idx", z3.IntSort(), z3.IntSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    p = z3.Int("p")

    def full_gather(i):
        # y = Gather(x, idx, axis=ga), read at output row i -- what the
        # original, unsplit Gather computes for row i of the full indices
        # tensor.
        return x(idx(i))

    # s_0, s_1, s_2 = Split(idx, axis=0) with axis-0 chunk sizes 2, 3, 2:
    # by Split's own semantics, chunk j's local row q is idx's row
    # (offset_j + q) -- nothing more than index-arithmetic bookkeeping.
    def gather_a(q):  # Gather(x, s_0, axis=ga)
        return x(idx(q))

    def gather_b(q):  # Gather(x, s_1, axis=ga)
        return x(idx(_LEN_A + q))

    def gather_c(q):  # Gather(x, s_2, axis=ga)
        return x(idx(_LEN_A + _LEN_B + q))

    def concat_read(i):
        # Concat(Gather(x,s_0), Gather(x,s_1), Gather(x,s_2), axis=ga), read
        # at global row i -- what runTransform's rewritten graph computes.
        return z3.If(
            i < _LEN_A,
            gather_a(i),
            z3.If(
                i < _LEN_A + _LEN_B,
                gather_b(i - _LEN_A),
                gather_c(i - _LEN_A - _LEN_B),
            ),
        )

    total = _LEN_A + _LEN_B + _LEN_C
    prove(
        z3.Implies(
            z3.And(p >= 0, p < total),
            consumer(full_gather(p)) == consumer(concat_read(p)),
        )
    )


def test_split_large_gather_chunk_selection_is_load_bearing():
    # Negative control: reading the *wrong* chunk (s_0, offset 0) at row p's
    # local offset within s_1 (i.e. p - _LEN_A) is not generally the correct
    # answer -- confirming the chunk/offset selection above is load-bearing,
    # not a vacuously-true rewrite (idx is uninterpreted, so gather_a(p -
    # _LEN_A) = x(idx(p - _LEN_A)) has no forced relationship to the correct
    # x(idx(p))).
    x = z3.Function("x", z3.IntSort(), z3.RealSort())
    idx = z3.Function("idx", z3.IntSort(), z3.IntSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    p = z3.Int("p")

    def full_gather(i):
        return x(idx(i))

    def gather_a(q):  # Gather(x, s_0, axis=ga)
        return x(idx(q))

    in_b_chunk = z3.And(p >= _LEN_A, p < _LEN_A + _LEN_B)
    wrong_chunk_claim = z3.Implies(
        in_b_chunk,
        consumer(full_gather(p)) == consumer(gather_a(p - _LEN_A)),
    )

    solver = z3.Solver()
    solver.add(z3.Not(wrong_chunk_claim))
    assert solver.check() == z3.sat, (
        "reading the wrong chunk should not generally match -- chunk "
        "selection would be vacuous"
    )


def _gather_model(data_sig, out_rank, ga, indices):
    # `indices` is attached as a numpy-built initializer after parsing (per
    # CLAUDE.md: large/constant arrays are built programmatically, not
    # spelled out as ONNX text literals) -- and it must be a genuine
    # initializer (not e.g. a same-shaped graph input) so onnxsim's own
    # random-sample equivalence check (run by `simplify_isolated_extra`)
    # feeds it the same in-bounds values on both the original and simplified
    # graphs, rather than fresh random indices that could go out of bounds.
    out_dims = ",".join(["?"] * out_rank)
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g ({data_sig}) => (float[{out_dims}] Y)
        {{
          Y = Gather<axis={ga}>(data, indices)
        }}
        """
    )
    model.graph.initializer.extend(
        [onnx.numpy_helper.from_array(indices.astype(np.int64), name="indices")]
    )
    return model


def _gather_model_dynamic_indices(data_sig, indices_sig, out_rank, ga):
    # Same shape as `_gather_model`, but `indices` is a plain graph *input*
    # (its shape still fully static, which is all `patternMatchPredicate`
    # requires) rather than a constant initializer. Used only for the
    # structural check below, where we want the pass's own `Split` node to
    # survive intact -- see that test for why a constant `indices` doesn't
    # let us observe it directly.
    out_dims = ",".join(["?"] * out_rank)
    return parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g ({data_sig}, {indices_sig} indices) => (float[{out_dims}] Y)
        {{
          Y = Gather<axis={ga}>(data, indices)
        }}
        """
    )


# ga=1 (axis 1 of `data`, a 2-D tensor) is deliberately *not* 0 (the axis
# `idx` is split along) in both tests below, to confirm the pass keeps the
# two straight: the new Gathers and the Concat all use ga=1, while the Split
# -- always along `idx`'s own axis 0 -- uses axis 0 regardless of ga.
_DIM0, _PER_ROW = 3, 25000  # total 75000 > kMaxIndicesPerGather (65536)


def test_split_large_gather_pass_matches_basic_firing_case_structure():
    # dim0=3, per_row=25000 -> chunk_rows = max(1, 65536 // 25000) = 2 ->
    # ComputeChunkSizes greedily chunks dim0=3 into [2, 1] (k=2).
    #
    # `indices` is a dynamic (non-constant) graph input here purely so the
    # `Split` node this pass creates is observable at all: onnxsim folds any
    # node whose entire input is compile-time-constant into a plain
    # initializer as part of its own (pass-independent) simplification, and
    # since `Split`'s only real input here is `indices` itself, a *constant*
    # indices tensor gets its `Split` folded away immediately -- see
    # `test_split_large_gather_pass_matches_basic_firing_case_values` below,
    # which uses concrete indices values instead and observes exactly that.
    # `check_n=0` skips onnxsim's own random-sample correctness check, which
    # would otherwise feed `indices` out-of-bounds values; that check_n=1
    # differential correctness role is exactly what the sibling values-based
    # test below covers instead.
    expected_chunk_sizes = _compute_chunk_sizes(_DIM0, _PER_ROW)
    assert expected_chunk_sizes == [2, 1]

    model = _gather_model_dynamic_indices(
        "float[5,10] data", f"int64[{_DIM0},{_PER_ROW}]", out_rank=3, ga=1
    )
    sim_model, ops = simplify_isolated_extra(model, "split_large_gather", check_n=0)
    assert ops["Split"] == 1
    assert ops["Concat"] == 1
    assert ops["Gather"] == len(expected_chunk_sizes)

    concat_node = producer(sim_model, "Y")
    assert concat_node.op_type == "Concat"
    assert onnx.helper.get_node_attr_value(concat_node, "axis") == 1  # ga, not 0

    gather_nodes = [producer(sim_model, name) for name in concat_node.input]
    assert all(g.op_type == "Gather" for g in gather_nodes)
    assert all(g.input[0] == "data" for g in gather_nodes)
    assert all(onnx.helper.get_node_attr_value(g, "axis") == 1 for g in gather_nodes)

    # The Split -- unlike the Gathers and Concat above -- always splits
    # `idx`'s own axis 0, independent of ga.
    (split_node,) = [n for n in sim_model.graph.node if n.op_type == "Split"]
    assert split_node.input[0] == "indices"
    assert onnx.helper.get_node_attr_value(split_node, "axis") == 0
    (split_sizes_init,) = [
        init for init in sim_model.graph.initializer if init.name == split_node.input[1]
    ]
    assert list(onnx.numpy_helper.to_array(split_sizes_init)) == expected_chunk_sizes

    # Each Gather's own indices input is one of the Split's outputs, in order.
    assert [g.input[1] for g in gather_nodes] == list(split_node.output)


def test_split_large_gather_pass_matches_basic_firing_case_values():
    # Same shape as the structural test above, but with concrete, in-bounds
    # `indices` values, run through onnxsim's own random-sample correctness
    # check (check_n=1) -- the actual differential check against the
    # compiled pass's runtime behavior, not just its output graph shape.
    #
    # Because `indices` is now a compile-time constant, onnxsim's own
    # (pass-independent) constant folding immediately reduces the `Split`
    # this pass creates down to two constant chunk initializers -- so unlike
    # the structural test above, no `Split` node remains to inspect
    # directly. `ComputeChunkSizes`'s arithmetic is instead confirmed by
    # comparing those two folded chunk tensors against the same chunks
    # sliced out by hand with plain numpy.
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 10, size=(_DIM0, _PER_ROW))
    expected_chunk_sizes = _compute_chunk_sizes(_DIM0, _PER_ROW)
    assert expected_chunk_sizes == [2, 1]
    expected_chunks = np.split(indices, np.cumsum(expected_chunk_sizes)[:-1], axis=0)

    model = _gather_model("float[5,10] data", out_rank=3, ga=1, indices=indices)
    sim_model, ops = simplify_isolated_extra(model, "split_large_gather", check_n=1)
    assert ops["Split"] == 0  # folded away, see docstring above
    assert ops["Concat"] == 1
    assert ops["Gather"] == len(expected_chunk_sizes)

    concat_node = producer(sim_model, "Y")
    assert concat_node.op_type == "Concat"
    assert onnx.helper.get_node_attr_value(concat_node, "axis") == 1  # ga, not 0

    gather_nodes = [producer(sim_model, name) for name in concat_node.input]
    assert all(g.op_type == "Gather" for g in gather_nodes)
    assert all(g.input[0] == "data" for g in gather_nodes)
    assert all(onnx.helper.get_node_attr_value(g, "axis") == 1 for g in gather_nodes)

    actual_chunks = []
    for g in gather_nodes:
        (init,) = [
            init for init in sim_model.graph.initializer if init.name == g.input[1]
        ]
        actual_chunks.append(onnx.numpy_helper.to_array(init))
    assert [c.shape[0] for c in actual_chunks] == expected_chunk_sizes
    for actual, expected in zip(actual_chunks, expected_chunks):
        np.testing.assert_array_equal(actual, expected)


def test_split_large_gather_declines_under_the_limit():
    # indices: shape [2, 30000], total 60000 <= kMaxIndicesPerGather (65536)
    # -- the predicate's very first size check fails, so the pass must not
    # fire at all: no Split/Concat, and the single Gather survives untouched.
    rng = np.random.default_rng(1)
    indices = rng.integers(0, 5, size=(2, 30000))
    model = _gather_model("float[5] data", out_rank=2, ga=0, indices=indices)
    sim_model, ops = simplify_isolated_extra(model, "split_large_gather", check_n=1)
    assert ops["Gather"] == 1
    assert ops["Split"] == 0
    assert ops["Concat"] == 0
    gather_node = producer(sim_model, "Y")
    assert gather_node.input == ["data", "indices"]


def test_split_large_gather_declines_when_per_row_alone_exceeds_the_limit():
    # indices: shape [2, 65537]. total = 131074 > kMaxIndicesPerGather, so
    # the pass gets past the predicate's first check -- but per_row = 65537
    # already exceeds kMaxIndicesPerGather on its own, so no split along
    # axis 0 (dim0=2, not <=1, so this is *not* the dim0<=1 decline branch)
    # could bring any one chunk under the limit; ComputeChunkSizes returns
    # {} and the pass declines.
    rng = np.random.default_rng(2)
    dim0, per_row = 2, _K_MAX_INDICES_PER_GATHER + 1
    indices = rng.integers(0, 5, size=(dim0, per_row))
    assert _compute_chunk_sizes(dim0, per_row) == []

    model = _gather_model("float[5] data", out_rank=2, ga=0, indices=indices)
    sim_model, ops = simplify_isolated_extra(model, "split_large_gather", check_n=1)
    assert ops["Gather"] == 1
    assert ops["Split"] == 0
    assert ops["Concat"] == 0
    gather_node = producer(sim_model, "Y")
    assert gather_node.input == ["data", "indices"]
