"""Formal check for FuseQKV (fuse_qkv.h).

onnxsim registers its own copy of this pass (``onnxsim/passes/fuse_qkv.h``)
via ``RegisterOrReplace``, overwriting the upstream onnx-optimizer entry of
the same name
(``third_party/onnx-optimizer/onnxoptimizer/passes/fuse_qkv.h``) in the pass
registry that ``getPassName()`` keys into -- so onnxsim's version is the one
that actually runs, the same override pattern
``test_formal_verify_fuse_add_bias_into_conv.py`` documents for that pass.
The two are byte-for-byte identical apart from one added guard in
``runTransform`` (see below); onnxsim's version fixes a real crash in the
upstream original.

``patternMatchPredicate`` matches any ``MatMul`` node whose **first** input
(``input(0)``) has *exactly 3 uses* -- i.e. some shared value ``X`` feeds
exactly 3 different consumers (candidate Q/K/V projections). Note this
matches on the *shared input*, not on any property distinguishing the
matched node itself: all 3 of those consumers (if all are themselves
qualifying MatMuls) satisfy the predicate identically, so which one
``patternMatchPredicate`` happens to visit first only affects which node
becomes ``n`` -- confirmed empirically below (see
``test_fuse_qkv_pass_matches_basic_case``), it's the first one in the
node list, since that's the first one the pass's node walk reaches.

``runTransform`` (onnxsim's version):

1. **The onnxsim-only fix**: first checks ``n->input(0)->node()->kind() ==
   kParam`` -- i.e. whether ``X``'s own producer is the graph's placeholder
   "node" for a plain graph input, meaning there is no real node to anchor
   the new fused ``Concat`` after. If so, it declines (``return false``)
   immediately. Upstream's original has no such check: it unconditionally
   does ``cat->insertAfter(PrevNode(n, 0))``, which for a graph-input ``X``
   resolves to the ``kParam`` placeholder -- a node that is never part of
   the real node list ``insertAfter`` requires, tripping an
   ``ONNX_ASSERT`` and aborting *onnxsim's entire simplify() pipeline*
   outright, not just this one pass's match. This is exactly the common
   self-attention shape: ``Q = MatMul(X, Wq)`` etc. reading the graph input
   ``X`` directly with no intervening node. Confirmed empirically below
   (``test_fuse_qkv_declines_when_shared_input_is_a_graph_input``) that
   onnxsim's version declines gracefully here instead of crashing.
2. For each of the 3 uses of ``X``: bails unless it is operand **0** of
   another ``MatMul`` (``use.offset != 0`` bails), that MatMul's own output
   has *exactly 1 use*, and that MatMul's operand 1 (the weight) is a
   compile-time constant (``IsConstantTensor``).
3. Fetches the 3 weight tensors (``q_t``/``k_t``/``v_t``, called Q/K/V only
   by the *order* ``uses()`` happens to return -- the header comment notes
   "q k v not a one-to-one correspondence actually") and requires them to
   have the **identical shape**; declines otherwise.
4. Builds a new ``Concat`` (inputs ``[Wq, Wk, Wv]`` in that order, axis =
   ``q_t.sizes().size() - 1``, i.e. the weights' own last/output axis), a
   new ``MatMul(X, Concat(...))``, and a new 3-output ``Split`` (axis
   ``-1``) of that MatMul's output into chunks sized
   ``[Wq.sizes().back(), Wk.sizes().back(), Wv.sizes().back()]`` (an
   initializer input for opset 13+, a ``split`` attribute pre-13 -- both
   confirmed empirically below).
5. Rewires the **original** 3 separate MatMul nodes' consumers onto the
   Split's 3 respective outputs (Q's consumers -> output 0, K's -> output
   1, V's -> output 2), then destroys **only** the matched node ``n``
   (``NodeDestroyType::DestroyOne``) -- the *other two* original MatMul
   nodes are left in the graph, now producing values nothing reads
   (dangling), not destroyed. Confirmed empirically below: with Q/K/V
   matched in source order, the pass visits the Q-producing MatMul first
   (so it becomes ``n`` and is destroyed), while the original K- and
   V-producing MatMul nodes survive as dead code.

Formal content: this is a genuine BLOCK MATRIX MULTIPLICATION identity --
matrix multiplication distributes over column-concatenation of the
right-hand operand: ``X @ [Wq | Wk | Wv] == [X@Wq | X@Wk | X@Wv]`` (weights
concatenated along their *output*/last axis, multiplied once, then split
back apart along that same axis). Each output column of the fused product is
``X`` dotted with one column of the concatenated weight matrix, which is
exactly a column of whichever of ``Wq``/``Wk``/``Wv`` that column came from
-- matching, index for index, the corresponding column of the
corresponding separate ``X@W`` product. Modeled below with ``X`` and each of
``Wq``/``Wk``/``Wv`` as uninterpreted ``Int, Int -> Real`` functions sharing
one contraction dimension ``K`` (all three MatMuls share the same ``X``) and
one output width ``N`` (the predicate's own identical-shape requirement),
concrete-sized (``K=2``, ``N=2``, fused width ``3*N=6``) to keep the
contraction sums finite for Z3, mirroring
``test_formal_verify_fuse_transpose_into_gemm.py``'s and
``test_formal_verify_fuse_matmul_into_conv.py``'s own concrete ``_M``/``_K``/
``_N`` matrix proofs rather than a fully symbolic-dimensioned one. Composing
with an arbitrary uninterpreted ``consumer`` then proves substitution safety
for any downstream reader of the split outputs, matching this suite's
established style (see e.g. ``test_formal_verify_eliminate_nop_split.py``).
"""

import numpy as np
import onnx
from _formal_verify_common import isolate, prove, simplify_isolated, z3
from onnx import parser

import onnxsim

_K = 2  # shared contraction dim -- all 3 MatMuls read the same X, so share K
_N = 2  # per-head (Q/K/V) output width -- the predicate requires all 3 equal


def _concat_cols(weights, k, q):
    """``Concat(Wq, Wk, Wv, axis=-1)`` read at (row=k, col=q): whichever of
    the 3 weight matrices column ``q`` falls into (a third of the ``3*_N``
    fused width each), re-indexed to that matrix's own local column."""
    third, offset = divmod(q, _N)
    return weights[third](k, offset)


def _matmul(A, B, K):
    """``MatMul(A, B)`` read at (row, col), contracting over ``range(K)``."""

    def out(row, col):
        return sum(A(row, k) * B(k, col) for k in range(K))

    return out


def test_block_matmul_distributes_over_weight_concat_is_sound():
    """``X @ Concat(Wq, Wk, Wv, axis=-1)``, read back at fused column
    ``third * N + offset`` (exactly what the new ``Split`` node does -- it
    only re-indexes, it does not recompute), equals the corresponding
    separate ``X @ W_third`` read at its own local column ``offset`` -- for
    every row, every third (Q/K/V), and every local column. This is exactly
    the algebra ``runTransform`` relies on: fuse via Concat+MatMul, then
    split the result back into 3 pieces that must reproduce the 3 original
    separate MatMuls' outputs untouched.
    """
    X = z3.Function("X", z3.IntSort(), z3.IntSort(), z3.RealSort())
    Wq = z3.Function("Wq", z3.IntSort(), z3.IntSort(), z3.RealSort())
    Wk = z3.Function("Wk", z3.IntSort(), z3.IntSort(), z3.RealSort())
    Wv = z3.Function("Wv", z3.IntSort(), z3.IntSort(), z3.RealSort())
    weights = (Wq, Wk, Wv)

    def concat(k, q):
        return _concat_cols(weights, k, q)

    fused = _matmul(X, concat, _K)  # X @ Concat(Wq, Wk, Wv, axis=-1)

    claims = []
    for third, w in enumerate(weights):
        original = _matmul(X, w, _K)  # the original separate X @ W_third
        for row in range(_K):
            for offset in range(_N):
                split_read = fused(row, third * _N + offset)
                claims.append(split_read == original(row, offset))
    prove(z3.And(*claims))


def test_block_matmul_composition_with_consumer_is_sound():
    """The identity above survives substitution into an arbitrary
    downstream consumer -- proving it's safe for *any* reader of the split
    outputs (exactly what ``tryReplacingAllUsesWith`` rewires onto), not
    only a concrete one.
    """
    X = z3.Function("X", z3.IntSort(), z3.IntSort(), z3.RealSort())
    Wq = z3.Function("Wq", z3.IntSort(), z3.IntSort(), z3.RealSort())
    Wk = z3.Function("Wk", z3.IntSort(), z3.IntSort(), z3.RealSort())
    Wv = z3.Function("Wv", z3.IntSort(), z3.IntSort(), z3.RealSort())
    weights = (Wq, Wk, Wv)
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    def concat(k, q):
        return _concat_cols(weights, k, q)

    fused = _matmul(X, concat, _K)

    claims = []
    for third, w in enumerate(weights):
        original = _matmul(X, w, _K)
        for row in range(_K):
            for offset in range(_N):
                split_read = fused(row, third * _N + offset)
                claims.append(consumer(split_read) == consumer(original(row, offset)))
    prove(z3.And(*claims))


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _qkv_model(k=_K, n=_N, opset=13, ir_version=10):
    rng = np.random.default_rng(0)
    Wq = rng.standard_normal((k, n))
    Wk = rng.standard_normal((k, n))
    Wv = rng.standard_normal((k, n))
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        g (float[4,{k}] RawX) => (float[4,{n}] Q, float[4,{n}] Kk, float[4,{n}] V)
        {{
          X = Relu(RawX)
          Q = MatMul(X, Wq)
          Kk = MatMul(X, Wk)
          V = MatMul(X, Wv)
        }}
        """
    )
    model.graph.initializer.extend([_f32(Wq, "Wq"), _f32(Wk, "Wk"), _f32(Wv, "Wv")])
    return model


def test_fuse_qkv_pass_matches_basic_case():
    # X is NOT a graph input directly -- it's Relu(RawX) -- so the
    # graph-input decline guard doesn't trip and the fuse actually happens.
    # skip_constant_folding=True bypasses simplify_isolated (which only
    # controls the *optimizer pass* list): onnxsim's constant folding is a
    # separate step that always runs regardless of skipped_optimizers, and
    # since Wq/Wk/Wv are all constants it would otherwise fold the new
    # Concat node away into a single initializer before this test could
    # inspect it directly (confirmed empirically).
    model = _qkv_model()
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("fuse_qkv"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"

    nodes = list(sim_model.graph.node)
    by_op = {}
    for n in nodes:
        by_op.setdefault(n.op_type, []).append(n)
    assert len(by_op["Concat"]) == 1
    assert len(by_op["Split"]) == 1
    assert len(by_op["MatMul"]) == 3  # 1 fused + 2 dangling originals (Kk, V)

    cat = by_op["Concat"][0]
    assert list(cat.input) == ["Wq", "Wk", "Wv"]
    assert [a.i for a in cat.attribute if a.name == "axis"] == [1]  # sizes()-1

    fused_matmul = next(n for n in by_op["MatMul"] if cat.output[0] in n.input)
    assert list(fused_matmul.input) == ["X", cat.output[0]]

    split = by_op["Split"][0]
    assert list(split.input[:1]) == [fused_matmul.output[0]]
    assert [a.i for a in split.attribute if a.name == "axis"] == [-1]
    split_sizes_init = next(
        onnx.numpy_helper.to_array(i)
        for i in sim_model.graph.initializer
        if i.name == split.input[1]
    )
    np.testing.assert_array_equal(split_sizes_init, [_N, _N, _N])

    # Q/K/V's original consumers now read Split's 3 outputs, in that order.
    assert list(split.output) == ["Q", "Kk", "V"]

    # The two ORIGINAL MatMul nodes that are not the matched node `n`
    # survive, dangling: still reading X and their own constant weight, but
    # producing a value nothing (no node, no graph output) consumes.
    dangling = [n for n in by_op["MatMul"] if n is not fused_matmul]
    assert len(dangling) == 2
    dangling_weights = {n.input[1] for n in dangling}
    assert dangling_weights == {"Wk", "Wv"}
    all_outputs_used = {inp for n in nodes for inp in n.input} | set(
        sim_model.graph.output[i].name for i in range(len(sim_model.graph.output))
    )
    for n in dangling:
        assert n.output[0] not in all_outputs_used, "dangling MatMul should be dead"

    # The matched-and-destroyed node was the Q-producing original MatMul:
    # patternMatchPredicate visits nodes in graph order (Q, Kk, V), and Q's
    # original MatMul is the one no longer present at all (not even
    # dangling) -- confirmed by it being neither the fused MatMul nor one
    # of the 2 survivors above.
    assert fused_matmul.input[1] == cat.output[0]


def test_fuse_qkv_declines_when_shared_input_is_a_graph_input():
    # The single most important differential test in this file: X IS the
    # graph's own input directly (no intervening node) feeding 3 MatMuls
    # with constant weights -- exactly the shape that crashes upstream's
    # unfixed original (n->input(0)->node() resolves to the kParam
    # placeholder, which trips insertAfter's ONNX_ASSERT and aborts the
    # entire simplify() pipeline). onnxsim's added guard declines this match
    # gracefully instead: no crash, no fuse, all three MatMuls survive
    # untouched.
    rng = np.random.default_rng(0)
    Wq = rng.standard_normal((_K, _N))
    Wk = rng.standard_normal((_K, _N))
    Wv = rng.standard_normal((_K, _N))
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,{_K}] X) => (float[4,{_N}] Q, float[4,{_N}] Kk, float[4,{_N}] V)
        {{
          Q = MatMul(X, Wq)
          Kk = MatMul(X, Wk)
          V = MatMul(X, Wv)
        }}
        """
    )
    model.graph.initializer.extend([_f32(Wq, "Wq"), _f32(Wk, "Wk"), _f32(Wv, "Wv")])
    # No crash -- this call completing at all is the point of this test.
    sim_model, ops = simplify_isolated(model, "fuse_qkv")
    assert ops["MatMul"] == 3
    assert ops["Concat"] == 0
    assert ops["Split"] == 0
    assert [n.input[1] for n in sim_model.graph.node if n.op_type == "MatMul"] == [
        "Wq",
        "Wk",
        "Wv",
    ]


def test_fuse_qkv_declines_on_mismatched_weight_shapes():
    # Wk is [K, N+1] -- not the same shape as Wq/Wk's [K, N] -- so
    # `q_t->sizes() != k_t->sizes()` trips and the pass declines outright.
    rng = np.random.default_rng(0)
    Wq = rng.standard_normal((_K, _N))
    Wk = rng.standard_normal((_K, _N + 1))
    Wv = rng.standard_normal((_K, _N))
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,{_K}] RawX) => (float[4,{_N}] Q, float[4,{_N + 1}] Kk, float[4,{_N}] V)
        {{
          X = Relu(RawX)
          Q = MatMul(X, Wq)
          Kk = MatMul(X, Wk)
          V = MatMul(X, Wv)
        }}
        """
    )
    model.graph.initializer.extend([_f32(Wq, "Wq"), _f32(Wk, "Wk"), _f32(Wv, "Wv")])
    _, ops = simplify_isolated(model, "fuse_qkv")
    assert ops["MatMul"] == 3
    assert ops["Concat"] == 0
    assert ops["Split"] == 0


def test_fuse_qkv_declines_when_a_weight_is_not_constant():
    # Kk's "weight" is a runtime graph input, not a constant -- fails
    # IsConstantTensor(use.user, 1), so the pass declines.
    rng = np.random.default_rng(0)
    Wq = rng.standard_normal((_K, _N))
    Wv = rng.standard_normal((_K, _N))
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,{_K}] RawX, float[{_K},{_N}] WkRuntime)
            => (float[4,{_N}] Q, float[4,{_N}] Kk, float[4,{_N}] V)
        {{
          X = Relu(RawX)
          Q = MatMul(X, Wq)
          Kk = MatMul(X, WkRuntime)
          V = MatMul(X, Wv)
        }}
        """
    )
    model.graph.initializer.extend([_f32(Wq, "Wq"), _f32(Wv, "Wv")])
    _, ops = simplify_isolated(model, "fuse_qkv")
    assert ops["MatMul"] == 3
    assert ops["Concat"] == 0
    assert ops["Split"] == 0


def test_fuse_qkv_declines_when_a_use_output_has_more_than_one_consumer():
    # Kk also feeds a second graph output (via Identity) -- Kk's own output
    # has 2 uses, tripping `use.user->output()->uses().size() != 1`, so the
    # pass declines.
    model = _qkv_model()
    y_out = onnx.helper.make_tensor_value_info("Kk2", onnx.TensorProto.FLOAT, [4, _N])
    model.graph.output.append(y_out)
    model.graph.node.append(onnx.helper.make_node("Identity", ["Kk"], ["Kk2"]))
    _, ops = simplify_isolated(model, "fuse_qkv")
    assert ops["MatMul"] == 3
    assert ops["Concat"] == 0
    assert ops["Split"] == 0


def test_fuse_qkv_pass_matches_pre_opset13_split_attribute_form():
    # Same basic-case fuse, but on an opset < 13 model: the new Split's
    # sizes are carried as a `split` INTS attribute instead of a second
    # (initializer) input -- the other half of runTransform's
    # `opset_version >= 13` branch.
    model = _qkv_model(opset=11, ir_version=8)
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=3,
        skipped_optimizers=isolate("fuse_qkv"),
        skip_constant_folding=True,
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"

    split = next(n for n in sim_model.graph.node if n.op_type == "Split")
    assert len(split.input) == 1  # no sizes-initializer input on this opset
    split_attrs = {a.name: list(a.ints) for a in split.attribute if a.name == "split"}
    assert split_attrs["split"] == [_N, _N, _N]
    assert list(split.output) == ["Q", "Kk", "V"]
