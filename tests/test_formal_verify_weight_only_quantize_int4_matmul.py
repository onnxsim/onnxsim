"""Formal check for WeightOnlyQuantizeInt4MatMul (opt-in; onnxsim's own
``onnxsim/passes/weight_only_quantize_int4_matmul.h``): the same "only the
constant weight is quantized, the activation ``X`` is left completely
untouched, no calibration data" design as
``weight_only_quantize_matmul.h`` (``test_formal_verify_weight_only_
quantize_matmul.py`` -- READ THAT FILE FIRST, this one only documents what
differs), narrowed to INT4 and made BLOCK-WISE. It rewrites ``Y = MatMul(X,
W)`` (or a "vanilla" ``Gemm``, bias left untouched) -- ``W`` a constant 2-D
FLOAT32 tensor whose reduction dimension ``K`` is evenly divisible by
``kBlockSize = 32``, ``X`` FLOAT32 -- into::

    Wq, Ws := block-wise symmetric INT4 quantization of W (computed ONCE, at
              pass-transform time, from W's static values) -- a SEPARATE
              scale per (block-of-K, output-channel) pair, not one scale per
              whole output channel
    Wdq = DequantizeLinear(Wq, Ws, axis=<reduction_axis>, block_size=32)
    Y   = MatMul(X, Wdq)

Two genuinely new wrinkles vs. the INT8 whole-channel pass:

1. INT4, not INT8: ``Wq``'s values live in ``[-7, 7]``, not ``[-127, 127]``,
   confirmed by reading ``TryQuantizeWeightBlockwiseInt4InPlace``
   (``quantize_matmul_common.h``) directly. Its formula, per (block, output
   channel) pair::

       scale[block, n] = max(|W[k, n]| for k in that block) / 7
                          (or 1.0 for an all-zero block, so no scale is 0)
       Wq[k, n] = clip(round(W[k, n] / scale[block_of(k), n]), -7, 7)

   -- exactly ``QuantizeWeightPerChannelInPlace``'s per-channel scheme
   (``/ 127``, whole-channel scale) with the max/round/clip taken over one
   block of ``K`` at a time instead of the whole channel, and ``/ 7`` instead
   of ``/ 127``. Because no test weight below is built to force clipping, the
   narrower ``[-7, 7]`` range changes nothing about the round-trip bound's
   *shape* -- ``|W - Wdq| <= scale / 2`` holds regardless of the code's own
   bit width, as it is a property of round-to-nearest, not of the range's
   width, as long as no clipping occurs.

2. Block-wise granularity: ``Ws`` has shape ``[K / 32, N]`` (or ``[N, K /
   32]`` for Gemm's ``transB=1`` layout) -- one scale per 32-element block of
   the reduction dimension, per output channel -- not a flat ``[N]`` the
   INT8 weight-only pass produces. Confirmed via the pass's C++ (this is the
   part worth re-reading carefully, since it differs from EVERY prior
   ``DequantizeLinear``-based pass in this suite): the ``axis`` attribute
   this pass sets on ``DequantizeLinear`` is ``reduction_axis`` --
   ``wdq->i_(kaxis, reduction_axis)`` -- i.e. the axis the BLOCKING happens
   along, NOT the output-channel axis every whole-channel
   ``DequantizeLinear``-based pass in this suite (``weight_only_quantize_
   matmul``, ``static_quantize_matmul``, ...) sets ``axis`` to. For a plain
   MatMul (``W`` stored ``[K, N]``, MatMul's native layout, not transposed)
   that is axis 0; for Gemm's ``transB=1`` layout (``W`` stored ``[N, K]``)
   that is axis 1 -- exactly backwards from ``weight_only_quantize_matmul``'s
   own ``channel_axis`` convention (0 / 1 swapped) for the very same two
   node shapes, because that pass's ``axis`` names the channel axis while
   this one's names the reduction axis, and reduction_axis = 1 -
   channel_axis.

This is still a *single-operand* special case of ``quantized_mac_bound``'s
general MAC bound (``test_formal_verify_quantized_mac_bound.py``), with
``eps_x := 0`` exactly as in ``weight_only_quantize_matmul``'s own proof --
but now with a PER-BLOCK error bound rather than a single per-channel one:
each tap ``k``'s own rounding-error budget is ``Ws[block_of(k), n] / 2``,
which can genuinely DIFFER from one tap to the next depending on which block
it falls in, rather than one shared ``Ws[n] / 2`` for every tap the way the
INT8 pass's bound has it. The derived bound is the same per-tap-error-
bound-times-``|X|`` sum::

    |MatMul(X, W)[i, n] - MatMul(X, Wdq)[i, n]|
        <= sum_k (Ws[block_of(k), n] / 2) * |X[i, k]|

-- proved below with a small concrete ``_K = 4`` split across TWO blocks of
``_BLOCK_SIZE = 2`` (taps 0-1 share one scale ``Ws0``, taps 2-3 share a
genuinely DIFFERENT scale ``Ws1``), using the direct-error-variable ``ew``
idiom (mirroring ``quantized_mac_bound``'s own ``ew``/``ex`` idiom and
``weight_only_quantize_matmul``'s own working formulation): each ``ew[k]``
is bounded by ITS OWN tap's block scale (``Ws0 / 2`` for ``k`` in block 0,
``Ws1 / 2`` for ``k`` in block 1), not a single shared bound, so the block
structure is genuinely exercised -- Z3 explores ``Ws0`` and ``Ws1``
independently, including cases where they differ arbitrarily -- rather than
accidentally collapsing back to the single-scale case. A further test below
confirms that collapse is not merely "untested" but actually UNSOUND: naively
reusing block 0's scale as if it bounded every tap's error (the INT8 pass's
single-``Ws`` shape) is not a theorem once ``Ws1`` may exceed ``Ws0``, which
is exactly why this pass's bound must sum a *per-tap* term rather than
factoring out one shared scale the way the whole-channel pass's bound does.

A bias-variant test (Gemm's bias, left untouched, cancels the same way as in
``weight_only_quantize_matmul``'s own proof) and a negative-control test
(the bound needs the rounding hypotheses at all) round out the Z3 side,
mirroring the INT8 file's own structure. No consumer-composition step is
added, for the same reason ``weight_only_quantize_matmul``'s proof gives:
an arbitrary consumer need not be Lipschitz, so a numeric *bound* (not an
equality) implies nothing about ``|consumer(a) - consumer(b)|`` in general.

Differential tests build a plain float ``MatMul``/``Gemm`` via
``onnx.parser`` (per ``CLAUDE.md``) at opset 21 (INT4 tensors and
``DequantizeLinear``'s ``block_size`` attribute both need it) with ``K = 64``
(exactly two 32-element blocks), run the real pass alone via
``simplify_isolated_extra``, and confirm: the exact ``MatMul(X,
DequantizeLinear(Wq, Ws, axis=<reduction_axis>, block_size=32))`` chain
fires with the expected ``reduction_axis`` (0 for plain MatMul, 1 for Gemm's
``transB=1``); ``Wq``'s values lie in ``[-7, 7]`` and ``Ws``'s SHAPE reflects
``K / 32`` blocks (``[K / 32, N]`` / ``[N, K / 32]``), not a flat ``[N]`` --
itself a meaningful differential confirmation of the block-wise (vs
whole-channel) scheme; ``Wq``/``Ws`` match an independent numpy
re-implementation of ``TryQuantizeWeightBlockwiseInt4InPlace``'s formula
above; ``X`` is passed through completely unchanged; the actual runtime
output (via onnxruntime, ``onnxsim.quantize_weight_only_int4``) stays within
the per-block bound proved above; a ``K`` not divisible by ``kBlockSize = 32``
(e.g. 50) declines outright; and a pre-opset-21 model declines outright
(INT4 tensors / ``block_size`` both need opset >= 21, unlike the INT8 pass's
opset >= 13).

INT4-packing note: ONNX's wire format for the INT4 tensor type packs two
4-bit codes per byte in ``raw_data`` (confirmed by reading
``TryQuantizeWeightBlockwiseInt4InPlace``'s own packing comment and code).
This was checked empirically against the onnx version installed here (see
the differential tests below, which read ``Wq`` straight off the pass's own
initializer): ``onnx.numpy_helper.to_array`` unpacks a raw-data-packed INT4
tensor correctly with no extra handling needed -- no wrinkle to work around,
just something to confirm rather than assume, per the task's own framing of
this file as having unusually large investigative surface.

The firing/bound tests below pass ``check_n=0`` to ``simplify_isolated_extra``
for the same reason ``weight_only_quantize_matmul``'s tests do: this is a
genuinely lossy INT4 rewrite, coarser than the INT8 one, so onnxsim's own
random-input equivalence check (default ``check_n=3``, tolerance
``rtol=1e-4``/``atol=1e-5``) does not apply -- confirmed empirically.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 4  # concrete number of contraction taps, split across two blocks below
# -- enough to exercise a genuine cross-block difference while keeping the Z3
# query small (matches this suite's usual small-_K convention, e.g.
# quantized_mac_bound's own _K = 2; doubled here since a single block would
# not exercise the block-wise structure this pass's proof is actually about).
_BLOCK_SIZE = 2  # so taps {0, 1} are one block, {2, 3} a second, different one
_NUM_BLOCKS = _K // _BLOCK_SIZE


def _block_of(k):
    return k // _BLOCK_SIZE


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _bound_formulas():
    """Builds the Z3 vocabulary for the single-operand, PER-BLOCK
    bounded-error claim: ``X`` has no error term at all (never quantized),
    only ``W`` does, via the free per-tap error variable ``ew`` (mirroring
    ``quantized_mac_bound``'s own ``ew``/``ex`` idiom). Unlike
    ``weight_only_quantize_matmul``'s single shared ``Ws``, there is one
    scale variable PER BLOCK (``Ws[b]``), and each tap's own rounding bound
    is its OWN block's scale, not one shared scale for every tap. Returns
    ``(float_matmul, dequant_matmul, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k, n] - Wdq[k, n]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]  # per-block scale Ws[block, n]

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        # Each tap's error is bounded by ITS OWN block's scale -- Ws0 for
        # taps 0-1, a genuinely different Ws1 for taps 2-3 -- not one shared
        # bound for every tap.
        *[_abs(ew[k]) <= Ws[_block_of(k)] / 2 for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = sum((Ws[_block_of(k)] / 2) * _abs(X[k]) for k in range(_K))

    return float_matmul, dequant_matmul, rounding_bounds, bound


def test_weight_only_quantize_int4_matmul_error_is_bounded():
    # The genuine bounded-error claim: given W's own per-BLOCK rounding
    # bound (|W[k, n] - Wdq[k, n]| <= Ws[block_of(k), n] / 2 -- the scale
    # that applies to element k depends on WHICH BLOCK k falls in) and X
    # completely unchanged, the true float dot product and the one computed
    # against the dequantized weight cannot differ by more than
    # sum_k (Ws[block_of(k), n] / 2) * |X[i, k]| -- quantized_mac_bound's own
    # bound with eps_x fixed to 0, now with a genuinely per-tap eps_w term.
    # Ws0 and Ws1 are free and independent here, so this holds even when the
    # two blocks' scales differ arbitrarily -- the block structure is
    # genuinely exercised, not accidentally collapsed to a single scale.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_weight_only_quantize_int4_matmul_bias_variant_error_is_bounded():
    # The "+ Bias" branch (a Gemm with a bias input): this pass never
    # touches Gemm's bias C at all, so adding the same Bias(n) to both the
    # true and the dequantized computation leaves their difference -- and
    # therefore the bound on it -- unchanged; Bias cancels out of the error
    # term algebraically, exactly as in weight_only_quantize_matmul's proof.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_weight_only_quantize_int4_matmul_negative_control_requires_rounding_bound():
    # Sanity check that the bound proved above is genuine, not vacuous: with
    # no error budget assumed on ew at all (only Ws0, Ws1 > 0), the same
    # bound is not a theorem -- Z3 must find a real counterexample.
    float_matmul, dequant_matmul, _rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul

    solver = z3.Solver()
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any rounding-error budget on ew -- "
        "negative control is vacuous"
    )


def test_weight_only_quantize_int4_matmul_uniform_bound_is_unsound_across_blocks():
    # New content this pass's proof needs that weight_only_quantize_matmul's
    # single-scale bound does not: confirms the per-block sum is not merely
    # untested-but-equivalent to a single shared scale -- it is genuinely
    # NECESSARY. If one instead (incorrectly) bounded every tap's error by
    # block 0's scale alone (Ws0 / 2), as the INT8 pass's single-Ws bound
    # would effectively do were it applied here, that claim is NOT a theorem
    # once block 1's real scale Ws1 can exceed Ws0 -- Z3 finds a
    # counterexample where block 1's actual rounding error (up to Ws1 / 2)
    # overflows the too-small Ws0-only budget. This is exactly why the sound
    # bound above must sum a per-tap term (Ws[block_of(k), n] / 2) rather
    # than factoring out one shared scale.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= Ws[_block_of(k)] / 2 for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    error = float_matmul - dequant_matmul

    # The naive/wrong claim: bound every tap's contribution using ONLY
    # block 0's scale Ws0, as if it were the single shared scale the INT8
    # pass's bound has.
    uniform_bound = (Ws[0] / 2) * sum(_abs(X[k]) for k in range(_K))

    solver = z3.Solver()
    solver.add(rounding_bounds)
    solver.add(z3.Not(z3.And(error <= uniform_bound, -error <= uniform_bound)))
    assert solver.check() == z3.sat, (
        "the naive single-scale (block 0 only) bound holds even though "
        "block 1 has its own, potentially larger scale -- this pass's "
        "per-block sum is not actually load-bearing, which would be wrong"
    )


def test_weight_only_quantize_int4_matmul_dequantizelinear_block_axis_semantics_matches_rounding_bound():
    # New content this pass's proof needs that weight_only_quantize_matmul's
    # does not: that pass's DequantizeLinear axis names the output-channel
    # axis (one scale per whole channel); this pass's axis names the
    # REDUCTION axis, with block_size grouping consecutive reduction indices
    # under one shared scale. Confirms DequantizeLinear's own blocked-axis
    # formula -- elements sharing a block index share one scale, a different
    # block gets its own independent scale -- is exactly what feeds the
    # rounding bound above, using three elements: k=0 and k=1 (same block,
    # scale ws_b0) and k=2 (a different block, its own scale ws_b1).
    w0, w1, w2, ws_b0, ws_b1 = z3.Reals("w0 w1 w2 ws_b0 ws_b1")
    wq0, wq1, wq2 = z3.Ints("wq0 wq1 wq2")  # round(w / ws): integer within 0.5
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        ws_b0 > 0,
        ws_b1 > 0,
        # k=0, k=1: SAME block (block_size groups them) -> SAME scale ws_b0.
        wq0 - w0 / ws_b0 <= half,
        w0 / ws_b0 - wq0 <= half,
        wq1 - w1 / ws_b0 <= half,
        w1 / ws_b0 - wq1 <= half,
        # k=2: a DIFFERENT block -> its own, independent scale ws_b1.
        wq2 - w2 / ws_b1 <= half,
        w2 / ws_b1 - wq2 <= half,
    )
    # DequantizeLinear(Wq, Ws, axis=<reduction axis>, block_size=...)'s own
    # defining formula for one element: Wdq[k] = Wq[k] * Ws[block_of(k)],
    # zero_point implicit 0 (symmetric).
    wdq0 = z3.ToReal(wq0) * ws_b0
    wdq1 = z3.ToReal(wq1) * ws_b0
    wdq2 = z3.ToReal(wq2) * ws_b1

    e0, e1, e2 = w0 - wdq0, w1 - wdq1, w2 - wdq2
    prove(
        z3.Implies(
            hypotheses,
            z3.And(
                e0 <= ws_b0 / 2,
                -e0 <= ws_b0 / 2,
                e1 <= ws_b0 / 2,
                -e1 <= ws_b0 / 2,
                e2 <= ws_b1 / 2,
                -e2 <= ws_b1 / 2,
            ),
        )
    )


def _model(body, initializer=(), opset=21, ir_version=10):
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
    return numpy_helper.from_array(array.astype(np.float32), name)


def _dequantizelinear_node(model, output_name):
    node = producer(model, output_name)
    assert node.op_type == "DequantizeLinear"
    return node


def _int_attr(node, name):
    return next(a.i for a in node.attribute if a.name == name)


def _quantize_weight_blockwise_int4_in_place(weight, channel_axis, block_size):
    """Independent numpy re-implementation of
    ``TryQuantizeWeightBlockwiseInt4InPlace`` (quantize_matmul_common.h):
    per-(block-of-reduction-axis, channel) symmetric INT4 quantization *in
    weight's own layout* (no transpose) -- scale = max(|block|) / 7 (or 1.0
    for an all-zero block), codes = round(w / scale) clipped to [-7, 7].
    """
    reduction_axis = 1 - channel_axis
    k = weight.shape[reduction_axis]
    assert k % block_size == 0
    num_blocks = k // block_size

    # Move the reduction axis to axis 0 to compute per-block scales simply,
    # then move it back.
    w_r = np.moveaxis(weight, reduction_axis, 0)  # [K, C]
    scale = np.stack(
        [
            np.max(np.abs(w_r[b * block_size : (b + 1) * block_size]), axis=0)
            for b in range(num_blocks)
        ],
        axis=0,
    )  # [num_blocks, C]
    scale = np.where(scale > 0, scale / 7.0, 1.0).astype(np.float32)

    scale_bcast_r = np.repeat(scale, block_size, axis=0)  # [K, C]
    codes_r = np.clip(np.round(w_r / scale_bcast_r), -7, 7).astype(np.int8)

    codes = np.moveaxis(codes_r, 0, reduction_axis)
    scale_out = np.moveaxis(scale, 0, reduction_axis)
    return codes, scale_out


def test_weight_only_quantize_int4_matmul_pass_fires_and_matches_scheme():
    # Build a plain float MatMul (K=64, exactly two 32-element blocks) and
    # run the real weight_only_quantize_int4_matmul rewrite alone, then
    # confirm both the node chain's shape (MatMul(X, DequantizeLinear(Wq,
    # Ws, axis=0, block_size=32))) and the quantized weight's exact values.
    rng = np.random.default_rng(0)
    rows, K, N = 4, 64, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    # check_n=0: see the module docstring for why the built-in random-input
    # check does not apply to a genuinely lossy INT4 rewrite.
    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int4_matmul", check_n=0
    )

    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    x_input, w_input = matmul_node.input
    # X (the activation) is passed through completely unchanged.
    assert x_input == "X"

    dql_node = _dequantizelinear_node(sim_model, w_input)
    # Plain MatMul: W stored [K, N] (not transposed), channel_axis=1, so the
    # REDUCTION axis (this pass's own `axis` attribute) is 0 -- unlike
    # weight_only_quantize_matmul, whose `axis` names the channel axis (1)
    # for this exact same node shape.
    assert _int_attr(dql_node, "axis") == 0
    assert _int_attr(dql_node, "block_size") == 32

    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in sim_model.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in sim_model.graph.initializer if i.name == ws_name)
    wq = numpy_helper.to_array(wq_init)
    ws = numpy_helper.to_array(ws_init)
    assert wq.shape == (K, N)
    # Block-wise, not whole-channel: K/32 = 2 blocks per output channel, not
    # a flat [N] the INT8 weight-only pass would produce here.
    assert ws.shape == (K // 32, N)
    assert wq.min() >= -7
    assert wq.max() <= 7

    expected_wq, expected_ws = _quantize_weight_blockwise_int4_in_place(weight, 1, 32)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_int4_matmul_gemm_transb_uses_reduction_axis_one():
    # PyTorch nn.Linear layout: weight [N, K], Gemm(X, W, B, transB=1). Per
    # the header comment, channel_axis=0 here, so the REDUCTION axis (this
    # pass's `axis` attribute) is 1 -- exactly backwards from
    # weight_only_quantize_matmul's channel_axis=0 for this same node shape,
    # since that pass's axis names the channel axis while this one's names
    # the reduction axis.
    rng = np.random.default_rng(1)
    rows, K, N = 3, 64, 2
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32)
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int4_matmul", check_n=0
    )

    gemm_node = producer(sim_model, "Y")
    assert gemm_node.op_type == "Gemm"
    x_input, w_input, b_input = gemm_node.input
    assert x_input == "X"
    assert b_input == "B"  # bias untouched

    dql_node = _dequantizelinear_node(sim_model, w_input)
    assert _int_attr(dql_node, "axis") == 1
    assert _int_attr(dql_node, "block_size") == 32

    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in sim_model.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in sim_model.graph.initializer if i.name == ws_name)
    wq = numpy_helper.to_array(wq_init)
    ws = numpy_helper.to_array(ws_init)
    assert wq.shape == (N, K)
    assert ws.shape == (N, K // 32)
    assert wq.min() >= -7
    assert wq.max() <= 7

    expected_wq, expected_ws = _quantize_weight_blockwise_int4_in_place(weight, 0, 32)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_int4_matmul_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring weight_only_quantize_matmul's own
    # analogous test: run the real quantized graph through onnxruntime and
    # confirm every output element's error against the true float MatMul
    # stays within the per-block bound the proof above derives -- Ws read
    # directly from the static initializer, block index computed per k.
    rng = np.random.default_rng(2)
    rows, K, N = 4, 64, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.8
    x = rng.standard_normal((rows, K)).astype(np.float32) * 2.0
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.quantize_weight_only_int4(model)
    dql_node = next(n for n in quantized.graph.node if n.op_type == "DequantizeLinear")
    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in quantized.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in quantized.graph.initializer if i.name == ws_name)
    ws = numpy_helper.to_array(ws_init)  # [K/32, N]
    assert numpy_helper.to_array(wq_init).shape == (K, N)
    assert ws.shape == (K // 32, N)

    sess = ort.InferenceSession(quantized.SerializeToString())
    (y_quant,) = sess.run(["Y"], {"X": x})

    y_float = x @ weight
    error = np.abs(y_float - y_quant)

    eps_w = ws / 2.0  # shape [K/32, N]: eps_w[block_of(k), n]
    block_of_k = np.arange(K) // 32  # [K]
    # bound[i, n] = sum_k eps_w[block_of(k), n] * |x[i, k]|
    per_tap_eps = eps_w[block_of_k, :]  # [K, N]
    bound = np.einsum("ik,kn->in", np.abs(x), per_tap_eps)
    assert np.all(error <= bound + 1e-5)

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs: the actual
    # error should sit well inside the proved worst-case bound, not merely
    # avoid violating it. INT4's much narrower [-7, 7] code range (vs
    # INT8's [-127, 127]) is meaningfully coarser per weight element even
    # with block-wise scales -- confirmed empirically that a fixed
    # rtol/atol as tight as weight_only_quantize_matmul's own analogous
    # check (0.05 / 1e-2) fails here -- so this compares against the
    # proved bound itself rather than picking a new ad hoc tolerance.
    assert np.mean(error) < 0.5 * np.mean(bound)


def test_weight_only_quantize_int4_matmul_declines_when_k_not_divisible_by_block_size():
    # TryQuantizeWeightBlockwiseInt4InPlace (and this pass's own
    # patternMatchPredicate) requires K % kBlockSize == 0 (kBlockSize=32); a
    # ragged last block is explicitly left unhandled. K=50 must survive
    # untouched.
    rng = np.random.default_rng(3)
    rows, K, N = 4, 50, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int4_matmul", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["MatMul"]


def test_weight_only_quantize_int4_matmul_declines_pre_opset21():
    # INT4 tensors and DequantizeLinear's `block_size` attribute both need
    # opset >= 21 (patternMatchPredicate's own check) -- unlike
    # weight_only_quantize_matmul's opset >= 13. A plain MatMul at opset 20
    # must survive untouched even though K is divisible by 32.
    rng = np.random.default_rng(4)
    rows, K, N = 4, 64, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=20,
        ir_version=9,
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int4_matmul", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["MatMul"]
