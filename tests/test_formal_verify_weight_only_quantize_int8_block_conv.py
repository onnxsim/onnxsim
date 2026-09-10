"""Formal check for WeightOnlyQuantizeInt8BlockConv (opt-in; onnxsim's own
``onnxsim/passes/weight_only_quantize_int8_block_conv.h``): the Conv-shaped
sibling of ``weight_only_quantize_int8_block_matmul.h``, related to it
exactly the way ``weight_only_quantize_int4_conv.h``
(``test_formal_verify_weight_only_quantize_int4_conv.py``, this file's
direct structural template -- READ THAT FILE FIRST, this one only documents
what differs) relates to ``weight_only_quantize_int4_matmul.h``. It rewrites
``Y = Conv(X, W)`` (optional bias, a third input, left completely untouched)
-- ``W`` a constant FLOAT32 tensor, rank >= 3 (``[Cout, Cin/groups, k...]``),
``X`` FLOAT32, whose flattened ``inner = Cin/groups * prod(k...)`` is evenly
divisible by ``kBlockSize = 32`` -- into::

    Wq_flat, Ws_flat := block-wise symmetric INT8 quantization (computed
        ONCE, at pass-transform time) of W FIRST RESHAPED to [Cout, inner]
        (TryQuantizeConvWeightBlockwiseInt8Flat, quantize_conv_common.h) --
        a separate scale per (output channel, block-of-inner) pair
    Wdq_flat = DequantizeLinear(Wq_flat, Ws_flat, axis=1, block_size=32)
    Wdq      = Reshape(Wdq_flat, W's ORIGINAL [Cout, Cin/groups, k...] shape)
    Y        = Conv(X, Wdq)

The only real difference from the INT4 Conv template is bit width, not
structure -- exactly the same relationship
``weight_only_quantize_int8_block_matmul.h`` has to
``weight_only_quantize_int4_matmul.h``. Confirmed by reading
``TryQuantizeConvWeightBlockwiseInt8Flat`` directly
(``quantize_conv_common.h``) rather than assumed from the INT4 file's
formula: it is *precisely* ``TryQuantizeConvWeightBlockwiseInt4Flat``'s own
flatten-to-``[Cout, inner]``-then-block-that scheme, with ``/ 7`` replaced by
``/ 127`` and the clip range widened from ``[-7, 7]`` to ``[-127, 127]``::

    scale[c, block] = max(|W_flat[c, j]| for j in that block) / 127
                       (or 1.0 for an all-zero block, so no scale is 0)
    Wq_flat[c, j] = clip(round(W_flat[c, j] / scale[c, block_of(j)]), -127, 127)

-- symmetric (no zero point: ``DequantizeLinear`` is called with only
``Wq_flat``/``Ws_flat``, no third ``zero_points`` input, so ORT's own
implicit default of 0 applies), matching this suite's every other
``DequantizeLinear``-based weight-only pass. ``Wq_flat``/``Ws_flat`` are
genuinely 2-D (``[Cout, inner]`` / ``[Cout, inner / 32]``) even when ``W``
itself is rank > 2, and the caller reshapes ``DequantizeLinear``'s 2-D
output back to ``W``'s original shape via an explicit ``Reshape`` node --
the same one node neither the plain (per-channel, unblocked)
``weight_only_quantize_conv.h`` INT8 pass nor
``weight_only_quantize_int8_block_matmul.h`` (MatMul/Gemm's weight is
already 2-D end to end) ever need.

Axis/block_size semantics, confirmed from the C++ (``wdq->i_(kaxis, 1);
wdq->i_(Symbol("block_size"), kBlockSize);``), not assumed to match
``weight_only_quantize_int8_block_matmul.h``'s own convention verbatim:
that MatMul/Gemm pass sets ``DequantizeLinear``'s ``axis`` to
``reduction_axis`` -- which axis that is *depends on Gemm's ``transB``*
(0 for a plain ``[K, N]`` weight, 1 for a transposed ``[N, K]`` one), because
MatMul/Gemm's weight has a genuine channel-vs-reduction AXIS CHOICE Conv's
layout never offers. Conv's weight ``[Cout, Cin/groups, k...]`` always puts
the output channel on axis 0 UNCONDITIONALLY (no ``transB``-like knob
exists for Conv), so this pass never needs a ``channel_axis`` parameter at
all -- it always flattens to ``[Cout, inner]`` first (output channel pinned
to axis 0 of the FLAT tensor) and then blocks ``axis=1`` (the flattened
``inner`` axis) UNCONDITIONALLY, always. The Reshape immediately after
restores ``W``'s original multi-axis shape, at which point "axis 0 is Cout"
is once again exactly true of the tensor Conv itself consumes, same as
before the rewrite ever ran.

This is still a *single-operand* special case of ``quantized_mac_bound``'s
general MAC bound (``eps_x := 0``, ``X`` never touched), with the same
per-tap-differs-by-block structure ``weight_only_quantize_int4_conv``'s own
proof establishes: each flattened tap ``j``'s own rounding-error budget is
``Ws_flat[c, block_of(j)] / 2``, which can genuinely differ from one tap to
the next depending on which block it falls in::

    |Conv(X, W)[..., c, ...] - Conv(X, Wdq)[..., c, ...]|
        <= sum_j (Ws_flat[c, block_of(j)] / 2) * |X_patch_flat[j]|

-- the round-to-nearest bound ``|W - Wdq| <= scale / 2`` holds regardless of
the code's own bit width (INT8's wider ``[-127, 127]`` vs. INT4's
``[-7, 7]``) as long as no clipping occurs, so the bound's *shape* is
identical to the INT4 Conv file's; only the concrete code range differs, and
that range plays no role in the Z3 formulas below (which never mention 7 or
127 at all -- ``ew``'s bound is stated purely in terms of the block's own
scale). The Z3 vocabulary and proofs below are therefore reused near-
verbatim from ``weight_only_quantize_int4_conv``'s own five proofs, EXCEPT
sized down to ``_K = 2`` split across two SEPARATE one-element blocks
(``_BLOCK_SIZE = 1``) rather than that file's ``_K = 4`` / ``_BLOCK_SIZE =
2`` -- matching ``test_formal_verify_weight_only_quantize_matmul_nbits.py``'s
own ``_BLOCK_SIZE = 1`` idiom for exercising genuinely independent per-block
scales with the smallest case that does so (two one-tap blocks is already
the minimal shape where "taps do NOT all share one scale" is exercised).
Every bound-proving query below uses the DIRECT-ERROR-VARIABLE idiom (a free
``ew`` bounded directly by the rounding hypothesis, not reconstructed from
separate code/scale multiplicands in the same query) -- see
``test_formal_verify_dynamic_quantize_matmul.py``'s own module docstring for
why a combined nonlinear reconstruction is a documented Z3 hang risk; the
one query below that DOES reconstruct from an integer code
(the axis-semantics test) is a small, non-summed identity query, the same
shape every other weight-only pass's own analogous "does
``DequantizeLinear``'s literal semantics match what the bound assumes" test
already uses safely.

A bias-variant test (Conv's own optional bias, untouched, cancels
algebraically the same way as every prior pass in this family), a negative-
control test (the bound needs the rounding hypotheses at all), and a
"naive single-shared-scale bound is unsound across blocks" negative control
(mirroring ``weight_only_quantize_int4_matmul``/``_conv`` and
``weight_only_quantize_matmul_nbits``'s own versions of this check) round
out the Z3 side. No consumer-composition step is added, for the same reason
every template file gives: an arbitrary consumer need not be Lipschitz, so a
numeric *bound* (not an equality) implies nothing about
``|consumer(a) - consumer(b)|`` in general.

UNLIKE ``weight_only_quantize_matmul_nbits.h``, this pass has NO ragged-block
handling at all: confirmed directly from ``patternMatchPredicate`` --
``return InnerSize(w_t->sizes()) % kBlockSize == 0;`` -- a hard boolean
gate with no "pad the last block" fallback anywhere in the file (unlike
``QuantizeWeightForMatMulNBits``'s ``ceil(K / block_size)`` and fixed dead-
position zero-point code). So there is no ragged-block Z3 lemma to prove
here; instead, the differential tests below include a DEDICATED test
confirming an ``inner`` not divisible by 32 makes the Conv survive
completely untouched (not merely "quantized differently").

Differential tests build a plain float ``Conv`` (rank >= 3 weight) via
``onnx.parser`` (per ``CLAUDE.md``) at opset 21 (``DequantizeLinear``'s
``block_size`` attribute needs it -- confirmed from
``patternMatchPredicate``'s own ``opset < 21`` check, the same floor as the
INT4 Conv pass even though plain per-channel INT8 needs only opset 13),
with a flattened reduction size (``Cin/groups * prod(kernel dims)``) that IS
a multiple of 32, run the real pass alone via ``simplify_isolated_extra``
(``"weight_only_quantize_int8_block_conv"``) and, for the full-pipeline
numeric-bound check, via the dedicated Python entry point
``onnxsim.quantize_weight_only_int8_block`` (confirmed by reading
``onnxsim/quantize_entry.cpp``'s ``QuantizeWeightOnlyInt8Block``, which
registers both ``weight_only_quantize_int8_block_matmul`` AND
``weight_only_quantize_int8_block_conv`` together, and
``onnxsim/onnx_simplifier.py``'s wrapper of the same name). Confirmed: the
exact chain ``Conv(X, Reshape(DequantizeLinear(Wq_flat, Ws_flat, axis=1,
block_size=32), shape=W's original shape))`` fires; the ``Reshape``'s shape-
initializer values match ``W``'s original ``[Cout, Cin/groups, k...]`` shape
exactly (not the flattened ``[Cout, inner]`` shape); ``Wq_flat``'s values
lie in ``[-127, 127]`` and ``Ws_flat``'s SHAPE is ``[Cout, inner / 32]`` --
matched against an independent numpy re-implementation of
``TryQuantizeConvWeightBlockwiseInt8Flat``'s own formula above; ``X`` and
any bias pass through completely unchanged; a flattened reduction size NOT
divisible by 32 declines outright (plain ``Conv`` survives untouched, no
quantization at all); a pre-opset-21 model declines outright; and the real
onnxruntime output stays within the per-block bound proved above.

IMPORTANT: for the one differential test that checks a numeric bound
against REAL onnxruntime execution, graph optimization is explicitly
disabled (``SessionOptions.graph_optimization_level =
GraphOptimizationLevel.ORT_DISABLE_ALL``) -- see
``test_ort_matmul_nbits_workaround.py``'s own module docstring for the
precedent: this suite found and fixed a real bug where onnxruntime's
DEFAULT optimization level silently fuses certain quantized-graph shapes
into a hardware-specific fused kernel that is a genuinely different code
path from what these proofs reason about. This pass's own
``DequantizeLinear`` output never reaches a MatMul/Gemm directly (a
``Reshape`` always sits in between, feeding ``Conv``, not ``MatMul``/
``Gemm``), so the specific ``MatMulNBitsFusion`` bug that workaround targets
does not apply here -- but disabling optimization for any such bound-
checking session is this suite's now-established default regardless, not
something to re-derive case by case.

The firing/bound tests below pass ``check_n=0`` to ``simplify_isolated_extra``
for the same reason every pass in this quantization family does: this is a
genuinely lossy INT8-at-block-granularity rewrite, and onnxsim's own
built-in random-input equivalence check does not apply to it (confirmed
empirically by every prior file in this family).
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 2  # concrete number of contraction taps -- matches quantized_mac_bound's
# own _K and weight_only_quantize_matmul_nbits' own idiom (not the larger
# _K = 4 weight_only_quantize_int4_matmul/_conv use): with _BLOCK_SIZE = 1
# below, two SEPARATE one-element blocks is already the minimal case that
# exercises genuinely independent per-block scales.
_BLOCK_SIZE = 1  # each tap is its own block: Ws0 for tap 0, a genuinely
# different Ws1 for tap 1 -- the strongest (smallest) case of "taps do NOT
# all share one scale".
_NUM_BLOCKS = _K // _BLOCK_SIZE


def _block_of(k):
    return k // _BLOCK_SIZE


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _bound_formulas():
    """Builds the Z3 vocabulary for the single-operand, PER-BLOCK
    bounded-error claim: ``X`` has no error term at all (never quantized),
    only ``W`` does, via the free per-tap error variable ``ew`` (the direct-
    error-variable idiom -- see the module docstring for why, not a
    code/scale-multiplicand reconstruction). There is one scale variable PER
    BLOCK (``Ws[b]``), and each tap's own rounding bound is its OWN block's
    scale, not one shared scale for every tap. Returns ``(float_conv,
    dequant_conv, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # one output position's
    # flattened receptive-field patch of X, true float values
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W_flat[c, k], true float
    # weight for output channel c
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W_flat[c, k] - Wdq_flat[c, k]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]  # per-block scale
    # Ws_flat[c, block]

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        # Each tap's error is bounded by ITS OWN block's scale -- Ws0 for
        # tap 0, a genuinely different Ws1 for tap 1 -- not one shared bound.
        *[_abs(ew[k]) <= Ws[_block_of(k)] / 2 for k in range(_K)],
    )

    float_conv = sum(X[k] * W[k] for k in range(_K))
    dequant_conv = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = sum((Ws[_block_of(k)] / 2) * _abs(X[k]) for k in range(_K))

    return float_conv, dequant_conv, rounding_bounds, bound


def test_weight_only_quantize_int8_block_conv_error_is_bounded():
    # The genuine bounded-error claim: given W_flat's own per-BLOCK rounding
    # bound (|W_flat[c, j] - Wdq_flat[c, j]| <= Ws_flat[c, block_of(j)] / 2)
    # and X completely unchanged, the true float dot product (one output
    # element's flattened receptive-field contraction) and the one computed
    # against the dequantized-and-reshaped-back weight cannot differ by more
    # than sum_j (Ws_flat[c, block_of(j)] / 2) * |X_patch_flat[j]| --
    # quantized_mac_bound's own bound with eps_x fixed to 0, with a
    # genuinely per-tap eps_w term. Ws0 and Ws1 are free and independent
    # here, so this holds even when the two blocks' scales differ
    # arbitrarily.
    float_conv, dequant_conv, rounding_bounds, bound = _bound_formulas()
    error = float_conv - dequant_conv
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_weight_only_quantize_int8_block_conv_bias_variant_error_is_bounded():
    # Conv's own optional bias (a third input, a per-output-channel additive
    # term this pass never touches): adding the same Bias(c) to both the
    # true and the dequantized computation leaves their difference -- and
    # therefore the bound on it -- unchanged; Bias cancels out algebraically.
    float_conv, dequant_conv, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_conv + bias) - (dequant_conv + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_weight_only_quantize_int8_block_conv_negative_control_requires_rounding_bound():
    # Sanity check that the bound proved above is genuine, not vacuous: with
    # no error budget assumed on ew at all (only Ws0, Ws1 > 0), the same
    # bound is not a theorem -- Z3 must find a real counterexample.
    float_conv, dequant_conv, _rounding_bounds, bound = _bound_formulas()
    error = float_conv - dequant_conv

    solver = z3.Solver()
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any rounding-error budget on ew -- "
        "negative control is vacuous"
    )


def test_weight_only_quantize_int8_block_conv_uniform_bound_is_unsound_across_blocks():
    # New content this pass's proof needs that weight_only_quantize_conv's
    # single-scale (per-channel, unblocked) bound does not: confirms the
    # per-block sum is not merely untested-but-equivalent to a single shared
    # scale -- it is genuinely NECESSARY. If one instead (incorrectly)
    # bounded every tap's error by block 0's scale alone (Ws0 / 2), that
    # claim is NOT a theorem once block 1's real scale Ws1 can exceed Ws0 --
    # Z3 finds a counterexample where block 1's actual rounding error (up to
    # Ws1 / 2) overflows the too-small Ws0-only budget.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= Ws[_block_of(k)] / 2 for k in range(_K)],
    )

    float_conv = sum(X[k] * W[k] for k in range(_K))
    dequant_conv = sum(X[k] * dequant_w[k] for k in range(_K))
    error = float_conv - dequant_conv

    # The naive/wrong claim: bound every tap's contribution using ONLY
    # block 0's scale Ws0, as if it were a single shared scale.
    uniform_bound = (Ws[0] / 2) * sum(_abs(X[k]) for k in range(_K))

    solver = z3.Solver()
    solver.add(rounding_bounds)
    solver.add(z3.Not(z3.And(error <= uniform_bound, -error <= uniform_bound)))
    assert solver.check() == z3.sat, (
        "the naive single-scale (block 0 only) bound holds even though "
        "block 1 has its own, potentially larger scale -- this pass's "
        "per-block sum is not actually load-bearing, which would be wrong"
    )


def test_weight_only_quantize_int8_block_conv_dequantizelinear_block_axis_semantics_matches_rounding_bound():
    # Confirms DequantizeLinear's own blocked-axis formula -- elements
    # sharing a block index share one scale, a different block gets its own
    # independent scale -- is exactly what feeds the rounding bound above,
    # using three elements: j=0 and j=1 (same block, scale ws_b0) and j=2
    # (a different block, its own scale ws_b1).
    w0, w1, w2, ws_b0, ws_b1 = z3.Reals("w0 w1 w2 ws_b0 ws_b1")
    wq0, wq1, wq2 = z3.Ints("wq0 wq1 wq2")  # round(w / ws): integer within 0.5
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        ws_b0 > 0,
        ws_b1 > 0,
        # j=0, j=1: SAME block (block_size groups them) -> SAME scale ws_b0.
        wq0 - w0 / ws_b0 <= half,
        w0 / ws_b0 - wq0 <= half,
        wq1 - w1 / ws_b0 <= half,
        w1 / ws_b0 - wq1 <= half,
        # j=2: a DIFFERENT block -> its own, independent scale ws_b1.
        wq2 - w2 / ws_b1 <= half,
        w2 / ws_b1 - wq2 <= half,
    )
    # DequantizeLinear(Wq_flat, Ws_flat, axis=1, block_size=...)'s own
    # defining formula for one element: Wdq_flat[c, j] = Wq_flat[c, j] *
    # Ws_flat[c, block_of(j)], zero_point implicit 0 (symmetric).
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


def _reshape_node(model, output_name):
    node = producer(model, output_name)
    assert node.op_type == "Reshape"
    return node


def _int_attr(node, name):
    return next(a.i for a in node.attribute if a.name == name)


def _quantize_conv_weight_blockwise_int8_flat(weight, block_size):
    """Independent numpy re-implementation of
    ``TryQuantizeConvWeightBlockwiseInt8Flat`` (quantize_conv_common.h):
    ``weight`` ([Cout, Cin/groups, k...]) is FIRST reshaped (row-major, via
    plain ``np.reshape`` -- confirmed to match the pass's own row-major flat
    buffer read by the differential tests below) to ``[Cout, inner]``, then
    quantized per-(output channel, block-of-inner) pair: scale =
    max(|block|) / 127 (or 1.0 for an all-zero block), codes = round(w /
    scale) clipped to [-127, 127]. Conv's flattened layout always puts the
    output channel on axis 0 and the block structure along axis 1 -- no
    axis choice to make (unlike weight_only_quantize_int8_block_matmul's
    own analogous helper, which can block along either axis depending on
    Gemm's transB).
    """
    cout = weight.shape[0]
    inner = weight.size // cout
    assert inner % block_size == 0
    num_blocks = inner // block_size

    w_flat = weight.reshape(cout, inner)  # [Cout, inner], row-major
    scale = np.stack(
        [
            np.max(np.abs(w_flat[:, b * block_size : (b + 1) * block_size]), axis=1)
            for b in range(num_blocks)
        ],
        axis=1,
    )  # [Cout, num_blocks]
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)

    scale_bcast = np.repeat(scale, block_size, axis=1)  # [Cout, inner]
    codes = np.clip(np.round(w_flat / scale_bcast), -127, 127).astype(np.int8)
    return codes, scale


def test_weight_only_quantize_int8_block_conv_pass_fires_and_matches_scheme_single_block():
    # Minimal case: Cin=2, kH=4, kW=4 -> inner = 2*4*4 = 32, exactly ONE
    # block. Confirms the flatten-quantize-dequantize-reshape chain fires
    # correctly even in the simplest (single-block) case before the
    # multi-block test below exercises a genuine cross-block difference.
    rng = np.random.default_rng(0)
    cout, cin, kh, kw = 2, 2, 4, 4
    hw = 5
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{hw - kh + 1},{hw - kw + 1}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    # check_n=0: see the module docstring for why the built-in random-input
    # check does not apply to a genuinely lossy quantizing rewrite.
    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int8_block_conv", check_n=0
    )

    conv_node = producer(sim_model, "Y")
    assert conv_node.op_type == "Conv"
    x_input, w_input = conv_node.input
    # X (the activation) is passed through completely unchanged.
    assert x_input == "X"

    reshape_node = _reshape_node(sim_model, w_input)
    wdq_flat_name, shape_name = reshape_node.input
    shape_init = next(i for i in sim_model.graph.initializer if i.name == shape_name)
    # The Reshape's target shape is W's ORIGINAL [Cout, Cin, kH, kW] shape,
    # not the flattened [Cout, inner] shape DequantizeLinear itself produces.
    np.testing.assert_array_equal(
        numpy_helper.to_array(shape_init), np.array(weight.shape, dtype=np.int64)
    )

    dql_node = _dequantizelinear_node(sim_model, wdq_flat_name)
    assert _int_attr(dql_node, "axis") == 1
    assert _int_attr(dql_node, "block_size") == 32

    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in sim_model.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in sim_model.graph.initializer if i.name == ws_name)
    wq = numpy_helper.to_array(wq_init)
    ws = numpy_helper.to_array(ws_init)
    inner = cin * kh * kw
    assert wq.shape == (cout, inner)
    # Exactly one block per output channel here.
    assert ws.shape == (cout, inner // 32)
    assert ws.shape == (cout, 1)
    assert wq.min() >= -127
    assert wq.max() <= 127

    expected_wq, expected_ws = _quantize_conv_weight_blockwise_int8_flat(weight, 32)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_int8_block_conv_pass_fires_and_matches_scheme_two_blocks():
    # Cin=1, kH=8, kW=8 -> inner = 1*8*8 = 64, exactly TWO 32-element blocks
    # -- actually exercises a genuine cross-block scale difference, unlike
    # the single-block case above.
    rng = np.random.default_rng(1)
    cout, cin, kh, kw = 2, 1, 8, 8
    hw = 9
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{hw - kh + 1},{hw - kw + 1}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int8_block_conv", check_n=0
    )

    conv_node = producer(sim_model, "Y")
    assert conv_node.op_type == "Conv"
    x_input, w_input = conv_node.input
    assert x_input == "X"

    reshape_node = _reshape_node(sim_model, w_input)
    wdq_flat_name, shape_name = reshape_node.input
    shape_init = next(i for i in sim_model.graph.initializer if i.name == shape_name)
    np.testing.assert_array_equal(
        numpy_helper.to_array(shape_init), np.array(weight.shape, dtype=np.int64)
    )

    dql_node = _dequantizelinear_node(sim_model, wdq_flat_name)
    assert _int_attr(dql_node, "axis") == 1
    assert _int_attr(dql_node, "block_size") == 32

    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in sim_model.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in sim_model.graph.initializer if i.name == ws_name)
    wq = numpy_helper.to_array(wq_init)
    ws = numpy_helper.to_array(ws_init)
    inner = cin * kh * kw
    assert wq.shape == (cout, inner)
    # Block-wise, not whole-channel: inner/32 = 2 blocks per output channel,
    # not a flat [Cout] the per-channel weight_only_quantize_conv pass would
    # produce here.
    assert ws.shape == (cout, inner // 32)
    assert ws.shape == (cout, 2)
    assert wq.min() >= -127
    assert wq.max() <= 127
    # The two blocks' scales genuinely differ (not merely "two entries that
    # happen to be equal") -- confirms the block structure is exercised, not
    # accidentally degenerate.
    assert not np.allclose(ws[:, 0], ws[:, 1])

    expected_wq, expected_ws = _quantize_conv_weight_blockwise_int8_flat(weight, 32)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_int8_block_conv_bias_and_x_are_untouched():
    # Conv's own optional bias input (input 2) and its activation input
    # (input 0): confirm both survive the rewrite completely unchanged --
    # only the weight input is ever replaced, and only via the
    # Reshape(DequantizeLinear(...)) chain.
    rng = np.random.default_rng(2)
    cout, cin, kh, kw = 2, 1, 8, 8
    hw = 9
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.5
    bias = rng.standard_normal(cout).astype(np.float32)
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{hw - kh + 1},{hw - kw + 1}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int8_block_conv", check_n=0
    )

    conv_node = producer(sim_model, "Y")
    assert conv_node.op_type == "Conv"
    x_input, w_input, b_input = conv_node.input
    assert x_input == "X"
    assert b_input == "B"  # bias untouched, still the ORIGINAL initializer

    b_init = next(i for i in sim_model.graph.initializer if i.name == "B")
    np.testing.assert_array_equal(numpy_helper.to_array(b_init), bias)

    reshape_node = _reshape_node(sim_model, w_input)
    wdq_flat_name, _shape_name = reshape_node.input
    dql_node = _dequantizelinear_node(sim_model, wdq_flat_name)
    assert _int_attr(dql_node, "axis") == 1
    assert _int_attr(dql_node, "block_size") == 32


def test_weight_only_quantize_int8_block_conv_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring every prior file in this family: run the
    # real quantized graph (via onnxsim.quantize_weight_only_int8_block,
    # which applies both the MatMul/Gemm and Conv block-wise INT8 rewrites --
    # confirmed by reading quantize_entry.cpp's QuantizeWeightOnlyInt8Block)
    # through onnxruntime and confirm every output element's error against
    # the true float Conv stays within the per-block bound the proof above
    # derives -- Ws_flat read directly from the static initializer, block
    # index computed per flattened tap.
    #
    # Graph optimization is explicitly disabled for this session -- see the
    # module docstring and test_ort_matmul_nbits_workaround.py's own
    # precedent: this suite treats disabling optimization as the default for
    # any bound-checking InferenceSession, not something to re-derive.
    rng = np.random.default_rng(3)
    cout, cin, kh, kw = 3, 2, 8, 8
    hw = 10
    oh, ow = hw - kh + 1, hw - kw + 1
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.8
    x = rng.standard_normal((1, cin, hw, hw)).astype(np.float32) * 2.0
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{oh},{ow}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.quantize_weight_only_int8_block(model)
    dql_node = next(n for n in quantized.graph.node if n.op_type == "DequantizeLinear")
    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in quantized.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in quantized.graph.initializer if i.name == ws_name)
    ws = numpy_helper.to_array(ws_init)  # [Cout, inner/32]
    inner = cin * kh * kw
    assert numpy_helper.to_array(wq_init).shape == (cout, inner)
    assert ws.shape == (cout, inner // 32)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        quantized.SerializeToString(),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    (y_quant,) = sess.run(["Y"], {"X": x})

    # True float Conv output, computed independently via a brute-force
    # sliding-window sum (no padding, stride 1 -- matches the model above).
    # The patch is flattened in the SAME [Cin, kH, kW] row-major order W's
    # own per-channel flatten uses, so block_of(j) lines up between the two.
    y_float = np.zeros((1, cout, oh, ow), dtype=np.float64)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                patch = x[0, :, i : i + kh, j : j + kw].astype(np.float64)
                y_float[0, c, i, j] = np.sum(patch * weight[c].astype(np.float64))

    error = np.abs(y_float - y_quant)

    eps_w = ws / 2.0  # shape [Cout, inner/32]
    block_of_j = np.arange(inner) // 32  # [inner]
    per_tap_eps = eps_w[:, block_of_j]  # [Cout, inner]: eps_w[c, block_of(j)]
    bound = np.zeros((1, cout, oh, ow), dtype=np.float64)
    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                patch_flat = x[0, :, i : i + kh, j : j + kw].astype(np.float64).ravel()
                bound[0, c, i, j] = np.sum(per_tap_eps[c] * np.abs(patch_flat))

    assert np.all(error <= bound + 1e-6)

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs -- INT8's much
    # wider [-127, 127] code range should track the true float output far
    # more tightly than either INT4 sibling file's own analogous check.
    assert np.mean(error) < 0.5 * np.mean(bound)


def test_weight_only_quantize_int8_block_conv_declines_when_inner_not_divisible_by_block_size():
    # TryQuantizeConvWeightBlockwiseInt8Flat (and this pass's own
    # patternMatchPredicate: `InnerSize(w_t->sizes()) % kBlockSize == 0`) has
    # NO ragged-block fallback at all -- unlike weight_only_quantize_matmul_
    # nbits.h's ceil()-based scheme, a non-divisible inner size makes the
    # whole node decline, not "quantize with a smaller/padded last block".
    # Cin=1, kH=3, kW=3 -> inner=9, not divisible by 32 -- the Conv must
    # survive completely untouched (still exactly one plain float Conv node,
    # no DequantizeLinear/Reshape inserted at all).
    rng = np.random.default_rng(4)
    cout, cin, kh, kw = 2, 1, 3, 3
    hw = 4
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{hw - kh + 1},{hw - kw + 1}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int8_block_conv", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["Conv"]
    conv_node = sim_model.graph.node[0]
    # The weight input is still the original initializer, byte-for-byte.
    w_init = next(
        i for i in sim_model.graph.initializer if i.name == conv_node.input[1]
    )
    np.testing.assert_array_equal(numpy_helper.to_array(w_init), weight)


def test_weight_only_quantize_int8_block_conv_declines_pre_opset21():
    # DequantizeLinear's `block_size` attribute needs opset >= 21
    # (patternMatchPredicate's own check) -- unlike the plain per-channel
    # weight_only_quantize_conv pass's opset >= 13. A plain Conv at opset 20
    # must survive untouched even though inner (64) is divisible by 32.
    rng = np.random.default_rng(5)
    cout, cin, kh, kw = 2, 1, 8, 8
    hw = 9
    weight = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[1,{cin},{hw},{hw}] X) => (float[1,{cout},{hw - kh + 1},{hw - kw + 1}] Y)
        {{
          Y = Conv<kernel_shape = [{kh}, {kw}]>(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=20,
        ir_version=9,
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int8_block_conv", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["Conv"]
