"""Formal check for WeightOnlyQuantizeInt4Conv (opt-in; onnxsim's own
``onnxsim/passes/weight_only_quantize_int4_conv.h``): the Conv-layer sibling
of ``weight_only_quantize_int4_matmul.h``
(``test_formal_verify_weight_only_quantize_int4_matmul.py``, this file's
direct template -- READ THAT FILE FIRST, this one only documents what
differs), combining that pass's block-wise INT4 scheme with
``weight_only_quantize_conv.h``'s Conv-specific plumbing
(``test_formal_verify_weight_only_quantize_conv.py``). It rewrites ``Y =
Conv(X, W)`` (optional bias, a third input, left completely untouched) --
``W`` a constant FLOAT32 tensor, rank >= 3 (``[Cout, Cin/groups, k...]``),
``X`` FLOAT32, whose flattened ``inner = Cin/groups * prod(k...)`` is evenly
divisible by ``kBlockSize = 32`` -- into::

    Wq_flat, Ws_flat := block-wise symmetric INT4 quantization (computed
        ONCE, at pass-transform time) of W FIRST RESHAPED to [Cout, inner]
        (TryQuantizeConvWeightBlockwiseInt4Flat, quantize_conv_common.h) --
        a separate scale per (output channel, block-of-inner) pair
    Wdq_flat = DequantizeLinear(Wq_flat, Ws_flat, axis=1, block_size=32)
    Wdq      = Reshape(Wdq_flat, W's ORIGINAL [Cout, Cin/groups, k...] shape)
    Y        = Conv(X, Wdq)

The ONE genuinely new structural wrinkle vs. both templates: Conv's weight
``[Cout, Cin/groups, k...]`` has no single clean reduction axis to block
along the way MatMul's ``K`` does (``weight_only_quantize_conv.h``'s own
per-*channel* INT8 scheme sidesteps this by putting one scale on the WHOLE
channel, never blocking at all) -- every one of ``Cin/groups`` and the
kernel's spatial dims jointly contributes to one output pixel's sum, so
blocking any single one of Conv's own axes would put an independent scale on
every kernel spatial position for little compression benefit (e.g. a 3x3
kernel). This pass instead FLATTENS every axis but the output channel into
one sequence of length ``inner``, blocks *that* flattened sequence exactly
like a MatMul weight's ``K`` axis would be blocked, and RESHAPES the
dequantized result back to ``W``'s original rank/shape afterwards. Confirmed
by reading ``TryQuantizeConvWeightBlockwiseInt4Flat`` directly
(``quantize_conv_common.h``): it first computes Conv's usual flat ``inner =
prod(sizes[1:])``, reads ``W``'s data as one row-major-flat buffer (so
indexing ``data[c * inner + j]`` is exactly ``W.reshape(Cout, inner)[c, j]``
for a row-major reshape -- the same flattening order ``numpy``'s default
``reshape`` uses, confirmed empirically below rather than assumed), then
applies *precisely* ``TryQuantizeWeightBlockwiseInt4InPlace``'s own formula
(``weight_only_quantize_int4_matmul.h``'s own quantizer) to that ``[Cout,
inner]`` matrix, with the output channel pinned to axis 0 and the block
structure along axis 1 (the flattened axis) -- there is no ``channel_axis``
choice to make here the way Gemm's ``transB`` gives that pass, since Conv's
layout always puts the output channel first::

    scale[c, block] = max(|W_flat[c, j]| for j in that block) / 7
                       (or 1.0 for an all-zero block, so no scale is 0)
    Wq_flat[c, j] = clip(round(W_flat[c, j] / scale[c, block_of(j)]), -7, 7)

``Wq_flat``/``Ws_flat`` are therefore genuinely 2-D (``[Cout, inner]`` /
``[Cout, inner / 32]``) even when ``W`` itself is rank > 2 (e.g. a 4-D
``[Cout, Cin, kH, kW]``) -- the caller (``weight_only_quantize_int4_conv.h``)
reshapes ``DequantizeLinear``'s 2-D output back to ``W``'s original shape via
an explicit ``Reshape`` node, which is the ONE node neither template file
needed: ``weight_only_quantize_int4_matmul.h`` never reshapes because
MatMul/Gemm's weight is already 2-D end to end, and
``weight_only_quantize_conv.h`` never reshapes because ITS INT8 scheme keeps
``Wq``/``Ws`` in ``W``'s own original shape throughout (a per-channel scale
broadcasts fine against any rank).

Per the task's own framing (matching how ``weight_only_quantize_conv.py``'s
own docstring related itself to its MatMul sibling): there is NO new Z3
content here beyond what ``test_formal_verify_weight_only_quantize_int4_
matmul.py`` already proves. Once a spatial output position and output
channel ``c`` are fixed, that one output element is exactly a dot product of
the receptive-field patch of ``X`` (``inner`` values, flattened the same way
``W`` is) against ``Wdq_flat[c, :]`` -- i.e. precisely the "one output
element is a linear combination of weight and input values, block-quantized
along one axis" shape that file's own Z3 formulation already models via a
small concrete contraction dimension ``_K`` split across two blocks. This is
still a *single-operand* special case of ``quantized_mac_bound``'s general
MAC bound (``eps_x := 0``, ``X`` never touched), with the SAME per-tap-
differs-by-block structure: each tap ``j``'s own rounding-error budget is
``Ws_flat[c, block_of(j)] / 2``, which can genuinely differ from one tap to
the next depending on which block it falls in, rather than one shared
``Ws[c] / 2`` for every tap the way ``weight_only_quantize_conv``'s own
(unblocked, per-channel) INT8 bound has it::

    |Conv(X, W)[..., c, ...] - Conv(X, Wdq)[..., c, ...]|
        <= sum_j (Ws_flat[c, block_of(j)] / 2) * |X_patch_flat[j]|

So the Z3 vocabulary and BOTH of the INT4 MatMul file's interesting proofs
are reused near-verbatim below, with only the docstring wording made
Conv-flavored: the main per-block bound (``_K = 4`` split across two blocks
of ``_BLOCK_SIZE = 2``, taps 0-1 sharing ``Ws0``, taps 2-3 a genuinely
different ``Ws1``), the bias-cancellation variant (Conv's own optional bias,
untouched), the negative control (the bound needs the rounding hypotheses at
all), the "naive uniform-scale collapse is unsound" negative control (block
1's real scale can exceed block 0's, so bounding every tap by block 0's scale
alone is not a theorem), and the ``DequantizeLinear`` block-axis semantics
check (elements sharing a block index share one scale; a different block
gets its own independent scale) -- exactly ``weight_only_quantize_int4_
matmul``'s own five proofs, since the underlying per-block rounding-bound
math is identical once working at the level of one output element's
flattened receptive-field-times-weight contraction. No consumer-composition
step is added, for the same reason both template files give: an arbitrary
consumer need not be Lipschitz, so a numeric *bound* (not an equality)
implies nothing about ``|consumer(a) - consumer(b)|`` in general.

The genuinely NEW content in this file is entirely in the differential
tests, confirming the flatten-then-block-then-reshape-back mechanics against
the real compiled pass: two Conv weight shapes built via ``onnx.parser`` (per
``CLAUDE.md``) at opset 21 (INT4 tensors and ``DequantizeLinear``'s
``block_size`` attribute both need it) -- ``[Cout=2, Cin=2, kH=4, kW=4]``
(``inner = 32``, exactly ONE block, the simplest case) and ``[Cout=2, Cin=1,
kH=8, kW=8]`` (``inner = 64``, exactly TWO blocks, to actually exercise a
cross-block difference) -- run through the real pass alone via
``simplify_isolated_extra``, confirming: the exact chain ``Conv(X,
Reshape(DequantizeLinear(Wq_flat, Ws_flat, axis=1, block_size=32), shape=W's
original shape))`` fires; the ``Reshape``'s shape-initializer values match
``W``'s original ``[Cout, Cin/groups, kH, kW]`` exactly (not the flattened
``[Cout, inner]`` shape); ``Wq_flat``'s values lie in ``[-7, 7]`` and
``Ws_flat``'s SHAPE is ``[Cout, inner / 32]`` -- confirmed against an
independent numpy re-implementation of
``TryQuantizeConvWeightBlockwiseInt4Flat``'s own formula above (``W``
reshaped to ``[Cout, inner]`` via plain ``np.reshape``, i.e. row-major, then
the same per-(channel, block) quantization formula the INT4 MatMul file's
own numpy helper uses, adapted so the channel is always axis 0 and the block
structure is always along axis 1); ``X`` and any bias pass through
completely unchanged (Conv's OTHER inputs untouched, mirroring both template
files' own distinguishing check); the actual runtime output (via
onnxruntime, ``onnxsim.quantize_weight_only_int4`` -- which applies both the
MatMul and Conv INT4 rewrites, confirmed by reading ``quantize_entry.cpp``)
stays within the per-block bound proved above; an ``inner`` not divisible by
``kBlockSize = 32`` (``Cin=1, kH=3, kW=3`` -> ``inner=9``) declines outright;
and a pre-opset-21 model declines outright (same opset floor as the INT4
MatMul pass, unlike the INT8 Conv pass's opset >= 13).

The firing/bound tests below pass ``check_n=0`` to ``simplify_isolated_extra``
for the same reason both template files' tests do: this is a genuinely lossy
INT4 rewrite, so onnxsim's own random-input equivalence check (default
``check_n=3``, tolerance ``rtol=1e-4``/``atol=1e-5``) does not apply --
confirmed empirically here too.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 4  # concrete number of contraction taps, split across two blocks below --
# stands in for one output position's flattened receptive-field-times-weight
# contraction (inner = Cin/groups * prod(k...) in a real Conv), matching
# weight_only_quantize_int4_matmul's own _K (doubled vs. quantized_mac_bound's
# plain _K = 2 to exercise a genuine cross-block difference).
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
    ``quantized_mac_bound``'s own ``ew``/``ex`` idiom, and identical to
    ``weight_only_quantize_int4_matmul``'s own formulation). There is one
    scale variable PER BLOCK (``Ws[b]``), and each tap's own rounding bound
    is its OWN block's scale, not one shared scale for every tap. Returns
    ``(float_conv, dequant_conv, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # one output position's
    # flattened receptive-field patch of X, true float values
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[c, ...] flattened, true
    # float weight for output channel c
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W_flat[c, k] - Wdq_flat[c, k]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]  # per-block scale
    # Ws_flat[c, block]

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        # Each tap's error is bounded by ITS OWN block's scale -- Ws0 for
        # taps 0-1, a genuinely different Ws1 for taps 2-3 -- not one shared
        # bound for every tap.
        *[_abs(ew[k]) <= Ws[_block_of(k)] / 2 for k in range(_K)],
    )

    float_conv = sum(X[k] * W[k] for k in range(_K))
    dequant_conv = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = sum((Ws[_block_of(k)] / 2) * _abs(X[k]) for k in range(_K))

    return float_conv, dequant_conv, rounding_bounds, bound


def test_weight_only_quantize_int4_conv_error_is_bounded():
    # The genuine bounded-error claim: given W_flat's own per-BLOCK rounding
    # bound (|W_flat[c, j] - Wdq_flat[c, j]| <= Ws_flat[c, block_of(j)] / 2 --
    # the scale that applies to element j depends on WHICH BLOCK j falls in)
    # and X completely unchanged, the true float dot product (one output
    # element's flattened receptive-field contraction) and the one computed
    # against the dequantized-and-reshaped-back weight cannot differ by more
    # than sum_j (Ws_flat[c, block_of(j)] / 2) * |X_patch_flat[j]| --
    # quantized_mac_bound's own bound with eps_x fixed to 0, now with a
    # genuinely per-tap eps_w term. Ws0 and Ws1 are free and independent
    # here, so this holds even when the two blocks' scales differ
    # arbitrarily -- the block structure is genuinely exercised, not
    # accidentally collapsed to a single scale.
    float_conv, dequant_conv, rounding_bounds, bound = _bound_formulas()
    error = float_conv - dequant_conv
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_weight_only_quantize_int4_conv_bias_variant_error_is_bounded():
    # Conv's own optional bias (a third input, a per-output-channel additive
    # term this pass never touches): adding the same Bias(c) to both the
    # true and the dequantized computation leaves their difference -- and
    # therefore the bound on it -- unchanged; Bias cancels out of the error
    # term algebraically, exactly as in both template files' own proofs.
    float_conv, dequant_conv, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_conv + bias) - (dequant_conv + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_weight_only_quantize_int4_conv_negative_control_requires_rounding_bound():
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


def test_weight_only_quantize_int4_conv_uniform_bound_is_unsound_across_blocks():
    # New content this pass's proof needs that weight_only_quantize_conv's
    # single-scale (per-channel, unblocked) bound does not: confirms the
    # per-block sum is not merely untested-but-equivalent to a single shared
    # scale -- it is genuinely NECESSARY. If one instead (incorrectly)
    # bounded every tap's error by block 0's scale alone (Ws0 / 2), as the
    # INT8 Conv pass's single-Ws bound would effectively do were it applied
    # here, that claim is NOT a theorem once block 1's real scale Ws1 can
    # exceed Ws0 -- Z3 finds a counterexample where block 1's actual
    # rounding error (up to Ws1 / 2) overflows the too-small Ws0-only
    # budget. This is exactly why the sound bound above must sum a per-tap
    # term (Ws[block_of(k), n] / 2) rather than factoring out one shared
    # scale -- identical in substance to weight_only_quantize_int4_matmul's
    # own version of this test.
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
    # block 0's scale Ws0, as if it were the single shared scale the INT8
    # Conv pass's bound has.
    uniform_bound = (Ws[0] / 2) * sum(_abs(X[k]) for k in range(_K))

    solver = z3.Solver()
    solver.add(rounding_bounds)
    solver.add(z3.Not(z3.And(error <= uniform_bound, -error <= uniform_bound)))
    assert solver.check() == z3.sat, (
        "the naive single-scale (block 0 only) bound holds even though "
        "block 1 has its own, potentially larger scale -- this pass's "
        "per-block sum is not actually load-bearing, which would be wrong"
    )


def test_weight_only_quantize_int4_conv_dequantizelinear_block_axis_semantics_matches_rounding_bound():
    # New content this pass's proof needs that weight_only_quantize_conv's
    # (unblocked, axis=0-only) DequantizeLinear check does not: confirms
    # DequantizeLinear's own blocked-axis formula -- elements sharing a block
    # index share one scale, a different block gets its own independent
    # scale -- is exactly what feeds the rounding bound above, using three
    # elements: j=0 and j=1 (same block, scale ws_b0) and j=2 (a different
    # block, its own scale ws_b1). Identical in substance to
    # weight_only_quantize_int4_matmul's own version of this test.
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


def _int_attr(node, name):
    return next(a.i for a in node.attribute if a.name == name)


def _quantize_conv_weight_blockwise_int4_flat(weight, block_size):
    """Independent numpy re-implementation of
    ``TryQuantizeConvWeightBlockwiseInt4Flat`` (quantize_conv_common.h):
    ``weight`` ([Cout, Cin/groups, k...]) is FIRST reshaped (row-major, via
    plain ``np.reshape`` -- confirmed to match the pass's own row-major flat
    buffer read by the differential tests below) to ``[Cout, inner]``, then
    quantized per-(output channel, block-of-inner) pair: scale = max(|block|)
    / 7 (or 1.0 for an all-zero block), codes = round(w / scale) clipped to
    [-7, 7]. Unlike ``weight_only_quantize_int4_matmul``'s own analogous
    helper (which can block along either axis depending on ``channel_axis``),
    Conv's flattened layout always puts the output channel on axis 0 and the
    block structure along axis 1 -- no axis choice to make.
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
    scale = np.where(scale > 0, scale / 7.0, 1.0).astype(np.float32)

    scale_bcast = np.repeat(scale, block_size, axis=1)  # [Cout, inner]
    codes = np.clip(np.round(w_flat / scale_bcast), -7, 7).astype(np.int8)
    return codes, scale


def _reshape_node(model, output_name):
    node = producer(model, output_name)
    assert node.op_type == "Reshape"
    return node


def test_weight_only_quantize_int4_conv_pass_fires_and_matches_scheme_single_block():
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
    # check does not apply to a genuinely lossy INT4 rewrite.
    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_int4_conv", check_n=0
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
    assert wq.min() >= -7
    assert wq.max() <= 7

    expected_wq, expected_ws = _quantize_conv_weight_blockwise_int4_flat(weight, 32)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_int4_conv_pass_fires_and_matches_scheme_two_blocks():
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
        model, "weight_only_quantize_int4_conv", check_n=0
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
    # not a flat [Cout] the INT8 weight_only_quantize_conv pass would
    # produce here.
    assert ws.shape == (cout, inner // 32)
    assert ws.shape == (cout, 2)
    assert wq.min() >= -7
    assert wq.max() <= 7
    # The two blocks' scales genuinely differ (not merely "two entries that
    # happen to be equal") -- confirms the block structure is exercised, not
    # accidentally degenerate.
    assert not np.allclose(ws[:, 0], ws[:, 1])

    expected_wq, expected_ws = _quantize_conv_weight_blockwise_int4_flat(weight, 32)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_weight_only_quantize_int4_conv_bias_and_x_are_untouched():
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
        model, "weight_only_quantize_int4_conv", check_n=0
    )

    conv_node = producer(sim_model, "Y")
    assert conv_node.op_type == "Conv"
    x_input, w_input, b_input = conv_node.input
    assert x_input == "X"
    assert b_input == "B"  # bias untouched

    reshape_node = _reshape_node(sim_model, w_input)
    wdq_flat_name, _shape_name = reshape_node.input
    dql_node = _dequantizelinear_node(sim_model, wdq_flat_name)
    assert _int_attr(dql_node, "axis") == 1
    assert _int_attr(dql_node, "block_size") == 32


def test_weight_only_quantize_int4_conv_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring both template files' own analogous tests:
    # run the real quantized graph (via onnxsim.quantize_weight_only_int4,
    # which applies both the MatMul and Conv INT4 rewrites -- confirmed by
    # reading quantize_entry.cpp's QuantizeWeightOnlyInt4) through
    # onnxruntime and confirm every output element's error against the true
    # float Conv stays within the per-block bound the proof above derives --
    # Ws_flat read directly from the static initializer, block index
    # computed per flattened tap.
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

    quantized = onnxsim.quantize_weight_only_int4(model)
    dql_node = next(n for n in quantized.graph.node if n.op_type == "DequantizeLinear")
    wq_name, ws_name = dql_node.input
    wq_init = next(i for i in quantized.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in quantized.graph.initializer if i.name == ws_name)
    ws = numpy_helper.to_array(ws_init)  # [Cout, inner/32]
    inner = cin * kh * kw
    assert numpy_helper.to_array(wq_init).shape == (cout, inner)
    assert ws.shape == (cout, inner // 32)

    sess = ort.InferenceSession(quantized.SerializeToString())
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
    # loose worst-case bound") for these well-scaled inputs -- consistent
    # with INT4 quantization's coarser [-7, 7] code range, not bitwise
    # equal, mirroring both template files' own reasoning for comparing
    # against the proved bound itself rather than an ad hoc tolerance.
    assert np.mean(error) < 0.5 * np.mean(bound)


def test_weight_only_quantize_int4_conv_declines_when_inner_not_divisible_by_block_size():
    # TryQuantizeConvWeightBlockwiseInt4Flat (and this pass's own
    # patternMatchPredicate) requires inner % kBlockSize == 0 (kBlockSize=32);
    # a ragged remainder is explicitly left unhandled. Cin=1, kH=3, kW=3 ->
    # inner=9, not divisible by 32 -- the Conv must survive untouched.
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
        model, "weight_only_quantize_int4_conv", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["Conv"]


def test_weight_only_quantize_int4_conv_declines_pre_opset21():
    # INT4 tensors and DequantizeLinear's `block_size` attribute both need
    # opset >= 21 (patternMatchPredicate's own check) -- unlike
    # weight_only_quantize_conv's opset >= 13. A plain Conv at opset 20 must
    # survive untouched even though inner (64) is divisible by 32.
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
        model, "weight_only_quantize_int4_conv", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["Conv"]
