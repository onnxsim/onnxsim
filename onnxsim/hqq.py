"""HQQ -- Half-Quadratic Quantization (Badri & Shaji, 2023, "Half-Quadratic
Quantization of Large Machine Learning Models",
https://mobiusml.github.io/hqq_blog/). One of the notable weight-only PTQ
techniques ``torchao`` implements (its ``Int4WeightOnlyConfig`` offers
``int4_choose_qparams_algorithm="hqq"`` as an alternative to plain min/max --
onnxsim ports the *algorithm*, not that code, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq` -- torchao has no ONNX export path).

Unlike every other onnxsim-native PTQ technique so far
(:mod:`onnxsim.adaround`, :mod:`onnxsim.awq`, :mod:`onnxsim.gptq`,
:mod:`onnxsim.autoround`), HQQ needs **no calibration data at all** -- it
never runs the model, only looks at each weight tensor's own values. Its
angle: a block's naive min/max-derived quantization range is set by
whichever one or two elements happen to be the most extreme, so a couple of
outliers can force a scale wide enough to blur every other, "normal"
element in that block down toward a handful of coarse levels. HQQ instead
picks the affine quantization parameters (scale, zero-point) to minimize a
**robust** loss -- an ``Lp`` norm with ``p < 1`` on the reconstruction
residual, which (unlike the ordinary ``L2``/mean-squared-error a naive
min/max range implicitly optimizes) tolerates a few large residuals on
outlier elements in exchange for a tighter, more accurate fit on the
majority.

This module optimizes the affine zero-point via **Iteratively Reweighted
Least Squares (IRLS)** -- a standard, textbook algorithm for minimizing an
``Lp`` (``p < 2``) loss by alternating (a) reweighting each element
inversely by its own current residual magnitude raised to the ``p - 2``
power, so elements with large residuals count for less in the next step,
and (b) re-solving the now-ordinary weighted-least-squares problem for the
zero-point in closed form. This converges to the same *kind* of solution
HQQ's own paper describes (a robust fit that downweights outliers) and
optimizes the same objective (:math:`\\sum_k |w_k - \\hat w_k|^p`); it is
not a line-for-line reproduction of the paper's own half-quadratic-splitting
solver, which this module does not claim to replicate exactly, only to
solve the same problem via a different, independently-verifiable classical
method.

The scale itself is set once from each block's own min/max range (the
standard affine-quantization initialization) and held fixed -- only the
zero-point is refined by IRLS, matching typical HQQ implementations, which
report most of the benefit comes from the zero-point fit.

Since ``quantize_weight_only_int4``'s own scheme is symmetric (zero-point
always 0 -- see ``weight_only_quantize_int4_matmul.h``), and HQQ's whole
premise is an *asymmetric* (nonzero zero-point) affine fit, this module
produces its own standalone quantization -- unlike
:mod:`onnxsim.adaround`/:mod:`onnxsim.awq`/:mod:`onnxsim.gptq` (which refine
an already-``quantize_weight_only_int4``-quantized model in place), this
takes the plain float model and quantizes it directly into a
``DequantizeLinear(Wq, Ws, Wz, axis=..., block_size=...)`` structure of its
own -- unsigned 4-bit codes and zero-point (``[0, 15]``), the natural
representation for an affine (as opposed to symmetric) quantizer.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim._onnx_compat import UINT4 as _UINT4
from onnxsim.bias_correction import _all_names, _unique_name


def _match_matmul_like(node: onnx.NodeProto):
    """Mirrors ``MatchMatMulLike`` (``passes/quantize_matmul_common.h``):
    a MatMul, or a Gemm with ``transA=0``, ``alpha=1`` and (when it has a
    bias) ``beta=1``. Returns ``(x_name, w_name, weight_transposed)`` or
    ``None``.
    """
    attrs = {a.name: a for a in node.attribute}
    if node.op_type == "MatMul":
        if len(node.input) != 2:
            return None
        return node.input[0], node.input[1], False
    if node.op_type == "Gemm":
        num_inputs = len(node.input)
        if num_inputs not in (2, 3):
            return None
        trans_a = attrs.get("transA")
        if trans_a is not None and trans_a.i != 0:
            return None
        alpha = attrs.get("alpha")
        if alpha is not None and alpha.f != 1.0:
            return None
        if num_inputs == 3:
            beta = attrs.get("beta")
            if beta is not None and beta.f != 1.0:
                return None
        trans_b = attrs.get("transB")
        weight_transposed = bool(trans_b is not None and trans_b.i)
        return node.input[0], node.input[1], weight_transposed
    return None


def _irls_affine_quantize_blockwise(
    w_nk: np.ndarray, block_size: int, num_iterations: int, lp_norm: float
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Returns ``(codes_nk, scale_blocks, zero_blocks)`` for ``w_nk``
    ([N, K], output channel first): unsigned 4-bit codes in ``[0, 15]``, a
    scale and (IRLS-refined) zero-point per ``(output channel, block-of-K)``
    group, each shape ``[N, K // block_size]``. Assumes ``K % block_size ==
    0``.
    """
    n, k = w_nk.shape
    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)

    lo = blocks.min(axis=2)
    hi = blocks.max(axis=2)
    scale = np.maximum((hi - lo) / 15.0, 1e-12)  # [N, num_blocks]
    scale3 = scale[:, :, np.newaxis]
    zero = -lo / scale  # [N, num_blocks]; code 0 maps back to the block min

    eps = 1e-8
    for _ in range(num_iterations):
        zero3 = zero[:, :, np.newaxis]
        code = np.round(blocks / scale3 + zero3)
        d = blocks - scale3 * code  # residual before the zero-point term
        dequant = scale3 * (code - zero3)
        resid = blocks - dequant
        # IRLS reweighting: elements with a larger current residual count
        # for less in the next weighted-least-squares solve, the mechanism
        # by which this converges toward an Lp<2 (robust) fit instead of
        # the ordinary L2 one a single unweighted solve would give.
        weight = np.power(np.abs(resid) + eps, lp_norm - 2.0)
        # Closed-form weighted-least-squares solution for a shared additive
        # zero-point per block, minimizing sum(weight * (d + scale*zero)^2).
        zero = -np.sum(weight * d, axis=2) / (scale * np.sum(weight, axis=2) + eps)

    zero = np.clip(np.round(zero), 0.0, 15.0)
    zero3 = zero[:, :, np.newaxis]
    codes = np.clip(np.round(blocks / scale3 + zero3), 0.0, 15.0)
    return codes.reshape(n, k), scale, zero


def _pack_uint4(codes: np.ndarray) -> bytes:
    # Low-nibble-first, matching ONNX's documented UINT4/INT4 raw_data
    # packing (byte[i] = code[2i] | (code[2i+1] << 4)); codes here are
    # already unsigned [0, 15], so no sign-bit handling is needed.
    flat = codes.astype(np.int64).ravel()
    nibbles = (flat & 0xF).astype(np.uint8)
    lo = nibbles[0::2]
    hi = nibbles[1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)
    return packed.tobytes()


def quantize_weight_only_int4_hqq(
    model: Union[str, onnx.ModelProto],
    block_size: int = 32,
    num_iterations: int = 10,
    lp_norm: float = 0.7,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``block_size``) into HQQ-style asymmetric block-wise INT4 -- see this
    module's own docstring for the technique. Needs no calibration data:
    every quantization decision comes from the weight tensor's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per (output-channel, block) quantization
            group along the reduction dimension
    :param num_iterations: IRLS reweighting steps refining each block's
            zero-point
    :param lp_norm: the robust loss exponent IRLS targets (``p < 2``; the
            HQQ paper's own typical default, ``0.7``, favors tolerating a
            few large residuals over spreading error evenly -- lower values
            downweight outliers more aggressively, ``p = 2`` would recover
            an ordinary (non-robust) least-squares fit)
    :returns: ``model`` with every matched layer's weight replaced by
            ``DequantizeLinear(Wq, Ws, Wz, axis=<reduction axis>,
            block_size=block_size)`` feeding the original MatMul/Gemm node;
            layers with a non-constant, non-2-D, or non-block-divisible
            weight are left untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    opset_ge_21 = any(
        o.domain in ("", "ai.onnx") and o.version >= 21 for o in out.opset_import
    )
    if not opset_ge_21:
        return (
            model  # UINT4 tensors and DequantizeLinear's block_size both need opset 21+
        )

    nodes = list(graph.node)
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue

        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K]
        k = w_nk.shape[1]
        if k % block_size != 0:
            continue

        codes_nk, scale_blocks, zero_blocks = _irls_affine_quantize_blockwise(
            w_nk, block_size, num_iterations, lp_norm
        )
        codes_orig = codes_nk if weight_transposed else codes_nk.T
        scale_orig = scale_blocks if weight_transposed else scale_blocks.T
        zero_orig = zero_blocks if weight_transposed else zero_blocks.T
        assert codes_orig.shape == (dim0, dim1)

        wq = onnx.TensorProto()
        wq.name = _unique_name(f"{w_name}_hqq_q", taken_names)
        wq.data_type = _UINT4
        wq.dims.extend(codes_orig.shape)
        wq.raw_data = _pack_uint4(codes_orig)
        graph.initializer.append(wq)

        ws = onnx.numpy_helper.from_array(
            scale_orig.astype(np.float32),
            name=_unique_name(f"{w_name}_hqq_scale", taken_names),
        )
        graph.initializer.append(ws)

        wz = onnx.TensorProto()
        wz.name = _unique_name(f"{w_name}_hqq_zero", taken_names)
        wz.data_type = _UINT4
        wz.dims.extend(zero_orig.shape)
        wz.raw_data = _pack_uint4(zero_orig)
        graph.initializer.append(wz)

        # w_nk normalizes to [N, K] (K = axis 1); transposing codes/scale
        # back to the original storage layout puts K on axis 1 when the
        # weight was already stored [N, K] (weight_transposed), else K
        # lands on axis 0 (MatMul's own untransposed [K, N] layout).
        reduction_axis = 1 if weight_transposed else 0
        dq_out = _unique_name(f"{w_name}_hqq_dq", taken_names)
        dq_node = onnx.helper.make_node(
            "DequantizeLinear",
            [wq.name, ws.name, wz.name],
            [dq_out],
            name=_unique_name(f"{w_name}_hqq_dequant", taken_names),
            axis=reduction_axis,
            block_size=block_size,
        )
        graph.node.insert(
            next(i for i, n in enumerate(graph.node) if n is node), dq_node
        )
        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = dq_out

    return out
