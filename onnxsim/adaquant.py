"""AdaQuant -- the layer-wise calibration half of "Improving Post Training
Neural Quantization: Layer-wise Calibration and Integer Programming" (Hubara,
Nahshan, Hanani, Banner, Soudry, 2020/2021,
https://arxiv.org/abs/2006.10518). The paper has two independent
contributions bundled under one name: (1) AdaQuant itself, a per-layer
gradient-descent reconstruction-error minimization jointly over weight
rounding *and* the activation's own clipping range, and (2) a separate
integer-programming solver that allocates a mixed per-layer bit-width budget
across a whole network. **Only (1) is ported here.** Bit-width allocation is
a distinct, much larger-scope optimization problem (a global budget-
constrained search over per-layer precision choices, not a per-layer
reconstruction step) with no relationship to the rounding/clipping math this
module implements, and is not attempted -- see
``docs/exl3-quantization-survey.md`` for this repository's own precedent of
documenting an explicit "not ported" scope decision rather than silently
under-delivering on a paper's title.

:func:`onnxsim.apply_adaround` already ports this paper's own closest
relative -- Nagel et al.'s AdaRound -- which optimizes only *weight
rounding* (a learned additive per-element perturbation, via the same
"rectified sigmoid" relaxation this module reuses) against a layer's
reconstruction error, leaving the activation's own quantization range fixed
at whatever calibration (:func:`onnxsim.calibrate`'s min-max or entropy
search) picked. AdaQuant's own insight is that this leaves error on the
table: the reconstruction loss ``||W_float @ X - W_quant @ Xdq||^2`` is a
*joint* function of both the weight's rounding and the activation's own
clip range (``Xdq = dequantize(quantize(X, scale, zero_point))``), and the
cross term between them (a weight rounding choice that would be optimal
against one activation clip range is not, in general, optimal against a
different one) is invisible to an optimizer that only ever touches one side.
AdaQuant instead optimizes both by gradient descent on the *same* loss,
per layer, starting from AdaRound-style weight-rounding relaxation and the
calibrated activation range as its warm start -- still layer-wise (each
layer optimized independently and sequentially, no cross-layer
interaction, exactly :func:`onnxsim.apply_adaround`'s own scope), still
plain numpy with hand-derived gradients, no autodiff framework.

Consequently this module's *target scheme* also differs from
:mod:`onnxsim.adaround`/:mod:`onnxsim.awq`/:mod:`onnxsim.gptq` (which all
share :func:`onnxsim.quantize_weight_only_int4`'s weight-only INT4 scheme):
AdaQuant only makes sense where the *activation* is itself quantized with a
learnable clip range, i.e. :func:`onnxsim.quantize_static`'s W8A8 QDQ
scheme, not a weight-only one. Of the two static-quantization graph shapes
this repository produces, this module targets :func:`quantize_static`'s QDQ
format (``QuantizeLinear``/``DequantizeLinear`` pairs wrapping an untouched
MatMul/Gemm) rather than :func:`onnxsim.quantize_qoperator_gemm`'s QGemm
format (a fused ``com.microsoft`` contrib op): the QDQ shape keeps the
activation's scale/zero-point and the weight's codes/scale as ordinary,
independently addressable initializers feeding standard ONNX
Quantize/DequantizeLinear nodes, exactly the "locate an initializer by the
node that consumes it, rewrite it in place" contract every onnxsim
``apply_*`` correction pass already relies on -- QGemm's fused contrib-op
inputs are no harder to *locate*, but there is no equivalent standard-ONNX
node to rewrite them through, and this module has no need to touch QGemm's
own compute node at all.

For every MatMul, or "vanilla" Gemm (``transA=0``, ``alpha=1``, ``beta=1``),
present (by node output name) in both ``float_model`` and ``quantized_model``
whose ``quantized_model`` shape matches
:func:`onnxsim.quantize_static`'s output exactly --

.. code-block:: text

    Xq  = QuantizeLinear(X, Xs, Xzp)        -- Xs/Xzp: asymmetric uint8
    Xdq = DequantizeLinear(Xq, Xs, Xzp)
    Wdq = DequantizeLinear(Wq, Ws, axis=<W's output-channel axis>)
    Y   = MatMul(Xdq, Wdq)                  -- or Gemm

-- this recovers each weight element's quantization bin the same way
:mod:`onnxsim.adaround` does and optimizes a continuous rectified-sigmoid
relaxation of its floor/ceil choice, *jointly* with ``Xs``/``Xzp`` (kept
continuous during optimization, projected back to an integer zero-point in
``[0, 255]`` only at the very end), against real activations captured from
``float_model`` -- minimizing
``||X @ W_float - dequantize(quantize(X, Xs, Xzp)) @ W_hat||^2`` by a small
hand-rolled Adam loop over all three parameter groups at once. The
activation branch's gradients are the standard "straight-through" quantized-
affine gradients (as in Bengio et al.'s STE and, applied to a *learnable*
clip range specifically, Esser et al. 2019's LSQ): ``round()``'s local
gradient is taken as 1 everywhere it isn't saturated by the surrounding
clip, and the clip's own gradient is 1 (pass-through) inside the clamped
range and 0 outside it -- composed step by step through the actual
quantize-then-dequantize computation graph, not algebraically simplified
first (naively simplifying ``dequantize(quantize(x, s)) ≈ x`` under a
literal round-is-identity substitution would erase the very rounding-error
signal the scale gradient needs). The weight's own quantization *scale*
stays exactly what :func:`quantize_static` calibrated -- only which integer
each element rounds to is optimized, the same restriction
:mod:`onnxsim.adaround` already imposes and this module inherits unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _GAMMA, _ZETA, _h_and_dhdv, _node_outputs
from onnxsim.bias_correction import _activation_rows, _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data

_WEIGHT_N_MIN = -127.0
_WEIGHT_N_MAX = 127.0
_ACT_N_MAX = 255.0


@dataclass
class _Candidate:
    output_name: str
    float_node: onnx.NodeProto
    w_float_init: onnx.TensorProto
    wq_name: str
    ws_name: str
    channel_axis: int
    weight_transposed: bool
    x_scale_name: str
    x_zp_name: str


def _find_static_qdq_candidates(
    float_model: onnx.ModelProto, quantized_model: onnx.ModelProto
) -> List[_Candidate]:
    q_by_output = _node_outputs(quantized_model.graph)
    f_by_output = _node_outputs(float_model.graph)
    q_init = {t.name: t for t in quantized_model.graph.initializer}
    f_init = {t.name: t for t in float_model.graph.initializer}

    candidates: List[_Candidate] = []
    for out_name, qn in q_by_output.items():
        if qn.op_type not in ("MatMul", "Gemm") or len(qn.input) < 2:
            continue
        fn = f_by_output.get(out_name)
        if fn is None or fn.op_type != qn.op_type or len(fn.input) < 2:
            continue

        w_float_init = f_init.get(fn.input[1])
        if (
            w_float_init is None
            or w_float_init.data_type != onnx.TensorProto.FLOAT
            or len(w_float_init.dims) != 2
        ):
            continue

        # Weight branch: Wdq = DequantizeLinear(Wq, Ws, axis=...), symmetric
        # (no zero_point input) -- exactly static_quantize_matmul.h's shape.
        wdq = q_by_output.get(qn.input[1])
        if wdq is None or wdq.op_type != "DequantizeLinear" or len(wdq.input) != 2:
            continue
        wq_init = q_init.get(wdq.input[0])
        ws_init = q_init.get(wdq.input[1])
        if (
            wq_init is None
            or ws_init is None
            or wq_init.data_type != onnx.TensorProto.INT8
            or list(wq_init.dims) != list(w_float_init.dims)
        ):
            continue
        axis = 1
        for attr in wdq.attribute:
            if attr.name == "axis":
                axis = attr.i

        # Activation branch: Xdq = DequantizeLinear(Xq, Xs, Xzp), Xq =
        # QuantizeLinear(X, Xs, Xzp) -- both sharing the same scale/zero
        # point initializers, exactly static_quantize_matmul.h's shape.
        xdq = q_by_output.get(qn.input[0])
        if xdq is None or xdq.op_type != "DequantizeLinear" or len(xdq.input) != 3:
            continue
        xq_node = q_by_output.get(xdq.input[0])
        if (
            xq_node is None
            or xq_node.op_type != "QuantizeLinear"
            or len(xq_node.input) != 3
        ):
            continue
        x_scale_name, x_zp_name = xdq.input[1], xdq.input[2]
        if xq_node.input[1] != x_scale_name or xq_node.input[2] != x_zp_name:
            continue
        x_scale_init = q_init.get(x_scale_name)
        x_zp_init = q_init.get(x_zp_name)
        if (
            x_scale_init is None
            or x_zp_init is None
            or x_zp_init.data_type != onnx.TensorProto.UINT8
        ):
            continue

        weight_transposed = False
        if qn.op_type == "Gemm":
            for attr in qn.attribute:
                if attr.name == "transB":
                    weight_transposed = bool(attr.i)

        candidates.append(
            _Candidate(
                output_name=out_name,
                float_node=fn,
                w_float_init=w_float_init,
                wq_name=wq_init.name,
                ws_name=ws_init.name,
                channel_axis=axis,
                weight_transposed=weight_transposed,
                x_scale_name=x_scale_name,
                x_zp_name=x_zp_name,
            )
        )
    return candidates


def _optimize_adaquant(
    w_nk: np.ndarray,
    scale_n: np.ndarray,
    x: np.ndarray,
    x_scale0: float,
    x_zp0: float,
    num_iterations: int,
    weight_learning_rate: float,
    activation_learning_rate: float,
    reg_param: float,
    warm_start: float,
    beta_range: Tuple[float, float],
) -> Tuple[np.ndarray, float, int]:
    """Jointly optimizes one layer's weight-rounding relaxation and its
    activation's (scale, zero_point) against ``||x @ w_nk.T -
    dequantize(quantize(x, s, zp)) @ w_hat.T||^2``, both by gradient descent
    on the same loss. ``w_nk``/``scale_n`` are ``w_nk``'s float weight and
    its (fixed, per-output-channel, symmetric) quantization scale, both
    laid out ``[N, K]``/``[N]`` (output channel first) regardless of the
    op's own storage layout; ``x`` is real calibration activations captured
    from the float model, shape ``[num_samples, K]``.

    Returns ``(codes_nk, x_scale, x_zero_point)``: the optimized integer
    weight codes (shape like ``w_nk``, values in
    ``[-127, 127]``), the optimized activation scale, and its optimized
    zero-point (rounded to the nearest integer in ``[0, 255]``, uint8's
    range).
    """
    n, k = w_nk.shape
    scale_nk = np.repeat(scale_n[:, None], k, axis=1)

    ratio = w_nk / scale_nk
    floor_base = np.floor(ratio)
    frac = np.clip(ratio - floor_base, 1e-4, 1.0 - 1e-4)
    sig0 = np.clip((frac - _GAMMA) / (_ZETA - _GAMMA), 1e-4, 1.0 - 1e-4)
    v = np.log(sig0 / (1.0 - sig0))

    log_s = float(np.log(max(x_scale0, 1e-8)))
    zp = float(np.clip(x_zp0, 0.0, _ACT_N_MAX))

    m_v, v2_v = np.zeros_like(v), np.zeros_like(v)
    m_s = v2_s = m_zp = v2_zp = 0.0
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    warm_start_iters = int(num_iterations * warm_start)
    beta_start, beta_end = beta_range
    n_elems = x.shape[0] * n

    for t in range(num_iterations):
        s_x = np.exp(log_s)

        h, dh_dv = _h_and_dhdv(v)
        raw_w = floor_base + h
        w_hat = np.clip(raw_w, _WEIGHT_N_MIN, _WEIGHT_N_MAX) * scale_nk
        active_w = (raw_w > _WEIGHT_N_MIN) & (raw_w < _WEIGHT_N_MAX)

        # Straight-through quantize-dequantize of the activation: computed
        # step by step (round, then add zero_point, then clip) so each
        # step's own local gradient (1 for round via STE, 1 for the
        # zero_point addition, 1 inside/0 outside the clip) composes
        # correctly instead of being algebraically cancelled out.
        rounded = np.round(x / s_x)
        xq_raw = rounded + zp
        active_x = (xq_raw > 0.0) & (xq_raw < _ACT_N_MAX)
        xq = np.clip(xq_raw, 0.0, _ACT_N_MAX)
        xdq = (xq - zp) * s_x

        y_hat = xdq @ w_hat.T  # [S, N]
        y_float = x @ w_nk.T  # [S, N]
        dl_dy = 2.0 * (y_hat - y_float) / n_elems

        dl_dw_hat = dl_dy.T @ xdq  # [N, K]
        dl_dh = dl_dw_hat * np.where(active_w, scale_nk, 0.0)
        grad_v = dl_dh * dh_dv

        if t >= warm_start_iters:
            progress = (t - warm_start_iters) / max(
                1, num_iterations - warm_start_iters - 1
            )
            beta = beta_start + (beta_end - beta_start) * progress
            u = 2.0 * h - 1.0
            abs_u = np.abs(u)
            dreg_dh = -2.0 * reg_param * beta * np.sign(u) * np.power(abs_u, beta - 1.0)
            grad_v = grad_v + dreg_dh * dh_dv

        dl_dxdq = dl_dy @ w_hat  # [S, K]
        dxdq_ds = (xq - zp) - np.where(active_x, x / s_x, 0.0)
        dxdq_dzp = s_x * (active_x.astype(np.float64) - 1.0)
        grad_log_s = float(np.sum(dl_dxdq * dxdq_ds)) * s_x
        grad_zp = float(np.sum(dl_dxdq * dxdq_dzp))

        m_v = beta1 * m_v + (1.0 - beta1) * grad_v
        v2_v = beta2 * v2_v + (1.0 - beta2) * (grad_v * grad_v)
        bias_c1 = 1.0 - beta1 ** (t + 1)
        bias_c2 = 1.0 - beta2 ** (t + 1)
        v = v - weight_learning_rate * (m_v / bias_c1) / (
            np.sqrt(v2_v / bias_c2) + adam_eps
        )

        m_s = beta1 * m_s + (1.0 - beta1) * grad_log_s
        v2_s = beta2 * v2_s + (1.0 - beta2) * (grad_log_s * grad_log_s)
        log_s = log_s - activation_learning_rate * (m_s / bias_c1) / (
            (v2_s / bias_c2) ** 0.5 + adam_eps
        )

        m_zp = beta1 * m_zp + (1.0 - beta1) * grad_zp
        v2_zp = beta2 * v2_zp + (1.0 - beta2) * (grad_zp * grad_zp)
        zp = zp - activation_learning_rate * (m_zp / bias_c1) / (
            (v2_zp / bias_c2) ** 0.5 + adam_eps
        )
        zp = float(np.clip(zp, 0.0, _ACT_N_MAX))

    h_final, _ = _h_and_dhdv(v)
    codes_nk = np.clip(floor_base + np.round(h_final), _WEIGHT_N_MIN, _WEIGHT_N_MAX)
    x_scale = float(np.exp(log_s))
    x_zero_point = int(np.clip(round(zp), 0, 255))
    return codes_nk, x_scale, x_zero_point


def apply_adaquant(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    weight_learning_rate: float = 0.1,
    activation_learning_rate: float = 0.01,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes AdaQuant-style joint weight-rounding + activation-clip-range
    calibration for every :func:`onnxsim.quantize_static`-quantized
    MatMul/"vanilla" Gemm layer present (by node output name) in both
    ``float_model`` and ``quantized_model``, using real activations captured
    from ``float_model``. See this module's own docstring for the technique
    and its relationship to :func:`onnxsim.apply_adaround`.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_static`. Layers quantized by any other
            scheme (including :func:`onnxsim.quantize_qoperator_gemm`'s
            QGemm format -- see this module's own docstring for why), or
            left unquantized, are left untouched. Assumes ``quantized_model``
            was produced from ``float_model`` without renaming any
            MatMul/Gemm node's own output tensor -- true of every onnxsim
            ``quantize_*`` function.
    :param calibration_data: representative input batches to optimize
            against. Each batch is a ``{input_name: np.ndarray}`` dict
            matching ``float_model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative optimization target than random
            input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_iterations: Adam steps to run per layer
    :param weight_learning_rate: Adam learning rate for the per-element
            weight-rounding relaxation (same role as
            :func:`onnxsim.apply_adaround`'s ``learning_rate``)
    :param activation_learning_rate: Adam learning rate for the
            activation's scale (optimized in log-space) and zero-point
    :param reg_param: weight of the regularization term that pulls each
            weight element's rounding relaxation toward a hard 0/1
            (floor/ceil) decision
    :param warm_start: fraction of ``num_iterations`` (from the start) run
            with the weight-rounding regularization term disabled
    :param beta_range: ``(beta_start, beta_end)`` for the weight-rounding
            regularization term's exponent, linearly annealed across the
            iterations after ``warm_start``
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's weight INT8
            codes and activation (scale, zero_point) initializers rewritten
            to their jointly-optimized values (same shapes/dtypes -- the
            weight's own per-channel scale and the graph structure are
            untouched)
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    candidates = _find_static_qdq_candidates(float_model, quantized_model)
    if not candidates:
        return quantized_model

    probe_names = sorted({c.float_node.input[0] for c in candidates})
    float_probe = _add_probe_outputs(float_model, probe_names)

    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        out = backend.run_model(float_probe, batch, providers=providers)
        for name in probe_names:
            activations[name].append(np.asarray(out[name], dtype=np.float64))

    q_init = {t.name: t for t in quantized_model.graph.initializer}
    optimized_codes: Dict[str, np.ndarray] = {}
    optimized_scale: Dict[str, float] = {}
    optimized_zp: Dict[str, int] = {}

    for c in candidates:
        acts = _activation_rows(activations[c.float_node.input[0]])
        if not acts:
            continue  # no usable activation (no feature axis); skip
        x = np.concatenate(acts, axis=0)

        w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        dim0, dim1 = w.shape
        if c.weight_transposed:
            w_nk = w  # already [N, K]
        else:
            w_nk = w.T  # [K, N] -> [N, K]
        if x.shape[1] != w_nk.shape[1]:
            continue  # activation's feature dim doesn't match K; skip

        ws_init = q_init.get(c.ws_name)
        if ws_init is None:
            continue
        scale_n = onnx.numpy_helper.to_array(ws_init).astype(np.float64).reshape(-1)
        if scale_n.shape[0] != w_nk.shape[0]:
            continue

        x_scale_init = q_init.get(c.x_scale_name)
        x_zp_init = q_init.get(c.x_zp_name)
        if x_scale_init is None or x_zp_init is None:
            continue
        x_scale0 = float(onnx.numpy_helper.to_array(x_scale_init).reshape(-1)[0])
        x_zp0 = float(onnx.numpy_helper.to_array(x_zp_init).reshape(-1)[0])

        codes_nk, x_scale, x_zp = _optimize_adaquant(
            w_nk,
            scale_n,
            x,
            x_scale0,
            x_zp0,
            num_iterations=num_iterations,
            weight_learning_rate=weight_learning_rate,
            activation_learning_rate=activation_learning_rate,
            reg_param=reg_param,
            warm_start=warm_start,
            beta_range=beta_range,
        )
        codes_orig = codes_nk if c.weight_transposed else codes_nk.T
        assert codes_orig.shape == (dim0, dim1)
        optimized_codes[c.wq_name] = codes_orig.astype(np.int8)
        optimized_scale[c.x_scale_name] = x_scale
        optimized_zp[c.x_zp_name] = x_zp

    if not optimized_codes:
        return quantized_model

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    for t in corrected.graph.initializer:
        codes = optimized_codes.get(t.name)
        if codes is not None:
            t.CopyFrom(onnx.numpy_helper.from_array(codes, name=t.name))
            continue
        scale = optimized_scale.get(t.name)
        if scale is not None:
            t.CopyFrom(
                onnx.numpy_helper.from_array(
                    np.array(scale, dtype=np.float32), name=t.name
                )
            )
            continue
        zp = optimized_zp.get(t.name)
        if zp is not None:
            t.CopyFrom(
                onnx.numpy_helper.from_array(np.array(zp, dtype=np.uint8), name=t.name)
            )

    return corrected
