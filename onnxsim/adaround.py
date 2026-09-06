"""AIMET's Adaptive Rounding (AdaRound) -- Nagel et al., 2020, "Up or Down?
Adaptive Rounding for Post-Training Quantization"
(https://arxiv.org/abs/2004.10568). The third AIMET PTQ technique ported to
onnxsim after Cross-Layer Equalization (:func:`onnxsim.cross_layer_equalize`)
and empirical Bias Correction (:func:`onnxsim.correct_bias`).

Every ``quantize_weight_only_int4*`` pass rounds each weight element to its
*own* nearest quantization level (round-to-nearest, "RTN"). That minimizes
each element's individual error, but a Conv/Gemm/MatMul's output is a sum
over many weight elements times correlated activations -- the rounding
choice (floor vs. ceil) that minimizes one element's error in isolation is
not, in general, the choice that minimizes the *layer's own output* error,
because rounding errors across elements can reinforce or cancel depending on
the actual activation distribution. AdaRound instead directly optimizes,
per weight element, whether to round down or up so as to minimize the
layer's reconstruction error (``||W_float @ x - W_quant @ x||^2``) on real
calibration data -- consistently measurably better than RTN at the same bit
width, with no change to the quantization grid itself (same scale, same
integer range) or to the runtime op the model deploys as.

This targets :func:`onnxsim.quantize_weight_only_int4`'s output specifically
(``DequantizeLinear(Wq, Ws, axis=<reduction axis>, block_size=...)`` feeding
a MatMul/Gemm, ``Wq`` symmetric INT4 in ``[-7, 7]``, one scale per
``(block, output channel)`` -- see ``weight_only_quantize_int4_matmul.h``):
for every such MatMul/Gemm present (by node output name) in both
``float_model`` and ``quantized_model``, it re-derives each weight element's
quantization bin (``floor(w / scale)``, the same bin RTN already rounds
within) and optimizes a continuous per-element relaxation of the floor/ceil
choice -- the "rectified sigmoid" relaxation the AdaRound paper introduces --
via a small hand-rolled Adam loop, minimizing reconstruction error against
real activations captured by running ``float_model`` on calibration data.
Once optimized, each relaxation is rounded to its nearest hard choice
(floor or ceil) and the packed INT4 initializer is rewritten in place; the
scale, the graph structure, and every other tensor are left untouched.

No calibration data, activation quantization, or gradient framework other
than what this module implements itself is required -- everything here is
plain numpy, matching :mod:`onnxsim.bias_correction`'s "post-hoc adjustment
from real activations" style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data

# The rectified-sigmoid relaxation's own shape parameters (Nagel et al.
# 2020, Sec. 4.1): h(v) = clip(sigmoid(v) * (zeta - gamma) + gamma, 0, 1)
# stretches sigmoid's (0, 1) range slightly past [0, 1] before clipping, so
# h can actually reach exactly 0 or 1 (a plain sigmoid never does) while
# still being smooth and optimizable everywhere in between.
_ZETA = 1.1
_GAMMA = -0.1


@dataclass
class _Candidate:
    output_name: str
    float_node: onnx.NodeProto
    w_float_init: onnx.TensorProto
    wq_name: str
    ws_init: onnx.TensorProto
    axis: int
    block_size: int
    weight_transposed: bool


def _node_outputs(graph: onnx.GraphProto) -> Dict[str, onnx.NodeProto]:
    m: Dict[str, onnx.NodeProto] = {}
    for n in graph.node:
        if n.output:
            m[n.output[0]] = n
    return m


def _find_int4_matmul_candidates(
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

        dq = q_by_output.get(qn.input[1])
        if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) < 2:
            continue
        wq_init = q_init.get(dq.input[0])
        ws_init = q_init.get(dq.input[1])
        if (
            wq_init is None
            or ws_init is None
            or wq_init.data_type != onnx.TensorProto.INT4
            or list(wq_init.dims) != list(w_float_init.dims)
        ):
            continue

        axis = 1
        block_size = None
        for attr in dq.attribute:
            if attr.name == "axis":
                axis = attr.i
            elif attr.name == "block_size":
                block_size = attr.i
        if not block_size:
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
                ws_init=ws_init,
                axis=axis,
                block_size=block_size,
                weight_transposed=weight_transposed,
            )
        )
    return candidates


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _h_and_dhdv(v: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    s = _sigmoid(v)
    raw = s * (_ZETA - _GAMMA) + _GAMMA
    h = np.clip(raw, 0.0, 1.0)
    active = (raw > 0.0) & (raw < 1.0)
    dh_dv = np.where(active, s * (1.0 - s) * (_ZETA - _GAMMA), 0.0)
    return h, dh_dv


def _optimize_rounding(
    w_nk: np.ndarray,
    scale_nk: np.ndarray,
    x: np.ndarray,
    n_min: float,
    n_max: float,
    num_iterations: int,
    learning_rate: float,
    reg_param: float,
    warm_start: float,
    beta_range: "tuple[float, float]",
) -> np.ndarray:
    """Returns the optimized integer codes (shape like ``w_nk``, values in
    ``[n_min, n_max]``) for one weight matrix, given its layer's activation
    ``x`` (shape ``[num_samples, K]``) captured from the float model.
    ``w_nk``/``scale_nk`` are the float weight and its (already
    block-broadcast) per-element scale, both laid out ``[N, K]`` (output
    channel first) regardless of the op's own storage layout.
    """
    y_float = x @ w_nk.T  # [num_samples, N]

    ratio = w_nk / scale_nk
    floor_base = np.floor(ratio)
    frac = np.clip(ratio - floor_base, 1e-4, 1.0 - 1e-4)
    # Inverse of h(v) at v's initial point: start each element's relaxation
    # at round-to-nearest's own choice (h == frac puts the soft weight
    # exactly at the un-rounded ratio, the least-biased starting point).
    sig0 = (frac - _GAMMA) / (_ZETA - _GAMMA)
    sig0 = np.clip(sig0, 1e-4, 1.0 - 1e-4)
    v = np.log(sig0 / (1.0 - sig0))

    m = np.zeros_like(v)
    v2 = np.zeros_like(v)
    adam_beta1, adam_beta2, adam_eps = 0.9, 0.999, 1e-8

    warm_start_iters = int(num_iterations * warm_start)
    beta_start, beta_end = beta_range
    n = x.shape[0] * w_nk.shape[0]

    for t in range(num_iterations):
        h, dh_dv = _h_and_dhdv(v)
        raw2 = floor_base + h
        w_hat = np.clip(raw2, n_min, n_max) * scale_nk
        active_w = (raw2 > n_min) & (raw2 < n_max)

        y_hat = x @ w_hat.T
        dl_dy = 2.0 * (y_hat - y_float) / n
        dl_dw_hat = dl_dy.T @ x  # [N, K]
        dl_dh = dl_dw_hat * np.where(active_w, scale_nk, 0.0)
        grad = dl_dh * dh_dv

        if t >= warm_start_iters:
            progress = (t - warm_start_iters) / max(
                1, num_iterations - warm_start_iters - 1
            )
            beta = beta_start + (beta_end - beta_start) * progress
            u = 2.0 * h - 1.0
            abs_u = np.abs(u)
            dreg_dh = -2.0 * reg_param * beta * np.sign(u) * np.power(abs_u, beta - 1.0)
            grad = grad + dreg_dh * dh_dv

        m = adam_beta1 * m + (1.0 - adam_beta1) * grad
        v2 = adam_beta2 * v2 + (1.0 - adam_beta2) * (grad * grad)
        m_hat = m / (1.0 - adam_beta1 ** (t + 1))
        v_hat = v2 / (1.0 - adam_beta2 ** (t + 1))
        v = v - learning_rate * m_hat / (np.sqrt(v_hat) + adam_eps)

    h_final, _ = _h_and_dhdv(v)
    return np.clip(floor_base + np.round(h_final), n_min, n_max)


def _pack_int4(codes: np.ndarray) -> bytes:
    # Same low-nibble-first packing as weight_only_quantize_int4_matmul.h's
    # TryQuantizeWeightBlockwiseInt4InPlace: byte[i] = (code[2i] & 0xF) |
    # ((code[2i+1] & 0xF) << 4). `codes` is always even-length here (K is a
    # multiple of an even block_size, per this whole scheme's own
    # precondition), so no odd-count padding is needed.
    flat = codes.astype(np.int64).ravel()
    nibbles = (flat & 0xF).astype(np.uint8)
    lo = nibbles[0::2]
    hi = nibbles[1::2]
    packed = (lo | (hi << 4)).astype(np.uint8)
    return packed.tobytes()


def apply_adaround(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    learning_rate: float = 0.1,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: "tuple[float, float]" = (20.0, 2.0),
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes AdaRound-style adaptive rounding for every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model``, using
    real activations captured from ``float_model``. See this module's own
    docstring for the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized) are left untouched.
            Assumes ``quantized_model`` was produced from ``float_model``
            without renaming any MatMul/Gemm node's own output tensor --
            true of every onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches to optimize the
            rounding on. Each batch is a ``{input_name: np.ndarray}`` dict
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
    :param learning_rate: Adam learning rate for the per-element rounding
            relaxation
    :param reg_param: weight of the regularization term that pulls each
            element's relaxation toward a hard 0/1 (floor/ceil) decision
    :param warm_start: fraction of ``num_iterations`` (from the start) run
            with the regularization term disabled, letting reconstruction
            error alone shape the relaxation before it is pulled toward a
            hard decision
    :param beta_range: ``(beta_start, beta_end)`` for the regularization
            term's exponent, linearly annealed across the iterations after
            ``warm_start``
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its AdaRound-optimized codes (same
            shape, dtype, and scale -- only which integer each element
            rounds to changes)
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    candidates = _find_int4_matmul_candidates(float_model, quantized_model)
    if not candidates:
        return quantized_model

    probe_names = sorted({c.float_node.input[0] for c in candidates})
    float_probe = _add_probe_outputs(float_model, probe_names)

    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        out = backend.run_model(float_probe, batch, providers=providers)
        for name in probe_names:
            activations[name].append(np.asarray(out[name], dtype=np.float64))

    optimized: Dict[str, np.ndarray] = {}
    for c in candidates:
        acts = activations[c.float_node.input[0]]
        acts = [a for a in acts if a.ndim == 2]
        if not acts:
            continue  # not a plain 2-D activation (batched/broadcast MatMul); skip
        x = np.concatenate(acts, axis=0)

        w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        scale = onnx.numpy_helper.to_array(c.ws_init).astype(np.float64)
        dim0, dim1 = w.shape

        # Normalize to [N, K] (output channel first) regardless of storage
        # layout, and broadcast the block-wise scale up to full [N, K].
        if c.weight_transposed:
            w_nk = w  # already [N, K]
            scale_blocks = scale  # already [N, K / block_size]
        else:
            w_nk = w.T  # [K, N] -> [N, K]
            scale_blocks = scale.T  # [K / block_size, N] -> [N, K / block_size]
        if x.shape[1] != w_nk.shape[1]:
            continue  # activation's feature dim doesn't match K; skip
        scale_nk = np.repeat(scale_blocks, c.block_size, axis=1)[:, : w_nk.shape[1]]

        codes_nk = _optimize_rounding(
            w_nk,
            scale_nk,
            x,
            n_min=-7.0,
            n_max=7.0,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            reg_param=reg_param,
            warm_start=warm_start,
            beta_range=beta_range,
        )
        codes_orig = codes_nk if c.weight_transposed else codes_nk.T
        assert codes_orig.shape == (dim0, dim1)
        optimized[c.wq_name] = codes_orig.astype(np.int8)

    if not optimized:
        return quantized_model

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    for t in corrected.graph.initializer:
        codes = optimized.get(t.name)
        if codes is None:
            continue
        t.raw_data = _pack_int4(codes)

    return corrected
