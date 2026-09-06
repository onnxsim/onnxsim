"""EasyQuant (Wu, Judd, Isaev, Micikevicius, 2020, "EasyQuant: Post-training
Quantization via Scale Optimization", https://arxiv.org/abs/2006.16669).
onnxsim ports the algorithm, not any framework's code, per the same
rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq` (EasyQuant's own
reference implementation quantizes live framework tensors with no ONNX
export path).

Every scale-calibration routine already in onnxsim (:func:`onnxsim.
calibrate`'s ``"minmax"``/``"entropy"``/``"mse"`` methods) picks a
quantization range from **one tensor's own observed distribution alone** --
a histogram, or a simple min/max -- with no knowledge of what that tensor
actually feeds into. EasyQuant's own idea is different in kind: instead of
asking "what threshold best represents this tensor's own values", ask "what
scale, for this weight and this activation *together*, makes the actual
downstream layer's quantized output (``X_q @ W_q^T`` for a MatMul/Gemm)
closest to the real float output" -- directly optimizing the metric that
matters (the layer's own output), the same reconstruction-error framing
:mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/:mod:`onnxsim.quantease` already
use for weight *rounding*, but applied here to the **scale** itself for a
plain W8A8 (INT8 weight, INT8 activation) quantizer, with no gradient
descent or Hessian at all -- just a **coordinate-descent grid search**:

1. Start from an ordinary per-output-channel weight scale and per-tensor
   activation scale (each tensor's own ``max(abs(.)) / 127``, the same
   starting point :func:`onnxsim.calibrate`'s ``"minmax"`` method would
   give).
2. **Weight step**: holding the activation scale fixed, search a small grid
   of candidate multipliers around each output channel's own current scale,
   picking whichever minimizes that channel's own quantized-output MSE
   against the float output -- exact and independent per output channel,
   since (for ``Y = X @ W^T``) column ``n`` of ``Y`` depends only on row
   ``n`` of ``W``, never on any other channel.
3. **Activation step**: holding the (now updated) weight scale fixed,
   search a small grid of candidate multipliers on the single activation
   scale, picking whichever maximizes the *whole* quantized layer output's
   cosine similarity against the float output -- the paper's own metric,
   and not separable per-channel the way the weight step is, so this step
   evaluates the full output.
4. Repeat 2-3 for a small, fixed number of rounds (each round only ever
   improves or holds the previous round's own chosen objective, since a
   grid search always considers "no change" as a candidate).

This module's own honest simplification of the paper: EasyQuant's own
reference procedure searches activation scale per-tensor exactly as here,
but this module's *weight* step decomposes the paper's own overall
cosine-similarity objective into independent per-channel MSE minimization
(justified above -- the decomposition is exact, not approximate, since
column ``n``'s error genuinely depends only on row ``n`` of ``W``) rather
than jointly re-evaluating whole-output cosine similarity for every
candidate combination of all channels at once (combinatorially
infeasible for a grid search) -- this module does not claim its weight
step reproduces the paper's own exact search procedure, only the same
data-driven, output-aware spirit.

Quantization itself is applied as a **float32 round-trip** (quantize then
immediately dequantize) exactly the same simplification :mod:`onnxsim.
attention_quantization`'s own per-token INT8 quantization already makes --
the weight side is folded directly into a new float32 initializer (no new
graph nodes needed, the same pattern every weight-only ``quantize_*``
function in this repo uses), while the activation side needs
``Div``/``Round``/``Clip``/``Mul`` nodes inserted at graph-run time (since
it's a runtime tensor, not a constant). This module has no lower-than-
float32 arithmetic ONNX op to express genuine ``int8 x int8`` execution in
anyway -- the same limitation :mod:`onnxsim.zeroquant`'s own docstring
names for onnxsim's other float-simulated activation-quantization passes.

**Scope note**: only ``MatMul`` and "vanilla" ``Gemm`` (``transA=0``,
``alpha=1``, and ``beta=1`` when a bias is present) with a constant 2-D
float32 weight are matched (via :func:`onnxsim.llm_int8._match_matmul_like`,
already shared with :mod:`onnxsim.llm_int8`) -- ``Conv`` is left untouched,
a scope decision consistent with several other onnxsim modules that target
only MatMul/Gemm.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import (
    _activation_rows,
    _add_probe_outputs,
    _all_names,
    _unique_name,
)
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.llm_int8 import _match_matmul_like


def _quantize_round_trip(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values / scale), -127.0, 127.0) * scale


def _search_scales(
    w_nk: np.ndarray,
    x: np.ndarray,
    num_iterations: int,
    num_candidates: int,
    search_span: float,
) -> "tuple[np.ndarray, float]":
    n, k = w_nk.shape
    eps = 1e-12

    base_w_scale = np.maximum(np.max(np.abs(w_nk), axis=1), eps) / 127.0
    base_a_scale = max(float(np.max(np.abs(x))), eps) / 127.0

    y_float = x @ w_nk.T  # [samples, N]
    multipliers = np.linspace(1.0 - search_span, 1.0 + search_span, num_candidates)
    multipliers = multipliers[multipliers > 0.0]

    w_scale = base_w_scale.copy()
    a_scale = base_a_scale

    for _ in range(num_iterations):
        x_q = _quantize_round_trip(x, a_scale)  # [samples, K], fixed this step

        for i in range(n):
            best_mse = None
            best_scale = w_scale[i]
            for m in multipliers:
                s = base_w_scale[i] * m
                w_q_row = _quantize_round_trip(w_nk[i, :], s)
                y_q_col = x_q @ w_q_row
                mse = float(np.mean((y_q_col - y_float[:, i]) ** 2))
                if best_mse is None or mse < best_mse:
                    best_mse, best_scale = mse, s
            w_scale[i] = best_scale

        w_q = _quantize_round_trip(w_nk, w_scale[:, None])
        y_float_norm = np.linalg.norm(y_float.ravel()) + eps
        best_cos = None
        best_a_scale = a_scale
        for m in multipliers:
            s = base_a_scale * m
            x_q_try = _quantize_round_trip(x, s)
            y_q = x_q_try @ w_q.T
            cos = float(
                np.dot(y_q.ravel(), y_float.ravel())
                / ((np.linalg.norm(y_q.ravel()) + eps) * y_float_norm)
            )
            if best_cos is None or cos > best_cos:
                best_cos, best_a_scale = cos, s
        a_scale = best_a_scale

    return w_scale, a_scale


def apply_easyquant(
    float_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 3,
    num_candidates: int = 21,
    search_span: float = 0.5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """W8A8-quantizes every matched MatMul/"vanilla" Gemm layer, choosing
    both the per-output-channel weight scale and the per-tensor activation
    scale via EasyQuant's own coordinate-descent search against real
    calibration activations -- see this module's own docstring for the
    technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param calibration_data: representative input batches to run the scale
            search against -- see :func:`onnxsim.correct_bias`'s own
            parameter of the same name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_iterations: coordinate-descent rounds (weight step, then
            activation step) -- each round only ever improves or holds the
            previous round's own chosen scales
    :param num_candidates: grid resolution per coordinate-descent step
    :param search_span: candidate multipliers span
            ``[1 - search_span, 1 + search_span]`` around each step's
            current scale
    :param providers: onnxruntime execution providers to run
            ``float_model`` on when capturing calibration activations
    :returns: ``float_model`` with every matched layer's weight replaced by
            its quantize-dequantize round-tripped float32 version, and a
            ``Div``/``Round``/``Clip``/``Mul`` round-trip inserted before
            its activation input -- layers with a non-constant, non-2-D
            weight, an activation with no feature axis at all (rank < 2;
            a higher-rank ``[batch, seq, K]`` one is flattened to
            ``[batch * seq, K]``, which is exact), or whose activation's
            feature dimension doesn't match the weight's own reduction
            size, are
            left untouched
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    initializer_map = {t.name: t for t in float_model.graph.initializer}
    candidates = []  # (x_name, w_init, weight_transposed, out_name)
    for node in float_model.graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, _bias_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
            or not node.output
        ):
            continue
        candidates.append((x_name, w_init, weight_transposed, node.output[0]))
    if not candidates:
        return float_model

    probe_names = list({x_name for x_name, _, _, _ in candidates})
    float_probe = _add_probe_outputs(float_model, probe_names)

    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        batch_out = backend.run_model(float_probe, batch, providers=providers)
        for name in probe_names:
            activations[name].append(np.asarray(batch_out[name], dtype=np.float64))

    out = onnx.ModelProto()
    out.CopyFrom(float_model)
    taken_names = _all_names(out.graph)
    # Nodes matched above by output-tensor name (a fresh copy's own nodes
    # are distinct Python objects from float_model's, even though
    # structurally identical) -- ONNX output tensor names are unique within
    # a graph, so this reliably finds each candidate's own node in `out`.
    node_by_output = {n.output[0]: n for n in out.graph.node if n.output}

    for x_name, w_init, weight_transposed, out_name in candidates:
        acts = _activation_rows(activations[x_name])
        if not acts:
            continue  # no usable activation (no feature axis); skip
        x = np.concatenate(acts, axis=0)

        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K]
        if x.shape[1] != w_nk.shape[1]:
            continue

        w_scale, a_scale = _search_scales(
            w_nk, x, num_iterations, num_candidates, search_span
        )
        w_quant_nk = _quantize_round_trip(w_nk, w_scale[:, None])
        w_quant = w_quant_nk if weight_transposed else w_quant_nk.T

        node = node_by_output[out_name]

        new_w_name = _unique_name(f"{w_init.name}_easyquant", taken_names)
        out.graph.initializer.append(
            onnx.numpy_helper.from_array(w_quant.astype(np.float32), name=new_w_name)
        )
        node.input[1] = new_w_name

        prefix = _unique_name(f"{node.output[0]}_easyquant", taken_names)
        scale_name = _unique_name(f"{prefix}_act_scale", taken_names)
        out.graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(a_scale, dtype=np.float32), name=scale_name
            )
        )
        neg_name = _unique_name(f"{prefix}_neg127", taken_names)
        out.graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(-127.0, dtype=np.float32), name=neg_name
            )
        )
        pos_name = _unique_name(f"{prefix}_pos127", taken_names)
        out.graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray(127.0, dtype=np.float32), name=pos_name
            )
        )

        new_nodes = []

        def _op(op_type, inputs, tag, **attrs):
            node_out_name = _unique_name(f"{prefix}_{tag}", taken_names)
            n = onnx.helper.make_node(
                op_type,
                inputs,
                [node_out_name],
                name=_unique_name(f"{prefix}_{tag}_node", taken_names),
                **attrs,
            )
            new_nodes.append(n)
            return node_out_name

        scaled = _op("Div", [x_name, scale_name], "scaled")
        rounded = _op("Round", [scaled], "rounded")
        clipped = _op("Clip", [rounded, neg_name, pos_name], "clipped")
        dequant = _op("Mul", [clipped, scale_name], "dequant")
        node.input[0] = dequant

        insertion_point = next(i for i, n in enumerate(out.graph.node) if n is node)
        for new_node in new_nodes:
            out.graph.node.insert(insertion_point, new_node)
            insertion_point += 1

    return out
