"""SpQR (Dettmers et al., 2023, "SpQR: A Sparse-Quantized Representation
for Near-Lossless LLM Weight Compression", https://arxiv.org/abs/2306.03078).
onnxsim ports the algorithm, not any framework's code, per the same
rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.spinquant`
(SpQR's own reference implementation quantizes live PyTorch weights via a
custom outlier-detection-and-packing pipeline, with no ONNX export path).

Ordinary block-wise RTN quantization (what
:func:`onnxsim.quantize_weight_only_int4` does) picks one scale per block
of the reduction dimension. A single unusually large weight *within* a
block forces that block's whole scale up, wasting resolution on every
other, ordinary-magnitude element sharing it -- the same "outlier" problem
:mod:`onnxsim.llm_int8`/:mod:`onnxsim.smoothquant` address for
*activations*, but here it is individual *weight* elements, scattered
throughout the matrix rather than confined to a few channels, so excluding
them channel-wise (like :mod:`onnxsim.llm_int8`) doesn't fit. SpQR's own
idea: identify the small fraction of weight elements whose quantization
error actually matters, exclude them from each block's own scale
computation (so the *rest* of the block quantizes tighter), and store an
exact correction for those specific elements as an explicit sparse
overlay -- ``W_reconstructed = block_quantized(W) + sparse_correction``,
with the correction defined as exactly ``W - block_quantized(W)`` at each
outlier position, so it cancels that position's quantization error
completely regardless of how the block-quantized value there was rounded.

**Picking which elements are outliers.** SpQR's own reference
implementation uses each weight's true contribution to the OBQ/GPTQ
objective, computed from the full inverse-Hessian of the layer's
calibration data -- expensive, and (like GPTQ's own column-by-column
update order) not independently verifiable without re-deriving the same
numerically delicate procedure. This module uses the classical
**diagonal-Hessian approximation** to that same objective instead: for a
squared-error objective with Hessian ``H = 2 X^T X``, OBQ's per-weight
error contribution ``w_k^2 / [H^{-1}]_{kk}`` reduces, when ``H`` is
approximated as diagonal, to ``w_k^2 * H_{kk} = w_k^2 * mean(X[:, k]^2)``
-- an ordinary, closed-form sensitivity score computed directly from the
weight and calibration activations, no matrix inversion involved. The
elements with the largest score (by default the top 1%, tunable via
``outlier_fraction``) are excluded from their block's scale computation
and become the sparse correction; every other element is quantized
normally.

**Storing the sparse correction efficiently.** Naively adding a dense
``[N, K]`` correction matrix back would cost as much storage as the
original float32 weight, defeating the point. Instead, only the
``num_outliers`` outlier ``(row, col)`` positions and their correction
values are stored (an initializer of shape ``[num_outliers, 2]`` plus one
of shape ``[num_outliers]`` -- at 1% density, a small fraction of the
dense weight's own footprint), and the graph reconstructs the full dense
correction at runtime via ``ScatterND`` into a ``ConstantOfShape``-produced
zero tensor -- both ordinary ONNX ops, no custom sparse tensor type or
contrib op needed:

    Before:
      Y = MatMul(X, W) [+ bias]                  -- W constant, [K, N], float32

    After:
      Wq  = <int4, per-(block, column) symmetric, outlier positions
             excluded from each block's own scale>
      Ws  = <float32, [K/block_size, N]>
      Wdq = DequantizeLinear(Wq, Ws, axis=0, block_size=block_size)  -- float32
      zeros = ConstantOfShape([K, N], value=0.0)
      correction = ScatterND(zeros, outlier_indices, outlier_values)  -- float32
      Wreconstructed = Wdq + correction
      Y = MatMul(X, Wreconstructed) [+ bias]

Every outlier position reconstructs *exactly* (the correction is defined
as the exact residual there, independent of how that position happened to
round), while the excluded-from-scale blocks quantize every ordinary
element more tightly than plain :func:`onnxsim.quantize_weight_only_int4`
would with the same outliers still dragging the scale up.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _pack_int4
from onnxsim.bias_correction import (
    _activation_rows,
    _add_probe_outputs,
    _all_names,
    _unique_name,
)
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.quip_sharp import _match_matmul_like


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def quantize_weight_only_spqr(
    model: Union[str, onnx.ModelProto],
    block_size: int = 16,
    outlier_fraction: float = 0.01,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies SpQR-style outlier-aware block-wise INT4 quantization (see
    this module's own docstring) to every MatMul/vanilla-Gemm layer with a
    constant 2-D float32 weight whose reduction dimension ``K`` is
    divisible by ``block_size``.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per quantization block along ``K``,
            matching :func:`onnxsim.quantize_weight_only_int4`'s own
            default granularity (SpQR's own typical choice is finer,
            8-32)
    :param outlier_fraction: fraction of each layer's weight elements
            (by count) excluded from block-scale computation and stored
            as an exact sparse correction instead -- SpQR's own paper
            uses roughly 1%
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to compute each weight element's sensitivity
            score -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data,
            a more representative sensitivity ranking than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight replaced by
            block-wise INT4 codes plus a sparse outlier correction (see
            the module docstring's diagram); output tensor name unchanged.
            Layers with a non-constant, non-2-D weight, a reduction
            dimension not divisible by ``block_size``, or no calibration
            activation available, are left untouched; a model with no
            matching layer, or an opset older than 21 (INT4's tensor type
            and ``DequantizeLinear``'s ``block_size`` attribute both need
            opset 21), is returned unchanged
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 21):
        return model

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    nodes = list(graph.node)
    candidates = []
    for node in nodes:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, bias_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((node, x_name, w_name, bias_name, weight_transposed))

    if not candidates:
        return out

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    probe_names = sorted({x_name for _, x_name, _, _, _ in candidates})
    probe_model = _add_probe_outputs(model, probe_names)
    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            arr = np.asarray(result[name], dtype=np.float64)
            activations[name].extend(_activation_rows([arr]))

    for node, x_name, w_name, bias_name, weight_transposed in candidates:
        acts = activations.get(x_name, [])
        if not acts:
            continue
        x = np.concatenate(acts, axis=0)

        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape
        if k % block_size != 0 or x.shape[1] != k:
            continue

        # Diagonal-Hessian OBQ sensitivity score -- see module docstring.
        h_k = np.mean(x**2, axis=0)  # [K]
        sensitivity = (w_nk**2) * h_k[np.newaxis, :]  # [N, K]

        num_outliers = int(round(outlier_fraction * n * k))
        num_blocks = k // block_size
        if num_outliers > 0:
            flat_idx = np.argpartition(sensitivity.ravel(), -num_outliers)[
                -num_outliers:
            ]
            outlier_rows, outlier_cols = np.unravel_index(flat_idx, (n, k))
            mask = np.ones((n, k), dtype=bool)
            mask[outlier_rows, outlier_cols] = False
        else:
            outlier_rows = np.empty(0, dtype=np.int64)
            outlier_cols = np.empty(0, dtype=np.int64)
            mask = np.ones((n, k), dtype=bool)

        blocks = w_nk.reshape(n, num_blocks, block_size)
        mask_blocks = mask.reshape(n, num_blocks, block_size)
        abs_masked = np.where(mask_blocks, np.abs(blocks), 0.0)
        scale_blocks = (
            np.maximum(abs_masked.max(axis=2), 1e-12) / 7.0
        )  # [N, num_blocks]
        scale_full = np.repeat(scale_blocks, block_size, axis=1)  # [N, K]

        codes_nk = np.clip(np.round(w_nk / scale_full), -7.0, 7.0)
        dequant_nk = codes_nk * scale_full

        prefix = f"{w_name}_spqr"
        codes_kn = codes_nk.T.astype(np.int64)  # [K, N]
        scale_kn = scale_blocks.T.astype(np.float32)  # [K/block_size, N]

        codes_name = _unique_name(f"{prefix}_codes", taken_names)
        codes_tensor = onnx.TensorProto()
        codes_tensor.name = codes_name
        codes_tensor.data_type = onnx.TensorProto.INT4
        codes_tensor.dims.extend([k, n])
        codes_tensor.raw_data = _pack_int4(codes_kn)
        graph.initializer.append(codes_tensor)

        scale_name = _unique_name(f"{prefix}_scale", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(scale_kn, name=scale_name)
        )

        new_nodes: List[onnx.NodeProto] = []

        def _new(op_type, inputs, out_suffix, **attrs):
            out_name = _unique_name(f"{prefix}_{out_suffix}", taken_names)
            n_ = onnx.helper.make_node(
                op_type,
                inputs,
                [out_name],
                name=_unique_name(f"{prefix}_{out_suffix}_node", taken_names),
                **attrs,
            )
            new_nodes.append(n_)
            return out_name

        w_dequant = _new(
            "DequantizeLinear",
            [codes_name, scale_name],
            "w_dequant",
            axis=0,
            block_size=block_size,
        )

        if num_outliers > 0:
            correction_values = (
                w_nk[outlier_rows, outlier_cols]
                - dequant_nk[outlier_rows, outlier_cols]
            )
            # [K, N]-layout indices, matching codes_kn/scale_kn's own
            # transposed storage: index[i] = [k_pos, n_pos].
            outlier_indices_kn = np.stack([outlier_cols, outlier_rows], axis=1)

            indices_name = _unique_name(f"{prefix}_outlier_indices", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    outlier_indices_kn.astype(np.int64), name=indices_name
                )
            )
            values_name = _unique_name(f"{prefix}_outlier_values", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    correction_values.astype(np.float32), name=values_name
                )
            )
            shape_name = _unique_name(f"{prefix}_shape", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    np.array([k, n], dtype=np.int64), name=shape_name
                )
            )

            zeros = _new(
                "ConstantOfShape",
                [shape_name],
                "zeros",
                value=onnx.numpy_helper.from_array(np.array([0.0], dtype=np.float32)),
            )
            correction = _new(
                "ScatterND", [zeros, indices_name, values_name], "correction"
            )
            w_reconstructed = _new("Add", [w_dequant, correction], "w_reconstructed")
        else:
            w_reconstructed = w_dequant

        core = _new("MatMul", [x_name, w_reconstructed], "core")

        old_output = node.output[0]
        if bias_name is not None:
            final = onnx.helper.make_node(
                "Add",
                [core, bias_name],
                [old_output],
                name=_unique_name(f"{prefix}_bias_add_node", taken_names),
            )
        else:
            final = onnx.helper.make_node(
                "Identity",
                [core],
                [old_output],
                name=_unique_name(f"{prefix}_identity_node", taken_names),
            )
        new_nodes.append(final)

        node_idx = next(i for i, n_ in enumerate(graph.node) if n_ is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, new_node)
        del graph.node[node_idx + len(new_nodes)]

    return out
