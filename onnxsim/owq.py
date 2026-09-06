"""OWQ (Lee, Park, Kim, Kim and Sung, 2023, "OWQ: Outlier-Aware Weight
Quantization for Efficient Fine-Tuning and Inference of Large Language
Models", https://arxiv.org/abs/2306.02272; AAAI 2024). A fifth lever
targeting :func:`onnxsim.quantize_weight_only_int4`'s output, alongside
:mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/:mod:`onnxsim.awq`/
:mod:`onnxsim.quantease` -- but unlike all four of those (which only ever
change *which integer* an already-fixed-scale quantizer rounds a column to),
OWQ instead rescues a small number of columns from quantization **entirely**,
restoring them to exact float32 precision via a correction term, based on a
different notion of "sensitive" than :mod:`onnxsim.awq`'s or
:mod:`onnxsim.spqr`'s own.

Three onnxsim modules already single out specific weight/activation
elements as needing special treatment, each by a different signal:
:mod:`onnxsim.llm_int8` excludes *activation* columns whose calibration
magnitude exceeds a fixed threshold from INT8 entirely (runtime-observed,
activation-side); :mod:`onnxsim.spqr` extracts individual outlier *weight
values* (not whole columns) into a sparse correction, by how far each
element's own error deviates from its layer's typical error; :mod:`onnxsim.
awq` rescales -- never excludes -- whole input channels in proportion to
their own average activation magnitude. OWQ's own signal is different from
all three: the classic Optimal Brain Surgeon (OBS) saliency metric --

    sensitivity_j = mean_n[(W[n, j] - RTN(W[n, j]))^2] / [H^-1]_jj

where the numerator is column ``j``'s own existing round-to-nearest
quantization error, squared and averaged over output channels ``n`` (a
per-column scalar proxy for the paper's own ``error_j^2`` term), and
``H = X^T X`` is the same calibration-derived Hessian :mod:`onnxsim.gptq`
already computes (this module reuses its ``_inverse_hessian_cholesky``
directly). ``[H^-1]_jj``
measures how much *other* columns could compensate for column ``j``'s own
error if it were left as-is (a small value means little compensation
capacity elsewhere, i.e. column ``j``'s own error matters more directly to
the layer's output) -- the same metric the original Optimal Brain Surgeon
pruning method (LeCun et al./Hassibi & Stork) uses to rank which weights to
remove, applied here to rank which *columns* are least safe to quantize.

For the top ``outlier_fraction`` columns by this score (the paper's own
default is a small fraction, ~0.1-1%), this module does not change how they
are quantized at all -- ``quantized_model``'s INT4 codes are left completely
untouched, including for those columns. Instead, it inserts a *correction*
term at graph-run time: ``Gather`` those columns of the activation, multiply
by a precomputed ``(W_float - W_rtn)`` residual (an exact, static
initializer), and ``Add`` the result to the layer's output. ``W_rtn`` here
is unpacked directly from ``quantized_model``'s own existing INT4
initializer -- not recomputed from scratch in Python -- specifically so the
residual is exact against whatever ``quantized_model`` actually contains:
recomputing round-to-nearest independently (the way, e.g.,
:mod:`onnxsim.adaround` does for its own starting point) can differ from
the real codes by a single rounding tie at a bin boundary, which would
silently turn this module's "exact restoration" claim into an
approximation off by that tie's own quantization step. -- the same "rename producer output, insert an op reproducing the
original name" mechanics :mod:`onnxsim.bias_correction`'s
``_apply_correction`` and :mod:`onnxsim.outlier_suppression_plus` already
use. Because the correction term is exactly ``Gather(X, weak) @ (W_float -
W_rtn)[weak]^T``, adding it to the INT4 branch's own output makes the
selected columns' contribution to the layer's output *exactly* what the
float model would have produced -- not an approximation, a full restoration
-- while every other column stays INT4-quantized as before, so this module
never needs to touch or resize any existing packed INT4 initializer.

Deliberately not ported: the paper's own additional fine-tuning step (a
short LoRA-style adaptation *after* the weak-column split, to recover
further accuracy) -- out of scope for the same reason QAT is throughout
this repo (see ``docs/nncf-comparison-future-work.md``'s own "Explicitly
out of scope" section): it needs a training loop, not a graph rewrite.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _find_int4_matmul_candidates
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.gptq import _inverse_hessian_cholesky


def _unpack_int4(tensor: onnx.TensorProto) -> np.ndarray:
    """Unpacks a ``TensorProto.INT4`` tensor's ``raw_data`` (the same
    low-nibble-first packing ``weight_only_quantize_int4_matmul.h`` uses:
    ``byte[i] = (code[2i] & 0xF) | ((code[2i+1] & 0xF) << 4)``) into a plain
    float64 array shaped ``tensor.dims``.
    """
    dims = list(tensor.dims)
    numel = int(np.prod(dims))
    raw = np.frombuffer(tensor.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    return codes.reshape(dims).astype(np.float64)


def apply_owq(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    outlier_fraction: float = 0.01,
    percdamp: float = 0.01,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Restores OWQ's most quantization-sensitive input columns (by the
    classic Optimal Brain Surgeon saliency metric) to exact float32
    precision, for every ``quantize_weight_only_int4``-quantized MatMul/Gemm
    layer present (by node output name) in both ``float_model`` and
    ``quantized_model``, using real activations captured from
    ``float_model``. See this module's own docstring for the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized), or whose activation
            input isn't a plain 2-D tensor, are left untouched. Assumes
            ``quantized_model`` was produced from ``float_model`` without
            renaming any MatMul/Gemm node's own output tensor -- true of
            every onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches to compute each
            layer's Hessian and per-column error from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``float_model``'s
            graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative Hessian than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param outlier_fraction: fraction of each layer's input columns to
            restore to full precision (the paper's own default range is
            roughly 0.1% to 1%), rounded to at least 1 column
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added to every diagonal entry before inversion), matching
            :mod:`onnxsim.gptq`'s own default -- keeps the inversion
            numerically stable when calibration data doesn't fully activate
            (or correlates too tightly across) a layer's input channels
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with a new ``Gather``/``MatMul``/``Add``
            correction inserted after every matched layer, restoring its
            most sensitive input columns' contribution to exact float32
            precision; the layer's own INT4 codes are never modified. A
            layer with fewer calibration-observed columns than needed for a
            meaningful split, or that OWQ found no columns worth restoring
            for (``outlier_fraction`` rounds to 0), is left untouched.
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

    wq_init_map = {t.name: t for t in quantized_model.graph.initializer}

    probe_names = sorted({c.float_node.input[0] for c in candidates})
    float_probe = _add_probe_outputs(float_model, probe_names)

    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        out = backend.run_model(float_probe, batch, providers=providers)
        for name in probe_names:
            activations[name].append(np.asarray(out[name], dtype=np.float64))

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    graph = corrected.graph
    taken_names = _all_names(graph)
    node_by_output = {n.output[0]: n for n in graph.node if n.output}

    for c in candidates:
        acts = [a for a in activations[c.float_node.input[0]] if a.ndim == 2]
        if not acts:
            continue  # not a plain 2-D activation; skip
        x = np.concatenate(acts, axis=0)

        w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if c.weight_transposed else w.T  # [N, K], output channel first
        k = w_nk.shape[1]
        if x.shape[1] != k:
            continue  # activation's feature dim doesn't match K; skip

        num_weak = int(round(outlier_fraction * k))
        if num_weak < 1:
            continue

        h = x.T @ x
        u = _inverse_hessian_cholesky(h, percdamp)  # inv(h) == u.T @ u
        h_inv_diag = np.maximum((u**2).sum(axis=0), 1e-12)  # [K]

        # Unpack quantized_model's own real INT4 codes (not a fresh
        # from-scratch RTN recomputation) so the residual below is exact
        # against what the graph actually contains -- see this module's own
        # docstring for why that distinction matters here.
        codes = _unpack_int4(wq_init_map[c.wq_name])
        scale = onnx.numpy_helper.to_array(c.ws_init).astype(np.float64)
        if c.weight_transposed:
            codes_nk = codes  # already [N, K]
            scale_blocks = scale  # already [N, K / block_size]
        else:
            codes_nk = codes.T  # [K, N] -> [N, K]
            scale_blocks = scale.T  # [K / block_size, N] -> [N, K / block_size]
        scale_full = np.repeat(scale_blocks, c.block_size, axis=1)
        w_rtn = codes_nk * scale_full
        col_error = np.mean((w_nk - w_rtn) ** 2, axis=0)  # [K]

        sensitivity = col_error / h_inv_diag
        weak_idx = np.argsort(-sensitivity)[:num_weak]
        weak_idx.sort()

        # Exact residual for the rescued columns only -- everywhere else,
        # quantized_model's own INT4 codes are left completely untouched.
        delta_w = (w_nk - w_rtn)[:, weak_idx]  # [N, num_weak]
        if not np.any(delta_w):
            continue  # RTN already exact on every selected column; no-op

        original_output = c.output_name
        qn = node_by_output.get(original_output)
        if qn is None:
            continue  # shouldn't happen -- output_name came from this graph

        idx_name = _unique_name(f"{original_output}_owq_weak_idx", taken_names)
        graph.initializer.append(
            onnx.helper.make_tensor(
                idx_name,
                onnx.TensorProto.INT64,
                [num_weak],
                weak_idx.astype(np.int64).tobytes(),
                raw=True,
            )
        )
        x_weak_name = _unique_name(f"{original_output}_owq_x_weak", taken_names)
        gather_node = onnx.helper.make_node(
            "Gather",
            [c.float_node.input[0], idx_name],
            [x_weak_name],
            name=_unique_name(f"{original_output}_owq_gather", taken_names),
            axis=-1,
        )

        delta_w_name = _unique_name(f"{original_output}_owq_delta_w", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                delta_w.T.astype(np.float32), name=delta_w_name
            )
        )
        corr_name = _unique_name(f"{original_output}_owq_correction", taken_names)
        matmul_node = onnx.helper.make_node(
            "MatMul",
            [x_weak_name, delta_w_name],
            [corr_name],
            name=_unique_name(f"{original_output}_owq_matmul", taken_names),
        )

        pre_name = _unique_name(f"{original_output}_owq_pre_correction", taken_names)
        node_idx = next(i for i, n in enumerate(graph.node) if n is qn)
        qn.output[0] = pre_name
        add_node = onnx.helper.make_node(
            "Add",
            [pre_name, corr_name],
            [original_output],
            name=_unique_name(f"{original_output}_owq_add", taken_names),
        )
        graph.node.insert(node_idx + 1, add_node)
        graph.node.insert(node_idx + 1, matmul_node)
        graph.node.insert(node_idx + 1, gather_node)
        node_by_output[original_output] = add_node

    return corrected
