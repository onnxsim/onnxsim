"""GPTAQ (Li, Yin, Lee, Xiao, Panda, 2025, "GPTAQ: Efficient Finetuning-Free
Quantization for Asymmetric Calibration", https://arxiv.org/abs/2504.02692)
-- a small, closed-form correction to :mod:`onnxsim.gptq` that this module
re-derives from first principles below rather than transcribing the paper's
own notation (the paper states the result but not every intermediate step;
the derivation here is onnxsim's own, checked algebraically, not a
transcription of the authors' proof).

The problem GPTAQ points out: real GPTQ quantizes a network layer by layer,
so by the time layer ``i`` is quantized, the activations flowing into it
already come from every *earlier* layer's own (already-quantized) weights
-- call that corrupted activation ``X``. But GPTQ's own per-column
objective (minimize ``||W X^T - Ŵ X^T||²``, ``W`` float, ``Ŵ`` quantized)
implicitly targets reconstructing *that corrupted signal*, not what the
original float network would have actually produced there (call the true,
never-corrupted activation ``X̃``). GPTQ calibrated this way is quietly
optimizing against its own accumulated error instead of correcting for it.

:mod:`onnxsim.gptq` sidesteps this by construction -- it always captures
activations from ``float_model`` alone, so every layer already calibrates
against ``X̃``, not a corrupted ``X`` at all. GPTAQ's asymmetric-calibration
idea is nonetheless available to onnxsim specifically *because* it already
threads both ``float_model`` and ``quantized_model`` through every
``apply_*`` correction pass here: this module captures ``X̃`` from
``float_model`` exactly like GPTQ, but *additionally* captures ``X`` at the
same probe point from ``quantized_model`` (whatever it was already
quantized/corrected by), and folds the gap between them into GPTQ's own
per-column procedure via one small, exact pre-computation.

Derivation. Write ``δX = X̃ - X`` (the accumulated upstream corruption at
this layer's input, fixed and independent of this layer's own weight
quantization). For one output row ``w`` (float) / ``ŵ`` (quantized) and
error ``e = w - ŵ``:

```
w X̃^T - ŵ X^T = w (X + δX)^T - (w - e) X^T = w δX^T + e X^T
```

so the per-row squared objective ``||w δX^T + e X^T||²`` expands to
``e H e^T + 2 c^T e^T + const`` with ``H = X^T X`` (GPTQ's own Hessian,
computed here from the *quantized*-model's activations, since that is
what multiplies ``e`` above) and the new linear term's coefficient
``c = X^T (δX w^T)`` -- a fixed, precomputable vector, since it only
depends on the already-known float weight row ``w`` and the fixed
activations, not on ``ŵ``. Completing the square (dropping the resulting
constant, which does not depend on ``ŵ``) shows this is *exactly* GPTQ's
own quadratic objective ``e' H e'^T``, but for a shifted error
``e' = (w + shift) - ŵ`` where ``shift = (H^{-1} c)^T`` -- i.e., GPTAQ is
GPTQ's own column algorithm, applied unchanged, to the weight matrix
``W + Shift`` instead of ``W``. This matches the paper's own description
of the fix as one small, closed-form residual term layered on top of
GPTQ, not a different algorithm.

When a candidate layer has no upstream quantization yet (``X == X̃``,
``δX ≈ 0``), ``Shift`` is (numerically) zero and this module's output
matches plain GPTQ's exactly -- the expected degenerate case, and one this
module's own tests check directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _find_int4_matmul_candidates, _node_outputs, _pack_int4
from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.gptq import _gptq_quantize_columns, _inverse_hessian_cholesky


def apply_gptaq(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes GPTAQ-style (asymmetric-calibration) sequential,
    Hessian-compensated rounding for every ``quantize_weight_only_int4``-
    quantized MatMul/Gemm layer present (by node output name) in both
    ``float_model`` and ``quantized_model``. See this module's own
    docstring for the technique and its relationship to
    :func:`onnxsim.apply_gptq`.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`, optionally already
            refined by other passes (e.g. :func:`onnxsim.apply_gptq`,
            :func:`onnxsim.correct_bias`) -- this is exactly what makes the
            asymmetric calibration meaningful: ``quantized_model``'s own
            activations at a candidate layer's input reflect whatever
            upstream layers' quantization already did. Layers quantized by
            any other scheme (or left unquantized), or whose activation
            input isn't a plain 2-D tensor, are left untouched. Assumes
            ``quantized_model`` was produced from ``float_model`` without
            renaming any MatMul/Gemm node's own output tensor -- true of
            every onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches, run through
            *both* models to capture the true (``float_model``) and
            corrupted (``quantized_model``) activation at each candidate's
            input -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data`.
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param percdamp: Hessian damping factor, matching
            :func:`onnxsim.apply_gptq`'s own parameter and default
    :param proc_block_size: GPTQ's own column-processing block size, passed
            through unchanged to :func:`onnxsim.apply_gptq`'s internals
    :param providers: onnxruntime execution providers to run both models on
            when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its GPTAQ-optimized codes (same shape,
            dtype, and scale -- only which integer each element rounds to
            changes)
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

    q_by_output = _node_outputs(quantized_model.graph)
    quant_probe_name: Dict[str, str] = {}
    for c in candidates:
        qn = q_by_output.get(c.output_name)
        if qn is not None and len(qn.input) >= 1:
            quant_probe_name[c.output_name] = qn.input[0]

    float_probe_names = [c.float_node.input[0] for c in candidates]
    float_probe = _add_probe_outputs(float_model, float_probe_names)
    quant_probe = _add_probe_outputs(quantized_model, list(quant_probe_name.values()))

    float_acts: Dict[str, List[np.ndarray]] = {name: [] for name in float_probe_names}
    quant_acts: Dict[str, List[np.ndarray]] = {
        name: [] for name in quant_probe_name.values()
    }
    for batch in calibration_data:
        out_f = backend.run_model(float_probe, batch, providers=providers)
        for name in float_probe_names:
            float_acts[name].append(np.asarray(out_f[name], dtype=np.float64))
        out_q = backend.run_model(quant_probe, batch, providers=providers)
        for name in quant_acts:
            quant_acts[name].append(np.asarray(out_q[name], dtype=np.float64))

    optimized: Dict[str, np.ndarray] = {}
    for c in candidates:
        quant_name = quant_probe_name.get(c.output_name)
        if quant_name is None:
            continue
        x_true_parts = [a for a in float_acts[c.float_node.input[0]] if a.ndim == 2]
        x_quant_parts = [a for a in quant_acts[quant_name] if a.ndim == 2]
        if not x_true_parts or len(x_true_parts) != len(x_quant_parts):
            continue
        if any(a.shape != b.shape for a, b in zip(x_true_parts, x_quant_parts)):
            continue
        x_true = np.concatenate(x_true_parts, axis=0)
        x_quant = np.concatenate(x_quant_parts, axis=0)

        w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        scale = onnx.numpy_helper.to_array(c.ws_init).astype(np.float64)
        dim0, dim1 = w.shape

        if c.weight_transposed:
            w_nk = w  # already [N, K]
            scale_blocks = scale  # already [N, K / block_size]
        else:
            w_nk = w.T  # [K, N] -> [N, K]
            scale_blocks = scale.T  # [K / block_size, N] -> [N, K / block_size]
        if x_quant.shape[1] != w_nk.shape[1]:
            continue  # activation's feature dim doesn't match K; skip

        delta_x = x_true - x_quant  # [S, K]: upstream quantization's own corruption
        h = x_quant.T @ x_quant  # [K, K]: GPTQ's own Hessian, from the corrupted signal
        hinv_u = _inverse_hessian_cholesky(
            h, percdamp
        )  # inverse(h) == hinv_u.T @ hinv_u

        r = delta_x @ w_nk.T  # [S, N]: this row's own float weight times the corruption
        c_mat = x_quant.T @ r  # [K, N]: the new objective's linear-term coefficient
        shift = (
            hinv_u.T @ (hinv_u @ c_mat)
        ).T  # [N, K]: inverse(h) @ c_mat, transposed

        codes_nk = _gptq_quantize_columns(
            w_nk + shift, scale_blocks, c.block_size, h, percdamp, proc_block_size
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
