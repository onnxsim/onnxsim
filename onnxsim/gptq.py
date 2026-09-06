"""GPTQ (Frantar et al., 2022, "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers", https://arxiv.org/abs/2210.17323) --
one of the notable weight-only PTQ techniques ``torchao`` implements
(``torchao/prototype/gptq/``, prototype status there; onnxsim ports the
*algorithm*, not that code -- see :mod:`onnxsim.awq`'s own docstring for why:
torchao has no ONNX export path). Third onnxsim-native PTQ technique
alongside :mod:`onnxsim.adaround` and :mod:`onnxsim.awq`, each pulling a
different lever on the exact same target scheme
(:func:`onnxsim.quantize_weight_only_int4`'s block-wise symmetric INT4):
AdaRound jointly optimizes every element's own rounding via gradient
descent; AWQ rescales whole input channels before quantizing; GPTQ instead
quantizes input channels **one at a time**, and after each one, propagates
its rounding error into every *not-yet-quantized* channel so their own
rounding can compensate for it -- a second-order (Hessian-based), greedy but
exact-per-step algorithm, not an iterative optimization.

The mechanism: for a layer's weight ``W`` ([N, K], output channel first) and
its calibration activations ``X`` ([samples, K]), the Hessian of the
layer's own squared reconstruction error with respect to ``W`` is
``H = X^T X`` (a ``[K, K]`` matrix, independent of ``W`` -- every output row
shares the same ``H``). Processing input channels left to right: quantize
channel ``i`` to its nearest grid point (using the same per-block scale
:func:`onnxsim.quantize_weight_only_int4` already computed), then charge the
resulting error to every remaining channel ``j > i`` in proportion to
``H^{-1}[i, j] / H^{-1}[i, i]`` -- the correction an optimal-brain-surgeon-
style argument shows exactly cancels that error's contribution to the
*layer's* reconstruction error, not just channel ``i``'s own. This module
uses the same Cholesky-based reformulation the GPTQ paper itself introduces
for efficiency (one ``H^{-1}`` Cholesky factorization up front, reused via
slicing for every column and block, rather than re-deriving a shrinking
inverse at every step) -- plain numpy, no framework dependency, matching
:mod:`onnxsim.adaround`/:mod:`onnxsim.awq`'s "post-hoc adjustment from real
activations" style.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _find_int4_matmul_candidates, _pack_int4
from onnxsim.bias_correction import _activation_rows, _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data


def _inverse_hessian_cholesky(h: np.ndarray, percdamp: float) -> np.ndarray:
    """Returns the upper-triangular Cholesky factor ``U`` of ``h``'s
    inverse (``inverse(h) == U.T @ U``) -- the GPTQ paper's own efficient
    reformulation, computed once per layer and reused (via slicing) for
    every column and block instead of re-deriving a shrinking inverse at
    each step. Channels ``h`` never saw any calibration activation for
    (an all-zero diagonal entry, i.e. a "dead" channel) get a fixed
    diagonal of 1 first, the standard GPTQ trick to keep ``h`` invertible
    without giving those channels any (meaningless, since nothing
    correlates with them) influence over other channels' corrections.
    """
    k = h.shape[0]
    diag = np.arange(k)
    dead = h[diag, diag] == 0.0
    h = h.copy()
    h[dead, dead] = 1.0
    damp = max(percdamp * float(np.mean(h[diag, diag])), 1e-8)
    h[diag, diag] += damp
    h_inv = np.linalg.inv(h)
    return np.linalg.cholesky(h_inv).T


def _gptq_quantize_columns(
    w_nk: np.ndarray,
    scale_blocks: np.ndarray,
    quant_block_size: int,
    h: np.ndarray,
    percdamp: float,
    proc_block_size: int,
) -> np.ndarray:
    """Returns GPTQ-optimized integer codes for ``w_nk`` ([N, K], output
    channel first), reusing ``scale_blocks``' existing per-(output channel,
    quantization group) scale (shape ``[N, K // quant_block_size]`` --
    unchanged from what :func:`onnxsim.quantize_weight_only_int4` already
    computed; GPTQ only changes which integer each element rounds to, never
    the scale). ``proc_block_size`` is GPTQ's own processing granularity
    (unrelated to ``quant_block_size``, the scale's granularity) -- how many
    columns' errors get propagated locally before a full cross-block
    update; larger values trade memory for fewer full-width updates, with
    no effect on the result other than floating-point summation order.
    """
    n, k = w_nk.shape
    hinv = _inverse_hessian_cholesky(h, percdamp)

    codes_nk = np.zeros((n, k), dtype=np.float64)
    w_work = w_nk.copy()

    for block_start in range(0, k, proc_block_size):
        block_end = min(block_start + proc_block_size, k)
        bs = block_end - block_start
        w1 = w_work[:, block_start:block_end].copy()
        err1 = np.zeros_like(w1)
        hinv1 = hinv[block_start:block_end, block_start:block_end]

        for i in range(bs):
            k_abs = block_start + i
            group = k_abs // quant_block_size
            s = scale_blocks[:, group]  # [N]
            w_col = w1[:, i]
            code_col = np.clip(np.round(w_col / s), -7.0, 7.0)
            codes_nk[:, k_abs] = code_col
            d = hinv1[i, i]
            err = (w_col - code_col * s) / d
            err1[:, i] = err
            if i + 1 < bs:
                w1[:, i + 1 :] -= np.outer(err, hinv1[i, i + 1 :])

        if block_end < k:
            w_work[:, block_end:] -= err1 @ hinv[block_start:block_end, block_end:]

    return codes_nk


def apply_gptq(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes GPTQ-style sequential, Hessian-compensated rounding for
    every ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present
    (by node output name) in both ``float_model`` and ``quantized_model``,
    using real activations captured from ``float_model``. See this module's
    own docstring for the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized), or whose activation
            input has no feature axis at all (rank < 2) are left
            untouched; a higher-rank ``[batch, seq, K]`` activation is
            flattened to ``[batch * seq, K]``, which is exact. Assumes
            ``quantized_model`` was produced from ``float_model`` without
            renaming any MatMul/Gemm node's own output tensor -- true of
            every onnxsim ``quantize_*`` function.
    :param calibration_data: representative input batches to compute each
            layer's Hessian from. Each batch is a
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
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added to every diagonal entry before inversion), matching the
            GPTQ paper's own default -- keeps the inversion numerically
            stable when calibration data doesn't fully activate (or
            correlates too tightly across) a layer's input channels
    :param proc_block_size: GPTQ's own column-processing block size (not
            the quantization scale's own block size, which this always
            reuses unchanged from ``quantized_model``) -- see
            :func:`_gptq_quantize_columns`
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its GPTQ-optimized codes (same shape,
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

    probe_names = sorted({c.float_node.input[0] for c in candidates})
    float_probe = _add_probe_outputs(float_model, probe_names)

    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        out = backend.run_model(float_probe, batch, providers=providers)
        for name in probe_names:
            activations[name].append(np.asarray(out[name], dtype=np.float64))

    optimized: Dict[str, np.ndarray] = {}
    for c in candidates:
        acts = _activation_rows(activations[c.float_node.input[0]])
        if not acts:
            continue  # no usable activation (no feature axis); skip
        x = np.concatenate(acts, axis=0)

        w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        scale = onnx.numpy_helper.to_array(c.ws_init).astype(np.float64)
        dim0, dim1 = w.shape

        if c.weight_transposed:
            w_nk = w  # already [N, K]
            scale_blocks = scale  # already [N, K / block_size]
        else:
            w_nk = w.T  # [K, N] -> [N, K]
            scale_blocks = scale.T  # [K / block_size, N] -> [N, K / block_size]
        if x.shape[1] != w_nk.shape[1]:
            continue  # activation's feature dim doesn't match K; skip

        h = x.T @ x
        codes_nk = _gptq_quantize_columns(
            w_nk, scale_blocks, c.block_size, h, percdamp, proc_block_size
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
