"""FOEM (2025, "First-Order Error Matters: Accurate Compensation for
Quantized Large Language Models", https://arxiv.org/abs/2507.11017).
onnxsim ports the *algorithm*, not any framework's code, per the same
rationale as :mod:`onnxsim.gptq`/:mod:`onnxsim.awq` (FOEM's own reference
implementation quantizes live PyTorch modules with no ONNX export path).

Read :mod:`onnxsim.gptq` first -- this module extends it directly, in the
sense the paper itself frames its own contribution: as an add-on
correction to GPTQ's own compensation, not a replacement for it.

GPTQ's own OBS-style correction, applied at each column ``i``, treats the
quantization error at that column (``w_col - code_col * s``) as *the*
error to propagate forward, implicitly via a second-order (Hessian-only)
Taylor argument -- one that quietly assumes the ``w_col`` it just quantized
is still (to first order) the *original* column of ``W``, undisturbed by
anything except its own rounding. That assumption is false by construction
partway through the pass: every earlier column's own forward-propagation
step (``w1[:, i+1:] -= outer(err, hinv1[i, i+1:])``, see
:func:`onnxsim.gptq._gptq_quantize_columns`) already nudged column ``i``'s
own pre-quantization value away from ``W``'s own true original column --
a **first-order deviation that accumulates column by column** and, by the
time a column late in the pass gets processed, is no longer negligible.
FOEM's own fix: alongside the fresh rounding error GPTQ already
compensates, also charge forward a (damped) fraction of *that* accumulated
deviation -- how far the column's current pre-quantization value has
already drifted from ``W``'s own untouched original column -- so later
columns' own compensation accounts for both sources of error, not just the
newest one.

This module's own version of the correction (a good-faith, numerically
verified reproduction of the paper's own described mechanism -- "measuring
the raw difference between the current compensated weights and the
untouched full weights, scaled by a small factor" -- rather than a
transcription of the paper's own derivation, which this module does not
claim to reproduce exactly): at column ``i``, alongside GPTQ's own
``err = (w_col - code_col * s) / d``, this module also computes
``drift = (w_col_before_quantizing - w_orig_col) / d`` (``w_orig_col``
being ``W``'s own untouched original column, never modified by any
previous correction) and propagates ``err + foem_beta * drift`` forward
instead of ``err`` alone -- ``foem_beta`` the "small
factor" the paper describes damping the first-order term by. This
module's own default (``0.005``) is deliberately conservative: swept
empirically against several toy calibration scenarios (see
``tests/test_foem.py``), a larger ``foem_beta`` here consistently
*increases* reconstruction error rather than reducing it (the added term
overshoots -- easy to do, since it is this module's own good-faith
reconstruction of the paper's mechanism rather than a verified transcription
of its exact derivation), while a small ``foem_beta`` gives a modest,
repeatable improvement over plain GPTQ across every scenario tested. This
module does not claim the improvement is universal or matches the paper's
own much larger reported gains (measured across real LLM benchmarks, not
one toy MatMul) -- only that, empirically, a small nonzero ``foem_beta``
does not hurt and sometimes measurably helps, which is the honest, verified
extent of what this port demonstrates. Reuses
:func:`onnxsim.gptq._inverse_hessian_cholesky` for the same Cholesky-based
``H^{-1}`` reformulation GPTQ itself already provides -- no new Hessian
machinery needed, exactly the paper's own "reuses the Cholesky factors
GPTQ already stores" efficiency claim.
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
from onnxsim.gptq import _inverse_hessian_cholesky


def _foem_quantize_columns(
    w_nk: np.ndarray,
    scale_blocks: np.ndarray,
    quant_block_size: int,
    h: np.ndarray,
    percdamp: float,
    proc_block_size: int,
    foem_beta: float,
) -> np.ndarray:
    """Returns FOEM-optimized integer codes for ``w_nk`` ([N, K], output
    channel first) -- see this module's own docstring. Mirrors
    :func:`onnxsim.gptq._gptq_quantize_columns`'s own structure and
    parameters exactly, adding only ``foem_beta`` (the damping factor on
    the additional first-order drift term).
    """
    n, k = w_nk.shape
    hinv = _inverse_hessian_cholesky(h, percdamp)

    codes_nk = np.zeros((n, k), dtype=np.float64)
    w_orig = w_nk  # never modified -- FOEM's own "untouched full weights"
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
            second_order_err = (w_col - code_col * s) / d
            # FOEM's own additional term: how far this column's own
            # pre-quantization value has already drifted from W's true
            # original column, due to every earlier column's own forward
            # propagation -- damped by foem_beta, not compensated in full.
            drift = (w_col - w_orig[:, k_abs]) / d
            err = second_order_err - foem_beta * drift
            err1[:, i] = err
            if i + 1 < bs:
                w1[:, i + 1 :] -= np.outer(err, hinv1[i, i + 1 :])

        if block_end < k:
            w_work[:, block_end:] -= err1 @ hinv[block_start:block_end, block_end:]

    return codes_nk


def apply_foem(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    foem_beta: float = 0.005,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes FOEM-style sequential, Hessian-*and*-first-order-drift-
    compensated rounding for every ``quantize_weight_only_int4``-quantized
    MatMul/Gemm layer present (by node output name) in both
    ``float_model`` and ``quantized_model``, using real activations
    captured from ``float_model``. See this module's own docstring for how
    this differs from :func:`onnxsim.gptq.apply_gptq` (which this module's
    own correction is an add-on to, not a replacement of).

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
            layer's Hessian from -- see :func:`onnxsim.gptq.apply_gptq`'s
            own parameter of the same name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param percdamp: Hessian damping factor, matching
            :func:`onnxsim.gptq.apply_gptq`'s own parameter of the same
            name and default
    :param proc_block_size: GPTQ's own column-processing block size,
            matching :func:`onnxsim.gptq.apply_gptq`'s own parameter of the
            same name and default
    :param foem_beta: damping factor on the additional first-order drift
            term (see this module's own docstring) -- ``0.0`` recovers
            plain GPTQ exactly; kept small by default since this module's
            own empirical sweeps show a larger value overshoots and
            increases error rather than reducing it
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its FOEM-optimized codes (same shape,
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
        codes_nk = _foem_quantize_columns(
            w_nk, scale_blocks, c.block_size, h, percdamp, proc_block_size, foem_beta
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
