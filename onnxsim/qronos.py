"""Qronos (2025, ICLR 2026, "Qronos: Correcting the Past by Shaping the
Future in Post-Training Quantization", https://arxiv.org/abs/2505.11695) --
a sequential, whole-model generalization of :mod:`onnxsim.gptq`.

**How this differs from** :mod:`onnxsim.gptq`: GPTQ's own error-compensation
scope is narrow in a specific way. Within *one* layer, after quantizing a
column, it charges that column's own leftover rounding error forward onto
the *still-unquantized* columns of the *same* weight matrix (via the
Hessian-inverse-Cholesky OBC/OBS mechanism :mod:`onnxsim.gptq` already
implements) -- but it implicitly assumes the *activations* feeding that
layer are exact. GPTQ's own Hessian ``H = X^T X`` is always computed from
``float_model``'s real activations, even when correcting a layer deep inside
an otherwise-already-quantized network -- it never accounts for the fact
that, at actual deployment, that layer's real input already carries
whatever error every upstream layer's own quantization introduced. Qronos's
distinguishing contribution is an explicit, more complete correction that
accounts for **both** (1) this layer's own weight-rounding error (same as
GPTQ) **and** (2) the error already baked into this layer's activations
because upstream layers were quantized first -- a genuine cross-layer error
term GPTQ's own per-layer Hessian doesn't model.

The mechanism: consider a single layer with float weight ``W`` ([N, K],
output channel first). Let ``X_float`` be the calibration activations this
layer would see if every upstream layer were still exact (what
:mod:`onnxsim.gptq` uses), and ``X_quant`` the activations this layer
*actually* sees once upstream layers have already been quantized (computed
by running the calibration data through the model with every upstream layer
already Qronos-corrected). GPTQ minimizes ``||(W - Wq) @ X_float||^2``
-- matching the *quantized* layer's output to the *float* layer's output,
both computed on the *same*, exact input, so the only error being modeled is
this layer's own rounding. Qronos instead minimizes what actually matters at
deployment: ``||W @ X_float - Wq @ X_quant||^2`` -- the float network's ideal
output (clean input, clean weight) versus the deployed network's actual
output (corrupted input, quantized weight).

Let ``dX = X_quant - X_float`` -- the upstream error itself, a genuinely
*small*, bounded quantity (it is nothing but earlier layers' own INT4
rounding noise), as opposed to ``X_float``/``X_quant`` themselves, which can
be arbitrarily large or ill-conditioned. Writing ``H = X_quant^T @ X_quant``
(the Hessian of the *real* input this layer receives) and reusing
:func:`onnxsim.gptq._inverse_hessian_cholesky` for its damped-Cholesky-of-
``H^{-1}`` reformulation, define

    ``W_opt = W - W @ dX @ X_quant^T @ H^{-1}``

-- ``W``'s own value, shifted by a Hessian-weighted correction driven only
by ``dX``. By construction ``W_opt @ X_quant`` recovers ``W @ X_float``
whenever ``H^{-1}`` exactly inverts ``H`` (substitute ``X_quant = X_float +
dX`` and expand), so ``W_opt`` is a least-squares solution for matching
``W``'s own ideal (clean-input) target using the *real*, corrupted input.
Basic OLS orthogonality then means minimizing the original objective over
quantized ``Wq`` is *exactly* equivalent to minimizing
``||(W_opt - Wq) @ X_quant||^2`` -- precisely GPTQ's own per-column greedy
objective, unchanged, with ``W_opt`` standing in for the float weight and
``H`` (from ``X_quant``, not ``X_float``) standing in for GPTQ's Hessian.
That is this module's whole implementation: compute ``W_opt`` and ``H`` as
above, then hand them unchanged to
:func:`onnxsim.gptq._gptq_quantize_columns` -- the same Cholesky-based
least-squares machinery GPTQ itself uses, reused rather than reimplemented.

Deriving ``W_opt`` as a shift proportional to ``dX`` (rather than
reconstructing it from scratch via ``Y @ X_quant @ H^{-1}`` for some
separately-computed ideal target ``Y`` -- algebraically identical when
``H^{-1}`` is exact, but numerically very different) matters in practice:
GPTQ's own damping deliberately makes ``H^{-1}`` an *inexact* inverse of
``H`` along near-singular directions (the correlated-calibration-channel
scenario :mod:`onnxsim.gptq`'s own docstring motivates), and reconstructing
``W_opt`` from scratch amplifies that inexactness by ``W``'s own
(unrelated, potentially large) magnitude. Shifting *from* ``W`` by an amount
proportional to ``dX`` instead keeps that same inexactness scaled by ``dX``
-- small by construction -- and makes the reduction to plain GPTQ exact
(bit-for-bit, not just approximately) whenever ``dX`` is exactly zero:
``W_opt == W`` and :func:`_gptq_quantize_columns` is called with the
identical arguments :func:`onnxsim.gptq.apply_gptq` itself would use. This
is always the case for a layer with no already-quantized upstream layer
feeding it (e.g. the very first layer) -- Qronos is a strict generalization
of GPTQ, not a different algorithm bolted on.

The paper frames this per-column update as alternating between an "error
correction" step (undoing the input's own already-baked-in error, this
module's ``W_opt`` shift) and a "diffusion" step (spreading this column's
own new rounding error onto future columns, GPTQ's own mechanism) -- both
happen here, just factored as one least-squares retarget followed by one
unmodified GPTQ pass, rather than interleaved column-by-column; the two
formulations are algebraically equivalent for this module's own (fixed,
already-computed) scale grid.

**Scope**: correcting for cross-layer error requires processing layers in
real forward-execution order, quantizing each one and using its own
already-quantized output as the next layer's real ("corrupted") calibration
activation -- unlike :mod:`onnxsim.gptq`, which computes every layer's
Hessian from the untouched float model independently and in any order.
:func:`apply_qronos` does exactly this for every
``quantize_weight_only_int4``-quantized MatMul/Gemm layer present, ordering
candidates by their node's position in ``float_model.graph.node`` (the ONNX
IR spec requires a graph's nodes be topologically sorted, so this order is a
valid, and for a plain feedforward stack the *only*, forward-execution
order) and re-probing the partially-Qronos-corrected model before each
subsequent layer. A layer with more than one immediate quantized predecessor
(e.g. a residual join) still gets a *some* real upstream-corrected input --
just not disentangled per branch -- since probing reads whatever single
tensor actually reaches that layer's input, regardless of how many quantized
paths fed into it upstream.
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
from onnxsim.gptq import _gptq_quantize_columns, _inverse_hessian_cholesky


def apply_qronos(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Qronos-corrects every ``quantize_weight_only_int4``-quantized
    MatMul/Gemm layer present (by node output name) in both ``float_model``
    and ``quantized_model``, processing layers in forward-execution order so
    each one's correction accounts for every upstream layer's own
    already-applied quantization error, not just its own rounding. See this
    module's own docstring for the technique and how it differs from
    :func:`onnxsim.gptq.apply_gptq` (which this reduces to when a layer has
    no already-quantized upstream layer feeding it).

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
            layer's target/Hessian from -- see
            :func:`onnxsim.gptq.apply_gptq`'s own parameter of the same name
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param percdamp: Hessian damping factor -- see
            :func:`onnxsim.gptq.apply_gptq`'s own parameter of the same name
    :param proc_block_size: GPTQ's own column-processing block size -- see
            :func:`onnxsim.gptq._gptq_quantize_columns`
    :param providers: onnxruntime execution providers to run the model on
            when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its Qronos-corrected codes (same shape,
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

    node_order = {id(n): i for i, n in enumerate(float_model.graph.node)}
    candidates = sorted(candidates, key=lambda c: node_order[id(c.float_node)])

    probe_names = sorted({c.float_node.input[0] for c in candidates})
    float_probe = _add_probe_outputs(float_model, probe_names)
    float_activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        out = backend.run_model(float_probe, batch, providers=providers)
        for name in probe_names:
            float_activations[name].append(np.asarray(out[name], dtype=np.float64))

    working = onnx.ModelProto()
    working.CopyFrom(quantized_model)
    working_init = {t.name: t for t in working.graph.initializer}

    any_optimized = False
    for c in candidates:
        probe_name = c.float_node.input[0]
        x_float_batches = _activation_rows(float_activations[probe_name])
        if not x_float_batches:
            continue  # no usable activation (no feature axis); skip
        x_float = np.concatenate(x_float_batches, axis=0)

        working_probe = _add_probe_outputs(working, [probe_name])
        x_quant_batches = []
        for batch in calibration_data:
            out = backend.run_model(working_probe, batch, providers=providers)
            x_quant_batches.append(np.asarray(out[probe_name], dtype=np.float64))
        x_quant_batches = _activation_rows(x_quant_batches)
        # Keep the float and quantized row sets aligned: `dx` below is their
        # elementwise difference, so a batch usable on one side but not the
        # other (no feature axis) would misalign samples.
        if len(x_quant_batches) != len(x_float_batches):
            continue
        x_quant = np.concatenate(x_quant_batches, axis=0)

        w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        scale = onnx.numpy_helper.to_array(c.ws_init).astype(np.float64)
        dim0, dim1 = w.shape

        if c.weight_transposed:
            w_nk = w  # already [N, K]
            scale_blocks = scale  # already [N, K / block_size]
        else:
            w_nk = w.T  # [K, N] -> [N, K]
            scale_blocks = scale.T  # [K / block_size, N] -> [N, K / block_size]
        if x_float.shape[1] != w_nk.shape[1] or x_quant.shape[1] != w_nk.shape[1]:
            continue  # activation's feature dim doesn't match K; skip

        dx = x_quant - x_float  # [samples, K], the upstream error itself
        h = x_quant.T @ x_quant  # Hessian of the *real* (corrupted) input
        u = _inverse_hessian_cholesky(h, percdamp)
        h_inv = u.T @ u
        # W_opt = W - W @ dX @ X_quant^T @ H^{-1} -- see this module's own
        # docstring for why shifting *from* w_nk by a dX-proportional amount
        # (rather than reconstructing the target from scratch) is what makes
        # this numerically well-behaved and an exact GPTQ reduction when
        # dX == 0.
        w_opt_nk = w_nk - w_nk @ dx.T @ x_quant @ h_inv

        codes_nk = _gptq_quantize_columns(
            w_opt_nk, scale_blocks, c.block_size, h, percdamp, proc_block_size
        )

        codes_orig = codes_nk if c.weight_transposed else codes_nk.T
        assert codes_orig.shape == (dim0, dim1)
        working_init[c.wq_name].raw_data = _pack_int4(codes_orig.astype(np.int8))
        any_optimized = True

    if not any_optimized:
        return quantized_model
    return working
