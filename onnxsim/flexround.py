"""FlexRound -- Learnable Rounding based on Element-wise Division for
Post-Training Quantization (Lee et al., 2023, ICML 2023,
https://arxiv.org/abs/2306.00317). Fourth onnxsim-native PTQ technique
alongside :mod:`onnxsim.adaround`, :mod:`onnxsim.gptq`, and
:mod:`onnxsim.awq`, each pulling a different lever on the exact same target
scheme (:func:`onnxsim.quantize_weight_only_int4`'s block-wise symmetric
INT4): AdaRound perturbs each weight element's own rounding *additively*
(:mod:`onnxsim.adaround`'s rectified-sigmoid ``delta``); GPTQ quantizes
input channels one at a time and propagates each one's rounding error into
every not-yet-quantized channel via a Hessian; AWQ rescales whole input
channels before quantizing. FlexRound instead reparametrizes the *divisor
itself*, multiplicatively, per weight element.

The paper's own formula (Eq. 1-2, per-tensor/per-channel uniform PTQ, a
linear layer): the quantized weight is ``W_hat = s1 * round(W / S)``, where
the *effective* per-element divisor ``S = s1 (x) S2 (x) s3`` ((x) = the
paper's element-wise product) decomposes a common quantization grid size
``s1`` (what a plain round-to-nearest quantizer already picks -- a scalar
or a per-output-channel vector in the paper), an element-wise learnable
correction ``S2`` (same shape as ``W``), and a per-output-channel learnable
correction ``s3``. (A 2D convolution's ``S`` gets a further per-input-
channel factor ``s4``; not applicable here since this module, like
:mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/:mod:`onnxsim.awq`, only
targets MatMul/Gemm.) All of ``S2``/``s3`` are initialized to 1, so
optimization starts exactly at round-to-nearest and only departs from it as
gradient descent (Adam, straight-through through ``round()`` -- Bengio
et al., 2013 -- the same "post-hoc adjustment from real activations" style
:mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/:mod:`onnxsim.awq` all share)
finds it worthwhile against ``||W X - W_hat X||^2`` on real calibration
activations.

The paper's own reciprocal-rule argument (Proposition 3.1) is precisely why
this differs from AdaRound's additive perturbation: because ``S`` divides
rather than adds, ``d(W/S)/dS = -W/S^2`` is proportional to ``W`` itself,
so (under a straight-through gradient) a large-magnitude weight naturally
receives a proportionally larger nudge and a small-magnitude one a
proportionally smaller one. AdaRound's additive ``delta``, by contrast, is
squashed into the same fixed range regardless of the weight's own
magnitude, so it costs a large weight the same absolute nudge as a small
one -- the paper's stated reason FlexRound's own reparametrization scales
better to heavy-tailed weight (or, in the paper's activation-quantization
experiments, activation) magnitude distributions than a fixed-range
additive perturbation does.

**What's ported vs. not, relative to the paper:**

- The paper jointly *learns* ``s1`` (the quantization grid size) alongside
  ``S2``/``s3``. This module does not: like every other onnxsim PTQ pass
  targeting :func:`onnxsim.quantize_weight_only_int4`'s output, it keeps
  the block scale ``quantized_model`` already computed completely
  unchanged and only rewrites *which integer* each element rounds to --
  ``s1`` here is fixed at that pre-existing per-(block, output channel)
  scale rather than optimized. This is the same scope restriction
  :mod:`onnxsim.adaround` and :mod:`onnxsim.gptq` both make, for the same
  reason: the scale is a shared tensor :func:`onnxsim.quantize_weight_only_int4`
  already committed to the graph, not a free parameter this pass owns.
- ``S2`` (element-wise) and ``s3`` (per-output-channel) are both
  implemented, matching the paper's own linear-layer formula (Eq. 2's
  ``S = s1 (x) S2 (x) s3``) exactly. The paper's ``s4`` (an additional
  per-input-channel factor) only applies to its 2D convolution formula,
  which is out of scope here for the reason above.
- The paper leaves the positivity constraint on ``S2``/``s3`` unspecified
  beyond "positive and learnable." This module enforces it by optimizing
  in log-space (``S2 = exp(v2)``, ``s3 = exp(v3)``, ``v2``/``v3``
  initialized to 0) -- a standard reparametrization for a
  positive-multiplicative learned quantity, and, as a useful side effect,
  algebraically simplifies the gradient (``d(S)/d(v2) == S`` and
  ``d(S)/d(v3) == S``; see ``_optimize_divisor`` below).
- The paper's own forward pass applies ``round()`` every iteration
  (straight-through for the backward pass). This module instead keeps the
  *continuous* relaxation ``clip(W / S, n_min, n_max)`` throughout
  optimization and rounds once at the very end -- the same structure
  :mod:`onnxsim.adaround` uses for its own relaxation. Unlike AdaRound,
  no regularization/annealing schedule pulls this relaxation toward a hard
  decision (the paper's own formulation doesn't have one either): the
  reconstruction loss alone shapes ``S2``/``s3`` from start to finish.
- The paper also explores W4A4 / activation-quantization variants (jointly
  reparametrizing the activation's own quantizer) and block-wise
  reconstruction for large language models; this module only implements
  weight-only quantization with plain, whole-layer reconstruction --
  matching every other onnxsim PTQ pass (see :mod:`onnxsim.adaround`'s own
  docstring) and this repo's existing calibration-driven-pass style.

No calibration data beyond what :mod:`onnxsim.calibration` already
provides, activation quantization, or gradient framework other than what
this module implements itself is required -- everything here is plain
numpy, matching :mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/
:mod:`onnxsim.awq`'s shared style.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _find_int4_matmul_candidates, _pack_int4
from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data


def _optimize_divisor(
    w_nk: np.ndarray,
    scale_nk: np.ndarray,
    x: np.ndarray,
    n_min: float,
    n_max: float,
    num_iterations: int,
    learning_rate: float,
    log_clip: float,
) -> np.ndarray:
    """Returns FlexRound-optimized integer codes (shape like ``w_nk``,
    values in ``[n_min, n_max]``) for one weight matrix, given its layer's
    activation ``x`` (shape ``[num_samples, K]``) captured from the float
    model. ``w_nk``/``scale_nk`` are the float weight and its (already
    block-broadcast) per-element scale, both laid out ``[N, K]`` (output
    channel first) regardless of the op's own storage layout.

    Implements the paper's Eq. 2 for a linear layer, with ``s1`` fixed at
    ``scale_nk`` (see this module's own docstring for why): the effective
    divisor is ``S = scale_nk * S2 * s3``, ``S2`` element-wise (``[N, K]``)
    and ``s3`` per-output-channel (``[N, 1]``), both parametrized in
    log-space (``S2 = exp(v2)``, ``s3 = exp(v3)``) to keep them positive by
    construction and initialized at ``v2 = v3 = 0`` (``S2 = s3 = 1``, so the
    very first forward pass is exactly round-to-nearest at ``scale_nk``,
    the paper's own stated initialization).

    The paper's own forward pass applies ``round()`` every iteration
    (straight-through for the backward pass). This implementation instead
    keeps the *continuous* relaxation ``clip(ratio, n_min, n_max)`` (no
    ``round()``) throughout the optimization loop, and only rounds once at
    the very end to produce integer codes -- the same structure
    :func:`onnxsim.adaround._optimize_rounding` uses for its own relaxation
    (optimize a smooth surrogate throughout, discretize once at the end).
    Empirically, re-rounding every iteration makes the loss surface a step
    function of ``v2``/``v3``, which destabilizes Adam's per-parameter
    second-moment estimate for no benefit here (there's no annealed
    regularizer, unlike AdaRound's, pulling values toward a hard decision
    mid-optimization that would need it); the continuous-throughout version
    below is mathematically the direct straight-through gradient of the
    quantity it actually optimizes, verified against finite differences.

    Gradients use a straight-through estimator through ``round()``
    (``codes ~= ratio`` for backward purposes, zeroed wherever clamping to
    ``[n_min, n_max]`` already saturated -- the same "active" masking
    :func:`onnxsim.adaround._optimize_rounding` applies to its own
    relaxation), combined with the log-parametrization's own simplification
    (``d(S)/d(v2) == S`` element-wise, ``d(S)/d(v3) == S`` summed over
    ``K``) so no explicit Jacobian of ``S2``/``s3`` w.r.t. ``S`` is needed.
    """
    y_float = x @ w_nk.T  # [num_samples, N]
    n = x.shape[0] * w_nk.shape[0]

    v2 = np.zeros_like(w_nk)
    v3 = np.zeros((w_nk.shape[0], 1), dtype=w_nk.dtype)

    m2 = np.zeros_like(v2)
    u2 = np.zeros_like(v2)
    m3 = np.zeros_like(v3)
    u3 = np.zeros_like(v3)
    adam_beta1, adam_beta2, adam_eps = 0.9, 0.999, 1e-8

    for t in range(num_iterations):
        s2 = np.exp(v2)
        s3 = np.exp(v3)
        s = scale_nk * s2 * s3  # [N, K], effective (element-wise) divisor

        ratio = w_nk / s
        active = (ratio > n_min) & (ratio < n_max)
        codes = np.clip(ratio, n_min, n_max)  # continuous surrogate this loop optimizes
        w_hat = codes * scale_nk  # deployed dequant always uses scale_nk, not s

        y_hat = x @ w_hat.T
        dl_dy = 2.0 * (y_hat - y_float) / n
        dl_dw_hat = dl_dy.T @ x  # [N, K]

        dw_hat_dratio = np.where(active, scale_nk, 0.0)
        dratio_ds = np.where(active, -w_nk / (s * s), 0.0)
        grad_s = dl_dw_hat * dw_hat_dratio * dratio_ds

        grad_sv = grad_s * s  # == dL/d(v2) elementwise, and dL/d(v3) pre-sum
        grad_v2 = grad_sv
        grad_v3 = grad_sv.sum(axis=1, keepdims=True)

        m2 = adam_beta1 * m2 + (1.0 - adam_beta1) * grad_v2
        u2 = adam_beta2 * u2 + (1.0 - adam_beta2) * (grad_v2 * grad_v2)
        m2_hat = m2 / (1.0 - adam_beta1 ** (t + 1))
        u2_hat = u2 / (1.0 - adam_beta2 ** (t + 1))
        v2 = v2 - learning_rate * m2_hat / (np.sqrt(u2_hat) + adam_eps)
        v2 = np.clip(v2, -log_clip, log_clip)

        m3 = adam_beta1 * m3 + (1.0 - adam_beta1) * grad_v3
        u3 = adam_beta2 * u3 + (1.0 - adam_beta2) * (grad_v3 * grad_v3)
        m3_hat = m3 / (1.0 - adam_beta1 ** (t + 1))
        u3_hat = u3 / (1.0 - adam_beta2 ** (t + 1))
        v3 = v3 - learning_rate * m3_hat / (np.sqrt(u3_hat) + adam_eps)
        v3 = np.clip(v3, -log_clip, log_clip)

    s_final = scale_nk * np.exp(v2) * np.exp(v3)
    return np.clip(np.round(w_nk / s_final), n_min, n_max)


def apply_flexround(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    learning_rate: float = 0.05,
    log_clip: float = 4.0,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes FlexRound-style learnable-division rounding for every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model``, using
    real activations captured from ``float_model``. See this module's own
    docstring for the technique.

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
    :param calibration_data: representative input batches to optimize the
            divisor on. Each batch is a ``{input_name: np.ndarray}`` dict
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
    :param learning_rate: Adam learning rate for the log-space element-wise
            (``S2``) and per-output-channel (``s3``) divisor corrections.
            The paper's own hyperparameter tables tune this per model/layer
            (values from ``1e-6`` to ``1e-1`` appear across its
            experiments) -- this reparametrization's own sensitivity to
            learning rate, not just a quirk of this port; too high
            overshoots past round-to-nearest's own optimum, too low never
            leaves it within a practical iteration budget
    :param log_clip: clamps ``log(S2)``/``log(s3)`` to
            ``[-log_clip, log_clip]`` after every step -- a numerical
            safety bound (not part of the paper's own formulation) keeping
            the learned divisor from drifting so far from the original
            scale that ``round(W / S)`` becomes numerically degenerate
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its FlexRound-optimized codes (same
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

        codes_nk = _optimize_divisor(
            w_nk,
            scale_nk,
            x,
            n_min=-7.0,
            n_max=7.0,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            log_clip=log_clip,
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
