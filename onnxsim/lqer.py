"""LQER (Zhang et al., 2024, "LQER: Low-Rank Quantization Error
Reconstruction for LLMs", https://arxiv.org/abs/2402.02446). onnxsim ports
the algorithm, not any framework's code, per the same rationale as
:mod:`onnxsim.low_rank_compensation`/:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`.

:mod:`onnxsim.low_rank_compensation` (ZeroQuant-V2's LoRC) already adds a
low-rank correction to a quantized layer's leftover error matrix
``residual = float_weight - dequantized_weight``, computed via that
matrix's own plain SVD -- no calibration data needed, but also blind to
which parts of ``residual`` actually matter: the Eckart-Young theorem's
"best rank-r approximation" is only best in an *unweighted*
Frobenius-norm sense, treating every input channel's contribution to
``residual`` as equally important. LQER's own contribution is exactly
this: since what actually reaches the model's output is ``X @ residual``
(not ``residual`` on its own), an input channel the real calibration data
drives with much larger activation energy than another deserves much more
weight in the low-rank fit, not equal weight.

**The technique**: for a candidate layer's ``residual`` (``[K, N]``,
ready for ``X @ residual``) and its own calibration-measured per-input-
channel activation RMS ``s`` (``[K]``, ``s_k = sqrt(mean_over_calibration(
X[:, k]^2))``), LQER row-scales the residual by ``s`` before taking the
SVD (``R' = diag(s) @ residual``), keeps the top ``r`` singular
components of ``R'``, and *un*-scales the result back
(``residual_approx = diag(1/s) @ R'_r``) before folding it into the same
``B @ A`` two-matmul correction :mod:`onnxsim.low_rank_compensation`
already uses. This is a weighted low-rank approximation minimizing
``sum_k s_k^2 * ||residual[k, :] - residual_approx[k, :]||^2`` --
exactly the AWQ-style "some channels matter more, weight by their own
activation energy" idea (:mod:`onnxsim.awq`'s own per-channel scaling),
applied here to *error compensation* rather than to the quantization
grid itself. Row-channels the calibration data never actually drives (an
all-zero or near-zero ``s_k``) contribute almost nothing to the weighted
objective, so LQER's fit spends its limited rank budget on the channels
that actually move the layer's real output, which
:mod:`onnxsim.low_rank_compensation`'s plain SVD cannot distinguish from
any other channel.

Everything else -- candidate matching (reusing
:mod:`onnxsim.adaround`'s ``_find_int4_matmul_candidates``/
``_node_outputs``), the resulting graph rewrite (two extra ``MatMul``
nodes plus an ``Add``, the original INT4 weight/scale left untouched) --
is identical to :mod:`onnxsim.low_rank_compensation`; only how ``B``/``A``
are computed differs. Falling back to an unweighted SVD (LoRC's own
formula) for any candidate whose input activation was never observed
during calibration keeps this a strict superset, never a regression,
of :mod:`onnxsim.low_rank_compensation`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _find_int4_matmul_candidates, _node_outputs
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data


def _per_channel_activation_rms(
    float_model: onnx.ModelProto,
    x_names: Sequence[str],
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]],
) -> Dict[str, np.ndarray]:
    """Per-input-channel activation RMS (``sqrt(mean(x_k^2))`` over every
    calibration sample and every leading/batch dimension) for each name in
    ``x_names``, over a plain 2-D (``[batch, K]``) activation only --
    matching :mod:`onnxsim.adaround`'s own "skip non-2-D activations"
    convention for the same reason (a batched/broadcast MatMul has no
    single well-defined per-channel statistic here).
    """
    probe_names = sorted(set(x_names))
    probe_model = _add_probe_outputs(float_model, probe_names)

    sum_sq: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim != 2:
                continue
            s = np.sum(x**2, axis=0)
            sum_sq[name] = sum_sq.get(name, 0.0) + s
            counts[name] = counts.get(name, 0) + x.shape[0]

    return {
        name: np.sqrt(s / max(counts[name], 1))
        for name, s in sum_sq.items()
        if counts[name] > 0
    }


def weighted_low_rank_correction(
    residual_kn: np.ndarray,
    channel_weights: Optional[np.ndarray],
    rank: int,
    eps: float = 1e-6,
) -> "tuple[np.ndarray, np.ndarray]":
    """The core LQER math, standalone and directly testable: a rank-``r``
    approximation of ``residual_kn`` (``[K, N]``) minimizing
    ``sum_k channel_weights[k]^2 * ||residual[k, :] - approx[k, :]||^2``
    instead of plain (unweighted) Frobenius error -- see this module's own
    docstring for the derivation (row-scale by ``channel_weights`` before
    the SVD, un-scale the result back afterward).

    :param residual_kn: ``[K, N]`` matrix to approximate
    :param channel_weights: ``[K]`` non-negative per-row weight (typically
            a per-input-channel activation RMS); ``None`` falls back to
            :mod:`onnxsim.low_rank_compensation`'s own plain, unweighted
            SVD (every channel weighted equally)
    :param rank: target rank ``r`` (**not** clamped to ``min(K, N)`` here --
            callers needing that do it themselves, matching
            :mod:`onnxsim.low_rank_compensation`'s own contract)
    :param eps: floor applied to ``channel_weights`` before dividing by it
    :returns: ``(b_kn, a_rn)`` with ``b_kn @ a_rn`` the rank-``r``
            approximation, shapes ``[K, r]``/``[r, N]``, float32
    """
    if channel_weights is None:
        u, sv, vt = np.linalg.svd(residual_kn, full_matrices=False)
        b_kn = u[:, :rank] * sv[np.newaxis, :rank]
        a_rn = vt[:rank, :]
        return b_kn.astype(np.float32), a_rn.astype(np.float32)

    s = np.maximum(channel_weights, eps)
    weighted = residual_kn * s[:, np.newaxis]
    u, sv, vt = np.linalg.svd(weighted, full_matrices=False)
    b_kn = (u[:, :rank] * sv[np.newaxis, :rank]) / s[:, np.newaxis]
    a_rn = vt[:rank, :]
    return b_kn.astype(np.float32), a_rn.astype(np.float32)


def apply_lqer(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    rank: int = 8,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    eps: float = 1e-6,
) -> onnx.ModelProto:
    """Adds an activation-weighted rank-``r`` low-rank correction to every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model`` --
    see this module's own docstring for the technique.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized) are left untouched.
            Assumes ``quantized_model`` was produced from ``float_model``
            without renaming any MatMul/Gemm node's own output tensor --
            true of every onnxsim ``quantize_*`` function.
    :param rank: the correction's rank ``r`` (clamped to
            ``min(r, N, K)`` per layer)
    :param calibration_data: representative input batches used to measure
            each layer's own per-input-channel activation RMS -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param eps: floor applied to each per-channel RMS before dividing by it,
            avoiding a division blow-up on a channel calibration barely
            drives
    :returns: ``quantized_model`` with every matched layer's output summed
            with a new rank-``r`` activation-weighted correction term (two
            chained ``MatMul`` nodes plus an ``Add``); the layer's own
            existing INT4 weight and scale are left completely untouched.
            A candidate whose activation was never observed during
            calibration (or isn't a plain 2-D ``[batch, K]`` shape) falls
            back to :mod:`onnxsim.low_rank_compensation`'s own unweighted
            SVD.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)

    candidates = _find_int4_matmul_candidates(float_model, quantized_model)
    if not candidates:
        return quantized_model

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    x_names = [c.float_node.input[0] for c in candidates]
    rms_by_name = _per_channel_activation_rms(
        float_model, x_names, calibration_data, providers
    )

    corrected = onnx.ModelProto()
    corrected.CopyFrom(quantized_model)
    graph = corrected.graph
    q_init = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    q_by_output = _node_outputs(graph)

    for c in candidates:
        wq_init = q_init[c.wq_name]
        codes = onnx.numpy_helper.to_array(wq_init).astype(np.float64)
        ws = onnx.numpy_helper.to_array(c.ws_init).astype(np.float64)
        scale_full = np.repeat(ws, c.block_size, axis=c.axis)
        slicer: List[slice] = [slice(None)] * codes.ndim
        slicer[c.axis] = slice(0, codes.shape[c.axis])
        w_dequant = codes * scale_full[tuple(slicer)]

        w_float = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
        residual = w_float - w_dequant  # original storage layout [dim0, dim1]
        residual_kn = residual if not c.weight_transposed else residual.T
        k, n = residual_kn.shape
        r = min(rank, k, n)
        if r <= 0:
            continue

        x_name = c.float_node.input[0]
        rms = rms_by_name.get(x_name)
        channel_weights = rms if rms is not None and rms.shape[0] == k else None
        b_kn, a_rn = weighted_low_rank_correction(residual_kn, channel_weights, r, eps)

        qn = q_by_output[c.output_name]

        prefix = f"{c.output_name}_lqer"
        b_name = _unique_name(f"{prefix}_b", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(b_kn, name=b_name))
        a_name = _unique_name(f"{prefix}_a", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(a_rn, name=a_name))

        old_output = qn.output[0]
        base_name = _unique_name(f"{prefix}_base", taken_names)
        qn.output[0] = base_name

        tmp_name = _unique_name(f"{prefix}_tmp", taken_names)
        tmp_node = onnx.helper.make_node(
            "MatMul",
            [x_name, b_name],
            [tmp_name],
            name=_unique_name(f"{prefix}_matmul1_node", taken_names),
        )
        lowrank_name = _unique_name(f"{prefix}_lowrank", taken_names)
        lowrank_node = onnx.helper.make_node(
            "MatMul",
            [tmp_name, a_name],
            [lowrank_name],
            name=_unique_name(f"{prefix}_matmul2_node", taken_names),
        )
        add_node = onnx.helper.make_node(
            "Add",
            [base_name, lowrank_name],
            [old_output],
            name=_unique_name(f"{prefix}_add_node", taken_names),
        )

        node_idx = next(i for i, nd in enumerate(graph.node) if nd is qn)
        graph.node.insert(node_idx + 1, tmp_node)
        graph.node.insert(node_idx + 2, lowrank_node)
        graph.node.insert(node_idx + 3, add_node)

    return corrected
