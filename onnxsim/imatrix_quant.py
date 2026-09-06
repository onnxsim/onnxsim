"""llama.cpp's "importance matrix" (``imatrix``) -- an activation-based
per-input-channel importance score, gathered by running calibration data
through the float model, that llama.cpp's quantizers (Q4_K, IQ4_NL, ...)
use to bias their round-to-nearest grid search toward minimizing
*importance-weighted* squared error rather than plain squared error.
Channels that matter more to the model's actual output (i.e. that carry
more activation energy) get a tighter fit; channels that are barely used
get less.

**What "importance" means here, and why mean-square activation is the
right proxy.** For a matched ``Y = X @ W`` layer, quantizing weight
element ``w_{r,j}`` (output row ``r``, input channel ``j``) with error
``e_{r,j} = w_{r,j} - \\hat w_{r,j}`` perturbs output element ``Y_{t,r}``
by ``sum_j e_{r,j} * X_{t,j}`` for calibration token ``t``. Treating the
per-channel error terms as uncorrelated (the usual diagonal
approximation used to justify e.g. GPTQ's/AWQ's own per-channel-weighted
objectives), the *expected squared output error* this introduces is
approximately ``sum_j e_{r,j}^2 * E[X_{t,j}^2]`` -- i.e. exactly a
per-channel-weighted sum of squared weight-quantization errors, with
weight ``importance_j = E[X_{t,j}^2]``, the channel's own mean-square
activation magnitude. That is precisely llama.cpp's own ``imatrix``
quantity (see llama.cpp's ``imatrix`` tool and e.g.
``ggml_compute_forward_out_prod`` /``IMatrixCollector`` in
``examples/imatrix``, which accumulate ``sum(x^2)`` per input channel
across calibration batches). Minimizing importance-weighted squared
quantization error is therefore, up to the diagonal approximation above,
minimizing the calibration data's own expected output error -- not just
an arbitrary reweighting.

**What this module does NOT attempt.** llama.cpp's real ``imatrix`` tool
runs a full model (with a real tokenizer and a real text corpus) through
``llama.cpp`` itself and dumps activation statistics to a ``.imatrix``
file consumed later by ``llama-quantize``. onnxsim has no calibration
data pipeline of that kind (see :mod:`onnxsim.calibration`'s own
docstring), so, exactly like every other calibration-based pass already
in this repository (:func:`onnxsim.apply_llm_int8`,
:func:`onnxsim.correct_bias`, :func:`onnxsim.quantize_static`), this
module computes the same *quantity* (per-channel mean-square activation)
from synthetic/random calibration data
(:func:`onnxsim.generate_random_calibration_data` by default, or any
user-supplied representative batches) run through the float model via
:func:`onnxsim.backend.run_model` -- onnxruntime when installed, onnx's
own pure-Python ``ReferenceEvaluator`` otherwise (no hard onnxruntime
dependency, matching every other onnxsim calibration-based pass).

**The quantizer itself.** Unlike :mod:`onnxsim.gguf_kquant` (Q4_K) and
:mod:`onnxsim.iq4_nl` (IQ4_NL), which reproduce specific GGUF block
*formats* (their own sub-block/codebook structure), this module reuses
this repository's own plain, already-verified symmetric block-wise INT4
scheme (:func:`onnxsim.omniquant._quantize_blockwise_int4_with_clip`,
also reused by :mod:`onnxsim.quarot`/:mod:`onnxsim.spinquant`: one scale
per ``block_size``-element block along the reduction axis ``K``, code
range ``[-7, 7]``) as the **plain, unweighted baseline** -- and adds, as
this module's own contribution, an importance-weighted grid search over
that same block's scale: instead of the plain block scale
(``max(abs(block)) / 7``), a small set of candidate scales around it is
evaluated, and whichever minimizes ``sum(importance_j * (w_j - dequant_j)
** 2)`` over the block is kept. This is the same "grid-search the
quantization grid against a weighted objective" idea llama.cpp's own
``make_qx_quants``/Q4_K encoder implement (see
:mod:`onnxsim.gguf_kquant`'s own honesty note on why this module does not
claim byte-for-bit parity with llama.cpp's own encoder internals, only
the same mechanism), applied on top of this repository's own existing
verified primitive rather than a new block format.

Weight-only, folded directly into a new float32 initializer (the
quantize-dequantize round trip), exactly the pattern
:func:`onnxsim.apply_gguf_q4_k_quantization`/
:func:`onnxsim.apply_iq4_nl_quantization` already established -- no new
ONNX tensor type below INT4 is needed since the result is stored as an
ordinary float32 tensor, and no new graph nodes are added (the importance
matrix only informs an *offline* weight-encoding decision, exactly how
llama.cpp itself uses it -- the runtime dequantization is unaffected).

**Scope**: only ``MatMul``/"vanilla" ``Gemm`` (via
:func:`onnxsim.llm_int8._match_matmul_like`) with a constant 2-D float32
weight are matched -- unlike :mod:`onnxsim.gguf_kquant`/
:mod:`onnxsim.iq4_nl` (which flatten weights of any rank, since they need
no activation), this module's importance score is only meaningful when a
weight's reduction axis ``K`` lines up one-to-one with a plain 2-D
activation's own feature axis, so ``Conv`` (whose weight has separate
spatial and channel axes) is out of scope, matching
:mod:`onnxsim.quarot`/:mod:`onnxsim.deepseek_fp8`'s own Conv-exclusion
decision. A layer whose reduction dimension ``K`` is not divisible by
``block_size``, or whose activation was never observed as a plain 2-D (or
higher-rank-but-flattenable) float tensor of the right width, is left
unquantized.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.llm_int8 import _match_matmul_like
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip

_MAX_CODE = 7.0  # symmetric 4-bit code range [-7, 7], matching
# onnxsim.omniquant._quantize_blockwise_int4_with_clip's own convention.


def compute_activation_importance(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Runs ``model`` over ``calibration_data`` (synthetic random data by
    default) and returns, for every activation input feeding a matched
    MatMul/vanilla-Gemm layer, that tensor's own per-channel mean-square
    activation magnitude -- llama.cpp's own ``imatrix`` quantity (see this
    module's own docstring for why mean-square activation is the right
    proxy for a channel's importance to the model's output).

    :param model: the float onnx ModelProto or file path to gather
            activation statistics from
    :param calibration_data: representative input batches, e.g. from
            :func:`onnxsim.generate_random_calibration_data` or
            :func:`onnxsim.load_huggingface_calibration_data`. Generated
            with :func:`onnxsim.generate_random_calibration_data` (using
            ``num_samples``/``seed``) when omitted.
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration
            on (see :func:`onnxsim.backend.run_model`); onnxruntime is
            used when installed, otherwise onnx's own pure-Python
            ``ReferenceEvaluator`` -- no hard onnxruntime dependency
    :returns: ``{activation_tensor_name: importance}``, ``importance`` a
            1-D float64 array of length ``activation.shape[-1]``, the
            mean of ``activation ** 2`` over every other (batch/token)
            dimension across all calibration batches. A tensor that was
            never observed with at least rank 1 across the calibration
            data is omitted.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    initializer_map = {t.name: t for t in model.graph.initializer}
    probe_names = set()
    for node in model.graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, _bias_name, _weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        probe_names.add(x_name)
    if not probe_names:
        return {}

    sorted_probe_names = sorted(probe_names)
    probe_model = _add_probe_outputs(model, sorted_probe_names)

    sum_sq: Dict[str, np.ndarray] = {}
    count: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in sorted_probe_names:
            x = np.asarray(result[name])
            if x.ndim == 0 or not np.issubdtype(x.dtype, np.floating):
                continue
            flat = x.reshape(-1, x.shape[-1]).astype(np.float64)  # [rows, channels]
            batch_sum_sq = np.sum(flat**2, axis=0)
            if name not in sum_sq:
                sum_sq[name] = batch_sum_sq
                count[name] = flat.shape[0]
            else:
                sum_sq[name] += batch_sum_sq
                count[name] += flat.shape[0]

    return {name: sum_sq[name] / count[name] for name in sum_sq if count[name] > 0}


def quantize_dequantize_int4_imatrix(
    w_nk: np.ndarray,
    importance_k: np.ndarray,
    block_size: int = 32,
    num_scale_candidates: int = 41,
    scale_search_range: Tuple[float, float] = (0.4, 1.6),
) -> np.ndarray:
    """Importance-weighted, block-wise INT4 quantize-dequantize round trip
    of ``w_nk`` (``[N, K]``, output channel first; ``K`` must be a
    multiple of ``block_size``).

    Starts from the same plain per-block scale
    :func:`onnxsim.omniquant._quantize_blockwise_int4_with_clip` already
    uses (``max(abs(block)) / 7``), then grid-searches
    ``num_scale_candidates`` candidate scales spaced across
    ``base_scale * scale_search_range`` and keeps, independently per
    ``(output row, block)``, whichever candidate minimizes
    ``sum(importance_j * (w_j - dequant_j) ** 2)`` over that block's own
    ``importance_k`` slice -- see this module's own docstring for why
    that quantity approximates expected output error. Since the plain
    per-block scale (candidate factor ``1.0``) is always one of the
    candidates tried, this can never do *worse* than
    :func:`quantize_dequantize_int4_plain` under uniform ``importance_k``
    (weighted SSE then equals plain unweighted SSE) -- though it need not
    exactly reproduce it: plain min/max scaling is not itself
    unweighted-SSE-optimal in general (clipping a mild outlier slightly can
    reduce total squared error), so the search can legitimately do
    strictly better even with every channel weighted equally. Both
    properties are verified in this module's own test file.

    :param w_nk: weight, ``[N, K]``, any float dtype
    :param importance_k: per-input-channel importance, length ``K``
    :param block_size: elements per quantization block along ``K``
    :param num_scale_candidates: number of candidate scale factors tried
            per block
    :param scale_search_range: ``(low, high)`` multipliers applied to the
            plain per-block scale to build the candidate grid
    :returns: ``w_nk``'s quantize-dequantize round trip, float64, same
            shape as ``w_nk``
    """
    w_nk = np.asarray(w_nk, dtype=np.float64)
    importance_k = np.asarray(importance_k, dtype=np.float64)
    n, k = w_nk.shape
    if k % block_size != 0:
        raise ValueError(f"K ({k}) must be a multiple of block_size ({block_size})")
    if importance_k.shape != (k,):
        raise ValueError(
            f"importance_k must have shape ({k},), got {importance_k.shape}"
        )

    num_blocks = k // block_size
    blocks = w_nk.reshape(n, num_blocks, block_size)
    imp_blocks = importance_k.reshape(num_blocks, block_size)

    base_scale = (
        np.maximum(np.abs(blocks).max(axis=2), 1e-12) / _MAX_CODE
    )  # [N, num_blocks]
    low, high = scale_search_range
    best_err = np.full((n, num_blocks), np.inf)
    best_scale = base_scale.copy()
    for factor in np.linspace(low, high, num_scale_candidates):
        trial_scale = np.maximum(base_scale * factor, 1e-12)  # [N, num_blocks]
        codes = np.clip(
            np.round(blocks / trial_scale[:, :, np.newaxis]), -_MAX_CODE, _MAX_CODE
        )
        recon = codes * trial_scale[:, :, np.newaxis]
        weighted_err = np.sum(
            imp_blocks[np.newaxis, :, :] * (blocks - recon) ** 2, axis=2
        )
        improved = weighted_err < best_err
        best_err = np.where(improved, weighted_err, best_err)
        best_scale = np.where(improved, trial_scale, best_scale)

    best_codes = np.clip(
        np.round(blocks / best_scale[:, :, np.newaxis]), -_MAX_CODE, _MAX_CODE
    )
    return (best_codes * best_scale[:, :, np.newaxis]).reshape(n, k)


def quantize_dequantize_int4_plain(
    w_nk: np.ndarray, block_size: int = 32
) -> np.ndarray:
    """Plain (unweighted) block-wise INT4 quantize-dequantize round trip of
    ``w_nk`` (``[N, K]``) -- the baseline
    :func:`quantize_dequantize_int4_imatrix` is compared against, built
    directly from this repository's own already-verified
    :func:`onnxsim.omniquant._quantize_blockwise_int4_with_clip`
    (``clip_ratio=1.0``, i.e. the plain min/max block scale).
    """
    w_nk = np.asarray(w_nk, dtype=np.float64)
    codes_nk, scale_blocks = _quantize_blockwise_int4_with_clip(w_nk, block_size, 1.0)
    return codes_nk * np.repeat(scale_blocks, block_size, axis=1)


def apply_imatrix_quantization(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    block_size: int = 32,
    num_scale_candidates: int = 41,
    scale_search_range: Tuple[float, float] = (0.4, 1.6),
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """Weight-only-quantizes every matched MatMul/vanilla-Gemm float32
    weight to INT4 (folded as a float32 quantize-dequantize round trip,
    see this module's own docstring), using real calibration activations
    to bias each block's scale search toward minimizing
    importance-weighted squared error (llama.cpp's ``imatrix`` idea).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute the
            importance matrix from -- see
            :func:`compute_activation_importance`
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration
            on (see :func:`onnxsim.backend.run_model`)
    :param block_size: elements per weight quantization block along
            ``K``, matching :func:`onnxsim.quantize_weight_only_int4`'s
            own default
    :param num_scale_candidates: candidate scale factors evaluated per
            block -- see :func:`quantize_dequantize_int4_imatrix`
    :param scale_search_range: multiplier range for the candidate scale
            grid -- see :func:`quantize_dequantize_int4_imatrix`
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by its
            importance-weighted INT4 round trip, same name and shape as
            the node's original weight input, stored under a new
            initializer name (the original initializer is left in the
            graph, unused). A layer with a non-constant/non-2-D weight, a
            reduction dimension not divisible by ``block_size``, or whose
            activation was never observed as a real-valued tensor across
            the calibration data, is left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    skip_names = set(skip_names) if skip_names is not None else frozenset()

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)

    candidates = []  # (node, x_name, w_name, weight_transposed)
    for node in graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        x_name, w_name, _bias_name, weight_transposed = match
        if w_name in skip_names:
            continue
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((node, x_name, w_name, weight_transposed))
    if not candidates:
        return out

    importance = compute_activation_importance(
        out,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        providers=providers,
    )

    quantized_names: Dict[str, str] = {}
    for node, x_name, w_name, weight_transposed in candidates:
        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape

        importance_k = importance.get(x_name)
        if importance_k is None or importance_k.shape[0] != k or k % block_size != 0:
            continue

        new_name = quantized_names.get(w_name)
        if new_name is None:
            w_quant_nk = quantize_dequantize_int4_imatrix(
                w_nk,
                importance_k,
                block_size=block_size,
                num_scale_candidates=num_scale_candidates,
                scale_search_range=scale_search_range,
            )
            w_quant = w_quant_nk if weight_transposed else w_quant_nk.T

            new_name = _unique_name(f"{w_name}_imatrix", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(w_quant.astype(np.float32), name=new_name)
            )
            quantized_names[w_name] = new_name
        node.input[1] = new_name

    return out
