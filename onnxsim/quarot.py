"""QuaRot (Ashkboos et al., 2024, "QuaRot: Outlier-Free 4-Bit Inference in
Rotated LLMs", https://arxiv.org/abs/2404.00456). onnxsim ports the
algorithm, not any framework's code, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.quip_sharp` (QuaRot's
own reference implementation rotates and quantizes live PyTorch weights
with custom CUDA kernels, with no ONNX export path).

Every weight-only INT4 scheme already in onnxsim (`quantize_weight_only_int4`
and everything built on it -- :mod:`onnxsim.spinquant`, :mod:`onnxsim.spqr`,
:mod:`onnxsim.quip_sharp`) leaves the *activation* flowing into the MatMul in
float32: only ``W`` ever gets quantized, so the MatMul itself still runs at
float precision and pays float memory bandwidth for ``X``.
:mod:`onnxsim.smoothquant` migrates activation outliers into the weight so a
*separate* W8A8 quantizer (e.g. :func:`onnxsim.quantize_static`) can handle
both operands -- but that still tops out at INT8, since a fixed per-channel
migration can only shrink outliers, not remove the need for extra headroom
entirely. QuaRot's own contribution: the same random-rotation idea
:mod:`onnxsim.quip_sharp` already uses to make a *weight* look
incoherent/outlier-free applies equally well to the *activation* -- rotating
the entire residual stream by a random orthogonal matrix removes activation
outliers too (the same concentration-of-measure argument, applied to
whichever vector is being conjugated), letting **both** operands of the
MatMul be quantized to INT4 with plain round-to-nearest and no calibration
data at all.

    Before:
      Y = MatMul(X, W) [+ bias]                 -- W constant, [K, N], float32

    After:
      U: initializer, float32 [K, K]            -- the random rotation
      Xrot = MatMul(X, U)                       -- runtime activation rotation
      Xq   = round_to_nearest_int4_per_token(Xrot)   -- data-free, no calibration
      Wtilde_hat = DequantizeLinear(Wtilde_q, Wtilde_s, axis=0, block_size=32)
                                                  -- INT4 codes, [K, N]
      Y = MatMul(Xq, Wtilde_hat) [+ bias]

Rotating by a random orthogonal matrix ``U`` is lossless on its own
(``U @ U.T == I``); only the two INT4 round-to-nearest steps after it lose
precision, exactly as much as their unrotated counterparts already do --
the point of the rotation is that it changes *which* values get quantized:
values spread evenly across directions, rather than concentrated in a few
outlier ones (weight *rows* for ``Wtilde``, or entries of a given token's
activation vector for ``Xrot``), quantize with far less error at the same
bit width. This module reuses :mod:`onnxsim.quip_sharp`'s own
``_random_orthogonal_matrix`` (a Haar-random matrix via QR-decomposing a
random Gaussian, not the real QuaRot's Hadamard-structured construction --
see that module's own docstring for why: the concentration argument only
needs a uniformly random rotation, not a Hadamard-structured one, and a
plain orthogonal matrix is simpler to construct correctly for any
dimension without power-of-2 padding, at the deployment cost of an
``O(n^2)`` dense MatMul instead of an ``O(n log n)`` Fast Walsh-Hadamard
Transform -- the same trade-off already made, and already explained, in
:mod:`onnxsim.quip_sharp`) and :mod:`onnxsim.omniquant`'s own
``_quantize_blockwise_int4_with_clip`` for the weight side, exactly as
:mod:`onnxsim.spinquant` does.

**Why the activation needs no calibration but the real QuaRot's weight
quantization can optionally use GPTQ.** :func:`apply_quarot` quantizes
*both* operands with plain round-to-nearest: the weight offline (from the
rotated weight's own values, like :mod:`onnxsim.spinquant`), the
activation online, per token, from that token's own rotated values (the
same data-free pattern :mod:`onnxsim.kv_cache_quantization`'s Value-style
KV-cache rewrite already uses -- ``scale = max(|x|) / 7`` computed at
graph-run time, no stored calibration statistics). The real QuaRot paper
additionally offers a GPTQ-based weight quantizer for a tighter error
bound; :func:`apply_quarot_gptq` (below) provides it: it is identical to
:func:`apply_quarot` in every respect (same candidate matching, same
per-layer rotation ``U``, same data-free per-token activation
quantization) except the *weight* is quantized via :mod:`onnxsim.gptq`'s
own Hessian-based column algorithm (evaluated in the *rotated* activation
space -- see :func:`apply_quarot_gptq`'s own docstring for exactly how the
two compose) instead of independent round-to-nearest, needing real
calibration activations only for that one step.

**Scope note.** The real QuaRot rotates the *entire residual stream* by a
single, shared rotation (fused into every adjacent weight matrix -- Q/K/V/O
projections, MLP up/down, embeddings -- so the rotation is "free" at
inference, and a second, smaller online rotation handles the attention
head dimension specifically). Fusing one global rotation across an entire
decoder stack needs a model-level graph walk (matching residual-stream
add/embedding boundaries) this module does not attempt; instead, matching
:mod:`onnxsim.spinquant`'s own scope, this module fits one independent
rotation **per matched layer**, at the cost of an explicit ``MatMul(X, U)``
per layer rather than a fused, "free" rotation -- the same graph-simplicity
trade-off :mod:`onnxsim.quip_sharp` already makes for its own two
rotations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _pack_int4
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.gptq import _gptq_quantize_columns
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip
from onnxsim.quip_sharp import _match_matmul_like, _random_orthogonal_matrix


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def apply_quarot(
    model: Union[str, onnx.ModelProto],
    seed: int = 0,
    block_size: int = 32,
    epsilon: float = 1e-12,
) -> onnx.ModelProto:
    """Applies QuaRot-style random-rotation preprocessing (see this
    module's own docstring) plus INT4 round-to-nearest quantization of
    *both* the weight and the activation to every MatMul/vanilla-Gemm
    layer with a constant 2-D float32 weight whose reduction dimension
    ``K`` is divisible by ``block_size``. Needs no calibration data: the
    rotation is random (not fit to any data, unlike
    :mod:`onnxsim.spinquant`), the weight's scale comes from the rotated
    weight's own values, and the activation's scale is computed fresh
    per token at graph-run time.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param seed: seed for the random rotation matrices (a fresh
            ``numpy.random.Generator`` is derived per matched layer, in
            graph node order, so results are deterministic and
            reproducible for a given model and seed)
    :param block_size: elements per weight quantization block along
            ``K``, matching :func:`onnxsim.quantize_weight_only_int4`'s
            own default
    :param epsilon: floor applied to a token's own max-abs activation
            value before using it as a scale, avoiding a divide-by-zero
            on an all-zero token
    :returns: ``model`` with every matched layer's weight and activation
            replaced by rotated, INT4-quantized versions (plus the
            original bias, if any); output tensor name unchanged. Layers
            with a non-constant, non-2-D weight, or a reduction dimension
            not divisible by ``block_size``, are left untouched; a model
            with no matching layer, or an opset older than 21 (INT4's
            tensor type and ``DequantizeLinear``'s ``block_size``
            attribute both need opset 21), is returned unchanged

    **Accepted, permanent divergence from the C++ port
    (``apply_quarot_cpp``).** The same ``seed`` does NOT produce the same
    rotation matrix, or therefore the same quantized output, on both sides
    -- and this is intentional, not an open gap. Two independent reasons,
    neither a correctness bug:

    1. Different orthogonalization algorithm. This function (via
       :func:`onnxsim.quip_sharp._random_orthogonal_matrix`) QR-decomposes
       a random Gaussian matrix with ``numpy.linalg.qr`` and then corrects
       ``Q``'s sign using ``R``'s own diagonal, because LAPACK's Householder
       QR picks each reflector's sign for numerical stability rather than
       at random, which biases ``Q`` without the fix. The C++ port
       (``passes/random_orthogonal.h``) instead uses Gram-Schmidt, whose
       natural diagonal (each row's Euclidean norm) is already nonnegative
       by construction -- so it needs no analogous correction and is
       already Haar-uniform on its own. A Monte Carlo check (K=5, 300k
       draws) found the two constructions' first-entry distributions
       statistically indistinguishable (max CDF gap ~0.0016, versus ~0.5
       against an *uncorrected* QR, which is visibly biased) -- i.e. both
       are valid, independent constructions of the same target
       distribution, not "the same algorithm, one right and one wrong."
    2. Different RNG derivation. This function sequences a single
       ``numpy.random.Generator`` across every matched layer in graph node
       order; the C++ port reseeds a fresh ``std::mt19937_64`` per matched
       node from a hash of the seed and that node's own unique id.

    Reconciling either difference for true bit-for-bit parity would mean
    reimplementing numpy's PCG64 bit generator and its ziggurat-based
    ``standard_normal`` sampler in C++ (not just swapping in a QR routine)
    -- disproportionate for a rotation whose only actual requirement, per
    the QuaRot/QuIP# papers themselves, is being *some* uniformly random
    orthogonal matrix, not a bit-specific one. ``apply_quarot`` and
    ``apply_quarot_cpp`` remain two independently-correct,
    non-interchangeable entry points; see ``passes/random_orthogonal.h``
    and ``passes/quarot.h`` for the full investigation this note
    summarizes.
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

    rng = np.random.default_rng(seed)

    for node, x_name, w_name, bias_name, weight_transposed in candidates:
        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape
        if k % block_size != 0:
            continue

        u = _random_orthogonal_matrix(k, rng)  # [K, K]
        w_tilde_nk = w_nk @ u  # [N, K] -- exact before quantization

        codes_nk, scale_blocks_nk = _quantize_blockwise_int4_with_clip(
            w_tilde_nk, block_size, 1.0
        )
        codes_kn = codes_nk.T.astype(np.int64)  # [K, N], ready for a plain MatMul
        scale_kn = scale_blocks_nk.T.astype(np.float32)  # [K/block_size, N]

        prefix = f"{w_name}_quarot"
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
        u_name = _unique_name(f"{prefix}_u", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(u.astype(np.float32), name=u_name)
        )
        eps_name = _unique_name(f"{prefix}_eps", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(epsilon, dtype=np.float32), name=eps_name
            )
        )
        seven_name = _unique_name(f"{prefix}_seven", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(7.0, dtype=np.float32), name=seven_name
            )
        )
        clip_min_name = _unique_name(f"{prefix}_clip_min", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(-7.0, dtype=np.float32), name=clip_min_name
            )
        )
        clip_max_name = _unique_name(f"{prefix}_clip_max", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(7.0, dtype=np.float32), name=clip_max_name
            )
        )
        axes_name = _unique_name(f"{prefix}_reduce_axes", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64), name=axes_name)
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

        x_rotated = _new("MatMul", [x_name, u_name], "x_rotated")

        # Data-free, per-token round-to-nearest INT4 activation
        # quantization -- simulated via an immediate dequantize (kept in
        # float32) rather than a true packed INT4 tensor, since X isn't
        # constant: scale = max(reduce_max(abs(x_rotated), axis=-1), eps) / 7
        abs_name = _new("Abs", [x_rotated], "x_abs")
        max_name = _new("ReduceMax", [abs_name, axes_name], "x_max", keepdims=1)
        safe_max_name = _new("Clip", [max_name, eps_name], "x_safe_max")
        x_scale = _new("Div", [safe_max_name, seven_name], "x_scale")
        x_scaled = _new("Div", [x_rotated, x_scale], "x_scaled")
        x_rounded = _new("Round", [x_scaled], "x_rounded")
        x_clipped = _new("Clip", [x_rounded, clip_min_name, clip_max_name], "x_clipped")
        x_dequant = _new("Mul", [x_clipped, x_scale], "x_dequant")

        w_dequant = _new(
            "DequantizeLinear",
            [codes_name, scale_name],
            "w_dequant",
            axis=0,
            block_size=block_size,
        )
        core = _new("MatMul", [x_dequant, w_dequant], "core")

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


def apply_quarot_gptq(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    epsilon: float = 1e-12,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Variant of :func:`apply_quarot` that quantizes the *rotated* weight
    via :mod:`onnxsim.gptq`'s Hessian-based column algorithm (see that
    module's own docstring for the algorithm itself) instead of plain
    round-to-nearest -- the real QuaRot paper's own optional, tighter
    weight quantizer that :func:`apply_quarot`'s own docstring flags as not
    reproduced there.

    Everything about the rotation and the activation is unchanged from
    :func:`apply_quarot`: the same candidate matching, the same per-layer
    random rotation ``U`` (this function reuses ``apply_quarot``'s own
    rotation-derivation loop verbatim -- one ``numpy.random.Generator(seed)``
    sequenced across every matched layer in graph node order -- so for a
    given ``seed`` the two functions derive byte-identical rotations per
    layer), and the same data-free, per-token round-to-nearest INT4
    activation quantization at graph-run time. Only the weight-quantization
    step differs, in three parts:

    1. capture each candidate layer's real (pre-rotation) activation ``X``
       from ``model`` using calibration data, exactly the pattern
       :func:`onnxsim.gptq.apply_gptq` already uses
       (:func:`onnxsim.bias_correction._add_probe_outputs` +
       :func:`onnxsim.backend.run_model`, including its 2-D-only filtering
       of what comes back);
    2. rotate that activation by the same layer's own ``U``
       (``X_rotated = X @ U``, mirroring the weight side's own
       ``Wtilde = W @ U``) and compute GPTQ's Hessian in the *rotated*
       space, ``H = X_rotated.T @ X_rotated`` -- the space
       :func:`onnxsim.gptq._gptq_quantize_columns` actually minimizes
       reconstruction error in, since it is ``Wtilde`` (not ``W``) being
       quantized here;
    3. quantize ``Wtilde`` via :func:`onnxsim.gptq._gptq_quantize_columns`
       using that Hessian and the *same* per-``(output channel, block)``
       scale :func:`onnxsim.omniquant._quantize_blockwise_int4_with_clip`
       already computes for it in :func:`apply_quarot` (that call's own
       round-to-nearest codes are discarded -- only its scale is reused,
       since GPTQ's column algorithm takes an already-computed scale and
       only changes which integer each element rounds to, never the scale
       itself).

    **Fallback for a layer with no usable calibration data.** A candidate
    layer whose captured activation is empty (no calibration batch reached
    it) or isn't a plain 2-D array, or whose feature dimension doesn't
    match the weight's own ``K``, is left completely alone by this
    function -- no rotation, no quantization -- rather than silently
    falling back to plain round-to-nearest quantization under GPTQ's own
    name (mirrors :func:`onnxsim.gptq.apply_gptq`'s own
    ``if not acts: continue`` skip). Note this differs from
    :func:`apply_quarot`, which never skips a layer for lack of data since
    it needs none.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            matched layer's rotated-space Hessian from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            much more representative Hessian than random input).
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for both the random rotation matrices (identical
            derivation to :func:`apply_quarot`'s own ``seed``) and the
            random calibration data (ignored for the latter if
            ``calibration_data`` is supplied)
    :param block_size: elements per weight quantization block along ``K``,
            matching :func:`apply_quarot`'s own default
    :param percdamp: Hessian damping factor, matching
            :func:`onnxsim.gptq.apply_gptq`'s own parameter and default
    :param proc_block_size: GPTQ's own column-processing block size (not
            the quantization scale's own block size) -- see
            :func:`onnxsim.gptq._gptq_quantize_columns`
    :param epsilon: floor applied to a token's own max-abs activation value
            before using it as a scale at graph-run time, matching
            :func:`apply_quarot`'s own parameter
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer that has usable
            calibration data rotated and INT4-quantized (GPTQ for the
            weight, data-free round-to-nearest for the activation, exactly
            like :func:`apply_quarot`); a matched layer without usable
            calibration data is left completely untouched. A model with no
            matching layer, or an opset older than 21, is returned
            unchanged (matching :func:`apply_quarot`).
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not _has_min_opset(model, 21):
        return model
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

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

    rng = np.random.default_rng(seed)

    # Pass 1: derive each eligible layer's rotation via the exact same
    # loop apply_quarot uses (matching node order, matching k % block_size
    # skip before drawing from rng) so the "same seed" yields the same U
    # per layer in both functions.
    rotated = []
    for node, x_name, w_name, bias_name, weight_transposed in candidates:
        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape
        if k % block_size != 0:
            continue

        u = _random_orthogonal_matrix(k, rng)  # [K, K]
        rotated.append((node, x_name, w_name, bias_name, w_nk, n, k, u))

    if not rotated:
        return out

    # Pass 2: capture every rotation-eligible layer's own (pre-rotation)
    # activation from the *original* model -- same probe pattern as
    # onnxsim.gptq.apply_gptq.
    probe_names = [r[1] for r in rotated]  # x_name
    probe_model = _add_probe_outputs(model, probe_names)
    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            activations[name].append(np.asarray(result[name], dtype=np.float64))

    for node, x_name, w_name, bias_name, w_nk, n, k, u in rotated:
        acts = [a for a in activations[x_name] if a.ndim == 2]
        if not acts:
            continue  # no usable calibration activation -- leave untouched
        x = np.concatenate(acts, axis=0)
        if x.shape[1] != k:
            continue  # activation's feature dim doesn't match K -- leave untouched

        w_tilde_nk = w_nk @ u  # [N, K] -- exact before quantization
        x_rotated = x @ u  # [S, K] -- same rotation, rotated-space Hessian
        h = x_rotated.T @ x_rotated  # [K, K]

        _, scale_blocks_nk = _quantize_blockwise_int4_with_clip(
            w_tilde_nk, block_size, 1.0
        )
        codes_nk = _gptq_quantize_columns(
            w_tilde_nk, scale_blocks_nk, block_size, h, percdamp, proc_block_size
        )
        codes_kn = codes_nk.T.astype(np.int64)  # [K, N], ready for a plain MatMul
        scale_kn = scale_blocks_nk.T.astype(np.float32)  # [K/block_size, N]

        prefix = f"{w_name}_quarot_gptq"
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
        u_name = _unique_name(f"{prefix}_u", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(u.astype(np.float32), name=u_name)
        )
        eps_name = _unique_name(f"{prefix}_eps", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(epsilon, dtype=np.float32), name=eps_name
            )
        )
        seven_name = _unique_name(f"{prefix}_seven", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(7.0, dtype=np.float32), name=seven_name
            )
        )
        clip_min_name = _unique_name(f"{prefix}_clip_min", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(-7.0, dtype=np.float32), name=clip_min_name
            )
        )
        clip_max_name = _unique_name(f"{prefix}_clip_max", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array(7.0, dtype=np.float32), name=clip_max_name
            )
        )
        axes_name = _unique_name(f"{prefix}_reduce_axes", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64), name=axes_name)
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

        x_rotated_name = _new("MatMul", [x_name, u_name], "x_rotated")

        # Data-free, per-token round-to-nearest INT4 activation
        # quantization -- simulated via an immediate dequantize (kept in
        # float32) rather than a true packed INT4 tensor, since X isn't
        # constant: scale = max(reduce_max(abs(x_rotated), axis=-1), eps) / 7
        abs_name = _new("Abs", [x_rotated_name], "x_abs")
        max_name = _new("ReduceMax", [abs_name, axes_name], "x_max", keepdims=1)
        safe_max_name = _new("Clip", [max_name, eps_name], "x_safe_max")
        x_scale = _new("Div", [safe_max_name, seven_name], "x_scale")
        x_scaled = _new("Div", [x_rotated_name, x_scale], "x_scaled")
        x_rounded = _new("Round", [x_scaled], "x_rounded")
        x_clipped = _new("Clip", [x_rounded, clip_min_name, clip_max_name], "x_clipped")
        x_dequant = _new("Mul", [x_clipped, x_scale], "x_dequant")

        w_dequant = _new(
            "DequantizeLinear",
            [codes_name, scale_name],
            "w_dequant",
            axis=0,
            block_size=block_size,
        )
        core = _new("MatMul", [x_dequant, w_dequant], "core")

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
