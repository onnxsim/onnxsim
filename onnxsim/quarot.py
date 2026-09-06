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

:func:`apply_quarot_fused` recovers the "free at inference" half of that
gap without the model-level residual-stream walk: it keeps the per-layer
rotation, but wherever a layer's activation comes straight out of another
constant-weight MatMul/Gemm that feeds nothing else, it folds that layer's
``U`` into the *producer's* weight offline (``W_P' = W_P @ U``,
``b_P' = b_P @ U``) so the producer already emits the rotated activation
and no runtime rotation node is emitted at all. That fold is exact
algebra, so the accuracy is :func:`apply_quarot`'s; only the runtime cost
goes away. It is still a per-layer rotation, and layers fed by anything
else (a model input, a LayerNorm, an activation, a residual ``Add``) still
pay for their explicit ``MatMul(X, U)`` -- see that function's own
docstring for the exact fusability conditions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

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

# (node, activation name, weight name, bias name or None, weight_transposed)
_Candidate = Tuple[onnx.NodeProto, str, str, Optional[str], bool]


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def _emit_quarot_layer(
    graph: onnx.GraphProto,
    taken_names: Set[str],
    node: onnx.NodeProto,
    prefix: str,
    w_tilde_nk: np.ndarray,
    u: Optional[np.ndarray],
    bias_name: Optional[str],
    block_size: int,
    epsilon: float,
) -> None:
    """Replaces ``node`` in ``graph`` with QuaRot's quantized layer: an
    INT4 ``DequantizeLinear`` of the already-rotated weight ``w_tilde_nk``
    ([N, K], output channel first), a data-free per-token INT4
    round-trip of the activation, the core MatMul, and the original bias
    (if any).

    ``u`` is the layer's input-side rotation ([K, K]). When it is not
    ``None`` the activation is rotated at graph-run time by an explicit
    ``MatMul(X, U)`` (what :func:`apply_quarot` does). When it is ``None``
    the caller has already folded that rotation into the *producer's*
    weight, so the incoming tensor is rotated already and neither the
    ``MatMul`` node nor the ``U`` initializer is emitted at all (what
    :func:`apply_quarot_fused` does).
    """
    n, k = w_tilde_nk.shape

    codes_nk, scale_blocks_nk = _quantize_blockwise_int4_with_clip(
        w_tilde_nk, block_size, 1.0
    )
    codes_kn = codes_nk.T.astype(np.int64)  # [K, N], ready for a plain MatMul
    scale_kn = scale_blocks_nk.T.astype(np.float32)  # [K/block_size, N]

    codes_name = _unique_name(f"{prefix}_codes", taken_names)
    codes_tensor = onnx.TensorProto()
    codes_tensor.name = codes_name
    codes_tensor.data_type = onnx.TensorProto.INT4
    codes_tensor.dims.extend([k, n])
    codes_tensor.raw_data = _pack_int4(codes_kn)
    graph.initializer.append(codes_tensor)

    scale_name = _unique_name(f"{prefix}_scale", taken_names)
    graph.initializer.append(onnx.numpy_helper.from_array(scale_kn, name=scale_name))
    u_name: Optional[str] = None
    if u is not None:
        u_name = _unique_name(f"{prefix}_u", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(u.astype(np.float32), name=u_name)
        )
    eps_name = _unique_name(f"{prefix}_eps", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(epsilon, dtype=np.float32), name=eps_name)
    )
    seven_name = _unique_name(f"{prefix}_seven", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.array(7.0, dtype=np.float32), name=seven_name)
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

    if u_name is None:
        # Fused: the producer's own weight already emits the rotated
        # activation, so there is nothing to rotate at run time.
        x_rotated = node.input[0]
    else:
        x_rotated = _new("MatMul", [node.input[0], u_name], "x_rotated")

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
        k = w_nk.shape[1]
        if k % block_size != 0:
            continue

        u = _random_orthogonal_matrix(k, rng)  # [K, K]
        w_tilde_nk = w_nk @ u  # [N, K] -- exact before quantization

        _emit_quarot_layer(
            graph,
            taken_names,
            node,
            f"{w_name}_quarot",
            w_tilde_nk,
            u,
            bias_name,
            block_size,
            epsilon,
        )

    return out


def _count_tensor_uses(graph: onnx.GraphProto) -> Dict[str, int]:
    """Counts how many *node inputs* across the whole model reference each
    tensor name. Subgraphs (``If``/``Loop``/``Scan`` bodies) are walked
    too, since an inner node may capture an outer-scope tensor by name --
    a tensor referenced from a subgraph must not be treated as
    single-consumer.
    """
    counts: Dict[str, int] = {}
    stack = [graph]
    while stack:
        g = stack.pop()
        for node in g.node:
            for name in node.input:
                if name:
                    counts[name] = counts.get(name, 0) + 1
            for attr in node.attribute:
                if attr.HasField("g"):
                    stack.append(attr.g)
                stack.extend(attr.graphs)
    return counts


def _fold_bias(b: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``b @ u`` on the bias's last (output-channel) axis, preserving its
    original ``[N]`` or ``[1, N]`` shape.
    """
    return (b.reshape(-1) @ u).reshape(b.shape)


def apply_quarot_fused(
    model: Union[str, onnx.ModelProto],
    seed: int = 0,
    block_size: int = 32,
    epsilon: float = 1e-12,
) -> onnx.ModelProto:
    """QuaRot with the activation rotation **fused into the producing
    layer's weight** wherever the graph allows it, so that rotation costs
    nothing at inference -- the "free at inference" property of the real
    QuaRot that :func:`apply_quarot` gives up (see this module's own
    docstring's scope note).

    Same matching, same rotations, same INT4 quantization as
    :func:`apply_quarot`; the only difference is *where* the rotation
    happens. If layer ``L``'s activation ``T`` is produced by another
    MatMul/vanilla-Gemm layer ``P`` with a constant weight, and ``T`` goes
    nowhere else, then instead of emitting ``MatMul(T, U)`` at run time
    the rotation is folded into ``P``'s own weight offline::

        T   = X_prev @ W_P  (+ b_P)
        W_P' = W_P @ U,  b_P' = b_P @ U
        =>  X_prev @ W_P' (+ b_P')  ==  (X_prev @ W_P + b_P) @ U  ==  T @ U

    so ``P`` directly emits the rotated activation ``L`` wants. This is
    exact algebra -- the fold itself loses nothing, and ``L``'s own weight
    is rotated exactly as :func:`apply_quarot` rotates it
    (``w_tilde_nk = w_nk @ u``), so ``(T @ U) @ (U.T @ W_L) == T @ W_L``
    still holds. The result is the same accuracy as
    :func:`apply_quarot` with strictly fewer runtime nodes (one
    ``MatMul`` and one ``[K, K]`` initializer fewer per fused layer).

    Note that the two rotations act on *different axes* of a weight: a
    layer's own input-side rotation acts on its reduction dim ``K``
    (right-multiply, ``w_nk @ u``), while an output-side fold acts on its
    output dim ``N`` (left-multiply, ``u_consumer.T @ w_nk``). A layer
    that is both a fused producer and a quantized consumer -- the middle
    layer of a chain ``A -> B -> C`` -- therefore gets **both**, and both
    are applied while the weight is still float, before it is quantized.
    Chains of any length are handled: this function plans every rotation
    first, then rotates every weight, and only then quantizes.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param seed: seed for the random rotation matrices. Sequenced exactly
            as :func:`apply_quarot` does it (a single
            ``numpy.random.Generator``, advanced once per quantized layer
            in graph node order), so for a given model and seed both
            functions fit the *same* rotation to the same layer
    :param block_size: elements per weight quantization block along ``K``
    :param epsilon: floor applied to a token's own max-abs activation
            value before using it as a scale
    :returns: ``model`` with every matched layer quantized as
            :func:`apply_quarot` does, except that a layer whose
            activation edge is fusable carries no runtime rotation node
            and no ``U`` initializer at all

    **Which edges are fusable.** ``L``'s activation edge is fused only
    when *all* of these hold; otherwise ``L`` falls back to
    :func:`apply_quarot`'s explicit ``MatMul(X, U)`` (never skipped, never
    silently mis-rotated):

    - the activation tensor is produced by a node in the main graph that
      :func:`onnxsim.quip_sharp._match_matmul_like` matches (a MatMul, or
      a Gemm with ``transA=0``/``alpha=1``/``beta=1``) whose weight is a
      constant 2-D float32 initializer;
    - that tensor is referenced by exactly one node input in the whole
      model (subgraph bodies included) -- a tensor consumed twice cannot
      be rotated for one consumer without corrupting the other;
    - that tensor is not a graph output (a graph output must keep its
      unrotated value);
    - the producer's own output dimension equals ``L``'s reduction
      dimension ``K``;
    - if the producer has a bias, that bias is a constant float32
      initializer shaped ``[N]`` or ``[1, N]`` -- the only shapes whose
      last axis is unambiguously the output channel axis, and so the only
      ones ``b @ U`` is defined for. A broadcast/scalar Gemm bias is not
      folded, and its layer is not fused.

    **Scope, stated plainly.** This is still a *per-layer* rotation, not
    the real QuaRot's single residual-stream-wide one: each matched layer
    still gets its own independent ``U``. What this function removes is
    only the *runtime cost* of that rotation, on the subset of edges where
    the algebra above applies. Layers fed by anything other than another
    constant-weight MatMul/Gemm -- a model input, a normalization, an
    activation function, a residual ``Add``, an attention softmax -- are
    not fusable and keep their explicit ``MatMul(X, U)``. In a real
    transformer that means the projections that immediately follow a
    LayerNorm or a residual add (i.e. most of them) still pay for their
    rotation; fusing those needs the model-level residual-stream walk this
    module still does not attempt.
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

    # --- Phase 1: plan ------------------------------------------------
    # Matching is identical to apply_quarot's, and so is the RNG
    # sequencing, so a given (model, seed) fits the very same rotation to
    # the very same layer in both functions.
    candidates: List[_Candidate] = []
    for node in list(graph.node):
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

    # Every constant-weight MatMul/Gemm is a potential fold target, even
    # one that is itself skipped for quantization (k % block_size != 0):
    # folding only needs its weight to be constant, not quantizable.
    producer_of: Dict[str, _Candidate] = {}
    for cand in candidates:
        producer_of[cand[0].output[0]] = cand

    rng = np.random.default_rng(seed)
    # Candidate indices that will actually be quantized, in graph node
    # order, each with its own input-side rotation U.
    quantized_order: List[int] = []
    own_rotation: Dict[int, np.ndarray] = {}
    for idx, (_node, _x_name, w_name, _bias_name, weight_transposed) in enumerate(
        candidates
    ):
        w_init = initializer_map[w_name]
        k = w_init.dims[1] if weight_transposed else w_init.dims[0]
        if k % block_size != 0:
            continue
        quantized_order.append(idx)
        own_rotation[idx] = _random_orthogonal_matrix(k, rng)

    graph_output_names = {o.name for o in graph.output}
    use_counts = _count_tensor_uses(graph)

    # id(producer node) -> the consumer rotation to fold into its output
    # axis. At most one entry per producer, since a fusable edge is by
    # definition single-consumer.
    fold_into: Dict[int, np.ndarray] = {}
    fused_indices: Set[int] = set()

    for idx in quantized_order:
        x_name = candidates[idx][1]
        u = own_rotation[idx]
        producer = producer_of.get(x_name)
        if producer is None:
            continue
        if x_name in graph_output_names or use_counts.get(x_name, 0) != 1:
            continue
        p_node, _p_x, p_w_name, p_bias_name, p_transposed = producer
        p_w_init = initializer_map[p_w_name]
        p_n = p_w_init.dims[0] if p_transposed else p_w_init.dims[1]
        if p_n != u.shape[0]:
            continue
        if p_bias_name is not None:
            p_b_init = initializer_map.get(p_bias_name)
            if p_b_init is None or p_b_init.data_type != onnx.TensorProto.FLOAT:
                continue
            if list(p_b_init.dims) not in ([p_n], [1, p_n]):
                continue
        fold_into[id(p_node)] = u
        fused_indices.add(idx)

    # --- Phase 2: rotate, all in float --------------------------------
    # Every weight that takes part -- as a quantized consumer, as a fused
    # producer, or (in an A -> B -> C chain) as both -- is rotated from
    # its original float values here, before anything is quantized.
    rotated_nk: Dict[int, np.ndarray] = {}
    rotated_bias: Dict[int, np.ndarray] = {}
    for idx, (node, _x_name, w_name, bias_name, weight_transposed) in enumerate(
        candidates
    ):
        fold_u = fold_into.get(id(node))
        own_u = own_rotation.get(idx)
        if fold_u is None and own_u is None:
            continue
        w = onnx.numpy_helper.to_array(initializer_map[w_name]).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        if fold_u is not None:
            # Output-side fold, on the N axis: the layer now emits T @ U.
            w_nk = fold_u.T @ w_nk
            if bias_name is not None:
                b = onnx.numpy_helper.to_array(initializer_map[bias_name]).astype(
                    np.float64
                )
                rotated_bias[idx] = _fold_bias(b, fold_u)
        if own_u is not None:
            # Input-side rotation, on the K axis: exactly apply_quarot's.
            w_nk = w_nk @ own_u
        rotated_nk[idx] = w_nk

    # --- Phase 3: emit -------------------------------------------------
    # A fused producer that is not itself quantized just gets its rotated
    # float weight (and bias) written back under fresh names -- the
    # original initializers may be shared with other nodes, so they are
    # never edited in place.
    quantized_indices = set(quantized_order)
    for idx, (node, _x_name, w_name, bias_name, weight_transposed) in enumerate(
        candidates
    ):
        if idx in quantized_indices or idx not in rotated_nk:
            continue
        w_nk = rotated_nk[idx]
        w_new = w_nk if weight_transposed else w_nk.T
        new_w_name = _unique_name(f"{w_name}_quarot_folded", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(w_new.astype(np.float32), name=new_w_name)
        )
        node.input[1] = new_w_name
        if idx in rotated_bias:
            assert bias_name is not None
            new_b_name = _unique_name(f"{bias_name}_quarot_folded", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    rotated_bias[idx].astype(np.float32), name=new_b_name
                )
            )
            node.input[2] = new_b_name

    for idx in quantized_order:
        node, _x_name, w_name, bias_name, _weight_transposed = candidates[idx]
        effective_bias = bias_name
        if idx in rotated_bias:
            assert bias_name is not None
            effective_bias = _unique_name(f"{bias_name}_quarot_folded", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    rotated_bias[idx].astype(np.float32), name=effective_bias
                )
            )
        _emit_quarot_layer(
            graph,
            taken_names,
            node,
            f"{w_name}_quarot",
            rotated_nk[idx],
            None if idx in fused_indices else own_rotation[idx],
            effective_bias,
            block_size,
            epsilon,
        )

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
