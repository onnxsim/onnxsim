"""AQLM (Egiazarian et al., 2024, "Extreme Compression of Large Language
Models via Additive Quantization", https://arxiv.org/abs/2401.06118).
onnxsim ports the algorithm, not any framework's code, per the same
rationale as :mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.hqq`
(AQLM's own reference implementation quantizes live PyTorch weights via a
custom calibration-aware beam-search optimizer, with no ONNX export path).

Every other codebook-based scheme in onnxsim (:mod:`onnxsim.nf4`'s one
*fixed* global codebook, :mod:`onnxsim.squeezellm`'s one *fit-per-group*
codebook) represents each weight element (NF4) or each small group of
elements (SqueezeLLM) as a single lookup into a single codebook. AQLM's
own idea, **additive** (a.k.a. residual) **quantization**: represent each
group instead as the *sum* of ``M`` lookups, one from each of ``M``
separate codebooks shared across every group in the layer:

    ŵ_group = C_1[i_1] + C_2[i_2] + ... + C_M[i_M]

With ``M`` codebooks of ``codebook_size`` entries each, this can represent
up to ``codebook_size ** M`` distinct group values while storing only
``M`` small codebooks plus ``M`` per-group indices -- far more
representational richness per stored bit than a single codebook of the
same total size, the same reason residual/product quantization is a
classical, well-established technique in the vector quantization
literature AQLM's own paper builds on and cites.

Fitting the ``M`` codebooks: this module uses the classical **greedy
residual k-means** strategy -- fit codebook 1 with ordinary k-means
(Lloyd's algorithm) to every group's own raw values (each of the many
groups in a layer treated as one point in ``group_dim``-dimensional
space, all sharing the *same* fitted codebook, unlike
:mod:`onnxsim.squeezellm`'s independent per-group codebooks), subtract
what it reconstructs, fit codebook 2 to *that residual*, and so on --
rather than AQLM's own more sophisticated joint beam-search code
assignment calibrated against a Hessian-weighted objective. Greedy
residual fitting is the textbook baseline additive/residual quantization
is built on, independently verifiable (see this module's own tests: more
codebooks can only reduce reconstruction error, since each new codebook
targets exactly the error the previous stages left over), and needs no
calibration data at all -- consistent with :mod:`onnxsim.hqq`/
:mod:`onnxsim.nf4`/:mod:`onnxsim.quip_sharp`'s own choice to solve the
same representational problem via a classical technique rather than risk
an unverifiable reproduction of a paper's own bespoke calibrated
optimizer.

Dequantization is ``M`` ordinary ``Gather`` operations (one per codebook,
exactly :mod:`onnxsim.nf4`'s own pattern, since every group shares the
same global codebook per stage -- no per-group indexing like
:mod:`onnxsim.squeezellm`'s ``GatherND`` is needed) followed by ``M - 1``
``Add`` nodes summing the stages together. No custom op or contrib
domain, and no opset requirement beyond ordinary ``Gather``/``Add``.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim.bias_correction import _all_names, _unique_name


def _match_matmul_like(node: onnx.NodeProto):
    """Mirrors ``MatchMatMulLike`` (``passes/quantize_matmul_common.h``):
    a MatMul, or a Gemm with ``transA=0``, ``alpha=1`` and (when it has a
    bias) ``beta=1``. Returns ``(x_name, w_name, weight_transposed)`` or
    ``None``.
    """
    attrs = {a.name: a for a in node.attribute}
    if node.op_type == "MatMul":
        if len(node.input) != 2:
            return None
        return node.input[0], node.input[1], False
    if node.op_type == "Gemm":
        num_inputs = len(node.input)
        if num_inputs not in (2, 3):
            return None
        trans_a = attrs.get("transA")
        if trans_a is not None and trans_a.i != 0:
            return None
        alpha = attrs.get("alpha")
        if alpha is not None and alpha.f != 1.0:
            return None
        if num_inputs == 3:
            beta = attrs.get("beta")
            if beta is not None and beta.f != 1.0:
                return None
        trans_b = attrs.get("transB")
        weight_transposed = bool(trans_b is not None and trans_b.i)
        return node.input[0], node.input[1], weight_transposed
    return None


def _fit_kmeans_codebook(
    data: np.ndarray, codebook_size: int, num_iterations: int, rng: np.random.Generator
) -> "tuple[np.ndarray, np.ndarray]":
    """Ordinary (unweighted) Lloyd's-algorithm k-means: fits
    ``codebook_size`` centroids (``[codebook_size, dim]``) shared across
    every row of ``data`` (``[num_points, dim]``), returning
    ``(centroids, assignment)`` with ``assignment`` (``[num_points]``) the
    index of each row's nearest centroid. Centroids are initialized from a
    random sample of ``data``'s own rows (padding by repeating the last
    sampled point if ``data`` has fewer rows than ``codebook_size``); an
    empty cluster keeps its previous centroid rather than going undefined.
    """
    num_points = data.shape[0]
    k = min(codebook_size, num_points)
    init_idx = rng.choice(num_points, size=k, replace=False)
    centroids = data[init_idx].copy()
    if k < codebook_size:
        pad = np.tile(centroids[-1:], (codebook_size - k, 1))
        centroids = np.vstack([centroids, pad])

    assignment = np.zeros(num_points, dtype=np.int64)
    for _ in range(num_iterations):
        dist = np.sum(
            (data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2
        )
        assignment = np.argmin(dist, axis=1)
        new_centroids = centroids.copy()
        for c in range(codebook_size):
            mask = assignment == c
            if np.any(mask):
                new_centroids[c] = data[mask].mean(axis=0)
        centroids = new_centroids

    dist = np.sum((data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    assignment = np.argmin(dist, axis=1)
    return centroids, assignment


def quantize_weight_only_aqlm(
    model: Union[str, onnx.ModelProto],
    group_dim: int = 8,
    num_codebooks: int = 2,
    codebook_size: int = 256,
    num_iterations: int = 10,
    seed: int = 0,
) -> onnx.ModelProto:
    """Quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight (whose reduction dimension ``K`` is evenly divisible by
    ``group_dim``) into AQLM-style additive (multi-codebook) quantization
    -- see this module's own docstring for the technique. Needs no
    calibration data: every codebook is fit directly to the weight's own
    values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param group_dim: elements per group (each ``group_dim``-element chunk
            of a row gets its own set of ``num_codebooks`` indices, one
            per codebook)
    :param num_codebooks: number of additive codebook stages ``M`` (the
            paper's own typical range is 1-2 for its most extreme
            compression settings, with more codebooks trading additional
            stored bits for lower reconstruction error)
    :param codebook_size: entries per codebook (2^8 = 256 is a typical
            choice, matching one byte per stored index)
    :param num_iterations: Lloyd's-algorithm iterations refining each
            stage's codebook
    :param seed: seed for the k-means centroid initialization (a fresh
            ``numpy.random.Generator`` is derived per matched layer, in
            graph node order, so results are deterministic and
            reproducible for a given model and seed)
    :returns: ``model`` with every matched layer's weight replaced by
            ``M`` ``Gather`` lookups (one per codebook) summed via
            ``Add`` and reshaped back to the weight's own shape, feeding
            the original MatMul/Gemm node; layers with a non-constant,
            non-2-D, or non-group-divisible weight are left untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

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
        x_name, w_name, weight_transposed = match
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or w_init.data_type != onnx.TensorProto.FLOAT
            or len(w_init.dims) != 2
        ):
            continue
        candidates.append((node, w_name, weight_transposed))

    if not candidates:
        return out

    rng = np.random.default_rng(seed)

    for node, w_name, weight_transposed in candidates:
        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        dim0, dim1 = w.shape
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape
        if k % group_dim != 0:
            continue

        num_groups = n * (k // group_dim)
        groups = w_nk.reshape(num_groups, group_dim)
        residual = groups.copy()

        codebooks: List[np.ndarray] = []
        codes: List[np.ndarray] = []
        for _ in range(num_codebooks):
            centroids, assignment = _fit_kmeans_codebook(
                residual, codebook_size, num_iterations, rng
            )
            codebooks.append(centroids)
            codes.append(assignment)
            residual = residual - centroids[assignment]

        prefix = f"{w_name}_aqlm"
        stage_outputs = []
        new_nodes: List[onnx.NodeProto] = []
        for m in range(num_codebooks):
            codebook_name = _unique_name(f"{prefix}_codebook{m}", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(
                    codebooks[m].astype(np.float32), name=codebook_name
                )
            )
            codes_name = _unique_name(f"{prefix}_codes{m}", taken_names)
            graph.initializer.append(
                onnx.numpy_helper.from_array(codes[m].astype(np.int64), name=codes_name)
            )
            stage_out = _unique_name(f"{prefix}_stage{m}", taken_names)
            gather_node = onnx.helper.make_node(
                "Gather",
                [codebook_name, codes_name],
                [stage_out],
                name=_unique_name(f"{prefix}_gather{m}_node", taken_names),
                axis=0,
            )
            new_nodes.append(gather_node)
            stage_outputs.append(stage_out)

        combined = stage_outputs[0]
        for m in range(1, num_codebooks):
            add_out = _unique_name(f"{prefix}_sum{m}", taken_names)
            add_node = onnx.helper.make_node(
                "Add",
                [combined, stage_outputs[m]],
                [add_out],
                name=_unique_name(f"{prefix}_add{m}_node", taken_names),
            )
            new_nodes.append(add_node)
            combined = add_out

        shape_name = _unique_name(f"{prefix}_shape", taken_names)
        graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.array([n, k], dtype=np.int64), name=shape_name
            )
        )
        unblocked_name = _unique_name(f"{prefix}_unblocked", taken_names)
        reshape_node = onnx.helper.make_node(
            "Reshape",
            [combined, shape_name],
            [unblocked_name],
            name=_unique_name(f"{prefix}_reshape_node", taken_names),
        )
        new_nodes.append(reshape_node)

        final_name = unblocked_name
        if not weight_transposed:
            final_name = _unique_name(f"{prefix}_transposed", taken_names)
            transpose_node = onnx.helper.make_node(
                "Transpose",
                [unblocked_name],
                [final_name],
                name=_unique_name(f"{prefix}_transpose_node", taken_names),
                perm=[1, 0],
            )
            new_nodes.append(transpose_node)

        node_idx = next(i for i, n in enumerate(graph.node) if n is node)
        for offset, new_node in enumerate(new_nodes):
            graph.node.insert(node_idx + offset, new_node)
        for i, inp in enumerate(node.input):
            if inp == w_name:
                node.input[i] = final_name

    return out
