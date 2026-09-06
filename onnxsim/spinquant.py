"""SpinQuant (Liu et al., 2024, "SpinQuant: LLM Quantization with Learned
Rotations", https://arxiv.org/abs/2405.16406). onnxsim ports the algorithm,
not any framework's code, per the same rationale as
:mod:`onnxsim.awq`/:mod:`onnxsim.gptq`/:mod:`onnxsim.quip_sharp` (SpinQuant's
own reference implementation learns rotation matrices end-to-end against
live PyTorch weights via a Cayley-manifold optimizer, with no ONNX export
path).

SpinQuant's core idea, shared with QuIP#/QuIP (see
:mod:`onnxsim.quip_sharp`): conjugating a weight by an orthogonal rotation
before quantizing it can make the rotated weight far more
quantization-friendly than the original -- fewer/less extreme outlier
directions for a uniform grid to waste precision on. QuIP#'s own choice is
a *random* rotation (a concentration-of-measure argument: any fixed
vector, rotated by a uniformly random orthogonal matrix, spreads out
evenly with high probability, regardless of the original weight's own
structure). SpinQuant's own contribution is that the rotation doesn't have
to be random -- a rotation *fit to the data* can do noticeably better,
since it can specifically target the directions the real weight/activation
distribution actually concentrates its outliers in, rather than relying on
a probabilistic argument that ignores that structure entirely.

SpinQuant's own reference implementation fits (typically four, one per
attention/MLP sub-block) rotation matrices per layer via gradient descent
on the quantized model's own end-to-end loss, constrained to the
orthogonal group via a Cayley-manifold optimizer -- calibration-aware,
differentiable-quantization machinery that is not independently verifiable
the way a closed-form procedure is (the same reason
:mod:`onnxsim.quip_sharp` doesn't reproduce QuIP#'s own randomized-Hadamard
construction verbatim, and :mod:`onnxsim.low_rank_compensation` uses
truncated SVD rather than a learned low-rank factorization). This module
instead reproduces SpinQuant's own "R1-only" ablation -- the paper's own
simplified configuration, reported to capture most of the improvement over
no rotation at all -- via a classical, closed-form substitute: fit a
*single* input-side rotation per layer as the eigenvector basis of that
layer's own calibration-activation covariance matrix (an ordinary,
symmetric eigendecomposition), then block-wise RTN-quantize the rotated
weight to INT4 exactly like :func:`onnxsim.quantize_weight_only_int4`
does. Reconstruction is exact before quantization, since rotating by an
orthogonal matrix is lossless:

    Before:
      Y = MatMul(X, W) [+ bias]                 -- W constant, [K, N], float32

    After:
      U: initializer, float32 [K, K]            -- the fitted rotation
      X_rotated = MatMul(X, U)
      Wtilde_hat = DequantizeLinear(Wtilde_q, Wtilde_s, axis=0, block_size=32)
                                                  -- INT4 codes, [K, N]
      Y = MatMul(X_rotated, Wtilde_hat) [+ bias]

Why the eigenvector basis specifically, rather than SpinQuant's own learned
rotation: it is the classical, closed-form answer to "which rotation makes
this data's second-moment structure as close to isotropic as possible" --
the eigenvectors of the covariance are exactly the directions ordinary PCA
identifies as carrying disproportionate variance, and rotating into that
basis spreads out what was concentrated along a few of them, the same
effect a learned rotation is chasing, achieved with ordinary linear algebra
(``numpy.linalg.eigh``) instead of an unverifiable optimization loop.
Unlike QuIP#'s random rotation, this needs calibration data (the whole
point is to target the real distribution's own structure) but, also unlike
QuIP#, needs no second, output-side rotation or non-uniform lattice
codebook: SpinQuant's contribution is specifically about the rotation
being *learned* (here: fit in closed form) rather than random, not about a
different quantization backend, so this module pairs it with the same
block-wise RTN backend :mod:`onnxsim.quantize_weight_only_int4` already
uses.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import _pack_int4
from onnxsim.bias_correction import (
    _activation_rows,
    _add_probe_outputs,
    _all_names,
    _unique_name,
)
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.omniquant import _quantize_blockwise_int4_with_clip
from onnxsim.quip_sharp import _match_matmul_like


def _has_min_opset(model: onnx.ModelProto, min_version: int) -> bool:
    return any(
        o.domain in ("", "ai.onnx") and o.version >= min_version
        for o in model.opset_import
    )


def apply_spinquant(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies SpinQuant-style learned-rotation preprocessing (its own
    "R1-only" configuration, fit in closed form -- see this module's own
    docstring) plus block-wise INT4 quantization to every MatMul/vanilla-
    Gemm layer with a constant 2-D float32 weight whose reduction dimension
    ``K`` is divisible by ``block_size``.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to fit each layer's own rotation -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and :func:`onnxsim.load_huggingface_calibration_data`
            (real data, a more representative rotation fit than random
            input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per quantization block along ``K``,
            matching :func:`onnxsim.quantize_weight_only_int4`'s own
            default
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight replaced by
            ``(X @ U) @ Ŵtilde`` (plus the original bias, if any), where
            ``U`` is the fitted rotation and ``Ŵtilde`` is reconstructed
            in-graph from packed INT4 codes and a per-block float32 scale;
            output tensor name unchanged. Layers with a non-constant,
            non-2-D weight, a reduction dimension not divisible by
            ``block_size``, or no calibration activation available, are
            left untouched; a model with no matching layer, or an opset
            older than 21 (INT4's tensor type and ``DequantizeLinear``'s
            ``block_size`` attribute both need opset 21), is returned
            unchanged
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

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    probe_names = sorted({x_name for _, x_name, _, _, _ in candidates})
    probe_model = _add_probe_outputs(model, probe_names)
    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            arr = np.asarray(result[name], dtype=np.float64)
            activations[name].extend(_activation_rows([arr]))

    for node, x_name, w_name, bias_name, weight_transposed in candidates:
        acts = activations.get(x_name, [])
        if not acts:
            continue
        x = np.concatenate(acts, axis=0)

        w_init = initializer_map[w_name]
        w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
        w_nk = w if weight_transposed else w.T  # [N, K], output channel first
        n, k = w_nk.shape
        if k % block_size != 0 or x.shape[1] != k:
            continue

        # PCA/eigenvector-basis rotation: the closed-form, classical
        # orthogonal matrix diagonalizing this layer's own calibration-
        # activation covariance -- see module docstring. eigh always
        # returns an orthonormal basis for any real symmetric matrix, so
        # this is exact and well-defined even for a rank-deficient
        # (few-sample) covariance.
        cov = x.T @ x / x.shape[0]  # [K, K]
        _eigvals, u = np.linalg.eigh(cov)  # u: [K, K], orthogonal columns

        w_tilde_nk = w_nk @ u  # [N, K] -- exact before quantization

        codes_nk, scale_blocks_nk = _quantize_blockwise_int4_with_clip(
            w_tilde_nk, block_size, 1.0
        )
        codes_kn = codes_nk.T.astype(np.int64)  # [K, N], ready for a plain MatMul
        scale_kn = scale_blocks_nk.T.astype(np.float32)  # [K/block_size, N]

        prefix = f"{w_name}_spinquant"
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
        w_dequant = _new(
            "DequantizeLinear",
            [codes_name, scale_name],
            "w_dequant",
            axis=0,
            block_size=block_size,
        )
        core = _new("MatMul", [x_rotated, w_dequant], "core")

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
