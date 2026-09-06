"""BRECQ (Li, Gong, Tan, Yang, Hu, Zhang, Yu, Wang, Gu, 2021, "BRECQ: Pushing
the Limit of Post-Training Quantization by Block Reconstruction",
https://arxiv.org/abs/2102.05426, ICLR 2021).

Read :mod:`onnxsim.adaround` first -- this module extends its own
rectified-sigmoid rounding relaxation directly, in exactly the sense BRECQ's
own paper frames its contribution: not a new relaxation mechanism, but a
different *objective* to optimize it against.

Every reconstruction-based pass already in onnxsim
(:mod:`onnxsim.adaround`, :mod:`onnxsim.adaquant`, :mod:`onnxsim.gptq`,
:mod:`onnxsim.foem`) optimizes rounding **one layer at a time**, minimizing
that single layer's own output reconstruction error
(``||W_float @ x - W_quant @ x||^2``) against calibration activations. BRECQ's
own point: this is only provably layer-optimal-implies-network-optimal under
an *independence* assumption between layers that a real network never
satisfies -- a later layer's own reconstruction error is computed from an
input that already carries the *previous* layer's rounding error, so the two
errors are correlated, and independently minimizing each layer's own error is
not, in general, the choice that minimizes a multi-layer **block**'s own
final output error (the errors can reinforce instead of cancelling, exactly
the same "rounding choices interact" observation AdaRound itself makes one
level down, at the per-weight-element scale). BRECQ's fix: optimize every
layer inside a block *jointly*, by gradient descent through the whole
block's own forward computation, against the *block's own final output*
reconstruction error -- not each internal layer's output independently. The
paper also weights that reconstruction loss by a diagonal Fisher-information
approximation (each output element's own squared task-loss-gradient
magnitude) instead of plain MSE, so elements the downstream task is more
sensitive to get prioritized; a full block-scale Hessian (as
:mod:`onnxsim.gptq` affords for a *single* layer) is prohibitively expensive
here, so a cheap diagonal approximation is the paper's own compromise.

**What this module simplifies, honestly:**

- *Block definition.* The paper auto-detects blocks in real CNN/transformer
  architectures via architecture-specific heuristics (a ResNet "BasicBlock",
  a transformer encoder layer, ...). This module does not attempt that --
  arbitrary-graph block auto-detection is a much larger, architecture-specific
  problem, not a reconstruction-objective one. Instead, the caller identifies
  a block by two tensor names: ``block_input_name`` (the activation entering
  the block, e.g. its own input or a previous block's own output) and
  ``block_output_name`` (the block's own final output, e.g. a residual Add's
  own output). This module then auto-discovers, by walking the float graph
  between those two names, which quantized MatMul/Gemm layers belong to the
  block and in what order -- the caller does not have to enumerate them.
- *Block topology.* Discovery only recognizes a **linear chain** of
  quantized MatMul/Gemm layers -- each layer's own activation input must be
  exactly the previous layer's own output, with no intervening node -- plus
  an *optional* trailing residual ``Add`` whose other input is
  ``block_input_name`` itself. This covers a ResNet "BasicBlock"'s own two
  stacked convolutions plus its skip connection, or a transformer FFN's own
  up/down projection pair plus its residual, exactly the shapes the paper's
  own Section 4 targets -- but not a block with an interleaved
  normalization/activation node (BatchNorm, ReLU, LayerNorm, GELU, ...)
  between its own quantized layers. Extending discovery to walk through a
  supported allowlist of elementwise ops is future work, not attempted here;
  see this module's own tests for exactly the topology exercised.
- *Fisher-information weighting.* The paper's own diagonal Fisher estimate
  is the squared gradient of a real downstream *task* loss (e.g.
  classification cross-entropy) with respect to each block output element,
  which requires a task loss this generic ONNX setting does not have access
  to. This module instead uses each output element's own empirical variance
  across calibration samples (normalized to a mean of 1, so the overall loss
  scale stays comparable to plain MSE) as a cheap, computable proxy for "how
  much this element's own value varies, and therefore plausibly matters" --
  a real approximation of an approximation, not a claim of reproducing the
  paper's own Fisher estimate.
- *Benchmark numbers.* This module does not claim to reproduce the paper's
  own reported ImageNet/BERT accuracy gains -- only the mechanism (joint
  block reconstruction beats independent per-layer reconstruction on the
  block's own final output, on a toy scenario engineered to make the two
  differ), verified empirically in ``tests/test_brecq.py`` the same way
  :mod:`onnxsim.foem`'s own docstring scopes its measured (not assumed)
  improvement. The paper's own claim is modest -- an incremental gain on top
  of AdaRound, not a dramatic one -- and this module's own tests confirm
  only that same modest, honest, measured direction: on a scenario
  engineered so a residual block's two layers' errors interact, jointly
  optimizing the whole block against its own final output measurably beats
  optimizing each layer independently (via :func:`onnxsim.apply_adaround`)
  against that same final output; it is not a claim that joint block
  optimization always wins by a wide margin, or at all on every topology.

Targets the exact same scheme :mod:`onnxsim.adaround`/:mod:`onnxsim.gptq`/
:mod:`onnxsim.foem` do (:func:`onnxsim.quantize_weight_only_int4`'s
block-wise symmetric INT4 MatMul/Gemm), reusing
:func:`onnxsim.adaround._find_int4_matmul_candidates` to locate individual
layer candidates and :func:`onnxsim.adaround._h_and_dhdv` for the identical
rectified-sigmoid relaxation. Plain numpy throughout, hand-derived gradients
backpropagated through the block's own chain -- no autodiff framework,
matching every other reconstruction-based pass in this repository.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.adaround import (
    _GAMMA,
    _ZETA,
    _Candidate,
    _find_int4_matmul_candidates,
    _h_and_dhdv,
    _node_outputs,
    _pack_int4,
)
from onnxsim.bias_correction import _activation_rows, _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data

_N_MIN = -7.0
_N_MAX = 7.0


def _layer_arrays(
    c: _Candidate,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], bool]:
    """Returns ``(w_nk, scale_nk, (dim0, dim1), weight_transposed)`` for one
    candidate -- the float weight and its (block-broadcast) per-element
    scale, both normalized to ``[N, K]`` (output channel first), matching
    :mod:`onnxsim.adaround`'s own normalization exactly.
    """
    w = onnx.numpy_helper.to_array(c.w_float_init).astype(np.float64)
    scale = onnx.numpy_helper.to_array(c.ws_init).astype(np.float64)
    dim0, dim1 = w.shape
    if c.weight_transposed:
        w_nk = w  # already [N, K]
        scale_blocks = scale  # already [N, K / block_size]
    else:
        w_nk = w.T  # [K, N] -> [N, K]
        scale_blocks = scale.T  # [K / block_size, N] -> [N, K / block_size]
    scale_nk = np.repeat(scale_blocks, c.block_size, axis=1)[:, : w_nk.shape[1]]
    return w_nk, scale_nk, (dim0, dim1), c.weight_transposed


def _discover_block_chain(
    float_model: onnx.ModelProto,
    quantized_model: onnx.ModelProto,
    block_input_name: str,
    block_output_name: str,
) -> Optional[Tuple[List[_Candidate], bool]]:
    """Walks the float graph from ``block_input_name`` to
    ``block_output_name``, returning ``(chain, has_residual)`` -- the
    ordered list of quantized MatMul/Gemm candidates found along the way,
    and whether the chain ends in a residual ``Add`` back to
    ``block_input_name`` -- or ``None`` if no such chain/candidate exists
    (an unrecognized topology, not an error: callers are expected to pass
    tensor names that actually delimit a block they know the shape of).
    See this module's own docstring for exactly which topologies are
    recognized.
    """
    candidates = _find_int4_matmul_candidates(float_model, quantized_model)
    by_input = {c.float_node.input[0]: c for c in candidates}

    chain: List[_Candidate] = []
    seen_outputs = set()
    cur = block_input_name
    while cur != block_output_name:
        c = by_input.get(cur)
        if c is None or c.output_name in seen_outputs:
            break
        seen_outputs.add(c.output_name)
        chain.append(c)
        cur = c.output_name

    if cur == block_output_name:
        return (chain, False) if chain else None

    if not chain:
        return None

    f_by_output = _node_outputs(float_model.graph)
    add_node = f_by_output.get(block_output_name)
    if (
        add_node is not None
        and add_node.op_type == "Add"
        and len(add_node.input) == 2
        and set(add_node.input) == {cur, block_input_name}
    ):
        return chain, True

    return None


def _fisher_diag(final_float: np.ndarray, eps: float) -> np.ndarray:
    """Empirical per-output-element variance across calibration samples,
    normalized to a mean of 1 -- see this module's own docstring for why
    this stands in for the paper's own task-loss-gradient-based Fisher
    diagonal.
    """
    var = final_float.var(axis=0)
    mean_var = var.mean()
    if mean_var <= eps:
        return np.ones_like(var)
    return (var + eps) / (mean_var + eps)


def _optimize_block_rounding(
    chain: List[_Candidate],
    has_residual: bool,
    x0: np.ndarray,
    final_float: np.ndarray,
    num_iterations: int,
    learning_rate: float,
    reg_param: float,
    warm_start: float,
    beta_range: Tuple[float, float],
    fisher_eps: float,
) -> List[np.ndarray]:
    """Jointly optimizes every layer's rounding relaxation in ``chain``
    against the block's own final output reconstruction error (post
    residual add, if ``has_residual``), Fisher-diagonal-weighted. Returns
    the optimized integer codes for each layer, in ``[N, K]`` layout, same
    order as ``chain``.
    """
    layers = [_layer_arrays(c) for c in chain]
    w_nks = [w for w, _, _, _ in layers]
    scale_nks = [s for _, s, _, _ in layers]
    floor_bases = [np.floor(w / s) for w, s in zip(w_nks, scale_nks)]

    v_list = []
    for w, s, floor_base in zip(w_nks, scale_nks, floor_bases):
        frac = np.clip(w / s - floor_base, 1e-4, 1.0 - 1e-4)
        # Same rectified-sigmoid inverse-at-init as onnxsim.adaround: start
        # each element's relaxation at round-to-nearest's own choice.
        sig0 = np.clip((frac - _GAMMA) / (_ZETA - _GAMMA), 1e-4, 1.0 - 1e-4)
        v_list.append(np.log(sig0 / (1.0 - sig0)))

    fisher = _fisher_diag(final_float, fisher_eps)

    m_list = [np.zeros_like(v) for v in v_list]
    v2_list = [np.zeros_like(v) for v in v_list]
    adam_beta1, adam_beta2, adam_eps = 0.9, 0.999, 1e-8

    warm_start_iters = int(num_iterations * warm_start)
    beta_start, beta_end = beta_range
    n_elems = x0.shape[0] * final_float.shape[1]
    num_layers = len(chain)

    for t in range(num_iterations):
        # Forward: run the whole chain with each layer's current relaxation.
        ys = [x0]
        h_list, dh_dv_list, w_hat_list, active_list = [], [], [], []
        for w_nk, scale_nk, floor_base, v in zip(w_nks, scale_nks, floor_bases, v_list):
            h, dh_dv = _h_and_dhdv(v)
            raw = floor_base + h
            w_hat = np.clip(raw, _N_MIN, _N_MAX) * scale_nk
            active = (raw > _N_MIN) & (raw < _N_MAX)
            h_list.append(h)
            dh_dv_list.append(dh_dv)
            w_hat_list.append(w_hat)
            active_list.append(active)
            ys.append(ys[-1] @ w_hat.T)

        final_hat = ys[-1] + x0 if has_residual else ys[-1]
        diff = final_hat - final_float
        grad_y = 2.0 * fisher[None, :] * diff / n_elems  # dL/dys[-1]

        # Backward through the chain, layer by layer, propagating the
        # gradient of the *block's own final output* loss back to each
        # layer's own weight relaxation -- the joint part of BRECQ: a
        # layer's own gradient here depends on every downstream layer's
        # current weights, not just its own output.
        grads_v_reversed: List[np.ndarray] = []
        for layer_idx in range(num_layers - 1, -1, -1):
            dl_dw_hat = grad_y.T @ ys[layer_idx]  # [N_l, K_l]
            dl_dh = dl_dw_hat * np.where(
                active_list[layer_idx], scale_nks[layer_idx], 0.0
            )
            grad_v = dl_dh * dh_dv_list[layer_idx]

            if t >= warm_start_iters:
                progress = (t - warm_start_iters) / max(
                    1, num_iterations - warm_start_iters - 1
                )
                beta = beta_start + (beta_end - beta_start) * progress
                u = 2.0 * h_list[layer_idx] - 1.0
                abs_u = np.abs(u)
                dreg_dh = (
                    -2.0 * reg_param * beta * np.sign(u) * np.power(abs_u, beta - 1.0)
                )
                grad_v = grad_v + dreg_dh * dh_dv_list[layer_idx]

            grads_v_reversed.append(grad_v)
            if layer_idx > 0:
                grad_y = grad_y @ w_hat_list[layer_idx]  # dL/dys[layer_idx]
        grads_v = list(reversed(grads_v_reversed))

        for layer_idx in range(num_layers):
            m_list[layer_idx] = (
                adam_beta1 * m_list[layer_idx] + (1.0 - adam_beta1) * grads_v[layer_idx]
            )
            v2_list[layer_idx] = adam_beta2 * v2_list[layer_idx] + (
                1.0 - adam_beta2
            ) * (grads_v[layer_idx] * grads_v[layer_idx])
            m_hat = m_list[layer_idx] / (1.0 - adam_beta1 ** (t + 1))
            v_hat = v2_list[layer_idx] / (1.0 - adam_beta2 ** (t + 1))
            v_list[layer_idx] = v_list[layer_idx] - learning_rate * m_hat / (
                np.sqrt(v_hat) + adam_eps
            )

    codes = []
    for floor_base, v in zip(floor_bases, v_list):
        h_final, _ = _h_and_dhdv(v)
        codes.append(np.clip(floor_base + np.round(h_final), _N_MIN, _N_MAX))
    return codes


def apply_brecq(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    blocks: Sequence[Tuple[str, str]],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    learning_rate: float = 0.1,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
    fisher_eps: float = 1e-3,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Optimizes BRECQ-style joint, Fisher-weighted block reconstruction for
    every ``quantize_weight_only_int4``-quantized MatMul/Gemm chain
    delimited by ``blocks``, using real activations captured from
    ``float_model``. See this module's own docstring for the technique, its
    block-discovery contract, and what it simplifies relative to the paper.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized) are left untouched.
            Assumes ``quantized_model`` was produced from ``float_model``
            without renaming any MatMul/Gemm node's own output tensor --
            true of every onnxsim ``quantize_*`` function.
    :param blocks: ``(block_input_name, block_output_name)`` pairs, one per
            residual/linear block to jointly optimize -- see this module's
            own docstring for exactly which topologies between those two
            tensor names are recognized. A pair whose topology isn't
            recognized (or that matches no quantized layer at all) is
            silently skipped, the same "no matching candidate" tolerance
            :func:`onnxsim.apply_adaround` and friends already have.
    :param calibration_data: representative input batches to optimize the
            rounding on. Each batch is a ``{input_name: np.ndarray}`` dict
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
    :param num_iterations: Adam steps to run per block
    :param learning_rate: Adam learning rate for the per-element rounding
            relaxation, same role as :func:`onnxsim.apply_adaround`'s
            parameter of the same name
    :param reg_param: weight of the regularization term that pulls each
            element's relaxation toward a hard 0/1 (floor/ceil) decision,
            applied independently per layer inside the block
    :param warm_start: fraction of ``num_iterations`` (from the start) run
            with the regularization term disabled
    :param beta_range: ``(beta_start, beta_end)`` for the regularization
            term's exponent, linearly annealed across the iterations after
            ``warm_start``
    :param fisher_eps: numerical floor added to the empirical per-element
            variance before normalizing it into the Fisher-diagonal weight
            (see this module's own docstring) -- keeps a near-constant
            output element (near-zero variance) from collapsing to a
            near-zero loss weight and losing all optimization signal
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every discovered block's layers'
            INT4 weight initializers rewritten to their jointly-optimized
            codes (same shape, dtype, and scale -- only which integer each
            element rounds to changes)
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    discovered = []
    for block_input_name, block_output_name in blocks:
        found = _discover_block_chain(
            float_model, quantized_model, block_input_name, block_output_name
        )
        if found is not None:
            discovered.append((block_input_name, block_output_name, *found))

    if not discovered:
        return quantized_model

    probe_names = sorted(
        {block_input_name for block_input_name, _, _, _ in discovered}
        | {block_output_name for _, block_output_name, _, _ in discovered}
    )
    float_probe = _add_probe_outputs(float_model, probe_names)

    activations: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        out = backend.run_model(float_probe, batch, providers=providers)
        for name in probe_names:
            activations[name].append(np.asarray(out[name], dtype=np.float64))

    optimized: Dict[str, np.ndarray] = {}
    for block_input_name, block_output_name, chain, has_residual in discovered:
        # Keep each batch's block-input/block-output pair together: only a
        # batch whose two probed tensors flatten to the *same* number of
        # rows is usable, and concatenating them independently (rather
        # than pairwise) could silently misalign samples if the two ever
        # disagreed per batch. A block preserves its token axis, so a
        # [batch, seq, K] input and its [batch, seq, N] output both
        # flatten to batch * seq rows, in the same order.
        x0_batches = []
        final_batches = []
        for xa, fa in zip(
            activations[block_input_name], activations[block_output_name]
        ):
            xr = _activation_rows([xa])
            fr = _activation_rows([fa])
            if xr and fr and xr[0].shape[0] == fr[0].shape[0]:
                x0_batches.append(xr[0])
                final_batches.append(fr[0])
        if not x0_batches:
            continue  # no usable activation pair; skip this block
        x0 = np.concatenate(x0_batches, axis=0)
        final_float = np.concatenate(final_batches, axis=0)

        first_w_nk, _, _, _ = _layer_arrays(chain[0])
        if x0.shape[1] != first_w_nk.shape[1]:
            continue  # activation's feature dim doesn't match the first layer's K; skip

        layer_codes = _optimize_block_rounding(
            chain,
            has_residual,
            x0,
            final_float,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            reg_param=reg_param,
            warm_start=warm_start,
            beta_range=beta_range,
            fisher_eps=fisher_eps,
        )
        for c, codes_nk in zip(chain, layer_codes):
            _, _, (dim0, dim1), weight_transposed = _layer_arrays(c)
            codes_orig = codes_nk if weight_transposed else codes_nk.T
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
