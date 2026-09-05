"""MoEQuant (Hu, Chen et al., 2025, ICML 2025, "MoEQuant: Enhancing
Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced
Sampling and Affinity Guidance", https://arxiv.org/abs/2505.03804).

Every weight-only quantizer already in onnxsim -- including
:mod:`onnxsim.gptq`, whose Hessian-accumulation and column-update code this
module reuses directly (:func:`onnxsim.gptq._gptq_quantize_columns`) -- treats
calibration data uniformly: run N batches through the model, accumulate
``H = X^T X`` equally from every sample. That is the right assumption for a
plain MatMul/Gemm layer, where every calibration token exercises the same
weight the same way. It breaks down for a Mixture-of-Experts layer's own
per-expert weights (:func:`onnxsim.pruning.apply_moe_expert_channel_pruning`'s
and :func:`onnxsim.pruning.apply_moe_whole_expert_pruning`'s own
``com.microsoft::MoE`` target), because *which* expert a token's activation
actually informs is itself decided by the router, and real routers are
learned, imbalanced, and data-dependent: a handful of popular experts see the
bulk of any calibration set's tokens, while rarer experts see only a trickle
-- sometimes none at all. Naively running :mod:`onnxsim.gptq`-style
calibration once per expert (uniformly over whichever tokens happen to route
there) inherits that imbalance directly into the Hessian: popular experts get
a well-conditioned, plentiful calibration signal, rare experts get a noisy,
near-singular one (or nothing), and the quantization quality gap between them
tracks router popularity rather than the actual reconstruction error each
expert's own weight contributes to the model's output.

This module's own contribution is **not a new quantization algorithm** --
sequential Hessian-compensated column quantization is exactly
:mod:`onnxsim.gptq`'s, reused as-is via a precomputed, per-expert Hessian.
It is a new **calibration methodology**, specific to MoE's sparse,
data-dependent routing, with two distinguishing pieces:

- **Affinity-Guided Quantization (AGQ)**: rather than counting every token
  a router sends to expert ``e`` equally, each token ``t``'s contribution to
  expert ``e``'s own Hessian is weighted by that token's own router
  affinity/gate weight for ``e`` -- ``softmax(router_probs)[t, e]`` --  so a
  token the router is confident about (a high gate weight) counts for more
  than one that only barely qualified into the top-``k`` selection. Applied
  to both of an expert's own weights: ``fc1_experts_weights``' Hessian is
  built directly from ``H``'s own input activation (the MoE node's own
  ``input[0]``, shared -- ungated -- by every expert), and
  ``fc2_experts_weights``' Hessian is built from that same expert's own
  ``fc1``-then-activation output (recomputed here in plain numpy from the
  captured input and ``fc1``'s *own float* weight -- this module quantizes
  each expert's ``fc1``/``fc2`` independently from real activations, the same
  per-candidate independence :func:`onnxsim.gptq.apply_gptq` already uses,
  rather than chaining a quantized ``fc1``'s own output into ``fc2``'s own
  calibration).
- **Expert-Balanced Self-Sampling (EBSS)**: independent of how a token's
  contribution is *weighted*, an expert that a calibration set routes many
  tokens to still ends up with proportionally more calibration influence than
  a rarely-used one -- a *raw-count* imbalance no per-token weighting alone
  corrects (uniformly rescaling one expert's whole Hessian by a constant
  changes GPTQ's optimal-brain-surgeon correction not at all: the column
  update ``H^{-1}[i, j] / H^{-1}[i, i]`` is a ratio, invariant to any uniform
  scale of ``H``). This module's own EBSS step instead *subsamples* an
  over-represented expert's routed-token pool down to a shared, balanced
  per-expert budget (``ceil(tokens * k / num_experts)``, drawn without
  replacement, weighted by each token's own AGQ affinity) before that
  pool ever reaches the Hessian -- a real change to *which* activation
  directions inform that expert's own calibration statistic, not just an
  overall magnitude adjustment. An under-represented expert (at or below the
  shared budget already) is left alone: this module's own EBSS can rebalance
  which of the *already-captured* calibration tokens count towards an
  over-represented expert, but it cannot fabricate calibration coverage a
  rarely-routed expert never received in the first place -- the paper's own
  EBSS instead re-samples/re-runs a live calibration data *loader* until
  every expert clears its own target, which needs repeated fresh forward
  passes this module does not perform. Supplying more/larger calibration
  batches (raising ``num_samples``, or passing real data via
  :func:`onnxsim.load_huggingface_calibration_data`) is this module's own
  route to genuinely more coverage for a rare expert; ``ebss=False`` disables
  the subsampling step entirely (AGQ affinity weighting alone, over every
  routed token) for comparison.

Scope: targets plain ``com.microsoft::MoE`` nodes matched by
:func:`onnxsim.pruning._match_moe_producer` (the exact same matcher
:func:`onnxsim.pruning.apply_moe_expert_channel_pruning` uses -- see that
module's own section comment for precisely which shapes/activations/``fc3``/
``swiglu`` combinations are declined and why), restricted further here to
FLOAT32 ``fc1_experts_weights``/``fc2_experts_weights`` (FLOAT16/BFLOAT16,
which that matcher itself accepts, are left untouched -- the same float32-only
restriction :mod:`onnxsim.gptq`/:mod:`onnxsim.adaround` already carry).
Quantization itself is INT4, block-wise symmetric (this module computes its
own initial round-to-nearest scale the same way
:func:`onnxsim.quantize_weight_only_int4` does -- one scale per 32-element
block of the reduction axis, per output row -- since, unlike
:mod:`onnxsim.gptq`, there is no existing ``MoE``-targeting int4 rewrite to
read a scale back from), but the result is **simulated ("fake") quantization**:
each expert's ``fc1``/``fc2`` float32 initializer is overwritten in place with
its dequantized (``code * scale``) reconstruction, rather than packed into
ONNX Runtime's real quantized ``com.microsoft::QMoE`` contrib op (a
fundamentally different, non-trivial column-packed/scale/zero-point storage
format -- see :mod:`onnxsim.pruning`'s own extensive "QMoE" section comment
for its exact contract). The graph keeps its original ``MoE`` node, dtype, and
shapes, so it stays exactly as loadable/runnable as the input model; emitting
a real ``QMoE`` rewrite (bandwidth reduction, not just simulated precision
loss) is deliberately left as future work, the same kind of documented
scoping decision :func:`onnxsim.pruning.apply_moe_expert_channel_pruning`'s
own section comment already makes for ``fc3``/``swiglu``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.gptq import _gptq_quantize_columns
from onnxsim.pruning import _find_moe_chains, _MoEChain

_erf = np.vectorize(math.erf)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


_ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0.0),
    "identity": lambda x: x,
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "gelu": lambda x: 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0))),
}


def _activation_name(node: onnx.NodeProto) -> str:
    for attr in node.attribute:
        if attr.name == "activation_type":
            return attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
    return "relu"


def _moe_k(node: onnx.NodeProto, num_experts: int) -> int:
    k = 1
    for attr in node.attribute:
        if attr.name == "k":
            k = attr.i
    return max(1, min(k, num_experts))


def _ebss_select(
    routed_idx: np.ndarray,
    affinity: np.ndarray,
    target_count: int,
    rng: np.random.Generator,
) -> "tuple[np.ndarray, np.ndarray]":
    """Expert-Balanced Self-Sampling's own per-expert token selection: when
    an expert's routed-token pool exceeds the shared, balanced
    ``target_count`` budget, draws exactly ``target_count`` of them without
    replacement, weighted by each token's own AGQ affinity -- a real
    selection of *which* activation directions inform that expert's own
    Hessian, not merely a uniform rescale (see this module's own docstring
    for why the distinction matters to GPTQ's own column update). An expert
    already at or below the budget is returned unchanged: this module has no
    fresh calibration data to add for it (see this module's own EBSS
    docstring section).
    """
    if target_count <= 0 or routed_idx.size <= target_count:
        return routed_idx, affinity
    weights = affinity / affinity.sum()
    chosen = rng.choice(routed_idx.size, size=target_count, replace=False, p=weights)
    return routed_idx[chosen], affinity[chosen]


def _quantize_expert_weight(
    w_nk: np.ndarray,
    h: np.ndarray,
    quant_block_size: int,
    percdamp: float,
    proc_block_size: int,
) -> np.ndarray:
    """Block-wise symmetric INT4 round-to-nearest scale (own initial estimate,
    the same per-``quant_block_size``-block-of-the-reduction-axis, per-output-
    row scheme :func:`onnxsim.quantize_weight_only_int4` uses), then
    :func:`onnxsim.gptq._gptq_quantize_columns`'s own GPTQ column-update
    against ``h`` -- reused unchanged, only ``h`` itself (this module's own
    AGQ/EBSS-weighted Hessian) differs from how :mod:`onnxsim.gptq` builds it.
    Returns the dequantized (``code * scale``) reconstruction, same shape as
    ``w_nk``.
    """
    n, k = w_nk.shape
    bs = quant_block_size if quant_block_size > 0 and k % quant_block_size == 0 else k
    blocks = w_nk.reshape(n, k // bs, bs)
    amax = np.max(np.abs(blocks), axis=2)
    scale_blocks = np.where(amax == 0.0, 1.0, amax / 7.0)
    codes = _gptq_quantize_columns(w_nk, scale_blocks, bs, h, percdamp, proc_block_size)
    scale_full = np.repeat(scale_blocks, bs, axis=1)
    return codes * scale_full


def apply_moequant(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    quant_block_size: int = 32,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    ebss: bool = True,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """MoEQuant-calibrated INT4 (simulated) quantization of every matched
    ``com.microsoft::MoE`` node's per-expert ``fc1_experts_weights``/
    ``fc2_experts_weights``. See this module's own docstring for the AGQ/EBSS
    calibration methodology and the exact scope (FLOAT32 only, simulated
    quantization, matcher shared with
    :func:`onnxsim.pruning.apply_moe_expert_channel_pruning`).

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches to capture each
            MoE node's own input activations and router gate scores from.
            Each batch is a ``{input_name: np.ndarray}`` dict matching
            ``model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data --
            the only way to raise a rarely-routed expert's own genuine
            coverage; see this module's own EBSS docstring section).
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied) and for EBSS's own weighted
            subsampling
    :param quant_block_size: INT4 block size along each expert weight's own
            reduction axis (``hidden_size`` for ``fc1``, ``inter_size`` for
            ``fc2``) -- falls back to one block spanning the whole axis when
            it doesn't evenly divide it, e.g. a small toy/test model.
    :param percdamp: GPTQ Hessian damping factor, passed straight through to
            :func:`onnxsim.gptq._gptq_quantize_columns` -- see
            :func:`onnxsim.gptq.apply_gptq`'s own docstring
    :param proc_block_size: GPTQ's own column-processing block size, passed
            straight through to
            :func:`onnxsim.gptq._gptq_quantize_columns`
    :param ebss: when true (default), cap an over-represented expert's routed
            calibration tokens down to a shared, balanced per-expert budget
            (Expert-Balanced Self-Sampling) before building its Hessian; when
            false, every token the router ever sent to an expert (in the
            top-``k`` sense) contributes, weighted only by AGQ affinity.
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched, FLOAT32 MoE node's per-expert
            ``fc1``/``fc2`` weight simulated-INT4-quantized in place. An
            expert that received no calibration tokens at all (top-``k``
            selected it for zero observed tokens) is left at its original
            float value -- there is nothing for AGQ/EBSS to weight.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    chains: List[_MoEChain] = [
        c for c in _find_moe_chains(model.graph) if c.node.input[0] and c.node.input[1]
    ]
    if not chains:
        return model

    probe_names = sorted(
        {c.node.input[0] for c in chains} | {c.node.input[1] for c in chains}
    )
    probe_model = _add_probe_outputs(model, probe_names)

    collected: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        out = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            arr = np.asarray(out[name])
            if arr.ndim == 2:
                collected[name].append(arr.astype(np.float64))

    result = onnx.ModelProto()
    result.CopyFrom(model)
    initializer_map = {t.name: t for t in result.graph.initializer}
    touched: Set[str] = set()
    rng = np.random.default_rng(seed)

    for chain in chains:
        weight_names = {chain.fc1_w, chain.fc2_w}
        if weight_names & touched:
            continue  # a shared/tied initializer another MoE node already quantized
        touched |= weight_names

        fc1_init = initializer_map[chain.fc1_w]
        fc2_init = initializer_map[chain.fc2_w]
        if (
            fc1_init.data_type != onnx.TensorProto.FLOAT
            or fc2_init.data_type != onnx.TensorProto.FLOAT
        ):
            continue  # FLOAT16/BFLOAT16 experts are out of scope -- see docstring

        x_chunks = collected.get(chain.node.input[0], [])
        r_chunks = collected.get(chain.node.input[1], [])
        if not x_chunks or not r_chunks:
            continue  # no usable (2-D) calibration activation observed

        x_all = np.concatenate(x_chunks, axis=0)
        r_all = np.concatenate(r_chunks, axis=0)
        n_tokens = min(x_all.shape[0], r_all.shape[0])
        x_all, r_all = x_all[:n_tokens], r_all[:n_tokens]
        if x_all.shape[1] != chain.hidden_size or r_all.shape[1] != chain.num_experts:
            continue  # calibration activation doesn't match this chain's own shapes

        k = _moe_k(chain.node, chain.num_experts)
        affinity_all = _softmax(r_all)
        top_k = np.argsort(-r_all, axis=1)[:, :k]
        target_count = -(-(n_tokens * k) // chain.num_experts)  # ceil division

        fc1_w = onnx.numpy_helper.to_array(fc1_init).astype(np.float64)
        fc2_w = onnx.numpy_helper.to_array(fc2_init).astype(np.float64)
        fc1_b = None
        if chain.fc1_b is not None:
            fc1_b = onnx.numpy_helper.to_array(initializer_map[chain.fc1_b]).astype(
                np.float64
            )
        act_fn = _ACTIVATIONS.get(_activation_name(chain.node), _ACTIVATIONS["relu"])

        new_fc1 = fc1_w.copy()
        new_fc2 = fc2_w.copy()

        for e in range(chain.num_experts):
            routed_idx = np.nonzero(np.any(top_k == e, axis=1))[0]
            if routed_idx.size == 0:
                continue  # never routed to in calibration -- leave this expert float

            affinity = affinity_all[routed_idx, e]
            if ebss:
                routed_idx, affinity = _ebss_select(
                    routed_idx, affinity, target_count, rng
                )

            sqrt_affinity = np.sqrt(affinity)[:, None]
            xe = x_all[routed_idx] * sqrt_affinity
            h1 = xe.T @ xe

            pre_activation = x_all[routed_idx] @ fc1_w[e].T
            if fc1_b is not None:
                pre_activation = pre_activation + fc1_b[e]
            inter_act = act_fn(pre_activation) * sqrt_affinity
            h2 = inter_act.T @ inter_act

            new_fc1[e] = _quantize_expert_weight(
                fc1_w[e], h1, quant_block_size, percdamp, proc_block_size
            )
            new_fc2[e] = _quantize_expert_weight(
                fc2_w[e], h2, quant_block_size, percdamp, proc_block_size
            )

        fc1_init.CopyFrom(
            onnx.numpy_helper.from_array(new_fc1.astype(np.float32), name=chain.fc1_w)
        )
        fc2_init.CopyFrom(
            onnx.numpy_helper.from_array(new_fc2.astype(np.float32), name=chain.fc2_w)
        )

    return result
