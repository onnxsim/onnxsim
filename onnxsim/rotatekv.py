"""RotateKV (Su, Wan, Zhu, Yang, Kang, Chen, Peng, Shao, He, Chen, Sui, Ma,
Yang, 2025, "RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for
LLMs via Outlier-Aware Adaptive Rotations", https://arxiv.org/abs/2501.16383).
onnxsim ports the algorithm, not the paper's own CUDA fast-Hadamard
kernels, per the same rationale as
:mod:`onnxsim.spinquant`/:mod:`onnxsim.quip_sharp` (no ONNX export path
for the paper's own reference implementation).

Distinguishing this module from its two nearest siblings, both already in
this repo:

- :mod:`onnxsim.kv_cache_quantization` (KIVI/KVQuant) already quantizes
  Key **per channel**, static, calibrated -- but it never touches the
  values themselves before quantizing, only picks a scale for whatever
  per-channel range the calibration data happens to show. When a handful
  of channels are persistent, large-magnitude outliers (KIVI's own
  motivating finding), a shared per-channel scale still has to cover that
  whole range, wasting precision on every *other*, non-outlier channel
  sharing the same scale.
- :mod:`onnxsim.qoq`'s :func:`onnxsim.apply_smooth_attention` (QServe's
  SmoothAttention) already addresses that outlier-channel problem for
  attention's Key -- but only with a **diagonal scale** migrated between
  Key and Query (``K_j /= s_j``, ``Q_j *= s_j`` per channel ``j``), the
  same ``(X/s)@(W*s)==X@W`` identity :mod:`onnxsim.smoothquant` uses for
  weights. A diagonal rescaling can shrink an outlier channel's
  *magnitude* but can never change *which* channels the outlier mass lands
  on -- it is fundamentally channel-preserving.

RotateKV's own distinguishing mechanism is a **rotation**, not a scale:
conjugating Key (and compensating Query) by an orthogonal matrix, the same
"rotate before quantizing to spread outlier mass across every channel
evenly" idea :mod:`onnxsim.spinquant`/:mod:`onnxsim.quip_sharp` already
use for *weights* -- but applied here to the **KV-cache activation
stream** itself, and, per the paper's own title ("Outlier-Aware Adaptive
Rotations"), *fit* per structural unit from that unit's own calibration
statistics (this module's closed-form substitute for the paper's own more
elaborate optimization -- see below) rather than either a single
fixed/random Hadamard rotation shared by the whole model (QuIP#-style) or
one whole-model learned rotation (SpinQuant's own "R1" -- one rotation for
the entire residual stream). A rotation, unlike a diagonal scale, can
redistribute outlier *mass* across channels, not merely rescale it --
exactly the paper's own reason for preferring it over SmoothAttention-
style migration when pushing Key down to its own aggressive (2-bit)
target.

Fitting: like :mod:`onnxsim.spinquant` (see that module's own docstring
for the rationale), this module substitutes the paper's own more involved,
non-closed-form outlier-aware construction (channel reordering plus a
Hadamard-based rotation, calibration-driven but fit via an optimization
procedure, not an eigendecomposition) with the same classical, closed-
form, independently-verifiable stand-in :mod:`onnxsim.spinquant` already
uses elsewhere in this repo: the eigenvector basis of the calibration
activation's own covariance (``numpy.linalg.eigh``) -- fit **per matched
KV-cache stream** (see below for what counts as one structural unit here)
from that stream's own calibration-activation statistics alone, literally
adaptive to (and only to) that one stream's own outlier structure -- the
paper's own core "outlier-aware, adaptive" claim minus its bespoke
optimizer, exactly like :mod:`onnxsim.spinquant` fits one rotation per
layer rather than reusing the paper's own gradient-based joint fit.

Scope/simplification relative to the paper (documented explicitly, the
same way :mod:`onnxsim.billm`'s own docstring documents its own scoped-
down simplification): the paper fits one rotation **per attention head**,
which requires splitting a fused/flat QKV projection's output channels
into per-head blocks first. Doing that split structurally in an arbitrary
ONNX graph (locating the Reshape/Transpose that turns one flat
``[..., num_heads * head_dim]`` projection output into per-head
``[..., num_heads, head_dim]`` tensors, for an unknown ``num_heads``) is
out of scope for this module; instead, this module fits and applies one
rotation per matched KV-cache stream
(:mod:`onnxsim.kv_cache_quantization`'s own
``Concat(past, new, axis=seq)`` pattern) -- which is exactly "one rotation
per head" whenever a graph already keeps each attention head's Key stream
as its own separate KV-cache Concat (a common decomposed-per-head export
shape), and is otherwise still a sound, exact, purely-additive "one
rotation per matched Key tensor" contribution -- the same "single-
rotation-per-Key-tensor" scoping this project already treats as a
legitimate, mergeable, scoped-down contribution on its own.

Graph rewrite -- per matched Key-style KV-cache stream (see
:mod:`onnxsim.kv_cache_quantization` for what "Key-style" means and how it
is matched) whose ``present_name`` also feeds -- directly, or through
exactly one ``Transpose`` (the common shape when the cache's own head-dim
axis is last but the attention math needs it second-to-last for
``QK^T``) -- some attention subgraph's ``QK^T`` MatMul
(:mod:`onnxsim.attention_quantization`'s own decomposed attention pattern)
as its ``Kt`` operand. A stream with no such attention consumer is left
untouched entirely: rotating Key alone with no way to compensate Query
would silently change the attention scores, and this module never does
that.

    Before:
      new_key: float32 [..., seq_new, head_dim]   -- this step's fresh Key
      present_key = Concat(past_key, new_key, axis=seq)   -- feeds the cache
      Kt = present_key, or Transpose(present_key)          -- feeds QK^T
      scores = MatMul(Q, Kt)

    After:
      R: initializer, float32 [head_dim, head_dim]   -- the fitted rotation
      new_key_rotated = MatMul(new_key, R)
      present_key = Concat(past_key, new_key_rotated, axis=seq)  -- Concat
                    node itself unchanged; now carries rotated data for
                    every token cached through this same (modified) graph
      Q_rotated = MatMul(Q, R)
      scores = MatMul(Q_rotated, Kt)     -- Kt still reads present_key
                    exactly as before, now transparently rotated

Exactness: for any orthogonal ``R`` (``R^T @ R == I``), rotating a
tensor's own head-dim axis by ``R`` and then transposing it commutes with
rotating the other, un-transposed operand of a dot product by that exact
same ``R`` on its own head-dim axis --
``(Q @ R) @ (X @ R)^T == (Q @ R) @ (R^T @ X^T) == Q @ (R @ R^T) @ X^T ==
Q @ X^T`` -- so the migration is provably exact (up to floating-point
rounding), the same "provably exact migration, then quantize" contract
:mod:`onnxsim.spinquant`/:mod:`onnxsim.smoothquant`/:mod:`onnxsim.qoq`
already use. This module only performs that migration -- it returns a
float-equivalent model, no quantization happens here at all -- meant to
run immediately before :func:`onnxsim.quantize_kv_cache` (Key-style, the
default) in a pipeline, the same relationship
:func:`onnxsim.apply_smooth_attention` already has with it:
:func:`onnxsim.quantize_kv_cache` then quantizes the *rotated* Key stream,
unaware anything changed -- this module never reimplements KV-cache
quantization itself.

A past-cache consistency note, shared with every other calibrated
KV-cache technique in this repo: ``R`` is a single fixed constant baked
into the graph, applied identically to *every* step's own freshly
computed Key for as long as this modified graph is used for one
generation -- so a genuinely empty starting cache (``seq_past == 0``) and
a cache populated entirely by prior calls to this same modified graph both
end up consistently rotated throughout. A cache spliced in from outside
this modified graph (e.g. from an unrotated run) would not be, but that is
already true of every other calibrated per-channel scale in
:mod:`onnxsim.kv_cache_quantization`, not something new here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.attention_quantization import _find_attention_candidates
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.kv_cache_quantization import (
    _find_kv_cache_candidates,
    _is_value_style,
    _KvCacheCandidate,
)


def _resolve_kt_source(
    kt_name: str, producer_by_output: Dict[str, onnx.NodeProto]
) -> str:
    """Unwraps at most one ``Transpose`` hop: ``Kt = Transpose(X)`` (the
    shape a KV-cache stream's own head-dim-last layout needs to become the
    head-dim-second-to-last layout ``QK^T`` wants) resolves to ``X``.
    Anything else -- no producer (a graph input/output), a non-Transpose
    producer, or a multi-input node -- resolves to ``kt_name`` itself,
    unchanged.
    """
    node = producer_by_output.get(kt_name)
    if node is not None and node.op_type == "Transpose" and len(node.input) == 1:
        return node.input[0]
    return kt_name


@dataclass
class _RotateKvTarget:
    kv_candidate: _KvCacheCandidate
    q_name: str
    qk_matmul: onnx.NodeProto


def _find_rotatekv_targets(graph: onnx.GraphProto) -> List[_RotateKvTarget]:
    kv_candidates = [
        c
        for c in _find_kv_cache_candidates(graph)
        if not _is_value_style(c.present_name, None)
    ]
    if not kv_candidates:
        return []

    producer_by_output: Dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            producer_by_output[out] = node

    attention_candidates = []
    seen = set()
    for a in _find_attention_candidates(graph):
        if id(a.qk_matmul) in seen:
            continue
        seen.add(id(a.qk_matmul))
        attention_candidates.append(a)

    kt_source_by_matmul = {
        id(a.qk_matmul): _resolve_kt_source(a.qk_matmul.input[1], producer_by_output)
        for a in attention_candidates
    }

    targets = []
    for c in kv_candidates:
        match = next(
            (
                a
                for a in attention_candidates
                if kt_source_by_matmul[id(a.qk_matmul)] == c.present_name
            ),
            None,
        )
        if match is None:
            continue  # no attention consumer found -- can't compensate Query
        targets.append(
            _RotateKvTarget(
                kv_candidate=c,
                q_name=match.qk_matmul.input[0],
                qk_matmul=match.qk_matmul,
            )
        )
    return targets


def apply_rotatekv(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Applies RotateKV-style outlier-aware rotation preprocessing (a
    closed-form per-stream rotation, fit from calibration data -- see this
    module's own docstring) to every matched Key-style KV-cache stream
    whose ``present_name`` also feeds some attention subgraph's ``QK^T``
    MatMul (see the module docstring for the exact match).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches (each a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs) used to fit each matched stream's own rotation from its
            own freshly computed Key activation -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data,
            a more representative rotation fit than random input)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to calibrate on
    :returns: ``model`` with every matched stream's fresh Key activation
            and its compensating Query tensor each replaced by
            ``MatMul(_, R)`` (the same fitted ``R`` for both -- see the
            module docstring's exactness argument), feeding the original
            Concat/QK^T nodes unchanged; a stream with no matching
            attention consumer, or whose calibration activation never
            appeared in any batch, is left untouched, as is the whole
            model when no stream matches at all
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    targets = _find_rotatekv_targets(graph)
    if not targets:
        return out

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    probe_names = sorted({t.kv_candidate.new_name for t in targets})
    probe_model = _add_probe_outputs(out, probe_names)

    samples: Dict[str, List[np.ndarray]] = {name: [] for name in probe_names}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            arr = np.asarray(result[name], dtype=np.float64)
            if arr.ndim == 0:
                continue
            samples[name].append(arr.reshape(-1, arr.shape[-1]))

    taken_names = _all_names(graph)

    for t in targets:
        c = t.kv_candidate
        batches = samples.get(c.new_name, [])
        if not batches:
            continue  # this stream's activation never appeared in any batch
        x = np.concatenate(batches, axis=0)
        head_dim = x.shape[1]
        if head_dim < 2:
            continue  # nothing to rotate

        # Closed-form, classical eigenvector-basis rotation -- see module
        # docstring. eigh always returns an orthonormal basis for any real
        # symmetric matrix, so this is exact and well-defined even for a
        # rank-deficient (few-sample) covariance.
        cov = x.T @ x / x.shape[0]  # [head_dim, head_dim]
        _eigvals, r = np.linalg.eigh(cov)  # r: [head_dim, head_dim], orthogonal
        r32 = r.astype(np.float32)

        prefix = f"{c.present_name}_rotatekv"
        r_name = _unique_name(f"{prefix}_r", taken_names)
        graph.initializer.append(onnx.numpy_helper.from_array(r32, name=r_name))

        new_key_rot_name = _unique_name(f"{c.new_name}_rotatekv", taken_names)
        new_key_node = onnx.helper.make_node(
            "MatMul",
            [c.new_name, r_name],
            [new_key_rot_name],
            name=_unique_name(f"{prefix}_new_key_node", taken_names),
        )

        q_rot_name = _unique_name(f"{t.q_name}_rotatekv", taken_names)
        q_node = onnx.helper.make_node(
            "MatMul",
            [t.q_name, r_name],
            [q_rot_name],
            name=_unique_name(f"{prefix}_q_node", taken_names),
        )

        if c.new_is_first_input:
            c.concat_node.input[0] = new_key_rot_name
        else:
            c.concat_node.input[1] = new_key_rot_name
        t.qk_matmul.input[0] = q_rot_name

        concat_idx = next(i for i, n in enumerate(graph.node) if n is c.concat_node)
        graph.node.insert(concat_idx, new_key_node)

        qk_idx = next(i for i, n in enumerate(graph.node) if n is t.qk_matmul)
        graph.node.insert(qk_idx, q_node)

    return out
