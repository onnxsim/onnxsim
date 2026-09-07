"""Label-free, block-wise quantization-aware fine-tuning -- ``docs/qat.md``'s
deliverable B, the "knowledge-distillation QAT" that stage 2 of that note
describes.

Read :mod:`onnxsim.brecq` first. It already optimizes a *block's own final
output* reconstruction error rather than each layer's own, which is the
objective this module keeps unchanged. Two things it does not do, and this
module does:

1. **Any topology :mod:`onnxsim.graph_grad` can differentiate.** BRECQ's own
   block discovery recognizes a strict linear chain of MatMul/Gemm layers
   (plus an optional trailing residual ``Add``), and its own docstring says
   why: every op shape between two quantized layers would mean another
   hand-derived backward pass. :mod:`onnxsim.graph_grad` removed that cost --
   it differentiates a slice of an ONNX graph by walking it in reverse and
   emitting ordinary ONNX nodes -- so a normalization, an activation, a
   GELU's ``Erf``, a residual, or a Softmax between two Linears is now just
   more nodes in the slice. That is the headline change here.
2. **The float weights themselves move.** Every reconstruction pass in this
   repository -- AdaRound, BRECQ, FOEM, FlexRound, AutoRound -- optimizes
   only *which of the two neighbouring integers* a weight rounds to. That
   restriction is what makes them rounding passes. Here the fp32 weight is
   the trained parameter, fake-quantized in the forward against the same
   block-wise INT4 grid, with a straight-through estimator through the
   ``round``/``clip``: an element can migrate several codes away from its
   round-to-nearest starting point if the block's output error says it
   should. That is what makes this QAT rather than a seventh rounding pass.

Optionally (``learn_scales=True``) each weight's per-block quantization
*scale* is trained alongside, LSQ-style (Esser et al., 2020, "Learned Step
Size Quantization"). The gradient is the one :mod:`onnxsim.autoround`
already derives in numpy -- ``d(w_hat)/d(scale)`` is ``code - w/scale`` where
the element is inside the clipping range and just ``code`` where it
saturates -- summed over each scale's own block.

**Everything runs as one ONNX step graph.** The fake-quant forward, the
backward emitted by :func:`onnxsim.graph_grad.build_backward`, and one
:func:`onnxsim.qat_graph.adam_update` per trained tensor are a single pure
``(constants, state, scalars) -> (next state, loss)`` function, driven by
:func:`onnxsim.qat_graph.run_step_graph`. So the loop reaches whatever
execution provider ``step_providers=`` names -- CUDA, an NPU EP, WebGPU in
the WASM build.

Everything this module *emits* stays inside
:data:`onnxsim.qat_graph.EP_FRIENDLY_OPS` to make that reach real rather
than nominal (no ``Round``:
:meth:`onnxsim.qat_graph.GraphBuilder.round_to_nearest` composes one out of
``Sign``/``Abs``/``Cast``, and this reuses it). Note the boundary, because it
is easy to over-read: the block's *own forward nodes are copied into the step
graph verbatim*, so a block containing a ``Relu``, a ``Softmax`` or a
``LayerNormalization`` produces a step graph containing those too. The
allowlist constrains the fake-quant, the backward and the optimizer -- the
parts this repository writes -- and says nothing about the block. Whether a
given block's step graph runs on a given accelerator therefore depends on
that backend's coverage of the block's own operators as well.

**What this deliberately is not, and does not claim.**

- *Not task-loss QAT.* There are no labels, no dataset API, no metric and no
  training lifecycle -- ``docs/qat.md``'s deliverable C, unchanged and still
  out of scope. The teacher's own activations are the only target, so the
  ceiling is "reproduce the float block", not "recover task accuracy the
  float block never had". Label-free distillation QAT is not paper-QAT
  accuracy and should not be advertised as it.
- *Not end-to-end training.* :func:`apply_qat` still trains one
  caller-named block per call, exactly :mod:`onnxsim.brecq`'s contract
  (``block_input_name`` / ``block_output_name``), and
  :func:`apply_qat_all_blocks` walks a whole model one block at a time --
  discovering the blocks with :func:`discover_qat_blocks` and, by default,
  feeding each one the student's own activation so it corrects what its
  predecessors left behind. What is still absent is a final pass that
  backpropagates through the *entire* graph at once against the model's own
  output; a block is always the unit of optimization, and the teacher's
  activations are always the target.
- *Not activation quantization.* This targets
  :func:`onnxsim.quantize_weight_only_int4`'s weight-only scheme, the same
  one AdaRound/BRECQ/FOEM target. Learnable activation scales exist in-tree
  (:mod:`onnxsim.adaquant`) but are not wired in here.
- *Calibration-scale, even minibatched.* ``batch_size=None`` (the default)
  is full-batch gradient descent: the whole calibration set is one static
  tensor baked into the step graph's shapes, as in every other
  reconstruction pass here. Passing a ``batch_size`` makes each step train
  on that many rows instead -- the set is still uploaded once and stays
  resident, and the step ``Gather``s its own rows out of it, so a step's
  cost stops scaling with the size of the set and a pass over the data
  performs many updates instead of one. What that does *not* do is lift the
  ceiling on how much data a run may use: the set remains one static tensor
  that has to fit in the execution provider's memory (:mod:`onnxsim.qat_graph`
  documents the alternative, which trades the residency away for an
  unbounded stream, and why it was not taken). This is still a
  calibration-scale budget, not a training-scale one, and there is still no
  task loss, no labels and no metric.
- *Measured, not assumed, and not a uniform win.* ``tests/test_qat.py``
  measures both claims rather than asserting them, and one of the two has a
  boundary worth stating here. On a two-Linear-plus-``Relu`` block the
  reconstruction error falls from 16.0 (round-to-nearest) to 6.5, and a
  GELU block's loss falls ~12x -- topologies no existing pass here can
  reconstruct at all. Against :func:`onnxsim.apply_adaround` on a *single*
  layer, where the objective is identical and only the parametrization
  differs, freeing the weight wins when the calibration activations are
  low-rank (rank 1: RTN 5.96, AdaRound 3.07, this 1.78) and **loses** when
  they are full-rank (rank 16: RTN 28.5, AdaRound 14.6, this 22.8). The
  reason is not subtle: a well-determined reconstruction problem has its
  optimum within one quantization step of round-to-nearest, so floor/ceil is
  all the freedom worth having and AdaRound's continuous rectified-sigmoid
  relaxation optimizes that restricted problem better than a hard
  straight-through estimator on a piecewise-constant loss does. Real
  calibration activations are strongly low-rank, which is why this is worth
  having -- but "QAT beats AdaRound" is not a claim this module makes.

**What the block contract accepts and refuses.** Refusing is loud
wherever the caller named the block: :func:`apply_qat` on a block it cannot
train raises :class:`ValueError`, never returning a silently unchanged
model. :func:`apply_qat_all_blocks` inverts that -- nobody named those
blocks, so an untrainable one is skipped with its reason recorded in the
returned :class:`QATBlockResult` and the walk continues. The slice is the
intersection of the two directions -- nodes downstream of
``block_input_name`` *and* upstream of ``block_output_name`` -- so naming a
boundary cannot drag in a subgraph on the far side of it. A tensor the slice
reads that is neither produced inside it nor an initializer is captured from
the float model as another teacher-forced constant (so a residual arriving
from further upstream, or a second graph input, is fine). It is refused if
``block_output_name`` is not produced by a node, if any node in the slice
has an op type :data:`onnxsim.graph_grad.SUPPORTED_OPS` does not cover
(including one no gradient reaches -- the same standard
:func:`onnxsim.graph_grad.build_backward` holds itself to), if the slice
contains no ``quantize_weight_only_int4``-quantized MatMul/Gemm to train, or
if the block's shapes cannot be inferred statically at opset 17.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper
import onnx.shape_inference

from onnxsim import backend, graph_grad, qat_graph
from onnxsim.adaround import _Candidate, _find_int4_matmul_candidates, _pack_int4
from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data

# quantize_weight_only_int4's symmetric INT4 range, same constants
# onnxsim.brecq pins for the same scheme.
_N_MIN = -7.0
_N_MAX = 7.0

# Every name this module introduces into the step graph starts here, so it
# cannot collide with a tensor name carried over from the float model.
_PREFIX = "qat__"


def _round_half_away(x: np.ndarray) -> np.ndarray:
    """Host-side twin of
    :meth:`onnxsim.qat_graph.GraphBuilder.round_to_nearest`.

    The export must round the trained master weights exactly the way the
    trained forward did, or the model that ships is not the model whose loss
    was measured. ``np.round`` is half-to-even and the step graph's rounding
    is half-away-from-zero; the two differ only on exact ties, but matching
    them costs one line.
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


@dataclass
class _Trained:
    """One quantized layer's trainable state inside the step graph.

    ``w`` is the fp32 master weight in the *storage* layout the graph's own
    initializer uses ([K, N] for a MatMul, [N, K] for a ``transB`` Gemm) --
    unlike :mod:`onnxsim.adaround` and :mod:`onnxsim.brecq`, which normalize
    to [N, K], because nothing here needs a normalized layout: the forward
    consumes the weight exactly where the block's own node reads it, and the
    scale's blocked axis is carried explicitly instead.
    """

    candidate: _Candidate
    w_input: str
    m_input: str
    v_input: str
    w_shape: Tuple[int, int]
    w_init: np.ndarray
    scale_axis: int
    scale_shape: Tuple[int, int]
    scale_init: np.ndarray
    scale_input: Optional[str] = None
    ms_input: Optional[str] = None
    vs_input: Optional[str] = None
    # Filled in as the graph is built.
    w_next: str = ""
    m_next: str = ""
    v_next: str = ""
    scale_next: str = ""
    ms_next: str = ""
    vs_next: str = ""


@dataclass(frozen=True)
class _Minibatch:
    """How one block's step graph reads a minibatch out of its calibration set.

    Present only when the caller asked for minibatching; ``None`` everywhere
    means the full-batch graph this module started with, node for node.

    The set itself stays a step-graph *constant* -- uploaded once and
    device-resident for the whole loop, exactly as in the full-batch case --
    and the step selects :attr:`size` of its :attr:`num_rows` rows with a
    ``Gather`` driven by :attr:`index_name`, a rank-1 int64 per-step input.
    See :mod:`onnxsim.qat_graph`'s module docstring for why the rows are
    selected inside the graph rather than fed to it, and what that choice
    does and does not buy.
    """

    size: int
    num_rows: int
    index_name: str = f"{_PREFIX}rows"


def _int64_const(b: qat_graph.GraphBuilder, values: Sequence[int]) -> str:
    """An int64 initializer, for the ``shape``/``axes`` tensor inputs
    ``Reshape`` and ``ReduceSum`` take from opset 13 on.
    :meth:`onnxsim.qat_graph.GraphBuilder.const` is float32-only, which is
    right for everything it was written for; these are the exceptions."""
    array = np.asarray(list(values), dtype=np.int64)
    name = b.name("i64")
    b.initializer.append(onnx.numpy_helper.from_array(array, name))
    return name


def _blocked_shapes(
    w_shape: Tuple[int, int], scale_shape: Tuple[int, int], axis: int, block_size: int
) -> Tuple[List[int], List[int]]:
    """The rank-3 views that turn a blocked scale into a full-size one and a
    full-size gradient back into a blocked one.

    A block-wise scale has one value per ``block_size`` consecutive weights
    along the reduction axis, so both directions are the same reshape: split
    that axis into ``(num_blocks, block_size)``, and the scale is the same
    tensor with a 1 in the ``block_size`` slot. Returns
    ``(split_weight_shape, scale_shape_with_a_1)``.
    """
    d = list(w_shape)
    s = list(scale_shape)
    if axis not in (0, 1):
        raise ValueError(f"unsupported blocked axis {axis} for a 2-D weight")
    other = 1 - axis
    if s[other] != d[other] or s[axis] * block_size != d[axis]:
        # np.repeat(...)[:, :k] is how the numpy passes tolerate a ragged
        # final block. Doing the same inside a graph would mean a Slice on a
        # dimension the accelerator backends compile statically, and the
        # scheme this targets never produces one (K is always a multiple of
        # its own block size), so it is refused instead of approximated.
        raise ValueError(
            f"weight shape {tuple(d)} is not an exact block-wise tiling of scale "
            f"shape {tuple(s)} with block_size {block_size} on axis {axis}"
        )
    split = d[:axis] + [s[axis], block_size] + d[axis + 1 :]
    with_one = s[:axis] + [s[axis], 1] + s[axis + 1 :]
    return split, with_one


def _broadcast_scale(
    b: qat_graph.GraphBuilder,
    scale: str,
    w_shape: Tuple[int, int],
    scale_shape: Tuple[int, int],
    axis: int,
    block_size: int,
) -> str:
    """A per-block scale expanded to one value per weight element.

    ``Expand`` would say this in one node, but it is outside
    :data:`onnxsim.qat_graph.EP_FRIENDLY_OPS`; multiplying by a constant of
    ones broadcasts identically and costs the size of one block.
    """
    split, with_one = _blocked_shapes(w_shape, scale_shape, axis, block_size)
    ones_shape = [1] * len(split)
    ones_shape[axis + 1] = block_size
    reshaped = b.op("Reshape", [scale, _int64_const(b, with_one)])
    tiled = b.mul(reshaped, b.const(np.ones(ones_shape, dtype=np.float32), "ones"))
    return b.op("Reshape", [tiled, _int64_const(b, list(w_shape))])


def _sum_over_blocks(
    b: qat_graph.GraphBuilder,
    grad: str,
    w_shape: Tuple[int, int],
    scale_shape: Tuple[int, int],
    axis: int,
    block_size: int,
) -> str:
    """The transpose of :func:`_broadcast_scale`: one scale is shared by a
    whole block of weights, so its gradient is the sum of theirs."""
    split, _ = _blocked_shapes(w_shape, scale_shape, axis, block_size)
    reshaped = b.op("Reshape", [grad, _int64_const(b, split)])
    return b.op(
        "ReduceSum", [reshaped, _int64_const(b, [axis + 1])], "blocksum", keepdims=0
    )


def _emit_fake_quant(
    b: qat_graph.GraphBuilder, w: str, scale_full: str, out_name: str
) -> Tuple[str, str, str]:
    """``w_hat = clip(round(w / s), -7, 7) * s``, written into ``out_name``.

    Returns ``(code, ratio, active)`` -- everything the straight-through
    backward below needs. ``active`` is a float 0/1 mask of the elements
    strictly inside the clipping range, which is where ``round``'s
    straight-through derivative of 1 is allowed to pass anything through at
    all; a saturated element's weight has no local influence on the block's
    output and must not be moved by the reconstruction gradient.

    Clipping happens *before* rounding rather than after. The two commute
    here because the bounds are integers, and doing it in this order leaves
    :meth:`onnxsim.qat_graph.GraphBuilder.round_to_nearest` with an argument
    already
    bounded to [-7, 7], where its float-to-int32 cast is exact.
    """
    ratio = b.div(w, scale_full)
    code = b.round_to_nearest(b.clip(ratio, _N_MIN, _N_MAX))
    b.nodes.append(onnx.helper.make_node("Mul", [code, scale_full], [out_name]))
    active = b.mul(
        b.greater_mask(ratio, _N_MIN),
        b.less_mask(ratio, _N_MAX),
    )
    return code, ratio, active


def _slice_block(
    graph: onnx.GraphProto, block_input_name: str, block_output_name: str
) -> Tuple[List[onnx.NodeProto], List[str]]:
    """The nodes that lie between ``block_input_name`` and
    ``block_output_name``: those downstream of the first *and* upstream of the
    second.

    Returns ``(nodes in graph order, externally-supplied tensor names)``.
    Intersecting the two directions is what makes the block boundary mean
    what a caller expects. Walking backwards from the output alone and merely
    stopping at the block input would keep following every *other* path out of
    the output -- a residual arriving from before the block would drag the
    whole earlier subgraph in with it, and the caller would find layers they
    never named being retrained.

    An "external" tensor is one the slice reads but does not produce and which
    is not an initializer: ``block_input_name`` itself, and anything entering
    the block sideways (that residual, an attention mask fed as a second graph
    input). Those are teacher-forced -- captured from the float model and fed
    to the step graph as constants, exactly as ``block_input_name`` is, which
    is the approximation every block-wise reconstruction method makes.
    """
    initializers = {t.name for t in graph.initializer}
    producer: Dict[str, int] = {}
    for index, node in enumerate(graph.node):
        for output in node.output:
            if output:
                producer[output] = index

    if block_output_name not in producer:
        raise ValueError(
            f"block output {block_output_name!r} is not produced by any node in the "
            "float graph; a block must end at a computed tensor"
        )

    # Forward: which nodes actually depend on the block input. The graph is
    # topologically ordered, so one pass suffices.
    downstream: Set[str] = {block_input_name}
    forward: Set[int] = set()
    for index, node in enumerate(graph.node):
        if any(name in downstream for name in node.input):
            forward.add(index)
            downstream.update(name for name in node.output if name)

    # Backward from the block output, confined to those nodes.
    used: Set[int] = set()
    external: Set[str] = set()
    seen: Set[str] = set()
    stack = [block_output_name]
    while stack:
        name = stack.pop()
        if not name or name in seen:
            continue
        seen.add(name)
        if name in initializers:
            continue
        owner = producer.get(name)
        if owner is None or owner not in forward:
            external.add(name)
            continue
        if owner in used:
            continue
        used.add(owner)
        stack.extend(graph.node[owner].input)

    nodes = [node for index, node in enumerate(graph.node) if index in used]
    return nodes, sorted(external)


def _refuse_unsupported(nodes: Sequence[onnx.NodeProto]) -> None:
    """Every op in the slice must have a gradient rule, checked before any
    calibration data is run.

    Tested against :data:`onnxsim.graph_grad.SUPPORTED_OPS` rather than by
    catching :class:`onnxsim.graph_grad.UnsupportedOpError` from
    ``build_backward``, which is what that module's own docstring asks
    callers who pick their own slice to do -- and it means the caller learns
    the block is out of scope in milliseconds rather than after a full
    activation capture.
    """
    unsupported = sorted(
        {n.op_type for n in nodes if n.op_type not in graph_grad.SUPPORTED_OPS}
    )
    if unsupported:
        raise graph_grad.UnsupportedOpError(
            f"the block contains {unsupported}, which onnxsim.graph_grad cannot "
            f"differentiate; it differentiates {sorted(graph_grad.SUPPORTED_OPS)}. "
            "Choose block boundaries that exclude those nodes."
        )


def _block_shapes(
    float_model: onnx.ModelProto,
    nodes: Sequence[onnx.NodeProto],
    externals: Dict[str, np.ndarray],
    block_output_name: str,
    block_output: np.ndarray,
) -> Dict[str, Sequence[int]]:
    """Static shapes for every tensor the slice touches -- what
    :func:`onnxsim.graph_grad.build_backward` requires of its caller.

    Inferred from a standalone model containing only the slice, whose inputs
    carry the *concrete* shapes of the captured calibration activations. The
    float model's own graph inputs are usually symbolic (``float[batch, K]``),
    and a symbolic dimension is exactly what a step graph cannot have: the
    accelerator backends this exists for compile a fixed graph, and undoing a
    broadcast at build time needs real numbers.

    Inference runs at opset 17, the pairing :mod:`onnxsim.qat_graph` emits, so
    a node the step graph could not legally carry is refused here rather than
    at session-creation time.
    """
    used = {name for node in nodes for name in node.input if name}
    initializers = [t for t in float_model.graph.initializer if t.name in used]
    inputs = [
        onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, list(value.shape)
        )
        for name, value in sorted(externals.items())
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            block_output_name, onnx.TensorProto.FLOAT, list(block_output.shape)
        )
    ]
    graph = onnx.helper.make_graph(
        list(nodes), "qat_block", inputs, outputs, initializer=initializers
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as error:  # noqa: BLE001 -- re-raised with the block's context
        raise ValueError(
            f"cannot statically infer the block's shapes at opset 17: {error}"
        ) from error

    shapes: Dict[str, Sequence[int]] = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    ):
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
        if any(d <= 0 for d in dims):
            raise ValueError(
                f"tensor {value.name!r} in the block has a non-static shape; "
                "block-wise QAT needs every shape known at build time"
            )
        shapes[value.name] = dims
    for t in inferred.graph.initializer:
        shapes[t.name] = list(t.dims)

    missing = sorted(
        {
            name
            for node in nodes
            for name in list(node.input) + list(node.output)
            if name
        }
        - set(shapes)
    )
    if missing:
        raise ValueError(
            f"shape inference did not produce a shape for {missing} in the block"
        )
    return shapes


def _capture(
    float_model: onnx.ModelProto,
    names: Sequence[str],
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[backend.Provider]],
) -> Dict[str, np.ndarray]:
    """Runs the float model on the calibration data and returns each probed
    tensor concatenated over batches along axis 0.

    Unlike :func:`onnxsim.bias_correction._activation_rows`, which the
    per-layer passes use, the captured rank is kept as-is. A single layer's
    reconstruction only ever needs the set of rows that multiply ``W``, so
    flattening ``[batch, seq, K]`` to ``[batch * seq, K]`` is exact there. A
    *block* is not rank-agnostic: it may contain a ``Softmax`` over a
    particular axis, a ``Reshape`` with a baked-in target, or a ``MatMul``
    that broadcasts over leading dimensions, and all three change meaning if
    the leading axes are collapsed. Concatenating along axis 0 keeps the
    block seeing the shape it was written for; it does assume axis 0 is a
    batch axis the block treats independently, which is what a calibration
    batch axis is.
    """
    probe = _add_probe_outputs(float_model, names)
    collected: Dict[str, List[np.ndarray]] = {name: [] for name in names}
    for batch in calibration_data:
        out = backend.run_model(probe, batch, providers=providers)
        for name in names:
            collected[name].append(np.asarray(out[name], dtype=np.float32))

    captured: Dict[str, np.ndarray] = {}
    for name, arrays in collected.items():
        trailing = {a.shape[1:] for a in arrays}
        if len(trailing) != 1:
            raise ValueError(
                f"calibration batches disagree on the shape of {name!r} "
                f"({sorted(trailing)}); every batch must differ only in its "
                "leading axis"
            )
        captured[name] = np.concatenate(arrays, axis=0)
    return captured


def _plan_trained(
    candidates: Sequence[_Candidate], learn_scales: bool
) -> List[_Trained]:
    """One :class:`_Trained` per quantized layer in the block, with its
    master weight seeded from the *float* model's own weight -- so step 0 of
    the loop reproduces round-to-nearest exactly, and every later step is a
    measured improvement on it rather than on an arbitrary re-initialization.
    """
    planned: List[_Trained] = []
    for i, candidate in enumerate(candidates):
        w = onnx.numpy_helper.to_array(candidate.w_float_init).astype(np.float32)
        scale = onnx.numpy_helper.to_array(candidate.ws_init).astype(np.float32)
        trained = _Trained(
            candidate=candidate,
            w_input=f"{_PREFIX}w{i}",
            m_input=f"{_PREFIX}mw{i}",
            v_input=f"{_PREFIX}vw{i}",
            w_shape=(int(w.shape[0]), int(w.shape[1])),
            w_init=w,
            scale_axis=candidate.axis,
            scale_shape=(int(scale.shape[0]), int(scale.shape[1])),
            scale_init=scale,
        )
        if learn_scales:
            trained.scale_input = f"{_PREFIX}s{i}"
            trained.ms_input = f"{_PREFIX}ms{i}"
            trained.vs_input = f"{_PREFIX}vs{i}"
        planned.append(trained)
    return planned


def _build_step_graph(
    trained: Sequence[_Trained],
    nodes: Sequence[onnx.NodeProto],
    shapes: Dict[str, Sequence[int]],
    block_initializers: Sequence[onnx.TensorProto],
    externals: Dict[str, np.ndarray],
    block_output_name: str,
    block_output_shape: Sequence[int],
    learn_scales: bool,
    batch: Optional[_Minibatch] = None,
) -> qat_graph.StepGraph:
    """The whole loop as one graph: fake-quant forward, block forward,
    reconstruction loss, backward, Adam.

    The ordering is the only subtle part. :func:`graph_grad.build_backward`
    reads forward tensors by name (including node *outputs*, where reusing a
    ``Sigmoid``/``Softmax`` result is cheaper than recomputing it), so every
    node it differentiates must already sit in the builder ahead of the nodes
    it appends. Hence: (optionally the minibatch gather,) fake-quant, then the
    block's own nodes verbatim, then the loss seed, then the backward, then
    the optimizer.

    ``externals`` and ``block_output_shape`` are always the *whole*
    calibration set's arrays and shape. With ``batch`` set they become the
    resident tables rather than the block's inputs, and the block's own
    tensors are ``batch.size`` rows gathered out of them -- so every shape
    from the block input downwards, the loss normalizer included, is a
    batch-sized shape, and none of the code below has to know which case it
    is in.
    """
    b = qat_graph.GraphBuilder(_PREFIX)
    b.initializer.extend(block_initializers)

    # 0. The minibatch, if there is one. Each captured tensor is declared at
    #    its full size and a Gather pulls this step's rows out of it under the
    #    name the block's own nodes were written against, so step 2 below can
    #    still splice those nodes in verbatim. The block never learns that its
    #    input stopped being a graph input.
    teacher = f"{_PREFIX}teacher"
    constants: Dict[str, Sequence[int]] = {}
    if batch is None:
        constants.update(
            {name: list(value.shape) for name, value in sorted(externals.items())}
        )
        constants[teacher] = list(block_output_shape)
    else:
        rows = batch.index_name
        # A captured tensor's table is ``qat__all_<its name>`` and the
        # teacher's is ``qat__teacher_all``; the two families cannot collide
        # whatever the model calls its tensors, since one starts ``qat__all_``
        # and the other ``qat__teacher_``.
        for name, value in sorted(externals.items()):
            table = f"{_PREFIX}all_{name}"
            constants[table] = list(value.shape)
            b.gather_rows(table, rows, name)
        constants[f"{_PREFIX}teacher_all"] = list(block_output_shape)
        b.gather_rows(f"{_PREFIX}teacher_all", rows, teacher)
        block_output_shape = [batch.size] + list(block_output_shape)[1:]

    # 1. Fake-quantize each trained master weight into the tensor name the
    #    block's own node already reads, so the block's nodes need no
    #    rewriting at all -- the weight initializer simply became a computed
    #    value.
    per_layer = []
    for t in trained:
        block_size = t.candidate.block_size
        if t.scale_input is None:
            scale = b.const(t.scale_init, "scale")
        else:
            scale = t.scale_input
        scale_full = _broadcast_scale(
            b, scale, t.w_shape, t.scale_shape, t.scale_axis, block_size
        )
        weight_name = t.candidate.float_node.input[1]
        code, ratio, active = _emit_fake_quant(b, t.w_input, scale_full, weight_name)
        per_layer.append((t, weight_name, scale_full, code, ratio, active))

    # 2. The block itself, node for node as the float graph wrote it.
    b.nodes.extend(nodes)

    # 3. The objective: MSE of the student block's output against the
    #    teacher's, and its gradient, which is the seed of the backward pass.
    #    ``block_output_shape`` is this step's shape, so the 2/n normalizer is
    #    the batch's element count and a minibatched gradient is the same
    #    *average* per-element quantity a full-batch one is -- which is what
    #    keeps one learning rate meaningful across batch sizes.
    diff = b.sub(block_output_name, teacher)
    n_elems = int(np.prod(list(block_output_shape)))
    dl_dy = b.mul(diff, b.const(2.0 / n_elems))

    # 4. The backward pass over the block, emitted as ONNX nodes.
    grads = graph_grad.build_backward(
        b,
        nodes,
        shapes,
        {block_output_name: dl_dy},
        [weight_name for _, weight_name, _, _, _, _ in per_layer],
    )

    # 5. Straight through the fake-quant, into the master weight and (if
    #    asked for) the scale, then one Adam step each.
    for t, weight_name, _scale_full, code, ratio, active in per_layer:
        g = grads[weight_name]  # dL/d(w_hat), in the weight's storage layout
        # STE: d(w_hat)/d(w) is 1 inside the clipping range and 0 outside.
        # The scale cancels -- w_hat = round(w/s)*s -- which is why a
        # straight-through weight gradient is just the masked output
        # gradient, with no scale factor anywhere.
        t.w_next, t.m_next, t.v_next = qat_graph.adam_update(
            b,
            t.w_input,
            b.mul(g, active),
            t.m_input,
            t.v_input,
            f"{_PREFIX}lr",
            "m_correction",
            "v_correction",
        )
        if t.scale_input is None:
            continue
        # LSQ's scale gradient, the same one onnxsim.autoround derives:
        # d(w_hat)/d(s) = code - w/s where the element is inside the clipping
        # range (the gap between the integer it rounds to and the exact
        # ratio) and just `code` where it saturates.
        dwhat_ds = b.sub(code, b.mul(active, ratio))
        g_scale = _sum_over_blocks(
            b,
            b.mul(g, dwhat_ds),
            t.w_shape,
            t.scale_shape,
            t.scale_axis,
            t.candidate.block_size,
        )
        t.scale_next, t.ms_next, t.vs_next = qat_graph.adam_update(
            b,
            t.scale_input,
            g_scale,
            str(t.ms_input),
            str(t.vs_input),
            f"{_PREFIX}lr_scale",
            "m_correction",
            "v_correction",
        )

    state: Dict[str, Tuple[Sequence[int], str]] = {}
    for t in trained:
        state[t.w_input] = (list(t.w_shape), t.w_next)
        state[t.m_input] = (list(t.w_shape), t.m_next)
        state[t.v_input] = (list(t.w_shape), t.v_next)
        if t.scale_input is not None:
            state[t.scale_input] = (list(t.scale_shape), t.scale_next)
            state[str(t.ms_input)] = (list(t.scale_shape), t.ms_next)
            state[str(t.vs_input)] = (list(t.scale_shape), t.vs_next)

    scalars = [f"{_PREFIX}lr", "m_correction", "v_correction"]
    if learn_scales:
        scalars.append(f"{_PREFIX}lr_scale")

    per_step: Optional[Dict[str, Tuple[Sequence[int], int]]] = None
    if batch is not None:
        per_step = {batch.index_name: ([batch.size], int(onnx.TensorProto.INT64))}

    return qat_graph.make_step_graph(
        b,
        constants=constants,
        state=state,
        scalars=scalars,
        loss=b.mean_square(diff),
        name="onnxsim_qat_step",
        per_step=per_step,
    )


@dataclass
class _BlockPlan:
    """Everything about one block that can be decided from the two graphs
    alone, before any calibration data exists.

    Separated out because both entry points need exactly this and nothing
    more: :func:`apply_qat` builds one from the names its caller supplied,
    and :func:`discover_qat_blocks` builds one per candidate boundary pair it
    proposes -- using the plan's construction as the *validation* of that
    proposal, so discovery and the single-block path can never disagree about
    what a legal block is.
    """

    input_name: str
    output_name: str
    nodes: List[onnx.NodeProto]
    externals: List[str]
    candidates: List[_Candidate]


def _plan_block(
    float_model: onnx.ModelProto,
    quantized_model: onnx.ModelProto,
    block_input_name: str,
    block_output_name: str,
) -> _BlockPlan:
    """Slices the block out of the float graph and checks the three things
    that make it trainable at all: it is non-empty, every op in it has a
    gradient rule, and at least one of its layers is INT4-quantized.

    Deliberately does *not* check shapes -- that needs the concrete
    calibration activations, so it happens later in :func:`_train_block`.
    """
    nodes, externals = _slice_block(
        float_model.graph, block_input_name, block_output_name
    )
    if not nodes:
        raise ValueError(
            f"no nodes lie between {block_input_name!r} and {block_output_name!r}"
        )
    _refuse_unsupported(nodes)

    slice_outputs = {out for node in nodes for out in node.output if out}
    candidates = [
        c
        for c in _find_int4_matmul_candidates(float_model, quantized_model)
        if c.output_name in slice_outputs
    ]
    if not candidates:
        raise ValueError(
            f"the block between {block_input_name!r} and {block_output_name!r} "
            "contains no quantize_weight_only_int4-quantized MatMul/Gemm layer to "
            "train"
        )
    return _BlockPlan(
        input_name=block_input_name,
        output_name=block_output_name,
        nodes=nodes,
        externals=externals,
        candidates=candidates,
    )


def _plan_minibatch(
    external_values: Dict[str, np.ndarray],
    teacher_output: np.ndarray,
    batch_size: Optional[int],
) -> Optional[_Minibatch]:
    """The block's minibatch plan, or ``None`` for the full-batch graph.

    ``None`` is returned for both ways of asking for full batch -- not
    passing a ``batch_size`` at all, and passing one at least as large as the
    calibration set -- and the second is the more interesting one. A batch
    that covers every row *is* the full-batch objective, so taking the
    full-batch path for it is not an approximation: it is the same
    computation, minus a ``Gather`` per captured tensor, and it keeps the
    default and the "batch_size larger than my data" case bit-for-bit
    identical to what this module did before minibatching existed. The
    alternative -- wrapping the index stream around and letting rows repeat
    inside a single batch -- would quietly reweight those rows.

    Minibatching needs a row axis, and this is where the assumption
    :func:`_capture` already makes ("axis 0 is a batch axis the block treats
    independently") stops being implicit: every captured tensor is sliced on
    axis 0 by the *same* index vector, so they must agree on how many rows
    they have. A block whose sideways input does not (a tensor computed from
    initializers alone, say) is refused for minibatching rather than sliced
    into nonsense -- and its full-batch path still works, which is what the
    error message says to do.
    """
    if batch_size is None:
        return None
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    leading = {int(value.shape[0]) for value in external_values.values()}
    leading.add(int(teacher_output.shape[0]))
    if len(leading) != 1:
        raise ValueError(
            "minibatching slices every captured tensor on axis 0 with one shared "
            f"index, so they must agree on their row count; got {sorted(leading)}. "
            "Leave batch_size unset to train this block full-batch."
        )
    num_rows = leading.pop()
    if batch_size >= num_rows:
        return None
    return _Minibatch(size=batch_size, num_rows=num_rows)


def _train_block(
    float_model: onnx.ModelProto,
    quantized_model: onnx.ModelProto,
    plan: _BlockPlan,
    external_values: Dict[str, np.ndarray],
    teacher_output: np.ndarray,
    *,
    num_iterations: int,
    learning_rate: float,
    learn_scales: bool,
    scale_learning_rate: float,
    lr_decay: bool,
    step_providers: Optional[Sequence[backend.Provider]],
    losses: Optional[List[float]],
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    batch_seed: int = 0,
) -> onnx.ModelProto:
    """Runs the whole optimization for one already-planned, already-captured
    block and returns ``quantized_model`` with that block's initializers
    rewritten.

    ``external_values`` are the block's inputs -- ``plan.input_name`` and
    anything entering sideways -- and ``teacher_output`` is the target. Which
    *model* those two came from is the caller's decision, and it is the whole
    difference between :func:`apply_qat`'s one-block contract and
    :func:`apply_qat_all_blocks`'s sequential walk: the target is always the
    teacher's, but the inputs may be the teacher's or the student's.
    """
    batch = _plan_minibatch(external_values, teacher_output, batch_size)
    # Shape inference sees one step's worth of rows, since that is what the
    # block's nodes -- and therefore the backward pass built from them -- will
    # actually be handed. The views are free; nothing is copied.
    if batch is None:
        block_inputs, block_target = external_values, teacher_output
    else:
        block_inputs = {k: v[: batch.size] for k, v in external_values.items()}
        block_target = teacher_output[: batch.size]

    shapes = _block_shapes(
        float_model, plan.nodes, block_inputs, plan.output_name, block_target
    )

    trained = _plan_trained(plan.candidates, learn_scales)
    trained_weight_names = {t.candidate.float_node.input[1] for t in trained}
    used = {name for node in plan.nodes for name in node.input if name}
    block_initializers = [
        t
        for t in float_model.graph.initializer
        if t.name in used and t.name not in trained_weight_names
    ]

    step = _build_step_graph(
        trained,
        plan.nodes,
        shapes,
        block_initializers,
        external_values,
        plan.output_name,
        list(teacher_output.shape),
        learn_scales,
        batch,
    )

    # The whole set is the constant either way; with a minibatch it is bound
    # under the private table names the gathers read instead of under the
    # block's own tensor names, and it is still uploaded exactly once.
    if batch is None:
        constants: Dict[str, np.ndarray] = dict(external_values)
        constants[f"{_PREFIX}teacher"] = teacher_output
    else:
        constants = {f"{_PREFIX}all_{k}": v for k, v in external_values.items()}
        constants[f"{_PREFIX}teacher_all"] = teacher_output
    state: Dict[str, np.ndarray] = {}
    for t in trained:
        state[t.w_input] = t.w_init
        state[t.m_input] = np.zeros_like(t.w_init)
        state[t.v_input] = np.zeros_like(t.w_init)
        if t.scale_input is not None:
            state[t.scale_input] = t.scale_init
            state[str(t.ms_input)] = np.zeros_like(t.scale_init)
            state[str(t.vs_input)] = np.zeros_like(t.scale_init)

    def scalars(t: int) -> Dict[str, float]:
        decay = 1.0 - t / num_iterations if lr_decay else 1.0
        values = {
            f"{_PREFIX}lr": learning_rate * decay,
            f"{_PREFIX}lr_scale": scale_learning_rate * decay,
        }
        if not learn_scales:
            del values[f"{_PREFIX}lr_scale"]
        values.update(qat_graph.adam_bias_corrections(t))
        return values

    feeds: Optional[Callable[[int], Dict[str, np.ndarray]]] = None
    if batch is not None:
        rows = qat_graph.minibatch_indices(
            batch.num_rows, batch.size, seed=batch_seed, shuffle=shuffle
        )
        index_name = batch.index_name

        def batch_rows(t: int) -> Dict[str, np.ndarray]:
            return {index_name: rows(t)}

        feeds = batch_rows

    final = qat_graph.run_step_graph(
        step,
        constants=constants,
        state=state,
        num_steps=num_iterations,
        scalars=scalars,
        providers=step_providers,
        losses=losses,
        feeds=feeds,
    )

    new_codes: Dict[str, np.ndarray] = {}
    new_scales: Dict[str, np.ndarray] = {}
    for t in trained:
        w = final[t.w_input].astype(np.float64)
        scale = (
            final[t.scale_input].astype(np.float64)
            if t.scale_input is not None
            else t.scale_init.astype(np.float64)
        )
        scale_full = np.repeat(scale, t.candidate.block_size, axis=t.scale_axis)
        codes = np.clip(_round_half_away(w / scale_full), _N_MIN, _N_MAX)
        new_codes[t.candidate.wq_name] = codes.astype(np.int8)
        if t.scale_input is not None:
            new_scales[t.candidate.ws_init.name] = scale.astype(np.float32)

    tuned = onnx.ModelProto()
    tuned.CopyFrom(quantized_model)
    for initializer in tuned.graph.initializer:
        codes = new_codes.get(initializer.name)
        if codes is not None:
            initializer.raw_data = _pack_int4(codes)
            continue
        scale_array = new_scales.get(initializer.name)
        if scale_array is not None:
            initializer.CopyFrom(
                onnx.numpy_helper.from_array(scale_array, name=initializer.name)
            )
    return tuned


def apply_qat(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    block_input_name: str,
    block_output_name: str,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 1000,
    learning_rate: float = 1e-4,
    learn_scales: bool = False,
    scale_learning_rate: float = 1e-5,
    lr_decay: bool = True,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    batch_seed: int = 0,
    providers: Optional[Sequence[backend.Provider]] = None,
    step_providers: Optional[Sequence[backend.Provider]] = None,
    losses: Optional[List[float]] = None,
) -> onnx.ModelProto:
    """Fine-tunes one block's INT4 weights against the float model's own
    output for that block -- label-free, teacher-distilled QAT. See this
    module's own docstring for the technique, what it refuses, and what it
    does not claim.

    The block is named the way :func:`onnxsim.apply_brecq` names one, by its
    input and output tensor. Unlike ``apply_brecq``, whatever lies between
    them is fair game as long as :mod:`onnxsim.graph_grad` has a rule for it
    -- an activation, a normalization, a GELU, a Softmax, a residual -- and
    unlike every rounding pass here, the fp32 weights themselves are what
    gets optimized.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path. It is the teacher: its activations at
            ``block_output_name`` are the only target, and its weights seed
            the trained master weights.
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`. Layers quantized by
            any other scheme (or left unquantized) are left untouched, and a
            block containing none of them is an error rather than a no-op.
            Assumes ``quantized_model`` was produced from ``float_model``
            without renaming any MatMul/Gemm node's own output tensor -- true
            of every onnxsim ``quantize_*`` function.
    :param block_input_name: the activation entering the block. The backward
            walk that discovers the block's nodes stops here.
    :param block_output_name: the block's own final output, the tensor whose
            reconstruction error is the loss.
    :param calibration_data: representative input batches. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``float_model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted) and
            :func:`onnxsim.load_huggingface_calibration_data` (real data, a
            far more representative target than random input). All batches
            are concatenated into one full-batch objective, so their shapes
            may differ only in the leading axis.
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_iterations: Adam steps -- optimizer steps -- to run over the
            block. That meaning is unchanged by ``batch_size``: it has always
            been the number of times the parameters are updated, and it still
            is. What ``batch_size`` changes is how much data each of those
            steps sees, and therefore how many *epochs* the same budget buys:
            with ``R`` calibration rows, a run covers
            ``num_iterations * batch_size / R`` epochs (full batch is
            ``batch_size = R``, hence exactly ``num_iterations`` epochs, one
            per step). So halving the batch size at a fixed
            ``num_iterations`` halves the data seen and the compute spent; to
            hold the *epoch* count fixed while minibatching, scale
            ``num_iterations`` up by ``R / batch_size``. ``lr_decay``
            likewise anneals over ``num_iterations`` steps regardless.
    :param learning_rate: Adam learning rate for the fp32 master weights.
            The natural scale to compare it against is the quantization step
            itself: an element has to travel about half a step to change
            which integer it rounds to, so ``num_iterations *
            learning_rate`` well below the typical step size means nothing
            can move at all, and well above it means everything thrashes.
    :param learn_scales: also train each weight's per-block quantization
            scale, LSQ-style. Off by default because it makes the problem
            non-convex in two coupled parameter sets at once (the same
            caution :mod:`onnxsim.autoround` documents) and because it
            rewrites the scale initializers, which the weight-only path
            leaves byte-identical.
    :param scale_learning_rate: Adam learning rate for the scales when
            ``learn_scales`` is on. Smaller than ``learning_rate`` by
            default: one scale is shared by a whole block of weights, so a
            step of the same size is a far larger change to the model.
    :param lr_decay: anneal both learning rates linearly to zero across the
            run. On by default because the objective is piecewise constant in
            the master weights -- the loss only moves when an element crosses
            a rounding boundary -- so a constant learning rate leaves the
            final iterate wherever the last step happened to put it, which
            can be worse than a step earlier. Annealing makes the end of the
            run settle instead. Turn it off to hold the rate fixed.
    :param batch_size: rows of the calibration set each step trains on.
            ``None``, the default, is full batch -- every step sees every
            row, which is what this module has always done and which stays
            bit-for-bit unchanged, ``Gather``-free graph included, because a
            full-batch run does not take the minibatching path at all. A
            ``batch_size`` at least as large as the row count is the same
            thing and takes the same path.

            What minibatching is *for*, stated honestly: a step's cost stops
            scaling with the size of the calibration set, so a larger set
            costs more epochs rather than a bigger, slower step -- and each
            pass over the data now performs ``R / batch_size`` updates
            instead of one, which is the ordinary reason stochastic gradient
            descent converges in fewer passes than full-batch descent. What
            it is *not*: a way to train on more data than fits in memory. The
            whole set is still one static tensor, resident on the execution
            provider's device, out of which each step gathers its rows --
            see :mod:`onnxsim.qat_graph`'s module docstring for that
            trade-off and the alternative that was not taken.

            The loss recorded in ``losses`` becomes the *batch's* loss, not
            the set's, so it is noisy: compare a smoothed tail against a
            smoothed head, not the last value against the first.
    :param shuffle: draw a fresh permutation of the rows each epoch, so
            consecutive steps see different rows rather than the same fixed
            partition every time round. On by default, and only meaningful
            when ``batch_size`` is set. ``False`` walks the rows in order,
            which is for reproducing a specific batch composition (a test, a
            debugging session) rather than for training.
    :param batch_seed: seed for that shuffling. Deliberately its own
            parameter rather than a second use of ``seed``: ``seed`` picks
            the random *calibration data* and is documented as ignored when
            the caller supplies their own, whereas the batch order matters in
            exactly the case where the caller did supply data. Two runs with
            the same ``batch_seed`` see identical batches in identical order.
            Batches are a pure function of ``(batch_seed, step index)``, not
            of a running generator, so an interrupted run resumes on the same
            schedule.
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing the teacher's activations
    :param step_providers: onnxruntime execution providers to run the
            optimization itself on, as an ONNX step graph
            (:mod:`onnxsim.qat_graph`) -- the way to reach a GPU, an NPU
            execution provider, or WebGPU with this loop. ``None`` means CPU.
            Unlike :func:`onnxsim.apply_adaround`, there is no host-numpy
            alternative here: the step graph *is* the implementation, so this
            selects where it runs rather than whether it is used.
    :param losses: when given, the reconstruction loss is appended to it once
            per step -- the cheapest way to see whether a block actually
            trained, and what the tests here assert on.
    :returns: ``quantized_model`` with the block's INT4 weight initializers
            (and, if ``learn_scales``, their scale initializers) rewritten.
            Every other byte of the model is untouched.
    :raises ValueError: if the block cannot be discovered, is not closed at
            statically-known shapes, contains no quantized layer to train, or
            (with ``batch_size`` set) has captured tensors that disagree
            about how many rows they have
    :raises onnxsim.graph_grad.UnsupportedOpError: if any node in the block
            has no gradient rule
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)

    plan = _plan_block(
        float_model, quantized_model, block_input_name, block_output_name
    )

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    captured = _capture(
        float_model,
        sorted(set(plan.externals) | {plan.output_name}),
        calibration_data,
        providers,
    )
    return _train_block(
        float_model,
        quantized_model,
        plan,
        {name: captured[name] for name in plan.externals},
        captured[plan.output_name],
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        learn_scales=learn_scales,
        scale_learning_rate=scale_learning_rate,
        lr_decay=lr_decay,
        batch_size=batch_size,
        shuffle=shuffle,
        batch_seed=batch_seed,
        step_providers=step_providers,
        losses=losses,
    )


@dataclass(frozen=True)
class QATBlock:
    """One trainable block :func:`discover_qat_blocks` found, named the way
    :func:`apply_qat` names one.

    ``input_name``/``output_name`` are exactly what a caller would have passed
    to :func:`apply_qat` by hand, so a plan is inspectable, diffable and
    replayable one block at a time. The rest is metadata about what the
    boundary pair actually resolved to -- useful for deciding whether the plan
    is the one you wanted before spending a training budget on it.
    """

    input_name: str
    output_name: str
    #: Output tensor of every ``quantize_weight_only_int4``-quantized
    #: MatMul/Gemm inside the block, in graph order. Never empty: a slice with
    #: nothing to train is not a block.
    quantized_outputs: Tuple[str, ...]
    #: Tensors the block reads but does not produce, ``input_name`` included.
    #: These are teacher-forced -- see :func:`_slice_block`.
    external_inputs: Tuple[str, ...]
    #: Op types inside the block, deduplicated and sorted. Every one of them
    #: is in :data:`onnxsim.graph_grad.SUPPORTED_OPS`, by construction.
    op_types: Tuple[str, ...]
    num_nodes: int


@dataclass
class QATBlockResult:
    """What happened to one block during :func:`apply_qat_all_blocks`.

    ``trained`` and ``skipped_reason`` are mutually exclusive: a block either
    trained (and ``losses`` has one entry per Adam step) or was skipped with a
    reason string. A skip is never silent and never fatal -- see
    :func:`apply_qat_all_blocks` for why that is the right trade here and
    the opposite of :func:`apply_qat`'s own loud refusal.
    """

    block: QATBlock
    trained: bool
    skipped_reason: Optional[str] = None
    losses: List[float] = field(default_factory=list)

    @property
    def initial_loss(self) -> Optional[float]:
        """The block's reconstruction error before the first Adam step, i.e.
        at round-to-nearest. ``None`` if the block was skipped."""
        return self.losses[0] if self.losses else None

    @property
    def final_loss(self) -> Optional[float]:
        """The block's reconstruction error after the last step. Compare it
        against :attr:`initial_loss` -- the *ratio* is the only meaningful
        number, since blocks differ in output scale and in element count."""
        return self.losses[-1] if self.losses else None


def _liveness_cuts(
    graph: onnx.GraphProto, primary_input: Optional[str]
) -> List[Tuple[int, str]]:
    """Every index at which the graph narrows to a single live activation,
    with the tensor that survives it.

    This is the whole of boundary discovery, and it is a liveness argument
    rather than a pattern match. Walk the nodes in their (topological) graph
    order and track which tensors are *live* at each gap between node ``p``
    and node ``p + 1``: produced at or before ``p``, and still read after it
    (a graph output counts as read by the outside world). A gap where exactly
    one tensor is live is a place the graph can be cut without severing
    anything else, so the slice on either side is self-contained -- which is
    precisely the property :func:`_slice_block` needs its two boundaries to
    have.

    The pleasant consequence is that residual connections *place* the
    boundaries instead of defeating them. Inside ``y = f(x) + x`` the skip
    tensor ``x`` is live alongside every intermediate, so no gap in the middle
    of the residual is a cut, and the first cut after ``x`` is the residual
    ``Add``'s own output -- exactly where a person would have drawn the block
    boundary of a ResNet BasicBlock or a transformer sub-layer, derived rather
    than special-cased.

    Two things are excluded from the live set:

    - **Initializers.** They are not activations; every block gets its own
      copy in its step graph.
    - **Graph inputs other than** ``primary_input``. A second graph input --
      an attention mask, a position id tensor -- is byte-identical in the
      teacher and the student, so teacher-forcing it into a block is exact
      rather than an approximation, and letting it span the whole graph would
      otherwise suppress every cut in a model that has one.

    ``primary_input`` itself is *kept* in the live set, so a residual from the
    model's own input still binds a block together (and its block ends at the
    residual's output, not before it).

    What this cannot see: a tensor computed purely from initializers -- a
    pre-transposed weight shared by several layers, say -- is counted as an
    ordinary live activation, so it suppresses cuts across its whole live
    range. That is conservative in the safe direction (fewer, larger blocks,
    or none) rather than the unsafe one.
    """
    initializers = {t.name for t in graph.initializer}
    graph_inputs = {i.name for i in graph.input if i.name not in initializers}
    ignored = {name for name in graph_inputs if name != primary_input}

    # A tensor is live until its last consumer; a graph output is live past
    # the end of the graph, so it is never dropped before the final gap.
    last_use: Dict[str, int] = {}
    for index, node in enumerate(graph.node):
        for name in node.input:
            if name and name not in initializers and name not in ignored:
                last_use[name] = index
    for out in graph.output:
        if out.name and out.name not in initializers and out.name not in ignored:
            last_use[out.name] = len(graph.node)

    cuts: List[Tuple[int, str]] = []
    live: Set[str] = {
        name
        for name in graph_inputs
        if name not in ignored and last_use.get(name, -1) > -1
    }
    if len(live) == 1:
        cuts.append((-1, next(iter(live))))
    for index, node in enumerate(graph.node):
        for name in node.output:
            if name and name not in ignored and last_use.get(name, -1) > index:
                live.add(name)
        live = {name for name in live if last_use.get(name, -1) > index}
        if len(live) == 1:
            cuts.append((index, next(iter(live))))
    return cuts


def _primary_graph_input(graph: onnx.GraphProto) -> Optional[str]:
    """The graph input the most nodes depend on -- the main activation path.

    A heuristic, and named as one. Models with several inputs almost always
    have one carrying the activations and the others carrying masks or ids,
    and "reaches the most nodes" separates those reliably in practice while
    being independent of naming conventions. Ties go to the earlier graph
    input. It only decides which input keeps its power to *prevent* a cut
    (see :func:`_liveness_cuts`); getting it wrong costs block granularity,
    not correctness, because every non-chosen input is teacher-forced exactly.
    """
    initializers = {t.name for t in graph.initializer}
    candidates = [i.name for i in graph.input if i.name not in initializers]
    if not candidates:
        return None

    best_name, best_reach = candidates[0], -1
    for name in candidates:
        reached = {name}
        count = 0
        for node in graph.node:
            if any(inp in reached for inp in node.input if inp):
                count += 1
                reached.update(out for out in node.output if out)
        if count > best_reach:
            best_name, best_reach = name, count
    return best_name


def discover_qat_blocks(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    max_layers_per_block: int = 2,
) -> List[QATBlock]:
    """Partitions the model into a sequence of blocks :func:`apply_qat` can
    train, without the caller naming a single tensor.

    This is the piece ``docs/qat.md`` lists as missing and
    :mod:`onnxsim.brecq`'s docstring explicitly declines to attempt ("the
    caller identifies a block by two tensor names"). BRECQ's reason was that
    auto-detection looked architecture-specific; it is not, once the question
    is asked in the right terms. Two properties make a slice trainable, and
    both are decidable from the graph:

    1. **Differentiable.** Every op inside must be in
       :data:`onnxsim.graph_grad.SUPPORTED_OPS`, since the step graph
       contains a backward pass over the block's own nodes.
    2. **Self-contained.** The block must be cuttable out of the graph
       without severing an activation that some other part of the graph is
       still using. :func:`_liveness_cuts` finds exactly those places, by
       liveness rather than by recognizing architectures -- read its
       docstring, it is the substance of this function.

    Blocks are then the spans between consecutive cuts, with two adjustments:

    - **Unsupported ops become gaps, not failures.** A span containing an op
      with no gradient rule cannot be a block, so it is skipped and the next
      block starts after it. A model with one ``Sin`` in the middle trains
      everything either side of it instead of being refused outright, which
      is the behaviour that makes whole-model QAT usable at all; the
      alternative -- :func:`apply_qat`'s loud refusal -- is right for a
      caller who *named* a block and wrong for a caller who named none.
    - **Spans are merged up to** ``max_layers_per_block`` **quantized
      layers.** A cut exists between every pair of layers in a plain MLP, so
      without merging every block would be a single layer and the whole point
      of block-wise reconstruction (letting layers inside a block cancel each
      other's error, :mod:`onnxsim.brecq`'s own argument) would be lost. The
      default of 2 is the paired-projection shape BRECQ's Section 4 targets
      -- a ResNet BasicBlock's two convolutions, a transformer FFN's up/down
      pair. A span with no quantized layer in it (a lone activation) never
      closes a block; it is absorbed into the next one.

    **What discovery cannot see, and therefore does not promise.** It never
    runs the model, so it cannot know whether a block's shapes are statically
    inferable -- a dynamic ``Reshape``, a symbolic dimension that survives
    inference -- and a block that fails on that is discovered here and
    skipped later, by :func:`apply_qat_all_blocks`, with the reason recorded.
    It also has no notion of which blocks *matter*: it will happily propose a
    block whose quantization error is already negligible, and it has no
    sensitivity metric to rank them (:mod:`onnxsim.precision_estimator` is
    where such a thing would come from). And it inherits every limit of
    :func:`_liveness_cuts`: a graph with a long-lived constant-derived tensor,
    or with multi-output branches that never reconverge, simply yields fewer
    or no cuts, and therefore fewer or no blocks -- an empty plan, not an
    error.

    :param float_model: the teacher, as an onnx ModelProto or a file path.
            Boundaries are found in *this* graph, since it is the one whose
            nodes the step graph differentiates.
    :param quantized_model: its :func:`onnxsim.quantize_weight_only_int4`
            counterpart, used only to find which layers are actually
            quantized -- a block must contain at least one.
    :param max_layers_per_block: how many quantized layers to merge into one
            block before closing it. 1 gives per-layer blocks (more, cheaper
            steps, no intra-block error cancellation); a large value gives
            one block per gap between undifferentiable ops.
    :returns: the blocks in graph order, possibly empty. Consecutive blocks
            need not be adjacent: a gap between two of them is a region
            nothing here can train.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if max_layers_per_block < 1:
        raise ValueError("max_layers_per_block must be at least 1")

    graph = float_model.graph
    cuts = _liveness_cuts(graph, _primary_graph_input(graph))
    quantized_outputs = {
        c.output_name
        for c in _find_int4_matmul_candidates(float_model, quantized_model)
    }

    # Walk the spans between consecutive cuts, accumulating them into blocks.
    # ``start`` is the cut the pending block opens at; ``layers`` counts the
    # quantized layers accumulated since then.
    pairs: List[Tuple[str, str]] = []
    start: Optional[Tuple[int, str]] = cuts[0] if cuts else None
    layers = 0
    for previous, current in zip(cuts, cuts[1:]):
        span = graph.node[previous[0] + 1 : current[0] + 1]
        if any(node.op_type not in graph_grad.SUPPORTED_OPS for node in span):
            # A gap. Close whatever was pending *before* it (the pending
            # block ends at the last cut that is still on the trainable side)
            # and reopen after it.
            if start is not None and layers and start[0] < previous[0]:
                pairs.append((start[1], previous[1]))
            start, layers = current, 0
            continue
        if start is None:
            start = previous
        layers += sum(
            1 for node in span for out in node.output if out in quantized_outputs
        )
        if layers >= max_layers_per_block:
            pairs.append((start[1], current[1]))
            start, layers = current, 0
    if start is not None and layers and cuts and start[0] < cuts[-1][0]:
        pairs.append((start[1], cuts[-1][1]))

    blocks: List[QATBlock] = []
    for input_name, output_name in pairs:
        try:
            plan = _plan_block(float_model, quantized_model, input_name, output_name)
        except ValueError:
            # Defensive: the span construction above already guarantees a
            # non-empty, supported, quantized slice. Rather than trust that
            # invariant, the plan itself is the check -- and a boundary pair
            # that somehow fails it is dropped rather than handed to a caller
            # who would only fail on it later.
            continue
        blocks.append(
            QATBlock(
                input_name=input_name,
                output_name=output_name,
                quantized_outputs=tuple(c.output_name for c in plan.candidates),
                external_inputs=tuple(plan.externals),
                op_types=tuple(sorted({n.op_type for n in plan.nodes})),
                num_nodes=len(plan.nodes),
            )
        )
    return blocks


def _capture_student_inputs(
    student: onnx.ModelProto,
    names: Sequence[str],
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[backend.Provider]],
    fallback: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """The student's own activations at ``names``, falling back to the
    teacher's for any name the student's graph does not have.

    ``quantize_weight_only_int4`` never renames a node's output, so an
    activation named in the float graph is named identically in the quantized
    one and this fallback is normally unused. It exists for the one case
    :func:`_slice_block` can produce that is not an activation: a tensor
    computed entirely from initializers, which a quantizer is free to fold or
    rewrite. Capturing the teacher's value for such a tensor is exact anyway.
    """
    present = {i.name for i in student.graph.input}
    present.update(out for node in student.graph.node for out in node.output if out)
    wanted = [name for name in names if name in present]
    captured = dict(fallback)
    if wanted:
        captured.update(_capture(student, wanted, calibration_data, providers))
    return {name: captured[name] for name in names}


def apply_qat_all_blocks(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    blocks: Optional[Sequence[QATBlock]] = None,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 1000,
    learning_rate: float = 1e-4,
    learn_scales: bool = False,
    scale_learning_rate: float = 1e-5,
    lr_decay: bool = True,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    batch_seed: int = 0,
    sequential: bool = True,
    max_layers_per_block: int = 2,
    providers: Optional[Sequence[backend.Provider]] = None,
    step_providers: Optional[Sequence[backend.Provider]] = None,
) -> Tuple[onnx.ModelProto, List[QATBlockResult]]:
    """Trains every block :func:`discover_qat_blocks` finds, in graph order --
    :func:`apply_qat` lifted from one caller-named block to the whole model.

    **The design decision that matters: where each block's input comes
    from.** Every block's optimization *target* is the teacher's output for
    that block; that is not in question and is what makes this label-free.
    The question is what to feed the block's input, and there are two honest
    answers:

    - ``sequential=True`` (**the default**): re-run the *student* -- the
      quantized model as tuned so far -- before each block, and feed that
      block the activation the deployed model will actually present to it.
      Block *k* therefore sees the error blocks 0..k-1 left behind and spends
      its own capacity correcting it, while still aiming at the teacher's
      clean output. This is the standard sequential block-reconstruction
      setup, and it is the reason the walk is worth more than *N* independent
      calls to :func:`apply_qat`. It costs one forward pass of the student
      per block -- inference, not training, and negligible beside
      ``num_iterations`` optimizer steps.
    - ``sequential=False``: capture everything once, from the float model,
      before any block is touched. This is what :mod:`onnxsim.adaround` and
      :mod:`onnxsim.brecq` do, and it is cheaper by exactly one forward pass
      per block. It is also strictly the *independence assumption*
      :mod:`onnxsim.brecq`'s own docstring identifies as the flaw in
      per-layer reconstruction, applied one level up: it assumes every
      earlier block was reconstructed perfectly, so a later block optimizes
      against an input the deployed model never produces.

    Sequential is the default because the assumption it drops is known to be
    false -- quantization error compounds down a network, that is the entire
    premise of block reconstruction. It is not, however, a free win:
    ``tests/test_qat.py`` measures both modes on a deliberately
    error-compounding model and records which one actually won, rather than
    asserting the expected direction. On a shallow model with small
    quantization error the two modes land within noise of each other, and on
    any model the sequential input is *noisier* -- the student's activation
    carries the earlier blocks' residual error, which acts a little like
    input jitter. That is usually a regularizer and occasionally a handicap.

    **Failures are per block, not per model.** A block that cannot be trained
    -- an op with no gradient rule that discovery could not have foreseen, a
    shape that will not infer statically, a slice with no quantized layer --
    is skipped, the reason is recorded in its :class:`QATBlockResult`, and
    the walk continues. That is the opposite of :func:`apply_qat`, which
    refuses loudly, and the difference is deliberate: refusing loudly is
    right when the caller *named* the thing that cannot be trained, and wrong
    when they named nothing and one block out of forty is unusual. Nothing is
    dropped silently -- every discovered block appears in the returned list,
    trained or not.

    :param float_model: the teacher (onnx ModelProto or file path). Its
            activations are every block's target and its weights seed every
            block's trained master weights.
    :param quantized_model: its :func:`onnxsim.quantize_weight_only_int4`
            counterpart -- the student, and the model that is returned with
            its INT4 initializers rewritten.
    :param blocks: the plan to walk. ``None`` runs :func:`discover_qat_blocks`
            with ``max_layers_per_block``. Pass an explicit list to inspect,
            filter or reorder the plan first -- e.g. to train only the blocks
            a sensitivity analysis flagged.
    :param calibration_data: representative input batches, as
            :func:`apply_qat` takes them. All batches are concatenated into
            one full-batch objective per block.
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for that random calibration data
    :param num_iterations: Adam (optimizer) steps per block, exactly as in
            :func:`apply_qat` -- see there for how ``batch_size`` relates
            steps to epochs. The total budget is this times the number of
            blocks, so a whole-model walk usually wants a smaller value than
            a single :func:`apply_qat` call would.
    :param learning_rate: Adam learning rate for the fp32 master weights
    :param learn_scales: also train each weight's per-block quantization
            scale, LSQ-style, in every block
    :param scale_learning_rate: Adam learning rate for those scales
    :param lr_decay: anneal both learning rates to zero within each block
    :param batch_size: rows per optimizer step, applied identically in every
            block -- ``None`` (the default) is full batch, unchanged. Note
            that the blocks share the *same* batch schedule, since each is
            trained by its own :func:`apply_qat`-equivalent loop starting from
            step 0 with the same ``batch_seed``; the rows are the same rows,
            because every block's activations were captured from the same
            calibration inputs in the same order, so block ``k`` and block
            ``k+1`` agree about what "row 7" means.
    :param shuffle: shuffle the rows per epoch, as :func:`apply_qat` does
    :param batch_seed: seed for that shuffling
    :param sequential: feed each block the student's own activation rather
            than the teacher's -- see above. ``True`` by default.
    :param max_layers_per_block: passed to :func:`discover_qat_blocks` when
            ``blocks`` is not given
    :param providers: execution providers for the activation captures (both
            the teacher's and, in sequential mode, the student's)
    :param step_providers: execution providers for the optimization itself,
            as an ONNX step graph -- the path to CUDA, an NPU EP or WebGPU
    :returns: ``(tuned model, one QATBlockResult per discovered block)``. The
            model is ``quantized_model`` with every successfully trained
            block's initializers rewritten and nothing else touched; if no
            block trained it is an unmodified copy.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)

    if blocks is None:
        blocks = discover_qat_blocks(
            float_model, quantized_model, max_layers_per_block=max_layers_per_block
        )
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    # Plans are built once, against the *original* quantized model. Training a
    # block rewrites initializer payloads and never the graph, so a plan --
    # which is nodes, tensor names and candidate metadata -- stays valid for
    # the whole walk. Master weights are seeded from the float model in every
    # case (see :func:`_plan_trained`), so no plan depends on the tuned state.
    plans: List[Optional[_BlockPlan]] = []
    results: List[QATBlockResult] = []
    for block in blocks:
        try:
            plans.append(
                _plan_block(
                    float_model, quantized_model, block.input_name, block.output_name
                )
            )
            results.append(QATBlockResult(block=block, trained=False))
        except ValueError as error:
            plans.append(None)
            results.append(
                QATBlockResult(
                    block=block, trained=False, skipped_reason=f"cannot plan: {error}"
                )
            )

    # One teacher pass for the whole walk: every block's target, and (in
    # capture-once mode) every block's input too. The teacher never changes,
    # so re-running it per block would buy nothing.
    wanted: Set[str] = set()
    for plan in plans:
        if plan is not None:
            wanted.update(plan.externals)
            wanted.add(plan.output_name)
    teacher = (
        _capture(float_model, sorted(wanted), calibration_data, providers)
        if wanted
        else {}
    )

    tuned = onnx.ModelProto()
    tuned.CopyFrom(quantized_model)
    for plan, result in zip(plans, results):
        if plan is None:
            continue
        if sequential:
            # The one extra forward pass this mode costs. It has to happen
            # here, not once up front, because ``tuned`` has changed since the
            # previous block: that is the entire point.
            inputs = _capture_student_inputs(
                tuned, plan.externals, calibration_data, providers, teacher
            )
        else:
            inputs = {name: teacher[name] for name in plan.externals}
        try:
            tuned = _train_block(
                float_model,
                tuned,
                plan,
                inputs,
                teacher[plan.output_name],
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                learn_scales=learn_scales,
                scale_learning_rate=scale_learning_rate,
                lr_decay=lr_decay,
                batch_size=batch_size,
                shuffle=shuffle,
                batch_seed=batch_seed,
                step_providers=step_providers,
                losses=result.losses,
            )
        except (ValueError, graph_grad.UnsupportedOpError) as error:
            # A step graph that was half-built cannot have touched ``tuned``
            # -- _train_block only rewrites initializers on a fresh copy, as
            # its very last act -- so the walk resumes from an intact model.
            result.losses.clear()
            result.skipped_reason = f"training failed: {error}"
            continue
        result.trained = True
    return tuned, results
