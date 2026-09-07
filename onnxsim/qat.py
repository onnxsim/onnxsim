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
the WASM build -- and it stays inside
:data:`onnxsim.qat_graph.EP_FRIENDLY_OPS` to make that real rather than
nominal (no ``Round``:
:meth:`onnxsim.qat_graph.GraphBuilder.round_to_nearest` composes
one out of ``Sign``/``Abs``/``Cast``, and this reuses it).

**What this deliberately is not, and does not claim.**

- *Not task-loss QAT.* There are no labels, no dataset API, no metric and no
  training lifecycle -- ``docs/qat.md``'s deliverable C, unchanged and still
  out of scope. The teacher's own activations are the only target, so the
  ceiling is "reproduce the float block", not "recover task accuracy the
  float block never had". Label-free distillation QAT is not paper-QAT
  accuracy and should not be advertised as it.
- *Not whole-model training.* One caller-named block per call, exactly
  :mod:`onnxsim.brecq`'s contract (``block_input_name`` /
  ``block_output_name``). A sliding window over blocks, and an end-to-end
  pass afterwards, are the caller's loop to write.
- *Not activation quantization.* This targets
  :func:`onnxsim.quantize_weight_only_int4`'s weight-only scheme, the same
  one AdaRound/BRECQ/FOEM target. Learnable activation scales exist in-tree
  (:mod:`onnxsim.adaquant`) but are not wired in here.
- *Full-batch gradient descent.* The whole calibration set is one static
  tensor baked into the step graph's shapes, as in every other
  reconstruction pass here -- no minibatching, no epochs. That is a
  calibration-scale budget, not a training-scale one.
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
everywhere: a caller who names a block this cannot train gets a
:class:`ValueError`, never a silently unchanged model. The slice is the
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

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

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
) -> qat_graph.StepGraph:
    """The whole loop as one graph: fake-quant forward, block forward,
    reconstruction loss, backward, Adam.

    The ordering is the only subtle part. :func:`graph_grad.build_backward`
    reads forward tensors by name (including node *outputs*, where reusing a
    ``Sigmoid``/``Softmax`` result is cheaper than recomputing it), so every
    node it differentiates must already sit in the builder ahead of the nodes
    it appends. Hence: fake-quant, then the block's own nodes verbatim, then
    the loss seed, then the backward, then the optimizer.
    """
    b = qat_graph.GraphBuilder(_PREFIX)
    b.initializer.extend(block_initializers)

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
    teacher = f"{_PREFIX}teacher"
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

    constants: Dict[str, Sequence[int]] = {
        name: list(value.shape) for name, value in sorted(externals.items())
    }
    constants[teacher] = list(block_output_shape)

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

    return qat_graph.make_step_graph(
        b,
        constants=constants,
        state=state,
        scalars=scalars,
        loss=b.mean_square(diff),
        name="onnxsim_qat_step",
    )


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
    :param num_iterations: Adam steps to run over the block
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
            statically-known shapes, or contains no quantized layer to train
    :raises onnxsim.graph_grad.UnsupportedOpError: if any node in the block
            has no gradient rule
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)

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

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    captured = _capture(
        float_model,
        sorted(set(externals) | {block_output_name}),
        calibration_data,
        providers,
    )
    teacher_output = captured[block_output_name]
    external_values = {name: captured[name] for name in externals}

    shapes = _block_shapes(
        float_model, nodes, external_values, block_output_name, teacher_output
    )

    trained = _plan_trained(candidates, learn_scales)
    trained_weight_names = {t.candidate.float_node.input[1] for t in trained}
    used = {name for node in nodes for name in node.input if name}
    block_initializers = [
        t
        for t in float_model.graph.initializer
        if t.name in used and t.name not in trained_weight_names
    ]

    step = _build_step_graph(
        trained,
        nodes,
        shapes,
        block_initializers,
        external_values,
        block_output_name,
        list(teacher_output.shape),
        learn_scales,
    )

    constants: Dict[str, np.ndarray] = dict(external_values)
    constants[f"{_PREFIX}teacher"] = teacher_output
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

    final = qat_graph.run_step_graph(
        step,
        constants=constants,
        state=state,
        num_steps=num_iterations,
        scalars=scalars,
        providers=step_providers,
        losses=losses,
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
