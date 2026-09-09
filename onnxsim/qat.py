"""Label-free, block-wise quantization-aware fine-tuning -- ``docs/qat.md``'s
deliverable B, the "knowledge-distillation QAT" that stage 2 of that note
describes.

The same machinery also runs with no quantizer in it at all
(``fake_quant=False``): :func:`apply_block_finetune` and
:func:`apply_block_finetune_all_blocks` train a model's *own float* weights
against a reference model's activations -- ordinary block-wise, label-free
fine-tuning for a model something else already changed. Everything below
applies to it unchanged except the fake-quantizer, which is the only
quantization-specific part of any of it.

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
- *Block-wise by default; end-to-end is reachable and measured worse.*
  :func:`apply_qat` trains one caller-named block per call, exactly
  :mod:`onnxsim.brecq`'s contract (``block_input_name`` /
  ``block_output_name``), and :func:`apply_qat_all_blocks` walks a whole
  model one block at a time -- discovering the blocks with
  :func:`discover_qat_blocks` and, by default, feeding each one the
  student's own activation so it corrects what its predecessors left
  behind.

  ``docs/qat.md`` lists "an optional end-to-end pass on the whole graph" as
  this stage's last open item, on the reading that a block is necessarily
  smaller than the model. It is not: **the whole graph is a legal block.**
  Naming the graph's own input and its own output -- ``apply_qat(float,
  quantized, "X", "Yout")``, or :func:`discover_qat_blocks` with
  ``max_layers_per_block`` above the model's layer count -- builds one step
  graph over every node, trains every quantized layer's master weight
  jointly inside it, and takes the loss against the float model's own final
  output. That is the end-to-end objective exactly, and it is the one slice
  with no teacher-forcing approximation left in it at all: a whole-graph
  slice's only external tensors are the graph's own inputs, whose values are
  identical in teacher and student by construction.

  So there was nothing here to build, and the only question was whether to
  recommend it. Measured against :func:`apply_qat_all_blocks` at an equal
  optimizer-step budget, on residual MLP stacks every one of whose ops has a
  gradient rule (``tests/test_qat.py``), it lands the same way every time:
  end-to-end reaches a **lower** error on the calibration set -- 12-26%
  lower on an eight-stage stack, unsurprisingly, since that is literally the
  quantity it minimizes and the block-wise walk only approximates it -- and
  a **higher** error on held-out inputs, 1-12% higher on the same model, on
  all eight seeds tried. It is buying calibration-set fit that does not
  transfer, which is :mod:`onnxsim.brecq`'s own argument for why the block
  is the right unit: requiring every intermediate activation to match the
  teacher is a far stronger constraint than requiring only the final output
  to, and at calibration scale that constraint is worth more than the
  freedom. Sixty-four times the data narrows the held-out gap from ~15% to
  ~9% without closing it, and by that budget end-to-end has stopped winning
  even on its own objective -- one global loss over sixteen coupled
  parameter groups is a harder problem than sixteen conditioned ones.

  It is also the expensive direction, in the way that decides whether a run
  fits at all. One end-to-end step touches every layer rather than one
  block's, so at an equal step count it costs roughly the block count more
  time (6x on an eight-block model, measured), and its working set is the
  whole model's rather than one block's: three fp32 copies of every trained
  weight (the master weight and Adam's two moments) plus every forward *and*
  backward intermediate at the calibration batch size, all resident
  simultaneously. Measured peak RSS on
  a one-million-parameter stack: 105 MB against the walk's 31 MB -- and the
  walk's is flat in depth where this one is linear in it.

  And it is all-or-nothing on operator coverage where the walk is not: one
  node without a gradient rule refuses the entire model, whereas
  :func:`apply_qat_all_blocks` turns that node into a gap and trains
  everything either side of it. :data:`onnxsim.graph_grad.SUPPORTED_OPS`
  covers 22 of the 202 operators in ONNX's default domain, so on a real
  model that is the binding constraint long before the accuracy question
  above is reached.

  Reach for it on a shallow model, where the two land within a few percent
  of each other in both directions. Do not reach for it on a deep one: it
  will report a better loss while shipping a worse model, which is why the
  loss ``losses=`` records has never been the number ``tests/test_qat.py``
  asserts on.
- *Weight-only by default; activation quantization is opt-in and changes
  the target scheme.* With ``learn_activation_scales=False`` (the default)
  this targets :func:`onnxsim.quantize_weight_only_int4`'s weight-only
  scheme, the same one AdaRound/BRECQ/FOEM target, and there is no
  activation quantizer anywhere in that model to train. With
  ``learn_activation_scales=True`` it targets
  :func:`onnxsim.quantize_static`'s QDQ scheme instead -- uint8 affine
  activations, per-output-channel symmetric INT8 weights -- and trains the
  weights *and* the activation quantizers of that scheme jointly. See
  :func:`apply_qat`'s own docstring for why one flag necessarily selects a
  scheme rather than adding a feature to the other one, and for what is
  refused. The activation gradients are :mod:`onnxsim.adaquant`'s, not a
  second derivation of them.
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

  The same standard applied to ``learn_activation_scales``, and it lands in
  the same shape: on a block whose activation range was calibrated from one
  unrepresentative outlier -- ~30x too wide, so activation quantization is
  the binding constraint -- training the quantizers alongside the weights
  takes the whole-model output error from 4.08 (weights alone) to 2.54, 38%
  better, on all three seeds tried. On the *same* block calibrated on
  representative data it does not help at all: 1.23 weights alone against
  1.23 joint at the default learning rate and 1.27 at 1e-1 -- a regression
  at both rates and on all three seeds. Min/max on
  representative data is already close to MSE-optimal, so there is little
  left for a learned clip range to find, and a second coupled parameter
  group makes a solved problem harder. This is a fix for a quantizer whose
  range is wrong, not a free improvement on one whose range is right.

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
contains no layer of the *targeted scheme* to train (which scheme that is
depends on ``learn_activation_scales``, so the same block can be trainable
under one and refused under the other -- see :func:`apply_qat`), or if the
block's shapes cannot be inferred statically at opset 17.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper
import onnx.shape_inference

from onnxsim import backend, graph_grad, qat_graph
from onnxsim.adaquant import _find_static_qdq_candidates
from onnxsim.adaround import _find_int4_matmul_candidates, _pack_int4
from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data

# quantize_weight_only_int4's symmetric INT4 range, same constants
# onnxsim.brecq pins for the same scheme.
_N_MIN = -7.0
_N_MAX = 7.0

# quantize_static's weight range: per-output-channel symmetric INT8, the same
# grid onnxsim.adaquant already optimizes rounding within.
_INT8_N_MIN = -127.0
_INT8_N_MAX = 127.0

# ...and its activation range: uint8, asymmetric (a learned zero-point), so
# the quantizer this module trains is exactly the (scale, zero_point) pair
# onnxsim.adaquant trains -- same bounds, same straight-through derivation.
_ACT_N_MIN = 0.0
_ACT_N_MAX = 255.0

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


@dataclass(frozen=True)
class _ActQuant:
    # Raw string: the Sphinx escaped space in ``:class:`_ActQuant`\ s`` below is
    # an invalid escape sequence to Python, which 3.12 reports as a
    # SyntaxWarning on import of the shipped package.
    r"""One activation quantizer this module may train: where it sits, which
    initializers it is stored in, and what calibration left it at.

    It describes an *edge*, not a tensor, and that is the whole of the
    "where does the quantizer sit" question. :func:`onnxsim.quantize_static`
    inserts one ``QuantizeLinear``/``DequantizeLinear`` pair per quantized
    node, with its own scale and zero-point initializers, even when two nodes
    read the same activation -- so a tensor feeding two MatMuls carries two
    independent quantizers and gets two independent :class:`_ActQuant`\ s.
    Training one per *tensor* would be a different (and lossier) model than
    the one that ships.
    """

    #: The activation the quantizer reads -- the layer's own ``input[0]``.
    tensor: str
    scale_name: str
    zp_name: str
    scale_init: float
    zp_init: float


@dataclass(frozen=True)
class _QuantizedLayer:
    """One quantized MatMul/Gemm this module can train, with the two schemes'
    differences already normalized away.

    Both schemes are "an integer code array plus a scale that tiles the
    weight", and everything downstream -- the fake-quant forward, the
    blocked-scale reshapes, the LSQ scale gradient, the export -- is written
    against that shape rather than against either scheme. The normalization
    is entirely in :func:`_from_int4` and :func:`_from_static`:

    - ``quantize_weight_only_int4``: 32-element blocks along the reduction
      axis, codes in ``[-7, 7]``, scale stored 2-D, no activation quantizer.
    - ``quantize_static``: one scale per output channel, codes in
      ``[-127, 127]``, scale stored 1-D. A per-channel scale *is* a
      block-wise scale whose block spans the whole reduction axis, so it maps
      onto the same rank-3 reshape with ``block_size`` = that axis's length
      and a scale of shape ``[1, N]`` (or ``[N, 1]`` for a ``transB`` Gemm's
      ``[N, K]`` weight). No special case is needed anywhere below.
    - **no quantization at all**: :func:`_from_float`, which is what turns
      this module into a plain fine-tuner. The layer's weight is trained as
      itself and written back as itself; there is no code array, no scale and
      no activation quantizer. See :attr:`fake_quant`.
    """

    output_name: str
    float_node: onnx.NodeProto
    w_float_init: onnx.TensorProto
    wq_name: str
    ws_name: str
    #: The scale's shape as the *model* stores it, for writing it back.
    ws_dims: Tuple[int, ...]
    #: The same scale in the normalized 2-D blocked view.
    scale_2d: np.ndarray
    axis: int
    block_size: int
    n_min: float
    n_max: float
    #: INT4 codes are packed two to a byte on export; INT8 codes are not.
    packed_int4: bool
    act: Optional[_ActQuant]
    #: Whether the step graph fake-quantizes this layer's weight on the way
    #: into the block. ``False`` is :func:`_from_float`'s third scheme: the
    #: block reads the master weight directly, the straight-through estimator
    #: has nothing to pass through, and the write-back stores fp32 rather than
    #: codes. Every field above describing the quantizer is then *unread* --
    #: and carries a value that would fail loudly rather than quietly if some
    #: future caller read one anyway (see :func:`_from_float`).
    fake_quant: bool = True


def _from_int4(candidate) -> _QuantizedLayer:
    """A :func:`onnxsim.quantize_weight_only_int4` layer, unchanged in
    substance from what this module trained before ``_QuantizedLayer``
    existed."""
    scale = onnx.numpy_helper.to_array(candidate.ws_init).astype(np.float32)
    return _QuantizedLayer(
        output_name=candidate.output_name,
        float_node=candidate.float_node,
        w_float_init=candidate.w_float_init,
        wq_name=candidate.wq_name,
        ws_name=candidate.ws_init.name,
        ws_dims=tuple(int(d) for d in candidate.ws_init.dims),
        scale_2d=scale,
        axis=candidate.axis,
        block_size=candidate.block_size,
        n_min=_N_MIN,
        n_max=_N_MAX,
        packed_int4=True,
        act=None,
    )


def _from_static(
    candidate, quantized_model: onnx.ModelProto
) -> Optional[_QuantizedLayer]:
    """A :func:`onnxsim.quantize_static` QDQ layer, or ``None`` if its
    initializers are not the shape that scheme produces.

    The ``None`` cases are the ones a hand-edited or third-party model can
    reach: a per-channel weight scale that is not 1-D of the output channel's
    length, or an activation scale/zero-point that is not a single value.
    They are dropped here rather than approximated, exactly as
    :func:`_find_int4_matmul_candidates` drops a layer whose ``block_size``
    it cannot read -- a layer this does not recognize is simply not trained,
    and a block with none left is refused loudly by :func:`_plan_block`.
    """
    q_init = {t.name: t for t in quantized_model.graph.initializer}
    ws_init = q_init.get(candidate.ws_name)
    x_scale_init = q_init.get(candidate.x_scale_name)
    x_zp_init = q_init.get(candidate.x_zp_name)
    if ws_init is None or x_scale_init is None or x_zp_init is None:
        return None

    dims = tuple(int(d) for d in candidate.w_float_init.dims)
    channel_axis = candidate.channel_axis
    if channel_axis not in (0, 1):
        return None
    scale = onnx.numpy_helper.to_array(ws_init).astype(np.float32).reshape(-1)
    if scale.shape[0] != dims[channel_axis]:
        return None
    # The blocked axis is the *other* one: one scale covers the whole
    # reduction, which is what "per output channel" means.
    blocked = 1 - channel_axis
    scale_2d = scale.reshape((1, -1) if blocked == 0 else (-1, 1))

    x_scale = onnx.numpy_helper.to_array(x_scale_init).astype(np.float64).reshape(-1)
    x_zp = onnx.numpy_helper.to_array(x_zp_init).astype(np.float64).reshape(-1)
    if x_scale.shape[0] != 1 or x_zp.shape[0] != 1:
        return None

    return _QuantizedLayer(
        output_name=candidate.output_name,
        float_node=candidate.float_node,
        w_float_init=candidate.w_float_init,
        wq_name=candidate.wq_name,
        ws_name=candidate.ws_name,
        ws_dims=tuple(int(d) for d in ws_init.dims),
        scale_2d=scale_2d,
        axis=blocked,
        block_size=dims[blocked],
        n_min=_INT8_N_MIN,
        n_max=_INT8_N_MAX,
        packed_int4=False,
        act=_ActQuant(
            tensor=candidate.float_node.input[0],
            scale_name=candidate.x_scale_name,
            zp_name=candidate.x_zp_name,
            scale_init=float(x_scale[0]),
            zp_init=float(x_zp[0]),
        ),
    )


def _from_float(node: onnx.NodeProto, w_init: onnx.TensorProto) -> _QuantizedLayer:
    """A plain, unquantized MatMul/Gemm -- the scheme that makes this module a
    fine-tuner rather than only a quantizer.

    Everything downstream is already written against "a master weight the
    block reads and an optimizer updates"; quantization is what sits *between*
    those two, and :attr:`_QuantizedLayer.fake_quant` is the switch that
    removes it. So this constructor's job is only to say which weight is
    trainable and where it is written back -- which, with no code array in the
    picture, is the weight initializer itself.

    The quantizer fields are unread when ``fake_quant`` is off, and they are
    filled in with values chosen to *break* rather than to look plausible:
    ``block_size`` of 0 makes the write-back's ``np.repeat`` produce an empty
    array and ``n_min == n_max == 0`` makes a fake-quant forward produce all
    zeros. A neutral-looking ``block_size`` of 1 with a unit scale would
    instead round the weights to integers and train on quietly, which is the
    failure mode worth ruling out: it is wrong, and it looks like a model
    that merely trained badly.
    """
    return _QuantizedLayer(
        output_name=node.output[0],
        float_node=node,
        w_float_init=w_init,
        # There is no separate code array: the tensor trained and the tensor
        # written back are the same one.
        wq_name=w_init.name,
        ws_name="",
        ws_dims=(),
        scale_2d=np.zeros((1, 1), dtype=np.float32),
        axis=0,
        block_size=0,
        n_min=0.0,
        n_max=0.0,
        packed_int4=False,
        act=None,
        fake_quant=False,
    )


def _find_float_layers(model: onnx.ModelProto) -> List[_QuantizedLayer]:
    """Every plain MatMul/Gemm in ``model`` whose weight is a 2-D fp32
    initializer.

    Scanned out of the *student* rather than the teacher, unlike the two
    quantized schemes, and that is the substantive difference between
    fine-tuning and QAT rather than an implementation detail. QAT seeds its
    master weights from the teacher because the student's weights are a
    lossy encoding of them and the teacher's are the thing being encoded.
    Fine-tuning has no such relationship: the student's weights are the
    starting point precisely because they are *not* the teacher's -- they
    were pruned, or simplified, or already tuned -- and re-seeding from the
    teacher would throw that away before the first step.

    ``Conv`` is here and is not in either quantized finder, which is not an
    oversight in those: ``adaround``'s INT4 finder and ``quantize_static``'s
    QDQ finder are both MatMul/Gemm-only, so no quantized scheme ever
    produces a Conv layer and there is nothing for them to train. Training a
    Conv's weight is therefore inherently a ``fake_quant=False`` feature,
    which is also what makes it cheap: the fake-quant path reads a weight as
    a 2-D grid of scale blocks (:func:`_blocked_shapes`), and none of that
    runs here.

    Rank is otherwise left alone -- a Conv's weight is
    ``[M, C/group, *kernel]``, rank 3 for a 1-D convolution and 5 for a 3-D
    one, and the training machinery is indifferent to which: the master
    weight is fed to the block's own node in the layout that node already
    reads, Adam's moments are ``zeros_like`` it, and the write-back stores it
    back unchanged. Only fp32 is still required, because the state tensors
    the loop carries are fp32.

    A weight of rank < 2 is skipped rather than trained: nothing here would
    break on one, but no MatMul, Gemm or Conv has a rank-1 weight, so such a
    tensor is something this function has misidentified.
    """
    initializers = {t.name: t for t in model.graph.initializer}
    layers: List[_QuantizedLayer] = []
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Gemm", "Conv") or len(node.input) < 2:
            continue
        if not node.output or not node.output[0]:
            continue
        w_init = initializers.get(node.input[1])
        if w_init is None:
            continue
        if w_init.data_type != onnx.TensorProto.FLOAT or len(w_init.dims) < 2:
            continue
        layers.append(_from_float(node, w_init))
    return layers


def _find_layers(
    float_model: onnx.ModelProto,
    quantized_model: onnx.ModelProto,
    activation_quant: bool,
    fake_quant: bool = True,
) -> List[_QuantizedLayer]:
    """Every layer of the scheme this run targets, in one list.

    The two finders are mutually exclusive by construction -- a weight
    dequantized from INT4 with a ``block_size`` is not one dequantized from
    per-channel INT8, and a weight-only model has no activation QDQ pair at
    all -- so this selects rather than merges. Which one is selected is the
    single decision ``learn_activation_scales`` makes; see :func:`apply_qat`.

    ``fake_quant=False`` selects neither: it is the third scheme, in which
    there is nothing quantized to look for and the trainable layers are just
    the student's own float ones.
    """
    if not fake_quant:
        return _find_float_layers(quantized_model)
    if not activation_quant:
        return [
            _from_int4(c)
            for c in _find_int4_matmul_candidates(float_model, quantized_model)
        ]
    layers = []
    for c in _find_static_qdq_candidates(float_model, quantized_model):
        layer = _from_static(c, quantized_model)
        if layer is not None:
            layers.append(layer)
    return layers


@dataclass
class _Trained:
    """One quantized layer's trainable state inside the step graph.

    ``w`` is the fp32 master weight in the *storage* layout the graph's own
    initializer uses ([K, N] for a MatMul, [N, K] for a ``transB`` Gemm,
    [M, C/group, *kernel] for a Conv) -- unlike :mod:`onnxsim.adaround` and
    :mod:`onnxsim.brecq`, which normalize to [N, K], because nothing here
    needs a normalized layout: the forward consumes the weight exactly where
    the block's own node reads it, and the scale's blocked axis is carried
    explicitly instead.

    :attr:`w_shape` is therefore whatever rank the layer's own weight has.
    Only the fake-quant path constrains it: :func:`_blocked_shapes` and the
    two functions built on it read a weight as a 2-D grid of scale blocks,
    which is what both quantized schemes produce and neither ever produces
    for a Conv -- ``adaround``'s INT4 finder and ``quantize_static``'s QDQ
    finder are both MatMul/Gemm-only. So a rank > 2 weight reaches here only
    with ``fake_quant=False``, where none of that code runs.
    """

    candidate: _QuantizedLayer
    w_input: str
    m_input: str
    w_shape: Tuple[int, ...]
    w_init: np.ndarray
    scale_axis: int
    scale_shape: Tuple[int, int]
    scale_init: np.ndarray
    # The weight's second Adam moment. Present (a real name) only when the
    # weight is trained with Adam; ``None`` when it is trained with SGD
    # momentum instead, which has only one state tensor (``m_input`` doubles
    # as that one moment buffer) and so no use for a second. See
    # :func:`_build_step_graph` and :func:`_plan_trained`.
    v_input: Optional[str] = None
    scale_input: Optional[str] = None
    ms_input: Optional[str] = None
    vs_input: Optional[str] = None
    # The layer's own activation quantizer, when one is being trained. Its
    # scale is carried in log space for the reason onnxsim.adaquant carries
    # it that way: a gradient step can then never drive the scale to zero or
    # negative, which would make the quantizer undefined.
    act: Optional[_ActQuant] = None
    log_scale_input: Optional[str] = None
    ma_input: Optional[str] = None
    va_input: Optional[str] = None
    zp_input: Optional[str] = None
    mz_input: Optional[str] = None
    vz_input: Optional[str] = None
    # Filled in as the graph is built.
    w_next: str = ""
    m_next: str = ""
    v_next: str = ""
    scale_next: str = ""
    ms_next: str = ""
    vs_next: str = ""
    log_scale_next: str = ""
    ma_next: str = ""
    va_next: str = ""
    zp_next: str = ""
    mz_next: str = ""
    vz_next: str = ""


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


def _blocked_weight_shape(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """``shape`` as the 2-D grid the fake-quant path reads a weight as.

    Only that path calls this. A weight of any other rank cannot legitimately
    reach it: both quantized finders are MatMul/Gemm-only, so a Conv's
    ``[M, C/group, *kernel]`` weight arrives only with ``fake_quant=False``,
    where none of the blocked-scale code runs.

    That is an invariant rather than a coincidence, so it is checked here
    instead of asserted in a comment. The failure it prevents is quiet: a
    rank-4 weight reinterpreted as a 2-D block grid produces a perfectly
    valid graph that trains the wrong thing.
    """
    if len(shape) != 2:
        raise ValueError(
            "the fake-quant path reads a weight as a 2-D grid of scale blocks, "
            f"but this one has shape {list(shape)}. No quantized scheme "
            "produces a layer of that rank -- both finders are MatMul/Gemm-only "
            "-- so a layer has been planned for the wrong scheme."
        )
    return (shape[0], shape[1])


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
    b: qat_graph.GraphBuilder,
    w: str,
    scale_full: str,
    out_name: str,
    n_min: float = _N_MIN,
    n_max: float = _N_MAX,
) -> Tuple[str, str, str]:
    """``w_hat = clip(round(w / s), n_min, n_max) * s``, written into
    ``out_name``.

    Returns ``(code, ratio, active)`` -- everything the straight-through
    backward below needs. ``active`` is a float 0/1 mask of the elements
    strictly inside the clipping range, which is where ``round``'s
    straight-through derivative of 1 is allowed to pass anything through at
    all; a saturated element's weight has no local influence on the block's
    output and must not be moved by the reconstruction gradient.

    Clipping happens *before* rounding rather than after. The two commute
    here because the bounds are integers, and doing it in this order leaves
    :meth:`onnxsim.qat_graph.GraphBuilder.round_to_nearest` with an argument
    already bounded to the grid, where its float-to-int32 cast is exact.

    ``n_min``/``n_max`` default to ``quantize_weight_only_int4``'s symmetric
    ``[-7, 7]``; ``quantize_static``'s per-channel INT8 weights pass
    ``[-127, 127]``. Nothing else about the fake-quant differs between the
    two schemes, which is why the grid is a pair of numbers rather than a
    second code path.
    """
    ratio = b.div(w, scale_full)
    code = b.round_to_nearest(b.clip(ratio, n_min, n_max))
    b.nodes.append(onnx.helper.make_node("Mul", [code, scale_full], [out_name]))
    active = b.mul(
        b.greater_mask(ratio, n_min),
        b.less_mask(ratio, n_max),
    )
    return code, ratio, active


def _take_nodes(b: qat_graph.GraphBuilder, start: int) -> List[onnx.NodeProto]:
    """The nodes ``b`` has accumulated since index ``start``, removed from it.

    :func:`_emit_activation_fake_quant` needs its nodes in two lists at once
    -- the graph's, and the subset :func:`onnxsim.graph_grad.build_backward`
    is asked to differentiate -- and a builder appends to only one. Lifting
    them back out is a line; duplicating :class:`onnxsim.qat_graph.GraphBuilder`'s
    naming and initializer bookkeeping to build them somewhere else would not
    be.
    """
    taken = b.nodes[start:]
    del b.nodes[start:]
    return taken


def _emit_activation_fake_quant(
    b: qat_graph.GraphBuilder, t: _Trained, x_shape: Sequence[int]
) -> Tuple[List[onnx.NodeProto], List[onnx.NodeProto], Dict[str, Sequence[int]], str]:
    """One layer's uint8 affine quantize-dequantize, with a learnable scale
    and zero-point, as nodes.

    Returns ``(all nodes, the differentiable subset, their shapes, the
    dequantized tensor's name)``. The caller splices the first list into the
    graph ahead of the layer that reads it and hands the second to
    :func:`onnxsim.graph_grad.build_backward`.

    **Why this emits the chain stage by stage rather than a
    ``QuantizeLinear``/``DequantizeLinear`` pair** is the point
    :mod:`onnxsim.adaquant`'s own docstring makes and this inherits: the
    scale's gradient lives entirely in the *difference* between ``x / s`` and
    its rounded value, so a formulation that hid the rounding inside one op
    -- or that "simplified" the round trip to the identity it almost is --
    would have nothing left to differentiate.

    **Why the gradients are not written out here.** :mod:`onnxsim.adaquant`
    derives them by hand:
    ``d(xdq)/ds = (xq - zp) - active * x/s``, ``d(xdq)/d(zp) = s * (active -
    1)``, ``d/d(log s) = s * d/ds``, and a straight-through ``d(xdq)/dx =
    active``. Every one of those falls out of
    :mod:`onnxsim.graph_grad`'s ordinary rules applied to the chain below,
    *provided* the round is expressed so the straight-through estimator is
    structural rather than asserted -- which is the one trick here:

        ``round(r)`` is emitted as ``r + residual`` where ``residual =
        round(r) - r`` is computed by nodes deliberately left **out** of the
        differentiated list.

    A tensor no differentiated node produces is a leaf, so the backward walk
    stops at it and the ``Add``'s other operand receives the whole incoming
    gradient -- which is exactly what "the derivative of round is 1" means.
    Include those two nodes instead and the residual's own ``-r`` cancels the
    ``+r``, leaving a zero gradient: the failure mode this shape exists to
    avoid. ``tests/test_qat.py`` checks the emitted gradients against
    adaquant's closed form rather than trusting the argument.

    The scale is read as ``exp(log_scale)`` inside the differentiated chain,
    so the log-space chain rule is the ``Exp`` rule and not a hand-applied
    factor.
    """
    act = t.act
    assert act is not None  # only called for a layer with a trained quantizer
    log_scale, zp = str(t.log_scale_input), str(t.zp_input)
    scalar: List[int] = []
    shapes: Dict[str, Sequence[int]] = {log_scale: scalar, zp: scalar}

    start = len(b.nodes)
    scale = b.op("Exp", [log_scale], "act_s")
    ratio = b.div(act.tensor, scale)
    head = _take_nodes(b, start)
    shapes[scale] = scalar
    shapes[ratio] = list(x_shape)

    # The stop-gradient half: these two nodes are in the graph but not in the
    # differentiated list, which is what makes the rounding a straight-through
    # estimator. See this function's docstring.
    start = len(b.nodes)
    residual = b.sub(b.round_to_nearest(ratio), ratio)
    rounding = _take_nodes(b, start)
    shapes[residual] = list(x_shape)

    start = len(b.nodes)
    rounded = b.add(ratio, residual)
    raw = b.add(rounded, zp)
    clipped = b.clip(raw, _ACT_N_MIN, _ACT_N_MAX)
    centred = b.sub(clipped, zp)
    xdq = b.mul(centred, scale)
    tail = _take_nodes(b, start)
    for name in (rounded, raw, clipped, centred, xdq):
        shapes[name] = list(x_shape)

    return head + rounding + tail, head + tail, shapes, xdq


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


def _tensor_elem_types(model: onnx.ModelProto) -> Dict[str, int]:
    """``{tensor name: ONNX element type}`` for every tensor ``model``
    declares a type for, after a best-effort shape-inference pass fills in
    whatever ``model`` did not already carry.

    Every tensor a block reads from outside itself used to be assumed
    float32 -- true of every op in :data:`onnxsim.graph_grad.SUPPORTED_OPS`
    until ``Gather`` joined it, whose ``indices`` input is a genuine integer
    tensor. This is the single place that assumption is replaced by the
    model's own answer, mirroring :func:`onnxsim.qat_interop._infer`'s
    reasoning: inference only *adds* value_info, so a model it fails on is
    simply used as-is, and a name still missing afterwards (never a graph
    input/output, never produced by a node, never an initializer) is left
    out of the map for the caller to default on.
    """
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:  # noqa: BLE001 -- inference is best-effort here
        inferred = model
    types: Dict[str, int] = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    ):
        elem_type = value.type.tensor_type.elem_type
        if elem_type:
            types[value.name] = elem_type
    for init in inferred.graph.initializer:
        types[init.name] = init.data_type
    return types


def _elem_type(types: Dict[str, int], name: str) -> int:
    """``types[name]``, defaulting to FLOAT for a name shape inference could
    not type -- the assumption every block-external tensor satisfied
    unconditionally before ``Gather`` made a non-float one possible.
    """
    return types.get(name, onnx.TensorProto.FLOAT)


def _np_elem_type(dtype: np.dtype) -> int:
    """The ONNX element type a captured array's own numpy dtype corresponds
    to -- what :func:`_build_step_graph` declares a constant's step-graph
    input as, now that :func:`_capture` preserves each tensor's real dtype
    instead of forcing float32 on all of them.
    """
    return int(onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype)))


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

    Every external is declared FLOAT here, save one exception: a tensor whose
    element type the *float model itself* declares (or shape inference over
    it infers) as something else -- in practice a ``Gather``'s ``indices``,
    an integer tensor entering the block sideways rather than the float
    activation every other block-external tensor is. Declaring it FLOAT
    regardless, the way this used to, is exactly what upset ONNX's own
    checker over the emitted step graph: a ``Gather`` node with a
    ``tensor(float)`` ``indices`` input is not a legal graph.
    """
    used = {name for node in nodes for name in node.input if name}
    initializers = [t for t in float_model.graph.initializer if t.name in used]
    elem_types = _tensor_elem_types(float_model)
    inputs = [
        onnx.helper.make_tensor_value_info(
            name, _elem_type(elem_types, name), list(value.shape)
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

    Each tensor is cast to the element type ``float_model`` itself declares
    (or shape inference infers) for it, not unconditionally to float32: a
    ``Gather``'s ``indices`` input is a genuine integer tensor, and casting
    its captured values to float32 the way every block-external tensor used
    to be cast would silently corrupt them (and disagree with the dtype
    :func:`_block_shapes` and :func:`onnxsim.qat_graph.make_step_graph` now
    declare for the same tensor). A name shape inference could not type falls
    back to float32, which is what this did unconditionally before.
    """
    probe = _add_probe_outputs(float_model, names)
    elem_types = _tensor_elem_types(probe)
    dtypes = {
        name: onnx.helper.tensor_dtype_to_np_dtype(_elem_type(elem_types, name))
        for name in names
    }
    collected: Dict[str, List[np.ndarray]] = {name: [] for name in names}
    for batch in calibration_data:
        out = backend.run_model(probe, batch, providers=providers)
        for name in names:
            collected[name].append(np.asarray(out[name], dtype=dtypes[name]))

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
    candidates: Sequence[_QuantizedLayer],
    learn_scales: bool,
    learn_activation_scales: bool = False,
    optimizer: str = "adam",
) -> List[_Trained]:
    """One :class:`_Trained` per quantized layer in the block, with its
    master weight seeded from the *float* model's own weight -- so step 0 of
    the loop reproduces round-to-nearest exactly, and every later step is a
    measured improvement on it rather than on an arbitrary re-initialization.

    The activation quantizer, when there is one, is seeded the same way: from
    what calibration already chose, so step 0 reproduces the quantized model
    as shipped and the run can only be measured against it. That is the same
    warm start :func:`onnxsim.apply_adaquant` uses, and the reason a run that
    helps nothing costs accuracy rather than losing it outright.

    ``optimizer`` picks what the *weight's own* state looks like: with
    ``"adam"`` (the default) a second-moment buffer ``v_input`` is allocated
    alongside ``m_input``; with ``"sgd_momentum"`` there is only the one
    momentum buffer Adam's ``m_input`` name already carries, so ``v_input``
    is left ``None`` -- see :class:`_Trained`. It never touches the scale or
    activation-quantizer state (``ms_input``/``vs_input``,
    ``ma_input``/``va_input``/``mz_input``/``vz_input``), which are always
    Adam's two moments regardless of this choice -- see :func:`_build_step_graph`.
    """
    planned: List[_Trained] = []
    for i, candidate in enumerate(candidates):
        w = onnx.numpy_helper.to_array(candidate.w_float_init).astype(np.float32)
        scale = candidate.scale_2d.astype(np.float32)
        trained = _Trained(
            candidate=candidate,
            w_input=f"{_PREFIX}w{i}",
            m_input=f"{_PREFIX}mw{i}",
            v_input=f"{_PREFIX}vw{i}" if optimizer == "adam" else None,
            w_shape=tuple(int(d) for d in w.shape),
            w_init=w,
            scale_axis=candidate.axis,
            scale_shape=(int(scale.shape[0]), int(scale.shape[1])),
            scale_init=scale,
        )
        if learn_scales:
            trained.scale_input = f"{_PREFIX}s{i}"
            trained.ms_input = f"{_PREFIX}ms{i}"
            trained.vs_input = f"{_PREFIX}vs{i}"
        if learn_activation_scales and candidate.act is not None:
            trained.act = candidate.act
            trained.log_scale_input = f"{_PREFIX}as{i}"
            trained.ma_input = f"{_PREFIX}mas{i}"
            trained.va_input = f"{_PREFIX}vas{i}"
            trained.zp_input = f"{_PREFIX}az{i}"
            trained.mz_input = f"{_PREFIX}maz{i}"
            trained.vz_input = f"{_PREFIX}vaz{i}"
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
    learn_activation_scales: bool = False,
    fake_quant: bool = True,
    preserve_sparsity: bool = False,
    optimizer: str = "adam",
) -> qat_graph.StepGraph:
    """The whole loop as one graph: fake-quant forward, block forward,
    reconstruction loss, backward, optimizer.

    ``optimizer`` (``"adam"`` or ``"sgd_momentum"``) picks what the *block's
    own weight* update looks like -- see :func:`apply_qat`'s docstring for
    the full scope boundary. It never reaches the LSQ scale update or the
    activation-quantizer updates below (the ``learn_scales``/
    ``learn_activation_scales`` extras): those are always Adam, unconditionally.
    With the default ``"adam"``, every node this function emits -- including
    which scalars it declares and which state tensors it threads -- is
    byte-identical to what it emitted before this parameter existed.

    The ordering is the only subtle part. :func:`graph_grad.build_backward`
    reads forward tensors by name (including node *outputs*, where reusing a
    ``Sigmoid``/``Softmax`` result is cheaper than recomputing it), so every
    node it differentiates must already sit in the builder ahead of the nodes
    it appends. Hence: (optionally the minibatch gather,) fake-quant, then the
    block's own nodes verbatim, then the loss seed, then the backward, then
    the optimizer.

    With ``learn_activation_scales`` the ordering gains one more rule: each
    trained layer's activation fake-quant is spliced in immediately before
    the layer that reads it, and its differentiable half joins the list
    handed to the backward. The block's other nodes are untouched, so the
    two features compose without either knowing about the other.

    With ``fake_quant=False`` step 1 disappears instead: the block's node
    reads the master weight itself, the loop is plain gradient descent on the
    block's weights, and everything else here -- the teacher, the loss, the
    backward, Adam, the minibatch -- is the same graph it always was. What
    makes that a *fine-tuner* rather than a no-op is that the teacher and the
    student are different models: the student is the pruned, simplified or
    otherwise altered one, and the block is trained to reproduce what the
    original produced at that point.

    ``preserve_sparsity`` adds one ``Mul`` per trained layer, zeroing the
    weight gradient wherever the master weight started at zero. That is enough
    to hold those elements at zero exactly, rather than merely near it: with a
    gradient of 0 every step, Adam's ``m`` and ``v`` stay 0, its step is
    ``lr * 0 / (sqrt(0) + eps)`` -- exactly 0 -- and the parameter never moves.
    No clean-up pass at the end, and no drift in between.

    ``externals`` and ``block_output_shape`` are always the *whole*
    calibration set's arrays and shape. With ``batch`` set they become the
    resident tables rather than the block's inputs, and the block's own
    tensors are ``batch.size`` rows gathered out of them -- so every shape
    from the block input downwards, the loss normalizer included, is a
    batch-sized shape, and none of the code below has to know which case it
    is in.
    """
    _refuse_unknown_optimizer(optimizer)
    b = qat_graph.GraphBuilder(_PREFIX)
    b.initializer.extend(block_initializers)

    # 0. The minibatch, if there is one. Each captured tensor is declared at
    #    its full size and a Gather pulls this step's rows out of it under the
    #    name the block's own nodes were written against, so step 2 below can
    #    still splice those nodes in verbatim. The block never learns that its
    #    input stopped being a graph input.
    teacher = f"{_PREFIX}teacher"
    # Each constant's declared element type is the captured array's own dtype
    # -- float for every external this ever ran on, and now whatever
    # non-float type a ``Gather``'s ``indices`` genuinely has, since
    # :func:`_capture` stopped forcing float32 on every block-external
    # tensor. The teacher/target is always float: it is the reconstruction
    # loss's own output, never a block-external a ``Gather`` could have made
    # non-float.
    constants: Dict[str, Tuple[Sequence[int], int]] = {}
    if batch is None:
        constants.update(
            {
                name: (list(value.shape), _np_elem_type(value.dtype))
                for name, value in sorted(externals.items())
            }
        )
        constants[teacher] = (list(block_output_shape), onnx.TensorProto.FLOAT)
    else:
        rows = batch.index_name
        # A captured tensor's table is ``qat__all_<its name>`` and the
        # teacher's is ``qat__teacher_all``; the two families cannot collide
        # whatever the model calls its tensors, since one starts ``qat__all_``
        # and the other ``qat__teacher_``.
        for name, value in sorted(externals.items()):
            table = f"{_PREFIX}all_{name}"
            constants[table] = (list(value.shape), _np_elem_type(value.dtype))
            b.gather_rows(table, rows, name)
        constants[f"{_PREFIX}teacher_all"] = (
            list(block_output_shape),
            onnx.TensorProto.FLOAT,
        )
        b.gather_rows(f"{_PREFIX}teacher_all", rows, teacher)
        block_output_shape = [batch.size] + list(block_output_shape)[1:]

    # 1. Fake-quantize each trained master weight into the tensor name the
    #    block's own node already reads, so the block's nodes need no
    #    rewriting at all -- the weight initializer simply became a computed
    #    value.
    # (layer, the tensor its gradient arrives on, and the fake-quant's own
    # intermediates). The last four are None exactly when there is no
    # fake-quant, which is what step 5 branches on.
    per_layer: List[
        Tuple[_Trained, str, Optional[str], Optional[str], Optional[str], Optional[str]]
    ] = []
    # Block tensor name -> what the block's own node should read instead.
    # Only ``fake_quant=False`` puts anything here: the master weight is
    # substituted for the weight initializer by *renaming one input*, rather
    # than by emitting an Identity -- a node whose whole job is to copy a
    # tensor is a node worth avoiding even though ``Identity`` is, as of
    # :mod:`onnxsim.graph_grad`'s templated "Add" rule, in EP_FRIENDLY_OPS:
    # renaming costs the execution provider nothing at all, where even an
    # allowlisted no-op node still costs a dispatch.
    weight_rewrites: Dict[str, str] = {}
    weight_shapes: Dict[str, Sequence[int]] = {}
    for t in trained:
        if not fake_quant:
            weight_name = t.candidate.float_node.input[1]
            weight_rewrites[weight_name] = t.w_input
            # The master weight is now a differentiated *leaf* of the block
            # rather than a value computed inside it, so the backward needs
            # its shape the way it needs the block's own tensors'.
            weight_shapes[t.w_input] = list(t.w_shape)
            # No quantizer, so no straight-through mask: the ``active`` slot
            # below is None and step 5 uses the raw gradient.
            per_layer.append((t, t.w_input, None, None, None, None))
            continue
        block_size = t.candidate.block_size
        if t.scale_input is None:
            scale = b.const(t.scale_init, "scale")
        else:
            scale = t.scale_input
        scale_full = _broadcast_scale(
            b,
            scale,
            _blocked_weight_shape(t.w_shape),
            t.scale_shape,
            t.scale_axis,
            block_size,
        )
        weight_name = t.candidate.float_node.input[1]
        code, ratio, active = _emit_fake_quant(
            b,
            t.w_input,
            scale_full,
            weight_name,
            t.candidate.n_min,
            t.candidate.n_max,
        )
        per_layer.append((t, weight_name, scale_full, code, ratio, active))

    # 2. The block itself, node for node as the float graph wrote it -- except
    #    that a layer whose activation quantizer is being trained reads a
    #    fake-quantized copy of its own input instead of the raw tensor. Only
    #    that one node is rewritten (one input name), so the quantizer lands
    #    on the *edge* the deployed QDQ pair occupies rather than on the
    #    tensor: two layers sharing an activation keep the two independent
    #    quantizers quantize_static gave them.
    act_shapes: Dict[str, Sequence[int]] = {}
    quantized_input = {t.candidate.output_name: t for t in trained if t.act is not None}
    forward: List[onnx.NodeProto] = []
    differentiated: List[onnx.NodeProto] = []
    for node in nodes:
        layer = quantized_input.get(node.output[0]) if node.output else None
        rewrites = [name for name in node.input if name in weight_rewrites]
        if layer is not None or rewrites:
            rewritten = onnx.NodeProto()
            rewritten.CopyFrom(node)
            if layer is not None:
                emitted, diff_nodes, extra, xdq = _emit_activation_fake_quant(
                    b, layer, shapes[node.input[0]]
                )
                forward.extend(emitted)
                differentiated.extend(diff_nodes)
                act_shapes.update(extra)
                rewritten.input[0] = xdq
            for i, name in enumerate(rewritten.input):
                if name in weight_rewrites:
                    rewritten.input[i] = weight_rewrites[name]
            node = rewritten
        forward.append(node)
        differentiated.append(node)
    b.nodes.extend(forward)

    # 3. The objective: MSE of the student block's output against the
    #    teacher's, and its gradient, which is the seed of the backward pass.
    #    ``block_output_shape`` is this step's shape, so the 2/n normalizer is
    #    the batch's element count and a minibatched gradient is the same
    #    *average* per-element quantity a full-batch one is -- which is what
    #    keeps one learning rate meaningful across batch sizes.
    diff = b.sub(block_output_name, teacher)
    n_elems = int(np.prod(list(block_output_shape)))
    dl_dy = b.mul(diff, b.const(2.0 / n_elems))

    # 4. The backward pass over the block, emitted as ONNX nodes. The
    #    activation quantizers' own parameters are targets alongside the
    #    weights: nothing else reaches them, since they are read only by the
    #    fake-quant chain.
    all_shapes = dict(shapes)
    all_shapes.update(act_shapes)
    all_shapes.update(weight_shapes)
    targets = [weight_name for _, weight_name, _, _, _, _ in per_layer]
    for t in trained:
        if t.act is not None:
            targets.extend([str(t.log_scale_input), str(t.zp_input)])
    grads = graph_grad.build_backward(
        b,
        differentiated,
        all_shapes,
        {block_output_name: dl_dy},
        targets,
    )

    # 5. Straight through the fake-quant, into the master weight and (if
    #    asked for) the scale, then one optimizer step each: the weight uses
    #    whichever of Adam/SGD-momentum ``optimizer`` names; the scale (like
    #    the activation quantizer below) is always Adam, regardless.
    for t, weight_name, _scale_full, quant_code, quant_ratio, ste_mask in per_layer:
        g = grads[weight_name]  # dL/d(w_hat), in the weight's storage layout
        # STE: d(w_hat)/d(w) is 1 inside the clipping range and 0 outside.
        # The scale cancels -- w_hat = round(w/s)*s -- which is why a
        # straight-through weight gradient is just the masked output
        # gradient, with no scale factor anywhere. Without a fake-quant there
        # is no clipping range and no estimator: ``g`` is already dL/dw.
        w_grad = g if ste_mask is None else b.mul(g, ste_mask)
        if preserve_sparsity:
            # The zeros the optimizer *started* from, held there. Emitted per
            # layer rather than hoisted, so the mask sits next to the gradient
            # it applies to and the builder's name counter stays a function of
            # emission order.
            w_grad = b.mul(w_grad, b.const((t.w_init != 0).astype(np.float32), "keep"))
        if optimizer == "adam":
            t.w_next, t.m_next, t.v_next = qat_graph.adam_update(
                b,
                t.w_input,
                w_grad,
                t.m_input,
                str(t.v_input),
                f"{_PREFIX}lr",
                "m_correction",
                "v_correction",
            )
        else:  # "sgd_momentum" -- the only other value _refuse_unknown_optimizer allows
            t.w_next, t.m_next = qat_graph.sgd_momentum_update(
                b,
                t.w_input,
                w_grad,
                t.m_input,
                f"{_PREFIX}lr",
            )
        if t.scale_input is not None:
            # LSQ's scale gradient, the same one onnxsim.autoround derives:
            # d(w_hat)/d(s) = code - w/s where the element is inside the
            # clipping range (the gap between the integer it rounds to and the
            # exact ratio) and just `code` where it saturates.
            dwhat_ds = b.sub(str(quant_code), b.mul(str(ste_mask), str(quant_ratio)))
            g_scale = _sum_over_blocks(
                b,
                b.mul(g, dwhat_ds),
                _blocked_weight_shape(t.w_shape),
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
        if t.act is None:
            continue
        # The activation quantizer's two parameters. Their gradients were
        # emitted by the backward walk over the fake-quant chain (see
        # _emit_activation_fake_quant for why that reproduces adaquant's
        # hand-derived ones exactly), so all that is left is an Adam step
        # each.
        t.log_scale_next, t.ma_next, t.va_next = qat_graph.adam_update(
            b,
            str(t.log_scale_input),
            grads[str(t.log_scale_input)],
            str(t.ma_input),
            str(t.va_input),
            f"{_PREFIX}lr_act",
            "m_correction",
            "v_correction",
        )
        zp_stepped, t.mz_next, t.vz_next = qat_graph.adam_update(
            b,
            str(t.zp_input),
            grads[str(t.zp_input)],
            str(t.mz_input),
            str(t.vz_input),
            f"{_PREFIX}lr_act",
            "m_correction",
            "v_correction",
        )
        # Re-clamped into uint8's range every step rather than only at export,
        # for onnxsim.adaquant's own reason: the forward's clip is what the
        # whole activation gradient is derived through, so a zero-point that
        # wandered outside the representable range would saturate every
        # element and silently kill the signal.
        t.zp_next = b.clip(zp_stepped, _ACT_N_MIN, _ACT_N_MAX)

    state: Dict[str, Tuple[Sequence[int], str]] = {}
    for t in trained:
        state[t.w_input] = (list(t.w_shape), t.w_next)
        state[t.m_input] = (list(t.w_shape), t.m_next)
        if t.v_input is not None:
            state[t.v_input] = (list(t.w_shape), t.v_next)
        if t.scale_input is not None:
            state[t.scale_input] = (list(t.scale_shape), t.scale_next)
            state[str(t.ms_input)] = (list(t.scale_shape), t.ms_next)
            state[str(t.vs_input)] = (list(t.scale_shape), t.vs_next)
        if t.act is not None:
            for name, out in (
                (str(t.log_scale_input), t.log_scale_next),
                (str(t.ma_input), t.ma_next),
                (str(t.va_input), t.va_next),
                (str(t.zp_input), t.zp_next),
                (str(t.mz_input), t.mz_next),
                (str(t.vz_input), t.vz_next),
            ):
                state[name] = ([], out)

    # "m_correction"/"v_correction" are Adam's bias-correction factors
    # (adam_update's own inputs). They are declared here -- and must be fed
    # by every caller of the resulting step graph -- exactly when *something*
    # in this block uses Adam: the weight itself (optimizer == "adam") or, if
    # neither is, the scale/activation updates above, which are always Adam
    # regardless of ``optimizer``. A step graph that declares neither of the
    # two extra optimizer flags and chose sgd_momentum for its one weight
    # never declares these -- and onnxruntime raises on a feed for an input a
    # graph never declared, so run_step_graph's own scalars callback (see
    # _train_block) must derive the identical condition rather than guess.
    scalars = [f"{_PREFIX}lr"]
    uses_adam = optimizer == "adam" or learn_scales or learn_activation_scales
    if uses_adam:
        scalars += ["m_correction", "v_correction"]
    if learn_scales:
        scalars.append(f"{_PREFIX}lr_scale")
    if learn_activation_scales:
        scalars.append(f"{_PREFIX}lr_act")

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
    candidates: List[_QuantizedLayer]


def _no_layers_message(
    float_model: onnx.ModelProto,
    quantized_model: onnx.ModelProto,
    block_input_name: str,
    block_output_name: str,
    learn_activation_scales: bool,
    fake_quant: bool = True,
) -> str:
    """Why this block has nothing to train, said in terms of the *scheme* the
    caller asked for.

    The bare fact ("no matching layer in this slice") is nearly useless when
    the real cause is that the model was quantized by a different
    ``quantize_*`` function than the flag selects -- a caller who turns
    ``learn_activation_scales`` on over a weight-only INT4 model has made a
    scheme error, not a boundary error, and would otherwise go looking at
    their tensor names. So the mismatch is detected and named. Naming it
    costs one extra scan of the model, on a path that is about to raise.
    """
    where = f"the block between {block_input_name!r} and {block_output_name!r}"
    if not fake_quant:
        return (
            f"{where} contains no MatMul/Gemm/Conv with an fp32 weight "
            "initializer of rank 2 or more to fine-tune (fake_quant=False "
            "trains the model's own float weights, so a layer whose weight is "
            "computed rather than stored, or stored at some other dtype, has "
            "nothing for the optimizer to hold)"
        )
    if learn_activation_scales:
        if _find_int4_matmul_candidates(float_model, quantized_model):
            return (
                "learn_activation_scales targets onnxsim.quantize_static's QDQ "
                "scheme (uint8 activations, per-output-channel INT8 weights), but "
                "this quantized model is an onnxsim.quantize_weight_only_int4 one "
                "-- a weight-only model has no activation quantizer anywhere in "
                "it to train. Re-quantize with onnxsim.quantize_static, or leave "
                "learn_activation_scales off to fine-tune the INT4 weights."
            )
        return (
            f"{where} contains no quantize_static-quantized MatMul/Gemm layer to "
            "train (learn_activation_scales targets that scheme; see apply_qat's "
            "docstring for why it is the only one with activation quantizers to "
            "train)"
        )
    message = (
        f"{where} contains no quantize_weight_only_int4-quantized MatMul/Gemm "
        "layer to train"
    )
    if _find_static_qdq_candidates(float_model, quantized_model):
        message += (
            "; this model's layers match onnxsim.quantize_static's QDQ scheme "
            "instead, which learn_activation_scales=True trains"
        )
    return message


def _plan_block(
    float_model: onnx.ModelProto,
    quantized_model: onnx.ModelProto,
    block_input_name: str,
    block_output_name: str,
    learn_activation_scales: bool = False,
    fake_quant: bool = True,
) -> _BlockPlan:
    """Slices the block out of the float graph and checks the three things
    that make it trainable at all: it is non-empty, every op in it has a
    gradient rule, and at least one of its layers is INT4-quantized.

    Deliberately does *not* check shapes -- that needs the concrete
    calibration activations, so it happens later in :func:`_train_block`.

    ``learn_activation_scales`` selects *which scheme* counts as quantized
    here -- ``quantize_weight_only_int4``'s INT4 weight-only layers, or
    ``quantize_static``'s QDQ ones -- so the same block can be trainable
    under one and refused under the other. That is the point rather than a
    wart: the two are different deployed models, and a run has to be aimed
    at the one that will ship.

    ``fake_quant=False`` replaces that question with a simpler one: the
    trainable layers are the student's own float MatMul/Gemms, and the third
    check becomes "at least one of those".
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
        for c in _find_layers(
            float_model, quantized_model, learn_activation_scales, fake_quant
        )
        if c.output_name in slice_outputs
    ]
    if not candidates:
        raise ValueError(
            _no_layers_message(
                float_model,
                quantized_model,
                block_input_name,
                block_output_name,
                learn_activation_scales,
                fake_quant,
            )
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
    learn_activation_scales: bool = False,
    activation_learning_rate: float = 1e-2,
    fake_quant: bool = True,
    preserve_sparsity: bool = False,
    optimizer: str = "adam",
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

    ``optimizer`` picks the block's own weight update (``"adam"`` or
    ``"sgd_momentum"``); see :func:`apply_qat` for the full scope boundary.
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

    trained = _plan_trained(
        plan.candidates, learn_scales, learn_activation_scales, optimizer
    )
    trained_weight_names = {t.candidate.float_node.input[1] for t in trained}
    used = {name for node in plan.nodes for name in node.input if name}
    # The block's *untrained* constants -- a LayerNorm's scale and bias, a
    # Gemm's C -- come from whichever model the trained weights came from, and
    # for the same reason. Under QAT that is the teacher, whose weights the
    # student encodes. Under fine-tuning it is the student, because the
    # student is a different model and quietly substituting the teacher's
    # constants into it would train the block to compensate for a
    # substitution the deployed model does not make.
    constant_source = float_model if fake_quant else quantized_model
    block_initializers = [
        t
        for t in constant_source.graph.initializer
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
        learn_activation_scales,
        fake_quant,
        preserve_sparsity,
        optimizer=optimizer,
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
        if t.v_input is not None:
            state[t.v_input] = np.zeros_like(t.w_init)
        if t.scale_input is not None:
            state[t.scale_input] = t.scale_init
            state[str(t.ms_input)] = np.zeros_like(t.scale_init)
            state[str(t.vs_input)] = np.zeros_like(t.scale_init)
        if t.act is not None:
            # Seeded from what calibration chose, so step 0 is the quantized
            # model as shipped. The 1e-8 floor is onnxsim.adaquant's, guarding
            # the degenerate calibrated scale of exactly 0 (a constant
            # activation) that log would otherwise turn into -inf.
            zero = np.zeros((), dtype=np.float32)
            state[str(t.log_scale_input)] = np.asarray(
                np.log(max(t.act.scale_init, 1e-8)), dtype=np.float32
            )
            state[str(t.ma_input)] = zero
            state[str(t.va_input)] = zero
            state[str(t.zp_input)] = np.asarray(
                np.clip(t.act.zp_init, _ACT_N_MIN, _ACT_N_MAX), dtype=np.float32
            )
            state[str(t.mz_input)] = zero
            state[str(t.vz_input)] = zero

    # Whether the step graph declared "m_correction"/"v_correction" at all --
    # onnxruntime raises on a feed for an input the graph never declared, so
    # this has to agree exactly with _build_step_graph's own "uses_adam"
    # condition (optimizer == "adam" or learn_scales or
    # learn_activation_scales) for whether those two scalars exist. Reading
    # it off ``step.model.graph.input`` directly, rather than recomputing
    # that condition a second time here, is what keeps the two checks from
    # ever drifting apart.
    uses_adam_scalars = any(
        inp.name == "m_correction" for inp in step.model.graph.input
    )

    def scalars(t: int) -> Dict[str, float]:
        decay = 1.0 - t / num_iterations if lr_decay else 1.0
        values = {
            f"{_PREFIX}lr": learning_rate * decay,
            f"{_PREFIX}lr_scale": scale_learning_rate * decay,
            f"{_PREFIX}lr_act": activation_learning_rate * decay,
        }
        if not learn_scales:
            del values[f"{_PREFIX}lr_scale"]
        if not learn_activation_scales:
            del values[f"{_PREFIX}lr_act"]
        if uses_adam_scalars:
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
    new_weights: Dict[str, np.ndarray] = {}
    new_scales: Dict[str, np.ndarray] = {}
    new_act_scales: Dict[str, float] = {}
    new_act_zps: Dict[str, int] = {}
    for t in trained:
        w = final[t.w_input].astype(np.float64)
        if not t.candidate.fake_quant:
            # Nothing to project back onto: the master weight *is* what the
            # model stores, so the write-back is the identity that the two
            # quantized schemes' rounding and clipping stand in for.
            new_weights[t.candidate.wq_name] = w.astype(np.float32)
            continue
        scale = (
            final[t.scale_input].astype(np.float64)
            if t.scale_input is not None
            else t.scale_init.astype(np.float64)
        )
        scale_full = np.repeat(scale, t.candidate.block_size, axis=t.scale_axis)
        codes = np.clip(
            _round_half_away(w / scale_full), t.candidate.n_min, t.candidate.n_max
        )
        new_codes[t.candidate.wq_name] = codes.astype(np.int8)
        if t.scale_input is not None:
            new_scales[t.candidate.ws_name] = scale.reshape(t.candidate.ws_dims).astype(
                np.float32
            )
        if t.act is not None:
            # Out of log space, and the zero-point back onto uint8's integer
            # grid -- the two projections onnxsim.adaquant makes at the same
            # point, and for the same reason: the optimizer needs them
            # continuous, the model can only store what it can store.
            new_act_scales[t.act.scale_name] = float(
                np.exp(float(final[str(t.log_scale_input)]))
            )
            new_act_zps[t.act.zp_name] = int(
                np.clip(round(float(final[str(t.zp_input)])), _ACT_N_MIN, _ACT_N_MAX)
            )

    tuned = onnx.ModelProto()
    tuned.CopyFrom(quantized_model)
    for initializer in tuned.graph.initializer:
        weight = new_weights.get(initializer.name)
        if weight is not None:
            initializer.CopyFrom(
                onnx.numpy_helper.from_array(weight, name=initializer.name)
            )
            continue
        codes = new_codes.get(initializer.name)
        if codes is not None:
            if initializer.data_type == onnx.TensorProto.INT4:
                initializer.raw_data = _pack_int4(codes)
            else:
                initializer.CopyFrom(
                    onnx.numpy_helper.from_array(codes, name=initializer.name)
                )
            continue
        scale_array = new_scales.get(initializer.name)
        if scale_array is not None:
            initializer.CopyFrom(
                onnx.numpy_helper.from_array(scale_array, name=initializer.name)
            )
            continue
        act_scale = new_act_scales.get(initializer.name)
        if act_scale is not None:
            initializer.CopyFrom(
                onnx.numpy_helper.from_array(
                    np.array(act_scale, dtype=np.float32), name=initializer.name
                )
            )
            continue
        act_zp = new_act_zps.get(initializer.name)
        if act_zp is not None:
            initializer.CopyFrom(
                onnx.numpy_helper.from_array(
                    np.array(act_zp, dtype=np.uint8), name=initializer.name
                )
            )
    return tuned


#: The block weight's own optimizer choice -- see :func:`apply_qat`'s
#: ``optimizer`` parameter for what each one means and, crucially, what it
#: does *not* apply to.
_VALID_OPTIMIZERS = frozenset({"adam", "sgd_momentum"})


def _refuse_unknown_optimizer(optimizer: str) -> None:
    """Shared by :func:`apply_qat`/:func:`apply_qat_all_blocks` (fail fast,
    before spending a capture or a training budget) and
    :func:`_build_step_graph` (defensive -- it is the function whose contract
    ``optimizer`` actually governs, so it checks its own argument rather than
    trusting every caller to have checked first)."""
    if optimizer not in _VALID_OPTIMIZERS:
        raise ValueError(
            f"optimizer must be one of {sorted(_VALID_OPTIMIZERS)}, got {optimizer!r}"
        )


def _refuse_quantizer_flags_without_fake_quant(
    fake_quant: bool, learn_scales: bool, learn_activation_scales: bool
) -> None:
    """``fake_quant=False`` and the two scale flags are a contradiction, not a
    combination.

    Both scale flags name a parameter of a quantizer, and with the fake-quant
    gone there is no quantizer for them to name -- no weight scale, no
    activation scale, no zero-point. Silently ignoring them would be the worse
    failure of the two available: a caller who asked to learn scales and got a
    model whose scales are exactly as they were has no way to tell that from a
    run in which learning them did not help.
    """
    if fake_quant:
        return
    asked = [
        name
        for name, on in (
            ("learn_scales", learn_scales),
            ("learn_activation_scales", learn_activation_scales),
        )
        if on
    ]
    if asked:
        raise ValueError(
            f"{' and '.join(asked)} cannot be used with fake_quant=False: "
            "both train a quantizer's parameters, and fake_quant=False is "
            "the mode with no quantizer in it. Fine-tuning trains the "
            "weights themselves."
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
    learn_activation_scales: bool = False,
    activation_learning_rate: float = 1e-2,
    lr_decay: bool = True,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    batch_seed: int = 0,
    providers: Optional[Sequence[backend.Provider]] = None,
    step_providers: Optional[Sequence[backend.Provider]] = None,
    losses: Optional[List[float]] = None,
    fake_quant: bool = True,
    preserve_sparsity: bool = False,
    optimizer: str = "adam",
) -> onnx.ModelProto:
    """Fine-tunes one block's quantized weights (and, opted in, its activation
    quantizers) against the float model's own
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
            :func:`onnxsim.quantize_weight_only_int4` -- or, with
            ``learn_activation_scales=True``, by
            :func:`onnxsim.quantize_static`. Layers quantized by any other
            scheme (or left unquantized) are left untouched, and a block
            containing none of them is an error rather than a no-op.
            Assumes ``quantized_model`` was produced from ``float_model``
            without renaming any MatMul/Gemm node's own output tensor -- true
            of every onnxsim ``quantize_*`` function.
    :param block_input_name: the activation entering the block. The backward
            walk that discovers the block's nodes stops here.
    :param block_output_name: the block's own final output, the tensor whose
            reconstruction error is the loss. Nothing stops these two names
            from being the graph's own input and its own output: that makes
            the block the whole model and the loss the model's own output
            error, which is the end-to-end pass rather than a surrogate for
            it. See this module's docstring for what that was measured to
            cost and to buy, and why it is not what
            :func:`apply_qat_all_blocks` does by default.
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
    :param learn_activation_scales: also train the *activation* quantizers
            -- each trained layer's own input (scale, zero_point) pair --
            LSQ-style, jointly with the weights. **This selects the target
            scheme**, which is the part to read before turning it on.

            *Which scheme, and why there is no choice.* onnxsim produces two
            static schemes this pass could plausibly aim at.
            :func:`onnxsim.quantize_weight_only_int4` (the default target,
            everything above) quantizes weights only: its activations are
            fp32, it has no activation quantizer anywhere, and there is
            therefore no parameter to train and no initializer to write a
            trained one back into. "Activation quantization on top of it"
            would mean *inserting* quantizers -- inventing a W4A8 model no
            ``quantize_*`` function here emits and no deployment story
            expects -- which is a quantizer's job, not a fine-tuner's. So
            this targets :func:`onnxsim.quantize_static`'s QDQ scheme
            instead: uint8 asymmetric activations
            (``QuantizeLinear``/``DequantizeLinear`` around each quantized
            node's input) and per-output-channel symmetric INT8 weights. That
            is the same scheme :func:`onnxsim.apply_adaquant` targets, for the
            same reason, and this reuses adaquant's straight-through
            derivation of the activation gradients rather than re-deriving
            them.

            *It follows that this flag also changes what the weights are.*
            A ``quantize_static`` model's weights are per-channel INT8, so
            with the flag on that is what the fp32 master weights are
            fake-quantized against; the INT4 grid and its 32-element blocks
            are simply not present in such a model. Weight training is
            unchanged in every other respect, and the two parameter groups
            are trained *jointly* on one loss -- which is AdaQuant's own
            argument for why it is worth doing at all: the reconstruction
            error is a joint function of both, and a weight choice that is
            optimal against one activation clip range is not optimal against
            another.

            *Which tensors get a trained quantizer.* Exactly one per trained
            layer: the quantizer on that layer's own activation input, where
            ``quantize_static`` put it. It is trained per *edge*, not per
            tensor -- two layers reading the same activation have two
            independent QDQ pairs in the deployed model and get two
            independent trained quantizers here. A layer in the block whose
            input has no QDQ pair is not a candidate at all (it is not a
            ``quantize_static`` layer), and a block with no candidates is
            refused rather than silently trained weight-only.

            *What training the block input's quantizer means, precisely.*
            The block's input activation is teacher-forced: the value fed in
            is the float model's, not the student's. The quantizer, though,
            sits **inside** the block -- ``QuantizeLinear`` consumes the
            block's input tensor and feeds the first layer -- so it is
            genuinely this block's parameter to train, and training it is not
            a boundary violation. What it *is* is a clip range fitted to the
            teacher's distribution of that tensor rather than the student's.
            Those coincide exactly at the model's own graph input, and drift
            apart deeper in the model as upstream quantization error
            accumulates. :func:`apply_qat_all_blocks` with ``sequential=True``
            (the default) closes that gap by re-running the student before
            each block, which matters more with this flag on than without it:
            a clip range is a property of the input distribution in a way a
            weight is not.

            Off by default, and not only for compatibility: it is a harder,
            non-convex problem in two coupled parameter groups (the caution
            :mod:`onnxsim.autoround` documents for its own clip ratio), it
            rewrites activation initializers a weight-only run leaves
            untouched, and it applies to a different quantized model
            entirely. Turning it on over a
            ``quantize_weight_only_int4`` model raises
            :class:`ValueError` naming the mismatch.

            *And it is measured rather than assumed.* On a block whose
            activation range was calibrated from one unrepresentative
            outlier it takes the whole-model output error 38% below what
            training the weights alone reaches; on the same block calibrated
            on representative data it is a regression at every learning
            rate tried, because min/max on
            representative data is already close to MSE-optimal and a second
            coupled parameter group makes a solved problem harder.
            ``tests/test_qat.py`` records both, with the numbers. Use it when
            the calibrated range is wrong -- and note that re-calibrating,
            when that is available, costs one forward pass rather than a
            training budget.
    :param activation_learning_rate: Adam learning rate for the activation
            scale (optimized in log space, so it can never reach zero or go
            negative) and its zero-point, when ``learn_activation_scales`` is
            on. Larger than ``scale_learning_rate`` by default because it is
            a *log*-space step for the scale and a step in uint8 codes for
            the zero-point -- neither is measured in the units the weight
            learning rate is.
    :param lr_decay: anneal every one of the learning rates above
            (weight, weight scale, activation quantizer) linearly to zero
            across the run. On by default because the objective is piecewise constant in
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
    :param preserve_sparsity: hold every weight element that starts at zero
            at zero for the whole run. Off by default, and it is the caller's
            call rather than a detected one, because "this element is zero" and
            "this element was pruned away" are the same bit pattern and only
            the caller knows which they meant.

            **Turn it on for an unstructured-pruned model.** Nothing else in
            the step graph masks the optimizer, so without it a pruned zero
            gets a gradient like any other element and leaves zero on the very
            first step: measured at 50% sparsity, 128 zeros per weight before
            and none after, while the loss fell five orders of magnitude. Every
            signal a caller would look at says that run went well, which is
            what makes it worth a parameter rather than a note.

            Structured pruning needs nothing here -- there the channel is gone
            from the tensor rather than zeroed inside it.

            The mask is the zero pattern of the weight the optimizer *starts*
            from, which is the master weight's seed: the student's own weight
            when fine-tuning, and the float model's when quantizing (so for
            QAT over a pruned model, prune first and pass the pruned model as
            ``float_model``, which is the ordinary order anyway). Elements it
            holds are held exactly, not approximately -- see
            :func:`_build_step_graph` for why one ``Mul`` is enough.
    :param fake_quant: with ``False``, drop the fake-quantizer and train the
            *second model's own float weights* directly. Everything else --
            the teacher, the reconstruction loss, the backward, Adam, the
            minibatch -- is unchanged, so this is the same block-wise,
            label-free distillation with the quantizer taken out of the
            middle: plain fine-tuning. :func:`apply_block_finetune` is this with a
            name that says so and without the parameters that stop meaning
            anything; prefer it, and see its docstring for when the two
            models differ enough for there to be something to learn.
            Incompatible with ``learn_scales`` and
            ``learn_activation_scales``, which have no scales to learn.
    :param optimizer: which optimizer trains the block's own weight --
            ``"adam"`` (the default) or ``"sgd_momentum"`` (classic
            heavy-ball momentum SGD, :func:`onnxsim.qat_graph.sgd_momentum_update`).
            **This is deliberately scoped to the weight alone.** When
            ``learn_scales`` and/or ``learn_activation_scales`` are also on,
            their LSQ scale / activation-quantizer parameters always train
            with Adam, regardless of what ``optimizer`` says -- so
            ``optimizer="sgd_momentum"`` with ``learn_scales=True`` trains
            the weight with SGD-momentum and the scale with Adam, in the same
            run. That is not an oversight: Adam's per-parameter state is two
            tensors shaped like the parameter (``m``, ``v``) plus two
            step-dependent bias-correction scalars, while SGD-momentum's is
            one tensor and no scalars, so a *uniform* optimizer choice across
            weight, scale and activation quantizer would mean plumbing that
            same either/or through three independently-shaped parameter
            groups instead of one -- a materially larger change for a
            feature nothing has asked to extend past the weight, which is
            also the one parameter every trained block always has (the scale
            and activation-quantizer state exist only when their own opt-in
            flags are on). Raises :class:`ValueError` if ``optimizer`` is
            neither of the two recognized strings.
    :returns: ``quantized_model`` with the block's quantized weight
            initializers (and, if ``learn_scales``, their scale
            initializers; if ``learn_activation_scales``, the activation
            scale and zero-point initializers too) rewritten. Every other
            byte of the model is untouched.
    :raises ValueError: if the block cannot be discovered, is not closed at
            statically-known shapes, contains no layer of the targeted
            scheme to train (including the scheme mismatch
            ``learn_activation_scales`` can be asked for), or (with
            ``batch_size`` set) has captured tensors that disagree about how
            many rows they have
    :raises onnxsim.graph_grad.UnsupportedOpError: if any node in the block
            has no gradient rule
    """
    _refuse_quantizer_flags_without_fake_quant(
        fake_quant, learn_scales, learn_activation_scales
    )
    _refuse_unknown_optimizer(optimizer)
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)

    plan = _plan_block(
        float_model,
        quantized_model,
        block_input_name,
        block_output_name,
        learn_activation_scales,
        fake_quant,
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
        learn_activation_scales=learn_activation_scales,
        activation_learning_rate=activation_learning_rate,
        lr_decay=lr_decay,
        batch_size=batch_size,
        shuffle=shuffle,
        batch_seed=batch_seed,
        step_providers=step_providers,
        losses=losses,
        fake_quant=fake_quant,
        preserve_sparsity=preserve_sparsity,
        optimizer=optimizer,
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
    learn_activation_scales: bool = False,
    fake_quant: bool = True,
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
            one block per gap between undifferentiable ops -- and on a model
            with no such gap, one block spanning the whole graph, which is
            the end-to-end objective itself rather than a surrogate for it.
            This module's docstring records why that is reachable but not
            the default.
    :param learn_activation_scales: plan for :func:`apply_qat`'s
            activation-quantization mode, i.e. count
            :func:`onnxsim.quantize_static` QDQ layers as the quantized ones
            rather than ``quantize_weight_only_int4`` INT4 layers. It has to
            be said here as well as at training time because "which layers
            are quantized" is what closes a block: the same graph partitions
            differently under the two schemes, and a plan made for one is
            not a plan for the other.
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
        for c in _find_layers(
            float_model, quantized_model, learn_activation_scales, fake_quant
        )
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
            plan = _plan_block(
                float_model,
                quantized_model,
                input_name,
                output_name,
                learn_activation_scales,
                fake_quant,
            )
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
    learn_activation_scales: bool = False,
    activation_learning_rate: float = 1e-2,
    lr_decay: bool = True,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    batch_seed: int = 0,
    sequential: bool = True,
    max_layers_per_block: int = 2,
    providers: Optional[Sequence[backend.Provider]] = None,
    step_providers: Optional[Sequence[backend.Provider]] = None,
    fake_quant: bool = True,
    preserve_sparsity: bool = False,
    optimizer: str = "adam",
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
    :param learn_activation_scales: train each block's activation quantizers
            alongside its weights, which also selects
            :func:`onnxsim.quantize_static`'s QDQ scheme as the target -- see
            :func:`apply_qat`, where the whole decision is written down. It is
            passed to :func:`discover_qat_blocks` as well when ``blocks`` is
            not given, since which layers count as quantized is what closes a
            block. This mode is where ``sequential=True`` earns the most: a
            clip range fitted to the teacher's activation distribution is a
            worse fit for the student's than a weight is, so re-running the
            student before each block matters more here than in the
            weight-only case.
    :param activation_learning_rate: Adam learning rate for those activation
            scales and zero-points
    :param lr_decay: anneal every learning rate to zero within each block
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
    :param optimizer: which optimizer trains each block's own weight --
            ``"adam"`` (the default) or ``"sgd_momentum"``, applied
            identically to every block in the walk. Scoped to the weight
            alone, exactly as in :func:`apply_qat`: with ``learn_scales``
            and/or ``learn_activation_scales`` also on, every block's scale
            and activation-quantizer parameters still train with Adam
            regardless of this choice -- see :func:`apply_qat`'s own
            ``optimizer`` paragraph for why that boundary was drawn where it
            was. Raises :class:`ValueError` if ``optimizer`` is neither of
            the two recognized strings.
    :returns: ``(tuned model, one QATBlockResult per discovered block)``. The
            model is ``quantized_model`` with every successfully trained
            block's initializers rewritten and nothing else touched; if no
            block trained it is an unmodified copy.
    """
    _refuse_quantizer_flags_without_fake_quant(
        fake_quant, learn_scales, learn_activation_scales
    )
    _refuse_unknown_optimizer(optimizer)
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)

    if blocks is None:
        blocks = discover_qat_blocks(
            float_model,
            quantized_model,
            max_layers_per_block=max_layers_per_block,
            learn_activation_scales=learn_activation_scales,
            fake_quant=fake_quant,
        )
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )

    # Plans are built once, against the *original* quantized model. Training a
    # block rewrites initializer payloads and never the graph, so a plan --
    # which is nodes, tensor names and candidate metadata -- stays valid for
    # the whole walk. No plan depends on the tuned state either: under QAT the
    # master weights are seeded from the float model (see
    # :func:`_plan_trained`), and with ``fake_quant=False`` they are seeded
    # from the student's own weights, which for a block about to be trained
    # are still the ones this plan captured -- each block is trained exactly
    # once, so nothing has rewritten them yet.
    plans: List[Optional[_BlockPlan]] = []
    results: List[QATBlockResult] = []
    for block in blocks:
        try:
            plans.append(
                _plan_block(
                    float_model,
                    quantized_model,
                    block.input_name,
                    block.output_name,
                    learn_activation_scales,
                    fake_quant,
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
                learn_activation_scales=learn_activation_scales,
                activation_learning_rate=activation_learning_rate,
                lr_decay=lr_decay,
                batch_size=batch_size,
                shuffle=shuffle,
                batch_seed=batch_seed,
                step_providers=step_providers,
                losses=result.losses,
                fake_quant=fake_quant,
                preserve_sparsity=preserve_sparsity,
                optimizer=optimizer,
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


def apply_block_finetune(
    reference_model: Union[str, onnx.ModelProto],
    model: Union[str, onnx.ModelProto],
    block_input_name: str,
    block_output_name: str,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 1000,
    learning_rate: float = 1e-4,
    lr_decay: bool = True,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    batch_seed: int = 0,
    providers: Optional[Sequence[backend.Provider]] = None,
    step_providers: Optional[Sequence[backend.Provider]] = None,
    losses: Optional[List[float]] = None,
    preserve_sparsity: bool = False,
) -> onnx.ModelProto:
    """Fine-tunes one block's float weights against a *reference* model's own
    output for that block -- :func:`apply_qat` with the quantizer taken out of
    the middle.

    Everything that made QAT here work is about distillation rather than about
    quantization: a block, a teacher's activation at its output, a mean
    squared reconstruction error, a backward pass emitted as ONNX, an Adam
    step. Quantization only ever entered at one point -- the fake-quantizer
    between the master weight and the block's node -- and removing it leaves
    ordinary block-wise fine-tuning. No labels, no loss function to choose, no
    training framework: the same step graph, run the same way, on whatever
    execution provider ``step_providers`` names.

    **The two models must differ, or there is nothing to learn.** The loss is
    the student block's output against the reference's, so a student that *is*
    the reference starts at zero loss and stays there. This is for the case
    where something has already changed the model and the change cost
    accuracy:

    - a **structurally** pruned model
      (:func:`onnxsim.apply_structured_pruning`,
      :func:`onnxsim.apply_attention_head_pruning` and friends), whose
      remaining weights can absorb some of what the removed ones did.
      *Unstructured* pruning
      (:func:`onnxsim.apply_magnitude_pruning`,
      :func:`onnxsim.apply_wanda_pruning`) is the case to avoid rather than
      the case to reach for: nothing here masks the optimizer, so every zero
      pruning left behind gets a gradient like any other element and is
      filled back in on the first step. Measured at 50% sparsity: 128 zeros
      per weight before, none after, while the loss fell five orders of
      magnitude -- so no signal a caller would look at says anything went
      wrong. See ``test_unstructured_sparsity_is_not_preserved``;
    - a model whose weights were quantized and dequantized back to fp32, or
      rewritten by any of this package's rounding passes;
    - a model already fine-tuned once, being tuned further against the
      original.

    It is *not* a way to fine-tune on new data or a new task: the objective is
    "reproduce what the reference model produced", which by construction
    cannot exceed the reference. Fitting a different target needs a different
    loss, and this module has one loss.

    **Against :func:`onnxsim.apply_pruning_finetune`,** which solves the same
    kind of problem and should usually be tried first for pruning. That one
    fits each layer *individually* and in *closed form* -- one ridge
    regression, one linear solve, no iteration, no learning rate, and an
    exactness argument this has no equivalent of. It is strictly better
    wherever it applies. What it cannot do is what a block buys: a layer
    pruned on both its input and its output channels at once is outside its
    channel-correspondence reconstruction and it declines to touch it, and a
    per-layer least-squares fit cannot let two layers with a nonlinearity
    between them trade error off against each other, because that objective
    is not a linear least-squares problem at all. This is the general,
    slower, weaker-guarantee alternative for those cases -- and, unlike a
    numpy solve, it is a step graph, so it runs on whatever
    ``step_providers`` names.

    :param reference_model: the teacher (onnx ModelProto or file path). Its
            activation at ``block_output_name`` is the only target. Its
            weights are *not* used to seed anything -- unlike
            :func:`apply_qat`, where the student's weights are a lossy
            encoding of the teacher's and so seeding from the teacher is the
            warm start. Here the student's weights are the starting point
            precisely because they are not the teacher's.
    :param model: the model being tuned (onnx ModelProto or file path). Every
            MatMul/Gemm in the block whose weight is a 2-D fp32 initializer is
            trained; the rest of the block is left alone, and a block with
            none of them is an error rather than a no-op. Assumed to have
            ``reference_model``'s topology and tensor names -- true of a
            pruned or requantized model, and the same assumption
            :func:`apply_qat` makes.
    :param block_input_name: the activation entering the block
    :param block_output_name: the block's own final output, the tensor whose
            reconstruction error is the loss
    :param calibration_data: representative input batches; see
            :func:`apply_qat`, whose meaning is unchanged
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for that random calibration data
    :param num_iterations: Adam steps to run over the block
    :param learning_rate: Adam learning rate for the weights. The default is
            :func:`apply_qat`'s, and is a starting point rather than a
            recommendation: QAT's objective is piecewise constant in the
            master weights (the loss only moves when an element crosses a
            rounding boundary), and this one is not, so a rate tuned for one
            is not tuned for the other.
    :param lr_decay: anneal the learning rate linearly to zero across the run
    :param batch_size: rows of the calibration set each step trains on;
            ``None`` is full batch. See :func:`apply_qat`.
    :param shuffle: draw a fresh permutation of the rows each epoch
    :param batch_seed: seed for that shuffling
    :param providers: onnxruntime execution providers for capturing the
            reference's activations
    :param step_providers: onnxruntime execution providers to run the
            optimization itself on
    :param losses: when given, the reconstruction loss is appended once per
            step
    :param preserve_sparsity: hold every weight element that starts at zero
            at zero for the whole run. Off by default, and it is the caller's
            call rather than a detected one, because "this element is zero" and
            "this element was pruned away" are the same bit pattern and only
            the caller knows which they meant.

            **Turn it on for an unstructured-pruned model.** Nothing else in
            the step graph masks the optimizer, so without it a pruned zero
            gets a gradient like any other element and leaves zero on the very
            first step: measured at 50% sparsity, 128 zeros per weight before
            and none after, while the loss fell five orders of magnitude. Every
            signal a caller would look at says that run went well, which is
            what makes it worth a parameter rather than a note.

            Structured pruning needs nothing here -- there the channel is gone
            from the tensor rather than zeroed inside it.

            The mask is the zero pattern of the weight the optimizer *starts*
            from, which is the master weight's seed: the student's own weight
            when fine-tuning, and the float model's when quantizing (so for
            QAT over a pruned model, prune first and pass the pruned model as
            ``float_model``, which is the ordinary order anyway). Elements it
            holds are held exactly, not approximately -- see
            :func:`_build_step_graph` for why one ``Mul`` is enough.
    :returns: ``model`` with the block's weight initializers rewritten. Every
            other byte is untouched.
    :raises ValueError: if the block cannot be discovered, is not closed at
            statically-known shapes, or contains no MatMul/Gemm with a 2-D
            fp32 weight initializer
    :raises onnxsim.graph_grad.UnsupportedOpError: if any node in the block
            has no gradient rule
    """
    return apply_qat(
        reference_model,
        model,
        block_input_name,
        block_output_name,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        batch_size=batch_size,
        shuffle=shuffle,
        batch_seed=batch_seed,
        providers=providers,
        step_providers=step_providers,
        losses=losses,
        fake_quant=False,
        preserve_sparsity=preserve_sparsity,
    )


def apply_block_finetune_all_blocks(
    reference_model: Union[str, onnx.ModelProto],
    model: Union[str, onnx.ModelProto],
    blocks: Optional[Sequence[QATBlock]] = None,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 1000,
    learning_rate: float = 1e-4,
    lr_decay: bool = True,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    batch_seed: int = 0,
    sequential: bool = True,
    max_layers_per_block: int = 2,
    providers: Optional[Sequence[backend.Provider]] = None,
    step_providers: Optional[Sequence[backend.Provider]] = None,
    preserve_sparsity: bool = False,
) -> Tuple[onnx.ModelProto, List[QATBlockResult]]:
    """:func:`apply_block_finetune` lifted to the whole model, exactly as
    :func:`apply_qat_all_blocks` lifts :func:`apply_qat`.

    The sequential default matters more here than it does under QAT, and for
    the same reason it matters at all: block *k* is fed the *student's* own
    activation, so it sees the error the earlier blocks left and can spend its
    capacity on it. Under QAT that error is quantization noise; here it is
    whatever the change to the model actually cost -- a pruned channel's
    absence, say -- which is exactly what a later block would otherwise never
    learn about.

    Discovery, per-block failure handling and the returned results are
    :func:`apply_qat_all_blocks`'s, unchanged. A block with no trainable
    MatMul/Gemm is skipped with a reason rather than raising.

    See :func:`apply_block_finetune` for what this is for, when the two models
    are different enough for there to be anything to learn, and how it
    compares with :func:`onnxsim.apply_pruning_finetune`.
    """
    return apply_qat_all_blocks(
        reference_model,
        model,
        blocks=blocks,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        batch_size=batch_size,
        shuffle=shuffle,
        batch_seed=batch_seed,
        sequential=sequential,
        max_layers_per_block=max_layers_per_block,
        providers=providers,
        step_providers=step_providers,
        fake_quant=False,
        preserve_sparsity=preserve_sparsity,
    )
