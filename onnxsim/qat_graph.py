"""One optimizer step, expressed as an ONNX graph, so it can run wherever an
ONNX model can run -- CPU, CUDA, an NPU execution provider, or WebGPU in a
browser -- instead of only in host numpy.

Six passes in this repo (:mod:`onnxsim.adaround`, :mod:`onnxsim.adaquant`,
:mod:`onnxsim.brecq`, :mod:`onnxsim.flexround`, :mod:`onnxsim.autoround`,
:mod:`onnxsim.omniquant`) already optimize a quantization parameter by Adam
with hand-derived, straight-through gradients. Every one of them runs that
loop in host numpy: their ``providers=`` argument reaches only the *capture*
of calibration activations from the float model, never the optimization
itself. That is the whole gap this module closes, and the reason it can be
closed cheaply is the "hand-derived" part:

    **A hand-derived backward pass is ordinary dataflow.** There is no tape,
    no autograd framework, and no ``Gradient`` operator involved -- just
    ``MatMul``/``Mul``/``Sub``/``Sigmoid``/``Clip``/``Sqrt`` over tensors. So
    it is expressible as a plain ONNX *inference* graph, and any runtime that
    can do inference can therefore train these parameters.

A **step graph** is that expression: a pure function

.. code-block:: text

    (fixed constants, mutable state, per-step scalars) -> (next state, loss)

with the optimizer's own state (Adam's two moments) carried in and out as
tensors rather than held in Python. Running it ``N`` times, feeding each
call's state outputs back in as the next call's state inputs
(:func:`run_step_graph`), *is* the optimization loop -- and it happens
wherever the execution provider says.

**What this buys, concretely.** :class:`onnxsim.backend.Runner` binds the
graph and a provider list once, so the same builder reaches
``CUDAExecutionProvider`` or an NPU EP (QNN, Core ML, OpenVINO -- see the
harnesses under ``scripts/``) from Python, and in the browser the WASM
build's model-executor trampoline (``docs/wasm_ort_web.md``) hands the same
graph to onnxruntime-web, whose provider list already offers ``webgpu`` and
WebNN's ``gpu``/``npu`` device types (``docs/webnn.md``).

The tensors also *stay* where that provider put them. :func:`run_step_graph`
binds the graph's constants and its state through onnxruntime's ``IOBinding``
(``bind_state=True``, the default, via
:meth:`onnxsim.backend.Runner.bind_loop`): the calibration activations are
uploaded once at setup, each step's state outputs become the next step's state
inputs without a round trip through host numpy, and only the per-step scalars
go up and the loss comes down. On CPU that saves a memcpy or two; on a
provider whose bus is PCIe it is the difference between a loop that is
arithmetic and a loop that is transfers.

**Minibatching, and where the batch lives.** A step graph's shapes are
static, so "the batch" is a fixed-size tensor either way; the question a
minibatched loop has to answer is where the *rest* of the data sits between
steps. Two answers, and this module implements the first:

- **The whole set stays resident and the graph selects rows.** The
  calibration set is one constant, uploaded once by ``bind_loop`` exactly as
  before, and each step feeds a rank-1 int64 index vector that a ``Gather``
  turns into that step's rows (:meth:`GraphBuilder.gather_rows`,
  :func:`minibatch_indices`). Per-step traffic stays at "a few scalars up, a
  loss down" -- ``batch_size`` int64s is not a transfer -- so the residency
  argument above survives a batch that changes every step. What it costs:
  one operator in :data:`EP_FRIENDLY_OPS` that is not arithmetic (see the
  note there for why ``Gather`` clears that bar where ``Round`` does not),
  a few nodes of graph, and the fact that the *set* must still fit in one
  static tensor on the device. Minibatching this way buys per-step compute
  and stochastic-gradient behaviour, not a larger-than-memory dataset.
- **The batch's rows are fed per step.** No new operator, no resident set,
  and no limit at all on how much data a run may stream -- but
  ``batch_size x width`` floats cross the bus every step, which is the exact
  cost :meth:`onnxsim.backend.Runner.bind_loop` was written to remove. The
  seam is open (``run_step_graph``'s ``feeds``, which is what carries the
  index vector in the chosen design), and its docstring records when this
  second answer would be the right one; a calibration set is small and a
  PCIe round trip per step is not, so it is not the default.

**What it does not buy, and the honest limits.**

- *Residency is the Python half only.* ``IOBinding`` covers onnxruntime in
  Python. The browser half of the same idea -- ORT-web's GPU-buffer tensors
  with ``preferredOutputLocation: "gpu-buffer"``, fed straight back in as the
  next step's inputs -- is still to do, so the WASM path re-sends its feeds
  every step.
- *Precision.* Step graphs are built in float32, not the float64 the numpy
  loops use: fp64 is what accelerators do not have. Results therefore agree
  with the numpy path closely rather than bit-exactly.
- *Determinism.* A non-CPU provider reassociates reductions, so a trained
  result is not reproducible the way ``tests/test_constant_fold_determinism.py``
  requires of folding. CPU stays the default everywhere in-tree.

See ``docs/qat.md`` for how this fits the larger QAT picture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx

from onnxsim import backend

# Same opset/IR pairing every other graph builder in this package uses (see
# onnxsim/gguf_reconstruct.py): opset 17 with the IR version opset 17 actually
# requires, rather than whatever the installed onnx's newest opset needs.
_OPSET = 17
_IR_VERSION = 8

# The operator set a step graph restricts itself to.
#
# The point of expressing a training step as an ONNX graph is that it runs
# wherever inference runs -- including onnxruntime-web's WebGPU backend and
# the WebNN/NPU execution providers, which implement far less than the full
# ONNX operator set. A step graph that reached for a convenient operator one
# of those cannot run would pass every numerical test and still be useless
# for the thing this module exists for, so the ops a builder may emit are
# pinned here and asserted in tests (``tests/test_qat_graph.py``,
# ``tests/test_graph_grad.py``, ``tests/test_adaquant_step_graph.py``).
#
# What is deliberately absent: control flow, boolean logic ops (a mask is a
# float 0/1 from ``Cast(Greater(...))``, multiplied in), ``Where``,
# ``Expand``, and ``Round`` -- which :mod:`onnxsim.adaquant` composes out of
# ``Sign``/``Abs``/``Cast`` instead. That composition was written when WebNN
# had no rounding operator at all; it has since gained ``roundEven``, and
# onnxruntime-web's WebNN EP maps ``Round`` onto it, so the reason has
# weakened rather than held. See the ``Conv`` note below, which is where that
# was re-checked and what it was checked against.
# Adding to this set is a real decision: check the operator actually has
# coverage on the WebGPU and WebNN backends first, not just on ORT's CPU
# kernels.
#
# ``Gather`` is the one addition made for a reason other than arithmetic, and
# it earns its place on exactly that criterion rather than on convenience.
# It is what lets a step graph read a *minibatch* out of a resident
# calibration set (:meth:`GraphBuilder.gather_rows`, and see
# :func:`minibatch_indices` for the loop around it): the whole set is uploaded
# once as a constant and the step selects rows from it by an index vector, so
# nothing large crosses the bus per step. The coverage is real on both
# backends this list exists for -- WebNN specifies ``gather`` (unlike
# ``round``, which it has no operator for at all, the case that shaped the
# rest of this list), and onnxruntime-web's WebGPU EP implements ``Gather``;
# it is also among the first ops every NPU EP in ``scripts/`` supports, since
# embedding lookup is inference, not training. The alternative that needed no
# new operator -- feeding the batch's rows themselves as a per-step input --
# is rejected in :func:`run_step_graph`'s own docstring, where the trade-off
# it loses is spelled out.
#
# ``Conv`` and ``ConvTranspose`` were considered for this set and deliberately
# left out, which is worth recording because the case for them looks strong:
# :func:`onnxsim.graph_grad._grad_conv` differentiates a convolution, and the
# textbook way to write that backward is a ``ConvTranspose`` for ``dX`` and a
# ``Conv`` over permuted axes for ``dW``. Both operators do have coverage --
# onnxruntime-web's WebGPU EP lists ``Conv`` and ``ConvTranspose``, and its
# WebNN EP maps them onto WebNN's ``conv2d``/``convTranspose2d`` -- so the
# membership test above is not what refuses them. What refuses them is the
# shape of that coverage:
#
# - It is 2-D only, on *both* backends. The WebGPU EP's own operator table
#   annotates ``Conv`` with "conv3d is not supported" and ``ConvTranspose``
#   with "ConvTranspose3d is not supported"; the WebNN EP's table restricts
#   both to "3-D or 4-D input and 'W'", and the WebNN specification defines
#   ``conv2d`` and ``convTranspose2d`` and no other convolution at all
#   ("Compute a 2-D convolution given 4-D input and filter tensors"). A rule
#   emitting them would therefore be a rule that runs for a 2-D convolution
#   and is dead in the browser for a 3-D one, which is precisely the failure
#   this list exists to prevent -- and a *silent* one, since the graph would
#   still be valid ONNX and still run on CPU.
# - ``dW``-as-a-``Conv`` is not attribute-for-attribute the forward: it swaps
#   strides with dilations and, when the stride does not divide the input,
#   needs its result cropped -- a ``Slice``, which is a second new member.
#   ``dX``-as-a-``ConvTranspose`` needs ``output_shape`` for the same reason.
#
# So ``_grad_conv`` is written as im2col instead -- one ``Gather`` with a
# constant index, one ``Mul`` by a 0/1 mask, one ``MatMul`` -- which adds
# nothing to this set, is rank-agnostic (1-D, 2-D and 3-D convolutions
# differentiate identically), and stays inside the ranks the WebNN spec
# *requires* implementations to support: its ``gather`` states allowed input
# rank 1 to N with 1 to 5 required, and its ``matmul`` "2 to N" with "2 to 5"
# required, against the rank-4 tensors that rule emits. What it costs instead
# is memory -- a materialized index table per convolution -- which is stated
# where the rule pays it rather than here.
#
# Sources checked when this was written (September 2026), rather than
# remembered: onnxruntime's ``js/web/docs/webgpu-operators.md`` and
# ``js/web/docs/webnn-operators.md`` on ``main``, and the WebNN
# specification's operator definitions. One thing they say that the paragraph
# above this one no longer does: WebNN today *has* gained a rounding
# operator (``roundEven``, which the WebNN EP maps ``Round`` onto). The
# composition in :mod:`onnxsim.adaquant` is not therefore wrong, but its
# reason has weakened, and this note is here so the next person to reach for
# ``Round`` re-checks rather than trusting the older sentence.
EP_FRIENDLY_OPS = frozenset(
    {
        "Abs",
        "Add",
        "Cast",
        "Clip",
        "Div",
        "Exp",
        "Gather",
        "Greater",
        "Less",
        "MatMul",
        "Mul",
        "Neg",
        "Pow",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Sigmoid",
        "Sign",
        "Sqrt",
        "Sub",
        "Transpose",
    }
)

# Adam's own standard hyper-parameters, matching the hand-rolled loops in
# adaround.py/adaquant.py/brecq.py exactly so a ported loop keeps its
# behaviour.
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


class GraphBuilder:
    """Accumulates nodes and initializers with unique names.

    Exists so a hand-derived gradient reads like the expression it is --
    ``g = b.mul(b.sub(y_hat, y), two_over_n)`` -- rather than like a pile of
    ``onnx.helper.make_node`` calls with hand-managed intermediate names.
    """

    def __init__(self, prefix: str = "") -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.initializer: List[onnx.TensorProto] = []
        self._prefix = prefix
        self._counter = 0

    def name(self, hint: str = "t") -> str:
        self._counter += 1
        return f"{self._prefix}{hint}_{self._counter}"

    def const(self, value, hint: str = "c") -> str:
        """A float32 initializer holding ``value`` (scalar or array)."""
        array = np.asarray(value, dtype=np.float32)
        name = self.name(hint)
        self.initializer.append(onnx.numpy_helper.from_array(array, name))
        return name

    def op(self, op_type: str, inputs: Sequence[str], hint: str = "", **attrs) -> str:
        out = self.name(hint or op_type.lower())
        self.nodes.append(onnx.helper.make_node(op_type, list(inputs), [out], **attrs))
        return out

    # The handful of operators the hand-derived gradients below actually use.
    # Deliberately kept to ops with broad execution-provider coverage: no
    # boolean logic ops (a mask is a float 0/1 from Cast(Greater), multiplied
    # in) and no Where, both of which are patchier on the WebNN/NPU backends
    # than plain arithmetic is.
    def add(self, a: str, b: str) -> str:
        return self.op("Add", [a, b])

    def sub(self, a: str, b: str) -> str:
        return self.op("Sub", [a, b])

    def mul(self, a: str, b: str) -> str:
        return self.op("Mul", [a, b])

    def div(self, a: str, b: str) -> str:
        return self.op("Div", [a, b])

    def matmul(self, a: str, b: str) -> str:
        return self.op("MatMul", [a, b])

    def transpose(self, a: str, perm: Optional[Sequence[int]] = None) -> str:
        if perm is None:
            return self.op("Transpose", [a])
        return self.op("Transpose", [a], perm=list(perm))

    def sqrt(self, a: str) -> str:
        return self.op("Sqrt", [a])

    def sigmoid(self, a: str) -> str:
        return self.op("Sigmoid", [a])

    def clip(self, a: str, low: float, high: float) -> str:
        return self.op("Clip", [a, self.const(low), self.const(high)])

    def greater_mask(self, a: str, threshold: float) -> str:
        """``(a > threshold)`` as a float32 0/1 tensor."""
        gt = self.op("Greater", [a, self.const(threshold)])
        return self.op("Cast", [gt], to=onnx.TensorProto.FLOAT)

    def less_mask(self, a: str, threshold: float) -> str:
        """``(a < threshold)`` as a float32 0/1 tensor."""
        lt = self.op("Less", [a, self.const(threshold)])
        return self.op("Cast", [lt], to=onnx.TensorProto.FLOAT)

    def round_to_nearest(self, a: str) -> str:
        """``round(a)``, composed rather than emitted as ``Round``.

        ``Round`` is deliberately absent from :data:`EP_FRIENDLY_OPS` -- and
        a fake-quant forward, which is what every caller of this wants it for,
        is exactly the code that must run on the accelerator backends. The
        original reason was that WebNN had no rounding operator at all, which
        is no longer true (it has ``roundEven``, and the WebNN EP maps
        ``Round`` onto it); see the ``Conv`` note beside
        :data:`EP_FRIENDLY_OPS` for when that was re-checked. The composition
        is kept because it works and is verified, not because the original
        argument still stands. A float-to-int32 ``Cast`` truncates toward
        zero, so truncating ``|a| + 0.5`` and re-applying the sign is
        round-half-away-from-zero. That differs from ``Round``'s (and numpy's)
        round-half-to-even on *exact* ties only: a value landing on a precise
        .5, which real calibration data does not do at any measurable rate,
        and which when it happens moves one element's code by one rather than
        changing the shape of the optimization.
        """
        magnitude = self.add(self.op("Abs", [a]), self.const(0.5))
        truncated = self.op("Cast", [magnitude], to=onnx.TensorProto.INT32)
        return self.mul(
            self.op("Sign", [a]),
            self.op("Cast", [truncated], to=onnx.TensorProto.FLOAT),
        )

    def mean_square(self, a: str) -> str:
        """``mean(a * a)`` as a scalar, for a reported loss."""
        sq = self.mul(a, a)
        return self.op("ReduceMean", [sq], keepdims=0)

    def gather_rows(self, table: str, index: str, out: Optional[str] = None) -> str:
        """Rows ``index`` of ``table``, i.e. ``table[index]`` along axis 0.

        The minibatching primitive. ``table`` is the whole calibration set,
        bound once as a step-graph constant and therefore resident on the
        execution provider's device for the life of the loop; ``index`` is a
        rank-1 int64 per-step input naming this step's rows. So what crosses
        the bus per step is ``batch_size`` 8-byte integers instead of
        ``batch_size x width`` floats, and the set is uploaded once rather
        than once per step -- the property :meth:`onnxsim.backend.Runner.bind_loop`
        exists to provide, preserved in the presence of a batch that changes
        every step.

        ``out`` names the output explicitly, for the caller who needs the
        gathered rows to carry a name some *other* nodes already read (in
        :mod:`onnxsim.qat`, the block's own input tensor: the block's nodes go
        into the graph verbatim, so the batch has to arrive under the name
        they were written against).
        """
        if out is None:
            return self.op("Gather", [table, index], "rows", axis=0)
        self.nodes.append(
            onnx.helper.make_node("Gather", [table, index], [out], axis=0)
        )
        return out


def adam_update(
    b: GraphBuilder,
    param: str,
    grad: str,
    m: str,
    v: str,
    lr: str,
    m_correction: str,
    v_correction: str,
    eps: float = ADAM_EPS,
) -> Tuple[str, str, str]:
    """Appends one Adam step to ``b`` and returns ``(param', m', v')``.

    ``m_correction``/``v_correction`` are the bias-correction *factors*
    ``1 / (1 - beta**t)``, passed in as scalars rather than derived from a step
    counter inside the graph: they are two host-side floats per step, so
    computing them outside costs nothing and keeps the graph free of the state
    that a ``Pow`` over a step counter would need.
    """
    beta1 = b.const(ADAM_BETA1)
    beta2 = b.const(ADAM_BETA2)
    one_minus_beta1 = b.const(1.0 - ADAM_BETA1)
    one_minus_beta2 = b.const(1.0 - ADAM_BETA2)

    m_next = b.add(b.mul(beta1, m), b.mul(one_minus_beta1, grad))
    v_next = b.add(b.mul(beta2, v), b.mul(one_minus_beta2, b.mul(grad, grad)))
    m_hat = b.mul(m_next, m_correction)
    v_hat = b.mul(v_next, v_correction)
    step = b.div(b.mul(lr, m_hat), b.add(b.sqrt(v_hat), b.const(eps)))
    param_next = b.sub(param, step)
    return param_next, m_next, v_next


@dataclass
class StepGraph:
    """A pure function performing one optimizer step.

    :param model: the graph itself. Its inputs are the fixed constants, the
            mutable state, and any per-step scalars; its outputs are the next
            state (and optionally a loss).
    :param state: ``{input name: output name}`` -- which output carries the
            next value of which input. :func:`run_step_graph` uses exactly this
            to close the loop.
    :param loss_name: an output holding a scalar loss, recorded per step when
            the caller asks for it. Optional: nothing in the loop needs it, it
            is for diagnostics.
    """

    model: onnx.ModelProto
    state: Dict[str, str]
    loss_name: Optional[str] = None


def make_step_graph(
    b: GraphBuilder,
    constants: Dict[str, Tuple[Sequence[int], int]],
    state: Dict[str, Tuple[Sequence[int], str]],
    scalars: Sequence[str] = (),
    loss: Optional[str] = None,
    name: str = "onnxsim_step",
    per_step: Optional[Dict[str, Tuple[Sequence[int], int]]] = None,
) -> StepGraph:
    """Wraps ``b``'s accumulated nodes into a :class:`StepGraph`.

    :param constants: ``{input name: (shape, onnx element type)}`` for the
            tensors that do not change across steps (calibration activations,
            the reconstruction target, a frozen scale, ...). Most of these are
            FLOAT -- an activation, a scale -- but a block-external tensor can
            be any dtype the source model gave it (a ``Gather``'s integer row
            ``indices``, captured whole as one of these when there is no
            minibatch), which is why this takes an element type per entry
            rather than assuming FLOAT for all of them the way it used to.
    :param state: ``{input name: (shape, output name)}`` for the tensors the
            step updates -- the parameter being optimized and the optimizer's
            own moments
    :param scalars: names of scalar (rank-0) per-step inputs, e.g. a learning
            rate or an annealed regularization weight
    :param loss: an optional scalar output name to expose as the loss
    :param per_step: ``{input name: (shape, onnx element type)}`` for per-step
            inputs that are neither float32 nor rank-0 -- in practice the
            int64 row index a minibatched loop feeds
            :meth:`GraphBuilder.gather_rows`. Kept separate from ``scalars``
            rather than generalizing it because the two are fed differently
            (``run_step_graph`` casts scalars to float32 and passes these
            through with the dtype the caller built them with) and because a
            scalar's shape and type never need saying.
    """
    inputs = [
        onnx.helper.make_tensor_value_info(n, elem_type, list(shape))
        for n, (shape, elem_type) in constants.items()
    ]
    inputs += [
        onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, list(shape))
        for n, (shape, _) in state.items()
    ]
    inputs += [
        onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [])
        for n in scalars
    ]
    inputs += [
        onnx.helper.make_tensor_value_info(n, elem_type, list(shape))
        for n, (shape, elem_type) in (per_step or {}).items()
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, list(shape))
        for _, (shape, out) in state.items()
    ]
    if loss is not None:
        outputs.append(
            onnx.helper.make_tensor_value_info(loss, onnx.TensorProto.FLOAT, [])
        )
    graph = onnx.helper.make_graph(
        b.nodes, name, inputs, outputs, initializer=b.initializer
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", _OPSET)]
    )
    model.ir_version = _IR_VERSION
    return StepGraph(
        model=model,
        state={n: out for n, (_, out) in state.items()},
        loss_name=loss,
    )


def _as_constant(v: np.ndarray) -> np.ndarray:
    """A step-graph constant's value, cast to float32 if it is
    floating-point and left alone otherwise.

    Every constant used to be forced to float32 unconditionally, back when
    every one of them was a float tensor. That is no longer true -- a
    :func:`onnxsim.qat.apply_qat` block containing a ``Gather`` can capture
    an integer ``indices`` tensor as one of these -- so only the
    floating-point ones (an activation, a scale, ...; also whichever numpy
    dtype a caller's own float64 host computation happened to produce, which
    this cast has always absorbed) are normalized to float32. A non-float
    array is passed through as the dtype it already is, matching the
    element type :func:`make_step_graph` declared its graph input as.
    """
    array = np.asarray(v)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32, copy=False)
    return array


def _run_bound_loop(
    bound: backend.BoundStepLoop,
    step: StepGraph,
    num_steps: int,
    step_feeds: Callable[[int], Dict[str, np.ndarray]],
    losses: Optional[List[float]],
) -> Optional[Dict[str, np.ndarray]]:
    """Drive ``bound`` for ``num_steps``, or return ``None`` if onnxruntime
    refuses to run the binding.

    Some execution providers accept a binding at setup time and only fail when
    a run actually reaches them, so "can this be bound?" is not fully knowable
    until the first ``run_with_iobinding``. Rather than leave the caller half
    way through a loop on a path that does not work, this reports the failure
    and lets :func:`run_step_graph` re-run the whole thing unbound: the step
    graph is a pure function of its state, so starting over from the same
    initial state reproduces exactly the same trajectory.

    ``losses`` is only extended once the whole loop has succeeded, so an
    abandoned attempt leaves no half-written diagnostics behind. The scalars
    callback is deliberately called *outside* the guarded region -- an
    exception from the caller's own code is the caller's bug, not a binding
    failure, and must not be swallowed into a silent fallback.
    """
    collected: List[float] = []
    want_loss = losses is not None and step.loss_name is not None
    for t in range(num_steps):
        feeds = step_feeds(t)
        try:
            out = bound.step(feeds)
        except Exception:
            return None
        if want_loss:
            collected.append(float(out[str(step.loss_name)]))
    if losses is not None:
        losses.extend(collected)
    return dict(bound.state())


def run_step_graph(
    step: StepGraph,
    constants: Dict[str, np.ndarray],
    state: Dict[str, np.ndarray],
    num_steps: int,
    scalars: Optional[Callable[[int], Dict[str, float]]] = None,
    providers: Optional[Sequence[backend.Provider]] = None,
    losses: Optional[List[float]] = None,
    bind_state: bool = True,
    feeds: Optional[Callable[[int], Dict[str, np.ndarray]]] = None,
) -> Dict[str, np.ndarray]:
    """Runs ``step`` ``num_steps`` times, threading its state through, and
    returns the final state.

    The session (and its execution providers) is created once for the whole
    loop, not once per step -- see :class:`onnxsim.backend.Runner`.

    :param constants: values for the step graph's constant inputs. A
            floating-point array is cast to float32 (a caller may hand this a
            float64 numpy computation and rely on that), but any other dtype
            -- in practice a ``Gather``'s captured integer ``indices``, when
            there is no minibatch -- is passed through as-is: casting it to
            float32 the way this used to unconditionally would silently
            corrupt it and then disagree with the dtype
            :func:`make_step_graph` declared that same input as.
    :param state: initial values for its state inputs
    :param num_steps: iterations to run
    :param scalars: called with the step index, returning that step's scalar
            inputs (learning rate, annealed regularization weight, Adam's bias
            corrections, ...). Adam's bias corrections are the reason this is a
            callback rather than a fixed dict: they change every step.
    :param providers: onnxruntime execution providers, in priority order, to
            run the step on. ``None`` means CPU.
    :param losses: when given, the step graph's loss output is appended to it
            once per step. Reading it back costs a scalar transfer per step.
    :param bind_state: keep the constants and the state resident on the
            execution provider's device across steps, via onnxruntime's
            ``IOBinding`` (:meth:`onnxsim.backend.Runner.bind_loop`), instead
            of re-sending every tensor as a feed on every step. This is the
            follow-up this module's docstring names, and it is on by default
            because it is what makes a non-CPU provider worth using: the
            constants go up once, the state never comes down between steps,
            and only the per-step scalars and (if asked for) the loss cross
            the bus. It is transparent -- the same numbers come back either
            way, and anything that stops the binding from working falls back
            to the feed-per-step path on its own, so a caller never has to
            know which one ran.

            Turn it off to force the feed-per-step path: when debugging a
            provider whose binding support is suspect and the unbound result
            is the reference to compare against, when a profiler's per-step
            input/output attribution is more useful than the speed, or when
            the extra device buffer per state tensor (binding double-buffers
            the state, see :class:`onnxsim.backend.BoundStepLoop`) is not
            affordable.
    :param feeds: called with the step index, returning that step's *tensor*
            inputs -- the ones declared through ``make_step_graph``'s
            ``per_step``, whose dtype is preserved rather than cast to
            float32. This is the seam a per-step minibatch goes through, and
            which of the two shapes of minibatching a caller gets depends
            entirely on what it puts here:

            - **An index vector into a resident set** (what
              :func:`onnxsim.apply_qat` does, via
              :meth:`GraphBuilder.gather_rows` and
              :func:`minibatch_indices`). The whole calibration set stays in
              ``constants``, uploaded once and never re-sent; the step graph
              selects its own rows. Per-step traffic is ``batch_size``
              int64s, which is nothing, and the residency this module is
              built around survives a batch that changes every step. The
              price is that the *set* still has to fit in one static tensor
              on the device: minibatching this way buys per-step compute
              (a step's FLOPs scale with the batch, not with the set) and
              stochastic-gradient behaviour, not a larger-than-memory set.
            - **The batch's rows themselves.** Then ``constants`` holds
              nothing large, the set can live in host memory or be streamed
              from disk, and the cap on how much data a run may use goes
              away entirely -- at the cost of copying ``batch_size x width``
              floats to the device on every single step, which is precisely
              the transfer :meth:`onnxsim.backend.Runner.bind_loop` exists to
              remove, and which on a PCIe bus turns the loop back into
              transfers. Nothing here forbids it; it is simply not what
              :mod:`onnxsim.qat` chose, because a calibration set is small
              and a PCIe round trip per step is not.
    """
    fetch = list(step.state.values())
    if losses is not None and step.loss_name is not None:
        fetch.append(step.loss_name)
    runner = backend.Runner(step.model, output_names=fetch, providers=providers)

    fixed = {k: _as_constant(v) for k, v in constants.items()}
    initial = {k: np.asarray(v, dtype=np.float32) for k, v in state.items()}

    def step_feeds(t: int) -> Dict[str, np.ndarray]:
        values: Dict[str, np.ndarray] = {}
        if scalars is not None:
            values.update(
                {k: np.asarray(v, dtype=np.float32) for k, v in scalars(t).items()}
            )
        if feeds is not None:
            # Passed through with the dtype the caller chose: an index vector
            # is int64, and casting it to float32 the way the scalars above
            # are cast would make it the wrong type for the graph input it
            # feeds.
            values.update({k: np.asarray(v) for k, v in feeds(t).items()})
        return values

    # A caller who left out one of the graph's state inputs gets the unbound
    # path's error about a missing feed, not a KeyError from the setup below.
    if bind_state and set(step.state) <= set(initial):
        bound = runner.bind_loop(
            fixed, {name: (out, initial[name]) for name, out in step.state.items()}
        )
        if bound is not None:
            final = _run_bound_loop(bound, step, num_steps, step_feeds, losses)
            if final is not None:
                return final

    current = dict(initial)
    for t in range(num_steps):
        inputs = dict(fixed)
        inputs.update(current)
        inputs.update(step_feeds(t))
        out = runner(inputs)
        current = {name: out[output] for name, output in step.state.items()}
        if losses is not None and step.loss_name is not None:
            losses.append(float(out[step.loss_name]))
    return current


def adam_bias_corrections(t: int) -> Dict[str, float]:
    """Adam's two bias-correction factors at (0-based) step ``t``, named the
    way :func:`adam_update`'s callers wire them: ``{"m_correction": ...,
    "v_correction": ...}``."""
    return {
        "m_correction": 1.0 / (1.0 - ADAM_BETA1 ** (t + 1)),
        "v_correction": 1.0 / (1.0 - ADAM_BETA2 ** (t + 1)),
    }


def minibatch_indices(
    num_rows: int,
    batch_size: int,
    seed: int = 0,
    shuffle: bool = True,
) -> Callable[[int], np.ndarray]:
    """The row indices step ``t`` should train on, as a pure function of ``t``.

    Feed the result to :func:`run_step_graph`'s ``feeds`` and the graph's own
    :meth:`GraphBuilder.gather_rows` and the loop is minibatched: the whole
    calibration set stays resident as a constant, and each step reads
    ``batch_size`` of its rows.

    Three decisions live here, and each of them is a decision rather than an
    accident:

    **The stream is an endless concatenation of permutations, chopped into
    fixed-size chunks.** Step ``t`` gets stream positions
    ``[t*B, (t+1)*B)``; position ``p`` is row ``perm[p // N][p % N]``, where
    ``perm[e]`` is epoch ``e``'s ordering of the ``N`` rows. So every row is
    visited exactly once per ``N`` positions consumed, no matter where the
    batch boundaries fall.

    **What happens when the batch size does not divide the row count**: the
    batch that would run off the end of an epoch is *completed from the front
    of the next epoch's permutation*, rather than being short. It has to be:
    a step graph's shapes are static -- that is the property that lets it
    compile for an NPU and lets :meth:`onnxsim.backend.Runner.bind_loop`
    pre-allocate its buffers -- so a ragged final batch would need a second
    graph, and every alternative that keeps one graph is worse. Dropping the
    tail (PyTorch's ``drop_last``) would silently discard up to
    ``batch_size - 1`` rows per epoch, and *systematically* the same rows
    whenever ``shuffle=False``. Padding by repeating a row would quietly
    reweight it. Wrapping keeps every row's visit count equal and every batch
    full; its only cost is that a straddling batch can contain the same row
    twice, when that row lands near the end of one permutation and the start
    of the next, which weights it double in that one step and not at all
    thereafter.

    **Shuffling and the seed.** ``shuffle=True`` draws a fresh permutation per
    epoch, which is what makes consecutive steps see different rows rather
    than the same ``B`` rows forever -- without it, an epoch's batches are
    the same fixed partition every time round and the "stochastic" in
    stochastic gradient descent is only the partition, never the composition.
    The permutation for epoch ``e`` is drawn from ``(seed, e)`` alone, not
    from a running generator, so this stays a pure function of ``t``: two runs
    with the same seed see identical batches, and (like the step graph itself)
    a loop can be stopped after ``N`` steps and resumed at step ``N`` without
    the batch order shifting. ``shuffle=False`` walks the rows in order, which
    is what a test that wants to see the batch composition directly, or a
    caller whose rows are already in a meaningful order, should use.

    :param num_rows: rows in the resident calibration set
    :param batch_size: rows per step. Larger than ``num_rows`` is allowed and
            simply means the wrap repeats rows within one batch, but callers
            normally clamp to the set size instead -- at that point the batch
            *is* the set and the full-batch path is both cheaper and exact.
    :param seed: seed for the per-epoch permutations
    :param shuffle: draw a new permutation per epoch instead of walking the
            rows in order
    """
    if num_rows < 1:
        raise ValueError(f"num_rows must be at least 1, got {num_rows}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    # Regenerating a permutation is O(num_rows) and a step is far more than
    # that, but a batch straddling an epoch boundary asks for two of them, so
    # the last two are kept rather than redrawn twice per step.
    cache: Dict[int, np.ndarray] = {}

    def permutation(epoch: int) -> np.ndarray:
        cached = cache.get(epoch)
        if cached is not None:
            return cached
        if shuffle:
            # Seeded by (seed, epoch) rather than advanced from a running
            # generator: that is what keeps this a pure function of the step
            # index.
            drawn = np.random.default_rng([seed, epoch]).permutation(num_rows)
        else:
            drawn = np.arange(num_rows)
        for stale in [e for e in cache if e < epoch - 1]:
            del cache[stale]
        cache[epoch] = drawn
        return drawn

    def rows(t: int) -> np.ndarray:
        indices = np.empty(batch_size, dtype=np.int64)
        filled = 0
        while filled < batch_size:
            epoch, offset = divmod(t * batch_size + filled, num_rows)
            take = min(num_rows - offset, batch_size - filled)
            indices[filled : filled + take] = permutation(epoch)[offset : offset + take]
            filled += take
        return indices

    return rows
