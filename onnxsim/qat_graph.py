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

# Adam's own standard hyper-parameters, matching the hand-rolled loops in
# adaround.py/adaquant.py/brecq.py exactly so a ported loop keeps its
# behaviour.
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
# ``Expand``, and ``Round`` -- WebNN has no rounding operator at all, which is
# why :mod:`onnxsim.adaquant` composes one out of ``Sign``/``Abs``/``Cast``.
# Adding to this set is a real decision: check the operator actually has
# coverage on the WebGPU and WebNN backends first, not just on ORT's CPU
# kernels.
EP_FRIENDLY_OPS = frozenset(
    {
        "Abs",
        "Add",
        "Cast",
        "Clip",
        "Div",
        "Exp",
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

    def mean_square(self, a: str) -> str:
        """``mean(a * a)`` as a scalar, for a reported loss."""
        sq = self.mul(a, a)
        return self.op("ReduceMean", [sq], keepdims=0)


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
    constants: Dict[str, Sequence[int]],
    state: Dict[str, Tuple[Sequence[int], str]],
    scalars: Sequence[str] = (),
    loss: Optional[str] = None,
    name: str = "onnxsim_step",
) -> StepGraph:
    """Wraps ``b``'s accumulated nodes into a :class:`StepGraph`.

    :param constants: ``{input name: shape}`` for the tensors that do not
            change across steps (calibration activations, the reconstruction
            target, a frozen scale, ...)
    :param state: ``{input name: (shape, output name)}`` for the tensors the
            step updates -- the parameter being optimized and the optimizer's
            own moments
    :param scalars: names of scalar (rank-0) per-step inputs, e.g. a learning
            rate or an annealed regularization weight
    :param loss: an optional scalar output name to expose as the loss
    """
    inputs = [
        onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, list(shape))
        for n, shape in constants.items()
    ]
    inputs += [
        onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, list(shape))
        for n, (shape, _) in state.items()
    ]
    inputs += [
        onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [])
        for n in scalars
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


def _run_bound_loop(
    bound: backend.BoundStepLoop,
    step: StepGraph,
    num_steps: int,
    scalar_feeds: Callable[[int], Dict[str, np.ndarray]],
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
        feeds = scalar_feeds(t)
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
) -> Dict[str, np.ndarray]:
    """Runs ``step`` ``num_steps`` times, threading its state through, and
    returns the final state.

    The session (and its execution providers) is created once for the whole
    loop, not once per step -- see :class:`onnxsim.backend.Runner`.

    :param constants: values for the step graph's constant inputs
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
    """
    fetch = list(step.state.values())
    if losses is not None and step.loss_name is not None:
        fetch.append(step.loss_name)
    runner = backend.Runner(step.model, output_names=fetch, providers=providers)

    fixed = {k: np.asarray(v, dtype=np.float32) for k, v in constants.items()}
    initial = {k: np.asarray(v, dtype=np.float32) for k, v in state.items()}

    def scalar_feeds(t: int) -> Dict[str, np.ndarray]:
        if scalars is None:
            return {}
        return {k: np.asarray(v, dtype=np.float32) for k, v in scalars(t).items()}

    # A caller who left out one of the graph's state inputs gets the unbound
    # path's error about a missing feed, not a KeyError from the setup below.
    if bind_state and set(step.state) <= set(initial):
        bound = runner.bind_loop(
            fixed, {name: (out, initial[name]) for name, out in step.state.items()}
        )
        if bound is not None:
            final = _run_bound_loop(bound, step, num_steps, scalar_feeds, losses)
            if final is not None:
                return final

    current = dict(initial)
    for t in range(num_steps):
        feeds = dict(fixed)
        feeds.update(current)
        feeds.update(scalar_feeds(t))
        out = runner(feeds)
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
