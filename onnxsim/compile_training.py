"""A ``torch.compile``-styled training loop, built entirely out of onnxsim's
own grad templating.

``torch.compile`` wraps an eager Python callable: the first call traces and
compiles it, and every later call with compatible inputs reuses that
compiled artifact instead of re-tracing. :func:`compile_training_loop` gives
an ONNX model's training step the same shape -- lazy compilation on first
call, a cached artifact on every call after that -- but what gets compiled is
not a Python trace. It is a single ONNX step graph, assembled the way
:mod:`onnxsim.qat` already assembles one for block-wise QAT:

- :func:`onnxsim.graph_grad.build_backward` differentiates the forward graph
  once, in reverse, emitting the gradient as ordinary ONNX nodes.
- :mod:`onnxsim.qat_graph` wires an :func:`~onnxsim.qat_graph.adam_update`
  (or :func:`~onnxsim.qat_graph.sgd_momentum_update`) onto each trained
  parameter and assembles forward, backward and optimizer into one
  :class:`~onnxsim.qat_graph.StepGraph`.
- :class:`onnxsim.backend.Runner` creates the onnxruntime session for that
  graph once; every subsequent call reuses it.

There is no torch dependency anywhere in this module, and no autograd tape:
differentiation happens once, at compile time, exactly as
:mod:`onnxsim.graph_grad`'s own module docstring describes. What this adds on
top of :mod:`onnxsim.qat_graph`'s ``run_step_graph`` -- which already runs a
:class:`~onnxsim.qat_graph.StepGraph` for a fixed number of steps with a
fixed set of constants -- is the calling convention: a plain callable that
compiles itself lazily on first use and takes a fresh batch of feeds on every
call, the shape an ordinary training loop actually has.

**Copying.** When onnxruntime is installed, :meth:`TrainingLoop.__call__`
never round-trips the trained parameters or the optimizer's own moments
through numpy between steps: they are kept as ``onnxruntime.OrtValue``
(:func:`onnxsim.backend.as_ort_value`/:meth:`onnxsim.backend.Runner.run_with_ort_values`)
and threaded straight from one step's output back in as the next step's
input. ``feeds`` -- the batch itself -- takes the same path: anything that
implements the DLPack protocol (a torch tensor, CPU or CUDA; a numpy array
new enough to implement it) is bound by reference rather than copied into a
fresh buffer first. A plain ``numpy.ndarray`` too old for ``__dlpack__``
still only pays the one copy ``OrtValue.ortvalue_from_numpy`` needs (and on
the CPU, not even that -- it aliases). Falls back to the plain numpy path
automatically when onnxruntime is not installed (the reference-evaluator
backend has no ``OrtValue``/DLPack concept at all); the numbers this returns
are identical either way, only the copying differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper
import onnx.shape_inference

from onnxsim import backend, graph_grad, qat_graph

# Same pairing onnxsim.qat_graph builds every step graph at, and the reason:
# qat_graph.make_step_graph fixes its own opset_imports to this regardless of
# what the forward model was authored against, so a node this module could
# not legally carry at opset 17 is refused here rather than at session
# creation time. See qat.py's _block_shapes for the same reasoning applied to
# a block instead of a whole model.
_OPSET = 17

# Every tensor name this module introduces starts here, so it cannot collide
# with a name carried over from the forward model -- the same convention
# onnxsim.qat.py's _PREFIX documents.
_PREFIX = "trainstep__"


def _int_shape(shape: Sequence[Union[int, str]]) -> Tuple[int, ...]:
    """``shape`` as a plain ``Tuple[int, ...]``.

    Every entry ``_static_shapes_and_types`` returns is already a real ``int``
    (a dynamic ``str`` dimension is refused there before this module ever
    sees it) -- this only narrows the type back down from the
    ``Sequence[Union[int, str]]`` :func:`onnxsim.graph_grad.build_backward`'s
    own ``shapes`` parameter needs, for the state/constants declarations
    below that (unlike that call) have no reason to carry a dimension this
    module can never produce.
    """
    return tuple(int(d) for d in shape)


def _to_numpy(value: Any) -> np.ndarray:
    """``value`` as a plain ``numpy.ndarray``, whether it already is one or
    is an ``onnxruntime.OrtValue`` (``TrainingLoop``'s own state, kept as
    ``OrtValue`` between calls when onnxruntime supports it -- see this
    module's docstring). A numpy array has no ``.numpy()`` method of its
    own, which is what tells the two apart here."""
    return value.numpy() if hasattr(value, "numpy") else value


def _static_shapes_and_types(
    model: onnx.ModelProto,
) -> Tuple[Dict[str, Sequence[Union[int, str]]], Dict[str, int]]:
    """Every tensor's static shape and element type, via shape inference.

    Raises if inference itself fails, or if any tensor's shape is not fully
    static -- a step graph's shapes are fixed at build time, the same
    requirement every other caller of :mod:`onnxsim.qat_graph` already meets.

    Typed ``Sequence[Union[int, str]]`` per tensor -- never actually a ``str``
    entry here, since every dimension is checked static above -- only because
    that is :func:`onnxsim.graph_grad.build_backward`'s own ``shapes``
    parameter type (it accepts a symbolic ``dim_param`` from callers that
    allow one) and ``Dict`` is invariant in its value type: a ``Dict[str,
    List[int]]`` is not a ``Dict[str, Sequence[Union[int, str]]]`` as far as
    mypy is concerned, even though every value satisfies it structurally.
    """
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as error:  # noqa: BLE001 -- re-raised with context
        raise ValueError(
            f"cannot statically infer the model's shapes at opset {_OPSET}: {error}"
        ) from error

    shapes: Dict[str, Sequence[Union[int, str]]] = {}
    elem_types: Dict[str, int] = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    ):
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
        if any(d <= 0 for d in dims):
            raise ValueError(
                f"tensor {value.name!r} has a non-static shape; "
                "compile_training_loop needs every shape known at compile time"
            )
        shapes[value.name] = dims
        elem_types[value.name] = value.type.tensor_type.elem_type
    for init in inferred.graph.initializer:
        shapes[init.name] = list(init.dims)
        elem_types[init.name] = init.data_type
    return shapes, elem_types


@dataclass
class TrainingLoop:
    """One training step, compiled lazily on first call and reused on every
    call after that.

    Do not construct this directly; use :func:`compile_training_loop`. Call
    the instance with one batch's feeds to run one optimizer step:

    .. code-block:: python

        loop = onnxsim.compile_training_loop(model, "loss", ["fc.weight"])
        for batch in batches:
            loss = loop({"x": batch.x, "y": batch.y}, lr=1e-3)

    The first call builds the step graph and the onnxruntime session for it
    ("compiles"); every call after that -- including the first -- runs one
    step and threads the trained parameters' and optimizer's state through to
    the next call, the way :func:`onnxsim.qat_graph.run_step_graph` threads a
    step graph's state across a fixed-length loop, except here the caller's
    own loop decides when to stop and what each step's batch is.
    """

    #: The forward model: its nodes are copied verbatim into the compiled
    #: step graph, so an op :mod:`onnxsim.graph_grad` cannot differentiate
    #: (:func:`onnxsim.graph_grad.supported_ops`) fails compilation with its
    #: own :class:`onnxsim.graph_grad.UnsupportedOpError`.
    model: onnx.ModelProto = field(repr=False)
    #: Name of a scalar (rank-0) tensor the model produces -- what the
    #: trained parameters are optimized against.
    loss_output: str
    #: Names of the model's own float32 initializers to train.
    params: Tuple[str, ...]
    #: ``"adam"`` (default) or ``"sgd_momentum"`` -- see
    #: :func:`onnxsim.qat_graph.adam_update` and
    #: :func:`onnxsim.qat_graph.sgd_momentum_update`.
    optimizer: str = "adam"
    #: onnxruntime execution providers for the compiled step, in priority
    #: order. ``None`` means CPU.
    providers: Optional[Sequence[backend.Provider]] = None

    _step: Optional[qat_graph.StepGraph] = field(default=None, init=False, repr=False)
    _runner: Optional[backend.Runner] = field(default=None, init=False, repr=False)
    #: The state as onnxruntime last returned it: an ``OrtValue`` per entry
    #: when :meth:`onnxsim.backend.Runner.supports_ort_values` (kept
    #: device-resident between calls -- see this module's own docstring),
    #: else a plain ``numpy.ndarray`` (the reference-evaluator fallback).
    #: Read through :func:`_to_numpy`, never assumed to be either.
    _state: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _t: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.optimizer not in ("adam", "sgd_momentum"):
            raise ValueError(
                f"unknown optimizer {self.optimizer!r}; use 'adam' or 'sgd_momentum'"
            )
        self.params = tuple(self.params)
        if not self.params:
            raise ValueError("params must name at least one trainable initializer")

    @property
    def compiled(self) -> bool:
        """Whether the step graph has been built yet. ``False`` until the
        first call, or until :attr:`step_graph`/:attr:`initial_state` is
        read."""
        return self._runner is not None

    @property
    def step_graph(self) -> qat_graph.StepGraph:
        """The compiled step graph, compiling now if this is the first
        access.

        Exposed so a caller can get at the actual compiled artifact -- to run
        it outside this loop's own ``__call__`` (a different runtime
        entirely, e.g. onnxruntime-web in a browser; see
        ``scripts/convertmodel/test/make_step_graph_fixtures.py``'s
        ``build_train_loop_demo``, which compiles a loop for exactly this)
        or to inspect it. Reading this before any call does not run a step;
        it only builds the graph and the onnxruntime session for it.
        """
        if self._runner is None:
            self._compile()
        assert self._step is not None
        return self._step

    @property
    def initial_state(self) -> Dict[str, np.ndarray]:
        """The step graph's state inputs at their starting values: the
        trained parameters as the forward model had them, and the
        optimizer's own moment buffers at zero. Keyed exactly as
        :attr:`step_graph`'s ``state`` dict, so a caller replaying the
        compiled step graph elsewhere can feed it directly.

        Compiles on first access, like :attr:`step_graph`. Meaningful only
        before the loop has actually been called -- read it before the first
        call, or keep a copy from then, since :meth:`__call__` advances
        :attr:`parameters`'s own state on every call after that.
        """
        self.step_graph  # noqa: B018 -- triggers _compile() for its side effect
        return {k: _to_numpy(v) for k, v in self._state.items()}

    def __call__(self, feeds: Dict[str, Any], lr: float) -> float:
        """Runs one step on ``feeds`` (the model's own non-trained input
        tensors, e.g. ``x``/``y``) at learning rate ``lr`` and returns the
        scalar loss.

        Compiles on the first call. Every call -- the first included --
        advances this instance's own trained-parameter and optimizer state by
        one step; :meth:`parameters` and :meth:`export` read the state as of
        the most recent call.

        ``feeds``' values are usually ``numpy.ndarray``, but anything
        implementing the DLPack protocol (a torch tensor, CPU or CUDA) is
        accepted directly -- see this module's own docstring on why that
        avoids a copy, and :func:`onnxsim.backend.as_ort_value` for exactly
        what "implementing DLPack" buys here.
        """
        if self._runner is None:
            self._compile()
        assert self._step is not None and self._runner is not None

        if self._runner.supports_ort_values():
            return self._call_with_ort_values(feeds, lr)
        return self._call_with_numpy(feeds, lr)

    def _call_with_ort_values(self, feeds: Dict[str, Any], lr: float) -> float:
        """:meth:`__call__`'s onnxruntime path: every tensor that crosses
        into or out of this step is an ``OrtValue``, so the trained
        parameters and the optimizer's own moments never touch numpy between
        calls, and a DLPack-capable ``feeds`` value never gets copied into a
        fresh buffer first. See this module's own docstring."""
        assert self._step is not None and self._runner is not None
        inputs = {k: backend.as_ort_value(v) for k, v in feeds.items()}
        inputs.update(self._state)
        inputs["lr"] = backend.as_ort_value(lr)
        if self.optimizer == "adam":
            for name, value in qat_graph.adam_bias_corrections(self._t).items():
                inputs[name] = backend.as_ort_value(value)

        out = self._runner.run_with_ort_values(inputs)
        self._state = {
            input_name: out[output_name]
            for input_name, output_name in self._step.state.items()
        }
        self._t += 1
        assert self._step.loss_name is not None
        return float(out[self._step.loss_name].numpy())

    def _call_with_numpy(self, feeds: Dict[str, Any], lr: float) -> float:
        """:meth:`__call__`'s fallback path, for when onnxruntime is not
        installed and every run therefore goes through the reference
        evaluator, which has no ``OrtValue``/DLPack concept at all: plain
        numpy in, plain numpy out, exactly as this loop worked before the
        onnxruntime path existed."""
        assert self._step is not None and self._runner is not None
        inputs = {k: np.asarray(v, dtype=np.float32) for k, v in feeds.items()}
        inputs.update(self._state)
        inputs["lr"] = np.asarray(lr, dtype=np.float32)
        if self.optimizer == "adam":
            for name, value in qat_graph.adam_bias_corrections(self._t).items():
                inputs[name] = np.asarray(value, dtype=np.float32)

        out = self._runner(inputs)
        self._state = {
            input_name: out[output_name]
            for input_name, output_name in self._step.state.items()
        }
        self._t += 1
        assert self._step.loss_name is not None
        return float(out[self._step.loss_name])

    def parameters(self) -> Dict[str, np.ndarray]:
        """Current trained values of each name in :attr:`params`."""
        if not self.compiled:
            raise RuntimeError(
                "parameters() is only available once the loop has compiled; "
                "call the loop at least once first"
            )
        return {name: _to_numpy(self._state[name]) for name in self.params}

    def export(self) -> onnx.ModelProto:
        """The forward model with each trained parameter's initializer
        replaced by its current value -- the model to actually ship.

        A no-op copy of :attr:`model` if the loop has never been called.
        """
        model = onnx.ModelProto()
        model.CopyFrom(self.model)
        if not self.compiled:
            return model
        trained = self.parameters()
        for init in model.graph.initializer:
            if init.name in trained:
                value = trained[init.name].astype(
                    onnx.helper.tensor_dtype_to_np_dtype(init.data_type), copy=False
                )
                init.CopyFrom(onnx.numpy_helper.from_array(value, init.name))
        return model

    def _compile(self) -> None:
        """Builds the step graph and creates its onnxruntime session.

        Differentiates the *whole* forward graph (not a caller-chosen slice,
        unlike :mod:`onnxsim.qat`'s block-wise machinery) with
        :attr:`loss_output` as the sole seed and :attr:`params` as the
        targets, then wires an optimizer update onto each target with
        :mod:`onnxsim.qat_graph` and assembles the result with
        :func:`onnxsim.qat_graph.make_step_graph`.

        A parameter's own initializer name is reused, unchanged, as its state
        input's name: the forward nodes copied into the step graph already
        reference it under that name, so the state input has to be named
        that for the graph to resolve. Only the optimizer's own per-parameter
        moment buffers, which no forward node references, get a fresh
        :meth:`onnxsim.qat_graph.GraphBuilder.name`.
        """
        model = self.model
        shapes, elem_types = _static_shapes_and_types(model)

        initializers = {t.name: t for t in model.graph.initializer}
        missing = [p for p in self.params if p not in initializers]
        if missing:
            raise ValueError(f"{missing} are not initializers of the model")
        not_float = [
            p
            for p in self.params
            if initializers[p].data_type != onnx.TensorProto.FLOAT
        ]
        if not_float:
            raise ValueError(f"{not_float} are not float32 initializers")

        loss_shape = shapes.get(self.loss_output)
        if loss_shape is None:
            raise ValueError(f"no static shape for loss output {self.loss_output!r}")
        if loss_shape:
            raise ValueError(
                f"loss output {self.loss_output!r} has shape {loss_shape}, "
                "but the loss must be a scalar"
            )

        b = qat_graph.GraphBuilder(prefix=_PREFIX)
        b.nodes = list(model.graph.node)
        trained = set(self.params)
        b.initializer = [t for t in model.graph.initializer if t.name not in trained]

        seed = b.const(np.array(1.0, dtype=np.float32), "loss_seed")
        grads = graph_grad.build_backward(
            b,
            nodes=list(model.graph.node),
            shapes=shapes,
            grad_outputs={self.loss_output: seed},
            targets=list(self.params),
        )

        state: Dict[str, Tuple[Sequence[int], str]] = {}
        scalars = ["lr"]
        if self.optimizer == "adam":
            scalars += ["m_correction", "v_correction"]

        initial_state: Dict[str, np.ndarray] = {}
        for p in self.params:
            w_shape = _int_shape(shapes[p])
            grad = grads[p]
            m_input = b.name("m")
            if self.optimizer == "adam":
                v_input = b.name("v")
                w_next, m_next, v_next = qat_graph.adam_update(
                    b, p, grad, m_input, v_input, "lr", "m_correction", "v_correction"
                )
                state[v_input] = (w_shape, v_next)
                initial_state[v_input] = np.zeros(w_shape, dtype=np.float32)
            else:
                w_next, m_next = qat_graph.sgd_momentum_update(
                    b, p, grad, m_input, "lr"
                )
            state[p] = (w_shape, w_next)
            state[m_input] = (w_shape, m_next)
            initial_state[p] = onnx.numpy_helper.to_array(initializers[p]).astype(
                np.float32
            )
            initial_state[m_input] = np.zeros(w_shape, dtype=np.float32)

        constants: Dict[str, Tuple[Sequence[int], int]] = {}
        for inp in model.graph.input:
            if inp.name in initializers:
                continue
            shape = shapes.get(inp.name)
            if shape is None:
                raise ValueError(f"no static shape for model input {inp.name!r}")
            constants[inp.name] = (
                _int_shape(shape),
                elem_types.get(inp.name, onnx.TensorProto.FLOAT),
            )

        self._step = qat_graph.make_step_graph(
            b,
            constants=constants,
            state=state,
            scalars=scalars,
            loss=self.loss_output,
            name="onnxsim_train_step",
        )
        self._runner = backend.Runner(
            self._step.model,
            output_names=list(self._step.state.values()) + [self.loss_output],
            providers=self.providers,
        )
        # Uploaded once, here, rather than on every call: from this point on
        # __call__'s onnxruntime path never sees a numpy array for its own
        # state again (see this module's own docstring).
        if self._runner.supports_ort_values():
            self._state = {k: backend.as_ort_value(v) for k, v in initial_state.items()}
        else:
            self._state = initial_state
        self._t = 0


def compile_training_loop(
    model: onnx.ModelProto,
    loss_output: str,
    params: Sequence[str],
    optimizer: str = "adam",
    providers: Optional[Sequence[backend.Provider]] = None,
) -> TrainingLoop:
    """Wraps ``model`` as a ``torch.compile``-styled training loop.

    Nothing is built yet -- the returned :class:`TrainingLoop` compiles its
    step graph lazily, on its first call, exactly when a ``torch.compile``-
    wrapped callable would first trace. See :class:`TrainingLoop` for the
    calling convention.

    :param model: the forward model. Every tensor's shape must be static (no
            symbolic dimensions) and every node's op type must be one
            :func:`onnxsim.graph_grad.build_backward` can differentiate
            (:func:`onnxsim.graph_grad.supported_ops`) -- checked at compile
            time, on the first call, not here.
    :param loss_output: name of a scalar (rank-0) tensor the model produces.
    :param params: names of the model's own float32 initializers to train.
    :param optimizer: ``"adam"`` (default) or ``"sgd_momentum"``.
    :param providers: onnxruntime execution providers for the compiled step,
            in priority order. ``None`` means CPU.
    """
    return TrainingLoop(
        model=model,
        loss_output=loss_output,
        params=tuple(params),
        optimizer=optimizer,
        providers=providers,
    )
