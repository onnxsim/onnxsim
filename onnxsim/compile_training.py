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
implements the DLPack protocol (a torch tensor, CPU, CUDA or ROCm/HIP; a numpy array
new enough to implement it) is bound by reference rather than copied into a
fresh buffer first. A plain ``numpy.ndarray`` too old for ``__dlpack__``
still only pays the one copy ``OrtValue.ortvalue_from_numpy`` needs (and on
the CPU, not even that -- it aliases). Falls back to the plain numpy path
automatically when onnxruntime is not installed (the reference-evaluator
backend has no ``OrtValue``/DLPack concept at all); the numbers this returns
are identical either way, only the copying differs.

**GPU execution (CUDA / ROCm / MIGraphX).** Genuinely zero-copy end to end:
pass ``providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`` on NVIDIA
(or ``["ROCMExecutionProvider", "CPUExecutionProvider"]`` /
``["MIGraphXExecutionProvider", "CPUExecutionProvider"]`` on AMD ROCm -- see
``scripts/amd/README.md`` for the wheel to install,
``onnxruntime-rocm``/``onnxruntime-migraphx``) to
:func:`compile_training_loop`/``onnxsim.compile_torch_training_loop`` and
feed device-resident tensors (a ``torch.Tensor`` already on ``"cuda"`` -- which
is also what a ROCm/HIP torch build reports its device as -- or any other
object whose ``__dlpack_device__`` reports the matching device) --
``as_ort_value`` is device-agnostic, so this needs no GPU-vendor-specific code
of its own, only a matching ``onnxruntime`` build and a session actually
configured to run on it. There is exactly one unavoidable host round trip: the
very first call uploads the forward model's own initializers
(``TrainingLoop``'s initial state), which start out as ordinary host bytes
inside the ONNX model -- there is no tensor on the caller's side yet to alias
for those. From the first call's own outputs on, the trained parameters and
optimizer moments are genuinely device-resident ``OrtValue``\\ s
(onnxruntime's GPU kernels produce them there, and nothing here ever calls
``.numpy()`` on them), so every call after the first pays no host transfer
for the state at all -- only ``feeds`` and the tiny per-step scalars (``lr``,
Adam's bias corrections) cross the bus, and a device-resident ``feeds``
tensor skips even that. See ``tests/test_compile_training.py``'s and
``tests/test_torch_training.py``'s own CUDA-gated tests (skipped without a
matching ``onnxruntime`` build and GPU) for this asserted directly against
``OrtValue.device_name()``, plus the ROCm/MIGraphX-gated training tests beside
them, which assert the same loop converges through those providers.

**MPS / WebGPU.** No zero-copy path exists between a PyTorch MPS tensor
(Apple's Metal backend) and onnxruntime's WebGPU execution provider today,
and that is not something this module can paper over. Two independent
blockers: (1) DLPack device mismatch -- an MPS tensor's own
``__dlpack_device__`` reports ``kDLMetal``, not ``kDLWebGPU``, so
``OrtValue.from_dlpack`` would reject it outright even if onnxruntime's
WebGPU EP accepted external buffers at all; (2) onnxruntime's WebGPU EP
does not expose a public API to import an externally created GPU buffer the
way the CUDA EP does via DLPack/``IOBinding`` -- its buffer manager is
internal. This repository's own WebGPU testing
(``scripts/convertmodel/test/webgpu_*_demo.test.mjs``) additionally runs
onnxruntime-web inside a headless Chromium browser via Playwright -- a
separate process from Python entirely -- so even a hypothetical MPS/WebGPU
DLPack bridge would still have to cross a process boundary, which is a copy
by construction. The practical option today is the same as for any other
unsupported source device: let ``feeds``/parameters cross to host memory
(``.cpu()``/``.numpy()``) before they reach this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.inliner
import onnx.numpy_helper
import onnx.shape_inference

from onnxsim import backend, grad_health, graph_grad, qat_graph

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


def _quantize_forward_for_training(
    model: onnx.ModelProto,
    params: Sequence[str],
    calibration_data: Optional[Sequence[Dict[str, np.ndarray]]],
) -> onnx.ModelProto:
    """Rewrites ``model``'s forward onto the INT8 grid while keeping every
    trained weight trainable in fp32 -- the ``quantize_forward`` half of
    :meth:`TrainingLoop._compile`.

    Runs :func:`onnxsim.calibration.quantize_static` (QDQ format: the
    MatMul/Gemm/Conv nodes stay, their inputs are rerouted through
    ``QuantizeLinear``/``DequantizeLinear`` pairs), then gives each trained
    weight a fp32 master initializer under its original name, feeding the
    weight's ``DequantizeLinear`` through a fresh ``QuantizeLinear`` -- a
    fake-quant chain: the forward computes exactly what INT8 inference
    would, while the master receives straight-through gradients (see
    :mod:`onnxsim.graph_grad`'s ``QuantizeLinear`` rule) and the optimizer
    keeps updating fp32.

    A trained weight the quantizer left untouched (not a constant 2-D/conv
    weight, or a missing calibration range) is left alone and keeps training
    in plain fp32 -- mixed precision degrades gracefully, layer by layer.
    """
    from onnxsim.calibration import quantize_static

    try:
        # Intermediate activations need value_info or the calibrator only
        # ever sees the graph inputs (and the tail of the network silently
        # trains fp32 -- including, worse, a half-quantized graph that trips
        # fusing runtimes).
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    quantized = quantize_static(model, calibration_data)
    trained = set(params)
    # Original weight uses, keyed by node output: the quantizer rewires a
    # node's inputs but never the node itself, so outputs identify nodes
    # across the rewrite.
    weight_uses: Dict[str, str] = {}
    for node in model.graph.node:
        if (
            node.op_type in ("MatMul", "Gemm", "Conv")
            and len(node.input) > 1
            and node.input[1] in trained
            and node.output
        ):
            weight_uses[node.output[0]] = node.input[1]
    if not weight_uses:
        return quantized

    float_inits = {t.name: t for t in model.graph.initializer}
    inits = {t.name: t for t in quantized.graph.initializer}
    by_output = {o: n for n in quantized.graph.node for o in n.output}
    # (param, weight-DQ node) pairs whose fake-quant master to build. The
    # stale float weight and the int8 codes are dropped by name below, so
    # collection and mutation stay strictly ordered: dropping first, then
    # creating the same-named masters, never both at once.
    jobs: Dict[str, List[onnx.NodeProto]] = {}
    drop_inits: List[str] = []
    for out, param in weight_uses.items():
        node = by_output.get(out)
        if node is None or len(node.input) < 2:
            continue
        dq = by_output.get(node.input[1])
        if (
            dq is None
            or dq.op_type != "DequantizeLinear"
            or not (2 <= len(dq.input) <= 3)
            or dq.input[0] not in inits
        ):
            continue  # weight left unquantized: trains fp32 as before
        wq = inits[dq.input[0]]
        if wq.data_type != onnx.TensorProto.INT8:
            continue
        master = float_inits.get(param)
        if master is None or master.data_type != onnx.TensorProto.FLOAT:
            continue
        stale = inits.get(param)
        if stale is not None:
            if any(stale.name in n.input for n in quantized.graph.node):
                raise ValueError(
                    f"trained weight {param!r} is still consumed in float "
                    "by some node after quantization; quantize_forward "
                    "needs every use quantized (or none)"
                )
            if stale.name not in drop_inits:
                drop_inits.append(stale.name)
        if wq.name not in drop_inits:
            drop_inits.append(wq.name)
        jobs.setdefault(param, []).append(dq)
    if drop_inits:
        doomed = set(drop_inits)
        kept = [t for t in quantized.graph.initializer if t.name not in doomed]
        del quantized.graph.initializer[:]
        quantized.graph.initializer.extend(kept)
        inits = {t.name: t for t in quantized.graph.initializer}
    for param, dqs in jobs.items():
        master = float_inits[param]
        ql_out = f"{param}__train_master_q"
        first = dqs[0]
        ql = onnx.helper.make_node(
            "QuantizeLinear", [param, first.input[1], *first.input[2:]], [ql_out]
        )
        # Insert directly before the first consuming DequantizeLinear: nodes
        # must stay topologically sorted.
        anchor = next(i for i, n in enumerate(quantized.graph.node) if n is first)
        quantized.graph.node.insert(anchor, ql)
        fresh = onnx.TensorProto()
        fresh.CopyFrom(master)
        quantized.graph.initializer.extend([fresh])
        for dq in dqs:
            dq.input[0] = ql_out
    return quantized


def _cast_backward_to_fp16(
    b: qat_graph.GraphBuilder,
    n_fwd: int,
    n_bwd: int,
    elem_types: Dict[str, int],
    grads: Dict[str, str],
) -> Dict[str, str]:
    """Rewrites the backward slice ``b.nodes[n_fwd:n_bwd]`` onto fp16 and
    returns the (possibly new) gradient names, one per target.

    Every edge from an fp32 source outside the slice (a forward tensor, a
    float initializer) into the slice gets a ``Cast`` to ``FLOAT16``; every
    gradient gets a ``Cast`` back to ``FLOAT`` right where the optimizer
    section will consume it. Non-float sources (int64 shapes/indices,
    boolean masks, quantized codes) pass through untouched, and anything
    the slice computes internally stays whatever dtype its (now fp16)
    inputs propagate. Casts insert directly before their consumer, so the
    node list stays topologically sorted.
    """
    float_sources = {
        t.name for t in b.initializer if t.data_type == onnx.TensorProto.FLOAT
    }
    float_sources.update(
        name for name, dtype in elem_types.items() if dtype == onnx.TensorProto.FLOAT
    )
    cast_suffix = 0
    fp16_names = set()

    def cast_to(dtype: int, src: str, hint: str) -> Tuple[str, onnx.NodeProto]:
        nonlocal cast_suffix
        out = f"{src}__bwd_{hint}_{cast_suffix}"
        cast_suffix += 1
        return out, onnx.helper.make_node("Cast", [src], [out], to=dtype)

    pending: List[Tuple[int, onnx.NodeProto]] = []
    for idx in range(n_fwd, n_bwd):
        node = b.nodes[idx]
        for j, inp in enumerate(node.input):
            if inp in float_sources and inp not in fp16_names:
                out, cast = cast_to(onnx.TensorProto.FLOAT16, inp, "f16")
                pending.append((idx, cast))
                node.input[j] = out
                fp16_names.add(out)
    offset = 0
    for idx, cast in pending:
        b.nodes.insert(idx + offset, cast)
        offset += 1
    # Rules emit mask/constant casts hardcoded to FLOAT (Greater/Less +
    # Cast, Where's cond cast, int-index casts). Inside the slice those
    # must match the surrounding fp16 math, so retarget the ones that do
    # not provably consume a static-fp32 tensor. (The fp32 grad
    # cast-backs appended below are created after this walk and keep
    # their dtype.)
    for idx in range(n_fwd, n_bwd + offset):
        node = b.nodes[idx]
        if node.op_type != "Cast":
            continue
        to = next((a.i for a in node.attribute if a.name == "to"), None)
        if to != onnx.TensorProto.FLOAT:
            continue
        if any(inp in float_sources for inp in node.input if inp):
            continue
        for attr in node.attribute:
            if attr.name == "to":
                attr.i = onnx.TensorProto.FLOAT16
    new_grads: Dict[str, str] = {}
    for param, grad in grads.items():
        out, cast = cast_to(onnx.TensorProto.FLOAT, grad, "f32")
        b.nodes.append(cast)
        new_grads[param] = out
    return new_grads


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


@dataclass(frozen=True)
class CustomOptimizer:
    """A per-parameter optimizer update rule, expressed as one reusable ONNX
    function, for :func:`compile_training_loop`'s/
    ``onnxsim.compile_torch_training_loop``'s own ``optimizer=`` argument --
    an alternative to the two builtin string choices (``"adam"``,
    ``"sgd_momentum"``, see :func:`onnxsim.qat_graph.adam_update`/
    :func:`onnxsim.qat_graph.sgd_momentum_update`) for a caller whose
    optimizer those two do not cover.

    Not constructed directly in the ordinary case; see
    ``onnxsim.torch_training.trace_torch_optimizer``, which builds one from
    an ordinary PyTorch function via the same ``torch.export`` FX pipeline
    :func:`onnxsim.torch_training.export_torch_module_to_onnx` uses for the
    forward model itself. Nothing here is torch-specific, though: any
    caller that can produce a model matching the shape below (by any means)
    can use one, exactly as any caller of :func:`compile_training_loop`
    itself can hand it a plain ONNX model with no torch involved anywhere.

    ``model`` must have exactly ``2 + num_state + 1`` inputs, in order:
    ``param``, ``grad``, ``num_state`` state tensors, then ``lr`` (a
    rank-0 scalar) -- and exactly ``1 + num_state`` outputs: the updated
    parameter, then the updated state tensors in the same order theirs came
    in. :meth:`TrainingLoop._compile` calls it once per trained parameter,
    each call wiring that parameter's own name, gradient, state buffers
    (freshly allocated and zero-initialized, :attr:`num_state` of them) and
    the step graph's shared ``"lr"`` input to those formal names.

    Traced once, against a fixed shape unrelated to any trained parameter's
    own shape, and reused unchanged at every parameter's own call site --
    which is only sound because an ordinary optimizer update is elementwise
    (nothing about it depends on a tensor's rank or size beyond
    broadcasting a scalar learning rate over it) and therefore genuinely
    does not care what shape it runs against once spliced in. A custom
    optimizer that is not elementwise in this sense (indexes into ``param``,
    reshapes it, reduces over some of its axes but not others) is out of
    scope for the same reason :mod:`onnxsim.qat_graph`'s own two builtin
    optimizers never needed to be anything but elementwise themselves.
    """

    #: A model-local ONNX function's worth of nodes -- see this class's own
    #: docstring for the exact input/output contract.
    model: onnx.ModelProto
    #: How many per-parameter state tensors this optimizer carries (Adam: 2,
    #: SGD-momentum: 1, plain SGD: 0).
    num_state: int

    def __post_init__(self) -> None:
        want = 2 + self.num_state + 1
        got = len(self.model.graph.input)
        if got != want:
            raise ValueError(
                f"a CustomOptimizer with num_state={self.num_state} needs a model "
                f"with {want} inputs (param, grad, {self.num_state} state tensors, "
                f"lr), got {got}"
            )
        want_out = 1 + self.num_state
        got_out = len(self.model.graph.output)
        if got_out != want_out:
            raise ValueError(
                f"a CustomOptimizer with num_state={self.num_state} needs a model "
                f"with {want_out} outputs (the updated parameter, then "
                f"{self.num_state} updated state tensors), got {got_out}"
            )


def _custom_optimizer_function(
    optimizer: CustomOptimizer, name: str
) -> onnx.FunctionProto:
    """``optimizer.model`` as one reusable :class:`onnx.FunctionProto`, for
    :meth:`GraphBuilder.call` to splice in once per trained parameter (see
    :class:`CustomOptimizer`'s own docstring for why the same traced function
    is sound to reuse unchanged at every call site).

    A ``FunctionProto`` has no initializer list of its own (unlike the
    ``ModelProto`` :func:`onnxsim.torch_training.trace_torch_optimizer`
    actually produces) -- any initializer the traced update needs (Adam's own
    beta/eps constants baked in by the traced Python code, say) is lowered to
    a ``Constant`` node feeding the same name instead, prepended before the
    model's own nodes.
    """
    constant_nodes = [
        onnx.helper.make_node("Constant", [], [init.name], value=init)
        for init in optimizer.model.graph.initializer
    ]
    opset_imports = list(optimizer.model.opset_import) or [
        onnx.helper.make_opsetid("", _OPSET)
    ]
    return onnx.helper.make_function(
        domain="onnxsim.custom_optimizer",
        fname=name,
        inputs=[i.name for i in optimizer.model.graph.input],
        outputs=[o.name for o in optimizer.model.graph.output],
        nodes=constant_nodes + list(optimizer.model.graph.node),
        opset_imports=opset_imports,
    )


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
    #: ``"adam"`` (default), ``"sgd_momentum"`` -- see
    #: :func:`onnxsim.qat_graph.adam_update` and
    #: :func:`onnxsim.qat_graph.sgd_momentum_update` -- or a
    #: :class:`CustomOptimizer` for an update rule neither of those two
    #: covers (see that class's own docstring, and
    #: ``onnxsim.torch_training.trace_torch_optimizer`` for the usual way to
    #: build one).
    optimizer: Union[str, CustomOptimizer] = "adam"
    #: Static loss-scale factor (default 1.0 = off): seeds the backward with
    #: this value instead of 1 and divides every parameter gradient back out
    #: before the optimizer update. A no-op mathematically at any value
    #: (exact for powers of two) -- its purpose is future lower-precision
    #: backward math, where unscaled gradients underflow: scaling keeps them
    #: representable through the backward, unscaling restores them for the
    #: fp32 update. Must be finite and positive.
    loss_scale: float = 1.0
    #: onnxruntime execution providers for the compiled step, in priority
    #: order. ``None`` means CPU.
    providers: Optional[Sequence[backend.Provider]] = None
    #: Execution providers for the forward pass only. ``None`` (the
    #: default) runs the whole fused step on :attr:`providers`. Set it --
    #: e.g. a VitisAI NPU entry with ``CPUExecutionProvider`` last -- to
    #: split execution instead: the forward runs on
    #: :attr:`forward_providers`, the backward+optimizer on
    #: :attr:`providers`, with the boundary activations crossing the host
    #: between the two sessions every step.
    #:
    #: The split exists for runtimes that fuse the forward fine but abort
    #: on the backward-containing whole (measured: the VitisAI EP's frontend
    #: mis-fuses a ``QuantizeLinear`` when the step graph carries the
    #: training backward alongside the QDQ forward). Split mode always
    #: crosses numpy (no device-resident state), while the fused mode keeps
    #: the ``OrtValue`` fast path.
    forward_providers: Optional[Sequence[backend.Provider]] = None
    #: Train with an INT8-quantized forward and an fp32 backward: when true,
    #: :meth:`_compile` first runs :func:`onnxsim.calibration.quantize_static`
    #: over the model (calibrated on :attr:`calibration_data`), then gives
    #: every trained weight a trainable fp32 master behind a fake-quant
    #: ``QuantizeLinear`` (straight-through differentiable -- see
    #: :mod:`onnxsim.graph_grad`'s ``QuantizeLinear`` rule), so the forward
    #: computes on the INT8 grid a QDQ-aware runtime fuses into integer
    #: kernels (e.g. the VitisAI NPU) while gradients, master weights and
    #: optimizer state all stay fp32. Integer backprop does not exist on
    #: such runtimes, so this split -- quantized forward, float backward --
    #: is the whole of what "INT8 training" can mean there.
    quantize_forward: bool = False
    #: Representative input batches calibrating :attr:`quantize_forward`'s
    #: activation ranges -- one ``{input_name: array}`` dict per batch, as
    #: :func:`onnxsim.calibration.quantize_static` takes them. ``None``
    #: falls back to that function's own random calibration data (a smoke
    #: test, not a deployment recipe). Ignored unless
    #: :attr:`quantize_forward` is true.
    calibration_data: Optional[Sequence[Dict[str, np.ndarray]]] = None
    #: Precision of the backward math: ``"float32"`` (default) or
    #: ``"float16"``. ``"float16"`` casts every fp32 edge entering the
    #: backward subgraph to fp16 and casts each parameter gradient back to
    #: fp32 before the optimizer update, so masters, moments and updates
    #: stay fp32 while the gradient math itself halves in width -- the
    #: standard AMP shape, minus the dynamic loss scaling (use
    #: :attr:`loss_scale` with a large power of two: unscaled fp16
    #: gradients underflow fast). Non-float edges (int64 shapes/indices,
    #: boolean masks, quantized codes) are left untouched.
    backward_precision: str = "float32"
    #: Input ranges ``{input: (lo, hi)}`` over the step's free inputs (data, labels, the loss
    #: seed, the parameters). When set, :meth:`_compile` checks the gradients it builds for
    #: underflow and rounding loss (:func:`onnxsim.grad_health.check_backward_health`) and keeps
    #: the verdict in :attr:`gradient_report`. Checked before loss scaling and the fp16 cast.
    gradient_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    #: Verdict of the gradient check requested by :attr:`gradient_ranges`; ``None`` until the
    #: step compiles, or always when that check was not requested.
    gradient_report: Optional[grad_health.GradHealthReport] = field(
        default=None, init=False, repr=False
    )

    _step: Optional[qat_graph.StepGraph] = field(default=None, init=False, repr=False)
    _runner: Optional[backend.Runner] = field(default=None, init=False, repr=False)
    #: Split-mode sessions (forward on :attr:`forward_providers`,
    #: backward+optimizer on :attr:`providers`); both ``None`` unless split
    #: execution is active. ``_needed`` is the boundary activation set the
    #: forward session returns and the backward session consumes.
    _runner_fwd: Optional[backend.Runner] = field(default=None, init=False, repr=False)
    _runner_bwd: Optional[backend.Runner] = field(default=None, init=False, repr=False)
    _needed: Tuple[str, ...] = field(default=(), init=False, repr=False)
    #: Split mode's state map (state input name -> updated-state output
    #: name), mirroring :attr:`step_graph`'s own for the backward session.
    _bwd_state: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    #: The state as onnxruntime last returned it: an ``OrtValue`` per entry
    #: when :meth:`onnxsim.backend.Runner.supports_ort_values` (kept
    #: device-resident between calls -- see this module's own docstring),
    #: else a plain ``numpy.ndarray`` (the reference-evaluator fallback).
    #: Read through :func:`_to_numpy`, never assumed to be either.
    _state: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _t: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, CustomOptimizer) and self.optimizer not in (
            "adam",
            "sgd_momentum",
        ):
            raise ValueError(
                f"unknown optimizer {self.optimizer!r}; use 'adam', 'sgd_momentum', "
                "or a CustomOptimizer instance"
            )
        self.params = tuple(self.params)
        if not self.params:
            raise ValueError("params must name at least one trainable initializer")
        if not (np.isfinite(self.loss_scale) and self.loss_scale > 0):
            raise ValueError(
                f"loss_scale must be finite and positive, got {self.loss_scale!r}"
            )
        if self.backward_precision not in ("float32", "float16"):
            raise ValueError(
                "backward_precision must be 'float32' or 'float16', "
                f"got {self.backward_precision!r}"
            )

    @property
    def compiled(self) -> bool:
        """Whether the step graph has been built yet. ``False`` until the
        first call, or until :attr:`step_graph`/:attr:`initial_state` is
        read."""
        return self._runner is not None or self._runner_fwd is not None

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
        if self._runner is None and self._runner_fwd is None:
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
        implementing the DLPack protocol (a torch tensor -- CPU, CUDA or
        ROCm/HIP) is accepted directly -- see this module's own docstring on
        why that avoids a copy, and :func:`onnxsim.backend.as_ort_value` for
        exactly what "implementing DLPack" buys here.
        """
        if self._runner is None and self._runner_fwd is None:
            self._compile()
        assert self._step is not None
        if self._runner_fwd is not None:
            return self._call_split(feeds, lr)
        assert self._runner is not None

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
        if self.quantize_forward:
            model = _quantize_forward_for_training(
                model, self.params, self.calibration_data
            )
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

        seed = b.const(np.array(self.loss_scale, dtype=np.float32), "loss_seed")
        n_fwd = len(b.nodes)
        grads = graph_grad.build_backward(
            b,
            nodes=list(model.graph.node),
            shapes=shapes,
            grad_outputs={self.loss_output: seed},
            targets=list(self.params),
        )
        if self.gradient_ranges is not None:
            self.gradient_report = grad_health.check_backward_health(
                b,
                shapes,
                grads,
                self.gradient_ranges,
                precision="fp16" if self.backward_precision == "float16" else "fp32",
                elem_types=elem_types,
            )
        if self.backward_precision == "float16":
            grads = _cast_backward_to_fp16(b, n_fwd, len(b.nodes), elem_types, grads)
        if self.loss_scale != 1.0:
            # Unscale: the seed above multiplies every gradient by the loss
            # scale (backprop is linear), so divide it back out before the
            # optimizer update. Exact for power-of-two scales; skipped
            # entirely at 1.0 so default graphs are byte-identical.
            inv = b.const(
                np.array(1.0 / self.loss_scale, dtype=np.float32), "loss_unscale"
            )
            grads = {p: b.mul(g, inv) for p, g in grads.items()}

        state: Dict[str, Tuple[Sequence[int], str]] = {}
        scalars = ["lr"]
        if self.optimizer == "adam":
            scalars += ["m_correction", "v_correction"]

        custom_fn: Optional[onnx.FunctionProto] = None
        if isinstance(self.optimizer, CustomOptimizer):
            custom_fn = _custom_optimizer_function(self.optimizer, b.name("custom_opt"))

        initial_state: Dict[str, np.ndarray] = {}
        for p in self.params:
            w_shape = _int_shape(shapes[p])
            grad = grads[p]
            if custom_fn is not None:
                assert isinstance(self.optimizer, CustomOptimizer)
                state_inputs = [b.name("s") for _ in range(self.optimizer.num_state)]
                outs = b.call(custom_fn, [p, grad, *state_inputs, "lr"])
                w_next, *state_next = outs
                for s_input, s_next in zip(state_inputs, state_next):
                    state[s_input] = (w_shape, s_next)
                    initial_state[s_input] = np.zeros(w_shape, dtype=np.float32)
            else:
                m_input = b.name("m")
                if self.optimizer == "adam":
                    v_input = b.name("v")
                    w_next, m_next, v_next = qat_graph.adam_update(
                        b,
                        p,
                        grad,
                        m_input,
                        v_input,
                        "lr",
                        "m_correction",
                        "v_correction",
                    )
                    state[v_input] = (w_shape, v_next)
                    initial_state[v_input] = np.zeros(w_shape, dtype=np.float32)
                else:
                    w_next, m_next = qat_graph.sgd_momentum_update(
                        b, p, grad, m_input, "lr"
                    )
                state[m_input] = (w_shape, m_next)
                initial_state[m_input] = np.zeros(w_shape, dtype=np.float32)
            state[p] = (w_shape, w_next)
            initial_state[p] = onnx.numpy_helper.to_array(initializers[p]).astype(
                np.float32
            )

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
        if self.forward_providers is None:
            self._runner = backend.Runner(
                self._step.model,
                output_names=list(self._step.state.values()) + [self.loss_output],
                providers=self.providers,
            )
            # Uploaded once, here, rather than on every call: from this point on
            # __call__'s onnxruntime path never sees a numpy array for its own
            # state again (see this module's own docstring).
            if self._runner.supports_ort_values():
                self._state = {
                    k: backend.as_ort_value(v) for k, v in initial_state.items()
                }
            else:
                self._state = initial_state
        else:
            self._build_split_runners(
                b,
                len(model.graph.node),
                model,
                shapes,
                elem_types,
                state,
                scalars,
                initial_state,
            )
            self._state = initial_state
        self._t = 0

    def _build_split_runners(
        self,
        b: qat_graph.GraphBuilder,
        n_fwd: int,
        model: onnx.ModelProto,
        shapes: Dict[str, Sequence[Union[int, str]]],
        elem_types: Dict[str, int],
        state: Dict[str, Tuple[Sequence[int], str]],
        scalars: Sequence[str],
        initial_state: Dict[str, np.ndarray],
    ) -> None:
        """Split execution: a forward session on :attr:`forward_providers`
        and a backward+optimizer session on :attr:`providers`.

        The builder holds forward nodes first (copied verbatim from the
        model) and everything derived after -- the split sits exactly on
        that boundary. Tensors the tail consumes from the head cross as
        plain graph outputs/inputs (``_needed``); trained parameters ride
        along as forward inputs (they are state, fed per step) and the fused
        :attr:`step_graph` above is still built (but never sessioned), so
        inspection, ``initial_state`` and ``export`` keep working unchanged.
        """
        fwd_nodes = list(b.nodes[:n_fwd])
        bwd_nodes = list(b.nodes[n_fwd:])
        fwd_out = {o for n in fwd_nodes for o in n.output if o}
        tail_in = {i for n in bwd_nodes for i in n.input if i}
        needed = sorted(t for t in (tail_in & fwd_out) if t != self.loss_output)

        inits = {t.name for t in b.initializer}

        def vinfo(name: str) -> onnx.ValueInfoProto:
            return onnx.helper.make_tensor_value_info(
                name,
                elem_types.get(name, onnx.TensorProto.FLOAT),
                _int_shape(shapes[name]),
            )

        # One opset_import per distinct function domain a templated gradient
        # rule (see graph_grad.py's "Templated rules" section) may have
        # registered on `b`, the same way make_step_graph does -- without
        # this, a call to e.g. GradMul is a node onnxruntime cannot resolve
        # ("No opset import for domain 'onnxsim.grad'").
        opset_imports = list(model.opset_import)
        if b.functions:
            domains = sorted({fn.domain for fn in b.functions})
            opset_imports += [onnx.helper.make_opsetid(d, 1) for d in domains]

        model_inputs = [i.name for i in model.graph.input if i.name not in inits]
        fwd_model = onnx.helper.make_model(
            onnx.helper.make_graph(
                fwd_nodes,
                "onnxsim_train_forward",
                [vinfo(n) for n in model_inputs]
                + [vinfo(p) for p in self.params if p not in model_inputs],
                [vinfo(self.loss_output)] + [vinfo(t) for t in needed],
                list(b.initializer),
            ),
            opset_imports=opset_imports,
            ir_version=model.ir_version,
        )
        bwd_inputs = (
            [vinfo(t) for t in needed]
            # State inputs (trained params and optimizer moments) carry
            # their own recorded shapes: backward-internal names like the
            # moment buffers never appear in the forward shapes dict.
            + [
                onnx.helper.make_tensor_value_info(
                    p, onnx.TensorProto.FLOAT, _int_shape(shape)
                )
                for p, (shape, _) in state.items()
            ]
            + [
                onnx.helper.make_tensor_value_info(s, onnx.TensorProto.FLOAT, [])
                for s in scalars
            ]
        )
        bwd_model = onnx.helper.make_model(
            onnx.helper.make_graph(
                bwd_nodes,
                "onnxsim_train_backward",
                bwd_inputs,
                [
                    onnx.helper.make_tensor_value_info(
                        out_name, onnx.TensorProto.FLOAT, _int_shape(shape)
                    )
                    for _, (shape, out_name) in state.items()
                ],
                list(b.initializer),
            ),
            opset_imports=opset_imports,
            ir_version=model.ir_version,
        )
        for proto in (fwd_model, bwd_model):
            proto.functions.extend(b.functions)
        if b.functions:
            # Expand every call site before either model reaches a runtime --
            # same reason make_step_graph does this for the fused path: no
            # execution provider needs to know about the private grad domain.
            fwd_model = onnx.inliner.inline_local_functions(fwd_model)
            bwd_model = onnx.inliner.inline_local_functions(bwd_model)
        self._needed = tuple(needed)
        self._bwd_state = {k: o for k, (_, o) in state.items()}
        self._runner_fwd = backend.Runner(
            fwd_model,
            output_names=[self.loss_output] + list(needed),
            providers=self.forward_providers,
        )
        self._runner_bwd = backend.Runner(
            bwd_model,
            output_names=[o for _, o in state.values()],
            providers=self.providers,
        )

    def _call_split(self, feeds: Dict[str, Any], lr: float) -> float:
        """One :meth:`__call__` step in split mode: forward on
        :attr:`forward_providers`, backward+optimizer on :attr:`providers`,
        boundary activations and state crossing as numpy."""
        assert self._runner_fwd is not None and self._runner_bwd is not None
        # Feed only what the forward session declares: a re-frozen forward
        # bakes the weights in as initializers instead of taking them as
        # state inputs (see the periodic re-freeze recipe), and extra feeds
        # are an error, not ignored. Without an onnxruntime session (the
        # reference-evaluator fallback) there is nothing to filter against.
        sess = getattr(self._runner_fwd, "_sess", None)
        get_inputs = getattr(sess, "get_inputs", None)
        declared = {i.name for i in get_inputs()} if get_inputs is not None else None
        fwd_in = {
            k: np.asarray(v, dtype=np.float32)
            for k, v in feeds.items()
            if declared is None or k in declared
        }
        for p in self.params:
            if declared is None or p in declared:
                fwd_in[p] = np.asarray(self._state[p], dtype=np.float32)
        fout = self._runner_fwd(fwd_in)
        bwd_in = {t: np.asarray(fout[t], dtype=np.float32) for t in self._needed}
        for k, v in self._state.items():
            bwd_in[k] = np.asarray(v, dtype=np.float32)
        bwd_in["lr"] = np.asarray(lr, dtype=np.float32)
        if self.optimizer == "adam":
            for name, value in qat_graph.adam_bias_corrections(self._t).items():
                bwd_in[name] = np.asarray(value, dtype=np.float32)
        out = self._runner_bwd(bwd_in)
        self._state = {
            input_name: out[output_name]
            for input_name, output_name in self._bwd_state.items()
        }
        self._t += 1
        return float(fout[self.loss_output])


def compile_training_loop(
    model: onnx.ModelProto,
    loss_output: str,
    params: Sequence[str],
    optimizer: Union[str, CustomOptimizer] = "adam",
    providers: Optional[Sequence[backend.Provider]] = None,
    quantize_forward: bool = False,
    calibration_data: Optional[Sequence[Dict[str, np.ndarray]]] = None,
    forward_providers: Optional[Sequence[backend.Provider]] = None,
    loss_scale: float = 1.0,
    backward_precision: str = "float32",
    gradient_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
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
    :param optimizer: ``"adam"`` (default), ``"sgd_momentum"``, or a
            :class:`CustomOptimizer` for an update rule neither builtin
            covers.
    :param providers: onnxruntime execution providers for the compiled step,
            in priority order. ``None`` means CPU.
    :param quantize_forward: train with an INT8-quantized forward and an
            fp32 backward -- see :attr:`TrainingLoop.quantize_forward`. Needs
            calibrated activation ranges: pass representative batches as
            :param:`calibration_data`, or omit it for random data (a smoke
            test only).
    :param calibration_data: representative input batches for
            :param:`quantize_forward`'s calibration, one
            ``{input_name: array}`` dict per batch. Ignored unless
            :param:`quantize_forward` is true.
    :param forward_providers: execution providers for the forward pass
            only -- see :attr:`TrainingLoop.forward_providers`. ``None``
            runs the whole fused step on :param:`providers`.
    :param loss_scale: static loss-scale factor, 1.0 (off) by default --
            see :attr:`TrainingLoop.loss_scale`.
    :param backward_precision: ``"float32"`` (default) or ``"float16"`` --
            see :attr:`TrainingLoop.backward_precision`. Combine ``"float16"``
            with a large :param:`loss_scale`: unscaled fp16 gradients
            underflow fast.
    :param gradient_ranges: input boxes for the gradient health check run at
            compile time -- see :attr:`TrainingLoop.gradient_ranges`.
    """
    return TrainingLoop(
        model=model,
        loss_output=loss_output,
        params=tuple(params),
        optimizer=optimizer,
        providers=providers,
        quantize_forward=quantize_forward,
        calibration_data=calibration_data,
        forward_providers=forward_providers,
        loss_scale=loss_scale,
        backward_precision=backward_precision,
        gradient_ranges=gradient_ranges,
    )
