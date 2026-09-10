"""A *real* ``torch.compile``-styled training loop: export an actual
``torch.nn.Module`` to ONNX via ``torch.export``'s FX graph, then hand the
result to :func:`onnxsim.compile_training_loop`.

:mod:`onnxsim.compile_training` gives an ONNX model the ``torch.compile``
calling convention (lazy compile-once, run-many) without any torch
dependency at all -- the model has to already be an ONNX ``ModelProto``.
This module is the on-ramp from an actual PyTorch model: it captures
``module``'s forward as an FX graph with ``torch.export.export`` (what
``torch.compile`` itself, and ``torch.onnx.export``'s modern exporter, both
build on) and converts that graph to ONNX with
``torch.onnx.export(..., dynamo=True)`` -- the dynamo/FX-based exporter,
not the older TorchScript tracer. What comes out the other end is an
ordinary :class:`onnxsim.compile_training.TrainingLoop`: every step after
that runs on onnxsim's own :mod:`onnxsim.graph_grad`/:mod:`onnxsim.qat_graph`
machinery, never on ``torch.autograd`` -- the forward graph is torch's, the
backward pass and the optimizer are onnxsim's.

``module.forward`` must return a single scalar tensor -- the loss -- the
same requirement :func:`onnxsim.compile_training_loop` already has for
``loss_output``, and for the same reason: a step graph has exactly one loss
output. Put the loss computation inside the module (``forward(self, x, y):
... ; return loss``) rather than composing it outside; there is no separate
loss-function export path here.

Needs ``torch >= 2.5`` (the ``onnxsim[torch-training]`` extra), the release
``torch.onnx.export``'s ``dynamo=True`` argument landed in. Not imported at
module load time -- only :func:`export_torch_module_to_onnx` and
:func:`compile_torch_training_loop` need it, and both raise a clear
:class:`ImportError` if it is missing rather than failing this module's own
import for every caller of :mod:`onnxsim.compile_training` who has no torch
installed at all.
"""

from __future__ import annotations

import copy
import inspect
import tempfile
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import onnx
import onnx.inliner

from onnxsim import backend
from onnxsim.compile_training import TrainingLoop, compile_training_loop

if TYPE_CHECKING:
    import torch

#: Positional example inputs (a tuple of tensors) or keyword ones (a dict),
#: the same two shapes :func:`torch.onnx.export` itself accepts for ``args``.
ExampleInputs = Union[Tuple["torch.Tensor", ...], Mapping[str, "torch.Tensor"]]

#: The two reductions torch's dynamo exporter lowers a full-tensor
#: `.mean()`/`.sum()` to -- see :func:`_fold_full_reduction_squeeze`.
_FULL_REDUCE_OPS = frozenset({"ReduceMean", "ReduceSum"})


def _inline_local_functions(model: onnx.ModelProto) -> onnx.ModelProto:
    """Expands every model-local ``FunctionProto`` call site into its own
    body nodes, via ``onnx.inliner.inline_local_functions`` -- the same call
    :func:`onnxsim.qat_graph.make_step_graph` already makes to expand
    :mod:`onnxsim.graph_grad`'s own templated rules before a step graph
    reaches a runtime.

    Which shape torch's dynamo exporter picks for a given op -- inline nodes
    directly, or a call to a local function (observed for both ``ReduceMean``
    and a whole ``aten::mean`` under different torch/onnxscript versions and
    platforms) -- is an implementation detail of that exporter version, not
    something a caller of ``torch.onnx.export`` controls. Left un-inlined, a
    function call node's own op type is whatever onnxscript named the
    function (``"aten_mean"``, not ``"ReduceMean"``), which
    :mod:`onnxsim.graph_grad` has no rule for under either name -- not
    because the operation is undifferentiable, but because it was never
    asked to differentiate the ops the function actually contains. Inlining
    first makes every version's export land at the same flat, function-free
    graph, so :func:`_fold_full_reduction_squeeze` and
    :func:`onnxsim.compile_training_loop`'s own differentiation see the same
    nodes regardless of which shape this particular export happened to take.

    A no-op, returning ``model`` unchanged, when there are no local
    functions to expand at all -- the common case, and the only one
    observed locally in this repository's own dev sandbox.
    """
    if not model.functions:
        return model
    return onnx.inliner.inline_local_functions(model)


def _fold_full_reduction_squeeze(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrites a full reduction's ``keepdims=1`` output immediately
    ``Squeeze``d back down to a scalar into one ``keepdims=0`` node.

    ``tensor.mean()`` with no explicit axis -- the ordinary way to compute a
    scalar loss, and so the single most common shape a torch training
    module's own forward takes -- is exactly this pair in torch's dynamo
    exporter's own decomposition: a ``ReduceMean`` that keeps every reduced
    axis as a literal size-1 dim, immediately followed by a ``Squeeze``
    dropping all of them. (``.sum()`` happens not to need this fold --
    observed to already lower straight to ``ReduceSum(keepdims=0)`` -- but
    nothing here assumes that stays true, hence covering both.)
    :mod:`onnxsim.graph_grad` has no gradient rule for ``Squeeze`` itself
    (nor should it grow one only for this: a *general* ``Squeeze`` can be
    squeezing an axis a caller cares to keep static-shape information about
    downstream, which the reduce-then-squeeze case never does), so left
    alone this decomposition would make ``compile_torch_training_loop`` fail
    on the single most ordinary training module with
    :class:`onnxsim.graph_grad.UnsupportedOpError` -- for an op ``graph_grad``
    already fully differentiates one attribute value away from being asked
    to.

    Only that one, narrow shape is rewritten, and only when it is
    unambiguous:

    - the reduce has no ``axes`` input/attribute of its own (opset >= 18
      carries it as an optional second input, opset < 18 as an attribute;
      absent either way means "every axis"),
    - its ``keepdims`` is 1,
    - its output feeds *only* the ``Squeeze`` (not also a graph output or
      another node), and
    - the ``Squeeze`` has no ``axes`` input either (meaning "every size-1
      axis" -- which, following an all-axes reduction, is every axis the
      reduce has).

    Any other combination -- an explicit ``axes`` on either node, more than
    one consumer, ``keepdims=0`` already -- is a real shape decision the
    graph is expressing and is left untouched; this is a fold of one
    specific decomposition artifact, not a general ``Squeeze`` eliminator.
    A build model unaffected by this pattern at all (no torch involved, or
    a torch forward that never reduces to a bare scalar this way) is
    returned unchanged, node for node.
    """
    producer: Dict[str, int] = {}
    consumer_counts: Dict[str, int] = {}
    for index, node in enumerate(model.graph.node):
        for output in node.output:
            if output:
                producer[output] = index
        for input_name in node.input:
            if input_name:
                consumer_counts[input_name] = consumer_counts.get(input_name, 0) + 1

    def has_axes(node: onnx.NodeProto) -> bool:
        if len(node.input) > 1 and node.input[1]:
            return True
        return any(attr.name == "axes" for attr in node.attribute)

    def keepdims(node: onnx.NodeProto) -> int:
        for attr in node.attribute:
            if attr.name == "keepdims":
                return attr.i
        return 1  # ReduceMean/ReduceSum's own default

    remove: set = set()
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Squeeze" or has_axes(node):
            continue
        reduce_index = producer.get(node.input[0])
        if reduce_index is None or reduce_index in remove:
            continue
        reduce_node = model.graph.node[reduce_index]
        if (
            reduce_node.op_type not in _FULL_REDUCE_OPS
            or has_axes(reduce_node)
            or keepdims(reduce_node) != 1
            or consumer_counts.get(reduce_node.output[0], 0) != 1
        ):
            continue
        reduce_node.output[0] = node.output[0]
        for attr in list(reduce_node.attribute):
            if attr.name == "keepdims":
                reduce_node.attribute.remove(attr)
        reduce_node.attribute.append(onnx.helper.make_attribute("keepdims", 0))
        remove.add(index)

    if not remove:
        return model
    kept_nodes: List[onnx.NodeProto] = [
        copy.deepcopy(n) for i, n in enumerate(model.graph.node) if i not in remove
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    return model


def _strip_default_noop_with_empty_axes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Drops a ``ReduceMean``/``ReduceSum`` node's ``noop_with_empty_axes``
    attribute when it is 0 (the schema's own default).

    ``noop_with_empty_axes`` is opset 18+ only, but onnxscript's own
    opset-downgrade converter cannot rewrite every graph torch's dynamo
    exporter produces (observed directly: asking
    :func:`export_torch_module_to_onnx` for ``opset_version=17`` still comes
    back an opset 18 model, with a logged warning that the downgrade
    "fallback is enabled"), and the reduce nodes it emits carry this
    attribute regardless -- even set to 0, its own no-op value, on every one
    observed. :func:`onnxsim.compile_training.TrainingLoop._compile` fixes
    its step graph's own ``opset_import`` to 17 no matter what a forward
    model declares (:mod:`onnxsim.qat_graph`'s own long-standing pairing,
    unrelated to torch and unaffected by this), so a node that is only
    legal from opset 18 on fails there, at session creation, with
    onnxruntime's own "Unrecognized attribute" error -- not a wrong answer,
    but a needless failure for an attribute whose value never differs from
    simply leaving it off. Dropping it when it is 0 changes nothing about
    what the node computes (the schema default over both opsets is exactly
    this: do not special-case empty ``axes``) and everything about whether
    onnxruntime accepts it at opset 17.

    A value of 1 is never stripped: that is a real behavior difference
    (treat an empty ``axes`` as a no-op instead of "reduce every axis"),
    which the fold above never produces and which torch's own exporter has
    no reason to either, but this function does not assume that and leaves
    a 1 exactly where it finds one.
    """
    for node in model.graph.node:
        if node.op_type not in _FULL_REDUCE_OPS:
            continue
        for attr in list(node.attribute):
            if attr.name == "noop_with_empty_axes" and attr.i == 0:
                node.attribute.remove(attr)
    return model


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "onnxsim.torch_training needs torch installed -- "
            "`pip install onnxsim[torch-training]` or `pip install 'torch>=2.5'`"
        ) from error
    if "dynamo" not in inspect.signature(torch.onnx.export).parameters:
        # A torch old enough to lack the dynamo=True argument entirely (pre-2.5,
        # where the FX-based exporter was the separate torch.onnx.dynamo_export()
        # function) would otherwise fail deep inside export_torch_module_to_onnx
        # with a confusing "unexpected keyword argument" -- refused here, at the
        # one place that already knows why.
        raise ImportError(
            f"onnxsim.torch_training needs torch >= 2.5 (found {torch.__version__}), "
            "the release torch.onnx.export's dynamo=True argument landed in"
        )
    return torch


def export_torch_module_to_onnx(
    module: "torch.nn.Module",
    example_inputs: ExampleInputs,
    *,
    input_names: Optional[Sequence[str]] = None,
    output_names: Sequence[str] = ("loss",),
    opset_version: int = 17,
) -> onnx.ModelProto:
    """Exports ``module`` to ONNX via ``torch.export``'s FX graph.

    ``module(*example_inputs)`` (or ``module(**example_inputs)`` for a dict)
    must run and return a single scalar tensor -- see this module's own
    docstring for why. Every input's shape is taken as static from
    ``example_inputs`` -- :func:`onnxsim.compile_training_loop` needs every
    shape known at compile time, so nothing here asks ``torch.export`` for a
    dynamic one.

    :param input_names: names for the ONNX graph's inputs, in the order
            ``example_inputs`` provides them (positional) or as given
            (keyword). Left as ``torch.export``'s own default parameter
            names when not given -- pass this when you want to control what
            :meth:`onnxsim.compile_training.TrainingLoop.__call__`'s
            ``feeds`` dict keys are.
    :param output_names: the model's own output names; must have exactly one
            entry, since only a single scalar loss is exported. Its own
            single entry is what :func:`compile_torch_training_loop` passes
            through as ``loss_output``.
    :param opset_version: ONNX opset to export at. 17 (the default) matches
            :mod:`onnxsim.qat_graph`'s own step-graph opset, so a node this
            export could not legally carry is refused here rather than
            later, inside :func:`onnxsim.compile_training_loop`'s own
            compile step.
    """
    torch = _import_torch()
    if len(output_names) != 1:
        raise ValueError(
            f"a training loop has exactly one loss output, got output_names="
            f"{list(output_names)!r}"
        )

    if isinstance(example_inputs, Mapping):
        args: Tuple[Any, ...] = ()
        kwargs: Dict[str, Any] = dict(example_inputs)
    else:
        args = tuple(example_inputs)
        kwargs = {}

    # Exported in eval mode, and restored to whatever mode the caller had it
    # in afterward. A step graph is one fixed computation, re-run as-is on
    # every call; eval mode is the only one where "the module's own forward"
    # already means that (BatchNorm's running stats fixed, Dropout off) --
    # training mode's own semantics (updating running stats as a side
    # effect, a randomly sampled mask) have no static-graph analogue this
    # module attempts, so exporting in training mode would silently bake in
    # one particular Dropout mask or a training-mode BatchNorm forward
    # nothing here ever threads the running-stat update for.
    was_training = module.training
    module.eval()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "model.onnx"
            try:
                torch.onnx.export(
                    module,
                    args,
                    str(onnx_path),
                    kwargs=kwargs,
                    input_names=list(input_names) if input_names is not None else None,
                    output_names=list(output_names),
                    opset_version=opset_version,
                    dynamo=True,
                    # No axis of any input is dynamic: every step graph
                    # onnxsim.compile_training_loop builds has fixed shapes
                    # throughout, so there is nothing to gain from tracing
                    # one as symbolic and it would only fail later, inside
                    # that function's own shape-inference check, with a less
                    # specific error.
                    dynamic_shapes=None,
                    # Both off, and for the same reason: the exporter's
                    # default optimization pass constant-folds a parameter
                    # used in a cheap enough expression (module's own w.T,
                    # here) straight into a renamed initializer --
                    # "permute", not "w" -- severing the link
                    # compile_torch_training_loop's own params= default
                    # (module.named_parameters()' qualified names) relies
                    # on. Leaving both off keeps every nn.Parameter a
                    # distinct, identically-named initializer, at the cost
                    # of a few extra nodes (an explicit Transpose here) that
                    # a real runtime would have folded away anyway -- this
                    # graph is differentiated and trained, never shipped
                    # as-is.
                    optimize=False,
                    do_constant_folding=False,
                )
            except ImportError as error:
                # torch.onnx's dynamo exporter imports onnxscript lazily,
                # deep inside torch.onnx.export itself -- surfacing here as
                # a bare "No module named 'onnxscript'" with no onnxsim
                # frame in the traceback at all if it is missing. Reraised
                # with the same install hint _import_torch already gives
                # for torch itself.
                raise ImportError(
                    "onnxsim.torch_training needs onnxscript installed too "
                    "(torch.onnx's own dynamo exporter dependency) -- "
                    "`pip install onnxsim[torch-training]` or "
                    f"`pip install onnxscript`: {error}"
                ) from error
            model = _inline_local_functions(onnx.load(str(onnx_path)))
            model = _fold_full_reduction_squeeze(model)
            return _strip_default_noop_with_empty_axes(model)
    finally:
        module.train(was_training)


def compile_torch_training_loop(
    module: "torch.nn.Module",
    example_inputs: ExampleInputs,
    loss_output: str = "loss",
    params: Optional[Sequence[str]] = None,
    optimizer: str = "adam",
    providers: Optional[Sequence[backend.Provider]] = None,
    opset_version: int = 17,
) -> TrainingLoop:
    """Wraps a real ``torch.nn.Module`` as a torch.compile-styled training
    loop, exported to ONNX and trained entirely by onnxsim's own grad
    templating -- never by ``torch.autograd``.

    .. code-block:: python

        class Regression(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(2, 3))

            def forward(self, x, y):
                y_hat = x @ self.w.T
                return ((y_hat - y) ** 2).mean()

        loop = onnxsim.compile_torch_training_loop(
            Regression(), (torch.zeros(8, 3), torch.zeros(8, 2))
        )
        for x, y in batches:
            loss = loop({"x": x.numpy(), "y": y.numpy()}, lr=1e-3)

    The returned :class:`~onnxsim.compile_training.TrainingLoop` is an
    ordinary one -- its ``__call__`` still takes numpy feeds, exactly as
    :func:`onnxsim.compile_training_loop` returns for a caller who already
    had an ONNX model. Only *building* the loop goes through torch; running
    it does not need torch installed at all, importable or not.

    :param module: exported via :func:`export_torch_module_to_onnx`, whose
            own docstring covers ``module.forward``'s single-scalar-output
            requirement, and ``example_inputs``'s two accepted shapes.
    :param loss_output: name for the model's single scalar output. Purely a
            label for the exported graph -- there is nothing to match it
            against on the torch side.
    :param params: names of the trained parameters, matching
            :func:`onnxsim.compile_training_loop`'s own ``params``. Defaults
            to every one of ``module.named_parameters()``'s qualified names
            (``"linear.weight"``, not ``"weight"``) -- what the dynamo
            exporter names the corresponding ONNX initializer, unlike the
            older TorchScript-based exporter, which does not preserve
            parameter names at all. Raises if a name from that default (or a
            caller-supplied ``params``) is not actually one of the exported
            model's initializers, rather than silently training a subset.
    :param optimizer: ``"adam"`` (default) or ``"sgd_momentum"``, passed
            through to :func:`onnxsim.compile_training_loop`.
    :param providers: onnxruntime execution providers for the compiled step,
            passed through to :func:`onnxsim.compile_training_loop`.
    :param opset_version: passed through to
            :func:`export_torch_module_to_onnx`.
    """
    model = export_torch_module_to_onnx(
        module,
        example_inputs,
        output_names=(loss_output,),
        opset_version=opset_version,
    )

    if params is None:
        params = [name for name, _ in module.named_parameters()]
    initializer_names = {init.name for init in model.graph.initializer}
    missing = [p for p in params if p not in initializer_names]
    if missing:
        raise ValueError(
            f"{missing} are not initializers of the exported ONNX model "
            f"(it has {sorted(initializer_names)}); the dynamo exporter may "
            "have folded, renamed, or dropped them -- pass params= explicitly "
            "to name the ones that survived export"
        )

    return compile_training_loop(
        model, loss_output, params, optimizer=optimizer, providers=providers
    )
