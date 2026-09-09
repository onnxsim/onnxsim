"""LoRA (low-rank adapter) injection and training -- onnxsim's own answer to
what ``tools/onnx-finetune`` needs a training-enabled ONNX Runtime build for.

``tools/onnx-finetune`` injects a trainable ``X @ A @ B`` branch around a
frozen weight and trains ``A``/``B`` via ``onnxruntime.training.artifacts``,
which needs ONNX Runtime built from source with ``--enable_training`` --
unavailable via ``pip install onnxruntime``. This module does the same graph
surgery, but trains the adapter with :mod:`onnxsim.graph_grad`'s hand-rolled
reverse-mode autodiff instead: the same machinery :mod:`onnxsim.qat` already
uses for block-wise quantization-reconstruction, which differentiates a
forward slice into an ordinary ONNX step graph and so runs on any inference
runtime, not just a training-enabled one.

**Why this needs (almost) no new gradient machinery.**
:func:`onnxsim.graph_grad.build_backward` already treats anything not in its
``targets`` list as frozen -- exactly "freeze the base weight ``W``, train
only the small ``A``/``B`` matrices" LoRA needs. Injection (this module's own
job) never modifies ``W``; training just asks ``build_backward`` for the
gradients of ``A``/``B`` alone, and the base branch's own nodes are appended
to the step graph verbatim, generating whatever gradient nodes reaching
``A``/``B`` requires along the way, exactly like reaching a QAT block's
trained weight does today.

**What this is not.** Reference-model distillation (:func:`train_lora` with
``reference_model=``) can only ever teach an adapter to reproduce that
reference -- see :func:`onnxsim.qat.apply_block_finetune`'s own docstring for
why that alone is not "fine-tuning on a new task." ``target_data=`` is the
escape hatch: a caller-supplied label tensor drives the same MSE step graph
directly, which is the actual point of LoRA fine-tuning in practice.

**Injection is plain graph surgery, not a step graph.** It follows
:func:`onnxsim.nf4.quantize_weight_only_nf4`'s own style -- edit an existing,
already-deployed ``ModelProto`` once with :mod:`onnx.helper`/
:mod:`onnx.numpy_helper`, not :class:`onnxsim.qat_graph.GraphBuilder` (which
exists for one-shot step-graph construction, not editing a model that is
already meant to be run as-is).

See ``tools/onnx-finetune/scripts/lora_surgery.py`` for the tool this ports
from. **QLoRA composition** (:func:`apply_qlora`) needs one extra step:
:func:`onnxsim.nf4.quantize_weight_only_nf4`'s dequantization chain includes
a ``Cast``, which has no rule in :data:`onnxsim.graph_grad.SUPPORTED_OPS` --
even though nothing needs a gradient through it, since it feeds the frozen
base weight, never the LoRA branch. :func:`_fold_frozen_prefixes` handles
this generically: any node whose entire input closure is constants (not the
adapter's own ``A``/``B``) is evaluated once and folded into a plain
initializer before ``build_backward`` ever sees it, which is exactly what a
step graph -- where the base weight never changes across steps anyway --
should do with it regardless of which quantization scheme produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend, graph_grad, nf4, qat, qat_graph
from onnxsim.bias_correction import _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data

# Every name this module introduces into a step graph starts here, so it
# cannot collide with a tensor name carried over from the model -- the same
# convention onnxsim.qat's own "qat__" prefix follows, kept distinct so the
# two never collide if a caller ever mixed both in one graph.
_PREFIX = "lora__"

_ELIGIBLE_OP_TYPES = ("MatMul", "Gemm", "Conv")


@dataclass
class LoraTarget:
    """One injected adapter, tied to the base weight it augments."""

    #: The frozen base initializer name. Never modified by injection or
    #: training -- the whole point of the low-rank branch.
    weight_name: str
    #: The original node's output tensor name -- what the closing ``Add``
    #: restores, so every existing downstream consumer needs no changes.
    node_output: str
    op_type: str  # "MatMul" | "Gemm" | "Conv"
    #: New initializer name, ``f"{weight_name}.lora_A"``.
    lora_a_name: str
    #: New initializer name, ``f"{weight_name}.lora_B"``.
    lora_b_name: str
    rank: int
    alpha: Optional[float]


@dataclass
class LoraAdapter:
    """Every adapter one :func:`inject_lora` call injected."""

    targets: List[LoraTarget] = field(default_factory=list)

    def parameter_names(self) -> List[str]:
        """Every ``lora_A``/``lora_B`` initializer name -- the ``targets``
        list :func:`train_lora` hands to :func:`onnxsim.graph_grad.build_backward`,
        and the ``skip_names`` :func:`apply_qlora` passes to
        :func:`onnxsim.nf4.quantize_weight_only_nf4`."""
        names: List[str] = []
        for t in self.targets:
            names.append(t.lora_a_name)
            names.append(t.lora_b_name)
        return names


def _attr_ints(node: onnx.NodeProto, name: str, default: List[int]) -> List[int]:
    for a in node.attribute:
        if a.name == name:
            return list(a.ints)
    return default


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for a in node.attribute:
        if a.name == name:
            return int(a.i)
    return default


def _scale_initializer(
    rank: int, alpha: Optional[float], taken_names: set
) -> Optional[Tuple[str, np.ndarray]]:
    """A scalar ``float32`` initializer holding ``alpha / rank``, or ``None``
    when ``alpha`` is not given -- the branch is then used unscaled, matching
    ``lora_surgery.py``'s own "optionally scaled" convention."""
    if alpha is None:
        return None
    name = _unique_name("lora_alpha_over_rank", taken_names)
    value = np.asarray(alpha / rank, dtype=np.float32)
    return name, value


def _inject_matmul(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    w_name: str,
    w_dims: Sequence[int],
    rank: int,
    alpha: Optional[float],
    rng: np.random.Generator,
    taken_names: set,
) -> LoraTarget:
    """``Y = X @ W``, ``W: [K, N]``. Branch: ``X @ A @ B``, ``A: [K, rank]``
    (Kaiming-normal, ``std = 1/sqrt(K)``), ``B: [rank, N]`` (zeros -- the
    branch is a numeric no-op until trained)."""
    k, n = int(w_dims[0]), int(w_dims[1])
    x_name = node.input[0]
    orig_output = node.output[0]

    a_name = _unique_name(f"{w_name}.lora_A", taken_names)
    a_value = (rng.standard_normal((k, rank)) / np.sqrt(k)).astype(np.float32)
    graph.initializer.append(onnx.numpy_helper.from_array(a_value, name=a_name))
    b_name = _unique_name(f"{w_name}.lora_B", taken_names)
    b_value = np.zeros((rank, n), dtype=np.float32)
    graph.initializer.append(onnx.numpy_helper.from_array(b_value, name=b_name))

    base_out = _unique_name(f"{w_name}.lora_base_out", taken_names)
    node.output[0] = base_out

    a_out = _unique_name(f"{w_name}.lora_a_out", taken_names)
    a_node = onnx.helper.make_node("MatMul", [x_name, a_name], [a_out])
    ab_out = _unique_name(f"{w_name}.lora_ab_out", taken_names)
    ab_node = onnx.helper.make_node("MatMul", [a_out, b_name], [ab_out])
    new_nodes = [a_node, ab_node]
    delta = ab_out

    scaled = _scale_initializer(rank, alpha, taken_names)
    if scaled is not None:
        scale_name, scale_value = scaled
        graph.initializer.append(
            onnx.numpy_helper.from_array(scale_value, name=scale_name)
        )
        scaled_out = _unique_name(f"{w_name}.lora_scaled", taken_names)
        new_nodes.append(
            onnx.helper.make_node("Mul", [ab_out, scale_name], [scaled_out])
        )
        delta = scaled_out

    new_nodes.append(onnx.helper.make_node("Add", [base_out, delta], [orig_output]))
    _insert_after(graph, node, new_nodes)

    return LoraTarget(
        weight_name=w_name,
        node_output=orig_output,
        op_type="MatMul",
        lora_a_name=a_name,
        lora_b_name=b_name,
        rank=rank,
        alpha=alpha,
    )


def _inject_gemm(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    w_name: str,
    w_dims: Sequence[int],
    rank: int,
    alpha: Optional[float],
    rng: np.random.Generator,
    taken_names: set,
) -> LoraTarget:
    """``Y = alpha*A'@B' + beta*C``, ``A' = X^T if transA else X``,
    ``B' = W^T if transB else W``. ``W``'s raw shape is ``[N, K]`` when
    ``transB`` else ``[K, N]``; the branch always computes in ``[K, N]``
    layout, so a ``transA`` input gets one extra ``Transpose`` before it
    feeds the branch (mirroring ``lora_surgery.py``'s own handling) -- the
    base node's own ``transA``/``transB``/``alpha``/``beta`` are untouched,
    since the branch is added to its already-computed output, not folded
    into it."""
    trans_a = _attr_int(node, "transA", 0)
    trans_b = _attr_int(node, "transB", 0)
    if trans_b:
        n, k = int(w_dims[0]), int(w_dims[1])
    else:
        k, n = int(w_dims[0]), int(w_dims[1])

    x_name = node.input[0]
    orig_output = node.output[0]
    branch_input = x_name
    new_nodes: List[onnx.NodeProto] = []
    if trans_a:
        branch_input = _unique_name(f"{w_name}.lora_xT", taken_names)
        new_nodes.append(
            onnx.helper.make_node("Transpose", [x_name], [branch_input], perm=[1, 0])
        )

    a_name = _unique_name(f"{w_name}.lora_A", taken_names)
    a_value = (rng.standard_normal((k, rank)) / np.sqrt(k)).astype(np.float32)
    graph.initializer.append(onnx.numpy_helper.from_array(a_value, name=a_name))
    b_name = _unique_name(f"{w_name}.lora_B", taken_names)
    b_value = np.zeros((rank, n), dtype=np.float32)
    graph.initializer.append(onnx.numpy_helper.from_array(b_value, name=b_name))

    base_out = _unique_name(f"{w_name}.lora_base_out", taken_names)
    node.output[0] = base_out

    a_out = _unique_name(f"{w_name}.lora_a_out", taken_names)
    new_nodes.append(onnx.helper.make_node("MatMul", [branch_input, a_name], [a_out]))
    ab_out = _unique_name(f"{w_name}.lora_ab_out", taken_names)
    new_nodes.append(onnx.helper.make_node("MatMul", [a_out, b_name], [ab_out]))
    delta = ab_out

    scaled = _scale_initializer(rank, alpha, taken_names)
    if scaled is not None:
        scale_name, scale_value = scaled
        graph.initializer.append(
            onnx.numpy_helper.from_array(scale_value, name=scale_name)
        )
        scaled_out = _unique_name(f"{w_name}.lora_scaled", taken_names)
        new_nodes.append(
            onnx.helper.make_node("Mul", [ab_out, scale_name], [scaled_out])
        )
        delta = scaled_out

    new_nodes.append(onnx.helper.make_node("Add", [base_out, delta], [orig_output]))
    _insert_after(graph, node, new_nodes)

    return LoraTarget(
        weight_name=w_name,
        node_output=orig_output,
        op_type="Gemm",
        lora_a_name=a_name,
        lora_b_name=b_name,
        rank=rank,
        alpha=alpha,
    )


def _inject_conv1x1(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    w_name: str,
    w_dims: Sequence[int],
    rank: int,
    alpha: Optional[float],
    rng: np.random.Generator,
    taken_names: set,
) -> LoraTarget:
    """``W: [out_ch, in_ch, 1, 1]``, ``kernel_shape == [1, 1]``,
    ``group == 1`` (both already checked by the caller). Branch:
    ``Conv(X, A) -> Conv(., B)``, ``A: [rank, in_ch, 1, 1]``,
    ``B: [out_ch, rank, 1, 1]``. The first branch conv carries the base
    node's own stride/pads/dilations (so its output lines up spatially with
    the base branch's); the second is a plain 1x1/stride-1 conv, since all
    spatial downsampling already happened in the first."""
    out_ch, in_ch = int(w_dims[0]), int(w_dims[1])
    x_name = node.input[0]
    orig_output = node.output[0]

    strides = _attr_ints(node, "strides", [1, 1])
    pads = _attr_ints(node, "pads", [0, 0, 0, 0])
    dilations = _attr_ints(node, "dilations", [1, 1])

    a_name = _unique_name(f"{w_name}.lora_A", taken_names)
    a_value = (rng.standard_normal((rank, in_ch, 1, 1)) / np.sqrt(in_ch)).astype(
        np.float32
    )
    graph.initializer.append(onnx.numpy_helper.from_array(a_value, name=a_name))
    b_name = _unique_name(f"{w_name}.lora_B", taken_names)
    b_value = np.zeros((out_ch, rank, 1, 1), dtype=np.float32)
    graph.initializer.append(onnx.numpy_helper.from_array(b_value, name=b_name))

    base_out = _unique_name(f"{w_name}.lora_base_out", taken_names)
    node.output[0] = base_out

    a_out = _unique_name(f"{w_name}.lora_a_out", taken_names)
    a_node = onnx.helper.make_node(
        "Conv",
        [x_name, a_name],
        [a_out],
        kernel_shape=[1, 1],
        strides=strides,
        pads=pads,
        dilations=dilations,
        group=1,
    )
    ab_out = _unique_name(f"{w_name}.lora_ab_out", taken_names)
    ab_node = onnx.helper.make_node(
        "Conv", [a_out, b_name], [ab_out], kernel_shape=[1, 1], group=1
    )
    new_nodes = [a_node, ab_node]
    delta = ab_out

    scaled = _scale_initializer(rank, alpha, taken_names)
    if scaled is not None:
        scale_name, scale_value = scaled
        graph.initializer.append(
            onnx.numpy_helper.from_array(scale_value, name=scale_name)
        )
        scaled_out = _unique_name(f"{w_name}.lora_scaled", taken_names)
        new_nodes.append(
            onnx.helper.make_node("Mul", [ab_out, scale_name], [scaled_out])
        )
        delta = scaled_out

    new_nodes.append(onnx.helper.make_node("Add", [base_out, delta], [orig_output]))
    _insert_after(graph, node, new_nodes)

    return LoraTarget(
        weight_name=w_name,
        node_output=orig_output,
        op_type="Conv",
        lora_a_name=a_name,
        lora_b_name=b_name,
        rank=rank,
        alpha=alpha,
    )


def _insert_after(
    graph: onnx.GraphProto, node: onnx.NodeProto, new_nodes: Sequence[onnx.NodeProto]
) -> None:
    """Splices ``new_nodes`` into ``graph.node`` immediately after ``node``,
    in order -- so the branch's own inputs (``node``'s original input, still
    available; the new initializers, always available) are already produced
    by the time each new node reads them, and the closing ``Add`` (last in
    ``new_nodes``) comes after everything it reads. Same technique
    :func:`onnxsim.nf4.quantize_weight_only_nf4` uses to splice its own
    dequant chain in."""
    insertion_point = next(i for i, n in enumerate(graph.node) if n is node) + 1
    for new_node in new_nodes:
        graph.node.insert(insertion_point, new_node)
        insertion_point += 1


def inject_lora(
    model: Union[str, onnx.ModelProto],
    rank: int = 8,
    alpha: Optional[float] = None,
    target_op_types: Sequence[str] = _ELIGIBLE_OP_TYPES,
    target_names: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Tuple[onnx.ModelProto, LoraAdapter]:
    """Injects a trainable low-rank adapter branch around every eligible
    ``MatMul``/``Gemm``/``Conv`` weight, leaving the base weight itself
    untouched and every other byte of the model unchanged.

    Eligible: ``MatMul``/``Gemm`` with a 2-D ``float32`` initializer at
    ``input[1]``; ``Conv`` with a 4-D ``float32`` initializer,
    ``kernel_shape == [1, 1]`` and ``group == 1`` (mirroring
    ``tools/onnx-finetune``'s own ``lora_surgery.py`` conditions, not
    :func:`onnxsim.qat._find_float_layers`'s looser ones -- that function
    admits any-shape ``Conv``, which a low-rank branch cannot represent).

    :param model: the model to inject into, or a file path.
    :param rank: the adapter's inner dimension.
    :param alpha: when given, the branch is scaled by ``alpha / rank``
            before being added to the base branch's output (LoRA's usual
            convention); when ``None``, the branch is added unscaled.
    :param target_op_types: restrict injection to these op types.
    :param target_names: restrict injection to weights with these initializer
            names; ``None`` means every eligible node.
    :param seed: seeds ``A``'s Kaiming-normal initialization. ``B`` always
            starts at zero, so injection is a numeric no-op until trained --
            checked directly by ``tests/test_lora.py``.
    :returns: ``(model with adapters injected, the injected LoraAdapter)``.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    taken_names = _all_names(graph)
    rng = np.random.default_rng(seed)

    adapter = LoraAdapter()
    for node in list(graph.node):
        if node.op_type not in target_op_types or len(node.input) < 2:
            continue
        w_name = node.input[1]
        if target_names is not None and w_name not in target_names:
            continue
        w_init = initializer_map.get(w_name)
        if w_init is None or w_init.data_type != onnx.TensorProto.FLOAT:
            continue

        if node.op_type == "MatMul":
            if len(w_init.dims) != 2:
                continue
            target = _inject_matmul(
                graph, node, w_name, w_init.dims, rank, alpha, rng, taken_names
            )
        elif node.op_type == "Gemm":
            if len(w_init.dims) != 2:
                continue
            target = _inject_gemm(
                graph, node, w_name, w_init.dims, rank, alpha, rng, taken_names
            )
        else:  # "Conv"
            if len(w_init.dims) != 4:
                continue
            kernel_shape = _attr_ints(node, "kernel_shape", list(w_init.dims[2:]))
            group = _attr_int(node, "group", 1)
            if kernel_shape != [1, 1] or group != 1:
                continue
            target = _inject_conv1x1(
                graph, node, w_name, w_init.dims, rank, alpha, rng, taken_names
            )
        adapter.targets.append(target)

    onnx.checker.check_model(out)
    return out, adapter


def _fold_frozen_prefixes(
    nodes: Sequence[onnx.NodeProto],
    model: onnx.ModelProto,
    non_foldable_names: Sequence[str],
) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto]]:
    """Constant-folds every node in ``nodes`` whose entire (transitive) input
    closure is initializers other than ``non_foldable_names`` -- in
    practice, a quantization scheme's dequantization chain feeding a frozen
    base weight (e.g. :func:`onnxsim.nf4.quantize_weight_only_nf4`'s
    ``Cast -> Gather -> Reshape -> Reshape -> Mul -> Reshape``), which may
    use an op :data:`onnxsim.graph_grad.SUPPORTED_OPS` has no rule for
    (``Cast``) even though nothing needs a gradient through it: the LoRA
    branch reads the block's own input, never the dequant chain's output.

    Deliberately general rather than NF4-specific: any node whose inputs are
    all constants is safe to fold regardless of which op or which
    quantization scheme produced them, and this covers "LoRA on top of any
    scheme with a non-differentiable dequant chain" uniformly.

    ``non_foldable_names`` must include every LoRA target's own ``A``/``B``
    initializer name -- these are trained, not fixed, so a node reading one
    must never be folded into a constant, however constant-looking its other
    inputs are.

    :returns: ``(nodes with the folded run removed, new initializers holding
            the folded values -- empty when nothing was foldable)``.
    """
    initializer_map = {t.name: t for t in model.graph.initializer}
    constant_names = set(initializer_map) - set(non_foldable_names)

    foldable_outputs = set(constant_names)
    folded_nodes: List[onnx.NodeProto] = []
    kept_nodes: List[onnx.NodeProto] = []
    for node in nodes:
        if node.input and all(
            (not name) or name in foldable_outputs for name in node.input
        ):
            folded_nodes.append(node)
            foldable_outputs.update(name for name in node.output if name)
        else:
            kept_nodes.append(node)

    if not folded_nodes:
        return list(nodes), []

    folded_output_names = {name for n in folded_nodes for name in n.output if name}
    boundary = sorted(
        {
            name
            for node in kept_nodes
            for name in node.input
            if name in folded_output_names
        }
    )
    if not boundary:
        # Every consumer of the fold was itself folded away too -- nothing a
        # kept node still needs, so there is nothing to materialize.
        return kept_nodes, []

    used_initializers = [
        initializer_map[name]
        for node in folded_nodes
        for name in node.input
        if name in initializer_map
    ]
    fold_graph = onnx.helper.make_graph(
        folded_nodes,
        "lora_fold",
        [],
        # A bare name, no declared type -- the same technique
        # onnxsim.bias_correction._add_probe_outputs uses to expose an
        # intermediate tensor as an output without knowing its type/shape
        # ahead of time.
        [onnx.ValueInfoProto(name=name) for name in boundary],
        initializer=used_initializers,
    )
    fold_model = onnx.helper.make_model(
        fold_graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    fold_model.ir_version = 8
    values = backend.run_model(fold_model, {}, providers=None)

    extra_initializers = [
        onnx.numpy_helper.from_array(np.asarray(values[name]), name=name)
        for name in boundary
    ]
    return kept_nodes, extra_initializers


def _build_lora_step_graph(
    adapter: LoraAdapter,
    nodes: Sequence[onnx.NodeProto],
    shapes: Dict[str, Sequence[int]],
    block_initializers: Sequence[onnx.TensorProto],
    externals: Dict[str, np.ndarray],
    block_output_name: str,
    block_output_shape: Sequence[int],
    batch: Optional["qat._Minibatch"] = None,
) -> qat_graph.StepGraph:
    """The whole LoRA training step as one graph: block forward (base branch
    + injected adapter branch, verbatim -- no substitution, since the base
    weight is never a target), reconstruction loss, backward restricted to
    the adapter's own ``A``/``B`` tensors, one Adam step each.

    Modeled directly on :func:`onnxsim.qat._build_step_graph`'s ordering,
    without the fake-quant/scale/activation-quantizer machinery that exists
    only because that function's caller has a *quantizer* to train --
    LoRA does not, so there is no substitution step: :func:`inject_lora`
    already left the block's nodes exactly as they should run.
    """
    b = qat_graph.GraphBuilder(_PREFIX)
    b.initializer.extend(block_initializers)

    teacher = f"{_PREFIX}teacher"
    constants: Dict[str, Tuple[Sequence[int], int]] = {}
    if batch is None:
        constants.update(
            {
                name: (list(value.shape), qat._np_elem_type(value.dtype))
                for name, value in sorted(externals.items())
            }
        )
        constants[teacher] = (list(block_output_shape), onnx.TensorProto.FLOAT)
    else:
        rows = batch.index_name
        for name, value in sorted(externals.items()):
            table = f"{_PREFIX}all_{name}"
            constants[table] = (list(value.shape), qat._np_elem_type(value.dtype))
            b.gather_rows(table, rows, name)
        constants[f"{_PREFIX}teacher_all"] = (
            list(block_output_shape),
            onnx.TensorProto.FLOAT,
        )
        b.gather_rows(f"{_PREFIX}teacher_all", rows, teacher)
        block_output_shape = [batch.size] + list(block_output_shape)[1:]

    b.nodes.extend(nodes)

    diff = b.sub(block_output_name, teacher)
    n_elems = int(np.prod(list(block_output_shape)))
    dl_dy = b.mul(diff, b.const(2.0 / n_elems))

    targets = adapter.parameter_names()
    grads = graph_grad.build_backward(
        b, nodes, shapes, {block_output_name: dl_dy}, targets
    )

    state: Dict[str, Tuple[Sequence[int], str]] = {}
    for param_name in targets:
        g = grads[param_name]
        shape = list(shapes[param_name])
        m_in, v_in = f"{_PREFIX}m_{param_name}", f"{_PREFIX}v_{param_name}"
        param_next, m_next, v_next = qat_graph.adam_update(
            b, param_name, g, m_in, v_in, f"{_PREFIX}lr", "m_correction", "v_correction"
        )
        state[param_name] = (shape, param_next)
        state[m_in] = (shape, m_next)
        state[v_in] = (shape, v_next)

    per_step: Optional[Dict[str, Tuple[Sequence[int], int]]] = None
    if batch is not None:
        per_step = {batch.index_name: ([batch.size], int(onnx.TensorProto.INT64))}

    return qat_graph.make_step_graph(
        b,
        constants=constants,
        state=state,
        scalars=[f"{_PREFIX}lr", "m_correction", "v_correction"],
        loss=b.mean_square(diff),
        name="onnxsim_lora_step",
        per_step=per_step,
    )


def _train_lora_block(
    model: onnx.ModelProto,
    adapter: LoraAdapter,
    nodes: Sequence[onnx.NodeProto],
    extra_initializers: Sequence[onnx.TensorProto],
    external_values: Dict[str, np.ndarray],
    teacher_output: np.ndarray,
    block_output_name: str,
    *,
    num_iterations: int,
    learning_rate: float,
    lr_decay: bool,
    batch_size: Optional[int],
    shuffle: bool,
    batch_seed: int,
    step_providers: Optional[Sequence[backend.Provider]],
    losses: Optional[List[float]],
) -> onnx.ModelProto:
    """Runs the LoRA training loop for one already-sliced, already-captured
    block and returns ``model`` with the adapter's ``A``/``B`` initializers
    rewritten. Mirrors :func:`onnxsim.qat._train_block`'s own structure.

    ``extra_initializers`` are :func:`_fold_frozen_prefixes`'s output, if
    any -- used only to build the step graph (whose own constants a folded
    quantization dequant chain becomes, since the base weight never changes
    across steps anyway) and never merged into the *returned* model, which
    is built from ``model`` unchanged: the deployed model keeps its real
    dequant chain, quantized weights included, exactly as
    :func:`apply_qlora` produced it.
    """
    batch = qat._plan_minibatch(external_values, teacher_output, batch_size)
    if batch is None:
        block_inputs, block_target = external_values, teacher_output
    else:
        block_inputs = {k: v[: batch.size] for k, v in external_values.items()}
        block_target = teacher_output[: batch.size]

    shape_source = model
    if extra_initializers:
        shape_source = onnx.ModelProto()
        shape_source.CopyFrom(model)
        shape_source.graph.initializer.extend(extra_initializers)

    shapes = qat._block_shapes(
        shape_source, nodes, block_inputs, block_output_name, block_target
    )

    param_names = set(adapter.parameter_names())
    used = {name for node in nodes for name in node.input if name}
    initializer_map = {t.name: t for t in shape_source.graph.initializer}
    block_initializers = [
        t
        for t in shape_source.graph.initializer
        if t.name in used and t.name not in param_names
    ]

    step = _build_lora_step_graph(
        adapter,
        nodes,
        shapes,
        block_initializers,
        external_values,
        block_output_name,
        list(teacher_output.shape),
        batch,
    )

    if batch is None:
        constants: Dict[str, np.ndarray] = dict(external_values)
        constants[f"{_PREFIX}teacher"] = teacher_output
    else:
        constants = {f"{_PREFIX}all_{k}": v for k, v in external_values.items()}
        constants[f"{_PREFIX}teacher_all"] = teacher_output

    state: Dict[str, np.ndarray] = {}
    for param_name in param_names:
        value = onnx.numpy_helper.to_array(initializer_map[param_name]).astype(
            np.float32
        )
        state[param_name] = value
        state[f"{_PREFIX}m_{param_name}"] = np.zeros_like(value)
        state[f"{_PREFIX}v_{param_name}"] = np.zeros_like(value)

    def scalars(t: int) -> Dict[str, float]:
        decay = 1.0 - t / num_iterations if lr_decay else 1.0
        values = {f"{_PREFIX}lr": learning_rate * decay}
        values.update(qat_graph.adam_bias_corrections(t))
        return values

    feeds = None
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

    tuned = onnx.ModelProto()
    tuned.CopyFrom(model)
    for initializer in tuned.graph.initializer:
        if initializer.name in param_names:
            initializer.CopyFrom(
                onnx.numpy_helper.from_array(
                    final[initializer.name].astype(np.float32), name=initializer.name
                )
            )
    onnx.checker.check_model(tuned)
    return tuned


def train_lora(
    model: Union[str, onnx.ModelProto],
    adapter: LoraAdapter,
    block_input_name: str,
    block_output_name: str,
    reference_model: Optional[Union[str, onnx.ModelProto]] = None,
    target_data: Optional[Sequence[np.ndarray]] = None,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 1000,
    learning_rate: float = 1e-3,
    lr_decay: bool = True,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    batch_seed: int = 0,
    providers: Optional[Sequence[backend.Provider]] = None,
    step_providers: Optional[Sequence[backend.Provider]] = None,
    losses: Optional[List[float]] = None,
) -> onnx.ModelProto:
    """Trains an :func:`inject_lora`-injected adapter's ``A``/``B``
    matrices, with the base model's own weights (and everything else outside
    the adapter) held fixed.

    The block is named the way :func:`onnxsim.apply_qat` names one, by its
    input and output tensor -- and, exactly as there, passing the graph's own
    input/output names trains the whole graph rather than a sub-block.

    Exactly one of ``reference_model``/``target_data`` is required:

    - ``reference_model``: label-free distillation. The block's own output
      is trained to match this model's activation at ``block_output_name``
      on the same calibration inputs -- see this module's docstring for why
      that alone cannot teach a new task, only reproduce the reference.
    - ``target_data``: the loss target directly, one array per
      ``calibration_data`` batch (same axis-0-concatenation contract
      :func:`onnxsim.qat._capture` uses) -- real supervised fine-tuning
      against caller-supplied labels. Requires ``calibration_data`` to be
      given explicitly too (there is nothing to randomly generate labels
      for).

    :param model: the LoRA-injected model (or file path) to train.
    :param adapter: the :class:`LoraAdapter` :func:`inject_lora` returned for
            ``model``.
    :param block_input_name: the activation entering the block.
    :param block_output_name: the block's own final output, whose
            reconstruction error against the target is the loss.
    :param reference_model: the teacher model (or file path); mutually
            exclusive with ``target_data``.
    :param target_data: caller-supplied loss targets; mutually exclusive
            with ``reference_model``.
    :param calibration_data: the block's input data. Random data is
            generated when omitted and ``reference_model`` is used;
            required when ``target_data`` is given.
    :returns: ``model`` with the adapter's initializers trained. Every other
            byte, base weights included, is untouched.
    :raises ValueError: if neither or both of ``reference_model``/
            ``target_data`` are given, if ``target_data`` is given without
            ``calibration_data``, or if ``adapter`` has no targets.
    :raises onnxsim.graph_grad.UnsupportedOpError: if any node in the block
            has no gradient rule.
    """
    if (reference_model is None) == (target_data is None):
        raise ValueError(
            "train_lora needs exactly one of reference_model or target_data"
        )
    if not adapter.targets:
        raise ValueError("adapter has no injected targets to train")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    nodes, externals = qat._slice_block(
        model.graph, block_input_name, block_output_name
    )
    nodes, extra_initializers = _fold_frozen_prefixes(
        nodes, model, adapter.parameter_names()
    )
    qat._refuse_unsupported(nodes)

    if target_data is not None:
        if calibration_data is None:
            raise ValueError(
                "target_data requires calibration_data -- the inputs it was "
                "computed for, since there is nothing to randomly generate "
                "labels for"
            )
    elif calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    external_values = qat._capture(
        model, sorted(set(externals)), calibration_data, providers
    )

    if reference_model is not None:
        if isinstance(reference_model, str):
            reference_model = onnx.load(reference_model, load_external_data=False)
        teacher_output = qat._capture(
            reference_model, [block_output_name], calibration_data, providers
        )[block_output_name]
    else:
        assert target_data is not None  # guaranteed by the XOR check above
        teacher_output = np.concatenate(
            [np.asarray(t, dtype=np.float32) for t in target_data], axis=0
        )

    return _train_lora_block(
        model,
        adapter,
        nodes,
        extra_initializers,
        external_values,
        teacher_output,
        block_output_name,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        batch_size=batch_size,
        shuffle=shuffle,
        batch_seed=batch_seed,
        step_providers=step_providers,
        losses=losses,
    )


def apply_qlora(
    model: Union[str, onnx.ModelProto],
    rank: int = 8,
    alpha: Optional[float] = None,
    target_op_types: Sequence[str] = _ELIGIBLE_OP_TYPES,
    target_names: Optional[Sequence[str]] = None,
    block_size: int = 64,
    seed: int = 0,
) -> Tuple[onnx.ModelProto, LoraAdapter]:
    """:func:`inject_lora`, then :func:`onnxsim.nf4.quantize_weight_only_nf4`
    on everything except the freshly-injected adapters -- QLoRA: a low-rank
    adapter trained on top of an NF4-quantized (4-bit) frozen base, the same
    composition ``tools/onnx-finetune``'s ``prepare_qlora.py`` builds.

    Order matters both ways: injection must happen first (NF4's matcher
    needs a plain initializer-fed ``MatMul``/``Gemm``, which quantizing
    first would replace with a dequant subgraph before injection ever saw
    it), and the injected ``A``/``B`` names must be excluded from
    quantization (``skip_names``) since they are otherwise indistinguishable
    from any other small 2-D weight by shape alone.

    Training the result works unmodified through :func:`train_lora` --
    :func:`_fold_frozen_prefixes` there handles the dequant chain's ``Cast``
    node, which :data:`onnxsim.graph_grad.SUPPORTED_OPS` has no rule for.

    :param block_size: NF4's per-block scale group size; see
            :func:`onnxsim.nf4.quantize_weight_only_nf4`.
    :returns: ``(model with adapters injected and the base weights
            NF4-quantized, the injected LoraAdapter)``.
    """
    injected, adapter = inject_lora(
        model,
        rank=rank,
        alpha=alpha,
        target_op_types=target_op_types,
        target_names=target_names,
        seed=seed,
    )
    quantized = nf4.quantize_weight_only_nf4(
        injected, block_size=block_size, skip_names=adapter.parameter_names()
    )
    return quantized, adapter


def export_lora_adapter(
    model: onnx.ModelProto,
    adapter: LoraAdapter,
    path: str,
    adapter_version: int = 0,
    model_version: int = 0,
) -> None:
    """Exports a trained adapter's ``A``/``B`` values to ONNX Runtime's own
    native ``.onnx_adapter`` format (``onnxruntime.AdapterFormat``, added in
    ORT 1.20) -- the format ``RunOptions.add_active_adapter``/
    ``onnxruntime.LoraAdapter`` swap in at inference time, the same one
    ``tools/onnx-finetune``'s ``export_onnx_adapter.py`` produces.

    A general ORT >=1.20 *inference-side* feature, not training-build
    -specific: the plain ``pip install onnxruntime`` package has it, unlike
    everything ``tools/onnx-finetune`` itself needs.

    ``model`` here is not declared with ``lora_A``/``lora_B`` as graph
    *inputs* the way ``tools/onnx-finetune``'s ``--adapter-inputs`` mode
    does (a separate, live-adapter-swap feature this module does not
    build) -- so the exported file round-trips through
    :meth:`onnxruntime.AdapterFormat.read_adapter`/``get_parameters`` with
    the trained values, but is not itself something
    ``RunOptions.add_active_adapter`` can swap into *this* model at
    inference time.

    :param model: the trained model :func:`train_lora` returned.
    :param adapter: the same :class:`LoraAdapter` used to train it.
    :param path: where to write the ``.onnx_adapter`` file.
    :param adapter_version: stored in the file; ORT surfaces it back on load.
    :param model_version: stored in the file; ORT surfaces it back on load.
    :raises ImportError: if onnxruntime is not installed, or is installed
            without ``AdapterFormat`` support (before 1.20).
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "export_lora_adapter needs the optional 'onnxruntime' package: "
            "pip install onnxruntime"
        ) from e
    if not hasattr(ort, "AdapterFormat"):
        raise ImportError(
            f"export_lora_adapter needs an onnxruntime build with AdapterFormat "
            f"export support (onnxruntime >= 1.20); found onnxruntime "
            f"{ort.__version__}, which does not have it. pip install -U onnxruntime"
        )

    initializer_map = {t.name: t for t in model.graph.initializer}
    params = {
        name: ort.OrtValue.ortvalue_from_numpy(
            onnx.numpy_helper.to_array(initializer_map[name])
        )
        for name in adapter.parameter_names()
    }
    fmt = ort.AdapterFormat()
    fmt.set_parameters(params)
    fmt.set_adapter_version(adapter_version)
    fmt.set_model_version(model_version)
    fmt.export_adapter(path)
