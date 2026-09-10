"""BatchNorm Recovery -- putting a trainable ``BatchNormalization`` back into
a graph that :func:`onnxsim.simplify` (or any equivalent BN-into-Conv fold)
already folded away, so the model can be fine-tuned again with the same
degrees of freedom a not-yet-fused checkpoint would have.

**Why the *original* BatchNorm can't be recovered exactly.**
``onnxsim/passes/fuse_bn_into_conv.h`` folds

::

    s  = scale / sqrt(var + eps)
    W' = W * s          (per output channel)
    b' = (b_conv - mean) * s + bias

into the preceding Conv/ConvTranspose's own weight and bias. Four per-channel
numbers (``scale``, ``bias``, ``mean``, ``var``) collapse into the two that
survive in ``W'``/``b'`` -- the fold is a many-to-one map, so a channel's
worth of information is genuinely gone once it happens. There is no algebraic
inverse: given only the fused ``W'``/``b'``, infinitely many
``(scale, bias, mean, var)`` tuples would have folded to the same result, so
"recover the BatchNorm" cannot mean "reconstruct the exact tuple that was
folded away". Nothing in this module claims to.

**What this module does instead.** :func:`insert_identity_bn` splices a
*fresh* ``BatchNormalization`` node right after each candidate Conv/
ConvTranspose, initialized to the identity transform
(``scale=1, bias=0, mean=0, var=1``, so ``(x - 0) / sqrt(1 + eps) * 1 + 0``
is numerically indistinguishable from ``x`` up to the ``eps`` rounding term
-- see ``test_insert_identity_bn_is_numerically_a_no_op``). That restores
the same graph *shape* a not-yet-fused checkpoint had -- a trainable
per-channel affine knob sitting right where the original BatchNorm used to
be -- without moving the model's output at all at the moment of insertion.
From there, two independent ways to make the inserted node non-trivial:

1. **Gradient descent.** ``onnxsim/graph_grad.py`` already differentiates
   ``BatchNormalization`` (``_grad_batch_normalization_templated``), so the
   inserted node's ``scale``/``bias`` (and, if desired, ``mean``/``var``) are
   immediately trainable by :func:`onnxsim.graph_grad.build_backward` or
   anything built on it, and by any external framework that reads the graph
   back in (a re-exported PyTorch module, a WebGPU/NPU training loop, ...).
   This module hands that machinery the tensors to train
   (:class:`RecoveredBatchNorm`'s ``scale_name``/``bias_name``/...); it does
   not itself run a training loop.
2. **Closed-form calibration** (:func:`calibrate_recovered_bn`), for anyone
   who wants a non-trivial BatchNorm without setting up a training loop at
   all: run real data through the model, measure the real per-channel
   mean/variance flowing into each recovered node, and set ``mean``/``var``
   to match -- exactly what a BatchNorm layer's own running statistics would
   have converged to had it been present and run in inference mode on that
   data. This is the same closed-form-from-calibration-data idiom
   ``onnxsim/bias_correction.py`` and ``onnxsim/norm_tweaking.py`` already
   use, adapted to fit statistics instead of an additive/affine correction.
   Optionally (``target_model=``), the node's ``scale``/``bias`` are further
   set so the recovered BatchNorm's own *output* distribution matches a
   reference model's activation at that same tensor name -- the same
   first-two-moments affine match ``apply_norm_tweaking`` performs for
   ``LayerNormalization``, simplified by the fact that the recovered node's
   own input is already whitened by construction (mean 0, unit variance) at
   that point, so matching a target's mean/std is just
   ``scale, bias = target_std, target_mean``.

:func:`recover_batch_norm` is both steps together -- the one function most
callers actually want. :func:`find_recoverable_convs` is the candidate
search :func:`insert_identity_bn` runs by default (exposed separately so a
caller can inspect or filter it before committing). :func:`find_recovered_bn_nodes`
re-discovers a model's own already-recovered nodes from the graph alone (by
the ``"_bn_recovery"`` marker in the inserted node's own name -- the same
name-suffix-as-marker idiom ``bias_correction.py``'s own inserted ``Add``
nodes use, just not currently re-discovered there), so a later calibration
call needs nothing passed forward from the call that inserted the nodes --
including across a save/reload round trip, since the marker is the node's
own name and survives serialization.

**Scope.** Only ``Conv``/``ConvTranspose`` nodes with a static (initializer)
weight are candidates -- the channel count the recovered BatchNorm needs
comes from that weight's own shape (output-channel axis 0 for ``Conv``, axis
1 times the node's own ``group`` attribute for ``ConvTranspose``, matching
the ONNX spec's weight layout for each). A Conv/ConvTranspose already
followed by a real ``BatchNormalization`` is skipped by
:func:`find_recoverable_convs` -- there is nothing to recover there. A
Conv/ConvTranspose's output feeding more than one consumer is not a problem
(unlike ``fuse_bn_into_conv``'s own single-consumer requirement): this
module only ever *adds* a node, and every existing consumer keeps the exact
same tensor name and therefore the exact same (identity-transformed, so
unchanged) value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs, _all_names, _unique_name
from onnxsim.calibration import Tensors, generate_random_calibration_data

_CONV_OPS = ("Conv", "ConvTranspose")


@dataclass
class RecoveredBatchNorm:
    """One recovered ``BatchNormalization`` node's identity: enough for a
    caller (:func:`calibrate_recovered_bn`, ``graph_grad``, an external
    trainer) to find and drive it without re-scanning the graph itself.

    :ivar conv_output_name: the tensor name the recovered node outputs --
            the same name the Conv/ConvTranspose it follows originally
            output, unchanged by recovery, so every pre-existing consumer
            (another node, a graph output, a ``value_info`` entry) still
            resolves correctly with no rewiring of its own.
    :ivar input_name: the tensor name the recovered node reads -- the
            Conv/ConvTranspose's own output, renamed to make room for the
            node above.
    :ivar node_name: the recovered ``BatchNormalization`` node's own name.
    :ivar scale_name: :ivar bias_name: :ivar mean_name: :ivar var_name:
            the four per-channel initializers the recovered node owns --
            trainable, or the four tensors :func:`calibrate_recovered_bn`
            overwrites in place.
    :ivar channels: the per-channel tensors' own length.
    """

    conv_output_name: str
    input_name: str
    node_name: str
    scale_name: str
    bias_name: str
    mean_name: str
    var_name: str
    channels: int


def _int_attr(node: onnx.NodeProto, name: str, default: int) -> int:
    for a in node.attribute:
        if a.name == name:
            return int(a.i)
    return default


def _conv_output_channels(
    node: onnx.NodeProto, weight: onnx.TensorProto
) -> Optional[int]:
    """The node's own total output-channel count, from its static weight's
    own shape -- ``Conv``'s weight is ``(out_channels, in_channels/group,
    ...)``, so axis 0 is already the total; ``ConvTranspose``'s is
    ``(in_channels, out_channels/group, ...)``, so axis 1 needs multiplying
    by the node's own ``group`` attribute (default 1) to reach the total.
    ``None`` if the weight has too few dims to have an output-channel axis
    at all.
    """
    dims = list(weight.dims)
    if node.op_type == "Conv":
        return int(dims[0]) if len(dims) >= 2 else None
    if node.op_type == "ConvTranspose":
        if len(dims) < 2:
            return None
        return int(dims[1]) * _int_attr(node, "group", 1)
    return None


def find_recoverable_convs(model: onnx.ModelProto) -> List[str]:
    """The output tensor names of every ``Conv``/``ConvTranspose`` node in
    ``model`` that :func:`insert_identity_bn` can splice a recovered
    BatchNorm after: a static (initializer) float weight, so the recovered
    node's own channel count is known, and not already followed by a real
    ``BatchNormalization`` (nothing to recover there).
    """
    initializers = {t.name: t for t in model.graph.initializer}
    has_bn_consumer: Set[str] = set()
    for n in model.graph.node:
        if n.op_type == "BatchNormalization" and n.input:
            has_bn_consumer.add(n.input[0])

    result: List[str] = []
    for n in model.graph.node:
        if n.op_type not in _CONV_OPS or not n.output:
            continue
        out = n.output[0]
        if out in has_bn_consumer or len(n.input) < 2:
            continue
        weight = initializers.get(n.input[1])
        if weight is None or weight.data_type != onnx.TensorProto.FLOAT:
            continue
        if _conv_output_channels(n, weight) is None:
            continue
        result.append(out)
    return result


def find_recovered_bn_nodes(model: onnx.ModelProto) -> List[RecoveredBatchNorm]:
    """Re-discovers every node :func:`insert_identity_bn` previously spliced
    into ``model`` -- by the ``"_bn_recovery"`` marker in the node's own
    name (see this module's own docstring) -- without needing the list
    ``insert_identity_bn`` itself returned. Survives a save/reload round
    trip: the marker is the node's own name, which ``onnx.save``/
    ``onnx.load`` round-trip unchanged.
    """
    initializers = {t.name: t for t in model.graph.initializer}
    recovered: List[RecoveredBatchNorm] = []
    for n in model.graph.node:
        if n.op_type != "BatchNormalization" or "_bn_recovery" not in n.name:
            continue
        if len(n.input) < 5 or not n.output:
            continue
        scale_init = initializers.get(n.input[1])
        channels = (
            int(scale_init.dims[0]) if scale_init is not None and scale_init.dims else 0
        )
        recovered.append(
            RecoveredBatchNorm(
                conv_output_name=n.output[0],
                input_name=n.input[0],
                node_name=n.name,
                scale_name=n.input[1],
                bias_name=n.input[2],
                mean_name=n.input[3],
                var_name=n.input[4],
                channels=channels,
            )
        )
    return recovered


def _insert_bn_after(
    graph: onnx.GraphProto,
    output_name: str,
    channels: int,
    epsilon: float,
    taken_names: Set[str],
) -> str:
    """Splices an identity-initialized ``BatchNormalization`` node right
    after the node currently producing ``output_name``, taking over that
    name for its own output -- the same rename-the-producer's-output,
    insert-the-new-node-under-the-old-name technique
    ``bias_correction.py``'s own ``_apply_correction`` uses for its inserted
    ``Add`` node, so every existing consumer of ``output_name`` (another
    node, a graph output, a ``value_info`` entry) needs no rewiring of its
    own. Returns the inserted node's own name.
    """
    producer_idx = None
    output_index = None
    for idx, n in enumerate(graph.node):
        for oi, out in enumerate(n.output):
            if out == output_name:
                producer_idx, output_index = idx, oi
                break
        if producer_idx is not None:
            break
    if producer_idx is None:
        raise ValueError(f"{output_name!r} is not produced by any node in this graph")

    pre_name = _unique_name(f"{output_name}_pre_bn_recovery", taken_names)
    graph.node[producer_idx].output[output_index] = pre_name

    scale_name = _unique_name(f"{output_name}_bn_recovery_scale", taken_names)
    bias_name = _unique_name(f"{output_name}_bn_recovery_bias", taken_names)
    mean_name = _unique_name(f"{output_name}_bn_recovery_mean", taken_names)
    var_name = _unique_name(f"{output_name}_bn_recovery_var", taken_names)
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.ones(channels, dtype=np.float32), name=scale_name
        )
    )
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.zeros(channels, dtype=np.float32), name=bias_name
        )
    )
    graph.initializer.append(
        onnx.numpy_helper.from_array(
            np.zeros(channels, dtype=np.float32), name=mean_name
        )
    )
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.ones(channels, dtype=np.float32), name=var_name)
    )

    node_name = _unique_name(f"{output_name}_bn_recovery", taken_names)
    bn_node = onnx.helper.make_node(
        "BatchNormalization",
        [pre_name, scale_name, bias_name, mean_name, var_name],
        [output_name],
        name=node_name,
        epsilon=float(epsilon),
    )
    graph.node.insert(producer_idx + 1, bn_node)
    return node_name


def insert_identity_bn(
    model: Union[str, onnx.ModelProto],
    conv_output_names: Optional[Sequence[str]] = None,
    epsilon: float = 1e-5,
) -> Tuple[onnx.ModelProto, List[RecoveredBatchNorm]]:
    """Splices an identity-initialized ``BatchNormalization`` node right
    after every named Conv/ConvTranspose output -- a graph-only rewrite,
    no calibration data needed, and numerically a no-op (up to the
    ``epsilon`` rounding term) at the moment it runs. See this module's own
    docstring for what "recovery" does and does not mean here.

    :param model: the onnx ModelProto or file path to recover BatchNorms
            into.
    :param conv_output_names: which Conv/ConvTranspose node outputs to
            splice a recovered BatchNorm after. ``None`` (the default) means
            every candidate :func:`find_recoverable_convs` finds. Passing an
            output name that is not a Conv/ConvTranspose node's own output
            with a static weight is an error rather than a silent skip --
            unlike the default search, an explicit list is assumed
            deliberate.
    :param epsilon: the recovered node's own ``epsilon`` attribute.
    :returns: ``(recovered_model, recovered)`` -- the rewritten model, and
            one :class:`RecoveredBatchNorm` per inserted node, in the same
            order as ``conv_output_names`` (or :func:`find_recoverable_convs`'s
            own order, when omitted).
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    target_names = (
        list(find_recoverable_convs(model))
        if conv_output_names is None
        else list(dict.fromkeys(conv_output_names))  # de-duplicate, preserve order
    )
    if not target_names:
        return model, []

    node_by_output = {n.output[0]: n for n in model.graph.node if n.output}
    initializers = {t.name: t for t in model.graph.initializer}
    channels_by_name: Dict[str, int] = {}
    for name in target_names:
        node = node_by_output.get(name)
        if node is None or node.op_type not in _CONV_OPS:
            raise ValueError(
                f"{name!r} is not a Conv/ConvTranspose node's own output in this model"
            )
        if len(node.input) < 2:
            raise ValueError(f"{name!r}'s node has no weight input")
        weight = initializers.get(node.input[1])
        if weight is None:
            raise ValueError(
                f"{name!r}'s weight ({node.input[1]!r}) is not a graph "
                "initializer -- its output channel count can't be determined"
            )
        channels = _conv_output_channels(node, weight)
        if channels is None:
            raise ValueError(f"can't determine {name!r}'s own output channel count")
        channels_by_name[name] = channels

    recovered_model = onnx.ModelProto()
    recovered_model.CopyFrom(model)
    graph = recovered_model.graph
    taken_names = _all_names(graph)

    inserted_node_names = [
        _insert_bn_after(graph, name, channels_by_name[name], epsilon, taken_names)
        for name in target_names
    ]

    by_node_name = {r.node_name: r for r in find_recovered_bn_nodes(recovered_model)}
    recovered = [by_node_name[n] for n in inserted_node_names]
    return recovered_model, recovered


def _channel_rows(x: np.ndarray) -> Optional[np.ndarray]:
    """Flattens ``x`` (``[N, C, ...]``, BatchNormalization's own layout) to
    ``[rows, C]``, the shape every per-channel statistic below is measured
    over -- ``None`` if ``x`` has no channel axis at all (rank < 2).
    """
    if x.ndim < 2:
        return None
    channels = x.shape[1]
    return np.moveaxis(x, 1, -1).reshape(-1, channels)


def calibrate_recovered_bn(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    target_model: Optional[Union[str, onnx.ModelProto]] = None,
    node_names: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Fits every recovered BatchNorm node already present in ``model``
    (found by :func:`find_recovered_bn_nodes` -- need not have been
    inserted by this same process, or even this same run) to real
    per-channel activation statistics measured on ``calibration_data``, in
    closed form. See this module's own docstring for the derivation.

    :param model: the onnx ModelProto or file path carrying the recovered
            BatchNorm node(s) to calibrate -- e.g. :func:`insert_identity_bn`'s
            own ``recovered_model``.
    :param calibration_data: representative input batches to measure
            statistics on. Each batch is a ``{input_name: np.ndarray}`` dict
            matching ``model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted).
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted.
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied).
    :param providers: onnxruntime execution providers to run ``model`` (and
            ``target_model``, if given) on.
    :param target_model: an optional reference model to additionally match
            each recovered node's own *output* distribution to -- see this
            module's own docstring's "closed-form calibration" section. Its
            activation is read from the tensor named
            ``RecoveredBatchNorm.conv_output_name`` (the same name the
            Conv/ConvTranspose that precedes the recovered node in ``model``
            originally output), so this only has an effect for a recovered
            node whose ``target_model`` counterpart still produces that
            exact tensor name -- true whenever ``target_model`` and
            ``model`` share the same Conv/ConvTranspose layer graph (e.g.
            ``target_model`` is the un-fused original the fused ``model``
            was produced from, or another checkpoint of the same
            architecture). ``None`` (the default) leaves ``scale``/``bias``
            at their identity values (``1``/``0``): the recovered node then
            becomes a pure whitening normalization -- mean 0, unit variance
            -- over ``calibration_data``'s own distribution, with no target
            distribution to match.
    :param node_names: restricts calibration to these recovered node names
            (:class:`RecoveredBatchNorm.node_name`) -- ``None`` (the
            default) calibrates every recovered node :func:`find_recovered_bn_nodes`
            finds.
    :returns: ``model`` with every calibrated node's own ``mean``/``var``
            (and, with ``target_model``, ``scale``/``bias``) initializers
            overwritten in place -- the same tensors, same names, new
            values, so this is safe (and idempotent) to call again on its
            own output, e.g. to recalibrate against new data.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if isinstance(target_model, str):
        target_model = onnx.load(target_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    recovered = find_recovered_bn_nodes(model)
    if node_names is not None:
        wanted = set(node_names)
        recovered = [r for r in recovered if r.node_name in wanted]
    if not recovered:
        return model

    input_probe = _add_probe_outputs(model, [r.input_name for r in recovered])
    target_probe = (
        _add_probe_outputs(target_model, [r.conv_output_name for r in recovered])
        if target_model is not None
        else None
    )

    x_sum: Dict[str, np.ndarray] = {}
    x_sumsq: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    t_sum: Dict[str, np.ndarray] = {}
    t_sumsq: Dict[str, np.ndarray] = {}
    t_counts: Dict[str, int] = {}

    for batch in calibration_data:
        x_out = backend.run_model(input_probe, batch, providers=providers)
        t_out = (
            backend.run_model(target_probe, batch, providers=providers)
            if target_probe is not None
            else None
        )
        for r in recovered:
            x2 = _channel_rows(np.asarray(x_out[r.input_name], dtype=np.float64))
            if x2 is None:
                continue
            if r.input_name in counts:
                x_sum[r.input_name] += x2.sum(axis=0)
                x_sumsq[r.input_name] += np.square(x2).sum(axis=0)
                counts[r.input_name] += x2.shape[0]
            else:
                x_sum[r.input_name] = x2.sum(axis=0)
                x_sumsq[r.input_name] = np.square(x2).sum(axis=0)
                counts[r.input_name] = x2.shape[0]

            if t_out is None:
                continue
            t = np.asarray(t_out[r.conv_output_name], dtype=np.float64)
            t2 = _channel_rows(t)
            if t2 is None or t2.shape[1] != x2.shape[1]:
                continue
            if r.conv_output_name in t_counts:
                t_sum[r.conv_output_name] += t2.sum(axis=0)
                t_sumsq[r.conv_output_name] += np.square(t2).sum(axis=0)
                t_counts[r.conv_output_name] += t2.shape[0]
            else:
                t_sum[r.conv_output_name] = t2.sum(axis=0)
                t_sumsq[r.conv_output_name] = np.square(t2).sum(axis=0)
                t_counts[r.conv_output_name] = t2.shape[0]

    calibrated = onnx.ModelProto()
    calibrated.CopyFrom(model)
    initializer_index = {t.name: i for i, t in enumerate(calibrated.graph.initializer)}

    def _overwrite(name: str, arr: np.ndarray) -> None:
        idx = initializer_index.get(name)
        if idx is None:
            return
        calibrated.graph.initializer[idx].CopyFrom(
            onnx.numpy_helper.from_array(arr.astype(np.float32), name=name)
        )

    for r in recovered:
        if r.input_name not in counts:
            continue
        n = counts[r.input_name]
        mean = x_sum[r.input_name] / n
        var = np.maximum(x_sumsq[r.input_name] / n - mean**2, 0.0)
        _overwrite(r.mean_name, mean)
        _overwrite(r.var_name, var)

        if r.conv_output_name in t_counts:
            tn = t_counts[r.conv_output_name]
            mean_t = t_sum[r.conv_output_name] / tn
            var_t = np.maximum(t_sumsq[r.conv_output_name] / tn - mean_t**2, 0.0)
            _overwrite(r.scale_name, np.sqrt(var_t))
            _overwrite(r.bias_name, mean_t)

    return calibrated


def recover_batch_norm(
    model: Union[str, onnx.ModelProto],
    conv_output_names: Optional[Sequence[str]] = None,
    epsilon: float = 1e-5,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    target_model: Optional[Union[str, onnx.ModelProto]] = None,
) -> onnx.ModelProto:
    """:func:`insert_identity_bn` followed by :func:`calibrate_recovered_bn`
    -- the one call most callers actually want: a model with a real,
    data-fit BatchNormalization back in place of what
    ``fuse_bn_into_conv`` (or an equivalent fold) had folded away, ready to
    ship as-is or to fine-tune further (e.g. by gradient descent through
    ``onnxsim.graph_grad``, using ``RecoveredBatchNorm.scale_name``/
    ``bias_name`` as the tensors to train). See this module's own docstring;
    see :func:`insert_identity_bn` and :func:`calibrate_recovered_bn` for
    each individual parameter's own meaning -- they're forwarded unchanged.

    :returns: the recovered (and calibrated) model. A model with no
            recoverable Conv/ConvTranspose at all (see
            :func:`find_recoverable_convs`) is returned unchanged, with no
            calibration data ever generated or run.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    recovered_model, recovered = insert_identity_bn(
        model, conv_output_names=conv_output_names, epsilon=epsilon
    )
    if not recovered:
        return recovered_model
    return calibrate_recovered_bn(
        recovered_model,
        calibration_data=calibration_data,
        num_samples=num_samples,
        seed=seed,
        providers=providers,
        target_model=target_model,
    )
