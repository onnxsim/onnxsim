"""QAT interop: consume and produce fake-quant (QDQ) graphs without training.

This is "deliverable A" of ``docs/qat.md`` -- the half of quantization-aware
training that is pure graph rewriting. Nothing here runs a training loop, and
nothing here needs calibration data for a tensor whose quantization
parameters the model already carries.

Two directions, and they are inverses of each other:

**Ingest** (:func:`quantize_static_keeping_qdq_scales`). A model exported from
PyTorch/TensorFlow QAT arrives as a QDQ graph whose scales and zero-points
were *learned* -- they are the output of hours of training, and they are
generally **not** what observing the tensor's min/max would produce (a trained
scale usually clips: it trades a few saturated outliers for a finer step on
the bulk of the distribution). :func:`onnxsim.quantize_static` would re-derive
them from calibration and throw the learned values away, silently, with no
error and only a small accuracy loss to show for it. So this module detects
the QDQ pairs already present, canonicalizes them back to the float graph the
rest of onnxsim's pipeline expects, re-runs the ordinary static-quantization
rewrite, and then writes the learned parameters back over the ones the rewrite
just computed. A tensor with no existing QDQ pair is calibrated normally, so a
*partially* quantized export is handled rather than refused -- and when every
quantizable tensor is annotated, no calibration data is needed or asked for at
all.

**Egress** (:func:`export_fake_quant`). Emit a QDQ model from a float model
plus a chosen scheme, naming the initializers an external trainer should make
learnable. Together with ingest this closes a round trip: onnxsim picks the
scheme, the user trains the scales in their framework, onnxsim re-imports
them.

Why write the learned values back *after* the rewrite instead of feeding them
in as a calibration range
---------------------------------------------------------------------------
``QuantizeStatic`` takes a ``{tensor: (min, max)}`` range and derives
``(scale, zero_point)`` from it in C++ (``ComputeAsymmetricUint8QuantParams``
in ``passes/static_quantize_matmul.h``). That map is invertible -- a learned
``(scale, zp)`` corresponds to the range ``(-zp*scale, (levels-zp)*scale)`` --
so the ranges *could* carry the learned values in. They are passed in, because
the pass will not fire on a tensor that has no range at all. But the round trip
goes through two float32 divisions and a ``round``, so it is only exact to
within an ulp or two, and "the learned value survives" is the entire point of
this path. Writing the exact learned tensor back over the emitted initializer
afterwards makes the guarantee bit-exact rather than nearly so.

What this module refuses
------------------------
Conservative in the house style: a QDQ pair it does not fully understand is
left exactly where it is rather than guessed at, and the caller is told which
one and why (:class:`SkippedQdq`, reachable from every entry point's result).
See :data:`SKIP_REASONS` for the list. In particular a pair whose scale is not
a constant initializer, whose zero-point has an unexpected dtype, or whose
per-axis scale length disagrees with the tensor's own dimension along the axis
it claims, is never canonicalized -- so it survives into the output untouched
instead of being re-derived from a guess.

Not covered, deliberately
-------------------------
- **Blocked quantization** (opset 21's ``block_size`` attribute) is refused
  rather than canonicalized; onnxsim's static-quantization rewrite has no
  blocked activation scheme to re-emit it into.
- **QDQ inside subgraphs** (``If``/``Loop`` bodies) is not detected, and a
  top-level pair whose dequantized output is referenced from a subgraph is
  refused.
- **Per-axis activation scales** are detected and reported but cannot be
  preserved: ``quantize_static``'s activation scheme is per-tensor. The tensor
  falls back to calibration and says so.
- **Asymmetric or non-int8 learned weight quantizers** are understood but not
  re-emitted: onnxsim's weight scheme is symmetric int8 per output channel, so
  such a weight is reported and falls back to onnxsim's own round-to-nearest
  scale. A symmetric int8 one *is* preserved, codes and all.
- **Per-layer bit-width selection** (``mixed_precision.py``,
  ``precision_estimator.py``) is not wired into :func:`export_fake_quant`; it
  emits one scheme for the whole model.

And one behaviour worth knowing rather than a gap: ingest runs the *whole*
static-quantization rewrite, so the model it returns may quantize more nodes
than the input model did -- an export that fake-quantized only some of its
layers comes back with the rest quantized too, from calibration.
``QdqIngestResult.recalibrated`` lists exactly those.
"""

import dataclasses
import json
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper

import onnxsim.onnxsim_cpp2py_export as C
from onnxsim.calibration import Tensors, calibrate, generate_random_calibration_data
from onnxsim.model_info import METADATA_PREFIX

__all__ = [
    "QdqAnnotation",
    "SkippedQdq",
    "QdqScan",
    "QdqIngestResult",
    "FakeQuantExport",
    "SKIP_REASONS",
    "LEARNABLE_SCALES_KEY",
    "LEARNABLE_ZERO_POINTS_KEY",
    "find_existing_qdq",
    "strip_existing_qdq",
    "quantize_static_keeping_qdq_scales",
    "export_fake_quant",
]


# Every reason string this module ever reports, with what it means. Kept as a
# single documented table (rather than only as string literals at the raise
# sites) so a caller can branch on a reason without reading the source, and so
# adding a refusal without documenting it is visibly incomplete.
SKIP_REASONS: Dict[str, str] = {
    # -- detection: the pattern is not a canonicalizable QDQ pair --
    "dequantize_input_not_recognized": (
        "DequantizeLinear's input is neither a QuantizeLinear output nor a "
        "constant integer initializer, so there is no float tensor to "
        "canonicalize back to"
    ),
    "mismatched_quantization_params": (
        "the QuantizeLinear and DequantizeLinear of the pair do not name the "
        "same scale/zero-point tensors (or the same axis), so the pair is not "
        "a round trip through one quantizer"
    ),
    "quantized_tensor_reused": (
        "the integer tensor between Quantize and Dequantize has another "
        "consumer (or is a graph output), so removing the pair would change "
        "what the graph computes"
    ),
    "dequantize_output_is_graph_output": (
        "the dequantized tensor is a graph output; rewiring its consumers "
        "would drop a name the model promises to produce"
    ),
    "referenced_in_subgraph": (
        "the dequantized tensor is read from inside an If/Loop body, which "
        "this module does not rewrite"
    ),
    # -- detection: the parameters themselves are not usable --
    "scale_not_constant": "the scale input is not a constant initializer",
    "zero_point_not_constant": ("the zero-point input is not a constant initializer"),
    "unsupported_scale_dtype": "the scale initializer is not float32",
    "unsupported_zero_point_dtype": (
        "the zero-point initializer's dtype is not one of uint8/int8/uint16/int16"
    ),
    "zero_point_shape_mismatch": ("the zero-point's shape differs from the scale's"),
    "non_positive_scale": "the scale is not finite and strictly positive",
    "blocked_quantization_unsupported": (
        "the pair uses opset-21 blocked quantization (block_size), which has "
        "no counterpart in onnxsim's static-quantization scheme"
    ),
    "axis_shape_unknown": (
        "the scale is per-axis but the quantized tensor's shape along that "
        "axis is not statically known, so the claim cannot be checked"
    ),
    "axis_shape_mismatch": (
        "the per-axis scale's length differs from the quantized tensor's "
        "dimension along the axis it claims"
    ),
    # -- ingest: detected and canonicalized, but not carried into the output --
    "not_a_quantizable_tensor": (
        "onnxsim's static quantization has no place for this tensor (it is "
        "not the activation or weight of a MatMul/Gemm/Conv it quantizes), so "
        "the pair was left in the graph untouched rather than re-emitted"
    ),
    "per_axis_activation_unsupported": (
        "the learned activation scale is per-axis; quantize_static's "
        "activation scheme is per-tensor, so this tensor was recalibrated"
    ),
    "activation_zero_point_dtype_mismatch": (
        "the learned activation zero-point's dtype is not the one this scheme "
        "emits (uint8 for int8, uint16 for int16)"
    ),
    "activation_zero_point_out_of_range": (
        "the learned activation zero-point lies outside the scheme's code "
        "range, so no (min, max) range reproduces it"
    ),
    "weight_dtype_not_int8": (
        "the learned weight quantization does not store int8 codes; onnxsim's "
        "weight scheme is symmetric int8, so it cannot carry it"
    ),
    "weight_zero_point_not_symmetric": (
        "the learned weight quantization is asymmetric (non-zero zero-point); "
        "onnxsim's weight scheme is symmetric int8, so it cannot carry it"
    ),
    "weight_scale_shape_mismatch": (
        "the learned weight scale's length or axis does not match the "
        "per-output-channel scale onnxsim emits for this weight"
    ),
    "quantize_node_not_emitted": (
        "the static-quantization rewrite did not quantize this tensor after "
        "all (e.g. the model's opset is below 13), so there was nothing to "
        "write the learned parameters into"
    ),
    "quantization_params_shared": (
        "the emitted scale/zero-point initializer is shared with another "
        "consumer, so overwriting it would change that consumer too"
    ),
}

# metadata_props keys :func:`export_fake_quant` stamps onto the model. See its
# docstring for why the names are carried *both* here and in the returned
# lists.
LEARNABLE_SCALES_KEY = METADATA_PREFIX + "qat.learnable_scales"
LEARNABLE_ZERO_POINTS_KEY = METADATA_PREFIX + "qat.learnable_zero_points"

_SUPPORTED_ZP_DTYPES = {
    onnx.TensorProto.UINT8,
    onnx.TensorProto.INT8,
    onnx.TensorProto.UINT16,
    onnx.TensorProto.INT16,
}

# (activation zero-point dtype, number of steps between the smallest and
# largest code) for each scheme this module can re-emit -- exactly what
# ComputeAsymmetricUint8QuantParams / ...Uint16QuantParams divide by.
_SCHEMES = {
    "int8": (onnx.TensorProto.UINT8, 255.0),
    "int16": (onnx.TensorProto.UINT16, 65535.0),
}

# The ops whose activation/weight quantize_static rewrites, so ingest can tell
# a QDQ pair it can re-emit from one it must leave alone.
_QUANTIZED_OPS = ("MatMul", "Gemm", "Conv")


@dataclasses.dataclass(frozen=True)
class QdqAnnotation:
    """One QuantizeLinear/DequantizeLinear pair already present in a model,
    read back as the quantization of a single float tensor.

    ``tensor_name`` is the *float* tensor the pair brackets -- the name the
    graph uses once the pair is canonicalized away, which is what every other
    part of onnxsim (calibration ranges,
    ``onnxsim_cpp2py_export.list_quantizable_activations``) addresses tensors
    by. ``scale``/``zero_point`` are the learned values themselves, already
    materialized as numpy arrays: they are the thing worth preserving, and a
    caller that only wants to *inspect* an export's learned parameters can
    stop here without going anywhere near the rest of this module.
    """

    tensor_name: str
    role: str  # "activation" (a runtime value) or "weight" (a constant)
    scale: np.ndarray  # float32, scalar for per-tensor
    zero_point: np.ndarray  # integer, same shape as `scale`
    zero_point_dtype: int  # an onnx.TensorProto enum value
    axis: Optional[int]  # None when the scale is per-tensor
    scale_name: str
    zero_point_name: Optional[str]  # None when the pair omitted the input
    quantize_node: Optional[str]  # output name of the QuantizeLinear, if any
    dequantize_node: str  # output name of the DequantizeLinear

    @property
    def per_tensor(self) -> bool:
        return self.axis is None


@dataclasses.dataclass(frozen=True)
class SkippedQdq:
    """A QDQ pattern this module declined to act on, and why.

    ``reason`` is a key of :data:`SKIP_REASONS`. Skipping is never silent and
    never fatal: the pair stays in the graph exactly as authored, so the model
    keeps whatever quantization it had -- the caller simply learns that
    onnxsim did not take responsibility for it.
    """

    tensor_name: str
    reason: str
    node: str  # output name of the node the refusal was raised on

    def __str__(self) -> str:
        return f"{self.tensor_name}: {self.reason} ({SKIP_REASONS[self.reason]})"


@dataclasses.dataclass(frozen=True)
class QdqScan:
    """What :func:`find_existing_qdq` found: the pairs it understood, and the
    ones it refused.
    """

    annotations: Tuple[QdqAnnotation, ...]
    skipped: Tuple[SkippedQdq, ...]

    def by_name(self) -> Dict[str, QdqAnnotation]:
        return {a.tensor_name: a for a in self.annotations}

    @property
    def activations(self) -> Tuple[QdqAnnotation, ...]:
        return tuple(a for a in self.annotations if a.role == "activation")

    @property
    def weights(self) -> Tuple[QdqAnnotation, ...]:
        return tuple(a for a in self.annotations if a.role == "weight")


@dataclasses.dataclass(frozen=True)
class QdqIngestResult:
    """The output of :func:`quantize_static_keeping_qdq_scales`.

    Four disjoint accounts of every quantizable tensor, so "did my learned
    scale survive?" is answerable without diffing two models:

    - ``preserved`` -- the learned scale/zero-point are in ``model``, bit-exact.
    - ``recalibrated`` -- no usable annotation, so the tensor was calibrated
      the ordinary way.
    - ``unpreserved`` -- an annotation was found and understood, but this
      scheme could not carry it; the tensor fell back to calibration.
    - ``scan.skipped`` -- a QDQ pattern was refused outright and left in the
      graph untouched.
    """

    model: onnx.ModelProto
    scan: QdqScan
    preserved: Tuple[str, ...]
    recalibrated: Tuple[str, ...]
    unpreserved: Tuple[SkippedQdq, ...]
    calibration_ran: bool


@dataclasses.dataclass(frozen=True)
class FakeQuantExport:
    """The output of :func:`export_fake_quant`: a QDQ model plus the names of
    the initializers an external trainer should make learnable.
    """

    model: onnx.ModelProto
    learnable_scales: Tuple[str, ...]
    learnable_zero_points: Tuple[str, ...]

    @property
    def learnable_tensors(self) -> Tuple[str, ...]:
        """Every initializer a QAT trainer may touch, scales first.

        Only ``learnable_scales`` are float tensors a gradient can move
        directly; ``learnable_zero_points`` are integer-typed, so a trainer
        that learns them too (LSQ+ does) has to round back to their dtype at
        every step. They are listed separately for exactly that reason.
        """
        return self.learnable_scales + self.learnable_zero_points


# --------------------------------------------------------------------------- #
# Graph inspection helpers
# --------------------------------------------------------------------------- #
def _load(model: Union[str, onnx.ModelProto]) -> onnx.ModelProto:
    if isinstance(model, str):
        return onnx.load(model, load_external_data=False)
    return model


def _infer(model: onnx.ModelProto) -> onnx.ModelProto:
    """``model`` with value_info filled in, or ``model`` unchanged on failure.

    Both entry points run this before handing a graph to the C++ rewrite,
    because ``ListQuantizableActivations``/``StaticQuantizeMatMul`` skip any
    node whose activation input has no *declared* element type -- an
    intermediate tensor of a graph that was never shape-inferred looks like
    "not float32" to them and is silently left unquantized. A QAT export's
    interesting tensors are exactly those intermediates, so inferring first is
    the difference between preserving a learned scale and reporting that there
    was nowhere to put it. Inference only adds value_info, so a model it fails
    on is simply used as-is.
    """
    try:
        return onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        return model


def _attr(node: onnx.NodeProto, name: str) -> Optional[onnx.AttributeProto]:
    for attr in node.attribute:
        if attr.name == name:
            return attr
    return None


def _iter_subgraphs(graph: onnx.GraphProto) -> Iterator[onnx.GraphProto]:
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                yield attr.g
                yield from _iter_subgraphs(attr.g)
            for sub in attr.graphs:
                yield sub
                yield from _iter_subgraphs(sub)


def _subgraph_reads(graph: onnx.GraphProto) -> Set[str]:
    """Every tensor name read by a node inside any nested subgraph.

    A subgraph can close over a name from the enclosing graph, so a top-level
    QDQ pair whose output is read from an ``If``/``Loop`` body cannot be
    rewired by only touching top-level nodes.
    """
    names: Set[str] = set()
    for sub in _iter_subgraphs(graph):
        for node in sub.node:
            names.update(node.input)
        names.update(o.name for o in sub.output)
    return names


def _use_counts(graph: onnx.GraphProto) -> Dict[str, int]:
    """How many times each tensor name is consumed at the top level, counting a
    graph output as one use.
    """
    counts: Dict[str, int] = {}
    for node in graph.node:
        for name in node.input:
            if name:
                counts[name] = counts.get(name, 0) + 1
    for out in graph.output:
        counts[out.name] = counts.get(out.name, 0) + 1
    return counts


def _producers(graph: onnx.GraphProto) -> Dict[str, onnx.NodeProto]:
    return {out: node for node in graph.node for out in node.output if out}


def _static_shapes(model: onnx.ModelProto) -> Dict[str, Tuple[Optional[int], ...]]:
    """``{tensor_name: shape}`` for every tensor whose shape is statically
    known, with ``None`` for a symbolic dimension.

    Shape inference is best-effort: it is only needed to *check* a per-axis
    scale's claim, and a tensor whose shape stays unknown is refused rather
    than assumed (``axis_shape_unknown``), so a failed inference costs
    coverage, never correctness.
    """
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        inferred = model
    shapes: Dict[str, Tuple[Optional[int], ...]] = {}
    for init in inferred.graph.initializer:
        shapes[init.name] = tuple(init.dims)
    values = (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    )
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shapes.setdefault(
            value.name,
            tuple(
                dim.dim_value if dim.HasField("dim_value") else None
                for dim in tensor_type.shape.dim
            ),
        )
    return shapes


# --------------------------------------------------------------------------- #
# Ingest: detection
# --------------------------------------------------------------------------- #
def _check_params(
    scale_init: Optional[onnx.TensorProto],
    zp_name: Optional[str],
    zp_init: Optional[onnx.TensorProto],
) -> Optional[str]:
    """The dtype/shape/value checks both QDQ shapes share, as a reason or None."""
    if scale_init is None:
        return "scale_not_constant"
    if scale_init.data_type != onnx.TensorProto.FLOAT:
        return "unsupported_scale_dtype"
    scale = onnx.numpy_helper.to_array(scale_init)
    if not np.all(np.isfinite(scale)) or not np.all(scale > 0):
        return "non_positive_scale"
    if zp_name:
        if zp_init is None:
            return "zero_point_not_constant"
        if zp_init.data_type not in _SUPPORTED_ZP_DTYPES:
            return "unsupported_zero_point_dtype"
        if tuple(zp_init.dims) != tuple(scale_init.dims):
            return "zero_point_shape_mismatch"
    return None


def _pair_axis(
    dq: onnx.NodeProto,
    q: Optional[onnx.NodeProto],
    scale: np.ndarray,
    quantized_name: str,
    shapes: Dict[str, Tuple[Optional[int], ...]],
) -> Union[Optional[int], str]:
    """The pair's quantization axis, or a skip reason.

    Returns ``None`` for a per-tensor (scalar or single-element) scale. For a
    per-axis one the claim is *checked* against the tensor's own shape rather
    than trusted: a scale of length C attached to an axis whose extent is not C
    is a malformed export, and guessing which of the two is right is exactly
    what this module does not do.
    """
    if scale.size == 1:
        return None
    axis_attr = _attr(dq, "axis")
    axis = int(axis_attr.i) if axis_attr is not None else 1
    if q is not None:
        q_axis_attr = _attr(q, "axis")
        q_axis = int(q_axis_attr.i) if q_axis_attr is not None else 1
        if q_axis != axis:
            return "mismatched_quantization_params"
    shape = shapes.get(quantized_name)
    if shape is None:
        return "axis_shape_unknown"
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        return "axis_shape_mismatch"
    if shape[axis] is None:
        return "axis_shape_unknown"
    if shape[axis] != scale.size:
        return "axis_shape_mismatch"
    return axis


def _scan_dequantize(
    dq: onnx.NodeProto,
    inits: Dict[str, onnx.TensorProto],
    producers: Dict[str, onnx.NodeProto],
    uses: Dict[str, int],
    outputs: Set[str],
    subgraph_reads: Set[str],
    shapes: Dict[str, Tuple[Optional[int], ...]],
) -> Union[QdqAnnotation, SkippedQdq, None]:
    """One DequantizeLinear node, read as an annotation or a refusal.

    ``None`` means "not a QDQ pattern at all" only in the cases where the node
    is not a usable DequantizeLinear to begin with; every recognized-but-
    unusable shape comes back as a :class:`SkippedQdq` rather than being
    dropped on the floor.
    """
    dq_out = dq.output[0]
    quantized_name = dq.input[0]
    q = producers.get(quantized_name)
    if q is not None and q.op_type != "QuantizeLinear":
        q = None
    source_is_initializer = quantized_name in inits

    if q is None and not source_is_initializer:
        return SkippedQdq(dq_out, "dequantize_input_not_recognized", dq_out)

    # The float tensor the pair brackets, and hence the name everything
    # downstream addresses it by. Shape 3 has no float tensor in the graph at
    # all -- the dequantized output *is* it -- so it takes that name.
    tensor_name = q.input[0] if q is not None else dq_out

    if _attr(dq, "block_size") is not None or (
        q is not None and _attr(q, "block_size") is not None
    ):
        return SkippedQdq(tensor_name, "blocked_quantization_unsupported", dq_out)
    if dq_out in outputs:
        return SkippedQdq(tensor_name, "dequantize_output_is_graph_output", dq_out)
    if dq_out in subgraph_reads or tensor_name in subgraph_reads:
        return SkippedQdq(tensor_name, "referenced_in_subgraph", dq_out)

    scale_name = dq.input[1]
    zp_name = dq.input[2] if len(dq.input) > 2 and dq.input[2] else None
    if q is not None:
        q_zp = q.input[2] if len(q.input) > 2 and q.input[2] else None
        if len(q.input) < 2 or q.input[1] != scale_name or q_zp != zp_name:
            return SkippedQdq(tensor_name, "mismatched_quantization_params", dq_out)
        # Removing the pair removes the integer tensor with it, so anything
        # else reading it would break.
        if uses.get(quantized_name, 0) != 1 or quantized_name in outputs:
            return SkippedQdq(tensor_name, "quantized_tensor_reused", dq_out)

    scale_init = inits.get(scale_name)
    zp_init = inits.get(zp_name) if zp_name else None
    reason = _check_params(scale_init, zp_name, zp_init)
    if reason is not None:
        return SkippedQdq(tensor_name, reason, dq_out)
    assert scale_init is not None  # _check_params rejected None already
    scale = onnx.numpy_helper.to_array(scale_init)

    axis = _pair_axis(dq, q, scale, quantized_name, shapes)
    if isinstance(axis, str):
        return SkippedQdq(tensor_name, axis, dq_out)

    if zp_init is not None:
        zero_point = onnx.numpy_helper.to_array(zp_init)
        zp_dtype = zp_init.data_type
    else:
        # An omitted zero_point is defined to be zero, in the quantized
        # tensor's own dtype -- which for shape 3 we can read off the stored
        # codes, and for shapes 1/2 defaults to uint8.
        zp_dtype = (
            inits[quantized_name].data_type
            if source_is_initializer
            else onnx.TensorProto.UINT8
        )
        if zp_dtype not in _SUPPORTED_ZP_DTYPES:
            return SkippedQdq(tensor_name, "unsupported_zero_point_dtype", dq_out)
        zero_point = np.zeros(scale.shape, dtype=np.int64)

    is_weight = source_is_initializer or tensor_name in inits
    return QdqAnnotation(
        tensor_name=tensor_name,
        role="weight" if is_weight else "activation",
        scale=np.asarray(scale, dtype=np.float32),
        zero_point=np.asarray(zero_point),
        zero_point_dtype=zp_dtype,
        axis=axis,
        scale_name=scale_name,
        zero_point_name=zp_name,
        quantize_node=q.output[0] if q is not None else None,
        dequantize_node=dq_out,
    )


def find_existing_qdq(model: Union[str, onnx.ModelProto]) -> QdqScan:
    """Detect the QuantizeLinear/DequantizeLinear pairs already in ``model``.

    Three shapes are recognized, which between them cover what QAT exporters
    and onnxsim's own :func:`onnxsim.quantize_static` emit:

    1. ``X -> QuantizeLinear -> DequantizeLinear -> ...`` where ``X`` is a
       runtime value -- a *fake-quantized activation*, the learned-scale case
       this whole module exists for.
    2. the same, where ``X`` is a float initializer -- a fake-quantized
       *weight*.
    3. ``Wq (integer initializer) -> DequantizeLinear -> ...`` with no
       Quantize -- a weight already stored in its quantized form, which is
       what ONNX Runtime's quantizer and ``quantize_static`` produce.

    Nothing is modified. The scan is purely informational, which makes it the
    right entry point for "what did my trainer actually learn?" -- and it is
    also the first step of :func:`strip_existing_qdq` and
    :func:`quantize_static_keeping_qdq_scales`, so the three always agree on
    what counts as a pair.

    :param model: onnx ModelProto object or file path
    :returns: a :class:`QdqScan` of understood pairs and refused ones
    """
    model = _load(model)
    graph = model.graph
    inits = {init.name: init for init in graph.initializer}
    producers = _producers(graph)
    uses = _use_counts(graph)
    outputs = {o.name for o in graph.output}
    subgraph_reads = _subgraph_reads(graph)
    shapes = _static_shapes(model)

    annotations: List[QdqAnnotation] = []
    skipped: List[SkippedQdq] = []
    for dq in graph.node:
        if dq.op_type != "DequantizeLinear" or len(dq.input) < 2 or not dq.output:
            continue
        found = _scan_dequantize(
            dq, inits, producers, uses, outputs, subgraph_reads, shapes
        )
        if isinstance(found, QdqAnnotation):
            annotations.append(found)
        elif isinstance(found, SkippedQdq):
            skipped.append(found)
    return QdqScan(tuple(annotations), tuple(skipped))


# --------------------------------------------------------------------------- #
# Ingest: canonicalization (removing the pairs)
# --------------------------------------------------------------------------- #
def _broadcast_along_axis(
    vector: np.ndarray, rank: int, axis: Optional[int]
) -> np.ndarray:
    """``vector`` reshaped so it broadcasts against a rank-``rank`` tensor along
    ``axis`` (or unchanged, for a per-tensor scalar).
    """
    if axis is None:
        return vector.reshape(())
    shape = [1] * rank
    shape[axis] = vector.size
    return vector.reshape(shape)


def _dequantize(codes: np.ndarray, ann: QdqAnnotation) -> np.ndarray:
    zero_point = _broadcast_along_axis(
        np.asarray(ann.zero_point, dtype=np.float64).reshape(-1), codes.ndim, ann.axis
    )
    scale = _broadcast_along_axis(
        np.asarray(ann.scale, dtype=np.float64).reshape(-1), codes.ndim, ann.axis
    )
    return ((codes.astype(np.float64) - zero_point) * scale).astype(np.float32)


def _strip(
    model: onnx.ModelProto, annotations: Sequence[QdqAnnotation]
) -> onnx.ModelProto:
    """``model`` with each named pair removed and its consumers rewired to the
    float tensor the pair bracketed.

    The result is an ordinary float graph, which is the only kind the rest of
    onnxsim's pipeline (``list_quantizable_activations``, ``QuantizeStatic``,
    the ``passes/``) knows how to reason about. Nothing else in the graph is
    touched: pairs not named here stay exactly as authored, and the only
    initializers removed are the ones the removed pairs were the last consumer
    of.
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    inits = {init.name: init for init in graph.initializer}
    producers = _producers(graph)

    drop_nodes: Set[str] = set()
    rewire: Dict[str, str] = {}
    orphan_candidates: Set[str] = set()
    materialized: List[onnx.TensorProto] = []

    for ann in annotations:
        dq = producers[ann.dequantize_node]
        drop_nodes.add(ann.dequantize_node)
        orphan_candidates.add(ann.scale_name)
        if ann.zero_point_name:
            orphan_candidates.add(ann.zero_point_name)
        if ann.quantize_node is not None:
            drop_nodes.add(ann.quantize_node)
            rewire[ann.dequantize_node] = ann.tensor_name
        else:
            # Shape 3: the float tensor exists only implicitly, so materialize
            # it under the dequantized output's own name -- consumers then need
            # no rewiring at all, and the name the rest of the pipeline sees is
            # the one the model already used for this value.
            codes_name = dq.input[0]
            orphan_candidates.add(codes_name)
            materialized.append(
                onnx.numpy_helper.from_array(
                    _dequantize(onnx.numpy_helper.to_array(inits[codes_name]), ann),
                    ann.tensor_name,
                )
            )

    kept = [
        node
        for node in graph.node
        if not node.output or node.output[0] not in drop_nodes
    ]
    del graph.node[:]
    graph.node.extend(kept)
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in rewire:
                node.input[i] = rewire[name]

    graph.initializer.extend(materialized)

    # Only the pairs' own parameter tensors are ever pruned, and only when
    # nothing at all still reads them -- an initializer that was already unused
    # before this call is left alone, so canonicalization never has a side
    # effect the caller did not ask for.
    still_used: Set[str] = set()
    for node in graph.node:
        still_used.update(node.input)
    for sub in _iter_subgraphs(graph):
        for node in sub.node:
            still_used.update(node.input)
    still_used.update(o.name for o in graph.output)
    still_used.update(i.name for i in graph.input)
    orphans = {name for name in orphan_candidates if name not in still_used}
    if orphans:
        surviving = [i for i in graph.initializer if i.name not in orphans]
        del graph.initializer[:]
        graph.initializer.extend(surviving)

    # value_info describing tensors that no longer exist would otherwise
    # outlive them and confuse shape inference.
    gone = drop_nodes | orphans
    surviving_info = [v for v in graph.value_info if v.name not in gone]
    del graph.value_info[:]
    graph.value_info.extend(surviving_info)
    return out


def strip_existing_qdq(
    model: Union[str, onnx.ModelProto],
) -> Tuple[onnx.ModelProto, QdqScan]:
    """Canonicalize ``model`` back to a float graph, returning it and the scan
    that says what was removed.

    "Canonicalize" rather than "dequantize": for a fake-quant pair the float
    tensor is already in the graph and the pair is simply deleted, so the
    result is the *original* float graph, not an approximation of it. Only a
    weight stored in integer form (scan shape 3) is actually rematerialized,
    as ``(codes - zero_point) * scale``.

    Useful on its own -- it is how you get a QAT export into any of onnxsim's
    float-model entry points -- but note that on its own it *discards* the
    learned parameters, which is the mistake this module exists to prevent.
    Use :func:`quantize_static_keeping_qdq_scales` to canonicalize and put them
    back.

    :param model: onnx ModelProto object or file path
    :returns: ``(float_model, scan)``
    """
    model = _load(model)
    scan = find_existing_qdq(model)
    return _strip(model, scan.annotations), scan


# --------------------------------------------------------------------------- #
# Ingest: the entry point
# --------------------------------------------------------------------------- #
def _range_for(scale: float, zero_point: float, levels: float) -> Tuple[float, float]:
    """The calibration range whose ``(scale, zero_point)`` is the given one.

    The exact inverse of ``ComputeAsymmetricUint8QuantParams``: that function
    computes ``scale = (hi - lo) / levels`` and ``zp = round(-lo / scale)``
    after widening ``[lo, hi]`` to include 0, so ``lo = -zp * scale`` and
    ``hi = lo + levels * scale`` reproduce it (and the widening is a no-op,
    since a zero-point inside the code range puts 0 inside the interval by
    construction). Float32 rounding makes the trip approximate, which is why
    the caller writes the learned tensors back afterwards; what this only has
    to get right is making the pass fire and landing in the right ballpark if
    the write-back is refused.
    """
    lo = -zero_point * scale
    return (lo, lo + levels * scale)


def _set_initializer_value(
    graph: onnx.GraphProto, name: str, value: np.ndarray
) -> None:
    """Overwrite initializer ``name``'s payload, keeping its name and dims."""
    for i, init in enumerate(graph.initializer):
        if init.name != name:
            continue
        replacement = onnx.numpy_helper.from_array(value.reshape(init.dims), name)
        graph.initializer[i].CopyFrom(replacement)
        return
    raise KeyError(name)


def _preservable_activation(
    ann: QdqAnnotation, zp_dtype: int, levels: float
) -> Optional[str]:
    """Why this activation annotation cannot be re-emitted, or None."""
    if ann.axis is not None:
        return "per_axis_activation_unsupported"
    if ann.zero_point_dtype != zp_dtype:
        return "activation_zero_point_dtype_mismatch"
    zero_point = float(np.asarray(ann.zero_point).reshape(-1)[0])
    if not 0.0 <= zero_point <= levels:
        return "activation_zero_point_out_of_range"
    return None


def _writeback_activation(
    qgraph: onnx.GraphProto,
    quantize_nodes: Dict[str, List[onnx.NodeProto]],
    uses: Dict[str, int],
    ann: QdqAnnotation,
) -> Optional[str]:
    """Put ``ann``'s learned scale/zero-point into the emitted QDQ pair(s).

    Plural: the rewrite is per-node, so an activation feeding two quantized
    nodes comes back bracketed by two independent QDQ pairs with two
    independent parameter initializers. All of them describe the same tensor
    and all of them get the same learned values -- writing only one would
    leave the model quantizing one tensor two different ways.

    Returns a skip reason if a pair the rewrite emitted is not the exclusive
    owner of its parameter initializers (so overwriting them would reach
    further than this one tensor), or if none was emitted at all.
    """
    nodes = [n for n in quantize_nodes.get(ann.tensor_name, []) if len(n.input) >= 3]
    if not nodes:
        return "quantize_node_not_emitted"
    # The rewrite creates one initializer per quantized tensor and wires it
    # into exactly the Quantize and the Dequantize of that pair -- two uses.
    # Anything else means this is not the graph we think it is.
    for node in nodes:
        if uses.get(node.input[1], 0) != 2 or uses.get(node.input[2], 0) != 2:
            return "quantization_params_shared"
    for node in nodes:
        scale_name, zp_name = node.input[1], node.input[2]
        _set_initializer_value(
            qgraph, scale_name, np.asarray(ann.scale, dtype=np.float32)
        )
        zp_init = next(i for i in qgraph.initializer if i.name == zp_name)
        zp_np = onnx.helper.tensor_dtype_to_np_dtype(zp_init.data_type)
        _set_initializer_value(
            qgraph, zp_name, np.asarray(ann.zero_point).astype(zp_np)
        )
    return None


def _writeback_weight(
    qgraph: onnx.GraphProto,
    weight: np.ndarray,
    wdq: onnx.NodeProto,
    uses: Dict[str, int],
    ann: QdqAnnotation,
) -> Optional[str]:
    """Put ``ann``'s learned weight scale into the emitted per-channel
    dequantization, re-deriving the integer codes from it.

    The scale cannot simply be swapped: the codes the rewrite computed are
    ``round(w / its own scale)``, so a different scale needs different codes or
    the weight changes value. Both are rewritten together, from the float
    weight this ingest canonicalized out of the input model -- which is the
    same float weight the trainer was fake-quantizing, so the result is the
    quantized weight the trainer was actually training against.
    """
    if ann.zero_point_dtype != onnx.TensorProto.INT8:
        return "weight_dtype_not_int8"
    if np.any(np.asarray(ann.zero_point) != 0):
        return "weight_zero_point_not_symmetric"
    codes_name, scale_name = wdq.input[0], wdq.input[1]
    if uses.get(codes_name, 0) != 1 or uses.get(scale_name, 0) != 1:
        return "quantization_params_shared"

    axis_attr = _attr(wdq, "axis")
    emitted_axis = int(axis_attr.i) if axis_attr is not None else 1
    emitted = next(i for i in qgraph.initializer if i.name == scale_name)
    channels = int(np.prod(emitted.dims)) if emitted.dims else 1

    learned = np.asarray(ann.scale, dtype=np.float32).reshape(-1)
    if ann.axis is None and learned.size == 1:
        # A per-tensor learned weight scale is coarser than the per-channel
        # scheme being emitted, but it is what was trained: broadcast it rather
        # than refuse, so the emitted weight is the one the trainer saw.
        learned = np.repeat(learned, channels)
    elif ann.axis != emitted_axis or learned.size != channels:
        return "weight_scale_shape_mismatch"

    scale = _broadcast_along_axis(learned.astype(np.float64), weight.ndim, emitted_axis)
    codes = np.clip(np.round(weight.astype(np.float64) / scale), -127, 127)
    _set_initializer_value(qgraph, codes_name, codes.astype(np.int8))
    _set_initializer_value(qgraph, scale_name, learned)
    return None


def quantize_static_keeping_qdq_scales(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
    scheme: str = "int8",
) -> QdqIngestResult:
    """Statically quantize ``model``, **keeping** the quantization parameters it
    already carries instead of re-deriving them.

    This is the ingest half of ``docs/qat.md``'s "QAT interop". Call it on a
    model exported from a QAT-trained network:
    :func:`onnxsim.quantize_static` would observe each activation's min/max
    over calibration data and compute a fresh ``(scale, zero_point)`` from it,
    discarding parameters that took a training run to produce -- and a learned
    scale is usually deliberately *tighter* than the observed range (clipping
    outliers to spend the codes where the values are), so re-deriving it is a
    real, silent regression rather than a wash.

    The pipeline is:

    1. :func:`find_existing_qdq` -- read the learned parameters off the graph.
    2. :func:`strip_existing_qdq` -- canonicalize back to the float graph, but
       only for the pairs onnxsim's static quantization can re-emit; a pair on
       a tensor it does not quantize is left in the model exactly as it was.
    3. calibrate -- but only the tensors that have no usable annotation. When
       there are none, no calibration runs and ``calibration_data`` is never
       looked at, so a fully-annotated export needs no data at all.
    4. quantize -- ordinary ``QuantizeStatic``, so the output is an ordinary
       onnxsim QDQ model that every downstream pass already understands.
    5. write the learned tensors back over what step 4 computed, bit-exactly.

    A partially quantized export -- some tensors annotated, some not -- is
    the normal case rather than an error: annotated tensors keep their learned
    parameters, the rest are calibrated.

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches for the tensors that
            still need calibrating (see :func:`onnxsim.quantize_static`).
            Ignored, and not generated, when every quantizable tensor is
            annotated.
    :param num_calibration_samples: number of random batches to generate when
            calibration is needed and ``calibration_data`` is not supplied
    :param seed: seed for that random calibration data
    :param providers: onnxruntime execution providers to calibrate on
    :param method: calibration range method for the un-annotated tensors,
            passed through to :func:`onnxsim.calibrate`
    :param scheme: ``"int8"`` (uint8 activations, :func:`onnxsim.quantize_static`)
            or ``"int16"`` (uint16 activations,
            :func:`onnxsim.quantize_static_int16`). An annotation whose
            zero-point dtype is not the scheme's is reported and recalibrated
            rather than reinterpreted.
    :returns: a :class:`QdqIngestResult` -- the model, plus which tensors kept
            their learned parameters, which were recalibrated, and which QDQ
            patterns were refused
    """
    if scheme not in _SCHEMES:
        raise ValueError(
            f"unknown scheme: {scheme!r} (expected one of {sorted(_SCHEMES)})"
        )
    zp_dtype, levels = _SCHEMES[scheme]
    model = _load(model)
    scan = find_existing_qdq(model)

    # Which pairs can be re-emitted at all is a property of the *float* graph,
    # so it takes a throwaway full strip to find out.
    probe = _infer(_strip(model, scan.annotations))
    candidates = set(C.list_quantizable_activations(probe.SerializeToString()))
    quantizable_weights = {
        node.input[1]
        for node in probe.graph.node
        if node.op_type in _QUANTIZED_OPS
        and len(node.input) >= 2
        and node.input[0] in candidates
    }

    keep: List[QdqAnnotation] = []
    unpreserved: List[SkippedQdq] = []
    for ann in scan.annotations:
        reclaimable = (
            ann.tensor_name in candidates
            if ann.role == "activation"
            else ann.tensor_name in quantizable_weights
        )
        if reclaimable:
            keep.append(ann)
        else:
            unpreserved.append(
                SkippedQdq(
                    ann.tensor_name, "not_a_quantizable_tensor", ann.dequantize_node
                )
            )
    float_model = _infer(_strip(model, keep))

    pinned: Dict[str, QdqAnnotation] = {}
    for ann in keep:
        if ann.role != "activation":
            continue
        reason = _preservable_activation(ann, zp_dtype, levels)
        if reason is None:
            pinned[ann.tensor_name] = ann
        else:
            unpreserved.append(SkippedQdq(ann.tensor_name, reason, ann.dequantize_node))

    need_calibration = candidates - set(pinned)
    ranges: Dict[str, Tuple[float, float]] = {}
    calibration_ran = bool(need_calibration)
    if calibration_ran:
        if calibration_data is None:
            calibration_data = generate_random_calibration_data(
                float_model, num_samples=num_calibration_samples, seed=seed
            )
        ranges = calibrate(
            float_model, calibration_data, providers=providers, method=method
        )
        # A tensor produced by a DequantizeLinear this module refused to touch
        # is already quantized; quantizing it again would nest a second QDQ
        # pair inside the first. The refusal is already in `scan.skipped`.
        dequantized = {
            out
            for node in float_model.graph.node
            if node.op_type == "DequantizeLinear"
            for out in node.output
        }
        ranges = {k: v for k, v in ranges.items() if k not in dequantized}
    for name, ann in pinned.items():
        ranges[name] = _range_for(
            float(np.asarray(ann.scale).reshape(-1)[0]),
            float(np.asarray(ann.zero_point).reshape(-1)[0]),
            levels,
        )

    quantize = C.quantize_static if scheme == "int8" else C.quantize_static_int16
    quantized = onnx.load_from_string(quantize(float_model.SerializeToString(), ranges))
    qgraph = quantized.graph
    uses = _use_counts(qgraph)
    producers = _producers(qgraph)
    quantize_nodes: Dict[str, List[onnx.NodeProto]] = {}
    for node in qgraph.node:
        if node.op_type == "QuantizeLinear" and node.input:
            quantize_nodes.setdefault(node.input[0], []).append(node)

    preserved: List[str] = []
    for name, ann in pinned.items():
        reason = _writeback_activation(qgraph, quantize_nodes, uses, ann)
        if reason is None:
            preserved.append(name)
        else:
            unpreserved.append(SkippedQdq(name, reason, ann.dequantize_node))

    weight_annotations = {a.tensor_name: a for a in keep if a.role == "weight"}
    if weight_annotations:
        float_nodes = {
            node.output[0]: node for node in float_model.graph.node if node.output
        }
        float_inits = {i.name: i for i in float_model.graph.initializer}
        for node in qgraph.node:
            if node.op_type not in _QUANTIZED_OPS or len(node.input) < 2:
                continue
            source = float_nodes.get(node.output[0])
            if source is None or len(source.input) < 2:
                continue
            if source.input[1] not in weight_annotations:
                continue
            ann = weight_annotations.pop(source.input[1])
            wdq = producers.get(node.input[1])
            if wdq is None or wdq.op_type != "DequantizeLinear":
                unpreserved.append(
                    SkippedQdq(
                        ann.tensor_name,
                        "quantize_node_not_emitted",
                        ann.dequantize_node,
                    )
                )
                continue
            weight = onnx.numpy_helper.to_array(float_inits[ann.tensor_name])
            reason = _writeback_weight(qgraph, weight, wdq, uses, ann)
            if reason is None:
                preserved.append(ann.tensor_name)
            else:
                unpreserved.append(
                    SkippedQdq(ann.tensor_name, reason, ann.dequantize_node)
                )
        for ann in weight_annotations.values():
            unpreserved.append(
                SkippedQdq(
                    ann.tensor_name, "quantize_node_not_emitted", ann.dequantize_node
                )
            )

    return QdqIngestResult(
        model=quantized,
        scan=scan,
        preserved=tuple(preserved),
        recalibrated=tuple(sorted(candidates - set(pinned))),
        unpreserved=tuple(unpreserved),
        calibration_ran=calibration_ran,
    )


# --------------------------------------------------------------------------- #
# Egress: emit a fake-quant model for an external trainer
# --------------------------------------------------------------------------- #
def _set_metadata(model: onnx.ModelProto, key: str, value: str) -> None:
    """``model.metadata_props[key] = value``, overwriting any existing entry.

    A local copy of ``model_info._set_metadata``'s three lines rather than an
    import of a private helper; the ``METADATA_PREFIX`` namespace those keys
    live in *is* imported, so the two stay consistent about where onnxsim's
    metadata goes.
    """
    for entry in model.metadata_props:
        if entry.key == key:
            entry.value = value
            return
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value


def _unique(name: str, taken: Set[str]) -> str:
    candidate = name
    suffix = 1
    while candidate in taken:
        candidate = f"{name}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _rename_initializers(graph: onnx.GraphProto, renames: Dict[str, str]) -> None:
    for init in graph.initializer:
        if init.name in renames:
            init.name = renames[init.name]
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in renames:
                node.input[i] = renames[name]
    surviving = [v for v in graph.value_info if v.name not in renames]
    del graph.value_info[:]
    graph.value_info.extend(surviving)


def export_fake_quant(
    model: Union[str, onnx.ModelProto],
    scheme: str = "int8",
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[str]] = None,
    method: str = "minmax",
    rename_parameters: bool = True,
) -> FakeQuantExport:
    """Emit a fake-quant (QDQ) model from a float model, naming the tensors an
    external trainer should make learnable.

    This is the egress half of ``docs/qat.md``'s "QAT interop", and the other
    end of :func:`quantize_static_keeping_qdq_scales`. onnxsim picks the scheme
    and computes a starting point for every quantization parameter by ordinary
    calibration; the user trains those parameters in whatever framework they
    already have (LSQ and friends differentiate the fake-quant with respect to
    its own step size); onnxsim re-imports the trained values through the
    ingest path. No training happens here -- the output is exactly what
    :func:`onnxsim.quantize_static` produces, plus stable names and an
    inventory.

    **How "these are the learnable tensors" is carried.** Both ways, and
    deliberately:

    - as :attr:`FakeQuantExport.learnable_scales` /
      ``learnable_zero_points`` -- the authoritative answer, in the obvious
      form, for a caller that is going to hand the names straight to an
      optimizer. Scales and zero-points are separate lists because they are
      not equally trainable: a scale is float32 and a gradient moves it
      directly, while a zero-point is an integer tensor a trainer must round
      back into after every step.
    - as ``metadata_props`` on the model, under
      :data:`LEARNABLE_SCALES_KEY` / :data:`LEARNABLE_ZERO_POINTS_KEY` (JSON
      arrays of names, namespaced with ``model_info.METADATA_PREFIX`` like
      every other key onnxsim writes -- see
      :func:`onnxsim.model_info.annotate_metadata`, which puts its metrics
      there under the same prefix). The returned lists die at the end of the
      process; a QAT round trip does not, because the model is saved to a file
      and picked up by a training script that never saw this call.

    The metadata is a convenience, never load-bearing: ingest re-derives
    everything structurally from the graph, so a trainer that rewrites the
    model (and drops its metadata_props, as most exporters do) still round
    trips. Do not treat the metadata as a guarantee that a scale was actually
    trained -- it only records which tensors were *offered*.

    Parameter names are rewritten to ``<tensor>_scale`` / ``<tensor>_zero_point``
    by default, since the rewrite's own generated names (``_v_5``) tell a user
    reading the list nothing about which layer they belong to. Pass
    ``rename_parameters=False`` to keep them.

    :param model: onnx ModelProto object or file path
    :param scheme: ``"int8"`` (uint8 activations, int8 per-output-channel
            weights) or ``"int16"`` (uint16 activations, same weights). Both
            are onnxsim's own static schemes -- see
            :func:`onnxsim.quantize_static` and
            :func:`onnxsim.quantize_static_int16`. Per-layer bit widths are not
            selected here; see ``onnxsim.mixed_precision``.
    :param calibration_data: representative input batches for the starting
            point (falls back to random data, as ``quantize_static`` does --
            fine here, since the values are about to be trained)
    :param num_calibration_samples: number of random batches when none given
    :param seed: seed for that random calibration data
    :param providers: onnxruntime execution providers to calibrate on
    :param method: calibration range method, passed to :func:`onnxsim.calibrate`
    :param rename_parameters: rename the emitted scale/zero-point initializers
            after the tensor they quantize
    :returns: a :class:`FakeQuantExport`
    """
    if scheme not in _SCHEMES:
        raise ValueError(
            f"unknown scheme: {scheme!r} (expected one of {sorted(_SCHEMES)})"
        )
    model = _infer(_load(model))
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    ranges = calibrate(model, calibration_data, providers=providers, method=method)
    quantize = C.quantize_static if scheme == "int8" else C.quantize_static_int16
    quantized = onnx.load_from_string(quantize(model.SerializeToString(), ranges))

    graph = quantized.graph
    producers = _producers(graph)
    float_nodes = {node.output[0]: node for node in model.graph.node if node.output}
    taken = {i.name for i in graph.initializer} | {v.name for v in graph.value_info}
    taken |= {v.name for v in list(graph.input) + list(graph.output)}

    renames: Dict[str, str] = {}
    scales: List[str] = []
    zero_points: List[str] = []

    def register(old: str, preferred: str, into: List[str]) -> None:
        new = old
        if rename_parameters:
            new = _unique(preferred, taken)
            renames[old] = new
        into.append(new)

    for node in graph.node:
        if node.op_type == "QuantizeLinear" and len(node.input) >= 3:
            register(node.input[1], f"{node.input[0]}_scale", scales)
            register(node.input[2], f"{node.input[0]}_zero_point", zero_points)
    for node in graph.node:
        if node.op_type not in _QUANTIZED_OPS or len(node.input) < 2:
            continue
        wdq = producers.get(node.input[1])
        source = float_nodes.get(node.output[0])
        if wdq is None or wdq.op_type != "DequantizeLinear" or source is None:
            continue
        # The weight's float name in the *input* model -- the emitted int8
        # codes have a generated name that says nothing about the layer.
        register(wdq.input[1], f"{source.input[1]}_scale", scales)

    if rename_parameters:
        _rename_initializers(graph, renames)
    _set_metadata(quantized, LEARNABLE_SCALES_KEY, json.dumps(scales))
    _set_metadata(quantized, LEARNABLE_ZERO_POINTS_KEY, json.dumps(zero_points))
    return FakeQuantExport(quantized, tuple(scales), tuple(zero_points))
