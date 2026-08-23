import argparse
import copy
import os
import re
import shutil
import sys
import tempfile
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import onnx  # type: ignore
import onnx.checker  # type: ignore
import onnx.helper  # type: ignore
import onnx.numpy_helper  # type: ignore
import onnx.shape_inference  # type: ignore
from google.protobuf.message import EncodeError
from rich import print
from rich.text import Text

import onnxsim.onnxsim_cpp2py_export as C

from . import backend, model_checking, model_info, profile_merge, version

TensorShape = List[int]
TensorShapes = Dict[str, TensorShape]
TensorShapesWithOptionalKey = Dict[Optional[str], TensorShape]
# A user-supplied graph rewriter: takes a model and returns a rewritten model,
# or mutates it in place and returns ``None``. Passed to ``simplify`` via
# ``custom_rewriter`` and run inside the simplification fixed point. Returning
# ``False`` reports that nothing was rewritten, which lets onnxsim skip copying
# an unchanged model back through the C++ core (see ``_GraphRewriterAdapter``).
ModelRewriter = Callable[[onnx.ModelProto], Union[onnx.ModelProto, bool, None]]
# A data-driven rewrite rule: a ``(pattern, replacement)`` pair of
# ``onnx.FunctionProto``. The pattern's inputs are wildcards binding to graph
# values, its body is the subgraph to match, and its outputs are the values
# rewired to the replacement's outputs. Unlike ``ModelRewriter`` (a Python
# callable), a rule is pure data, so the same rules work from the C and Rust
# bindings too. Build one with ``onnx.parser.parse_function`` (see the README).
FunctionRewriteRule = Tuple[onnx.FunctionProto, onnx.FunctionProto]
Unit = Literal["B", "KB", "MB", "GB", "TB"]

UNIT_MAP: dict[Unit, int] = {
    "B": 1,
    "KB": 1 << 10,
    "MB": 1 << 20,
    "GB": 1 << 30,
    "TB": 1 << 40,
}


def parse_size(size: str) -> int:
    m = re.fullmatch(r"([\d.]+)\s*([KMGT]?B)", size.strip(), re.I)
    if not m:
        raise ValueError(size)
    number: float = float(m.group(1))
    unit: Unit = m.group(2).upper()  # type: ignore
    return int(number * UNIT_MAP[unit])


def get_output_names(model: onnx.ModelProto) -> List[str]:
    output_names = [opt.name for opt in model.graph.output]
    return output_names


def remove_unused_output(
    model: onnx.ModelProto, unused_output: Sequence[str]
) -> onnx.ModelProto:
    unused_output_names = unused_output
    output_names = get_output_names(model)
    for unused_output_name in unused_output_names:
        if unused_output_name not in output_names:
            raise RuntimeError(
                f'The model doesn\'t have output named "{unused_output_name}"'
            )
    for graph_output in copy.deepcopy(model.graph.output):
        if graph_output.name in unused_output_names:
            model.graph.output.remove(graph_output)
    return model


def _default_domain_opset(model: onnx.ModelProto) -> int:
    """Opset version imported for the default (ai.onnx) domain, or 0 if none."""
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return imp.version
    return 0


# Value-baking fusions such as ``fuse_bn_into_conv`` materialise helper nodes --
# notably ``Cast`` -- using the modern operator encoding, where ``Cast``'s ``to``
# attribute is an INT (a ``TensorProto`` data-type enum). That encoding only
# became valid in opset 6; before that ``to`` was a STRING type name. Enabling
# those fusions on an older-opset graph therefore emits nodes the graph's own
# opset rejects, and onnx / onnxruntime abort with e.g. "Mismatched attribute
# type in 'Cast_0 : to'. Expected: 'STRING', actual: 'INT'" (the onnx-caffe2
# opset-3 ``resnet50-caffe2-v1-3`` hit exactly this). Below this opset we leave
# IR<4 models untouched so onnxoptimizer keeps treating their initializers as
# runtime inputs and skips the value-baking fusions -- the graph passes through
# unchanged rather than crashing.
_MIN_OPSET_FOR_INITIALIZER_FOLD = 6


def remove_initializer_from_input(model: onnx.ModelProto) -> onnx.ModelProto:
    # IR version 4 (ONNX 1.4) is the first that allows an initializer to *not*
    # also be a graph input. Older IR (v3 and below) required every initializer
    # to appear in ``graph.input``, and leaving it there makes onnxoptimizer
    # treat it as a runtime input rather than a constant
    # (``is_constant_initializer`` returns false), which silently blocks
    # value-baking fusions such as ``fuse_bn_into_conv`` -- e.g. the plain
    # Conv+BN chains of the opset-8 ``resnet101-v1-7`` were left completely
    # unsimplified. Bump such models to IR 4 so the removal below is legal and
    # the freed initializers fold like any other constant -- but only when the
    # opset is new enough for the ops those fusions insert; on an ancient-opset
    # graph the bump would let a fusion emit a node the opset rejects (see
    # ``_MIN_OPSET_FOR_INITIALIZER_FOLD``), so leave such models alone.
    if model.ir_version < 4 and (
        _default_domain_opset(model) < _MIN_OPSET_FOR_INITIALIZER_FOLD
    ):
        return model
    initializer_names = [x.name for x in model.graph.initializer]
    removed_any = False
    for graph_input in copy.deepcopy(model.graph.input):
        if graph_input.name in initializer_names:
            model.graph.input.remove(graph_input)
            removed_any = True
    if removed_any and model.ir_version < 4:
        model.ir_version = 4
    return model


def check_and_update_input_shapes(
    model: onnx.ModelProto, input_shapes: Optional[TensorShapesWithOptionalKey]
) -> Optional[TensorShapes]:
    if input_shapes is None:
        return None

    def get_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
        initializer_names = [x.name for x in model.graph.initializer]
        return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]

    def get_input_names(model: onnx.ModelProto) -> List[str]:
        input_names = [ipt.name for ipt in get_inputs(model)]
        return input_names

    input_names = get_input_names(model)
    if None in input_shapes:
        if len(input_names) == 1:
            input_shapes[input_names[0]] = input_shapes[None]
            del input_shapes[None]
        else:
            raise RuntimeError(
                'The model has more than 1 inputs, please use the format "input_name:dim0,dim1,...,dimN" in --input-shape'
            )
    for x in input_shapes:
        if x not in input_names:
            raise RuntimeError('The model doesn\'t have input named "{}"'.format(x))

    return input_shapes  # type: ignore


# A very very large threshold
DEFAULT_TENSOR_SIZE_THRESHOLDHOLD = "1.5GB"


# ONNX ``TensorProto`` element types that onnxoptimizer's tensor-value hashing
# (``cse_util.h``) knows how to hash. Any other type makes those passes raise
# ``RuntimeError: no supported data type: <N>``. We enumerate the *supported*
# types (rather than the unsupported ones) so that element types added to ONNX
# in the future are treated as unhashable by default instead of silently
# crashing the optimizer.
_CSE_HASHABLE_ELEM_TYPES = frozenset(
    {
        onnx.TensorProto.UNDEFINED,
        onnx.TensorProto.BOOL,
        onnx.TensorProto.INT8,
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.UINT32,
        onnx.TensorProto.UINT64,
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.COMPLEX64,
        onnx.TensorProto.COMPLEX128,
        onnx.TensorProto.STRING,
    }
)

# onnxoptimizer passes that hash tensor *values* via ``cse_util.h``. They crash
# on tensors whose element type they cannot hash -- for example the
# ``float8_e4m3fn`` zero points in NVIDIA ModelOpt fp8 QDQ models (see GitHub
# issue #348), or int4/uint4/float8 tensors in general.
_TENSOR_VALUE_HASHING_OPTIMIZERS = (
    "eliminate_common_subexpression",
    "eliminate_duplicate_initializer",
)


def _iter_tensor_data_types(graph: onnx.GraphProto):
    """Yield the ``data_type`` of every tensor stored inside ``graph``.

    Covers initializers, ``Constant`` (and other) tensor/tensors attributes and
    recurses into subgraphs, i.e. all the places onnxoptimizer might hash a
    tensor value.
    """
    for initializer in graph.initializer:
        yield initializer.data_type
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t.data_type
            for tensor in attr.tensors:
                yield tensor.data_type
            if attr.HasField("g"):
                yield from _iter_tensor_data_types(attr.g)
            for subgraph in attr.graphs:
                yield from _iter_tensor_data_types(subgraph)


def _has_cse_unhashable_tensor(model: onnx.ModelProto) -> bool:
    """Whether ``model`` contains a tensor onnxoptimizer's CSE cannot hash."""
    return any(
        data_type not in _CSE_HASHABLE_ELEM_TYPES
        for data_type in _iter_tensor_data_types(model.graph)
    )


def _formal_parameter_tuple(param) -> Tuple:
    """Marshal an ``onnx.defs.OpSchema.FormalParameter`` for the C++ importer."""
    return (
        param.name,
        param.description,
        param.type_str,
        int(param.option),
        bool(param.is_homogeneous),
        int(param.min_arity),
    )


def _register_schema_in_onnxsim(schema) -> None:
    """Register a single ``onnx.defs.OpSchema`` into onnxsim's C++ registry."""
    inputs = [_formal_parameter_tuple(p) for p in schema.inputs]
    outputs = [_formal_parameter_tuple(p) for p in schema.outputs]

    attributes = []
    for attr in schema.attributes.values():
        # ``default_value`` is an ``onnx.AttributeProto`` that nanobind
        # serializes across the library boundary; an UNDEFINED type means the
        # attribute has no default (it is either required or plainly optional).
        default_value = attr.default_value
        if default_value is None:
            default_value = onnx.AttributeProto()
        attributes.append(
            (
                attr.name,
                attr.description,
                int(attr.type),
                bool(attr.required),
                default_value,
            )
        )

    type_constraints = [
        (tc.type_param_str, list(tc.allowed_type_strs), tc.description)
        for tc in schema.type_constraints
    ]

    C._register_schema(
        schema.name,
        schema.domain,
        int(schema.since_version),
        schema.doc or "",
        inputs,
        outputs,
        attributes,
        type_constraints,
        bool(schema.has_type_and_shape_inference_function),
    )


def import_onnx_schemas() -> int:
    """Copy operator schemas from the Python ``onnx`` module into onnxsim.

    onnxsim links its own copy of the ONNX C++ library, so its operator schema
    registry is completely separate from the one the ``onnx`` Python module uses.
    A schema a user adds via ``onnx.defs.register_schema`` -- for example to
    describe a custom/user-defined operator -- is therefore invisible to
    onnxsim, and simplifying such a model fails ``check_model`` with
    "No Op registered for <op> ..." (GitHub issue #326).

    This imports every operator schema that onnxsim does not already know about
    from the ``onnx`` registry into onnxsim's registry, so custom operators pass
    validation and are preserved through simplification. Standard ONNX operators
    (already present in onnxsim) are left untouched. It is idempotent and safe to
    call repeatedly: an operator onnxsim already knows -- including one imported
    by a previous call -- is skipped.

    The type/shape inference function attached to an ``onnx`` schema is native
    code inside the ``onnx`` library and cannot be transferred directly, so when
    a schema has one, onnxsim registers a trampoline that calls it back through
    ``onnx.shape_inference.infer_node_outputs`` during shape inference. Custom
    operators without an inference function are still imported (shape inference
    flows past them).

    :return: the number of schemas imported.
    """
    import onnx.defs

    try:
        schemas = onnx.defs.get_all_schemas_with_history()
    except Exception:
        return 0

    imported = 0
    # Cache onnxsim's knowledge per (op, domain) so the native check runs once
    # per operator rather than once per registered version.
    onnxsim_knows: Dict[Tuple[str, str], bool] = {}
    for schema in schemas:
        try:
            key = (schema.name, schema.domain)
            known = onnxsim_knows.get(key)
            if known is None:
                known = C._has_schema(schema.name, schema.domain)
                onnxsim_knows[key] = known
            if known:
                continue
            _register_schema_in_onnxsim(schema)
            imported += 1
        except Exception:
            # A single unusual schema must never break simplification: skip it
            # and keep importing the rest.
            continue
    return imported


def export_safetensors(model: onnx.ModelProto, out_path: str) -> None:
    """Export ``model`` to a standalone safetensors archive at ``out_path``.

    Every initializer's bytes move into the archive with real, byte-accurate
    offsets -- openable by the ``safetensors`` Python package / HF tooling with
    no onnxsim involved -- and the graph itself is embedded alongside them, so
    ``out_path`` alone is both the model's weights and its graph. Reload it
    with :func:`import_safetensors`.
    """
    C.export_safetensors(model.SerializeToString(), out_path)


def import_safetensors(in_path: str) -> onnx.ModelProto:
    """Import a standalone safetensors archive back into an ``onnx.ModelProto``.

    ``in_path`` must be an archive produced by :func:`export_safetensors` (or
    any other tool following the same self-describing-archive convention: an
    embedded ``model.onnx`` entry alongside the tensors). Raises
    ``RuntimeError`` if the archive has no embedded onnxsim model, e.g. a
    plain weights-only safetensors file with no graph to import.
    """
    model = onnx.ModelProto()
    model.ParseFromString(C.import_safetensors(in_path))
    return model


def export_gguf(model: onnx.ModelProto, out_path: str) -> None:
    """GGUF counterpart of :func:`export_safetensors`."""
    C.export_gguf(model.SerializeToString(), out_path)


def import_gguf(in_path: str) -> onnx.ModelProto:
    """GGUF counterpart of :func:`import_safetensors`."""
    model = onnx.ModelProto()
    model.ParseFromString(C.import_gguf(in_path))
    return model


def import_gguf_weights(
    model: Union[str, onnx.ModelProto], gguf_path: str
) -> Tuple[onnx.ModelProto, List[str]]:
    """
    Hydrate ``model``'s initializers, by name, from ``gguf_path`` -- unlike
    :func:`import_gguf`, this works on any GGUF file, including a plain
    third-party weights-only checkpoint (e.g. a Hugging Face GGUF export)
    with no embedded onnxsim model: bring your own graph (e.g. exported by
    another tool for the same architecture) with initializers named to
    match the checkpoint's own tensor names, and this fills in their
    values.

    A K-quant tensor (``Q4_K``/``Q5_K``/``Q6_K``/``Q8_0`` -- the block
    format most real quantized checkpoints, e.g. Unsloth's GGUF exports,
    actually use for the bulk of their weights) is dequantized to float32
    in the process; every other quantized GGML format (the legacy ``Q4_0``
    family, every ``IQ*`` variant, ...) has no decoder here and is skipped
    -- see the second return value.

    :param model: onnx ModelProto object or file path -- the graph to
            hydrate. Its initializers' own declared dtype/shape are left
            alone except for a matched K-quant tensor, whose data_type is
            forced to FLOAT (the only meaningful type for a dequantized
            result) regardless of what the initializer previously declared.
    :param gguf_path: path to the GGUF file to pull weight values from
    :returns: ``(model, skipped)`` -- the hydrated model, and the names of
            GGUF tensors present in the file but skipped because their
            quantized format has no decoder here (the legacy ``Q4_0``
            family, every ``IQ*`` variant, ...). A GGUF tensor with no
            matching initializer name in ``model`` is simply not brought
            in -- it does NOT appear in ``skipped``, which reports only
            unsupported *formats*, not name mismatches.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    model_bytes, skipped = C.import_gguf_weights(model.SerializeToString(), gguf_path)
    result = onnx.ModelProto()
    result.ParseFromString(model_bytes)
    return result, skipped


def quantize_dynamic(model: Union[str, onnx.ModelProto]) -> onnx.ModelProto:
    """
    Dynamically quantize every MatMul, and every "vanilla" Gemm (transA=0,
    alpha=1, beta=1), whose weight is a constant 2-D float32 tensor.

    The weight is quantized to INT8 ahead of time (per output channel,
    symmetric, from its static values -- no calibration data is needed),
    while the activation is quantized to uint8 *in the graph* via
    ``DynamicQuantizeLinear``, which computes its own scale/zero-point from
    each run's actual input range. This mirrors the "dynamic quantization"
    scheme ONNX Runtime's ``quantize_dynamic`` applies to MatMul/Gemm.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Nodes that do not match (dynamic or non-2-D weights, non-default Gemm
    attributes, non-float32 operands, an opset older than 11 -- which
    ``DynamicQuantizeLinear`` requires) are left untouched. Consider calling
    :func:`simplify` before and/or after to clean up the graph.

    :param model: onnx ModelProto object or file path
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.quantize_dynamic(model.SerializeToString()))


def quantize_ternary(model: Union[str, onnx.ModelProto]) -> onnx.ModelProto:
    """
    Dynamically quantize every MatMul, and every "vanilla" Gemm (transA=0,
    alpha=1, beta=1), whose constant weight is *structurally ternary*: every
    element of every output column is one of ``{-s, 0, +s}`` for that
    column's own scale ``s``. This is the weight representation `BitNet
    b1.58 <https://github.com/microsoft/BitNet>`_ and similar ternary-weight
    models use internally, which a generic ONNX export still stores as a
    dense float32 initializer -- 16x larger than it needs to be, and running
    on the generic float MatMul kernel.

    A detected node gets exactly :func:`quantize_dynamic`'s rewrite
    (``DynamicQuantizeLinear`` + ``MatMulInteger`` + dequantize) -- the only
    difference is that the weight's INT8 encoding here is a **lossless**
    ``{-1, 0, 1}`` code (derived structurally, not by rounding), rather than
    a rounded approximation of the weight's full dynamic range. A node whose
    weight is not structurally ternary is left untouched by this call --
    combine with :func:`quantize_dynamic` (which fires on any constant
    float32 weight) if a model mixes ternary and ordinary layers and both
    should be quantized.

    This only targets standard ONNX operators, like the rest of onnxsim's
    quantization passes -- not a contrib op like
    ``com.microsoft::MatMulNBits``, which would pack the ternary codes down
    to 2 bits for a further ~4x weight-storage saving on top of what this
    function gets, at the cost of only running on onnxruntime builds that
    ship it. See docs/ternary-quantization.md.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Nodes that do not match (dynamic or non-2-D weights, non-default Gemm
    attributes, non-float32 operands, a non-ternary weight, an opset older
    than 11) are left untouched. Consider calling :func:`simplify` before
    and/or after to clean up the graph.

    :param model: onnx ModelProto object or file path
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.quantize_ternary(model.SerializeToString()))


def quantize_weight_only(model: Union[str, onnx.ModelProto]) -> onnx.ModelProto:
    """
    Weight-only quantize every MatMul, every "vanilla" Gemm (transA=0,
    alpha=1, beta=1), and every Conv, whose weight is a constant float32
    tensor (2-D for MatMul/Gemm, rank >= 3 for Conv).

    The weight is quantized to INT8 ahead of time (per output channel,
    symmetric, from its static values -- the same scheme
    :func:`quantize_dynamic`/:func:`quantize_static` use), inserting a single
    ``DequantizeLinear`` in its place. Unlike both of those, the activation is
    never touched: no ``DynamicQuantizeLinear``, no QuantizeLinear/
    DequantizeLinear pair, no calibration data of any kind. This only shrinks
    the model's weight storage (~4x for the quantized weights); it does not
    change activation precision or add any runtime quantize/dequantize cost
    to the activation path. This mirrors the "weight-only quantization"
    scheme most real-world weight-heavy ONNX deployments -- large linear/
    embedding layers in transformer-style decoders, for example -- actually
    ship, as opposed to full activation quantization.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Nodes that do not match (dynamic or unsupported-rank weights, non-default
    Gemm attributes, non-float32 operands, an opset older than 13 -- which
    ``DequantizeLinear``'s per-channel ``axis`` requires) are left untouched.
    Consider calling :func:`simplify` before and/or after to clean up the
    graph.

    :param model: onnx ModelProto object or file path
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.quantize_weight_only(model.SerializeToString()))


def quantize_weight_only_int4(model: Union[str, onnx.ModelProto]) -> onnx.ModelProto:
    """
    Block-wise INT4 weight-only quantize every MatMul, every "vanilla" Gemm
    (transA=0, alpha=1, beta=1), and every Conv, whose weight is a constant
    float32 tensor whose flattened reduction size -- ``K`` for MatMul/Gemm;
    ``Cin/groups * prod(kernel dims)`` for Conv -- is evenly divisible by 32.

    The weight is quantized to INT4 (values in ``[-7, 7]``) with a separate
    symmetric scale per 32-element block of that reduction, per output
    channel -- the GPTQ/AWQ-style block quantization real weight-heavy
    LLM/ASR deployments increasingly ship -- inserting a single
    ``DequantizeLinear(..., block_size=32)`` in its place (Conv's weight is
    flattened to 2-D for this, then a ``Reshape`` restores its original
    shape). Like :func:`quantize_weight_only`, the activation is never
    touched: no calibration data, no runtime quantize/dequantize cost on the
    activation path. Storage is roughly half of :func:`quantize_weight_only`'s
    INT8 scheme for a comparable accuracy cost, since block-local scales
    absorb most of what a single wider per-channel range would otherwise
    lose. Uses ONNX opset 21's INT4 tensor type and ``DequantizeLinear``'s
    ``block_size`` attribute -- standard ONNX, not a contrib op, so the
    result loads on any conformant opset-21+ runtime.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Nodes that do not match (dynamic or unsupported-rank weights, a
    reduction size not divisible by 32, non-default Gemm attributes,
    non-float32 operands, an opset older than 21) are left untouched.
    Consider calling :func:`simplify` before and/or after to clean up the
    graph.

    :param model: onnx ModelProto object or file path
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.quantize_weight_only_int4(model.SerializeToString()))


def quantize_weight_only_int16(model: Union[str, onnx.ModelProto]) -> onnx.ModelProto:
    """
    INT16 weight-only quantize every MatMul, every "vanilla" Gemm (transA=0,
    alpha=1, beta=1), and every Conv, whose weight is a constant float32
    tensor.

    The weight is quantized to INT16 (per output channel, symmetric, scale =
    ``max(|w|) / 32767``) with a single ``DequantizeLinear(axis=...)`` in its
    place -- the exact same per-channel scheme :func:`quantize_weight_only`
    uses, just with INT16's ~8x finer step (1/32767 relative) instead of
    INT8's 1/127. Like :func:`quantize_weight_only`, the activation is never
    touched: no calibration data, no runtime quantize/dequantize cost on the
    activation path.

    That extra resolution matters specifically for channels with a few
    extreme-outlier weights, where INT8's coarser step would leave the
    channel's *typical* (median-magnitude) weight rounding to within one
    quantization step of zero -- effectively lost.
    :func:`estimate_quantization_precision` flags exactly this case (a
    channel's ``max(|w|) / median(|w|)`` ratio past 127) and recommends INT16
    as one fix; this is that fix. The tradeoff: INT16 is only ~2x smaller
    than float32 (INT8 is ~4x), so use this for the specific outlier-heavy
    weights :func:`quantize_weight_only`'s INT8 handles poorly, not as a
    blanket replacement for it. Uses ONNX opset 21's INT16
    ``QuantizeLinear``/``DequantizeLinear`` type support -- standard ONNX,
    not a contrib op.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Nodes that do not match (dynamic or unsupported-rank weights, non-default
    Gemm attributes, non-float32 operands, an opset older than 21) are left
    untouched. Consider calling :func:`simplify` before and/or after to clean
    up the graph.

    :param model: onnx ModelProto object or file path
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.quantize_weight_only_int16(model.SerializeToString())
    )


def quantize_weight_only_int8_block(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    Block-wise INT8 weight-only quantize every MatMul, every "vanilla" Gemm
    (transA=0, alpha=1, beta=1), and every Conv, whose weight is a constant
    float32 tensor whose flattened reduction size -- ``K`` for MatMul/Gemm;
    ``Cin/groups * prod(kernel dims)`` for Conv -- is evenly divisible by 32.

    The weight is quantized to INT8 (values in ``[-127, 127]``) with a
    separate symmetric scale per 32-element block of that reduction, per
    output channel -- the same block-wise granularity
    :func:`quantize_weight_only_int4` uses, just at INT8's wider code range
    -- inserting a single ``DequantizeLinear(..., block_size=32)`` in its
    place (Conv's weight is flattened to 2-D for this, then a ``Reshape``
    restores its original shape). Like :func:`quantize_weight_only`, the
    activation is never touched: no calibration data, no runtime
    quantize/dequantize cost on the activation path.

    This sits between :func:`quantize_weight_only`'s single per-channel INT8
    scale (coarser, no block overhead) and
    :func:`quantize_weight_only_int4`'s block-wise INT4 (finer blocks, but
    only 15 representable codes per block): the same storage as
    :func:`quantize_weight_only` (INT8 codes are still 1 byte each; only the
    scale tensor grows, from one float per channel to one float per (block,
    channel) pair), with resolution closer to a per-block scheme. Uses ONNX
    opset 21's ``DequantizeLinear`` ``block_size`` attribute -- standard
    ONNX, not a contrib op -- the same opset floor as
    :func:`quantize_weight_only_int4`, even though plain INT8 itself needs
    only opset 13.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Nodes that do not match (dynamic or unsupported-rank weights, a
    reduction size not divisible by 32, non-default Gemm attributes,
    non-float32 operands, an opset older than 21) are left untouched.
    Consider calling :func:`simplify` before and/or after to clean up the
    graph.

    :param model: onnx ModelProto object or file path
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.quantize_weight_only_int8_block(model.SerializeToString())
    )


def quantize_fp16(
    model: Union[str, onnx.ModelProto], keep_io_types: bool = True
) -> onnx.ModelProto:
    """
    Convert every float32 weight -- and, by default, every internal
    activation -- in ``model`` to float16.

    Unlike every other ``quantize_*`` function in onnxsim, this needs no
    calibration data and has no scale/zero-point: float16 is still a
    floating-point format (5 exponent bits / 10 mantissa bits, versus
    float32's 8/23), not an integer scheme, so every float32 value is simply
    rounded to its nearest representable float16 value. A value outside
    float16's finite range (magnitude > 65504, including +-Inf itself) is
    clamped to float16's largest finite magnitude rather than rounded to an
    infinity.

    With ``keep_io_types`` (the default ``True``), the model's own external
    input/output types stay float32 -- a ``Cast`` is inserted right after
    each float32 graph input and right before each float32 graph output, so
    the model's public interface is unchanged and only its internal weights
    and compute switch to float16. Pass ``False`` to redeclare graph
    inputs/outputs float16 directly instead (no casts; callers must then
    feed/read float16 tensors).

    No node's op_type or attributes are touched, and there is no per-op
    float16-support check: an ordinary feedforward graph ends up computing
    end-to-end in float16 as a side effect of every value along the way now
    being float16-typed, since almost every ONNX op propagates its input
    dtype to its output dtype. A model containing an op with no float16
    kernel in the runtime it is deployed on will fail at *execution* time,
    not at conversion time here -- the same limitation every other
    float32-to-float16 model converter (e.g. ``onnxconverter-common``'s
    ``convert_float_to_float16``) has.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Only the top-level graph is converted -- nodes inside control-flow
    subgraphs (If/Loop/Scan bodies) are left untouched -- and an initializer
    whose name is also a graph input (the rarely-used ONNX "optional input
    with a default value" convention) is left alone entirely. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: onnx ModelProto object or file path
    :param keep_io_types: keep the graph's own external input/output types
            at float32 (inserting boundary Cast nodes) instead of
            redeclaring them float16 directly
    :returns: the converted onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.quantize_fp16(model.SerializeToString(), keep_io_types)
    )


def quantize_bf16(
    model: Union[str, onnx.ModelProto], keep_io_types: bool = True
) -> onnx.ModelProto:
    """
    Convert every float32 weight -- and, by default, every internal
    activation -- in ``model`` to bfloat16.

    The same calibration-free, whole-graph conversion as
    :func:`quantize_fp16`, just to a different narrow floating-point format:
    bfloat16 keeps float32's full 8-bit exponent range and narrows only the
    mantissa (7 bits instead of float32's 23), so every finite float32 value
    maps to a finite bfloat16 value -- unlike float16, no clamping is ever
    needed.

    With ``keep_io_types`` (the default ``True``), the model's own external
    input/output types stay float32 -- a ``Cast`` is inserted right after
    each float32 graph input and right before each float32 graph output, so
    the model's public interface is unchanged and only its internal weights
    and compute switch to bfloat16. Pass ``False`` to redeclare graph
    inputs/outputs bfloat16 directly instead (no casts; callers must then
    feed/read bfloat16 tensors).

    No node's op_type or attributes are touched, and there is no per-op
    bfloat16-support check: an ordinary feedforward graph ends up computing
    end-to-end in bfloat16 as a side effect of every value along the way now
    being bfloat16-typed, since almost every ONNX op propagates its input
    dtype to its output dtype. A model containing an op with no bfloat16
    kernel in the runtime it is deployed on will fail at *execution* time,
    not at conversion time here.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Only the top-level graph is converted -- nodes inside control-flow
    subgraphs (If/Loop/Scan bodies) are left untouched -- and an initializer
    whose name is also a graph input (the rarely-used ONNX "optional input
    with a default value" convention) is left alone entirely. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: onnx ModelProto object or file path
    :param keep_io_types: keep the graph's own external input/output types
            at float32 (inserting boundary Cast nodes) instead of
            redeclaring them bfloat16 directly
    :returns: the converted onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.quantize_bf16(model.SerializeToString(), keep_io_types)
    )


def quantize_fp8(
    model: Union[str, onnx.ModelProto],
    format: str = "e4m3",
    keep_io_types: bool = True,
) -> onnx.ModelProto:
    """
    Convert every float32 weight -- and, by default, every internal
    activation -- in ``model`` to an 8-bit floating-point format.

    The same calibration-free, whole-graph conversion as
    :func:`quantize_fp16`/:func:`quantize_bf16`, just to a much narrower
    floating-point format. ``format`` selects which one:

    - ``"e4m3"`` (the default): E4M3FN -- 4 exponent bits, 3 mantissa bits,
      max finite magnitude 448. No infinities. Typically used for weights.
    - ``"e5m2"``: 5 exponent bits, 2 mantissa bits, max finite magnitude
      57344 -- a dynamic range similar to float16. Typically used for
      gradients.

    Both are converted with saturation: a value whose magnitude exceeds the
    target format's max finite value (including +-Inf itself) is clamped to
    it rather than mapped to an infinity/NaN, the same design choice
    :func:`quantize_fp16` makes for its own out-of-range values. Rounding is
    round-to-nearest, ties-to-even -- float8's mantissa is only 2-3 bits
    wide, so exact ties are common enough on real data that the simpler
    ties-away-from-zero rule :func:`quantize_fp16`/:func:`quantize_bf16` use
    is not a good enough approximation here.

    With ``keep_io_types`` (the default ``True``), the model's own external
    input/output types stay float32 -- a ``Cast`` is inserted right after
    each float32 graph input and right before each float32 graph output, so
    the model's public interface is unchanged and only its internal weights
    and compute switch to the target float8 format. Pass ``False`` to
    redeclare graph inputs/outputs in that format directly instead (no
    casts; callers must then feed/read float8 tensors).

    No node's op_type or attributes are touched, and there is no per-op
    float8-support check: an ordinary feedforward graph ends up computing
    end-to-end in the target format as a side effect of every value along
    the way now being that format, since almost every ONNX op propagates its
    input dtype to its output dtype. A model containing an op with no float8
    kernel in the runtime it is deployed on will fail at *execution* time,
    not at conversion time here -- and as of most current runtimes, float8
    compute kernel coverage is narrower than even bfloat16's, so expect this
    to mostly be useful for storage-size reduction and for deployment
    targets with real float8 kernel support today.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Only the top-level graph is converted -- nodes inside control-flow
    subgraphs (If/Loop/Scan bodies) are left untouched -- and an initializer
    whose name is also a graph input (the rarely-used ONNX "optional input
    with a default value" convention) is left alone entirely. Consider
    calling :func:`simplify` before and/or after to clean up the graph.
    Casting to/from these types needs opset >= 19.

    :param model: onnx ModelProto object or file path
    :param format: target float8 format, ``"e4m3"`` (default) or ``"e5m2"``
    :param keep_io_types: keep the graph's own external input/output types
            at float32 (inserting boundary Cast nodes) instead of
            redeclaring them in the target format directly
    :returns: the converted onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.quantize_fp8(model.SerializeToString(), format, keep_io_types)
    )


def simplify(
    model: Union[str, onnx.ModelProto],
    check_n: int = 0,
    perform_optimization: bool = True,
    skip_fuse_bn: bool = False,
    overwrite_input_shapes=None,
    test_input_shapes=None,
    skipped_optimizers: Optional[List[str]] = None,
    skip_constant_folding=False,
    skip_shape_inference=False,
    input_data=None,
    dynamic_input_shape: bool = False,
    custom_lib: Optional[str] = None,
    include_subgraph: bool = False,
    unused_output: Optional[Sequence[str]] = None,
    tensor_size_threshold: str = DEFAULT_TENSOR_SIZE_THRESHOLDHOLD,
    mutable_initializer: bool = False,
    *,
    initializers_as_constants: bool = True,
    inline_functions: bool = False,
    import_custom_schemas: bool = True,
    input_shapes=None,
    target_opset_version: Optional[int] = None,
    custom_rewriter: Optional[ModelRewriter] = None,
    function_rewrite_rules: Optional[Sequence[FunctionRewriteRule]] = None,
    check_rtol: float = 1e-4,
    check_atol: float = 1e-5,
    input_fill: str = "random",
    providers: Optional[Sequence[backend.Provider]] = None,
    profile: Optional[str] = None,
    ort_profile: Optional[str] = None,
    merge_ort_profile: bool = False,
) -> Tuple[onnx.ModelProto, bool]:
    """
    :param model: onnx ModelProto object or file path
    :param check_n: The simplified model will be checked for `check_n` times by random inputs
    :param perform_optimization: Whether to run onnx optimizer on the model
    :param skip_fuse_bn: Skip fuse_bn_into_conv onnx optimizer
    :param overwrite_input_shapes: If the model has dynamic input shape, user must pass a fixed input shape
            for generating random inputs and checking equality.
    :param test_input_shapes: If the model has dynamic input shape, user must pass a fixed input shape
            for generating random inputs and checking equality.
    :param skipped_optimizers: Skip some specific onnx optimizers
    :param skip_constant_folding: Skip constant folding
    :param skip_shape_inference: Skip shape inference (sometimes shape inference will crash)
    :param input_data: Feed custom input data for checking if needed
    :param dynamic_input_shape: Deprecated. Not needed anymore.
    :param custom_lib: onnxruntime custom ops's shared library
    :param include_subgraph: Simplify subgraph (e.g. true graph and false graph of "If" operator) instead of only the main graph
    :param unused_output: name of unused outputs that will be eliminated from the model
    :param initializers_as_constants: Whether initializers are treated as constant tensors
            (the default, ``True``). When set to ``False``, initializers are treated as
            non-constant: constant folding leaves nodes that depend only on initializers in the
            graph, and the onnx optimizer's value-baking passes (e.g. ``fuse_bn_into_conv``) are
            told to leave initializer-backed weights alone, so the weights survive simplification
            as tunable tensors. ``Constant`` nodes are still folded either way. This is orthogonal
            to ``mutable_initializer`` (which controls whether initializers also remain graph
            inputs).
    :param inline_functions: When True, inline the model's local (model-defined) functions into
            the main graph before simplifying (using onnx's inliner), flattening function calls
            into plain ops so the optimizer, shape inference and constant folding can see through
            them. Schema-defined (built-in) functions are left alone. Defaults to False, which
            leaves the model's functions untouched.
    :param import_custom_schemas: Import operator schemas registered in the Python `onnx` module
            (e.g. via `onnx.defs.register_schema`) into onnxsim's own registry so models using
            custom operators pass validation. Set to False to disable this and leave onnxsim's
            registry untouched.
    :param input_shapes: Deprecated. Please use `overwrite_input_shapes` and/or `test_input_shapes` instead.
    :param target_opset_version: Convert the model to this opset version (of the default ONNX domain)
            before simplifying, using onnx's version converter (run inside the C++ core so every
            binding shares the behavior). This can be used to upgrade (or downgrade) the model's
            opset during simplification. When None (the default), the opset version is left unchanged.
    :param custom_rewriter: An optional callable ``ModelProto -> Optional[ModelProto]`` run as an extra
            stage inside onnxsim's simplification fixed point, interleaved with the built-in optimizer,
            shape inference and constant folding so a rewrite can unlock further simplification and vice
            versa. Use it to plug in a custom graph rewriter, e.g. an ``onnxscript.rewriter`` rule set:
            ``custom_rewriter=lambda m: onnxscript.rewriter.rewrite(m, pattern_rewrite_rules=my_rules)``.
            The callable may return a new ``ModelProto``, mutate and return ``None``, or return ``False``
            to report that it rewrote nothing. Returning ``False`` when no rewrite happened (for example
            when an ``onnxscript.rewriter`` pass reports ``PassResult.modified`` is ``False``) lets onnxsim
            skip copying an unchanged model back through the C++ core on that fixed-point round.
    :param function_rewrite_rules: An optional sequence of ``(pattern, replacement)`` pairs of
            ``onnx.FunctionProto`` describing data-driven rewrite rules matched and applied natively by
            onnxsim's C++ core (no onnxscript dependency), so the *same* rules also work from the C and
            Rust bindings. The pattern's inputs are wildcards binding to graph values, its body is the
            subgraph to match, and its outputs are rewired to the replacement's outputs; a node attribute
            written ``@name`` (a ``ref_attr_name``) is an attribute wildcard bound and substituted into the
            replacement. Build the FunctionProtos with ``onnx.parser.parse_function``. Runs inside the same
            fixed point as ``custom_rewriter`` and is mutually exclusive with it (passing both raises
            ``ValueError``).
    :param check_rtol: Relative tolerance used by the ``check_n`` verification when comparing the
            original and simplified outputs (``numpy.allclose``). The default (1e-4) is strict; raise
            it for very deep models where correct constant-folding/fusion reorders floating-point ops
            enough to accumulate a larger-but-benign difference (e.g. RF-DETR's XLarge segmentation
            variants -- see ``scripts/rfdetr/FAILURE_ANALYSIS.md``). Ignored when ``check_n == 0``.
    :param check_atol: Absolute tolerance counterpart of ``check_rtol``.
    :param input_fill: How to fill the random inputs generated for the ``check_n``
            verification when ``input_data`` is not supplied. One of ``"random"``
            (uniform ``[0, 1)``, the default), ``"ones"``, ``"zeros"`` or ``"arange"``
            (``0, 1, 2, ...`` in row-major order). Ignored when ``check_n == 0`` or
            when ``input_data`` is given.
    :param providers: onnxruntime execution providers used to run the model
            during constant folding, in priority order, for example
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` to fold on an
            NVIDIA GPU (falling back to CPU for ops CUDA cannot run). An entry may
            also be a ``(name, options)`` tuple as accepted by
            ``onnxruntime.InferenceSession``. ``None`` (the default) folds on the
            CPU. Non-CPU providers require onnxruntime to be installed (the CUDA
            provider specifically needs the ``onnxruntime-gpu`` build); a
            requested provider that the installed onnxruntime does not offer
            raises ``ValueError`` instead of silently falling back.
    :param profile: When set, profile every simplification fixed-point function
            (shape inference, the onnx-optimizer passes, constant folding and any
            custom rewriter) -- recording each one's wall-clock and CPU duration
            and the peak resident memory reached while it runs -- and write a
            Chrome Trace Event Format JSON to this path (an empty string uses
            ``onnxsim_profile.json``). Open the file in ``chrome://tracing`` or
            https://ui.perfetto.dev to see the flame graph; a per-function summary
            is also printed to stdout. The trace also records a "NodeCount"
            counter event after every round of each fixed-point loop, tagged
            with which loop produced it -- see ``onnxsim.profile_plot`` (or
            ``--node-reduction-plot``) to turn that into a node-count-per-round
            plot. Implemented in the C++ core via the ``ONNXSIM_PROFILE``
            environment variable, so it works from every binding; setting this
            argument simply sets that variable for the call.
    :param ort_profile: When set, turn on onnxruntime's own built-in session
            profiler for the onnxruntime sessions onnxsim runs while simplifying
            (the constant-folding sessions, plus the correctness-check runs when
            ``check_n`` > 0). This is separate from and complementary to
            ``profile``: where ``profile`` times onnxsim's fixed-point functions,
            ``ort_profile`` records onnxruntime's detailed per-operator execution
            within each session. The value is a file *prefix* (an empty string uses
            ``onnxsim_ort_profile``); onnxruntime writes one
            ``<prefix>_<timestamp>.json`` Chrome trace per session, so a run that
            folds in several batches produces several files. Implemented via the
            ``ONNXSIM_ORT_PROFILE`` environment variable, so it works from every
            binding; setting this argument simply sets that variable for the call.
    :param merge_ort_profile: Merge onnxruntime's own per-operator session
            profiles *into* onnxsim's ``profile`` trace, so the operator-level
            events show up directly under each ``OrtSession`` span in one unified
            flame graph -- rather than as the separate files ``ort_profile``
            writes. Implies profiling: if ``profile`` is not set it defaults to
            ``onnxsim_profile.json``. onnxruntime's traces are captured to a
            temporary directory and removed after merging, so no stray files are
            left behind. Takes precedence over ``ort_profile`` (which requests
            standalone files). Works for every executor, including the Python
            ``PyModelExecutor`` used by ``simplify()``.
    :return: A tuple (simplified model, success(True) or failed(False))
    """
    # Validate the requested execution providers up front. onnxsim's constant
    # folding catches per-op executor failures and leaves the op unfolded, so an
    # unavailable provider raised mid-fold would be swallowed and silently
    # degrade to no folding. Checking here turns a misconfigured provider (e.g.
    # CUDA requested without the onnxruntime-gpu build) into an immediate error.
    backend.validate_providers(providers)

    if dynamic_input_shape:
        print(
            Text(
                "WARNING: The argument `dynamic_input_shape=True` is not needed any more, onnxsim can now support dynamic input shapes natively, please refer to the latest documentation. An error will be raised in the future.",
                style="bold red",
            )
        )
    if input_shapes is not None:
        print(
            Text(
                "WARNING: The argument `input_shapes` is deprecated. Please use `overwrite_input_shapes` and/or `test_input_shapes` instead. An error will be raised in the future.",
                style="bold red",
            )
        )
        overwrite_input_shapes = input_shapes
        test_input_shapes = input_shapes

    # Bridge operator schemas registered in the Python ``onnx`` module (e.g. via
    # ``onnx.defs.register_schema`` for a custom operator) into onnxsim's own
    # separately linked schema registry, so models using custom operators pass
    # validation instead of failing with "No Op registered for ..." (issue #326).
    # Can be turned off via ``import_custom_schemas=False``.
    if import_custom_schemas:
        import_onnx_schemas()

    # Wrap the user-supplied rewriter (if any) so the C++ simplifier can call it
    # between optimization rounds. ``None`` leaves the pipeline unchanged.
    # ``custom_rewriter`` (a Python callable) and ``function_rewrite_rules`` (data
    # matched natively in C++) both drive the single rewriter slot, so at most one
    # may be given.
    if custom_rewriter is not None and function_rewrite_rules:
        raise ValueError(
            "custom_rewriter and function_rewrite_rules are mutually exclusive; "
            "pass only one."
        )
    if function_rewrite_rules:
        rewriter = C.make_function_proto_rewriter(
            [(pattern, replacement) for pattern, replacement in function_rewrite_rules]
        )
    elif custom_rewriter is not None:
        rewriter = _GraphRewriterAdapter(custom_rewriter)
    else:
        rewriter = None

    if not perform_optimization:
        # None means skip all optimizers
        skipped_optimizers = None
    elif skipped_optimizers is None:
        skipped_optimizers = []

    if skip_fuse_bn and skipped_optimizers is not None:
        skipped_optimizers.append("fuse_bn_into_conv")

    # onnxsim's model transforms -- input-shape overwrite, unused-output
    # removal, initializer-from-input folding, and the unhashable-tensor
    # optimizer skip -- all run natively in C++ (Simplify()/SimplifyPath()),
    # so this function never needs to parse the model into a Python object
    # itself: it only marshals options through to the C++ core. The one
    # exception is the deprecated "``None`` key means the model's single
    # input" convenience on ``overwrite_input_shapes``/``test_input_shapes``
    # (and the legacy ``input_shapes`` argument, which sets both) -- resolving
    # it needs the model's own input names, so it is the one case that still
    # requires loading the model up front.
    _shapes_need_model = (input_shapes is not None) or any(
        d and None in d for d in (overwrite_input_shapes, test_input_shapes)
    )

    # Fast path: no ``check_n`` verification (which needs the original model
    # loaded anyway to compare against), no shape dict needing the model to
    # resolve, and no onnxruntime-side session profiling requested (this
    # function's ``ort_profile``/``merge_ort_profile`` handling below sets up
    # ``ONNXSIM_ORT_PROFILE`` and merges the resulting traces into the onnxsim
    # profile after simplification -- machinery the fast path does not
    # replicate). A file path goes straight through the C++ core's native
    # path-based entry point (``C.simplify_path``), skipping the Python-level
    # parse-then-reserialize round trip a large model otherwise pays crossing
    # into C++ -- every initializer's bytes get materialized into a Python
    # ``ModelProto``, serialized back to a byte string, and reparsed on the
    # C++ side, all before simplification even starts. On an 833MB model this
    # dwarfs the actual simplification work (tens of seconds of marshalling
    # for ~5 seconds of real work). An in-memory ``ModelProto`` has no file to
    # hand to C++, so it pays one unavoidable ``SerializeToString``/
    # ``C.simplify`` bytes round trip -- still far cheaper than the removed
    # Python-side transforms, which needed the model fully materialized and
    # walked repeatedly. Falls back to the standard path below on ANY
    # exception -- e.g. a model whose tensors onnxoptimizer's CSE cannot hash
    # (issue #348), which the C++ core already detects and works around
    # itself -- so this is never less correct than before, only sometimes not
    # faster.
    if (
        check_n == 0
        and not _shapes_need_model
        and ort_profile is None
        and not merge_ort_profile
    ):
        _fast_threshold_bytes = parse_size(tensor_size_threshold)
        if _fast_threshold_bytes <= 2**31 - 9999:
            _fast_prev_profile_env = None
            _fast_profile_active = profile is not None
            _fast_profile_path = profile or "onnxsim_profile.json"
            if _fast_profile_active:
                _fast_prev_profile_env = os.environ.get("ONNXSIM_PROFILE")
                os.environ["ONNXSIM_PROFILE"] = _fast_profile_path
            try:
                if isinstance(model, str):
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        fast_out_path = os.path.join(tmpdirname, "opt.onnx")
                        C.simplify_path(
                            _get_model_executor(providers),
                            model,
                            fast_out_path,
                            skipped_optimizers,
                            not skip_constant_folding,
                            not skip_shape_inference,
                            _fast_threshold_bytes,
                            target_opset_version,
                            rewriter,
                            initializers_as_constants,
                            inline_functions,
                            mutable_initializer,
                            overwrite_input_shapes,
                            unused_output,
                        )
                        model_opt = onnx.load(fast_out_path)
                else:
                    model_bytes = model.SerializeToString()
                    if len(model_bytes) >= 2 * 1024 * 1024 * 1024:
                        raise EncodeError("Message larger than 2GiB")
                    model_opt_bytes = C.simplify(
                        _get_model_executor(providers),
                        model_bytes,
                        skipped_optimizers,
                        not skip_constant_folding,
                        not skip_shape_inference,
                        _fast_threshold_bytes,
                        target_opset_version,
                        rewriter,
                        initializers_as_constants,
                        inline_functions,
                        mutable_initializer,
                        overwrite_input_shapes,
                        unused_output,
                    )
                    if len(model_opt_bytes) == 0:
                        raise ValueError("Simplified model larger than 2GB")
                    model_opt = onnx.load_from_string(model_opt_bytes)
                check_ok = model_checking.compare(
                    model_opt,
                    None,
                    0,
                    test_input_shapes,
                    input_data,
                    custom_lib,
                    rtol=check_rtol,
                    atol=check_atol,
                    input_fill=input_fill,
                )
                return model_opt, check_ok
            except Exception:
                pass
            finally:
                if _fast_profile_active:
                    if _fast_prev_profile_env is None:
                        os.environ.pop("ONNXSIM_PROFILE", None)
                    else:
                        os.environ["ONNXSIM_PROFILE"] = _fast_prev_profile_env

    # Slow path: either check_n > 0 (needs the original model to compare
    # against) or a shape dict uses the "None key" convenience (needs the
    # model's own input names to resolve). The model transforms themselves
    # still run natively in C++ (passed through as arguments below); this
    # only loads the model for those two Python-side needs.
    #
    # Track whether we own the in-memory model. When the caller passes a file
    # path we load it here, so the resulting ``ModelProto`` is private to this
    # function and may be mutated freely (e.g. saved as external data without a
    # defensive copy). When the caller passes their own ``ModelProto`` we must
    # not mutate it.
    external_data_dir: Optional[str] = None
    if isinstance(model, str):
        model_owned = True
        external_data_dir = os.path.dirname(os.path.abspath(model))
        model = onnx.load(model, load_external_data=False)
    else:
        model_owned = False
    if overwrite_input_shapes is None:
        overwrite_input_shapes = {}
    overwrite_input_shapes = check_and_update_input_shapes(
        model, overwrite_input_shapes
    )
    test_input_shapes = check_and_update_input_shapes(model, test_input_shapes)

    tensor_size_threshold_bytes = parse_size(tensor_size_threshold)
    if tensor_size_threshold_bytes > 2**31 - 9999:
        raise ValueError("tensor_size_threshold should be less than 2GB")

    # Materialize the external tensor data now that the metadata-only phases are
    # over and the full model is about to be serialized for the C++ simplifier.
    # This is a no-op unless we loaded from a path above (and therefore deferred
    # it); a caller-provided ``ModelProto`` already carries its data inline.
    if external_data_dir is not None:
        onnx.load_external_data_for_model(model, external_data_dir)

    # Merging onnxruntime's profile into onnxsim's trace requires an onnxsim
    # trace to merge into, so it implies profiling.
    if merge_ort_profile and profile is None:
        profile = "onnxsim_profile.json"

    # Enable the C++ core's fixed-point profiler for the duration of this call by
    # setting ``ONNXSIM_PROFILE`` (read inside ``Simplify``), restoring any prior
    # value afterwards so profiling does not leak into later calls in the same
    # process. An empty string falls back to the default trace filename.
    _prev_profile_env = None
    _profile_active = profile is not None
    _profile_path = profile or "onnxsim_profile.json"
    if _profile_active:
        _prev_profile_env = os.environ.get("ONNXSIM_PROFILE")
        os.environ["ONNXSIM_PROFILE"] = _profile_path

    # Turn on onnxruntime's own session profiler by setting ``ONNXSIM_ORT_PROFILE``
    # (read by the executor). It has two modes: ``ort_profile`` writes standalone
    # trace files at the given prefix, while ``merge_ort_profile`` captures them to
    # a temporary directory to be merged into the onnxsim trace below (and takes
    # precedence). Either way, restore any prior value afterwards.
    _ort_merge_dir = (
        tempfile.mkdtemp(prefix="onnxsim_ort_ops_") if merge_ort_profile else None
    )
    if _ort_merge_dir is not None:
        _ort_env_prefix: Optional[str] = os.path.join(_ort_merge_dir, "session")
    elif ort_profile is not None:
        _ort_env_prefix = ort_profile or "onnxsim_ort_profile"
    else:
        _ort_env_prefix = None
    _prev_ort_profile_env = None
    _ort_profile_active = _ort_env_prefix is not None
    if _ort_env_prefix is not None:
        _prev_ort_profile_env = os.environ.get("ONNXSIM_ORT_PROFILE")
        os.environ["ONNXSIM_ORT_PROFILE"] = _ort_env_prefix

    try:
        model_bytes = model.SerializeToString()
        if len(model_bytes) >= 2 * 1024 * 1024 * 1024:
            model_bytes = None
            raise EncodeError("Message larger than 2GiB")
        model_opt_bytes = C.simplify(
            _get_model_executor(providers),
            model_bytes,
            skipped_optimizers,
            not skip_constant_folding,
            not skip_shape_inference,
            tensor_size_threshold_bytes,
            target_opset_version,
            rewriter,
            initializers_as_constants,
            inline_functions,
            mutable_initializer,
            overwrite_input_shapes,
            unused_output,
        )
        # The serialized original (~1x model) is not needed once the C++
        # simplifier has consumed it -- the large-model fallback below
        # re-serializes from ``model`` rather than reusing these bytes. Free it
        # now so it is not held alive while the simplified result is
        # deserialized, which would otherwise inflate peak memory for no reason.
        del model_bytes
        if len(model_opt_bytes) == 0:
            raise ValueError("Simplified model larger than 2GB")
        # With ``check_n == 0`` the original model is never read again:
        # ``model_checking.compare`` only touches it inside the ``range(check_n)``
        # loop, so it merely runs ``onnx.checker.check_model`` on the result.
        # Release the original before deserializing the result to lower peak
        # memory. Only do so when we own the model (a caller-provided
        # ``ModelProto`` is still referenced by the caller, so dropping our
        # reference would not free anything). This must come *after* the
        # ``len(model_opt_bytes) == 0`` check above -- that is the ">2GB
        # optimized model" trigger whose fallback re-simplifies from ``model``.
        if check_n == 0 and model_owned:
            model = None
        model_opt = onnx.load_from_string(model_opt_bytes)
        check_ok = model_checking.compare(
            model_opt,
            model,
            check_n,
            test_input_shapes,
            input_data,
            custom_lib,
            rtol=check_rtol,
            atol=check_atol,
            input_fill=input_fill,
        )
    except (EncodeError, ValueError, onnx.onnx_cpp2py_export.checker.ValidationError):
        if model is None:
            # We released the original model above because ``check_n == 0`` made
            # it unnecessary. The large-model fallback re-simplifies from it, so
            # it cannot run here. This is not the recoverable >2GB case (that is
            # caught by the ``len(model_opt_bytes) == 0`` check before the model
            # is freed), so surface the exception directly instead of crashing
            # on a ``None`` model.
            raise
        print(
            "[bold magenta]Simplified model larger than 2GB. Trying to save as external data...[/bold magenta]"
        )
        # large models try to convert through a temporary file
        with tempfile.TemporaryDirectory() as tmpdirname:
            # ``save_as_external_data=True`` mutates the model in place, moving
            # each initializer's ``raw_data`` out to the external data file. When
            # we own the model this both avoids a full ``deepcopy`` (which would
            # double peak memory for multi-GB models) and frees the in-memory
            # ``raw_data`` as it is streamed to disk. Only copy when the caller
            # owns the ``ModelProto`` and must not see it mutated.
            model_to_save = model if model_owned else copy.deepcopy(model)
            onnx.save(
                model_to_save,
                os.path.join(tmpdirname, "model.onnx"),
                save_as_external_data=True,
            )
            check_ok = C.simplify_path(
                _get_model_executor(providers),
                os.path.join(tmpdirname, "model.onnx"),
                os.path.join(tmpdirname, "opt.onnx"),
                skipped_optimizers,
                not skip_constant_folding,
                not skip_shape_inference,
                tensor_size_threshold_bytes,
                target_opset_version,
                rewriter,
                initializers_as_constants,
                inline_functions,
                mutable_initializer,
                overwrite_input_shapes,
                unused_output,
            )
            check_ok = model_checking.compare(
                os.path.join(tmpdirname, "opt.onnx"),
                os.path.join(tmpdirname, "model.onnx"),
                check_n,
                test_input_shapes,
                input_data,
                custom_lib,
                rtol=check_rtol,
                atol=check_atol,
                input_fill=input_fill,
            )
            model_opt = onnx.load(os.path.join(tmpdirname, "opt.onnx"))
    finally:
        if _profile_active:
            if _prev_profile_env is None:
                os.environ.pop("ONNXSIM_PROFILE", None)
            else:
                os.environ["ONNXSIM_PROFILE"] = _prev_profile_env
        if _ort_profile_active:
            if _prev_ort_profile_env is None:
                os.environ.pop("ONNXSIM_ORT_PROFILE", None)
            else:
                os.environ["ONNXSIM_ORT_PROFILE"] = _prev_ort_profile_env
        # Fold onnxruntime's captured per-operator traces into the onnxsim trace,
        # then drop the temporary files. Best-effort: a merge failure must not
        # sink an otherwise-successful simplification.
        if _ort_merge_dir is not None:
            try:
                profile_merge.merge_ort_traces_into_profile(
                    _profile_path, _ort_merge_dir
                )
            except Exception:  # noqa: BLE001 - profiling is best-effort
                pass
            finally:
                shutil.rmtree(_ort_merge_dir, ignore_errors=True)
    return model_opt, check_ok


class PyModelExecutor(C.ModelExecutor):
    def __init__(self, providers: Optional[Sequence[backend.Provider]] = None):
        super().__init__()
        # onnxruntime execution providers used to run the throwaway sub-models
        # the C++ core builds during constant folding. ``None`` means CPU only.
        self.providers = providers

    def Run(self, model_str: str, inputs_str: List[str]):
        model = onnx.ModelProto()
        model.ParseFromString(model_str)

        def deserialize_tp(tp_str):
            tp = onnx.TensorProto()
            tp.ParseFromString(tp_str)
            return tp

        input_tps = map(deserialize_tp, inputs_str)
        input_arrs = map(onnx.numpy_helper.to_array, input_tps)
        input_names = [x.name for x in model.graph.input]
        inputs = dict(zip(input_names, input_arrs))
        # This executor is only ever invoked by the C++ core's constant-folding
        # ``RunOps`` (see onnxsim.cpp) to run one throwaway fold-group
        # sub-model, never for the full-size correctness check -- so it is
        # always safe (and, given how many of these run per model, usually a
        # meaningful speedup) to skip onnxruntime's per-session thread-pool
        # spin-up.
        outputs = backend.run_model(
            model, inputs, providers=self.providers, single_threaded=True
        )
        # The inference backend may return a non-ndarray for an output (for
        # example onnxruntime yields an empty Python list for an empty sequence
        # output). onnx.numpy_helper.from_array only accepts numpy arrays, so
        # coerce any such value into an empty array instead of crashing with
        # "'list' object has no attribute 'shape'" (GitHub PR #249).
        return [
            onnx.numpy_helper.from_array(x).SerializeToString()
            if isinstance(x, np.ndarray)
            else x
            for x in outputs.values()
        ]


class _GraphRewriterAdapter(C.GraphRewriter):
    """Adapts a Python ``ModelProto -> ModelProto`` callable to the C++
    ``GraphRewriter`` interface.

    The C++ simplifier hands the model to ``Run`` as serialized protobuf bytes
    and expects the rewritten model back as bytes; this adapter deserializes,
    calls the user function, and re-serializes. A function that returns ``None``
    (i.e. mutates the model in place) is supported by falling back to the
    passed-in model.

    A function may instead return ``False`` to report that it rewrote nothing
    -- e.g. an ``onnxscript.rewriter`` pass whose ``PassResult.modified`` is
    ``False`` because no pattern matched. In that case this adapter returns
    empty bytes, the "model unchanged" sentinel, so the C++ core keeps the model
    it already has instead of re-serializing here and parsing an identical copy
    back. Because an onnxscript IR round-trip can reorder bytes even when no rule
    fires, the ``modified`` flag -- not a byte comparison -- is the reliable
    signal that nothing changed.
    """

    def __init__(self, fn: ModelRewriter):
        super().__init__()
        self._fn = fn

    def Run(self, model_str: bytes) -> bytes:
        model = onnx.ModelProto()
        model.ParseFromString(model_str)
        rewritten = self._fn(model)
        if rewritten is False:
            # Nothing was rewritten: return the empty "unchanged" sentinel and
            # skip re-serializing an identical model.
            return b""
        if rewritten is None:
            rewritten = model
        # ``rewritten`` is now a ModelProto: the ``False`` (unchanged) and
        # ``None`` (mutated in place) sentinels were handled above, and the
        # rewriter contract permits no other non-model return.
        assert not isinstance(rewritten, bool)
        return rewritten.SerializeToString()


_model_executor: Optional[PyModelExecutor] = None


def _get_model_executor(
    providers: Optional[Sequence[backend.Provider]] = None,
) -> PyModelExecutor:
    """Return the process-wide Python model executor, creating it on demand.

    The executor is passed explicitly to the C++ ``simplify``/``simplify_path``
    entry points instead of being registered as a global instance. The cached
    executor is reused only when it was built for the same ``providers`` request,
    so a later call asking for a different provider set gets a fresh executor
    rather than silently keeping the previous one.
    """
    global _model_executor
    if _model_executor is None or _model_executor.providers != providers:
        _model_executor = PyModelExecutor(providers)
    return _model_executor


def main():
    # onnxsim runs models through native libraries (onnx shape inference,
    # onnxoptimizer and onnxruntime). A malformed or unusual model can make one
    # of them crash with a segmentation fault instead of a Python exception,
    # which is very hard to diagnose from a bare "Segmentation fault" message
    # (see GitHub issue #426). Enabling faulthandler makes such native crashes
    # dump a Python traceback to stderr, pinpointing the phase that crashed.
    import faulthandler

    if not faulthandler.is_enabled():
        faulthandler.enable()

    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", help="Input ONNX model")
    parser.add_argument("output_model", help="Output ONNX model")
    parser.add_argument(
        "check_n",
        help="Check whether the output is correct with n random inputs",
        nargs="?",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--enable-fuse-bn",
        help="This option is deprecated. Fusing bn into conv is enabled by default.",
        action="store_true",
    )
    parser.add_argument(
        "--skip-fuse-bn", help="Skip fusing batchnorm into conv.", action="store_true"
    )
    parser.add_argument(
        "--skip-optimization",
        help="Skip all ONNX optimizers or some of them. To skip all optimizers, use `onnxsim a.onnx b.onnx --skip-optimization`. To skip some of optimizers, use something like `onnxsim a.onnx b.onnx --skip-optimization fuse_bn_into_conv fuse_pad_into_pool`.",
        type=str,
        nargs="*",
    )
    parser.add_argument(
        "--skip-constant-folding", help="Skip constant folding", action="store_true"
    )
    parser.add_argument(
        "--check-rtol",
        help="Relative tolerance for the check_n output comparison (default 1e-4). "
        "Raise it for very deep models whose correct op reordering accumulates a larger "
        "floating-point difference (e.g. RF-DETR XLarge).",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--check-atol",
        help="Absolute tolerance for the check_n output comparison (default 1e-5).",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--input-shape",
        help="This argument has been renamed to --overwrite-input-shape, please refer to it",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--overwrite-input-shape",
        help='Overwrite the input shape. The format is "input_name:dim0,dim1,...,dimN" or simply "dim0,dim1,...,dimN" when there is only one input, for example, "data:1,3,224,224" or "1,3,224,224". Note: you might want to use some visualization tools like netron to make sure what the input name and dimension ordering (NCHW or NHWC) is.',
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--test-input-shape",
        help='The input shape to generated random inputs for test, useful when the input shape is dynamic. The format is "input_name:dim0,dim1,...,dimN" or simply "dim0,dim1,...,dimN" when there is only one input, for example, "data:1,3,224,224" or "1,3,224,224". Note: you might want to use some visualization tools like netron to make sure what the input name and dimension ordering (NCHW or NHWC) is.',
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--skip-optimizer",
        help="Deprecated. Refer to --skip-optimization",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--skip-shape-inference", help="Skip shape inference", action="store_true"
    )
    parser.add_argument(
        "--enable-onnxruntime-optimization",
        help="Enable ONNX Runtime's ORT_ENABLE_BASIC level optimization.",
        action="store_true",
    )
    parser.add_argument(
        "--dynamic-input-shape",
        help="Deprecated. Not needed any more.",
        action="store_true",
    )
    parser.add_argument(
        "--input-data-path",
        help='input data, The value should be "input_name1:xxx1.bin"  "input_name2:xxx2.bin ...", input data should be a binary data file.',
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--input-fill",
        help="How to fill the random inputs generated for checking (check_n). "
        "'random' (uniform [0, 1), the default), 'ones', 'zeros' or 'arange' "
        "(0, 1, 2, ... in row-major order). Ignored when --input-data-path is given.",
        type=str,
        choices=model_checking.INPUT_FILL_CHOICES,
        default="random",
    )
    parser.add_argument(
        "--custom-lib", help="Deprecated. Not needed any more.", type=str
    )
    parser.add_argument(
        "--include-subgraph",
        help='Experimental feature. Simplify subgraph (e.g. true graph and false graph of "If" operator) instead of only the main graph',
        action="store_true",
    )
    parser.add_argument(
        "--unused-output",
        help="Name of unused outputs that will be eliminated from the model",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--no-large-tensor",
        help="Some ops like Tile and ConstantOfShape can produce large tensor and make the model size much larger. Specifying this flag to skip folding these ops, with loss of some optimization chances. It can be followed with a threshold, for example, --no-large-tensor 1M or --no-large-tensor 100KB. A simple '--no-large-tensor' means '--no-large-tensor 1KB'.",
        type=str,
        const="1KB",
        default=DEFAULT_TENSOR_SIZE_THRESHOLDHOLD,
        nargs="?",
        dest="tensor_size_threshold",
    )
    parser.add_argument(
        "--mutable-initializer",
        help="By ONNX specification, initializers can also serve as inputs. This allows users to overwrite their values during runtime, but some useful optimizations like fuse-conv-and-bn will not be applicable anymore. In almost all cases, having an initializer that is also an input is unintended (usually caused by a out-dated PyTorch). So onnxsim treats all initializers immutable to enabling all optimizations. If it is not wanted, you can specify '--mutable-initializer' to disable this behavior.",
        action="store_true",
    )
    parser.add_argument(
        "--initializers-as-non-constants",
        help="Treat initializers as non-constant tensors. By default onnxsim treats "
        "initializers as constants, so it constant-folds nodes that depend only on them "
        "and lets value-baking optimizers (e.g. fuse_bn_into_conv) fold their weights. "
        "Specify this flag to keep such weights untouched as tunable tensors; Constant "
        "nodes are still folded. This is independent of --mutable-initializer.",
        action="store_true",
    )
    parser.add_argument(
        "--save-as-external-data",
        help="Save parameters as external data. This will make the .onnx file much smaller, but the .onnx file will depend on the external data file (.data).",
        action="store_true",
    )
    parser.add_argument(
        "--skip-schema-import",
        help="By default onnxsim imports operator schemas registered in the Python 'onnx' module (e.g. via onnx.defs.register_schema) into its own registry so models with custom operators pass validation. Specify this flag to disable that import.",
        action="store_true",
    )
    parser.add_argument(
        "--target-opset",
        help="Convert the model to this opset version (of the default ONNX domain) before simplifying, for example '--target-opset 18'. Can be used to upgrade (or downgrade) the model's opset during simplification.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--inline-functions",
        help="Inline the model's local (model-defined) functions into the main graph before "
        "simplifying, so the optimizer, shape inference and constant folding can see through "
        "function calls into a flat op graph. Schema-defined (built-in) functions are left alone.",
        action="store_true",
    )
    parser.add_argument(
        "--providers",
        help="onnxruntime execution providers used to run the model during "
        "constant folding, in priority order, for example '--providers "
        "CUDAExecutionProvider CPUExecutionProvider' to fold on an NVIDIA GPU "
        "(falling back to CPU for ops CUDA cannot run). Defaults to CPU only. "
        "The CUDA provider requires the 'onnxruntime-gpu' build.",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--cuda",
        help="Shortcut for '--providers CUDAExecutionProvider "
        "CPUExecutionProvider'. Requires the 'onnxruntime-gpu' build.",
        action="store_true",
    )
    parser.add_argument(
        "--profile",
        help="Profile each simplification fixed-point function (shape inference, "
        "the onnx-optimizer passes, constant folding and any rewriter): record its "
        "wall-clock and CPU duration and the peak resident memory reached while it "
        "runs, print a per-function summary, and write a Chrome trace to the given "
        "path (defaults to 'onnxsim_profile.json'). Open the trace in "
        "chrome://tracing or https://ui.perfetto.dev to view the flame graph.",
        type=str,
        nargs="?",
        const="onnxsim_profile.json",
        default=None,
    )
    parser.add_argument(
        "--ort-profile",
        help="Turn on onnxruntime's own per-operator session profiler for the "
        "onnxruntime sessions onnxsim runs while simplifying (complementary to "
        "--profile). The value is a file prefix (defaults to "
        "'onnxsim_ort_profile'); onnxruntime writes one '<prefix>_<timestamp>.json' "
        "Chrome trace per session.",
        type=str,
        nargs="?",
        const="onnxsim_ort_profile",
        default=None,
    )
    parser.add_argument(
        "--merge-ort-profile",
        help="Merge onnxruntime's per-operator session profiles into onnxsim's "
        "--profile trace, so the operator-level events appear under each OrtSession "
        "span in one unified flame graph. Implies --profile (defaults to "
        "'onnxsim_profile.json').",
        action="store_true",
    )
    parser.add_argument(
        "--node-reduction-plot",
        help="After simplifying, plot node count per round for each "
        "simplification fixed-point loop (from the --profile trace's "
        "'NodeCount' events) and save it as a PNG at the given path (defaults "
        "to the --profile trace's path with '_node_reduction.png' appended). "
        "Implies --profile (defaults to 'onnxsim_profile.json') if not "
        "already given. Needs matplotlib: 'pip install onnxsim[plot]'.",
        type=str,
        nargs="?",
        const="",
        default=None,
    )
    parser.add_argument(
        "--graph-diff",
        help="Print a node- and value-level diff between the original and "
        "simplified graphs after simplification, matched by output tensor "
        "name: which nodes/values were removed, added, or changed (e.g. a "
        "Conv whose bias input got folded away).",
        action="store_true",
    )
    parser.add_argument(
        "--dynamic-quantize",
        help="After simplifying, dynamically quantize MatMul/Gemm weights to "
        "INT8 (per output channel, symmetric, from their static values) and "
        "quantize activations to uint8 at runtime via DynamicQuantizeLinear. "
        "No calibration data is needed. See onnxsim.quantize_dynamic.",
        action="store_true",
    )
    parser.add_argument(
        "--ternary-quantize",
        help="After simplifying, dynamically quantize MatMul/Gemm nodes "
        "whose constant weight is structurally ternary ({-s, 0, +s} per "
        "output column, e.g. BitNet b1.58) using a lossless ternary weight "
        "encoding, rather than quantize_dynamic's rounded full-range one. "
        "Nodes whose weight is not ternary are left untouched -- combine "
        "with --dynamic-quantize to also quantize those. See "
        "onnxsim.quantize_ternary.",
        action="store_true",
    )
    parser.add_argument(
        "--weight-only-quantize",
        help="After simplifying, weight-only quantize MatMul/Gemm/Conv "
        "weights to INT8 (per output channel, symmetric, from their static "
        "values), inserting a single DequantizeLinear per weight. "
        "Activations are never touched -- no calibration data is needed and "
        "no quantize/dequantize node is added on the activation path. See "
        "onnxsim.quantize_weight_only.",
        action="store_true",
    )
    parser.add_argument(
        "--weight-only-quantize-int4",
        help="After simplifying, block-wise INT4 weight-only quantize "
        "MatMul/Gemm/Conv weights (one symmetric scale per 32-element block "
        "of the flattened reduction dimension, per output channel), "
        "inserting a single DequantizeLinear(block_size=32) per weight. "
        "Activations are never touched. Needs opset >= 21 (INT4 tensors, "
        "DequantizeLinear's block_size). See onnxsim.quantize_weight_only_int4.",
        action="store_true",
    )
    parser.add_argument(
        "--weight-only-quantize-int16",
        help="After simplifying, INT16 weight-only quantize MatMul/Gemm/Conv "
        "weights (one symmetric scale per output channel, INT16's ~8x finer "
        "step than INT8's), inserting a single DequantizeLinear per weight. "
        "Activations are never touched. For channels with a few "
        "extreme-outlier weights, where INT8's coarser step loses the "
        "channel's typical-magnitude weights -- see "
        "onnxsim.estimate_quantization_precision's max_outlier_ratio check. "
        "Needs opset >= 21. See onnxsim.quantize_weight_only_int16.",
        action="store_true",
    )
    parser.add_argument(
        "--weight-only-quantize-int8-block",
        help="After simplifying, block-wise INT8 weight-only quantize "
        "MatMul/Gemm/Conv weights (one symmetric scale per 32-element block "
        "of the flattened reduction dimension, per output channel, values "
        "in [-127, 127]), inserting a single "
        "DequantizeLinear(block_size=32) per weight. Activations are never "
        "touched. Same storage as --weight-only-quantize's INT8, but "
        "resolution closer to --weight-only-quantize-int4's block-wise "
        "scheme. Needs opset >= 21. See "
        "onnxsim.quantize_weight_only_int8_block.",
        action="store_true",
    )
    parser.add_argument(
        "--fp16-quantize",
        help="After simplifying, convert every float32 weight (and, by "
        "default, every internal activation) to float16 -- no calibration "
        "data needed, since float16 is still a floating-point format, not "
        "an integer scheme. The model's own external input/output types "
        "stay float32 (a Cast is inserted at each boundary). See "
        "onnxsim.quantize_fp16.",
        action="store_true",
    )
    parser.add_argument(
        "--bf16-quantize",
        help="After simplifying, convert every float32 weight (and, by "
        "default, every internal activation) to bfloat16 -- no calibration "
        "data needed, since bfloat16 is still a floating-point format, not "
        "an integer scheme. Keeps float32's exponent range, so no clamping "
        "is needed. The model's own external input/output types stay "
        "float32 (a Cast is inserted at each boundary). See "
        "onnxsim.quantize_bf16.",
        action="store_true",
    )
    parser.add_argument(
        "--fp8-quantize",
        help="After simplifying, convert every float32 weight (and, by "
        "default, every internal activation) to an 8-bit floating-point "
        "format -- no calibration data needed, since float8 is still a "
        "floating-point format, not an integer scheme. Out-of-range values "
        "are saturated (clamped) rather than mapped to an infinity/NaN. The "
        "model's own external input/output types stay float32 (a Cast is "
        "inserted at each boundary). Needs opset >= 19. See "
        "onnxsim.quantize_fp8.",
        action="store_true",
    )
    parser.add_argument(
        "--fp8-format",
        help="Target float8 format for --fp8-quantize: 'e4m3' (E4M3FN, the "
        "default -- max finite magnitude 448, typically used for weights) "
        "or 'e5m2' (max finite magnitude 57344, a dynamic range similar to "
        "float16, typically used for gradients).",
        choices=["e4m3", "e5m2"],
        default="e4m3",
    )
    parser.add_argument(
        "--static-quantize",
        help="After simplifying, statically (calibration-based) quantize "
        "MatMul/Gemm/Conv weights and activations to INT8/uint8, inserting a "
        "QuantizeLinear/DequantizeLinear pair (QDQ format) with a fixed "
        "scale/zero-point calibrated from --calibration-dataset if given, "
        "else from random data. See onnxsim.quantize_static.",
        action="store_true",
    )
    parser.add_argument(
        "--static-quantize-int16",
        help="Same as --static-quantize, but a 'W8A16' scheme: weights stay "
        "INT8, while activations are quantized to uint16 instead of uint8 "
        "(an 8x finer calibrated step) -- useful for activations a QDQ "
        "round trip is unusually sensitive to. Needs opset >= 21. See "
        "onnxsim.quantize_static_int16.",
        action="store_true",
    )
    parser.add_argument(
        "--qoperator-quantize",
        help="After simplifying, statically (calibration-based) quantize "
        "MatMul/Gemm weights and activations to INT8/uint8 into the "
        "'QOperator' format (QLinearMatMul) rather than --static-quantize's "
        "QDQ format, with a fixed scale/zero-point calibrated from "
        "--calibration-dataset if given, else from random data. See "
        "onnxsim.quantize_qoperator.",
        action="store_true",
    )
    parser.add_argument(
        "--qoperator-quantize-elementwise",
        help="After simplifying, statically (calibration-based) quantize "
        "elementwise Add/Mul nodes (both inputs non-constant, e.g. a "
        "residual connection) into ONNX Runtime's 'com.microsoft' contrib "
        "ops QLinearAdd/QLinearMul, with a fixed scale/zero-point "
        "calibrated from --calibration-dataset if given, else from random "
        "data. Unlike the other --*-quantize flags, the result needs a "
        "com.microsoft-aware runtime (e.g. ONNX Runtime) to execute -- see "
        "onnxsim.quantize_qoperator_elementwise.",
        action="store_true",
    )
    parser.add_argument(
        "--qoperator-quantize-activation",
        help="After simplifying, statically (calibration-based) quantize "
        "standalone Sigmoid/LeakyRelu nodes into ONNX Runtime's "
        "'com.microsoft' contrib ops QLinearSigmoid/QLinearLeakyRelu, with a "
        "fixed scale/zero-point calibrated from --calibration-dataset if "
        "given, else from random data. Unlike the other --*-quantize flags "
        "(except --qoperator-quantize-elementwise), the result needs a "
        "com.microsoft-aware runtime (e.g. ONNX Runtime) to execute -- see "
        "onnxsim.quantize_qoperator_activation.",
        action="store_true",
    )
    parser.add_argument(
        "--qoperator-quantize-concat",
        help="After simplifying, statically (calibration-based) quantize "
        "Concat nodes (all inputs non-constant) into ONNX Runtime's "
        "'com.microsoft' contrib op QLinearConcat, with a fixed "
        "scale/zero-point calibrated from --calibration-dataset if given, "
        "else from random data. Unlike the other --*-quantize flags (except "
        "--qoperator-quantize-elementwise/-activation), the result needs a "
        "com.microsoft-aware runtime (e.g. ONNX Runtime) to execute -- see "
        "onnxsim.quantize_qoperator_concat.",
        action="store_true",
    )
    parser.add_argument(
        "--qoperator-quantize-softmax",
        help="After simplifying, statically (calibration-based) quantize "
        "standalone Softmax nodes into ONNX Runtime's 'com.microsoft' "
        "contrib op QLinearSoftmax, with a fixed scale/zero-point "
        "calibrated from --calibration-dataset if given, else from random "
        "data. Unlike the other --*-quantize flags (except "
        "--qoperator-quantize-elementwise/-activation/-concat), the result "
        "needs a com.microsoft-aware runtime (e.g. ONNX Runtime) to "
        "execute -- see onnxsim.quantize_qoperator_softmax.",
        action="store_true",
    )
    parser.add_argument(
        "--qoperator-quantize-pool",
        help="After simplifying, statically (calibration-based) quantize "
        "standalone AveragePool/GlobalAveragePool nodes into ONNX Runtime's "
        "'com.microsoft' contrib ops QLinearAveragePool/"
        "QLinearGlobalAveragePool, with a fixed scale/zero-point calibrated "
        "from --calibration-dataset if given, else from random data. "
        "Unlike the other --*-quantize flags (except "
        "--qoperator-quantize-elementwise/-activation/-concat/-softmax), "
        "the result needs a com.microsoft-aware runtime (e.g. ONNX "
        "Runtime) to execute -- see onnxsim.quantize_qoperator_pool.",
        action="store_true",
    )
    parser.add_argument(
        "--qoperator-quantize-where",
        help="After simplifying, statically (calibration-based) quantize "
        "Where nodes (both data operands non-constant) into ONNX Runtime's "
        "'com.microsoft' contrib op QLinearWhere, with a fixed "
        "scale/zero-point calibrated from --calibration-dataset if given, "
        "else from random data. Unlike the other --*-quantize flags "
        "(except --qoperator-quantize-elementwise/-activation/-concat/"
        "-softmax/-pool), the result needs a com.microsoft-aware runtime "
        "(e.g. ONNX Runtime) to execute -- see "
        "onnxsim.quantize_qoperator_where.",
        action="store_true",
    )
    parser.add_argument(
        "--qoperator-quantize-gemm",
        help="After simplifying, statically (calibration-based) quantize "
        "Gemm nodes (constant 2-D weight; any transA/transB/alpha, unlike "
        "--qoperator-quantize's vanilla-only QLinearMatMul path) into ONNX "
        "Runtime's 'com.microsoft' contrib op QGemm, with a fixed "
        "scale/zero-point calibrated from --calibration-dataset if given, "
        "else from random data. Unlike the other --*-quantize flags "
        "(except --qoperator-quantize-elementwise/-activation/-concat/"
        "-softmax/-pool/-where), the result needs a com.microsoft-aware "
        "runtime (e.g. ONNX Runtime) to execute -- see "
        "onnxsim.quantize_qoperator_gemm.",
        action="store_true",
    )
    parser.add_argument(
        "--calibration-dataset",
        help="Hugging Face Hub dataset id (e.g. 'mnist') to pull "
        "--calibration-samples real examples from for --static-quantize's, "
        "--static-quantize-int16's, --qoperator-quantize's, "
        "--qoperator-quantize-elementwise's, --qoperator-quantize-activation's, "
        "--qoperator-quantize-concat's, --qoperator-quantize-softmax's, "
        "--qoperator-quantize-pool's, --qoperator-quantize-where's, or "
        "--qoperator-quantize-gemm's calibration, instead of random data. "
        "See onnxsim.load_huggingface_calibration_data (needs the optional "
        "'datasets' package: pip install datasets).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--calibration-samples",
        help="Number of calibration batches/examples for --static-quantize, "
        "--qoperator-quantize, --qoperator-quantize-elementwise, "
        "--qoperator-quantize-activation, --qoperator-quantize-concat, "
        "--qoperator-quantize-softmax, --qoperator-quantize-pool, "
        "--qoperator-quantize-where, or --qoperator-quantize-gemm "
        "(default: 8).",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--calibration-method",
        help="Calibration range method for --static-quantize, "
        "--qoperator-quantize, --qoperator-quantize-elementwise, "
        "--qoperator-quantize-activation, --qoperator-quantize-concat, "
        "--qoperator-quantize-softmax, --qoperator-quantize-pool, "
        "--qoperator-quantize-where, or --qoperator-quantize-gemm: "
        "'minmax' "
        "(default) uses each tensor's observed min/max directly; 'entropy' "
        "instead searches for the clip threshold minimizing KL divergence "
        "between the observed and simulated-quantized distributions "
        "(TensorRT-style entropy calibration), which can give a tighter "
        "range for heavy-tailed activations but wants more calibration data "
        "than minmax to build a meaningful histogram. See "
        "onnxsim.calibrate.",
        type=str,
        choices=["minmax", "entropy"],
        default="minmax",
    )
    parser.add_argument(
        "-v", "--version", action="version", version="onnxsim " + version.version
    )

    class ListOptimizers(argparse.Action):
        def __call__(self, parser, ns, v, option_string=None):
            for p in C._list_optimizers():
                print(p)
            parser.exit()

    parser.add_argument(
        "--list-default-optimizers",
        help="List default optimizer pass names",
        nargs=0,
        action=ListOptimizers,
    )

    args = parser.parse_args()

    if args.enable_fuse_bn:
        print(
            Text(
                'WARNING: "--enable-fuse-bn" is not needed any more, because fuse bn is enabled by default. "--enable-fuse-bn" flag is ignored now and will raise an error in the future.',
                style="bold red",
            )
        )
    if args.dynamic_input_shape:
        print(
            Text(
                'WARNING: "--dynamic-input-shape" is not needed any more, onnxsim v0.4 now handles dynamic input shapes automatically. "--dynamic-input-shape" flag is ignored now and will raise an error in the future.',
                style="bold red",
            )
        )
    assert not (args.input_shape is not None and args.overwrite_input_shape is not None)
    if args.input_shape:
        print(
            Text(
                'WARNING: "--input-shape" is renamed to "--overwrite-input-shape". Please use it instead.',
                style="bold red",
            )
        )
        args.overwrite_input_shape = args.input_shape
    if args.include_subgraph:
        print(
            Text(
                "WARNING: subgraph optimization is not supported in v0.4 for now.",
                style="bold red",
            )
        )
    assert not (args.skip_optimizer is not None and args.skip_optimization is not None)
    if args.skip_optimizer:
        print(
            Text(
                'WARNING: "--skip-optimizer" is renamed to "--skip-optimization". Please use it instead.',
                style="bold red",
            )
        )
        args.skip_optimization = args.skip_optimizer
    if args.skip_optimization is None:
        # user doesn't specify --skip-optimization
        args.skip_optimization = []
    elif len(args.skip_optimization) == 0:
        # user specify --skip-optimization without any certain optimizer name
        # set it to None means skip all optimizations
        args.skip_optimization = None
    if args.skip_fuse_bn and args.skip_optimization is not None:
        args.skip_optimization.append("fuse_bn_into_conv")

    perform_optimization = False if args.skip_optimization is None else True

    # Resolve the execution providers for constant folding. ``--cuda`` is a
    # shortcut for the common GPU-then-CPU order; it and an explicit
    # ``--providers`` cannot both be given.
    if args.cuda and args.providers is not None:
        raise RuntimeError(
            "--cuda and --providers are mutually exclusive; --cuda is a shortcut "
            "for '--providers CUDAExecutionProvider CPUExecutionProvider'."
        )
    if args.cuda:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = args.providers

    def parse_shapes(shapes_arg):
        shapes = {}
        if shapes_arg is not None:
            for x in shapes_arg:
                if ":" not in x:
                    shapes[None] = list(map(int, x.split(",")))
                else:
                    pieces = x.split(":")
                    # for the input name like input:0
                    name, shape = (
                        ":".join(pieces[:-1]),
                        list(map(int, pieces[-1].split(","))),
                    )
                    shapes.update({name: shape})
        return shapes

    test_input_shapes = parse_shapes(args.test_input_shape)
    overwrite_input_shapes = parse_shapes(args.overwrite_input_shape)

    if args.enable_onnxruntime_optimization:
        if not backend.has_onnxruntime():
            raise RuntimeError(
                "--enable-onnxruntime-optimization requires onnxruntime, "
                "please install it by `pip install onnxruntime`."
            )
        import onnxruntime as rt

        tmp_file = tempfile.NamedTemporaryFile()
        sess_options = rt.SessionOptions()
        # Set graph optimization level
        sess_options.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        # To enable model serialization after graph optimization
        sess_options.optimized_model_filepath = tmp_file.name
        _ = rt.InferenceSession(
            args.input_model, sess_options, providers=["CPUExecutionProvider"]
        )

        # ``tmp_file`` stays referenced (and thus on disk) until ``main`` returns,
        # so ``simplify`` can load it below.
        model_path = tmp_file.name
    else:
        model_path = args.input_model

    # Load only the graph structure here, deferring the (potentially multi-GB)
    # external tensor data. The CLI needs this model just for the pre-flight
    # warnings below and the size/op diff printed at the end -- none of which read
    # raw tensor bytes: op counts come from graph structure and the reported size
    # is computed from external-data metadata (see ``model_info``). ``simplify``
    # is handed the *path* (not this ModelProto) so it owns its copy and performs
    # its own deferred load, keeping the weights out of memory here entirely.
    model = onnx.load(model_path, load_external_data=False)

    if args.tensor_size_threshold == DEFAULT_TENSOR_SIZE_THRESHOLDHOLD:
        for node in model.graph.node:
            if node.op_type in ["Tile", "ConstantOfShape", "Expand"]:
                print(
                    Text(
                        'Your model contains "Tile" ops or/and "ConstantOfShape" ops or/and "Expand" ops. Folding these ops can make the simplified model much larger. If it is not expected, please specify "--no-large-tensor" (which will lose some optimization chances)',
                        style="bold magenta",
                    )
                )
                break

    if not args.mutable_initializer:
        initializer_names = set([x.name for x in model.graph.initializer])
        input_names = set([x.name for x in model.graph.input])
        if len(initializer_names.intersection(input_names)) > 0:
            print(
                Text(
                    'Your model contains initializers that are also inputs. This is usually caused by an out-dated PyTorch. onnxsim treats all initializers immutable to enabling all optimizations. If it is not wanted, please specify "--mutable-initializer" to disable this behavior.',
                    style="bold magenta",
                )
            )

    # Plotting the node-reduction curve needs the "NodeCount" events a
    # profiled run writes, so turn --profile on if the user only asked for
    # the plot.
    if args.node_reduction_plot is not None and args.profile is None:
        args.profile = "onnxsim_profile.json"

    input_tensors = None
    if args.input_data_path is not None:
        input_tensors = {}
        for x in args.input_data_path:
            pieces = x.split(":")
            name, data = ":".join(pieces[:-1]), pieces[-1]
            input_tensors.update({name: np.load(data)})

    print("Simplifying...")

    model_opt, check_ok = simplify(
        model_path,
        args.check_n,
        perform_optimization,
        False,
        overwrite_input_shapes,
        test_input_shapes,
        args.skip_optimization,
        args.skip_constant_folding,
        args.skip_shape_inference,
        input_tensors,
        False,
        args.custom_lib,
        args.include_subgraph,
        args.unused_output,
        args.tensor_size_threshold,
        args.mutable_initializer,
        initializers_as_constants=not args.initializers_as_non_constants,
        inline_functions=args.inline_functions,
        import_custom_schemas=not args.skip_schema_import,
        target_opset_version=args.target_opset,
        check_rtol=args.check_rtol,
        check_atol=args.check_atol,
        input_fill=args.input_fill,
        providers=providers,
        profile=args.profile,
        ort_profile=args.ort_profile,
        merge_ort_profile=args.merge_ort_profile,
    )

    if args.node_reduction_plot is not None:
        from . import profile_plot

        plot_out = args.node_reduction_plot or None
        try:
            plot_path = profile_plot.plot_node_reduction(args.profile, plot_out)
        except RuntimeError as e:
            print(Text(f"WARNING: --node-reduction-plot failed: {e}", style="bold red"))
        else:
            print(f"Node reduction plot written to {plot_path}")

    if args.dynamic_quantize:
        print("Dynamically quantizing MatMul/Gemm weights to INT8...")
        model_opt = quantize_dynamic(model_opt)

    if args.ternary_quantize:
        print("Dynamically quantizing structurally-ternary MatMul/Gemm weights...")
        model_opt = quantize_ternary(model_opt)

    if args.weight_only_quantize:
        print("Weight-only quantizing MatMul/Gemm/Conv weights to INT8...")
        model_opt = quantize_weight_only(model_opt)

    if args.weight_only_quantize_int4:
        print("Block-wise INT4 weight-only quantizing MatMul/Gemm/Conv weights...")
        model_opt = quantize_weight_only_int4(model_opt)

    if args.weight_only_quantize_int16:
        print("INT16 weight-only quantizing MatMul/Gemm/Conv weights...")
        model_opt = quantize_weight_only_int16(model_opt)

    if args.weight_only_quantize_int8_block:
        print("Block-wise INT8 weight-only quantizing MatMul/Gemm/Conv weights...")
        model_opt = quantize_weight_only_int8_block(model_opt)

    if args.fp16_quantize:
        print("Converting float32 weights/activations to float16...")
        model_opt = quantize_fp16(model_opt)

    if args.bf16_quantize:
        print("Converting float32 weights/activations to bfloat16...")
        model_opt = quantize_bf16(model_opt)

    if args.fp8_quantize:
        print(
            f"Converting float32 weights/activations to float8 ({args.fp8_format})..."
        )
        model_opt = quantize_fp8(model_opt, format=args.fp8_format)

    if args.static_quantize:
        from . import calibration

        if args.calibration_dataset:
            print(
                f'Calibrating from Hugging Face dataset "{args.calibration_dataset}"...'
            )
            calibration_data = calibration.load_huggingface_calibration_data(
                args.calibration_dataset,
                model_opt,
                num_samples=args.calibration_samples,
            )
        else:
            print("Calibrating from random data...")
            calibration_data = calibration.generate_random_calibration_data(
                model_opt, num_samples=args.calibration_samples
            )
        print("Statically quantizing MatMul/Gemm/Conv weights and activations...")
        model_opt = calibration.quantize_static(
            model_opt, calibration_data=calibration_data, method=args.calibration_method
        )

    if args.static_quantize_int16:
        from . import calibration

        if args.calibration_dataset:
            print(
                f'Calibrating from Hugging Face dataset "{args.calibration_dataset}"...'
            )
            calibration_data = calibration.load_huggingface_calibration_data(
                args.calibration_dataset,
                model_opt,
                num_samples=args.calibration_samples,
            )
        else:
            print("Calibrating from random data...")
            calibration_data = calibration.generate_random_calibration_data(
                model_opt, num_samples=args.calibration_samples
            )
        print(
            "Statically quantizing MatMul/Gemm/Conv weights (INT8) and "
            "activations (uint16)..."
        )
        model_opt = calibration.quantize_static_int16(
            model_opt, calibration_data=calibration_data, method=args.calibration_method
        )

    if args.qoperator_quantize:
        from . import calibration

        if args.calibration_dataset:
            print(
                f'Calibrating from Hugging Face dataset "{args.calibration_dataset}"...'
            )
            calibration_data = calibration.load_huggingface_calibration_data(
                args.calibration_dataset,
                model_opt,
                num_samples=args.calibration_samples,
            )
        else:
            print("Calibrating from random data...")
            calibration_data = calibration.generate_random_calibration_data(
                model_opt, num_samples=args.calibration_samples
            )
        print(
            "Statically quantizing MatMul/Gemm weights and activations "
            "(QOperator format)..."
        )
        model_opt = calibration.quantize_qoperator(
            model_opt, calibration_data=calibration_data, method=args.calibration_method
        )

    if args.qoperator_quantize_elementwise:
        from . import calibration

        if args.calibration_dataset:
            print(
                f'Calibrating from Hugging Face dataset "{args.calibration_dataset}"...'
            )
            calibration_data = calibration.load_huggingface_calibration_data(
                args.calibration_dataset,
                model_opt,
                num_samples=args.calibration_samples,
            )
        else:
            print("Calibrating from random data...")
            calibration_data = calibration.generate_random_calibration_data(
                model_opt, num_samples=args.calibration_samples
            )
        print(
            "Statically quantizing elementwise Add/Mul nodes (QOperator "
            "format, com.microsoft contrib ops)..."
        )
        model_opt = calibration.quantize_qoperator_elementwise(
            model_opt, calibration_data=calibration_data, method=args.calibration_method
        )

    if args.qoperator_quantize_activation:
        from . import calibration

        if args.calibration_dataset:
            print(
                f'Calibrating from Hugging Face dataset "{args.calibration_dataset}"...'
            )
            calibration_data = calibration.load_huggingface_calibration_data(
                args.calibration_dataset,
                model_opt,
                num_samples=args.calibration_samples,
            )
        else:
            print("Calibrating from random data...")
            calibration_data = calibration.generate_random_calibration_data(
                model_opt, num_samples=args.calibration_samples
            )
        print(
            "Statically quantizing Sigmoid/LeakyRelu nodes (QOperator "
            "format, com.microsoft contrib ops)..."
        )
        model_opt = calibration.quantize_qoperator_activation(
            model_opt, calibration_data=calibration_data, method=args.calibration_method
        )

    if args.qoperator_quantize_concat:
        from . import calibration

        if args.calibration_dataset:
            print(
                f'Calibrating from Hugging Face dataset "{args.calibration_dataset}"...'
            )
            calibration_data = calibration.load_huggingface_calibration_data(
                args.calibration_dataset,
                model_opt,
                num_samples=args.calibration_samples,
            )
        else:
            print("Calibrating from random data...")
            calibration_data = calibration.generate_random_calibration_data(
                model_opt, num_samples=args.calibration_samples
            )
        print(
            "Statically quantizing Concat nodes (QOperator format, "
            "com.microsoft contrib ops)..."
        )
        model_opt = calibration.quantize_qoperator_concat(
            model_opt, calibration_data=calibration_data, method=args.calibration_method
        )

    if args.qoperator_quantize_softmax:
        from . import calibration

        if args.calibration_dataset:
            print(
                f'Calibrating from Hugging Face dataset "{args.calibration_dataset}"...'
            )
            calibration_data = calibration.load_huggingface_calibration_data(
                args.calibration_dataset,
                model_opt,
                num_samples=args.calibration_samples,
            )
        else:
            print("Calibrating from random data...")
            calibration_data = calibration.generate_random_calibration_data(
                model_opt, num_samples=args.calibration_samples
            )
        print(
            "Statically quantizing Softmax nodes (QOperator format, "
            "com.microsoft contrib ops)..."
        )
        model_opt = calibration.quantize_qoperator_softmax(
            model_opt, calibration_data=calibration_data, method=args.calibration_method
        )

    if args.qoperator_quantize_pool:
        from . import calibration

        if args.calibration_dataset:
            print(
                f'Calibrating from Hugging Face dataset "{args.calibration_dataset}"...'
            )
            calibration_data = calibration.load_huggingface_calibration_data(
                args.calibration_dataset,
                model_opt,
                num_samples=args.calibration_samples,
            )
        else:
            print("Calibrating from random data...")
            calibration_data = calibration.generate_random_calibration_data(
                model_opt, num_samples=args.calibration_samples
            )
        print(
            "Statically quantizing AveragePool/GlobalAveragePool nodes "
            "(QOperator format, com.microsoft contrib ops)..."
        )
        model_opt = calibration.quantize_qoperator_pool(
            model_opt, calibration_data=calibration_data, method=args.calibration_method
        )

    if args.qoperator_quantize_where:
        from . import calibration

        if args.calibration_dataset:
            print(
                f'Calibrating from Hugging Face dataset "{args.calibration_dataset}"...'
            )
            calibration_data = calibration.load_huggingface_calibration_data(
                args.calibration_dataset,
                model_opt,
                num_samples=args.calibration_samples,
            )
        else:
            print("Calibrating from random data...")
            calibration_data = calibration.generate_random_calibration_data(
                model_opt, num_samples=args.calibration_samples
            )
        print(
            "Statically quantizing Where nodes (QOperator format, "
            "com.microsoft contrib ops)..."
        )
        model_opt = calibration.quantize_qoperator_where(
            model_opt, calibration_data=calibration_data, method=args.calibration_method
        )

    if args.qoperator_quantize_gemm:
        from . import calibration

        if args.calibration_dataset:
            print(
                f'Calibrating from Hugging Face dataset "{args.calibration_dataset}"...'
            )
            calibration_data = calibration.load_huggingface_calibration_data(
                args.calibration_dataset,
                model_opt,
                num_samples=args.calibration_samples,
            )
        else:
            print("Calibrating from random data...")
            calibration_data = calibration.generate_random_calibration_data(
                model_opt, num_samples=args.calibration_samples
            )
        print(
            "Statically quantizing Gemm nodes (QOperator format, "
            "com.microsoft QGemm)..."
        )
        model_opt = calibration.quantize_qoperator_gemm(
            model_opt, calibration_data=calibration_data, method=args.calibration_method
        )

    try:
        if not args.save_as_external_data:
            onnx.save(model_opt, args.output_model)
        else:
            raise ValueError("save_as_external_data")
    except ValueError:
        # large models (>2GB) which onnx.save doesn't support,
        # or explicitly specified --save-as-external-data
        external_data_path = os.path.basename(args.output_model) + ".data"
        if os.path.exists(external_data_path):
            os.remove(external_data_path)
        onnx.save(
            copy.deepcopy(model_opt),
            args.output_model,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path,
        )

    if check_ok:
        print("Finish! Here is the difference:")
        model_info.print_simplifying_info(model, model_opt)
        if args.graph_diff:
            model_info.print_graph_diff(model, model_opt)
    else:
        print(
            'Check failed. Please be careful to use the simplified model, or try specifying "--skip-fuse-bn" or "--skip-optimization" (run "onnxsim -h" for details).'
        )
        print("Here is the difference after simplification:")
        model_info.print_simplifying_info(model, model_opt)
        if args.graph_diff:
            model_info.print_graph_diff(model, model_opt)
        sys.exit(1)
