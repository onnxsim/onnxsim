import argparse
import contextvars
import copy
import functools
import importlib.util
import inspect
import os
import re
import shutil
import sys
import tempfile
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import onnx  # type: ignore
import onnx.checker  # type: ignore
import onnx.helper  # type: ignore
import onnx.numpy_helper  # type: ignore
import onnx.shape_inference  # type: ignore
from google.protobuf.message import EncodeError

import onnxsim.onnxsim_cpp2py_export as C

from . import backend, model_checking, model_info, profile_merge, version
from ._rich_compat import Text, print
from .calibration import Tensors, generate_random_calibration_data
from .pruning import EmbeddingPruningResult, _ImportanceNorm
from .rptq import RptqLayerInfo

if TYPE_CHECKING:
    # onnxsim.lora itself imports from onnxsim.nf4, which imports back from
    # this module (quantize_weight_only_nf4_cpp) -- a real circular import
    # at module-load time, unlike RptqLayerInfo's own plain top-level import
    # just above (onnxsim.rptq never imports the onnxsim package itself).
    # inject_lora_cpp below imports onnxsim.lora lazily, inside its own
    # function body, for the same reason onnxsim.gptq's own apply_gptq
    # imports apply_gptq_cpp lazily; this TYPE_CHECKING-only import exists
    # purely so this file's own LoraAdapter annotation (written as a string
    # below) resolves for a type checker.
    from .lora import LoraAdapter

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

# Above this serialized size, ``simplify(..., output_path=...)`` and the CLI
# save the model as external data by default, even without
# ``--save-as-external-data`` -- see the ``external_data_threshold`` param
# below and ``main()``'s ``--external-data-threshold`` flag.
DEFAULT_EXTERNAL_DATA_THRESHOLD = "100MB"


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


def _formal_parameter_from_tuple(t: Tuple):
    """Rebuild an ``onnx.defs.OpSchema.FormalParameter`` from ``C._get_all_schemas``'s tuple."""
    name, description, type_str, option, is_homogeneous, min_arity = t
    return onnx.defs.OpSchema.FormalParameter(
        name,
        type_str,
        description,
        param_option=onnx.defs.OpSchema.FormalParameterOption(option),
        is_homogeneous=is_homogeneous,
        min_arity=min_arity,
    )


def _attribute_from_tuple(t: Tuple):
    """Rebuild an ``onnx.defs.OpSchema.Attribute`` from ``C._get_all_schemas``'s tuple."""
    name, description, attr_type, required, default_value = t
    if default_value.type != onnx.AttributeProto.UNDEFINED:
        return onnx.defs.OpSchema.Attribute(name, default_value, description)
    return onnx.defs.OpSchema.Attribute(
        name, onnx.defs.OpSchema.AttrType(attr_type), description, required=required
    )


def _register_schema_in_onnx(schema: Tuple) -> None:
    """Register a single onnxsim-registry schema (from ``C._get_all_schemas``) into ``onnx``."""
    (
        name,
        domain,
        since_version,
        doc,
        inputs,
        outputs,
        attributes,
        type_constraints,
        _has_inference_function,
    ) = schema

    onnx.defs.register_schema(
        onnx.defs.OpSchema(
            name,
            domain,
            since_version,
            doc,
            inputs=[_formal_parameter_from_tuple(p) for p in inputs],
            outputs=[_formal_parameter_from_tuple(p) for p in outputs],
            attributes=[_attribute_from_tuple(a) for a in attributes],
            type_constraints=[tuple(tc) for tc in type_constraints],
        )
    )


def export_onnx_schemas() -> int:
    """Copy operator schemas from onnxsim's internal registry into the Python ``onnx`` module.

    This is the counterpart to :func:`import_onnx_schemas`. onnxsim links its
    own copy of the ONNX C++ library (see that function's docstring), so its
    operator schema registry also holds schemas the ``onnx`` Python module's
    registry does not: onnxsim's built-in ONNX Runtime contrib-op schemas
    (``com.microsoft`` ops such as ``QLinearAdd`` or ``Attention``, see
    ``contrib_schemas.cpp``), and any custom schema previously bridged in via
    :func:`import_onnx_schemas` or ``onnxsim.onnxsim_cpp2py_export._register_schema``.

    Exporting these into ``onnx.defs`` lets other tools built on the ``onnx``
    Python module -- ``onnx.checker.check_model``, ``onnx.shape_inference``,
    third-party ``onnx.defs``-based tooling -- recognize onnxsim's operators
    without depending on onnxsim itself.

    It is idempotent and safe to call repeatedly: an operator ``onnx``
    already knows -- including one exported by a previous call -- is skipped.
    A schema's type/shape inference function is native code inside onnxsim
    and is not transferred; exported schemas carry none.

    :return: the number of schemas exported.
    """
    import onnx.defs

    try:
        schemas = C._get_all_schemas()
    except Exception:
        return 0

    exported = 0
    # Cache onnx's knowledge per (op, domain) so the native check runs once
    # per operator rather than once per registered version.
    onnx_knows: Dict[Tuple[str, str], bool] = {}
    for schema in schemas:
        try:
            key = (schema[0], schema[1])
            known = onnx_knows.get(key)
            if known is None:
                known = onnx.defs.has(key[0], domain=key[1])
                onnx_knows[key] = known
            if known:
                continue
            _register_schema_in_onnx(schema)
            exported += 1
        except Exception:
            # A single unusual schema must never break the export: skip it
            # and keep exporting the rest.
            continue
    return exported


def _extract_tensor_to_dict(
    tensor: onnx.TensorProto, key: str, tensor_bytes: Dict[str, bytes]
) -> None:
    if (
        tensor.data_location == onnx.TensorProto.EXTERNAL
        or tensor.data_type == onnx.TensorProto.STRING
        or not tensor.HasField("raw_data")
    ):
        return
    tensor_bytes[key] = tensor.raw_data
    tensor.ClearField("raw_data")


def _extract_attr_tensors_to_dict(
    attr: onnx.AttributeProto, path: str, tensor_bytes: Dict[str, bytes]
) -> None:
    if attr.HasField("t"):
        _extract_tensor_to_dict(attr.t, attr.t.name or f"{path}/t", tensor_bytes)
    for i, t in enumerate(attr.tensors):
        _extract_tensor_to_dict(t, t.name or f"{path}/tensors{i}", tensor_bytes)
    if attr.HasField("g"):
        _extract_graph_tensors_to_dict(attr.g, tensor_bytes)
    for g in attr.graphs:
        _extract_graph_tensors_to_dict(g, tensor_bytes)


def _extract_graph_tensors_to_dict(
    graph: onnx.GraphProto, tensor_bytes: Dict[str, bytes]
) -> None:
    """Reverse of :func:`_hydrate_graph_tensors_from_pool`: pulls every
    eligible tensor's ``raw_data`` out of ``graph`` (initializers and node
    attribute tensors, recursing into subgraphs) into ``tensor_bytes`` --
    keyed the same way tensor_pool_bridge.h's ForEachTensor derives pool
    keys -- and clears it from the tensor in place, so the ``ModelProto``
    this graph belongs to can be serialized across the FFI boundary without
    paying to encode/decode the (potentially huge) tensor bytes a second
    time; ``AdoptAllWithPlaceholderOffsets``'s C++ side expects exactly this
    stripped state when given an ``external_tensor_bytes`` map.
    """
    for i, init in enumerate(graph.initializer):
        _extract_tensor_to_dict(init, init.name or f"initializer{i}", tensor_bytes)
    for ni, node in enumerate(graph.node):
        node_path = node.name or f"node{ni}"
        for ai, attr in enumerate(node.attribute):
            _extract_attr_tensors_to_dict(attr, f"{node_path}/attr{ai}", tensor_bytes)


def _restore_tensor_raw_data(
    tensor: onnx.TensorProto, key: str, tensor_bytes: Dict[str, bytes]
) -> None:
    if key in tensor_bytes:
        tensor.raw_data = tensor_bytes[key]


def _restore_attr_tensors_raw_data(
    attr: onnx.AttributeProto, path: str, tensor_bytes: Dict[str, bytes]
) -> None:
    if attr.HasField("t"):
        _restore_tensor_raw_data(attr.t, attr.t.name or f"{path}/t", tensor_bytes)
    for i, t in enumerate(attr.tensors):
        _restore_tensor_raw_data(t, t.name or f"{path}/tensors{i}", tensor_bytes)
    if attr.HasField("g"):
        _restore_graph_tensors_raw_data(attr.g, tensor_bytes)
    for g in attr.graphs:
        _restore_graph_tensors_raw_data(g, tensor_bytes)


def _restore_graph_tensors_raw_data(
    graph: onnx.GraphProto, tensor_bytes: Dict[str, bytes]
) -> None:
    """Undoes :func:`_extract_graph_tensors_to_dict`: puts each extracted
    tensor's ``raw_data`` back so the caller's ``model`` is left exactly as
    it was, once the export call that needed it stripped out has returned
    (or raised). This restore is a plain in-memory field assignment --
    no serialization, no FFI crossing -- so it doesn't reintroduce the
    protobuf round-trip cost the extraction was written to avoid.
    """
    for i, init in enumerate(graph.initializer):
        _restore_tensor_raw_data(init, init.name or f"initializer{i}", tensor_bytes)
    for ni, node in enumerate(graph.node):
        node_path = node.name or f"node{ni}"
        for ai, attr in enumerate(node.attribute):
            _restore_attr_tensors_raw_data(attr, f"{node_path}/attr{ai}", tensor_bytes)


def export_safetensors(model: onnx.ModelProto, out_path: str) -> None:
    """Export ``model`` to a standalone safetensors archive at ``out_path``.

    Every initializer's bytes move into the archive with real, byte-accurate
    offsets -- openable by the ``safetensors`` Python package / HF tooling with
    no onnxsim involved -- and the graph itself is embedded alongside them, so
    ``out_path`` alone is both the model's weights and its graph. Reload it
    with :func:`import_safetensors`.

    ``model`` is left unchanged once this returns (including on error): each
    tensor's ``raw_data`` is transiently cleared before crossing into C++ (so
    the accompanying model-structure serialize doesn't also re-encode the
    -- potentially huge -- tensor bytes a second time) and restored
    afterwards, a plain in-memory assignment with no serialization or FFI
    cost of its own.
    """
    tensor_bytes: Dict[str, bytes] = {}
    _extract_graph_tensors_to_dict(model.graph, tensor_bytes)
    try:
        C.export_safetensors(model.SerializeToString(), tensor_bytes, out_path)
    finally:
        _restore_graph_tensors_raw_data(model.graph, tensor_bytes)


def import_safetensors(in_path: str) -> onnx.ModelProto:
    """Import a standalone safetensors archive back into an ``onnx.ModelProto``.

    ``in_path`` must be an archive produced by :func:`export_safetensors` (or
    any other tool following the same self-describing-archive convention: an
    embedded ``model.onnx`` entry alongside the tensors). Raises
    ``RuntimeError`` if the archive has no embedded onnxsim model, e.g. a
    plain weights-only safetensors file with no graph to import.
    """
    model_bytes, pool = C.import_safetensors(in_path)
    model = onnx.ModelProto()
    model.ParseFromString(model_bytes)
    _hydrate_graph_tensors_from_pool(model.graph, pool)
    return model


def _hydrate_tensor_from_pool(
    tensor: onnx.TensorProto, key: str, pool: C.TensorPool
) -> None:
    if tensor.data_location != onnx.TensorProto.EXTERNAL or key not in pool:
        return
    tensor.raw_data = pool.bytes(key)
    tensor.data_location = onnx.TensorProto.DEFAULT
    tensor.ClearField("external_data")


def _hydrate_attr_tensors_from_pool(
    attr: onnx.AttributeProto, path: str, pool: C.TensorPool
) -> None:
    if attr.HasField("t"):
        _hydrate_tensor_from_pool(attr.t, attr.t.name or f"{path}/t", pool)
    for i, t in enumerate(attr.tensors):
        _hydrate_tensor_from_pool(t, t.name or f"{path}/tensors{i}", pool)
    if attr.HasField("g"):
        _hydrate_graph_tensors_from_pool(attr.g, pool)
    for g in attr.graphs:
        _hydrate_graph_tensors_from_pool(g, pool)


def _hydrate_graph_tensors_from_pool(
    graph: onnx.GraphProto, pool: C.TensorPool
) -> None:
    """Mirrors tensor_pool_bridge.h's ForEachTensor/HydrateTensorProto in Python:
    hydrates every EXTERNAL tensor the pool resolved (initializers and node
    attribute tensors, recursing into subgraphs), keyed the same way the C++
    side pooled them (a tensor's own name, or -- for an unnamed attribute
    tensor -- a positional fallback) -- so this must stay in sync with that
    file's ForEachTensor if its key derivation ever changes.
    """
    for i, init in enumerate(graph.initializer):
        _hydrate_tensor_from_pool(init, init.name or f"initializer{i}", pool)
    for ni, node in enumerate(graph.node):
        node_path = node.name or f"node{ni}"
        for ai, attr in enumerate(node.attribute):
            _hydrate_attr_tensors_from_pool(attr, f"{node_path}/attr{ai}", pool)


def load_model(
    path: str, hydrate_all: bool = True
) -> Tuple[onnx.ModelProto, C.TensorPool]:
    """Load a model file, resolving its external weights through a
    :class:`onnxsim.onnxsim_cpp2py_export.TensorPool` instead of onnx's own
    external-data loader, and return that pool alongside the model.

    Dispatches on ``path``'s extension:

    * ``.safetensors`` / ``.gguf`` -- one of onnxsim's own self-describing
      archives (see :func:`export_safetensors` / :func:`export_gguf`): the
      graph and weights both live in this one file. Raises ``RuntimeError``
      if the archive has no embedded onnxsim model (e.g. a plain
      weights-only file with no graph to import).
    * anything else -- an ordinary ``.onnx`` file (produced by any exporter,
      not necessarily onnxsim). Behaves like ``onnx.load(path)`` (structure
      plus every tensor's values, inline) for any such model, but for one
      whose weights live in classic ONNX external data
      (``onnx.save(..., save_as_external_data=True)``, the default this
      package's own :func:`simplify` and CLI now use once a model's
      serialized size passes 100MB -- see ``DEFAULT_EXTERNAL_DATA_THRESHOLD``),
      loading itself (before any hydration) mmaps each *distinct* external
      file exactly once instead of onnx's own loader's one open+seek+read
      per tensor. Measured on a 190MB/2000-tensor external-data model
      (single shared weights file): resolving every tensor's location this
      way, with no data copied yet, took ~3ms, vs ~93ms for onnx.load to
      actually read them; hydrating every tensor too (``hydrate_all=True``,
      the default -- see below) still finished in ~65ms, ~1.4x faster than
      onnx.load's ~93ms, because the mmap turns each tensor's copy into a
      plain memory read instead of a seek + read syscall. A relative
      external-data ``location`` is resolved against this file's own
      directory, same as onnx's own loader; a tensor whose external file is
      missing or malformed is left as an unresolved ``EXTERNAL`` reference
      rather than failing the whole load, matching onnx's own loader's
      leniency for one bad reference in an otherwise-loadable model.

    :param path: path to the model file to load.
    :param hydrate_all: when ``True`` (the default), every tensor the pool
            resolves is also copied into the returned ``ModelProto`` as an
            ordinary in-memory tensor (one copy per tensor, straight from
            the mmap'd pool -- unavoidable, since ``onnx.TensorProto`` owns
            its bytes as a plain ``bytes`` field), ready for any onnxsim
            pass or onnx tool. Pass ``False`` to skip this copy entirely and
            leave the model's tensors as lazy ``EXTERNAL`` references
            instead -- the returned pool already holds their bytes
            zero-copy (``pool.bytes(name)`` copies out just the ones you
            ask for), so nothing is lost, but the model itself is not
            usable by code that doesn't know to hydrate from the pool on
            demand. See the measurements above for the difference this
            makes.
    :returns: ``(model, pool)`` -- the loaded model, and the ``TensorPool``
            every external tensor was resolved into (empty if ``path`` had
            no external weights to resolve). The pool supports ``len()``,
            ``name in pool``, ``pool.names()``, and, per tensor,
            ``pool.dtype(name)``, ``pool.shape(name)``, ``pool.bytes(name)``
            and ``pool.content_hash(name)``.

            For an ordinary ``.onnx`` file, the pool's entries may be a
            zero-copy memory mapping of the external-data file(s) on disk
            (see above): on Windows (unlike POSIX), the mapped file cannot
            be deleted or moved while the pool is still alive. Drop the
            pool (e.g. ``del pool``, or simply let it go out of scope)
            before deleting or replacing that file -- ``pool.bytes(name)``
            and friends already return independent copies, so extracting
            what you need and then dropping the pool is always safe.
    """
    model_bytes, pool = C.load_model(path)
    model = onnx.ModelProto()
    model.ParseFromString(model_bytes)
    if hydrate_all:
        _hydrate_graph_tensors_from_pool(model.graph, pool)
    return model, pool


def export_gguf(model: onnx.ModelProto, out_path: str) -> None:
    """GGUF counterpart of :func:`export_safetensors`, including its
    leaves-``model``-unchanged contract.
    """
    tensor_bytes: Dict[str, bytes] = {}
    _extract_graph_tensors_to_dict(model.graph, tensor_bytes)
    try:
        C.export_gguf(model.SerializeToString(), tensor_bytes, out_path)
    finally:
        _restore_graph_tensors_raw_data(model.graph, tensor_bytes)


def import_gguf(in_path: str) -> onnx.ModelProto:
    """GGUF counterpart of :func:`import_safetensors`."""
    model_bytes, pool = C.import_gguf(in_path)
    model = onnx.ModelProto()
    model.ParseFromString(model_bytes)
    _hydrate_graph_tensors_from_pool(model.graph, pool)
    return model


def _tensor_proto_nbytes(dtype: int, dims: Sequence[int]) -> Optional[int]:
    """Byte count a ``TensorProto``'s ``raw_data`` must have for `dtype`/`dims`,
    or ``None`` if `dtype` has no fixed-width numpy equivalent (e.g. STRING --
    never one of import_gguf_weights' own matched dtypes, so this only ever
    returns ``None`` defensively)."""
    try:
        itemsize = onnx.helper.tensor_dtype_to_np_dtype(dtype).itemsize
    except (KeyError, TypeError):
        return None
    n = itemsize
    for d in dims:
        n *= d
    return n


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
            Never mutated by this call, whether passed in directly or
            loaded from a path.
    :param gguf_path: path to the GGUF file to pull weight values from
    :returns: ``(model, skipped)`` -- the hydrated model (a distinct object
            from ``model``, if a ``ModelProto`` was passed in), and the
            names of GGUF tensors present in the file but skipped because
            either their quantized format has no decoder here (the legacy
            ``Q4_0`` family, every ``IQ*`` variant, ...) or, for a
            same-named match, the file's tensor decodes to a different byte
            count than ``model``'s initializer declares (dtype x dims) --
            e.g. a placeholder built for the wrong shape, or a checkpoint
            tensor sized for a different architecture configuration. Either
            way the initializer is left with its original value rather than
            overwritten with bytes that don't fit its declared shape. A
            GGUF tensor with no matching initializer name in ``model`` is
            simply not brought in -- it does NOT appear in ``skipped``,
            which reports only tensors the file and the graph both name but
            this call could not actually hydrate.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    # C++ only ever clears/rewrites a *matched* initializer's raw_data --
    # every unmatched one crosses back out untouched, still carrying
    # whatever `model` already had. But "matched" there means only "a
    # same-named GGUF entry exists": ImportModelWithGGUFToPool clears a
    # matched initializer's raw_data unconditionally, before this function
    # ever gets a chance to reject it below on a byte-count mismatch -- so
    # a rejected tensor's original value does NOT survive in `result` and
    # must be restored here from `model` (still unmutated, per this
    # function's own contract) rather than merely left alone.
    originals = {init.name: init for init in model.graph.initializer}
    model_bytes, matched, skipped = C.import_gguf_weights(
        model.SerializeToString(), gguf_path
    )
    result = onnx.ModelProto()
    result.ParseFromString(model_bytes)
    skipped = list(skipped)
    for init in result.graph.initializer:
        if init.name not in matched:
            continue
        dtype = matched.dtype(init.name)
        data = matched.bytes(init.name)
        # The C++ side decodes/copies purely from the GGUF entry's own shape
        # (see tensor_pool_gguf_bridge.h's HydrateTensorProtoFromGGUF), with
        # no awareness of `init`'s declared dims -- so a name match whose
        # actual byte count doesn't fit `init`'s shape (e.g. a hand-built
        # placeholder graph declared with the wrong shape for this
        # checkpoint) would otherwise leave a corrupt-length raw_data behind
        # instead of failing clearly. Treat that the same as an unsupported
        # quantization format: report it in `skipped` and restore the
        # initializer's original value (see this loop's own note above on
        # why that needs restoring rather than merely being left alone).
        expected = _tensor_proto_nbytes(dtype, init.dims)
        if expected is not None and expected != len(data):
            skipped.append(init.name)
            init.CopyFrom(originals[init.name])
            continue
        init.data_type = dtype
        init.raw_data = data
    return result, skipped


def read_gguf_metadata(path: str) -> dict:
    """
    Read a GGUF file's architecture hyperparameters and per-tensor
    name/shape/dtype list, without reading any tensor byte data -- cheap
    even against a multi-gigabyte real checkpoint.

    This is the piece :func:`import_gguf_weights` never surfaces: both
    functions parse the same GGUF header section, but ``import_gguf_weights``
    only ever looks at the ``general.alignment`` key before moving on to
    loading tensor *values* into an existing graph's initializers. Use this
    instead when what you need is the checkpoint's own description of its
    architecture -- e.g. ``general.architecture``, ``<arch>.block_count``,
    ``<arch>.attention.head_count``, ``<arch>.rope.freq_base`` -- to decide
    *what graph structure* to build in the first place (``import_gguf_weights``
    only ever fills in the values of a graph you already have).

    :param path: path to the GGUF file to read
    :returns: ``{"kv": {key: int | float | str | bool, ...},
            "tensors": [{"name": str, "shape": [int, ...],
            "ggml_type": int}, ...]}``. ``ggml_type`` is the raw GGML type
            code (see ``onnxsim/gguf_dtype.h``); an ARRAY-typed metadata
            value (e.g. ``tokenizer.ggml.tokens``, which alone can hold
            >100k strings in a real checkpoint) is omitted from ``"kv"``
            entirely rather than decoded.
    """
    return C.read_gguf_metadata(path)


def cross_layer_equalize(model: Union[str, onnx.ModelProto]) -> onnx.ModelProto:
    """
    Data-free Cross-Layer Equalization (CLE) -- the weight-equalization
    preprocessing technique from "Data-Free Quantization Through Weight
    Equalization and Bias Correction" (Nagel et al., 2019), also shipped as
    part of Qualcomm's AIMET toolkit.

    This is **not** a quantization scheme: no ``Quantize``/``DequantizeLinear``
    node is ever introduced, and the model's computed function is unchanged
    bit-for-bit -- only its internal weight *parameterization* changes. Run
    it *before* one of onnxsim's ``quantize_*`` functions to make the
    per-tensor or per-channel quantization that follows more accurate.

    For every pair of adjacent Conv layers ``Conv1 -> [activation] -> Conv2``
    where the activation (if any) is positive-homogeneous of degree 1 --
    ``f(a*x) == a*f(x)`` for every ``a > 0``, true of ``Relu``/``PRelu``/
    ``LeakyRelu`` and trivially true of "no activation at all" -- and both
    convs have ``group == 1``, this rescales each shared channel ``c`` by
    ``S[c] = sqrt(r1[c] / r2[c])`` (``r1``/``r2`` being Conv1's/Conv2's own
    per-channel weight range): Conv1's weight/bias for channel ``c`` divided
    by ``S[c]``, Conv2's weight for channel ``c`` multiplied by ``S[c]``.
    This makes the two layers' per-channel weight ranges identical -- the
    most balanced a fixed pair can be -- without changing the composed
    function at all, since the activation's positive homogeneity is exactly
    what lets ``S[c]`` and ``1/S[c]`` cancel across it. A single call already
    equalizes a whole chain of layers, not just one adjacent pair: onnxsim
    reruns this pass, along with every other registered pass, to a
    network-wide fixed point.

    Scope of this implementation (each just declines the match rather than
    mishandling it -- not correctness bugs):

    - Conv only (no ConvTranspose, no Gemm/MatMul-based fully-connected
      equalization).
    - ``group`` must be 1 on both convs.
    - FLOAT32 weights/bias only.
    - No "high-bias absorption" (AIMET's optional follow-up step for a
      following BatchNorm's bias) -- plain BN folding (already part of
      :func:`simplify`) upstream of this pass covers the common case fine on
      its own.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.

    :param model: onnx ModelProto object or file path
    :returns: the equalized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.cross_layer_equalize(model.SerializeToString()))


def quantize_dynamic(
    model: Union[str, onnx.ModelProto], verify: bool = False
) -> onnx.ModelProto:
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
    :param verify: opt-in (default ``False``): afterwards run
            :func:`onnxsim.quant_int_verify.attach_verification` on the result -- prove each integer
            layer cannot wrap its int32 accumulator, check the integer weights against ``model``'s,
            and note whether the model would give different results on a CPU whose ``u8 x s8``
            kernel saturates int16 pair sums. The return value is unchanged; a one-line summary is
            written to the result's ``metadata_props["onnxsim.quant_int_verify"]``, a
            ``RuntimeWarning`` is emitted when a soundness finding is refuted, and the check never
            raises. Call :func:`onnxsim.quant_int_verify.verify_model` for the full report.
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    out = onnx.load_from_string(C.quantize_dynamic(model.SerializeToString()))
    if verify:
        from . import quant_int_verify as _quant_int_verify

        _quant_int_verify.attach_verification(out, reference_model=model)
    return out


def quantize_dynamic_matmul_integer_to_float(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    Same rewrite as :func:`quantize_dynamic` -- same matching rules, same
    per-output-channel weight quantization, same runtime
    ``DynamicQuantizeLinear`` activation quantization -- but the dequantize
    step is a single ONNX Runtime "com.microsoft" contrib op,
    ``MatMulIntegerToFloat``, instead of :func:`quantize_dynamic`'s
    three-to-four separate standard-ONNX nodes (``MatMulInteger`` + ``Cast``
    + two ``Mul``s + an optional ``Add``): ``MatMulIntegerToFloat``'s own
    schema dequantizes and adds an optional bias directly, so this needs
    only ``DynamicQuantizeLinear`` plus the one contrib op.

    Unlike :func:`quantize_dynamic`, the result is **not** portable standard
    ONNX -- ``MatMulIntegerToFloat`` is an ONNX Runtime contrib op, so the
    quantized model needs a "com.microsoft"-aware runtime to execute. No
    calibration data is needed either way.

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
    return onnx.load_from_string(
        C.quantize_dynamic_matmul_integer_to_float(model.SerializeToString())
    )


def quantize_attention_dynamic(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    Dynamically quantizes every "com.microsoft" ``Attention`` node already
    present in ``model`` (this does not fuse attention itself -- run
    :func:`simplify` first to produce ``Attention`` nodes if the input model
    doesn't already have any; see :mod:`onnxsim`'s ``fuse_attention`` pass)
    into ``Attention``'s quantized counterpart, ``QAttention``.

    The merged Q/K/V weight is quantized to INT8 ahead of time (per output
    channel, symmetric, from its static values -- no calibration data is
    needed), while the activation is quantized to uint8 *in the graph* via
    ``DynamicQuantizeLinear``, which computes its own scale/zero-point from
    each run's actual input range -- mirroring :func:`quantize_dynamic`'s own
    scheme for MatMul/Gemm.

    ``Attention``'s optional ``qkv_hidden_sizes`` attribute lets V's hidden
    size differ from Q/K's (see :func:`simplify`'s ``fuse_attention`` pass),
    but ``QAttention``'s schema assumes an even three-way split -- a node
    with an uneven split is left unquantized rather than guessing how (or
    whether) the kernel would handle it.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, fuse_attention, or any
    other pass. Nodes that do not match (no ``Attention`` node, a
    non-constant or non-2-D weight, a non-float32 activation, an opset older
    than 11 -- which ``DynamicQuantizeLinear`` requires -- or an uneven
    ``qkv_hidden_sizes`` split) are left untouched. Consider calling
    :func:`simplify` before and/or after to produce ``Attention`` nodes and
    clean up the graph.

    :param model: onnx ModelProto object or file path
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.quantize_attention_dynamic(model.SerializeToString())
    )


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


def quantize_weight_only_matmul_nbits(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    Block-wise INT4 weight-only quantize every MatMul, and every "vanilla"
    Gemm (transA=0, alpha=1, beta=1), whose weight is a constant 2-D
    float32 tensor, into ONNX Runtime's own ``com.microsoft::MatMulNBits``
    contrib op.

    This is a **vendor-specific** counterpart to
    :func:`quantize_weight_only_int4`: same INT4 precision, same 32-element
    block-wise scale, but packed into ONNX Runtime's own single fused op --
    the format ORT's own GenAI/quantization tooling (Olive,
    ``onnxruntime.quantization``'s ``matmul_4bits_quantizer``, ...) emits
    for LLM/ASR weight compression -- instead of
    :func:`quantize_weight_only_int4`'s portable, standard-ONNX
    INT4-tensor-plus-``DequantizeLinear`` pair. Smaller and faster on ONNX
    Runtime specifically, at the cost of needing ORT (or another runtime
    implementing this contrib op) to run at all: unlike every other
    ``quantize_*`` function in onnxsim, the result does **not** load on an
    arbitrary conformant ONNX runtime.

    Unlike :func:`quantize_weight_only_int4` (which needs opset 21 for
    ONNX's own native INT4 tensor type and ``DequantizeLinear``'s
    ``block_size`` attribute), this needs no minimum standard opset --
    ``MatMulNBits`` is a self-contained contrib op -- and, unlike that
    function, does not require the reduction dimension to be evenly
    divisible by the 32-element block size: ``MatMulNBits`` itself defines
    ``k_blocks = ceil(K / block_size)``, so a ragged last block is
    quantized exactly like every other one instead of being declined.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Nodes that do not match (dynamic or non-2-D weights, non-default Gemm
    attributes, non-float32 operands) are left untouched. Consider calling
    :func:`simplify` before and/or after to clean up the graph.

    :param model: onnx ModelProto object or file path
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.quantize_weight_only_matmul_nbits(model.SerializeToString())
    )


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


def quantize_weight_only_mxfp4_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_mxfp4`: OCP
    Microscaling MXFP4 weight-only quantizes every MatMul and every
    "vanilla" Gemm (transA=0, alpha=1, beta=1) whose weight is a constant
    2-D float32 tensor whose reduction dimension ``K`` is evenly divisible
    by 32 (the OCP MX spec's own canonical block size).

    Unlike every other ``quantize_weight_only_*`` scheme, MXFP4's per-block
    scale is constrained to a pure power of two, and its 4-bit codes follow
    a fixed, non-uniform (E2M1 floating-point) codebook rather than an
    ordinary affine range -- see :func:`onnxsim.quantize_weight_only_mxfp4`'s
    own docstring for the format's full definition. Needs no calibration
    data: both the codebook and the per-block power-of-two scale come from
    the weight's own values. Like the pure-Python implementation, only
    ``MatMul``/``Gemm`` are handled -- neither port supports ``Conv``.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Layers with a non-constant, non-2-D, or non-block-divisible weight are
    left untouched. Consider calling :func:`simplify` before and/or after to
    clean up the graph.

    :param model: onnx ModelProto object or file path
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.quantize_weight_only_mxfp4(model.SerializeToString())
    )


def apply_double_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_double_quantization`: applies
    QLoRA-style double quantization (Dettmers et al., 2023, "QLoRA:
    Efficient Finetuning of Quantized LLMs", Section 3.2) to every
    ``DequantizeLinear`` node already present in ``model`` whose scale input
    is a constant float32 tensor with at least 64 values.

    Every block-wise/per-channel scale (e.g. one float32 value per 32-element
    INT4 block) is itself quantized to UINT8 with a single per-tensor
    meta-scale, reconstructed in-graph via a second, nested
    ``DequantizeLinear`` feeding the original node's own scale input -- see
    :func:`onnxsim.apply_double_quantization`'s own docstring for the exact
    rewrite. This is technique-agnostic: it composes with the output of any
    onnxsim block-wise/per-channel quantizer (or any other model containing
    ``DequantizeLinear`` nodes) unchanged.

    :param model: an already-quantized onnx ModelProto or file path
    :returns: ``model`` with every matching ``DequantizeLinear`` node's scale
            input double-quantized; a scale that is not a constant
            initializer, not float32, or too small (fewer than 64 values) is
            left untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_double_quantization(model.SerializeToString()))


def prune_magnitude_cpp(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    global_sparsity: bool = False,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_magnitude_pruning`: zeros the
    least-magnitude entries of every MatMul/vanilla-Gemm layer's constant
    2-D FLOAT/FLOAT16/BFLOAT16 weight, every Conv layer's constant 4-D
    FLOAT/FLOAT16/BFLOAT16 weight (ordinary, depthwise, and general grouped
    Conv alike), and every ``com.microsoft::Attention`` node's constant 2-D
    FLOAT/FLOAT16/BFLOAT16 merged QKV weight -- the data-free unstructured
    pruning baseline (Han et al., 2015). Full parity with the pure-Python
    :func:`onnxsim.apply_magnitude_pruning`.

    Within each output row (or, for Conv, each output filter), keeps the
    ``max(1, round(cols * (1 - sparsity)))`` highest-magnitude entries and
    zeros the rest, so a layer with row-dependent weight scale doesn't get
    some rows pruned to nothing and others left untouched.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Layers with a non-constant, non-2-D (MatMul/Gemm/Attention), or non-4-D
    (Conv) weight are left untouched.

    :param model: onnx ModelProto object or file path
    :param sparsity: target fraction of each row's (or, for Conv, each
            output filter's) entries to zero, ignored when ``n``/``m`` are
            given -- or, when `global_sparsity`, target fraction of every
            matched layer's entries *combined*
    :param n: keep the ``n`` highest-magnitude entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4). Must be given
            together with ``m``; incompatible with `global_sparsity`.
    :param m: group size for N:M pruning; see ``n``
    :param global_sparsity: the classic "global magnitude pruning" variant
            (Han et al., 2015): pools every matched layer's ``|W|`` entries
            into one ranking across the whole model and zeros exactly
            `sparsity`'s fraction of that pooled total, wherever it lands
            (no per-row/per-layer floor -- see
            :func:`onnxsim.apply_magnitude_pruning`'s own docstring).
            Incompatible with ``n``/``m``.
    :returns: ``model`` with every matched layer's weight pruned in place
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.prune_magnitude(model.SerializeToString(), sparsity, n, m, global_sparsity)
    )


def apply_structured_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
    global_sparsity: bool = False,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_structured_pruning`: removes
    whole output channels from MatMul/vanilla-Gemm and Conv layers -- real
    structural pruning (smaller weight tensors, smaller matmuls on any
    runtime), as opposed to :func:`onnxsim.prune_magnitude_cpp`'s value-only
    zeroing.

    For every MatMul/vanilla-Gemm or 2-D Conv "producer" node whose output
    feeds, through zero or more shape-preserving elementwise ops (an
    activation, or -- MatMul/Gemm only -- an Add/Mul against a constant
    per-channel bias/scale, or -- Conv only -- a depthwise Conv hop) with no
    other consumer anywhere along that path, into exactly one downstream
    "consumer" of the same family: ranks the producer's output channels by
    L2 norm of their own weight row/filter, drops the lowest-``sparsity``-
    fraction of them, and removes the corresponding rows/columns from the
    producer's weight (and bias, if constant) and every intermediate
    per-channel constant, and the matching columns/rows from the consumer's
    weight. A general grouped Conv (neither ``group=1`` nor fully depthwise)
    is matched too, as a producer and/or consumer, ranking/pruning each of
    its ``group`` channel blocks independently. The gated-FFN SwiGLU/GeGLU
    pattern is matched too -- two producers combined by a ``Mul`` (or ONNX
    opset-28+'s native ``SwiGLU`` node) feeding one consumer, both pruned to
    the same combined-importance-ranked channel indices. A Conv or
    MatMul/Gemm residual (skip-connection) chain is matched too -- a
    channel-preserving ``Add(a, b)`` where both operands are non-constant
    forces whichever real producer(s) feed ``a``/``b`` to agree on one
    shared channel-index set, resolved via a backward walk plus union-find
    grouping across such merge points that also covers a whole chain of
    such merges transitively sharing one spine channel count (a lone
    residual connection, or a linear stack of ``Add``-only merges). Once a
    group's shared channel-index set is established, it can also fan out
    *forward* to more than one independent ordinary consumer -- so a real
    multi-block ResNet/transformer stage's shared "post-block" tensor, read
    by both the next block's own first Conv/MatMul *and*, unchanged, that
    block's own ``Add``, is reached rather than declined; a general grouped
    Conv may take part in this merge too, as a producer, the primary
    consumer, and/or an extra fan-out branch, as long as every one of those
    that is grouped shares the exact same ``group`` count. For MatMul/Gemm
    specifically, a fused
    ``com.microsoft::SkipLayerNormalization``/``SkipSimplifiedLayerNormalization``
    node -- what onnxruntime's transformer optimizer collapses a bare
    residual ``Add`` plus the following LayerNorm into, and so what a
    fully-optimized transformer's own residual connections typically look
    like -- is recognized as an eligible merge point too, its own
    ``gamma``/``beta``/``bias`` constants riding along as a per-channel
    affine hop on the resolved chain; a Conv residual chain only ever sees a
    bare ``Add`` (there is no Conv analogue of that fused op). A fused
    ``com.microsoft::BiasGelu``/``FastGelu`` node is recognized as a
    per-channel hop too (MatMul/Gemm chains only), and
    ``com.microsoft::QuickGelu`` is a plain unary pass-through hop
    everywhere a unary activation is already allowed.

    A ``Concat``-merged skip connection (the U-Net-style encoder/decoder
    merge) is matched too, for both MatMul/Gemm (last-axis ``Concat`` only)
    and Conv (channel-axis ``Concat``): unlike ``Add``, a ``Concat``'s
    branches are structurally independent -- each owns a fixed, disjoint
    offset range of the merged channel range -- so each branch is ranked and
    pruned entirely on its own; only the shared downstream consumer's weight
    needs new slicing, at each branch's own fixed offset. A branch may
    itself resolve through a gated (SwiGLU/GeGLU) combine or a whole
    Add/SkipLayerNormalization residual group; a branch that fans out
    elsewhere, or would need to cross another ``Concat`` or a fused
    self-attention op boundary, declines the *entire* group, never partially
    pruned.

    Also prunes ``com.microsoft::MatMulNBits`` (block-quantized, weight-only
    int4/int8) chains -- the plain (producer -> consumer), gated
    (SwiGLU/GeGLU), and fused-``gate_up_proj`` (``MatMulNBitsMlp``) families,
    a C++-port subset of
    :func:`onnxsim.apply_structured_pruning_matmul_nbits` (its
    ``MatMulNBitsQkv`` variant is deliberately NOT ported here -- pruning a
    whole KV group needs the head-count-matching machinery
    :func:`onnxsim.apply_attention_head_pruning_cpp` already owns, so that
    one variant is wired into that entry point instead). Either side of a
    matched (non-``MatMulNBitsMlp``) chain may independently be a
    ``MatMulNBits`` node or a plain-float MatMul/vanilla-Gemm peer (at least
    one side must be ``MatMulNBits``); the producer's output channels are
    ranked by L2 norm of their own DEQUANTIZED weight row (never written
    back -- the actual rewrite always row/column-slices the existing packed
    ``B``/``scales``/``zero_points`` codes in place, re-packing a nibble-
    packed axis rather than ever re-quantizing a sliced float weight from
    scratch), and -- only when the consumer is itself ``MatMulNBits`` --
    that keep-set must land on whole ``block_size``-sized blocks of the
    consumer's own quantized ``K`` axis, or the whole chain is left
    untouched (an individual K-column can't be dropped without
    re-quantizing its block, out of scope).

    Several more quantized-weight chain families this C++ entry point
    additionally consolidates from their own SEPARATE pure-Python top-level
    functions (each still exists independently -- see this module's own
    "structured pruning" section comments in ``pruning.py`` for why each was
    kept a standalone function there rather than folded into the pure-Python
    :func:`onnxsim.apply_structured_pruning` itself: retrofitting a
    quantized-weight representation into machinery already extensively
    tested against float32/float16/bfloat16 weights only was judged not
    worth the regression risk, for a representation that never aliases a
    plain float weight anyway):

    * QDQ (a ``DequantizeLinear``-fed int8/uint8 weight, per-tensor,
      per-channel, or opset-21+ blockwise) -- plain and gated families, the
      C++ counterpart of :func:`onnxsim.apply_structured_pruning_qdq`.
    * ``com.microsoft::MatMulBnb4`` (bitsandbytes FP4/NF4 block-quantized
      weight) -- :func:`onnxsim.apply_structured_pruning_matmul_bnb4`'s C++
      counterpart.
    * ``MatMulBlockQuantizedFp8Weight``/``MatMulBlockQuantizedFp4Weight``
      (NVFP4/FP8 block-quantized weight) --
      :func:`onnxsim.apply_structured_pruning_matmul_block_quantized_fp8`/
      :func:`onnxsim.apply_structured_pruning_matmul_block_quantized_fp4`'s
      C++ counterparts.
    * QOperator static quantization (``QLinearConv``/``QLinearMatMul``/
      ``QGemm``) -- :func:`onnxsim.apply_structured_pruning_qoperator`'s C++
      counterpart.
    * ``com.microsoft::DynamicQuantizeMatMul``/``MatMulIntegerToFloat``
      (ORT's dynamic-quantization fusion) --
      :func:`onnxsim.apply_structured_pruning_dynamic_quantize_matmul`'s C++
      counterpart.
    * ``ConvInteger``-based dynamic quantization (ORT's unfused
      ``DynamicQuantizeLinear -> ConvInteger -> Cast -> Mul`` Conv pattern) --
      a PARTIAL C++ counterpart of
      :func:`onnxsim.apply_structured_pruning_dynamic_quantize_conv`: the
      ordinary same-family producer/consumer chain (plus its ``Clip``
      pass-through and depthwise-mid-chain hops) is matched and pruned here,
      but the Python reference's own extra
      ``GlobalAveragePool -> Flatten/Reshape -> {MatMul, Gemm}``
      classifier-head exception is NOT -- so, unlike every other family
      above, this one is still not a full behavioral match and
      :func:`onnxsim.apply_structured_pruning_dynamic_quantize_conv` itself
      remains pure Python, not aliased to this entry point.

    None of the pure-Python per-format functions above have a
    ``global_sparsity`` parameter of their own, so ``global_sparsity`` below
    has no defined meaning for any quantized-weight family here either --
    every one of them is always applied with its own local per-chain
    sparsity, unaffected by that flag.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched producer's output
            channels to remove (at least one channel is always kept) -- or,
            when ``global_sparsity``, target fraction of every eligible
            chain's channels *combined*; must be in ``[0, 1)``
    :param importance_norm: ``"l2"`` (default, unchanged from before this
            parameter existed) ranks by Li et al.'s own root-sum-square L2
            criterion; ``"l1"`` ranks by NNI's alternative L1 criterion
            (sum of absolute weight magnitude) instead -- mirrors the
            pure-Python :func:`onnxsim.apply_structured_pruning`'s own
            ``importance_norm`` parameter exactly. Applies identically
            whether or not ``global_sparsity`` is set.
    :param global_sparsity: pools every *eligible* matched chain's own
            per-channel importance into one ranking across the whole model
            and picks a single keep-count from ``sparsity``'s fraction of
            that pooled total, instead of every chain being cut by the same
            fraction independently -- mirrors the pure-Python
            :func:`onnxsim.apply_structured_pruning`'s own
            ``global_sparsity`` mode exactly, including which chains are
            "eligible" (an ordinary, single-producer chain with no extra
            fan-out consumer branch and no general grouped Conv on either
            side -- a gated pair, a residual/merge group, a
            ``Concat``-merged branch, and any general grouped Conv chain
            are all left completely untouched in this mode instead) and the
            per-chain floor of at least one surviving channel. Default
            ``False`` -- every pre-existing caller's behavior is unchanged.
    :returns: ``model`` with every matched chain's tensors resized in place;
            anything not matching the exact topology above (branching, a
            non-constant bias, a consumer whose reduction dimension doesn't
            line up, ...) is left completely untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_structured_pruning(
            model.SerializeToString(), sparsity, importance_norm, global_sparsity
        )
    )


def apply_structured_wanda_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
    epsilon: float = 1e-8,
    providers: Optional[Sequence[backend.Provider]] = None,
    global_sparsity: bool = False,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_structured_wanda_pruning`: the
    calibrated upgrade of :func:`onnxsim.apply_structured_pruning_cpp`,
    exactly as the pure-Python :func:`onnxsim.apply_wanda_pruning` is to
    :func:`onnxsim.apply_magnitude_pruning`. This is onnxsim's first
    calibration-driven (not purely data-free) C++ structured-pruning entry
    point -- see :func:`onnxsim.simplify`'s own use of a
    :class:`onnxsim.onnx_simplifier.PyModelExecutor` for the general "hand a
    Python-backed model executor to the C++ core" pattern this reuses; every
    prior C++-ported pruning pass (:func:`onnxsim.apply_structured_pruning_cpp`
    and friends) is purely graph-structural and needs no executor at all.

    Same real structural channel removal and topology matching as
    :func:`onnxsim.apply_structured_pruning_cpp` (a single producer -> ...
    -> consumer chain, a gated SwiGLU/GeGLU pair, or a bounded Conv or
    MatMul/Gemm residual/merge group; a ``Concat``-merged branch; the
    fused-``gate_up_proj`` split-gated FFN shape -- see that function's own
    docstring for the full topology list), EXCEPT the additional
    quantized-weight chain families it also matches
    (``MatMulNBits``/QDQ/``MatMulBnb4``/FP8/FP4 block-quantized/QOperator/
    ``DynamicQuantizeMatMul``) are out of scope here too -- mirroring the
    pure-Python :func:`onnxsim.apply_structured_wanda_pruning`, which has no
    quantized-weight counterpart either. Each matched chain's output
    channels are ranked by ``||W_row||_2 * ||X||_2`` -- L2 norm of that
    channel's own weight row (or, for Conv, whole filter), times the L2 norm
    of the *activation* actually flowing through that channel over
    ``calibration_data`` (captured right where the chain feeds into its
    consumer -- or, for a ``Concat`` branch, right where it feeds into the
    ``Concat`` node -- reduced over every axis but the channel one) --
    instead of weight magnitude alone. The fused-``gate_up_proj`` split-gated
    chain family is matched but always ranked by plain weight magnitude only
    (never calibrated), matching the pure-Python original's own scope
    decision for that one family.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            chain's consumer-side activation norm on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched chain's output
            channels to remove (at least one channel is always kept) -- or,
            when ``global_sparsity``, target fraction of every eligible
            chain's channels *combined*; must be in ``[0, 1)``
    :param importance_norm: ``"l2"`` (default) or ``"l1"`` -- selects the
            *weight*-magnitude term ``||W_row||`` only, exactly as it does
            for :func:`onnxsim.apply_structured_pruning_cpp`; the
            *activation*-norm term ``||X||_2`` stays L2 unconditionally
            either way, per Wanda's own ``|W_ij| * ||X_j||_2`` definition.
            Applies identically whether or not ``global_sparsity`` is set.
    :param epsilon: floor applied to the accumulated per-channel activation
            norm, avoiding every channel of an all-zero activation tying at
            exactly the weight-only importance
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations (passed to the shared
            :func:`onnxsim.onnx_simplifier._get_model_executor` process-wide
            executor, the same one :func:`onnxsim.simplify` itself uses)
    :param global_sparsity: the structural analogue of
            :func:`onnxsim.apply_structured_pruning_cpp`'s own
            ``global_sparsity`` mode, applied to this function's own
            ``||W_row||_2 * ||X||_2`` metric -- see that function's own
            docstring for the full mechanism, the per-chain floor, and
            exactly which chains are "eligible" to take part in the pool.
            Default ``False`` -- every pre-existing caller's behavior is
            unchanged.
    :returns: ``model`` with every matched chain's tensors resized in place;
            anything not matching that exact topology falls back to
            :func:`onnxsim.apply_structured_pruning_cpp`'s own plain
            weight-magnitude ranking if no matching activation was ever
            observed for that chain's consumer (including whenever
            ``calibration_data`` is empty)
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Each batch crosses the pybind boundary as a {input_name: TensorProto}
    # map (not raw ndarrays) -- the same onnx::TensorProto nanobind caster
    # every other proto crosses this boundary with (see cpp2py_export.cc's
    # own apply_structured_wanda_pruning binding comment), so this is the
    # only conversion needed on the Python side; the name -> positional
    # reordering ModelExecutor::Run itself requires happens entirely on the
    # C++ side (WandaCalibrationStats in structured_pruning_entry.cpp).
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_structured_wanda_pruning(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            sparsity,
            epsilon,
            importance_norm,
            global_sparsity,
        )
    )


def apply_attention_head_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_attention_head_pruning`: removes
    whole attention heads -- or, for grouped-query attention, whole KV
    groups -- from every matched fused self-attention block whose output
    feeds, optionally through a single shape-preserving ``Reshape``, exactly
    one downstream MatMul/vanilla-Gemm's reduction dimension (the output
    projection) -- the attention analogue of
    :func:`onnxsim.apply_structured_pruning_cpp`, at head (or KV-group)
    instead of single-channel granularity. Every matched producer/consumer
    weight (and bias) may be FLOAT, FLOAT16, or BFLOAT16 -- read/ranked/
    sliced in double precision and written back down to its own original
    dtype, so a surviving entry's exact bit pattern is preserved.

    Three families of fused op are matched:

    * A single merged QKV weight/bias -- ``com.microsoft::Attention``,
      ``DecoderMaskedSelfAttention``, or ``PackedAttention`` (the three share
      ``Attention``'s own weight/bias input positions and are matched
      identically, net of ``DecoderMaskedSelfAttention``'s own schema
      quirks -- no ``qkv_hidden_sizes`` attribute, a required, must-stay-
      dynamic ``past`` input, no fused ``do_rotary``). Ranks every head by
      the combined Frobenius norm of its own Q, K, and V weight columns,
      drops the lowest-``sparsity``-fraction of heads (at least one head is
      always kept), and removes the corresponding column blocks from the
      merged QKV weight (and bias, if present), decrementing
      ``num_heads``/``qkv_hidden_sizes`` accordingly (the latter left alone
      for ``DecoderMaskedSelfAttention``, which has no such attribute), and
      the matching row block from the output projection's weight.
    * Separate, un-merged Q/K/V producers -- ``GroupQueryAttention``, plain
      ``ai.onnx::Attention`` (opset 24+), ``MultiHeadAttention``,
      ``PackedMultiHeadAttention``, ``DecoderMaskedMultiHeadAttention``,
      ``PagedAttention``, ``LinearAttention``, or ``SparseAttention``. Ranks
      every *KV group* (a KV head and the ``num_heads / kv_num_heads`` query
      heads the kernel maps to it) by the combined Frobenius norm of that
      group's own Q+K+V weight block, drops the lowest-``sparsity``-fraction
      of groups (at least one group is always kept), and removes the
      corresponding column blocks from all three producers (and their
      biases, or a ``MultiHeadAttention``-style single combined Q+K+V bias,
      if present) together with the matching row block from the output
      projection's weight, decrementing the query head count and
      ``kv_num_heads`` by the number of groups dropped -- so their ratio
      (query heads per KV head) is unchanged. An individual query head is
      never dropped on its own: only a whole group, since neither kernel has
      a way to keep a KV head alive for some, but not all, of the query
      heads that shared it.
    * The fully decomposed ("eager SDPA export") shape -- separate Q/K/V
      Linear producers (or one shared packed-QKV-then-``Split`` producer)
      each reshaped/transposed to BNSH, an optional ``repeat_kv``
      (``Unsqueeze``/``Expand``/``Reshape``) broadcast of K/V up to the full
      query head count, optional decomposed RoPE and/or Q/K-norm pass-
      throughs, ``MatMul(Q, K^T)`` -> scale -> optional additive mask ->
      ``Softmax`` -> ``MatMul(., V)`` -> transpose/reshape back, feeding the
      output projection. Ranked and pruned the same KV-group way as the
      separate-producer family above (or, for true MQA -- a single shared KV
      head -- by individual query head instead, since group-granularity
      ranking can never distinguish anything when there is only one group).

    Also prunes ``com.microsoft::MatMulNBitsQkv`` (a block-quantized,
    weight-only int4/int8 fused QKV projection) at whole-KV-group
    granularity, reusing this function's own GQA/plain-Attention head-count-
    matching machinery -- the C++ port's counterpart of
    :func:`onnxsim.apply_structured_pruning_matmul_nbits`'s own
    ``MatMulNBitsQkv`` handling (deliberately consolidated here rather than
    in :func:`onnxsim.apply_structured_pruning_cpp`, which explicitly defers
    to this function for that one op -- see that function's own docstring).

    The calibration-driven Wanda upgrade of the three fused-op families above
    (mirroring the pure-Python :func:`onnxsim.apply_attention_head_wanda_pruning`)
    is :func:`onnxsim.apply_attention_head_wanda_pruning_cpp` -- which does
    NOT include the ``MatMulNBitsQkv`` family, mirroring pruning.py's own
    scope (no calibration-driven counterpart of
    :func:`onnxsim.apply_structured_pruning_matmul_nbits` exists there
    either).

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched block's heads (or, for
            the separate-producer/decomposed families, KV groups) to remove
            (at least one is always kept); must be in ``[0, 1)``
    :param importance_norm: ``"l2"`` (default, unchanged from before this
            parameter existed) ranks by the combined Frobenius (L2) norm of
            each head's/KV group's own weight block, mirroring the
            pure-Python :func:`onnxsim.apply_attention_head_pruning`'s own
            default; ``"l1"`` ranks by the sum of absolute weight magnitude
            across that same block instead.
    :returns: ``model`` with every matched block's tensors resized in
            place; anything not matching that exact topology (a
            non-constant or unsupported-dtype weight, a packed-QKV
            GroupQueryAttention node, an ai.onnx Attention node with
            differing Q/K/V head sizes or without explicit
            ``q_num_heads``/``kv_num_heads`` attributes, a consumer whose
            reduction dimension doesn't line up, ...) is left completely
            untouched. A non-empty constant past-KV-cache or attention-
            mask/bias input no longer forces this on its own: every matched
            family now validates such an input and either slices it in
            place (a genuine per-head/per-KV-group constant), leaves it
            untouched (a broadcast shape needing no slicing), or splices in
            a new ``Gather`` node (a genuinely per-head *dynamic* mask/bias)
            -- only a shape that doesn't statically resolve either way still
            declines the whole match.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_attention_head_pruning(
            model.SerializeToString(), sparsity, importance_norm
        )
    )


def apply_attention_head_wanda_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
    epsilon: float = 1e-8,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_attention_head_wanda_pruning`:
    the calibrated upgrade of :func:`onnxsim.apply_attention_head_pruning_cpp`,
    exactly as :func:`onnxsim.apply_structured_wanda_pruning_cpp` is to
    :func:`onnxsim.apply_structured_pruning_cpp` -- same real
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed calibration
    machinery (see that function's own docstring for the general pattern),
    applied to attention-head pruning instead of plain structured pruning.

    Same real head (or, for the separate-Q/K/V-producer and decomposed
    families, whole-KV-group) removal and topology matching as
    :func:`onnxsim.apply_attention_head_pruning_cpp` -- every one of that
    function's families *except* the fused, block-quantized
    ``com.microsoft::MatMulNBitsQkv`` variant (mirroring pruning.py's own
    ``apply_attention_head_wanda_pruning``, which has no calibration-driven
    counterpart of that quantized-weight case either): a matched
    ``com.microsoft::Attention``/``DecoderMaskedSelfAttention``/
    ``PackedAttention`` (single merged QKV weight); a matched
    ``com.microsoft::GroupQueryAttention``, plain ``ai.onnx::Attention``
    (opset 24+), ``MultiHeadAttention``, ``PackedMultiHeadAttention``,
    ``DecoderMaskedMultiHeadAttention``, ``PagedAttention``,
    ``LinearAttention``, or ``SparseAttention`` block (separate, un-merged
    Q/K/V producers); or the fully decomposed ("eager SDPA export") shape --
    each whose output feeds, optionally through a single shape-preserving
    ``Reshape``, exactly one downstream MatMul/vanilla-Gemm's reduction
    dimension -- see that function's own docstring for the full topology and
    per-block-kind removal details. Each unit's importance is
    ``||W||_F * ||X||_2`` -- the plain
    Frobenius-norm weight score times the combined (root-sum-square)
    activation norm of that unit's own slice of the *output projection's*
    input, captured over calibration data -- instead of weight magnitude
    alone. For a plain ``com.microsoft::Attention`` block this is per head,
    exactly as before; for a ``GroupQueryAttention``/plain
    ``ai.onnx::Attention`` block the activation norm is combined
    (root-sum-square) over every query head a KV group owns, mirroring how
    the plain weight importance is already combined across that group's own
    Q+K+V producers.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            block's output-projection-side activation norm on. Each batch
            is a ``{input_name: np.ndarray}`` dict matching ``model``'s
            graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched block's heads (or, for
            GroupQueryAttention/plain ai.onnx Attention, KV groups) to
            remove (at least one is always kept); must be in ``[0, 1)``
    :param importance_norm: ``"l2"`` (default) or ``"l1"`` -- selects the
            *weight*-magnitude term only, exactly as it does for
            :func:`onnxsim.apply_attention_head_pruning_cpp`; the
            *activation*-norm term stays L2 unconditionally either way.
    :param epsilon: floor applied to the accumulated per-unit activation
            norm, avoiding every unit of an all-zero activation tying at
            exactly the weight-only importance
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations (passed to the shared
            :func:`onnxsim.onnx_simplifier._get_model_executor` process-wide
            executor, the same one :func:`onnxsim.simplify` itself uses)
    :returns: ``model`` with every matched block's tensors resized in place;
            anything not matching that exact topology falls back to
            :func:`onnxsim.apply_attention_head_pruning_cpp`'s own plain
            Frobenius-norm ranking if no matching activation was ever
            observed for that block's consumer (including whenever
            ``calibration_data`` is empty)
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_structured_wanda_pruning_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_attention_head_wanda_pruning(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            sparsity,
            epsilon,
            importance_norm,
        )
    )


def apply_sparsegpt_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_sparsegpt_pruning`: SparseGPT
    (Frantar & Alistarh, 2023, "SparseGPT: Massive Language Models Can Be
    Accurately Pruned in One-Shot", https://arxiv.org/abs/2301.00774) --
    zeros the least-important entries of every matched layer's constant 2-D
    weight to an unstructured or N:M sparsity pattern, using a sequential,
    Hessian-error-compensating algorithm ported from GPTQ (same authors,
    same Cholesky-factored inverse Hessian reformulation) rather than
    magnitude/Wanda's one-shot static importance score -- every *kept*
    entry may also change value, having accumulated compensation for every
    entry pruned before it. Same real
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed calibration
    machinery as :func:`onnxsim.apply_structured_wanda_pruning_cpp` (see
    that function's own docstring for the general pattern), but -- unlike
    every other ``*_cpp`` pruning entry point in this module -- this NEVER
    changes any tensor's shape: only individual weight entries are
    rewritten in place, and every matched layer is processed completely
    independently, with no producer/consumer chain-walking at all.

    Matches every plain ``MatMul``/vanilla-``Gemm`` node with a constant 2-D
    FLOAT32/FLOAT16/BFLOAT16 weight (this already includes ``com.microsoft::
    GroupQueryAttention``'s own separate Q/K/V projections -- ordinary
    MatMul/Gemm nodes feeding it, not a weight the op itself owns), every
    ``com.microsoft::Attention`` node's constant 2-D FLOAT32/FLOAT16/
    BFLOAT16 merged QKV weight, and every 2-D ``Conv``/``FusedConv`` node
    (ordinary/depthwise/general-grouped alike) with a constant 4-D
    FLOAT32/FLOAT16/BFLOAT16 ``[out_channels, in_channels/group, kh, kw]``
    weight -- read upcast to float64, written back down to each weight's
    own original dtype, exactly like
    :func:`onnxsim.apply_sparsegpt_pruning`'s own ``_to_f64``/``_from_f64``
    convention (though note SparseGPT's Hessian-compensated update, unlike
    plain masking, *recomputes* every kept entry's own value, so a fp16/bf16
    weight's surviving entries do not reproduce their pre-pruning bit
    pattern the way a plain-masking C++ port's FLOAT16/BFLOAT16 support
    does). Now at full, verified parity with the pure-Python
    :func:`onnxsim.apply_sparsegpt_pruning`, which is itself a thin alias
    for this function (see that function's own docstring) -- Conv's own
    im2col cross-covariance Hessian (``H = patches.T @ patches``, with a
    genuinely per-group Hessian and column-processing split for grouped/
    depthwise Conv) was verified numerically against the pure-Python
    original across ordinary/depthwise/general-grouped Conv, unstructured
    and N:M sparsity, ``auto_pad``, and dilation (see
    ``tests/test_sparsegpt_pruning_cpp.py``'s own Conv section), despite
    that Python original having no correct upstream SparseGPT reference of
    its own to port from (see its own module docstring for the three
    independent verification legs that make it trustworthy ground truth
    regardless). A Conv node whose spatial attributes are malformed (a
    ``kernel_shape`` disagreeing with the weight's own shape, an
    unrecognized ``auto_pad``, non-positive ``strides``/``dilations``, or a
    malformed explicit ``pads``) is left completely untouched, never
    guessed at, exactly like a layer with no observed calibration
    activation.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of entries to zero (shared per
            ``proc_block_size``-wide column block, not per row -- the one
            deliberate departure from every other unstructured-pruning
            function this port faithfully reproduces, matching the
            reference SparseGPT implementation's own behavior), ignored
            when ``n``/``m`` are given
    :param n: keep the ``n`` highest-importance entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4, per-row).
            Must be given together with ``m``.
    :param m: group size for N:M pruning; see ``n``
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added to every diagonal entry before inversion), matching
            :func:`onnxsim.apply_sparsegpt_pruning`'s own default
    :param proc_block_size: column-processing block size -- both the
            lazy-update granularity (how many columns' errors accumulate
            locally before a full cross-block update) and, for
            unstructured sparsity only, the width each shared per-block
            threshold is computed over
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations (passed to the shared
            :func:`onnxsim.onnx_simplifier._get_model_executor` process-wide
            executor, the same one :func:`onnxsim.simplify` itself uses)
    :returns: ``model`` with every matched layer's weight rewritten in
            place to the target pattern; a MatMul/Gemm/Attention layer with
            no observed 2-D calibration activation (dead input, an
            otherwise-empty ``calibration_data``, or every batch's
            activation isn't FLOAT32/FLOAT16/BFLOAT16), or a Conv layer
            with malformed spatial attributes or no observed usable 4-D
            activation for any one of its groups (a grouped/depthwise Conv
            is never partially pruned), is left completely untouched --
            there is no data-free fallback for SparseGPT
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_structured_wanda_pruning_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_sparsegpt_pruning(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            sparsity,
            n,
            m,
            percdamp,
            proc_block_size,
        )
    )


def apply_wanda_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    epsilon: float = 1e-8,
    global_sparsity: bool = False,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_wanda_pruning`: Wanda pruning
    (Sun et al., 2023, "A Simple and Effective Pruning Approach for Large
    Language Models", https://arxiv.org/abs/2306.11695) -- the calibration-
    driven upgrade of :func:`onnxsim.apply_magnitude_pruning`'s data-free
    baseline, zeroing the least-important entries of every matched layer's
    constant 2-D weight to an unstructured or N:M sparsity pattern using
    ``|W_ij| * ||X_j||_2`` (weight magnitude times its reduction-dimension
    entry's CALIBRATED activation L2-norm) as the importance metric, instead
    of plain ``|W|`` -- a one-shot static score, unlike
    :func:`onnxsim.apply_sparsegpt_pruning_cpp`'s own sequential Hessian-
    error-compensating algorithm (this pass never recomputes a *kept*
    entry's value, only zeros dropped ones, exactly like
    :func:`onnxsim.prune_magnitude_cpp`). Same real
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed calibration
    machinery as :func:`onnxsim.apply_structured_wanda_pruning_cpp`/
    :func:`onnxsim.apply_sparsegpt_pruning_cpp` (see either function's own
    docstring for the general pattern).

    Matches every plain ``MatMul``/vanilla-``Gemm`` node with a constant 2-D
    FLOAT32/FLOAT16/BFLOAT16 weight (this already includes
    ``com.microsoft::GroupQueryAttention``'s own separate Q/K/V projections
    -- ordinary MatMul/Gemm nodes feeding it, not a weight the op itself
    owns), every ``com.microsoft::Attention`` node's constant 2-D
    FLOAT32/FLOAT16/BFLOAT16 merged QKV weight, and every 2-D ``Conv``
    node's constant 4-D FLOAT32/FLOAT16/BFLOAT16 weight -- ordinary
    (``group=1``), fully depthwise (``group == in_channels ==
    out_channels``), and general grouped (``1 < group < in_channels``)
    alike (read out upcast to float64, written back down to the original
    dtype, exactly mirroring the pure-Python
    :func:`onnxsim.apply_wanda_pruning`'s own ``_to_f64``/``_from_f64``
    round trip -- masking never recomputes a surviving entry's own value, so
    this reproduces its exact original bit pattern). TRUE parity with the
    pure-Python :func:`onnxsim.apply_wanda_pruning` on every one of these
    candidate families, including Conv's own from-scratch im2col
    per-receptive-field-offset activation norm (``_conv_patch_sq_sum``'s C++
    mirror, ``ConvPatchSqSum``) and grouped/depthwise group-relative norm
    expansion (``_conv_group_relative_norm``'s C++ mirror,
    ``ConvGroupRelativeNorm`` -- see ``structured_pruning_entry.h``'s own
    ``ApplyWandaPruning`` declaration comment for the full mechanism).

    TRUE parity now also covers a prior gap unrelated to Conv: a
    full-regression check (every existing MatMul/Gemm/Attention candidate's
    live output against the pure-Python reference, not just Conv coverage)
    once surfaced a genuine, pre-existing divergence -- this port's own
    calibration statistic (``WandaCalibrationStats``, shared with
    :func:`onnxsim.apply_structured_wanda_pruning_cpp`/
    ``ApplyAttentionHeadWandaPruning``) used to compute a real
    per-channel-axis activation norm for a MatMul/Gemm candidate's
    activation at ANY rank, whereas the pure-Python reference requires
    exactly rank 2 for that same statistic and falls back to plain
    magnitude for anything else (e.g. a rank-3, batched/sequence activation
    feeding a plain 2-D MatMul weight). That gap is now closed --
    ``WandaCalibrationStats`` takes a ``require_rank2`` set (this pass'
    plain MatMul/Gemm candidates only; its Attention candidates keep the
    any-rank->=2 treatment the pure-Python reference's own separate
    ``attn_act_norm`` statistic always gave them, and the other two callers
    above pass no such set at all, matching their own Python references'
    complete lack of a rank restriction) -- see ``structured_pruning_entry.h``'s
    own ``ApplyWandaPruning`` declaration comment for the full writeup and
    ``tests/test_wanda_pruning_cpp.py``'s own module docstring for the
    regression test that caught the original gap.

    Unlike :func:`onnxsim.apply_sparsegpt_pruning_cpp` (which has no data-
    free fallback at all -- its entire mechanism IS the Hessian), a matched
    layer with NO observed calibration activation for its own input (dead
    input, an otherwise-empty ``calibration_data``, or every batch's
    activation isn't FLOAT32/FLOAT16/BFLOAT16) still gets pruned here, just
    to PLAIN MAGNITUDE importance (``|W_ij|`` alone) instead -- mirrors the
    pure-Python :func:`onnxsim.apply_wanda_pruning`'s own per-layer
    fallback exactly.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's activation norm on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each row's entries to zero, ignored
            when ``n``/``m`` are given
    :param n: keep the ``n`` highest-importance entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4, per output
            row). Must be given together with ``m``; incompatible with
            ``global_sparsity``.
    :param m: group size for N:M pruning; see ``n``
    :param epsilon: floor applied to the accumulated per-channel activation
            norm, avoiding every entry of an all-zero channel tying at
            exactly-zero importance
    :param global_sparsity: pools every matched layer's own importance into
            one ranking across the whole model and picks a single
            keep-count from ``sparsity``'s fraction of that pooled total,
            mirroring the pure-Python :func:`onnxsim.apply_wanda_pruning`'s
            own ``global_sparsity`` mode (see that function's own docstring
            for the full mechanism and its honestly-noted cross-layer-scale
            caveat). Incompatible with ``n``/``m``.
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations (passed to the shared
            :func:`onnxsim.onnx_simplifier._get_model_executor` process-wide
            executor, the same one :func:`onnxsim.simplify` itself uses)
    :returns: ``model`` with every matched layer's weight zeroed in place to
            the target pattern; a layer with no observed calibration
            activation falls back to plain-magnitude importance rather than
            being left untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_structured_wanda_pruning_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_wanda_pruning(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            sparsity,
            n,
            m,
            epsilon,
            global_sparsity,
        )
    )


def apply_imatrix_quantization_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[backend.Provider]] = None,
    block_size: int = 32,
    num_scale_candidates: int = 41,
    scale_search_range: Tuple[float, float] = (0.4, 1.6),
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_imatrix_quantization`:
    llama.cpp's "importance matrix" (imatrix) applied to this repository's
    own plain block-wise INT4 weight quantizer -- see
    ``onnxsim/imatrix_quant.py``'s own module docstring for the technique
    (why mean-square activation is a sound per-channel importance proxy,
    and why this is a weighted grid-search extension of the plain block
    quantizer rather than a new GGUF block format).

    Same real calibration machinery as :func:`onnxsim.apply_wanda_pruning_cpp`/
    :func:`onnxsim.apply_sparsegpt_pruning_cpp` -- a live
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyImatrixQuantization`` in ``imatrix_quant_entry.h`` for the full
    scope, including why this calibration-driven pass follows
    ``ApplyWandaPruning``'s own protobuf-level shape rather than
    :func:`onnxsim.apply_quarot_cpp`'s data-free ``PredicateBasedPass`` one).

    Matches every plain ``MatMul``/vanilla-``Gemm`` node with a constant 2-D
    FLOAT32 weight (``Conv`` is out of scope, matching the pure-Python
    :func:`onnxsim.apply_imatrix_quantization`'s own scope decision -- see
    that function's own docstring for why).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute the
            importance matrix from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations (passed to the shared
            :func:`onnxsim.onnx_simplifier._get_model_executor` process-wide
            executor, the same one :func:`onnxsim.simplify` itself uses)
    :param block_size: elements per weight quantization block along ``K``
    :param num_scale_candidates: candidate scale factors evaluated per
            block -- see :func:`onnxsim.quantize_dequantize_int4_imatrix`
    :param scale_search_range: ``(low, high)`` multiplier range for the
            candidate scale grid -- see
            :func:`onnxsim.quantize_dequantize_int4_imatrix`
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched layer's weight replaced by its
            importance-weighted INT4 round trip, stored under a new
            initializer name (the original initializer is left in the
            graph, unused). A layer with a non-constant/non-2-D weight, a
            reduction dimension not divisible by ``block_size``, or whose
            activation was never observed as a FLOAT32 tensor across the
            calibration data, is left untouched.

    **Accepted divergence from the pure-Python
    :func:`onnxsim.apply_imatrix_quantization`.** Not required to be
    bit-for-bit identical: the pure-Python reference upcasts every weight
    and activation to float64 throughout, whereas this port reads/writes
    FLOAT32 protobuf tensors and accumulates calibration statistics in
    float64 internally only -- both are the exact same algorithm
    (:func:`onnxsim.quantize_dequantize_int4_imatrix`'s own weighted grid
    search, ported scalar-loop-for-scalar-loop, see
    ``onnxsim/passes/imatrix_quant.h``), so results are expected to track
    each other unusually closely, but a floating-point rounding difference
    at a scale-search tie is possible and not treated as a bug -- the same
    "two independently-correct, non-interchangeable entry points" contract
    :func:`onnxsim.apply_quarot_cpp`/:func:`onnxsim.apply_iq4_nl_quantization_cpp`
    already establish for this codebase's other ``*_cpp`` ports.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_wanda_pruning_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    scale_lo, scale_hi = scale_search_range
    return onnx.load_from_string(
        C.apply_imatrix_quantization(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            block_size,
            num_scale_candidates,
            scale_lo,
            scale_hi,
            list(skip_names) if skip_names is not None else [],
        )
    )


def apply_outlier_suppression_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    alpha: float = 0.5,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_outlier_suppression`: Outlier
    Suppression (Wei et al., 2022) "Gamma Migration" -- folds the
    SmoothQuant-style per-channel migration scale directly into every
    matched ``LayerNormalization``'s gamma/bias (adding zero runtime
    nodes), compensated by scaling every downstream MatMul/vanilla-Gemm
    consumer's weight rows by the same scale. Returns a float model --
    pass the result to a W8A8 quantizer (e.g.
    :func:`onnxsim.quantize_static`) to actually quantize it.

    Same real calibration machinery as
    :func:`onnxsim.apply_imatrix_quantization_cpp` -- a live
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyOutlierSuppression`` in ``outlier_suppression_entry.h`` for the
    full scope).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            LayerNormalization output channel's activation range on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param alpha: the migration strength, identical in meaning to
            :func:`onnxsim.apply_smoothquant_cpp`'s own ``alpha``
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value (and to the scale itself) before computing it
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched ``LayerNormalization`` migrated
            in place; a ``LayerNormalization`` with any non-MatMul/Gemm
            consumer (or whose output is itself a graph output) is left
            completely untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_imatrix_quantization_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_outlier_suppression(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            alpha,
            epsilon,
        )
    )


def apply_llm_int8_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    outlier_threshold: float = 6.0,
    epsilon: float = 1e-8,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_llm_int8`: LLM.int8()
    (Dettmers et al., 2022) -- decomposes every matched MatMul/vanilla-Gemm
    layer with a constant 2-D FLOAT32 weight into a float32 outlier part
    plus a vector-wise INT8 part computed via ``MatMulInteger`` (per-row
    activation scales at runtime, per-output-channel weight scales
    offline, uint8 activation at zero-point 128). Every rewritten layer
    keeps its original output tensor name.

    Same real calibration machinery as
    :func:`onnxsim.apply_outlier_suppression_cpp` -- a live
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyLlmInt8`` in ``llm_int8_entry.h`` for the full scope).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to find each
            layer's outlier channels on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param outlier_threshold: an input channel is treated as an outlier if
            its activation magnitude exceeds this anywhere in the
            calibration data (the paper's own default, ``6.0``)
    :param epsilon: floor applied to a zero row/weight-column max-abs
            value before dividing by it
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer replaced by its
            outlier/INT8 decomposition; layers with a non-constant,
            non-2-D weight, a non-2-D activation, no (or all) outlier
            channels, or an int32-unsafe non-outlier reduction depth, are
            left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_outlier_suppression_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_llm_int8(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            outlier_threshold,
            epsilon,
        )
    )


def apply_spqr_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 16,
    outlier_fraction: float = 0.01,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_spqr`: SpQR
    (Dettmers et al., 2023) -- outlier-aware block-wise INT4 quantization
    where per-element outliers (by Hessian-diagonal-weighted sensitivity)
    are excluded from their own block's scale computation and stored
    instead as an exact sparse correction.

    Same real calibration machinery as :func:`onnxsim.apply_llm_int8_cpp`
    -- a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see ``ApplySpqr``
    in ``spqr_entry.h`` for the full scope, including its one documented
    divergence: outlier-position storage order can differ from the pure
    Python reference, though the SET of outlier positions and the
    reconstructed model are numerically identical).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            weight element's sensitivity score -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per quantization block along ``K``
    :param outlier_fraction: fraction of each layer's weight elements (by
            count) excluded from block-scale computation and stored as an
            exact sparse correction instead
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by
            block-wise INT4 codes plus a sparse outlier correction.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_spqr(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            block_size,
            outlier_fraction,
        )
    )


def apply_rptq_reorder_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_clusters: int = 4,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> Tuple[onnx.ModelProto, Dict[str, RptqLayerInfo]]:
    """
    C++-backed port of :func:`onnxsim.apply_rptq_reorder`: RPTQ (Yuan et
    al., 2023) -- clusters every matched MatMul/vanilla-Gemm layer's input
    channels by their own calibration abs-max (a plain Lloyd's-algorithm
    k-means) and permutes them -- plus the weight's matching ``K``-axis
    rows -- so same-cluster channels sit contiguously, via a new ``Gather``
    inserted before the layer. An exact reordering, not a quantization --
    see this module's own :func:`apply_rptq_reorder` and
    ``onnxsim/rptq.py``'s module docstring for the full technique.

    Same real calibration machinery as :func:`onnxsim.apply_spqr_cpp` -- a
    live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyRptqReorder`` in ``rptq_entry.h`` for the full scope, including
    its one documented divergence: the k-means fit's own RNG does not
    reproduce numpy's PCG64 bit-for-bit, though the reordered model's own
    correctness -- an exact permutation identity -- never depends on
    clustering quality).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's calibration range on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied) and for the k-means centroid
            tie-breaking
    :param num_clusters: number of channel clusters to reorder each matched
            layer's input into
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: a ``(model, layer_info)`` pair, matching
            :func:`onnxsim.apply_rptq_reorder`'s own return contract
            exactly -- ``layer_info`` maps each matched layer's original
            activation input name to a reconstructed, real, public
            :class:`onnxsim.rptq.RptqLayerInfo`.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    model_bytes, layers = C.apply_rptq_reorder(
        _get_model_executor(providers),
        model.SerializeToString(),
        calibration_data_pb,
        seed,
        num_clusters,
    )
    out_model = onnx.load_from_string(model_bytes)
    layer_info: Dict[str, RptqLayerInfo] = {
        x_name: RptqLayerInfo(
            x_name=x_name,
            w_name=w_name,
            gather_output=gather_output,
            permutation=np.asarray(permutation, dtype=np.int64),
            cluster_bounds=list(cluster_bounds),
        )
        for x_name, w_name, gather_output, permutation, cluster_bounds in layers
    }
    return out_model, layer_info


def inject_lora_cpp(
    model: Union[str, onnx.ModelProto],
    rank: int = 8,
    alpha: Optional[float] = None,
    target_op_types: Sequence[str] = ("MatMul", "Gemm", "Conv"),
    target_names: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> "Tuple[onnx.ModelProto, LoraAdapter]":
    """
    C++-backed port of :func:`onnxsim.lora.inject_lora`: splices a trainable
    low-rank ``X @ A @ B`` branch around every eligible ``MatMul``/``Gemm``/
    ``Conv`` weight, leaving the base weight itself untouched and every
    other byte of the model unchanged. See this module's own
    :func:`onnxsim.lora.inject_lora` and ``onnxsim/lora.py``'s module
    docstring for the full technique.

    Data-free and single-model, unlike :func:`onnxsim.apply_rptq_reorder_cpp`
    above -- no executor or calibration data crosses into C++ (see
    ``InjectLora`` in ``lora_entry.h`` for the full scope, including its one
    documented divergence: ``A``'s Kaiming-normal initializer is drawn from
    a ``std::mt19937_64`` stream, not numpy's PCG64-backed
    ``default_rng``, so the same ``seed`` does not reproduce the Python's
    exact ``A`` values -- immaterial to injected-model behavior since ``B``
    always starts at zero, so the branch is a numeric no-op until trained
    regardless of what ``A`` drew).

    :param model: the model to inject into, or a file path.
    :param rank: the adapter's inner dimension.
    :param alpha: when given, the branch is scaled by ``alpha / rank``
            before being added to the base branch's output; when ``None``,
            the branch is added unscaled.
    :param target_op_types: restrict injection to these op types.
    :param target_names: restrict injection to weights with these
            initializer names; ``None`` means every eligible node.
    :param seed: seeds ``A``'s Kaiming-normal initialization (see this
            function's own divergence note above).
    :returns: ``(model with adapters injected, the injected LoraAdapter)``,
            matching :func:`onnxsim.lora.inject_lora`'s own return contract
            exactly -- reconstructed into the real, public
            :class:`onnxsim.lora.LoraAdapter`/:class:`onnxsim.lora.LoraTarget`.
    """
    # Imported lazily (inside the function body, not at module scope) to
    # avoid a circular import: onnxsim.lora itself imports from onnxsim.nf4,
    # which imports quantize_weight_only_nf4_cpp back from this module --
    # importing onnxsim.lora at this module's own top level would deadlock
    # the import machinery the same way onnxsim.gptq's own lazy
    # `from onnxsim.onnx_simplifier import apply_gptq_cpp` avoids.
    from .lora import LoraAdapter, LoraTarget

    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    model_bytes, targets = C.inject_lora(
        model.SerializeToString(),
        rank,
        alpha,
        list(target_op_types),
        target_names is not None,
        list(target_names) if target_names is not None else [],
        seed,
    )
    out_model = onnx.load_from_string(model_bytes)
    adapter = LoraAdapter(
        targets=[
            LoraTarget(
                weight_name=weight_name,
                node_output=node_output,
                op_type=op_type,
                lora_a_name=lora_a_name,
                lora_b_name=lora_b_name,
                rank=int(rank_out),
                alpha=alpha_out,
            )
            for weight_name, node_output, op_type, lora_a_name, lora_b_name, rank_out, alpha_out in targets
        ]
    )
    return out_model, adapter


def apply_spinquant_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_spinquant`: SpinQuant (Liu et
    al., 2024), "R1-only" variant -- fits a single dense ``[K, K]``
    rotation per matched MatMul/vanilla-Gemm layer as the eigenvector basis
    of that layer's own calibration-activation covariance (a closed-form
    substitute for SpinQuant's own learned, Cayley-manifold-optimized
    rotation), conjugates the weight by it, then block-wise INT4-quantizes
    the result. See this module's own :func:`apply_spinquant` and
    ``onnxsim/spinquant.py``'s module docstring for the full technique.

    Same real calibration machinery as :func:`onnxsim.apply_spqr_cpp` -- a
    live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplySpinquant`` in ``spinquant_entry.h`` for the full scope,
    including its one documented divergence: the eigendecomposition here is
    a hand-rolled Jacobi eigenvalue algorithm rather than LAPACK's own, so
    individual rotation columns are not expected to match the Python
    reference sign-for-sign or order-for-order -- only that the fitted
    rotation is orthogonal and the reconstructed model's numerical behavior
    is comparably close).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches used to fit each
            layer's own rotation -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per quantization block along ``K``
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight replaced by
            rotated, INT4-quantized versions (plus the original bias, if
            any); output tensor name unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_spinquant(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            block_size,
        )
    )


def apply_paroquant_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    alpha: float = 0.5,
    num_angle_steps: int = 9,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_paroquant`: ParoQuant (Liang et
    al., 2025) -- combines a SmoothQuant-style per-channel scale with many
    independent, cheap 2x2 (pairwise, Givens) rotations on fixed adjacent-
    -channel pairs within each quantization block -- each angle grid-
    -searched against its own block's INT4 reconstruction error -- instead
    of :func:`onnxsim.apply_spinquant`'s single dense rotation, then block-
    -wise INT4-quantizes the result. See this module's own
    :func:`apply_paroquant` and ``onnxsim/paroquant.py``'s module docstring
    for the full technique.

    Same real calibration machinery as :func:`onnxsim.apply_spqr_cpp` -- a
    live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyParoquant`` in ``paroquant_entry.h`` for the full scope -- unlike
    :func:`onnxsim.apply_spinquant_cpp`'s eigendecomposition, the per-pair
    angle search here has no RNG/LAPACK-equivalent algorithm choice to
    diverge on, so fitted angles are expected to agree closely with the
    Python reference).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches used to measure
            each layer's own input channel activation range for the
            SmoothQuant-style scale -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per quantization block along ``K``, and per
            pairwise-rotation grouping (must be even)
    :param alpha: the channel-scale migration strength, same meaning and
            default as :func:`onnxsim.apply_smoothquant`'s own ``alpha``
    :param num_angle_steps: grid points per pair for the Givens angle
            search, evenly spaced over ``[-pi/4, pi/4]`` inclusive
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value before computing the SmoothQuant-style scale
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight and activation
            replaced by scaled-and-pairwise-rotated, INT4-quantized
            versions (plus the original bias, if any); output tensor name
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_paroquant(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            block_size,
            alpha,
            num_angle_steps,
            epsilon,
        )
    )


def apply_duquant_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    outlier_fraction: float = 0.05,
    epsilon: float = 1e-12,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_duquant`: DuQuant (Lin et al.,
    2024) -- a calibration-ranked permutation that redistributes each
    matched MatMul/vanilla-Gemm layer's worst outlier input channels
    one-per-block across the quantization grouping, composed with an
    independent Haar-random orthogonal rotation applied *within* each
    block, then INT4 round-to-nearest quantization of *both* the weight
    (offline, block-wise) and the activation (data-free, per-token, at
    graph-run time). See this module's own :func:`apply_duquant` and
    ``onnxsim/duquant.py``'s module docstring for the full technique.

    Same real calibration machinery as :func:`onnxsim.apply_spinquant_cpp`
    -- a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see ``ApplyDuquant``
    in ``duquant_entry.h`` for the full scope, including its own accepted
    numerical scope: this port seeds a fresh RNG per matched layer -- rather
    than reproducing ``apply_duquant``'s own single, sequentially-advancing
    ``numpy.random.Generator`` thread across every layer and block -- and
    its own block-local rotation is a Gram-Schmidt construction, not
    ``_random_orthogonal_matrix``'s sign-corrected QR, so individual
    rotation columns and near-tied outlier-channel assignments are not
    expected to match the Python reference exactly; the permute-rotate-
    -then-quantize composition stays exact before quantization for any
    orthogonal rotation regardless).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches used to rank each
            layer's own input channels by outlier magnitude -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied) and for the block-local
            rotation matrices
    :param block_size: elements per quantization block along ``K``,
            matching :func:`onnxsim.quantize_weight_only_int4`'s own
            default
    :param outlier_fraction: fraction of a layer's own input channels (by
            count) ranked as outliers and redistributed one-per-block
            across the permutation
    :param epsilon: floor applied to a token's own max-abs activation value
            before using it as a scale, avoiding a divide-by-zero on an
            all-zero token
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight and activation
            replaced by permuted-and-rotated, INT4-quantized versions (plus
            the original bias, if any); output tensor name unchanged.
            Layers with a non-constant, non-2-D weight, a reduction
            dimension not divisible by ``block_size``, or no calibration
            activation available, are left untouched; a model with no
            matching layer, or an opset older than 21, is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_duquant(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            seed,
            block_size,
            outlier_fraction,
            epsilon,
        )
    )


def quantize_weight_only_pb_llm_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    salient_ratio: float = 0.15,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_pb_llm`: PB-LLM
    (Shang et al., 2024, ICLR) -- a structured mixed-precision binarizer:
    per matched layer, the ``salient_ratio`` fraction of input channels
    with the highest Hessian-diagonal-weighted magnitude stay INT8, every
    other channel is binarized to ~1 bit/element.

    Same real calibration machinery as :func:`onnxsim.apply_llm_int8_cpp`
    -- a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see ``ApplyPbLlm``
    in ``pb_llm_entry.h`` for the full scope).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian diagonal from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param salient_ratio: fraction of each layer's input columns (by
            count, most salient first) kept at INT8; the rest are
            binarized
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by
            ``Mul(Cast(Code), Scale)`` feeding the original MatMul/Gemm
            node.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.quantize_weight_only_pb_llm(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            salient_ratio,
        )
    )


def quantize_weight_only_squeezellm_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    bits: int = 4,
    outlier_fraction: float = 0.0045,
    num_kmeans_iterations: int = 20,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_squeezellm`:
    SqueezeLLM (Kim et al., 2023) -- sensitivity-weighted per-group
    codebook (a real ``GatherND``-based graph rewrite, not folded to a
    single initializer) plus a dense-and-sparse outlier correction.

    Same real calibration machinery as :func:`onnxsim.apply_llm_int8_cpp`
    -- a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplySqueezeLlm`` in ``squeezellm_entry.h`` for the full scope). This
    technique's own weighted k-means fit initializes deterministically (no
    RNG anywhere), so this port is expected to track the Python reference
    numerically closely, not just structurally.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's sensitivity on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: elements per ``(output channel, block)`` codebook
            group along the reduction dimension
    :param bits: codebook size is ``2 ** bits`` centroids per group
    :param outlier_fraction: fraction of weight elements (by magnitude,
            across the whole tensor) excluded from the k-means fit and
            corrected back to their exact original value instead
    :param num_kmeans_iterations: weighted Lloyd's-algorithm iterations
            refining each group's codebook
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by a
            per-group codebook lookup plus a sparse outlier correction.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.quantize_weight_only_squeezellm(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            block_size,
            bits,
            outlier_fraction,
            num_kmeans_iterations,
        )
    )


def apply_billm_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 128,
    percdamp: float = 0.01,
    max_salient_search: int = 30,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_billm`: BiLLM
    (Huang et al., 2024, ICML) -- a genuine ~1-bit-average weight
    binarizer: Hessian-guided salient-column selection, a two-level binary
    residual approximation for salient columns, plain flat binary for the
    rest, and OBC-style forward error compensation.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` -- a
    live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see ``ApplyBillm``
    in ``billm_entry.h`` for the full scope, including its accepted
    numerical scope: the dense inverse/Cholesky at this algorithm's heart
    -- reused from ``apply_gptq_cpp``'s own machinery -- use scalar
    double-precision kernels rather than LAPACK).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param block_size: BiLLM's own salient-column-search/OBC-compensation
            block width
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added before inversion)
    :param max_salient_search: bounds how many leading (most Hessian-
            salient) columns of a block the search tries as "the salient
            group"
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by a
            two-level binary residual (salient columns) or a flat binary
            level (non-salient columns).
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_billm(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            block_size,
            percdamp,
            max_salient_search,
        )
    )


def quantize_kv_cache_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    value_output_names: Optional[Sequence[str]] = None,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_kv_cache`: per-channel
    static INT8 for a matched ``Concat(past, new, axis=seq)`` stream's own
    Key-style values, per-token dynamic INT8 (data-free) for Value-style
    ones.

    Same real calibration machinery as :func:`onnxsim.apply_llm_int8_cpp`
    -- a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyKvCacheQuantization`` in ``kv_cache_quantization_entry.h`` for
    the full scope).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            Key-style stream's own per-channel scale from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param value_output_names: which matched streams' ``present`` output
            names get Value-style (per-token) treatment instead of the
            default Key-style (per-channel) treatment; when omitted, a
            stream whose ``present`` output name contains ``".value"`` is
            treated as Value-style instead
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched KV-cache stream quantized to
            INT8.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.quantize_kv_cache(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            list(value_output_names) if value_output_names is not None else [],
        )
    )


def apply_owq_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    outlier_fraction: float = 0.01,
    percdamp: float = 0.01,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_owq`: OWQ (Lee, Park, Kim, Kim
    and Sung, 2023, AAAI 2024) -- rescues the top ``outlier_fraction``
    OBS-salient columns of an already-``quantize_weight_only_int4``-quantized
    layer back to exact float32 precision via an additive
    ``Gather``/``MatMul``/``Add`` correction; ``quantized_model``'s own
    INT4 codes are never modified.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` -- a
    live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the float model in C++ (see
    ``ApplyOwq`` in ``owq_entry.h`` for the full scope, including its
    accepted numerical scope: the dense inverse/Cholesky at this
    algorithm's heart -- reused from ``apply_gptq_cpp``'s own machinery --
    use scalar double-precision kernels rather than LAPACK).

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to compute each
            layer's Hessian and per-column error from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param outlier_fraction: fraction of each layer's input columns to
            restore to full precision
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added before inversion)
    :param providers: onnxruntime execution providers to run
            ``float_model`` on when capturing calibration activations
    :returns: ``quantized_model`` with a new ``Gather``/``MatMul``/``Add``
            correction inserted after every matched layer.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_owq(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            outlier_fraction,
            percdamp,
        )
    )


def apply_gear_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    rank: int = 4,
    outlier_fraction: float = 0.05,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gear`: GEAR (Kang et al., 2024)
    -- low-rank-plus-sparse residual compensation layered on top of
    :func:`onnxsim.quantize_kv_cache`'s own static per-channel INT8 base
    quantization, applied only to a freshly-produced KV-cache token.

    Same real calibration machinery as :func:`onnxsim.apply_llm_int8_cpp`
    -- a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see ``ApplyGear``
    in ``gear_entry.h`` for the full scope, including its accepted
    numerical scope: the reconstructed low-rank projector matches the
    Python reference's own up to ordinary floating-point rounding, not
    sign-for-sign for individual singular vectors).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to fit each
            matched stream's own low-rank projector and sparse mask from --
            see :func:`onnxsim.generate_random_calibration_data` (the
            default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param rank: the low-rank correction's own rank (clamped to
            ``min(rank, head_dim, num_calibration_rows)`` per stream)
    :param outlier_fraction: fraction of each stream's channels kept as an
            explicit sparse correction after the low-rank term is
            subtracted
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched KV-cache stream's own
            reconstruction error corrected by a low-rank-plus-sparse term.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_gear(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            rank,
            outlier_fraction,
        )
    )


def apply_rotatekv_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_rotatekv`: RotateKV (Su et al.,
    2025) -- fits a per-stream orthogonal rotation from a matched KV-cache
    stream's own calibration-activation covariance and applies it to both
    the stream's fresh Key and its compensating Query, exact by
    construction for any orthogonal ``R``.

    Same real calibration machinery as :func:`onnxsim.apply_llm_int8_cpp`
    -- a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyRotateKv`` in ``rotatekv_entry.h`` for the full scope, including
    its permanent divergence: the fitted rotation is the eigenvector basis
    of a from-scratch Jacobi eigensolver, not LAPACK's own ``eigh``, so the
    exact basis differs from the pure Python reference's own -- both are
    valid orthogonal rotations with the same reconstruction-quality
    property).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to fit each
            matched stream's own rotation from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched KV-cache stream's own fresh Key
            and compensating Query rotated by a fitted orthogonal matrix.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_rotatekv(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
        )
    )


def apply_gptq_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gptq`: GPTQ (Frantar et al.,
    2022) sequential, Hessian-compensated rounding for every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model``,
    reusing that scheme's own per-block scales and changing only which
    integer each element rounds to.

    Same real calibration machinery as
    :func:`onnxsim.apply_llm_int8_cpp` -- a live
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the float model in C++ (see
    ``ApplyGptq`` in ``gptq_entry.h`` for the full scope, including its
    accepted numerical scope: the dense inverse/Cholesky at this
    algorithm's heart use scalar double-precision kernels rather than
    LAPACK, so codes can differ from the reference in rare rounding ties
    while reconstruction error tracks it).

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to compute each
            layer's Hessian from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added before inversion)
    :param proc_block_size: GPTQ's own column-processing block size (not
            the quantization scale's own block size, reused unchanged)
    :param providers: onnxruntime execution providers to run
            ``float_model`` on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its GPTQ-optimized codes.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_gptq(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            percdamp,
            proc_block_size,
        )
    )


def apply_gptaq_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gptaq`: GPTAQ (Li, Yin, Lee,
    Xiao, Panda, 2025) -- a small, closed-form asymmetric-calibration
    correction layered on top of :func:`onnxsim.apply_gptq_cpp`'s own
    sequential, Hessian-compensated rounding. See :mod:`onnxsim.gptaq`'s
    own module docstring for the full first-principles derivation: probes
    BOTH ``float_model`` (the true activation) AND ``quantized_model`` (the
    corrupted activation, reflecting whatever upstream layers' quantization
    already did) at each candidate layer's own input, folding the gap
    between them into GPTQ's own per-column procedure via one small, exact
    pre-computation.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through BOTH models in C++ (unlike
    ``apply_gptq_cpp``, which only ever probes the float model) -- see
    ``ApplyGptaq`` in ``gptaq_entry.h`` for the full scope, including its
    accepted numerical scope (shared with ``ApplyGptq``'s own: the dense
    inverse/Cholesky kernels at this algorithm's heart use scalar
    double-precision arithmetic rather than LAPACK, so codes can differ
    from the reference in rare rounding ties while reconstruction error
    tracks it).

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`, optionally already
            refined by other passes (e.g. :func:`onnxsim.apply_gptq_cpp`,
            :func:`onnxsim.correct_bias`) -- this is exactly what makes the
            asymmetric calibration meaningful
    :param calibration_data: representative input batches, run through
            *both* models to capture the true and corrupted activation at
            each candidate's input -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param percdamp: Hessian damping factor, matching
            :func:`onnxsim.apply_gptq_cpp`'s own parameter and default
    :param proc_block_size: GPTQ's own column-processing block size (not
            the quantization scale's own block size, reused unchanged)
    :param providers: onnxruntime execution providers to run both models on
            when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its GPTAQ-optimized codes.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_gptaq(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            percdamp,
            proc_block_size,
        )
    )


def apply_quantease_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_epochs: int = 4,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_quantease`: QuantEase (Behdin,
    Acharya, Gupta, Song, Zhu and Keerthi, 2023) -- plain cyclic coordinate
    descent for every ``quantize_weight_only_int4``-quantized MatMul/Gemm
    layer present (by node output name) in both ``float_model`` and
    ``quantized_model``, reusing that scheme's own per-block scales and
    changing only which integer each element rounds to. See this module's
    own :func:`apply_quantease` and ``onnxsim/quantease.py``'s module
    docstring for the full technique and its derivation relative to
    :func:`onnxsim.apply_gptq_cpp`/:func:`onnxsim.apply_adaround_cpp`.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` -- a
    live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the float model in C++ (see
    ``ApplyQuantease`` in ``quantease_entry.h`` for the full scope -- unlike
    :func:`onnxsim.apply_gptq_cpp`'s own Cholesky-inverse note, this port
    needs no matrix inversion/factorization at all, so there is no linear-
    -algebra routine whose algorithm choice could diverge from numpy's own).

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to compute each
            layer's Hessian from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_epochs: number of full cyclic sweeps over every column
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its QuantEase-optimized codes.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_quantease(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            num_epochs,
        )
    )


def apply_lqer_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    rank: int = 8,
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[backend.Provider]] = None,
    eps: float = 1e-6,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_lqer`: LQER (Zhang et al.,
    2024) -- an activation-weighted generalization of
    :func:`onnxsim.apply_low_rank_compensation_cpp`'s own plain
    (unweighted) SVD, row-scaling the residual by a calibration-measured
    per-input-channel activation RMS before the SVD (and un-scaling the
    result back afterward) so the low-rank fit spends its rank budget on
    the input channels calibration data actually drives, not every
    channel equally.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through ``float_model`` in C++ (see
    ``ApplyLqer`` in ``lqer_entry.h`` for the full scope, including its
    accepted numerical scope: this port's own SVD is a hand-rolled Jacobi
    solver rather than ``numpy.linalg.svd``'s own LAPACK routine -- see
    :func:`onnxsim.apply_low_rank_compensation_cpp`'s own identical note).

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`; this is the model
            that gets corrected and returned
    :param rank: the correction's rank ``r`` (clamped to
            ``min(r, N, K)`` per layer)
    :param calibration_data: representative input batches used to measure
            each layer's own per-input-channel activation RMS -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :param eps: floor applied to each per-channel RMS before dividing by
            it, avoiding a division blow-up on a channel calibration
            barely drives
    :returns: ``quantized_model`` with every matched layer's output summed
            with a new rank-``r`` activation-weighted correction term (two
            chained ``MatMul`` nodes plus an ``Add``); the layer's own
            existing INT4 weight/scale are left completely untouched. A
            layer whose activation was never observed as a plain 2-D
            tensor falls back to the same unweighted SVD
            :func:`onnxsim.apply_low_rank_compensation_cpp` uses.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_lqer(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            rank,
            eps,
        )
    )


def apply_norm_tweaking_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[backend.Provider]] = None,
    eps: float = 1e-6,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_norm_tweaking`: Norm Tweaking
    (Li, Xu, Ni, Chen, Ye, Sun, 2023) -- recalibrates every matched
    ``LayerNormalization`` node's own ``scale``/``bias`` parameters in
    ``quantized_model`` so its output distribution's per-channel mean and
    standard deviation, measured on ``calibration_data``, matches
    ``float_model``'s own (a closed-form moment-matching fit folded
    directly into a new ``scale``/``bias`` initializer pair -- no new
    graph nodes).

    Unlike every other calibration-driven ``*_cpp`` port that only probes
    ``float_model`` (e.g. :func:`onnxsim.apply_gptq_cpp`), this one runs
    BOTH ``float_model`` and ``quantized_model`` through a live
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor for every
    calibration batch, since what needs correcting is precisely the
    quantized model's own CURRENT output distribution (see ``ApplyNormTweaking``
    in ``norm_tweaking_entry.h`` for the full scope, including its accepted
    numerical scope).

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), e.g. from
            :func:`onnxsim.quantize_weight_only_int4` or any ``quantize_*``
            function
    :param calibration_data: representative input batches to measure each
            node's own float-vs-quantized output distribution on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run both models on
    :param eps: added to the measured quantized-side standard deviation
            before dividing, to avoid blowing up on a (near-)constant
            channel
    :returns: ``quantized_model`` with every matched ``LayerNormalization``
            node's ``scale``/``bias`` initializers replaced by tweaked
            copies (new initializers, uniquely named).
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_norm_tweaking(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            eps,
        )
    )


def apply_dac_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[backend.Provider]] = None,
    min_expected_error_reduction: float = 0.5,
    correction_threshold: float = 1e-12,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_dac`: D2Quant's
    Deviation-Aware Correction (Yan et al., 2026) -- measures, per channel
    and per ``LayerNormalization`` node, how much of that channel's own
    quantization-induced output deviation is a consistent, directional
    mean shift (as opposed to unstructured noise), and folds the shift
    directly into that same ``LayerNormalization``'s own bias for every
    channel where the shift dominates.

    Same dual-model calibration machinery as
    :func:`onnxsim.apply_norm_tweaking_cpp` -- a live
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor runs
    ``calibration_data`` through BOTH models in C++ (see ``ApplyDac`` in
    ``dac_entry.h`` for the full scope, including its accepted numerical
    scope). ``onnxsim/d2quant.py``'s own ``apply_dsq`` is a separate
    technique from the same paper, already delegating to its own verified
    C++ port (:func:`onnxsim.apply_dsq_cpp`) -- unaffected by this
    function.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path)
    :param calibration_data: representative input batches to measure each
            LayerNormalization's own deviation on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run both models on
    :param min_expected_error_reduction: only correct a channel whose
            measured deviation's own ``mu^2 / (mu^2 + sigma^2)`` reaches
            this fraction (in ``[0, 1)``) -- the paper's own closed-form
            expected squared-error reduction from applying the correction
    :param correction_threshold: skip a LayerNormalization whose largest
            per-channel correction (after the ratio gate above) never
            exceeds this in absolute value
    :returns: ``quantized_model`` with a per-channel mean-shift correction
            folded into every measurably-shifted, gated-in
            ``LayerNormalization``'s own bias
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_dac(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            min_expected_error_reduction,
            correction_threshold,
        )
    )


def apply_adaround_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    learning_rate: float = 0.1,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_adaround`: Nagel et al.
    (2020)'s AdaRound -- a rectified-sigmoid relaxation of each weight
    element's floor/ceil rounding decision, optimized by a hand-rolled
    Adam loop to minimize a layer's own reconstruction error against real
    calibration activations, rather than round-to-nearest's
    per-element-independent choice -- see ``onnxsim/adaround.py``'s own
    module docstring for the full technique.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the float model in C++ (see
    ``ApplyAdaround`` in ``adaround_entry.h`` for the full scope).

    Same accepted-numerical-scope class as
    :func:`onnxsim.apply_tesseraq_cpp` (not every other calibration-driven
    ``*_cpp`` port in this codebase, all either closed-form or
    closed-form-apart-from-RNG): this is an iterative Adam optimization,
    so cross-language floating-point agreement is measured empirically
    (see tests/test_adaround_cpp.py) rather than assumed, and this
    function is not aliased from :func:`onnxsim.apply_adaround`.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to optimize the
            rounding on -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied) -- this port has no RNG of
            its own, so ``seed`` only ever affects which calibration
            batches get generated
    :param num_iterations: Adam steps to run per layer
    :param learning_rate: Adam learning rate for the per-element rounding
            relaxation
    :param reg_param: weight of the regularization term that pulls each
            element's relaxation toward a hard 0/1 (floor/ceil) decision
    :param warm_start: fraction of ``num_iterations`` (from the start) run
            with the regularization term disabled
    :param beta_range: ``(beta_start, beta_end)`` for the regularization
            term's exponent, linearly annealed after ``warm_start``
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its AdaRound-optimized codes.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    beta_start, beta_end = beta_range
    return onnx.load_from_string(
        C.apply_adaround(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            num_iterations,
            learning_rate,
            reg_param,
            warm_start,
            beta_start,
            beta_end,
        )
    )


def apply_adaquant_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    weight_learning_rate: float = 0.1,
    activation_learning_rate: float = 0.01,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_adaquant`: Hubara, Nahshan,
    Hanani, Banner, Soudry (2020/2021)'s AdaQuant -- a per-layer joint
    Adam optimization over the same rectified-sigmoid weight-rounding
    relaxation :func:`onnxsim.apply_adaround_cpp` uses, PLUS the
    activation's own (scale, zero_point), against
    :func:`onnxsim.quantize_static`'s W8A8 QDQ scheme (not
    :func:`onnxsim.quantize_weight_only_int4`'s blocked-INT4 one) -- see
    ``onnxsim/adaquant.py``'s own module docstring for the full technique
    and its relationship to AdaRound.

    Same real calibration machinery as :func:`onnxsim.apply_adaround_cpp`
    -- a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the float model in C++ (see
    ``ApplyAdaquant`` in ``adaquant_entry.h`` for the full scope).

    Same accepted-numerical-scope class as :func:`onnxsim.apply_adaround_cpp`:
    this is an iterative Adam optimization over three parameter groups
    jointly, so cross-language floating-point agreement is measured
    empirically (see tests/test_adaquant_cpp.py) rather than assumed.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_static`. Layers quantized by any other
            scheme (including :func:`onnxsim.quantize_qoperator_gemm`'s
            QGemm format), or left unquantized, are left untouched.
    :param calibration_data: representative input batches to optimize
            against -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied) -- this port has no RNG of
            its own, so ``seed`` only ever affects which calibration
            batches get generated
    :param num_iterations: Adam steps to run per layer
    :param weight_learning_rate: Adam learning rate for the per-element
            weight-rounding relaxation (same role as
            :func:`onnxsim.apply_adaround_cpp`'s ``learning_rate``)
    :param activation_learning_rate: Adam learning rate for the
            activation's scale (optimized in log-space) and zero-point
    :param reg_param: weight of the regularization term that pulls each
            weight element's rounding relaxation toward a hard 0/1
            (floor/ceil) decision
    :param warm_start: fraction of ``num_iterations`` (from the start) run
            with the weight-rounding regularization term disabled
    :param beta_range: ``(beta_start, beta_end)`` for the weight-rounding
            regularization term's exponent, linearly annealed after
            ``warm_start``
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's weight INT8
            codes and activation (scale, zero_point) initializers rewritten
            to their jointly-optimized values.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_adaround_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    beta_start, beta_end = beta_range
    return onnx.load_from_string(
        C.apply_adaquant(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            num_iterations,
            weight_learning_rate,
            activation_learning_rate,
            reg_param,
            warm_start,
            beta_start,
            beta_end,
        )
    )


def apply_omniquant_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_clip_steps: int = 20,
    num_alpha_steps: int = 20,
    min_clip_ratio: float = 0.5,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_omniquant`: OmniQuant (Shao et
    al., 2023)'s Learnable Weight Clipping (LWC) and Learnable Equivalent
    Transformation (LET), applied to every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer shared (by
    node output name) between ``float_model`` and ``quantized_model`` --
    see ``onnxsim/omniquant.py``'s own module docstring for the full
    technique.

    Same real calibration machinery as :func:`onnxsim.apply_adaround_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the float model in C++ (see
    ``ApplyOmniquant`` in ``omniquant_entry.h`` for the full scope).

    Unlike :func:`onnxsim.apply_adaround_cpp`/:func:`onnxsim.apply_tesseraq_cpp`
    (iterative Adam optimizations), this is a bounded, deterministic
    coordinate-descent GRID search -- no RNG, no hand-rolled dense linear
    algebra -- so this port is expected to track the Python reference
    closely, including exact-tie behavior at every grid point boundary (see
    tests/test_omniquant_cpp.py for the measured agreement), and this
    function is not aliased from :func:`onnxsim.apply_omniquant`.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to search and
            measure the transform on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_clip_steps: grid points for the LWC clip ratio (see
            ``onnxsim/omniquant.py``'s own docstring for the grid layout)
    :param num_alpha_steps: grid points for the LET migration-strength
            exponent
    :param min_clip_ratio: the LWC grid's lower bound
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every layer OmniQuant measurably
            improved rewritten (its INT4 weight/scale, and -- only where
            LET was found to help -- a new ``Sub``/``Mul``/``Add`` inserted
            around it).
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_adaround_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_omniquant(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            num_clip_steps,
            num_alpha_steps,
            min_clip_ratio,
        )
    )


def apply_affinequant_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_clip_steps: int = 20,
    num_alpha_steps: int = 20,
    min_clip_ratio: float = 0.5,
    affine_block_size: int = 8,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_affinequant`: AffineQuant (Ma
    et al., 2024, ICLR)'s block-diagonal generalization of
    :func:`onnxsim.apply_omniquant_cpp`'s own Learnable Equivalent
    Transformation -- see ``onnxsim/affinequant.py``'s own module
    docstring for the full technique and how it generalizes OmniQuant's
    own diagonal transform.

    Same real calibration machinery, and same bounded-grid-search (not
    iterative Adam) numerical character, as
    :func:`onnxsim.apply_omniquant_cpp` -- see that function's own
    docstring. This port's own block-affine candidate additionally uses a
    hand-rolled cyclic Jacobi eigendecomposition (no LAPACK/BLAS linked
    into this codebase) to fit each block's own rotation; see
    ``affinequant_entry.h`` for this port's own accepted
    eigendecomposition-algorithm divergence. This function is not aliased
    from :func:`onnxsim.apply_affinequant`.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to search and
            measure the transform on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_clip_steps: grid points for the LWC clip ratio
    :param num_alpha_steps: grid points for the LET migration-strength
            exponent, used identically for both the diagonal and the
            block-affine candidate
    :param min_clip_ratio: the LWC grid's lower bound
    :param affine_block_size: the size of each independently-rotated block
            of activation channels in the block-affine candidate -- see
            ``onnxsim/affinequant.py``'s own docstring's "Scope" section
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every layer AffineQuant measurably
            improved rewritten (its INT4 weight/scale, and -- only where a
            diagonal or block-affine transform was found to help -- a new
            ``Sub``/(optionally ``MatMul``)/``Mul``/``Add`` inserted
            around it).
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_affinequant(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            num_clip_steps,
            num_alpha_steps,
            min_clip_ratio,
            affine_block_size,
        )
    )


def apply_brecq_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    blocks: Sequence[Tuple[str, str]],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    learning_rate: float = 0.1,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
    fisher_eps: float = 1e-3,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_brecq`: BRECQ (Li et al.,
    2021, ICLR)'s joint, Fisher-diagonal-weighted block reconstruction --
    see ``onnxsim/brecq.py``'s own module docstring for the full technique,
    its block-discovery contract, and what it simplifies relative to the
    paper.

    Same real calibration machinery as :func:`onnxsim.apply_adaround_cpp`,
    and the same accepted-numerical-scope class (an iterative Adam
    optimization, so cross-language floating-point agreement is measured
    empirically -- see tests/test_brecq_cpp.py -- rather than assumed),
    extended to jointly optimize an entire caller-delimited block of
    layers rather than one layer independently. This function is not
    aliased from :func:`onnxsim.apply_brecq`.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param blocks: ``(block_input_name, block_output_name)`` pairs, one per
            residual/linear block to jointly optimize -- see
            ``onnxsim/brecq.py``'s own docstring for exactly which
            topologies between those two tensor names are recognized
    :param calibration_data: representative input batches to optimize the
            rounding on -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_iterations: Adam steps to run per block
    :param learning_rate: Adam learning rate for the per-element rounding
            relaxation
    :param reg_param: weight of the regularization term that pulls each
            element's relaxation toward a hard 0/1 (floor/ceil) decision,
            applied independently per layer inside the block
    :param warm_start: fraction of ``num_iterations`` (from the start) run
            with the regularization term disabled
    :param beta_range: ``(beta_start, beta_end)`` for the regularization
            term's exponent, linearly annealed after ``warm_start``
    :param fisher_eps: numerical floor added to the empirical per-element
            variance before normalizing it into the Fisher-diagonal weight
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every discovered block's layers'
            INT4 weight initializers rewritten to their jointly-optimized
            codes.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    beta_start, beta_end = beta_range
    return onnx.load_from_string(
        C.apply_brecq(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            list(blocks),
            calibration_data_pb,
            num_iterations,
            learning_rate,
            reg_param,
            warm_start,
            beta_start,
            beta_end,
            fisher_eps,
        )
    )


def apply_mixed_precision_quantization_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    high_bits_fraction: float = 0.2,
    block_size: int = 32,
    sensitivity_metric: str = "hessian_diag",
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_mixed_precision_quantization`:
    quantizes every matched MatMul/vanilla-Gemm layer to block-wise INT8 or
    INT4, chosen per layer from a calibration-driven Hessian-diagonal (or
    full-Hessian) sensitivity score, budgeted so the top
    ``high_bits_fraction`` most-sensitive layers get INT8 -- see
    ``onnxsim/mixed_precision.py``'s own module docstring for the full
    technique.

    Same real calibration machinery as :func:`onnxsim.apply_llm_int8_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyMixedPrecisionQuantization`` in ``mixed_precision_entry.h`` for
    the full scope).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches used to measure
            each layer's own per-input-channel activation energy -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param high_bits_fraction: fraction of matched layers (by count, most
            sensitive first) that get block-wise INT8 instead of INT4
    :param block_size: elements per quantization block along ``K``, for
            both the INT4 and INT8 tiers
    :param sensitivity_metric: ``"hessian_diag"`` (the default) or
            ``"full_hessian"`` -- which per-layer sensitivity score to rank
            candidate layers by. Raises :class:`ValueError` for any other
            value.
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight replaced by
            block-wise INT4 or INT8 codes plus a per-block float32 scale;
            layers with a non-constant, non-2-D weight, a reduction
            dimension not divisible by ``block_size``, or no calibration
            activation available, are left untouched; a model with no
            matching layer, or an opset older than 21, is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if sensitivity_metric not in ("hessian_diag", "full_hessian"):
        raise ValueError(
            "sensitivity_metric must be one of ('hessian_diag', "
            f"'full_hessian'), got {sensitivity_metric!r}"
        )
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_mixed_precision_quantization(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            high_bits_fraction,
            block_size,
            sensitivity_metric,
        )
    )


def apply_smooth_attention_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_smooth_attention`: QoQ's
    SmoothAttention -- migrates Key's per-channel quantization difficulty
    into Query (which stays float) for every decomposed attention subgraph
    (``MatMul(Q,Kt) -> [Mul/Div] -> [Add] -> Softmax -> MatMul(_,V)``, the
    same pattern :func:`onnxsim.apply_attention_quantization` matches) --
    see ``onnxsim/qoq.py``'s own module docstring, point "2. SmoothAttention",
    for the full technique. Returns a float model, provably equivalent to
    the input up to floating-point rounding -- no quantization happens
    here at all.

    Same real calibration machinery as :func:`onnxsim.apply_llm_int8_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplySmoothAttention`` in ``smooth_attention_entry.h`` for the full
    scope, including why it transcribes rather than reuses
    ``attention_quantization.h``'s own Node/Value-IR matcher).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            Key head-dim channel's activation range on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param epsilon: floor applied to every per-channel Key max-abs value
            before dividing by it
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched subgraph's ``Q`` operand
            multiplied by a new per-head-dim-channel scale ``s`` and its
            ``Kt`` (Key, transposed) operand divided by the same ``s``; a
            subgraph whose Key tensor never appeared as a plain-enough
            (rank >= 2) probe, or a model with no matching subgraph at
            all, is left untouched for that subgraph (or returned
            unchanged, respectively).
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_smooth_attention(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            epsilon,
        )
    )


def apply_ptq4vit_quantization_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_calibration_samples: int = 8,
    seed: int = 0,
    providers: Optional[Sequence[backend.Provider]] = None,
    n_levels: int = 256,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_ptq4vit_quantization`: PTQ4ViT's
    twin uniform quantization -- applied to every standalone ``Softmax``
    output and every GELU output (a standalone ``Gelu`` node, or the
    standard ``Erf``-decomposed GELU's final ``Mul``) -- see
    ``onnxsim/ptq4vit.py``'s own module docstring for the full technique.

    Same real calibration machinery as :func:`onnxsim.apply_llm_int8_cpp`,
    except this pass captures each matched tensor's own REAL VALUES (not a
    reduction) to run PTQ4ViT's own reconstruction-error split search
    against -- see ``ApplyPtq4Vit`` in ``ptq4vit_entry.h`` for the full
    scope.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to observe each
            matched tensor's real values from -- see
            :func:`onnxsim.calibration.generate_random_calibration_data`
            (the default when omitted)
    :param num_calibration_samples: number of random batches to generate
            when ``calibration_data`` is not supplied
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param providers: onnxruntime execution providers to run calibration on
    :param n_levels: number of uniform levels *each* sub-quantizer uses
    :returns: ``model`` with every matched tensor's consuming nodes
            rewired to read the twin-uniform quantize-dequantize round
            trip of it instead. A model with no matching ``Softmax``/GELU
            pattern, or an opset older than 11, is returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_calibration_samples, seed=seed
        )
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_ptq4vit_quantization(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            n_levels,
        )
    )


def apply_autoround_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    learning_rate: float = 0.1,
    clip_learning_rate: float = 0.03,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
    clip_ratio_range: Tuple[float, float] = (0.5, 1.5),
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_autoround`: Cheng et al.
    (2023)'s AutoRound -- jointly optimizes AdaRound's own rectified-sigmoid
    rounding relaxation AND a second, per-(output-channel, block) clip-ratio
    parameter that lets the effective scale move during the same Adam
    optimization, closing the one gap between
    :func:`onnxsim.apply_adaround_cpp` (fixed-scale, rounding-only) and
    AutoRound proper -- see ``onnxsim/autoround.py``'s own module docstring
    for the full technique.

    Same real calibration machinery as :func:`onnxsim.apply_adaround_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the float model in C++ (see
    ``ApplyAutoround`` in ``autoround_entry.h`` for the full scope).

    Same accepted-numerical-scope class as :func:`onnxsim.apply_adaround_cpp`
    (an iterative Adam optimization, so cross-language floating-point
    agreement is measured empirically -- see tests/test_autoround_cpp.py --
    rather than assumed), plus one extra divergence source unique to
    AutoRound: because the scale itself moves every step, an element whose
    ratio sits within an ulp of an integer boundary can take a different
    quantization bin between the two implementations. This port's own
    safety net (always also running AdaRound's own fixed-scale
    optimization and keeping whichever candidate has the lower measured
    reconstruction error) bounds the practical impact: this function can
    never regress reconstruction error below :func:`onnxsim.apply_adaround_cpp`'s
    own verified-agreeing optimum for the same layer and calibration data.

    Unlike :func:`onnxsim.apply_adaround_cpp` (which never rewrites a
    layer's scale), a layer whose joint optimization beats the AdaRound-only
    safety net also gets its scale initializer rewritten to the optimized
    per-block scale.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to optimize
            against -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_iterations: Adam steps to run per layer
    :param learning_rate: Adam learning rate for the per-element rounding
            relaxation
    :param clip_learning_rate: Adam learning rate for the per-block
            clip-ratio parameter
    :param reg_param: weight of the regularization term that pulls each
            rounding element's relaxation toward a hard 0/1 decision
    :param warm_start: fraction of ``num_iterations`` (from the start) run
            with the rounding regularization term disabled
    :param beta_range: ``(beta_start, beta_end)`` for the rounding
            regularization term's exponent, linearly annealed after
            ``warm_start``
    :param clip_ratio_range: ``(min, max)`` bounds the optimized scale can
            move to, expressed as a multiple of the original (RTN) scale.
            Must satisfy ``min + max == 2.0`` for the optimization to start
            at the unmodified scale.
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its AutoRound-optimized codes, and its
            scale initializer rewritten whenever doing so actually improves
            reconstruction error.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    beta_start, beta_end = beta_range
    clip_ratio_min, clip_ratio_max = clip_ratio_range
    return onnx.load_from_string(
        C.apply_autoround(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            num_iterations,
            learning_rate,
            clip_learning_rate,
            reg_param,
            warm_start,
            beta_start,
            beta_end,
            clip_ratio_min,
            clip_ratio_max,
        )
    )


def apply_flexround_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 300,
    learning_rate: float = 0.05,
    log_clip: float = 4.0,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_flexround`: Lee et al. (2023)'s
    FlexRound -- "learnable-division rounding", the fourth onnxsim-native
    PTQ technique alongside :func:`onnxsim.apply_adaround_cpp`,
    :func:`onnxsim.apply_gptq_cpp`, and :func:`onnxsim.apply_awq_cpp`. See
    ``onnxsim/flexround.py``'s own module docstring for the full technique:
    reparametrizes the divisor itself (``S = scale * S2 * s3``, both
    ``S2``/``s3`` learnable and log-space parametrized) rather than
    AdaRound's own additive rounding perturbation.

    Same real calibration machinery as :func:`onnxsim.apply_adaround_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the float model in C++ (see
    ``ApplyFlexround`` in ``flexround_entry.h`` for the full scope).

    Same accepted-numerical-scope class as :func:`onnxsim.apply_adaround_cpp`:
    this is an iterative Adam optimization, so cross-language floating-point
    agreement is measured empirically (see tests/test_flexround_cpp.py)
    rather than assumed.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to optimize the
            divisor on -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_iterations: Adam steps to run per layer
    :param learning_rate: Adam learning rate for the log-space element-wise
            (``S2``) and per-output-channel (``s3``) divisor corrections
    :param log_clip: clamps ``log(S2)``/``log(s3)`` to
            ``[-log_clip, log_clip]`` after every step
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its FlexRound-optimized codes (same
            shape, dtype, and scale -- only which integer each element
            rounds to changes).
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_flexround(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            num_iterations,
            learning_rate,
            log_clip,
        )
    )


def apply_foem_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    foem_beta: float = 0.005,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_foem`: FOEM ("First-Order Error
    Matters", 2025) -- a sequential, Hessian-*and*-first-order-drift-
    compensated extension of :func:`onnxsim.apply_gptq_cpp`'s own rounding.
    See ``onnxsim/foem.py``'s own module docstring for the full technique:
    alongside GPTQ's own Hessian-compensated rounding error, also charges
    forward a damped (``foem_beta``) fraction of how far each column's own
    pre-quantization value has already drifted from the weight's true
    original column, due to every earlier column's own forward propagation.
    ``foem_beta == 0.0`` recovers plain GPTQ exactly.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` -- a
    live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the float model in C++ (see
    ``ApplyFoem`` in ``foem_entry.h`` for the full scope, including its
    accepted numerical scope -- shared with ``ApplyGptq``'s own).

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to compute each
            layer's Hessian from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param percdamp: Hessian damping factor, matching
            :func:`onnxsim.apply_gptq_cpp`'s own parameter and default
    :param proc_block_size: GPTQ's own column-processing block size,
            matching :func:`onnxsim.apply_gptq_cpp`'s own parameter and
            default
    :param foem_beta: damping factor on the additional first-order drift
            term -- ``0.0`` recovers plain GPTQ exactly
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its FOEM-optimized codes.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_foem(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            percdamp,
            proc_block_size,
            foem_beta,
        )
    )


def apply_slim_llm_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    target_bits: float = 3.0,
    low_bits: int = 2,
    high_bits: int = 4,
    group_size: int = 32,
    percdamp: float = 0.01,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_slim_llm`: SliM-LLM (Huang,
    Shao, Dong, Luo, Qiao et al., 2024) -- salience-driven mixed-precision
    quantization that picks a bit-width per GROUP within a layer's own
    weight (contrast :mod:`onnxsim.mixed_precision`, which picks one
    bit-width per whole layer) -- see ``onnxsim/slim_llm.py``'s own module
    docstring for the full technique (salience score, bit assignment, and
    storage format).

    Same real calibration machinery as :func:`onnxsim.apply_llm_int8_cpp`
    -- a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplySlimLlm`` in ``slim_llm_entry.h`` for the full scope, including
    its own accepted numerical scope: the dense inverse/Cholesky behind
    the per-column salience score uses scalar double-precision kernels
    rather than LAPACK).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches used to build
            each layer's Hessian and per-column error -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param target_bits: target average bits/weight for each matched layer
    :param low_bits: bit-width for the least salient groups in each layer
    :param high_bits: bit-width for the most salient groups in each layer
    :param group_size: elements per quantization group along K
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added to every diagonal entry before inversion)
    :param providers: onnxruntime execution providers to run calibration on
    :returns: ``model`` with every matched layer's weight replaced by
            per-group mixed ``low_bits``/``high_bits`` INT8-stored codes
            plus a per-group float32 scale and a parallel per-group
            bit-width INT64 initializer; layers with a non-constant,
            non-2-D weight, a reduction dimension not divisible by
            ``group_size``, or no calibration activation available, are
            left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_slim_llm(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            target_bits,
            low_bits,
            high_bits,
            group_size,
            percdamp,
        )
    )


def apply_moequant_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    quant_block_size: int = 32,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    ebss: bool = True,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_moequant`: MoEQuant (Hu, Chen
    et al., 2025) -- Expert-Balanced Self-Sampling (EBSS) plus
    Affinity-Guided Quantization (AGQ), a calibration METHODOLOGY (not a
    new quantization algorithm -- the column-update machinery is
    apply_gptq's own, reused as-is) for ``com.microsoft::MoE`` nodes' per-
    expert weights -- see ``onnxsim/moequant.py``'s own module docstring
    for the full technique.

    Same real calibration machinery as :func:`onnxsim.apply_llm_int8_cpp`
    -- a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyMoequant`` in ``moequant_entry.h`` for the full scope, including
    its own two accepted numerical divergences: EBSS's own weighted-
    without-replacement subsampling does not reproduce numpy's own RNG
    stream, and the reused GPTQ machinery's dense inverse/Cholesky uses
    scalar double-precision kernels rather than LAPACK).

    :param model: onnx ModelProto object or file path
    :param calibration_data: representative input batches to capture each
            MoE node's own input activations and router gate scores from
            -- see :func:`onnxsim.generate_random_calibration_data` (the
            default when omitted)
    :param num_samples: random batches to generate when ``calibration_data``
            is omitted
    :param seed: seed for the random calibration data AND for this port's
            own independent EBSS subsampling RNG (NOT the same random
            stream as :func:`onnxsim.apply_moequant`'s own numpy
            ``Generator`` -- see ``ApplyMoequant``'s own accepted
            numerical scope note)
    :param quant_block_size: INT4 block size along each expert weight's own
            reduction axis
    :param percdamp: GPTQ Hessian damping factor
    :param proc_block_size: GPTQ's own column-processing block size
    :param ebss: when true (default), cap an over-represented expert's
            routed calibration tokens down to a shared, balanced per-expert
            budget before building its Hessian
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched, FLOAT32 MoE node's per-expert
            ``fc1``/``fc2`` weight simulated-INT4-quantized in place.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_moequant(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            quant_block_size,
            percdamp,
            proc_block_size,
            ebss,
            seed,
        )
    )


def apply_qronos_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_qronos`: a sequential,
    whole-model generalization of GPTQ that additionally accounts for the
    error already baked into a layer's activations because upstream
    layers were quantized first, not just this layer's own rounding --
    see ``onnxsim/qronos.py``'s own module docstring for the full
    technique and how it reduces exactly to
    :func:`onnxsim.apply_gptq_cpp` when a layer has no already-quantized
    upstream layer feeding it.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyQronos`` in ``qronos_entry.h`` for the full scope, including
    its accepted numerical scope, shared with :func:`onnxsim.apply_gptq_cpp`'s
    own). Unlike every other calibration-driven ``*_cpp`` port in this
    codebase, the executor is invoked once per matched layer (not once up
    front for all of them): each layer's correction re-probes the
    progressively-corrected quantized model, processing layers in
    ``float_model``'s own node order.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to compute each
            layer's target/Hessian from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added before inversion)
    :param proc_block_size: GPTQ's own column-processing block size (not
            the quantization scale's own block size, reused unchanged)
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            initializer rewritten to its Qronos-corrected codes (same
            shape, dtype, and scale -- only which integer each element
            rounds to changes).
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_qronos(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            percdamp,
            proc_block_size,
        )
    )


def apply_tesseraq_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_bits: int = 4,
    num_iterations: int = 400,
    par_rounds: int = 4,
    learning_rate: float = 0.1,
    scale_learning_rate: float = 0.01,
    reg_param: float = 0.01,
    warm_start: float = 0.2,
    beta_range: Tuple[float, float] = (20.0, 2.0),
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_tesseraq`: TesseraQ's own
    "Progressive Adaptive Rounding" (PAR) -- an AdaRound-style
    rectified-sigmoid rounding relaxation, optimized by a hand-rolled Adam
    loop jointly with each weight block's own dequantization scale (in
    log-space), with a coarse-to-fine element-by-element hardening
    schedule across ``par_rounds`` rounds instead of a single monolithic
    anneal -- see ``onnxsim/tesseraq.py``'s own module docstring for the
    full technique.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the float model in C++ (see
    ``ApplyTesseraq`` in ``tesseraq_entry.h`` for the full scope).

    Unlike every other calibration-driven ``*_cpp`` port in this codebase
    (all either closed-form or, for :func:`onnxsim.apply_quarot_gptq_cpp`/
    :func:`onnxsim.apply_gptvq_cpp`, closed-form apart from an
    independently-seeded RNG), this is an iterative Adam optimization: see
    ``tesseraq_entry.h``'s own accepted numerical scope note for what that
    means for cross-language floating-point agreement, and
    tests/test_tesseraq_cpp.py for how closely this tracks the pure-Python
    reference in practice.

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to optimize the
            reconstruction against -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied) -- this port has no RNG of
            its own (unlike :func:`onnxsim.apply_quarot_gptq_cpp`'s/
            :func:`onnxsim.apply_gptvq_cpp`'s own), so ``seed`` only ever
            affects which calibration batches get generated
    :param num_bits: effective signed bit width PAR rounds each element
            into (2..4) -- see :func:`onnxsim.apply_tesseraq`'s own
            parameter of the same name
    :param num_iterations: total Adam steps to run per layer, split evenly
            across ``par_rounds``
    :param par_rounds: number of Progressive Adaptive Rounding rounds
    :param learning_rate: Adam learning rate for the per-element rounding
            relaxation
    :param scale_learning_rate: Adam learning rate for each weight block's
            dequantization scale
    :param reg_param: weight of the regularization term pulling each
            still-soft element toward a hard 0/1 decision
    :param warm_start: fraction of the total iteration budget run with the
            regularization term disabled
    :param beta_range: ``(beta_start, beta_end)`` for the regularization
            term's exponent, linearly annealed after ``warm_start``
    :param providers: onnxruntime execution providers to run ``float_model``
            on when capturing calibration activations
    :returns: ``quantized_model`` with every matched layer's INT4 weight
            codes and per-block scale initializers rewritten to their
            PAR-optimized values.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    beta_start, beta_end = beta_range
    return onnx.load_from_string(
        C.apply_tesseraq(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            num_bits,
            num_iterations,
            par_rounds,
            learning_rate,
            scale_learning_rate,
            reg_param,
            warm_start,
            beta_start,
            beta_end,
        )
    )


def apply_awq_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_alpha_steps: int = 20,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_awq`: AWQ (Lin et al., 2023)
    grid-searched per-channel weight rescaling for every
    ``quantize_weight_only_int4``-quantized MatMul/Gemm layer present (by
    node output name) in both ``float_model`` and ``quantized_model``,
    re-quantizing from scratch at each grid point over ``[0, 1]`` and
    keeping the exponent with the lowest reconstruction error.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the float model in C++ (see
    ``ApplyAwq`` in ``awq_entry.h`` for the full scope).

    :param float_model: the original (unquantized) onnx ModelProto or file
            path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`
    :param calibration_data: representative input batches to search and
            measure the rescaling on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_alpha_steps: grid points for the per-channel scale exponent
            ``alpha``, evenly spaced over ``[0, 1]`` inclusive
    :param providers: onnxruntime execution providers to run
            ``float_model`` on when capturing calibration activations
    :returns: ``quantized_model`` with every measurably improved layer
            rewritten (INT4 weight/scale replaced, compensating ``Mul``
            inserted); unimproved layers are left completely untouched.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            float_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_awq(
            _get_model_executor(providers),
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            calibration_data_pb,
            num_alpha_steps,
        )
    )


def apply_quarot_gptq_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    block_size: int = 32,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    epsilon: float = 1e-12,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_quarot_gptq`: the real QuaRot
    (Ashkboos et al., 2024) paper's optional, tighter weight quantizer --
    identical to :func:`onnxsim.apply_quarot_cpp` in every respect (same
    candidate matching, same per-layer random rotation, same data-free
    per-token INT4 activation quantization) except the *weight* is
    quantized via :func:`onnxsim.apply_gptq_cpp`'s own Hessian-based column
    algorithm, evaluated in the rotated activation space, instead of
    independent round-to-nearest.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through ``model`` in C++ (see
    ``ApplyQuarotGptq`` in ``quarot_gptq_entry.h`` for the full scope,
    including its own permanent RNG divergence from the Python reference --
    shared with :func:`onnxsim.apply_quarot_cpp`'s own -- and its accepted
    numerical scope, shared with :func:`onnxsim.apply_gptq_cpp`'s own).

    Unlike :func:`onnxsim.apply_gptq_cpp`/:func:`onnxsim.apply_awq_cpp`,
    there is no separate ``quantized_model`` argument: like
    :func:`onnxsim.apply_quarot_cpp`, this pass derives its own rotation and
    quantizes from ``model`` alone.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            matched layer's rotated-space Hessian from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the per-layer random rotation (independent of,
            and not comparable to, :func:`onnxsim.apply_quarot_gptq`'s own
            ``seed`` -- see ``quarot_gptq_entry.h``) and for the random
            calibration data (ignored for the latter if ``calibration_data``
            is supplied)
    :param block_size: elements per weight quantization block along K,
            matching :func:`onnxsim.apply_quarot_cpp`'s own default
    :param percdamp: Hessian damping factor, matching
            :func:`onnxsim.apply_gptq_cpp`'s own parameter and default
    :param proc_block_size: GPTQ's own column-processing block size (not
            the quantization scale's own block size)
    :param epsilon: floor applied to a token's own max-abs rotated-activation
            value before using it as a scale at graph-run time, matching
            :func:`onnxsim.apply_quarot_cpp`'s own parameter
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer that has usable calibration
            data rotated and INT4-quantized; a matched layer without usable
            calibration data is left completely untouched. A model with no
            matching layer, or an opset older than 21, is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_quarot_gptq(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            seed,
            block_size,
            percdamp,
            proc_block_size,
            epsilon,
        )
    )


def apply_gptvq_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    vector_dim: int = 2,
    num_centroids: int = 256,
    num_iterations: int = 10,
    percdamp: float = 0.01,
    providers: Optional[Sequence[backend.Provider]] = None,
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_gptvq`: GPTVQ
    (Van Baalen et al., 2024) -- a genuine combination of
    :func:`onnxsim.apply_gptq_cpp`'s own sequential, Hessian-compensated
    correction with a k-means-fit vector codebook (like
    :mod:`onnxsim.aqlm`'s own single shared codebook): small groups of
    consecutive input-channel columns of every matched MatMul/vanilla-Gemm
    layer's constant 2-D FLOAT32 weight are jointly quantized against the
    codebook, then each group's resulting per-column residual is
    propagated into every not-yet-quantized column exactly like GPTQ's own
    per-column correction (see ``onnxsim/gptvq.py``'s own module docstring
    for the full technique).

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` --
    a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through ``model`` in C++ (see ``ApplyGptvq``
    in ``gptvq_entry.h`` for the full scope, including its own permanent
    RNG divergence from the Python reference for the k-means codebook fit
    -- not aliased to :func:`onnxsim.quantize_weight_only_gptvq` for the
    same reason :func:`onnxsim.apply_quarot_cpp` is not aliased to
    :func:`onnxsim.apply_quarot`).

    Rewires only the matched node's weight input (a
    ``Gather``+``Reshape``[+``Transpose``] chain); the node itself,
    including any bias, is left otherwise unchanged.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the per-layer k-means codebook initialization
            (independent of, and not comparable to,
            :func:`onnxsim.quantize_weight_only_gptvq`'s own ``seed`` --
            see ``gptvq_entry.h``) and for the random calibration data
            (ignored for the latter if ``calibration_data`` is supplied)
    :param vector_dim: elements per group (each ``vector_dim``-element
            chunk of consecutive input-channel columns is jointly
            quantized to a single codebook entry)
    :param num_centroids: entries in the layer's own fitted codebook
    :param num_iterations: Lloyd's-algorithm iterations fitting the
            codebook
    :param percdamp: Hessian damping factor, matching
            :func:`onnxsim.apply_gptq_cpp`'s own parameter and default
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``model`` with every matched, eligible layer's weight
            replaced by a codebook lookup reconstructing it in the
            weight's own units; a layer with a non-constant, non-2-D
            weight, a name in ``skip_names``, a reduction dimension not
            divisible by ``vector_dim``, or no usable calibration
            activation, is left completely untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_gptq_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_gptvq(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            seed,
            vector_dim,
            num_centroids,
            num_iterations,
            percdamp,
            list(skip_names) if skip_names is not None else [],
        )
    )


def apply_smoothquant_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    alpha: float = 0.5,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_smoothquant`: SmoothQuant
    (Xiao et al., 2022) migration -- rescales every matched
    MatMul/vanilla-Gemm layer's constant 2-D FLOAT32 weight columns by the
    per-channel migration scale ``s`` in place and inserts a ``Mul`` node
    dividing that layer's activation input by the same ``s``. Returns a
    float model -- pass the result to a W8A8 quantizer (e.g.
    :func:`onnxsim.quantize_static`) to actually quantize it.

    Same real calibration machinery as
    :func:`onnxsim.apply_imatrix_quantization_cpp` -- a live
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplySmoothQuant`` in ``smoothquant_entry.h`` for the full scope).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's activation range on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param alpha: the migration strength (``0.5`` splits difficulty evenly
            on a log scale)
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value (and to ``s`` itself) before computing ``s``
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer migrated; layers with a
            non-constant/non-2-D weight, or whose activation was never
            observed as a plain 2-D tensor matching the weight's reduction
            dimension, are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_imatrix_quantization_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_smoothquant(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            alpha,
            epsilon,
        )
    )


def apply_fptq_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    alpha: float = 0.5,
    outlier_ratio_threshold: float = 10.0,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_fptq`: FPTQ (Li, Zhang, Li,
    Yao, Zhang, Chu, Sun, Du and Xie, 2023) migration -- shares
    :func:`onnxsim.apply_smoothquant_cpp`'s own core mechanism (rescale a
    matched MatMul/vanilla-Gemm layer's constant 2-D FLOAT32 weight columns
    by a per-channel migration scale ``s`` in place, dividing the
    activation by the same ``s`` via a new ``Mul`` node), replacing
    SmoothQuant's own power-law scale with FPTQ's own logarithmic-
    equalization scale on layers whose activation has an outlier channel
    at least ``outlier_ratio_threshold`` times the layer's own typical
    (geometric-mean) channel scale. See :mod:`onnxsim.fptq`'s own module
    docstring for the technique. Returns a float model -- pass the result
    to onnxsim's own W4 weight-only quantizer and a W8A8 activation
    quantizer afterward to realize the paper's actual W4A8 recipe.

    Same real calibration machinery as
    :func:`onnxsim.apply_smoothquant_cpp` -- a live
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see ``ApplyFptq``
    in ``fptq_entry.h`` for the full scope).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's activation range on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param alpha: the migration strength used on "tractable" layers,
            identical in meaning to :func:`onnxsim.apply_smoothquant_cpp`'s
            own ``alpha``
    :param outlier_ratio_threshold: a layer is classified "intractable"
            (and gets the logarithmic scale instead of the power-law one)
            when its largest per-channel activation max is at least this
            many times its per-channel geometric-mean activation max
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value (and to the geometric-mean reference and ``s``
            itself) before dividing
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer migrated; layers with a
            non-constant/non-2-D weight, or whose activation was never
            observed as a plain 2-D tensor matching the weight's reduction
            dimension, are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_smoothquant_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_fptq(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            alpha,
            outlier_ratio_threshold,
            epsilon,
        )
    )


def apply_easyquant_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    num_iterations: int = 3,
    num_candidates: int = 21,
    search_span: float = 0.5,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_easyquant`: EasyQuant (Wu,
    Judd, Isaev, Micikevicius, 2020) -- W8A8-quantizes every matched
    MatMul/vanilla-Gemm layer, choosing both the per-output-channel weight
    scale and the per-tensor activation scale via EasyQuant's own
    coordinate-descent GRID SEARCH (no gradients or Hessian) against real
    calibration activations. See :mod:`onnxsim.easyquant`'s own module
    docstring for the technique.

    Same real calibration machinery as
    :func:`onnxsim.apply_smoothquant_cpp` -- a live
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyEasyquant`` in ``easyquant_entry.h`` for the full scope,
    including its accepted numerical scope: a bounded, deterministic grid
    search with no RNG or hand-rolled dense linear algebra, so this port
    is expected to track the Python reference closely, including exact-tie
    behavior at every grid point boundary).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to run the scale
            search against -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param num_iterations: coordinate-descent rounds (weight step, then
            activation step) -- each round only ever improves or holds the
            previous round's own chosen scales
    :param num_candidates: grid resolution per coordinate-descent step
    :param search_span: candidate multipliers span
            ``[1 - search_span, 1 + search_span]`` around each step's
            current scale
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by its
            quantize-dequantize round-tripped float32 version, and a
            ``Div``/``Round``/``Clip``/``Mul`` round-trip inserted before
            its activation input; layers with a non-constant, non-2-D
            weight, an activation with no feature axis at all (rank < 2; a
            higher-rank ``[batch, seq, K]`` one is flattened to
            ``[batch * seq, K]``, which is exact), or whose activation's
            feature dimension doesn't match the weight's own reduction
            size, are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_smoothquant_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_easyquant(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            num_iterations,
            num_candidates,
            search_span,
        )
    )


def apply_outlier_suppression_plus_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    alpha: float = 0.5,
    epsilon: float = 1e-5,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_outlier_suppression_plus`:
    Outlier Suppression+ (Wei et al., 2023) channel-wise shifting and
    scaling -- recenters every matched MatMul/vanilla-Gemm layer's
    constant 2-D FLOAT32 activation channels around zero, rescales the
    weight columns by the per-channel scale ``s`` in place, inserts a
    ``Sub``+``Mul`` pair applying the shift and scale before the layer,
    and inserts an ``Add`` after it restoring the shift's constant
    contribution to the output. Returns a float model -- pass the result
    to a W8A8 quantizer (e.g. :func:`onnxsim.quantize_static`) to actually
    quantize it.

    Same real calibration machinery as
    :func:`onnxsim.apply_outlier_suppression_cpp` -- a live
    :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see
    ``ApplyOutlierSuppressionPlus`` in
    ``outlier_suppression_plus_entry.h`` for the full scope).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's activation range on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param alpha: the scaling step's migration strength, identical in
            meaning to :func:`onnxsim.apply_smoothquant_cpp`'s own
            ``alpha`` (applied to the *shifted* activation's range)
    :param epsilon: floor applied to every per-channel activation/weight
            max-abs value (and to the scale itself) before computing it
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer shifted, scaled, and
            output-corrected; layers with a non-constant/non-2-D weight,
            or whose activation was never observed as a plain 2-D tensor
            matching the weight's reduction dimension, are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_outlier_suppression_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_outlier_suppression_plus(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            alpha,
            epsilon,
        )
    )


def apply_moe_expert_channel_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_moe_expert_channel_pruning`:
    removes intermediate (``inter_size``) channels from every expert of a
    matched ``com.microsoft::MoE`` node at once -- real structural pruning
    (smaller ``fc1_experts_weights``/``fc2_experts_weights``, smaller
    per-expert matmuls on any runtime), as opposed to value-only zeroing.

    For every matched ``MoE`` node: ranks every ``inter_size`` index by
    combined (root-sum-square) L2 norm of ``fc1_experts_weights``' own row
    (across every expert and ``hidden_size`` at once) and
    ``fc2_experts_weights``' own column (same reduction), plus
    ``fc1_experts_bias``'s own entry when present, drops the lowest-
    ``sparsity``-fraction of indices (at least one is always kept), and
    removes the matching row from ``fc1_experts_weights``/
    ``fc1_experts_bias`` and column from ``fc2_experts_weights``, identically
    across every expert. ``num_experts``, ``k``, and every node attribute are
    untouched -- pruning ``inter_size`` changes no other tensor's shape
    anywhere in the graph, including the node's own output (always equal to
    ``input``'s shape).

    A node with ``fc3_experts_weights`` present, a ``swiglu``/unrecognized
    ``activation_type`` (or nonzero ``swiglu_fusion``), a non-constant or
    tied/shared ``fc1``/``fc2`` weight (or bias), or a shape this pass
    doesn't recognize (e.g. mismatched ``hidden_size`` between
    ``fc1_experts_weights``/``fc2_experts_weights``) is left completely
    untouched -- the same conservative "decline rather than mis-slice" bar
    every other chain-matcher in this codebase holds.

    ``fc1_experts_weights``/``fc2_experts_weights``/``fc1_experts_bias`` may
    be FLOAT, FLOAT16, or BFLOAT16 -- full parity with the pure-Python
    :func:`onnxsim.apply_moe_expert_channel_pruning` (itself now a thin
    alias for this function; see IsSupportedFloatDtype/ReadTensorAsF64/
    WriteF64TensorAs in ``onnxsim/structured_pruning_entry.cpp``'s own "MoE
    expert-intermediate-channel pruning" section for the read-upcast/
    write-downcast mechanics -- a surviving weight's own value is only ever
    reordered/dropped, never recomputed, so this round-trips every
    FLOAT16/BFLOAT16 bit pattern exactly). Whole-expert pruning (shrinking
    ``num_experts`` itself, calibration-driven) is a deliberately separate
    entry point, :func:`onnxsim.apply_moe_whole_expert_pruning_cpp`.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched node's ``inter_size``
            channels to remove (at least one channel is always kept); must
            be in ``[0, 1)``
    :returns: ``model`` with every matched ``MoE`` node's ``fc1``/``fc2``
            tensors resized in place; anything not matching the exact
            topology above is left completely untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_moe_expert_channel_pruning(model.SerializeToString(), sparsity)
    )


def apply_qmoe_expert_channel_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_qmoe_expert_channel_pruning`:
    removes intermediate (``inter_size``) channels from every expert of a
    matched ``com.microsoft::QMoE`` node at once -- the quantized-weight
    counterpart of :func:`onnxsim.apply_structured_pruning_cpp`, targeting
    ``QMoE``'s own packed ``uint8`` ``fc1_experts_weights``/
    ``fc2_experts_weights`` (plus their ``scales``/``zero_points``/
    ``global_scale`` operands, co-sliced in lockstep) instead of plain float
    weights.

    Supports ``quant_type='int'`` (``expert_weight_bits`` in ``{2, 4, 8}``,
    with no ``block_size`` -- whole-row per-channel scale -- or a groupwise
    ``block_size``) and ``quant_type='nvfp4'`` (E2M1-packed weights,
    ``float8e4m3fn`` per-block scales, a required per-expert ``float32``
    global scale, ``block_size`` fixed at 16 -- schema-derived only, since no
    onnxruntime build has a CPU kernel for any FP4/FP8 ``quant_type`` to
    verify against). ``'fp4'``, ``'fp8'``, and ``'wfp4afp8'`` remain out of
    scope, as does ``fc3_experts_weights``, ``router_weights``, a
    ``swiglu``/unrecognized ``activation_type``, and a CUTLASS-prepacked
    (``weights_prepacked`` outside ``{-1, 0}``) weight layout.

    Ranks every ``inter_size`` index by combined (root-sum-square) L2 norm of
    ``fc1_experts_weights``'/``fc2_experts_weights``' own DEQUANTIZED row/
    column (never written back -- the actual rewrite always slices the
    existing packed codes/scales/zero_points in place, re-packing a
    sub-byte-packed axis rather than ever re-quantizing a sliced float
    weight from scratch) plus ``fc1_experts_bias``'s own entry when present,
    drops the lowest-``sparsity``-fraction of indices (at least one always
    kept, floored to a multiple of ``8 / expert_weight_bits`` -- or, with
    ``block_size`` set, to whole ``block_size``-sized groups, since
    ``fc2_experts_weights``' own quantization blocks group along
    ``inter_size`` and a value can't be dropped out of a shared-scale group
    without re-quantizing it), and removes the matching row from ``fc1``'s
    own weight/scales/bias/zero_points and column from ``fc2``'s own weight
    (plus, only when ``block_size`` is set, ``fc2``'s own scales/
    zero_points too), identically across every expert. ``num_experts``, `k`,
    and every node attribute are untouched.

    ``fc1``/``fc2`` scales and bias may be FLOAT, FLOAT16, or BFLOAT16 --
    full parity with the pure-Python :func:`onnxsim.apply_qmoe_expert_
    channel_pruning` (itself now a thin alias for this function; the packed
    ``uint8`` `fc1`/`fc2` weights themselves, and `zero_points`, are
    unaffected either way -- always UINT8 regardless of the *activation*/
    scale dtype). The complementary whole-expert-removal pass is a separate
    entry point, :func:`onnxsim.apply_qmoe_whole_expert_pruning_cpp`.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched node's `inter_size`
            channels (or, with `block_size` set, `block_size`-sized groups)
            to remove (at least one channel is always kept); must be in
            ``[0, 1)``
    :returns: ``model`` with every matched ``QMoE`` node's `fc1`/`fc2`
            tensors resized in place; anything not matching the exact
            topology above is left completely untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_qmoe_expert_channel_pruning(model.SerializeToString(), sparsity)
    )


def apply_moe_whole_expert_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_moe_whole_expert_pruning`: the
    calibration-driven complementary technique to
    :func:`onnxsim.apply_moe_expert_channel_pruning_cpp`'s own `inter_size`
    pruning -- drops whole experts (shrinks the `num_experts` leading axis)
    from a matched ``com.microsoft::MoE`` node and its upstream router
    projection at once, exactly like :func:`onnxsim.apply_structured_wanda_
    pruning_cpp` is this onnxsim's first calibration-driven MoE C++ pruning
    entry point: the executor actually runs `model` over `calibration_data`
    to capture real router activations (via
    :func:`onnxsim.onnx_simplifier._get_model_executor`, the same
    process-wide executor :func:`onnxsim.simplify` itself uses).

    For each matched ``MoE`` node whose upstream router is a single, untied,
    constant-weight MatMul/vanilla-Gemm feeding `router_probs` (and nothing
    else): ranks every expert by its mean router *gate weight* --
    ``softmax(router_probs)`` averaged over every calibration token, NOT raw
    logit magnitude (no shared scale across experts) and NOT exact top-k
    selection frequency (would require re-deriving onnxruntime's own top-k +
    renormalization semantics) -- drops the lowest-``sparsity``-fraction of
    experts, with `num_experts_to_keep` silently FLOORED at the node's own
    `k` attribute (`k` itself is never modified -- pruning below `k` experts
    remaining is a hard onnxruntime execution failure, confirmed
    empirically, not merely suboptimal). Every dropped expert's own
    `fc1_experts_weights`/`fc2_experts_weights` (and `fc1_experts_bias`, if
    present) row, and the router projection's own matching output column
    (weight and bias, if present), are removed together. A chain whose
    `router_probs` was never observed during calibration (e.g.
    ``calibration_data=[]``, or a chain matched only inside a nested
    subgraph -- see below) falls back to each expert's own combined
    `fc1`/`fc2`(+bias) L2 weight norm, the same "no matching activation
    observed" fallback :func:`onnxsim.apply_structured_wanda_pruning_cpp`
    already uses.

    A node with `fc3_experts_weights` present, a `swiglu`/unrecognized
    `activation_type`, `use_sparse_mixer=1`, a non-constant or tied/shared
    weight anywhere in the chain (including the router projection), a
    `router_probs` with more than one consumer, or any other shape this
    pass doesn't recognize is left completely untouched.

    Subgraph-aware (matching/slicing) for every matched ``MoE`` node, at any
    ``If``/``Loop``/``Scan``/``BeamSearch``-family nesting depth -- but the
    calibration-based *ranking* only ever runs over chains matched in the
    TOP-LEVEL graph (the probe-output injection this relies on can only
    ever append to the top-level graph's own outputs), so a chain matched
    only inside a nested subgraph always falls back to the weight-norm
    ranking above -- still correctly pruned, never silently skipped.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to rank experts by
            mean router gate weight on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched node's `num_experts` to
            remove (floored at the node's own `k`, so fewer may actually be
            removed -- never more); must be in ``[0, 1)``
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration router activations (passed to the
            shared :func:`onnxsim.onnx_simplifier._get_model_executor`
            process-wide executor)
    :returns: ``model`` with every matched ``MoE`` node's `fc1`/`fc2`(/`fc1`
            bias) and its router projection's weight(/bias) resized in
            place; anything not matching that exact topology is left
            completely untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing as
    # apply_structured_wanda_pruning_cpp -- see that wrapper's own comment;
    # the name -> positional reordering happens entirely on the C++ side
    # (MoeRouterGateCalibrationStats in structured_pruning_entry.cpp).
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_moe_whole_expert_pruning(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            sparsity,
        )
    )


def apply_qmoe_whole_expert_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_qmoe_whole_expert_pruning`: the
    quantized-weight counterpart of
    :func:`onnxsim.apply_moe_whole_expert_pruning_cpp` -- drops whole
    experts from a matched ``com.microsoft::QMoE`` node and its upstream
    router projection, ranked by the exact same calibration-based mean
    router gate weight (`router_probs` is `QMoE`'s own second input too,
    upstream of and oblivious to its quantized `fc1`/`fc2`, so this needs no
    `QMoE`-specific calibration logic at all -- the same shared
    ``MoeRouterGateCalibrationStats`` C++ helper is reused unchanged), and
    the identical `k`-floor safety property.

    Unlike :func:`onnxsim.apply_qmoe_expert_channel_pruning_cpp`'s own
    `inter_size` slice (which needs the full packed-axis unpack/select/
    repack machinery, since `inter_size` sometimes falls on a
    sub-byte-packed axis), whole-expert pruning needs no such handling at
    all: `num_experts` is every per-expert QMoE tensor's own LEADING axis
    (`fc1`/`fc2` weights, scales, biases, zero_points, and -- for
    `quant_type='nvfp4'` -- their own `global_scale`) regardless of
    `quant_type`/`block_size`, so every one is a plain raw-element axis-0
    index-select, plus the router projection's own weight/bias slice.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to rank experts by
            mean router gate weight on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched node's `num_experts` to
            remove (floored at the node's own `k`); must be in ``[0, 1)``
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration router activations
    :returns: ``model`` with every matched ``QMoE`` node's `fc1`/`fc2`
            tensors (and their own `scales`/`bias`/`zero_points`/
            `global_scale`, where present) and its router projection's
            weight(/bias) resized in place; anything not matching that
            exact topology is left completely untouched
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_qmoe_whole_expert_pruning(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            sparsity,
        )
    )


def apply_transformer_block_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.25,
    num_blocks_to_drop: Optional[int] = None,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_transformer_block_pruning`:
    depth/block-level pruning that drops whole redundant pre-norm
    transformer residual sub-blocks (``x = x + SelfAttn(LN(x))`` or
    ``x = x + MLP(LN(x))``) wholesale, rather than shrinking every block a
    little the way every other ``apply_*_cpp`` entry point does. A
    GENUINELY DIFFERENT KIND of pass from every other C++-backed pruning
    entry point in this module: it deletes whole graph nodes and rewires
    their consumers straight through to the dropped block's own input
    (real graph surgery, changing the graph's own topology), not tensors
    resized in place. Same executor-as-first-argument shape as
    :func:`onnxsim.apply_moe_whole_expert_pruning_cpp` -- the executor
    actually runs `model` over `calibration_data` (via
    :func:`onnxsim.onnx_simplifier._get_model_executor`, the same
    process-wide executor :func:`onnxsim.simplify` itself uses) to capture
    each candidate block's own input/output activations.

    See :func:`onnxsim.apply_transformer_block_pruning`'s own docstring
    (the pure-Python reference this ports) and
    ``onnxsim/structured_pruning_entry.cpp``'s own "Transformer block
    (depth) pruning" section comment for the exact matched pattern: a
    pre-norm residual sub-block whose merge is a bare ``Add`` (never a
    fused ``SkipLayerNormalization``-family node in that role) and whose
    entry norm is either a plain, unfused ``LayerNormalization``/
    ``RMSNormalization``/``SimplifiedLayerNormalization`` node, or a fused
    ``SkipLayerNormalization``/``SkipSimplifiedLayerNormalization`` node's
    own optional fourth output standing in for the block's own raw input
    -- so a model already run through onnxruntime's own transformer
    optimizer is matched too. Attention and MLP/FFN blocks are matched and
    dropped fully independently, never only as a paired "whole layer". A
    KV-cache-bearing attention block needs no dedicated detection to
    always decline safely on its own (its own ``present_key``/
    ``present_value`` graph outputs trip the same generic "no
    block-internal output may leak outside the block" check every
    candidate is held to).

    Every candidate is confirmed shape-safe (using real
    ``onnx::shape_inference`` output, not just whatever ``value_info``
    `model` already happened to carry) before ranking ever runs. Candidates
    are ranked by mean cosine similarity between their own input/output
    over `calibration_data` -- the literature-standard ("Block Influence"/
    ShortGPT-style) redundancy signal -- and the highest-similarity ones
    are dropped first, up to the target count, skipping (not failing) any
    candidate whose own interior overlaps an already-committed one's.

    Subgraph-aware (matching/selection/committing) exactly like
    :func:`onnxsim.apply_transformer_block_pruning`'s own docstring: the
    calibration-based ranking only ever runs over candidates matched in
    the TOP-LEVEL graph (the probe-output injection this relies on can
    only ever append to the top-level graph's own outputs), so a candidate
    matched only inside a nested subgraph is conservatively ranked last
    (never preferentially dropped) but is still genuinely droppable if
    `target` calls for it; `sparsity`/`num_blocks_to_drop` are pooled
    ACROSS THE WHOLE MODEL, not computed independently per graph.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            candidate block's own input/output similarity on. Each batch
            is a ``{input_name: np.ndarray}`` dict matching ``model``'s
            graph inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: fraction of *matched candidate* blocks to drop
            (rounded to the nearest whole block), ignored when
            ``num_blocks_to_drop`` is given
    :param num_blocks_to_drop: an explicit number of blocks to drop
            instead of a fraction, silently capped at however many
            candidates were actually matched
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations (passed to the shared
            :func:`onnxsim.onnx_simplifier._get_model_executor` process-wide
            executor)
    :returns: ``model`` with the target number of matched candidate blocks
            -- whichever ones ranked most redundant -- deleted and their
            own consumers rewired to read straight through to their own
            block's own input; unchanged (a byte-for-byte copy) if no
            candidate was matched, ``sparsity``/``num_blocks_to_drop``
            rounds to zero blocks, or ``calibration_data`` never gives any
            candidate a valid (non-degenerate) token to rank on
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing as
    # apply_structured_wanda_pruning_cpp -- see that wrapper's own comment;
    # the name -> positional reordering happens entirely on the C++ side
    # (TransformerBlockSimilarity in structured_pruning_entry.cpp).
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_transformer_block_pruning(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            sparsity,
            num_blocks_to_drop,
        )
    )


def _embedding_pruning_result_from_cpp(
    model_bytes: bytes,
    matched: bool,
    kept_token_ids: List[int],
    lm_head_pruned: bool,
) -> EmbeddingPruningResult:
    """Reconstructs the real, public :class:`onnxsim.pruning.EmbeddingPruningResult`
    -- the exact same dataclass :func:`onnxsim.apply_embedding_vocab_pruning`/
    :func:`onnxsim.apply_embedding_vocab_magnitude_pruning` already return --
    from the C++ binding's own ``(model_bytes, matched, kept_token_ids,
    lm_head_pruned)`` tuple. ``id_map`` (``{old_token_id: new_token_id}``) is
    not carried across the nanobind boundary at all -- it is exactly
    ``{tok: i for i, tok in enumerate(kept_token_ids)}``, trivial to
    reconstruct here, so there is no reason for the C++ side to also
    serialize an int64->int64 map.
    """
    model = onnx.load_from_string(model_bytes)
    if not matched:
        return EmbeddingPruningResult(model=model, matched=False)
    id_map = {old: new for new, old in enumerate(kept_token_ids)}
    return EmbeddingPruningResult(
        model=model,
        matched=True,
        kept_token_ids=list(kept_token_ids),
        id_map=id_map,
        lm_head_pruned=lm_head_pruned,
    )


def apply_embedding_vocab_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    keep_token_ids: Optional[Sequence[int]] = None,
    drop_token_ids: Optional[Sequence[int]] = None,
    input_name: Optional[str] = None,
) -> EmbeddingPruningResult:
    """C++-backed port of :func:`onnxsim.apply_embedding_vocab_pruning`:
    shrinks a matched token-embedding table's vocabulary axis -- a plain
    ``Gather``'s ``data`` input feeding a graph input's token-id tensor --
    (and, where a tied or confidently-auto-identified untied ``lm_head``
    exists, its own vocab-logits projection too) down to a caller-supplied,
    explicit keep-set.

    **Contract change, unmissable on purpose, identical to the pure-Python
    original**: the pruned model this returns does **not** accept the same
    ``input_ids`` values the original model did. Token id ``i`` is only
    ever a valid input going forward if ``i in result.id_map`` -- feed
    ``result.id_map[i]`` in its place before running ``result.model``. See
    :class:`onnxsim.pruning.EmbeddingPruningResult`'s own docstring for the
    full return-value contract.

    Matches all three producer shapes the pure-Python original recognizes:
    a plain ``Gather``, ``com.microsoft::EmbedLayerNormalization``, or
    ``com.microsoft::GatherBlockQuantized`` (the block-quantized embedding
    shape -- verified TRUE full parity across both of its own sub-8-bit
    packing conventions, native ``tensor(uint4)``/``tensor(int4)`` and
    manually-packed ``tensor(uint8)``; see ``onnxsim/structured_pruning_
    entry.cpp``'s own "Embedding vocabulary pruning" section comment for the
    full empirical detail). Its ``lm_head`` auto-detection recognizes
    ``MatMul``/vanilla ``Gemm``/``com.microsoft::FusedGemm``/
    ``GemmFastGelu``, and the embedding table/``lm_head`` weight/bias may be
    FLOAT, FLOAT16, OR BFLOAT16 -- all matching the pure-Python original's
    own scope exactly. See that same section comment for the exact matched
    topology.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.

    :param model: the original onnx ModelProto or file path
    :param keep_token_ids: the exact set of original token ids to keep, in
            any order/with any duplicates. Give exactly one of
            ``keep_token_ids``/``drop_token_ids``.
    :param drop_token_ids: the exact set of original token ids to drop;
            every other id in ``range(vocab_size)`` is kept. Give exactly
            one of ``keep_token_ids``/``drop_token_ids``.
    :param input_name: when the graph has more than one structurally-
            eligible embedding ``Gather``, names which one to target by its
            token-id operand's graph input name. Required in that case --
            omitted, an ambiguous graph declines the whole call rather than
            guessing. A name that matches no eligible producer raises
            ``ValueError``.
    :returns: an :class:`onnxsim.pruning.EmbeddingPruningResult` -- see its
            own docstring. ``matched`` is False (model left completely
            untouched) for any topology this pass doesn't confidently
            recognize.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    result = C.apply_embedding_vocab_pruning(
        model.SerializeToString(),
        list(keep_token_ids) if keep_token_ids is not None else None,
        list(drop_token_ids) if drop_token_ids is not None else None,
        input_name,
    )
    return _embedding_pruning_result_from_cpp(*result)


def apply_embedding_vocab_magnitude_pruning_cpp(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    protect_token_ids: Optional[Sequence[int]] = None,
    input_name: Optional[str] = None,
) -> EmbeddingPruningResult:
    """C++-backed port of :func:`onnxsim.apply_embedding_vocab_magnitude_pruning`:
    the importance-ranked variant of
    :func:`onnxsim.apply_embedding_vocab_pruning_cpp` -- drops the lowest-
    L2-norm ``sparsity`` fraction of vocabulary rows (combined,
    root-sum-square, with a matched untied ``lm_head``'s own per-row weight
    norm when one is identified), needing no calibration data at all.

    **This mode's safety bar is meaningfully weaker than
    :func:`onnxsim.apply_embedding_vocab_pruning_cpp`'s own explicit
    keep-set** -- see that function's own docstring and
    :func:`onnxsim.apply_embedding_vocab_magnitude_pruning`'s own (identical
    caveat, ported verbatim): a small embedding-row norm means a token was
    initialized/trained with small weights, not that it is safe to drop
    from a real deployment's input space. ``protect_token_ids`` (below)
    covers the common, cheap case of guaranteeing a known-important set
    never gets ranked away by norm alone, but does not by itself make
    norm-based ranking a safe substitute for the explicit-keep-set entry
    point in general.

    Same contract change as :func:`onnxsim.apply_embedding_vocab_pruning_cpp`
    -- see its own docstring and
    :class:`onnxsim.pruning.EmbeddingPruningResult`'s.

    Same matched topology as
    :func:`onnxsim.apply_embedding_vocab_pruning_cpp` (all three producer
    shapes, including ``GatherBlockQuantized``, plus every ``lm_head``-node-
    type/dtype shape) -- see that function's own docstring. For
    ``GatherBlockQuantized``, the per-row L2 norm this ranks by is computed
    from the dequantized-for-ranking-only embedding table (never written
    back) -- see ``onnxsim/structured_pruning_entry.cpp``'s own
    ``GatherBlockQuantizedDequantized``.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of ``vocab_size`` to drop (floored so
            at least one row, and every ``protect_token_ids`` row, always
            survives); must be in ``[0, 1)``
    :param protect_token_ids: token ids to always keep regardless of their
            own norm ranking
    :param input_name: identical to
            :func:`onnxsim.apply_embedding_vocab_pruning_cpp`'s own
            ``input_name``
    :returns: an :class:`onnxsim.pruning.EmbeddingPruningResult`
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    result = C.apply_embedding_vocab_magnitude_pruning(
        model.SerializeToString(),
        sparsity,
        list(protect_token_ids) if protect_token_ids is not None else None,
        input_name,
    )
    return _embedding_pruning_result_from_cpp(*result)


def apply_any_precision_llm_cpp(
    model: Union[str, onnx.ModelProto],
    bits: int = 4,
    max_bits: int = 8,
    block_size: int = 32,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_any_precision_llm`: weight-only
    quantizes every MatMul/vanilla-Gemm layer with a constant 2-D float32
    weight to ``bits`` bits per element, per (output channel, ``block_size``
    -element K-block), via a nested bit-plane code built once to
    ``max_bits`` (repeated within-bin bisection at each bin's own current
    min/max midpoint) and truncated down to ``bits`` by a plain integer
    right-shift -- see :func:`onnxsim.apply_any_precision_llm`'s own
    docstring for the full rationale (Park et al., 2024, ICML 2024,
    "Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized
    LLMs"). One quantization pass serves any precision up to ``max_bits``
    instead of re-quantizing from scratch per bit-width.

    Every matched element is replaced by its own quantize-dequantize
    (per-bin-mean reconstruction) round trip; the result stays float32
    (same shape/dtype as the original weight) -- this is a compute-only
    rewrite, not a compressed storage format, since no ONNX tensor type
    below INT4 exists to store 3/5/6/7-bit codes natively.

    Unlike that pure-Python implementation, this port's per-bin bisection
    groups indices via a hash map rather than numpy's own reduction order,
    so results are not expected to match bit-for-bit -- only to be
    similarly accurate (both satisfy the same exact nesting invariant and
    the same monotonically-improving-with-more-bits property; see
    :func:`onnxsim.apply_any_precision_llm`'s own docstring for why an
    earlier, affine-fit-based reconstruction did NOT have that property).

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param bits: the precision level to materialize, ``1 <= bits <=
            max_bits``
    :param max_bits: the highest bit-width the nested code tree is built to
    :param block_size: elements per (output-channel, K-block) reconstruction
            group, matching :func:`onnxsim.quantize_weight_only_int4`'s own
            default block size convention
    :returns: the quantized onnx ModelProto
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_any_precision_llm(model.SerializeToString(), bits, max_bits, block_size)
    )


def apply_quarot_cpp(
    model: Union[str, onnx.ModelProto],
    seed: int = 0,
    block_size: int = 32,
    epsilon: float = 1e-12,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_quarot`: applies QuaRot-style
    random-rotation preprocessing (Ashkboos et al., 2024, "QuaRot:
    Outlier-Free 4-Bit Inference in Rotated LLMs") plus INT4
    round-to-nearest quantization of *both* the weight and the activation to
    every MatMul/vanilla-Gemm layer with a constant 2-D float32 weight whose
    reduction dimension ``K`` is divisible by ``block_size``.

    Rotating the whole residual stream by a random orthogonal matrix removes
    activation outliers the same way block quantization already tolerates
    weight outliers, letting both MatMul operands drop to INT4 with no
    calibration data at all -- see :func:`onnxsim.apply_quarot`'s own
    docstring for the full rationale. Unlike that pure-Python
    implementation, this port derives a fresh rotation per matched layer
    from ``seed`` independently of graph node order, so results are *not*
    expected to match the Python port bit-for-bit -- only to be similarly
    accurate.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Layers with a non-constant, non-2-D weight, or a reduction dimension not
    divisible by ``block_size``, are left untouched. Consider calling
    :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param seed: seed for the per-layer random rotation matrices
    :param block_size: elements per weight quantization block along ``K``,
            matching :func:`onnxsim.quantize_weight_only_int4`'s own default
    :param epsilon: floor applied to a token's own max-abs rotated-activation
            value before using it as a scale, avoiding a divide-by-zero on
            an all-zero token
    :returns: the rotated-and-quantized onnx ModelProto; a model with no
            matching layer, or an opset older than 21, is returned unchanged
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_quarot(model.SerializeToString(), seed, block_size, epsilon)
    )


def apply_iq4_nl_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_iq4_nl_quantization`: weight-only
    quantizes every MatMul/vanilla-Gemm layer with a constant 2-D float32
    weight into llama.cpp's IQ4_NL format -- a fixed, 16-entry non-uniform
    ("non-linear") codebook, one scale per 32-element block of the weight's
    own flattened storage (``scale = max(|block|) / max(|codebook|)``, each
    element snapped to whichever codebook entry times that scale is
    closest). See :func:`onnxsim.apply_iq4_nl_quantization`'s own docstring
    for the full rationale and, importantly, this format's own codebook
    provenance -- this repo could not find or verify llama.cpp's real IQ4_NL
    codebook (`kvalues_iq4nl`) anywhere in-tree, so both this port and its
    Python counterpart use their own computationally-derived, honestly
    documented non-uniform codebook instead, not a transcription of
    llama.cpp's own table.

    Unlike that pure-Python implementation, this port does not support
    quantizing ``Conv`` weights, and is not guaranteed to be bit-for-bit
    identical to it -- though, having no accumulation or
    iterative-refinement step at all (every element's own code comes from a
    single per-block max-abs scale and a 16-way nearest-neighbor scan over a
    shared fixed codebook), it is expected to track the Python port's own
    results unusually closely among this repo's ``*_cpp`` ports.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.
    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            IQ4_NL quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_iq4_nl(model.SerializeToString()))


def apply_gguf_q4_0_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gguf_q4_0_quantization`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into llama.cpp's legacy Q4_0 format -- one
    symmetric scale per 32-element block of the weight's own flattened
    storage, no separate min (``dequant = (code - 8) * d``). See
    :func:`onnxsim.apply_gguf_q4_0_quantization`'s own docstring for the
    full rationale and this format's own encoder-provenance honesty note.

    Unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            Q4_0 quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_gguf_q4_0_quantization(model.SerializeToString())
    )


def apply_gguf_q4_1_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gguf_q4_1_quantization`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into llama.cpp's legacy Q4_1 format -- one scale and
    one explicit min per 32-element block of the weight's own flattened
    storage (``dequant = code * d + m``). See
    :func:`onnxsim.apply_gguf_q4_1_quantization`'s own docstring for the
    full rationale and this format's own encoder-provenance honesty note.

    Unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            Q4_1 quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_gguf_q4_1_quantization(model.SerializeToString())
    )


def apply_gguf_q5_0_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gguf_q5_0_quantization`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into llama.cpp's legacy Q5_0 format -- one
    symmetric scale per 32-element block of the weight's own flattened
    storage, no separate min (``dequant = (code - 16) * d``). See
    :func:`onnxsim.apply_gguf_q5_0_quantization`'s own docstring for the
    full rationale and this format's own encoder-provenance honesty note.

    Unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            Q5_0 quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_gguf_q5_0_quantization(model.SerializeToString())
    )


def apply_gguf_q5_1_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gguf_q5_1_quantization`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into llama.cpp's legacy Q5_1 format -- one scale and
    one explicit min per 32-element block of the weight's own flattened
    storage (``dequant = code * d + m``). See
    :func:`onnxsim.apply_gguf_q5_1_quantization`'s own docstring for the
    full rationale and this format's own encoder-provenance honesty note.

    Unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            Q5_1 quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_gguf_q5_1_quantization(model.SerializeToString())
    )


def apply_gguf_q8_0_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gguf_q8_0_quantization`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into llama.cpp's Q8_0 format -- one plain
    32-element block sharing a single fp16 scale, signed 8-bit code with
    no bias/min (``dequant = code * d``). See
    :func:`onnxsim.apply_gguf_q8_0_quantization`'s own docstring for the
    full rationale and this format's own encoder-provenance honesty note.

    Unlike :func:`onnxsim.apply_gguf_q8_0_quantization` and unlike
    :func:`simplify`, this does not support an ``include_conv`` option,
    and this does not run shape inference, constant folding or any other
    simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            Q8_0 quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_gguf_q8_0_quantization(model.SerializeToString())
    )


def apply_gguf_q2_k_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gguf_q2_k_quantization`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into llama.cpp's Q2_K K-quant format -- a
    256-element super-block split into 16 sub-blocks of 16, each with its
    own asymmetric affine (scale, min) pair re-quantized to 4-bit codes
    relative to one shared super-block (d, dmin) reference pair, times a
    2-bit element code (``dequant = d*sc_j*q - dmin*m_j``). See
    :func:`onnxsim.apply_gguf_q2_k_quantization`'s own docstring for the
    full rationale and this format's own encoder-provenance honesty note.

    Unlike :func:`onnxsim.apply_gguf_q2_k_quantization` and unlike
    :func:`simplify`, this does not support an ``include_conv`` option,
    and this does not run shape inference, constant folding or any other
    simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            Q2_K quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_gguf_q2_k_quantization(model.SerializeToString())
    )


def apply_gguf_q3_k_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gguf_q3_k_quantization`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into llama.cpp's Q3_K K-quant format -- a
    256-element super-block split into 16 sub-blocks of 16, each with its
    own 6-bit unsigned scale code (restricted by this encoder to a
    non-negative offset) times one shared float16 super-block scale,
    times the format's own asymmetric 3-bit element code
    (``dequant = d_all * (sc_j - 32) * q``). See
    :func:`onnxsim.apply_gguf_q3_k_quantization`'s own docstring for the
    full rationale and this format's own encoder-provenance honesty note.

    Unlike :func:`onnxsim.apply_gguf_q3_k_quantization` and unlike
    :func:`simplify`, this does not support an ``include_conv`` option,
    and this does not run shape inference, constant folding or any other
    simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            Q3_K quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_gguf_q3_k_quantization(model.SerializeToString())
    )


def apply_gguf_q4_k_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gguf_q4_k_quantization`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into llama.cpp's Q4_K K-quant format -- a
    256-element super-block split into 8 sub-blocks of 32, each with its
    own asymmetric affine (scale, min) pair re-quantized to 6-bit codes
    relative to one shared super-block (d, dmin) reference pair, times a
    4-bit element code (``dequant = d*sc_j*q - dmin*m_j``). See
    :func:`onnxsim.apply_gguf_q4_k_quantization`'s own docstring for the
    full rationale and this format's own encoder-provenance honesty note.

    Unlike :func:`onnxsim.apply_gguf_q4_k_quantization` and unlike
    :func:`simplify`, this does not support an ``include_conv`` option,
    and this does not run shape inference, constant folding or any other
    simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            Q4_K quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_gguf_q4_k_quantization(model.SerializeToString())
    )


def apply_gguf_q5_k_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gguf_q5_k_quantization`:
    identical to :func:`onnxsim.apply_gguf_q4_k_quantization_cpp` except
    the element code is 5 bits (``[0, 31]``) rather than Q4_K's 4. See
    :func:`onnxsim.apply_gguf_q5_k_quantization`'s own docstring for the
    full rationale and this format's own encoder-provenance honesty note.

    Unlike :func:`onnxsim.apply_gguf_q5_k_quantization` and unlike
    :func:`simplify`, this does not support an ``include_conv`` option,
    and this does not run shape inference, constant folding or any other
    simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            Q5_K quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_gguf_q5_k_quantization(model.SerializeToString())
    )


def apply_gguf_ternary_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gguf_ternary_quantization`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight using BitNet b1.58's published absmean ternary rule,
    as shipped by llama.cpp's GGUF TQ1_0/TQ2_0 tensor types -- one shared
    scale ``d = mean(|block|)`` per 256-element block of the weight's own
    flattened storage, each element restricted to ``{-1, 0, +1}`` times
    that scale. See :func:`onnxsim.apply_gguf_ternary_quantization`'s own
    docstring for the full rationale and this format's own honesty note.

    Unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            ternary quantize-dequantize round-tripped float32 version,
            stored under a *new* initializer (the original initializer is
            left in the graph, unused). A model with no matching layer is
            returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_gguf_ternary_quantization(model.SerializeToString())
    )


def apply_fp6_llm_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_fp6_llm_quantization`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight using FP6-LLM's E3M2 6-bit floating-point format --
    one shared scale ``s = max(|block|) / 28.0`` per 64-element block of
    the weight's own flattened storage, each element snapped to the
    nearest of FP6 E3M2's 64 representable values times that scale. See
    :func:`onnxsim.apply_fp6_llm_quantization`'s own docstring for the
    full rationale. Unlike its Python counterpart, this port only
    implements the paper's primary E3M2 format.

    Unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            FP6 quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_fp6_llm(model.SerializeToString()))


def apply_gguf_q6_k_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_gguf_q6_k_quantization`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into llama.cpp's Q6_K K-quant format -- a
    256-element super-block split into 16 sub-blocks of 16 elements, each
    sharing one 8-bit scale code times a shared float16 super-block scale,
    times a symmetric 6-bit element code. See
    :func:`onnxsim.apply_gguf_q6_k_quantization`'s own docstring for the
    full rationale and this format's own encoder-provenance honesty note.

    Unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass.

    Layers with a non-constant, non-2-D weight are left untouched. Consider
    calling :func:`simplify` before and/or after to clean up the graph.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            Q6_K quantize-dequantize round-tripped float32 version, stored
            under a *new* initializer (the original initializer is left in
            the graph, unused). A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_gguf_q6_k_quantization(model.SerializeToString())
    )


def apply_leptoquant_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_leptoquant`: weight-only
    quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight into AngelSlim's LeptoQuant format -- a 128x128-tile
    FLOAT8E4M3FN round trip, refined by a 5-point outlier-fraction grid
    search per tile. See :func:`onnxsim.apply_leptoquant`'s own docstring
    for the full rationale.

    Unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            LeptoQuant round-tripped float32 version, stored under a
            *new* initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_leptoquant(model.SerializeToString()))


def quantize_weight_only_nf4_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_nf4`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into bitsandbytes' NF4 format -- a fixed 16-value
    non-uniform codebook, one scale per 64-element (output-channel,
    K-block) group. See :func:`onnxsim.quantize_weight_only_nf4`'s own
    docstring for the full rationale.

    Unlike :func:`onnxsim.quantize_weight_only_nf4`, this port folds the
    round trip directly into a replacement float32 initializer instead of
    building it out of Gather/Reshape/Mul graph nodes, and unlike
    :func:`simplify`, this does not run shape inference, constant folding
    or any other simplification pass.

    A layer whose reduction dimension isn't evenly divisible by 64 is
    left completely untouched.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            NF4 round-tripped float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.quantize_weight_only_nf4(model.SerializeToString()))


def quantize_weight_only_if4_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_if4`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into the IF4 format -- per (output-channel,
    16-element K-block), tries both a plain INT4 grid and MXFP4's own
    E2M1 codebook and keeps whichever reconstructs that block with lower
    MSE. See :func:`onnxsim.quantize_weight_only_if4`'s own docstring for
    the full rationale.

    Unlike :func:`onnxsim.quantize_weight_only_if4`, this port folds the
    round trip directly into a replacement float32 initializer instead of
    building it out of Cast/Gather/Reshape/Mul graph nodes, and unlike
    :func:`simplify`, this does not run shape inference, constant folding
    or any other simplification pass.

    A layer whose reduction dimension isn't evenly divisible by 16 is
    left completely untouched.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            IF4 round-tripped float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.quantize_weight_only_if4(model.SerializeToString()))


def quantize_weight_only_nvfp4_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_nvfp4`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight into NVIDIA's NVFP4 format -- shares OCP MXFP4's
    E2M1 element codebook, with a two-level (per-tensor global scale,
    E4M3-rounded per-block scale) rule. See
    :func:`onnxsim.quantize_weight_only_nvfp4`'s own docstring for the
    full rationale.

    Unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass. A layer whose reduction
    dimension isn't evenly divisible by 16 is left completely untouched.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            NVFP4 round-tripped float32 version (via the same
            Cast/Gather/Reshape/Mul graph shape
            :func:`onnxsim.quantize_weight_only_mxfp4_cpp` already uses).
            A model with no matching layer is returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.quantize_weight_only_nvfp4(model.SerializeToString())
    )


def apply_deepseek_fp8_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of the *weight* half of
    :func:`onnxsim.apply_deepseek_fp8`: weight-only quantizes every
    MatMul/vanilla-Gemm layer with a constant 2-D float32 weight via a
    real FLOAT8E4M3FN round trip, one scale per 128x128 tile. See
    :func:`onnxsim.apply_deepseek_fp8`'s own docstring for the full
    rationale.

    Unlike :func:`onnxsim.apply_deepseek_fp8`, this port does not also
    insert the activation-quantization (W8A8) graph rewrite -- weight-only
    scope, matching every other data-free ``*_cpp`` port in this repo.
    Unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            block-FP8 round-tripped float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_deepseek_fp8(model.SerializeToString()))


def apply_kmeans_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_kmeans`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight by fitting a 16-centroid codebook per layer via
    Lloyd's algorithm. See
    :func:`onnxsim.quantize_weight_only_kmeans`'s own docstring for the
    full rationale.

    Unlike :func:`onnxsim.quantize_weight_only_kmeans`, this port folds
    the round trip directly into a replacement float32 initializer
    instead of building it out of Cast/Gather graph nodes, and unlike
    :func:`simplify`, this does not run shape inference, constant folding
    or any other simplification pass.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            k-means-fitted float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_kmeans_quantization(model.SerializeToString()))


def apply_hqq_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_int4_hqq`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight via an asymmetric affine INT4 quantizer whose
    zero-point is IRLS-refined per (output-channel, 32-element K-block)
    group. See :func:`onnxsim.quantize_weight_only_int4_hqq`'s own
    docstring for the full rationale.

    Unlike :func:`onnxsim.quantize_weight_only_int4_hqq`, this port folds
    the round trip directly into a replacement float32 initializer
    instead of building a real ``DequantizeLinear`` node with packed
    UINT4 codes, and unlike :func:`simplify`, this does not run shape
    inference, constant folding or any other simplification pass.

    A layer whose reduction dimension isn't evenly divisible by 32 is
    left completely untouched.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            HQQ-quantized float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_hqq(model.SerializeToString()))


def apply_ibert_gelu_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_ibert_gelu`: unlike every
    other ``*_cpp`` port in this module, this is not a weight quantizer
    -- it replaces every standalone ``Erf`` node anywhere in the graph
    with I-BERT's own closed-form second-order polynomial approximation
    (``sign(x) * (a*(clip(|x|, max=-b)+b)**2 + c)``), built from ordinary
    Abs/Clip/Add/Mul/Sign ops. See :func:`onnxsim.apply_ibert_gelu`'s own
    docstring for the full rationale.

    Both sides build the exact same five ONNX ops from the exact same
    three float32 constants, so this port is expected to be numerically
    identical to :func:`onnxsim.apply_ibert_gelu` up to onnxruntime's own
    float32 evaluation of the resulting graph.

    :param model: the original onnx ModelProto or file path
    :returns: ``model`` with every standalone Erf node replaced. A model
            with no Erf node is returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_ibert_gelu(model.SerializeToString()))


def apply_ibert_softmax_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_ibert_softmax`: unlike every
    weight-only ``*_cpp`` port in this module, this is a nonlinear-
    activation rewrite, not a weight quantizer -- it replaces every
    standalone ``Softmax`` node with I-BERT's own integer-friendly
    exp-approximation subgraph (a base-2 exponent decomposition plus a
    quadratic polynomial fit, then an ordinary division for the row
    normalization). See :func:`onnxsim.apply_ibert_softmax`'s own
    docstring for the full rationale and its own scope note (the final
    normalization is a plain division, not the paper's own integer-only
    iterative reciprocal).

    Needs opset 18+ (``ReduceMax``/``ReduceSum``'s axes-as-input form); a
    model below that opset is returned unchanged.

    :param model: the original onnx ModelProto or file path
    :returns: ``model`` with every standalone Softmax node replaced. A
            model with no Softmax node, or below opset 18, is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_ibert_softmax(model.SerializeToString()))


def apply_adpq_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_adpq`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight via AdpQ's own calibration-free salient/non-
    salient split -- per (output-channel, 128-element K-group), a median/
    MAD-based adaptive threshold (borrowed from Adaptive LASSO) decides
    which elements are salient (reconstructed exactly) versus quantized
    to a symmetric 7-level-per-side grid. See
    :func:`onnxsim.quantize_weight_only_adpq`'s own docstring for the
    full rationale.

    Unlike :func:`onnxsim.quantize_weight_only_adpq`, this port folds the
    round trip directly into a replacement float32 initializer instead of
    building a real ``DequantizeLinear``+``ScatterND``+``Add`` graph
    rewrite, and unlike :func:`simplify`, this does not run shape
    inference, constant folding or any other simplification pass.

    A layer whose reduction dimension isn't evenly divisible by 128 is
    left completely untouched.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            AdpQ-quantized float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_adpq(model.SerializeToString()))


def apply_icquant_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_icquant`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight via ICQuant's own single-outlier-per-block scheme
    -- per (output-channel, 32-element K-block), the single largest-
    magnitude element is excluded from the block's own scale computation
    and reconstructed exactly; the rest quantize to a symmetric
    7-level-per-side grid. See
    :func:`onnxsim.quantize_weight_only_icquant`'s own docstring for the
    full rationale and its own note that the paper's "combinadic" index
    encoding is a pure storage detail with no effect on the reconstructed
    values (this port skips it entirely).

    Unlike :func:`onnxsim.quantize_weight_only_icquant`, this port folds
    the round trip directly into a replacement float32 initializer
    instead of building a real INT4/``DequantizeLinear``/``ScatterND``/
    ``MatMul``/``Add`` graph rewrite, and unlike :func:`simplify`, this
    does not run shape inference, constant folding or any other
    simplification pass.

    A layer whose reduction dimension isn't evenly divisible by 32 is
    left completely untouched.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            ICQuant-quantized float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_icquant(model.SerializeToString()))


def apply_olive_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_olive`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight via OliVe's own outlier-victim pair scheme -- per
    (output-channel, 32-element K-block), adjacent element pairs with
    exactly one outlier member become an OVP pair (the outlier gets a
    wider code, its neighbor a narrower "victim" code at the same total
    bit budget as two ordinary elements). See
    :func:`onnxsim.quantize_weight_only_olive`'s own docstring for the
    full rationale.

    Unlike :func:`onnxsim.quantize_weight_only_olive`, this port folds
    the round trip directly into a replacement float32 initializer
    instead of building a real ``DequantizeLinear`` x2 + ``Cast`` +
    ``Where`` + ``MatMul``[+``Add``] graph rewrite, and unlike
    :func:`simplify`, this does not run shape inference, constant folding
    or any other simplification pass.

    A layer whose reduction dimension isn't evenly divisible by 32 is
    left completely untouched.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            OliVe-quantized float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_olive(model.SerializeToString()))


def apply_aqlm_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_aqlm`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight via AQLM's own additive/residual multi-codebook
    scheme -- each (output-channel, 8-element K-block) group is
    reconstructed as the sum of 2 codebook lookups, each codebook fit via
    greedy residual Lloyd's-algorithm k-means. See
    :func:`onnxsim.quantize_weight_only_aqlm`'s own docstring for the
    full rationale.

    Unlike :func:`onnxsim.quantize_weight_only_aqlm`, this port folds the
    reconstruction directly into a replacement float32 initializer
    instead of building a Gather+Add-chain graph rewrite, and unlike
    :func:`simplify`, this does not run shape inference, constant folding
    or any other simplification pass. Unlike
    :func:`onnxsim.apply_kmeans_quantization_cpp`'s own scalar
    percentile-based initialization (which matches its Python counterpart
    exactly outside a narrow edge case), this port's own k-means
    initialization is deterministically magnitude-sorted rather than
    :func:`onnxsim.quantize_weight_only_aqlm`'s own seeded-random sample
    -- a genuinely different (though algorithmically identical) fit, not
    merely a floating-point-order divergence.

    A layer whose reduction dimension isn't evenly divisible by 8 is
    left completely untouched.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            AQLM-reconstructed float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_aqlm(model.SerializeToString()))


def apply_drop_by_drop_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_drop_by_drop`:
    weight-only quantizes every MatMul/vanilla-Gemm layer with a constant
    2-D float32 weight via Drop-by-Drop's own additive multi-bitwidth
    codebook scheme -- each (output-channel, 8-element K-block) group is
    reconstructed by 4 additive residual codebook stages, each an
    *importance-weighted* Lloyd's-algorithm k-means fit (weighted by the
    group's own fixed original-magnitude RMS). See
    :func:`onnxsim.quantize_weight_only_drop_by_drop`'s own docstring for
    the full rationale.

    Unlike :func:`onnxsim.quantize_weight_only_drop_by_drop`, this port
    folds the reconstruction directly into a replacement float32
    initializer instead of building a Gather+Add-chain graph rewrite, and
    unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass. Like
    :func:`onnxsim.apply_aqlm_cpp`'s own divergence, this port's own
    k-means initialization is deterministically magnitude-sorted rather
    than :func:`onnxsim.quantize_weight_only_drop_by_drop`'s own
    seeded-random sample, applied to each of the 4 weighted fits.

    A layer whose reduction dimension isn't evenly divisible by 8 is
    left completely untouched.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            Drop-by-Drop-reconstructed float32 version, stored under a
            *new* initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_drop_by_drop(model.SerializeToString()))


def apply_lo_bcq_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of the weight-side half of
    :func:`onnxsim.quantize_weight_only_lo_bcq`: weight-only quantizes
    every MatMul/vanilla-Gemm layer with a constant 2-D float32 weight via
    LO-BCQ's own block-clustered codebook scheme -- blocks of the
    reduction dimension are clustered by their own [mean, std] feature
    vector into 4 groups, then each cluster fits its own small codebook
    from only its own currently-assigned blocks, alternating cluster
    refit and block reassignment. See
    :func:`onnxsim.quantize_weight_only_lo_bcq`'s own docstring for the
    full rationale.

    Unlike :func:`onnxsim.quantize_weight_only_lo_bcq`, this port folds
    the reconstruction directly into a replacement float32 initializer
    instead of building a Gather/GatherElements/Reshape graph rewrite,
    and unlike :func:`simplify`, this does not run shape inference,
    constant folding or any other simplification pass. This port's own
    per-cluster/fallback 1-D codebook fits are deterministic
    (percentile-derived centroids, matching
    :func:`onnxsim.apply_kmeans_quantization_cpp`'s own precedent), while
    its own block-clustering step's initialization is deterministically
    feature-norm-sorted rather than
    :func:`onnxsim.quantize_weight_only_lo_bcq`'s own seeded-random
    sample (matching :func:`onnxsim.apply_aqlm_cpp`'s own precedent).

    A layer whose reduction dimension isn't evenly divisible by 32 is
    left completely untouched.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            LO-BCQ-reconstructed float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_lo_bcq(model.SerializeToString()))


def apply_quip_sharp_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_quip_sharp`: weight-only
    quantizes every MatMul/vanilla-Gemm layer with a constant 2-D
    float32 weight via QuIP#'s own incoherence-processing-plus-E8-lattice
    scheme -- conjugates the weight by a pair of random orthogonal
    matrices (``Wtilde = V @ W @ U``) so its entries look i.i.d.
    Gaussian, then jointly quantizes each 8-element group to the nearest
    E8 lattice point. See :func:`onnxsim.apply_quip_sharp`'s own
    docstring for the full rationale.

    Unlike :func:`onnxsim.apply_quip_sharp`'s own graph rewrite (which
    keeps U, V and the packed INT4 lattice codes as explicit initializers
    and new MatMul nodes), this port folds the entire
    rotate/quantize/rotate-back sandwich into a single replacement weight
    initializer -- exact, not an approximation, since U/V are square and
    only sandwich the weight -- and unlike :func:`simplify`, this does
    not run shape inference, constant folding or any other
    simplification pass. This port's own random orthogonal matrices are
    generated via a Gram-Schmidt construction (the same precedent
    :func:`onnxsim.apply_quarot_cpp` already establishes for this
    module), not :func:`onnxsim.apply_quip_sharp`'s own
    QR-with-sign-correction one -- both are independently Haar-uniform;
    cross-language bit parity is explicitly not a goal.

    A layer whose reduction dimension isn't evenly divisible by 8 is
    left completely untouched.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by its
            QuIP#-reconstructed float32 version, stored under a *new*
            initializer. A model with no matching layer is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_quip_sharp(model.SerializeToString()))


def apply_attention_quantization_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_attention_quantization`:
    quantizes the decomposed attention subgraph's own Q/K/V operands
    (data-free, per-token dynamic INT8, ``scale = max(|x|, axis=-1) /
    127``) and the Softmax output itself (a fixed UINT8 scale of ``1/255``,
    since a Softmax output's range is guaranteed ``[0, 1]``). See
    :func:`onnxsim.apply_attention_quantization`'s own docstring for the
    full rationale.

    Unlike every weight-only ``*_cpp`` port in this module, this is not a
    fold-to-initializer pass -- none of the four quantized tensors is a
    constant weight, so this builds real new quantize/dequantize graph
    nodes instead. Unlike :func:`simplify`, this does not run shape
    inference, constant folding or any other simplification pass. No
    RNG/fitting step anywhere in this technique, so this port is expected
    to track the pure-Python reference closely, up to ordinary
    floating-point summation-order differences.

    Only the common, decomposed
    ``MatMul(Q,Kt) -> [Mul] -> [Add] -> Softmax -> MatMul(_,V)`` subgraph is
    matched, on an opset 18+ model (``ReduceMax``'s axes-as-input form).

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched attention subgraph's own Q/K/V
            and Softmax-output operands replaced by an INT8
            quantize-dequantize round trip. A model with no matching
            subgraph, or an opset older than 18, is returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_attention_quantization(model.SerializeToString())
    )


def apply_zeroquant_cpp(
    model: Union[str, onnx.ModelProto],
    block_size: int = 32,
    epsilon: float = 1e-12,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_zeroquant`: applies
    ZeroQuant-style W8A8 quantization (Yao et al., 2022) -- group-wise INT8
    weight quantization paired with per-token dynamic INT8 activation
    quantization, executed as a real ``int8 x int8`` integer matmul -- to
    every MatMul/vanilla-Gemm layer with a constant 2-D float32 weight whose
    reduction dimension ``K`` is divisible by ``block_size``. See
    :func:`onnxsim.apply_zeroquant`'s own docstring for the full rationale.

    Unlike every other per-token-dynamic-INT8 activation-quantization pass
    in this module (which immediately dequantizes back to float32,
    simulating precision loss), this one feeds the quantized activation into
    a genuine ``MatMulInteger`` -- so, like :func:`onnxsim.apply_quarot_cpp`,
    this cannot fold to a single replacement weight (the activation's own
    per-token scale is a runtime value) and builds real new graph nodes
    instead. Unlike :func:`simplify`, this does not run shape inference,
    constant folding or any other simplification pass. No RNG/fitting step
    anywhere in this technique, so this port is expected to track the
    pure-Python reference closely, up to ordinary floating-point
    summation-order differences.

    Needs no calibration data: the weight's per-group scales come from the
    weight's own static values, and the activation's per-token scale is
    computed fresh at graph-run time from that token's own values.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param block_size: elements per weight quantization group along ``K``,
            matching :func:`onnxsim.quantize_weight_only_int8_block`'s own
            default
    :param epsilon: floor applied to a weight group's own max-abs value
            (and, at graph-run time, a token's own quantization range)
            before using it as a scale, avoiding a divide-by-zero on an
            all-zero group or token
    :returns: ``model`` with every matched layer's weight and activation
            replaced by a group-wise/per-token INT8 ``MatMulInteger``
            pipeline (plus the original bias, if any). Layers with a
            non-constant, non-2-D weight, or a reduction dimension not
            divisible by ``block_size``, are left untouched. A model whose
            opset is older than 18, or with ``block_size`` not a positive
            divisor within the safe int32-accumulation bound, is returned
            unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_zeroquant(model.SerializeToString(), block_size, epsilon)
    )


def apply_intactkv_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_intactkv`: not a quantizer, but
    a *companion* pass (IntactKV, Liu et al., 2024) that splits a KV-cache
    stream's own fixed-length leading "pivot" prefix (attention-sink
    tokens, disproportionately sensitive to quantization error) out into
    its own always-exact stream, so a following KV-cache quantizer can
    leave the pivots untouched forever and quantize only the remaining,
    still-growing "rest" of the cache. See
    :func:`onnxsim.apply_intactkv`'s own docstring for the full rationale.

    Matches a ``Concat(past, new, axis=seq)`` KV-cache stream (``past``: a
    float32 graph input consumed by nothing else, whose Concat output is
    directly a graph output) and rewrites it into a new fixed-size
    ``*_pivot`` graph input/output plus an ordinary ``*_rest`` stream,
    reconstructed back under the ORIGINAL ``present_*`` name/binding via a
    new ``Concat`` node -- real graph input/output surgery, not a
    fold-to-initializer or a same-shape node rewrite. Unlike
    :func:`simplify`, this does not run shape inference, constant folding
    or any other simplification pass. No RNG/fitting step anywhere in this
    technique (a closed-form structural transformation), so this port is
    expected to track the pure-Python reference exactly.

    This port hardcodes :func:`onnxsim.apply_intactkv`'s own default
    ``num_pivot_tokens=4`` rather than exposing it as a parameter.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched KV-cache stream split into a
            ``*_pivot``/``*_rest`` pair, reconstructed under the original
            name. A model with no matching stream is returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_intactkv(model.SerializeToString()))


def apply_kbvq_moe_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_kbvq_moe`: fits a KLT (PCA)
    basis shared across a ``com.microsoft::MoE`` router group's own experts
    (KBVQ-MoE, Xu et al., 2026), then vector-quantizes each expert's own
    residual against that shared basis with an ordinary per-expert k-means
    codebook. See :func:`onnxsim.apply_kbvq_moe`'s own docstring for the
    full rationale.

    Unlike :func:`simplify`, this does not run shape inference, constant
    folding or any other simplification pass. Only FLOAT32
    ``fc1_experts_weights``/``fc2_experts_weights`` are quantized
    (FLOAT16/BFLOAT16 left untouched, per-tensor, matching
    :func:`onnxsim.apply_kbvq_moe`'s own scope exactly). This port's own
    shared-basis fit has no RNG (an ordinary deterministic SVD); its
    per-expert residual codebook reuses this module's own
    :func:`onnxsim.apply_kmeans_quantization_cpp` k-means fit, whose only
    divergence from the pure-Python reference is the same deterministic,
    percentile-initialized scheme that function's own docstring already
    documents. This port hardcodes :func:`onnxsim.apply_kbvq_moe`'s own
    defaults (``rank=4``, ``bits=4``, ``kmeans_iters=20``) rather than
    exposing them as parameters.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched MoE router group's own FLOAT32
            expert weights replaced by their shared-basis-plus-residual-
            codebook reconstruction, stored under *new* initializers. A
            model with no matching MoE node is returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_kbvq_moe(model.SerializeToString()))


def quantize_weight_only_llm_fp4_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_llm_fp4`
    (LLM-FP4, Liu et al., 2023): searches, per matched MatMul/vanilla-Gemm
    layer's constant 2-D float32 weight, an exponent/mantissa FP4 format
    split (E1M2/E2M1/E3M0) and a per-block real-valued clip-ratio scale,
    picking whichever combination minimizes reconstruction MSE. Needs no
    calibration data. This port hardcodes
    :func:`onnxsim.quantize_weight_only_llm_fp4`'s own defaults
    (``block_size=32``, ``formats=("e1m2", "e2m1", "e3m0")``,
    ``num_scale_candidates=17``, ``min_clip_ratio=0.5``) rather than
    exposing them as parameters, and omits its own ``skip_names``
    parameter.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by a
            searched FP4 codebook/codes/scale reconstruction feeding the
            original node; layers with a non-constant, non-2-D, or
            non-block-divisible weight are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.quantize_weight_only_llm_fp4(model.SerializeToString())
    )


def apply_llm_fp4_activation_quantization_cpp(
    model: Union[str, onnx.ModelProto],
    epsilon: float = 1e-12,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_llm_fp4_activation_quantization`:
    completes W4A4 for every :func:`onnxsim.quantize_weight_only_llm_fp4`-
    quantized layer by inserting a **per-token, data-free** FP4 quantize/
    dequantize round-trip on that layer's own activation input, reusing
    that same layer's own already-baked codebook. This is NOT the LLM-FP4
    paper's own activation-quantization design -- read
    :func:`onnxsim.apply_llm_fp4_activation_quantization`'s own docstring
    (in particular its "Honesty note") before using; the paper's own
    calibrated per-tensor scheme is
    :func:`onnxsim.apply_llm_fp4_activation_quantization_per_tensor_cpp`.
    Needs no calibration data.

    :param model: the original or weight-quantized onnx ModelProto or
            file path -- must already have been passed through
            :func:`onnxsim.quantize_weight_only_llm_fp4`/
            :func:`onnxsim.quantize_weight_only_llm_fp4_cpp` for this
            function to do anything
    :param epsilon: floor applied to a token's own max-abs value before
            using it as a scale, avoiding a divide-by-zero on an
            all-zero token
    :returns: ``model`` with every already-FP4-weight-quantized layer's
            activation input replaced by its own per-token FP4 quantize/
            dequantize round-trip; a layer whose weight input isn't fed
            by exactly :func:`onnxsim.quantize_weight_only_llm_fp4`'s own
            dequantization pattern is left completely untouched. A model
            with an opset older than 18 is returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_llm_fp4_activation_quantization(model.SerializeToString(), epsilon)
    )


def apply_llm_fp4_activation_quantization_per_tensor_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    clip_ratios: Optional[Sequence[float]] = None,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of
    :func:`onnxsim.apply_llm_fp4_activation_quantization_per_tensor`: the
    **calibrated per-tensor** half of the LLM-FP4 paper's own activation-
    quantization design. Fits one real-valued per-tensor scale per
    already-:func:`onnxsim.quantize_weight_only_llm_fp4`-quantized
    layer's activation from calibration data (the same clip-ratio grid
    search the weight side already uses, applied to the whole captured
    activation as a single group), bakes it in as a constant initializer,
    then inserts a *static* FP4 quantize/dequantize round-trip. This
    implements only the quantizer half of the paper's own design -- the
    migration half (pushing per-channel activation outliers into the
    preceding weight/LayerNormalization) is the caller's job, via
    :func:`onnxsim.apply_smoothquant`/
    :func:`onnxsim.apply_outlier_suppression` run first -- see
    :func:`onnxsim.apply_llm_fp4_activation_quantization_per_tensor`'s
    own docstring for the full three-call sequence and honesty note.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` -- a
    live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++.

    :param model: a weight-quantized onnx ModelProto or file path -- must
            already have been passed through
            :func:`onnxsim.quantize_weight_only_llm_fp4`/
            :func:`onnxsim.quantize_weight_only_llm_fp4_cpp` for this
            function to do anything
    :param calibration_data: representative input batches to fit each
            activation's scale on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param clip_ratios: per-tensor clip-ratio candidates to grid-search;
            defaults to 17 points evenly spaced over ``[0.5, 1.0]``
            (:func:`onnxsim.quantize_weight_only_llm_fp4`'s own weight-side
            default grid)
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every already-FP4-weight-quantized layer's
            activation input replaced by a static, per-tensor FP4
            quantize/dequantize round-trip driven by a constant scale
            initializer; a layer whose weight input isn't fed by exactly
            :func:`onnxsim.quantize_weight_only_llm_fp4`'s own
            dequantization pattern, or whose activation no calibration
            batch reached (or for which every captured value was zero or
            non-finite), is left completely untouched. A model with an
            opset older than 13 is returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_llm_fp4_activation_quantization_per_tensor(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            None if clip_ratios is None else [float(r) for r in clip_ratios],
        )
    )


def apply_bwa_ptq_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    group_size: int = 128,
    max_em_iters: int = 10,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_bwa_ptq`: Binary Weight-
    Activation PTQ (Song et al., 2025, ACL Findings), weight side only
    (W(1+1)) -- binarizes every matched MatMul/vanilla-Gemm layer's
    weight to exactly 1 sign bit + 1 group-select bit/element via
    Hessian-weighted two-scale binary EM. See
    :func:`onnxsim.apply_bwa_ptq`'s own module docstring for the full
    technique and its documented scope (the paper's own A(1x4)
    activation-side bit-plane decomposition is deliberately not ported --
    it is numerically equivalent to ordinary calibrated INT4 activation
    quantization).

    Same real calibration machinery as :func:`onnxsim.apply_billm_cpp` -- a
    live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++ (see ``ApplyBwaPtq``
    in ``bwa_ptq_entry.h`` for the full scope, including its accepted
    numerical scope).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian diagonal from -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param group_size: contiguous reduction-axis elements sharing one
            pair of candidate scales
    :param max_em_iters: upper bound on EM iterations per group
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight replaced by its
            two-scale binary reconstruction.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_bwa_ptq(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            group_size,
            max_em_iters,
        )
    )


def apply_pruning_finetune_cpp(
    original_model: Union[str, onnx.ModelProto],
    pruned_model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    reg_param: float = 1e-2,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_pruning_finetune`: for every
    surviving MatMul/vanilla-Gemm layer present (by node output name) in
    both ``original_model`` and ``pruned_model``, re-solves its weight
    (and bias, if it has one) as a closed-form ridge-regression fit
    against ``original_model``'s own real activations -- see
    :func:`onnxsim.apply_pruning_finetune`'s own module docstring for the
    full technique, its "exactness argument", and its scope boundaries.

    Same real calibration machinery as :func:`onnxsim.apply_gptq_cpp` -- a
    live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through ``original_model`` in C++ (see
    ``ApplyPruningFinetune`` in ``finetune_entry.h`` for the full scope,
    including its accepted numerical scope: the dense linear solve at
    this fit's heart uses a scalar double-precision Gaussian-elimination
    kernel rather than LAPACK).

    :param original_model: the pre-pruning onnx ModelProto or file path
    :param pruned_model: ``original_model`` after
            :func:`onnxsim.apply_structured_pruning_cpp`/
            :func:`onnxsim.apply_attention_head_pruning_cpp` (or their
            Wanda-calibrated counterparts) -- onnx ModelProto or file
            path
    :param calibration_data: representative input batches to fit each
            layer's own reconstruction on -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param reg_param: ridge-regression regularization strength, relative
            to the calibration data's own scale
    :param providers: onnxruntime execution providers to run
            ``original_model`` on when capturing calibration activations
    :returns: ``pruned_model`` with every matched layer's weight (and
            bias, if it has one) replaced by its fine-tuned fit; a layer
            whose weight isn't a clean row/column subsequence of its own
            ``original_model`` counterpart, or whose captured activation
            has no feature axis at all, is left completely untouched.
    """
    if isinstance(original_model, str):
        original_model = onnx.load(original_model, load_external_data=False)
    if isinstance(pruned_model, str):
        pruned_model = onnx.load(pruned_model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            original_model, num_samples=num_samples, seed=seed
        )
    # Same {input_name: TensorProto}-per-batch crossing convention as
    # apply_llm_int8_cpp -- see that function's own comment.
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in batch.items()
        }
        for batch in calibration_data
    ]
    return onnx.load_from_string(
        C.apply_pruning_finetune(
            _get_model_executor(providers),
            original_model.SerializeToString(),
            pruned_model.SerializeToString(),
            calibration_data_pb,
            reg_param,
        )
    )


def apply_qoq_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_weight_only_qoq` (QServe's
    QoQ quantization, Lin et al., 2024): progressive (INT8-then-INT4)
    block-wise weight quantization, folded into one combined per-group
    scale. Needs no calibration data. This port hardcodes
    :func:`onnxsim.quantize_weight_only_qoq`'s own defaults
    (``block_size=32``, ``int8_clip_max=119``) rather than exposing them as
    parameters.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched layer's weight replaced by
            ``DequantizeLinear(Wq, Ws, ...)`` feeding the original node;
            layers with a non-constant, non-2-D, or non-block-divisible
            weight are left untouched.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_qoq(model.SerializeToString()))


def apply_dsq_cpp(
    model: Union[str, onnx.ModelProto],
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_dsq` (D2Quant's Dual-Scale
    Quantizer, Yan et al., 2026): a per-input-channel auxiliary scale for
    SwiGLU/GLU down-projection weights, fit by alternating blockwise INT4
    quantization and a closed-form per-column weighted least squares, then
    absorbed into the paired up-projection's own raw weight. Needs no
    calibration data. This port hardcodes :func:`onnxsim.apply_dsq`'s own
    defaults (``block_size=32``, ``num_iterations=15``) rather than
    exposing them as parameters.

    :param model: the original (unquantized) onnx ModelProto or file path
    :returns: ``model`` with every matched down-projection's weight
            replaced by ``DequantizeLinear(Wq, Ws, ...)`` and the paired
            up-projection's own weight rescaled in place; a model with no
            matching SwiGLU/GLU block is returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(C.apply_dsq(model.SerializeToString()))


def quantize_embedding_binary_cpp(
    model: Union[str, onnx.ModelProto],
    output_name: Optional[str] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.quantize_embedding_binary`: binarizes
    a model's own embedding output at inference time (sign-thresholded,
    then 8 consecutive elements packed MSB-first into one ``uint8`` byte --
    exactly ``numpy.packbits(x > 0, axis=-1, bitorder="big")``). Needs no
    calibration data. Unlike every other ``*_cpp`` port in this module,
    this one targets a whole graph OUTPUT declaration directly rather than
    a matched node.

    :param model: the original (unquantized) onnx ModelProto or file path
    :param output_name: which graph output to binarize; if omitted, the
            graph must have exactly one float32 output (declining
            otherwise, rather than guessing)
    :returns: ``model`` with the resolved output's dtype changed to
            ``uint8`` and its last dimension divided by 8; a model whose
            output can't be resolved, whose last dimension isn't known
            statically, or isn't a multiple of 8, is returned unchanged
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    return onnx.load_from_string(
        C.quantize_embedding_binary(
            model.SerializeToString(), output_name if output_name is not None else ""
        )
    )


def apply_daq_cpp(
    base_model: Union[str, onnx.ModelProto],
    post_trained_model: Union[str, onnx.ModelProto],
    metric: str = "cosine",
    skip_names: Optional[Iterable[str]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_daq`: FP8-quantizes every
    matched MatMul/vanilla-Gemm weight of ``post_trained_model`` with a
    per-layer scalar scale chosen to preserve that layer's fine-tuning
    update (``ΔW = W_post - W_base``) rather than to minimize its raw
    reconstruction error. See :func:`onnxsim.apply_daq`'s own docstring
    for the full rationale.

    Unlike every other ``*_cpp`` port in this module, this one is
    data-free yet still takes two full model arguments (matched by node
    output name, the same correspondence assumption
    :func:`onnxsim.apply_gptq_cpp`/:func:`onnxsim.apply_qronos_cpp` make
    about their own two model arguments) -- neither model is ever run, and
    no calibration data or ``providers`` argument is needed.

    :param base_model: the pre-fine-tuning checkpoint (onnx ModelProto or
            file path)
    :param post_trained_model: the fine-tuned checkpoint (onnx ModelProto
            or file path); this is the model that gets quantized and
            returned
    :param metric: ``"cosine"`` (default) or ``"sign_preservation"`` --
            see :func:`onnxsim.apply_daq`'s own docstring
    :param skip_names: weight initializer names to leave unquantized even
            if otherwise eligible
    :returns: ``post_trained_model`` with every matched layer's weight
            replaced by its delta-aware FP8 round trip. A layer with no
            ``base_model`` counterpart, a shape mismatch, or a
            within-tolerance-zero ``ΔW`` is left completely untouched.
    """
    if isinstance(base_model, str):
        base_model = onnx.load(base_model, load_external_data=False)
    if isinstance(post_trained_model, str):
        post_trained_model = onnx.load(post_trained_model, load_external_data=False)
    skip_names = list(skip_names) if skip_names is not None else []
    return onnx.load_from_string(
        C.apply_daq(
            base_model.SerializeToString(),
            post_trained_model.SerializeToString(),
            metric,
            skip_names,
        )
    )


def apply_low_rank_compensation_cpp(
    float_model: Union[str, onnx.ModelProto],
    quantized_model: Union[str, onnx.ModelProto],
    rank: int = 8,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_low_rank_compensation`: adds a
    rank-``r`` low-rank correction (ZeroQuant-V2's LoRC, Yao et al., 2023)
    to every ``quantize_weight_only_int4``-quantized MatMul/Gemm layer
    matched (by node output name) between ``float_model`` and
    ``quantized_model``, canceling out that much of its existing
    quantization error via the Eckart-Young-optimal rank-``r``
    approximation of the reconstruction error matrix. See
    :func:`onnxsim.apply_low_rank_compensation`'s own docstring for the
    full rationale.

    Unlike every other ``*_cpp`` weight-only port in this module (which
    folds its correction into a replacement weight initializer), this
    one's correction is *additive to a matched layer's output*: it adds
    two new small ``MatMul`` nodes plus an ``Add`` node
    (``Y = X @ Wq_dequant + (X @ B) @ A``), leaving the layer's own
    existing INT4 weight and scale completely untouched. Like
    :func:`onnxsim.apply_daq_cpp`, this is data-free (no calibration data
    or ``providers`` argument) yet still takes two full model arguments.
    This port's own SVD is a hand-rolled Jacobi SVD rather than
    ``numpy.linalg.svd``'s own LAPACK routine (no linear-algebra library
    is linked into this codebase) -- individual singular vectors/values
    are not expected to match sign-for-sign or bit-for-bit, but the
    reconstructed rank-``r`` correction itself is (up to ordinary
    floating-point rounding, by the Eckart-Young theorem's own
    uniqueness).

    :param float_model: the original (unquantized) onnx ModelProto or
            file path
    :param quantized_model: a quantized version of ``float_model`` (onnx
            ModelProto or file path), produced by
            :func:`onnxsim.quantize_weight_only_int4`; this is the model
            that gets corrected and returned
    :param rank: the correction's rank ``r`` (clamped to
            ``min(r, N, K)`` per layer)
    :returns: ``quantized_model`` with every matched layer's output
            summed with a new rank-``r`` correction term. A layer with no
            ``float_model`` counterpart, a shape mismatch, or a clamped
            rank ``<= 0`` is left completely untouched.
    """
    if isinstance(float_model, str):
        float_model = onnx.load(float_model, load_external_data=False)
    if isinstance(quantized_model, str):
        quantized_model = onnx.load(quantized_model, load_external_data=False)
    return onnx.load_from_string(
        C.apply_low_rank_compensation(
            float_model.SerializeToString(),
            quantized_model.SerializeToString(),
            rank,
        )
    )


def apply_svdquant_cpp(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    rank: int = 32,
    block_size: int = 32,
    smooth_alpha: Optional[float] = 0.5,
    providers: Optional[Sequence[backend.Provider]] = None,
) -> onnx.ModelProto:
    """
    C++-backed port of :func:`onnxsim.apply_svdquant`: SVDQuant / Nunchaku
    (Li, Lin, Zhang, et al., 2024) -- an optional SmoothQuant-style
    migration, then for every matched MatMul/vanilla-Gemm layer a low-rank/
    residual split of the weight via truncated SVD, keeping the rank-
    ``rank`` dominant/outlier structure in a full-precision low-rank branch
    and block-wise INT4-quantizing only the (now much more uniform)
    residual. See this module's own :func:`apply_svdquant` and
    ``onnxsim/svdquant.py``'s module docstring for the full technique and
    scope.

    Same real calibration machinery as :func:`onnxsim.apply_smoothquant_cpp`
    -- a live :class:`onnxsim.onnx_simplifier.PyModelExecutor`-backed
    :func:`onnxsim.onnx_simplifier._get_model_executor` executor actually
    runs ``calibration_data`` through the model in C++, but only when
    ``smooth_alpha`` is not ``None`` (see ``ApplySvdquant`` in
    ``svdquant_entry.h`` for the full scope, including its own accepted
    numerical scope: this port's own SVD is a hand-rolled Jacobi solver
    rather than ``numpy.linalg.svd``'s own LAPACK routine -- see
    :func:`onnxsim.apply_low_rank_compensation_cpp`'s own identical note).

    :param model: the original (unquantized) onnx ModelProto or file path
    :param calibration_data: representative input batches, forwarded to the
            SmoothQuant-style migration step (ignored entirely when
            ``smooth_alpha`` is ``None``) -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted and ``smooth_alpha`` is not ``None``)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted and ``smooth_alpha`` is not
            ``None``
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied, or if ``smooth_alpha`` is
            ``None``)
    :param rank: the low-rank branch's rank ``r`` (clamped to
            ``min(r, N, K)`` per layer)
    :param block_size: elements per quantization block along the
            residual's reduction dimension
    :param smooth_alpha: migration strength forwarded to the SmoothQuant-
            style preprocessing step's own ``alpha``; pass ``None`` to skip
            migration entirely and decompose the raw weight instead
    :param providers: onnxruntime execution providers to run the smoothing
            step's calibration on (ignored if ``smooth_alpha`` is ``None``)
    :returns: a model with every matched layer's weight replaced by a
            block-wise INT4-quantized residual plus a full-precision
            low-rank correction; layers with a non-constant, non-2-D
            weight, or a reduction dimension not divisible by
            ``block_size``, are left untouched. A model with no matching
            layer, or an opset older than 21, is returned unchanged.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if smooth_alpha is not None and calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    calibration_data_pb = [
        {
            name: onnx.numpy_helper.from_array(np.asarray(arr), name)
            for name, arr in (batch or {}).items()
        }
        for batch in (calibration_data or [])
    ]
    return onnx.load_from_string(
        C.apply_svdquant(
            _get_model_executor(providers),
            model.SerializeToString(),
            calibration_data_pb,
            rank,
            block_size,
            smooth_alpha,
        )
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


#: Ops most ONNX runtimes/NPU targets either refuse outright or support only
#: partially -- what :func:`inline_local_functions` checks for after
#: inlining and simplification. Matches the set the codebase already names
#: informally elsewhere (e.g. :func:`quantize_fp8`'s own docstring above).
_CONTROL_FLOW_OPS = frozenset({"If", "Loop", "Scan"})


def inline_local_functions(
    model: Union[str, onnx.ModelProto],
    *,
    convert_version: bool = True,
    providers: Optional[Sequence[backend.Provider]] = None,
    skipped_optimizers: Optional[List[str]] = None,
    extra_optimizers: Optional[List[str]] = None,
) -> onnx.ModelProto:
    """Inline every model-local function call, then verify no control flow
    (``If``/``Loop``/``Scan``) survived.

    :func:`simplify`'s own ``inline_functions=True`` already does the first
    half of this -- inline, then run the full fixed point (shape inference,
    constant folding, the optimizer's default pass set) so the result is not
    left raw. That fixed point already includes onnx-optimizer's
    ``eliminate_if_with_const_cond`` by default (see
    ``tests/test_formal_verify_eliminate_if_with_const_cond.py`` for why it
    is sound), so an ``If`` a function's body introduces is already
    eliminated automatically *when its condition is resolvable at compile
    time* -- collapsed into just the taken branch, inlined directly into the
    parent graph.

    What ``simplify(inline_functions=True)`` alone does not give you:

    - A function living in its own domain needs that domain listed in the
      model's own ``opset_import`` before the inliner -- or even
      ``onnx.checker`` -- will look at it, which a model assembled by hand
      (e.g. :mod:`onnxsim.qat_graph`'s ``GraphBuilder``) does not always
      have; this fills it in first, so inlining does not silently no-op on a
      hand-assembled graph the way it would otherwise.
    - Confirmation that inlining actually finished the job: if a surviving
      ``If``/``Loop``/``Scan``'s condition or trip count is not a compile-time
      constant (real, data-dependent control flow), no pass can legally
      eliminate it, and most ONNX runtimes -- particularly NPU/WebNN/WebGPU
      execution providers -- do not execute it at all. Shipping that graph
      to such a target fails at *inference* time with an opaque "unsupported
      op" error, far from the inlining call that produced it. This raises
      ``ValueError`` right here instead, naming exactly which node it is.

    Returns a new :class:`onnx.ModelProto`; ``model`` is not modified.
    Returns ``model`` unchanged (as a ``ModelProto`` if given a path) when it
    has no local functions to inline.

    :param model: onnx ModelProto object or file path
    :param convert_version: passed to ``onnx.inliner.inline_local_functions``:
            automatically convert a function's own opset version to the
            model's before inlining it, rather than requiring them to
            already match. Defaults to True, since a function is commonly
            authored once, against a fixed opset, and then inlined into
            models exported over time against a range of newer ones.
    :param providers: forwarded to :func:`simplify` for its own constant
            folding
    :param skipped_optimizers: forwarded to :func:`simplify`
    :param extra_optimizers: forwarded to :func:`simplify`
    :raises ValueError: if an ``If``/``Loop``/``Scan`` node remains after
            inlining and simplification -- see above
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if not model.functions:
        return model

    from onnx import inliner as onnx_inliner

    # Every function's own domain has to be a model opset_import entry
    # before the inliner will look at it at all -- see this function's own
    # docstring. Copy first: onnx.inliner.inline_local_functions itself
    # already returns a fresh ModelProto rather than mutating its argument,
    # and this stays consistent with that instead of mutating the caller's
    # model as a side effect of this fixup.
    model = copy.deepcopy(model)
    imported_domains = {entry.domain for entry in model.opset_import}
    for fn in model.functions:
        if fn.domain not in imported_domains:
            model.opset_import.append(onnx.helper.make_opsetid(fn.domain, 1))
            imported_domains.add(fn.domain)

    inlined = onnx_inliner.inline_local_functions(
        model, convert_version=convert_version
    )
    inlined, _ = simplify(
        inlined,
        skipped_optimizers=skipped_optimizers,
        extra_optimizers=extra_optimizers,
        providers=providers,
    )

    survivors = [
        f"{node.op_type} {node.name!r}"
        for node in inlined.graph.node
        if node.op_type in _CONTROL_FLOW_OPS
    ]
    if survivors:
        raise ValueError(
            "inline_local_functions: control flow survived inlining and "
            "simplification: " + ", ".join(survivors) + ". Most ONNX "
            "runtimes do not execute If/Loop/Scan; this means a function's "
            "condition or trip count is not a compile-time constant, so it "
            "could not be eliminated automatically and needs a real fix "
            "upstream rather than a legalizer pass."
        )
    return inlined


# --------------------------------------------------------------------------
# Default-on equivalence certification (onnxsim.certify) around simplify()
# --------------------------------------------------------------------------

#: Models larger than this (serialized) are not certified by default: the check
#: needs a second copy of the original, and the SMT encoding targets small windows.
_CERTIFY_MAX_BYTES = 64 * 1024 * 1024
#: Wall-clock budget (seconds) for the whole default-on certification.
_CERTIFY_TOTAL_SECONDS = 10.0


def _certify_snapshot(model, output_path, explicit):
    """A private copy of the original model for ``certify``, or ``(None, reason)``."""
    if output_path is not None:
        return None, "output_path streams the result to disk"
    try:
        if isinstance(model, str):
            if os.path.getsize(model) > _CERTIFY_MAX_BYTES:
                return None, "model is larger than the certify size limit"
            snap = onnx.load(model, load_external_data=False)
            if any(
                t.data_location == onnx.TensorProto.EXTERNAL
                for t in snap.graph.initializer
            ):
                return None, "model uses external data"
            return snap, ""
        if model.ByteSize() > _CERTIFY_MAX_BYTES:
            return None, "model is larger than the certify size limit"
        snap = onnx.ModelProto()
        snap.CopyFrom(model)
        return snap, ""
    except Exception as e:  # never let a certify prerequisite break simplify
        return None, f"could not snapshot the original ({e})"


def _certify_result(orig, simplified, kwargs, explicit):
    """Run certify and return ``(status, detail)``; never raises."""
    from . import certify as _certify
    from . import ranges as _ranges

    # Only annotations on graph *inputs* bound anything certify can use. An output-only
    # annotation leaves every input unbounded, so it must not defeat the "unproven, annotate
    # input ranges" downgrade below (it used to: any annotation at all counted).
    graph_inputs = {i.name for i in orig.graph.input} - {
        t.name for t in orig.graph.initializer
    }
    annotated = {
        name: box
        for name, box in _ranges.get_ranges(orig).items()
        if name in graph_inputs
    }
    try:
        report = _certify.certify(
            orig,
            simplified,
            input_ranges=annotated or None,
            atol=kwargs.get("check_atol", 1e-5),
            rtol=kwargs.get("check_rtol", 1e-4),
            max_work=100_000,
            timeout_ms=5_000,
            total_timeout_ms=int(_CERTIFY_TOTAL_SECONDS * 1000),
        )
    except (
        ValueError
    ) as e:  # e.g. inputs/outputs changed on purpose (unused_output, ...)
        return "not-run", str(e)
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"
    if report.ok:
        return "proved", ", ".join(sorted(set(report.outputs.values())))
    refuted = [w for w in report.windows if w.status == _certify.REFUTED]
    if (
        refuted
        and not annotated
        and all(_certify.NEEDS_RANGES_HINT in w.detail for w in refuted)
    ):
        return (
            "unproven-no-ranges",
            "differs only for very large inputs (rounding of re-computed constants?); "
            "annotate input ranges with onnxsim.ranges.set_range to decide",
        )
    if refuted:
        return "refuted", refuted[0].detail
    first = next((w for w in report.windows if w.status == _certify.SKIPPED), None)
    return "skipped", first.detail if first else "not all outputs proved"


def _with_certify(fn):
    """Wrap ``simplify`` so it also certifies its own result (best effort, on by default).

    ``certify=None`` (default): run when z3-solver is installed and the model is small
    enough; stay silent unless the verdict is ``refuted``.  The environment
    variable ``ONNXSIM_CERTIFY=0`` disables this default process-wide. ``certify=True``: run, and
    report every outcome including why it did not run. ``certify=False``: skip. The
    verdict is recorded in the returned model's ``metadata_props`` (``onnxsim.certify``,
    ``onnxsim.certify.detail``); it never changes the simplified model otherwise and a
    failure inside certification never fails ``simplify``.
    """

    @functools.wraps(fn)
    def wrapper(*args, certify=None, **kwargs):
        # ONNXSIM_CERTIFY=0 turns the *default* off process-wide (an explicit
        # certify=True still wins). Test suites that call simplify() incidentally,
        # not to test certification, set it: Z3 proofs are sensitive to the Z3
        # state earlier work leaves in the process, so a default-on run inside
        # unrelated tests made a pure-Z3 test in CI's formal-verification job hang.
        if certify is None and os.environ.get(
            "ONNXSIM_CERTIFY", ""
        ).strip().lower() in (
            "0",
            "false",
            "off",
            "no",
        ):
            certify = False
        model = args[0] if args else kwargs.get("model")
        explicit = certify is True
        snap, why = None, ""
        if certify is not False:
            if importlib.util.find_spec("z3") is None:
                why = "z3-solver is not installed (pip install z3-solver)"
            else:
                snap, why = _certify_snapshot(
                    model, kwargs.get("output_path"), explicit
                )
        result = fn(*args, **kwargs)
        if certify is False:
            return result
        if snap is None:
            if explicit:
                print(f"Certify: not run -- {why}")
            return result
        simplified = result[0]
        status, detail = _certify_result(snap, simplified, kwargs, explicit)
        if status == "error":
            print(
                Text(f"WARNING: certify failed internally: {detail}", style="bold red")
            )
        elif status == "refuted":
            print(
                Text(
                    "WARNING: certify could not prove the simplified model equals the original and "
                    f"found a counterexample: {detail}",
                    style="bold red",
                )
            )
        elif explicit:
            print(f"Certify: {status} ({detail})")
        for key, value in (
            ("onnxsim.certify", status),
            ("onnxsim.certify.detail", detail[:300]),
        ):
            if isinstance(simplified, onnx.ModelProto):
                for p in list(simplified.metadata_props):
                    if p.key == key:
                        simplified.metadata_props.remove(p)
                entry = simplified.metadata_props.add()
                entry.key, entry.value = key, value
        return result

    sig = inspect.signature(fn)
    wrapper.__signature__ = sig.replace(
        parameters=list(sig.parameters.values())
        + [
            inspect.Parameter(
                "certify",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[bool],
            )
        ]
    )
    return wrapper


def _set_metadata(model: onnx.ModelProto, key: str, value: str) -> None:
    for p in list(model.metadata_props):
        if p.key == key:
            model.metadata_props.remove(p)
    entry = model.metadata_props.add()
    entry.key, entry.value = key, value


def _nan_verdict(model: onnx.ModelProto) -> Tuple[str, str]:
    from . import nan_check

    rep = nan_check.check_nan(model)
    status = "finite" if rep.finite else "nan-free" if rep.nan_free else "nan reachable"
    details = [
        f"{h.tensor} ({h.op_type}) {h.kind}: {h.detail}" for h in rep.hazards[:3]
    ]
    if rep.unmodelled:
        details.append(f"unmodelled ops {rep.unmodelled}")
    if rep.unanalysed:
        details.append(f"{len(rep.unanalysed)} tensor(s) without an interval")
    return status, "; ".join(details)


def _gradient_verdict(model: onnx.ModelProto, precision: str) -> Tuple[str, str]:
    from . import grad_health

    rep = grad_health.check_gradient_health(model, precision=precision)
    status = "healthy" if rep.healthy else "unhealthy"
    details = [f"{f.tensor} {f.kind}: {f.detail}" for f in rep.findings[:3]]
    if rep.unanalysed:
        details.append(f"{len(rep.unanalysed)} gradient(s) without a bound")
    return status, "; ".join(details)


def _run_hazard_check(
    model: onnx.ModelProto, key: str, label: str, verdict: Callable[[], Tuple[str, str]]
) -> None:
    try:
        status, detail = verdict()
    except Exception as e:  # a checker failure must never fail simplify
        status, detail = "error", f"{type(e).__name__}: {e}"
        print(Text(f"WARNING: {label} failed internally: {detail}", style="bold red"))
    else:
        style = "bold yellow" if status in ("nan reachable", "unhealthy") else None
        print(
            Text(f"{label}: {status}" + (f" ({detail})" if detail else ""), style=style)
        )
    _set_metadata(model, key, f"{status}: {detail}"[:1000])


def _with_hazard_checks(fn):
    """Wrap ``simplify`` so it can also check its own result for NaN/Inf hazards and for
    gradient health (both opt-in, both off by default).

    ``check_hazards=True`` runs :func:`onnxsim.nan_check.check_nan` on the simplified model;
    ``check_gradients=True`` runs :func:`onnxsim.grad_health.check_gradient_health` on it with its
    graph outputs taken as gradients, at ``gradient_precision``. Input boxes come from
    ``onnxsim.ranges`` annotations. Verdicts are printed and recorded in ``metadata_props``
    (``onnxsim.nan_check``, ``onnxsim.gradient_health``); a failing check is reported, never raised.
    """

    @functools.wraps(fn)
    def wrapper(
        *args,
        check_hazards=False,
        check_gradients=False,
        gradient_precision="fp32",
        **kwargs,
    ):
        if check_gradients and gradient_precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"unknown gradient_precision {gradient_precision!r}")
        result = fn(*args, **kwargs)
        simplified = result[0]
        if not (check_hazards or check_gradients) or not isinstance(
            simplified, onnx.ModelProto
        ):
            return result
        if check_hazards:
            _run_hazard_check(
                simplified,
                "onnxsim.nan_check",
                "NaN check",
                lambda: _nan_verdict(simplified),
            )
        if check_gradients:
            _run_hazard_check(
                simplified,
                "onnxsim.gradient_health",
                "Gradient health",
                lambda: _gradient_verdict(simplified, gradient_precision),
            )
        return result

    sig = inspect.signature(fn)
    wrapper.__signature__ = sig.replace(
        parameters=list(sig.parameters.values())
        + [
            inspect.Parameter(
                name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=ann
            )
            for name, default, ann in (
                ("check_hazards", False, bool),
                ("check_gradients", False, bool),
                ("gradient_precision", "fp32", str),
            )
        ]
    )
    return wrapper


@_with_hazard_checks
@_with_certify
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
    target_opset_version: Optional[Union[int, Literal["latest"]]] = None,
    extra_optimizers: Optional[List[str]] = None,
    custom_rewriter: Optional[ModelRewriter] = None,
    function_rewrite_rules: Optional[Sequence[FunctionRewriteRule]] = None,
    check_rtol: float = 1e-4,
    check_atol: float = 1e-5,
    input_fill: str = "random",
    providers: Optional[Sequence[backend.Provider]] = None,
    gemm_fusion_backend: Literal[
        "ort_cpu", "unrestricted", "webgpu", "webnn"
    ] = "ort_cpu",
    profile: Optional[str] = None,
    ort_profile: Optional[str] = None,
    merge_ort_profile: bool = False,
    output_path: Optional[str] = None,
    external_data_threshold: str = DEFAULT_EXTERNAL_DATA_THRESHOLD,
) -> Tuple[onnx.ModelProto, bool]:
    """
    :param model: onnx ModelProto object or file path
    :param check_n: The simplified model will be checked for `check_n` times by random inputs
    :param certify: Also try to *prove* the result equals the original with Z3 (``onnxsim.certify``).
            ``None`` (default): best effort -- runs when z3-solver is installed and the model is
            small, silent unless it finds a counterexample. ``True``: always report the outcome,
            including why it did not run. ``False``: skip. The verdict is stored in the returned
            model's ``metadata_props`` (``onnxsim.certify``). Input ranges come from
            ``onnxsim.ranges`` annotations on the model; without them a proof can only hold for
            unbounded inputs, which a folded BatchNorm cannot satisfy exactly.
    :param check_hazards: Opt in: report the NaN/Inf hazards of the simplified model
            (:func:`onnxsim.nan_check.check_nan`). Input boxes come from ``onnxsim.ranges``
            annotations; without them most arithmetic is reported. Stored in
            ``metadata_props["onnxsim.nan_check"]``.
    :param check_gradients: Opt in: check the simplified model's graph outputs as gradients for
            underflow and rounding loss (:func:`onnxsim.grad_health.check_gradient_health`).
            Stored in ``metadata_props["onnxsim.gradient_health"]``.
    :param gradient_precision: Format the gradients are computed in for ``check_gradients``:
            ``"fp32"`` (default), ``"fp16"`` or ``"bf16"``.
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
            opset during simplification. Pass the literal string ``"latest"`` to resolve to the
            highest default-domain opset version this onnxsim build's compiled-in onnx schema
            registry supports (note this can differ from the pip-installed ``onnx`` package's own
            ``onnx.defs.onnx_opset_version()``, since onnxsim vendors its own onnx fork). When None
            (the default), the opset version is left unchanged.
    :param extra_optimizers: The counterpart to ``skipped_optimizers``: run these named onnx-optimizer
            passes in addition to the default fuse/elimination set, rather than excluding some of it.
            This is how a pass registered as ``PassType::Other`` -- excluded from the default set
            because it is a graph-shape rewrite rather than a pure node reduction or fusion (e.g. a
            defusion that trades a backend-specific op for a more portable but larger equivalent) --
            gets opted into, without changing what runs by default for every other caller. List valid
            names with ``onnxsim --list-other-optimizers`` (``--list-default-optimizers`` lists the
            ones already on by default, which do not need naming here). Has no effect when
            ``perform_optimization`` is ``False``. An unknown name raises, since -- unlike a typo in
            ``skipped_optimizers``, which just means nothing new is skipped -- a typo here means the
            pass you asked for silently never runs.
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
            ``onnxruntime.InferenceSession`` -- e.g. AMD's Ryzen AI NPU via
            ``[("VitisAIExecutionProvider", {"config_file": "vaip_config.json"}),
            "CPUExecutionProvider"]`` (the Vitis AI EP ships in AMD's Ryzen AI
            Software bundle, not in the stock ``onnxruntime`` wheel; keep
            ``CPUExecutionProvider`` last so unsupported ops fall back to the
            CPU). ``None`` (the default) folds on the
            CPU. Non-CPU providers require onnxruntime to be installed (the CUDA
            provider specifically needs the ``onnxruntime-gpu`` build, the NPU
            provider AMD's Ryzen AI bundle); a
            requested provider that the installed onnxruntime does not offer
            raises ``ValueError`` instead of silently falling back.
    :param gemm_fusion_backend: Which runtime the ``fuse_matmul_add_bias_into_gemm``
            passes should assume will execute the *simplified* model -- unrelated to
            ``providers`` above, which only picks constant folding's execution
            provider during simplification itself. ``"ort_cpu"`` (the default)
            restricts the MatMul+Add -> Gemm fusion to FLOAT32 operands: ONNX
            Runtime's CPU execution provider has no fast Gemm kernel for FLOAT16 (it
            falls back to a naive path) even though its MatMul kernel does, so
            fusing a FLOAT16 MatMul+Add there was measured making that op ~70x
            *slower* to run, not faster. ``"unrestricted"`` fuses regardless of
            operand dtype (onnx-optimizer's original, backend-agnostic behavior) --
            use it when the simplified model will run somewhere ORT CPU's FP16 Gemm
            slowness does not apply (a different runtime, a different execution
            provider such as CUDA, or an ONNX Runtime build with a real FP16 Gemm
            kernel). ``"webgpu"`` is for models headed to onnxruntime-web's WebGPU
            execution provider: unlike ORT's CPU EP, WebGPU has its own FP16 Gemm
            kernel (including subgroup-matrix-accelerated paths on GPUs that
            support them), so there is no CPU-specific naive-FP16 fallback to avoid
            here -- it is handled identically to ``"unrestricted"`` (fuse
            regardless of dtype). Pair it with
            :func:`onnxsim.check_webgpu_attention_support` on the simplified
            output, which flags a separate WebGPU gap this flag does not cover:
            ``com.microsoft::Attention`` nodes using mask/past-present inputs,
            which onnxruntime-web's WebGPU Attention kernel does not yet
            accelerate and will silently run on a fallback execution provider
            instead. ``"webnn"`` is likewise for models headed to
            onnxruntime-web's WebNN execution provider, which delegates to the
            platform's own ML stack (DirectML / Core ML / the platform NN API)
            -- routinely used precisely for NPU/GPU FP16 (and lower-precision)
            acceleration, not a naive fallback path -- so it is handled
            identically to ``"unrestricted"`` too. Pair it with
            :func:`onnxsim.check_webnn_support` on the simplified output,
            which flags WebNN gaps this flag does not cover: INT64 graph
            inputs/outputs and non-constant ``Reshape``/``Expand`` shape
            inputs, both of which fall a node (or, for INT64 boundaries,
            potentially the whole session) back to onnxruntime-web's ``wasm``
            fallback instead of running on WebNN. Implemented in the C++ core
            via the ``ONNXSIM_GEMM_FUSION_BACKEND`` environment variable (as
            ``"webgpu"``/``"webnn"`` are translated to ``"unrestricted"``
            there, since that is the only distinction the C++ side actually
            makes), so it works from every binding; setting this argument
            simply sets that variable for the call.
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
    :param output_path: Save the simplified model to this path directly, instead of (or as
            well as) relying on the caller to save the returned ``ModelProto`` themselves.
            Requires ``model`` to be a file path (there is no file to redirect otherwise).
            When ``model`` is a path and ``check_n == 0`` (the default), this lets the fast
            path skip reading the result back with its tensor data inline (default
            ``load_external_data=True``) purely to satisfy this function's return contract,
            loading it structure-only (``load_external_data=False``) instead -- correct for
            inspecting the graph (shapes, node counts, ...), but call
            ``onnx.load_external_data_for_model(model_opt)`` yourself if you need actual
            tensor values from it afterward. Note this is a secondary optimization: the
            dominant peak-memory cost for a large external-data model was traced to the C++
            core's own working copy of the model, not to this reload -- see
            ``bench/RESULTS_synthetic_decoder_oom.md`` for the measurements and the (now
            separately fixed, via ``SimplifyConsumeInput`` in ``onnxsim.cpp``) root cause.
            Outside the fast-path case (``check_n > 0``, or another reason it doesn't apply)
            the model is still saved to ``output_path`` before returning, just without that
            reload-skipping benefit, since the full model has to be materialized anyway to
            run the correctness check.
    :param external_data_threshold: When saving to ``output_path``, save the model as
            external data (weights in a sibling ``.data`` file, same as passing
            ``save_as_external_data=True`` to ``onnx.save``) whenever its serialized size
            exceeds this threshold -- by default ``"100MB"``, so a model that size or larger
            gets external data with no extra argument needed. A model over 2GB always saves
            as external data regardless of this setting (inline serialization is not possible
            past that size). Accepts the same ``"<number><unit>"`` strings as
            ``tensor_size_threshold`` (e.g. ``"500MB"``, ``"2GB"``); raise it (e.g. to
            ``"2GB"``) to keep the pre-existing behavior of only externalizing when inline
            saving is impossible.
    :return: A tuple (simplified model, success(True) or failed(False))
    """
    # Validate the requested execution providers up front. onnxsim's constant
    # folding catches per-op executor failures and leaves the op unfolded, so an
    # unavailable provider raised mid-fold would be swallowed and silently
    # degrade to no folding. Checking here turns a misconfigured provider (e.g.
    # CUDA requested without the onnxruntime-gpu build) into an immediate error.
    backend.validate_providers(providers)

    # The C++ core's ``ParseGemmFusionBackend`` (gemm_fusion_backend.cpp) only
    # recognizes "ort_cpu"/"unrestricted" -- "webgpu"/"webnn" are Python-facing
    # aliases for "unrestricted" (see this parameter's docstring for why they
    # behave identically), translated here so the env var below always carries
    # a name the C++ side understands.
    _gemm_fusion_backend_for_cpp = (
        "unrestricted"
        if gemm_fusion_backend in ("webgpu", "webnn")
        else gemm_fusion_backend
    )

    # ``target_opset_version="latest"`` resolves against the C++ core's own
    # compiled-in onnx schema registry (the same one ConvertOpsetVersion uses),
    # not the pip-installed `onnx` package's `onnx.defs.onnx_opset_version()` --
    # the two can differ since onnxsim vendors its own onnx fork.
    if target_opset_version == "latest":
        target_opset_version = C.max_default_domain_opset_version()
    elif target_opset_version is not None and not isinstance(target_opset_version, int):
        raise ValueError(
            "target_opset_version must be an int, the string 'latest', or None, "
            f"got {target_opset_version!r}"
        )

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
    if output_path is not None and not isinstance(model, str):
        raise ValueError(
            "output_path requires `model` to be a file path (there is no file to "
            "redirect the saved result from otherwise)"
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
            _fast_prev_gemm_backend_env = os.environ.get("ONNXSIM_GEMM_FUSION_BACKEND")
            os.environ["ONNXSIM_GEMM_FUSION_BACKEND"] = _gemm_fusion_backend_for_cpp
            try:
                if isinstance(model, str):
                    # ``output_path`` given: write the result there directly instead of a
                    # throwaway temporary file, and skip loading it back with data inline.
                    # A structure-only load still satisfies this function's
                    # ``ModelProto``-returning contract for callers who only need the graph
                    # shape, not the tensor values. See ``output_path``'s docstring above for
                    # why this is a secondary optimization rather than the main fix for
                    # bench/TODO_large_decoder_submodule_oom.md -- that turned out to be
                    # inside the C++ core (``SimplifyConsumeInput`` in ``onnxsim.cpp``), not
                    # here; see ``bench/RESULTS_synthetic_decoder_oom.md`` for how that was
                    # found.
                    if output_path is not None:
                        C.simplify_path(
                            _get_model_executor(providers),
                            model,
                            output_path,
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
                            extra_optimizers,
                        )
                        check_ok = model_checking.compare(
                            output_path,
                            None,
                            0,
                            test_input_shapes,
                            input_data,
                            custom_lib,
                            rtol=check_rtol,
                            atol=check_atol,
                            input_fill=input_fill,
                        )
                        model_opt = onnx.load(output_path, load_external_data=False)
                        return model_opt, check_ok
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
                            extra_optimizers,
                        )
                        # check_n == 0 is guaranteed on this path (the fast-path
                        # gate above requires it), so compare() only needs
                        # model_opt for the checker call below -- check straight
                        # from disk (the checker's own C++ file loader) rather
                        # than loading it into a ModelProto first just to have
                        # the checker re-serialize it right back to bytes.
                        check_ok = model_checking.compare(
                            fast_out_path,
                            None,
                            0,
                            test_input_shapes,
                            input_data,
                            custom_lib,
                            rtol=check_rtol,
                            atol=check_atol,
                            input_fill=input_fill,
                        )
                        model_opt = onnx.load(fast_out_path)
                    return model_opt, check_ok
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
                        extra_optimizers,
                    )
                    if len(model_opt_bytes) == 0:
                        raise ValueError("Simplified model larger than 2GB")
                    # Same idea as the path branch above: check_n == 0 here too,
                    # so hand the checker the bytes we already have instead of
                    # loading them into a ModelProto (below) and making the
                    # checker serialize that right back to bytes.
                    check_ok = model_checking.compare(
                        model_opt_bytes,
                        None,
                        0,
                        test_input_shapes,
                        input_data,
                        custom_lib,
                        rtol=check_rtol,
                        atol=check_atol,
                        input_fill=input_fill,
                    )
                    model_opt = onnx.load_from_string(model_opt_bytes)
                return model_opt, check_ok
            except Exception:
                pass
            finally:
                if _fast_profile_active:
                    if _fast_prev_profile_env is None:
                        os.environ.pop("ONNXSIM_PROFILE", None)
                    else:
                        os.environ["ONNXSIM_PROFILE"] = _fast_prev_profile_env
                if _fast_prev_gemm_backend_env is None:
                    os.environ.pop("ONNXSIM_GEMM_FUSION_BACKEND", None)
                else:
                    os.environ["ONNXSIM_GEMM_FUSION_BACKEND"] = (
                        _fast_prev_gemm_backend_env
                    )

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

    # Select which runtime fuse_matmul_add_bias_into_gemm(_batched) should
    # assume will execute the simplified model, for the duration of this call
    # (read inside ``Simplify`` via ``ONNXSIM_GEMM_FUSION_BACKEND`` -- see
    # gemm_fusion_backend.h), restoring any prior value afterwards so this
    # does not leak into later calls in the same process.
    _prev_gemm_backend_env = os.environ.get("ONNXSIM_GEMM_FUSION_BACKEND")
    os.environ["ONNXSIM_GEMM_FUSION_BACKEND"] = _gemm_fusion_backend_for_cpp

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
            extra_optimizers,
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
            # At check_n == 0, compare() only feeds this to the checker (see the
            # comment above), which accepts bytes directly -- pass the bytes we
            # already have instead of the ModelProto we just deserialized them
            # into, so the checker isn't made to re-serialize it right back.
            model_opt_bytes if check_n == 0 else model_opt,
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
                extra_optimizers,
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
        if _prev_gemm_backend_env is None:
            os.environ.pop("ONNXSIM_GEMM_FUSION_BACKEND", None)
        else:
            os.environ["ONNXSIM_GEMM_FUSION_BACKEND"] = _prev_gemm_backend_env
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
    if output_path is not None:
        # Reached with ``output_path`` set only when the fast path above either
        # doesn't apply (e.g. ``check_n > 0``, which needs ``model_opt``
        # materialized anyway to run the correctness check) or fell back from an
        # internal error. Either way ``model_opt`` is already a full ModelProto
        # here, so there is no reload to avoid -- just honor the save request.
        _external_data_threshold_bytes = parse_size(external_data_threshold)
        try:
            if model_opt.ByteSize() > _external_data_threshold_bytes:
                raise ValueError("external_data_threshold")
            onnx.save(model_opt, output_path)
        except (ValueError, EncodeError):
            external_data_path = os.path.basename(output_path) + ".data"
            if os.path.exists(external_data_path):
                os.remove(external_data_path)
            onnx.save(
                model_opt,
                output_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=external_data_path,
            )
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


# Set by ``onnxsim.rpc.remote_executor`` so constant folding can run on a remote device.
_executor_override: contextvars.ContextVar[Optional[C.ModelExecutor]] = (
    contextvars.ContextVar("onnxsim_executor_override", default=None)
)


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
    override = _executor_override.get()
    if override is not None:
        return override
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
        "--enable-optimization",
        help="The counterpart to --skip-optimization: run these named optimizer passes in addition to the default set, rather than excluding some of it. Use for a pass not already on by default (see --list-other-optimizers for valid names), for example `onnxsim a.onnx b.onnx --enable-optimization fuse_matmul_add_bias_into_gemm_batched`.",
        type=str,
        nargs="+",
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
        "--certify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also try to prove the simplified model equals the original with Z3 (onnxsim.certify; needs "
        "z3-solver). The default is best effort: it runs when z3-solver is installed and the model is "
        "small, and only speaks up on a counterexample. --certify always reports the outcome; "
        "--no-certify skips it. Input ranges come from onnxsim.ranges annotations on the model "
        "(python -m onnxsim.ranges).",
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
        "--external-data-threshold",
        help="Save parameters as external data whenever the simplified model's serialized size exceeds this, without needing --save-as-external-data. Accepts the same '<number><unit>' syntax as --no-large-tensor, e.g. '500MB' or '2GB'. Defaults to '100MB'; raise it (e.g. to '2GB') to only externalize when inline saving is impossible, matching pre-existing behavior.",
        type=str,
        default=DEFAULT_EXTERNAL_DATA_THRESHOLD,
    )
    parser.add_argument(
        "--skip-schema-import",
        help="By default onnxsim imports operator schemas registered in the Python 'onnx' module (e.g. via onnx.defs.register_schema) into its own registry so models with custom operators pass validation. Specify this flag to disable that import.",
        action="store_true",
    )
    parser.add_argument(
        "--target-opset",
        help="Convert the model to this opset version (of the default ONNX domain) before simplifying, for example '--target-opset 18'. Can be used to upgrade (or downgrade) the model's opset during simplification. Pass 'latest' to resolve to the highest opset version this onnxsim build supports.",
        type=lambda s: s if s == "latest" else int(s),
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
        "The CUDA provider requires the 'onnxruntime-gpu' build; AMD's NPU "
        "provider ('VitisAIExecutionProvider', from AMD's Ryzen AI Software "
        "bundle) is accepted the same way, keeping CPUExecutionProvider last "
        "for fallback. Provider *options* (e.g. the Vitis AI EP's "
        "'config_file') need the Python API's (name, options) tuple form.",
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
        "--emit-mlir",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Also emit the simplified model as MLIR (see --mlir-target). "
        "Optionally give an output path; when the flag is passed without one, the "
        "MLIR is written next to the output model with a '.mlir' extension.",
    )
    parser.add_argument(
        "--mlir-target",
        choices=["torch", "onnx"],
        default="torch",
        help="Which MLIR dialect --emit-mlir produces: 'torch' (Torch dialect, via "
        "torch-mlir; pip install torch-mlir) or 'onnx' (ONNX dialect, via the "
        "onnx-mlir compiler binary). Default: torch.",
    )
    parser.add_argument(
        "--onnx-mlir",
        metavar="PATH",
        default=None,
        help="Path to the onnx-mlir compiler binary, used with "
        "'--mlir-target onnx'. Defaults to $ONNX_MLIR, "
        "$ONNX_MLIR_HOME/bin/onnx-mlir, then the onnx-mlir on PATH.",
    )
    parser.add_argument(
        "--emit-coreml",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Also convert the simplified model to Core ML (via coremltools; "
        "pip install coremltools). Requires fully static input shapes. Optionally "
        "give an output path (a '.mlpackage' directory, or '.mlmodel' with "
        "'--coreml-format neuralnetwork'); when the flag is passed without one, it "
        "is written next to the output model with a '.mlpackage'/'.mlmodel' "
        "extension.",
    )
    parser.add_argument(
        "--coreml-format",
        choices=["mlprogram", "neuralnetwork"],
        default="mlprogram",
        help="Core ML model type --emit-coreml produces: 'mlprogram' (the modern "
        "'.mlpackage' format) or 'neuralnetwork' (the legacy '.mlmodel' format). "
        "Default: mlprogram.",
    )
    parser.add_argument(
        "--coreml-compute-units",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
        default="ALL",
        help="Which compute devices the --emit-coreml model may run on. Default: ALL.",
    )
    parser.add_argument(
        "--coreml-io-dtype",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Dtype of the --emit-coreml model's float inputs and outputs. 'fp16' "
        "matches the precision an ML Program already computes in, so Core ML stops "
        "converting every float tensor at the model boundary on each call (needs "
        "--coreml-format mlprogram and iOS16/macOS13 or newer). Default: fp32.",
    )
    parser.add_argument(
        "--coreml-minimum-deployment-target",
        default=None,
        metavar="TARGET",
        help="Minimum OS version the --emit-coreml model must run on, e.g. 'iOS16' "
        "or 'macOS13'. Defaults to coremltools' own default.",
    )
    parser.add_argument(
        "--emit-tflite",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Also convert the simplified model to TensorFlow Lite (via "
        "TensorFlow; pip install tensorflow). Requires fully static input shapes. "
        "Optionally give an output path; when the flag is passed without one, it "
        "is written next to the output model with a '.tflite' extension.",
    )
    parser.add_argument(
        "--tflite-optimize",
        action="store_true",
        help="Enable TFLite's default post-training (dynamic-range) quantization "
        "when converting with --emit-tflite. Only applies to --tflite-backend "
        "builtin.",
    )
    parser.add_argument(
        "--tflite-backend",
        choices=["builtin", "onnx2tf"],
        default="builtin",
        help="Which ONNX-to-TensorFlow translator --emit-tflite uses: 'builtin' "
        "(default, covers a practical op subset, keeps the model's NCHW I/O "
        "layout) or 'onnx2tf' (via https://github.com/PINTO0309/onnx2tf; pip "
        "install onnx2tf -- far broader op coverage, but a much heavier "
        "dependency and a default channel-last I/O layout). Reach for onnx2tf "
        "when a model hits an unsupported op with the builtin translator.",
    )
    parser.add_argument(
        "--tflite-int8",
        action="store_true",
        help="Full-integer quantize when converting with --emit-tflite "
        "(TFLITE_BUILTINS_INT8, with quantized I/O). This is the quantization "
        "the Coral Edge TPU requires; without it, use --tflite-optimize for "
        "plain dynamic-range quantization instead (the two are mutually "
        "exclusive). Only applies to --tflite-backend builtin. Calibration "
        "uses uniform-random data unless the model needs real inputs -- see "
        "--tflite-calibration-samples.",
    )
    parser.add_argument(
        "--tflite-calibration-samples",
        default=None,
        type=int,
        metavar="N",
        help="Random calibration batches for --tflite-int8/--tflite-edgetpu "
        "(default: 100). Random data is enough to produce a valid quantized "
        "model, but real representative inputs give better accuracy -- for "
        "that, use the Python API's representative_dataset= instead.",
    )
    parser.add_argument(
        "--tflite-io-dtype",
        choices=["uint8", "int8"],
        default=None,
        help="Quantized model I/O type for --tflite-int8/--tflite-edgetpu "
        "(default: uint8). Quantized I/O avoids a CPU-side quantize/dequantize "
        "pair at each model boundary on the Edge TPU.",
    )
    parser.add_argument(
        "--tflite-flex",
        action="store_true",
        help="Allow TensorFlow Flex (SELECT_TF_OPS) kernels when converting "
        "with --emit-tflite for the few ops TFLite has no builtin for (e.g. "
        "Atan) instead of failing conversion. Flex ops always run on the CPU, "
        "so a model that needs Flex cannot target the Edge TPU (mutually "
        "exclusive with --tflite-int8/--tflite-edgetpu). Only applies to "
        "--tflite-backend builtin.",
    )
    parser.add_argument(
        "--tflite-layout",
        choices=["nchw", "nhwc"],
        default="nchw",
        help="Dimension order of the --emit-tflite model's 4-D tensors "
        "(default: nchw, ONNX's own order, with transposes around conv/pool). "
        "'nhwc' carries 4-D tensors channel-last end to end and emits no "
        "transposes -- required for Edge TPU compilation of larger models, "
        "whose NCHW entry transpose the compiler refuses ('large activation "
        "tensors'). Only applies to --tflite-backend builtin.",
    )
    parser.add_argument(
        "--tflite-edgetpu",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Compile the --emit-tflite model for the Coral Edge TPU with "
        "edgetpu_compiler (implies --tflite-int8 with quantized I/O). "
        "Optionally give an output path; when the flag is passed without one, "
        "it is written next to the .tflite file with a '_edgetpu' suffix. "
        "Only applies to --tflite-backend builtin.",
    )
    parser.add_argument(
        "--tflite-edgetpu-compiler",
        default=None,
        metavar="PATH",
        help="Explicit path to the edgetpu_compiler binary for --tflite-edgetpu "
        "(default: $EDGETPU_COMPILER or PATH lookup).",
    )
    parser.add_argument(
        "--tflite-edgetpu-check",
        action="store_true",
        help="Before converting with --emit-tflite, statically check the "
        "simplified model against the Edge TPU requirements (static shapes, "
        "supported ops) and print the findings. Needs nothing but onnx.",
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
        "--dynamic-quantize-matmul-integer-to-float",
        help="Same as --dynamic-quantize, but the dequantize step is a "
        "single ONNX Runtime 'com.microsoft' contrib op "
        "(MatMulIntegerToFloat) instead of a MatMulInteger+Cast+Mul(+Add) "
        "node chain. Unlike --dynamic-quantize, the result needs a "
        "com.microsoft-aware runtime (e.g. ONNX Runtime) to execute. See "
        "onnxsim.quantize_dynamic_matmul_integer_to_float.",
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

    class ListOtherOptimizers(argparse.Action):
        def __call__(self, parser, ns, v, option_string=None):
            for p in C._list_other_optimizers():
                print(p)
            parser.exit()

    parser.add_argument(
        "--list-other-optimizers",
        help="List optimizer pass names valid for --enable-optimization (i.e. registered but not already part of the default set listed by --list-default-optimizers)",
        nargs=0,
        action=ListOtherOptimizers,
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
        extra_optimizers=args.enable_optimization,
        check_rtol=args.check_rtol,
        certify=args.certify,
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

    if args.dynamic_quantize_matmul_integer_to_float:
        print(
            "Dynamically quantizing MatMul/Gemm weights to INT8 "
            "(com.microsoft MatMulIntegerToFloat)..."
        )
        model_opt = quantize_dynamic_matmul_integer_to_float(model_opt)

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

    _external_data_threshold_bytes = parse_size(args.external_data_threshold)
    try:
        if not args.save_as_external_data and (
            model_opt.ByteSize() <= _external_data_threshold_bytes
        ):
            onnx.save(model_opt, args.output_model)
        else:
            raise ValueError("save_as_external_data")
    except (ValueError, EncodeError):
        # large models (>2GB) which onnx.save doesn't support, explicitly
        # specified --save-as-external-data, or the model's serialized size
        # exceeds --external-data-threshold (100MB by default)
        external_data_path = os.path.basename(args.output_model) + ".data"
        if os.path.exists(external_data_path):
            os.remove(external_data_path)
        # Mutate ``model_opt`` in place (no deepcopy): ``save_as_external_data=True``
        # moves each initializer's raw_data out to the external file, so the same
        # object is left with external references instead of inline bytes. That
        # matters below: model_info re-serializes ``model_opt`` to report size/diff,
        # which would otherwise hit this same >2GB EncodeError on the very model we
        # just went out of our way to save. main() doesn't need the inline-data
        # version of model_opt again after this point.
        onnx.save(
            model_opt,
            args.output_model,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path,
        )

    if args.emit_mlir is not None:
        from onnxsim import mlir_export

        if args.emit_mlir:
            mlir_path = args.emit_mlir
        else:
            mlir_path = os.path.splitext(args.output_model)[0] + ".mlir"
        dialect = "ONNX" if args.mlir_target == "onnx" else "Torch"
        print(f"Emitting {dialect}-dialect MLIR to {mlir_path} ...")
        mlir_kwargs = {}
        if args.mlir_target == "onnx" and args.onnx_mlir:
            mlir_kwargs["onnx_mlir"] = args.onnx_mlir
        try:
            mlir_export.export_mlir(
                model_opt, mlir_path, target=args.mlir_target, **mlir_kwargs
            )
        except RuntimeError as e:
            print(Text(str(e), style="bold red"))
            sys.exit(1)
        print(f"MLIR written to {mlir_path}")

    if args.emit_coreml is not None:
        from onnxsim import coreml_export

        if args.emit_coreml:
            coreml_path = args.emit_coreml
        else:
            ext = ".mlpackage" if args.coreml_format == "mlprogram" else ".mlmodel"
            coreml_path = os.path.splitext(args.output_model)[0] + ext
        print(f"Converting to Core ML ({args.coreml_format}) at {coreml_path} ...")
        try:
            coreml_export.export_coreml(
                model_opt,
                coreml_path,
                convert_to=args.coreml_format,
                compute_units=args.coreml_compute_units,
                minimum_deployment_target=args.coreml_minimum_deployment_target,
                io_dtype=args.coreml_io_dtype,
            )
        except RuntimeError as e:
            print(Text(str(e), style="bold red"))
            sys.exit(1)
        print(f"Core ML model written to {coreml_path}")

    if args.emit_tflite is not None:
        from onnxsim import tflite_export

        if args.tflite_optimize and args.tflite_backend != "builtin":
            print(
                Text(
                    "--tflite-optimize only applies to --tflite-backend builtin.",
                    style="bold red",
                )
            )
            sys.exit(1)
        edgetpu_active = args.tflite_edgetpu is not None
        int8_active = bool(args.tflite_int8) or edgetpu_active
        if args.tflite_optimize and int8_active:
            print(
                Text(
                    "--tflite-optimize and --tflite-int8/--tflite-edgetpu are "
                    "mutually exclusive.",
                    style="bold red",
                )
            )
            sys.exit(1)
        if int8_active and args.tflite_backend != "builtin":
            print(
                Text(
                    "--tflite-int8/--tflite-edgetpu only apply to "
                    "--tflite-backend builtin (use onnx2tf's own quantization "
                    "options via the Python API instead).",
                    style="bold red",
                )
            )
            sys.exit(1)
        if args.tflite_flex and args.tflite_backend != "builtin":
            print(
                Text(
                    "--tflite-flex only applies to --tflite-backend builtin.",
                    style="bold red",
                )
            )
            sys.exit(1)
        if args.tflite_flex and int8_active:
            print(
                Text(
                    "--tflite-flex and --tflite-int8/--tflite-edgetpu are "
                    "mutually exclusive.",
                    style="bold red",
                )
            )
            sys.exit(1)
        if args.tflite_layout != "nchw" and args.tflite_backend != "builtin":
            print(
                Text(
                    "--tflite-layout only applies to --tflite-backend builtin "
                    "(onnx2tf is channel-last by default).",
                    style="bold red",
                )
            )
            sys.exit(1)
        if args.tflite_edgetpu_check:
            from onnxsim import edgetpu_export

            report = edgetpu_export.check_onnx_for_edgetpu(model_opt)
            print(report.summary())
            if report.errors:
                print(
                    Text(
                        "Edge TPU compatibility errors above must be fixed "
                        "before the model can run on the device.",
                        style="bold red",
                    )
                )
        if args.emit_tflite:
            tflite_path = args.emit_tflite
        else:
            tflite_path = os.path.splitext(args.output_model)[0] + ".tflite"
        print(
            f"Converting to TensorFlow Lite (backend: {args.tflite_backend}) at "
            f"{tflite_path} ..."
        )
        tflite_kwargs = {"backend": args.tflite_backend}
        if args.tflite_backend == "builtin":
            # onnx2tf rejects onnxsim's builtin-only options (and is
            # channel-last by default), so io_layout stays on the builtin path.
            tflite_kwargs["io_layout"] = args.tflite_layout
        if args.tflite_optimize:
            tflite_kwargs["optimizations"] = ["DEFAULT"]
        if args.tflite_flex:
            tflite_kwargs["flex_ops"] = True
        if int8_active:
            tflite_kwargs["int8_quantize"] = True
            if args.tflite_calibration_samples is not None:
                tflite_kwargs["num_calibration_samples"] = (
                    args.tflite_calibration_samples
                )
            tflite_kwargs["inference_io_dtype"] = args.tflite_io_dtype or "uint8"
        elif (
            args.tflite_calibration_samples is not None
            or args.tflite_io_dtype is not None
        ):
            print(
                Text(
                    "--tflite-calibration-samples/--tflite-io-dtype only apply "
                    "with --tflite-int8/--tflite-edgetpu.",
                    style="bold red",
                )
            )
            sys.exit(1)
        try:
            tflite_bytes = tflite_export.export_tflite(
                model_opt, tflite_path, **tflite_kwargs
            )
        except (RuntimeError, ValueError) as e:
            print(Text(str(e), style="bold red"))
            sys.exit(1)
        print(f"TFLite model written to {tflite_path}")
        if edgetpu_active:
            from onnxsim import edgetpu_export

            if args.tflite_edgetpu:
                edgetpu_path = args.tflite_edgetpu
            else:
                stem, _ = os.path.splitext(tflite_path)
                edgetpu_path = stem + "_edgetpu.tflite"
            print(f"Compiling for the Edge TPU at {edgetpu_path} ...")
            try:
                result = edgetpu_export.compile_for_edgetpu(
                    tflite_bytes,
                    output_path=edgetpu_path,
                    compiler=args.tflite_edgetpu_compiler,
                )
            except (RuntimeError, ValueError) as e:
                print(Text(str(e), style="bold red"))
                sys.exit(1)
            print(result.summary())
            if not result.success:
                print(Text("Edge TPU compilation failed.", style="bold red"))
                sys.exit(1)
            if not result.fully_mapped:
                print(
                    Text(
                        "Warning: part of the model falls back to the CPU; "
                        "see the operator statuses above.",
                        style="bold yellow",
                    )
                )
            else:
                print(f"Edge TPU model written to {edgetpu_path}")

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
