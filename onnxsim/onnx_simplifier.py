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
from .calibration import Tensors, generate_random_calibration_data
from .pruning import EmbeddingPruningResult, _ImportanceNorm

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
    the weight's own values. Unlike the pure-Python implementation, ``Conv``
    layers are not (yet) handled -- only ``MatMul``/``Gemm``.

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

    None of the pure-Python per-format functions above have a
    ``global_sparsity`` parameter of their own, so ``global_sparsity`` below
    has no defined meaning for any quantized-weight family here either --
    every one of them is always applied with its own local per-chain
    sparsity, unaffected by that flag. ``com.microsoft::ConvInteger``-based
    dynamic quantization
    (:func:`onnxsim.apply_structured_pruning_dynamic_quantize_conv`) has no
    C++ port at all yet and is NOT matched here.

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
    groups -- from every matched ``com.microsoft::Attention``,
    ``com.microsoft::GroupQueryAttention``, or plain ``ai.onnx::Attention``
    node whose output feeds, optionally through a single shape-preserving
    ``Reshape``, exactly one downstream MatMul/vanilla-Gemm's reduction
    dimension (the output projection) -- the attention analogue of
    :func:`onnxsim.apply_structured_pruning_cpp`, at head (or KV-group)
    instead of single-channel granularity.

    For each matched plain ``com.microsoft::Attention`` block (a single
    merged QKV weight/bias): ranks every head by the combined Frobenius norm
    of its own Q, K, and V weight columns, drops the lowest-``sparsity``-
    fraction of heads (at least one head is always kept), and removes the
    corresponding column blocks from the merged QKV weight (and bias, if
    present), decrementing ``num_heads``/``qkv_hidden_sizes`` accordingly,
    and the matching row block from the output projection's weight.

    For each matched ``GroupQueryAttention`` or plain ``ai.onnx::Attention``
    block (separate, un-merged Q/K/V producers): ranks every *KV group* (a
    KV head and the ``num_heads / kv_num_heads`` query heads the kernel maps
    to it) by the combined Frobenius norm of that group's own Q+K+V weight
    block, drops the lowest-``sparsity``-fraction of groups (at least one
    group is always kept), and removes the corresponding column blocks from
    all three producers (and their biases, if present) together with the
    matching row block from the output projection's weight, decrementing the
    query head count and ``kv_num_heads`` by the number of groups dropped --
    so their ratio (query heads per KV head) is unchanged. An individual
    query head is never dropped on its own: only a whole group, since
    neither kernel has a way to keep a KV head alive for some, but not all,
    of the query heads that shared it.

    The calibration-driven Wanda upgrade of this same matching/ranking
    (mirroring the pure-Python :func:`onnxsim.apply_attention_head_wanda_pruning`)
    is :func:`onnxsim.apply_attention_head_wanda_pruning_cpp`.

    This is a single, self-contained graph rewrite: unlike :func:`simplify`,
    it does not run shape inference, constant folding, or any other pass.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched block's heads (or, for
            GroupQueryAttention/plain ai.onnx Attention, KV groups) to
            remove (at least one is always kept); must be in ``[0, 1)``
    :param importance_norm: ``"l2"`` (default, unchanged from before this
            parameter existed) ranks by the combined Frobenius (L2) norm of
            each head's/KV group's own weight block, mirroring the
            pure-Python :func:`onnxsim.apply_attention_head_pruning`'s own
            default; ``"l1"`` ranks by the sum of absolute weight magnitude
            across that same block instead.
    :returns: ``model`` with every matched block's tensors resized in
            place; anything not matching that exact topology (a
            non-constant weight, a packed-QKV GroupQueryAttention node, a
            GroupQueryAttention/plain ai.onnx Attention node with a
            non-empty constant past-KV-cache or attention-mask input, an
            ai.onnx Attention node with differing Q/K/V head sizes or
            without explicit ``q_num_heads``/``kv_num_heads`` attributes, a
            consumer whose reduction dimension doesn't line up, ...) is
            left completely untouched
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

    Same real head (or, for ``GroupQueryAttention``/plain ``ai.onnx::Attention``,
    whole-KV-group) removal and topology matching as
    :func:`onnxsim.apply_attention_head_pruning_cpp` (a matched
    ``com.microsoft::Attention``, ``com.microsoft::GroupQueryAttention``, or
    plain ``ai.onnx::Attention`` block whose output feeds, optionally through
    a single shape-preserving ``Reshape``, exactly one downstream
    MatMul/vanilla-Gemm's reduction dimension -- see that function's own
    docstring for the full topology and per-block-kind removal details), but
    each unit's importance is ``||W||_F * ||X||_2`` -- the plain
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
    target_opset_version: Optional[Union[int, Literal["latest"]]] = None,
    extra_optimizers: Optional[List[str]] = None,
    custom_rewriter: Optional[ModelRewriter] = None,
    function_rewrite_rules: Optional[Sequence[FunctionRewriteRule]] = None,
    check_rtol: float = 1e-4,
    check_atol: float = 1e-5,
    input_fill: str = "random",
    providers: Optional[Sequence[backend.Provider]] = None,
    gemm_fusion_backend: Literal["ort_cpu", "unrestricted"] = "ort_cpu",
    profile: Optional[str] = None,
    ort_profile: Optional[str] = None,
    merge_ort_profile: bool = False,
    output_path: Optional[str] = None,
    external_data_threshold: str = DEFAULT_EXTERNAL_DATA_THRESHOLD,
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
            ``onnxruntime.InferenceSession``. ``None`` (the default) folds on the
            CPU. Non-CPU providers require onnxruntime to be installed (the CUDA
            provider specifically needs the ``onnxruntime-gpu`` build); a
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
            kernel). Implemented in the C++ core via the
            ``ONNXSIM_GEMM_FUSION_BACKEND`` environment variable, so it works from
            every binding; setting this argument simply sets that variable for the
            call.
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
            os.environ["ONNXSIM_GEMM_FUSION_BACKEND"] = gemm_fusion_backend
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
    os.environ["ONNXSIM_GEMM_FUSION_BACKEND"] = gemm_fusion_backend

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
        if args.emit_tflite:
            tflite_path = args.emit_tflite
        else:
            tflite_path = os.path.splitext(args.output_model)[0] + ".tflite"
        print(
            f"Converting to TensorFlow Lite (backend: {args.tflite_backend}) at "
            f"{tflite_path} ..."
        )
        tflite_kwargs = {"backend": args.tflite_backend}
        if args.tflite_optimize:
            tflite_kwargs["optimizations"] = ["DEFAULT"]
        try:
            tflite_export.export_tflite(model_opt, tflite_path, **tflite_kwargs)
        except RuntimeError as e:
            print(Text(str(e), style="bold red"))
            sys.exit(1)
        print(f"TFLite model written to {tflite_path}")

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
