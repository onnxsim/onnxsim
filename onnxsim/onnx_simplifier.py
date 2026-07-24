import argparse

import copy
import os
import sys
import re
import tempfile
from typing import List, Literal, Dict, Union, Optional, Tuple, Sequence
from rich.text import Text
from rich import print
import numpy as np
from google.protobuf.message import EncodeError

import onnx  # type: ignore
import onnx.checker  # type: ignore
import onnx.helper  # type: ignore
import onnx.shape_inference  # type: ignore
import onnx.numpy_helper  # type: ignore
import onnxsim.onnxsim_cpp2py_export as C
from . import backend
from . import model_info
from . import model_checking
from . import version


TensorShape = List[int]
TensorShapes = Dict[str, TensorShape]
TensorShapesWithOptionalKey = Dict[Optional[str], TensorShape]
Unit = Literal["B", "KB", "MB", "GB", "TB"]

UNIT_MAP: dict[Unit, int] = {
    "B": 1,
    "KB": 1 << 10,
    "MB": 1 << 20,
    "GB": 1 << 30,
    "TB": 1 << 40,
}

def get_output_names(model: onnx.ModelProto) -> List[str]:
    output_names = [opt.name for opt in model.graph.output]
    return output_names


def get_dynamic_input_dims(model: onnx.ModelProto) -> Dict[str, List[int]]:
    """Return, for each real graph input, the axis indices that are dynamic.

    A dimension is *dynamic* when its size is not a fixed positive integer --
    i.e. it carries a symbolic ``dim_param`` (such as "batch" or "width") or a
    non-positive ``dim_value`` (0 or -1, which some exporters use to mean
    "unknown"). Initializers that also appear in ``graph.input`` are skipped:
    they are constants, not inputs. Only inputs with at least one dynamic axis
    appear in the returned mapping.
    """
    initializer_names = {x.name for x in model.graph.initializer}
    dynamic: Dict[str, List[int]] = {}
    for ipt in model.graph.input:
        if ipt.name in initializer_names:
            continue
        tensor_type = ipt.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        axes = [
            i
            for i, dim in enumerate(tensor_type.shape.dim)
            if not (dim.HasField("dim_value") and dim.dim_value > 0)
        ]
        if axes:
            dynamic[ipt.name] = axes
    return dynamic


def _format_input_shape_hint(
    model: onnx.ModelProto, name: str, dynamic_axes: Sequence[int]
) -> str:
    """Render an input's shape as an ``--overwrite-input-shape`` suggestion.

    Static axes keep their concrete size; dynamic axes show their symbolic name
    (or ``?`` when anonymous) as a placeholder the user must replace with a
    concrete size, e.g. ``x:batch,3,H,W``.
    """
    for ipt in model.graph.input:
        if ipt.name == name:
            dims = []
            for i, dim in enumerate(ipt.type.tensor_type.shape.dim):
                if i in dynamic_axes:
                    dims.append(dim.dim_param or "?")
                else:
                    dims.append(str(dim.dim_value))
            return "{}:{}".format(name, ",".join(dims))
    return name


def _warn_if_shape_ops_remain(model_opt: onnx.ModelProto) -> None:
    """Explain leftover ``Shape`` ops caused by dynamic input dimensions.

    onnxsim removes shape-related ops (``Shape``, ``Reshape``, ...) by
    constant-folding them, which is only possible when the dimensions they read
    are statically known. When an input keeps a dynamic dimension, the ``Shape``
    ops that read it genuinely depend on runtime values and cannot be folded, so
    they remain in the output -- which then breaks converters that do not support
    dynamic shapes (e.g. ncnn's ``onnx2ncnn``, see GitHub issue #314). Point the
    user at ``--overwrite-input-shape`` so they can make the model fully static.
    """
    remaining_shape_ops = sum(
        1 for node in model_opt.graph.node if node.op_type == "Shape"
    )
    dynamic_inputs = get_dynamic_input_dims(model_opt)
    if not remaining_shape_ops or not dynamic_inputs:
        return
    hints = " ".join(
        '"{}"'.format(_format_input_shape_hint(model_opt, name, axes))
        for name, axes in dynamic_inputs.items()
    )
    print(
        Text(
            f'The simplified model still contains {remaining_shape_ops} "Shape" '
            "op(s). They read input dimensions that are dynamic "
            f"({', '.join(dynamic_inputs)}), so onnxsim cannot constant-fold "
            "them away. If you need a fully static model (for example for "
            "converters like onnx2ncnn that do not support dynamic shapes), "
            "re-run with a fixed input shape, e.g. --overwrite-input-shape "
            f"{hints} (replace each symbolic/? dimension with a concrete size).",
            style="bold magenta",
        )
    )


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


def remove_initializer_from_input(model: onnx.ModelProto) -> onnx.ModelProto:
    initializer_names = [x.name for x in model.graph.initializer]
    for graph_input in copy.deepcopy(model.graph.input):
        if graph_input.name in initializer_names:
            model.graph.input.remove(graph_input)
    return model


def check_and_update_input_shapes(model: onnx.ModelProto, input_shapes: Optional[TensorShapesWithOptionalKey]) -> Optional[TensorShapes]:
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
                'The model has more than 1 inputs, please use the format "input_name:dim0,dim1,...,dimN" in --input-shape')
    for x in input_shapes:
        if x not in input_names:
            raise RuntimeError(
                'The model doesn\'t have input named "{}"'.format(x))

    return input_shapes  # type: ignore


# A very very large threshold
DEFAULT_TENSOR_SIZE_THRESHOLDHOLD = '1.5GB'


# ONNX ``TensorProto`` element types that onnxoptimizer's tensor-value hashing
# (``cse_util.h``) knows how to hash. Any other type makes those passes raise
# ``RuntimeError: no supported data type: <N>``. We enumerate the *supported*
# types (rather than the unsupported ones) so that element types added to ONNX
# in the future are treated as unhashable by default instead of silently
# crashing the optimizer.
_CSE_HASHABLE_ELEM_TYPES = frozenset({
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
})

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
            (attr.name, attr.description, int(attr.type), bool(attr.required), default_value)
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


def _snapshot_doc_strings(model: onnx.ModelProto) -> dict:
    """Capture the ``doc_string`` fields that the C++ optimizer discards."""
    return {
        "model": model.doc_string,
        "graph": model.graph.doc_string,
        "inputs": {i.name: i.doc_string for i in model.graph.input},
        "outputs": {o.name: o.doc_string for o in model.graph.output},
    }


def _restore_doc_strings(model: onnx.ModelProto, snapshot: dict) -> None:
    """Restore doc strings captured by :func:`_snapshot_doc_strings`.

    Only fields that the optimizer left empty are restored, so any doc string
    produced by the optimizer itself takes precedence.
    """
    if not model.doc_string and snapshot["model"]:
        model.doc_string = snapshot["model"]
    if not model.graph.doc_string and snapshot["graph"]:
        model.graph.doc_string = snapshot["graph"]
    for ipt in model.graph.input:
        if not ipt.doc_string and snapshot["inputs"].get(ipt.name):
            ipt.doc_string = snapshot["inputs"][ipt.name]
    for opt in model.graph.output:
        if not opt.doc_string and snapshot["outputs"].get(opt.name):
            opt.doc_string = snapshot["outputs"][opt.name]


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
    import_custom_schemas: bool = True,
    input_shapes=None,
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
    :param import_custom_schemas: Import operator schemas registered in the Python `onnx` module
            (e.g. via `onnx.defs.register_schema`) into onnxsim's own registry so models using
            custom operators pass validation. Set to False to disable this and leave onnxsim's
            registry untouched.
    :param input_shapes: Deprecated. Please use `overwrite_input_shapes` and/or `test_input_shapes` instead.
    :return: A tuple (simplified model, success(True) or failed(False))
    """
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

    if not perform_optimization:
        # None means skip all optimizers
        skipped_optimizers = None
    elif skipped_optimizers is None:
        skipped_optimizers = []

    if skip_fuse_bn and skipped_optimizers is not None:
        skipped_optimizers.append("fuse_bn_into_conv")
    # Track whether we own the in-memory model. When the caller passes a file
    # path we load it here, so the resulting ``ModelProto`` is private to this
    # function and may be mutated freely (e.g. saved as external data without a
    # defensive copy). When the caller passes their own ``ModelProto`` we must
    # not mutate it.
    model_owned = isinstance(model, str)
    if model_owned:
        model = onnx.load(model)
    if overwrite_input_shapes is None:
        overwrite_input_shapes = {}
    overwrite_input_shapes = check_and_update_input_shapes(
        model, overwrite_input_shapes)
    test_input_shapes = check_and_update_input_shapes(
        model, test_input_shapes)

    for name, input_shape in overwrite_input_shapes.items():
        for ipt in model.graph.input:
            if ipt.name == name:
                for i, dim in enumerate(ipt.type.tensor_type.shape.dim):
                    # A non-positive value means "keep the original (possibly
                    # dynamic) dimension" rather than hardcoding an invalid size
                    # such as 0, which would make the model impossible to run
                    # (see GitHub issue #237).
                    if input_shape[i] > 0:
                        dim.dim_value = input_shape[i]
    if unused_output is not None:
        model = remove_unused_output(model, unused_output)
    if not mutable_initializer and model.ir_version >= 4:
        model = remove_initializer_from_input(model)

    # onnxoptimizer's common-subexpression / duplicate-initializer passes hash
    # tensor values and crash with "no supported data type: <N>" on element
    # types they cannot hash, such as the float8 zero points produced by NVIDIA
    # ModelOpt fp8 QDQ models (GitHub issue #348). When such a tensor is present
    # we transparently skip those two passes so the rest of the simplification
    # still runs, instead of failing outright. ``skipped_optimizers is None``
    # means "skip every optimizer", so there is nothing to add in that case.
    if skipped_optimizers is not None and _has_cse_unhashable_tensor(model):
        added = [
            opt for opt in _TENSOR_VALUE_HASHING_OPTIMIZERS
            if opt not in skipped_optimizers
        ]
        if added:
            skipped_optimizers.extend(added)
            print(
                Text(
                    "The model contains tensors with element types that "
                    "onnxoptimizer cannot hash (e.g. float8/int4 in NVIDIA "
                    "ModelOpt fp8 QDQ models). Skipping the optimizers "
                    f"{added} to avoid a crash; all other simplifications "
                    "still run.",
                    style="bold magenta",
                )
            )

    # The C++ optimizer re-serializes the graph and drops the `doc_string`
    # fields on the model, graph and input/output value infos. Snapshot them
    # here so they can be restored on the simplified model (GitHub issue #428).
    doc_strings = _snapshot_doc_strings(model)

    def parse_size(size: str) -> int:
        m = re.fullmatch(r"([\d.]+)\s*([KMGT]?B)", size.strip(), re.I)
        if not m:
            raise ValueError(size)
        number: float = float(m.group(1))
        unit: Unit = m.group(2).upper()  # type: ignore
        return int(number * UNIT_MAP[unit])

    tensor_size_threshold = parse_size(tensor_size_threshold)
    if tensor_size_threshold > 2**31 - 9999:
        raise ValueError("tensor_size_threshold should be less than 2GB")

    try:
        model_bytes = model.SerializeToString()
        if len(model_bytes) >= 2 * 1024 * 1024 * 1024:
            model_bytes = None
            raise EncodeError("Message larger than 2GiB")
        model_opt_bytes = C.simplify(
            _get_model_executor(),
            model_bytes,
            skipped_optimizers,
            not skip_constant_folding,
            not skip_shape_inference,
            tensor_size_threshold,
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
            model_opt, model, check_n, test_input_shapes, input_data, custom_lib
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
        print("[bold magenta]Simplified model larger than 2GB. Trying to save as external data...[/bold magenta]")
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
                os.path.join(tmpdirname, 'model.onnx'),
                save_as_external_data=True,
            )
            check_ok = C.simplify_path(
                _get_model_executor(),
                os.path.join(tmpdirname, 'model.onnx'),
                os.path.join(tmpdirname, 'opt.onnx'),
                skipped_optimizers,
                not skip_constant_folding,
                not skip_shape_inference,
                tensor_size_threshold,
            )
            check_ok = model_checking.compare(
                os.path.join(tmpdirname, 'opt.onnx'),
                os.path.join(tmpdirname, 'model.onnx'),
                check_n, test_input_shapes, input_data, custom_lib
            )
            model_opt = onnx.load(os.path.join(tmpdirname, 'opt.onnx'))
    _restore_doc_strings(model_opt, doc_strings)
    _warn_if_shape_ops_remain(model_opt)
    return model_opt, check_ok


class PyModelExecutor(C.ModelExecutor):
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
        outputs = backend.run_model(model, inputs)
        # The inference backend may return a non-ndarray for an output (for
        # example onnxruntime yields an empty Python list for an empty sequence
        # output). onnx.numpy_helper.from_array only accepts numpy arrays, so
        # coerce any such value into an empty array instead of crashing with
        # "'list' object has no attribute 'shape'" (GitHub PR #249).
        return [
            onnx.numpy_helper.from_array(x).SerializeToString() if isinstance(x, np.ndarray) else x
            for x in outputs.values()
        ]


_model_executor: Optional[PyModelExecutor] = None


def _get_model_executor() -> PyModelExecutor:
    """Return the process-wide Python model executor, creating it on demand.

    The executor is passed explicitly to the C++ ``simplify``/``simplify_path``
    entry points instead of being registered as a global instance.
    """
    global _model_executor
    if _model_executor is None:
        _model_executor = PyModelExecutor()
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
    parser.add_argument("--skip-constant-folding", help="Skip constant folding", action="store_true")
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
        const='1KB',
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
        "--save-as-external-data",
        help="Save parameters as external data. This will make the .onnx file much smaller, but the .onnx file will depend on the external data file (.data).",
        action="store_true",
        )
    parser.add_argument(
        "--skip-schema-import",
        help="By default onnxsim imports operator schemas registered in the Python 'onnx' module (e.g. via onnx.defs.register_schema) into its own registry so models with custom operators pass validation. Specify this flag to disable that import.",
        action="store_true",
        )
    parser.add_argument('-v', '--version', action='version', version='onnxsim ' + version.version)

    class ListOptimizers(argparse.Action):
        def __call__(self, parser, ns, v, option_string=None):
            for p in C._list_optimizers():
                print(p)
            parser.exit()

    parser.add_argument("--list-default-optimizers", help="List default optimizer pass names", nargs=0, action=ListOptimizers)

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

    def parse_shapes(shapes_arg):
        shapes = {}
        if shapes_arg is not None:
            for x in shapes_arg:
                if ':' not in x:
                    shapes[None] = list(map(int, x.split(',')))
                else:
                    pieces = x.split(':')
                    # for the input name like input:0
                    name, shape = ':'.join(
                        pieces[:-1]), list(map(int, pieces[-1].split(',')))
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
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # To enable model serialization after graph optimization
        sess_options.optimized_model_filepath = tmp_file.name
        _ = rt.InferenceSession(args.input_model, sess_options, providers=["CPUExecutionProvider"])

        model = onnx.load(tmp_file.name)
    else:
        model = onnx.load(args.input_model)

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

    input_tensors = None
    if args.input_data_path is not None:
        input_tensors = {}
        for x in args.input_data_path:
            pieces = x.split(':')
            name, data = ':'.join(pieces[:-1]), pieces[-1]
            input_tensors.update({name: np.load(data)})

    print("Simplifying...")

    model_opt, check_ok = simplify(
        model,
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
        import_custom_schemas=not args.skip_schema_import,
    )

    try:
        if not args.save_as_external_data:
            onnx.save(model_opt, args.output_model)
        else:
            raise ValueError("save_as_external_data")
    except ValueError:
        # large models (>2GB) which onnx.save doesn't support,
        # or explicitly specified --save-as-external-data
        external_data_path = os.path.basename(args.output_model) + '.data'
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
    else:
        print(
            'Check failed. Please be careful to use the simplified model, or try specifying "--skip-fuse-bn" or "--skip-optimization" (run "onnxsim -h" for details).'
        )
        print("Here is the difference after simplification:")
        model_info.print_simplifying_info(model, model_opt)
        sys.exit(1)
