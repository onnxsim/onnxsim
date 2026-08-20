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


def load_model(path: str) -> onnx.ModelProto:
    """Load an ``.onnx`` model from ``path``, the same way :func:`simplify`
    loads a path by default.

    Structurally identical to ``onnx.load(path)`` -- the returned
    ``ModelProto`` is fully self-contained, with every tensor's data inline
    -- but any classic external data (the ``<name>.onnx`` + ``<name>.data``
    pair ``onnx.save(..., save_as_external_data=True)`` produces) is
    memory-mapped straight into onnxsim's C++ ``TensorPool`` and copied once
    from there into ``raw_data``, rather than first being read into a
    throwaway buffer the way ``onnx.load_external_data_for_model`` does.
    Models with no external data (or none large enough to matter) pay no
    extra cost either way.
    """
    model = onnx.load(path, load_external_data=False)
    model.ParseFromString(
        C.hydrate_external_data_pooled(
            model.SerializeToString(), os.path.dirname(os.path.abspath(path))
        )
    )
    return model


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
            is also printed to stdout. Implemented in the C++ core via the
            ``ONNXSIM_PROFILE`` environment variable, so it works from every
            binding; setting this argument simply sets that variable for the call.
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
    # Uses the same TensorPool-backed, memory-mapped loading path as
    # ``load_model`` and the fast path's ``C.simplify_path`` below, rather than
    # ``onnx.load_external_data_for_model``'s buffered read.
    if external_data_dir is not None:
        model.ParseFromString(
            C.hydrate_external_data_pooled(model.SerializeToString(), external_data_dir)
        )

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
        "--graph-diff",
        help="Print a node- and value-level diff between the original and "
        "simplified graphs after simplification, matched by output tensor "
        "name: which nodes/values were removed, added, or changed (e.g. a "
        "Conv whose bias input got folded away).",
        action="store_true",
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
