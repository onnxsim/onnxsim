import copy
import dataclasses
import math
import warnings
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeGuard,
    Union,
)

import numpy as np
import onnx
from onnx import defs, helper, numpy_helper, shape_inference
from onnx.external_data_helper import ExternalDataInfo, uses_external_data

from onnxsim._rich_compat import Table, Text, print

try:
    import sympy
except ImportError:  # sympy is an optional dependency; see _tensor_shape below.
    sympy = None

try:
    from onnx import inliner as onnx_inliner
except ImportError:  # onnx.inliner was added in onnx 1.14; see ModelInfo.__init__.
    onnx_inliner = None

try:
    import onnx_ir
    from onnx_shape_inference import infer_symbolic_shapes
except ImportError:  # optional dependency; see ModelInfo._infer_shapes below.
    onnx_ir = None
    infer_symbolic_shapes = None


__all__ = [
    "ModelInfo",
    "print_simplifying_info",
    "annotate_metadata",
    "METADATA_PREFIX",
    "GraphDiff",
    "NodeDiffEntry",
    "diff_graphs",
    "print_graph_diff",
    "WeightQuantizationError",
    "weight_quantization_error",
    "print_weight_quantization_error",
]

# metadata_props keys written by ``annotate_metadata`` are namespaced with this
# prefix (e.g. "onnxsim.macs") so they never collide with other producers.
METADATA_PREFIX = "onnxsim."


def _iter_graph_tensors(graph: onnx.GraphProto) -> Iterable[onnx.TensorProto]:
    """Yield every ``TensorProto`` stored in ``graph``, recursing into subgraphs.

    Covers initializers as well as tensors carried in node attributes (e.g. the
    ``value`` of a ``Constant``), matching every place a model may hold tensor
    data.
    """
    for initializer in graph.initializer:
        yield initializer
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            for tensor in attr.tensors:
                yield tensor
            if attr.HasField("g"):
                yield from _iter_graph_tensors(attr.g)
            for subgraph in attr.graphs:
                yield from _iter_graph_tensors(subgraph)


def _external_data_size(graph: onnx.GraphProto) -> int:
    """Total bytes of tensor data held in external files, from metadata alone.

    ``ExternalDataInfo(tensor).length`` reads the ``length`` entry of a tensor's
    ``external_data`` record, so this never loads the data itself: the size of a
    model whose weights live on disk can be reported without materializing them.
    """
    total = 0
    for tensor in _iter_graph_tensors(graph):
        if uses_external_data(tensor):
            total += ExternalDataInfo(tensor).length or 0
    return total


def human_readable_size(num, suffix="B"):
    # A symbolic byte count (dynamic shapes + sympy) is printed as the formula
    # itself, matching human_readable_num.
    if _is_symbolic(num):
        return _factor_or_str(num)
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def human_readable_num(num, suffix=""):
    # A symbolic MAC count (dynamic shapes + sympy) is printed as the formula
    # itself, e.g. "512*batch*seq**2 + 5419008*batch".
    if _is_symbolic(num):
        return _factor_or_str(num)
    for unit in ["", "K", "M", "G", "T", "P", "E"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Z{suffix}"


def human_readable_density(num, suffix=" FLOP/Byte"):
    # Arithmetic intensity is a small ratio; print it plainly. When symbolic
    # (dynamic shapes cancel unevenly) fall back to the factored formula.
    if _is_symbolic(num):
        return f"{_factor_or_str(num)}{suffix}"
    return f"{float(num):.2f}{suffix}"


# A dimension is a concrete int, a symbolic size (a sympy Symbol standing in for
# a ``dim_param`` such as "batch"), or None when the size is entirely unknown.
Dim = Union[int, "sympy.Expr", None]
# A MAC count is an int, or a sympy expression once any symbolic dim is involved.
Macs = Union[int, "sympy.Expr"]
ShapeMap = Dict[str, List[Dim]]
# Bytes-per-element for each tensor, keyed like ShapeMap. Drives the memory
# metrics (access traffic, peak footprint) alongside the shapes.
DTypeMap = Dict[str, int]


def _dim_symbol(name: str) -> "sympy.Expr":
    # Same dim_param name -> same symbol, so a dynamic dim shared across tensors
    # (e.g. "batch") combines correctly in the accumulated formula. Sizes are
    # positive integers, which lets sympy simplify/factor the result.
    return sympy.Symbol(name, positive=True, integer=True)


def _prod(vals: Iterable[Macs]) -> Macs:
    result: Macs = 1
    for v in vals:
        result = result * v
    return result


def _tensor_shape(type_proto: onnx.TypeProto) -> Optional[List[Dim]]:
    tensor_type = type_proto.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    shape: List[Dim] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.dim_param:
            # Dynamic (symbolic) dimension. With sympy we keep it as a symbol so
            # the MAC total becomes a formula in terms of it; without sympy we
            # assume 1 and report per-sample MACs (as onnx-tool does).
            shape.append(_dim_symbol(dim.dim_param) if sympy is not None else 1)
        else:
            # Rank is known but this dimension is entirely unknown (no value and
            # no name); it stays None and disables MAC counting for the node.
            shape.append(None)
    return shape


def _is_symbolic(value: Macs) -> bool:
    return (
        sympy is not None and isinstance(value, sympy.Expr) and bool(value.free_symbols)
    )


# sympy.factor() on a multivariate polynomial is at best exponential in its
# number of free symbols. Real models with properly-named/deduplicated dynamic
# dims (e.g. "batch", "sequence") only ever contribute a handful of distinct
# symbols, but a model whose shape inference didn't unify intermediate dynamic
# dims back to the named input dims can produce hundreds of distinct symbols --
# there factor() doesn't error, it just never returns in practical time. Skip
# it above this threshold; it is purely a formatting nicety, never worth more
# than a bounded amount of work.
_MAX_FACTOR_FREE_SYMBOLS = 16


def _factor_or_str(value: "sympy.Expr") -> str:
    # sympy.factor() is purely cosmetic here (a nicer-looking formula for the
    # report), but on models with many unresolved symbolic dims its polynomial
    # arithmetic can recurse deep enough to blow Python's recursion limit (seen
    # in practice on real-world models with 1000+ nodes, e.g. VOICEVOX's
    # predict_sing_f0.onnx), or simply take intractably long without ever
    # erroring (see ``_MAX_FACTOR_FREE_SYMBOLS`` above). Fall back to the
    # unfactored expression rather than crashing or hanging the whole report
    # over a formatting nicety.
    if len(value.free_symbols) > _MAX_FACTOR_FREE_SYMBOLS:
        return str(value)
    try:
        return str(sympy.factor(value))
    except RecursionError:
        return str(value)


def _representative_number(value: Macs) -> int:
    # Collapse a (possibly symbolic) MAC count to a single number by setting
    # every free dimension to 1. Used only for ordering and the summary table's
    # highlighting -- never for the reported value, which stays symbolic.
    # ``xreplace`` (a direct, purely syntactic tree substitution), not ``subs``
    # (which layers on structural-equality/simplification passes meant for
    # pattern-based substitution): every replacement here is an exact Symbol
    # swapped for a literal, and ``subs`` on that over hundreds of free symbols
    # -- as models with undeduplicated dynamic dims can produce -- takes
    # minutes where ``xreplace`` takes a fraction of a second.
    if sympy is not None and isinstance(value, sympy.Expr):
        value = value.xreplace({s: sympy.Integer(1) for s in value.free_symbols})
    return int(value)


def _max_macs(a: Macs, b: Macs) -> Macs:
    # Larger of two possibly-symbolic values, chosen by representative magnitude
    # (all free dims -> 1). The winner is returned intact, so a symbolic peak is
    # reported as its formula while the comparison itself stays decidable.
    return a if _representative_number(a) >= _representative_number(b) else b


def _collect_shapes(graph: onnx.GraphProto, inherited: ShapeMap) -> ShapeMap:
    shapes: ShapeMap = dict(inherited)
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        shape = _tensor_shape(value_info.type)
        if shape is not None:
            shapes[value_info.name] = shape
    for initializer in graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    return shapes


def _elem_size(elem_type: int) -> Optional[int]:
    """Bytes per element of an ONNX tensor element type, or None if it has no
    fixed width (e.g. STRING) or is unmapped. Unknown widths disable the memory
    metrics for the tensor rather than guessing.
    """
    try:
        dtype = helper.tensor_dtype_to_np_dtype(elem_type)
    except Exception:
        return None
    if dtype.kind in ("O", "U", "S", "V"):  # object / string / void: no fixed size
        return None
    return int(dtype.itemsize)


def _collect_dtypes(graph: onnx.GraphProto, inherited: DTypeMap) -> DTypeMap:
    dtypes: DTypeMap = dict(inherited)
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        esize = _elem_size(value_info.type.tensor_type.elem_type)
        if esize is not None:
            dtypes[value_info.name] = esize
    for initializer in graph.initializer:
        esize = _elem_size(initializer.data_type)
        if esize is not None:
            dtypes[initializer.name] = esize
    return dtypes


def _tensor_bytes(name: str, shapes: ShapeMap, dtypes: DTypeMap) -> Optional[Macs]:
    """Size in bytes of a named tensor, or None when its shape or element size is
    unknown. May be symbolic when a dimension is a ``dim_param``.
    """
    shape = shapes.get(name)
    esize = dtypes.get(name)
    if not _known(shape) or esize is None:
        return None
    return _prod(shape) * esize


def _node_memory_access(
    node: onnx.NodeProto, shapes: ShapeMap, dtypes: DTypeMap
) -> Macs:
    """Bytes a node moves in a forward pass: every input read plus every output
    written. Weights read from memory count as reads. Tensors with unknown size
    contribute 0 (best-effort, matching the MAC counters).
    """
    total: Macs = 0
    for name in list(node.input) + list(node.output):
        if not name:  # optional/omitted operand
            continue
        nbytes = _tensor_bytes(name, shapes, dtypes)
        if nbytes is not None:
            total += nbytes
    return total


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


def _known(shape: Optional[List[Dim]]) -> TypeGuard[List[Macs]]:
    return shape is not None and len(shape) > 0 and all(d is not None for d in shape)


# --- Per-op MAC (multiply-accumulate) counters --------------------------------
# Each counter returns the number of MACs for a single node, or 0 when the
# required tensor shapes are unknown. FLOPs are reported as 2 * MACs.
# Coverage is intentionally limited to the compute-dominant operators; these
# account for the vast majority of a typical model's arithmetic.
_MAC_COUNTERS: Dict[str, Callable[[onnx.NodeProto, ShapeMap], Macs]] = {}


def _register(*op_types: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        for op_type in op_types:
            _MAC_COUNTERS[op_type] = fn
        return fn

    return deco


# QLinearConv reorders inputs: x, x_scale, x_zp, w, w_scale, w_zp, ... so the
# weight is input[3] rather than input[1]. Everything else about the MAC count
# (a quantized conv does the same multiply-accumulates as its float twin) is
# identical, so the counters below just point at the right operand indices.
@_register("Conv", "ConvInteger")
def _conv_macs(node: onnx.NodeProto, shapes: ShapeMap) -> int:
    return _conv_macs_impl(node, shapes, weight_idx=1)


@_register("QLinearConv")
def _qlinearconv_macs(node: onnx.NodeProto, shapes: ShapeMap) -> int:
    return _conv_macs_impl(node, shapes, weight_idx=3)


def _conv_macs_impl(node: onnx.NodeProto, shapes: ShapeMap, weight_idx: int) -> Macs:
    # weight: [out_channels, in_channels / group, *kernel_shape]
    # output: [batch, out_channels, *spatial_out]
    weight = shapes.get(node.input[weight_idx])
    output = shapes.get(node.output[0])
    if not _known(weight) or not _known(output):
        return 0
    in_channels_per_group = weight[1]
    kernel = weight[2:]
    return _prod(output) * in_channels_per_group * _prod(kernel)


@_register("ConvTranspose")
def _conv_transpose_macs(node: onnx.NodeProto, shapes: ShapeMap) -> int:
    # weight: [in_channels, out_channels / group, *kernel_shape]
    # input:  [batch, in_channels, *spatial_in]
    x = shapes.get(node.input[0])
    weight = shapes.get(node.input[1])
    if not _known(x) or not _known(weight):
        return 0
    out_channels_per_group = weight[1]
    kernel = weight[2:]
    return _prod(x) * out_channels_per_group * _prod(kernel)


@_register("Gemm")
def _gemm_macs(node: onnx.NodeProto, shapes: ShapeMap) -> int:
    a = shapes.get(node.input[0])
    b = shapes.get(node.input[1])
    if not _known(a) or not _known(b) or len(a) != 2 or len(b) != 2:
        return 0
    trans_a = _attr_int(node, "transA", 0)
    trans_b = _attr_int(node, "transB", 0)
    m, k = (a[1], a[0]) if trans_a else (a[0], a[1])
    n = b[0] if trans_b else b[1]
    return m * n * k


# MatMul, MatMulInteger and QLinearMatMul all take A as input[0] and produce Y
# as output[0]; the quantized variants only add scale/zero-point operands, which
# don't affect the multiply-accumulate count.
@_register("MatMul", "MatMulInteger", "QLinearMatMul")
def _matmul_macs(node: onnx.NodeProto, shapes: ShapeMap) -> Macs:
    # output: [*batch, M, N]; contraction dim K is the last dim of input A.
    a = shapes.get(node.input[0])
    output = shapes.get(node.output[0])
    if not _known(a) or not _known(output):
        return 0
    k = a[-1]
    return _prod(output) * k


@_register("Attention")
def _attention_macs(node: onnx.NodeProto, shapes: ShapeMap) -> Macs:
    """Scaled dot-product attention (ai.onnx opset 23+).

    Cost is dominated by the two batched matmuls, QK^T and (softmax)V; the
    softmax/scale/mask are elementwise and negligible by comparison. For heads
    ``h``, query/key sequence lengths ``sq``/``skv`` and per-head sizes ``d``
    (QK) and ``dv`` (V):

        QK^T : batch * h * sq * skv * d
        P V  : batch * h * sq * skv * dv

    Q/K/V are either 4D ``(batch, num_heads, seq, head_size)`` or 3D
    ``(batch, seq, num_heads * head_size)``; in the 3D form the head count comes
    from the ``q_num_heads`` / ``kv_num_heads`` attributes. Grouped-query
    attention (kv heads < q heads) still evaluates all ``q_num_heads`` query
    heads, so the query head count drives both matmuls.
    """
    q = shapes.get(node.input[0])
    k = shapes.get(node.input[1])
    v = shapes.get(node.input[2])
    if not _known(q) or not _known(k) or not _known(v):
        return 0

    if len(q) == 4 and len(k) == 4 and len(v) == 4:
        batch, q_heads, sq, d = q
        skv = k[2]
        dv = v[3]
    elif len(q) == 3 and len(k) == 3 and len(v) == 3:
        q_heads = _attr_int(node, "q_num_heads", 0)
        kv_heads = _attr_int(node, "kv_num_heads", 0)
        if q_heads <= 0 or kv_heads <= 0:
            return 0  # head split is unknowable without the attributes
        batch, sq, _ = q
        skv = k[1]
        d = k[2] // kv_heads  # per-head size (K packs kv_heads * head_size)
        dv = v[2] // kv_heads
    else:
        return 0

    qk = _prod([batch, q_heads, sq, skv, d])
    pv = _prod([batch, q_heads, sq, skv, dv])
    return qk + pv


def _type_map(graph: onnx.GraphProto) -> Dict[str, onnx.TypeProto]:
    types: Dict[str, onnx.TypeProto] = {}
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value_info.type.ByteSize():
            types[value_info.name] = value_info.type
    for initializer in graph.initializer:
        types[initializer.name] = helper.make_tensor_type_proto(
            initializer.data_type, list(initializer.dims)
        )
    return types


def _schema_function_body(
    node: onnx.NodeProto, types: Dict[str, onnx.TypeProto], opsets: Dict[str, int]
) -> Optional[onnx.FunctionProto]:
    """The expanded body of a context-dependent schema function, or None.

    Only context-dependent functions (e.g. Attention, LayerNormalization) are
    handled: their body is generated from the node's attributes and input types,
    so it needs no attribute plumbing when turned into a local function.
    """
    version = opsets.get(node.domain, opsets.get("", 1))
    try:
        schema = defs.get_schema(node.op_type, version, node.domain)
    except Exception:
        return None
    if not schema.has_context_dependent_function:
        return None
    # Every non-optional input must have a known type to generate the body.
    if any(name and types.get(name) is None for name in node.input):
        return None
    input_types = [
        (
            types[name].SerializeToString()
            if name
            else onnx.TypeProto().SerializeToString()
        )
        for name in node.input
    ]
    try:
        body_bytes = schema.get_context_dependent_function(
            node.SerializeToString(), input_types
        )
    except Exception:
        return None
    body = onnx.FunctionProto()
    body.ParseFromString(body_bytes)
    return body


def _expand_schema_functions(model: onnx.ModelProto) -> None:
    """In place: turn schema-registered function ops that have no bespoke MAC
    counter into uniquely-named model-local functions, so a later inlining pass
    exposes the compute (MatMuls, Convs, ...) inside their bodies.

    Best-effort and top-level only: a node whose body cannot be generated is
    left untouched (and thus counts as 0, as before). Ops with a bespoke counter
    are skipped so their exact counters keep winning.
    """
    opsets = {imp.domain: imp.version for imp in model.opset_import}
    types = _type_map(ModelInfo._infer_shapes(model).graph)
    existing = {(f.domain, f.name) for f in model.functions}
    new_functions = []
    for i, node in enumerate(model.graph.node):
        if node.op_type in _MAC_COUNTERS or (node.domain, node.op_type) in existing:
            continue
        body = _schema_function_body(node, types, opsets)
        if body is None:
            continue
        domain = f"__macs_expand_{i}"
        body.name = f"{node.op_type}__expanded"
        body.domain = domain
        new_functions.append(body)
        node.op_type = body.name
        node.domain = domain
        del node.attribute[:]  # the body is already specialized for these attrs
        model.opset_import.append(helper.make_opsetid(domain, 1))
    model.functions.extend(new_functions)


def _poly_to_metric(poly: List[Tuple[int, List[str]]]) -> Macs:
    """Rebuild a metric from the C++ polynomial form -- a list of
    ``(coeff, [dim_name, ...])`` terms -- into the value the pure-Python counters
    produced: a plain int when concrete, or the same sympy expression (dim_params
    as positive-integer symbols) when a dynamic dimension is involved. Without
    sympy every symbol collapses to 1, matching the old per-sample fallback.
    """
    if sympy is None:
        return sum((coeff for coeff, _ in poly), 0)
    total: Macs = 0
    for coeff, monomial in poly:
        term: Macs = coeff
        for name in monomial:
            term = term * _dim_symbol(name)
        total = total + term
    return total


def _cpp_metrics(
    model: onnx.ModelProto, run_shape_inference: bool = True
) -> Tuple[Dict[str, int], int, Macs, Macs, Macs]:
    """Delegate the metric counting to the C++ implementation and rebuild the
    symbolic metrics as sympy expressions. Returns
    ``(op_nums, model_size, macs, mem_access, memory_footprint)``.
    """
    # Imported lazily so importing ``onnxsim.model_info`` never forces the
    # compiled extension at module load, and cannot form an import cycle with the
    # package __init__.
    from onnxsim import onnxsim_cpp2py_export as _C

    op_nums, model_size, macs, mem_access, footprint = _C._model_metrics(
        model.SerializeToString(), run_shape_inference
    )
    return (
        dict(op_nums),
        model_size,
        _poly_to_metric(macs),
        _poly_to_metric(mem_access),
        _poly_to_metric(footprint),
    )


class ModelInfo:
    """
    Model info contains:
    1. Num of every op
    2. Model size
    3. Count of the top-level graph's own initializers (``initializer_count``)
    4. MACs / FLOPs of the compute-dominant operators: Conv, ConvTranspose,
       Gemm, MatMul, Attention, and the quantized twins (ConvInteger,
       QLinearConv, MatMulInteger, QLinearMatMul). Shapes come from ONNX shape
       inference -- or, when the optional ``onnx-shape-inference`` package
       (https://github.com/justinchuby/onnx-shape-inference) is installed, its
       symbolic shape inference, which resolves more shapes via data
       propagation through chains like Shape -> Slice -> Concat -> Reshape;
       nodes whose shapes still cannot be inferred contribute 0. Dynamic
       dimensions (``dim_param``, e.g. "batch") become sympy symbols when sympy
       is installed, so ``macs`` / ``flops`` may be a symbolic formula; without
       sympy they are assumed 1 (per-sample MACs). Function ops are expanded
       before counting, so the compute inside their bodies is included in the
       MAC total while op counts still list the function op itself: model-local
       functions (and nested ones) are inlined, and schema-registered function
       ops without a bespoke counter fall back to their context-dependent body
       (best-effort).
    5. Memory metrics, derived statically from the same inferred shapes (no
       runtime execution needed):
       - ``mem_access``: total bytes read and written across a forward pass --
         every node's inputs (weights included) plus its outputs.
       - ``memory_footprint``: peak bytes resident at once, from a liveness pass
         over the topologically-ordered nodes -- weights stay resident while each
         activation lives only from where it is produced to its last use.
       - ``compute_density``: arithmetic intensity, ``flops / mem_access``
         (FLOP per byte), the roofline ratio.
       Tensors whose shape or element size is unknown contribute 0, so these are
       best-effort lower bounds; with dynamic dims they may be symbolic too.
    """

    def get_info(
        self,
        graph: onnx.GraphProto,
        inherited_shapes: Optional[ShapeMap] = None,
        inherited_dtypes: Optional[DTypeMap] = None,
    ) -> Tuple[Dict[str, int], Macs, Macs]:
        if inherited_shapes is None:
            inherited_shapes = {}
        if inherited_dtypes is None:
            inherited_dtypes = {}
        shapes = _collect_shapes(graph, inherited_shapes)
        dtypes = _collect_dtypes(graph, inherited_dtypes)
        op_nums: Dict[str, int] = defaultdict(int)
        macs = 0
        mem_access = 0
        for node in graph.node:
            op_nums[node.op_type] += 1
            counter = _MAC_COUNTERS.get(node.op_type)
            if counter is not None:
                try:
                    macs += counter(node, shapes)
                except Exception as e:
                    warnings.warn(
                        f"Failed to count MACs for {node.op_type} node "
                        f"'{node.name}' ({e}); it is excluded from the total.",
                        stacklevel=2,
                    )
            mem_access += _node_memory_access(node, shapes, dtypes)
            for attr in node.attribute:
                sub_graphs = []
                if attr.HasField("g"):
                    sub_graphs.append(attr.g)
                sub_graphs.extend(attr.graphs)
                for sub_graph in sub_graphs:
                    sub_op_nums, sub_macs, sub_mem = self.get_info(
                        sub_graph, shapes, dtypes
                    )
                    op_nums = defaultdict(
                        int,
                        {
                            k: op_nums[k] + sub_op_nums[k]
                            for k in set(op_nums) | set(sub_op_nums)
                        },
                    )
                    macs += sub_macs
                    mem_access += sub_mem
        op_nums["Constant"] += len(graph.initializer)
        return op_nums, macs, mem_access

    def __init__(self, model: onnx.ModelProto):
        # Delegate the counting to the single C++ implementation, which returns
        # op counts, model size, and the compute/memory metrics as polynomials;
        # _cpp_metrics rebuilds the symbolic ones into the same sympy expressions
        # the pure-Python counters used to produce. Op counts and size describe
        # the model as authored, so a function op is reported as a single op and
        # the size is the serialized graph plus external-data lengths (correct
        # whether or not the weights on disk have been loaded).
        op_nums, self.model_size, macs, mem_access, footprint = _cpp_metrics(model)
        self.op_nums = defaultdict(int, op_nums)
        # The top-level graph's own initializer count -- unlike
        # ``op_nums["Constant"]``, which folds initializers (recursively,
        # subgraphs included) together with actual ``Constant`` nodes, this is
        # reported as its own "Initializers" row so a change there (weights
        # folded into a fused node, duplicate initializers deduplicated, ...)
        # is visible on its own.
        self.initializer_count = len(model.graph.initializer)
        # A function op's compute lives in its body, so recount MACs and the
        # memory metrics on the function-expanded (inlined) graph -- the counters
        # then see the MatMuls, Convs, etc. inside every function instance. The
        # expanded model already carries shapes (inferred with data propagation),
        # so C++ need not infer them again.
        expanded = self._expanded_macs_model(model)
        if expanded is not None:
            _, _, macs, mem_access, footprint = _cpp_metrics(
                expanded, run_shape_inference=False
            )
        self.macs = macs
        self.mem_access = mem_access
        self.memory_footprint = footprint

    @staticmethod
    def _expanded_macs_model(model: onnx.ModelProto) -> Optional[onnx.ModelProto]:
        """A shape-inferred model with function bodies exposed for MAC counting,
        or None when there is nothing to expand.

        Model-local functions are inlined, and schema-registered function ops
        without a bespoke counter are first converted to local functions (see
        ``_expand_schema_functions``). Shapes are inferred with data propagation
        so the dynamic reshapes inside generated bodies (e.g. Attention) resolve
        and the internal MatMuls get concrete shapes.
        """
        if onnx_inliner is None:
            return None
        try:
            work = copy.deepcopy(model)
            _expand_schema_functions(work)
            if not work.functions:
                return None
            inlined = onnx_inliner.inline_local_functions(work)
            return ModelInfo._infer_shapes(inlined, data_prop=True)
        except Exception as e:
            warnings.warn(
                f"Failed to expand function bodies ({e}); MACs inside function "
                "bodies are not counted.",
                stacklevel=2,
            )
            return None

    @staticmethod
    def _infer_shapes(
        model: onnx.ModelProto, data_prop: bool = False
    ) -> onnx.ModelProto:
        # onnx-shape-inference (https://github.com/justinchuby/onnx-shape-inference),
        # when installed, resolves more shapes than onnx's own shape_inference: it
        # always does data propagation and tracks values through chains like
        # Shape -> Slice -> Concat -> Reshape, so dynamic reshapes that onnx leaves
        # unknown often still get a shape here. Its dim_param names for dynamic
        # dims are still picked up as sympy symbols by _tensor_shape below.
        if infer_symbolic_shapes is not None:
            try:
                inferred = infer_symbolic_shapes(onnx_ir.from_proto(model))
                return onnx_ir.to_proto(inferred)
            except Exception as e:
                warnings.warn(
                    f"onnx-shape-inference failed ({e}); falling back to "
                    "onnx.shape_inference.",
                    stacklevel=2,
                )
        try:
            return shape_inference.infer_shapes(model, data_prop=data_prop)
        except Exception as e:
            # Shape inference can fail (e.g. models > 2GB); MACs then fall back
            # to 0 for nodes without pre-existing value_info.
            warnings.warn(
                f"Shape inference failed ({e}); MACs/FLOPs may be underestimated "
                "for nodes without existing shape info.",
                stacklevel=2,
            )
            return model

    def _peak_memory_footprint(
        self,
        graph: onnx.GraphProto,
        inherited_shapes: Optional[ShapeMap] = None,
        inherited_dtypes: Optional[DTypeMap] = None,
    ) -> Macs:
        """Peak bytes resident during a forward pass of ``graph``.

        A simple liveness pass over the topologically-ordered nodes (ONNX
        requires that order): weights (initializers) stay resident throughout,
        while every other tensor lives from where it is produced until its last
        consumer -- graph outputs live to the end. The peak is the largest total
        of resident-weights + live-activations at any node. Control-flow
        subgraphs add their own recursive peak on top of the live set at the
        owning node (a conservative bound, since captured tensors are counted in
        both). Unknown-size tensors contribute 0.
        """
        shapes = _collect_shapes(graph, inherited_shapes or {})
        dtypes = _collect_dtypes(graph, inherited_dtypes or {})

        def nbytes(name: str) -> Macs:
            size = _tensor_bytes(name, shapes, dtypes)
            return size if size is not None else 0

        weight_names = {init.name for init in graph.initializer}
        resident: Macs = 0
        for name in weight_names:
            resident += nbytes(name)

        # Last node index that consumes each tensor; graph outputs are "consumed"
        # at the end so they stay live for the whole pass.
        last_use: Dict[str, int] = {}
        for i, node in enumerate(graph.node):
            for name in node.input:
                if name:
                    last_use[name] = i
        end = len(graph.node)
        for out in graph.output:
            last_use[out.name] = end

        def live_bytes(live: set) -> Macs:
            total: Macs = resident
            for name in live:
                total += nbytes(name)
            return total

        # Graph inputs (non-weight) are available from the start.
        live = {inp.name for inp in graph.input if inp.name not in weight_names}
        peak = live_bytes(live)
        for i, node in enumerate(graph.node):
            for name in node.output:
                if name and name not in weight_names:
                    live.add(name)
            current = live_bytes(live)
            # Nested subgraphs run while this node's live set is held.
            for attr in node.attribute:
                sub_graphs = []
                if attr.HasField("g"):
                    sub_graphs.append(attr.g)
                sub_graphs.extend(attr.graphs)
                for sub_graph in sub_graphs:
                    sub_peak = self._peak_memory_footprint(sub_graph, shapes, dtypes)
                    current = current + sub_peak
            peak = _max_macs(peak, current)
            for name in [n for n in live if last_use.get(n) == i]:
                live.discard(name)
        return peak

    @property
    def flops(self) -> int:
        return self.macs * 2

    @property
    def compute_density(self) -> Macs:
        """Arithmetic intensity: FLOPs per byte of memory traffic (roofline).

        0 when no traffic is known. Symbolic when the dynamic dimensions do not
        cancel between FLOPs and bytes.
        """
        if _representative_number(self.mem_access) == 0:
            return 0
        return self.flops / self.mem_access


def print_simplifying_info(
    model_ori: onnx.ModelProto, model_opt: onnx.ModelProto
) -> None:
    """
    --------------------------------------------------------
    |             | original model | simplified model |
    --------------------------------------------------------
    | ****        | ****           | ****             |
    --------------------------------------------------------
    | Model Size  | ****           | ****             |
    --------------------------------------------------------
    """
    ori_info = ModelInfo(model_ori)
    opt_info = ModelInfo(model_opt)
    table = Table()
    table.add_column("")
    table.add_column("Original Model")
    table.add_column("Simplified Model")

    def add_row(
        table: Table,
        key,
        ori_data,
        opt_data,
        is_better: Callable[[Any, Any], Any],
        postprocess: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if postprocess is None:
            postprocess = str
        if is_better(opt_data, ori_data):
            table.add_row(
                key,
                postprocess(ori_data),
                Text(postprocess(opt_data), style="bold green1"),
            )
        else:
            table.add_row(key, postprocess(ori_data), postprocess(opt_data))

    for key in sorted(
        list(set(ori_info.op_nums.keys()) | set(opt_info.op_nums.keys()))
    ):
        add_row(
            table,
            key,
            ori_info.op_nums[key],
            opt_info.op_nums[key],
            lambda opt, ori: opt < ori,
        )
    add_row(
        table,
        "Model Size",
        ori_info.model_size,
        opt_info.model_size,
        lambda opt, ori: opt < ori,
        postprocess=human_readable_size,
    )
    add_row(
        table,
        "Initializers",
        ori_info.initializer_count,
        opt_info.initializer_count,
        lambda opt, ori: opt < ori,
    )

    # MACs/FLOPs may be symbolic, for which "<" yields an undecidable sympy
    # relational; compare representative magnitudes (all free dims -> 1) so the
    # highlighting still works without raising.
    def macs_improved(opt: Macs, ori: Macs) -> bool:
        return _representative_number(opt) < _representative_number(ori)

    add_row(
        table,
        "MACs",
        ori_info.macs,
        opt_info.macs,
        macs_improved,
        postprocess=human_readable_num,
    )
    add_row(
        table,
        "FLOPs",
        ori_info.flops,
        opt_info.flops,
        macs_improved,
        postprocess=human_readable_num,
    )
    add_row(
        table,
        "Memory Access",
        ori_info.mem_access,
        opt_info.mem_access,
        macs_improved,
        postprocess=human_readable_size,
    )
    add_row(
        table,
        "Memory Footprint",
        ori_info.memory_footprint,
        opt_info.memory_footprint,
        macs_improved,
        postprocess=human_readable_size,
    )
    # Compute density (FLOP/Byte): a change here isn't strictly "better or
    # worse", so it is reported without highlighting.
    add_row(
        table,
        "Compute Density",
        ori_info.compute_density,
        opt_info.compute_density,
        lambda opt, ori: False,
        postprocess=human_readable_density,
    )
    print(table)


# --------------------------------------------------------------------------- #
# Node/value level diff between the original and simplified graph
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class NodeDiffEntry:
    """One node's identity for diffing: enough to describe it on a diff line,
    not the full ``NodeProto`` (attributes are omitted; a node whose attributes
    changed but whose op_type/inputs/outputs did not is not detected as
    "changed" -- see ``diff_graphs``).
    """

    op_type: str
    name: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]


def _node_diff_entry(node: onnx.NodeProto) -> NodeDiffEntry:
    return NodeDiffEntry(
        op_type=node.op_type,
        name=node.name,
        inputs=tuple(node.input),
        outputs=tuple(node.output),
    )


@dataclasses.dataclass(frozen=True)
class GraphDiff:
    """Node- and value-level diff between an original and simplified graph,
    matched by name -- see ``diff_graphs``.
    """

    removed_nodes: List[NodeDiffEntry]
    added_nodes: List[NodeDiffEntry]
    # (before, after) pairs that produce the same output(s) but whose op_type
    # or inputs changed, e.g. a Conv whose bias input was folded away.
    changed_nodes: List[Tuple[NodeDiffEntry, NodeDiffEntry]]
    removed_values: List[str]
    added_values: List[str]


def _graph_value_names(graph: onnx.GraphProto) -> Set[str]:
    # Every named tensor the (top-level) graph produces or consumes: node
    # outputs, initializers and graph inputs. Node *inputs* that are neither an
    # initializer nor another node's output are graph inputs already, so this
    # set covers every value a diff could plausibly care about.
    names: Set[str] = set()
    for node in graph.node:
        names.update(n for n in node.output if n)
    names.update(init.name for init in graph.initializer)
    names.update(inp.name for inp in graph.input)
    return names


def diff_graphs(model_ori: onnx.ModelProto, model_opt: onnx.ModelProto) -> GraphDiff:
    """Diff ``model_ori``'s top-level graph against ``model_opt``'s, matched by
    name rather than position, so the result reflects what simplification
    actually did to named values instead of a meaningless positional diff.

    Nodes are matched by their output tensor name(s) -- unique within a graph
    by the ONNX spec, and the identity a downstream consumer actually depends
    on -- so a node is reported as "changed" (rather than removed + added) only
    when simplification kept the same output name(s) but altered the op_type or
    inputs (e.g. folding a Conv's bias into its weight). This is precisely why
    onnxsim tries to preserve value names across simplification passes where
    possible: it is what keeps this diff (and any other name-keyed tooling)
    meaningful instead of turning every fused/folded node into an unrelated
    remove+add pair.

    Only the top-level graph is compared -- nodes inside control-flow
    subgraphs (If/Loop/Scan bodies) are not matched across models, since names
    are only required to be unique within their own graph scope.
    """
    ori_by_output = {
        _node_diff_entry(n).outputs: _node_diff_entry(n) for n in model_ori.graph.node
    }
    opt_by_output = {
        _node_diff_entry(n).outputs: _node_diff_entry(n) for n in model_opt.graph.node
    }

    removed_nodes = []
    changed_nodes = []
    for outputs, before in ori_by_output.items():
        after = opt_by_output.get(outputs)
        if after is None:
            removed_nodes.append(before)
        elif (after.op_type, after.inputs) != (before.op_type, before.inputs):
            changed_nodes.append((before, after))
    added_nodes = [
        after
        for outputs, after in opt_by_output.items()
        if outputs not in ori_by_output
    ]

    ori_values = _graph_value_names(model_ori.graph)
    opt_values = _graph_value_names(model_opt.graph)

    return GraphDiff(
        removed_nodes=removed_nodes,
        added_nodes=added_nodes,
        changed_nodes=changed_nodes,
        removed_values=sorted(ori_values - opt_values),
        added_values=sorted(opt_values - ori_values),
    )


def _node_label(entry: NodeDiffEntry) -> str:
    name = entry.name or "/".join(entry.outputs)
    return f"{entry.op_type} ({name})"


def print_graph_diff(
    model_ori: onnx.ModelProto, model_opt: onnx.ModelProto, limit: int = 50
) -> None:
    """Print the node- and value-level diff between ``model_ori`` and
    ``model_opt`` (see ``diff_graphs``), in a unified-diff style: ``-`` for
    what simplification removed, ``+`` for what it added, ``~`` for a node kept
    under the same output name(s) but changed. Each section is capped at
    ``limit`` entries (with a "... and N more" line) so a large model's diff
    stays readable.
    """
    diff = diff_graphs(model_ori, model_opt)

    def print_section(title: str, count: int) -> None:
        print(f"[bold]{title} ({count}):[/bold]")

    def print_capped(lines: List[Text]) -> None:
        for line in lines[:limit]:
            print(line)
        if len(lines) > limit:
            print(Text(f"  ... and {len(lines) - limit} more", style="dim"))

    print(Text("Graph diff (matched by node output / value name):", style="bold"))

    print_section("Nodes removed", len(diff.removed_nodes))
    print_capped(
        [Text(f"  - {_node_label(n)}", style="red") for n in diff.removed_nodes]
    )

    print_section("Nodes added", len(diff.added_nodes))
    print_capped(
        [Text(f"  + {_node_label(n)}", style="green") for n in diff.added_nodes]
    )

    print_section("Nodes changed", len(diff.changed_nodes))
    changed_lines = []
    for before, after in diff.changed_nodes:
        label = _node_label(after)
        if before.op_type != after.op_type:
            changed_lines.append(
                Text(
                    f"  ~ {label}: {before.op_type} -> {after.op_type}", style="yellow"
                )
            )
        else:
            changed_lines.append(
                Text(
                    f"  ~ {label}: inputs {list(before.inputs)} -> {list(after.inputs)}",
                    style="yellow",
                )
            )
    print_capped(changed_lines)

    print_section("Values removed", len(diff.removed_values))
    print_capped([Text(f"  - {v}", style="red") for v in diff.removed_values])

    print_section("Values added", len(diff.added_values))
    print_capped([Text(f"  + {v}", style="green") for v in diff.added_values])


def _metric_str(value: Macs) -> str:
    # A metadata value is always a string. A symbolic metric is stored as its
    # factored formula (e.g. "512*batch"); a concrete one as its plain number.
    if _is_symbolic(value):
        return _factor_or_str(value)
    return str(value)


def _supports_metadata(proto) -> bool:
    # metadata_props on Node/Graph/ValueInfo/Tensor was added in newer onnx
    # releases; skip the level cleanly on older ones instead of crashing.
    return any(f.name == "metadata_props" for f in proto.DESCRIPTOR.fields)


def _set_metadata(proto, key: str, value: str) -> None:
    """Set ``key`` -> ``value`` in ``proto.metadata_props``, overwriting any
    existing entry with that key. No-op if the proto has no metadata_props.
    """
    if not _supports_metadata(proto):
        return
    for entry in proto.metadata_props:
        if entry.key == key:
            entry.value = value
            return
    entry = proto.metadata_props.add()
    entry.key = key
    entry.value = value


def _annotate_graph(
    graph: onnx.GraphProto,
    prefix: str,
    inherited_shapes: ShapeMap,
    inherited_dtypes: DTypeMap,
) -> Tuple[Macs, Macs]:
    """Write per-node and per-value metrics onto ``graph`` in place and return
    the graph's ``(macs, mem_access)`` subtotal.

    Per-value tensors (inputs, outputs, value_info and initializers) get a
    ``<prefix>bytes`` entry; each node gets ``macs`` / ``flops`` / ``mem_access``;
    the graph itself gets its own aggregate of those three. Values come from the
    graph as authored, so a function-op node counts only what a bespoke counter
    knows (0 otherwise) -- the model-level totals additionally include function
    bodies (see ``annotate_metadata``).
    """
    shapes = _collect_shapes(graph, inherited_shapes)
    dtypes = _collect_dtypes(graph, inherited_dtypes)

    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        nbytes = _tensor_bytes(value_info.name, shapes, dtypes)
        if nbytes is not None:
            _set_metadata(value_info, prefix + "bytes", _metric_str(nbytes))
    for initializer in graph.initializer:
        nbytes = _tensor_bytes(initializer.name, shapes, dtypes)
        if nbytes is not None:
            _set_metadata(initializer, prefix + "bytes", _metric_str(nbytes))

    macs: Macs = 0
    mem_access: Macs = 0
    for node in graph.node:
        counter = _MAC_COUNTERS.get(node.op_type)
        node_macs: Macs = 0
        if counter is not None:
            try:
                node_macs = counter(node, shapes)
            except Exception:
                node_macs = 0
        node_mem = _node_memory_access(node, shapes, dtypes)
        _set_metadata(node, prefix + "macs", _metric_str(node_macs))
        _set_metadata(node, prefix + "flops", _metric_str(node_macs * 2))
        _set_metadata(node, prefix + "mem_access", _metric_str(node_mem))
        macs += node_macs
        mem_access += node_mem
        for attr in node.attribute:
            sub_graphs = []
            if attr.HasField("g"):
                sub_graphs.append(attr.g)
            sub_graphs.extend(attr.graphs)
            for sub_graph in sub_graphs:
                sub_macs, sub_mem = _annotate_graph(sub_graph, prefix, shapes, dtypes)
                macs += sub_macs
                mem_access += sub_mem

    _set_metadata(graph, prefix + "macs", _metric_str(macs))
    _set_metadata(graph, prefix + "flops", _metric_str(macs * 2))
    _set_metadata(graph, prefix + "mem_access", _metric_str(mem_access))
    return macs, mem_access


def annotate_metadata(
    model: onnx.ModelProto, prefix: str = METADATA_PREFIX
) -> onnx.ModelProto:
    """Return a shape-inferred copy of ``model`` with the computed metrics stored
    in ``metadata_props`` at three levels, so downstream tools can read them back:

    - **Model** (and every graph): ``<prefix>macs``, ``flops``, ``mem_access``.
      The model additionally carries ``memory_footprint``, ``compute_density``
      and ``model_size``. Model-level totals come from :class:`ModelInfo`, so
      they include the compute inside inlined function bodies; a graph's own
      entries are the sum over the nodes it literally contains.
    - **Node**: ``<prefix>macs`` / ``flops`` / ``mem_access`` for that node.
    - **Value** (inputs, outputs, value_info, initializers): ``<prefix>bytes``.

    Values are strings (a symbolic metric is its formula, e.g. ``"512*batch"``).
    Tensors of unknown shape/dtype are simply left unannotated. The input model
    is never mutated; the returned copy is shape-inferred so intermediate values
    carry the shapes the per-value/per-node metrics are derived from.
    """
    info = ModelInfo(model)
    # Work on an inferred copy: infer_shapes returns a fresh model on success,
    # but the original object on failure -- deep-copy first so the caller's model
    # is never touched and the annotations land on populated value_info.
    work = ModelInfo._infer_shapes(copy.deepcopy(model))

    _annotate_graph(work.graph, prefix, {}, {})

    totals = {
        "macs": info.macs,
        "flops": info.flops,
        "mem_access": info.mem_access,
        "memory_footprint": info.memory_footprint,
        "compute_density": info.compute_density,
        "model_size": info.model_size,
    }
    for key, value in totals.items():
        _set_metadata(work, prefix + key, _metric_str(value))
    # Overwrite the top graph's authored-node aggregates with the model totals so
    # graph- and model-level macs/flops/mem_access agree (they differ only when
    # function bodies contribute compute the authored nodes don't show).
    for key in ("macs", "flops", "mem_access"):
        _set_metadata(work.graph, prefix + key, _metric_str(totals[key]))
    return work


# --------------------------------------------------------------------------- #
# Weight quantization error: original float weight vs. its dequantized
# (quantize -> dequantize round-trip) reconstruction
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class WeightQuantizationError:
    """Accuracy metrics for one weight quantized by :func:`onnxsim.quantize_static`,
    comparing the original float weight against ``(w_q - zero_point) * scale`` --
    the inverse of the quantize step, i.e. rematerializing the weight through its
    own quantization parameters -- computed by :func:`weight_quantization_error`.
    """

    node: str  # the quantized node's name, or its output name(s) if unnamed
    weight_name: str  # the original float weight's initializer/tensor name
    shape: Tuple[int, ...]
    axis: int  # the per-channel quantization axis
    mse: float
    max_abs_error: float
    relative_l2: float  # ||w - dequant||_2 / ||w||_2
    cosine_similarity: float
    sqnr_db: float  # 20*log10(||w||_2 / ||w - dequant||_2); +inf if exact
    enob_bits: float  # (sqnr_db - 1.76) / 6.02 -- SQNR re-expressed as "effective bits"
    psnr_db: float  # 20*log10(max|w| / rmse); the peak- rather than energy-normalized twin of sqnr_db
    histogram_js_divergence: float  # Jensen-Shannon divergence (nats) between w's and dequant's value histograms -- a *distributional* check, complementing the element-wise metrics above
    per_channel_relative_l2: List[float]  # one entry per channel along `axis`


def _resolve_constant(graph: onnx.GraphProto, name: str) -> Optional[onnx.TensorProto]:
    """The constant ``TensorProto`` backing value ``name`` -- an initializer, or a
    ``Constant`` node's ``value`` attribute -- or ``None`` if ``name`` is not
    constant (or not found).
    """
    for initializer in graph.initializer:
        if initializer.name == name:
            return initializer
    for node in graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == name:
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    return attr.t
    return None


def _broadcast_along_axis(values: np.ndarray, ndim: int, axis: int) -> np.ndarray:
    """Reshape a per-channel 1-D array (or a scalar) so it broadcasts against an
    ``ndim``-D array along ``axis`` -- turns ``DequantizeLinear``'s ``scale`` /
    ``zero_point`` (one value per channel, or a single scalar for per-tensor
    quantization) into the shape numpy needs to apply it channel-wise.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        return values
    shape = [1] * ndim
    shape[axis] = values.shape[0]
    return values.reshape(shape)


def _histogram_js_divergence(
    a: np.ndarray, b: np.ndarray, num_bins: int = 256
) -> float:
    """Jensen-Shannon divergence (nats; 0 = identical histograms, ``ln(2)`` =
    disjoint) between ``a``'s and ``b``'s value histograms over their shared
    range -- unlike the element-wise metrics in :func:`_tensor_error_metrics`
    (which compare ``a[i]`` against ``b[i]``), this compares the *shape* of the
    two distributions, so it can catch distortion (e.g. many distinct values
    collapsing onto one quantization level) that averages out of the
    element-wise metrics.

    Unlike plain KL divergence, JS needs no epsilon-smoothing for empty bins:
    the mixture ``m = (p + q) / 2`` is positive everywhere either input is, so
    ``p * log(p / m)`` never divides by zero.
    """
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 0.0  # every value (in both arrays) is identical

    p, _ = np.histogram(a, bins=num_bins, range=(lo, hi))
    q, _ = np.histogram(b, bins=num_bins, range=(lo, hi))
    p = p.astype(np.float64) / max(p.sum(), 1)
    q = q.astype(np.float64) / max(q.sum(), 1)
    m = 0.5 * (p + q)

    def _kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log(x[mask] / y[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _tensor_error_metrics(
    original: np.ndarray, dequantized: np.ndarray, axis: int
) -> Tuple[float, float, float, float, float, float, float, float, List[float]]:
    """``(mse, max_abs_error, relative_l2, cosine_similarity, sqnr_db, enob_bits,
    psnr_db, histogram_js_divergence, per_channel_relative_l2)`` between
    ``original`` and ``dequantized`` (same shape), the last computed per channel
    along ``axis``.
    """
    orig = original.astype(np.float64)
    deq = dequantized.astype(np.float64)
    diff = deq - orig

    mse = float(np.mean(diff**2)) if diff.size else 0.0
    max_abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0

    norm_orig = float(np.linalg.norm(orig))
    norm_diff = float(np.linalg.norm(diff))
    if norm_orig > 0:
        relative_l2 = norm_diff / norm_orig
        sqnr_db = (
            math.inf if norm_diff == 0 else 20.0 * math.log10(norm_orig / norm_diff)
        )
    else:
        # An all-zero original weight: any reconstruction error is undefined as a
        # *relative* quantity, so report exact-zero as perfect and anything else
        # as unbounded rather than dividing by zero.
        relative_l2 = 0.0 if norm_diff == 0 else math.inf
        sqnr_db = math.inf if norm_diff == 0 else -math.inf

    norm_deq = float(np.linalg.norm(deq))
    denom = norm_orig * norm_deq
    cosine_similarity = (
        float(np.dot(orig.ravel(), deq.ravel()) / denom) if denom > 0 else 1.0
    )

    # ENOB re-expresses SQNR in "effective bits" (the standard ADC/DSP
    # conversion); +-inf propagates through the arithmetic unchanged.
    enob_bits = (sqnr_db - 1.76) / 6.02

    peak_orig = float(np.max(np.abs(orig))) if orig.size else 0.0
    if mse == 0.0:
        psnr_db = math.inf
    elif peak_orig == 0.0:
        psnr_db = -math.inf
    else:
        psnr_db = 20.0 * math.log10(peak_orig / math.sqrt(mse))

    histogram_js_divergence = _histogram_js_divergence(orig, deq)

    moved_orig = np.moveaxis(orig, axis, 0)
    moved_deq = np.moveaxis(deq, axis, 0)
    per_channel_relative_l2 = []
    for c in range(moved_orig.shape[0]):
        c_norm_orig = float(np.linalg.norm(moved_orig[c]))
        c_norm_diff = float(np.linalg.norm(moved_deq[c] - moved_orig[c]))
        if c_norm_orig > 0:
            per_channel_relative_l2.append(c_norm_diff / c_norm_orig)
        else:
            per_channel_relative_l2.append(0.0 if c_norm_diff == 0 else math.inf)

    return (
        mse,
        max_abs_error,
        relative_l2,
        cosine_similarity,
        sqnr_db,
        enob_bits,
        psnr_db,
        histogram_js_divergence,
        per_channel_relative_l2,
    )


def weight_quantization_error(
    model_before: onnx.ModelProto, model_after: onnx.ModelProto
) -> List[WeightQuantizationError]:
    """Measure how much each weight :func:`onnxsim.quantize_static` (the QDQ,
    calibration-based pass) perturbed, by rematerializing every quantized weight
    through its ``DequantizeLinear`` -- ``w_dequant = (w_q - zero_point) * scale``,
    the inverse of the quantize step -- and diffing that reconstruction against
    the original float weight from ``model_before``.

    ``model_after`` must be the direct result of running
    :func:`onnxsim.quantize_static` on ``model_before`` (or on a simplified copy
    of it -- node *identity*, not exact byte content, is what matters here): a
    ``Conv``/``MatMul``/``Gemm`` node is matched between the two models by its
    output name, since ``quantize_static`` never renames or removes these nodes,
    only rewires their ``X``/``W`` inputs (see
    ``onnxsim/passes/static_quantize_matmul.h`` and ``static_quantize_conv.h``).
    A match is only reported when the matched node's weight input traces back to
    a ``DequantizeLinear`` whose inputs are a constant int8/uint8 tensor and a
    constant float scale -- i.e. only where ``quantize_static`` actually
    quantized that weight.

    Not applicable to :func:`onnxsim.quantize_dynamic`: that pass replaces the
    MatMul/Gemm node itself with ``MatMulInteger`` rather than dequantizing the
    weight back to float in the graph, so there is no ``DequantizeLinear`` to
    walk back through here. Its quantized weight and per-channel scale are still
    plain initializers though (the ``B`` input of the ``MatMulInteger`` node, and
    the constant operand of the ``Mul`` that combines it with the activation's
    runtime-computed scale) -- extract them by name and feed
    ``(w_q * w_scale)`` through the same reconstruction by hand if needed.

    :param model_before: the float model, before quantization
    :param model_after: ``onnxsim.quantize_static(model_before, ...)``'s result
    :returns: one :class:`WeightQuantizationError` per matched weight, in the
            order its node appears in ``model_after``
    """
    graph_before = model_before.graph
    graph_after = model_after.graph
    nodes_before = {tuple(n.output): n for n in graph_before.node if n.output}

    results: List[WeightQuantizationError] = []
    for node in graph_after.node:
        if node.op_type not in ("Conv", "MatMul", "Gemm") or len(node.input) < 2:
            continue
        before_node = nodes_before.get(tuple(node.output))
        if before_node is None or before_node.op_type != node.op_type:
            continue

        dq = next(
            (
                n
                for n in graph_after.node
                if n.op_type == "DequantizeLinear"
                and n.output
                and n.output[0] == node.input[1]
            ),
            None,
        )
        if dq is None or len(dq.input) < 2:
            continue

        wq_t = _resolve_constant(graph_after, dq.input[0])
        ws_t = _resolve_constant(graph_after, dq.input[1])
        w_orig_t = _resolve_constant(graph_before, before_node.input[1])
        if wq_t is None or ws_t is None or w_orig_t is None:
            continue
        if wq_t.data_type not in (
            onnx.TensorProto.INT8,
            onnx.TensorProto.UINT8,
        ):
            continue

        wq = numpy_helper.to_array(wq_t)
        w_orig = numpy_helper.to_array(w_orig_t)
        if wq.shape != w_orig.shape:
            continue  # unexpected shape mismatch; skip rather than guess

        axis = 1
        for attr in dq.attribute:
            if attr.name == "axis":
                axis = attr.i
        axis %= wq.ndim

        zero_point = np.array(0.0)
        if len(dq.input) >= 3 and dq.input[2]:
            zp_t = _resolve_constant(graph_after, dq.input[2])
            if zp_t is not None:
                zero_point = numpy_helper.to_array(zp_t)

        ws = numpy_helper.to_array(ws_t)
        w_dequant = (
            wq.astype(np.float64) - _broadcast_along_axis(zero_point, wq.ndim, axis)
        ) * _broadcast_along_axis(ws, wq.ndim, axis)

        (
            mse,
            max_abs_error,
            relative_l2,
            cosine_similarity,
            sqnr_db,
            enob_bits,
            psnr_db,
            histogram_js_divergence,
            per_channel_relative_l2,
        ) = _tensor_error_metrics(w_orig, w_dequant, axis)

        results.append(
            WeightQuantizationError(
                node=before_node.name or "/".join(before_node.output),
                weight_name=before_node.input[1],
                shape=tuple(w_orig.shape),
                axis=axis,
                mse=mse,
                max_abs_error=max_abs_error,
                relative_l2=relative_l2,
                cosine_similarity=cosine_similarity,
                sqnr_db=sqnr_db,
                enob_bits=enob_bits,
                psnr_db=psnr_db,
                histogram_js_divergence=histogram_js_divergence,
                per_channel_relative_l2=per_channel_relative_l2,
            )
        )
    return results


def print_weight_quantization_error(
    results: List[WeightQuantizationError], limit: int = 50
) -> None:
    """Pretty-print :func:`weight_quantization_error`'s results as a table,
    worst (highest relative L2 error) weight first, capped at ``limit`` rows (with
    a "... and N more" line) so a large model's report stays readable.
    """

    def _fmt(value: float) -> str:
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.1f}"

    table = Table()
    table.add_column("Node")
    table.add_column("Weight")
    table.add_column("Shape")
    table.add_column("MSE")
    table.add_column("Max |Δ|")
    table.add_column("Relative L2")
    table.add_column("Cosine Sim.")
    table.add_column("SQNR (dB)")
    table.add_column("ENOB (bits)")
    table.add_column("PSNR (dB)")
    table.add_column("JS Div.")

    ordered = sorted(results, key=lambda r: r.relative_l2, reverse=True)
    for r in ordered[:limit]:
        table.add_row(
            r.node,
            r.weight_name,
            "x".join(str(d) for d in r.shape),
            f"{r.mse:.3e}",
            f"{r.max_abs_error:.3e}",
            f"{r.relative_l2:.4f}",
            f"{r.cosine_similarity:.6f}",
            _fmt(r.sqnr_db),
            _fmt(r.enob_bits),
            _fmt(r.psnr_db),
            f"{r.histogram_js_divergence:.4f}",
        )
    print(table)
    if len(ordered) > limit:
        print(Text(f"... and {len(ordered) - limit} more", style="dim"))
