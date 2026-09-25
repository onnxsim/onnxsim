"""Convert a (simplified) ONNX model to Core ML.

onnxsim's job stops at a cleaned-up ``onnx.ModelProto``. Apple platforms want that
graph as a Core ML model (``.mlpackage``) instead. This module bridges the two by
walking the ONNX graph node by node and building the equivalent `MIL
<https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html>`_
(Model Intermediate Language) program with `coremltools
<https://github.com/apple/coremltools>`_'s own builder, then handing that program to
``coremltools.convert(..., source="milinternal")`` to produce the actual Core ML model.

coremltools itself dropped its ONNX frontend in version 7 (it only converts
TensorFlow/PyTorch models, or an in-memory MIL program), so there is no off-the-shelf
"convert this ONNX model" call to lean on -- this translator plays that role, one ONNX
op at a time. It covers a practical subset of ops (common to CNN/MLP/transformer-block
graphs: conv, pooling, normalization, matmul/gemm, elementwise math, reshapes,
reductions, ...); a node whose op isn't in ``SUPPORTED_ONNX_OPS`` raises a
``RuntimeError`` naming the op, rather than silently producing a wrong model.

Feeding a *simplified* model in is the point, same as with ``mlir_export.py``: onnxsim's
constant folding turns shape-manipulation subgraphs and foldable weight arithmetic into
plain initializers, so more of the graph lands on this translator's supported-op list
and the emitted Core ML model is smaller.

Like onnxruntime for constant folding and torch-mlir/onnx-mlir for MLIR (see
``backend.py`` / ``mlir_export.py``), coremltools is an **optional** dependency: nothing
here is imported at ``import onnxsim`` time, only when ``--emit-coreml`` / the
``export_coreml`` API run. A missing coremltools raises a ``RuntimeError`` with an
install hint.

Conversion itself needs no macOS-specific functionality (MIL construction and
``.mlpackage`` serialization are pure Python/protobuf), so it runs the same on Linux,
macOS, or Windows. Only *loading the produced model back for a prediction* needs Core
ML's runtime, i.e. an Apple OS -- this module defaults to ``skip_model_load=True`` so
conversion still succeeds off of macOS; pass ``skip_model_load=False`` on macOS to get a
model that's ready to run.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import numpy_helper

_COREML_INSTALL_HINT = (
    "coremltools is required to export Core ML models but is not installed. "
    "Install it with `pip install coremltools`."
)

# ONNX TensorProto dtypes this translator can carry through to Core ML, downcasting
# where Core ML/MIL has no matching type (float64 -> float32, int64 -> int32; Core ML
# arrays have no 64-bit numeric type).
_NP_DOWNCAST = {
    np.dtype(np.float64): np.float32,
    np.dtype(np.int64): np.int32,
}
_SUPPORTED_NP_DTYPES = {np.float32, np.float16, np.int32, np.bool_, np.int8, np.uint8}


def has_coremltools() -> bool:
    """Whether coremltools is importable in this environment."""
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


def _import_coremltools():
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError(_COREML_INSTALL_HINT) from exc
    return ct


def _import_mil():
    try:
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.input_types import RangeDim, TensorType
        from coremltools.converters.mil.mil import Function, Program, types
    except ImportError as exc:
        raise RuntimeError(_COREML_INSTALL_HINT) from exc
    return mb, types, Function, Program, RangeDim, TensorType


def _as_mil_array(arr: np.ndarray) -> np.ndarray:
    """Downcast ``arr`` to a dtype Core ML/MIL supports, or raise."""
    arr = np.asarray(arr)
    if arr.dtype == np.int64:
        # Saturate rather than wrap: ONNX graphs routinely use INT64_MAX/MIN as
        # "unbounded" sentinels (e.g. a Slice `ends` meaning "to the end of this
        # axis"), and a plain .astype(int32) wraps those around to -1/0 instead of
        # a large in-range number, silently corrupting the sentinel.
        arr = np.clip(arr, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    target = _NP_DOWNCAST.get(arr.dtype)
    if target is not None:
        arr = arr.astype(target)
    if arr.dtype.type not in _SUPPORTED_NP_DTYPES:
        raise RuntimeError(
            f"Unsupported tensor dtype {arr.dtype} for Core ML export (supported: "
            "float16, float32, float64, int32, int64, bool, int8, uint8; 64-bit "
            "types are downcast to their 32-bit equivalent; int8/uint8 are "
            "constants only, e.g. a quantization zero point)."
        )
    return arr


def _onnx_elem_type_to_mil(elem_type: int, types) -> Any:
    TP = onnx.TensorProto
    mapping = {
        TP.FLOAT: types.fp32,
        TP.FLOAT16: types.fp16,
        TP.DOUBLE: types.fp32,
        TP.INT32: types.int32,
        TP.INT64: types.int32,
        TP.BOOL: types.bool,
    }
    if elem_type not in mapping:
        raise RuntimeError(
            f"Unsupported input dtype {TP.DataType.Name(elem_type)}; onnxsim's Core "
            "ML exporter supports float16, float32, float64, int32, int64, and bool "
            "graph inputs (64-bit types are represented as their 32-bit equivalent "
            "in the exported model)."
        )
    return mapping[elem_type]


def _make_input_spec(
    value_info: onnx.ValueInfoProto,
    mb,
    types,
    dynamic_shapes: Dict[str, Tuple[int, int, int]],
    range_dims: Dict[str, Any],
    RangeDim,
    TensorType,
    io_dtype: Optional[str] = None,
):
    """Build this input's MIL ``TensorSpec`` and, if it has a dynamic dim, the
    matching coremltools ``TensorType`` (for ``ct.convert(inputs=...)``; ``None``
    otherwise).

    Each *named* dynamic dimension (an ONNX ``dim_param``) present as a key in
    ``dynamic_shapes`` gets one shared ``RangeDim`` -- reused by name across every
    input that has it (via ``range_dims``), so e.g. all of a KV cache's
    ``past_key_values.*.key``/``value`` inputs vary together. A ``dim_param`` not
    in ``dynamic_shapes`` is still rejected as non-static: this is opt-in, not a
    blanket "allow anything".
    """
    tt = value_info.type.tensor_type
    mil_shape: List[Any] = []
    ct_shape: List[Any] = []
    has_dynamic = False
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            mil_shape.append(int(d.dim_value))
            ct_shape.append(int(d.dim_value))
            continue
        name = d.dim_param
        bounds = dynamic_shapes.get(name)
        if bounds is None:
            raise RuntimeError(
                f"Input '{value_info.name}' has a non-static dimension "
                f"({name or '?'!r}); onnxsim's Core ML exporter requires fully "
                "static input shapes by default. Give the model concrete input "
                "shapes first (e.g. via --overwrite-input-shape or "
                "onnx.tools.update_model_dims), or mark this dimension dynamic "
                "via the `dynamic_shapes` argument, e.g. "
                f"dynamic_shapes={{{name!r}: (lower_bound, default, upper_bound)}}."
            )
        if name not in range_dims:
            lower, default, upper = bounds
            range_dims[name] = RangeDim(
                lower_bound=lower, upper_bound=upper, default=default, symbol=name
            )
        rd = range_dims[name]
        mil_shape.append(rd.symbol)
        ct_shape.append(rd)
        has_dynamic = True
    dtype = _onnx_elem_type_to_mil(tt.elem_type, types)
    if io_dtype == "fp16" and dtype == types.fp32:
        # Declare the *model boundary* as fp16 (see convert_to_coreml's `io_dtype`).
        # Only float inputs move: an int/bool input has no fp16 equivalent, and
        # Core ML would reject one anyway.
        dtype = types.fp16
    spec = mb.TensorSpec(shape=tuple(mil_shape), dtype=dtype)
    ct_input = (
        TensorType(name=value_info.name, shape=tuple(ct_shape)) if has_dynamic else None
    )
    return spec, ct_input


def _node_attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        if isinstance(v, onnx.TensorProto):
            v = numpy_helper.to_array(v)
        elif isinstance(v, bytes):
            v = v.decode("utf-8")
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], bytes):
            v = [x.decode("utf-8") for x in v]
        attrs[a.name] = v
    return attrs


def _pad_type_and_pads(attrs: Dict[str, Any], n: int):
    """Map ONNX ``auto_pad``/``pads`` to a MIL ``(pad_type, pad)`` pair.

    ``pad`` is ``None`` when ``pad_type`` alone determines the padding (``"valid"``,
    ``"same"``, ``"same_lower"``); otherwise it is MIL's interleaved
    ``[before_0, after_0, before_1, after_1, ...]`` form, converted from ONNX's
    ``[before_0, ..., before_{n-1}, after_0, ..., after_{n-1}]`` form.
    """
    auto_pad = attrs.get("auto_pad", "NOTSET")
    if auto_pad == "SAME_UPPER":
        return "same", None
    if auto_pad == "SAME_LOWER":
        return "same_lower", None
    if auto_pad == "VALID":
        return "valid", None
    pads = attrs.get("pads")
    if not pads or not any(pads):
        return "valid", None
    begins, ends = pads[:n], pads[n:]
    interleaved: List[int] = []
    for b, e in zip(begins, ends):
        interleaved += [int(b), int(e)]
    return "custom", interleaved


def _const_scalar(var) -> float:
    if var.val is None:
        raise RuntimeError(
            "expected a compile-time constant here, but the tensor is only known at "
            "runtime (onnxsim's Core ML exporter can't trace dynamic values)"
        )
    return float(np.asarray(var.val).reshape(-1)[0])


class _Lowerer:
    """Walks an ONNX graph once, building the equivalent MIL ops as it goes."""

    def __init__(self, mb, types, opset: int, matmul_to_conv: bool = False):
        self.mb = mb
        self.types = types
        self.opset = opset
        self.matmul_to_conv = matmul_to_conv
        self._values: Dict[str, Any] = {}
        self._counter = 0

    def bind(self, name: str, var) -> None:
        self._values[name] = var

    def get(self, name: str):
        if name not in self._values:
            raise RuntimeError(
                f"reference to unknown tensor '{name}' (the graph may not be "
                "topologically sorted, or the producing node uses an unsupported "
                "feature)"
            )
        return self._values[name]

    def make_const(self, name: str, arr: np.ndarray):
        self._counter += 1
        return self.mb.const(
            val=_as_mil_array(arr), name=f"{name}__const{self._counter}"
        )

    def fresh_name(self, node: onnx.NodeProto, suffix: Optional[str] = None) -> str:
        self._counter += 1
        base = (
            node.output[0]
            if node.output and node.output[0]
            else (node.name or node.op_type)
        )
        if suffix:
            base = f"{base}_{suffix}"
        return f"{base}__{self._counter}"

    def fresh_io_name(self, base: str, suffix: str) -> str:
        """A unique name for a cast inserted at the graph's I/O boundary, where
        there is no ONNX node to derive one from (see ``convert_to_coreml``'s
        ``io_dtype``). Shares ``_counter`` with ``fresh_name``/``make_const`` so a
        graph tensor literally named ``x_from_fp16__1`` still can't collide."""
        self._counter += 1
        return f"{base}_{suffix}__{self._counter}"

    def lower_node(self, node: onnx.NodeProto) -> None:
        handler = _OP_HANDLERS.get(node.op_type)
        if handler is None:
            raise RuntimeError(
                f"ONNX op '{node.op_type}' is not supported by onnxsim's Core ML "
                f"exporter (node {node.name or node.output[0]!r}). Supported ops: "
                + ", ".join(sorted(_OP_HANDLERS))
            )
        ins = [self.get(name) if name else None for name in node.input]
        attrs = _node_attrs(node)
        try:
            outs = handler(self, node, ins, attrs)
        except Exception as exc:
            raise RuntimeError(
                f"failed to convert ONNX node {node.name or node.output[0]!r} "
                f"({node.op_type}) to Core ML: {exc}"
            ) from exc
        for out_name, var in zip(node.output, outs):
            if out_name:
                self._values[out_name] = var


# ---------------------------------------------------------------------------
# Op handlers. Each handler is ``(lowerer, node, ins, attrs) -> list[Var]``, with
# ``ins[i]`` the already-converted MIL value for ``node.input[i]`` (``None`` for an
# omitted optional input) and ``attrs`` the node's parsed attribute dict.
# ---------------------------------------------------------------------------

_OP_HANDLERS: Dict[str, Callable] = {}


def _register(*op_types: str):
    def deco(fn):
        for op_type in op_types:
            _OP_HANDLERS[op_type] = fn
        return fn

    return deco


def _simple_unary(mil_name: str):
    def handler(lowerer, node, ins, attrs):
        fn = getattr(lowerer.mb, mil_name)
        return [fn(x=ins[0], name=lowerer.fresh_name(node))]

    return handler


def _simple_binary(mil_name: str):
    def handler(lowerer, node, ins, attrs):
        fn = getattr(lowerer.mb, mil_name)
        return [fn(x=ins[0], y=ins[1], name=lowerer.fresh_name(node))]

    return handler


for _onnx_op, _mil_op in [
    ("Relu", "relu"),
    ("Sigmoid", "sigmoid"),
    ("Tanh", "tanh"),
    ("Exp", "exp"),
    ("Log", "log"),
    ("Sqrt", "sqrt"),
    ("Erf", "erf"),
    ("Identity", "identity"),
    ("Sin", "sin"),
    ("Cos", "cos"),
    ("Abs", "abs"),
    ("Floor", "floor"),
    ("Round", "round"),
]:
    _OP_HANDLERS[_onnx_op] = _simple_unary(_mil_op)

for _onnx_op, _mil_op in [
    ("Add", "add"),
    ("Sub", "sub"),
    ("Mul", "mul"),
    ("Div", "real_div"),
    ("Pow", "pow"),
    ("Equal", "equal"),
    ("LessOrEqual", "less_equal"),
    ("GreaterOrEqual", "greater_equal"),
    ("Greater", "greater"),
    ("And", "logical_and"),
]:
    _OP_HANDLERS[_onnx_op] = _simple_binary(_mil_op)


@_register("Not")
def _lower_not(lowerer, node, ins, attrs):
    return [lowerer.mb.logical_not(x=ins[0], name=lowerer.fresh_name(node))]


def _resize_nearest_indices(
    input_size: int,
    output_size: int,
    scale: float,
    coordinate_mode: str,
    nearest_mode: str,
) -> np.ndarray:
    """The source index each output position samples, for one spatial axis.

    Shared by the gather expansion below and the ``resize_nearest_neighbor``
    eligibility check, so both describe the same sampling rule.
    """
    output = np.arange(output_size, dtype=np.float64)
    if coordinate_mode == "asymmetric":
        source = output / scale
    elif coordinate_mode == "half_pixel":
        source = (output + 0.5) / scale - 0.5
    else:
        raise RuntimeError(
            f"Resize nearest coordinate_transformation_mode {coordinate_mode!r} "
            f"is not supported"
        )
    if nearest_mode == "floor":
        indices = np.floor(source)
    elif nearest_mode == "ceil":
        indices = np.ceil(source)
    elif nearest_mode == "round_prefer_floor":
        indices = np.ceil(source - 0.5)
    elif nearest_mode == "round_prefer_ceil":
        indices = np.floor(source + 0.5)
    else:
        raise RuntimeError(f"Resize nearest_mode {nearest_mode!r} is not supported")
    return np.clip(indices, 0, input_size - 1).astype(np.int32)


def _resize_nearest_is_repeating(input_size: int, indices: np.ndarray) -> bool:
    """Whether this nearest Resize is a pure integer replication of the input.

    ``resize_nearest_neighbor`` is a real Core ML kernel rather than a pair of
    index gathers materializing an intermediate feature map, but it implements
    its own fixed sampling rule, and `scripts/apple/README.md`'s Resize note
    records why these were gathers in the first place (a Core ML resize runtime
    limitation). So rather than assume the two rules agree, this requires the
    strongest case where agreement is provable and was measured on Core ML: every
    input position repeated a whole number of times (``2x2 -> 4x4``,
    ``3x4 -> 6x8``, ``4x4 -> 8x8`` all match ONNX Runtime exactly).

    Non-integer factors are deliberately excluded rather than compared against an
    assumed formula: measured on Core ML, a downscale (``4x4 -> 2x2``) or a
    fractional upscale (``2x2 -> 3x3``) samples *different* rows than ONNX does.
    Those keep the gather path, so the documented "preserves ONNX's sampling
    rule" property cannot regress.
    """
    output_size = int(indices.size)
    if output_size % input_size or output_size < input_size:
        return False
    repeats = output_size // input_size
    expected = np.repeat(np.arange(input_size, dtype=np.int32), repeats)
    return bool(np.array_equal(indices, expected))


@_register("Resize")
def _lower_resize(lowerer, node, ins, attrs):
    """Lower static 2-D ONNX Resize ops used by CNN feature pyramids."""
    if len(ins[0].shape) != 4:
        raise RuntimeError(
            f"Resize expects NCHW rank 4 input, got rank {len(ins[0].shape)}"
        )
    mode = attrs.get("mode", b"nearest")
    if isinstance(mode, bytes):
        mode = mode.decode()
    coordinate_mode = attrs.get("coordinate_transformation_mode", b"half_pixel")
    if isinstance(coordinate_mode, bytes):
        coordinate_mode = coordinate_mode.decode()
    sampling_modes = {
        "half_pixel": "UNALIGN_CORNERS",
        "align_corners": "ALIGN_CORNERS",
    }
    height, width = ins[0].shape[-2:]
    if not all(isinstance(d, (int, np.integer)) for d in (height, width)):
        raise RuntimeError("Resize requires static spatial dimensions")

    sizes_in = ins[3] if len(ins) > 3 and ins[3] is not None else None
    scales_in = ins[2] if len(ins) > 2 and ins[2] is not None else None
    if sizes_in is not None:
        if sizes_in.val is None:
            raise RuntimeError("Resize requires a compile-time-constant 'sizes' input")
        sizes = np.asarray(sizes_in.val).reshape(-1)
        if sizes.size != 4:
            raise RuntimeError(f"Resize 'sizes' must have 4 entries, got {sizes.size}")
        target_size = [int(v) for v in sizes]
        input_shape = tuple(ins[0].shape)
        if any(
            isinstance(d, (int, np.integer)) and target_size[axis] != int(d)
            for axis, d in enumerate(input_shape[:2])
        ):
            raise RuntimeError(
                "Resize only supports unchanged N/C dimensions, got input shape "
                f"{input_shape} and sizes {sizes.tolist()}"
            )
        target_height, target_width = target_size[-2:]
        if target_height <= 0 or target_width <= 0:
            raise RuntimeError(
                f"Resize output sizes must be positive, got {sizes.tolist()}"
            )
        scales = np.array(
            [
                1.0,
                1.0,
                target_height / float(height),
                target_width / float(width),
            ],
            dtype=np.float64,
        )
    elif scales_in is not None:
        if scales_in.val is None:
            raise RuntimeError("Resize requires a compile-time-constant 'scales' input")
        scales = np.asarray(scales_in.val).reshape(-1)
        if scales.size != 4:
            raise RuntimeError(
                f"Resize 'scales' must have 4 entries, got {scales.size}"
            )
        if not np.all(scales[:2] == 1):
            raise RuntimeError(
                f"Resize only supports N/C scale 1, got scales {scales.tolist()}"
            )
        if scales[-2] <= 0 or scales[-1] <= 0:
            raise RuntimeError(
                f"Resize spatial scales must be positive, got {scales.tolist()}"
            )
        target_height = int(round(int(height) * float(scales[-2])))
        target_width = int(round(int(width) * float(scales[-1])))
    else:
        raise RuntimeError(
            "Resize needs a compile-time-constant 'sizes' or 'scales' input"
        )

    resize_name = lowerer.fresh_name(node)
    if mode == "nearest":
        nearest_mode = attrs.get("nearest_mode", b"round_prefer_floor")
        if isinstance(nearest_mode, bytes):
            nearest_mode = nearest_mode.decode()

        height_indices = _resize_nearest_indices(
            int(height), target_height, scales[-2], coordinate_mode, nearest_mode
        )
        width_indices = _resize_nearest_indices(
            int(width), target_width, scales[-1], coordinate_mode, nearest_mode
        )
        if _resize_nearest_is_repeating(
            int(height), height_indices
        ) and _resize_nearest_is_repeating(int(width), width_indices):
            return [
                lowerer.mb.resize_nearest_neighbor(
                    x=ins[0],
                    target_size_height=target_height,
                    target_size_width=target_width,
                    name=resize_name,
                )
            ]
        height_index_var = lowerer.make_const(
            resize_name + "_height_indices", height_indices
        )
        width_index_var = lowerer.make_const(
            resize_name + "_width_indices", width_indices
        )
        resized_height = lowerer.mb.gather(
            x=ins[0],
            indices=height_index_var,
            axis=2,
            name=lowerer.fresh_name(node, "nearest_height"),
        )
        return [
            lowerer.mb.gather(
                x=resized_height,
                indices=width_index_var,
                axis=3,
                name=resize_name,
            )
        ]
    if mode != "linear":
        raise RuntimeError(f"Resize mode {mode!r} is not supported")
    if coordinate_mode not in sampling_modes:
        raise RuntimeError(
            f"Resize coordinate_transformation_mode {coordinate_mode!r} is not supported"
        )
    return [
        lowerer.mb.resize_bilinear(
            x=ins[0],
            target_size_height=target_height,
            target_size_width=target_width,
            sampling_mode=sampling_modes[coordinate_mode],
            name=resize_name,
        )
    ]


@_register("DepthToSpace")
def _lower_depth_to_space(lowerer, node, ins, attrs):
    """Lower ONNX NCHW DepthToSpace as a reshape and axis permutation."""
    shape = ins[0].shape
    if len(shape) != 4 or not all(isinstance(d, (int, np.integer)) for d in shape):
        raise RuntimeError("DepthToSpace requires a static rank-4 input")
    n, channels, height, width = (int(d) for d in shape)
    block = int(attrs.get("blocksize", 0))
    if block <= 0 or channels % (block * block):
        raise RuntimeError(f"invalid DepthToSpace blocksize {block} for shape {shape}")
    out_channels = channels // (block * block)
    mode = attrs.get("mode", b"DCR")
    if isinstance(mode, bytes):
        mode = mode.decode()
    if mode == "DCR":
        reshaped = lowerer.mb.reshape(
            x=ins[0],
            shape=[n, block, block, out_channels, height, width],
            name=lowerer.fresh_name(node, "reshape"),
        )
        perm = [0, 3, 4, 1, 5, 2]
    elif mode == "CRD":
        reshaped = lowerer.mb.reshape(
            x=ins[0],
            shape=[n, out_channels, block, block, height, width],
            name=lowerer.fresh_name(node, "reshape"),
        )
        perm = [0, 1, 4, 2, 5, 3]
    else:
        raise RuntimeError(f"DepthToSpace mode {mode!r} is not supported")
    transposed = lowerer.mb.transpose(
        x=reshaped, perm=perm, name=lowerer.fresh_name(node, "transpose")
    )
    return [
        lowerer.mb.reshape(
            x=transposed,
            shape=[n, out_channels, height * block, width * block],
            name=lowerer.fresh_name(node),
        )
    ]


def _mil_dim_equal(da, db) -> bool:
    """Whether two MIL shape dims are provably the same extent."""
    if isinstance(da, (int, np.integer)) and isinstance(db, (int, np.integer)):
        return int(da) == int(db)
    # Symbolic/dynamic dims: only ever produced from this module's own
    # `dynamic_shapes` RangeDims, so two non-int dims are treated as the same
    # extent (spare broadcasts would only add no-op fill/add pairs, but the
    # leaner graph is easier to read back in a model dump).
    return not isinstance(da, (int, np.integer)) and not isinstance(
        db, (int, np.integer)
    )


def _mil_shapes_match(a_shape, b_shape) -> bool:
    """Whether two MIL shapes are provably identical (same rank, every dim
    equal per :func:`_mil_dim_equal`)."""
    return len(a_shape) == len(b_shape) and all(
        _mil_dim_equal(da, db) for da, db in zip(a_shape, b_shape)
    )


def _broadcast_to_shape(lowerer, node, x, shape_var, suffix: str):
    """Broadcast MIL var ``x`` up to ``shape_var``'s shape (the 1-D runtime
    shape vector, e.g. from ``mb.shape``), via the same fill/add idiom the
    dynamic ``Expand`` path uses: ``fill`` accepts a runtime (non-constant)
    shape, and adding a same-shaped zero tensor forces ``x`` to broadcast up
    to it, exactly following numpy/ONNX right-aligned broadcast rules. A
    1-element constant instead becomes a scalar ``fill`` directly (no add)."""
    is_bool = x.dtype == lowerer.types.bool
    if is_bool:
        x = lowerer.mb.cast(
            x=x, dtype="int32", name=lowerer.fresh_name(node, suffix + "_boolcast")
        )
    np_dtype = lowerer.types.nptype_from_builtin(x.dtype)
    if x.val is not None and np.asarray(x.val).size == 1:
        out = lowerer.mb.fill(
            shape=shape_var,
            value=np_dtype(np.asarray(x.val).reshape(-1)[0]),
            name=lowerer.fresh_name(node, suffix + "_fill"),
        )
    else:
        zeros = lowerer.mb.fill(
            shape=shape_var,
            value=np_dtype(0),
            name=lowerer.fresh_name(node, suffix + "_zeros"),
        )
        out = lowerer.mb.add(
            x=x, y=zeros, name=lowerer.fresh_name(node, suffix + "_bcast")
        )
    if is_bool:
        out = lowerer.mb.cast(
            x=out, dtype="bool", name=lowerer.fresh_name(node, suffix)
        )
    return out


def _qdq_scale_and_zp(lowerer, node, ins, attrs, op_name: str):
    """Shared (scale, zero_point, axis) handling for the QDQ pair.

    Both MIL `quantize`/`dequantize` need compile-time-constant scale (and
    zero point); a runtime-computed scale (e.g. re-estimated per step) has no
    lowering here. Returns `(scale_var, zero_point_var_or_None, axis_or_None)`
    with the scale dtype-matched to the input exactly the way the dynamic
    `Expand` path matches its `fill` value (a bare Python/numpy scalar would
    land on MIL's default dtype instead).
    """
    scale = ins[1]
    if scale.val is None:
        raise RuntimeError(f"{op_name} needs a compile-time-constant 'scale' input")
    zp = ins[2] if len(ins) > 2 and ins[2] is not None else None
    if zp is not None and zp.val is None:
        raise RuntimeError(
            f"{op_name} needs a compile-time-constant 'zero_point' input"
        )
    axis = int(attrs.get("axis", 1))
    return scale, zp, axis


@_register("QuantizeLinear")
def _op_quantize_linear(lowerer, node, ins, attrs):
    x = ins[0]
    scale, zp, axis = _qdq_scale_and_zp(lowerer, node, ins, attrs, "QuantizeLinear")
    np_dtype = lowerer.types.nptype_from_builtin(x.dtype)
    if np.asarray(scale.val).dtype != np_dtype:
        scale = lowerer.make_const(
            f"{node.output[0]}_scale",
            np.asarray(scale.val).astype(np_dtype),
        )
    if zp is None:
        output_dtype = "int8"
        zp_arg: Dict[str, Any] = {}
    else:
        zp_np_dtype = np.asarray(zp.val).dtype
        if zp_np_dtype == np.dtype(np.int8):
            output_dtype = "int8"
        elif zp_np_dtype == np.dtype(np.uint8):
            output_dtype = "uint8"
        else:
            raise RuntimeError(
                f"QuantizeLinear 'zero_point' must be int8 or uint8 (got {zp_np_dtype})"
            )
        zp_arg = {"zero_point": zp}
    kwargs: Dict[str, Any] = {
        "input": x,
        "scale": scale,
        "output_dtype": output_dtype,
        "name": lowerer.fresh_name(node),
    }
    kwargs.update(zp_arg)
    if np.asarray(scale.val).ndim > 0:
        kwargs["axis"] = axis
    return [lowerer.mb.quantize(**kwargs)]


@_register("DequantizeLinear")
def _op_dequantize_linear(lowerer, node, ins, attrs):
    x = ins[0]
    scale, zp, axis = _qdq_scale_and_zp(lowerer, node, ins, attrs, "DequantizeLinear")
    if zp is not None:
        zp_np_dtype = np.asarray(zp.val).dtype
        # A QDQ Conv/Gemm bias is int32 with its own per-channel scale, which
        # MIL's `dequantize` does not accept (it only takes int8/uint8 input).
        # Dequantizing an int32 zero-point bias is exactly a cast plus a
        # scale-and-zero-point multiply, so lower it that way and let the
        # broadcast happen in `mul`: the scale is 1-D along `axis` (Conv's
        # output channels), and a bias is applied to the conv output, so
        # broadcasting the (1-D scale, 1-D bias) pair over the NCHW output
        # is what ONNX's own per-channel DequantizeLinear asks for.
        if zp_np_dtype == np.dtype(np.int32):
            zp_arr = np.asarray(zp.val)
            cast = lowerer.mb.cast(
                x=x,
                dtype=lowerer.types.builtin_to_string(scale.dtype),
                name=lowerer.fresh_name(node, "cast"),
            )
            scaled = lowerer.mb.mul(
                x=cast, y=scale, name=lowerer.fresh_name(node, "scale")
            )
            if np.all(zp_arr == 0):
                return [scaled]
            zp_f = lowerer.make_const(
                f"{node.output[0]}_zp_f32", zp_arr.astype(np.float32)
            )
            return [
                lowerer.mb.add(
                    x=scaled,
                    y=zp_f,
                    name=lowerer.fresh_name(node),
                )
            ]
        if zp_np_dtype not in (np.dtype(np.int8), np.dtype(np.uint8)):
            raise RuntimeError(
                "DequantizeLinear 'zero_point' must be int8, uint8 or int32 "
                f"(got {zp_np_dtype})"
            )
    kwargs = {
        "input": x,
        "scale": scale,
        "name": lowerer.fresh_name(node),
    }
    if zp is not None:
        kwargs["zero_point"] = zp
    if np.asarray(scale.val).ndim > 0:
        kwargs["axis"] = axis
    return [lowerer.mb.dequantize(**kwargs)]


@_register("Where")
def _op_where(lowerer, node, ins, attrs):
    cond, a, b = ins
    # MIL `select` needs all three inputs at the same shape: coremltools
    # accepts a broadcastable triple at conversion time, but E5RT's ANE shape
    # propagation rejects it ("Failed to PropagateInputTensorShapes ...
    # for select: Incompatible Shape"), failing the whole model under
    # CPU_AND_NE/ALL while CPU_ONLY runs fine (seen on a dynamic-shape
    # transformer decoder whose attention mask does
    # `Where(mask, const[1], scores[1,heads,S,S])`). Explicitly broadcast the
    # deficient inputs first, using a max-rank input as the shape source.
    shapes = [tuple(v.shape) for v in (cond, a, b)]
    if not (
        _mil_shapes_match(shapes[0], shapes[1])
        and _mil_shapes_match(shapes[0], shapes[2])
    ):
        ranked = sorted((cond, a, b), key=lambda v: len(v.shape))
        ref = ranked[-1]
        ref_shape = tuple(ref.shape)
        shape_var = lowerer.mb.shape(
            x=ref, name=lowerer.fresh_name(node, "bcast_shape")
        )
        if not _mil_shapes_match(tuple(cond.shape), ref_shape):
            cond = _broadcast_to_shape(lowerer, node, cond, shape_var, "cond")
        if not _mil_shapes_match(tuple(a.shape), ref_shape):
            a = _broadcast_to_shape(lowerer, node, a, shape_var, "a")
        if not _mil_shapes_match(tuple(b.shape), ref_shape):
            b = _broadcast_to_shape(lowerer, node, b, shape_var, "b")
    return [lowerer.mb.select(cond=cond, a=a, b=b, name=lowerer.fresh_name(node))]


@_register("IsNaN")
def _op_isnan(lowerer, node, ins, attrs):
    x = ins[0]
    return [lowerer.mb.not_equal(x=x, y=x, name=lowerer.fresh_name(node))]


@_register("Expand")
def _op_expand(lowerer, node, ins, attrs):
    x = ins[0]
    shape_var = ins[1]
    if shape_var.val is not None:
        target = [int(v) for v in shape_var.val]
        x_shape = list(x.shape)
        aligned_target = (
            [1] * (len(x_shape) - len(target)) + target
            if len(target) < len(x_shape)
            else target
        )
        if len(target) > len(x_shape):
            x_shape = [1] * (len(target) - len(x_shape)) + x_shape
            x = lowerer.mb.reshape(
                x=x, shape=x_shape, name=lowerer.fresh_name(node, "reshape")
            )
            aligned_target = target
        if any(not isinstance(s, (int, np.integer)) or s < 1 for s in x_shape):
            raise RuntimeError("Expand requires static positive input dimensions")
        if any(
            s not in (1, t) and t not in (1, s) for s, t in zip(x_shape, aligned_target)
        ):
            raise RuntimeError(
                f"Expand shapes are not broadcast-compatible: {x_shape} -> {target}"
            )
        output_shape = [max(s, t) for s, t in zip(x_shape, aligned_target)]
        reps = [o // s for s, o in zip(x_shape, output_shape)]
        return [lowerer.mb.tile(x=x, reps=reps, name=lowerer.fresh_name(node))]

    # The target shape is itself only known at runtime (e.g. it depends on a KV
    # cache's dynamic length) -- `tile` needs concrete integer repeat counts, so
    # there's no way to pick those at conversion time. Broadcast-add a same-shaped
    # zero tensor instead: MIL's `fill` accepts a non-constant shape, and adding a
    # tensor of the target shape forces `x` to broadcast up to it.
    is_bool = x.dtype == lowerer.types.bool
    if is_bool:
        x = lowerer.mb.cast(
            x=x, dtype="int32", name=lowerer.fresh_name(node, "boolcast")
        )
    # `fill`'s output dtype follows its `value`'s Python type (`int` always
    # becomes MIL's int32, a bare `float` always becomes fp32 regardless of
    # context); match `x`'s (post-bool-cast) dtype explicitly via a numpy
    # scalar so the `add` below doesn't reject the two operands as mismatched
    # types -- e.g. an fp16 model's `x` next to fp32 `zeros`.
    np_dtype = lowerer.types.nptype_from_builtin(x.dtype)
    fill_value = np_dtype(0)
    zeros = lowerer.mb.fill(
        shape=shape_var, value=fill_value, name=lowerer.fresh_name(node, "zeros")
    )
    out = lowerer.mb.add(
        x=x, y=zeros, name=lowerer.fresh_name(node, "expand" if is_bool else None)
    )
    if is_bool:
        out = lowerer.mb.cast(x=out, dtype="bool", name=lowerer.fresh_name(node))
    return [out]


@_register("Neg")
def _op_neg(lowerer, node, ins, attrs):
    np_dtype = lowerer.types.nptype_from_builtin(ins[0].dtype)
    return [lowerer.mb.mul(x=ins[0], y=np_dtype(-1.0), name=lowerer.fresh_name(node))]


@_register("LeakyRelu")
def _op_leaky_relu(lowerer, node, ins, attrs):
    alpha = float(attrs.get("alpha", 0.01))
    return [lowerer.mb.leaky_relu(x=ins[0], alpha=alpha, name=lowerer.fresh_name(node))]


@_register("Gelu")
def _op_gelu(lowerer, node, ins, attrs):
    approximate = attrs.get("approximate", "none")
    mode = "TANH_APPROXIMATION" if approximate == "tanh" else "EXACT"
    return [lowerer.mb.gelu(x=ins[0], mode=mode, name=lowerer.fresh_name(node))]


@_register("Clip")
def _op_clip(lowerer, node, ins, attrs):
    lo = attrs.get("min")
    hi = attrs.get("max")
    if len(ins) > 1 and ins[1] is not None:
        lo = _const_scalar(ins[1])
    if len(ins) > 2 and ins[2] is not None:
        hi = _const_scalar(ins[2])
    lo = float(lo) if lo is not None else float(np.finfo(np.float32).min)
    hi = float(hi) if hi is not None else float(np.finfo(np.float32).max)
    return [lowerer.mb.clip(x=ins[0], alpha=lo, beta=hi, name=lowerer.fresh_name(node))]


@_register("Softmax")
def _op_softmax(lowerer, node, ins, attrs):
    x = ins[0]
    default_axis = -1 if lowerer.opset >= 13 else 1
    axis = attrs.get("axis", default_axis)
    axis = axis if axis >= 0 else axis + x.rank
    if lowerer.opset >= 13:
        return [lowerer.mb.softmax(x=x, axis=axis, name=lowerer.fresh_name(node))]
    # Pre-opset-13 Softmax coerces x into 2-D at `axis` (flattening the leading and
    # trailing dimensions together) and applies softmax over the trailing axis, unlike
    # MIL's softmax which normalizes a single axis in place -- reproduce that by
    # reshaping down to 2-D, softmaxing, then reshaping back.
    shape = x.shape
    if any(not isinstance(d, (int, np.integer)) for d in shape):
        raise RuntimeError(
            "Softmax below opset 13 requires a fully static input shape to reproduce "
            "its flatten-then-normalize semantics"
        )
    lead = int(np.prod(shape[:axis])) if axis > 0 else 1
    trail = int(np.prod(shape[axis:]))
    flat = lowerer.mb.reshape(
        x=x, shape=[lead, trail], name=lowerer.fresh_name(node, "flat")
    )
    sm = lowerer.mb.softmax(x=flat, axis=-1, name=lowerer.fresh_name(node, "softmax"))
    return [lowerer.mb.reshape(x=sm, shape=list(shape), name=lowerer.fresh_name(node))]


def _matmul_as_conv1x1(lowerer, node, x, w):
    """If ``lowerer.matmul_to_conv`` is set and this MatMul is a linear
    projection (``x [B, S, C_in] @ w [C_in, C_out]`` with a compile-time-constant
    2-D ``w`` -- the shape every attention/MLP projection in a transformer
    decoder takes), lower it to a 1x1/pointwise ``conv`` instead of MIL's native
    ``matmul``. Returns the output ``Var``, or ``None`` if the shapes don't
    qualify (the caller falls back to ``matmul`` in that case).

    Why: per the M4 Neural Engine's own reverse-engineered behavior (see
    scripts/apple/README.md's "Theoretical ceiling" section), the ANE's compute
    array parallelizes over *convolution* output channels -- its native ``matmul``
    path only reaches ~30% of peak throughput on a single op, where a
    channel-parallel conv1x1 chain reaches ~99%. Only ``x``'s rank-3
    ``[batch, sequence, features]`` shape (this pipeline's actual shape, since
    ``export_llm_to_coreml.py`` always pins ``batch_size=1``) is handled; any
    other rank falls back to ``matmul`` rather than risk a shape bug on an
    untested case.

    The rewrite is a pure transpose/conv/transpose (MIL's ``conv`` accepts
    ``[n, C_in, *d_in]`` for ``1 <= len(d_in) <= 3``, so 1-D conv's ``[n, C_in,
    L]`` is exactly ``x`` transposed) -- no shape needs to be known or computed
    ahead of time, so this is as dynamic-shape-safe as the matmul it replaces.
    The weight transpose/reshape happens once, in Python, on the constant
    array itself (never as graph ops), so it costs nothing at runtime.
    """
    if not lowerer.matmul_to_conv:
        return None
    if w.val is None or w.rank != 2:
        return None
    if x.rank != 3:
        return None

    # ONNX MatMul convention: w is [C_in, C_out]. MIL conv wants weight
    # [C_out, C_in, K]; K=1 here (pointwise/1x1-equivalent).
    conv_w = lowerer.make_const(
        node.input[1], np.ascontiguousarray(w.val.T)[:, :, None]
    )
    xt = lowerer.mb.transpose(
        x=x, perm=[0, 2, 1], name=lowerer.fresh_name(node, "conv_in")
    )
    conv_out = lowerer.mb.conv(
        x=xt,
        weight=conv_w,
        strides=[1],
        pad_type="valid",
        dilations=[1],
        groups=1,
        name=lowerer.fresh_name(node, "conv"),
    )
    return lowerer.mb.transpose(
        x=conv_out, perm=[0, 2, 1], name=lowerer.fresh_name(node)
    )


@_register("MatMul")
def _op_matmul(lowerer, node, ins, attrs):
    conv_out = _matmul_as_conv1x1(lowerer, node, ins[0], ins[1])
    if conv_out is not None:
        return [conv_out]
    return [lowerer.mb.matmul(x=ins[0], y=ins[1], name=lowerer.fresh_name(node))]


@_register("Gemm")
def _op_gemm(lowerer, node, ins, attrs):
    a, b = ins[0], ins[1]
    c = ins[2] if len(ins) > 2 else None
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    trans_a = bool(attrs.get("transA", 0))
    trans_b = bool(attrs.get("transB", 0))
    y = lowerer.mb.matmul(
        x=a,
        y=b,
        transpose_x=trans_a,
        transpose_y=trans_b,
        name=lowerer.fresh_name(node, "matmul"),
    )
    if alpha != 1.0:
        np_dtype = lowerer.types.nptype_from_builtin(y.dtype)
        y = lowerer.mb.mul(
            x=y, y=np_dtype(alpha), name=lowerer.fresh_name(node, "alpha")
        )
    if c is not None:
        if beta == 1.0:
            term = c
        else:
            np_dtype = lowerer.types.nptype_from_builtin(c.dtype)
            term = lowerer.mb.mul(
                x=c, y=np_dtype(beta), name=lowerer.fresh_name(node, "beta")
            )
        y = lowerer.mb.add(x=y, y=term, name=lowerer.fresh_name(node))
    return [y]


@_register("Conv")
def _op_conv(lowerer, node, ins, attrs):
    x, w = ins[0], ins[1]
    bias = ins[2] if len(ins) > 2 else None
    n = x.rank - 2
    pad_type, pad = _pad_type_and_pads(attrs, n)
    kwargs = dict(
        x=x,
        weight=w,
        strides=list(attrs.get("strides", [1] * n)),
        pad_type=pad_type,
        dilations=list(attrs.get("dilations", [1] * n)),
        groups=int(attrs.get("group", 1)),
        name=lowerer.fresh_name(node),
    )
    if pad is not None:
        kwargs["pad"] = pad
    if bias is not None:
        kwargs["bias"] = bias
    return [lowerer.mb.conv(**kwargs)]


@_register("ConvTranspose")
def _op_conv_transpose(lowerer, node, ins, attrs):
    x, w = ins[0], ins[1]
    bias = ins[2] if len(ins) > 2 else None
    output_padding = attrs.get("output_padding")
    if output_padding and any(output_padding):
        raise RuntimeError(
            "ConvTranspose with a non-zero 'output_padding' is not supported (the "
            "output shape it implies isn't derivable from strides/pads alone)"
        )
    n = x.rank - 2
    pad_type, pad = _pad_type_and_pads(attrs, n)
    kwargs = dict(
        x=x,
        weight=w,
        strides=list(attrs.get("strides", [1] * n)),
        pad_type=pad_type,
        dilations=list(attrs.get("dilations", [1] * n)),
        groups=int(attrs.get("group", 1)),
        name=lowerer.fresh_name(node),
    )
    if pad is not None:
        kwargs["pad"] = pad
    if bias is not None:
        kwargs["bias"] = bias
    return [lowerer.mb.conv_transpose(**kwargs)]


@_register("BatchNormalization")
def _op_batch_norm(lowerer, node, ins, attrs):
    if len(node.output) > 1:
        raise RuntimeError(
            "training-mode BatchNormalization (with running-stats outputs) is not "
            "supported; only inference-mode (single-output) BatchNormalization is"
        )
    x, scale, bias, mean, var = ins[:5]
    eps = float(attrs.get("epsilon", 1e-5))
    return [
        lowerer.mb.batch_norm(
            x=x,
            mean=mean,
            variance=var,
            gamma=scale,
            beta=bias,
            epsilon=eps,
            name=lowerer.fresh_name(node),
        )
    ]


@_register("InstanceNormalization")
def _op_instance_norm(lowerer, node, ins, attrs):
    x, scale, bias = ins[:3]
    eps = float(attrs.get("epsilon", 1e-5))
    return [
        lowerer.mb.instance_norm(
            x=x, gamma=scale, beta=bias, epsilon=eps, name=lowerer.fresh_name(node)
        )
    ]


@_register("LayerNormalization")
def _op_layer_norm(lowerer, node, ins, attrs):
    x = ins[0]
    scale = ins[1] if len(ins) > 1 and ins[1] is not None else None
    bias = ins[2] if len(ins) > 2 and ins[2] is not None else None
    axis = int(attrs.get("axis", -1))
    axis = axis if axis >= 0 else axis + x.rank
    eps = float(attrs.get("epsilon", 1e-5))
    kwargs = dict(
        x=x, axes=list(range(axis, x.rank)), epsilon=eps, name=lowerer.fresh_name(node)
    )
    if scale is not None:
        kwargs["gamma"] = scale
    if bias is not None:
        kwargs["beta"] = bias
    return [lowerer.mb.layer_norm(**kwargs)]


def _pool(mil_name: str, extra_kwargs: Optional[Callable] = None):
    def handler(lowerer, node, ins, attrs):
        if len(node.output) > 1:
            raise RuntimeError(
                f"{node.op_type} with an Indices output is not supported"
            )
        dilations = attrs.get("dilations")
        if dilations and any(d != 1 for d in dilations):
            raise RuntimeError(f"{node.op_type} with dilations != 1 is not supported")
        x = ins[0]
        n = x.rank - 2
        pad_type, pad = _pad_type_and_pads(attrs, n)
        kwargs = dict(
            x=x,
            kernel_sizes=list(attrs["kernel_shape"]),
            strides=list(attrs.get("strides", [1] * n)),
            pad_type=pad_type,
            ceil_mode=bool(attrs.get("ceil_mode", 0)),
            name=lowerer.fresh_name(node),
        )
        if pad is not None:
            kwargs["pad"] = pad
        if extra_kwargs is not None:
            kwargs.update(extra_kwargs(attrs))
        return [getattr(lowerer.mb, mil_name)(**kwargs)]

    return handler


_OP_HANDLERS["MaxPool"] = _pool("max_pool")
_OP_HANDLERS["AveragePool"] = _pool(
    "avg_pool",
    lambda attrs: {
        "exclude_padding_from_average": not bool(attrs.get("count_include_pad", 0))
    },
)


def _global_reduce(mil_name: str):
    def handler(lowerer, node, ins, attrs):
        x = ins[0]
        axes = list(range(2, x.rank))
        fn = getattr(lowerer.mb, mil_name)
        return [fn(x=x, axes=axes, keep_dims=True, name=lowerer.fresh_name(node))]

    return handler


_OP_HANDLERS["GlobalAveragePool"] = _global_reduce("reduce_mean")
_OP_HANDLERS["GlobalMaxPool"] = _global_reduce("reduce_max")


def _reduce(mil_name: str):
    def handler(lowerer, node, ins, attrs):
        x = ins[0]
        axes = attrs.get("axes")
        if len(ins) > 1 and ins[1] is not None:
            axes = [int(v) for v in ins[1].val]
        keep_dims = bool(attrs.get("keepdims", 1))
        kwargs = dict(x=x, keep_dims=keep_dims, name=lowerer.fresh_name(node))
        if axes is not None:
            kwargs["axes"] = [int(a) if a >= 0 else int(a) + x.rank for a in axes]
        return [getattr(lowerer.mb, mil_name)(**kwargs)]

    return handler


_OP_HANDLERS["ReduceMean"] = _reduce("reduce_mean")
_OP_HANDLERS["ReduceSum"] = _reduce("reduce_sum")
_OP_HANDLERS["ReduceMax"] = _reduce("reduce_max")


@_register("Reshape")
def _op_reshape(lowerer, node, ins, attrs):
    # MIL's reshape accepts a non-constant `shape` input directly (unlike most
    # other shape-consuming ops here), so a `shape` derived from a dynamic Shape
    # op (e.g. "keep the KV cache's current length, flatten the rest") works with
    # no special-casing -- pass it straight through either way.
    return [lowerer.mb.reshape(x=ins[0], shape=ins[1], name=lowerer.fresh_name(node))]


@_register("Transpose")
def _op_transpose(lowerer, node, ins, attrs):
    x = ins[0]
    perm = list(attrs.get("perm", list(reversed(range(x.rank)))))
    return [lowerer.mb.transpose(x=x, perm=perm, name=lowerer.fresh_name(node))]


@_register("Flatten")
def _op_flatten(lowerer, node, ins, attrs):
    axis = int(attrs.get("axis", 1))
    return [lowerer.mb.flatten2d(x=ins[0], axis=axis, name=lowerer.fresh_name(node))]


@_register("Squeeze")
def _op_squeeze(lowerer, node, ins, attrs):
    x = ins[0]
    axes = None
    if len(ins) > 1 and ins[1] is not None:
        axes = [int(v) for v in ins[1].val]
    elif "axes" in attrs:
        axes = list(attrs["axes"])
    kwargs = dict(x=x, name=lowerer.fresh_name(node))
    if axes is not None:
        kwargs["axes"] = [int(a) if a >= 0 else int(a) + x.rank for a in axes]
    return [lowerer.mb.squeeze(**kwargs)]


@_register("Unsqueeze")
def _op_unsqueeze(lowerer, node, ins, attrs):
    x = ins[0]
    if len(ins) > 1 and ins[1] is not None:
        axes = [int(v) for v in ins[1].val]
    else:
        axes = list(attrs["axes"])
    out_rank = x.rank + len(axes)
    axes = sorted(a if a >= 0 else a + out_rank for a in axes)
    return [lowerer.mb.expand_dims(x=x, axes=axes, name=lowerer.fresh_name(node))]


@_register("Concat")
def _op_concat(lowerer, node, ins, attrs):
    axis = int(attrs["axis"])
    axis = axis if axis >= 0 else axis + ins[0].rank
    # Drop inputs that are provably empty along the concat axis (static 0):
    # concatenating them is a no-op semantically, and backends have been
    # observed to miscompile the pattern -- E5RT/MPS computes a downstream
    # shape of 97 for Phi-3's full-rotary `[96:96]`-of-96 empty pass-through
    # slice concatenated back onto the rotated part, failing plan build with
    # "mps.concat op invalid input tensor shapes" on an otherwise valid
    # model (verified empty by executing the subgraph on ONNX Runtime).
    # An input empty along any *other* axis is left alone (the result is
    # genuinely empty there, not a droppable no-op).
    live = [
        v
        for v in ins
        if not (
            len(v.shape) > axis
            and isinstance(v.shape[axis], (int, np.integer))
            and int(v.shape[axis]) == 0
        )
    ]
    if not live:
        live = ins[:1]
    if len(live) == 1:
        return [live[0]]
    return [lowerer.mb.concat(values=live, axis=axis, name=lowerer.fresh_name(node))]


@_register("Split")
def _op_split(lowerer, node, ins, attrs):
    x = ins[0]
    axis = int(attrs.get("axis", 0))
    axis = axis if axis >= 0 else axis + x.rank
    split_sizes = attrs.get("split")
    if len(ins) > 1 and ins[1] is not None:
        split_sizes = [int(v) for v in ins[1].val]
    name = lowerer.fresh_name(node)
    if split_sizes is not None:
        outs = lowerer.mb.split(
            x=x, split_sizes=[int(s) for s in split_sizes], axis=axis, name=name
        )
    else:
        outs = lowerer.mb.split(x=x, num_splits=len(node.output), axis=axis, name=name)
    return list(outs)


@_register("Pad")
def _op_pad(lowerer, node, ins, attrs):
    x = ins[0]
    if len(ins) > 1 and ins[1] is not None:
        pads_flat = [int(v) for v in ins[1].val]
    else:
        pads_flat = [int(v) for v in attrs["pads"]]
    mode = attrs.get("mode", "constant")
    mil_mode = {"constant": "constant", "reflect": "reflect", "edge": "replicate"}.get(
        mode
    )
    if mil_mode is None:
        raise RuntimeError(
            f"Pad mode {mode!r} is not supported (supported: constant, reflect, edge)"
        )
    const_val = 0.0
    if len(ins) > 2 and ins[2] is not None:
        const_val = _const_scalar(ins[2])
    elif "value" in attrs:
        const_val = float(attrs["value"])
    n = x.rank
    begins, ends = pads_flat[:n], pads_flat[n:]
    interleaved: List[int] = []
    for b, e in zip(begins, ends):
        interleaved += [b, e]
    return [
        lowerer.mb.pad(
            x=x,
            pad=interleaved,
            mode=mil_mode,
            constant_val=float(const_val),
            name=lowerer.fresh_name(node),
        )
    ]


def _slice_scalar_element(lowerer, var, index: int, node, suffix: str):
    """Element ``index`` of 1-D int tensor ``var``, as a length-1 piece: a plain
    Python ``int`` if statically known, else a shape-``(1,)`` MIL Var."""
    if var.val is not None:
        return int(var.val[index])
    return lowerer.mb.slice_by_index(
        x=var,
        begin=[index],
        end=[index + 1],
        stride=[1],
        name=lowerer.fresh_name(node, suffix),
    )


def _axis_size(lowerer, x, axis: int, node, suffix: str):
    """Size of ``x`` along ``axis``: a plain Python ``int`` if statically known,
    else a shape-``(1,)`` MIL Var (via MIL's runtime ``shape`` op)."""
    dim = x.shape[axis]
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    full_shape = lowerer.mb.shape(x=x, name=lowerer.fresh_name(node, suffix + "_shape"))
    return lowerer.mb.slice_by_index(
        x=full_shape,
        begin=[axis],
        end=[axis + 1],
        stride=[1],
        name=lowerer.fresh_name(node, suffix),
    )


def _as_slice_piece(value):
    """A shape-``(1,)`` int32 array for a Python ``int`` piece; ``value`` itself
    (already a MIL Var) otherwise. Used to mix static and dynamic Slice
    begin/end pieces in one ``mb.concat``."""
    return value if not isinstance(value, int) else np.array([value], dtype=np.int32)


def _clamp_nonneg_slice_bound(lowerer, value, dim, node, suffix: str):
    """Clamp a Slice ``start``/``end`` value into ``[0, dim]`` for a positive
    stride, wrapping a negative value via ``value + dim`` first (ONNX Slice's own
    rule). ``value``/``dim`` are each a Python ``int`` or a shape-``(1,)`` MIL Var
    (see ``_slice_scalar_element``/``_axis_size``); returns a Python ``int`` only
    when both inputs are, else a MIL Var.
    """
    if isinstance(value, int) and isinstance(dim, int):
        wrapped = value + dim if value < 0 else value
        return max(0, min(wrapped, dim))
    value_t, dim_t = _as_slice_piece(value), _as_slice_piece(dim)
    zero = np.array([0], dtype=np.int32)
    wrapped = lowerer.mb.select(
        cond=lowerer.mb.less(
            x=value_t, y=zero, name=lowerer.fresh_name(node, suffix + "_neg")
        ),
        a=lowerer.mb.add(
            x=value_t, y=dim_t, name=lowerer.fresh_name(node, suffix + "_wrap")
        ),
        b=value_t,
        name=lowerer.fresh_name(node, suffix + "_sel"),
    )
    hi_clamped = lowerer.mb.minimum(
        x=wrapped, y=dim_t, name=lowerer.fresh_name(node, suffix + "_min")
    )
    return lowerer.mb.maximum(
        x=hi_clamped, y=zero, name=lowerer.fresh_name(node, suffix)
    )


@_register("Slice")
def _op_slice(lowerer, node, ins, attrs):
    x = ins[0]
    rank = x.rank

    if len(ins) > 1:
        starts_var, ends_var = ins[1], ins[2]
        axes_var = ins[3] if len(ins) > 3 and ins[3] is not None else None
        steps_var = ins[4] if len(ins) > 4 and ins[4] is not None else None
        if axes_var is not None and axes_var.val is None:
            raise RuntimeError("Slice requires a compile-time-constant 'axes' input")
        if steps_var is not None and steps_var.val is None:
            raise RuntimeError("Slice requires a compile-time-constant 'steps' input")
        if axes_var is not None:
            axes = [int(v) for v in axes_var.val]
        elif starts_var.val is not None:
            axes = list(range(len(starts_var.val)))
        elif ends_var.val is not None:
            axes = list(range(len(ends_var.val)))
        else:
            raise RuntimeError(
                "Slice needs a compile-time-constant 'axes' input when both "
                "'starts' and 'ends' are dynamic (there is no way to tell how "
                "many axes are being sliced)"
            )
        steps = (
            [int(v) for v in steps_var.val]
            if steps_var is not None
            else [1] * len(axes)
        )
        starts = [
            _slice_scalar_element(lowerer, starts_var, i, node, "start")
            for i in range(len(axes))
        ]
        ends = [
            _slice_scalar_element(lowerer, ends_var, i, node, "end")
            for i in range(len(axes))
        ]
    else:
        axes = [int(v) for v in attrs.get("axes", range(len(attrs["starts"])))]
        steps = [1] * len(axes)
        starts = [int(v) for v in attrs["starts"]]
        ends = [int(v) for v in attrs["ends"]]

    # An axis this node doesn't slice at all is left alone via begin_mask/end_mask,
    # rather than pre-filled with `shape[a]` -- that axis's size doesn't need to be
    # known at conversion time then (e.g. a dynamic KV-cache axis untouched by this
    # particular Slice).
    begin_list: List[Any] = [0] * rank
    end_list: List[Any] = [0] * rank
    stride = [1] * rank
    begin_mask, end_mask = [True] * rank, [True] * rank
    for s, e, a, st in zip(starts, ends, axes, steps):
        a = a if a >= 0 else a + rank
        dim = _axis_size(lowerer, x, a, node, f"dim{a}")
        is_dynamic = not (
            isinstance(s, int) and isinstance(e, int) and isinstance(dim, int)
        )
        if st < 0:
            if is_dynamic:
                raise RuntimeError(
                    f"Slice on axis {a} with a negative stride and a dynamic "
                    "start/end/axis-size is not supported"
                )
            # Reproduce ONNX Slice's own clamp algorithm (numpy-style
            # negative-index wraparound, then clamp) rather than Python's
            # `slice.indices()`: for a negative stride, a fully-clamped end of -1
            # means "include index 0", but MIL's `end` re-wraps a literal -1 to
            # `dim - 1` (numpy indexing semantics), which would silently turn
            # that into an empty slice. Route that one case through `end_mask`
            # (MIL's "ignore `end`, go to the natural boundary" flag) instead.
            s = s + dim if s < 0 else s
            e = e + dim if e < 0 else e
            s = max(0, min(s, dim - 1))
            e = max(-1, min(e, dim - 1))
            begin_list[a], stride[a] = s, st
            begin_mask[a] = False
            if e == -1:
                end_mask[a] = True  # ignored: end_list[a] below is a don't-care
            else:
                end_list[a] = e
                end_mask[a] = False
        else:
            begin_list[a] = _clamp_nonneg_slice_bound(
                lowerer, s, dim, node, f"begin{a}"
            )
            end_list[a] = _clamp_nonneg_slice_bound(lowerer, e, dim, node, f"end{a}")
            stride[a] = st
            begin_mask[a] = False
            end_mask[a] = False

    if all(isinstance(v, int) for v in begin_list + end_list):
        begin, end = begin_list, end_list
    else:
        begin = lowerer.mb.concat(
            values=[_as_slice_piece(v) for v in begin_list],
            axis=0,
            name=lowerer.fresh_name(node, "begin"),
        )
        end = lowerer.mb.concat(
            values=[_as_slice_piece(v) for v in end_list],
            axis=0,
            name=lowerer.fresh_name(node, "end"),
        )

    return [
        lowerer.mb.slice_by_index(
            x=x,
            begin=begin,
            end=end,
            stride=stride,
            begin_mask=begin_mask,
            end_mask=end_mask,
            name=lowerer.fresh_name(node),
        )
    ]


@_register("Gather")
def _op_gather(lowerer, node, ins, attrs):
    x, indices = ins[0], ins[1]
    axis = int(attrs.get("axis", 0))
    axis = axis if axis >= 0 else axis + x.rank
    # MIL's gather has no bool overload (unlike most other ops here); round-trip
    # through int32 for a bool `x` (e.g. gathering rows out of a boolean mask).
    is_bool = x.dtype == lowerer.types.bool
    if is_bool:
        x = lowerer.mb.cast(
            x=x, dtype="int32", name=lowerer.fresh_name(node, "boolcast")
        )
    out = lowerer.mb.gather(
        x=x,
        indices=indices,
        axis=axis,
        name=lowerer.fresh_name(node, "gather" if is_bool else None),
    )
    if is_bool:
        out = lowerer.mb.cast(x=out, dtype="bool", name=lowerer.fresh_name(node))
    return [out]


@_register("GatherND")
def _op_gather_nd(lowerer, node, ins, attrs):
    x, indices = ins[:2]
    batch_dims = int(attrs.get("batch_dims", 0))
    kwargs = {"x": x, "indices": indices, "name": lowerer.fresh_name(node)}
    if batch_dims:
        kwargs["batch_dims"] = batch_dims
    return [lowerer.mb.gather_nd(**kwargs)]


@_register("Tile")
def _op_tile(lowerer, node, ins, attrs):
    reps = ins[1]
    if reps.val is None:
        raise RuntimeError("Tile requires a compile-time-constant 'repeats' input")
    return [
        lowerer.mb.tile(
            x=ins[0], reps=[int(v) for v in reps.val], name=lowerer.fresh_name(node)
        )
    ]


@_register("Shape")
def _op_shape(lowerer, node, ins, attrs):
    x = ins[0]
    shape = x.shape
    rank = len(shape)
    start = int(attrs.get("start", 0))
    start = start if start >= 0 else start + rank
    end = int(attrs.get("end", rank))
    end = end if end >= 0 else end + rank
    sliced = shape[start:end]
    if all(isinstance(d, (int, np.integer)) for d in sliced):
        return [lowerer.make_const(node.output[0], np.array(sliced, dtype=np.int64))]
    # At least one requested dim is only known at runtime (e.g. a KV cache's
    # dynamic length) -- emit MIL's own runtime `shape` op instead of a constant.
    whole_range = (start, end) == (0, rank)
    full = lowerer.mb.shape(
        x=x, name=lowerer.fresh_name(node, None if whole_range else "full")
    )
    if whole_range:
        return [full]
    return [
        lowerer.mb.slice_by_index(
            x=full, begin=[start], end=[end], stride=[1], name=lowerer.fresh_name(node)
        )
    ]


@_register("ConstantOfShape")
def _op_constant_of_shape(lowerer, node, ins, attrs):
    shape_var = ins[0]
    value = attrs.get("value")
    if value is not None:
        value = np.asarray(value)
        fill_val, dtype = value.reshape(-1)[0], value.dtype
    else:
        fill_val, dtype = 0.0, np.float32
    if shape_var.val is None:
        # The target shape is only known at runtime (e.g. derived from a KV
        # cache's dynamic length) -- MIL's `fill` op takes a non-constant shape,
        # unlike materializing a numpy array here. Keep `fill_val` as a numpy
        # scalar (not a bare Python `int`/`float` via `.item()`): MIL infers a
        # bare Python float as fp32 regardless of `dtype` here, which would
        # produce e.g. an fp32 fill for an fp16 model.
        mil_arr = _as_mil_array(np.array(fill_val, dtype=dtype))
        fill_val = mil_arr.dtype.type(mil_arr.reshape(-1)[0])
        return [
            lowerer.mb.fill(
                shape=shape_var, value=fill_val, name=lowerer.fresh_name(node)
            )
        ]
    shape = [int(v) for v in shape_var.val]
    return [lowerer.make_const(node.output[0], np.full(shape, fill_val, dtype=dtype))]


@_register("Range")
def _op_range(lowerer, node, ins, attrs):
    start, limit, delta = ins
    return [
        lowerer.mb.range_1d(
            start=start, end=limit, step=delta, name=lowerer.fresh_name(node)
        )
    ]


_ONNX_TO_MIL_CAST = {
    onnx.TensorProto.FLOAT: "fp32",
    onnx.TensorProto.FLOAT16: "fp16",
    onnx.TensorProto.DOUBLE: "fp32",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT64: "int32",
    onnx.TensorProto.BOOL: "bool",
}


@_register("Cast")
def _op_cast(lowerer, node, ins, attrs):
    to = int(attrs["to"])
    dtype = _ONNX_TO_MIL_CAST.get(to)
    if dtype is None:
        raise RuntimeError(
            f"Cast to {onnx.TensorProto.DataType.Name(to)} is not supported "
            "(supported targets: float16, float32, double, int32, int64, bool)"
        )
    return [lowerer.mb.cast(x=ins[0], dtype=dtype, name=lowerer.fresh_name(node))]


@_register("Constant")
def _op_constant(lowerer, node, ins, attrs):
    if "value" in attrs:
        arr = np.asarray(attrs["value"])
    elif "value_float" in attrs:
        arr = np.array(attrs["value_float"], dtype=np.float32)
    elif "value_floats" in attrs:
        arr = np.array(list(attrs["value_floats"]), dtype=np.float32)
    elif "value_int" in attrs:
        arr = np.array(attrs["value_int"], dtype=np.int32)
    elif "value_ints" in attrs:
        arr = np.array(list(attrs["value_ints"]), dtype=np.int32)
    else:
        raise RuntimeError(
            "Constant node has no supported value attribute (supported: value, "
            "value_float(s), value_int(s))"
        )
    return [lowerer.make_const(node.output[0], arr)]


SUPPORTED_ONNX_OPS = tuple(sorted(_OP_HANDLERS))


def _build_mil_program(
    model: onnx.ModelProto,
    mb,
    types,
    Function,
    Program,
    RangeDim,
    TensorType,
    dynamic_shapes: Optional[Dict[str, Tuple[int, int, int]]] = None,
    matmul_to_conv: bool = False,
    opset_version: Optional[Any] = None,
    io_dtype: Optional[str] = None,
    state: Optional[Dict[str, str]] = None,
):
    graph = model.graph
    initializer_names = {t.name for t in graph.initializer}
    opset = 1
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            opset = imp.version

    lowerer = _Lowerer(mb=mb, types=types, opset=opset, matmul_to_conv=matmul_to_conv)

    # `{state input name: state output name}` -- graph inputs held resident
    # on-device across predict() calls (Core ML states) instead of crossing
    # the boundary every call. The matching outputs are written back into
    # the states and dropped from the model interface.
    state = state or {}
    state_inputs = set(state)
    state_outputs = set(state.values())
    graph_inputs = {i.name for i in graph.input} - initializer_names
    for name in state_inputs:
        if name not in graph_inputs:
            raise RuntimeError(
                f"state input {name!r} is not a graph input (states can only "
                "wrap model inputs, never initializers or intermediates)."
            )
    graph_output_names = {o.name for o in graph.output}
    for name in state_outputs:
        if name not in graph_output_names:
            raise RuntimeError(f"state output {name!r} is not a graph output.")

    dynamic_shapes = dynamic_shapes or {}
    range_dims: Dict[str, Any] = {}
    input_specs = {}
    flexible_inputs = []
    graph_dtypes: Dict[str, Any] = {}
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        if inp.name in state_inputs:
            # Resident weight: states only support fp16, so declare fp16 and
            # cast back to the graph dtype at the boundary -- the same shape
            # as the io_dtype boundary-cast pattern below. States need fully
            # static shapes (a state is a fixed on-device buffer).
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                if not d.HasField("dim_value"):
                    raise RuntimeError(
                        f"State input '{inp.name}' has a non-static dimension; "
                        "Core ML states are fixed on-device buffers, so state "
                        "inputs must be fully static."
                    )
                dims.append(int(d.dim_value))
            input_specs[inp.name] = mb.StateTensorSpec(tuple(dims), dtype=types.fp16)
            graph_dtypes[inp.name] = _onnx_elem_type_to_mil(
                inp.type.tensor_type.elem_type, types
            )
            continue
        spec, ct_input = _make_input_spec(
            inp, mb, types, dynamic_shapes, range_dims, RangeDim, TensorType, io_dtype
        )
        input_specs[inp.name] = spec
        graph_dtypes[inp.name] = _onnx_elem_type_to_mil(
            inp.type.tensor_type.elem_type, types
        )
        if ct_input is not None:
            flexible_inputs.append(ct_input)

    with Function(input_specs, opset_version=opset_version) as func:
        state_vars: Dict[str, Any] = {}
        for name, var in func.inputs.items():
            if name in state_inputs:
                state_vars[name] = var
                var = mb.read_state(
                    input=var,
                    name=lowerer.fresh_io_name(name, "read_state"),
                )
            if var.dtype != graph_dtypes[name]:
                # `io_dtype="fp16"` declared this input fp16 where the ONNX graph
                # says fp32. Cast straight back at the boundary so every op below
                # is lowered against exactly the dtype it would see without the
                # flag -- the graph body stays byte-identical, only the model's
                # declared input type changes. coremltools' own fp16
                # compute-precision pass folds the round trip away.
                var = mb.cast(
                    x=var,
                    dtype=types.builtin_to_string(graph_dtypes[name]),
                    name=lowerer.fresh_io_name(name, "from_fp16"),
                )
            lowerer.bind(name, var)
        for init in graph.initializer:
            lowerer.bind(
                init.name, lowerer.make_const(init.name, numpy_helper.to_array(init))
            )
        for node in graph.node:
            lowerer.lower_node(node)
        out_to_state = {v: k for k, v in state.items()}
        outputs = []
        for out in graph.output:
            if out.name in out_to_state:
                # Written back into the state instead of returned: the caller
                # reads it via MLState, never over the predict() boundary.
                # (The update's return value is deliberately left unconsumed;
                # routing it into an output would re-send the state every
                # call, which is exactly the traffic this drops.)
                src = lowerer.get(out.name)
                if src.dtype != types.fp16:
                    src = mb.cast(
                        x=src,
                        dtype="fp16",
                        name=lowerer.fresh_io_name(out.name, "to_state"),
                    )
                lowerer.mb.coreml_update_state(
                    state=state_vars[out_to_state[out.name]],
                    value=src,
                )
                continue
            var = lowerer.get(out.name)
            if io_dtype == "fp16" and var.dtype == types.fp32:
                # Terminal cast: the Core ML model's declared output dtype is the
                # dtype of the op that produces it.
                outputs.append(mb.cast(x=var, dtype="fp16", name=out.name))
            else:
                outputs.append(mb.identity(x=var, name=out.name))
        if not outputs:
            raise RuntimeError(
                "state= consumed every model output; at least one "
                "non-state output (e.g. a loss) is required."
            )
        func.set_outputs(outputs)

    prog = Program()
    prog.add_function("main", func)
    return prog, (flexible_inputs or None)


def _resolve_compute_units(ct, compute_units: Union[str, Any]):
    if isinstance(compute_units, str):
        try:
            return getattr(ct.ComputeUnit, compute_units)
        except AttributeError as exc:
            valid = ", ".join(u.name for u in ct.ComputeUnit)
            raise RuntimeError(
                f"Unknown compute_units {compute_units!r}; valid values: {valid}"
            ) from exc
    return compute_units


def _resolve_deployment_target(ct, target: Union[str, Any]):
    if isinstance(target, str):
        try:
            return getattr(ct.target, target)
        except AttributeError as exc:
            raise RuntimeError(f"Unknown minimum_deployment_target {target!r}") from exc
    return target


def _resolve_io_dtype(ct, io_dtype: Optional[str], convert_to: str, resolved_target):
    """Validate ``io_dtype`` and return ``(normalized, resolved_target)``.

    ``normalized`` is ``"fp16"`` or ``None`` ("leave the boundary alone"), and
    ``resolved_target`` is bumped to iOS16/macOS13 when fp16 I/O is requested
    without an explicit target -- that is the first Core ML version whose
    ``MLMultiArray`` model interface can be declared float16 at all.
    """
    if io_dtype is None or io_dtype == "fp32":
        return None, resolved_target
    if io_dtype != "fp16":
        raise RuntimeError(
            f"Unknown io_dtype {io_dtype!r}; valid values: 'fp32' (the default, "
            "float model inputs/outputs declared float32) or 'fp16'."
        )
    if convert_to != "mlprogram":
        raise RuntimeError(
            "io_dtype='fp16' requires convert_to='mlprogram'; the legacy "
            "'neuralnetwork' format has no float16 model interface."
        )
    floor = ct.target.iOS16
    if resolved_target is None:
        return "fp16", floor
    if int(resolved_target) < int(floor):
        raise RuntimeError(
            f"io_dtype='fp16' needs minimum_deployment_target iOS16/macOS13 or "
            f"newer (got {resolved_target.name}); float16 model inputs/outputs "
            "did not exist before then."
        )
    return "fp16", resolved_target


def _raise_target_to_floor(ct, floor, resolved_target, reason: str, label: str):
    """Raise ``resolved_target`` to at least ``floor``, or refuse an older one.

    Shared by every per-feature deployment floor below. The ``max`` matters
    when several features apply at once: each call must be free to *raise* the
    target, never to lower it back to its own floor. A resolver that returned
    its floor unconditionally would make a later, higher resolver read that
    value as the caller's explicit choice and refuse to raise past it -- which
    is exactly what a QDQ model with a nearest Resize did (iOS15 pinned by the
    Resize rule, then rejected by the iOS17 QDQ rule).
    """
    if resolved_target is None:
        return floor
    if int(resolved_target) < int(floor):
        raise RuntimeError(
            f"{reason} needs minimum_deployment_target {label} or newer "
            f"(got {resolved_target.name}); that Core ML op did not exist "
            "before then."
        )
    return resolved_target


def _resolve_state_target(ct, state, resolved_target):
    """Bump the deployment floor to iOS18 when states are requested.

    `read_state`/`coreml_update_state` only exist from the iOS18 op version
    -- same pattern as the iOS16 fp16-I/O bump and the iOS17 QDQ bump.
    """
    if not state:
        return resolved_target
    return _raise_target_to_floor(
        ct, ct.target.iOS18, resolved_target, "state=", "iOS18/macOS15"
    )


def _resolve_gather_nd_target(ct, model: onnx.ModelProto, resolved_target):
    uses_batch_dims = any(
        node.op_type == "GatherND" and int(_node_attrs(node).get("batch_dims", 0)) > 0
        for node in model.graph.node
    )
    if not uses_batch_dims:
        return resolved_target
    return _raise_target_to_floor(
        ct,
        ct.target.iOS16,
        resolved_target,
        "GatherND with batch_dims > 0",
        "iOS16/macOS13",
    )


def _resolve_resize_target(ct, model: onnx.ModelProto, resolved_target):
    """Bump the deployment floor to iOS15 when the graph has a nearest Resize.

    The fast path emits MIL's `resize_nearest_neighbor` (iOS15), so a graph that
    may take it must both build and convert at that target or newer. The check is
    deliberately coarse -- a nearest Resize that ends up ineligible and lowered to
    gathers still raises the floor -- same trade-off (and same reason: keeping the
    predicate stateless) as `_resolve_gather_nd_target` above.
    """
    if not any(
        node.op_type == "Resize"
        and _node_attrs(node).get("mode", "nearest") == "nearest"
        for node in model.graph.node
    ):
        return resolved_target
    return _raise_target_to_floor(
        ct, ct.target.iOS15, resolved_target, "a nearest Resize", "iOS15/macOS12"
    )


def _resolve_quantized_target(ct, model: onnx.ModelProto, resolved_target):
    """Bump the deployment floor to iOS17 when the graph holds QDQ nodes.

    MIL's `quantize`/`dequantize` only exist from the iOS17 op version, so a
    model using them must both build and convert at that target or newer --
    same shape as `_resolve_io_dtype`'s iOS16 bump for fp16 I/O: raise the
    floor silently when the caller didn't pin one, refuse an explicitly older
    one rather than emitting a model `coremlcompiler` would reject.
    """
    if not any(
        node.op_type in ("QuantizeLinear", "DequantizeLinear")
        for node in model.graph.node
    ):
        return resolved_target
    return _raise_target_to_floor(
        ct,
        ct.target.iOS17,
        resolved_target,
        "QuantizeLinear/DequantizeLinear",
        "iOS17/macOS14",
    )


def convert_to_coreml(
    model: onnx.ModelProto,
    *,
    convert_to: str = "mlprogram",
    compute_units: Union[str, Any] = "ALL",
    compute_precision: Optional[Any] = None,
    minimum_deployment_target: Optional[Union[str, Any]] = None,
    skip_model_load: bool = True,
    dynamic_shapes: Optional[Dict[str, Tuple[int, int, int]]] = None,
    matmul_to_conv: bool = False,
    io_dtype: Optional[str] = None,
    state: Optional[Dict[str, str]] = None,
):
    """Convert an ONNX model to an in-memory Core ML model.

    Parameters
    ----------
    model:
        The ONNX model to convert. Typically the output of :func:`onnxsim.simplify`.
        Every graph input dimension must be either static or named in
        ``dynamic_shapes``; every node's op must be one of ``SUPPORTED_ONNX_OPS``.
    convert_to:
        Core ML model type: ``"mlprogram"`` (the modern ``.mlpackage`` format,
        default) or ``"neuralnetwork"`` (the legacy ``.mlmodel`` format).
    compute_units:
        Which compute devices the model may run on: ``"ALL"`` (default),
        ``"CPU_ONLY"``, ``"CPU_AND_GPU"``, or ``"CPU_AND_NE"``. Accepts either the
        string name or a ``coremltools.ComputeUnit`` member.
    compute_precision:
        Forwarded to ``coremltools.convert`` (e.g. ``coremltools.precision.FLOAT16``);
        left at coremltools' own default when ``None``.
    minimum_deployment_target:
        Minimum OS version the model must run on, e.g. ``"iOS16"``/``"macOS13"``, or a
        ``coremltools.target`` member. Left at coremltools' own default when ``None``.
        Raised automatically when the graph needs a newer op: a nearest ``Resize``
        (iOS15), a batched ``GatherND`` or fp16 I/O (iOS16), QDQ nodes (iOS17), or
        ``state=`` (iOS18). An explicitly older target is refused rather than
        producing a model ``coremlcompiler`` would reject.
    skip_model_load:
        Skip compiling/loading the produced model for prediction (default ``True``).
        Compiling requires Apple's Core ML toolchain and only succeeds on macOS; leave
        this ``True`` to convert models on any OS, or pass ``False`` on macOS to get a
        model that's ready to call ``.predict()`` on.
    dynamic_shapes:
        Opt in specific ONNX input dimensions as dynamic instead of requiring every
        dimension to be static. Maps an ONNX ``dim_param`` name (e.g.
        ``"past_sequence_length"``, the dimension that grows as a KV cache fills)
        to a ``(lower_bound, default, upper_bound)`` tuple of concrete sizes. Every
        input that shares the same ``dim_param`` name varies together (the common
        case: a KV cache's ``past_key_values.*.key``/``value`` inputs). A
        ``dim_param`` an ONNX exporter wrote as a derived expression (e.g.
        ``"past_sequence_length + sequence_length"``, seen on an attention mask)
        needs its own entry with that exact string as the key -- it is not
        automatically inferred from the terms it names. ``None`` (the default)
        requires every dimension to be static, as before.
    matmul_to_conv:
        Lower a linear-projection ``MatMul`` (``x [batch, sequence, C_in] @ w
        [C_in, C_out]`` with compile-time-constant ``w``, rank-3 ``x`` -- the
        shape every attention/MLP projection in a transformer decoder takes)
        to a 1x1/pointwise ``conv`` instead of MIL's native ``matmul``
        (default ``False``). The Apple Neural Engine's compute array
        parallelizes over convolution output channels; its native matmul path
        measures well below conv1x1's throughput on the same hardware -- see
        `scripts/apple/README.md <../scripts/apple/README.md>`_'s "Theoretical
        ceiling" section for the measurements this is based on and whether it
        actually helps *this* pipeline's shapes (mostly single-token decode
        steps, unlike the deep, wide chains that finding was measured on).
        Any ``MatMul`` that doesn't match that exact shape (a non-constant or
        non-2-D weight, or ``x`` of any rank other than 3) is left on the
        ``matmul`` path unchanged.
    io_dtype:
        Dtype of the *model interface* -- the float inputs and outputs Core ML
        exposes to a caller. ``None``/``"fp32"`` (the default) leaves them
        float32, coremltools' own default; ``"fp16"`` declares every float input
        and output float16 instead.

        An ML Program already computes in float16 by default, so a float32
        interface makes Core ML convert every float input down to fp16 on the way
        in and every float output back up on the way out, moving twice the bytes
        across the boundary in the process. Declaring the boundary fp16 skips
        both conversions; the ANE reverse-engineering work this follows measures
        direct fp16 I/O at roughly 37% faster than fp32 I/O on the same buffers
        (`maderix/ANE <https://github.com/maderix/ANE>`_, "How It Works" step 3).
        It costs no accuracy relative to the default either: the computation was
        already fp16 either way, so a float32 output is an upcast fp16 value.

        Worth most on a graph that hands the same large float tensors back and
        forth every call -- a decoder's KV cache, which arrives as
        ``past_key_values_*`` inputs and leaves as ``present_*`` outputs on every
        single decode step. Integer and bool inputs are unaffected (Core ML has
        no fp16 form for them). Requires ``convert_to="mlprogram"`` and raises
        ``minimum_deployment_target`` to iOS16/macOS13 when one isn't given.
    state:
        Hold graph inputs resident on-device across ``predict()`` calls
        instead of sending them every call: ``{input name: output name}``
        (the same shape as :class:`onnxsim.qat_graph.StepGraph`'s ``state``
        mapping). Each named input becomes a Core ML state (fp16, static
        shape required), each named output is written back into its state via
        ``coreml_update_state`` and dropped from the model interface -- the
        caller reads it back with ``MLState.read_state``, never over the
        ``predict()`` boundary. ``None`` (the default) leaves the interface
        alone. Requires ``convert_to="mlprogram"`` and raises
        ``minimum_deployment_target`` to iOS18/macOS15 when one isn't given
        (states did not exist before then).

    Returns
    -------
    coremltools.models.MLModel

    Raises
    ------
    RuntimeError
        If coremltools is not installed, an input has a dimension that's neither
        static nor named in ``dynamic_shapes``, or the graph uses an ONNX
        op/feature this translator does not support.
    """
    ct = _import_coremltools()
    mb, types, Function, Program, RangeDim, TensorType = _import_mil()

    # Resolved once and threaded into both _build_mil_program (as the MIL
    # program's own opset_version) and ct.convert's minimum_deployment_target
    # below -- building the program directly at the requested target's opset
    # (rather than always at coremltools' lowest default and relying on
    # ct.convert's own op-version-upgrade pass to bridge the gap) is required
    # for correctness: that upgrade pass has been observed to leave newly-
    # applicable optional inputs (e.g. Gather's `validate_indices`, added at
    # the iOS17 op version) unset in the serialized spec, which coremlcompiler
    # then rejects at load time ("Required param 'validate_indices' is
    # missing") -- building at the real target from the start lets MIL's own
    # builder synthesize each op's version-appropriate default inputs itself.
    resolved_target = (
        _resolve_deployment_target(ct, minimum_deployment_target)
        if minimum_deployment_target is not None
        else None
    )
    io_dtype, resolved_target = _resolve_io_dtype(
        ct, io_dtype, convert_to, resolved_target
    )
    resolved_target = _resolve_gather_nd_target(ct, model, resolved_target)
    resolved_target = _resolve_quantized_target(ct, model, resolved_target)
    resolved_target = _resolve_resize_target(ct, model, resolved_target)
    resolved_target = _resolve_state_target(ct, state, resolved_target)

    prog, flexible_inputs = _build_mil_program(
        model,
        mb,
        types,
        Function,
        Program,
        RangeDim,
        TensorType,
        dynamic_shapes,
        matmul_to_conv,
        opset_version=resolved_target,
        io_dtype=io_dtype,
        state=state,
    )

    kwargs: Dict[str, Any] = {}
    if compute_precision is not None:
        kwargs["compute_precision"] = compute_precision
    if resolved_target is not None:
        kwargs["minimum_deployment_target"] = resolved_target
    if flexible_inputs is not None:
        kwargs["inputs"] = flexible_inputs
    if state:
        # `common::canonicalize_inplace_pattern` reorders `coreml_update_state`
        # ops and crashes on clustered trailing updates (KeyError inserting
        # before an already-removed op -- reproduced with a 12-state training
        # step while smaller stateful graphs convert fine). The pass only
        # canonicalizes update placement for memory reuse, so skipping it is
        # functionally neutral; without it the stateful conversion below
        # succeeds. Pinned by test (a coremltools rename would surface there,
        # since `remove_passes` silently no-ops on unknown names).
        from coremltools.converters.mil.mil.passes.pass_pipeline import (
            PassPipeline,
        )

        pipeline = PassPipeline.get_pipeline("default")
        pipeline.remove_passes({"common::canonicalize_inplace_pattern"})
        kwargs["pass_pipeline"] = pipeline

    return ct.convert(
        prog,
        source="milinternal",
        convert_to=convert_to,
        compute_units=_resolve_compute_units(ct, compute_units),
        skip_model_load=skip_model_load,
        **kwargs,
    )


def export_coreml(
    model: onnx.ModelProto,
    output_path: Optional[str] = None,
    **kwargs,
):
    """Convert ``model`` to Core ML, optionally saving it to ``output_path``.

    This is the public entry point used by the ``onnxsim --emit-coreml`` CLI and is
    re-exported as ``onnxsim.export_coreml``. It returns the ``MLModel`` regardless of
    whether ``output_path`` is given, so it is equally usable in-memory.

    Parameters
    ----------
    model:
        The ONNX model to convert (usually the output of :func:`onnxsim.simplify`).
    output_path:
        If given, the model is saved here with ``MLModel.save`` -- a ``.mlpackage``
        directory for ``convert_to="mlprogram"`` (the default), or a ``.mlmodel`` file
        for ``convert_to="neuralnetwork"``. If ``None``, the model is only returned.

    Other keyword arguments are forwarded to :func:`convert_to_coreml`.

    Returns
    -------
    coremltools.models.MLModel
    """
    mlmodel = convert_to_coreml(model, **kwargs)
    if output_path is not None:
        mlmodel.save(output_path)
    return mlmodel
