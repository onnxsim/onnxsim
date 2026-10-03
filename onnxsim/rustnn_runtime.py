"""Runs (simplified) ONNX models natively through rustnn, a Rust
implementation of the W3C WebNN API (https://github.com/rustnn/rustnn), via
its ``pywebnn`` Python bindings (``pip install pywebnn``, imported as
``webnn``).

Everything else WebNN-related in onnxsim (``onnxsim.webnn_target``,
``scripts/convertmodel/webnn.mjs``, ``docs/webnn.md``) targets
onnxruntime-web's WebNN execution provider inside a browser, so it can only
ever be exercised from a Chromium page with the WebNN flag on. rustnn
implements the same ``MLContext`` / ``MLGraphBuilder`` / ``MLGraph`` API as a
native library, backed by ONNX Runtime, TensorRT-RTX, Core ML, LiteRT or
CANN (chosen at context creation), so a WebNN graph can be built, run and
timed from plain Python -- in CI, or next to tinygrad on the same machine
(see ``onnxsim.webnn_tinygrad_tuning``).

rustnn doesn't load ONNX itself (its sibling ``onnx2webnn`` is a separate
Rust CLI), so :func:`build_webnn_graph` lowers an ONNX graph onto
``MLGraphBuilder`` calls directly. The lowering is deliberately the same
shape as a browser WebNN graph builder would see from onnxruntime-web's
WebNN EP: static shapes only, and shape-like inputs (``Reshape``'s shape,
``Slice``'s starts/ends, ``Clip``'s bounds, ...) must be constants -- the
same constraint :func:`onnxsim.webnn_target.check_webnn_support` flags. So
run it on the *output* of ``onnxsim.simplify``, whose constant folding and
shape inference are exactly what make a graph lowerable.

:data:`WEBNN_SUPPORTED_OPS` is checked by op type alone
(:func:`find_unsupported_webnn_ops`, which needs neither rustnn nor pywebnn
installed); attribute combinations WebNN can't express (a 3-D ``Conv``,
``AveragePool`` with ``count_include_pad=1`` and non-zero pads, ...) raise
:class:`WebnnLoweringError` from :func:`build_webnn_graph`.

Verified against pywebnn 0.5.12 on its ``onnx`` (ONNX Runtime CPU) backend.
rustnn is marked experimental upstream and its API is still moving.
"""

from __future__ import annotations

import functools
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
from onnx import numpy_helper

from .einsum_decompose import decompose_einsum

__all__ = [
    "WEBNN_SUPPORTED_OPS",
    "WebnnLoweringError",
    "RustnnTiming",
    "RustnnSession",
    "find_unsupported_webnn_ops",
    "build_webnn_graph",
    "probe_rustnn",
]


class WebnnLoweringError(ValueError):
    """An ONNX node (or graph input) can't be expressed as WebNN builder
    calls -- see this module's docstring for which constraints apply."""


def _require_webnn():
    try:
        import webnn
    except ImportError as e:
        raise ImportError(
            "onnxsim.rustnn_runtime needs the optional 'pywebnn' package "
            "(rustnn's Python bindings): pip install pywebnn"
        ) from e
    return webnn


_ONNX_TO_WEBNN_DTYPE = {
    onnx.TensorProto.FLOAT: "float32",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.UINT32: "uint32",
    onnx.TensorProto.INT64: "int64",
    onnx.TensorProto.UINT64: "uint64",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.UINT8: "uint8",
    # WebNN has no bool; comparison ops produce uint8, as Chromium's
    # lowering and onnxruntime-web's WebNN EP both do.
    onnx.TensorProto.BOOL: "uint8",
}


def _create_context(device_type: str, backend: str, power_preference: str = "default"):
    """``webnn.ML().create_context(...)``, passing ``backend=`` only when a
    specific one is asked for: the released pywebnn 0.5.12 predates that
    keyword (it picks the backend from ``device_type`` alone), while rustnn's
    ``main`` accepts it."""
    webnn = _require_webnn()
    kwargs = {"power_preference": power_preference, "device_type": device_type}
    if backend != "auto":
        kwargs["backend"] = backend
    try:
        return webnn.ML().create_context(**kwargs)
    except TypeError as e:
        if "backend" in kwargs and "backend" in str(e):
            raise RuntimeError(
                f"this pywebnn build can't select backend={backend!r}; install a "
                "pywebnn built from rustnn main (maturin develop) or use backend='auto'"
            ) from e
        raise


def _webnn_dtype(elem_type: int) -> str:
    try:
        return _ONNX_TO_WEBNN_DTYPE[elem_type]
    except KeyError:
        raise WebnnLoweringError(
            f"ONNX dtype {onnx.TensorProto.DataType.Name(elem_type)} has no WebNN equivalent"
        ) from None


_UNARY = {
    "Abs": "abs",
    "Ceil": "ceil",
    "Floor": "floor",
    "Neg": "neg",
    "Sign": "sign",
    "Exp": "exp",
    "Log": "log",
    "Sqrt": "sqrt",
    "Reciprocal": "reciprocal",
    "Sin": "sin",
    "Cos": "cos",
    "Tan": "tan",
    "Erf": "erf",
    "Identity": "identity",
    "Relu": "relu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Softplus": "softplus",
    "Softsign": "softsign",
    "HardSwish": "hard_swish",
    "Not": "logical_not",
    "Round": "round_even",
}

_BINARY = {
    "Add": "add",
    "Sub": "sub",
    "Mul": "mul",
    "Div": "div",
    "Pow": "pow",
    "Equal": "equal",
    "Greater": "greater",
    "GreaterOrEqual": "greater_or_equal",
    "Less": "lesser",
    "LessOrEqual": "lesser_or_equal",
    "And": "logical_and",
    "Or": "logical_or",
    "Xor": "logical_xor",
    "MatMul": "matmul",
}

# Variadic in ONNX, folded pairwise.
_VARIADIC = {"Max": "max", "Min": "min", "Sum": "add"}

_REDUCE = {
    "ReduceSum": "reduce_sum",
    "ReduceMean": "reduce_mean",
    "ReduceMax": "reduce_max",
    "ReduceMin": "reduce_min",
    "ReduceProd": "reduce_product",
    "ReduceL1": "reduce_l1",
    "ReduceL2": "reduce_l2",
    "ReduceLogSum": "reduce_log_sum",
    "ReduceLogSumExp": "reduce_log_sum_exp",
    "ReduceSumSquare": "reduce_sum_square",
}

_OTHER = {
    "Constant",
    "LeakyRelu",
    "Elu",
    "HardSigmoid",
    "Clip",
    "Gelu",
    "Softmax",
    "Cast",
    "Where",
    "Gemm",
    "Conv",
    "ConvTranspose",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "BatchNormalization",
    "InstanceNormalization",
    "LayerNormalization",
    "Reshape",
    "Flatten",
    "Transpose",
    "Concat",
    "Squeeze",
    "Unsqueeze",
    "Expand",
    "Slice",
    "Split",
    "Gather",
    "Pad",
    "ArgMax",
    "ArgMin",
    "PRelu",
}

#: ONNX op types (default domain) :func:`build_webnn_graph` knows how to lower.
WEBNN_SUPPORTED_OPS = (
    frozenset(_UNARY)
    | frozenset(_BINARY)
    | frozenset(_VARIADIC)
    | frozenset(_REDUCE)
    | frozenset(_OTHER)
)


def find_unsupported_webnn_ops(model: Union[str, onnx.ModelProto]) -> Dict[str, int]:
    """Op types in ``model``'s main graph that :func:`build_webnn_graph` has
    no lowering for, with their node counts (empty when every op type is
    covered). Needs neither rustnn nor pywebnn.

    This is an op-type check only: a covered op can still be rejected by
    :func:`build_webnn_graph` for its attributes or for a non-constant
    shape-like input.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    counts: Dict[str, int] = {}
    for node in model.graph.node:
        key = (
            node.op_type
            if node.domain in ("", "ai.onnx")
            else f"{node.domain}::{node.op_type}"
        )
        if node.domain in ("", "ai.onnx") and node.op_type in WEBNN_SUPPORTED_OPS:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _static_shape(value_info: onnx.ValueInfoProto) -> List[int]:
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise WebnnLoweringError(
                f"graph input {value_info.name!r} has a non-static dimension "
                f"({dim.dim_param or '?'}); WebNN graphs need static shapes -- "
                "pass input_shapes= or simplify with overwrite_input_shapes"
            )
        shape.append(dim.dim_value)
    return shape


class _Lowering:
    """Walks ``graph.node`` in order, keeping every value either as a numpy
    array (initializers, ``Constant`` outputs, and shape-like values that are
    only ever read as attributes) or as a WebNN ``MLOperand``."""

    def __init__(
        self,
        builder,
        graph: onnx.GraphProto,
        input_shapes: Mapping[str, Sequence[int]],
        coreml: bool = False,
    ):
        self.b = builder
        # rustnn 0.5.12's Core ML backend drops conv/gemm biases, only has
        # int32 arg-reductions, and mis-handles a few ops outright; see
        # build_webnn_graph and docs/rustnn.md.
        self.coreml = coreml
        self.consts: Dict[str, np.ndarray] = {
            init.name: numpy_helper.to_array(init) for init in graph.initializer
        }
        self.ops: Dict[str, Any] = {}
        self.input_dtypes: Dict[str, str] = {}
        for vi in graph.input:
            if vi.name in self.consts:
                continue
            shape = (
                list(input_shapes[vi.name])
                if vi.name in input_shapes
                else _static_shape(vi)
            )
            dtype = _webnn_dtype(vi.type.tensor_type.elem_type)
            self.input_dtypes[vi.name] = dtype
            self.ops[vi.name] = builder.input(vi.name, [int(d) for d in shape], dtype)

    # -- value access --------------------------------------------------
    def operand(self, name: str):
        if name in self.ops:
            return self.ops[name]
        if name in self.consts:
            arr = self.consts[name]
            if arr.dtype == np.bool_:
                arr = arr.astype(np.uint8)
            elif arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            op = self.b.constant(np.ascontiguousarray(arr))
            self.ops[name] = op
            return op
        raise WebnnLoweringError(f"value {name!r} is used before it is produced")

    def const(
        self, node: onnx.NodeProto, index: int, what: str
    ) -> Optional[np.ndarray]:
        if index >= len(node.input) or not node.input[index]:
            return None
        name = node.input[index]
        if name not in self.consts:
            raise WebnnLoweringError(
                f"{node.op_type} node {node.name or node.output[0]!r}: {what} input "
                f"{name!r} must be a constant for WebNN (simplify the model first)"
            )
        return self.consts[name]

    @staticmethod
    def shape(op) -> List[int]:
        return [int(d) for d in op.shape]

    # -- lowering ------------------------------------------------------
    def lower(self, node: onnx.NodeProto) -> None:
        if (
            node.domain not in ("", "ai.onnx")
            or node.op_type not in WEBNN_SUPPORTED_OPS
        ):
            raise WebnnLoweringError(f"no WebNN lowering for op {node.op_type!r}")
        if node.op_type == "Constant":
            attr = node.attribute[0]
            if attr.name not in (
                "value",
                "value_float",
                "value_floats",
                "value_int",
                "value_ints",
            ):
                raise WebnnLoweringError(f"Constant with {attr.name} is not supported")
            val = onnx.helper.get_attribute_value(attr)
            self.consts[node.output[0]] = (
                numpy_helper.to_array(val)
                if isinstance(val, onnx.TensorProto)
                else np.asarray(
                    val, dtype=np.float32 if "float" in attr.name else np.int64
                )
            )
            return
        result = self._lower(node, _attrs(node))
        outs = result if isinstance(result, (list, tuple)) else [result]
        for name, op in zip(node.output, outs):
            if name:
                self.ops[name] = op

    def _lower(self, node: onnx.NodeProto, a: Dict[str, Any]):
        b, t = self.b, node.op_type
        x = lambda i=0: self.operand(node.input[i])  # noqa: E731
        if t in _UNARY:
            return getattr(b, _UNARY[t])(x())
        if t in _BINARY:
            return getattr(b, _BINARY[t])(x(0), x(1))
        if t in _VARIADIC:
            out = x(0)
            for i in range(1, len(node.input)):
                out = getattr(b, _VARIADIC[t])(out, x(i))
            return out
        if t in _REDUCE:
            return self._reduce(node, a)
        return getattr(self, f"_op_{t}")(node, a, x)

    def _reduce(self, node, a):
        inp = self.operand(node.input[0])
        rank = len(inp.shape)
        axes = a.get("axes")
        if axes is None:
            c = self.const(node, 1, "axes")
            axes = None if c is None else [int(v) for v in c]
        if not axes:
            if a.get("noop_with_empty_axes", 0):
                return self.b.identity(inp)
            axes = list(range(rank))
        axes = [ax % rank for ax in axes]
        return getattr(self.b, _REDUCE[node.op_type])(
            inp, axes=axes, keep_dimensions=bool(a.get("keepdims", 1))
        )

    def _op_LeakyRelu(self, node, a, x):
        return self.b.leaky_relu(x(), alpha=a.get("alpha", 0.01))

    def _op_Elu(self, node, a, x):
        return self.b.elu(x(), alpha=a.get("alpha", 1.0))

    def _op_HardSigmoid(self, node, a, x):
        return self.b.hard_sigmoid(
            x(), alpha=a.get("alpha", 0.2), beta=a.get("beta", 0.5)
        )

    def _op_PRelu(self, node, a, x):
        return self.b.prelu(x(0), x(1))

    def _op_Clip(self, node, a, x):
        lo, hi = a.get("min"), a.get("max")
        c = self.const(node, 1, "min")
        lo = float(c) if c is not None else lo
        c = self.const(node, 2, "max")
        hi = float(c) if c is not None else hi
        return self.b.clamp(x(), min_value=lo, max_value=hi)

    def _op_Gelu(self, node, a, x):
        if a.get("approximate", b"none") not in (b"none", "none"):
            raise WebnnLoweringError("Gelu(approximate='tanh') has no WebNN equivalent")
        return self.b.gelu(x())

    def _op_Softmax(self, node, a, x):
        inp = x()
        return self.b.softmax(inp, a.get("axis", -1) % len(inp.shape))

    def _op_Cast(self, node, a, x):
        return self.b.cast(x(), _webnn_dtype(a["to"]))

    def _reject_on_coreml(self, node, why: str):
        if self.coreml:
            raise WebnnLoweringError(
                f"{node.op_type}: not supported on rustnn's Core ML backend ({why})"
            )

    def _add_bias(self, out, bias, channel_axis: int):
        """``out + bias`` with a 1-D ``bias`` broadcast along ``channel_axis``
        -- the Core ML workaround for the fused bias rustnn 0.5.12 drops."""
        shape = [1] * len(out.shape)
        shape[channel_axis] = int(bias.shape[0])
        return self.b.add(out, self.b.reshape(bias, shape))

    def _op_Where(self, node, a, x):
        self._reject_on_coreml(node, "MIL select rejects WebNN's uint8 condition")
        return self.b.where_(x(0), x(1), x(2))

    def _op_Gemm(self, node, a, x):
        c = x(2) if len(node.input) > 2 and node.input[2] else None
        alpha, beta = a.get("alpha", 1.0), a.get("beta", 1.0)
        kwargs = dict(
            alpha=alpha,
            a_transpose=bool(a.get("transA", 0)),
            b_transpose=bool(a.get("transB", 0)),
        )
        if self.coreml and c is not None:
            out = self.b.gemm(x(0), x(1), **kwargs)
            if beta != 1.0:
                c = self.b.mul(c, self.b.constant(np.array(beta, np.float32)))
            return self.b.add(out, c)
        return self.b.gemm(x(0), x(1), c=c, beta=beta, **kwargs)

    def _spatial(
        self, node, a, kernel: Sequence[int], in_hw: Sequence[int], transpose=False
    ):
        """ONNX ``pads``/``auto_pad`` -> WebNN ``[top, bottom, left, right]``."""
        strides = list(a.get("strides", [1, 1]))
        dilations = list(a.get("dilations", [1, 1]))
        auto_pad = a.get("auto_pad", b"NOTSET")
        auto_pad = auto_pad.decode() if isinstance(auto_pad, bytes) else auto_pad
        if auto_pad in ("NOTSET", "VALID"):
            p = (
                list(a.get("pads", [0, 0, 0, 0]))
                if auto_pad == "NOTSET"
                else [0, 0, 0, 0]
            )
        elif auto_pad in ("SAME_UPPER", "SAME_LOWER") and not transpose:
            p = [0, 0, 0, 0]
            for i in range(2):
                out = -(-in_hw[i] // strides[i])
                total = max(
                    0,
                    (out - 1) * strides[i]
                    + (kernel[i] - 1) * dilations[i]
                    + 1
                    - in_hw[i],
                )
                small, big = total // 2, total - total // 2
                p[i], p[i + 2] = (
                    (small, big) if auto_pad == "SAME_UPPER" else (big, small)
                )
        else:
            raise WebnnLoweringError(
                f"{node.op_type}: auto_pad={auto_pad} is not supported"
            )
        return strides, dilations, [p[0], p[2], p[1], p[3]]

    def _require_2d(self, node, op):
        if len(op.shape) != 4:
            raise WebnnLoweringError(
                f"{node.op_type}: only 2-D (NCHW) is supported by WebNN, got rank {len(op.shape)}"
            )

    def _op_Conv(self, node, a, x):
        inp, w = x(0), x(1)
        self._require_2d(node, inp)
        kernel = self.shape(w)[2:]
        strides, dilations, pads = self._spatial(node, a, kernel, self.shape(inp)[2:])
        bias = x(2) if len(node.input) > 2 and node.input[2] else None
        out = self.b.conv2d(
            inp,
            w,
            strides=strides,
            dilations=dilations,
            pads=pads,
            groups=a.get("group", 1),
            bias=None if self.coreml else bias,
        )
        return self._add_bias(out, bias, 1) if self.coreml and bias is not None else out

    def _op_ConvTranspose(self, node, a, x):
        inp, w = x(0), x(1)
        self._require_2d(node, inp)
        strides, dilations, pads = self._spatial(
            node, a, self.shape(w)[2:], self.shape(inp)[2:], transpose=True
        )
        bias = x(2) if len(node.input) > 2 and node.input[2] else None
        kwargs = {}
        if "output_padding" in a:
            kwargs["output_padding"] = list(a["output_padding"])
        if "output_shape" in a:
            kwargs["output_sizes"] = list(a["output_shape"])[-2:]
        out = self.b.conv_transpose2d(
            inp,
            w,
            strides=strides,
            dilations=dilations,
            pads=pads,
            groups=a.get("group", 1),
            filter_layout="iohw",
            bias=None if self.coreml else bias,
            **kwargs,
        )
        return self._add_bias(out, bias, 1) if self.coreml and bias is not None else out

    def _pool(self, node, a, x, fn):
        inp = x()
        self._require_2d(node, inp)
        kernel = list(a["kernel_shape"])
        in_hw = self.shape(inp)[2:]
        strides, dilations, pads = self._spatial(node, a, kernel, in_hw)
        if len(node.output) > 1 and node.output[1]:
            raise WebnnLoweringError(
                f"{node.op_type}: the Indices output is not supported"
            )
        kwargs = {}
        if a.get("ceil_mode", 0):
            begin_end = [(pads[0], pads[1]), (pads[2], pads[3])]
            spans = [
                in_hw[i] + sum(begin_end[i]) - (kernel[i] - 1) * dilations[i] - 1
                for i in range(2)
            ]
            if any(span % stride for span, stride in zip(spans, strides)):
                # Floor and ceil rounding disagree here; only newer rustnn
                # builds take outputShapeRounding (pywebnn 0.5.12 doesn't).
                kwargs["output_shape_rounding"] = "ceil"
        try:
            return fn(
                inp,
                window_dimensions=kernel,
                strides=strides,
                dilations=dilations,
                pads=pads,
                **kwargs,
            )
        except TypeError as e:
            raise WebnnLoweringError(
                f"{node.op_type}: ceil_mode=1 changes the output shape here and this "
                f"pywebnn build has no output_shape_rounding ({e})"
            ) from e

    def _op_MaxPool(self, node, a, x):
        return self._pool(node, a, x, self.b.max_pool2d)

    def _op_AveragePool(self, node, a, x):
        if a.get("count_include_pad", 0) and any(a.get("pads", [0])):
            # WebNN's averagePool2d excludes padding from the divisor.
            raise WebnnLoweringError(
                "AveragePool: count_include_pad=1 with padding is not supported"
            )
        if not self.coreml:
            return self._pool(node, a, x, self.b.average_pool2d)
        inp = x()
        self._require_2d(node, inp)
        in_hw = self.shape(inp)[2:]
        kernel = list(a["kernel_shape"])
        strides, dilations, pads = self._spatial(node, a, kernel, in_hw)
        if not any(pads):
            return self._pool(node, a, x, self.b.average_pool2d)
        if a.get("ceil_mode", 0):
            raise WebnnLoweringError(
                "AveragePool: ceil_mode=1 with padding is not supported on Core ML"
            )
        # rustnn 0.5.12's Core ML averagePool2d counts padding in the divisor
        # (unlike WebNN / ONNX count_include_pad=0). Zero-pad explicitly, pool
        # unpadded, then rescale by kernel taps / valid taps per output
        # position: exact, and backend-independent.
        top, bottom, left, right = pads
        padded = self._pad_without_mil_pad(
            inp, [0, 0, top, left, 0, 0, bottom, right], "constant", 0.0
        )
        out = self.b.average_pool2d(
            padded,
            window_dimensions=kernel,
            strides=strides,
            dilations=dilations,
            pads=[0, 0, 0, 0],
        )
        out_hw = self.shape(out)[2:]
        counts = []
        for i, pb in enumerate((top, left)):
            pos = (
                np.arange(out_hw[i])[:, None] * strides[i]
                - pb
                + np.arange(kernel[i])[None, :] * dilations[i]
            )
            counts.append(((pos >= 0) & (pos < in_hw[i])).sum(axis=1))
        valid = np.outer(counts[0], counts[1]).astype(np.float64)
        scale = (kernel[0] * kernel[1] / valid).astype(np.float32)
        return self.b.mul(out, self.b.constant(scale.reshape(1, 1, *out_hw)))

    def _op_GlobalAveragePool(self, node, a, x):
        inp = x()
        self._require_2d(node, inp)
        return self.b.global_average_pool(inp)

    def _op_GlobalMaxPool(self, node, a, x):
        inp = x()
        self._require_2d(node, inp)
        return self.b.global_max_pool(inp)

    def _op_BatchNormalization(self, node, a, x):
        if len([o for o in node.output if o]) > 1 or a.get("training_mode", 0):
            raise WebnnLoweringError(
                "BatchNormalization: only inference mode is supported"
            )
        return self.b.batch_normalization(
            x(0),
            x(3),
            x(4),
            scale=x(1),
            bias=x(2),
            epsilon=a.get("epsilon", 1e-5),
            axis=1,
        )

    def _op_InstanceNormalization(self, node, a, x):
        inp = x(0)
        self._require_2d(node, inp)
        return self.b.instance_normalization(
            inp, scale=x(1), bias=x(2), epsilon=a.get("epsilon", 1e-5)
        )

    def _op_LayerNormalization(self, node, a, x):
        self._reject_on_coreml(node, "rustnn 0.5.12 computes wrong values")
        if len([o for o in node.output if o]) > 1:
            raise WebnnLoweringError(
                "LayerNormalization: Mean/InvStdDev outputs are not supported"
            )
        inp = x(0)
        rank = len(inp.shape)
        axis = a.get("axis", -1) % rank
        bias = x(2) if len(node.input) > 2 and node.input[2] else None
        # WebNN wants scale/bias shaped like the normalized dims exactly;
        # ONNX lets them broadcast, so expand constants up front.
        norm_shape = self.shape(inp)[axis:]
        scale = self._broadcast_const(node.input[1], norm_shape)
        if bias is not None:
            bias = self._broadcast_const(node.input[2], norm_shape)
        return self.b.layer_normalization(
            inp,
            scale=scale,
            bias=bias,
            epsilon=a.get("epsilon", 1e-5),
            axes=list(range(axis, rank)),
        )

    def _broadcast_const(self, name: str, shape: List[int]):
        if name in self.consts and list(self.consts[name].shape) != shape:
            return self.b.constant(
                np.ascontiguousarray(np.broadcast_to(self.consts[name], shape))
            )
        return self.operand(name)

    def _op_Reshape(self, node, a, x):
        inp = x()
        target = [int(v) for v in self.const(node, 1, "shape")]
        in_shape = self.shape(inp)
        if not a.get("allowzero", 0):
            target = [in_shape[i] if v == 0 else v for i, v in enumerate(target)]
        if -1 in target:
            known = int(np.prod([v for v in target if v != -1]))
            target[target.index(-1)] = int(np.prod(in_shape)) // max(known, 1)
        return self.b.reshape(inp, target)

    def _op_Flatten(self, node, a, x):
        inp = x()
        shape = self.shape(inp)
        axis = a.get("axis", 1) % (len(shape) + 1) if shape else 0
        return self.b.reshape(
            inp, [int(np.prod(shape[:axis])), int(np.prod(shape[axis:]))]
        )

    def _op_Transpose(self, node, a, x):
        inp = x()
        perm = a.get("perm", list(reversed(range(len(inp.shape)))))
        return self.b.transpose(inp, permutation=list(perm))

    def _op_Concat(self, node, a, x):
        ins = [self.operand(n) for n in node.input if n]
        return self.b.concat(ins, a["axis"] % len(ins[0].shape))

    def _axes(self, node, a, index):
        axes = a.get("axes")
        if axes is None:
            c = self.const(node, index, "axes")
            axes = None if c is None else [int(v) for v in c]
        return axes

    def _op_Squeeze(self, node, a, x):
        inp = x()
        shape = self.shape(inp)
        axes = self._axes(node, a, 1)
        axes = (
            [ax % len(shape) for ax in axes]
            if axes
            else [i for i, d in enumerate(shape) if d == 1]
        )
        return self.b.reshape(inp, [d for i, d in enumerate(shape) if i not in axes])

    def _op_Unsqueeze(self, node, a, x):
        inp = x()
        shape = self.shape(inp)
        rank = len(shape) + len(self._axes(node, a, 1))
        axes = sorted(ax % rank for ax in self._axes(node, a, 1))
        out, it = [], iter(shape)
        for i in range(rank):
            out.append(1 if i in axes else next(it))
        return self.b.reshape(inp, out)

    def _op_Expand(self, node, a, x):
        inp = x()
        target = [int(v) for v in self.const(node, 1, "shape")]
        out = list(np.broadcast_shapes(tuple(self.shape(inp)), tuple(target)))
        return self.b.expand(inp, out)

    def _op_Slice(self, node, a, x):
        inp = x()
        shape = self.shape(inp)
        starts = [int(v) for v in self.const(node, 1, "starts")]
        ends = [int(v) for v in self.const(node, 2, "ends")]
        c = self.const(node, 3, "axes")
        axes = (
            [int(v) % len(shape) for v in c]
            if c is not None
            else list(range(len(starts)))
        )
        c = self.const(node, 4, "steps")
        steps = [int(v) for v in c] if c is not None else [1] * len(starts)
        full_starts, sizes, strides = [0] * len(shape), list(shape), [1] * len(shape)
        for ax, s, e, st in zip(axes, starts, ends, steps):
            if st != 1:
                self._reject_on_coreml(node, "rustnn ignores slice strides there")
            if st <= 0:
                raise WebnnLoweringError(
                    "Slice: negative/zero steps are not supported by WebNN"
                )
            dim = shape[ax]
            s = min(max(s + dim if s < 0 else s, 0), dim)
            e = min(max(e + dim if e < 0 else e, 0), dim)
            # WebNN's size is the window span; the output holds ceil(size / stride).
            full_starts[ax], sizes[ax], strides[ax] = s, max(0, e - s), st
        return self.b.slice(inp, full_starts, sizes, strides=strides)

    def _op_Split(self, node, a, x):
        inp = x()
        shape = self.shape(inp)
        axis = a.get("axis", 0) % len(shape)
        split = a.get("split")
        if split is None:
            c = self.const(node, 1, "split")
            split = [int(v) for v in c] if c is not None else None
        if split is None:
            n = a.get("num_outputs", len(node.output))
            chunk = -(-shape[axis] // n)
            split = [chunk] * (n - 1) + [shape[axis] - chunk * (n - 1)]
        return self.b.split(inp, [int(s) for s in split], axis=axis)

    def _op_Gather(self, node, a, x):
        inp = x(0)
        axis = a.get("axis", 0) % len(inp.shape)
        name = node.input[1]
        if name in self.consts:
            idx = self.consts[name].astype(np.int64)
            idx = np.where(idx < 0, idx + self.shape(inp)[axis], idx)
            indices = self.b.constant(np.ascontiguousarray(idx.astype(np.int32)))
        else:
            indices = self.operand(name)
        return self.b.gather(inp, indices, axis=axis)

    def _op_Pad(self, node, a, x):
        inp = x()
        rank = len(inp.shape)
        pads = a.get("pads")
        if pads is None:
            pads = [int(v) for v in self.const(node, 1, "pads")]
        c = self.const(node, 3, "axes")
        if c is not None:
            full = [0] * (2 * rank)
            axes = [int(v) % rank for v in c]
            for i, ax in enumerate(axes):
                full[ax], full[ax + rank] = pads[i], pads[i + len(axes)]
            pads = full
        if any(p < 0 for p in pads):
            raise WebnnLoweringError("Pad: negative pads are not supported by WebNN")
        mode = a.get("mode", b"constant")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        mode = {"constant": "constant", "edge": "edge", "reflect": "reflection"}.get(
            mode
        )
        if mode is None:
            raise WebnnLoweringError(
                "Pad: only constant/edge/reflect modes are supported"
            )
        value = self.const(node, 2, "constant_value")
        fill = float(value.reshape(-1)[0]) if value is not None and value.size else 0.0
        if self.coreml:
            return self._pad_without_mil_pad(inp, [int(p) for p in pads], mode, fill)
        kwargs = {"mode": mode}
        if value is not None and value.size:
            kwargs["value"] = fill
        # pywebnn takes ONNX's own [begin_0.., end_0..] layout as one list.
        return self.b.pad(inp, [int(p) for p in pads], **kwargs)

    def _pad_without_mil_pad(self, inp, pads, mode: str, fill: float):
        """Pad without WebNN's ``pad``: rustnn 0.5.12's Core ML backend emits
        MIL ``pad`` with no ``mode``, which Core ML refuses to load. Exact
        rewrite, one padded axis at a time, as a ``concat`` along that axis:
        constant pads add constant blocks; edge/reflection pads add unit-
        stride slices of the input following numpy's ``edge``/``reflect``
        index maps (ONNX's), consecutive indices merged into one slice.
        (Only unit strides: rustnn ignores slice strides on Core ML, and
        chained ``gather`` on different axes mis-infers ranks.)"""
        rank = len(inp.shape)
        out = inp
        for ax in range(rank):
            before, after = pads[ax], pads[ax + rank]
            if not before and not after:
                continue
            shape = self.shape(out)
            n = shape[ax]
            if mode == "constant":
                ins = []
                for k in (before, None, after):
                    if k is None:
                        ins.append(out)
                    elif k:
                        blk = list(shape)
                        blk[ax] = k
                        ins.append(
                            self.b.constant(np.full(blk, fill, dtype=np.float32))
                        )
                out = self.b.concat(ins, ax)
                continue
            if mode == "reflection" and (before >= n or after >= n):
                raise WebnnLoweringError(
                    "Pad: reflect pads must be smaller than the padded dimension"
                )
            idx = np.pad(
                np.arange(n),
                (before, after),
                mode="edge" if mode == "edge" else "reflect",
            ).tolist()
            runs: List[List[int]] = []  # [start, length] of ascending index runs
            for i in idx:
                if runs and runs[-1][0] + runs[-1][1] == i:
                    runs[-1][1] += 1
                else:
                    runs.append([i, 1])
            ins = []
            for start, length in runs:
                if start == 0 and length == n:
                    ins.append(out)
                    continue
                starts = [0] * rank
                sizes = list(shape)
                starts[ax], sizes[ax] = start, length
                ins.append(self.b.slice(out, starts, sizes))
            out = self.b.concat(ins, ax) if len(ins) > 1 else ins[0]
        return out

    def _arg(self, node, a, x, fn):
        if a.get("select_last_index", 0):
            raise WebnnLoweringError(
                f"{node.op_type}: select_last_index=1 is not supported"
            )
        inp = x()
        return fn(
            inp,
            a.get("axis", 0) % len(inp.shape),
            keep_dimensions=bool(a.get("keepdims", 1)),
            # Core ML has no int64 tensors; RustnnSession.run casts back.
            output_data_type="int32" if self.coreml else "int64",
        )

    def _op_ArgMax(self, node, a, x):
        return self._arg(node, a, x, self.b.arg_max)

    def _op_ArgMin(self, node, a, x):
        return self._arg(node, a, x, self.b.arg_min)


def _is_coreml(context) -> bool:
    try:
        return context.backend_info().get("backend") == "coreml"
    except Exception:
        return False


def _build(
    context,
    model: onnx.ModelProto,
    input_shapes: Optional[Mapping[str, Sequence[int]]],
) -> Tuple[Any, Dict[str, List[int]]]:
    """:func:`build_webnn_graph`, plus the logical shape of each graph output
    the Core ML path flattened (see below)."""
    coreml = _is_coreml(context)
    builder = context.create_graph_builder()
    # Batch-matmul einsums become Transpose/MatMul, which WebNN has kernels for;
    # anything left over still raises its usual unsupported-op error.
    model = decompose_einsum(model)
    lowering = _Lowering(builder, model.graph, input_shapes or {}, coreml=coreml)
    for node in model.graph.node:
        try:
            lowering.lower(node)
        except WebnnLoweringError:
            raise
        except (
            Exception
        ) as e:  # pywebnn validation errors are plain RuntimeError/ValueError
            label = node.name or (node.output[0] if node.output else node.op_type)
            raise WebnnLoweringError(
                f"rustnn rejected {node.op_type} node {label!r}: {e}"
            ) from e
    outputs, shapes = {}, {}
    for vi in model.graph.output:
        op = lowering.operand(vi.name)
        # WebNN graph outputs must be computed operands, not inputs/constants.
        if vi.name in lowering.consts or vi.name in lowering.input_dtypes:
            op = builder.identity(op)
        if coreml and op.data_type not in ("float32", "float16"):
            # rustnn 0.5.12 only recognizes type code 3 for Int32, but Core
            # ML reports MLMultiArrayDataTypeInt32 as 0x20020, so int outputs
            # fall into its Float32 branch and read back as ~0. Return them
            # as float32 instead; RustnnSession.run casts back to the ONNX
            # dtype (exact for integers up to 2**24, or 2048 if Core ML
            # computes the cast in float16).
            op = builder.cast(op, "float32")
        if coreml and len(op.shape) > 1:
            # rustnn 0.5.12 reads Core ML float32 outputs as contiguous and
            # ignores MLMultiArray.strides, but ANE-produced outputs have
            # 64-byte-aligned rows. A 1-D output has no row padding, so
            # flatten here and let RustnnSession.run reshape it back.
            shapes[vi.name] = _Lowering.shape(op)
            op = builder.reshape(op, [int(np.prod(shapes[vi.name]))])
        outputs[vi.name] = op
    return builder.build(outputs), shapes


def build_webnn_graph(
    context,
    model: onnx.ModelProto,
    input_shapes: Optional[Mapping[str, Sequence[int]]] = None,
):
    """Lowers ``model``'s main graph onto a fresh ``MLGraphBuilder`` from
    ``context`` (a ``webnn.MLContext``) and builds it.

    On rustnn's Core ML backend (``backend_info()["backend"] == "coreml"``,
    what pywebnn 0.5.12 picks for ``device_type="npu"``) the lowering works
    around that backend's bugs: conv/gemm biases become explicit adds,
    arg-reductions return int32, non-float outputs are returned as float32,
    ``Pad`` becomes an exact gather/concat (its MIL ``pad`` lacks ``mode``),
    and every output is flattened to 1-D (use :class:`RustnnSession`, which
    reshapes and casts them back). Ops it computes wrongly there (``Where``,
    ``LayerNormalization``, strided ``Slice``) raise
    :class:`WebnnLoweringError` instead.

    :param input_shapes: static shapes for graph inputs whose ONNX shape has
            symbolic dimensions.
    :raises WebnnLoweringError: an op, dtype, or attribute combination WebNN
            can't express, or a shape-like input that isn't constant.
    :returns: the built ``webnn.MLGraph``.
    """
    return _build(context, model, input_shapes)[0]


@dataclass(frozen=True)
class RustnnTiming:
    """Wall-clock latency of one ``MLContext.compute`` call, host-to-host
    (it includes copying inputs in and outputs out, like the browser-side
    ``performance.now()`` timings ``webgpu_kernel_tuner.mjs`` reports).
    """

    median_ms: float
    min_ms: float
    runs: int


def _time_calls(
    fn: Callable[[], Any], warmup: int, runs: int
) -> Tuple[RustnnTiming, Any]:
    out = None
    for _ in range(warmup):
        out = fn()
    samples = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        out = fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return RustnnTiming(statistics.median(samples), min(samples), len(samples)), out


class RustnnSession:
    """An ONNX model lowered to a WebNN graph on one rustnn ``MLContext``.

    :param model: the ONNX model (or path); see :func:`build_webnn_graph`
            for what it must satisfy.
    :param device_type: WebNN device hint: ``"auto"``, ``"cpu"``, ``"gpu"``
            or ``"npu"``.
    :param backend: rustnn backend: ``"auto"``, ``"onnx"``, ``"trtx"``,
            ``"coreml"``, ``"litert"`` or ``"cann"`` (the non-ONNX-Runtime
            ones need a pywebnn built with that feature).
    :param power_preference: ``"default"``, ``"high-performance"`` or
            ``"low-power"``.
    """

    def __init__(
        self,
        model: Union[str, onnx.ModelProto],
        *,
        device_type: str = "auto",
        backend: str = "auto",
        power_preference: str = "default",
        input_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    ):
        if isinstance(model, str):
            model = onnx.load(model)
        self.model = model
        self.context = _create_context(device_type, backend, power_preference)
        self.graph, self._output_shapes = _build(self.context, model, input_shapes)
        self._output_dtypes = {
            vi.name: onnx.helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
            for vi in model.graph.output
            if vi.type.tensor_type.elem_type
        }
        inits = {i.name for i in model.graph.initializer}
        self.input_names = [vi.name for vi in model.graph.input if vi.name not in inits]
        self.output_names = [vi.name for vi in model.graph.output]

    def backend_info(self) -> Dict[str, Any]:
        """rustnn's own report of the backend it actually selected."""
        return dict(self.context.backend_info())

    def run(self, feeds: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Runs the graph once; outputs are keyed by ONNX output name and cast
        back to the ONNX-declared dtype (pywebnn returns float64/uint8 for
        some outputs)."""
        result = self.context.compute(
            self.graph, {k: np.ascontiguousarray(v) for k, v in feeds.items()}
        )
        out = {}
        for name in self.output_names:
            # Undo the Core ML path's output flattening (a no-op otherwise).
            arr = np.asarray(result[name])
            if name in self._output_shapes:
                arr = arr.reshape(self._output_shapes[name])
            dtype = self._output_dtypes.get(name)
            out[name] = arr.astype(dtype, copy=False) if dtype is not None else arr
        return out

    def benchmark(
        self, feeds: Mapping[str, np.ndarray], warmup: int = 2, runs: int = 10
    ) -> Tuple[RustnnTiming, Dict[str, np.ndarray]]:
        """Times :meth:`run`; returns the timing and the last run's outputs."""
        feeds = {k: np.ascontiguousarray(v) for k, v in feeds.items()}
        return _time_calls(lambda: self.run(feeds), warmup, runs)


@functools.lru_cache(maxsize=None)
def probe_rustnn(device_type: str = "auto", backend: str = "auto") -> Tuple[bool, str]:
    """Builds and runs a ``relu(x + 1)`` canary on the requested rustnn
    context: ``(True, "")`` when that works, else ``(False, reason)`` -- a
    missing pywebnn, a backend feature pywebnn wasn't built with, or a
    backend library (e.g. ONNX Runtime's shared library) it can't load.
    """
    try:
        ctx = _create_context(device_type, backend)
        b = ctx.create_graph_builder()
        x = b.input("x", [2, 2], "float32")
        y = b.relu(b.add(x, b.constant(np.ones((2, 2), np.float32))))
        graph = b.build({"y": y})
        got = np.asarray(
            ctx.compute(graph, {"x": np.array([[-2, -1], [0, 1]], np.float32)})["y"]
        )
        if not np.allclose(got, [[0, 0], [1, 2]]):
            return False, f"canary produced wrong values: {got.tolist()}"
    except Exception as e:  # anything here means "unusable", with a reason
        return False, f"{type(e).__name__}: {e}"
    return True, ""
