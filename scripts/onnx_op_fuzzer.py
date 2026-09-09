#!/usr/bin/env python3
"""Schema-driven differential fuzzer: onnxruntime vs. onnx's ReferenceEvaluator.

The motivating gap: onnxsim falls back to ``onnx.reference.ReferenceEvaluator``
for constant folding / correctness checks when onnxruntime is not installed
(see ``onnxsim/backend.py``), but the reference evaluator does not implement
every op form onnxruntime does -- a missing dtype in some op's kernel table, a
newer opset version, an attribute combination nobody ported. Those gaps only
show up when a model happens to hit one, so this walks ONNX's own operator
schemas (``onnx.defs``) instead of waiting to trip over the next one.

For each op this script knows how to construct, it builds one minimal model
per allowed input dtype (the "dtype sweep"), and -- for a handful of ops with
enumerated string attributes (``Resize``'s ``mode``, ``Pad``'s ``mode``, ...)
-- one model per attribute value at a fixed dtype (the "attribute sweep").
Each model is run on both backends and classified:

    both_ok        -- both ran and agreed (the common, uninteresting case)
    ref_gap        -- onnxruntime ran, the reference evaluator raised
                       ***this is the thing this script is trying to find***
    ort_gap        -- the reference evaluator ran, onnxruntime raised
    both_fail      -- both raised (usually just an invalid generated case)
    value_mismatch -- both ran but produced different results
    invalid_model  -- onnx.checker rejected the generated model (a bug in this
                       script's generator, not a finding about either backend)

This is deliberately narrower than a general-purpose model fuzzer like
NNSmith (https://github.com/ise-uiuc/nnsmith, ASPLOS'23): NNSmith solves for
whole *graphs* of arbitrary shape via an SMT solver and is aimed at crashing
compilers, not at pinpointing which single-op dtype/attribute combination a
specific backend lacks. Here the shape per op is fixed by hand (see the
``_FAMILIES`` builders below) and the only things varied are dtype and, for a
few ops, one enumerated attribute -- which is exactly the axis onnxruntime
and the reference evaluator tend to diverge on.

Coverage is intentionally partial: only ops with a registered "family" below
are exercised (``--list`` prints which ops in the current opset have one).
Extending coverage means adding an op to ``_OP_TO_FAMILY`` (and a new family
function if none of the existing shapes fit) -- there is no attempt to
synthesize valid shapes/attributes generically for arbitrary schemas, since
that is precisely the hard problem tools like NNSmith build a solver for.

Runs in-process (no subprocess isolation): a Python exception from either
backend is caught and classified, but a native crash (e.g. a segfault deep in
an onnxruntime kernel) will take this script down with it. That has not been
observed in practice for the op forms exercised here, but is worth knowing
before pointing this at a wide, unfamiliar dtype/attribute matrix.

Usage:
    python scripts/onnx_op_fuzzer.py                    # sweep every known op
    python scripts/onnx_op_fuzzer.py --ops Resize Pad    # just these ops
    python scripts/onnx_op_fuzzer.py --list              # coverage, no run
    python scripts/onnx_op_fuzzer.py --output report.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnx.defs as defs
from onnx import NodeProto, TensorProto, ValueInfoProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

try:
    import onnxruntime as ort

    _HAS_ORT = True
except ImportError:
    ort = None  # type: ignore[assignment]
    _HAS_ORT = False


def _pick_ir_and_opset() -> Tuple[int, int]:
    """The (ir_version, opset) pair to build fuzz models with.

    ``onnx.defs.onnx_opset_version()``/``onnx.IR_VERSION`` are whatever the
    installed ``onnx`` package defines as latest, which routinely races ahead
    of the installed ``onnxruntime`` (a newer onnx pulled in without a
    matching onnxruntime release yet, or vice versa). Building models at an
    opset/IR ORT does not know about would fail every single case with the
    same "unsupported opset" load error, which looks exactly like a wall of
    genuine reference-evaluator gaps and would drown them out. So probe a
    trivial one-node model down from the installed maximum until onnxruntime
    actually loads it, and use that. Skipped when onnxruntime isn't
    installed, since the reference evaluator alone tracks the installed onnx
    package's own opset by construction.
    """
    max_ir, max_opset = onnx.IR_VERSION, defs.onnx_opset_version()
    if not _HAS_ORT:
        return max_ir, max_opset
    for ir in range(max_ir, max(max_ir - 3, 0), -1):
        for opset in range(max_opset, 0, -1):
            graph = helper.make_graph(
                [helper.make_node("Identity", ["x"], ["y"])],
                "probe",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            )
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=ir
            )
            try:
                ort.InferenceSession(
                    model.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                return ir, opset
            except Exception:
                continue
    return max_ir, max_opset


_IR_VERSION, _OPSET = _pick_ir_and_opset()


# ---------------------------------------------------------------------------
# dtype <-> numpy plumbing
# ---------------------------------------------------------------------------


def _np_dtype(elem_type: int):
    """``numpy`` dtype for an ONNX elem_type, or None if this host can't
    represent it (e.g. a sub-byte float type without ``ml_dtypes`` installed).
    """
    try:
        return helper.tensor_dtype_to_np_dtype(elem_type)
    except Exception:
        return None


def _random_array(elem_type: int, shape: Sequence[int], rng: np.random.RandomState):
    """Best-effort random tensor of `elem_type`/`shape`. None if this host
    cannot build one (not a finding -- just a generation limitation)."""
    if elem_type == TensorProto.STRING:
        n = int(np.prod(shape)) if len(shape) else 1
        return np.array([f"s{i}" for i in range(n)], dtype=object).reshape(shape)
    np_dtype = _np_dtype(elem_type)
    if np_dtype is None:
        return None
    try:
        if np_dtype == np.bool_:
            return (rng.rand(*shape) > 0.5) if shape else np.array(rng.rand() > 0.5)
        if np.issubdtype(np_dtype, np.complexfloating):
            real = rng.uniform(-1, 1, size=shape)
            imag = rng.uniform(-1, 1, size=shape)
            return (real + 1j * imag).astype(np_dtype)
        if np.issubdtype(np_dtype, np.integer):
            info = np.iinfo(np_dtype)
            lo, hi = max(info.min, -8), min(info.max, 8)
            return rng.randint(lo, hi + 1, size=shape).astype(np_dtype)
        # Float family, including bfloat16/float8*/float4* via ml_dtypes: these
        # report as floating-point numpy dtypes once registered.
        return (rng.uniform(-1, 1, size=shape)).astype(np_dtype)
    except Exception:
        return None


def _elem_name(elem_type: int) -> str:
    return TensorProto.DataType.Name(elem_type).lower()


# ---------------------------------------------------------------------------
# model assembly
# ---------------------------------------------------------------------------


@dataclass
class CaseGraph:
    """Everything needed to build+run one fuzz case."""

    nodes: List[NodeProto]
    inputs: List[ValueInfoProto]
    outputs: List[ValueInfoProto]
    initializers: List[Any]
    feeds: Dict[str, np.ndarray]


def _vi(name: str, elem_type: int, shape: Sequence[int]) -> ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, list(shape))


def _init(name: str, array: np.ndarray):
    return numpy_helper.from_array(np.asarray(array), name)


def _build_model(case: CaseGraph, op_type: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        case.nodes,
        f"fuzz_{op_type}",
        case.inputs,
        case.outputs,
        case.initializers,
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", _OPSET)],
        ir_version=_IR_VERSION,
    )


# ---------------------------------------------------------------------------
# op families -- each returns a CaseGraph for one op_type/dtype combination,
# or None if this family cannot build a case for that dtype/op on this host.
# Shapes are fixed by hand per family; only dtype (and, for a few ops, one
# enumerated attribute) varies. See the module docstring for why.
# ---------------------------------------------------------------------------

Family = Callable[
    [str, int, Dict[str, Any], np.random.RandomState], Optional[CaseGraph]
]


def _family_unary(op, dtype, attrs, rng):
    shape = (2, 3, 4)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    node = helper.make_node(op, ["x"], ["y"], **attrs)
    return CaseGraph(
        [node], [_vi("x", dtype, shape)], [_vi("y", dtype, shape)], [], {"x": x}
    )


def _family_binary(op, dtype, attrs, rng):
    a_shape, b_shape = (2, 3, 4), (3, 4)
    a = _random_array(dtype, a_shape, rng)
    b = _random_array(dtype, b_shape, rng)
    if a is None or b is None:
        return None
    node = helper.make_node(op, ["a", "b"], ["y"], **attrs)
    return CaseGraph(
        [node],
        [_vi("a", dtype, a_shape), _vi("b", dtype, b_shape)],
        [_vi("y", dtype, a_shape)],
        [],
        {"a": a, "b": b},
    )


def _family_pow(op, dtype, attrs, rng):
    shape = (2, 3, 4)
    base = _random_array(dtype, shape, rng)
    if base is None:
        return None
    exponent = np.full(shape, 2, dtype=np.int64)
    node = helper.make_node(op, ["base", "exp"], ["y"])
    return CaseGraph(
        [node],
        [_vi("base", dtype, shape), _vi("exp", TensorProto.INT64, shape)],
        [_vi("y", dtype, shape)],
        [],
        {"base": base, "exp": exponent},
    )


def _family_compare(op, dtype, attrs, rng):
    shape = (2, 3, 4)
    a = _random_array(dtype, shape, rng)
    b = _random_array(dtype, shape, rng)
    if a is None or b is None:
        return None
    node = helper.make_node(op, ["a", "b"], ["y"], **attrs)
    return CaseGraph(
        [node],
        [_vi("a", dtype, shape), _vi("b", dtype, shape)],
        [_vi("y", TensorProto.BOOL, shape)],
        [],
        {"a": a, "b": b},
    )


def _family_cast(op, dtype, attrs, rng):
    shape = (2, 3, 4)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    to = TensorProto.INT64 if dtype == TensorProto.FLOAT else TensorProto.FLOAT
    node = helper.make_node(op, ["x"], ["y"], to=to, **attrs)
    return CaseGraph(
        [node], [_vi("x", dtype, shape)], [_vi("y", to, shape)], [], {"x": x}
    )


def _family_reduce(op, dtype, attrs, rng):
    shape = (2, 3, 4)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    schema = defs.get_schema(op)
    has_axes_input = any(i.name == "axes" for i in schema.inputs)
    node_attrs = {"keepdims": 1, **attrs}
    if has_axes_input:
        axes = _init("axes", np.array([1], dtype=np.int64))
        inits = [axes]
        node_inputs = ["x", "axes"]
    else:
        node_attrs["axes"] = [1]
        inits = []
        node_inputs = ["x"]
    node = helper.make_node(op, node_inputs, ["y"], **node_attrs)
    out_shape = (2, 1, 4)
    return CaseGraph(
        [node], [_vi("x", dtype, shape)], [_vi("y", dtype, out_shape)], inits, {"x": x}
    )


def _family_argreduce(op, dtype, attrs, rng):
    shape = (2, 3, 4)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    node = helper.make_node(op, ["x"], ["y"], axis=1, keepdims=1, **attrs)
    return CaseGraph(
        [node],
        [_vi("x", dtype, shape)],
        [_vi("y", TensorProto.INT64, (2, 1, 4))],
        [],
        {"x": x},
    )


def _family_conv(op, dtype, attrs, rng):
    x_shape, w_shape, y_shape = (1, 3, 8, 8), (4, 3, 3, 3), (1, 4, 8, 8)
    x = _random_array(dtype, x_shape, rng)
    w = _random_array(dtype, w_shape, rng)
    if x is None or w is None:
        return None
    node = helper.make_node(op, ["x", "w"], ["y"], pads=[1, 1, 1, 1], **attrs)
    return CaseGraph(
        [node],
        [_vi("x", dtype, x_shape)],
        [_vi("y", dtype, y_shape)],
        [_init("w", w)],
        {"x": x},
    )


def _family_pool(op, dtype, attrs, rng):
    x_shape, y_shape = (1, 3, 8, 8), (1, 3, 6, 6)
    x = _random_array(dtype, x_shape, rng)
    if x is None:
        return None
    node = helper.make_node(op, ["x"], ["y"], kernel_shape=[3, 3], **attrs)
    return CaseGraph(
        [node], [_vi("x", dtype, x_shape)], [_vi("y", dtype, y_shape)], [], {"x": x}
    )


def _family_global_pool(op, dtype, attrs, rng):
    x_shape, y_shape = (1, 3, 8, 8), (1, 3, 1, 1)
    x = _random_array(dtype, x_shape, rng)
    if x is None:
        return None
    node = helper.make_node(op, ["x"], ["y"], **attrs)
    return CaseGraph(
        [node], [_vi("x", dtype, x_shape)], [_vi("y", dtype, y_shape)], [], {"x": x}
    )


def _family_gemm(op, dtype, attrs, rng):
    a, b, c = (4, 8), (8, 16), (16,)
    a_arr, b_arr, c_arr = (
        _random_array(dtype, a, rng),
        _random_array(dtype, b, rng),
        _random_array(dtype, c, rng),
    )
    if a_arr is None or b_arr is None or c_arr is None:
        return None
    node = helper.make_node(op, ["a", "b", "c"], ["y"], **attrs)
    return CaseGraph(
        [node],
        [_vi("a", dtype, a), _vi("b", dtype, b), _vi("c", dtype, c)],
        [_vi("y", dtype, (4, 16))],
        [],
        {"a": a_arr, "b": b_arr, "c": c_arr},
    )


def _family_matmul(op, dtype, attrs, rng):
    a, b = (4, 8), (8, 16)
    a_arr, b_arr = _random_array(dtype, a, rng), _random_array(dtype, b, rng)
    if a_arr is None or b_arr is None:
        return None
    node = helper.make_node(op, ["a", "b"], ["y"], **attrs)
    return CaseGraph(
        [node],
        [_vi("a", dtype, a), _vi("b", dtype, b)],
        [_vi("y", dtype, (4, 16))],
        [],
        {"a": a_arr, "b": b_arr},
    )


def _family_topk(op, dtype, attrs, rng):
    shape = (2, 6)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    k = _init("k", np.array([2], dtype=np.int64))
    node = helper.make_node(op, ["x", "k"], ["values", "indices"], axis=-1, **attrs)
    return CaseGraph(
        [node],
        [_vi("x", dtype, shape)],
        [_vi("values", dtype, (2, 2)), _vi("indices", TensorProto.INT64, (2, 2))],
        [k],
        {"x": x},
    )


def _family_gather(op, dtype, attrs, rng):
    shape = (4, 5)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    idx = _init("idx", np.array([0, 2], dtype=np.int64))
    node = helper.make_node(op, ["x", "idx"], ["y"], axis=0, **attrs)
    return CaseGraph(
        [node], [_vi("x", dtype, shape)], [_vi("y", dtype, (2, 5))], [idx], {"x": x}
    )


def _family_gather_elements(op, dtype, attrs, rng):
    shape = (3, 4)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    idx = _init("idx", np.array([[0, 1, 2, 0]], dtype=np.int64))
    node = helper.make_node(op, ["x", "idx"], ["y"], axis=0, **attrs)
    return CaseGraph(
        [node], [_vi("x", dtype, shape)], [_vi("y", dtype, (1, 4))], [idx], {"x": x}
    )


def _family_scatter_elements(op, dtype, attrs, rng):
    shape = (3, 4)
    x = _random_array(dtype, shape, rng)
    updates = _random_array(dtype, (1, 4), rng)
    if x is None or updates is None:
        return None
    idx = _init("idx", np.array([[0, 1, 2, 0]], dtype=np.int64))
    node = helper.make_node(op, ["x", "idx", "u"], ["y"], axis=0, **attrs)
    return CaseGraph(
        [node],
        [_vi("x", dtype, shape), _vi("u", dtype, (1, 4))],
        [_vi("y", dtype, shape)],
        [idx],
        {"x": x, "u": updates},
    )


def _family_scatter_nd(op, dtype, attrs, rng):
    shape = (4, 4)
    x = _random_array(dtype, shape, rng)
    updates = _random_array(dtype, (2, 4), rng)
    if x is None or updates is None:
        return None
    idx = _init("idx", np.array([[0], [2]], dtype=np.int64))
    node = helper.make_node(op, ["x", "idx", "u"], ["y"], **attrs)
    return CaseGraph(
        [node],
        [_vi("x", dtype, shape), _vi("u", dtype, (2, 4))],
        [_vi("y", dtype, shape)],
        [idx],
        {"x": x, "u": updates},
    )


def _family_concat(op, dtype, attrs, rng):
    a_shape, b_shape = (2, 3), (2, 5)
    a, b = _random_array(dtype, a_shape, rng), _random_array(dtype, b_shape, rng)
    if a is None or b is None:
        return None
    node = helper.make_node(op, ["a", "b"], ["y"], axis=1, **attrs)
    return CaseGraph(
        [node],
        [_vi("a", dtype, a_shape), _vi("b", dtype, b_shape)],
        [_vi("y", dtype, (2, 8))],
        [],
        {"a": a, "b": b},
    )


def _family_split(op, dtype, attrs, rng):
    shape = (2, 6)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    node = helper.make_node(op, ["x"], ["y0", "y1"], axis=1, num_outputs=2, **attrs)
    return CaseGraph(
        [node],
        [_vi("x", dtype, shape)],
        [_vi("y0", dtype, (2, 3)), _vi("y1", dtype, (2, 3))],
        [],
        {"x": x},
    )


def _family_pad(op, dtype, attrs, rng):
    shape = (2, 3)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    pads = _init("pads", np.array([0, 1, 0, 1], dtype=np.int64))
    node = helper.make_node(op, ["x", "pads"], ["y"], **attrs)
    return CaseGraph(
        [node], [_vi("x", dtype, shape)], [_vi("y", dtype, (3, 4))], [pads], {"x": x}
    )


def _family_resize(op, dtype, attrs, rng):
    shape = (1, 1, 4, 4)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    scales = _init("scales", np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
    node = helper.make_node(op, ["x", "", "scales"], ["y"], **attrs)
    return CaseGraph(
        [node],
        [_vi("x", dtype, shape)],
        [_vi("y", dtype, (1, 1, 8, 8))],
        [scales],
        {"x": x},
    )


def _family_quantize_linear(op, dtype, attrs, rng):
    # Sweeps `dtype` as the *output* (quantized) type; `x`/`y_scale` stay float32.
    shape = (2, 4)
    x = _random_array(TensorProto.FLOAT, shape, rng)
    zp = _random_array(dtype, (), rng)
    if x is None or zp is None:
        return None
    scale = _init("scale", np.array(0.1, dtype=np.float32))
    zero_point = _init("zp", np.asarray(zp))
    node = helper.make_node(op, ["x", "scale", "zp"], ["y"], **attrs)
    return CaseGraph(
        [node],
        [_vi("x", TensorProto.FLOAT, shape)],
        [_vi("y", dtype, shape)],
        [scale, zero_point],
        {"x": x},
    )


def _family_dequantize_linear(op, dtype, attrs, rng):
    # Sweeps `dtype` as the *input* (quantized) type; output stays float32.
    shape = (2, 4)
    x = _random_array(dtype, shape, rng)
    zp = _random_array(dtype, (), rng)
    if x is None or zp is None:
        return None
    scale = _init("scale", np.array(0.1, dtype=np.float32))
    zero_point = _init("zp", np.asarray(zp))
    node = helper.make_node(op, ["x", "scale", "zp"], ["y"], **attrs)
    return CaseGraph(
        [node],
        [_vi("x", dtype, shape)],
        [_vi("y", TensorProto.FLOAT, shape)],
        [scale, zero_point],
        {"x": x},
    )


def _family_where(op, dtype, attrs, rng):
    shape = (2, 3)
    cond = rng.rand(*shape) > 0.5
    x, y = _random_array(dtype, shape, rng), _random_array(dtype, shape, rng)
    if x is None or y is None:
        return None
    node = helper.make_node(op, ["c", "x", "y"], ["z"], **attrs)
    return CaseGraph(
        [node],
        [
            _vi("c", TensorProto.BOOL, shape),
            _vi("x", dtype, shape),
            _vi("y", dtype, shape),
        ],
        [_vi("z", dtype, shape)],
        [],
        {"c": cond, "x": x, "y": y},
    )


def _family_shapelike(op, dtype, attrs, rng):
    shape = (2, 3, 4)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    node = helper.make_node(op, ["x"], ["y"], **attrs)
    out_shape = (len(shape),) if op == "Shape" else ()
    return CaseGraph(
        [node],
        [_vi("x", dtype, shape)],
        [_vi("y", TensorProto.INT64, out_shape)],
        [],
        {"x": x},
    )


def _family_structural_same(op, dtype, attrs, rng):
    """Ops whose output is the same dtype/shape family as the input, with a
    fixed structural (int64) side input or attribute. One function per op
    covers the different side-input shapes; see ``_STRUCTURAL_CONFIG``."""
    cfg = _STRUCTURAL_CONFIG[op]
    shape = cfg["in_shape"]
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    inits = []
    node_inputs = ["x"]
    for name, arr in cfg.get("side_inputs", {}).items():
        inits.append(_init(name, arr))
        node_inputs.append(name)
    node = helper.make_node(op, node_inputs, ["y"], **{**cfg.get("attrs", {}), **attrs})
    return CaseGraph(
        [node],
        [_vi("x", dtype, shape)],
        [_vi("y", dtype, cfg["out_shape"])],
        inits,
        {"x": x},
    )


_STRUCTURAL_CONFIG: Dict[str, Dict[str, Any]] = {
    "Flatten": {"in_shape": (2, 3, 4), "out_shape": (2, 12), "attrs": {"axis": 1}},
    "Transpose": {"in_shape": (2, 3, 4), "out_shape": (4, 3, 2)},
    "Reshape": {
        "in_shape": (2, 3, 4),
        "out_shape": (6, 4),
        "side_inputs": {"shape": np.array([6, 4], dtype=np.int64)},
    },
    "Squeeze": {
        "in_shape": (2, 1, 4),
        "out_shape": (2, 4),
        "side_inputs": {"axes": np.array([1], dtype=np.int64)},
    },
    "Unsqueeze": {
        "in_shape": (2, 4),
        "out_shape": (2, 1, 4),
        "side_inputs": {"axes": np.array([1], dtype=np.int64)},
    },
    "Expand": {
        "in_shape": (1, 4),
        "out_shape": (3, 4),
        "side_inputs": {"shape": np.array([3, 4], dtype=np.int64)},
    },
    "Tile": {
        "in_shape": (2, 3),
        "out_shape": (2, 6),
        "side_inputs": {"repeats": np.array([1, 2], dtype=np.int64)},
    },
    "Slice": {
        "in_shape": (4, 5),
        "out_shape": (2, 3),
        "side_inputs": {
            "starts": np.array([0, 0], dtype=np.int64),
            "ends": np.array([2, 3], dtype=np.int64),
        },
    },
    "DepthToSpace": {
        "in_shape": (1, 8, 2, 2),
        "out_shape": (1, 2, 4, 4),
        "attrs": {"blocksize": 2},
    },
    "SpaceToDepth": {
        "in_shape": (1, 2, 4, 4),
        "out_shape": (1, 8, 2, 2),
        "attrs": {"blocksize": 2},
    },
}


def _family_nonzero(op, dtype, attrs, rng):
    shape = (2, 3)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    node = helper.make_node(op, ["x"], ["y"], **attrs)
    return CaseGraph(
        [node],
        [_vi("x", dtype, shape)],
        [_vi("y", TensorProto.INT64, (len(shape), None))],
        [],
        {"x": x},
    )


def _family_compress(op, dtype, attrs, rng):
    shape = (4, 3)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    cond = _init("cond", np.array([True, False, True, False]))
    node = helper.make_node(op, ["x", "cond"], ["y"], axis=0, **attrs)
    return CaseGraph(
        [node], [_vi("x", dtype, shape)], [_vi("y", dtype, (2, 3))], [cond], {"x": x}
    )


def _family_onehot(op, dtype, attrs, rng):
    values = _random_array(dtype, (2,), rng)
    if values is None:
        return None
    indices = _init("indices", np.array([0, 1, 2, 0], dtype=np.int64))
    depth = _init("depth", np.array(3, dtype=np.int64))
    vals = _init("values", values)
    node = helper.make_node(op, ["indices", "depth", "values"], ["y"], axis=-1, **attrs)
    return CaseGraph([node], [], [_vi("y", dtype, (4, 3))], [indices, depth, vals], {})


def _family_cumsum(op, dtype, attrs, rng):
    shape = (2, 4)
    x = _random_array(dtype, shape, rng)
    if x is None:
        return None
    axis = _init("axis", np.array(1, dtype=np.int64))
    node = helper.make_node(op, ["x", "axis"], ["y"], **attrs)
    return CaseGraph(
        [node], [_vi("x", dtype, shape)], [_vi("y", dtype, shape)], [axis], {"x": x}
    )


def _family_norm3(op, dtype, attrs, rng):
    """LayerNormalization / InstanceNormalization: (X, Scale, B), Y same shape."""
    shape = (2, 3, 4)
    x = _random_array(dtype, shape, rng)
    if op == "InstanceNormalization":
        param_shape = (3,)
    else:
        param_shape = (4,)
    scale = _random_array(dtype, param_shape, rng)
    bias = _random_array(dtype, param_shape, rng)
    if x is None or scale is None or bias is None:
        return None
    node = helper.make_node(op, ["x", "scale", "bias"], ["y"], **attrs)
    return CaseGraph(
        [node],
        [
            _vi("x", dtype, shape),
            _vi("scale", dtype, param_shape),
            _vi("bias", dtype, param_shape),
        ],
        [_vi("y", dtype, shape)],
        [],
        {"x": x, "scale": scale, "bias": bias},
    )


# ---------------------------------------------------------------------------
# op -> family registry, and the small set of enumerated-attribute sweeps
# ---------------------------------------------------------------------------

_OP_TO_FAMILY: Dict[str, Family] = {}


def _register(family: Family, *ops: str) -> None:
    for op in ops:
        _OP_TO_FAMILY[op] = family


_register(
    _family_unary,
    "Relu",
    "Sigmoid",
    "Tanh",
    "Abs",
    "Neg",
    "Sqrt",
    "Exp",
    "Log",
    "Floor",
    "Ceil",
    "Round",
    "Sign",
    "Reciprocal",
    "Softplus",
    "Softsign",
    "Elu",
    "LeakyRelu",
    "Selu",
    "HardSigmoid",
    "Celu",
    "Mish",
    "Not",
    "BitwiseNot",
    "Identity",
    "Erf",
    "Softmax",
    "LogSoftmax",
    "Hardmax",
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Sinh",
    "Cosh",
    "Asinh",
    "Acosh",
    "Atanh",
    "IsNaN",
)
_register(
    _family_binary,
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Max",
    "Min",
    "Mod",
    "And",
    "Or",
    "Xor",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "PRelu",
)
_register(_family_pow, "Pow")
_register(_family_compare, "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual")
_register(_family_cast, "Cast")
_register(
    _family_reduce,
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceSumSquare",
)
_register(_family_argreduce, "ArgMax", "ArgMin")
_register(_family_conv, "Conv")
_register(_family_pool, "MaxPool", "AveragePool")
_register(_family_global_pool, "GlobalAveragePool", "GlobalMaxPool")
_register(_family_gemm, "Gemm")
_register(_family_matmul, "MatMul")
_register(_family_topk, "TopK")
_register(_family_gather, "Gather")
_register(_family_gather_elements, "GatherElements")
_register(_family_scatter_elements, "ScatterElements")
_register(_family_scatter_nd, "ScatterND")
_register(_family_concat, "Concat")
_register(_family_split, "Split")
_register(_family_pad, "Pad")
_register(_family_resize, "Resize")
_register(_family_quantize_linear, "QuantizeLinear")
_register(_family_dequantize_linear, "DequantizeLinear")
_register(_family_where, "Where")
_register(_family_shapelike, "Shape", "Size")
_register(_family_structural_same, *_STRUCTURAL_CONFIG.keys())
_register(_family_nonzero, "NonZero")
_register(_family_compress, "Compress")
_register(_family_onehot, "OneHot")
_register(_family_cumsum, "CumSum")
_register(_family_norm3, "LayerNormalization", "InstanceNormalization")

# op -> {attr_name: [values to sweep, at a fixed representative dtype]}.
# Only ops where a wrong/missing value is plausibly its own reference-evaluator
# gap (distinct from the dtype sweep) are listed here; this is deliberately
# small and meant to grow as new gaps are found.
_ATTR_SWEEPS: Dict[str, Dict[str, List[Any]]] = {
    "Resize": {
        "mode": ["nearest", "linear", "cubic"],
        "coordinate_transformation_mode": [
            "half_pixel",
            "pytorch_half_pixel",
            "align_corners",
            "asymmetric",
            "tf_crop_and_resize",
        ],
        "nearest_mode": ["round_prefer_floor", "round_prefer_ceil", "floor", "ceil"],
    },
    "Pad": {"mode": ["constant", "reflect", "edge", "wrap"]},
    "ScatterElements": {"reduction": ["none", "add", "mul", "min", "max"]},
    "ScatterND": {"reduction": ["none", "add", "mul", "min", "max"]},
    "DepthToSpace": {"mode": ["DCR", "CRD"]},
    "ReduceSum": {"noop_with_empty_axes": [0, 1]},
    "Gelu": {"approximate": ["none", "tanh"]},
}
_register(_family_unary, "Gelu")


def _default_dtype(op: str) -> int:
    """A representative allowed dtype for `op`'s primary input, for the
    attribute sweep (which fixes dtype and varies one attribute instead)."""
    schema = defs.get_schema(op)
    primary = schema.inputs[0]
    for tc in schema.type_constraints:
        if tc.type_param_str == primary.type_str:
            for candidate in ("tensor(float)", "tensor(int64)"):
                if candidate in tc.allowed_type_strs:
                    return getattr(TensorProto, candidate[7:-1].upper())
            return _type_str_to_elem(tc.allowed_type_strs[0])
    raise KeyError(f"no type constraint {primary.type_str} for {op}")


def _type_str_to_elem(type_str: str) -> int:
    # "tensor(float16)" -> TensorProto.FLOAT16
    name = type_str[len("tensor(") : -1].upper()
    return getattr(TensorProto, name)


# Which type constraint to sweep, when it isn't simply the first input's (e.g.
# QuantizeLinear's interesting axis is its *output* dtype T3, not input x's T1).
_SWEEP_TYPE_OVERRIDE: Dict[str, str] = {"QuantizeLinear": "T3"}


def _swept_dtypes(op: str) -> List[int]:
    schema = defs.get_schema(op)
    type_str = _SWEEP_TYPE_OVERRIDE.get(op, schema.inputs[0].type_str)
    for tc in schema.type_constraints:
        if tc.type_param_str == type_str:
            # Every family builder here only knows about plain tensors -- skip
            # seq(tensor(...))/optional(...)/sparse_tensor(...) entries a
            # constraint like Identity's may also allow.
            return [
                _type_str_to_elem(t)
                for t in tc.allowed_type_strs
                if t.startswith("tensor(") and t.endswith(")")
            ]
    return []


# ---------------------------------------------------------------------------
# running + classifying one case
# ---------------------------------------------------------------------------


@dataclass
class Result:
    op: str
    label: str
    status: str
    detail: str = ""


def _run_reference(model: onnx.ModelProto, feeds: Dict[str, np.ndarray]):
    sess = ReferenceEvaluator(model)
    return sess.run(None, feeds)


def _run_ort(model: onnx.ModelProto, feeds: Dict[str, np.ndarray]):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _values_match(ref_out, ort_out) -> Optional[str]:
    """None if every output matches; otherwise a short reason it didn't."""
    if len(ref_out) != len(ort_out):
        return f"{len(ref_out)} reference outputs vs {len(ort_out)} onnxruntime outputs"
    for i, (r, o) in enumerate(zip(ref_out, ort_out)):
        r, o = np.asarray(r), np.asarray(o)
        if r.shape != o.shape:
            return f"output {i}: shape {r.shape} vs {o.shape}"
        if r.dtype == object or o.dtype == object or r.dtype == np.bool_:
            if not np.array_equal(r, o):
                return f"output {i}: values differ (dtype {r.dtype})"
            continue
        try:
            if not np.allclose(
                r.astype(np.float64),
                o.astype(np.float64),
                rtol=1e-2,
                atol=1e-2,
                equal_nan=True,
            ):
                diff = np.max(np.abs(r.astype(np.float64) - o.astype(np.float64)))
                return f"output {i}: max abs diff {diff:.4g}"
        except (TypeError, ValueError):
            if not np.array_equal(r, o):
                return f"output {i}: values differ (dtype {r.dtype})"
    return None


def _run_case(op: str, label: str, case: CaseGraph) -> Result:
    try:
        model = _build_model(case, op)
        onnx.checker.check_model(model, full_check=False)
    except Exception as exc:
        return Result(op, label, "invalid_model", f"{type(exc).__name__}: {exc}")

    ref_err = ort_err = None
    ref_out = ort_out = None
    try:
        ref_out = _run_reference(model, case.feeds)
    except Exception as exc:
        ref_err = f"{type(exc).__name__}: {exc}"

    if not _HAS_ORT:
        status = "ref_only_ok" if ref_err is None else "ref_only_fail"
        return Result(op, label, status, ref_err or "")

    try:
        ort_out = _run_ort(model, case.feeds)
    except Exception as exc:
        ort_err = f"{type(exc).__name__}: {exc}"

    if ref_err is None and ort_err is None:
        mismatch = _values_match(ref_out, ort_out)
        if mismatch is None:
            return Result(op, label, "both_ok")
        return Result(op, label, "value_mismatch", mismatch)
    if ref_err is not None and ort_err is None:
        return Result(op, label, "ref_gap", ref_err)
    if ref_err is None and ort_err is not None:
        return Result(op, label, "ort_gap", ort_err)
    return Result(op, label, "both_fail", f"ref={ref_err} | ort={ort_err}")


# ---------------------------------------------------------------------------
# driving the sweep
# ---------------------------------------------------------------------------


def iter_cases(op: str, seed: int) -> List[Tuple[str, CaseGraph]]:
    """All (label, CaseGraph) pairs for `op`: the dtype sweep, plus the
    attribute sweep if one is registered."""
    family = _OP_TO_FAMILY[op]
    rng = np.random.RandomState(seed)
    out: List[Tuple[str, CaseGraph]] = []
    for dtype in _swept_dtypes(op):
        case = family(op, dtype, {}, rng)
        if case is not None:
            out.append((f"dtype={_elem_name(dtype)}", case))
    attr_sweep = _ATTR_SWEEPS.get(op)
    if attr_sweep:
        default_dtype = _default_dtype(op)
        for attr_name, values in attr_sweep.items():
            for value in values:
                case = family(op, default_dtype, {attr_name: value}, rng)
                if case is not None:
                    out.append((f"{attr_name}={value}", case))
    return out


def known_ops() -> List[str]:
    return sorted(_OP_TO_FAMILY)


def all_default_domain_ops() -> List[str]:
    return sorted({s.name for s in defs.get_all_schemas() if s.domain == ""})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ops",
        nargs="*",
        default=None,
        help="only fuzz these ops (default: all known)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--output", default=None, help="CSV path for the full per-case report"
    )
    ap.add_argument(
        "--list", action="store_true", help="print coverage (known vs. total) and exit"
    )
    ap.add_argument(
        "--show",
        default="ref_gap",
        choices=sorted(
            {
                "ref_gap",
                "ort_gap",
                "value_mismatch",
                "both_fail",
                "invalid_model",
                "all",
            }
        ),
        help="which status to print case detail for (default: ref_gap, the interesting one)",
    )
    args = ap.parse_args()

    total_ops = all_default_domain_ops()
    if args.list:
        covered = known_ops()
        print(
            f"{len(covered)}/{len(total_ops)} default-domain ops have a fuzz generator."
        )
        missing = sorted(set(total_ops) - set(covered))
        print(f"\n{len(missing)} without one (add to _OP_TO_FAMILY to cover):")
        for name in missing:
            print(f"  {name}")
        return 0

    if not _HAS_ORT:
        print(
            "onnxruntime is not installed -- running the reference evaluator alone "
            "(no differential comparison; install onnxruntime to find ref_gap cases).",
            file=sys.stderr,
        )

    ops = args.ops or known_ops()
    unknown = [op for op in ops if op not in _OP_TO_FAMILY]
    if unknown:
        print(
            f"no fuzz generator for: {', '.join(unknown)} (see --list)", file=sys.stderr
        )
        ops = [op for op in ops if op in _OP_TO_FAMILY]

    rows: List[Result] = []
    for op in ops:
        for label, case in iter_cases(op, args.seed):
            rows.append(_run_case(op, label, case))

    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(
        f"{len(rows)} cases across {len(ops)} ops: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )

    show = [r for r in rows if args.show == "all" or r.status == args.show]
    if show:
        print(f"\n{args.show} ({len(show)}):")
        for r in show:
            print(f"  {r.op:24} {r.label:45} {r.detail[:160]}")

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["op", "label", "status", "detail"])
            for r in rows:
                w.writerow([r.op, r.label, r.status, r.detail])
        print(f"\nwrote {args.output} ({len(rows)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
