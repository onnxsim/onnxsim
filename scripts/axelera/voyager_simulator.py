#!/usr/bin/env python3
"""A partition *estimate* for Axelera's Voyager SDK Metis AIPU compiler, from
its own published per-operator support docs (`voyager_ops.py`) -- no
Docker/device/compiler involved. Shaped like `scripts/axera/pulsar2_
simulator.py`'s `partition()`/`coverage()`, but this one can go one level
deeper than op-type membership for many operators: Voyager SDK's docs give
formal `rule`/`allow_config` predicates for most "Constrained" ops (unlike
Axera's Pulsar2, which only publishes an op-type list -- see
`pulsar2_ops.py`'s docstring), so `evaluate_constraints()` attempts to check
those predicates against a node's actual (statically-known) shapes and
constant inputs.

**This is an estimate, from docs alone, with no execution behind it at
all** -- read this before trusting any of its output:

- **No compiler, no hardware, no numeric check.** Unlike `scripts/axera/
  pulsar2_simulator.py`, there is no `simulate()`/numeric-comparison
  function here at all, and this was never run against Voyager SDK's real
  compiler. `deploy.py` needs `axelera-types`/`axelera-runtime`, proprietary
  packages served only from Axelera's own private `axelera_runtime` package
  index (see `installer_support.py` in a voyager-sdk checkout) -- not on
  PyPI, not in the public git repo, and this environment has no credentials
  for it. So there is no way to actually run Voyager SDK's compiler here,
  not even its no-hardware `--pipe=quantized` calibration/quantization path.
  Nothing below was checked against real compiler output.
- **`evaluate_constraints()` is best-effort and fails closed.** It resolves
  a rule/allow_config expression's referenced names (a node's input shapes,
  constness, and constant values; its attributes; opset-17 ONNX-spec
  attribute defaults for the handful where one is unambiguous, e.g. Conv's
  `auto_pad` defaulting to `"NOTSET"`) against a small per-op parameter
  table below, then evaluates the expression with Python's `eval` in a
  restricted namespace. Anything it can't confidently resolve -- a dynamic
  shape, a non-constant tensor referenced by value, an op not in the table,
  any exception during evaluation -- comes back as `"unknown"`, never a
  guessed True/False. It also does not "fix" apparent oddities in Axelera's
  own constraint DSL (see `voyager_ops.py`'s docstring re: `B.shape==(0)`);
  it evaluates exactly the text the docs publish.
- **Only opset 17 is covered** (Voyager SDK's own recommended default for
  ONNX export -- see `scrape_onnx_support_docs.py`'s docstring for the
  cross-opset check backing this).
- **Attribute-less op-type partitioning (`partition()`) is the part you can
  trust most**: it's a direct, literal reading of the "Supported operators"
  table in `docs/reference/compiler/onnx-support.md`.

Use this to get a rough first read (does this graph look AIPU-friendly? did
onnxsim's simplification plausibly help or hurt that?) before ever trusting
Voyager SDK's own `deploy.py` output, which is authoritative and this is
not.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import onnx
from onnx import numpy_helper, shape_inference
from voyager_ops import (
    VOYAGER_OP_LEVEL,
    VOYAGER_OP_SUPPORT,
    VOYAGER_PROSE_ONLY_CONSTRAINTS,
)

#: ONNX-spec attribute defaults that are unambiguous constants (not
#: shape/rank-dependent, unlike e.g. `dilations`/`strides`, which the docs'
#: own rules already handle via explicit `is None` checks -- see this
#: module's docstring). Per (op_type, attr_name).
_ONNX_ATTR_DEFAULTS: Dict[Tuple[str, str], Any] = {
    ("Conv", "auto_pad"): "NOTSET",
    ("Conv", "group"): 1,
    ("ConvTranspose", "auto_pad"): "NOTSET",
    ("ConvTranspose", "group"): 1,
    ("AveragePool", "auto_pad"): "NOTSET",
    ("AveragePool", "count_include_pad"): 0,
    ("MaxPool", "auto_pad"): "NOTSET",
    ("MaxPool", "storage_order"): 0,
    ("Gemm", "transA"): 0,
    ("Split", "axis"): 0,
    ("Reshape", "allowzero"): 0,
}

# Per op_type: ordered tensor-input names as ONNX declares them, and each
# formal parameter's binding kind -- 'shape' (bind a _ShapeProps proxy),
# 'value' (bind the tensor's resolved constant value), or 'attr' (bind the
# node attribute's value). 'Y' is a synthetic name bound to the node's own
# (single) output. Built from Voyager SDK's own opset-17 doc pages for each
# op's **Parameters** list and **AIPU Acceleration Constraints** block --
# see this module's docstring for the accuracy caveats.
_OP_PARAM_SPEC: Dict[str, Dict[str, tuple]] = {
    "Add": {"A": ("input", "shape"), "B": ("input", "shape")},
    "Mul": {"A": ("input", "shape"), "B": ("input", "shape")},
    "Sub": {"A": ("input", "shape"), "B": ("input", "shape")},
    "Clip": {"min": ("input", "value"), "max": ("input", "value")},
    "Concat": {"axis": ("attr", None)},  # 'inputs' handled specially
    "Conv": {
        "W": ("input", "shape"),
        "auto_pad": ("attr", None),
        "group": ("attr", None),
        "dilations": ("attr", None),
        "strides": ("attr", None),
    },
    "AveragePool": {
        "auto_pad": ("attr", None),
        "pads": ("attr", None),
        "kernel_shape": ("attr", None),
        "count_include_pad": ("attr", None),
    },
    "MaxPool": {"auto_pad": ("attr", None), "storage_order": ("attr", None)},
    "Gemm": {"transA": ("attr", None)},
    "HardSigmoid": {"alpha": ("attr", None), "beta": ("attr", None)},
    "Selu": {"alpha": ("attr", None), "gamma": ("attr", None)},
    "Pad": {
        "data": ("input", "shape"),
        "pads": ("input", "value"),
        "constant_value": ("input", "value"),
        "mode": ("attr", None),
    },
    "Resize": {
        "X": ("input", "shape"),
        "Y": ("output", "shape"),
        "roi": ("input", "value"),
        "mode": ("attr", None),
        "coordinate_transformation_mode": ("attr", None),
        "nearest_mode": ("attr", None),
    },
    "Slice": {
        "data": ("input", "shape"),
        "starts": ("input", "value"),
        "ends": ("input", "value"),
        "axes": ("input", "value"),
        "steps": ("input", "value"),
    },
    "Split": {"axis": ("attr", None)},
    "Transpose": {"perm": ("attr", None)},
    "ConvTranspose": {
        "auto_pad": ("attr", None),
        "group": ("attr", None),
        "pads": ("attr", None),
        "output_shape": ("attr", None),
    },
    "Reshape": {
        "data": ("input", "shape"),
        "shape": ("input", "value"),
        "allowzero": ("attr", None),
    },
    "PRelu": {"X": ("input", "shape"), "slope": ("input", "shape")},
}

# ONNX input name -> positional index, for ops whose rules/allow_config
# reference an input by name (needed to look up node.input[i]). Only ops
# with a 'value' or 'shape' input binding above need an entry here.
_OP_INPUT_INDEX: Dict[str, Dict[str, int]] = {
    "Add": {"A": 0, "B": 1},
    "Mul": {"A": 0, "B": 1},
    "Sub": {"A": 0, "B": 1},
    "Clip": {"input": 0, "min": 1, "max": 2},
    "Conv": {"X": 0, "W": 1, "B": 2},
    "Pad": {"data": 0, "pads": 1, "constant_value": 2},
    "Resize": {"X": 0, "roi": 1, "scales": 2, "sizes": 3},
    "Slice": {"data": 0, "starts": 1, "ends": 2, "axes": 3, "steps": 4},
    "Reshape": {"data": 0, "shape": 1},
    "PRelu": {"X": 0, "slope": 1},
}


@dataclass
class _ShapeProps:
    shape: Optional[Tuple[int, ...]]
    is_constant: bool
    size: Optional[int]


class _Unresolvable(Exception):
    """A name the expression needs could not be statically resolved."""


def _onnx_attr_native(node: onnx.NodeProto, name: str, default: Any) -> Any:
    for attr in node.attribute:
        if attr.name != name:
            continue
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        if attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        if attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        if attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        if attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode("utf-8")
        if attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode("utf-8") for s in attr.strings]
        raise _Unresolvable(f"unsupported attribute type for {name!r}")
    return default


def _native_constant(value) -> Any:
    """A numpy array/scalar as a plain Python value (list/int/float/str),
    so it behaves under ==, %, indexing, len() the way the DSL expects."""
    arr = value
    if hasattr(arr, "tolist"):
        arr = arr.tolist()
    return arr


class _ShapeCache:
    """Statically-known 4D+ shapes and constant-ness, via one `onnx.shape_
    inference` pass over the whole model, cached for the model's lifetime.
    """

    def __init__(self, model: onnx.ModelProto):
        self._graph = model.graph
        self._initializers = {i.name: i for i in model.graph.initializer}
        self._producers = {
            out: node for node in model.graph.node for out in node.output
        }
        inferred = shape_inference.infer_shapes(model)
        self._value_info = {vi.name: vi for vi in inferred.graph.value_info}
        self._value_info.update({vi.name: vi for vi in inferred.graph.input})
        self._value_info.update({vi.name: vi for vi in inferred.graph.output})

    def shape(self, tensor_name: str) -> Optional[Tuple[int, ...]]:
        if tensor_name in self._initializers:
            return tuple(numpy_helper.to_array(self._initializers[tensor_name]).shape)
        vi = self._value_info.get(tensor_name)
        if vi is None or not vi.type.HasField("tensor_type"):
            return None
        dims = vi.type.tensor_type.shape.dim
        if not all(d.HasField("dim_value") for d in dims):
            return None
        return tuple(d.dim_value for d in dims)

    def is_constant(self, tensor_name: str) -> bool:
        if tensor_name in self._initializers:
            return True
        producer = self._producers.get(tensor_name)
        return producer is not None and producer.op_type == "Constant"

    def constant_value(self, tensor_name: str):
        if tensor_name in self._initializers:
            return numpy_helper.to_array(self._initializers[tensor_name])
        producer = self._producers.get(tensor_name)
        if producer is not None and producer.op_type == "Constant":
            for attr in producer.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
        raise _Unresolvable(f"{tensor_name!r} is not a statically-known constant")

    def shape_props(self, tensor_name: str) -> _ShapeProps:
        shape = self.shape(tensor_name)
        size = None
        if shape is not None:
            size = 1
            for d in shape:
                size *= d
        return _ShapeProps(
            shape=shape, is_constant=self.is_constant(tensor_name), size=size
        )


_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Store,  # only reachable as a comprehension target in `eval` mode
    ast.Attribute,
    ast.Subscript,
    ast.Index,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.USub,
    ast.comprehension,
    ast.ListComp,
    ast.GeneratorExp,
)

_SAFE_BUILTINS = {
    "len": len,
    "abs": abs,
    "all": all,
    "any": any,
    "min": min,
    "max": max,
    "round": round,
}


class _NpStub:
    """Just enough of `numpy` for the DSL's `np.array_equal(...)` calls."""

    @staticmethod
    def array_equal(a, b) -> bool:
        return list(a) == list(b)


def _check_ast_safety(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise _Unresolvable(
                f"expression uses disallowed syntax: {type(node).__name__}"
            )


def _build_namespace(
    op_type: str, node: onnx.NodeProto, cache: _ShapeCache
) -> Dict[str, Any]:
    """Every name this op's spec knows how to bind, best-effort: a name
    whose *own* resolution fails (e.g. a non-constant tensor referenced by
    value) is simply omitted rather than aborting the whole namespace --
    `eval` raising `NameError` for a *used* missing name still gets caught
    by `evaluate_expr`, but an *unused* one no longer needs to resolve at
    all. Free variables Python itself scopes (a comprehension's `for x in
    ...` target, `np`, and everything in `_SAFE_BUILTINS`) are handled by
    `eval`'s own namespace chain, not listed here.
    """
    spec = _OP_PARAM_SPEC.get(op_type, {})
    input_index = _OP_INPUT_INDEX.get(op_type, {})
    ns: Dict[str, Any] = {}

    if op_type == "Concat":
        ns["inputs"] = [cache.shape_props(t) for t in node.input]

    for name, (kind, sub) in spec.items():
        try:
            if kind == "attr":
                default = _ONNX_ATTR_DEFAULTS.get((op_type, name))
                ns[name] = _onnx_attr_native(node, name, default)
            elif kind == "output":
                if len(node.output) != 1:
                    continue
                ns[name] = cache.shape_props(node.output[0])
            elif kind == "input":
                if name not in input_index:
                    continue
                idx = input_index[name]
                if idx >= len(node.input) or not node.input[idx]:
                    ns[name] = None  # optional input, absent
                    continue
                tensor_name = node.input[idx]
                if sub == "shape":
                    ns[name] = cache.shape_props(tensor_name)
                elif sub == "value":
                    ns[name] = _native_constant(cache.constant_value(tensor_name))
        except _Unresolvable:
            continue
    return ns


def evaluate_expr(
    op_type: str, node: onnx.NodeProto, cache: _ShapeCache, expr: str
) -> Optional[bool]:
    """Evaluate one `rule`/`allow_config` expression against `node`.

    Returns True/False if it could be statically resolved, or None
    ("unknown") if any referenced name/operation could not be. Never
    raises -- every failure mode collapses to None. See this module's
    docstring for why that's the only safe default.
    """
    try:
        tree = ast.parse(expr, mode="eval")
        _check_ast_safety(tree)
        ns = _build_namespace(op_type, node, cache)
        ns["np"] = _NpStub()
        code = compile(tree, "<voyager-constraint>", "eval")
        return bool(eval(code, {"__builtins__": _SAFE_BUILTINS}, ns))
    except _Unresolvable:
        return None
    except Exception:
        # Any other failure (TypeError from a None operand, IndexError from
        # a shorter-than-expected shape, ...) also means "can't tell" --
        # never surfaces as a guessed True/False.
        return None


@dataclass
class ConstraintResult:
    op_type: str
    level: str  # 'Supported' | 'Constrained' | 'CPU fallback'
    verdict: str  # 'ok' | 'violated' | 'unknown' | 'prose_only' | 'not_applicable'
    detail: List[str] = field(default_factory=list)


def evaluate_constraints(node: onnx.NodeProto, cache: _ShapeCache) -> ConstraintResult:
    """Best-effort per-node compatibility check against Voyager SDK's
    published opset-17 AIPU constraints. See this module's docstring for
    what "best-effort" means here -- this is not a substitute for running
    the real compiler.
    """
    entry = VOYAGER_OP_SUPPORT.get(node.op_type)
    if entry is None:
        return ConstraintResult(node.op_type, "CPU fallback", "not_applicable")
    if entry["level"] == "Supported":
        return ConstraintResult(node.op_type, "Supported", "ok")
    if node.op_type in VOYAGER_PROSE_ONLY_CONSTRAINTS:
        return ConstraintResult(
            node.op_type, "Constrained", "prose_only", detail=[entry["notes"] or ""]
        )

    if entry["rules"]:
        # AND semantics: every rule must hold.
        results = [evaluate_expr(node.op_type, node, cache, r) for r in entry["rules"]]
        if any(r is False for r in results):
            bad = [r for r, ok in zip(entry["rules"], results) if ok is False]
            return ConstraintResult(node.op_type, "Constrained", "violated", detail=bad)
        if any(r is None for r in results):
            return ConstraintResult(node.op_type, "Constrained", "unknown")
        return ConstraintResult(node.op_type, "Constrained", "ok")
    else:
        # OR semantics: at least one allow_config must hold.
        results = [
            evaluate_expr(node.op_type, node, cache, r) for r in entry["allow_config"]
        ]
        if any(r is True for r in results):
            return ConstraintResult(node.op_type, "Constrained", "ok")
        if any(r is None for r in results):
            return ConstraintResult(node.op_type, "Constrained", "unknown")
        return ConstraintResult(
            node.op_type, "Constrained", "violated", detail=list(entry["allow_config"])
        )


@dataclass
class Partition:
    npu_nodes: List[str]
    cpu_fallback_nodes: List[str]
    cpu_fallback_op_types: Dict[str, int] = field(default_factory=dict)

    @property
    def npu_node_fraction(self) -> float:
        total = len(self.npu_nodes) + len(self.cpu_fallback_nodes)
        return len(self.npu_nodes) / total if total else 1.0


def partition(model: onnx.ModelProto) -> Partition:
    """Op-type-only classification against `voyager_ops.VOYAGER_OP_LEVEL`:
    any op_type Voyager SDK's docs list at all (Supported or Constrained)
    counts as NPU-eligible here, matching the docs' own framing that a
    "Constrained" op "is still hardware-accelerated -- it just has
    attribute limits". Use `evaluate_constraints()` for the (best-effort,
    per-node) attribute-level check.
    """
    npu: List[str] = []
    cpu: List[str] = []
    cpu_types: Dict[str, int] = {}
    for node in model.graph.node:
        label = node.name or f"<{node.op_type}>"
        if node.op_type in VOYAGER_OP_LEVEL:
            npu.append(label)
        else:
            cpu.append(label)
            cpu_types[node.op_type] = cpu_types.get(node.op_type, 0) + 1
    return Partition(npu, cpu, cpu_types)


def coverage(model: onnx.ModelProto) -> str:
    """'full' (every node's op_type is documented AIPU-eligible), 'none', or
    'partial'."""
    p = partition(model)
    if not p.cpu_fallback_nodes:
        return "full"
    if not p.npu_nodes:
        return "none"
    return "partial"


def evaluate_all_constraints(model: onnx.ModelProto) -> List[ConstraintResult]:
    """`evaluate_constraints()` for every node in `model`, in graph order."""
    cache = _ShapeCache(model)
    return [evaluate_constraints(node, cache) for node in model.graph.node]
