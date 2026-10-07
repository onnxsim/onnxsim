"""Numerical-safety linter: where can this model produce inf / nan / overflow for *some* input?

``lint(model, input_ranges, dtype)`` propagates the input box through the graph with
:mod:`onnxsim.interval` and flags operations that **can** misbehave for some input inside
the box:

==================  ===========================================================================
rule                what it flags
==================  ===========================================================================
``div-by-zero``     ``Div`` / ``Reciprocal`` whose denominator interval contains 0
``log-domain``      ``Log`` on a domain that includes <= 0
``sqrt-domain``     ``Sqrt`` on a domain that includes < 0
``pow-domain``      ``Pow`` with a constant exponent: negative base with a fractional exponent,
                    or a base that can be 0 with a negative exponent
``exp-overflow``    ``Exp`` input above ln(dtype max) (88.7 fp32/bf16, 11.09 fp16); a
                    decomposed softmax without the max-subtraction gets a tailored fix
``saturation``      ``Sigmoid`` / ``Softplus`` / ``Tanh`` driven past ln(dtype max): a naive
                    ``exp`` based implementation overflows there (implementation dependent)
``norm-variance``   ``LayerNormalization`` / ``RMSNormalization`` / ``SimplifiedLayerNormalization``
                    whose variance (or sum of squares) can overflow the dtype, or whose variance
                    can be 0 while ``epsilon`` underflows to 0 in the dtype
``range-overflow``  any float tensor whose interval exceeds the dtype's largest finite value
                    (65504 for fp16) -- reported where the overflow *arises*, not downstream
``const-overflow``  a constant that is not representable in the dtype (e.g. a ``-3.4e38`` softmax
                    mask becomes ``-inf`` in fp16)
``range-underflow`` fp16: constants that flush to 0 or go subnormal; activations that stay subnormal
``int32-wrap``      ``MatMulInteger`` / ``ConvInteger`` accumulator that can leave the int32 range
``cast-range``      ``Cast`` to a narrower integer / fp16 type whose range the input exceeds
``dead-op``         (info) ``Clip`` / ``Relu`` that never changes its input or always saturates;
                    ``Where`` whose condition is decided
==================  ===========================================================================

What a finding means -- read this before acting on one
------------------------------------------------------
* A finding is a **sound over-approximation**: *if the real inputs stay in the box and the
  interval analysis is as tight as it can be, this can happen for some input in the box*.
  It is **not** a confirmed bug (``certainty="possible"``). Plain intervals lose correlations,
  so on deep networks the hulls grow far faster than the real activations (ResNet18: the
  layer4 ReLU hull is ~1e20 for an observed maximum of ~10): most deep ``range-overflow``
  findings are looseness, not bugs.
* With ``witness=N`` the linter searches (random, corners, hill-climbing; at most ``N`` runs of
  the model in onnxruntime) for an input in the box that actually breaks the condition in the
  fp32 reference execution, and upgrades the finding to ``certainty="confirmed"`` only then. For
  fp16/bf16 the run is still fp32: "confirmed" means *the exact value is outside the dtype's
  range*, not that an fp16 kernel was executed. ``observed`` records the extreme the search saw
  (so a certified 6e8 next to an observed 80 shows how loose the bound is).
* Everything depends on the input ranges you declare (``onnxsim.ranges`` annotations or the
  ``input_ranges`` argument). An input with no range is unbounded; findings that exist only
  because of that are suppressed and counted (``suppressed_unbounded``) unless
  ``include_unbounded=True``.
* Ops :mod:`onnxsim.interval` cannot bound leave their outputs unknown and stop the analysis
  downstream; for ``Gather``, ``LayerNormalization``/``RMSNormalization``, ``Cast``, ``Gelu`` and a
  decomposed softmax the linter *cuts* the graph there instead: the op's output becomes a fresh
  input with a range derived from the op's mathematics (e.g. ``|LayerNorm(x)| <= sqrt(N)|gamma| +
  |beta|``), which is sound and lets the analysis continue.
* Not modelled: fp16 *underflow of intermediate sums*, accumulation order, implementation-specific
  fused kernels (several findings say "implementation dependent"), ``ArgMax``/``TopK`` ties, and
  anything outside real arithmetic over the declared box.

Nothing here is wired into ``simplify()``; it is opt-in (``python -m onnxsim.numeric_lint``).
"""

import argparse
import dataclasses
import json
import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from . import interval as _interval
from . import ranges as _ranges
from ._onnx_compat import INT4, UINT4

CAN_FAIL = "can-fail"
CAN_LOSE_PRECISION = "can-lose-precision"
INFO = "info"
SEVERITIES = (CAN_FAIL, CAN_LOSE_PRECISION, INFO)
POSSIBLE = "possible"
CONFIRMED = "confirmed"

_F32_MAX = 3.4028234663852886e38
_BF16_MAX = 3.3895313892515355e38
_PRECISION: Dict[str, Dict[str, float]] = {
    "fp32": dict(max=_F32_MAX, min_normal=2.0**-126, min_sub=2.0**-149),
    "fp16": dict(max=65504.0, min_normal=2.0**-14, min_sub=2.0**-24),
    "bf16": dict(max=_BF16_MAX, min_normal=2.0**-126, min_sub=2.0**-133),
}
for _p in _PRECISION.values():
    _p["exp_over"] = math.log(_p["max"])

_FLOAT_TYPES = {
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    TensorProto.BFLOAT16,
    TensorProto.DOUBLE,
}
_INT_TYPES = {
    TensorProto.INT8,
    TensorProto.UINT8,
    TensorProto.INT16,
    TensorProto.UINT16,
    TensorProto.INT32,
    TensorProto.UINT32,
    TensorProto.INT64,
    TensorProto.UINT64,
}
_CAST_RANGE: Dict[int, Tuple[float, float]] = {
    TensorProto.INT8: (-128.0, 127.0),
    TensorProto.UINT8: (0.0, 255.0),
    TensorProto.INT16: (-32768.0, 32767.0),
    TensorProto.UINT16: (0.0, 65535.0),
    TensorProto.INT32: (-(2.0**31), 2.0**31 - 1),
    TensorProto.UINT32: (0.0, 2.0**32 - 1),
    TensorProto.INT64: (-(2.0**63), 2.0**63 - 1),
    TensorProto.UINT64: (0.0, 2.0**64 - 1),
    TensorProto.FLOAT16: (-65504.0, 65504.0),
    INT4: (-8.0, 7.0),
    UINT4: (0.0, 15.0),
}
_INT32_MAX = 2**31 - 1
# constants: flushed-to-zero fraction (of the nonzero values) at which fp16 underflow is a warning
_FLUSH_FRACTION = 1e-3
_NP_FOR_TYPE: Dict[int, Any] = {
    TensorProto.FLOAT: np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.FLOAT16: np.float16,
    TensorProto.INT8: np.int8,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT16: np.int16,
    TensorProto.UINT16: np.uint16,
    TensorProto.INT32: np.int32,
    TensorProto.UINT32: np.uint32,
    TensorProto.INT64: np.int64,
    TensorProto.UINT64: np.uint64,
    TensorProto.BOOL: np.bool_,
}
_NORM_OPS = {
    "LayerNormalization",
    "RMSNormalization",
    "SimplifiedLayerNormalization",
}

# (loss of generality note) check: values -> (confirmed, progress, observed). ``progress``
# grows as the input gets closer to violating the rule; ``observed`` is the extreme seen.
Check = Callable[[Dict[str, np.ndarray]], Tuple[bool, float, float]]


@dataclasses.dataclass
class Finding:
    """One place the model can misbehave for some input in the declared box."""

    rule: str
    severity: str
    node: str
    op_type: str
    tensor: str
    message: str
    interval: Tuple[float, float]  # the interval the claim rests on
    assumption: Dict[str, Tuple[float, float]]  # input ranges the analysis assumed
    fix: str
    certainty: str = POSSIBLE
    unbounded: bool = False
    implementation_dependent: bool = False
    refined: bool = False  # survived CROWN refinement (refine=True)
    observed: Optional[float] = None  # extreme seen by the witness search
    observed_what: str = ""
    witness_runs: int = 0
    _check: Optional[Check] = dataclasses.field(default=None, repr=False, compare=False)
    _tensors: Tuple[str, ...] = dataclasses.field(default=(), repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            k: v
            for k, v in dataclasses.asdict(self).items()
            if not k.startswith("_") and k != "interval"
        }
        d["interval"] = [_jnum(self.interval[0]), _jnum(self.interval[1])]
        d["assumption"] = {
            k: [_jnum(a), _jnum(b)] for k, (a, b) in self.assumption.items()
        }
        d["observed"] = None if self.observed is None else _jnum(self.observed)
        return d

    def __str__(self) -> str:
        tag = "CONFIRMED" if self.certainty == CONFIRMED else "possible"
        impl = " (implementation dependent)" if self.implementation_dependent else ""
        obs = ""
        if self.observed is not None:
            obs = f" [witness search saw {self.observed_what} = {self.observed:.4g}]"
        return (
            f"[{self.severity}/{tag}] {self.rule} at {self.node} ({self.op_type}): "
            f"{self.message}{impl}{obs}\n    fix: {self.fix}"
        )


def _jnum(x: float) -> Any:
    return x if math.isfinite(x) else ("inf" if x > 0 else "-inf" if x < 0 else "nan")


@dataclasses.dataclass
class LintReport:
    findings: List[Finding]
    dtype: str
    assumptions: Dict[str, Tuple[float, float]]
    unannotated_inputs: List[str]
    suppressed_unbounded: int = 0
    consequences: int = (
        0  # overflow reports skipped because an input already overflowed
    )
    unanalysed: List[str] = dataclasses.field(
        default_factory=list
    )  # tensors with no bound
    cuts: int = 0
    refined_away: int = 0
    seconds: float = 0.0
    witness_runs: int = 0
    witness_note: str = ""
    witnesses: Dict[int, Dict[str, np.ndarray]] = dataclasses.field(
        default_factory=dict
    )
    _model: Any = dataclasses.field(default=None, repr=False, compare=False)
    _known: Any = dataclasses.field(default=None, repr=False, compare=False)

    def hull(self, tensor: str) -> Optional[Tuple[float, float]]:
        """The interval the linter used for ``tensor`` (None if it could not bound it)."""
        return None if self._known is None else self._known.hull(tensor)

    def by(
        self, severity: Optional[str] = None, certainty: Optional[str] = None
    ) -> List[Finding]:
        return [
            f
            for f in self.findings
            if (severity is None or f.severity == severity)
            and (certainty is None or f.certainty == certainty)
        ]

    @property
    def can_fail(self) -> List[Finding]:
        return self.by(CAN_FAIL)

    @property
    def confirmed(self) -> List[Finding]:
        return self.by(certainty=CONFIRMED)

    @property
    def ok(self) -> bool:
        """No ``can-fail`` finding under the declared ranges (a sound statement, not a guarantee of accuracy)."""
        return not self.can_fail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dtype": self.dtype,
            "ok": self.ok,
            "assumptions": {
                k: [_jnum(a), _jnum(b)] for k, (a, b) in self.assumptions.items()
            },
            "unannotated_inputs": self.unannotated_inputs,
            "counts": {s: len(self.by(s)) for s in SEVERITIES}
            | {"confirmed": len(self.confirmed)},
            "suppressed_unbounded": self.suppressed_unbounded,
            "consequences": self.consequences,
            "cuts": self.cuts,
            "refined_away": self.refined_away,
            "unanalysed_tensors": len(self.unanalysed),
            "seconds": round(self.seconds, 3),
            "witness_runs": self.witness_runs,
            "witness_note": self.witness_note,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), **kw)

    def replay(self, index: int) -> Dict[str, np.ndarray]:
        """Re-run a confirmed finding's witness input in onnxruntime; returns every tensor it needs."""
        if index not in self.witnesses:
            raise KeyError(f"finding {index} has no stored witness")
        f = self.findings[index]
        return _run_exposed(self._model, self.witnesses[index], list(f._tensors))

    def __str__(self) -> str:
        c = {s: len(self.by(s)) for s in SEVERITIES}
        head = (
            f"numeric lint ({self.dtype}): {c[CAN_FAIL]} can-fail, "
            f"{c[CAN_LOSE_PRECISION]} can-lose-precision, {c[INFO]} info"
            f"; {len(self.confirmed)} confirmed"
        )
        notes = []
        if self.unannotated_inputs:
            notes.append(
                f"inputs without a range: {', '.join(self.unannotated_inputs)}"
            )
        if self.suppressed_unbounded:
            notes.append(
                f"{self.suppressed_unbounded} findings suppressed (they rest on an unbounded input)"
            )
        if self.consequences:
            notes.append(
                f"{self.consequences} downstream overflow reports folded into earlier ones"
            )
        if self.unanalysed:
            notes.append(
                f"{len(self.unanalysed)} tensors have no bound "
                "(an op the interval analysis does not model)"
            )
        if self.witness_note:
            notes.append(self.witness_note)
        lines = [head] + [f"  note: {n}" for n in notes]
        for f in self.findings:
            if f.severity != INFO:
                lines.append(str(f))
        infos = self.by(INFO)
        if infos:
            lines.append(f"  ({len(infos)} info findings; see report.by('info'))")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Hull bookkeeping
# --------------------------------------------------------------------------


class _Known:
    """Tensor bounds merged over the cut iterations of the analysis."""

    def __init__(self) -> None:
        self.arr: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.hulls: Dict[str, Tuple[float, float]] = {}
        self._cache: Dict[str, Optional[Tuple[float, float]]] = {}

    def update(self, res: Any) -> None:
        for k, (lo, hi) in res.intervals.items():
            self.arr[k] = (np.asarray(lo), np.asarray(hi))
            self._cache.pop(k, None)
        for k, r in res.ranged.items():
            self.hulls[k] = (float(r.hull[0]), float(r.hull[1]))
            self._cache.pop(k, None)

    def hull(self, name: str) -> Optional[Tuple[float, float]]:
        if name in self._cache:
            return self._cache[name]
        h: Optional[Tuple[float, float]] = None
        if name in self.arr:
            lo, hi = self.arr[name]
            if lo.size:
                a, b = float(np.min(lo)), float(np.max(hi))
                h = (
                    -math.inf if math.isnan(a) else a,
                    math.inf if math.isnan(b) else b,
                )
            else:
                h = (0.0, 0.0)
        elif name in self.hulls:
            h = self.hulls[name]
        self._cache[name] = h
        return h

    def is_point(self, name: str) -> bool:
        if name in self.arr:
            lo, hi = self.arr[name]
            return lo.shape == hi.shape and bool(np.array_equal(lo, hi))
        h = self.hulls.get(name)
        return h is not None and h[0] == h[1]


def _finite(h: Optional[Tuple[float, float]]) -> bool:
    return h is not None and math.isfinite(h[0]) and math.isfinite(h[1])


def _unknown(h: Optional[Tuple[float, float]]) -> bool:
    """No information: not bounded at all, on either side (an op the interval analysis does not model)."""
    return h is None or (h[0] == -math.inf and h[1] == math.inf)


def _maxabs(h: Tuple[float, float]) -> float:
    return max(abs(h[0]), abs(h[1]))


# --------------------------------------------------------------------------
# Witness checks (closures: values -> (confirmed, progress, observed))
# --------------------------------------------------------------------------


def _nanmax_abs(a: np.ndarray) -> float:
    a = np.abs(np.asarray(a, dtype=np.float64))
    fin = a[np.isfinite(a)]
    return float(fin.max()) if fin.size else 0.0


def _chk_abs_gt(tensor: str, limit: float) -> Check:
    def f(v: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        a = np.asarray(v[tensor], dtype=np.float64)
        nonfin = not bool(np.isfinite(a).all())
        m = math.inf if nonfin else float(np.max(np.abs(a))) if a.size else 0.0
        return (nonfin or m > limit), m, m

    return f


def _chk_nonfinite(out: str, prog: str, mode: str) -> Check:
    """Confirmed when ``out`` holds inf/nan; progress follows the offending operand ``prog``."""

    def f(v: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        o = np.asarray(v[out], dtype=np.float64)
        conf = not bool(np.isfinite(o).all())
        x = np.asarray(v[prog], dtype=np.float64)
        if not x.size:
            return conf, -math.inf, math.nan
        if mode == "min":
            m = float(np.min(x))
            return conf, -m, m
        m = float(np.min(np.abs(x)))
        return conf, -m, m

    return f


def _chk_exp(x: str, y: str, limit: float) -> Check:
    def f(v: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        out = np.asarray(v[y], dtype=np.float64)
        inp = np.asarray(v[x], dtype=np.float64)
        conf = (not bool(np.isfinite(out).all())) or (
            bool(out.size) and float(np.max(np.abs(out))) > limit
        )
        m = float(np.max(inp)) if inp.size else -math.inf
        return conf, m, m

    return f


def _chk_outside(tensor: str, lo: float, hi: float) -> Check:
    def f(v: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        a = np.asarray(v[tensor], dtype=np.float64)
        if not a.size:
            return False, -math.inf, math.nan
        top = float(np.max(a)) - hi
        bot = lo - float(np.min(a))
        worst = max(top, bot)
        obs = float(np.max(np.abs(a)))
        return bool(worst > 0), worst, obs

    return f


def _row_var(x: np.ndarray, axis: int, kind: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    ax = tuple(range(axis % x.ndim, x.ndim))
    if kind == "rms":
        return np.mean(x * x, axis=ax)
    return np.var(x, axis=ax)


def _chk_norm_var(x: str, axis: int, kind: str, limit: float, mode: str) -> Check:
    def f(v: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        rv = _row_var(v[x], axis, kind)
        if mode == "max":
            m = float(np.max(rv)) if rv.size else 0.0
            return m > limit, m, m
        m = float(np.min(rv)) if rv.size else math.inf
        return m == 0.0, -m, m

    return f


def _chk_int_acc(a: str, b: np.ndarray, za: float, zb: float) -> Check:
    bb = np.asarray(b, dtype=np.int64) - int(zb)

    def f(v: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        aa = np.asarray(v[a], dtype=np.int64) - int(za)
        acc = np.matmul(aa, bb)
        m = float(np.max(np.abs(acc))) if acc.size else 0.0
        return m > _INT32_MAX, m, m

    return f


def _chk_small(tensor: str, limit: float) -> Check:
    def f(v: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        a = np.abs(np.asarray(v[tensor], dtype=np.float64))
        nz = a[a > 0]
        if not nz.size:
            return False, -math.inf, math.nan
        m = float(nz.min())
        return m < limit, -m, m

    return f


# --------------------------------------------------------------------------
# The analysis
# --------------------------------------------------------------------------


def _dtype_cfg(dtype: str) -> Dict[str, float]:
    if dtype not in _PRECISION:
        raise ValueError(f"dtype must be one of {sorted(_PRECISION)}, got {dtype!r}")
    return _PRECISION[dtype]


def _attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _tensor_types(
    model: onnx.ModelProto,
) -> Tuple[Dict[str, int], Dict[str, Optional[List[int]]]]:
    types: Dict[str, int] = {}
    shapes: Dict[str, Optional[List[int]]] = {}
    try:
        g = onnx.shape_inference.infer_shapes(model).graph
    except Exception:
        g = model.graph
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        tt = vi.type.tensor_type
        types[vi.name] = tt.elem_type
        if tt.HasField("shape"):
            dims = [
                d.dim_value if d.HasField("dim_value") else -1 for d in tt.shape.dim
            ]
            shapes[vi.name] = None if any(x < 0 for x in dims) else dims
        else:
            shapes[vi.name] = None
    for t in model.graph.initializer:
        types[t.name] = t.data_type
        shapes[t.name] = list(t.dims)
    return types, shapes


def _prod(xs: Sequence[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def _consts(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for t in model.graph.initializer:
        try:
            out[t.name] = numpy_helper.to_array(t)
        except Exception:
            pass
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output:
            for a in n.attribute:
                if a.name == "value":
                    out[n.output[0]] = numpy_helper.to_array(a.t)
                elif a.name == "value_float":
                    out[n.output[0]] = np.array(a.f, dtype=np.float32)
                elif a.name == "value_int":
                    out[n.output[0]] = np.array(a.i, dtype=np.int64)
                elif a.name == "value_floats":
                    out[n.output[0]] = np.array(list(a.floats), dtype=np.float32)
                elif a.name == "value_ints":
                    out[n.output[0]] = np.array(list(a.ints), dtype=np.int64)
    return out


# Ops whose output bound follows from their mathematics. Each takes (node, ctx) and returns
# {output_name: (lo, hi)} (scalars or arrays broadcastable to the output) or None.
def _cut_gather(
    node: onnx.NodeProto, c: "_CutCtx"
) -> Optional[Dict[str, Tuple[Any, Any]]]:
    data = c.consts.get(node.input[0])
    axis = int(_attrs(node).get("axis", 0))
    if data is not None and data.size:
        d = np.asarray(data, dtype=np.float64)
        ih = c.known.hull(node.input[1]) if len(node.input) > 1 else None
        if axis == 0 and d.ndim >= 1 and _finite(ih):
            assert ih is not None
            a = max(0, int(math.floor(ih[0])))
            b = min(d.shape[0] - 1, int(math.ceil(ih[1])))
            if a <= b:
                d = d[a : b + 1]
        return {node.output[0]: (float(np.min(d)), float(np.max(d)))}
    h = c.known.hull(node.input[0])
    if _finite(h):
        assert h is not None
        return {node.output[0]: h}
    return None


def _cut_layernorm(
    node: onnx.NodeProto, c: "_CutCtx"
) -> Optional[Dict[str, Tuple[Any, Any]]]:
    shape = c.shapes.get(node.input[0])
    if not shape:
        return None
    axis = int(_attrs(node).get("axis", -1)) % len(shape)
    n = _prod(shape[axis:])
    root = math.sqrt(n)
    scale = c.consts.get(node.input[1]) if len(node.input) > 1 else None
    bias = (
        c.consts.get(node.input[2]) if len(node.input) > 2 and node.input[2] else None
    )
    if scale is None:
        sh = c.known.hull(node.input[1]) if len(node.input) > 1 else (1.0, 1.0)
        if not _finite(sh):
            return None
        assert sh is not None
        g: Any = max(abs(sh[0]), abs(sh[1]))
    else:
        g = np.abs(np.asarray(scale, dtype=np.float64))
    if bias is None:
        if len(node.input) > 2 and node.input[2]:
            bh = c.known.hull(node.input[2])
            if not _finite(bh):
                return None
            assert bh is not None
            blo, bhi = bh
        else:
            blo, bhi = 0.0, 0.0
        lo: Any = blo - root * g
        hi: Any = bhi + root * g
    else:
        b = np.asarray(bias, dtype=np.float64)
        lo, hi = b - root * g, b + root * g
    return {node.output[0]: (lo, hi)}


def _cut_rmsnorm(
    node: onnx.NodeProto, c: "_CutCtx"
) -> Optional[Dict[str, Tuple[Any, Any]]]:
    shape = c.shapes.get(node.input[0])
    if not shape:
        return None
    axis = int(_attrs(node).get("axis", -1)) % len(shape)
    root = math.sqrt(_prod(shape[axis:]))
    scale = c.consts.get(node.input[1]) if len(node.input) > 1 else None
    if scale is None:
        return None
    g = np.abs(np.asarray(scale, dtype=np.float64)) * root
    return {node.output[0]: (-g, g)}


def _cut_cast(
    node: onnx.NodeProto, c: "_CutCtx"
) -> Optional[Dict[str, Tuple[Any, Any]]]:
    to = int(_attrs(node).get("to", 0))
    h = c.known.hull(node.input[0])
    if to == TensorProto.BOOL:
        return {node.output[0]: (0.0, 1.0)}
    if not _finite(h):
        return None
    assert h is not None
    lo, hi = h
    if to in _CAST_RANGE and to != TensorProto.FLOAT16:
        tlo, thi = _CAST_RANGE[to]
        if lo < tlo or hi > thi:
            return {node.output[0]: (tlo, thi)}
        return {node.output[0]: (math.floor(lo), math.ceil(hi))}
    return {node.output[0]: (lo, hi)}


def _cut_gelu(
    node: onnx.NodeProto, c: "_CutCtx"
) -> Optional[Dict[str, Tuple[Any, Any]]]:
    h = c.known.hull(node.input[0])
    if not _finite(h):
        return None
    assert h is not None
    return {node.output[0]: (-0.1701, max(0.0, h[1]))}


def _cut_where(
    node: onnx.NodeProto, c: "_CutCtx"
) -> Optional[Dict[str, Tuple[Any, Any]]]:
    # whichever branch the condition picks, the result lies in the union of the two branches
    hx, hy = c.known.hull(node.input[1]), c.known.hull(node.input[2])
    if hx is None or hy is None:
        return None
    lo, hi = min(hx[0], hy[0]), max(hx[1], hy[1])
    if lo == -math.inf and hi == math.inf:
        return None
    return {node.output[0]: (lo, hi)}


def _cut_bool(
    node: onnx.NodeProto, c: "_CutCtx"
) -> Optional[Dict[str, Tuple[Any, Any]]]:
    return {node.output[0]: (0.0, 1.0)}


def _cut_constant_of_shape(
    node: onnx.NodeProto, c: "_CutCtx"
) -> Optional[Dict[str, Tuple[Any, Any]]]:
    v = 0.0
    for a in node.attribute:
        if a.name == "value":
            arr = numpy_helper.to_array(a.t)
            if arr.size:
                v = float(np.asarray(arr).reshape(-1)[0])
    return {node.output[0]: (v, v)}


def _cut_abs(
    node: onnx.NodeProto, c: "_CutCtx"
) -> Optional[Dict[str, Tuple[Any, Any]]]:
    h = c.known.hull(node.input[0])
    if h is None:
        return None
    lo = 0.0 if h[0] <= 0 <= h[1] else min(abs(h[0]), abs(h[1]))
    return {node.output[0]: (lo, max(abs(h[0]), abs(h[1])))}


def _cut_unit(
    node: onnx.NodeProto, c: "_CutCtx"
) -> Optional[Dict[str, Tuple[Any, Any]]]:
    return {node.output[0]: (-1.0, 1.0)}  # Sin, Cos, Sign


def _cut_reciprocal(
    node: onnx.NodeProto, c: "_CutCtx"
) -> Optional[Dict[str, Tuple[Any, Any]]]:
    h = c.known.hull(node.input[0])
    if h is None or not _finite(h) or h[0] <= 0 <= h[1]:
        return None
    a, b = 1.0 / h[0], 1.0 / h[1]
    return {node.output[0]: (min(a, b), max(a, b))}


_CUT_OPS: Dict[str, Callable[..., Optional[Dict[str, Tuple[Any, Any]]]]] = {
    "Gather": _cut_gather,
    "LayerNormalization": _cut_layernorm,
    "SimplifiedLayerNormalization": _cut_rmsnorm,
    "RMSNormalization": _cut_rmsnorm,
    "Cast": _cut_cast,
    "Gelu": _cut_gelu,
    "Where": _cut_where,
    "Equal": _cut_bool,
    "Greater": _cut_bool,
    "GreaterOrEqual": _cut_bool,
    "Less": _cut_bool,
    "LessOrEqual": _cut_bool,
    "And": _cut_bool,
    "Or": _cut_bool,
    "Xor": _cut_bool,
    "Not": _cut_bool,
    "IsNaN": _cut_bool,
    "IsInf": _cut_bool,
    "ConstantOfShape": _cut_constant_of_shape,
    "Abs": _cut_abs,
    "Sin": _cut_unit,
    "Cos": _cut_unit,
    "Sign": _cut_unit,
    "Reciprocal": _cut_reciprocal,
}
_NONNEG_PRODUCERS = {"Exp", "Sigmoid", "Softplus", "Relu", "Abs"}


@dataclasses.dataclass
class _CutCtx:
    known: _Known
    consts: Dict[str, np.ndarray]
    shapes: Dict[str, Optional[List[int]]]


def _softmax_div(node: onnx.NodeProto, producers: Dict[str, onnx.NodeProto]) -> bool:
    """``Div(e, ReduceSum(e))`` with ``e >= 0`` (a decomposed softmax): the result is in [0, 1]."""
    if node.op_type != "Div" or len(node.input) != 2:
        return False
    num, den = node.input
    pd, pn = producers.get(den), producers.get(num)
    return bool(
        pd is not None
        and pd.op_type == "ReduceSum"
        and pd.input
        and pd.input[0] == num
        and pn is not None
        and pn.op_type in _NONNEG_PRODUCERS
    )


def _analyse(
    model: onnx.ModelProto,
    ranges: Dict[str, Tuple[Any, Any]],
    input_shapes: Optional[Dict[str, Sequence[Any]]],
    shapes: Dict[str, Optional[List[int]]],
    types: Dict[str, int],
    consts: Dict[str, np.ndarray],
    max_iter: int = 64,
) -> Tuple[_Known, int, List[str]]:
    """Propagate intervals, *cutting* at ops whose output bound follows from their maths."""
    known = _Known()
    cur = model
    extra: Dict[str, Tuple[Any, Any]] = {}
    extra_shapes: Dict[str, Sequence[Any]] = dict(input_shapes or {})
    cuts = 0
    producers = {o: n for n in model.graph.node for o in n.output if o}
    ctx = _CutCtx(known, consts, shapes)
    for _ in range(max_iter):
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", RuntimeWarning
            )  # all-NaN slices in interval's Div hull
            res = _interval.propagate(cur, {**ranges, **extra}, extra_shapes or None)
        known.update(res)
        todo: List[Tuple[onnx.NodeProto, Dict[str, Tuple[Any, Any]]]] = []
        for node in cur.graph.node:
            outs = [o for o in node.output if o]
            if not outs:
                continue
            # interval.propagate fills an unmodelled op's outputs with (-inf, inf) when it knows
            # the shape, or leaves them out when it does not: both mean "no information"
            missing = any(
                (o not in res.intervals and o not in res.ranged)
                or _unknown(known.hull(o))
                for o in outs
            )
            is_softmax = _softmax_div(node, producers)
            if not (missing and node.op_type in _CUT_OPS) and not is_softmax:
                continue
            if node.input and any(
                i and known.hull(i) is None and i not in consts for i in node.input
            ):
                continue
            if is_softmax and not missing:
                cut: Optional[Dict[str, Tuple[Any, Any]]] = {node.output[0]: (0.0, 1.0)}
                # only worth cutting when the interval hull is much looser than [0, 1]
                h = known.hull(node.output[0])
                if h is not None and h[1] <= 1.0 + 1e-9 and h[0] >= -1e-9:
                    continue
            elif is_softmax:
                cut = {node.output[0]: (0.0, 1.0)}
            else:
                cut = _CUT_OPS[node.op_type](node, ctx)
            if cut is None:
                continue
            todo.append((node, cut))
        if not todo:
            break
        drop = {id(n) for n, _ in todo}
        new = onnx.ModelProto()
        new.CopyFrom(cur)
        keep = [n for n in cur.graph.node if id(n) not in drop]
        del new.graph.node[:]
        new.graph.node.extend(keep)
        applied = 0
        for node, cut in todo:
            for out, (lo, hi) in cut.items():
                shp = shapes.get(out)
                if out in {i.name for i in new.graph.input}:
                    continue
                etype = types.get(out, TensorProto.FLOAT)
                dims = [int(d) for d in shp] if shp is not None else None
                vi = onnx.helper.make_tensor_value_info(out, etype, dims)
                new.graph.input.append(vi)
                extra[out] = (lo, hi)
                if dims is None:
                    extra_shapes[out] = [None]
                applied += 1
                cuts += 1
        if not applied:
            break
        cur = new
    unanalysed = [
        o for n in model.graph.node for o in n.output if o and known.hull(o) is None
    ]
    return known, cuts, unanalysed


class _Linter:
    def __init__(
        self,
        model: onnx.ModelProto,
        known: _Known,
        dtype: str,
        assumptions: Dict[str, Tuple[float, float]],
        unannotated: List[str],
        include_unbounded: bool,
        types: Dict[str, int],
        shapes: Dict[str, Optional[List[int]]],
        consts: Dict[str, np.ndarray],
    ) -> None:
        self.model, self.known, self.dtype = model, known, dtype
        self.cfg = _dtype_cfg(dtype)
        self.assumptions, self.unannotated = assumptions, unannotated
        self.include_unbounded = include_unbounded
        self.types, self.shapes, self.consts = types, shapes, consts
        self.findings: List[Finding] = []
        self.suppressed = 0
        self.consequences = 0
        self.producers = {o: n for n in model.graph.node for o in n.output if o}
        self.flagged_nodes: Set[str] = set()
        self.unmodelled: Set[str] = set()
        # tensors that depend on an input with no declared range
        init = {t.name for t in model.graph.initializer}
        self.tainted: Set[str] = {
            i.name
            for i in model.graph.input
            if i.name not in init and i.name in unannotated
        }
        for n in model.graph.node:
            if any(i in self.tainted for i in n.input):
                self.tainted.update(o for o in n.output if o)

    # -- helpers
    def is_float(self, name: str) -> bool:
        return self.types.get(name) in _FLOAT_TYPES

    def hull(self, name: str) -> Optional[Tuple[float, float]]:
        return self.known.hull(name)

    def eff_hull(self, name: str) -> Optional[Tuple[float, float]]:
        """Hull of ``name``; ``x - ReduceMax(x)`` is tightened to ``[lo - hi, 0]`` (a dependency
        plain intervals cannot see)."""
        h = self.hull(name)
        p = self.producers.get(name)
        if p is not None and p.op_type == "Sub" and len(p.input) == 2:
            q = self.producers.get(p.input[1])
            xh = self.hull(p.input[0])
            if (
                q is not None
                and q.op_type == "ReduceMax"
                and q.input
                and q.input[0] == p.input[0]
            ):
                if _finite(xh):
                    assert xh is not None
                    return (xh[0] - xh[1], 0.0)
        return h

    def const(self, name: str) -> Optional[np.ndarray]:
        return self.consts.get(name)

    def emit(
        self,
        rule: str,
        node: onnx.NodeProto,
        nid: str,
        severity: str,
        tensor: str,
        message: str,
        hull: Tuple[float, float],
        fix: str,
        check: Optional[Check] = None,
        tensors: Sequence[str] = (),
        involved: Sequence[str] = (),
        impl: bool = False,
        what: str = "",
    ) -> Optional[Finding]:
        if _unknown(hull) and not any(
            t in self.tainted for t in list(involved) + [tensor]
        ):
            # "anything" because an op is not modelled, not because an input is unbounded: no claim
            self.unmodelled.add(tensor)
            return None
        unb = (not math.isfinite(hull[0]) or not math.isfinite(hull[1])) and any(
            t in self.tainted for t in list(involved) + [tensor]
        )
        if unb and not self.include_unbounded:
            self.suppressed += 1
            return None
        f = Finding(
            rule=rule,
            severity=severity,
            node=nid,
            op_type=node.op_type,
            tensor=tensor,
            message=message,
            interval=(float(hull[0]), float(hull[1])),
            assumption=dict(self.assumptions),
            fix=fix,
            unbounded=unb,
            implementation_dependent=impl,
            observed_what=what,
            _check=check,
            _tensors=tuple(dict.fromkeys(tensors)),
        )
        self.findings.append(f)
        self.flagged_nodes.add(nid)
        return f

    # -- rules ---------------------------------------------------------
    def run(self) -> None:
        for idx, node in enumerate(self.model.graph.node):
            nid = node.name or f"{node.op_type}_{idx}"
            rule = getattr(self, "_r_" + node.op_type, None)
            if rule is not None:
                rule(node, nid)
        for idx, node in enumerate(self.model.graph.node):
            nid = node.name or f"{node.op_type}_{idx}"
            self._generic(node, nid)
        self._constants()

    def _elementwise_zero(self, name: str) -> Tuple[int, int, Tuple[float, float]]:
        """(#elements whose interval contains 0, #elements, hull of those elements)."""
        if name in self.known.arr:
            lo, hi = self.known.arr[name]
            lo = np.asarray(lo, dtype=np.float64)
            hi = np.asarray(hi, dtype=np.float64)
            m = (lo <= 0) & (hi >= 0)
            if m.any():
                return (
                    int(m.sum()),
                    int(m.size),
                    (float(lo[m].min()), float(hi[m].max())),
                )
            return 0, int(m.size), (0.0, 0.0)
        h = self.hull(name)
        if h is not None and h[0] <= 0 <= h[1]:
            return 1, 1, h
        return 0, 1, (0.0, 0.0)

    def _r_Div(self, node: onnx.NodeProto, nid: str) -> None:
        den, out = node.input[1], node.output[0]
        k, n, h = self._elementwise_zero(den)
        if not k:
            return
        integer = self.types.get(out) in _INT_TYPES
        self.emit(
            "div-by-zero",
            node,
            nid,
            CAN_FAIL,
            den,
            f"denominator '{den}' can be 0 for some input ({k} of {n} elements have an interval "
            f"containing 0, hull [{h[0]:.4g}, {h[1]:.4g}]): "
            + (
                "integer division by zero is undefined"
                if integer
                else "result is inf/nan"
            ),
            h,
            "add a small epsilon to the denominator (x / (d + eps)), clamp it away from 0, or "
            "narrow the declared input range if the real data cannot reach 0",
            check=_chk_nonfinite(out, den, "minabs"),
            tensors=[den, out],
            involved=[den],
            what="min |denominator|",
        )

    def _r_Reciprocal(self, node: onnx.NodeProto, nid: str) -> None:
        x, out = node.input[0], node.output[0]
        k, n, h = self._elementwise_zero(x)
        if k:
            self.emit(
                "div-by-zero",
                node,
                nid,
                CAN_FAIL,
                x,
                f"Reciprocal input '{x}' can be 0 for some input ({k} of {n} elements, hull "
                f"[{h[0]:.4g}, {h[1]:.4g}]): result is inf",
                h,
                "add a small epsilon before the reciprocal or clamp the input away from 0",
                check=_chk_nonfinite(out, x, "minabs"),
                tensors=[x, out],
                involved=[x],
                what="min |input|",
            )

    def _r_Log(self, node: onnx.NodeProto, nid: str) -> None:
        x, out = node.input[0], node.output[0]
        h = self.hull(x)
        if h is not None and h[0] <= 0:
            kind = "-inf at 0" if h[0] == 0 else "nan for negative inputs"
            self.emit(
                "log-domain",
                node,
                nid,
                CAN_FAIL,
                x,
                f"Log input '{x}' can be <= 0 (hull [{h[0]:.4g}, {h[1]:.4g}]): {kind}",
                h,
                "clamp the input to a small positive value (Max(x, eps)) or add eps before the Log",
                check=_chk_nonfinite(out, x, "min"),
                tensors=[x, out],
                involved=[x],
                what="min input",
            )

    def _r_Sqrt(self, node: onnx.NodeProto, nid: str) -> None:
        x, out = node.input[0], node.output[0]
        h = self.hull(x)
        if h is not None and h[0] < 0:
            self.emit(
                "sqrt-domain",
                node,
                nid,
                CAN_FAIL,
                x,
                f"Sqrt input '{x}' can be negative (hull [{h[0]:.4g}, {h[1]:.4g}]): result is nan",
                h,
                "clamp the input at 0 (Relu / Max(x, 0)) or add a positive epsilon",
                check=_chk_nonfinite(out, x, "min"),
                tensors=[x, out],
                involved=[x],
                what="min input",
            )

    def _r_Pow(self, node: onnx.NodeProto, nid: str) -> None:
        base, expo, out = node.input[0], node.input[1], node.output[0]
        e = self.const(expo)
        h = self.hull(base)
        if e is None or h is None or e.size != 1:
            return
        ev = float(np.asarray(e).reshape(-1)[0])
        if h[0] < 0 and ev != math.floor(ev):
            self.emit(
                "pow-domain",
                node,
                nid,
                CAN_FAIL,
                base,
                f"Pow base '{base}' can be negative (hull [{h[0]:.4g}, {h[1]:.4g}]) with the "
                f"fractional exponent {ev:g}: result is nan",
                h,
                "take the absolute value / clamp the base at 0 first",
                check=_chk_nonfinite(out, base, "min"),
                tensors=[base, out],
                involved=[base],
                what="min base",
            )
        elif ev < 0 and h[0] <= 0 <= h[1]:
            self.emit(
                "pow-domain",
                node,
                nid,
                CAN_FAIL,
                base,
                f"Pow base '{base}' can be 0 (hull [{h[0]:.4g}, {h[1]:.4g}]) with the negative "
                f"exponent {ev:g}: result is inf",
                h,
                "add an epsilon to the base or clamp it away from 0",
                check=_chk_nonfinite(out, base, "minabs"),
                tensors=[base, out],
                involved=[base],
                what="min |base|",
            )

    def _softmax_context(self, node: onnx.NodeProto) -> bool:
        """Is this ``Exp`` the numerator of a decomposed softmax (``Div(e, ReduceSum(e))``)?"""
        e = node.output[0]
        for n in self.model.graph.node:
            if n.op_type == "ReduceSum" and n.input and n.input[0] == e:
                return True
        return False

    def _r_Exp(self, node: onnx.NodeProto, nid: str) -> None:
        x, out = node.input[0], node.output[0]
        h = self.eff_hull(x)
        thr = self.cfg["exp_over"]
        if h is not None and h[1] > thr:
            soft = self._softmax_context(node)
            fix = (
                "this Exp feeds a normalising sum (a decomposed softmax) without the max "
                "subtraction: use the Softmax op, or compute exp(x - ReduceMax(x))"
                if soft
                else "bound the input (Clip / Min) or compute in a wider dtype; for a softmax subtract "
                "the row max first"
            )
            self.emit(
                "exp-overflow",
                node,
                nid,
                CAN_FAIL,
                x,
                f"Exp input '{x}' can reach {h[1]:.4g} > ln({self.dtype} max) = {thr:.2f}: result "
                f"overflows to inf in {self.dtype}"
                + (" (softmax numerator without max subtraction)" if soft else ""),
                h,
                fix,
                check=_chk_exp(x, out, self.cfg["max"]),
                tensors=[x, out],
                involved=[x],
                what="max input",
            )

    def _saturating(self, node: onnx.NodeProto, nid: str, what: str) -> None:
        x = node.input[0]
        h = self.hull(x)
        thr = self.cfg["exp_over"]
        if h is not None and (h[1] > thr or h[0] < -thr):
            self.emit(
                "saturation",
                node,
                nid,
                CAN_LOSE_PRECISION,
                x,
                f"{what} input '{x}' reaches [{h[0]:.4g}, {h[1]:.4g}], beyond +-ln({self.dtype} "
                f"max) = +-{thr:.2f}: the op saturates, and an implementation built on a naive "
                f"exp() overflows there",
                h,
                "clip the input to +-ln(dtype max) before the op, or use a numerically stable kernel",
                impl=True,
                tensors=[x],
                involved=[x],
            )

    def _r_Sigmoid(self, node: onnx.NodeProto, nid: str) -> None:
        self._saturating(node, nid, "Sigmoid")

    def _r_Tanh(self, node: onnx.NodeProto, nid: str) -> None:
        self._saturating(node, nid, "Tanh")

    def _r_Softplus(self, node: onnx.NodeProto, nid: str) -> None:
        x = node.input[0]
        h = self.hull(x)
        thr = self.cfg["exp_over"]
        if h is not None and h[1] > thr:
            self.emit(
                "saturation",
                node,
                nid,
                CAN_LOSE_PRECISION,
                x,
                f"Softplus input '{x}' can reach {h[1]:.4g} > ln({self.dtype} max) = {thr:.2f}: "
                f"log(1 + exp(x)) overflows in a naive implementation",
                h,
                "use x + log1p(exp(-x)) for large x (a stable kernel) or clip the input",
                impl=True,
                tensors=[x],
                involved=[x],
            )

    def _r_Softmax(self, node: onnx.NodeProto, nid: str) -> None:
        x = node.input[0]
        h = self.hull(x)
        thr = self.cfg["exp_over"]
        if (
            h is not None
            and _finite(h)
            and (h[1] - h[0]) > thr
            and self.dtype == "fp16"
        ):
            self.emit(
                "saturation",
                node,
                nid,
                INFO,
                x,
                f"Softmax logits '{x}' can span {h[1] - h[0]:.4g} > ln(fp16 max) = {thr:.2f}: a "
                f"kernel without max subtraction overflows",
                h,
                "check that the target runtime's fp16 Softmax subtracts the row max",
                impl=True,
                tensors=[x],
                involved=[x],
            )

    def _norm(self, node: onnx.NodeProto, nid: str) -> None:
        x = node.input[0]
        shape = self.shapes.get(x)
        h = self.hull(x)
        if not shape or h is None or not _finite(h):
            return
        attrs = _attrs(node)
        axis = int(attrs.get("axis", -1)) % len(shape)
        n = _prod(shape[axis:])
        kind = "rms" if node.op_type != "LayerNormalization" else "ln"
        eps = float(attrs.get("epsilon", 1e-5))
        cfg = self.cfg
        if kind == "rms":
            var_ub = _maxabs(h) ** 2
            what = "mean of squares"
        else:
            var_ub = (h[1] - h[0]) ** 2 / 4.0
            what = "variance"
        if var_ub > cfg["max"]:
            self.emit(
                "norm-variance",
                node,
                nid,
                CAN_FAIL,
                x,
                f"{what} of '{x}' over {n} features can reach {var_ub:.4g} > {self.dtype} max "
                f"{cfg['max']:.6g} (input hull [{h[0]:.4g}, {h[1]:.4g}]): overflow to inf",
                (0.0, var_ub),
                "scale the input down before the norm, or run the norm statistics in fp32",
                check=_chk_norm_var(x, axis, kind, cfg["max"], "max"),
                tensors=[x],
                involved=[x],
                what=f"max row {what}",
            )
        elif n * var_ub > cfg["max"]:
            self.emit(
                "norm-variance",
                node,
                nid,
                CAN_FAIL,
                x,
                f"the sum of squares behind the {what} of '{x}' can reach {n * var_ub:.4g} "
                f"(N={n}) > {self.dtype} max {cfg['max']:.6g}: overflows if the kernel sums before "
                f"dividing by N",
                (0.0, n * var_ub),
                "accumulate the statistics in fp32 (check the target kernel) or scale the input down",
                check=_chk_norm_var(x, axis, kind, cfg["max"] / max(n, 1), "max"),
                tensors=[x],
                involved=[x],
                impl=True,
                what=f"max row {what}",
            )
        # can a row be constant (LayerNorm variance 0) / all-zero (RMS)? from the per-element boxes
        can_zero = True
        if x in self.known.arr:
            lo, hi = self.known.arr[x]
            lo = np.asarray(lo, dtype=np.float64).reshape(-1, n)
            hi = np.asarray(hi, dtype=np.float64).reshape(-1, n)
            if kind == "ln":
                can_zero = bool(np.any(lo.max(axis=1) <= hi.min(axis=1)))
            else:
                can_zero = bool(np.any((lo <= 0).all(axis=1) & (hi >= 0).all(axis=1)))
        eps_in = eps if eps >= cfg["min_sub"] / 2 else 0.0
        if can_zero and eps_in == 0.0:
            self.emit(
                "norm-variance",
                node,
                nid,
                CAN_FAIL,
                x,
                f"{what} of '{x}' can be 0 (a row can be constant) and epsilon={eps:g} "
                f"{'is 0' if eps == 0 else f'underflows to 0 in {self.dtype}'}: 1/sqrt(0) = inf, "
                f"0 * inf = nan",
                (0.0, 0.0),
                "use epsilon >= 1e-5 (fp16: >= 6e-5 to stay normal)",
                check=_chk_norm_var(x, axis, kind, 0.0, "min"),
                tensors=[x],
                involved=[x],
                what=f"min row {what}",
            )
        elif can_zero and eps_in < cfg["min_normal"] and self.dtype == "fp16":
            self.emit(
                "norm-variance",
                node,
                nid,
                INFO,
                x,
                f"epsilon={eps:g} is subnormal in fp16 (< {cfg['min_normal']:.3g}) and a row of "
                f"'{x}' can be constant: 1/sqrt(eps) loses precision (true of most fp16 LayerNorms)",
                (0.0, 0.0),
                "use epsilon >= 6.1e-5 in fp16",
                tensors=[x],
                involved=[x],
            )

    _r_LayerNormalization = _norm
    _r_RMSNormalization = _norm
    _r_SimplifiedLayerNormalization = _norm

    def _int_acc(self, node: onnx.NodeProto, nid: str) -> None:
        a, b = node.input[0], node.input[1]
        out = node.output[0]
        ah = self.hull(a)
        if ah is None:
            at = self.types.get(a)
            ah = (_CAST_RANGE[at][0], _CAST_RANGE[at][1]) if at in _CAST_RANGE else None
        if ah is None or not _finite(ah):
            return

        def zp(i: int) -> float:
            if len(node.input) > i and node.input[i]:
                z = self.const(node.input[i])
                if z is not None and z.size:
                    return float(np.max(np.asarray(z, dtype=np.float64)))
            return 0.0

        za, zb = zp(2), zp(3)
        amax = max(abs(ah[0] - za), abs(ah[1] - za))
        w = self.const(b)
        if w is not None:
            wd = np.asarray(w, dtype=np.float64) - zb
            if node.op_type == "ConvInteger":
                per = np.abs(wd).reshape(wd.shape[0], -1).sum(axis=1)
            elif wd.ndim >= 2:
                per = np.abs(wd).reshape(-1, wd.shape[-1]).sum(axis=0)
            else:
                per = np.array([np.abs(wd).sum()])
            bound = float(per.max()) * amax
            detail = "from the constant weights"
        else:
            bt = self.types.get(b)
            if bt not in _CAST_RANGE:
                return
            bmax = max(abs(_CAST_RANGE[bt][0] - zb), abs(_CAST_RANGE[bt][1] - zb))
            ws = self.shapes.get(b)
            if not ws:
                return
            kdepth = (
                _prod(ws[:-1]) if node.op_type == "MatMulInteger" else _prod(ws[1:])
            )
            bound = kdepth * bmax * amax
            detail = f"from the dtype range of the weights (K={kdepth})"
        if bound > _INT32_MAX:
            chk: Optional[Check] = None
            if node.op_type == "MatMulInteger" and w is not None:
                chk = _chk_int_acc(a, np.asarray(w), za, zb)
            self.emit(
                "int32-wrap",
                node,
                nid,
                CAN_FAIL,
                out,
                f"int32 accumulator of {node.op_type} can reach +-{bound:.6g} > 2^31-1 "
                f"({detail}; input hull [{ah[0]:.4g}, {ah[1]:.4g}], zero point {za:g}): wraps around",
                (-bound, bound),
                "reduce the reduction depth (split the MatMul), narrow the activation range, or "
                "accumulate in a wider type",
                check=chk,
                tensors=[a, out],
                involved=[a],
                what="max |accumulator|",
            )

    _r_MatMulInteger = _int_acc
    _r_ConvInteger = _int_acc

    def _r_Cast(self, node: onnx.NodeProto, nid: str) -> None:
        to = int(_attrs(node).get("to", 0))
        x = node.input[0]
        h = self.hull(x)
        if to not in _CAST_RANGE or h is None:
            return
        src = self.types.get(x)
        if to == TensorProto.FLOAT16 and src not in (
            TensorProto.FLOAT,
            TensorProto.DOUBLE,
        ):
            return
        tlo, thi = _CAST_RANGE[to]
        if h[0] < tlo or h[1] > thi:
            is_f16 = to == TensorProto.FLOAT16
            self.emit(
                "cast-range",
                node,
                nid,
                CAN_FAIL,
                x,
                f"Cast of '{x}' to {'fp16' if is_f16 else 'a narrower integer type'} whose range is "
                f"[{tlo:g}, {thi:g}]: the input can span [{h[0]:.4g}, {h[1]:.4g}] -> "
                + (
                    "overflows to inf"
                    if is_f16
                    else "saturates or is undefined (implementation defined)"
                ),
                h,
                "clip the input to the target range first, or keep the wider type",
                check=_chk_outside(x, tlo, thi),
                tensors=[x],
                involved=[x],
                what="max |input|",
            )

    # -- info-level rules
    def _r_Clip(self, node: onnx.NodeProto, nid: str) -> None:
        x = node.input[0]
        h = self.hull(x)
        if h is None or not _finite(h):
            return
        lo, hi = -math.inf, math.inf
        attrs = _attrs(node)
        for pos, key, which in ((1, "min", "lo"), (2, "max", "hi")):
            v: Optional[float] = None
            if len(node.input) > pos and node.input[pos]:
                c = self.const(node.input[pos])
                if c is not None and c.size:
                    v = float(np.asarray(c).reshape(-1)[0])
            elif key in attrs:
                v = float(attrs[key])
            if v is not None:
                if which == "lo":
                    lo = v
                else:
                    hi = v
        if lo == -math.inf and hi == math.inf:
            return
        if h[0] >= lo and h[1] <= hi:
            self.emit(
                "dead-op",
                node,
                nid,
                INFO,
                x,
                f"Clip never changes its input: '{x}' stays in [{h[0]:.4g}, {h[1]:.4g}] inside "
                f"[{lo:g}, {hi:g}] (under the declared ranges)",
                h,
                "the Clip can be removed if the declared range is a real contract",
                tensors=[x],
            )
        elif h[1] <= lo or h[0] >= hi:
            self.emit(
                "dead-op",
                node,
                nid,
                INFO,
                x,
                f"Clip always saturates: '{x}' in [{h[0]:.4g}, {h[1]:.4g}] is entirely "
                f"{'below' if h[1] <= lo else 'above'} [{lo:g}, {hi:g}]",
                h,
                "the output is constant under the declared ranges; check the range assumption",
                tensors=[x],
            )

    def _r_Relu(self, node: onnx.NodeProto, nid: str) -> None:
        x = node.input[0]
        h = self.hull(x)
        if h is None or not _finite(h):
            return
        if h[0] >= 0:
            self.emit(
                "dead-op",
                node,
                nid,
                INFO,
                x,
                f"Relu is the identity: '{x}' is >= {h[0]:.4g} under the declared ranges",
                h,
                "the Relu can be removed if the declared range is a real contract",
                tensors=[x],
            )
        elif h[1] <= 0:
            self.emit(
                "dead-op",
                node,
                nid,
                INFO,
                x,
                f"Relu always outputs 0: '{x}' is <= {h[1]:.4g} under the declared ranges",
                h,
                "the whole branch is dead under the declared ranges",
                tensors=[x],
            )

    def _decided(self, cond: str) -> Optional[bool]:
        h = self.hull(cond)
        if h is not None and h[0] == h[1] and h[0] in (0.0, 1.0):
            return bool(h[0])
        p = self.producers.get(cond)
        if p is None or len(p.input) != 2:
            return None
        a, b = self.hull(p.input[0]), self.hull(p.input[1])
        if a is None or b is None:
            return None
        t = p.op_type
        if t == "Greater":
            return True if a[0] > b[1] else False if a[1] <= b[0] else None
        if t == "GreaterOrEqual":
            return True if a[0] >= b[1] else False if a[1] < b[0] else None
        if t == "Less":
            return True if a[1] < b[0] else False if a[0] >= b[1] else None
        if t == "LessOrEqual":
            return True if a[1] <= b[0] else False if a[0] > b[1] else None
        if t == "Equal":
            if a[1] < b[0] or b[1] < a[0]:
                return False
            if a[0] == a[1] == b[0] == b[1]:
                return True
        return None

    def _r_Where(self, node: onnx.NodeProto, nid: str) -> None:
        d = self._decided(node.input[0])
        if d is not None:
            self.emit(
                "dead-op",
                node,
                nid,
                INFO,
                node.input[0],
                f"Where condition '{node.input[0]}' is always {str(d).lower()} under the declared "
                f"ranges: only the '{node.input[1] if d else node.input[2]}' branch is ever taken",
                (0.0, 1.0),
                "the Where can be replaced by the taken branch if the declared range is a contract",
                tensors=[node.input[0]],
            )

    # -- generic rules over every tensor
    def _generic(self, node: onnx.NodeProto, nid: str) -> None:
        cfg = self.cfg
        for out in node.output:
            # constants are the const-overflow rule's business, not this one's
            if not out or not self.is_float(out) or out in self.consts:
                continue
            h = self.hull(out)
            if h is None:
                continue
            ins = [i for i in node.input if i and i not in self.consts]
            if _unknown(h) or any(_unknown(self.hull(i)) for i in ins):
                if (
                    _unknown(h)
                    and nid not in self.flagged_nodes
                    and not any(i in self.tainted for i in ins)
                ):
                    self.unmodelled.add(out)
                continue
            # an input that already overflows (a constant mask included) makes this a consequence
            ins_over = any(
                ih is not None and _maxabs(ih) > cfg["max"]
                for ih in (self.hull(i) for i in node.input if i)
            )
            if (_maxabs(h) > cfg["max"]) and node.op_type != "Exp":
                if ins_over:
                    self.consequences += 1
                    continue
                if nid in self.flagged_nodes:
                    continue
                finite = _finite(h)
                self.emit(
                    "range-overflow",
                    node,
                    nid,
                    CAN_FAIL,
                    out,
                    f"'{out}' can reach {'+-inf' if not finite else f'{_maxabs(h):.4g}'} > "
                    f"{self.dtype} max {cfg['max']:.6g} (interval [{h[0]:.4g}, {h[1]:.4g}]) while its "
                    f"inputs are still in range: overflow to inf",
                    h,
                    "rescale the computation (e.g. divide an input first), accumulate in fp32, or "
                    "narrow the declared range if the real activations cannot get this large",
                    check=_chk_abs_gt(out, cfg["max"]),
                    tensors=[out],
                    involved=list(ins),
                    what="max |value|",
                )
            elif (
                self.dtype == "fp16"
                and _finite(h)
                and 0 < _maxabs(h) < cfg["min_normal"]
                and out not in self.consts
            ):
                self.emit(
                    "range-underflow",
                    node,
                    nid,
                    CAN_LOSE_PRECISION,
                    out,
                    f"'{out}' stays below the smallest normal fp16 number "
                    f"({cfg['min_normal']:.3g}): max |value| <= {_maxabs(h):.4g} loses precision "
                    f"or flushes to 0",
                    h,
                    "scale the tensor up (and the consumer down), or keep this part in fp32",
                    check=_chk_small(out, cfg["min_normal"]),
                    tensors=[out],
                    involved=list(ins),
                    what="min nonzero |value|",
                )

    def _constants(self) -> None:
        cfg = self.cfg
        if self.dtype == "fp32":
            return
        init_names = {t.name for t in self.model.graph.initializer}
        seen: Set[str] = set()
        owners: Dict[str, onnx.NodeProto] = {}
        for n in self.model.graph.node:
            for o in n.output:
                owners[o] = n
        for name, arr in self.consts.items():
            if name in seen or arr.dtype.kind != "f" or not arr.size:
                continue
            seen.add(name)
            a = np.abs(arr.astype(np.float64))
            node = owners.get(name)
            if node is None:
                node = onnx.NodeProto()
                node.op_type = "Initializer"
            nid = (node.name or name) if name not in init_names else name
            fin = a[np.isfinite(a)]
            big = int((fin > cfg["max"]).sum())
            if big:
                m = float(fin.max())
                mask_like = float(arr.min()) < -cfg["max"]
                f = self.emit(
                    "const-overflow",
                    node,
                    nid,
                    CAN_FAIL,
                    name,
                    f"constant '{name}' has {big} value(s) beyond {self.dtype} max "
                    f"{cfg['max']:.6g} (largest {m:.4g}): becomes +-inf in {self.dtype}"
                    + (
                        " (a very negative value is typically an attention mask: a fully masked "
                        "softmax row then becomes nan)"
                        if mask_like
                        else ""
                    ),
                    (float(arr.min()), float(arr.max())),
                    "use a finite mask value that fits the dtype (e.g. -1e4 / -65504 in fp16)"
                    if mask_like
                    else "rescale the constant or keep the op that uses it in a wider dtype",
                    check=None,
                    tensors=[name],
                )
                if f is not None:
                    # no input is involved: the value is read straight from the model
                    f.certainty = CONFIRMED
            if self.dtype == "fp16":
                nz = fin[fin > 0]
                flush = int((nz < cfg["min_sub"]).sum())
                sub = int(((nz >= cfg["min_sub"]) & (nz < cfg["min_normal"])).sum())
                # a handful of tiny weights among millions is noise; flag a real fraction
                if flush:
                    frac = flush / max(nz.size, 1)
                    self.emit(
                        "range-underflow",
                        node,
                        nid,
                        CAN_LOSE_PRECISION if frac >= _FLUSH_FRACTION else INFO,
                        name,
                        f"constant '{name}': {flush} of {arr.size} values ({frac:.2%} of the nonzero "
                        f"ones) are below fp16's smallest subnormal ({cfg['min_sub']:.3g}) and flush "
                        f"to 0 ({sub} more are subnormal)",
                        (float(arr.min()), float(arr.max())),
                        "rescale the tensor (fold a scale into the neighbouring op) or keep it in fp32",
                        tensors=[name],
                    )
                elif sub:
                    self.emit(
                        "range-underflow",
                        node,
                        nid,
                        INFO,
                        name,
                        f"constant '{name}': {sub} of {arr.size} values are subnormal in fp16 "
                        f"(reduced precision)",
                        (float(arr.min()), float(arr.max())),
                        "fine unless these values matter; otherwise rescale",
                        tensors=[name],
                    )


# --------------------------------------------------------------------------
# Witness search
# --------------------------------------------------------------------------


class _Exposed:
    """One onnxruntime session that also returns the intermediate tensors a finding needs."""

    def __init__(self, model: onnx.ModelProto, tensors: Sequence[str]) -> None:
        import onnxruntime as ort

        init = {t.name for t in model.graph.initializer}
        self.consts = _consts(model)
        produced = {o for n in model.graph.node for o in n.output if o}
        self.tensors = list(dict.fromkeys(tensors))
        want = [t for t in self.tensors if t in produced and t not in init]
        m = onnx.ModelProto()
        m.CopyFrom(model)
        have = {o.name for o in m.graph.output}
        for t in want:
            if t not in have:
                m.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
        self.names = [o.name for o in m.graph.output]
        self.sess = ort.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]
        )

    def run(self, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = dict(zip(self.names, self.sess.run(None, feeds)))
        out.update({k: np.asarray(v) for k, v in feeds.items()})
        for t in self.tensors:
            if t not in out and t in self.consts:
                out[t] = self.consts[t]
        return out


def _run_exposed(
    model: onnx.ModelProto, feeds: Dict[str, np.ndarray], tensors: List[str]
) -> Dict[str, np.ndarray]:
    return _Exposed(model, tensors).run(feeds)


class _Sampler:
    def __init__(
        self,
        model: onnx.ModelProto,
        ranges: Dict[str, Tuple[Any, Any]],
        input_shapes: Optional[Dict[str, Sequence[Any]]],
        seed: int,
    ) -> None:
        init = {t.name for t in model.graph.initializer}
        self.rng = np.random.default_rng(seed)
        self.specs: List[
            Tuple[str, Tuple[int, ...], Any, np.ndarray, np.ndarray, bool]
        ] = []
        for vi in model.graph.input:
            if vi.name in init:
                continue
            tt = vi.type.tensor_type
            override = (input_shapes or {}).get(vi.name)
            dims: List[int] = []
            for k, d in enumerate(tt.shape.dim):
                if (
                    override is not None
                    and k < len(override)
                    and isinstance(override[k], int)
                ):
                    dims.append(int(override[k]))
                elif d.HasField("dim_value") and d.dim_value > 0:
                    dims.append(int(d.dim_value))
                else:
                    dims.append(1)
            npt = _NP_FOR_TYPE.get(tt.elem_type, np.float32)
            is_int = np.issubdtype(npt, np.integer) or npt is np.bool_
            if vi.name in ranges:
                lo, hi = ranges[vi.name]
            else:
                lo, hi = (0.0, 10.0) if is_int else (-1.0, 1.0)
            lo_a = np.broadcast_to(np.asarray(lo, dtype=np.float64), dims).copy()
            hi_a = np.broadcast_to(np.asarray(hi, dtype=np.float64), dims).copy()
            self.specs.append((vi.name, tuple(dims), npt, lo_a, hi_a, bool(is_int)))

    def _make(
        self, fn: Callable[[np.ndarray, np.ndarray, bool], np.ndarray]
    ) -> Dict[str, np.ndarray]:
        out = {}
        for name, _dims, npt, lo, hi, is_int in self.specs:
            v = fn(lo, hi, is_int)
            out[name] = np.clip(np.rint(v) if is_int else v, lo, hi).astype(npt)
        return out

    def lo(self) -> Dict[str, np.ndarray]:
        return self._make(lambda lo, hi, i: lo)

    def hi(self) -> Dict[str, np.ndarray]:
        return self._make(lambda lo, hi, i: hi)

    def mid(self) -> Dict[str, np.ndarray]:
        return self._make(lambda lo, hi, i: (lo + hi) / 2)

    def uniform(self) -> Dict[str, np.ndarray]:
        return self._make(lambda lo, hi, i: lo + (hi - lo) * self.rng.random(lo.shape))

    def corner(self) -> Dict[str, np.ndarray]:
        return self._make(
            lambda lo, hi, i: np.where(self.rng.random(lo.shape) < 0.5, lo, hi)
        )

    def perturb(self, x: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        frac = float(self.rng.choice([0.01, 0.1, 0.5, 1.0]))
        mode = int(self.rng.integers(3))
        out = {}
        for name, _dims, npt, lo, hi, is_int in self.specs:
            cur = np.asarray(x[name], dtype=np.float64)
            sel = self.rng.random(lo.shape) < frac
            if mode == 0:  # push the chosen coordinates to a bound
                new = np.where(self.rng.random(lo.shape) < 0.5, lo, hi)
            elif mode == 1:  # resample
                new = lo + (hi - lo) * self.rng.random(lo.shape)
            else:  # small step
                new = cur + 0.1 * (hi - lo) * self.rng.standard_normal(lo.shape)
            v = np.where(sel, new, cur)
            out[name] = np.clip(np.rint(v) if is_int else v, lo, hi).astype(npt)
        return out


def _witness_search(
    model: onnx.ModelProto,
    findings: List[Finding],
    ranges: Dict[str, Tuple[Any, Any]],
    input_shapes: Optional[Dict[str, Sequence[Any]]],
    budget: int,
    seed: int,
    data: Sequence[Dict[str, np.ndarray]] = (),
) -> Tuple[int, str, Dict[int, Dict[str, np.ndarray]]]:
    cand = [i for i, f in enumerate(findings) if f._check is not None]
    if not cand:
        return 0, "no finding has a witness check", {}
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return 0, "witness search needs onnxruntime", {}
    need: List[str] = []
    for i in cand:
        need.extend(findings[i]._tensors)
    sampler = _Sampler(model, ranges, input_shapes, seed)
    try:
        exposed = _Exposed(model, need)
    except Exception as e:  # unsupported op / bad model for onnxruntime
        return (
            0,
            f"onnxruntime could not load the model ({type(e).__name__}); witnesses skipped",
            {},
        )
    best_prog = {i: -math.inf for i in cand}
    best_in: Dict[int, Dict[str, np.ndarray]] = {}
    best_obs: Dict[int, float] = {}
    confirmed: Dict[int, Dict[str, np.ndarray]] = {}
    runs = 0

    def evaluate(feeds: Dict[str, np.ndarray]) -> bool:
        nonlocal runs
        try:
            vals = exposed.run(feeds)
        except Exception:
            return False
        runs += 1
        for i in cand:
            f = findings[i]
            assert f._check is not None
            try:
                conf, prog, obs = f._check(vals)
            except KeyError:
                continue
            improved = prog > best_prog[i]
            if improved:
                best_prog[i] = prog
                best_in[i] = {k: v.copy() for k, v in feeds.items()}
                best_obs[i] = obs
            if conf and (i not in confirmed or improved):
                confirmed[i] = {k: v.copy() for k, v in feeds.items()}
                best_obs[i] = obs
        return True

    for feeds in data:  # the caller's real samples are the best witnesses there are
        if runs >= budget:
            break
        if not evaluate({k: np.asarray(v) for k, v in feeds.items()}):
            return (
                runs,
                "onnxruntime could not run a witness_data sample; witnesses skipped",
                {},
            )
    seeds = [sampler.mid, sampler.lo, sampler.hi]
    seeds += [sampler.uniform] * 4 + [sampler.corner] * 4
    for maker in seeds:
        if runs >= budget:
            break
        if not evaluate(maker()):
            return runs, "onnxruntime could not run the model; witnesses skipped", {}
    it = 0
    while runs < budget:
        open_ = [i for i in cand if i not in confirmed and i in best_in]
        if not open_:
            break
        i = open_[it % len(open_)]
        it += 1
        evaluate(sampler.perturb(best_in[i]))
    for i in cand:
        f = findings[i]
        f.witness_runs = runs
        if i in best_obs:
            f.observed = best_obs[i]
        if i in confirmed:
            f.certainty = CONFIRMED
    return runs, "", confirmed


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def lint(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None,
    dtype: str = "fp32",
    witness: int = 0,
    witness_seed: int = 0,
    input_shapes: Optional[Dict[str, Sequence[Any]]] = None,
    include_unbounded: bool = False,
    refine: bool = False,
    witness_data: Optional[Sequence[Dict[str, np.ndarray]]] = None,
) -> LintReport:
    """Lint ``model`` for numerical hazards over the input box; see the module docstring.

    :param witness_data: real input samples (``{input_name: array}``) tried first by the witness
        search, ahead of the random / corner candidates; they count against the ``witness``
        run budget.

    :param input_ranges: ``{input: (lo, hi)}`` (scalars or arrays); merged over the model's own
        ``onnxsim.range.*`` annotations. An input with neither is unbounded.
    :param dtype: the precision the model will be *run* in: ``"fp32"``, ``"fp16"`` or ``"bf16"``.
    :param witness: at most this many onnxruntime runs searching for an input that confirms a
        finding (0 = off). A finding is ``confirmed`` only if a replayable input really breaks it.
    :param input_shapes: ``{input: [dim, ...]}`` for dynamic inputs (also used for witnesses).
    :param include_unbounded: also report findings that exist only because an input has no range.
    :param refine: re-check findings with CROWN bounds (:mod:`onnxsim.crown`) and drop the ones it
        proves safe. Dense, so only for small models; silently skipped if CROWN fails.
    """
    t0 = time.perf_counter()
    _dtype_cfg(dtype)
    merged: Dict[str, Tuple[Any, Any]] = dict(_ranges.get_ranges(model))
    for k, (lo, hi) in (input_ranges or {}).items():
        merged[k] = (np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
    init = {t.name for t in model.graph.initializer}
    inputs = [i.name for i in model.graph.input if i.name not in init]
    unannotated = [i for i in inputs if i not in merged]
    assumptions = {
        k: (float(np.min(lo)), float(np.max(hi)))
        for k, (lo, hi) in merged.items()
        if k in inputs
    }
    types, shapes = _tensor_types(model)
    consts = _consts(model)
    known, cuts, unanalysed = _analyse(
        model, merged, input_shapes, shapes, types, consts
    )
    lin = _Linter(
        model,
        known,
        dtype,
        assumptions,
        unannotated,
        include_unbounded,
        types,
        shapes,
        consts,
    )
    lin.run()
    findings = lin.findings
    unanalysed = sorted(set(unanalysed) | lin.unmodelled)
    refined_away = 0
    if refine and any(f.severity != INFO for f in findings):
        findings, refined_away = _refine(
            model,
            merged,
            findings,
            lin,
            known,
            dtype,
            assumptions,
            unannotated,
            include_unbounded,
            types,
            shapes,
            consts,
        )
    report = LintReport(
        findings=findings,
        dtype=dtype,
        assumptions=assumptions,
        unannotated_inputs=unannotated,
        suppressed_unbounded=lin.suppressed,
        consequences=lin.consequences,
        unanalysed=unanalysed,
        cuts=cuts,
        refined_away=refined_away,
        _model=model,
        _known=known,
    )
    if witness > 0:
        runs, note, wit = _witness_search(
            model,
            findings,
            merged,
            input_shapes,
            witness,
            witness_seed,
            witness_data or (),
        )
        report.witness_runs, report.witness_note, report.witnesses = runs, note, wit
    report.seconds = time.perf_counter() - t0
    return report


def _refine(
    model: onnx.ModelProto,
    ranges: Dict[str, Tuple[Any, Any]],
    findings: List[Finding],
    lin: "_Linter",
    known: _Known,
    dtype: str,
    assumptions: Dict[str, Tuple[float, float]],
    unannotated: List[str],
    include_unbounded: bool,
    types: Dict[str, int],
    shapes: Dict[str, Optional[List[int]]],
    consts: Dict[str, np.ndarray],
) -> Tuple[List[Finding], int]:
    try:
        from . import crown as _crown

        names = sorted({t for f in findings if f.severity != INFO for t in f._tensors})
        names = [n for n in names if n not in consts]
        if not names:
            return findings, 0
        tb = _crown.bounds(model, ranges, output=names, method="crown")
        for name, b in tb.items():
            lo, hi = (
                np.asarray(b.lo, dtype=np.float64),
                np.asarray(b.hi, dtype=np.float64),
            )
            old = known.arr.get(name)
            if old is not None and old[0].shape == lo.shape:
                lo = np.maximum(lo, np.asarray(old[0], dtype=np.float64))
                hi = np.minimum(hi, np.asarray(old[1], dtype=np.float64))
            known.arr[name] = (lo, hi)
            known._cache.pop(name, None)
    except Exception:
        return findings, 0
    again = _Linter(
        model,
        known,
        dtype,
        assumptions,
        unannotated,
        include_unbounded,
        types,
        shapes,
        consts,
    )
    again.run()
    for f in again.findings:
        f.refined = True
    before = sum(1 for f in findings if f.severity != INFO)
    after = sum(1 for f in again.findings if f.severity != INFO)
    lin.suppressed = again.suppressed
    lin.consequences = again.consequences
    return again.findings, max(0, before - after)


def _parse_range(spec: str) -> Tuple[str, float, float]:
    name, _, rest = spec.rpartition("=")
    if not name or "," not in rest:
        raise argparse.ArgumentTypeError(f"expected NAME=LO,HI, got {spec!r}")
    lo, hi = rest.split(",", 1)
    return name, float(lo), float(hi)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m onnxsim.numeric_lint",
        description="Flag operations that can produce inf/nan/overflow for some input in a declared box.",
    )
    ap.add_argument("model")
    ap.add_argument("--dtype", default="fp32", choices=sorted(_PRECISION))
    ap.add_argument(
        "--range",
        action="append",
        default=[],
        type=_parse_range,
        metavar="NAME=LO,HI",
        help="input range (repeatable); merged over the model's onnxsim.range.* annotations",
    )
    ap.add_argument(
        "--witness",
        type=int,
        default=0,
        metavar="N",
        help="search up to N onnxruntime runs for inputs that confirm findings",
    )
    ap.add_argument("--include-unbounded", action="store_true")
    ap.add_argument(
        "--refine",
        action="store_true",
        help="re-check with CROWN bounds (small models only)",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--info", action="store_true", help="also print info-level findings"
    )
    args = ap.parse_args(argv)
    m = onnx.load(args.model)
    rep = lint(
        m,
        {n: (lo, hi) for n, lo, hi in args.range} or None,
        dtype=args.dtype,
        witness=args.witness,
        include_unbounded=args.include_unbounded,
        refine=args.refine,
    )
    if args.json:
        print(rep.to_json(indent=2))
    else:
        print(rep)
        if args.info:
            for f in rep.by(INFO):
                print(f)
    return 1 if rep.can_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
