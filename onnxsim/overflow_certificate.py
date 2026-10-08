"""Overflow and wraparound certificates for integer and fixed-point graphs.

A zero-knowledge circuit, or an integer-only accelerator, computes in a fixed ring: a signed
``b``-bit word, or a prime field ``F_p`` whose elements are read back as integers in
``[-(p-1)/2, (p-1)/2]``. A value outside that window wraps around, and the circuit then computes
a different function from the model. :func:`certify_no_overflow` uses the interval analysis of
:mod:`onnxsim.interval` to decide whether any value the graph can reach, for inputs in the declared
box, stays inside the window.

Per node it checks:

* every tensor of integer type (every tensor, for a field), against the window. An integer tensor
  whose dtype range already lies inside the window is certified from its dtype alone;
* the accumulator of each MatMul/Gemm/Conv and of ``MatMulInteger``, ``ConvInteger``,
  ``QLinearMatMul`` and ``QLinearConv``. Its bound is the worst partial sum,
  ``sum_k |w_k| * max|x - zx|`` over the reduction plus the bias, so intermediate overflow is
  covered and not only the final value. A non-constant weight uses ``K * max|x| * max|w|``;
* the input of every ``Cast`` to an integer type, against that type's full range.

Soundness: a tensor or accumulator whose bound cannot be computed (unbounded interval, dynamic
reduction depth, non-constant or per-channel zero point) is listed as ``unanalysed``, and a graph
with any such entry never ``fits``. A graph input of integer type defaults to its dtype range;
other inputs need a range from ``input_ranges`` or an ``onnxsim.range.*`` annotation.

Bounds are computed in float64, so they are exact for integer-valued operands below 2**53. For
non-integer operands they carry float64 rounding error.
"""

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from onnxsim import interval as _interval
from onnxsim import ranges as _ranges

_FLOAT_TYPES = {
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.FLOAT16,
    onnx.TensorProto.DOUBLE,
    onnx.TensorProto.BFLOAT16,
}
_NATIVE = {
    **_interval._INT_RANGE,
    onnx.TensorProto.INT64: (-(2**63), 2**63 - 1),
    onnx.TensorProto.UINT64: (0, 2**64 - 1),
}
_DEFAULT_INPUT = {**_NATIVE, onnx.TensorProto.BOOL: (0, 1)}
_INTEGER_OPS = {"MatMulInteger", "ConvInteger", "QLinearMatMul", "QLinearConv"}
_ACCUMULATORS = {"MatMul", "Gemm", "Conv"} | _INTEGER_OPS
# (activation, activation zero point, weight, weight zero point, bias) input indices
_OPERANDS = {
    "MatMul": (0, None, 1, None, None),
    "Gemm": (0, None, 1, None, 2),
    "Conv": (0, None, 1, None, 2),
    "MatMulInteger": (0, 2, 1, 3, None),
    "ConvInteger": (0, 2, 1, 3, None),
    "QLinearMatMul": (0, 2, 3, 5, None),
    "QLinearConv": (0, 2, 3, 5, 8),
}


@dataclasses.dataclass(frozen=True)
class Window:
    name: str
    lo: int
    hi: int


@dataclasses.dataclass(frozen=True)
class OverflowFinding:
    node: str
    tensor: str
    kind: str  # "tensor", "accumulator" or "cast"
    lo: int  # integer lower bound on the checked values
    hi: int  # integer upper bound on the checked values
    required_bits: int  # smallest signed width holding [lo, hi]
    fits: bool


@dataclasses.dataclass
class OverflowReport:
    window: str
    findings: List[OverflowFinding]
    unanalysed: List[str]  # "<node>: <reason>"

    @property
    def violations(self) -> List[OverflowFinding]:
        return [f for f in self.findings if not f.fits]

    @property
    def fits(self) -> bool:
        return not self.violations and not self.unanalysed


class _Skip(Exception):
    pass


def _window(bits: Optional[int], field_modulus: Optional[int]) -> Window:
    if (bits is None) == (field_modulus is None):
        raise ValueError("pass exactly one of bits or field_modulus")
    if bits is not None:
        if not 2 <= bits <= 64:
            raise ValueError(f"bits must be in [2, 64], got {bits}")
        return Window(f"int{bits}", -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    p = int(field_modulus or 0)
    if p < 3 or p % 2 == 0:
        raise ValueError(f"field modulus must be odd and at least 3, got {p}")
    half = (p - 1) // 2
    return Window(f"F_{p}", -half, half)


def _required_bits(lo: int, hi: int) -> int:
    def need(v: int) -> int:
        return v.bit_length() + 1 if v >= 0 else (-v - 1).bit_length() + 1

    return max(need(lo), need(hi))


def _feeds(model: onnx.ModelProto, input_ranges: Optional[Dict]) -> Dict:
    feeds = dict(input_ranges or {})
    annotated = _ranges.get_ranges(model)
    consts = {t.name for t in model.graph.initializer}
    elem = _interval._elem_types(model)
    for vi in model.graph.input:
        n = vi.name
        if n in feeds or n in annotated or n in consts:
            continue
        if elem.get(n) in _DEFAULT_INPUT:
            feeds[n] = _DEFAULT_INPUT[elem[n]]
    return feeds


def _hull(result: _interval.IntervalResult, name: str) -> Tuple[float, float]:
    if name in result.ranged:
        return result.ranged[name].hull
    if name not in result.intervals:
        raise _Skip(f"no interval for {name}")
    lo, hi = result.intervals[name]
    if np.size(lo) == 0:
        return 0.0, 0.0
    return float(np.min(lo)), float(np.max(hi))


def _bounded(result: _interval.IntervalResult, name: str) -> Tuple[float, float]:
    lo, hi = _hull(result, name)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise _Skip(f"unbounded {name}")
    return lo, hi


def _shape(result: _interval.IntervalResult, name: str) -> Tuple[int, ...]:
    if name in result.ranged or name not in result.intervals:
        raise _Skip(f"dynamic or unknown shape of {name}")
    return result.intervals[name][0].shape


def _zero_point(node: onnx.NodeProto, idx: Optional[int], consts: Dict) -> float:
    if idx is None or idx >= len(node.input) or not node.input[idx]:
        return 0.0
    name = node.input[idx]
    if name not in consts:
        raise _Skip(f"zero point {name} is not constant")
    z = consts[name].astype(np.float64)
    if not np.any(z):
        return 0.0
    if z.size != 1:
        raise _Skip(f"per-channel zero point {name}")
    return float(z.reshape(-1)[0])


def _centered_mag(lo: float, hi: float, zp: float) -> float:
    return max(abs(lo - zp), abs(hi - zp))


def _accumulator_bound(
    node: onnx.NodeProto, result: _interval.IntervalResult, consts: Dict
) -> float:
    x_i, zx_i, w_i, zw_i, b_i = _OPERANDS[node.op_type]
    attrs = _interval._attrs(node)
    is_gemm = node.op_type == "Gemm"
    alpha = abs(float(attrs.get("alpha", 1.0))) if is_gemm else 1.0
    zx = _zero_point(node, zx_i, consts)
    zw = _zero_point(node, zw_i, consts)
    mx = _centered_mag(*_bounded(result, node.input[x_i]), zx)
    wname = node.input[w_i]
    if wname in consts:
        W = consts[wname].astype(np.float64) - zw
        if is_gemm and attrs.get("transB", 0):
            W = W.T
        A = np.abs(W)
        if is_gemm:
            s = A.sum(axis=0).max(initial=0.0)
        elif node.op_type in ("Conv", "ConvInteger", "QLinearConv"):
            s = A.reshape(A.shape[0], -1).sum(axis=1).max(initial=0.0)
        elif A.ndim == 1:
            s = A.sum()
        else:
            s = A.sum(axis=-2).max(initial=0.0)
        acc = alpha * s * mx
    else:
        mw = _centered_mag(*_bounded(result, wname), zw)
        if node.op_type in ("Conv", "ConvInteger", "QLinearConv"):
            depth = int(np.prod(_shape(result, wname)[1:]))
        elif is_gemm:
            a = _shape(result, node.input[x_i])
            depth = a[0] if attrs.get("transA", 0) else a[1]
        else:
            depth = _shape(result, node.input[x_i])[-1]
        acc = alpha * depth * mx * mw
    if b_i is not None and b_i < len(node.input) and node.input[b_i]:
        blo, bhi = _bounded(result, node.input[b_i])
        beta = abs(float(attrs.get("beta", 1.0))) if is_gemm else 1.0
        acc += beta * max(abs(blo), abs(bhi))
    return acc


def _finding(
    node: str, tensor: str, kind: str, lo: float, hi: float, window: Window
) -> OverflowFinding:
    ilo, ihi = math.floor(lo), math.ceil(hi)
    return OverflowFinding(
        node,
        tensor,
        kind,
        ilo,
        ihi,
        _required_bits(ilo, ihi),
        window.lo <= ilo and ihi <= window.hi,
    )


def certify_no_overflow(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    *,
    bits: Optional[int] = None,
    field_modulus: Optional[int] = None,
) -> OverflowReport:
    """Check that no reachable value of ``model`` leaves a signed ``bits``-bit word, or a field.

    :param input_ranges: ``{input: (lo, hi)}``, merged over the model's ``onnxsim.range.*``
        annotations. Integer graph inputs without a range use their dtype range.
    :param bits: check against the signed ``bits``-bit window ``[-2**(bits-1), 2**(bits-1)-1]``.
    :param field_modulus: check against the window ``[-(p-1)/2, (p-1)/2]`` of an odd prime ``p``.
        Every tensor is checked, since a field circuit computes on integers only.

    Exactly one of ``bits`` and ``field_modulus`` must be given. ``report.fits`` is true only when
    every checked value is bounded and inside the window.
    """
    window = _window(bits, field_modulus)
    field = field_modulus is not None
    result = _interval.propagate(model, _feeds(model, input_ranges))
    elem = _interval._elem_types(model)
    consts = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    report = OverflowReport(window.name, [], [])

    def unanalysed(node_name: str, reason: str) -> None:
        entry = f"{node_name}: {reason}"
        if entry not in report.unanalysed:
            report.unanalysed.append(entry)

    def in_scope(name: str) -> bool:
        return field or elem.get(name) not in _FLOAT_TYPES

    for k, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}_{k}"
        covered = node.op_type in ("MatMulInteger", "ConvInteger")
        if node.op_type == "Cast" and node.input[0]:
            to = int(_interval._attrs(node).get("to", 0))
            if to in _NATIVE:
                covered = True
                src = node.input[0]
                try:
                    lo, hi = _bounded(result, src)
                except _Skip as e:
                    unanalysed(name, str(e))
                else:
                    native = Window(onnx.TensorProto.DataType.Name(to), *_NATIVE[to])
                    report.findings.append(_finding(name, src, "cast", lo, hi, native))
                    if in_scope(node.output[0]):
                        report.findings.append(
                            _finding(name, node.output[0], "tensor", lo, hi, window)
                        )
        if node.op_type in _ACCUMULATORS and node.output[0]:
            if node.op_type in _INTEGER_OPS or in_scope(node.output[0]):
                try:
                    acc = _accumulator_bound(node, result, consts)
                except _Skip as e:
                    unanalysed(name, str(e))
                else:
                    report.findings.append(
                        _finding(name, node.output[0], "accumulator", -acc, acc, window)
                    )
        if covered:
            continue  # output is the accumulator, or bounded by the input hull above
        for o in node.output:
            if not o or not in_scope(o):
                continue
            try:
                lo, hi = _bounded(result, o)
            except _Skip as e:
                unanalysed(name, str(e))
                continue
            report.findings.append(_finding(name, o, "tensor", lo, hi, window))
    return report
