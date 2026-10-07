"""Verify that an integer-quantized ONNX graph computes what its fake-quant reference does.

Given an integer-form graph -- ``MatMulInteger`` / ``ConvInteger`` /
``QLinearMatMul`` / ``QLinearConv`` and onnxsim's own
``DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul`` chain -- this module
answers, per layer and with certificates, three questions:

1. **NO-WRAP** (soundness): can the int32 accumulator wrap around for any input
   the layer can actually receive? The accumulator of one output channel is
   ``sum_k w_k * d_k`` with ``d_k = a_k - a_zero_point`` ranging over a box, so its
   extremes are attained at box vertices and are computed *exactly* in closed form
   (``P*d_hi - N*d_lo`` and ``P*d_lo - N*d_hi`` with ``P``/``N`` the sums of the
   positive/negative weights). A wrap is therefore not "possible", it is
   *reachable*, and the report carries a concrete input that reproduces it. For
   small layers the same claim is proved a second time from the two's-complement
   semantics themselves with Z3 bit-vectors (:func:`z3_no_wrap`); a disagreement
   between the two routes is reported as an internal inconsistency, never ignored.
2. **REQUANT-EXACT** (soundness for ``QLinear*``): the integer pipeline computes
   ``saturate(rne(fl32(acc) * fl32(sa*sb/sy)) + zy)``; the ideal (real-arithmetic)
   specification computes ``saturate(rne(acc * sa*sb/sy) + zy)``. The two can only
   differ where a half-way tie ``k + 1/2`` lies between the exact and the fp32
   value, i.e. in a window of at most ``eps*|t|`` around each of at most 255
   ties. That is a *finite* set of accumulators computed exactly
   (:func:`requant_differences`), so the claim is decided for **all** accumulators
   at once -- this decomposition (per-output dot product, plus the requantisation
   function checked once over its whole input range) is what scales to large
   layers. A differing accumulator is reported with an input that reaches it, or
   marked "reachability undecided" when the solver cannot say.
3. **TIE HAZARDS** (informational): the same set, viewed as "places where two
   runtimes that round differently (half-even vs half-away, fp32 vs fp64
   multiplier) silently disagree by 1 LSB", plus whether ``Cast<float>(acc)`` is
   exact (``|acc| <= 2**24``).

For onnxsim's dynamic-quantization chain it additionally proves the *relative*
error of ``Cast<float>(acc) * (Xs * Ws)`` against the exact real value
``acc * Xs * Ws`` (:func:`dynamic_rel_error_bound`), and, when a float reference
model is supplied, that the integer weights are the round-to-nearest
quantization of the reference weights.

What a verdict means -- so it is not over-read
==============================================

* ``proved`` / ``refuted`` are exact decisions about the *integer* semantics
  modelled here (two's-complement int32 accumulation, round-half-to-even,
  saturation, fp32 multiply). They say nothing about a runtime that deviates from
  that model; ``probe_u8s8_saturation`` exists because some AVX2 u8 x s8 kernels
  saturate int16 pair sums, which is *not* wraparound and not modelled.
* ``skipped`` always carries a reason (unsupported op, non-constant scale,
  unknown shape, per-row zero point, ...). It is never silently ``proved``.
* ``severity="soundness"`` findings decide :attr:`ModelReport.ok`;
  ``severity="informational"`` ones (tie hazards, exactness of the float cast)
  never do.
* Z3 use is deliberately small and bounded: every solver call has a timeout, a
  fresh ``z3.Context`` is used per proof (Z3 proofs are state-sensitive -- see
  the CI hang fixed in PR #2082 -- so no proof may depend on state left by an
  earlier one), and nothing here needs ``ForAll``.
"""

import dataclasses
import math
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
FP32_EXACT_INT = 2**24
_U32 = Fraction(1, 2**24)  # unit roundoff of fp32 with round-to-nearest
_MAX_REACH_CHECKS = 200  # candidate tie accumulators checked for reachability before giving up (bounds the cost)

PROVED = "proved"
REFUTED = "refuted"
SKIPPED = "skipped"
SOUNDNESS = "soundness"
INFORMATIONAL = "informational"

_INT_RANGES = {
    onnx.TensorProto.UINT8: (0, 255),
    onnx.TensorProto.INT8: (-128, 127),
    onnx.TensorProto.UINT16: (0, 65535),
    onnx.TensorProto.INT16: (-32768, 32767),
}
_NP_DTYPE = {
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.UINT16: np.uint16,
    onnx.TensorProto.INT16: np.int16,
}


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Finding:
    """One verdict about one property of one layer."""

    kind: str  # "no-wrap" | "requant-exact" | "tie-hazard" | "float-cast-exact" | ...
    severity: str  # SOUNDNESS | INFORMATIONAL
    verdict: str  # PROVED | REFUTED | SKIPPED
    detail: str = ""
    method: str = ""
    counterexample: Optional[Dict[str, np.ndarray]] = None


@dataclasses.dataclass
class LayerReport:
    name: str
    op_type: str
    findings: List[Finding] = dataclasses.field(default_factory=list)

    def of(self, kind: str) -> List[Finding]:
        return [f for f in self.findings if f.kind == kind]

    @property
    def sound(self) -> bool:
        """No soundness finding was refuted (skipped ones are not claimed either way)."""
        return not any(
            f.severity == SOUNDNESS and f.verdict == REFUTED for f in self.findings
        )

    @property
    def proved(self) -> bool:
        """Every soundness finding was proved (and there was at least one)."""
        s = [f for f in self.findings if f.severity == SOUNDNESS]
        return bool(s) and all(f.verdict == PROVED for f in s)


@dataclasses.dataclass
class ModelReport:
    layers: List[LayerReport]
    notes: List[str] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.layers) and all(layer.proved for layer in self.layers)

    def __str__(self) -> str:
        out = [f"quant_int_verify: {'OK' if self.ok else 'NOT PROVED'}"]
        for layer in self.layers:
            out.append(f"  {layer.name} ({layer.op_type})")
            for f in layer.findings:
                out.append(
                    f"    [{f.severity[:4]}] {f.kind}: {f.verdict} -- {f.detail}"
                )
        out += [f"  note: {n}" for n in self.notes]
        return "\n".join(out)


# --------------------------------------------------------------------------
# Exact arithmetic helpers
# --------------------------------------------------------------------------


def rne_fraction(v: Fraction) -> int:
    """Round half to even of an exact rational."""
    fl = v.numerator // v.denominator
    rem = v - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        return fl + 1
    return fl


def fp32_requant(acc: int, m32: Any, zy: int, qmin: int, qmax: int) -> int:
    """The runtime's integer->integer requantisation: ``sat(rne(fl32(acc)*m32) + zy)``.

    ``np.float32(int)`` converts through a double (exact for |acc| < 2**53) and
    then rounds once to fp32 with round-to-nearest-even; ``np.rint`` is half-even.
    """
    r = np.float32(float(acc)) * np.float32(m32)
    q = int(np.rint(r)) + zy
    return max(qmin, min(qmax, q))


def real_requant(acc: int, m_real: Fraction, zy: int, qmin: int, qmax: int) -> int:
    """The ideal specification: ``sat(rne(acc * m) + zy)`` in exact arithmetic."""
    q = rne_fraction(Fraction(acc) * m_real) + zy
    return max(qmin, min(qmax, q))


def _frac(x: Any) -> Fraction:
    return Fraction(float(x))


def requant_multiplier(sa: Any, sb: Any, sy: Any) -> Tuple[Fraction, Any]:
    """``(exact real multiplier, fp32 multiplier a runtime uses)`` for ``sa*sb/sy``.

    The exact value is the rational ``sa*sb/sy`` of the fp32 scales as stored. The
    fp32 multiplier is ``fl32(fl32(sa*sb)/sy)``; onnxruntime was observed to agree with
    every other evaluation order on random data, but they are not provably identical
    for every input, which is why callers pass whichever multiplier their runtime uses.
    """
    m_real = _frac(sa) * _frac(sb) / _frac(sy)
    m32 = np.float32(np.float32(np.float32(sa) * np.float32(sb)) / np.float32(sy))
    return m_real, m32


def requant_rel_eps(m_real: Fraction, m32: Any) -> Fraction:
    """Sound relative error of the fp32 pipeline vs the exact one: ``|p - v| <= eps*|v|``.

    ``p = fl(fl(acc) * m32)``: the cast contributes <= u, the multiply <= u, and the
    multiplier itself differs from the exact one by ``|m32 - m| / m``.
    """
    e0 = abs(_frac(m32) - m_real) / m_real
    return (1 + e0) * (1 + _U32) ** 2 - 1


def requant_differences(
    m_real: Fraction,
    m32: Any,
    zy: int,
    qmin: int,
    qmax: int,
    acc_lo: int,
    acc_hi: int,
    max_candidates: int = 2_000_000,
) -> Optional[List[int]]:
    """Every accumulator in ``[acc_lo, acc_hi]`` where fp32 and exact requant differ.

    Exact and complete: the two pipelines produce the same real number up to a
    relative ``eps``, so their roundings can differ only if a tie ``t = k + 1/2`` lies
    between them, hence only if ``|acc*m - t| <= eps*|t| / (1 - eps)``. Only ties whose
    two neighbouring outputs are not both saturated matter, and there are at most
    ``qmax - qmin`` of them. Each window is enumerated and both pipelines evaluated
    exactly. Returns ``None`` if the windows hold more than ``max_candidates`` values
    (then nothing is claimed).
    """
    eps = requant_rel_eps(m_real, m32)
    diffs: List[int] = []
    seen = 0
    for k in range(qmin - zy, qmax - zy):
        t = Fraction(2 * k + 1, 2)
        delta = eps * abs(t) / (1 - eps)
        lo = max(acc_lo, math.ceil((t - delta) / m_real))
        hi = min(acc_hi, math.floor((t + delta) / m_real))
        if hi < lo:
            continue
        seen += hi - lo + 1
        if seen > max_candidates:
            return None
        for acc in range(lo, hi + 1):
            if fp32_requant(acc, m32, zy, qmin, qmax) != real_requant(
                acc, m_real, zy, qmin, qmax
            ):
                diffs.append(acc)
    return sorted(set(diffs))


def dynamic_rel_error_bound(acc_abs_max: int) -> Fraction:
    """Relative error bound of ``Cast<float>(acc) * (Xs * Ws)`` vs the exact ``acc*Xs*Ws``.

    ``fl(fl(acc) * fl(Xs*Ws))`` is ``acc*Xs*Ws*(1+d1)(1+d2)(1+d3)`` with ``|d_i| <= u``
    (``d1 = 0`` when ``|acc| <= 2**24`` because the cast is then exact). The product
    ``(1+d1)(1+d2)(1+d3)`` is multilinear in the ``d_i``, so its extremes over the box
    are at the 8 corners; they are enumerated exactly. Sound for any operand values.
    """
    u = _U32
    d1s = [Fraction(0)] if acc_abs_max <= FP32_EXACT_INT else [-u, u]
    worst = Fraction(0)
    for a in d1s:
        for b in (-u, u):
            for c in (-u, u):
                worst = max(worst, abs((1 + a) * (1 + b) * (1 + c) - 1))
    return worst


# --------------------------------------------------------------------------
# Accumulator ranges (closed form, exact)
# --------------------------------------------------------------------------


def accumulator_range(
    w: np.ndarray, d_lo: int, d_hi: int, bias: int = 0
) -> Tuple[int, int]:
    """Exact ``(min, max)`` of ``bias + sum_k w_k * d_k`` over ``d_k in [d_lo, d_hi]``.

    ``w`` is the integer weight column already offset by its zero point. Each term
    is independent, so the extremes sit at box vertices: ``max = P*d_hi - N*d_lo`` and
    ``min = P*d_lo - N*d_hi`` where ``P``/``N`` are the sums of positive weights and
    of absolute negative weights.
    """
    wi = np.asarray(w, dtype=np.int64).reshape(-1)
    p = int(wi[wi > 0].sum())
    n = int((-wi[wi < 0]).sum())
    return bias + p * d_lo - n * d_hi, bias + p * d_hi - n * d_lo


def _vertex_input(w: np.ndarray, d_lo: int, d_hi: int, maximise: bool) -> np.ndarray:
    """The vertex ``d`` reaching the max (or min) of ``sum w_k d_k``."""
    wi = np.asarray(w, dtype=np.int64).reshape(-1)
    hi_if_pos = d_hi if maximise else d_lo
    lo_if_pos = d_lo if maximise else d_hi
    return np.where(wi >= 0, hi_if_pos, lo_if_pos).astype(np.int64)


# --------------------------------------------------------------------------
# Z3 encoders (small layers; fresh Context + timeout per proof)
# --------------------------------------------------------------------------


def _z3():
    try:
        import z3
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("this check needs z3-solver (the 'verify' extra)") from e
    return z3


def z3_no_wrap(
    w: Sequence[int],
    d_lo: int,
    d_hi: int,
    bias: int = 0,
    timeout_ms: int = 10_000,
) -> Tuple[str, Optional[List[int]]]:
    """Decide from two's-complement semantics whether ``bias + sum w_k d_k`` can wrap int32.

    Builds the 32-bit accumulation exactly as hardware does (sign-extended products
    summed modulo 2**32) next to the unbounded sum, and asks Z3 for ``d`` with
    ``sign_extend(bv_acc) != exact``. Returns ``("proved", None)`` (no wrap for any
    ``d`` in the box), ``("refuted", d)`` with a reaching input, or ``("skipped", None)``
    on timeout. Intended for small ``len(w)``; uses its own ``z3.Context``.
    """
    z3 = _z3()
    ctx = z3.Context()
    n = len(w)
    # Pure bit-vectors (Z3's Int<->BV conversions are far slower): 32-bit modular
    # accumulation next to a 64-bit exact one. The 64-bit sum cannot itself overflow for
    # the small layers this is meant for (|w| < 2**31, |d| < 2**17, n <= a few dozen).
    ds = [z3.BitVec(f"d{k}", 32, ctx) for k in range(n)]
    s = z3.Solver(ctx=ctx)
    s.set("timeout", timeout_ms)
    for d in ds:
        s.add(
            d >= z3.BitVecVal(d_lo, 32, ctx), d <= z3.BitVecVal(d_hi, 32, ctx)
        )  # signed comparisons
    wrapped = z3.BitVecVal(bias, 32, ctx)
    exact = z3.BitVecVal(bias, 64, ctx)
    for wk, d in zip(w, ds):
        wrapped = wrapped + z3.BitVecVal(int(wk), 32, ctx) * d
        exact = exact + z3.BitVecVal(int(wk), 64, ctx) * z3.SignExt(32, d)
    s.add(z3.SignExt(32, wrapped) != exact)
    r = s.check()
    if r == z3.unsat:
        return PROVED, None
    if r == z3.sat:
        m = s.model()
        return REFUTED, [m.eval(d, model_completion=True).as_signed_long() for d in ds]
    return SKIPPED, None


def z3_requant_expr(
    ctx: Any, acc_bv: Any, m32: Any, zy: int, qmin: int, qmax: int
) -> Any:
    """Z3 expression (Int) of ``sat(rne(fl32(acc) * m32) + zy)`` using the floating-point theory.

    ``acc_bv`` is a signed 32-bit bit-vector. ``fpToFP`` from a signed bit-vector rounds
    to nearest-even like the runtime's cast; the multiply is an fp32 multiply with
    RNE; ``fpRoundToIntegral(RNE)`` is half-even rounding. Exposed so tests can check
    the encoding against onnxruntime/numpy on concrete accumulators.
    """
    z3 = _z3()
    # Every FP constructor that takes an optional ``ctx`` defaults to the GLOBAL context, which
    # would silently mix contexts with our fresh one -- so ``ctx`` is passed explicitly throughout.
    rm = z3.RNE(ctx)
    fp = z3.Float32(ctx)
    a = z3.fpSignedToFP(rm, acc_bv, fp, ctx)
    m = z3.FPVal(float(np.float32(m32)), None, fp, ctx)
    prod = z3.fpMul(rm, a, m, ctx)
    rounded = z3.fpRoundToIntegral(rm, prod, ctx)
    q = (
        z3.BV2Int(z3.fpToSBV(rm, rounded, z3.BitVecSort(64, ctx), ctx), is_signed=True)
        + zy
    )
    return z3.If(
        q < qmin, z3.IntVal(qmin, ctx), z3.If(q > qmax, z3.IntVal(qmax, ctx), q)
    )


def z3_eval_requant(
    acc: int, m32: Any, zy: int, qmin: int, qmax: int, timeout_ms: int = 10_000
) -> Optional[int]:
    """Evaluate :func:`z3_requant_expr` on one concrete accumulator (fresh context)."""
    z3 = _z3()
    ctx = z3.Context()
    s = z3.Solver(ctx=ctx)
    s.set("timeout", timeout_ms)
    x = z3.BitVec("acc", 32, ctx)
    out = z3.Int("out", ctx)
    s.add(x == z3.BitVecVal(acc, 32, ctx))
    s.add(out == z3_requant_expr(ctx, x, m32, zy, qmin, qmax))
    if s.check() != z3.sat:
        return None
    return s.model().eval(out, model_completion=True).as_long()


def z3_reach(
    w: Sequence[int],
    d_lo: int,
    d_hi: int,
    target: int,
    bias: int = 0,
    timeout_ms: int = 10_000,
) -> Tuple[str, Optional[List[int]]]:
    """Is ``bias + sum w_k d_k == target`` reachable with ``d_k in [d_lo, d_hi]`` (integers)?"""
    z3 = _z3()
    ctx = z3.Context()
    s = z3.Solver(ctx=ctx)
    s.set("timeout", timeout_ms)
    ds = [z3.Int(f"d{k}", ctx) for k in range(len(w))]
    for d in ds:
        s.add(d >= d_lo, d <= d_hi)
    s.add(
        z3.Sum([z3.IntVal(int(wk), ctx) * d for wk, d in zip(w, ds)]) + bias == target
    )
    r = s.check()
    if r == z3.sat:
        m = s.model()
        return "reachable", [m.eval(d, model_completion=True).as_long() for d in ds]
    if r == z3.unsat:
        return "unreachable", None
    return "unknown", None


def _greedy_reach(
    w: np.ndarray,
    d_lo: int,
    d_hi: int,
    target: int,
    bias: int = 0,
    order: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Constructively find ``d`` in the box with ``bias + w.d == target``, or ``None``.

    Starts at the minimising vertex and adds the deficit tap by tap in steps of ``|w_k|``
    (largest weights first; pass a precomputed ``order`` to avoid re-sorting). Always
    verified exactly before it is returned.
    """
    wi = np.asarray(w, dtype=np.int64).reshape(-1)
    d = _vertex_input(wi, d_lo, d_hi, maximise=False).copy()
    deficit = target - (bias + int((wi * d).sum()))
    if deficit < 0:
        return None
    span = d_hi - d_lo
    for k in order if order is not None else np.argsort(-np.abs(wi)):
        wk = int(abs(wi[k]))
        if wk == 0 or deficit == 0:
            continue
        steps = min(span, deficit // wk)
        if steps:
            d[k] += steps if wi[k] > 0 else -steps
            deficit -= steps * wk
    if deficit != 0:
        return None
    return d if bias + int((wi * d).sum()) == target else None


def _reach(
    w: np.ndarray,
    d_lo: int,
    d_hi: int,
    target: int,
    bias: int,
    z3_max_k: int,
    timeout_ms: int,
    order: Optional[np.ndarray] = None,
) -> Tuple[str, Optional[np.ndarray]]:
    """Is accumulator ``target`` reachable? ``("reachable", d)`` / ``("unreachable", None)`` / ``("unknown", None)``."""
    wi = np.asarray(w, dtype=np.int64).reshape(-1)
    lo, hi = accumulator_range(wi, d_lo, d_hi, bias)
    if target < lo or target > hi:
        return "unreachable", None
    g = _greedy_reach(wi, d_lo, d_hi, target, bias, order)
    if g is not None:
        return "reachable", g
    nz = wi[wi != 0]
    if len(nz) <= z3_max_k:
        try:
            verdict, d = z3_reach(nz.tolist(), d_lo, d_hi, target, bias, timeout_ms)
        except ImportError:
            return "unknown", None
        if d is None:
            return verdict, None
        full = np.full(wi.shape, d_lo, dtype=np.int64)
        full[wi != 0] = d
        return verdict, full
    return "unknown", None


# --------------------------------------------------------------------------
# Graph helpers
# --------------------------------------------------------------------------


def _attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _consts(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    out = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output:
            for a in n.attribute:
                if a.name == "value":
                    out[n.output[0]] = numpy_helper.to_array(a.t)
    return out


def _scalar(consts: Dict[str, np.ndarray], name: str) -> Optional[Any]:
    """The value of a scalar-like constant input, or ``None`` if absent/non-constant/non-scalar."""
    if not name or name not in consts:
        return None
    a = consts[name]
    return a.reshape(-1)[0] if a.size == 1 else None


def _elem_types(model: onnx.ModelProto) -> Dict[str, int]:
    try:
        g = onnx.shape_inference.infer_shapes(model).graph
    except Exception:
        g = model.graph
    out: Dict[str, int] = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        if vi.type.HasField("tensor_type"):
            out[vi.name] = vi.type.tensor_type.elem_type
    for t in model.graph.initializer:
        out[t.name] = t.data_type
    return out


def _shapes(model: onnx.ModelProto) -> Dict[str, Tuple[int, ...]]:
    try:
        g = onnx.shape_inference.infer_shapes(model).graph
    except Exception:
        g = model.graph
    out: Dict[str, Tuple[int, ...]] = {}
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        tt = vi.type.tensor_type
        if vi.type.HasField("tensor_type") and tt.HasField("shape"):
            dims = [
                d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else None
                for d in tt.shape.dim
            ]
            if None not in dims:
                out[vi.name] = tuple(int(d) for d in dims if d is not None)
    return out


@dataclasses.dataclass
class _Geometry:
    """Receptive-field facts for a Conv layer, used to decide whether padding matters."""

    padded: bool
    interior: Optional[
        Tuple[int, int]
    ]  # an output position whose taps all lie inside, if known
    x_hw: Optional[Tuple[int, int]]


def _conv_geometry(
    attrs: Dict[str, Any], kshape: Tuple[int, ...], x_shape: Optional[Tuple[int, ...]]
) -> _Geometry:
    kh, kw = kshape[-2], kshape[-1]
    strides = list(attrs.get("strides", [1, 1]))
    dil = list(attrs.get("dilations", [1, 1]))
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    padded = any(p != 0 for p in pads)
    if x_shape is None or len(x_shape) != 4:
        return _Geometry(padded, None if padded else (0, 0), None)
    h, w = x_shape[2], x_shape[3]
    interior = None
    oy = -(-pads[0] // strides[0])
    ox = -(-pads[1] // strides[1])
    oh = (h + pads[0] + pads[2] - dil[0] * (kh - 1) - 1) // strides[0] + 1
    ow = (w + pads[1] + pads[3] - dil[1] * (kw - 1) - 1) // strides[1] + 1
    if (
        oy < oh
        and ox < ow
        and oy * strides[0] - pads[0] + dil[0] * (kh - 1) <= h - 1
        and ox * strides[1] - pads[1] + dil[1] * (kw - 1) <= w - 1
    ):
        interior = (oy, ox)
    return _Geometry(padded, interior, (h, w))


def _bound_with_padding(
    w: np.ndarray, d_lo: int, d_hi: int, padded: bool
) -> Tuple[int, int, bool, bool]:
    """``(min, max, min_exact_if_interior, max_exact_if_interior)`` of ``sum w_k d_k`` over a Conv tap set.

    Padded taps contribute exactly 0 (the input is padded with its zero point, so
    ``d = 0``). With padding present, a tap may be absent, so each term is clamped
    towards 0 for the *bound*; the bound is attained iff no clamp was needed.
    """
    wi = np.asarray(w, dtype=np.int64).reshape(-1)
    c_hi = np.maximum(wi * d_lo, wi * d_hi)
    c_lo = np.minimum(wi * d_lo, wi * d_hi)
    if not padded:
        return int(c_lo.sum()), int(c_hi.sum()), True, True
    return (
        int(np.minimum(c_lo, 0).sum()),
        int(np.maximum(c_hi, 0).sum()),
        bool((c_lo <= 0).all()),
        bool((c_hi >= 0).all()),
    )


# --------------------------------------------------------------------------
# Layer analysis
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _Layer:
    """Everything the checks need, extracted once from a node."""

    name: str
    op_type: str
    wq: np.ndarray  # [channels, taps], already offset by the weight zero point (int64)
    d_lo: int  # range of (a - a_zero_point)
    d_hi: int
    bias: np.ndarray  # int32 per channel (zeros if none)
    is_conv: bool
    geometry: Optional[_Geometry]
    za: int
    a_dtype: int
    a_shape: Optional[Tuple[int, ...]]
    conv_attrs: Dict[str, Any]
    wshape: Tuple[int, ...]
    requant: Optional[Dict[str, Any]]  # QLinear*: scales / zero points / dtype
    notes: List[str]


def _act_range_from(
    za: int, a_dtype: int, act_range: Optional[Tuple[int, int]]
) -> Tuple[int, int, str]:
    qlo, qhi = _INT_RANGES.get(a_dtype, (0, 255))
    note = ""
    if act_range is not None:
        qlo, qhi = max(qlo, int(act_range[0])), min(qhi, int(act_range[1]))
        note = f"activation codes restricted to [{qlo}, {qhi}]"
    return qlo - za, qhi - za, note


def _extract_layer(
    node: onnx.NodeProto,
    consts: Dict[str, np.ndarray],
    elem_types: Dict[str, int],
    shapes: Dict[str, Tuple[int, ...]],
    act_range: Optional[Tuple[int, int]],
) -> Tuple[Optional[_Layer], Optional[str]]:
    """Build a :class:`_Layer` from an integer matmul/conv node, or ``(None, reason)``."""
    op = node.op_type
    ins = list(node.input) + [""] * 9
    if op in ("MatMulInteger", "ConvInteger"):
        a, w, azp, wzp = ins[0], ins[1], ins[2], ins[3]
        qlin = None
        bias_name = ""
    elif op in ("QLinearMatMul", "QLinearConv"):
        a, asc, azp, w, wsc, wzp, ysc, yzp = ins[:8]
        bias_name = ins[8] if op == "QLinearConv" else ""
        qlin = (asc, wsc, ysc, yzp)
    else:
        return None, f"unsupported op {op}"
    if w not in consts:
        return None, "weight is not a constant"
    wq = consts[w].astype(np.int64)
    za = 0
    if azp:
        z = _scalar(consts, azp)
        if z is None:
            return None, "activation zero point is not a scalar constant"
        za = int(z)
    a_dtype = elem_types.get(a) or (
        onnx.helper.np_dtype_to_tensor_dtype(consts[azp].dtype)
        if azp in consts
        else onnx.TensorProto.UINT8
    )
    notes: List[str] = []
    if a_dtype not in _INT_RANGES:
        return None, f"unsupported activation dtype {a_dtype}"
    if not azp and a not in elem_types:
        notes.append("activation dtype unknown, assumed uint8")
    if wzp:
        if wzp not in consts:
            return None, "weight zero point is not a constant"
        zp = consts[wzp].astype(np.int64)
    else:
        zp = np.zeros((), dtype=np.int64)
    d_lo, d_hi, rn = _act_range_from(za, a_dtype, act_range)
    if rn:
        notes.append(rn)
    is_conv = op in ("ConvInteger", "QLinearConv")
    attrs = _attrs(node)
    if is_conv:
        if wq.ndim != 4:
            return None, "only 2-D convolution is handled"
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            return None, "auto_pad is not handled"
        if zp.ndim == 1 and zp.size == wq.shape[0]:
            wq = wq - zp.reshape(-1, 1, 1, 1)
        else:
            wq = wq - zp.reshape(()) if zp.size == 1 else wq
        wmat = wq.reshape(wq.shape[0], -1)
        geom: Optional[_Geometry] = _conv_geometry(
            attrs, tuple(wq.shape), shapes.get(a)
        )
    else:
        if wq.ndim != 2:
            return None, "only 2-D (K, N) MatMul weights are handled"
        if zp.ndim == 1 and zp.size == wq.shape[1]:
            wq = wq - zp.reshape(1, -1)
        elif zp.size == 1:
            wq = wq - zp.reshape(())
        else:
            return None, "per-row weight zero point is not handled"
        wmat = wq.T  # [N, K]
        geom = None
    bias = np.zeros(wmat.shape[0], dtype=np.int64)
    if bias_name:
        if bias_name not in consts:
            return None, "bias is not a constant"
        bias = consts[bias_name].astype(np.int64).reshape(-1)
    requant = None
    if qlin is not None:
        asc, wsc, ysc, yzp = qlin
        sa, sy, zy = _scalar(consts, asc), _scalar(consts, ysc), _scalar(consts, yzp)
        sw = consts.get(wsc)
        if sa is None or sy is None or zy is None or sw is None:
            return None, "requantisation scales / output zero point are not constants"
        ydtype = onnx.helper.np_dtype_to_tensor_dtype(consts[yzp].dtype)
        if ydtype not in _INT_RANGES:
            return None, f"unsupported output dtype {ydtype}"
        sw = sw.reshape(-1)
        if sw.size not in (1, wmat.shape[0]):
            return None, "weight scale is neither per-tensor nor per-output-channel"
        requant = {"sa": sa, "sw": sw, "sy": sy, "zy": int(zy), "ydtype": ydtype}
    return (
        _Layer(
            name=node.name or f"{op}_{node.output[0]}",
            op_type=op,
            wq=wmat,
            d_lo=d_lo,
            d_hi=d_hi,
            bias=bias,
            is_conv=is_conv,
            geometry=geom,
            za=za,
            a_dtype=a_dtype,
            a_shape=shapes.get(a),
            conv_attrs=attrs,
            wshape=tuple(consts[w].shape),
            requant=requant,
            notes=notes,
        ),
        None,
    )


def _channel_ranges(
    layer: _Layer,
) -> Tuple[List[int], List[int], List[bool], List[bool]]:
    """Per-channel ``(min, max, min_attained, max_attained)`` of the int32 accumulator."""
    mins: List[int] = []
    maxs: List[int] = []
    amin: List[bool] = []
    amax: List[bool] = []
    for c in range(layer.wq.shape[0]):
        b = (
            int(layer.bias[c])
            if layer.bias.size > 1
            else int(layer.bias.reshape(-1)[0])
        )
        if layer.is_conv and layer.geometry is not None:
            lo, hi, e_lo, e_hi = _bound_with_padding(
                layer.wq[c], layer.d_lo, layer.d_hi, layer.geometry.padded
            )
            interior = layer.geometry.interior is not None
            mins.append(lo + b)
            maxs.append(hi + b)
            amin.append(e_lo and (interior or not layer.geometry.padded))
            amax.append(e_hi and (interior or not layer.geometry.padded))
        else:
            lo, hi = accumulator_range(layer.wq[c], layer.d_lo, layer.d_hi, b)
            mins.append(lo)
            maxs.append(hi)
            amin.append(True)
            amax.append(True)
    return mins, maxs, amin, amax


def _conv_counterexample(
    layer: _Layer, d: np.ndarray, co: int
) -> Dict[str, np.ndarray]:
    """An input tensor putting tap vector ``d`` at one fully-interior output position."""
    wshape = layer.wshape
    cout, cg, kh, kw = wshape
    attrs = layer.conv_attrs
    strides = list(attrs.get("strides", [1, 1]))
    dil = list(attrs.get("dilations", [1, 1]))
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    group = int(attrs.get("group", 1))
    oy = -(-pads[0] // strides[0])
    ox = -(-pads[1] // strides[1])
    h = oy * strides[0] - pads[0] + dil[0] * (kh - 1) + 1
    w = ox * strides[1] - pads[1] + dil[1] * (kw - 1) + 1
    if layer.geometry is not None and layer.geometry.x_hw is not None:
        h, w = max(h, layer.geometry.x_hw[0]), max(w, layer.geometry.x_hw[1])
    x = np.full((1, cg * group, h, w), layer.za, dtype=np.int64)
    g = co // (cout // group)
    dd = d.reshape(cg, kh, kw)
    for c in range(cg):
        for i in range(kh):
            for j in range(kw):
                x[
                    0,
                    g * cg + c,
                    oy * strides[0] - pads[0] + i * dil[0],
                    ox * strides[1] - pads[1] + j * dil[1],
                ] = layer.za + dd[c, i, j]
    return {"a": x.astype(_NP_DTYPE[layer.a_dtype])}


def _no_wrap_findings(layer: _Layer, z3_max_k: int, timeout_ms: int) -> List[Finding]:
    mins, maxs, amin, amax = _channel_ranges(layer)
    worst_hi = max(maxs)
    worst_lo = min(mins)
    c_hi = int(np.argmax(maxs))
    c_lo = int(np.argmin(mins))
    taps = layer.wq.shape[1]
    method = (
        "closed-form vertex bounds (exact)"
        if not (layer.is_conv and layer.geometry and layer.geometry.padded)
        else "closed-form bounds"
    )
    over_hi = worst_hi > INT32_MAX
    under_lo = worst_lo < INT32_MIN
    if not over_hi and not under_lo:
        detail = (
            f"every accumulator lies in [{worst_lo}, {worst_hi}] "
            f"(int32 is [{INT32_MIN}, {INT32_MAX}]; {taps} taps/channel, "
            f"{layer.wq.shape[0]} channels, head-room {INT32_MAX - max(worst_hi, -worst_lo)})"
        )
        findings = [Finding("no-wrap", SOUNDNESS, PROVED, detail, method)]
        findings += _z3_cross_check(
            layer, c_hi, c_lo, z3_max_k, timeout_ms, expect_wrap=False
        )
        return findings
    # Overflow bound exceeded: is it attained (reachable)?
    for direction, bad, c, attained in (
        ("max", over_hi, c_hi, amax[c_hi]),
        ("min", under_lo, c_lo, amin[c_lo]),
    ):
        if not bad:
            continue
        val = worst_hi if direction == "max" else worst_lo
        if not attained:
            return [
                Finding(
                    "no-wrap",
                    SOUNDNESS,
                    SKIPPED,
                    f"the bound {val} exceeds int32, but with padding it is only an upper bound and its "
                    f"attainability could not be confirmed for this input geometry",
                    method,
                )
            ]
        d = _vertex_input(
            layer.wq[c], layer.d_lo, layer.d_hi, maximise=(direction == "max")
        )
        if layer.is_conv:
            cex = _conv_counterexample(layer, d, c)
        else:
            cex = {"a": (d + layer.za).reshape(1, -1).astype(_NP_DTYPE[layer.a_dtype])}
        wrapped = ((val - INT32_MIN) % 2**32) + INT32_MIN
        out = [
            Finding(
                "no-wrap",
                SOUNDNESS,
                REFUTED,
                f"channel {c} reaches an accumulator of {val} (> int32 {'max' if direction == 'max' else 'min'}); "
                f"the kernel wraps it to {wrapped}",
                method,
                counterexample=cex,
            )
        ]
        out += _z3_cross_check(layer, c, c, z3_max_k, timeout_ms, expect_wrap=True)
        return out
    return []  # pragma: no cover


def _z3_cross_check(
    layer: _Layer,
    c_hi: int,
    c_lo: int,
    z3_max_k: int,
    timeout_ms: int,
    expect_wrap: bool,
) -> List[Finding]:
    """Re-derive the no-wrap verdict from two's-complement semantics for small layers."""
    if layer.is_conv and layer.geometry is not None and layer.geometry.padded:
        return []
    for c in dict.fromkeys((c_hi, c_lo)):
        w = layer.wq[c]
        nz = w[w != 0]
        if len(nz) > z3_max_k:
            return []
        try:
            verdict, _ = z3_no_wrap(
                nz.tolist(),
                layer.d_lo,
                layer.d_hi,
                int(layer.bias[c] if layer.bias.size > 1 else layer.bias[0]),
                timeout_ms,
            )
        except ImportError:
            return []
        if verdict == SKIPPED:
            return [
                Finding(
                    "no-wrap-z3",
                    INFORMATIONAL,
                    SKIPPED,
                    f"Z3 gave up on channel {c}",
                    "z3",
                )
            ]
        if (verdict == REFUTED) != expect_wrap:
            return [
                Finding(
                    "internal-inconsistency",
                    SOUNDNESS,
                    REFUTED,
                    f"closed form and the Z3 two's-complement encoding disagree on channel {c} "
                    f"(closed form says wrap={expect_wrap}, Z3 says wrap={verdict == REFUTED})",
                    "z3",
                )
            ]
    return [
        Finding(
            "no-wrap-z3",
            INFORMATIONAL,
            PROVED,
            "Z3 bit-vector semantics agree with the closed form",
            "z3 (fresh context)",
        )
    ]


def _requant_findings(layer: _Layer, z3_max_k: int, timeout_ms: int) -> List[Finding]:
    rq = layer.requant
    assert rq is not None
    qmin, qmax = _INT_RANGES[rq["ydtype"]]
    mins, maxs, _, _ = _channel_ranges(layer)
    out: List[Finding] = []
    worst_abs = max(max(abs(v) for v in mins), max(abs(v) for v in maxs))
    out.append(
        Finding(
            "float-cast-exact",
            INFORMATIONAL,
            PROVED if worst_abs <= FP32_EXACT_INT else REFUTED,
            f"max |accumulator| = {worst_abs}; Cast<float> is exact up to 2**24 = {FP32_EXACT_INT}"
            + (
                ""
                if worst_abs <= FP32_EXACT_INT
                else " -- beyond that the cast rounds (relative error <= 2**-24)"
            ),
            "closed form",
        )
    )
    # Candidate tie accumulators per channel are exact (see requant_differences); which of them
    # an input can actually reach is decided lazily -- the first reachable one is enough to
    # refute "bit-exact", and re-checking thousands of candidates over a 70000-tap channel
    # would only repeat the same answer.
    candidates: List[Tuple[int, int]] = []
    incomplete = False
    sw = rq["sw"]
    for c in range(layer.wq.shape[0]):
        s_w = sw[c] if sw.size > 1 else sw[0]
        m_real, m32 = requant_multiplier(rq["sa"], s_w, rq["sy"])
        lo, hi = max(mins[c], INT32_MIN), min(maxs[c], INT32_MAX)
        diffs = requant_differences(m_real, m32, rq["zy"], qmin, qmax, lo, hi)
        if diffs is None:
            incomplete = True
            continue
        candidates += [(c, acc) for acc in diffs]
    reachable: Optional[Tuple[int, int, Optional[np.ndarray]]] = None
    undecided = 0
    checked = 0
    orders: Dict[int, np.ndarray] = {}
    for c, acc in candidates:
        if layer.is_conv:
            # Reaching one specific accumulator through a convolution's tap structure is not decided here.
            undecided += 1
            continue
        if checked >= _MAX_REACH_CHECKS:
            undecided += 1
            continue
        checked += 1
        b = int(layer.bias[c] if layer.bias.size > 1 else layer.bias[0])
        if c not in orders:
            orders[c] = np.argsort(-np.abs(layer.wq[c]))
        status, d = _reach(
            layer.wq[c], layer.d_lo, layer.d_hi, acc, b, z3_max_k, timeout_ms, orders[c]
        )
        if status == "reachable":
            reachable = (c, acc, d)
            break
        if status == "unknown":
            undecided += 1
    # Soundness: whatever differs, it differs by at most 1 LSB. Differences only occur at
    # ties (enumerated above), and the fp32 value is within eps*|v| of the exact one;
    # eps*|v| < 1 for every v the clamp can distinguish, so the roundings differ by <= 1.
    worst_eps = max(
        requant_rel_eps(
            *requant_multiplier(rq["sa"], (sw[c] if sw.size > 1 else sw[0]), rq["sy"])
        )
        for c in range(layer.wq.shape[0])
    )
    reach_v = max(abs(qmin - rq["zy"]), abs(qmax - rq["zy"])) + 1
    within = worst_eps * reach_v < 1
    out.append(
        Finding(
            "requant-1lsb",
            SOUNDNESS,
            PROVED if within and not incomplete else SKIPPED,
            f"fp32 requantisation is within relative {float(worst_eps):.3g} of the ideal value, so the "
            f"integer output differs from the ideal specification by at most 1 LSB wherever it differs"
            if within and not incomplete
            else "could not bound the requantisation difference",
            "tie-window enumeration",
        )
    )
    if incomplete:
        out.append(
            Finding(
                "requant-bit-exact",
                INFORMATIONAL,
                SKIPPED,
                "too many candidate accumulators to enumerate",
            )
        )
    elif reachable is not None:
        c, acc, d = reachable
        cex = None
        if d is not None:
            cex = {"a": (d + layer.za).reshape(1, -1).astype(_NP_DTYPE[layer.a_dtype])}
        out.append(
            Finding(
                "requant-bit-exact",
                INFORMATIONAL,
                REFUTED,
                f"accumulator {acc} of channel {c} is reachable and rounds differently from the ideal "
                f"specification ({len(candidates)} candidate tie accumulators in total); a half-way tie that a "
                f"runtime rounding half-away, or using a double multiplier, would resolve differently",
                "tie-window enumeration (exact) + reachability witness",
                counterexample=cex,
            )
        )
    elif undecided:
        out.append(
            Finding(
                "requant-bit-exact",
                INFORMATIONAL,
                SKIPPED,
                f"{len(candidates)} candidate tie accumulators; reachability of {undecided} could not be decided",
                "tie-window enumeration (exact)",
            )
        )
    else:
        out.append(
            Finding(
                "requant-bit-exact",
                INFORMATIONAL,
                PROVED,
                "the fp32 pipeline equals the ideal real-arithmetic requantisation for every reachable accumulator"
                + (
                    f" ({len(candidates)} candidate tie accumulators, none reachable)"
                    if candidates
                    else ""
                ),
                "tie-window enumeration (exact)",
            )
        )
    return out


def verify_layer(
    node: onnx.NodeProto,
    consts: Dict[str, np.ndarray],
    act_range: Optional[Tuple[int, int]] = None,
    act_dtype: Optional[int] = None,
    x_shape: Optional[Tuple[int, ...]] = None,
    z3_max_k: int = 24,
    timeout_ms: int = 10_000,
) -> LayerReport:
    """Verify one integer layer (``MatMulInteger`` / ``ConvInteger`` / ``QLinearMatMul`` / ``QLinearConv``).

    :param consts: name -> value of every constant the node reads (weights, scales, zero points, bias).
    :param act_range: integer range ``(lo, hi)`` the activation codes can actually take; defaults to the
        activation dtype's full range, which is sound but pessimistic.
    :param act_dtype: ``onnx.TensorProto`` dtype of the activation when it cannot be read off a zero point.
    :param x_shape: activation shape, used for convolution padding geometry.
    :param z3_max_k: re-prove the no-wrap claim with Z3 bit-vectors when a channel has at most this many taps.
    """
    types = {node.input[0]: act_dtype} if act_dtype is not None and node.input else {}
    shapes = (
        {node.input[0]: tuple(x_shape)} if x_shape is not None and node.input else {}
    )
    layer, why = _extract_layer(node, consts, types, shapes, act_range)
    rep = LayerReport(node.name or node.op_type, node.op_type)
    if layer is None:
        rep.findings.append(
            Finding("no-wrap", SOUNDNESS, SKIPPED, why or "unsupported")
        )
        return rep
    rep.name = layer.name
    rep.findings += _no_wrap_findings(layer, z3_max_k, timeout_ms)
    if layer.requant is not None:
        rep.findings += _requant_findings(layer, z3_max_k, timeout_ms)
    for n in layer.notes:
        rep.findings.append(Finding("note", INFORMATIONAL, PROVED, n))
    return rep


# --------------------------------------------------------------------------
# onnxsim's dynamic-quantization chain
#   Xq, Xs, Xzp = DynamicQuantizeLinear(X); Acc = MatMulInteger(Xq, Wq, Xzp)
#   Y = Cast<float>(Acc) * (Xs * Ws)
# --------------------------------------------------------------------------


def _dynamic_zero_points(float_range: Optional[Tuple[float, float]]) -> Tuple[int, int]:
    """Range of the zero point ``DynamicQuantizeLinear`` can choose, given a float hull of its input.

    The runtime extends the data range to include 0 and sets
    ``zp = clamp(rne(-min'/scale))`` with ``scale = (max'-min')/255``. A non-negative
    input therefore always gets ``zp = 0``, a non-positive one ``zp = 255``; a sign-mixed
    (or unknown) range can produce any zero point in ``[0, 255]``.
    """
    if float_range is None:
        return 0, 255
    lo, hi = float_range
    if lo >= 0:
        return 0, 0
    if hi <= 0:
        return 255, 255
    return 0, 255


def _dynamic_accumulator_range(
    w: np.ndarray, zp_lo: int, zp_hi: int
) -> Tuple[int, int, int, int]:
    """Exact ``(min, max, zp_at_min, zp_at_max)`` of ``sum w_k (xq_k - zp)`` with ``xq in [0,255]``, ``zp in [zp_lo, zp_hi]``.

    For a fixed zero point the extremes are at vertices (``P*(255-zp) + N*zp`` for the
    maximum); that is linear in ``zp``, so over the zero-point interval it is attained at an
    endpoint. The activation codes and the zero point are *coupled* (one zero point for the
    whole tensor), which is why this is tighter than ``255 * sum|w|``.
    """
    wi = np.asarray(w, dtype=np.int64).reshape(-1)
    p = int(wi[wi > 0].sum())
    n = int((-wi[wi < 0]).sum())
    best_hi = max(((p * (255 - z) + n * z), z) for z in (zp_lo, zp_hi))
    best_lo = min(((-p * z - n * (255 - z)), z) for z in (zp_lo, zp_hi))
    return best_lo[0], best_hi[0], best_lo[1], best_hi[1]


def _dynamic_counterexample(
    w: np.ndarray, zp: int, maximise: bool, name: str
) -> Dict[str, np.ndarray]:
    """A float input making ``DynamicQuantizeLinear`` pick zero point ``zp`` and the vertex codes."""
    wi = np.asarray(w, dtype=np.int64).reshape(-1)
    want_hi = (wi >= 0) if maximise else (wi < 0)  # taps wanting the top code (255)
    if zp == 0:
        x = np.where(want_hi, 1.0, 0.0)
    else:  # zp == 255: data in [-1, 0]; code 255 <-> 0.0, code 0 <-> -1.0
        x = np.where(want_hi, 0.0, -1.0)
    return {name: x.reshape(1, -1).astype(np.float32)}


def _find_dynamic_chain(model: onnx.ModelProto) -> List[Dict[str, Any]]:
    producer = {o: n for n in model.graph.node for o in n.output if o}
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for n in model.graph.node:
        for i in n.input:
            if i:
                consumers.setdefault(i, []).append(n)
    chains: List[Dict[str, Any]] = []
    for n in model.graph.node:
        if n.op_type != "MatMulInteger" or len(n.input) < 3:
            continue
        dql = producer.get(n.input[0])
        if (
            dql is None
            or dql.op_type != "DynamicQuantizeLinear"
            or n.input[2] != dql.output[2]
        ):
            continue
        ws_name = None
        for c in consumers.get(dql.output[1], []):
            if c.op_type == "Mul":
                other = [i for i in c.input if i != dql.output[1]]
                if other:
                    ws_name = other[0]
        chains.append({"dql": dql, "mm": n, "ws": ws_name})
    return chains


def weight_fidelity(
    w_ref: np.ndarray,
    wq: np.ndarray,
    scale: np.ndarray,
    zero_point: Any = 0,
    axis: Optional[int] = None,
) -> Tuple[str, str, Optional[Tuple[int, ...]]]:
    """Are the integer weights the round-to-nearest quantization of the reference weights?

    Checks ``|w_ref - scale * (wq - zp)| <= scale/2 + |w_ref| * 2**-23`` for every element
    (the slack is the fp32 rounding of the quantizer's own division, which scales with
    ``|w_ref|``). Returns
    ``(verdict, detail, index_of_first_violation)``. ``axis`` names the output-channel
    axis of ``wq`` when ``scale`` is per-channel.
    """
    wq = np.asarray(wq, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    if s.size > 1:
        if axis is None:
            axis = int(np.argmax(np.array(wq.shape) == s.size))
        shp = [1] * wq.ndim
        shp[axis] = -1
        s = s.reshape(shp)
    deq = s * (wq - np.asarray(zero_point, dtype=np.float64))
    if deq.shape != np.shape(w_ref):
        return SKIPPED, f"shape mismatch {deq.shape} vs {np.shape(w_ref)}", None
    wr = np.asarray(w_ref, dtype=np.float64)
    err = np.abs(wr - deq)
    # The quantizer computes q = rne(fl32(w / s)). fp32 division is correctly rounded, so
    # fl32(w/s) = (w/s)(1+d) with |d| <= 2**-24, and therefore |w - s*q| <= s/2 + |w| * 2**-24.
    # The extra term grows with |w/s| (up to 127 steps), so a *fixed* slack is wrong: on a
    # 4096 x 4096 layer a fixed 2**-22 flagged 9 perfectly good weights sitting within ~1e-6 of
    # a tie. The slack is doubled (2**-23) as margin.
    tol = np.broadcast_to(s, deq.shape) * 0.5 + np.abs(wr) * 2.0**-23
    bad = err > tol
    if not bad.any():
        return (
            PROVED,
            f"all {err.size} integer weights are within half a quantization step of the reference "
            f"(max error {float(err.max()):.3g}, step {float(np.max(s)):.3g})",
            None,
        )
    idx = tuple(int(i) for i in np.argwhere(bad)[0])
    return (
        REFUTED,
        f"{int(bad.sum())} of {err.size} weights are more than half a step from the reference "
        f"(first at {idx}: |ref - deq| = {float(err[idx]):.3g} > {float(np.broadcast_to(s, deq.shape)[idx] * 0.5):.3g}); "
        f"likely clipped or mis-scaled",
        idx,
    )


def _fidelity_finding(
    wq: np.ndarray,
    scale: np.ndarray,
    zp: Any,
    reference: onnx.ModelProto,
    axis: Optional[int],
) -> Finding:
    refs = [numpy_helper.to_array(t) for t in reference.graph.initializer]
    cands = [r for r in refs if r.shape == wq.shape and r.dtype.kind == "f"]
    if axis is None and wq.ndim == 2:
        cands += [
            r.T
            for r in refs
            if r.ndim == 2 and r.T.shape == wq.shape and r.dtype.kind == "f"
        ]
    if not cands:
        return Finding(
            "weight-fidelity",
            SOUNDNESS,
            SKIPPED,
            "no reference weight of matching shape",
        )
    best = None
    for r in cands:
        v, d, i = weight_fidelity(r, wq, scale, zp, axis)
        if v == PROVED:
            return Finding(
                "weight-fidelity", SOUNDNESS, PROVED, d, "exhaustive element check"
            )
        if best is None:
            best = (v, d, i, r)
    assert best is not None
    return Finding(
        "weight-fidelity",
        SOUNDNESS,
        best[0],
        best[1] + " (closest of the reference tensors with this shape)",
        "exhaustive element check",
    )


def probe_u8s8_saturation() -> Optional[bool]:
    """Run a tiny extreme-value ``MatMulInteger`` on onnxruntime and say whether it deviates from exact int32 arithmetic.

    Some AVX2 u8 x s8 kernels sum adjacent products in saturating int16, which is not
    modelled by wraparound semantics. ``True``: deviation seen; ``False``: exact on this
    machine; ``None``: onnxruntime unavailable. This is a property of one runtime on one
    CPU and proves nothing about others.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    k = 64
    a = np.full((1, k), 255, dtype=np.uint8)
    b = np.full((k, 1), 127, dtype=np.int8)
    g = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMulInteger", ["a", "b"], ["y"])],
        "probe",
        [onnx.helper.make_tensor_value_info("a", onnx.TensorProto.UINT8, [1, k])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.INT32, [1, 1])],
        [numpy_helper.from_array(b, "b")],
    )
    m = onnx.helper.make_model(g, opset_imports=[onnx.helper.make_opsetid("", 13)])
    m.ir_version = 8
    got = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"a": a})[0]
    return int(got[0, 0]) != k * 255 * 127


def _float_hull(
    model: onnx.ModelProto, input_ranges: Optional[Dict[str, Tuple[Any, Any]]]
) -> Dict[str, Tuple[float, float]]:
    """Float hulls of every tensor from interval analysis (empty if unavailable)."""
    try:
        from . import interval as _interval

        res = _interval.propagate(model, input_ranges)
        return {k: res.hull(k) for k in res.intervals}
    except Exception:
        return {}


def verify_model(
    integer_model: onnx.ModelProto,
    reference_model: Optional[onnx.ModelProto] = None,
    input_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None,
    z3_max_k: int = 24,
    timeout_ms: int = 10_000,
    probe_runtime: bool = False,
) -> ModelReport:
    """Verify every integer layer of ``integer_model`` (see the module docstring).

    :param reference_model: optional float / QDQ model the integer one was derived from. When
        given, each integer weight tensor is also checked to be the round-to-nearest
        quantization of a reference weight (``weight-fidelity``).
    :param input_ranges: ``{graph input: (lo, hi)}``. Float inputs feeding a ``QuantizeLinear`` or
        ``DynamicQuantizeLinear`` narrow the reachable activation codes (and, for the dynamic
        chain, the zero point); integer graph inputs give the code range directly.
    :param probe_runtime: also run :func:`probe_u8s8_saturation` and note the result.
    """
    consts = _consts(integer_model)
    elem = _elem_types(integer_model)
    shapes = _shapes(integer_model)
    hulls = (
        _float_hull(integer_model, input_ranges)
        if input_ranges
        or any(p.key.startswith("onnxsim.range.") for p in integer_model.metadata_props)
        else {}
    )
    producer = {o: n for n in integer_model.graph.node for o in n.output if o}
    chains = {id(c["mm"]): c for c in _find_dynamic_chain(integer_model)}
    layers: List[LayerReport] = []
    notes: List[str] = []
    if probe_runtime:
        sat = probe_u8s8_saturation()
        notes.append(
            "runtime probe unavailable (no onnxruntime)"
            if sat is None
            else (
                "onnxruntime's u8 x s8 MatMulInteger DEVIATES from exact int32 arithmetic on extreme values on this CPU "
                "(int16 pair saturation); wraparound-based verdicts do not describe it"
                if sat
                else "onnxruntime's u8 x s8 MatMulInteger matched exact int32 arithmetic on an extreme-value probe (this CPU only)"
            )
        )
    for node in integer_model.graph.node:
        if node.op_type not in (
            "MatMulInteger",
            "ConvInteger",
            "QLinearMatMul",
            "QLinearConv",
        ):
            continue
        chain = chains.get(id(node))
        if chain is not None:
            layers.append(
                _verify_dynamic(
                    node,
                    chain,
                    consts,
                    hulls,
                    reference_model,
                    z3_max_k,
                    timeout_ms,
                    integer_model,
                )
            )
            continue
        a = node.input[0]
        act_range = _static_act_range(a, producer, consts, hulls, input_ranges, elem)
        rep = verify_layer(
            node,
            consts,
            act_range=act_range,
            act_dtype=elem.get(a),
            x_shape=shapes.get(a),
            z3_max_k=z3_max_k,
            timeout_ms=timeout_ms,
        )
        if reference_model is not None:
            _add_static_fidelity(rep, node, consts, reference_model)
        layers.append(rep)
    if not layers:
        notes.append("no integer layers found")
    return ModelReport(layers, notes)


def _static_act_range(
    a: str,
    producer: Dict[str, onnx.NodeProto],
    consts: Dict[str, np.ndarray],
    hulls: Dict[str, Tuple[float, float]],
    input_ranges: Optional[Dict[str, Tuple[Any, Any]]],
    elem: Dict[str, int],
) -> Optional[Tuple[int, int]]:
    """Reachable integer code range of activation ``a``, or ``None`` for the dtype's full range."""
    qlo, qhi = _INT_RANGES.get(elem.get(a, onnx.TensorProto.UINT8), (0, 255))
    p = producer.get(a)
    if p is not None and p.op_type == "QuantizeLinear" and p.input[0] in hulls:
        s = _scalar(consts, p.input[1])
        zp = _scalar(consts, p.input[2]) if len(p.input) > 2 else 0
        if s is not None and zp is not None:
            lo, hi = hulls[p.input[0]]
            if np.isfinite(lo) and np.isfinite(hi):

                def code(v: float) -> int:
                    """The code QuantizeLinear gives a float: ``sat(rne(v / s) + zp)``."""
                    return max(
                        qlo, min(qhi, rne_fraction(_frac(v) / _frac(s)) + int(zp))
                    )

                return code(lo), code(hi)
    if p is None and input_ranges and a in input_ranges:
        lo, hi = input_ranges[a]
        return int(np.floor(np.min(lo))), int(np.ceil(np.max(hi)))
    return None


def _add_static_fidelity(
    rep: LayerReport,
    node: onnx.NodeProto,
    consts: Dict[str, np.ndarray],
    reference: onnx.ModelProto,
) -> None:
    ins = list(node.input) + [""] * 9
    if node.op_type == "QLinearMatMul":
        w, wsc, wzp = ins[3], ins[4], ins[5]
        axis = 1
    elif node.op_type == "QLinearConv":
        w, wsc, wzp = ins[3], ins[4], ins[5]
        axis = 0
    else:
        return  # MatMulInteger / ConvInteger carry no scale to compare against
    if w not in consts or wsc not in consts:
        return
    zp = consts[wzp] if wzp in consts else 0
    rep.findings.append(
        _fidelity_finding(
            consts[w],
            consts[wsc],
            zp,
            reference,
            axis if consts[wsc].size > 1 else None,
        )
    )


def _verify_dynamic(
    node: onnx.NodeProto,
    chain: Dict[str, Any],
    consts: Dict[str, np.ndarray],
    hulls: Dict[str, Tuple[float, float]],
    reference: Optional[onnx.ModelProto],
    z3_max_k: int,
    timeout_ms: int,
    model: onnx.ModelProto,
) -> LayerReport:
    name = node.name or f"MatMulInteger_{node.output[0]}"
    rep = LayerReport(name, "DynamicQuantizeLinear+MatMulInteger")
    wq_name = node.input[1]
    if wq_name not in consts or consts[wq_name].ndim != 2:
        rep.findings.append(
            Finding(
                "no-wrap", SOUNDNESS, SKIPPED, "weight is not a constant 2-D tensor"
            )
        )
        return rep
    wq = consts[wq_name].astype(np.int64)
    wzp = (
        int(_scalar(consts, node.input[3]) or 0)
        if len(node.input) > 3 and node.input[3]
        else 0
    )
    w = (wq - wzp).T  # [N, K]
    x_name = chain["dql"].input[0]
    fr = hulls.get(x_name)
    zp_lo, zp_hi = _dynamic_zero_points(
        fr if fr is not None and np.isfinite(fr[0]) and np.isfinite(fr[1]) else None
    )
    mins, maxs, zmin, zmax = [], [], [], []
    for c in range(w.shape[0]):
        lo, hi, zl, zh = _dynamic_accumulator_range(w[c], zp_lo, zp_hi)
        mins.append(lo)
        maxs.append(hi)
        zmin.append(zl)
        zmax.append(zh)
    worst_hi, worst_lo = max(maxs), min(mins)
    method = "closed form with the zero point coupled to the codes (exact)"
    if worst_hi <= INT32_MAX and worst_lo >= INT32_MIN:
        rep.findings.append(
            Finding(
                "no-wrap",
                SOUNDNESS,
                PROVED,
                f"every accumulator lies in [{worst_lo}, {worst_hi}] for zero points in [{zp_lo}, {zp_hi}] "
                f"(the naive bound 255*sum|w| would be {255 * int(np.abs(w).sum(axis=1).max())})",
                method,
            )
        )
    else:
        over_hi = worst_hi > INT32_MAX
        c = int(np.argmax(maxs)) if over_hi else int(np.argmin(mins))
        zp = zmax[c] if over_hi else zmin[c]
        val = worst_hi if over_hi else worst_lo
        cex = _dynamic_counterexample(w[c], zp, over_hi, x_name)
        wrapped = ((val - INT32_MIN) % 2**32) + INT32_MIN
        rep.findings.append(
            Finding(
                "no-wrap",
                SOUNDNESS,
                REFUTED,
                f"channel {c} reaches {val} with zero point {zp}; the kernel wraps it to {wrapped}",
                method,
                counterexample=cex,
            )
        )
    worst_abs = max(abs(worst_hi), abs(worst_lo))
    rep.findings.append(
        Finding(
            "float-cast-exact",
            INFORMATIONAL,
            PROVED if worst_abs <= FP32_EXACT_INT else REFUTED,
            f"max |accumulator| = {worst_abs}; Cast<float> is exact up to 2**24",
            "closed form",
        )
    )
    eps = dynamic_rel_error_bound(worst_abs)
    rep.findings.append(
        Finding(
            "dynamic-rel-error",
            SOUNDNESS,
            PROVED,
            f"Cast<float>(acc) * (Xs * Ws) is within relative {float(eps):.3g} of the exact acc*Xs*Ws "
            f"for every operand value (three fp32 roundings; the cast is exact when |acc| <= 2**24)",
            "multilinear corner enumeration (exact)",
        )
    )
    if reference is not None and chain["ws"] in consts:
        ws = consts[chain["ws"]]
        rep.findings.append(
            _fidelity_finding(
                consts[wq_name], ws, wzp, reference, 1 if ws.size > 1 else None
            )
        )
    return rep


# --------------------------------------------------------------------------
# Replay: run a counterexample on onnxruntime
# --------------------------------------------------------------------------


def replay_no_wrap(
    model: onnx.ModelProto, node_name: str, finding: Finding
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a ``no-wrap`` counterexample on onnxruntime: ``(observed int32, exact int64)`` accumulators.

    For a refuted layer these differ -- the observed value is the wrapped one. Builds an
    integer twin of the layer (``MatMulInteger`` / ``ConvInteger`` with the same weights and
    zero points; for the dynamic chain, ``DynamicQuantizeLinear`` + ``MatMulInteger``) so the
    accumulator itself is observable even for ``QLinear*`` layers, whose output is requantised.
    """
    import onnxruntime as ort

    assert finding.counterexample is not None
    consts = _consts(model)
    node = next(
        n
        for n in model.graph.node
        if (n.name or f"{n.op_type}_{n.output[0]}") == node_name or n.name == node_name
    )
    ((key, x),) = finding.counterexample.items()
    if node.op_type in ("MatMulInteger", "ConvInteger", "QLinearMatMul", "QLinearConv"):
        ins = list(node.input) + [""] * 9
        if node.op_type.startswith("QLinear"):
            w, azp, wzp = ins[3], ins[2], ins[5]
        else:
            w, azp, wzp = ins[1], ins[2], ins[3]
        twin_op = "MatMulInteger" if "MatMul" in node.op_type else "ConvInteger"
        inits = [numpy_helper.from_array(consts[w], "w")]
        tin = ["a", "w"]
        a_dtype = onnx.helper.np_dtype_to_tensor_dtype(x.dtype)
        if azp and azp in consts:
            inits.append(numpy_helper.from_array(consts[azp], "azp"))
            tin.append("azp")
        if wzp and wzp in consts:
            if len(tin) == 2:
                inits.append(
                    numpy_helper.from_array(np.zeros((), dtype=x.dtype), "azp")
                )
                tin.append("azp")
            inits.append(numpy_helper.from_array(consts[wzp], "wzp"))
            tin.append("wzp")
        twin = onnx.helper.make_node(
            twin_op,
            tin,
            ["acc"],
            **(
                {}
                if twin_op == "MatMulInteger"
                else {
                    k: v
                    for k, v in _attrs(node).items()
                    if k in ("pads", "strides", "dilations", "group", "kernel_shape")
                }
            ),
        )
        g = onnx.helper.make_graph(
            [twin], "twin", [onnx.helper.make_tensor_value_info("a", a_dtype, list(x.shape))],
            [onnx.helper.make_tensor_value_info("acc", onnx.TensorProto.INT32, None)], inits,
        )  # fmt: skip
        m = onnx.helper.make_model(g, opset_imports=[onnx.helper.make_opsetid("", 13)])
        m.ir_version = 8
        observed = ort.InferenceSession(
            m.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"a": x})[0]
        za = int(consts[azp].reshape(-1)[0]) if azp and azp in consts else 0
        wz = consts[wzp].astype(np.int64) if wzp and wzp in consts else 0
        wi = consts[w].astype(np.int64) - wz
        if twin_op == "MatMulInteger":
            exact = (x.astype(np.int64) - za) @ wi
        else:
            exact = _exact_conv(x.astype(np.int64) - za, wi, _attrs(node))
        return observed, exact
    raise ValueError(f"cannot replay {node.op_type}")  # pragma: no cover


def replay_dynamic_no_wrap(
    model: onnx.ModelProto, node_name: str, finding: Finding
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay a dynamic-chain counterexample: ``(observed int32, exact int64)`` of ``MatMulInteger``'s output."""
    import onnxruntime as ort

    assert finding.counterexample is not None
    ((key, x),) = finding.counterexample.items()
    m = onnx.ModelProto()
    m.CopyFrom(model)
    mm = next(
        n
        for n in m.graph.node
        if n.op_type == "MatMulInteger"
        and (n.name == node_name or f"MatMulInteger_{n.output[0]}" == node_name)
    )
    dql = next(
        n
        for n in m.graph.node
        if n.op_type == "DynamicQuantizeLinear" and n.output[0] == mm.input[0]
    )
    del m.graph.output[:]
    for name, t in (
        (mm.output[0], onnx.TensorProto.INT32),
        (dql.output[0], onnx.TensorProto.UINT8),
        (dql.output[2], onnx.TensorProto.UINT8),
    ):
        m.graph.output.append(onnx.helper.make_tensor_value_info(name, t, None))
    outs = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {key: x})
    consts = _consts(model)
    wq = consts[mm.input[1]].astype(np.int64)
    exact = (outs[1].astype(np.int64) - outs[2].astype(np.int64)) @ wq
    return outs[0], exact


def _exact_conv(d: np.ndarray, w: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
    """Exact int64 2-D convolution of ``d`` (already offset by the zero point) with ``w``; padding adds 0."""
    strides = list(attrs.get("strides", [1, 1]))
    dil = list(attrs.get("dilations", [1, 1]))
    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    group = int(attrs.get("group", 1))
    n, c, h, wd = d.shape
    cout, cg, kh, kw = w.shape
    dp = np.pad(d, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    oh = (dp.shape[2] - dil[0] * (kh - 1) - 1) // strides[0] + 1
    ow = (dp.shape[3] - dil[1] * (kw - 1) - 1) // strides[1] + 1
    out = np.zeros((n, cout, oh, ow), dtype=np.int64)
    per = cout // group
    for co in range(cout):
        g = co // per
        for i in range(kh):
            for j in range(kw):
                patch = dp[
                    :,
                    g * cg : (g + 1) * cg,
                    i * dil[0] : i * dil[0] + oh * strides[0] : strides[0],
                    j * dil[1] : j * dil[1] + ow * strides[1] : strides[1],
                ]
                out[:, co] += (patch * w[co, :, i, j].reshape(1, -1, 1, 1)).sum(axis=1)
    return out
