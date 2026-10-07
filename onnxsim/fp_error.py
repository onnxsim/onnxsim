"""Certified floating-point roundoff bounds for ONNX graphs.

``onnxsim.interval``, ``onnxsim.crown``, ``onnxsim.zonotope`` and ``onnxsim.certify``
all reason about the *real-number* semantics of a model, but a model runs in float32
(or float16/bfloat16). This module bounds the gap: for every input in a box,

    |execution(x) - real_semantics(x)|  <=  bound            (per output element)

``roundoff_bound`` computes ``bound`` by first-order error propagation with the
standard model ``fl(a op b) = (a op b)(1 + d), |d| <= u`` (``u`` = 2**-24 for fp32), carried
next to the real-value intervals of :func:`onnxsim.interval.propagate`.
``tolerance_for(orig, simplified, ...)`` combines it with the zonotope difference bound
into a certified ``atol`` for comparing the two models' *executions*, replacing the
arbitrary constant ``simplify(check_atol=...)`` / ``certify(atol=...)`` otherwise use.

What is proved, and what is only assumed -- read this before trusting a number:

PROVED (pure arithmetic, no hypothesis about any particular runtime)
  * Add/Sub/Mul/Div/Neg/Pow(2)/Relu/Clip/Max/Min and the data-movement ops follow from the
    model above and, for the exact ops, from 1-Lipschitzness.
  * A dot product / convolution of ``n`` terms is bounded by ``gamma_n * sum|w||x|`` with
    ``gamma_n = n*u / (1 - n*u)`` (Higham, "Accuracy and Stability of Numerical Algorithms",
    Thm 3.5 / eq. (3.12)). That bound holds for **any summation order** -- sequential,
    pairwise, blocked, SIMD lanes, tree reductions -- and with fused multiply-add, which is
    what makes it valid for kernels that order their reductions differently from each other.
  * Softmax with the usual max-subtraction: input sensitivity ``expm1(2*||e||_inf)`` plus a
    relative error built from the argument rounding, the exp error, the n-term sum and the
    final division (derivation in ``_softmax``).

ASSUMED (a hypothesis about the runtime, stated and parameterised -- not a theorem)
  * **Library functions** (Sigmoid, Tanh, Exp, Erf, Sqrt, Log, Softplus) are not correctly
    rounded in fast kernels. Their error is the :class:`LibmModel` ``rel*u*|f| + abs*u``.
    The defaults are about 3x the worst error measured for onnxruntime 1.x CPU fp32 on
    2 million points per range (max observed, in units of ``u``: Exp 1.3 rel, Erf 1.4 rel,
    Sqrt 1.0 rel, Tanh 5.3 rel, Softplus 5.5 rel, Log 4 rel / 1.4 abs near 1, Sigmoid 3.0
    abs / 4 rel away from the tails). They are *measurements of one runtime*; a different
    kernel, or fp16/bf16 (where they are reused in units of that precision), can be worse.
    Pass your own ``lib=`` to certify against a runtime you have characterised.
  * **Graph execution as written.** Runtime graph rewrites change roundoff. The BatchNorm bound
    is made robust to the most common one (folding BatchNorm into the preceding Conv/Gemm
    weights) by scaling its constant-rounding term with ``sum|w||x|`` of the producer instead
    of ``|conv output|``; other fusions (fused GELU, fused attention, LayerNorm kernels) are not
    modelled. Winograd/FFT convolution has larger error than any direct summation, so it is
    **not** covered; a kernel that accumulates in a *wider* format only makes the bound
    conservative.
  * **No overflow or NaN.** If a computed magnitude could exceed the format's largest finite
    number the error is reported ``inf``. Underflow is covered by an absolute term ``eta``.
  * **Inputs are the numbers actually fed.** The input box describes values already
    representable in the target precision (error 0 at the inputs). Constants stored in a wider
    format than the analysed precision are rounded: a relative error ``u`` on each weight.
  * The real-value ranges come from :func:`onnxsim.interval.propagate`, which encloses real
    semantics in float64 with a ``1e-14`` relative widening; its own float64 rounding
    (~1e-16 relative) is far below every bound here and is not added.

Anything without a rule yields an infinite error for that tensor and a note, never a
silently small bound. Errors are elementwise arrays shaped like the tensor.
"""

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

from . import interval as _interval

_INF = float("inf")


@dataclasses.dataclass(frozen=True)
class Precision:
    """A floating-point format: unit roundoff ``u``, largest finite value, and ``eta``.

    ``eta`` is the absolute error bound that underflow can add to a product (half the
    smallest subnormal), the term the textbook model ``(1+d)`` leaves out.
    """

    name: str
    u: float
    fmax: float
    eta: float
    mantissa_bits: int  # explicit significand bits + 1


PRECISIONS: Dict[str, Precision] = {
    "fp32": Precision("fp32", 2.0**-24, 3.4028234663852886e38, 2.0**-150, 24),
    "fp16": Precision("fp16", 2.0**-11, 65504.0, 2.0**-25, 11),
    "bf16": Precision("bf16", 2.0**-8, 3.3895313892515355e38, 2.0**-134, 8),
}

_STORED_BITS = {"float16": 11, "float32": 24, "float64": 53}

#: ``ranges="auto"`` uses zonotope ranges only up to this many input elements (the generator
#: matrices are dense), else it stays with interval ranges and says so in the notes.
_ZONOTOPE_MAX_INPUTS = 4096

#: The tight pass gives every rounding event its own noise symbol; the generator matrices are
#: dense, so it only runs up to this many (input + noise) elements.
_TIGHT_MAX_SYMBOLS = 8192

#: Ops whose float result is the real result of the computed inputs (no rounding of their own):
#: comparisons/selections, sign flips, and pure data movement. They add no noise symbol.
_EXACT_OPS = frozenset(
    {
        "Neg", "Identity", "Relu", "Abs", "Clip", "Max", "Min", "Constant", "ConstantOfShape",
        "Reshape", "Flatten", "Transpose", "Squeeze", "Unsqueeze", "Slice", "Gather", "Expand",
        "Tile", "DepthToSpace", "SpaceToDepth", "MaxPool", "GlobalMaxPool", "ReduceMax",
        "ReduceMin", "Split", "Dropout", "Pad", "Concat", "Cast",
    }
)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class LibmModel:
    """Error of the runtime's elementary functions: ``rel*u*|f(z)| + abs*u`` per call.

    Defaults are ~3x the worst error observed for onnxruntime 1.x CPU fp32; see the
    module docstring. This is a measurement of one runtime, not a guarantee.
    """

    rel: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "Sigmoid": 12.0,
            "Tanh": 16.0,
            "Exp": 4.0,
            "Erf": 4.0,
            "Sqrt": 2.0,
            "Log": 12.0,
            "Softplus": 16.0,
        }
    )
    abs_: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "Sigmoid": 8.0,
            "Tanh": 0.0,
            "Exp": 0.0,
            "Erf": 0.0,
            "Sqrt": 0.0,
            "Log": 4.0,
            "Softplus": 4.0,
        }
    )


DEFAULT_LIBM = LibmModel()


@dataclasses.dataclass
class RoundoffResult:
    """Certified roundoff error of a graph; see :func:`roundoff_bound`."""

    precision: str
    errors: Dict[str, np.ndarray]  # every analysed tensor: elementwise abs error bound
    outputs: Dict[str, np.ndarray]  # graph outputs only
    notes: List[str]
    unsupported: List[str]  # op types that fell back to an infinite error

    def bound(self, name: str) -> float:
        """Largest elementwise error bound of tensor ``name`` (``inf`` if unbounded)."""
        e = self.errors[name]
        return float(np.max(e)) if np.size(e) else 0.0

    @property
    def worst(self) -> float:
        """Largest bound over all graph outputs (``inf`` if any is unbounded or unknown)."""
        if not self.outputs:
            return _INF
        return float(
            max(np.max(v) if np.size(v) else 0.0 for v in self.outputs.values())
        )

    @property
    def bounded(self) -> bool:
        return bool(np.isfinite(self.worst))


def _gamma(n: float, u: float) -> float:
    """Higham's ``gamma_n = n*u / (1 - n*u)``; ``inf`` once ``n*u >= 1`` (no bound)."""
    nu = n * u
    return nu / (1.0 - nu) if nu < 1.0 else _INF


def _is_float(a: np.ndarray) -> bool:
    return np.asarray(a).dtype.kind == "f"


class _NoRule(Exception):
    """No sound roundoff rule for this node: its error is reported infinite."""


class _Analyzer:
    def __init__(
        self,
        model: onnx.ModelProto,
        input_ranges: Optional[Dict[str, Tuple]],
        prec: Precision,
        lib: LibmModel,
        bn_ulps: float,
        weights_exact: Optional[bool],
        input_exact: bool,
        range_source: str = "interval",
    ) -> None:
        self.model, self.prec, self.lib, self.bn_ulps = model, prec, lib, bn_ulps
        self.input_ranges = input_ranges
        self.input_exact = input_exact
        self.weights_exact = weights_exact
        self.notes: List[str] = []
        self.real = _interval.propagate(model, input_ranges)
        self.iv = self.real.intervals
        if range_source not in ("interval", "zonotope", "auto"):
            raise ValueError(
                f"unknown ranges {range_source!r}; expected interval, zonotope or auto"
            )
        if range_source != "interval":
            self._tighten_with_zonotopes(
                input_ranges, strict=range_source == "zonotope"
            )
        self.runner = _interval._Runner(model)
        self.opset = max(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
            default=0,
        )
        self.err: Dict[str, np.ndarray] = {}
        self.absmag: Dict[
            str, np.ndarray
        ] = {}  # sum|w||x| bound of linear producers (BN fusion)
        self.local: Dict[
            str, np.ndarray
        ] = {}  # per tensor: radius of the rounding noise its op adds
        self.tight_ok = (
            True  # False once a node has rounding the noise model does not record
        )
        self.overflowed = (
            False  # a value may exceed the format's range: execution may yield inf
        )
        self.unsupported: List[str] = []

    def _tighten_with_zonotopes(self, input_ranges, strict: bool) -> None:
        """Intersect the interval ranges with zonotope ranges (both enclose the real value).

        Plain interval arithmetic inflates a magnitude by about ``sum|w|`` per layer (the
        dependency problem); every roundoff term scales with these magnitudes, so tighter
        ranges directly tighten the bound. The intersection of two sound enclosures is sound.
        """
        from . import zonotope

        n_in = sum(
            int(np.prod(self.iv[vi.name][0].shape))
            for vi in self.model.graph.input
            if vi.name in self.iv
        )
        if not strict and n_in > _ZONOTOPE_MAX_INPUTS:
            self.notes.append(
                f"zonotope ranges skipped: {n_in} input elements > {_ZONOTOPE_MAX_INPUTS}"
            )
            return
        try:
            z = zonotope.propagate(self.model, input_ranges)
        except Exception as e:  # unbounded input, memory, unsupported shape...
            if strict:
                raise
            self.notes.append(
                f"zonotope ranges unavailable ({type(e).__name__}: {e}); interval ranges used"
            )
            return
        for name, (il, ih) in list(self.iv.items()):
            if name not in z.tensors or not _is_float(il):
                continue
            zl, zh = z.bounds(name)
            if zl.shape != il.shape:
                continue
            lo, hi = np.maximum(il, zl), np.minimum(ih, zh)
            ok = lo <= hi  # both enclose the truth; a crossing can only be float noise
            self.iv[name] = (np.where(ok, lo, il), np.where(ok, hi, ih))

    def tighten_errors(self) -> None:
        """Replace the ``|W|``-style forward bounds by a zonotope difference (correlation-aware).

        The forward recursion carries a layer's error through ``|W|``, i.e. it assumes the
        error of every unit can align against the sign of every weight, and it compounds that
        per layer (about ``sum|w|`` per layer: ~1e4x pessimistic at depth 3, ~1e8x at depth 6
        in measurements). The true propagation is ``W @ e``, where signs cancel.

        Here every rounding event becomes a bounded noise symbol. Its radius is the local
        radius the first pass already certified (the rounding a node adds when applied to its
        *computed* inputs, evaluated against sound magnitude bounds), so the executed model is
        contained in "the real model with those bounded perturbations added after each node".
        :func:`onnxsim.zonotope.bound_difference` of that perturbed model against the clean one,
        on shared noise symbols, then gives ``|perturbed - clean|`` with the weights acting as
        ``W`` not ``|W|``. Both bounds are sound; the elementwise minimum is returned.
        """
        from . import zonotope

        g = self.model.graph
        if self.overflowed:
            # The noise model assumes finite arithmetic. If a value may overflow, execution can
            # return inf and no finite difference bound applies; keep the first pass's infinity.
            self.notes.append("tight pass skipped: a value may overflow the format")
            return
        if not self.tight_ok:
            self.notes.append(
                "tight pass skipped: a node's rounding is not expressible as noise (e.g. Gemm alpha/beta)"
            )
            return
        if not self.input_exact or any(
            np.any(self.err[t.name]) for t in g.initializer if t.name in self.err
        ):
            self.notes.append(
                "tight pass skipped: inputs/constants are not exactly representable in this precision"
            )
            return
        names = [t for t, r in self.local.items() if np.any(r)]
        if any(not np.all(np.isfinite(self.local[t])) for t in names):
            self.notes.append("tight pass skipped: a rounding radius is unbounded")
            return
        inits = {t.name for t in g.initializer}
        n_in = sum(
            self.iv[vi.name][0].size
            for vi in g.input
            if vi.name not in inits and vi.name in self.iv
        )
        n_noise = sum(self.local[t].size for t in names)
        if n_in + n_noise > _TIGHT_MAX_SYMBOLS:
            self.notes.append(
                f"tight pass skipped: {n_in + n_noise} noise symbols > {_TIGHT_MAX_SYMBOLS}"
            )
            return
        if not names:
            return
        targets = set(names)
        noise_in = {t: f"__fp_noise_{k}" for k, t in enumerate(names)}

        def extended(noisy: bool) -> onnx.ModelProto:
            m = onnx.ModelProto()
            m.CopyFrom(self.model)
            del m.graph.value_info[:]
            for t in names:  # the clean model declares (and ignores) the noise inputs
                shape = self.iv[t][0].shape
                m.graph.input.append(
                    onnx.helper.make_tensor_value_info(
                        noise_in[t], onnx.TensorProto.FLOAT, list(shape)
                    )
                )
            if not noisy:
                return m
            nodes = []
            for node in m.graph.node:
                adds = []
                for idx, o in enumerate(node.output):
                    if o in targets:
                        node.output[idx] = o + "__pre"
                        adds.append(
                            onnx.helper.make_node(
                                "Add",
                                [o + "__pre", noise_in[o]],
                                [o],
                                name=o + "__noise",
                            )
                        )
                nodes.append(node)
                nodes.extend(adds)
            del m.graph.node[:]
            m.graph.node.extend(nodes)
            return m

        try:
            ranges = dict(self.input_ranges or {})
            ranges.update({noise_in[t]: (-self.local[t], self.local[t]) for t in names})
            diff = zonotope.bound_difference(extended(False), extended(True), ranges)
        except (
            Exception
        ) as e:  # never let the optional tightening break the sound bound
            self.notes.append(
                f"tight pass failed ({type(e).__name__}: {e}); forward bound kept"
            )
            return
        improved = 0
        for o in self.model.graph.output:
            d = diff.max_abs.get(o.name)
            if d is None or o.name not in self.err or d.shape != self.err[o.name].shape:
                continue
            new = np.minimum(self.err[o.name], d)
            improved += int(np.any(new < self.err[o.name]))
            self.err[o.name] = new
        self.notes.extend(f"tight pass (zonotope): {n}" for n in diff.notes)
        self.notes.append(
            f"tight pass: tightened {improved} of {len(self.model.graph.output)} outputs"
        )

    # ---- helpers -----------------------------------------------------------
    def note(self, node: onnx.NodeProto, msg: str) -> None:
        self.notes.append(f"{node.op_type} {node.name or node.output[0]}: {msg}")

    def mag(self, name: str) -> np.ndarray:
        lo, hi = self.iv[name]
        return np.maximum(np.abs(lo), np.abs(hi)).astype(np.float64)

    def u(self) -> float:
        return self.prec.u

    def _const_error(self, arr: np.ndarray) -> np.ndarray:
        """Representation error of a stored constant when analysed at ``self.prec``."""
        if not _is_float(arr):
            return np.zeros(arr.shape, dtype=np.float64)
        exact = self.weights_exact
        if exact is None:
            exact = _STORED_BITS.get(arr.dtype.name, 99) <= self.prec.mantissa_bits
        if exact:
            return np.zeros(arr.shape, dtype=np.float64)
        return self.prec.u * np.abs(arr.astype(np.float64)) + self.prec.eta

    def _finish(self, node: onnx.NodeProto, name: str, e: np.ndarray) -> np.ndarray:
        """Broadcast to the tensor's shape, kill NaNs (sound: -> inf), apply the overflow guard."""
        lo, _ = self.iv[name]
        e = np.broadcast_to(np.asarray(e, dtype=np.float64), lo.shape).copy()
        e[np.isnan(e)] = _INF
        if _is_float(lo):
            # Partial sums of a dot product are bounded by sum|a||b|, not by the final value.
            peak = np.maximum(self.mag(name), self.absmag.get(name, 0.0))
            over = (peak + e) > self.prec.fmax
            if np.any(over & np.isfinite(e)):
                self.note(
                    node,
                    "value may exceed the largest finite number: error reported inf",
                )
                self.overflowed = True
            e[over] = _INF
        return e

    # ---- driver ------------------------------------------------------------
    def run(self) -> None:
        g = self.model.graph
        for t in g.initializer:
            if t.name in self.iv:
                self.err[t.name] = self._const_error(numpy_helper.to_array(t))
        inits = {t.name for t in g.initializer}
        for vi in g.input:
            if vi.name in inits or vi.name not in self.iv:
                continue
            lo, _ = self.iv[vi.name]
            self.err[vi.name] = (
                np.zeros(lo.shape)
                if self.input_exact
                else self.prec.u * self.mag(vi.name) + self.prec.eta
            )
        for node in g.node:
            outs = [o for o in node.output if o]
            if any(o not in self.iv for o in outs):
                continue  # no shape/interval for an output: nothing to say
            ins = [x for x in node.input if x]
            if any(x not in self.iv or x not in self.err for x in ins):
                continue
            if all(not _is_float(self.iv[o][0]) for o in outs):
                for (
                    o
                ) in outs:  # integer/bool tensors (shapes, indices) carry no roundoff
                    self.err[o] = np.zeros(self.iv[o][0].shape)
                continue
            handler = getattr(self, "op_" + node.op_type, None)
            try:
                if node.domain not in ("", "ai.onnx") or handler is None:
                    raise _NoRule(f"no roundoff rule for {node.op_type}")
                with np.errstate(
                    invalid="ignore", over="ignore"
                ):  # inf * 0 -> nan is mapped to inf
                    results = handler(node)
            except _NoRule as e:
                if node.op_type not in self.unsupported:
                    self.unsupported.append(node.op_type)
                self.note(node, f"{e}; error reported inf")
                results = [np.full(self.iv[o][0].shape, _INF) for o in outs]
            except Exception as e:  # an evaluator limitation: stay sound
                if node.op_type not in self.unsupported:
                    self.unsupported.append(node.op_type)
                self.note(
                    node,
                    f"internal limitation ({type(e).__name__}); error reported inf",
                )
                results = [np.full(self.iv[o][0].shape, _INF) for o in outs]
            for o, err_o in zip(outs, results):
                if not _is_float(self.iv[o][0]):
                    self.err[o] = np.zeros(
                        self.iv[o][0].shape
                    )  # integer tensors are exact
                else:
                    self.err[o] = self._finish(node, o, err_o)
                    if node.op_type not in _EXACT_OPS and o not in self.local:
                        self.tight_ok = (
                            False  # rounding this analysis cannot express as noise
                        )

    # ---- constants and casts ----------------------------------------------
    def op_Constant(self, node):
        return [self._const_error(np.asarray(self.iv[node.output[0]][0]))]

    op_ConstantOfShape = op_Constant

    def op_Cast(self, node):
        x = node.input[0]
        to = int(_interval._attrs(node)["to"])
        if not _is_float(self.iv[x][0]):
            # integer -> float: exact below 2**mantissa_bits, else one rounding of relative u
            loc = self.u() * self.mag(x) + self.prec.eta
            self._local(node, loc)
            return [loc]
        if (
            to in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE)
            and self.prec.name == "fp32"
        ):
            return [self.err[x]]  # float32 -> float32/float64: no further rounding
        raise _NoRule("Cast to a narrower float is not modelled")

    # ---- elementwise -------------------------------------------------------
    def _two(self, node):
        a, b = node.input[0], node.input[1]
        return (self.mag(a), self.err[a]), (self.mag(b), self.err[b]), node.output[0]

    def _local(self, node, radius, index: int = 0):
        """Record the radius of the rounding noise this node adds on top of its (real) function."""
        self.local[node.output[index]] = np.asarray(radius, dtype=np.float64)

    def op_Add(self, node):
        (_, ea), (_, eb), o = self._two(node)
        e = ea + eb
        loc = self.u() * (self.mag(o) + e)  # rounding of the result of an add/sub
        self._local(node, loc)
        return [e + loc]

    def op_Sub(self, node):
        return self.op_Add(node)

    def op_Mul(self, node):
        (ma, ea), (mb, eb), _ = self._two(node)
        u = self.u()
        # |a^b^ - ab| <= ea|b^| + |a|eb <= ea(mb+eb) + ma*eb ; rounding u|a^b^| ; underflow eta
        loc = u * (ma + ea) * (mb + eb) + self.prec.eta
        self._local(node, loc)
        return [ea * (mb + eb) + ma * eb + loc]

    def op_Div(self, node):
        (ma, ea), (mb, eb), _ = self._two(node)
        lo, hi = self.iv[node.input[1]]
        dmin = np.where(lo > 0, lo, np.where(hi < 0, -hi, 0.0)).astype(np.float64)
        gap = dmin - eb
        if np.any(gap <= 0):
            self.note(node, "denominator range (widened by its error) may contain 0")
        with np.errstate(all="ignore"):
            safe = np.where(gap > 0, gap, np.nan)
            # |a^/b^ - a/b| <= ea/(dmin-eb) + ma*eb/(dmin*(dmin-eb)) ; rounding u*|a^/b^|
            loc = self.u() * (ma + ea) / safe + self.prec.eta
            e = ea / safe + ma * eb / (dmin * safe) + loc
        self._local(node, np.where(np.isnan(loc), _INF, loc))
        return [e]

    def op_Reciprocal(self, node):
        x = node.input[0]
        lo, hi = self.iv[x]
        dmin = np.where(lo > 0, lo, np.where(hi < 0, -hi, 0.0)).astype(np.float64)
        gap = dmin - self.err[x]
        with np.errstate(all="ignore"):
            safe = np.where(gap > 0, gap, np.nan)
            loc = self.u() / safe + self.prec.eta
            e = self.err[x] / (dmin * safe) + loc
        self._local(node, np.where(np.isnan(loc), _INF, loc))
        return [e]

    def op_Neg(self, node):
        return [self.err[node.input[0]]]

    op_Identity = op_Neg
    op_Relu = op_Neg
    op_Abs = op_Neg
    op_Clip = op_Neg  # min/max are exact comparisons; 1-Lipschitz

    def op_Max(self, node):
        e = self.err[node.input[0]]
        for x in node.input[1:]:
            e = np.maximum(e, self.err[x])  # |max(a,b)-max(a^,b^)| <= max(ea, eb)
        return [e]

    op_Min = op_Max

    def op_LeakyRelu(self, node):
        alpha = float(_interval._attrs(node).get("alpha", 0.01))
        x = node.input[0]
        e = self.err[x]
        loc = self.u() * abs(alpha) * (self.mag(x) + e)
        self._local(node, loc)
        return [max(1.0, abs(alpha)) * e + loc]

    def op_Pow(self, node):
        x = node.input[0]
        ex = self.iv.get(node.input[1])
        if (
            ex is None
            or not np.array_equal(ex[0], ex[1])
            or np.asarray(ex[0]).size != 1
            or float(np.asarray(ex[0]).reshape(-1)[0]) != 2.0
        ):
            raise _NoRule("only Pow with the constant exponent 2")  # noqa: E501
        m, e = self.mag(x), self.err[x]
        loc = self.u() * (m + e) ** 2 + self.prec.eta
        self._local(node, loc)
        return [e * (2 * m + e) + loc]

    # ---- data movement: apply the op to the error array -------------------
    def _move(self, node):
        data = node.input[0]
        extra = [self.iv[x][0] for x in node.input[1:] if x]
        return self.runner.run(node, [self.err[data]] + extra)

    op_Reshape = op_Flatten = op_Transpose = op_Squeeze = op_Unsqueeze = _move
    op_Slice = op_Gather = op_Expand = op_Tile = op_DepthToSpace = op_SpaceToDepth = (
        _move
    )
    op_MaxPool = op_GlobalMaxPool = op_ReduceMax = op_ReduceMin = op_Split = _move
    op_Dropout = _move

    def op_Pad(self, node):
        # Padding value is 0 (exact); run on the error array with the constant_value input dropped.
        extra = [self.iv[x][0] for x in node.input[1:2] if x]
        return self.runner.run(node, [self.err[node.input[0]]] + extra)

    def op_Concat(self, node):
        return self.runner.run(node, [self.err[x] for x in node.input])

    # ---- library functions -------------------------------------------------
    @staticmethod
    def _f(op: str, z: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            if op == "Sigmoid":
                return 1.0 / (1.0 + np.exp(-z))
            if op == "Tanh":
                return np.tanh(z)
            if op == "Exp":
                return np.exp(z)
            if op == "Erf":
                from math import erf

                return np.vectorize(erf, otypes=[np.float64])(z)
            if op == "Sqrt":
                return np.sqrt(np.maximum(z, 0.0))
            if op == "Log":
                return np.log(z)
            if op == "Softplus":
                return np.log1p(np.exp(z))
        raise _NoRule(op)

    def _elementary(self, node):
        op = node.op_type
        x = node.input[0]
        e = self.err[x]
        lo, hi = self.iv[x]
        lo_h, hi_h = (
            lo.astype(np.float64) - e,
            hi.astype(np.float64) + e,
        )  # range of the computed input
        with np.errstate(all="ignore"):
            if op == "Sigmoid":
                lip = np.full(e.shape, 0.25)
            elif op == "Tanh" or op == "Softplus":
                lip = np.ones(e.shape)
            elif op == "Exp":
                lip = np.exp(hi_h)
            elif op == "Erf":
                lip = np.full(e.shape, 2.0 / math.sqrt(math.pi))
            elif op == "Sqrt":
                lip = np.where(
                    lo_h > 0, 0.5 / np.sqrt(np.where(lo_h > 0, lo_h, 1.0)), _INF
                )
            elif op == "Log":
                lip = np.where(lo_h > 0, 1.0 / np.where(lo_h > 0, lo_h, 1.0), _INF)
            else:  # pragma: no cover - guarded by the op_* aliases below
                raise _NoRule(op)
            # all these are increasing: the largest |f| over the computed range is at an end
            if op in ("Sqrt", "Log"):
                lo_h = np.maximum(
                    lo_h, 0.0 if op == "Sqrt" else np.finfo(np.float64).tiny
                )
            mf = np.maximum(np.abs(self._f(op, lo_h)), np.abs(self._f(op, hi_h))) * (
                1 + 1e-12
            )
            mf = np.where(np.isnan(mf), _INF, mf)
            # |f^(x^) - f(x)| <= |f(x^) - f(x)| + |f^(x^) - f(x^)|  <=  L*e + (rel*u*|f| + abs*u)
            lib = self.lib.rel[op] * self.u() * mf + self.lib.abs_[op] * self.u()
            loc = lib + self.prec.eta
            out = np.where(e == 0, 0.0, lip * e) + loc
        self._local(node, np.where(np.isnan(loc), _INF, loc))
        return [out]

    op_Sigmoid = op_Tanh = op_Exp = op_Erf = op_Sqrt = op_Log = op_Softplus = (
        _elementary
    )

    # ---- linear / bilinear -------------------------------------------------
    def _depth(self, node, a_shape, b_shape) -> int:
        t = node.op_type
        at = _interval._attrs(node)
        if t == "MatMul":
            return int(a_shape[-1]) if len(a_shape) else 1
        if t == "Gemm":
            return int(a_shape[0] if at.get("transA", 0) else a_shape[1])
        g = int(at.get("group", 1))
        if t == "Conv":
            return int(np.prod(b_shape[1:], dtype=np.int64))
        if t == "ConvTranspose":
            return int((b_shape[0] // g) * np.prod(b_shape[2:], dtype=np.int64))
        raise _NoRule(t)

    def _bilinear(self, node):
        t = node.op_type
        a, b = node.input[0], node.input[1]
        c = node.input[2] if len(node.input) > 2 and node.input[2] else None
        ma, ea, mb, eb = self.mag(a), self.err[a], self.mag(b), self.err[b]
        n = self._depth(node, ma.shape, mb.shape)
        u, eta = self.u(), self.prec.eta
        ov = {"alpha": 1.0, "beta": 0.0} if t == "Gemm" else {}

        def op(x, y):
            return self.runner.run(node, [x, y], **ov)[0]

        prop = op(ea, mb + eb) + op(
            ma, eb
        )  # |sum a^b^ - sum ab| <= sum ea(|b|+eb) + |a| eb
        mag = op(ma + ea, mb + eb)  # sum |a^||b^|
        at = _interval._attrs(node)
        alpha, beta = float(at.get("alpha", 1.0)), float(at.get("beta", 1.0))
        scaled = t == "Gemm" and (alpha != 1.0 or beta != 1.0)
        if c is None or scaled:
            loc = _gamma(n, u) * mag + n * eta
            e = prop + loc
            self.absmag[node.output[0]] = mag
            if c is None and not scaled:
                self._local(node, loc)
                return [e]
            # Gemm with non-unit alpha/beta: scale dot and bias separately, then add.
            mreal = op(ma, mb)
            if alpha != 1.0:
                e = abs(alpha) * e + u * abs(alpha) * (mreal + e) + eta
            if c is None:
                return [e]
            mc, ec = self.mag(c), self.err[c]
            if beta != 1.0:
                ec = abs(beta) * ec + u * abs(beta) * (mc + ec) + eta
            return [(e + ec) * (1 + u) + u * self.mag(node.output[0])]
        # fused bias: one more term in the same accumulation
        mc, ec = self.mag(c), self.err[c]
        if t in ("Conv", "ConvTranspose") and mc.ndim == 1:
            shp = [1, -1] + [1] * (mag.ndim - 2)
            mc, ec = mc.reshape(shp), ec.reshape(shp)
        loc = _gamma(n + 1, u) * (mag + mc + ec) + (n + 1) * eta
        self._local(node, loc)
        self.absmag[node.output[0]] = mag + mc + ec
        return [prop + ec + loc]

    op_MatMul = op_Gemm = op_Conv = op_ConvTranspose = _bilinear

    def op_BatchNormalization(self, node):
        if _interval._attrs(node).get("training_mode", 0):
            raise _NoRule("training_mode=1")
        x = node.input[0]
        consts = []
        for nm in node.input[1:5]:
            lo, hi = self.iv[nm]
            if not np.array_equal(lo, hi):
                raise _NoRule("BatchNormalization parameters must be constants")
            consts.append(np.asarray(lo, dtype=np.float64))
        scale, bias, mean, var = consts
        eps = float(_interval._attrs(node).get("epsilon", 1e-5))
        s = scale / np.sqrt(var + eps)
        shp = [1, -1] + [1] * (self.mag(x).ndim - 2)
        s, beta, mu = s.reshape(shp), bias.reshape(shp), mean.reshape(shp)
        u, eta = self.u(), self.prec.eta
        rho = self.bn_ulps * u
        # The scale/shift constants are re-derived at run time (sqrt, div, mul, sub) or, when the
        # runtime folds BatchNorm into the preceding weights, rounded into W' = W*s. Either way each
        # constant is perturbed by a relative rho, which acts on sum|w||x| of the producer in the
        # folded case -- hence the substitution of its absmag for |x| below.
        mx = self.absmag.get(x, self.mag(x))
        ex = self.err[x]
        mx_hat = np.maximum(mx, self.mag(x)) + ex
        tau = rho * (np.abs(beta) + np.abs(mu * s))
        l1 = rho * np.abs(s) * mx_hat + u * np.abs(s) * (1 + rho) * mx_hat + eta + tau
        e1 = (
            np.abs(s) * ex + l1
        )  # propagated input error (BN is affine in x) + constants/mul
        loc = l1 + u * (
            self.mag(node.output[0]) + e1
        )  # ... + rounding of the final add
        self._local(node, loc)
        return [np.abs(s) * ex + loc]

    # ---- reductions / pooling ---------------------------------------------
    def _avg(self, node, n: float, sum_only: bool):
        x = node.input[0]
        extra = [
            self.iv[i][0] for i in node.input[1:] if i
        ]  # e.g. ReduceSum's axes input
        ex = self.runner.run(node, [self.err[x]] + extra)[0]
        mx = self.runner.run(node, [self.mag(x) + self.err[x]] + extra)[0]
        u = self.u()
        g = _gamma(n if sum_only else n + 1, u)
        loc = g * mx + n * self.prec.eta
        self._local(node, loc)
        return [ex + loc]

    def op_ReduceSum(self, node):
        x, o = node.input[0], node.output[0]
        n = float(np.prod(self.iv[x][0].shape, dtype=np.int64)) / max(
            1, self.iv[o][0].size
        )
        return self._avg(node, n, True)

    def op_ReduceMean(self, node):
        x, o = node.input[0], node.output[0]
        n = float(np.prod(self.iv[x][0].shape, dtype=np.int64)) / max(
            1, self.iv[o][0].size
        )
        return self._avg(node, n, False)

    def op_GlobalAveragePool(self, node):
        n = float(np.prod(self.iv[node.input[0]][0].shape[2:], dtype=np.int64))
        return self._avg(node, n, False)

    def op_AveragePool(self, node):
        n = float(np.prod(_interval._attrs(node)["kernel_shape"], dtype=np.int64))
        return self._avg(node, n, False)

    # ---- softmax -----------------------------------------------------------
    def op_Softmax(self, node):
        if self.opset < 13:
            raise _NoRule("Softmax before opset 13 flattens to 2-D (not modelled)")
        x = node.input[0]
        lo, hi = self.iv[x]
        ax = int(_interval._attrs(node).get("axis", -1))
        n = lo.shape[ax]
        e_inf = np.max(self.err[x], axis=ax, keepdims=True)
        spread = (
            np.max(hi, axis=ax, keepdims=True) - np.min(lo, axis=ax, keepdims=True)
        ).astype(np.float64) + 2 * e_inf
        u = self.u()
        rel_exp = self.lib.rel["Exp"] * u
        with np.errstate(all="ignore"):
            # Input sensitivity: log softmax_i = x_i - logsumexp(x), logsumexp is 1-Lipschitz in
            # the inf-norm, so |d log y_i| <= 2*e_inf and |y^_i - y_i| <= y_i * expm1(2*e_inf) <= expm1(2 e).
            e_in = np.expm1(2.0 * e_inf)
            # Computation (max-subtracted): argument z = x - max rounds with relative error u
            # (|z| <= spread) -> exp relative error expm1(u*spread); exp library error rel_exp.
            eps1 = (1 + np.expm1(u * spread)) * (1 + rel_exp) - 1
            g = _gamma(max(n - 1, 0), u)
            denom = (1 - eps1) * (1 - g)
            rel_c = np.where(denom > 0, (1 + eps1) * (1 + u) / denom - 1, _INF)
            loc = rel_c * np.exp(2.0 * e_inf) + self.prec.eta
            e = e_in + loc
        self._local(node, np.where(np.isnan(loc), _INF, loc))
        return [e]


def roundoff_bound(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    precision: str = "fp32",
    lib: Optional[LibmModel] = None,
    bn_ulps: float = 8.0,
    weights_exact: Optional[bool] = None,
    input_exact: bool = True,
    ranges: str = "auto",
    tight: bool = True,
) -> RoundoffResult:
    """Certified bound on ``|float execution - real semantics|`` for inputs in a box.

    :param input_ranges: ``{input: (lo, hi)}`` (merged over ``onnxsim.ranges`` annotations on
        the model). Without a box the real-value ranges are unbounded and every error that
        depends on a magnitude is ``inf`` -- the bound is meaningless without one.
    :param precision: ``"fp32"`` (default), ``"fp16"`` or ``"bf16"``: the format the model is
        executed in. Constants stored in a wider format are rounded to it (relative error ``u``).
    :param lib: error model of the elementary functions (see :class:`LibmModel`); an assumption.
    :param bn_ulps: relative error, in units of ``u``, of BatchNorm's re-derived constants
        (covers the unfused run-time form and the folded-into-weights form).
    :param weights_exact: force constants exact (``True``) or rounded (``False``); default is
        automatic from the stored dtype versus ``precision``.
    :param input_exact: the box holds numbers already representable in ``precision``.
    :param ranges: where the real-value magnitudes come from. ``"interval"``: plain interval
        arithmetic (cheap, but inflates magnitudes ~``sum|w|`` per layer, so deep-net bounds are
        very loose). ``"zonotope"``: intersect with :func:`onnxsim.zonotope.propagate` ranges
        (much tighter; dense, small/medium nets only; raises if it cannot run). ``"auto"``
        (default): zonotope when affordable, else interval, noted in ``notes``.
    :param tight: after the forward bound, also run the correlation-aware pass
        (:meth:`_Analyzer.tighten_errors`) and keep the elementwise minimum of the two sound
        bounds. Needs a bounded box and a modest graph (dense generators); otherwise skipped
        with a note and the forward bound stands.

    Returns per-tensor and per-output elementwise bounds. Tensors the analysis cannot size
    (unknown shape) are absent from ``errors``; an output without a bound is absent from
    ``outputs`` and ``worst`` is then ``inf``.
    """
    if precision not in PRECISIONS:
        raise ValueError(
            f"unknown precision {precision!r}; expected one of {sorted(PRECISIONS)}"
        )
    a = _Analyzer(
        model,
        input_ranges,
        PRECISIONS[precision],
        lib or DEFAULT_LIBM,
        bn_ulps,
        weights_exact,
        input_exact,
        ranges,
    )
    a.run()
    if tight:
        a.tighten_errors()
    outputs = {o.name: a.err[o.name] for o in model.graph.output if o.name in a.err}
    missing = [o.name for o in model.graph.output if o.name not in a.err]
    for m in missing:
        a.notes.append(f"output {m}: no shape/interval information; no bound")
    if a.real.unsupported:
        a.notes.append(
            "interval fallback (unbounded) used for: " + ", ".join(a.real.unsupported)
        )
    return RoundoffResult(precision, a.err, outputs, a.notes, a.unsupported)


@dataclasses.dataclass
class Tolerance:
    """Certified tolerance for comparing the executions of two models; see :func:`tolerance_for`."""

    atol: Dict[str, float]  # per output: |orig_run - simplified_run| <= atol (absolute)
    real_difference: Dict[
        str, float
    ]  # part from real-arithmetic reasoning (zonotope bound)
    roundoff_orig: Dict[str, float]
    roundoff_simplified: Dict[str, float]
    notes: List[str]

    @property
    def worst(self) -> float:
        return float(max(self.atol.values())) if self.atol else _INF

    @property
    def bounded(self) -> bool:
        return bool(np.isfinite(self.worst))


def tolerance_for(
    orig: onnx.ModelProto,
    simplified: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    precision: str = "fp32",
    lib: Optional[LibmModel] = None,
) -> Tolerance:
    """Certified ``atol`` so that ``|run(orig) - run(simplified)| <= atol`` for every input in the box.

    ``|o^ - s^| <= |o^ - o| + |o - s| + |s - s^|``: the two models' roundoff bounds
    (:func:`roundoff_bound`) plus the real-arithmetic difference bound
    (:func:`onnxsim.zonotope.bound_difference`). Absolute only -- there is no rtol, which keeps
    it sound where an output crosses zero. All assumptions of :func:`roundoff_bound` apply to both
    models (notably the library-function error model and execution-as-written).
    """
    from . import zonotope

    diff = zonotope.bound_difference(orig, simplified, input_ranges)
    ra = roundoff_bound(orig, input_ranges, precision, lib)
    rb = roundoff_bound(simplified, input_ranges, precision, lib)
    names = [o.name for o in orig.graph.output]
    atol: Dict[str, float] = {}
    d_real: Dict[str, float] = {}
    r_a: Dict[str, float] = {}
    r_b: Dict[str, float] = {}
    notes = (
        [f"zonotope: {n}" for n in diff.notes]
        + [f"orig: {n}" for n in ra.notes]
        + [f"simplified: {n}" for n in rb.notes]
    )
    for nm in names:
        d = diff.max_abs.get(nm)
        d_real[nm] = (
            float(np.max(d))
            if d is not None and np.size(d)
            else (_INF if d is None else 0.0)
        )
        r_a[nm] = float(np.max(ra.outputs[nm])) if nm in ra.outputs else _INF
        r_b[nm] = float(np.max(rb.outputs[nm])) if nm in rb.outputs else _INF
        atol[nm] = d_real[nm] + r_a[nm] + r_b[nm]
    return Tolerance(atol, d_real, r_a, r_b, notes)
