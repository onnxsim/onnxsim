"""Certified bounds on ``|quantized(x) - float(x)|`` for quantized ONNX models.

``verify(float_model, quantized_model, input_ranges)`` proves, for every input in a box,
an upper bound on how far each output of a quantized model can be from the float model it
came from -- instead of measuring the drop on a calibration set the way
``onnxsim.accuracy.measure_accuracy_drop`` does, or assuming the scheme's full integer
range the way ``onnxsim.precision_estimator`` does.

How. The quantized model is converted to an equivalent *float graph with bounded noise*:

* ``QuantizeLinear -> DequantizeLinear`` (the value a quantized tensor really carries) is
  ``x`` plus a rounding error ``e`` with ``|e| <= scale / 2`` *inside* the representable range
  ``[(qmin - zp) * scale, (qmax - zp) * scale]``. That is the lemma proved with Z3 in
  ``tests/test_formal_verify_quantize_round_trip.py``; the MAC composition lemma in
  ``tests/test_formal_verify_quantized_mac_bound.py`` is what the product-graph machinery
  below applies automatically through every MatMul/Conv/Gemm.
* **Clipping is accounted for, not assumed away.** A value outside the representable range
  saturates, and its error is the distance beyond the range, which ``scale / 2`` does not
  bound. The noise radius of every site is therefore
  ``max(scale / 2, hi - Rmax, Rmin - lo)`` where ``[lo, hi]`` is the *propagated* range of the
  tensor entering the quantizer (including upstream noise). Because that range depends on the
  radii of earlier sites, the radii are iterated to a fixed point; if it is not reached the
  affected radii become ``inf`` (never a wrong finite bound). An unaccounted clip is the
  classic way a quantization verifier is unsound, which is why this is not optional.
* ``DequantizeLinear`` on a *constant* (per-tensor, per-axis or blockwise, int4 through int16)
  becomes the exact dequantized constant, so the float-versus-quantized *weight* difference is
  exact structure, not noise. Constant ``QuantizeLinear`` is evaluated exactly.
* ``QLinearConv`` / ``QLinearMatMul`` become the float op on the dequantized operands followed
  by the output quantizer's noise site (the integer accumulation is exact in real arithmetic).

The float model and the converted model then go through
:func:`onnxsim.zonotope.bound_difference` with one extra graph input per noise site (range
``[-r, r]``), so shared structure cancels exactly and the noise symbols are shared.

What a bound means -- and does not:

* It is a statement in **real arithmetic**. float32 evaluation of either model adds roundoff
  (about ``1e-6`` relative per op); budget for it, or see ``onnxsim.fp_error`` once available.
* The noise model only needs ``round`` to return an integer within ``0.5`` of its argument,
  so the tie rule does not matter.
* **The integer pipeline's exactness against the QDQ semantics is not verified here.** That
  the int32 accumulator does not overflow is only *checked* (worst case over the full integer
  ranges) and reported as a hazard.
* Ops this module has no rule for in the *quantized-domain* (``MatMulInteger``,
  ``ConvInteger``, ``DynamicQuantizeLinear``, ``QLinearAdd`` ...) make the affected bound
  ``inf`` with a note. Nothing unsupported is ever guessed.
* Ranges propagated through the converted model use interval arithmetic by default
  (``range_method="zonotope"`` is tighter and slower); this only affects how much clipping
  noise is charged, never soundness.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import backward_diff as _backward_diff
from . import interval as _interval
from . import ranges as _ranges
from . import zonotope as _zonotope
from ._onnx_compat import INT4, UINT4

# Representable integer range of each quantized element type.
_QRANGE: Dict[int, Tuple[int, int]] = {
    TensorProto.UINT8: (0, 255),
    TensorProto.INT8: (-128, 127),
    TensorProto.UINT16: (0, 65535),
    TensorProto.INT16: (-32768, 32767),
    # UINT4/INT4 come from _onnx_compat, not TensorProto: that attribute is missing on onnx < 1.16
    # and a module-level access would break `import onnxsim` there (tests/test_onnx_compat.py).
    UINT4: (0, 15),
    INT4: (-8, 7),
}
_NOISE_PREFIX = "__qnoise_"
_INT32_MAX = 2**31 - 1


class _Unsupported(Exception):
    """The quantized model uses something this converter cannot represent soundly."""


@dataclasses.dataclass
class _Site:
    """One place where quantization rounds a tensor (a ``QuantizeLinear`` or a QLinear* output)."""

    label: str
    kind: str
    input_name: (
        str  # tensor (in the converted graph) whose range decides the clipping error
    )
    noise_name: str
    shape: Tuple[int, ...]
    half_step: np.ndarray  # scale / 2, broadcast to ``shape``
    r_min: np.ndarray  # lowest representable value, broadcast to ``shape``
    r_max: np.ndarray
    neg: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(())
    )  # noise in [-neg, pos]
    pos: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(()))
    can_clip: bool = False
    dynamic: bool = (
        False  # DynamicQuantizeLinear: scale = (max - min) / 255 of the data itself
    )

    @property
    def radius(self) -> np.ndarray:
        return np.maximum(self.neg, self.pos)


@dataclasses.dataclass
class SiteReport:
    """What one quantizer contributes (see :attr:`QuantVerifyReport.sites`)."""

    label: str
    kind: str
    half_step: float  # largest scale / 2 over the tensor
    radius: float  # noise radius charged, including any clipping term
    clipped: bool  # the propagated range exceeds the representable range
    contribution: Optional[
        float
    ]  # worst-case |float - quantized| with ONLY this site's noise


@dataclasses.dataclass
class QuantVerifyReport:
    """Result of :func:`verify`."""

    outputs: Dict[
        str, np.ndarray
    ]  # per output: elementwise certified bound on |float - quantized|
    ref_min_abs: Dict[
        str, np.ndarray
    ]  # per output: lower bound on |quantized| (for rtol)
    weights_only: Optional[
        float
    ]  # worst bound with every activation quantizer made exact
    sites: List[SiteReport]
    hazards: List[
        str
    ]  # reasons a bound is infinite or an assumption needs your attention
    notes: List[str]

    @property
    def worst(self) -> float:
        """Largest certified per-element error over all outputs (``inf`` if any is unbounded)."""
        vals = [float(np.max(v)) if np.size(v) else 0.0 for v in self.outputs.values()]
        return max(vals) if vals else 0.0

    @property
    def bounded(self) -> bool:
        return bool(np.isfinite(self.worst))

    def within(self, atol: float, rtol: float = 0.0) -> bool:
        """Sound check that ``|float - quantized| <= atol + rtol * |quantized|`` everywhere."""
        for name, bound in self.outputs.items():
            if not np.all(np.isfinite(bound)):
                return False
            if not np.all(bound <= atol + rtol * self.ref_min_abs[name]):
                return False
        return True

    def __str__(self) -> str:
        lines = [f"quantization verify: worst |float - quantized| <= {self.worst:.6g}"]
        if self.weights_only is not None:
            lines.append(f"  weights/constants only: {self.weights_only:.6g}")
        for s in sorted(self.sites, key=lambda s: -(s.contribution or 0.0)):
            c = "n/a" if s.contribution is None else f"{s.contribution:.6g}"
            lines.append(
                f"  site {s.label} ({s.kind}): scale/2 {s.half_step:.4g}, radius {s.radius:.4g}"
                f"{' CLIPPED' if s.clipped else ''}, alone {c}"
            )
        lines += [f"  hazard: {h}" for h in self.hazards]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Quantization parameter helpers
# --------------------------------------------------------------------------


def _expand(
    vec: np.ndarray, shape: Tuple[int, ...], axis: int, block: int = 0
) -> np.ndarray:
    """Broadcast a per-tensor / per-axis / blockwise scale or zero point to ``shape``."""
    vec = np.asarray(vec)
    if block:
        ax = axis % len(shape)
        full = np.repeat(vec, block, axis=ax)
        idx = [slice(None)] * len(shape)
        idx[ax] = slice(0, shape[ax])
        out = full[tuple(idx)]
        if out.shape != tuple(shape):
            raise _Unsupported(f"block scale shape {vec.shape} does not tile {shape}")
        return out
    if vec.size == 1:
        return vec.reshape(())
    ax = axis % len(shape)
    if vec.ndim != 1 or vec.shape[0] != shape[ax]:
        raise _Unsupported(
            f"scale shape {vec.shape} does not match axis {ax} of {shape}"
        )
    view = [1] * len(shape)
    view[ax] = vec.shape[0]
    return vec.reshape(view)


def _fake_quant(
    x: np.ndarray, s: np.ndarray, zp: np.ndarray, lo: int, hi: int
) -> np.ndarray:
    """``DQ(Q(x))`` exactly as ONNX defines it (round half to even, saturate)."""
    q = np.clip(np.rint(x.astype(np.float32) / s.astype(np.float32)) + zp, lo, hi)
    return ((q - zp) * s.astype(np.float32)).astype(np.float32)


def _dequant(q: np.ndarray, s: np.ndarray, zp: np.ndarray) -> np.ndarray:
    return ((q.astype(np.int64) - zp.astype(np.int64)) * s.astype(np.float32)).astype(
        np.float32
    )


def _attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


class _Converter:
    """Rewrites a quantized model into a float graph with one noise input per rounding site."""

    def __init__(self, model: onnx.ModelProto, clamp: bool = True):
        self.model = model
        self.clamp = clamp
        g = model.graph
        self.consts: Dict[str, np.ndarray] = {}
        self.const_dtype: Dict[str, int] = {}
        for t in g.initializer:
            self.consts[t.name] = numpy_helper.to_array(t)
            self.const_dtype[t.name] = t.data_type
        for n in g.node:
            if n.op_type == "Constant" and not n.domain:
                v = _attr(n, "value")
                if v is not None:
                    self.consts[n.output[0]] = numpy_helper.to_array(v)
                    self.const_dtype[n.output[0]] = v.data_type
        try:
            inferred = onnx.shape_inference.infer_shapes(model).graph
        except Exception:
            inferred = g
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        for vi in (
            list(inferred.input) + list(inferred.value_info) + list(inferred.output)
        ):
            tt = vi.type.tensor_type
            if vi.type.HasField("tensor_type") and tt.HasField("shape"):
                dims = [
                    d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 0
                    for d in tt.shape.dim
                ]
                if dims and all(d > 0 for d in dims):
                    self.shapes[vi.name] = tuple(dims)
        for t in g.initializer:
            self.shapes[t.name] = tuple(t.dims)
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self.sites: List[_Site] = []
        self.hazards: List[str] = []
        self.notes: List[str] = []
        # tensors that carry the *dequantized* value of an integer tensor, with the params used
        self.qparams: Dict[str, Tuple[np.ndarray, np.ndarray, int, int, int]] = {}
        self.real_const: Dict[str, np.ndarray] = {}
        # DynamicQuantizeLinear's scale/zero-point outputs, and the integer accumulators built on them
        self.dyn_scales: set = set()
        self.dyn_zps: set = set()
        self.pending: Dict[
            str, Tuple[str, np.ndarray, np.ndarray]
        ] = {}  # acc -> (a_real, wq, wzp)
        self.scale_chain: Dict[
            str, np.ndarray
        ] = {}  # combined scale tensor -> weight scale const

    # -- helpers -----------------------------------------------------------
    def _need_const(self, name: str, what: str, node: onnx.NodeProto) -> np.ndarray:
        if name not in self.consts:
            raise _Unsupported(f"{node.op_type} {what} {name!r} is not a constant")
        return self.consts[name]

    def _qrange(self, zp_name: str, node: onnx.NodeProto) -> Tuple[int, int]:
        if zp_name and zp_name in self.const_dtype:
            dt = self.const_dtype[zp_name]
        else:
            dt = int(_attr(node, "output_dtype", TensorProto.UINT8))
        if dt not in _QRANGE:
            raise _Unsupported(
                f"{node.op_type} with quantized type {dt} is not supported"
            )
        return _QRANGE[dt]

    def _zp(self, name: str, node: onnx.NodeProto, like: np.ndarray) -> np.ndarray:
        if not name:
            return np.zeros_like(like, dtype=np.int64)
        return self._need_const(name, "zero point", node).astype(np.int64)

    def _add_site(
        self,
        node: onnx.NodeProto,
        kind: str,
        input_name: str,
        out_name: str,
        scale: np.ndarray,
        zp: np.ndarray,
        qmin: int,
        qmax: int,
        axis: int,
    ) -> str:
        shape = self.shapes.get(input_name) or self.shapes.get(out_name)
        if shape is None:
            raise _Unsupported(
                f"{kind} {node.name or out_name!r}: tensor shape is not static"
            )
        s = _expand(scale, shape, axis)
        z = _expand(zp, shape, axis)
        half = np.broadcast_to(np.asarray(s, dtype=np.float64) / 2.0, shape).copy()
        r_min = np.broadcast_to(
            (qmin - np.asarray(z, dtype=np.float64)) * s, shape
        ).copy()
        r_max = np.broadcast_to(
            (qmax - np.asarray(z, dtype=np.float64)) * s, shape
        ).copy()
        k = len(self.sites)
        noise = f"{_NOISE_PREFIX}{k}"
        self.sites.append(
            _Site(
                node.name or out_name,
                kind,
                input_name,
                noise,
                shape,
                half,
                r_min,
                r_max,
                half.copy(),
                half.copy(),
            )
        )
        return noise

    def _emit_site(self, node: onnx.NodeProto, noise: str, x: str, out: str) -> None:
        """``out = DQ(Q(x))``: exact saturation (``clamp``) plus rounding noise (``scale/2``).

        ``clamp(x, a, b) = a + Relu(x - a) - Relu(x - b)``: stable (so free) whenever the range of
        ``x`` stays inside ``[a, b]``. With ``clamp=False`` the saturation error is instead charged
        as extra independent noise radius (looser; kept for comparison and as a fallback).
        """
        label = node.name or out
        if not self.clamp:
            self.nodes.append(
                helper.make_node("Add", [x, noise], [out], name=label + "_q")
            )
            return
        site = next(s for s in self.sites if s.noise_name == noise)
        a, b = f"{noise}__lo", f"{noise}__hi"
        self.inits.append(numpy_helper.from_array(site.r_min.astype(np.float32), a))
        self.inits.append(numpy_helper.from_array(site.r_max.astype(np.float32), b))
        t = [f"{noise}__t{i}" for i in range(5)]
        self.nodes += [
            helper.make_node("Sub", [x, a], [t[0]], name=label + "_c0"),
            helper.make_node("Relu", [t[0]], [t[1]], name=label + "_c1"),
            helper.make_node("Sub", [x, b], [t[2]], name=label + "_c2"),
            helper.make_node("Relu", [t[2]], [t[3]], name=label + "_c3"),
            helper.make_node("Add", [a, t[1]], [t[4]], name=label + "_c4"),
        ]
        clamped = f"{noise}__clamped"
        self.nodes.append(
            helper.make_node("Sub", [t[4], t[3]], [clamped], name=label + "_c5")
        )
        self.nodes.append(
            helper.make_node("Add", [clamped, noise], [out], name=label + "_q")
        )

    # -- conversion --------------------------------------------------------
    def run(self) -> onnx.ModelProto:
        g = self.model.graph
        for node in g.node:
            if node.domain not in ("", "ai.onnx", "com.microsoft"):
                self.nodes.append(node)
                continue
            prefix = "_ms_" if node.domain == "com.microsoft" else "_do_"
            fn = getattr(self, prefix + node.op_type, None)
            if fn is not None:
                fn(node)
            elif node.op_type in _UNSUPPORTED_QUANT_OPS:
                raise _Unsupported(
                    f"{node.op_type} is a quantized-domain op with no rule here"
                )
            elif node.domain == "com.microsoft" and node.op_type.startswith(
                ("QLinear", "MatMulNBits")
            ):
                raise _Unsupported(f"{node.domain}::{node.op_type} is not supported")
            else:
                self.nodes.append(node)
        self._check_defined()
        new = onnx.ModelProto()
        new.CopyFrom(self.model)
        del new.graph.node[:]
        new.graph.node.extend(self.nodes)
        new.graph.initializer.extend(self.inits)
        keep = {t.name for t in self.inits}
        # a constant Constant-node DQ replacement may collide with an existing initializer name
        seen: Dict[str, int] = {}
        for i, t in enumerate(new.graph.initializer):
            seen[t.name] = i
        dedup = [t for i, t in enumerate(new.graph.initializer) if seen[t.name] == i]
        del new.graph.initializer[:]
        new.graph.initializer.extend(dedup)
        del keep
        for site in self.sites:
            new.graph.input.append(
                helper.make_tensor_value_info(
                    site.noise_name, TensorProto.FLOAT, list(site.shape)
                )
            )
        return new

    def _emit_const(self, name: str, arr: np.ndarray) -> None:
        self.inits.append(numpy_helper.from_array(arr.astype(np.float32), name))
        self.real_const[name] = arr.astype(np.float32)
        self.consts[name] = arr.astype(np.float32)
        self.const_dtype[name] = TensorProto.FLOAT

    def _do_QuantizeLinear(self, node: onnx.NodeProto) -> None:
        x, s_name = node.input[0], node.input[1]
        zp_name = node.input[2] if len(node.input) > 2 else ""
        out = node.output[0]
        if _attr(node, "block_size", 0):
            raise _Unsupported(
                "blockwise QuantizeLinear on an activation is not supported"
            )
        scale = self._need_const(s_name, "scale", node)
        qmin, qmax = self._qrange(zp_name, node)
        zp = self._zp(zp_name, node, scale)
        axis = int(_attr(node, "axis", 1))
        if x in self.consts:  # quantizing a constant: exact
            xs = self.consts[x]
            s = _expand(scale, xs.shape, axis)
            z = _expand(zp, xs.shape, axis)
            self._emit_const(out, _fake_quant(xs, s, z, qmin, qmax))
            return
        noise = self._add_site(
            node, "QuantizeLinear", x, out, scale, zp, qmin, qmax, axis
        )
        self._emit_site(node, noise, x, out)
        self.qparams[out] = (scale, zp, qmin, qmax, axis)

    def _do_DequantizeLinear(self, node: onnx.NodeProto) -> None:
        x, s_name = node.input[0], node.input[1]
        zp_name = node.input[2] if len(node.input) > 2 else ""
        out = node.output[0]
        scale = self._need_const(s_name, "scale", node)
        axis = int(_attr(node, "axis", 1))
        block = int(_attr(node, "block_size", 0))
        if x in self.consts and x not in self.real_const:
            q = self.consts[x]
            zp = self._zp(zp_name, node, scale)
            s = _expand(scale, q.shape, axis, block)
            z = _expand(zp, q.shape, axis, block)
            self._emit_const(out, _dequant(q, s, z))
            return
        if x in self.real_const:  # already a real-valued constant (a folded Q->DQ)
            self.nodes.append(helper.make_node("Identity", [x], [out]))
            return
        if x in self.qparams:  # DQ of a tensor that carries Q's dequantized value
            s0, z0, _, _, ax0 = self.qparams[x]
            zp = self._zp(zp_name, node, scale)
            if not (
                np.array_equal(np.asarray(s0), np.asarray(scale))
                and np.array_equal(np.asarray(z0), np.asarray(zp))
            ):
                raise _Unsupported(
                    f"DequantizeLinear {node.name or out!r} uses different scale/zero point than the "
                    "QuantizeLinear that produced its input"
                )
            self.nodes.append(helper.make_node("Identity", [x], [out]))
            self.qparams[out] = self.qparams[x]
            return
        raise _Unsupported(
            f"DequantizeLinear {node.name or out!r}: input {x!r} is neither a constant nor produced "
            "by a QuantizeLinear / QLinear op this module understands"
        )

    def _qlinear_operand(
        self, node: onnx.NodeProto, v: str, s_name: str, zp_name: str, axis: int
    ) -> str:
        """The dequantized value of one QLinear* operand, as a tensor name in the converted graph."""
        scale = self._need_const(s_name, "scale", node)
        zp = self._zp(zp_name, node, scale)
        if v in self.consts and v not in self.real_const:
            q = self.consts[v]
            name = f"{v}__dq_{len(self.inits)}"
            s = _expand(scale, q.shape, axis if np.size(scale) > 1 else 0)
            z = _expand(zp, q.shape, axis if np.size(zp) > 1 else 0)
            self._emit_const(name, _dequant(q, s, z))
            return name
        if v in self.qparams:
            s0, z0, _, _, _ = self.qparams[v]
            if not (
                np.array_equal(np.asarray(s0), np.asarray(scale))
                and np.array_equal(np.asarray(z0), np.asarray(zp))
            ):
                raise _Unsupported(
                    f"{node.op_type} {node.name!r}: operand {v!r} scale/zero point differ from its producer"
                )
            return v
        raise _Unsupported(
            f"{node.op_type} {node.name!r}: operand {v!r} is not a constant or a Q output"
        )

    def _acc_hazard(
        self,
        node: onnx.NodeProto,
        w_q: np.ndarray,
        w_zp: np.ndarray,
        x_zp: np.ndarray,
        x_range: Tuple[int, int],
        depth_axes: Tuple[int, ...],
    ) -> None:
        """Worst-case |int32 accumulator| over the full integer ranges, as a hazard if it can wrap."""
        wd = np.abs(w_q.astype(np.int64) - w_zp.astype(np.int64))
        per_out = wd.sum(axis=depth_axes)
        xmax = max(
            abs(x_range[0] - int(np.max(x_zp))), abs(x_range[1] - int(np.min(x_zp)))
        )
        worst = int(np.max(per_out)) * int(xmax)
        if worst > _INT32_MAX:
            self.hazards.append(
                f"{node.op_type} {node.name or node.output[0]!r}: worst-case |int32 accumulator| "
                f"{worst} > {_INT32_MAX} over the full integer ranges; the integer pipeline may wrap "
                "(the bound here assumes it does not)"
            )

    def _do_QLinearConv(self, node: onnx.NodeProto) -> None:
        x, xs, xz, w, ws, wz, ys, yz = node.input[:8]
        bias = node.input[8] if len(node.input) > 8 and node.input[8] else ""
        out = node.output[0]
        x_real = self._qlinear_operand(node, x, xs, xz, 1)
        w_real = self._qlinear_operand(node, w, ws, wz, 0)
        ins = [x_real, w_real]
        if bias:
            b_q = self._need_const(bias, "bias", node)
            x_scale = self._need_const(xs, "scale", node)
            w_scale = self._need_const(ws, "scale", node)
            b_name = f"{bias}__dq_{len(self.inits)}"
            self._emit_const(
                b_name,
                b_q.astype(np.float64)
                * np.asarray(x_scale, np.float64)
                * np.asarray(w_scale, np.float64),
            )
            ins.append(b_name)
        wq = self._need_const(w, "weights", node)
        qmin_x, qmax_x = self._qrange(xz, node)
        self._acc_hazard(
            node,
            wq,
            _expand(self._zp(wz, node, self.consts[ws]), wq.shape, 0)
            if np.size(self._zp(wz, node, self.consts[ws])) > 1
            else self._zp(wz, node, self.consts[ws]),
            self._zp(xz, node, self.consts[xs]),
            (qmin_x, qmax_x),
            tuple(range(1, wq.ndim)),
        )
        tmp = f"{out}__qlinear_pre"
        conv_attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        self.nodes.append(
            helper.make_node(
                "Conv", ins, [tmp], name=(node.name or out) + "_f", **conv_attrs
            )
        )
        self._finish_qlinear(node, tmp, out, ys, yz)

    def _do_QLinearMatMul(self, node: onnx.NodeProto) -> None:
        a, as_, az, b, bs, bz, ys, yz = node.input[:8]
        out = node.output[0]
        a_real = self._qlinear_operand(node, a, as_, az, 0)
        b_axis = -1
        b_real = self._qlinear_operand(node, b, bs, bz, b_axis)
        if b in self.consts and self.consts[b].ndim == 2:
            bq = self.consts[b]
            bz_arr = self._zp(bz, node, self.consts[bs])
            self._acc_hazard(
                node,
                bq,
                np.asarray(bz_arr),
                self._zp(az, node, self.consts[as_]),
                self._qrange(az, node),
                (0,),
            )
        tmp = f"{out}__qlinear_pre"
        self.nodes.append(
            helper.make_node(
                "MatMul", [a_real, b_real], [tmp], name=(node.name or out) + "_f"
            )
        )
        self._finish_qlinear(node, tmp, out, ys, yz)

    def _finish_qlinear(
        self, node: onnx.NodeProto, tmp: str, out: str, ys: str, yz: str
    ) -> None:
        scale = self._need_const(ys, "output scale", node)
        qmin, qmax = self._qrange(yz, node)
        zp = self._zp(yz, node, scale)
        # tmp has no declared shape: record the output's
        if out in self.shapes:
            self.shapes[tmp] = self.shapes[out]
        noise = self._add_site(node, node.op_type, tmp, out, scale, zp, qmin, qmax, 1)
        self._emit_site(node, noise, tmp, out)
        self.qparams[out] = (scale, zp, qmin, qmax, 1)

    def _check_defined(self) -> None:
        """Every tensor the converted graph reads must be produced; a leaked integer-domain
        tensor (one of the accumulators or dynamic scales consumed by an unrecognised op)
        would otherwise make the analysis silently wrong."""
        defined = {t.name for t in self.model.graph.initializer} | {
            t.name for t in self.inits
        }
        defined |= {i.name for i in self.model.graph.input}
        for n in self.nodes:
            for o in n.output:
                defined.add(o)
        defined |= set(self.consts) | {site.noise_name for site in self.sites}
        for n in self.nodes:
            for i in n.input:
                if i and i not in defined:
                    raise _Unsupported(
                        f"{n.op_type} {n.name or n.output[0]!r} reads {i!r}, an integer-domain tensor "
                        "this module could not rewrite (only the DynamicQuantizeLinear -> MatMulInteger "
                        "-> Cast -> Mul pattern and MatMulIntegerToFloat are understood)"
                    )

    def _do_DynamicQuantizeLinear(self, node: onnx.NodeProto) -> None:
        x = node.input[0]
        y, y_scale, y_zp = node.output[0], node.output[1], node.output[2]
        shape = self.shapes.get(x)
        if shape is None:
            raise _Unsupported(
                f"DynamicQuantizeLinear {node.name or y!r}: tensor shape is not static"
            )
        k = len(self.sites)
        noise = f"{_NOISE_PREFIX}{k}"
        zero = np.zeros(shape)
        inf = np.full(shape, np.inf)
        self.sites.append(
            _Site(
                node.name or y,
                "DynamicQuantizeLinear",
                x,
                noise,
                shape,
                zero,
                -inf,
                inf,
                zero.copy(),
                zero.copy(),
                dynamic=True,
            )
        )
        self.nodes.append(
            helper.make_node("Add", [x, noise], [y], name=(node.name or y) + "_q")
        )
        self.dyn_scales.add(y_scale)
        self.dyn_zps.add(y_zp)
        self.qparams[y] = (np.array(0.0), np.array(0), 0, 255, 1)

    def _do_MatMulInteger(self, node: onnx.NodeProto) -> None:
        a, b = node.input[0], node.input[1]
        a_zp = node.input[2] if len(node.input) > 2 else ""
        b_zp = node.input[3] if len(node.input) > 3 else ""
        if a not in self.qparams:
            raise _Unsupported(
                f"MatMulInteger {node.name!r}: activation {a!r} is not the output of a quantizer"
            )
        if a_zp and a_zp not in self.dyn_zps:
            raise _Unsupported(
                f"MatMulInteger {node.name!r}: only the dynamic-quantization zero point is understood"
            )
        wq = self._need_const(b, "weights", node)
        wzp = self._zp(b_zp, node, np.zeros(()))
        self.pending[node.output[0]] = (a, wq, wzp)
        if wq.ndim == 2:
            self._acc_hazard(node, wq, wzp, np.array(0), (0, 255), (0,))

    def _do_Cast(self, node: onnx.NodeProto) -> None:
        if node.input[0] in self.pending:
            self.pending[node.output[0]] = self.pending[node.input[0]]
            return
        self.nodes.append(node)

    def _do_Mul(self, node: onnx.NodeProto) -> None:
        p, q = node.input[0], node.input[1]
        for a, b in ((p, q), (q, p)):
            if (
                a in self.dyn_scales and b in self.consts
            ):  # combined = y_scale * weight_scale
                self.scale_chain[node.output[0]] = np.asarray(self.consts[b])
                return
            if (
                a in self.pending and b in self.scale_chain
            ):  # acc * combined = the real matmul
                a_real, wq, wzp = self.pending[a]
                ws = self.scale_chain[b]
                w_real = _dequant(
                    wq,
                    _expand(ws, wq.shape, 1),
                    _expand(wzp, wq.shape, 1) if np.size(wzp) > 1 else wzp,
                )
                name = f"{node.output[0]}__wreal"
                self._emit_const(name, w_real)
                self.nodes.append(
                    helper.make_node(
                        "MatMul",
                        [a_real, name],
                        [node.output[0]],
                        name=(node.name or node.output[0]) + "_f",
                    )
                )
                return
        if (
            p in self.pending
            or q in self.pending
            or p in self.dyn_scales
            or q in self.dyn_scales
        ):
            raise _Unsupported(
                f"Mul {node.name or node.output[0]!r} uses an integer accumulator or dynamic scale in an unrecognised way"
            )
        self.nodes.append(node)

    def _ms_MatMulIntegerToFloat(self, node: onnx.NodeProto) -> None:
        a, b, a_scale, b_scale = (
            node.input[0],
            node.input[1],
            node.input[2],
            node.input[3],
        )
        a_zp = node.input[4] if len(node.input) > 4 else ""
        b_zp = node.input[5] if len(node.input) > 5 else ""
        bias = node.input[6] if len(node.input) > 6 else ""
        if a not in self.qparams or a_scale not in self.dyn_scales:
            raise _Unsupported(
                f"MatMulIntegerToFloat {node.name!r}: only dynamically quantized activations are understood"
            )
        if a_zp and a_zp not in self.dyn_zps:
            raise _Unsupported(
                f"MatMulIntegerToFloat {node.name!r}: unexpected activation zero point"
            )
        wq = self._need_const(b, "weights", node)
        ws = self._need_const(b_scale, "weight scale", node)
        wzp = self._zp(b_zp, node, np.zeros(()))
        name = f"{node.output[0]}__wreal"
        self._emit_const(
            name,
            _dequant(
                wq,
                _expand(ws, wq.shape, 1),
                _expand(wzp, wq.shape, 1) if np.size(wzp) > 1 else wzp,
            ),
        )
        out = node.output[0]
        if bias:
            tmp = f"{out}__mm"
            self.nodes.append(
                helper.make_node(
                    "MatMul", [a, name], [tmp], name=(node.name or out) + "_f"
                )
            )
            self.nodes.append(helper.make_node("Add", [tmp, bias], [out]))
        else:
            self.nodes.append(
                helper.make_node(
                    "MatMul", [a, name], [out], name=(node.name or out) + "_f"
                )
            )
        if wq.ndim == 2:
            self._acc_hazard(node, wq, wzp, np.array(0), (0, 255), (0,))


_UNSUPPORTED_QUANT_OPS = frozenset(
    {
        "ConvInteger",
        "QLinearAdd",
        "QLinearMul",
        "QLinearAveragePool",
        "QLinearGlobalAveragePool",
        "QLinearConcat",
        "QLinearSigmoid",
        "QLinearLeakyRelu",
        "QGemm",
    }
)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _model_inputs(model: onnx.ModelProto) -> List[str]:
    inits = {t.name for t in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in inits]


def _noise_ranges(
    sites: List[_Site], active: Optional[int]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for k, s in enumerate(sites):
        if active is None or active == k:
            out[s.noise_name] = (
                -np.broadcast_to(s.neg, s.shape),
                np.broadcast_to(s.pos, s.shape),
            )
        else:
            out[s.noise_name] = (np.zeros(s.shape), np.zeros(s.shape))
    return out


def _site_ranges(
    conv: onnx.ModelProto, ranges: Dict[str, Tuple], names: List[str], method: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    got: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if method == "zonotope":
        try:
            zres = _zonotope.propagate(conv, ranges)
            for n in names:
                if n in zres.tensors:
                    got[n] = zres.bounds(n)
            return got
        except Exception:  # unbounded input, unsupported op ...: fall back to intervals
            got = {}
    res = _interval.propagate(conv, ranges)
    for n in names:
        if n in res.intervals:
            lo, hi = res.intervals[n]
            got[n] = (np.asarray(lo, np.float64), np.asarray(hi, np.float64))
    return got


def _clip_noise(
    site: _Site, lo: np.ndarray, hi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Noise interval ``[-neg, pos]`` of a quantizer whose input ranges over ``[lo, hi]``.

    Inside the representable range the rounding error is within ``+-scale/2``. Above the range
    the value saturates to ``Rmax``, so the error is ``Rmax - x <= 0`` and at least ``Rmax - hi``;
    below it, the error is ``Rmin - x >= 0`` and at most ``Rmin - lo``. The union over all
    inputs is therefore ``[-max(scale/2, hi - Rmax), +max(scale/2, Rmin - lo)]``.
    """
    if site.dynamic:
        # scale = (max(x, 0) - min(x, 0)) / 255 over the whole tensor, at most the box hull's; the
        # zero point is rounded, which moves the representable range by <= scale/2 but leaves every
        # value of the data within scale/2 of some code, so |error| <= scale/2 (padded for fp32).
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            inf = np.full(site.shape, np.inf)
            return inf, inf.copy()
        s_max = (max(float(np.max(hi)), 0.0) - min(float(np.min(lo)), 0.0)) / 255.0
        r = np.full(site.shape, s_max / 2.0 * (1.0 + 1e-6))
        return r, r.copy()
    with np.errstate(invalid="ignore"):
        over = np.maximum(0.0, hi - site.r_max)
        under = np.maximum(0.0, site.r_min - lo)
        neg = np.maximum(site.half_step, over)
        pos = np.maximum(site.half_step, under)
    return np.where(np.isnan(neg), np.inf, neg), np.where(np.isnan(pos), np.inf, pos)


def _settle_noise(
    converted: onnx.ModelProto,
    sites: List[_Site],
    ranges: Dict[str, Tuple],
    range_method: str,
    max_iter: int,
    only_dynamic: bool = False,
) -> bool:
    """Grow every site's noise interval to a fixed point; ``False`` if none is reached.

    ``only_dynamic`` restricts this to DynamicQuantizeLinear sites, whose noise radius is a
    function of the tensor's range even when saturation is modelled exactly elsewhere.
    """
    for _ in range(max_iter):
        if not sites:
            return True
        rng_all = dict(ranges)
        rng_all.update(_noise_ranges(sites, None))
        got = _site_ranges(
            converted, rng_all, [s.input_name for s in sites], range_method
        )
        changed = False
        for s in sites:
            if only_dynamic and not s.dynamic:
                continue
            if s.input_name not in got:
                neg = pos = np.full(s.shape, np.inf)
            else:
                lo, hi = got[s.input_name]
                neg, pos = _clip_noise(
                    s, np.broadcast_to(lo, s.shape), np.broadcast_to(hi, s.shape)
                )
            if np.any(neg > s.neg * (1 + 1e-12) + 1e-300) or np.any(
                pos > s.pos * (1 + 1e-12) + 1e-300
            ):
                s.neg, s.pos = np.maximum(s.neg, neg), np.maximum(s.pos, pos)
                changed = True
        if not changed:
            return True
    return False


def _infinite_report(
    float_model: onnx.ModelProto, hazards: List[str], notes: Optional[List[str]] = None
) -> QuantVerifyReport:
    outs = {o.name: np.array(np.inf) for o in float_model.graph.output}
    return QuantVerifyReport(
        outs, {k: np.array(0.0) for k in outs}, None, [], hazards, notes or []
    )


def verify(
    float_model: onnx.ModelProto,
    quantized_model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    atol: Optional[float] = None,
    range_method: str = "interval",
    clipping: str = "noise",
    max_iter: int = 8,
    breakdown: bool = True,
    max_sites_for_breakdown: int = 24,
    engine: str = "zonotope",
    device: Optional[str] = None,
    precision: Optional[str] = None,
) -> QuantVerifyReport:
    """Certified per-output bound on ``|float_model(x) - quantized_model(x)|`` for ``x`` in a box.

    :param input_ranges: ``{input: (lo, hi)}`` over the *real* graph inputs, merged over the
        float model's own ``onnxsim.range.*`` annotations. Every input needs a finite range; an
        unbounded input gives an infinite bound with a hazard.
    :param atol: optional; only used to set ``report.within`` convenience in the caller -- the
        report itself always carries the bounds (``report.within(atol)`` is the check).
    :param range_method: ``"interval"`` (default) or ``"zonotope"`` for the ranges that decide
        how much *clipping* error each quantizer is charged. Only affects tightness.
    :param clipping: ``"clamp"`` (default) models saturation exactly as ``clamp(x, Rmin, Rmax)``
        plus ``scale/2`` rounding noise; ``"noise"`` charges it as extra independent noise radius
        (looser, kept for comparison).
    :param breakdown: also compute each site's stand-alone contribution (one extra analysis per
        site, skipped above ``max_sites_for_breakdown`` sites).
    :param engine: ``"zonotope"`` (default) carries a dense symbol array through every layer: tight,
        but time and memory grow with the square of the image area (110 s and 7 GB for a 3-layer
        conv net at 32x32). ``"backward"`` bounds the *difference* of the two graphs with a backward
        CROWN pass (:mod:`onnxsim.backward_diff`): cost ~ outputs x graph, so it suits models with
        few outputs; it is somewhat looser where correlation across layers matters. Both are sound.
    :param device: ``None`` / ``"cpu"`` (default), ``"cuda"`` / ``"cuda:N"``, ``"torch-cpu"`` or
        ``"auto"``: where the ``"backward"`` engine's CROWN pass runs (:mod:`onnxsim._device`).
        Only the backward engine has a device backend; ``engine="zonotope"`` raises for anything
        but the default.
    :param precision: ``"float64"`` (default) or ``"float32"`` (sound, error-tracked; needs a
        torch device). Backward engine only.
    """
    del atol  # accepted for API symmetry; use QuantVerifyReport.within(atol)
    if engine not in ("zonotope", "backward"):
        raise ValueError(f"engine must be 'zonotope' or 'backward', got {engine!r}")
    if engine == "zonotope" and (
        device not in (None, "cpu") or precision not in (None, "float64")
    ):
        raise ValueError(
            "device= / precision= apply to engine='backward' only; the zonotope engine "
            "runs on the CPU in float64"
        )
    real_inputs = _model_inputs(float_model)
    if set(real_inputs) != set(_model_inputs(quantized_model)):
        raise ValueError(
            f"graph inputs differ: {sorted(real_inputs)} vs {sorted(_model_inputs(quantized_model))}"
        )
    f_outs = [o.name for o in float_model.graph.output]
    q_outs = [o.name for o in quantized_model.graph.output]
    if len(f_outs) != len(q_outs):
        raise ValueError(f"graph outputs differ: {f_outs} vs {q_outs}")
    ranges: Dict[str, Tuple] = dict(_ranges.get_ranges(float_model))
    ranges.update(input_ranges or {})
    missing = [n for n in real_inputs if n not in ranges]
    if missing:
        return _infinite_report(
            float_model, [f"input(s) {missing} have no range; pass input_ranges"]
        )

    try:
        conv = _Converter(quantized_model, clamp=(clipping == "clamp"))
        converted = conv.run()
    except _Unsupported as e:
        return _infinite_report(float_model, [str(e)])
    sites, hazards, notes = conv.sites, list(conv.hazards), list(conv.notes)

    # rename converted outputs to the float model's names (positional)
    if f_outs != q_outs:
        for fo, qo in zip(f_outs, q_outs):
            if fo != qo:
                converted.graph.node.append(helper.make_node("Identity", [qo], [fo]))
        for i, fo in enumerate(f_outs):
            converted.graph.output[i].name = fo

    # the float reference gets the same graph inputs, unused, so noise symbols are shared
    ref = onnx.ModelProto()
    ref.CopyFrom(float_model)
    for s in sites:
        ref.graph.input.append(
            helper.make_tensor_value_info(
                s.noise_name, TensorProto.FLOAT, list(s.shape)
            )
        )

    # radii: scale/2 always; in "noise" mode plus the clipping term, iterated to a fixed point
    # (monotone: radii only grow). In "clamp" mode saturation is part of the graph, so one range
    # pass only decides which sites can saturate, for the report.
    converged = not sites
    if clipping == "clamp":
        converged = _settle_noise(
            converted, sites, ranges, range_method, max_iter, only_dynamic=True
        )
        if not converged:
            hazards.append(
                f"dynamic-quantization scales did not reach a fixed point in {max_iter} iterations; they are unbounded"
            )
            for s in sites:
                if s.dynamic:
                    s.neg = s.pos = np.full(s.shape, np.inf)
        if sites:
            rng_all = dict(ranges)
            rng_all.update(_noise_ranges(sites, None))
            got = _site_ranges(
                converted, rng_all, [s.input_name for s in sites], range_method
            )
            for s in sites:
                if s.input_name not in got:
                    s.can_clip = True
                    continue
                lo, hi = got[s.input_name]
                s.can_clip = bool(
                    np.any(np.broadcast_to(hi, s.shape) > s.r_max)
                    or np.any(np.broadcast_to(lo, s.shape) < s.r_min)
                )
    else:
        converged = _settle_noise(converted, sites, ranges, range_method, max_iter)
        if not converged:
            hazards.append(
                f"clipping radii did not reach a fixed point in {max_iter} iterations; affected sites are unbounded"
            )
            for s in sites:
                s.neg = s.pos = np.full(s.shape, np.inf)
    reports: List[SiteReport] = []
    for s in sites:
        if s.dynamic:
            clipped = False  # DynamicQuantizeLinear derives its range from the data: nothing saturates
            if not np.all(np.isfinite(s.radius)):
                hazards.append(
                    f"site {s.label}: unbounded input range, so the dynamic scale is unbounded"
                )
        elif clipping == "clamp":
            clipped = s.can_clip
            if clipped:
                hazards.append(
                    f"site {s.label}: the propagated range exceeds the representable range "
                    f"[{float(np.min(s.r_min)):.6g}, {float(np.max(s.r_max)):.6g}], so inputs in the box can saturate; "
                    "saturation is modelled exactly (clamp), not assumed away"
                )
        else:
            clipped = bool(np.any(s.radius > s.half_step * (1 + 1e-9)))
            if clipped and np.all(np.isfinite(s.radius)):
                hazards.append(
                    f"site {s.label}: propagated range exceeds the representable range; clipping error "
                    f"is included in its radius ({float(np.max(s.radius)):.6g} vs scale/2 {float(np.max(s.half_step)):.6g})"
                )
            elif not np.all(np.isfinite(s.radius)):
                hazards.append(
                    f"site {s.label}: unbounded input range, so its clipping error is unbounded"
                )
        half = float(np.max(s.radius)) if s.dynamic else float(np.max(s.half_step))
        reports.append(
            SiteReport(s.label, s.kind, half, float(np.max(s.radius)), clipped, None)
        )
    if any(not np.all(np.isfinite(s.radius)) for s in sites):
        out = _infinite_report(float_model, hazards, notes)
        out.sites = reports
        return out

    def bound(
        active: Optional[int], zero_all: bool = False
    ) -> _zonotope.DifferenceBound:
        rng = dict(ranges)
        if zero_all:
            rng.update(
                {s.noise_name: (np.zeros(s.shape), np.zeros(s.shape)) for s in sites}
            )
        else:
            rng.update(_noise_ranges(sites, active))
        if engine == "backward":
            return _backward_diff.bound_difference(
                ref, converted, rng, device=device, precision=precision
            )
        return _zonotope.bound_difference(ref, converted, rng)

    total = bound(None)
    notes += total.notes
    weights_only = None
    if sites:
        weights_only = bound(None, zero_all=True).worst
        if breakdown and len(sites) <= max_sites_for_breakdown:
            for k, rep in enumerate(reports):
                rep.contribution = bound(k).worst
        elif breakdown:
            notes.append(
                f"per-site breakdown skipped: {len(sites)} sites > {max_sites_for_breakdown}"
            )
    if not total.bounded:
        hazards.append(
            "the certified bound is infinite (an op without an analysis rule, or an unbounded range)"
        )
    return QuantVerifyReport(
        {k: np.asarray(v) for k, v in total.max_abs.items()},
        {k: np.asarray(v) for k, v in total.ref_min_abs.items()},
        weights_only,
        reports,
        hazards,
        notes,
    )


def observed_error(
    float_model: onnx.ModelProto,
    quantized_model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    n: int = 200,
    seed: int = 0,
    adversarial: int = 0,
) -> float:
    """Largest ``|float - quantized|`` seen in onnxruntime for inputs in the box.

    An *empirical lower bound* on the true worst case, for sanity-checking a certified bound:
    a certified bound below this number is a bug. ``n`` uniform random inputs are tried first.
    With ``adversarial > 0`` a corner search follows: random vertices of the box (where the
    worst case of piecewise-linear networks usually sits), then ``adversarial`` hill-climbing
    steps that flip a random subset of input elements to the opposite bound and keep the flip
    whenever the error grows. Uniform samples sit far from the corners, so without this the
    ratio ``certified / observed`` mostly measures how weak the empirical baseline is.
    """
    import onnxruntime as ort

    ranges: Dict[str, Tuple] = dict(_ranges.get_ranges(float_model))
    ranges.update(input_ranges or {})
    rng = np.random.default_rng(seed)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess_f = ort.InferenceSession(
        float_model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    sess_q = ort.InferenceSession(
        quantized_model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    shapes = {
        i.name: [d if isinstance(d, int) else 1 for d in i.shape]
        for i in sess_f.get_inputs()
    }

    def err(feed: Dict[str, np.ndarray]) -> float:
        a = sess_f.run(None, feed)
        b = sess_q.run(None, feed)
        return max(
            float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
            for x, y in zip(a, b)
        )

    worst = 0.0
    for _ in range(n):
        worst = max(
            worst,
            err(
                {
                    k: _ranges.sample(ranges[k], shp, np.float32, rng)
                    for k, shp in shapes.items()
                }
            ),
        )
    if adversarial > 0:
        lohi = {
            k: tuple(
                np.broadcast_to(np.asarray(b, np.float64), shp).astype(np.float32)
                for b in ranges[k]
            )
            for k, shp in shapes.items()
        }
        best_e, best = -1.0, {}
        for _ in range(max(8, n // 10)):  # random vertices
            feed = {
                k: np.where(rng.random(shapes[k]) < 0.5, lohi[k][0], lohi[k][1]).astype(
                    np.float32
                )
                for k in shapes
            }
            e = err(feed)
            if e > best_e:
                best_e, best = e, feed
        for _ in range(adversarial):  # hill climb: flip a random subset of elements
            cand = {}
            for k, v in best.items():
                flip = rng.random(v.shape) < max(0.02, 1.0 / max(1, v.size) * 4)
                cand[k] = np.where(
                    flip, np.where(v == lohi[k][0], lohi[k][1], lohi[k][0]), v
                ).astype(np.float32)
            e = err(cand)
            if e > best_e:
                best_e, best = e, cand
        worst = max(worst, best_e)
    return worst


def verify_against_annotation(
    quantized_model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    range_method: str = "interval",
):
    """Do the model's annotated output ranges (``onnxsim.ranges``) still hold after quantization?

    The quantized model is analysed as the float graph with bounded rounding/saturation noise
    described in the module docstring, and every annotated output range is checked
    with :func:`onnxsim.crown.verify_output_ranges` over the input box and the noise radii.
    Returns ``{output: RangeVerdict}`` (``proved`` means *proven* for every input in the box and
    every rounding outcome; ``False`` means not proved, not violated). Raises ``ValueError`` when
    the model cannot be analysed (an unsupported quantized-domain op, an input without a range).
    """
    from . import crown as _crown

    ranges: Dict[str, Tuple] = dict(_ranges.get_ranges(quantized_model))
    ranges.update(input_ranges or {})
    missing = [n for n in _model_inputs(quantized_model) if n not in ranges]
    if missing:
        raise ValueError(f"input(s) {missing} have no range; pass input_ranges")
    try:
        conv = _Converter(quantized_model, clamp=False)
        converted = conv.run()
    except _Unsupported as e:
        raise ValueError(str(e)) from e
    if not _settle_noise(converted, conv.sites, ranges, range_method, 8):
        raise ValueError(
            "clipping noise did not reach a fixed point (a quantizer input is unbounded)"
        )
    ranges.update(_noise_ranges(conv.sites, None))
    return _crown.verify_output_ranges(converted, ranges)


__all__ = [
    "QuantVerifyReport",
    "SiteReport",
    "observed_error",
    "verify",
    "verify_against_annotation",
]
