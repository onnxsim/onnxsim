"""Backward (CROWN-style) certified bound on ``|orig(x) - converted(x)|`` over an input box.

Why this exists. ``onnxsim.zonotope.bound_difference`` is tight but carries a dense
``(symbols x tensor size)`` generator array through every layer; the symbol count grows with
``C*H*W`` (one per input element, one per rounding site element, one per unstable Relu), so
memory and conv work grow with the *square* of the image area (a 3-layer 8-channel conv net:
1 s at 8x8, 10 s at 16x16, 110 s and 7 GB at 32x32). A *backward* pass costs about
``(#outputs x graph size)`` and never carries thousands of symbols through the layers, which
is why CROWN scales to real networks; a classifier head has 10-1000 outputs.

The construction (every step is exact algebra or a sound enclosure):

1. **The difference network.** For two graphs that share their real inputs, track for every pair
   of aligned tensors ``(a, b)`` the *difference tensor* ``D = a - b`` explicitly:

   * linear layer with constants ``W_A, W_B`` and biases:
     ``D_out = W_B D_in + (W_A - W_B) a_in + (b_A - b_B)``  (exact; the shared part cancels
     structurally, a layer whose constants are identical contributes nothing);
   * a rounding-noise site ``b_out = b_in + e``: ``D_out = D_in - e``;
   * add / sub / shape ops / average pooling: the same op on the ``D`` inputs;
   * a paired nonlinearity ``f`` (Relu, Clip, Sigmoid, Tanh): by the mean value theorem
     ``f(a) - f(b) = m (a - b)`` with ``m`` inside the range ``[m_lo, m_hi]`` of ``f'`` over the
     union of the two pre-activation ranges, so
     ``D_out = m_mid D_pre + e'``, ``|e'| <= (m_hi - m_lo)/2 * max|D_pre|`` -- *not* two
     independent relaxations. Independent relaxation is what makes a plain product graph
     ``orig - converted`` degenerate to interval-style numbers (23.6 / 51.4 on exact rewrites);
   * anything without a rule (an aligned node of another op type): ``D_out`` is a box from the
     two *independent* intervals of the outputs, ``[lo_A - hi_B, hi_A - lo_B]`` (sound, loses
     correlation, noted).

2. **One CROWN pass.** The float model's own graph (it supplies the activations ``a_in`` the
   ``(W_A - W_B)`` terms read) plus the difference chain form one ONNX graph whose outputs are
   the ``D`` of the model outputs. ``crown.bounds`` bounds them backward; coefficients on shared
   tensors accumulate, so cancellation is automatic and the nonlinearities of the float graph
   only ever see the tiny ``(W_A - W_B)`` coefficients.

Soundness, so the numbers are not over-read:

* Alignment (which tensor of ``converted`` corresponds to which of ``orig``) affects *tightness
  only*. Each identity above holds for any two tensors that really are the inputs of nodes with
  the stated op and constants, and the intervals used for slope ranges and ``max|D_pre|`` are
  enclosures of the actual tensors. A wrong pairing makes ``D`` large, never the bound wrong.
* ``max|D_pre|`` for each pair comes from forward interval propagation *of the difference
  chain* (cheap, structure-aware, and for the first layer exact), intersected with what the two
  independent intervals imply.
* Float64 arithmetic with a small relative widening of every ``e'`` radius (``1e-9``); that is not
  directed rounding. The bounds describe the real-number functions; float32 execution can exceed
  them by float32 rounding (the tests budget ``1e-4`` relative).
* Outputs that cannot be aligned get a sound interval-difference bound and a note.

What it is not: it is *looser* than the zonotope engine where correlation through several layers
matters (backward CROWN relaxes the float graph's own Relu once per neuron), and its cost for a
tensor with ``n`` output elements is ``n`` rows, so a model whose output is a large feature map
costs as much as the zonotope. It is meant for models with few outputs.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import crown as _crown
from . import interval as _interval
from . import ranges as _ranges
from . import zonotope as _zonotope
from .zonotope import DifferenceBound

_SLACK = 1e-9  # relative widening of every box radius (float64, not directed rounding)
_Iv = Tuple[np.ndarray, np.ndarray]
_LINEAR = ("Conv", "Gemm", "MatMul")
_STRUCTURAL = (
    "Flatten",
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "AveragePool",
    "GlobalAveragePool",
    "Slice",
)
_PAIRED = ("Relu", "Clip", "Sigmoid", "Tanh")


class _Lost(Exception):
    """Alignment of a node failed (it falls back to a box or is dropped)."""


@dataclasses.dataclass
class _Chain:
    """A difference tensor ``D = a - b``: its graph name and interval (``None`` = exactly 0)."""

    name: str
    lo: np.ndarray
    hi: np.ndarray


def _attr_sig(node: onnx.NodeProto) -> Tuple:
    return tuple(
        sorted(
            (a.name, a.SerializeToString(deterministic=True)) for a in node.attribute
        )
    )


def _consts(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    out = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    runner = _interval._Runner(model)
    for node in model.graph.node:
        ins = [x for x in node.input if x]
        if node.op_type == "Constant" and not node.domain:
            for a in node.attribute:
                if a.name == "value":
                    out[node.output[0]] = numpy_helper.to_array(a.t)
            continue
        if ins and all(x in out for x in ins) and "" not in node.input:
            try:
                res = runner.run(node, [np.asarray(out[x]) for x in ins])
            except Exception:
                continue
            for o, r in zip(node.output, res):
                if o:
                    out[o] = np.asarray(r)
    return out


def _real_inputs(model: onnx.ModelProto) -> List[str]:
    init = {t.name for t in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in init]


def _used_inputs(model: onnx.ModelProto) -> set:
    return {x for n in model.graph.node for x in n.input if x}


def _slopes(
    kind: str,
    node: onnx.NodeProto,
    consts: Dict[str, np.ndarray],
    lo: np.ndarray,
    hi: np.ndarray,
):
    """Elementwise range ``[m_lo, m_hi]`` of ``f'`` over ``[lo, hi]`` (for the mean value theorem)."""
    one, zero = np.ones_like(lo), np.zeros_like(lo)
    if kind == "Relu":
        stable_on, stable_off = lo >= 0, hi <= 0
        return (np.where(stable_on, one, zero), np.where(stable_off, zero, one))
    if kind == "Clip":
        lo_c = (
            consts.get(node.input[1]) if len(node.input) > 1 and node.input[1] else None
        )
        hi_c = (
            consts.get(node.input[2]) if len(node.input) > 2 and node.input[2] else None
        )
        mn = -np.inf if lo_c is None else float(np.asarray(lo_c).reshape(-1)[0])
        mx = np.inf if hi_c is None else float(np.asarray(hi_c).reshape(-1)[0])
        inside = (lo >= mn) & (hi <= mx)
        outside = (hi <= mn) | (lo >= mx)
        return (np.where(inside, one, zero), np.where(outside, zero, one))

    def d(x: np.ndarray) -> np.ndarray:
        if kind == "Sigmoid":
            s = 1.0 / (1.0 + np.exp(-x))
            return s * (1.0 - s)
        return 1.0 - np.tanh(x) ** 2

    peak = d(np.zeros(()))
    dl, dh = d(lo), d(hi)
    d_max = np.where((lo < 0) & (hi > 0), peak, np.maximum(dl, dh))
    return np.minimum(dl, dh), d_max


class _Builder:
    def __init__(
        self,
        orig: onnx.ModelProto,
        conv: onnx.ModelProto,
        ranges: Dict[str, Tuple],
        notes: List[str],
    ) -> None:
        self.A, self.B, self.ranges, self.notes = orig, conv, ranges, notes
        self.ia = _interval.propagate(orig, ranges)
        self.ib = _interval.propagate(conv, ranges)
        self.ca, self.cb = _consts(orig), _consts(conv)
        self.runner = _interval._Runner(orig)
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self.extra_inputs: Dict[
            str, Tuple[Tuple[int, ...], np.ndarray, np.ndarray]
        ] = {}
        self.refs: set = set()  # tensors of A the chain reads
        self.amap: Dict[str, str] = {}
        self.delta: Dict[str, Optional[_Chain]] = {}
        self.lost: List[str] = []
        self.k = 0
        used_a = _used_inputs(orig)
        self.real = [n for n in _real_inputs(orig) if n in used_a]
        self.noise = {
            n for n in _real_inputs(conv) if n not in used_a and n in _used_inputs(conv)
        }
        for n in self.real:
            self.amap[n] = n
            self.delta[n] = None
        # Identity nodes of the original are transparent: ``y = Identity(a2)`` is the tensor ``a2``
        self.alias: Dict[str, str] = {}
        for node in orig.graph.node:
            if (
                node.op_type == "Identity"
                and len(node.input) == 1
                and len(node.output) == 1
                and node.input[0] not in self.ca
            ):
                self.alias[node.output[0]] = node.input[0]
        self.virtual: set = set()
        self.a_index: Dict[Tuple, List[onnx.NodeProto]] = {}
        for node in orig.graph.node:
            key = self._key(
                node.op_type,
                [self.ca.get(i) is not None for i in node.input],
                [self.canon(i) for i in node.input],
            )
            self.a_index.setdefault(key, []).append(node)
        self.b_consumers: Dict[str, List[str]] = {}
        for node in conv.graph.node:
            for x in node.input:
                if x:
                    self.b_consumers.setdefault(x, []).append(node.op_type)
        self._fuse(orig)

    def canon(self, name: str) -> str:
        while name in self.alias:
            name = self.alias[name]
        return name

    # -- virtual fused nodes of the original ---------------------------------------
    # ``simplify`` fuses nodes (BatchNorm into the Conv before it, MatMul + Add into a Gemm), so
    # the converted graph has one node where the original has two. The original graph is left
    # alone; for *matching* we describe the pair as one virtual node whose constants are computed
    # in float64 (an exact identity over the reals: ``W * s`` and ``(b - mu) * s + beta``).
    def _fuse(self, orig: onnx.ModelProto) -> None:
        consumers: Dict[str, List[onnx.NodeProto]] = {}
        producer: Dict[str, onnx.NodeProto] = {}
        for n in orig.graph.node:
            for x in n.input:
                if x:
                    consumers.setdefault(x, []).append(n)
            for o in n.output:
                if o:
                    producer[o] = n
        for n in orig.graph.node:
            first = producer.get(n.input[0]) if n.input else None
            if first is None or len(consumers.get(first.output[0], [])) != 1:
                continue
            fused: Optional[onnx.NodeProto] = None
            if n.op_type == "BatchNormalization" and first.op_type in _LINEAR:
                fused = self._fuse_bn(first, n)
            elif n.op_type == "Add" and first.op_type == "MatMul":
                fused = self._fuse_matmul_add(first, n)
            if fused is None:
                continue
            key = self._key(
                fused.op_type,
                [self.ca.get(i) is not None for i in fused.input],
                [self.canon(i) for i in fused.input],
            )
            self.a_index.setdefault(key, []).append(fused)
            self.virtual.add(id(fused))

    def _virtual(
        self, op: str, ins: List[str], out: str, attrs: Sequence[onnx.AttributeProto]
    ) -> onnx.NodeProto:
        node = helper.make_node(op, ins, [out])
        node.attribute.extend(attrs)
        return node

    def _fuse_bn(
        self, lin: onnx.NodeProto, bn: onnx.NodeProto
    ) -> Optional[onnx.NodeProto]:
        w = self.ca.get(lin.input[1]) if len(lin.input) > 1 else None
        params = [self.ca.get(x) for x in bn.input[1:5]]
        if (
            w is None
            or any(p is None for p in params)
            or _interval._attrs(bn).get("training_mode", 0)
        ):
            return None
        scale, beta, mean, var = (np.asarray(p, dtype=np.float64) for p in params)  # type: ignore[arg-type]
        s = scale / np.sqrt(var + float(_interval._attrs(bn).get("epsilon", 1e-5)))
        w = np.asarray(w, dtype=np.float64)
        a = _interval._attrs(lin)
        if lin.op_type == "Conv":
            wf = w * s.reshape([-1] + [1] * (w.ndim - 1))
            nout = w.shape[0]
        elif lin.op_type == "Gemm":
            if w.ndim != 2:
                return None
            wf = w * (s.reshape(-1, 1) if a.get("transB", 0) else s.reshape(1, -1))
            nout = w.shape[0] if a.get("transB", 0) else w.shape[1]
        else:  # MatMul
            if w.ndim != 2:
                return None
            wf, nout = w * s.reshape(1, -1), w.shape[1]
        b = np.zeros(nout)
        if lin.op_type == "Conv" and len(lin.input) > 2 and lin.input[2]:
            b = np.asarray(self.ca.get(lin.input[2], b), dtype=np.float64)
        if lin.op_type == "Gemm" and len(lin.input) > 2 and lin.input[2]:
            b = np.broadcast_to(
                np.asarray(self.ca.get(lin.input[2], b), dtype=np.float64)
                * float(a.get("beta", 1.0)),
                (nout,),
            )
        bf = (b - mean) * s + beta
        wn, bn_ = f"{bn.output[0]}__fw", f"{bn.output[0]}__fb"
        self.ca[wn], self.ca[bn_] = wf, bf
        ins = [lin.input[0], wn, bn_]
        attrs = [x for x in lin.attribute if x.name != "beta"]
        op = lin.op_type if lin.op_type != "MatMul" else "Gemm"
        if lin.op_type == "Gemm":
            attrs.append(helper.make_attribute("beta", 1.0))
        return self._virtual(op, ins, bn.output[0], attrs)

    def _fuse_matmul_add(
        self, mm: onnx.NodeProto, add: onnx.NodeProto
    ) -> Optional[onnx.NodeProto]:
        w = self.ca.get(mm.input[1])
        other = add.input[1] if add.input[0] == mm.output[0] else add.input[0]
        c = self.ca.get(other)
        if w is None or c is None or np.ndim(w) != 2 or np.ndim(c) > 1:
            return None
        return self._virtual(
            "Gemm", [mm.input[0], mm.input[1], other], add.output[0], []
        )

    # -- plumbing ---------------------------------------------------------------
    @staticmethod
    def _key(op: str, is_const: List[bool], names: List[str]) -> Tuple:
        return (op, tuple(None if c else n for c, n in zip(is_const, names)))

    def _name(self, tag: str) -> str:
        self.k += 1
        return f"bd{self.k}_{tag}"

    def _init(self, arr: np.ndarray, tag: str) -> str:
        name = self._name(tag)
        self.inits.append(
            numpy_helper.from_array(np.asarray(arr, dtype=np.float64), name)
        )
        return name

    def _input(
        self, shape: Tuple[int, ...], lo: np.ndarray, hi: np.ndarray, tag: str
    ) -> str:
        name = self._name(tag)
        self.extra_inputs[name] = (
            shape,
            np.broadcast_to(lo, shape).copy(),
            np.broadcast_to(hi, shape).copy(),
        )
        return name

    def _node(
        self,
        op: str,
        ins: List[str],
        tag: str,
        attrs: Optional[Sequence[onnx.AttributeProto]] = None,
        **kw: Any,
    ) -> str:
        out = self._name(tag)
        n = helper.make_node(op, ins, [out], **kw)
        if attrs:
            n.attribute.extend(attrs)
        self.nodes.append(n)
        return out

    @staticmethod
    def _widen(lo: np.ndarray, hi: np.ndarray) -> _Iv:
        pad = _SLACK * (np.abs(lo) + np.abs(hi)) + 1e-300
        return lo - pad, hi + pad

    def _iv(self, model_iv: "_interval.IntervalResult", name: str) -> Optional[_Iv]:
        v = model_iv.intervals.get(name)
        if v is None:
            return None
        return np.asarray(v[0], dtype=np.float64), np.asarray(v[1], dtype=np.float64)

    # -- interval arithmetic on the difference chain ------------------------------
    @staticmethod
    def _fwd(node: onnx.NodeProto, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        a = _interval._attrs(node)
        if node.op_type == "Conv":
            nd = x.ndim - 2
            if nd != 2:
                raise _Lost("Conv with non-2D spatial dims")
            return _zonotope._conv2d(
                x,
                w,
                list(a.get("strides", [1, 1])),
                list(a.get("pads", [0, 0, 0, 0])),
                list(a.get("dilations", [1, 1])),
                int(a.get("group", 1)),
            )
        if node.op_type == "Gemm":
            xa = x.T if a.get("transA", 0) else x
            wb = w.T if a.get("transB", 0) else w
            return float(a.get("alpha", 1.0)) * (xa @ wb)
        return np.matmul(x, w)

    def _lin_iv(self, node: onnx.NodeProto, iv: _Iv, w: np.ndarray) -> _Iv:
        lo, hi = iv
        c, r = (lo + hi) / 2.0, (hi - lo) / 2.0
        oc = self._fwd(node, c, w)
        orad = np.abs(self._fwd(node, r, np.abs(w)))
        return self._widen(oc - orad, oc + orad)

    @staticmethod
    def _sum(a: _Iv, b: _Iv) -> _Iv:
        return a[0] + b[0], a[1] + b[1]

    # -- chain emission ------------------------------------------------------------
    def _add_terms(self, terms: List[Tuple[str, _Iv]]) -> Optional[_Chain]:
        if not terms:
            return None
        name, iv = terms[0]
        for n2, iv2 in terms[1:]:
            name = self._node("Add", [name, n2], "sum")
            iv = self._sum(iv, iv2)
        return _Chain(name, iv[0], iv[1])

    def _linear(
        self, nb: onnx.NodeProto, na: onnx.NodeProto, ins_b: List[str]
    ) -> Optional[_Chain]:
        op = nb.op_type
        xb = nb.input[0]
        wb = self.cb.get(nb.input[1])
        wa = self.ca.get(na.input[1])
        if wb is None or wa is None or wb.shape != wa.shape:
            raise _Lost(f"{op} weights are not matching constants")
        d_in = self.delta.get(xb)
        a_in = self.amap[xb]
        terms: List[Tuple[str, _Iv]] = []
        attrs = list(na.attribute)
        if op == "Gemm":
            attrs = [a for a in attrs if a.name not in ("beta",)]
        if d_in is not None:
            wn = self._init(wb, "W")
            out = self._node(op, [d_in.name, wn], "lin_d", attrs)
            terms.append(
                (out, self._lin_iv(na, (d_in.lo, d_in.hi), wb.astype(np.float64)))
            )
        dw = wa.astype(np.float64) - wb.astype(np.float64)
        if np.any(dw != 0):
            ai = self._iv(self.ia, a_in)
            if ai is None:
                raise _Lost(f"no interval for {a_in!r}")
            self.refs.add(a_in)
            wn = self._init(dw, "dW")
            out = self._node(op, [a_in, wn], "lin_w", attrs)
            terms.append((out, self._lin_iv(na, ai, dw)))
        # bias difference
        db: Optional[np.ndarray] = None
        if op == "Conv":
            ba = self.ca.get(na.input[2]) if len(na.input) > 2 and na.input[2] else None
            bb = self.cb.get(nb.input[2]) if len(nb.input) > 2 and nb.input[2] else None
            if ba is not None or bb is not None:
                z = np.zeros(wa.shape[0])
                db = (z if ba is None else ba.astype(np.float64)) - (
                    z if bb is None else bb.astype(np.float64)
                )
                db = db.reshape([1, -1] + [1] * (wa.ndim - 2))
        elif op == "Gemm":
            ca_ = (
                self.ca.get(na.input[2]) if len(na.input) > 2 and na.input[2] else None
            )
            cb_ = (
                self.cb.get(nb.input[2]) if len(nb.input) > 2 and nb.input[2] else None
            )
            if ca_ is not None or cb_ is not None:
                shape = np.broadcast_shapes(
                    *(np.shape(c) for c in (ca_, cb_) if c is not None)
                )
                z = np.zeros(shape)
                db = float(_interval._attrs(na).get("beta", 1.0)) * (
                    (z if ca_ is None else ca_.astype(np.float64))
                    - (z if cb_ is None else cb_.astype(np.float64))
                )
        if db is not None and np.any(db != 0):
            cn = self._init(db, "db")
            if terms:
                name, iv = terms[0]
                for n2, iv2 in terms[1:]:
                    name = self._node("Add", [name, n2], "sum")
                    iv = self._sum(iv, iv2)
                shifted = self._node("Add", [name, cn], "bias")
                d = np.broadcast_to(db, np.broadcast(iv[0], db).shape)
                return _Chain(shifted, iv[0] + d, iv[1] + d)
            return self._const_chain(db, na.output[0])
        return self._add_terms(terms)

    def _const_chain(self, arr: np.ndarray, a_out: str) -> _Chain:
        """A difference that is a constant tensor (nothing varies): broadcast to the output shape."""
        iv = self._iv(self.ia, a_out)
        if iv is None:
            raise _Lost(f"no shape for {a_out!r}")
        full = np.broadcast_to(arr, iv[0].shape).astype(np.float64)
        return _Chain(self._init(full, "const"), full.copy(), full.copy())

    def _paired(self, nb: onnx.NodeProto, na: onnx.NodeProto) -> Optional[_Chain]:
        op = nb.op_type
        xb = nb.input[0]
        d_pre = self.delta.get(xb)
        a_pre = self.amap[xb]
        if d_pre is None:
            return None  # identical inputs, identical nonlinearity: exactly equal
        ia_, ib_ = self._iv(self.ia, a_pre), self._iv(self.ib, xb)
        if ia_ is None or ib_ is None:
            raise _Lost("no interval for a pre-activation")
        la, ua = ia_
        lb, ub = ib_
        # each pre-activation is also confined by the other one and the difference interval
        la2, ua2 = np.maximum(la, lb + d_pre.lo), np.minimum(ua, ub + d_pre.hi)
        lb2, ub2 = np.maximum(lb, la - d_pre.hi), np.minimum(ub, ua - d_pre.lo)
        if np.any(la2 > ua2) or np.any(lb2 > ub2):
            la2, ua2, lb2, ub2 = la, ua, lb, ub
        lo_u, hi_u = np.minimum(la2, lb2), np.maximum(ua2, ub2)
        m_lo, m_hi = _slopes(op, na, self.ca, lo_u, hi_u)
        mid, w = (m_lo + m_hi) / 2.0, (m_hi - m_lo) / 2.0
        dmax = np.maximum(np.abs(d_pre.lo), np.abs(d_pre.hi))
        rad = w * dmax * (1.0 + _SLACK) + 1e-300 * (w > 0)
        mn = self._init(mid, "m")
        scaled = self._node("Mul", [d_pre.name, mn], "pair_m")
        mlo, mhi = mid * d_pre.lo, mid * d_pre.hi
        lo, hi = np.minimum(mlo, mhi), np.maximum(mlo, mhi)
        if np.any(rad > 0):
            en = self._input(tuple(rad.shape), -rad, rad, "pair_e")
            scaled = self._node("Add", [scaled, en], "pair_x")
            lo, hi = lo - rad, hi + rad
        return _Chain(scaled, *self._widen(lo, hi))

    def _box(self, a_out: str, b_out: str, why: str) -> Optional[_Chain]:
        ia_, ib_ = self._iv(self.ia, a_out), self._iv(self.ib, b_out)
        if ia_ is None or ib_ is None:
            raise _Lost("no interval for the box fallback")
        lo, hi = ia_[0] - ib_[1], ia_[1] - ib_[0]
        self._note(f"precision lost at {why}: independent interval box")
        name = self._input(tuple(lo.shape), lo, hi, "box")
        return _Chain(name, lo, hi)

    def _note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)

    # -- main loop ---------------------------------------------------------------
    def run(self) -> None:
        for nb in self.B.graph.node:
            outs = [o for o in nb.output if o]
            ins = [x for x in nb.input if x]
            if nb.domain not in ("", "ai.onnx") or len(outs) != 1:
                self.lost.append(f"{nb.op_type} {outs[0] if outs else ''}".strip())
                continue
            if nb.op_type == "Constant" and not nb.domain:
                continue  # consumers read it from self.cb; it must never get a difference of its own
            if ins and all(x in self.cb for x in ins) and "" not in nb.input:
                continue  # a constant: folded into self.cb already
            try:
                self._node_b(nb)
            except _Lost as e:
                self.lost.append(f"{nb.op_type} {outs[0]}: {e}")

    def _node_b(self, nb: onnx.NodeProto) -> None:
        out_b = nb.output[0]
        ins = list(nb.input)
        noise_ins = [x for x in ins if x in self.noise]
        if nb.op_type == "Add" and len(noise_ins) == 1:
            other = ins[0] if ins[1] == noise_ins[0] else ins[1]
            if other not in self.amap:
                raise _Lost("noise added to an unaligned tensor")
            e = noise_ins[0]
            d = self.delta[other]
            lo_hi = self.ranges.get(e)
            if lo_hi is None:
                raise _Lost("noise input without a range")
            elo = np.asarray(lo_hi[0], dtype=np.float64)
            ehi = np.asarray(lo_hi[1], dtype=np.float64)
            shape = np.broadcast_shapes(np.shape(elo), np.shape(ehi))
            if d is None:
                name = self._node("Neg", [e], "noise")
                lo, hi = -np.broadcast_to(ehi, shape), -np.broadcast_to(elo, shape)
            else:
                name = self._node("Sub", [d.name, e], "noise")
                lo, hi = d.lo - ehi, d.hi - elo
            self.amap[out_b] = self.amap[other]
            self.delta[out_b] = _Chain(name, lo, hi)
            return
        if nb.op_type == "Identity":
            if ins[0] not in self.amap:
                raise _Lost("Identity of an unaligned tensor")
            self.amap[out_b] = self.amap[ins[0]]
            self.delta[out_b] = self.delta[ins[0]]
            return
        is_const = [x in self.cb for x in ins]
        if any((not c) and (x not in self.amap) for c, x in zip(is_const, ins) if x):
            raise _Lost("input is not aligned")
        key_names = [self.amap.get(x, x) if x else x for x in ins]
        key = self._key(nb.op_type, is_const, key_names)
        na = self._pick(self.a_index.get(key, []), nb)
        if na is None:
            raise _Lost("no matching node in the original graph")
        out_a = na.output[0]
        chain = self._rule(nb, na, is_const)
        if chain is not None:
            ia_, ib_ = self._iv(self.ia, out_a), self._iv(self.ib, out_b)
            if ia_ is not None and ib_ is not None and ia_[0].shape == chain.lo.shape:
                chain.lo = np.maximum(chain.lo, ia_[0] - ib_[1])
                chain.hi = np.minimum(chain.hi, ia_[1] - ib_[0])
                chain.hi = np.maximum(chain.hi, chain.lo)  # never an inverted interval
        self.amap[out_b] = out_a
        self.delta[out_b] = chain

    def _pick(
        self, cands: List[onnx.NodeProto], nb: onnx.NodeProto
    ) -> Optional[onnx.NodeProto]:
        """The original node ``nb`` corresponds to.

        A real node and a virtual fused node can both fit (``Conv`` vs ``Conv+BatchNorm``). The
        converted graph still has the BatchNorm / Add after ``nb`` if it kept the original's
        structure; if it has not, ``nb`` is the fused one.
        """
        real = [c for c in cands if id(c) not in self.virtual]
        fused = [c for c in cands if id(c) in self.virtual]
        follows = self.b_consumers.get(nb.output[0], [])
        keeps_split = any(op in ("BatchNormalization", "Add") for op in follows)
        order = real + fused if keeps_split or not fused else fused + real
        for c in order:
            if self._same_attrs(c, nb):
                return c
        return None

    def _same_attrs(self, na: onnx.NodeProto, nb: onnx.NodeProto) -> bool:
        if id(na) not in self.virtual:
            return _attr_sig(na) == _attr_sig(nb)
        defaults = (
            {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}
            if nb.op_type == "Gemm"
            else {}
        )
        da, db = _interval._attrs(na), _interval._attrs(nb)
        return all(
            da.get(k, defaults.get(k)) == db.get(k, defaults.get(k))
            for k in set(da) | set(db)
        )

    def _rule(
        self, nb: onnx.NodeProto, na: onnx.NodeProto, is_const: List[bool]
    ) -> Optional[_Chain]:
        op = nb.op_type
        ins = list(nb.input)
        out_b, out_a = nb.output[0], na.output[0]
        # identical op on identical tensors with identical constants: exactly equal, whatever the op
        if all(self.delta.get(x) is None for x, c in zip(ins, is_const) if x and not c):
            if all(
                self.ca.get(ia) is not None and np.array_equal(self.ca[ia], self.cb[x])
                for x, ia, c in zip(ins, na.input, is_const)
                if x and c
            ):
                return None
        try:
            if op in _LINEAR and len(ins) >= 2 and is_const[1] and not is_const[0]:
                return self._linear(nb, na, ins)
            if op in _PAIRED and not is_const[0]:
                return self._paired(nb, na)
            if op in ("Add", "Sub") and len(ins) == 2:
                return self._addsub(nb, na, is_const)
            if op == "Mul" and len(ins) == 2 and is_const.count(True) == 1:
                return self._mul(nb, na, is_const)
            if op == "MaxPool" and not is_const[0]:
                # |maxpool(a) - maxpool(b)| <= window-max of |a - b|: a box on the output difference
                d = self.delta[ins[0]]
                if d is None:
                    return None
                dmax = np.maximum(np.abs(d.lo), np.abs(d.hi))
                m = np.asarray(self.runner.run(na, [dmax])[0], dtype=np.float64)
                name = self._input(tuple(m.shape), -m, m, "pool")
                return _Chain(name, -m, m)
            if op in _STRUCTURAL and not is_const[0]:
                d = self.delta[ins[0]]
                if d is None:
                    return None
                extra = [x for x in ins[1:] if x]
                if any(x not in self.cb for x in extra):
                    raise _Lost("shape operand is not constant")
                arrs = [np.asarray(self.cb[x]) for x in extra]
                lo = self.runner.run(na, [d.lo] + arrs)[0]
                hi = self.runner.run(na, [d.hi] + arrs)[0]
                names = [d.name] + [self._shape_const(a) for a in arrs]
                return _Chain(self._node(op, names, "str", list(na.attribute)), lo, hi)
        except _Lost:
            raise
        return self._box(out_a, out_b, op)

    def _shape_const(self, arr: np.ndarray) -> str:
        name = self._name("shape")
        self.inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def _addsub(
        self, nb: onnx.NodeProto, na: onnx.NodeProto, is_const: List[bool]
    ) -> Optional[_Chain]:
        op = nb.op_type
        x0, x1 = nb.input
        if not is_const[0] and not is_const[1]:
            d0, d1 = self.delta[x0], self.delta[x1]
            if d0 is None and d1 is None:
                return None
            if d1 is None:
                return d0
            if d0 is None:
                if op == "Add":
                    return d1
                name = self._node("Neg", [d1.name], "neg")
                return _Chain(name, -d1.hi, -d1.lo)
            name = self._node(op, [d0.name, d1.name], op.lower())
            if op == "Add":
                return _Chain(name, d0.lo + d1.lo, d0.hi + d1.hi)
            return _Chain(name, d0.lo - d1.hi, d0.hi - d1.lo)
        if is_const[0] and is_const[1]:
            raise _Lost("constant node")
        t, c = (0, 1) if is_const[1] else (1, 0)
        xt, xc = nb.input[t], nb.input[c]
        ac = self.ca.get(na.input[c])
        bc = self.cb.get(xc)
        if ac is None or bc is None:
            raise _Lost("constant operand missing in one model")
        d = self.delta[xt]
        dc = ac.astype(np.float64) - bc.astype(np.float64)
        # Add(x, c): a - b = D_x + (c_A - c_B); Sub(x, c): D_x - (c_A - c_B); Sub(c, x): (c_A - c_B) - D_x
        if t == 0:
            shift = dc if op == "Add" else -dc
            dsign = 1.0
        else:
            shift = dc
            dsign = 1.0 if op == "Add" else -1.0
        has_shift = bool(np.any(shift != 0))
        if d is None and not has_shift:
            return None
        if d is None:
            return self._const_chain(shift, na.output[0])
        name, lo, hi = d.name, d.lo, d.hi
        if dsign < 0:
            name = self._node("Neg", [name], "neg")
            lo, hi = -d.hi, -d.lo
        if has_shift:
            cn = self._init(shift, "c")
            name = self._node("Add", [name, cn], "shift")
            lo, hi = lo + shift, hi + shift
        return _Chain(name, lo, hi)

    def _mul(
        self, nb: onnx.NodeProto, na: onnx.NodeProto, is_const: List[bool]
    ) -> Optional[_Chain]:
        t, c = (0, 1) if is_const[1] else (1, 0)
        xt = nb.input[t]
        cb_ = self.cb.get(nb.input[c])
        ca_ = self.ca.get(na.input[c])
        if cb_ is None or ca_ is None:
            raise _Lost("Mul constant missing in one model")
        d = self.delta[xt]
        terms: List[Tuple[str, _Iv]] = []
        if d is not None:
            cn = self._init(cb_.astype(np.float64), "c")
            name = self._node("Mul", [d.name, cn], "mul_d")
            p, q = d.lo * cb_, d.hi * cb_
            terms.append((name, (np.minimum(p, q), np.maximum(p, q))))
        dc = ca_.astype(np.float64) - cb_.astype(np.float64)
        if np.any(dc != 0):
            a_in = self.amap[xt]
            ai = self._iv(self.ia, a_in)
            if ai is None:
                raise _Lost(f"no interval for {a_in!r}")
            self.refs.add(a_in)
            dn = self._init(dc, "dc")
            name = self._node("Mul", [a_in, dn], "mul_c")
            p, q = ai[0] * dc, ai[1] * dc
            terms.append((name, (np.minimum(p, q), np.maximum(p, q))))
        return self._add_terms(terms)


def _model_graph(
    b: _Builder, orig: onnx.ModelProto, out_names: List[str], chains: List[_Chain]
) -> Tuple[onnx.ModelProto, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """The float graph (only what the chain reads) plus the chain, as one float64 ONNX model."""
    g = orig.graph
    prod: Dict[str, onnx.NodeProto] = {}
    for n in g.node:
        for o in n.output:
            if o:
                prod[o] = n
    need: List[onnx.NodeProto] = []
    seen: set = set()
    consts_needed: set = set()

    def visit(t: str) -> None:
        if t in seen:
            return
        seen.add(t)
        n = prod.get(t)
        if n is None:
            consts_needed.add(t)
            return
        for x in n.input:
            if x:
                visit(x)
        need.append(n)

    for t in sorted(b.refs):
        visit(t)
    inits = {t.name: t for t in g.initializer}
    a_inputs = {i.name: i for i in g.input}
    new_inits: List[onnx.TensorProto] = []
    for name in sorted(
        consts_needed | {x for n in need for x in n.input if x and x in inits}
    ):
        if name in inits:
            arr = numpy_helper.to_array(inits[name])
            if arr.dtype.kind == "f":  # shape/axes constants stay integers
                arr = arr.astype(np.float64)
            new_inits.append(numpy_helper.from_array(arr, name))
    inputs: List[onnx.ValueInfoProto] = []
    for name in sorted(consts_needed):
        if name in a_inputs and name not in inits:
            vi = a_inputs[name]
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            inputs.append(
                helper.make_tensor_value_info(name, TensorProto.DOUBLE, shape)
            )
    ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, (eshape, lo, hi) in b.extra_inputs.items():
        inputs.append(
            helper.make_tensor_value_info(name, TensorProto.DOUBLE, list(eshape))
        )
        ranges[name] = (lo, hi)
    # real inputs and the noise inputs the chain reads
    used_chain = {x for n in b.nodes for x in n.input}
    for name in list(a_inputs):
        if (
            (name in used_chain or name in consts_needed)
            and name not in inits
            and name not in {i.name for i in inputs}
        ):
            vi = a_inputs[name]
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            inputs.append(
                helper.make_tensor_value_info(name, TensorProto.DOUBLE, shape)
            )
    for name in sorted(b.noise):
        if name in used_chain and name not in {i.name for i in inputs}:
            shape = list(
                np.broadcast_shapes(
                    np.shape(b.ranges[name][0]), np.shape(b.ranges[name][1])
                )
            )
            inputs.append(
                helper.make_tensor_value_info(name, TensorProto.DOUBLE, shape)
            )
    outputs = [
        helper.make_tensor_value_info(c.name, TensorProto.DOUBLE, list(c.lo.shape))
        for c in chains
    ]
    graph = helper.make_graph(
        need + b.nodes,
        "backward_diff",
        inputs,
        outputs,
        new_inits + b.inits,
    )
    model = helper.make_model(graph, opset_imports=list(orig.opset_import))
    model.ir_version = orig.ir_version
    return model, ranges


def _interval_difference(
    ia: _interval.IntervalResult, ib: _interval.IntervalResult, name: str
) -> Optional[np.ndarray]:
    a, b = ia.intervals.get(name), ib.intervals.get(name)
    if a is None or b is None:
        return None
    return np.maximum(np.abs(a[0] - b[1]), np.abs(a[1] - b[0]))


def bound_difference(
    orig: onnx.ModelProto,
    converted: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    outputs: Optional[Union[str, Sequence[str]]] = None,
    method: str = "crown",
    refine: Union[bool, str] = False,
    max_refine_elements: int = 4096,
) -> DifferenceBound:
    """Certified per-output bound on ``|orig(x) - converted(x)|`` by backward propagation.

    Same contract and result type as :func:`onnxsim.zonotope.bound_difference` (``max_abs``,
    ``ref_min_abs``, ``notes``), so it can stand in for it. ``orig`` and ``converted`` must share
    their real inputs and output names; extra inputs of either model (the rounding noise inputs
    ``quant_verify`` adds) need ranges in ``input_ranges``, like every real input.

    :param outputs: restrict to these output names (default: all). The cost grows with the number
        of output *elements*; pass the logits, not a feature map.
    :param method: ``crown`` (default), ``ibp``, or ``alpha`` (needs torch) for the final pass.
    :param refine: tighten intermediate boxes with CROWN first (``True``), never (``False``, the
        default), or ``"auto"`` (only when no tensor of the analysed graph has more than
        ``max_refine_elements``). Default off because it bought almost nothing here: on 3-, 5- and
        8-layer conv classifiers it changed the certified bound by under 1.1% (5.703 vs 5.761,
        239.3 vs 240.6, 3.643e4 vs 3.645e4) and cost 50-80x the time (4.4 s vs 0.08 s, 28.6 s vs 0.37 s).
    """
    in_a = set(_real_inputs(orig))
    names = [o.name for o in orig.graph.output]
    if names != [o.name for o in converted.graph.output]:
        raise ValueError("graph outputs differ")
    if isinstance(outputs, str):
        outputs = [outputs]
    names = list(outputs) if outputs is not None else names
    ranges: Dict[str, Tuple] = dict(_ranges.get_ranges(orig))
    ranges.update(input_ranges or {})
    notes: List[str] = []
    inf = {o: np.array(np.inf) for o in names}
    zero = {o: np.array(0.0) for o in names}
    missing = [
        n for n in _real_inputs(orig) if n not in ranges and n in _used_inputs(orig)
    ]
    if missing:
        return DifferenceBound(inf, zero, [f"unbounded input: {missing} have no range"])
    for n in in_a & set(_real_inputs(converted)):
        lo, hi = ranges[n]
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            return DifferenceBound(inf, zero, [f"unbounded input: {n}"])
    try:
        b = _Builder(orig, converted, ranges, notes)
        b.run()
    except (
        Exception
    ) as e:  # nothing could be analysed: fall back to plain intervals below
        notes.append(
            f"backward engine failed ({type(e).__name__}: {e}); interval difference used"
        )
        ia, ib = (
            _interval.propagate(orig, ranges),
            _interval.propagate(converted, ranges),
        )
        fb_abs: Dict[str, np.ndarray] = {}
        fb_ref: Dict[str, np.ndarray] = {}
        for o in names:
            d = _interval_difference(ia, ib, o)
            fb_abs[o] = np.array(np.inf) if d is None else d
            iv = ib.intervals.get(o)
            fb_ref[o] = (
                np.array(0.0)
                if iv is None
                else np.maximum(0.0, np.maximum(iv[0], -iv[1]))
            )
        return DifferenceBound(fb_abs, fb_ref, notes)
    for item in b.lost:
        notes.append(f"alignment lost at {item}")

    ia, ib = b.ia, b.ib
    max_abs: Dict[str, np.ndarray] = {}
    ref_min: Dict[str, np.ndarray] = {}
    chains: List[_Chain] = []
    aligned: List[str] = []
    for o in names:
        iv = ib.intervals.get(o)
        ref_min[o] = (
            np.array(0.0) if iv is None else np.maximum(0.0, np.maximum(iv[0], -iv[1]))
        )
        ch = b.delta.get(o)
        if o in b.amap and b.amap[o] == b.canon(o) and o in b.delta:
            if ch is None:
                max_abs[o] = (
                    np.zeros_like(ref_min[o]) if np.ndim(ref_min[o]) else np.array(0.0)
                )
                continue
            if not (np.all(np.isfinite(ch.lo)) and np.all(np.isfinite(ch.hi))):
                max_abs[o] = np.full(ch.lo.shape, np.inf)
                continue
            chains.append(ch)
            aligned.append(o)
            continue
        d = _interval_difference(ia, ib, o)
        max_abs[o] = np.array(np.inf) if d is None else d
        notes.append(f"output {o}: not aligned, independent interval difference used")
    if chains:
        model, extra = _model_graph(b, orig, aligned, chains)
        in_ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        graph_inputs = {i.name for i in model.graph.input}
        for n, r in ranges.items():
            if n in graph_inputs:
                in_ranges[n] = (
                    np.asarray(r[0], dtype=np.float64),
                    np.asarray(r[1], dtype=np.float64),
                )
        in_ranges.update(extra)
        do_refine = refine is True
        if refine == "auto":
            els = max((int(np.size(v[0])) for v in b.ia.intervals.values()), default=0)
            do_refine = els <= max_refine_elements
        res = _crown.bounds(
            model,
            in_ranges,
            output=[c.name for c in chains],
            method=method,
            refine=do_refine,
        )
        for o, ch in zip(aligned, chains):
            tb = res[ch.name]
            lo = np.maximum(
                tb.lo, ch.lo
            )  # never looser than the chain's own forward interval
            hi = np.minimum(tb.hi, ch.hi)
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            max_abs[o] = np.maximum(np.abs(lo), np.abs(hi))
    return DifferenceBound(max_abs, ref_min, notes)
