"""Zonotope (affine-arithmetic) analysis of ONNX models, and certified bounds on
``original(x) - simplified(x)``.

Why this exists. ``onnxsim.interval`` and CROWN-style bound propagation treat the two
branches of a product graph ``orig(x) - simplified(x)`` independently, so the
relaxation of a ``Relu`` in one branch cannot cancel against the same ``Relu`` in the
other: Conv -> BatchNorm -> Relu came out at 23.6 and MatMul -> Add -> Relu -> MatMul
-> Add at 51.4 for rewrites that are exact. A *zonotope* keeps every value as an affine
form over shared noise symbols,

    x_i = c_i + sum_j G[j, i] * eps_j,      eps_j in [-1, 1],

so structure the two models share cancels exactly (``x - x`` is the zero form, two
identical branches stay identical) and only genuinely different structure costs
precision. This is the abstract domain of DeepZ (Singh et al., "Fast and Effective
Robustness Certification", NeurIPS 2018).

Soundness argument (what each piece guarantees, and where it is not rigorous):

* Affine ops (Gemm / MatMul / Conv / BatchNormalization / Add / Sub / Mul by a
  constant / shape ops / averaging) map a zonotope to the exact image zonotope: the
  generator matrix goes through the same linear map as the centre, so no precision is
  lost and no new symbols are needed.
* ``Relu``: with concretised bounds ``[l, u]`` of a neuron, ``l >= 0`` and ``u <= 0``
  are exact. For ``l < 0 < u`` the DeepZ parallelogram is used:
  ``lam*x <= relu(x) <= lam*x - lam*l`` with ``lam = u/(u-l)``, so
  ``relu(x) = lam*x + mu + mu*eps_new`` with ``mu = -lam*l/2`` and a fresh symbol.
* ``Sigmoid`` / ``Tanh``: the derivative is unimodal and peaks at 0, so its minimum on
  ``[l, u]`` is at an endpoint. With ``lam = min(f'(l), f'(u))`` the function
  ``f(x) - lam*x`` is non-decreasing on ``[l, u]``, hence lies in
  ``[f(l) - lam*l, f(u) - lam*u]``; that band is covered by a fresh symbol.
* An op with no rule here is enclosed by an *interval* box from ``onnxsim.interval``
  (fresh symbol per element: sound, but all correlation through that op is lost and a
  ``precision lost at <op>`` note is recorded). If even that is unbounded, the tensor is
  ``Top`` (unbounded) and so is everything computed from it -- never a wrong bound.
* Symbol-count guard: once a tensor has more than ``max_symbols`` generators, the
  smallest ones are replaced by one fresh *diagonal* symbol per element whose coefficient
  is ``sum |g_j|`` over the dropped generators. ``sum g_j eps_j`` lies in
  ``[-sum|g_j|, sum|g_j|]`` elementwise, so the replacement encloses it; it only loses
  correlation with other tensors.
* Arithmetic is float64 with a small relative widening in :meth:`Zonotope.bounds`
  (default ``1e-9``). That is not directed rounding; it is several orders of magnitude
  above the float64 rounding of networks of this size. The bounds describe the
  *real-number* function; evaluating the model in float32 can exceed them by float32
  rounding (about 1e-6 relative per op), which callers should budget for (the default
  ``slack`` in the test-suite soundness check is ``1e-4``).

Scope: generators are stored densely per tensor (``k x tensor size``), so this targets
small windows and models, not full-size networks. Integrating this into
``onnxsim.certify`` for nonlinear windows is planned; it is deliberately standalone for
now.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import numpy_helper

from . import interval as _interval
from . import ranges as _ranges

# Dense generators cost (symbols x tensor size) float64s. 16384 keeps a 2-layer Conv window at
# 16x16 exact-cancelling (4096 does not: see ``_consolidate``) at roughly 150 MB per live tensor.
DEFAULT_MAX_SYMBOLS = 16384
_REL_SLACK = 1e-9
_NONLINEAR = frozenset({"Relu", "Sigmoid", "Tanh"})


class _Space:
    """Allocator of globally unique noise-symbol ids, shared by every model in a run."""

    def __init__(self) -> None:
        self.next = 0

    def fresh(self, n: int) -> np.ndarray:
        ids = np.arange(self.next, self.next + n, dtype=np.int64)
        self.next += n
        return ids


class Zonotope:
    """Affine form ``c + sum_j G[j] * eps_j`` per tensor element, ``eps_j in [-1, 1]``.

    ``c`` has the tensor's shape; ``G`` has shape ``(k,) + c.shape`` and ``ids`` (sorted,
    length ``k``) names the global symbol of each generator row.
    """

    __slots__ = ("c", "ids", "G")

    def __init__(self, c: np.ndarray, ids: np.ndarray, G: np.ndarray) -> None:
        self.c = np.asarray(c, dtype=np.float64)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.G = np.asarray(G, dtype=np.float64).reshape(
            (len(self.ids),) + self.c.shape
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.c.shape

    @property
    def size(self) -> int:
        return int(self.c.size)

    @classmethod
    def constant(cls, c: np.ndarray) -> "Zonotope":
        c = np.asarray(c, dtype=np.float64)
        return cls(c, np.zeros(0, np.int64), np.zeros((0,) + c.shape))

    def radius(self) -> np.ndarray:
        if len(self.ids) == 0:
            return np.zeros_like(self.c)
        return np.abs(self.G).sum(axis=0)

    def bounds(self, rel_slack: float = _REL_SLACK) -> Tuple[np.ndarray, np.ndarray]:
        """Concretise to an elementwise box ``(lo, hi)`` (widened by ``rel_slack``)."""
        r = self.radius()
        pad = rel_slack * (np.abs(self.c) + r)
        return self.c - r - pad, self.c + r + pad


class _Top:
    """An unbounded tensor (shape may be unknown). Anything computed from it is Top."""

    def __init__(self, shape: Optional[Tuple[int, ...]] = None) -> None:
        self.shape = shape


Value = Union[np.ndarray, Zonotope, _Top]


# --------------------------------------------------------------------------
# Zonotope algebra
# --------------------------------------------------------------------------


def _lift(g: np.ndarray, ndim: int) -> np.ndarray:
    """Give generators ``(k, *S)`` the rank of an ``ndim``-dimensional tensor (broadcast-ready)."""
    return g.reshape((g.shape[0],) + (1,) * (ndim - (g.ndim - 1)) + g.shape[1:])


def _union(a: Zonotope, b: Zonotope) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.union1d(a.ids, b.ids)

    def place(z: Zonotope) -> np.ndarray:
        out = np.zeros((len(ids),) + z.c.shape)
        if len(z.ids):
            out[np.searchsorted(ids, z.ids)] = z.G
        return out

    return ids, place(a), place(b)


def _add(a: Zonotope, b: Zonotope, sign: float = 1.0) -> Zonotope:
    ids, ga, gb = _union(a, b)
    nd = max(a.c.ndim, b.c.ndim)
    c = a.c + sign * b.c
    g = _lift(ga, nd) + sign * _lift(gb, nd)
    return Zonotope(c, ids, np.broadcast_to(g, (len(ids),) + c.shape))


def _scale(z: Zonotope, k: np.ndarray) -> Zonotope:
    """Elementwise multiply by a constant array ``k`` (broadcasting)."""
    k = np.asarray(k, dtype=np.float64)
    c = z.c * k
    g = _lift(z.G, c.ndim) * k
    return Zonotope(c, z.ids, np.broadcast_to(g, (len(z.ids),) + c.shape))


def _new_symbols(space: _Space, radius: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fresh diagonal symbols, one per element with a positive ``radius``."""
    flat = radius.reshape(-1)
    idx = np.flatnonzero(flat > 0)
    ids = space.fresh(len(idx))
    g = np.zeros((len(idx), flat.size))
    g[np.arange(len(idx)), idx] = flat[idx]
    return ids, g.reshape((len(idx),) + radius.shape)


def _with_new(
    z: Zonotope, c: np.ndarray, g: np.ndarray, new_radius: np.ndarray, space: _Space
) -> Zonotope:
    ids_n, g_n = _new_symbols(space, new_radius)
    return Zonotope(c, np.concatenate([z.ids, ids_n]), np.concatenate([g, g_n]))


def _relu(z: Zonotope, space: _Space) -> Zonotope:
    lo, hi = z.bounds()
    pos, neg = lo >= 0, hi <= 0
    unstable = ~(pos | neg)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(pos, 1.0, np.where(neg, 0.0, hi / (hi - lo)))
    mu = np.where(unstable, -lam * lo / 2.0, 0.0)
    return _with_new(z, lam * z.c + mu, _lift(z.G, z.c.ndim) * lam, mu, space)


def _sigmoid_f(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_d(x: np.ndarray) -> np.ndarray:
    s = _sigmoid_f(x)
    return s * (1.0 - s)


def _tanh_d(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def _unimodal(z: Zonotope, space: _Space, f, df) -> Zonotope:
    """Sigmoid/Tanh-style relaxation (see module docstring)."""
    lo, hi = z.bounds()
    lam = np.minimum(df(lo), df(hi))
    band_lo = f(lo) - lam * lo
    band_hi = f(hi) - lam * hi
    mid = (band_lo + band_hi) / 2.0
    rad = (band_hi - band_lo) / 2.0 + 1e-12  # covers f's own float64 rounding
    return _with_new(z, lam * z.c + mid, _lift(z.G, z.c.ndim) * lam, rad, space)


def _consolidate(z: Zonotope, space: _Space, max_symbols: int) -> Zonotope:
    k = len(z.ids)
    if k <= max_symbols:
        return z
    # Drop enough of the smallest generators that, even after adding one diagonal symbol
    # per element, the count is back under the cap (best effort when size > cap).
    ndrop = min(k, max(1, k - max_symbols + z.size))
    mag = np.abs(z.G).reshape(k, -1).sum(axis=1)
    order = np.argsort(mag, kind="stable")
    drop, keep = order[:ndrop], np.sort(order[ndrop:])
    resid = np.abs(z.G[drop]).sum(axis=0)
    ids_n, g_n = _new_symbols(space, resid)
    return Zonotope(
        z.c,
        np.concatenate([z.ids[keep], ids_n]),
        np.concatenate([z.G[keep], g_n]),
    )


def _relu_slopes(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Range of ``relu'`` over ``[lo, hi]`` (elementwise): exactly 1, exactly 0, or ``[0, 1]``."""
    return np.where(lo >= 0, 1.0, 0.0), np.where(hi <= 0, 0.0, 1.0)


def _unimodal_slopes(df) -> Any:
    """Slope range of a function whose derivative ``df`` is unimodal and peaks at 0."""

    def slopes(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nearest0 = np.clip(0.0, lo, hi)
        return np.minimum(df(lo), df(hi)), df(nearest0)

    return slopes


def _slope_apply(
    dx: Zonotope, s_lo: np.ndarray, s_hi: np.ndarray, space: "_Space"
) -> Zonotope:
    """Enclose ``f(a) - f(b)`` given an enclosure ``dx`` of ``a - b`` and the slope range of ``f``.

    Mean-value theorem: ``f(a) - f(b) = f'(xi) * (a - b)`` with ``xi`` between ``a`` and ``b``,
    so ``f'(xi)`` lies in ``[s_lo, s_hi]`` when that range was taken over a set containing both.
    Then ``s*d = s_mid*d + (s - s_mid)*d`` and ``|(s - s_mid)*d| <= half_width * max|d|``.
    """
    mid, half = (s_lo + s_hi) / 2.0, (s_hi - s_lo) / 2.0
    lo, hi = dx.bounds()
    rad = half * np.maximum(np.abs(lo), np.abs(hi)) * (1.0 + 1e-12)
    out = _scale(dx, mid)
    return _with_new(out, out.c, out.G, rad, space)


def _box_zonotope(lo: np.ndarray, hi: np.ndarray, space: _Space) -> Zonotope:
    c, r = (lo + hi) / 2.0, (hi - lo) / 2.0
    ids, g = _new_symbols(space, r)
    return Zonotope(c, ids, g)


# --------------------------------------------------------------------------
# Numeric kernels
# --------------------------------------------------------------------------


def _conv2d(
    x: np.ndarray,
    w: np.ndarray,
    strides: List[int],
    pads: List[int],
    dil: List[int],
    group: int,
) -> np.ndarray:
    b, c, h, wd = x.shape
    m, cg, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])))
    eh, ew = dil[0] * (kh - 1) + 1, dil[1] * (kw - 1) + 1
    win = np.lib.stride_tricks.sliding_window_view(xp, (eh, ew), axis=(2, 3))
    win = win[:, :, :: strides[0], :: strides[1], :: dil[0], :: dil[1]]
    oh, ow = win.shape[2], win.shape[3]
    out = np.zeros((b, m, oh, ow))
    mg = m // group
    for g in range(group):
        wg = w[g * mg : (g + 1) * mg]
        xg = win[:, g * cg : (g + 1) * cg]
        out[:, g * mg : (g + 1) * mg] = np.einsum("bchwij,mcij->bmhw", xg, wg)
    return out


def _attrs(node: onnx.NodeProto) -> Dict[str, Any]:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _ref_run(
    node: onnx.NodeProto, arrays: List[np.ndarray], opsets
) -> List[np.ndarray]:
    from onnx.reference import ReferenceEvaluator

    n = onnx.NodeProto()
    n.CopyFrom(node)
    names = [f"i{k}" for k in range(len(arrays))]
    del n.input[:]
    n.input.extend(names)
    del n.output[:]
    n.output.extend(f"o{k}" for k in range(max(1, len(node.output))))
    arrays = [
        np.asarray(a, dtype=np.float64)
        if np.asarray(a).dtype.kind == "f"
        else np.asarray(a)
        for a in arrays
    ]
    g = onnx.helper.make_graph(
        [n],
        "n",
        [
            onnx.helper.make_tensor_value_info(
                nm, onnx.helper.np_dtype_to_tensor_dtype(a.dtype), a.shape
            )
            for nm, a in zip(names, arrays)
        ],
        [onnx.helper.make_empty_tensor_value_info(o) for o in n.output],
    )
    m = onnx.helper.make_model(g, opset_imports=list(opsets))
    return list(ReferenceEvaluator(m).run(None, dict(zip(names, arrays))))


# --------------------------------------------------------------------------
# Graph evaluation
# --------------------------------------------------------------------------


class _Failed(Exception):
    """A rule could not be applied; the caller falls back to an interval box."""


class _Evaluator:
    """Evaluates one model's graph on zonotopes (sharing a ``_Space`` with other models)."""

    def __init__(
        self,
        model: onnx.ModelProto,
        space: _Space,
        max_symbols: int,
        record: Optional[List[Tuple[str, Zonotope, Zonotope]]] = None,
        pair: Optional[List[Tuple[str, Zonotope, Zonotope]]] = None,
    ) -> None:
        self.model, self.space, self.max_symbols = model, space, max_symbols
        # ``record``: the first model of a difference run logs (kind, input, output) of every
        # nonlinearity. ``pair``: the second model replays that log, pairing its j-th
        # nonlinearity of each kind with the first model's j-th (see ``_paired``).
        self.record, self.pair = record, pair
        self.nl_count: Dict[str, int] = {}
        self.opsets = list(model.opset_import)
        self.notes: List[str] = []
        self.values: Dict[str, Value] = {}
        try:
            g = onnx.shape_inference.infer_shapes(model).graph
        except Exception:
            g = model.graph
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        for vi in list(g.input) + list(g.value_info) + list(g.output):
            tt = vi.type.tensor_type
            if vi.type.HasField("tensor_type") and tt.HasField("shape"):
                dims = [
                    d.dim_value if d.HasField("dim_value") else 0 for d in tt.shape.dim
                ]
                if all(d > 0 for d in dims):
                    self.shapes[vi.name] = tuple(dims)

    # -- helpers
    def _note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)

    def _const(self, v: Value) -> bool:
        return isinstance(v, np.ndarray)

    def _as_z(self, v: Value) -> Zonotope:
        if isinstance(v, Zonotope):
            return v
        if isinstance(v, np.ndarray):
            return Zonotope.constant(v)
        raise _Failed("unbounded operand")

    def _post(self, z: Zonotope) -> Zonotope:
        out = _consolidate(z, self.space, self.max_symbols)
        if out is not z:
            self._note(
                f"symbol cap ({self.max_symbols}) reached: generators were merged into boxes, "
                "so correlation with the other model is lost and bounds may be loose "
                "(raise max_symbols)"
            )
        return out

    def run(self, inputs: Dict[str, Zonotope]) -> Dict[str, Value]:
        g = self.model.graph
        for t in g.initializer:
            self.values[t.name] = numpy_helper.to_array(t)
        self.values.update(inputs)
        for node in g.node:
            args: List[Optional[Value]] = [
                self.values.get(x) if x else None for x in node.input
            ]
            outs = [o for o in node.output if o]
            if any(x and self.values.get(x) is None for x in node.input):
                for o in outs:
                    self.values[o] = _Top(self.shapes.get(o))
                continue
            if any(isinstance(a, _Top) for a in args):
                self._note(f"unbounded operand reaches {node.op_type}")
                for o in outs:
                    self.values[o] = _Top(self.shapes.get(o))
                continue
            try:
                res = self._apply(node, args)
            except Exception as e:  # no rule / shape problem: fall back, never guess
                res = self._fallback(node, args, e)
            for o, v in zip(outs, res):
                self.values[o] = v
        return self.values

    def _apply(self, node: onnx.NodeProto, args: List[Optional[Value]]) -> List[Value]:
        if node.domain not in ("", "ai.onnx"):
            raise _Failed(f"domain {node.domain!r}")
        present = [a for a in args if a is not None]
        if present and all(self._const(a) for a in present) and "" not in node.input:
            return list(_ref_run(node, [np.asarray(a) for a in present], self.opsets))
        if not node.input:
            return list(_ref_run(node, [], self.opsets))
        fn = getattr(self, "op_" + node.op_type, None)
        if fn is None:
            raise _Failed(f"no zonotope rule for {node.op_type}")
        out = fn(node, args)
        res = out if isinstance(out, list) else [out]
        res = [self._post(r) if isinstance(r, Zonotope) else r for r in res]
        if (
            self.record is not None
            and node.op_type in _NONLINEAR
            and isinstance(args[0], Zonotope)
            and isinstance(res[0], Zonotope)
        ):
            self.record.append((node.op_type, args[0], res[0]))
        return res

    def _fallback(
        self, node: onnx.NodeProto, args: List[Optional[Value]], why: Exception
    ) -> List[Value]:
        outs = [o for o in node.output if o]
        box = _interval_box(node, args, self.opsets)
        self._note(f"precision lost at {node.op_type} ({why})")
        if box is None:
            return [_Top(self.shapes.get(o)) for o in outs]
        boxed: List[Value] = [_box_zonotope(lo, hi, self.space) for lo, hi in box]
        return boxed[: len(outs)]

    # -- elementwise / arithmetic
    def op_Identity(self, node, args):
        return args[0]

    op_Dropout = op_Identity

    def op_Neg(self, node, args):
        z = self._as_z(args[0])
        return Zonotope(-z.c, z.ids, -z.G)

    def _binary(self, args, sign):
        a, b = args[0], args[1]
        if self._const(b):
            return _add(self._as_z(a), Zonotope.constant(np.asarray(b)), sign)
        if self._const(a):
            return _add(Zonotope.constant(np.asarray(a)), self._as_z(b), sign)
        return _add(self._as_z(a), self._as_z(b), sign)

    def op_Add(self, node, args):
        return self._binary(args, 1.0)

    def op_Sub(self, node, args):
        return self._binary(args, -1.0)

    def op_Mul(self, node, args):
        a, b = args[0], args[1]
        if self._const(b):
            return _scale(self._as_z(a), np.asarray(b))
        if self._const(a):
            return _scale(self._as_z(b), np.asarray(a))
        raise _Failed("product of two non-constant tensors")

    def _paired(self, kind: str, z: Zonotope, slopes) -> Optional[Zonotope]:
        """Second-model nonlinearity: ``out_b = out_a - (f(x_a) - f(x_b))`` for the paired node.

        Sound for ANY pairing, right or wrong: ``out_a`` already encloses ``f(x_a)``, and the
        subtracted term encloses ``f(x_a) - f(x_b)`` from the *difference* ``x_a - x_b`` (a
        zonotope over the shared symbols, tiny when the pre-activations agree). A bad pairing
        only makes ``x_a - x_b`` large, hence the bound loose -- never wrong. Returns ``None``
        when there is nothing to pair with (the caller then evaluates ``f`` independently).
        """
        if self.pair is None:
            return None
        j = self.nl_count.get(kind, 0)
        self.nl_count[kind] = j + 1
        recs = [r for r in self.pair if r[0] == kind]
        if j >= len(recs) or recs[j][1].shape != z.shape:
            return None
        _, x_a, out_a = recs[j]
        dx = _add(x_a, z, -1.0)
        lo_a, hi_a = x_a.bounds()
        lo_b, hi_b = z.bounds()
        s_lo, s_hi = slopes(np.minimum(lo_a, lo_b), np.maximum(hi_a, hi_b))
        return _add(out_a, _slope_apply(dx, s_lo, s_hi, self.space), -1.0)

    def op_Relu(self, node, args):
        z = self._as_z(args[0])
        out = self._paired("Relu", z, _relu_slopes)
        return out if out is not None else _relu(z, self.space)

    def op_Sigmoid(self, node, args):
        z = self._as_z(args[0])
        out = self._paired("Sigmoid", z, _unimodal_slopes(_sigmoid_d))
        return (
            out if out is not None else _unimodal(z, self.space, _sigmoid_f, _sigmoid_d)
        )

    def op_Tanh(self, node, args):
        z = self._as_z(args[0])
        out = self._paired("Tanh", z, _unimodal_slopes(_tanh_d))
        return out if out is not None else _unimodal(z, self.space, np.tanh, _tanh_d)

    # -- linear algebra
    def _matmul(self, a: Value, b: Value) -> Zonotope:
        if self._const(b) and not self._const(a):
            z, w = self._as_z(a), np.asarray(b, dtype=np.float64)
            if z.c.ndim < 2 or w.ndim != 2:
                raise _Failed("MatMul rank")
            return Zonotope(z.c @ w, z.ids, z.G @ w)
        if self._const(a) and not self._const(b):
            w, z = np.asarray(a, dtype=np.float64), self._as_z(b)
            if z.c.ndim < 2 or w.ndim != 2:
                raise _Failed("MatMul rank")
            return Zonotope(w @ z.c, z.ids, w @ z.G)
        raise _Failed("MatMul of two non-constant tensors")

    def op_MatMul(self, node, args):
        return self._matmul(args[0], args[1])

    def op_Gemm(self, node, args):
        at = _attrs(node)
        a, b = args[0], args[1]

        def maybe_t(v: Value, flag: int) -> Value:
            if not flag:
                return v
            if isinstance(v, np.ndarray):
                return v.T
            z = self._as_z(v)
            return Zonotope(z.c.T, z.ids, np.swapaxes(z.G, 1, 2))

        y = self._matmul(
            maybe_t(a, at.get("transA", 0)), maybe_t(b, at.get("transB", 0))
        )
        alpha, beta = float(at.get("alpha", 1.0)), float(at.get("beta", 1.0))
        if alpha != 1.0:
            y = _scale(y, np.asarray(alpha))
        if len(args) > 2 and args[2] is not None:
            c = args[2]
            c = (
                np.asarray(c) * beta
                if self._const(c)
                else _scale(self._as_z(c), np.asarray(beta))
            )
            y = _add(y, self._as_z(c))
        return y

    def op_BatchNormalization(self, node, args):
        if _attrs(node).get("training_mode", 0):
            raise _Failed("training_mode")
        z = self._as_z(args[0])
        if not all(self._const(a) for a in args[1:5]):
            raise _Failed("non-constant BatchNormalization parameters")
        scale, bias, mean, var = (np.asarray(a, dtype=np.float64) for a in args[1:5])
        s = scale / np.sqrt(var + float(_attrs(node).get("epsilon", 1e-5)))
        shp = [1, -1] + [1] * (z.c.ndim - 2)
        s, bias, mean = s.reshape(shp), bias.reshape(shp), mean.reshape(shp)
        return Zonotope((z.c - mean) * s + bias, z.ids, z.G * s)

    def op_Conv(self, node, args):
        at = _attrs(node)
        z = self._as_z(args[0])
        if not self._const(args[1]):
            raise _Failed("non-constant Conv weights")
        w = np.asarray(args[1], dtype=np.float64)
        bias = None
        if len(args) > 2 and args[2] is not None:
            if not self._const(args[2]):
                raise _Failed("non-constant Conv bias")
            bias = np.asarray(args[2], dtype=np.float64)
        if z.c.ndim != 4 or at.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            raise _Failed("Conv form")
        strides = list(at.get("strides", [1, 1]))
        dil = list(at.get("dilations", [1, 1]))
        pads = list(at.get("pads", [0, 0, 0, 0]))
        group = int(at.get("group", 1))
        c = _conv2d(z.c, w, strides, pads, dil, group)
        k = len(z.ids)
        if k:
            n, ch, h, wd = z.c.shape
            g = _conv2d(z.G.reshape(k * n, ch, h, wd), w, strides, pads, dil, group)
            g = g.reshape((k, n) + g.shape[1:])
        else:
            g = np.zeros((0,) + c.shape)
        if bias is not None:
            c = c + bias.reshape(1, -1, 1, 1)
        return Zonotope(c, z.ids, g)

    # -- shape ops
    def op_Flatten(self, node, args):
        z = self._as_z(args[0])
        axis = int(_attrs(node).get("axis", 1))
        axis = axis + z.c.ndim if axis < 0 else axis
        rows = int(np.prod(z.c.shape[:axis], dtype=np.int64))
        return Zonotope(z.c.reshape(rows, -1), z.ids, z.G.reshape(len(z.ids), rows, -1))

    def op_Reshape(self, node, args):
        z = self._as_z(args[0])
        if not self._const(args[1]):
            raise _Failed("non-constant Reshape target")
        if _attrs(node).get("allowzero", 0):
            raise _Failed("allowzero")
        shape = [int(s) for s in np.asarray(args[1])]
        shape = [z.c.shape[i] if s == 0 else s for i, s in enumerate(shape)]
        c = z.c.reshape(shape)
        return Zonotope(c, z.ids, z.G.reshape((len(z.ids),) + c.shape))

    def op_Transpose(self, node, args):
        z = self._as_z(args[0])
        perm = _attrs(node).get("perm") or list(range(z.c.ndim))[::-1]
        return Zonotope(
            np.transpose(z.c, perm),
            z.ids,
            np.transpose(z.G, [0] + [p + 1 for p in perm]),
        )

    def _raw_axes(self, node, args):
        at = _attrs(node)
        if len(args) > 1 and args[1] is not None and self._const(args[1]):
            return [int(a) for a in np.asarray(args[1]).reshape(-1)]
        return [int(a) for a in at.get("axes", [])]

    def _axes(self, node, args, nd):
        return [a + nd if a < 0 else a for a in self._raw_axes(node, args)]

    def op_Squeeze(self, node, args):
        z = self._as_z(args[0])
        axes = self._axes(node, args, z.c.ndim) or [
            i for i, d in enumerate(z.c.shape) if d == 1
        ]
        return Zonotope(
            np.squeeze(z.c, tuple(axes)),
            z.ids,
            np.squeeze(z.G, tuple(a + 1 for a in axes)),
        )

    def op_Unsqueeze(self, node, args):
        z = self._as_z(args[0])
        raw = self._raw_axes(node, args)
        out_nd = z.c.ndim + len(raw)
        axes = [a + out_nd if a < 0 else a for a in raw]
        return Zonotope(
            np.expand_dims(z.c, tuple(axes)),
            z.ids,
            np.expand_dims(z.G, tuple(a + 1 for a in axes)),
        )

    def op_Concat(self, node, args):
        zs = [self._as_z(a) for a in args if a is not None]
        axis = int(_attrs(node).get("axis", 0))
        axis = axis + zs[0].c.ndim if axis < 0 else axis
        ids = zs[0].ids
        for z in zs[1:]:
            ids = np.union1d(ids, z.ids)
        gs = []
        for z in zs:
            g = np.zeros((len(ids),) + z.c.shape)
            if len(z.ids):
                g[np.searchsorted(ids, z.ids)] = z.G
            gs.append(g)
        return Zonotope(
            np.concatenate([z.c for z in zs], axis=axis),
            ids,
            np.concatenate(gs, axis=axis + 1),
        )

    def op_GlobalAveragePool(self, node, args):
        z = self._as_z(args[0])
        ax = tuple(range(2, z.c.ndim))
        return Zonotope(
            z.c.mean(axis=ax, keepdims=True),
            z.ids,
            z.G.mean(axis=tuple(a + 1 for a in ax), keepdims=True),
        )

    def _reduce(self, node, args, fn):
        z = self._as_z(args[0])
        axes = self._axes(node, args, z.c.ndim) or list(range(z.c.ndim))
        keep = bool(_attrs(node).get("keepdims", 1))
        ax, gax = tuple(axes), tuple(a + 1 for a in axes)
        return Zonotope(
            fn(z.c, axis=ax, keepdims=keep), z.ids, fn(z.G, axis=gax, keepdims=keep)
        )

    def op_ReduceMean(self, node, args):
        return self._reduce(node, args, np.mean)

    def op_ReduceSum(self, node, args):
        return self._reduce(node, args, np.sum)


def _interval_box(
    node: onnx.NodeProto, args: List[Optional[Value]], opsets
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """Interval enclosure of ``node``'s outputs from ``onnxsim.interval`` (None if unbounded)."""
    inputs, inits, ranges, names = [], [], {}, []
    for k, a in enumerate(args):
        if a is None:
            names.append("")
            continue
        nm = f"i{k}"
        names.append(nm)
        if isinstance(a, np.ndarray):
            inits.append(numpy_helper.from_array(a, nm))
        else:
            if not isinstance(
                a, Zonotope
            ):  # an unbounded operand has no box to enclose
                return None
            lo, hi = a.bounds()
            inputs.append(
                onnx.helper.make_tensor_value_info(
                    nm, onnx.TensorProto.DOUBLE, lo.shape
                )
            )
            ranges[nm] = (lo, hi)
    n = onnx.NodeProto()
    n.CopyFrom(node)
    del n.input[:]
    n.input.extend(names)
    outs = [f"o{k}" for k in range(len(node.output))]
    del n.output[:]
    n.output.extend(outs)
    g = onnx.helper.make_graph(
        [n],
        "fb",
        inputs,
        [onnx.helper.make_empty_tensor_value_info(o) for o in outs],
        inits,
    )
    m = onnx.helper.make_model(g, opset_imports=list(opsets))
    m.ir_version = 8
    try:
        res = _interval.propagate(m, ranges)
    except Exception:
        return None
    boxes = []
    for o, real in zip(outs, node.output):
        if not real:
            continue
        if o not in res.intervals:
            return None
        lo, hi = res.intervals[o]
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            return None
        boxes.append((np.asarray(lo, np.float64), np.asarray(hi, np.float64)))
    return boxes


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


@dataclasses.dataclass
class ZonotopeResult:
    """Per-tensor abstract values of one model (see :func:`propagate`)."""

    tensors: Dict[str, Value]
    notes: List[str]

    def bounds(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Elementwise ``(lo, hi)`` enclosing ``name`` for every input in the box."""
        v = self.tensors[name]
        if isinstance(v, Zonotope):
            return v.bounds()
        if isinstance(v, _Top):
            shape = v.shape if v.shape is not None else ()
            return np.full(shape, -np.inf), np.full(shape, np.inf)
        a = np.asarray(v, dtype=np.float64)
        return a, a


def _input_zonotopes(
    models: List[onnx.ModelProto],
    input_ranges: Optional[Dict[str, Tuple]],
    space: _Space,
) -> Dict[str, Zonotope]:
    ref = models[0]
    inits = {t.name for t in ref.graph.initializer}
    ranges: Dict[str, Tuple] = dict(_ranges.get_ranges(ref))
    ranges.update(input_ranges or {})
    out: Dict[str, Zonotope] = {}
    for vi in ref.graph.input:
        if vi.name in inits:
            continue
        tt = vi.type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else 0 for d in tt.shape.dim]
        if not dims or any(d <= 0 for d in dims):
            raise ValueError(f"input {vi.name!r} needs a fully static shape")
        if vi.name not in ranges:
            raise ValueError(
                f"input {vi.name!r} has no finite range; pass input_ranges"
            )
        lo, hi = (
            np.broadcast_to(np.asarray(b, dtype=np.float64), dims).copy()
            for b in ranges[vi.name]
        )
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise ValueError(f"input {vi.name!r} has an unbounded range")
        if np.any(lo > hi):
            raise ValueError(f"input {vi.name!r} has an empty range")
        out[vi.name] = _box_zonotope(lo, hi, space)
    return out


def propagate(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
) -> ZonotopeResult:
    """Propagate the input box through ``model`` as zonotopes.

    :param input_ranges: ``{input: (lo, hi)}`` (scalars or arrays broadcastable to the input
        shape), merged over the model's own ``onnxsim.range.*`` annotations. Every graph input
        needs a finite, static-shape range, else ``ValueError``.
    :param max_symbols: generator cap per tensor before the sound consolidation step.
    """
    space = _Space()
    ev = _Evaluator(model, space, max_symbols)
    values = ev.run(_input_zonotopes([model], input_ranges, space))
    return ZonotopeResult(dict(values), ev.notes)


@dataclasses.dataclass
class DifferenceBound:
    """Certified bound on ``|orig - simplified|`` over the input box, per output."""

    max_abs: Dict[str, np.ndarray]  # elementwise upper bound on |orig - simplified|
    ref_min_abs: Dict[str, np.ndarray]  # elementwise lower bound on |simplified|
    notes: List[str]

    @property
    def worst(self) -> float:
        """Largest per-element bound over all outputs (``inf`` if any is unbounded)."""
        if not self.max_abs:
            return 0.0
        return float(
            max(np.max(v) if np.size(v) else 0.0 for v in self.max_abs.values())
        )

    @property
    def bounded(self) -> bool:
        return bool(np.isfinite(self.worst))

    def within(self, atol: float = 1e-5, rtol: float = 1e-4) -> bool:
        """Sound check that ``|orig - simplified| <= atol + rtol * |simplified|`` everywhere.

        Sufficient condition used: ``max|diff| <= atol + rtol * min|simplified|`` where
        ``min|simplified|`` is a lower bound of ``|simplified|`` over the box (zero whenever the
        output's range contains 0). Conservative: it can say ``False`` for an equivalent pair,
        never ``True`` for a pair that differs by more than the tolerance.
        """
        for name, d in self.max_abs.items():
            if not np.all(np.isfinite(d)):
                return False
            if not np.all(d <= atol + rtol * self.ref_min_abs[name]):
                return False
        return True


def bound_difference(
    orig: onnx.ModelProto,
    simplified: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
) -> DifferenceBound:
    """Certified per-output bound on ``|orig(x) - simplified(x)|`` for ``x`` in the input box.

    Both models are evaluated on the *same* input zonotope (shared noise symbols) and their
    outputs subtracted, so everything the two share cancels. An unbounded or unsupported
    situation yields ``inf`` for the affected outputs and a note -- never a NaN or a bound that
    is too small. Inputs and outputs must have the same names in both models.
    """
    in_a = {i.name for i in orig.graph.input} - {t.name for t in orig.graph.initializer}
    in_b = {i.name for i in simplified.graph.input} - {
        t.name for t in simplified.graph.initializer
    }
    if in_a != in_b:
        raise ValueError(f"graph inputs differ: {sorted(in_a)} vs {sorted(in_b)}")
    out_a = [o.name for o in orig.graph.output]
    if out_a != [o.name for o in simplified.graph.output]:
        raise ValueError("graph outputs differ")
    space = _Space()
    notes: List[str] = []
    try:
        inputs = _input_zonotopes([orig, simplified], input_ranges, space)
    except ValueError as e:
        return DifferenceBound(
            {o: np.array(np.inf) for o in out_a},
            {o: np.array(0.0) for o in out_a},
            [f"unbounded input: {e}"],
        )
    log: List[Tuple[str, Zonotope, Zonotope]] = []
    ev_a = _Evaluator(orig, space, max_symbols, record=log)
    va = ev_a.run(inputs)
    ev_b = _Evaluator(simplified, space, max_symbols, pair=log)
    vb = ev_b.run(inputs)
    notes += [f"orig: {n}" for n in ev_a.notes] + [
        f"simplified: {n}" for n in ev_b.notes
    ]
    max_abs: Dict[str, np.ndarray] = {}
    ref_min: Dict[str, np.ndarray] = {}
    for o in out_a:
        a, b = va[o], vb[o]
        if isinstance(a, _Top) or isinstance(b, _Top):
            max_abs[o], ref_min[o] = np.array(np.inf), np.array(0.0)
            continue
        za, zb = ev_a._as_z(a), ev_b._as_z(b)
        if za.shape != zb.shape:
            max_abs[o], ref_min[o] = np.array(np.inf), np.array(0.0)
            notes.append(f"output {o}: shapes differ {za.shape} vs {zb.shape}")
            continue
        lo, hi = _add(za, zb, -1.0).bounds()
        max_abs[o] = np.maximum(np.abs(lo), np.abs(hi))
        blo, bhi = zb.bounds()
        ref_min[o] = np.maximum(0.0, np.maximum(blo, -bhi))
    return DifferenceBound(max_abs, ref_min, notes)


def proves_equal(
    orig: onnx.ModelProto,
    simplified: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
) -> bool:
    """True only if ``|orig - simplified| <= atol + rtol*|simplified|`` is *proved* over the box.

    See :meth:`DifferenceBound.within` for exactly what is checked. ``False`` means "not proved",
    not "different".
    """
    return bound_difference(orig, simplified, input_ranges, max_symbols).within(
        atol, rtol
    )
