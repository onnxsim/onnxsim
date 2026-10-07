"""Translation validation of a simplified model against its original, with Z3.

``certify(orig, simplified)`` tries to *prove* the two models compute the same
outputs, instead of sampling random inputs the way ``onnxsim.model_checking``
does. It needs only the two models -- no rewrite log from the C++ passes -- and
works in four layers, cheapest first:

1. **Structural hashing.** Every tensor of both models gets a canonical id built
   from its op, attributes and input ids (initializers by exact bytes, graph
   inputs by name). Two tensors with the same id are computed identically, so
   everything onnxsim left alone is proved equal for free, whatever its size.
2. **Congruence peeling.** If two differing tensors come from the same op with
   the same attributes, it is enough to prove their inputs pairwise equal. This
   keeps nonlinear ops that onnxsim did not touch (``Relu`` after a fused
   ``Conv``) out of the solver.
3. **SMT window.** What is left -- the region a pass actually rewrote -- is
   encoded over exact rationals (Z3 reals) between shared cut points, and Z3 is
   asked for an input where the two sides differ by more than ``atol + rtol*|b|``.
   ``unsat`` is a proof; ``sat`` is a counterexample you can replay.
4. **Spatial shrink** (only when a window is too large for step 3). If every op
   in the window is spatially local (2-D Conv, BatchNormalization, Add/Sub/Mul,
   Neg, Identity, Relu, Clip) and every constant meeting a spatial tensor is
   uniform over space, the window is re-encoded at a smaller spatial size that
   keeps all of its border behaviour, and the verdict is ``proved-reduced``.
   ``proved-reduced`` counts as proved (``CertifyReport.ok``) but is a distinct
   label because the proof rests on the argument below, not on the full-size
   encoding. The argument, in short: an output position depends on a bounded patch
   plus which borders it touches; all interior positions are the same function; a
   top position is fixed by its index and a bottom position by its offset from
   the end. A smaller extent with the same residue modulo the total stride, no
   position touching both borders, and at least one interior position therefore
   contains a twin of every full-size position, with an identical symbolic
   expression. Proving a pointwise tolerance for all of them proves it at full
   size. (Details and the exact conditions: ``_plan_reduction``.) What it does
   NOT cover: windows with any other op (MatMul, Gemm, Reshape, pooling, ...),
   positional constants, per-position input ranges, non-NCHW layouts -- those
   keep their ``skipped`` verdict, with the reason.

   A difference found at the reduced size is lifted to the original shape by
   embedding the violating position's patch (a bottom position is shifted by the
   amount removed) and, when onnxruntime is available, replayed on the two real
   models. ``refuted`` is only reported if the replay reproduces it (or cannot
   run, which the detail text says); a replay that disagrees downgrades to
   ``skipped``, because a false alarm is worse than a skip.

5. **Zonotope fallback** (only for an output that is still ``skipped`` after steps 1-4, and
   only when every graph input has a finite range). ``onnxsim.zonotope`` evaluates both
   models on the *same* input box as affine forms over shared noise symbols and bounds
   ``|orig - simplified|`` per output element; shared structure cancels exactly, which is
   what lets it handle a ``Relu``/``Sigmoid``/``Tanh`` between two rewritten layers that the
   Z3 encoding gives up on. If the certified bound satisfies
   ``bound <= atol + rtol * min|simplified|`` (``min|simplified|`` is a lower bound of
   ``|simplified|`` over the box, zero whenever the output can reach 0, so ``rtol`` only
   helps when it is justified) the verdict is ``proved-affine``; it counts as proved. A bound
   that is merely too large proves nothing: the output stays ``skipped`` with the bound in the
   detail. This step can never produce ``refuted``, because an over-approximation exceeding
   the tolerance does not show a real difference. It is bounded by a cost estimate made from
   shapes alone (the zonotope code is not interruptible), by the overall time budget, and it
   never raises.

What a proof means, so it is not over-read:

* It is a statement about *real* arithmetic over the float32/float64 constants
  as stored. It says the rewritten window cannot differ by more than the
  tolerance for any input in the declared range. It does not model fp32
  rounding of the *evaluation* (ONNX Runtime, reordered kernels). The tolerance
  is what absorbs the rounding of re-computed constants (a folded BatchNorm
  scale, for instance).
* Constants derived through ``sqrt`` (BatchNorm's ``scale / sqrt(var + eps)``)
  are computed in float64 and then taken exactly.
* The op semantics below are *our* encoding of ONNX and are part of the trusted
  base; they are checked against onnxruntime in ``tests/test_certify.py``.
* A window the encoder cannot handle (unsupported op, unknown or symbolic
  shape, too many multiply-adds) is reported ``skipped`` with the reason --
  never silently treated as proved.
* Leaves other than graph inputs (shared intermediate tensors) are left
  unconstrained, which is sound but can only make a proof harder.

z3-solver is an optional dependency (the ``verify`` extra).
"""

import dataclasses
import hashlib
import importlib.util
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

# Statuses a window can end in. Everything but "proved-*" means certification
# of that output did not succeed.
NEEDS_RANGES_HINT = "no input_ranges given"
PROVED_STRUCTURAL = "proved-structural"
PROVED_CONGRUENCE = "proved-congruence"
PROVED_SMT = "proved-smt"
PROVED_REDUCED = "proved-reduced"
PROVED_AFFINE = "proved-affine"
REFUTED = "refuted"
SKIPPED = "skipped"

_NONDETERMINISTIC = {
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    "Multinomial",
    "Bernoulli",
    "Dropout",
}


@dataclasses.dataclass
class Window:
    """One proof obligation: the original tensor ``orig`` against ``simplified``."""

    orig: str
    simplified: str
    status: str
    detail: str = ""
    counterexample: Optional[Dict[str, np.ndarray]] = None
    # Sub-obligations a congruence proof rests on (not part of repr/eq).
    children: List["Window"] = dataclasses.field(
        default_factory=list, repr=False, compare=False
    )


@dataclasses.dataclass
class CertifyReport:
    outputs: Dict[str, str]  # graph output name -> status
    windows: List[Window]

    @property
    def ok(self) -> bool:
        return all(s.startswith("proved") for s in self.outputs.values())

    def __str__(self) -> str:
        lines = [f"certify: {'OK' if self.ok else 'NOT PROVED'}"]
        lines += [f"  output {k}: {v}" for k, v in self.outputs.items()]
        for w in self.windows:
            if w.status != PROVED_STRUCTURAL:
                lines.append(
                    f"  window {w.orig} ~ {w.simplified}: {w.status} {w.detail}".rstrip()
                )
        return "\n".join(lines)


class _Skip(Exception):
    pass


class _TooLarge(_Skip):
    """The window exceeds ``max_work`` -- the one skip a spatial shrink can cure."""


class _NoShrink(Exception):
    """The window cannot be soundly re-encoded at a smaller spatial size (reason in args[0])."""


def _z3():
    try:
        import z3
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "onnxsim.certify needs z3-solver (pip install z3-solver, the 'verify' extra)"
        ) from e
    return z3


# --------------------------------------------------------------------------
# Structural hashing of both models into one shared id space
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _Def:
    kind: str  # "input" | "const" | "op"
    name: str = ""  # input name / op_type
    domain: str = ""
    attrs: Tuple = ()
    inputs: Tuple[int, ...] = ()
    out_index: int = 0
    value: Optional[np.ndarray] = None  # const
    node: Optional[onnx.NodeProto] = None


class _Space:
    def __init__(self):
        self.key_to_id: Dict[tuple, int] = {}
        self.defs: List[_Def] = []
        self.shape: Dict[int, Optional[Tuple[int, ...]]] = {}
        self.dtype: Dict[int, Optional[int]] = {}

    def intern(self, key: tuple, make) -> int:
        if key not in self.key_to_id:
            self.key_to_id[key] = len(self.defs)
            self.defs.append(make())
        return self.key_to_id[key]

    def const(self, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr)
        key = (
            "const",
            arr.dtype.str,
            arr.shape,
            hashlib.sha1(arr.tobytes()).hexdigest(),
        )
        return self.intern(key, lambda: _Def("const", value=arr))


def _attr_key(node: onnx.NodeProto) -> Tuple:
    return tuple(
        sorted(
            (a.name, a.SerializeToString(deterministic=True)) for a in node.attribute
        )
    )


def _shapes(model: onnx.ModelProto) -> Dict[str, Tuple[Optional[Tuple[int, ...]], int]]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    out: Dict[str, Tuple[Optional[Tuple[int, ...]], int]] = {}
    g = inferred.graph
    for vi in list(g.input) + list(g.value_info) + list(g.output):
        tt = vi.type.tensor_type
        if not vi.type.HasField("tensor_type") or not tt.HasField("shape"):
            out[vi.name] = (None, tt.elem_type)
            continue
        dims = [d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim]
        known = tuple(d for d in dims if d is not None)
        out[vi.name] = (known if len(known) == len(dims) else None, tt.elem_type)
    for t in g.initializer:
        out[t.name] = (tuple(t.dims), t.data_type)
    return out


def _index(
    space: _Space, model: onnx.ModelProto, with_shapes: bool = True
) -> Dict[str, int]:
    """Map each tensor name of ``model`` to its id in ``space``."""
    g = model.graph
    shapes = _shapes(model) if with_shapes else {}
    tid: Dict[str, int] = {}
    inits = {t.name for t in g.initializer}

    def note(name, i):
        s = shapes.get(name)
        if s is not None:
            space.shape.setdefault(i, s[0])
            space.dtype.setdefault(i, s[1])

    for t in g.initializer:
        tid[t.name] = space.const(numpy_helper.to_array(t))
        note(t.name, tid[t.name])
    for vi in g.input:
        if vi.name in inits:
            continue
        i = space.intern(("input", vi.name), lambda vi=vi: _Def("input", name=vi.name))
        tid[vi.name] = i
        note(vi.name, i)
    for node in g.node:
        for o_idx, out in enumerate(node.output):
            if not out:
                continue
            if node.op_type == "Constant" and not node.domain:
                v = next((a for a in node.attribute if a.name == "value"), None)
                if v is not None:
                    tid[out] = space.const(numpy_helper.to_array(v.t))
                    note(out, tid[out])
                    continue
            ins = tuple(tid[x] if x else -1 for x in node.input)
            key = ("op", node.domain, node.op_type, _attr_key(node), ins, o_idx)
            i = space.intern(
                key,
                lambda node=node, ins=ins, o=o_idx: _Def(
                    "op", node.op_type, node.domain, _attr_key(node), ins, o, node=node
                ),
            )
            tid[out] = i
            note(out, i)
    return tid


# --------------------------------------------------------------------------
# SMT encoding of ONNX ops over exact rationals
# --------------------------------------------------------------------------


class _Encoder:
    def __init__(self, space: _Space, input_ranges, max_work: int):
        self.z3 = _z3()
        self.space = space
        self.input_ranges = input_ranges or {}
        self.max_work = max_work
        self.work = 0
        self.memo: Dict[int, np.ndarray] = {}
        self.leaves: Dict[int, np.ndarray] = {}  # id -> array of z3 variables
        self.constraints: list = []
        self.cut: set = set()
        self.zero = self.z3.RealVal(0)
        self.has_bounds = False
        # id -> reduced leaf shape, set when a spatial shrink is in effect
        self.shape_override: Dict[int, Tuple[int, ...]] = {}
        self._real_cache: Dict[float, Any] = {}

    # -- helpers
    def real(self, v):
        from fractions import Fraction

        key = float(v)
        hit = self._real_cache.get(key)
        if hit is None:
            f = Fraction(key)
            hit = self._real_cache[key] = self.z3.RealVal(
                f"{f.numerator}/{f.denominator}"
            )
        return hit

    def lift(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype.kind not in "fiu":
            raise _Skip(f"constant of dtype {arr.dtype} is not numeric")
        out = np.empty(arr.shape, dtype=object)
        flat = out.reshape(-1)
        for k, v in enumerate(arr.reshape(-1)):
            flat[k] = self.real(v)
        return out

    def spend(self, n: int):
        self.work += int(n)
        if self.work > self.max_work:
            raise _TooLarge(f"window too large (> {self.max_work} multiply-adds)")

    def concrete(self, i: int) -> np.ndarray:
        d = self.space.defs[i]
        if d.kind != "const":
            raise _Skip(
                f"{d.name or 'tensor'} operand must be a constant for this encoding"
            )
        return d.value

    def leaf(self, i: int) -> np.ndarray:
        if i in self.leaves:
            return self.leaves[i]
        shape = self.shape_override.get(i, self.space.shape.get(i))
        if shape is None:
            raise _Skip("unknown or symbolic shape at a cut point")
        if self.space.dtype.get(i) not in (
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
        ):
            raise _Skip("non-float tensor at a cut point")
        n = int(np.prod(shape, dtype=np.int64))
        self.spend(n)
        z3 = self.z3
        vs = np.empty(n, dtype=object)
        d = self.space.defs[i]
        rng = self.input_ranges.get(d.name) if d.kind == "input" else None
        for k in range(n):
            vs[k] = z3.Real(f"v{i}_{k}")
        arr = vs.reshape(shape)
        if rng is not None:
            self.has_bounds = True
            lo, hi = (
                np.broadcast_to(np.asarray(b, dtype=np.float64), shape) for b in rng
            )
            for v, a, b in zip(arr.reshape(-1), lo.reshape(-1), hi.reshape(-1)):
                self.constraints += [v >= self.real(a), v <= self.real(b)]
        self.leaves[i] = arr
        return arr

    # -- evaluation
    def value(self, i: int) -> np.ndarray:
        if i in self.memo:
            return self.memo[i]
        d = self.space.defs[i]
        if d.kind == "const":
            r = self.lift(d.value)
        elif i in self.cut or d.kind == "input":
            r = self.leaf(i)
        else:
            r = self.op(d)
        self.memo[i] = r
        return r

    def op(self, d: _Def) -> np.ndarray:
        if d.domain not in ("", "ai.onnx"):
            raise _Skip(f"unsupported domain {d.domain!r}")
        fn = getattr(self, "op_" + d.name, None)
        if fn is None:
            raise _Skip(f"unsupported op {d.name}")
        if d.out_index != 0:
            raise _Skip(f"{d.name} output #{d.out_index}")
        return fn(d)

    def ins(self, d: _Def) -> List[np.ndarray]:
        return [self.value(i) for i in d.inputs if i >= 0]

    def attr(self, d: _Def, name, default=None):
        if d.node is None:
            return default
        for a in d.node.attribute:
            if a.name == name:
                return onnx.helper.get_attribute_value(a)
        return default

    def _bcast_work(self, *arrs):
        self.spend(max(a.size for a in arrs))

    def op_Add(self, d):
        a, b = self.ins(d)
        self._bcast_work(a, b)
        return a + b

    def op_Sub(self, d):
        a, b = self.ins(d)
        self._bcast_work(a, b)
        return a - b

    def op_Mul(self, d):
        a, b = self.ins(d)
        self._bcast_work(a, b)
        return a * b

    def op_Neg(self, d):
        (a,) = self.ins(d)
        return -a

    def op_Identity(self, d):
        (a,) = self.ins(d)
        return a

    def op_Relu(self, d):
        (a,) = self.ins(d)
        self.spend(a.size)
        z3, zero = self.z3, self.zero
        out = np.empty(a.shape, dtype=object)
        for k, e in enumerate(a.reshape(-1)):
            out.reshape(-1)[k] = z3.If(e > 0, e, zero)
        return out

    def op_Clip(self, d):
        a = self.value(d.inputs[0])
        lo = (
            self.concrete(d.inputs[1])
            if len(d.inputs) > 1 and d.inputs[1] >= 0
            else None
        )
        hi = (
            self.concrete(d.inputs[2])
            if len(d.inputs) > 2 and d.inputs[2] >= 0
            else None
        )
        if (lo is not None and lo.size != 1) or (hi is not None and hi.size != 1):
            raise _Skip("Clip with non-scalar bounds")
        self.spend(a.size)
        z3 = self.z3
        out = np.empty(a.shape, dtype=object)
        for k, e in enumerate(a.reshape(-1)):
            if lo is not None:
                e = z3.If(
                    e < self.real(lo.reshape(-1)[0]), self.real(lo.reshape(-1)[0]), e
                )
            if hi is not None:
                e = z3.If(
                    e > self.real(hi.reshape(-1)[0]), self.real(hi.reshape(-1)[0]), e
                )
            out.reshape(-1)[k] = e
        return out

    def op_Transpose(self, d):
        (a,) = self.ins(d)
        perm = self.attr(d, "perm")
        return np.transpose(a, perm) if perm else np.transpose(a)

    def op_Flatten(self, d):
        (a,) = self.ins(d)
        axis = self.attr(d, "axis", 1)
        axis = axis + a.ndim if axis < 0 else axis
        return a.reshape(int(np.prod(a.shape[:axis], dtype=np.int64)), -1)

    def op_Reshape(self, d):
        a = self.value(d.inputs[0])
        shape = [int(s) for s in self.concrete(d.inputs[1])]
        if self.attr(d, "allowzero", 0):
            raise _Skip("Reshape allowzero=1")
        shape = [a.shape[k] if s == 0 else s for k, s in enumerate(shape)]
        return a.reshape(shape)

    def _matmul(self, a, b):
        self.spend(a.size * (b.shape[-1] if b.ndim > 1 else 1))
        if a.ndim == 0 or b.ndim == 0:
            raise _Skip("MatMul with scalar")
        return np.matmul(a, b)

    def op_MatMul(self, d):
        a, b = self.ins(d)
        return self._matmul(a, b)

    def op_Gemm(self, d):
        ins = self.ins(d)
        a, b = ins[0], ins[1]
        if self.attr(d, "transA", 0):
            a = a.T
        if self.attr(d, "transB", 0):
            b = b.T
        y = self._matmul(a, b)
        alpha, beta = self.attr(d, "alpha", 1.0), self.attr(d, "beta", 1.0)
        if alpha != 1.0:
            y = y * self.real(alpha)
        if len(ins) > 2:
            c = ins[2] if beta == 1.0 else ins[2] * self.real(beta)
            y = y + c
        return y

    def op_BatchNormalization(self, d):
        if self.attr(d, "training_mode", 0):
            raise _Skip("BatchNormalization training_mode=1")
        if d.node is not None and len(d.node.output) > 1 and any(d.node.output[1:]):
            raise _Skip("BatchNormalization training outputs")
        x = self.value(d.inputs[0])
        scale, bias, mean, var = (
            self.concrete(i).astype(np.float64) for i in d.inputs[1:5]
        )
        eps = float(self.attr(d, "epsilon", 1e-5))
        s = scale / np.sqrt(var + eps)
        shape = [1, -1] + [1] * (x.ndim - 2)
        s_l, b_l, m_l = (self.lift(v.reshape(shape)) for v in (s, bias, mean))
        self.spend(x.size)
        return (x - m_l) * s_l + b_l

    def op_Conv(self, d):
        ins = self.ins(d)
        x, w = ins[0], ins[1]
        bias = ins[2] if len(ins) > 2 else None
        if self.attr(d, "auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            raise _Skip("Conv auto_pad")
        nd = x.ndim - 2
        strides = list(self.attr(d, "strides", [1] * nd))
        dil = list(self.attr(d, "dilations", [1] * nd))
        pads = list(self.attr(d, "pads", [0] * (2 * nd)))
        group = int(self.attr(d, "group", 1))
        if nd != 2:
            raise _Skip(f"Conv with {nd} spatial dims")
        n, c, h, wd = x.shape
        m, cg, kh, kw = w.shape
        xp = np.full(
            (n, c, h + pads[0] + pads[2], wd + pads[1] + pads[3]),
            self.zero,
            dtype=object,
        )
        xp[:, :, pads[0] : pads[0] + h, pads[1] : pads[1] + wd] = x
        oh = (xp.shape[2] - dil[0] * (kh - 1) - 1) // strides[0] + 1
        ow = (xp.shape[3] - dil[1] * (kw - 1) - 1) // strides[1] + 1
        self.spend(n * m * oh * ow * cg * kh * kw)
        z3 = self.z3
        out = np.empty((n, m, oh, ow), dtype=object)
        mg = m // group
        for b_ in range(n):
            for mo in range(m):
                g = mo // mg
                for i in range(oh):
                    for j in range(ow):
                        terms = []
                        for cc in range(cg):
                            for ki in range(kh):
                                for kj in range(kw):
                                    terms.append(
                                        w[mo, cc, ki, kj]
                                        * xp[
                                            b_,
                                            g * cg + cc,
                                            i * strides[0] + ki * dil[0],
                                            j * strides[1] + kj * dil[1],
                                        ]
                                    )
                        acc = z3.Sum(terms)
                        out[b_, mo, i, j] = acc + bias[mo] if bias is not None else acc
        return out


# --------------------------------------------------------------------------
# Proof driver
# --------------------------------------------------------------------------


def _cone(space: _Space, root: int) -> set:
    seen, stack = set(), [root]
    while stack:
        i = stack.pop()
        if i in seen or i < 0:
            continue
        seen.add(i)
        stack.extend(space.defs[i].inputs)
    return seen


# --------------------------------------------------------------------------
# Spatial shrink of windows that are too large to encode at full size
# --------------------------------------------------------------------------
#
# Why this is sound. For a window built only from spatially local ops (below),
# one output position depends on a bounded *patch* of input pixels. Take the
# patch of output position ``p`` along one spatial axis, in input pixels:
# ``[p*S - lead, p*S + tail)`` with ``S`` the product of strides, ``lead`` the
# padding offset and ``tail`` the far extent. Three kinds of position exist:
#
# * top-affected    (``p*S - lead < 0``): the patch touches top padding;
# * bottom-affected (``p*S + tail > N``): the patch touches bottom padding;
# * interior        (neither): no padded cell is involved at any layer, so every
#                   interior position is the *same* function of its own, free,
#                   patch variables (one output step = ``S`` pixels).
#
# A top-affected position's function depends only on its index ``p`` (the top is
# anchored at 0). A bottom-affected position's function depends only on how far
# it is from the bottom, i.e. on its offset from the output length ``O``. So if
# a smaller extent ``N' = N - k*S`` (the same residue mod ``S``, which keeps the
# bottom pattern a rigid translation) has *no* position that is both top- and
# bottom-affected, and at least one interior position, then the set of distinct
# position functions at ``N'`` equals the set at ``N``, and each full-size
# position has a twin with an identical symbolic expression. The tolerance test
# ``|a - b| <= atol + rtol*|b|`` is pointwise, so proving it for every position
# at ``N'`` proves it for every position at ``N``. Batch is shrunk to 1 (every
# allowed op is independent across the batch); channels are untouched.
#
# What it requires, and refuses otherwise (``_NoShrink``): every op is Conv (2-D),
# BatchNormalization, Add/Sub/Mul, Neg, Identity, Relu or Clip; every constant
# that meets a spatial tensor is uniform over batch and space (so Conv weights
# and BN parameters, which are per-channel, are fine, a positional bias is not);
# every leaf has the same rank-4 spatial size; input ranges are uniform over
# batch and space. The argument is axis-separable, so H and W shrink independently.

_LOCAL_UNARY = {"Identity", "Relu", "Clip", "Neg"}
_LOCAL_BINARY = {"Add", "Sub", "Mul"}


@dataclasses.dataclass(frozen=True)
class _Axis:
    """Patch geometry along one spatial axis, in input pixels (see above)."""

    stride: int
    lead: int
    tail: int


@dataclasses.dataclass
class _Reduction:
    shapes: Dict[int, Tuple[int, ...]]  # leaf id -> reduced shape
    full_hw: Tuple[int, int]
    reduced_hw: Tuple[int, int]
    axes: Tuple[_Axis, _Axis]  # merged geometry of the two roots
    leaves: List[int]

    def describe(self) -> str:
        f, r = self.full_hw, self.reduced_hw
        return f"{f[0]}x{f[1]} -> {r[0]}x{r[1]} spatial"


@dataclasses.dataclass
class _Result:
    """Outcome of one solver run over a window (``status``: unsat / sat / skipped)."""

    status: str
    detail: str = ""
    cex: Optional[Dict[str, np.ndarray]] = None
    cex_by_id: Dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    only_inputs: bool = True  # the witness mentions graph inputs only
    too_large: bool = False  # skipped because of ``max_work``
    bad_index: Optional[Tuple[int, ...]] = (
        None  # output element that violates the tolerance
    )
    reduction: Optional[_Reduction] = None  # set when solved at a reduced spatial size


def _uniform_const(shape: Tuple[int, ...]) -> bool:
    if len(shape) > 4:
        return False
    padded = (1,) * (4 - len(shape)) + tuple(shape)
    return padded[0] == 1 and padded[2] == 1 and padded[3] == 1


def _node_attrs(d: "_Def") -> Dict[str, Any]:
    if d.node is None:
        return {}
    return {x.name: onnx.helper.get_attribute_value(x) for x in d.node.attribute}


def _plan_reduction(
    space: "_Space", a: int, b: int, cut: set, input_ranges
) -> _Reduction:
    """Find the smallest spatial extent that is provably equivalent (see above)."""
    defs = space.defs
    leaves: List[int] = []
    memo: Dict[int, Tuple[_Axis, _Axis]] = {}

    def is_const(i: int) -> bool:
        return defs[i].kind == "const"

    def geo(i: int) -> Tuple[_Axis, _Axis]:
        if i in memo:
            return memo[i]
        d = defs[i]
        if d.kind == "const":
            raise _NoShrink("a constant is used as a spatial tensor")
        ins = [x for x in d.inputs if x >= 0]
        if d.kind == "input" or i in cut:
            if i not in leaves:
                leaves.append(i)
            g = (_Axis(1, 0, 1), _Axis(1, 0, 1))
        elif d.domain not in ("", "ai.onnx") or d.out_index != 0:
            raise _NoShrink(f"{d.name} is not a spatially local op")
        elif d.name == "Conv":
            if (
                len(ins) < 2
                or is_const(ins[0])
                or not all(is_const(x) for x in ins[1:])
            ):
                raise _NoShrink("Conv with non-constant weights")
            w = defs[ins[1]].value
            if w is None or w.ndim != 4:
                raise _NoShrink("only 2-D Conv can be shrunk")
            at = _node_attrs(d)
            if at.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
                raise _NoShrink("Conv auto_pad")
            strides = list(at.get("strides", [1, 1]))
            dil = list(at.get("dilations", [1, 1]))
            pads = list(at.get("pads", [0, 0, 0, 0]))
            if min(pads) < 0:
                raise _NoShrink("negative Conv padding")
            gx = geo(ins[0])
            g = (
                _Axis(
                    gx[0].stride * strides[0],
                    gx[0].lead + pads[0] * gx[0].stride,
                    gx[0].tail + ((w.shape[2] - 1) * dil[0] - pads[0]) * gx[0].stride,
                ),
                _Axis(
                    gx[1].stride * strides[1],
                    gx[1].lead + pads[1] * gx[1].stride,
                    gx[1].tail + ((w.shape[3] - 1) * dil[1] - pads[1]) * gx[1].stride,
                ),
            )
        elif d.name == "BatchNormalization" or d.name in _LOCAL_UNARY:
            if is_const(ins[0]) or not all(is_const(x) for x in ins[1:]):
                raise _NoShrink(f"{d.name} with a non-constant extra operand")
            g = geo(ins[0])
        elif d.name in _LOCAL_BINARY:
            if len(ins) != 2:
                raise _NoShrink(f"{d.name} arity")
            live = [x for x in ins if not is_const(x)]
            if not live:
                raise _NoShrink(f"{d.name} of two constants")
            for x in ins:
                cval = defs[x].value
                if cval is not None:
                    shp = tuple(cval.shape)
                    if not _uniform_const(shp):
                        raise _NoShrink(
                            f"{d.name} constant of shape {shp} varies over space or batch"
                        )
            gs = [geo(x) for x in live]
            for ax in range(2):
                if len({gg[ax].stride for gg in gs}) > 1:
                    raise _NoShrink(f"{d.name} joins branches with different strides")
            g = (
                _Axis(
                    gs[0][0].stride,
                    max(gg[0].lead for gg in gs),
                    max(gg[0].tail for gg in gs),
                ),
                _Axis(
                    gs[0][1].stride,
                    max(gg[1].lead for gg in gs),
                    max(gg[1].tail for gg in gs),
                ),
            )
        else:
            raise _NoShrink(f"{d.name} is not a spatially local op")
        memo[i] = g
        return g

    def slen(i: int, n: int, ax: int, m: Dict[int, int]) -> int:
        """Spatial length of tensor ``i`` along ``ax`` when the leaves have extent ``n``."""
        if i in m:
            return m[i]
        d = defs[i]
        if d.kind == "input" or i in cut:
            r = n
        elif d.name == "Conv":
            ins = [x for x in d.inputs if x >= 0]
            at = _node_attrs(d)
            w = defs[ins[1]].value
            if w is None:
                raise _NoShrink("Conv with non-constant weights")
            st = list(at.get("strides", [1, 1]))[ax]
            dl = list(at.get("dilations", [1, 1]))[ax]
            pads = list(at.get("pads", [0, 0, 0, 0]))
            span = slen(ins[0], n, ax, m) + pads[ax] + pads[2 + ax]
            r = (span - dl * (w.shape[2 + ax] - 1) - 1) // st + 1
        else:
            lives = [slen(x, n, ax, m) for x in d.inputs if x >= 0 and not is_const(x)]
            if not lives or len(set(lives)) != 1:
                raise _NoShrink("operands of an elementwise op differ in spatial size")
            r = lives[0]
        m[i] = r
        return r

    ga, gb = geo(a), geo(b)
    axes: List[_Axis] = []
    for ax in range(2):
        if ga[ax].stride != gb[ax].stride:
            raise _NoShrink("the two sides have different total strides")
        axes.append(
            _Axis(
                ga[ax].stride,
                max(ga[ax].lead, gb[ax].lead),
                max(ga[ax].tail, gb[ax].tail),
            )
        )
    shapes = {i: space.shape.get(i) for i in leaves}
    if not leaves or any(sh is None or len(sh) != 4 for sh in shapes.values()):
        raise _NoShrink("leaves must be rank-4 NCHW tensors with known shapes")
    hw = {tuple(sh[2:]) for sh in shapes.values() if sh is not None}
    if len(hw) != 1:
        raise _NoShrink("leaves have different spatial sizes")
    full_hw = next(iter(hw))
    for i in leaves:
        d = defs[i]
        rng = input_ranges.get(d.name) if d.kind == "input" and input_ranges else None
        if rng is not None and not all(_uniform_const(tuple(np.shape(x))) for x in rng):
            raise _NoShrink(f"input range for {d.name!r} varies over space or batch")

    def classify(ax: int, n: int) -> Optional[Tuple[int, int, int]]:
        """``(output length, #top, #bottom)`` at extent ``n``, or ``None`` if not shrink-safe."""
        g = axes[ax]
        la, lb = slen(a, n, ax, {}), slen(b, n, ax, {})
        if la != lb or la < 1:
            return None
        top = bottom = interior = 0
        for p in range(la):
            t, bt = p * g.stride - g.lead < 0, p * g.stride + g.tail > n
            if t and bt:
                return None  # a position the full-size image would not have
            top += t
            bottom += bt
            interior += not (t or bt)
        return (la, top, bottom) if interior >= 1 else None

    reduced: List[int] = []
    for ax in range(2):
        n_full = full_hw[ax]
        full = classify(ax, n_full)
        if full is None:
            raise _NoShrink("the full-size window has no interior position to preserve")
        known = space.shape.get(a)
        if known is not None and len(known) == 4 and known[2 + ax] != full[0]:
            raise _NoShrink("output size disagrees with the shape-propagation formula")
        best = n_full
        for k in range(1, max(0, (n_full - 1) // axes[ax].stride) + 1):
            n_try = n_full - k * axes[ax].stride
            if n_try < 1:
                break
            c = classify(ax, n_try)
            if c is not None and c[1] == full[1] and c[2] == full[2]:
                best = n_try
        if best == n_full:
            raise _NoShrink("no smaller extent preserves the border structure")
        reduced.append(best)
    out_shapes: Dict[int, Tuple[int, ...]] = {}
    for i in leaves:
        sh = shapes[i]
        assert sh is not None
        out_shapes[i] = (1, sh[1], reduced[0], reduced[1])
    return _Reduction(
        out_shapes,
        (full_hw[0], full_hw[1]),
        (reduced[0], reduced[1]),
        (axes[0], axes[1]),
        leaves,
    )


class _Prover:
    def __init__(
        self,
        space,
        names,
        input_ranges,
        atol,
        rtol,
        max_work,
        timeout_ms,
        deadline=None,
        replay=None,
    ):
        self.space, self.names = space, names
        self.input_ranges, self.atol, self.rtol = input_ranges, atol, rtol
        self.max_work, self.timeout_ms = max_work, timeout_ms
        self.deadline = deadline
        # ``replay(a, b, leaf_values) -> Optional[bool]``: re-run the window on the
        # original models at full size (see ``_make_replay``); ``None`` disables it.
        self.replay = replay
        self.memo: Dict[Tuple[int, int], Window] = {}
        self.windows: List[Window] = []

    def label(self, i):
        return self.names.get(i, f"#{i}")

    def prove(self, a: int, b: int) -> Window:
        if (a, b) in self.memo:
            return self.memo[(a, b)]
        w = self._prove(a, b)
        self.memo[(a, b)] = w
        return w

    def _prove(self, a: int, b: int) -> Window:
        def mk(status, detail="", cex=None):
            return Window(self.label(a), self.label(b), status, detail, cex)

        if a == b:
            return mk(PROVED_STRUCTURAL)
        da, db = self.space.defs[a], self.space.defs[b]
        if (
            da.kind == db.kind == "op"
            and (da.domain, da.name, da.attrs, da.out_index)
            == (db.domain, db.name, db.attrs, db.out_index)
            and da.name not in _NONDETERMINISTIC
            and len(da.inputs) == len(db.inputs)
        ):
            subs = [
                self.prove(x, y)
                for x, y in zip(da.inputs, db.inputs)
                if x >= 0 and y >= 0
            ]
            if all(s.status.startswith("proved") for s in subs) and all(
                (x < 0) == (y < 0) for x, y in zip(da.inputs, db.inputs)
            ):
                w = mk(PROVED_CONGRUENCE, f"same {da.name}, inputs proved equal")
                w.children = subs
                return w
            if not any(x.status == REFUTED for x in subs):
                # Nothing refuted, nothing proved: the whole-window encoding
                # contains the failed sub-window, so retrying it only repeats it.
                bad = next((x for x in subs if not x.status.startswith("proved")), None)
                if bad is not None and all(
                    (x < 0) == (y < 0) for x, y in zip(da.inputs, db.inputs)
                ):
                    w = mk(
                        SKIPPED, f"input {bad.orig} ~ {bad.simplified}: {bad.detail}"
                    )
                    w.children = subs
                    return w
        return self._smt(a, b, mk)

    def _smt(self, a, b, mk) -> Window:
        if self.deadline is not None and time.monotonic() > self.deadline:
            return mk(SKIPPED, "certify time budget exhausted")
        ca, cb = _cone(self.space, a), _cone(self.space, b)
        # Shared non-constant tensors are cut points: proven equal by hashing,
        # so they become one shared free variable instead of being re-encoded.
        # Proofs under cuts are sound (the variable over-approximates the real
        # tensor), but a *counterexample* under cuts may not be reachable from
        # any real input -- so a cut-level witness is re-checked uncut below.
        cut = {i for i in ca & cb if self.space.defs[i].kind == "op"}
        res = self._attempt(a, b, cut)
        if res.status == "sat" and not res.only_inputs:
            res = self._attempt(a, b, set())
            if res.status == "skipped":
                res.detail = (
                    "a difference exists at an intermediate tensor shared by both models, but it "
                    f"could not be confirmed at the graph inputs ({res.detail})"
                )
        if res.status == "unsat":
            if res.reduction is not None:
                return mk(
                    PROVED_REDUCED,
                    f"{res.detail}; encoded at a reduced shape ({res.reduction.describe()}), "
                    "all ops spatially local -- see the certify module notes on shrinking",
                )
            return mk(PROVED_SMT, res.detail)
        if res.status == "sat":
            if res.reduction is not None:
                return self._confirm_reduced(a, b, res, mk)
            return mk(REFUTED, res.detail, res.cex)
        return mk(SKIPPED, res.detail)

    def _estimate_work(self, a, b, cut) -> int:
        """Multiply-adds the encoder would spend, from shapes alone (never over-estimates).

        Mirrors ``_Encoder``'s own accounting for the ops it counts, and counts 0 for
        anything it cannot size, so ``estimate > max_work`` implies the full-size
        encoding would have been rejected too -- without paying to build it first.
        """
        space = self.space
        total = 0
        for i in _cone(space, a) | _cone(space, b):
            d = space.defs[i]
            shape = space.shape.get(i)
            if d.kind == "const" or shape is None:
                continue
            size = int(np.prod(shape, dtype=np.int64))
            if d.kind == "input" or i in cut:
                total += size
            elif d.name == "Conv":
                w = space.defs[d.inputs[1]].value if len(d.inputs) > 1 else None
                if w is not None and w.ndim == 4:
                    total += size * int(w.shape[1] * w.shape[2] * w.shape[3])
            elif d.name in ("Add", "Sub", "Mul", "Relu", "Clip", "BatchNormalization"):
                total += size
        return total

    def _attempt(self, a, b, cut) -> "_Result":
        """Solve at full size; if (only) the size is the obstacle, retry spatially shrunk."""
        if self._estimate_work(a, b, cut) > self.max_work:
            res = _Result(
                "skipped",
                f"window too large (> {self.max_work} multiply-adds)",
                too_large=True,
            )
        else:
            res = self._solve(a, b, cut)
        if not (res.status == "skipped" and res.too_large):
            return res
        try:
            reduction = _plan_reduction(self.space, a, b, cut, self.input_ranges)
        except _NoShrink as why:
            res.detail += f"; not shrinkable: {why}"
            return res
        shrunk = self._solve(a, b, cut, reduction)
        if shrunk.status == "skipped":
            shrunk.detail = (
                f"{shrunk.detail} (even after shrinking to {reduction.describe()})"
            )
        return shrunk

    def _lift(
        self, red: "_Reduction", res: "_Result"
    ) -> Optional[Dict[int, np.ndarray]]:
        """Embed a reduced-shape counterexample into the full shape (class-preserving).

        The violating output position keeps its border class: a bottom-affected
        position is moved down by ``N - N'``, others stay put (see the shrink notes).
        Only the position's patch is copied; the rest is filled with a value inside
        each input's range (0 where there is none), so the lifted input stays in the box.
        """
        if res.bad_index is None or len(res.bad_index) != 4:
            return None
        lifted: Dict[int, np.ndarray] = {}
        for i in red.leaves:
            small = res.cex_by_id.get(i)
            full_shape = self.space.shape.get(i)
            if small is None or full_shape is None:
                return None
            d = self.space.defs[i]
            fill = 0.0
            rng = self.input_ranges.get(d.name) if d.kind == "input" else None
            if rng is not None:
                lo = float(np.min(np.asarray(rng[0], dtype=np.float64)))
                hi = float(np.max(np.asarray(rng[1], dtype=np.float64)))
                fill = min(max(0.0, lo), hi)
            big = np.full(full_shape, fill, dtype=np.float64)
            spans = []
            for ax, p in enumerate(res.bad_index[2:]):
                g = red.axes[ax]
                n_small, n_full = red.reduced_hw[ax], red.full_hw[ax]
                lo_px = max(0, p * g.stride - g.lead)
                hi_px = min(n_small, p * g.stride + g.tail)
                shift = n_full - n_small if p * g.stride + g.tail > n_small else 0
                spans.append((lo_px, hi_px, shift))
            (h0, h1, hs), (w0, w1, ws) = spans
            big[0, :, h0 + hs : h1 + hs, w0 + ws : w1 + ws] = small[0, :, h0:h1, w0:w1]
            lifted[i] = big
        return lifted

    def _confirm_reduced(self, a, b, res: "_Result", mk) -> Window:
        """A difference found at a reduced shape: lift, replay on the real models, then report.

        The verdict does not depend on the replay (the shrink argument already says the
        full-size window has the same position functions), but a false "refuted" is a
        loud warning in ``simplify``, so a replay that *disagrees* downgrades to
        ``skipped`` instead of being ignored.
        """
        red = res.reduction
        assert red is not None
        lifted = self._lift(red, res)
        cex = res.cex
        where = f"found at a reduced shape ({red.describe()})"
        if lifted is None:
            return mk(
                REFUTED,
                f"{res.detail}; {where}, no full-size counterexample could be built",
                cex,
            )
        by_name = {
            (
                self.space.defs[i].name
                if self.space.defs[i].kind == "input"
                else f"tensor#{i}"
            ): v
            for i, v in lifted.items()
        }
        confirmed = None if self.replay is None else self.replay(a, b, lifted)
        if confirmed is True:
            return mk(
                REFUTED,
                f"{res.detail}; {where}, lifted to full size and confirmed on the original models",
                by_name,
            )
        if confirmed is False:
            return mk(
                SKIPPED,
                f"a difference was found at a reduced shape ({red.describe()}) but did not reproduce "
                "when replayed on the original models at full size (margin below float32 rounding?)",
            )
        return mk(
            REFUTED,
            f"{res.detail}; {where}, lifted to full size but not replayed (onnxruntime unavailable "
            "or the window could not be extracted)",
            by_name,
        )

    def _solve(self, a, b, cut, reduction: Optional[_Reduction] = None) -> "_Result":
        """Solve one window, optionally with its leaves shrunk to ``reduction.shapes``."""
        z3 = _z3()
        enc = _Encoder(self.space, self.input_ranges, self.max_work)
        enc.cut = cut
        if reduction is not None:
            enc.shape_override = dict(reduction.shapes)
        try:
            va, vb = enc.value(a), enc.value(b)
            if va.shape != vb.shape:
                return _Result("sat", f"output shapes differ: {va.shape} vs {vb.shape}")
            s = z3.Solver()
            s.add(*enc.constraints)
            atol, rtol = enc.real(self.atol), enc.real(self.rtol)
            # One query per output element, each with its own share of the
            # time budget: a single big Or over every element is far harder
            # for the solver than the same facts checked one at a time.
            n = va.size
            budget = self.timeout_ms
            if self.deadline is not None:
                budget = min(
                    budget, max(500, int((self.deadline - time.monotonic()) * 1000))
                )
            s.set("timeout", max(500, budget // max(1, min(n, 8))))
            r = z3.unsat
            bad_flat = -1
            for k, (x, y) in enumerate(zip(va.reshape(-1), vb.reshape(-1))):
                diff = x - y
                ady = z3.If(y >= 0, y, -y)
                s.push()
                s.add(z3.If(diff >= 0, diff, -diff) > atol + rtol * ady)
                r = s.check()
                if r != z3.unsat:
                    bad_flat = k
                    break
                s.pop()
        except _TooLarge as e:
            return _Result("skipped", str(e), too_large=True)
        except _Skip as e:
            return _Result("skipped", str(e))
        if r == z3.unsat:
            return _Result(
                "unsat",
                f"{va.size} elements, {enc.work} multiply-adds",
                reduction=reduction,
            )
        if r == z3.sat:
            m = s.model()
            cex: Dict[str, np.ndarray] = {}
            by_id: Dict[int, np.ndarray] = {}
            only_inputs = True
            for i, arr in enc.leaves.items():
                d = self.space.defs[i]
                only_inputs &= d.kind == "input"
                vals = np.array(
                    [
                        float(m.eval(v, model_completion=True).as_fraction())
                        for v in arr.reshape(-1)
                    ]
                ).reshape(arr.shape)
                cex[d.name if d.kind == "input" else f"tensor#{i}"] = vals
                by_id[i] = vals
            bad_index = (
                tuple(int(x) for x in np.unravel_index(bad_flat, va.shape))
                if bad_flat >= 0
                else None
            )
            big = max((float(np.abs(v).max()) for v in cex.values()), default=0.0)
            note = "solver found an input where the two sides differ"
            if not enc.has_bounds:
                note += (
                    f" (counterexample magnitude {big:.3g}; {NEEDS_RANGES_HINT}, so this may be"
                    " float rounding of re-computed constants amplified by a huge input -- give"
                    " input_ranges to certify over the realistic domain)"
                )
            return _Result(
                "sat",
                note,
                cex,
                by_id,
                only_inputs,
                bad_index=bad_index,
                reduction=reduction,
            )
        hint = (
            "" if enc.has_bounds else "; no input_ranges given, so inputs are unbounded"
        )
        return _Result("skipped", f"solver gave up ({s.reason_unknown()}){hint}")


def _make_replay(orig, simplified, ta, tb, space, atol, rtol):
    """Build ``replay(a, b, leaf_values)``: run the window on both real models at full size.

    Extracts each side's sub-graph (leaf tensors -> root tensor) with
    ``onnx.utils.Extractor`` and runs it in onnxruntime on the lifted leaf values.
    Returns ``True`` if the roots differ beyond the tolerance, ``False`` if they do
    not, and ``None`` when it cannot run (no onnxruntime, extraction failed, ...).
    """
    if importlib.util.find_spec("onnxruntime") is None:
        return None
    inv_a: Dict[int, str] = {}
    for n, i in ta.items():
        inv_a.setdefault(i, n)
    inv_b: Dict[int, str] = {}
    for n, i in tb.items():
        inv_b.setdefault(i, n)

    def replay(a: int, b: int, lifted: Dict[int, np.ndarray]) -> Optional[bool]:
        try:
            import onnx.utils
            import onnxruntime as ort

            outs = []
            for model, inv, root in ((orig, inv_a, a), (simplified, inv_b, b)):
                cone = _cone(space, root)
                ids = [i for i in lifted if i in cone]
                names = [inv[i] for i in ids]
                sub = onnx.utils.Extractor(model).extract_model(names, [inv[root]])
                sess = ort.InferenceSession(
                    sub.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                feeds = {}
                for i, n in zip(ids, names):
                    dt = (
                        np.float64
                        if space.dtype.get(i) == onnx.TensorProto.DOUBLE
                        else np.float32
                    )
                    feeds[n] = lifted[i].astype(dt)
                outs.append(np.asarray(sess.run(None, feeds)[0], dtype=np.float64))
            oa, ob = outs
            if oa.shape != ob.shape:
                return True
            return bool(np.any(np.abs(oa - ob) > atol + rtol * np.abs(ob)))
        except Exception:
            return None

    return replay


# Upper bound, in float64 elements, on what the zonotope fallback may allocate (256 MB estimated;
# measured peak is about 1.4x the estimate: 28.6M estimated -> ~330 MB extra RSS, 0.5 s). Zonotope
# generators are dense (symbols x tensor size) and the evaluator keeps every tensor, so memory,
# not time, is the limit. It only runs for outputs the SMT steps already gave up on.
_ZONOTOPE_MAX_ELEMENTS = 32_000_000
_ZONOTOPE_NONLINEAR = frozenset({"Relu", "Sigmoid", "Tanh"})


def _without_range_annotations(model: onnx.ModelProto) -> onnx.ModelProto:
    """``model`` minus its ``onnxsim.range.*`` annotations (a copy only if it has any).

    ``zonotope`` merges a model's own annotations over the ``input_ranges`` it is given, while
    ``certify``'s documented rule is "inputs not listed in ``input_ranges`` are unbounded".
    Stripping keeps the zonotope step from proving more than the Z3 steps were allowed to assume.
    """
    prefix = "onnxsim.range."
    if not any(p.key.startswith(prefix) for p in model.metadata_props):
        return model
    out = onnx.ModelProto()
    out.CopyFrom(model)
    keep = [
        (p.key, p.value) for p in out.metadata_props if not p.key.startswith(prefix)
    ]
    del out.metadata_props[:]
    for k, v in keep:
        entry = out.metadata_props.add()
        entry.key, entry.value = k, v
    return out


def _zonotope_cost(
    orig: onnx.ModelProto, simplified: onnx.ModelProto, n_in: int, cap: int
) -> int:
    """Rough float64-element count the zonotope evaluation would hold, from shapes alone.

    Walks both models in order: a tensor carries as many generators as were created upstream
    (the input symbols plus one fresh symbol per element of every earlier nonlinearity, in
    either model), capped as the evaluator caps them. Over-estimates rather than under.
    """
    running, total = n_in, 0
    for model in (orig, simplified):
        shapes = _shapes(model)
        for node in model.graph.node:
            for out in node.output:
                shape = shapes.get(out, (None, 0))[0] if out else None
                if shape is None:
                    continue
                size = int(np.prod(shape, dtype=np.int64)) if shape else 1
                if node.op_type in _ZONOTOPE_NONLINEAR:
                    running += size
                total += min(running, cap + size) * size
    return total


def _zonotope_check(
    orig: onnx.ModelProto,
    simplified: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]],
    atol: float,
    rtol: float,
    outputs: List[str],
    deadline: Optional[float] = None,
) -> Dict[str, Tuple[bool, str]]:
    """Try to prove each of ``outputs`` equal with zonotopes: ``{output: (proved, detail)}``.

    Sound by construction (see the module notes, step 5) and never raises; every reason for not
    attempting or not proving is returned as text.
    """

    def none(why: str) -> Dict[str, Tuple[bool, str]]:
        return {o: (False, f"zonotope not used: {why}") for o in outputs}

    try:
        from . import zonotope as _zono

        # Same rule as the Z3 steps: only the explicit ``input_ranges`` bound the inputs
        # (a model's own annotations reach here via ``simplify``, which passes them in).
        boxes: Dict[str, Tuple] = dict(input_ranges or {})
        inits = {t.name for t in orig.graph.initializer}
        n_in = 0
        for vi in orig.graph.input:
            if vi.name in inits:
                continue
            if vi.name not in boxes:
                return none(f"input {vi.name!r} has no range")
            lo, hi = (np.asarray(b, dtype=np.float64) for b in boxes[vi.name])
            if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
                return none(f"input {vi.name!r} has an unbounded range")
            dims = [
                d.dim_value if d.HasField("dim_value") else 0
                for d in vi.type.tensor_type.shape.dim
            ]
            if not dims or any(d <= 0 for d in dims):
                return none(f"input {vi.name!r} has no static shape")
            n_in += int(np.prod(dims, dtype=np.int64))
        cost = _zonotope_cost(orig, simplified, n_in, _zono.DEFAULT_MAX_SYMBOLS)
        if cost > _ZONOTOPE_MAX_ELEMENTS:
            return none(
                f"too large (about {cost / 1e6:.0f}M elements > {_ZONOTOPE_MAX_ELEMENTS / 1e6:.0f}M)"
            )
        if deadline is not None and time.monotonic() > deadline:
            return none("certify time budget exhausted")
        bound = _zono.bound_difference(
            _without_range_annotations(orig),
            _without_range_annotations(simplified),
            boxes,
        )
    except Exception as e:  # certification must never break its caller
        return none(f"failed ({type(e).__name__}: {e})")

    notes = "; ".join(bound.notes[:2])
    suffix = f" [{notes}]" if notes else ""
    verdicts: Dict[str, Tuple[bool, str]] = {}
    for o in outputs:
        d, ref = bound.max_abs.get(o), bound.ref_min_abs.get(o)
        if d is None or ref is None:
            verdicts[o] = (False, "zonotope not used: no bound for this output")
            continue
        if not np.all(np.isfinite(d)):
            verdicts[o] = (False, f"zonotope bound is unbounded{suffix}")
            continue
        worst = float(np.max(d)) if np.size(d) else 0.0
        proved = bool(np.all(d <= atol + rtol * ref))
        verdicts[o] = (
            proved,
            f"zonotope: max|orig-simplified| <= {worst:.3g} over the input box "
            f"(tolerance {atol:g} + {rtol:g}*min|simplified|){suffix}",
        )
    return verdicts


def certify(
    orig: onnx.ModelProto,
    simplified: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    max_work: int = 300_000,
    timeout_ms: int = 60_000,
    total_timeout_ms: Optional[int] = None,
) -> CertifyReport:
    """Try to prove ``simplified`` computes the same outputs as ``orig``.

    :param input_ranges: ``{input_name: (lo, hi)}`` boxes (scalars or arrays
        broadcastable to the input's shape). Inputs not listed are unbounded --
        the proof then holds for every real input.
    :param atol, rtol: the same tolerance meaning as ``simplify(check_atol=,
        check_rtol=)``: two outputs agree when ``|a - b| <= atol + rtol * |b|``.
    :param max_work: budget of multiply-adds / variables per SMT window; a
        bigger window is reported ``skipped``.
    :param timeout_ms: per-window solver timeout.
    :param total_timeout_ms: wall-clock budget for the whole call; windows reached after it
        is spent are reported ``skipped``. ``None`` means no overall limit.
    """
    space = _Space()
    deadline = (
        None
        if total_timeout_ms is None
        else time.monotonic() + total_timeout_ms / 1000.0
    )
    ta, tb = _index(space, orig, False), _index(space, simplified, False)
    in_a = {i.name for i in orig.graph.input} - {t.name for t in orig.graph.initializer}
    in_b = {i.name for i in simplified.graph.input} - {
        t.name for t in simplified.graph.initializer
    }
    if in_a != in_b:
        raise ValueError(f"graph inputs differ: {sorted(in_a)} vs {sorted(in_b)}")
    out_a = [o.name for o in orig.graph.output]
    out_b = [o.name for o in simplified.graph.output]
    if out_a != out_b:
        raise ValueError(f"graph outputs differ: {out_a} vs {out_b}")
    if any(ta[o] != tb[o] for o in out_a):
        # Something differs: only now pay for shape inference (needed by the SMT encoding).
        for model, tid in ((orig, ta), (simplified, tb)):
            for name, (shape, elem) in _shapes(model).items():
                if name in tid:
                    space.shape.setdefault(tid[name], shape)
                    space.dtype.setdefault(tid[name], elem)
    names: Dict[int, str] = {}
    for name, i in list(tb.items()) + list(ta.items()):
        names.setdefault(i, name)
    prover = _Prover(
        space,
        names,
        input_ranges,
        atol,
        rtol,
        max_work,
        timeout_ms,
        deadline,
        replay=_make_replay(orig, simplified, ta, tb, space, atol, rtol),
    )
    outputs, windows, seen = {}, [], set()
    tops: Dict[str, Window] = {o: prover.prove(ta[o], tb[o]) for o in out_a}
    for o, w in tops.items():
        outputs[o] = w.status
    # Last resort for outputs still ``skipped`` (never for ``refuted``): zonotope bounds.
    skipped = [o for o in out_a if outputs[o] == SKIPPED]
    if skipped:
        for o, (proved, detail) in _zonotope_check(
            orig, simplified, input_ranges, atol, rtol, skipped, deadline
        ).items():
            old = tops[o]
            if proved:
                tops[o] = Window(
                    o, o, PROVED_AFFINE, f"{detail} (SMT window: {old.detail})"
                )
                outputs[o] = PROVED_AFFINE
            else:
                old.detail = f"{old.detail}; {detail}" if old.detail else detail

    def collect(w: Window):
        # Only obligations the final verdict actually rests on -- not sub-proofs
        # that failed inside a congruence attempt later settled by SMT.
        if id(w) in seen:
            return
        seen.add(id(w))
        if w.status != PROVED_STRUCTURAL:
            windows.append(w)
        for c in w.children:
            collect(c)

    for o in out_a:
        collect(tops[o])
    return CertifyReport(outputs, windows)


def _concrete_eval(
    model: onnx.ModelProto, feeds: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Evaluate ``model`` through the SMT encoding on concrete inputs (test hook).

    Used by ``tests/test_certify.py`` to check the op encodings above against
    onnx's reference evaluator, since the encoding is part of what a proof trusts.
    """
    z3 = _z3()
    space = _Space()
    tid = _index(space, model)
    enc = _Encoder(space, None, 10**9)
    for name, arr in feeds.items():
        enc.leaves[tid[name]] = enc.lift(np.asarray(arr))
    out = {}
    for o in model.graph.output:
        v = enc.value(tid[o.name])
        flat = [float(z3.simplify(e).as_fraction()) for e in v.reshape(-1)]
        out[o.name] = np.array(flat).reshape(v.shape)
    return out
