"""Translation validation of a simplified model against its original, with Z3.

``certify(orig, simplified)`` tries to *prove* the two models compute the same
outputs, instead of sampling random inputs the way ``onnxsim.model_checking``
does. It needs only the two models -- no rewrite log from the C++ passes -- and
works in three layers, cheapest first:

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
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

# Statuses a window can end in. Everything but "proved-*" means certification
# of that output did not succeed.
NEEDS_RANGES_HINT = "no input_ranges given"
PROVED_STRUCTURAL = "proved-structural"
PROVED_CONGRUENCE = "proved-congruence"
PROVED_SMT = "proved-smt"
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

    # -- helpers
    def real(self, v):
        from fractions import Fraction

        f = Fraction(float(v))
        return self.z3.RealVal(f"{f.numerator}/{f.denominator}")

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
            raise _Skip(f"window too large (> {self.max_work} multiply-adds)")

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
        shape = self.space.shape.get(i)
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
    ):
        self.space, self.names = space, names
        self.input_ranges, self.atol, self.rtol = input_ranges, atol, rtol
        self.max_work, self.timeout_ms = max_work, timeout_ms
        self.deadline = deadline
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
        status, detail, cex, only_inputs = self._solve(a, b, cut)
        if status == "sat" and not only_inputs:
            status, detail, cex, only_inputs = self._solve(a, b, set())
            if status == "skipped":
                detail = (
                    "a difference exists at an intermediate tensor shared by both models, but it "
                    f"could not be confirmed at the graph inputs ({detail})"
                )
        if status == "unsat":
            return mk(PROVED_SMT, detail)
        if status == "sat":
            return mk(REFUTED, detail, cex)
        return mk(SKIPPED, detail)

    def _solve(self, a, b, cut):
        """Returns ``(status, detail, counterexample, witness_uses_only_graph_inputs)``."""
        z3 = _z3()
        enc = _Encoder(self.space, self.input_ranges, self.max_work)
        enc.cut = cut
        try:
            va, vb = enc.value(a), enc.value(b)
            if va.shape != vb.shape:
                return (
                    "sat",
                    f"output shapes differ: {va.shape} vs {vb.shape}",
                    None,
                    True,
                )
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
            for x, y in zip(va.reshape(-1), vb.reshape(-1)):
                diff = x - y
                ady = z3.If(y >= 0, y, -y)
                s.push()
                s.add(z3.If(diff >= 0, diff, -diff) > atol + rtol * ady)
                r = s.check()
                if r != z3.unsat:
                    break
                s.pop()
        except _Skip as e:
            return "skipped", str(e), None, True
        if r == z3.unsat:
            return "unsat", f"{va.size} elements, {enc.work} multiply-adds", None, True
        if r == z3.sat:
            m = s.model()
            cex, only_inputs = {}, True
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
            big = max((float(np.abs(v).max()) for v in cex.values()), default=0.0)
            note = "solver found an input where the two sides differ"
            if not enc.has_bounds:
                note += (
                    f" (counterexample magnitude {big:.3g}; {NEEDS_RANGES_HINT}, so this may be"
                    " float rounding of re-computed constants amplified by a huge input -- give"
                    " input_ranges to certify over the realistic domain)"
                )
            return "sat", note, cex, only_inputs
        hint = (
            "" if enc.has_bounds else "; no input_ranges given, so inputs are unbounded"
        )
        return "skipped", f"solver gave up ({s.reason_unknown()}){hint}", None, True


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
        space, names, input_ranges, atol, rtol, max_work, timeout_ms, deadline
    )
    outputs, windows, seen = {}, [], set()

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
        w = prover.prove(ta[o], tb[o])
        outputs[o] = w.status
        collect(w)
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
