"""VNN-LIB properties and VNN-COMP style verdicts for onnxsim's bound engines.

The neural-network-verification community states a robustness / safety question as an ONNX
network plus a ``.vnnlib`` file, and answers it with one of three verdicts. This module reads
the VNN-COMP subset of VNN-LIB and answers it with ``onnxsim.interval`` / ``zonotope`` /
``crown``, so those engines can be run on the same public benchmarks as the established
verifiers and compared with *independent* references, not only with our own sampling tests.

Convention (the VNN-COMP one): a property describes the **unsafe** region. A *counterexample*
is an input ``x`` inside the input box whose output ``y = net(x)`` satisfies the output
constraints.

* ``unsat``  -- verified safe: no input in the box produces an unsafe output.
* ``sat``    -- a counterexample was found **and replayed on onnxruntime** (never claimed from
  an over-approximation).
* ``unknown``-- neither (the bounds were too loose, the budget ran out, ...).
* ``unsupported`` -- the property or the network is outside what this module handles; it says
  why and never guesses.

Property structure. Every ``(assert F)`` is conjoined; ``F`` may nest ``and`` / ``or`` over
atoms ``(<= a b)``, ``(< a b)``, ``(>= a b)``, ``(> a b)``, ``(= a b)`` whose operands are linear
expressions built from numbers, ``X_i`` / ``Y_j`` and ``+``, ``-``, ``*`` (by a constant), ``/``
(by a constant). The result is a disjunction of *clauses*; a clause is an input box plus linear
output atoms ``a . Y <= b`` (strict ones are kept strict for replay, and treated as the weaker
non-strict ones when proving, which only makes ``unsat`` harder, i.e. stays sound).

Not supported, and rejected loudly: ``not``, ``=>``, ``xor``, any atom mixing inputs and
outputs, a constraint over several ``X`` variables (only per-variable bounds define the
box), VNN-LIB 2.0 syntax (``declare-network`` ...), non-linear terms.

Exclusion rule (derived, and what ``verify`` implements). For a clause with atoms
``a_j . Y <= b_j`` over the input box ``B``, the clause has a point iff some ``x in B`` has
``a_j . net(x) <= b_j`` for every ``j``. It is therefore *empty* (safe) as soon as for ONE
atom ``min over B of a_j . net(x) > b_j``. Given a sound lower bound ``lb_j`` the test is
``lb_j > b_j + margin`` (``margin`` a tiny float-rounding allowance). The property is
verified safe iff every clause is empty. Branch and bound applies the same test per region
of the input box (a region is decided when some atom is infeasible on it), which is
strictly stronger than testing atoms one at a time.

Soundness caveat: the bound engines enclose the *real-number* function, while onnxruntime
runs float32. An ``unsat`` whose margin to the boundary is within float32 rounding of a real
counterexample could in principle disagree with a replay; ``verify`` reports such a clash as
``inconsistent`` instead of choosing one silently.
"""

import argparse
import dataclasses
import itertools
import os
import re
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from . import crown as _crown
from . import zonotope as _zonotope

_MARGIN = (
    1e-9  # allowance (relative to max(1, |b|)) between a proven bound and the threshold
)
_MAX_CLAUSES = 200_000


class VnnlibError(ValueError):
    """The property text is malformed or outside the supported subset."""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    text = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", text)


def _sexprs(tokens: List[str]) -> List[Any]:
    out: List[Any] = []
    stack: List[List[Any]] = []
    for t in tokens:
        if t == "(":
            stack.append([])
        elif t == ")":
            if not stack:
                raise VnnlibError("unbalanced ')'")
            done = stack.pop()
            (stack[-1] if stack else out).append(done)
        else:
            (stack[-1] if stack else out).append(t)
    if stack:
        raise VnnlibError("unbalanced '('")
    return out


_VAR = re.compile(r"^([XY])_(\d+)$")
_Lin = Tuple[Dict[Tuple[str, int], float], float]


def _lin(e: Any) -> _Lin:
    """A linear expression as ``({(kind, index): coefficient}, constant)``."""
    if isinstance(e, str):
        m = _VAR.match(e)
        if m:
            return {(m.group(1), int(m.group(2))): 1.0}, 0.0
        try:
            return {}, float(e)
        except ValueError:
            raise VnnlibError(f"unknown symbol {e!r}") from None
    if not e:
        raise VnnlibError("empty expression")
    head, args = e[0], e[1:]
    if head == "+":
        coefs: Dict[Tuple[str, int], float] = {}
        const = 0.0
        for a in args:
            c, k = _lin(a)
            for v, x in c.items():
                coefs[v] = coefs.get(v, 0.0) + x
            const += k
        return coefs, const
    if head == "-":
        if len(args) == 1:
            c, k = _lin(args[0])
            return {v: -x for v, x in c.items()}, -k
        c0, k0 = _lin(args[0])
        coefs = dict(c0)
        const = k0
        for a in args[1:]:
            c, k = _lin(a)
            for v, x in c.items():
                coefs[v] = coefs.get(v, 0.0) - x
            const -= k
        return coefs, const
    if head == "*":
        rc: Dict[Tuple[str, int], float] = {}
        rk = 1.0
        for a_ in args:
            c, k = _lin(a_)
            if c and rc:
                raise VnnlibError(
                    "non-linear term: product of two variable expressions"
                )
            if c:  # variable operand times the constants gathered so far
                rc, rk = {v: x * rk for v, x in c.items()}, k * rk
            else:  # constant operand
                rc, rk = {v: x * k for v, x in rc.items()}, rk * k
        return rc, rk
    if head == "/":
        if len(args) != 2:
            raise VnnlibError("'/' takes two operands")
        c, k = _lin(args[0])
        dc, dk = _lin(args[1])
        if dc or dk == 0.0:
            raise VnnlibError("division by a non-constant or by zero")
        return {v: x / dk for v, x in c.items()}, k / dk
    raise VnnlibError(f"unsupported operator {head!r} in a linear expression")


@dataclasses.dataclass
class YAtom:
    """One output constraint ``a . Y <= b`` (``<`` when ``strict``)."""

    a: np.ndarray
    b: float
    strict: bool = False


@dataclasses.dataclass
class Clause:
    """An input box and the output atoms that must all hold for a point to be unsafe."""

    lo: np.ndarray
    hi: np.ndarray
    atoms: List[YAtom]


@dataclasses.dataclass
class Property:
    """A parsed ``.vnnlib``: the unsafe region is the union of ``clauses``."""

    n_inputs: int
    n_outputs: int
    clauses: List[Clause]

    def input_hull(self) -> Tuple[np.ndarray, np.ndarray]:
        """The smallest box containing every clause's input box."""
        if not self.clauses:
            z = np.zeros(self.n_inputs)
            return z, z
        return (
            np.min([c.lo for c in self.clauses], axis=0),
            np.max([c.hi for c in self.clauses], axis=0),
        )

    def violated_by(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Is ``(x, y)`` a counterexample? ``x`` flat, ``y`` flat; exact, float64, no slack."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        for c in self.clauses:
            if not (np.all(x >= c.lo) and np.all(x <= c.hi)):
                continue
            if all((a.a @ y < a.b) if a.strict else (a.a @ y <= a.b) for a in c.atoms):
                return True
        return False


_Clauseish = Tuple[Dict[int, List[float]], List[Tuple[Dict[int, float], float, bool]]]


def _atoms_of(e: Any) -> List[_Clauseish]:
    """DNF of a formula: a list of ``(x bounds, y atoms)`` clauses."""
    if isinstance(e, str) or not e:
        raise VnnlibError(f"expected a formula, got {e!r}")
    head, args = e[0], e[1:]
    if head == "and":
        clauses: List[_Clauseish] = [({}, [])]
        for a in args:
            sub = _atoms_of(a)
            merged: List[_Clauseish] = []
            for (xb, ya), (xb2, ya2) in itertools.product(clauses, sub):
                nb = {k: list(v) for k, v in xb.items()}
                for k, (lo, hi) in xb2.items():
                    cur = nb.setdefault(k, [-np.inf, np.inf])
                    cur[0], cur[1] = max(cur[0], lo), min(cur[1], hi)
                merged.append((nb, ya + ya2))
            clauses = merged
            if len(clauses) > _MAX_CLAUSES:
                raise VnnlibError(
                    f"more than {_MAX_CLAUSES} clauses after expanding 'and'/'or'"
                )
        return clauses
    if head == "or":
        out: List[_Clauseish] = []
        for a in args:
            out += _atoms_of(a)
        if len(out) > _MAX_CLAUSES:
            raise VnnlibError(f"more than {_MAX_CLAUSES} clauses after expanding 'or'")
        return out
    if head in ("<=", "<", ">=", ">", "="):
        if len(args) != 2:
            raise VnnlibError(f"{head!r} takes two operands")
        c1, k1 = _lin(args[0])
        c2, k2 = _lin(args[1])
        coefs: Dict[Tuple[str, int], float] = dict(c1)
        for v, x in c2.items():
            coefs[v] = coefs.get(v, 0.0) - x
        const = k1 - k2  # coefs . v + const  (op)  0
        coefs = {v: x for v, x in coefs.items() if x != 0.0}
        kinds = {k for k, _ in coefs}
        pieces: List[Tuple[Dict[Tuple[str, int], float], float, bool]] = []
        if head in ("<=", "<"):
            pieces.append((coefs, const, head == "<"))
        elif head in (">=", ">"):
            pieces.append(({v: -x for v, x in coefs.items()}, -const, head == ">"))
        else:
            pieces.append((coefs, const, False))
            pieces.append(({v: -x for v, x in coefs.items()}, -const, False))
        if not kinds:  # a constant comparison: decide it now
            if all(k <= 0.0 if not s else k < 0.0 for _, k, s in pieces):
                return [({}, [])]
            return []
        if kinds == {"X", "Y"}:
            raise VnnlibError("an atom mixing inputs and outputs is not supported")
        x_bounds: Dict[int, List[float]] = {}
        y_atoms: List[Tuple[Dict[int, float], float, bool]] = []
        for pc, pk, strict in pieces:
            if kinds == {"X"}:
                if len(pc) != 1:
                    raise VnnlibError(
                        "a constraint over several input variables is not supported "
                        "(only per-variable bounds define the box)"
                    )
                (_, idx), cf = next(iter(pc.items()))
                bound = -pk / cf  # cf * x + pk <= 0
                cur = x_bounds.setdefault(idx, [-np.inf, np.inf])
                if cf > 0:
                    cur[1] = min(cur[1], bound)
                else:
                    cur[0] = max(cur[0], bound)
            else:
                y_atoms.append(({idx: cf for (_, idx), cf in pc.items()}, pk, strict))
        return [(x_bounds, y_atoms)]
    raise VnnlibError(f"unsupported formula operator {head!r}")


def parse(text: str) -> Property:
    """Parse VNN-LIB text (the VNN-COMP subset) into a :class:`Property`."""
    nx = ny = 0
    declared = set()
    formula: List[Any] = []
    for form in _sexprs(_tokenize(text)):
        if isinstance(form, str) or not form:
            raise VnnlibError(f"unexpected top-level token {form!r}")
        if form[0] == "declare-const":
            if len(form) != 3 or form[2] != "Real":
                raise VnnlibError(f"unsupported declaration {form!r}")
            m = _VAR.match(form[1])
            if not m:
                raise VnnlibError(f"variable {form[1]!r} is not X_<i> or Y_<j>")
            declared.add(form[1])
            if m.group(1) == "X":
                nx = max(nx, int(m.group(2)) + 1)
            else:
                ny = max(ny, int(m.group(2)) + 1)
        elif form[0] == "assert":
            if len(form) != 2:
                raise VnnlibError("assert takes one formula")
            formula.append(form[1])
        else:
            raise VnnlibError(f"unsupported top-level form {form[0]!r}")
    if not nx or not ny:
        raise VnnlibError("the property declares no inputs or no outputs")
    clauses = _atoms_of(["and"] + formula)
    out: List[Clause] = []
    for xb, ya in clauses:
        lo = np.full(nx, -np.inf)
        hi = np.full(nx, np.inf)
        for i, (lo_i, hi_i) in xb.items():
            if i >= nx:
                raise VnnlibError(f"X_{i} was not declared")
            lo[i], hi[i] = lo_i, hi_i
        if np.any(lo > hi):
            continue  # an empty input box: no unsafe point in this clause
        atoms = []
        for coefs, k, strict in ya:
            a = np.zeros(ny)
            for j, c in coefs.items():
                if j >= ny:
                    raise VnnlibError(f"Y_{j} was not declared")
                a[j] = c
            atoms.append(YAtom(a, -k, strict))  # a.Y + k <= 0   <=>   a.Y <= -k
        out.append(Clause(lo, hi, atoms))
    return Property(nx, ny, out)


def parse_file(path: str) -> Property:
    with open(path) as f:
        return parse(f.read())


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

ENGINES = ("ibp", "zonotope", "crown", "alpha", "prima", "bab", "bab-beta")

UNSAT, SAT, UNKNOWN, UNSUPPORTED = "unsat", "sat", "unknown", "unsupported"


class _Unsupported(Exception):
    """The network or property is outside what ``verify`` handles."""


@dataclasses.dataclass
class Verdict:
    status: str  # unsat | sat | unknown | unsupported
    engine: str
    seconds: float
    detail: str = ""
    counterexample: Optional[Tuple[np.ndarray, np.ndarray]] = (
        None  # (x, y), replayed on ORT
    )
    clauses: int = 0  # clauses of the property
    regions: int = 0  # regions bounded by branch and bound (sum over clauses)
    # the engines proved safety but an onnxruntime replay found a violation (float32 vs
    # real arithmetic at a razor-thin margin, or a bug): never resolved silently
    inconsistent: bool = False


def _single_io(model: onnx.ModelProto) -> Tuple[str, List[int]]:
    inits = {t.name for t in model.graph.initializer}
    ins = [i for i in model.graph.input if i.name not in inits]
    if len(ins) != 1 or len(model.graph.output) != 1:
        raise _Unsupported("only networks with one input and one output are supported")
    shape = []
    for k, d in enumerate(ins[0].type.tensor_type.shape.dim):
        if d.HasField("dim_value") and d.dim_value > 0:
            shape.append(int(d.dim_value))
        elif k == 0:
            shape.append(1)  # a dynamic batch dimension is run at 1
        else:
            raise _Unsupported("dynamic non-batch input dimension")
    return ins[0].name, shape


def _output_size(model: onnx.ModelProto) -> Optional[int]:
    """Number of output elements (batch 1), or None if shape inference cannot tell."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        out = inferred.graph.output[0].type.tensor_type
        if not out.HasField("shape"):
            return None
        n = 1
        for k, d in enumerate(out.shape.dim):
            if d.HasField("dim_value") and d.dim_value > 0:
                n *= int(d.dim_value)
            elif k != 0:
                return None
        return n
    except Exception:  # shape inference is a convenience check only
        return None


def _spec_model(model: onnx.ModelProto, a: np.ndarray) -> Tuple[onnx.ModelProto, str]:
    """``model`` plus an output ``__vnn_spec = Flatten(Y) @ A^T`` (one column per atom)."""
    a32 = a.astype(np.float32)
    if not np.array_equal(a32.astype(np.float64), a):
        raise _Unsupported(
            "an output coefficient is not exactly representable in float32"
        )
    m = onnx.ModelProto()
    m.CopyFrom(model)
    out = m.graph.output[0].name
    m.graph.node.append(helper.make_node("Flatten", [out], ["__vnn_flat"], axis=1))
    m.graph.initializer.append(
        numpy_helper.from_array(np.ascontiguousarray(a32.T), "__vnn_A")
    )
    m.graph.node.append(
        helper.make_node("MatMul", ["__vnn_flat", "__vnn_A"], ["__vnn_spec"])
    )
    m.graph.output.append(
        helper.make_tensor_value_info("__vnn_spec", TensorProto.FLOAT, [1, a.shape[0]])
    )
    return m, "__vnn_spec"


def _ranges(
    name: str, shape: Sequence[int], lo: np.ndarray, hi: np.ndarray
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return {name: (lo.reshape(shape), hi.reshape(shape))}


def _bounds(
    model: onnx.ModelProto,
    rng: Dict[str, Tuple[np.ndarray, np.ndarray]],
    tensor: str,
    engine: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sound ``(lo, hi)`` of ``tensor`` over the box with one of the root-bound engines."""
    if engine == "zonotope":
        try:
            lo, hi = _zonotope.propagate(model, rng).bounds(tensor)
        except KeyError:
            raise _Unsupported(
                f"zonotope has no abstract value for {tensor!r}"
            ) from None
        return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    tb = _crown.bounds(model, rng, output=tensor, method=engine)[tensor]
    return np.asarray(tb.lo, dtype=np.float64), np.asarray(tb.hi, dtype=np.float64)


def net_output_bounds(
    model: onnx.ModelProto, lo: np.ndarray, hi: np.ndarray, engine: str = "crown"
) -> Tuple[np.ndarray, np.ndarray]:
    """Bounds of the network output (flattened) over the flat input box ``[lo, hi]``.

    A thin helper for comparing engines (or an external reference) on identical
    network + box; ``verify`` itself bounds the property's linear specs instead.
    """
    if engine not in ("ibp", "zonotope", "crown", "alpha", "prima"):
        raise ValueError(f"engine {engine!r} is not a root-bound engine")
    name, shape = _single_io(model)
    out = model.graph.output[0].name
    rng = _ranges(
        name, shape, np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    )
    blo, bhi = _bounds(model, rng, out, engine)
    return blo.reshape(-1), bhi.reshape(-1)


class _ClauseBab(_crown._Bab):
    """Branch and bound where a region is *decided* when SOME atom of the clause is infeasible.

    ``crown._Bab`` decides a region when every output element lies inside a target range.
    The clause's emptiness test is a disjunction (one infeasible atom suffices), so the
    per-element violation is replaced by: the smallest violation over atoms, placed on the
    atom closest to being infeasible (which is also the element ``_worst`` then tightens).
    This relies on the private ``_violation`` hook of ``crown._Bab``.
    """

    def __init__(
        self,
        model: onnx.ModelProto,
        rng: Dict[str, Tuple[np.ndarray, np.ndarray]],
        spec: str,
        thresholds: np.ndarray,
        leaf_method: str,
    ) -> None:
        super().__init__(
            model,
            rng,
            [spec],
            "auto",
            leaf_method,
            30,
            0.2,
            {spec: (np.zeros(1), np.zeros(1))},
            0.0,
            0,
        )
        self._thr = np.asarray(thresholds, dtype=np.float64)

    def _violation(self, n: str, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        flat = np.asarray(lo, dtype=np.float64).reshape(-1)
        v = np.maximum(
            self._thr - flat, 0.0
        )  # 0: this atom is infeasible on the region
        out = np.zeros_like(v)
        j = int(np.argmin(v))
        out[j] = v[j]
        return out.reshape(np.shape(lo))


# --------------------------------------------------------------------------
# Counterexample search (always replayed on onnxruntime)
# --------------------------------------------------------------------------


def _ort_session(model: onnx.ModelProto) -> Any:
    import onnxruntime as ort

    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def _inside_f32(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """``x`` as float32 with every element inside ``[lo, hi]`` as float64 (nudged if needed)."""
    x32 = np.clip(x, lo, hi).astype(np.float32)
    for _ in range(4):
        bad_hi = x32.astype(np.float64) > hi
        bad_lo = x32.astype(np.float64) < lo
        if not (bad_hi.any() or bad_lo.any()):
            break
        x32 = np.where(bad_hi, np.nextafter(x32, np.float32(-np.inf)), x32)
        x32 = np.where(bad_lo, np.nextafter(x32, np.float32(np.inf)), x32).astype(
            np.float32
        )
    return x32


def _replay(
    sess: Any, name: str, shape: Sequence[int], prop: Property, x: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    y = sess.run(None, {name: x.reshape(shape).astype(np.float32)})[0]
    if prop.violated_by(x.reshape(-1), np.asarray(y).reshape(-1)):
        return x.reshape(-1).astype(np.float64), np.asarray(
            y, dtype=np.float64
        ).reshape(-1)
    return None


class _TorchNet:
    """A small torch interpreter for MLP/CNN ONNX graphs: enough for PGD, never for verdicts."""

    _OPS = frozenset(
        {
            "Sub", "Add", "Mul", "Div", "Neg", "MatMul", "Gemm", "Relu", "Sigmoid", "Tanh",
            "Flatten", "Reshape", "Transpose", "Conv", "BatchNormalization", "Identity",
            "AveragePool", "GlobalAveragePool", "Constant",
        }
    )  # fmt: skip

    def __init__(self, model: onnx.ModelProto) -> None:
        import torch

        self.torch = torch
        self.nodes = list(model.graph.node)
        self.consts = {
            t.name: torch.tensor(numpy_helper.to_array(t))
            for t in model.graph.initializer
        }
        for n in self.nodes:  # Constant nodes are folded into the constant table
            if n.op_type == "Constant":
                value = next((k for k in n.attribute if k.name == "value"), None)
                if value is None:
                    raise _Unsupported("Constant node without a tensor value")
                self.consts[n.output[0]] = torch.tensor(numpy_helper.to_array(value.t))
        self.nodes = [n for n in self.nodes if n.op_type != "Constant"]
        self.inp, _ = _single_io(model)
        self.out = model.graph.output[0].name
        for n in self.nodes:
            if n.op_type not in self._OPS:
                raise _Unsupported(f"PGD interpreter has no op {n.op_type}")

    def __call__(self, x: Any) -> Any:
        t = self.torch
        F = t.nn.functional
        env: Dict[str, Any] = dict(self.consts)
        env[self.inp] = x
        b = x.shape[0]
        for n in self.nodes:
            a = [env[i] for i in n.input if i]
            at = {k.name: helper.get_attribute_value(k) for k in n.attribute}
            op = n.op_type
            if op == "Sub":
                r = a[0] - a[1]
            elif op == "Add":
                r = a[0] + a[1]
            elif op == "Mul":
                r = a[0] * a[1]
            elif op == "Div":
                r = a[0] / a[1]
            elif op == "Neg":
                r = -a[0]
            elif op == "MatMul":
                r = a[0] @ a[1]
            elif op == "Gemm":
                u, v = a[0], a[1]
                u = u.T if at.get("transA", 0) else u
                v = v.T if at.get("transB", 0) else v
                r = at.get("alpha", 1.0) * (u @ v)
                if len(a) > 2:
                    r = r + at.get("beta", 1.0) * a[2]
            elif op == "Relu":
                r = t.relu(a[0])
            elif op == "Sigmoid":
                r = t.sigmoid(a[0])
            elif op == "Tanh":
                r = t.tanh(a[0])
            elif op == "Flatten":
                ax = at.get("axis", 1)
                r = a[0].reshape(int(np.prod(a[0].shape[:ax])), -1)
            elif op == "Reshape":
                shp = [int(s) for s in a[1].tolist()]
                if shp and shp[0] == 1:
                    shp[0] = b  # the batch dimension of the traced graph
                r = a[0].reshape(shp)
            elif op == "Transpose":
                r = a[0].permute(*at["perm"])
            elif op == "Identity":
                r = a[0]
            elif op == "Conv":
                pads = at.get("pads", [0, 0, 0, 0])
                if pads[0] != pads[2] or pads[1] != pads[3]:
                    raise _Unsupported("asymmetric Conv padding")
                r = F.conv2d(
                    a[0],
                    a[1],
                    a[2] if len(a) > 2 else None,
                    stride=tuple(at.get("strides", [1, 1])),
                    padding=(pads[0], pads[1]),
                    dilation=tuple(at.get("dilations", [1, 1])),
                    groups=at.get("group", 1),
                )
            elif op == "BatchNormalization":
                r = F.batch_norm(
                    a[0], a[3], a[4], a[1], a[2], False, 0.0, at.get("epsilon", 1e-5)
                )
            elif op == "AveragePool":
                pads = at.get("pads", [0, 0, 0, 0])
                r = F.avg_pool2d(
                    a[0],
                    tuple(at["kernel_shape"]),
                    tuple(at.get("strides", [1, 1])),
                    (pads[0], pads[1]),
                )
            else:  # GlobalAveragePool
                r = a[0].mean(dim=(2, 3), keepdim=True)
            env[n.output[0]] = r
        return env[self.out]


def _attack(
    model: onnx.ModelProto,
    prop: Property,
    clause: Clause,
    name: str,
    shape: Sequence[int],
    seed: int,
    seconds: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Look for a counterexample inside ``clause``'s box: samples, corners, then PGD."""
    t0 = time.monotonic()
    lo, hi = clause.lo, clause.hi
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    rng = np.random.default_rng(seed)
    sess = _ort_session(model)
    a_mat = np.stack([at.a for at in clause.atoms])
    b_vec = np.array([at.b for at in clause.atoms])

    def try_x(x: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return _replay(sess, name, shape, prop, _inside_f32(x.reshape(-1), lo, hi))

    cands = [(lo + hi) / 2.0, lo, hi]
    cands += [lo + (hi - lo) * rng.integers(0, 2, lo.shape) for _ in range(64)]
    cands += [lo + (hi - lo) * rng.random(lo.shape) for _ in range(192)]
    for x in cands:
        got = try_x(x)
        if got is not None:
            return got
        if time.monotonic() - t0 > seconds:
            return None
    if len(shape) < 2:
        return None  # PGD needs a leading batch dimension
    try:
        return _pgd(model, try_x, lo, hi, a_mat, b_vec, shape, rng, t0, seconds)
    except Exception:  # torch missing, an op the interpreter lacks, ...: sampling only
        return None


def _pgd(
    model: onnx.ModelProto,
    try_x: Any,
    lo: np.ndarray,
    hi: np.ndarray,
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    shape: Sequence[int],
    rng: np.random.Generator,
    t0: float,
    seconds: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    import torch

    net = _TorchNet(model)
    n_start, steps = 64, 200
    xs = tuple(shape[1:])
    lo_t = torch.tensor(lo.reshape(xs), dtype=torch.float32)
    hi_t = torch.tensor(hi.reshape(xs), dtype=torch.float32)
    width = hi_t - lo_t
    a_t = torch.tensor(a_mat, dtype=torch.float32)
    b_t = torch.tensor(b_vec, dtype=torch.float32)
    first = True
    while True:  # restart rounds until the budget is spent (thin unsafe regions need many starts)
        x0 = lo + (hi - lo) * rng.random((n_start,) + lo.shape)
        if first:
            x0[0] = (lo + hi) / 2.0
            first = False
        x = torch.tensor(x0.reshape((n_start,) + xs), dtype=torch.float32)
        best_f = torch.full((n_start,), float("inf"))
        best_x = x.clone()
        for step in range(steps):
            x = x.detach().requires_grad_(True)
            y = net(x).reshape(n_start, -1)
            f = (
                (y @ a_t.T - b_t).max(dim=1).values
            )  # unsafe iff max_j(a_j.y - b_j) <= 0
            f.sum().backward()
            with torch.no_grad():
                fd = f.detach()
                better = fd < best_f
                best_f = torch.where(better, fd, best_f)
                best_x[better] = x.detach()[better]
                for i in (
                    torch.nonzero(fd <= 1e-6).flatten().tolist()[:4]
                ):  # looks unsafe: replay it
                    got = try_x(x[i].detach().numpy().astype(np.float64))
                    if got is not None:
                        return got
                g = x.grad
                dead = (
                    g.abs().flatten(1).sum(1) == 0
                )  # a flat (dead ReLU) region: restart there
                lr = width * (
                    0.05 * 0.95**step + 5e-5
                )  # cooled step: sign steps overshoot a kinked f
                x_new = torch.minimum(
                    torch.maximum(x.detach() - lr * g.sign(), lo_t), hi_t
                )
                if bool(dead.any()):
                    x_new[dead] = lo_t + width * torch.rand(
                        (int(dead.sum().item()),) + xs
                    )
                x = x_new
            if time.monotonic() - t0 > seconds:
                break
        for i in torch.argsort(best_f)[
            :8
        ].tolist():  # last look at this round's best iterates
            got = try_x(best_x[i].numpy().astype(np.float64))
            if got is not None:
                return got
        if time.monotonic() - t0 > seconds:
            return None


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def _as_model(m: Any) -> onnx.ModelProto:
    return onnx.load(m) if isinstance(m, str) else m


def _as_property(p: Any) -> Property:
    if isinstance(p, Property):
        return p
    if "(" in p:
        return parse(p)
    return parse_file(p)


def _finite(c: Clause) -> bool:
    return bool(np.all(np.isfinite(c.lo)) and np.all(np.isfinite(c.hi)))


def find_counterexample(
    model: Any, prop: Any, seconds: float = 10.0, seed: int = 0
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Search every clause for an input whose output replays as unsafe on onnxruntime.

    Independent of the bound engines (it never reads a bound), so it doubles as their
    soundness oracle: a point found here inside a region an engine proved safe is a bug.
    ``seconds`` is the total budget, split over the clauses. Returns ``(x, y)`` or None;
    None only means "not found" -- never "safe".
    """
    net = _as_model(model)
    pr = _as_property(prop)
    name, shape = _single_io(net)
    clauses = [c for c in pr.clauses if _finite(c)]
    per = max(0.2, seconds / max(1, len(clauses)))
    for c in clauses:
        if not c.atoms:  # the whole box is unsafe
            pt = _inside_f32((c.lo + c.hi) / 2, c.lo, c.hi)
            got = _replay(_ort_session(net), name, shape, pr, pt)
        else:
            got = _attack(net, pr, c, name, shape, seed, per)
        if got is not None:
            return got
    return None


def verify(
    model: Any,
    prop: Any,
    engine: str = "crown",
    timeout: float = 60.0,
    budget: int = 200,
    attack: bool = True,
    seed: int = 0,
    attack_seconds: float = 5.0,
) -> Verdict:
    """Answer one VNN-COMP instance: ``unsat`` (safe), ``sat`` (replayed counterexample) or ``unknown``.

    :param model: ONNX model or path (one input, one output; dynamic batch is run at 1).
    :param prop: ``.vnnlib`` text, path, or a parsed :class:`Property`.
    :param engine: how the output specs are bounded: ``ibp``, ``zonotope``, ``crown``,
        ``alpha`` and ``prima`` bound the whole input box once; ``bab`` and ``bab-beta``
        add input / ReLU splitting (``budget`` regions per clause, ``timeout`` seconds in
        total; ``bab-beta`` uses beta-CROWN leaves and needs torch).
    :param attack: also look for a counterexample (samples, corners, PGD with torch) in the
        clauses the bounds could not exclude. A ``sat`` always carries a point replayed on
        onnxruntime.
    :param attack_seconds: total time for that search, split over the open clauses (a thin
        unsafe region can need many random restarts; 5 s is cheap, tens of seconds find more).
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    t0 = time.monotonic()

    def done(status: str, detail: str = "", cex: Any = None, **kw: Any) -> Verdict:
        return Verdict(status, engine, time.monotonic() - t0, detail, cex, **kw)

    def safe(detail: str, regions: int = 0) -> Verdict:
        # Every ``unsat`` passes through here. A counterexample that replays on onnxruntime
        # inside ANY clause (the open ones were attacked above for longer, so this is a short
        # extra look) means the proof is wrong: reported as sat + inconsistent, never ignored.
        if attack:
            for c in pr.clauses:
                got = _attack(
                    net, pr, c, name, shape, seed + 1, min(1.0, attack_seconds)
                )
                if got is not None:
                    return done(
                        SAT,
                        "SOUNDNESS ALARM: a counterexample replays inside a clause the engine "
                        "claimed excluded",
                        got,
                        clauses=nclauses,
                        regions=regions,
                        inconsistent=True,
                    )
        return done(UNSAT, detail, clauses=nclauses, regions=regions)

    root_engine = "crown" if engine.startswith("bab") else engine
    try:
        net = _as_model(model)
        pr = _as_property(prop)
        name, shape = _single_io(net)
        if int(np.prod(shape)) != pr.n_inputs:
            raise _Unsupported(
                f"the property has {pr.n_inputs} inputs but the network takes {int(np.prod(shape))}"
            )
        n_out = _output_size(net)
        if n_out is not None and n_out != pr.n_outputs:
            raise _Unsupported(
                f"the property has {pr.n_outputs} outputs but the network produces {n_out}"
            )
    except (VnnlibError, _Unsupported) as e:
        return done(UNSUPPORTED, str(e))
    nclauses = len(pr.clauses)
    if nclauses == 0:
        return done(UNSAT, "the unsafe region is empty", clauses=0)

    open_clauses: List[Tuple[Clause, onnx.ModelProto, str, np.ndarray]] = []
    try:
        for c in pr.clauses:
            if not c.atoms:  # the whole box is unsafe: any point is a counterexample
                if _finite(c):
                    pt = _inside_f32((c.lo + c.hi) / 2, c.lo, c.hi)
                    got = _replay(_ort_session(net), name, shape, pr, pt)
                    if got is not None:
                        return done(SAT, "no output constraint", got, clauses=nclauses)
                return done(
                    UNKNOWN,
                    "a clause with no output constraint and no finite box",
                    clauses=nclauses,
                )
            a = np.stack([at.a for at in c.atoms])
            b = np.array([at.b for at in c.atoms])
            thr = b + _MARGIN * np.maximum(1.0, np.abs(b))
            spec_model, spec = _spec_model(net, a)
            if _finite(c):
                lo_b = _bounds(
                    spec_model, _ranges(name, shape, c.lo, c.hi), spec, root_engine
                )[0]
            else:
                lo_b = np.full(len(b), -np.inf)  # an unbounded input box proves nothing
            if not np.any(
                lo_b.reshape(-1) > thr
            ):  # no atom is infeasible: clause stays open
                open_clauses.append((c, spec_model, spec, thr))
    except _Unsupported as e:
        return done(UNSUPPORTED, str(e), clauses=nclauses)
    except (ValueError, NotImplementedError) as e:
        return done(UNSUPPORTED, f"{type(e).__name__}: {e}", clauses=nclauses)
    if not open_clauses:
        return safe("every clause is excluded by the root bounds")

    if attack:
        per = max(0.2, attack_seconds / len(open_clauses))
        for c, _, _, _ in open_clauses:
            got = _attack(net, pr, c, name, shape, seed, per)
            if got is not None:
                return done(
                    SAT, "counterexample replayed on onnxruntime", got, clauses=nclauses
                )

    regions = 0
    if engine.startswith("bab"):
        leaf = "beta" if engine == "bab-beta" else "crown"
        for k, (c, spec_model, spec, thr) in enumerate(open_clauses):
            left = max(timeout - (time.monotonic() - t0), 0.0) / (len(open_clauses) - k)
            try:
                bab = _ClauseBab(
                    spec_model, _ranges(name, shape, c.lo, c.hi), spec, thr, leaf
                )
                res = bab.run(budget, left)
            except (ValueError, NotImplementedError, ImportError) as e:
                return done(UNSUPPORTED, f"{type(e).__name__}: {e}", clauses=nclauses)
            regions += res.regions
            if not res.proved:
                return done(
                    UNKNOWN,
                    f"branch and bound did not exclude clause {k}",
                    clauses=nclauses,
                    regions=regions,
                )
        return safe("branch and bound excluded every clause", regions)
    return done(
        UNKNOWN, "the bounds do not exclude the unsafe region", clauses=nclauses
    )


# --------------------------------------------------------------------------
# VNN-COMP tool protocol (2025): run_instance.sh writes the result file the harness reads
# --------------------------------------------------------------------------

VNNCOMP_WORD = {
    UNSAT: "unsat",
    SAT: "sat",
    UNKNOWN: "unknown",
    UNSUPPORTED: "unknown",
}
_WATCHDOG_MARGIN = (
    5.0  # seconds before the limit at which the "timeout" word is written
)


def _format_counterexample(x: np.ndarray, y: np.ndarray) -> str:
    entries = [f"(X_{i} {float(v)!r})" for i, v in enumerate(np.ravel(x))]
    entries += [f"(Y_{j} {float(v)!r})" for j, v in enumerate(np.ravel(y))]
    return "(\n" + "\n".join(entries) + "\n)\n"


def _write_result(path: str, word: str, counterexample: str = "") -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(word + "\n" + counterexample)
    os.replace(tmp, path)


def run_instance(
    onnx_path: str,
    vnnlib_path: str,
    results_path: str,
    timeout: float,
    engine: str = "bab",
    budget: int = 200,
) -> str:
    """Answer one VNN-COMP instance, write its one-word result file, and return the word.

    The first line is ``unsat``, ``sat`` (followed by the replayed counterexample as
    ``(X_i v)`` / ``(Y_j v)`` lines), ``unknown`` (including unsupported properties, which
    the protocol has no separate word for), ``timeout`` or ``error``. If the answer is not
    ready close to ``timeout`` seconds, a watchdog writes ``timeout`` and ends the process,
    so the harness never reads a missing or stale file.
    """
    lock = threading.Lock()
    settled: list[str] = []

    def settle(word: str, counterexample: str = "") -> bool:
        with lock:
            if settled:
                return False
            settled.append(word)
            _write_result(results_path, word, counterexample)
            return True

    def on_deadline() -> None:
        if settle("timeout"):
            os._exit(0)

    margin = min(_WATCHDOG_MARGIN, 0.1 * timeout)
    watchdog = threading.Timer(max(timeout - margin, 0.0), on_deadline)
    watchdog.daemon = True
    watchdog.start()
    counterexample = ""
    try:
        v = verify(
            onnx_path,
            vnnlib_path,
            engine=engine,
            timeout=max(timeout - margin, 1.0),
            budget=budget,
        )
        word = VNNCOMP_WORD[v.status]
        if v.status == SAT and v.counterexample is not None:
            counterexample = _format_counterexample(*v.counterexample)
        print(
            f"{word}: {v.status} via {v.engine} in {v.seconds:.2f}s: {v.detail}",
            file=sys.stderr,
        )
        if v.inconsistent:
            print(
                "warning: the verdict is inconsistent, see the detail above",
                file=sys.stderr,
            )
    except Exception:
        traceback.print_exc()
        word = "error"
    settle(word, counterexample)
    watchdog.cancel()
    return word


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m onnxsim.vnnlib", description=__doc__.split("\n")[0]
    )
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="answer one VNN-COMP instance (run_instance protocol)"
    )
    run.add_argument("onnx")
    run.add_argument("vnnlib")
    run.add_argument("results")
    run.add_argument("timeout", type=float, help="seconds")
    run.add_argument("--engine", default="bab", choices=ENGINES)
    run.add_argument("--budget", type=int, default=200, help="BaB regions per clause")
    args = ap.parse_args(argv)
    run_instance(
        args.onnx, args.vnnlib, args.results, args.timeout, args.engine, args.budget
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
