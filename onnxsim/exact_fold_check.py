"""Independent check that constant folding kept the values it folded.

onnxsim folds constant subgraphs with its own engine and stores the results as initializers.
This module recomputes each such initializer from the original model's constant subgraph, in
float64, and compares it with the simplified value. Two comparisons are made:

* ``evaluation``: the subgraph is run by the ONNX reference evaluator with float32 values promoted
  to float64. The simplified value must lie within ``(nodes + 1) * 2**-24 * S`` of the reference,
  where ``S`` is the largest magnitude of any intermediate. That is a rounding budget scaled by the
  largest intermediate, not a sound bound: heavy cancellation can exceed it.
* ``freivalds``: for a large constant MatMul or Gemm, ``A (B r)`` is compared with ``C r`` for
  random +-1 vectors ``r`` instead of recomputing ``A @ B``. A mismatch beyond the tolerance is a
  real discrepancy. Agreement is evidence, not a proof: a wrong ``C`` that moves an output by more
  than the tolerance is caught with probability at least 1/2 per trial.

Names are matched between the two models. A simplified initializer whose name is also an
initializer or pinned input of the original is compared with that value exactly, so a rewrite that
legitimately changes a weight (BN fusion, quantization) is reported as a mismatch. A name the
original computes from a non-constant input, or from an operator the evaluator cannot run, is
``unanalysed``. ``input_ranges`` pins an input to a value when its lower and upper bounds are equal.
"""

import dataclasses
from typing import Dict, List, Optional, Sequence

import numpy as np
import onnx
from onnx import numpy_helper

_UNIT_ROUNDOFF_FP32 = 2.0**-24
_FREIVALDS_MIN_FLOPS = 1 << 22
_FREIVALDS_TRIALS = 3
_SEED = 0
_TINY = float(np.finfo(np.float64).tiny)
_NON_CONSTANT_OPS = frozenset(
    {
        "If",
        "Loop",
        "Scan",
        "RandomUniform",
        "RandomNormal",
        "RandomUniformLike",
        "RandomNormalLike",
        "Multinomial",
        "Bernoulli",
    }
)


@dataclasses.dataclass(frozen=True)
class FoldFinding:
    tensor: str
    method: str  # "initializer", "evaluation" or "freivalds"
    max_abs: float
    max_rel: float
    tolerance: float
    ok: bool


@dataclasses.dataclass
class FoldCheckReport:
    findings: List[FoldFinding]
    unanalysed: List[str]

    @property
    def matches(self) -> bool:
        return all(f.ok for f in self.findings)

    @property
    def complete(self) -> bool:
        return not self.unanalysed


def check_constant_fold(
    original: onnx.ModelProto,
    simplified: onnx.ModelProto,
    input_ranges: Optional[dict] = None,
    freivalds_min_flops: int = _FREIVALDS_MIN_FLOPS,
    trials: int = _FREIVALDS_TRIALS,
) -> FoldCheckReport:
    """Compare each initializer of ``simplified`` with the value ``original`` computes for it.

    The Freivalds check is probabilistic. Set ``freivalds_min_flops`` to ``0`` to use it for every
    constant MatMul or Gemm.
    """
    checker = _Checker(original, input_ranges or {})
    findings: List[FoldFinding] = []
    unanalysed: List[str] = []
    for init in simplified.graph.initializer:
        try:
            finding = checker.check(init, freivalds_min_flops, trials)
        except Exception:  # an evaluator limitation or an unsupported op: stay sound
            finding = None
        if finding is None:
            unanalysed.append(init.name)
        else:
            findings.append(finding)
    return FoldCheckReport(findings, unanalysed)


class _Checker:
    def __init__(self, model: onnx.ModelProto, input_ranges: dict):
        self.model = model
        self.original_initializers = {
            i.name: numpy_helper.to_array(i) for i in model.graph.initializer
        }
        self.values: Dict[str, np.ndarray] = dict(self.original_initializers)
        for name, (lo, hi) in input_ranges.items():
            if lo is not None and hi is not None and np.array_equal(lo, hi):
                self.values.setdefault(name, np.asarray(lo))
        self.producer = {o: n for n in model.graph.node for o in n.output if o}

    def check(
        self, init: onnx.TensorProto, min_flops: int, trials: int
    ) -> Optional[FoldFinding]:
        name = init.name
        sim = numpy_helper.to_array(init)
        if name in self.values:
            return _compare(
                name, "initializer", sim, self.values[name], 0.0, exact=True
            )
        nodes = self._constant_nodes(name)
        if nodes is None:
            return None
        producer = self.producer[name]
        if producer.op_type in ("MatMul", "Gemm"):
            finding = self._freivalds(name, producer, sim, min_flops, trials)
            if finding is not None:
                return finding
        return self._evaluate(name, nodes, sim)

    def _constant_nodes(self, target: str) -> Optional[List[onnx.NodeProto]]:
        """Nodes computing ``target`` from initializers, pinned inputs and constants, in order."""
        order: List[onnx.NodeProto] = []
        emitted = set()
        stack = [(target, False)]
        while stack:
            tensor, expanded = stack.pop()
            if not tensor or tensor in self.values:
                continue
            node = self.producer.get(tensor)
            if node is None or node.op_type in _NON_CONSTANT_OPS:
                return None
            if id(node) in emitted:
                continue
            if expanded:
                emitted.add(id(node))
                order.append(node)
                continue
            stack.append((tensor, True))
            stack.extend((i, False) for i in node.input)
        return order

    def _run(
        self, nodes: Sequence[onnx.NodeProto], outputs: List[str]
    ) -> List[np.ndarray]:
        from onnx.reference import ReferenceEvaluator

        inits: Dict[str, np.ndarray] = {}
        body: List[onnx.NodeProto] = []
        for node in nodes:
            if node.op_type == "Constant":
                if [a.name for a in node.attribute] != ["value"]:
                    raise NotImplementedError("Constant with a non-tensor value")
                inits[node.output[0]] = _promote(
                    numpy_helper.to_array(node.attribute[0].t)
                )
                continue
            body.append(node)
            for i in node.input:
                if i and i in self.values:
                    inits.setdefault(i, _promote(self.values[i]))
        graph = onnx.helper.make_graph(
            body,
            "constant_fold_check",
            [],
            [],
            initializer=[numpy_helper.from_array(v, n) for n, v in inits.items()],
        )
        model = onnx.helper.make_model(
            graph,
            ir_version=self.model.ir_version,
            opset_imports=list(self.model.opset_import),
        )
        return ReferenceEvaluator(model).run(outputs, {})

    def _value(self, tensor: str) -> Optional[np.ndarray]:
        if tensor in self.values:
            return _promote(self.values[tensor])
        nodes = self._constant_nodes(tensor)
        if not nodes:
            return None
        return self._run(nodes, [tensor])[0]

    def _evaluate(
        self, name: str, nodes: List[onnx.NodeProto], sim: np.ndarray
    ) -> FoldFinding:
        outputs = list(dict.fromkeys(o for n in nodes for o in n.output if o))
        results = dict(zip(outputs, self._run(nodes, outputs)))
        ref = results[name]
        if not np.issubdtype(ref.dtype, np.floating):
            return _compare(name, "evaluation", sim, ref, 0.0, exact=True)
        scale = max(
            (
                _magnitude(v)
                for v in results.values()
                if np.issubdtype(v.dtype, np.floating)
            ),
            default=0.0,
        )
        tol = (len(nodes) + 1) * _UNIT_ROUNDOFF_FP32 * scale
        return _compare(name, "evaluation", sim, _promote(ref), tol, exact=False)

    def _freivalds(
        self,
        name: str,
        node: onnx.NodeProto,
        sim: np.ndarray,
        min_flops: int,
        trials: int,
    ) -> Optional[FoldFinding]:
        bias_name = None
        if node.op_type == "Gemm":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if (
                attrs.get("alpha", 1.0) != 1.0
                or attrs.get("beta", 1.0) != 1.0
                or attrs.get("transA", 0)
                or attrs.get("transB", 0)
            ):
                return None
            if len(node.input) > 2 and node.input[2]:
                bias_name = node.input[2]
        a = self._value(node.input[0])
        b = self._value(node.input[1])
        bias = self._value(bias_name) if bias_name else None
        if a is None or b is None or (bias_name and bias is None):
            return None
        if a.ndim < 2 or b.ndim < 2:
            return None
        m, k = a.shape[-2:]
        n = b.shape[-1]
        batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        if int(np.prod(batch, dtype=np.int64)) * m * k * n < min_flops:
            return None
        if sim.shape != batch + (m, n):
            return FoldFinding(
                name, "freivalds", float("inf"), float("inf"), 0.0, False
            )

        sim64 = sim.astype(np.float64)
        abs_a, abs_b, abs_s = np.abs(a), np.abs(b), np.abs(sim64)
        abs_bias = np.abs(bias) if bias is not None else 0.0
        rng = np.random.default_rng(_SEED)
        max_abs = 0.0
        max_rel = 0.0
        max_tol = 0.0
        ok = True
        for _ in range(trials):
            r = rng.choice([-1.0, 1.0], size=(n, 1))
            abs_r = np.abs(r)
            lhs = a @ (b @ r)
            if bias is not None:
                lhs = lhs + bias @ r
            rhs = sim64 @ r
            mag = abs_a @ (abs_b @ abs_r) + abs_s @ abs_r
            if bias is not None:
                mag = mag + abs_bias @ abs_r
            tol = (k + 2) * _UNIT_ROUNDOFF_FP32 * mag
            diff = np.abs(lhs - rhs)
            if not np.all(np.isfinite(diff)):
                return FoldFinding(
                    name, "freivalds", float("inf"), float("inf"), 0.0, False
                )
            max_abs = max(max_abs, float(diff.max()))
            max_rel = max(max_rel, float((diff / np.maximum(mag, _TINY)).max()))
            max_tol = max(max_tol, float(tol.max()))
            ok = ok and bool(np.all(diff <= tol))
        return FoldFinding(name, "freivalds", max_abs, max_rel, max_tol, ok)


def _promote(value: np.ndarray) -> np.ndarray:
    return value.astype(np.float64) if value.dtype == np.float32 else value


def _magnitude(value: np.ndarray) -> float:
    return float(np.max(np.abs(value.astype(np.float64)))) if value.size else 0.0


def _compare(
    name: str,
    method: str,
    sim: np.ndarray,
    ref: np.ndarray,
    tol: float,
    exact: bool,
) -> FoldFinding:
    if sim.shape != ref.shape:
        return FoldFinding(name, method, float("inf"), float("inf"), tol, False)
    if exact:
        same = bool(np.all((sim == ref) | (np.isnan(sim) & np.isnan(ref))))
        with np.errstate(invalid="ignore"):
            max_abs = float(
                np.max(
                    np.abs(sim.astype(np.float64) - ref.astype(np.float64)), initial=0.0
                )
            )
        return FoldFinding(
            name, method, max_abs, 0.0 if same else float("inf"), 0.0, same
        )
    sim64 = sim.astype(np.float64)
    ref64 = ref.astype(np.float64)
    finite = np.isfinite(sim64) & np.isfinite(ref64)
    nonfinite_match = np.array_equal(
        np.isnan(sim64), np.isnan(ref64)
    ) and np.array_equal(sim64[~finite], ref64[~finite], equal_nan=True)
    if not nonfinite_match:
        return FoldFinding(name, method, float("inf"), float("inf"), tol, False)
    diff = np.abs(sim64 - ref64)[finite]
    max_abs = float(diff.max()) if diff.size else 0.0
    scale = float(np.max(np.abs(ref64[finite]))) if finite.any() else 0.0
    max_rel = max_abs / max(scale, _TINY)
    return FoldFinding(name, method, max_abs, max_rel, tol, max_abs <= tol)
