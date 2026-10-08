"""Static NaN and Inf hazard detection on top of interval propagation.

``interval.propagate`` bounds every float tensor over the input boxes. A node is a
*hazard* when the operand intervals admit an input for which its IEEE operation yields
NaN (``kind="nan"``) or an infinity (``kind="inf"``), including overflow of an output
past the float range of the model. The analysis over-approximates: with no hazard, no
NaN or Inf is reachable inside the boxes; a reported hazard may be unreachable when the
intervals are loose. Inputs without a range are unbounded, so without ``input_ranges``
(or ``onnxsim.range.*`` annotations) nearly every arithmetic node is reported.

Ops without a rule are listed in ``unmodelled``. They are checked only for infinite
operands; their own NaN behaviour (e.g. a naive softmax) is assumed, not proved.
"""

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx

from onnxsim import interval as _interval

_STANDARD_DOMAINS = ("", "ai.onnx")

# Ops that create no NaN from infinite operands and whose outputs are not checked here
# beyond the generic overflow check.
_INF_SAFE = frozenset(
    {
        "Abs", "Cast", "Ceil", "Clip", "Concat", "Constant", "Equal", "Erf", "Exp",
        "Expand", "Flatten", "Floor", "Gather", "Greater", "GreaterOrEqual", "Identity",
        "Less", "LessOrEqual", "Max", "Min", "Neg", "Pad", "Relu", "Reshape", "Round",
        "Shape", "Sigmoid", "Slice", "Softplus", "Split", "Squeeze", "Tanh", "Tile",
        "Transpose", "Unsqueeze", "Where",
    }
)  # fmt: skip

_RULED = frozenset({"Add", "Sub", "Mul", "Div", "Reciprocal", "Sqrt", "Log", "Pow"})


@dataclasses.dataclass(frozen=True)
class Hazard:
    tensor: str  # first output of the node that may produce the value
    op_type: str
    kind: str  # "nan" or "inf"
    count: int  # elements that may be affected
    detail: str


@dataclasses.dataclass
class NanReport:
    hazards: List[Hazard]
    unmodelled: List[str]  # op types whose NaN behaviour is assumed, not checked
    unanalysed: List[str]  # output tensors of nodes whose operands have no interval

    @property
    def nan_free(self) -> bool:
        """No NaN is reachable through the checked ops, and every node was analysed."""
        return not self.unanalysed and not self._any("nan")

    @property
    def finite(self) -> bool:
        """No NaN or Inf is reachable through the checked ops, and every node was analysed."""
        return self.nan_free and not self._any("inf")

    @property
    def complete(self) -> bool:
        """Every op has a rule (no assumed NaN behaviour) and every node was analysed."""
        return not self.unmodelled and not self.unanalysed

    def _any(self, kind: str) -> bool:
        return any(h.kind == kind for h in self.hazards)


def _pos_inf(iv) -> np.ndarray:
    return iv[1] == np.inf


def _neg_inf(iv) -> np.ndarray:
    return iv[0] == -np.inf


def _inf(iv) -> np.ndarray:
    return _pos_inf(iv) | _neg_inf(iv)


def _zero(iv) -> np.ndarray:
    return (iv[0] <= 0) & (iv[1] >= 0)


def _rules(op: str, ins) -> List[Tuple[str, np.ndarray, str]]:
    """``(kind, mask, detail)`` per hazard of a ruled op; ``mask`` marks affected elements."""
    a = ins[0]
    b = ins[1] if len(ins) > 1 else a
    if op == "Sqrt":
        return [("nan", a[0] < 0, "argument may be negative")]
    if op == "Log":
        return [
            ("nan", a[0] < 0, "argument may be negative"),
            ("inf", a[0] <= 0, "argument may be zero"),
        ]
    if op == "Reciprocal":
        return [("inf", _zero(a), "argument may be zero")]
    if op == "Div":
        return [
            ("nan", (_zero(a) & _zero(b)) | (_inf(a) & _inf(b)), "0/0 or inf/inf"),
            ("inf", _zero(b) | _inf(a), "division by zero or by infinity"),
        ]
    if op == "Add":
        return [
            (
                "nan",
                (_pos_inf(a) & _neg_inf(b)) | (_neg_inf(a) & _pos_inf(b)),
                "inf + -inf",
            ),
            ("inf", _inf(a) | _inf(b), "infinite operand"),
        ]
    if op == "Sub":
        return [
            (
                "nan",
                (_pos_inf(a) & _pos_inf(b)) | (_neg_inf(a) & _neg_inf(b)),
                "inf - inf",
            ),
            ("inf", _inf(a) | _inf(b), "infinite operand"),
        ]
    if op == "Mul":
        return [
            ("nan", (_zero(a) & _inf(b)) | (_inf(a) & _zero(b)), "0 * inf"),
            ("inf", _inf(a) | _inf(b), "infinite operand"),
        ]
    if op == "Pow":
        y_int = (b[0] == b[1]) & (b[0] == np.round(b[0])) & np.isfinite(b[0])
        return [
            ("nan", (a[0] < 0) & ~y_int, "negative base with a non-integer exponent"),
            (
                "inf",
                (_zero(a) & (b[0] < 0)) | _inf(a) | _inf(b),
                "zero base or infinite operand",
            ),
        ]
    raise KeyError(op)


def _node_hazards(op: str, ins, tensor: str) -> List[Hazard]:
    if op in _RULED:
        found = [(k, m, d) for k, m, d in _rules(op, ins) if np.any(m)]
        return [Hazard(tensor, op, k, int(np.count_nonzero(m)), d) for k, m, d in found]
    if op in _INF_SAFE:
        return []
    count = sum(int(np.count_nonzero(_inf(x))) for x in ins)
    if count:
        return [
            Hazard(tensor, op, "nan", count, "unmodelled op, operand may be infinite")
        ]
    return []


def _float_max(model: onnx.ModelProto) -> float:
    for vi in model.graph.input:
        if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            return float(np.finfo(np.float16).max)
    return float(np.finfo(np.float32).max)


def check_nan(
    model: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
) -> NanReport:
    """Report the NaN and Inf hazards of ``model`` over the boxes in ``input_ranges``.

    :param input_ranges: ``{input: (lo, hi)}``, merged over the model's ``onnxsim.range.*``
        annotations. An input with neither is unbounded.
    """
    res = _interval.propagate(model, input_ranges)
    fmax = _float_max(model)
    hazards: List[Hazard] = []
    unmodelled: List[str] = []
    unanalysed: List[str] = []
    for node in model.graph.node:
        outs = [o for o in node.output if o]
        tensor = outs[0] if outs else node.name
        if node.domain not in _STANDARD_DOMAINS:
            if node.op_type not in unmodelled:
                unmodelled.append(node.op_type)
            continue
        if node.op_type not in _RULED and node.op_type not in _INF_SAFE:
            if node.op_type not in unmodelled:
                unmodelled.append(node.op_type)
        operands = [n for n in node.input if n]
        if any(n in res.ranged or n not in res.intervals for n in operands):
            unanalysed.append(tensor)
            continue
        ins = [
            (
                np.asarray(res.intervals[n][0], np.float64),
                np.asarray(res.intervals[n][1], np.float64),
            )
            for n in operands
        ]
        if not ins:
            continue
        found = _node_hazards(node.op_type, ins, tensor)
        hazards.extend(found)
        if any(h.kind == "inf" for h in found) or any(_inf(x).any() for x in ins):
            continue
        for out in outs:
            if out not in res.intervals:
                continue
            lo, hi = res.intervals[out]
            over = int(np.count_nonzero((np.abs(lo) > fmax) | (np.abs(hi) > fmax)))
            if over:
                hazards.append(
                    Hazard(
                        out, node.op_type, "inf", over, f"may overflow past {fmax:.3g}"
                    )
                )
    return NanReport(hazards, unmodelled, unanalysed)
