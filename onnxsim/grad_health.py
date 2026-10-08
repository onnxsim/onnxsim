"""Gradient health of a training graph: vanishing and imprecise gradients.

A training graph is an ordinary ONNX graph whose outputs include gradients, so the analyses of
:mod:`onnxsim.interval` and :mod:`onnxsim.fp_error` apply to it unchanged. For each gradient
tensor this module reports two things over the input box:

* **vanishing** -- an upper bound on ``|g|`` at or below the precision's underflow range. A
  gradient that is ``dead`` is identically zero over the box; ``flushed`` values round to zero in
  the target format; ``subnormal`` values keep only part of their significand. The bound is
  proved (interval arithmetic on the real semantics), so a vanishing verdict holds for every
  point of the box, not just the sampled ones.
* **imprecise** -- the roundoff bound of :func:`onnxsim.fp_error.roundoff_bound` is at least
  ``rel_tol`` times the magnitude bound, so the rounded gradient may have the wrong sign or be
  dominated by rounding. Unbounded roundoff is reported the same way.

Caveats. Interval magnitudes grow with depth (see ``fp_error``'s ``ranges`` option), so a
gradient can be reported imprecise only because of loose bounds. Ops without an error model are
reported through ``fp_error``'s ``unsupported`` list and make their gradients imprecise.
"""

import dataclasses
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.inliner

from onnxsim import fp_error as _fp_error
from onnxsim import interval as _interval

if TYPE_CHECKING:
    from onnxsim import qat_graph

_NORMAL_MIN = {"fp32": 2.0**-126, "fp16": 2.0**-14, "bf16": 2.0**-126}
_SUBNORMAL_MIN = {"fp32": 2.0**-149, "fp16": 2.0**-24, "bf16": 2.0**-133}
VANISHING_KINDS = frozenset({"dead", "flushed", "subnormal"})
# Same pairing qat_graph.make_step_graph uses, so the checked graph is the one that would run.
_OPSET = 17
_IR_VERSION = 8


@dataclasses.dataclass(frozen=True)
class GradFinding:
    tensor: str
    kind: str  # "dead", "flushed", "subnormal" or "imprecise"
    magnitude: float  # upper bound on |value| over the box
    error: float  # largest elementwise roundoff bound
    detail: str


@dataclasses.dataclass
class GradHealthReport:
    precision: str
    findings: List[GradFinding]
    unanalysed: List[str]  # gradients with no interval or roundoff bound

    @property
    def vanishing_free(self) -> bool:
        return not any(f.kind in VANISHING_KINDS for f in self.findings)

    @property
    def precise(self) -> bool:
        return not any(f.kind == "imprecise" for f in self.findings)

    @property
    def healthy(self) -> bool:
        return self.vanishing_free and self.precise and not self.unanalysed


def check_gradient_health(
    model: onnx.ModelProto,
    input_ranges: Optional[dict] = None,
    grads: Optional[Sequence[str]] = None,
    precision: str = "fp16",
    rel_tol: float = 1.0,
) -> GradHealthReport:
    """Check the gradients of ``model`` for underflow and rounding loss.

    :param input_ranges: ``{input: (lo, hi)}`` over every graph input of the training step: data,
        parameters and the upstream gradient. Without a box every magnitude is unbounded.
    :param grads: tensors to check; defaults to the graph outputs.
    :param precision: ``"fp32"``, ``"fp16"`` or ``"bf16"``, the format the step runs in.
    :param rel_tol: a gradient is imprecise when its roundoff bound is at least this fraction of
        its magnitude bound.
    """
    if precision not in _NORMAL_MIN:
        raise ValueError(f"unknown precision {precision!r}")
    res = _interval.propagate(model, input_ranges)
    rounding = _fp_error.roundoff_bound(
        model, input_ranges, precision=precision, input_exact=False
    )
    names = list(grads) if grads else [o.name for o in model.graph.output]
    normal_min, subnormal_min = _NORMAL_MIN[precision], _SUBNORMAL_MIN[precision]
    findings: List[GradFinding] = []
    unanalysed: List[str] = []
    for g in names:
        if g not in res.intervals or g not in rounding.errors:
            unanalysed.append(g)
            continue
        mag = _magnitude(res.intervals[g])
        err = rounding.bound(g)
        if mag == 0.0:
            findings.append(
                GradFinding(g, "dead", 0.0, err, "gradient is zero over the box")
            )
        elif mag <= subnormal_min / 2:
            findings.append(
                GradFinding(
                    g,
                    "flushed",
                    mag,
                    err,
                    f"|g| <= {mag:.3g} rounds to zero in {precision}",
                )
            )
        elif mag < normal_min:
            findings.append(
                GradFinding(
                    g,
                    "subnormal",
                    mag,
                    err,
                    f"|g| <= {mag:.3g} is subnormal in {precision}",
                )
            )
        if mag > 0.0 and (not np.isfinite(err) or err >= rel_tol * mag):
            findings.append(
                GradFinding(
                    g,
                    "imprecise",
                    mag,
                    err,
                    f"roundoff {err:.3g} against magnitude {mag:.3g} in {precision}",
                )
            )
    return GradHealthReport(precision, findings, unanalysed)


def check_backward_health(
    b: "qat_graph.GraphBuilder",
    shapes: Dict[str, Sequence[Union[int, str]]],
    grads: Dict[str, str],
    input_ranges: Optional[dict] = None,
    precision: str = "fp16",
    rel_tol: float = 1.0,
    elem_types: Optional[Dict[str, int]] = None,
) -> GradHealthReport:
    """Check the gradients :func:`onnxsim.graph_grad.build_backward` appended to ``b``.

    ``b`` must hold the forward nodes as well as the backward ones (the normal case: the same
    builder the forward was built with). ``grads`` is the ``{target: gradient tensor}`` map that
    call returned, and ``shapes`` the static shapes of the step's free inputs, as
    ``build_backward`` takes them. ``input_ranges`` is keyed by those free inputs. ``elem_types``
    gives the ONNX element type of any free input that is not float32 (indices, masks).
    """
    grad_names = sorted(set(grads.values()))
    model = _builder_model(b, shapes, grad_names, elem_types or {})
    return check_gradient_health(
        model,
        input_ranges,
        grads=grad_names,
        precision=precision,
        rel_tol=rel_tol,
    )


def _builder_model(
    b: "qat_graph.GraphBuilder",
    shapes: Dict[str, Sequence[Union[int, str]]],
    outputs: Sequence[str],
    elem_types: Dict[str, int],
) -> onnx.ModelProto:
    produced = {o for n in b.nodes for o in n.output if o}
    constant = {t.name for t in b.initializer}
    free: List[str] = []
    for n in b.nodes:
        for name in n.input:
            if name and name not in produced | constant and name not in free:
                free.append(name)
    missing = [name for name in free if name not in shapes]
    if missing:
        raise ValueError(f"no static shape for step inputs {missing}")
    inputs = [
        onnx.helper.make_tensor_value_info(
            name,
            elem_types.get(name, onnx.TensorProto.FLOAT),
            list(shapes[name]),
        )
        for name in free
    ]
    graph = onnx.helper.make_graph(
        b.nodes,
        "backward_health",
        inputs,
        [onnx.helper.make_empty_tensor_value_info(o) for o in outputs],
        initializer=b.initializer,
    )
    opset_imports = [onnx.helper.make_opsetid("", _OPSET)]
    domains = sorted({fn.domain for fn in b.functions})
    opset_imports += [onnx.helper.make_opsetid(d, 1) for d in domains]
    model = onnx.helper.make_model(
        graph, opset_imports=opset_imports, functions=list(b.functions)
    )
    model.ir_version = _IR_VERSION
    if b.functions:
        model = onnx.inliner.inline_local_functions(model)
    return model


def _magnitude(iv: Tuple[np.ndarray, np.ndarray]) -> float:
    lo, hi = iv
    if np.size(lo) == 0:
        return 0.0
    return float(np.max(np.maximum(np.abs(lo), np.abs(hi))))
