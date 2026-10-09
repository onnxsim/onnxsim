"""Certified error bound for a graph rewrite that is exact in real arithmetic.

BatchNormalization folding (``fuse_bn_into_conv``) and constant folding change the float
execution but not the real function. For two graphs ``original`` and ``fused`` with the same
inputs and outputs, :func:`fusion_error_bound` returns, per output, a bound on
``|run(original) - run(fused)|`` over every input in the box, in three parts:

* ``roundoff_original`` / ``roundoff_fused``: :func:`onnxsim.fp_error.roundoff_bound` for each
  graph. The fused graph's folded constants are read as the stored values (exact in the
  real semantics), so the rounding that created them is not counted here.
* ``real_difference``: :func:`onnxsim.zonotope.bound_difference` on the two real semantics.
  The folded constants are real numbers equal to the stored values, so this term carries the
  rounding of the fold (``w * gamma / sqrt(var + eps)`` stored in fp32) and propagates it through
  every later layer.

The absolute bound is their sum (:func:`onnxsim.fp_error.tolerance_for`). ``ulps`` expresses it
in units of the precision's unit roundoff ``u`` times the output's magnitude bound
(``absolute / (u * max|output|)``, not the exact ULP of the output).

Assumptions, all inherited from :mod:`onnxsim.fp_error`:

* round-to-nearest arithmetic in ``precision``, with the stated elementary-function error model;
* no overflow or underflow beyond the modelled subnormal term;
* the input box ``input_ranges`` (merged with ``onnxsim.ranges`` annotations) is the whole input
  domain; without a box every magnitude is unbounded and every bound is ``inf``;
* the two graphs have identical input and output names and shapes.

The bound is sound, not tight: intervals and zonotopes widen with depth.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import onnx

from . import fp_error as _fp_error
from . import interval as _interval


def fusion_error_bound(
    original: onnx.ModelProto,
    fused: onnx.ModelProto,
    input_ranges: Optional[Dict[str, Tuple]] = None,
    precision: str = "fp32",
) -> dict:
    """Certified per-output bound on ``|run(original) - run(fused)|`` for inputs in the box.

    Returns ``{"outputs": {name: {...}}, "worst_absolute": float, "worst_ulps": float,
    "notes": [...]}``. Each output entry has ``absolute``, ``ulps``, ``real_difference``,
    ``roundoff_original`` and ``roundoff_fused``. ``inf`` means the analysis could not bound it.
    """
    if precision not in _fp_error.PRECISIONS:
        raise ValueError(
            f"unknown precision {precision!r}; expected one of {sorted(_fp_error.PRECISIONS)}"
        )
    orig_outputs = [o.name for o in original.graph.output]
    fused_outputs = [o.name for o in fused.graph.output]
    if orig_outputs != fused_outputs:
        raise ValueError(f"graph outputs differ: {orig_outputs} vs {fused_outputs}")

    tol = _fp_error.tolerance_for(original, fused, input_ranges, precision)
    magnitudes = _interval.propagate(original, input_ranges).intervals
    u = _fp_error.PRECISIONS[precision].u

    outputs: Dict[str, dict] = {}
    for name in orig_outputs:
        absolute = tol.atol[name]
        mag = _magnitude(magnitudes.get(name))
        outputs[name] = {
            "absolute": absolute,
            "ulps": _ulps(absolute, u, mag),
            "real_difference": tol.real_difference[name],
            "roundoff_original": tol.roundoff_orig[name],
            "roundoff_fused": tol.roundoff_simplified[name],
        }
    return {
        "outputs": outputs,
        "worst_absolute": tol.worst,
        "worst_ulps": max((o["ulps"] for o in outputs.values()), default=0.0),
        "notes": list(tol.notes),
    }


def _magnitude(iv) -> float:
    if iv is None:
        return float("inf")
    lo, hi = iv
    if np.size(lo) == 0:
        return 0.0
    return float(np.max(np.maximum(np.abs(lo), np.abs(hi))))


def _ulps(absolute: float, u: float, magnitude: float) -> float:
    if absolute == 0.0:
        return 0.0
    if not np.isfinite(absolute) or not np.isfinite(magnitude) or magnitude == 0.0:
        return float("inf")
    return absolute / (u * magnitude)
