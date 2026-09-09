"""Formal check for the uniform-affine quantize/dequantize round trip.

``onnxsim.calibration.quantize_static`` (and every other uniform-affine
scheme in this repo) inserts a standard ONNX QuantizeLinear/DequantizeLinear
pair around each quantized tensor::

    q  = saturate(round(x / scale) + zero_point)
    x' = (q - zero_point) * scale

This is the first of two lemmas needed for a formal (not sampled)
worst-case error bound on a quantized model: the round-trip bound proved
here, and a follow-up MAC-composition lemma (not yet in this repo) that
combines per-element round-trip bounds into an end-to-end bound for a
quantized MatMul/Conv/Gemm.

Unlike the other tests in this file family (which each isolate one of
onnxsim's own optimizer rewrites via ``skipped_optimizers``), this one
isolates a property of two *ONNX* operators onnxsim's quantization relies
on but does not implement itself -- there is no onnxsim pass to isolate
here, so the differential check instead runs the two ops through
onnxruntime directly.

Soundness only needs one fact about ``round``: it returns *some* integer
within 0.5 of its argument -- true for round-half-to-even (ONNX's
documented tie-breaking rule) and for every other correct
nearest-integer rounding rule, since the tie-breaking choice only matters
for exact half-integers, where either neighbor is still within the bound.
Modeling ``round(v)`` this way, rather than picking one specific tie rule,
proves the bound for any of them at once.

The proof assumes no saturation (the quantized code stays in range):
saturation clips to a fixed code and can introduce unbounded error by
construction -- a real, separate failure mode of a too-narrow calibration
range, not something ``scale / 2`` ever bounds -- so it is an explicit side
condition here, not silently ignored.
"""

import numpy as np
import pytest
from _formal_verify_common import prove, z3
from onnx import parser

ort = pytest.importorskip("onnxruntime")


def test_quantize_round_trip_is_sound():
    x, scale, zero_point = z3.Reals("x scale zero_point")
    n = z3.Int("n")  # round(x / scale): *some* integer within 0.5 of x/scale.
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        scale > 0,
        n - x / scale <= half,
        x / scale - n <= half,
    )
    # Not saturated: the quantized code is exactly n + zero_point (no clip),
    # so dequantizing recovers n * scale -- zero_point cancels exactly.
    quantized_code = n + zero_point
    dequantized = (quantized_code - zero_point) * scale

    error = x - dequantized
    prove(z3.Implies(hypotheses, z3.And(error <= scale / 2, -error <= scale / 2)))


def test_quantize_round_trip_matches_onnxruntime():
    # Differential check: run the real QuantizeLinear/DequantizeLinear pair
    # (the ops onnxsim's own quantize_static inserts) through onnxruntime,
    # and confirm every element's round-trip error is within scale/2, for a
    # scale/zero_point/input range chosen so nothing saturates -- matching
    # the proof's own side condition.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1000] X) => (float[1000] Y)
        <float scale = {0.1}, int8 zero_point = {0}>
        {
          q = QuantizeLinear(X, scale, zero_point)
          Y = DequantizeLinear(q, scale, zero_point)
        }
        """
    )
    sess = ort.InferenceSession(model.SerializeToString())
    rng = np.random.default_rng(0)
    # int8 range is [-128, 127]; keep |X / scale| well clear of that so
    # nothing saturates.
    x = (rng.random(1000).astype(np.float32) - 0.5) * 10.0  # in [-5, 5)
    (y,) = sess.run(None, {"X": x})
    np.testing.assert_array_less(np.abs(x - y), 0.1 / 2 + 1e-6)
