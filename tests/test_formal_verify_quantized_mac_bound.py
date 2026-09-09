"""Formal check for the quantized-MAC composition bound -- not an onnxsim
pass or a single ONNX op, but a derived lemma combining the round-trip
bound proved in test_formal_verify_quantize_round_trip.py with a
dot-product's own linearity. This is the second of the two lemmas needed
for an end-to-end worst-case quantization error bound (the round-trip
bound is the first): every output element of a quantized MatMul/Conv/Gemm
*is* exactly this kind of dot product.

Claim: if every quantized weight ``w_i`` and activation ``x_i`` satisfies
its own round-trip bound (``|w_i - qw_i| <= eps_w``, ``|x_i - qx_i| <=
eps_x``, each ``scale / 2`` per test_formal_verify_quantize_round_trip.py),
then the quantized dot product ``sum(qw_i * qx_i)`` deviates from the float
dot product ``sum(w_i * x_i)`` by at most::

    eps_x * sum(|w_i|) + eps_w * sum(|x_i|) + K * eps_w * eps_x

Per-tap derivation: writing ``qw = w - ew``, ``qx = x - ex``, a single tap's
error ``w*x - qw*qx`` expands to ``w*ex + ew*x - ew*ex``, whose absolute
value the triangle inequality bounds by
``|w|*|ex| + |ew|*|x| + |ew|*|ex| <= |w|*eps_x + eps_w*|x| + eps_w*eps_x``.
Summing K taps and applying the triangle inequality once more over the sum
gives the claim above. Z3 checks the *sum* bound directly in one shot
rather than trusting that by-hand algebra composes correctly across taps --
an error in how the per-tap bounds combine (e.g. an accidentally tighter or
looser constant, or a missing cross term) would show up as a counterexample
here even though each individual tap's own bound is correct in isolation.
"""

import numpy as np
import onnx
import pytest
from _formal_verify_common import prove, z3
from onnx import parser

ort = pytest.importorskip("onnxruntime")

_K = 2  # concrete number of MAC taps -- enough to exercise the cross-tap sum


def _abs(v):
    return z3.If(v >= 0, v, -v)


def test_quantized_mac_error_is_bounded():
    w = [z3.Real(f"w{i}") for i in range(_K)]
    x = [z3.Real(f"x{i}") for i in range(_K)]
    ew = [z3.Real(f"ew{i}") for i in range(_K)]  # w_i - qw_i
    ex = [z3.Real(f"ex{i}") for i in range(_K)]  # x_i - qx_i
    eps_w, eps_x = z3.Reals("eps_w eps_x")

    hypotheses = z3.And(
        eps_w >= 0,
        eps_x >= 0,
        *[_abs(ew[i]) <= eps_w for i in range(_K)],
        *[_abs(ex[i]) <= eps_x for i in range(_K)],
    )

    qw = [w[i] - ew[i] for i in range(_K)]
    qx = [x[i] - ex[i] for i in range(_K)]

    float_dot = sum(w[i] * x[i] for i in range(_K))
    quant_dot = sum(qw[i] * qx[i] for i in range(_K))
    error = float_dot - quant_dot

    bound = (
        eps_x * sum(_abs(w[i]) for i in range(_K))
        + eps_w * sum(_abs(x[i]) for i in range(_K))
        + _K * eps_w * eps_x
    )

    prove(z3.Implies(hypotheses, z3.And(error <= bound, -error <= bound)))


def test_quantized_mac_bound_matches_onnxruntime():
    # Differential check: quantize a real activation and weight (the same
    # QuantizeLinear/DequantizeLinear formula validated in
    # test_formal_verify_quantize_round_trip.py) with onnxruntime, MatMul
    # the dequantized results, and confirm every output element's error
    # against the float MatMul stays within the bound the proof above
    # derives -- for scale/value ranges chosen so nothing saturates.
    K, N = 4, 3
    rng = np.random.default_rng(0)
    x = (rng.random(K).astype(np.float32) - 0.5) * 10.0  # in [-5, 5)
    w = (rng.random((K, N)).astype(np.float32) - 0.5) * 10.0

    scale_x, scale_w = np.float32(0.1), np.float32(0.1)
    eps_x, eps_w = scale_x / 2, scale_w / 2

    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4] X) => (float[3] Y)
        <float scale_x = {0.1}, int8 zp_x = {0}, float scale_w = {0.1}, int8 zp_w = {0}>
        {
          qx = QuantizeLinear(X, scale_x, zp_x)
          dqx = DequantizeLinear(qx, scale_x, zp_x)
          qw = QuantizeLinear(W, scale_w, zp_w)
          dqw = DequantizeLinear(qw, scale_w, zp_w)
          Y = MatMul(dqx, dqw)
        }
        """
    )
    model.graph.initializer.append(onnx.numpy_helper.from_array(w, "W"))
    sess = ort.InferenceSession(model.SerializeToString())
    (y_quant,) = sess.run(None, {"X": x})

    y_float = x @ w
    error = np.abs(y_float - y_quant)
    bound = eps_x * np.abs(w).sum(axis=0) + eps_w * np.abs(x).sum() + K * eps_w * eps_x
    assert np.all(error <= bound + 1e-6)
