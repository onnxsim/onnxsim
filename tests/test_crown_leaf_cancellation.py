"""CROWN must sum the coefficients that reach the same leaf BEFORE concretising them.

``crown.bounds`` used to concretise every arrival at a leaf (a graph input, or a tensor with no
backward rule) on its own, so paths from the same leaf never cancelled: ``Sub(x, x)`` over
``x in [-R, R]`` came out as ``[-2R, 2R]``, exactly the plain-interval answer, instead of 0.
"""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import crown

R = 4e4
BOX = {"x": (-R, R)}


def _model(body, initializer=None, shape="[1,4]"):
    m = parser.parse_model(
        f'<ir_version: 8, opset_import: ["" : 13]> g (float{shape} x) => (float{shape} y) {{ {body} }}'
    )
    m.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    return m


def _bound(model, method="crown", box=BOX):
    r = crown.bounds(model, box, method=method)["y"]
    return np.asarray(r.lo), np.asarray(r.hi)


_W = np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32)

_CANCELS = {
    "sub_x_x": ("y = Sub(x, x)", None),
    "identity_minus_x": ("i = Identity(x)\n y = Sub(i, x)", None),
    "add_x_neg_x": ("n = Neg(x)\n y = Add(x, n)", None),
    "two_identical_matmuls": (
        "a = MatMul(x, W)\n b = MatMul(x, W)\n y = Sub(a, b)",
        {"W": _W},
    ),
}


@pytest.mark.parametrize("case", sorted(_CANCELS))
def test_paths_from_the_same_input_cancel_exactly(case):
    body, init = _CANCELS[case]
    lo, hi = _bound(_model(body, init))
    # exact cancellation, up to the 1e-9 relative widening on the output, never ~R
    assert np.all(np.abs(lo) <= 1e-3) and np.all(np.abs(hi) <= 1e-3), (lo, hi)


def test_cancellation_is_strictly_tighter_than_intervals():
    m = _model("y = Sub(x, x)")
    ibp = crown.bounds(m, BOX, method="ibp")["y"]
    c = crown.bounds(m, BOX, method="crown")["y"]
    assert float(np.max(np.asarray(ibp.hi))) == pytest.approx(2 * R, rel=1e-6)
    assert float(np.max(np.asarray(c.hi))) < 1e-3


def test_partial_cancellation_still_sound_and_never_looser_than_intervals():
    # y = x + (x * 0.25): both arrivals at x merge into one coefficient 1.25 -> width 1.25*(2R)
    m = _model(
        "c = Constant<value = float[4] {0.25, 0.25, 0.25, 0.25}>()\n a = Mul(x, c)\n y = Add(x, a)"
    )
    lo, hi = _bound(m)
    assert np.allclose(hi, 1.25 * R, rtol=1e-6) and np.allclose(
        lo, -1.25 * R, rtol=1e-6
    )


def test_residual_skip_from_the_input_is_sound():
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((6, 6)) * 0.4).astype(np.float32)
    m = _model(
        "a = MatMul(x, W)\n r = Relu(a)\n y = Add(r, x)", {"W": w}, shape="[1,6]"
    )
    box = {"x": (-1.0, 1.0)}
    ibp = crown.bounds(m, box, method="ibp")["y"]
    c = crown.bounds(m, box, method="crown")["y"]
    assert np.all(np.asarray(c.lo) >= np.asarray(ibp.lo) - 1e-9)
    assert np.all(np.asarray(c.hi) <= np.asarray(ibp.hi) + 1e-9)
    sess = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for _ in range(300):
        x = rng.uniform(-1, 1, (1, 6)).astype(np.float32)
        y = sess.run(None, {"x": x})[0]
        assert np.all(y >= np.asarray(c.lo) - 1e-4) and np.all(
            y <= np.asarray(c.hi) + 1e-4
        )


def test_unbounded_leaf_stays_unbounded_not_nan():
    lo, hi = _bound(_model("y = Sub(x, x)"), box={})
    assert not np.isnan(lo).any() and not np.isnan(hi).any()


def test_two_separate_relus_do_not_cancel_but_stay_sound_and_inside_intervals():
    # Linear paths cancel; two separate nonlinear nodes are relaxed independently, so
    # Relu(x) - Relu(x) is NOT exactly 0 here. (Pairing such nodes is what zonotope /
    # backward_diff do.) It must stay sound, i.e. contain 0, and never exceed the interval bound.
    m = _model("a = Relu(x)\n b = Relu(x)\n y = Sub(a, b)")
    lo, hi = _bound(m)
    ilo, ihi = _bound(m, method="ibp")
    assert np.all(lo <= 0.0) and np.all(hi >= 0.0)
    assert np.all(lo >= ilo - 1e-9) and np.all(hi <= ihi + 1e-9)
