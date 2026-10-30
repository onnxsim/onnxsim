"""Calibration of a NEW_OPS graph (row-scaled norm: Abs, ReduceMax, ReduceMean over
128- and 2560-wide rows) sees the constant MatMul weights rounded to 8 bits per output
column (``u16_retarget._u8_weights``), as Pulsar2 does. Host checks (a parser-built
graph, no build): the rounding grid, dynamic MatMuls left alone, and the default
(``weight_u8=None``: on for NEW_OPS graphs only, older graphs unchanged).

The scratch-gated check reads a native Pulsar2 build of the row-scaled norm (the
``ident`` build: ``AXERA_U16_IDENT_BUILD``, skipped when absent, so CI with the
variable at ``/nonexistent`` skips it) and compares ``calibrate_graph``'s scales with
the build's own ``quant_axmodel.json`` in float32 ulps.
"""

import os
import sys

import numpy as np
import onnx
import pytest
from onnx import parser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import u16_retarget as ur  # noqa: E402

F = np.float32
IDENT = os.environ.get(
    "AXERA_U16_IDENT_BUILD",
    "/mnt/data/cache/claude-work/tmp/qwen-fuse/Q/ident/"
    "templates_alyrrsqkrk4096_v0_s777",
)


def _model(body, inputs="(float[1,4] x)", outputs="(float[1,3] y)"):
    return parser.parse_model(
        f'<ir_version: 8, opset_import: ["": 17]> agraph {inputs} => {outputs} {body}'
    )


def _ulps(a, b):
    return abs(int(np.float32(a).view(np.int32)) - int(np.float32(b).view(np.int32)))


def test_u8_weights_round_each_column_on_its_own_grid():
    # column 0 absmax 1.0: s0 = 1.0 / 127.5, 1.0 / s0 = 127.5 -> rint 128 -> clipped 127,
    # 0.5 / s0 = 63.75 -> 64. Column 1 absmax 0.7: s1 = 0.7 / 127.5; 0.25 / s1 = 45.54 ->
    # 46, -0.7 / s1 = -127.5 -> rint -128 (even) and in range.
    m = _model(
        "<float[2,2] W = {1.0, 0.25, 0.5, -0.7}> { y = MatMul(x, W) }",
        inputs="(float[1,2] x)",
        outputs="(float[1,2] y)",
    )
    w_in = onnx.numpy_helper.to_array(m.graph.initializer[0]).copy()
    w = onnx.numpy_helper.to_array(ur._u8_weights(m).graph.initializer[0])
    s0 = np.abs(w_in).max(axis=0)[0] / F(127.5)
    s1 = np.abs(w_in).max(axis=0)[1] / F(127.5)
    assert w[0, 0] == F(127) * s0
    assert w[1, 0] == F(64) * s0
    assert w[0, 1] == F(46) * s1
    assert w[1, 1] == F(-128) * s1
    # the input model is untouched
    assert np.array_equal(onnx.numpy_helper.to_array(m.graph.initializer[0]), w_in)


def test_u8_weights_leave_dynamic_matmul_alone():
    m = _model("{ y = MatMul(x, x2) }", inputs="(float[1,4] x, float[4,3] x2)")
    out = ur._u8_weights(m)
    assert len(out.graph.initializer) == 0
    assert out.graph.node[0].op_type == "MatMul"


@pytest.fixture
def build(tmp_path):
    """A build directory (``model.onnx`` only) and its samples: calibrate_graph takes
    them from ``samples`` in place of the dataset tars."""
    rng = np.random.default_rng(3)
    w = rng.uniform(-0.1, 0.1, (4, 3)).astype(F)

    def make(body, name):
        m = _model(
            f"<float[4,3] W = {{{', '.join(repr(float(v)) for v in w.reshape(-1))}}}> {body}"
        )
        d = tmp_path / name
        d.mkdir()
        onnx.save(m, str(d / "model.onnx"))
        samples = {"x": [rng.standard_normal((1, 4)).astype(F) for _ in range(6)]}
        return str(d), samples

    return make


def test_default_is_on_for_new_ops_only(build):
    plain, s = build("{ y = MatMul(x, W) }", "plain")
    assert ur.calibrate_graph(plain, samples=s) == ur.calibrate_graph(
        plain, samples=s, weight_u8=False
    )
    new, s2 = build(
        "{ a = Abs(x) \n mx = ReduceMax(a) \n d = Div(x, mx) \n y = MatMul(d, W) }",
        "new",
    )
    on = ur.calibrate_graph(new, samples=s2)
    assert on == ur.calibrate_graph(new, samples=s2, weight_u8=True)
    assert on != ur.calibrate_graph(new, samples=s2, weight_u8=False)


#: the ident build's native scales (``quant_axmodel.json``), float32 values
NATIVE = {
    "m0_1": 1.2851435258198762e-06,
    "m0": 1.406031856276968e-06,
    "qm": 2.361458791710902e-05,
    "km": 2.202740688517224e-05,
    "um": 1.5259012798196636e-05,
    "m_1": 1.5258838175213896e-05,
    "m": 1.5258858184097335e-05,
}
#: the ReduceMean outputs with the float weights are 0.1-0.24% off (26987 ulp for m0_1)
#: with Pulsar2's float ops (``u16_retarget._pulsar_forward``, the default for NEW_OPS
#: graphs) the scales sit 0-5 ulps off; ``qm`` (a 128-wide mean) is 3 ulps off, one row
#: of 32 short by 1 ulp in its sum (the width-128 ReduceMean order is still open)
ULP = {"m0_1": 0, "m0": 5, "qm": 3, "km": 0, "um": 0, "m_1": 0, "m": 0}


@pytest.mark.skipif(
    not os.path.exists(os.path.join(IDENT, "model.onnx")),
    reason="needs the native ident build (AXERA_U16_IDENT_BUILD)",
)
def test_ident_build_reducemean_scales_match_native():
    native = ur.quant_of(IDENT)
    cal = ur.calibrate_graph(IDENT)
    plain = ur.calibrate_graph(IDENT, weight_u8=False)
    for k, want in NATIVE.items():
        assert native[k][0] == want, k  # the stored values are the build's
        assert _ulps(cal[k][0], want) <= ULP[k], k
    # the float weights miss the two windowed means by thousands of ulps
    assert _ulps(plain["m0_1"][0], NATIVE["m0_1"]) > 1000
