"""Formal check for FuseBNIntoConv (fuse_bn_into_conv.h).

The rewrite folds BatchNormalization's affine transform into the preceding
Conv's weight/bias::

    new_W = W * s
    new_b = (B - mean) * s + bias
    s     = scale / sqrt(var + eps)

where ``B`` is the Conv's own pre-existing bias, or 0 if it had none. This
exact formula is also what ``tests/test_fusion_patterns.py``'s
``test_fuse_conv_bn_into_conv_double`` validates numerically against the
compiled pass in float64.

Two independent facts make the fusion sound, combined here into one real
(non-floating-point) arithmetic identity per output element:

1. Conv is linear in its weights for a fixed input, so scaling every weight
   by ``s`` scales the pre-BN output by ``s`` (modeled below with a 2-tap dot
   product -- enough taps to exercise linearity/distributivity, not just a
   single-weight special case).
2. BN's own affine transform, expanded algebraically, is exactly
   ``s * conv_out + (B - mean) * s + bias``.

``sqrt`` is modeled as a real variable ``sq`` constrained by
``sq * sq == var + eps`` and ``sq > 0`` -- the same side condition
fuse_bn_into_conv.h's own division relies on -- rather than as an
uninterpreted function, so Z3's real (nonlinear) arithmetic can discharge the
identity directly instead of reasoning about an opaque sqrt symbol.
"""

import numpy as np
import onnx
from onnx import parser

from _formal_verify_common import prove, simplify_isolated, z3


def test_fuse_bn_into_conv_is_sound():
    w0, w1, x0, x1, B = z3.Reals("w0 w1 x0 x1 B")
    scale, bias, mean, var, eps = z3.Reals("scale bias mean var eps")
    sq, s = z3.Reals("sq s")

    conv_out = w0 * x0 + w1 * x1 + B  # Conv(x, W) + B

    side_conditions = z3.And(
        sq * sq == var + eps,
        sq > 0,
        s * sq == scale,  # s == scale / sq, kept polynomial (no division)
    )

    original = (conv_out - mean) * s + bias  # BatchNormalization, in terms of s
    fused_conv_out = (w0 * s) * x0 + (w1 * s) * x1  # Conv(x, W * s), no bias
    fused_bias = (B - mean) * s + bias
    fused = fused_conv_out + fused_bias

    prove(z3.Implies(side_conditions, original == fused))


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def test_fuse_bn_into_conv_pass_matches():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 3, 3, 3))
    scale = rng.random(8) + 0.5
    bias = rng.standard_normal(8)
    mean = rng.standard_normal(8)
    var = rng.random(8) + 0.5
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[1,3,16,16] X) => (float[1,8,16,16] Y)
        {
          c = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(X, W)
          Y = BatchNormalization(c, scale, bias, mean, var)
        }
        """
    )
    model.graph.initializer.extend(
        [
            _f32(W, "W"),
            _f32(scale, "scale"),
            _f32(bias, "bias"),
            _f32(mean, "mean"),
            _f32(var, "var"),
        ]
    )
    sim_model, ops = simplify_isolated(model, "fuse_bn_into_conv")
    assert ops["BatchNormalization"] == 0
    assert ops["Conv"] == 1

    conv_node = next(n for n in sim_model.graph.node if n.op_type == "Conv")
    by_name = {
        init.name: onnx.numpy_helper.to_array(init)
        for init in sim_model.graph.initializer
    }
    fused_w = by_name[conv_node.input[1]]
    fused_b = by_name[conv_node.input[2]]

    # The exact formula just proven sound above, computed independently with
    # numpy (eps matches fuse_bn_into_conv.h's hardcoded default of 1e-5).
    s = scale / np.sqrt(var + 1e-5)
    expected_w = W * s.reshape(-1, 1, 1, 1)
    expected_b = (0.0 - mean) * s + bias
    np.testing.assert_allclose(fused_w, expected_w, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(fused_b, expected_b, rtol=1e-4, atol=1e-6)
