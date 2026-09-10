"""Tests for ``onnxsim.bn_recovery`` -- reinserting a trainable
``BatchNormalization`` after a Conv/ConvTranspose a fold (e.g.
``onnxsim/passes/fuse_bn_into_conv.h``, exercised via ``onnxsim.simplify``)
already folded away, and (optionally) fitting it to real data. See
``onnxsim/bn_recovery.py``'s own docstring for the technique and what it
does and does not claim to recover.
"""

import collections

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.bn_recovery import (
    RecoveredBatchNorm,
    _conv_output_channels,
    calibrate_recovered_bn,
    find_recoverable_convs,
    find_recovered_bn_nodes,
    insert_identity_bn,
    recover_batch_norm,
)

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=17, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _run(model, feeds):
    import onnxruntime as _ort

    sess = _ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, feeds)
    return dict(zip(out_names, outs))


def _identity_conv_model(c, bias, batch=1, spatial=4):
    # A 1x1 Conv whose weight is the identity matrix, so its output is
    # exactly `x + bias` (per channel, no cross-channel mixing, no spatial
    # change) -- makes the pre-BatchNorm activation's own per-channel
    # statistics computable in closed form from the calibration inputs
    # alone, the same "known injected value" idiom
    # tests/test_bias_correction.py uses.
    w = np.zeros((c, c, 1, 1), dtype=np.float32)
    for i in range(c):
        w[i, i, 0, 0] = 1.0
    return _model(
        f"""
        g (float[{batch},{c},{spatial},{spatial}] x) => (float[{batch},{c},{spatial},{spatial}] y)
        {{
          y = Conv<kernel_shape = [1, 1]>(x, w, b)
        }}
        """,
        initializer=[_f32(w, "w"), _f32(bias, "b")],
    )


def test_find_recoverable_convs_skips_conv_already_followed_by_bn():
    model = _model(
        """
        g (float[1,3,4,4] x) => (float[1,3,4,4] y2)
        {
          y1 = Conv<kernel_shape = [1, 1]>(x, w1, b1)
          y1r = Relu(y1)
          y2 = Conv<kernel_shape = [1, 1]>(y1r, w2, b2)
          y2n = BatchNormalization(y2, scale, bias, mean, var)
        }
        """,
        initializer=[
            _f32(np.eye(3).reshape(3, 3, 1, 1), "w1"),
            _f32(np.zeros(3), "b1"),
            _f32(np.eye(3).reshape(3, 3, 1, 1), "w2"),
            _f32(np.zeros(3), "b2"),
            _f32(np.ones(3), "scale"),
            _f32(np.zeros(3), "bias"),
            _f32(np.zeros(3), "mean"),
            _f32(np.ones(3), "var"),
        ],
    )
    onnx.checker.check_model(model)
    assert find_recoverable_convs(model) == ["y1"]


def test_insert_identity_bn_is_numerically_a_no_op():
    rng = np.random.default_rng(0)
    bias = rng.standard_normal(4).astype(np.float32) * 0.1
    model = _identity_conv_model(4, bias)
    onnx.checker.check_model(model)

    recovered_model, recovered = insert_identity_bn(model)
    onnx.checker.check_model(recovered_model)
    assert len(recovered) == 1
    assert recovered[0].conv_output_name == "y"
    assert recovered[0].channels == 4
    assert [n.op_type for n in recovered_model.graph.node] == [
        "Conv",
        "BatchNormalization",
    ]

    x = rng.standard_normal((1, 4, 4, 4)).astype(np.float32)
    before = _run(model, {"x": x})["y"]
    after = _run(recovered_model, {"x": x})["y"]
    np.testing.assert_allclose(after, before, rtol=1e-4, atol=1e-6)


def test_insert_identity_bn_on_conv_transpose():
    rng = np.random.default_rng(1)
    c = 3
    w = np.zeros((c, c, 1, 1), dtype=np.float32)
    for i in range(c):
        w[i, i, 0, 0] = 1.0
    bias = rng.standard_normal(c).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[1,{c},4,4] x) => (float[1,{c},4,4] y)
        {{
          y = ConvTranspose<kernel_shape = [1, 1]>(x, w, b)
        }}
        """,
        initializer=[_f32(w, "w"), _f32(bias, "b")],
    )
    onnx.checker.check_model(model)

    recovered_model, recovered = insert_identity_bn(model)
    onnx.checker.check_model(recovered_model)
    assert recovered[0].channels == c

    x = rng.standard_normal((1, c, 4, 4)).astype(np.float32)
    before = _run(model, {"x": x})["y"]
    after = _run(recovered_model, {"x": x})["y"]
    np.testing.assert_allclose(after, before, rtol=1e-4, atol=1e-6)


def test_conv_output_channels_grouped_conv_transpose():
    # A pure metadata computation, not a runnable graph (grouped
    # ConvTranspose's own conv math isn't what's under test here) -- a bare
    # NodeProto/TensorProto built with onnx.helper is the natural fit,
    # there is no graph body for the text parser to read.
    node = onnx.helper.make_node(
        "ConvTranspose",
        ["x", "w"],
        ["y"],
        kernel_shape=[1, 1],
        group=2,
    )
    # ConvTranspose weight layout: (in_channels, out_channels / group, kH, kW)
    weight = onnx.numpy_helper.from_array(
        np.zeros((4, 3, 1, 1), dtype=np.float32), name="w"
    )
    assert _conv_output_channels(node, weight) == 6  # 3 * group(2)


def test_insert_identity_bn_explicit_name_must_be_a_conv_output():
    model = _identity_conv_model(2, np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError):
        insert_identity_bn(model, conv_output_names=["does_not_exist"])


def test_calibrate_recovered_bn_recovers_known_statistics():
    rng = np.random.default_rng(2)
    c = 4
    bias = np.array([0.5, -1.0, 2.0, 0.0], dtype=np.float32)
    model = _identity_conv_model(c, bias, batch=2, spatial=3)
    recovered_model, recovered = insert_identity_bn(model)
    r = recovered[0]

    calib = [
        {"x": rng.standard_normal((2, c, 3, 3)).astype(np.float32)} for _ in range(6)
    ]
    calibrated = calibrate_recovered_bn(recovered_model, calibration_data=calib)
    onnx.checker.check_model(calibrated)

    # z = x + bias (identity 1x1 conv, per-channel, no spatial change), so
    # the expected per-channel mean/var are computable directly from the
    # same calibration inputs.
    all_x = np.concatenate(
        [np.moveaxis(b["x"], 1, -1).reshape(-1, c) for b in calib], axis=0
    )
    expected_mean = all_x.mean(axis=0) + bias
    expected_var = all_x.var(axis=0)

    init_by_name = {t.name: t for t in calibrated.graph.initializer}
    got_mean = onnx.numpy_helper.to_array(init_by_name[r.mean_name])
    got_var = onnx.numpy_helper.to_array(init_by_name[r.var_name])
    np.testing.assert_allclose(got_mean, expected_mean, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(got_var, expected_var, rtol=1e-4, atol=1e-5)
    # scale/bias stay at their identity values with no target_model.
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(init_by_name[r.scale_name]), np.ones(c)
    )
    np.testing.assert_allclose(
        onnx.numpy_helper.to_array(init_by_name[r.bias_name]), np.zeros(c)
    )


def test_calibrate_recovered_bn_with_target_model_matches_target_distribution():
    rng = np.random.default_rng(3)
    c = 3
    bias = np.zeros(c, dtype=np.float32)
    target_bias = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    student = _identity_conv_model(c, bias, batch=1, spatial=4)
    target = _identity_conv_model(c, target_bias, batch=1, spatial=4)

    recovered_model, _ = insert_identity_bn(student)
    calib = [
        {"x": rng.standard_normal((1, c, 4, 4)).astype(np.float32)} for _ in range(6)
    ]
    calibrated = calibrate_recovered_bn(
        recovered_model, calibration_data=calib, target_model=target
    )
    onnx.checker.check_model(calibrated)

    # The recovered node's own output should now closely reproduce
    # target's activation on the same calibration data it was fit on.
    for batch in calib:
        got = _run(calibrated, batch)["y"]
        want = _run(target, batch)["y"]
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4)


def test_find_recovered_bn_nodes_roundtrips_through_serialization():
    model = _identity_conv_model(3, np.zeros(3, dtype=np.float32))
    recovered_model, recovered = insert_identity_bn(model)

    reloaded = onnx.ModelProto()
    reloaded.ParseFromString(recovered_model.SerializeToString())

    found = find_recovered_bn_nodes(reloaded)
    assert len(found) == 1
    assert found[0].node_name == recovered[0].node_name
    assert found[0].scale_name == recovered[0].scale_name
    assert found[0].channels == recovered[0].channels


def test_recover_batch_norm_end_to_end_after_fusion():
    # The scenario from onnxsim.bn_recovery's own docstring: a graph whose
    # BatchNormalization has already been folded into its preceding Conv (by
    # onnxsim's own fuse_bn_into_conv, exercised here through
    # onnxsim.simplify).
    rng = np.random.default_rng(4)
    c_in, c_out = 3, 4
    w = rng.standard_normal((c_out, c_in, 3, 3)).astype(np.float32) * 0.2
    b = rng.standard_normal(c_out).astype(np.float32) * 0.05
    scale = (0.8 + 0.4 * rng.random(c_out)).astype(np.float32)
    shift = rng.standard_normal(c_out).astype(np.float32) * 0.1
    mean = rng.standard_normal(c_out).astype(np.float32) * 0.1
    var = (0.5 + rng.random(c_out)).astype(np.float32)

    original = _model(
        f"""
        g (float[1,{c_in},8,8] x) => (float[1,{c_out},8,8] y)
        {{
          conv = Conv<kernel_shape = [3, 3], pads = [1, 1, 1, 1]>(x, w, b)
          y = BatchNormalization(conv, scale, shift, mean, var)
        }}
        """,
        initializer=[
            _f32(w, "w"),
            _f32(b, "b"),
            _f32(scale, "scale"),
            _f32(shift, "shift"),
            _f32(mean, "mean"),
            _f32(var, "var"),
        ],
    )
    onnx.checker.check_model(original)

    fused, check_ok = onnxsim.simplify(original, check_n=2)
    assert check_ok
    op_counts = collections.Counter(n.op_type for n in fused.graph.node)
    assert op_counts.get("BatchNormalization", 0) == 0  # fused away

    x = rng.standard_normal((1, c_in, 8, 8)).astype(np.float32)
    original_out = _run(original, {"x": x})["y"]
    fused_out = _run(fused, {"x": x})["y"]
    np.testing.assert_allclose(fused_out, original_out, rtol=1e-4, atol=1e-5)

    # insert_identity_bn alone -- no calibration data, so no distribution
    # assumption enters at all -- is exactly the numeric no-op the module's
    # own docstring promises: the fused graph's own output is untouched.
    structurally_recovered, _ = insert_identity_bn(fused)
    onnx.checker.check_model(structurally_recovered)
    assert "BatchNormalization" in {
        n.op_type for n in structurally_recovered.graph.node
    }
    bare_out = _run(structurally_recovered, {"x": x})["y"]
    np.testing.assert_allclose(bare_out, original_out, rtol=1e-4, atol=1e-5)

    # recover_batch_norm's own closed-form calibration, with no target_model,
    # deliberately fits the recovered node to *whatever distribution
    # calibration_data carries* -- unrelated random data here -- so it is
    # *not* expected to reproduce `original`'s own numbers (see this
    # module's own docstring: recovering the *original* folded-away BN
    # exactly is impossible by construction). What it should still do is
    # produce a valid, structurally recovered model.
    calib = [
        {"x": rng.standard_normal((1, c_in, 8, 8)).astype(np.float32)}
        for _ in range(4)
    ]
    recovered = recover_batch_norm(fused, calibration_data=calib)
    onnx.checker.check_model(recovered)
    assert "BatchNormalization" in {n.op_type for n in recovered.graph.node}

    # Pointing recover_batch_norm at `original` as target_model, though,
    # asks it to match the recovered node's own output distribution back to
    # `original`'s -- and since the fused graph's pre-recovery activation is
    # already numerically identical to `original`'s own final output (shown
    # above), that closed-form affine match closely reproduces it, including
    # off the exact calibration inputs.
    matched = recover_batch_norm(
        fused, calibration_data=calib, target_model=original
    )
    onnx.checker.check_model(matched)
    matched_out = _run(matched, {"x": x})["y"]
    np.testing.assert_allclose(matched_out, original_out, rtol=1e-3, atol=1e-4)


def test_recover_batch_norm_is_a_no_op_with_nothing_to_recover():
    model = _model(
        """
        g (float[1,3,4,4] x) => (float[1,3,4,4] y)
        {
          y = Relu(x)
        }
        """
    )
    result = recover_batch_norm(model)
    assert result.SerializeToString() == model.SerializeToString()
