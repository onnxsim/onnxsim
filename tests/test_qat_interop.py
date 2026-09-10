"""Tests for ``onnxsim.qat_interop`` -- the graph-only half of QAT.

The claim under test is narrow and easy to get silently wrong: a QDQ model
exported from a QAT-trained network carries scales and zero-points that took a
training run to produce, and ``onnxsim.quantize_static`` would recompute them
from observed min/max without complaining. So every model built here has
learned parameters *deliberately far* from what calibration would derive --
the tests assert both that the learned values come out the other side
bit-exact and (separately) that plain calibration really would have produced
something else, which is what makes the first assertion mean anything.

The second theme is refusal: a QDQ pair this module does not fully understand
must be left alone rather than guessed at, and the caller must be able to find
out. Each refusal is exercised with a model that triggers exactly it.
"""

import json

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim import qat_interop as qi

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _u8(value, name):
    return onnx.numpy_helper.from_array(np.asarray(value, dtype=np.uint8), name)


def _i8(value, name):
    return onnx.numpy_helper.from_array(np.asarray(value, dtype=np.int8), name)


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _check_and_run(model, feeds):
    """Everything this module produces must both validate and execute."""
    onnx.checker.check_model(model)
    outputs = _run(model, feeds)
    for out in outputs:
        assert np.all(np.isfinite(out))
    return outputs


def _initializers(model):
    return {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}


def _activation_params(model, tensor_name):
    """The (scale, zero_point) the QDQ pair around ``tensor_name`` carries."""
    inits = _initializers(model)
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and node.input[0] == tensor_name:
            return inits[node.input[1]], inits[node.input[2]]
    raise AssertionError(f"{tensor_name} is not quantized in this model")


def _weight_params(model, consumer_output):
    """The (codes, scale) of the dequantized weight feeding ``consumer_output``."""
    inits = _initializers(model)
    producers = {out: node for node in model.graph.node for out in node.output}
    for node in model.graph.node:
        if node.output[0] != consumer_output:
            continue
        dq = producers[node.input[1]]
        assert dq.op_type == "DequantizeLinear"
        return inits[dq.input[0]], inits[dq.input[1]]
    raise AssertionError(f"no node produces {consumer_output}")


# The learned activation quantizer every QAT-style model below carries. It
# clips hard: over the calibration data used here, min/max calibration derives
# a scale several times larger (asserted in
# test_learned_activation_scale_is_not_what_calibration_would_derive), which is
# the realistic case -- a trained scale buys resolution on the bulk of the
# distribution by saturating the tails.
#
# The exact pair matters, and not only for realism. Ingest hands the C++
# rewrite a calibration range equivalent to the learned quantizer and then
# writes the learned tensors back over what the rewrite computed; for many
# (scale, zero-point) pairs that range round trip happens to be exact in
# float32 anyway, and a test built on one of those would pass with the
# write-back deleted. This pair is one the round trip loses (the range-derived
# scale comes back as 0.011300001), so asserting bit-exactness here actually
# tests the thing it names.
LEARNED_SCALE = 0.0113
LEARNED_ZP = 91

RNG = np.random.default_rng(0)
W1 = (RNG.standard_normal((4, 6)) * 0.5).astype(np.float32)
W2 = (RNG.standard_normal((6, 3)) * 0.5).astype(np.float32)
# Deliberately wide: the observed range is ~40x the learned quantizer's.
CALIBRATION = [{"X": (RNG.standard_normal((2, 4)) * 4.0).astype(np.float32)}]
FEEDS = {"X": (RNG.standard_normal((2, 4)) * 0.3).astype(np.float32)}


def _float_matmul():
    return _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          Y = MatMul(X, W1)
        }
        """,
        initializer=[_f32(W1, "W1")],
    )


def _qat_matmul(scale=LEARNED_SCALE, zero_point=LEARNED_ZP):
    """One MatMul whose activation arrives already fake-quantized."""
    return _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          Xq = QuantizeLinear(X, x_scale, x_zp)
          Xdq = DequantizeLinear(Xq, x_scale, x_zp)
          Y = MatMul(Xdq, W1)
        }
        """,
        initializer=[
            _f32(W1, "W1"),
            _f32(scale, "x_scale"),
            _u8(zero_point, "x_zp"),
        ],
    )


def _qat_chain():
    """Two MatMuls, only the first one's activation annotated -- the
    partially-quantized export a real QAT pipeline produces when only part of
    the network was wrapped in fake-quant modules.
    """
    return _model(
        """
        g (float[2, 4] X) => (float[2, 3] Y)
        {
          Xq = QuantizeLinear(X, x_scale, x_zp)
          Xdq = DequantizeLinear(Xq, x_scale, x_zp)
          H = MatMul(Xdq, W1)
          Y = MatMul(H, W2)
        }
        """,
        initializer=[
            _f32(W1, "W1"),
            _f32(W2, "W2"),
            _f32(LEARNED_SCALE, "x_scale"),
            _u8(LEARNED_ZP, "x_zp"),
        ],
    )


# --------------------------------------------------------------------------- #
# The point of the whole module: learned parameters survive
# --------------------------------------------------------------------------- #
def test_learned_activation_scale_is_not_what_calibration_would_derive():
    # Guards every other assertion in this file: if calibration happened to
    # land on the learned values, "the learned values survived" would be
    # vacuous.
    calibrated = onnxsim.quantize_static(_float_matmul(), calibration_data=CALIBRATION)
    scale, zero_point = _activation_params(calibrated, "X")
    assert float(scale) > 3 * LEARNED_SCALE
    assert int(zero_point) != LEARNED_ZP


def test_learned_activation_scale_survives_ingest():
    result = qi.quantize_static_keeping_qdq_scales(
        _qat_matmul(), calibration_data=CALIBRATION
    )
    assert result.preserved == ("X",)
    assert result.recalibrated == ()
    assert result.unpreserved == ()
    scale, zero_point = _activation_params(result.model, "X")
    # Bit-exact, not merely close: the value was trained, so re-deriving it
    # through the calibration-range round trip is not good enough.
    assert float(scale) == np.float32(LEARNED_SCALE)
    assert int(zero_point) == LEARNED_ZP
    _check_and_run(result.model, FEEDS)


def test_fully_annotated_model_needs_no_calibration_data():
    # No `calibration_data`, and none is generated either: every quantizable
    # tensor already carries a learned quantizer, so there is nothing to
    # observe. This is the property that makes ingest usable on a model whose
    # real input data the user does not have.
    result = qi.quantize_static_keeping_qdq_scales(_qat_matmul())
    assert result.calibration_ran is False
    assert result.preserved == ("X",)
    assert float(_activation_params(result.model, "X")[0]) == np.float32(LEARNED_SCALE)


def test_partially_annotated_model_keeps_one_and_calibrates_the_other():
    result = qi.quantize_static_keeping_qdq_scales(
        _qat_chain(), calibration_data=CALIBRATION
    )
    assert result.calibration_ran is True
    assert "X" in result.preserved
    assert result.recalibrated == ("H",)

    scale, zero_point = _activation_params(result.model, "X")
    assert float(scale) == np.float32(LEARNED_SCALE)
    assert int(zero_point) == LEARNED_ZP
    # H had no annotation, so it must have been calibrated the ordinary way --
    # i.e. it is quantized at all, and not with X's parameters.
    h_scale, _ = _activation_params(result.model, "H")
    assert float(h_scale) > 0
    assert float(h_scale) != float(scale)
    _check_and_run(result.model, FEEDS)


def test_learned_weight_scale_and_codes_survive_ingest():
    # A per-output-channel weight quantizer trained 20% tighter than
    # round-to-nearest would pick -- the LSQ-style learned weight scale.
    learned = (np.abs(W1).max(axis=0) / 127.0 * 0.8).astype(np.float32)
    model = _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          Xq = QuantizeLinear(X, x_scale, x_zp)
          Xdq = DequantizeLinear(Xq, x_scale, x_zp)
          Wq = QuantizeLinear<axis = 1>(W1, w_scale, w_zp)
          Wdq = DequantizeLinear<axis = 1>(Wq, w_scale, w_zp)
          Y = MatMul(Xdq, Wdq)
        }
        """,
        initializer=[
            _f32(W1, "W1"),
            _f32(LEARNED_SCALE, "x_scale"),
            _u8(LEARNED_ZP, "x_zp"),
            _f32(learned, "w_scale"),
            _i8(np.zeros(6), "w_zp"),
        ],
    )
    result = qi.quantize_static_keeping_qdq_scales(model)
    assert result.calibration_ran is False
    assert set(result.preserved) == {"X", "W1"}

    codes, scale = _weight_params(result.model, "Y")
    assert np.array_equal(scale, learned)
    # The codes are re-derived from the learned scale rather than kept from
    # the rewrite's own round-to-nearest scale -- swapping only the scale would
    # change what the weight dequantizes to.
    assert np.array_equal(codes, np.clip(np.round(W1 / learned), -127, 127))
    _check_and_run(result.model, FEEDS)


def test_one_activation_feeding_two_layers_gets_both_pairs():
    # The rewrite is per-node, so a shared activation comes back bracketed by
    # two independent QDQ pairs with two independent scale initializers.
    # Writing the learned value into only one of them would leave the model
    # quantizing a single tensor two different ways.
    other = (RNG.standard_normal((4, 5)) * 0.5).astype(np.float32)
    model = _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y, float[2, 5] Z)
        {
          Xq = QuantizeLinear(X, x_scale, x_zp)
          Xdq = DequantizeLinear(Xq, x_scale, x_zp)
          Y = MatMul(Xdq, W1)
          Z = MatMul(Xdq, W3)
        }
        """,
        initializer=[
            _f32(W1, "W1"),
            _f32(other, "W3"),
            _f32(LEARNED_SCALE, "x_scale"),
            _u8(LEARNED_ZP, "x_zp"),
        ],
    )
    result = qi.quantize_static_keeping_qdq_scales(model)
    assert result.preserved == ("X",)
    inits = _initializers(result.model)
    quantizers = [n for n in result.model.graph.node if n.op_type == "QuantizeLinear"]
    assert len(quantizers) == 2
    for node in quantizers:
        assert float(inits[node.input[1]]) == np.float32(LEARNED_SCALE)
        assert int(inits[node.input[2]]) == LEARNED_ZP
    _check_and_run(result.model, FEEDS)


def test_learned_parameters_survive_on_a_conv():
    # quantize_static covers Conv as well as MatMul/Gemm, and Conv's weight
    # puts its output channel on axis 0 rather than 1 -- so the axis agreement
    # between the learned quantizer and the emitted one is a different case,
    # not the same one with different numbers.
    rng = np.random.default_rng(7)
    weight = (rng.standard_normal((3, 2, 3, 3)) * 0.4).astype(np.float32)
    learned = (np.abs(weight).max(axis=(1, 2, 3)) / 127.0 * 0.75).astype(np.float32)
    model = _model(
        """
        g (float[1, 2, 5, 5] X) => (float[1, 3, 3, 3] Y)
        {
          Xq = QuantizeLinear(X, x_scale, x_zp)
          Xdq = DequantizeLinear(Xq, x_scale, x_zp)
          Wq = QuantizeLinear<axis = 0>(K, w_scale, w_zp)
          Wdq = DequantizeLinear<axis = 0>(Wq, w_scale, w_zp)
          Y = Conv(Xdq, Wdq)
        }
        """,
        initializer=[
            _f32(weight, "K"),
            _f32(LEARNED_SCALE, "x_scale"),
            _u8(LEARNED_ZP, "x_zp"),
            _f32(learned, "w_scale"),
            _i8(np.zeros(3), "w_zp"),
        ],
    )
    result = qi.quantize_static_keeping_qdq_scales(model)
    assert result.calibration_ran is False
    assert set(result.preserved) == {"X", "K"}
    scale, zero_point = _activation_params(result.model, "X")
    assert float(scale) == np.float32(LEARNED_SCALE)
    assert int(zero_point) == LEARNED_ZP
    codes, weight_scale = _weight_params(result.model, "Y")
    assert np.array_equal(weight_scale, learned)
    assert np.array_equal(
        codes,
        np.clip(np.round(weight / learned.reshape(3, 1, 1, 1)), -127, 127),
    )
    _check_and_run(
        result.model, {"X": rng.standard_normal((1, 2, 5, 5)).astype(np.float32)}
    )


def test_ingest_stays_close_to_the_model_it_ingested():
    # End-to-end sanity: preserving the learned quantizers should reproduce the
    # QAT export's own numerics, not merely produce *a* quantized model.
    model = _qat_matmul()
    (before,) = _run(model, FEEDS)
    result = qi.quantize_static_keeping_qdq_scales(model)
    (after,) = _check_and_run(result.model, FEEDS)
    assert np.allclose(before, after, atol=2e-2)


# --------------------------------------------------------------------------- #
# Round trip: emit, "train", re-import
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme", ["int8", "int16"])
def test_round_trip_through_an_external_trainer(scheme):
    export = qi.export_fake_quant(
        _float_matmul(), scheme=scheme, calibration_data=CALIBRATION
    )
    assert export.learnable_scales == ("X_scale", "W1_scale")
    assert export.learnable_zero_points == ("X_zero_point",)
    _check_and_run(export.model, FEEDS)

    # What a trainer does to those tensors: move them. 0.6x is well outside
    # anything calibration would land on, so a re-derived scale is
    # distinguishable from a preserved one.
    trained = onnx.ModelProto()
    trained.CopyFrom(export.model)
    wanted = {}
    for init in trained.graph.initializer:
        if init.name not in export.learnable_scales:
            continue
        moved = (onnx.numpy_helper.to_array(init) * np.float32(0.6)).astype(np.float32)
        init.CopyFrom(onnx.numpy_helper.from_array(moved, init.name))
        wanted[init.name] = moved

    result = qi.quantize_static_keeping_qdq_scales(
        trained, calibration_data=CALIBRATION, scheme=scheme
    )
    assert result.calibration_ran is False
    assert result.unpreserved == ()
    assert result.scan.skipped == ()

    scale, _ = _activation_params(result.model, "X")
    assert np.array_equal(scale.reshape(-1), wanted["X_scale"].reshape(-1))
    _, weight_scale = _weight_params(result.model, "Y")
    assert np.array_equal(weight_scale, wanted["W1_scale"])
    _check_and_run(result.model, FEEDS)


def test_export_records_learnable_tensors_in_metadata_too():
    export = qi.export_fake_quant(_float_matmul(), calibration_data=CALIBRATION)
    props = {p.key: p.value for p in export.model.metadata_props}
    assert json.loads(props[qi.LEARNABLE_SCALES_KEY]) == list(export.learnable_scales)
    assert json.loads(props[qi.LEARNABLE_ZERO_POINTS_KEY]) == list(
        export.learnable_zero_points
    )
    # Scales are float and directly trainable; zero-points are integer tensors
    # a trainer has to round back into, so they are listed apart.
    inits = _initializers(export.model)
    for name in export.learnable_scales:
        assert inits[name].dtype == np.float32
    for name in export.learnable_zero_points:
        assert inits[name].dtype in (np.uint8, np.uint16)
    assert export.learnable_tensors == (
        export.learnable_scales + export.learnable_zero_points
    )


def test_export_keeps_generated_names_when_asked():
    export = qi.export_fake_quant(
        _float_matmul(), calibration_data=CALIBRATION, rename_parameters=False
    )
    inits = _initializers(export.model)
    assert all(name in inits for name in export.learnable_tensors)
    assert "X_scale" not in inits


@pytest.mark.parametrize("bad", ["int4", "fp8", ""])
def test_unknown_scheme_is_refused(bad):
    with pytest.raises(ValueError, match="unknown scheme"):
        qi.export_fake_quant(_float_matmul(), scheme=bad)
    with pytest.raises(ValueError, match="unknown scheme"):
        qi.quantize_static_keeping_qdq_scales(_qat_matmul(), scheme=bad)


# --------------------------------------------------------------------------- #
# Canonicalization
# --------------------------------------------------------------------------- #
def test_strip_restores_the_float_graph():
    stripped, scan = qi.strip_existing_qdq(_qat_matmul())
    assert [a.tensor_name for a in scan.annotations] == ["X"]
    assert [n.op_type for n in stripped.graph.node] == ["MatMul"]
    # The pair's own parameters go with it; the float weight stays.
    names = {i.name for i in stripped.graph.initializer}
    assert names == {"W1"}
    _check_and_run(stripped, FEEDS)


def test_strip_rematerializes_an_integer_weight():
    # Scan shape 3: a weight stored in its quantized form with no Quantize in
    # front of it, which is what quantize_static and ORT's quantizer emit.
    codes = np.clip(np.round(W1 / 0.01), -127, 127).astype(np.int8)
    model = _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          Wdq = DequantizeLinear(Wq, w_scale)
          Y = MatMul(X, Wdq)
        }
        """,
        initializer=[
            onnx.numpy_helper.from_array(codes, "Wq"),
            _f32(0.01, "w_scale"),
        ],
    )
    stripped, scan = qi.strip_existing_qdq(model)
    assert [(a.tensor_name, a.role) for a in scan.annotations] == [("Wdq", "weight")]
    assert [n.op_type for n in stripped.graph.node] == ["MatMul"]
    assert np.allclose(_initializers(stripped)["Wdq"], codes.astype(np.float32) * 0.01)


def test_strip_leaves_untouched_what_it_refuses():
    model = _blocked_activation()
    stripped, scan = qi.strip_existing_qdq(model)
    assert scan.annotations == ()
    assert [n.op_type for n in stripped.graph.node] == [
        n.op_type for n in model.graph.node
    ]


# --------------------------------------------------------------------------- #
# Refusals: each one reported, each one left alone
# --------------------------------------------------------------------------- #
def _non_constant_scale():
    # The scale reaches QuantizeLinear through an Identity, so it is a node
    # output rather than an initializer: computable, but not readable off the
    # graph without folding, which this module deliberately will not do.
    return _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          s = Identity(x_scale)
          Xq = QuantizeLinear(X, s, x_zp)
          Xdq = DequantizeLinear(Xq, s, x_zp)
          Y = MatMul(Xdq, W1)
        }
        """,
        initializer=[_f32(W1, "W1"), _f32(LEARNED_SCALE, "x_scale"), _u8(3, "x_zp")],
    )


def _unexpected_zero_point_dtype():
    # int4 is a legal QuantizeLinear zero-point type in opset 21 and a
    # dtype onnxsim's static scheme has no counterpart for. Built with
    # onnx.helper rather than numpy_helper because a 4-bit tensor has no
    # plain numpy dtype to build it from.
    return _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          Xq = QuantizeLinear(X, x_scale, x_zp)
          Xdq = DequantizeLinear(Xq, x_scale, x_zp)
          Y = MatMul(Xdq, W1)
        }
        """,
        initializer=[
            _f32(W1, "W1"),
            _f32(LEARNED_SCALE, "x_scale"),
            onnx.helper.make_tensor("x_zp", onnx.TensorProto.INT4, [], [3]),
        ],
    )


def _axis_shape_mismatch():
    # A per-channel weight quantizer claiming axis 1 of a [4, 6] weight, with
    # only 5 scales. One of the two is wrong and there is no way to tell which.
    return _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          Wq = QuantizeLinear<axis = 1>(W1, w_scale, w_zp)
          Wdq = DequantizeLinear<axis = 1>(Wq, w_scale, w_zp)
          Y = MatMul(X, Wdq)
        }
        """,
        initializer=[
            _f32(W1, "W1"),
            _f32(np.full(5, 0.01), "w_scale"),
            _i8(np.zeros(5), "w_zp"),
        ],
    )


def _blocked_activation():
    return _model(
        """
        g (float[4, 6] X) => (float[4, 3] Y)
        {
          Xq = QuantizeLinear<axis = 1, block_size = 3>(X, b_scale, b_zp)
          Xdq = DequantizeLinear<axis = 1, block_size = 3>(Xq, b_scale, b_zp)
          Y = MatMul(Xdq, W2)
        }
        """,
        initializer=[
            _f32(W2, "W2"),
            _f32(np.full((4, 2), 0.01), "b_scale"),
            _u8(np.zeros((4, 2)), "b_zp"),
        ],
    )


def _reused_quantized_tensor():
    # Xq escapes to a second consumer, so deleting the pair would delete a
    # tensor something else still reads.
    return _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y, uint8[2, 4] Z)
        {
          Xq = QuantizeLinear(X, x_scale, x_zp)
          Xdq = DequantizeLinear(Xq, x_scale, x_zp)
          Z = Identity(Xq)
          Y = MatMul(Xdq, W1)
        }
        """,
        initializer=[_f32(W1, "W1"), _f32(LEARNED_SCALE, "x_scale"), _u8(3, "x_zp")],
    )


def _dequantized_graph_output():
    return _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y, float[2, 4] Xdq)
        {
          Xq = QuantizeLinear(X, x_scale, x_zp)
          Xdq = DequantizeLinear(Xq, x_scale, x_zp)
          Y = MatMul(Xdq, W1)
        }
        """,
        initializer=[_f32(W1, "W1"), _f32(LEARNED_SCALE, "x_scale"), _u8(3, "x_zp")],
    )


def _mismatched_pair():
    # Quantize and Dequantize disagree about the quantizer: the pair is not a
    # round trip through one set of parameters, so there is no single
    # (scale, zero-point) to preserve.
    return _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          Xq = QuantizeLinear(X, x_scale, x_zp)
          Xdq = DequantizeLinear(Xq, other_scale, x_zp)
          Y = MatMul(Xdq, W1)
        }
        """,
        initializer=[
            _f32(W1, "W1"),
            _f32(LEARNED_SCALE, "x_scale"),
            _f32(LEARNED_SCALE * 2, "other_scale"),
            _u8(3, "x_zp"),
        ],
    )


def _non_positive_scale():
    return _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          Xq = QuantizeLinear(X, x_scale, x_zp)
          Xdq = DequantizeLinear(Xq, x_scale, x_zp)
          Y = MatMul(Xdq, W1)
        }
        """,
        initializer=[_f32(W1, "W1"), _f32(0.0, "x_scale"), _u8(3, "x_zp")],
    )


REFUSALS = {
    "scale_not_constant": _non_constant_scale,
    "unsupported_zero_point_dtype": _unexpected_zero_point_dtype,
    "axis_shape_mismatch": _axis_shape_mismatch,
    "blocked_quantization_unsupported": _blocked_activation,
    "quantized_tensor_reused": _reused_quantized_tensor,
    "dequantize_output_is_graph_output": _dequantized_graph_output,
    "mismatched_quantization_params": _mismatched_pair,
    "non_positive_scale": _non_positive_scale,
}


@pytest.mark.parametrize("reason", sorted(REFUSALS))
def test_refused_pattern_is_reported_and_left_in_the_graph(reason):
    model = REFUSALS[reason]()
    onnx.checker.check_model(model)
    scan = qi.find_existing_qdq(model)
    assert scan.annotations == ()
    assert [s.reason for s in scan.skipped] == [reason]
    # Every reported reason is a documented one, and its string form says so.
    assert reason in qi.SKIP_REASONS
    assert qi.SKIP_REASONS[reason] in str(scan.skipped[0])

    # Refusing means leaving the model's own quantization exactly where it is,
    # not dropping it: the pair is still there afterwards.
    result = qi.quantize_static_keeping_qdq_scales(model, num_calibration_samples=2)
    op_types = [n.op_type for n in result.model.graph.node]
    assert op_types.count("QuantizeLinear") >= 1
    assert result.scan.skipped == scan.skipped


def test_per_axis_activation_is_understood_but_cannot_be_carried():
    # Detection succeeds -- the pair is well formed and its axis checks out --
    # but quantize_static's activation scheme is per-tensor, so the tensor is
    # recalibrated and the caller is told exactly that.
    model = _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          Xq = QuantizeLinear<axis = 1>(X, x_scale, x_zp)
          Xdq = DequantizeLinear<axis = 1>(Xq, x_scale, x_zp)
          Y = MatMul(Xdq, W1)
        }
        """,
        initializer=[
            _f32(W1, "W1"),
            _f32(np.full(4, LEARNED_SCALE), "x_scale"),
            _u8(np.full(4, LEARNED_ZP), "x_zp"),
        ],
    )
    scan = qi.find_existing_qdq(model)
    assert [(a.tensor_name, a.axis) for a in scan.annotations] == [("X", 1)]

    result = qi.quantize_static_keeping_qdq_scales(model, calibration_data=CALIBRATION)
    assert result.preserved == ()
    assert result.recalibrated == ("X",)
    assert [(s.tensor_name, s.reason) for s in result.unpreserved] == [
        ("X", "per_axis_activation_unsupported")
    ]
    _check_and_run(result.model, FEEDS)


@pytest.mark.parametrize(
    "zero_point,reason",
    [
        (("int8", 5), "weight_zero_point_not_symmetric"),
        (("uint8", 0), "weight_dtype_not_int8"),
    ],
)
def test_asymmetric_weight_quantizer_is_reported_not_reinterpreted(zero_point, reason):
    # onnxsim's weight scheme is symmetric int8. A learned weight quantizer
    # that is not is understood but cannot be re-emitted, so the weight falls
    # back to onnxsim's own round-to-nearest scale -- and says so.
    dtype, value = zero_point
    zp = np.full(6, value, dtype=np.dtype(dtype))
    model = _model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          Wq = QuantizeLinear<axis = 1>(W1, w_scale, w_zp)
          Wdq = DequantizeLinear<axis = 1>(Wq, w_scale, w_zp)
          Y = MatMul(X, Wdq)
        }
        """,
        initializer=[
            _f32(W1, "W1"),
            _f32(np.full(6, 0.01), "w_scale"),
            onnx.numpy_helper.from_array(zp, "w_zp"),
        ],
    )
    onnx.checker.check_model(model)
    result = qi.quantize_static_keeping_qdq_scales(model, calibration_data=CALIBRATION)
    assert result.preserved == ()
    assert [(s.tensor_name, s.reason) for s in result.unpreserved] == [("W1", reason)]
    _, scale = _weight_params(result.model, "Y")
    assert not np.allclose(scale, 0.01)
    _check_and_run(result.model, FEEDS)


def test_qdq_on_a_tensor_onnxsim_does_not_quantize_is_left_alone():
    # A fake-quantized Add output: a real QAT export annotates far more
    # tensors than onnxsim's static quantization has a place for. The pair
    # stays in the graph untouched, and the reason says why.
    model = _model(
        """
        g (float[2, 4] X) => (float[2, 4] Y)
        {
          H = Add(X, X)
          Hq = QuantizeLinear(H, x_scale, x_zp)
          Hdq = DequantizeLinear(Hq, x_scale, x_zp)
          Y = Relu(Hdq)
        }
        """,
        initializer=[_f32(LEARNED_SCALE, "x_scale"), _u8(LEARNED_ZP, "x_zp")],
    )
    scan = qi.find_existing_qdq(model)
    assert [a.tensor_name for a in scan.annotations] == ["H"]

    result = qi.quantize_static_keeping_qdq_scales(model, calibration_data=CALIBRATION)
    assert result.preserved == ()
    assert [(s.tensor_name, s.reason) for s in result.unpreserved] == [
        ("H", "not_a_quantizable_tensor")
    ]
    assert [n.op_type for n in result.model.graph.node] == [
        n.op_type for n in model.graph.node
    ]
    scale, zero_point = _activation_params(result.model, "H")
    assert float(scale) == np.float32(LEARNED_SCALE)
    assert int(zero_point) == LEARNED_ZP
    _check_and_run(result.model, FEEDS)


def test_every_reported_reason_is_documented():
    # A reason with no entry in SKIP_REASONS would render as a KeyError in
    # SkippedQdq.__str__ the first time a user printed it.
    models = [factory() for factory in REFUSALS.values()] + [
        _qat_matmul(),
        _qat_chain(),
        _axis_shape_mismatch(),
    ]
    seen = set()
    for model in models:
        result = qi.quantize_static_keeping_qdq_scales(model, num_calibration_samples=2)
        seen.update(s.reason for s in result.scan.skipped)
        seen.update(s.reason for s in result.unpreserved)
    assert seen
    assert seen <= set(qi.SKIP_REASONS)


# --------------------------------------------------------------------------- #
# QONNX/Brevitas ingest: a ``Quant`` node is a different graph shape than a
# QuantizeLinear/DequantizeLinear pair (see qat_interop.py's own "QONNX/
# Brevitas ingest" docstring section), so it gets its own small model helper --
# ``_model`` above hardcodes ``opset_import: ["": opset]`` with no room for the
# extra "qonnx.custom_op.general" domain a Quant node needs declared.
# --------------------------------------------------------------------------- #
def _qonnx_model(body, initializer=(), opset=21, ir_version=10):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}, "qonnx.custom_op.general": 1]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _qonnx_activation_quant(
    signed=0, zero_point=LEARNED_ZP, scale=LEARNED_SCALE, bitwidth=8.0
):
    """One MatMul whose activation arrives via a single QONNX ``Quant`` node
    instead of a QuantizeLinear/DequantizeLinear pair -- what
    ``brevitas.export.export_qonnx`` emits for a quantized activation.
    """
    return _qonnx_model(
        f"""
        g (float[2, 4] X) => (float[2, 6] Y)
        {{
          Xdq = qonnx.custom_op.general.Quant<signed = {signed}, narrow = 0>(X, x_scale, x_zp, x_bw)
          Y = MatMul(Xdq, W1)
        }}
        """,
        initializer=[
            _f32(W1, "W1"),
            _f32(scale, "x_scale"),
            _f32(float(zero_point), "x_zp"),
            _f32(bitwidth, "x_bw"),
        ],
    )


def _qonnx_weight_quant(scale=0.01, zero_point=0.0, bitwidth=8.0):
    """One MatMul whose weight arrives pre-wrapped in a QONNX ``Quant`` node --
    what ``export_qonnx`` emits for a quantized (signed, symmetric) weight.
    """
    return _qonnx_model(
        f"""
        g (float[2, 4] X) => (float[2, 6] Y)
        {{
          Wdq = qonnx.custom_op.general.Quant<signed = 1, narrow = 0>(W1, w_scale, w_zp, w_bw)
          Y = MatMul(X, Wdq)
        }}
        """,
        initializer=[
            _f32(W1, "W1"),
            _f32(scale, "w_scale"),
            _f32(zero_point, "w_zp"),
            _f32(bitwidth, "w_bw"),
        ],
    )


def test_qonnx_quant_is_detected_as_a_fourth_pattern_shape():
    model = _qonnx_activation_quant()
    scan = qi.find_existing_qdq(model)
    assert [a.tensor_name for a in scan.annotations] == ["X"]
    ann = scan.annotations[0]
    assert ann.is_qonnx_quant is True
    assert ann.role == "activation"
    assert float(ann.scale) == np.float32(LEARNED_SCALE)
    assert int(ann.zero_point) == LEARNED_ZP
    assert ann.zero_point_dtype == onnx.TensorProto.UINT8


def test_qonnx_quant_activation_survives_ingest():
    result = qi.quantize_static_keeping_qdq_scales(
        _qonnx_activation_quant(), calibration_data=CALIBRATION
    )
    assert result.preserved == ("X",)
    assert result.recalibrated == ()
    assert result.unpreserved == ()
    scale, zero_point = _activation_params(result.model, "X")
    assert float(scale) == np.float32(LEARNED_SCALE)
    assert int(zero_point) == LEARNED_ZP
    # The ingested model is an ordinary QDQ model: the Quant node is gone.
    assert "Quant" not in [n.op_type for n in result.model.graph.node]
    _check_and_run(result.model, FEEDS)


def test_qonnx_quant_weight_survives_ingest():
    result = qi.quantize_static_keeping_qdq_scales(
        _qonnx_weight_quant(), calibration_data=CALIBRATION
    )
    assert "W1" in result.preserved
    codes, scale = _weight_params(result.model, "Y")
    assert np.allclose(scale, 0.01)
    assert np.array_equal(codes, np.clip(np.round(W1 / 0.01), -127, 127))
    _check_and_run(result.model, FEEDS)


def test_strip_existing_qdq_canonicalizes_a_quant_node():
    float_model, scan = qi.strip_existing_qdq(_qonnx_weight_quant())
    assert [a.tensor_name for a in scan.annotations] == ["W1"]
    assert [n.op_type for n in float_model.graph.node] == ["MatMul"]
    # W1 was rewired straight to the original float initializer, not
    # rematerialized -- unlike an integer-stored QDQ weight (scan shape 3),
    # a Quant node's own first input already is the float tensor.
    assert np.array_equal(
        onnx.numpy_helper.to_array(
            next(i for i in float_model.graph.initializer if i.name == "W1")
        ),
        W1,
    )


def test_qonnx_bitwidth_other_than_8_is_reported():
    model = _qonnx_activation_quant(bitwidth=4.0)
    scan = qi.find_existing_qdq(model)
    assert scan.annotations == ()
    assert [s.reason for s in scan.skipped] == ["qonnx_bitwidth_unsupported"]


def test_qonnx_per_channel_scale_is_reported():
    model = _qonnx_weight_quant()
    for init in model.graph.initializer:
        if init.name == "w_scale":
            init.CopyFrom(_f32(np.full(6, 0.01), "w_scale"))
    scan = qi.find_existing_qdq(model)
    assert scan.annotations == ()
    assert [s.reason for s in scan.skipped] == ["qonnx_per_channel_scale_unsupported"]


def test_qonnx_non_integral_zero_point_is_reported():
    model = _qonnx_activation_quant(zero_point=0)
    for init in model.graph.initializer:
        if init.name == "x_zp":
            init.CopyFrom(_f32(0.5, "x_zp"))
    scan = qi.find_existing_qdq(model)
    assert scan.annotations == ()
    assert [s.reason for s in scan.skipped] == ["qonnx_zero_point_not_integral"]


def test_qonnx_bipolar_quant_is_recognized_but_unsupported():
    model = _qonnx_model(
        """
        g (float[2, 4] X) => (float[2, 6] Y)
        {
          Xdq = qonnx.custom_op.general.BipolarQuant(X, x_scale)
          Y = MatMul(Xdq, W1)
        }
        """,
        initializer=[_f32(W1, "W1"), _f32(LEARNED_SCALE, "x_scale")],
    )
    scan = qi.find_existing_qdq(model)
    assert scan.annotations == ()
    assert [s.reason for s in scan.skipped] == ["qonnx_op_unsupported"]

    result = qi.quantize_static_keeping_qdq_scales(model, num_calibration_samples=2)
    # Refusing means leaving the node exactly where it is.
    assert "BipolarQuant" in [n.op_type for n in result.model.graph.node]
