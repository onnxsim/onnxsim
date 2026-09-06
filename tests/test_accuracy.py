"""Tests for ``onnxsim.accuracy`` -- the unified ``QuantizationConfig``/
``quantize()`` dispatcher and the empirical ``measure_accuracy_drop`` tool.

Models are built via ``onnx.parser.parse_model``. ``measure_accuracy_drop``
and the ``quantize()``-dispatched calibration-based schemes execute the
model (through ``onnxsim.backend``, onnxruntime when installed), so this
mirrors ``test_dynamic_quantize_matmul_integer_to_float.py``'s
``pytest.importorskip("onnxruntime")`` guard -- a bare ``import
onnxruntime`` would fail *collection* (not skip the test) on a platform
onnxruntime doesn't ship wheels for.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.accuracy import (
    AccuracyDropReport,
    OutputAccuracyStats,
    _initializer_nbytes,
)

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


def _linear_model(K=64, N=32, opset=21, seed=0):
    rng = np.random.default_rng(seed)
    w = onnx.numpy_helper.from_array(
        (rng.standard_normal((K, N)) * 0.3).astype(np.float32), "W"
    )
    b = onnx.numpy_helper.from_array(
        (rng.standard_normal(N) * 0.1).astype(np.float32), "B"
    )
    return _model(
        f"""
        g (float[4,{K}] X) => (float[4,{N}] Y)
        {{
          mm = MatMul(X, W)
          Y = Add(mm, B)
        }}
        """,
        initializer=[w, b],
        opset=opset,
    )


# --------------------------------------------------------------------------- #
# QuantizationConfig / quantize() dispatcher
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "config,expected_new_op",
    [
        (onnxsim.QuantizationConfig(scheme="dynamic"), "DynamicQuantizeLinear"),
        (
            onnxsim.QuantizationConfig(scheme="dynamic_fused"),
            "MatMulIntegerToFloat",
        ),
        (
            onnxsim.QuantizationConfig(scheme="weight_only", dtype="int8"),
            "DequantizeLinear",
        ),
        (
            onnxsim.QuantizationConfig(
                scheme="weight_only", dtype="int8", granularity="per_block"
            ),
            "DequantizeLinear",
        ),
        (
            onnxsim.QuantizationConfig(scheme="weight_only", dtype="int16"),
            "DequantizeLinear",
        ),
        (
            onnxsim.QuantizationConfig(scheme="weight_only", dtype="int4"),
            "DequantizeLinear",
        ),
        (onnxsim.QuantizationConfig(scheme="static"), "QuantizeLinear"),
        (
            onnxsim.QuantizationConfig(scheme="static_int16", dtype="int16"),
            "QuantizeLinear",
        ),
        (onnxsim.QuantizationConfig(scheme="qoperator"), "QLinearMatMul"),
        (onnxsim.QuantizationConfig(scheme="float", dtype="float16"), "Cast"),
        (onnxsim.QuantizationConfig(scheme="float", dtype="bfloat16"), "Cast"),
        (onnxsim.QuantizationConfig(scheme="float", dtype="float8_e4m3"), "Cast"),
        (onnxsim.QuantizationConfig(scheme="float", dtype="float8_e5m2"), "Cast"),
    ],
)
def test_quantize_dispatches_every_scheme(config, expected_new_op):
    # weight_only/int4 needs a reduction depth divisible by 32; every other
    # scheme is happy with the same K=64.
    model = _linear_model(K=64, N=32)
    quantized = onnxsim.quantize(model, config)
    onnx.checker.check_model(quantized)
    ops = {n.op_type for n in quantized.graph.node}
    assert expected_new_op in ops


def test_quantize_ternary_scheme_dispatches_and_is_a_noop_on_non_ternary_weight():
    model = _linear_model(K=64, N=32)
    quantized = onnxsim.quantize(model, onnxsim.QuantizationConfig(scheme="ternary"))
    onnx.checker.check_model(quantized)
    # The weight isn't structurally ternary, so the pass declines -- still a
    # valid dispatch, just a no-op rewrite.
    assert {n.op_type for n in quantized.graph.node} == {"MatMul", "Add"}


def test_quantize_unknown_scheme_raises_value_error():
    model = _linear_model()
    with pytest.raises(ValueError, match="unknown QuantizationConfig.scheme"):
        onnxsim.quantize(model, onnxsim.QuantizationConfig(scheme="not-a-scheme"))


def test_quantize_invalid_dtype_for_scheme_raises_value_error():
    model = _linear_model()
    with pytest.raises(ValueError, match="does not support dtype"):
        onnxsim.quantize(
            model, onnxsim.QuantizationConfig(scheme="dynamic", dtype="int16")
        )


def test_quantize_invalid_granularity_raises_value_error():
    model = _linear_model()
    with pytest.raises(ValueError, match="granularity"):
        onnxsim.quantize(
            model,
            onnxsim.QuantizationConfig(
                scheme="weight_only", dtype="int8", granularity="bogus"
            ),
        )


def _int4_matmul_model(K=64, N=16, seed=0):
    # A bare MatMul (no bias Add) matching the model shape
    # tests/test_awq.py, tests/test_gptq.py, and tests/test_gptaq.py
    # exercise apply_awq/apply_gptq/apply_gptaq against directly -- the
    # candidate-matching those passes do (by node output name, present in
    # both the float and quantized model) is exact regardless, but reusing
    # the same shape keeps the calibration helpers below meaningful.
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [onnx.numpy_helper.from_array(weight, "W")],
    )


def _salient_channel_calibration(K=64, num_samples=64, salient_channels=(3, 7), seed=1):
    # Matches tests/test_awq.py's own calibration helper: a handful of
    # channels with much larger activation magnitude than the rest reliably
    # gives AWQ's per-channel rescale something to improve on.
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, K)).astype(np.float32)
    for c in salient_channels:
        x[:, c] *= 20.0
    return [{"X": x}]


def _correlated_calibration(K=64, num_samples=64, rank=6, seed=1):
    # Matches tests/test_gptq.py's own calibration helper: channels that
    # are linear combinations of a handful of latent factors, so GPTQ's
    # off-diagonal Hessian terms have real correlation to compensate for.
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, K)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, K)).astype(np.float32) * 0.05
    return [{"X": x.astype(np.float32)}]


# --------------------------------------------------------------------------- #
# QuantizationConfig awq/gptq/gptaq/double_quant pipeline flags
# --------------------------------------------------------------------------- #
def test_quantize_weight_only_int4_default_flags_match_direct_call():
    model = _int4_matmul_model()
    config = onnxsim.QuantizationConfig(scheme="weight_only", dtype="int4")

    dispatched = onnxsim.quantize(model, config)
    direct = onnxsim.quantize_weight_only_int4(model)

    assert dispatched.SerializeToString() == direct.SerializeToString()


def test_quantize_weight_only_int4_awq_flag_matches_direct_apply_awq():
    model = _int4_matmul_model(K=64, N=16, seed=0)
    calibration_data = _salient_channel_calibration(K=64, num_samples=64, seed=1)
    config = onnxsim.QuantizationConfig(
        scheme="weight_only",
        dtype="int4",
        awq=True,
        calibration_data=calibration_data,
    )

    dispatched = onnxsim.quantize(model, config)

    rtn = onnxsim.quantize_weight_only_int4(model)
    direct = onnxsim.apply_awq(model, rtn, calibration_data=calibration_data)

    assert dispatched.SerializeToString() == direct.SerializeToString()
    # AWQ must have actually run (inserted its compensating Mul, changing
    # the weight/scale initializers too) -- not just re-produced plain RTN.
    assert dispatched.SerializeToString() != rtn.SerializeToString()


def test_quantize_weight_only_int4_gptq_flag_matches_direct_apply_gptq():
    model = _int4_matmul_model(K=64, N=16, seed=0)
    calibration_data = _correlated_calibration(K=64, num_samples=64, seed=1)
    config = onnxsim.QuantizationConfig(
        scheme="weight_only",
        dtype="int4",
        gptq=True,
        calibration_data=calibration_data,
    )

    dispatched = onnxsim.quantize(model, config)

    rtn = onnxsim.quantize_weight_only_int4(model)
    direct = onnxsim.apply_gptq(model, rtn, calibration_data=calibration_data)

    assert dispatched.SerializeToString() == direct.SerializeToString()
    assert dispatched.SerializeToString() != rtn.SerializeToString()


def test_quantize_weight_only_int4_chains_awq_gptq_double_quant_in_fixed_order():
    model = _int4_matmul_model(K=64, N=16, seed=0)
    calibration_data = _salient_channel_calibration(K=64, num_samples=64, seed=1)
    config = onnxsim.QuantizationConfig(
        scheme="weight_only",
        dtype="int4",
        awq=True,
        gptq=True,
        double_quant=True,
        calibration_data=calibration_data,
    )

    dispatched = onnxsim.quantize(model, config)

    manual = onnxsim.quantize_weight_only_int4(model)
    manual = onnxsim.apply_awq(model, manual, calibration_data=calibration_data)
    manual = onnxsim.apply_gptq(model, manual, calibration_data=calibration_data)
    manual = onnxsim.apply_double_quantization(manual)

    assert dispatched.SerializeToString() == manual.SerializeToString()


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"scheme": "dynamic"},
        {"scheme": "static"},
        {"scheme": "float", "dtype": "float16"},
    ],
)
def test_int4_pipeline_flags_silently_ignored_for_other_schemes(config_kwargs):
    model = _linear_model(K=64, N=32)
    plain = onnxsim.quantize(model, onnxsim.QuantizationConfig(**config_kwargs))
    flagged = onnxsim.quantize(
        model,
        onnxsim.QuantizationConfig(
            **config_kwargs,
            awq=True,
            gptq=True,
            gptaq=True,
            double_quant=True,
        ),
    )
    assert plain.SerializeToString() == flagged.SerializeToString()


def test_quantize_static_passes_through_calibration_settings():
    model = _linear_model(K=8, N=4)
    rng = np.random.default_rng(5)
    calibration_data = [{"X": rng.standard_normal((4, 8)).astype(np.float32)}]
    quantized = onnxsim.quantize(
        model,
        onnxsim.QuantizationConfig(
            scheme="static",
            calibration_data=calibration_data,
            calibration_method="minmax",
        ),
    )
    onnx.checker.check_model(quantized)
    assert "QuantizeLinear" in {n.op_type for n in quantized.graph.node}


# --------------------------------------------------------------------------- #
# measure_accuracy_drop
# --------------------------------------------------------------------------- #
def test_measure_accuracy_drop_reports_small_but_nonzero_error_for_int8():
    model = _linear_model(K=64, N=32)
    quantized = onnxsim.quantize_dynamic(model)

    report = onnxsim.measure_accuracy_drop(model, quantized, num_samples=16, seed=1)
    assert isinstance(report, AccuracyDropReport)
    assert report.num_samples == 16
    assert set(report.per_output) == {"Y"}
    stats = report.per_output["Y"]
    assert isinstance(stats, OutputAccuracyStats)
    # INT8 quantization is lossy but should stay well within a coarse bound
    # for a small, well-conditioned random matrix.
    assert 0.0 < report.worst_relative_l2 < 0.2
    assert report.worst_cosine_distance < 0.1
    assert report.all_finite


def test_measure_accuracy_drop_identity_quantization_is_exact():
    # "Quantizing" with fp16 keep_io_types=True and then immediately casting
    # back is not exact, but comparing a model against *itself* must report
    # exactly zero error -- a basic sanity check on the metric plumbing.
    model = _linear_model(K=16, N=4)
    report = onnxsim.measure_accuracy_drop(model, model, num_samples=4, seed=2)
    assert report.worst_relative_l2 == 0.0
    assert report.worst_cosine_distance == 0.0
    assert report.all_finite


def test_measure_accuracy_drop_casts_inputs_for_keep_io_types_false():
    # fp16 with keep_io_types=False redeclares the graph's own inputs as
    # float16 -- the float32 calibration data must be auto-cast to match, or
    # the quantized model's session would reject it outright.
    model = _linear_model(K=16, N=4)
    quantized = onnxsim.quantize_fp16(model, keep_io_types=False)
    assert quantized.graph.input[0].type.tensor_type.elem_type == (
        onnx.TensorProto.FLOAT16
    )

    report = onnxsim.measure_accuracy_drop(model, quantized, num_samples=4, seed=3)
    assert report.worst_relative_l2 < 0.01
    assert report.all_finite


def test_measure_accuracy_drop_uses_supplied_calibration_data():
    model = _linear_model(K=8, N=4)
    quantized = onnxsim.quantize_dynamic(model)
    calibration_data = [{"X": np.ones((4, 8), dtype=np.float32)}]

    report = onnxsim.measure_accuracy_drop(
        model, quantized, calibration_data=calibration_data
    )
    assert report.num_samples == 1


# --------------------------------------------------------------------------- #
# recommend_quantization() / quantize_auto()
# --------------------------------------------------------------------------- #
def test_recommend_quantization_returns_first_candidate_meeting_budget():
    model = _linear_model(K=64, N=64, opset=21, seed=0)

    recommendation = onnxsim.recommend_quantization(model, accuracy_budget=0.05)

    assert recommendation.meets_budget
    assert recommendation.report.worst_relative_l2 < 0.05
    assert recommendation.report.all_finite
    # Not pinning down *which* scheme wins here: weight_only/int4's own error
    # on this model sits close enough to 0.05 that it can land on either side
    # of the budget depending on the execution backend's exact int4
    # rounding (observed both ~0.10 and <0.05 across environments) --
    # test_recommend_quantization_tries_more_compressed_schemes_first below
    # covers the search-order behavior with a budget decisively wide of that
    # race instead.
    assert _initializer_nbytes(recommendation.quantized_model) < _initializer_nbytes(
        model
    )


def test_recommend_quantization_tries_more_compressed_schemes_first():
    # weight_only/int4 (block-wise 4-bit) sorts before dynamic/int8 in
    # DEFAULT_QUANTIZATION_CANDIDATES; a budget loose enough for int4's own
    # (lossier) error should pick it over the less-compressed schemes tried
    # after it.
    model = _linear_model(K=64, N=64, opset=21, seed=0)

    recommendation = onnxsim.recommend_quantization(model, accuracy_budget=0.3)

    assert recommendation.meets_budget
    assert recommendation.config.scheme == "weight_only"
    assert recommendation.config.dtype == "int4"


def test_recommend_quantization_shrink_check_rejects_a_noop_win():
    # A model whose weight isn't structurally ternary makes "ternary" (this
    # module's first, most-aggressive candidate) a no-op -- quantize()
    # returns the model unchanged, which would look like a perfect
    # (zero-error) win without the reachable-initializer shrink check.
    model = _linear_model(K=64, N=64, opset=21, seed=0)

    recommendation = onnxsim.recommend_quantization(model, accuracy_budget=1.0)

    assert recommendation.config.scheme != "ternary"


def test_recommend_quantization_falls_back_to_least_lossy_when_no_candidate_fits():
    model = _linear_model(K=64, N=64, opset=21, seed=0)

    recommendation = onnxsim.recommend_quantization(model, accuracy_budget=1e-9)

    assert not recommendation.meets_budget
    assert recommendation.report.all_finite
    # A generous upper bound, not a tight one: the exact least-lossy error
    # depends on the execution backend's numerics (see the comment on
    # test_recommend_quantization_returns_first_candidate_meeting_budget),
    # this just confirms the fallback is a real, small-but-nonzero drop
    # rather than something badly broken.
    assert 0 < recommendation.report.worst_relative_l2 < 0.01


def test_recommend_quantization_raises_when_no_candidate_applies():
    model = _linear_model(K=64, N=64, opset=21, seed=0)

    with pytest.raises(ValueError):
        onnxsim.recommend_quantization(model, accuracy_budget=0.05, candidates=[])


def test_recommend_quantization_custom_candidates_restricts_search():
    model = _linear_model(K=64, N=64, opset=21, seed=0)

    recommendation = onnxsim.recommend_quantization(
        model,
        accuracy_budget=0.5,
        candidates=[onnxsim.QuantizationConfig(scheme="dynamic")],
    )

    assert recommendation.config.scheme == "dynamic"


def test_quantize_auto_returns_quantized_model():
    model = _linear_model(K=64, N=64, opset=21, seed=0)

    quantized = onnxsim.quantize_auto(model, accuracy_budget=0.05)

    onnx.checker.check_model(quantized)
    report = onnxsim.measure_accuracy_drop(model, quantized, seed=1)
    assert report.worst_relative_l2 < 0.05


def test_quantize_auto_warns_when_no_candidate_meets_budget():
    model = _linear_model(K=64, N=64, opset=21, seed=0)

    with pytest.warns(UserWarning, match="no quantization candidate met"):
        onnxsim.quantize_auto(model, accuracy_budget=1e-9)
