"""Tests for ``onnxsim.apply_mixed_precision_quantization`` -- see
``onnxsim/mixed_precision.py`` for the technique (calibration-driven
per-layer choice between block-wise INT8 and block-wise INT4) -- and for
``onnxsim.search_mixed_precision_for_budget``, the accuracy-aware search
over that dispatcher's own ``high_bits_fraction`` (see that function's
docstring in ``onnxsim/mixed_precision.py``).
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _vi(name, shape):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(nodes, inputs, outputs, initializer, opset=21):
    graph = onnx.helper.make_graph(nodes, "g", inputs, outputs, initializer)
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=10
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def _two_layer_model(K=32, H=16, N=8, seed=0, opset=21):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((K, H)).astype(np.float32) * 0.5
    # w2's rows get planted, large-magnitude outliers, making it far more
    # sensitive to INT4 quantization than w1 -- so a good sensitivity
    # ranking should pick THIS layer for the INT8 tier.
    w2 = rng.standard_normal((H, N)).astype(np.float32) * 0.05
    w2[0, :] = 20.0
    nodes = [
        onnx.helper.make_node("MatMul", ["X", "W1"], ["H1"]),
        onnx.helper.make_node("MatMul", ["H1", "W2"], ["Y"]),
    ]
    return _model(
        nodes,
        [_vi("X", ["batch", K])],
        [_vi("Y", ["batch", N])],
        [_f32(w1, "W1"), _f32(w2, "W2")],
        opset=opset,
    )


def test_mixed_precision_output_stays_close_to_float_via_onnxruntime():
    model = _two_layer_model(K=32, H=16, N=8, seed=0)
    q = onnxsim.apply_mixed_precision_quantization(
        model, block_size=8, high_bits_fraction=0.5, num_samples=16, seed=1
    )
    onnx.checker.check_model(q)

    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 32)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(q, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_mixed_precision_picks_the_more_sensitive_layer_for_int8():
    model = _two_layer_model(K=32, H=16, N=8, seed=3)
    q = onnxsim.apply_mixed_precision_quantization(
        model, block_size=8, high_bits_fraction=0.5, num_samples=32, seed=4
    )
    codes_by_prefix = {
        t.name: t for t in q.graph.initializer if t.name.endswith("_codes")
    }
    w2_codes = next(t for name, t in codes_by_prefix.items() if name.startswith("W2_"))
    w1_codes = next(t for name, t in codes_by_prefix.items() if name.startswith("W1_"))
    # W2 has the planted outlier row -- it must be the INT8 (more precise)
    # tier, while the ordinary W1 stays at INT4.
    assert w2_codes.data_type == onnx.TensorProto.INT8
    assert w1_codes.data_type == onnx.TensorProto.INT4


def test_mixed_precision_zero_fraction_matches_all_int4():
    model = _two_layer_model(K=32, H=16, N=8, seed=5)
    q = onnxsim.apply_mixed_precision_quantization(
        model, block_size=8, high_bits_fraction=0.0, num_samples=16, seed=6
    )
    codes = [t for t in q.graph.initializer if t.name.endswith("_codes")]
    assert len(codes) == 2
    assert all(t.data_type == onnx.TensorProto.INT4 for t in codes)


def test_mixed_precision_one_fraction_matches_all_int8():
    model = _two_layer_model(K=32, H=16, N=8, seed=7)
    q = onnxsim.apply_mixed_precision_quantization(
        model, block_size=8, high_bits_fraction=1.0, num_samples=16, seed=8
    )
    codes = [t for t in q.graph.initializer if t.name.endswith("_codes")]
    assert len(codes) == 2
    assert all(t.data_type == onnx.TensorProto.INT8 for t in codes)


def test_mixed_precision_declines_when_k_not_divisible_by_block_size():
    rng = np.random.default_rng(9)
    weight = rng.standard_normal((20, 4)).astype(np.float32)  # 20 not a multiple of 8
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", ["batch", 20])], [_vi("Y", ["batch", 4])], [_f32(weight, "W")]
    )
    q = onnxsim.apply_mixed_precision_quantization(model, block_size=8)
    assert q.SerializeToString() == model.SerializeToString()


def test_mixed_precision_declines_non_constant_weight():
    nodes = [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])]
    model = _model(
        nodes, [_vi("X", [4, 32]), _vi("W", [32, 4])], [_vi("Y", [4, 4])], []
    )
    q = onnxsim.apply_mixed_precision_quantization(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_mixed_precision_noop_when_no_matmul_present():
    nodes = [onnx.helper.make_node("Relu", ["X"], ["Y"])]
    model = _model(nodes, [_vi("X", [4, 4])], [_vi("Y", [4, 4])], [])
    result = onnxsim.apply_mixed_precision_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_mixed_precision_declines_below_opset21():
    model = _two_layer_model(K=32, H=16, N=8, opset=13)
    result = onnxsim.apply_mixed_precision_quantization(model)
    assert result.SerializeToString() == model.SerializeToString()


# --------------------------------------------------------------------------- #
# search_mixed_precision_for_budget
# --------------------------------------------------------------------------- #
# Named `_text_model` (rather than reusing `_model` above) since this repo's
# CLAUDE.md asks new test model-building code to go through `onnx.parser`,
# but `_model` above already names the onnx.helper-based builder the earlier
# tests in this file use.
def _text_model(body, initializer=(), opset=21, ir_version=10):
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


def _search_test_model(seed=0, outlier=20.0):
    # Two chained MatMuls, K=32/H=16/N=8 (block_size=8 divides both). W2's
    # first row (every output channel's weight for hidden unit 0) is set to
    # a large-but-not-extreme outlier: big enough (400x W2's other weights'
    # scale) that the sensitivity ranking unambiguously ranks W2 above W1 on
    # any platform (their MSE*activation-energy scores differ by ~5 orders
    # of magnitude, nowhere near a rounding-boundary tie -- see this repo's
    # own CLAUDE.md note on `tests/test_gptaq.py`'s prior single-seed
    # flakiness for why that margin matters), while still leaving room for
    # promoting more layers to INT8 to actually reduce the measured output
    # error (an outlier so extreme it saturates the block scale would make
    # INT4-vs-INT8 equally (in)accurate for that block, which would defeat
    # the point of this test).
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((32, 16)) * 0.5).astype(np.float32)
    w2 = (rng.standard_normal((16, 8)) * 0.05).astype(np.float32)
    w2[0, :] = outlier
    return _text_model(
        """
        agraph (float[batch, 32] X) => (float[batch, 8] Y)
        {
            H1 = MatMul(X, W1)
            Y = MatMul(H1, W2)
        }
        """,
        initializer=[_f32(w1, "W1"), _f32(w2, "W2")],
    )


def test_search_mixed_precision_promotes_a_layer_to_meet_a_tight_budget():
    model = _search_test_model(seed=0, outlier=20.0)
    result = onnxsim.search_mixed_precision_for_budget(
        model, accuracy_budget=0.15, block_size=8, num_samples=16, seed=1
    )
    assert result.meets_budget
    assert result.report.all_finite
    assert result.report.worst_relative_l2 < 0.15
    # Starts at the smallest fraction and has to promote at least one layer
    # (in fact needs every eligible layer at INT8 here) before meeting a
    # tight budget -- a single INT8 layer alone (fractions 0.3/0.5) still
    # leaves worst_relative_l2 far above 0.15.
    assert (
        result.fractions_tried[0] == onnxsim.mixed_precision.DEFAULT_SEARCH_FRACTIONS[0]
    )
    assert len(result.fractions_tried) > 1
    assert result.high_bits_fraction > 0.0


def test_search_mixed_precision_stops_at_first_fraction_for_generous_budget():
    model = _search_test_model(seed=0, outlier=20.0)
    result = onnxsim.search_mixed_precision_for_budget(
        model, accuracy_budget=1.0, block_size=8, num_samples=16, seed=1
    )
    assert result.meets_budget
    default_fractions = onnxsim.mixed_precision.DEFAULT_SEARCH_FRACTIONS
    assert result.fractions_tried == [default_fractions[0]]
    assert result.high_bits_fraction == default_fractions[0]


def test_search_mixed_precision_exhausts_fractions_for_impossible_budget():
    model = _search_test_model(seed=0, outlier=20.0)
    fractions = (0.0, 0.1, 0.4, 1.0)
    result = onnxsim.search_mixed_precision_for_budget(
        model,
        accuracy_budget=1e-12,
        fractions=fractions,
        block_size=8,
        num_samples=16,
        seed=1,
    )
    assert not result.meets_budget
    assert result.fractions_tried == list(fractions)
    assert result.high_bits_fraction == fractions[-1]


def test_search_mixed_precision_winner_matches_direct_call():
    model = _search_test_model(seed=0, outlier=20.0)
    calibration_data = onnxsim.generate_random_calibration_data(
        model, num_samples=16, seed=1
    )
    result = onnxsim.search_mixed_precision_for_budget(
        model,
        accuracy_budget=0.15,
        block_size=8,
        calibration_data=calibration_data,
    )
    direct = onnxsim.apply_mixed_precision_quantization(
        model,
        calibration_data=calibration_data,
        high_bits_fraction=result.high_bits_fraction,
        block_size=8,
    )
    assert result.quantized_model.SerializeToString() == direct.SerializeToString()


def test_search_mixed_precision_respects_custom_fractions():
    model = _search_test_model(seed=0, outlier=20.0)
    calibration_data = onnxsim.generate_random_calibration_data(
        model, num_samples=16, seed=1
    )
    custom_fractions = (0.0, 1.0)
    result = onnxsim.search_mixed_precision_for_budget(
        model,
        # Impossibly tight so the search never stops early -- every fraction
        # in `custom_fractions` must actually be tried.
        accuracy_budget=1e-12,
        fractions=custom_fractions,
        block_size=8,
        calibration_data=calibration_data,
    )
    assert result.fractions_tried == list(custom_fractions)
    for frac in result.fractions_tried:
        assert frac in custom_fractions
    # None of the default sweep's own intermediate values (e.g. 0.2, 0.3,
    # 0.5, 0.75) were ever tried.
    default_fractions = onnxsim.mixed_precision.DEFAULT_SEARCH_FRACTIONS
    for frac in default_fractions:
        if frac not in custom_fractions:
            assert frac not in result.fractions_tried
