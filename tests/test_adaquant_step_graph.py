"""Tests for ``onnxsim.adaquant``'s step-graph path -- AdaQuant's joint
optimization over weight rounding *and* the activation's (scale, zero_point)
expressed as an ONNX graph (``onnxsim/qat_graph.py``, ``docs/qat.md``) so it
can run on a GPU, an NPU execution provider, or WebGPU instead of only in host
numpy.

``onnxsim.adaround``'s own port is tested in ``tests/test_qat_graph.py``;
these are the questions that port's tests cannot answer for this one, because
AdaQuant optimizes strictly more. Three things matter:

1. the graph really is the same optimization as the numpy loop it replaces,
   over all *three* parameter groups at once -- not merely another one that
   also happens to help;
2. it stays inside the operator set the accelerator backends actually
   implement (a step graph that reached for a convenient op no NPU execution
   provider supports would pass every numerical test here and still be useless
   for what it exists for); and
3. ``apply_adaquant(..., step_providers=...)`` still produces a valid,
   improved model end to end.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

# The allowlist tests/test_qat_graph.py pins for every step graph, imported
# rather than copied so the two cannot drift: whatever that file decides the
# accelerator backends cover, this graph is held to as well.
from test_qat_graph import _ALLOWED_OPS

import onnxsim
from onnxsim import adaquant

ort = pytest.importorskip("onnxruntime")

_NUMPY_LOOP_KWARGS = dict(
    num_iterations=200,
    weight_learning_rate=0.1,
    activation_learning_rate=0.01,
    reg_param=0.01,
    warm_start=0.2,
    beta_range=(20.0, 2.0),
)


def _layer_case(seed, rows=40, n=8, k=32):
    """One layer's optimization problem: real-shaped activations, a float
    weight with its symmetric per-output-channel INT8 scale, and the min-max
    calibrated activation (scale, zero_point) AdaQuant starts from.

    The activation gets a couple of large, positively-shifted channels for the
    same reason ``tests/test_adaquant.py`` does: a calibrated zero-point that
    is neither 0 nor 128 is what gives the joint optimization something to do
    on the activation side at all.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, k))
    for channel in (3, 7):
        x[:, channel % k] = x[:, channel % k] * 15.0 + 8.0
    w = rng.standard_normal((n, k)) * 0.5
    scale_n = np.abs(w).max(axis=1) / 127.0

    x_min, x_max = min(0.0, float(x.min())), max(0.0, float(x.max()))
    x_scale = (x_max - x_min) / 255.0
    x_zp = float(np.clip(round(-x_min / x_scale), 0, 255))
    return x, w, scale_n, x_scale, x_zp


def _reconstruction_error(x, w, scale_n, codes, x_scale, x_zp):
    """``||x @ w.T - dequantize(quantize(x)) @ (codes * scale).T||`` -- the
    loss AdaQuant actually minimizes, including the activation's own
    quantization, evaluated at a candidate solution."""
    scale_nk = np.repeat(scale_n[:, None], w.shape[1], axis=1)
    xq = np.clip(np.round(x / x_scale) + x_zp, 0.0, 255.0)
    xdq = (xq - x_zp) * x_scale
    return float(np.linalg.norm(x @ w.T - xdq @ (codes * scale_nk).T))


def _round_to_nearest_codes(w, scale_n):
    scale_nk = np.repeat(scale_n[:, None], w.shape[1], axis=1)
    return np.clip(np.round(w / scale_nk), -127.0, 127.0)


def test_adaquant_step_graph_agrees_with_the_numpy_loop():
    """The step-graph path is the same optimization ``_optimize_adaquant``
    performs, over all three parameter groups jointly.

    Measured over five seeds, the two paths pick the *identical* floor/ceil
    decision for at least 99% of weight elements, land on the same integer
    zero-point, and agree on the activation scale to better than 1% relative
    -- the residual being float32-vs-float64 boundary rounding, which for this
    algorithm has two entry points rather than AdaRound's one: an element
    whose relaxation sits near h = 0.5, and an activation whose ``x / scale``
    sits within an ulp of a .5 boundary and so quantizes to a different
    integer in the two paths.
    """
    for seed in range(5):
        x, w, scale_n, x_scale0, x_zp0 = _layer_case(seed)

        numpy_codes, numpy_scale, numpy_zp = adaquant._optimize_adaquant(
            w, scale_n, x, x_scale0, x_zp0, **_NUMPY_LOOP_KWARGS
        )
        graph_codes, graph_scale, graph_zp = adaquant._optimize_adaquant_on_graph(
            w, scale_n, x, x_scale0, x_zp0, providers=None, **_NUMPY_LOOP_KWARGS
        )

        assert (numpy_codes == graph_codes).mean() > 0.99
        assert graph_zp == numpy_zp
        assert graph_scale == pytest.approx(numpy_scale, rel=1e-2)

        # And they arrive at the same place on the objective, which is better
        # than the round-to-nearest weights and calibrated range they started
        # from.
        rtn_err = _reconstruction_error(
            x, w, scale_n, _round_to_nearest_codes(w, scale_n), x_scale0, x_zp0
        )
        numpy_err = _reconstruction_error(
            x, w, scale_n, numpy_codes, numpy_scale, numpy_zp
        )
        graph_err = _reconstruction_error(
            x, w, scale_n, graph_codes, graph_scale, graph_zp
        )
        assert graph_err < rtn_err
        assert graph_err == pytest.approx(numpy_err, rel=0.05)


def test_adaquant_step_graph_optimizes_the_activation_range_too():
    """The activation branch is actually being optimized, not just carried:
    the scale and zero-point the graph path returns have moved off the
    calibrated warm start, and the loss is lower *at fixed weights* than it
    would be with the calibrated range."""
    x, w, scale_n, x_scale0, x_zp0 = _layer_case(1)
    codes, x_scale, x_zp = adaquant._optimize_adaquant_on_graph(
        w, scale_n, x, x_scale0, x_zp0, providers=None, **_NUMPY_LOOP_KWARGS
    )

    assert x_scale != pytest.approx(x_scale0, rel=1e-6)
    assert 0 <= x_zp <= 255
    tuned = _reconstruction_error(x, w, scale_n, codes, x_scale, x_zp)
    warm_start = _reconstruction_error(x, w, scale_n, codes, x_scale0, x_zp0)
    assert tuned < warm_start


def test_adaquant_step_graph_is_a_valid_model_declaring_all_nine_state_tensors():
    step = adaquant._build_adaquant_step_graph(8, 3, 4)
    onnx.checker.check_model(step.model)

    input_names = {i.name for i in step.model.graph.input}
    output_names = {o.name for o in step.model.graph.output}
    assert set(step.state) <= input_names
    assert set(step.state.values()) <= output_names
    assert step.loss_name in output_names
    # Three parameter groups, each with its own pair of Adam moments: the
    # relaxation is per weight element, the activation's two are rank-0.
    assert set(step.state) == {
        "v",
        "m_v",
        "vv_v",
        "log_s",
        "m_s",
        "vv_s",
        "zp",
        "m_zp",
        "vv_zp",
    }


def test_adaquant_step_graph_stays_within_broadly_supported_ops():
    step = adaquant._build_adaquant_step_graph(8, 3, 4)
    used = {node.op_type for node in step.model.graph.node}
    assert used <= _ALLOWED_OPS, f"unsupported ops in step graph: {used - _ALLOWED_OPS}"


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


def _matmul_model(K, N, seed):
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [onnx.numpy_helper.from_array(weight, "W")],
    )


def _calibration(K, rows, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, K)).astype(np.float32)
    for channel in (3, 7):
        x[:, channel] = x[:, channel] * 15.0 + 8.0
    return x


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)[0].astype(np.float64)


def test_apply_adaquant_on_a_step_graph_beats_the_untuned_quantized_model():
    """End to end through the public API: ``step_providers`` moves the joint
    optimization onto an execution provider, and what comes back is still a
    valid, AdaQuant-improved model whose untouched tensors are untouched."""
    K, N, rows = 64, 16, 64
    model = _matmul_model(K, N, seed=0)
    x = _calibration(K, rows, seed=1)
    calibration_data = [{"X": x}]

    quant = onnxsim.quantize_static(model, calibration_data=calibration_data)
    assert any(n.op_type == "QuantizeLinear" for n in quant.graph.node)

    tuned = onnxsim.apply_adaquant(
        model,
        quant,
        calibration_data=calibration_data,
        num_iterations=200,
        step_providers=["CPUExecutionProvider"],
    )
    onnx.checker.check_model(tuned)

    reference = _run(model, {"X": x})
    quant_err = np.linalg.norm(reference - _run(quant, {"X": x}))
    tuned_err = np.linalg.norm(reference - _run(tuned, {"X": x}))
    assert tuned_err < quant_err

    # AdaQuant promises to rewrite exactly three kinds of initializer -- the
    # weight's INT8 codes and the activation's scale/zero-point -- and to
    # leave everything else, the weight's own per-channel scale included,
    # byte-identical. Shapes and dtypes are preserved throughout.
    before = {t.name: t for t in quant.graph.initializer}
    after = {t.name: t for t in tuned.graph.initializer}
    assert set(before) == set(after)
    ql = next(n for n in tuned.graph.node if n.op_type == "QuantizeLinear")
    activation_params = {ql.input[1], ql.input[2]}
    weight_codes = {
        n.input[0]
        for n in tuned.graph.node
        if n.op_type == "DequantizeLinear" and n.input[0] in before
    }
    changed = {
        name
        for name in before
        if before[name].SerializeToString() != after[name].SerializeToString()
    }
    assert changed <= activation_params | weight_codes
    assert changed & weight_codes, "no weight was actually retuned"
    for name in before:
        assert list(before[name].dims) == list(after[name].dims)
        assert before[name].data_type == after[name].data_type


def test_apply_adaquant_step_providers_are_validated():
    """An execution provider the installed onnxruntime does not have fails
    loudly rather than silently running on the CPU -- backend.validate_providers'
    own contract, inherited by the step-graph path."""
    K, N, rows = 32, 8, 16
    model = _matmul_model(K, N, seed=2)
    x = _calibration(K, rows, seed=3)
    calibration_data = [{"X": x}]
    quant = onnxsim.quantize_static(model, calibration_data=calibration_data)

    with pytest.raises(ValueError, match="not available"):
        onnxsim.apply_adaquant(
            model,
            quant,
            calibration_data=calibration_data,
            num_iterations=5,
            step_providers=["NoSuchExecutionProvider"],
        )
