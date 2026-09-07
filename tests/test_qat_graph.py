"""Tests for ``onnxsim.qat_graph`` -- one optimizer step expressed as an ONNX
graph, so an optimization loop that used to be host numpy can run on any
execution provider (a GPU, an NPU EP, or WebGPU in the WASM build). See
``onnxsim/qat_graph.py`` and ``docs/qat.md``.

Two things are worth testing about that claim, and both are here: that the
graph really does optimize (it agrees with the numpy Adam loop it replaces,
on the generic machinery *and* on the first real caller,
``onnxsim.adaround``), and that it stays inside an operator set the
accelerator backends actually implement -- a step graph that reached for a
convenient op no NPU execution provider supports would pass every numerical
test here and still be useless for what it exists for.
"""

import numpy as np
import onnx
import pytest

import onnxsim
from onnxsim import adaround, qat_graph

ort = pytest.importorskip("onnxruntime")

# Operators a step graph may use. Everything here is a plain arithmetic,
# comparison or reduction op with broad coverage across onnxruntime's
# execution providers (including onnxruntime-web's WebGPU backend and the
# WebNN/NPU ones). Ops deliberately kept out: control flow, boolean logic,
# Where, anything that would make a graph runnable only on the CPU.
_ALLOWED_OPS = {
    "Abs",
    "Add",
    "Cast",
    "Clip",
    "Div",
    "Greater",
    "Less",
    "MatMul",
    "Mul",
    "Pow",
    "ReduceMean",
    "Sigmoid",
    "Sign",
    "Sqrt",
    "Sub",
    "Transpose",
}


def _linear_fit_step_graph(rows, k, n):
    """A step graph fitting ``y = x @ W.T`` by Adam -- the smallest useful
    exercise of the builder, the Adam nodes and the state plumbing."""
    b = qat_graph.GraphBuilder()
    y_hat = b.matmul("x", b.transpose("w"))
    diff = b.sub(y_hat, "y")
    grad = b.mul(b.matmul(b.transpose(diff), "x"), b.const(2.0 / (rows * n)))
    w_next, m_next, v_next = qat_graph.adam_update(
        b, "w", grad, "m", "vv", "lr", "m_correction", "v_correction"
    )
    return qat_graph.make_step_graph(
        b,
        constants={"x": [rows, k], "y": [rows, n]},
        state={
            "w": ([n, k], w_next),
            "m": ([n, k], m_next),
            "vv": ([n, k], v_next),
        },
        scalars=["lr", "m_correction", "v_correction"],
        loss=b.mean_square(diff),
    )


def _numpy_adam_linear_fit(x, y, num_steps, lr=0.1):
    """The same fit, as the hand-rolled numpy Adam loop this repo's
    reconstruction passes all use."""
    rows, k = x.shape
    n = y.shape[1]
    w = np.zeros((n, k))
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for t in range(num_steps):
        grad = 2.0 * ((x @ w.T - y).T @ x) / (rows * n)
        m = qat_graph.ADAM_BETA1 * m + (1.0 - qat_graph.ADAM_BETA1) * grad
        v = qat_graph.ADAM_BETA2 * v + (1.0 - qat_graph.ADAM_BETA2) * grad * grad
        m_hat = m / (1.0 - qat_graph.ADAM_BETA1 ** (t + 1))
        v_hat = v / (1.0 - qat_graph.ADAM_BETA2 ** (t + 1))
        w = w - lr * m_hat / (np.sqrt(v_hat) + qat_graph.ADAM_EPS)
    return w


def _run_linear_fit(step, x, y, num_steps, lr=0.1, losses=None, providers=None):
    # w, and Adam's two moments, all share the parameter's [n, k] shape.
    zeros = np.zeros((y.shape[1], x.shape[1]))
    return qat_graph.run_step_graph(
        step,
        constants={"x": x, "y": y},
        state={"w": zeros, "m": zeros, "vv": zeros},
        num_steps=num_steps,
        scalars=lambda t: dict(lr=lr, **qat_graph.adam_bias_corrections(t)),
        providers=providers,
        losses=losses,
    )


def test_step_graph_adam_matches_the_numpy_loop_it_replaces():
    rng = np.random.default_rng(3)
    rows, k, n = 32, 5, 3
    x = rng.standard_normal((rows, k))
    w_true = rng.standard_normal((n, k))
    y = x @ w_true.T

    step = _linear_fit_step_graph(rows, k, n)
    final = _run_linear_fit(step, x, y, num_steps=400)
    reference = _numpy_adam_linear_fit(x, y, num_steps=400)

    # Both recover the generating weight; the step graph computes in float32
    # (what accelerators have) against the loop's float64, so they agree to
    # float32 precision rather than exactly.
    assert np.abs(final["w"] - w_true).max() < 1e-5
    assert np.abs(final["w"] - reference).max() < 1e-5


def test_step_graph_reports_a_decreasing_loss():
    rng = np.random.default_rng(4)
    rows, k, n = 24, 4, 2
    x = rng.standard_normal((rows, k))
    y = x @ rng.standard_normal((n, k)).T

    losses = []
    step = _linear_fit_step_graph(rows, k, n)
    _run_linear_fit(step, x, y, num_steps=200, losses=losses)

    assert len(losses) == 200
    assert losses[-1] < losses[0] * 1e-3


def test_step_graph_is_a_pure_function_of_its_state():
    """Running N steps and then M more is the same as running N + M: nothing
    is carried between calls except the state the graph declares."""
    rng = np.random.default_rng(5)
    rows, k, n = 16, 4, 3
    x = rng.standard_normal((rows, k))
    y = x @ rng.standard_normal((n, k)).T
    step = _linear_fit_step_graph(rows, k, n)

    straight = _run_linear_fit(step, x, y, num_steps=60)
    part = _run_linear_fit(step, x, y, num_steps=25)
    resumed = qat_graph.run_step_graph(
        step,
        constants={"x": x, "y": y},
        state=part,
        num_steps=35,
        scalars=lambda t: dict(lr=0.1, **qat_graph.adam_bias_corrections(t + 25)),
    )
    for name in ("w", "m", "vv"):
        np.testing.assert_allclose(straight[name], resumed[name], rtol=0, atol=1e-6)


def test_step_graph_is_a_valid_model_and_declares_its_state():
    step = _linear_fit_step_graph(8, 3, 2)
    onnx.checker.check_model(step.model)

    input_names = {i.name for i in step.model.graph.input}
    output_names = {o.name for o in step.model.graph.output}
    assert set(step.state) <= input_names
    assert set(step.state.values()) <= output_names
    assert step.loss_name in output_names


@pytest.mark.parametrize(
    "step",
    [
        _linear_fit_step_graph(8, 3, 2),
        adaround._build_rounding_step_graph(8, 3, 4, -7.0, 7.0),
    ],
    ids=["linear_fit", "adaround"],
)
def test_step_graphs_stay_within_broadly_supported_ops(step):
    used = {node.op_type for node in step.model.graph.node}
    assert used <= _ALLOWED_OPS, f"unsupported ops in step graph: {used - _ALLOWED_OPS}"


def _rounding_case(seed, rows=40, n=8, k=32):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, k))
    w = rng.standard_normal((n, k)) * 0.5
    scale = np.repeat(np.abs(w).max(axis=1, keepdims=True) / 7.0, k, axis=1)
    return x, w, scale


def _reconstruction_error(x, w, scale, codes):
    return float(np.linalg.norm(x @ w.T - x @ (codes * scale).T))


def test_adaround_step_graph_agrees_with_the_numpy_loop():
    """The step-graph path is the same optimization as ``_optimize_rounding``,
    not merely another one that also happens to help: the two pick the same
    floor/ceil decision for all but a handful of elements (the ones whose
    relaxation lands near the 0.5 boundary, where float32 and float64 can
    round apart) and land on the same reconstruction error."""
    kwargs = dict(
        n_min=-7.0,
        n_max=7.0,
        num_iterations=200,
        learning_rate=0.1,
        reg_param=0.01,
        warm_start=0.2,
        beta_range=(20.0, 2.0),
    )
    for seed in range(3):
        x, w, scale = _rounding_case(seed)
        numpy_codes = adaround._optimize_rounding(w, scale, x, **kwargs)
        graph_codes = adaround._optimize_rounding_on_graph(
            w, scale, x, providers=None, **kwargs
        )
        assert (numpy_codes == graph_codes).mean() > 0.95

        rtn = np.clip(np.round(w / scale), -7.0, 7.0)
        rtn_err = _reconstruction_error(x, w, scale, rtn)
        numpy_err = _reconstruction_error(x, w, scale, numpy_codes)
        graph_err = _reconstruction_error(x, w, scale, graph_codes)
        assert graph_err < rtn_err
        assert graph_err == pytest.approx(numpy_err, rel=0.1)


def test_apply_adaround_on_a_step_graph_beats_round_to_nearest():
    """End to end through the public API: ``step_providers`` moves the
    optimization onto an execution provider and the result is still an
    AdaRound-improved model."""
    rng = np.random.default_rng(11)
    k, n, batch = 64, 16, 32
    weight = rng.standard_normal((k, n)).astype(np.float32) * 0.5
    float_model = onnx.parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 21]>
        g (float[{batch},{k}] X) => (float[{batch},{n}] Y) {{
          Y = MatMul(X, W)
        }}
        """
    )
    float_model.graph.initializer.append(onnx.numpy_helper.from_array(weight, "W"))
    quant_model = onnxsim.quantize_weight_only_int4(float_model)

    x = rng.standard_normal((batch, k)).astype(np.float32)
    tuned = onnxsim.apply_adaround(
        float_model,
        quant_model,
        calibration_data=[{"X": x}],
        num_iterations=200,
        step_providers=["CPUExecutionProvider"],
    )
    onnx.checker.check_model(tuned)

    def run(model):
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        return sess.run(None, {"X": x})[0].astype(np.float64)

    reference = run(float_model)
    rtn_err = np.linalg.norm(reference - run(quant_model))
    tuned_err = np.linalg.norm(reference - run(tuned))
    assert tuned_err < rtn_err


def test_apply_adaround_step_providers_are_validated():
    """An execution provider the installed onnxruntime does not have fails
    loudly rather than silently running on the CPU -- backend.validate_providers'
    own contract, inherited by the step-graph path."""
    rng = np.random.default_rng(12)
    k, n, batch = 32, 8, 8
    float_model = onnx.parser.parse_model(
        f"""
        <ir_version: 10, opset_import: ["": 21]>
        g (float[{batch},{k}] X) => (float[{batch},{n}] Y) {{
          Y = MatMul(X, W)
        }}
        """
    )
    float_model.graph.initializer.append(
        onnx.numpy_helper.from_array(
            (rng.standard_normal((k, n)) * 0.5).astype(np.float32), "W"
        )
    )
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    with pytest.raises(ValueError, match="not available"):
        onnxsim.apply_adaround(
            float_model,
            quant_model,
            calibration_data=[
                {"X": rng.standard_normal((batch, k)).astype(np.float32)}
            ],
            num_iterations=5,
            step_providers=["NoSuchExecutionProvider"],
        )
