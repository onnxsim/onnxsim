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
from onnxsim import adaround, backend, qat_graph

ort = pytest.importorskip("onnxruntime")

# The operator set a step graph may emit is the package's own, not a copy of
# it: two allowlists that drift apart would each certify a graph the other
# rejects. See qat_graph.EP_FRIENDLY_OPS for what earns a place in it.
_ALLOWED_OPS = qat_graph.EP_FRIENDLY_OPS


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


# --- Device-resident state (``bind_state``) ----------------------------------
#
# ``run_step_graph`` keeps the constants and the state on the execution
# provider's device between steps, through onnxruntime's ``IOBinding``
# (``onnxsim.backend.Runner.bind_loop``). That is an optimization, so the tests
# below are about it changing *nothing*: the same numbers as the feed-per-step
# path, no corruption from onnxruntime reusing a bound buffer, and a clean
# fallback whenever the binding cannot be set up or cannot run.


def _linear_fit_case(seed, rows=48, k=8, n=6):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, k))
    y = x @ rng.standard_normal((n, k)).T
    return x, y, np.zeros((n, k))


def test_bound_and_unbound_loops_agree():
    """The whole contract of ``bind_state``: it moves tensors, not numbers.

    Exact equality, not a tolerance -- same graph, same provider, same values,
    only a different way of handing onnxruntime the buffers -- so any
    divergence here is a real bug rather than float noise.
    """
    x, y, zeros = _linear_fit_case(20)
    step = _linear_fit_step_graph(x.shape[0], x.shape[1], y.shape[1])

    bound_losses: list = []
    unbound_losses: list = []
    bound = _run_linear_fit(step, x, y, num_steps=150, losses=bound_losses)
    unbound = qat_graph.run_step_graph(
        step,
        constants={"x": x, "y": y},
        state={"w": zeros, "m": zeros, "vv": zeros},
        num_steps=150,
        scalars=lambda t: dict(lr=0.1, **qat_graph.adam_bias_corrections(t)),
        losses=unbound_losses,
        bind_state=False,
    )

    for name in ("w", "m", "vv"):
        np.testing.assert_array_equal(bound[name], unbound[name])
    assert bound_losses == unbound_losses
    assert len(bound_losses) == 150


def test_bound_state_is_not_corrupted_by_buffer_reuse():
    """onnxruntime may hand a bound output buffer back on the next run, so a
    loop that fed a step's output straight back in as the next step's input --
    while binding that same buffer as an output again -- could read values a
    kernel is concurrently overwriting. ``BoundStepLoop`` double-buffers to
    make each run's reads and writes disjoint.

    The check: one bound loop of twelve chained steps against twelve one-step
    bound loops. Every one-step loop allocates its own binding and its own
    buffers, so no buffer can possibly survive from one step into the next
    there; if reuse were corrupting the chained loop, the two would drift
    apart. They agree exactly instead.
    """
    x, y, zeros = _linear_fit_case(21)
    step = _linear_fit_step_graph(x.shape[0], x.shape[1], y.shape[1])

    chained = _run_linear_fit(step, x, y, num_steps=12)

    fresh = {"w": zeros, "m": zeros, "vv": zeros}
    for t in range(12):
        fresh = qat_graph.run_step_graph(
            step,
            constants={"x": x, "y": y},
            state=fresh,
            num_steps=1,
            scalars=lambda _, t=t: dict(lr=0.1, **qat_graph.adam_bias_corrections(t)),
        )

    for name in ("w", "m", "vv"):
        np.testing.assert_array_equal(chained[name], fresh[name])


def test_bound_loop_leaves_the_callers_arrays_alone():
    """The state arrays a caller passes in are inputs, not scratch space.

    Worth its own test because onnxruntime's CPU ``OrtValue`` wraps the numpy
    buffer it is given rather than copying it, so binding a caller's array
    directly as one half of the ping-pong pair would quietly turn the caller's
    initial state into the loop's working memory.
    """
    x, y, _ = _linear_fit_case(22)
    step = _linear_fit_step_graph(x.shape[0], x.shape[1], y.shape[1])
    start = {
        name: np.full((y.shape[1], x.shape[1]), 0.25, dtype=np.float32)
        for name in ("w", "m", "vv")
    }
    before = {name: value.copy() for name, value in start.items()}

    final = qat_graph.run_step_graph(
        step,
        constants={"x": x, "y": y},
        state=start,
        num_steps=8,
        scalars=lambda t: dict(lr=0.1, **qat_graph.adam_bias_corrections(t)),
    )

    for name, value in before.items():
        np.testing.assert_array_equal(start[name], value)
    assert not np.array_equal(final["w"], before["w"])


def test_bound_loop_falls_back_when_a_bound_run_fails(monkeypatch):
    """A provider that accepts a binding and then refuses to run against it --
    an onnxruntime build without ``IOBinding`` support for that EP -- must land
    the caller on the feed-per-step path, with the right answer and exactly one
    loss per step rather than a half-written list from the abandoned attempt.
    """
    x, y, zeros = _linear_fit_case(23)
    step = _linear_fit_step_graph(x.shape[0], x.shape[1], y.shape[1])
    reference_losses: list = []
    reference = qat_graph.run_step_graph(
        step,
        constants={"x": x, "y": y},
        state={"w": zeros, "m": zeros, "vv": zeros},
        num_steps=40,
        scalars=lambda t: dict(lr=0.1, **qat_graph.adam_bias_corrections(t)),
        losses=reference_losses,
        bind_state=False,
    )

    attempts = []

    def refuse(self, iobinding, run_options=None):
        attempts.append(iobinding)
        raise RuntimeError("this provider has no IOBinding support")

    monkeypatch.setattr(ort.InferenceSession, "run_with_iobinding", refuse)

    losses: list = []
    final = _run_linear_fit(step, x, y, num_steps=40, losses=losses)

    # The binding was really tried -- which is also this file's evidence that
    # the default path through ``run_step_graph`` is the bound one.
    assert len(attempts) == 1
    for name in ("w", "m", "vv"):
        np.testing.assert_array_equal(final[name], reference[name])
    assert losses == reference_losses


def test_bind_state_falls_back_to_the_reference_evaluator(monkeypatch):
    """Without onnxruntime there is no binding and no device to bind to, and
    ``bind_state=True`` still has to run -- through onnx's reference evaluator,
    which is what the whole backend degrades to (onnxsim/onnxsim#441)."""
    x, y, zeros = _linear_fit_case(24, rows=16, k=4, n=3)
    step = _linear_fit_step_graph(x.shape[0], x.shape[1], y.shape[1])
    reference = _run_linear_fit(step, x, y, num_steps=30)

    monkeypatch.setattr(backend, "_HAS_ONNXRUNTIME", False)
    assert backend.Runner(step.model).bind_loop({}, {}) is None

    final = _run_linear_fit(step, x, y, num_steps=30)
    for name in ("w", "m", "vv"):
        # Both compute in float32; the two implementations' arithmetic differs
        # in the last bits, and 30 Adam steps amplify that a little.
        np.testing.assert_allclose(final[name], reference[name], rtol=0, atol=1e-5)


def test_bind_loop_declines_an_output_it_cannot_preallocate():
    """A bound output needs its buffer before the run that fills it, so a
    dynamic output shape is not bindable. ``bind_loop`` says so by returning
    ``None`` -- the caller's cue to run the ordinary way -- instead of raising.
    """
    model = onnx.parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 17]>
        g (float[N,2] s) => (float[N,2] out) {
          out = Add(s, s)
        }
        """
    )
    runner = backend.Runner(model)
    state = {"s": ("out", np.ones((3, 2), dtype=np.float32))}
    assert runner.bind_loop({}, state) is None
    # ... and the unbound call it falls back to still works.
    np.testing.assert_array_equal(
        runner({"s": np.ones((3, 2), dtype=np.float32)})["out"],
        np.full((3, 2), 2.0, dtype=np.float32),
    )


def test_bind_loop_declines_a_state_output_of_a_different_shape():
    """State threading requires the output to be shaped like the input it feeds
    back into. A graph that reshapes its state is not one this loop can carry,
    and saying so up front beats discovering it as a binding error mid-loop."""
    model = onnx.parser.parse_model(
        """
        <ir_version: 8, opset_import: ["": 17]>
        g (float[2,3] s) => (float[3,2] out) {
          out = Transpose<perm = [1, 0]>(s)
        }
        """
    )
    runner = backend.Runner(model)
    assert (
        runner.bind_loop({}, {"s": ("out", np.ones((2, 3), dtype=np.float32))}) is None
    )


def test_runner_call_still_returns_an_ordered_dict():
    """``Runner.__call__`` is the interface constant folding and the other
    callers use; the binding path is additive and must not have touched it."""
    x, y, _ = _linear_fit_case(25, rows=8, k=3, n=2)
    step = _linear_fit_step_graph(x.shape[0], x.shape[1], y.shape[1])
    zeros = np.zeros((y.shape[1], x.shape[1]), dtype=np.float32)
    runner = backend.Runner(step.model, output_names=[step.state["w"]])
    out = runner(
        {
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "w": zeros,
            "m": zeros,
            "vv": zeros,
            "lr": np.asarray(0.1, dtype=np.float32),
            "m_correction": np.asarray(10.0, dtype=np.float32),
            "v_correction": np.asarray(1000.0, dtype=np.float32),
        }
    )
    assert list(out) == [step.state["w"]]
    assert out[step.state["w"]].shape == zeros.shape
