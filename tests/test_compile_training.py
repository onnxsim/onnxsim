"""Tests for ``onnxsim.compile_training`` -- a ``torch.compile``-styled
training loop built out of onnxsim's own grad templating
(:mod:`onnxsim.graph_grad` for the backward pass, :mod:`onnxsim.qat_graph`
for the optimizer and the step graph itself).

The correctness claim under test is the same one ``tests/test_graph_grad.py``
and ``tests/test_qat_graph.py`` already check piece by piece -- a correct
backward pass composed with a correct Adam step trains a model -- so the
focus here is the thing this module actually adds on top of those: the lazy
compile-once-then-reuse calling convention, state threaded across independent
calls rather than one fixed-length loop, and :meth:`TrainingLoop.export`.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim import backend, graph_grad

ort = pytest.importorskip("onnxruntime")

_HEADER = '<ir_version: 8, opset_import: ["": 17]>'


def _model(body: str, initializer=()) -> onnx.ModelProto:
    model = parser.parse_model(f"{_HEADER}\n{body}")
    model.graph.initializer.extend(initializer)
    return model


def _f32(array: np.ndarray, name: str) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _linear_model(rows: int = 8, k: int = 3, n: int = 2, seed: int = 0):
    """``loss = mean((x @ w^T - y) ** 2)``, with ``w`` the trained parameter.

    A well-determined linear regression: Adam should drive the loss down
    substantially in a couple hundred steps, which is the correctness signal
    the tests below rely on rather than a closed-form comparison.
    """
    rng = np.random.default_rng(seed)
    model = _model(
        f"""
        agraph (float[{rows},{k}] x, float[{rows},{n}] y) => (float loss)
        {{
            wt = Transpose<perm=[1,0]>(w)
            y_hat = MatMul(x, wt)
            diff = Sub(y_hat, y)
            sq = Mul(diff, diff)
            loss = ReduceMean<keepdims=0>(sq)
        }}
        """,
        initializer=[_f32(rng.standard_normal((n, k)) * 0.1, "w")],
    )
    onnx.checker.check_model(model)

    w_true = rng.standard_normal((n, k)).astype(np.float32)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    y = x @ w_true.T
    return model, x, y


def test_lazy_compile_then_reuse():
    """``compiled`` flips only once the loop is actually called, and the
    step graph/session built on that first call is what every later call
    reuses -- torch.compile's own first-call-traces, later-calls-reuse shape.
    """
    model, x, y = _linear_model()
    loop = onnxsim.compile_training_loop(model, "loss", ("w",))
    assert not loop.compiled

    first_loss = loop({"x": x, "y": y}, lr=1e-2)
    assert loop.compiled
    step, runner = loop._step, loop._runner

    loop({"x": x, "y": y}, lr=1e-2)
    assert loop._step is step
    assert loop._runner is runner
    assert isinstance(first_loss, float)


def test_step_graph_and_initial_state_compile_without_running_a_step():
    """:attr:`TrainingLoop.step_graph`/:attr:`TrainingLoop.initial_state` let
    a caller get at the compiled artifact -- to run it through a different
    runtime entirely, e.g. ``scripts/convertmodel``'s WebGPU CI fixtures --
    without needing to run a step through this loop's own ``__call__``.
    """
    model, x, y = _linear_model()
    loop = onnxsim.compile_training_loop(model, "loss", ("w",))
    assert not loop.compiled

    step = loop.step_graph
    assert loop.compiled
    assert "w" in step.state
    assert step.loss_name == "loss"

    state = loop.initial_state
    assert set(state) == set(step.state)
    np.testing.assert_array_equal(
        state["w"], onnx.numpy_helper.to_array(model.graph.initializer[0])
    )
    # Every non-trained state entry (Adam's moments) starts at zero.
    for name, value in state.items():
        if name != "w":
            assert np.all(value == 0.0)

    # Reading it again after compiling does not run a step or otherwise
    # change anything.
    assert loop.step_graph is step
    np.testing.assert_array_equal(state["w"], loop.initial_state["w"])


def test_state_stays_an_ort_value_between_calls():
    """The trained parameter and the optimizer's own moments never round-trip
    through numpy between steps when onnxruntime is available -- see
    ``onnxsim/compile_training.py``'s own module docstring on why. Reading
    them out (:attr:`TrainingLoop.parameters`) still works exactly as before;
    only the internal storage differs.
    """
    model, x, y = _linear_model()
    loop = onnxsim.compile_training_loop(model, "loss", ("w",))
    loop({"x": x, "y": y}, lr=5e-2)
    assert loop._runner.supports_ort_values()
    for value in loop._state.values():
        assert not isinstance(value, np.ndarray)
        assert hasattr(value, "numpy")  # an OrtValue
    assert isinstance(loop.parameters()["w"], np.ndarray)


def test_ort_value_and_numpy_paths_agree(monkeypatch):
    """``__call__``'s two branches -- the onnxruntime ``OrtValue``/DLPack
    path and the plain-numpy path used when onnxruntime is not installed --
    must compute identical numbers; only how much gets copied between steps
    differs. Forces the fallback via monkeypatch rather than actually
    uninstalling onnxruntime, since this whole test file needs it either way.

    The patch targets ``loop_numpy``'s own ``Runner`` instance, not the
    class: patching the class would flip *every* loop's dispatch, including
    ``loop_ort``'s, onto the numpy branch the moment it is applied -- exactly
    the failure mode this test exists to catch, so it must not fall into it
    itself.
    """
    model, x, y = _linear_model()
    loop_ort = onnxsim.compile_training_loop(model, "loss", ("w",))

    loop_numpy = onnxsim.compile_training_loop(model, "loss", ("w",))
    loop_numpy.step_graph  # compile via the real, OrtValue-capable path first
    monkeypatch.setattr(loop_numpy._runner, "supports_ort_values", lambda: False)
    # ... then convert the state that compile produced back to plain numpy,
    # matching what an actual no-onnxruntime compile would have stored.
    loop_numpy._state = {k: v.numpy() for k, v in loop_numpy._state.items()}

    for _ in range(20):
        loss_ort = loop_ort({"x": x, "y": y}, lr=5e-2)
        loss_numpy = loop_numpy({"x": x, "y": y}, lr=5e-2)
        assert loss_ort == pytest.approx(loss_numpy, rel=1e-5)
    np.testing.assert_allclose(
        loop_ort.parameters()["w"], loop_numpy.parameters()["w"], rtol=1e-5
    )
    assert isinstance(loop_numpy._state["w"], np.ndarray)
    assert not isinstance(loop_ort._state["w"], np.ndarray)


def test_feeds_accept_a_dlpack_capable_array_directly():
    """A feed value need not be a numpy array -- anything implementing
    ``__dlpack__`` (here, a numpy array wrapped so only that protocol is
    exposed, ruling out any other code path recognizing it) is accepted via
    :func:`onnxsim.backend.as_ort_value` with no manual conversion."""

    class _DlpackOnly:
        """Exposes only ``__dlpack__``/``__dlpack_device__`` -- not
        ``__array__`` or the buffer protocol -- so a numpy fallback path
        that didn't actually go through DLPack would fail outright instead
        of quietly working anyway."""

        def __init__(self, array: np.ndarray) -> None:
            self._array = array

        def __dlpack__(self, *args, **kwargs):
            return self._array.__dlpack__(*args, **kwargs)

        def __dlpack_device__(self):
            return self._array.__dlpack_device__()

    model, x, y = _linear_model()
    loop = onnxsim.compile_training_loop(model, "loss", ("w",))
    losses = [
        loop({"x": _DlpackOnly(x), "y": _DlpackOnly(y)}, lr=5e-2) for _ in range(50)
    ]
    assert losses[-1] < 0.5 * losses[0]


def test_training_loop_reduces_loss():
    model, x, y = _linear_model()
    loop = onnxsim.compile_training_loop(model, "loss", ("w",))

    losses = [loop({"x": x, "y": y}, lr=5e-2) for _ in range(300)]
    assert losses[-1] < 0.02 * losses[0]
    # Not just the last step: the loop should have made steady progress, not
    # one lucky step among many bad ones.
    assert np.mean(losses[-20:]) < 0.1 * np.mean(losses[:20])


def test_sgd_momentum_optimizer_also_trains():
    model, x, y = _linear_model()
    loop = onnxsim.compile_training_loop(
        model, "loss", ("w",), optimizer="sgd_momentum"
    )
    losses = [loop({"x": x, "y": y}, lr=5e-2) for _ in range(300)]
    assert losses[-1] < 0.1 * losses[0]


def test_export_reflects_trained_parameters():
    model, x, y = _linear_model()
    loop = onnxsim.compile_training_loop(model, "loss", ("w",))
    for _ in range(50):
        loop({"x": x, "y": y}, lr=5e-2)

    exported = loop.export()
    onnx.checker.check_model(exported)
    (w_init,) = [t for t in exported.graph.initializer if t.name == "w"]
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(w_init), loop.parameters()["w"]
    )
    # And the exported model is a plain, ordinary forward model: running it
    # reproduces the trained w's own prediction.
    out = backend.run_model(exported, {"x": x, "y": y})
    w = loop.parameters()["w"]
    expected = float(np.mean((x @ w.T - y) ** 2))
    assert out["loss"] == pytest.approx(expected, rel=1e-4)


def test_export_before_any_call_is_the_original_model():
    model, _, _ = _linear_model()
    loop = onnxsim.compile_training_loop(model, "loss", ("w",))
    exported = loop.export()
    (w_before,) = [t for t in model.graph.initializer if t.name == "w"]
    (w_after,) = [t for t in exported.graph.initializer if t.name == "w"]
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(w_before), onnx.numpy_helper.to_array(w_after)
    )


def test_state_persists_across_independent_calls():
    """Parameters keep moving from where the previous call left them, not
    from the original model's initializer -- the state-threading contract a
    real training loop needs, one call at a time rather than one fixed-length
    ``run_step_graph`` loop."""
    model, x, y = _linear_model()
    loop = onnxsim.compile_training_loop(model, "loss", ("w",))

    loop({"x": x, "y": y}, lr=5e-2)
    w_after_one = loop.parameters()["w"].copy()
    loop({"x": x, "y": y}, lr=5e-2)
    w_after_two = loop.parameters()["w"]

    assert not np.allclose(w_after_one, w_after_two)


def test_unknown_optimizer_is_refused_immediately():
    model, _, _ = _linear_model()
    with pytest.raises(ValueError, match="optimizer"):
        onnxsim.compile_training_loop(model, "loss", ("w",), optimizer="rmsprop")


def test_missing_parameter_is_refused():
    model, x, y = _linear_model()
    loop = onnxsim.compile_training_loop(model, "loss", ("not_a_param",))
    with pytest.raises(ValueError, match="not_a_param"):
        loop({"x": x, "y": y}, lr=1e-2)


def test_non_scalar_loss_is_refused():
    model = _model(
        """
        agraph (float[4,3] x) => (float[4,3] y)
        {
            y = Mul(x, w)
        }
        """,
        initializer=[_f32(np.ones(3), "w")],
    )
    loop = onnxsim.compile_training_loop(model, "y", ("w",))
    with pytest.raises(ValueError, match="scalar"):
        loop({"x": np.zeros((4, 3), dtype=np.float32)}, lr=1e-2)


def test_unsupported_op_is_refused_loudly():
    """Softplus has no rule in :mod:`onnxsim.graph_grad`
    (:data:`onnxsim.graph_grad.SUPPORTED_OPS`), so compiling a model that
    uses it must fail loudly rather than silently skip differentiating it --
    the same discipline :mod:`onnxsim.graph_grad`'s own docstring describes.
    """
    model = _model(
        """
        agraph (float[4,3] x) => (float loss)
        {
            scaled = Mul(x, w)
            act = Softplus(scaled)
            loss = ReduceMean<keepdims=0>(act)
        }
        """,
        initializer=[_f32(np.ones(3), "w")],
    )
    loop = onnxsim.compile_training_loop(model, "loss", ("w",))
    with pytest.raises(graph_grad.UnsupportedOpError):
        loop({"x": np.zeros((4, 3), dtype=np.float32)}, lr=1e-2)
