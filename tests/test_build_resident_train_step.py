"""Tests for ``scripts/axera/build_resident_train_step.py`` -- the in-graph
SGD update that lets a resident runner keep a training step's trainable
weights entirely device-side (see ``docs/axera-on-device-training-
handoff.md``'s "Weights resident with in-graph updates" section, and
``scripts/axera/tools/resident_runner.c``, which is what actually binds a
state output back to its own input's device buffer between steps -- neither
needs a real AX650N to check that the *graph* computes what it should).

Everything here runs on the CPU reference/onnxruntime; no Docker, no device.
"""

import os
import sys

import numpy as np
import onnx
import pytest
from onnx import parser

ort = pytest.importorskip("onnxruntime")

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import build_resident_train_step as brts  # noqa: E402


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _forward_model():
    """`x -> Conv -> Relu -> Flatten -> Gemm -> logits`: small enough to run
    instantly, but exercises both a trainable `Conv` weight and a trainable
    `Gemm` weight, plus the `Flatten` this module's own docstring says must
    be legalized to `Reshape` *before* `graph_grad.build_backward` ever sees
    it (`onnxsim.graph_grad` has no gradient rule for `Flatten` itself).
    """
    rng = np.random.default_rng(0)
    cw = rng.standard_normal((2, 1, 3, 3)).astype(np.float32) * 0.3
    gw = rng.standard_normal((10, 32)).astype(np.float32) * 0.2
    gb = rng.standard_normal((10,)).astype(np.float32) * 0.1
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,1,4,4] x) => (float[1,10] logits)
        {
          h = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(x, cw)
          r = Relu(h)
          f = Flatten<axis=1>(r)
          logits = Gemm<transB=1>(f, gw, gb)
        }
        """
    )
    model.graph.initializer.extend([_f32(cw, "cw"), _f32(gw, "gw"), _f32(gb, "gb")])
    return model


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(output_names, feeds)


def test_add_mse_loss_matches_manual_computation():
    forward = _forward_model()
    with_loss = brts.add_mse_loss(forward, "logits", num_classes=10)
    onnx.checker.check_model(with_loss)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
    y = rng.standard_normal((1, 10)).astype(np.float32)

    (logits,) = _run(forward, {"x": x}, ["logits"])
    (loss,) = _run(with_loss, {"x": x, "y": y}, ["loss"])
    assert loss.shape == ()
    assert np.allclose(loss, np.mean((logits - y) ** 2), atol=1e-5)


def test_state_output_is_sgd_update_of_the_input():
    """`w_next` for each trained param must be exactly `w - lr * grad`, with
    `lr=0` a pure pass-through -- the cheapest end-to-end check that the
    in-graph `Mul`/`Sub` update is wired to the right tensors."""
    forward = _forward_model()
    with_loss = brts.add_mse_loss(forward, "logits", num_classes=10)
    step_model, state = brts.build_resident_step(with_loss, params=["cw", "gw"])
    onnx.checker.check_model(step_model)

    assert set(state) == {"cw", "gw"}
    # lr was reshaped from rank 0 to rank 1 -- see this module's docstring.
    lr_input = next(i for i in step_model.graph.input if i.name == "lr")
    assert [d.dim_value for d in lr_input.type.tensor_type.shape.dim] == [1]

    initializers = {t.name: t for t in forward.graph.initializer}
    cw0 = onnx.numpy_helper.to_array(initializers["cw"])
    gw0 = onnx.numpy_helper.to_array(initializers["gw"])

    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
    y = rng.standard_normal((1, 10)).astype(np.float32)

    feeds = {"x": x, "y": y, "lr": np.array([0.0], np.float32), "cw": cw0, "gw": gw0}
    out_names = [o.name for o in step_model.graph.output]
    outs = dict(zip(out_names, _run(step_model, feeds, out_names)))
    assert np.array_equal(outs[state["cw"]], cw0)
    assert np.array_equal(outs[state["gw"]], gw0)


def test_in_graph_gradient_matches_finite_differences():
    """The gradient implied by the in-graph update (`grad = (w -
    w_next) / lr`) must agree with a numeric directional derivative of the
    *original* forward+loss graph -- the same kind of check
    `tests/test_axera_training_legalize.py` runs for each legalization rule,
    here covering the whole build_backward + in-graph-update pipeline at
    once."""
    forward = _forward_model()
    with_loss = brts.add_mse_loss(forward, "logits", num_classes=10)
    step_model, state = brts.build_resident_step(with_loss, params=["cw", "gw"])

    initializers = {t.name: t for t in forward.graph.initializer}
    w0 = {p: onnx.numpy_helper.to_array(initializers[p]) for p in state}

    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
    y = rng.standard_normal((1, 10)).astype(np.float32)

    out_names = [o.name for o in step_model.graph.output]
    feeds = {"x": x, "y": y, "lr": np.array([1.0], np.float32), **w0}
    outs = dict(zip(out_names, _run(step_model, feeds, out_names)))
    grads = {p: (w0[p] - outs[state[p]]).astype(np.float64) for p in state}

    def loss_at(weights):
        # `with_loss` still carries each trained weight as a plain
        # initializer (only build_resident_step's own internal builder
        # promotes them to graph inputs) -- so overriding one for this probe
        # means replacing its initializer, not feeding it as a run() input.
        probe = onnx.ModelProto()
        probe.CopyFrom(with_loss)
        for init in probe.graph.initializer:
            if init.name in weights:
                init.CopyFrom(
                    onnx.numpy_helper.from_array(weights[init.name], init.name)
                )
        (loss,) = _run(probe, {"x": x, "y": y}, ["loss"])
        return float(loss)

    for p in state:
        d = rng.standard_normal(w0[p].shape).astype(np.float64)
        eps = 1e-2 / (np.linalg.norm(d) + 1e-12)
        w_plus = dict(w0)
        w_plus[p] = (w0[p].astype(np.float64) + eps * d).astype(np.float32)
        w_minus = dict(w0)
        w_minus[p] = (w0[p].astype(np.float64) - eps * d).astype(np.float32)
        fd = (loss_at(w_plus) - loss_at(w_minus)) / (2 * eps)
        predicted = float((grads[p] * d).sum())
        assert predicted == pytest.approx(fd, rel=0.05, abs=1e-6), p
