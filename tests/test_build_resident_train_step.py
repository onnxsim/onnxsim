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

    feeds = {
        "x": x,
        "y": y,
        "lr": np.array([0.0], np.float32),
        "grad_seed": np.array([1.0], np.float32),
        "cw": cw0,
        "gw": gw0,
    }
    out_names = [o.name for o in step_model.graph.output]
    outs = dict(zip(out_names, _run(step_model, feeds, out_names)))
    assert np.array_equal(outs[state["cw"]], cw0)
    assert np.array_equal(outs[state["gw"]], gw0)


def test_grad_seed_is_a_runtime_input_that_linearly_scales_the_gradient():
    """`grad_seed` used to be `b.const(1.0)` -- baked in at build time, so no
    per-step loss-scaling controller could ever vary it (see
    `docs/axera-on-device-training-handoff.md`'s "The ceiling: the gradient
    dies" section for why that mattered: the whole point of loss scaling is
    a runtime-varying multiplier). Now a real scalar graph input like `lr`.

    `build_backward`'s gradient is linear in its seed by construction (the
    seed is literally the initial dL/d(loss) the chain rule multiplies
    through), so `grad_seed=S` must return exactly `S` times the
    `grad_seed=1` gradient -- extracted via the same `w - w_next` at `lr=1`
    trick `test_set_batch_gradient_is_the_mean_of_per_sample_gradients` uses.
    This is what makes a non-1.0 seed value meaningful *before* it ever
    reaches Pulsar2's calibration/`layer_configs` machinery: get this wrong
    on host and no amount of on-device `FP32` layer config can save it.
    """
    forward = _forward_model()
    with_loss = brts.add_mse_loss(forward, "logits", num_classes=10)
    step_model, state = brts.build_resident_step(with_loss, params=["cw", "gw"])
    onnx.checker.check_model(step_model)

    seed_input = next(i for i in step_model.graph.input if i.name == "grad_seed")
    assert [d.dim_value for d in seed_input.type.tensor_type.shape.dim] == [1]

    initializers = {t.name: t for t in forward.graph.initializer}
    cw0 = onnx.numpy_helper.to_array(initializers["cw"])
    gw0 = onnx.numpy_helper.to_array(initializers["gw"])

    rng = np.random.default_rng(4)
    x = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
    y = rng.standard_normal((1, 10)).astype(np.float32)
    out_names = [o.name for o in step_model.graph.output]

    def grad_at(seed):
        feeds = {
            "x": x,
            "y": y,
            "lr": np.array([1.0], np.float32),
            "grad_seed": np.array([seed], np.float32),
            "cw": cw0,
            "gw": gw0,
        }
        outs = dict(zip(out_names, _run(step_model, feeds, out_names)))
        return cw0 - outs[state["cw"]], gw0 - outs[state["gw"]]

    cw_grad1, gw_grad1 = grad_at(1.0)
    for scale in (1000.0, 2.0**16, 2.0**20):
        cw_grad_s, gw_grad_s = grad_at(scale)
        assert np.allclose(cw_grad_s, cw_grad1 * scale, rtol=1e-4), scale
        assert np.allclose(gw_grad_s, gw_grad1 * scale, rtol=1e-4), scale


def test_set_batch_gradient_is_the_mean_of_per_sample_gradients():
    """A batch-N step's gradient must equal the average of N independent
    batch-1 steps' gradients -- the same relationship
    `docs/axera-on-device-training-handoff.md`'s "Batching" section measured
    on real AX650N hardware for resnet18, checked here on host for the same
    reason every other in-graph-update property is: no docker, no device
    needed to catch a batch-handling bug in `set_batch`/`add_mse_loss`
    before it reaches the compiler.

    Extracts each gradient as `w - w_next` at `lr=1` (the same trick
    `test_state_output_is_sgd_update_of_the_input` relies on via `lr=0`, just
    solved for the gradient instead of asserting a pass-through)."""
    base = _forward_model()
    initializers = {t.name: t for t in base.graph.initializer}
    cw0 = onnx.numpy_helper.to_array(initializers["cw"])
    gw0 = onnx.numpy_helper.to_array(initializers["gw"])

    rng = np.random.default_rng(3)
    n = 4
    xs = rng.standard_normal((n, 1, 4, 4)).astype(np.float32)
    ys = rng.standard_normal((n, 10)).astype(np.float32)
    lr1 = np.array([1.0], np.float32)

    model1 = brts.add_mse_loss(_forward_model(), "logits", num_classes=10)
    step1, state1 = brts.build_resident_step(model1, params=["cw", "gw"])
    onnx.checker.check_model(step1)
    out_names1 = [o.name for o in step1.graph.output]

    per_sample_grad = {"cw": [], "gw": []}
    for i in range(n):
        feeds = {
            "x": xs[i : i + 1],
            "y": ys[i : i + 1],
            "lr": lr1,
            "grad_seed": np.array([1.0], np.float32),
            "cw": cw0,
            "gw": gw0,
        }
        outs = dict(zip(out_names1, _run(step1, feeds, out_names1)))
        per_sample_grad["cw"].append(cw0 - outs[state1["cw"]])
        per_sample_grad["gw"].append(gw0 - outs[state1["gw"]])
    grad_avg = {p: np.mean(np.stack(g), axis=0) for p, g in per_sample_grad.items()}

    forward_n = brts.set_batch(_forward_model(), n)
    model_n = brts.add_mse_loss(forward_n, "logits", num_classes=10)
    # `set_batch` must produce the same batch dimension `add_mse_loss` reads
    # `y`'s shape from -- checked directly, not just implied by the numeric
    # match below.
    y_input = next(i for i in model_n.graph.input if i.name == "y")
    assert [d.dim_value for d in y_input.type.tensor_type.shape.dim] == [n, 10]

    step_n, state_n = brts.build_resident_step(model_n, params=["cw", "gw"])
    onnx.checker.check_model(step_n)
    out_names_n = [o.name for o in step_n.graph.output]
    feeds_n = {
        "x": xs,
        "y": ys,
        "lr": lr1,
        "grad_seed": np.array([1.0], np.float32),
        "cw": cw0,
        "gw": gw0,
    }
    outs_n = dict(zip(out_names_n, _run(step_n, feeds_n, out_names_n)))

    for p in ("cw", "gw"):
        grad_batch = cw0 - outs_n[state_n[p]] if p == "cw" else gw0 - outs_n[state_n[p]]
        assert np.allclose(grad_avg[p], grad_batch, atol=1e-5), p

    # batch-N's step graph is the same shape/structure as batch-1's -- no
    # extra nodes from taking a different code path for N != 1.
    assert len(step_n.graph.node) == len(step1.graph.node)


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
    feeds = {
        "x": x,
        "y": y,
        "lr": np.array([1.0], np.float32),
        "grad_seed": np.array([1.0], np.float32),
        **w0,
    }
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


def test_linearize_trainable_convs_matches_conv_and_drops_the_weight_transpose():
    """`_linearize_trainable_convs`'s whole reason to exist:
    `legalize.act_weight_conv_to_matmul` (run later, on the *step* graph)
    legalizes a live-weight `Conv` by transposing its weight into matmul
    layout -- measured on real AX650N hardware at 89.6% of the step's
    `AxTranspose` cost, recomputed from scratch every step for a weight that
    is now resident state and barely changes step to step
    (`docs/axera-on-device-training-handoff.md`). This checks the
    replacement directly: same numbers as `Conv` (including the two
    resnet18 geometries this actually has to handle -- a strided, biased 1x1
    downsample and a padded, stride-1, biased 3x3), and no `Transpose` node
    reads the weight at all.
    """
    rng = np.random.default_rng(4)
    for cin, cout, k, size, stride, pad, has_bias in (
        (16, 32, 1, 8, 2, 0, True),  # resnet18's downsample conv
        (16, 16, 3, 8, 1, 1, True),  # resnet18's 3x3 conv, post-BN-fold bias
        (16, 16, 3, 8, 1, 1, False),  # no bias, the pre-fold shape
    ):
        out = (size + 2 * pad - k) // stride + 1
        cw = (rng.standard_normal((cout, cin, k, k)) * 0.2).astype(np.float32)
        cb = (rng.standard_normal(cout) * 0.1).astype(np.float32) if has_bias else None
        model = parser.parse_model(
            f"""
            <
              ir_version: 10,
              opset_import: ["": 17]
            >
            g (float[1,{cin},{size},{size}] x) => (float[1,{cout},{out},{out}] y)
            {{
              y = Conv<kernel_shape=[{k},{k}], strides=[{stride},{stride}],
                       pads=[{pad},{pad},{pad},{pad}]>(x, cw{", cb" if has_bias else ""})
            }}
            """
        )
        inits = [_f32(cw, "cw")]
        if has_bias:
            inits.append(_f32(cb, "cb"))
        model.graph.initializer.extend(inits)

        x = rng.standard_normal((1, cin, size, size)).astype(np.float32)
        (ref,) = _run(model, {"x": x}, ["y"])

        linearized = brts._linearize_trainable_convs(
            onnx.shape_inference.infer_shapes(model), ["cw"]
        )
        onnx.checker.check_model(linearized)
        assert not any(
            n.op_type == "Transpose" and "cw" in n.input for n in linearized.graph.node
        )
        (got,) = _run(linearized, {"x": x}, ["y"])
        assert got.shape == ref.shape
        assert np.allclose(got, ref, atol=1e-4), (cin, cout, k, stride, pad, has_bias)


def _bottleneck_model():
    """`x -> 1x1 -> Relu -> 3x3 -> Relu -> 1x1 -> Flatten -> Gemm -> logits`:
    a resnet50-style bottleneck block's conv shape (channel-reduce 1x1,
    spatial 3x3, channel-expand 1x1, all three trainable and chained), rather
    than the single isolated conv `test_linearize_trainable_convs_...` above
    already covers. Exercises the same `_linearize_trainable_convs` path this
    module's docstring describes, but for *multiple* trainable convs of
    different kernel sizes feeding each other -- the shape a real resnet50
    `layer4.2` block has and resnet18's basic blocks do not. See
    `docs/axera-on-device-training-handoff.md`'s "A different architecture:
    resnet50, first compile" section, which verified this same shape
    combination on the real model (cosine 0.99994 against finite
    differences); this is the from-scratch, hardware-free regression test for
    it.
    """
    rng = np.random.default_rng(5)
    w1 = (rng.standard_normal((4, 8, 1, 1)) * 0.2).astype(np.float32)  # reduce
    w2 = (rng.standard_normal((4, 4, 3, 3)) * 0.2).astype(np.float32)  # spatial
    w3 = (rng.standard_normal((8, 4, 1, 1)) * 0.2).astype(np.float32)  # expand
    gw = (rng.standard_normal((5, 8 * 4 * 4)) * 0.1).astype(np.float32)
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,8,4,4] x) => (float[1,5] logits)
        {
          h1 = Conv<kernel_shape=[1,1]>(x, w1)
          r1 = Relu(h1)
          h2 = Conv<kernel_shape=[3,3], pads=[1,1,1,1]>(r1, w2)
          r2 = Relu(h2)
          h3 = Conv<kernel_shape=[1,1]>(r2, w3)
          r3 = Relu(h3)
          f = Flatten<axis=1>(r3)
          logits = Gemm<transB=1>(f, gw)
        }
        """
    )
    model.graph.initializer.extend(
        [_f32(w1, "w1"), _f32(w2, "w2"), _f32(w3, "w3"), _f32(gw, "gw")]
    )
    return model


def test_bottleneck_block_gradient_matches_finite_differences():
    """The full `build_resident_step` pipeline (not just
    `_linearize_trainable_convs` in isolation) on a resnet50-bottleneck-
    shaped chain of trainable convs, all promoted to state at once -- the
    combination the isolated-conv test above doesn't exercise."""
    forward = _bottleneck_model()
    with_loss = brts.add_mse_loss(forward, "logits", num_classes=5)
    step_model, state = brts.build_resident_step(with_loss, params=["w1", "w2", "w3"])

    initializers = {t.name: t for t in forward.graph.initializer}
    w0 = {p: onnx.numpy_helper.to_array(initializers[p]) for p in state}

    rng = np.random.default_rng(6)
    x = rng.standard_normal((1, 8, 4, 4)).astype(np.float32)
    y = rng.standard_normal((1, 5)).astype(np.float32)

    out_names = [o.name for o in step_model.graph.output]
    feeds = {
        "x": x,
        "y": y,
        "lr": np.array([1.0], np.float32),
        "grad_seed": np.array([1.0], np.float32),
        **w0,
    }
    outs = dict(zip(out_names, _run(step_model, feeds, out_names)))
    grads = {p: (w0[p] - outs[state[p]]).astype(np.float64) for p in state}

    def loss_at(weights):
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
        w_plus, w_minus = dict(w0), dict(w0)
        w_plus[p] = (w0[p].astype(np.float64) + eps * d).astype(np.float32)
        w_minus[p] = (w0[p].astype(np.float64) - eps * d).astype(np.float32)
        fd = (loss_at(w_plus) - loss_at(w_minus)) / (2 * eps)
        predicted = float((grads[p] * d).sum())
        assert predicted == pytest.approx(fd, rel=0.05, abs=1e-6), p
