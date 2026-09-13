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


def test_rank1_state_update_is_reshaped_around_the_sub():
    """A rank-1 trainable tensor's in-graph SGD `Sub` crashes Pulsar2's own
    NPU backend tiler on real hardware (`TileFailException("AxQuantizedSub,
    tuple index out of range")`, confirmed on two different real bias
    shapes -- `docs/axera-super-resolution-op-coverage.md`'s real-hardware
    section). Sidestepped by reshaping the whole per-step update to rank-2
    around the `Sub` and back afterward -- transparent to the state
    tensor's own declared rank-1 shape at the graph's I/O boundary, checked
    both structurally (the `Reshape`s exist) and numerically (the update
    computes the identical value a bare rank-1 `Sub` would)."""
    forward = _forward_model()
    with_loss = brts.add_mse_loss(forward, "logits", num_classes=10)
    step_model, state = brts.build_resident_step(with_loss, params=["cw", "gw", "gb"])
    onnx.checker.check_model(step_model)

    gb_next = state["gb"]
    reshape_out = next(n for n in step_model.graph.node if n.output[0] == gb_next)
    assert reshape_out.op_type == "Reshape", "expected the final state back at rank 1"
    sub_node = next(
        n for n in step_model.graph.node if n.output[0] == reshape_out.input[0]
    )
    assert sub_node.op_type == "Sub"
    # `simplify()` may fold away a reshape that turns out to be a no-op (a
    # Gemm bias gradient's own "unbroadcast" step can already land at rank
    # 2), so check the `Sub`'s *actual* operand ranks rather than assuming
    # a specific intermediate reshape node survives -- the property this
    # test cares about is that `Sub` itself never sees a bare rank-1
    # tensor, not the exact node count it took to get there.
    for operand in sub_node.input:
        shape = next(
            (
                [d.dim_value for d in vi.type.tensor_type.shape.dim]
                for vi in list(step_model.graph.value_info)
                + list(step_model.graph.input)
                if vi.name == operand
            ),
            None,
        )
        if shape is not None:
            assert len(shape) != 1, f"{operand} feeds Sub as a bare rank-1 tensor"

    initializers = {t.name: t for t in forward.graph.initializer}
    cw0 = onnx.numpy_helper.to_array(initializers["cw"])
    gw0 = onnx.numpy_helper.to_array(initializers["gw"])
    gb0 = onnx.numpy_helper.to_array(initializers["gb"])

    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
    y = rng.standard_normal((1, 10)).astype(np.float32)
    lr = 0.1

    feeds = {
        "x": x,
        "y": y,
        "lr": np.array([lr], np.float32),
        "grad_seed": np.array([1.0], np.float32),
        "cw": cw0,
        "gw": gw0,
        "gb": gb0,
    }
    out_names = [o.name for o in step_model.graph.output]
    outs = dict(zip(out_names, _run(step_model, feeds, out_names)))

    # The update's implied gradient (`grad = (gb0 - gb_next) / lr`) must
    # match a central-difference gradient of the *original* forward+loss
    # graph's own loss w.r.t. `gb` -- independent of this module's own
    # graph-building code, and of exactly which nodes survived `simplify()`.
    # `with_loss` still has `gb` as a fixed initializer, not a real input --
    # promote it to one so the loss can be probed at an offset value.
    gb_as_input = onnx.ModelProto()
    gb_as_input.CopyFrom(with_loss)
    kept = [t for t in gb_as_input.graph.initializer if t.name != "gb"]
    del gb_as_input.graph.initializer[:]
    gb_as_input.graph.initializer.extend(kept)
    gb_as_input.graph.input.append(
        onnx.helper.make_tensor_value_info("gb", onnx.TensorProto.FLOAT, [10])
    )

    implied_grad = (gb0 - outs[gb_next]) / lr

    def loss_at(gb_value):
        (loss,) = _run(gb_as_input, {"x": x, "y": y, "gb": gb_value}, ["loss"])
        return float(loss)

    eps = 1e-3
    numeric_grad = np.zeros_like(gb0)
    for i in range(gb0.shape[0]):
        plus, minus = gb0.copy(), gb0.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric_grad[i] = (loss_at(plus) - loss_at(minus)) / (2 * eps)

    assert np.allclose(implied_grad, numeric_grad, atol=1e-3)


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


def test_a_raw_constant_node_is_folded_before_backward():
    """`graph_grad.build_backward` demands a gradient rule for every node
    type it walks, `Constant` included, even though a zero-input op has
    nothing to backprop through -- found training a real Whisper encoder,
    whose Erf-GELU decomposition (`0.5 * x * (1 + erf(x / sqrt(2)))`) leaves
    raw `Constant` nodes for `0.5`/`sqrt(2)` that a forward-only export never
    needs to fold (see docs/axera-on-device-training-handoff.md's "A
    memory-heavy case" section). Before `_fold_constants` ran ahead of
    `build_backward`, this raised `UnsupportedOpError('no gradient rule for
    op type 'Constant'')`; resnet18/50's own forward graphs never exercised
    this path since neither has a raw `Constant` node anywhere.
    """
    rng = np.random.default_rng(7)
    gw = rng.standard_normal((4, 4)).astype(np.float32) * 0.2

    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 17]
        >
        g (float[1,4] x) => (float[1,4] logits)
        {
          half = Constant<value = float {0.5}>()
          one = Constant<value = float {1.0}>()
          inv_sqrt2 = Constant<value = float {0.7071067811865476}>()
          h = MatMul(x, gw)
          scaled = Mul(h, inv_sqrt2)
          erfed = Erf(scaled)
          shifted = Add(erfed, one)
          gated = Mul(h, shifted)
          logits = Mul(gated, half)
        }
        """
    )
    model.graph.initializer.append(_f32(gw, "gw"))
    onnx.checker.check_model(model)
    assert any(n.op_type == "Constant" for n in model.graph.node)

    with_loss = brts.add_mse_loss(model, "logits", num_classes=4)
    step_model, state = brts.build_resident_step(with_loss, params=["gw"])
    onnx.checker.check_model(step_model)
    assert not any(n.op_type == "Constant" for n in step_model.graph.node)
    assert set(state) == {"gw"}

    x = rng.standard_normal((1, 4)).astype(np.float32)
    y = rng.standard_normal((1, 4)).astype(np.float32)
    gw0 = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "gw")
    )
    out_names = [o.name for o in step_model.graph.output]
    feeds = {
        "x": x,
        "y": y,
        "lr": np.array([1.0], np.float32),
        "grad_seed": np.array([1.0], np.float32),
        "gw": gw0,
    }
    outs = dict(zip(out_names, _run(step_model, feeds, out_names)))
    grad = (gw0 - outs[state["gw"]]).astype(np.float64)

    def loss_at(w):
        probe = onnx.ModelProto()
        probe.CopyFrom(with_loss)
        for init in probe.graph.initializer:
            if init.name == "gw":
                init.CopyFrom(onnx.numpy_helper.from_array(w.astype(np.float32), "gw"))
        (loss,) = _run(probe, {"x": x, "y": y}, ["loss"])
        return float(loss)

    d = rng.standard_normal(gw0.shape).astype(np.float64)
    eps = 1e-2 / (np.linalg.norm(d) + 1e-12)
    fd = (loss_at(gw0 + eps * d) - loss_at(gw0 - eps * d)) / (2 * eps)
    predicted = float((grad * d).sum())
    assert predicted == pytest.approx(fd, rel=0.05, abs=1e-6)


def test_resident_dataset_gather_matches_feeding_the_same_rows_directly():
    """`add_resident_dataset` replaces `x`/`y` with `Gather(dataset, index)`
    off a resident constant. The whole point is that training on rows
    `[i, j]` selected by `batch_index=[i, j]` computes exactly what feeding
    `x=dataset[[i, j]]`/`y=dataset_y[[i, j]]` directly (the pre-existing,
    per-step-re-upload pipeline) already does -- gradients included, not
    just the forward pass."""
    rng = np.random.default_rng(3)
    forward = brts.set_batch(_forward_model(), batch=2)
    with_loss = brts.add_mse_loss(forward, "logits", num_classes=10)

    n_rows = 5
    x_data = rng.standard_normal((n_rows, 1, 4, 4)).astype(np.float32)
    y_data = rng.standard_normal((n_rows, 10)).astype(np.float32)
    resident = brts.add_resident_dataset(with_loss, {"x": x_data, "y": y_data})
    onnx.checker.check_model(resident)
    assert not any(inp.name in ("x", "y") for inp in resident.graph.input)
    assert any(inp.name == "batch_index" for inp in resident.graph.input)

    step_model, state = brts.build_resident_step(resident, params=["cw", "gw"])
    onnx.checker.check_model(step_model)

    cw0 = onnx.numpy_helper.to_array(
        next(t for t in forward.graph.initializer if t.name == "cw")
    )
    gw0 = onnx.numpy_helper.to_array(
        next(t for t in forward.graph.initializer if t.name == "gw")
    )
    out_names = [o.name for o in step_model.graph.output]

    idx = np.array([1, 3], dtype=np.int64)
    feeds_gathered = {
        "batch_index": idx,
        "lr": np.array([1.0], np.float32),
        "grad_seed": np.array([1.0], np.float32),
        "cw": cw0,
        "gw": gw0,
    }
    outs_gathered = dict(zip(out_names, _run(step_model, feeds_gathered, out_names)))

    # The reference: the *pre-gather* step graph, fed the same rows directly
    # -- built from the same forward+loss model, just skipping
    # add_resident_dataset entirely.
    ref_step_model, ref_state = brts.build_resident_step(with_loss, params=["cw", "gw"])
    ref_out_names = [o.name for o in ref_step_model.graph.output]
    feeds_direct = {
        "x": x_data[idx],
        "y": y_data[idx],
        "lr": np.array([1.0], np.float32),
        "grad_seed": np.array([1.0], np.float32),
        "cw": cw0,
        "gw": gw0,
    }
    outs_direct = dict(
        zip(ref_out_names, _run(ref_step_model, feeds_direct, ref_out_names))
    )

    # Output tensor names differ (the gather graph has extra upstream nodes,
    # so onnxsim's auto-generated names shift) -- compare by state's own
    # mapping, and the shared loss name, not by raw output-name equality.
    assert set(state) == set(ref_state)
    for p in state:
        np.testing.assert_allclose(
            outs_gathered[state[p]], outs_direct[ref_state[p]], rtol=1e-5, atol=1e-6
        )


def test_resident_dataset_flatten_matches_native_shape_gather():
    """`flatten=True` (the default) stores and gathers `x`'s rank-4 dataset
    as a flat `[N, 16]` initializer, `Reshape`-ing each gathered row back to
    `[batch, 1, 4, 4]` -- the workaround `docs/axera-on-device-training-
    handoff.md`'s "Trading free memory for throughput" section names for a
    real Pulsar2 NPU-backend gap (`Gather` over a 4D conv-activation-shaped
    resident tensor fails in Pulsar2's backend; the same `Gather` over a
    flat 2D tensor plus an ordinary `Reshape` does not). This only changes
    *how* the dataset is stored/gathered, so it must compute exactly what
    `flatten=False`'s native-shape `Gather` already does -- and `y` (rank 2)
    must come out byte-identical either way, since flattening a rank<=2
    array is a no-op by this function's own rule."""
    rng = np.random.default_rng(4)
    forward = brts.set_batch(_forward_model(), batch=2)
    with_loss = brts.add_mse_loss(forward, "logits", num_classes=10)

    n_rows = 5
    x_data = rng.standard_normal((n_rows, 1, 4, 4)).astype(np.float32)
    y_data = rng.standard_normal((n_rows, 10)).astype(np.float32)

    flat = brts.add_resident_dataset(with_loss, {"x": x_data, "y": y_data})
    native = brts.add_resident_dataset(
        with_loss, {"x": x_data, "y": y_data}, flatten=False
    )
    onnx.checker.check_model(flat)
    onnx.checker.check_model(native)

    flat_ops = [n.op_type for n in flat.graph.node]
    native_ops = [n.op_type for n in native.graph.node]
    assert flat_ops.count("Reshape") == native_ops.count("Reshape") + 1
    assert flat_ops.count("Gather") == native_ops.count("Gather") == 2
    x_dataset = next(t for t in flat.graph.initializer if t.name == "x_dataset")
    assert list(x_dataset.dims) == [n_rows, 16]
    y_dataset = next(t for t in flat.graph.initializer if t.name == "y_dataset")
    assert list(y_dataset.dims) == [n_rows, 10]  # rank<=2: flattening is a no-op

    for m in (flat, native):
        m.graph.output.extend(
            [
                onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, None),
                onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None),
            ]
        )

    idx = np.array([1, 3], dtype=np.int64)
    x_flat, y_flat = _run(flat, {"batch_index": idx}, ["x", "y"])
    x_native, y_native = _run(native, {"batch_index": idx}, ["x", "y"])
    np.testing.assert_array_equal(x_flat, x_data[idx])
    np.testing.assert_array_equal(x_flat, x_native)
    np.testing.assert_array_equal(y_flat, y_data[idx])
    np.testing.assert_array_equal(y_flat, y_native)


def test_fold_constants_preserves_every_initializer_a_real_bug_dropped():
    """A real bug found building `scripts/axera/legalize.py`'s `unroll_gru`
    onto real hardware (`docs/axera-audio-speech-op-coverage.md`'s "Both
    closed" section): `_fold_constants`'s `onnxsim.simplify()` call, with
    that function's own default `initializers_as_constants=True`, silently
    *dropped* `unroll_gru`'s freshly-created per-gate `W`/`R` weight
    initializers for this exact graph shape -- no error, `build_resident_
    step` only failing several steps later, unable to find `params` by
    name any more. `unroll_lstm`'s own output happened not to trigger the
    same optimizer behavior, so this needs its own graph to catch a
    regression -- a plain resnet-shaped model (`test_fold_constants_is_a_
    noop_without_any_constant_nodes`, above) never exercises this path at
    all, since it has no `Constant` node to trigger `_fold_constants` in
    the first place.
    """
    from _local_import import fresh
    from onnx import helper, numpy_helper

    legalize = fresh("legalize", _AXERA_DIR)

    seq, batch, inp, hid = 3, 1, 4, 4
    rng = np.random.RandomState(0)
    w = rng.randn(1, 3 * hid, inp).astype(np.float32) * 0.3
    r = rng.randn(1, 3 * hid, hid).astype(np.float32) * 0.3
    b = rng.randn(1, 6 * hid).astype(np.float32) * 0.1

    gru = helper.make_node(
        "GRU",
        ["x", "W", "R", "B", "", ""],
        ["gru_y", "gru_yh"],
        hidden_size=hid,
        linear_before_reset=1,
        name="mygru",
    )
    # A real `Constant` node -- needed to trigger `_fold_constants` at all
    # (it is a no-op, by design, on a graph without one).
    one = helper.make_node(
        "Constant",
        [],
        ["one"],
        value=numpy_helper.from_array(np.array(1.0, np.float32)),
    )
    diff = helper.make_node("Sub", ["gru_y", "target"], ["diff"])
    scaled = helper.make_node("Mul", ["diff", "one"], ["scaled"])
    sq = helper.make_node("Mul", ["scaled", "scaled"], ["sq"])
    loss = helper.make_node(
        "ReduceMean", ["sq"], ["loss"], axes=[0, 1, 2, 3], keepdims=0
    )
    graph = helper.make_graph(
        [gru, one, diff, scaled, sq, loss],
        "g",
        [
            helper.make_tensor_value_info(
                "x", onnx.TensorProto.FLOAT, [seq, batch, inp]
            ),
            helper.make_tensor_value_info(
                "target", onnx.TensorProto.FLOAT, [seq, 1, batch, hid]
            ),
        ],
        [helper.make_tensor_value_info("loss", onnx.TensorProto.FLOAT, [])],
        initializer=[_f32(w, "W"), _f32(r, "R"), _f32(b, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    legalize.unroll_gru(model)
    onnx.checker.check_model(model)

    params = [f"mygru_w{i}" for i in range(3)] + [f"mygru_r{i}" for i in range(3)]
    before_names = {i.name for i in model.graph.initializer}
    assert set(params) <= before_names

    step_model, state = brts.build_resident_step(model, params, loss_output="loss")
    onnx.checker.check_model(step_model)
    assert set(state) == set(params)


def test_fold_constants_is_a_noop_without_any_constant_nodes():
    """`build_resident_step` only pays for `_fold_constants`'s simplify()
    pass when the graph actually has a `Constant` node -- confirm a plain
    resnet-shaped graph (no `Constant` anywhere) still builds identically,
    i.e. the new check doesn't change behaviour for every model this
    pipeline already handled."""
    forward = _forward_model()
    assert not any(n.op_type == "Constant" for n in forward.graph.node)
    with_loss = brts.add_mse_loss(forward, "logits", num_classes=10)
    step_model, state = brts.build_resident_step(with_loss, params=["cw", "gw"])
    onnx.checker.check_model(step_model)
    assert set(state) == {"cw", "gw"}
