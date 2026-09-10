"""End-to-end tests for scripts/generate_distillation_step_graph.py --
onnx-finetune's only distillation implementation, built on onnxsim's own
``graph_grad``/``qat_graph`` autodiff instead of ``onnxruntime.training``.

Everything here runs on a *plain* ``onnxruntime`` -- the whole point of this
path. Three deliberately independent checks, not one relying on the others:

- :func:`test_gradient_matches_finite_differences` is the rigorous one, in
  the same spirit as every builtin rule in tests/test_graph_grad.py: the
  analytic gradient against an independent float64 finite difference, at a
  fixed batch size. Loss decreasing over many steps (the training test)
  only proves *a* descent direction was taken -- a sign error in, say, only
  the soft-loss term could still show the loss going down if the hard-loss
  term dominates, and would not be caught by that test alone.
- :func:`test_gradient_matches_finite_differences_across_batch_sizes` is the
  same rigor, but the point is the *dynamic batch* itself: one compiled
  step graph, the finite-difference check repeated at two different batch
  sizes without rebuilding anything -- proving the backward graph the batch
  axis flows through is genuinely batch-size-agnostic, not merely "declared
  with a dim_param but only actually correct at the size it happened to be
  built against".
- :func:`test_step_graph_trains_on_plain_onnxruntime` is the practical one:
  the actual artifact a caller would run, driven the way the native CLI/WASM
  binding will drive it (feed a batch, feed weights back, repeat), across a
  training run whose batch size changes step to step.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.helper
import onnx.inliner
import onnx.numpy_helper
import pytest
from onnx.reference import ReferenceEvaluator

# generate_distillation_step_graph.py needs the onnxsim package importable
# (onnxsim.graph_grad/qat_graph) -- the *compiled* onnxsim extension, not
# just a pip package (see CLAUDE.md). onnx-finetune-lora.yml's
# lora-scripts job deliberately runs `pytest tools/onnx-finetune/tests`
# without building onnxsim at all ("no compiled onnxsim extension needed"),
# so this must skip there rather than error out at collection time.
pytest.importorskip("onnxsim")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_distillation_step_graph import (  # noqa: E402
    _build_forward_loss_and_grads,
    build_distillation_step_graph,
    labels_to_onehot,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run(script, *args):
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], check=True)


@pytest.fixture
def toy_models(tmp_path):
    teacher_path = tmp_path / "teacher.onnx"
    student_path = tmp_path / "student.onnx"
    _run(
        "make_toy_classifier.py",
        "-o",
        str(teacher_path),
        "--input-dim",
        "8",
        "--hidden-dim",
        "32",
        "--num-classes",
        "4",
        "--seed",
        "1",
    )
    _run(
        "make_toy_classifier.py",
        "-o",
        str(student_path),
        "--input-dim",
        "8",
        "--hidden-dim",
        "8",
        "--num-classes",
        "4",
        "--seed",
        "2",
    )
    return teacher_path, student_path


def _as_double(model: onnx.ModelProto) -> onnx.ModelProto:
    doubled = onnx.ModelProto()
    doubled.CopyFrom(model)
    for value in list(doubled.graph.input) + list(doubled.graph.output):
        if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            value.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    converted = []
    for initializer in doubled.graph.initializer:
        array = onnx.numpy_helper.to_array(initializer)
        if array.dtype == np.float32:
            array = array.astype(np.float64)
        converted.append(onnx.numpy_helper.from_array(array, initializer.name))
    del doubled.graph.initializer[:]
    doubled.graph.initializer.extend(converted)
    return doubled


def _grad_and_loss_models(fwd):
    """``(grad_model, loss_model)``: plain ONNX graphs exposing, respectively,
    every trainable weight's raw gradient and the scalar loss -- shared setup
    between the two finite-difference tests below.

    Declares every per-step input's shape exactly as ``fwd`` itself declares
    it (batch included, dynamic where ``fwd`` says so) rather than
    concretizing anything: proving a *dynamic* graph checks out numerically
    means checking the graph as exported, not a version of it nailed down to
    one size.
    """
    b = fwd.b
    inputs = [
        onnx.helper.make_tensor_value_info(
            fwd.input_name, onnx.TensorProto.FLOAT, fwd.input_shape
        ),
        onnx.helper.make_tensor_value_info(
            fwd.teacher_logits_name, onnx.TensorProto.FLOAT, fwd.logits_shape
        ),
        onnx.helper.make_tensor_value_info(
            fwd.labels_onehot_name, onnx.TensorProto.FLOAT, fwd.logits_shape
        ),
        onnx.helper.make_tensor_value_info(
            fwd.batch_size_name, onnx.TensorProto.FLOAT, []
        ),
    ] + [
        onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, list(value.shape)
        )
        for name, value in fwd.trainable.items()
    ]

    def _finish(nodes, outputs):
        # A templated rule (graph_grad.py's "Add"/"BatchNormalization")
        # appends a call to a model-local function rather than plain ops
        # directly, so it has to be attached and inlined before the result is
        # a plain graph a checker/runtime can handle -- the same thing
        # onnxsim.qat_graph.make_step_graph does for a real step graph, and
        # what tests/test_graph_grad.py's own `_backward_model` does.
        opset_imports = [onnx.helper.make_opsetid("", 17)]
        opset_imports += [onnx.helper.make_opsetid(fn.domain, 1) for fn in b.functions]
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                list(nodes), "probe", inputs, outputs, initializer=list(b.initializer)
            ),
            functions=list(b.functions),
            opset_imports=opset_imports,
        )
        model.ir_version = 8
        onnx.checker.check_model(model)
        if b.functions:
            model = onnx.inliner.inline_local_functions(model)
        return model

    grad_outputs = [
        onnx.helper.make_tensor_value_info(
            fwd.grads[name], onnx.TensorProto.FLOAT, list(value.shape)
        )
        for name, value in fwd.trainable.items()
    ]
    # A plain graph exposing every raw gradient directly, bypassing Adam
    # entirely: Adam's own step-1 update is `lr * sign(gradient)` up to the
    # `eps` guard (see onnxsim.qat_graph.adam_update), which does not cleanly
    # invert back to the gradient's own magnitude. Needs the *full* node list
    # (forward + loss + backward) since the gradients are backward tensors.
    grad_model = _finish(b.nodes, grad_outputs)
    # The finite-difference reference, by contrast, must be forward+loss
    # *only* -- see _ForwardLossGrads.forward_and_loss_nodes's own docstring
    # for why the backward nodes cannot come along for this one.
    loss_model = _finish(
        fwd.forward_and_loss_nodes,
        [onnx.helper.make_tensor_value_info(fwd.combined, onnx.TensorProto.FLOAT, [])],
    )
    return grad_model, loss_model


def _random_feeds(fwd, rng, batch_size):
    feeds = {
        fwd.input_name: rng.standard_normal([batch_size, fwd.input_shape[1]]).astype(
            np.float32
        ),
        fwd.teacher_logits_name: rng.standard_normal(
            [batch_size, fwd.num_classes]
        ).astype(np.float32),
        fwd.labels_onehot_name: labels_to_onehot(
            rng.integers(0, fwd.num_classes, size=batch_size), fwd.num_classes
        ),
        fwd.batch_size_name: np.asarray(float(batch_size), dtype=np.float32),
    }
    for name, value in fwd.trainable.items():
        feeds[name] = value
    return feeds


def _finite_difference_grad(loss_model, feeds, target):
    evaluator = ReferenceEvaluator(_as_double(loss_model))
    feeds64 = {k: np.asarray(v, dtype=np.float64) for k, v in feeds.items()}
    flat = feeds64[target].reshape(-1)
    grad_fd = np.empty_like(flat)
    h = 1e-4
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + h
        plus = float(evaluator.run(None, feeds64)[0])
        flat[i] = original - h
        minus = float(evaluator.run(None, feeds64)[0])
        flat[i] = original
        grad_fd[i] = (plus - minus) / (2.0 * h)
    return grad_fd.reshape(feeds64[target].shape)


@pytest.mark.parametrize("target", ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"])
def test_gradient_matches_finite_differences(toy_models, target):
    ort = pytest.importorskip("onnxruntime")
    _teacher_path, student_path = toy_models
    student = onnx.load(str(student_path))
    fwd = _build_forward_loss_and_grads(student, temperature=2.0, alpha=0.5)
    grad_model, loss_model = _grad_and_loss_models(fwd)

    feeds = _random_feeds(fwd, np.random.default_rng(3), batch_size=6)
    session = ort.InferenceSession(
        grad_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    analytic = session.run([fwd.grads[target]], feeds)[0]

    grad_fd = _finite_difference_grad(loss_model, feeds, target)
    np.testing.assert_allclose(analytic, grad_fd, rtol=2e-3, atol=2e-4)


def test_gradient_matches_finite_differences_across_batch_sizes(toy_models):
    """The dynamic-batch-specific check: the *same compiled graph* (``fwd``
    built once) checked against an independent finite difference at two
    different batch sizes, proving the backward graph's own correctness does
    not depend on which size it happened to see."""
    ort = pytest.importorskip("onnxruntime")
    _teacher_path, student_path = toy_models
    student = onnx.load(str(student_path))
    fwd = _build_forward_loss_and_grads(student, temperature=2.0, alpha=0.5)
    grad_model, loss_model = _grad_and_loss_models(fwd)
    session = ort.InferenceSession(
        grad_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    target = "fc1.weight"
    for batch_size, seed in [(5, 10), (13, 11)]:
        feeds = _random_feeds(fwd, np.random.default_rng(seed), batch_size)
        analytic = session.run([fwd.grads[target]], feeds)[0]
        grad_fd = _finite_difference_grad(loss_model, feeds, target)
        np.testing.assert_allclose(analytic, grad_fd, rtol=2e-3, atol=2e-4)


def test_step_graph_has_no_training_only_ops(toy_models):
    """The whole point of this path: every node is a standard-domain op a
    plain (non-training) onnxruntime already implements."""
    _teacher_path, student_path = toy_models
    student = onnx.load(str(student_path))
    step, _initial_state, _fwd = build_distillation_step_graph(student)
    domains = {node.domain for node in step.model.graph.node}
    assert domains <= {""}, f"expected only the default onnx domain, found {domains}"


def test_step_graph_trains_on_plain_onnxruntime(toy_models):
    """The practical check: run the actual artifact the way the native
    CLI/WASM binding will -- feed a batch, feed weights back, repeat -- via a
    plain ``onnxruntime.InferenceSession`` (no ``onnxruntime.training``
    import anywhere in this test), with the batch size itself changing from
    step to step, and watch the loss go down.
    """
    ort = pytest.importorskip("onnxruntime")
    teacher_path, student_path = toy_models
    student = onnx.load(str(student_path))
    step, state, _fwd = build_distillation_step_graph(
        student, temperature=2.0, alpha=0.5
    )

    teacher_session = ort.InferenceSession(
        str(teacher_path), providers=["CPUExecutionProvider"]
    )
    step_session = ort.InferenceSession(
        step.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    output_names = [o.name for o in step_session.get_outputs()]

    rng = np.random.default_rng(0)
    input_dim, num_classes = 8, 4
    # A fixed pool of 64 rows, with a *different-sized* random slice of it
    # drawn each step -- proving the one compiled graph really does accept
    # whatever batch size a step hands it, not just a fixed size chosen once.
    pool_size = 64
    x_pool = rng.standard_normal((pool_size, input_dim)).astype(np.float32)
    labels_pool = rng.integers(0, num_classes, size=pool_size)
    teacher_logits_pool = teacher_session.run(None, {"input": x_pool})[0]
    batch_sizes = [32, 8, 64, 1, 17]

    losses = []
    for t in range(50):
        batch_size = batch_sizes[t % len(batch_sizes)]
        idx = rng.choice(pool_size, size=batch_size, replace=False)
        x = x_pool[idx]
        teacher_logits = teacher_logits_pool[idx]
        onehot = labels_to_onehot(labels_pool[idx], num_classes)

        feeds = dict(state)
        feeds["input"] = x
        feeds["teacher_logits"] = teacher_logits
        feeds["labels_onehot"] = onehot
        feeds["batch_size"] = np.asarray(float(batch_size), dtype=np.float32)
        feeds["lr"] = np.asarray(0.05, dtype=np.float32)
        feeds["m_correction"] = np.asarray(
            1.0 / (1.0 - 0.9 ** (t + 1)), dtype=np.float32
        )
        feeds["v_correction"] = np.asarray(
            1.0 / (1.0 - 0.999 ** (t + 1)), dtype=np.float32
        )

        out = dict(zip(output_names, step_session.run(output_names, feeds)))
        loss = float(out[step.loss_name])
        assert np.isfinite(loss)
        losses.append(loss)
        state = {name: out[out_name] for name, out_name in step.state.items()}

    assert losses[-1] < losses[0]
