"""End-to-end tests for scripts/generate_distillation_step_graph.py --
onnx-finetune's distillation support built on onnxsim's own
``graph_grad``/``qat_graph`` autodiff instead of ``onnxruntime.training``.

Unlike tests/test_distillation.py (which needs a training-enabled
onnxruntime build: ``onnxruntime.training.artifacts``/``.api``), everything
here runs on a *plain* ``onnxruntime`` -- the whole point of this path. The
two tests below are deliberately independent checks, not one relying on the
other:

- :func:`test_gradient_matches_finite_differences` is the rigorous one, in
  the same spirit as every builtin rule in tests/test_graph_grad.py: the
  analytic gradient against an independent float64 finite difference. Loss
  decreasing over many steps (the other test) only proves *a* descent
  direction was taken -- a sign error in, say, only the soft-loss term could
  still show the loss going down if the hard-loss term dominates, and would
  not be caught by that test alone.
- :func:`test_step_graph_trains_on_plain_onnxruntime` is the practical one:
  the actual artifact a caller would run, driven the way the native CLI/WASM
  binding will drive it (feed a batch, feed weights back, repeat).
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
        "make_toy_classifier.py", "-o", str(teacher_path),
        "--input-dim", "8", "--hidden-dim", "32", "--num-classes", "4", "--seed", "1",
    )
    _run(
        "make_toy_classifier.py", "-o", str(student_path),
        "--input-dim", "8", "--hidden-dim", "8", "--num-classes", "4", "--seed", "2",
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


@pytest.mark.parametrize("target", ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"])
def test_gradient_matches_finite_differences(toy_models, target):
    ort = pytest.importorskip("onnxruntime")
    _teacher_path, student_path = toy_models
    student = onnx.load(str(student_path))
    batch_size = 6
    fwd = _build_forward_loss_and_grads(student, batch_size, temperature=2.0, alpha=0.5)
    b = fwd.b

    inputs = [
        onnx.helper.make_tensor_value_info(fwd.input_name, onnx.TensorProto.FLOAT, fwd.input_shape),
        onnx.helper.make_tensor_value_info(
            fwd.teacher_logits_name, onnx.TensorProto.FLOAT, fwd.logits_shape
        ),
        onnx.helper.make_tensor_value_info(
            fwd.labels_onehot_name, onnx.TensorProto.FLOAT, [fwd.rows, fwd.num_classes]
        ),
    ] + [
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, list(value.shape))
        for name, value in fwd.trainable.items()
    ]

    grad_name = fwd.grads[target]
    grad_shape = list(fwd.trainable[target].shape)

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

    # A plain graph exposing the raw gradient directly, bypassing Adam
    # entirely: Adam's own step-1 update is `lr * sign(gradient)` up to the
    # `eps` guard (see onnxsim.qat_graph.adam_update), which does not cleanly
    # invert back to the gradient's own magnitude. Needs the *full* node list
    # (forward + loss + backward) since `grad_name` is a backward tensor.
    grad_model = _finish(
        b.nodes,
        [onnx.helper.make_tensor_value_info(grad_name, onnx.TensorProto.FLOAT, grad_shape)],
    )
    # The finite-difference reference, by contrast, must be forward+loss
    # *only* -- see _ForwardLossGrads.forward_and_loss_nodes's own docstring
    # for why the backward nodes cannot come along for this one.
    loss_model = _finish(
        fwd.forward_and_loss_nodes,
        [onnx.helper.make_tensor_value_info(fwd.combined, onnx.TensorProto.FLOAT, [])],
    )

    rng = np.random.default_rng(3)
    feeds = {
        fwd.input_name: rng.standard_normal(fwd.input_shape).astype(np.float32),
        fwd.teacher_logits_name: rng.standard_normal(fwd.logits_shape).astype(np.float32),
        fwd.labels_onehot_name: labels_to_onehot(
            rng.integers(0, fwd.num_classes, size=batch_size), fwd.num_classes
        ),
    }
    for name, value in fwd.trainable.items():
        feeds[name] = value

    session = ort.InferenceSession(grad_model.SerializeToString(), providers=["CPUExecutionProvider"])
    analytic = session.run([grad_name], feeds)[0]

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
    grad_fd = grad_fd.reshape(feeds64[target].shape)

    np.testing.assert_allclose(analytic, grad_fd, rtol=2e-3, atol=2e-4)


def test_step_graph_has_no_training_only_ops(toy_models):
    """The whole point of this path: every node is a standard-domain op a
    plain (non-training) onnxruntime already implements."""
    _teacher_path, student_path = toy_models
    student = onnx.load(str(student_path))
    step, _initial_state, _fwd = build_distillation_step_graph(student, batch_size=8)
    domains = {node.domain for node in step.model.graph.node}
    assert domains <= {""}, f"expected only the default onnx domain, found {domains}"


def test_step_graph_trains_on_plain_onnxruntime(toy_models):
    """The practical check: run the actual artifact the way the native
    CLI/WASM binding will -- feed a batch, feed weights back, repeat -- via a
    plain ``onnxruntime.InferenceSession`` (no ``onnxruntime.training``
    import anywhere in this test), and watch the loss go down.
    """
    ort = pytest.importorskip("onnxruntime")
    teacher_path, student_path = toy_models
    student = onnx.load(str(student_path))
    batch_size = 32
    step, state, _fwd = build_distillation_step_graph(student, batch_size, temperature=2.0, alpha=0.5)

    teacher_session = ort.InferenceSession(str(teacher_path), providers=["CPUExecutionProvider"])
    step_session = ort.InferenceSession(
        step.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    output_names = [o.name for o in step_session.get_outputs()]

    rng = np.random.default_rng(0)
    input_dim, num_classes = 8, 4
    x = rng.standard_normal((batch_size, input_dim)).astype(np.float32)
    labels = rng.integers(0, num_classes, size=batch_size)
    teacher_logits = teacher_session.run(None, {"input": x})[0]
    onehot = labels_to_onehot(labels, num_classes)

    losses = []
    for t in range(50):
        feeds = dict(state)
        feeds["input"] = x
        feeds["teacher_logits"] = teacher_logits
        feeds["labels_onehot"] = onehot
        feeds["lr"] = np.asarray(0.05, dtype=np.float32)
        feeds["m_correction"] = np.asarray(1.0 / (1.0 - 0.9 ** (t + 1)), dtype=np.float32)
        feeds["v_correction"] = np.asarray(1.0 / (1.0 - 0.999 ** (t + 1)), dtype=np.float32)

        out = dict(zip(output_names, step_session.run(output_names, feeds)))
        loss = float(out[step.loss_name])
        assert np.isfinite(loss)
        losses.append(loss)
        state = {name: out[out_name] for name, out_name in step.state.items()}

    assert losses[-1] < losses[0]
