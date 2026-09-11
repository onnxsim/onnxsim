"""Tests for ``onnxsim.distributed`` (see ``onnxsim/distributed.py`` and
``docs/distributed-data-parallel-plan.md``) -- real multi-process
data-parallel training built on ``onnxsim.graph_grad``/``onnxsim.qat_graph``.

Two things are checked independently, deliberately kept apart:

- :func:`test_gradient_averaging_matches_combined_batch_reference` -- the
  core mathematical claim (averaging equal-sized workers' gradients equals
  the gradient over their concatenated batch), checked directly against
  :func:`onnxsim.distributed.build_gradient_step_graph`'s own output with no
  multiprocessing involved at all, so a failure here can't be blamed on
  process orchestration.
- :func:`test_train_data_parallel_reduces_loss_via_real_multiprocessing` --
  that :func:`onnxsim.distributed.train_data_parallel` actually spawns real
  OS processes and trains correctly end to end.
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.shape_inference
import pytest

from onnxsim import backend, distributed

ort = pytest.importorskip("onnxruntime")


def _shapes_of(model: onnx.ModelProto):
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    ):
        shapes[value.name] = [d.dim_value for d in value.type.tensor_type.shape.dim]
    for initializer in inferred.graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    return shapes


def _make_mlp(rng, batch_size, d_in=4, d_hidden=6, d_out=3):
    """A tiny 2-layer MLP (MatMul/Add/Relu/MatMul/Add), every weight
    trainable -- small enough that a handful of local steps visibly moves
    the loss, the same bar ``tests/test_lora.py`` sets for its own toy
    models."""
    w1 = (rng.standard_normal((d_in, d_hidden)) * 0.3).astype(np.float32)
    b1 = np.zeros(d_hidden, dtype=np.float32)
    w2 = (rng.standard_normal((d_hidden, d_out)) * 0.3).astype(np.float32)
    b2 = np.zeros(d_out, dtype=np.float32)

    nodes = [
        onnx.helper.make_node("MatMul", ["input", "W1"], ["mm1"]),
        onnx.helper.make_node("Add", ["mm1", "b1"], ["add1"]),
        onnx.helper.make_node("Relu", ["add1"], ["relu1"]),
        onnx.helper.make_node("MatMul", ["relu1", "W2"], ["mm2"]),
        onnx.helper.make_node("Add", ["mm2", "b2"], ["output"]),
    ]
    initializers = [
        onnx.numpy_helper.from_array(w1, "W1"),
        onnx.numpy_helper.from_array(b1, "b1"),
        onnx.numpy_helper.from_array(w2, "W2"),
        onnx.numpy_helper.from_array(b2, "b2"),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "toy_mlp",
        [
            onnx.helper.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT, [batch_size, d_in]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "output", onnx.TensorProto.FLOAT, [batch_size, d_out]
            )
        ],
        initializer=initializers,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    shapes = _shapes_of(model)
    param_names = ["W1", "b1", "W2", "b2"]
    initial_params = {
        name: onnx.numpy_helper.to_array(t).copy()
        for name, t in zip(param_names, initializers)
    }
    return (
        model,
        nodes,
        initializers,
        shapes,
        param_names,
        initial_params,
        (d_in, d_hidden, d_out),
    )


def test_gradient_averaging_matches_combined_batch_reference():
    rng = np.random.default_rng(0)
    batch_size = 5
    (
        model,
        nodes,
        initializers,
        shapes,
        param_names,
        _initial,
        (d_in, _d_hidden, d_out),
    ) = _make_mlp(rng, batch_size)

    per_worker_model, grad_names, loss_name = distributed.build_gradient_step_graph(
        nodes,
        shapes,
        initializers,
        "input",
        [batch_size, d_in],
        "target",
        [batch_size, d_out],
        "output",
        param_names,
    )
    per_worker_runner = backend.Runner(
        per_worker_model,
        output_names=[grad_names[n] for n in param_names] + [loss_name],
    )

    x_a = rng.standard_normal((batch_size, d_in)).astype(np.float32)
    y_a = rng.standard_normal((batch_size, d_out)).astype(np.float32)
    x_b = rng.standard_normal((batch_size, d_in)).astype(np.float32) + 3.0
    y_b = rng.standard_normal((batch_size, d_out)).astype(np.float32) - 3.0

    params = {
        n: onnx.numpy_helper.to_array(t) for n, t in zip(param_names, initializers)
    }

    out_a = per_worker_runner({"input": x_a, "target": y_a, **params})
    out_b = per_worker_runner({"input": x_b, "target": y_b, **params})
    averaged = {
        name: np.mean([out_a[grad_names[name]], out_b[grad_names[name]]], axis=0)
        for name in param_names
    }

    combined_model, combined_grad_names, _combined_loss = (
        distributed.build_gradient_step_graph(
            nodes,
            shapes,
            initializers,
            "input",
            [2 * batch_size, d_in],
            "target",
            [2 * batch_size, d_out],
            "output",
            param_names,
        )
    )
    combined_runner = backend.Runner(
        combined_model, output_names=[combined_grad_names[n] for n in param_names]
    )
    combined_out = combined_runner(
        {
            "input": np.concatenate([x_a, x_b], axis=0),
            "target": np.concatenate([y_a, y_b], axis=0),
            **params,
        }
    )

    for name in param_names:
        np.testing.assert_allclose(
            averaged[name],
            combined_out[combined_grad_names[name]],
            rtol=1e-4,
            atol=1e-5,
        )


def test_build_apply_step_graph_moves_params_in_the_gradients_direction():
    rng = np.random.default_rng(0)
    shapes = {"W": [3, 2]}
    step = distributed.build_apply_step_graph(shapes, ["W"])
    runner = backend.Runner(step.model, output_names=list(step.state.values()))

    w = rng.standard_normal((3, 2)).astype(np.float32)
    grad = np.ones((3, 2), dtype=np.float32)
    out = runner(
        {
            "W": w,
            "m__W": np.zeros_like(w),
            "v__W": np.zeros_like(w),
            "grad__W": grad,
            "lr": np.array(0.1, dtype=np.float32),
            "m_correction": np.array(1.0, dtype=np.float32),
            "v_correction": np.array(1.0, dtype=np.float32),
        }
    )
    w_next = out[step.state["W"]]
    # A positive gradient everywhere must move every element of W down.
    assert np.all(w_next < w)


def test_train_data_parallel_requires_at_least_one_shard():
    rng = np.random.default_rng(0)
    (
        _model,
        nodes,
        initializers,
        shapes,
        param_names,
        initial_params,
        (d_in, _h, d_out),
    ) = _make_mlp(rng, 4)
    grad_model, grad_names, loss_name = distributed.build_gradient_step_graph(
        nodes,
        shapes,
        initializers,
        "input",
        [4, d_in],
        "target",
        [4, d_out],
        "output",
        param_names,
    )
    with pytest.raises(ValueError, match="at least one worker"):
        distributed.train_data_parallel(
            grad_model,
            grad_names,
            loss_name,
            "input",
            "target",
            param_names,
            shapes,
            initial_params,
            [],
            num_steps=1,
            batch_size=4,
            learning_rate=1e-2,
        )


def test_train_data_parallel_reduces_loss_via_real_multiprocessing():
    rng = np.random.default_rng(0)
    batch_size = 4
    (
        _model,
        nodes,
        initializers,
        shapes,
        param_names,
        initial_params,
        (d_in, _h, d_out),
    ) = _make_mlp(rng, batch_size)
    grad_model, grad_names, loss_name = distributed.build_gradient_step_graph(
        nodes,
        shapes,
        initializers,
        "input",
        [batch_size, d_in],
        "target",
        [batch_size, d_out],
        "output",
        param_names,
    )

    # Two workers holding different random samples of the *same* underlying
    # linear relationship -- the realistic data-parallel scenario (one
    # dataset sharded across workers), not the federated-learning one
    # tests/test_federated.py's own two clients model (there, each client's
    # data follows a genuinely different relationship, which is exactly
    # what per-client LoRA adapters, not one gradient-averaged shared model,
    # are suited to fit). A single set of weights trained by averaging
    # gradients from two *conflicting* teachers can only converge to a
    # compromise fit for both, not drive either one's loss sharply down --
    # that is a real property of gradient-averaged DP, not a bug, but it
    # would make a noisy, threshold-fragile test here.
    true_w = (rng.standard_normal((d_in, d_out)) * 0.5).astype(np.float32)

    def make_shard(local_rng):
        x = local_rng.standard_normal((32, d_in)).astype(np.float32)
        y = x @ true_w + 0.05 * local_rng.standard_normal((32, d_out)).astype(
            np.float32
        )
        return distributed.WorkerShard(inputs=x, targets=y.astype(np.float32))

    shard_a = make_shard(rng)
    shard_b = make_shard(rng)

    def whole_shard_loss(params, shard):
        eval_model, _grads, eval_loss_name = distributed.build_gradient_step_graph(
            nodes,
            shapes,
            initializers,
            "input",
            [shard.inputs.shape[0], d_in],
            "target",
            [shard.inputs.shape[0], d_out],
            "output",
            param_names,
        )
        runner = backend.Runner(eval_model, output_names=[eval_loss_name])
        out = runner({"input": shard.inputs, "target": shard.targets, **params})
        return float(out[eval_loss_name])

    loss_a_before = whole_shard_loss(initial_params, shard_a)
    loss_b_before = whole_shard_loss(initial_params, shard_b)

    final_params, step_losses = distributed.train_data_parallel(
        grad_model,
        grad_names,
        loss_name,
        "input",
        "target",
        param_names,
        shapes,
        initial_params,
        [shard_a, shard_b],
        num_steps=150,
        batch_size=batch_size,
        learning_rate=2e-2,
        seed=1,
    )

    assert len(step_losses) == 150
    assert all(np.isfinite(loss) for loss in step_losses)

    loss_a_after = whole_shard_loss(final_params, shard_a)
    loss_b_after = whole_shard_loss(final_params, shard_b)
    assert loss_a_after < 0.1 * loss_a_before, (loss_a_before, loss_a_after)
    assert loss_b_after < 0.1 * loss_b_before, (loss_b_before, loss_b_after)
