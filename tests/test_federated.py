"""Tests for ``onnxsim.federated`` (see ``onnxsim/federated.py``) -- FedAvg
round orchestration on top of :mod:`onnxsim.lora`'s single-client training.

Two things are checked independently: that :func:`onnxsim.federated.fedavg`
computes the weighted average it claims to, against a hand-computed result
rather than trusting its own arithmetic; and that a federated round/training
run actually reduces each client's own local loss while leaving the base
model's weights untouched, the same property ``tests/test_lora.py`` checks
for a single client.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim import federated, lora

ort = pytest.importorskip("onnxruntime")


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


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _run(model, feeds, output_names=None):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = output_names or [o.name for o in model.graph.output]
    return sess.run(names, feeds)


def _injected_matmul_model(rng, in_dim=8, out_dim=8, rank=4):
    w = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{in_dim}] X) => (float[batch,{out_dim}] Y) {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(w, "W")],
    )
    return lora.inject_lora(model, rank=rank, seed=0), w


def test_fedavg_matches_hand_computed_weighted_average():
    states = [
        {"A": np.array([1.0, 2.0], dtype=np.float32)},
        {"A": np.array([3.0, 6.0], dtype=np.float32)},
    ]
    averaged = federated.fedavg(states, weights=[1.0, 3.0])
    # (1*[1,2] + 3*[3,6]) / 4 = [10, 20] / 4 = [2.5, 5.0]
    np.testing.assert_allclose(averaged["A"], [2.5, 5.0], rtol=1e-6)


def test_fedavg_uniform_weights_is_a_plain_mean():
    states = [
        {"A": np.array([2.0, 4.0], dtype=np.float32)},
        {"A": np.array([4.0, 8.0], dtype=np.float32)},
    ]
    averaged = federated.fedavg(states, weights=[1.0, 1.0])
    np.testing.assert_allclose(averaged["A"], [3.0, 6.0], rtol=1e-6)


def test_fedavg_rejects_empty_or_mismatched_input():
    with pytest.raises(ValueError, match="at least one"):
        federated.fedavg([], [])
    with pytest.raises(ValueError, match="weights"):
        federated.fedavg([{"A": np.zeros(1, dtype=np.float32)}], [])
    with pytest.raises(ValueError, match="positive"):
        federated.fedavg(
            [{"A": np.zeros(1, dtype=np.float32)}, {"A": np.zeros(1, dtype=np.float32)}],
            [0.0, 0.0],
        )
    with pytest.raises(ValueError, match="same tensors"):
        federated.fedavg(
            [
                {"A": np.zeros(1, dtype=np.float32)},
                {"B": np.zeros(1, dtype=np.float32)},
            ],
            [1.0, 1.0],
        )


def test_federated_client_requires_exactly_one_of_reference_model_or_target_data():
    with pytest.raises(ValueError, match="exactly one of"):
        federated.FederatedClient(calibration_data=[{"X": np.zeros((1, 1))}])
    with pytest.raises(ValueError, match="exactly one of"):
        federated.FederatedClient(
            calibration_data=[{"X": np.zeros((1, 1))}],
            target_data=[np.zeros((1, 1))],
            reference_model=object(),
        )


def test_federated_client_weight_defaults_to_batch_count():
    client = federated.FederatedClient(
        calibration_data=[{"X": np.zeros((1, 1))}, {"X": np.zeros((1, 1))}],
        target_data=[np.zeros((1, 1)), np.zeros((1, 1))],
    )
    assert client.weight() == 2

    weighted = federated.FederatedClient(
        calibration_data=[{"X": np.zeros((1, 1))}],
        target_data=[np.zeros((1, 1))],
        num_examples=128,
    )
    assert weighted.weight() == 128


def test_extract_and_apply_adapter_state_round_trip():
    rng = np.random.default_rng(0)
    (injected, adapter), _ = _injected_matmul_model(rng)

    state = federated.extract_adapter_state(injected, adapter)
    assert set(state) == set(adapter.parameter_names())

    bumped = {name: value + 1.0 for name, value in state.items()}
    applied = federated.apply_adapter_state(injected, bumped)
    onnx.checker.check_model(applied)

    round_tripped = federated.extract_adapter_state(applied, adapter)
    for name in state:
        np.testing.assert_allclose(round_tripped[name], state[name] + 1.0)

    # Only the adapter's own tensors moved -- the base weight is untouched.
    w_before = onnx.numpy_helper.to_array(
        next(t for t in injected.graph.initializer if t.name == "W")
    )
    w_after = onnx.numpy_helper.to_array(
        next(t for t in applied.graph.initializer if t.name == "W")
    )
    np.testing.assert_array_equal(w_before, w_after)


def test_run_federated_round_requires_at_least_one_client():
    rng = np.random.default_rng(0)
    (injected, adapter), _ = _injected_matmul_model(rng)
    with pytest.raises(ValueError, match="at least one client"):
        federated.run_federated_round(injected, adapter, [], "X", "Y")


def test_run_federated_training_reduces_every_clients_loss_and_freezes_base_weight():
    rng = np.random.default_rng(0)
    (injected, adapter), w = _injected_matmul_model(rng, in_dim=8, out_dim=8, rank=4)

    # Two clients with different, non-overlapping local data and targets --
    # simulating the non-IID case FedAvg is meant to handle, not two shards
    # of one shared dataset.
    x_a = rng.standard_normal((16, 8)).astype(np.float32)
    y_a = rng.standard_normal((16, 8)).astype(np.float32)
    x_b = rng.standard_normal((16, 8)).astype(np.float32) + 5.0
    y_b = rng.standard_normal((16, 8)).astype(np.float32) - 5.0

    client_a = federated.FederatedClient(
        calibration_data=[{"X": x_a}], target_data=[y_a]
    )
    client_b = federated.FederatedClient(
        calibration_data=[{"X": x_b}], target_data=[y_b]
    )

    def client_loss(model, x, y):
        (pred,) = _run(model, {"X": x}, ["Y"])
        return float(np.mean((pred - y) ** 2))

    loss_a_before = client_loss(injected, x_a, y_a)
    loss_b_before = client_loss(injected, x_b, y_b)

    round_losses = []
    trained = federated.run_federated_training(
        injected,
        adapter,
        [client_a, client_b],
        "X",
        "Y",
        num_rounds=20,
        local_iterations=10,
        learning_rate=1e-2,
        round_losses=round_losses,
    )
    onnx.checker.check_model(trained)

    assert len(round_losses) == 20
    assert all(len(losses) == 2 for losses in round_losses)

    loss_a_after = client_loss(trained, x_a, y_a)
    loss_b_after = client_loss(trained, x_b, y_b)
    assert loss_a_after < 0.5 * loss_a_before, (loss_a_before, loss_a_after)
    assert loss_b_after < 0.5 * loss_b_before, (loss_b_before, loss_b_after)

    w_after = onnx.numpy_helper.to_array(
        next(t for t in trained.graph.initializer if t.name == "W")
    )
    np.testing.assert_array_equal(w, w_after)


def test_run_federated_round_does_not_mutate_the_global_model():
    rng = np.random.default_rng(0)
    (injected, adapter), _ = _injected_matmul_model(rng)
    before = federated.extract_adapter_state(injected, adapter)

    x = rng.standard_normal((8, 8)).astype(np.float32)
    y = rng.standard_normal((8, 8)).astype(np.float32)
    client = federated.FederatedClient(calibration_data=[{"X": x}], target_data=[y])

    federated.run_federated_round(
        injected, adapter, [client], "X", "Y", local_iterations=5
    )

    after = federated.extract_adapter_state(injected, adapter)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


def test_run_federated_round_matches_manual_fedavg_of_independent_client_runs():
    rng = np.random.default_rng(0)
    (injected, adapter), _ = _injected_matmul_model(rng, in_dim=6, out_dim=6, rank=3)

    x_a = rng.standard_normal((8, 6)).astype(np.float32)
    y_a = rng.standard_normal((8, 6)).astype(np.float32)
    x_b = rng.standard_normal((8, 6)).astype(np.float32)
    y_b = rng.standard_normal((8, 6)).astype(np.float32)

    client_a = federated.FederatedClient(
        calibration_data=[{"X": x_a}], target_data=[y_a], num_examples=1
    )
    client_b = federated.FederatedClient(
        calibration_data=[{"X": x_b}], target_data=[y_b], num_examples=3
    )

    trained_a = lora.train_lora(
        injected, adapter, "X", "Y",
        target_data=[y_a], calibration_data=[{"X": x_a}],
        num_iterations=5, learning_rate=1e-2,
    )
    trained_b = lora.train_lora(
        injected, adapter, "X", "Y",
        target_data=[y_b], calibration_data=[{"X": x_b}],
        num_iterations=5, learning_rate=1e-2,
    )
    expected = federated.fedavg(
        [
            federated.extract_adapter_state(trained_a, adapter),
            federated.extract_adapter_state(trained_b, adapter),
        ],
        weights=[1, 3],
    )

    next_global, _ = federated.run_federated_round(
        injected,
        adapter,
        [client_a, client_b],
        "X",
        "Y",
        local_iterations=5,
        learning_rate=1e-2,
    )
    actual = federated.extract_adapter_state(next_global, adapter)

    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-5, atol=1e-6)
