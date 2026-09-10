"""Tests for ``onnxsim.torch_training`` -- exporting a real ``torch.nn.Module``
via ``torch.export``'s FX graph and training it entirely on onnxsim's own
grad templating (:mod:`onnxsim.compile_training`), never on ``torch.autograd``.

Needs ``torch >= 2.5`` and ``onnxscript`` (the ``onnxsim[torch-training]``
extra); skipped entirely otherwise, the same way the rest of this repo skips
an optional-dependency-gated test file.
"""

import numpy as np
import onnx
import pytest

from onnxsim import graph_grad, torch_training

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
ort = pytest.importorskip("onnxruntime")


class _Regression(torch.nn.Module):
    """``loss = mean((x @ w^T - y) ** 2)``, the same objective
    ``tests/test_compile_training.py``'s own ``_linear_model`` fits, now
    authored as an ordinary torch module instead of ``onnx.parser`` text --
    what this module's whole point is exercising.

    Deliberately ``(diff * diff)``, not ``diff ** 2``: torch's dynamo
    exporter lowers ``**`` to ``Pow``, which has no rule in
    :mod:`onnxsim.graph_grad` (:data:`onnxsim.graph_grad.SUPPORTED_OPS`);
    ``*`` lowers to ``Mul``, which does. That gap -- a real one, not a test
    artifact -- is exercised directly in
    ``test_pow_is_refused_with_the_ops_graph_grad_actually_differentiates``.
    """

    def __init__(self, n: int = 2, k: int = 3, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(n, k, generator=g) * 0.1)

    def forward(self, x, y):
        y_hat = x @ self.w.T
        diff = y_hat - y
        return (diff * diff).mean()


def _example_and_batch(rows=8, k=3, n=2, seed=1):
    example = (torch.zeros(rows, k), torch.zeros(rows, n))
    rng = np.random.default_rng(seed)
    w_true = rng.standard_normal((n, k)).astype(np.float32)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    y = x @ w_true.T
    return example, w_true, x, y


def test_export_produces_a_static_shape_onnx_model_named_w():
    module = _Regression()
    example, *_ = _example_and_batch()
    model = torch_training.export_torch_module_to_onnx(
        module, example, input_names=["x", "y"], output_names=("loss",)
    )
    onnx.checker.check_model(model)
    assert [i.name for i in model.graph.input] == ["x", "y"]
    assert [o.name for o in model.graph.output] == ["loss"]
    # The parameter's own qualified name, unmangled -- what
    # compile_torch_training_loop's params= default relies on.
    assert [t.name for t in model.graph.initializer] == ["w"]
    # Every op is one onnxsim.graph_grad can differentiate -- see _Regression's
    # own docstring on why this is a real constraint, not automatic.
    ops = {n.op_type for n in model.graph.node}
    assert ops <= graph_grad.supported_ops()


def test_export_folds_the_full_reduction_squeeze():
    """``tensor.mean()`` with no explicit axis is exactly the decomposition
    :func:`onnxsim.torch_training._fold_full_reduction_squeeze` exists for
    (see its own docstring): a keepdims=1 ReduceMean immediately Squeezed
    back to a scalar. Folded, no Squeeze should remain at all.
    """
    module = _Regression()
    example, *_ = _example_and_batch()
    model = torch_training.export_torch_module_to_onnx(module, example)
    ops = [n.op_type for n in model.graph.node]
    assert "Squeeze" not in ops
    assert "ReduceMean" in ops


def test_compile_torch_training_loop_trains_to_match_the_true_weight():
    module = _Regression()
    example, w_true, x, y = _example_and_batch()
    loop = torch_training.compile_torch_training_loop(module, example)

    losses = [loop({"x": x, "y": y}, lr=5e-2) for _ in range(300)]
    assert losses[-1] < 1e-6 * losses[0]
    np.testing.assert_allclose(loop.parameters()["w"], w_true, atol=1e-3)


def test_params_default_to_named_parameters():
    module = _Regression()
    example, *_ = _example_and_batch()
    loop = torch_training.compile_torch_training_loop(module, example)
    assert loop.params == ("w",)


def test_params_can_be_overridden_explicitly():
    module = _Regression()
    example, *_ = _example_and_batch()
    loop = torch_training.compile_torch_training_loop(module, example, params=("w",))
    assert loop.params == ("w",)


def test_dict_example_inputs_are_accepted():
    module = _Regression()
    (x_ex, y_ex), w_true, x, y = _example_and_batch()
    loop = torch_training.compile_torch_training_loop(module, {"x": x_ex, "y": y_ex})
    losses = [loop({"x": x, "y": y}, lr=5e-2) for _ in range(50)]
    assert losses[-1] < 0.5 * losses[0]


def test_sgd_momentum_optimizer_also_trains():
    module = _Regression()
    example, w_true, x, y = _example_and_batch()
    loop = torch_training.compile_torch_training_loop(
        module, example, optimizer="sgd_momentum"
    )
    losses = [loop({"x": x, "y": y}, lr=5e-2) for _ in range(300)]
    assert losses[-1] < 0.1 * losses[0]


def test_missing_param_name_is_refused_with_the_exported_initializers_listed():
    module = _Regression()
    example, *_ = _example_and_batch()
    with pytest.raises(ValueError, match="not_a_param"):
        torch_training.compile_torch_training_loop(
            module, example, params=("not_a_param",)
        )


def test_output_names_must_be_exactly_one():
    module = _Regression()
    example, *_ = _example_and_batch()
    with pytest.raises(ValueError, match="one loss output"):
        torch_training.export_torch_module_to_onnx(
            module, example, output_names=("loss", "extra")
        )


def test_pow_is_refused_with_the_ops_graph_grad_actually_differentiates():
    """``**`` (``torch.pow``/``Tensor.__pow__``) lowers to ONNX ``Pow``, which
    has no gradient rule in :mod:`onnxsim.graph_grad` -- see _Regression's own
    docstring. Training such a module must fail loudly, the same discipline
    ``tests/test_compile_training.py``'s own
    ``test_unsupported_op_is_refused_loudly`` already covers for a
    hand-written ONNX model; this is the same failure reached from a real
    torch module instead.
    """

    class PowRegression(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2, 3))

        def forward(self, x, y):
            diff = x @ self.w.T - y
            return (diff**2).mean()

    example, *_ = _example_and_batch()
    loop = torch_training.compile_torch_training_loop(PowRegression(), example)
    with pytest.raises(graph_grad.UnsupportedOpError):
        loop.step_graph
