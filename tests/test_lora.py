"""Tests for ``onnxsim.lora`` (see ``onnxsim/lora.py``) -- LoRA adapter
injection and training on top of :mod:`onnxsim.graph_grad`/
:mod:`onnxsim.qat_graph`, onnxsim's own answer to what
``tools/onnx-finetune`` needs a training-enabled ONNX Runtime build for.

Two things are checked independently, the same split
``tests/test_graph_grad_templates.py`` used this session for the templated
gradient rules: that :func:`onnxsim.lora.inject_lora`'s graph surgery is
correct on its own (a numeric no-op at init, the base weight never touched),
and that the gradient :func:`onnxsim.graph_grad.build_backward` computes for
an injected adapter's own ``A``/``B`` matrices agrees with an independent
implementation -- ``torch.autograd`` on the identical computation -- rather
than merely trusting :func:`onnxsim.lora.train_lora`'s own loss curve.
"""

import builtins
import sys
import types

import numpy as np
import onnx
import onnx.inliner
import onnx.numpy_helper
import pytest
from onnx import parser

from onnxsim import graph_grad, lora, qat, qat_graph

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


def test_inject_lora_freezes_the_base_weight_and_is_a_noop_at_init():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((6, 8)).astype(np.float32)
    model = _model(
        """
        g (float[batch,6] X) => (float[batch,8] Y) {
          Y = MatMul(X, W)
        }
        """,
        [_f32(w, "W")],
    )

    injected, adapter = lora.inject_lora(model, rank=3, seed=0)
    onnx.checker.check_model(injected)

    assert len(adapter.targets) == 1
    target = adapter.targets[0]
    assert target.weight_name == "W"
    assert target.op_type == "MatMul"

    # The base weight is never read by anything new -- byte-for-byte
    # unchanged.
    w_after = onnx.numpy_helper.to_array(
        next(t for t in injected.graph.initializer if t.name == "W")
    )
    np.testing.assert_array_equal(w, w_after)

    # B starts at zero, so the injected model computes exactly what the
    # original did.
    x = rng.standard_normal((4, 6)).astype(np.float32)
    (y_before,) = _run(model, {"X": x})
    (y_after,) = _run(injected, {"X": x})
    np.testing.assert_allclose(y_before, y_after, rtol=1e-5, atol=1e-5)


def test_inject_lora_gemm_transA_and_conv_1x1():
    rng = np.random.default_rng(0)
    # transA=1, transB=0 (default): A' = X^T is [M, K], B' = Wg is [K, N]
    # directly -- so Wg's raw shape is [K, N] = [6, 8], not [N, K].
    w_gemm = rng.standard_normal((6, 8)).astype(np.float32)
    xc = rng.standard_normal((1, 3, 4, 4)).astype(np.float32)
    wc = rng.standard_normal((5, 3, 1, 1)).astype(np.float32)
    model = _model(
        """
        g (float[6,batch] X, float[1,3,4,4] Xc) => (float[batch,8] Y, float[1,5,4,4] Yc) {
          Y = Gemm <transA = 1> (X, Wg)
          Yc = Conv <kernel_shape = [1, 1]> (Xc, Wc)
        }
        """,
        [_f32(w_gemm, "Wg"), _f32(wc, "Wc")],
    )

    injected, adapter = lora.inject_lora(model, rank=2, seed=0)
    onnx.checker.check_model(injected)

    by_op = {t.op_type: t for t in adapter.targets}
    assert set(by_op) == {"Gemm", "Conv"}

    op_types = [n.op_type for n in injected.graph.node]
    assert "Transpose" in op_types  # transA=1's extra branch input transpose
    conv_target = by_op["Conv"]
    a_conv = next(
        onnx.numpy_helper.to_array(t)
        for t in injected.graph.initializer
        if t.name == conv_target.lora_a_name
    )
    assert a_conv.shape == (2, 3, 1, 1)  # [rank, in_ch, 1, 1]

    x = rng.standard_normal((6, 4)).astype(np.float32)
    (y_before, yc_before) = _run(model, {"X": x, "Xc": xc})
    (y_after, yc_after) = _run(injected, {"X": x, "Xc": xc})
    np.testing.assert_allclose(y_before, y_after, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(yc_before, yc_after, rtol=1e-5, atol=1e-5)


def test_train_lora_reduces_loss_and_freezes_base_weight():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 8)).astype(np.float32)
    model = _model(
        """
        g (float[batch,8] X) => (float[batch,8] Y) {
          Y = MatMul(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    injected, adapter = lora.inject_lora(model, rank=4, seed=0)

    x = rng.standard_normal((32, 8)).astype(np.float32)
    target_data = rng.standard_normal((32, 8)).astype(np.float32)

    losses = []
    trained = lora.train_lora(
        injected,
        adapter,
        "X",
        "Y",
        target_data=[target_data],
        calibration_data=[{"X": x}],
        num_iterations=500,
        learning_rate=1e-2,
        losses=losses,
    )
    onnx.checker.check_model(trained)

    assert losses[-1] < 0.3 * losses[0], (losses[0], losses[-1])

    w_after = onnx.numpy_helper.to_array(
        next(t for t in trained.graph.initializer if t.name == "W")
    )
    np.testing.assert_array_equal(w, w_after)

    b_after = onnx.numpy_helper.to_array(
        next(
            t
            for t in trained.graph.initializer
            if t.name == adapter.targets[0].lora_b_name
        )
    )
    assert not np.allclose(b_after, 0.0)  # actually moved from its zero init


def test_train_lora_requires_exactly_one_of_reference_model_or_target_data():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 4)).astype(np.float32)
    model = _model(
        """
        g (float[batch,4] X) => (float[batch,4] Y) {
          Y = MatMul(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    injected, adapter = lora.inject_lora(model, rank=2, seed=0)
    with pytest.raises(ValueError, match="exactly one of"):
        lora.train_lora(injected, adapter, "X", "Y")


def test_train_lora_gradient_matches_torch_autograd():
    torch = pytest.importorskip(
        "torch", reason="this comparison specifically needs torch.autograd"
    )

    rng = np.random.default_rng(0)
    w = rng.standard_normal((6, 6)).astype(np.float32)
    model = _model(
        """
        g (float[batch,6] X) => (float[batch,6] Y) {
          Y = MatMul(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    injected, adapter = lora.inject_lora(model, rank=3, alpha=2.0, seed=0)
    target = adapter.targets[0]

    a0 = onnx.numpy_helper.to_array(
        next(t for t in injected.graph.initializer if t.name == target.lora_a_name)
    )
    # B starts at zero, which makes d(loss)/dA identically zero regardless of
    # whether the branch is wired correctly (A's gradient is scaled by B) --
    # not a fair check of the mechanism. Seed both away from their init so
    # every gradient this compares is actually exercised.
    b0 = rng.standard_normal((3, 6)).astype(np.float32)
    for t in injected.graph.initializer:
        if t.name == target.lora_b_name:
            t.CopyFrom(onnx.numpy_helper.from_array(b0, name=target.lora_b_name))

    x = rng.standard_normal((4, 6)).astype(np.float32)
    seed = rng.standard_normal((4, 6)).astype(np.float32)

    # The real graph inject_lora produced, differentiated by the real
    # graph_grad.build_backward w.r.t. A/B -- exactly the composition
    # train_lora relies on, minus the Adam-optimizer wrapper (whose own
    # first-step update is not simply proportional to the raw gradient, so
    # comparing it directly would not isolate the gradient computation).
    dummy_y = np.zeros((4, 6), dtype=np.float32)
    shapes = qat._block_shapes(
        injected, list(injected.graph.node), {"X": x}, "Y", dummy_y
    )
    b = qat_graph.GraphBuilder("t_")
    grads = graph_grad.build_backward(
        b,
        list(injected.graph.node),
        shapes,
        {"Y": "dY"},
        [target.lora_a_name, target.lora_b_name],
    )
    nodes = list(injected.graph.node) + list(b.nodes)
    outputs = []
    for name, grad in [
        ("dA", grads[target.lora_a_name]),
        ("dB", grads[target.lora_b_name]),
    ]:
        nodes.append(onnx.helper.make_node("Identity", [grad], [name]))
        outputs.append(name)
    graph = onnx.helper.make_graph(
        nodes,
        "grad_check",
        [
            onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4, 6]),
            onnx.helper.make_tensor_value_info("dY", onnx.TensorProto.FLOAT, [4, 6]),
        ],
        [
            onnx.helper.make_tensor_value_info("dA", onnx.TensorProto.FLOAT, [6, 3]),
            onnx.helper.make_tensor_value_info("dB", onnx.TensorProto.FLOAT, [3, 6]),
        ],
        initializer=list(injected.graph.initializer) + list(b.initializer),
    )
    # The base MatMul's own residual Add (`Y = base + lora_branch * alpha`)
    # goes through graph_grad._RULES's templated "Add" rule -- a call to the
    # checked-in GradAdd FunctionProto, not plain ops -- so b.functions has to
    # be attached and every call site inlined before this is a plain graph a
    # runtime can execute, exactly as qat_graph.make_step_graph does for a
    # real step graph.
    opset_imports = [onnx.helper.make_opsetid("", 17)]
    opset_imports += [onnx.helper.make_opsetid(fn.domain, 1) for fn in b.functions]
    backward_model = onnx.helper.make_model(
        graph, functions=list(b.functions), opset_imports=opset_imports
    )
    backward_model.ir_version = 8
    onnx.checker.check_model(backward_model)
    if b.functions:
        backward_model = onnx.inliner.inline_local_functions(backward_model)
    da, db = _run(backward_model, {"X": x, "dY": seed}, outputs)

    tx = torch.tensor(x)
    tw = torch.tensor(w)
    ta = torch.tensor(a0, requires_grad=True)
    tb = torch.tensor(b0, requires_grad=True)
    talpha_over_rank = torch.tensor(2.0 / 3)
    ty = tx @ tw + (tx @ ta @ tb) * talpha_over_rank
    loss = (ty * torch.tensor(seed)).sum()
    grad_a, grad_b = torch.autograd.grad(loss, [ta, tb])

    np.testing.assert_allclose(da, grad_a.numpy(), rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(db, grad_b.numpy(), rtol=2e-3, atol=2e-4)


def test_apply_qlora_composition_is_smaller_and_valid():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 64)).astype(np.float32)
    model = _model(
        """
        g (float[batch,64] X) => (float[batch,64] Y) {
          Y = MatMul(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    injected, _ = lora.inject_lora(model, rank=4, seed=0)
    quantized, adapter = lora.apply_qlora(model, rank=4, block_size=32, seed=0)
    onnx.checker.check_model(quantized)

    param_names = set(adapter.parameter_names())
    # No node reads the original float32 "W" anymore -- it was rewired to
    # read the NF4 dequant chain's own output instead (onnxsim.nf4 leaves
    # the now-dead initializer in place rather than pruning it, the same as
    # every other onnxsim quantize_* function).
    read_names = {name for n in quantized.graph.node for name in n.input}
    assert "W" not in read_names
    for t in quantized.graph.initializer:
        if t.name in param_names:
            assert t.data_type == onnx.TensorProto.FLOAT
    codes = [
        t for t in quantized.graph.initializer if t.data_type == onnx.TensorProto.UINT8
    ]
    assert codes, "expected at least one NF4 code tensor"
    # The actual compression QLoRA buys: one packed uint8 per weight element
    # (4 bits used, one byte stored -- onnxsim.nf4 does not sub-byte-pack)
    # against float32's 4 bytes -- a 4x reduction for the base weight,
    # independent of onnxsim.nf4.quantize_weight_only_nf4's own convention
    # of leaving the now-dead float32 "W" initializer in the graph rather
    # than pruning it (unrelated to this module, and not exercised by
    # tests/test_nf4.py either).
    w_bytes = next(t for t in injected.graph.initializer if t.name == "W").ByteSize()
    codes_bytes = sum(t.ByteSize() for t in codes)
    assert codes_bytes < w_bytes / 3


def test_apply_qlora_then_train_lora_does_not_raise_on_the_nf4_dequant_chain():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 64)).astype(np.float32)
    model = _model(
        """
        g (float[batch,64] X) => (float[batch,64] Y) {
          Y = MatMul(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    quantized, adapter = lora.apply_qlora(model, rank=8, block_size=32, seed=0)
    assert any(n.op_type == "Cast" for n in quantized.graph.node), (
        "test assumes the NF4 dequant chain (which includes Cast, absent "
        "from graph_grad.SUPPORTED_OPS) actually landed in the block"
    )

    x = rng.standard_normal((64, 64)).astype(np.float32)
    target_data = rng.standard_normal((64, 64)).astype(np.float32)
    losses = []
    trained = lora.train_lora(
        quantized,
        adapter,
        "X",
        "Y",
        target_data=[target_data],
        calibration_data=[{"X": x}],
        num_iterations=300,
        learning_rate=1e-2,
        losses=losses,
    )
    onnx.checker.check_model(trained)
    assert losses[-1] < losses[0]

    # The dequant chain itself is untouched in the returned (deployed)
    # model -- the fold only ever affected the internal step graph.
    assert any(n.op_type == "Cast" for n in trained.graph.node)
    codes_before = {
        t.name: t.raw_data
        for t in quantized.graph.initializer
        if t.data_type == onnx.TensorProto.UINT8
    }
    codes_after = {
        t.name: t.raw_data
        for t in trained.graph.initializer
        if t.data_type == onnx.TensorProto.UINT8
    }
    assert codes_before == codes_after


def test_discover_lora_blocks_finds_liveness_cut_boundaries():
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((6, 8)).astype(np.float32)
    w2 = rng.standard_normal((8, 4)).astype(np.float32)
    model = _model(
        """
        g (float[batch,6] X) => (float[batch,4] Y) {
          H = MatMul(X, W1)
          R = Relu(H)
          Y = MatMul(R, W2)
        }
        """,
        [_f32(w1, "W1"), _f32(w2, "W2")],
    )

    injected, adapter = lora.inject_lora(model, rank=2, seed=0)
    onnx.checker.check_model(injected)
    assert len(adapter.targets) == 2

    # Per-adapter blocks: one boundary per injected MatMul, matching where a
    # caller would have named train_lora's block_input_name/block_output_name
    # by hand.
    per_layer = lora.discover_lora_blocks(injected, adapter, max_targets_per_block=1)
    assert [(b.input_name, b.output_name, b.target_outputs) for b in per_layer] == [
        ("X", "H", ("H",)),
        ("H", "Y", ("Y",)),
    ]

    # max_targets_per_block=2 (the default) merges both into one block
    # spanning the whole graph -- no gap between them since Relu is in
    # graph_grad.SUPPORTED_OPS.
    merged = lora.discover_lora_blocks(injected, adapter)
    assert len(merged) == 1
    assert merged[0].input_name == "X"
    assert merged[0].output_name == "Y"
    assert merged[0].target_outputs == ("H", "Y")

    # The discovered blocks are directly usable by train_lora.
    x = rng.standard_normal((5, 6)).astype(np.float32)
    target_data = rng.standard_normal((5, 4)).astype(np.float32)
    losses = []
    trained = lora.train_lora(
        injected,
        adapter,
        merged[0].input_name,
        merged[0].output_name,
        target_data=[target_data],
        calibration_data=[{"X": x}],
        num_iterations=200,
        learning_rate=1e-2,
        losses=losses,
    )
    onnx.checker.check_model(trained)
    assert losses[-1] < losses[0]


def test_discover_lora_blocks_rejects_non_positive_max_targets_per_block():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((6, 8)).astype(np.float32)
    model = _model(
        """
        g (float[batch,6] X) => (float[batch,8] Y) {
          Y = MatMul(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    injected, adapter = lora.inject_lora(model, rank=2, seed=0)

    with pytest.raises(ValueError, match="max_targets_per_block"):
        lora.discover_lora_blocks(injected, adapter, max_targets_per_block=0)


def test_export_lora_adapter_round_trips_through_adapterformat(tmp_path):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 8)).astype(np.float32)
    model = _model(
        """
        g (float[batch,8] X) => (float[batch,8] Y) {
          Y = MatMul(X, W)
        }
        """,
        [_f32(w, "W")],
    )
    injected, adapter = lora.inject_lora(model, rank=3, seed=0)
    x = rng.standard_normal((16, 8)).astype(np.float32)
    target = rng.standard_normal((16, 8)).astype(np.float32)
    trained = lora.train_lora(
        injected,
        adapter,
        "X",
        "Y",
        target_data=[target],
        calibration_data=[{"X": x}],
        num_iterations=50,
    )

    path = tmp_path / "adapter.onnx_adapter"
    lora.export_lora_adapter(
        trained, adapter, str(path), adapter_version=3, model_version=7
    )

    fmt = ort.AdapterFormat.read_adapter(str(path))
    assert fmt.get_adapter_version() == 3
    assert fmt.get_model_version() == 7

    params = fmt.get_parameters()
    initializer_map = {t.name: t for t in trained.graph.initializer}
    for name in adapter.parameter_names():
        expected = onnx.numpy_helper.to_array(initializer_map[name])
        np.testing.assert_array_equal(params[name].numpy(), expected)


def test_export_lora_adapter_missing_onnxruntime_raises_clear_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("No module named 'onnxruntime'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install onnxruntime"):
        lora.export_lora_adapter(
            onnx.ModelProto(), lora.LoraAdapter(), "unused.onnx_adapter"
        )


def test_export_lora_adapter_old_onnxruntime_without_adapterformat_raises_clear_error(
    monkeypatch,
):
    fake_ort = types.SimpleNamespace(__version__="1.15.0")  # no AdapterFormat
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(ImportError, match="AdapterFormat"):
        lora.export_lora_adapter(
            onnx.ModelProto(), lora.LoraAdapter(), "unused.onnx_adapter"
        )
