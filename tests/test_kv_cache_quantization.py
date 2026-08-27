"""Tests for ``onnxsim.quantize_kv_cache`` -- see ``onnxsim/kv_cache_quantization.py``
for the technique (static, per-channel INT8 quantization of a decoder's
``Concat(past, new, axis=seq)`` KV-cache stream).
"""

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _vi(name, shape, dtype=onnx.TensorProto.FLOAT):
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def _kv_cache_model(batch=1, heads=2, head_dim=4, opset=13, symbolic_seq=True):
    # new_key is deliberately routed through an Identity node rather than
    # being a bare graph input: a real exported decoder's "new" K/V is
    # always a freshly computed activation (the current step's projection
    # output), never a raw model input -- and keeping the test model that
    # way avoids a genuine ambiguity the matcher can't resolve from
    # structure alone (nothing distinguishes "the persistent cache" from
    # "this step's fresh token" if *both* Concat operands are plain graph
    # inputs with a single consumer).
    seq_past = "seq_past" if symbolic_seq else 3
    seq_present = "seq_present" if symbolic_seq else 4
    nodes = [
        onnx.helper.make_node("Identity", ["new_key_raw"], ["new_key"]),
        onnx.helper.make_node(
            "Concat", ["past_key", "new_key"], ["present_key"], axis=2
        ),
        onnx.helper.make_node("ReduceSum", ["present_key"], ["summary"], keepdims=0),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [
            _vi("past_key", [batch, heads, seq_past, head_dim]),
            _vi("new_key_raw", [batch, heads, 1, head_dim]),
        ],
        [
            _vi("present_key", [batch, heads, seq_present, head_dim]),
            _vi("summary", []),
        ],
        [],
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=8
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-6)


def test_kv_cache_quantization_changes_input_and_output_to_int8():
    model = _kv_cache_model()
    q = onnxsim.quantize_kv_cache(model, num_samples=4, seed=0)
    onnx.checker.check_model(q)

    past_vi = next(i for i in q.graph.input if i.name == "past_key")
    present_vi = next(o for o in q.graph.output if o.name == "present_key")
    assert past_vi.type.tensor_type.elem_type == onnx.TensorProto.INT8
    assert present_vi.type.tensor_type.elem_type == onnx.TensorProto.INT8
    # new_key_raw/the Identity's "new_key" output are untouched float --
    # only Concat's own operand is rewired to a freshly quantized tensor.
    new_vi = next(i for i in q.graph.input if i.name == "new_key_raw")
    assert new_vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    quantize_node = next(n for n in q.graph.node if n.op_type == "QuantizeLinear")
    assert quantize_node.input[0] == "new_key"


def test_kv_cache_quantization_inserts_quantize_dequantize_around_concat():
    model = _kv_cache_model()
    q = onnxsim.quantize_kv_cache(model, num_samples=4, seed=0)
    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("QuantizeLinear") == 1
    assert op_types.count("DequantizeLinear") == 1
    assert "Concat" in op_types
    # ReduceSum (the "attention math" stand-in) must consume the
    # dequantized float tensor, not the raw int8 Concat output.
    reduce_node = next(n for n in q.graph.node if n.op_type == "ReduceSum")
    dequant_node = next(n for n in q.graph.node if n.op_type == "DequantizeLinear")
    assert reduce_node.input[0] == dequant_node.output[0]


def test_kv_cache_quantization_two_step_round_trip_close_to_float():
    # Mirrors how tools/onnx-deploy's KvCachePipeline actually drives a
    # decoder: step 0 starts from an empty cache, step 1 feeds step 0's own
    # "present" output straight back in as "past" -- for the quantized
    # model, that output is already INT8, exactly as a real pipeline would
    # receive it, with no manual conversion by the caller.
    #
    # Compared against the *dequantized cache tensor itself* (all 24
    # elements), not the downstream ReduceSum scalar: summing so few
    # elements to one number lets ordinary cancellation inflate the
    # relative error on the sum far past the actual per-element
    # quantization error, which is what this test is really checking.
    batch, heads, head_dim = 1, 2, 6
    model = _kv_cache_model(batch=batch, heads=heads, head_dim=head_dim)
    q = onnxsim.quantize_kv_cache(model, num_samples=64, seed=1)
    dequant_node = next(n for n in q.graph.node if n.op_type == "DequantizeLinear")
    q_probe = onnx.ModelProto()
    q_probe.CopyFrom(q)
    q_probe.graph.output.append(onnx.ValueInfoProto(name=dequant_node.output[0]))

    rng = np.random.default_rng(3)
    empty_past = np.zeros((batch, heads, 0, head_dim), dtype=np.float32)
    new0 = rng.standard_normal((batch, heads, 1, head_dim)).astype(np.float32)
    new1 = rng.standard_normal((batch, heads, 1, head_dim)).astype(np.float32)

    float_present0, _ = _run(model, {"past_key": empty_past, "new_key_raw": new0})
    float_present1, _ = _run(model, {"past_key": float_present0, "new_key_raw": new1})

    empty_past_q = np.zeros((batch, heads, 0, head_dim), dtype=np.int8)
    q_present0, _ = _run(q, {"past_key": empty_past_q, "new_key_raw": new0})
    assert q_present0.dtype == np.int8
    q_present1, _, q_present1_f = _run(
        q_probe, {"past_key": q_present0, "new_key_raw": new1}
    )
    assert q_present1.dtype == np.int8

    # A generous bound, not a tight one: with a small (64-sample) random
    # calibration set, this step's own random draw occasionally lands an
    # element past the calibrated per-channel max, clipping it -- a real,
    # expected property of any calibrated int8 scheme, not a bug in this
    # module. Matches the tolerance other onnxsim quantizers use for a
    # small random-data end-to-end check (e.g. AQLM's own 0.3).
    assert _rel_l2(float_present1, q_present1_f) < 0.2


def test_kv_cache_quantization_declines_when_past_has_other_consumers():
    nodes = [
        onnx.helper.make_node("Identity", ["new_key_raw"], ["new_key"]),
        onnx.helper.make_node(
            "Concat", ["past_key", "new_key"], ["present_key"], axis=2
        ),
        onnx.helper.make_node("ReduceSum", ["present_key"], ["summary"], keepdims=0),
        # A second, direct consumer of past_key -- the pattern must decline
        # rather than silently break this other use.
        onnx.helper.make_node("ReduceSum", ["past_key"], ["past_summary"], keepdims=0),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "g",
        [_vi("past_key", [1, 2, 3, 4]), _vi("new_key_raw", [1, 2, 1, 4])],
        [
            _vi("present_key", [1, 2, 4, 4]),
            _vi("summary", []),
            _vi("past_summary", []),
        ],
        [],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8
    )
    q = onnxsim.quantize_kv_cache(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_kv_cache_quantization_noop_without_kv_cache_pattern():
    nodes = [onnx.helper.make_node("Relu", ["x"], ["y"])]
    graph = onnx.helper.make_graph(
        nodes, "g", [_vi("x", [4, 4])], [_vi("y", [4, 4])], []
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8
    )
    result = onnxsim.quantize_kv_cache(model)
    assert result.SerializeToString() == model.SerializeToString()


def test_kv_cache_quantization_noop_below_opset13():
    model = _kv_cache_model(opset=12)
    result = onnxsim.quantize_kv_cache(model)
    assert result.SerializeToString() == model.SerializeToString()
