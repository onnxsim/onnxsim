"""Tests for the integer ``BitwiseXor`` -> ``And``/``Or``/``Sub`` rewrite.

The rewrite exists because the Instant-NGP hash grid that nerfstudio-style NeRF
encoders are built around uses ``torch.bitwise_xor``, which lowers to ONNX
``BitwiseXor`` (opset 18). onnxruntime runs that fine, but Core ML has no MIL
bitwise op, Qualcomm QNN HTP has no known kernel for it, and TFLite's
``BITWISE_XOR`` is a builtin only (unavailable to the int8/uint8 delegate graphs
this repo builds) with no integer ``BitwiseAnd``/``BitwiseOr`` to fall back to.

The identity is ``a ^ b == (a | b) - (a & b)``, which is exact for every integer
width because ``a | b >= a & b`` bitwise -- every AND bit is also an OR bit -- so
the subtraction never borrows and cannot wrap.

The numeric assertions here use ``assert_array_equal``, not ``allclose``: this is
a bit-exact rewrite and a tolerance would hide a wrong result.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from onnxsim.bitwise_decompose import decompose_bitwise_xor

# Every integer type ONNX permits for the bitwise ops. BitwiseXor on a float
# would be a malformed model, and rewriting it would turn a NaN payload into a
# number, so the pass refuses anything outside this set.
INTEGER_TYPES = [
    ("uint8", TensorProto.UINT8),
    ("int8", TensorProto.INT8),
    ("uint16", TensorProto.UINT16),
    ("int16", TensorProto.INT16),
    ("uint32", TensorProto.UINT32),
    ("int32", TensorProto.INT32),
    ("uint64", TensorProto.UINT64),
    ("int64", TensorProto.INT64),
]


def _xor_model(dtype, a, b, opset=18, extra_attrs=None):
    node = helper.make_node("BitwiseXor", ["a", "b"], ["c"], **(extra_attrs or {}))
    graph = helper.make_graph(
        [node],
        "g",
        [
            helper.make_tensor_value_info("a", dtype, list(a.shape)),
            helper.make_tensor_value_info("b", dtype, list(b.shape)),
        ],
        [helper.make_tensor_value_info("c", dtype, list(a.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _run(model, a, b):
    import onnxruntime as ort

    return ort.InferenceSession(model.SerializeToString()).run(None, {"a": a, "b": b})[
        0
    ]


def _samples(dtype_name, dtype_enum):
    """Random values plus the extremes, for one integer type."""
    rng = np.random.default_rng(0)
    np_dt = np.dtype(dtype_name)
    info = np.iinfo(np_dt)
    # Stay inside the non-negative range so the bound is representable as a
    # numpy integer; signed extremes are appended explicitly.
    top = min(int(info.max), np.iinfo(np.int64).max - 1)
    specials = [0, 1, int(info.max)]
    if np_dt.kind == "i":
        specials += [-1, int(info.min)]
    a = np.concatenate(
        [rng.integers(0, top + 1, 2000).astype(np_dt), np.array(specials, np_dt)]
    )
    b = np.concatenate(
        [rng.integers(0, top + 1, 2000).astype(np_dt), np.array(specials[::-1], np_dt)]
    )
    return a, b


@pytest.mark.parametrize(
    "dtype_name,dtype_enum", INTEGER_TYPES, ids=[t for t, _ in INTEGER_TYPES]
)
def test_rewrite_is_bit_exact_for_every_integer_type(dtype_name, dtype_enum):
    a, b = _samples(dtype_name, dtype_enum)
    model = _xor_model(dtype_enum, a, b)

    rewritten = decompose_bitwise_xor(model)
    assert [n.op_type for n in rewritten.graph.node] == [
        "BitwiseOr",
        "BitwiseAnd",
        "Sub",
    ]
    # The rewrite must be a valid model, including for the strict checker that
    # also runs shape inference over the new intermediates.
    onnx.checker.check_model(rewritten, full_check=True)

    assert_array_equal = np.testing.assert_array_equal
    assert_array_equal(_run(rewritten, a, b), _run(model, a, b))


def test_all_ones_and_all_zeros_edges():
    """255^255, 0^255, 255^0 -- the borrow-heavy corners of the subtraction."""
    a = np.array([255, 0, 255, 0], np.uint8)
    b = np.array([255, 255, 0, 0], np.uint8)
    model = _xor_model(TensorProto.UINT8, a, b)
    rewritten = decompose_bitwise_xor(model)
    expected = np.array([0, 255, 255, 0], np.uint8)
    np.testing.assert_array_equal(_run(model, a, b), expected)
    np.testing.assert_array_equal(_run(rewritten, a, b), expected)


def test_model_without_bitwise_xor_is_returned_by_identity():
    graph = helper.make_graph(
        [helper.make_node("Relu", ["a"], ["c"])],
        "g",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    assert decompose_bitwise_xor(model) is model


def test_float_operands_are_left_alone():
    """A BitwiseXor on a float type is malformed, but rewriting it would turn a
    NaN bit pattern into a number -- so the pass must refuse it and leave the
    error for the target to report."""
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("BitwiseXor", ["a", "b"], ["c"])],
            "g",
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("c", TensorProto.FLOAT, [2])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10
    assert decompose_bitwise_xor(model) is model


def test_direction_attribute_is_left_alone():
    """A ``direction`` attribute must not be silently reinterpreted.

    ``direction`` is not part of ``BitwiseXor``'s schema -- ONNX rejects it, and
    the pass leaves such a node alone rather than assuming the bitwise reading.
    The model here is deliberately malformed (built without ``check_model``) so
    the guard is what is under test, not the schema.
    """
    graph = helper.make_graph(
        [helper.make_node("BitwiseXor", ["a", "b"], ["c"], direction="LOGICAL")],
        "g",
        [
            helper.make_tensor_value_info("a", TensorProto.UINT8, [2]),
            helper.make_tensor_value_info("b", TensorProto.UINT8, [2]),
        ],
        [helper.make_tensor_value_info("c", TensorProto.UINT8, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    assert decompose_bitwise_xor(model) is model


def test_rewrite_does_not_mutate_its_input():
    a, b = _samples("int32", TensorProto.INT32)
    model = _xor_model(TensorProto.INT32, a, b)
    before = [(n.op_type, list(n.output)) for n in model.graph.node]
    decompose_bitwise_xor(model)
    assert [(n.op_type, list(n.output)) for n in model.graph.node] == before


def test_two_bitwise_xors_in_one_graph_both_lower():
    a, b = _samples("int64", TensorProto.INT64)
    graph = helper.make_graph(
        [
            helper.make_node("BitwiseXor", ["a", "b"], ["t"]),
            helper.make_node("BitwiseXor", ["t", "b"], ["c"]),
        ],
        "g",
        [
            helper.make_tensor_value_info("a", TensorProto.INT64, [len(a)]),
            helper.make_tensor_value_info("b", TensorProto.INT64, [len(b)]),
        ],
        [helper.make_tensor_value_info("c", TensorProto.INT64, [len(a)])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10

    rewritten = decompose_bitwise_xor(model)
    assert "BitwiseXor" not in [n.op_type for n in rewritten.graph.node]
    assert [n.op_type for n in rewritten.graph.node].count("Sub") == 2
    # Intermediate names must be distinct or the second rewrite would clobber
    # the first one's tensor.
    outputs = [o for n in rewritten.graph.node for o in n.output]
    assert len(outputs) == len(set(outputs))
    onnx.checker.check_model(rewritten, full_check=True)
    np.testing.assert_array_equal(_run(rewritten, a, b), _run(model, a, b))
