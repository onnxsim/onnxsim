"""Per-ONNX-op coverage for the AX650, checked offline.

`scripts/axera/op_coverage.py` classifies every op in the ai.onnx default
domain against the vendor's published support list, and counts a model's nodes
by whether the NPU can take them. Neither needs Docker or a card, so unlike
the rest of the Axera suite this runs on a stock CI runner.

The check that earns its keep is the first one: an entry in the support list
that is not a real ONNX op name matches nothing, so it silently understates
coverage. That is not hypothetical -- `TopK` was on the list spelled `Topk`,
and the real-hardware sweep had already built a `TopK` node successfully.
"""

import os
import sys

import numpy as np
import onnx
import onnx.defs
from onnx import TensorProto, helper, numpy_helper

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import op_coverage  # noqa: E402
import pulsar2_ops  # noqa: E402


def test_every_supported_op_is_a_real_onnx_op_or_a_named_vendor_one():
    """A support-list entry that is not an ai.onnx op name can never match a
    node, so it costs coverage silently. The only entries allowed to do that
    are the vendor's own fused ops, named explicitly."""
    onnx_ops = set(op_coverage.onnx_op_types())
    stray = set(pulsar2_ops.AX650_SUPPORTED_OPS) - onnx_ops
    assert stray <= op_coverage.VENDOR_NATIVE | {"Topk"}, stray


def test_topk_is_matched_under_its_onnx_spelling():
    """Axera's list spells it `Topk`; ONNX's operator is `TopK`. The
    real-hardware sweep built a `TopK` node, so classifying one as unsupported
    was a misclassification rather than a missing capability."""
    assert "TopK" in pulsar2_ops.AX650_SUPPORTED_OPS
    assert op_coverage.classify("TopK") == op_coverage.ELIGIBLE
    assert not pulsar2_ops.unsupported_on_ax650(
        helper.make_model(
            helper.make_graph(
                [helper.make_node("TopK", ["x", "k"], ["v", "i"], largest=1, sorted=1)],
                "g",
                [
                    helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8]),
                    helper.make_tensor_value_info("k", TensorProto.INT64, [1]),
                ],
                [
                    helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 3]),
                    helper.make_tensor_value_info("i", TensorProto.INT64, [1, 3]),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
        )
    )


def test_the_table_partitions_the_whole_operator_set():
    """Every ai.onnx op lands in exactly one bucket, and the buckets add up."""
    ops = op_coverage.onnx_op_types()
    table = op_coverage.coverage_table()
    assert sum(len(v) for v in table.values()) == len(ops)
    seen = [op for bucket in table.values() for op in bucket]
    assert sorted(seen) == ops
    assert len(set(seen)) == len(seen)
    # The NPU takes a substantial minority of ONNX; a sweeping claim either way
    # would be wrong, and this pins which.
    assert 0.3 < len(table[op_coverage.ELIGIBLE]) / len(ops) < 0.6


def test_model_usage_counts_nodes_without_loading_weights(tmp_path):
    """Op types do not depend on weights, and these graphs routinely carry
    hundreds of megabytes of external data, so the inventory must not need
    them."""
    weight = numpy_helper.from_array(np.zeros((4, 4, 3, 3), np.float32), "w")
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["x", "w"], ["c"], kernel_shape=[3, 3], pads=[1] * 4
            ),
            helper.make_node("Relu", ["c"], ["r"]),
            helper.make_node("Shape", ["r"], ["s"]),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])],
        [helper.make_tensor_value_info("s", TensorProto.INT64, [4])],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = str(tmp_path / "m.onnx")
    onnx.save(
        model,
        path,
        save_as_external_data=True,
        location="m.onnx.data",
        size_threshold=0,
    )
    os.remove(str(tmp_path / "m.onnx.data"))  # the weights are gone; ops remain

    total, eligible, blocked = op_coverage.summarise(path)
    assert total == 3
    assert eligible == 2  # Conv and Relu
    assert dict(blocked) == {"Shape": 1}
