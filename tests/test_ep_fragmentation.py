"""Tests for ``onnxsim._ep_fragmentation.estimate_fragmentation`` and the
``estimate_webgpu_islands``/``estimate_webnn_islands`` wrappers built on it.
"""

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import parser

from onnxsim._ep_fragmentation import estimate_fragmentation
from onnxsim.webgpu_target import estimate_webgpu_islands
from onnxsim.webnn_target import estimate_webnn_islands


def _model(body, initializer=(), opset=17, ir_version=10, extra_opset_imports=""):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}{extra_opset_imports}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _chain_graph():
    # a -> b -> c -> d, a plain linear chain of 4 nodes.
    model = _model(
        """
        g (float[2] x) => (float[2] y)
        {
          a = Identity(x)
          b = Identity(a)
          c = Identity(b)
          y = Identity(c)
        }
        """
    )
    return model.graph


def test_estimate_fragmentation_no_flags_is_one_island():
    graph = _chain_graph()
    report = estimate_fragmentation(graph, set())
    assert report.island_count == 1
    assert report.boundary_edge_count == 0
    assert report.flagged_node_names == []


def test_estimate_fragmentation_middle_node_splits_into_two_islands():
    graph = _chain_graph()  # nodes: 0=a, 1=b, 2=c, 3=y
    report = estimate_fragmentation(graph, {1})  # flag "b"
    # {a} and {c, y} are now disconnected from each other.
    assert report.island_count == 2
    assert report.boundary_edge_count == 2  # a->b and b->c
    assert report.flagged_node_names == ["b"]


def test_estimate_fragmentation_flags_at_both_ends_leave_one_island():
    graph = _chain_graph()  # 0=a, 1=b, 2=c, 3=y
    report = estimate_fragmentation(graph, {0, 3})  # flag "a" and "y"
    # only {b, c} remain, still directly connected to each other.
    assert report.island_count == 1
    assert report.boundary_edge_count == 2  # a->b and c->y
    assert report.flagged_node_names == ["a", "y"]


def test_estimate_fragmentation_all_flagged_has_no_islands():
    graph = _chain_graph()
    report = estimate_fragmentation(graph, {0, 1, 2, 3})
    assert report.island_count == 0
    assert report.boundary_edge_count == 0


def test_estimate_fragmentation_independent_branches_are_separate_islands():
    model = _model(
        """
        g (float[2] x1, float[2] x2) => (float[2] y1, float[2] y2)
        {
          y1 = Identity(x1)
          y2 = Identity(x2)
        }
        """
    )
    report = estimate_fragmentation(model.graph, set())
    assert report.island_count == 2
    assert report.boundary_edge_count == 0


def test_estimate_webgpu_islands_attention_with_mask_fragments_graph():
    B, S, H, NH = 2, 5, 32, 4
    rng = np.random.default_rng(0)
    wqkv = onnx.numpy_helper.from_array(
        (rng.standard_normal((H, H * 3)) * 0.1).astype(np.float32), "wqkv"
    )
    bqkv = onnx.numpy_helper.from_array(
        (rng.standard_normal(H * 3) * 0.1).astype(np.float32), "bqkv"
    )
    model = _model(
        f"""
        g (float[{B},{S},{H}] x, int32[{B}] mask_index) => (float[{B},{S},{H}] y)
        {{
          pre = Identity(x)
          attn = com.microsoft.Attention<num_heads = {NH}, qkv_hidden_sizes = [{H}, {H}, {H}]>(pre, wqkv, bqkv, mask_index)
          y = Identity(attn)
        }}
        """,
        [wqkv, bqkv],
        extra_opset_imports=', "com.microsoft": 1',
    )
    report = estimate_webgpu_islands(model)
    # {pre} and {y} are split apart by the flagged Attention node.
    assert report.island_count == 2
    assert report.boundary_edge_count == 2
    assert len(report.flagged_node_names) == 1


def test_estimate_webgpu_islands_clean_attention_is_one_island():
    B, S, H, NH = 2, 5, 32, 4
    rng = np.random.default_rng(0)
    wqkv = onnx.numpy_helper.from_array(
        (rng.standard_normal((H, H * 3)) * 0.1).astype(np.float32), "wqkv"
    )
    bqkv = onnx.numpy_helper.from_array(
        (rng.standard_normal(H * 3) * 0.1).astype(np.float32), "bqkv"
    )
    model = _model(
        f"""
        g (float[{B},{S},{H}] x) => (float[{B},{S},{H}] y)
        {{
          pre = Identity(x)
          attn = com.microsoft.Attention<num_heads = {NH}, qkv_hidden_sizes = [{H}, {H}, {H}]>(pre, wqkv, bqkv)
          y = Identity(attn)
        }}
        """,
        [wqkv, bqkv],
        extra_opset_imports=', "com.microsoft": 1',
    )
    report = estimate_webgpu_islands(model)
    assert report.island_count == 1
    assert report.boundary_edge_count == 0
    assert report.flagged_node_names == []


def test_estimate_webnn_islands_non_constant_reshape_fragments_graph():
    model = _model(
        """
        g (float[2,3] x, float[6] like) => (float[?] y)
        {
          pre = Identity(x)
          shape = Shape(like)
          reshaped = Reshape(pre, shape)
          y = Identity(reshaped)
        }
        """
    )
    report = estimate_webnn_islands(model)
    # pre, shape, and y each only touch the flagged Reshape node, not each
    # other -- three separate one-node islands, three boundary edges.
    assert report.island_count == 3
    assert report.boundary_edge_count == 3
    assert len(report.flagged_node_names) == 1
