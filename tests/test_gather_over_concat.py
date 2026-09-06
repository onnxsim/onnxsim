"""Tests for the opt-in ``rewrite_gather_over_concat`` pass.

Every model is built with the ONNX text format parser (``onnx.parser``) per
CLAUDE.md's convention for this repo's tests. Rewriting models are run
through ``onnxsim.simplify(..., extra_optimizers=["rewrite_gather_over_concat"])``,
which numerically equivalence-checks the rewritten graph against the
original ``Concat`` + ``Gather`` pair (via onnxruntime, or the onnx
reference evaluator when onnxruntime is not installed) using concrete
``input_data`` -- see ``tests/test_gathernd_to_gather.py``'s
``_simplify_and_check`` helper, which this mirrors. Since ``extra_optimizers``
runs on top of onnxsim's default fuse/elimination set (not instead of it),
a ``Concat`` left with no remaining consumers after the rewrite is expected
to disappear from the final graph too.
"""

import collections

import numpy as np
from onnx import parser

import onnxsim


def _model(body, opset=17, ir_version=10):
    return parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def _simplify_and_check(model, input_data, check_n=1):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
        extra_optimizers=["rewrite_gather_over_concat"],
    )
    assert check_ok, "rewritten graph failed onnxsim's equivalence check"
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    return sim_model, op_types


def _simplify_and_assert_declined(model, input_data, check_n=1):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
        extra_optimizers=["rewrite_gather_over_concat"],
    )
    assert check_ok
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert op_types["Concat"] == 1, op_types
    return sim_model, op_types


# --------------------------------------------------------------------------- #
# All (constant) indices land in one middle input's segment -- the Concat
# (and its now-unreferenced first/last inputs) disappears entirely once the
# Gather is rewired straight onto that one segment.
# --------------------------------------------------------------------------- #


def test_single_segment_selects_middle_input():
    body = """
    agraph (float[2,4] x0, float[3,4] x1, float[5,4] x2) => (float[2,4] Y)
    <int64[2] idx = {3, 4}>
    {
      c = Concat <axis=0> (x0, x1, x2)
      Y = Gather <axis=0> (c, idx)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(0)
    input_data = {
        "x0": rng.randn(2, 4).astype(np.float32),
        "x1": rng.randn(3, 4).astype(np.float32),
        "x2": rng.randn(5, 4).astype(np.float32),
    }
    sim_model, op_types = _simplify_and_check(model, input_data)
    assert op_types["Concat"] == 0, op_types
    assert op_types["Gather"] == 1, op_types


# --------------------------------------------------------------------------- #
# Negative index -- must be normalized (wrapped into [0, total)) before
# resolving which segment it lands in and shifting it to a local offset.
# --------------------------------------------------------------------------- #


def test_negative_index_selects_last_input():
    body = """
    agraph (float[2,4] x0, float[3,4] x1) => (float[1,4] Y)
    <int64[1] idx = {-1}>
    {
      c = Concat <axis=0> (x0, x1)
      Y = Gather <axis=0> (c, idx)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(1)
    input_data = {
        "x0": rng.randn(2, 4).astype(np.float32),
        "x1": rng.randn(3, 4).astype(np.float32),
    }
    sim_model, op_types = _simplify_and_check(model, input_data)
    assert op_types["Concat"] == 0, op_types
    assert op_types["Gather"] == 1, op_types


# --------------------------------------------------------------------------- #
# The shared axis need not be 0.
# --------------------------------------------------------------------------- #


def test_non_leading_axis():
    body = """
    agraph (float[4,2] x0, float[4,3] x1) => (float[4,2] Y)
    <int64[2] idx = {3, 4}>
    {
      c = Concat <axis=1> (x0, x1)
      Y = Gather <axis=1> (c, idx)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(2)
    input_data = {
        "x0": rng.randn(4, 2).astype(np.float32),
        "x1": rng.randn(4, 3).astype(np.float32),
    }
    sim_model, op_types = _simplify_and_check(model, input_data)
    assert op_types["Concat"] == 0, op_types
    assert op_types["Gather"] == 1, op_types


# --------------------------------------------------------------------------- #
# Declined: the constant indices span more than one Concat input segment --
# a single Gather can only ever be rewired onto one upstream value.
# --------------------------------------------------------------------------- #


def test_declined_indices_span_multiple_segments():
    body = """
    agraph (float[2,4] x0, float[3,4] x1) => (float[2,4] Y)
    <int64[2] idx = {1, 2}>
    {
      c = Concat <axis=0> (x0, x1)
      Y = Gather <axis=0> (c, idx)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(3)
    input_data = {
        "x0": rng.randn(2, 4).astype(np.float32),
        "x1": rng.randn(3, 4).astype(np.float32),
    }
    _simplify_and_assert_declined(model, input_data)


# --------------------------------------------------------------------------- #
# Declined: indices are not a compile-time constant.
# --------------------------------------------------------------------------- #


def test_declined_dynamic_indices():
    body = """
    agraph (float[2,4] x0, float[3,4] x1, int64[1] idx) => (float[1,4] Y)
    {
      c = Concat <axis=0> (x0, x1)
      Y = Gather <axis=0> (c, idx)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(4)
    input_data = {
        "x0": rng.randn(2, 4).astype(np.float32),
        "x1": rng.randn(3, 4).astype(np.float32),
        "idx": np.array([2], dtype=np.int64),
    }
    _simplify_and_assert_declined(model, input_data)


# --------------------------------------------------------------------------- #
# Declined: Gather's axis differs from Concat's -- outside this pass's scope
# (see the header comment's declines list).
# --------------------------------------------------------------------------- #


def test_declined_axis_mismatch():
    body = """
    agraph (float[2,4] x0, float[3,4] x1) => (float[5,1] Y)
    <int64[1] idx = {0}>
    {
      c = Concat <axis=0> (x0, x1)
      Y = Gather <axis=1> (c, idx)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(5)
    input_data = {
        "x0": rng.randn(2, 4).astype(np.float32),
        "x1": rng.randn(3, 4).astype(np.float32),
    }
    _simplify_and_assert_declined(model, input_data)
