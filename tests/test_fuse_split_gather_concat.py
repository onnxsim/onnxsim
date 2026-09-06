"""Tests for the default-on ``fuse_split_gather_concat`` pass.

Unlike ``rewrite_gather_over_concat`` (its mirror-image sibling, opt-in and
constant-indices-only), this fusion is an unconditional identity -- it never
needs ``idx`` to be a compile-time constant, or any shape to be statically
known (beyond a rank when an ``axis`` attribute is negative) -- so it runs by
default and every model below is simplified with a plain
``onnxsim.simplify(...)`` call. Every model is built with the ONNX text
format parser (``onnx.parser``) per CLAUDE.md's convention for this repo's
tests, and equivalence-checked against concrete ``input_data`` the same way
as the other migrated test files.
"""

import collections

import numpy as np
from onnx import parser

import onnxsim


def _model(body, opset=18, ir_version=10):
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
    )
    assert check_ok, "fused graph failed onnxsim's equivalence check"
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    return sim_model, op_types


def _simplify_and_assert_declined(model, input_data, check_n=1):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
    )
    assert check_ok
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert op_types["Split"] == 1, op_types
    assert op_types["Concat"] == 1, op_types
    return sim_model, op_types


# --------------------------------------------------------------------------- #
# The textbook shape: Split(idx) into k pieces, Gather the same x with each
# (same axis), Concat the results back along that axis -- collapses to one
# Gather(x, idx). idx is a *dynamic* graph input throughout this file: this
# fusion never needs it to be constant.
# --------------------------------------------------------------------------- #


def test_split_gather_concat_collapses_to_one_gather():
    body = """
    agraph (float[10,4] x, int64[6] idx) => (float[6,4] Y)
    <int64[3] splits = {2, 2, 2}>
    {
      s0, s1, s2 = Split <axis=0> (idx, splits)
      g0 = Gather <axis=0> (x, s0)
      g1 = Gather <axis=0> (x, s1)
      g2 = Gather <axis=0> (x, s2)
      Y = Concat <axis=0> (g0, g1, g2)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(0)
    input_data = {
        "x": rng.randn(10, 4).astype(np.float32),
        "idx": rng.randint(0, 10, size=(6,)).astype(np.int64),
    }
    sim_model, op_types = _simplify_and_check(model, input_data)
    assert op_types["Split"] == 0, op_types
    assert op_types["Concat"] == 0, op_types
    assert op_types["Gather"] == 1, op_types


# --------------------------------------------------------------------------- #
# The shared axis need not be 0, and Gather's axis need not equal Split's --
# only ca == ga + sa (in the output's own frame) is required.
# --------------------------------------------------------------------------- #


def test_non_leading_axis():
    body = """
    agraph (float[3,10] x, int64[6] idx) => (float[3,6] Y)
    <int64[2] splits = {4, 2}>
    {
      s0, s1 = Split <axis=0> (idx, splits)
      g0 = Gather <axis=1> (x, s0)
      g1 = Gather <axis=1> (x, s1)
      Y = Concat <axis=1> (g0, g1)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(1)
    input_data = {
        "x": rng.randn(3, 10).astype(np.float32),
        "idx": rng.randint(0, 10, size=(6,)).astype(np.int64),
    }
    sim_model, op_types = _simplify_and_check(model, input_data)
    assert op_types["Split"] == 0, op_types
    assert op_types["Concat"] == 0, op_types
    assert op_types["Gather"] == 1, op_types


# --------------------------------------------------------------------------- #
# Negative-index values are irrelevant here (unlike rewrite_gather_over_concat,
# this fusion never inspects idx's values), but it should still fire.
# --------------------------------------------------------------------------- #


def test_negative_indices_still_fires():
    body = """
    agraph (float[10,4] x, int64[6] idx) => (float[6,4] Y)
    <int64[3] splits = {2, 2, 2}>
    {
      s0, s1, s2 = Split <axis=0> (idx, splits)
      g0 = Gather <axis=0> (x, s0)
      g1 = Gather <axis=0> (x, s1)
      g2 = Gather <axis=0> (x, s2)
      Y = Concat <axis=0> (g0, g1, g2)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(2)
    input_data = {
        "x": rng.randn(10, 4).astype(np.float32),
        "idx": rng.randint(-10, 10, size=(6,)).astype(np.int64),
    }
    sim_model, op_types = _simplify_and_check(model, input_data)
    assert op_types["Gather"] == 1, op_types


# --------------------------------------------------------------------------- #
# Declined: the Gathers don't all share the same data input.
# --------------------------------------------------------------------------- #


def test_declined_different_data_inputs():
    body = """
    agraph (float[10,4] x0, float[10,4] x1, int64[6] idx) => (float[6,4] Y)
    <int64[3] splits = {2, 2, 2}>
    {
      s0, s1, s2 = Split <axis=0> (idx, splits)
      g0 = Gather <axis=0> (x0, s0)
      g1 = Gather <axis=0> (x1, s1)
      g2 = Gather <axis=0> (x0, s2)
      Y = Concat <axis=0> (g0, g1, g2)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(3)
    input_data = {
        "x0": rng.randn(10, 4).astype(np.float32),
        "x1": rng.randn(10, 4).astype(np.float32),
        "idx": rng.randint(0, 10, size=(6,)).astype(np.int64),
    }
    _simplify_and_assert_declined(model, input_data)


# --------------------------------------------------------------------------- #
# Declined: Concat's inputs are reordered relative to Split's outputs.
# --------------------------------------------------------------------------- #


def test_declined_reordered_pieces():
    body = """
    agraph (float[10,4] x, int64[6] idx) => (float[6,4] Y)
    <int64[3] splits = {2, 2, 2}>
    {
      s0, s1, s2 = Split <axis=0> (idx, splits)
      g0 = Gather <axis=0> (x, s0)
      g1 = Gather <axis=0> (x, s1)
      g2 = Gather <axis=0> (x, s2)
      Y = Concat <axis=0> (g1, g0, g2)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(4)
    input_data = {
        "x": rng.randn(10, 4).astype(np.float32),
        "idx": rng.randint(0, 10, size=(6,)).astype(np.int64),
    }
    _simplify_and_assert_declined(model, input_data)


# --------------------------------------------------------------------------- #
# Declined: Concat only consumes a strict subset of Split's outputs.
# --------------------------------------------------------------------------- #


def test_declined_partial_split_consumption():
    body = """
    agraph (float[10,4] x, int64[6] idx) => (float[4,4] Y, float[2,4] Z)
    <int64[3] splits = {2, 2, 2}>
    {
      s0, s1, s2 = Split <axis=0> (idx, splits)
      g0 = Gather <axis=0> (x, s0)
      g1 = Gather <axis=0> (x, s1)
      Y = Concat <axis=0> (g0, g1)
      Z = Gather <axis=0> (x, s2)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(5)
    input_data = {
        "x": rng.randn(10, 4).astype(np.float32),
        "idx": rng.randint(0, 10, size=(6,)).astype(np.int64),
    }
    sim_model, check_ok = onnxsim.simplify(model, check_n=1, input_data=input_data)
    assert check_ok
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert op_types["Split"] == 1, op_types
