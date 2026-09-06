"""Tests for the opt-in ``split_large_gather`` pass.

The mirror image of ``fuse_split_gather_concat``: it rewrites a ``Gather``
whose indices tensor is "too large" (more than
``SplitLargeGather::kMaxIndicesPerGather`` == 65536 elements, see that C++
class) into ``Concat(Gather(x, Split(idx, axis=0)_0, axis=ga), ...,
axis=ga)`` -- useful for size-limited hardware backends that cannot run one
oversized ``Gather``. Since 65536 elements is too large to spell out as a
literal in an ONNX text-format model, every model here uses a dynamic
``idx`` graph input (any shape/dtype is fine, since this pass never inspects
index values, only ``idx``'s static shape) sized just over or under that
threshold, and equivalence-checks against concrete ``input_data`` the same
way as the other migrated test files.
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
        extra_optimizers=["split_large_gather"],
    )
    assert check_ok, "split graph failed onnxsim's equivalence check"
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    return sim_model, op_types


def _simplify_and_assert_declined(model, input_data, check_n=1):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
        extra_optimizers=["split_large_gather"],
    )
    assert check_ok
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert op_types["Split"] == 0, op_types
    assert op_types["Concat"] == 0, op_types
    assert op_types["Gather"] == 1, op_types
    return sim_model, op_types


# --------------------------------------------------------------------------- #
# indices' total element count (100_000) exceeds the 65536 limit: splits into
# Concat(Gather(x, Split(idx, axis=0)), ...).
# --------------------------------------------------------------------------- #


def test_large_1d_indices_get_split():
    n = 100_000
    body = f"""
    agraph (float[10,4] x, int64[{n}] idx) => (float[{n},4] Y)
    {{
      Y = Gather <axis=0> (x, idx)
    }}
    """
    model = _model(body)
    rng = np.random.RandomState(0)
    input_data = {
        "x": rng.randn(10, 4).astype(np.float32),
        "idx": rng.randint(0, 10, size=(n,)).astype(np.int64),
    }
    sim_model, op_types = _simplify_and_check(model, input_data)
    assert op_types["Gather"] > 1, op_types
    assert op_types["Split"] == 1, op_types
    assert op_types["Concat"] == 1, op_types

    # Every chunk Split produces must itself be at or under the limit --
    # the whole point of the rewrite. At opset 17, split sizes are the
    # Split node's second input, resolved as an initializer.
    split_node = next(n for n in sim_model.graph.node if n.op_type == "Split")
    initializers = {init.name: init for init in sim_model.graph.initializer}
    sizes_init = initializers[split_node.input[1]]
    chunk_sizes = list(sizes_init.int64_data) or list(
        np.frombuffer(sizes_init.raw_data, dtype=np.int64)
    )
    assert sum(chunk_sizes) == n, chunk_sizes
    assert all(s <= 65536 for s in chunk_sizes), chunk_sizes


def test_non_leading_gather_axis_still_uses_same_axis_for_concat():
    n = 100_000
    body = f"""
    agraph (float[4,10] x, int64[{n}] idx) => (float[4,{n}] Y)
    {{
      Y = Gather <axis=1> (x, idx)
    }}
    """
    model = _model(body)
    rng = np.random.RandomState(1)
    input_data = {
        "x": rng.randn(4, 10).astype(np.float32),
        "idx": rng.randint(0, 10, size=(n,)).astype(np.int64),
    }
    sim_model, op_types = _simplify_and_check(model, input_data)
    assert op_types["Split"] == 1, op_types
    assert op_types["Concat"] == 1, op_types


def test_2d_indices_split_along_axis0():
    # dim0=200, per-row=1000 -> per_row <= 65536, splits along axis 0.
    rows, cols = 200, 1000
    body = f"""
    agraph (float[20,4] x, int64[{rows},{cols}] idx) => (float[{rows},{cols},4] Y)
    {{
      Y = Gather <axis=0> (x, idx)
    }}
    """
    model = _model(body)
    rng = np.random.RandomState(2)
    input_data = {
        "x": rng.randn(20, 4).astype(np.float32),
        "idx": rng.randint(0, 20, size=(rows, cols)).astype(np.int64),
    }
    sim_model, op_types = _simplify_and_check(model, input_data)
    assert op_types["Split"] == 1, op_types
    assert op_types["Concat"] == 1, op_types


# --------------------------------------------------------------------------- #
# Declined: indices' total element count is at or under the limit.
# --------------------------------------------------------------------------- #


def test_declined_small_indices():
    body = """
    agraph (float[10,4] x, int64[6] idx) => (float[6,4] Y)
    {
      Y = Gather <axis=0> (x, idx)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(3)
    input_data = {
        "x": rng.randn(10, 4).astype(np.float32),
        "idx": rng.randint(0, 10, size=(6,)).astype(np.int64),
    }
    _simplify_and_assert_declined(model, input_data)


# --------------------------------------------------------------------------- #
# Declined: indices' shape isn't fully statically known.
# --------------------------------------------------------------------------- #


def test_declined_dynamic_indices_shape():
    body = """
    agraph (float[10,4] x, int64[N] idx) => (float[N,4] Y)
    {
      Y = Gather <axis=0> (x, idx)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(4)
    n = 100_000
    input_data = {
        "x": rng.randn(10, 4).astype(np.float32),
        "idx": rng.randint(0, 10, size=(n,)).astype(np.int64),
    }
    _simplify_and_assert_declined(model, input_data)


# --------------------------------------------------------------------------- #
# Declined: per-row element count (the product of every axis but 0) alone
# already exceeds the limit -- no split along axis 0 can help.
# --------------------------------------------------------------------------- #


def test_declined_large_per_row():
    body = """
    agraph (float[3,4] x, int64[2,100000] idx) => (float[2,100000,4] Y)
    {
      Y = Gather <axis=0> (x, idx)
    }
    """
    model = _model(body)
    rng = np.random.RandomState(5)
    input_data = {
        "x": rng.randn(3, 4).astype(np.float32),
        "idx": rng.randint(0, 3, size=(2, 100_000)).astype(np.int64),
    }
    _simplify_and_assert_declined(model, input_data)


# --------------------------------------------------------------------------- #
# Not opted in: the pass never runs unless explicitly requested.
# --------------------------------------------------------------------------- #


def test_not_run_by_default():
    n = 100_000
    body = f"""
    agraph (float[10,4] x, int64[{n}] idx) => (float[{n},4] Y)
    {{
      Y = Gather <axis=0> (x, idx)
    }}
    """
    model = _model(body)
    rng = np.random.RandomState(6)
    input_data = {
        "x": rng.randn(10, 4).astype(np.float32),
        "idx": rng.randint(0, 10, size=(n,)).astype(np.int64),
    }
    sim_model, check_ok = onnxsim.simplify(model, check_n=1, input_data=input_data)
    assert check_ok
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert op_types["Gather"] == 1, op_types
    assert op_types["Split"] == 0, op_types
