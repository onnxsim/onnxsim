"""RK3588 real-NPU compatibility test.

The sibling ``tests/test_rknn_compat.py`` only exercises ``rknn-toolkit2``'s
**PC simulator**. This file covers the higher fidelity tier added for the
NanoPC-T6 (RK3588): the original and onnxsim-simplified graphs are compiled with
``target_platform="rk3588"``, uploaded over SSH, and run on the board's actual
NPU through ``librknnrt.so``, with the results compared against each other.

Because it needs both ``rknn-toolkit2`` (an x86-64 Linux wheel) **and** a
reachable RK3588 board with ``librknnrt.so``, it is skipped unless
``RKNN_3588_HOST`` is set::

    RKNN_3588_HOST=nanopc-t6.tailf0b7b1.ts.net \\
    RKNN_SSH_PASSWORD=... pytest tests/test_rknn3588_compat.py

Set ``RKNN_3588_REQUIRE=1`` to turn "no board configured" into a failure, which
is what a dedicated hardware CI job wants.
"""

import os
import sys

import pytest

_RKNN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "rknn"
)

HOST = os.environ.get("RKNN_3588_HOST", "")
USER = os.environ.get("RKNN_3588_USER", "pi")
PASSWORD = os.environ.get("RKNN_SSH_PASSWORD")
REQUIRED = os.environ.get("RKNN_3588_REQUIRE", "") in {"1", "true", "yes"}

pytestmark = pytest.mark.skipif(
    not HOST,
    reason="set RKNN_3588_HOST to a reachable RK3588 board to run the real-NPU check",
)


@pytest.fixture(scope="module")
def harness():
    if _RKNN_DIR not in sys.path:
        sys.path.insert(0, _RKNN_DIR)
    try:
        import rknn.api  # noqa: F401
    except Exception:
        if REQUIRED:
            pytest.fail("RKNN_3588_REQUIRE is set but rknn-toolkit2 is unavailable")
        pytest.skip("rknn-toolkit2 is unavailable on this host")
    import run_rknn3588_compat as module

    return module


def test_board_is_an_rk3588(harness, tmp_path):
    """A misconfigured host must not silently report another SoC's results.

    Compiles the smallest suite model and runs it once, asserting the runtime
    itself identifies the device as an RK3588 with all 3 NPU cores.
    """
    import numpy as np
    import onnx as onnx_mod
    from common.ep_numerics import random_feeds  # noqa: E402
    from common.synthetic_models import build  # noqa: E402

    model = build("conv_bn_relu")
    onnx_path = str(tmp_path / "conv_bn_relu.onnx")
    onnx_mod.save(model, onnx_path)
    feeds = random_feeds(model, seed=0)
    np.savez(str(tmp_path / "conv_bn_relu.feeds.npz"), **feeds)
    harness.compile_rknn(
        onnx_path, str(tmp_path / "conv_bn_relu.rknn"), [feeds["x"]], "x"
    )

    board = harness.run_on_board(
        HOST,
        USER,
        PASSWORD,
        str(tmp_path / "conv_bn_relu.rknn"),
        str(tmp_path / "conv_bn_relu.feeds.npz"),
        "/tmp/onnxsim-rknn-test",
        2,
        5,
    )
    assert board["device"]["soc"] == "RKNN_SOC_RK3588", board["device"]
    assert board["device"]["cores"] == 3, board["device"]


@pytest.mark.parametrize(
    "name",
    [
        "conv_bn_relu",
        "foldable_shape_reshape",
        "redundant_transpose",
        "sigmoid_mul_swish",
    ],
)
def test_simplify_does_not_change_rk3588_output(harness, tmp_path, name):
    """The real regression signal: simplification must not change what the
    RK3588 NPU computes, with both graphs built from identical calibration."""
    from common.synthetic_models import build

    row = harness._check_model(
        name,
        build(name),
        str(tmp_path),
        HOST,
        USER,
        PASSWORD,
        "/tmp/onnxsim-rknn-test",
        rtol=2e-2,
        atol=2e-2,
        warmup=5,
        iterations=20,
    )
    assert row["status"] == "ok", row
    # Measured exactly 0.0 on the connected NanoPC-T6 for every model in the
    # suite. A small nonzero value would still be acceptable (INT8 requantization
    # of a restructured graph can reorder a rounding step); a large one is not.
    assert row["diff_sim_vs_simp"] < 2e-2, row


def test_float_buffer_size_accounts_for_stride_padding():
    """Regression guard for a verified NanoPC-T6 failure.

    ``rknn_build()`` rewrites default I/O dtype to int8, and a ``want_float``
    buffer must cover the tensor's *stride-padded* extent at 4 bytes/element.
    Sizing from ``n_elems`` alone under-allocates for padded tensors and the
    runtime rejects it with
    ``rknn_set_io_mem, input memory size(1728) < model input size(2304)``.
    """
    if _RKNN_DIR not in sys.path:
        sys.path.insert(0, _RKNN_DIR)
    import nanopc_rknn_runner as runner

    class _Attr:
        # sigmoid_mul_swish's input, as rknn_query actually reported it on the
        # board: [1,12,12,3], int8, size=432 but size_with_stride=576.
        n_elems = 432
        size = 432
        size_with_stride = 576
        type = runner.RKNN_TENSOR_INT8

    assert runner._float_buffer_size(_Attr()) == 576 * 4


def test_input_channels_only_for_rank4(tmp_path):
    """``mean_values``/``std_values`` must be omitted for non-rank-4 inputs.

    RKNN rejects a mismatched length outright: a 1-element vector for a
    ``[1,32,16,16]`` input fails with ``The len of mean_values ([0.0]) for
    input 0 is wrong, expect 32!``.
    """
    if _RKNN_DIR not in sys.path:
        sys.path.insert(0, _RKNN_DIR)
    import onnx  # noqa: E402
    import run_rknn3588_compat as harness  # noqa: E402

    def _save(shape, path):
        from onnx import TensorProto, helper

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [shape[-1]])
        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph([node], "g", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, path)

    rank4 = str(tmp_path / "rank4.onnx")
    rank2 = str(tmp_path / "rank2.onnx")
    _save([1, 32, 16, 16], rank4)
    _save([8, 16], rank2)
    assert harness._input_channels(rank4) == 32
    assert harness._input_channels(rank2) is None
