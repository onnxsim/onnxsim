"""Tests for the opt-in ``rewrite_gridsample_to_gather`` pass.

Every model is a single 2-D ``GridSample`` node, built with the ONNX text
format parser (``onnx.parser``, no torch dependency) per CLAUDE.md's
convention for this repo's tests. Each is run through
``onnxsim.simplify(..., extra_optimizers=["rewrite_gridsample_to_gather"])``,
which numerically equivalence-checks the rewritten graph against the original
``GridSample`` node (via onnxruntime, or the onnx reference evaluator when
onnxruntime is not installed) on an actual random ``grid`` *input* -- not a
folded constant -- generated with a wide enough range to include values
outside ``[-1, 1]`` on every test, so padding-mode handling (not just
in-bounds sampling) is always exercised. ``input_data`` pins that grid (and
``X``) so the same values are used for both the original and rewritten graphs
-- see ``tests/test_pocket_tts.py``'s ``check_n=1`` / fixed-``input_data``
idiom, which this mirrors.
"""

import collections
import importlib.util

import numpy as np
import pytest
from onnx import parser

import onnxsim

# The onnx reference evaluator implements only GridSample-20's spelling of the
# interpolation modes ("linear"/"nearest"/"cubic") and raises outright on the
# opset-16..19 "bilinear"/"bicubic" spelling of the same modes. onnxsim's
# equivalence check has to run the *original* (unrewritten) GridSample node, so
# the pre-opset-20 tests below need onnxruntime, which implements both
# spellings -- each one only on the opset range that uses it.
requires_ort = pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="running a pre-opset-20 GridSample node needs onnxruntime",
)


def _model(
    x_shape,
    grid_shape,
    out_shape,
    mode,
    padding_mode,
    align_corners,
    opset=20,
    ir_version=10,
):
    # ``mode=None`` omits the attribute entirely, exercising the schema
    # default ("bilinear" before opset 20, "linear" from 20 on -- the same
    # mode either way).
    mode_attr = "" if mode is None else f'mode="{mode}", '
    body = f"""
    <
      ir_version: {ir_version},
      opset_import: ["": {opset}]
    >
    agraph (float{x_shape} X, float{grid_shape} grid) => (float{out_shape} Y)
    {{
      Y = GridSample <{mode_attr}padding_mode="{padding_mode}", align_corners={align_corners}> (X, grid)
    }}
    """
    return parser.parse_model(body)


def _rand_grid(rng, shape, lo=-1.6, hi=1.6):
    # A wide enough range that, with overwhelming probability, some values
    # fall outside [-1, 1] -- so zeros/border/reflection padding-mode
    # handling is actually exercised, not just in-bounds bilinear/nearest
    # sampling.
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


def _simplify_and_check(model, input_data, check_n=1):
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        input_data=input_data,
        extra_optimizers=["rewrite_gridsample_to_gather"],
    )
    assert check_ok, "rewritten graph failed onnxsim's equivalence check"
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert "GridSample" not in op_types, op_types
    assert "GatherND" in op_types, op_types
    return sim_model, op_types


def _simplify_expect_declined(model):
    """Simplifies with the pass opted in and asserts it left the node alone.

    No equivalence check (``check_n=0``): nothing about the graph changed, and
    the modes this is used for ("cubic"/"bicubic") are exactly the ones the
    pass declines to decompose.
    """
    sim_model, _ = onnxsim.simplify(
        model,
        extra_optimizers=["rewrite_gridsample_to_gather"],
    )
    op_types = collections.Counter(n.op_type for n in sim_model.graph.node)
    assert op_types == collections.Counter({"GridSample": 1}), op_types
    return sim_model


# --------------------------------------------------------------------------- #
# bilinear ("linear") x {align_corners} x {padding_mode}
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("align_corners", [0, 1])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_linear_all_padding_modes(padding_mode, align_corners):
    rng = np.random.RandomState(0)
    X = rng.randn(2, 3, 5, 7).astype(np.float32)
    grid = _rand_grid(rng, (2, 4, 6, 2))

    model = _model(
        "[2,3,5,7]",
        "[2,4,6,2]",
        "[2,3,4,6]",
        mode="linear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


# --------------------------------------------------------------------------- #
# nearest x {padding_mode}, one align_corners setting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_nearest_all_padding_modes(padding_mode):
    rng = np.random.RandomState(1)
    X = rng.randn(2, 3, 5, 7).astype(np.float32)
    grid = _rand_grid(rng, (2, 4, 6, 2))

    model = _model(
        "[2,3,5,7]",
        "[2,4,6,2]",
        "[2,3,4,6]",
        mode="nearest",
        padding_mode=padding_mode,
        align_corners=0,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


# --------------------------------------------------------------------------- #
# Grid values outside [-1, 1] -- explicit, dedicated coverage of padding-mode
# handling beyond what the wide random range above already exercises
# incidentally: a grid built entirely from out-of-range values (some barely
# so, some far enough to need more than one reflection fold).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_out_of_range_grid_values(padding_mode):
    X = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    # A mix of just-out-of-range and far-out-of-range (multi-reflection)
    # coordinates on both axes, broadcast across the batch/output grid.
    base = np.array(
        [
            [-1.2, 1.3],
            [2.5, -2.7],
            [5.9, -6.1],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )
    grid = np.broadcast_to(base, (2, 4, 4, 2)).copy()

    model = _model(
        "[2,3,4,4]",
        "[2,4,4,2]",
        "[2,3,4,4]",
        mode="linear",
        padding_mode=padding_mode,
        align_corners=0,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


# --------------------------------------------------------------------------- #
# Dynamic (symbolic) H/W -- the pass must derive H/W from Shape(X) at
# runtime, never assume a static shape.
# --------------------------------------------------------------------------- #


def test_dynamic_input_shape():
    rng = np.random.RandomState(2)
    X = rng.randn(2, 3, 9, 11).astype(np.float32)
    grid = _rand_grid(rng, (2, 4, 6, 2))

    model = _model(
        "[N,3,H,W]",
        "[N,4,6,2]",
        "[N,3,4,6]",
        mode="linear",
        padding_mode="zeros",
        align_corners=0,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


def test_dynamic_input_shape_reflection_nearest():
    # Also cover the reflection/nearest combination under a dynamic shape --
    # H/W feed directly into the per-axis reflect bounds and the rounding
    # path, both of which must stay purely runtime-derived.
    rng = np.random.RandomState(3)
    X = rng.randn(1, 2, 6, 8).astype(np.float32)
    grid = _rand_grid(rng, (1, 3, 5, 2))

    model = _model(
        "[N,2,H,W]",
        "[N,3,5,2]",
        "[N,2,3,5]",
        mode="nearest",
        padding_mode="reflection",
        align_corners=1,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


# --------------------------------------------------------------------------- #
# Pre-opset-20 mode spelling. GridSample-20 renamed two of the three
# interpolation modes -- "bilinear" -> "linear" and "bicubic" -> "cubic" (the
# schema default with them) -- without changing what they compute, so an
# opset-16..19 model spells the very same mode differently. The pass must
# recognize both spellings; a node it would decompose at opset 20 must not
# survive untouched merely for having been exported at opset 16.
# --------------------------------------------------------------------------- #


@requires_ort
@pytest.mark.parametrize("opset", [16, 19])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_bilinear_pre_opset_20(opset, padding_mode):
    rng = np.random.RandomState(4)
    X = rng.randn(2, 3, 5, 7).astype(np.float32)
    grid = _rand_grid(rng, (2, 4, 6, 2))

    model = _model(
        "[2,3,5,7]",
        "[2,4,6,2]",
        "[2,3,4,6]",
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=0,
        opset=opset,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


@requires_ort
@pytest.mark.parametrize("opset", [16, 19])
def test_nearest_pre_opset_20(opset):
    # "nearest" is spelled the same in both opset ranges -- covered here to
    # pin that the added spelling handling did not disturb it.
    rng = np.random.RandomState(5)
    X = rng.randn(1, 2, 6, 8).astype(np.float32)
    grid = _rand_grid(rng, (1, 3, 5, 2))

    model = _model(
        "[1,2,6,8]",
        "[1,3,5,2]",
        "[1,2,3,5]",
        mode="nearest",
        padding_mode="reflection",
        align_corners=1,
        opset=opset,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


@pytest.mark.parametrize("opset", [16, 20])
def test_default_mode(opset):
    # No `mode` attribute at all: the schema default is "bilinear" before
    # opset 20 and "linear" from 20 on -- the same mode under both names, so
    # the pass must decompose it either way.
    rng = np.random.RandomState(6)
    X = rng.randn(2, 3, 5, 7).astype(np.float32)
    grid = _rand_grid(rng, (2, 4, 6, 2))

    model = _model(
        "[2,3,5,7]",
        "[2,4,6,2]",
        "[2,3,4,6]",
        mode=None,
        padding_mode="zeros",
        align_corners=0,
        opset=opset,
    )
    _simplify_and_check(model, {"X": X, "grid": grid})


@pytest.mark.parametrize(
    "opset,mode", [(16, "bicubic"), (20, "cubic")], ids=["bicubic-16", "cubic-20"]
)
def test_declines_cubic(opset, mode):
    # The cubic mode is out of scope for this rewrite under either of its
    # names -- recognizing the pre-opset-20 spelling must not turn "bicubic"
    # into something the pass thinks it can decompose.
    model = _model(
        "[2,3,5,7]",
        "[2,4,6,2]",
        "[2,3,4,6]",
        mode=mode,
        padding_mode="zeros",
        align_corners=0,
        opset=opset,
    )
    _simplify_expect_declined(model)
