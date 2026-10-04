"""Tests for the NeRF-shaped op coverage added to the exporters.

Three additions, all aimed at graphs that torch emits for NeRF / novel-view-
synthesis models, and all three were previously refused by every hand-written
exporter:

* ``onnxsim/einsum_decompose.py`` -- rewrites batched-matrix-multiply
  ``Einsum`` into ``Transpose``/``MatMul`` before lowering, so Core ML, TFLite
  and WebNN all gain einsum support from one implementation. Lives at the ONNX
  level rather than inside each exporter's op table, so the coverage is shared
  and is testable without coremltools/TensorFlow installed.
* ``Sin``/``Cos`` on TFLite -- the sine/cosine positional-encoding head.
* ``GridSample`` on Core ML -- the sampling/warping op.

The einsum decomposition is the bulk of this file and needs nothing but
onnx/onnxruntime. The Core ML GridSample lowering is checked numerically via
MIL's own constant folding (the ``_mil_const_value`` trick already used by
``test_coreml_export.py``), so it needs coremltools but not a Core ML runtime.
"""

import pathlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, numpy_helper, parser

from onnxsim.einsum_decompose import decompose_einsum

F = TensorProto.FLOAT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(body: str, initializer=(), opset: int = 17, ir_version: int = 8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(initializer)
    onnx.checker.check_model(model)
    return model


def _einsum_model(equation: str, shape1, shape2, value1=None, value2=None):
    """A two-operand ``Einsum`` with the output shape derived from the equation.

    Built with ``onnx.helper`` rather than the text format because the text
    format has no spelling for a string attribute such as ``equation``.
    """
    from onnx import helper

    lhs, out = equation.split("->")
    term1, term2 = lhs.split(",")
    dims = {
        **{c: d for c, d in zip(term1, shape1)},
        **{c: d for c, d in zip(term2, shape2)},
    }
    out_shape = [dims[c] for c in out]
    graph = helper.make_graph(
        [helper.make_node("Einsum", ["A", "B"], ["C"], equation=equation)],
        "einsum",
        [
            helper.make_tensor_value_info("A", F, list(shape1)),
            helper.make_tensor_value_info("B", F, list(shape2)),
        ],
        [helper.make_tensor_value_info("C", F, out_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


def _run(model, feed):
    return ort.InferenceSession(model.SerializeToString()).run(None, feed)


def _assert_same_result(original, rewritten, feed):
    """The rewritten graph must be a valid ONNX model producing the same values."""
    onnx.shape_inference.infer_shapes(rewritten, strict_mode=True)
    onnx.checker.check_model(rewritten, full_check=True)
    expected = _run(original, feed)
    actual = _run(rewritten, feed)
    assert len(actual) == len(expected)
    for a, b in zip(expected, actual):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Einsum decomposition
# ---------------------------------------------------------------------------

# Equations of the batched-matmul form, i.e. exactly one axis shared by both
# operands and summed over, one free axis per side. These are what torch emits
# for linear layers, attention projections and ray/sample contractions.
BATCHED_MATMUL_EQUATIONS = [
    # (equation, operand-1 shape, operand-2 shape, expected lowered ops)
    ("ij,jd->id", (5, 7), (7, 3), ["MatMul"]),  # no transpose needed either side
    ("ab,bc->ac", (5, 3), (3, 6), ["MatMul"]),
    ("bj,jk->bk", (4, 6), (6, 2), ["MatMul"]),
    ("ij,jk->ik", (7, 3), (3, 9), ["MatMul"]),
    ("nc,cd->nd", (6, 8), (8, 3), ["MatMul"]),
    # a shared batch axis must lead both operands' matrix axes
    ("bij,bjk->bik", (2, 5, 3), (2, 3, 4), ["MatMul"]),
    # operand 2's free axis precedes the contracted one, so it needs a transpose
    ("mj,nj->mn", (5, 3), (7, 3), ["Transpose", "MatMul"]),
]

# Equations that are NOT a single matrix multiplication. Each must be left
# untouched so the exporter still raises its normal "unsupported op" error
# rather than lowering it to something wrong. Entries are
# ``(equation, operand-1 shape, operand-2 shape)``; ``shape2`` is ``None`` for
# the one-operand forms.
NON_BATCHED_MATMUL_EQUATIONS = [
    ("ii->i", (4, 4), None),  # trace
    ("i,j->ij", (3,), (4,)),  # outer product
    ("ij->", (3, 4), None),  # full reduction
    ("ij,jk->", (3, 4), (4, 5)),  # reduction to a scalar
    ("im,im->im", (2, 3), (2, 3)),  # shared axes that survive
    ("ijk,jl->ikl", (2, 3, 4), (3, 5)),  # two shared axes both kept
    # Per-axis broadcast: the batch axis is on one operand only.
    ("bnc,cd->bnd", (4, 6, 8), (8, 3)),
    ("bj,bk->bk", (4, 6), (4, 7)),
    ("abc,cd->abd", (2, 3, 4), (4, 5)),
    ("nchw,ncd->ndh", (2, 3, 8, 8), (2, 3, 5)),
    # Output order interleaves batch and free axes differently from MatMul's
    # `batch ++ [m, n]`, which would need an output transpose.
    ("bnc,cd->bdn", (4, 6, 8), (8, 3)),
]

_TWO_OPERAND_REFUSED = [e for e in NON_BATCHED_MATMUL_EQUATIONS if e[2] is not None]
_SINGLE_OPERAND_REFUSED = [e for e in NON_BATCHED_MATMUL_EQUATIONS if e[2] is None]


@pytest.mark.parametrize(
    "equation,shape1,shape2,expected_ops",
    BATCHED_MATMUL_EQUATIONS,
    ids=[eq for eq, _, _, _ in BATCHED_MATMUL_EQUATIONS],
)
def test_einsum_batch_matmul_is_lowered(equation, shape1, shape2, expected_ops):
    model = _einsum_model(equation, shape1, shape2)
    rewritten = decompose_einsum(model)
    assert [n.op_type for n in rewritten.graph.node] == expected_ops
    rng = np.random.default_rng(0)
    feed = {
        "A": rng.random(shape1).astype(np.float32),
        "B": rng.random(shape2).astype(np.float32),
    }
    _assert_same_result(model, rewritten, feed)


@pytest.mark.parametrize(
    "equation,shape1,shape2",
    _TWO_OPERAND_REFUSED,
    ids=[eq for eq, _, _ in _TWO_OPERAND_REFUSED],
)
def test_einsum_non_matmul_is_left_alone(equation, shape1, shape2):
    """Refused forms stay ``Einsum``, so the exporter reports them as unsupported."""
    model = _einsum_model(equation, shape1, shape2)
    assert decompose_einsum(model) is model
    assert [n.op_type for n in model.graph.node] == ["Einsum"]


@pytest.mark.parametrize(
    "equation,shape1",
    [(eq, s1) for eq, s1, _ in _SINGLE_OPERAND_REFUSED],
    ids=[eq for eq, _, _ in _SINGLE_OPERAND_REFUSED],
)
def test_einsum_single_operand_is_left_alone(equation, shape1):
    """A one-operand einsum (trace / full reduce) is never a matmul."""
    from onnx import helper

    graph = helper.make_graph(
        [helper.make_node("Einsum", ["A"], ["C"], equation=equation)],
        "einsum",
        [helper.make_tensor_value_info("A", F, list(shape1))],
        [helper.make_tensor_value_info("C", F, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    model.graph.output[0].type.tensor_type.ClearField("shape")
    assert decompose_einsum(model) is model


def test_einsum_model_without_einsum_is_returned_by_identity():
    model = _model(
        """
        relu (float[2,3] x) => (float[2,3] y)
        {
            y = Relu (x)
        }
        """
    )
    assert decompose_einsum(model) is model


def test_einsum_decomposition_does_not_mutate_its_input():
    model = _einsum_model("ij,jd->id", (5, 7), (7, 3))
    before = [n.op_type for n in model.graph.node]
    decompose_einsum(model)
    assert [n.op_type for n in model.graph.node] == before


def test_einsum_whitespace_in_equation_is_tolerated():
    """ONNX permits surrounding whitespace in ``equation``; it must not defeat
    the pattern match."""
    from onnx import helper

    graph = helper.make_graph(
        [helper.make_node("Einsum", ["A", "B"], ["C"], equation=" ij , jd -> id ")],
        "einsum",
        [
            helper.make_tensor_value_info("A", F, [5, 7]),
            helper.make_tensor_value_info("B", F, [7, 3]),
        ],
        [helper.make_tensor_value_info("C", F, [5, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    assert [n.op_type for n in decompose_einsum(model).graph.node] == ["MatMul"]


def test_einsum_transposed_operands_get_fresh_unique_names():
    """Generated Transpose outputs must not collide with existing tensor names."""
    model = _einsum_model("bj,bk->bk", (4, 6), (4, 7))
    model.graph.node[0].name = "C_einsum_b_1"  # collide with our naming scheme
    rewritten = decompose_einsum(model)
    names = {n for n in rewritten.graph.value_info}
    for node in rewritten.graph.node:
        names.update(node.output)
        names.update(t for t in node.input if t)
    outputs = [t for node in rewritten.graph.node for t in node.output]
    assert len(outputs) == len(set(outputs))
    onnx.checker.check_model(rewritten, full_check=True)


def test_einsum_multiple_einsums_in_one_graph_all_lower():
    from onnx import helper

    graph = helper.make_graph(
        [
            helper.make_node("Einsum", ["A", "B"], ["t"], equation="ij,jd->id"),
            helper.make_node("Einsum", ["t", "C"], ["Y"], equation="ij,jk->ik"),
        ],
        "einsum",
        [
            helper.make_tensor_value_info("A", F, [5, 7]),
            helper.make_tensor_value_info("B", F, [7, 3]),
            helper.make_tensor_value_info("C", F, [3, 4]),
        ],
        [helper.make_tensor_value_info("Y", F, [5, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    rewritten = decompose_einsum(model)
    assert "Einsum" not in [n.op_type for n in rewritten.graph.node]
    rng = np.random.default_rng(1)
    feed = {
        "A": rng.random((5, 7)).astype(np.float32),
        "B": rng.random((7, 3)).astype(np.float32),
        "C": rng.random((3, 4)).astype(np.float32),
    }
    _assert_same_result(model, rewritten, feed)


def test_einsum_is_absent_from_every_exporter_op_table():
    """Einsum is decomposed on the way in, so no exporter claims to lower it --
    which is what keeps an un-rewritable equation a hard error rather than a
    silently wrong graph."""
    for module in ("coreml_export", "tflite_export", "rustnn_runtime"):
        imported = pytest.importorskip(
            f"onnxsim.{module}", reason=f"onnxsim.{module} not importable"
        )
        assert "Einsum" not in getattr(imported, "SUPPORTED_ONNX_OPS", ())


def test_tflite_lowering_entry_point_decomposes_einsum(monkeypatch):
    """``_build_concrete_function`` is the one place TFLite lowers a graph, so the
    decomposition happening there covers every caller (CLI, Edge TPU, int8).

    The fake ``_Lowerer`` records the graph it was handed and then stops the
    lowering, so this needs no TensorFlow installed.
    """
    tflite_export = pytest.importorskip(
        "onnxsim.tflite_export", reason="onnxsim.tflite_export not importable"
    )
    model = _einsum_model("ij,jd->id", (5, 7), (7, 3))

    class _Stop(Exception):
        """Stops the lowering right after the graph is prepared."""

    def _fake_init(self, tf, nhwc=False):
        self.graph = tflite_export.decompose_einsum(model).graph
        raise _Stop

    monkeypatch.setattr(tflite_export._Lowerer, "__init__", _fake_init)
    with pytest.raises(_Stop):
        tflite_export._build_concrete_function(model, None)


# ---------------------------------------------------------------------------
# TFLite: Sin / Cos positional-encoding head
# ---------------------------------------------------------------------------


def test_tflite_supports_sin_and_cos():
    tflite_export = pytest.importorskip(
        "onnxsim.tflite_export", reason="onnxsim.tflite_export not importable"
    )
    assert "Sin" in tflite_export.SUPPORTED_ONNX_OPS
    assert "Cos" in tflite_export.SUPPORTED_ONNX_OPS


def test_tflite_sin_cos_handlers_are_elementwise_unary():
    """They must route through the shared ``_simple_unary`` path, i.e. keep the
    channel layout of the tensor they read rather than transposing it."""
    tflite_export = pytest.importorskip(
        "onnxsim.tflite_export", reason="onnxsim.tflite_export not importable"
    )
    for op in ("Sin", "Cos"):
        assert tflite_export._OP_HANDLERS[op].__qualname__.startswith("_simple_unary")


# ---------------------------------------------------------------------------
# Core ML: GridSample
# ---------------------------------------------------------------------------


def test_coreml_supports_gridsample():
    coreml_export = pytest.importorskip(
        "onnxsim.coreml_export", reason="onnxsim.coreml_export not importable"
    )
    assert "GridSample" in coreml_export.SUPPORTED_ONNX_OPS


def _mil_program(model):
    """Build the MIL program for ``model`` (coremltools needed, not macOS)."""
    coreml_export = pytest.importorskip(
        "onnxsim.coreml_export", reason="coremltools is not installed"
    )
    prog, _flexible = coreml_export._build_mil_program(
        model, *coreml_export._import_mil()
    )
    return prog.functions["main"]


def _gridsample_model(
    x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
):
    """A GridSample with both inputs as initializers, so MIL can fold it."""
    from onnx import helper

    out_shape = [int(x.shape[0]), int(x.shape[1])] + [int(d) for d in grid.shape[1:-1]]
    graph = helper.make_graph(
        [
            helper.make_node(
                "GridSample",
                ["x", "grid"],
                ["y"],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=int(align_corners),
            )
        ],
        "gridsample",
        [],
        [helper.make_tensor_value_info("y", F, out_shape)],
        [
            numpy_helper.from_array(x.astype(np.float32), "x"),
            numpy_helper.from_array(grid.astype(np.float32), "grid"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


GRIDSAMPLE_CASES = [
    # (x shape, grid shape, mode, padding_mode, align_corners)
    ((2, 3, 8, 8), (2, 4, 4, 2), "bilinear", "zeros", False),
    ((1, 2, 5, 5), (1, 3, 3, 2), "bilinear", "zeros", True),
    ((1, 2, 5, 5), (1, 3, 3, 2), "bilinear", "border", False),
    ((1, 2, 5, 5), (1, 3, 3, 2), "nearest", "zeros", False),
    ((1, 2, 5, 5), (1, 3, 3, 2), "nearest", "border", True),
    ((2, 4, 6, 6, 6), (2, 2, 2, 2, 3), "bilinear", "zeros", False),  # 3-D trilinear
    ((1, 2, 6, 6, 6), (1, 2, 2, 2, 3), "nearest", "zeros", False),
]


@pytest.mark.parametrize(
    "x_shape,grid_shape,mode,padding_mode,align_corners",
    GRIDSAMPLE_CASES,
    ids=[f"{m}-{p}-align{a}" for _, _, m, p, a in GRIDSAMPLE_CASES],
)
def test_coreml_gridsample_lowers_to_well_formed_mil(
    x_shape, grid_shape, mode, padding_mode, align_corners
):
    """The lowered MIL graph must type-check end to end and land on the ONNX
    output shape.

    Numeric equivalence is covered by
    ``test_coreml_gridsample_index_math_matches_onnx_reference`` below: MIL does
    not constant-fold ``gather_nd``, so the values cannot be read back off the
    program the way the other Core ML tests here do. What this checks is the
    thing a shape bug would break first -- every op's output shape, which is
    fully static here.
    """
    pytest.importorskip("coremltools", reason="coremltools is not installed")
    rng = np.random.default_rng(0)
    x = rng.random(x_shape).astype(np.float32)
    grid = rng.random(grid_shape).astype(np.float32) * 3.0 - 1.5
    model = _gridsample_model(x, grid, mode, padding_mode, align_corners)

    main = _mil_program(model)
    expected_shape = [x_shape[0], x_shape[1]] + list(grid_shape[1:-1])
    assert tuple(main.outputs[0].shape) == tuple(expected_shape)
    ops = [op.op_type for op in main.operations]
    assert "gather_nd" in ops, "sampling must be built from gather_nd corner taps"
    if mode in ("bilinear", "linear"):
        # 2-D bilinear weights 4 corners, 3-D trilinear weights 8.
        assert ops.count("mul") >= 4 if len(x_shape) == 4 else ops.count("mul") >= 8
    else:
        assert "mul" in ops or padding_mode == "border"


def _replay_gridsample_mil(main, inputs):
    """Evaluate the emitted sampling graph in numpy, op by op.

    The handler emits, per corner: ``gather_nd`` through one channel-last view of
    ``x``, then a ``mul`` by the bilinear weight it computed, then (for
    ``padding_mode="zeros"``) a ``mul`` by the per-corner validity mask, then an
    ``add`` into the accumulator -- followed by one transpose and one reshape
    back to the ONNX output order. Every operand is either one of ``inputs`` or a
    constant the handler emitted, so walking the op list in order evaluates
    exactly the graph that was produced.
    """
    values = {}
    for op in main.operations:
        kind = op.op_type

        def get(key):
            var = op.inputs[key]
            return values[var.name] if var.name in values else inputs[var.name]

        out = op.outputs[0].name
        if kind == "const":
            values[out] = np.asarray(op.outputs[0].val)
        elif kind == "transpose":
            values[out] = np.transpose(get("x"), np.asarray(get("perm")).tolist())
        elif kind == "reshape":
            values[out] = get("x").reshape(np.asarray(get("shape")).tolist())
        elif kind == "gather_nd":
            idx, src = get("indices"), get("x")
            flat = idx.shape[1]
            values[out] = np.stack(
                [
                    src[
                        (idx[:, f, 0],)
                        + tuple(idx[:, f, 1 + j] for j in range(idx.shape[2] - 1))
                    ]
                    for f in range(flat)
                ],
                axis=1,
            )
        elif kind == "mul":
            values[out] = get("x") * get("y")
        elif kind == "add":
            values[out] = get("x") + get("y")
        elif kind == "identity":
            values[out] = get("x")
        else:  # pragma: no cover - guards against an unexpected new op
            raise AssertionError(f"unexpected op in GridSample lowering: {kind}")
    return values[main.outputs[0].name]


def test_coreml_gridsample_sampling_matches_onnx_reference():
    """The emitted Core ML sampling graph reproduces ONNX's own GridSample."""
    pytest.importorskip("coremltools", reason="coremltools is not installed")
    from onnx import helper
    from onnx.reference import ReferenceEvaluator

    rng = np.random.default_rng(0)
    for x_shape, grid_shape, mode, padding_mode, align_corners in GRIDSAMPLE_CASES:
        x = rng.random(x_shape).astype(np.float32)
        # Deliberately sample outside [-1, 1] so the padding path is exercised.
        grid = rng.random(grid_shape).astype(np.float32) * 3.0 - 1.5
        model = _gridsample_model(x, grid, mode, padding_mode, align_corners)
        main = _mil_program(model)

        # The graph is all-initializer, so its only non-constant input is `x`
        # itself (the grid is folded into the handler's own constants).
        got = _replay_gridsample_mil(main, {"x": x})

        node = helper.make_node(
            "GridSample",
            ["X", "G"],
            ["Y"],
            mode="linear" if mode == "bilinear" else mode,
            padding_mode=padding_mode,
            align_corners=int(align_corners),
        )
        g = helper.make_graph(
            [node],
            "gs",
            [
                helper.make_tensor_value_info("X", F, list(x_shape)),
                helper.make_tensor_value_info("G", F, list(grid_shape)),
            ],
            [helper.make_tensor_value_info("Y", F, None)],
        )
        ref_model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
        ref_model.ir_version = 9
        onnx_ref = ReferenceEvaluator(ref_model).run(None, {"X": x, "G": grid})[0]

        assert got.shape == tuple(onnx_ref.shape)
        np.testing.assert_allclose(got, onnx_ref, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# The shared GridSample plan
# ---------------------------------------------------------------------------


def test_both_exporters_reduce_gridsample_to_the_same_plan():
    """The Core ML and TFLite lowerings share `grid_sample_plan`, so a future
    change to the sampling arithmetic cannot land in one and miss the other.

    What is asserted: both modules import the shared plan, and the host-side
    reference both defer to is itself plan-driven.
    """
    from onnxsim import coreml_export, tflite_export

    for module in (coreml_export, tflite_export):
        source = pathlib.Path(module.__file__).read_text()
        assert "from .gridsample_plan import grid_sample_plan" in source, (
            f"{module.__name__} does not import the shared GridSample plan"
        )
    # Core ML has no grid-sample kernel, so its handler emits MIL ops straight
    # from the plan and must not recompute coordinates or corner weights.
    coreml_src = pathlib.Path(coreml_export.__file__).read_text()
    handler = coreml_src.split('@_register("GridSample")')[1].split("@_register")[0]
    assert "scoord" not in handler, "the Core ML handler still derives coordinates"
    assert "np.floor" not in handler, "the Core ML handler still derives corners"
    # TFLite emits TensorFlow ops (so it keeps its own coordinate math for the
    # graph), but its numpy reference and const-folding path must agree with the
    # plan rather than duplicate it.
    assert "grid_sample_plan(" in coreml_src
    tflite_src = pathlib.Path(tflite_export.__file__).read_text()
    reference = tflite_src.split("def _numpy_grid_sample(")[1].split("@_register")[0]
    assert "grid_sample_plan(" in reference, (
        "the TFLite numpy reference no longer reduces via the shared plan"
    )


def test_grid_sample_plan_is_referenceable_without_coremltools():
    """The plan is pure numpy, so it is usable (and testable) on its own."""
    from onnxsim.gridsample_plan import grid_sample_plan

    rng = np.random.default_rng(0)
    grid = rng.random((2, 3, 3, 2)).astype(np.float32) * 3.0 - 1.5
    bilinear = grid_sample_plan((2, 3, 8, 8), (2, 3, 3, 2), grid)
    nearest = grid_sample_plan((2, 3, 8, 8), (2, 3, 3, 2), grid, mode="nearest")
    assert len(bilinear) == 4  # 2-D -> 4 corners
    assert len(nearest) == 1
    for tap in bilinear + nearest:
        assert tap.indices.shape == (2, 9, 3)  # (N, flat, 1 + xd)
        assert tap.indices.dtype == np.int32
        assert tap.gather_weight.shape == (2, 9)
        assert tap.valid.dtype == bool
    # Bilinear weights over the 4 corners sum to 1 per sample.
    total = sum(t.gather_weight for t in bilinear)
    np.testing.assert_allclose(total, np.ones((2, 9)), atol=1e-6)


def test_grid_sample_plan_rejects_unsupported_mode_and_padding():
    from onnxsim.gridsample_plan import grid_sample_plan

    grid = np.zeros((1, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="mode"):
        grid_sample_plan((1, 1, 4, 4), (1, 2, 2, 2), grid, mode="bicubic")
    with pytest.raises(ValueError, match="padding_mode"):
        grid_sample_plan((1, 1, 4, 4), (1, 2, 2, 2), grid, padding_mode="reflection")


def test_grid_sample_plan_matches_onnx_reference():
    """Applying the plan's own indices and weights reproduces ONNX's GridSample.

    This is the plan applied independently of ``_numpy_grid_sample`` (which both
    exporters also consume), so a bug in either the plan or one of its two
    consumers shows up here as a mismatch rather than cancelling out.
    """
    from onnx import helper
    from onnx.reference import ReferenceEvaluator

    from onnxsim.gridsample_plan import grid_sample_plan

    rng = np.random.default_rng(0)
    for x_shape, grid_shape, mode, padding_mode, align_corners in GRIDSAMPLE_CASES:
        x = rng.random(x_shape).astype(np.float32)
        grid = rng.random(grid_shape).astype(np.float32) * 3.0 - 1.5
        plan = grid_sample_plan(
            x_shape, grid_shape, grid, mode, padding_mode, align_corners
        )
        n, xd = x_shape[0], len(x_shape) - 2
        out_shape = list(grid_shape[1:-1])
        # (N, *spatial, C); each plan index row is a batch column followed by one
        # index per spatial axis of this channel-last view.
        x_cl = np.moveaxis(x, 1, -1)
        batch = np.broadcast_to(
            np.arange(n).reshape([n] + [1] * len(out_shape)), [n] + out_shape
        )
        acc = 0
        for tap in plan:
            coords = tap.indices[:, :, 1:].reshape([n] + out_shape + [-1])
            vals = x_cl[(batch,) + tuple(coords[..., j] for j in range(xd))]
            vals = np.moveaxis(vals, -1, 1)  # (N, C, *out)
            per_sample = [n, 1] + out_shape
            term = vals * tap.gather_weight.reshape(per_sample)
            if padding_mode == "zeros":
                term = term * tap.valid.reshape(per_sample)
            acc = acc + term

        node = helper.make_node(
            "GridSample",
            ["X", "G"],
            ["Y"],
            mode="linear" if mode == "bilinear" else mode,
            padding_mode=padding_mode,
            align_corners=int(align_corners),
        )
        graph = helper.make_graph(
            [node],
            "gs",
            [
                helper.make_tensor_value_info("X", F, list(x_shape)),
                helper.make_tensor_value_info("G", F, list(grid_shape)),
            ],
            [helper.make_tensor_value_info("Y", F, None)],
        )
        ref_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 20)]
        )
        ref_model.ir_version = 9
        ref = ReferenceEvaluator(ref_model).run(None, {"X": x, "G": grid})[0]
        assert acc.shape == ref.shape
        np.testing.assert_allclose(acc, ref, atol=1e-5, rtol=1e-4)


def test_coreml_gridsample_rejects_unsupported_mode():
    pytest.importorskip("coremltools", reason="coremltools is not installed")
    rng = np.random.default_rng(0)
    x = rng.random((1, 2, 4, 4)).astype(np.float32)
    grid = rng.random((1, 2, 2, 2)).astype(np.float32)
    model = _gridsample_model(x, grid, mode="bicubic")
    with pytest.raises(RuntimeError, match="mode"):
        _mil_program(model)


def test_coreml_gridsample_rejects_unsupported_padding_mode():
    pytest.importorskip("coremltools", reason="coremltools is not installed")
    rng = np.random.default_rng(0)
    x = rng.random((1, 2, 4, 4)).astype(np.float32)
    grid = rng.random((1, 2, 2, 2)).astype(np.float32)
    model = _gridsample_model(x, grid, padding_mode="reflection")
    with pytest.raises(RuntimeError, match="padding_mode"):
        _mil_program(model)
