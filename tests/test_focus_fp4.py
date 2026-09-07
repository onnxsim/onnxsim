"""Tests for ``onnxsim.quantize_weight_only_mxfp4_focus`` (FOCUS's CRS /
DGS decoupled quantization scale, see ``onnxsim/focus_fp4.py``) -- a
data-free per-block search that changes only which E2M1 *codes* a block
gets, leaving the stored power-of-two E8M0 dequantization scale, and every
other byte of the emitted graph, exactly as
``onnxsim.quantize_weight_only_mxfp4`` produces it.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim.focus_fp4 import FOCUS_DGS_SUB_BLOCK_SIZE, _coefficient_grid
from onnxsim.mx_quantization import MXFP4_CODEBOOK

ort = pytest.importorskip("onnxruntime")

CODEBOOK = np.asarray(MXFP4_CODEBOOK, dtype=np.float64)


def _f32(array, name):
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _model(body, initializer=(), opset=13, ir_version=8):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _matmul_model(K=64, N=16, weight=None, seed=0, scale=0.5):
    if weight is None:
        rng = np.random.default_rng(seed)
        weight = rng.standard_normal((K, N)).astype(np.float32) * scale
    return _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
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


def _initializer(model, name):
    return next(t for t in model.graph.initializer if t.name == name)


def _dequantize(model, w_name="W", block_size=32):
    """Decodes the emitted MXFP4 initializers back to float, exactly as
    ``tests/test_mx_quantization.py``'s own hand decoder does -- the FOCUS
    variant emits the identically-named initializers, which is itself part
    of what is under test here.
    """
    codes = onnx.numpy_helper.to_array(_initializer(model, f"{w_name}_mxfp4_q")).astype(
        np.int64
    )
    scale = onnx.numpy_helper.to_array(
        _initializer(model, f"{w_name}_mxfp4_scale")
    ).astype(np.float64)
    return CODEBOOK[codes] * np.repeat(scale, block_size, axis=0)


def _block_objective(weight, dequantized, block_size=32, lam=1.0):
    """The module's own per-block objective ``sum(D^2) + lambda*(sum D)^2``,
    summed over blocks -- ``lambda=0`` recovers plain (unnormalized)
    element-wise squared error.
    """
    delta = dequantized - weight.astype(np.float64)
    k, n = delta.shape
    blocked = delta.reshape(k // block_size, block_size, n)
    square = float(np.sum(blocked * blocked))
    return square + lam * float(np.sum(np.sum(blocked, axis=1) ** 2))


def _mse(weight, dequantized):
    delta = dequantized - weight.astype(np.float64)
    return float(np.mean(delta * delta))


# --------------------------------------------------------------------------
# The central claim: the deployed format is unchanged, only the codes differ.
# --------------------------------------------------------------------------


def test_focus_emits_the_identical_format_only_codes_differ():
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((128, 32)).astype(np.float32) * 0.5
    model = _matmul_model(weight=weight)
    plain = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)
    focus = onnxsim.quantize_weight_only_mxfp4_focus(model, block_size=32)
    onnx.checker.check_model(focus)

    # Same initializers, in the same order, with the same shapes and dtypes.
    assert [t.name for t in plain.graph.initializer] == [
        t.name for t in focus.graph.initializer
    ]
    for a, b in zip(plain.graph.initializer, focus.graph.initializer):
        assert (a.name, list(a.dims), a.data_type) == (
            b.name,
            list(b.dims),
            b.data_type,
        )

    # Same nodes, in the same order, with the same inputs/outputs.
    assert [(n.op_type, list(n.input), list(n.output)) for n in plain.graph.node] == [
        (n.op_type, list(n.input), list(n.output)) for n in focus.graph.node
    ]

    # Every initializer is bit-identical except the code bytes themselves --
    # including the power-of-two dequantization scale, which is the whole
    # point of CRS/DGS: it stays exactly hardware-legal and unchanged.
    differing = {
        a.name
        for a, b in zip(plain.graph.initializer, focus.graph.initializer)
        if a.SerializeToString() != b.SerializeToString()
    }
    assert differing == {"W_mxfp4_q"}

    # ... and swapping the codes back makes the two models byte-identical,
    # so nothing downstream can distinguish the formats at all.
    patched = onnx.ModelProto()
    patched.CopyFrom(focus)
    _initializer(patched, "W_mxfp4_q").CopyFrom(_initializer(plain, "W_mxfp4_q"))
    assert patched.SerializeToString() == plain.SerializeToString()


def test_focus_scale_is_still_a_power_of_two():
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((64, 16)).astype(np.float32) * 3.7
    model = _matmul_model(weight=weight)
    focus = onnxsim.quantize_weight_only_mxfp4_focus(model, block_size=32)

    scale = (
        onnx.numpy_helper.to_array(_initializer(focus, "W_mxfp4_scale"))
        .astype(np.float64)
        .ravel()
    )
    log2_scale = np.log2(scale)
    assert np.all(np.abs(log2_scale - np.round(log2_scale)) < 1e-9)


def test_focus_codes_stay_in_range():
    rng = np.random.default_rng(6)
    weight = rng.standard_normal((64, 8)).astype(np.float32) * 3
    model = _matmul_model(weight=weight)
    focus = onnxsim.quantize_weight_only_mxfp4_focus(model, block_size=32)
    codes = onnx.numpy_helper.to_array(_initializer(focus, "W_mxfp4_q"))
    assert codes.dtype == np.uint8
    assert np.all(codes >= 0) and np.all(codes <= 15)


def test_focus_uses_standard_ops_only_and_runs():
    model = _matmul_model(K=64, N=16, seed=3)
    focus = onnxsim.quantize_weight_only_mxfp4_focus(model, block_size=32)
    onnx.checker.check_model(focus)
    op_types = {n.op_type for n in focus.graph.node}
    assert op_types <= {"MatMul", "Cast", "Gather", "Reshape", "Mul"}
    assert all(n.domain in ("", "ai.onnx") for n in focus.graph.node)

    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(focus, {"X": x})
    assert np.all(np.isfinite(q_y))
    assert _rel_l2(float_y, q_y) < 0.3


def test_focus_gemm_transb():
    rng = np.random.default_rng(5)
    K, N = 128, 12
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[batch,{K}] X) => (float[batch,{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W)
        }}
        """,
        initializer=[_f32(weight, "W")],
    )
    plain = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)
    focus = onnxsim.quantize_weight_only_mxfp4_focus(model, block_size=32)
    onnx.checker.check_model(focus)
    # The transposed path must preserve the format identity too.
    assert (
        _initializer(plain, "W_mxfp4_scale").SerializeToString()
        == _initializer(focus, "W_mxfp4_scale").SerializeToString()
    )

    x = rng.standard_normal((4, K)).astype(np.float32)
    (float_y,) = _run(model, {"X": x})
    (q_y,) = _run(focus, {"X": x})
    assert _rel_l2(float_y, q_y) < 0.3


# --------------------------------------------------------------------------
# Strict generalization: the degenerate settings reproduce plain MXFP4.
# --------------------------------------------------------------------------


def test_focus_with_unit_coefficient_reproduces_plain_mxfp4_exactly():
    rng = np.random.default_rng(11)
    weight = rng.standard_normal((128, 16)).astype(np.float32) * 0.5
    model = _matmul_model(weight=weight)
    plain = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)
    focus = onnxsim.quantize_weight_only_mxfp4_focus(
        model, block_size=32, coefficients=(1.0,)
    )
    assert focus.SerializeToString() == plain.SerializeToString()


def test_focus_with_zero_aggregate_weight_reproduces_plain_mxfp4_exactly():
    """With ``aggregate_error_weight=0`` the objective is plain element-wise
    MSE, for which per-element nearest rounding at the *fixed* dequantization
    scale is already optimal -- so the full 17-candidate CRS grid plus DGS
    refinement provably cannot improve on plain MXFP4, and must return its
    codes untouched. This is the module's own honesty note, asserted.
    """
    rng = np.random.default_rng(12)
    weight = rng.standard_normal((256, 16)).astype(np.float32) * 0.5
    model = _matmul_model(weight=weight)
    plain = onnxsim.quantize_weight_only_mxfp4(model, block_size=32)
    for sub_block_size in (None, FOCUS_DGS_SUB_BLOCK_SIZE):
        focus = onnxsim.quantize_weight_only_mxfp4_focus(
            model,
            block_size=32,
            sub_block_size=sub_block_size,
            aggregate_error_weight=0.0,
        )
        assert focus.SerializeToString() == plain.SerializeToString()
        # ... which is exactly the statement "element-wise reconstruction MSE
        # is <= plain MXFP4's": at this setting it is equal, and it provably
        # cannot be lower.
        assert _mse(weight, _dequantize(focus)) == _mse(weight, _dequantize(plain))


def test_coefficient_grid_always_contains_exactly_one_and_starts_there():
    grid = _coefficient_grid(17, 1.25)
    assert grid[0] == 1.0  # ties in the search keep plain MXFP4's own codes
    assert len(grid) == 17
    assert np.isclose(grid.min(), 1 / 1.25) and np.isclose(grid.max(), 1.25)
    assert np.count_nonzero(grid == 1.0) == 1
    # Degenerate configurations collapse to the plain-MXFP4 choice.
    assert list(_coefficient_grid(1, 1.25)) == [1.0]
    assert list(_coefficient_grid(17, 1.0)) == [1.0]


# --------------------------------------------------------------------------
# Accuracy. Aggregated over several seeds with a generous measured margin --
# per-seed inequalities on this kind of comparison have bitten this repo
# before (see tests/test_gptaq.py's history).
# --------------------------------------------------------------------------

_ACCURACY_SEEDS = range(8)


def _accuracy_case(seed, K=128, N=32):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _matmul_model(weight=weight)
    plain = _dequantize(onnxsim.quantize_weight_only_mxfp4(model, block_size=32))
    crs = _dequantize(
        onnxsim.quantize_weight_only_mxfp4_focus(
            model, block_size=32, sub_block_size=None
        )
    )
    dgs = _dequantize(
        onnxsim.quantize_weight_only_mxfp4_focus(
            model, block_size=32, sub_block_size=FOCUS_DGS_SUB_BLOCK_SIZE
        )
    )
    return weight, plain, crs, dgs


def test_focus_never_increases_the_block_objective_it_optimizes():
    """Monotone by construction: ``c_i = 1`` is always a candidate, and DGS
    refinement starts from CRS's own solution accepting only strict
    improvements. Checked per seed, since it is a guarantee, not a trend.
    """
    for seed in _ACCURACY_SEEDS:
        weight, plain, crs, dgs = _accuracy_case(seed)
        plain_objective = _block_objective(weight, plain)
        crs_objective = _block_objective(weight, crs)
        dgs_objective = _block_objective(weight, dgs)
        assert crs_objective <= plain_objective
        assert dgs_objective <= crs_objective  # DGS is a refinement of CRS


def test_focus_improves_the_block_objective_by_a_wide_margin():
    plain_total = crs_total = dgs_total = 0.0
    for seed in _ACCURACY_SEEDS:
        weight, plain, crs, dgs = _accuracy_case(seed)
        plain_total += _block_objective(weight, plain)
        crs_total += _block_objective(weight, crs)
        dgs_total += _block_objective(weight, dgs)
    # Measured across these seeds: CRS ~39% better, DGS ~43% better. 15% is a
    # deliberately loose floor on a comfortably reproducible gap.
    assert crs_total <= 0.85 * plain_total
    assert dgs_total <= 0.85 * plain_total
    assert dgs_total <= crs_total


def test_focus_reduces_layer_output_error_under_correlated_activations():
    """The weight where "the coupling actually costs something": a layer whose
    inputs are mean-shifted (as post-GELU/post-ReLU activations are), so the
    block's *aggregate* rounding error -- which plain nearest rounding does
    nothing about -- dominates the output error.

    Measured with real matmuls, not the module's own surrogate objective.
    """
    plain_total = crs_total = dgs_total = 0.0
    iid_plain_total = iid_crs_total = 0.0
    for seed in _ACCURACY_SEEDS:
        weight, plain, crs, dgs = _accuracy_case(seed, K=256, N=32)
        rng = np.random.default_rng(1000 + seed)
        w64 = weight.astype(np.float64)
        # E[x_i x_j] = sigma^2 * delta_ij + mu^2, i.e. the module's own
        # I + lambda*11^T input model with lambda = mu^2 / sigma^2 = 1.
        x = 1.0 + rng.standard_normal((2048, weight.shape[0]))
        reference = x @ w64
        plain_total += float(np.mean((x @ plain - reference) ** 2))
        crs_total += float(np.mean((x @ crs - reference) ** 2))
        dgs_total += float(np.mean((x @ dgs - reference) ** 2))

        x_iid = rng.standard_normal((2048, weight.shape[0]))
        reference_iid = x_iid @ w64
        iid_plain_total += float(np.mean((x_iid @ plain - reference_iid) ** 2))
        iid_crs_total += float(np.mean((x_iid @ crs - reference_iid) ** 2))

    # Measured across these seeds: CRS ~37% and DGS ~42% lower output error.
    assert crs_total <= 0.85 * plain_total
    assert dgs_total <= 0.85 * plain_total
    assert dgs_total <= crs_total

    # And the honest other half of the trade, also asserted rather than
    # merely documented: under *uncorrelated* inputs the extra freedom cannot
    # help (plain nearest rounding is already optimal there), so FOCUS pays a
    # modest penalty instead. Measured ~9%; 40% is a loose ceiling on it.
    assert iid_crs_total >= iid_plain_total
    assert iid_crs_total <= 1.4 * iid_plain_total


# --------------------------------------------------------------------------
# No-op guards, matching tests/test_mx_quantization.py's own coverage.
# --------------------------------------------------------------------------


def test_focus_skips_non_block_divisible_k():
    model = _matmul_model(K=48, N=8, seed=7)  # 48 is not a multiple of 32
    focus = onnxsim.quantize_weight_only_mxfp4_focus(model, block_size=32)
    assert focus.SerializeToString() == model.SerializeToString()


def test_focus_skips_non_constant_weight():
    model = _model(
        """
        g (float[4,64] X, float[64,4] W) => (float[4,4] Y)
        {
          Y = MatMul(X, W)
        }
        """
    )
    focus = onnxsim.quantize_weight_only_mxfp4_focus(model)
    assert focus.SerializeToString() == model.SerializeToString()


def test_focus_skips_non_float32_weight():
    rng = np.random.default_rng(21)
    weight = rng.standard_normal((64, 8))
    model = _model(
        """
        g (double[batch,64] X) => (double[batch,8] Y)
        {
          Y = MatMul(X, W)
        }
        """,
        initializer=[onnx.numpy_helper.from_array(weight.astype(np.float64), "W")],
    )
    focus = onnxsim.quantize_weight_only_mxfp4_focus(model)
    assert focus.SerializeToString() == model.SerializeToString()


def test_focus_skip_names_leaves_matched_weight_untouched():
    rng = np.random.default_rng(8)
    w_base = rng.standard_normal((64, 16)).astype(np.float32) * 0.5
    w_other = rng.standard_normal((64, 4)).astype(np.float32) * 0.1
    model = _model(
        """
        g (float[batch,64] X) => (float[batch,16] Y, float[batch,4] H)
        {
          Y = MatMul(X, W)
          H = MatMul(X, W_other)
        }
        """,
        initializer=[_f32(w_base, "W"), _f32(w_other, "W_other")],
    )
    focus = onnxsim.quantize_weight_only_mxfp4_focus(
        model, block_size=32, skip_names=["W_other"]
    )
    onnx.checker.check_model(focus)

    names = {t.name for t in focus.graph.initializer}
    assert "W_mxfp4_q" in names
    assert "W_other_mxfp4_q" not in names
    other_out = next(
        onnx.numpy_helper.to_array(t)
        for t in focus.graph.initializer
        if t.name == "W_other"
    )
    assert np.array_equal(other_out, w_other)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sub_block_size": 7},  # does not divide block_size
        {"sub_block_size": 0},
        {"aggregate_error_weight": -1.0},
        {"coefficients": ()},
        {"coefficients": (1.0, 0.0)},
        {"coefficients": (-1.0,)},
    ],
)
def test_focus_rejects_invalid_configuration(kwargs):
    model = _matmul_model(K=64, N=8, seed=9)
    with pytest.raises(ValueError):
        onnxsim.quantize_weight_only_mxfp4_focus(model, block_size=32, **kwargs)
