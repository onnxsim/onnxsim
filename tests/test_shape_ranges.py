"""Tests for onnxsim.shape_ranges (ranged dimensions) and their use in onnxsim.interval.

Two layers: unit tests of the shape rules against NumPy/Python ground truth, and
soundness tests of ``interval.propagate`` on graphs with data-dependent ops
(``NonZero``, ``TopK`` with a runtime ``K``, ``Compress``, ``Unique``,
``NonMaxSuppression``, ``Range``, ``ConstantOfShape``) -- sample inputs, run
onnxruntime with every intermediate tensor exposed, and require each observed
*shape and value* to lie inside what the analysis claims.
"""

import contextlib
import io
import pathlib
import re

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import interval as I
from onnxsim import shape_ranges as S


def _model(body, initializer=None, opset=15, ir_version=8):
    model = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    model.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    onnx.checker.check_model(model)
    return model


def _all_tensors(model, feeds):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    produced = [o for n in m.graph.node for o in n.output if o]
    del m.graph.output[:]
    m.graph.output.extend(onnx.helper.make_empty_tensor_value_info(o) for o in produced)
    sess = ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return dict(zip(produced, sess.run(None, feeds)))


def _sparse(rng, shape, lo=-1.0, hi=1.0):
    """Uniform values with a random fraction of exact zeros, so NonZero's count varies."""
    x = rng.uniform(lo, hi, shape).astype(np.float32)
    return x * (rng.random(shape) < rng.uniform(0.15, 1.0))


def _assert_encloses(res, model, feeds, skip=()):
    for name, val in _all_tensors(model, feeds).items():
        if name in skip:
            continue
        known = name in res.intervals or name in res.ranged
        assert known, f"{name} has no interval or ranged shape"
        assert res.contains(name, val), (
            f"{name}: observed shape {val.shape} range "
            f"[{np.min(val) if val.size else None}, {np.max(val) if val.size else None}] "
            f"escaped {S.shape_str(res.shape(name))} hull {res.hull(name)}"
        )


# ---- shape rules vs ground truth ----------------------------------------------


def test_dim_arithmetic_and_set_operations():
    a, b = S.rng(2, 5), S.rng(0, None)
    assert (a + a) == S.rng(4, 10)
    assert (a * S.exact(3)) == S.rng(6, 15)
    assert (S.exact(0) * b).exact and (S.exact(0) * b).value == 0  # 0 * unbounded = 0
    assert S.intersect(a, S.rng(4, 9)) == S.rng(4, 5)
    assert S.intersect(a, S.rng(6, 9)) is None
    assert S.hull(a, S.rng(7, 9)) == S.rng(2, 9)
    assert a.contains(2) and not a.contains(6) and b.contains(10**9)
    with pytest.raises(ValueError):
        S.Dim(3, 2)


def test_broadcast_contains_every_numpy_result():
    rng = np.random.default_rng(0)
    for _ in range(300):
        rank = rng.integers(1, 4)
        a_shape = [int(rng.integers(1, 5)) for _ in range(rank)]
        b_shape = [d if rng.random() < 0.6 else 1 for d in a_shape]
        want = np.broadcast_shapes(tuple(a_shape), tuple(b_shape))
        a = tuple(
            S.rng(max(0, d - 1), d + 1) if rng.random() < 0.5 else S.exact(d)
            for d in a_shape
        )
        b = tuple(
            S.rng(1, d + 1) if rng.random() < 0.5 else S.exact(d) for d in b_shape
        )
        got = S.broadcast2(a, b)
        assert got is not None and S.contains_shape(got, want), (a, b, want)
    assert S.broadcast2((S.exact(3),), (S.exact(4),)) is None  # genuinely incompatible


def test_slice_rule_contains_every_python_slice():
    rng = np.random.default_rng(1)
    for _ in range(400):
        lo = int(rng.integers(0, 6))
        hi = lo + int(rng.integers(0, 6))
        start, end = int(rng.integers(-8, 8)), int(rng.integers(-8, 8))
        step = int(rng.choice([1, 1, 2, 3, -1, -2]))
        out = S.slice_((S.rng(lo, hi),), [start], [end], [0], [step])
        assert out is not None
        for n in range(lo, hi + 1):
            assert out[0].contains(len(range(n)[start:end:step])), (
                n,
                start,
                end,
                step,
                out,
            )


def test_slice_length_is_not_monotone_in_the_dim():
    # starts=-3, ends=2: the kept length *shrinks* as the dim grows past 5 -- so evaluating
    # only the endpoints of the dim range would be unsound.
    lens = [len(range(n)[-3:2]) for n in range(0, 10)]
    assert lens != sorted(lens)
    out = S.slice_((S.rng(0, 9),), [-3], [2])
    assert out[0].lo == min(lens) and out[0].hi == max(lens)


def test_reshape_infers_minus_one_over_a_ranged_element_count():
    shape = (S.rng(0, 12), S.exact(2))  # between 0 and 24 elements
    out = S.reshape(shape, np.array([-1, 4]), np.array([-1, 4]))
    assert out is not None and out[1].exact and out[1].value == 4
    for n in (0, 4, 8, 24):
        assert out[0].contains(n // 4)
    # a ranged entry that might be 0 or -1 must not be guessed
    assert S.reshape(shape, np.array([-1, 0]), np.array([1, 4])) is None
    # 0 copies the input dim
    z = S.reshape((S.exact(3), S.rng(1, 4)), np.array([0, -1]), np.array([0, -1]))
    assert z is not None and z[0].value == 3


def test_range_count_matches_numpy_arange():
    rng = np.random.default_rng(2)
    for _ in range(300):
        s = float(rng.integers(-5, 6))
        lim = float(rng.integers(-5, 12))
        d = float(rng.choice([1, 2, 3, -1, -2, 0.5]))
        got = S.range_count((s, s), (lim, lim), (d, d))
        assert got.contains(len(np.arange(s, lim, d))), (s, lim, d, got)
    # interval inputs: every concrete choice stays inside
    dim = S.range_count((0, 0), (0, 10), (1, 1))
    assert dim == S.rng(0, 10)
    assert S.range_count((0, 0), (5, 5), (-1, 1)) == S.Dim(
        0, None
    )  # delta may be 0: no bound


def test_data_dependent_shape_rules():
    x = (S.exact(3), S.exact(4))
    assert S.nonzero(x) == (S.exact(2), S.rng(0, 12))
    assert S.nonzero(x, value_hull=(0.5, 2.0))[1] == S.exact(
        12
    )  # excludes 0: all non-zero
    assert S.nonzero(x, value_hull=(-2.0, -0.1))[1] == S.exact(12)
    assert S.nonzero(x, value_hull=(0.0, 0.0))[1] == S.exact(0)
    assert S.nonzero(x, definitely=5, possibly=9)[1] == S.rng(5, 9)
    assert S.topk(x, -1, S.rng(1, 3)) == (S.exact(3), S.rng(1, 3))
    assert S.topk(x, 1, S.rng(2, 99)) == (
        S.exact(3),
        S.rng(2, 4),
    )  # K cannot exceed the dim
    assert S.compress(x, 0, S.exact(3)) == (S.rng(0, 3), S.exact(4))
    assert S.compress(x, None, S.exact(12)) == (S.rng(0, 12),)
    y, idx, inv, cnt = S.unique(x, None)
    assert y == (S.rng(1, 12),) and inv == (S.exact(12),)
    assert S.nms(
        (S.exact(1), S.exact(5), S.exact(4)),
        (S.exact(1), S.exact(2), S.exact(5)),
        S.exact(3),
    ) == (
        S.rng(0, 6),
        S.exact(3),
    )
    assert S.constant_of_shape(np.array([2.0, 0.0]), np.array([2.0, 7.0])) == (
        S.exact(2),
        S.rng(0, 7),
    )


def test_layout_rules():
    a = (S.exact(2), S.rng(0, 5))
    b = (S.exact(2), S.rng(3, 4))
    assert S.concat([a, b], 1) == (S.exact(2), S.rng(3, 9))
    assert S.gather(a, (S.exact(7),), 0) == (S.exact(7), S.rng(0, 5))
    assert S.gather_nd((S.exact(3), S.exact(4)), (S.rng(0, 9), S.exact(2))) == (
        S.rng(0, 9),
    )
    assert (
        S.gather_nd((S.exact(3), S.exact(4)), (S.exact(5), S.rng(1, 2))) is None
    )  # K not exact
    assert S.unsqueeze(a, [0]) == (S.exact(1),) + a
    assert S.squeeze((S.exact(1), S.rng(0, 5)), [0]) == (S.rng(0, 5),)
    assert (
        S.squeeze((S.rng(0, 1),), None) is None
    )  # might or might not be a 1: rank unknown
    assert S.matmul((S.exact(2), S.rng(0, 9)), (S.rng(0, 9), S.exact(3))) == (
        S.exact(2),
        S.exact(3),
    )
    assert S.reduce(a, [1], keepdims=False) == (S.exact(2),)
    assert S.reduced_count(a, [1]) == S.rng(0, 5)
    assert S.numel(a) == S.rng(0, 10)
    assert S.tile(a, (S.exact(2), S.exact(3))) == (S.exact(4), S.rng(0, 15))
    assert S.flatten((S.exact(2), S.rng(1, 3), S.exact(4)), 1) == (
        S.exact(2),
        S.rng(4, 12),
    )


# ---- interval.propagate: soundness on data-dependent graphs ---------------------

_NZ_GATHER = """
m (float[3,4] x) => (float s) {
  nz = NonZero(x)
  shp = Constant<value=int64[1] {-1}>()
  f = Reshape(x, shp)
  nz1 = NonZero(f)
  ax = Constant<value=int64[1] {0}>()
  idx = Squeeze(nz1, ax)
  g = Gather(f, idx)
  s = ReduceSum<keepdims=0>(g)
}"""


def test_nonzero_gather_reducesum_is_sound_and_ranged():
    model = _model(_NZ_GATHER)
    res = I.propagate(model, {"x": (-1.0, 1.0)})
    assert res.unsupported == []
    assert [str(d) for d in res.shape("nz")] == ["2", "[0,12]"]
    assert res.ranged["nz"].hull == (
        0.0,
        3.0,
    )  # row i holds an index in [0, shape[i]-1]
    assert "g" in res.ranged and res.ranged["g"].hull == (-1.0, 1.0)  # data hull
    assert "s" in res.intervals and res.hull("s") == (
        -12.0,
        12.0,
    )  # <= 12 items in [-1, 1]
    assert {"nz", "nz1", "idx", "g"} <= set(res.ranged_names)
    rng = np.random.default_rng(3)
    for _ in range(80):
        _assert_encloses(res, model, {"x": _sparse(rng, (3, 4))})


def test_all_positive_input_makes_nonzero_exact():
    model = _model(_NZ_GATHER)
    res = I.propagate(model, {"x": (0.5, 1.0)})
    # every element is provably non-zero: N is exactly numel, so the shape is static again
    assert "nz1" in res.intervals and res.intervals["nz1"][0].shape == (1, 12)
    assert res.ranged.get("nz1") is None
    lo, hi = res.hull("s")
    assert lo == pytest.approx(6.0) and hi == pytest.approx(
        12.0
    )  # 12 items in [0.5, 1]
    rng = np.random.default_rng(4)
    for _ in range(40):
        _assert_encloses(
            res, model, {"x": rng.uniform(0.5, 1.0, (3, 4)).astype(np.float32)}
        )


def test_all_zero_input_makes_nonzero_empty_and_sum_zero():
    res = I.propagate(_model(_NZ_GATHER), {"x": (0.0, 0.0)})
    assert res.intervals["nz1"][0].shape == (1, 0)
    assert res.hull("s") == (0.0, 0.0)


def test_per_element_ranges_count_definitely_and_possibly_nonzero():
    model = _model("m (float[2,3] x) => (int64[2,?] nz) { nz = NonZero(x) }")
    lo = np.array([[1.0, 0.0, -2.0], [0.0, 0.0, -1.0]])
    hi = np.array([[2.0, 0.0, -1.0], [3.0, 0.0, 1.0]])
    res = I.propagate(model, {"x": (lo, hi)})
    # definitely non-zero: (0,0) and (0,2); certainly zero: (0,1) and (1,1); the rest might be
    assert [str(d) for d in res.shape("nz")] == ["2", "[2,4]"]
    rng = np.random.default_rng(5)
    for _ in range(80):
        x = rng.uniform(lo, hi).astype(np.float32)
        x[(rng.random(x.shape) < 0.5) & (lo <= 0) & (hi >= 0)] = 0.0
        _assert_encloses(res, model, {"x": x})


def test_nonzero_into_gathernd_and_matmul_contraction_over_the_dynamic_dim():
    model = _model(
        """
        m (float[3,4] x) => (float p) {
          nz = NonZero(x)
          t = Transpose<perm=[1,0]>(nz)
          g = GatherND(x, t)
          nzf = Cast<to=1>(nz)
          nzt = Transpose<perm=[1,0]>(nzf)
          p = MatMul(nzf, nzt)
        }"""
    )
    res = I.propagate(model, {"x": (-1.0, 1.0)})
    assert [str(d) for d in res.shape("t")] == ["[0,12]", "2"]
    assert [str(d) for d in res.shape("g")] == ["[0,12]"]
    # contraction over N in [0, 12] items, each product in [0, 3]*[0, 3]: sum in [0, 108]
    assert "p" in res.intervals and res.intervals["p"][0].shape == (2, 2)
    assert res.hull("p") == (0.0, 108.0)
    rng = np.random.default_rng(6)
    for _ in range(60):
        _assert_encloses(res, model, {"x": _sparse(rng, (3, 4))})


def test_reductions_over_a_possibly_empty_dim_fall_back_soundly():
    model = _model(
        """
        m (float[3,4] x) => (float mean, float mx) {
          nz = NonZero(x)
          nzf = Cast<to=1>(nz)
          mean = ReduceMean<keepdims=0>(nzf)
          mx = ReduceMax<keepdims=0>(nzf)
        }"""
    )
    res = I.propagate(model, {"x": (-1.0, 1.0)})
    # N can be 0, and the mean/max of nothing is undefined: no finite claim is made
    assert res.hull("mean") == (-np.inf, np.inf) and res.hull("mx") == (-np.inf, np.inf)
    # with at least one provable non-zero element the hull is the data hull
    res2 = I.propagate(model, {"x": (0.5, 1.0)})
    lo, hi = res2.hull("mean")
    assert lo == 0.0 and hi == pytest.approx(3.0)


def test_unsupported_op_on_a_ranged_tensor_leaves_outputs_unknown():
    model = _model(
        """
        m (float[3,4] x) => (float[?,?] y) {
          nz = NonZero(x)
          nzf = Cast<to=1>(nz)
          y = Trilu(nzf)
        }"""
    )
    res = I.propagate(model, {"x": (-1.0, 1.0)})
    assert "Trilu" in res.unsupported
    assert "y" not in res.intervals and "y" not in res.ranged  # unknown, never guessed
    assert "nzf" in res.ranged


def test_topk_runtime_k_is_sound():
    model = _model(
        """
        m (float[2,6] x, int64[1] k) => (float[2,?] v, int64[2,?] i) {
          v, i = TopK(x, k)
        }"""
    )
    res = I.propagate(model, {"x": (-2.0, 3.0), "k": (1, 4)})
    assert [str(d) for d in res.shape("v")] == ["2", "[1,4]"]
    assert res.ranged["v"].hull == (-2.0, 3.0)
    assert res.ranged["i"].hull == (0.0, 5.0)
    rng = np.random.default_rng(7)
    for _ in range(60):
        feeds = {
            "x": rng.uniform(-2, 3, (2, 6)).astype(np.float32),
            "k": np.array([rng.integers(1, 5)], dtype=np.int64),
        }
        _assert_encloses(res, model, feeds)


@pytest.mark.parametrize("largest", [1, 0])
def test_topk_constant_k_uses_order_statistic_bounds(largest):
    model = _model(
        f"""
        m (float[2,5] x) => (float[2,3] v, int64[2,3] i) {{
          k = Constant<value=int64[1] {{3}}>()
          v, i = TopK<largest={largest}>(x, k)
        }}"""
    )
    rng = np.random.default_rng(8)
    lo = rng.uniform(-3, 0, (2, 5))
    hi = lo + rng.uniform(0, 3, (2, 5))
    res = I.propagate(model, {"x": (lo, hi)})
    v_lo, v_hi = res.intervals["v"]
    assert v_lo.shape == (2, 3) and "v" not in res.ranged
    # j-th largest lies between the j-th largest lower bound and the j-th largest upper
    # bound (j-th smallest likewise)
    if largest:
        want_lo, want_hi = -np.sort(-lo, axis=1)[:, :3], -np.sort(-hi, axis=1)[:, :3]
    else:
        want_lo, want_hi = np.sort(lo, axis=1)[:, :3], np.sort(hi, axis=1)[:, :3]
    np.testing.assert_allclose(v_lo, want_lo)
    np.testing.assert_allclose(v_hi, want_hi)
    assert res.hull("i") == (0.0, 4.0)
    for _ in range(80):
        _assert_encloses(res, model, {"x": rng.uniform(lo, hi).astype(np.float32)})


def test_compress_sharpens_with_a_known_condition_and_is_sound():
    body = """
        m (float[6] x, bool[6] c) => (float[?] y) {
          y = Compress<axis=0>(x, c)
        }"""
    model = _model(body)
    rng = np.random.default_rng(9)
    for rng_c, lo_n, hi_n in (((1, 1), 6, 6), ((0, 1), 0, 6), ((0, 0), 0, 0)):
        res = I.propagate(model, {"x": (-1.0, 1.0), "c": rng_c})
        d = res.shape("y")[0]
        assert (d.lo, d.hi) == (lo_n, hi_n), (rng_c, d)
        for _ in range(40):
            if rng_c == (1, 1):
                c = np.ones(6, dtype=bool)
            elif rng_c == (0, 0):
                c = np.zeros(6, dtype=bool)
            else:
                c = rng.random(6) < 0.5
            _assert_encloses(
                res, model, {"x": rng.uniform(-1, 1, 6).astype(np.float32), "c": c}
            )


def test_compare_nonzero_gather_pipeline_is_sound():
    # "where-compress" in the usual ONNX spelling: indices of the elements passing a test
    model = _model(
        """
        m (float[3,4] x) => (float[?] sel) {
          zero = Constant<value=float {0.25}>()
          c = Greater(x, zero)
          nz = NonZero(c)
          t = Transpose<perm=[1,0]>(nz)
          sel = GatherND(x, t)
        }"""
    )
    res = I.propagate(model, {"x": (-1.0, 1.0)})
    assert "sel" in res.ranged and res.ranged["sel"].hull == (-1.0, 1.0)
    assert [str(d) for d in res.shape("sel")] == ["[0,12]"]
    rng = np.random.default_rng(10)
    for _ in range(60):
        _assert_encloses(
            res, model, {"x": rng.uniform(-1, 1, (3, 4)).astype(np.float32)}
        )


def test_unique_and_nms_are_sound():
    uniq = _model("m (float[2,3] x) => (float[?] y) { y, idx, inv, cnt = Unique(x) }")
    res = I.propagate(uniq, {"x": (0.0, 3.0)})
    assert [str(d) for d in res.shape("y")] == ["[1,6]"]
    rng = np.random.default_rng(11)
    for _ in range(40):
        x = rng.integers(0, 4, (2, 3)).astype(np.float32)
        _assert_encloses(res, uniq, {"x": x})

    nms = _model(
        """
        m (float[1,5,4] boxes, float[1,2,5] scores) => (int64[?,3] sel) {
          mx = Constant<value=int64 {3}>()
          sel = NonMaxSuppression(boxes, scores, mx)
        }"""
    )
    res = I.propagate(nms, {"boxes": (0.0, 1.0), "scores": (0.0, 1.0)})
    assert [str(d) for d in res.shape("sel")] == ["[0,6]", "3"]
    assert res.ranged["sel"].hull == (0.0, 4.0)  # batch 0, class 0..1, box 0..4
    for _ in range(40):
        b = rng.uniform(0, 1, (1, 5, 4)).astype(np.float32)
        b[..., 2:] = b[..., :2] + rng.uniform(0.05, 0.5, (1, 5, 2)).astype(np.float32)
        _assert_encloses(
            res,
            nms,
            {"boxes": b, "scores": rng.uniform(0, 1, (1, 2, 5)).astype(np.float32)},
        )


def test_range_and_constant_of_shape_follow_ranged_extents():
    model = _model(
        """
        m (int64 n, float[3,4] x) => (int64[?] r, float[2,?] c) {
          zero = Constant<value=int64 {0}>()
          one = Constant<value=int64 {1}>()
          r = Range(zero, n, one)
          nz = NonZero(x)
          s = Shape(nz)
          c = ConstantOfShape<value=float[1] {1.5}>(s)
        }"""
    )
    res = I.propagate(model, {"n": (0, 10), "x": (-1.0, 1.0)})
    assert [str(d) for d in res.shape("r")] == ["[0,10]"]
    assert res.ranged["r"].hull == (0.0, 10.0)
    assert [str(d) for d in res.shape("c")] == ["2", "[0,12]"]
    assert res.ranged["c"].hull == (1.5, 1.5)
    rng = np.random.default_rng(12)
    for _ in range(40):
        feeds = {
            "n": np.array(int(rng.integers(0, 11)), dtype=np.int64),
            "x": _sparse(rng, (3, 4)),
        }
        _assert_encloses(res, model, feeds)


def test_integer_division_of_a_ranged_count_stays_a_superset_of_truncation():
    model = _model(
        """
        m (float[3,4] x) => (int64 q) {
          nz = NonZero(x)
          s = Shape(nz)
          i1 = Constant<value=int64 {1}>()
          n = Gather(s, i1)
          five = Constant<value=int64 {5}>()
          q = Div(n, five)
        }"""
    )
    res = I.propagate(model, {"x": (-1.0, 1.0)})
    lo, hi = res.hull("q")
    assert (
        lo == 0.0 and 3.0 <= hi < 3.0001
    )  # real quotient hull is [0, 2.4]; truncation widens
    rng = np.random.default_rng(13)
    for _ in range(60):
        _assert_encloses(res, model, {"x": _sparse(rng, (3, 4))})
    # the widening is only for tensors derived from ranged ones: a plain float Div is untouched
    plain = _model("m (float[2] x, float[2] y) => (float z) { z = Div(x, y) }")
    r = I.propagate(plain, {"x": (0.0, 3.0), "y": (2.0, 2.0)})
    assert r.hull("z")[1] == pytest.approx(1.5)


def test_dynamic_input_dim_becomes_ranged_and_reduces_soundly():
    model = _model(
        """
        m (float[N,4] x) => (float[4] s) {
          r = Relu(x)
          s = ReduceSum<keepdims=0>(r, axes)
        }""",
        {"axes": np.array([0], dtype=np.int64)},
    )
    res = I.propagate(model, {"x": (0.0, 1.0)})
    assert "x" in res.ranged and str(res.ranged["x"].shape[0]) == "N:[0,inf]"
    assert res.hull("s") == (0.0, np.inf)  # unbounded batch: unbounded sum, honestly
    res = I.propagate(model, {"x": (0.0, 1.0)}, input_shapes={"x": [(1, 8), 4]})
    lo, hi = res.hull("s")
    assert lo == 0.0 and hi == pytest.approx(8.0)
    rng = np.random.default_rng(14)
    for _ in range(40):
        n = int(rng.integers(1, 9))
        _assert_encloses(
            res, model, {"x": rng.uniform(0, 1, (n, 4)).astype(np.float32)}
        )


def test_static_graphs_are_unaffected():
    rng = np.random.default_rng(15)
    k = 4
    model = _model(
        """
        m (float[1,3,6,6] x) => (float[1,5] y) {
          c = Conv<pads=[1,1,1,1]>(x, W, B)
          r = Relu(c)
          a = Add(r, c)
          p = GlobalAveragePool(a)
          f = Flatten(p)
          y = Gemm<transB=1>(f, W2, B2)
        }""",
        dict(
            W=rng.standard_normal((k, 3, 3, 3)).astype(np.float32),
            B=rng.standard_normal(k).astype(np.float32),
            W2=rng.standard_normal((5, k)).astype(np.float32),
            B2=rng.standard_normal(5).astype(np.float32),
        ),
    )
    res = I.propagate(model, {"x": (-1.0, 1.0)})
    assert res.ranged == {} and res.ranged_names == []
    assert res.unsupported == []
    for name in ("c", "r", "a", "p", "f", "y"):
        assert name in res.intervals and res.intervals[name][0].dtype == np.float64
    for _ in range(20):
        _assert_encloses(
            res, model, {"x": rng.uniform(-1, 1, (1, 3, 6, 6)).astype(np.float32)}
        )
    # IntervalResult built the old way (two positional fields) still works
    legacy = I.IntervalResult(res.intervals, [])
    assert legacy.ranged == {} and legacy.hull("y") == res.hull("y")


def test_doc_snippets_run_and_print_what_the_doc_says():
    doc = pathlib.Path(__file__).resolve().parents[1] / "docs" / "shape-ranges.md"
    text = doc.read_text()
    pairs = re.findall(r"```python\n(.*?)```\n\n```text\n(.*?)```", text, flags=re.S)
    assert len(pairs) >= 4
    namespace: dict = {}
    for code, expected in pairs:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, str(doc), "exec"), namespace)
        assert out.getvalue() == expected, f"doc output drifted for:\n{code}"
