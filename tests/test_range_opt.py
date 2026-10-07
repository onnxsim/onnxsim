"""Tests for onnxsim.range_opt (range-driven simplification and dtype narrowing).

Every rule gets a positive case, a negative case where the interval does NOT justify it (it must not
fire), and an equivalence check: the rewritten model equals the original on sampled inputs inside the
declared box (onnxruntime, float rounding only). Where the model relies on the box, an input outside
it is shown to change the output -- that is what the recorded precondition guards.
"""

import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper, parser

from onnxsim import range_opt as RO

INF = float("inf")


def _model(body, initializer=None, opset=15, ir_version=8):
    m = parser.parse_model(
        f'<ir_version: {ir_version}, opset_import: ["" : {opset}]> {body}'
    )
    m.graph.initializer.extend(
        numpy_helper.from_array(v, k) for k, v in (initializer or {}).items()
    )
    return m


def _session(m):
    return ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def _sample(m, box, rng):
    feeds = {}
    for vi in m.graph.input:
        tt = vi.type.tensor_type
        shape = [d.dim_value for d in tt.shape.dim]
        lo, hi = box[vi.name]
        if tt.elem_type in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
            dt = np.int64 if tt.elem_type == onnx.TensorProto.INT64 else np.int32
            feeds[vi.name] = rng.integers(int(lo), int(hi) + 1, size=shape).astype(dt)
        else:
            feeds[vi.name] = (lo + (hi - lo) * rng.random(shape)).astype(np.float32)
    return feeds


def _assert_equivalent(orig, new, box, n=40, seed=0, rtol=1e-5, atol=1e-6):
    """new == orig on ``n`` inputs sampled inside ``box`` (float rounding only)."""
    rng = np.random.default_rng(seed)
    so, sn = _session(orig), _session(new)
    for _ in range(n):
        feeds = _sample(orig, box, rng)
        a = so.run(None, feeds)
        nf = {
            vi.name: feeds[vi.name].astype(
                np.int32
                if vi.type.tensor_type.elem_type == onnx.TensorProto.INT32
                else feeds[vi.name].dtype
            )
            for vi in new.graph.input
        }
        b = sn.run(None, nf)
        for x, y in zip(a, b):
            np.testing.assert_allclose(y, x, rtol=rtol, atol=atol)


def _ops(m):
    return [n.op_type for n in m.graph.node]


def _fired(props, rule):
    return [p for p in props if p.rule == rule and p.applies]


def _meta(m, key):
    return next((p.value for p in m.metadata_props if p.key == key), None)


# ---- dead Relu ------------------------------------------------------------------------------

_RELU = "m (float[4] x) => (float[4] y) { a = Add(x, C)  y = Relu(a) }"
_C = {"C": np.full(4, 0.5, np.float32)}


def test_dead_relu_fires_only_when_the_interval_proves_it():
    m = _model(_RELU, _C)
    box = {"x": (1.0, 2.0)}
    props = RO.analyze(m, box)
    assert len(_fired(props, "dead_relu")) == 1
    new, log = RO.apply(m, box)
    assert "Relu" not in _ops(new) and log[0]["rule"] == "dead_relu"
    assert log[0]["proof"]["hull"] == pytest.approx([1.5, 2.5])
    _assert_equivalent(m, new, box)
    # x in [-1, 2] gives a + 0.5 in [-0.5, 2.5]: not provably non-negative
    assert not _fired(RO.analyze(m, {"x": (-1.0, 2.0)}), "dead_relu")
    # no range -> unbounded -> nothing provable
    assert not _fired(RO.analyze(m, {}), "dead_relu")


def test_relu_after_sigmoid_is_unconditional_and_needs_no_precondition():
    m = _model("m (float[4] x) => (float[4] y) { s = Sigmoid(x)  y = Relu(s) }")
    box = {"x": (-3.0, 3.0)}
    (rw,) = _fired(RO.analyze(m, box), "dead_relu")
    assert rw.unconditional
    new, _ = RO.apply(m, box)
    assert "Relu" not in _ops(new)
    assert not any(p.key.startswith(RO.PRECONDITION_PREFIX) for p in new.metadata_props)
    _assert_equivalent(m, new, box)


def test_margin_demands_a_gap():
    m = _model("m (float[4] x) => (float[4] y) { y = Relu(x) }")
    assert _fired(RO.analyze(m, {"x": (0.0, 1.0)}), "dead_relu")
    assert not _fired(RO.analyze(m, {"x": (0.0, 1.0)}, margin=0.1), "dead_relu")
    assert _fired(RO.analyze(m, {"x": (0.2, 1.0)}, margin=0.1), "dead_relu")


def test_bypass_keeps_a_graph_output_name_with_an_identity():
    m = _model("m (float[4] x) => (float[4] y) { y = Relu(x) }")
    new, _ = RO.apply(m, {"x": (1.0, 2.0)})
    assert _ops(new) == ["Identity"] and new.graph.output[0].name == "y"
    _assert_equivalent(m, new, {"x": (1.0, 2.0)})


def test_outside_the_box_the_rewritten_model_differs_which_the_guard_catches():
    m = _model(_RELU, _C)
    box = {"x": (1.0, 2.0)}
    new, _ = RO.apply(m, box)
    bad = {"x": np.full(4, -1.0, np.float32)}
    assert not np.allclose(
        _session(m).run(None, bad)[0], _session(new).run(None, bad)[0]
    )
    with pytest.raises(RO.PreconditionViolation, match="leaves the declared box"):
        RO.check_precondition(new, bad)
    assert len(RO.check_precondition(new, bad, raise_on_violation=False)) == 1
    good = {"x": np.full(4, 1.5, np.float32)}
    assert RO.check_precondition(new, good) == []


# ---- Abs ------------------------------------------------------------------------------------


def test_abs_identity_neg_or_nothing():
    m = _model("m (float[4] x) => (float[4] y) { y = Abs(x) }")
    new, _ = RO.apply(m, {"x": (1.0, 2.0)})
    assert _ops(new) == ["Identity"]
    _assert_equivalent(m, new, {"x": (1.0, 2.0)})
    new, log = RO.apply(m, {"x": (-3.0, -1.0)})
    assert _ops(new) == ["Neg"] and log[0]["rule"] == "dead_abs"
    _assert_equivalent(m, new, {"x": (-3.0, -1.0)})
    assert not _fired(RO.analyze(m, {"x": (-1.0, 1.0)}), "dead_abs")


# ---- Clip -----------------------------------------------------------------------------------

_RELU6 = "m (float[4] x) => (float[4] y) { s = Sigmoid(x)  a = Mul(s, K)  y = Clip(a, LO, HI) }"
_CLIP_INIT = {"LO": np.float32(0.0), "HI": np.float32(6.0)}


def test_dead_clip_removed_entirely():
    m = _model(_RELU6, {"K": np.float32(5.0), **_CLIP_INIT})
    box = {"x": (-2.0, 2.0)}
    new, log = RO.apply(m, box)
    assert "Clip" not in _ops(new)
    assert any(r["rule"] == "dead_clip" and r["unconditional"] for r in log)
    _assert_equivalent(m, new, box)


def test_clip_drops_only_the_redundant_bound():
    m = _model(_RELU6, {"K": np.float32(7.0), **_CLIP_INIT})  # Sigmoid * 7 in [0, 7]
    box = {"x": (-9.0, 9.0)}
    new, _ = RO.apply(m, box)
    clip = next(n for n in new.graph.node if n.op_type == "Clip")
    assert clip.input[1] == "" and clip.input[2] == "HI"  # min dropped, max kept
    _assert_equivalent(m, new, box)
    # and the max bound really is still needed: the value reaches above 6
    sess = _session(new)
    assert sess.run(None, {"x": np.full(4, 9.0, np.float32)})[0].max() <= 6.0


def test_unconditional_means_the_same_rewrite_not_a_weaker_one():
    # Clip(Relu(Add(x, c)), 0, 6): with every input unbounded only the *min* bound is provably
    # redundant (a Relu output is >= 0). Removing the whole Clip needs x in the box, so it must
    # NOT be labelled unconditional -- otherwise no precondition would be recorded for it.
    m = _model(
        "m (float[4] x) => (float[4] y) { a = Add(x, C)  b = Relu(a)  y = Clip(b, LO, HI) }",
        {"C": np.full(4, 0.5, np.float32), **_CLIP_INIT},
    )
    props = RO.analyze(m, {"x": (1.0, 2.0)})
    (clip,) = [p for p in props if p.rule == "dead_clip"]
    assert clip.action[0] == "bypass" and not clip.unconditional
    new, log = RO.apply(m, {"x": (1.0, 2.0)})
    assert _ops(new) == ["Add", "Identity"]
    assert RO.preconditions(new)  # the box the whole-Clip removal relies on is recorded
    with pytest.raises(RO.PreconditionViolation):
        RO.check_precondition(new, {"x": np.full(4, 50.0, np.float32)})
    # and the weaker rewrite alone really is unconditional
    only = RO.analyze(m, {"x": (-1e9, 1e9)})
    assert [
        p
        for p in only
        if p.rule == "dead_clip" and p.action[0] == "clip_drop" and p.unconditional
    ]


def test_clip_not_touched_when_neither_bound_is_proved():
    m = _model(_RELU6, {"K": np.float32(7.0), **_CLIP_INIT})
    m2 = _model("m (float[4] x) => (float[4] y) { y = Clip(x, LO, HI) }", _CLIP_INIT)
    assert not _fired(RO.analyze(m2, {"x": (-1.0, 9.0)}), "dead_clip")
    assert RO.analyze(m2, {"x": (1.0, 5.0)})  # fully inside: removable
    assert _fired(RO.analyze(m, {"x": (-9.0, 9.0)}), "dead_clip")  # one-sided only


def test_clip_attribute_form_opset9():
    m = _model(
        "m (float[4] x) => (float[4] y) { y = Clip<min=0.0, max=6.0>(x) }", opset=9
    )
    new, _ = RO.apply(m, {"x": (1.0, 5.0)})
    assert _ops(new) == ["Identity"]
    _assert_equivalent(m, new, {"x": (1.0, 5.0)})
    new, _ = RO.apply(m, {"x": (1.0, 9.0)})  # min redundant -> attribute removed
    clip = next(n for n in new.graph.node if n.op_type == "Clip")
    assert [a.name for a in clip.attribute] == ["max"]
    _assert_equivalent(m, new, {"x": (1.0, 9.0)})


# ---- Min / Max ------------------------------------------------------------------------------


def test_max_and_min_against_constants():
    mx = _model(
        "m (float[4] x) => (float[4] y) { y = Max(x, C) }",
        {"C": np.full(4, 1.0, np.float32)},
    )
    new, _ = RO.apply(mx, {"x": (2.0, 3.0)})
    assert _ops(new) == ["Identity"]
    _assert_equivalent(mx, new, {"x": (2.0, 3.0)})
    assert not _fired(RO.analyze(mx, {"x": (0.0, 3.0)}), "dead_minmax")
    mn = _model(
        "m (float[4] x) => (float[4] y) { y = Min(x, C) }",
        {"C": np.full(4, 1.0, np.float32)},
    )
    new, _ = RO.apply(mn, {"x": (0.0, 0.5)})
    assert _ops(new) == ["Identity"]
    assert not _fired(RO.analyze(mn, {"x": (0.0, 2.0)}), "dead_minmax")


def test_max_of_two_tensors_and_broadcast_guard():
    m = _model("m (float[4] a, float[4] b) => (float[4] y) { y = Max(a, b) }")
    box = {"a": (5.0, 6.0), "b": (0.0, 1.0)}
    new, _ = RO.apply(m, box)
    assert _ops(new) == ["Identity"]
    _assert_equivalent(m, new, box)
    # b broadcasts a: the output has b's shape, so a cannot stand in for it
    mb = _model("m (float[1] a, float[4] b) => (float[4] y) { y = Max(a, b) }")
    assert not _fired(RO.analyze(mb, {"a": (5.0, 6.0), "b": (0.0, 1.0)}), "dead_minmax")


# ---- Where / If -----------------------------------------------------------------------------

_WHERE = "m (float[4] x, float[4] a, float[4] b) => (float[4] y) { c = Greater(x, T)  y = Where(c, a, b) }"


def test_decided_where_takes_the_branch():
    m = _model(_WHERE, {"T": np.full(4, 1.0, np.float32)})
    box = {"x": (2.0, 3.0), "a": (-1.0, 1.0), "b": (-1.0, 1.0)}
    new, log = RO.apply(m, box)
    assert "Where" not in _ops(new) and log[0]["proof"]["decided"] is True
    _assert_equivalent(m, new, box)
    box_else = {"x": (-3.0, 0.5), "a": (-1.0, 1.0), "b": (-1.0, 1.0)}
    new, log = RO.apply(m, box_else)
    assert log[0]["proof"]["decided"] is False
    _assert_equivalent(m, new, box_else)
    undecided = {"x": (0.0, 3.0), "a": (-1.0, 1.0), "b": (-1.0, 1.0)}
    assert not _fired(RO.analyze(m, undecided), "decided_where")


def test_decided_where_through_not_and_logic():
    m = _model(
        "m (float[4] x, float[4] a, float[4] b) => (float[4] y) "
        "{ c = Less(x, T)  n = Not(c)  y = Where(n, a, b) }",
        {"T": np.full(4, 1.0, np.float32)},
    )
    box = {"x": (2.0, 3.0), "a": (0.0, 1.0), "b": (0.0, 1.0)}
    new, _ = RO.apply(m, box)
    assert "Where" not in _ops(new)
    _assert_equivalent(m, new, box)


def test_where_broadcast_guard():
    m = _model(
        "m (float[4] x, float[1] a, float[4] b) => (float[4] y) "
        "{ c = Greater(x, T)  y = Where(c, a, b) }",
        {"T": np.full(4, 1.0, np.float32)},
    )
    box = {"x": (2.0, 3.0), "a": (0.0, 1.0), "b": (0.0, 1.0)}
    assert not _fired(
        RO.analyze(m, box), "decided_where"
    )  # a would change the output shape


_IF = """m (float[3] x, float[1] s) => (float[3] y) {
  t = Constant<value = float[1] {1.0}>()
  c = Greater(s, t)
  y = If<then_branch = gt () => (float[3] o) { r = Relu(x)  o = Neg(r) },
         else_branch = ge () => (float[3] o2) { o2 = Abs(x) }>(c)
}"""


def test_decided_if_is_inlined_to_the_taken_branch():
    m = _model(_IF)
    for s_box, kept in (((2.0, 3.0), "Neg"), ((-3.0, 0.5), "Abs")):
        box = {"x": (-2.0, 2.0), "s": s_box}
        new, log = RO.apply(m, box)
        assert "If" not in _ops(new) and kept in _ops(new)
        assert any(r["rule"] == "decided_if" for r in log)
        onnx.checker.check_model(new)
        _assert_equivalent(m, new, box)
    assert not _fired(RO.analyze(m, {"x": (-2.0, 2.0), "s": (0.0, 3.0)}), "decided_if")


def test_if_with_nested_subgraph_is_left_alone():
    nested = """m (float[3] x, float[1] s) => (float[3] y) {
  t = Constant<value = float[1] {1.0}>()
  c = Greater(s, t)
  y = If<then_branch = gt () => (float[3] o) {
           d = Greater(s, t)
           o = If<then_branch = g2 () => (float[3] p) { p = Relu(x) },
                  else_branch = g3 () => (float[3] q) { q = Neg(x) }>(d) },
         else_branch = ge () => (float[3] o2) { o2 = Abs(x) }>(c)
}"""
    m = _model(nested)
    assert not _fired(RO.analyze(m, {"x": (-2.0, 2.0), "s": (2.0, 3.0)}), "decided_if")


# ---- Cast -----------------------------------------------------------------------------------


def test_cast_noop():
    m = _model("m (float[4] x) => (float[4] y) { y = Cast<to=1>(x) }")
    new, log = RO.apply(m, {"x": (0.0, 1.0)})
    assert _ops(new) == ["Identity"] and log[0]["rule"] == "cast_noop"


def test_cast_roundtrip_lossless_widening_is_unconditional():
    m = _model(
        "m (int32[4] x) => (int32[4] y) { w = Cast<to=7>(x)  y = Cast<to=6>(w) }"
    )
    new, log = RO.apply(m, {"x": (0, 5)})
    assert _ops(new) == ["Identity"] and log[0]["unconditional"]
    _assert_equivalent(m, new, {"x": (0, 5)})
    mf = _model(
        "m (float[4] x) => (float[4] y) { w = Cast<to=11>(x)  y = Cast<to=1>(w) }"
    )
    new, log = RO.apply(mf, {"x": (-1.0, 1.0)})
    assert _ops(new) == ["Identity"] and log[0]["unconditional"]


def test_cast_roundtrip_through_a_narrower_type_needs_the_range():
    m = _model(
        "m (int64[4] x) => (int64[4] y) { n = Cast<to=6>(x)  y = Cast<to=7>(n) }"
    )
    new, log = RO.apply(m, {"x": (-100, 100)})
    assert _ops(new) == ["Identity"] and not log[0]["unconditional"]
    _assert_equivalent(m, new, {"x": (-100, 100)})
    assert not _fired(RO.analyze(m, {"x": (0, 2**33)}), "cast_roundtrip")
    assert not _fired(RO.analyze(m, {}), "cast_roundtrip")


def test_cast_roundtrip_refuses_lossy_float16():
    m = _model(
        "m (float[4] x) => (float[4] y) { h = Cast<to=10>(x)  y = Cast<to=1>(h) }"
    )
    assert not _fired(RO.analyze(m, {"x": (-1.0, 1.0)}), "cast_roundtrip")


# ---- integer narrowing ----------------------------------------------------------------------

_GATHER = "m (float[10,4] d) => (float[3,4] y) { y = Gather(d, I) }"


def test_narrow_const_gather_indices():
    m = _model(_GATHER, {"I": np.array([0, 5, 9], np.int64)})
    new, log = RO.apply(m, {"d": (0.0, 1.0)})
    idx = next(t for t in new.graph.initializer if t.name == "I")
    assert idx.data_type == onnx.TensorProto.INT32
    assert log[0]["rule"] == "narrow_const_index" and log[0]["unconditional"]
    _assert_equivalent(m, new, {"d": (0.0, 1.0)})
    big = _model(_GATHER, {"I": np.array([0, 5, 2**40], np.int64)})
    assert not _fired(RO.analyze(big, {"d": (0.0, 1.0)}), "narrow_const_index")


def test_narrow_skips_a_tensor_that_something_else_needs_as_int64():
    m = _model(
        "m (float[10,4] d) => (float[3,4] y, float[12] z) "
        "{ y = Gather(d, I)  r = Reshape(d, S)  z = Reshape(y, S2) }",
        {
            "I": np.array([0, 5, 9], np.int64),
            "S": np.array([40], np.int64),
            "S2": np.array([12], np.int64),
        },
    )
    assert any(p.proof.get("tensor") == "I" for p in RO.analyze(m, {"d": (0.0, 1.0)}))
    # S is only a Reshape shape operand (int64 required): never narrowed
    assert not any(
        p.proof.get("tensor") in ("S", "S2") for p in RO.analyze(m, {"d": (0.0, 1.0)})
    )
    shared = _model(
        "m (float[10,4] d) => (float[3,4] y, float[3] z) "
        "{ y = Gather(d, I)  z = Cast<to=1>(I) }",
        {"I": np.array([0, 5, 9], np.int64)},
    )
    assert not any(
        p.proof.get("tensor") == "I" for p in RO.analyze(shared, {"d": (0.0, 1.0)})
    )


def test_slice_bounds_are_narrowed_and_open_ends_clamped():
    m = _model(
        "m (float[10,4] d) => (float[?,?] y) { y = Slice(d, ST, EN, AX) }",
        {
            "ST": np.array([2], np.int64),
            "EN": np.array([2**63 - 1], np.int64),  # "to the end"
            "AX": np.array([0], np.int64),
        },
    )
    new, log = RO.apply(m, {"d": (0.0, 1.0)})
    types = {t.name: t.data_type for t in new.graph.initializer}
    assert all(types[k] == onnx.TensorProto.INT32 for k in ("ST", "EN", "AX"))
    en = numpy_helper.to_array(next(t for t in new.graph.initializer if t.name == "EN"))
    assert en[0] == 2**31 - 1 and any(r["proof"].get("clamped") for r in log)
    _assert_equivalent(m, new, {"d": (0.0, 1.0)})


def test_slice_with_a_computed_operand_is_not_narrowed():
    m = _model(
        "m (float[10,4] d, int64[1] n) => (float[?,?] y) { e = Add(n, ONE)  y = Slice(d, ST, e, AX) }",
        {
            "ONE": np.array([1], np.int64),
            "ST": np.array([2], np.int64),
            "AX": np.array([0], np.int64),
        },
    )
    assert not any(
        p.rule == "narrow_const_index"
        for p in RO.analyze(m, {"d": (0.0, 1.0), "n": (3, 6)})
    )


_EMBED = "m (int64[2,5] ids) => (float[2,5,4] y) { y = Gather(W, ids) }"


def test_narrow_input_is_opt_in_and_changes_the_interface():
    w = np.arange(400, dtype=np.float32).reshape(100, 4)
    m = _model(_EMBED, {"W": w})
    box = {"ids": (0, 99)}
    props = RO.analyze(m, box)
    (rw,) = [p for p in props if p.rule == "narrow_input"]
    assert not rw.applies and rw.interface_change  # reported, not applied by default
    new, _ = RO.apply(m, box)
    assert new.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.INT64
    new, log = RO.apply(m, box, allow_interface_change=True)
    assert new.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.INT32
    assert any(r["rule"] == "narrow_input" and r["interface_change"] for r in log)
    _assert_equivalent(m, new, box)
    # the guard rejects an id the narrowed model was never proved for
    assert (
        RO.check_precondition(
            new, {"ids": np.array([[0, 99]])}, raise_on_violation=False
        )
        == []
    )
    with pytest.raises(RO.PreconditionViolation):
        RO.check_precondition(new, {"ids": np.array([[0, 100]])})


def test_narrow_input_refuses_other_consumers_and_wide_ranges():
    w = np.zeros((100, 4), np.float32)
    other = _model(
        "m (int64[2,5] ids) => (float[2,5,4] y, int64[2,5] z) { y = Gather(W, ids)  z = Add(ids, ids) }",
        {"W": w},
    )
    assert not [
        p for p in RO.analyze(other, {"ids": (0, 99)}) if p.rule == "narrow_input"
    ]
    m = _model(_EMBED, {"W": w})
    assert not [
        p for p in RO.analyze(m, {"ids": (0, 2**40)}) if p.rule == "narrow_input"
    ]
    assert not [p for p in RO.analyze(m, {}) if p.rule == "narrow_input"]


def test_narrow_input_through_cast_consumers():
    m = _model("m (int64[2,5] mask) => (float[2,5] y) { y = Cast<to=1>(mask) }")
    new, _ = RO.apply(m, {"mask": (0, 1)}, allow_interface_change=True)
    assert new.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.INT32
    _assert_equivalent(m, new, {"mask": (0, 1)})


# ---- decomposed softmax ---------------------------------------------------------------------

_SM = (
    "m (float[2,8] x) => (float[2,8] y) { "
    "mx = ReduceMax<axes=[1], keepdims=1>(x)  d = Sub(x, mx)  e = Exp(d)  "
    "s = ReduceSum<axes=[1], keepdims=1>(e)  y = Div(e, s) }"
)


def test_softmax_max_subtraction_dropped_only_in_a_safe_range():
    m = _model(_SM, opset=11)
    box = {"x": (-5.0, 5.0)}
    new, log = RO.apply(m, box)
    ops = _ops(new)
    assert (
        "Sub" not in ops and "ReduceMax" not in ops
    )  # the orphaned ReduceMax is pruned too
    assert any(r["rule"] == "softmax_no_max" for r in log)
    _assert_equivalent(m, new, box, rtol=1e-5, atol=1e-7)
    # hi + ln(8) > 88: exp could overflow; lo < -80: the denominator could underflow
    assert not _fired(RO.analyze(m, {"x": (-5.0, 90.0)}), "softmax_no_max")
    assert not _fired(RO.analyze(m, {"x": (-100.0, 5.0)}), "softmax_no_max")
    assert not _fired(RO.analyze(m, {}), "softmax_no_max")
    # the number of summed elements matters: hi = 87 is below 88 but 8 * exp(87) overflows float32
    # (hi + ln 8 = 89.08), whereas hi = 85.5 still gives a finite, normal sum (87.58 <= 88)
    assert not _fired(RO.analyze(m, {"x": (-5.0, 87.0)}), "softmax_no_max")
    edge = {"x": (-5.0, 85.5)}
    assert _fired(RO.analyze(m, edge), "softmax_no_max")
    new_edge, _ = RO.apply(m, edge)
    x = np.full((2, 8), 85.5, np.float32)
    assert np.isfinite(_session(new_edge).run(None, {"x": x})[0]).all()
    _assert_equivalent(m, new_edge, edge, rtol=1e-5, atol=1e-7)


def test_outside_the_softmax_box_dropping_the_max_really_overflows():
    m = _model(_SM, opset=11)
    new, _ = RO.apply(m, {"x": (-5.0, 5.0)})
    x = np.full((2, 8), 100.0, np.float32)
    assert np.isfinite(_session(m).run(None, {"x": x})[0]).all()
    assert not np.isfinite(_session(new).run(None, {"x": x})[0]).all()


def test_softmax_pattern_must_be_a_normalisation():
    unnormalised = _model(
        "m (float[2,8] x) => (float[2,8] y) { mx = ReduceMax<axes=[1], keepdims=1>(x)  d = Sub(x, mx)  y = Exp(d) }",
        opset=11,
    )
    assert not _fired(RO.analyze(unnormalised, {"x": (-5.0, 5.0)}), "softmax_no_max")
    mismatched = _model(
        _SM.replace("ReduceSum<axes=[1]", "ReduceSum<axes=[0]"), opset=11
    )
    assert not _fired(RO.analyze(mismatched, {"x": (-5.0, 5.0)}), "softmax_no_max")


# ---- report-only ----------------------------------------------------------------------------


def test_fp16_risk_is_reported_never_applied():
    m = _model(
        "m (float[4] x) => (float[4] y) { y = Mul(x, K) }", {"K": np.float32(1000.0)}
    )
    props = RO.analyze(m, {"x": (0.0, 100.0)})
    (rep,) = [p for p in props if p.rule == "fp16_risk"]
    assert not rep.applies and rep.proof["risk_count"] == 1
    new, log = RO.apply(m, {"x": (0.0, 100.0)})
    assert _ops(new) == ["Mul"]
    assert any(r["rule"] == "fp16_risk" and not r["applied"] for r in log)
    safe = RO.analyze(m, {"x": (0.0, 1.0)})
    assert [p for p in safe if p.rule == "fp16_risk"][0].proof["risk_count"] == 0


def test_fp16_flags_exp_operands_and_accumulators():
    m = _model(
        "m (float[1,4] x) => (float[1,4] y) { e = Exp(x)  y = MatMul(e, W) }",
        {"W": np.full((4, 4), 10000.0, np.float32)},
    )
    # x in [0, 1]: exp is fine, but the accumulator bound sum|w| * max|x| = 4e4 * e = 108730 > 65504
    (rep,) = [p for p in RO.analyze(m, {"x": (0.0, 1.0)}) if p.rule == "fp16_risk"]
    assert any("accumulator" in u["why"] for u in rep.proof["risk"])
    # x in [0, 20]: the Exp operand itself exceeds ln(65504) = 11.09
    (rep,) = [p for p in RO.analyze(m, {"x": (0.0, 20.0)}) if p.rule == "fp16_risk"]
    assert any("exp operand" in u["why"] for u in rep.proof["risk"])
    clean = _model("m (float[1,4] x) => (float[1,4] y) { y = Exp(x) }")
    (rep,) = [p for p in RO.analyze(clean, {"x": (0.0, 1.0)}) if p.rule == "fp16_risk"]
    assert rep.proof["risk_count"] == 0


def test_fp16_report_keeps_unbounded_apart_from_proven_risk():
    m = _model(
        "m (float[4] x) => (float[4] y) { s = Mul(x, K)  y = Add(s, K) }",
        {"K": np.float32(2.0)},
    )
    (rep,) = [
        p for p in RO.analyze(m, {}) if p.rule == "fp16_risk"
    ]  # no range: nothing is proven
    assert rep.proof["risk_count"] == 0 and rep.proof["unbounded_count"] >= 1
    assert rep.proof["safe_count"] == 0
    (rep,) = [p for p in RO.analyze(m, {"x": (0.0, 10.0)}) if p.rule == "fp16_risk"]
    assert rep.proof["unbounded_count"] == 0 and rep.proof["safe_count"] == 2


def test_int64_fits_report_counts_computed_tensors():
    m = _model(
        "m (float[2,3] x) => (float[6] y) { s = Shape(x)  n = ReduceProd(s)  r = Reshape(x, SH)  y = Identity(r) }",
        {"SH": np.array([6], np.int64)},
    )
    reps = [p for p in RO.analyze(m, {"x": (0.0, 1.0)}) if p.rule == "int64_fits_int32"]
    assert reps and not reps[0].applies


# ---- apply semantics -------------------------------------------------------------------------


def test_apply_refuses_to_run_without_ranges():
    m = _model(_RELU, _C)
    with pytest.raises(ValueError, match="needs at least one input range"):
        RO.apply(m)
    with pytest.raises(ValueError):
        RO.apply(m, {})


def test_apply_uses_the_models_own_range_annotations():
    from onnxsim import ranges

    m = _model(_RELU, _C)
    ranges.set_range(m, "x", 1.0, 2.0)
    new, log = RO.apply(m)
    assert "Relu" not in _ops(new) and log


def test_precondition_and_log_are_recorded():
    m = _model(_RELU, _C)
    new, log = RO.apply(m, {"x": (1.0, 2.0)})
    pre = RO.preconditions(new)
    assert set(pre) == {"x"} and float(pre["x"][0]) == 1.0 and float(pre["x"][1]) == 2.0
    recorded = json.loads(_meta(new, RO.LOG_KEY))
    assert recorded[0]["rule"] == "dead_relu" and "description" in recorded[0]
    assert RO.preconditions(m) == {}
    # the original is not modified
    assert "Relu" in _ops(m) and not any(
        p.key.startswith("onnxsim.") for p in m.metadata_props
    )


def test_second_apply_finds_nothing_more():
    m = _model(_RELU, _C)
    new, _ = RO.apply(m, {"x": (1.0, 2.0)})
    again, log = RO.apply(new, {"x": (1.0, 2.0)})
    assert not [r for r in log if r["applied"]]
    assert _ops(again) == _ops(new)


def test_rules_argument_restricts_and_rejects_unknown_names():
    m = _model(_RELU, _C)
    assert not _fired(
        RO.analyze(m, {"x": (1.0, 2.0)}, rules=["dead_clip"]), "dead_relu"
    )
    with pytest.raises(ValueError, match="unknown rule"):
        RO.analyze(m, {"x": (1.0, 2.0)}, rules=["nope"])
    with pytest.raises(ValueError):
        RO.analyze(m, {"x": (1.0, 2.0)}, engine="zonotope")


def test_check_precondition_on_a_model_with_no_box_never_violates():
    m = _model("m (float[4] x) => (float[4] y) { s = Sigmoid(x)  y = Relu(s) }")
    new, _ = RO.apply(m, {"x": (-1.0, 1.0)})
    assert RO.check_precondition(new, {"x": np.full(4, 1e9, np.float32)}) == []


# ---- randomized soundness --------------------------------------------------------------------


def _random_chain(rng):
    """A random op chain over a [4] input plus a random box, built from the ops the rules cover."""
    lo = float(rng.uniform(-3, 3))
    hi = lo + float(rng.uniform(0.1, 4))
    lines, inits, cur = [], {}, "x"
    for k in range(int(rng.integers(2, 6))):
        op = str(
            rng.choice(
                ["Add", "Mul", "Relu", "Abs", "Clip", "Max", "Min", "Neg", "Sigmoid"]
            )
        )
        out = f"t{k}"
        if op in ("Add", "Mul", "Max", "Min"):
            inits[f"c{k}"] = (rng.uniform(-3, 3, 4)).astype(np.float32)
            lines.append(f"{out} = {op}({cur}, c{k})")
        elif op == "Clip":
            a = float(rng.uniform(-4, 1))
            inits[f"lo{k}"], inits[f"hi{k}"] = (
                np.float32(a),
                np.float32(a + rng.uniform(0.2, 6)),
            )
            lines.append(f"{out} = Clip({cur}, lo{k}, hi{k})")
        else:
            lines.append(f"{out} = {op}({cur})")
        cur = out
    lines.append(f"y = Identity({cur})")
    body = "m (float[4] x) => (float[4] y) { " + "  ".join(lines) + " }"
    return _model(body, inits), {"x": (lo, hi)}


def test_random_chains_stay_equivalent_inside_their_box():
    fired = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        m, box = _random_chain(rng)
        new, log = RO.apply(m, box, margin=1e-6)
        fired += len([r for r in log if r["applied"]])
        _assert_equivalent(m, new, box, n=15, seed=seed, rtol=1e-5, atol=1e-5)
    assert fired >= 15  # the generator really exercises the rules


# ---- tighter bounds with CROWN ---------------------------------------------------------------

_CORR = "m (float[4] x) => (float[4] y) { r = Relu(x)  d = Sub(r, r)  a = Add(d, C)  y = Relu(a) }"


def test_crown_engine_proves_what_plain_intervals_cannot():
    m = _model(_CORR, {"C": np.full(4, 0.1, np.float32)})
    box = {"x": (-1.0, 1.0)}
    # intervals: Sub(r, r) in [-1, 1] loses the correlation, so the last Relu is undecided
    assert (
        _fired(RO.analyze(m, box), "dead_relu") == []
    )  # (the first Relu has x in [-1, 1]: not dead either)
    tight = _fired(RO.analyze(m, box, engine="crown"), "dead_relu")
    assert len(tight) == 1 and tight[0].proof["engine"] == "crown"
    new, _ = RO.apply(m, box, engine="crown")
    assert (
        _ops(new).count("Relu") == 1
    )  # the first Relu (x in [-1, 1]) stays, the last is gone
    _assert_equivalent(m, new, box)


# ---- independent confirmation ----------------------------------------------------------------


def test_certify_confirms_small_rewrites():
    pytest.importorskip("z3")
    from onnxsim import certify

    # certify has no encoding for Where/Greater/Abs/Sigmoid, so only rewrites over the ops it
    # supports (Add, Relu, Clip) can be confirmed this way
    clip = _model(
        "m (float[4] x) => (float[4] y) { a = Add(x, C)  y = Clip(a, LO, HI) }",
        {"C": np.full(4, 0.5, np.float32), **_CLIP_INIT},
    )
    cases = [
        (_model(_RELU, _C), {"x": (1.0, 2.0)}),  # dead Relu
        (clip, {"x": (0.0, 3.0)}),  # Clip removed entirely
        (clip, {"x": (0.0, 9.0)}),  # only the redundant min bound dropped
    ]
    for m, box in cases:
        new, log = RO.apply(m, box)
        assert [r for r in log if r["applied"]]
        rep = certify.certify(m, new, input_ranges=box)
        assert rep.ok, str(rep)
    # certify refutes the same rewrite outside the declared box: the precondition is real
    m = _model(_RELU, _C)
    new, _ = RO.apply(m, {"x": (1.0, 2.0)})
    assert not certify.certify(m, new, input_ranges={"x": (-2.0, 2.0)}).ok


# ---- CLI -------------------------------------------------------------------------------------


def test_cli_analyze_and_apply(tmp_path, capsys):
    src = tmp_path / "m.onnx"
    onnx.save(_model(_RELU, _C), str(src))
    assert RO.main([str(src), "--range", "x=1,2"]) == 0
    out = capsys.readouterr().out
    assert "dead_relu" in out and "needs box" in out
    dst = tmp_path / "o.onnx"
    assert RO.main([str(src), "--range", "x=1,2", "--apply", "-o", str(dst)]) == 0
    assert "wrote" in capsys.readouterr().out
    assert "Relu" not in _ops(onnx.load(str(dst)))
    with pytest.raises(SystemExit):
        RO.main([str(src), "--range", "x=1,2", "--apply"])  # needs -o
    with pytest.raises(ValueError):
        RO.main([str(src), "--apply", "-o", str(dst)])  # no ranges


def test_module_is_importable_without_touching_new_dtypes_at_import_time():
    # tests/test_onnx_compat.py scans every module; this is the cheap local guard
    src = open(RO.__file__).read()
    assert "TensorProto.UINT4" not in src and "TensorProto.INT4" not in src
    assert os.path.exists(RO.__file__)
