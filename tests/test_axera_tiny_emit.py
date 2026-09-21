"""Offline tests for scripts/axera/tiny_emit.py.

No Docker, no card: the tinygrad pattern match runs against a real
tinygrad graph (skipped if tinygrad is not installed), and the scale
field map is pinned against the committed neg_1x8 fixture. Device
execution of emitted streams is documented in the module docstring and
the README, not asserted here.
"""

import gzip
import os
import struct
import sys

import numpy as np
import pytest

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import tiny_emit  # noqa: E402
from mcode import (  # noqa: E402
    FULL_RULE,
    check,
    decode,
    encode,
    stream_bounds,
)

_FIXTURES = os.path.join(_AXERA_DIR, "fixtures")


def _blob(name):
    with gzip.open(os.path.join(_FIXTURES, name + ".mcode.gz"), "rb") as f:
        return f.read()


def test_trace_neg_matches_mul_by_minus_one():
    tg = pytest.importorskip("tinygrad")
    t = tg.Tensor.empty(1, 8)
    got = tiny_emit.trace_neg(-t)
    assert got["shape"] == [1, 8]


def test_trace_neg_rejects_other_graphs():
    tg = pytest.importorskip("tinygrad")
    t = tg.Tensor.empty(1, 8)
    with pytest.raises(ValueError):
        tiny_emit.trace_neg(t + t)
    with pytest.raises(ValueError):
        tiny_emit.trace_neg(t * 2.0)


def test_trace_add_matches_two_tensor_add():
    tg = pytest.importorskip("tinygrad")
    a, b = tg.Tensor.empty(1, 8), tg.Tensor.empty(1, 8)
    got = tiny_emit.trace_add(a + b)
    assert got["shape"] == [1, 8]


def test_trace_add_rejects_other_graphs():
    tg = pytest.importorskip("tinygrad")
    a, b = tg.Tensor.empty(1, 8), tg.Tensor.empty(1, 8)
    with pytest.raises(ValueError):
        tiny_emit.trace_add(a * b)  # wrong op
    with pytest.raises(ValueError):
        tiny_emit.trace_add(a + 2.0)  # scalar add, not two-tensor


def test_trace_mul_matches_two_tensor_multiply():
    tg = pytest.importorskip("tinygrad")
    a, b = tg.Tensor.empty(1, 8), tg.Tensor.empty(1, 8)
    got = tiny_emit.trace_mul(a * b)
    assert got["shape"] == [1, 8]


def test_trace_mul_rejects_other_graphs():
    tg = pytest.importorskip("tinygrad")
    a, b = tg.Tensor.empty(1, 8), tg.Tensor.empty(1, 8)
    with pytest.raises(ValueError):
        tiny_emit.trace_mul(a + b)  # wrong op
    with pytest.raises(ValueError):
        tiny_emit.trace_mul(-a)  # scalar multiply (Neg's own shape)
    with pytest.raises(ValueError):
        tiny_emit.trace_mul(a * 2.0)  # scalar multiply, not two-tensor


def test_trace_relu_matches_compare_and_select():
    tg = pytest.importorskip("tinygrad")
    t = tg.Tensor.empty(1, 8)
    got = tiny_emit.trace_relu(t.relu())
    assert got["shape"] == [1, 8]


def test_trace_relu_rejects_other_graphs():
    tg = pytest.importorskip("tinygrad")
    a, b = tg.Tensor.empty(1, 8), tg.Tensor.empty(1, 8)
    with pytest.raises(ValueError):
        tiny_emit.trace_relu(a.sigmoid())  # wrong top-level op
    with pytest.raises(ValueError):
        # Same WHERE/CMPLT-against-0.0 shape, but the true branch is a
        # *different* tensor than the one compared against 0 -- the
        # identity check, not just shape/dtype matching, must catch this.
        tiny_emit.trace_relu((0 < a).where(b, 0))


def test_trace_sigmoid_matches_reciprocal_chain():
    tg = pytest.importorskip("tinygrad")
    t = tg.Tensor.empty(1, 8)
    got = tiny_emit.trace_sigmoid(t.sigmoid())
    assert got["shape"] == [1, 8]


def test_trace_sigmoid_rejects_other_graphs():
    tg = pytest.importorskip("tinygrad")
    t = tg.Tensor.empty(1, 8)
    with pytest.raises(ValueError):
        tiny_emit.trace_sigmoid(t.relu())  # wrong top-level op
    with pytest.raises(ValueError):
        # 1/(1+2^(-2x)) -- same shape, wrong constant (not -log2(e)).
        tiny_emit.trace_sigmoid((1.0 + (t * -2.0).exp2()).reciprocal())


def test_minmax_scale_matches_pulsar2_to_1e10():
    rng = np.random.default_rng(0)
    samples = [(rng.uniform(-2.5, 2.5, (1, 8))).astype(np.float32) for _ in range(8)]
    got = tiny_emit.minmax_scale(samples)
    # Pulsar2's recorded output scale for this exact calibration.
    assert got == pytest.approx(0.0194994397, rel=1e-6)


def test_find_scale_words_locates_the_quadruple():
    blob = _blob("neg_1x8")
    scale = float(np.float32(0.0076893349))
    assert tiny_emit.find_scale_words(blob, scale) == [1288, 1295, 1302, 1309]


def test_emit_neg_round_trips_and_stays_clean():
    blob = _blob("neg_1x8")
    out = tiny_emit.emit_neg(
        blob, float(np.float32(0.0076893349)), float(np.float32(0.0132596185))
    )
    lo, hi = stream_bounds(out)
    assert encode(decode(out, start=lo, end=hi, **FULL_RULE)) == out[lo:hi]
    assert check(out) == []


# Mul input-side scales. Source of truth for the values is
# tests/test_axera_mcode_reciprocal.py (fixture builds' quant JSON);
# w2 (2x/2x ranges, same short site-B form as base) was scratched in
# ~/npu-scratch/t9-mul/mul_1x8_w2 and is committed as a fixture here.
_BASE = (0.007799775805324316, 0.0076893349178135395, 0.007151617668569088)
_W2 = (0.015599551610648632, 0.015378669835627079, 0.028606470674276352)
_X10 = (0.07799775898456573, 0.0007689335034228861, 0.007151617202907801)
_X01 = (0.0007725197938270867, 0.07746022194623947, 0.005844127852469683)


def _assert_slot_run(blob, value, stride, width=4):
    """The float32 word for ``value`` forms a clean x4 stride run."""
    pat = struct.pack("<f", value)[:width]
    assert len(tiny_emit._strided_run(blob, pat, stride)) == 4


def _assert_family_matches(patched, target, value, stride, width=4):
    _assert_slot_run(patched, value, stride, width)
    _assert_slot_run(target, value, stride, width)


def test_patch_mul_short_to_short_matches_w2_slots():
    """base -> w2 (both short site-B form): every input-side family lands
    on w2's own slot bytes, and the patched stream stays structurally
    clean. Full-stream equality is explicitly out of scope: the output
    quads (z 4x, emitter.py's domain), the manifest string table (x/y
    name order swaps build to build), the magnitude-adaptive S-unit
    programs and one input-driven single (1836: c9 -> cd) do not
    transplant -- see the transplant analysis in the PR."""
    patched = tiny_emit.patch_mul_scales(_blob("mul_1x8"), _BASE, _W2)
    target = _blob("mul_1x8_w2")
    nx, ny, nz = _W2
    _assert_family_matches(patched, target, 1.0 / nx, 8)
    _assert_family_matches(patched, target, nz / (nx * ny), 8)
    _assert_family_matches(patched, target, 1.0 / ny, 6, width=3)
    assert check(patched) == []


def test_patch_mul_full_to_full_matches_x01_slots():
    """x10 -> x01 (both full site-B form): same contract as above."""
    patched = tiny_emit.patch_mul_scales(_blob("mul_1x8_recip_x10"), _X10, _X01)
    target = _blob("mul_1x8_recip_x01")
    nx, ny, nz = _X01
    _assert_family_matches(patched, target, 1.0 / nx, 8)
    _assert_family_matches(patched, target, nz / (nx * ny), 8)
    _assert_family_matches(patched, target, 1.0 / ny, 7)
    assert check(patched) == []


def test_patch_mul_preserves_reference_site_b_form():
    """base (short B) -> x01 scales (full B in its own build): sites A and
    C still land on x01's slot bytes, while site B keeps the reference's
    short form carrying the new value -- patching rewrites values, it
    does not recompile programs."""
    patched = tiny_emit.patch_mul_scales(_blob("mul_1x8"), _BASE, _X01)
    target = _blob("mul_1x8_recip_x01")
    nx, ny, nz = _X01
    _assert_family_matches(patched, target, 1.0 / nx, 8)
    _assert_family_matches(patched, target, nz / (nx * ny), 8)
    _assert_slot_run(patched, 1.0 / ny, 6, width=3)
    assert check(patched) == []


# Output scale quad (z's own scale, `03 <f32> 81 82` x4 stride 7): source
# of truth is TestOutputScaleQuads in tests/test_axera_mcode_reciprocal.py.


@pytest.mark.parametrize(
    "src,dst,dst_z",
    [
        ("mul_1x8", "mul_1x8_w2", _W2[2]),
        ("mul_1x8_recip_x10", "mul_1x8_recip_x01", _X01[2]),
    ],
)
def test_patch_mul_output_quad_matches_target(src, dst, dst_z):
    src_z = {"mul_1x8": _BASE[2], "mul_1x8_recip_x10": _X10[2]}[src]
    patched = tiny_emit.patch_mul_output_quad(_blob(src), src_z, dst_z)
    target = _blob(dst)
    _assert_family_matches(patched, target, dst_z, 7)
    assert check(patched) == []


def test_patch_mul_output_quad_round_trips():
    blob = _blob("mul_1x8")
    out_and_back = tiny_emit.patch_mul_output_quad(
        tiny_emit.patch_mul_output_quad(blob, _BASE[2], _W2[2]), _W2[2], _BASE[2]
    )
    assert out_and_back == blob


_RESHAPE_GATHER_SCALE = 0.0076232957653701305
_RESHAPE_GATHER_TARGET_SCALE = 0.02


def test_patch_reshape_gather_scales_changes_only_verified_scale_fields():
    blob = _blob("reshape_gather_fresh")
    patched = tiny_emit.patch_reshape_gather_scales(
        blob,
        _RESHAPE_GATHER_SCALE,
        _RESHAPE_GATHER_TARGET_SCALE,
        _RESHAPE_GATHER_SCALE,
        _RESHAPE_GATHER_TARGET_SCALE,
    )

    reciprocal = struct.pack("<f", 1.0 / _RESHAPE_GATHER_TARGET_SCALE)
    output_scale = struct.pack("<f", _RESHAPE_GATHER_TARGET_SCALE)
    assert tiny_emit._strided_run(patched, reciprocal, 8)
    assert tiny_emit._strided_run(patched, output_scale, 7)

    changed = {
        i for i, (before, after) in enumerate(zip(blob, patched)) if before != after
    }
    expected = {
        i
        for offset, stride in ((1067, 8), (1377, 7))
        for start in (offset + stride * index for index in range(4))
        for i in range(start, start + 4)
    }
    assert changed == expected
    assert check(patched) == []


def test_patch_reshape_gather_scales_round_trips():
    blob = _blob("reshape_gather_fresh")
    patched = tiny_emit.patch_reshape_gather_scales(
        blob,
        _RESHAPE_GATHER_SCALE,
        _RESHAPE_GATHER_TARGET_SCALE,
        _RESHAPE_GATHER_SCALE,
        _RESHAPE_GATHER_TARGET_SCALE,
    )
    restored = tiny_emit.patch_reshape_gather_scales(
        patched,
        _RESHAPE_GATHER_TARGET_SCALE,
        _RESHAPE_GATHER_SCALE,
        _RESHAPE_GATHER_TARGET_SCALE,
        _RESHAPE_GATHER_SCALE,
    )
    assert restored == blob


@pytest.mark.parametrize(
    "old_x,old_z,new_x,new_z",
    [
        (_RESHAPE_GATHER_SCALE, 0.01, 0.02, 0.02),
        (_RESHAPE_GATHER_SCALE, _RESHAPE_GATHER_SCALE, 0.02, 0.03),
        (0.0, 0.0, 0.02, 0.02),
        (_RESHAPE_GATHER_SCALE, _RESHAPE_GATHER_SCALE, float("nan"), float("nan")),
    ],
)
def test_patch_reshape_gather_scales_rejects_invalid_scale_relations(
    old_x, old_z, new_x, new_z
):
    with pytest.raises(ValueError):
        tiny_emit.patch_reshape_gather_scales(
            _blob("reshape_gather_fresh"), old_x, new_x, old_z, new_z
        )


def test_patch_mul_output_quad_rejects_bad_frame():
    # x_scale's own reciprocal-family words never carry the 03../8182
    # frame, so patching "the output scale" by a value that only happens
    # to collide with an unframed word must fail loudly, not silently
    # patch the wrong bytes.
    with pytest.raises(ValueError):
        tiny_emit.patch_mul_output_quad(_blob("mul_1x8"), 1.0 / _BASE[0], 1.0)


# x's zero point, literal-byte form only (`02 10 1b <zp_x> 83 36`): source
# of truth is TestZpXLiteralByteWhenPresent, which also documents that
# most builds do NOT use this form -- not predictable from zp_x's value.


def test_patch_mul_zp_x_matches_target():
    patched = tiny_emit.patch_mul_zp_x(_blob("mul_1x8_zp33sweep"), 33, 35)
    target = _blob("mul_1x8_zp35sweep")
    unit = bytes.fromhex("02101b") + bytes([35]) + bytes.fromhex("8336")
    assert unit in patched
    assert unit in target
    assert check(patched) == []


def test_patch_mul_zp_x_round_trips():
    blob = _blob("mul_1x8_zp33sweep")
    out_and_back = tiny_emit.patch_mul_zp_x(
        tiny_emit.patch_mul_zp_x(blob, 33, 35), 35, 33
    )
    assert out_and_back == blob


def test_patch_mul_zp_x_raises_when_form_absent():
    # mul_1x8 (base) has zp_x=128 but uses one of the opaque forms, not
    # the literal one -- the function must say so, not silently no-op.
    with pytest.raises(ValueError):
        tiny_emit.patch_mul_zp_x(_blob("mul_1x8"), 128, 100)


def test_patch_mul_zp_x_rejects_out_of_range():
    with pytest.raises(ValueError):
        tiny_emit.patch_mul_zp_x(_blob("mul_1x8_zp33sweep"), 33, 256)
