"""Host-only checks of the s8 weight-table helpers in ``llm_build_dtype_analysis``.

``s8_add_to_codes`` edits single weight codes of an encoded s8 block. The edit must
change exactly those codes (and the row sums of their column blocks), and nothing else.
The Qwen3 scale rule is checked against its definition. No Pulsar2, no Docker, no
checkpoint: the blocks are encoded with ``encode_block`` from random weights.
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import llm_build_dtype_analysis as lda  # noqa: E402


def _block(cin, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.randn(lda.ROW_BLOCK, cin) * 0.02).astype(np.float32)


@pytest.mark.parametrize("cin", [256, 1024, 3072])
def test_encode_decode_round_trip(cin):
    w = _block(cin)
    data = lda.encode_block(w, "s8")
    got = lda.decode_block(data, "s8", cin)["q"]
    want, _ = lda.quantize_int(w, 8)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("cin", [256, 1024, 3072])
def test_patch_changes_only_the_edited_codes(cin):
    w = _block(cin, seed=1)
    data = lda.encode_block(w, "s8")
    before = lda.decode_block(data, "s8", cin)
    edits = [(0, 0, 8), (5, cin - 1, -12), (31, cin // 2, 40)]
    new = lda.s8_add_to_codes(data, 0, cin, edits)
    after = lda.decode_block(new, "s8", cin)
    expect = before["q"].copy()
    for row, col, delta in edits:
        expect[row, col] += delta
    np.testing.assert_array_equal(after["q"], expect)
    # the file bytes that changed are the nibble bytes of the edited codes and their row sums
    changed = np.flatnonzero(
        np.frombuffer(data, np.uint8) != np.frombuffer(new, np.uint8)
    )
    allowed = set()
    for row, col, _ in edits:
        byte, _sh, tail = lda.s8_code_site(row, col, cin)
        allowed |= {byte, byte + lda.CHUNK}
        allowed |= set(
            range(
                tail + (row % lda.ROW_BLOCK) * 8, tail + (row % lda.ROW_BLOCK) * 8 + 4
            )
        )
    assert set(changed.tolist()) <= allowed


def test_row_sum_moves_by_half_the_delta():
    cin = 1024
    data = lda.encode_block(_block(cin, seed=2), "s8")
    before = lda.decode_block(data, "s8", cin)["rowsum"]
    new = lda.s8_add_to_codes(data, 0, cin, [(3, 7, 10)])
    after = lda.decode_block(new, "s8", cin)["rowsum"]
    diff = np.asarray(after) - np.asarray(before)
    assert np.isclose(
        diff[3].sum(), -5.0
    )  # -(delta / 2), in the column block that holds col 7
    assert np.allclose(np.delete(diff, 3, axis=0), 0)


def test_out_of_range_edit_is_refused():
    cin = 256
    data = lda.encode_block(_block(cin, seed=3), "s8")
    q = lda.decode_block(data, "s8", cin)["q"]
    with pytest.raises(ValueError):
        lda.s8_add_to_codes(data, 0, cin, [(0, 0, 127 - int(q[0, 0]) + 1)])


def test_qwen3_rule_is_max_over_127():
    w = _block(64, seed=4)
    q, scale = lda.quantize_int_qwen3(w, 8)
    peak = w[np.arange(len(w)), np.argmax(np.abs(w), axis=1)]
    np.testing.assert_allclose(scale, np.abs(peak) / 127, rtol=1e-6)
    assert scale.min() > 0
    assert q.min() >= -128 and q.max() <= 127
    np.testing.assert_array_equal(
        q, np.clip(np.floor(w / scale[:, None] + 0.5), -128, 127)
    )


def test_rule_selection_by_model_type():
    assert lda.s8_scale_rule("qwen3") is lda.quantize_int_qwen3
    assert lda.s8_scale_rule("llama") is lda.quantize_int
