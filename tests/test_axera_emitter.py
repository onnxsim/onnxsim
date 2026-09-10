"""Learning an AX650N weight table's encoding as a bit map, checked offline.

`scripts/axera/emitter.py` does not implement the seven weight layouts
`README.md` reverse-engineered. It observes that all of them are bit
permutations and reads the permutation off a handful of same-shape builds.
The tests here build synthetic "compilers" -- a known permutation, a known
bit-plane split, a table with scaffolding mixed in -- so the method can be
checked without Docker, a card, or a committed multi-megabyte fixture.
"""

import os
import sys

import numpy as np

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import emitter  # noqa: E402


def _fake_compiler(n_codes, n_table_bytes, seed=0, const_bits=None):
    """A permutation that scatters every code bit into a fixed table position.

    Returns `encode(codes) -> table`, standing in for `pulsar2 build` at one
    shape: the placement is fixed, only the codes move.
    """
    rng = np.random.default_rng(seed)
    n_code_bits = n_codes * 8
    n_table_bits = n_table_bytes * 8
    slots = rng.permutation(n_table_bits)[:n_code_bits]
    scaffold = rng.integers(0, 2, n_table_bits).astype(np.uint8)
    if const_bits is not None:
        scaffold[:] = const_bits

    def encode(codes):
        bits = np.unpackbits(np.asarray(codes, np.uint8).ravel(), bitorder="little")
        out = scaffold.copy()
        out[slots] = bits
        return np.packbits(out, bitorder="little")

    return encode, slots


def _random_weights(shape, seed):
    return (np.random.default_rng(seed).standard_normal(shape) * 0.1).astype(np.float32)


def test_codes_are_the_quantiser_plus_the_storage_offset():
    """The table stores unsigned codes: the per-channel symmetric quantiser
    with 128 added, which is what `README.md` read out of real tables."""
    import precision

    w = _random_weights((4, 3, 3), 1)
    codes = emitter.codes_of(w)
    assert codes.dtype == np.uint8
    step = np.abs(w).max(axis=(1, 2), keepdims=True) / 127.5
    assert np.array_equal(
        codes.astype(np.int16) - 128,
        np.clip(np.rint(w / step), -128, 127).astype(np.int16),
    )
    # and it agrees with the module that does the same job without the offset
    assert np.allclose(
        (codes.astype(np.float32) - 128) * step,
        precision.quantise_dequantise(w, axis=0),
    )


def test_a_permutation_is_learned_exactly_from_enough_builds():
    """Thirty-two samples name one code bit out of 3456 with room to spare."""
    shape = (12, 12, 3)
    n_codes = int(np.prod(shape))
    encode, slots = _fake_compiler(n_codes, 700, seed=5)
    ws = [_random_weights(shape, s) for s in range(32)]
    codes = [emitter.codes_of(w) for w in ws]
    tables = [encode(c) for c in codes]

    origin, const, ambiguous = emitter.learn(codes, tables)
    assert len(ambiguous) == 0
    assert emitter.collisions(codes, origin) == 0
    mapped = np.flatnonzero(origin != emitter.CONST)
    # every code bit that actually varied is accounted for, at the right slot
    for code_bit, table_bit in enumerate(slots):
        if origin[table_bit] != emitter.CONST:
            assert origin[table_bit] == code_bit
    assert len(mapped) >= 0.99 * len(slots)


def test_the_learned_map_writes_an_unseen_weight_set_byte_exactly():
    """The point of the exercise: emit a table the compiler was never asked
    for, and have it match what the compiler would have produced."""
    shape = (12, 12, 3)
    encode, _ = _fake_compiler(int(np.prod(shape)), 700, seed=6)
    ws = [_random_weights(shape, s) for s in range(32)]
    codes = [emitter.codes_of(w) for w in ws]
    origin, _, _ = emitter.learn(codes, [encode(c) for c in codes])

    held_out = emitter.codes_of(_random_weights(shape, 999))
    got = emitter.emit_table(encode(codes[0]), origin, held_out)
    assert np.array_equal(got, encode(held_out))


def test_bit_planes_are_no_harder_than_a_permutation():
    """The real format interleaves bits from several codes into one byte. That
    is still a permutation, so nothing about the method changes."""
    shape = (8, 8, 3)
    n_codes = int(np.prod(shape))

    def encode(codes):
        bits = np.unpackbits(np.asarray(codes, np.uint8).ravel(), bitorder="little")
        planes = bits.reshape(n_codes, 8).T.ravel()  # plane-major
        return np.packbits(planes, bitorder="little")

    ws = [_random_weights(shape, s) for s in range(32)]
    codes = [emitter.codes_of(w) for w in ws]
    origin, _, ambiguous = emitter.learn(codes, [encode(c) for c in codes])
    assert len(ambiguous) == 0
    held_out = emitter.codes_of(_random_weights(shape, 4242))
    assert np.array_equal(
        emitter.emit_table(encode(codes[0]), origin, held_out), encode(held_out)
    )


def test_too_few_builds_shows_up_as_collisions():
    """The method's own failure mode is detectable, which is what makes it
    safe to use: with four samples the signatures are four bits wide and
    thousands of code bits share each one."""
    shape = (12, 12, 3)
    encode, _ = _fake_compiler(int(np.prod(shape)), 700, seed=7)
    ws = [_random_weights(shape, s) for s in range(32)]
    codes = [emitter.codes_of(w) for w in ws]
    tables = [encode(c) for c in codes]
    few = emitter.learn(codes[:4], tables[:4])[0]
    assert emitter.collisions(codes[:4], few) > 100
    many = emitter.learn(codes, tables)[0]
    assert emitter.collisions(codes, many) == 0


def test_scaffolding_that_is_not_a_weight_bit_is_reported_not_guessed():
    """A table also holds things computed from the weights rather than copied
    from them -- per-channel scales, biases. Those must come back as
    unexplained, so an emitter knows it has to model them."""
    shape = (8, 8, 3)
    n_codes = int(np.prod(shape))
    encode, _ = _fake_compiler(n_codes, 400, seed=8)

    def encode_with_field(codes):
        table = encode(codes).copy()
        # a byte that is a function of the weights but not a copy of any bit
        table[-1] = np.uint8(int(np.asarray(codes, np.int64).sum()) & 0xFF)
        return table

    ws = [_random_weights(shape, s) for s in range(32)]
    codes = [emitter.codes_of(w) for w in ws]
    origin, const, ambiguous = emitter.learn(
        codes, [encode_with_field(c) for c in codes]
    )
    assert len(ambiguous) > 0
    assert all(bit // 8 == 399 for bit in ambiguous)


def test_constant_scaffolding_is_kept_from_the_reference():
    shape = (8, 8, 3)
    encode, _ = _fake_compiler(int(np.prod(shape)), 400, seed=9, const_bits=1)
    ws = [_random_weights(shape, s) for s in range(32)]
    codes = [emitter.codes_of(w) for w in ws]
    tables = [encode(c) for c in codes]
    origin, const, _ = emitter.learn(codes, tables)
    got = emitter.emit_table(tables[0], origin, codes[5])
    assert np.array_equal(got, tables[5])


def test_the_map_round_trips_through_a_file(tmp_path):
    shape = (8, 8, 3)
    encode, _ = _fake_compiler(int(np.prod(shape)), 400, seed=10)
    ws = [_random_weights(shape, s) for s in range(32)]
    codes = [emitter.codes_of(w) for w in ws]
    origin, const, ambiguous = emitter.learn(codes, [encode(c) for c in codes])
    path = str(tmp_path / "map.npz")
    emitter.save_map(path, origin, const, ambiguous, shape)
    o2, c2, a2, s2 = emitter.load_map(path)
    assert np.array_equal(o2, origin) and np.array_equal(c2, const)
    assert np.array_equal(a2, ambiguous) and s2 == shape


def _mcode_samples(n, scales, zeros, free_values):
    """Streams shaped like a real one: constant, except an output scale written
    four times, a one-byte zero point, and a few bytes that move for reasons of
    their own."""
    base = np.arange(64, dtype=np.uint8) * 3
    out = []
    for i in range(n):
        m = base.copy()
        m[10] = np.uint8(zeros[i])
        for off in (20, 28, 36, 44):
            m[off : off + 4] = np.frombuffer(np.float32(scales[i]).tobytes(), np.uint8)
        for j, off in enumerate((5, 7)):
            m[off] = free_values[i][j]
        out.append(m)
    return out


def test_the_mcode_fields_that_follow_the_output_quantisation_are_found():
    rng = np.random.default_rng(3)
    n = 20
    scales = (0.02 + rng.random(n) * 0.01).astype(np.float32)
    zeros = rng.integers(110, 150, n)
    free = rng.integers(0, 64, (n, 2)).astype(np.uint8)
    fields = emitter.learn_mcode(_mcode_samples(n, scales, zeros, free), scales, zeros)
    assert fields["scale_offsets"] == [20, 28, 36, 44]
    assert fields["zero_offsets"] == [10]
    assert fields["free_offsets"] == [5, 7]
    assert fields["outliers"] == []


def test_a_stream_that_cannot_hold_a_literal_is_reported_not_silently_wrong():
    """Two of 48 real builds encode a field with an extra byte instead of
    inline, shifting everything after it. No rule tried predicted which, so the
    values that actually failed are recorded and refused."""
    rng = np.random.default_rng(4)
    n = 12
    scales = (0.02 + rng.random(n) * 0.01).astype(np.float32)
    zeros = rng.integers(110, 150, n)
    zeros[3] = 128
    free = np.zeros((n, 2), np.uint8)
    samples = _mcode_samples(n, scales, zeros, free)
    samples[3] = np.roll(samples[3], 1)  # the shifted stream

    fields = emitter.learn_mcode(samples, scales, zeros)
    assert fields["outliers"] == [3]
    assert fields["unpatchable"]["zero"] == [128]
    emitter.emit_mcode(samples[0], fields, scales[0], zeros[0])  # fine
    try:
        emitter.emit_mcode(samples[0], fields, scales[0], 128)
    except ValueError as exc:
        assert "128" in str(exc)
    else:
        raise AssertionError("writing the unpatchable zero point should refuse")


def test_emit_mcode_changes_only_the_quantisation_fields():
    rng = np.random.default_rng(5)
    n = 16
    scales = (0.02 + rng.random(n) * 0.01).astype(np.float32)
    zeros = rng.integers(110, 150, n)
    free = np.zeros((n, 2), np.uint8)
    samples = _mcode_samples(n, scales, zeros, free)
    fields = emitter.learn_mcode(samples, scales, zeros)
    for i in range(n):
        got = emitter.emit_mcode(samples[0], fields, scales[i], zeros[i])
        assert np.array_equal(got, samples[i])


def test_the_requant_block_is_the_quantised_convolutions_constant_term():
    """`bias[c] = zy - zx * sum(q_c) * M_c` and `M_c = x_scale*w_scale/y_scale`
    -- read off 48 real builds, where the formula reproduces all 48x32 stored
    float32s to 6.1e-05."""
    w = _random_weights((5, 4, 3), 2)
    codes = emitter.codes_of(w)
    x_scale, x_zero, y_scale, y_zero = 0.0279, 113.0, 0.0255, 131.0
    block = emitter.requant_block(
        codes, x_scale, x_zero, y_scale, y_zero, emitter.weight_scales(w)
    )
    got = block.copy().view(np.float32)
    assert len(got) == 10
    m = x_scale * emitter.weight_scales(w) / y_scale
    q = codes.reshape(5, -1).astype(np.int64) - 128
    assert np.allclose(got[5:], m, rtol=1e-6)
    assert np.allclose(got[:5], y_zero - x_zero * q.sum(axis=1) * m, rtol=1e-5)


def test_weight_scales_match_the_quantiser():
    w = _random_weights((6, 4, 3), 3)
    assert np.allclose(emitter.weight_scales(w), np.abs(w).max(axis=(1, 2)) / 127.5)


def test_the_llm_quantiser_is_not_the_convolution_one():
    """`llm_build` divides by 128, takes the scale from the *signed* peak and
    negates it, and rounds ties toward +inf -- so a row's extreme weight lands
    on code 0 and zero lands on 128."""
    w = np.array([[-1.0, 0.5, 0.0, 0.25], [2.0, -1.0, 0.0, 0.5]], dtype=np.float32)
    codes = emitter.llm_codes_of(w)
    assert codes.dtype == np.uint8
    # the extreme of each row is code 0, and an exact zero is 128
    assert codes[0, 0] == 0 and codes[1, 0] == 0
    assert codes[0, 2] == 128 and codes[1, 2] == 128
    # and it differs from the convolution quantiser, which is symmetric
    conv = emitter.codes_of(w.reshape(2, 4, 1))
    assert not np.array_equal(conv.reshape(2, 4), codes)


def test_the_llm_quantiser_puts_a_negative_peak_at_code_zero_too():
    """The sign of the extreme is kept, not its magnitude: a row whose largest
    weight is positive still lands that weight on 0."""
    codes = emitter.llm_codes_of(np.array([[3.0, -1.0, 0.0]], dtype=np.float32))
    assert codes[0, 0] == 0
    assert codes[0, 2] == 128
    assert codes[0, 1] > 128
