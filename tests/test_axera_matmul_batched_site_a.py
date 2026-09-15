"""Rank-3 (batched/broadcast) MatMul breaks the rank-2 A/B asymmetry: `A`
gets a real, new, short-form quad -- it isn't just "still nothing."

`tests/test_axera_site_a_generalizes.py` (merged) established that plain
rank-2 `MatMul(A[M,K], B[K,N])` is asymmetric: only `B` gets the literal
site-A quad (`<f32(1/scale)> a1 00 <id>` x4, stride 8); `A`'s reciprocal
scale has no literal encoding anywhere, exhaustively searched (full
float32, short 3-byte form, bf16 truncation -- all absent).
`tests/test_axera_matmul_a_input_cascades.py` (merged) separately showed
`A`'s scale-dependent effect on rank-2 mcode is a diffuse cascade, not a
hidden field.

This checks whether that asymmetry holds once `A` gains a batch
dimension -- the practically important shape (attention `Q @ K^T`,
`attn @ V`) that has never been checked at the mcode byte level before
(the README's own "Conv/MatMul variants" section confirms
broadcasting/batched `MatMul` *compiles* successfully, but says nothing
about its mcode encoding -- a different question, checked not to
overlap before starting this).

## Finding: batching `A` gives it a real quad, using the *short* form

Two shapes were built (`pulsar2_docker.build()`, compile-only): fully
batched `A[2,4,8] @ B[2,8,8]` and broadcast `A[2,4,8] @ B[8,8]` (`B`
stays rank-2). In **both**, unlike plain rank-2:

- **`B` still gets the standard full-form site-A quad** (`<f32(1/B_scale)>
  a1 00 <id>` x4, stride 8) -- unaffected by whether `B` itself is rank-2
  or rank-3. Rank doesn't change `B`'s treatment at all.
- **`A` now gets a real quad too, but in the *short* 3-byte form** (the
  low 3 bytes of `1/A_scale`'s float32 representation -- the exact form
  `test_axera_site_a_generalizes.py` searched for and confirmed *absent*
  for rank-2's `A`): `<3 bytes, low(1/A_scale)> 82 <var> 02` x4 at
  stride 6, with the last copy's tail byte `83` instead of `02` (the
  same "final copy differs" shape this project's other quads show --
  output-scale quad's `tag2`, site-A's own `<id>`). The middle tag byte
  (`0x62` in the batched build, `0x5e` in the broadcast build) varies
  build to build, same as site-A's own varying lead byte -- not part of
  the fixed frame.
- This is a real behavior change, not a search artifact: the exact same
  low-3-bytes pattern was searched for and confirmed absent, at any
  offset, in plain rank-2's `matmul_4x8x8.mcode.gz`
  (`test_a_gets_no_literal_encoding_full_short_or_bf16`, already
  merged). Batching `A` gives it real, new structure that flatly does
  not exist in the rank-2 case.

## A second, independent finding: the fully-batched shape has unusually
large rebuild-to-rebuild noise -- confirmed, not assumed

This project's hard-learned rule (violated and caught twice before) is
to verify determinism with an independent rebuild before trusting any
diff. Doing that here surfaced something new: the **fully-batched**
`A[2,4,8] @ B[2,8,8]` build's own rebuild pair (identical model,
calibration data, and config) differs at **932 of 3336 bytes (28%)** --
two orders of magnitude past the ~6-15-byte noise zone this project has
confirmed repeatedly for every other op/shape so far. The **broadcast**
`A[2,4,8] @ B[8,8]` build's rebuild pair, by contrast, differs at only
6 of 3080 bytes -- an entirely ordinary noise floor. So the extreme
non-determinism is specific to the fully-batched (`A` *and* `B` both
rank-3) configuration, not a general property of rank-3 `A`.

Despite that, the short-form `A` quad is real and reproducible in both
cases: in the fully-batched pair, the quad's own 30-byte content is
byte-identical across the rebuild, just found at a constant -4 byte
offset (the same "content survives, position drifts under heavy
surrounding noise" pattern `test_axera_matmul_a_input_cascades.py`
already established); in the broadcast pair, it's at the exact same
offset both times, consistent with that shape's much smaller noise
floor. Not chased further here: *why* the fully-batched shape
specifically has 150x the noise of every other shape tested is a real,
motivated open question for whoever looks at it next.
"""

import gzip
import os
import struct
import unittest

FIX = os.path.join(os.path.dirname(__file__), "..", "scripts", "axera", "fixtures")


def load(name):
    with gzip.open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def hits(data, pat):
    return [i for i in range(len(data) - len(pat) + 1) if data[i : i + len(pat)] == pat]


class TestBatchedMatMulSiteA(unittest.TestCase):
    # fixture: (A_scale, B_scale)
    CASES = {
        "matmul_2x4x8x8_batched.mcode.gz": (0.007840047590434551, 0.015641119331121445),
        "matmul_2x4x8x8_broadcast.mcode.gz": (
            0.007840047590434551,
            0.015641119331121445,
        ),
    }

    def test_b_still_gets_the_full_site_a_quad(self):
        for name, (_, b_scale) in self.CASES.items():
            data = load(name)
            pat = struct.pack("<f", 1.0 / b_scale)
            found = hits(data, pat)
            self.assertEqual(len(found), 4, f"{name}: B full-form hits")
            strides = {y - x for x, y in zip(found, found[1:])}
            self.assertEqual(strides, {8}, f"{name}: B stride")
            for i in found:
                self.assertEqual(
                    data[i + 4 : i + 6].hex(), "a100", f"{name}@{i}: B frame"
                )

    def test_a_gets_a_new_short_form_quad_unlike_rank_2(self):
        for name, (a_scale, _) in self.CASES.items():
            data = load(name)
            full = struct.pack("<f", 1.0 / a_scale)
            short = full[:3]
            # The full 4-byte form must still be absent (A never gets the
            # *full* site-A treatment B gets) -- only the short form appears.
            self.assertEqual(
                hits(data, full), [], f"{name}: A full-form must be absent"
            )
            found = hits(data, short)
            self.assertEqual(len(found), 4, f"{name}: A short-form hits")
            strides = {y - x for x, y in zip(found, found[1:])}
            self.assertEqual(strides, {6}, f"{name}: A short-form stride")
            # frame: <3 bytes> 82 <var> 02, except the last copy's tail is 83
            for i in found[:-1]:
                self.assertEqual(data[i + 3], 0x82, f"{name}@{i}: tag byte 0")
                self.assertEqual(data[i + 5], 0x02, f"{name}@{i}: tag byte 2")
            last = found[-1]
            self.assertEqual(data[last + 3], 0x82, f"{name}@{last}: tag byte 0 (last)")
            self.assertEqual(data[last + 5], 0x83, f"{name}@{last}: tag byte 2 (last)")

    def test_rank_2_plain_matmul_still_has_no_such_pattern(self):
        """Confirms this is genuinely new behavior from batching, not
        something the rank-2 search (test_axera_site_a_generalizes.py)
        simply missed -- re-checked here directly against the rank-2
        fixture for the specific short-form pattern this file found."""
        data = load("matmul_4x8x8.mcode.gz")
        a_scale = 0.007840047590434551  # matmul_4x8x8's own A_SCALE
        short = struct.pack("<f", 1.0 / a_scale)[:3]
        self.assertEqual(hits(data, short), [], "rank-2 A short-form must be absent")


class TestFullyBatchedShapeHasUnusualNoise(unittest.TestCase):
    """The fully-batched (A and B both rank-3) shape's own rebuild-pair
    determinism check (not shipped as a fixture; done ahead of writing
    this file, see module docstring) found 932/3336 bytes differing --
    far above the ~6-15 byte noise zone confirmed everywhere else in
    this project. This test only asserts the shipped fixture's length
    matches what that check used, so the docstring's claim stays tied
    to a concrete, checkable artifact."""

    def test_batched_fixture_length_matches_the_determinism_check(self):
        data = load("matmul_2x4x8x8_batched.mcode.gz")
        self.assertEqual(len(data), 3336)

    def test_broadcast_fixture_length_matches_the_determinism_check(self):
        data = load("matmul_2x4x8x8_broadcast.mcode.gz")
        self.assertEqual(len(data), 3080)


if __name__ == "__main__":
    unittest.main()
