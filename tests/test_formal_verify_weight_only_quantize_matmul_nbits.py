"""Formal check for WeightOnlyQuantizeMatMulNBits (opt-in; onnxsim's own
``onnxsim/passes/weight_only_quantize_matmul_nbits.h``): the same "only the
constant weight is quantized, the activation ``X`` is left completely
untouched" design as ``weight_only_quantize_matmul.h``/``weight_only_
quantize_int4_matmul.h`` -- READ ``test_formal_verify_weight_only_quantize_
int4_matmul.py`` FIRST, this file only documents what differs -- but this
pass rewrites ``Y = MatMul(X, W)`` (or a "vanilla" ``Gemm``, bias passed
through as ``MatMulNBits``' own optional bias input) into ONNX Runtime's
*vendor-specific* ``com.microsoft::MatMulNBits`` contrib op instead of
standard-ONNX opset-21 INT4-tensor-plus-``DequantizeLinear``::

    Wq, Ws := QuantizeWeightForMatMulNBits(W)   # packed uint8 nibbles + scales
    Y = com.microsoft::MatMulNBits(X, Wq, Ws, K=K, N=N, bits=4, block_size=32)

Two genuinely new wrinkles this file's proof needs that ``weight_only_
quantize_int4_matmul``'s does not (read ``QuantizeWeightForMatMulNBits`` in
``weight_only_quantize_matmul_nbits.h`` in full -- it is where both live):

1. **RAGGED blocking, not exact.** ``weight_only_quantize_int4_matmul``
   requires ``K % kBlockSize == 0`` and declines otherwise (a ragged last
   block is "left to a future extension"). This pass instead always fires
   (given a valid ``block_size`` and a constant 2-D float32 weight):
   ``k_blocks = ceil(K / block_size)``, so the last block may hold fewer
   than ``block_size`` REAL elements. For ``k >= K`` (past the real weight,
   inside that ragged last block only), ``code_at`` returns the FIXED
   zero-point code ``8`` *unconditionally* -- not a rounded value of any
   real data, since none exists there -- which dequantizes to exactly
   ``(8 - 8) * scale == 0``. The header comment calls these positions
   "never read by the kernel (bounded by K)": ``MatMulNBits``' own ``K``
   attribute tells the ORT kernel where to stop accumulating, so in
   principle *any* byte value there would be a don't-care; the pass fills
   it with a well-defined value anyway (0 after dequant) purely so the
   initializer's own contents are never undefined. ``block_size`` itself
   is a fixed compile-time constant here (``static constexpr int64_t
   kBlockSize = 32`` on the pass struct, confirmed by reading the header),
   not a runtime parameter the way ``weight_only_quantize_int4_matmul``'s
   own ``kBlockSize`` is *also* fixed at 32 but at least conceptually
   pluggable per-call in that file's sibling ``TryQuantizeWeightBlockwiseInt4
   InPlace`` helper -- here there is no Python-visible knob at all (see
   ``onnxsim.quantize_weight_only_matmul_nbits``'s signature: no
   ``block_size`` argument, unlike the INT4 pass's None here either, but
   confirmed by reading the pass struct directly).

2. **A different zero-point convention.** ``weight_only_quantize_int4_
   matmul`` has no zero point at all -- its ``TryQuantizeWeightBlockwiseInt4
   InPlace`` codes are signed, ``code = clip(round(w / scale), -7, 7)``,
   stored directly as ONNX's native signed INT4 type. This pass has no
   int4-native tensor type available (packed uint8 nibbles only), so
   ``QuantizeWeightForMatMulNBits`` instead centers codes on
   ``MatMulNBits``' own documented default zero point ``2^(bits-1) = 8``:
   ``code = clip(round(w / scale), -8, 7) + 8`` (an UNSIGNED code in
   ``[0, 15]``), and the op's own semantics dequantize as
   ``Wdq = (code - zero_point) * scale = (code - 8) * scale``. Confirmed
   independently below (``test_matmul_nbits_offset_quantization_matches_
   rounding_bound``, not assumed from the other pass's file) that this is
   purely a REPRESENTATIONAL shift, not a different error bound: writing
   ``code = wq + 8`` for the very same integer ``wq`` the INT4 pass would
   have produced, ``(code - 8) * scale == wq * scale`` -- the ``+8``/``-8``
   cancel exactly -- so the round-to-nearest bound
   ``|W[k, n] - Wdq[k, n]| <= Ws[block, n] / 2`` has the *identical shape*
   as the INT4 pass's own bound, independent of which convention encodes
   the signed code.

This is still a *single-operand* special case of ``quantized_mac_bound``'s
general MAC bound, ``eps_x := 0`` exactly as in ``weight_only_quantize_
matmul``'s and ``weight_only_quantize_int4_matmul``'s own proofs, with a
PER-BLOCK error bound (each tap's own budget is ITS block's ``Ws[block, n]
/ 2``, not one shared bound for every tap) reused verbatim from ``weight_
only_quantize_int4_matmul``'s own per-block generalization -- proved below
with a concrete ``_K = 2`` split across two SEPARATE one-element blocks
(``_BLOCK_SIZE = 1``), matching ``quantized_mac_bound``'s own ``_K = 2``
convention for concrete bound queries rather than ``weight_only_quantize_
int4_matmul``'s own larger ``_K = 4`` -- two independently-scaled blocks is
already the minimal case that exercises "the block structure is genuinely
used, not accidentally collapsed to a single scale" (the same claim a
dedicated negative-control test below confirms is not vacuous: naively
reusing one block's scale to bound the other's error is UNSOUND once the
two scales can differ). Every bound-proving query below uses the DIRECT-
ERROR-VARIABLE idiom (a free ``ew`` bounded directly by the rounding
hypothesis) rather than reconstructing a dequantized value from separate
code/zero-point/scale multiplicands inside the same query -- see
``test_formal_verify_dynamic_quantize_matmul.py``'s own module docstring
for the incident writeup (a combined nonlinear reconstruction hung Z3 past
a minute); the one place this file DOES reconstruct from an integer code
(the offset-convention test) is a single-tap, non-summed identity query,
the same shape ``weight_only_quantize_matmul.py``'s and ``weight_only_
quantize_int4_matmul.py``'s own analogous "does ``DequantizeLinear``'s
literal semantics match what the bound assumes" tests already use safely.

The genuinely NEW Z3 content past that reused per-block bound is about the
ragged last block specifically -- two lemmas, not one, since "the bound only
needs to hold over the real prefix" is actually two separate claims:

- The FIXED dead-position code (``8``) dequantizes to EXACTLY ``0`` for
  every scale, unconditionally -- an identity, not merely a bound, since
  there is no real data there for a "rounding error" to even be defined
  against (``test_matmul_nbits_dead_position_code_dequantizes_to_exactly_
  zero``).
- Extending a block's error sum from its real prefix out to its full
  nominal ``block_size`` width changes NOTHING, because the extra "tap"
  has no real activation to multiply against at all -- ``X`` only ever has
  ``K`` columns; a dead position, at nominal width, is exactly a term
  multiplied by an activation of 0, regardless of what value (the pass's
  own 0, or anything else) sits in the dequantized slot there. So the
  per-block bound already proved over real taps needs no ragged-block
  special case; the "extra" nominal width simply never contributes
  (``test_matmul_nbits_ragged_block_bound_needs_only_real_prefix``).

A bias-variant test (Gemm's bias, left untouched, cancels the same way as
every prior pass in this family) rounds out the Z3 side.

Differential tests build a plain float ``MatMul``/``Gemm`` via
``onnx.parser`` (per ``CLAUDE.md``) with ``K = 40``, ``block_size = 32``
(``kBlockSize``, a fixed compile-time constant, confirmed by reading the
header struct) -- one full block (taps 0-31) and one genuinely RAGGED last
block (taps 32-39 real, 40-63 dead) -- run the real pass via
``onnxsim.quantize_weight_only_matmul_nbits`` (the dedicated Python entry
point, confirmed by reading ``onnxsim/quantize_entry.cpp``'s
``QuantizeWeightOnlyMatMulNBits`` and ``onnxsim/onnx_simplifier.py``'s
wrapper of the same name -- mirroring ``quantize_weight_only_int4``'s own
dedication) and, for the structural/firing checks, also via ``simplify_
isolated_extra`` (the pass is registered as an opt-in "other" optimizer
under its own name, ``weight_only_quantize_matmul_nbits``, confirmed via
``C._list_other_optimizers()``). Confirmed: the ``com.microsoft`` opset
import is added; the produced ``MatMulNBits`` node's ``K``/``N``/``bits``/
``block_size`` attributes match the real weight's shape; unpacking the
returned packed-nibble ``Wq`` tensor BY HAND (low nibble first, per the
header comment) reproduces an independent numpy re-implementation of
``QuantizeWeightForMatMulNBits``'s formula for every REAL (``k < K``) code,
and every DEAD (``k >= K``, last block only) nibble decodes to exactly
``8``; ``MatMulNBits`` has a working ONNX Runtime CPU kernel, confirmed
FIRST with a minimal hand-built model (independent of this pass's C++
entirely) before trusting the pass's own output against it; the real
quantized graph's numeric output stays within the proved per-block bound
against the true float ``MatMul``, for an ``X`` whose values are genuinely
nonzero everywhere, including the real columns immediately adjacent to the
ragged block's own real/dead boundary (so an off-by-one in where "real"
stops would show up as a bound violation, not merely as multiplying an
already-zero column); the ORT kernel truly IGNORES whatever is packed past
``K`` rather than accidentally reading it -- confirmed by corrupting the
dead nibbles to arbitrary non-zero-point codes post-quantization and
checking the output is bit-for-bit UNCHANGED; a Gemm bias variant
(``MatMulNBits``' own optional bias input, added via the two ``Undefined``
placeholders for ``zero_points``/``g_idx`` exactly as the header describes);
and a weight whose ``K`` IS evenly divisible by ``block_size`` (the
non-ragged, degenerate case: ``k_blocks == K / block_size`` exactly, no dead
positions at all).

Every bound-checking differential test passes ``check_n=0`` to ``simplify_
isolated_extra`` for the same reason every pass in this quantization family
does: this is a genuinely lossy 4-bit rewrite, coarser than onnxsim's own
default random-input equivalence check's tolerance, confirmed empirically by
every prior file in this family.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 2  # concrete number of contraction taps -- matches quantized_mac_bound's
# own _K (and weight_only_quantize_matmul's), not weight_only_quantize_int4_
# matmul's larger _K = 4: two SEPARATE one-element blocks (_BLOCK_SIZE = 1
# below) is already the minimal case that exercises independent per-block
# scales, so there is no need for the larger query the INT4 file's
# 2-taps-per-block layout used.
_BLOCK_SIZE = 1  # each tap is its own block: Ws0 for tap 0, a genuinely
# different Ws1 for tap 1 -- the strongest (smallest) case of "taps do NOT
# all share one scale". Block-SHARING semantics (multiple taps under one
# scale) is exercised separately below, in
# test_matmul_nbits_offset_quantization_matches_rounding_bound.
_NUM_BLOCKS = _K // _BLOCK_SIZE


def _block_of(k):
    return k // _BLOCK_SIZE


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _bound_formulas():
    """Builds the Z3 vocabulary for the single-operand, PER-BLOCK
    bounded-error claim -- reused verbatim (down-sized to ``_K = 2``) from
    ``weight_only_quantize_int4_matmul``'s own per-block generalization of
    ``weight_only_quantize_matmul``'s single-scale bound: ``X`` has no error
    term at all (never quantized), only ``W`` does, via the free per-tap
    error variable ``ew``, with one scale variable PER BLOCK (``Ws[b]``).
    This bound's *shape* does not depend on this pass's own ``+8``
    zero-point offset at all (see the module docstring and
    ``test_matmul_nbits_offset_quantization_matches_rounding_bound`` for why
    that offset cancels out of the dequantized value entirely) -- it is
    exactly the same claim as the INT4 pass's, restated here to confirm this
    pass's own C++ produces a ``Wdq`` satisfying it too.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k, n] - Wdq[k, n]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]  # per-block scale Ws[n, block]

    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        # Each tap's error is bounded by ITS OWN block's scale -- Ws0 for
        # tap 0, a genuinely different Ws1 for tap 1 -- not one shared bound.
        *[_abs(ew[k]) <= Ws[_block_of(k)] / 2 for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))

    bound = sum((Ws[_block_of(k)] / 2) * _abs(X[k]) for k in range(_K))

    return float_matmul, dequant_matmul, rounding_bounds, bound


def test_matmul_nbits_error_is_bounded():
    # The genuine bounded-error claim: given W's own per-BLOCK rounding
    # bound (|W[k, n] - Wdq[k, n]| <= Ws[block_of(k), n] / 2) and X
    # completely unchanged, the true float dot product and the one computed
    # against MatMulNBits' dequantized weight cannot differ by more than
    # sum_k (Ws[block_of(k), n] / 2) * |X[i, k]| -- quantized_mac_bound's
    # own bound with eps_x fixed to 0, with a genuinely per-tap eps_w term.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_matmul_nbits_bias_variant_error_is_bounded():
    # The "+ Bias" branch (a Gemm with a bias input, routed to MatMulNBits'
    # own optional bias input): this pass never touches the bias's values at
    # all, so adding the same Bias(n) to both the true and the dequantized
    # computation leaves their difference -- and therefore the bound on it
    # -- unchanged; Bias cancels out of the error term algebraically.
    float_matmul, dequant_matmul, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_matmul + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_matmul_nbits_negative_control_requires_rounding_bound():
    # Sanity check that the bound proved above is genuine, not vacuous: with
    # no error budget assumed on ew at all (only Ws0, Ws1 > 0), the same
    # bound is not a theorem -- Z3 must find a real counterexample.
    float_matmul, dequant_matmul, _rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_matmul

    solver = z3.Solver()
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any rounding-error budget on ew -- "
        "negative control is vacuous"
    )


def test_matmul_nbits_uniform_bound_is_unsound_across_blocks():
    # New content this pass's proof needs that a single shared-Ws bound does
    # not: the per-block sum is not merely untested-but-equivalent to one
    # shared scale -- it is genuinely NECESSARY. If one instead (incorrectly)
    # bounded every tap's error using block 0's scale alone (Ws0 / 2), that
    # claim is NOT a theorem once block 1's real scale Ws1 can exceed Ws0 --
    # Z3 finds a counterexample where block 1's actual rounding error (up to
    # Ws1 / 2) overflows the too-small Ws0-only budget.
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    Ws = [z3.Real(f"Ws{b}") for b in range(_NUM_BLOCKS)]

    dequant_w = [W[k] - ew[k] for k in range(_K)]
    rounding_bounds = z3.And(
        *[Ws[b] > 0 for b in range(_NUM_BLOCKS)],
        *[_abs(ew[k]) <= Ws[_block_of(k)] / 2 for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_matmul = sum(X[k] * dequant_w[k] for k in range(_K))
    error = float_matmul - dequant_matmul

    # The naive/wrong claim: bound every tap's contribution using ONLY
    # block 0's scale Ws0.
    uniform_bound = (Ws[0] / 2) * sum(_abs(X[k]) for k in range(_K))

    solver = z3.Solver()
    solver.add(rounding_bounds)
    solver.add(z3.Not(z3.And(error <= uniform_bound, -error <= uniform_bound)))
    assert solver.check() == z3.sat, (
        "the naive single-scale (block 0 only) bound holds even though "
        "block 1 has its own, potentially larger scale -- this pass's "
        "per-block sum is not actually load-bearing, which would be wrong"
    )


def test_matmul_nbits_offset_quantization_matches_rounding_bound():
    # New content this pass's proof needs that weight_only_quantize_int4_
    # matmul's does not: that pass's codes are signed and zero-point-free
    # (code = wq directly, wq in [-7, 7]); this pass instead has NO
    # int4-native tensor type -- only a packed uint8 nibble format -- so
    # QuantizeWeightForMatMulNBits centers codes on MatMulNBits' own
    # documented default zero point (code = wq + 8, an UNSIGNED code in
    # [0, 15], for the very same integer wq the INT4 pass would produce) and
    # MatMulNBits' own semantics dequantize as Wdq = (code - 8) * scale.
    # Confirmed here, independently, that this is a pure representational
    # shift, not a different error bound: the +8/-8 cancel exactly, so the
    # dequantized value -- and therefore the round-to-nearest bound -- is
    # IDENTICAL in shape to the INT4 pass's own zero-point-free bound. Also
    # exercises block-SHARING (unlike the per-tap-own-block setup above):
    # w0 and w1 fall in the SAME block (shared scale ws_b0, shared code
    # offset), w2 in a different block (its own, independent ws_b1) --
    # mirroring QuantizeWeightForMatMulNBits' own per-block scale loop,
    # which computes exactly one scale per (block, output channel) and
    # applies the identical code_at formula to every real tap in it.
    w0, w1, w2, ws_b0, ws_b1 = z3.Reals("w0 w1 w2 ws_b0 ws_b1")
    wq0, wq1, wq2 = z3.Ints("wq0 wq1 wq2")  # round(w / ws): integer within 0.5
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        ws_b0 > 0,
        ws_b1 > 0,
        # k=0, k=1: SAME block -> SAME scale ws_b0.
        wq0 - w0 / ws_b0 <= half,
        w0 / ws_b0 - wq0 <= half,
        wq1 - w1 / ws_b0 <= half,
        w1 / ws_b0 - wq1 <= half,
        # k=2: a DIFFERENT block -> its own, independent scale ws_b1.
        wq2 - w2 / ws_b1 <= half,
        w2 / ws_b1 - wq2 <= half,
    )
    # This pass's own code_at (for a REAL position) and MatMulNBits' own
    # dequantization formula: code = wq + 8 (an unsigned nibble in [0, 15]
    # once clamped -- clamping is irrelevant to this bound, which only
    # concerns the round-to-nearest property, exactly like every prior
    # pass's own analogous test in this suite), Wdq = (code - 8) * scale.
    code0 = wq0 + 8
    code1 = wq1 + 8
    code2 = wq2 + 8
    wdq0 = (z3.ToReal(code0) - 8) * ws_b0
    wdq1 = (z3.ToReal(code1) - 8) * ws_b0
    wdq2 = (z3.ToReal(code2) - 8) * ws_b1

    e0, e1, e2 = w0 - wdq0, w1 - wdq1, w2 - wdq2
    prove(
        z3.Implies(
            hypotheses,
            z3.And(
                e0 <= ws_b0 / 2,
                -e0 <= ws_b0 / 2,
                e1 <= ws_b0 / 2,
                -e1 <= ws_b0 / 2,
                e2 <= ws_b1 / 2,
                -e2 <= ws_b1 / 2,
            ),
        )
    )


def test_matmul_nbits_dead_position_code_dequantizes_to_exactly_zero():
    # New content specific to this pass's RAGGED blocking (weight_only_
    # quantize_int4_matmul has no such case -- it declines a ragged K
    # entirely): for a position past the real weight in a block's own
    # nominal width (k >= K), code_at returns the FIXED code 8
    # unconditionally -- not a rounded value of any real data, since none
    # exists there. This is an IDENTITY, not merely a bound: the
    # dequantized value is EXACTLY 0 for every possible scale, with no
    # rounding hypothesis needed at all (there is nothing to round).
    code = z3.Int("code")
    s = z3.Real("s")
    dequant = (z3.ToReal(code) - 8) * s
    prove(z3.Implies(code == 8, dequant == 0))


def test_matmul_nbits_ragged_block_bound_needs_only_real_prefix():
    # New content this pass's proof needs that no prior pass in this family
    # does: the per-block bound proved above (test_matmul_nbits_error_is_
    # bounded) is stated over REAL taps only (X[k] for k in [0, K)) -- this
    # confirms that is not an oversight: extending a ragged block's error
    # sum out to its full NOMINAL block_size width changes nothing, because
    # X simply has no column there to multiply against at all (MatMulNBits'
    # own K attribute is exactly "how many columns X has", not a separate
    # runtime check) -- modeled here as multiplying by an activation of 0,
    # which is what "no such column exists" amounts to algebraically. The
    # phantom weight/dequant values at that position (W1_phantom, Wdq1_
    # phantom) are left completely FREE/unconstrained -- not even fixed to
    # the pass's own actual 0 -- to show the real prefix's bound holds
    # regardless of what (if anything) a dead position's value happens to
    # be, not merely for the pass's own specific choice.
    X0, W0, ew0, Ws = z3.Reals("X0 W0 ew0 Ws")
    W1_phantom, Wdq1_phantom = z3.Reals("W1_phantom Wdq1_phantom")

    rounding_bounds = z3.And(Ws > 0, _abs(ew0) <= Ws / 2)

    error_real_prefix_only = (X0 * W0) - (X0 * (W0 - ew0))
    # The "dead" slot's activation is 0 -- there is no k = K column of X --
    # so it contributes 0 to BOTH the true and dequantized sums, regardless
    # of W1_phantom / Wdq1_phantom.
    error_full_nominal_width = (X0 * W0 + 0 * W1_phantom) - (
        X0 * (W0 - ew0) + 0 * Wdq1_phantom
    )

    bound = (Ws / 2) * _abs(X0)
    prove(
        z3.Implies(
            rounding_bounds,
            z3.And(
                error_real_prefix_only == error_full_nominal_width,
                error_full_nominal_width <= bound,
                -error_full_nominal_width <= bound,
            ),
        )
    )


# --- Differential tests -----------------------------------------------------

_KBLOCK = 32  # this pass's kBlockSize -- a fixed C++ compile-time constant,
# confirmed by reading the header struct; unlike weight_only_quantize_int4_
# matmul, there is no Python-visible block_size parameter to pass.


def _model(body, initializer=(), opset=17, ir_version=10):
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


def _f32(array, name):
    return numpy_helper.from_array(array.astype(np.float32), name)


def _int_attr(node, name):
    return next(a.i for a in node.attribute if a.name == name)


def _quantize_weight_for_matmul_nbits(w, transposed, block_size):
    """Independent numpy re-implementation of
    ``QuantizeWeightForMatMulNBits`` (weight_only_quantize_matmul_nbits.h):
    ``w`` is a 2-D float32 array laid out as ``w_t`` is (``[N, K]`` when
    ``transposed`` else ``[K, N]``). Returns ``(packed uint8 [N, k_blocks,
    blob_size], scale float32 [N, k_blocks], codes uint8 [N, k_blocks *
    block_size])`` -- ``codes`` is the UNPACKED per-position code (real
    positions per the round/clamp/+8 formula, dead positions fixed at 8),
    handy for checking real vs. dead positions separately.
    """
    dim0, dim1 = w.shape
    K = dim1 if transposed else dim0
    N = dim0 if transposed else dim1
    k_blocks = -(-K // block_size)  # ceil(K / block_size)
    blob_size = block_size // 2

    def at(k, n):
        return float(w[n, k] if transposed else w[k, n])

    scale = np.ones((N, k_blocks), dtype=np.float32)
    codes = np.full((N, k_blocks * block_size), 8, dtype=np.uint8)
    for n in range(N):
        for kb in range(k_blocks):
            k0 = kb * block_size
            k1 = min(K, k0 + block_size)
            m = 0.0
            for k in range(k0, k1):
                m = max(m, abs(at(k, n)))
            s = m / 7.0 if m > 0.0 else 1.0
            scale[n, kb] = s
            for k in range(k0, k1):
                q = np.clip(np.round(at(k, n) / s), -8.0, 7.0)
                codes[n, k] = np.uint8(q + 8.0)

    packed = np.zeros((N, k_blocks, blob_size), dtype=np.uint8)
    for n in range(N):
        for kb in range(k_blocks):
            k0 = kb * block_size
            for j in range(blob_size):
                lo = codes[n, k0 + 2 * j]
                hi = codes[n, k0 + 2 * j + 1]
                packed[n, kb, j] = (lo & 0x0F) | ((hi & 0x0F) << 4)

    return packed, scale, codes


def _find_uint8_init(model, exclude_name=None):
    return next(
        i
        for i in model.graph.initializer
        if i.data_type == onnx.TensorProto.UINT8 and i.name != exclude_name
    )


def _find_float_init(model, exclude_name):
    return next(
        i
        for i in model.graph.initializer
        if i.data_type == onnx.TensorProto.FLOAT and i.name != exclude_name
    )


def test_matmul_nbits_pass_fires_ragged_block_matches_scheme():
    # K=40 with this pass's fixed block_size=32 gives k_blocks=2: one FULL
    # block (taps 0-31) and one genuinely RAGGED last block (taps 32-39
    # real, 40-63 dead -- 8 real positions, 24 dead ones). Confirms this
    # pass fires here at all (weight_only_quantize_int4_matmul would decline
    # this exact K outright), the com.microsoft opset import, the node's
    # attributes, and the packed codes against the independent
    # re-implementation above -- both the real prefix AND that every dead
    # nibble decodes to exactly 8.
    rng = np.random.default_rng(0)
    rows, K, N = 4, 40, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.6
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_matmul_nbits", check_n=0
    )

    nbits_node = producer(sim_model, "Y")
    assert nbits_node.op_type == "MatMulNBits"
    assert nbits_node.domain == "com.microsoft"
    assert any(
        o.domain == "com.microsoft" and o.version == 1 for o in sim_model.opset_import
    )

    x_input = nbits_node.input[0]
    assert x_input == "X"  # the activation is passed through unchanged

    assert _int_attr(nbits_node, "K") == K
    assert _int_attr(nbits_node, "N") == N
    assert _int_attr(nbits_node, "bits") == 4
    assert _int_attr(nbits_node, "block_size") == _KBLOCK

    wq_name = nbits_node.input[1]
    ws_name = nbits_node.input[2]
    wq_init = next(i for i in sim_model.graph.initializer if i.name == wq_name)
    ws_init = next(i for i in sim_model.graph.initializer if i.name == ws_name)
    wq = numpy_helper.to_array(wq_init)  # [N, k_blocks, blob_size]
    ws = numpy_helper.to_array(ws_init)  # [N, k_blocks]
    k_blocks = -(-K // _KBLOCK)
    assert wq.shape == (N, k_blocks, _KBLOCK // 2)
    assert ws.shape == (N, k_blocks)

    _expected_packed, expected_scale, expected_codes = (
        _quantize_weight_for_matmul_nbits(weight, False, _KBLOCK)
    )
    np.testing.assert_array_equal(wq, _expected_packed)
    np.testing.assert_allclose(ws, expected_scale, rtol=1e-6)

    # Unpack by hand (low nibble first, per the header comment) and check
    # real vs. dead positions separately.
    for n in range(N):
        for kb in range(k_blocks):
            k0 = kb * _KBLOCK
            for j in range(_KBLOCK // 2):
                byte = int(wq[n, kb, j])
                lo, hi = byte & 0x0F, (byte >> 4) & 0x0F
                for offset, code in ((0, lo), (1, hi)):
                    k = k0 + 2 * j + offset
                    if k < K:
                        assert code == expected_codes[n, k]
                    else:
                        assert code == 8  # dead position: fixed zero-point code


def test_matmul_nbits_pass_fires_full_blocks_only_when_k_divisible():
    # The non-ragged, degenerate case: K=64 is exactly two full 32-element
    # blocks (k_blocks == K / block_size exactly), so no dead positions
    # exist at all -- every code decodes from the same round/clamp/+8
    # formula, with no "code == 8, unconditionally" branch ever taken.
    rng = np.random.default_rng(1)
    rows, K, N = 4, 64, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.6
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "weight_only_quantize_matmul_nbits", check_n=0
    )
    nbits_node = producer(sim_model, "Y")
    assert nbits_node.op_type == "MatMulNBits"
    k_blocks = K // _KBLOCK
    assert k_blocks * _KBLOCK == K  # exactly divisible, no ragged remainder

    wq_name, ws_name = nbits_node.input[1], nbits_node.input[2]
    wq = numpy_helper.to_array(
        next(i for i in sim_model.graph.initializer if i.name == wq_name)
    )
    ws = numpy_helper.to_array(
        next(i for i in sim_model.graph.initializer if i.name == ws_name)
    )
    assert wq.shape == (N, k_blocks, _KBLOCK // 2)
    assert ws.shape == (N, k_blocks)

    _expected_packed, expected_scale, _codes = _quantize_weight_for_matmul_nbits(
        weight, False, _KBLOCK
    )
    np.testing.assert_array_equal(wq, _expected_packed)
    np.testing.assert_allclose(ws, expected_scale, rtol=1e-6)


def test_matmul_nbits_ort_kernel_matches_documented_semantics():
    # Confirms EMPIRICALLY (not merely assumed) that ONNX Runtime actually
    # implements com.microsoft::MatMulNBits with a working CPU kernel that
    # matches the documented low-nibble-first, code-minus-8 dequantization
    # semantics -- built completely independently of this pass's own C++
    # (a hand-packed model, no onnxsim involved at all), with a genuinely
    # RAGGED block (block_size=16, the smallest block size ORT's own kernel
    # accepts -- 4, used by the Z3 tests above, is NOT one of ORT's
    # supported sizes, confirmed empirically) so the K-bounded dead region
    # is exercised here too, not just in the pass's own output.
    block_size = 16
    rng = np.random.default_rng(2)
    K, N = 20, 2  # one full block (0-15) + a ragged block (16-19 real, 4 dead)
    rows = 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    packed, scale, _codes = _quantize_weight_for_matmul_nbits(weight, False, block_size)

    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": 17, "com.microsoft": 1]
        >
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = com.microsoft.MatMulNBits<K={K},N={N},bits=4,block_size={block_size}>(X, B, S)
        }}
        """
    )
    model.graph.initializer.extend(
        [numpy_helper.from_array(packed, "B"), numpy_helper.from_array(scale, "S")]
    )
    onnx.checker.check_model(model)

    x = rng.standard_normal((rows, K)).astype(np.float32)
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y,) = sess.run(["Y"], {"X": x})

    # Independent expected dequantization, built straight from packed/scale
    # (not from `weight` directly), so this really checks the OP's own
    # semantics rather than round-tripping the quantizer against itself.
    k_blocks = packed.shape[1]
    dequant = np.zeros((K, N), dtype=np.float32)
    for n in range(N):
        for kb in range(k_blocks):
            s = scale[n, kb]
            k0 = kb * block_size
            for j in range(block_size // 2):
                byte = int(packed[n, kb, j])
                lo, hi = byte & 0x0F, (byte >> 4) & 0x0F
                if k0 + 2 * j < K:
                    dequant[k0 + 2 * j, n] = (lo - 8) * s
                if k0 + 2 * j + 1 < K:
                    dequant[k0 + 2 * j + 1, n] = (hi - 8) * s
    y_expected = x @ dequant
    np.testing.assert_allclose(y, y_expected, atol=1e-5, rtol=1e-5)


def test_matmul_nbits_output_within_proved_bound_for_ragged_block():
    # Differential check mirroring weight_only_quantize_int4_matmul's own
    # analogous test: run the real quantized graph through onnxruntime and
    # confirm every output element's error against the true float MatMul
    # stays within the per-block bound proved above -- Ws read directly
    # from the static initializer, block index computed per k. X's values
    # are genuinely nonzero everywhere, INCLUDING the real columns 32-39
    # immediately adjacent to the ragged block's real/dead boundary, so an
    # off-by-one in where "real" stops (e.g. treating k=32..38 as dead too)
    # would show up as a bound violation here, not merely as multiplying an
    # already-near-zero column.
    rng = np.random.default_rng(3)
    rows, K, N = 4, 40, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.6
    x = rng.standard_normal((rows, K)).astype(np.float32) * 2.0
    x[:, K - 8 : K] += 5.0  # boundary-adjacent real columns: emphatically nonzero
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.quantize_weight_only_matmul_nbits(model)
    onnx.checker.check_model(quantized)
    nbits_node = next(n for n in quantized.graph.node if n.op_type == "MatMulNBits")
    wq_name, ws_name = nbits_node.input[1], nbits_node.input[2]
    ws = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == ws_name)
    )  # [N, k_blocks]
    wq = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == wq_name)
    )
    k_blocks = -(-K // _KBLOCK)
    assert wq.shape == (N, k_blocks, _KBLOCK // 2)
    assert ws.shape == (N, k_blocks)

    sess = ort.InferenceSession(
        quantized.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_quant,) = sess.run(["Y"], {"X": x})

    y_float = x @ weight
    error = np.abs(y_float - y_quant)

    eps_w = (ws / 2.0).T  # [k_blocks, N]: eps_w[block_of(k), n]
    block_of_k = np.arange(K) // _KBLOCK  # [K], real taps only -- dead
    # positions (k >= K) never appear here at all, matching test_matmul_
    # nbits_ragged_block_bound_needs_only_real_prefix's own claim.
    per_tap_eps = eps_w[block_of_k, :]  # [K, N]
    bound = np.einsum("ik,kn->in", np.abs(x), per_tap_eps)
    assert np.all(error <= bound + 1e-4)


def test_matmul_nbits_kernel_ignores_corrupted_dead_nibbles():
    # New content this pass's proof needs that no prior pass in this family
    # does: confirms the ORT kernel truly IGNORES whatever is packed past K
    # -- bounded by the node's own K attribute -- rather than merely
    # happening to read a value (8, dequantizing to 0) that is harmless by
    # coincidence. Corrupts every dead nibble to an arbitrary NON-zero-point
    # code post-quantization and checks the runtime output is bit-for-bit
    # unchanged: if the kernel's own accumulation loop went past K, this
    # corruption would change the result.
    rng = np.random.default_rng(4)
    rows, K, N = 3, 40, 2
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.6
    x = rng.standard_normal((rows, K)).astype(np.float32)
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.quantize_weight_only_matmul_nbits(model)
    nbits_node = next(n for n in quantized.graph.node if n.op_type == "MatMulNBits")
    wq_name = nbits_node.input[1]
    wq_init = next(i for i in quantized.graph.initializer if i.name == wq_name)
    wq = numpy_helper.to_array(wq_init)  # [N, k_blocks, blob_size]
    N_, k_blocks, blob_size = wq.shape
    assert N_ == N

    sess = ort.InferenceSession(
        quantized.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_before,) = sess.run(["Y"], {"X": x})

    corrupted = wq.copy()
    for n in range(N):
        for kb in range(k_blocks):
            k0 = kb * _KBLOCK
            for j in range(blob_size):
                byte = int(corrupted[n, kb, j])
                lo, hi = byte & 0x0F, (byte >> 4) & 0x0F
                if k0 + 2 * j >= K:
                    lo = 3  # anything other than the pass's own 8
                if k0 + 2 * j + 1 >= K:
                    hi = 12
                corrupted[n, kb, j] = (lo & 0x0F) | ((hi & 0x0F) << 4)
    assert not np.array_equal(corrupted, wq), (
        "test setup: no dead nibble existed to corrupt"
    )

    corrupted_model = onnx.ModelProto()
    corrupted_model.CopyFrom(quantized)
    for init in corrupted_model.graph.initializer:
        if init.name == wq_name:
            init.CopyFrom(numpy_helper.from_array(corrupted, wq_name))

    sess2 = ort.InferenceSession(
        corrupted_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_after,) = sess2.run(["Y"], {"X": x})
    np.testing.assert_array_equal(y_before, y_after)


def test_matmul_nbits_gemm_transb_bias_variant():
    # PyTorch nn.Linear layout: weight [N, K], Gemm(X, W, B, transB=1), with
    # a genuinely ragged K so the bias path and the ragged-block path are
    # exercised together. Confirms the bias is routed to MatMulNBits' own
    # optional bias input (index 5) via two Undefined placeholders for
    # zero_points/g_idx (indices 3, 4), exactly as the header describes, and
    # that the runtime output still stays within the proved bound.
    rng = np.random.default_rng(5)
    rows, K, N = 3, 40, 4
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.6
    bias = rng.standard_normal(N).astype(np.float32)
    x = rng.standard_normal((rows, K)).astype(np.float32) * 1.5
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    quantized = onnxsim.quantize_weight_only_matmul_nbits(model)
    onnx.checker.check_model(quantized)
    nbits_node = next(n for n in quantized.graph.node if n.op_type == "MatMulNBits")
    assert len(nbits_node.input) == 6
    assert nbits_node.input[0] == "X"
    assert nbits_node.input[3] == ""  # zero_points: skipped
    assert nbits_node.input[4] == ""  # g_idx: skipped
    assert nbits_node.input[5] == "B"  # bias: passed through unchanged

    wq_name, ws_name = nbits_node.input[1], nbits_node.input[2]
    wq = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == wq_name)
    )
    ws = numpy_helper.to_array(
        next(i for i in quantized.graph.initializer if i.name == ws_name)
    )
    k_blocks = -(-K // _KBLOCK)
    assert wq.shape == (N, k_blocks, _KBLOCK // 2)
    assert ws.shape == (N, k_blocks)

    expected_packed, expected_scale, _codes = _quantize_weight_for_matmul_nbits(
        weight, True, _KBLOCK
    )
    np.testing.assert_array_equal(wq, expected_packed)
    np.testing.assert_allclose(ws, expected_scale, rtol=1e-6)

    sess = ort.InferenceSession(
        quantized.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (y_quant,) = sess.run(["Y"], {"X": x})
    y_float = x @ weight.T + bias
    error = np.abs(y_float - y_quant)

    eps_w = (ws / 2.0).T  # [k_blocks, N]
    block_of_k = np.arange(K) // _KBLOCK
    per_tap_eps = eps_w[block_of_k, :]  # [K, N]
    bound = np.einsum("ik,kn->in", np.abs(x), per_tap_eps)  # Bias cancels out
    assert np.all(error <= bound + 1e-4)
