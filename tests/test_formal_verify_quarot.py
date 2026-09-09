"""Formal check for Quarot (opt-in; onnxsim's own ``onnxsim/passes/quarot.h``,
``onnxsim/passes/random_orthogonal.h``): QuaRot-style rotation preprocessing
(Ashkboos et al., 2024, "QuaRot: Outlier-Free 4-Bit Inference in Rotated
LLMs") plus INT4 round-to-nearest quantization of *both* MatMul operands.
This is a genuinely new kind of algebra for this suite -- every other
opt-in-quantization file so far (``weight_only_quantize_int4_matmul``,
``dynamic_quantize_matmul``, ...) quantizes the operands *as given*; this
pass first applies an exact, invertible change of basis (a random orthogonal
rotation) to *both* operands before quantizing either one, and the
soundness of that change of basis -- not any quantization scheme -- is this
file's main new content.

It rewrites ``Y = MatMul(X, W) [+ bias]`` (``W`` a constant 2-D FLOAT32
tensor whose reduction dimension ``K`` is divisible by ``block_size``
(default 32), on an opset >= 21 model; a "vanilla" Gemm -- transA=0,
alpha=1, beta=1 -- is handled identically, its bias carried through
unchanged) into::

    U           = a fresh random orthogonal [K, K] initializer, one per layer
    Xrot        = MatMul(X, U)                         -- exact
    Xq_dequant  = round_to_nearest_int4_per_token(Xrot) -- Abs/ReduceMax/
                  Clip(eps)/Div/Round/Clip/Mul, kept in float32
    Wtilde_hat  = DequantizeLinear(Wtilde_q, Wtilde_s, axis=0, block_size)
                  -- block-wise INT4 quantization of Wtilde_kn (below)
    Y           = MatMul(Xq_dequant, Wtilde_hat) [+ bias]

--------------------------------------------------------------------------
Part 1: the exact rotation-cancellation identity (the genuinely new part)
--------------------------------------------------------------------------

Re-deriving the C++'s index algebra independently (confirmed with a
standalone numpy check before writing any Z3, per this file's own
instructions -- not merely trusted from a prompt): let ``W_kn`` be the
original weight in its native ``[K, N]`` layout (``W_kn[k, n]``), and
``w_nk[n, k] := W_kn[k, n]`` its transpose, output-channel-first (this is
exactly what ``ReadWeightNK`` -- quantize_matmul_common.h -- produces,
``quarot.h``'s own ``w_nk``). ``runTransform`` computes, for every output
row ``r`` (an ``n`` index) and every column ``c`` (a ``K``-index)::

    w_tilde_nk[r, c] = sum_k w_nk[r, k] * U[k, c]        -- i.e. w_nk @ U

and then stores its OWN transpose directly into ``[K, N]`` layout
(``codes_kn[c * N + r] = quantize(w_tilde_nk[r * K + c])`` -- literally
``Wtilde_kn[c, r] := w_tilde_nk[r, c]``, avoiding a separate Transpose node
since packing already-nibble-packed INT4 data afterward would need
unpacking/repacking anyway). Substituting::

    Wtilde_kn[c, n] = w_tilde_nk[n, c]
                    = sum_k w_nk[n, k] * U[k, c]
                    = sum_k W_kn[k, n] * U[k, c]                 (w_nk[n,k]=W_kn[k,n])
                    = sum_k U[k, c] * W_kn[k, n]
                    = sum_k U^T[c, k] * W_kn[k, n]                 (U^T[c,k] := U[k,c])
                    = (U^T @ W_kn)[c, n]

so **``Wtilde_kn = U^T @ W_kn`` exactly**, no quantization involved yet.
Meanwhile ``Xrot = X @ U`` exactly (``runTransform``'s own literal
``MatMul(info.x, u_v)``). So the rewritten computation, ignoring
quantization entirely, is::

    Xrot @ Wtilde_kn = (X @ U) @ (U^T @ W_kn) = X @ (U @ U^T) @ W_kn

which equals the ORIGINAL ``X @ W_kn`` **exactly when ``U @ U^T = I``** --
i.e. exactly when ``U`` is orthogonal (``RandomOrthogonalMatrix``'s own
documented postcondition, random_orthogonal.h: "the result already
satisfies ``U @ U^T == I``"). This is the pass's real soundness claim, and
it is a claim about ``U`` alone -- independent of, and prior to, any
quantization error. ``tests/verify_quarot_algebra.py``-style scratch numpy
(K=5, N=3, a QR-derived orthogonal ``U``) confirmed this derivation
end-to-end (``Wtilde_kn == U.T @ W_kn`` and ``Xrot @ Wtilde_kn == X @
W_kn`` both held to float64 precision, and both failed for a non-orthogonal
``U``) before any Z3 was written.

Z3 models this with ``K = 2`` (small but genuinely 2-dimensional, enough to
exercise the off-diagonal orthogonality cross-term ``u00*u10 + u01*u11 ==
0`` -- matching this suite's usual small-``K`` convention,
e.g. ``quantized_mac_bound``'s own ``_K = 2``) and ``N = 2`` (so ``W`` is a
genuine matrix, not a single column, and the claim is checked for both
output columns at once): ``U``'s four entries are free/uninterpreted Z3
Reals (a genuinely symbolic orthogonal matrix, not one concrete numeric
example), the orthogonality hypothesis is the three explicit equations
``U @ U^T == I`` expands to for ``K=2``, and ``X``/``W`` are free Reals too.
The claim is proved as nested sums (``sum_k X[k] * W[k][n] == sum_c (sum_k
X[k] * U[k][c]) * (sum_k U[k][c] * W[k][n])`` for each ``n``) rather than
via any symbolic matrix-multiply helper, exactly as suggested: it is Z3's
own job to verify the double-sum identity, not a hand-rolled matrix layer's.

A negative-control test drops the orthogonality hypothesis entirely (``U``
an arbitrary, unconstrained ``2x2`` matrix) and confirms Z3 finds a genuine
counterexample -- the identity is NOT a theorem for a general
"rotation-shaped" matrix, only for a genuinely orthogonal one. This is
exactly why the pass needs ``RandomOrthogonalMatrix`` (Gram-Schmidt on a
Gaussian matrix, which IS Haar-orthogonal by construction -- see that
file's own header comment) rather than any old square matrix.

A consumer-composition test (this suite's usual substitution-safety idiom,
e.g. ``test_formal_verify_eliminate_identity.py``) confirms that, since the
rewritten computation is *exactly* equal to the original given
orthogonality (not merely bounded), any arbitrary downstream consumer of
both output columns together sees the same result either way.

--------------------------------------------------------------------------
Part 2: quantization error, layered on top of the exact identity
--------------------------------------------------------------------------

On top of the identity above, both ``Xrot`` and ``Wtilde_kn`` are then
quantized to INT4 before the final MatMul, so the pass's REAL output is
lossy overall even though the rotation itself is exact. This file does not
re-derive the INT4 per-block weight bound from scratch -- that is exactly
``test_formal_verify_weight_only_quantize_int4_matmul.py``'s own proof
(``|Wtilde_kn[c, n] - Wtilde_hat[c, n]| <= Ws[block_of(c), n] / 2``, one
scale per 32-element block of ``c`` and output channel ``n``, reused here
unchanged -- this pass's own ``DequantizeLinear(..., axis=0,
block_size=...)`` is textually the same call shape). The activation side is
new here: unlike every prior *weight-only* pass in this suite (``eps_x :=
0``), ``Xq_dequant`` genuinely quantizes ``Xrot`` too, per TOKEN (row ``i``),
with scale ``scale_x(i) = Clip(max_c |Xrot[i, c]|, epsilon) / 7`` -- an
ordinary round-to-nearest bound, ``|Xrot[i, c] - Xq_dequant[i, c]| <=
scale_x(i) / 2``, of exactly the same *shape* ``quantize_round_trip``/
``quantized_mac_bound`` already establish, just with a per-token rather
than per-tensor scale.

Combining the two lemmas by substitution in prose rather than one combined
Z3 query (mirroring how ``test_formal_verify_dynamic_quantize_matmul.py``'s
own docstring composes its ring identity and its bound lemma -- empirically,
this suite has repeatedly found that handing Z3 rotation-identity-shaped and
rounding-bound-shaped nonlinear arithmetic in the SAME query blows up past
usable runtimes, e.g. ``dynamic_quantize_matmul``'s own documented >90s
hang, so the two are kept as separate, already-tractable queries and
combined here by hand instead): write ``Y_star[i, n] := MatMul(Xrot,
Wtilde_kn)[i, n] [+ bias]`` -- the exact, unquantized rotated computation.
By Part 1's identity (``Xrot = X @ U`` exactly, ``Wtilde_kn = U^T @ W_kn``
exactly, ``U @ U^T = I``), ``Y_star[i, n]`` is EXACTLY the true float
``MatMul(X, W)[i, n] [+ bias]`` -- no error contributed by the rotation at
all. The pass's actual output ``Y[i, n]`` then differs from ``Y_star[i, n]``
only by the SAME two-operand quantized-MAC error ``quantized_mac_bound``
already bounds in general (``test_formal_verify_quantized_mac_bound.py``'s
``eps_x * sum(|w_i|) + eps_w * sum(|x_i|) + K * eps_w * eps_x``), substituting
its free ``eps_x``/``eps_w`` with THIS pass's own two, now both genuinely
nonzero (unlike the weight-only files' ``eps_x := 0``) scales::

    eps_x(i)      := scale_x(i) / 2           (per-token, from Xq_dequant)
    eps_w(c, n)   := Ws[block_of(c), n] / 2   (per-block, from Wtilde_hat)

    |Y_true[i, n] - Y[i, n]|
        <= eps_x(i) * sum_c |Wtilde_kn[c, n]|
         + sum_c eps_w(c, n) * |Xrot[i, c]|
         + K * eps_x(i) * max_c eps_w(c, n)

(the third, cross, term needs one shared scalar bound across taps the way
``quantized_mac_bound``'s own ``K * eps_w * eps_x`` term does; substituting
the per-block ``eps_w(c, n)``'s own worst case over ``c`` keeps this a valid
upper bound even though ``eps_w`` genuinely varies by block, exactly the
same per-tap-vs-uniform reasoning
``weight_only_quantize_int4_matmul``'s own uniform-bound-is-unsound test
demonstrates). No new Z3 query is added for this combined statement --
Part 1's identity plus quantized_mac_bound's already-proved general lemma
(instantiated with a per-token ``eps_x`` and a per-block ``eps_w``, which
that lemma's proof never assumed were per-tensor/per-channel constants, only
that EACH tap's own error is bounded by SOME nonnegative scalar) already
imply it. The differential test below confirms this combined bound
empirically against a real onnxruntime run, the same way the sibling INT4
weight-only file's own analogous test does for its one-operand case.

A brief note on ``epsilon``: the per-token scale is ``Clip(max_c
|Xrot[i, c]|, epsilon) / 7``, not ``max_c |Xrot[i, c]| / 7`` directly --
without the ``Clip`` floor, an all-zero token drives ``max_c |Xrot[i, c]|``
to exactly 0, making the scale 0 and the following ``Div`` a division by
zero. A tiny Z3 lemma below confirms the floored scale is always strictly
positive given ``epsilon > 0``, and that it equals exactly ``epsilon`` (not
merely "some small number") on that exact zero-token boundary case.

--------------------------------------------------------------------------
Differential tests
--------------------------------------------------------------------------

Unlike every other file in this suite, this pass takes a random per-layer
rotation seeded by a caller-supplied ``seed`` that has NO path through
``simplify()``'s ``extra_optimizers``/``skipped_optimizers`` interface (that
interface is just a list of pass names -- ``QuarotSeed()`` is a
process-global set by ``ApplyQuarot`` in ``quantize_entry.cpp``
immediately before calling ``OptimizeFixed``, exactly the way
``QuarotBlockSize()``/``QuarotEpsilon()`` are). The differential tests below
therefore call :func:`onnxsim.apply_quarot_cpp` directly (per the header
comments in ``onnxsim/passes/quarot.h`` and ``onnxsim/onnx_simplifier.py``'s
own docstring for that function) rather than this suite's usual
``simplify_isolated_extra`` -- the only way to get a specific, reproducible
seed from Python. ``tests/test_quarot_cpp.py`` already covers this same
entry point far more exhaustively (block_size/epsilon threading, Gemm+bias,
non-block-divisible/pre-opset21 declines, cross-checking the block
quantization math against ``onnxsim.omniquant``'s own routine, ...); the
tests here are the smaller self-contained set this suite's own convention
expects each ``test_formal_verify_*.py`` file to carry, focused on
confirming the SPECIFIC claims proved above (rotation orthogonality,
determinism, and the combined error bound) against the real compiled pass.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, z3
from onnx import parser

import onnxsim

_K = 2  # small but genuinely 2-D -- see module docstring for why this
# matches quantized_mac_bound's own _K = 2 rather than something larger.
_N = 2  # two output columns, so W is a genuine matrix and both columns'
# identities are checked at once, not just a single-column special case.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _rotation_identity_formulas():
    """Builds the Z3 vocabulary for the exact rotation-cancellation identity
    (Part 1 of the module docstring): ``U`` is a free, symbolic ``K x K``
    matrix; ``orthogonality`` is the explicit ``U @ U^T == I`` equations for
    ``K = 2``; ``Wtilde`` is built from ``U^T @ W`` via the derivation above
    (NOT hand-simplified -- Z3 verifies the double-sum identity itself).
    Returns ``(orthogonality, y_true, y_rewrite)`` where the latter two are
    length-``_N`` lists, one entry per output column.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], one activation row
    # W[k][n]: the original weight in its native [K, N] layout.
    W = [[z3.Real(f"W{k}_{n}") for n in range(_N)] for k in range(_K)]
    # U[k][c]: the random rotation, K x K, fully symbolic/general.
    U = [[z3.Real(f"U{k}_{c}") for c in range(_K)] for k in range(_K)]

    # Orthogonality hypothesis U @ U^T == I, spelled out as the three
    # explicit equations it expands to for K=2 (row norms are 1, distinct
    # rows are perpendicular) -- exactly the module docstring's u00/u01/
    # u10/u11 example.
    orthogonality = z3.And(
        sum(U[0][c] * U[0][c] for c in range(_K)) == 1,
        sum(U[1][c] * U[1][c] for c in range(_K)) == 1,
        sum(U[0][c] * U[1][c] for c in range(_K)) == 0,
    )

    # Xrot[c] = sum_k X[k] * U[k][c]  -- runTransform's literal MatMul(X, U).
    x_rot = [sum(X[k] * U[k][c] for k in range(_K)) for c in range(_K)]

    # Wtilde[c][n] = sum_k U[k][c] * W[k][n]  -- i.e. (U^T @ W)[c, n], the
    # derivation above (NOT w_nk @ U transposed by hand -- the whole point
    # is that Z3 checks the resulting double sum matches, not that this
    # helper re-asserts the conclusion).
    w_tilde = [
        [sum(U[k][c] * W[k][n] for k in range(_K)) for n in range(_N)]
        for c in range(_K)
    ]

    y_true = [sum(X[k] * W[k][n] for k in range(_K)) for n in range(_N)]
    y_rewrite = [sum(x_rot[c] * w_tilde[c][n] for c in range(_K)) for n in range(_N)]

    return orthogonality, y_true, y_rewrite


def test_quarot_rotation_cancellation_identity_holds_given_orthogonality():
    # The core new claim (Part 1): rotating X by U and (via this pass's own
    # transpose-and-rotate construction) W by U^T is EXACTLY invertible --
    # X @ (U @ U^T) @ W == X @ W -- precisely when U @ U^T == I. Both output
    # columns are checked at once (y_true[n] == y_rewrite[n] for n=0,1).
    orthogonality, y_true, y_rewrite = _rotation_identity_formulas()
    prove(
        z3.Implies(
            orthogonality,
            z3.And(*[y_true[n] == y_rewrite[n] for n in range(_N)]),
        )
    )


def test_quarot_rotation_cancellation_composes_with_arbitrary_consumer():
    # Substitution-safety idiom (test_formal_verify_eliminate_identity.py):
    # since the rewritten computation is EXACTLY equal to the original given
    # orthogonality (not merely bounded), any downstream consumer of the
    # full two-column output row sees the identical result either way, for
    # every possible consumer -- not just the one this file happens to
    # differential-test below.
    orthogonality, y_true, y_rewrite = _rotation_identity_formulas()
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort(), z3.RealSort())
    prove(
        z3.Implies(
            orthogonality,
            consumer(y_true[0], y_true[1]) == consumer(y_rewrite[0], y_rewrite[1]),
        )
    )


def test_quarot_negative_control_orthogonality_is_necessary():
    # Sanity check that Part 1's hypothesis is load-bearing, not vacuous:
    # for a genuinely ARBITRARY (unconstrained) K x K matrix U -- not
    # orthogonal -- the rotation-cancellation identity is not a theorem.
    # This is exactly why the pass needs RandomOrthogonalMatrix's own
    # Gram-Schmidt construction rather than any square matrix.
    _orthogonality, y_true, y_rewrite = _rotation_identity_formulas()
    solver = z3.Solver()
    solver.add(z3.Not(z3.And(*[y_true[n] == y_rewrite[n] for n in range(_N)])))
    assert solver.check() == z3.sat, (
        "the rotation-cancellation identity holds even for a non-orthogonal "
        "U -- negative control is vacuous, and orthogonality would not "
        "actually be a necessary hypothesis"
    )


def test_quarot_activation_scale_epsilon_floor_prevents_division_by_zero():
    # A brief but genuine correctness detail (see module docstring): the
    # per-token activation scale is Clip(max_abs, epsilon) / 7, not plain
    # max_abs / 7 -- the Clip floor exists specifically so an all-zero
    # token (max_abs == 0) still yields a strictly positive scale, avoiding
    # a division by zero in the following Div node.
    max_abs = z3.Real("max_abs")
    epsilon = z3.Real("epsilon")
    safe_max = z3.If(max_abs >= epsilon, max_abs, epsilon)  # Clip(., min=epsilon)

    hypotheses = z3.And(max_abs >= 0, epsilon > 0)
    prove(z3.Implies(hypotheses, safe_max > 0))
    # And specifically on the documented failure case (an exactly-zero
    # token): the floor engages and the scale becomes exactly epsilon / 7,
    # not merely "some positive number".
    prove(
        z3.Implies(
            z3.And(max_abs == 0, epsilon > 0),
            safe_max / 7 == epsilon / 7,
        )
    )


def _model(body, initializer=(), opset=21, ir_version=10):
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


def _walk_quarot_chain(model, output_name):
    """Walks the documented node chain backward from ``output_name``
    (mirroring this suite's usual ``producer()``-based structural checks),
    returning the key nodes/initializer names so callers can assert on
    shapes/attributes without re-deriving the chain each time. Handles both
    the plain and "+ bias" variants.
    """
    node = producer(model, output_name)
    bias_name = None
    if node.op_type == "Add":
        core_out, bias_name = node.input
        core = producer(model, core_out)
    else:
        core = node
    assert core.op_type == "MatMul"
    xq_name, wdq_name = core.input

    wdq = producer(model, wdq_name)
    assert wdq.op_type == "DequantizeLinear"

    x_dequant = producer(model, xq_name)
    assert x_dequant.op_type == "Mul"
    x_clipped_name, x_scale_name = x_dequant.input

    x_clipped = producer(model, x_clipped_name)
    assert x_clipped.op_type == "Clip"
    x_rounded_name = x_clipped.input[0]

    x_rounded = producer(model, x_rounded_name)
    assert x_rounded.op_type == "Round"

    x_scaled = producer(model, x_rounded.input[0])
    assert x_scaled.op_type == "Div"
    x_rot_name, x_scale_name2 = x_scaled.input
    assert x_scale_name2 == x_scale_name

    x_scale = producer(model, x_scale_name)
    assert x_scale.op_type == "Div"
    x_safe_max_name = x_scale.input[0]

    x_safe_max = producer(model, x_safe_max_name)
    assert x_safe_max.op_type == "Clip"
    x_max_name = x_safe_max.input[0]

    x_max = producer(model, x_max_name)
    assert x_max.op_type == "ReduceMax"
    x_abs_name = x_max.input[0]

    x_abs = producer(model, x_abs_name)
    assert x_abs.op_type == "Abs"

    x_rot = producer(model, x_abs.input[0])
    assert x_rot.op_type == "MatMul"
    x_name, u_name = x_rot.input
    assert x_rot_name == x_abs.input[0]

    return {
        "x_name": x_name,
        "u_name": u_name,
        "wdq_node": wdq,
        "bias_name": bias_name,
    }


def test_quarot_cpp_pass_fires_with_documented_node_chain():
    # Confirms the real compiled pass produces exactly the node chain the
    # module docstring (and quarot.h's own header comment) describes, for a
    # plain MatMul: MatMul(X, U) -> Abs -> ReduceMax -> Clip -> Div -> Round
    # -> Clip -> Mul -> MatMul(., DequantizeLinear(Wq, Ws, axis=0,
    # block_size)).
    rng = np.random.default_rng(0)
    rows, K, N, block_size = 4, 64, 4, 32
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_quarot_cpp(model, seed=0, block_size=block_size)

    chain = _walk_quarot_chain(quantized, "Y")
    assert chain["bias_name"] is None
    assert chain["x_name"] == "X"

    wdq = chain["wdq_node"]
    axis = next(a.i for a in wdq.attribute if a.name == "axis")
    bs = next(a.i for a in wdq.attribute if a.name == "block_size")
    assert axis == 0
    assert bs == block_size

    u_init = next(t for t in quantized.graph.initializer if t.name == chain["u_name"])
    assert list(u_init.dims) == [K, K]

    wq_name, ws_name = wdq.input
    wq_init = next(t for t in quantized.graph.initializer if t.name == wq_name)
    ws_init = next(t for t in quantized.graph.initializer if t.name == ws_name)
    assert list(wq_init.dims) == [K, N]
    assert list(ws_init.dims) == [K // block_size, N]


def test_quarot_cpp_gemm_bias_variant_fires_with_add_node():
    # The "+ bias" branch: a vanilla Gemm's bias is carried through
    # unchanged via a trailing Add, per the module docstring.
    rng = np.random.default_rng(1)
    rows, K, N = 3, 32, 5
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32) * 0.1
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    quantized = onnxsim.apply_quarot_cpp(model, seed=0)
    chain = _walk_quarot_chain(quantized, "Y")
    assert chain["bias_name"] == "B"


def test_quarot_cpp_rotation_matrix_is_genuinely_orthogonal():
    # THE single most important differential confirmation for Part 1's
    # proof: the real pass's own random U (not a hand-picked example)
    # actually satisfies the orthogonality hypothesis the Z3 proof requires
    # -- U @ U.T close to the identity matrix.
    rng = np.random.default_rng(2)
    rows, K, N = 4, 32, 6
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_quarot_cpp(model, seed=123)
    chain = _walk_quarot_chain(quantized, "Y")
    u = numpy_helper.to_array(
        next(t for t in quantized.graph.initializer if t.name == chain["u_name"])
    ).astype(np.float64)
    np.testing.assert_allclose(u @ u.T, np.eye(K), atol=1e-4)


def test_quarot_cpp_seed_determinism():
    # Same seed -> same rotation matrix (and therefore the same quantized
    # weight); different seeds -> a genuinely different rotation. Confirms
    # the seed actually threads through QuarotSeed() into the per-node RNG
    # derivation, rather than the pass using some fixed matrix regardless.
    rng = np.random.default_rng(3)
    rows, K, N = 4, 32, 4
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    def _rotation(seed):
        quantized = onnxsim.apply_quarot_cpp(model, seed=seed)
        chain = _walk_quarot_chain(quantized, "Y")
        return numpy_helper.to_array(
            next(t for t in quantized.graph.initializer if t.name == chain["u_name"])
        )

    u_a1 = _rotation(seed=7)
    u_a2 = _rotation(seed=7)
    u_b = _rotation(seed=8)
    np.testing.assert_array_equal(u_a1, u_a2)
    assert not np.allclose(u_a1, u_b)


def test_quarot_cpp_output_within_combined_bound_and_close_to_float():
    # The full end-to-end sanity check: combines Part 1's exact rotation
    # identity with Part 2's two-operand quantized-MAC bound (both derived
    # in the module docstring) against a real onnxruntime run, mirroring
    # weight_only_quantize_int4_matmul's own analogous single-operand test.
    rng = np.random.default_rng(4)
    rows, K, N, block_size = 8, 64, 5, 32
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.6
    x = rng.standard_normal((rows, K)).astype(np.float32) * 1.5
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_quarot_cpp(model, seed=5, block_size=block_size)
    chain = _walk_quarot_chain(quantized, "Y")
    u = numpy_helper.to_array(
        next(t for t in quantized.graph.initializer if t.name == chain["u_name"])
    ).astype(np.float64)
    _wq_name, ws_name = chain["wdq_node"].input
    ws = numpy_helper.to_array(
        next(t for t in quantized.graph.initializer if t.name == ws_name)
    ).astype(np.float64)  # [K / block_size, N]

    # Xrot / Wtilde_kn: the exact (unquantized) rotated values Part 1's
    # identity is about -- Xrot @ Wtilde_kn should equal X @ W up to
    # floating-point rounding (U itself is stored as float32, so this
    # isn't bit-exact, just as close as float32 orthogonality allows),
    # independent of any INT4 quantization.
    x_rot = x.astype(np.float64) @ u
    w_tilde_kn = u.T @ weight.astype(np.float64)
    y_star = x_rot @ w_tilde_kn
    np.testing.assert_allclose(
        y_star, x.astype(np.float64) @ weight.astype(np.float64), rtol=1e-5, atol=1e-5
    )

    # Run the real quantized graph.
    sess = ort.InferenceSession(quantized.SerializeToString())
    (y_quant,) = sess.run(["Y"], {"X": x})
    y_float = x.astype(np.float64) @ weight.astype(np.float64)
    error = np.abs(y_float - y_quant)

    # Reconstruct the per-token activation scale exactly as the pass's own
    # graph computes it (epsilon default 1e-12, negligible here).
    scale_x = np.maximum(np.max(np.abs(x_rot), axis=1), 1e-12) / 7.0  # [rows]
    eps_x = scale_x / 2.0  # [rows]

    block_of_k = np.arange(K) // block_size
    eps_w = (ws / 2.0)[block_of_k, :]  # [K, N], per-tap block scale

    bound = (
        eps_x[:, None] * np.abs(w_tilde_kn).sum(axis=0)[None, :]
        + np.einsum("ik,kn->in", np.abs(x_rot), eps_w)
        + K * eps_x[:, None] * eps_w.max(axis=0)[None, :]
    )
    assert np.all(error <= bound + 1e-4)

    # And a loose but meaningful end-to-end sanity check: INT4 on BOTH
    # operands is coarser than any single-operand INT4/INT8 pass in this
    # suite, so this uses a looser tolerance than those files' own
    # (confirmed empirically -- tighter thresholds like
    # weight_only_quantize_int4_matmul's rtol=0.05 fail here).
    rel_l2 = np.linalg.norm((y_float - y_quant).ravel()) / max(
        np.linalg.norm(y_float.ravel()), 1e-6
    )
    assert rel_l2 < 0.5


def test_quarot_cpp_declines_when_k_not_divisible_by_block_size():
    # patternMatchPredicate requires K % block_size == 0 (default
    # block_size=32); a non-divisible K must survive completely untouched.
    rng = np.random.default_rng(6)
    rows, K, N = 4, 48, 4  # 48 is not a multiple of 32
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.5
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.apply_quarot_cpp(model, seed=0)
    assert quantized.SerializeToString() == model.SerializeToString()
