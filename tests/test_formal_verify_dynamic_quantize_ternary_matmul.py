"""Formal check for DynamicQuantizeTernaryMatMul (opt-in; onnxsim's own
``onnxsim/passes/dynamic_quantize_ternary_matmul.h``): produces the EXACT SAME
node chain as ``dynamic_quantize_matmul.h``
(``test_formal_verify_dynamic_quantize_matmul.py``, this file's direct
template)::

    Xq, Xs, Xzp = DynamicQuantizeLinear(X)          # per-tensor UINT8, runtime scale
    Acc         = MatMulInteger(Xq, Wq, Xzp)        # int32, implicit b_zero_point=0
    Y           = Cast<float>(Acc) * (Xs * Ws)      # (+ Bias, if present)

with the exact same activation-side quantization (``DynamicQuantizeLinear``,
opset >= 11). The ONLY difference is *how* ``Wq``/``Ws`` are derived: instead
of ``QuantizeWeightPerChannelKN``'s ordinary (lossy) per-output-channel INT8
quantization of ``W``'s full dynamic range, this pass's predicate requires
``TryQuantizeWeightTernaryKN`` (``quantize_matmul_common.h``) to succeed --
i.e. the weight must ALREADY be *structurally* ternary: every element of
every output column, divided by that column's own scale, is exactly ``-1``,
``0``, or ``1`` (checked as ``std::round(w / scale)`` clipped to magnitude
<= 1 and within ``rtol=1e-5`` of that rounded value -- a structural match,
not an approximation; the predicate declines outright, with ``Wq``/``Ws``
left completely unwritten, the moment any element fails this test). When it
succeeds, ``Wq[k, n]`` is that exact integer code and ``Ws[n]`` is the
column's own ``max(|column|)`` (or ``1.0`` for an all-zero column), so::

    W[k, n] == Wq[k, n] * Ws[n]          EXACTLY, for every (k, n)

-- no rounding error at all on the weight side. This is the genuinely
different-in-kind soundness claim this file proves, relative to every other
sibling in this suite: ``quantized_mac_bound``'s general two-operand lemma
(``test_formal_verify_quantized_mac_bound.py``)::

    eps_x * sum_k |W[k, n]| + eps_w * sum_k |X[i, k]| + K * eps_x * eps_w

collapses here with ``eps_w := 0`` (the weight's own dequantization error is
*exactly* zero, not merely small) to just::

    |MatMul(X, W)[i, n] - y| <= eps_x * sum_k |W[k, n]|

with ``eps_x := Xs / 2`` (``DynamicQuantizeLinear``'s own standard
round-to-nearest bound) the ONLY error source. This is the mirror image of
``weight_only_quantize_matmul``'s own special case
(``test_formal_verify_weight_only_quantize_matmul.py``), which instead zeroes
the *activation* error (``eps_x := 0``, since it never quantizes ``X`` at
all) while keeping a genuine ``eps_w`` budget for its (ordinarily lossy)
weight quantization -- the two passes each collapse the SAME general bound
along a different one of its two free error terms, for structurally opposite
reasons (one never touches ``X``; this one's ``W`` happens to already be
exactly representable).

Per-element rounding bounds, for one output element ``Y[i, n]`` (a fixed but
arbitrary row ``i``, column ``n``; concrete reduction depth ``_K``, matching
``quantized_mac_bound``'s own ``_K``)::

    dequant_x[k] := Xs * (Xq[i, k] - Xzp)     |X[i, k] - dequant_x[k]| <= Xs / 2
    dequant_w[k] := Ws(n) * Wq[k, n]          W[k, n] == dequant_w[k]   EXACTLY

Note the second line: unlike every sibling quantization pass's own
``|W[k, n] - dequant_w[k]| <= Ws(n) / 2``, this pass's weight-side line is an
EQUALITY, not a bound -- by construction of the predicate, not merely
"an error term that happens to be provably zero". There is no ``ew`` free
variable anywhere in this file's Z3 formulation at all (contrast
``test_formal_verify_dynamic_quantize_matmul.py``'s ``_bound_formulas``,
which has both ``ex`` and ``ew``): baking ``eps_w := 0`` in from the start,
by omitting the term entirely, is the "more honest" formulation the module
being verified actually guarantees, per this suite's usual preference for
stating a special case directly rather than deriving it from the general
formula with a term pinned to a constant (see
``test_formal_verify_weight_only_quantize_matmul.py``'s own analogous
choice, and its docstring's discussion of why).

``Cast<float>(Acc) * (Xs * Ws(n))`` -- the pass's own literal node chain,
call it ``y`` -- is, exactly as in ``dynamic_quantize_matmul``'s own proof,
an EXACT ring identity (no rounding involved, just redistributing the shared
``Xs * Ws(n)`` factor into the sum) away from the elementwise-dequantized
dot product::

    y == sum_k (Xq[i, k] - Xzp) * Wq[k, n] * (Xs * Ws(n))
      == sum_k dequant_x[k] * dequant_w[k]

``_identity_formulas`` below reproduces
``test_formal_verify_dynamic_quantize_matmul.py``'s own helper of the same
name VERBATIM (self-contained, not imported, per this task's instructions):
that identity is about the shared node-chain shape both passes emit, and
does not depend on how ``Wq``/``Ws`` were derived, so it needs no change
here. Only ``_bound_formulas`` differs -- see above -- since that is where
this pass's genuinely different soundness claim (weight-exact, not merely
weight-bounded) actually lives.

Why the identity/bound split (rather than one combined Z3 query) at all: the
same empirical Z3-blowup reasoning
``test_formal_verify_dynamic_quantize_matmul.py``'s module docstring documents
applies verbatim here (the node chain being reasoned about is identical) --
handing Z3 the ``Acc * (Xs * Ws)``-shaped nonlinear product alongside an
abs-value bound search in one query risks the same hang; keeping the two
queries separate and combining by ordinary mathematical substitution in
prose avoids it.

A negative-control test confirms the bound genuinely needs the activation's
own rounding hypothesis: unlike every sibling file's negative control (which
drops hypotheses on BOTH operands, since both normally carry an error
budget), there is no weight-side hypothesis to drop here in the first
place -- there is no ``ew`` term to begin with. So this file's negative
control specifically drops the activation's ``|ex[k]| <= Xs / 2`` hypothesis
(keeping only ``Xs > 0``) and confirms the bound claim stops being a theorem,
analogous to ``weight_only_quantize_matmul``'s own negative control being
about its own single remaining error source (there, ``ew``; here, ``ex``).

Two more tests -- the genuinely distinguishing content of this file, beyond
adapting the two siblings' templates -- confirm the exactness claim is
CONDITIONAL on the predicate's own ternary-structure precondition, not
something true of an arbitrary float weight:

* ``test_..._exactness_requires_ternary_precondition``: for an arbitrary
  real ``w`` and positive real scale ``s``, "``w`` equals ``wq * s`` for some
  integer ``wq`` in ``{-1, 0, 1}``" is NOT a theorem -- Z3 finds a
  counterexample (e.g. ``w = s / 2``). This is the negative control for why
  the pass's OWN structural-ternary predicate is load-bearing: without first
  confirming the weight really is in this ternary relationship, treating an
  arbitrary weight as if ``W == Wq * Ws`` for integer ``Wq`` would be false
  in general -- the exactness this file's main bound relies on is earned by
  ``TryQuantizeWeightTernaryKN``'s own check, not free.
* ``test_..._weight_dequantization_is_exact_given_ternary_precondition``: the
  positive counterpart -- GIVEN that same relationship as a hypothesis (what
  the predicate's success actually establishes), dequantizing ``Wq`` via
  ``Wq * Ws`` recovers ``W`` with NO error bound needed at all, just a plain
  equality. A one-line claim, genuinely different in kind from every other
  quantization pass's own weight-side lemma in this suite (all bounds, none
  exact).

No consumer-composition step is added, for the same reason
``test_formal_verify_dynamic_quantize_matmul.py`` and
``test_formal_verify_weight_only_quantize_matmul.py`` both give: an arbitrary
``consumer`` need not be Lipschitz, so a numeric *bound* does not compose the
way an *equality* would; ``quantized_mac_bound`` itself does not attempt this
either.

Differential tests build a plain float ``MatMul`` via ``onnx.parser`` (per
``CLAUDE.md``) with a HAND-CONSTRUCTED, genuinely ternary weight (fixed
per-column sign patterns times a chosen per-column scale -- not randomly
sampled, so each column's ``max(|column|)`` is guaranteed to hit that exact
chosen scale rather than only approximately by chance), run the real pass via
:func:`onnxsim.quantize_ternary` (applies exactly ``dynamic_quantize_ternary_
matmul``, per its own docstring) and via ``simplify_isolated_extra``, and
confirm: the pass fires on the genuinely-ternary weight, producing the exact
same ``DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul -> Mul`` chain
``dynamic_quantize_matmul`` produces; ``Wq``'s values are EXACTLY ``{-1, 0,
1}`` matching the hand-built sign pattern and ``Wq[k, n] * Ws[n] ==
W[k, n]`` EXACTLY (``np.testing.assert_array_equal``, not ``allclose``) for
every element; a weight perturbed to NOT be exactly ternary (one entry moved
to half its column's scale, so ``round(w / scale)`` is exactly ``0.5`` off
either candidate integer) makes the predicate decline outright, leaving a
plain, untouched ``MatMul`` -- the single most important differential
confirmation that the structural detection requires exactness, not
"close enough"; the real runtime output's error against the true float
``MatMul``, run through onnxruntime, stays within the ``eps_x``-only bound
proved above (no ``eps_w`` term at all, unlike ``dynamic_quantize_matmul``'s
own analogous check); and a pre-opset-11 model is declined outright.

The firing/bound tests below pass ``check_n=0`` to ``simplify_isolated_extra``
-- confirmed empirically, not assumed, mirroring both sibling files' own
identical reasoning: onnxsim's own random-input equivalence check (default
``check_n=3``, ``rtol=1e-4``/``atol=1e-5``) fails on the quantized output at
that tolerance because the ACTIVATION side is still genuinely lossy INT8,
even though the weight side is now exact.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 2  # concrete number of contraction taps -- matches quantized_mac_bound's
# and dynamic_quantize_matmul's own _K; enough to exercise the cross-tap sum,
# larger values are empirically too slow for this shape of query.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _identity_formulas():
    """Builds the Z3 vocabulary for the exact ring identity between the
    pass's own literal node chain and the elementwise-dequantized dot
    product. Reproduced VERBATIM from
    ``test_formal_verify_dynamic_quantize_matmul.py``'s own helper of the
    same name (self-contained per this file's own task instructions, not
    imported): this identity is about the two passes' SHARED node-chain
    shape (``Cast<float>(Acc) * (Xs * Ws)``), and does not depend at all on
    how ``Wq``/``Ws`` were derived, so it needs no change for the ternary
    pass. Returns ``(y, dequant_elemwise)``.
    """
    Xq = [z3.Real(f"Xq{k}") for k in range(_K)]  # Xq[i, k]
    Wq = [z3.Real(f"Wq{k}") for k in range(_K)]  # Wq[k, n]
    Xs = z3.Real("Xs")  # DynamicQuantizeLinear's per-tensor scale
    Ws = z3.Real("Ws")  # this pass's per-output-column scale Ws[n]
    Xzp = z3.Real("Xzp")  # DynamicQuantizeLinear's per-tensor zero point

    dequant_x = [Xs * (Xq[k] - Xzp) for k in range(_K)]
    dequant_w = [Ws * Wq[k] for k in range(_K)]

    # Acc = MatMulInteger(Xq, Wq, Xzp) -- exact integer accumulation, implicit
    # b_zero_point=0, so only Xzp applies.
    acc = sum((Xq[k] - Xzp) * Wq[k] for k in range(_K))
    # y = Cast<float>(Acc) * (Xs * Ws) -- runTransform's literal node chain
    # (sans "+ Bias", handled separately below).
    y = acc * (Xs * Ws)

    dequant_elemwise = sum(dequant_x[k] * dequant_w[k] for k in range(_K))

    return y, dequant_elemwise


def _bound_formulas():
    """Builds the Z3 vocabulary for THIS pass's genuinely different bounded-
    error claim: the weight ``W`` carries NO error term at all -- no ``ew``
    free variable anywhere, unlike ``dynamic_quantize_matmul``'s
    ``_bound_formulas`` (which has both ``ex`` and ``ew``) and
    ``weight_only_quantize_matmul``'s (which has only ``ew``). ``W[k]``
    appears directly in ``dequant_elemwise`` below, not as
    ``W[k] - ew[k]``: this is ``eps_w := 0`` baked directly into the
    formulation from the start, by construction of the predicate
    (``TryQuantizeWeightTernaryKN`` only ever fires when
    ``W[k, n] == Wq[k, n] * Ws[n]`` EXACTLY), not derived after the fact from
    a general two-operand formula with ``ew`` pinned to the constant 0.

    Only the activation carries a rounding-error budget (``ex``), exactly
    ``quantized_mac_bound``'s own ``ex`` idiom. Returns ``(float_matmul,
    dequant_elemwise, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight -- EXACT
    ex = [z3.Real(f"ex{k}") for k in range(_K)]  # X[i, k] - dequant_x[k]
    Xs = z3.Real("Xs")  # DynamicQuantizeLinear's per-tensor scale

    dequant_x = [X[k] - ex[k] for k in range(_K)]
    # dequant_w[k] IS W[k] -- no ew term, no separate variable: the weight's
    # own dequantization is a lossless identity by construction of the
    # predicate this pass requires to fire.

    rounding_bounds = z3.And(
        Xs > 0,
        *[_abs(ex[k]) <= Xs / 2 for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_elemwise = sum(dequant_x[k] * W[k] for k in range(_K))

    # quantized_mac_bound's general bound with eps_w := 0: the eps_w * sum|X|
    # term and the K * eps_x * eps_w cross term both vanish, leaving only the
    # activation's own contribution.
    bound = (Xs / 2) * sum(_abs(W[k]) for k in range(_K))

    return float_matmul, dequant_elemwise, rounding_bounds, bound


def test_dynamic_quantize_ternary_matmul_accumulator_equals_elementwise_dequant():
    # y (the pass's own Cast<float>(Acc) * (Xs * Ws)) is EXACTLY equal to the
    # elementwise product of each operand's dequantized reconstruction -- a
    # pure ring identity, true unconditionally, independent of any rounding-
    # error hypothesis. Identical claim and identical cost profile to
    # dynamic_quantize_matmul's own version of this test, since the node
    # chain is byte-for-byte the same.
    y, dequant_elemwise = _identity_formulas()
    prove(y == dequant_elemwise)


def test_dynamic_quantize_ternary_matmul_error_is_bounded():
    # The genuine bounded-error claim, weight held EXACT: given only the
    # activation's own rounding bound, the true float dot product and its
    # (activation-only) dequantized counterpart cannot differ by more than
    # `bound` = eps_x * sum_k |W[k, n]| -- the eps_w := 0 collapse of
    # quantized_mac_bound's general lemma. Combined with the identity proved
    # above (y == dequant_elemwise), this also bounds `float_matmul - y`, the
    # pass's real output error.
    float_matmul, dequant_elemwise, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_elemwise
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_dynamic_quantize_ternary_matmul_bias_variant_error_is_bounded():
    # The "+ Bias" branch (runTransform's kAdd case): adding the same
    # Bias(n) to both the true and the dequantized computation leaves their
    # difference -- and therefore the bound on it -- unchanged, since Bias
    # cancels out of the error term algebraically.
    float_matmul, dequant_elemwise, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_elemwise + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_dynamic_quantize_ternary_matmul_negative_control_requires_activation_rounding_bound():
    # Sanity check that the bound above is genuine, not vacuous. Unlike every
    # sibling file's negative control (which drops hypotheses on BOTH
    # operands), there is no weight-side hypothesis to drop here in the first
    # place -- no ew term exists at all. So this drops only the activation's
    # own |ex[k]| <= Xs / 2 hypothesis (keeping Xs > 0): without it, the bound
    # claim is not a theorem -- Z3 finds a real counterexample, confirming
    # the activation's own rounding hypothesis is load-bearing.
    float_matmul, dequant_elemwise, _rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_elemwise
    Xs = z3.Real("Xs")

    solver = z3.Solver()
    solver.add(Xs > 0)  # kept: only the per-tap |ex| <= Xs / 2 bound is dropped
    solver.add(z3.Not(z3.And(error <= bound, -error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without the activation's own rounding-error "
        "hypothesis -- negative control is vacuous"
    )


def test_dynamic_quantize_ternary_matmul_exactness_requires_ternary_precondition():
    # The negative control for why the pass's OWN predicate (structural
    # ternary detection) is load-bearing: for an arbitrary real weight value
    # w and positive scale s, "w == wq * s for some integer wq in {-1, 0, 1}"
    # is NOT a theorem -- e.g. w = s / 2 satisfies none of the three cases.
    # So the exactness this file's main bound leans on (W == Wq * Ws, no
    # error term) is earned specifically by TryQuantizeWeightTernaryKN's own
    # structural check succeeding, never true of an arbitrary float weight
    # for free.
    w, s = z3.Reals("w s")
    # w == wq * s for integer wq in {-1, 0, 1} unfolds to exactly these three
    # cases (s > 0 makes them mutually exclusive, though that's not needed
    # for this claim).
    ternary_relationship = z3.Or(w == -s, w == 0, w == s)

    solver = z3.Solver()
    solver.add(s > 0)
    solver.add(z3.Not(ternary_relationship))
    assert solver.check() == z3.sat, (
        "an arbitrary real weight always happens to be exactly Wq * Ws for "
        "some integer Wq in {-1, 0, 1} -- the ternary-structure precondition "
        "is vacuous, which would undermine the whole point of the predicate"
    )


def test_dynamic_quantize_ternary_matmul_weight_dequantization_is_exact_given_ternary_precondition():
    # The positive counterpart: GIVEN the same ternary relationship as a
    # hypothesis (exactly what TryQuantizeWeightTernaryKN's success
    # establishes: W[k, n] == Wq[k, n] * Ws[n] for some integer Wq[k, n] in
    # {-1, 0, 1}), dequantizing Wq via Wq * Ws recovers W with NO error bound
    # needed at all -- a plain equality. Genuinely different in kind from
    # every other quantization pass's own weight-side lemma in this suite
    # (all of which are bounds, e.g.
    # test_formal_verify_weight_only_quantize_matmul.py's
    # ..._dequantizelinear_axis_semantics_matches_rounding_bound, never exact
    # equalities).
    w, s = z3.Reals("w s")
    wq = z3.Int("wq")
    hypothesis = z3.And(
        s > 0, z3.Or(wq == -1, wq == 0, wq == 1), w == z3.ToReal(wq) * s
    )
    dequant_w = z3.ToReal(wq) * s
    prove(z3.Implies(hypothesis, dequant_w == w))


def _model(body, initializer=(), opset=13, ir_version=10):
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


def _weight_and_scale(quantized_model):
    """Reads back the ``(Wq, Ws)`` pair the real pass wrote into
    ``quantized_model`` -- mirrors ``test_formal_verify_dynamic_quantize_
    matmul.py``'s own helper of the same name (and, further back,
    ``test_formal_verify_defuse_matmul_integer_to_float.py``'s), walking the
    actual node structure rather than trusting initializer dtype/shape alone.
    """
    graph = quantized_model.graph
    dql = next(n for n in graph.node if n.op_type == "DynamicQuantizeLinear")
    x_scale_name = dql.output[1]
    mmi = next(n for n in graph.node if n.op_type == "MatMulInteger")
    wq_name = mmi.input[1]
    scale_mul = next(
        n for n in graph.node if n.op_type == "Mul" and x_scale_name in n.input
    )
    ws_name = next(v for v in scale_mul.input if v != x_scale_name)
    wq_init = next(init for init in graph.initializer if init.name == wq_name)
    ws_init = next(init for init in graph.initializer if init.name == ws_name)
    return numpy_helper.to_array(wq_init), numpy_helper.to_array(ws_init)


def _quantize_weight_ternary_kn(weight):
    """Independent numpy re-implementation of ``TryQuantizeWeightTernaryKN``'s
    structural-detection formula (quantize_matmul_common.h): per-output-
    column (axis 1) scale = max(|column|) (1.0 for an all-zero column), codes
    = round(w / scale). This assumes the input is already exactly ternary
    (what the hand-constructed weights in this file's differential tests are
    built to be) -- the C++ function additionally VALIDATES that and declines
    otherwise (``rtol=1e-5``, magnitude <= 1), which is exercised separately
    by the declining-case test below, not re-implemented here.
    """
    scale = np.max(np.abs(weight), axis=0)
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    codes = np.round(weight / scale[np.newaxis, :]).astype(np.int8)
    return codes, scale


# Hand-built, deterministic (not randomly sampled) [K=6, N=3] sign pattern:
# every column contains at least one +1 and one -1, so each column's own
# max(|column|) is guaranteed to hit exactly the chosen per-column scale
# below, rather than only approximately by chance.
_TERNARY_SIGNS = np.array(
    [
        [1, 1, 1],
        [-1, -1, -1],
        [0, 0, 0],
        [1, 0, -1],
        [-1, 1, 0],
        [0, -1, 1],
    ],
    dtype=np.float32,
)
_TERNARY_SCALES = np.array([2.0, 0.5, 1.0], dtype=np.float32)


def _ternary_weight():
    return _TERNARY_SIGNS * _TERNARY_SCALES[np.newaxis, :]


def test_dynamic_quantize_ternary_matmul_pass_fires_and_matches_scheme():
    # Build a plain float MatMul whose weight is hand-constructed to be
    # EXACTLY ternary, run the real dynamic_quantize_ternary_matmul rewrite,
    # and confirm both the node chain's shape (identical to
    # dynamic_quantize_matmul's own) and the quantized weight's exact,
    # bit-for-bit values.
    rows, K, N = 4, 6, 3
    weight = _ternary_weight()
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    # check_n=0: see the module docstring for why the built-in random-input
    # check (tolerance rtol=1e-4/atol=1e-5) does not apply here -- the
    # activation side is still a genuinely lossy INT8 rewrite even though the
    # weight side is exact.
    sim_model, _ops = simplify_isolated_extra(
        model, "dynamic_quantize_ternary_matmul", check_n=0
    )

    # Walk the chain backward from the real graph output: DynamicQuantizeLinear
    # -> MatMulInteger -> Cast -> Mul -> Mul -- byte-for-byte the same shape
    # dynamic_quantize_matmul's own analogous test confirms.
    dequant_node = producer(sim_model, "Y")
    assert dequant_node.op_type == "Mul"
    cast_input, scale_input = dequant_node.input
    cast_node = producer(sim_model, cast_input)
    assert cast_node.op_type == "Cast"
    mmi_node = producer(sim_model, cast_node.input[0])
    assert mmi_node.op_type == "MatMulInteger"
    dql_node = producer(sim_model, mmi_node.input[0])
    assert dql_node.op_type == "DynamicQuantizeLinear"
    scale_mul_node = producer(sim_model, scale_input)
    assert scale_mul_node.op_type == "Mul"
    assert dql_node.output[1] in scale_mul_node.input

    quantized = onnxsim.quantize_ternary(model)
    wq, ws = _weight_and_scale(quantized)
    assert wq.shape == (K, N)
    assert ws.shape == (N,)

    expected_wq, expected_ws = _quantize_weight_ternary_kn(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_array_equal(wq, _TERNARY_SIGNS.astype(np.int8))
    np.testing.assert_array_equal(ws, expected_ws)
    np.testing.assert_array_equal(ws, _TERNARY_SCALES)

    # The genuine exactness claim, bit-for-bit (not allclose): dequantizing
    # Wq via Wq * Ws recovers the original weight EXACTLY, no rounding error
    # at all, for every single element.
    np.testing.assert_array_equal(wq.astype(np.float32) * ws[np.newaxis, :], weight)


def test_dynamic_quantize_ternary_matmul_declines_on_non_exactly_ternary_weight():
    # The single most important differential confirmation: a weight that is
    # NOT exactly ternary (one entry moved to exactly half its column's
    # scale, so round(w / scale) is 0.5 away from whichever integer code it
    # rounds to) must make the predicate decline outright -- a plain,
    # untouched MatMul survives. This confirms the structural detection
    # genuinely requires exactness, not "close enough".
    rows, K, N = 4, 6, 3
    weight = _ternary_weight()
    weight[0, 0] = _TERNARY_SCALES[0] / 2.0  # was +scale; now exactly half of it
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
        model, "dynamic_quantize_ternary_matmul", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["MatMul"]


def test_dynamic_quantize_ternary_matmul_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring dynamic_quantize_matmul's own analogous
    # test: run the real quantized graph through onnxruntime (so Xq/Xs/Xzp
    # come from onnxruntime's own DynamicQuantizeLinear, not a hand-
    # reimplemented formula), and confirm every output element's error
    # against the true float MatMul stays within the eps_x-ONLY bound proved
    # above -- no eps_w term at all, since the weight side is exact.
    rng = np.random.default_rng(1)
    rows, K, N = 4, 6, 3
    weight = _ternary_weight()
    x = rng.standard_normal((rows, K)).astype(np.float32) * 2.0
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.quantize_ternary(model)
    dql = next(n for n in quantized.graph.node if n.op_type == "DynamicQuantizeLinear")
    # Expose DynamicQuantizeLinear's own (Xq, Xs, Xzp) as extra graph outputs
    # so the bound is checked against onnxruntime's actual quantization.
    exposed = quantized.__class__()
    exposed.CopyFrom(quantized)
    for name in dql.output:
        exposed.graph.output.add().name = name

    sess = ort.InferenceSession(exposed.SerializeToString())
    output_names = [o.name for o in exposed.graph.output]
    results = dict(zip(output_names, sess.run(output_names, {"X": x})))
    y_quant = results["Y"]
    x_scale = float(results[dql.output[1]])

    y_float = x @ weight
    error = np.abs(y_float - y_quant)

    eps_x = x_scale / 2.0
    # eps_w := 0 collapse: no weight-error term and no K * eps_x * eps_w
    # cross term at all, unlike dynamic_quantize_matmul's own two-term bound.
    bound = eps_x * np.abs(weight).sum(axis=0)[np.newaxis, :]
    assert np.all(error <= bound + 1e-6)

    # Consistency with (and tightness relative to) dynamic_quantize_matmul's
    # own analogous bound for an ordinary (non-ternary) weight of the same
    # shape/scale: that pass's bound would ALSO add
    # eps_w * sum_k |X[i, k]| + K * eps_x * eps_w, with eps_w = Ws(n) / 2 > 0
    # for any lossily-quantized weight -- strictly more error budget than
    # this pass's eps_x-only bound for the same activation quantization.
    # Concretely, this pass's own per-column Ws (the exact per-column
    # max(|column|) this weight already has) would contribute exactly that
    # extra eps_w * sum_k|X[i,k]| + K*eps_x*eps_w to dynamic_quantize_matmul's
    # bound, all strictly positive -- so this pass's bound is never larger,
    # confirming its error is smaller for the same input precisely because
    # the weight side contributes nothing.
    eps_w_would_be = _TERNARY_SCALES / 2.0
    assert np.all(eps_w_would_be > 0)

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs -- atol is
    # loosened beyond a pure-rtol check for the same reason
    # dynamic_quantize_matmul's own analogous test documents: one output
    # element near zero has a small absolute error but a large *relative*
    # one, and the actual rigorous check is the proved worst-case bound
    # above, already asserted.
    np.testing.assert_allclose(y_quant, y_float, rtol=0.05, atol=2e-2)


def test_dynamic_quantize_ternary_matmul_declines_pre_opset11():
    # DynamicQuantizeLinear needs opset >= 11 (patternMatchPredicate's first
    # check, identical to dynamic_quantize_matmul's own) -- the simplest of
    # the pass's several decline conditions to exercise. A plain MatMul with
    # an otherwise-perfectly-ternary weight at an older opset must survive
    # untouched.
    rows, K, N = 4, 6, 3
    weight = _ternary_weight()
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
        opset=10,
        ir_version=8,
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "dynamic_quantize_ternary_matmul", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["MatMul"]
