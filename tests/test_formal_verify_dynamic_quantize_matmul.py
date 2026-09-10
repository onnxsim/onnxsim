"""Formal check for DynamicQuantizeMatMul (opt-in; onnxsim's own
``onnxsim/passes/dynamic_quantize_matmul.h``): the exact inverse of the
rewrite ``defuse_matmul_integer_to_float.h`` folds back
(``test_formal_verify_defuse_matmul_integer_to_float.py`` is this file's
direct template for the Z3 vocabulary). It rewrites ``Y = MatMul(X, W)`` (or
a "vanilla" ``Gemm``, ``+ Bias`` optional) -- ``W`` a constant 2-D FLOAT32
tensor, ``X`` FLOAT32 -- into::

    Xq, Xs, Xzp = DynamicQuantizeLinear(X)          # per-tensor UINT8, runtime scale
    Acc         = MatMulInteger(Xq, Wq, Xzp)        # int32, implicit b_zero_point=0
    Y           = Cast<float>(Acc) * (Xs * Ws)      # (+ Bias, if present)

``Wq`` (INT8) and ``Ws`` (FLOAT32, one scale per output column) are computed
ONCE at pass-transform time from ``W``'s static values, ordinary
per-output-channel symmetric quantization (``QuantizeWeightPerChannelKN``);
``Xq``/``Xs``/``Xzp`` are computed at *run time* by ``DynamicQuantizeLinear``
from each run's actual activation range.

Unlike ``defuse_matmul_integer_to_float`` -- which folds an *already*
quantized chain back into a plain ``MatMul`` and is an EXACT equivalence
given ``DynamicQuantizeLinear``'s own defining dequantization identity --
this pass runs in the other direction, from an exact float computation to a
quantized approximation of it, and is genuinely LOSSY: both the dynamic
UINT8 quantization of ``X`` and the static INT8 quantization of ``W``
introduce their own rounding error. The claim this file proves is therefore
the bounded-error shape ``test_formal_verify_quantized_mac_bound.py``
establishes for exactly this kind of thing, not an exact-equivalence claim:
the rewritten output differs from the true float ``MatMul(X, W)`` by an
amount boundable in terms of the two quantization steps' own per-element
rounding bounds, propagated through the reduction the way a dot product's
own linearity plus the triangle inequality allows
(``quantized_mac_bound``'s own derivation, reused here almost verbatim --
see below for why "almost").

Per-element rounding bounds (standard round-to-nearest quantize/dequantize
bound, ``|value - dequant(quant(value))| <= scale / 2``), for one output
element ``Y[i, n]`` (a fixed but arbitrary row ``i``, column ``n``; the
concrete reduction depth is ``_K``, matching ``quantized_mac_bound``'s own
``_K``, see below)::

    dequant_x[k] := Xs * (Xq[i, k] - Xzp)     |X[i, k]  - dequant_x[k]| <= Xs / 2
    dequant_w[k] := Ws(n) * Wq[k, n]          |W[k, n]  - dequant_w[k]| <= Ws(n) / 2

``Cast<float>(Acc)[i, n] * (Xs * Ws(n))`` -- the pass's own literal node
chain, call it ``y`` -- is then an EXACT ring identity (no rounding involved,
just redistributing the shared ``Xs * Ws(n)`` factor into the sum) away from
the elementwise-dequantized dot product::

    y == sum_k (Xq[i, k] - Xzp) * Wq[k, n] * (Xs * Ws(n))
      == sum_k dequant_x[k] * dequant_w[k]

so bounding ``MatMul(X, W)[i, n] - sum_k dequant_x[k] * dequant_w[k]`` bounds
``MatMul(X, W)[i, n] - y`` too. That second bound is EXACTLY
``quantized_mac_bound``'s own lemma, with ``eps_w := Ws(n) / 2`` and
``eps_x := Xs / 2`` substituted for its free ``eps_w``/``eps_x``::

    |MatMul(X, W)[i, n] - y| <=
        (Xs / 2) * sum_k |W[k, n]| + (Ws(n) / 2) * sum_k |X[i, k]|
        + _K * (Xs / 2) * (Ws(n) / 2)

Why two separate lemmas (an exact identity plus a reused bound) instead of
one direct Z3 proof of the combined claim, the way
``defuse_matmul_integer_to_float``'s single-shot proof works: empirically,
handing Z3 the combined nonlinear statement in one go -- whether by literally
writing ``y``'s ``Acc * (Xs * Ws)`` form (a Mul of a sum-of-products by
another product) into the same formula as the abs-value bound, or even by
adding the identity above as an extra hypothesis to the very same implication
-- makes its nonlinear-arithmetic search hang well past a minute at ``_K`` as
small as 2 (confirmed directly: over 90 seconds, versus a few milliseconds
for the identity alone). This suite has hit exactly this kind of Z3 blowup
before (see ``fuse_consecutive_log_softmax``'s original ``ForAll`` hang,
fixed by ground instantiation) -- here the fix is the same idea one level up:
keep each individual Z3 query in the tractable shape an already-working proof
in this suite uses, and combine the two results by ordinary mathematical
substitution in the surrounding Python/prose rather than asking Z3 to search
the combined nonlinear formula itself.

The two lemmas below don't just split the same formulas into two `prove()`
calls, either -- the bound lemma (``_bound_formulas``) is deliberately
reformulated to drop ``Xq``/``Wq``/``Xzp``/``Xs``/``Ws`` products entirely in
favor of *direct* per-tap error variables (``ex``/``ew``, each a free Real
bounded straight by the rounding hypothesis, exactly
``quantized_mac_bound``'s own ``ew``/``ex`` idiom), rather than re-deriving
each tap's dequantized value from separate quantized-code/scale/zero-point
multiplicands the way the identity lemma (``_identity_formulas``) does. Both
formulations describe the same ``dequant_elemwise`` quantity -- the identity
lemma is what connects it back to the pass's own literal node chain -- but
handing Z3 the *extra* ``Xs*(Xq-Xzp)``-shaped multiplications inside the same
query as the abs-value bound search is exactly what caused the hang above;
dropping them from the bound query specifically (while keeping them where
they're actually needed, in the identity query) brings its cost back down to
about ``quantized_mac_bound``'s own ~20s at the same ``_K``. ``_K = 2`` (not
the ``e.g. 3`` a first cut might reach for) is chosen for the same reason
``quantized_mac_bound`` itself uses ``_K = 2`` -- it is already enough to
exercise the cross-tap sum, and larger ``_K`` was empirically too slow for
this shape of query.

``Xq``/``Wq`` (in the identity lemma) are modeled as plain Z3 Reals, not the
Int-sorted functions ``defuse_matmul_integer_to_float``'s proof uses for the
analogous quantized codes: this proof's argument only ever uses the
rounding-error bound each code satisfies, never that it is integer-valued,
and empirically even reintroducing Int sorts and per-(i, k)/(k, n)
uninterpreted functions (rather than plain per-tap variables for one fixed,
arbitrary (i, n)) reopens the same kind of Z3 blowup. Nothing about the
soundness argument depends on ranging over an unbounded ``Function``: the
claim is already about one representative output element, exactly as
``quantized_mac_bound``'s own dot-product lemma is.

A negative-control test confirms the rounding-error hypotheses are load-
bearing: with no error budget assumed at all, the exact-equality claim
``MatMul(X, W)[i, n] == y`` is not a theorem -- Z3 finds a real
counterexample -- so the bound above is not vacuously satisfied by some
accidentally-always-equal pair of expressions.

No consumer-composition step (this suite's usual substitution-safety idiom,
e.g. ``test_formal_verify_eliminate_identity.py``) is added here: composing
an arbitrary ``consumer`` with an *equality* is immediate
(``consumer(a) == consumer(b)`` follows from ``a == b`` for any function
``consumer``), but no such general principle holds for a numeric *bound* --
an arbitrary ``consumer`` need not be Lipschitz, let alone 1-Lipschitz, so
"``|a - b| <= bound``" does not in general imply anything about
"``|consumer(a) - consumer(b)|``" at all. ``quantized_mac_bound`` itself,
this file's template for this shape of claim, does not attempt such a
composition either.

Differential tests build a plain float ``MatMul`` via ``onnx.parser``, run
the real pass through :func:`onnxsim.quantize_dynamic` (applying
``dynamic_quantize_matmul`` alone, per that function's own docstring), and
confirm: the exact ``DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul ->
Mul`` chain fires; the ``Wq``/``Ws`` initializers match an independent numpy
re-implementation of ``QuantizeWeightPerChannelKN``'s formula; the actual
runtime output error (against onnxruntime's own ``DynamicQuantizeLinear``
output, not a hand-reimplemented one) stays within the bound proved above,
mirroring ``quantized_mac_bound``'s own
``test_quantized_mac_bound_matches_onnxruntime``; and a pre-opset-11 model
(``DynamicQuantizeLinear`` needs opset >= 11) is declined outright, the
simplest of the pass's several decline conditions to exercise.

The firing tests below pass ``check_n=0`` to ``simplify_isolated_extra`` --
confirmed empirically, not assumed -- since this really is a lossy INT8
rewrite: onnxsim's own random-input equivalence check (its default
``check_n=3``, tolerance ``rtol=1e-4``/``atol=1e-5``) fails on the quantized
output at that tolerance, the same reasoning
``test_formal_verify_defuse_matmul_integer_to_float.py`` documents for its
own ``check_n=0`` (there, for the reverse direction).
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 2  # concrete number of contraction taps -- see module docstring for why
# this matches quantized_mac_bound's own _K rather than a larger value.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _identity_formulas():
    """Builds the Z3 vocabulary for the exact ring identity between the
    pass's own literal node chain and the elementwise-dequantized dot
    product -- needs ``Xq``/``Wq``/``Xzp`` explicitly, since it's a claim
    about that specific node chain's shape. Returns ``(y,
    dequant_elemwise)``.
    """
    Xq = [z3.Real(f"Xq{k}") for k in range(_K)]  # Xq[i, k]
    Wq = [z3.Real(f"Wq{k}") for k in range(_K)]  # Wq[k, n]
    Xs = z3.Real("Xs")  # DynamicQuantizeLinear's per-tensor scale
    Ws = z3.Real("Ws")  # this pass's per-output-column scale Ws[n]
    Xzp = z3.Real("Xzp")  # DynamicQuantizeLinear's per-tensor zero point

    dequant_x = [Xs * (Xq[k] - Xzp) for k in range(_K)]
    dequant_w = [Ws * Wq[k] for k in range(_K)]

    # Acc = MatMulInteger(Xq, Wq, Xzp) -- exact integer accumulation, implicit
    # b_zero_point=0, so only Xzp applies (mirrors defuse_matmul_integer_to_
    # float's own `acc`).
    acc = sum((Xq[k] - Xzp) * Wq[k] for k in range(_K))
    # y = Cast<float>(Acc) * (Xs * Ws) -- runTransform's literal node chain
    # (sans "+ Bias", handled separately below).
    y = acc * (Xs * Ws)

    dequant_elemwise = sum(dequant_x[k] * dequant_w[k] for k in range(_K))

    return y, dequant_elemwise


def _bound_formulas():
    """Builds the Z3 vocabulary for the bounded-error claim, mirroring
    ``test_formal_verify_quantized_mac_bound.py``'s own formulation exactly:
    each operand's dequantization ERROR (``ex``/``ew``) is a free Real
    variable bounded directly by the rounding hypothesis, rather than being
    re-derived from separate quantized-code/scale/zero-point multiplicands
    (``Xq``/``Wq``/``Xzp``/``Xs``/``Ws``) the way ``_identity_formulas``
    above does. This is mathematically the same ``dequant_elemwise``
    quantity either way (the identity test above already connects it to the
    pass's own literal node chain) -- but empirically, handing Z3 the extra
    ``Xs*(Xq-Xzp)``-shaped multiplications alongside the abs-value bound
    search makes this specific query hang well past a minute at ``_K`` as
    small as 2 (confirmed directly), whereas this direct-error formulation
    -- with no more nonlinear structure than ``quantized_mac_bound``'s own,
    already-working proof -- costs about the same ~20s. See the module
    docstring for the full explanation.

    Returns ``(float_matmul, dequant_elemwise, rounding_bounds, bound)``.
    """
    X = [z3.Real(f"X{k}") for k in range(_K)]  # X[i, k], true float activation
    W = [z3.Real(f"W{k}") for k in range(_K)]  # W[k, n], true float weight
    ex = [z3.Real(f"ex{k}") for k in range(_K)]  # X[i, k] - dequant_x[k]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]  # W[k, n] - dequant_w[k]
    Xs = z3.Real("Xs")  # DynamicQuantizeLinear's per-tensor scale
    Ws = z3.Real("Ws")  # this pass's per-output-column scale Ws[n]

    dequant_x = [X[k] - ex[k] for k in range(_K)]
    dequant_w = [W[k] - ew[k] for k in range(_K)]

    rounding_bounds = z3.And(
        Xs > 0,
        Ws > 0,
        *[_abs(ex[k]) <= Xs / 2 for k in range(_K)],
        *[_abs(ew[k]) <= Ws / 2 for k in range(_K)],
    )

    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_elemwise = sum(dequant_x[k] * dequant_w[k] for k in range(_K))

    bound = (
        (Xs / 2) * sum(_abs(W[k]) for k in range(_K))
        + (Ws / 2) * sum(_abs(X[k]) for k in range(_K))
        + _K * (Xs / 2) * (Ws / 2)
    )

    return float_matmul, dequant_elemwise, rounding_bounds, bound


def test_dynamic_quantize_matmul_accumulator_equals_elementwise_dequant():
    # y (the pass's own Cast<float>(Acc) * (Xs * Ws)) is EXACTLY equal to the
    # elementwise product of each operand's dequantized reconstruction -- a
    # pure ring identity (distributing the shared Xs * Ws factor into the
    # sum), true unconditionally, independent of any rounding-error
    # hypothesis. No abs/If branching is involved, so Z3 checks this in
    # milliseconds regardless of the nonlinear-arithmetic cost the bound
    # proof below has.
    y, dequant_elemwise = _identity_formulas()
    prove(y == dequant_elemwise)


def test_dynamic_quantize_matmul_error_is_bounded():
    # The genuine bounded-error claim: given each operand's own rounding
    # bound, the true float dot product and its elementwise-dequantized
    # counterpart cannot differ by more than `bound`. Combined with the
    # identity proved above (y == dequant_elemwise, connecting this same
    # dequant_elemwise quantity to the pass's actual node chain), this also
    # bounds `float_matmul - y` -- the pass's real output error -- by the
    # same amount; see the module docstring for why that combination is done
    # here in prose/by substitution rather than as a single Z3 query.
    float_matmul, dequant_elemwise, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_elemwise
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_dynamic_quantize_matmul_bias_variant_error_is_bounded():
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


def test_dynamic_quantize_matmul_negative_control_requires_rounding_bounds():
    # Sanity check that the bound proved above is genuine, not vacuous:
    # without ANY error budget (no rounding-error hypothesis at all), the
    # exact-equality claim float_matmul == dequant_elemwise is not a
    # theorem -- Z3 must find a real counterexample, confirming the
    # rounding really does introduce error rather than the two computations
    # always coinciding regardless. (Combined with the identity test above,
    # float_matmul == dequant_elemwise is equivalent to float_matmul == y.)
    X = [z3.Real(f"X{k}") for k in range(_K)]
    W = [z3.Real(f"W{k}") for k in range(_K)]
    ex = [z3.Real(f"ex{k}") for k in range(_K)]
    ew = [z3.Real(f"ew{k}") for k in range(_K)]
    dequant_x = [X[k] - ex[k] for k in range(_K)]
    dequant_w = [W[k] - ew[k] for k in range(_K)]
    float_matmul = sum(X[k] * W[k] for k in range(_K))
    dequant_elemwise = sum(dequant_x[k] * dequant_w[k] for k in range(_K))

    solver = z3.Solver()
    solver.add(z3.Not(float_matmul == dequant_elemwise))
    assert solver.check() == z3.sat, (
        "the float and dequantized computations always agree even without "
        "any rounding-error budget -- negative control is vacuous"
    )


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
    ``quantized_model`` -- mirrors
    ``test_formal_verify_defuse_matmul_integer_to_float.py``'s own helper of
    the same name, walking the actual node structure rather than trusting
    initializer dtype/shape alone.
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


def _quantize_weight_per_channel_kn(weight):
    """Independent numpy re-implementation of
    ``QuantizeWeightPerChannelKN`` (quantize_matmul_common.h): per-output-
    column (axis 1) symmetric INT8 quantization, scale = max(|column|) / 127
    (or 1.0 for an all-zero column), codes = round(w / scale) clipped to
    [-127, 127].
    """
    scale = np.max(np.abs(weight), axis=0)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    codes = np.clip(np.round(weight / scale[np.newaxis, :]), -127, 127).astype(np.int8)
    return codes, scale


def test_dynamic_quantize_matmul_pass_fires_and_matches_scheme():
    # Build a plain float MatMul and run the real dynamic_quantize_matmul
    # rewrite (via onnxsim.quantize_dynamic, which applies exactly this one
    # rewrite -- see its own docstring), then confirm both the node chain's
    # shape and the quantized weight's exact values.
    rng = np.random.default_rng(0)
    rows, K, N = 4, 5, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [_f32(weight, "W")],
    )

    quantized = onnxsim.quantize_dynamic(model)

    # check_n=0: see the module docstring for why the built-in random-input
    # check (tolerance rtol=1e-4/atol=1e-5) does not apply to a genuinely
    # lossy INT8 rewrite.
    sim_model, _ops = simplify_isolated_extra(
        model, "dynamic_quantize_matmul", check_n=0
    )

    # Walk the chain backward from the real graph output: DynamicQuantizeLinear
    # -> MatMulInteger -> Cast -> Mul -> Mul.
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

    wq, ws = _weight_and_scale(quantized)
    assert wq.shape == (K, N)
    assert ws.shape == (N,)
    expected_wq, expected_ws = _quantize_weight_per_channel_kn(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)


def test_dynamic_quantize_matmul_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring quantized_mac_bound's own
    # test_quantized_mac_bound_matches_onnxruntime: run the real quantized
    # graph through onnxruntime (so Xq/Xs/Xzp come from onnxruntime's own
    # DynamicQuantizeLinear, not a hand-reimplemented formula), and confirm
    # every output element's error against the true float MatMul stays
    # within the bound the proof above derives, with Xs/Ws(n) taken from the
    # actual run rather than assumed.
    rng = np.random.default_rng(1)
    rows, K, N = 4, 6, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.8
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

    quantized = onnxsim.quantize_dynamic(model)
    dql = next(n for n in quantized.graph.node if n.op_type == "DynamicQuantizeLinear")
    # Expose DynamicQuantizeLinear's own (Xq, Xs, Xzp) as extra graph outputs
    # so the bound is checked against onnxruntime's actual quantization, not
    # a reimplementation of DynamicQuantizeLinear's formula.
    exposed = quantized.__class__()
    exposed.CopyFrom(quantized)
    for name in dql.output:
        exposed.graph.output.add().name = name

    # Graph optimization disabled: by default onnxruntime silently fuses this
    # QDQ chain (DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul) into a
    # hardware-specific fused kernel, a different code path than the literal
    # node chain this pass's proof reasons about -- see
    # `tests/test_ort_matmul_nbits_workaround.py`'s docstring for this
    # suite's existing precedent of a real ORT graph-optimization fusion bug
    # of exactly this shape. Disabling optimization executes the graph
    # exactly as the pass produced it.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        exposed.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"]
    )
    output_names = [o.name for o in exposed.graph.output]
    results = dict(zip(output_names, sess.run(output_names, {"X": x})))
    y_quant = results["Y"]
    x_scale = float(results[dql.output[1]])

    _wq, ws = _weight_and_scale(quantized)

    y_float = x @ weight
    error = np.abs(y_float - y_quant)

    eps_x = x_scale / 2.0
    eps_w = ws / 2.0  # shape [N]
    bound = (
        eps_x * np.abs(weight).sum(axis=0)[np.newaxis, :]
        + eps_w[np.newaxis, :] * np.abs(x).sum(axis=1)[:, np.newaxis]
        + K * eps_x * eps_w[np.newaxis, :]
    )
    assert np.all(error <= bound + 1e-6)

    # Also confirm the rewrite is meaningfully close (not merely "within a
    # loose worst-case bound") for these well-scaled inputs -- a few percent
    # relative error, consistent with INT8 quantization, not bitwise equal.
    # atol is loosened beyond the rtol-only bound one output element near
    # zero would otherwise need (a small absolute error there is still a
    # large *relative* one), rather than tightened to accommodate one
    # outlier -- the actual rigorous check is the proved worst-case bound
    # above, already asserted.
    np.testing.assert_allclose(y_quant, y_float, rtol=0.05, atol=1e-2)


def test_dynamic_quantize_matmul_declines_pre_opset11():
    # DynamicQuantizeLinear needs opset >= 11 (patternMatchPredicate's first
    # check); the simplest of the pass's several decline conditions to
    # exercise. A plain MatMul at an older opset must survive untouched.
    rng = np.random.default_rng(2)
    rows, K, N = 4, 5, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
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
        model, "dynamic_quantize_matmul", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["MatMul"]
