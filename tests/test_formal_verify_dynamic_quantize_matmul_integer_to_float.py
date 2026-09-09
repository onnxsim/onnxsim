"""Formal check for DynamicQuantizeMatMulIntegerToFloat (opt-in; onnxsim's own
``onnxsim/passes/dynamic_quantize_matmul_integer_to_float.h``):
``test_formal_verify_dynamic_quantize_matmul.py`` is this file's direct
template -- READ THAT FILE'S MODULE DOCSTRING FIRST, since almost everything
there applies here unchanged. This pass applies the EXACT SAME "dynamic
quantization" rewrite (same predicate: opset >= 11, float32 ``X``, constant
2-D float32 ``W``, ``IsSafeInt32ReductionDepth``; same per-output-channel
``QuantizeWeightPerChannelKN`` weight scheme; same runtime
``DynamicQuantizeLinear`` activation quantization) to ``Y = MatMul(X, W)`` (or
a "vanilla" ``Gemm``, ``+ Bias`` optional) -- the ONLY difference is which
node(s) do the dequantization::

    Xq, Xs, Xzp = DynamicQuantizeLinear(X)                          # per-tensor UINT8, runtime scale
    Y           = MatMulIntegerToFloat(Xq, Wq, Xs, Ws, Xzp, "" [,Bias])  # one "com.microsoft" contrib op

instead of ``dynamic_quantize_matmul``'s own four-to-five standard-ONNX nodes
(``MatMulInteger`` -> ``Cast`` -> ``Mul`` -> ``Mul`` [-> ``Add``]).
``MatMulIntegerToFloat``'s own schema semantics (per this pass's header
comment) are true int8 compute, dequantize, and optionally add a bias, ALL IN
ONE OP -- i.e. it computes exactly::

    MatMulIntegerToFloat(Xq, Wq, Xs, Ws, Xzp, "", Bias)
        == Cast<float>(MatMulInteger(Xq, Wq, Xzp)) * (Xs .* Ws) [+ Bias]

which is the SAME formula ``dynamic_quantize_matmul``'s own multi-node chain
computes -- just as one op's own documented semantics instead of several
nodes' composed semantics. The 6th input position (``b_zero_point``) is
omitted (symmetric weight quantization, i.e. always 0) via the standard ONNX
empty-string placeholder for a skipped middle-position optional input, the
same ``kUndefined``-node mechanism other passes in this family use for a
skipped optional slot (e.g. ``mask_index``-shaped gaps elsewhere).

Since the underlying algebra is identical to ``dynamic_quantize_matmul``'s
own, the Z3 proof content here is IDENTICAL in substance to that file's own:
the same two-lemma structure (an exact ring identity connecting the node's
own literal output formula to the elementwise-dequantized dot product, then
the reused bounded-error lemma, matching ``quantized_mac_bound``'s own
derivation), the same ``_K = 2``, and the same direct-error-variable idiom
for the bound lemma from the start -- ``dynamic_quantize_matmul``'s own
module docstring documents in detail why handing Z3 the combined nonlinear
statement (or even the intermediate ``Xs*(Xq-Xzp)``-shaped multiplications
inside the *same* query as the abs-value bound search) causes its
nonlinear-arithmetic search to hang past a minute at ``_K`` as small as 2;
that same fix -- splitting an exact ring identity from a bound lemma stated
directly in per-tap error variables, and combining the two by ordinary
mathematical substitution in prose rather than asking Z3 to search the
combined formula -- is reused here unchanged, this file only reusing the
vocabulary (self-contained, per this suite's convention: nothing is imported
from that file).

Per-element rounding bounds, ring identity, and bound derivation are
otherwise verbatim identical to ``dynamic_quantize_matmul``'s own -- see that
file's module docstring for the full derivation. The only thing that differs
between the two files' proofs is which node chain the identity lemma is
proving equal to ``dequant_elemwise``: there, the explicit
``Cast<float>(Acc) * (Xs * Ws)`` node chain; here, ``MatMulIntegerToFloat``'s
own one-op defining formula (algebraically the exact same expression).

A negative-control test confirms the rounding-error hypotheses are load-
bearing, and no consumer-composition step is added, for the same reasons
``dynamic_quantize_matmul``'s own module docstring gives (a numeric *bound*
does not compose through an arbitrary, not-necessarily-Lipschitz consumer the
way an *equality* does).

Differential tests build a plain float ``MatMul``/bias ``Gemm`` via
``onnx.parser``, run the real pass through
:func:`onnxsim.quantize_dynamic_matmul_integer_to_float`, and confirm: the
rewrite produces exactly ``DynamicQuantizeLinear`` followed by a SINGLE
``MatMulIntegerToFloat`` node (domain ``com.microsoft``) -- unlike
``dynamic_quantize_matmul``'s own multi-node dequantization chain -- fed by
``DynamicQuantizeLinear``'s three outputs plus the statically-quantized
``Wq``/``Ws``, with the 6th input (``b_zero_point``) wired as the empty-string
placeholder rather than simply absent; the ``Wq``/``Ws`` initializers match
the SAME independent numpy re-implementation of ``QuantizeWeightPerChannelKN``
``test_formal_verify_dynamic_quantize_matmul.py`` uses; the model's
``opset_import`` gains a ``com.microsoft`` entry (version 1) once the pass
fires, a structural check specific to this pass (its sibling never touches
opset imports, since it only emits standard-ONNX ops); the actual runtime
output error, checked by running the real compiled ``MatMulIntegerToFloat``
ONNX Runtime contrib kernel (confirmed present in this environment -- see
below), stays within the bound proved above; and a pre-opset-11 model is
declined outright, the simplest of the pass's several decline conditions to
exercise.

ONNX Runtime (1.29.0 in this environment) DOES ship a working CPU
``MatMulIntegerToFloat`` kernel: confirmed empirically by running the
compiled rewrite's own output through ``onnxruntime.InferenceSession`` before
writing these tests (a plain ``MatMul`` and a bias ``Gemm`` variant both
executed and produced results close to the true float computation). So the
runtime-error differential test below runs the real contrib kernel, exactly
like ``test_formal_verify_dynamic_quantize_matmul.py``'s own
``test_..._output_is_close_to_float_within_proved_bound`` does for its sibling
node chain -- no numpy fallback is needed here.

The firing tests below pass ``check_n=0`` to ``simplify_isolated_extra``, for
the same reason ``dynamic_quantize_matmul``'s own tests do (confirmed
empirically): this is a genuinely lossy INT8 rewrite, and onnxsim's own
random-input equivalence check (default tolerance) fails on the quantized
output.
"""

import numpy as np
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 2  # concrete number of contraction taps -- matches quantized_mac_bound's
# and dynamic_quantize_matmul's own _K; see the module docstring for why.


def _abs(v):
    return z3.If(v >= 0, v, -v)


def _identity_formulas():
    """Builds the Z3 vocabulary for the exact ring identity between
    ``MatMulIntegerToFloat``'s own one-op defining formula and the
    elementwise-dequantized dot product -- needs ``Xq``/``Wq``/``Xzp``
    explicitly, since it's a claim about that specific formula's shape.
    Returns ``(y, dequant_elemwise)``.
    """
    Xq = [z3.Real(f"Xq{k}") for k in range(_K)]  # Xq[i, k]
    Wq = [z3.Real(f"Wq{k}") for k in range(_K)]  # Wq[k, n]
    Xs = z3.Real("Xs")  # DynamicQuantizeLinear's per-tensor scale
    Ws = z3.Real("Ws")  # this pass's per-output-column scale Ws[n]
    Xzp = z3.Real("Xzp")  # DynamicQuantizeLinear's per-tensor zero point

    dequant_x = [Xs * (Xq[k] - Xzp) for k in range(_K)]
    dequant_w = [Ws * Wq[k] for k in range(_K)]

    # MatMulIntegerToFloat's own defining formula: true int8 compute (exact
    # integer accumulation, implicit b_zero_point=0 via the empty-string
    # placeholder, so only Xzp applies), then dequantize -- all in one op,
    # rather than dynamic_quantize_matmul's separate MatMulInteger + Cast +
    # Mul node chain (algebraically the same expression either way).
    acc = sum((Xq[k] - Xzp) * Wq[k] for k in range(_K))
    y = acc * (Xs * Ws)

    dequant_elemwise = sum(dequant_x[k] * dequant_w[k] for k in range(_K))

    return y, dequant_elemwise


def _bound_formulas():
    """Builds the Z3 vocabulary for the bounded-error claim, direct-error-
    variable formulation from the start (matching
    ``dynamic_quantize_matmul``'s own FIXED, working version, not its
    original nonlinear-blowup attempt -- see the module docstring). Returns
    ``(float_matmul, dequant_elemwise, rounding_bounds, bound)``.
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


def test_dynamic_quantize_matmul_integer_to_float_accumulator_equals_elementwise_dequant():
    # y (MatMulIntegerToFloat's own one-op defining formula) is EXACTLY equal
    # to the elementwise product of each operand's dequantized
    # reconstruction -- a pure ring identity (distributing the shared
    # Xs * Ws factor into the sum), true unconditionally, independent of any
    # rounding-error hypothesis. No abs/If branching, so Z3 checks this in
    # milliseconds regardless of the nonlinear-arithmetic cost the bound
    # proof below has.
    y, dequant_elemwise = _identity_formulas()
    prove(y == dequant_elemwise)


def test_dynamic_quantize_matmul_integer_to_float_error_is_bounded():
    # The genuine bounded-error claim: given each operand's own rounding
    # bound, the true float dot product and its elementwise-dequantized
    # counterpart cannot differ by more than `bound`. Combined with the
    # identity proved above (y == dequant_elemwise, connecting this same
    # dequant_elemwise quantity to MatMulIntegerToFloat's actual defining
    # formula), this also bounds `float_matmul - y` -- the pass's real
    # output error -- by the same amount; see the module docstring (and
    # dynamic_quantize_matmul's own) for why that combination is done here
    # in prose/by substitution rather than as a single Z3 query.
    float_matmul, dequant_elemwise, rounding_bounds, bound = _bound_formulas()
    error = float_matmul - dequant_elemwise
    prove(z3.Implies(rounding_bounds, z3.And(error <= bound, -error <= bound)))


def test_dynamic_quantize_matmul_integer_to_float_bias_variant_error_is_bounded():
    # The "+ Bias" branch: MatMulIntegerToFloat's own Bias input, added
    # directly in the same op rather than via a separate Add node. Adding
    # the same Bias(n) to both the true and the dequantized computation
    # leaves their difference -- and therefore the bound on it -- unchanged,
    # since Bias cancels out of the error term algebraically. Same
    # cancellation dynamic_quantize_matmul's own bias variant relies on.
    float_matmul, dequant_elemwise, rounding_bounds, bound = _bound_formulas()
    bias = z3.Real("Bias")
    error_with_bias = (float_matmul + bias) - (dequant_elemwise + bias)
    prove(
        z3.Implies(
            rounding_bounds, z3.And(error_with_bias <= bound, -error_with_bias <= bound)
        )
    )


def test_dynamic_quantize_matmul_integer_to_float_negative_control_requires_rounding_bounds():
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


def _weight_and_scale(quantized_model, mmitf_node):
    """Reads back the ``(Wq, Ws)`` pair the real pass wrote into
    ``quantized_model``, given the ``MatMulIntegerToFloat`` node itself --
    input positions ``Wq, Xs, Ws`` are fixed by the op's own schema, unlike
    ``dynamic_quantize_matmul``'s sibling helper, which has to hunt down a
    ``Mul`` node by elimination since there's no single node whose input
    list documents the pairing directly.
    """
    graph = quantized_model.graph
    wq_name = mmitf_node.input[1]
    ws_name = mmitf_node.input[3]
    wq_init = next(init for init in graph.initializer if init.name == wq_name)
    ws_init = next(init for init in graph.initializer if init.name == ws_name)
    return numpy_helper.to_array(wq_init), numpy_helper.to_array(ws_init)


def _quantize_weight_per_channel_kn(weight):
    """Independent numpy re-implementation of
    ``QuantizeWeightPerChannelKN`` (quantize_matmul_common.h) -- identical to
    ``test_formal_verify_dynamic_quantize_matmul.py``'s own helper of the
    same name, since both passes share this exact weight-quantization
    scheme: per-output-column (axis 1) symmetric INT8 quantization,
    scale = max(|column|) / 127 (or 1.0 for an all-zero column), codes =
    round(w / scale) clipped to [-127, 127].
    """
    scale = np.max(np.abs(weight), axis=0)
    scale = np.where(scale > 0, scale / 127.0, 1.0).astype(np.float32)
    codes = np.clip(np.round(weight / scale[np.newaxis, :]), -127, 127).astype(np.int8)
    return codes, scale


def test_dynamic_quantize_matmul_integer_to_float_pass_fires_and_matches_scheme():
    # Build a plain float MatMul and run the real
    # dynamic_quantize_matmul_integer_to_float rewrite (via
    # onnxsim.quantize_dynamic_matmul_integer_to_float, which applies exactly
    # this one rewrite -- see its own docstring), then confirm both the node
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

    quantized = onnxsim.quantize_dynamic_matmul_integer_to_float(model)

    # check_n=0: see the module docstring for why the built-in random-input
    # check (tolerance rtol=1e-4/atol=1e-5) does not apply to a genuinely
    # lossy INT8 rewrite.
    sim_model, _ops = simplify_isolated_extra(
        model, "dynamic_quantize_matmul_integer_to_float", check_n=0
    )

    # Exactly one node (MatMulIntegerToFloat) sits between DynamicQuantizeLinear
    # and the graph output -- unlike dynamic_quantize_matmul's own multi-node
    # dequantization chain.
    mmitf_node = producer(sim_model, "Y")
    assert mmitf_node.op_type == "MatMulIntegerToFloat"
    assert mmitf_node.domain == "com.microsoft"
    assert len(mmitf_node.input) == 6  # no Bias for a plain MatMul
    dql_node = producer(sim_model, mmitf_node.input[0])
    assert dql_node.op_type == "DynamicQuantizeLinear"
    assert mmitf_node.input[2] == dql_node.output[1]  # Xs
    assert mmitf_node.input[4] == dql_node.output[2]  # Xzp
    # b_zero_point (index 5) is the empty-string placeholder, not absent.
    assert mmitf_node.input[5] == ""

    # The model's opset_import gains a com.microsoft entry (version 1) --
    # a structural check specific to this pass, since its sibling
    # (dynamic_quantize_matmul) only emits standard-ONNX ops and never
    # touches opset imports.
    ms_opsets = [op for op in sim_model.opset_import if op.domain == "com.microsoft"]
    assert len(ms_opsets) == 1
    assert ms_opsets[0].version == 1
    # The original model had no com.microsoft import at all.
    assert not [op for op in model.opset_import if op.domain == "com.microsoft"]

    wq, ws = _weight_and_scale(sim_model, mmitf_node)
    assert wq.shape == (K, N)
    assert ws.shape == (N,)
    expected_wq, expected_ws = _quantize_weight_per_channel_kn(weight)
    np.testing.assert_array_equal(wq, expected_wq)
    np.testing.assert_allclose(ws, expected_ws, rtol=1e-6)

    # quantize_dynamic_matmul_integer_to_float (the whole-model convenience
    # wrapper) produces the exact same shape.
    q_mmitf = next(
        n for n in quantized.graph.node if n.op_type == "MatMulIntegerToFloat"
    )
    assert q_mmitf.domain == "com.microsoft"
    q_wq, q_ws = _weight_and_scale(quantized, q_mmitf)
    np.testing.assert_array_equal(q_wq, expected_wq)
    np.testing.assert_allclose(q_ws, expected_ws, rtol=1e-6)


def test_dynamic_quantize_matmul_integer_to_float_bias_variant_wires_bias_input():
    # A "vanilla" Gemm with a bias: MatMulIntegerToFloat gets a 7th input
    # (Bias) appended directly, rather than a separate Add node the way
    # dynamic_quantize_matmul's own chain needs.
    rng = np.random.default_rng(4)
    rows, K, N = 4, 5, 3
    weight = rng.standard_normal((K, N)).astype(np.float32) * 0.7
    bias = rng.standard_normal((N,)).astype(np.float32) * 0.3
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    sim_model, _ops = simplify_isolated_extra(
        model, "dynamic_quantize_matmul_integer_to_float", check_n=0
    )

    mmitf_node = producer(sim_model, "Y")
    assert mmitf_node.op_type == "MatMulIntegerToFloat"
    assert len(mmitf_node.input) == 7
    assert mmitf_node.input[5] == ""  # b_zero_point still the empty placeholder
    assert mmitf_node.input[6] == "B"  # Bias appended directly, no Add node


def test_dynamic_quantize_matmul_integer_to_float_output_is_close_to_float_within_proved_bound():
    # Differential check mirroring dynamic_quantize_matmul's own
    # test_..._output_is_close_to_float_within_proved_bound: run the real
    # quantized graph through onnxruntime's actual MatMulIntegerToFloat
    # contrib kernel (confirmed present and working in this environment --
    # see module docstring) so Xq/Xs/Xzp come from onnxruntime's own
    # DynamicQuantizeLinear, and confirm every output element's error
    # against the true float MatMul stays within the bound the proof above
    # derives, with Xs/Ws(n) taken from the actual run rather than assumed.
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

    quantized = onnxsim.quantize_dynamic_matmul_integer_to_float(model)
    dql = next(n for n in quantized.graph.node if n.op_type == "DynamicQuantizeLinear")
    mmitf = next(n for n in quantized.graph.node if n.op_type == "MatMulIntegerToFloat")
    # Expose DynamicQuantizeLinear's own (Xq, Xs, Xzp) as extra graph outputs
    # so the bound is checked against onnxruntime's actual quantization, not
    # a reimplementation of DynamicQuantizeLinear's formula.
    exposed = quantized.__class__()
    exposed.CopyFrom(quantized)
    for name in dql.output:
        exposed.graph.output.add().name = name

    sess = ort.InferenceSession(
        exposed.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    output_names = [o.name for o in exposed.graph.output]
    results = dict(zip(output_names, sess.run(output_names, {"X": x})))
    y_quant = results["Y"]
    x_scale = float(results[dql.output[1]])

    _wq, ws = _weight_and_scale(quantized, mmitf)

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


def test_dynamic_quantize_matmul_integer_to_float_declines_pre_opset11():
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
        model, "dynamic_quantize_matmul_integer_to_float", check_n=0
    )
    assert [n.op_type for n in sim_model.graph.node] == ["MatMul"]
