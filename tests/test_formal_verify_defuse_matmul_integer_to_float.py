"""Formal check for DefuseMatMulIntegerToFloat (opt-in; onnxsim's own
``onnxsim/passes/defuse_matmul_integer_to_float.h``): the exact inverse of
``dynamic_quantize_matmul.h``'s rewrite. It folds the pattern that pass emits
-- ``Xq, Xs, Xzp = DynamicQuantizeLinear(X)``; ``Acc = MatMulInteger(Xq, Wq,
Xzp)`` (``Wq`` a constant INT8 ``[K, N]``, implicit ``b_zero_point = 0``);
``Y = Cast<float>(Acc) * (Xs * Ws)`` (``Ws`` a constant FLOAT32 ``[N]``
per-column scale); optionally ``Y = Y + Bias`` -- back into a single
``Y = MatMul(X, Dequant(Wq, Ws))`` (``+ Bias`` if present), by dequantizing
the constant weight to float ONCE, at pass-transform time
(``Dequant(Wq, Ws)[k, n] = Wq[k, n] * Ws[n]``, see ``DequantizeWeightKN``),
rather than leaving it a runtime ``DequantizeLinear`` node.

The genuine algebraic content, spelled out in the header comment: with
implicit ``b_zero_point = 0``, ``MatMulInteger(Xq, Wq, Xzp)[i, n] = sum_k
(Xq[i, k] - Xzp) * Wq[k, n]`` -- an exact *integer* sum (only the activation
zero-point applies). Combined with the DEFINING relationship
``DynamicQuantizeLinear`` establishes between its float input ``X`` and its
quantized triple ``(Xq, Xs, Xzp)`` -- the standard asymmetric-quantization
dequantization identity ``X[i, k] == Xs * (Xq[i, k] - Xzp)`` -- the rewrite's
soundness follows algebraically: ``Cast<float>(Acc)[i, n] * (Xs * Ws[n]) ==
MatMul(X, Dequant(Wq, Ws))[i, n] == sum_k X[i, k] * Wq[k, n] * Ws[n]``.

That dequantization identity is taken here as an AXIOM about what ``X`` and
its quantized triple satisfy -- it is ``DynamicQuantizeLinear``'s own
contract for the values it produces, not something onnxsim itself verifies at
runtime. This is the crux of why the rewrite is sound despite operating on
already-lossily-quantized data: it is an *exact* identity given that the
axiom holds for the concrete ``(Xq, Xs, Xzp)`` triple actually in the graph,
which it does by construction, since ``DynamicQuantizeLinear`` is *defined*
to produce exactly that relationship. The proof below has Z3 genuinely use
that axiom to derive the equivalence -- it does not start from two formulas
that are already textually identical -- and a negative-control test confirms
that without it, the two computations can disagree, so the axiom is doing
real work.

The proof models a concrete contraction depth ``_K`` and output width ``_N``:
``Xq``/``Wq`` are uninterpreted ``Int, Int -> Int`` functions (literal
quantized integer codes), ``Ws`` an uninterpreted ``Int -> Real`` function
(per-output-column scale), ``Xs``/``Xzp`` free Real/Int scalars
(``DynamicQuantizeLinear`` produces one scale/zero-point for the whole
activation tensor, not per-element), and ``X`` an uninterpreted
``Int, Int -> Real`` function constrained by the dequantization axiom. The
axiom is ground-instantiated over the small concrete ``_K`` (a Python-level
conjunction, not a Z3 ``ForAll``) -- this suite has hit ``ForAll``-related Z3
hangs before (``fuse_consecutive_log_softmax``'s original proof, fixed by
ground-instantiating small-domain axioms instead of quantifying them), and
ground instantiation is simple here since the row index ``i`` the proof
concerns is already a single free (implicitly universally quantified, via
``prove()``) variable shared between the axiom and the claim -- no separate
quantifier over ``i`` is needed at all.

Both the equivalence and its ``+ Bias`` variant are additionally composed
with an arbitrary uninterpreted ``consumer`` (this suite's standard
substitution-safety idiom, e.g. test_formal_verify_eliminate_identity.py) to
confirm whatever reads the rewritten output downstream sees the same value
the original chain would have produced.

Differential tests build the exact recognized pattern by first running the
real ``dynamic_quantize_matmul`` pass (via :func:`onnxsim.quantize_dynamic`,
which applies that single rewrite directly with no other pass involved) on a
plain float ``MatMul``/``Gemm``, then reading back the ``Wq``/``Ws``
initializers it produced -- more robust than hand-assembling the
``DynamicQuantizeLinear``/``MatMulInteger``/``Cast``/``Mul`` chain's node
shapes and dtypes via ``onnx.parser`` text, and it guarantees the input to
``defuse_matmul_integer_to_float`` is byte-for-byte the same pattern
production code emits. ``simplify_isolated_extra`` then runs
``defuse_matmul_integer_to_float`` alone; per its own docstring, isolating one
opt-in pass runs it without its usual companion dead-code pass
(``eliminate_deadend`` is a default pass, skipped here like every other one),
so the old ``DynamicQuantizeLinear``/``MatMulInteger``/``Cast``/``Mul`` chain
is left dangling as dead code rather than actually removed -- op-count
assertions below therefore walk from the live graph output via ``producer()``
instead of checking that those op types disappear entirely.

The firing tests below pass ``check_n=0`` to ``simplify_isolated_extra``,
disabling onnxsim's own random-input equivalence check for this one step --
not because the rewrite is unsound, but because that checker compares actual
runtime outputs at its default tight tolerance (rtol=1e-4, atol=1e-5), and
the *quantized* graph (before defuse) and the *defused* graph (after) are
genuinely not numerically identical at that tolerance: this pass's own
matmul node reuses the ORIGINAL float activation ``X`` (the value
``DynamicQuantizeLinear`` itself reads its input from, per
``MatchMatMulIntegerDequant``'s ``info.x = dql->inputs()[0]``), not a
reconstruction of it via the ``Xs * (Xq - Xzp)`` dequantization formula. The
Z3 proof above's ``dequant_axiom`` is an idealized statement that this
reconstruction recovers ``X`` exactly; real ``DynamicQuantizeLinear`` rounds,
so it does not, in general. The defused graph is therefore *more* accurate
than the quantized one it replaces (it skips the activation-quantization
step entirely) rather than bit-identical to it -- exactly the tradeoff
``tests/test_simple.py``'s own
``test_defuse_matmul_integer_to_float_undoes_dynamic_quantize_matmul``
documents ("Only W went through lossy INT8 quantization ... this should be
close, not bitwise equal"). What IS checked exactly here, with ordinary
``np.testing.assert_allclose`` at a tight tolerance, is the one piece that
really is a bit-for-bit deterministic transform: that the new ``MatMul``'s
weight initializer equals ``Wq * Ws`` computed independently in numpy.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_K = 3  # concrete contraction dim -- enough to exercise the summation
_N = 2  # concrete output-channel dim


def _formulas():
    """Builds the shared Z3 vocabulary/formulas for the soundness proof.

    Returns ``(dequant_axiom, cast_and_scale, dequantized_matmul, consumer,
    Bias)`` -- see the module docstring for what each represents.
    """
    Xq = z3.Function("Xq", z3.IntSort(), z3.IntSort(), z3.IntSort())
    Wq = z3.Function("Wq", z3.IntSort(), z3.IntSort(), z3.IntSort())
    Ws = z3.Function("Ws", z3.IntSort(), z3.RealSort())
    X = z3.Function("X", z3.IntSort(), z3.IntSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())
    Bias = z3.Function("Bias", z3.IntSort(), z3.RealSort())
    i, n = z3.Ints("i n")
    Xs = z3.Real("Xs")
    Xzp = z3.Int("Xzp")

    # DynamicQuantizeLinear's own defining contract relating float X to its
    # quantized triple (Xq, Xs, Xzp), ground-instantiated over the concrete
    # k in range(_K) for the SAME free row index i used below -- no ForAll
    # needed, since i is already implicitly universal via prove().
    dequant_axiom = z3.And(
        *[X(i, k) == Xs * (z3.ToReal(Xq(i, k)) - z3.ToReal(Xzp)) for k in range(_K)]
    )

    # Cast<float>(MatMulInteger(Xq, Wq, Xzp))[i, n] * (Xs * Ws[n]): the
    # integer accumulation (implicit b_zero_point=0, so only Xzp applies) is
    # exact int arithmetic, cast to Real only once, matching Cast<float>
    # running after the integer accumulator.
    acc = sum((Xq(i, k) - Xzp) * Wq(k, n) for k in range(_K))
    cast_and_scale = z3.ToReal(acc) * (Xs * Ws(n))

    # MatMul(X, Dequant(Wq, Ws))[i, n], with Dequant(Wq, Ws)[k, n] =
    # Wq[k, n] * Ws[n] per DequantizeWeightKN.
    dequantized_matmul = sum(X(i, k) * (z3.ToReal(Wq(k, n)) * Ws(n)) for k in range(_K))

    return dequant_axiom, cast_and_scale, dequantized_matmul, consumer, Bias, n


def test_defuse_matmul_integer_to_float_is_sound():
    dequant_axiom, cast_and_scale, dequantized_matmul, consumer, _Bias, _n = _formulas()

    prove(z3.Implies(dequant_axiom, cast_and_scale == dequantized_matmul))
    prove(
        z3.Implies(
            dequant_axiom,
            consumer(cast_and_scale) == consumer(dequantized_matmul),
        )
    )


def test_defuse_matmul_integer_to_float_bias_variant_is_sound():
    # The "+ Bias" branch (runTransform's kAdd case): the same equivalence
    # holds with "+ Bias[n]" added to both sides, since Bias is added
    # identically to either side after the (already-proven-equal) matmul
    # term.
    dequant_axiom, cast_and_scale, dequantized_matmul, consumer, Bias, n = _formulas()

    lhs = cast_and_scale + Bias(n)
    rhs = dequantized_matmul + Bias(n)

    prove(z3.Implies(dequant_axiom, lhs == rhs))
    prove(z3.Implies(dequant_axiom, consumer(lhs) == consumer(rhs)))


def test_defuse_matmul_integer_to_float_negative_control_requires_dequant_axiom():
    # Sanity check that the proof above is genuine, not vacuous: without the
    # dequantization axiom constraining X, Cast<float>(Acc) * (Xs * Ws) (which
    # does not mention X at all) and MatMul(X, Dequant(Wq, Ws)) (which is
    # built entirely from X) need not agree -- Z3 must find a real
    # counterexample, confirming the axiom is doing real, load-bearing work
    # rather than the equivalence holding for any X regardless.
    _dequant_axiom, cast_and_scale, dequantized_matmul, _consumer, _Bias, _n = (
        _formulas()
    )

    solver = z3.Solver()
    solver.add(z3.Not(cast_and_scale == dequantized_matmul))
    assert solver.check() == z3.sat, (
        "the two computations always agree even without the dequantization "
        "axiom -- negative control is vacuous"
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
    """Reads back the ``(Wq, Ws)`` pair ``dynamic_quantize_matmul`` (run via
    :func:`onnxsim.quantize_dynamic`) wrote into ``quantized_model`` -- the
    constant INT8 [K, N] weight feeding ``MatMulInteger`` and the constant
    FLOAT32 [N] per-column scale multiplied against the activation scale.

    Found by walking the actual node structure (mirroring
    ``MatchMatMulIntegerDequant`` itself), not by initializer dtype/shape
    alone: an optional bias initializer (FLOAT32, shape [N], same as ``Ws``)
    would otherwise be ambiguous with ``Ws`` in the bias-variant model.
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


def test_defuse_matmul_integer_to_float_pass_fires_and_dequantizes_correctly():
    # Build a plain float MatMul, then run the real dynamic_quantize_matmul
    # rewrite (dynamic_quantize_matmul.h, applied standalone via
    # onnxsim.quantize_dynamic) on it to get the EXACT DynamicQuantizeLinear
    # + MatMulInteger + Cast + Mul + Mul pattern defuse_matmul_integer_to_float
    # recognizes, with real quantized Wq/Ws values.
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
    assert next(n for n in quantized.graph.node if "Y" in n.output).op_type == "Mul"
    wq, ws = _weight_and_scale(quantized)
    assert wq.shape == (K, N)
    assert ws.shape == (N,)
    expected_dequant = wq.astype(np.float32) * ws[np.newaxis, :]

    # check_n=0: see the module docstring for why the built-in random-input
    # check does not apply to this comparison (quantized-vs-defused, not
    # float-vs-defused).
    sim_model, _ops = simplify_isolated_extra(
        quantized, "defuse_matmul_integer_to_float", check_n=0
    )

    matmul_node = producer(sim_model, "Y")
    assert matmul_node.op_type == "MatMul"
    assert matmul_node.input[0] == "X"
    new_w_init = next(
        init
        for init in sim_model.graph.initializer
        if init.name == matmul_node.input[1]
    )
    assert list(new_w_init.dims) == [K, N]
    np.testing.assert_allclose(
        numpy_helper.to_array(new_w_init), expected_dequant, rtol=1e-6
    )


def test_defuse_matmul_integer_to_float_bias_variant_fires_and_dequantizes_correctly():
    # PyTorch nn.Linear layout (weight [N, K], Gemm(X, W, B, transB=1)) with a
    # bias -- dynamic_quantize_matmul.h adds the bias back in float after
    # dequantization (an Add node); defuse_matmul_integer_to_float must
    # recognize its own Add branch and produce MatMul(X, Dequant) + Bias.
    rng = np.random.default_rng(1)
    rows, K, N = 3, 4, 2
    weight = rng.standard_normal((N, K)).astype(np.float32) * 0.5
    bias = rng.standard_normal(N).astype(np.float32)
    model = _model(
        f"""
        g (float[{rows},{K}] X) => (float[{rows},{N}] Y)
        {{
          Y = Gemm<transB = 1>(X, W, B)
        }}
        """,
        [_f32(weight, "W"), _f32(bias, "B")],
    )

    quantized = onnxsim.quantize_dynamic(model)
    assert next(n for n in quantized.graph.node if "Y" in n.output).op_type == "Add"
    wq, ws = _weight_and_scale(quantized)
    assert wq.shape == (K, N)
    assert ws.shape == (N,)
    expected_dequant = wq.astype(np.float32) * ws[np.newaxis, :]

    sim_model, _ops = simplify_isolated_extra(
        quantized, "defuse_matmul_integer_to_float", check_n=0
    )

    add_node = producer(sim_model, "Y")
    assert add_node.op_type == "Add"
    matmul_input, bias_input = add_node.input
    assert bias_input == "B"
    matmul_node = producer(sim_model, matmul_input)
    assert matmul_node.op_type == "MatMul"
    assert matmul_node.input[0] == "X"
    new_w_init = next(
        init
        for init in sim_model.graph.initializer
        if init.name == matmul_node.input[1]
    )
    assert list(new_w_init.dims) == [K, N]
    np.testing.assert_allclose(
        numpy_helper.to_array(new_w_init), expected_dequant, rtol=1e-6
    )


def test_defuse_matmul_integer_to_float_declines_explicit_b_zero_point():
    # Negative control: a MatMulInteger with an explicit (4th) b_zero_point
    # input is a different, more general shape than the exact 3-input pattern
    # dynamic_quantize_matmul.h emits (implicit b_zero_point=0) --
    # MatchMatMulIntegerDequant requires exactly 3 inputs, so this must be
    # left completely untouched.
    rng = np.random.default_rng(2)
    rows, K, N = 3, 4, 2
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
    quantized = onnxsim.quantize_dynamic(model)

    mmi_node = next(n for n in quantized.graph.node if n.op_type == "MatMulInteger")
    assert len(mmi_node.input) == 3
    b_zero_point = onnx.numpy_helper.from_array(
        np.array(0, dtype=np.int8), "b_zero_point"
    )
    quantized.graph.initializer.append(b_zero_point)
    mmi_node.input.append("b_zero_point")
    onnx.checker.check_model(quantized)

    sim_model, _ops = simplify_isolated_extra(
        quantized, "defuse_matmul_integer_to_float"
    )
    y_node = producer(sim_model, "Y")
    assert y_node.op_type == "Mul"
    mmi_node = next(n for n in sim_model.graph.node if n.op_type == "MatMulInteger")
    assert len(mmi_node.input) == 4
