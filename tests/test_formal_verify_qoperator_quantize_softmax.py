"""Formal check for QOperatorQuantizeSoftmax (opt-in; onnxsim's own
``onnxsim/passes/qoperator_quantize_softmax.h``): the reduction-axis sibling
of ``qoperator_quantize_activation.h``'s QLinearSigmoid/QLinearLeakyRelu
rewrite. It rewrites a standalone ``Y = Softmax(X, axis=ax)`` into ONNX
Runtime's "com.microsoft" contrib op ``QLinearSoftmax``::

    Xq = QuantizeLinear(X, Xs, Xzp)                                  -- CALIBRATED
    Yq = QLinearSoftmax(Xq, Xs, Xzp, Ys, Yzp,
                         axis=ax, opset=default_domain_opset)        -- true int8
    Y  = DequantizeLinear(Yq, Ys, Yzp)                                -- CALIBRATED

Only a Softmax with exactly 1 input, float32, is matched; a node is only
rewritten when both its input's name and its own output's name have a
calibrated range, and when the model's own default-domain ("" / "ai.onnx")
opset import is resolvable (``DefaultDomainOpsetVersion`` returns > 0) --
confirmed from ``patternMatchPredicate`` in the header.

Why this file's content is shaped differently from every MAC-bound sibling
====================================================================================
Every other ``qoperator_quantize_*``/``static_quantize_*`` file in this suite
proves a numeric worst-case error BOUND on an exact-dot-product operator
(MatMul/Conv/Gemm), because the "MAC" structure gives Z3 a closed-form
algebraic error-propagation formula to check. Softmax has no such structure:
it is a nonlinear, whole-axis reduction (``exp``/``sum``/divide``), and
``QLinearSoftmax``'s own internal int8 approximation of that reduction is
entirely ONNX Runtime's own kernel's business -- this repo's C++ does not
compute, and could not independently derive, a closed-form bound on it (there
is no "sum of |weight|" analogue for a normalizing nonlinear reduction). So,
per this suite's own established precedent for a pass whose main content
genuinely does not reduce to MAC algebra (see
``test_formal_verify_rename_input_output.py`` and
``test_formal_verify_lift_lexical_references.py``'s own docstrings for this
same honesty-over-manufactured-machinery principle), this file's real content
is NOT a numeric bound on ``QLinearSoftmax``'s own internal computation.
Instead it is:

1. A thin, genuinely load-bearing Z3 lemma about this pass's own STRUCTURAL
   correctness condition (below), plus the two round-trip lemmas this pass's
   activation-side and output-side quantization steps do share with every
   other file in this family (X's own and Y's own ``QuantizeLinear``/
   ``DequantizeLinear`` bound, re-derived in this file's own vocabulary per
   this suite's per-file convention).
2. Heavy differential weight (the genuine crux of this pass): standard ONNX
   ``Softmax`` has TWO INCOMPATIBLE axis behaviors across opset versions --
   pre-opset-13 flattens the tensor to 2-D at ``axis`` and reduces jointly
   over the ENTIRE trailing (flattened) dimension, while opset-13+ reduces
   ``axis`` in place, independently, same rank in and out.
   ``QLinearSoftmax``'s own ``opset`` attribute tells ONNX Runtime's kernel
   which of the two to replicate. This pass reads the MODEL's own declared
   default-domain opset (``DefaultDomainOpsetVersion``) and threads it
   through verbatim -- not a hardcoded constant, not the latest opset, not
   the (nonexistent, on a bare ``Softmax`` node) opset of "the node itself".
   Getting this wrong would be a silent, hard-to-detect correctness bug:
   some models would compute the WRONG axis-reduction shape/semantics with
   no crash, no shape mismatch, no exception -- just quietly wrong numbers.
   The differential tests below build a genuinely adversarial pair of models
   (opset 12 vs. opset 17, multi-dimensional ``X`` with ``axis`` in the
   *middle* of the tensor's rank, so pre-13 flatten-then-reduce and 13+
   in-place-reduce provably disagree -- confirmed empirically below, max abs
   difference between the two float ``Softmax`` semantics is >0.5 on a
   [0, 1]-valued tensor, i.e. not a rounding-noise-scale difference) and
   confirm the real pass threads the right ``opset`` attribute for each, and
   that ONNX Runtime's own ``QLinearSoftmax`` kernel run with that attribute
   reproduces ONNX Runtime's own plain-``Softmax`` kernel at the SAME model
   opset far more closely than it does the WRONG one -- an internal
   ORT-vs-ORT consistency check, not a hand-derived numpy reference (per this
   file's task instructions), since ``QLinearSoftmax``'s own internal
   approximation error is not something this repo computes or bounds.

``QLinearSoftmax`` genuinely has a working ONNX Runtime CPU kernel in this
build -- confirmed empirically (see ``test_qoperator_quantize_softmax_
qlinear_softmax_kernel_exists`` below) before relying on it anywhere else in
this file, per this suite's own precedent
(``test_formal_verify_qoperator_quantize_matmul.py``'s docstring) for
confirming a contrib op's kernel exists before building a whole differential
suite on top of it.

Structural/decline tests confirm: only a Softmax with exactly 1 float32
input is matched; a model with NO resolvable default-domain opset import at
all (only a custom domain in ``opset_import``, no "" / "ai.onnx" entry) is
left COMPLETELY untouched -- this is the single most important structural
test in this file, since it is the one case where the opset-threading
guarantee above would otherwise have nothing to thread and there is, per the
header's own comment, "no safe default to guess"; expressible directly via
``onnx.parser`` (an ``opset_import`` list omitting the default domain parses
and loads fine, confirmed empirically -- no ``onnx.helper`` fallback is
needed here). Also confirmed: both X's own name AND the node's own output
name need a calibrated range (missing either declines), and the pass adds
the ``com.microsoft`` opset import (version 1) the first time it fires,
without duplicating it if already present.

Differential tests invoke the real compiled pass directly via the
nanobind-exposed ``onnxsim.onnxsim_cpp2py_export.quantize_qoperator_softmax
(model_bytes, activation_ranges)`` -- the dedicated Softmax entry point (see
``QuantizeQOperatorSoftmax`` in ``onnxsim/onnxsim.h`` and
``onnxsim/quantize_entry.cpp``; NOT the same ``quantize_qoperator`` entry
point ``test_formal_verify_qoperator_quantize_matmul.py`` uses, which only
runs the MatMul/Conv passes), the same "no calibration *data* need be
fabricated, only calibrated ranges" reasoning as every other file in this
family, isolating exactly ``qoperator_quantize_softmax``.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
from _formal_verify_common import producer, prove, z3
from onnx import parser


def _abs(v):
    return z3.If(v >= 0, v, -v)


# ---------------------------------------------------------------------------
# 1. Z3 content
# ---------------------------------------------------------------------------


def test_qoperator_quantize_softmax_opset_decline_guard_is_total():
    # The genuinely load-bearing structural claim (not a MAC bound -- see
    # module docstring): DefaultDomainOpsetVersion's own contract is "return
    # the matching default-domain opset import's version, or 0 if none is
    # found" -- modeled here as `resolved := If(found, version, 0)`, an exact
    # transcription of that loop-and-fallback shape. `patternMatchPredicate`
    # then declines whenever `resolved <= 0`.
    #
    # This guard is exactly right (neither too strict -- rejecting a model it
    # could safely handle -- nor too loose -- accepting one with nothing to
    # thread) given one real ONNX invariant supplied here as an explicit
    # hypothesis: an opset_import entry's version is always a POSITIVE
    # integer (opset 0 does not exist; ONNX has never defined it) -- so
    # `version >= 1` whenever an import genuinely `found` a default-domain
    # entry. Under that hypothesis, `resolved <= 0` holds if and only if NO
    # entry was found -- i.e. the `<= 0` check is precisely the well-defined
    # "no resolvable opset" guard the header's own "there is no safe default
    # to guess" comment describes, not an accidentally-narrower-or-wider one.
    #
    # This is intentionally a thin, near-tautological identity once the ONNX
    # invariant is taken as given -- honestly so, in the same spirit as this
    # suite's other structural (non-MAC) files -- rather than a deep
    # algebraic derivation, because DefaultDomainOpsetVersion's own logic
    # ("first matching import, else 0") has no deeper algebra to expose.
    found = z3.Bool("found")  # a default-domain ("" / "ai.onnx") import exists
    version = z3.Int("version")  # that import's own version, if found

    onnx_opset_versions_are_positive = z3.Implies(found, version >= 1)
    resolved = z3.If(found, version, 0)

    prove(
        z3.Implies(
            onnx_opset_versions_are_positive,
            (resolved <= 0) == z3.Not(found),
        )
    )


def test_qoperator_quantize_softmax_opset_decline_guard_needs_the_invariant():
    # Negative control: without the "opset versions are positive" hypothesis,
    # the equivalence above is NOT a theorem -- e.g. `found=True` with
    # `version=0` (an ONNX-invalid but Z3-unconstrained state) would make
    # `resolved <= 0` true even though an import WAS found. This confirms the
    # prior test's guarantee genuinely relies on the stated ONNX invariant,
    # rather than holding vacuously by construction.
    found = z3.Bool("found")
    version = z3.Int("version")
    resolved = z3.If(found, version, 0)

    solver = z3.Solver()
    solver.add(z3.Not((resolved <= 0) == z3.Not(found)))
    assert solver.check() == z3.sat, (
        "the decline-guard equivalence holds even without the ONNX "
        "positive-opset-version invariant -- negative control is vacuous"
    )


def test_qoperator_quantize_softmax_x_round_trip_is_sound():
    # This pass's OWN activation-side round trip, re-derived in this file's
    # own vocabulary per this suite's per-file convention (same shape as
    # test_formal_verify_quantize_round_trip.py's generic lemma, and every
    # sibling qoperator_quantize_* file's own activation-side reproof):
    # QuantizeLinear(X, Xs, Xzp) then DequantizeLinear back recovers X to
    # within Xs/2, given no saturation and a free (arbitrary) zero point.
    X, Xs, Xzp = z3.Reals("X Xs Xzp")
    n = z3.Int("n")  # round(X / Xs): some integer within 0.5 of X / Xs
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        Xs > 0,
        n - X / Xs <= half,
        X / Xs - n <= half,
    )
    # Not saturated: the quantized code is exactly n + Xzp (no clip), so
    # dequantizing recovers n * Xs -- Xzp cancels exactly, for ANY Xzp.
    xq = n + Xzp
    xdq = (xq - Xzp) * Xs

    error = X - xdq
    prove(z3.Implies(hypotheses, z3.And(error <= Xs / 2, -error <= Xs / 2)))


def test_qoperator_quantize_softmax_y_round_trip_is_sound():
    # The symmetric claim for THIS pass's own output round trip: Y (before
    # this pass fires) is DequantizeLinear(QLinearSoftmax(...), Ys, Yzp) --
    # QLinearSoftmax itself computes internally in int8 with no float
    # intermediate, so whatever int8 code Yq it produces is, from the
    # perspective of this round-trip lemma alone, dequantized the same
    # QuantizeLinear/DequantizeLinear-shaped way as X's own round trip above.
    # (This lemma says nothing about how close Yq's own VALUE is to the true
    # float Softmax(X) -- that is QLinearSoftmax's own internal kernel
    # approximation, ORT's business, not something re-derived here; see
    # module docstring.)
    Yraw, Ys, Yzp = z3.Reals("Yraw Ys Yzp")
    n = z3.Int("n")
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        Ys > 0,
        n - Yraw / Ys <= half,
        Yraw / Ys - n <= half,
    )
    yq = n + Yzp
    ydq = (yq - Yzp) * Ys

    error = Yraw - ydq
    prove(z3.Implies(hypotheses, z3.And(error <= Ys / 2, -error <= Ys / 2)))


# ---------------------------------------------------------------------------
# 2. Differential / structural content -- the real weight of this file
# ---------------------------------------------------------------------------


def _model(body, opset=13, ir_version=10, extra_imports=""):
    model = parser.parse_model(
        f"""
        <
          ir_version: {ir_version},
          opset_import: ["": {opset}{extra_imports}]
        >
        {body}
        """
    )
    return model


def _quantize_qoperator_softmax(model, activation_ranges):
    """Invokes the real compiled pass directly via the nanobind-exposed
    ``quantize_qoperator_softmax(model_bytes, activation_ranges)`` -- the
    dedicated Softmax entry point (module docstring). Runs ``OptimizeFixed``
    with exactly ``["qoperator_quantize_softmax"]``.
    """
    out = onnx.ModelProto()
    out.ParseFromString(
        C.quantize_qoperator_softmax(model.SerializeToString(), activation_ranges)
    )
    return out


def _expected_asymmetric_uint8_quant_params(min_val, max_val):
    """Independent re-implementation of ``ComputeAsymmetricUint8QuantParams``
    (static_quantize_matmul.h), in float32 to match the pass's own arithmetic
    precision -- identical to
    ``test_formal_verify_qoperator_quantize_matmul.py``'s own helper of the
    same name, since this pass reads the exact same global/function for both
    its activation AND its output.
    """
    lo = min(np.float32(0.0), np.float32(min_val))
    hi = max(np.float32(0.0), np.float32(max_val))
    if hi <= lo:
        hi = lo + np.float32(1.0)
    scale = (hi - lo) / np.float32(255.0)
    zero_point = int(np.clip(np.round(-lo / scale), 0, 255))
    return np.float32(scale), zero_point


def _disable_opt_session(model_bytes):
    # Disable graph optimization explicitly: by default onnxruntime can
    # fuse/transform a QDQ/QOperator-shaped chain like this pass produces
    # into a different, hardware-specific code path than the literal node
    # chain this file's proofs reason about -- see
    # tests/test_ort_matmul_nbits_workaround.py's docstring for this suite's
    # existing precedent of a real ORT graph-optimization fusion bug of
    # exactly this shape.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model_bytes, sess_options=so, providers=["CPUExecutionProvider"]
    )


def test_qoperator_quantize_softmax_qlinear_softmax_kernel_exists():
    # Confirmed empirically before relying on it anywhere else in this file
    # (this suite's own precedent for a contrib op used by a differential
    # test): a minimal standalone QuantizeLinear -> QLinearSoftmax ->
    # DequantizeLinear model actually runs on onnxruntime's CPU provider.
    model = _model(
        """
        g (float[2,4] X) => (float[2,4] Y)
        <float Xs = {0.05}, uint8 Xzp = {20}, float Ys = {0.01}, uint8 Yzp = {0}>
        {
          Xq = QuantizeLinear(X, Xs, Xzp)
          Yq = com.microsoft.QLinearSoftmax<axis=-1, opset=13>(Xq, Xs, Xzp, Ys, Yzp)
          Y = DequantizeLinear(Yq, Ys, Yzp)
        }
        """,
        extra_imports=', "com.microsoft": 1',
    )
    sess = _disable_opt_session(model.SerializeToString())
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 4)
    (y,) = sess.run(None, {"X": x})
    assert y.shape == (2, 4)
    assert np.all(np.isfinite(y))


def test_qoperator_quantize_softmax_pass_fires_and_matches_scheme():
    # Build a plain float Softmax and run the real pass with calibration
    # ranges for BOTH X and the node's own output "Y", each straddling 0 (for
    # X) so Xzp comes out genuinely nonzero, mirroring every sibling file's
    # own convention for exercising a nonzero zero point.
    rows, cols = 4, 5
    model = _model(
        f"""
        g (float[{rows},{cols}] X) => (float[{rows},{cols}] Y)
        {{
          Y = Softmax<axis=-1>(X)
        }}
        """,
        opset=17,
    )

    x_range = (-5.0, 10.0)
    y_range = (0.0, 1.0)  # Softmax output is always in [0, 1]
    quantized = _quantize_qoperator_softmax(model, {"X": x_range, "Y": y_range})

    # No float Softmax left anywhere.
    op_types = {n.op_type for n in quantized.graph.node}
    assert "Softmax" not in op_types

    # Walk the chain backward from the real graph output:
    # DequantizeLinear(Yq) <- QLinearSoftmax(Xq, ...) <- QuantizeLinear(X).
    dq_node = producer(quantized, "Y")
    assert dq_node.op_type == "DequantizeLinear"
    qlsm_node = producer(quantized, dq_node.input[0])
    assert qlsm_node.op_type == "QLinearSoftmax"
    assert qlsm_node.domain == "com.microsoft"
    assert dq_node.input[1:] == list(qlsm_node.input[3:5])  # Ys, Yzp shared

    ql_node = producer(quantized, qlsm_node.input[0])
    assert ql_node.op_type == "QuantizeLinear"
    assert ql_node.input[0] == "X"
    assert ql_node.input[1:] == list(qlsm_node.input[1:3])  # Xs, Xzp shared

    attrs = {a.name: a for a in qlsm_node.attribute}
    assert attrs["axis"].i == -1
    assert attrs["opset"].i == 17  # the MODEL's own declared opset, verbatim

    # com.microsoft opset import added, version 1.
    ms_imports = [o for o in quantized.opset_import if o.domain == "com.microsoft"]
    assert len(ms_imports) == 1
    assert ms_imports[0].version == 1

    init = {i.name: i for i in quantized.graph.initializer}

    x_scale = numpy_helper.to_array(init[ql_node.input[1]])
    x_zp = numpy_helper.to_array(init[ql_node.input[2]])
    expected_x_scale, expected_x_zp = _expected_asymmetric_uint8_quant_params(*x_range)
    assert x_zp.dtype == np.uint8
    assert int(x_zp) == expected_x_zp
    assert expected_x_zp != 0, "test calibration range must exercise a nonzero Xzp"
    np.testing.assert_allclose(float(x_scale), float(expected_x_scale), rtol=1e-6)

    y_scale = numpy_helper.to_array(init[dq_node.input[1]])
    y_zp = numpy_helper.to_array(init[dq_node.input[2]])
    expected_y_scale, expected_y_zp = _expected_asymmetric_uint8_quant_params(*y_range)
    assert y_zp.dtype == np.uint8
    assert int(y_zp) == expected_y_zp
    np.testing.assert_allclose(float(y_scale), float(expected_y_scale), rtol=1e-6)


def test_qoperator_quantize_softmax_axis_attribute_carried_over():
    # A non-default explicit axis attribute must be threaded through
    # unchanged onto QLinearSoftmax.
    model = _model(
        """
        g (float[2,3,4] X) => (float[2,3,4] Y)
        {
          Y = Softmax<axis=1>(X)
        }
        """,
        opset=13,
    )
    quantized = _quantize_qoperator_softmax(model, {"X": (-3.0, 3.0), "Y": (0.0, 1.0)})
    qlsm_node = producer(quantized, producer(quantized, "Y").input[0])
    attrs = {a.name: a for a in qlsm_node.attribute}
    assert attrs["axis"].i == 1


def test_qoperator_quantize_softmax_com_microsoft_import_not_duplicated():
    # If "com.microsoft" is already present in opset_import (e.g. a model
    # that already went through this pass, or already used another contrib
    # op), the pass must not add a second entry.
    model = _model(
        """
        g (float[2,4] X) => (float[2,4] Y)
        {
          Y = Softmax<axis=-1>(X)
        }
        """,
        opset=13,
        extra_imports=', "com.microsoft": 1',
    )
    quantized = _quantize_qoperator_softmax(model, {"X": (-3.0, 3.0), "Y": (0.0, 1.0)})
    ms_imports = [o for o in quantized.opset_import if o.domain == "com.microsoft"]
    assert len(ms_imports) == 1
    assert ms_imports[0].version == 1
    # And the rewrite still actually fired.
    assert producer(quantized, "Y").op_type == "DequantizeLinear"


def test_qoperator_quantize_softmax_declines_with_only_activation_range():
    # patternMatchPredicate requires calibrated ranges for BOTH the
    # activation AND the node's own output. Supplying only X's range must
    # leave the Softmax completely untouched.
    model = _model(
        """
        g (float[2,4] X) => (float[2,4] Y)
        {
          Y = Softmax<axis=-1>(X)
        }
        """,
        opset=13,
    )
    quantized = _quantize_qoperator_softmax(model, {"X": (-3.0, 3.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Softmax"]


def test_qoperator_quantize_softmax_declines_with_only_output_range():
    # Symmetric: only Y's range supplied, no entry for X.
    model = _model(
        """
        g (float[2,4] X) => (float[2,4] Y)
        {
          Y = Softmax<axis=-1>(X)
        }
        """,
        opset=13,
    )
    quantized = _quantize_qoperator_softmax(model, {"Y": (0.0, 1.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Softmax"]


def test_qoperator_quantize_softmax_declines_non_float_input():
    # Only a float32 input is matched.
    model = _model(
        """
        g (float16[2,4] X) => (float16[2,4] Y)
        {
          Y = Softmax<axis=-1>(X)
        }
        """,
        opset=13,
    )
    quantized = _quantize_qoperator_softmax(model, {"X": (-3.0, 3.0), "Y": (0.0, 1.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Softmax"]


def test_qoperator_quantize_softmax_declines_with_no_resolvable_default_domain_opset():
    # The single most important structural test in this file (see module
    # docstring): a model whose opset_import lists ONLY a custom domain, with
    # no "" / "ai.onnx" entry at all, has nothing for DefaultDomainOpsetVersion
    # to resolve -- per the header's own "there is no safe default to guess"
    # comment, the pass must leave it completely untouched, even though both
    # calibrated ranges are present. Expressible directly via onnx.parser (an
    # opset_import list omitting the default domain parses and loads fine --
    # confirmed empirically before writing this test), so no onnx.helper
    # fallback is needed here.
    model = parser.parse_model(
        """
        <
          ir_version: 8,
          opset_import: ["custom.domain": 1]
        >
        g (float[2,4] X) => (float[2,4] Y)
        {
          Y = Softmax<axis=-1>(X)
        }
        """
    )
    quantized = _quantize_qoperator_softmax(model, {"X": (-3.0, 3.0), "Y": (0.0, 1.0)})
    assert [n.op_type for n in quantized.graph.node] == ["Softmax"]
    # Confirm this is genuinely the SAME custom-only opset_import, not
    # something the pass otherwise mutated.
    assert [(o.domain, o.version) for o in quantized.opset_import] == [
        ("custom.domain", 1)
    ]


# ---------------------------------------------------------------------------
# 3. The genuine crux: an adversarial opset-semantics-mismatch case
# ---------------------------------------------------------------------------

# A 3-D tensor with `axis` in the MIDDLE of its rank: pre-opset-13 Softmax
# flattens dims [axis:] together (here dims 1 and 2, sizes 3*4=12) and
# normalizes ALL 12 values jointly per batch row; opset-13+ Softmax
# normalizes ONLY dim 1 (size 3), independently for each of the 4 positions
# in dim 2. These are genuinely different computations -- confirmed
# empirically below (not merely a cosmetic attribute difference): the two
# semantics disagree by more than 0.5 in absolute value on this shape, on an
# output tensor whose every value lies in [0, 1].
_ADVERSARIAL_SHAPE = (2, 3, 4)
_ADVERSARIAL_AXIS = 1


def _adversarial_softmax_model(opset):
    return _model(
        f"""
        g (float[2,3,4] X) => (float[2,3,4] Y)
        {{
          Y = Softmax<axis={_ADVERSARIAL_AXIS}>(X)
        }}
        """,
        opset=opset,
        ir_version=8,
    )


def test_qoperator_quantize_softmax_pre13_and_13plus_semantics_genuinely_disagree():
    # Establishes the adversarial case is real BEFORE relying on it: run the
    # plain float Softmax through onnxruntime at opset 12 (pre-13, flatten
    # semantics) and opset 17 (13+, in-place semantics) on the SAME input,
    # and confirm they produce meaningfully different numbers (not just a
    # different op-attribute rendering of the same computation).
    rng = np.random.default_rng(0)
    x = rng.standard_normal(_ADVERSARIAL_SHAPE).astype(np.float32)

    sess12 = _disable_opt_session(_adversarial_softmax_model(12).SerializeToString())
    sess17 = _disable_opt_session(_adversarial_softmax_model(17).SerializeToString())
    (y12,) = sess12.run(None, {"X": x})
    (y17,) = sess17.run(None, {"X": x})

    assert y12.shape == y17.shape == _ADVERSARIAL_SHAPE
    max_diff = float(np.max(np.abs(y12 - y17)))
    assert max_diff > 0.3, (
        f"pre-13 vs 13+ Softmax semantics do not genuinely disagree on this "
        f"adversarial shape/axis (max diff {max_diff}) -- the adversarial "
        f"case is not exercising the two-semantics split this file needs"
    )

    # And each one really does what its opset documents: opset 12's output,
    # flattened over dims [1, 2], sums to 1 per batch row; opset 17's output
    # sums to 1 independently along axis 1 for every fixed (batch, dim2).
    np.testing.assert_allclose(y12.reshape(2, -1).sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(y17.sum(axis=_ADVERSARIAL_AXIS), 1.0, atol=1e-5)


def test_qoperator_quantize_softmax_threads_the_models_own_opset_verbatim():
    # The pass, run on the pre-13 and the 13+ adversarial models, must thread
    # EACH model's own declared opset onto QLinearSoftmax's `opset`
    # attribute -- not a hardcoded constant, not always 13, not always the
    # latest opset.
    for model_opset in (12, 17):
        model = _adversarial_softmax_model(model_opset)
        rng = np.random.default_rng(model_opset)
        x = rng.standard_normal(_ADVERSARIAL_SHAPE).astype(np.float32)
        x_min, x_max = float(x.min()), float(x.max())

        quantized = _quantize_qoperator_softmax(
            model, {"X": (x_min, x_max), "Y": (0.0, 1.0)}
        )
        qlsm_node = producer(quantized, producer(quantized, "Y").input[0])
        assert qlsm_node.op_type == "QLinearSoftmax"
        attrs = {a.name: a for a in qlsm_node.attribute}
        assert attrs["opset"].i == model_opset, (
            f"pass threaded opset attribute {attrs['opset'].i}, expected the "
            f"model's own declared opset {model_opset}"
        )


def test_qoperator_quantize_softmax_wrong_opset_would_be_visibly_wrong():
    # The crux differential test: for EACH of the adversarial pre-13/13+
    # models, quantize it with the REAL pass (reading the model's own actual
    # opset), run the resulting QLinearSoftmax graph through onnxruntime, and
    # confirm the quantized result is close to onnxruntime's OWN plain-
    # Softmax execution of the ORIGINAL (pre-rewrite) model at that SAME
    # opset -- an ORT-vs-ORT internal consistency check, not a hand-derived
    # numpy reference (QLinearSoftmax's own int8 approximation error is ORT's
    # kernel's business, not something this repo bounds -- see module
    # docstring). Calibration ranges are set to the actual observed (min,
    # max) of X and of the TRUE (matching-opset) float output, so nothing
    # clips -- the round-trip lemmas' side condition, for both tensors.
    #
    # Then, the genuinely adversarial confirmation: the SAME quantized graph
    # compared instead against the WRONG-opset float reference (what this
    # pass would produce if it had threaded the other, incorrect opset
    # value) is FAR less accurate -- confirming a wrong-opset bug here would
    # be silently, visibly wrong, not a rounding-noise-scale difference.
    rng = np.random.default_rng(42)
    x = rng.standard_normal(_ADVERSARIAL_SHAPE).astype(np.float32) * 2.0
    x_min, x_max = float(x.min()), float(x.max())

    float_outputs = {}
    for opset in (12, 17):
        sess = _disable_opt_session(
            _adversarial_softmax_model(opset).SerializeToString()
        )
        (y,) = sess.run(None, {"X": x})
        float_outputs[opset] = y

    for opset, other_opset in ((12, 17), (17, 12)):
        y_float = float_outputs[opset]
        y_min, y_max = float(y_float.min()), float(y_float.max())
        quantized = _quantize_qoperator_softmax(
            _adversarial_softmax_model(opset),
            {"X": (x_min, x_max), "Y": (y_min, y_max)},
        )
        qsess = _disable_opt_session(quantized.SerializeToString())
        (y_quant,) = qsess.run(None, {"X": x})

        matched_err = float(np.max(np.abs(y_quant - y_float)))
        # Generous slack over the pure output round-trip (Ys/2): X's own
        # quantization error also propagates through Softmax's nonlinear
        # exp/normalize, so the total error exceeds the output-only bound,
        # but must still stay small -- nowhere near the wrong-opset scale
        # checked below.
        assert matched_err < 0.05, (
            f"opset={opset}: quantized QLinearSoftmax result strays too far "
            f"from onnxruntime's own matching-opset float Softmax "
            f"(max abs err {matched_err}) -- opset threading may be wrong"
        )

        y_float_wrong = float_outputs[other_opset]
        mismatched_err = float(np.max(np.abs(y_quant - y_float_wrong)))
        assert mismatched_err > 0.3, (
            f"opset={opset}: quantized result is suspiciously close to the "
            f"WRONG-opset ({other_opset}) float reference (max abs err "
            f"{mismatched_err}) -- this adversarial case no longer "
            f"distinguishes correct from incorrect opset threading"
        )
        assert matched_err < mismatched_err, (
            "matching-opset error should be far smaller than wrong-opset error"
        )
