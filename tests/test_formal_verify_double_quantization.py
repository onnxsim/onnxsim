"""Formal check for DoubleQuantization (opt-in; onnxsim's own
``onnxsim/passes/double_quantization.h``) -- QLoRA-style double
quantization (Dettmers et al., 2023, "QLoRA: Efficient Finetuning of
Quantized LLMs", Section 3.2). READ THE HEADER IN FULL FIRST.

This pass is STRUCTURALLY UNLIKE every other pass in this suite: it has no
"live float weight" to quantize at all. It is a second pass over an
*already-quantized* graph, run once per ``DequantizeLinear`` node whose
scale input (input 1) is a constant FLOAT32 tensor with at least
``kMinElements = 64`` values (confirmed directly from the header:
``static constexpr int64_t kMinElements = 64;`` and
``MatchDoubleQuantizationCandidate``'s own ``numel < min_elements`` decline
-- a boundary test below confirms 63 declines and 64 fires, exactly). It
rewrites::

    Before:  Whatever = DequantizeLinear(Codes, Scale, ...)   -- Scale: constant, >= 64 values
    After:   ScaleCodes: initializer, uint8, same shape as Scale
             MetaScale:  initializer, float32 scalar, max(|Scale|) / 255
             ScaleHat  = DequantizeLinear(ScaleCodes, MetaScale)
             Whatever  = DequantizeLinear(Codes, ScaleHat, ...)  -- attrs unchanged

Confirmed from ``runTransform`` line by line, not assumed:

* The inner scheme is genuinely SYMMETRIC UINT8, zero_point implicitly 0 --
  ``codes[i] = clip(round(scale_data[i] / meta_scale), 0, 255)`` with no
  offset, and the inner ``DequantizeLinear`` is built with only two inputs
  (``ScaleCodes``, ``MetaScale``), no zero-point input at all. This is
  sound specifically *because* ``scale_data`` (an absmax-derived
  quantization scale) is always non-negative -- the header's own comment
  ("always non-negative, so a plain unsigned 0..255 range needs no
  zero-point offset") is confirmed here, not assumed.
* ``meta_scale = max(|Scale|) / 255`` exactly (``std::max(max_abs, 1e-12)``
  only guards a degenerate all-zero ``Scale``, irrelevant whenever
  ``max(|Scale|) > 0`` as in every test below).
* Only ``n``'s (the original node's) scale input (input index 1) is
  replaced (``n->replaceInput(1, inner_dq->output())``) -- every other
  input (codes, zero-point) and every attribute on the OUTER node is
  untouched, confirmed structurally below with a non-trivial zero-point.
* The fixed-point argument in the header comment holds up: the rewritten
  ``ScaleHat`` is a node output, not a constant initializer, so
  ``FetchConstantTensor`` (``pass_util.h``) never matches it a second
  time; the inner node's own scale (``MetaScale``) is a scalar (1 value),
  always under ``kMinElements = 64`` -- confirmed empirically below via a
  byte-identical second application.
* The original float32 ``Scale`` initializer is left in the graph,
  unreferenced by any node (dead code), exactly as the header says --
  confirmed below by checking it is still present in
  ``graph.initializer`` but is not consumed by any node's input list.

**Entry point.** ``double_quantization`` IS registered as an opt-in
("other") pass -- ``"double_quantization" in
onnxsim.onnxsim_cpp2py_export._list_other_optimizers()`` is ``True`` -- so
``simplify_isolated_extra(model, "double_quantization")`` (this pass needs
no extra runtime parameters, unlike calibration-range-based passes) DOES
fire the real rewrite; a smoke test below confirms this route works, for
parity with every other opt-in-pass file in this suite. But
``simplify_isolated_extra`` goes through ``onnxsim.simplify()``'s full
wrapper, which does its own general cleanup independent of
``skipped_optimizers`` -- empirically confirmed to prune the very
now-unreferenced ``Scale`` initializer this file's own header-fidelity
claim needs to observe, and to rename every value on each call so a
second ``simplify_isolated_extra`` call is only structurally, not
byte-for-byte, identical to the first. So every structural/differential
test below instead uses ``onnxsim.apply_double_quantization_cpp``
(``onnx_simplifier.py``'s dedicated Python-visible port of
``ApplyDoubleQuantization``, ``quantize_entry.cpp``, which calls
``OptimizeFixed`` with only this one pass name and nothing else) --
mirroring ``test_double_quantization_cpp.py``'s own convention for this
exact pass. It runs the rewrite alone with no other cleanup, which is
exactly what lets the dead-initializer and byte-identical-fixed-point
claims below be checked directly against the pass itself, not against
``simplify()``'s incidental behavior around it.

**The genuinely new claim this file proves.** Every other pass in this
suite bounds error against a LIVE float value being quantized for the
first time. Here there is no such value: ``Scale`` is already just a
number, and this pass introduces a SECOND, independent rounding error on
top of whatever error the ORIGINAL quantization scheme (that produced
``Codes`` and ``Scale`` in the first place) already had. This file does
NOT reprove that original scheme's own round-trip bound (every other
formal-verify file in this suite already covers its own scheme -- see
``test_formal_verify_quantize_round_trip.py``'s own module docstring for
the base affine lemma). Instead it isolates and bounds EXACTLY this
pass's own new contribution:

    escale        := Scale[i] - ScaleHat[i]                     (bounded by MetaScale/2, symmetric)
    new_error[i]  := Whatever_after[i] - Whatever_with_exact_Scale[i]
                   = Codes_adj[i] * ScaleHat[i] - Codes_adj[i] * Scale[i]
                   = -Codes_adj[i] * escale[i]                  (Codes_adj := Codes - ZeroPoint, held fixed)

so ``|new_error[i]| = |Codes_adj[i] * escale[i]| <= |Codes_adj[i]| *
(MetaScale / 2)``, and since ``MetaScale = max(|Scale|) / 255``, the
WORST CASE over the whole tensor is ``max(|Codes_adj|) * max(|Scale|) /
510`` -- independent of which technique produced ``Codes``/``Scale`` in
the first place (only their own magnitudes matter), making the header's
own "technique-agnostic" claim quantitative. A composition lemma then
shows how this NEW error stacks (by the triangle inequality, abstractly --
not by reproving any one scheme's own bound formula) with whatever error
budget the original scheme already had.

Differential tests build a standalone ``DequantizeLinear(Codes, Scale,
ZeroPoint)`` via ``onnx.parser`` (per ``CLAUDE.md``) with an ASYMMETRIC,
non-trivial per-element ``ZeroPoint`` on the OUTER node (genuinely
different from the pass's own always-symmetric inner scheme) and a
varied, non-constant ``Scale`` with >= 64 elements, run the real compiled
pass via ``apply_double_quantization_cpp``, and confirm the node chain,
``ScaleCodes``/``MetaScale``'s exact values (against an independent numpy
re-implementation), the untouched outer attributes/zero-point, the dead
``Scale`` initializer, the exact fixed point, and -- via a real
onnxruntime execution with graph optimization explicitly disabled (this
suite's now-established default; see
``tests/test_ort_matmul_nbits_workaround.py``'s docstring for the ORT
fusion-bug precedent) -- that the true numeric error against the
exact-``Scale`` computation stays within the bound proved above,
including the worst-case-over-the-tensor equality.
"""

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated_extra, z3
from onnx import parser

import onnxsim

_MIN_ELEMENTS = 64  # DoubleQuantization::kMinElements (double_quantization.h)


def _abs(v):
    return z3.If(v >= 0, v, -v)


# ---------------------------------------------------------------------------
# Z3 proofs
# ---------------------------------------------------------------------------


def test_double_quantization_scale_round_trip_is_sound():
    # The base round-trip lemma reused in THIS file's own vocabulary
    # (Scale/ScaleHat/MetaScale/ScaleCodes), for THIS pass's own specific
    # scheme: symmetric UINT8 (zero_point implicitly 0 -- omitted entirely
    # from the inner DequantizeLinear, confirmed from runTransform), so the
    # bound is the plain scale/2 formula, not the asymmetric affine
    # zero-point-half-range formula an affine scheme would need. As in
    # test_formal_verify_quantize_round_trip.py, ``n`` models "*some*
    # integer within 0.5 of Scale / MetaScale" -- true for any correct
    # nearest-integer rounding rule -- and no saturation is assumed (a side
    # condition, matching that file's own).
    scale, meta_scale = z3.Reals("Scale MetaScale")
    n = z3.Int("ScaleCodes")
    half = z3.RealVal(1) / 2

    hypotheses = z3.And(
        meta_scale > 0,
        n - scale / meta_scale <= half,
        scale / meta_scale - n <= half,
    )
    # ScaleHat = DequantizeLinear(ScaleCodes, MetaScale): zero_point
    # omitted, i.e. always 0 -- confirmed from the header's own comment and
    # runTransform's own inner_dq construction (only 2 inputs).
    scale_hat = z3.ToReal(n) * meta_scale
    escale = scale - scale_hat

    prove(
        z3.Implies(
            hypotheses, z3.And(escale <= meta_scale / 2, -escale <= meta_scale / 2)
        )
    )


def test_double_quantization_new_error_is_bounded_by_codes_times_half_metascale():
    # The pass's own NEW error, isolated: Whatever = Codes_adj * Scale_used
    # (DequantizeLinear is linear in its own scale input, with Codes_adj :=
    # Codes - ZeroPoint held completely fixed by this pass -- only the
    # scale changes). Using the direct-error-variable idiom (escale free,
    # bounded by the rounding hypothesis directly, rather than
    # reconstructed from a code/meta_scale product inside this same query --
    # see test_formal_verify_dynamic_quantize_matmul.py's own module
    # docstring for why that reconstruction is a documented Z3 hang risk)
    # to isolate exactly the multiplication this pass's own error
    # introduces.
    codes_adj, meta_scale, escale = z3.Reals("CodesAdj MetaScale escale")
    rounding_bound = z3.And(meta_scale > 0, _abs(escale) <= meta_scale / 2)

    new_error = -codes_adj * escale  # Whatever_after - Whatever_with_exact_Scale
    bound = _abs(codes_adj) * (meta_scale / 2)

    prove(z3.Implies(rounding_bound, z3.And(new_error <= bound, -new_error <= bound)))


def test_double_quantization_worst_case_tensor_error_bound():
    # The genuinely quantitative, "technique-agnostic" conclusion: given
    # this element's own |Codes_adj| bounded by the tensor-wide max
    # |Codes_adj| ever takes, and MetaScale defined (this pass's own
    # formula) as max(|Scale|) / 255, the worst-case additional error this
    # pass introduces anywhere in the tensor is max(|Codes_adj|) *
    # max(|Scale|) / 510 -- bounded purely via Codes's and Scale's own
    # magnitudes, regardless of which original scheme produced them.
    codes_adj, escale, meta_scale, max_codes, max_scale = z3.Reals(
        "CodesAdj escale MetaScale MaxCodesAdj MaxScale"
    )
    hypotheses = z3.And(
        max_scale >= 0,
        max_codes >= 0,
        meta_scale == max_scale / 255,  # this pass's own MetaScale formula
        _abs(codes_adj) <= max_codes,
        _abs(escale) <= meta_scale / 2,  # the round-trip lemma above
    )
    new_error = -codes_adj * escale
    worst_case_bound = max_codes * max_scale / 510

    prove(
        z3.Implies(
            hypotheses,
            z3.And(new_error <= worst_case_bound, -new_error <= worst_case_bound),
        )
    )


def test_double_quantization_worst_case_bound_equality_simplifies():
    # A standalone algebraic lemma for the simplification itself (not just
    # prose): (max_codes * (max_scale / 255)) / 2 is EXACTLY max_codes *
    # max_scale / 510 -- i.e. "half of max(|Scale|)/255" and
    # "max(|Scale|)/510" are the same bound, for every real max_codes,
    # max_scale (an unconditional algebraic identity, needing no sign
    # hypotheses at all).
    max_codes, max_scale = z3.Reals("MaxCodesAdj MaxScale")
    prove((max_codes * (max_scale / 255)) / 2 == max_codes * max_scale / 510)


def test_double_quantization_negative_control_requires_escale_rounding_bound():
    # Sanity check the bound above is genuine, not vacuous: with no error
    # budget assumed on escale at all (only MetaScale > 0), the same bound
    # is not a theorem -- Z3 must find a real counterexample.
    codes_adj, meta_scale, escale = z3.Reals("CodesAdj MetaScale escale")
    new_error = -codes_adj * escale
    bound = _abs(codes_adj) * (meta_scale / 2)

    solver = z3.Solver()
    solver.add(meta_scale > 0)
    solver.add(z3.Not(z3.And(new_error <= bound, -new_error <= bound)))
    assert solver.check() == z3.sat, (
        "the bound holds even without any rounding-error budget on escale -- "
        "negative control is vacuous"
    )


def test_double_quantization_composes_with_original_scheme_error():
    # How this pass's own NEW error stacks with whatever error the
    # ORIGINAL quantization scheme already had against the true float
    # value W -- abstractly, via the triangle inequality, WITHOUT
    # reproving any one scheme's own bound formula (that is each such
    # scheme's own formal-verify file's job, e.g.
    # test_formal_verify_quantize_round_trip.py): eps_orig := W -
    # Dequant_exact (using the exact, undegraded Scale) is bounded by some
    # abstract bound_orig >= 0; this pass's own new_error is bounded by
    # |Codes_adj| * (MetaScale/2) as proved above. The combined error
    # against the true float W, using this pass's own rewritten graph
    # (ScaleHat instead of Scale), cannot exceed their sum.
    w, dequant_exact, dequant_final = z3.Reals("W DequantExact DequantFinal")
    codes_adj, meta_scale, escale, eps_orig, bound_orig = z3.Reals(
        "CodesAdj MetaScale escale eps_orig bound_orig"
    )

    hypotheses = z3.And(
        meta_scale > 0,
        bound_orig >= 0,
        _abs(escale) <= meta_scale / 2,  # this pass's own round-trip lemma
        _abs(eps_orig) <= bound_orig,  # the ORIGINAL scheme's own (abstract) bound
        eps_orig == w - dequant_exact,
        dequant_final == dequant_exact - codes_adj * escale,  # = Codes_adj * ScaleHat
    )
    total_error = w - dequant_final
    new_error_bound = _abs(codes_adj) * (meta_scale / 2)
    combined_bound = bound_orig + new_error_bound

    prove(
        z3.Implies(
            hypotheses,
            z3.And(total_error <= combined_bound, -total_error <= combined_bound),
        )
    )


# ---------------------------------------------------------------------------
# Differential / structural tests
# ---------------------------------------------------------------------------


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


def _dequantize_linear_model(n_elem, seed=0, with_zero_point=True):
    """A standalone, already-"post-quantization"-looking graph: a single
    DequantizeLinear(Codes, Scale[, ZeroPoint]) node whose Scale is a
    constant float32 tensor with ``n_elem`` varied, nonzero values (not
    all-equal, so max(|Scale|) is a genuine nontrivial bound), and an
    ASYMMETRIC, per-element ZeroPoint on the OUTER node -- deliberately
    different from this pass's own always-symmetric inner scheme, so the
    "outer attrs/zero-point untouched" claim is checked against a case
    that would actually break if the pass touched them.
    """
    rng = np.random.default_rng(seed)
    codes = rng.integers(-128, 128, size=(n_elem,)).astype(np.int8)
    scale = (rng.random(n_elem).astype(np.float32) * 3.0 + 0.01) * (
        1 + (np.arange(n_elem) % 7)
    ).astype(np.float32)
    initializer = [
        numpy_helper.from_array(codes, name="Codes"),
        numpy_helper.from_array(scale, name="Scale"),
    ]
    if with_zero_point:
        zero_point = rng.integers(-5, 6, size=(n_elem,)).astype(np.int8)
        initializer.append(numpy_helper.from_array(zero_point, name="ZeroPoint"))
        node_inputs = "Codes, Scale, ZeroPoint"
    else:
        node_inputs = "Codes, Scale"

    model = _model(
        f"""
        g (float[1] dummy) => (float[{n_elem}] Whatever)
        {{
          Whatever = DequantizeLinear<axis = 0>({node_inputs})
        }}
        """,
        initializer,
    )
    return model, codes, scale, (zero_point if with_zero_point else None)


def _reference_scale_codes(scale):
    """Independent numpy re-implementation of runTransform's own scale
    quantization: meta_scale = max(|scale|) / 255 (no zero-point, unsigned
    0..255 range), codes = round(scale / meta_scale) clipped to [0, 255].
    """
    scale64 = scale.astype(np.float64)
    meta_scale = max(float(np.abs(scale64).max()), 1e-12) / 255.0
    codes = np.clip(np.round(scale64 / meta_scale), 0, 255).astype(np.uint8)
    return codes, np.float32(meta_scale)


def test_double_quantization_smoke_via_simplify_isolated_extra():
    # Confirms the "real Python-visible entry point" question up front:
    # double_quantization IS registered as an opt-in pass, so the usual
    # simplify_isolated_extra route this suite's other opt-in-pass files
    # use DOES fire the rewrite (walking backward from the graph output,
    # since simplify_isolated_extra skips the default dead-code-
    # elimination pass, mirroring every other opt-in differential test in
    # this suite). Every other test below instead uses
    # apply_double_quantization_cpp -- see the module docstring for why.
    model, _codes, _scale, _zp = _dequantize_linear_model(96, seed=1)
    sim_model, _ops = simplify_isolated_extra(model, "double_quantization", check_n=0)

    outer = producer(sim_model, "Whatever")
    assert outer.op_type == "DequantizeLinear"
    inner = producer(sim_model, outer.input[1])
    assert inner.op_type == "DequantizeLinear"
    assert len(inner.input) == 2  # ScaleCodes, MetaScale -- no zero-point


def test_double_quantization_pass_fires_and_matches_scheme():
    n_elem = 96  # comfortably above kMinElements = 64
    model, codes, scale, zero_point = _dequantize_linear_model(n_elem, seed=7)
    onnx.checker.check_model(model)

    q = onnxsim.apply_double_quantization_cpp(model)
    onnx.checker.check_model(q)

    outer = next(
        n
        for n in q.graph.node
        if n.op_type == "DequantizeLinear" and "Whatever" in n.output
    )
    assert list(outer.input) == [
        "Codes",
        outer.input[1],
        "ZeroPoint",
    ]  # Codes/ZeroPoint untouched
    assert [(a.name, a.i) for a in outer.attribute] == [("axis", 0)]  # attrs untouched

    inner = next(n for n in q.graph.node if outer.input[1] in n.output)
    assert inner.op_type == "DequantizeLinear"
    assert len(inner.input) == 2  # ScaleCodes, MetaScale: no zero-point (symmetric)

    scale_codes_name, meta_scale_name = inner.input
    scale_codes_init = next(
        i for i in q.graph.initializer if i.name == scale_codes_name
    )
    meta_scale_init = next(i for i in q.graph.initializer if i.name == meta_scale_name)

    assert scale_codes_init.data_type == onnx.TensorProto.UINT8
    assert list(scale_codes_init.dims) == [n_elem]  # same shape as Scale
    assert meta_scale_init.data_type == onnx.TensorProto.FLOAT
    assert list(meta_scale_init.dims) == []  # scalar

    scale_codes = numpy_helper.to_array(scale_codes_init)
    meta_scale = float(numpy_helper.to_array(meta_scale_init))
    assert scale_codes.min() >= 0
    assert scale_codes.max() <= 255

    expected_codes, expected_meta_scale = _reference_scale_codes(scale)
    np.testing.assert_array_equal(scale_codes, expected_codes)
    np.testing.assert_allclose(meta_scale, expected_meta_scale, rtol=1e-6)
    np.testing.assert_allclose(meta_scale, np.abs(scale).max() / 255.0, rtol=1e-6)


def test_double_quantization_declines_scale_tensor_below_min_elements():
    # Exact boundary: kMinElements - 1 = 63 elements must decline outright.
    model, _codes, _scale, _zp = _dequantize_linear_model(_MIN_ELEMENTS - 1, seed=9)
    q = onnxsim.apply_double_quantization_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_double_quantization_fires_at_exactly_min_elements():
    # The other side of the same boundary: exactly kMinElements = 64
    # elements must fire (numel < min_elements is the decline condition,
    # so numel == min_elements does not decline).
    model, _codes, _scale, _zp = _dequantize_linear_model(_MIN_ELEMENTS, seed=10)
    q = onnxsim.apply_double_quantization_cpp(model)
    assert q.SerializeToString() != model.SerializeToString()
    op_types = [n.op_type for n in q.graph.node]
    assert op_types.count("DequantizeLinear") == 2


def test_double_quantization_declines_dynamic_scale_input():
    # A scale that is a graph INPUT (not a constant initializer) -- e.g.
    # any Value-style per-token/dynamic scale -- must be left untouched;
    # FetchConstantTensor only matches an actual constant initializer (or
    # a Constant node's own embedded value).
    model = _model(
        """
        g (float[80] Scale, int8[80] Codes) => (float[80] Whatever)
        {
          Whatever = DequantizeLinear<axis = 0>(Codes, Scale)
        }
        """
    )
    q = onnxsim.apply_double_quantization_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_double_quantization_noop_without_dequantize_linear():
    model = _model(
        """
        g (float[4,4] X) => (float[4,4] Y)
        {
          Y = Relu(X)
        }
        """
    )
    q = onnxsim.apply_double_quantization_cpp(model)
    assert q.SerializeToString() == model.SerializeToString()


def test_double_quantization_original_scale_initializer_left_dead():
    # The header's own note: the original float32 Scale initializer is
    # left in the graph, unreferenced, rather than removed.
    model, _codes, _scale, _zp = _dequantize_linear_model(96, seed=11)
    q = onnxsim.apply_double_quantization_cpp(model)

    assert "Scale" in {i.name for i in q.graph.initializer}
    used_as_input = {name for n in q.graph.node for name in n.input}
    assert "Scale" not in used_as_input


def test_double_quantization_is_a_fixed_point():
    # The header's own fixed-point argument, confirmed empirically: the
    # rewritten outer node's scale input (ScaleHat) is a node output, not
    # a constant initializer, so it never re-matches; the inner node's own
    # scale (MetaScale) is a scalar, always under kMinElements. Applying
    # the pass to its own output must be an EXACT no-op.
    model, _codes, _scale, _zp = _dequantize_linear_model(96, seed=12)
    once = onnxsim.apply_double_quantization_cpp(model)
    twice = onnxsim.apply_double_quantization_cpp(once)
    assert twice.SerializeToString() == once.SerializeToString()


def _disabled_opt_session(model_bytes):
    # onnxruntime's DEFAULT graph optimization level can silently fuse or
    # prune shapes differently than the plain node chain these proofs
    # reason about (see test_ort_matmul_nbits_workaround.py's docstring
    # for the precedent this suite found and fixed) -- disabled explicitly
    # here rather than relying on the default.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model_bytes, sess_options=so, providers=["CPUExecutionProvider"]
    )


def test_double_quantization_output_is_close_to_exact_scale_within_proved_bound():
    # The genuine numeric differential check: run the pass's own rewritten
    # graph through onnxruntime, and confirm every output element's error
    # against the value the ORIGINAL, exact (undegraded) Scale would have
    # produced stays within |Codes_adj| * (MetaScale / 2) -- the bound
    # test_double_quantization_new_error_is_bounded_by_codes_times_half_
    # metascale proves -- and that the worst-case-over-the-tensor bound
    # (max(|Codes_adj|) * max(|Scale|) / 510) matches the largest per-
    # element bound actually realized, tying the Z3 lemma above to real
    # values.
    n_elem = 96
    model, codes, scale, zero_point = _dequantize_linear_model(n_elem, seed=13)

    q = onnxsim.apply_double_quantization_cpp(model)
    outer = next(n for n in q.graph.node if "Whatever" in n.output)
    inner = next(n for n in q.graph.node if outer.input[1] in n.output)
    scale_codes_name, meta_scale_name = inner.input
    meta_scale_init = next(i for i in q.graph.initializer if i.name == meta_scale_name)
    meta_scale = float(numpy_helper.to_array(meta_scale_init))

    sess = _disabled_opt_session(q.SerializeToString())
    (whatever_final,) = sess.run(["Whatever"], {"dummy": np.zeros(1, dtype=np.float32)})

    codes_adj = codes.astype(np.float64) - zero_point.astype(np.float64)
    whatever_exact = codes_adj * scale.astype(
        np.float64
    )  # using the ORIGINAL, exact Scale
    error = np.abs(whatever_final.astype(np.float64) - whatever_exact)

    per_element_bound = np.abs(codes_adj) * (meta_scale / 2.0)
    assert np.all(error <= per_element_bound + 1e-6)

    worst_case_bound = np.abs(codes_adj).max() * np.abs(scale).max() / 510.0
    np.testing.assert_allclose(worst_case_bound, per_element_bound.max(), rtol=1e-6)
    assert error.max() <= worst_case_bound + 1e-6
