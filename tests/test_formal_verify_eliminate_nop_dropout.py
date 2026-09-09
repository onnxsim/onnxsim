"""Formal check for EliminateNopDropout (eliminate_nop_dropout.h).

Per ``third_party/onnx-optimizer/onnxoptimizer/passes/eliminate_nop_dropout.h``,
``patternMatchPredicate`` matches a Dropout node with a ``ratio`` *attribute*
(pre-opset-12 Dropout) equal to exactly ``0.0``, and ``runTransform`` rewires
every use of every output of the node (data output *and* the optional mask
output) directly to the node's single data input, then destroys the node --
mirroring ``eliminate_identity.h``'s handling of Identity.

**onnxsim ships and registers its own, different implementation of this
pass.** ``onnxsim/passes/eliminate_nop_dropout.h`` defines a second
``EliminateNopDropout`` under the same ``onnxsim_passes`` sub-namespace, and
``RegisterCustomOptimizerPasses()`` (called before every ``simplify()``, see
``onnxsim/onnxsim.cpp``) registers it via ``RegisterOrReplace`` keyed by
``getPassName()`` -- so it *replaces* the third_party pass of the same name in
the global registry. The version actually compiled into
``onnxsim_cpp2py_export`` and exercised by every test below is onnxsim's own,
not the third_party one. It differs in three material ways, all confirmed
empirically here (throwaway scripts, not committed) rather than assumed from
either header's comments:

1. **The mask-output question.** The third_party ``runTransform`` loops over
   *every* output of the Dropout node -- including the optional second "mask"
   output, which old-opset Dropout types as ``bool`` -- and calls
   ``tryReplacingAllUsesWith(output, node->input())`` for each. This is a real
   latent type-mismatch risk, not just a style nit:
   ``Value::replaceAllUsesWith`` (``third_party/onnx/onnx/common/ir.h``)
   propagates the *old* value's ``elemType`` onto the *new* value
   (``newValue->setElemType(this->elemType())``), so replacing a bool-typed
   mask's uses with the (shared, float) data input would retroactively stamp
   that shared input Value as ``bool`` -- corrupting every *other* use of the
   same input, not only the mask's former consumer. onnxsim's own
   ``patternMatchPredicate`` sidesteps this rather than risk it: it declines
   outright whenever the node has more than one output and that second
   (mask) output has any consumer at all
   (``node->outputs().size() > 1 && !node->outputs()[1]->uses().empty()``),
   and ``runTransform`` only ever rewires ``outputs()[0]``. Empirically (see
   the differential tests below): a two-output Dropout whose mask output has
   *zero* consumers -- declared but unused, which is the overwhelmingly
   common shape for an inference-mode ONNX export, since exporters keep the
   optional mask output around unconditionally -- is still eliminated, exactly
   like the single-output case. A two-output Dropout whose mask output *is*
   consumed by something is left untouched: the pass declines rather than
   apply the upstream loop's dubious rewrite. (Old-opset Dropout's mask
   output is unconditionally bool per its own ONNX schema regardless of what
   a graph output's declared type says -- onnxruntime's shape inference
   enforces this and rejects a float-typed consumer outright, discovered
   while building the differential test below -- so every realistic
   well-typed consumer of the mask *is* a type mismatch against the float
   data input, which is exactly the scenario the guard exists to avoid; the
   guard is phrased unconditionally, on "does the mask have any use", rather
   than trying to detect the mismatch itself.)

2. **The opset-12+ carve-out.** The third_party header's comment says the
   pass deliberately does *not* handle opset 12+ Dropout, where ``ratio``
   moved from an attribute to an input, "since it supports training-friendly
   models, for which the Dropout ops are required" -- ``hasAttribute(kratio)``
   is false for such nodes, so the third_party predicate never matches them.
   **This does not hold for onnxsim's own replacement.** Its
   ``patternMatchPredicate`` explicitly branches on ``hasAttribute(kratio)``:
   when absent (opset 12+), it instead requires input 1 (``ratio``) and input
   2 (``training_mode``) to each be omitted or a constant all-zero/all-false
   tensor, and eliminates the node when so. Differential tests below confirm
   this empirically: opset 13 Dropout with a constant-0.0 ``ratio`` input and
   omitted ``training_mode`` *is* eliminated by the real compiled pass, and a
   constant-``true`` ``training_mode`` input blocks it. This is a real,
   deliberate behavioral divergence from the upstream comment, not a bug in
   this test file -- it is worth calling out precisely because trusting the
   third_party comment (which correctly describes the third_party pass, only
   not the pass onnxsim actually runs) would have produced a wrong prediction
   for a "declines" test here.

3. **Only the plain ``value`` Constant attribute is recognized.** The
   opset-12+ path's "is this input a constant zero/false" check goes through
   ``IsConstantTensor``/``FetchConstantTensor``
   (``onnxoptimizer/passes/pass_util.h``), which only special-cases
   ``node->hasAttribute(kvalue)`` -- the plain ``value`` (Tensor) attribute of
   Constant. A ``Constant<value_float=0.0>()`` node is an equally valid,
   equally constant-0.0 ONNX Constant (per the Constant operator's own spec,
   which offers ``value``/``value_float``/``value_int``/etc. as mutually
   exclusive, equally legal ways to spell a constant), but is invisible to
   this pass -- confirmed empirically: swapping a passing test's Constant
   from ``value = float r = {0.0}`` to ``value_float = 0.0`` makes the
   Dropout survive untouched. Not unsound (declining is always safe, just
   suboptimal), but worth documenting since it shapes how the opset-12+
   differential tests below must construct their models.

Soundness of the underlying rewrite (once the predicate holds) rests on ONNX
Dropout's own reference semantics
(``onnx.reference.ops.op_dropout._dropout``): the output equals the input
whenever ``drop_probability == 0 or not training_mode``. For every opset this
pass's predicate actually matches -- pre-opset-12 attribute-form Dropout
(``training_mode`` cannot be expressed at all, so it is implicitly always
``False``) and opset-12+ Dropout with ``training_mode`` omitted-or-false --
that condition holds and Dropout is a pure, unconditional, pointwise identity
between its data output and its data input, regardless of tensor rank. The
proof below models a tensor abstractly as an uninterpreted function from
index (``Int``) to value (``Real``) -- rank 1 is enough, since nothing about
this argument depends on rank -- and an uninterpreted ``dropout(ratio, x)``
function standing in for one element's transformation, with the axiom
``ForAll([x], dropout(0.0, x) == x)`` (mirroring
``test_formal_verify_eliminate_nop_cast.py``'s ``cast_to_own_type_is_identity``
axiom). Composing with an arbitrary uninterpreted ``consumer`` proves
substitution safety for any downstream consumer of any one element, exactly
as the other nop-elimination proofs in this repo do.
"""

from _formal_verify_common import producer, prove, simplify_isolated, z3
from onnx import parser


def test_eliminate_nop_dropout_is_sound():
    i = z3.Int("i")  # arbitrary tensor index -- rank 1 is enough
    ratio = z3.Real("ratio")
    x = z3.Real("x")
    tensor = z3.Function("tensor", z3.IntSort(), z3.RealSort())  # the data input
    dropout = z3.Function("dropout", z3.RealSort(), z3.RealSort(), z3.RealSort())
    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    # General law: Dropout with ratio 0 (and, for every opset this pass's
    # predicate actually matches, training_mode implicitly or explicitly
    # false) drops nothing -- a value-preserving no-op, for every element
    # value x, independent of tensor rank or shape.
    dropout_at_zero_ratio_is_identity = z3.ForAll([x], dropout(0.0, x) == x)

    # patternMatchPredicate's actual hypothesis (attribute form): ratio == 0.
    predicate_holds = ratio == 0.0

    dropout_output_i = dropout(
        ratio, tensor(i)
    )  # Dropout<ratio>(tensor)'s value at index i
    prove(
        z3.Implies(
            z3.And(dropout_at_zero_ratio_is_identity, predicate_holds),
            consumer(dropout_output_i) == consumer(tensor(i)),
        )
    )


def test_eliminate_nop_dropout_pass_matches():
    # Baseline differential check: single-output Dropout<ratio=0.0> at an
    # old-enough opset for ratio to legally be an attribute (opset 11) -- the
    # compiled pass, run alone, removes it and rewires its consumer directly
    # to X.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 11]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Dropout<ratio = 0.0>(X)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_dropout")
    assert ops["Dropout"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_dropout_pass_matches_with_unused_mask_output():
    # Two-output Dropout<ratio=0.0> (data + mask), mask output declared but
    # with zero actual consumers -- the extremely common shape for an
    # inference-mode export. onnxsim's guard only blocks on the mask output
    # having a *use*, so this is still eliminated exactly like the
    # single-output case.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 11]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          a, mask = Dropout<ratio = 0.0>(X)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_dropout")
    assert ops["Dropout"] == 0
    assert ops["Relu"] == 1


def test_eliminate_nop_dropout_declines_when_mask_output_is_consumed():
    # Two-output Dropout<ratio=0.0> whose mask output IS consumed, here by a
    # correctly bool-typed ``Not`` (old-opset Dropout's mask output is
    # unconditionally bool per its ONNX schema regardless of what a graph
    # output's own declared type says, and onnxruntime's shape inference
    # enforces this -- confirmed while building this test: wiring the mask
    # into a float-only op like Relu, or declaring a float graph output type
    # for it, makes onnxruntime reject the model outright with "Type Error:
    # Type 'tensor(bool)' ... is invalid" before the pass is even relevant).
    # So this is a realistic, well-typed consumer, not a contrived one --
    # and onnxsim's patternMatchPredicate still declines unconditionally
    # whenever the mask output has any use at all, precisely to avoid the
    # upstream pass's dubious "rewire every output, including the mask,
    # straight to the float data input" rewrite documented above (which
    # *would* have produced an ill-typed graph here, feeding X's float
    # values into Not and silently retyping X's own Value as bool via
    # ``Value::replaceAllUsesWith``'s elemType propagation). The pass, run
    # alone, must leave the Dropout node untouched.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 11]
        >
        g (float[4,8] X) => (float[4,8] Y, bool[4,8] M)
        {
          a, mask = Dropout<ratio = 0.0>(X)
          Y = Relu(a)
          M = Not(mask)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_dropout")
    assert ops["Dropout"] == 1


def test_eliminate_nop_dropout_declines_on_nonzero_ratio():
    # Edge case from patternMatchPredicate: ratio attribute is 0.5, not 0, so
    # Dropout genuinely drops values -- not a no-op -- and the predicate
    # declines.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 11]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          a = Dropout<ratio = 0.5>(X)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_dropout")
    assert ops["Dropout"] == 1


def test_eliminate_nop_dropout_pass_matches_opset12_plus_ratio_input():
    # opset 13: ratio is an INPUT, not an attribute (a constant-0.0 tensor
    # here), training_mode is omitted (defaults to false per the ONNX
    # reference semantics). Contrary to the third_party header's comment --
    # which describes the third_party pass, not the one onnxsim actually
    # registers and runs -- onnxsim's own predicate does handle this opset
    # 12+ shape and eliminates it, since input 1 (ratio) is a constant zero
    # and input 2 (training_mode) is omitted.
    #
    # The Constant must use the plain ``value`` (Tensor) attribute, not the
    # ``value_float``/``value_int``/etc. scalar-attribute variants: onnxsim's
    # ``FetchConstantTensor`` (``onnxoptimizer/passes/pass_util.h``) only
    # special-cases ``node->hasAttribute(kvalue)``, so a
    # ``Constant<value_float=0.0>()`` node -- equally a valid, equally
    # constant-0.0 ONNX Constant -- is invisible to this pass's
    # ``IsConstantTensor``/``IsOmittedOrConstantZero`` and does *not* trigger
    # elimination (confirmed empirically: swapping this Constant's attribute
    # from ``value = float r = {0.0}`` to ``value_float = 0.0`` makes this
    # exact test fail, the Dropout node surviving untouched). Also note ONNX
    # Runtime's own shape inference rejects a non-scalar ``ratio`` input
    # ("Ratio of Dropout must be a scalar"), so the tensor literal below is
    # deliberately written with no dims (``float r = {0.0}``, a rank-0
    # scalar), not ``float[1] {0.0}``.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          ratio = Constant<value = float r = {0.0}>()
          a = Dropout<seed = 0>(X, ratio)
          Y = Relu(a)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "eliminate_nop_dropout")
    assert ops["Dropout"] == 0
    # eliminate_deadend is skipped too (only eliminate_nop_dropout is
    # active), so the now-dead ``ratio`` Constant is left lying around --
    # walk back from the real graph output to find the live computation
    # rather than overcounting via a raw op-type Counter.
    assert producer(sim_model, "Y").op_type == "Relu"


def test_eliminate_nop_dropout_declines_when_training_mode_input_is_true():
    # opset 13: ratio input is a constant 0.0, but training_mode is a
    # constant `true` input -- a genuine training-mode-capable Dropout that
    # must be preserved (matching the ONNX reference semantics: training_mode
    # true means the op can actually drop values at runtime, independent of
    # what ratio happens to be set to here). The predicate declines.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          ratio = Constant<value = float r = {0.0}>()
          training_mode = Constant<value = bool tm = {1}>()
          a = Dropout<seed = 0>(X, ratio, training_mode)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_dropout")
    assert ops["Dropout"] == 1


def test_eliminate_nop_dropout_declines_on_nonzero_ratio_input():
    # opset 13: ratio input is a constant 0.5 (training_mode omitted) -- a
    # genuine drop probability, not a no-op. The predicate declines.
    model = parser.parse_model(
        """
        <
          ir_version: 10,
          opset_import: ["": 13]
        >
        g (float[4,8] X) => (float[4,8] Y)
        {
          ratio = Constant<value = float r = {0.5}>()
          a = Dropout<seed = 0>(X, ratio)
          Y = Relu(a)
        }
        """
    )
    _, ops = simplify_isolated(model, "eliminate_nop_dropout")
    assert ops["Dropout"] == 1
