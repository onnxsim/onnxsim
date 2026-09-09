"""Formal check for FuseConsecutiveLogSoftmax (fuse_consecutive_log_softmax.h).

``patternMatchPredicate`` matches ``Log(Softmax(X, axis=k))`` -- a ``Log``
node whose sole input is the output of a ``Softmax`` node -- only when that
Softmax's output has exactly 1 use (``node->input()->uses().size() == 1``),
i.e. the Log is its sole consumer.

``runTransform`` creates a brand-new ``LogSoftmax`` node with the SAME
``axis`` attribute as the original Softmax, reading the SAME input ``X`` the
Softmax read; the old Log node's consumers are rewired onto the new
LogSoftmax node's output, and the old Log node is destroyed
(``NodeDestroyType::DestroyOne``). The old Softmax node itself is left
dangling (0 uses) -- not destroyed by this pass -- the same
dangling-producer pattern seen throughout this repo's other fusion-pass
formal-verify tests (cleanup is ``eliminate_deadend``'s job, a separate
default pass).

Unlike most passes in this fusion-pass family (mostly structural/index-
algebra arguments), this one rests on a genuine real-analysis identity:
``Log(Softmax(x)_i) == LogSoftmax(x)_i`` for every index ``i`` along the
softmax axis, because

* ``Softmax(x)_i = exp(x_i) / sum_j exp(x_j)``
* ``LogSoftmax(x)_i`` is DEFINED (per the ONNX LogSoftmax spec, and
  mathematically) as ``x_i - Log(sum_j exp(x_j))``
* ``Log(Softmax(x)_i) = Log(exp(x_i) / sum_j exp(x_j))
                       = Log(exp(x_i)) - Log(sum_j exp(x_j))``   [quotient law,
                                                                   valid since
                                                                   exp is always
                                                                   > 0]
                       ``= x_i - Log(sum_j exp(x_j))``            [Log(Exp(y)) == y]

  which is exactly ``LogSoftmax(x)_i``.

This is modeled below with an uninterpreted ``logsumexp`` real constant
standing in for ``Log(sum_j exp(x_j))`` -- the actual sum-over-exp
computation is never itself modeled, only that BOTH Softmax's own
denominator and LogSoftmax's own defining equation refer to the exact same
quantity ``sum_j exp(x_j)`` for the same input ``x`` and axis -- plus two
axioms capturing the two real-analysis laws actually used
(``ForAll([a, b], a>0 => b>0 => log(a/b) == log(a) - log(b))`` and
``ForAll([x], log(exp(x)) == x)``). The proof below derives the elementwise
identity from those two axioms plus each op's own defining equation, walking
through the ``log(a/b) = log(a) - log(b)`` step in Z3 exactly as the
source-level derivation above does, rather than asserting the two
definitions are equal outright. This is closer in spirit to
``test_formal_verify_fuse_bn_into_conv.py``'s handling of BatchNorm's
``sqrt``/variance side condition (a real piece of nontrivial algebra) than
to most of this repo's other "index remapping" proofs.
"""

import numpy as np
import onnxruntime as ort
from _formal_verify_common import producer, prove, simplify_isolated, z3
from onnx import parser


def test_fuse_consecutive_log_softmax_is_sound():
    log = z3.Function("log", z3.RealSort(), z3.RealSort())
    exp = z3.Function("exp", z3.RealSort(), z3.RealSort())

    # X(i) == x_i: an uninterpreted per-index reader of the (arbitrary,
    # symbolic) input tensor along the softmax axis.
    X = z3.Function("X", z3.IntSort(), z3.RealSort())
    i = z3.Int("i")

    # sum_exp == sum_j exp(x_j): the shared axis-reduction quantity both
    # Softmax's denominator and LogSoftmax's own logsumexp term refer to.
    # Its actual computation (the sum itself) is never modeled -- only that
    # both ops' defining equations name the exact same quantity.
    sum_exp = z3.Real("sum_exp")
    # logsumexp == Log(sum_j exp(x_j)): the uninterpreted stand-in
    # LogSoftmax's own defining equation uses, tied to sum_exp by the axiom
    # below (not asserted equal to the conclusion outright).
    logsumexp = z3.Real("logsumexp")

    def softmax_i(i):
        return exp(X(i)) / sum_exp  # Softmax(x)_i, ONNX Softmax's own definition

    def log_softmax_i(i):
        return X(i) - logsumexp  # LogSoftmax(x)_i, ONNX LogSoftmax's own definition

    # The two real-analysis laws the source-level derivation above actually
    # uses, instantiated directly at the concrete points this proof needs
    # (a=exp(X(i)), b=sum_exp; y=X(i)) rather than left as ForAll-quantified
    # axioms for Z3 to instantiate itself. prove()'s own implicit universal
    # quantification over every free variable here (i, sum_exp, logsumexp,
    # and X itself) already makes this argument general for every index and
    # every input -- a ForAll would only ask Z3 to re-derive, via its own
    # quantifier-instantiation heuristics, the single ground instantiation
    # this proof actually needs. Ground instantiation avoids a real
    # observed fragility: with the ForAll form, this same proof passed in
    # well under a second in isolation but, when run after many other
    # Solver() instances earlier in the same test session, occasionally took
    # far longer -- nonlinear real arithmetic combined with quantified
    # uninterpreted functions is a case where Z3's E-matching can be
    # sensitive to accumulated internal solver state. The ground form sidesteps
    # that instability entirely (mirroring the same fast-instantiation fix
    # test_formal_verify_eliminate_nop_reshape.py's own proof uses for an
    # unrelated ForAll-over-div/mod performance concern).
    log_quotient_here = log(exp(X(i)) / sum_exp) == log(exp(X(i))) - log(sum_exp)
    log_exp_here = log(exp(X(i))) == X(i)

    side_conditions = z3.And(
        sum_exp > 0,
        exp(X(i)) > 0,  # exp is always strictly positive -- the quotient law's a>0 need
        logsumexp == log(sum_exp),  # ties logsumexp to the SAME sum_exp quantity
    )

    consumer = z3.Function("consumer", z3.RealSort(), z3.RealSort())

    prove(
        z3.Implies(
            z3.And(log_quotient_here, log_exp_here, side_conditions),
            consumer(log(softmax_i(i))) == consumer(log_softmax_i(i)),
        )
    )


def _model(body, opset=13):
    return parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}]
        >
        {body}
        """
    )


def test_fuse_consecutive_log_softmax_pass_matches():
    model = _model(
        """
        g (float[2,3] X) => (float[2,3] Y)
        {
          s = Softmax<axis = 1>(X)
          Y = Log(s)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_log_softmax")
    assert ops["LogSoftmax"] == 1
    assert ops["Log"] == 0

    # The live output is produced directly by the new LogSoftmax node, fed
    # by the ORIGINAL input X (not by the old Softmax's output) and carrying
    # the SAME axis attribute the Softmax had.
    fused = producer(sim_model, "Y")
    assert fused.op_type == "LogSoftmax"
    assert list(fused.input) == ["X"]
    axis_attr = next(a.i for a in fused.attribute if a.name == "axis")
    assert axis_attr == 1

    # The old Softmax node is left dangling (0 uses) -- not itself destroyed
    # by this pass (eliminate_deadend is skipped by simplify_isolated).
    dangling = [n for n in sim_model.graph.node if n.output[0] == "s"]
    assert len(dangling) == 1 and dangling[0].op_type == "Softmax"

    # Numeric check: this is a genuine mathematical identity, so a mismatch
    # here would indicate a real bug, not just a structural difference.
    # simplify_isolated's own check_ok already asserted onnxsim's internal
    # equivalence check passed; run onnxruntime directly too so the exact
    # claim ("the output values are the same") is visible here.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3)).astype(np.float32)
    orig_sess = ort.InferenceSession(model.SerializeToString())
    fused_sess = ort.InferenceSession(sim_model.SerializeToString())
    (orig_out,) = orig_sess.run(None, {"X": x})
    (fused_out,) = fused_sess.run(None, {"X": x})
    np.testing.assert_allclose(orig_out, fused_out, rtol=1e-5, atol=1e-6)


def test_fuse_consecutive_log_softmax_declines_multi_use_softmax():
    # The Softmax's output also feeds a second graph output directly -- more
    # than 1 use, so the predicate declines: both Softmax and Log survive,
    # still chained.
    model = _model(
        """
        g (float[2,3] X) => (float[2,3] Y, float[2,3] S)
        {
          s = Softmax<axis = 1>(X)
          Y = Log(s)
          S = Identity(s)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_log_softmax")
    assert ops["LogSoftmax"] == 0
    assert ops["Softmax"] == 1
    assert ops["Log"] == 1

    log_node = producer(sim_model, "Y")
    assert log_node.op_type == "Log"
    assert log_node.input[0] == "s"
    softmax_node = producer(sim_model, log_node.input[0])
    assert softmax_node.op_type == "Softmax"
    assert list(softmax_node.input) == ["X"]


def test_fuse_consecutive_log_softmax_declines_non_softmax_producer():
    # Log's input is produced by Sigmoid, not Softmax -- CheckKind's kind
    # check fails outright, so the predicate declines without even looking
    # at use counts.
    model = _model(
        """
        g (float[2,3] X) => (float[2,3] Y)
        {
          s = Sigmoid(X)
          Y = Log(s)
        }
        """
    )
    sim_model, ops = simplify_isolated(model, "fuse_consecutive_log_softmax")
    assert ops["LogSoftmax"] == 0
    assert ops["Sigmoid"] == 1
    assert ops["Log"] == 1

    log_node = producer(sim_model, "Y")
    assert log_node.op_type == "Log"
    assert log_node.input[0] == "s"
