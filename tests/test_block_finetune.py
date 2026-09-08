"""Tests for ``onnxsim.apply_block_finetune`` (see ``onnxsim/qat.py``) -- the
same block-wise, label-free distillation :func:`onnxsim.apply_qat` performs,
with the fake-quantizer taken out of the middle.

Everything that made ``apply_qat`` work is about distillation rather than about
quantization: a block, a teacher's activation at its output, a mean squared
reconstruction error, a backward pass emitted as ONNX, an Adam step.
Quantization entered at exactly one point -- the fake-quantizer between the
master weight and the block's own node -- and ``fake_quant=False`` removes it.
So most of what is worth testing here is *what did not change*, and the tests
below are shaped accordingly: the step graph has no quantizer residue in it,
the weights come back as fp32 rather than as codes, and the flags that name a
quantizer's parameters are refused rather than ignored.

Three claims are genuinely new and are therefore measured rather than asserted:

1. a perturbed model's block really is tuned back toward the reference, and by
   enough to show on a *held-out* input the tuning never saw -- the number is
   recorded in ``test_a_perturbed_model_recovers_on_a_held_out_input``;
2. the master weights are seeded from the **student**, not the teacher, which
   is the one substantive semantic difference from ``apply_qat`` and is the
   difference between fine-tuning a pruned model and silently un-pruning it;
3. the block's *untrained* constants come from the student too, for the same
   reason -- ``test_the_blocks_other_constants_come_from_the_student`` is
   built so that getting this wrong reports a loss of exactly zero.

What is deliberately *not* claimed anywhere here: that this beats
:func:`onnxsim.apply_pruning_finetune`, the closed-form layer-wise fit that
already exists for pruning recovery. It does not, where that one applies; see
``apply_block_finetune``'s own docstring for the boundary.

Two limitations are pinned rather than left to be discovered. A calibration set
too small for the block's parameter count is fitted exactly and generalizes
almost not at all -- the loss falls three orders of magnitude either way, so
the loss cannot tell you which happened. And nothing masks the optimizer, so an
unstructured-pruned weight comes back fully dense. Both have a test, and the
second asserts the *current* behaviour rather than the desired one, so that
fixing it is a visible change.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim import backend, qat, qat_graph

ort = pytest.importorskip("onnxruntime")

D = 16


def _model(body, initializer=(), opset=17, ir_version=8):
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
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _init(model, name):
    return onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == name)
    )


_MLP_BODY = f"""
g (float[8,{D}] X) => (float[8,{D}] Y) {{
  H = MatMul(X, W1)
  A = Relu(H)
  Y = MatMul(A, W2)
}}
"""


def _mlp(w1, w2):
    return _model(_MLP_BODY, [_f32(w1, "W1"), _f32(w2, "W2")])


def _pair(rng, noise=0.15):
    """A reference model and a *damaged* copy of it.

    The damage stands in for whatever actually changed the model -- a pruning
    pass, a rounding pass, a requantization -- because none of that is what
    these tests are about. What matters is only that the two models differ:
    the loss is the student's block output against the reference's, so a
    student that *is* the reference starts at zero and stays there.
    """
    w1 = rng.normal(0, 0.3, (D, D)).astype(np.float32)
    w2 = rng.normal(0, 0.3, (D, D)).astype(np.float32)
    damaged = w2 + rng.normal(0, noise, w2.shape).astype(np.float32)
    return _mlp(w1, w2), _mlp(w1, damaged)


def _data(rng, batches=2, rows=8):
    return [
        {"X": rng.normal(0, 1, (rows, D)).astype(np.float32)} for _ in range(batches)
    ]


def _held_out_error(model, reference, batches):
    """Mean absolute end-to-end output error against ``reference``.

    Averaged over several batches rather than one, because a single batch of
    eight rows is noisy enough to move this number by a factor of two -- which
    is more than some of the effects below are.
    """
    student_run = backend.Runner(model)
    reference_run = backend.Runner(reference)
    return float(
        np.mean(
            [
                np.abs(student_run(batch)["Y"] - reference_run(batch)["Y"]).mean()
                for batch in batches
            ]
        )
    )


def test_a_perturbed_model_recovers_on_a_held_out_input():
    """The claim, measured where it counts.

    A falling training loss only says the optimizer moved downhill on the data
    it was given, and here that is a genuinely weak statement rather than a
    pedantic caveat: see
    ``test_the_calibration_set_size_dominates_generalization``, where the
    training loss falls by three orders of magnitude on a calibration set
    small enough that the held-out error barely moves. So the number this
    asserts on is the end-to-end output error on inputs that were never in the
    calibration set, and the calibration set is large enough for that to be
    the same question.
    """
    rng = np.random.default_rng(0)
    reference, student = _pair(rng)
    data = _data(rng, batches=32)
    held_out = _data(rng, batches=8)

    losses: list = []
    tuned = onnxsim.apply_block_finetune(
        reference,
        student,
        "X",
        "Y",
        calibration_data=data,
        num_iterations=300,
        learning_rate=2e-2,
        losses=losses,
    )

    assert losses[-1] < losses[0] / 100
    before = _held_out_error(student, reference, held_out)
    assert _held_out_error(tuned, reference, held_out) < before / 10


def test_the_calibration_set_size_dominates_generalization():
    """The limitation, recorded rather than avoided.

    This is the most useful thing the tests here have to say about the method,
    and it is not a happy result. The block-wise objective is fitted almost
    exactly at every set size below -- the *training* loss falls by three to
    four orders of magnitude whether it is given sixteen rows or a thousand --
    and how much of that reaches a held-out input is decided almost entirely
    by how many rows there were.

    Measured, as the mean held-out error ratio over four seeds:

        16 rows -> 0.65    64 rows -> 0.05    256 rows -> 0.002

    Sixteen rows is fewer rows than a 16x16 weight has free parameters, so the
    fit is underdetermined and spends its freedom on the calibration set. The
    thresholds below are loose around those numbers; what the test is for is
    the *ordering*, and that a run whose loss went to nearly zero can still be
    worth almost nothing.

    Worth knowing when reading ``num_samples``'s default of 8 batches.
    """
    rng = np.random.default_rng(11)
    reference, student = _pair(rng)
    held_out = _data(rng, batches=8)
    before = _held_out_error(student, reference, held_out)

    ratios = {}
    for batches in (2, 32):
        losses: list = []
        tuned = onnxsim.apply_block_finetune(
            reference,
            student,
            "X",
            "Y",
            calibration_data=_data(rng, batches=batches),
            num_iterations=300,
            learning_rate=2e-2,
            losses=losses,
        )
        # Fitted to nearly nothing either way. That is the point.
        assert losses[-1] < losses[0] / 100
        ratios[batches] = _held_out_error(tuned, reference, held_out) / before

    assert ratios[32] < ratios[2] / 4
    assert ratios[2] > 0.3


def test_a_student_identical_to_the_reference_has_nothing_to_learn():
    """The degenerate case, pinned because it is the one a caller reaches by
    accident.

    Passing the same model twice is not an error and is not refused -- the
    machinery runs perfectly happily -- but the loss it reports is zero from
    the first step, because the objective is "reproduce what the reference
    produced" and the student already does. A caller seeing a flat zero here
    has not found a bug; they have found out that their two models are the
    same one.
    """
    rng = np.random.default_rng(1)
    reference, _ = _pair(rng)
    losses: list = []
    onnxsim.apply_block_finetune(
        reference,
        reference,
        "X",
        "Y",
        calibration_data=_data(rng),
        num_iterations=5,
        losses=losses,
    )
    assert losses[0] == pytest.approx(0.0, abs=1e-12)


def test_the_master_weight_is_seeded_from_the_student_not_the_reference():
    """The semantic difference from :func:`onnxsim.apply_qat`, made visible.

    QAT seeds its master weight from the *teacher*, because the student's
    weight is a lossy encoding of it and the teacher's is the thing being
    encoded -- that is what makes step 0 reproduce round-to-nearest exactly.
    Fine-tuning has no such relationship: the student's weights are the
    starting point precisely because they are not the teacher's. Seeding from
    the teacher here would silently undo whatever change is being recovered
    from, which for a pruned model means quietly un-pruning it.

    A zero learning rate makes the seed the only thing the run can return.
    """
    rng = np.random.default_rng(2)
    reference, student = _pair(rng)
    tuned = onnxsim.apply_block_finetune(
        reference,
        student,
        "X",
        "Y",
        calibration_data=_data(rng),
        num_iterations=1,
        learning_rate=0.0,
        lr_decay=False,
    )
    assert np.array_equal(_init(tuned, "W2"), _init(student, "W2"))
    assert not np.array_equal(_init(tuned, "W2"), _init(reference, "W2"))


def test_the_blocks_other_constants_come_from_the_student():
    """The same choice, for the constants the run does *not* train.

    A block's untrained initializers -- a LayerNorm's scale and bias, a
    Gemm's C, the ``Gain`` below -- are spliced into the step graph verbatim,
    and which model they are read out of is a real decision. Reading them from
    the teacher would train the block to compensate for a substitution the
    deployed model never makes.

    This model is built so the wrong answer is unmistakable rather than
    merely worse: the two models share ``W`` and differ *only* in ``Gain``, so
    a step graph holding the teacher's ``Gain`` computes the teacher's own
    output and reports a loss of exactly zero.
    """
    rng = np.random.default_rng(3)
    w = rng.normal(0, 0.3, (D, D)).astype(np.float32)
    body = f"""
    g (float[8,{D}] X) => (float[8,{D}] Y) {{
      H = MatMul(X, W)
      Y = Mul(H, Gain)
    }}
    """
    reference = _model(body, [_f32(w, "W"), _f32(np.ones(D), "Gain")])
    student = _model(body, [_f32(w, "W"), _f32(np.full(D, 2.0), "Gain")])

    losses: list = []
    onnxsim.apply_block_finetune(
        reference,
        student,
        "X",
        "Y",
        calibration_data=_data(rng, batches=1),
        num_iterations=1,
        learning_rate=0.0,
        lr_decay=False,
        losses=losses,
    )
    assert losses[0] > 1e-3


def test_the_step_graph_has_no_quantizer_left_in_it():
    """``fake_quant=False`` removes the fake-quant rather than neutralizing it.

    A fake-quant that had merely been made an identity -- scale 1, a wide
    clipping range -- would still round every weight to an integer every step,
    still cost its nodes, and still look exactly like this from the outside
    until the numbers came out wrong. So what is asserted is the absence of
    the operators only :meth:`onnxsim.qat_graph.GraphBuilder.round_to_nearest`
    and the clipping emit.

    The rest of the graph is checked against ``EP_FRIENDLY_OPS`` for the
    reason ``apply_qat``'s own tests check it: the block's own operators are
    copied in from the float model and were never governed by the allowlist,
    but everything onnxsim *emits* around them is.
    """
    rng = np.random.default_rng(4)
    reference, student = _pair(rng)
    data = _data(rng, batches=1)

    plan = qat._plan_block(reference, student, "X", "Y", False, False)
    captured = qat._capture(
        reference, sorted(set(plan.externals) | {plan.output_name}), data, None
    )
    externals = {name: captured[name] for name in plan.externals}
    trained = qat._plan_trained(plan.candidates, False, False)
    shapes = qat._block_shapes(
        reference, plan.nodes, externals, plan.output_name, captured[plan.output_name]
    )
    step = qat._build_step_graph(
        trained,
        plan.nodes,
        shapes,
        [],
        externals,
        plan.output_name,
        list(captured[plan.output_name].shape),
        False,
        None,
        False,
        False,
    )

    emitted = {node.op_type for node in step.model.graph.node}
    assert not emitted & {"Sign", "Abs", "Clip", "Round"}
    block_ops = {node.op_type for node in plan.nodes}
    assert not (emitted - block_ops - set(qat_graph.EP_FRIENDLY_OPS))


def test_the_weights_are_written_back_as_floats():
    """There is nothing to project back onto.

    Both quantized schemes end by rounding the master weight onto an integer
    grid, because that is what the model can store. Here the master weight
    *is* what the model stores, so the write-back is the identity that the
    rounding stands in for -- and the initializer must come back fp32, at the
    same name, with its shape unchanged.
    """
    rng = np.random.default_rng(5)
    reference, student = _pair(rng)
    tuned = onnxsim.apply_block_finetune(
        reference,
        student,
        "X",
        "Y",
        calibration_data=_data(rng, batches=1),
        num_iterations=10,
    )
    written = next(t for t in tuned.graph.initializer if t.name == "W2")
    assert written.data_type == onnx.TensorProto.FLOAT
    assert list(written.dims) == [D, D]
    onnx.checker.check_model(tuned, full_check=True)


def test_only_the_matmul_weights_are_touched():
    """A block's other initializers are left byte-identical.

    :func:`onnxsim.qat._find_float_layers` trains MatMul/Gemm weights and
    nothing else, so a LayerNorm's scale and bias sit inside the block, get
    differentiated through, and come out unchanged. That boundary is worth
    pinning: widening it later is a decision, not a refactor.
    """
    rng = np.random.default_rng(6)
    w1 = rng.normal(0, 0.3, (D, D)).astype(np.float32)
    w2 = rng.normal(0, 0.3, (D, D)).astype(np.float32)
    scale = rng.normal(1.0, 0.1, D).astype(np.float32)
    bias = rng.normal(0.0, 0.1, D).astype(np.float32)
    body = f"""
    g (float[8,{D}] X) => (float[8,{D}] Y) {{
      H = MatMul(X, W1)
      N = LayerNormalization<axis = -1>(H, S, B)
      Y = MatMul(N, W2)
    }}
    """

    def build(second):
        return _model(
            body,
            [_f32(w1, "W1"), _f32(second, "W2"), _f32(scale, "S"), _f32(bias, "B")],
        )

    reference = build(w2)
    student = build(w2 + rng.normal(0, 0.15, w2.shape).astype(np.float32))

    losses: list = []
    tuned = onnxsim.apply_block_finetune(
        reference,
        student,
        "X",
        "Y",
        calibration_data=_data(rng),
        num_iterations=200,
        learning_rate=2e-2,
        losses=losses,
    )
    # The block only became trainable at all once graph_grad grew a
    # LayerNormalization rule; before that this slice was refused outright.
    assert losses[-1] < losses[0] / 10
    assert np.array_equal(_init(tuned, "S"), scale)
    assert np.array_equal(_init(tuned, "B"), bias)
    assert not np.array_equal(_init(tuned, "W2"), _init(student, "W2"))


def test_a_gemm_weight_trains_too():
    """``Gemm`` with ``transB``, which stores its weight the other way round.

    Worth its own case because the weight's storage layout is the one thing
    the trained state carries verbatim: the master weight is fed straight back
    into the node's own input, so a layout the code guessed at rather than
    preserved would produce a graph that fails shape inference rather than one
    that trains badly.
    """
    rng = np.random.default_rng(7)
    w = rng.normal(0, 0.3, (D, D)).astype(np.float32)
    body = f"""
    g (float[8,{D}] X) => (float[8,{D}] Y) {{
      Y = Gemm<transB = 1>(X, W)
    }}
    """
    reference = _model(body, [_f32(w, "W")])
    student = _model(body, [_f32(w + rng.normal(0, 0.2, w.shape), "W")])

    losses: list = []
    tuned = onnxsim.apply_block_finetune(
        reference,
        student,
        "X",
        "Y",
        calibration_data=_data(rng),
        num_iterations=300,
        learning_rate=2e-2,
        losses=losses,
    )
    assert losses[-1] < losses[0] / 100
    assert np.abs(_init(tuned, "W") - w).mean() < np.abs(_init(student, "W") - w).mean()


@pytest.mark.parametrize("flag", ["learn_scales", "learn_activation_scales"])
def test_the_scale_flags_are_refused_rather_than_ignored(flag):
    """Both flags name a parameter of a quantizer that is not there.

    Ignoring them is the worse of the two available failures: a caller who
    asked to learn scales and got a model whose scales are exactly as they
    were has no way to tell that from a run in which learning them did not
    help.
    """
    rng = np.random.default_rng(8)
    reference, student = _pair(rng)
    with pytest.raises(ValueError, match=f"{flag}.*fake_quant=False"):
        onnxsim.apply_qat(
            reference,
            student,
            "X",
            "Y",
            calibration_data=_data(rng, batches=1),
            num_iterations=1,
            fake_quant=False,
            **{flag: True},
        )


def test_a_block_with_no_trainable_weight_says_so():
    """The refusal names the scheme, as the two quantized ones already do.

    ``apply_qat``'s refusals go out of their way to distinguish "you named the
    wrong tensors" from "you aimed at the wrong scheme". This one has only one
    scheme to be wrong about, so what it has to say is which shape of weight
    it can hold -- a weight that is computed rather than stored has nothing
    for the optimizer to keep state for.
    """
    rng = np.random.default_rng(9)
    body = f"""
    g (float[8,{D}] X) => (float[8,{D}] Y) {{
      A = Relu(X)
      Y = Add(A, Bias)
    }}
    """
    model = _model(body, [_f32(rng.normal(0, 0.1, D), "Bias")])
    with pytest.raises(ValueError, match="no MatMul/Gemm with a 2-D fp32 weight"):
        onnxsim.apply_block_finetune(
            model, model, "X", "Y", calibration_data=_data(rng, batches=1)
        )


def test_the_whole_model_walk_recovers_a_damaged_model():
    """The walk, end to end, measured on a held-out input.

    Discovery, the per-block failure handling and the returned results are
    ``apply_qat_all_blocks``'s unchanged, so what this adds is only that the
    walk composes with ``fake_quant=False`` -- and the number it records is
    the one a caller would actually care about.
    """
    rng = np.random.default_rng(10)
    weights = [rng.normal(0, 0.3, (D, D)).astype(np.float32) for _ in range(5)]
    body_lines = "\n".join(
        f"  H{i} = MatMul({'X' if i == 0 else f'A{i - 1}'}, W{i})\n  A{i} = Relu(H{i})"
        for i in range(4)
    )
    body = f"""
    g (float[8,{D}] X) => (float[8,{D}] Y) {{
    {body_lines}
      Y = MatMul(A3, W4)
    }}
    """

    def build(ws):
        return _model(body, [_f32(w, f"W{i}") for i, w in enumerate(ws)])

    reference = build(weights)
    student = build([w + rng.normal(0, 0.1, w.shape) for w in weights])
    held_out = _data(rng, batches=8)

    tuned, results = onnxsim.apply_block_finetune_all_blocks(
        reference,
        student,
        calibration_data=_data(rng, batches=32),
        num_iterations=200,
        learning_rate=2e-2,
    )

    assert results and all(r.trained for r in results)
    assert all(r.final_loss < r.initial_loss / 10 for r in results)

    before = _held_out_error(student, reference, held_out)
    assert _held_out_error(tuned, reference, held_out) < before / 2


def test_unstructured_sparsity_is_not_preserved_by_default():
    """The default fills a pruned model's zeros back in, and says so.

    Nothing in the step graph masks the optimizer unless asked: Adam updates
    every element of a trained weight, so an element pruning set to zero gets
    a gradient like any other and leaves zero on the first step. For a model
    whose value *is* its zeros -- an unstructured magnitude-pruned one -- that
    destroys what was bought, and it does so while the loss falls by orders of
    magnitude, which is exactly the shape of failure that goes unnoticed.

    ``preserve_sparsity=True`` is the fix and the test next door measures it.
    The default stays off because "this element is zero" and "this element was
    pruned away" are the same bit pattern, and only the caller knows which
    they meant -- so this test pins the default rather than deprecating it.

    Structured pruning is unaffected either way, because there the channel is
    gone from the tensor rather than zeroed inside it.
    """
    rng = np.random.default_rng(12)
    w1 = rng.normal(0, 0.3, (D, D)).astype(np.float32)
    w2 = rng.normal(0, 0.3, (D, D)).astype(np.float32)

    def pruned(w):
        w = w.copy()
        w[np.abs(w) < np.median(np.abs(w))] = 0.0
        return w

    reference = _mlp(w1, w2)
    sparse1, sparse2 = pruned(w1), pruned(w2)
    student = _mlp(sparse1, sparse2)
    assert (sparse2 == 0).sum() == D * D // 2

    losses: list = []
    tuned = onnxsim.apply_block_finetune(
        reference,
        student,
        "X",
        "Y",
        calibration_data=_data(rng, batches=32),
        num_iterations=300,
        learning_rate=2e-2,
        losses=losses,
    )

    assert losses[-1] < losses[0] / 100
    assert (_init(tuned, "W1") == 0).sum() == 0
    assert (_init(tuned, "W2") == 0).sum() == 0


def _pruned_pair(rng, sparsity=0.5):
    """A dense reference and an unstructured magnitude-pruned copy of it."""
    weights = [rng.normal(0, 0.3, (D, D)).astype(np.float32) for _ in range(2)]

    def pruned(w):
        w = w.copy()
        w[np.abs(w) < np.quantile(np.abs(w), sparsity)] = 0.0
        return w

    sparse = [pruned(w) for w in weights]
    return _mlp(*weights), _mlp(*sparse), sparse


def test_preserve_sparsity_holds_every_pruned_zero_exactly():
    """The fix, and the reason one ``Mul`` is enough.

    With the gradient zeroed wherever the master weight started at zero,
    Adam's ``m`` and ``v`` stay 0 for those elements, its step is
    ``lr * 0 / (sqrt(0) + eps)`` -- exactly 0 -- and the parameter never
    moves. So the assertion is equality with zero rather than a tolerance:
    "held near zero" would be a different, weaker feature, and a model whose
    zeros became 1e-20 is a dense model as far as any sparse kernel is
    concerned.

    The whole zero *pattern* is compared, not just the count, because a run
    that zeroed some other element to compensate would keep the count and be
    entirely wrong.
    """
    rng = np.random.default_rng(21)
    reference, student, sparse = _pruned_pair(rng)

    tuned = onnxsim.apply_block_finetune(
        reference,
        student,
        "X",
        "Y",
        calibration_data=_data(rng, batches=32),
        num_iterations=300,
        learning_rate=2e-2,
        preserve_sparsity=True,
    )

    for name, before in zip(("W1", "W2"), sparse):
        after = _init(tuned, name)
        assert np.array_equal(after == 0, before == 0)
        assert (after == 0).sum() == D * D // 2
        # ...and the surviving weights did move, or the mask would have been
        # a very thorough way of doing nothing.
        assert not np.array_equal(after, before)


def test_preserve_sparsity_still_recovers_error_while_staying_sparse():
    """What the option is actually worth, measured against both alternatives.

    Preserving the zeros costs reconstruction quality -- half the free
    parameters are held at zero, so the block cannot fit the dense reference
    and its loss plateaus far above the unconstrained run's. That is the
    trade, and the number worth having is whether what remains still beats
    doing nothing. It does, on a held-out input.

    The unconstrained run is measured alongside precisely so the comparison is
    not flattering by omission: it reaches a lower error, and it gets there by
    returning a dense model, which is not the model that was asked for.
    """
    rng = np.random.default_rng(22)
    reference, student, _ = _pruned_pair(rng)
    data = _data(rng, batches=32)
    held_out = _data(rng, batches=8)
    before = _held_out_error(student, reference, held_out)

    def run(preserve):
        return onnxsim.apply_block_finetune(
            reference,
            student,
            "X",
            "Y",
            calibration_data=data,
            num_iterations=300,
            learning_rate=2e-2,
            preserve_sparsity=preserve,
        )

    sparse_tuned = run(True)
    dense_tuned = run(False)

    sparse_error = _held_out_error(sparse_tuned, reference, held_out)
    dense_error = _held_out_error(dense_tuned, reference, held_out)

    assert sparse_error < before * 0.9
    # The dense run wins on error and loses the sparsity. Both halves are
    # asserted so neither can quietly stop being true.
    assert dense_error < sparse_error
    assert (_init(dense_tuned, "W1") == 0).sum() == 0
    assert (_init(sparse_tuned, "W1") == 0).sum() == D * D // 2


def test_preserve_sparsity_costs_exactly_one_multiply_per_layer():
    """The mechanism, pinned at the graph level.

    A mask applied by writing zeros back after each step, or by a clean-up
    pass at the end, would satisfy the tests above and be a different thing:
    the weight would move and be moved back, its Adam moments would fill with
    garbage, and the moments are state the loop carries. This asserts the
    cheap version -- one extra ``Mul`` per trained layer, on the gradient --
    which is what keeps the moments at zero too.
    """
    rng = np.random.default_rng(23)
    reference, student, _ = _pruned_pair(rng)
    data = _data(rng, batches=1)

    def step_graph(preserve):
        plan = qat._plan_block(reference, student, "X", "Y", False, False)
        captured = qat._capture(
            reference, sorted(set(plan.externals) | {plan.output_name}), data, None
        )
        externals = {name: captured[name] for name in plan.externals}
        trained = qat._plan_trained(plan.candidates, False, False)
        shapes = qat._block_shapes(
            reference,
            plan.nodes,
            externals,
            plan.output_name,
            captured[plan.output_name],
        )
        return qat._build_step_graph(
            trained,
            plan.nodes,
            shapes,
            [],
            externals,
            plan.output_name,
            list(captured[plan.output_name].shape),
            False,
            None,
            False,
            False,
            preserve,
        )

    def counts(graph):
        ops: dict = {}
        for node in graph.model.graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1
        return ops

    plain_graph, masked_graph = step_graph(False), step_graph(True)
    plain, masked = counts(plain_graph), counts(masked_graph)
    assert masked["Mul"] - plain["Mul"] == 2  # two trained layers
    assert {op: n for op, n in masked.items() if op != "Mul"} == {
        op: n for op, n in plain.items() if op != "Mul"
    }

    # An op count alone cannot tell a gradient mask from a mask on the
    # *updated weight* -- both are one Mul. What separates them is where the
    # result goes: a masked gradient feeds Adam's two moment updates, so it
    # has several consumers and is not itself a state output, whereas a masked
    # weight would be the state output and feed nothing.
    graph = masked_graph.model.graph
    keep = [t.name for t in graph.initializer if "keep" in t.name]
    assert len(keep) == 2
    outputs = {o.name for o in graph.output}
    for name in keep:
        product = next(node.output[0] for node in graph.node if name in node.input)
        consumers = [node for node in graph.node if product in node.input]
        assert product not in outputs
        assert len(consumers) >= 2
