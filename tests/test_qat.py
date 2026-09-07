"""Tests for ``onnxsim.apply_qat`` (see ``onnxsim/qat.py``) -- label-free,
block-wise quantization-aware fine-tuning: the float model is the teacher, the
quantized model is the student, and the loss is the block's own output
reconstruction error.

Two claims separate this from the six rounding passes already in the tree, and
each is *measured* here rather than assumed:

1. the block may be any topology :mod:`onnxsim.graph_grad` can differentiate,
   including the activation-between-two-Linears shape
   :func:`onnxsim.apply_brecq` refuses outright (asserted directly: BRECQ
   returns the model unchanged on the very graph this trains);
2. the fp32 weights themselves move, not only their floor/ceil choice.

Claim 2 is the one with a nuance, and the test that makes it records the
nuance instead of hiding it: freeing the weights wins where the reconstruction
problem is *underdetermined*, and loses to AdaRound's smoother relaxation
where it is not. Both directions are measured below.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim
from onnxsim import graph_grad, qat, qat_graph

ort = pytest.importorskip("onnxruntime")

# quantize_weight_only_int4's own fixed block size, so every weight dimension
# below is a multiple of it.
D = 32


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
    return onnx.numpy_helper.from_array(array.astype(np.float32), name)


def _quantize_chain_int4(model, weight_names):
    # Verbatim in spirit from tests/test_brecq.py, and for the same reason:
    # onnxsim.quantize_weight_only_int4 only quantizes a MatMul whose
    # activation input is a graph input, so a multi-layer chain comes back
    # with only its first layer quantized. To get a fully quantized block,
    # quantize each named weight's MatMul in isolation -- where it *is* the
    # first layer -- through the real production pass, then splice the
    # resulting DequantizeLinear and its Wq/Ws back into the chain. The codes
    # are exactly the ones the real pass produces; only the assembly is by
    # hand.
    original = {t.name: t for t in model.graph.initializer}
    quantized = onnx.ModelProto()
    quantized.CopyFrom(model)

    nodes = []
    initializers = [
        t for t in quantized.graph.initializer if t.name not in weight_names
    ]
    for index, node in enumerate(quantized.graph.node):
        if node.op_type not in ("MatMul", "Gemm") or node.input[1] not in weight_names:
            nodes.append(node)
            continue
        weight_name = node.input[1]
        weight = original[weight_name]
        k, n = weight.dims[0], weight.dims[1]
        isolated = _model(
            f"""
            h (float[batch,{k}] Ain) => (float[batch,{n}] Aout)
            {{
              Aout = MatMul(Ain, {weight_name})
            }}
            """,
            [weight],
        )
        isolated_q = onnxsim.quantize_weight_only_int4(isolated)
        dq = next(x for x in isolated_q.graph.node if x.op_type == "DequantizeLinear")
        wq = next(t for t in isolated_q.graph.initializer if t.name == dq.input[0])
        ws = next(t for t in isolated_q.graph.initializer if t.name == dq.input[1])

        suffix = f"_{index}"
        wq_renamed = onnx.TensorProto()
        wq_renamed.CopyFrom(wq)
        wq_renamed.name += suffix
        ws_renamed = onnx.TensorProto()
        ws_renamed.CopyFrom(ws)
        ws_renamed.name += suffix
        dq_out = f"{weight_name}_dq{suffix}"

        new_dq = onnx.NodeProto()
        new_dq.CopyFrom(dq)
        new_dq.input[0] = wq_renamed.name
        new_dq.input[1] = ws_renamed.name
        new_dq.output[0] = dq_out
        initializers.extend([wq_renamed, ws_renamed])
        nodes.append(new_dq)

        new_node = onnx.NodeProto()
        new_node.CopyFrom(node)
        new_node.input[1] = dq_out
        nodes.append(new_node)

    del quantized.graph.node[:]
    quantized.graph.node.extend(nodes)
    del quantized.graph.initializer[:]
    quantized.graph.initializer.extend(initializers)
    return quantized


def _dequantize_int4_for(model, matmul_output_name):
    """The effective float weight the quantized model actually deploys for one
    MatMul -- same decode as ``tests/test_brecq.py``'s own helper."""
    matmul = next(n for n in model.graph.node if n.output[0] == matmul_output_name)
    dq = next(n for n in model.graph.node if n.output[0] == matmul.input[1])
    wq = next(t for t in model.graph.initializer if t.name == dq.input[0])
    ws = next(t for t in model.graph.initializer if t.name == dq.input[1])
    block_size = next(a.i for a in dq.attribute if a.name == "block_size")
    axis = next((a.i for a in dq.attribute if a.name == "axis"), 1)

    dims = list(wq.dims)
    numel = int(np.prod(dims))
    raw = np.frombuffer(wq.raw_data, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    codes = np.empty(numel, dtype=np.int8)
    codes[0::2] = lo[: (numel + 1) // 2]
    codes[1::2] = hi[: numel // 2]
    codes = codes.reshape(dims).astype(np.float64)

    scale = onnx.numpy_helper.to_array(ws).astype(np.float64)
    return codes * np.repeat(scale, block_size, axis=axis)


def _relu_block_model(seed=0):
    """Two Linears with a ``Relu`` between them, plus a residual.

    The single node in the middle is the whole point: :mod:`onnxsim.brecq`'s
    block discovery requires each layer's activation input to be *exactly* the
    previous layer's output, so this topology is invisible to it -- which
    ``test_a_block_brecq_cannot_discover_at_all`` asserts rather than assumes.
    """
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    w2 = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    return _model(
        f"""
        g (float[batch,{D}] X) => (float[batch,{D}] Yout)
        {{
          Y1 = MatMul(X, W1)
          A1 = Relu(Y1)
          Y2 = MatMul(A1, W2)
          Yout = Add(Y2, X)
        }}
        """,
        [_f32(w1, "W1"), _f32(w2, "W2")],
    )


def _correlated_calibration(rank, num_samples=64, noise=0.05, seed=100):
    """Calibration rows spanning only ``rank`` directions.

    ``rank`` is the knob every measured comparison here turns. A low-rank
    activation makes ``||X (W - W_hat)||`` massively underdetermined -- whole
    subspaces of integer weights reconstruct the layer equally well, and the
    good ones are nowhere near round-to-nearest. A full-rank one pins the
    optimum next to RTN, where floor/ceil is all the freedom there is to use.
    """
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((num_samples, rank)).astype(np.float32)
    projection = rng.standard_normal((rank, D)).astype(np.float32)
    x = latent @ projection
    x += rng.standard_normal((num_samples, D)).astype(np.float32) * noise
    return x


def _relu_block_error(model, x, w1, w2):
    """``||teacher - student||`` for the whole ``_relu_block_model`` block,
    computed from the deployed integer weights rather than from the training
    loop's own numbers."""
    hidden = np.maximum(x.astype(np.float64) @ _dequantize_int4_for(model, "Y1"), 0.0)
    student = hidden @ _dequantize_int4_for(model, "Y2") + x.astype(np.float64)
    teacher = np.maximum(x.astype(np.float64) @ w1.astype(np.float64), 0.0) @ w2.astype(
        np.float64
    ) + x.astype(np.float64)
    return np.linalg.norm(teacher - student)


def _weights_of(model):
    return {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}


def _quant_tensors_for(model, matmul_output_name):
    """``(codes initializer name, scale initializer name)`` for one quantized
    MatMul, found the way the pass itself finds them: through the
    ``DequantizeLinear`` feeding the node's weight input."""
    matmul = next(n for n in model.graph.node if n.output[0] == matmul_output_name)
    dq = next(n for n in model.graph.node if n.output[0] == matmul.input[1])
    return dq.input[0], dq.input[1]


def test_a_block_brecq_cannot_discover_at_all_trains_below_round_to_nearest():
    """The headline: an activation between two Linears.

    :func:`onnxsim.apply_brecq` -- the closest existing pass, and the one that
    already optimizes the *block's* output rather than each layer's -- returns
    this model completely unchanged, because its discovery walks only a linear
    MatMul/Gemm chain. Asserted here, not assumed, so the claim cannot rot.

    Measured on this scenario (rank-2 calibration, seed 0): round-to-nearest
    leaves a block reconstruction error of ~16.0; 1000 Adam steps take it to
    ~6.5, a ~60% reduction, with the training loss falling ~5x.
    """
    model = _relu_block_model(seed=0)
    x = _correlated_calibration(rank=2)
    calibration_data = [{"X": x}]
    quant = _quantize_chain_int4(model, {"W1", "W2"})

    unchanged = onnxsim.apply_brecq(
        model, quant, blocks=[("X", "Yout")], calibration_data=calibration_data
    )
    assert unchanged.SerializeToString() == quant.SerializeToString()

    w1 = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W1")
    )
    w2 = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W2")
    )

    losses = []
    tuned = onnxsim.apply_qat(
        model, quant, "X", "Yout", calibration_data=calibration_data, losses=losses
    )
    onnx.checker.check_model(tuned)

    rtn_error = _relu_block_error(quant, x, w1, w2)
    qat_error = _relu_block_error(tuned, x, w1, w2)
    assert qat_error < 0.6 * rtn_error
    # The loop really optimized, rather than the improvement coming from
    # somewhere else: the reported block loss falls monotonically enough to
    # end several times below where it started.
    assert losses[-1] < 0.3 * losses[0]


def test_a_gelu_block_trains():
    """The other topology ``docs/qat.md`` names: a transformer FFN, GELU in
    its exact ``erf`` form, with the block input feeding both the first
    projection and the residual (so the backward has to accumulate two
    gradient paths into it). Nothing about the pass is special-cased for it --
    it is simply more nodes with rules in
    :data:`onnxsim.graph_grad.SUPPORTED_OPS`."""
    rng = np.random.default_rng(0)
    hidden = 64
    w1 = (rng.standard_normal((D, hidden)) * 0.3).astype(np.float32)
    w2 = (rng.standard_normal((hidden, D)) * 0.3).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{D}] X) => (float[batch,{D}] Yout)
        <float half = {{0.5}}, float one = {{1.0}}, float inv_sqrt2 = {{0.70710678}}>
        {{
          H = MatMul(X, W1)
          S = Mul(H, inv_sqrt2)
          E = Erf(S)
          Ep = Add(E, one)
          Hh = Mul(half, Ep)
          G = Mul(H, Hh)
          P = MatMul(G, W2)
          Yout = Add(P, X)
        }}
        """,
        [_f32(w1, "W1"), _f32(w2, "W2")],
    )
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    x = _correlated_calibration(rank=2)

    losses = []
    tuned = onnxsim.apply_qat(
        model, quant, "X", "Yout", calibration_data=[{"X": x}], losses=losses
    )
    onnx.checker.check_model(tuned)
    # ~0.35 -> ~0.03 as measured; the assertion is deliberately looser than
    # the observed margin so it tracks the mechanism, not the seed.
    assert losses[-1] < 0.25 * losses[0]


def test_freeing_the_weights_beats_optimizing_only_their_rounding():
    """Claim 2, isolated as cleanly as it can be.

    The block here is a *single* layer, so ``apply_qat``'s objective is
    literally the one :func:`onnxsim.apply_adaround` already minimizes for
    that layer -- the block's output *is* the layer's output. The only thing
    that differs is what is free: AdaRound may push each element to the
    integer below or the integer above, and nowhere else; ``apply_qat`` moves
    the fp32 weight itself, so an element may migrate several codes.

    Measured, rank-1 calibration, seed 0: RTN 5.96, AdaRound 3.07, QAT 1.78 --
    a 42% further reduction on top of AdaRound. Across seeds 0-7 the direction
    held every time, by between 0.5% (seed 7) and 44%.

    **This is scenario-dependent, and the dependence is the interesting
    part.** Repeat the same experiment with full-rank calibration
    (``rank=16``) and it inverts: RTN 28.5, AdaRound 14.6, QAT 22.8 -- AdaRound
    wins by 36%. That is not a defect in either. When the activations span
    every direction, the reconstruction optimum sits within one quantization
    step of round-to-nearest, floor/ceil is therefore all the freedom that is
    useful, and AdaRound's continuous rectified-sigmoid relaxation optimizes
    that restricted problem better than a hard straight-through estimator on a
    piecewise-constant loss does. When the activations are low-rank the
    optimum is far away, outside AdaRound's box entirely, and only a free
    weight can reach it. Real calibration activations are strongly low-rank,
    which is why this is worth having -- but "QAT always beats AdaRound" is
    not a claim this module makes, and the second half of this test measures
    exactly the case where it is false.
    """
    model = _relu_block_model(seed=0)
    w1 = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W1")
    ).astype(np.float64)

    def layer_error(candidate_model, x):
        teacher = x.astype(np.float64) @ w1
        student = x.astype(np.float64) @ _dequantize_int4_for(candidate_model, "Y1")
        return np.linalg.norm(teacher - student)

    def errors(rank):
        x = _correlated_calibration(rank=rank)
        calibration_data = [{"X": x}]
        quant = _quantize_chain_int4(model, {"W1", "W2"})
        adaround = onnxsim.apply_adaround(
            model, quant, calibration_data=calibration_data
        )
        tuned = onnxsim.apply_qat(
            model, quant, "X", "Y1", calibration_data=calibration_data
        )
        return (
            layer_error(quant, x),
            layer_error(adaround, x),
            layer_error(tuned, x),
        )

    rtn, adaround, tuned = errors(rank=1)
    assert tuned < adaround < rtn

    # The honest other half: where the problem is well determined, optimizing
    # only the rounding -- with a better-conditioned relaxation -- wins.
    rtn_full, adaround_full, tuned_full = errors(rank=16)
    assert adaround_full < rtn_full
    assert tuned_full < rtn_full
    assert adaround_full < tuned_full


def test_an_unsupported_op_in_the_block_is_refused():
    """A block containing an op :mod:`onnxsim.graph_grad` has no rule for is
    an error, not a quietly unchanged model. Silently skipping is how a caller
    ends up believing a block was fine-tuned when it never was."""
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{D}] X) => (float[batch,{D}] Y2)
        {{
          Y1 = MatMul(X, W)
          S = Sin(Y1)
          Y2 = MatMul(S, W)
        }}
        """,
        [_f32(w, "W")],
    )
    quant = _quantize_chain_int4(model, {"W"})
    with pytest.raises(graph_grad.UnsupportedOpError, match="Sin"):
        onnxsim.apply_qat(
            model,
            quant,
            "X",
            "Y2",
            calibration_data=[{"X": _correlated_calibration(rank=4)}],
        )


def test_a_block_with_nothing_quantized_in_it_is_refused():
    """The other half of the same contract: a block that matches no
    ``quantize_weight_only_int4`` layer has nothing to train, so saying so
    beats returning the input unchanged and letting the caller assume it
    worked."""
    model = _relu_block_model(seed=0)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    with pytest.raises(ValueError, match="no quantize_weight_only_int4"):
        onnxsim.apply_qat(
            model,
            quant,
            "Y1",
            "A1",  # just the Relu -- no quantized layer inside
            calibration_data=[{"X": _correlated_calibration(rank=4)}],
        )


def test_a_block_output_that_is_not_computed_is_refused():
    model = _relu_block_model(seed=0)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    with pytest.raises(ValueError, match="not produced by any node"):
        onnxsim.apply_qat(
            model,
            quant,
            "X",
            "X",
            calibration_data=[{"X": _correlated_calibration(rank=4)}],
        )


def test_only_the_blocks_own_weight_initializers_change():
    """Everything outside the trained block must come back byte-identical --
    including the second block's weights, every scale, and the graph itself.
    A pass that rewrote more than it claimed would be nearly impossible to
    notice downstream."""
    rng = np.random.default_rng(3)
    weights = [(rng.standard_normal((D, D)) * 0.3).astype(np.float32) for _ in range(3)]
    model = _model(
        f"""
        g (float[batch,{D}] X) => (float[batch,{D}] Y3)
        {{
          Y1 = MatMul(X, W1)
          A1 = Relu(Y1)
          Y2 = MatMul(A1, W2)
          A2 = Relu(Y2)
          Y3 = MatMul(A2, W3)
        }}
        """,
        [_f32(w, f"W{i + 1}") for i, w in enumerate(weights)],
    )
    quant = _quantize_chain_int4(model, {"W1", "W2", "W3"})
    tuned = onnxsim.apply_qat(
        model,
        quant,
        "X",
        "Y2",  # the first two layers only; W3's layer is outside the block
        calibration_data=[{"X": _correlated_calibration(rank=2)}],
    )

    # The graph -- nodes, inputs, outputs, opset, everything but initializer
    # payloads -- is untouched.
    before = onnx.ModelProto()
    before.CopyFrom(quant)
    after = onnx.ModelProto()
    after.CopyFrom(tuned)
    del before.graph.initializer[:]
    del after.graph.initializer[:]
    assert before.SerializeToString() == after.SerializeToString()

    trained = {_quant_tensors_for(quant, name)[0] for name in ("Y1", "Y2")}
    old = {t.name: t for t in quant.graph.initializer}
    new = {t.name: t for t in tuned.graph.initializer}
    assert set(old) == set(new)
    changed = {
        name
        for name in old
        if old[name].SerializeToString() != new[name].SerializeToString()
    }
    assert changed <= trained
    # ...and it did in fact change something, so the assertion above is not
    # passing vacuously.
    assert changed


def test_the_weight_only_path_leaves_every_scale_byte_identical():
    """``learn_scales=False`` is the default precisely because it keeps this
    guarantee, the same one :func:`onnxsim.apply_adaround` makes."""
    model = _relu_block_model(seed=1)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    tuned = onnxsim.apply_qat(
        model,
        quant,
        "X",
        "Yout",
        calibration_data=[{"X": _correlated_calibration(rank=2)}],
    )
    old, new = _weights_of(quant), _weights_of(tuned)
    for matmul in ("Y1", "Y2"):
        scale_name = _quant_tensors_for(quant, matmul)[1]
        np.testing.assert_array_equal(old[scale_name], new[scale_name])


def test_learn_scales_moves_the_scales_and_still_reconstructs():
    """LSQ's scale gradient wired in. The scales move (so the gradient is
    reaching them at all) and the block still reconstructs better than
    round-to-nearest -- the honest bar, since jointly optimizing two coupled
    parameter sets is a harder problem than either alone, exactly the caution
    :mod:`onnxsim.autoround` documents for its own clip ratio."""
    model = _relu_block_model(seed=0)
    x = _correlated_calibration(rank=2)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    w1 = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W1")
    )
    w2 = onnx.numpy_helper.to_array(
        next(t for t in model.graph.initializer if t.name == "W2")
    )

    tuned = onnxsim.apply_qat(
        model,
        quant,
        "X",
        "Yout",
        calibration_data=[{"X": x}],
        learn_scales=True,
        scale_learning_rate=1e-4,
    )
    onnx.checker.check_model(tuned)

    old, new = _weights_of(quant), _weights_of(tuned)
    moved = 0
    for matmul in ("Y1", "Y2"):
        scale_name = _quant_tensors_for(quant, matmul)[1]
        before, after = old[scale_name], new[scale_name]
        assert before.shape == after.shape
        if not np.array_equal(before, after):
            moved += 1
    assert moved == 2

    assert _relu_block_error(tuned, x, w1, w2) < _relu_block_error(quant, x, w1, w2)


def test_end_to_end_on_the_cpu_step_provider():
    """The whole loop through ``step_providers=``, which is the boundary
    ``docs/qat.md`` says carries this to CUDA, an NPU EP or WebGPU
    unmodified. CPU is the one CI can assert on; the point of the test is
    that the provider path is exercised at all and produces a model
    onnxruntime will actually load."""
    model = _relu_block_model(seed=2)
    x = _correlated_calibration(rank=2, num_samples=32)
    quant = _quantize_chain_int4(model, {"W1", "W2"})

    tuned = onnxsim.apply_qat(
        model,
        quant,
        "X",
        "Yout",
        calibration_data=[{"X": x}],
        num_iterations=50,
        providers=["CPUExecutionProvider"],
        step_providers=["CPUExecutionProvider"],
    )
    onnx.checker.check_model(tuned)

    session = ort.InferenceSession(
        tuned.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (out,) = session.run(None, {"X": x})
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


def test_codes_stay_inside_the_int4_grid():
    """A free fp32 weight can wander anywhere; the exported codes may not.
    ``quantize_weight_only_int4``'s grid is symmetric ``[-7, 7]`` and the
    clip in the fake-quant forward is the only thing keeping the export
    inside it."""
    model = _relu_block_model(seed=4)
    quant = _quantize_chain_int4(model, {"W1", "W2"})
    tuned = onnxsim.apply_qat(
        model,
        quant,
        "X",
        "Yout",
        calibration_data=[{"X": _correlated_calibration(rank=2) * 4}],
        learning_rate=1e-2,  # deliberately far too large, to push the weights out
    )
    checked = 0
    for t in tuned.graph.initializer:
        if t.data_type != onnx.TensorProto.INT4:
            continue
        checked += 1
        numel = int(np.prod(list(t.dims)))
        raw = np.frombuffer(t.raw_data, dtype=np.uint8)
        lo = (raw & 0x0F).astype(np.int8)
        hi = ((raw >> 4) & 0x0F).astype(np.int8)
        lo = np.where(lo >= 8, lo - 16, lo)
        hi = np.where(hi >= 8, hi - 16, hi)
        codes = np.empty(numel, dtype=np.int8)
        codes[0::2] = lo[: (numel + 1) // 2]
        codes[1::2] = hi[: numel // 2]
        assert np.all(codes >= -7) and np.all(codes <= 7)
    assert checked == 2


def test_the_step_graph_stays_inside_the_execution_provider_allowlist(monkeypatch):
    """The reason all of this is expressed as ONNX rather than numpy is that
    it must run on WebGPU and NPU execution providers, and an op none of them
    implement would pass every numerical test above while making the whole
    exercise pointless. So: everything the optimizer machinery emits -- the
    fake-quant forward, the loss, the backward, Adam -- stays inside
    :data:`onnxsim.qat_graph.EP_FRIENDLY_OPS`. The block's *own* nodes are
    excluded, since those are whatever the user's model already contains."""
    model = _relu_block_model(seed=0)
    quant = _quantize_chain_int4(model, {"W1", "W2"})

    captured = {}
    real = qat_graph.run_step_graph

    def spy(step, **kwargs):
        captured["step"] = step
        return real(step, **kwargs)

    monkeypatch.setattr(qat.qat_graph, "run_step_graph", spy)
    onnxsim.apply_qat(
        model,
        quant,
        "X",
        "Yout",
        calibration_data=[{"X": _correlated_calibration(rank=2, num_samples=8)}],
        num_iterations=1,
        learn_scales=True,
    )

    block_ops = {n.op_type for n in model.graph.node}
    emitted = {n.op_type for n in captured["step"].model.graph.node} - block_ops
    assert emitted <= set(qat_graph.EP_FRIENDLY_OPS), sorted(
        emitted - set(qat_graph.EP_FRIENDLY_OPS)
    )


def test_a_residual_arriving_from_upstream_of_the_block_is_teacher_forced():
    """A block whose output adds in a tensor produced *before*
    ``block_input_name`` is still a closed block: the extra tensor is captured
    from the float model and fed in as another constant, exactly as the block
    input itself is. That is the same teacher-forcing every block-wise
    reconstruction method does, and without it the discovery would have to
    refuse a shape real residual networks are full of."""
    rng = np.random.default_rng(5)
    w0 = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    w1 = (rng.standard_normal((D, D)) * 0.3).astype(np.float32)
    model = _model(
        f"""
        g (float[batch,{D}] X) => (float[batch,{D}] Yout)
        {{
          Y0 = MatMul(X, W0)
          A0 = Relu(Y0)
          Y1 = MatMul(A0, W1)
          Yout = Add(Y1, Y0)
        }}
        """,
        [_f32(w0, "W0"), _f32(w1, "W1")],
    )
    quant = _quantize_chain_int4(model, {"W0", "W1"})
    losses = []
    tuned = onnxsim.apply_qat(
        model,
        quant,
        "A0",  # the block starts after the activation; Y0 enters sideways
        "Yout",
        calibration_data=[{"X": _correlated_calibration(rank=2)}],
        losses=losses,
    )
    onnx.checker.check_model(tuned)
    assert losses[-1] < losses[0]

    # Only the layer inside the block (W1's) was retrained; W0's codes are
    # outside it and must be untouched.
    old, new = _weights_of(quant), _weights_of(tuned)
    outside = _quant_tensors_for(quant, "Y0")[0]
    np.testing.assert_array_equal(old[outside], new[outside])


def test_a_transb_gemm_trains_on_the_other_blocked_axis():
    """A ``Gemm`` with ``transB=1`` stores its weight ``[N, K]`` and blocks
    its scale along axis 1, the mirror image of a ``MatMul``'s ``[K, N]``
    weight blocked along axis 0. Both directions of the reshape that expands
    a per-block scale to per-element (and sums a per-element gradient back
    into per-block) are therefore exercised only if both layouts are tested,
    and a transposed reshape is exactly the kind of bug that produces a
    plausible-looking but wrong model."""
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal((64, D)) * 0.3).astype(np.float32)  # [N, K]
    model = _model(
        f"""
        g (float[8,{D}] X) => (float[8,64] Y)
        {{
          Y = Gemm <transB = 1> (X, W)
        }}
        """,
        [_f32(weight, "W")],
    )
    quant = onnxsim.quantize_weight_only_int4(model)
    x = _correlated_calibration(rank=2, num_samples=8)

    losses = []
    tuned = onnxsim.apply_qat(
        model,
        quant,
        "X",
        "Y",
        calibration_data=[{"X": x}],
        learn_scales=True,
        losses=losses,
    )
    onnx.checker.check_model(tuned)
    assert losses[-1] < 0.5 * losses[0]

    codes_name, scale_name = _quant_tensors_for(quant, "Y")
    old, new = _weights_of(quant), _weights_of(tuned)
    assert list(old[scale_name].shape) == list(new[scale_name].shape) == [64, 1]
    assert not np.array_equal(old[codes_name], new[codes_name])
