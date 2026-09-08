"""Tests for ``onnxsim.autoround``'s step-graph path -- AutoRound's joint
optimization over weight rounding *and* the per-block clip ratio (and so the
scale) expressed as an ONNX graph (``onnxsim/qat_graph.py``, ``docs/qat.md``)
so it can run on a GPU, an NPU execution provider, or WebGPU instead of only
in host numpy.

``onnxsim.adaround``'s port is tested in ``tests/test_qat_graph.py`` and
``onnxsim.adaquant``'s in ``tests/test_adaquant_step_graph.py``; these are the
questions neither can answer for this one. AutoRound differs from both in a
way that matters for a port: its quantization *bin* moves. AdaRound and
AdaQuant both compute ``floor(w / scale)`` once, in host float64, and feed it
in as a loop constant, because their weight scale never changes; AutoRound's
scale is the thing being optimized, so the division and the flooring happen
inside the graph, every step, in float32. A bin that flips is a discontinuous
change to that element's gradient, so this is the port where "agrees closely,
not exactly" needs measuring rather than asserting. Four things matter:

1. the graph really is the same optimization the numpy loop performs, over
   both parameter groups at once -- checked both at a single step, where
   float32 round-off has not had time to compound, and over a full 200-step
   run;
2. how far the two drift over a full run, stated as measured numbers next to
   a control (the same numpy loop, in float32) that says how much of the
   drift is precision rather than the port;
3. it stays inside the operator set the accelerator backends actually
   implement -- a step graph that reached for a convenient op no NPU
   execution provider supports would pass every numerical test here and still
   be useless for what it exists for; and
4. ``apply_autoround(..., step_providers=...)`` still produces a valid,
   improved model end to end, and still keeps AutoRound's own promise never
   to come out worse than ``apply_adaround`` on the same layer.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest

# The INT4 decoding and outlier-weight helpers tests/test_autoround.py already
# has, and the operator allowlist tests/test_qat_graph.py pins for every step
# graph: both imported rather than copied so they cannot drift. The numpy
# path's tests and the step graph's should be looking at exactly the same
# layer, and whatever that file decides the accelerator backends cover, this
# graph is held to as well.
from test_autoround import _decode_int4, _model, _outlier_block_weight, _scale_tensor
from test_qat_graph import _ALLOWED_OPS

import onnxsim
from onnxsim import autoround

ort = pytest.importorskip("onnxruntime")

_BLOCK_SIZE = 32

_LOOP_KWARGS = dict(
    n_min=-7.0,
    n_max=7.0,
    num_iterations=200,
    learning_rate=0.1,
    clip_learning_rate=0.03,
    reg_param=0.01,
    warm_start=0.2,
    beta_range=(20.0, 2.0),
    clip_ratio_range=(0.5, 1.5),
)


def _layer_case(seed, rows=64, n=8, k=64, block_size=_BLOCK_SIZE):
    """One layer's optimization problem: real-shaped activations and a float
    weight with its abs-max per-block INT4 scale, laid out ``[N, K]`` the way
    ``apply_autoround`` normalizes to.

    Each (output channel, block) gets one large outlier, for the same reason
    ``tests/test_autoround.py`` does: a block whose abs-max scale is set by a
    single outlier leaves every other element under-resolved, which is
    exactly the situation a movable clip ratio can improve and a fixed scale
    cannot. Without one, the joint search has nothing to find, the safety net
    keeps AdaRound's fixed-scale answer on both paths, and a test that
    believed it was exercising this module would be exercising
    :mod:`onnxsim.adaround`.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, k)) * 0.1
    for block in range(k // block_size):
        column = rng.integers(0, block_size, size=n) + block * block_size
        w[np.arange(n), column] = rng.choice([-1.0, 1.0], size=n) * 3.0
    scale_blocks = np.abs(w.reshape(n, k // block_size, block_size)).max(axis=2) / 7.0
    x = np.random.default_rng(seed + 1000).standard_normal((rows, k))
    return x, w, scale_blocks


def _joint(x, w, scale_blocks, path, dtype=np.float64):
    """The joint optimizer's own answer on one path, *before* the AdaRound
    safety net gets to choose between it and a fixed-scale run.

    Comparing the two paths' end-to-end output would not be comparing the two
    optimizers: :func:`onnxsim.autoround._keep_better_of` picks whichever of
    the two candidates measures better, and where the joint and fixed-scale
    optima are close the two paths can legitimately pick differently -- which
    says nothing about whether the graph computes the same step. So this
    stops at the parameters and collapses them itself.

    ``dtype`` exists for the float32 control: running the *numpy* loop in
    float32 is what separates "the graph differs from the numpy loop" from
    "float32 differs from float64".
    """
    w_d, scale_d, x_d = w.astype(dtype), scale_blocks.astype(dtype), x.astype(dtype)
    y_float = x_d @ w_d.T
    _, v0, c0 = autoround._init_autoround(w_d, scale_d, _BLOCK_SIZE)
    args = (w_d, scale_d, _BLOCK_SIZE, x_d, y_float, v0, c0)
    if path == "graph":
        v, c = autoround._joint_loop_on_graph(*args, providers=None, **_LOOP_KWARGS)
    else:
        v, c = autoround._joint_loop(*args, **_LOOP_KWARGS)
    codes, scale_opt, scale_eff = autoround._autoround_results(
        w.astype(np.float64),
        scale_blocks.astype(np.float64),
        _BLOCK_SIZE,
        np.asarray(v, dtype=np.float64),
        np.asarray(c, dtype=np.float64),
        _LOOP_KWARGS["n_min"],
        _LOOP_KWARGS["n_max"],
        _LOOP_KWARGS["clip_ratio_range"],
    )
    return codes, scale_opt, scale_eff


def _reconstruction_error(x, w, codes, scale_eff):
    """``||x @ w.T - x @ (codes * scale).T||`` -- the loss AutoRound
    minimizes, evaluated at a candidate solution."""
    return float(np.linalg.norm(x @ w.T - x @ (codes * scale_eff).T))


def _broadcast(scale_blocks, k):
    return np.repeat(scale_blocks, _BLOCK_SIZE, axis=1)[:, :k]


def test_autoround_step_graph_takes_the_same_first_step_as_the_numpy_loop():
    """One step, where float32 round-off has not had time to compound: the
    graph's Adam update is the numpy loop's, to float32's own precision.

    This is the tightest statement available about the hand-derived gradients
    themselves -- a mis-derived clip-ratio gradient, a block sum over the
    wrong axis, or a ``floor`` that was really a truncation would all show up
    here as an O(1) disagreement rather than the O(1e-6) one measured.

    The statistic is the *median* relative step, and that is deliberate.
    Adam's first update is ``lr * g / (|g| + 1e-8)``, which is ``lr * sign(g)``
    for any gradient comfortably above its epsilon but a genuinely
    precision-dependent fraction of ``lr`` for a gradient near it -- and both
    parameter groups have some of those (a saturated weight element; a block
    whose clip gradient is a sum of ~2000 signed terms that nearly cancel).
    Measured across seeds: the median relative difference is ~8e-7 for the
    relaxation and ~3e-3 for the clip parameter, while the worst tenth of
    elements, the ones in that epsilon regime, differ by whatever fraction of
    a step float32 puts them at.
    """
    kwargs = dict(_LOOP_KWARGS, num_iterations=1)
    for seed in range(3):
        x, w, scale_blocks = _layer_case(seed, n=16, k=128)
        _, v0, c0 = autoround._init_autoround(w, scale_blocks, _BLOCK_SIZE)
        args = (w, scale_blocks, _BLOCK_SIZE, x, x @ w.T, v0, c0)

        v_numpy, c_numpy = autoround._joint_loop(*args, **kwargs)
        v_graph, c_graph = autoround._joint_loop_on_graph(
            *args, providers=None, **kwargs
        )

        v_step = np.maximum(np.abs(v0 - v_numpy), 1e-30)
        c_step = np.maximum(np.abs(c0 - c_numpy), 1e-30)
        assert np.median(np.abs(v_numpy - v_graph) / v_step) < 1e-5
        assert np.median(np.abs(c_numpy - c_graph) / c_step) < 1e-2


def test_autoround_step_graph_agrees_with_the_numpy_loop():
    """Over a full 200-step run, the step graph is the same optimization
    ``_joint_loop`` performs -- over the rounding relaxation and the clip
    ratio jointly, not merely another one that also happens to help.

    Measured over eight seeds, the two paths pick the identical floor/ceil
    decision for at least 97.7% of weight elements, land on clip ratios
    within 5.4% of each other, and reach reconstruction errors within 1.1%.
    The parameters agree far less tightly than the objective does, and that
    is the honest shape of this port: the loss surface here has a
    discontinuity per element (the bin ``floor(w / scale_eff)``, recomputed
    against a scale that moves), so float32 does not merely add noise to a
    trajectory, it occasionally sends it down a different one that ends
    somewhere just as good.

    The control is the same numpy loop run in float32, and it is what makes
    those numbers interpretable: it disagrees with the float64 loop by the
    same order -- rather worse, in fact, on the clip ratios (>=95.9% of
    codes, clip ratios within 10.4%) -- so what is being measured is the
    precision an execution provider offers, not the port.
    """
    for seed in range(8):
        x, w, scale_blocks = _layer_case(seed)
        k = w.shape[1]

        numpy_codes, numpy_scale, numpy_eff = _joint(x, w, scale_blocks, "numpy")
        graph_codes, graph_scale, graph_eff = _joint(x, w, scale_blocks, "graph")
        control_codes, control_scale, _ = _joint(
            x, w, scale_blocks, "numpy", dtype=np.float32
        )

        assert (numpy_codes == graph_codes).mean() > 0.97
        assert np.abs(graph_scale / numpy_scale - 1.0).max() < 0.1

        numpy_err = _reconstruction_error(x, w, numpy_codes, numpy_eff)
        graph_err = _reconstruction_error(x, w, graph_codes, graph_eff)
        assert graph_err == pytest.approx(numpy_err, rel=0.05)

        # Both are a real improvement on round-to-nearest at the original
        # scale, which is the point of running any of this.
        scale_nk = _broadcast(scale_blocks, k)
        rtn = np.clip(np.round(w / scale_nk), -7.0, 7.0)
        assert graph_err < _reconstruction_error(x, w, rtn, scale_nk)

        # The control: float32 in host numpy diverges from float64 by the
        # same order the graph does, so the divergence is precision, not
        # something the graph does differently.
        assert (numpy_codes == control_codes).mean() > 0.95
        assert np.abs(control_scale / numpy_scale - 1.0).max() < 0.15


def test_autoround_step_graph_actually_moves_the_clip_ratio():
    """The clip parameter is being optimized, not just carried: the scale the
    graph path returns has moved off the original RTN scale, stayed inside
    the bounded range the reparameterization promises, and bought a lower
    reconstruction error than the same rounding search at the fixed scale
    reaches. Without this, every assertion above would still pass with
    ``grad_c`` wired to zero."""
    x, w, scale_blocks = _layer_case(1)
    k = w.shape[1]
    codes, scale_opt, scale_eff = _joint(x, w, scale_blocks, "graph")

    ratio = scale_opt / scale_blocks
    assert not np.allclose(ratio, 1.0)
    cmin, cmax = _LOOP_KWARGS["clip_ratio_range"]
    assert np.all(ratio > cmin) and np.all(ratio < cmax)

    fixed_scale_codes = autoround._optimize_rounding_on_graph(
        w,
        _broadcast(scale_blocks, k),
        x,
        _LOOP_KWARGS["n_min"],
        _LOOP_KWARGS["n_max"],
        _LOOP_KWARGS["num_iterations"],
        _LOOP_KWARGS["learning_rate"],
        _LOOP_KWARGS["reg_param"],
        _LOOP_KWARGS["warm_start"],
        _LOOP_KWARGS["beta_range"],
        providers=None,
    )
    joint_err = _reconstruction_error(x, w, codes, scale_eff)
    fixed_err = _reconstruction_error(
        x, w, fixed_scale_codes, _broadcast(scale_blocks, k)
    )
    assert joint_err < fixed_err


def test_autoround_step_graph_is_a_valid_model_declaring_its_six_state_tensors():
    step = autoround._build_autoround_step_graph(8, 3, 8, 2, 4, -7.0, 7.0, (0.5, 1.5))
    onnx.checker.check_model(step.model)

    input_names = {i.name for i in step.model.graph.input}
    output_names = {o.name for o in step.model.graph.output}
    assert set(step.state) <= input_names
    assert set(step.state.values()) <= output_names
    assert step.loss_name in output_names
    # Two parameter groups, each with its own pair of Adam moments: the
    # relaxation is per weight element, the clip parameter per (output
    # channel, block).
    assert set(step.state) == {"v", "m_v", "vv_v", "c", "m_c", "vv_c"}

    shapes = {
        i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
        for i in step.model.graph.input
    }
    assert shapes["v"] == [3, 8]
    assert shapes["c"] == [3, 2]


def test_autoround_step_graph_stays_within_broadly_supported_ops():
    """The whole point of expressing the step as a graph is that it runs on
    backends implementing far less than all of ONNX. ``Floor`` in particular
    is not in the set -- see ``onnxsim.autoround._floor`` for what is emitted
    instead."""
    step = autoround._build_autoround_step_graph(8, 3, 8, 2, 4, -7.0, 7.0, (0.5, 1.5))
    used = {node.op_type for node in step.model.graph.node}
    assert used <= _ALLOWED_OPS, f"unsupported ops in step graph: {used - _ALLOWED_OPS}"
    assert "Floor" not in used


def test_composed_floor_matches_numpy_floor_including_negatives():
    """``_floor``'s composition (truncate toward zero, then correct the
    negative non-integers) is exact, not approximate -- including at the two
    cases a truncation-based floor gets wrong if the correction is misplaced:
    negative integers, where truncation is already right, and values just
    below one."""
    from onnxsim import qat_graph

    b = qat_graph.GraphBuilder()
    out = autoround._floor(b, "a")
    values = np.array(
        [-3.0, -2.5, -2.0, -1.999, -1.0, -0.5, -1e-7, 0.0, 0.5, 1.0, 2.999, 7.0],
        dtype=np.float32,
    )
    graph = onnx.helper.make_graph(
        b.nodes,
        "floor",
        [
            onnx.helper.make_tensor_value_info(
                "a", onnx.TensorProto.FLOAT, [len(values)]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                out, onnx.TensorProto.FLOAT, [len(values)]
            )
        ],
        initializer=b.initializer,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (got,) = sess.run(None, {"a": values})
    np.testing.assert_array_equal(got, np.floor(values))


def test_step_providers_none_never_builds_a_graph(monkeypatch):
    """``step_providers=None`` is the untouched float64 numpy path, not the
    step graph with a CPU provider: the two are not bit-for-bit the same
    thing, and the default has to stay the exact, deterministic one."""

    def refuse(*args, **kwargs):
        raise AssertionError("the numpy path must not build a step graph")

    monkeypatch.setattr(autoround, "_build_autoround_step_graph", refuse)

    K, N, batch = 32, 4, 32
    weight = _outlier_block_weight(K=K, N=N, seed=10)
    float_model = _matmul_model(weight, K, N, batch)
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    x = np.random.default_rng(11).standard_normal((batch, K)).astype(np.float32)
    onnxsim.apply_autoround(
        float_model,
        quant_model,
        calibration_data=[{"X": x}],
        num_iterations=20,
    )


def _matmul_model(weight, K, N, batch):
    return _model(
        f"""
        g (float[{batch},{K}] X) => (float[{batch},{N}] Y)
        {{
          Y = MatMul(X, W)
        }}
        """,
        [onnx.numpy_helper.from_array(weight.astype(np.float32), "W")],
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)[0].astype(np.float64)


def test_apply_autoround_on_a_step_graph_beats_the_untuned_quantized_model():
    """End to end through the public API: ``step_providers`` moves the joint
    optimization onto an execution provider, and what comes back is still a
    valid, AutoRound-improved model -- including the guarantee AutoRound
    makes about itself, that it never lands worse than
    :func:`onnxsim.apply_adaround` would on the same layer and data."""
    K, N, batch = 32, 4, 64
    weight = _outlier_block_weight(K=K, N=N, seed=10)
    float_model = _matmul_model(weight, K, N, batch)
    quant_model = onnxsim.quantize_weight_only_int4(float_model)

    x = np.random.default_rng(11).standard_normal((batch, K)).astype(np.float32)
    calibration_data = [{"X": x}]

    tuned = onnxsim.apply_autoround(
        float_model,
        quant_model,
        calibration_data=calibration_data,
        num_iterations=200,
        step_providers=["CPUExecutionProvider"],
    )
    onnx.checker.check_model(tuned)

    reference = _run(float_model, {"X": x})
    quant_err = np.linalg.norm(reference - _run(quant_model, {"X": x}))
    tuned_err = np.linalg.norm(reference - _run(tuned, {"X": x}))
    assert tuned_err < quant_err

    # The never-worse-than-AdaRound guarantee, on the step-graph path and
    # against an AdaRound run on the same path, so both sides are optimized
    # in the same precision.
    adaround_model = onnxsim.apply_adaround(
        float_model,
        quant_model,
        calibration_data=calibration_data,
        num_iterations=200,
        step_providers=["CPUExecutionProvider"],
    )
    y_float = x.astype(np.float64) @ weight.astype(np.float64)
    ada_err = np.linalg.norm(
        y_float - x.astype(np.float64) @ _decode_int4(adaround_model)
    )
    auto_err = np.linalg.norm(y_float - x.astype(np.float64) @ _decode_int4(tuned))
    assert auto_err <= ada_err + 1e-6

    # AutoRound rewrites exactly two kinds of initializer -- the weight's
    # INT4 codes and its per-block scale -- and leaves everything else
    # byte-identical, with every shape and dtype preserved.
    dq = next(n for n in tuned.graph.node if n.op_type == "DequantizeLinear")
    before = {t.name: t for t in quant_model.graph.initializer}
    after = {t.name: t for t in tuned.graph.initializer}
    assert set(before) == set(after)
    changed = {
        name
        for name in before
        if before[name].SerializeToString() != after[name].SerializeToString()
    }
    assert changed <= {dq.input[0], dq.input[1]}
    assert dq.input[0] in changed, "no weight was actually retuned"
    for name in before:
        assert list(before[name].dims) == list(after[name].dims)
        assert before[name].data_type == after[name].data_type

    # The codes are still legal INT4 in the symmetric [-7, 7] range the
    # quantizer's own kernels assume.
    codes = np.round(
        _decode_int4(tuned)
        / _broadcast(
            onnx.numpy_helper.to_array(_scale_tensor(tuned)).astype(np.float64).T, K
        ).T
    )
    assert np.all(np.abs(codes) <= 7)


def test_apply_autoround_step_providers_are_validated():
    """An execution provider the installed onnxruntime does not have fails
    loudly rather than silently running on the CPU -- backend.validate_providers'
    own contract, inherited by the step-graph path."""
    K, N, batch = 32, 4, 16
    weight = _outlier_block_weight(K=K, N=N, seed=2)
    float_model = _matmul_model(weight, K, N, batch)
    quant_model = onnxsim.quantize_weight_only_int4(float_model)
    x = np.random.default_rng(3).standard_normal((batch, K)).astype(np.float32)

    with pytest.raises(ValueError, match="not available"):
        onnxsim.apply_autoround(
            float_model,
            quant_model,
            calibration_data=[{"X": x}],
            num_iterations=5,
            step_providers=["NoSuchExecutionProvider"],
        )
