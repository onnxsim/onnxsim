#!/usr/bin/env python3
"""Emit real QAT step graphs as ONNX fixtures, for the execution-provider test.

``onnxsim/qat_graph.py`` pins the operators a step graph may contain
(``EP_FRIENDLY_OPS``) and justifies the set by claiming those operators have
coverage on onnxruntime-web's WebGPU backend and the WebNN/NPU execution
providers. Three python test files assert that the emitted graphs stay inside
the set; nothing checked that the set is *right*. ``step_graph_ep.test.mjs``
does -- by loading these graphs into onnxruntime-web and running them on every
execution provider it can reach.

The graphs here are produced by the library's own builders, never hand-written,
so a fixture cannot drift into describing a step graph onnxsim does not emit:

  * ``step_adaround.onnx``     -- ``onnxsim.adaround._build_rounding_step_graph``
  * ``step_adaquant.onnx``     -- ``onnxsim.adaquant._build_adaquant_step_graph``
  * ``step_qat_backward.onnx`` -- ``onnxsim.graph_grad.build_backward`` composed
    with ``qat_graph.adam_update``/``make_step_graph``, the way
    ``onnxsim.qat._build_step_graph`` composes them. The forward slice is
    written here rather than sliced out of a quantized model (which would drag
    in the whole ``apply_qat`` pipeline for no extra operator coverage), and is
    deliberately built from ``EP_FRIENDLY_OPS`` members only, so the *whole*
    fixture stays inside the set under test. A real ``apply_qat`` step graph
    embeds its block's forward nodes verbatim and can therefore contain ops
    outside the set (``Relu``, ``Softmax``, ...); that is a property of the
    caller's model, not of what the builders emit, and is out of scope here.
  * ``step_minibatch.onnx``    -- ``qat_graph.GraphBuilder.gather_rows`` reading
    a minibatch out of a resident calibration set, which is the only thing
    ``Gather`` is in the allowlist for. It is also the only fixture with a
    *non-float* input (the rank-1 int64 row index, declared through
    ``make_step_graph``'s ``per_step``), and therefore the one most likely to
    find a backend limit: int64 is exactly what onnxruntime-web's WebNN backend
    is known to reject.

Between them the three cover every operator in ``EP_FRIENDLY_OPS`` -- the
script asserts that, so the set growing a member with no fixture behind it
fails here rather than going quietly unverified.

Each graph is accompanied, in ``step_graphs.json``, by the values to feed it
(constants, initial state, and the per-step scalars), the state wiring
``run_step_graph`` closes the loop with, and the loss trajectory the same feeds
produce through onnxruntime's CPU provider in Python. The Node test replays
exactly that loop and compares, so "onnxruntime-web on this EP agrees with
onnxruntime on CPU" is a numeric check rather than a claim.

A fifth graph, ``step_qat_hf_demo.onnx`` (``build_qat_hf_demo``, below), is
generated here too but is not part of the ``step_graphs.json`` manifest or
``step_graph_ep.test.mjs``'s own EP-coverage measurement -- it is driven with
*live* data (a real Hugging Face photo) by a different consumer,
``webgpu_hf_demo.test.mjs``; see that function's own docstring for why it
needs its own file.

Regenerate (from this directory, with onnxsim importable from the repo root)::

    python3 make_step_graph_fixtures.py

Rerun it after changing any of the four builders; the committed ``.onnx``
files are the point of the fixture, so they must be regenerated and committed
deliberately rather than rebuilt at test time.
"""

import json
import pathlib
import sys
from typing import Dict, List, Sequence

import numpy as np
import onnx

HERE = pathlib.Path(__file__).parent
# Prefer the repo's own onnxsim over anything installed, so the fixtures always
# come from the working tree being tested.
sys.path.insert(0, str((HERE / ".." / ".." / "..").resolve()))

import onnxruntime as ort  # noqa: E402
from onnxsim import adaquant, adaround, graph_grad, qat_graph  # noqa: E402

# One shared seed: every array below is drawn from it, so a regeneration with
# unchanged builders produces byte-identical fixtures.
SEED = 20260907

# How many steps the recorded reference trajectory (and the Node test) runs.
# Four is enough for the second step's loss to have moved measurably while
# keeping the JSON small.
NUM_STEPS = 4


def _f32(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def _tensor_entry(array: np.ndarray) -> Dict:
    """One tensor as the Node test wants it: dims plus flat float32 data."""
    array = _f32(array)
    return {"dims": list(array.shape), "data": [float(v) for v in array.ravel()]}


def _op_histogram(model: onnx.ModelProto) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return dict(sorted(counts.items()))


def _reference_losses(
    step: qat_graph.StepGraph,
    constants: Dict[str, np.ndarray],
    state: Dict[str, np.ndarray],
    scalars: Sequence[Dict[str, float]],
    per_step: Sequence[Dict[str, np.ndarray]],
) -> List[float]:
    """The loss at each step from onnxruntime's CPU provider in Python.

    Deliberately *not* ``qat_graph.run_step_graph``: that may take the
    IOBinding path, and what the Node test replays is the plain feed-per-step
    loop. Running the same loop here is what makes the two comparable.
    """
    session = ort.InferenceSession(
        step.model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    fetch = list(step.state.values()) + [str(step.loss_name)]
    current = {k: _f32(v) for k, v in state.items()}
    losses = []
    for t, values in enumerate(scalars):
        feeds = {k: _f32(v) for k, v in constants.items()}
        feeds.update(current)
        feeds.update({k: _f32(v) for k, v in values.items()})
        # Non-float per-step inputs (the int64 minibatch index) pass through
        # with the dtype they were built with, as run_step_graph does.
        feeds.update({k: v for k, v in (per_step[t] if per_step else {}).items()})
        out = session.run(fetch, feeds)
        result = dict(zip(fetch, out))
        current = {name: result[o] for name, o in step.state.items()}
        losses.append(float(result[str(step.loss_name)]))
    return losses


def _annealed_scalars(
    base: Dict[str, float], warm_start_steps: int, reg_param: float
) -> List[Dict[str, float]]:
    """The scalar feeds for ``NUM_STEPS`` steps, on the warm-start-then-anneal
    schedule ``adaround``/``adaquant`` both drive their step graph with (their
    ``_optimize_*_on_graph``'s own ``scalars`` callback, with the iteration
    count shrunk to ``NUM_STEPS``)."""
    beta_start, beta_end = 20.0, 2.0
    steps = []
    for t in range(NUM_STEPS):
        values = dict(base)
        if t >= warm_start_steps:
            progress = (t - warm_start_steps) / max(
                1, NUM_STEPS - warm_start_steps - 1
            )
            values["reg_scale"] = reg_param
            values["beta"] = beta_start + (beta_end - beta_start) * progress
        else:
            # The regularizer is off by weight, not by a second graph; beta
            # still needs a value Pow can evaluate.
            values["reg_scale"] = 0.0
            values["beta"] = 1.0
        values.update(qat_graph.adam_bias_corrections(t))
        steps.append(values)
    return steps


def build_adaround(rng: np.random.Generator) -> Dict:
    """AdaRound's rounding step: one weight matrix's floor/ceil relaxation."""
    num_rows, n, k = 8, 4, 8
    # int4's code range, which is what apply_adaround runs on.
    n_min, n_max = -8.0, 7.0

    step = adaround._build_rounding_step_graph(num_rows, n, k, n_min, n_max)

    w = _f32(rng.normal(scale=0.5, size=(n, k)))
    scale = _f32(np.abs(w).max(axis=1, keepdims=True) / n_max + 1e-3)
    scale = _f32(np.repeat(scale, k, axis=1))
    x = _f32(rng.normal(size=(num_rows, k)))
    # Exactly the warm start both optimization paths share, then jogged off it.
    # At the warm start h(v) reproduces the un-rounded ratio exactly, so the
    # reconstruction loss is 0 to float precision -- a fine place for the real
    # loop to begin and a useless one to compare two runtimes at. The state fed
    # in here is therefore a warm start a few steps in, which is what every
    # step but the first actually sees.
    v0, floor_base = adaround._init_relaxation(w.astype(np.float64), scale.astype(np.float64))
    v0 = v0 + rng.normal(scale=1.5, size=v0.shape)

    constants = {
        "x": x,
        "y_float": _f32(x @ w.T),
        "floor_base": _f32(floor_base),
        "scale": scale,
    }
    state = {"v": _f32(v0), "m": np.zeros((n, k), np.float32), "vv": np.zeros((n, k), np.float32)}
    scalars = _annealed_scalars({"lr": 0.1}, warm_start_steps=1, reg_param=0.01)
    return _package(
        "adaround",
        "step_adaround.onnx",
        "onnxsim.adaround._build_rounding_step_graph"
        f"(num_rows={num_rows}, n={n}, k={k}, n_min={n_min}, n_max={n_max})",
        step,
        constants,
        state,
        scalars,
    )


def build_adaquant(rng: np.random.Generator) -> Dict:
    """AdaQuant's step: rounding relaxation *and* the activation scale and
    zero-point, so nine state tensors and three Adam updates -- including the
    rank-0 state tensors, which are the shape an accelerator backend is most
    likely to be unhappy with."""
    num_rows, n, k = 8, 4, 8
    step = adaquant._build_adaquant_step_graph(num_rows, n, k)

    w = _f32(rng.normal(scale=0.5, size=(n, k)))
    scale_n = _f32(np.abs(w).max(axis=1) / 127.0 + 1e-4)
    x = _f32(np.abs(rng.normal(size=(num_rows, k))))  # a post-Relu activation
    x_scale0 = float(x.max() / 255.0)
    scale_nk, v0, floor_base, log_s0, zp0 = adaquant._init_adaquant(
        w.astype(np.float64), scale_n.astype(np.float64), x_scale0, 0.0
    )
    # Off the warm start, for the reason build_adaround gives.
    v0 = v0 + rng.normal(scale=1.5, size=v0.shape)

    constants = {
        "x": x,
        "y_float": _f32(x @ w.T),
        "floor_base": _f32(floor_base),
        "scale": _f32(scale_nk),
    }
    zero = np.zeros((), np.float32)
    state = {
        "v": _f32(v0),
        "m_v": np.zeros((n, k), np.float32),
        "vv_v": np.zeros((n, k), np.float32),
        "log_s": _f32(log_s0),
        "m_s": zero,
        "vv_s": zero,
        "zp": _f32(zp0),
        "m_zp": zero,
        "vv_zp": zero,
    }
    scalars = _annealed_scalars(
        {"w_lr": 0.1, "a_lr": 0.01}, warm_start_steps=1, reg_param=0.01
    )
    return _package(
        "adaquant",
        "step_adaquant.onnx",
        f"onnxsim.adaquant._build_adaquant_step_graph(num_rows={num_rows}, n={n}, k={k})",
        step,
        constants,
        state,
        scalars,
    )


def build_qat_backward(rng: np.random.Generator) -> Dict:
    """A ``graph_grad.build_backward`` composition: a forward block, the
    reconstruction loss against a teacher output, the emitted backward, and one
    Adam step per trained tensor -- ``onnxsim.qat._build_step_graph``'s own
    ordering (forward first, since the backward rules read forward tensors by
    name).

    The forward is chosen for which *gradient rules* it exercises rather than
    for realism: the broadcast ``Add`` (whose VJP is the ``ReduceSum`` +
    ``Reshape`` un-broadcast), ``Reshape``, ``Div`` (whose VJP is where ``Neg``
    comes from), ``Transpose`` and ``ReduceSum`` are the three operators
    neither rounding step graph above contains.
    """
    rows, kin, hidden = 4, 6, 5
    x_shape = (rows, kin)
    w_shape = (kin, hidden)
    b_shape = (hidden,)
    flat_shape = (2, 10)  # rows * hidden == 20
    out_shape = (10, 1)

    b = qat_graph.GraphBuilder("qat_")
    # 1. The forward slice, node for node as a block of a real graph would be.
    denom = b.const(_f32(rng.uniform(1.0, 2.0, size=flat_shape)), "denom")
    shape_const = onnx.numpy_helper.from_array(
        np.asarray(flat_shape, dtype=np.int64), "qat_flat_shape"
    )
    axes_const = onnx.numpy_helper.from_array(np.asarray([1], dtype=np.int64), "qat_axes")
    b.initializer.extend([shape_const, axes_const])

    forward = [
        onnx.helper.make_node("MatMul", ["x", "w"], ["h"]),
        onnx.helper.make_node("Sigmoid", ["h"], ["hs"]),
        onnx.helper.make_node("Add", ["hs", "bias"], ["hb"]),
        onnx.helper.make_node("Reshape", ["hb", "qat_flat_shape"], ["hr"]),
        onnx.helper.make_node("Div", ["hr", denom], ["hd"]),
        onnx.helper.make_node("Transpose", ["hd"], ["ht"]),
        onnx.helper.make_node("ReduceSum", ["ht", "qat_axes"], ["y"], keepdims=1),
    ]
    b.nodes.extend(forward)

    shapes = {
        "x": x_shape,
        "w": w_shape,
        "bias": b_shape,
        "h": (rows, hidden),
        "hs": (rows, hidden),
        "hb": (rows, hidden),
        "hr": flat_shape,
        denom: flat_shape,
        "hd": flat_shape,
        "ht": tuple(reversed(flat_shape)),
        "y": out_shape,
    }

    # 2. The objective, and 3. its gradient as the backward pass's seed.
    diff = b.sub("y", "teacher")
    dl_dy = b.mul(diff, b.const(2.0 / float(np.prod(out_shape))))
    grads = graph_grad.build_backward(b, forward, shapes, {"y": dl_dy}, ["w", "bias"])

    # 4. One Adam step per trained tensor.
    w_next, mw_next, vw_next = qat_graph.adam_update(
        b, "w", grads["w"], "mw", "vw", "lr", "m_correction", "v_correction"
    )
    b_next, mb_next, vb_next = qat_graph.adam_update(
        b, "bias", grads["bias"], "mb", "vb", "lr", "m_correction", "v_correction"
    )
    step = qat_graph.make_step_graph(
        b,
        constants={
            "x": (list(x_shape), onnx.TensorProto.FLOAT),
            "teacher": (list(out_shape), onnx.TensorProto.FLOAT),
        },
        state={
            "w": (list(w_shape), w_next),
            "mw": (list(w_shape), mw_next),
            "vw": (list(w_shape), vw_next),
            "bias": (list(b_shape), b_next),
            "mb": (list(b_shape), mb_next),
            "vb": (list(b_shape), vb_next),
        },
        scalars=["lr", "m_correction", "v_correction"],
        loss=b.mean_square(diff),
        name="onnxsim_qat_backward_step",
    )

    x = _f32(rng.normal(size=x_shape))
    constants = {"x": x, "teacher": _f32(rng.normal(scale=0.5, size=out_shape))}
    state = {
        "w": _f32(rng.normal(scale=0.5, size=w_shape)),
        "mw": np.zeros(w_shape, np.float32),
        "vw": np.zeros(w_shape, np.float32),
        "bias": _f32(rng.normal(scale=0.1, size=b_shape)),
        "mb": np.zeros(b_shape, np.float32),
        "vb": np.zeros(b_shape, np.float32),
    }
    scalars = []
    for t in range(NUM_STEPS):
        values = {"lr": 0.1}
        values.update(qat_graph.adam_bias_corrections(t))
        scalars.append(values)
    return _package(
        "qat_backward",
        "step_qat_backward.onnx",
        "onnxsim.graph_grad.build_backward + qat_graph.adam_update/make_step_graph",
        step,
        constants,
        state,
        scalars,
    )


def build_minibatch(rng: np.random.Generator) -> Dict:
    """A step that reads its batch out of a resident calibration set with
    ``GraphBuilder.gather_rows``.

    This is the only thing ``Gather`` is in ``EP_FRIENDLY_OPS`` for, and the
    only step-graph shape with a non-float input: the row index is a rank-1
    int64 per-step input, declared through ``make_step_graph``'s ``per_step``.
    Both halves are the point of the fixture -- an EP that implements every
    arithmetic operator in the set and still cannot take an int64 input cannot
    run a minibatched loop at all.

    The forward is a plain linear reconstruction (the objective every rounding
    pass in the tree optimizes, minus the quantizer) so that the gathering, not
    the arithmetic around it, is what the fixture is about.
    """
    total, batch, kin, out = 16, 4, 6, 3

    b = qat_graph.GraphBuilder("mb_")
    index = "index"
    xb = b.gather_rows("x_all", index)
    yb = b.gather_rows("y_all", index)
    y_hat = b.matmul(xb, "w")
    diff = b.sub(y_hat, yb)
    dl_dy = b.mul(diff, b.const(2.0 / float(batch * out)))
    grad = b.matmul(b.transpose(xb), dl_dy)  # [kin, out]
    w_next, mw_next, vw_next = qat_graph.adam_update(
        b, "w", grad, "mw", "vw", "lr", "m_correction", "v_correction"
    )
    step = qat_graph.make_step_graph(
        b,
        constants={
            "x_all": ([total, kin], onnx.TensorProto.FLOAT),
            "y_all": ([total, out], onnx.TensorProto.FLOAT),
        },
        state={
            "w": ([kin, out], w_next),
            "mw": ([kin, out], mw_next),
            "vw": ([kin, out], vw_next),
        },
        scalars=["lr", "m_correction", "v_correction"],
        per_step={index: ([batch], onnx.TensorProto.INT64)},
        loss=b.mean_square(diff),
        name="onnxsim_minibatch_step",
    )

    x_all = _f32(rng.normal(size=(total, kin)))
    w_true = _f32(rng.normal(scale=0.5, size=(kin, out)))
    constants = {"x_all": x_all, "y_all": _f32(x_all @ w_true)}
    state = {
        "w": np.zeros((kin, out), np.float32),
        "mw": np.zeros((kin, out), np.float32),
        "vw": np.zeros((kin, out), np.float32),
    }
    scalars = []
    per_step = []
    for t in range(NUM_STEPS):
        values = {"lr": 0.1}
        values.update(qat_graph.adam_bias_corrections(t))
        scalars.append(values)
        # A different batch every step, which is what makes the index a
        # per-step input rather than another constant.
        per_step.append(
            {index: rng.permutation(total)[:batch].astype(np.int64)}
        )
    return _package(
        "minibatch",
        "step_minibatch.onnx",
        "qat_graph.GraphBuilder.gather_rows + adam_update/make_step_graph",
        step,
        constants,
        state,
        scalars,
        per_step,
    )


def build_qat_hf_demo(rng: np.random.Generator) -> Dict:
    """A tiny two-layer QAT step graph, structurally identical in spirit to
    ``build_qat_backward`` but meant to be *driven*, not replayed: unlike
    every other fixture here, its ``x``/``teacher`` constants are declared
    (as ``make_step_graph`` always declares constants -- see that function's
    docstring) but never baked into ``step_qat_hf_demo.json`` with values,
    because the whole point of ``webgpu_hf_demo.test.mjs`` is to feed them
    something this script cannot reach: a real photo fetched live from a
    Hugging Face dataset, inside a real browser, on WebGPU. This script runs
    in plain Python with no network access and has no opinion about what that
    photo will be -- it only fixes the *shape* of the problem.

    The forward is deliberately smaller than ``build_qat_backward``'s (two
    ``MatMul``s, two ``Add``s, one ``Sigmoid`` -- all already covered by other
    fixtures, so this adds no new operator to verify) and takes a flattened
    8x8 grayscale image (64 features) down to a single scalar, regressed
    toward a fixed zero target. "Fit a real photo to zero" is an arbitrary
    objective, not a meaningful one -- chosen because it needs no labels, only
    a real ``x``, which is all ``hf_datasets.mjs`` promises. What this proves
    is that the whole pipeline (a real photo in, a real gradient/Adam step
    graph, WebGPU execution) runs end to end and the loss actually moves.
    """
    in_dim, hidden = 64, 8
    b = qat_graph.GraphBuilder("hfdemo_")
    forward = [
        onnx.helper.make_node("MatMul", ["x", "w1"], ["h0"]),
        onnx.helper.make_node("Add", ["h0", "b1"], ["h1"]),
        onnx.helper.make_node("Sigmoid", ["h1"], ["h2"]),
        onnx.helper.make_node("MatMul", ["h2", "w2"], ["h3"]),
        onnx.helper.make_node("Add", ["h3", "b2"], ["y"]),
    ]
    b.nodes.extend(forward)
    shapes = {
        "x": (1, in_dim),
        "w1": (in_dim, hidden),
        "b1": (hidden,),
        "h0": (1, hidden),
        "h1": (1, hidden),
        "h2": (1, hidden),
        "w2": (hidden, 1),
        "b2": (1,),
        "h3": (1, 1),
        "y": (1, 1),
    }

    diff = b.sub("y", "teacher")
    dl_dy = b.mul(diff, b.const(2.0))
    grads = graph_grad.build_backward(
        b, forward, shapes, {"y": dl_dy}, ["w1", "b1", "w2", "b2"]
    )

    state_vars = {}
    for name, grad_name, shape in [
        ("w1", grads["w1"], (in_dim, hidden)),
        ("b1", grads["b1"], (hidden,)),
        ("w2", grads["w2"], (hidden, 1)),
        ("b2", grads["b2"], (1,)),
    ]:
        next_v, next_m, next_v2 = qat_graph.adam_update(
            b, name, grad_name, f"m_{name}", f"v_{name}", "lr",
            "m_correction", "v_correction",
        )
        state_vars[name] = (shape, next_v, f"m_{name}", next_m, f"v_{name}", next_v2)

    state = {}
    for name, (shape, next_v, m_name, next_m, v_name, next_v2) in state_vars.items():
        state[name] = (list(shape), next_v)
        state[m_name] = (list(shape), next_m)
        state[v_name] = (list(shape), next_v2)

    step = qat_graph.make_step_graph(
        b,
        constants={
            "x": ([1, in_dim], onnx.TensorProto.FLOAT),
            "teacher": ([1, 1], onnx.TensorProto.FLOAT),
        },
        state=state,
        scalars=["lr", "m_correction", "v_correction"],
        loss=b.mean_square(diff),
        name="onnxsim_qat_hf_demo_step",
    )
    onnx.checker.check_model(step.model, full_check=True)
    onnx.save(step.model, HERE / "step_qat_hf_demo.onnx")

    # A quick offline sanity check with synthetic data (this script has no
    # network access): the graph should still be a working optimizer step,
    # decreasing loss on *some* input, before the browser ever sees it with a
    # real one. Not shipped as a "match this" reference -- see the docstring.
    init_state = {
        "w1": _f32(rng.normal(scale=0.3, size=(in_dim, hidden))),
        "b1": np.zeros(hidden, np.float32),
        "w2": _f32(rng.normal(scale=0.3, size=(hidden, 1))),
        "b2": np.zeros(1, np.float32),
    }
    for name in list(init_state):
        init_state[f"m_{name}"] = np.zeros_like(init_state[name])
        init_state[f"v_{name}"] = np.zeros_like(init_state[name])
    sanity_x = _f32(rng.normal(size=(1, in_dim)))
    sanity_teacher = np.zeros((1, 1), np.float32)
    # lr is deliberately small: at t=0 Adam's bias correction makes its
    # first update roughly lr * sign(gradient) regardless of magnitude (see
    # this function's own commit message/PR description for the derivation),
    # which for a batch of one real photo can otherwise overshoot hard enough
    # to make the loss curve look broken rather than noisy-but-improving.
    NUM_DEMO_STEPS = 40
    sanity_scalars = []
    for t in range(NUM_DEMO_STEPS):
        values = {"lr": 0.01}
        values.update(qat_graph.adam_bias_corrections(t))
        sanity_scalars.append(values)
    sanity_losses = _reference_losses(
        step,
        {"x": sanity_x, "teacher": sanity_teacher},
        init_state,
        sanity_scalars,
        [],
    )
    if not (sanity_losses[-1] < 0.5 * sanity_losses[0]):
        raise SystemExit(
            "step_qat_hf_demo.onnx sanity check: loss did not meaningfully "
            f"decrease over {NUM_DEMO_STEPS} synthetic steps "
            f"({sanity_losses[0]:.6g} -> {sanity_losses[-1]:.6g}); the graph "
            "is broken before it ever reaches a browser"
        )

    manifest = {
        "file": "step_qat_hf_demo.onnx",
        "opset": qat_graph._OPSET,
        "irVersion": qat_graph._IR_VERSION,
        "loss": step.loss_name,
        "inputDim": in_dim,
        # Declared shape/dtype only -- no baked values. webgpu_hf_demo.test.mjs
        # supplies "x" (a real photo, flattened+resized to [1, inputDim]) and
        # "teacher" (fixed zero) itself.
        "constants": {
            "x": {"dims": [1, in_dim], "dtype": "float32"},
            "teacher": {"dims": [1, 1], "dtype": "float32"},
        },
        "state": {
            name: {"dims": list(shape), "output": out, "data": [float(v) for v in init_state[name].ravel()]}
            for name, (shape, out) in state.items()
        },
        "scalars": [dict(s) for s in sanity_scalars],
    }
    (HERE / "step_qat_hf_demo.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        f"wrote step_qat_hf_demo.onnx + step_qat_hf_demo.json; offline sanity "
        f"loss {sanity_losses[0]:.6g} -> {sanity_losses[-1]:.6g}"
    )


def _package(
    name: str,
    filename: str,
    builder: str,
    step: qat_graph.StepGraph,
    constants: Dict[str, np.ndarray],
    state: Dict[str, np.ndarray],
    scalars: Sequence[Dict[str, float]],
    per_step: Sequence[Dict[str, np.ndarray]] = (),
) -> Dict:
    """Check the graph, write it, and describe it for the manifest."""
    onnx.checker.check_model(step.model, full_check=True)
    onnx.save(step.model, HERE / filename)
    losses = _reference_losses(step, constants, state, scalars, per_step)
    return {
        "name": name,
        "file": filename,
        "builder": builder,
        "ops": _op_histogram(step.model),
        "loss": step.loss_name,
        "constants": {k: _tensor_entry(v) for k, v in constants.items()},
        "state": {
            k: dict(_tensor_entry(v), output=step.state[k]) for k, v in state.items()
        },
        "scalars": [dict(s) for s in scalars],
        # Per-step non-float inputs, one entry per step, carrying their dtype
        # so the Node test can build the right typed array.
        "perStep": [
            {k: {"dims": list(v.shape), "dtype": str(v.dtype), "data": [int(i) for i in v.ravel()]}
             for k, v in values.items()}
            for values in per_step
        ],
        "referenceLosses": losses,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    graphs = [
        build_adaround(rng),
        build_adaquant(rng),
        build_qat_backward(rng),
        build_minibatch(rng),
    ]

    covered = set()
    for graph in graphs:
        covered |= set(graph["ops"])
    missing = sorted(qat_graph.EP_FRIENDLY_OPS - covered)
    if missing:
        raise SystemExit(
            "these EP_FRIENDLY_OPS members appear in no fixture, so the Node "
            f"test cannot say anything about them: {missing}. Extend one of "
            "the graphs above (or record deliberately why it cannot be "
            "covered) rather than shipping an unverifiable claim."
        )
    # The converse would mean a builder emitting an op its own allowlist bans;
    # the python tests assert it too, but a fixture is what the Node test
    # actually runs, so check the thing being shipped.
    extra = sorted(covered - set(qat_graph.EP_FRIENDLY_OPS))
    if extra:
        raise SystemExit(f"fixture contains ops outside EP_FRIENDLY_OPS: {extra}")

    manifest = {
        "epFriendlyOps": sorted(qat_graph.EP_FRIENDLY_OPS),
        "opset": qat_graph._OPSET,
        "irVersion": qat_graph._IR_VERSION,
        "numSteps": NUM_STEPS,
        "seed": SEED,
        "graphs": graphs,
    }
    (HERE / "step_graphs.json").write_text(json.dumps(manifest, indent=1) + "\n")

    for graph in graphs:
        print(
            f"wrote {graph['file']}: {sum(graph['ops'].values())} nodes, "
            f"{len(graph['ops'])} distinct ops, "
            f"loss {graph['referenceLosses'][0]:.6g} -> "
            f"{graph['referenceLosses'][-1]:.6g}"
        )
    print(f"wrote step_graphs.json; {len(covered)} of EP_FRIENDLY_OPS covered")

    # Separate from the four above: this one is driven with live data by
    # webgpu_hf_demo.test.mjs, not replayed against a baked reference
    # trajectory, so it gets its own file rather than joining step_graphs.json
    # (see build_qat_hf_demo's own docstring).
    build_qat_hf_demo(np.random.default_rng(SEED))


if __name__ == "__main__":
    main()
