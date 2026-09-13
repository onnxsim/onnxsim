#!/usr/bin/env python3
"""Build a resnet18-shaped training-step graph with the weight update *in
the graph*, for a resident runner to keep entirely device-side.

`finetune.py`'s loop treats a compiled training step as a pure function:
feed weights and a batch, get back a gradient, and do `w -= lr * grad` in
host numpy. That means every trainable weight crosses the host<->device
boundary twice a step (once as input, once as the returned gradient) for no
reason but that the update lives off-device -- 42.9 MB of the resnet18
step's 42.9 MB moved is exactly this, measured in
`docs/axera-on-device-training-handoff.md`.

This module builds a different graph for the same step: each trainable
weight is a **state** tensor (`onnxsim.qat_graph.StepGraph`'s sense) that is
both an input and an output, with `w_next = w - lr * grad` computed by
ordinary `Mul`/`Sub` nodes inside the graph itself (plain SGD, no momentum --
matching `finetune.train()`'s own host update exactly, so the two are
directly comparable). A resident runner
(`scripts/axera/tools/resident_runner.c`) can then bind each state output
back to its own input's device buffer between `Execute()` calls and never
send the weight across the host boundary at all; only the batch (`x`, `y`)
goes in and the scalar loss comes out.

Pipeline, and why the order matters:

1. Append a scalar loss to the forward model (`Sub`/`Mul`/`ReduceMean`, with
   explicit `axes` -- see `legalize.avgpool_ceil_to_floor`'s and
   `global_pool_to_reduce`'s own docstrings for the vendor bug this dodges).
2. Legalize *forward*-graph blockers that are differentiability blockers,
   not just Pulsar2-compile-time ones: `avgpool_ceil_to_floor`,
   `flatten_to_reshape`, `global_pool_to_reduce`. `onnxsim.graph_grad` has no
   rule for `Flatten`/`GlobalAveragePool` and declines `ceil_mode=1`
   outright, so these three must run *before* `build_backward`, not after
   (unlike the rest of `legalize.TRAINING_RULES`, which only needs to run on
   the finished step graph).
3. `onnxsim.graph_grad.build_backward()` differentiates the (now legal)
   forward+loss graph against the chosen trainable weights.
4. Plain SGD, in-graph: `step = lr * grad; w_next = w - step` per weight,
   via `onnxsim.qat_graph.GraphBuilder.mul`/`.sub` directly -- not
   `qat_graph.sgd_momentum_update`, which always carries a momentum buffer
   as extra state; this is the zero-momentum case `finetune.py`'s host loop
   already implements, so there is nothing to gain from carrying one.
5. `onnxsim.qat_graph.make_step_graph()` wraps it into a `StepGraph` and
   simplifies with the same skip list `docs/axera-on-device-training-
   handoff.md`'s "Simplify the gradient graph" section describes
   (`fuse_matmul_add_bias_into_gemm`/`fuse_transpose_into_gemm` -- a training
   graph's whole point is a *live* weight, and both fusions would rebuild
   exactly the `Gemm`/transpose-into-`Gemm` shape this graph exists to
   avoid).
6. The remaining `legalize.TRAINING_RULES` (`inline_local_functions`,
   `avgpool_ceil_to_floor` again for any pool the update graph itself
   introduced, `neg_to_mul`, `rank0_to_rank1`, `gemm_to_matmul`,
   `act_weight_conv_to_matmul`) make the *step* graph -- backward pass and
   update included -- something `pulsar2 build` will actually lower.
7. One more `onnxsim.simplify()` pass, same skip list: `act_weight_conv_to_
   matmul`'s per-tap expansion multiplies node count, and consecutive taps
   slicing the same tensors collapse under common-subexpression elimination
   -- worth doing after legalization, not only before it (same finding the
   handoff's own "Simplify the gradient graph" section records).

One graph-construction quirk worth naming: `qat_graph.make_step_graph`
declares every scalar input (here, just `lr`) at rank 0. Pulsar2's
calibration step concatenates each input across calibration samples, and a
rank-0 tensor cannot be concatenated -- the same failure
`legalize.rank0_to_rank1` exists to fix for a rank-0 *output* (the loss).
There is no equivalent rule for a rank-0 *input*, so this module reshapes
`lr` to `[1]` directly after `make_step_graph` returns; broadcasting a `[1]`
scalar against every weight's `Mul` is identical arithmetic.

See `tests/test_build_resident_train_step.py` for a from-scratch synthetic
model exercising this whole pipeline without resnet18 or real hardware, and
`docs/axera-on-device-training-handoff.md` for the measured speed this
bought on a real AX650N.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _local_import import ensure_repo_onnxsim, fresh  # noqa: E402

ensure_repo_onnxsim()

# scripts/axera/legalize.py and scripts/axelera/legalize.py are two
# different, same-named modules sharing one `sys.modules["legalize"]` entry
# -- a plain `import legalize` risks silently getting axelera's copy if
# something upstream already claimed that bare name. `fresh` reloads
# directly from this file's own directory regardless of what's cached.
legalize = fresh("legalize", HERE)

from onnxsim import graph_grad, qat_graph  # noqa: E402
from onnxsim.compile_training import _static_shapes_and_types  # noqa: E402

#: Applied to the *step* graph (forward + loss + backward + in-graph
#: update), after `make_step_graph`'s own simplify. Mirrors
#: `legalize.TRAINING_RULES` minus the three rules that had to run on the
#: forward graph earlier, before differentiation -- see this module's
#: docstring.
_POST_BACKWARD_RULES = (
    "inline_local_functions",
    "avgpool_ceil_to_floor",
    "neg_to_mul",
    "rank0_to_rank1",
    "gemm_to_matmul",
    "act_weight_conv_to_matmul",
)


def _linearize_trainable_convs(
    model: onnx.ModelProto, params: Sequence[str]
) -> onnx.ModelProto:
    """Rewrites each `Conv` node whose weight is in `params` from `Conv(x,
    w)` into the *same* im2col-as-gather identity `onnxsim.graph_grad`'s own
    `_grad_conv` already uses internally for this op's backward -- one
    `Gather`/`Mul`(mask)/`MatMul`, no `Transpose` of the weight at all.

    Why this exists, and why `legalize.act_weight_conv_to_matmul` is not
    enough: this weight is about to become **state** -- an input *and* an
    output of a graph run repeatedly by a resident runner that keeps state
    device-side between calls (see this module's docstring). Left as a plain
    `Conv(x, w)`, `act_weight_conv_to_matmul` legalizes it by transposing `w`
    from `[Cout, Cin, k...]` into `[k..., Cin, Cout]` *inside the compiled
    graph* -- correct for a one-shot inference graph, but on a real AX650N
    that transpose is recomputed from scratch on every `Execute()` call even
    though the weight barely changes step to step: measured (real hardware,
    `pulsar2 build --compiler.npu_perf`) at 89.6% of the whole step's
    `AxTranspose` cost and 13.3% of total NPU cycles, entirely from the two
    512-channel trainable convs (`docs/axera-on-device-training-
    handoff.md`'s "AxTranspose/AxSlice glue" section).

    The fix is not to *move* that transpose to build time (an earlier
    version of this function tried exactly that, pre-transposing the weight
    once in host numpy and keeping state in the transposed layout -- it
    numerically works, but `graph_grad.build_backward` cannot differentiate
    the `Pad`/`Slice`/`Concat` nodes `act_weight_conv_to_matmul`'s own
    construction needs, none of which have a gradient rule, built-in or
    registerable without hand-writing three new ones). The fix is to **avoid
    needing a weight transpose in the first place**: `_grad_conv`'s own
    docstring spells out the identity --

        col[c, t, o] = X[c, position(o, t)]      (im2col: one gather)
        Y[m, o]      = sum_{c, t} W[m, c, t] * col[c, t, o]

    -- where "W reshaped to `[M, C*K]`" is `w.reshape(Cout, -1)`, a plain
    C-order reshape of the *original* `[Cout, Cin, k...]` layout with no
    data movement at all, because `Cin` is already `W`'s second axis and
    `k...` are already trailing -- unlike `act_weight_conv_to_matmul`'s
    `[k..., Cin, Cout]`, which moves `Cout` from first to last. Building the
    forward pass with this identity (reusing `graph_grad`'s own
    `_conv_geometry`/`_im2col_indices` so the index/mask tables are exactly
    the ones `_grad_conv` would derive for the same node) means: no weight
    transpose ever appears, `Gather`/`Mul`/`MatMul`/`Reshape` are all
    builtin-differentiable so `build_backward` needs no custom gradient
    registration either, and the gradient it produces is already in `w`'s
    original, unchanged shape -- so **state stays exactly the shape it
    always was**, nothing above this function (`params`, `state`,
    `shapes[p]`) changes at all.

    A `Conv` this function declines (see `_conv_geometry`'s own refusals:
    not 1-D/2-D, `auto_pad`, a geometry that does not reproduce its declared
    output shape, ...; grouped convs are declined here directly, since the
    identity above assumes `group=1`) is left untouched and still caught by
    `act_weight_conv_to_matmul` later in `_POST_BACKWARD_RULES` -- this
    function is an optimization for the common case (an ungrouped 1-D/2-D
    conv, which is every trainable conv resnet18 has), not a superset of
    that rule's coverage.
    """
    shapes = legalize._value_shapes(model)
    initializers = {t.name: t for t in model.graph.initializer}
    trained = set(params)
    out_nodes = []
    linearized = set()

    for node in model.graph.node:
        w = node.input[1] if node.op_type == "Conv" and len(node.input) > 1 else None
        if w is None or w not in trained or not legalize._is_initializer(model, w):
            out_nodes.append(node)
            continue

        w_array = numpy_helper.to_array(initializers[w])
        x_shape, y_shape = shapes.get(node.input[0]), shapes.get(node.output[0])
        if not x_shape or not y_shape:
            out_nodes.append(node)
            continue
        try:
            group, kernel, strides, dilations, pads_begin = graph_grad._conv_geometry(
                node, tuple(x_shape), tuple(w_array.shape), tuple(y_shape)
            )
        except graph_grad.UnsupportedOpError:
            out_nodes.append(node)
            continue
        if group != 1:
            out_nodes.append(node)  # the reshape identity below assumes group=1
            continue

        cout, cin = int(w_array.shape[0]), int(w_array.shape[1])
        in_dims, out_dims = [int(d) for d in x_shape[2:]], [int(d) for d in y_shape[2:]]
        taps = graph_grad._prod(kernel)
        in_count, out_count = graph_grad._prod(in_dims), graph_grad._prod(out_dims)
        index, mask = graph_grad._im2col_indices(
            in_dims, out_dims, kernel, strides, dilations, pads_begin
        )

        stem = node.name or node.output[0]
        made = []

        def const(array, hint):
            name = legalize._unique_name(model, f"{stem}_{hint}")
            model.graph.initializer.append(numpy_helper.from_array(array, name))
            return name

        def op(op_type, inputs, hint, **attrs):
            out = legalize._unique_name(model, f"{stem}_{hint}")
            made.append(
                helper.make_node(
                    op_type,
                    inputs,
                    [out],
                    name=legalize._unique_name(model, f"{stem}_{hint.upper()}"),
                    **attrs,
                )
            )
            return out

        batch = int(x_shape[0])
        x3 = op(
            "Reshape",
            [node.input[0], const(np.array([batch, cin, in_count], np.int64), "x3s")],
            "x3",
        )
        gathered = op(
            "Gather",
            [x3, const(np.array(index, np.int64), "idx")],
            "gathered",
            axis=2,
        )
        masked = op(
            "Mul",
            [
                gathered,
                const(mask.reshape(1, 1, taps * out_count).astype(np.float32), "mask"),
            ],
            "masked",
        )
        col = op(
            "Reshape",
            [masked, const(np.array([batch, cin * taps, out_count], np.int64), "cols")],
            "col",
        )
        w2 = op(
            "Reshape",
            [w, const(np.array([cout, cin * taps], np.int64), "w2s")],
            "w2",
        )
        y3 = op(
            "MatMul", [w2, col], "mm"
        )  # [Cout,C*K] x [batch,C*K,out] -> [batch,Cout,out]
        if len(node.input) > 2 and node.input[2]:
            bias4 = op(
                "Reshape",
                [node.input[2], const(np.array([1, cout, 1], np.int64), "bs")],
                "bias3",
            )
            y3 = op("Add", [y3, bias4], "biased")
        # [batch, Cout, out_count] -> [batch, Cout, *out_dims], the original
        # Conv's own declared output shape/name.
        made.append(
            helper.make_node(
                "Reshape",
                [y3, const(np.array([batch, cout] + out_dims, np.int64), "ys")],
                [node.output[0]],
                name=legalize._unique_name(model, f"{stem}_Y"),
            )
        )

        out_nodes.extend(made)
        linearized.add(w)

    if linearized:
        del model.graph.node[:]
        model.graph.node.extend(out_nodes)
    return model


def set_batch(model: onnx.ModelProto, batch: int) -> onnx.ModelProto:
    """Returns a copy of `model` with its first input's leading (batch) dim
    set to `batch`, and every cached shape downstream of it invalidated so
    the next shape-inference pass (every caller in this module runs one --
    `legalize._value_shapes`/`_static_shapes_and_types`) recomputes them
    instead of reading stale ones.

    `resnet18d_folded.onnx` (and any similarly-exported forward model) has no
    batch-specific shape baked in anywhere downstream of `x`: pooling
    (`AveragePool`/`GlobalAveragePool`), `Flatten`
    (`legalize.flatten_to_reshape` reads the batch dim from the *current*
    static shape at legalize time, not a constant), and `Gemm` are all
    batch-preserving ops with no reshape target that names `1` explicitly.
    So changing just the declared input shape and clearing the stale cached
    ones is the whole fix -- see `docs/axera-on-device-training-
    handoff.md`'s batch-scaling section for the real batch sweep this makes
    possible.
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    g = out.graph
    g.input[0].type.tensor_type.shape.dim[0].Clear()
    g.input[0].type.tensor_type.shape.dim[0].dim_value = batch
    del g.value_info[:]
    for value in g.output:
        if value.type.tensor_type.shape.dim:
            value.type.tensor_type.shape.dim[0].Clear()
    return out


def add_mse_loss(
    model: onnx.ModelProto, logits: str, num_classes: int
) -> onnx.ModelProto:
    """Returns a copy of `model` with a `y` input and a scalar MSE `loss`
    output appended: `loss = mean((logits - y) ** 2)`.

    A plain elementwise MSE is the simplest loss `graph_grad` differentiates
    exactly, and is all this module needs to demonstrate the in-graph
    update -- swap in a different loss (cross-entropy, ...) by building one
    yourself and skipping this helper.

    `y`'s batch dimension is read from `logits`'s own static shape (via shape
    inference), not hardcoded to 1 -- `model`'s declared input batch size is
    what determines the whole step graph's batch size (see
    `docs/axera-on-device-training-handoff.md`'s batch-scaling section), and
    `ReduceMean(axes=[0, 1])` below already averages over *both* the batch
    and class axes regardless of what the batch dimension is, so nothing
    else in this function is batch-size-specific.
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    g = out.graph
    logits_shape = legalize._value_shapes(model)[logits]
    batch = int(logits_shape[0])
    g.input.append(
        helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, num_classes])
    )
    g.node.extend(
        [
            helper.make_node("Sub", [logits, "y"], ["loss_diff"], name="loss_diff"),
            helper.make_node(
                "Mul", ["loss_diff", "loss_diff"], ["loss_sq"], name="loss_sq"
            ),
            helper.make_node(
                "ReduceMean",
                ["loss_sq"],
                ["loss"],
                name="loss_mean",
                axes=[0, 1],
                keepdims=0,
            ),
        ]
    )
    g.output.append(helper.make_tensor_value_info("loss", TensorProto.FLOAT, []))
    return out


def add_resident_dataset(
    model: onnx.ModelProto,
    data: Dict[str, np.ndarray],
    index_name: str = "batch_index",
    flatten: bool = True,
) -> onnx.ModelProto:
    """Returns a copy of `model` with each of `data`'s named inputs replaced
    by a `Gather` off a resident constant, selected by one shared per-step
    index vector -- `onnxsim.qat_graph.GraphBuilder.gather_rows`'s pattern
    (see its own docstring) applied to this pipeline for the first time.

    `flatten` (default `True`) stores and gathers every rank>=3 array as a
    flat `[N, prod(shape[1:])]` initializer, `Reshape`-ing each gathered row
    back to its real shape (`[batch, *shape[1:]]`) immediately afterward,
    rather than gathering the native `[N, C, H, W]` shape directly. This is
    the workaround `docs/axera-on-device-training-handoff.md`'s "Trading
    free memory for throughput" section names and confirms fixes a real
    Pulsar2 NPU-backend gap: `Gather` over a 4D, conv-activation-shaped
    resident tensor fails identically in Pulsar2's backend compiler
    regardless of dataset size, while the same `Gather` over a flat 2D
    tensor (plus an ordinary, separately-supported `Reshape`) compiles and
    runs. Pass `flatten=False` to get the old, pre-workaround behavior (e.g.
    to reproduce that failure, or for a rank<=2 dataset where flattening is
    a no-op anyway).

    Every training step measured for this pipeline so far re-uploads `x`/`y`
    fresh every call (`resident_runner.c`'s main loop:
    `axclrtMemcpy(in_bufs[x_in], hx, ..., AXCL_MEMCPY_HOST_TO_DEVICE)`, every
    iteration) even though the trainable weights are already kept resident.
    This is a different, complementary residency: `data[name]`'s full
    `[N, ...]` array becomes a plain graph **initializer** -- baked into the
    compiled `.axmodel` and resident on-device from load, the same way a
    frozen conv weight already is, not merely "uploaded once" the way
    `qat_graph`'s own `bind_loop` keeps a bound tensor resident for an
    onnxruntime session. What crosses the host boundary each step is
    `index_name`, a rank-1 `int64` of length `batch` (`batch` read from
    `data[name]`'s *current* declared input shape, i.e. call this after
    `set_batch`, not before) -- `qat_graph.minibatch_indices` already
    generates the exact index stream this expects.

    All of `data`'s tensors are gathered by the *same* `index_name` -- they
    must therefore share row count `N` along axis 0 (true for any `x`/`y`
    pair drawn from one dataset; not checked here beyond the `Gather`'s own
    shape inference catching a mismatch).

    Call this on the forward+loss model, before `build_resident_step`: the
    replaced inputs must already be gone by the time `build_resident_step`
    walks `model.graph.input` to decide the step graph's own per-step
    constants, and `graph_grad.build_backward` needs the inserted `Gather`
    nodes present in the graph it walks (Gather already has a rule --
    `graph_grad._grad_gather`, exercised elsewhere in this pipeline for
    conv-as-matmul taps -- so no new gradient machinery is needed; nothing
    downstream ever asks for a gradient *of* the dataset or the index, since
    neither is in `build_resident_step`'s `params`).
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    g = out.graph

    kept_inputs = [inp for inp in g.input if inp.name not in data]
    del g.input[:]
    g.input.extend(kept_inputs)

    batch = None
    gathers = []
    for name, array in data.items():
        array = np.asarray(array, dtype=np.float32)
        row_shape = array.shape[1:]
        do_flatten = flatten and array.ndim > 2
        if do_flatten:
            stored = array.reshape(array.shape[0], -1)
        else:
            stored = array
        g.initializer.append(numpy_helper.from_array(stored, f"{name}_dataset"))
        gather_out = f"{name}_gathered" if do_flatten else name
        gathers.append(
            helper.make_node(
                "Gather",
                [f"{name}_dataset", index_name],
                [gather_out],
                name=f"gather_{name}",
                axis=0,
            )
        )
        if do_flatten:
            shape_init = numpy_helper.from_array(
                np.array([-1, *row_shape], dtype=np.int64), f"{name}_reshape_shape"
            )
            g.initializer.append(shape_init)
            gathers.append(
                helper.make_node(
                    "Reshape",
                    [gather_out, f"{name}_reshape_shape"],
                    [name],
                    name=f"reshape_{name}",
                )
            )
        shape = legalize._value_shapes(model)[name]
        if batch is None:
            batch = int(shape[0])
        elif batch != int(shape[0]):
            raise ValueError(
                f"{name!r}'s declared batch {shape[0]} doesn't match the "
                f"other resident inputs' {batch} -- call set_batch first so "
                "every gathered input agrees on batch size"
            )

    # Insert before every other node: the gathers must run before anything
    # that reads `x`/`y` by name, and nothing in `data` depends on anything
    # else in the graph, so the front of the list is always valid.
    new_nodes = gathers + list(g.node)
    del g.node[:]
    g.node.extend(new_nodes)

    g.input.append(
        helper.make_tensor_value_info(index_name, TensorProto.INT64, [batch])
    )
    del g.value_info[:]
    return out


def _fold_constants(model: onnx.ModelProto) -> onnx.ModelProto:
    """Removes every `Constant` node, folding each into an initializer of
    the same name/value -- a `Constant` node is an initializer wearing a
    node's clothes: same `TensorProto`, no inputs.

    Runs `onnxsim.simplify()` first (same skip list `build_resident_step`'s
    own final simplify uses -- `fuse_matmul_add_bias_into_gemm`/`fuse_
    transpose_into_gemm` skipped, so this doesn't reshuffle which
    initializer name carries which weight before the caller picks `params`
    by name), which folds most duplicate `Constant`s via CSE; whatever
    survives that (not every model's constants collapse to a single shared
    one) is folded here by hand, since `simplify()` does not guarantee zero
    remain.

    `initializers_as_constants=False`: this runs *before* `params` are
    promoted to state I/O (still plain initializers at this point), and
    `initializers_as_constants=True` (`simplify()`'s own default) lets the
    optimizer fold/eliminate them as ordinary constant data -- found
    concretely by `scripts/axera/legalize.py`'s `unroll_gru` output: with
    the default, this call silently dropped the per-gate `W`/`R`
    initializers `unroll_gru` had just created, no error, `build_resident_
    step` only noticing several steps later when `params` could no longer
    be found. `unroll_lstm`'s own output happened not to trigger this same
    optimizer behavior, which is exactly the kind of silent, model-specific
    failure this flag closes off generally rather than routing around once.
    """
    from onnxsim import simplify as _simplify

    model, ok = _simplify(
        model,
        initializers_as_constants=False,
        skipped_optimizers=[
            "fuse_matmul_add_bias_into_gemm",
            "fuse_transpose_into_gemm",
        ],
    )
    if not ok:
        raise RuntimeError(
            "pre-backward constant-folding simplify() failed its own check"
        )

    fold_nodes = [n for n in model.graph.node if n.op_type == "Constant"]
    if not fold_nodes:
        return model
    keep = [n for n in model.graph.node if n.op_type != "Constant"]
    del model.graph.node[:]
    model.graph.node.extend(keep)
    for n in fold_nodes:
        (attr,) = n.attribute
        t = onnx.TensorProto()
        t.CopyFrom(attr.t)
        t.name = n.output[0]
        model.graph.initializer.append(t)
    return model


def build_resident_step(
    forward_and_loss: onnx.ModelProto,
    params: Sequence[str],
    loss_output: str = "loss",
) -> Tuple[onnx.ModelProto, Dict[str, str]]:
    """The full pipeline (this module's docstring, steps 2-7) over a forward
    model that already has its loss appended (`add_mse_loss`, or your own).

    :param params: names of `forward_and_loss`'s own float32 initializers to
            train -- promoted to state (input *and* output) rather than left
            as fixed initializers.
    :returns: `(step_model, state)`, where `state` maps each trainable
            weight's name to the output name carrying its updated value --
            exactly `qat_graph.StepGraph.state`, restricted to `params`
            (the optimizer's own extra state, if any, is not exposed here
            since plain SGD carries none).
    """
    model = onnx.ModelProto()
    model.CopyFrom(forward_and_loss)

    # Forward-graph blockers that must be fixed before build_backward can
    # even walk the graph -- see this module's docstring, step 2.
    legalize.avgpool_ceil_to_floor(model)
    legalize.flatten_to_reshape(model)
    legalize.global_pool_to_reduce(model)
    onnx.checker.check_model(model)

    # A fourth forward-graph blocker, found training a Whisper encoder (see
    # docs/axera-on-device-training-handoff.md's "A memory-heavy case"
    # section): graph_grad.build_backward demands a gradient rule for every
    # node type it walks, `Constant` included, even though a zero-input op
    # has nothing to backprop through. resnet18d/resnet50d's forward exports
    # never have one, but a raw Erf-GELU decomposition's `0.5`/`sqrt(2)`
    # literals do, and no earlier step here folds them. Only pay for this
    # (a simplify() pass over what may be a large graph) when there is
    # something to fold.
    if any(n.op_type == "Constant" for n in model.graph.node):
        model = _fold_constants(model)
        onnx.checker.check_model(model)

    # Pre-transpose any trainable Conv's weight into matmul layout now, in
    # host numpy, once -- instead of leaving `act_weight_conv_to_matmul` to
    # rebuild that transpose on the device every step. See
    # `_linearize_trainable_convs`'s own docstring for the measured cost
    # this removes; `params`'/`state`'s names are unaffected, only the
    # linearized weights' shapes change.
    model = _linearize_trainable_convs(model, params)
    onnx.checker.check_model(model)

    shapes, elem_types = _static_shapes_and_types(model)
    initializers = {t.name: t for t in model.graph.initializer}
    missing = [p for p in params if p not in initializers]
    if missing:
        raise ValueError(f"{missing} are not initializers of the forward model")

    b = qat_graph.GraphBuilder(prefix="resident_step__")
    b.nodes = list(model.graph.node)
    trained = set(params)
    b.initializer = [t for t in model.graph.initializer if t.name not in trained]

    # A runtime-fed scalar, not `b.const` -- baking the seed in at 1.0 is
    # what made the original loss-scaling probe's finding possible in the
    # first place (a constant can never be varied per step). Declared as a
    # scalar graph input the same way "lr" is, below, via `scalars=`.
    grads = graph_grad.build_backward(
        b,
        nodes=list(model.graph.node),
        shapes=shapes,
        grad_outputs={loss_output: "grad_seed"},
        targets=list(params),
    )

    state: Dict[str, Tuple[Sequence[int], str]] = {}
    for p in params:
        w_shape = tuple(int(d) for d in shapes[p])
        step = b.mul("lr", grads[p])
        w_next = b.sub(p, step)
        state[p] = (w_shape, w_next)

    constants: Dict[str, Tuple[Sequence[int], int]] = {}
    for inp in model.graph.input:
        if inp.name in initializers:
            continue
        shape = shapes.get(inp.name)
        if shape is None:
            raise ValueError(f"no static shape for model input {inp.name!r}")
        constants[inp.name] = (
            tuple(int(d) for d in shape),
            elem_types.get(inp.name, TensorProto.FLOAT),
        )

    step_graph = qat_graph.make_step_graph(
        b,
        constants=constants,
        state=state,
        scalars=["lr", "grad_seed"],
        loss=loss_output,
        name="resident_train_step",
    )
    step_model = step_graph.model

    # rank0_to_rank1 only fixes a scalar *output* (the loss); a scalar
    # *input* hits the identical Pulsar2 calibration failure -- see this
    # module's own docstring.
    for inp in step_model.graph.input:
        if inp.name in ("lr", "grad_seed"):
            del inp.type.tensor_type.shape.dim[:]
            inp.type.tensor_type.shape.dim.add().dim_value = 1

    legalize.legalize(step_model, _POST_BACKWARD_RULES)
    onnx.checker.check_model(step_model)

    from onnxsim import simplify as _simplify

    step_model, ok = _simplify(
        step_model,
        skipped_optimizers=[
            "fuse_matmul_add_bias_into_gemm",
            "fuse_transpose_into_gemm",
        ],
    )
    if not ok:
        raise RuntimeError("post-legalize simplify() failed its own correctness check")

    return step_model, step_graph.state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "forward_onnx", help="forward model, already simplified/BN-folded"
    )
    parser.add_argument("output_onnx")
    parser.add_argument(
        "--logits", default="logits", help="forward model's output tensor"
    )
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument(
        "--param",
        action="append",
        required=True,
        dest="params",
        help="a trainable initializer's name; repeat for each",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="override the forward model's declared batch size (see set_batch)",
    )
    args = parser.parse_args(argv)

    forward = onnx.load(args.forward_onnx)
    if args.batch is not None:
        forward = set_batch(forward, args.batch)
    with_loss = add_mse_loss(forward, args.logits, args.num_classes)
    step_model, state = build_resident_step(with_loss, args.params)

    print(f"step graph: {len(step_model.graph.node)} nodes")
    for p, out_name in state.items():
        print(f"  state: {p} -> {out_name}")
    onnx.save(step_model, args.output_onnx)
    print("wrote", args.output_onnx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
