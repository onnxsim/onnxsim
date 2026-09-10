#!/usr/bin/env python3
"""Build a self-contained knowledge-distillation training-step graph, using
onnxsim's own reverse-mode differentiator (``onnxsim.graph_grad``) instead of
``onnxruntime.training``.

``generate_artifacts.py --loss distillation`` (``distillation_loss.py``) gets
its gradient from ``onnxruntime.training.artifacts``/``onnxblock`` -- real,
but only available via a training-enabled onnxruntime build (the
``onnxruntime-training`` PyPI wheel, or a from-source ``--enable_training_apis``
build for the C++ CLI/WASM binding, see ../README.md and wasm/README.md).
``onnxsim.lora``/``onnxsim.qat`` already sidestep that dependency entirely for
LoRA and QAT by hand-differentiating with ``onnxsim.graph_grad`` and running
the result on a *plain* onnxruntime (no training APIs at all). This script
does the same thing for distillation: it emits ONE ordinary ONNX model --
forward, KD loss, and backward, plus an Adam step, all baked into a single
graph via ``onnxsim.qat_graph.make_step_graph`` -- runnable by repeatedly
calling ``session.Run()``/``InferenceSession.run()`` on a plain (non-training)
session and feeding each step's outputs back in as the next step's weight
inputs.

**The loss**, matching ``distillation_loss.py``'s formula (Hinton et al.):
``alpha`` * soft-target cross-entropy (temperature-scaled) + ``(1 - alpha)``
* hard-label cross-entropy. Built from ``onnxsim.qat_graph.GraphBuilder``
primitives rather than the fused ``LogSoftmax``/``SoftmaxCrossEntropyLoss``
ops onnxblock uses, since neither has a VJP rule in ``graph_grad`` (and this
is deliberately not the place to add one -- see graph_grad.py's own
"arithmetic primitives, not fused ops" stance):

- ``log_softmax(x)`` is ``Log(Softmax(x))`` -- two rules ``graph_grad``
  already has. ``Softmax``'s own max-subtraction happens inside the op for
  numerical stability, so this composition is not less stable than a fused
  LogSoftmax would be for the logit magnitudes a training loop produces.
- picking out each row's target-class log-probability (what
  ``SoftmaxCrossEntropyLoss`` does internally) is a ``Mul`` against a one-hot
  label matrix followed by a ``ReduceSum`` -- both ops ``graph_grad`` already
  differentiates, unlike ``Gather`` (whose rule only covers a single
  constant-axis table lookup, not "row i, column labels[i]" for a batch of
  different columns per row -- see ``graph_grad._grad_gather``'s own
  docstring). The one-hot matrix itself is built on the *host*, from the
  caller's integer labels, and handed in as an ordinary float32 input
  (:func:`labels_to_onehot`) -- turning an integer index into a one-hot row is
  data preparation, not a gradient, and doing it off-graph keeps every node
  ``build_backward`` is asked to differentiate inside ``graph_grad.SUPPORTED_OPS``
  (notably keeping ``Cast``/``Greater``/``Less``, which have no VJP rule, out
  of the differentiated slice entirely).

**What is different from ``generate_artifacts.py``'s output.** That produces
four separate artifact files (``training_model.onnx``, ``eval_model.onnx``,
``optimizer_model.onnx``, ``checkpoint``) consumed by
``onnxruntime.training.api``'s stateful ``Module``/``Optimizer``/
``CheckpointState`` objects. This produces one :class:`onnxsim.qat_graph.StepGraph`
(a single ``ModelProto`` plus a ``{input name: output name}`` state map) meant
for :func:`onnxsim.qat_graph.run_step_graph` or a hand-rolled loop -- there is
no separate eval-only graph and no persisted optimizer-state file format; a
caller that wants to pause/resume training saves the state dict's numpy
arrays itself. There is also no ``additional_output_names`` facility here:
only the combined loss is exposed as a step-graph output, not the soft/hard
breakdown ``distillation_loss.py`` exposes.

**The one real limitation this path has that the onnxruntime.training one
does not: a fixed batch size.** ``graph_grad.build_backward`` needs the
static shape of every tensor the slice touches (undoing a broadcast and
undoing a reduction are both shape arithmetic -- see graph_grad.py's own
module docstring), so the student model's batch dimension is concretized to
``--batch-size`` before shape inference ever runs, and the emitted step graph
only accepts batches of exactly that size. This is the same static-shapes
trade a QAT/LoRA step graph already makes, in exchange for a graph that needs
no training-capable runtime to execute at all.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.shape_inference

from onnxsim import graph_grad, qat_graph


def labels_to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """The host-side half of the "no Cast/Greater/Less in the differentiated
    slice" design this module's docstring explains: turn integer class
    indices into the float32 one-hot matrix the step graph's
    ``labels_onehot`` input expects."""
    labels = np.asarray(labels, dtype=np.int64)
    return np.eye(num_classes, dtype=np.float32)[labels]


def _shapes_of(model: onnx.ModelProto) -> Dict[str, List[int]]:
    """Every tensor's static shape, the way ``graph_grad.build_backward``
    needs it -- inputs, outputs, intermediate ``value_info``, and
    initializers alike."""
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes: Dict[str, List[int]] = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    ):
        shapes[value.name] = [d.dim_value for d in value.type.tensor_type.shape.dim]
    for initializer in inferred.graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    return shapes


def _concretize_batch(model: onnx.ModelProto, batch_size: int) -> onnx.ModelProto:
    """``model`` with its first input's leading (batch) dimension fixed to
    ``batch_size``, and every downstream shape cleared so shape inference
    recomputes them from that concrete value rather than leaving the old
    symbolic/dynamic dimension behind.

    ``build_backward`` cannot work with a symbolic batch axis at all (see this
    module's own docstring), so this is not optional the way it might be for
    an ordinary inference model.
    """
    fixed = onnx.ModelProto()
    fixed.CopyFrom(model)
    original_input = fixed.graph.input[0]
    tail = [d.dim_value for d in original_input.type.tensor_type.shape.dim[1:]]
    new_input = onnx.helper.make_tensor_value_info(
        original_input.name,
        original_input.type.tensor_type.elem_type,
        [batch_size] + tail,
    )
    fixed.graph.input[0].CopyFrom(new_input)
    for value in fixed.graph.output:
        value.type.tensor_type.ClearField("shape")
    del fixed.graph.value_info[:]
    return fixed


def _int64_const(b: qat_graph.GraphBuilder, values, hint: str = "i") -> str:
    """An int64 initializer -- the ``axes``/``shape`` inputs ``ReduceSum`` and
    ``Reshape`` take as tensors from opset 13 on. Mirrors
    ``graph_grad._Backward.int64_const`` (not reusable here directly: that
    one lives on the private context ``build_backward`` constructs for its
    own rules, not on ``GraphBuilder`` itself)."""
    array = np.asarray(list(values), dtype=np.int64)
    name = b.name(hint)
    b.initializer.append(onnx.numpy_helper.from_array(array, name))
    return name


class _ForwardLossGrads:
    """Everything :func:`build_distillation_step_graph` needs before it wires
    in Adam -- also exactly what a caller that wants to check the raw
    gradient (rather than an Adam-updated weight, from which the gradient's
    own *magnitude* is not cleanly recoverable: Adam's step-1 update is
    ``lr * sign(gradient)`` up to the ``eps`` guard, see ``adam_update``)
    needs, which is why this is its own function rather than inlined into
    :func:`build_distillation_step_graph`."""

    def __init__(
        self, b, grads, combined, trainable, input_name, input_shape,
        logits_shape, rows, num_classes, forward_and_loss_nodes,
    ):
        self.b = b
        self.grads = grads
        self.combined = combined
        self.trainable = trainable
        self.input_name = input_name
        self.input_shape = input_shape
        self.logits_shape = logits_shape
        self.rows = rows
        self.num_classes = num_classes
        # `b.nodes` as of just before `build_backward` appended anything --
        # forward and loss only, no backward nodes. A finite-difference
        # reference for the loss must be built from exactly this list, not
        # `b.nodes` in full: the reference evaluator executes every node in a
        # graph unconditionally (no dead-code elimination), so a backward
        # node's `Cast(..., to=FLOAT)` (graph_grad's mask helpers hardcode
        # float32 regardless of the surrounding graph's own dtype) would
        # otherwise choke a float64 finite-difference reference with a
        # dtype-mismatched Mul it never needed to run at all.
        self.forward_and_loss_nodes = forward_and_loss_nodes
        # The two fixed per-step tensor input names every caller (the Adam
        # wrapper below, and a test building its own gradient-output model on
        # top of `b`) needs to know to feed a batch in.
        self.teacher_logits_name = "teacher_logits"
        self.labels_onehot_name = "labels_onehot"


def _build_forward_loss_and_grads(
    student: onnx.ModelProto,
    batch_size: int,
    temperature: float,
    alpha: float,
) -> _ForwardLossGrads:
    """The differentiable half: the student's own forward nodes, the KD loss
    on top of them, and ``graph_grad.build_backward``'s gradient for every
    trainable weight -- everything in ``b`` up to, but not including, the
    Adam update. Appends nothing an Adam step or a plain gradient-output
    model couldn't equally build on top of.
    """
    if len(student.graph.output) != 1:
        raise ValueError(
            f"expected exactly one output (the logits), got "
            f"{[o.name for o in student.graph.output]}"
        )
    fixed = _concretize_batch(student, batch_size)
    input_name = fixed.graph.input[0].name
    logits_name = fixed.graph.output[0].name

    trainable = {
        init.name: onnx.numpy_helper.to_array(init).copy()
        for init in fixed.graph.initializer
    }
    if not trainable:
        raise ValueError("student model has no initializers to train")

    # A probe model to learn every tensor's shape from, with each trainable
    # weight declared as a plain input (a concrete shape, standing in for the
    # step-graph *state* input it becomes below) instead of the initializer it
    # is in `fixed` -- onnx.shape_inference needs every name resolved one way
    # or the other, and a step graph's own weights are graph inputs, not
    # initializers (see the note beside `b.nodes.extend` below for why).
    probe_inputs = [fixed.graph.input[0]] + [
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, list(value.shape))
        for name, value in trainable.items()
    ]
    probe_graph = onnx.helper.make_graph(
        list(fixed.graph.node), "probe", probe_inputs, list(fixed.graph.output)
    )
    probe_model = onnx.helper.make_model(probe_graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    probe_model.ir_version = 8
    forward_shapes = _shapes_of(probe_model)

    logits_shape = forward_shapes[logits_name]
    if len(logits_shape) < 2:
        raise ValueError(
            f"distillation needs a >=2D logits tensor (batch, ..., classes); "
            f"{logits_name!r} has shape {logits_shape}"
        )
    num_classes = logits_shape[-1]
    rows = 1
    for d in logits_shape[:-1]:
        rows *= d

    b = qat_graph.GraphBuilder()
    b.nodes.extend(fixed.graph.node)
    # Deliberately not `b.initializer.extend(fixed.graph.initializer)`: every
    # one of these becomes a step-graph *state* input below instead of a
    # baked-in constant, so the step graph can be driven from a different
    # weight value on every call -- the same "the block's own weight is an
    # ordinary graph input, not an initializer" move onnxsim.qat's block
    # training already makes.

    flat_logits = logits_name
    teacher_logits = "teacher_logits"
    labels_onehot = "labels_onehot"
    if len(logits_shape) != 2:
        shape_const = _int64_const(b, [rows, num_classes], "flatten_shape")
        flat_logits = b.op("Reshape", [logits_name, shape_const])

    t_const = b.const(float(temperature), "t")
    t_sq_const = b.const(float(temperature) * float(temperature), "t_sq")
    alpha_const = b.const(float(alpha), "alpha")
    one_minus_alpha_const = b.const(1.0 - float(alpha), "one_minus_alpha")

    # --- soft loss: -mean_i(sum_c(softmax(teacher/T) * log(softmax(student/T)))) * T^2
    student_scaled = b.div(flat_logits, t_const)
    teacher_scaled = b.div(teacher_logits, t_const)
    student_log_probs = b.op("Log", [b.op("Softmax", [student_scaled], axis=-1)])
    teacher_probs = b.op("Softmax", [teacher_scaled], axis=-1)
    per_token = b.mul(teacher_probs, student_log_probs)
    soft_axes = _int64_const(b, [-1], "axes")
    per_position = b.op("ReduceSum", [per_token, soft_axes], keepdims=1)
    soft_mean = b.op("ReduceMean", [per_position], keepdims=0)
    soft_loss = b.mul(b.op("Neg", [soft_mean]), t_sq_const)

    # --- hard loss: -mean_i(sum_c(onehot(y) * log(softmax(student))))
    log_probs = b.op("Log", [b.op("Softmax", [flat_logits], axis=-1)])
    hard_axes = _int64_const(b, [-1], "axes")
    selected = b.op("ReduceSum", [b.mul(labels_onehot, log_probs), hard_axes], keepdims=1)
    hard_loss = b.op("Neg", [b.op("ReduceMean", [selected], keepdims=0)])

    combined = b.add(b.mul(soft_loss, alpha_const), b.mul(hard_loss, one_minus_alpha_const))

    # Shapes for everything build_backward might ask about: the forward
    # network's own (from the probe above) plus every tensor the loss
    # construction just added, gotten the same way -- run real shape
    # inference over the combined graph rather than tracking each new
    # intermediate's shape by hand, which is exactly as error-prone as the
    # hand-derivation this module exists to avoid.
    loss_probe_inputs = probe_inputs + [
        onnx.helper.make_tensor_value_info(teacher_logits, onnx.TensorProto.FLOAT, logits_shape),
        onnx.helper.make_tensor_value_info(labels_onehot, onnx.TensorProto.FLOAT, [rows, num_classes]),
    ]
    loss_probe_graph = onnx.helper.make_graph(
        list(b.nodes),
        "loss_probe",
        loss_probe_inputs,
        [onnx.helper.make_tensor_value_info(combined, onnx.TensorProto.FLOAT, [])],
        initializer=list(b.initializer),
    )
    loss_probe_model = onnx.helper.make_model(
        loss_probe_graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    loss_probe_model.ir_version = 8
    shapes = _shapes_of(loss_probe_model)

    grad_seed = b.const(1.0, "loss_grad_seed")
    forward_and_loss_nodes = list(b.nodes)
    grads = graph_grad.build_backward(
        b, forward_and_loss_nodes, shapes, {combined: grad_seed}, list(trainable)
    )

    return _ForwardLossGrads(
        b, grads, combined, trainable, input_name, forward_shapes[input_name],
        logits_shape, rows, num_classes, forward_and_loss_nodes,
    )


def build_distillation_step_graph(
    student: onnx.ModelProto,
    batch_size: int,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> Tuple[qat_graph.StepGraph, Dict[str, np.ndarray], _ForwardLossGrads]:
    """Returns ``(step, initial_state, fwd)``. ``fwd`` is the same
    :class:`_ForwardLossGrads` :func:`_build_forward_loss_and_grads` returned
    internally -- exposed to the caller too so ``main()`` (and any other
    caller writing out the per-step tensor metadata a non-Python runtime
    needs, see :func:`write_manifest_and_initial_state`) does not have to
    recompute ``input_name``/``logits_shape``/``rows``/``num_classes`` a
    second time by re-deriving them from ``student``.

    ``step.model`` trains the whole of ``student`` against a frozen teacher's
    logits and hard labels; ``initial_state`` seeds every weight from
    ``student``'s own initializer values and every Adam moment at zero, ready
    to hand straight to :func:`onnxsim.qat_graph.run_step_graph`.

    ``step.model``'s per-step inputs are ``student``'s own input name (batch
    of ``batch_size`` rows), ``"teacher_logits"`` (the frozen teacher's output
    on the same batch, computed by the caller each step), and
    ``"labels_onehot"`` (:func:`labels_to_onehot` of the batch's integer
    labels) -- plus ``"lr"``/``"m_correction"``/``"v_correction"`` scalars,
    the same three every ``onnxsim.qat_graph`` Adam step graph takes (see
    :func:`onnxsim.qat_graph.adam_bias_corrections`).
    """
    fwd = _build_forward_loss_and_grads(student, batch_size, temperature, alpha)
    b, grads, combined, trainable = fwd.b, fwd.grads, fwd.combined, fwd.trainable

    state: Dict[str, Tuple[list, str]] = {}
    initial_state: Dict[str, np.ndarray] = {}
    for name, value in trainable.items():
        shape = list(value.shape)
        m_name, v_name = f"{name}__m", f"{name}__v"
        next_name, m_next, v_next = qat_graph.adam_update(
            b, name, grads[name], m_name, v_name, "lr", "m_correction", "v_correction"
        )
        state[name] = (shape, next_name)
        state[m_name] = (shape, m_next)
        state[v_name] = (shape, v_next)
        initial_state[name] = value
        initial_state[m_name] = np.zeros(shape, dtype=np.float32)
        initial_state[v_name] = np.zeros(shape, dtype=np.float32)

    per_step = {
        fwd.input_name: (fwd.input_shape, onnx.TensorProto.FLOAT),
        fwd.teacher_logits_name: (fwd.logits_shape, onnx.TensorProto.FLOAT),
        fwd.labels_onehot_name: ([fwd.rows, fwd.num_classes], onnx.TensorProto.FLOAT),
    }
    step = qat_graph.make_step_graph(
        b,
        constants={},
        state=state,
        scalars=["lr", "m_correction", "v_correction"],
        loss=combined,
        name="onnxsim_distillation_step",
        per_step=per_step,
    )
    onnx.checker.check_model(step.model)
    return step, initial_state, fwd


def write_manifest_and_initial_state(
    step: qat_graph.StepGraph,
    initial_state: Dict[str, np.ndarray],
    trainable_names: List[str],
    input_name: str,
    input_shape: List[int],
    teacher_logits_name: str,
    logits_shape: List[int],
    labels_onehot_name: str,
    rows: int,
    num_classes: int,
    manifest_path: str,
    initial_state_path: str,
) -> None:
    """Writes the two files a caller with no ONNX protobuf parser at hand
    (the native CLI, see ../src/main.cpp) needs to actually run ``step``:

    - ``manifest_path``: a flat, line-oriented text manifest (matching
      ``onnxsim/qat_parity_fixtures.txt``'s own reasoning for the same
      choice -- no JSON parser is vendored here either) naming every
      per-step tensor input, the loss output, and -- the one thing the raw
      ONNX graph itself cannot reveal -- the ``{state input name: state
      output name}`` mapping :class:`onnxsim.qat_graph.StepGraph` carries,
      since ``GraphBuilder``'s own autogenerated output names (``sub_130``,
      not something derivable from the input name) are exactly the point of
      threading state through a step graph rather than mutating it in place.
    - ``initial_state_path``: every trainable weight's starting value (from
      the student model's own initializers), concatenated float32,
      row-major, in the same order the manifest's ``weight`` lines list them
      -- everything else in ``initial_state`` (each weight's ``__m``/``__v``
      Adam moments) starts at zero, which the reader can produce itself
      without needing it spelled out here.
    """
    with open(manifest_path, "w") as f:
        f.write(f"input_name {input_name}\n")
        f.write(f"input_shape {' '.join(str(d) for d in input_shape)}\n")
        f.write(f"teacher_logits_name {teacher_logits_name}\n")
        f.write(f"teacher_logits_shape {' '.join(str(d) for d in logits_shape)}\n")
        f.write(f"labels_onehot_name {labels_onehot_name}\n")
        f.write(f"rows {rows}\n")
        f.write(f"num_classes {num_classes}\n")
        f.write(f"loss_name {step.loss_name}\n")
        for name, out_name in step.state.items():
            shape = list(initial_state[name].shape)
            f.write(f"state {name} {out_name} {' '.join(str(d) for d in shape)}\n")
        for name in trainable_names:
            shape = list(initial_state[name].shape)
            f.write(f"weight {name} {' '.join(str(d) for d in shape)}\n")

    with open(initial_state_path, "wb") as f:
        for name in trainable_names:
            initial_state[name].astype(np.float32).tofile(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("student", help="path to the student .onnx model")
    p.add_argument("-o", "--output", required=True, help="path to write the step graph to")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()

    student = onnx.load(args.student)
    step, initial_state, fwd = build_distillation_step_graph(
        student, args.batch_size, args.temperature, args.alpha
    )
    onnx.save(step.model, args.output)

    manifest_path = args.output + ".manifest.txt"
    initial_state_path = args.output + ".initial_state.bin"
    write_manifest_and_initial_state(
        step, initial_state, list(fwd.trainable),
        fwd.input_name, fwd.input_shape,
        fwd.teacher_logits_name, fwd.logits_shape, fwd.labels_onehot_name,
        fwd.rows, fwd.num_classes,
        manifest_path, initial_state_path,
    )
    print(
        f"wrote {args.output} ({len(step.model.graph.node)} nodes), "
        f"{manifest_path}, {initial_state_path}"
    )


if __name__ == "__main__":
    main()
