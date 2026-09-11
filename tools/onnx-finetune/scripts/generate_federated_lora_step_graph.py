#!/usr/bin/env python3
"""Build a self-contained, data-free LoRA training-step graph for a federated
round's *client* half -- the browser-side counterpart to
``onnxsim.federated``'s server-side FedAvg loop (see ``onnxsim/federated.py``
and ``tests/test_federated.py``), runnable via the official
``onnxruntime-web`` package with no custom WASM build, exactly the same
relationship ``generate_distillation_step_graph.py`` has to
``../wasm/distill_step_graph/step_graph_runner.mjs``.

**Why a step graph is the right thing to ship to a client at all.** A step
graph (:mod:`onnxsim.qat_graph`) bakes forward, loss, backward and an Adam
update into one ordinary ONNX graph -- but critically, the calibration
activations and training targets are *graph inputs* (bound once and reused
across steps for residency, see ``onnxsim.qat_graph``'s own module
docstring), never baked into the model as initializers. So the ``.onnx`` file
this script writes contains onnxsim's own architecture and the base model's
frozen weights, but *no client's training data whatsoever* -- a client feeds
its own private activations/targets as ordinary ``session.run()`` inputs,
exactly as it would with any other inference graph. That is what makes
shipping this file to an untrusted browser client safe in the first place.

**What crosses the wire, in each direction.** Down to the client: this
script's four outputs -- the step graph itself, the base (LoRA-injected)
model the client trains against, a text manifest describing every per-step
tensor name/shape, and the *current global round's* adapter values as a flat
initial-state file. Back up from the client: only the trained ``A``/``B``
adapter values (:func:`export_trained_state`'s counterpart on the JS side),
never the client's data and never the base model's own weights, which the
client never modifies (see ``onnxsim.lora``'s own "the base weight is never
touched" invariant). ``../scripts/federated_lora_aggregate.py`` is the
server-side FedAvg step that turns several clients' exported states into the
next round's base model and initial state.

**The one real restriction this generator adds over ``onnxsim.lora.train_lora``
itself.** ``train_lora`` differentiates a block whose shapes are captured
from real calibration data at call time, so nothing about it needs a fixed
batch size in advance. A step graph shipped to a browser ahead of any
client's data cannot do that -- there is no calibration run to capture shapes
from -- so this script needs the block's input/output shape declared with a
fixed batch size up front (``--batch-size``), and refuses a block whose only
external input is not exactly its own ``block_input_name`` (a residual or
side-channel input reaching the block from elsewhere in the graph has no
value to probe its shape from either). This is a real, narrower restriction
than ``train_lora``'s own, not an oversight -- see
:func:`build_federated_lora_step_graph`'s docstring.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import lora, qat, qat_graph


def _resolve_static_shape(
    model: onnx.ModelProto, tensor_name: str, batch_size: int
) -> List[int]:
    """``tensor_name``'s shape from ``model``'s own inputs/outputs/
    ``value_info``, with its leading (batch) dimension replaced by
    ``batch_size`` regardless of what that dimension already was -- static or
    a ``dim_param`` -- since a federated round's client-side batch is picked
    freely per :class:`onnxsim.federated.FederatedClient`, not fixed by the
    base model.

    :raises ValueError: if ``tensor_name`` is not found, or if any dimension
            other than the leading one is not a concrete integer -- this
            generator (unlike ``generate_distillation_step_graph.py``) does
            not support a dynamic non-batch dimension, since
            :mod:`onnxsim.lora`'s own step graphs (built via
            ``onnxsim.qat_graph.mean_square``'s ``ReduceMean``) are not
            dynamic-batch-safe to begin with -- see this module's own
            docstring.
    """
    for value_info in (
        list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    ):
        if value_info.name != tensor_name:
            continue
        shape: List[int] = []
        for axis, dim in enumerate(value_info.type.tensor_type.shape.dim):
            if axis == 0:
                shape.append(batch_size)
            elif dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                raise ValueError(
                    f"{tensor_name!r} has a non-batch dynamic dimension "
                    f"({dim.dim_param!r} at axis {axis}); "
                    "generate_federated_lora_step_graph needs every dimension "
                    "but the leading (batch) one to be static"
                )
        return shape
    raise ValueError(
        f"{tensor_name!r} not found among {model.graph.name}'s inputs, "
        "outputs, or value_info"
    )


def build_federated_lora_step_graph(
    injected: onnx.ModelProto,
    adapter: lora.LoraAdapter,
    block_input_name: str,
    block_output_name: str,
    batch_size: int,
) -> Tuple[qat_graph.StepGraph, Dict[str, np.ndarray]]:
    """The client-shippable half of :func:`onnxsim.lora.train_lora` --
    everything up to (not including) actually running a step, since running
    it is the client's job, on the client's own data.

    Reuses :mod:`onnxsim.qat`'s own block-slicing machinery
    (``_slice_block``/``_refuse_unsupported``/``_block_shapes``) and
    :func:`onnxsim.lora._build_lora_step_graph` directly -- the same pieces
    ``train_lora`` itself calls internally -- with dummy, zero-valued arrays
    standing in for calibration data purely to supply :func:`_block_shapes`
    the concrete shapes ``onnxsim.graph_grad.build_backward`` needs; their
    *values* are never read.

    :raises ValueError: if the block reads any external input other than
            ``block_input_name`` itself (see this module's own docstring),
            or if :func:`_resolve_static_shape` refuses a non-batch dynamic
            dimension.
    :raises onnxsim.graph_grad.UnsupportedOpError: if the block contains an
            op :mod:`onnxsim.graph_grad` has no gradient rule for.
    """
    nodes, externals = qat._slice_block(injected.graph, block_input_name, block_output_name)
    qat._refuse_unsupported(nodes)
    if externals != [block_input_name]:
        raise ValueError(
            "generate_federated_lora_step_graph only supports a block whose "
            f"sole external input is its own block_input_name; the block "
            f"between {block_input_name!r} and {block_output_name!r} also "
            f"reads {[e for e in externals if e != block_input_name]}, which "
            "has no calibration run to capture a shape from ahead of time"
        )

    input_shape = _resolve_static_shape(injected, block_input_name, batch_size)
    output_shape = _resolve_static_shape(injected, block_output_name, batch_size)
    dummy_input = np.zeros(input_shape, dtype=np.float32)
    dummy_output = np.zeros(output_shape, dtype=np.float32)

    shapes = qat._block_shapes(injected, nodes, {block_input_name: dummy_input}, block_output_name, dummy_output)

    used = {name for node in nodes for name in node.input if name}
    param_names = set(adapter.parameter_names())
    initializer_map = {t.name: t for t in injected.graph.initializer}
    block_initializers = [
        t
        for t in injected.graph.initializer
        if t.name in used and t.name not in param_names
    ]

    step = lora._build_lora_step_graph(
        adapter,
        nodes,
        shapes,
        block_initializers,
        {block_input_name: dummy_input},
        block_output_name,
        output_shape,
        batch=None,
    )
    onnx.checker.check_model(step.model)

    initial_state: Dict[str, np.ndarray] = {}
    for name in adapter.parameter_names():
        value = onnx.numpy_helper.to_array(initializer_map[name]).astype(np.float32)
        initial_state[name] = value
        initial_state[f"{lora._PREFIX}m_{name}"] = np.zeros_like(value)
        initial_state[f"{lora._PREFIX}v_{name}"] = np.zeros_like(value)

    return step, initial_state


def write_manifest_and_initial_state(
    step: qat_graph.StepGraph,
    initial_state: Dict[str, np.ndarray],
    adapter: lora.LoraAdapter,
    block_input_name: str,
    input_shape: Sequence[int],
    block_output_name: str,
    output_shape: Sequence[int],
    manifest_path: str,
    initial_state_path: str,
) -> None:
    """Writes the manifest+binary pair
    ``tools/onnx-finetune/wasm/federated_lora/step_graph_runner.mjs`` and
    :mod:`federated_lora_aggregate` both read, in the same flat line-oriented
    text format ``generate_distillation_step_graph.py`` uses for the same
    reason (no JSON parser vendored on the C++/wasm side).

    Only ``weight`` lines (the adapter's own ``A``/``B`` initializers) are
    written to ``initial_state_path`` -- every ``__m``/``__v`` Adam moment
    starts at zero every round (a federated client always starts local
    training with a fresh optimizer state, see ``onnxsim.federated``'s own
    module docstring), so a reader can produce those itself without them
    being spelled out on disk.
    """
    with open(manifest_path, "w") as f:
        f.write(f"input_name {block_input_name}\n")
        f.write(f"input_shape {' '.join(str(d) for d in input_shape)}\n")
        f.write(f"teacher_name {lora._PREFIX}teacher\n")
        f.write(f"teacher_shape {' '.join(str(d) for d in output_shape)}\n")
        f.write(f"lr_name {lora._PREFIX}lr\n")
        f.write(f"loss_name {step.loss_name}\n")
        for name, out_name in step.state.items():
            shape = list(initial_state[name].shape)
            f.write(f"state {name} {out_name} {' '.join(str(d) for d in shape)}\n")
        for name in adapter.parameter_names():
            shape = list(initial_state[name].shape)
            f.write(f"weight {name} {' '.join(str(d) for d in shape)}\n")

    with open(initial_state_path, "wb") as f:
        for name in adapter.parameter_names():
            initial_state[name].astype(np.float32).tofile(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="path to the plain (not yet LoRA-injected) .onnx model")
    p.add_argument("-o", "--output", required=True, help="path to write the step graph to")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument(
        "--block-input-name", default=None,
        help="defaults to the model's own single input",
    )
    p.add_argument(
        "--block-output-name", default=None,
        help="defaults to the model's own single output",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = onnx.load(args.model)
    injected, adapter = lora.inject_lora(model, rank=args.rank, alpha=args.alpha, seed=args.seed)
    if not adapter.targets:
        raise ValueError(
            f"{args.model} has no eligible MatMul/Gemm/Conv weight to inject "
            "a LoRA adapter into -- see onnxsim.lora.inject_lora's own "
            "eligibility rules"
        )

    block_input_name = args.block_input_name or injected.graph.input[0].name
    block_output_name = args.block_output_name or injected.graph.output[0].name

    step, initial_state = build_federated_lora_step_graph(
        injected, adapter, block_input_name, block_output_name, args.batch_size
    )
    onnx.save(step.model, args.output)

    base_model_path = args.output + ".base_model.onnx"
    onnx.save(injected, base_model_path)

    adapter_names_path = args.output + ".adapter_names.json"
    with open(adapter_names_path, "w") as f:
        json.dump(adapter.parameter_names(), f)

    manifest_path = args.output + ".manifest.txt"
    initial_state_path = args.output + ".initial_state.bin"
    write_manifest_and_initial_state(
        step,
        initial_state,
        adapter,
        block_input_name,
        _resolve_static_shape(injected, block_input_name, args.batch_size),
        block_output_name,
        _resolve_static_shape(injected, block_output_name, args.batch_size),
        manifest_path,
        initial_state_path,
    )
    print(
        f"wrote {args.output} ({len(step.model.graph.node)} nodes, "
        f"batch size {args.batch_size}), {base_model_path}, {adapter_names_path}, "
        f"{manifest_path}, {initial_state_path}"
    )


if __name__ == "__main__":
    main()
