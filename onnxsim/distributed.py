"""Real, multi-process data-parallel training over onnxsim's own step-graph
machinery -- see ``docs/distributed-data-parallel-plan.md`` for the design
rationale and why this stays deliberately scoped to data parallelism rather
than tensor/pipeline parallelism.

**The split this module makes that no other onnxsim training path needs.**
:func:`onnxsim.qat_graph.make_step_graph` fuses forward, backward, and an
optimizer update into one graph for every existing caller
(:mod:`onnxsim.lora`, :mod:`onnxsim.qat`), because every existing caller
trains alone. Data-parallel training needs a gradient *average* to happen
between backward and the optimizer step, so this module splits those two
into separate graphs instead:

- :func:`build_gradient_step_graph` -- forward, an MSE loss, and
  :func:`onnxsim.graph_grad.build_backward` restricted to the parameters
  being trained. No optimizer, no state threading: a worker's weights are
  fed fresh from the parent process every step (see
  :func:`train_data_parallel`), never carried between calls.
- :func:`build_apply_step_graph` -- exactly the same
  :func:`onnxsim.qat_graph.adam_update`-as-a-step-graph idiom every other
  onnxsim training path already uses, just fed an already-averaged gradient
  instead of one it computed itself.

**The orchestration.** :func:`train_data_parallel` spawns one real OS
process per data shard (``multiprocessing.Process``, not a thread or an
in-process loop), each running its own ``onnxruntime`` session on the
gradient-step graph against its own private data; the parent process
collects every worker's gradients over a ``multiprocessing.Pipe``, averages
them, and runs the (single, shared) apply-step to produce the next round's
weights, which it broadcasts back down the same pipes. This is genuine
inter-process parallelism -- unlike :mod:`onnxsim.federated`'s sequential
simulation of federated clients, or ``scripts/distributed_tp_sketch.py``'s
single-process numpy stand-in for NCCL -- it just runs on one machine's CPU
cores rather than a GPU cluster or a real network, since that is what is
available to develop and test this against. See the plan doc's "Left for
later" section for what a real multi-node transport would need to change
(the gradient-averaging math here would not).
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnx.inliner

from onnxsim import backend, graph_grad, qat_graph

_OPSET = 17
_IR_VERSION = 8


def build_gradient_step_graph(
    nodes: Sequence[onnx.NodeProto],
    shapes: Dict[str, Sequence[Union[int, str]]],
    initializers: Sequence[onnx.TensorProto],
    input_name: str,
    input_shape: Sequence[int],
    target_name: str,
    target_shape: Sequence[int],
    output_name: str,
    param_names: Sequence[str],
) -> Tuple[onnx.ModelProto, Dict[str, str], str]:
    """A plain (non-step, non-stateful) ONNX graph computing an MSE loss's
    gradient with respect to every name in ``param_names``.

    Unlike :func:`onnx.qat_graph.make_step_graph`, this declares every
    parameter as a plain graph *input* (fed fresh each call) with a plain
    gradient tensor as the matching *output* -- there is nothing to thread
    between calls, since :func:`train_data_parallel` never reuses one
    worker's own copy of the weights across steps (see this module's
    docstring). Using ``make_step_graph``'s state-threading machinery for a
    graph that has no state to thread would be forcing a fit rather than
    reusing one; building the plain graph directly, the way
    ``onnxsim.qat._block_shapes``'s own probe models do, is the more honest
    match.

    :param nodes: the forward pass's own nodes (in graph order).
    :param shapes: static shapes for every tensor ``nodes`` touches, as
            :func:`onnxsim.graph_grad.build_backward` requires.
    :param initializers: every initializer ``nodes`` reads (frozen weights
            not in ``param_names``, and any of ``param_names`` themselves --
            included here only so :func:`onnxsim.backend.run_model`-style
            callers can inspect a self-contained model; the *training* path
            in :func:`train_data_parallel` always overrides these via feeds,
            since it declares each of ``param_names`` as an input below too).
    :param input_name: the model's own input activation.
    :param target_name: the regression target this step's MSE loss is
            computed against.
    :param output_name: the forward pass's own final output.
    :param param_names: which initializers to differentiate with respect to
            -- everything else in ``nodes`` is treated as frozen, exactly as
            :func:`onnxsim.graph_grad.build_backward`'s own ``targets``
            argument works.
    :returns: ``(model, {param name: gradient tensor name}, loss_name)``.
    """
    b = qat_graph.GraphBuilder("dp__")
    b.initializer.extend(initializers)
    b.nodes.extend(nodes)

    diff = b.sub(output_name, target_name)
    n_elems = int(np.prod(list(target_shape)))
    dl_dy = b.mul(diff, b.const(2.0 / n_elems))
    loss_name = b.mean_square(diff)

    grads = graph_grad.build_backward(
        b, list(nodes), shapes, {output_name: dl_dy}, list(param_names)
    )

    inputs = [
        onnx.helper.make_tensor_value_info(
            input_name, onnx.TensorProto.FLOAT, list(input_shape)
        ),
        onnx.helper.make_tensor_value_info(
            target_name, onnx.TensorProto.FLOAT, list(target_shape)
        ),
    ]
    inputs += [
        onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, list(shapes[name])
        )
        for name in param_names
    ]

    grad_names = {name: grads[name] for name in param_names}
    outputs = [
        onnx.helper.make_tensor_value_info(
            grad_names[name], onnx.TensorProto.FLOAT, list(shapes[name])
        )
        for name in param_names
    ]
    outputs.append(
        onnx.helper.make_tensor_value_info(loss_name, onnx.TensorProto.FLOAT, [])
    )

    graph = onnx.helper.make_graph(
        b.nodes,
        "onnxsim_dp_gradient_step",
        inputs,
        outputs,
        initializer=[t for t in b.initializer if t.name not in param_names],
    )
    opset_imports = [onnx.helper.make_opsetid("", _OPSET)]
    if b.functions:
        domains = sorted({fn.domain for fn in b.functions})
        opset_imports += [onnx.helper.make_opsetid(d, 1) for d in domains]
    model = onnx.helper.make_model(
        graph, opset_imports=opset_imports, functions=list(b.functions)
    )
    model.ir_version = _IR_VERSION
    if b.functions:
        model = onnx.inliner.inline_local_functions(model)
    onnx.checker.check_model(model)
    return model, grad_names, loss_name


def build_apply_step_graph(
    shapes: Dict[str, Sequence[int]], param_names: Sequence[str]
) -> qat_graph.StepGraph:
    """The shared optimizer step every worker's already-averaged gradient
    feeds into -- one :func:`onnxsim.qat_graph.adam_update` call per
    parameter, wired through :func:`onnxsim.qat_graph.make_step_graph`'s
    existing state mechanism exactly like every other onnxsim training path
    already does. The only thing distinguishing this from, say,
    :mod:`onnxsim.lora`'s own Adam wiring is that the gradient
    (``f"grad__{name}"``) is a *per-step* input here rather than something
    this same graph computed via backward -- :func:`train_data_parallel`
    supplies it fresh each call, already averaged across every worker.
    """
    b = qat_graph.GraphBuilder("dpopt__")
    state: Dict[str, Tuple[Sequence[int], str]] = {}
    per_step: Dict[str, Tuple[Sequence[int], int]] = {}
    for name in param_names:
        shape = list(shapes[name])
        grad_name = f"grad__{name}"
        per_step[grad_name] = (shape, onnx.TensorProto.FLOAT)
        m_name, v_name = f"m__{name}", f"v__{name}"
        param_next, m_next, v_next = qat_graph.adam_update(
            b, name, grad_name, m_name, v_name, "lr", "m_correction", "v_correction"
        )
        state[name] = (shape, param_next)
        state[m_name] = (shape, m_next)
        state[v_name] = (shape, v_next)

    return qat_graph.make_step_graph(
        b,
        constants={},
        state=state,
        scalars=["lr", "m_correction", "v_correction"],
        name="onnxsim_dp_apply_step",
        per_step=per_step,
    )


@dataclass
class WorkerShard:
    """One worker's local data for the whole run -- never sent anywhere,
    never seen by the parent process or any other worker. Only that
    worker's *gradients*, computed locally, ever leave it (see
    :func:`train_data_parallel`)."""

    inputs: np.ndarray
    targets: np.ndarray

    def sample(
        self, rng: np.random.Generator, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, self.inputs.shape[0], size=batch_size)
        return self.inputs[idx], self.targets[idx]


def _worker_main(
    worker_id: int,
    grad_model_bytes: bytes,
    input_name: str,
    target_name: str,
    grad_output_names: List[str],
    loss_name: str,
    conn,
) -> None:
    """One data-parallel worker's whole life: build a local ORT session
    once, then repeatedly receive ``(params, batch_input, batch_target)``
    from the parent and send back ``(worker_id, gradients, loss)`` -- until
    a ``None`` message tells it to exit. Runs in its own OS process (see
    :func:`train_data_parallel`), so this function's own imports and the ORT
    session it creates are never shared with the parent or any sibling
    worker.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(grad_model_bytes, providers=["CPUExecutionProvider"])
    output_names = grad_output_names + [loss_name]
    try:
        while True:
            message = conn.recv()
            if message is None:
                return
            params, batch_input, batch_target = message
            feeds = {input_name: batch_input, target_name: batch_target}
            feeds.update(params)
            results = session.run(output_names, feeds)
            grads = results[:-1]
            loss = float(results[-1])
            conn.send((worker_id, grads, loss))
    finally:
        conn.close()


def train_data_parallel(
    grad_model: onnx.ModelProto,
    grad_names: Dict[str, str],
    loss_name: str,
    input_name: str,
    target_name: str,
    param_names: Sequence[str],
    shapes: Dict[str, Sequence[int]],
    initial_params: Dict[str, np.ndarray],
    shards: Sequence[WorkerShard],
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], List[float]]:
    """Trains ``param_names`` for ``num_steps`` synchronous data-parallel
    steps across ``len(shards)`` real OS processes -- one per shard, each
    running its own copy of ``grad_model`` against its own private data (see
    :class:`WorkerShard`) -- and returns ``(final params, per-step mean
    loss across workers)``.

    Every step: the current parameters are broadcast to every worker; each
    worker samples its own random batch (:meth:`WorkerShard.sample`) and
    computes local gradients; the parent averages the same-named gradient
    across every worker (equal weight per worker, matching equal-sized
    shards -- unequal shards would need
    :func:`onnxsim.federated.fedavg`-style weighting instead, not
    implemented here) and runs one shared :func:`build_apply_step_graph`
    Adam step to produce the next parameters.

    :param grad_model: the model :func:`build_gradient_step_graph` returned.
    :param grad_names: the ``{param name: gradient tensor name}`` mapping
            :func:`build_gradient_step_graph` also returned.
    :param loss_name: that same call's ``loss_name``.
    :param shards: one per worker; ``len(shards)`` is the worker count.
    :param batch_size: each worker's own local batch size per step -- every
            worker samples this many rows independently, so the effective
            combined batch size per step is ``batch_size * len(shards)``.
    :param seed: seeds the parent's own batch-sampling RNG; each worker's
            samples are therefore reproducible run to run (single-process
            parent, one RNG), but are not required to be identical to what a
            single-process reference run would sample as its own combined
            batch -- see ``tests/test_distributed.py`` for how the two are
            actually compared.
    :raises ValueError: if ``shards`` is empty.
    """
    if not shards:
        raise ValueError("train_data_parallel needs at least one worker shard")

    apply_step = build_apply_step_graph(shapes, param_names)
    apply_runner = backend.Runner(
        apply_step.model, output_names=list(apply_step.state.values())
    )

    model_bytes = grad_model.SerializeToString()
    ordered_grad_names = [grad_names[name] for name in param_names]

    parent_conns = []
    processes = []
    for worker_id, _shard in enumerate(shards):
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(
            target=_worker_main,
            args=(
                worker_id,
                model_bytes,
                input_name,
                target_name,
                ordered_grad_names,
                loss_name,
                child_conn,
            ),
        )
        process.start()
        child_conn.close()
        parent_conns.append(parent_conn)
        processes.append(process)

    params = {
        name: np.asarray(initial_params[name], dtype=np.float32) for name in param_names
    }
    opt_m = {name: np.zeros_like(params[name]) for name in param_names}
    opt_v = {name: np.zeros_like(params[name]) for name in param_names}

    rng = np.random.default_rng(seed)
    losses: List[float] = []
    try:
        for t in range(num_steps):
            for conn, shard in zip(parent_conns, shards):
                batch_input, batch_target = shard.sample(rng, batch_size)
                conn.send((params, batch_input, batch_target))

            grads_by_worker: List[List[np.ndarray]] = [None] * len(shards)  # type: ignore[list-item]
            step_losses = []
            for conn in parent_conns:
                worker_id, grads, loss = conn.recv()
                grads_by_worker[worker_id] = grads
                step_losses.append(loss)
            losses.append(float(np.mean(step_losses)))

            feeds: Dict[str, np.ndarray] = dict(params)
            for i, name in enumerate(param_names):
                averaged = np.mean([grads[i] for grads in grads_by_worker], axis=0)
                feeds[f"grad__{name}"] = averaged.astype(np.float32)
                feeds[f"m__{name}"] = opt_m[name]
                feeds[f"v__{name}"] = opt_v[name]
            feeds["lr"] = np.array(learning_rate, dtype=np.float32)
            corrections = qat_graph.adam_bias_corrections(t)
            feeds["m_correction"] = np.array(
                corrections["m_correction"], dtype=np.float32
            )
            feeds["v_correction"] = np.array(
                corrections["v_correction"], dtype=np.float32
            )

            out = apply_runner(feeds)
            for name in param_names:
                params[name] = out[apply_step.state[name]]
                opt_m[name] = out[apply_step.state[f"m__{name}"]]
                opt_v[name] = out[apply_step.state[f"v__{name}"]]
    finally:
        for conn in parent_conns:
            conn.send(None)
            conn.close()
        for process in processes:
            process.join()

    return params, losses
