#!/usr/bin/env python3
"""The server-side half of one federated round -- FedAvg several clients'
trained LoRA states (each in the exact byte layout
``generate_federated_lora_step_graph.py``'s ``initial_state.bin`` uses, and
``step_graph_runner.mjs``'s ``exportTrainedState`` produces in the browser)
into the next round's global model.

This is a thin CLI wrapper around :mod:`onnxsim.federated` -- the same
:func:`onnxsim.federated.fedavg`/:func:`onnxsim.federated.apply_adapter_state`
already exercised by ``tests/test_federated.py`` in-process, here reading
each client's contribution from a file instead of a Python dict, since a real
federated round's clients are separate processes (browser tabs) that can only
communicate via files/HTTP bodies. Nothing about the aggregation math
differs; only how each client's state arrives does.

**Usage**::

    federated_lora_aggregate.py \\
        --manifest step.onnx.manifest.txt \\
        --base-model step.onnx.base_model.onnx \\
        --client alice_round0.bin --client bob_round0.bin \\
        -o round1.base_model.onnx

Pass ``--weight`` once per ``--client`` (same order) to weight the average by
each client's local example count, i.e. FedAvg proper
(:meth:`onnxsim.federated.FederatedClient.weight`'s role) rather than a
uniform mean; omitted weights default to ``1`` for every client.

The output is both a new ``--base-model``-shaped ``.onnx`` file (a complete,
deployable model, exactly the shape ``onnxsim.federated.run_federated_round``
returns) and a fresh ``<output>.initial_state.bin`` alongside it, ready to
hand to the *next* round's :func:`generate_federated_lora_step_graph`-produced
clients without them needing to parse the ``.onnx`` file's initializers
themselves.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnx.numpy_helper


def read_weight_layout(manifest_path: str) -> List[Tuple[str, List[int]]]:
    """The ``weight`` lines of a manifest ``generate_federated_lora_step_graph.py``
    wrote -- ``[(name, shape), ...]``, in the same order
    ``initial_state.bin``/``exportTrainedState`` concatenate them in. This is
    the Python-side mirror of ``step_graph_runner.mjs``'s own
    ``parseManifest`` -- kept as independent, format-compatible parsing on
    each side rather than shared code, the same relationship this repo's own
    C++/JS/Python manifest readers already have for the distillation step
    graph.
    """
    layout: List[Tuple[str, List[int]]] = []
    with open(manifest_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "weight":
                name, shape = parts[1], [int(d) for d in parts[2:]]
                layout.append((name, shape))
    if not layout:
        raise ValueError(f"{manifest_path} has no 'weight' lines")
    return layout


def read_client_state(
    path: str, layout: List[Tuple[str, List[int]]]
) -> Dict[str, np.ndarray]:
    """One client's raw float32 file, sliced into a ``{name: array}`` dict
    per ``layout`` -- the inverse of
    ``generate_federated_lora_step_graph.py``'s own
    ``write_manifest_and_initial_state``, and of ``step_graph_runner.mjs``'s
    ``exportTrainedState``: both write exactly this concatenation."""
    raw = np.fromfile(path, dtype=np.float32)
    expected = sum(int(np.prod(shape)) for _, shape in layout)
    if raw.size != expected:
        raise ValueError(
            f"{path} has {raw.size} float32 values but the manifest's weight "
            f"layout expects {expected}"
        )
    state: Dict[str, np.ndarray] = {}
    offset = 0
    for name, shape in layout:
        count = int(np.prod(shape))
        state[name] = raw[offset : offset + count].reshape(shape)
        offset += count
    return state


def fedavg(
    client_states: List[Dict[str, np.ndarray]], weights: List[float]
) -> Dict[str, np.ndarray]:
    """Delegates to :func:`onnxsim.federated.fedavg` -- imported lazily so
    this script's file-parsing halves above are usable (and testable) even
    in a checkout where the ``onnxsim`` C++ extension has not been built,
    matching this tool's own general stance of depending on
    ``onnxsim.graph_grad``/``onnxsim.qat_graph`` only where it actually needs
    the differentiator, never just to move bytes around."""
    from onnxsim import federated

    return federated.fedavg(client_states, weights)


def apply_state(model: onnx.ModelProto, state: Dict[str, np.ndarray]) -> onnx.ModelProto:
    from onnxsim import federated

    return federated.apply_adapter_state(model, state)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument(
        "--client", action="append", required=True, dest="clients",
        help="one client's trained-state .bin file; repeat once per client",
    )
    p.add_argument(
        "--weight", action="append", type=float, dest="weights", default=None,
        help="this client's FedAvg weight (local example count); repeat once "
        "per --client, same order. Defaults to 1 for every client.",
    )
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    weights = args.weights if args.weights is not None else [1.0] * len(args.clients)
    if len(weights) != len(args.clients):
        raise ValueError(
            f"got {len(args.clients)} --client but {len(weights)} --weight; "
            "pass exactly one --weight per --client, or none at all"
        )

    layout = read_weight_layout(args.manifest)
    client_states = [read_client_state(path, layout) for path in args.clients]
    averaged = fedavg(client_states, weights)

    base_model = onnx.load(args.base_model)
    next_global = apply_state(base_model, averaged)
    onnx.save(next_global, args.output)

    initial_state_path = args.output + ".initial_state.bin"
    with open(initial_state_path, "wb") as f:
        for name, _shape in layout:
            averaged[name].astype(np.float32).tofile(f)

    print(
        f"aggregated {len(args.clients)} client(s) (weights={weights}) -> "
        f"{args.output}, {initial_state_path}"
    )


if __name__ == "__main__":
    main()
