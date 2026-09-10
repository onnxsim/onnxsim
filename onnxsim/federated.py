"""Federated averaging on top of :mod:`onnxsim.lora` -- training a shared
LoRA adapter across clients that each hold private data, without any client's
data (or the base model's own weights) ever leaving where it started.

**Why LoRA is the right unit to federate, not the whole model.** A federated
round communicates whatever a client trains, every round, to every
participant -- so its cost is set by how much *state* one round of local
training touches, not by how big the base model is. :func:`onnxsim.lora.train_lora`
already freezes everything except an injected adapter's own ``A``/``B``
matrices (:meth:`onnxsim.lora.LoraAdapter.parameter_names`) and returns a
model with only those initializers changed. Federating LoRA training instead
of full fine-tuning means each round moves rank-sized deltas, not full
weight tensors -- the same reason LoRA is attractive for on-device
personalization in the first place, here reused for its communication cost
rather than its memory cost.

**What this module adds, concretely.** :func:`onnxsim.lora.train_lora` already
does everything one *client's* local training needs. The only genuinely new
piece is round orchestration: broadcast the current global adapter state to
every client, train each independently on its own data starting from that
same state (:func:`run_federated_round`), and average the results back into
one global state (:func:`fedavg`) -- classic FedAvg, applied to an adapter's
parameters instead of a whole model's. Each client's local optimizer state
(Adam's ``m``/``v`` moments) is not part of what gets averaged or carried
between rounds: :func:`onnxsim.lora.train_lora` always starts a client's local
steps from a fresh zeroed optimizer state (see ``onnxsim.lora._train_lora_block``),
exactly as FedAvg's own definition assumes.

**What this does not implement.** Secure aggregation (the server seeing only
the *sum* of client updates, never one client's own) and differential-privacy
noise are both real requirements for production cross-device FL, and neither
is here -- :func:`fedavg` is a plain, visible weighted average. Non-IID-aware
aggregation (FedProx, SCAFFOLD, FedOpt-style server optimizers) is also out of
scope: this is FedAvg, the baseline every one of those improves on, not a
replacement for it. Adding either is a reason to extend this module later,
not a gap in what it claims to do now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import onnx.numpy_helper

from onnxsim import backend, lora
from onnxsim.calibration import Tensors


@dataclass
class FederatedClient:
    """One participant's local, private data for one federated round.

    Never crosses into another client's or the server's hands -- only the
    adapter state :func:`run_federated_round` extracts *after* local training
    does. Exactly one of ``reference_model``/``target_data`` must be given,
    mirroring :func:`onnxsim.lora.train_lora`'s own contract: a client with
    real local labels uses ``target_data`` (actual supervised fine-tuning on
    that client's data); a client with unlabeled data can still contribute by
    label-free distillation against a shared ``reference_model``. Nothing
    requires every client in one round to pick the same option.

    :param calibration_data: this client's local input batches, in the same
            axis-0-concatenation contract :func:`onnxsim.lora.train_lora` takes.
    :param target_data: this client's local labels, one array per
            ``calibration_data`` batch. Mutually exclusive with
            ``reference_model``.
    :param reference_model: a shared teacher model this client distills
            against instead of using its own labels. Mutually exclusive with
            ``target_data``.
    :param num_examples: this client's example count, for FedAvg's weighted
            average (:func:`fedavg`). Defaults to ``len(calibration_data)``,
            i.e. one example per batch -- pass this explicitly when a batch
            holds more than one example, or when a client should be weighted
            by something other than a raw batch count.
    """

    calibration_data: Sequence[Tensors]
    target_data: Optional[Sequence[np.ndarray]] = None
    reference_model: Optional[Union[str, onnx.ModelProto]] = None
    num_examples: Optional[int] = None

    def __post_init__(self) -> None:
        if (self.reference_model is None) == (self.target_data is None):
            raise ValueError(
                "FederatedClient needs exactly one of reference_model or target_data"
            )

    def weight(self) -> int:
        """This client's FedAvg weight -- its local example count."""
        if self.num_examples is not None:
            return self.num_examples
        return len(self.calibration_data)


def extract_adapter_state(
    model: onnx.ModelProto, adapter: lora.LoraAdapter
) -> Dict[str, np.ndarray]:
    """The current value of every one of ``adapter``'s ``A``/``B``
    initializers in ``model`` -- what one round communicates, in either
    direction: a client's return trip after local training, or (via
    :func:`apply_adapter_state`) the server's broadcast of the next round's
    starting point."""
    initializer_map = {t.name: t for t in model.graph.initializer}
    return {
        name: onnx.numpy_helper.to_array(initializer_map[name]).astype(np.float32)
        for name in adapter.parameter_names()
    }


def apply_adapter_state(
    model: onnx.ModelProto, state: Dict[str, np.ndarray]
) -> onnx.ModelProto:
    """``model`` with every initializer named in ``state`` overwritten by its
    value, and everything else -- base weights included -- byte-for-byte
    unchanged. The inverse of :func:`extract_adapter_state`, and how a
    federated round's averaged result (or a fresh round's broadcast of the
    previous round's global state) becomes the model the next round of local
    training starts from."""
    out = onnx.ModelProto()
    out.CopyFrom(model)
    for initializer in out.graph.initializer:
        if initializer.name in state:
            initializer.CopyFrom(
                onnx.numpy_helper.from_array(
                    state[initializer.name].astype(np.float32),
                    name=initializer.name,
                )
            )
    onnx.checker.check_model(out)
    return out


def fedavg(
    client_states: Sequence[Dict[str, np.ndarray]],
    weights: Sequence[float],
) -> Dict[str, np.ndarray]:
    """The weighted average of several clients' adapter states, tensor by
    tensor -- McMahan et al.'s FedAvg, applied to whatever
    :func:`extract_adapter_state` returned for each client rather than to a
    whole model's weights.

    Accumulates in float64 and casts back to float32 at the end, so summing
    many clients' contributions does not lose precision that the individual
    float32 states never had reason to carry.

    :param client_states: one :func:`extract_adapter_state` result per
            client that took part in the round, all for the same adapter (so
            all with identical key sets).
    :param weights: one non-negative weight per entry of ``client_states``,
            typically :meth:`FederatedClient.weight`'s local example count.
    :raises ValueError: if ``client_states`` is empty, if ``weights`` has a
            different length, if the weights do not sum to a positive number,
            or if the client states do not all share the same tensor names.
    """
    if not client_states:
        raise ValueError("fedavg needs at least one client state")
    if len(weights) != len(client_states):
        raise ValueError(
            f"fedavg got {len(client_states)} client states but {len(weights)} weights"
        )
    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"fedavg weights must sum to a positive number, got {total}")

    names = set(client_states[0])
    for state in client_states[1:]:
        if set(state) != names:
            raise ValueError("fedavg needs every client state to name the same tensors")

    averaged: Dict[str, np.ndarray] = {}
    for name in names:
        acc = np.zeros_like(client_states[0][name], dtype=np.float64)
        for state, weight in zip(client_states, weights):
            acc += state[name].astype(np.float64) * (weight / total)
        averaged[name] = acc.astype(np.float32)
    return averaged


def run_federated_round(
    global_model: onnx.ModelProto,
    adapter: lora.LoraAdapter,
    clients: Sequence[FederatedClient],
    block_input_name: str,
    block_output_name: str,
    local_iterations: int = 10,
    learning_rate: float = 1e-3,
    lr_decay: bool = True,
    providers: Optional[Sequence[backend.Provider]] = None,
    step_providers: Optional[Sequence[backend.Provider]] = None,
) -> Tuple[onnx.ModelProto, List[float]]:
    """One federated round: every client in ``clients`` trains
    :func:`onnxsim.lora.train_lora` independently for ``local_iterations``
    steps, starting from ``global_model``'s current adapter state, on its own
    data alone; the resulting adapter states are combined by :func:`fedavg`,
    weighted by each client's :meth:`FederatedClient.weight`, into the next
    global state.

    Every client starts from the *same* broadcast state -- ``global_model``
    is read, never mutated, so training one client cannot see another's
    updates from the same round (that is next round's global state, not this
    one's).

    :param global_model: an :func:`onnxsim.lora.inject_lora`-injected model
            holding the current global adapter state.
    :param adapter: the :class:`onnxsim.lora.LoraAdapter`
            :func:`onnxsim.lora.inject_lora` returned alongside
            ``global_model``.
    :param clients: this round's participants. At least one is required; a
            production deployment would also sample a subset of a larger
            client population per round, which is a policy this function
            leaves to its caller -- pass exactly the clients meant to
            participate.
    :param block_input_name: the activation entering the trained block, the
            same argument :func:`onnxsim.lora.train_lora` takes.
    :param block_output_name: the block's own output, whose reconstruction
            error against each client's target is that client's local loss.
    :param local_iterations: local optimizer steps each client runs before
            reporting back -- FedAvg's ``E`` (local epochs), expressed here as
            a step count rather than an epoch count since
            :func:`onnxsim.lora.train_lora` itself takes one.
    :param learning_rate: local learning rate, shared by every client this
            round.
    :param lr_decay: anneal each client's local learning rate to zero over
            its own ``local_iterations``, the same flag
            :func:`onnxsim.lora.train_lora` takes -- decay resets every
            round, since each client's local run is independent.
    :param providers: onnxruntime execution providers for capturing each
            client's calibration activations (only used when a client trains
            against a ``reference_model``; unused for ``target_data``
            clients, matching :func:`onnxsim.lora.train_lora`).
    :param step_providers: onnxruntime execution providers to run local
            training's step graph on.
    :returns: ``(next global model, one final local loss per client, in
            ``clients`` order)``. A client whose local run recorded no losses
            (impossible with ``local_iterations >= 1``, kept only because
            ``train_lora``'s own ``losses`` list could in principle be empty)
            reports ``nan`` rather than raising.
    :raises ValueError: if ``clients`` is empty.
    """
    if not clients:
        raise ValueError("run_federated_round needs at least one client")

    client_states: List[Dict[str, np.ndarray]] = []
    weights: List[float] = []
    round_losses: List[float] = []
    for client in clients:
        losses: List[float] = []
        tuned = lora.train_lora(
            global_model,
            adapter,
            block_input_name,
            block_output_name,
            reference_model=client.reference_model,
            target_data=client.target_data,
            calibration_data=client.calibration_data,
            num_iterations=local_iterations,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            providers=providers,
            step_providers=step_providers,
            losses=losses,
        )
        client_states.append(extract_adapter_state(tuned, adapter))
        weights.append(client.weight())
        round_losses.append(losses[-1] if losses else float("nan"))

    averaged = fedavg(client_states, weights)
    next_global = apply_adapter_state(global_model, averaged)
    return next_global, round_losses


def run_federated_training(
    global_model: onnx.ModelProto,
    adapter: lora.LoraAdapter,
    clients: Sequence[FederatedClient],
    block_input_name: str,
    block_output_name: str,
    num_rounds: int = 5,
    local_iterations: int = 10,
    learning_rate: float = 1e-3,
    lr_decay: bool = True,
    providers: Optional[Sequence[backend.Provider]] = None,
    step_providers: Optional[Sequence[backend.Provider]] = None,
    round_losses: Optional[List[List[float]]] = None,
) -> onnx.ModelProto:
    """Runs :func:`run_federated_round` ``num_rounds`` times, feeding each
    round's resulting global model in as the next round's starting point.

    The same fixed ``clients`` list takes part in every round; sampling a
    different subset of a larger population per round is, as in
    :func:`run_federated_round`, left to the caller -- call
    :func:`run_federated_round` directly, once per round, to vary who
    participates.

    :param round_losses: when given, one entry per round is appended -- the
            list :func:`run_federated_round` returned for that round (one
            final loss per client). Diagnostic only; nothing here reads it
            back.
    :returns: the global model after all ``num_rounds`` rounds.
    """
    model = global_model
    for _ in range(num_rounds):
        model, losses = run_federated_round(
            model,
            adapter,
            clients,
            block_input_name,
            block_output_name,
            local_iterations=local_iterations,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            providers=providers,
            step_providers=step_providers,
        )
        if round_losses is not None:
            round_losses.append(losses)
    return model
