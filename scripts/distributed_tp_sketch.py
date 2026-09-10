"""A runnable sketch of the Megatron-style tensor-parallel + data-parallel
training flow described in ``docs/`` discussion of ONNX's distributed-training
ops -- turning that conceptual sketch into code that actually executes and
checks its own arithmetic, the same way this repo's tests compare a graph's
gradient against an independent computation rather than trusting it.

**Two halves, deliberately kept apart.**

1. :func:`build_inspectable_rank_graph` builds one rank's *real* ONNX graph,
   using the actual collective op types ONNX Runtime's (legacy, largely
   unmaintained today) ``orttraining`` module defines with working gradient
   rules: ``MegatronF``/``MegatronG`` (the tensor-parallel "conjugate pair" --
   identity-forward/all-reduce-backward and its mirror) and ``NcclAllReduce``
   (data-parallel gradient averaging, selected by a ``group_type`` attribute).
   This graph is built and printed for inspection only -- it is never run.
   Running it for real needs an onnxruntime build compiled with
   ``--enable_training --use_nccl``, which is nowhere close to what
   ``pip install onnxruntime`` gives you (see this repo's own ``CLAUDE.md``:
   even the *wheel build* of onnxsim itself never compiles ONNX Runtime, let
   alone a training+NCCL one), and it needs an actual multi-GPU NCCL fabric
   besides. Neither exists in a sandbox, so this half exists purely so the
   node types, domains, and attributes discussed are something you can look
   at rather than take on faith.

2. :func:`run_tensor_and_data_parallel_step` and friends do the *identical
   math* in plain numpy, across ``tp_size * dp_size`` simulated ranks in one
   process -- standing in for what ``MegatronF``/``MegatronG``/
   ``NcclAllReduce`` would do on a real NCCL fabric, using the exact same
   Adam update rule :mod:`onnxsim.qat_graph` uses for every other training
   loop in this repo. :func:`main` then checks the simulated multi-rank
   result against a single-rank, non-parallel reference computation on the
   same combined data -- forward output, gradients, and post-optimizer
   weights all have to agree, which is only true if the collectives were
   simulated in the right place with the right reduction. That is the
   property this script actually verifies; it is not just a demo that runs
   without crashing.

**What this is not.** Not a contribution to :mod:`onnxsim` itself -- nothing
here is imported by the package, which is why it lives under ``scripts/``
rather than ``onnxsim/``. Not a working distributed training system: real
tensor parallelism needs a real NCCL fabric and a real multi-process launch
(``mpirun``/``torchrun``-equivalent), neither of which this script attempts.
And not a claim that the ``com.microsoft`` attribute names in
:func:`build_inspectable_rank_graph` are pinned to one exact onnxruntime
version -- they are reconstructed from ``orttraining``'s own source
(``training_op_defs.cc``, ``gradient_builder.cc``, ``collective_defs.cc``)
rather than a live schema registry, since that domain's training ops are not
registered in a default onnxruntime build and so cannot be checked against a
live schema here either. Treat that half as a documentation artifact, not a
tested contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import onnx
from onnx import helper

from onnxsim.qat_graph import ADAM_BETA1, ADAM_BETA2, ADAM_EPS


# ---------------------------------------------------------------------------
# Half 1: the real ONNX graph, for inspection only.
# ---------------------------------------------------------------------------


def build_inspectable_rank_graph(
    tp_size: int, d_in: int, d_hidden: int, d_out: int
) -> onnx.ModelProto:
    """One rank's forward-only graph for a column-parallel-then-row-parallel
    MLP block (the same block :func:`run_tensor_and_data_parallel_step`
    simulates), using ``com.microsoft``'s real collective op types.

    Only the forward pass is built here: ``MegatronF``/``MegatronG`` carry
    their own registered gradient rules in a real ``orttraining`` build (that
    is the entire point of the "conjugate pair" -- a differentiable graph
    transformation library can call ``build_backward`` on this exactly like
    any other op, without a hand-written collective-aware backward pass), so
    there is nothing distributed-specific to add on the backward side of a
    *real* build. This script's own backward pass (in the numpy half) is
    hand-derived instead, since :mod:`onnxsim.graph_grad` has no rule for a
    domain onnxruntime's default build does not even register.

    ``d_hidden`` must be divisible by ``tp_size`` -- each rank owns a
    ``d_hidden / tp_size`` column slice of ``W1`` and the matching row slice
    of ``W2``, exactly the split :func:`shard_weights` computes for the
    numpy simulation.
    """
    if d_hidden % tp_size != 0:
        raise ValueError(f"d_hidden={d_hidden} must be divisible by tp_size={tp_size}")
    shard = d_hidden // tp_size

    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["batch", d_in])
    w1 = helper.make_tensor_value_info("w1_shard", onnx.TensorProto.FLOAT, [d_in, shard])
    w2 = helper.make_tensor_value_info("w2_shard", onnx.TensorProto.FLOAT, [shard, d_out])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, ["batch", d_out])

    nodes = [
        # MegatronF: identity in the forward direction. Its entire reason to
        # exist is the *backward* rule a real build registers for it
        # (all-reduce of the incoming gradient across the tensor-parallel
        # group) -- see this function's docstring.
        helper.make_node("MegatronF", ["x"], ["x_f"], domain="com.microsoft"),
        helper.make_node("MatMul", ["x_f", "w1_shard"], ["z"]),
        helper.make_node("Tanh", ["z"], ["h"]),
        helper.make_node("MatMul", ["h", "w2_shard"], ["y_partial"]),
        # MegatronG: all-reduce in the forward direction -- this is the node
        # that turns each rank's partial sum (row-parallel W2's contraction
        # axis is sharded) into the true, replicated output.
        helper.make_node("MegatronG", ["y_partial"], ["y"], domain="com.microsoft"),
    ]

    graph = helper.make_graph(nodes, f"tp_rank_forward_shard_{shard}", [x, w1, w2], [y])
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            # Version 1 of a domain this repo's onnx package does not
            # register a schema for -- see the module docstring's third
            # paragraph for why this model is never passed to a checker or a
            # runtime.
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    return model


def build_data_parallel_gradient_sync_graph(d_in: int, shard: int) -> onnx.ModelProto:
    """The other collective a real per-rank *training* graph would contain:
    ``NcclAllReduce`` averaging one rank's local weight-shard gradient across
    the *data*-parallel group, immediately before the optimizer update reads
    it -- the classic DP gradient-average, orthogonal to the tensor-parallel
    pair above (it reduces across a different mesh axis).

    ``group_type`` is reconstructed from ``orttraining``'s own
    ``NcclAllReduce`` schema (a small enum: global/data/node-local/
    cross-node/horizontal/model); ``"data"`` selects the data-parallel
    communicator specifically, as opposed to a tensor- or pipeline-parallel
    one a *different* node in the same real graph might reduce across.
    """
    grad_in = helper.make_tensor_value_info(
        "grad_w1_shard_local", onnx.TensorProto.FLOAT, [d_in, shard]
    )
    grad_out = helper.make_tensor_value_info(
        "grad_w1_shard_dp_averaged", onnx.TensorProto.FLOAT, [d_in, shard]
    )
    node = helper.make_node(
        "NcclAllReduce",
        ["grad_w1_shard_local"],
        ["grad_w1_shard_dp_averaged"],
        domain="com.microsoft",
        group_type="data",
    )
    graph = helper.make_graph(
        [node], "dp_gradient_sync", [grad_in], [grad_out]
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


# ---------------------------------------------------------------------------
# Half 2: the same math, in numpy, standing in for a real NCCL fabric.
# ---------------------------------------------------------------------------


def activation(z: np.ndarray) -> np.ndarray:
    """Stands in for the GELU a real Megatron MLP uses -- swapped for
    ``tanh`` purely so this script needs no dependency beyond numpy (numpy
    has no ``erf``). Nothing about the parallelism pattern being checked
    depends on which pointwise nonlinearity sits here."""
    return np.tanh(z)


def activation_grad(z: np.ndarray) -> np.ndarray:
    t = np.tanh(z)
    return 1.0 - t * t


@dataclass
class RankShard:
    """One tensor-parallel rank's own slice of the block's two weights, plus
    that shard's own Adam moments -- each TP rank keeps a full, un-sharded
    optimizer state for the parameters *it* owns, exactly as real Megatron-LM
    does (ZeRO-style optimizer-state sharding is a separate, orthogonal
    technique this sketch does not add)."""

    w1: np.ndarray  # [d_in, d_hidden / tp_size] -- column-parallel
    w2: np.ndarray  # [d_hidden / tp_size, d_out] -- row-parallel
    m1: np.ndarray
    v1: np.ndarray
    m2: np.ndarray
    v2: np.ndarray


def shard_weights(w1_full: np.ndarray, w2_full: np.ndarray, tp_size: int) -> List[RankShard]:
    """Splits one un-sharded ``(W1, W2)`` pair into ``tp_size``
    :class:`RankShard`\\ s -- ``W1`` column-wise (its output/hidden axis),
    ``W2`` row-wise (its input/hidden axis, the contraction axis the
    row-parallel matmul sums over). Every TP rank in every data-parallel
    replica starts from the *same* shard, exactly like a real job's initial
    broadcast of the model before training begins.
    """
    d_hidden = w1_full.shape[1]
    if d_hidden % tp_size != 0:
        raise ValueError(f"d_hidden={d_hidden} must be divisible by tp_size={tp_size}")
    shard = d_hidden // tp_size
    shards = []
    for r in range(tp_size):
        w1_r = w1_full[:, r * shard : (r + 1) * shard].copy()
        w2_r = w2_full[r * shard : (r + 1) * shard, :].copy()
        shards.append(
            RankShard(w1_r, w2_r, np.zeros_like(w1_r), np.zeros_like(w1_r),
                       np.zeros_like(w2_r), np.zeros_like(w2_r))
        )
    return shards


def tp_forward(
    shards: Sequence[RankShard], x: np.ndarray
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """One tensor-parallel group's forward pass -- ``MegatronF``'s forward
    (identity on ``x``, so every rank reads the same replicated activation)
    followed by each rank's local column-parallel/row-parallel matmul pair,
    followed by ``MegatronG``'s forward (summing every rank's partial output
    -- the all-reduce a row-parallel contraction axis requires before its
    result means anything)."""
    zs, hs, y_partials = [], [], []
    for shard in shards:
        z = x @ shard.w1
        h = activation(z)
        y_partials.append(h @ shard.w2)
        zs.append(z)
        hs.append(h)
    y = sum(y_partials)
    return y, zs, hs


def tp_backward(
    shards: Sequence[RankShard],
    x: np.ndarray,
    zs: Sequence[np.ndarray],
    hs: Sequence[np.ndarray],
    dl_dy: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """The tensor-parallel group's backward pass. ``MegatronG``'s backward
    is Identity: since ``y = sum(y_partial_r)``, ``d(y)/d(y_partial_r) = 1``
    for every rank, so every rank receives the *same* ``dl_dy`` unchanged --
    no communication needed here, which is the whole reason the pair is
    asymmetric (all-reduce forward, identity backward) rather than
    all-reduce on both sides.

    Each rank's own weight gradients are then local and require no further
    communication -- a column-parallel/row-parallel split gives every rank a
    *disjoint* slice of each weight, so there is nothing to reduce. The one
    remaining collective is ``MegatronF``'s backward: every rank's local
    contribution to ``dl/dx`` must be all-reduced, since ``x`` was
    replicated and each rank only computed its own shard's share of how
    ``x`` affected the loss.
    """
    grads_w1, grads_w2, dx_partials = [], [], []
    for shard, z, h in zip(shards, zs, hs):
        # MegatronG backward: Identity -- dl_dy reaches every rank unchanged.
        grad_w2 = h.T @ dl_dy
        dl_dh = dl_dy @ shard.w2.T
        dl_dz = dl_dh * activation_grad(z)
        grad_w1 = x.T @ dl_dz
        dx_partials.append(dl_dz @ shard.w1.T)
        grads_w1.append(grad_w1)
        grads_w2.append(grad_w2)
    dx = sum(dx_partials)  # MegatronF backward: all-reduce.
    return grads_w1, grads_w2, dx


def adam_step(
    param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray, lr: float, t: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The exact update rule :func:`onnxsim.qat_graph.adam_update` builds as
    an ONNX graph, reused here directly in numpy. A real onnxsim-based
    deployment would run this rank's local optimizer step as a step graph
    exactly the way :mod:`onnxsim.lora`/:mod:`onnxsim.qat` already do; there
    is no distributed-specific reason to depart from that once the gradient
    this function receives has already been correctly reduced, which is
    :func:`run_tensor_and_data_parallel_step`'s job, not this one's.
    """
    m_next = ADAM_BETA1 * m + (1 - ADAM_BETA1) * grad
    v_next = ADAM_BETA2 * v + (1 - ADAM_BETA2) * (grad * grad)
    m_hat = m_next / (1 - ADAM_BETA1 ** t)
    v_hat = v_next / (1 - ADAM_BETA2 ** t)
    param_next = param - lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
    return param_next, m_next, v_next


def run_tensor_and_data_parallel_step(
    replicas: List[List[RankShard]],
    xs: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    lr: float,
    step_t: int,
) -> float:
    """One full distributed training step across ``dp_size`` data-parallel
    replicas, each an independent ``tp_size``-way tensor-parallel group --
    ``replicas[d][r]`` is data-parallel replica ``d``'s tensor-parallel rank
    ``r``, mutated in place exactly like a real rank would update its own
    local weights.

    Order of operations mirrors the real per-rank graph: local
    forward/backward within each TP group (no cross-replica communication
    yet), then ``NcclAllReduce(group_type="data")`` averaging each
    (``d``-independent) TP rank's own gradient shard across every DP
    replica, then a local Adam step per shard. Returns the mean loss across
    every replica, for logging only.
    """
    dp_size = len(replicas)
    tp_size = len(replicas[0])
    losses = []

    # Local forward + backward, independently per data-parallel replica.
    per_replica_grads_w1: List[List[np.ndarray]] = []
    per_replica_grads_w2: List[List[np.ndarray]] = []
    for d in range(dp_size):
        y, zs, hs = tp_forward(replicas[d], xs[d])
        diff = y - targets[d]
        losses.append(float(np.mean(diff * diff)))
        dl_dy = diff * (2.0 / diff.size)
        grads_w1, grads_w2, _dx = tp_backward(replicas[d], xs[d], zs, hs, dl_dy)
        per_replica_grads_w1.append(grads_w1)
        per_replica_grads_w2.append(grads_w2)

    # NcclAllReduce(group_type="data"): average each TP rank's gradient
    # shard across the data-parallel group -- same rank index ``r``, summed
    # over the ``d`` axis instead of the ``r`` axis MegatronG summed over.
    for r in range(tp_size):
        avg_grad_w1 = sum(per_replica_grads_w1[d][r] for d in range(dp_size)) / dp_size
        avg_grad_w2 = sum(per_replica_grads_w2[d][r] for d in range(dp_size)) / dp_size
        for d in range(dp_size):
            shard = replicas[d][r]
            shard.w1, shard.m1, shard.v1 = adam_step(
                shard.w1, avg_grad_w1, shard.m1, shard.v1, lr, step_t
            )
            shard.w2, shard.m2, shard.v2 = adam_step(
                shard.w2, avg_grad_w2, shard.m2, shard.v2, lr, step_t
            )

    return float(np.mean(losses))


def run_reference_step(
    w1: np.ndarray, w2: np.ndarray, m1, v1, m2, v2, x: np.ndarray, target: np.ndarray,
    lr: float, step_t: int,
):
    """The same block, computed with no parallelism at all, on the
    *combined* batch every data-parallel replica's ``x``/``target`` would
    concatenate into. This is what
    :func:`run_tensor_and_data_parallel_step`'s result must agree with: DP
    gradient-averaging over equal-sized replica batches is mathematically
    identical to computing the mean-loss gradient over the concatenated
    batch directly, and un-sharding ``W1``/``W2`` before the matmuls makes
    tensor parallelism's own reduction (``MegatronG``'s all-reduce) a no-op
    by construction -- there is only one rank, so there is nothing to sum.
    """
    z = x @ w1
    h = activation(z)
    y = h @ w2
    diff = y - target
    loss = float(np.mean(diff * diff))
    dl_dy = diff * (2.0 / diff.size)
    grad_w2 = h.T @ dl_dy
    dl_dh = dl_dy @ w2.T
    dl_dz = dl_dh * activation_grad(z)
    grad_w1 = x.T @ dl_dz
    w1, m1, v1 = adam_step(w1, grad_w1, m1, v1, lr, step_t)
    w2, m2, v2 = adam_step(w2, grad_w2, m2, v2, lr, step_t)
    return w1, w2, m1, v1, m2, v2, loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--d-in", type=int, default=8)
    parser.add_argument("--d-hidden", type=int, default=16)
    parser.add_argument("--d-out", type=int, default=8)
    parser.add_argument("--batch-per-replica", type=int, default=6)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=== Half 1: the real ONNX collective ops, for inspection ===\n")
    rank_graph = build_inspectable_rank_graph(args.tp_size, args.d_in, args.d_hidden, args.d_out)
    print(helper.printable_graph(rank_graph.graph))
    shard = args.d_hidden // args.tp_size
    dp_sync_graph = build_data_parallel_gradient_sync_graph(args.d_in, shard)
    print()
    print(helper.printable_graph(dp_sync_graph.graph))

    print("\n=== Half 2: simulating what those ops would do, in numpy ===\n")
    rng = np.random.default_rng(args.seed)
    w1_full = (rng.standard_normal((args.d_in, args.d_hidden)) / np.sqrt(args.d_in)).astype(np.float32)
    w2_full = (rng.standard_normal((args.d_hidden, args.d_out)) / np.sqrt(args.d_hidden)).astype(np.float32)

    replicas = [shard_weights(w1_full, w2_full, args.tp_size) for _ in range(args.dp_size)]

    xs = [
        rng.standard_normal((args.batch_per_replica, args.d_in)).astype(np.float32)
        for _ in range(args.dp_size)
    ]
    targets = [
        rng.standard_normal((args.batch_per_replica, args.d_out)).astype(np.float32)
        for _ in range(args.dp_size)
    ]

    ref_w1, ref_w2 = w1_full.copy(), w2_full.copy()
    ref_m1 = ref_v1 = np.zeros_like(ref_w1)
    ref_m2 = ref_v2 = np.zeros_like(ref_w2)
    x_combined = np.concatenate(xs, axis=0)
    target_combined = np.concatenate(targets, axis=0)

    for t in range(1, args.steps + 1):
        sim_loss = run_tensor_and_data_parallel_step(replicas, xs, targets, args.lr, t)
        ref_w1, ref_w2, ref_m1, ref_v1, ref_m2, ref_v2, ref_loss = run_reference_step(
            ref_w1, ref_w2, ref_m1, ref_v1, ref_m2, ref_v2,
            x_combined, target_combined, args.lr, t,
        )
        if t == 1 or t % 5 == 0 or t == args.steps:
            print(f"step {t:3d}  simulated (tp={args.tp_size},dp={args.dp_size}) loss="
                  f"{sim_loss:.6f}   non-parallel reference loss={ref_loss:.6f}")

    # The check that actually matters: un-sharding every replica's rank-0
    # trained shards back into one matrix must reproduce the reference
    # exactly (up to float32 rounding) -- every replica trains the *same*
    # weights, so any one of them (here replica 0) is the whole answer.
    trained_w1 = np.concatenate([shard.w1 for shard in replicas[0]], axis=1)
    trained_w2 = np.concatenate([shard.w2 for shard in replicas[0]], axis=0)
    np.testing.assert_allclose(trained_w1, ref_w1, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(trained_w2, ref_w2, rtol=1e-4, atol=1e-5)
    for d in range(1, args.dp_size):
        other_w1 = np.concatenate([shard.w1 for shard in replicas[d]], axis=1)
        other_w2 = np.concatenate([shard.w2 for shard in replicas[d]], axis=0)
        np.testing.assert_allclose(other_w1, trained_w1, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(other_w2, trained_w2, rtol=1e-6, atol=1e-7)

    print(
        f"\nOK: {args.tp_size}-way tensor-parallel x {args.dp_size}-way data-parallel "
        f"training for {args.steps} steps matches the non-parallel reference, "
        f"and every data-parallel replica converged to identical weights."
    )


if __name__ == "__main__":
    main()
