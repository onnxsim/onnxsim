# Distributed (data-parallel) training for onnxsim: plan

**Status: initial implementation, scope deliberately narrow.** This is the
"distributed" counterpart to `onnxsim/federated.py`'s FedAvg work
(`docs/`-adjacent design note, not a comparison writeup like
`nncf-comparison-future-work.md`). It records why data parallelism -- and
specifically *not* tensor/pipeline parallelism -- is the only form of
"distributed training" that fits onnxsim's existing architecture without a
large, separate investment, and what `onnxsim/distributed.py` actually
implements as a first version.

## Why data parallelism, and not tensor/pipeline parallelism

Every training-capable piece of onnxsim (`onnxsim.graph_grad`,
`onnxsim.qat_graph`, `onnxsim.lora`, `onnxsim.qat`) is built around **one
self-contained ONNX graph per process**: forward, loss, backward, and
(usually) an optimizer step, all baked into ordinary ONNX nodes and run
through a single `InferenceSession`. Real tensor parallelism needs the
*graph itself* split across devices with explicit collectives inserted at
exactly the points a sharded contraction axis requires them (see
`scripts/distributed_tp_sketch.py` and the survey behind it) -- ONNX has no
maintained, first-class way to express that today (ORT's own
`MegatronF`/`MegatronG`/`Send`/`Recv` collective-with-gradient ops are a
mostly-abandoned corner of `orttraining`), and hand-rolling the equivalent
gradient rules inside `onnxsim.graph_grad` would be a large, narrowly-useful
investment for a project whose actual niche is fine-tuning/QAT/LoRA on
modestly-sized models, not multi-billion-parameter pretraining.

Plain gradient-averaging data parallelism needs none of that. It only
requires:

1. Splitting **gradient computation** from **optimizer application** --
   currently fused into one graph by `onnxsim.qat_graph.make_step_graph`
   (see its own module docstring) for every existing caller, because every
   existing caller trains alone, on one device. Once split, one all-reduce
   between the two is enough.
2. A gradient average across workers -- an ordinary Python-level
   collective, not an ONNX graph construct. ONNX/ORT stay the *local*
   compute engine on each worker; the collective lives entirely in the
   orchestration layer around them, the same shape vLLM/DeepSpeed give
   PyTorch (local compute) and NCCL (the collective) as separate concerns.

That is a small, self-contained addition on top of existing, tested
primitives (`graph_grad.build_backward`, `qat_graph.adam_update`,
`qat_graph.make_step_graph`) -- not a new differentiation engine, not a new
IR, and not a dependency on any onnxruntime build beyond the plain one
`pip install onnxruntime` already gives every other onnxsim training path.

## What "distributed" means here, concretely

Real OS-level parallelism -- one Python **process** per worker
(`multiprocessing.Process`), each running its own `onnxruntime`
`InferenceSession` on its own private data shard, communicating gradients
back to a parameter-server-style parent process over `multiprocessing.Pipe`.
This is a real, if minimal, distributed system (genuine inter-process
communication, not a single-process simulation like `onnxsim/federated.py`'s
sequential client loop or `scripts/distributed_tp_sketch.py`'s numpy-only
collective stand-in) -- it just runs on one machine's CPU cores rather than
a GPU cluster, since that's what's available to develop and test this
against. Swapping the transport (`multiprocessing.Pipe` &rarr; MPI/NCCL/a
real network) is future work explicitly deferred, not attempted here; see
"Left for later" below.

## Architecture

```
                    ┌─────────────────────────┐
                    │   parent process         │
                    │                           │
     ┌─────current  │  1. broadcast params      │
     │   params      │  2. collect N gradients   │
     │              │  3. average them           │
     │              │  4. apply_step (Adam)      │
     │              │     -- one ORT session,    │
     │              │        one call per step   │
     │              └─────────────────────────┘
     │                    │            ▲
     ▼                    ▼            │
┌─────────┐        ┌─────────┐   gradients
│ worker 0 │        │ worker 1 │   (+ local loss,
│ own ORT  │  ...   │ own ORT  │    for logging)
│ session, │        │ session, │
│ own data │        │ own data │
│ shard    │        │ shard    │
└─────────┘        └─────────┘
  grad_step graph      grad_step graph
  (forward+backward,   (identical graph,
   no optimizer)        different process)
```

- **`build_gradient_step_graph`**: forward + MSE loss + backward via
  `onnxsim.graph_grad.build_backward`, restricted to `param_names` exactly
  like `onnxsim.lora`'s block training does -- but with no optimizer wired
  in and no state threading, since a worker's own weights never persist
  between steps: they're fed fresh from the parent every step (the
  broadcast in the diagram above), and the worker's only job is turning
  `(current params, local batch)` into `(gradients, loss)`.
- **`build_apply_step_graph`**: exactly one `onnxsim.qat_graph.adam_update`
  call per parameter, wired through `make_step_graph`'s existing state
  mechanism -- this is *not* new graph-building logic, it's the same
  Adam-as-a-step-graph idiom every other onnxsim training path already uses,
  just fed an already-averaged gradient instead of one it computed itself.
- **`train_data_parallel`**: the orchestration loop above. Runs entirely in
  the parent process except for each worker's own forward/backward, which is
  real parallel work across OS processes.

## Correctness property this is expected to hold

Averaging equal-sized workers' gradients of a mean-loss is mathematically
identical to computing that same mean-loss's gradient over the concatenated
combined batch directly -- the same equivalence
`scripts/distributed_tp_sketch.py` checks for its own tensor+data-parallel
simulation. `tests/test_distributed.py` checks it here too: N-worker
data-parallel training must match a single-process, non-parallel reference
run on the concatenated data, to float32 tolerance, every step.

## Left for later (explicitly out of scope for this version)

- **Real network transport.** `multiprocessing.Pipe` only works within one
  machine. A real multi-node version would swap this for MPI or a thin
  socket protocol; the gradient-averaging *logic* above does not change,
  only how bytes move between processes.
- **ZeRO-style optimizer-state sharding.** Every worker-independent
  parameter here has its Adam state held once, in the parent process --
  fine at the scale onnxsim actually targets, not a fit for models too big
  for one machine's memory (which is also not what tensor/pipeline
  parallelism being out of scope, above, was ever going to solve here
  either).
- **Fault tolerance / elastic membership.** A worker dying mid-run currently
  hangs the parent waiting on its pipe. A production version would need
  timeouts and the ability to drop/replace a worker, the same gap flagged
  for the federated round-trip protocol in the FL track's own discussion.
- **Overlap of communication with compute.** Every step here is fully
  synchronous (broadcast, wait for all gradients, average, apply) with no
  attempt to prefetch or overlap, unlike FSDP2's overlapped all-gathers.
  Worth revisiting once there's a real workload to profile against.
