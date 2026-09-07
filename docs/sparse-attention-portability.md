# Porting AngelSlim's sparse attention to onnxsim: a negative result

**Status: research note.** This records an investigation into whether any of
[AngelSlim](https://github.com/Tencent/AngelSlim)'s **sparse attention**
algorithms (Stem, MInference, FlexPrefill, XAttention, FlashPrefill,
VecAttention, CoSA) can be ported into onnxsim, and why the answer is **no**
for all seven. It is not a roadmap and not a rejection of the underlying
research -- the algorithms are real and they work where they were designed to
work. It exists so that the next person who asks "can we do sparse attention
too?" does not have to re-derive the answer, and so that the one framing under
which a port *would* become worthwhile is written down precisely enough to
recognise if it ever arrives.

The conclusion in one sentence: **sparse attention's entire benefit is a
*kernel that skips work at run time*, and a static ONNX graph rewrite cannot
skip work -- the two ONNX/ORT constructs that come closest (a baked `-inf`
attention mask, and `com.microsoft::SparseAttention`'s CSR block mask on the
CPU EP) both compute the full dense attention anyway, and measure 1.18x-2.0x
*slower* than plain causal attention while keeping only 28.6-28.8% of the score
entries.**

## What AngelSlim's sparse attention actually is

AngelSlim scopes it explicitly as a **Prefill acceleration module**: its own
doc's first sentence is that the core goal is to "dynamically skip unimportant
attention blocks during inference, significantly reducing Prefill compute and
latency" (`docs/source/features/sparse_attention/index.md`). It is not a
compression technique -- nothing gets smaller, no weights are touched, the
checkpoint is unchanged. It is a *serving-time kernel swap*: AngelSlim
monkey-patches each attention module's `forward` (`build_attn_forward` ->
`modules/forward.py`) so the prefill call routes to a block-sparse
Triton/CUDA kernel instead of FlashAttention/SDPA.

Constraints AngelSlim itself lists for **all** algorithms: no padding mask
(batch=1 or unpadded input), some kernels require batch=1, prefix-cache /
chunked prefill silently falls back to dense (sparsity only engages on a true
first fill, `k_len == q_len`), no sliding-window layers, no tensor parallelism
(`WORLD_SIZE>1` is rejected), no `output_attentions=True`, and -- directly
relevant to this repo -- **it cannot be combined with quantization**. Only Stem
is documented and stable; the other six are experimental or reference-only.

### The seven algorithms, and where each one's sparsity pattern comes from

The decisive axis for a static-graph port is not "which paper" but **is the
kept-block set a function of position only, or of the activations?**

| Algorithm | Selection rule | Pattern is... | Real kernel needs |
|---|---|---|---|
| **Stem** | Triton strided group-GEMM block scoring + value-norm bonus, then per-layer top-k of blocks | **data-dependent** | Triton + CUDA; `block-sparse-attn` (or the non-public `hpc` C++ ext) |
| **MInference / `a_shape`** | sink (first `n_init`) + sliding window (last `n_local`), StreamingLLM-style | **positional only** | Triton + CUDA (`streaming_kernel`) |
| **MInference / `tri_shape`** | `a_shape` for all but the last `n_last` queries; those attend full-causal | **positional only** | Triton + CUDA (`streaming_kernel`) |
| **MInference / `minference`** | per-head top-k vertical columns + top-k slash diagonals, estimated from the last-64 queries' softmax | **data-dependent** | Triton + a JIT-built `convert_vertical_slash_indexes` CUDA ext (needs a runtime CUDA toolchain) |
| **FlexPrefill** | per head, smallest key-block set whose cumulative attention mass covers `gamma` | **data-dependent** | Triton + `flash_attn`; head_dim in {16,32,64,128} |
| **XAttention** | per head, top-p over antidiagonal-strided block scores | **data-dependent** | `block_sparse_attn`; block_size==128, batch==1 |
| **FlashPrefill** | per query-block, keep blocks with `s[I,J] >= alpha * max_J s[I,J]` (+ fixed sink 256 / window 512) | **data-dependent** (plus a fixed part) | `block_sparse_attn`; dense path needs `flash_attn` |
| **VecAttention** | per query-block, MinP on logits: keep key `j` iff `qk[j] - max_j qk >= log(threshold)` (+ sink + local band) | **data-dependent** (plus a fixed part) | `vllm_flash_attn.sparse_attn_func` from an unreleased vLLM-FA fork |
| **CoSA** | per query-block, top-p over a stride-subsampled `Q@K^T` proxy, emitted priority-ordered | **data-dependent** | Triton + the non-public internal `hpc` extension |

Six of the seven algorithms -- Stem, FlexPrefill, XAttention, FlashPrefill,
VecAttention, CoSA -- plus one of MInference's own three variants, take a
top-k / top-p / threshold decision **on the actual `Q@K^T` values of the request
being served**. ONNX can express that computation (`TopK` exists), but
expressing it does not help: you would compute the scores in order to decide which scores
not to compute. The selection estimate itself costs a fraction of a dense
attention only because the kernel then skips most of the real one -- and
`MatMul`/`Softmax` in an ONNX graph have no "skip these blocks" input.

So the only candidates that clear the first filter at all are **A-Shape and
Tri-Shape**: their masks are pure functions of `(q_len, k_len, n_init,
n_local, n_last)` and can be materialised as a constant at graph-rewrite time.
(Verified in AngelSlim's own `minference/reference.py`: `a_shape_attention`
builds `keep = (sink | window) & causal` from `torch.arange` position indices
only, and `tri_shape_attention` splits the query range at `-n_last` and gives
the tail a plain causal mask. No tensor values enter either.)

## The crux: does a static rewrite buy anything on ONNX Runtime?

A fixed mask *is* expressible in ONNX. The question is whether any op onnxsim
can emit will then actually do less work. Every ORT attention op in 1.29.0 was
checked (`onnxruntime.capi._pybind_state.get_all_operator_schema()`), and the
two that advertise sparsity were read at source level and benchmarked.

### What the ops actually do

| Op | Sparsity knob | Does it skip work? |
|---|---|---|
| `ai.onnx::Attention` (opset 23/24) | none -- only a dense `attn_mask` input, `is_causal`, `softcap` | No such knob exists |
| `com.microsoft::MultiHeadAttention` | dense `attention_bias` only | No |
| `com.microsoft::GroupQueryAttention` | `local_window_size` (Mistral-style sliding window) | **No, on the CPU EP.** `gqa_attention_base.h` runs the full `GemmEx(sequence_length, total_seqlen, head_size)` `QK^T`, *then* zeroes the pre-window region, then a full dense `probs@V`. The window narrows the softmax, not the GEMMs. |
| `com.microsoft::LongformerAttention` | `window` + global-attention flags | Window semantics, but a fixed Longformer shape (symmetric two-sided window, global flags, and a *fused* QKV projection) -- not an A-Shape/Tri-Shape carrier. CUDA-only: session creation on the CPU EP fails with `NOT_IMPLEMENTED: Could not find an implementation for LongformerAttention(1)` |
| `com.microsoft::SparseAttention` | **CSR block mask** (`block_row_indices` / `block_col_indices`) -- the Phi-3-small block-sparse op | **CPU EP: no. CUDA EP: yes.** See below. |

`com.microsoft::SparseAttention` is the one op in the whole surface whose
*interface* is exactly what a static A-Shape port would want: a per-layout
block mask in CSR form, supplied as an ordinary tensor, which is precisely
what a graph rewrite can bake in. Its two implementations diverge completely:

- **CUDA EP** (`contrib_ops/cuda/sparse/sparse_attention_v1/sparse_attention_triton.py`)
  genuinely skips. The Triton kernel's inner loop is literally
  `for col_idx_idx in range(start_l, end_l)` over the CSR non-zeros for the
  current query block -- masked blocks are never loaded and never multiplied.
  It is gated on: compute capability 7.5 / 8.0 / 8.6 / 8.9 / 9.0,
  **`head_size == 128`**, **`sparse_block_size == 64`**, fp16/bf16 only, and
  past/present KV sharing one buffer.
- **CPU EP** (`contrib_ops/cpu/sparse/sparse_attention_base.h`) does **not**
  skip. It runs the full dense `GemmEx(sequence_length, total_seq_len,
  head_size)` `QK^T`, then expands the CSR mask into a per-row `int32` array
  and writes `std::numeric_limits<T>::lowest()` into the masked score
  positions, then a full dense `probs @ V`. It is the "compute dense, then add
  `-inf`" form with extra bookkeeping on top -- and it additionally requires
  IOBinding with past/present bound to the same `OrtValue` (it hard-fails
  otherwise).

### Measured

Reproducible via `bench/sparse_attention_probe.py` (`python
bench/sparse_attention_probe.py all --seq 4096 --profile`), which is the
committed form of both probes below. Numbers here are from this repo's sandbox:
Intel Xeon @ 2.80GHz, 4 vCPU, `onnxruntime==1.29.0` CPU EP,
`intra_op_num_threads=4`, fp32, batch=1, 8 heads, head_dim=64, seq_len=4096;
A-Shape configured as `n_init=128, n_local=512`. Medians of repeated runs after
a warm-up; ranges are across three full runs of the script (a 4-vCPU sandbox is
a noisy place to measure, which is why the ranges are wide and why the per-node
profile below matters more than the wall-clock).

**1. `com.microsoft::SparseAttention`, dense-causal CSR vs A-Shape CSR**
(`sparse_block_size=64`: 2080 causal blocks vs 595 A-Shape blocks, 28.6%):

| block mask | median | vs causal |
|---|---|---|
| full causal (2080 blocks) | 177-190 ms | 1.00x |
| A-Shape (595 blocks, 28.6%) | 283-352 ms | **0.50-0.67x, i.e. 1.5-2.0x slower** |

A 3.5x reduction in non-zero blocks buys nothing, because the GEMM is dense
either way; the sparse layout only *adds* the per-row mask expansion and
`lowest()`-store loop, which the dense layout skips entirely (the kernel's
`has_sparse` flag is false exactly when a row's non-zero count equals `row+1`).
Output deltas of 0.33-0.52 max-abs confirm the mask is being applied -- this is
not a no-op that failed to take effect.

**2. A baked `-inf` mask on the decomposed subgraph and on `MultiHeadAttention`**
(the form onnxsim could actually emit today; A-Shape keeps 28.8% of the causal
entries at token granularity). The script runs the causal and A-Shape sessions
**interleaved**, so ordering and thermal drift hit both equally:

| graph | causal | A-Shape | ratio |
|---|---|---|---|
| `MatMul`/`Mul`/`Add`/`Softmax`/`MatMul` (ai.onnx) | 424-436 ms | 535-584 ms | **1.24-1.38x slower** |
| `com.microsoft::MultiHeadAttention` + `attention_bias` | 412-436 ms | 637-770 ms | **1.55-1.77x slower** |

Standalone (one session at a time, not interleaved) the same comparison gives
1.29x and 1.18x; substituting a finite `-1e30` for `-inf` gives 1.30x, so the
slowdown is not an infinity slow-path artifact. **In no configuration, on any
op, was the A-Shape mask ever faster.**

Per-node profiling (`--profile`; min over 7 runs, which drops warm-up noise)
isolates where the difference is and, more importantly, where it is not:

| node | causal | A-Shape |
|---|---|---|
| `Add` (mask) | 29.9 ms | 32.4 ms |
| `FusedMatMul` (`QK^T`) | 40.3 ms | 38.4 ms |
| `MatMul` (`probs@V`) | 39.5 ms | 40.5 ms |
| `Softmax` | 103.9 ms | **165.0 ms** |

The two matmuls -- the `O(S^2 d)` work that sparse attention exists to
eliminate -- cost **the same**, to within noise, whether 100% or 28.8% of the
mask is kept. That is the whole result. Adding a mask does not remove arithmetic
from a dense GEMM; it only changes the numbers that go into it.

## Per-algorithm verdict

| Algorithm | Portable to onnxsim? | Why not |
|---|---|---|
| Stem | No | Data-dependent per-layer block top-k, computed from a Triton scoring GEMM at run time. Even AngelSlim's own `allow_pseudo_sparse` fallback (dense masked attention) still needs Triton for scoring and is explicitly documented as "a correctness/debug path, not a performance path". |
| MInference `a_shape` | Pattern yes, benefit **no** | Mask is static and expressible, but every ORT op onnxsim can emit computes dense attention behind it -- measured 1.18-2.0x slower, never once faster, see above. |
| MInference `tri_shape` | Pattern yes, benefit **no** | Same as `a_shape`. Worse in graph terms: the query range splits at `-n_last`, so a faithful rewrite is two attention subgraphs plus a `Concat` -- more nodes, still dense. |
| MInference `minference` | No | Per-head top-k over columns *and* diagonals of a live softmax; then a CUDA index extension JIT-compiled at run time. |
| FlexPrefill | No | Per-head gamma-coverage over live block scores. |
| XAttention | No | Per-head top-p over live strided block scores. |
| FlashPrefill | No | Per-query-block `alpha * max` threshold over live block scores. Its *fixed* part (sink 256 + window 512) is A-Shape and inherits A-Shape's verdict. |
| VecAttention | No | Per-head MinP over live logits; fast path needs an unreleased vLLM-FlashAttention fork. Its fixed part is again sink + local band. |
| CoSA | No | Per-query-block top-p over a live Triton proxy; attention kernel is the non-public internal `hpc` extension, with no fallback at all. |

## Why "ship the A-Shape mask anyway" is the wrong call

A mask-inserting pass is easy to write. It would be dishonest in two different
directions at once, so it is not being written:

- **It is not acceleration.** Calling it "sparse attention" would import the
  name of a technique whose entire content is a speedup, while delivering a
  measured *slowdown* on the runtime this repo targets. The measurements above
  exist so that this is a fact, not an opinion.
- **It is not a quantization/compression pass either.** It changes what the
  model computes (A-Shape drops real attention edges -- the max absolute output
  difference vs dense in the `SparseAttention` benchmark above was 0.33-0.52,
  i.e. substantial, not numerical noise), while making the model neither smaller nor
  faster. Every other onnxsim pass that perturbs outputs buys something
  measurable for it: bits, bytes, or latency. This one would buy nothing.
  A-Shape/StreamingLLM masking *is* a legitimate long-context technique -- its
  payoff is bounded memory and length extrapolation in an autoregressive
  *serving loop* with a rolling KV cache, which is a runtime policy, not a
  property of a static graph.

Note also AngelSlim's own constraint that sparse attention **cannot be combined
with quantization** -- which is essentially all of onnxsim's surface.

For contrast, `onnxsim/tensorrt_sparsity.py` is the repo's existing precedent
for a pass whose payoff is hardware-conditional and unmeasurable on a CPU
sandbox, and it ships. The difference is that it is **value-preserving** (it
canonicalises `MatMul` to `Gemm`, computing exactly the same thing) and its
only job is to make an *already-produced* 2:4 pattern *eligible* for a kernel
that really does skip work. A static A-Shape rewrite fails both legs: it
changes the outputs, and there is no ORT-emittable op that becomes eligible for
anything as a result.

## What would have to change for a port to become worthwhile

Recorded precisely, so it is recognisable if it happens:

1. **A CUDA-EP `com.microsoft::SparseAttention` emission path.** This is the
   only concrete route that exists today. It would mean a pass that rewrites a
   matched GQA/KV-cache attention block into `SparseAttention` with baked
   A-Shape CSR `block_row_indices`/`block_col_indices`. onnxsim already has
   partial `SparseAttention` awareness -- `onnxsim/structured_pruning_entry.cpp`
   (`MatchSparseAttentionProducer`, `FindSparseAttentionChains`) head-prunes
   through the op -- so the matching machinery is not the hard part. The
   blockers are that it would be *untestable in CI* (the pass's only exercisable
   path, the CPU EP, is the slow one), and that the CUDA kernel's preconditions
   are narrow: sm 7.5-9.0, `head_size == 128`, `sparse_block_size == 64`,
   fp16/bf16, past/present sharing one buffer. It also still changes outputs,
   so it needs an accuracy story that a pure eligibility pass like
   `tensorrt_sparsity.py` does not.
2. **A CPU-EP block-sparse kernel in ORT.** If `sparse_attention_base.h` ever
   grew a real block-skipping path (iterate the CSR non-zeros and GEMM only
   those tiles, FlashAttention-style online softmax) instead of dense-GEMM +
   `lowest()`, the measurement in this note flips sign and A-Shape/Tri-Shape
   become a genuinely useful static rewrite. Worth re-checking on ORT
   upgrades: `python bench/sparse_attention_probe.py blockmask` is exactly that
   check -- one `SparseAttention` node run twice, causal CSR vs A-Shape CSR,
   same inputs. A number above 1.0x there means this note needs revisiting.
3. **A standard ONNX block-sparsity attribute.** `ai.onnx::Attention` (opset
   23/24) has no window or block-mask knob at all -- only a dense `attn_mask`.
   If ONNX standardised one and ORT implemented it on CPU, this stops being a
   contrib-op-and-GPU-only question.
4. **Data-dependent selection is a different problem entirely** and is not on
   this list. Expressing top-k-over-live-scores in ONNX is possible and
   pointless: the graph would compute the full score matrix to decide which
   parts of the full score matrix to skip, and then compute it again densely.
   No plausible change to ONNX or onnxsim fixes that; it needs a fused kernel,
   which is a runtime concern, not a graph-rewriting one. Stem, `minference`,
   FlexPrefill, XAttention, FlashPrefill, VecAttention and CoSA are all in this
   category.

## Explicitly out of scope (recorded so it isn't re-litigated)

- **A mask-inserting pass named after sparse attention.** See above: no
  speedup, no size reduction, changed outputs. If someone later wants
  StreamingLLM-style A-Shape masking as an explicit *behaviour* change, it
  should be proposed on its own accuracy/long-context merits, under its own
  name, with the "this is not faster" fact stated up front -- not as a port of
  AngelSlim's Prefill accelerator.
- **Vendoring or reimplementing Triton/CUDA kernels.** onnxsim is stateless
  ONNX protobuf rewriting targeting ONNX Runtime; it has no kernel-authoring
  layer and adding one to host block-sparse attention would be a second,
  fundamentally different code path -- the same reasoning that keeps QAT out
  (`docs/nncf-comparison-future-work.md`, "Explicitly out of scope").
- **Serving-loop techniques generally.** Sparse *prefill* is one instance of a
  broader class (prefix caching, chunked prefill, speculative decoding) whose
  benefit lives in a request-serving loop, not in the model file. onnxsim
  rewrites the model file.

## Sources

Primary sources actually read for this note:

- AngelSlim, `docs/source/features/sparse_attention/index.md` -- the module's
  scope statement, the seven-algorithm status table, and the shared constraint
  list (including "cannot be combined with quantization").
- AngelSlim, `docs/source/features/sparse_attention/stem.md` -- Stem's
  block-scoring / top-k-schedule / block-sparse-attention pipeline and its
  Triton/`block-sparse-attn`/`hpc` requirements.
- AngelSlim, `angelslim/compressor/sparsity/registry.py`,
  `algorithms/minference/{algorithm,prefill,reference}.py`,
  `algorithms/minference/kernels/__init__.py` -- the exact selection rule for
  each MInference variant and the kernel-availability gate.
- AngelSlim, `algorithms/{flexprefill,xattention,flashprefill,vecattention,cosa}/algorithm.py`
  -- each algorithm's selection rule and hard requirements.
- ONNX Runtime 1.29.0, `onnxruntime/contrib_ops/cpu/sparse/sparse_attention_base.h`
  -- the CPU `SparseAttention` kernel's dense-GEMM-then-`lowest()` structure.
- ONNX Runtime 1.29.0, `onnxruntime/contrib_ops/cuda/sparse/sparse_attention.cc`
  and `sparse_attention_v1/sparse_attention_triton.py` -- the CUDA kernel's
  CSR-non-zero inner loop and its sm / head_size / block_size gates.
- ONNX Runtime 1.29.0, `onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h`
  -- `local_window_size` masking after a full dense GEMM.
- ONNX Runtime 1.29.0 op schemas, enumerated in-process via
  `onnxruntime.capi._pybind_state.get_all_operator_schema()`; `ai.onnx::Attention`
  via `onnx.defs.get_schema("Attention")` (onnx 1.22.0).
- Benchmarks: `bench/sparse_attention_probe.py` in this repo, run in this
  repo's sandbox as described in "Measured" above.

Related reading in this repo: `bench/sparse_attention_probe.py` (the probe
behind every number above), `onnxsim/tensorrt_sparsity.py` (the
hardware-conditional-but-value-preserving sparsity precedent),
`onnxsim/attention_quantization.py` (the decomposed-attention subgraph shape a
mask rewrite would have targeted), and `docs/nncf-comparison-future-work.md`
(the research-note format this file follows).
