# Optimization notes: InstantHMR v1.1.0 ONNX graphs

These notes come from running onnxsim 0.8.1 (built from this repo at `0d06e22`) on the
InstantHMR v1.1.0 ONNX graphs, and from checking which graph patterns remain after
simplification. Conclusion first: onnxsim already does nearly all of the useful work on the
fit graphs. The remaining opportunities are small, or they belong in the exporter or in a
runtime rather than in onnxsim.

## Source and scope

- Weights: Hugging Face `momolesang/InstantHMR` (the v1.1.0 GitHub release assets were not
  reachable from the sandbox). Files: `instanthmr-{medium,nano}.onnx`,
  `instanthmr-{medium,nano}-fit{1,2}.onnx`, `mhr-body.onnx`.
- Exporter: `tools/export_onnx.py` in the InstantHMR repo at tag `v1.1.0`.
- Inputs in the tests were random, so the results cover graph structure and numerical
  equivalence only, not pose accuracy.

## Results

| Model | Nodes before | Nodes after onnxsim | Max output diff |
|---|---|---|---|
| `instanthmr-medium` | 2629 | 1011 | 6.6e-7 |
| `instanthmr-nano` | 2626 | 1008 | 7.2e-7 |
| `instanthmr-medium-fit1` | 2286 | 2282 | 1.5e-5 (fp16) |
| `instanthmr-medium-fit2` | 2308 | 2304 | 6.1e-5 (fp16) |
| `instanthmr-nano-fit2` | 2305 | 2301 | 2.4e-4 (fp16) |
| `mhr-body` | 1169 | 699 | 0 |

### Why the fit graphs shrink less

`export_fit` calls `onnxsim.simplify` before it saves a fit graph. The plain network graph
gets no such pass. The fit graphs therefore already have the constant and shape bookkeeping
removed. For `medium` that bookkeeping is 972 `Constant` and 86 `Shape` nodes; the fit graphs
have none. Re-running onnxsim on `medium-fit1` removes only 4 nodes.

## Patterns checked

Each count below comes from `medium-fit1` unless noted.

| Pattern | Count | Verdict |
|---|---|---|
| Cast→Cast | 0 | nothing to do |
| Unsqueeze→Squeeze, adjacent | 0 | nothing to do |
| Reshape→Reshape, Transpose→Transpose | 0 | nothing to do |
| Constant-only subgraphs | 0 | already folded |
| Identity-like ops (`x*1`, `x+0`, `Identity`) | 0 | nothing to do |
| Dead nodes | 0 | nothing to do |
| Duplicate nodes (CSE candidates) | 3 | minor |
| Elementwise ops with a constant operand | 278 of 909 | mostly from the polynomial in `atan2_poly`; small gain |
| Elementwise→elementwise, single-use edge | 548 | fusion candidate for a runtime, not for onnxsim |
| Longest elementwise-only chain | 13 | same as above |
| Gather→Gather, adjacent | 28 | see below; not worth a pass |
| Slice→Slice, adjacent | 19 | see below; not worth a pass |
| Gather of Concat, constant scalar index | 0 | no matching pattern |

### Gather and Slice

- **Slice→Slice (19 pairs):** all indices are constant. In every pair the inner Slice's output
  also has other consumers, so merging the two Slices adds a node and keeps the old one.
  Node count does not drop.
- **Gather→Gather (28 pairs):** 24 pairs have a shared inner Gather, so they can't merge. The
  other 4 are single-use pairs of scalar Gathers on two different original axes, such as
  `x[:, 0][:, 0]`. A single ONNX `Gather` cannot select along two axes, and the one-node
  replacement is `GatherND`, which may be slower. The gain would be 4 nodes.
- These Gathers and Slices come mostly from the `cont_head` Concat chains. They move real
  data, so they are not redundant.

### Elementwise chains

`atan2_poly` (in `tools/export_onnx.py`) replaces `torch.atan2` with a polynomial built from
basic ops, because onnxruntime's CUDA provider has no Atan kernel. The polynomial's constants
sit in separate Mul and Add nodes, so no `x*c1*c2` constant merge applies. Folding the
constants into one coefficient per Horner step would save a few dozen nodes at most.

## Where real gains would come from

1. **Exporter, `atan2_poly`:** a native `Atan` op would remove most of the polynomial. The
   source comment says the change would keep the CUDA provider from capturing a CUDA graph,
   so this is a trade-off, not a free change. Measure CUDA-graph capture before changing it.
2. **Runtime fusion:** ONNX Runtime and TensorRT already fuse elementwise chains at load
   time. Node counts in the file overstate the kernel count at runtime. Profile on the target
   runtime before optimizing the graph further.
3. **onnxsim:** no pass for the patterns above is worth writing on current evidence. The
   Gather and Slice patterns do not reduce node count, and the constant-merge pattern has too
   few matches to justify it.

## Reproducing

The analysis scripts were written in a scratch directory outside this repo, so they are not
committed. To reproduce the counts, load each graph with `onnx.load`, count nodes per
`op_type`, and check each pattern on the producer and consumer of each edge. Compare outputs
before and after `onnxsim.simplify` with `onnxruntime` on the same random inputs.
