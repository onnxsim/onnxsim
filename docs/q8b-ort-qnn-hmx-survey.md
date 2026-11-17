# ORT-QNN HTP/HMX survey for Radxa Dragon Q8B

Survey date: 2026-11-17. Target: SC8280XP, Linux ARM64, Hexagon v68.
The board's verified tinygrad DSP test does not establish HMX execution.
Its installed ORT 1.32.0 wheel exposes CPU and Azure EPs, not QNN; the
Hexagon SDK 6.4.0.2 installation does not install QAIRT/QNN.

## Release scope

The standalone `onnxruntime/onnxruntime-qnn` repository is the relevant
upstream for QNN EP 2.x. Its latest published release is **2.8.0**, dated
2026-11-12, with QAIRT **2.52.0**, built against ORT **1.29.0** and declaring
ORT compatibility **>=1.24.1**. Earlier releases examined: 2.7.0
(2026-10-13), 2.6.0 (2026-09-10), and 2.5.0 (2026-08-20).
These QNN EP version numbers are distinct from both ORT and Hexagon SDK
versions. Sources: [published releases](https://github.com/onnxruntime/onnxruntime-qnn/releases),
[2.8.0 release notes](https://github.com/onnxruntime/onnxruntime-qnn/blob/v2.8.0/docs/release-notes.md).

Also inspected main at `1b73a511fbac87bcda0542607f32a64a89e3e23e`.
Changes present on main are not assumed to be in a published wheel.

## Improvements that matter for matrix-heavy language models

| Change | Evidence and availability | Q8B relevance |
|---|---|---|
| Native block-quantized Conv/MatMulNBits | [PR #604](https://github.com/onnxruntime/onnxruntime-qnn/pull/604), released in 2.7.0, QAIRT >=2.51 | Not the native path on v68: architecture guard is v81+. |
| MatMul LUT selection | [PR #807](https://github.com/onnxruntime/onnxruntime-qnn/pull/807), released in 2.7.0 | Current source enables this only on Windows with QAIRT >=2.51; unavailable through this option on Q8B Linux. |
| LPBQ Conv lowering | [PR #483](https://github.com/onnxruntime/onnxruntime-qnn/pull/483), released in 2.6.0 | A distinct representation/path from native BQ; backend validation is still required. |
| Native RoPE | [PR #659](https://github.com/onnxruntime/onnxruntime-qnn/pull/659), released in 2.7.0 | Can reduce graph decomposition; no Q8B speed measurement established. |
| Graph splitting for preparation | [PR #645](https://github.com/onnxruntime/onnxruntime-qnn/pull/645), released in 2.7.0, QAIRT >=2.49 | Startup optimization, not faster token decoding. |
| Multiple NPU cores | [PR #790](https://github.com/onnxruntime/onnxruntime-qnn/pull/790), released in 2.8.0 | Requests discovered hardware cores; cannot create extra cores on v68. |
| Native MatMul instead of FullyConnected | [PR #821](https://github.com/onnxruntime/onnxruntime-qnn/pull/821), merged 2026-10-27; found on inspected main, absent from v2.8.0's `ModelSettings` | Potential startup/host-memory improvement; not a released 2.8.0 feature in the examined tag. |

Native BQ avoids the fallback's INT16-to-FP16 activation conversion and
FP16-to-INT16 output conversion. The implementation checks v81 or newer,
unsigned 16-bit activations, symmetric 4-bit weights, blocks of 32/64/128,
and output channels divisible by 32. This is a specific eligibility rule,
not a general promise that any INT4 model uses HMX.
[Tagged implementation](https://github.com/onnxruntime/onnxruntime-qnn/blob/v2.8.0/onnxruntime/core/providers/qnn/builder/qnn_bq_utils.cc).

MatMul LUT is passed as a graph-finalization setting to QNN, with an opt-out.
The public ORT code does not implement its matrix instructions. The
Windows/SDK gate is explicit in
[`qnn_def.h`](https://github.com/onnxruntime/onnxruntime-qnn/blob/v2.8.0/onnxruntime/core/providers/qnn/builder/qnn_def.h).
Do not extrapolate this optimization to Linux merely from the PR title.

Graph splitting partitions preparation work. The PR reports about 30%
less preparation time using one thread and 78% using eight, but these are
upstream examples, not Q8B results. Loading an existing compiled context
gets no benefit from this preparation step.
[PR #645](https://github.com/onnxruntime/onnxruntime-qnn/pull/645).

`use_native_matmul=1` on inspected main keeps static ONNX weights in their
original orientation, avoiding host transposes and some rank flattening.
It defaults off and leaves several block-quantized/fused paths unchanged.
This targets session initialization and peak memory; a decode speedup must
be measured separately. [PR #821](https://github.com/onnxruntime/onnxruntime-qnn/pull/821).

## Memory, deployment and correctness

Recent configuration surfaces worth evaluating with an actual QNN runtime:

- `htp_share_resource_optimization` shares VTCM backup storage across sessions.
- `htp_reused_io_limit_mb` describes the peak registered I/O budget for DSP
  placement estimates. It is a hint, not a bound enforced on allocations.
- `prepare_and_load` separates preparation/loading for larger graphs.
- Native RoPE, MatMul/Add/Relu fusion and operator coverage can reduce graph
  boundaries independently of matrix-kernel speed.

Windows ARM64 weight file mapping and v81+ extended uDMA should not be
counted as Q8B Linux improvements. The current configuration guide also
documents detailed/optrace profiling and disabling CPU fallback, both
needed to distinguish NPU execution from a successful CPU fallback.
[Configuration guide](https://github.com/onnxruntime/onnxruntime-qnn/blob/1b73a511fbac87bcda0542607f32a64a89e3e23e/docs/execution_providers/QNN-ExecutionProvider.md).

Main includes explicit v68 restrictions: quantized Conv with 16-bit
activations cannot use 16-bit weights, and float SpaceToDepth fusion is
rejected on v68. A successful graph conversion on a newer architecture
therefore does not establish compatibility here.
[Conv guard](https://github.com/onnxruntime/onnxruntime-qnn/blob/1b73a511fbac87bcda0542607f32a64a89e3e23e/onnxruntime/core/providers/qnn/builder/opbuilder/conv_op_builder.cc),
[SpaceToDepth guard](https://github.com/onnxruntime/onnxruntime-qnn/blob/1b73a511fbac87bcda0542607f32a64a89e3e23e/onnxruntime/core/providers/qnn/builder/qnn_node_group/slice_concat_spacetodepth_fusion.cc).

## What to try on this board

1. Establish a supported Linux ARM64 QAIRT/QNN installation with a matching
   v68 backend/skel and firmware before changing model quantization.
2. Run a small quantized MatMul with CPU fallback disabled. Save the QNN
   graph, placement logs and detailed profile; success alone does not prove
   which internal HMX kernel QNN selected.
3. Check A8W8/A16W8 support before pursuing newer BQ paths. Compare identical
   inputs and output tolerances, recording conversion and RPC costs.
4. Measure model loading, prompt prefill, and cached decoding separately.
   Weight-sharing or preparation improvements must not be presented as
   token-throughput improvements.
5. For tinygrad, treat these as scheduling/representation ideas to evaluate,
   not kernels that can be copied from the ORT wrapper. The inspected public
   code configures QNN's HTP backend; it does not expose the backend's HMX
   microkernel implementation.

No ORT-QNN performance measurements were run in this survey.
