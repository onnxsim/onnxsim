# Android demo plan: what a nerfstudio NeRF needs to run on a phone

Status: **feasible with two unresolved blockers, neither of which is in onnxsim.**

This is the working plan for turning the exported nerfacto graph into an on-device
demo on the QNN HTP path the rest of `scripts/android/` uses. It records what is
already true, what was measured for this PR, and what still has to be built.

## Where it stands

| Stage | Status |
|---|---|
| nerfstudio `NerfactoField` -> ONNX (opset 18, `dynamo=True`) | **done**, verified in CI |
| `onnxsim.simplify` preserves it | **done**, bit-exact (`density max_abs_diff = 0.0`) |
| ORT runs the simplified graph | **done**, 5.96e-08 on rgb |
| `BitwiseXor` -> `And`/`Or`/`Sub` rewrite | **done in this PR**, bit-exact |
| int8 quantization (`full_qdq`) | should work -- `Gather` is already a shared-qparam op and indices are never quantized |
| QNN HTP accepts the rewritten graph | **unknown, never probed** |
| HTP `Gather` at nerfstudio's table size | **known to fail at defaults**, see below |

## The one thing already settled

The Instant-NGP space hash is `torch.bitwise_xor`, and `BitwiseXor` has no
kernel on the backends that matter:

* **QNN HTP** -- no known kernel; never probed. This is the target.
* **TFLite** -- has a `BITWISE_XOR` *builtin* (code 160), but builtins are
  unavailable to the int8/uint8 delegate graphs this repo builds, and TFLite has
  no integer `BitwiseAnd`/`BitwiseOr` (only bool `LOGICAL_AND`/`LOGICAL_OR`).
* **Core ML** -- MIL has **no bitwise op at all**, so `a ^ b == (a | b) - (a & b)`
  would just move the failure to `BitwiseOr`. Core ML is deliberately not wired
  up to call the rewrite. Verified against `SSAOpRegistry.core_ops`.

So this PR adds `onnxsim/bitwise_decompose.py`, which rewrites the op exactly
(for every integer width, no overflow -- `a | b >= a & b` bitwise, so the
subtraction never borrows) and is verified bit-exact against the original graph
on the real exported nerfacto model:

```
exported:  220 nodes, BitwiseXor=16
rewritten: 252 nodes, BitwiseXor=0, BitwiseOr=16, BitwiseAnd=16
onnx.checker: OK
density/rgb bit-exact vs the original graph:  True / True
after onnxsim.simplify (250 nodes): still bit-exact
```

It is exposed as a function rather than a default `simplify()` pass, because it
is a legalisation rewrite for a target that needs it. The QNN probe has to come
first: until the HTP is known to accept `BitwiseOr`/`BitwiseAnd`, the rewrite is
untested against the only thing that matters.

## Blocker 1: the hash tables are far too big for the HTP

`scripts/android/vision_models/fastbev/README.md` records the measured ceiling:
fp16 `Gather` on the HTP finalizes only for a few-MB table, and a `67585 x 64`
table fails even for 10k indices. Its 41 MB uint8 volume gather costs 30 ms+.

nerfacto's defaults, per level-set (`log2_hashmap_size` levels x features_per_level
rows of `geo_feat_dim + levels*features_per_level`):

| config | fp16 | uint8 |
|---|---|---|
| defaults (`log2_hashmap_size=17`) | **130 MB** | **65 MB** |
| small (`log2_hashmap_size=14`) | 4.2 MB | 2.1 MB |

Defaults are ~20x past the measured limit. int8 QDQ roughly halves it, which
still leaves 65 MB. The repo's documented answers, both with prior art:

* int8/uint8 tables, which do finalize on the HTP (fastbev's gathers are uint8),
* or move the gather to the DSP. `fastbev`'s `fbgather` does exactly this --
  4 uint8 tables + int32 LUTs in one FastRPC call, **8.1 ms** with 4 threads,
  byte-exact under `qemu-hexagon-static`.

A demo therefore has to shrink `log2_hashmap_size` hard (14 or lower), or accept
a DSP gather. Either way it is a smaller model than nerfacto's defaults, and
should be labelled as such rather than presented as "nerfacto on a phone".

## Blocker 2: nothing has been probed on real hardware

No measurement exists for any of this on an HTP:

* does the rewritten graph finalize at all,
* what `BitwiseOr`/`BitwiseAnd` cost per sample,
* whether int8 `Gather` at ~2 MB holds its throughput,
* what the end-to-end frame time is.

`scripts/android/htp_exploration/qnn_shell/probe_gather.py` is the existing
tool for the `Gather` half of that; a small `probe_bitwise.cpp` alongside it
would answer the rest. Until then the honest statement is that this is a plan,
not an estimate.

## The realistic demo

Follow the shape of `scripts/android/vision_models/superres/` (XLSR: 5.9 ms
int8 at 1080p, uint8 NHWC I/O, one strict HTP session, already in the demo app
at 54 FPS), with `nss/` as the precedent for anything graphics-shaped (a 148 K
param CNN on the HTP at 2.70 ms, everything else hand-written OpenCL on the
Adreno).

Concretely: a reduced nerfacto (`log2_hashmap_size<=14`, `geo_feat_dim` small),
int8-quantized with `full_qdq` + `quantized_io`, run through
`qnn_run_multi` under the same `QNN_PERF=burst` + phone lock every other number
in those READMEs depends on. Expected shape of the work:

1. `probe_bitwise.cpp` -- confirm `BitwiseOr`/`BitwiseAnd` finalize on the HTP
   and time them. **This is the gate; nothing else should start first.**
2. Probe `Gather` at the chosen table size, int8, with `probe_gather.py`.
3. Only then wire up `phone.sh` and measure end to end.
4. If the HTP refuses the bitwise ops, the fallback is the `fastbev` pattern:
   a hand-written HVX kernel behind FastRPC. There is direct prior art for the
   move (`scatter_rewrite.py` rewrote `ScatterElements` -> `ScatterND` for the
   same reason, bit-exact and 1.6x faster).

## What is *not* blocked

Two routes need none of the above and could be demoed today if a NeRF-shaped
graph is wanted before the HTP work:

* **Drop the hash grid.** A NeRF with plain sine/cosine positional encoding and
  an MLP is `MatMul`/`Gemm`/`Sin`/`Cos`/`Softmax` -- every one already in the
  TFLite and Core ML tables and native on the HTP. `tests/test_nerf_onnx_integration.py`
  builds exactly this shape.
* **Precompute the hash indices** host-side (the `fastbev` LUT pattern, or bake
  them as initializers for a fixed set of sample positions). Exact, but it pins
  the demo to fixed viewpoints.

Neither is nerfstudio's nerfacto; both are legitimate demos of the export path
and neither needs a new pass.

## Out of scope

`splatfacto` is not reachable: `gsplat` is fused CUDA with no ONNX representation
at all. And nerfstudio's `implementation="tcnn"` is a CUDA-only fused kernel, so
the `torch` path is the only one that ports.