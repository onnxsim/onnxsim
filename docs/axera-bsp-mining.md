# Axera's public BSPs: what they add to the MCode decode

Two public AXERA-TECH repositories had not been examined by this project:
`ax650n_bsp_sdk` (AX650N) and `ax620e_bsp_sdk` (AX620Q/AX630C). Earlier work mined
the card's own `ax_npu.ko` and `libax_interpreter.so` (`scripts/axera/README.md`,
firmware section) and the AX620A headers in `sipeed/axpi_bsp_sdk`. This note
covers the rest: the BSPs' NPU host libraries, and a corpus of 152 MatMul models
that Axera compiled with its own toolchain.

**Summary.** The host libraries treat MCode as opaque bytes; nothing in them
decodes or patches it. The MatMul corpus is more useful. It validates this
project's codec against a second compiler, it shows a multi-core segment layout,
and its int8/int16 pairs isolate the TENG queue better than any sweep this
project has built. It does not yield a TENG generator.

## The host libraries: no MCode decoding

Sources: `msp/out/lib/libax_engine.a` (3.5 MB, 106 members) and
`libax_interpreter.a` from the AX650N BSP; `libax_engine.a`, `libax_engine_tiny.a`
and `libax_interpreter.a` from the AX620E BSP. They were read with `ar`, `nm`,
`strings` and `pahole` only. Nothing is encrypted or license-gated here.

- **`libax_engine.a` ships with DWARF debug info** (built from
  `DailyBuild_ax650_backend_mp_v1.45.0`). Its 82 struct types are all host-side:
  graph and scheduler objects, nanopb ONNX messages, and the axmodel `extra_data`
  schema (`pulsar2ext.pb`). The runtime keeps a neu model as a pair of names
  (`neu_pair { name, key }`) and a batch/stride table
  (`neutron_param { count, params, weights }`). It never looks inside the blob.
- **The `extra_data` metadata schema is now named field by field**:
  `AxmodelExtra { tensor_extras, version, subgraphs, build_info, hardware_type }`,
  `Subgraph { name, type: ONNX|NPUDotNeu, dotneus }`,
  `DotNeuMeta { neu_key, batch, extra_inputs }`,
  `DotNeuInput { name, type: Const|Dynamic|DynParam, const_data_key,
  dyn_param_expr_key }`, `TensorExtra { name, layout, color_space, enable_stride,
  group_index }` and `BuildInfo { static_batch_sizes, max_dynamic_batch_size,
  npu_mode }`. This is the base64 `extra_data` metadata prop on every compiled
  `.axmodel`. It holds no MCode content.
- **`libax_interpreter.a` is one object, `npu_hal2.o`**, with no debug info. It is
  an ioctl client for `/dev/npu` (`AX_NPU_Create_handle`, `AX_NPU_Run_task`, ...),
  as the README already found for the card's `.so`. Its version is
  `V1.45.0_P39_20240830` on AX650 and `V2.0.0_P7_20240513` on AX620E. It exports
  `AX_NPU_Set_debug_conf`, and `ax_npu.ko` has `is_npu_debug_dumpmcode_enalbe`. So
  the driver has a debug mode that dumps MCode, but its argument struct is not in
  any shipped header. Calling it is a device experiment and was not tried.
- Nothing names TENG, MAU, verbs, opcodes or segment kinds. No string contains the
  verb bytes (`a1`-`a3`, `a7`-`a9`) as a table. The engine vocabulary stops at
  "neutron", "dot neu", "WBT" and "OCM".

## The vendor MatMul corpus

`msp/sample/ive/data/ive/matmul/matmul_models/` in the AX650N BSP holds 152
models. Each is one `neu mode` node computing `R[M, N] = X[M, K] · Y[N, K]^T`. Both
inputs are live and already quantized (int8 or int16), and `R` is FP32. There are
no calibration literals, so the calibration confound that
`docs/axera-teng2-calibration-isolation.md` chased cannot arise here.

| generation | models | IR / opset | segments | dyn params |
| --- | --- | --- | --- | --- |
| `npu1` | 40 = {s8, s16} × K {256, 512} × N {10000, 100000} × M {16, 32, 48, 64, 96} | 7 / 13 | 5 | none |
| `npu3` | 112 = {s8, s16} × K × N × M {16, 32, 48, 64, 96, 192, 384} × 2 variants | 8 / 16 | 15 | half carry `npu_dyn_params = "R.stride0"` |

Both generations report producer `Pulsar2`. `bsp_matmul_corpus.py table DIR`
reproduces the full table.

### 1. The codec validates on a second compiler

`mcode.check()` passes on **151 of 152** streams. The one exception
(`npu3 s8`) explains 93.9% of its non-zero bytes, just under the 0.94 floor. That
floor was set on this project's own pulsar2 7.0-lite builds. These streams come
from different compiler builds, both older and newer in IR terms, and include
15-segment multi-core layouts. The segment table, verb set, tag set and `a7`
boundary rule all hold unchanged.

### 2. `npu3` streams have 15 segments: probably 3 cores × 5 queues

Our single-core (`NPU1`) builds always have five queues: conv, conv, teng, cv,
sdma. The `npu3` streams have 15, in a fixed pattern:

- segments 0-5 are large and similar in size;
- 6-8 are 544 B in every s8 model;
- 9 varies, while 10 and 11 are usually 32 B;
- 12 is large, while 13 and 14 are 32 B.

The reading "conv × 6, teng × 3, cv × 3, sdma × 3, where only core 0 runs CV and
SDMA" fits every row. It is an inference from sizes, not decoded. `ENGINES` in the
script encodes it.

### 3. The dtype toggle isolates TENG

At identical (generation, K, N, M, variant), the s16 build is larger than the s8
build by engine:

| engine | npu1 median (min-max) | npu3 median (min-max) |
| --- | --- | --- |
| conv | 4.9× (3.2-6.9) | 4.1× (0.1-9.7) |
| **teng** | **78×** (15-188) | **9.1×** (1.4-185) |
| cv | 1.4× | 2.3× |
| dma | 2.1× | 2.0× |

For **s8, TENG is a small fixed-size program**: 640 B (608 B in 2 of 20) on
`npu1`, and 544 B per core on all 56 `npu3` models. For s16 it is large and
unique per shape. Conv growing about 4× and DMA about 2× is what you would expect
if int16 is computed as int8 partial products (DMA moves twice the bytes). TENG
growing far more then suggests TENG does the recombination, which s8 does not
need. **That is a hypothesis.** It fits the sizes and nothing here tests it.

### 4. The s8 TENG program is a template that ignores M

The s8 TENG queues fall into a few **byte-identical** templates:

- `npu1`: 20 models give 6 distinct contents. For example,
  `(K=256, N=10000, M∈{16,32,48,64})` and `(512, 10000, 16)` all share one
  640-byte program.
- `npu3` core 0 (segment 6), static-stride variant: 28 models give 9 contents.
  One of them covers 11 shapes: all K=256/512, N=10000 with M ≤ 96, plus 256×192.

So within a template the exact M (and sometimes K) is not encoded in TENG at all.
It lives in the conv and DMA queues. Between templates the program is rewritten,
not patched: 471 of 640 (npu1) and 461 of 544 (npu3) positions differ. That is
the same wall as every other TENG study in this project. What is new is that, for
s8 matmul, a lookup of about 9 templates covers every shape in the corpus. Which
template a shape gets does not follow K, N or M in any simple way.

### 5. Tensor sizes as immediates

The null control searched for the value + d, d ∈ {3, 5, 7, 11, 13, 17}, as LE32.
It hit 0.4-2% of the time.

- **Whole-tensor byte counts sit in the header/tail**, outside every segment, in
  100% of models: `M·K·e` (X), `N·K·e` (Y) and `M·N·4` (R), where e is the element
  size. This matches the tensor I/O table the README already decoded, now
  confirmed on a second compiler.
- **The output row stride `N·4` is inside the DMA queue**: segment 4 in 100% of
  `npu1` s8 models and 50% of s16 models, and segment 12 in `npu3`.
- **It partly tracks the dynamic-stride variant**, but only as an association. In
  `npu3` s16, the `R.stride0` builds omit `N·4` from segment 12 in 25 of 28 cases,
  and static builds carry it in 22 of 28. In s8 the split is 17/28 and 17/28. That
  is not a rule, so none is claimed.

## What this does not do

- It does not give a TENG generator. Templates are stable within a cluster and
  rewritten between them, and the cluster assignment is not decoded.
- None of this is our target op or compiler. These are vendor IVE matmuls from
  other Pulsar2 builds, not ResNet18-step ops from pulsar2 7.0-lite.
- No device run was made. The debug-dump hook (`AX_NPU_Set_debug_conf`) and the
  AXCL `log.level=0` trace from `docs/axera-teng2-toolchain-mining.md` remain the
  untried runtime leads.

## Reproduction

The corpus is about 22 MB. Download it from
`AXERA-TECH/ax650n_bsp_sdk/msp/sample/ive/data/ive/matmul/matmul_models/`, then:

```
scripts/axera/bsp_matmul_corpus.py table DIR
scripts/axera/bsp_matmul_corpus.py check DIR
```

`tests/test_axera_bsp_matmul_corpus.py` checks the key facts against four
committed models under `scripts/axera/fixtures/bsp_matmul/`. It needs no network
or device.
