# Axera Pulsar2/AXCL compatibility check

Verifies that `onnxsim`'s output stays friendly to **Pulsar2**, the compiler
behind Axera's AXCL toolchain that turns an ONNX model into a `.axmodel` for
the AX6xx/AX8xx NPU line. Based on the handoff notes at
[`../../../junk/axcl-axmodel-onnxsim-notes.md`](../../../junk/axcl-axmodel-onnxsim-notes.md),
and since verified against a real **AX650N** (PCIe, via the AXCL host driver
and `axcl_run_model`) and a real compiled `.axmodel`
(`AXERA-TECH/YOLOv8`'s `AX650/yolov8n_640x640_npu1.axmodel`).

## ⚠️ Confirmed on real hardware: onnxsim corrupts compiled `.axmodel` files

**Do not run `onnxsim.simplify()` on an already-compiled `.axmodel`.** This
was verified end-to-end: `axcl_run_model` ran the real file successfully
(~4.8ms/inference on the NPU), then `simplify()` on that same file dropped
its NPU weight/command data and the result failed to even load
(`axcl_run_model` -> "Create model handle failed").

Root cause: the compiled subgraph is a single node, `op_type="neu mode"`,
whose NPU weight/command blobs are ordinary `graph.initializer` tensors
(`npu_params`, `npu_dyn_params`, `<name>_b<N>_neu`) referenced **only** by
name inside a JSON string in the node's `npu_graph_info` attribute --
**not** as a declared node input. Both onnxsim's own constant-folding
cleanup and onnx-optimizer's `eliminate_unused_initializer`/
`eliminate_deadend` passes treat unreferenced-as-input initializers as dead
and drop them; a fresh shape-inference pass also drops the `graph.value_info`
entries describing those tensors, which the real device's loader also needs.

**No combination of `simplify()`'s public parameters avoids this** --
confirmed by exhausting them: `skip_constant_folding=True` alone,
`skipped_optimizers=["eliminate_unused_initializer", "eliminate_deadend"]`
alone, and even both together plus `skip_shape_inference=True`, all still
produced a file `axcl_run_model` refused to load. See `pulsar2_ops.py`'s
docstring for the full record. `pulsar2_ops.has_out_of_band_npu_data()` /
`pulsar2_backend.unsafe_for_simplify()` detect this **before** calling
`simplify()`, and `worker.py` uses it as a hard pre-flight guard
(`pulsar2_unsafe_for_simplify` status) rather than ever calling `simplify()`
on such a model. `tests/test_pulsar2_compat.py::
test_onnxsim_corrupts_a_compiled_npu_subgraph` reproduces the bug against a
synthetic fixture so it's caught in CI without needing the real device --
this confirms the handoff notes' own recommendation to only ever simplify
*pre*-`pulsar2 build` ONNX (approach (b) in the notes), never a compiled
`.axmodel` (approach (a)).

## ✅ Also confirmed on real hardware: approach (b) itself is safe

The real Pulsar2 toolchain (`pulsar2:6.0-lite`, matching the AX650N's
installed firmware) was loaded via Docker and used to actually build two
real `onnxmodelzoo` models end to end -- ONNX -> `pulsar2 build` ->
`.axmodel` -> run on the real AX650N:

- **`resnet18d_Opset18`**: both the original ONNX and its onnxsim-simplified
  twin (onnxsim folded 117 dangling weight-as-input entries down to 1 real
  input, same 56 nodes) compiled to a single NPU subgraph with **identical
  compiler-reported `max_cycle` (1,318,764)**. Running both `.axmodel`s on
  the real device with the same input produced **bit-identical output**
  (`np.array_equal` `True`, max abs diff `0.0`). This is the concrete,
  positive counterpart to the corruption finding above: simplifying
  *pre*-compile ONNX (approach (b)) is safe.
- **`googlenet-6`** (opset 9, uses `LRN`): `pulsar2 build` did not gracefully
  fall `LRN` back to CPU -- it hard-failed the whole build at the frontend
  parse stage (`KeyError('dont support LRN opr in AXOPS/ONNXOPS/CUSTOM_OPS')`)
  before any CPU/NPU partitioning happened. Also below Pulsar2's documented
  minimum opset (11) for AX650. Useful negative data point: an unsupported
  op isn't always "less NPU-friendly," sometimes it's a hard build failure.

This also directly answered an open question from the handoff notes: Axera
publishes the real AX650 NPU op-support list in Pulsar2's own docs
(`appendix/op_support_list_ax650.html`) -- 92 ops, opset >= 11 required.
It's now `pulsar2_ops.AX650_SUPPORTED_OPS` / `AX650_MIN_OPSET`, and
`pulsar2_backend.ax650_build_risks()` uses it to predict (not guarantee) the
two failure modes seen above *before* attempting a real build.

## This one is not like its siblings

[`scripts/qualcomm`](../qualcomm) (QNN), [`scripts/intel`](../intel)
(OpenVINO), and [`scripts/amd`](../amd) (MIGraphX) each wrap a **real**
compiler via a pip-installable ONNX Runtime execution provider, so they
measure actual compile/run behavior. Pulsar2 has neither a PyPI package nor
an ORT execution provider -- it ships as a Docker image -- so there is no
compiler to invoke here for testing *pre*-compile ONNX. (What real hardware
*can* do -- run an already-compiled `.axmodel` via the `axcl_run_model` CLI
-- is a different, narrower thing; see the corruption finding above, which
is exactly what that access was used for.)

So the coverage side of this harness is a **static heuristic**, not a
compiler check: it flags onnx op types that are extremely unlikely to run on
*any* fixed-function NPU (control flow, sequence/optional types, string ops,
data-dependent-shape ops), plus a non-standard ONNX `domain` check that
turned out *not* to be how Axera actually marks a compiled subgraph (see
`pulsar2_ops.py`'s docstring -- the real marker is `op_type="neu mode"` in
the plain default domain). See `pulsar2_ops.py`'s docstring for the full
reasoning and its explicit `CPU_ONLY_OPS` caveats.

## What it checks

For each model:

0. If it already has a compiled Axera NPU subgraph node (`op_type="neu
   mode"`) -> `pulsar2_unsafe_for_simplify`, **without calling `simplify()`
   at all** (see the corruption finding above).
1. Otherwise, `simplify` the model with onnxsim.
2. Compute the static Pulsar2-NPU-blocker set (`pulsar2_ops.blocking_ops`) for
   the original and the simplified graph.
3. If simplification **introduced** a blocking op type that wasn't already
   present -> `pulsar2_regression` (a failure): simplification likely folded
   something into a form Pulsar2's NPU partitioner would reject, pushing more
   of the graph onto its CPU fallback path than before.
4. If simplification dropped NPU weight/command data a compiled subgraph
   node still references -> `pulsar2_data_corrupted` (shouldn't be reachable
   given step 0, checked anyway as defense in depth).
5. If onnxsim's own correctness check reported a mismatch ->
   `simplify_check_failed`.

A model that already has a blocker *before* simplification, or still has one
after but didn't gain a new one, passes (`ok`) -- that's a property of the
input graph, not something onnxsim introduced.

## No-Docker/no-device simulator + compatible quantizer

`pulsar2_simulator.py` and `pulsar2_quantizer.py` turn the confirmed-real
data above into something you can query without the ~1GB Docker image or
physical hardware:

- **`pulsar2_quantizer.quantize_like_pulsar2()`** reproduces Pulsar2's real
  PTQ *numeric convention* -- read directly off a real `quant_axmodel.onnx`
  from the `resnet18d` conversion: **U8 (uint8), per-tensor, asymmetric**
  activations and **S8 (int8), per-channel, symmetric** weights, MinMax
  calibration. It turns out **onnxsim already has a quantizer with exactly
  this convention** -- `onnxsim.quantize_static(method="minmax")`
  (`onnxsim/calibration.py`, an "asymmetric uint8 affine quantization" per
  its own C++ pass's comment) -- so this is now a thin wrapper over
  onnxsim's own quantizer rather than a hand-rolled equivalent built on
  `onnxruntime.quantization`. It does **not** reproduce Pulsar2's actual
  quantized IR: that file's ops are proprietary (`AxQuantizedConv`,
  `AxQuantizeLinear`, ... all in the plain default domain, not standard ONNX
  `QuantizeLinear`/`DequantizeLinear`, and not executable by onnxruntime),
  and onnxsim's quantizer only quantizes Conv/MatMul/"vanilla" Gemm nodes
  where Pulsar2 quantizes essentially the whole graph -- see its docstring.
- **`pulsar2_simulator.py`** adds `partition()`/`coverage()` (per-node
  `AX650_SUPPORTED_OPS` membership -- correctly predicted both real
  conversions: "full" for `resnet18d`, "partial" with
  `{"LRN": 2, "Dropout": 1}` for `googlenet-6`) and `simulate()` (runs the
  quantized graph through onnxruntime's CPU EP as an fp32-vs-INT8 estimate).
  Validated against real hardware: on `resnet18d` with the same input image,
  this simulator's INT8 output had **0.938 cosine similarity** to the real
  device's actual output, close to fp32-vs-real's own **0.949** -- similar
  *magnitude* of quantization noise, but **not** rank/bit-accurate (top-5
  didn't match between fp32, simulated, and real on that input). Both
  degrade gracefully (`SIMULATOR_AVAILABLE`/`PULSAR2_QUANTIZER_AVAILABLE`)
  when `onnxruntime` isn't installed (onnxsim's own `quantize_static` only
  imports it lazily, inside `calibrate()`); `partition()`/`coverage()` need
  only `onnx` and always work.

Use these for a fast first read before spending time on a real
`pulsar2 build` -- always confirm anything that matters on the real
toolchain and hardware, the same way this README's findings were confirmed.

## Real NPU profiling: `chrome://tracing`-compatible trace.json

Confirmed real (this is a genuine Pulsar2 feature, not something this repo
implements): passing `--compiler.npu_perf` to a real `pulsar2 build` writes
`${output_dir}/compiler/debug/subgraph_npu_0/b1/trace.json` -- a standard
Chrome Trace Event Format file (`{"traceEvents": [...], "displayTimeUnit":
...}`, each event `{"ph": "X", "pid": "subgraph_npu_0", "tid": "teng2", ...,
"args": {...}}`) that loads directly in `chrome://tracing` (or Edge's
`edge://tracing`), with one lane per NPU IP (`teng`/`sdma`/`cv`/`conv`) and
one span per hardware task -- op names, dependencies, ddr-swap/load/store
colors. Also pass `--debug.dump_frontend_graph` to get
`frontend/optimized_quant_axmodel.onnx` (openable in Netron) so trace task
labels can be matched back to the algorithm graph. A flat CSV covering the
same data (`op_profile.csv`, one row per op: cycles, bandwidth, tensor
shapes) is written alongside it.

Reproduced against the real `resnet18d_Opset18` build used throughout this
README:

```bash
docker run --rm -v "$PWD:/data" pulsar2:6.0-lite \
  pulsar2 build --target_hardware AX650 \
  --input model/resnet18d.onnx --output_dir output/resnet18d_trace \
  --config config/resnet18d_build_config.json \
  --compiler.npu_perf --debug.dump_frontend_graph
```

This needs a real `pulsar2 build`, not just a compiled `.axmodel` -- it's
generated at compile time from the cycle model, not measured live on-device
by `axcl_run_model`/`ax_run_model` (those only report aggregate min/max/avg
latency). **Automated**: `convert_onnxmodelzoo.py --profile` passes this
through automatically (see below); see Pulsar2's own docs
(`other_tools/profiling.html`) for the full trace-UI reference.

## Digging into a compiled `.axmodel`'s `neu mode` node

Prompted by "could we generate `.axmodel` without Axera tools?" -- short
answer still no (see below), but here's what direct inspection of a real
compiled file, plus a real `--compiler.npu_perf` trace, actually shows.

**The node's own attributes** (from a real `pulsar2 build` of a tiny Mistral
checkpoint via `build_from_hf_checkpoint()`):

```
neu_name: "subgraph_npu_0"
npu_graph_info: {"name": "subgraph_npu_0", "dotneus": [{"neu_key":
                 "subgraph_npu_0_b1_neu", "batch": 1, "extra_inputs":
                 [{"name": "params", "const_data_key": "npu_params"}]}]}
outputs_info: {"lm_head.matmul.94": ["FP32", [1, 8, 32000]]}
version: <int>
```

`neu_key`/`const_data_key` just name ordinary `graph.initializer` UINT8
blobs (see `pulsar2_ops.py`'s docstring for why onnxsim's own dead-code
elimination strips these): `npu_params` (21MB here -- the raw weight
dump, no header, just concatenated tensor bytes at offsets the other blob
names), `npu_dyn_params` (0 bytes for a static-shape model), and the
`<neu_key>`-named blob itself (28KB here) -- the actual compiled program.

**The compiled-program blob is a FlatBuffers container**, confirmed by
hand-decoding its first 32 bytes against the public FlatBuffers spec: byte
0 is a valid root-table uoffset (28), which resolves through a
well-formed vtable (size 24, 7 populated field slots) -- not a coincidence,
a real, spec-conformant FlatBuffers root table. Scanning the blob for
embedded strings surfaces the vtable's field names, present twice (once
right after the header, once duplicated near the very end of the buffer --
consistent with FlatBuffers' bottom-up buffer-construction convention):

```
params, ddr_swap, lm_head.matmul.94_offset, lm_head.matmul.94,
position_ids_offset, position_ids, input_ids_offset, input_ids, _ocm_base
```

i.e. a **tensor I/O offset table**: a `<name>`/`<name>_offset` pair per
graph input/output, plus `_ocm_base` (the AX650's on-chip SRAM base
address) and `ddr_swap` (matches the real, timestamped "add ddr swap..."
compiler pass -- see the build-phase breakdown two sections up). The
remaining ~27KB in the middle of the blob (>95% of it) has no further
embedded strings or hand-decodable structure -- almost certainly the
actual scheduled NPU instruction stream, in a proprietary, undocumented
encoding.

**No usable Axera-provided FlatBuffers schema was found.** Searched (inside
`pulsar2:6.0-lite`): no `*.fbs`/`*.bfbs`/`*_generated.{h,py}` files
anywhere under the image; no `import flatbuffers`/`from flatbuffers` in any
plaintext `.py` file under `/opt/pulsar2` (the only such hits anywhere in
the image are ONNX Runtime's own unrelated `.ort`-format schema, bundled as
a dependency); no relevant field-name strings (`ddr_swap`, `_ocm_base`,
`neu_key`, ...) or `flatbuffers`/`.fbs` mentions in any of the five
`backend/*/*_cmodel.so` libraries (these are almost certainly cycle-accurate
NPU functional simulators used for verification, not the FlatBuffers
writer). The `flatbuffers` PyPI package itself **is** installed in the
image (confirming the container format), but whatever Python code actually
constructs this schema lives inside the Pyarmor-obfuscated `yamain`/
`yasched`/`opset` modules (see `pulsar2_ops.py`'s docstring) -- not
recoverable by inspection.

**A real `trace.json` (see the profiling section above), though, gives away
almost the entire semantic content of that opaque instruction stream --
in plain, readable JSON, no reverse engineering needed.** For the same
tiny Mistral build (`--profile`), the 635 trace events reveal:

- **Five named parallel execution engines** (`tid` values): `conv0`/`conv1`
  (213/210 events -- the MAC/matmul compute units), `cv3` (78 events -- a
  vector/elementwise unit: RMSNorm, RoPE rotation, Softmax), `sdma4` (76
  events -- a system-DMA/prefetch engine), `teng2` (58 events -- handles the
  embedding gather and I/O staging). A real, confirmed heterogeneous
  multi-engine architecture, not a single monolithic "NPU core."
- **Named memory regions**, matching the FlatBuffers offset table above:
  `ocm_base` (871 references -- the dominant, fast on-chip working memory),
  `params` (154 -- DRAM-resident weights), `ddr_swap` (2 -- DRAM staging for
  spilled tensors), plus the three named I/O tensors.
- **Original ONNX op names are preserved end to end** (e.g.
  `model.layers.0.q_rope.rot.41`, `model.layers.0.attn_norm.var_eps.18`),
  each lowered to a small set of NPU primitives: `onnx.FullyConnected`
  (417/635 events -- every projection and the FFN, all lowered to the same
  primitive), `AxQuantizedMatMul` (3), `onnx.Silu` (1).
- **The trace's own time units are NPU cycles** (scaled by 1000, despite
  the file's `displayTimeUnit: "ns"`): summing every event's `dur` gives a
  total schedule span of ~287,231, matching this exact build's own reported
  `max_cycle=287,211` (see `BuildResult.max_cycle`) to within rounding.

Net effect on "could we generate `.axmodel` without Axera tools": no change
to the answer, but a much better-understood boundary. The *container*
(FlatBuffers) and the full *dataflow graph* (`trace.json`, when
`--profile` is used) are both now understood well enough to write a reader
without Docker. Actually *producing* a correct, hardware-loadable
instruction stream from that dataflow graph -- real quantization, tiling,
scheduling, and codegen into an undocumented ISA -- still requires
Pulsar2's own (obfuscated) compiler backend.

### What's op-specific vs. boilerplate, across real Conv/MatMul variants

The single-model dig above raises an obvious question: how much of that
FlatBuffers offset table and instruction stream is generic wrapper vs.
op-specific? Answered by compiling 9 small, hand-built ONNX graphs (plain
`MatMul`, `Gemm` with bias, dense `Conv` 3x3, 1x1 pointwise, depthwise
(`group=C`), grouped (`group=2`), stride-2, dilation-2, and a batched
(rank-3) `MatMul`) through the same real `pulsar2 build --compiler.npu_perf`
and inspecting each result the same way:

- **The FlatBuffers field-name table is identical across every one of the
  9 models**: always exactly `params`, `<input>_offset`, `<output>_offset`,
  `_ocm_base`, and the graph name -- regardless of kernel size, stride,
  dilation, groups, or op family. **No op parameter ever shows up as a
  named field.** Conv's stride/padding/dilation/group and Gemm's alpha/
  beta/transpose flags are entirely opaque, baked into the unlabeled
  instruction bytes -- this table is pure I/O bookkeeping, not a
  semantically rich IR.
- **`ddr_swap` is a real, conditional field**, not a fixed part of the
  schema: present in the earlier 28-layer LLM build (something spilled to
  DRAM), absent from all 9 of these small models (everything fit in OCM).
- **Exactly two compute "primitive families" appear**, cleanly split by op
  family: every `Conv` variant -- dense, 1x1, depthwise, grouped, strided,
  dilated, no exceptions -- lowers to `Pre_AxTranspose` -> `AxQuantizedConv`
  (x6 tiles) -> `Post_AxTranspose` (almost certainly a NCHW<->NHWC layout
  swap around a channel-last-native conv engine); `MatMul` lowers to
  `onnx.FullyConnected` instead. Op parameters change *within* those
  primitives (invisibly) but never *which* primitive gets picked.
- Trace event **naming isn't fully consistent**: `matmul2d`'s compute
  events are labeled `op_1:onnx.FullyConnected_<tile>_<tile>` (primitive
  name visible), but `gemm_bias`'s and the batched `matmul_batched3d`'s are
  labeled directly after their own output tensor (`y_0_0`, `y_1_2`, ...) --
  the same underlying compute, differently named depending on some
  internal fusion/naming decision, not a reliable way to detect op type
  from the trace alone.
- **Cycle cost and weight-blob size don't scale the way FLOP count would
  predict, at this tiny (16x16, 4-8 channel) test size.** 1x1 pointwise
  conv costs nearly as many cycles as full dense 3x3 (2052 vs. 2096) --
  fixed per-tile overhead dominates raw MAC count here. Most strikingly,
  **dilated conv's stored weight blob is 2.6x larger than a plain conv with
  the identical (8,4,3,3) kernel shape** (3.66KB vs. ~1.4KB) despite having
  the same number of logical weight values -- strong evidence the compiler
  materializes a real, zero-expanded ("atrous") kernel footprint for
  dilation rather than an actually-sparse dilated MAC pattern, and it's
  also the most expensive op tested by cycle count (2228).
- Every model tiles into a small, similar instruction count regardless of
  large parameter differences at this scale: all 6 `Conv` variants compile
  to exactly 6 `AxQuantizedConv` sub-tiles each; `matmul2d` to 6
  `FullyConnected` sub-tiles; the batched matmul to 7. Tiling granularity
  here looks governed by fixed hardware tile-size constants more than by
  the specific op's shape/parameters -- this may well change at larger,
  more realistic tensor sizes where tiling actually has to split work up.

### External corroboration: a real hardware teardown

An independent third-party writeup --
[jas-hacks.blogspot.com's AX650N/Sipeed M4N teardown](https://jas-hacks.blogspot.com/2024/09/ax650n-sipeed-maix-iv-axerapi-pro-npu.html)
(not Axera's own documentation; treat specific numbers as one outside
source's reporting, and any interpretive claims -- explicitly flagged
below -- as that author's own inference, not confirmed fact) -- gives real
names and numbers for the hardware this repo's own trace.json digging
above only inferred generically:

- The NPU ("Neutron") is described as 13 execution units + 3 SDMA units:
  3 Convolution Units (handling depthwise/grouped conv, dilation, and
  ConvTranspose), 3 Computer Vision Units (image normalize/resize/clip/
  warp), 3 Tensor Units (activation, pooling, elementwise, reduction), and
  a single Matrix Arithmetic Unit (int8/int16 in, fp16/fp32 out). This
  lines up with this repo's own trace.json engine names in outline --
  `conv0`/`conv1` for the Convolution Units, `sdma4` for an SDMA unit --
  but not exactly: our own LLM trace's `cv3` engine ran RMSNorm/RoPE
  elementwise math, which reads as "Tensor Unit" work by this source's own
  description, not "Computer Vision Unit" work, and we never observed a
  distinct engine for the single Matrix Arithmetic Unit despite compiling
  real `MatMul`/`Gemm` graphs above (both engine-name schemes may not be
  directly comparable, or the compiler may route elementwise math onto
  whichever engine family has spare capacity rather than a fixed
  CV-vs-tensor split). Reported, not reconciled -- a real open question
  for anyone digging further.
- **On-chip memory (OCM): reported as 11.5MB, address space ending at
  `0xAFFFFF`.** `0xAFFFFF + 1 = 0xB00000 = 11,534,336` bytes = exactly
  11MiB by that address range -- close to but not exactly the "11.5MB"
  figure quoted; direct, checkable confirmation that `ocm_base`'s byte
  offsets seen in this repo's own trace.json digging above (all under
  ~3.2MB in our tiny test models) sit well inside a real, multi-megabyte
  on-chip SRAM, not some other memory space.
- 8GB total SoC RAM, split 4GB Linux / 4GB "CMM" (Contiguous Memory Model)
  for peripherals -- CMM is almost certainly what this repo's own findings
  call `params`/DRAM-resident weight storage and `ddr_swap` staging.
- Claimed performance: 72 TOPS mixed precision (18.0 TOPS@INT8, 43.2
  TOPS@INT4 and 10.8 TOPS@INT8 "from NPU alone" per Axera's own SDK docs,
  per that source). **The author's own interpretation** (not a measured
  fact): a single Matrix Arithmetic Unit instance may bottleneck LLM
  inference, since every `MatMul`/`Gemm` in a transformer routes through
  it. That's a plausible complementary explanation for *why* LLM inference
  is slow on this hardware, alongside (not instead of) the very different,
  independently-confirmed bottleneck this repo's own `demo_hf_llm_chat.py`
  measured: `axcl_run_model`'s ~700ms-per-invocation process/model-reload
  overhead, which has nothing to do with the NPU's own compute engines at
  all and would dominate regardless of how many Matrix Arithmetic Units
  existed.
- Real production reference point (**not comparable to
  `demo_hf_llm_chat.py`'s own measured tokens/sec** -- different
  measurement entirely: real `ax-llm` + KV-cache decode via Pulsar2's own
  `llm_build()` path, not this repo's re-run-the-whole-model-per-token
  `build_from_hf_checkpoint()` path, and no per-call CLI-reload overhead
  since it's a persistent server): Phi-3 Mini reported at ~4.4 tokens/sec
  on the AX650N, vs. ~6.46 tokens/sec on an RK3588 for comparison.
- Confirms real ONNX-level `.axmodel` structure from an outside source
  independently: "axmodel files contain a mix of ONNX data and an internal
  graph representation" sent to the NPU kernel driver -- matching this
  repo's own finding of an ordinary ONNX container wrapping an opaque,
  FlatBuffers-framed internal representation.
- **Confirmed, and more specific than reported**: `gemm_bias` above used
  default `alpha=1.0, beta=1.0` and compiled fine; a `Gemm` with
  non-default values (`alpha=2.0, beta=0.5`) **fails outright**, not
  merely "restricted" -- a real `pulsar2 build` on that graph throws
  `KeyError: 'dont support AxQuantizedGemm opr in AXOPS/ONNXOPS/
  CUSTOM_OPS'` before quantization even runs. Default-alpha/beta `Gemm`
  apparently lowers to the same path as a plain `MatMul` + bias-add (hence
  `gemm_bias` succeeding above); any other `alpha`/`beta` maps to a
  distinct, entirely unimplemented `AxQuantizedGemm` op. Confirms the
  blog's suspicion with a precise, reproducible mechanism.

### Differential analysis: how elementwise ops and Conv bias get encoded

The dig above characterizes one model's compiled output; this pushes
further with **differential analysis** -- compiling many near-identical
graphs and byte-diffing the results to locate exactly where a specific,
controlled value ends up. Test graph throughout: `Add`/`Sub`/`Mul`/`Div`
between a `float[1,4]` input and a uniform-broadcast constant, or a `Conv`
with a bias term -- varying only the constant/bias value between builds.

**A "trivial" fast-path exists for small uniform constants, scale=1,
zero_point=0, storing the constant's own integer value as a raw byte** --
but confirmed real by testing across all four ops, **the trivial *set*
is op-specific, not a shared threshold**:

- `Add`: exactly the uniform values `{0, 1, 2}` are trivial; `3` and up,
  any negative value, and any non-integer are not.
- `Sub`: only `{0, 1}` -- `Sub(x, 2.0)` is *not* trivial, unlike
  `Add(x, 2.0)`. Not simply "`Sub(x, c)` lowers to `Add(x, -c)`" either:
  that would predict `Sub(x, 1.0)` (i.e. `Add(x, -1.0)`) to behave like
  `Add`'s confirmed-non-trivial negative case, but it doesn't -- it's
  trivial, storing `01 01 01 01` same as `Add(x, 1.0)`.
  `Mul`: every uniform value tried (`0`, `2`, `3`) was trivial -- no
  "rich" encoding observed for `Mul` at all.
- `Div`: triviality depends on **the constant's reciprocal**, not the
  constant itself -- `Div(x, 0.5)` (reciprocal `2.0`) is trivial, storing
  `02 02 02 02`, while `Div(x, 2.0)` and `Div(x, 3.0)` (reciprocals `0.5`,
  `0.333...`, non-integer) saturate every element to `0xff`. Consistent
  with `Div(x, c)` being compiled as `Mul(x, 1/c)` internally.

**Any non-integer value saturates every element to `0xff` (255)**,
regardless of magnitude -- confirmed across `Add`/`Sub`/`Div`'s non-integer
cases (`0.5`, `3.14159`, and `Div`'s non-integer effective reciprocals).
It's specifically about exact integer-valuedness of whatever value is
actually being quantized (the reciprocal, for `Div`) -- not "small enough."

**A uniform broadcast is required for the trivial path, even when every
individual element already qualifies**: `Add` with the mixed constant
`[1, 1, 2, 2]` (every element in the "trivial" set `{0,1,2}`) still gets
the non-trivial encoding, because the *tensor* isn't a uniform single-value
broadcast. Mixed constants still store their exact literal integer values
per element in the non-trivial path (`[1,2,3,4]` -> `01 02 03 04`), so
"non-trivial" doesn't mean "imprecise" -- it means "not the degenerate
single-value fast path."

**Confirmed real, reproducible compiler bug**: `Mul(x, 1.0)` and
`Div(x, 1.0)` both crash a real `pulsar2 build` with the identical
`NotImplementedError: Seems config of input(y) doesn't exist`. Multiplying
or dividing by the identity constant appears to get eliminated by
Pulsar2's own frontend graph optimizer (`x*1=x`, `x/1=x`) before
quantization runs, leaving the declared graph output with no producing
node. `tests/test_axera_neu_format_arith_ops.py::
test_mul_and_div_by_one_crash_the_real_build` locks this in.

**`Div(x, 0.0)` doesn't error -- it stores literal IEEE-754 `+Infinity`**:
`00 00 80 7f` (float32 `+inf`) repeated once per element, a third,
distinct byte-length class from the other two, and the only case found
where this field holds genuine float32 data instead of an integer code --
a sensible fallback once the "true" quantized value is undefined.

**A field that resists decoding, isolated but not solved**: `Add`/`Sub`'s
non-trivial encoding appends 4 extra bytes past the per-element values.
Ten-plus decodings were tried and rejected (float32, uint32, a `bf16`
pair, an `fp16` pair, `xxhash32`/`xxhash64` of several byte encodings of
the constant, a hand-computed asymmetric output-quantization scale/
zero-point from the real calibration data) -- none matched. What *is*
confirmed: holding the constant fixed (`c=99`) and varying only the
input's calibration scale (x1, x100, x0.01) changed these bytes
completely while the constant's own quantized bytes stayed identical --
so the field depends on the input/output's calibration range, not the
constant alone. The two 16-bit halves are also mathematically coupled:
treating them as `(pair1, pair2)`, `pair2 * scale_y ≈ pair1` held to
within rounding across all three calibration scales, where `scale_y` is
the real `(max-min)/255` output range independently computed from the
actual calibration samples used. A real, non-arbitrary (value,
value-expressed-in-quantization-units) pair -- just one whose absolute
unit/format wasn't identified.

**`Conv`'s bias term, by contrast, decodes cleanly**: byte-diffing five
otherwise-identical `Conv` builds that differ only in bias value locates
the bias-dependent region precisely, and it holds two 4-element
(one per output channel) plain `float32` arrays -- not further-obfuscated
integer codes. One array is small (~0.0026-0.0037) and shrinks
monotonically as the bias value grows, consistent with a per-channel
requantization multiplier `M_channel = input_scale * weight_scale_channel
/ output_scale` (a larger bias widens the calibrated output range, so
`output_scale` grows and `M_channel` shrinks) -- exactly the standard
quantized-conv parameterization real edge-inference runtimes use. The
other array (larger magnitude, ~-159 to 242) plausibly holds a quantized
bias term but wasn't independently re-derived from scratch. Real,
recognizable structure here, in clear contrast to `Add`/`Sub`'s still-
opaque field above.

### The AX650N card's own firmware: real terminology, one real dead end

Prompted by "could you analyze firmware loaded to AX650N?" -- the AXCL host
package ships the actual firmware pushed to the card over PCIe at
`/lib/firmware/axcl/ax650_card.pac` (155MB). It's a real, parseable
container, not an opaque blob: a fixed header (magic, board name
`AX650_card`, firmware version `V2.25.0`) followed by 5 fixed-size (464
byte) partition-table entries, each holding a name/type/filename plus an
`(offset, size)` pair (`<Q Q` little-endian) into the file. Confirmed by
parsing it and checking the last partition's `offset + size` matches the
file's exact total size:

```
UBOOT  (u-boot.bin)        offset=2448        size=1,069,223
DTB    (AX650_card.dtb)    offset=1,071,671   size=192,178
ATF    (atf_bl31.img)      offset=1,263,849   size=28,736
KERNEL (Image)             offset=1,292,585   size=20,285,448
ROOTFS (rootfs.ext4)       offset=21,578,033  size=134,217,728
```

**The card runs a full, independent embedded Linux system** (u-boot -> ARM
Trusted Firmware -> a real Linux `Image` -> an ext4 root filesystem) that
mediates PCIe register/DMA access for the host-side `axcl` API -- not a
single-purpose NPU firmware blob. `rootfs.ext4` is browsable read-only with
`debugfs` (no mount/root needed) and its `/soc/ko/ax_npu.ko` -- the actual
NPU kernel driver -- is **not stripped**, so its full symbol table is
directly readable with `nm`. That symbol table resolves several things this
README had only inferred from outside:

- **The compiled instruction blob is internally called "mcode"**, not just
  this project's "neu mode blob" label -- confirmed by real log format
  strings (`"mcode[%u] size is %u"`, `"mcode error"`) and functions
  (`dump_mcode_with_handle`, `modify_mcode_crc`, `npu_cv_outer_update_mcode`,
  `is_npu_debug_dumpmcode_enalbe`). It's CRC-protected
  (`npu_get_cmd_crc_state`/`npu_get_data_crc_state`, a `crc_table`/
  `crc_update`), consistent with the FlatBuffers dig above finding no
  further structure in >95% of the blob -- it may simply be checksum-opaque
  binary, not additionally obfuscated.
- **Execution is a command queue (`cmdq`), not a fetch-decode-execute CPU
  loop**: `cmdq_write_instruct`, `cmdq_set_eu_idle`, `cmdq_set_wait_cycle`,
  `cmdq_set_clear_job_id`, `cmdq_connect_use_jump` (a queue-to-queue jump
  primitive), `cmdq_update_sync`, backed by `sync_manager_*` functions
  (job IDs, interrupt-clear, wait-bypass, timer thresholds). This reframes
  what to look for in "mcode": fixed-format command/DMA descriptors with
  sync primitives, not a general-purpose instruction set -- a materially
  different (and more tractable) reverse-engineering target than "unknown
  CPU ISA".
- **"EU" (execution unit) and 5 queue types are named directly**:
  `npu_eu_mask_2_sub_id`/`get_eu_class_mask` alongside five identically-
  compiled (`.isra.0`, i.e. the same source macro instantiated per name)
  queue setters -- `npu_dma_set_queue`, `npu_mau_set_queue`,
  `npu_potato_set_queue`, `npu_sdma_set_queue`, `npu_warp_set_queue`. `MAU`
  (Matrix Arithmetic Unit) independently corroborates the earlier
  jas-hacks.blogspot.com teardown's "1 Matrix Arithmetic Unit" claim;
  `sdma` matches the real trace.json engine name (`sdma4`) exactly.
- **OCM terminology matches exactly**: `get_vnpu_ocm_base`/
  `get_vnpu_ocm_size`/`print_vnpu_ocm_1k_contents`, consistent with the
  FlatBuffers offset table's `_ocm_base` field and the blog's independently
  measured 11.5MB OCM region.
- `libax_interpreter.so` (also on the card, `/opt/lib`) turned out to be a
  **misleading name for this investigation**: despite being 22KB and
  plausibly containing a bytecode interpreter, every one of its imported
  symbols is a lifecycle call into `ax_npu.ko`'s own userspace API
  (`AX_NPU_Create_task`, `AX_NPU_Run_task`, ...) -- it's a thin client, not
  an mcode interpreter. Combined with `cmdq`/`sync_manager` being the only
  execution-adjacent code in the driver itself, this confirms **there is no
  software interpreter for mcode anywhere in the accessible stack** -- the
  NPU's execution units decode it directly in hardware. The only place mcode
  gets *decoded in software* at all is Pulsar2's own verification tooling.

**A real dead end, reported for completeness rather than pursued further**:
Pulsar2's own backend simulators (`/opt/pulsar2/backend/ax650npu/
ax650npu_cmodel.so` inside the Docker image, one per target chip) export a
genuine, undocumented-elsewhere **mcode assembler and disassembler** as
plain C symbols -- `mcode_new`, `mcode_dump`, `mcode_size`,
`mcode_disassemble`, `assembler_eu`, `assembler_ctrl`, `disassembler_eu` --
exactly the tool that would turn this whole investigation's differential
byte-diffing into direct disassembly. **It's commercially licensed**: even
just `dlopen()`-ing the library (via `ctypes.CDLL`, before calling any
function) fails with `Sentinel LDK Protection System: Sentinel key not
found (H0007)` -- a hardware/software dongle-gated check from a real
third-party licensing product (Thales Sentinel LDK), fired from the
library's own load-time constructor. Not pursued further: this is a
legitimate commercial licensing control on Axera's own tooling, not a
technical obstacle to route around.

### Public prior-generation API headers fill in "mcode"'s place in the stack

Following up on the firmware dig above with public research (no Docker, no
device -- just Axera's own publicly-redistributed SDK) turned up a real,
legitimately public source that independently corroborates and extends the
firmware's terminology: **`sipeed/axpi_bsp_sdk`**
(github.com/sipeed/axpi_bsp_sdk), Sipeed's Axera-authorized public BSP
release for the **AX620A** (an earlier chip in the same lineage as AX650N,
predating Pulsar2). Its `msp/out/include/*.h` headers are real, unobfuscated
C, not reverse-engineered -- and they name the exact same concepts this
README's differential analysis and firmware dig had only inferred from
outside, one generation earlier:

- **`AX_NPU_SDK_EX_Create_handle(handle, dotNeuAddr, dotNeuLen)`** --
  confirms "dot-neu" is literally the raw compiled-model buffer handed to
  the NPU runtime, one-to-one with the FlatBuffers `npu_graph_info`
  attribute's `"dotneus"` JSON field this README already decoded from a
  real AX650 `.axmodel`. Same name, two chip generations apart.
- **`AX_NPU_SDK_EX_MODEL_CMM_INFO_T`** breaks a dot-neu's memory footprint
  into exactly four named parts: `nModelMcodeSize` (the main compiled NN
  program -- what this whole investigation has been calling "mcode"),
  `nCvPreProcessMcodeSize` (a *separate* compiled program specifically for
  ISP/CV pre-processing -- mcode isn't unique to NN inference), `nWbtSize`
  ("Wbt" = **Weight Table**, confirmed by `run_neu_v2.cpp`'s
  `--wbt-index`/`pWbtNames`/`nWbtNum` -- named, indexable weight blobs, the
  direct ancestor of AX650's `const_data_key`-named initializers), and
  `nRingbufferSize` (matches `ax_npu.ko`'s real `g_ddr_ringbuf`/
  `g_ddr_ringbuf_lock` symbols found in the firmware dig, and plausibly the
  ancestor of AX650's FlatBuffers `ddr_swap` field).
- **The container hierarchy is now clear across the whole stack**: your
  ONNX graph compiles into a **"Joint" model** (`AX_JOINT_CreateHandle`,
  `AX_JOINT_GetJointModelType` -- confirmed by `joint.h`, the modern,
  newer-than-`dot-neu` unified API in the same SDK -- almost certainly the
  internal name for what ships publicly as a single `.axmodel` file), which
  bundles one or more **"dot-neu"** subgraphs (matching this README's own
  finding that a real LLM per-layer `.axmodel` has *two* `neu mode` nodes,
  decode and prefill), each made of **mcode** (+ optional
  CvPreProcessMcode) referencing named **Wbt** weight tables and using a
  DDR **ringbuffer** for spillover.
- **A real, plausible explanation for `SpatialTransformer`'s odd calling
  convention** (found during this project's op-coverage sweep: `theta` is
  six separate scalar node *attributes*, not a tensor input): both the
  AX620A and (still, per `ax_interpreter_external_api.h`'s modern
  `AX_JOINT_RTV_TYPE_T` copy) newer generations have a **Runtime Variable
  (RTV)** mechanism -- `AX_NPU_RTV_AFFINE`, `AX_NPU_RTV_WARP_CCM`,
  `AX_NPU_RTV_WARP_MAT33`, and a dozen ISP-coefficient variants -- small,
  named slots inside an already-compiled mcode that get *patched* at
  inference time without recompiling, generated by a separate tool the
  sample code calls out by name: **`librosetta`**. `SpatialTransformer`'s
  affine matrix is exactly the shape of data an RTV slot exists for --
  plausible, not confirmed, since AX650's own RTV support (if any) wasn't
  independently verified here.

This is public, Axera-authorized documentation (a board vendor's official
BSP redistribution), not a leak or a bypass -- a clean source distinct from,
and unaffected by, the Sentinel-gated tooling above. It doesn't reveal
mcode's actual instruction encoding (a different chip generation, and these
headers are the *host API* around dot-neu, not dot-neu's own internal
format), but it resolves what several previously-separate, only-inferred
pieces of terminology actually mean and how they relate to each other.

### Applying the new vocabulary to a real `.axmodel`, and a real mcode-size finding

With "dot-neu / mcode / Wbt / ringbuffer" now understood as real terms (not
this project's own labels), re-examining a freshly-built real `.axmodel`
confirms the mapping directly: the `<neu_key>`-named `graph.initializer`
blob this README has been calling "the compiled program" **is mcode**, and
`npu_params` **is the Wbt** (Weight Table). `AX_ENGINE_CMM_INFO`/
`AX_JOINT_MODEL_CMM_INFO` (the AX650-era public structs, checked directly
against this host's real `axcl_npu_type.h`) only expose a single aggregate
`nCMMSize` -- the older SDK's four-way mcode/CvPreProcessMcode/Wbt/Ringbuffer
breakdown isn't in the public host API for this generation, so the mapping
below comes from directly instrumenting real builds, not a queryable API.
(`axcl_ut_npu`'s own `Case12_AXCL_ENGINE_GetCMMUsage` test exists and passes
against a real device, confirming the call works at all, but reveals nothing
about the finer split without writing a custom harness -- not attempted,
low expected value for real device-state risk.)

**Wbt scales exactly linearly, 1280 bytes per identical op**: compiling ten
otherwise-identical single-input-single-output graphs with 1 through 10
sequential `Conv` layers (same shape throughout, `pads=[1,1,1,1]` to keep
every intermediate the same size) gives `npu_params` sizes of 1320, 2600,
3880, 5160, 6440, ... -- **every consecutive delta is exactly 1280 bytes**.
Confirms Wbt really is what its name says: a flat, uncompressed,
one-record-per-op concatenation of quantized weight (+ scale/bias) data,
not a smarter const-data store.

**mcode does *not* scale linearly -- but every delta is an exact multiple of
32 bytes**: the same ten builds' mcode (`<neu_key>` blob) sizes are 2984,
3248, 3408, 3440, 3568, 3600, 3728, 3792, 3888, 3920. Deltas from the second
step onward: 160, 32, 128, 32, 128, 64, 96, 32 -- every single one is `32 *
{1,2,3,4,5}`, never an in-between value, across 8 independent
measurements. (The very first delta, 2984 -> 3248 = 264, breaks the
pattern -- plausibly a one-time structural cost specific to a trivial
single-op graph, not re-tested further.) This is real, repeatable
structure: **mcode's command queue appears to allocate in 32-byte-aligned
units, with a variable (not fixed) number of units assigned per identical
op** -- consistent with `cmdq_write_instruct` emitting a different number
of queue entries per instance of the same op depending on scheduling
context (buffer/sync setup needs), rather than one fixed-size record per
op the way Wbt's constant stride would suggest.

**The growing region is not a simple append, even though the graph only
grows by appending one more `Conv`**: diffing consecutive builds' mcode
byte-for-byte finds only a **36-byte common prefix** (matches this
README's own already-decoded FlatBuffers root-table + vtable header) and a
**158-byte common suffix** (the tail string table, consistent with
FlatBuffers' bottom-up construction convention already documented above) --
identical across every pair tested. Everything in between differs
completely, even for the *unchanged* first N-1 conv layers' commands, not
just the newly-added one. Pulsar2's compiler re-serializes/re-addresses
essentially the whole command stream on any topology change (consistent
with cmdq entries containing absolute buffer addresses, job IDs, or
OCM-residency decisions that legitimately do shift when one more op is
added anywhere in the graph) rather than treating mcode as an append-only
log. This bounds what future differential analysis targeting the cmdq
record format itself can assume: byte-position stability across even
trivially-different graphs cannot be relied on outside the fixed
header/footer.

### The 32-byte mcode unit holds under shape variation too, and Wbt reveals real channel-tiling structure

The op-count sweep above varies *how many* identical ops there are; this
sweep instead holds op-count at 1 and varies *shape* -- output channels,
input channels, and kernel size, each independently, for a single `Conv` --
to separate "cost per op" from "cost per unit of tensor data."

**The 32-byte mcode finding generalizes**: every single-`Conv` shape change
tested still moves mcode's size by an exact multiple of 32 bytes -- cout
`{4,8,16,32,64}` gives deltas `{512, 32, 0, 64}`; cin `{4,8,16,32,64}` gives
`{192, 32, 0, 32}`; kernel size `{1,3,5,7}` gives `{-224, 928, 352}`
(k=1->3 is negative -- a smaller kernel with a *larger* mcode, see below).
19 independent measurements now, across two completely different kinds of
model variation (repeating an op vs. reshaping one), zero exceptions: mcode
appears to always serialize in exact 32-byte increments, a real structural
property of the format rather than an artifact specific to adding ops.

**Wbt's cost is *not* proportional to channel count -- it's flat across a
range, then jumps**, consistent with output-channel tiling:

```
cout:    4     8    16    32    64
Wbt:  1320  1448  1448  1448  2856
```

`Wbt` is identical across `cout` 8, 16, and *32* (1448 bytes each), then
roughly doubles at 64 -- consistent with an output-channel tile width of 32
(any count up to one tile's worth costs the same; a second tile is only
needed past it). It cleanly decomposes at the two largest sizes: `Wbt -
(cout * cin * k^2 [int8 weight] + cout * 4 [f32 per-channel scale] + cout *
4 [f32 per-channel bias])` equals **exactly 40 bytes** for both cout=32
(1448 - 1408 = 40) and cout=64 (2856 - 2816 = 40) -- a small, constant,
plausibly-a-tensor-header overhead once tiling effects are past. Below the
32-channel tile boundary (cout 4, 8, 16), that same subtraction gives 1144,
1096, and 744 respectively -- real, shrinking, but not yet explained by any
formula tried; `cout=4` is *smaller* than `cout=8`/`cout=16` even though
they're otherwise byte-identical to each other, an unexplained outlier at
the smallest size tested (the op-count sweep above hit the same kind of
"smallest case is different" edge, for what it's worth).

**Input channels show a similar flat-then-jump shape, but no clean floor
was found**:

```
cin:    4     8    16    32    64
Wbt: 1320  1256  1256  2408  4712
```

Flat across `cin` 8-16, roughly doubling at 32 and again at 64 (consistent
with an input-channel tile width of 16 -- half `cout`'s apparent 32,
plausible for a real MAC array with different input/output parallelism).
Padding `cin` up to the nearest multiple of 16 before applying the same
subtraction formula above gives a *matching* 648-byte residual for both
cin=8 and cin=16 (real, clean), but the residual keeps growing at cin=32/64
(1224, 2376) rather than settling to a constant the way `cout`'s did --
genuinely not resolved here; per-input-channel data (e.g. a real hardware
design can need per-input-channel handling that a purely per-output-channel
model like the `cout` case doesn't) is a plausible reason, not confirmed.

**Kernel size is the most surprising: not monotonic, and not `k^2`-scaled**:

```
k:      1     3     5     7
Wbt: 1448  1320  2472  4776
```

`k=1` costs *more* than `k=3` despite having 1/9th the raw weight data --
plausibly a real, different internal handling for 1x1 convolutions (closer
to a MatMul in some NPU designs) rather than the general conv path. `k=3`
to `k=5` to `k=7` grow much faster than raw element count would predict
(a `k^2` model predicts a 2.78x jump from 3 to 5; the real jump is 1.87x),
consistent with the kernel being broken into fixed-size sub-tiles (e.g. 3x3
blocks) with each additional tile costing a full tile's worth regardless of
how much of it the real kernel uses -- plausible, not confirmed with only
four data points.

**Bottom line**: the 32-byte mcode-serialization-unit finding is now solid
across two independent kinds of experiment. The Wbt tiling-granularity
story is real and reproducible (flat regions, clean doublings, an exact
40-byte per-tensor floor once large enough) but only partially explained --
a genuine, well-scoped target for further differential analysis, not a
closed question.

### How much of mcode do we actually understand, for a real model?

Everything above characterizes tiny, single-purpose synthetic graphs.
Applying it to a real, full-size, production model --
`resnet18d_Opset18` (56 nodes, 8 distinct ONNX op types: `Conv` x22,
`Relu` x19, `Add` x8, `AveragePool` x3, `MaxPool`, `GlobalAveragePool`,
`Flatten`, `Gemm`) -- gives a real, quantified, honestly small answer.

**Byte-level: roughly 1%.** This model's real compiled mcode blob is
49,080 bytes. Scanning it for the known FlatBuffers structure (the header,
vtable, and the tensor-name/`_offset` string table this README already
decoded, duplicated near the start and end per FlatBuffers' bottom-up
convention) accounts for only ~505 of those bytes -- the front copy spans
byte 0 to ~253, the back copy spans ~48,828 to the end. **The remaining
~48,575 bytes (>99%) are completely uninterpreted** beyond the general
"changes happen in 32-byte units" behavioral finding above, which says
nothing about what any specific byte in that region means. A blind scan
of that whole opaque region for embedded ASCII strings (the same technique
that found the tensor-name table) turns up nothing but statistical noise
(~20 spurious 3-4-byte "printable" runs, exactly what you'd expect by
chance in ~48KB of dense binary data) -- confirming there's no other
human-readable structure hiding in there for this technique to find.

**Primitive/op-level: about 1 of 7, and even that one is partial.**
Rebuilding the same model with `--profile` and inspecting the real
`trace.json` (see the profiling section above) shows the *dataflow* is
well understood -- but that's a different, much coarser layer than mcode's
actual byte encoding. The 1433 real scheduled events resolve to:

```
AxQuantizedConv          996  (69.5%)  -- Conv, with Relu fused in (see below)
LOAD_XXH128_DEDUP        343  (23.9%)  -- content-addressed weight DMA loads
"/fc/Gemm_*"              48  ( 3.3%)  -- the final FC layer, kept its ONNX name
AxQuantizedAdd            22  ( 1.5%)  -- the residual Add
AxMaxPool                   8  ( 0.6%)  (4 spatial-split sub-events x2)
AxQuantizedAvgPool           6  ( 0.4%)
AxQuantizedNormalize          4  ( 0.3%)  -- see below, not an ONNX op at all
AxQuantizedGlobAvgPool         2  ( 0.1%)
RTV_IO_EVENT                    2  ( 0.1%)  -- see below
Post_AxTranspose, final DequantizeLinear -- 1 event each
```

Of these 7 real distinct *compute* primitive families (`AxQuantizedConv`,
the FC `Gemm`, `AxQuantizedAdd`, `AxMaxPool`, `AxQuantizedAvgPool`,
`AxQuantizedNormalize`, `AxQuantizedGlobAvgPool` -- separate from the
housekeeping categories: weight-load dedup, RTV I/O, transpose, final
dequant), this whole investigation has only ever differentially decoded
**one narrow field of one of them**: `AxQuantizedConv`'s per-channel bias
array (two float32 arrays, one identified as a real requantization
multiplier -- see the differential-analysis section above). The dominant
primitive by far (`AxQuantizedConv`'s actual weight/compute encoding, 69.5%
of all scheduled
work) is not decoded at all, and `AxQuantizedAdd`, `AvgPool`, `MaxPool`,
`GlobalAvgPool`, `Normalize`, and the FC `Gemm` path have had zero
differential analysis directed at them specifically. (The earlier Add/Sub
differential analysis tested `Add`/`Sub` against a *constant* operand, not
`AxQuantizedAdd`'s real use here -- a residual add of two activations --
so it doesn't actually transfer to this real usage.)

**Real findings along the way, not previously documented**, even though
the underlying bytes remain opaque:

- **`Relu` produces no separate primitive or scheduled event at all** --
  it's fused directly into the preceding `AxQuantizedConv`, at zero
  additional schedule cost. Real, free activation fusion.
- **`Flatten` likewise produces nothing** -- a pure reshape, no compute or
  DMA needed.
- **Image normalization (the `calibration_mean`/`calibration_std` from
  `input_processors`, not part of the original ONNX graph at all) compiles
  to its own explicit primitive, `AxQuantizedNormalize`** -- confirming
  Pulsar2's input pre-processing pipeline is baked into mcode as a real,
  first-class compiled op, not a host-side step.
- **RTV events are real and present even for a plain CNN's ordinary
  tensor I/O** (`__rtv_x`, `__rtv_212`, one per graph input/output) --
  independent, real-model confirmation that the Runtime Variable mechanism
  from the public prior-generation SDK research above is still live in the
  current stack, and not restricted to ISP/CV use cases the way the public
  header's RTV enum values (mostly `WARP_*`/`HAAR_*`/`YDRC_*`) might
  suggest on their own.
- **Nearly a quarter of the entire real execution schedule (23.9%, 343 of
  1433 events) is weight-loading DMA, deduplicated by content hash**
  (`ld:xxh128:<hash>`) -- a real, previously-undocumented systems
  optimization: identical weight blocks (plausible for a residual network
  with repeated block structure) get loaded once and reused via
  content-addressing, rather than re-transferred per use.

**Bottom line**: this investigation understands the *container* (FlatBuffers)
and, separately, the real *dataflow schedule* (via `trace.json`) well. It
does not understand mcode's actual instruction/command encoding in any
usable sense for a real model -- at best a single narrow field of the
single most common primitive. Wbt is in better shape: its gross
composition (weight + per-channel scale + per-channel bias + a tiling
floor) is characterized, even though the bias array's own bit-level
quantization format was never independently re-derived either. Turning
"~1% of mcode's bytes explained" into real coverage would need the same
byte-diffing technique demonstrated for Conv's bias, scaled up to cover
`AxQuantizedAdd`/`AvgPool`/`MaxPool`/`GlobalAvgPool`/`Normalize`/`Gemm`,
and critically, `AxQuantizedConv`'s dominant weight/compute payload itself
-- each a real, well-scoped, but separately time-consuming target.

**Update, after the further mcode-focused sections below**: the ~1% figure
above is unchanged for raw "exact meaning known" bytes -- Conv's bias
lives in Wbt, not mcode, so none of that work adds to mcode's own count --
but the *map* of mcode is now materially more complete than "1% known,
99% blank":

- **Two distinct periodic fields are now precisely located inside a real
  `AxQuantizedConv` command** (not just Wbt) -- the original 4-repeats/
  7-byte-stride field, confirmed real and dilation/groups-sensitive across
  six independent experiments, plus a second, similarly-shaped field that
  only activates once dilation reaches 4. Neither is decoded at the bit
  level, but both are now real, reproducible targets with exact byte
  offsets, not part of the undifferentiated opaque mass.
- **The confirmed non-deterministic region needs no further decoding at
  all** -- it's understood to be a functionally-inert internal label
  permutation (bit-identical real device output regardless of which
  permutation a build lands on), not an encoded parameter. That's a small
  but real subtraction from the "mystery" pile: bytes whose *role* is now
  fully explained, even without knowing the exact label values' meaning.
- **A real, quantified bound on how much of mcode is even distinct**: the
  43.4%-of-bytes-are-exact-duplicates finding (from the self-similarity
  scan elsewhere in this README) means the effective amount of *unique*
  content to decode in a real model is well under its raw byte count --
  most of what's left unexamined is copies of a smaller number of real
  templates, not independent unique data.
- **A negative result narrows where to keep looking**: profiling a real
  two-op chain found no separately-scheduled "transfer" event for
  inter-op data movement, meaning whatever addressing the intermediate
  buffer needs is folded into the existing per-op command bytes rather
  than existing as its own, separately-findable region -- ruling out one
  plausible place further decoding might have focused on.

None of this changes the honest headline (still roughly 1% exactly
decoded, for a real model, and the dominant `AxQuantizedConv` payload
itself still opaque) -- but "what's left to figure out" is now a
materially smaller, better-characterized target than when this section
was first written.

### A first real crack at `AxQuantizedConv`'s command encoding

Taking up that target directly: since Conv's actual *weight values* live
in Wbt (already reasonably well understood), what's left opaque in mcode
for `AxQuantizedConv` is the *command* that invokes it -- stride, padding,
dilation, groups, and whatever addressing/scheduling those imply. Sweeping
each independently (single `Conv`, shape otherwise fixed) confirmed the
now-familiar problem: most pairs differ in total mcode length (a stride or
padding change usually changes the output tensor's declared shape too),
which triggers the wholesale re-serialization this README already
documented -- diffing two differently-sized blobs mostly shows *that*
effect, not the parameter's own encoding.

**The fix: pick pairs that happen to serialize to the *same total length*.**
`dilation=2` vs `dilation=3` (padding adjusted to keep output shape
identical, so *only* dilation differs) both compiled to exactly 3528
bytes; `groups=2` vs `groups=4` both compiled to exactly 3176 bytes. With
length held constant, a full byte-level diff (not just common-prefix/
suffix) is meaningful, and both pairs show real, tight, structured
differences instead of a wholesale rewrite:

- Two **byte-identical 94-byte and 60-byte blocks**, each appearing twice
  in the blob (at a fixed +608-byte separation for the dilation pair),
  containing a handful of scattered single-byte differences -- plausibly
  one shared command template instantiated once per major compute engine
  (`conv0`/`conv1` split evenly in this README's own `resnet18d` profiling
  above), each copy separately patched with a few dilation/groups-sensitive
  bytes.
- **A real, precisely-located periodic field**: exactly **4 repeats of a
  3-byte value, each occurring every 7 bytes**, changing identically with
  both dilation (`1a3b80`->`4b186f`, repeated at offsets 2561/2568/2575/
  2582) and groups (`924e5c`->`af9636`, repeated 7 bytes apart). The exact
  same 4x/7-byte-stride shape in two independent experiments is real
  structure, not noise.
- **A control experiment rules out the obvious explanation**: re-running
  the dilation pair with `cout=8` instead of 4 still shows *exactly 4*
  repeats (at the same 7-byte stride) -- so this field is not
  "one entry per output channel." It's much more likely one entry per
  **spatial tile**: this README's own `resnet18d` profiling above already
  found real primitives split into named `_s0`/`_s1`/`_s2`/`_s3`
  sub-events (`AxMaxPool`, `AxQuantizedNormalize`) -- independent evidence
  for a fixed 4-way spatial tiling convention in this compiler, which this
  new byte-level finding now corroborates from a completely different
  angle.

**Still not decoded**: the exact bit-level meaning of that 3-byte
per-tile field, or of the scattered single-byte differences inside the
94-byte/60-byte shared blocks. This is a real, precise *location* (mcode
byte offsets 2561-2601 in this specific build, always 4 entries at a
7-byte stride) for future work to target, not a solved encoding -- but
it's the first time this investigation has isolated a small, structured,
non-header/footer region of `AxQuantizedConv`'s own command at all,
against a real production-scale finding (4-way spatial tiling) rather than
a synthetic-model artifact.

### Does this transfer to `resnet18d` itself? A negative result, and a bigger positive one

The natural next question: does the periodic-field finding above, found on
a tiny single-`Conv` synthetic model, actually show up inside a real,
22-`Conv` model's mcode? Tested directly: edited `resnet18d_Opset18`'s own
ONNX graph in place, changing exactly one real layer's
(`/layer1/layer1.0/conv1/Conv`) `dilations` from `[1,1]` to `[2,2]`
(padding adjusted to `[2,2,2,2]` to keep output shape, and therefore every
downstream layer, identical), and rebuilt for real -- weights untouched, a
purely structural edit. It compiled successfully (`max_cycle` 1,318,764 ->
1,327,463, a plausible small increase for one costlier layer).

**Negative result: the same-total-length trick that made the synthetic
diff clean does not carry over.** Unlike the tiny model (where a careful
padding choice reliably produced an identical total mcode length), editing
one internal layer of a real 22-layer network changed the *whole* blob's
length -- 49,080 -> 49,912 bytes. A second attempt (`dilation=3`, `pad=3`
on the same layer) gave a third, still-different length (50,808). Neither
of the two edited variants matches the baseline or each other, so the
clean "diff the whole blob" technique that worked on the synthetic model
doesn't directly apply here -- a single internal attribute change cascades
into a differently-sized whole program in a way that (at least on the two
tries attempted) doesn't coincidentally realign. This is itself a real,
useful negative result: it bounds how far the synthetic-model technique
generalizes on its own, without a smarter localization method or a lot of
brute-force retrying.

**A different technique gives a much bigger, and arguably more useful,
positive result.** Since diffing two *different* builds didn't work
cleanly, look for self-similarity *within* the one real, unmodified
`resnet18d` mcode blob instead -- no second build needed. Scanning it for
exact-duplicate byte windows (any substring that occurs more than once)
turns up an amount of internal repetition that is astronomically
impossible by chance: 1,225 distinct duplicated 32-byte windows (4,138
total window instances, out of only ~49,000 possible positions -- for
context, a truly random 49KB blob would have a vanishingly small chance of
containing even one repeated 32-byte sequence, let alone thousands, since
there are `256^32` possible values). Extending every matching seed to its
maximal exact-duplicate run and merging overlaps gives a precise, real
number: **21,289 of the blob's 49,080 bytes (43.4%) are byte-for-byte
identical to some other span elsewhere in the same file**, including one
exactly-matching run of 177 bytes (at offsets 278 and 10,994) and dozens
more in the 40-90 byte range.

This is real, direct, large-scale confirmation of the shared-command-
template hypothesis the tiny synthetic model first suggested (the 94-byte/
60-byte blocks repeated exactly twice, one guess being "once per compute
engine") -- just demonstrated a different way, and at a scale (43% of a
real model's mcode) that makes clear most of a real compiled program's
bytes are copies of other bytes in the same file, not each independently
carrying unique information. It does **not** mean 43% of mcode is
*understood* -- none of these repeated templates have been decoded either,
this only establishes that they repeat -- but it substantially shrinks the
amount of genuinely distinct content anyone would need to decode to cover
the rest: on this evidence, well under 49,080 bytes' worth of *distinct*
templates, not 49,080 bytes of independent unique data.

### Decomposing a real model into a few real ops at a time: real extraction works, but doesn't restore localization on its own

A natural idea, given the last section's negative result: rather than
editing the whole 22-`Conv` `resnet18d` at once, cut *real* subgraphs
(genuine topology and trained weights, via `onnx.utils.Extractor`) out of
it -- small enough, hopefully, for the clean same-length diffing that
worked on a synthetic single-`Conv` model to work again, while still using
representative real weights and shapes instead of made-up ones.

**Real subgraph extraction itself works cleanly**: `Extractor(model)
.extract_model([input_name], [output_name])` on a shape-inferred
`resnet18d` produces valid, checker-passing standalone ONNX graphs for any
internal cut -- confirmed for a single real stem `Conv` alone
(`/conv1/conv1.0/Conv`, real 3->32-channel trained weights) and a full real
5-op residual block (`Conv`/`Relu`/`Conv`/`Add`/`Relu`, real 64-channel
weights, real skip-connection `Add`) with only real, existing tensor names
as new inputs/outputs. Both compile through the real toolchain
unmodified.

**But five separate attempts at reproducing the clean, same-length
dilation trick on real-weight single-`Conv` slices all failed**, each
ruling out one plausible cause: the real stem `Conv` (cin=3, cout=32) at
its real 224x224 input size (4568 -> 4600 bytes, off by 32); the *same*
real `Conv` resized down to 16x16 -- matching the earlier tractable
synthetic test's spatial scale exactly -- to rule out spatial size (3632 ->
3664, still off by 32, still a wholesale-rewrite-scale diff underneath);
a different real layer's `Conv` (cin=cout=64, both powers of two, clear of
the tiling-boundary behavior the `cout` sweep found near 32) to rule out
channel-count "unfriendliness" (3440 -> 4152, off by 712); the same, with
its bias input stripped entirely, to rule out bias presence (identical
result, 3440 -> 4152); and the same cin=cout=64 slice compared at dilation
2 vs 3 specifically (not 1 vs 2), to exactly match the earlier synthetic
test's own dilation transition rather than assuming any transition is
equivalent (4152 -> 4888, off by 736). **None matched.**

**A sixth, decisive test ruled out weight values too, and points back to
shape after all.** Before concluding it was about real-vs-random weight
*values*, that was tested directly: the cin=cout=64 slice, same dilation
2-vs-3 transition, but with its real trained weights replaced by i.i.d.
random Gaussian weights (matching the one synthetic test that *did* work,
same generation code, different shape). Result: **4152 -> 4888 bytes,
byte-for-byte the same sizes as the real-weight version of the same
shape.** Weight values -- real or random -- made no difference whatsoever
at this shape. That rules out the weight-values explanation cleanly: it's
the shape itself (cin=cout=64) that reliably produces a mismatched pair,
independent of what's actually in the tensors.

**So the honest conclusion is narrower than either single-cause story**:
whether a same-length pair exists is a property of *shape* (channel
counts, at least, since spatial size, bias presence, and weight values
were all ruled out above) -- but not in a simple "friendly vs. unfriendly
channel count" way either. Of the three distinct channel-count
combinations tested for this exact question, only the original synthetic
test's cin=cout=4 produced a match; cin=3/cout=32 and cin=cout=64 both did
not, across six separate real dilation-pair experiments in this section
alone. On the evidence gathered so far, a matching pair looks like a coincidental
alignment of whatever tiling arithmetic the compiler runs for that
specific shape, not a systematically reachable property -- there may be a
real, discoverable rule underneath (the earlier `cout`/`cin`
tiling-granularity sweeps found real structure in a related question, just
not this one), but it wasn't found here.

**Bottom line for "decompose into a few ops to expand coverage"**: the
*technique* (real subgraph extraction) is validated and reusable --
`Extractor` cleanly produces small, real, checker-valid, toolchain-
buildable slices of any real model, letting future differential analysis
target real ops with real weights instead of only synthetic ones. But "few
ops" alone does not by itself restore the localized-diff property that
made the earlier decoding progress possible -- across every real shape
tried from `resnet18d` (six separate dilation-pair experiments), none
reproduced a same-length pair the way the one tiny synthetic shape
happened to. Making
further progress this way would need either a systematic shape sweep large
enough to find which specific `(cin, cout, ...)` combinations do produce
matching pairs (if any beyond the one already found), or abandoning
same-length localization in favor of mining un-localized diffs directly
(as the resnet18d self-similarity scan above did, successfully, without
needing localization at all).

### Chaining two real convs: genuine inter-op data transfer, and a real cross-op coupling finding

Every experiment up to this point used a *single* op -- no genuine
inter-op data transfer, since the one op's input/output are both graph
boundaries, addressed through the tensor I/O offset table this README
already decoded. Chaining two ops introduces a real, so-far-unexamined
piece: the intermediate activation between them lives entirely inside the
compiled program's own OCM/DDR addressing, with no external name to look
up. Testing this directly: two chained `Conv`s (`cin=cout=mid=4`, the one
shape combination confirmed to give same-length pairs) with data
genuinely flowing from the first op's output into the second op's input.

**The same-length trick survives the jump from one op to two, at this
shape.** Varying only the first `Conv`'s dilation (2 vs 3, padding
compensated to hold every shape downstream constant) gave two builds at
an identical 3,920 bytes; varying only the *second* `Conv`'s dilation
(first held completely fixed) gave two builds at an identical 3,952
bytes. Both diffs are real and structured, not wholesale rewrites --
confirming the technique generalizes past a single op, at least for this
shape.

**Correction (see "Is mcode deterministic?" below): the offset-858-876
claim originally made here was wrong.** This section first reported a
"new" diff pattern at byte offset 858-876 (four single-byte changes at a
6-byte stride) as a candidate for the intermediate buffer's own
addressing. Directly rebuilding the *identical* config three times (no
parameter changed at all) later confirmed that exact byte region is
**non-deterministic noise**, not a dilation-dependent signal -- rebuilding
the very same `two_conv_d2` model three times in a row produced three
different mcode blobs, differing only at these same 4-6 byte positions
each time. That fully explains the "new content" that seemed to appear
here: it was never caused by chaining two ops, it was present (and just
as spurious) even with nothing changed between builds. Left in place with
this correction rather than silently rewritten, since it's a real example
of a finding this project got wrong before checking determinism -- see
below for what's actually confirmed stable.

**An unexpected, genuinely new finding: an op's own command bytes are not
independent of *downstream* ops.** Varying the *second* conv's dilation
while leaving the first conv's own attributes completely untouched still
changes bytes in the region corresponding to the *first* conv's own
per-op command (the same relative area the 94-byte/60-byte templates
occupy) -- not just adding new bytes near the second conv. An op that
didn't itself change still gets re-encoded because something later in the
graph did. This sharpens (and partly explains) the earlier "mcode isn't
append-only" finding: it's not only that changing a graph's *topology*
forces a wholesale re-serialization -- even within a fixed, matching-size
two-op program, one op's encoding depends on what happens after it, not
just on its own attributes and its own inputs.

**The periodic field looks genuinely global, not per-op.** The same
4-repeats/7-byte-stride signature appears in *both* experiments --
varying conv1's dilation and varying conv2's -- landing at different
absolute offsets (2727 vs 2759) simply because the two builds have
different overall sizes, but with the identical shape otherwise. A field
that reacts to a dilation change no matter which of the two convs it
belongs to is further, independent evidence for this being a shared,
graph-wide resource (plausibly the 4-way spatial-tiling table this
README's `resnet18d` profiling already found evidence for) rather than
something scoped to one specific op's own command.

The periodic field and the cross-op coupling finding are both confirmed
stable under repeated identical builds (see below) -- real, precisely-
located targets for future work, not solved encodings, but not noise
either. Whether connecting two ops surfaces genuinely *new* content tied
specifically to the intermediate buffer's own addressing remains an open
question -- the one candidate found here didn't hold up.

### Is `.axmodel` deterministic? No -- and that correction above is why this matters

Every differential-analysis finding in this whole investigation assumes
that rebuilding the *same* model with the *same* config twice produces
the *same* bytes, so any observed diff is caused by the one thing that
changed. That assumption was never directly checked until it produced a
wrong finding (immediately above). Checked properly now, by rebuilding
one exact model/config three separate times with no changes at all:

- **`Wbt` (`npu_params`) is fully deterministic**: byte-identical
  (matching SHA-256) across three independent builds, every time tested.
- **`mcode` is *not* fully deterministic**: three rebuilds of the
  identical `two_conv_d2` model produced three different mcode blobs
  (same length, 3,920 bytes, every time -- only the *content* differs).
  The non-determinism is small and localized, not pervasive: 3-4 bytes
  differ per pair of runs, always at the same handful of positions (byte
  offsets 858/864/870/876 in that specific build), cycling through what
  looks like a small fixed set of values (`0x10`/`0x20`/`0x30`/`0x40`) in
  different orders each time -- consistent with a non-deterministic
  assignment of interchangeable resource/job IDs (which of several
  equivalent parallel slots gets which label) rather than genuinely
  random data corruption. The same experiment on the single-`Conv`
  dilation model (from earlier in this README) found the same thing, in
  the same relative region (offsets ~859-882), plus one additional
  isolated stray byte elsewhere (offset 3232, differing in only one of
  two run-pairs) -- confirming the non-determinism isn't confined to one
  specific model shape.
- **The overall `.axmodel` file is never byte-identical across rebuilds**
  even though `Wbt` alone is -- three rebuilds of the same config gave
  three different file SHA-256 hashes at the same file size, entirely
  because of `mcode`'s non-determinism above (nothing else in the file
  differed).

**Practical impact, checked directly rather than assumed**: the two
already-committed regression tests that depend on comparing mcode across
builds (the periodic 4-repeats/7-byte field, and the cross-op coupling
finding) were both re-examined against the confirmed noisy byte ranges
above and neither overlaps with them -- the periodic field sits at a
completely different offset range in every build tested, and the cross-op
coupling test only inspects the first 800 bytes, entirely below where the
noise was ever observed to start (858+ in every model tested so far).
Both findings hold up. The one finding that *did* turn out to be an
artifact (the "new content at 858-876" claim above) is the one case where
this wasn't checked before publishing it -- corrected in place rather
than removed, as a real example of why this check matters for any future
differential-analysis claim in this space: a same-length, structured-
looking diff is not automatically signal, and this non-determinism is
exactly the kind of thing that can masquerade as one.

**Where does the non-determinism actually come from -- metadata, or the
mcode generation algorithm itself?** Checked directly rather than
guessed, by looking at *which* values appear at the noisy positions, not
just that they differ. For the clean `two_conv_d2` case (four single-byte
positions, offsets 858/864/870/876), the exact values seen across all
three rebuilds are:

```
run1: 858=0x10  864=0x30  870=0x20  876=0x40
run2: 858=0x30  864=0x20  870=0x40  876=0x10
run3: 858=0x20  864=0x40  870=0x30  876=0x10
```

**Every single run has the identical multiset `{0x10, 0x20, 0x30,
0x40}`** at these four positions -- only *which position gets which
value* changes. This is decisive: it's the unmistakable signature of a
fixed, small set of interchangeable labels (plausibly per-tile or
per-job identifiers, given this project's other evidence for 4-way
spatial tiling) being assigned to four equivalent slots in a
non-deterministic *order* -- consistent with iteration over an unordered
container (hash-map/hash-set bucket order depending on pointer values or
ASLR) or parallel-task completion order in the compiler, not with
embedded metadata. A real timestamp, build UUID, PID, or similar tracking
value would need to reproduce the *exact same four values* across three
independent builds run at different wall-clock times, just shuffled --
astronomically unlikely for anything resembling real metadata, and
trivially expected for a label-assignment race. The messier single-`Conv`
case (a wider, ~24-byte noisy region rather than four isolated bytes)
didn't resolve to as clean a single-byte permutation on inspection, but
occupies the same narrow relative region and is consistent with the same
underlying mechanism at a different granularity (e.g. multi-byte records
being reordered rather than single label bytes) rather than a second,
unrelated source. No timestamp-like field (a large, monotonically
distinct value) was found anywhere in either the noisy region or the rest
of the file across any of the rebuilds -- the file's own metadata-shaped
fields (`version`, `neu_name`, the JSON attributes) were separately
confirmed identical across every rebuild in this section.

### What a real, profiled two-conv chain's trace.json actually shows

Following up on "does data transfer between the two convs show up as its
own event," the `two_conv_d2` model was rebuilt with `--profile` to check
directly rather than infer from mcode bytes alone. The real trace (17
events total, comparable in structure to the `resnet18d` profiling
elsewhere in this README) shows:

- **No separate event for the inter-op transfer at all.** The first
  `Conv`'s last scheduled event on the `conv1` engine ends at the exact
  timestamp the second `Conv`'s first event on `conv1` begins (`0.63475 +
  0.575 = 1.20975`, matching to the fifth decimal place in the raw trace).
  There is no gap, no separate "copy"/"transfer"/"sync" event between
  them. The hand-off is either free (same engine, same OCM location,
  nothing to move) or its cost is folded into one of the adjacent events
  rather than broken out on its own.
- **Asymmetric engine usage between the two convs**: the first `Conv` runs
  on *both* `conv0` and `conv1` in parallel (matching pairs of events at
  identical timestamps on each); the second `Conv` runs on `conv1` alone
  -- `conv0` does nothing after the first `Conv` finishes. Not every op in
  a chain gets the same 2-engine split this README's `resnet18d` profiling
  showed for that model's convs; whether an op is split across both engines
  or run on just one is itself a real scheduling decision with no
  visible cost model exposed here. This asymmetry is also a plausible
  partial explanation for the mcode size non-linearity this README's
  op-count sweep found earlier (a 32-byte-unit delta per added op, but not
  a *constant* one) -- not every op costs the same number of engine-command
  copies.
- **Both weight-load events happen up front, at `ts=0`**, one per real
  weight tensor (`ld:xxh128:...` on `cv3` and `sdma4`) -- not interleaved
  between the two convs' execution the way a naive "load weights right
  before you need them" schedule might. Confirms weight loading is
  planned globally ahead of compute, consistent with the content-addressed
  weight-load-dominates-the-schedule finding from the `resnet18d`
  profiling elsewhere in this README, just at a much smaller scale (2
  distinct real tensors here, nothing to deduplicate against each other).
- **RTV events fire for both the graph input and the graph output**
  (`__rtv_x`, `__rtv_y`) -- consistent with, and now confirmed for a
  genuinely multi-op graph (not just the single-op case checked
  previously), the finding that RTV isn't scoped to ISP/CV use cases.

Net effect: the profiler confirms there's no dedicated, separately-timed
"transfer" step to go looking for in mcode -- if the intermediate buffer's
address/size needs to be encoded anywhere (and it must, since the two
ops' reads and writes have to agree on where it lives), it's folded into
one of the two convs' own commands rather than existing as its own
identifiable unit, which is a real, useful negative constraint on where
to look next.

### Following up on determinism: the label noise is functionally harmless, and stays small at real-model scale

Two direct follow-ups to the determinism finding above, both confirmed
rather than assumed.

**Does the byte-level non-determinism actually change the computed
result?** Ran all three of the earlier `two_conv_d2` rebuilds (same
config, three different mcode blobs due to the label-permutation noise)
on the real AX650N with the identical input. **All three produce
bit-identical output** (`np.array_equal` `True` pairwise, max absolute
difference `0.0`, across every element checked). The non-deterministic
label reassignment this README already traced to a small set of
interchangeable slot IDs really is cosmetic at the semantic level --
whichever arbitrary label a slot gets, the hardware executes the same
computation. This is reassuring for the toolchain generally (rebuilding
a real model doesn't silently change its answers), and it sharpens what
"non-determinism" means here: an internal bookkeeping artifact with a
confirmed-zero behavioral footprint, not a source of real output
variance.

**Does the non-determinism stay this small at real-model scale, or could
it explain the large size deltas found when decomposing `resnet18d`
earlier?** This mattered directly: that earlier section's negative result
rested on real dilation edits changing `resnet18d`'s mcode length by
700-900+ bytes, and it's worth checking that isn't just inflated noise at
a bigger scale. Rebuilding the real, *unmodified* `resnet18d` twice (no
parameter changed) gives: **`Wbt` byte-identical** (as at small scale),
and **mcode's total length exactly stable at 49,080 bytes both times**,
with only **16 bytes of internal noise** out of 49,080 -- proportionally
smaller, not larger, than the small synthetic models' noise. This
confirms the earlier `resnet18d` decomposition experiments' 700-900-byte
size deltas are real, parameter-driven effects, not an artifact of
non-determinism growing with model size -- the negative result there
stands.

### Extending the periodic field across a wider dilation range: real values, no simple formula yet, and a new threshold effect

With determinism now understood well enough to trust same-length diffs
again, the periodic 4-repeats/7-byte-stride field was pushed further:
built the single-`Conv` dilation model at dilation `{2,3,4,5,6}` (padding
compensated each time). Four of the five (`2,3,4,6`) happened to
serialize to the same 3,528-byte length, letting all six pairs be
compared directly; `5` serialized to a different length (3,560) and was
left out of this specific comparison.

**The field's value is real and dilation-dependent across the whole
range tested, but doesn't reduce to an obvious arithmetic function of
dilation alone.** The confirmed field (still at the same relative
position, offset 2561 in this build) takes a different 3-byte value for
every dilation compared against the baseline (`d=2`: `1a3b80`; `d=3`:
`4b186f`; `d=4`: `244082`; `d=6`: `550f5b`) -- real signal, not noise
(this exact offset never appeared among the confirmed noisy positions in
any determinism check above), but neither the full 3-byte value nor its
individual bytes move monotonically or linearly with dilation. Consistent
with this project's earlier finding that a structurally similar
undecoded field (Add/Sub's non-trivial encoding) depends on
*calibration-derived* values rather than the raw attribute directly, this
field plausibly encodes something computed from the dilated receptive
field's effect on quantization ranges, not the integer dilation value
itself -- a real, motivated hypothesis, not yet confirmed.

**A new, real, threshold-like effect**: comparing `d=4` and `d=6` against
the `d=2` baseline (but *not* `d=3` vs `d=2`) surfaces a second pair of
matching 3-byte runs at entirely different offsets (552/1160 for `d=4`;
754/1362 for `d=6`) that don't exist at all in the `d=3`-vs-`d=2`
comparison. Same shape as the already-known field -- a value repeated
identically twice, at a fixed separation -- but a distinct location that
only activates once dilation reaches 4. A plausible explanation: larger
dilation values push the effective receptive field past some internal
boundary (an OCM tile edge, or a maximum supported "trivial" receptive
field size) that requires an extra encoded record once crossed --
consistent with the general pattern this README has already found of
Pulsar2's compiler behavior changing in threshold/tile-boundary ways
rather than smoothly. Not chased further here, but a real, precisely
reproducible lead (two exact offsets, two exact dilation thresholds) for
future work.

### Two more Conv attributes tried: a real asymmetry, and a second non-determinism zone found by a false lead

Continuing to work through Conv's remaining untested attributes:

**Asymmetric kernel shape (`3x1` vs `1x3`) reveals a real, new asymmetry.**
Same total weight count either way (`cin*cout*3*1 == cin*cout*1*3`), so
`Wbt` came out byte-identical in size (1,320 bytes both) as expected --
but mcode did *not*: 3,528 bytes for `3x1` vs 3,208 bytes for `1x3`, a
real 320-byte (10-unit) difference driven by orientation alone, not by
how much weight data there is. A genuine, real finding: the compiler
treats a "tall" and a "wide" kernel of otherwise identical size
differently, plausibly because how it scans/tiles the input differs by
row vs. column direction. Not further decoded (no same-length pair here
to diff cleanly), but a real, motivated target for whoever chases the
scanning-order encoding next.

**`auto_pad="SAME_UPPER"` vs. the numerically-equivalent explicit `pads`
looked at first like a real, tiny signal -- and turned out to be a false
lead that found something else useful instead.** Both compiled to the
identical 2,984-byte mcode length, and diffing them found only 4 bytes
different, at a location (offsets 301/303/323/325) never seen in any
prior section of this README -- a plausible candidate for auto_pad
leaving some small trace even after normalization. **Checked properly
before believing it**: rebuilding the *`auto_pad="NOTSET"` config alone*,
twice, with nothing changed, reproduced a nearly identical diff pattern
(5 bytes, offsets 301/303/317/319/325, same multiset-of-values signature
as the already-confirmed non-determinism elsewhere in this README). The
"auto_pad signal" was never real -- it was this project's second
encounter with the same class of non-deterministic label noise, just at
a location not seen before. **Real, useful takeaway**: `auto_pad` appears
to fully normalize to its explicit-padding equivalent before
quantization, with no detectable functional difference in mcode --  a
clean negative result, now that the false positive has been ruled out.

**More importantly, methodologically**: this confirms mcode's
non-determinism is not confined to the one zone (offsets ~858-882)
characterized earlier -- there are at least two independent noisy
regions (~301-325 as well), and likely more not yet stumbled into. Any
future same-length-diff finding in this space needs its own determinism
check (rebuild the *unchanged* config and confirm the observed diff
isn't reproduced by noise alone) before being trusted, not just a check
against the two zones already known -- this project got this exact kind
of false positive twice now, at two different locations.

### Expanding past Conv: MaxPool, the real residual Add, and GlobalAveragePool, cross-checked against `--profile`

Every mcode structural finding so far came from `Conv`. Directly
extending coverage to three of `resnet18d`'s other real primitive
families found in its own `trace.json` profiling earlier in this
README -- `AxMaxPool`, `AxQuantizedAdd` (the real residual add of two
activations, not the old constant-broadcast `Add` tested early in this
investigation), and `AxQuantizedGlobAvgPool` -- using small controlled
models and, per the user's suggestion, checking `--profile` metadata
alongside mcode bytes this time rather than bytes alone.

**A real, clean architectural split by execution engine, confirmed
directly**: `AxMaxPool`, the real residual `AxQuantizedAdd`, and
`AxQuantizedGlobAvgPool` all schedule on **`teng2`** (the same engine
`AxQuantizedNormalize` used in the `resnet18d` profiling earlier) --
never on `conv0`/`conv1`, which are reserved for `AxQuantizedConv` and
`Gemm`-family MAC work. A clean, real, generalizable rule confirmed
across four distinct primitive types now: **non-MAC ops run on `teng2`;
MAC ops run on the conv engines.** `Pre_AxTranspose`'s own engine
placement is context-dependent -- `cv3` when it wraps a `MaxPool` or
`GlobalAvgPool`, `teng2` when it wraps a `Conv` in the residual-block
test -- a real difference not chased down further here.

**`MaxPool`'s `ceil_mode` is a third confirmed false lead, same
signature as before.** Comparing `ceil_mode=0` vs `ceil_mode=1` on a
shape where both give the identical output size produced a same-length
pair with 5 differing bytes -- but a determinism check (rebuilding the
`ceil_mode=0` config alone, twice) reproduced the same positions and the
same exact multiset of values (`{0x13, 0x20, 0x23, 0x30, 0x40}`) already
seen in the `auto_pad` false lead. **This is the same noise signature
appearing a third time, now confirmed on a completely different op type
and model** -- strong, direct evidence this non-determinism is a global
property of mcode generation, not tied to Conv, to any one model shape,
or to any one byte region.

**`MaxPool`'s kernel size behaves like Conv's did**: comparing a `2x2`
kernel against a `4x4` kernel with padding chosen to hold the same output
shape gives *different* total mcode lengths (2,408 vs 2,440 bytes, a
32-byte-unit-consistent delta) -- real, but not a same-length pair, so no
clean localized diff was possible here the way the dilation experiments
allowed for Conv.

### `Gemm` joins the MAC engines, and a real, substantial signal from `transB`

Continuing to work through resnet18d's remaining real primitives: `Gemm`
(the final FC layer, 3.3% of resnet18d's real schedule) and
`AveragePool` (distinct from `GlobalAveragePool`, used 3x in
`resnet18d`'s real downsample paths).

**`Gemm` schedules on `conv1`** -- joining `Conv` in the MAC-engine
category rather than `teng2`'s non-MAC group, extending the same clean
split found above to a second op type. Its trace events keep the output
tensor's own name (`y_0_0`, `y_0_1`, `y_0_2`) rather than being renamed to
an `AxQuantized*`-style primitive the way `Conv`/`Pool`/`Add` are --
matching, exactly, the un-renamed `"/fc/Gemm_*"` event names already seen
in this README's real `resnet18d` profiling. **`AveragePool` schedules on
`teng2`**, joining `MaxPool`/`Add`/`GlobalAvgPool`/`Normalize` in the
non-MAC category, but with a real difference from `MaxPool` at the same
input size: two sub-events (`AxQuantizedAvgPool_0_0` and
`_1_0`) rather than one -- plausibly a real two-pass sum-then-divide
structure specific to averaging, not investigated further here.

**Correction, caught the same way the `auto_pad` false lead was**: this
section first reported this `Gemm` shape as showing *zero*
non-deterministic noise across a rebuild, based on a single rebuild pair.
That turned out to be a lucky draw, not a real property of the shape --
a second, independent pair of rebuilds (done while writing an automated
regression test for this finding, which caught the discrepancy) showed
the familiar ~6-byte noise at the same `~301-325` zone already confirmed
for `Conv`/`MaxPool`/`auto_pad`/`ceil_mode`. This shape is not
noise-free after all; it has the same known noise as everything else
tested so far.

**`transB=0` vs `transB=1` still produces a real, substantial signal, net
of that noise.** Of the 95 originally-reported differing bytes, the two
at offsets 319/325 fall inside the confirmed noisy zone and are not
trustworthy as `transB`-specific signal on their own (this exact
`gemm_base` config's own noise realization could easily land differently
there by chance alone, independent of `transB`). The remaining ~93 bytes
sit well outside any confirmed noise zone and hold up as real: a large,
clean 85-byte contiguous block (offset 1620-1705, containing non-trivial
repeated structure -- `f129ff3b81` and `f5852513c8` each appearing three
times) plus smaller diffs at 1617, 1708, and 1712. Consistent with a real
per-tile or per-output-column memory-access-pattern encoding that has to
change because `transB` genuinely changes whether the weight matrix is
traversed row-major or column-major. This remains, net of the correction,
by a wide margin the largest and cleanest real signal isolated in this
whole investigation -- a strong, well-motivated, precisely-located target
(offset 1620, 85 bytes) for whoever attempts the next level of decoding.
Also a second, independent confirmation of the broader determinism
lesson: a single rebuild pair is not enough to call something noise-free,
and this project has now made and caught that exact mistake twice.

### `Conv`'s `group` attribute: grouped convolutions parallelize across both MAC engines, dense ones don't

Untested until now: `Conv`'s `group` attribute (depthwise/grouped
convolution, common in real MobileNet/ResNeXt-style architectures, though
not `resnet18d` itself). A dense `Conv(cin=4, cout=4, 3x3, group=1)` and a
grouped version at the same shape (`group=2`, and full depthwise
`group=4`) all build successfully -- no compiler crash the way `Mul`/`Div`
by 1.0 hit earlier -- but schedule genuinely differently, confirmed stable
across two independent rebuilds of each config:

- **`group=1` (dense): 3 sub-events, all on a single engine (`conv1`).**
  Same one-engine pattern already seen for ordinary `Conv`/`Gemm` at small
  shapes elsewhere in this README.
- **`group=2` and `group=4` (any grouped conv): 6 sub-events, split evenly
  across *both* `conv0` and `conv1`** (3 each). This is a real, binary
  split on "is this Conv grouped at all," not something that scales with
  the group count -- `group=2` (2 groups) and `group=4` (4 groups, full
  depthwise) produce the identical 6-event, both-engines pattern, not 2 vs.
  4 proportional sub-events. Consistent with the compiler parallelizing a
  grouped conv's independent groups across the two MAC engines as a fixed
  strategy, rather than a per-group unit of scheduling.
- mcode grows despite Wbt shrinking: at this shape, dense `group=1` has a
  1320-byte Wbt / 2984-byte mcode, while full depthwise `group=4` has a
  *smaller* 1256-byte Wbt (fewer real weight values: `4*1*3*3=36` vs.
  `4*4*3*3=144`) but a *larger* 3176-byte mcode (+192 bytes, +6.4%) --
  the extra command bytes are the cost of coordinating two engines instead
  of one, not weight-data volume.

This is now regression-tested (`test_grouped_conv_splits_across_two_mac_engines_dense_does_not`)
via `--profile` engine/event-count assertions, following the same
determinism-checked pattern as the rest of this file.

### Does the two-engine split transfer to `resnet18d` itself? Yes, but it corrects the framing above

The natural next question for the finding above (per this README's own
established pattern -- see "Does this transfer to `resnet18d` itself?"):
does a real, unmodified `resnet18d_Opset18` build's `--profile` trace show
the same single-engine-vs-two-engine split? Checked directly against a
real profiled build (`convert_onnxmodelzoo.py --models resnet18d_Opset18
--profile`, real AX650N, confirmed bit-identical device output between the
original and onnxsim-simplified model as usual): **all 15 of
`resnet18d`'s distinct real Conv ops schedule on *both* `conv0` and
`conv1`** -- none stays on a single engine, even though `resnet18d` has no
grouped convolutions anywhere in it (confirmed above: this architecture
doesn't use `group>1`).

**This means the "grouped vs. dense" framing above was incomplete, not
wrong.** Grouping isn't the underlying trigger for the two-engine split;
channel count is, and the earlier experiment's `cin=cout=4` dense baseline
just happened to sit right at the edge of a real, sharp threshold.
Isolated directly by sweeping `cin=cout` for an otherwise-identical dense
(`group=1`) `Conv`, confirmed stable across independent rebuilds at both
ends: **`cin=cout<=4` stays on a single engine (`conv1`, 3 sub-events);
`cin=cout>=5` splits across both `conv0` and `conv1` (6 sub-events)** --
a precise, real, and surprisingly small cutover point. Every real
`resnet18d` layer has far more than 5 channels (64 to 512), so all 15
land unconditionally on the two-engine side of this threshold -- fully
explaining the profiled result without needing any grouping-specific
mechanism.

Reconciling both findings: dense convs cross into the two-engine regime
once total channel count passes this small threshold, while *any* grouped
conv (confirmed down to `group=2`/`group=4` at only 4 total channels,
`cin=cout=4`/`cin_per_group=1`) crosses into it regardless of size --
two independent triggers for the same underlying two-engine scheduling
strategy, not one unified rule. This refines, rather than replaces, the
earlier regression test: `test_grouped_conv_splits_across_two_mac_engines_dense_does_not`'s
`group=1` case is still correctly single-engine, precisely because it
was chosen at `cin=cout=4` -- right at (not below) the real threshold.
A second test now locks in the size-threshold side of this directly.

**`Gemm` has the same two-regime split, but at a much higher, distinct
threshold.** Checked directly against a real profiled resnet18d fc layer
(`k=512, n=1000`, the real shape): 48 sub-events split across both `conv0`
and `conv1` (16 tiles x 3 sub-events), matching the real trace exactly.
But unlike `Conv`, neither `k` nor `n` alone drives it: `Gemm(k=512,
n=16)` and `Gemm(k=16, n=512)` -- each with 8,192 weight elements --
*both* stay on a single engine, while `Gemm(k=n=256)` (65,536 elements)
splits and `Gemm(k=n=128)` (16,384 elements) does not, confirmed stable
across independent rebuilds at both ends. So `Gemm`'s cutover needs a much
larger, roughly-square shape to trigger, sitting somewhere in the 128-256
range for `k=n`, in clear contrast to `Conv`'s tiny 4-vs-5-channel
threshold. This is a real, confirmed difference in the two ops' tiling
strategies, not a single formula ported across op types -- reported
honestly as "a real threshold exists, at a different scale per op," not
as a unified quantity (candidates like raw weight-element count and
output-element count were both checked and neither cleanly explains both
ops' thresholds together).

### Beyond passive diffing: running our own hand-patched mcode on real hardware

Everything above (and in every earlier section) only ever *observes*
Pulsar2's own compiler output -- building variant ONNX graphs and diffing
what the real compiler produces. This project has no mcode generator; it
never emits mcode itself. This section goes one step further for the
first time: directly editing a real, working `.axmodel`'s mcode bytes by
hand (loading it as the ONNX protobuf it is, overwriting the `neu_key`
initializer's `raw_data`, resaving) and running the hand-patched result on
the real AX650N -- a much stronger causal test than comparing two
independently-compiled outputs, since it can construct byte patterns the
real compiler would never produce as a whole.

**Splicing confirms the noise zone is truly a swappable, inert label, not
just something two builds happen to agree is inert.** Building the same
`_two_conv_model` three times gave three real mcode blobs differing only
at 3 of the confirmed noise-zone positions (858, 870, 876 -- byte 864
happened to coincide across all three this time, consistent with a small
multiset randomly permuted per build). Constructing a hybrid -- build A's
mcode, but with build B's value spliced in at position 858 -- gives a byte
sequence that is **not** identical to any of the three real builds (a
genuinely novel combination, confirmed by direct comparison), yet it
loaded and ran on the real device with **bit-identical output** to the
unpatched original. This is real, direct proof by construction, not
correlation: the compiler's own output never had to agree with itself for
this to work, because the byte pattern tested was never compiled as a
whole by anything.

(The regression test locking this in uses 5 rebuilds and mixes noisy
positions across all of them, not just one -- caught during development:
with only 3 rebuilds, occasionally just one position actually varies, and
splicing only that one reproduces another real build byte-for-byte rather
than a genuinely novel combination. Same false-positive-adjacent lesson
this README has hit before elsewhere: verify the "novel" claim directly
rather than assume it.)

**Probing the still-opaque majority region with single-byte flips found
something new: it isn't uniformly load-bearing.** Flipping all 8 bits of
one byte (`^= 0xFF`) at 8 candidate offsets spread through the same
3,920-byte mcode blob (avoiding the header, footer, and known noise
positions) split cleanly into two real, reproducible outcomes, confirmed
stable across two different random inputs and repeat runs:

- **Offsets 400, 2000, 3000, 3400: the flip is completely inert** -- the
  patched model ran and produced bit-identical output to the unpatched
  original, for every input tried.
- **Offsets 700, 1000, 1500, 2500: the flip reliably faults the runtime**,
  every time, with the identical real error: `[ERROR] Run model
  failed{0x8030070C}` / `Request api(11) return failed(-2147090294)`. The
  device itself stayed healthy afterward (`axcl-smi` and the driver both
  confirmed fine) -- this is a clean, graceful runtime-level rejection,
  not a hardware lockup like the PCIe-driver crash this README's hardware
  section separately covers.

This is a real, useful, previously-unknown signal for future decoding
work, found without decoding a single new bit: `0x8030070C` is Pulsar2's
own real, reproducible signature for "this mcode program is structurally
invalid" -- almost certainly evidence of an internal checksum, opcode
validity check, or address-range check the runtime performs before or
during execution, not a full re-verification of program *semantics*
(since a corrupted-but-still-valid-looking byte can also just silently
produce identical output, as the inert offsets show). Knowing which
offsets fall on which side of this line, confirmed empirically rather
than guessed, is a real, concrete, well-scoped map for whoever attempts
to instrument or bisect further -- and a hard existence proof that some
of that 99%-opaque region cannot be padding: a mechanism this precise and
reproducible almost certainly means it's live, checked, structural data.

### Hand-patching the *decoded* Wbt requantization scale: confirms the field, corrects the mental model of how it acts

Everything above hand-patches still-*undecoded* mcode bytes. This applies
the same causal-intervention technique to a field this README already
claims to have decoded -- `Conv`'s per-channel requantization multiplier
`M_channel` in Wbt (`input_scale * weight_scale_channel / output_scale`,
identified earlier purely by correlation: it shrinks monotonically as
bias grows). A `Conv(cin=cout=4)` with a distinctly non-uniform bias
(`[0, 5, 10, 15]`, so each channel's own float32 value is individually
identifiable instead of accidentally coinciding across channels) locates
each channel's `M_channel` value at a precise Wbt offset, present in 4
identical repeated copies (64 bytes apart) -- consistent with this
README's earlier "Wbt reveals real channel-tiling structure" finding.

**Multiplying channel 0's value by 0.5 at all 4 repeated copies, confirmed
stable across two independent rebuilds**, cleanly isolates to exactly that
channel: channels 1-3's real device output are **bit-identical** to the
unpatched baseline, direct proof the offset identification and per-channel
indexing are both correct. But channel 0's actual change **refutes the
simple mental model** ("this scales the final float output by the same
factor") the earlier correlation-only description implied:

```
channel 0, baseline:  min=-1.73 max=1.73 mean=0.034 std=0.649 (42 unique values)
channel 0, M x 0.5:   min=-2.41 max=-1.13 mean=-1.976 std=0.299 (17 unique values)
```

Halving `M_channel` did **not** halve the float output (that would predict
a new mean near 0.017, still centered on zero) -- it collapsed the channel
to a much narrower, shifted band with far fewer distinct values. That
signature -- reduced spread, fewer unique output codes, a shifted center
-- is exactly what happens when a scale factor used *inside* int8
requantization (`int8_code = round(int32_accumulator * M) + zero_point`)
gets halved: the accumulator's full dynamic range collapses toward
`zero_point` in int8-code space *before* a separate, untouched
output-dequantization step converts back to float32, rather than `M`
being applied as an external multiplier on the already-dequantized float
result. This is consistent with -- and actually a more precise,
causally-verified version of -- the standard quantized-conv formula this
README already named, it just corrects exactly where the multiplication
happens in the pipeline. A real, verified example of this project's
recurring lesson: a correlation-only finding (an array that "shrinks as
bias grows") can misdescribe the actual mechanism even when the general
hypothesis is right, and only a direct intervention exposes that.

### The bit-flip probe on the real `resnet18d` mcode: a third outcome appears, and the blob is ~83% live

The single-byte-flip probe above only ever saw two outcomes on a tiny
two-Conv model (inert, or a `0x8030070C` fault), across just 8 offsets.
Running the same probe across the **real, unmodified `resnet18d_Opset18`
mcode** (49,080 bytes; 41 offsets evenly spaced every 1,200 bytes through
the interior, one flipped byte per run, each on a fresh copy of the same
base file, all fed the identical fixed `uint8 [1,224,224,3]` input) gives
a quantified, real-model-scale picture, and finds something the tiny
model never showed:

```
25 / 41  (61%)  FAULT      -- runtime rejects it: 0x8030070C, same code every time
 9 / 41  (22%)  DIFFERENT  -- runs, but the 1000-class logits change
 7 / 41  (17%)  identical  -- runs, bit-identical output
```

**The `DIFFERENT` class is new, and it is real computation.** Every one of
those 9 flips ran to completion with no error, yet produced different
logits -- and in **7 of the 9 the predicted class itself changed** (e.g.
argmax 305 -> 567, 908, 834, 143, 111, 533, 318), with max-abs logit
deltas from 0.18 up to 7.29. The other 2 shifted the logits only mildly
(0.18, 0.29) and kept the argmax. These are the first bytes this project
has ever identified whose effect on the *actual computation* is directly
observable -- a genuine instruction/parameter class, distinct from the
checksum/opcode-validation class that faults, and from the inert class.
All three outcomes were verified to be deterministic and not device noise:
the unpatched baseline is bit-identical across 3 repeated runs; three of
the `DIFFERENT` flips (9900, 32700, 48300) reproduce the *same* changed
output on rerun; and two `identical` flips (12300, 26700) stay
bit-identical against a second, different random input. The device stayed
healthy throughout 25 consecutive faults (`axcl-smi` fine afterward) --
the fault is graceful every time, never a lockup.

**Bottom line for "how live is a real model's mcode":** by this sampling,
**~83% of `resnet18d`'s mcode bytes are load-bearing** (61% checked
structurally, 22% affecting real output), and only ~17% are inert -- far
more live than the tiny model's 50/50 split suggested, and a direct,
quantified counterpart to the earlier "43.4% of bytes are exact
duplicates" finding: a byte being a *copy* of another span does not make
it dead, since the hardware evidently reads those copies. The 9
output-changing offsets are the most concrete decoding targets this
README has produced so far -- each is a real, known-live byte whose
effect on a real classifier is already measurable, so the next step
(bisecting each one's *neighbors* to map the extent of its field, and
correlating the resulting logit change with which layer's Conv it sits
in) needs no new technique at all.

### Bisecting the live bytes' neighbors: an identical field template 15,600 bytes apart, and every bit of one byte is live

Following the step named above: flip each byte in a 17-byte window
(`X-8..X+8`) around three of the output-changing offsets, one flip per
run, and classify each as fault (`F`), identical (`=`), or different
(`D`). Then flip each of the 8 bits of one such byte individually.

```
X=9900:   [FFFF=FFFDDFFFFDFF]
X=32700:  [FF==D=FDDDDDF=FF=]
X=48300:  [FF==D=FDDDDDF=FF=]     <- byte-for-byte the same signature as 32700
```

**Two offsets 15,600 bytes apart have the identical 17-position live-byte
signature.** Not merely similar bytes: the same layout of which
positions fault, which are inert, and which change the output --
including the same contiguous 5-byte live run at `X-1..X+3` and the same
isolated live byte at `X-4`. This is the earlier "43.4% of bytes are
exact duplicates / repeated command templates" finding seen from the
*hardware's* side: the same command template recurs, and its internal
field structure recurs with it. `9900` has a different, narrower layout
(a 2-byte live pair at `X..X+1`, one isolated live byte at `X+6`, faults
everywhere else), i.e. a different template.

**Two different bytes that are functionally interchangeable.** Within
both templates, flipping `X-4` and flipping `X-1` (e.g. 32696 vs 32699,
48296 vs 48299) produce the **bit-identical full 1000-logit output** --
confirmed on rerun. Two distinct bytes, 3 apart, whose corruption drives
the computation to exactly the same state: consistent with two copies of
one value being combined (or a field where those two bytes play the same
role), and a concrete, reproducible handle on this template's internal
redundancy.

**Every bit of byte 9900 is live, and the effect is not bit-weighted.**
All 8 single-bit flips run without fault and all change the output --
no bit is padding. But the max-abs logit delta does *not* grow with bit
significance (bit0 3.41, bit1 1.54, bit2 2.15, bit3 1.97, bit4 1.08, bit5
1.08, bit6 2.15, bit7 2.98), as a plain little-endian integer or a
float32 byte would predict. Instead the deltas fall on a grid: bit4 and
bit5 give exactly 1.077231, bit2 and bit6 exactly 2.154462 = 2 x 1.077231
(equal max-abs, but *different* full outputs -- so it's the magnitude
that quantizes, not the result). That grid is the output tensor's own
int8 dequantization step showing through, which is why it cannot be used
to read the byte's encoding off the logits directly. Reported as what it
is: a fully-live, non-bit-weighted numeric-ish field whose encoding is
still not identified -- the exact-equality pairs are the lead, not a
decoding.

### Inside one repeated template: a gate, three address-like bytes, and a checked sign bit

Two follow-up probes on the `FF==D=FDDDDDF=FF=` template, again no
rebuild needed, sharpen what its live bytes actually are.

**The interchangeable `X-4`/`X-1` pair is a gate, not a value.** Flipping
`X-4` alone, `X-1` alone, or **both at once** lands in the byte-identical
output at both template instances (32700: 3.986/argmax 111; 48300:
1.293/argmax 305), and never back at baseline. A double flip that neither
cancels (as two XOR-combined copies would) nor compounds (as two
independent numeric contributions would) means the two bytes don't carry
a *value* -- they gate a condition that either holds or doesn't, and any
disturbance drops the computation into one fixed fallback state.

**Per-bit across the 5-byte live run (`X-1..X+3`) at 32700:**

```
byte    bit0   bit1   bit2   bit3   bit4   bit5   bit6   bit7
32699  3.986  3.986  3.986  3.986    =      =      =      =     <- X-1
32700  5.135  3.016  5.063  6.176  6.068  6.751  5.530  5.099   <- X
32701  5.386  6.284  3.447  6.894  6.858  2.370  7.469  3.052
32702  3.591  5.745  5.997  3.375  4.776  4.453  4.596  6.104
32703  5.925  4.740  5.889  3.627  5.853  4.309  5.673    F     <- X+3
```

- **`X-1` is half dead, half gate.** Its low nibble (bits 0-3) each
  trips the *same* 3.986/argmax-111 state as `X-4` does; its high nibble
  (bits 4-7) is completely inert. So `X-4` and the low nibble of `X-1`
  are one gate -- the same fallback state now replicated across seven
  independent single-flip measurements plus the double flip.
- **`X..X+2` are three fully-live bytes with large, non-bit-weighted
  effects** (every bit changes the output, deltas 2.4-7.5, no monotone
  trend with bit index; e.g. at 32701 bit 6 gives 7.47 but bit 7 gives
  3.05). A magnitude field would scale with bit significance; a field
  where *any* corrupted bit makes the hardware read the wrong data --
  an address, offset, or index -- would look exactly like this. Reported
  as "address-like," not as a decoded address.
- **`X+3`'s MSB is checked, its other 7 bits are live.** Bit 7 of the
  run's last byte is the one single-bit flip in the whole 40-flip matrix
  that faults (`0x8030070C`); bits 0-6 all run and change the output. A
  sign/valid/reserved bit sitting at the top of a 4-byte little-endian
  field, checked by the same validator the `F` positions hit, is the
  simplest reading.

Net: this template's live region reads as `[gate byte] [gate nibble |
dead nibble] [3 address-like bytes] [7 live bits | 1 checked bit]` --
the most detailed internal map of any mcode command this README has
produced, obtained entirely from single-bit interventions on the real
`resnet18d` blob.

**A caveat on the fault class, found the hard way: `0x8030070C` can hit
a valid model, rarely.** While locking the template findings into a
regression test, the *unpatched* baseline of a freshly-built `resnet18d`
faulted with the same `0x8030070C` -- once, immediately after a dense
burst of roughly 65 deliberately-faulting runs plus a Docker build. The
same file ran fine seconds later, and so did the known-good build, so it
was neither a bad build nor a wedged device (no kernel/PCIe events;
telemetry normal). Trying to reproduce it on purpose failed: 10 trials of
a deterministic-fault run immediately followed by a valid run gave **0/10
transient faults** (and 10/10 deliberate faults, reconfirming that class
is deterministic). So this is a rare event -- one in well over 200 valid
runs this session -- not something a single preceding fault triggers,
and its trigger is not identified. The tests now retry a `0x8030070C`
exactly once (`_run_retry_once`): safe, because a genuinely faulting
byte faults on every attempt, so a second fault is a real one, while a
transient recovers -- and it never masks anything, since no other error
is retried. Worth knowing for anyone running large fault-injection
sweeps on this hardware.

### What the output-changing bytes *are*: 8 of 9 are control-like, 1 is data-like

The 22% "runs, but changes the output" class above says a byte is live
and computational, but not what *kind* of byte it is. A cheap
discriminator: run the same flipped model against **different inputs**.
A corrupted *weight or data* value interacts with the input, so its
effect should vary with the input; a corrupted *control* value (an
address, offset, index, or fixed constant) redirects the computation the
same way regardless of what flows through it. Each of the 9
output-changing flips against three independent random `uint8` inputs
(seeds 42, 7, 123; all three baselines classify as 305):

```
offset   seed42          seed7           seed123         shifted class
 6300    305  0.18       305  0.18       305  0.14       same
 9900    567  3.81       567  3.81       567  3.81       same
20700    143  0.50       143  0.50       143  0.50       same
24300    111  0.97       111  1.04       111  1.01       same
32700    908  4.45       908  4.42       908  4.42       same
36300    533  5.75       533  5.85       533  5.82       same
43500    305  0.29        65  0.32        60  0.29       DIFFERS
44700    318  1.47       318  1.47       318  1.51       same
48300    834  7.29       834  7.29       834  7.33       same
         (argmax, max-abs logit delta vs. that input's own baseline)
```

**Eight of the nine are input-independent**: the same flip lands the
classifier on the same wrong class with a near-identical delta on every
input -- 9900 goes to 567 at 3.81 all three times, 48300 to 834 at
7.29/7.29/7.33. That is control-like behavior, and it lines up with the
template map above: 32700 and 48300 are the template whose live run
reads as address-like bytes, and a wrong address fetches the same wrong
data no matter the input. **Exactly one, 43500, is input-dependent** --
the shifted class moves with the input (305, 65, 60) and the delta stays
small (~0.3): the signature of a corrupted weight/data value whose effect
is modulated by what it multiplies. So the "changes the output" class is
itself ~8:1 control-like to data-like at this sampling, which is also a
useful hint about mcode's overall composition: it is dominated by
control/addressing, with real numeric data mostly living elsewhere (Wbt)
-- exactly what the `AxQuantizedConv` weight-in-Wbt / commands-in-mcode
split found much earlier would predict.

### The one data-like byte, probed: the gate recurs in a third template, and the prediction is only half right

The control-vs-data reading above makes a falsifiable prediction: a
data-like byte should be *bit-weighted* (its high bits should matter
more than its low bits), unlike the control bytes at 9900/32700 whose
every bit produced a large, unordered effect. Probing 43500 the same
two ways:

```
window:  F===D=FDDDFFF=FF=      (43492..43508; control template was FF==D=FDDDDDF=FF=)
  43496 (X-4): D 2.442 / argmax 116
  43499 (X-1): D 2.442 / argmax 116      <- bit-identical output to X-4
  43500 (X):   D 0.287 / argmax 305
  43501 (X+1): D 1.149 / argmax 106

per-bit at 43500:
  bit0 0.323  bit1 0.395  bit2 0.287  bit3 0.359     (argmax stays 305)
  bit4 0.682  bit5 1.149  bit7 1.508                 (argmax -> 741, 741, 106)
  bit6 0.323                                         (argmax 305; == bit0 exactly)
```

**The gate is now seen in a third template.** The window shares the
control template's skeleton -- an isolated live byte at `X-4`, the run
starting at `X-1` -- and `X-4` and `X-1` again yield the bit-identical
output. That makes the "any disturbance trips one fixed state"
`X-4`/`X-1` structure a recurring feature of these command templates
(three templates, ~10 independent measurements), not a quirk of one.
The live run here is 3 bytes (`X-1..X+1`) rather than 5, so this is a
related variant, not the same template.

**The bit-weighting prediction is half right, and reported as such.**
High bits dominate: bits 4, 5, and 7 produce the largest deltas and are
the only ones that move the predicted class, while bits 0-3 perturb the
logits without changing the class. That is more ordered than the control
bytes ever were. But it is not a clean weighting: bit 6 is small (0.323,
exactly equal to bit 0 -- the output's int8 grid again), so a strict
"delta grows with bit index" test fails. One further observation does
fit a *signed* numeric value specifically: flipping all 8 bits at once
changes the output *less* (0.287) than flipping bit 7 alone (1.508).
XOR-ing every bit of a small two's-complement value is approximately a
negation and lands near zero, whereas flipping only the top bit moves it
by half the range -- so this is exactly the ordering a small signed byte
would show. Consistent with a signed quantized parameter, not a
decoding of one.

### The opaque region is a 32-bit-word instruction stream: `a1 00 xx yy` headers, never adjacent, each followed by an operand

The hardware probes above pointed at *which* bytes matter; comparing the
raw bytes of the probed windows -- zero device runs -- shows *why*, and
it is the most concrete structural decoding of mcode this project has
produced.

**The two control-template instances differ at exactly the three
"address-like" bytes and nowhere else.** The 17-byte windows around
32700 and 48300 are identical at all 14 positions the probes classified
as fault, inert, or gate, and differ only at `X..X+2` -- precisely the
bytes whose flips gave large, unordered, input-independent effects:

```
32700:  05 00 00 00 | a1 00 40 02 | 8c 4c 06 00 | a1 00 50 01 | 00
48300:  05 00 00 00 | a1 00 40 02 | a0 49 b3 00 | a1 00 50 01 | 00
43500:  00 00 00 00 | a1 00 50 03 | 80 64 1e 00 | a1 00 50 01 | 00
```

Segmented into 4-byte words (all three windows start on a 4-byte
boundary), the layout is the same everywhere: a **header word
`a1 00 xx yy`**, then a **32-bit little-endian operand** (`0x00064c8c`,
`0x00b349a0`, `0x001e6480`), then the next header. Every probe result now
has a place: the gate at `X-4` is the header's first byte `a1`; the
"gate nibble" at `X-1` is the low nibble of the header's last byte
(`02`/`03`); the three address-like bytes are the operand's low three
bytes; the checked MSB at `X+3` is bit 7 of the operand's top byte
(`00`); and the data-like instance simply has a different header
(`50 03` vs `40 02`), i.e. a different instruction kind whose operand
the hardware treats differently.

**This is the format of the whole blob, in two different models, not a
local coincidence** (all of it local byte analysis, deterministic,
re-checkable in seconds):

```
                             resnet18d (49,080 B)     tiny two-Conv (3,920 B)
length % 4                   0                        0
`a1 00` words, 4-byte aligned  1,197                  58
`a1 00` byte pairs, unaligned  167 (~56 per phase)    49 (~16 per phase)
header followed by a header  0 / 1,197                0 / 58
even gaps between headers    98% (1,174 / 1,196)      86% (49 / 57)
```

Aligned headers are ~21x (resnet18d) and ~3.6x (tiny) more frequent than
the same byte pair at any other phase, so the alignment is real, not
chance. **No header is ever immediately followed by another** (random
placement would give ~117 such pairs in resnet18d), which is the
`[header][>=1 operand]` structure exactly. Gaps between headers are
dominantly 2 words -- one operand -- with 4, 8, 10, 12 next; a first
version of this section said "strictly even," which was an overstatement
from reading only the top gap sizes: it is 98% and 86%, i.e. 2-word
granularity with exceptions, and 32 bytes is four such 2-word
instructions, reconciling the very first "32-byte mcode unit" finding.

**The instruction mix is model-dependent; the format is not.** In
`resnet18d` three header kinds dominate -- `50 01` (724), and `40 02`
and `50 03` at *exactly* 181 each (the two probed control bytes were
`40 02`, the data-like one `50 03`, so a `40 02`/`50 03` pair per op is
the natural reading, e.g. an address-set and a data-set). The tiny model
has a different top set (`80 02`, `b0 03`, `b0 0b`, `30 04`), as a
different op/shape mix should. One more tie-in: the tiny model's four
known non-deterministic noise bytes (858/864/870/876) fall in `23 00 xx
82`-style words, never in an `a1 00` header, carrying exactly the known
`{0x10,0x20,0x30,0x40}` label values -- the label noise is a field of a
different word family.

**What this does and does not settle.** It settles that the "99% opaque"
region is a parseable stream of 4-byte words with a recognizable header
kind, so "decoding mcode" now means "decoding operand semantics per
header kind," a far smaller and better-posed problem. It does *not*
identify what `a1` means, what `xx yy` encode beyond "kind," or confirm
that operands are addresses -- though the input-dependence split
(control-like `40 02` vs data-like `50 03`) is exactly what "an address
to an output buffer" vs "an address to weights" would produce, which is
now the leading hypothesis and one a single targeted test (patch an
operand to another instance's value) could check.

### `40 02` operands are weight-table offsets; `40 02`/`50 03` come in pairs one 32-byte unit apart; `50 01` is a 4-valued flag

Still zero device runs -- pairing and operand statistics over all 1,197
headers, then one size cross-check.

**`40 02` and `50 03` are paired one-to-one.** Every one of the 181
`50 03` headers has a `40 02` before it at a word distance of 8 (166
cases), 10 (14), or 44 (1), and the reverse histogram -- each `40 02` to
its next `50 03` -- is identical. Eight words is 32 bytes: the pair sits
exactly one "32-byte mcode unit" apart, the unit this whole
investigation began with, with three other two-word instructions in
between.

**Operand statistics separate the kinds cleanly:**

```
kind        n    distinct   operand range          reading
a1 00 40 02  181    181     0x000000..0xb45ba0     address-like, one per op
a1 00 50 03  181    176     0x000000..0x2f7040     address-like, smaller region
a1 00 50 01  724      4     {0x1 .. 0x01000000}    enumerated flag, not an address
```

**And the size cross-check decodes `40 02`.** `resnet18d`'s Wbt
(`npu_params`) is 11,855,108 bytes = `0x00b4e504`; the largest `40 02`
operand is `0x00b45ba0` -- within 0.3% of the Wbt's end -- and the 181
values are all distinct. So **`a1 00 40 02 <operand>` sets this op's
offset into the weight table**: a real, quantitative, independently
checkable semantic for one instruction kind, obtained without any
device. `50 03`'s operands stop at ~3.1 MB, about a quarter of the Wbt,
so they address a different, smaller region -- the size is consistent
with an activation/intermediate-buffer arena, but that is inferred from
size alone and is not confirmed.

**Correction, stated in place.** The previous section guessed the
opposite assignment -- `40 02` as an output-buffer address and `50 03`
as a weight address -- from the input-dependence split alone. The Wbt
span settles it the other way. The two observations still fit together:
corrupting a Wbt offset can land the op on per-channel scale/bias data
that saturates its output to a fixed value regardless of input
(input-independent, as 32700/48300 showed), while corrupting a pointer
into an activation region reads data that still varies with the input
(input-dependent, as 43500 showed). A related caveat on that
input-dependence experiment: its three "different inputs" were all
independent uniform-noise images, which are statistically alike -- a
weaker test than three genuinely different photographs. It still had
discriminating power (43500 did vary), but the 8-of-9 count should be
read with that in mind.

The most common instruction, `50 01`, takes only four distinct operand
values across 724 uses -- a small enumerated control/flag word, not a
pointer; which flags is not identified.

### Patching a `40 02` operand on the device: in-range predictions hold, the bounds-check prediction fails

The section above named one targeted test; here it is, on the real
device. Overwrite instance 32700's whole 4-byte operand (`0x00064c8c`)
and run, one variant per fresh copy, same fixed input:

```
A <- B's operand 0x00b349a0 (valid, in range)   runs, D  argmax 905
B <- A's operand 0x00064c8c (valid, in range)   runs, D  argmax 832
A <- 0                                          runs, D  argmax 925
A <- last in-range word (Wbt end - 4)           runs, D  argmax 18  (smallest change, 0.83)
A <- 1 MB past the Wbt's end                    runs, D  argmax 530
A <- 0x7fffff00 (~2 GB, far past)               runs, D  argmax 794
```

**Confirmed:** the operand is live and causal for any value -- every
variant ran and changed the output, deterministically (the two past-end
variants and the swap each reproduce the identical output across three
reruns). Swapping two instances' offsets works in both directions.

**Refuted, and worth stating plainly:** the prediction that an
out-of-range offset would fault. It does not. The runtime accepts 1 MB
past the Wbt's end, and `0x7fffff00`, without any error -- there is **no
bounds check on this operand**. So the "weight-table offset" reading
rests on the static evidence (181 distinct values whose maximum matches
the Wbt's size to 0.3%), not on any runtime enforcement; mechanically
this behaves like a raw base-plus-offset read, and a past-end value
simply reads whatever stable, mapped memory sits there (deterministic
across reruns, so not uninitialized garbage). It also sharpens what the
`0x8030070C` validator is: every fault in this whole investigation came
from a *header/format* byte, never from an operand value, however
absurd -- the validator checks instruction structure, not operands.

One suggestive detail, single data point: pointing the op at the last
in-range word produced the *smallest* change of all six (0.83, vs
2.4-7.5), as reading a near-empty tail of the weight table would.

### `a1 00 xx yy` is a field write: `xx` is a 16-byte-granular field offset, `yy` a bank, and the map is shared across models

Re-running the header statistics on the tiny two-Conv blob, side by
side with `resnet18d`, corrects the "instruction kinds" framing above
into something more specific and better supported -- all local, zero
device runs.

**Three checked regularities, both models:**

- **`xx` is a multiple of `0x10` in every header: 58/58 and
  1,197/1,197.** Not a single exception in 1,255 headers -- a 16-byte
  granular offset, not an opcode.
- **The `xx` values form ladders within each `yy`, and the ladders are
  the same in both models.** Bank `0x02`: `0x10,0x20,0x40,...,0xd0` in
  the tiny model, `0x10..0xd0` in `resnet18d`; bank `0x03`:
  `0x30,0x50..0xe0` vs `0x30,0x50..0xf0`; bank `0x04`: `0x30..0xe0` vs
  `0x40..0xf0`; bank `0x01`: `{0x50,0x60}` in both. A shared map of
  fields, not a per-model vocabulary.
- **`50 01` takes the identical operand set in both models:**
  `{0x1, 0x100, 0x100000, 0x1000000}` -- single bits 0, 8, 20, 24. A
  one-hot enable/mode register, written 724 times in `resnet18d` and 4
  times in the tiny model, with the same four values.

**So the reading is: `a1 00 xx yy <value>` writes a 32-bit value to
field `xx` of bank `yy`** -- a descriptor or register-file write. That
one reading explains everything found so far at once: a write is
`[selector][value]`, so no selector is ever immediately followed by
another (0/1,255); writes are two words, so the 2-word granularity and
the 32-byte (four-write) unit; the "gate" is the selector word (corrupt
it and the value goes to the wrong field, or nowhere -- one fixed
fallback state); the validator faults on malformed *selectors*, never on
*values*; `40 02` is the Wbt-offset field (181 distinct values spanning
the Wbt in `resnet18d`; the single write in the tiny model is `0`, the
first op's offset -- and `resnet18d`'s values include
`0x14f80, 0x14f82, 0x14f84`, byte-granular steps, as INT8 weight
offsets would be); and the input-independent vs input-dependent split is
just which field got a bad value.

**Corrections, stated in place.** (1) "The instruction mix is
model-dependent; the format is not" -- better: the *field map* is
shared; a large model re-writes a few fields (Wbt offset, an address in
bank 3, the `50 01` flags) once per op, while a small model touches many
fields once each as setup. (2) The tiny-model "structure holds" table
earlier rested on 58 headers with many one-offs, which was thin; the
ladders and the identical `50 01` set are the real, strong tiny-model
evidence. (3) "Operands are address-like" holds for address *fields*
(`40 02`, `50 03`); many bank-2/3/4 fields in the tiny model carry
high-entropy packed values (`0xa1020c81`, `0x8130180f`, ...) -- packed
configuration words, not addresses.

What this still does not settle: the meaning of any field other than
`40 02` (Wbt offset, quantitatively) and `50 01` (a one-hot flag, values
known, meaning not), and what the banks are. But "decode mcode" has
now gone from "49 KB of opaque bytes" to "a register map with ~50 named
fields in a dozen banks, each written with a 32-bit value" -- a
concrete, enumerable target.

### Typing the field map: three per-op fields are 91% of a real model's mcode; the rest is one-time setup

Counting, per field, how many times `resnet18d` writes it and how many
distinct values it uses (local, zero device runs) turns the enumerable
map into a typed one:

```
field         writes  distinct   value range               what it looks like
a1 00 50 01     724       4      {bit 0, 8, 20, 24}         one-hot flag, per op
a1 00 40 02     181     181      0x0 .. 0xb45ba0 (= Wbt)    Wbt offset, per op
a1 00 50 03     181     176      0x0 .. 0x2f7040 (~3.1 MB)  address into a smaller region, per op
a1 00 30 03      16      10      0x1 .. 0x30009683          packed config, occasional
a1 00 20 02      15       9      0x1 .. 0x93ff              small counts/sizes, occasional
(6 more fields written 3-4 times, all high-entropy packed words, e.g. 0x888c97ff, 0xc0031883)
(57 fields written once or twice)
```

**68 distinct fields in total; 57 of them are written fewer than three
times.** Only three fields are written at per-op scale, and those three
account for **1,086 of the 1,197 writes (91%)**. In other words, a
real model's mcode is overwhelmingly a per-op loop of "set the flag,
set the weight offset, set the other address" against a fixed
configuration laid down once. That also says where the remaining
decoding effort is worth spending: the semantics of exactly three
fields would cover 91% of what the hardware is told, and two of them
are already pinned quantitatively (`40 02`) or by value set (`50 01`).
The obvious next target is `50 03`'s ~3.1 MB region -- the same
size-matching argument that decoded `40 02` against the Wbt can be run
against the model's activation tensors.

### `50 03` is not whole activation tensors; it is a 64-byte-aligned tile arena

Running that size-matching argument (local; shape inference on the
cached `resnet18d` ONNX vs. the 176 distinct `50 03` operands):

- **The whole-tensor hypothesis fails, and is reported as such.** The
  operands' consecutive differences -- 4,096; 37,120; 14,080; 23,040;
  7,936; ... -- are not activation-tensor sizes (`resnet18d` has 10
  distinct INT8 activation sizes, from 512 to 802,816 bytes); only 2 of
  the top 10 deltas coincide with a tensor size (25,088 and 512),
  consistent with chance. Neither the largest tensor (802,816 B) nor the
  sum of all intermediates (7.59 MB) equals the 3.1 MB span.
- **What does hold is alignment and scale.** 175 of the 176 operands
  are exact multiples of 64; the smallest non-zero one is 37,120
  (`0x9100`, also the most frequent delta); the deltas are
  tile-sized (a few KB to a few tens of KB), not tensor-sized.

So `50 03` addresses a 64-byte-aligned arena of *tiles*, into which
the compiler places activation chunks -- which fits both the earliest
"32-byte unit / channel tiling" findings and the `_ocm_base` string
this README decoded from the mcode header table long ago (on-chip
memory has exactly this shape: small, aligned, tile-addressed). Whether
the ~3.1 MB span is the on-chip memory's actual size is not confirmed
against any spec here, and the tile-to-tensor mapping is not
decoded; the field's *kind* -- aligned tile address -- is.

### `50 01` is a per-op four-step sequence, written in a fixed order -- not an engine selector

The most-written field, `50 01`, takes exactly four one-hot values
(`0x1, 0x100, 0x100000, 0x1000000` -- bits 0, 8, 20, 24) in both
models. Four values and four profiled engines (`conv0`, `conv1`,
`teng2`, `sdma4`) invite the guess "it selects an engine." Testing that
guess, locally, refutes it:

- **Every op writes all four, in the same fixed order.** Splitting
  `resnet18d`'s header stream at each `40 02` (Wbt-offset) write, the
  `50 01` values between consecutive ops are `bit8, bit0, bit20, bit24`
  in **180 of 180** segments, and the tiny two-Conv model's single op
  writes exactly the same four in the same order. Each value occurs
  exactly 181 times -- once per op -- so 724 = 4 x 181 is not "four
  choices" but "four steps."
- So `50 01` is a **per-op sequence register**: every op pulses it
  four times in a fixed order. Which four steps -- four stage triggers,
  four engine-enable pulses issued in sequence, a four-phase handshake
  -- is not identified; what is settled is that it is not a per-op
  choice of one engine.

A count that lines up on the way: `sdma4` has exactly **181** events in
`resnet18d`'s trace -- one per op, one per `40 02` write -- consistent
with each op's Wbt-offset write being followed by one weight DMA on the
system DMA engine. (`conv0`/`conv1` have 522 events each; `teng2` 46;
`cv3` 162.)

### The four `50 01` steps on the device: three required, one optional, and a correction about what the validator checks

`50 01` is written four times per op in a fixed order (`bit8, bit0,
bit20, bit24`). Editing those four writes for one real `resnet18d` op
on the device, one variant per run, then repeating on a second op:

```
                                      op @ word 8174   op @ word 12074
skip bit8   (write 0 instead)         FAULT            FAULT
skip bit0                             FAULT            (not repeated)
skip bit20                            FAULT            (not repeated)
skip bit24                            identical        identical
swap bit8 <-> bit0                    identical        identical
swap bit20 <-> bit24                  FAULT            (not repeated)
all four -> 0                         runs, D (815)    runs, D (832)
```

- **`bit8`, `bit0`, `bit20` are each required**: removing any one of
  them faults with the usual `0x8030070C`.
- **`bit24` is optional**: removing it leaves the output bit-identical
  on both ops -- whatever the fourth step does, correctness does not
  depend on it (a timing or profiling pulse is the natural guess).
- **`bit8` and `bit0` are order-free** (swapped, identical output);
  **`bit20` must precede `bit24`** (swapped, fault; one op).
- **Removing all four does *not* fault** -- the op runs, with a
  different result, on both ops. So the check is not "a step is
  missing" but "the steps present are inconsistent": a partial
  sequence is rejected, an absent one falls back to some default mode.

**Correction, stated in place.** The `40 02` operand section above
concluded that "every fault in this whole investigation came from a
header/format byte, never from an operand value, however absurd." That
held for every *address* value tried, and it is wrong as a general
statement: the `50 01` step values above are operand values, and
zeroing one of them faults. The consistent reading of all of it is that
`0x8030070C` is a **runtime sequencing rejection** rather than a static
format check -- it fires when the command stream is inconsistent as a
*sequence* (a corrupted selector word, a partial step set, a
misordered step), and stays silent for values that do not break
sequencing (an out-of-range address is still a well-formed step). That
also explains why it never fired on any `40 02` value and why a wholly
absent step set runs.

### The `50 01` step set is the op's dispatch, and its fourth step is inert model-wide

Two device probes that each settle one question with a single decisive
comparison.

**An op with no step set does not execute.** Zeroing all four `50 01`
writes of one real `resnet18d` op makes the model run with a changed
result (argmax 305 -> 815). Doing the same *and* also pointing that
op's `40 02` Wbt offset at a different instance's weights gives the
**byte-identical** changed result. If the op still executed in some
"default mode," its weights would matter and the two outputs would
differ; they do not. So the four `50 01` writes are what dispatch the
op -- remove them and the op is simply skipped, its weight offset never
read, and the model's output is whatever the downstream layers make of
a missing contribution. This also reinterprets the "removing all four
does not fault" observation: there is no partial sequence left for the
sequencer to reject, because there is no op.

**`bit24` is inert for correctness across the entire model.** The
per-op probe found that zeroing the fourth step (`bit24`) leaves one
op's output bit-identical. Zeroing it in **all 181 ops at once**, in one
patched model, leaves the whole network's output **bit-identical to the
unpatched baseline**. Whatever the fourth write does -- a completion or
profiling pulse is the natural guess -- no part of `resnet18d`'s
computed result depends on it. Combined with the previous section: each
op is dispatched by three required steps (`bit8`, `bit0` in either
order, then `bit20`), followed by one step that can be dropped
model-wide without changing a single output bit.

### Five verbs, a readable per-op program, a variable-length preamble, and the arena size in the header

Three local, zero-device-run checks that complete the structural
picture -- and correct one earlier statement.

**`a1` is one of five verbs over the same selector.** Tallying every
phase-0 word of the shape `XX 00 <multiple of 0x10> yy` in the
`resnet18d` blob, `a1` is joined by four more first bytes with the same
`xx yy` field/bank selector and the same per-op multiples:

```
verb  writes  distinct selectors   what it writes
a1     1197   68                   field writes (the map above)
a8      365   5    -> 30 02 x181, 40 03 x181   one bank-2 and one bank-3 write per op
a2      359   2    -> 00 00 x358               ~2 per op, fixed selector
a3      185   4    -> 00 00 x182               1 per op
a9      181   1    -> 00 00 x181               exactly 1 per op
```

**With all five verbs counted, the bulk is a uniform two-word stream.**
From byte 328 there are 2,287 verb headers and **97.5% of consecutive
headers are exactly two words apart** (2,228 of 2,285): `[verb 00 xx yy]
[32-bit operand]`, eight bytes per instruction, throughout. The earlier
"gaps of 4, 8, 10 words between `a1` headers" were other verbs in
between, not longer instructions.

**The per-op program is now readable.** Splitting at each `a1 40 02`
(Wbt offset), 64 of 180 ops are exactly this eleven-instruction
sequence, and 40 more differ only by one extra `a2`:

```
a1 40 02 <Wbt offset>      set weight offset
a1 50 01 bit8              step 1 (required)
a1 50 01 bit0              step 2 (required, order-free with step 1)
a8 40 03 <...>             bank-3 write
a1 50 03 <arena address>   set tile-arena address
a1 50 01 bit20             step 3 (required)
a3 00 00 <...>
a1 50 01 bit24             step 4 (optional model-wide)
a9 00 00 <...>
a2 00 00 <...>
a8 30 02 <...>             bank-2 write
```

Each of the four `50 01` steps sits immediately before a different verb,
which reads naturally as "arm, then do" -- but that is an interpretation;
the sequence itself is the measured fact (22 distinct per-op patterns
in total, dominated by these two).

**Correction, stated in place: the stream is not 4-byte aligned from the
blob's first byte.** The first field write sits at byte 297 (phase 1):
`a1 00 40 02 ff 00 00 00 | a8 00 50 02 00 00 00 00 | a1 00 60 02 80 14 03
00 | a1 00 70 02 81 0a 03`, where that last slot is **seven** bytes -- a
3-byte operand -- and only from about byte 328 onward do headers settle
at phase 0 and stay there (1,192 of the bulk's 1,320 `a1 00` pairs; the
scattered phase-1/2/3 ones lie inside packed operand words). So the
preamble has variable-length instructions, the bulk does not, and a
parser must not assume 8-byte slots from offset 0. The earlier word
indexing from the blob start was right for the bulk by coincidence of
where the preamble's odd slots happen to end.

**The tile arena's size is declared in the FlatBuffers header.** Header
words 72 and 76 hold `4096` and `0x002ff000` (3,141,632). The largest
`50 03` operand, `0x2f7040`, sits 32,704 bytes below `0x2ff000` -- less
than one typical tile (the most common `50 03` delta is 37,120) -- i.e.
the last tile fits exactly under it. A declared size that bounds every
arena address to within one tile is strong evidence that word 76 is the
arena size and word 72 its page/alignment, which also links the header
region (decoded first, long ago) to the field map for the first time.
Whether `0x2ff000` is the physical on-chip memory size is still not
checked against a spec.

### The whole map generalizes to a second real architecture: `mnasnet_small`

Everything above about mcode's structure was established on `resnet18d`
(plus a tiny synthetic two-Conv model). A real second architecture,
`mnasnet_small_Opset17` -- depthwise-separable, 133 NPU ops vs.
`resnet18d`'s 181, a 2.77 MB weight table vs. 11.86 MB -- compiled fully
on the NPU (one fused subgraph), ran on the real AX650N, and was
bit-identical between the original and onnxsim-simplified graphs, as
always. Its mcode (78,584 bytes) checked against every structural claim,
all local:

```
claim                                   resnet18d            mnasnet_small
header words 72 / 76                    4096 / 0x2ff000      4096 / 0x2ff000   (identical)
largest 50 03 below the arena, by       32,704 B (< 1 tile)  26,848 B (< 1 tile)
first field write                       byte 297, phase 1    byte 297, phase 1
five verbs present                      yes                  yes
40 02 / 50 03 / a9 / a8:3002 / a8:4003  181 each             133 each   (= op count)
50 01 writes per op                     4.00                 4.00
50 01 operand set                       {bit0,8,20,24}       {bit0,8,20,24}
50 01 order per op                      bit8,bit0,bit20,bit24 (180/180)  same (132/132)
dominant per-op template                64 of 180 ops        34 of 132 ops
40 02 max / Wbt size                    0.997, all distinct  0.984, all distinct
sdma4 trace events                      181 (= ops)          133 (= ops)
two-word tiling of the bulk             97.5%                92.3%
50 03 operands 64-byte aligned          175/176              112/133
```

**Every per-op invariant holds exactly on the second model**, with the
op count simply changing from 181 to 133: the five verbs, the one write
each of `40 02`/`50 03`/`a9`/both `a8` selectors per op, four `50 01`
writes per op in the same one-hot set and the same fixed order, the same
dominant eleven-instruction template, `40 02` spanning the weight table
(all distinct, 98.4% of a Wbt one quarter the size), and one `sdma4` DMA
per op. **The arena size `0x2ff000` is identical**, so it is a platform
constant of the AX650N, not something the compiler sizes per model --
and both models' arena addresses stay under it by less than one tile.
The depthwise convs also show up where the engine-split finding said
they would: both MAC engines busy (`conv0` 909, `conv1` 855 events).

**Two honest deltas.** The bulk's two-word tiling is 92% here vs. 97.5%
(a few 16-word gaps -- larger operands or a different verb the tally
does not know), and `50 03` alignment is 84% vs. 99% (depthwise tiles
plausibly use a finer granule). Neither touches the per-op invariants;
both are worth knowing before treating the parser as complete.

Net: the field-write reading, the verb set, the per-op program, and the
two quantitatively decoded fields are properties of the compiler and
the hardware, not of `resnet18d`.

### Correction: the instruction runs are two-word, but they are only 37% of the bulk -- the rest is verb-free regions

Chasing the small residue in the two-word tiling (97.5% on `resnet18d`,
92% on `mnasnet`) exposed an overstatement two sections up, which is
corrected here in place rather than rewritten there.

**What was overstated.** "With all five verbs counted, the bulk is a
uniform two-word stream." The 97.5% figure is real, but it measures the
spacing *between consecutive verb headers* -- and the gap histogram it
came from listed only the most common gap sizes. The full histogram has
a long tail of very large gaps (in `resnet18d`: 1,047; 1,042; 688; 510;
487 words ...). Counting words instead of gaps:

```
                          resnet18d          mnasnet_small
bulk words                12,188             19,564
verb-header + operand     4,574   (37.5%)    3,742   (19.1%)
verb-free regions (>=8w)  44 regions,        127 regions,
                          7,494 w (61.5%)    15,680 w (80.3%)
regions per op            0.24               0.95
largest region            1,047 words        1,597 words
```

So the correct statement is: **the five-verb instruction runs are
tightly two-word (97.5%) but make up only 37% of `resnet18d`'s bulk and
19% of `mnasnet`'s; the remainder is 44 (resp. 127) verb-free regions
of other word families** -- `23 00`, `16 00`, `84 08`, `00 80` in
`resnet18d`; `81 20` and `20 03` dominating in `mnasnet` -- whose
content is high-entropy and whose format is not decoded. That family is
where the tiny model's four non-deterministic noise bytes lived (`23 00
xx 82` words), which fits these regions being data/descriptor tables
rather than instructions, but "table" is an inference; only the
counts, sizes and families are measured. `mnasnet`'s ~one region per op
(0.95), versus `resnet18d`'s one per four ops, is also why its tiling
residue was larger -- not a sixth verb: no other first byte >= 0x80
occurs more than 5 times in the `VV 00 x0 yy` shape in either model.

**Two more prologue facts from the same pass.** From the first field
write at byte 297, the two real classifiers share a **byte-identical
158-byte prologue** (they share only 21 bytes with the tiny synthetic
model, whose input is float `4x16x16` rather than `uint8` NHWC
`224x224` -- so the prologue plausibly carries input/preprocessing
setup). Inside it, all three models carry the same compact ladder
`00 90 84 08, 00 a0 84 0a, 00 b0 84 08, ... 00 f0 84 08`: seven 4-byte
units of the shape `00 xx 84 vv` -- field `xx` stepping `0x10`, bank
`0x84`, a **one-byte value** and **no verb byte**. That is a second,
short-form write encoding, alongside the 8-byte `[VV 00 xx yy][32-bit]`
form and the 7-byte 3-byte-operand slots already noted; a parser needs
all three.

### Second correction: the "verb-free regions" are the same instruction stream, drifted off the 4-byte grid

The section above split the bulk into two-word verb runs (37%) and
"verb-free regions" (63%) and floated "data/descriptor tables" for the
latter. Testing that directly, locally, overturns it -- stated in place.

**The regions are full of instructions the grid could not see.** Counting
verb-shaped `VV 00 x0 yy` words *inside* the regions at every byte
phase: `resnet18d` has 73 / 43 / 54 of them at phases 1 / 2 / 3 (and
zero at phase 0 only because phase 0 is what defined the regions);
`mnasnet` has 197 / 185 / 160. The compact `00 xx 84 vv` short-form
write appears **417** times inside `resnet18d`'s regions and **580**
times inside `mnasnet`'s, at any phase. The regions are not tables; they
are stretches where the instruction stream's alignment has drifted off
a fixed 4-byte grid -- exactly what 7-byte and 4-byte instructions
interleaved with 8-byte ones must produce. Fixed-size records are also
ruled out on their own terms: the regions' best internal periods are
4-8 bytes with weak match (0.11 / 0.26), their sizes are rarely 32- or
64-byte multiples, and only 0.7% / 2.9% of their 32-byte windows repeat.

**A variable-length walker recovers what the grid missed -- and shows
what is still unknown.** Walking from byte 297 and consuming an 8-byte
verb instruction, a 7-byte one when the next header lands at +7, or a
4-byte compact write, and otherwise stepping one unknown byte:

```
                       fixed 4-byte grid    variable-length walker (3 known forms)
resnet18d  (48,531 B)  37.7% explained      43.8%   (2,443 x 8-byte, 16 x 7-byte, 398 x compact)
mnasnet    (78,035 B)  19.2%                27.4%   (2,367, 49, 535)
tiny       ( 3,371 B)  15.9%                39.4%   (  116,  8,  86)
```

The walker beats the grid everywhere, most on the tiny model, and the
bytes it cannot explain sit overwhelmingly in **short** gaps (the most
common unknown-run lengths are 5, 12, 1, 47, 6 and 7 bytes) with a few
long stretches (the longest 1,352 bytes in `resnet18d`). Short gaps
between recognized instructions are what additional, not-yet-known
short instruction forms look like -- not what a data table looks like.

So the corrected picture, replacing "37% two-word runs + 63% regions":
**mcode's bulk is one variable-length instruction stream with at least
three encodings, of which three forms are known and explain 27-44% of
the bytes; the rest is further forms, mostly short, plus a few long
stretches, all undecoded.** The "37%" and "verb-free" figures were
artifacts of measuring a variable-length stream on a fixed grid.

### The regions fault like instructions on the device -- and a chance baseline withdraws the "short-form family"

**Device census: the former "verb-free regions" are validated
structure.** Flipping one byte at the start, middle and end of each of
the ten largest regions in the real `resnet18d` blob (30 flips, one
patched model per run, retry-once on the known transient):

```
region (byte, len)   start / mid / end
(18228, 4180)        F D =
( 6844, 4160)        F F F
(15436, 2744)        F D F
(11812, 2032)        D F =
( 4896, 1940)        F F F
(  400, 1772)        = F F
( 3200, 1688)        F F F
(29188, 1352)        F F F
(28032, 1148)        F F F
( 2180, 1012)        F F F
                     24 fault / 3 change output / 3 inert
```

**80% of flips inside these regions fault** with `0x8030070C` -- a
higher rate than the original whole-blob sweep (61%). A data table
answers corruption with wrong values (`D`) or nothing (`=`); a
sequenced instruction stream answers it with the validator. This is the
hardware-side confirmation of the previous section's reading, and it is
the strongest single result in this arc: the regions are instructions.
The device stayed healthy throughout (70 C, memory at baseline).
One honest variance note: re-running the four largest regions' twelve
flips on a *fresh* build gave 7 faults, not the 10 seen here -- per-flip
outcomes at a fixed offset can differ between builds (the known
non-deterministic label bytes are the obvious suspect), so the
regression test asserts a majority of faults, not the exact count.

**A chance baseline, applied late, withdraws a claim.** Tokenizing the
bytes the walker could not explain suggested a family of short
bank-tagged writes -- `00 ff TT vv` (4 B), `01 ff xx TT vv` (5 B),
bare `TT vv` (2 B), tag `TT` in `0x81..0x84`, with the prefix and tag
tending to sum to `0x84` -- and a walker taught those forms "explained"
61-70% of the bulk. Checked against chance, that dissolves: `0x00` is
28.5% of `resnet18d`'s bulk bytes and tag bytes 6.5%, so a `00 ?? TT`
pattern arises at ~1.8% of positions by chance against 2.7% observed --
**only 1.5x chance** (1.3x on `mnasnet`, 1.4x on the tiny model) -- and
the bare 2-byte form is at or below chance. The verb forms, by
contrast, are more than 100x chance (five specific first bytes, a zero,
a multiple of 16). So: the compact `00 xx 84 vv` **prologue ladder** is
real (a stepping field across three models is not chance), but
generalizing it to a stream-wide short-form family is **not supported**
at present, and the "a parser needs all three forms" sentence two
sections up is too strong -- the third form is established in the
prologue only. Coverage figures that lean on the loose short forms are
inflated; the defensible number remains the three-form walker's 27-44%,
of which the specific verb instructions are the reliable part.

Net, stated carefully: the regions are instructions (device-confirmed),
their alignment drifts (off-phase verbs), and their *encoding* is still
mostly unknown -- the next real step is a form-by-form decoding with a
chance baseline applied *before* each form is admitted, not after.

### Third correction: with the right null, the short-form family is real after all

Doing exactly that -- applying a chance baseline *properly* -- reverses
the withdrawal just above. The "1.5x chance" figure came from an
independence estimate (`P(byte == 00) x P(byte in 0x81..0x84)`) against
the greedy walker's start rate. That is the wrong null for a
*positional* pattern: what makes `00 ?? TT ??` an instruction form is
that a tag byte sits at a fixed offset after a prefix byte, and an
independence product says nothing about offsets. The right null keeps
every byte and destroys only the order: shuffle the non-verb bytes of
each bulk (20 permutations, identical byte histogram) and count each
pattern in the shuffles.

```
pattern                       resnet18d: observed / null (+-sd)   ratio    z
00 ?? TT ??  (4-byte form)    1485 /  382 (+-13)                  3.9x    82
01 ?? ?? TT ??  (5-byte)       496 /  175 (+-12)                  2.8x    27
02 ... TT ??  (6-byte)         229 /   68 (+- 5)                  3.4x    31
03 ... TT ??  (7-byte)         154 /   63 (+- 8)                  2.4x    12
00 x0 84 ??  (prologue ladder) 393 /   29 (+- 5)                 13.7x    72
prefix + tag == 0x84          1286 /  191 (+-15)                  6.7x    75
                              mnasnet: 3.9x / 4.1x / 3.8x / 4.8x / 9.4x / 8.3x, z up to 172
```

Every short form is far above its permutation null -- z-scores from 12
to 172 -- and the prefix/tag complement (`00<->84`, `01<->83`,
`02<->82`, `03<->81`) is 6.7-8.3x chance in both models. So the family
of short bank-tagged writes **is real statistical structure**, the
"withdrawal" above was an over-correction from a mis-specified
baseline, and the honest state is: the forms exist; their exact widths
are the best reading of where the tag byte lands (offset `prefix + 2`);
and the coverage they add on top of the verb instructions (to ~61-63%
of the bulk without the permissive 2-byte form) is legitimate. The
bare 2-byte `TT vv` form remains unsupported (it has no positional
structure to test) and stays out of the count.

The methodological lesson is the durable part: three corrections in a
row on one topic came from three different baselines -- a fixed grid,
an independence estimate, and finally a permutation null. Only the last
is appropriate for order-dependent structure, and it is the one this
README should have reached for first.

**The width rule, measured.** With the permutation null in hand, the
"widths are the best reading" hedge can be replaced by a measurement:
for each prefix `p` in `0..3`, where after the prefix does a tag byte
sit more often than chance?

```
                         offset after the prefix (observed / null)
prefix    k=1    k=2    k=3    k=4    k=5           mnasnet peak
p=0      0.25   3.93   0.19   0.41   0.33           k=2  3.74x
p=1      0.80   0.20   2.93   0.09   0.65           k=3  4.01x
p=2      1.44   0.43   0.37   3.40   1.10           k=4  3.72x
p=3      0.95   0.77   1.63   0.49   2.49           k=5  4.72x
```

The enrichment peaks at **exactly `k = p + 2`** for every prefix in
both models, with every other offset at or below chance -- and at that
offset the complement tag `0x84 - p` is elevated 5-12x while other tags
are 0.9-3.3x. So the unit is `[p] [p+1 payload bytes] [0x84 - p]
[value]`, **`p + 4` bytes long**: 4, 5, 6, 7 bytes for `p = 0..3`. The
prefix encodes the payload length and, most of the time, the bank. That
is a decoded *format* for the short instructions, not their semantics
-- what the payload bytes and the value mean per bank is still open --
but it is the first part of the non-verb encoding that can be stated as
a rule rather than a pattern.

**One step into the payload: the bank tag selects the payload's kind.**
For the dominant 4-byte units `00 ff TT vv` (1,485 in `resnet18d`,
3,500 in `mnasnet`), the field byte `ff` is a multiple of `0x10` far
more often for tag `0x84` than for the others -- and bank `0x84`'s
commonest fields are exactly the prologue ladder's (`0x80, 0xb0, 0x70,
0xc0, 0xa0`), i.e. the same 16-byte-granular field space the `a1` verbs
use. Banks `0x81..0x83` instead carry small integers in `ff` (`1, 5, 8,
9, 0xe, 0xd`), which read as indices or counts rather than field
offsets. Some units are constants (`00 b0 84 08` occurs 61 of 61 times
in `resnet18d`); others carry a small numeric range (`00 80 84 vv` takes
16 values, `0x56..0x5c`). Overall `ff` is 16-granular in only 35.6% of
units against a 20.4% shuffled null (25.3% vs 21.1% in `mnasnet`), so
this is a per-bank distinction, not a property of the family --
reported as the shape of the data, not a decoding of it.

### The layout that explains all of it: two configuration blocks, then 180 clean op programs

Splitting the whole token stream at every `a1 40 02` write and asking
where the short units and unknown bytes actually are gives the blob's
real layout, and it is simpler than the section-by-section picture
suggested:

```
resnet18d (49,080 B)                       verbs  short units  unknown bytes
  bytes    297 .. 11,433  config block A       50          750         7,282
  bytes 11,433 .. 30,816  config block B      159        1,444        11,396
  bytes 30,816 .. 48,496  180 op programs   ~2,160            0             0
  bytes 48,496 .. 48,828  epilogue             11            0           244
  (then the back FlatBuffers copy)

mnasnet_small (78,584 B)
  bytes    297 .. 19,977  config block A      141        1,574        10,756
  bytes 19,977 .. 64,672  config block B      502        4,242        17,576
  bytes 64,672 .. 78,008  132 op programs   ~1,650            0             0
  bytes 78,008 .. 78,332  epilogue             12            0           228
```

**180 of 183 `resnet18d` segments, and 132 of 135 in `mnasnet`, contain
no short unit and no unknown byte at all** -- each is exactly the
eleven-instruction verb template (~96 bytes) and nothing else. Every
short-form unit and every undecoded byte in the entire blob sits in two
large configuration blocks at the front (bounded by two *setup* `40 02`
writes whose operands -- `ff 00 00 00` and one more -- are not real
op offsets) plus a ~330-byte epilogue. The blocks are 62% of
`resnet18d`'s blob and 82% of `mnasnet`'s.

This is why the coverage numbers looked the way they did: the "37%" of
the bulk that the verb walker explained was, to first order, *the
entire op-program region*, which is parsed end to end, while the
"unexplained 63%" was the configuration blocks, where verbs are a
minority among short units and undecoded bytes. It also corrects the
earlier reading that the regions were interleaved "about one per four
ops": they are not interleaved with ops at all. The per-op program is
done; the frontier is precisely two configuration blocks whose short
units follow the width rule and whose remaining bytes (18.9 KB in
`resnet18d`) are the actual undecoded content of mcode.

### Into config block B: the width rule reaches prefix 4, and a candidate second header family

First pass over the larger configuration block (`resnet18d` bytes
11,433..30,816; `mnasnet` 19,977..64,672), asking what the undecoded
residue is made of once every validated instruction is removed.

**The width rule extends, a little.** Testing prefixes `p = 4..15` the
same way (tag byte at offset `p + 2`, against a shuffled null of the
block): `p = 4` holds in both models (2.6x, n = 68; 2.4x, n = 203);
`p = 5` and `p = 9` reach ~2.2x in `mnasnet` only; `p = 6..8` and
`10..15` sit at chance. Admitting `p <= 4` shrinks `resnet18d`'s block-B
residue from 11,396 to 11,134 bytes -- the family is `p = 0..4` (units
of 4..8 bytes), and it is not where most of the residue goes. Block B
is **42.6%** explained in `resnet18d` and **63.4%** in `mnasnet`.

**What remains is not records, and not noise.** The residue's runs are
still instruction-sized (2, 4, 8, 6, 15, 13, 7, 5 bytes; longest 136),
and its best internal period is weak in absolute terms (5 bytes, 8.3%
matching) but **three times** the same residue shuffled (2.7%) -- real
local structure, not fixed-size records. Its most common 2-byte pairs
are `23 00` (139), `00 00`, `03 00`, `18 9f`, `8b 18`, `16 00` in
`resnet18d` and `00 00`, `16 00`, `04 00`, `9f 16`, `01 00`, `16 04` in
`mnasnet`: a set of `XX 00` pairs with `XX` in `{0x23, 0x16, 0x03,
0x04, 0x01}` -- the same `23 00` family the tiny model's four
non-deterministic noise bytes lived in. That reads as a **second header
family** (`XX 00` with a first byte outside the verb set), i.e. more
instruction forms with their own widths, and it is the next thing to
validate with the permutation method -- the width-rule test above is
exactly the template for it.

### Fourth correction: seventeen tags, the trailing byte is the register, and `23 00` was a prefix byte

Running that validation changed three earlier readings. All numbers are
block B of the cached `resnet18d` and `mnasnet_small` builds plus the
tiny two-conv model, against a shuffled copy of the same block.

**The tag set is not 0x81..0x84.** Admitting any tag byte in
0x81..0x9f (everything below the verb bytes 0xa1..0xa9) at offset
`p + 2` lifts block B from 42.6% to **82.7%** explained in `resnet18d`
while the null only moves from 10.7% to 23.3%. Per tag, the ratio over
the null splits cleanly: 0x81..0x86, 0x89..0x8d, 0x94, 0x95, 0x9b..0x9d
and 0x9f are 2x..9x above it in both models (0x9b 8.6x, 0x9d 6.0x, 0x95
5.9x, 0x8b 4.8x, 0x9c 4.8x, 0x9f 3.6x in `resnet18d`, n = 416 for
0x9f), while 0x90, 0x91, 0x96..0x99 sit *at or below* it (0.1x..0.5x)
and 0x87, 0x88, 0x8e, 0x8f, 0x92, 0x93, 0x9a, 0x9e reach 2x in one
model only. With that data-driven set of **17 tags** and `p <= 4`,
block B is **76.1%** explained in `resnet18d` (null 19.5%, 3.9x),
**81.8%** in `mnasnet` (null 22.7%, 3.6x) and 75.7% in the tiny model
(null 26.5%); the residue falls from 11,134 to **4,636** bytes and from
16,990 to **8,146** bytes. The `tag = 0x84 - p` pairing the third
correction reported is the *dominant* pairing, not a rule: 0x84 takes
`p = 0` in 539 of 601 units, but 0x81 takes `p = 3` in only 48 of 282
`resnet18d` units (1,261 of 1,748 in `mnasnet`). The tag does not
encode the width; `p` does.

**The trailing byte is the register, and the payload is the value.**
The byte after the tag is even in **2,909 of 2,911** `resnet18d` units,
6,248 of 6,253 `mnasnet` units and 258 of 258 tiny-model units (99.9%),
against a 64% background; the first payload byte is even at exactly the
background rate (59.6%, 63.4%, 77.1% vs 64.5%, 63.8%, 71.7%). Only a
2-byte-granular offset looks like that. So the short unit reads
`[p][p+1-byte value][tag][register]`, not `[p][field][tag][value]` as
the layout section had it -- the verb form `a1 00 xx yy <32-bit>`
addresses at 16-byte granularity through `xx`, the short form at 2-byte
granularity through its last byte. The test helper's tuple still
returns the first payload byte in the slot it always did; only the name
changed.

**`23 00` is a prefix byte, not a header.** With the wide tag set,
1-byte residue runs are the largest class (381 of 1,178 runs in
`resnet18d`), and they sit *immediately before an ordinary unit*: `23`
precedes a `p = 0` unit 139 times, `03` 102 times (plus 13 before a
`p = 1` unit), `3c` 21 times; `mnasnet` uses `04` (104), `2d` (50),
`03` (46), `05` (43), `1c` (42). The `23 00 ...` pairs the previous
section counted are `23` followed by the `00` that opens a `p = 0`
unit. That is a one-byte prefix with model-specific values on ordinary
units -- what it modifies is open -- and not a second header family.
The tiny model's four non-deterministic bytes are inside one such
prefixed unit.

**What does not validate.** A payload-less `[tag][even byte]` pair
(the natural `p = -1`) would push `resnet18d` to 89.5% explained, but
the null rises to 40.8% with it (ratio 2.2 from 3.9) and per tag it is
at or below chance -- 0x84 pairs 59 real vs 266 shuffled, 0x9f 207 vs
315, 0x81 126 vs 226; only 0x87 (112 vs 76; 183 vs 105) and 0x88 (154
vs 101; 137 vs 86) exceed it, at ~1.5x. Not admitted. A 2-byte `e1 XX`
pair (odd `XX`) is 21 vs 13 in `resnet18d` and 86 vs 31 in `mnasnet`
-- a candidate, not a form. No tag byte has a fixed argument length
(for every tag with n > 100 the modal distance to the next tag byte
holds < 50% of cases), so the 0x85..0x9f family are tags in the same
width rule, not opcodes of their own.

**Both configuration blocks open with the same 158 bytes.** Block A
and block B of `resnet18d` and of `mnasnet` all begin with the same
158-byte prologue (four-way identical); the tiny model has the same
prologue with one operand changed -- the `a1 00 60 02` write at +16
carries `0x31480` (201,856) in both 224x224 models and `0xd80` (3,456)
for the 1x4x16x16 input -- and its shorter blocks (608 and 2,231 bytes)
lack the region that follows. The first per-model bytes after the
prologue are `p = 2` units on tag 0x81/0x82 whose payloads are
`09 bf 01` vs `09 c3 01` and `09 80 03` vs `09 3c 05` with `09 00 07`
shared: 16-bit parts 447 vs 451, 896 vs 1,340, 1,792, the first model
quantities in the block and unmatched to any count tracked here
(op count, block sizes, token counts).

Test: `test_resnet18d_short_form_tags_are_seventeen_wide_and_trail_a_register`
in `tests/test_axera_mcode_structure.py` (fresh `resnet18d` build,
fixed-seed null, no device).

### Fifth correction: every tag byte is real, so is the bare pair -- the explained-bytes null was the wrong null

The section above rejected 14 tag bytes and the payload-less
`[tag][even]` pair because admitting them raised the *shuffled* block's
explained fraction almost as much as the real one's. That comparison is
sound for a common form and useless for a rare one: shuffling the block
leaves ~60% of it unexplained, and a rare pattern then matches by
chance inside that residue about as often as it occurs for real. The
right null for a rare form is conditional: given the form's own anchor
byte, does the byte the rule constrains behave as the rule says, and
does it do so in the shuffled block?

**Every byte in 0x81..0x9f is a tag.** Admitting all 31 as p-unit tags,
the trailing byte is even in essentially every real unit *for every
tag*: 0x87 52/52, 0x88 95/95, 0x8e 26/26, 0x92 31/31, 0x93 22/22 in
`resnet18d` (the "rejected" ones), alongside 0x84 601/601 and 0x9f
416/416; `mnasnet` likewise (0x87 14/14, 0x88 50/50, 0x92 26/26, 0x93
51/51, 0x9c 166/166). In the shuffled block the same units are even at
50..70% for every tag (0x84 100/151, 0x9f 79/116). The 17-tag set was
a false rejection of the rarer tags, not a property of the format.

**The bare `[tag][register]` pair is real.** Among residue runs of
exactly two bytes that begin with a tag byte, the second byte is even in
**265 of 265** (`resnet18d`), **598 of 600** (`mnasnet`) and 29 of 29
(tiny) -- against a 60% background inside the residue and 7 of 12 in
the shuffled block. That is the `p = -1` width: a register touched with
no payload. Two more forms show the same signature and are noted, not
yet tokenized: `e1 XX` with `XX` *odd* in 13/14, 48/48 and 7/7 cases
(the verb byte 0xa1 with bit 6 set), and 0xa1 itself acting as a tag
when not followed by `00` -- bare `a1 5e`-style pairs are even 8/8 and
20/20, and `01 70 04 a1 5e` is a `p = 1` unit on tag 0xa1.

**Where block B stands.** With all tags, `p <= 4` and bare pairs, block
B is **93.5%** explained in `resnet18d` (residue 1,267 of 19,383
bytes), **93.3%** in `mnasnet` (3,004 of 44,695) and 85.9% in the tiny
model. The shuffled block reaches 42.9% under the same rule -- the
inflation that misled the section above, and the reason the parity
counts, not that figure, carry the claim. The residue is now mostly the
one-byte prefixes: 641 of 762 residue runs in `resnet18d` (`23` 150,
`03` 118, `16` 72, `3c` 28, `05` 21, `0d` 19), 992 of 1,415 in `mnasnet`
(`04` 115, `05` 99, `03` 62, `1c` 52, `2d` 50), and 559 + 63 of those
641 are followed directly by a p-unit or a bare pair. The multi-byte
leftovers are a few 6..7-byte runs that sit immediately before a verb
(`09 20 02 00 00 01 00`, `09 c0 0c 80 fe 01 01`, `08 0b 55 fc 55 01`)
-- a candidate pre-verb form with too few instances to test yet.

Test: `test_resnet18d_every_tag_and_the_bare_pair_pass_the_parity_null`
in `tests/test_axera_mcode_structure.py`; the previous test's bare-pair
assertion is replaced by the corrected claim.

### Sixth correction: there is no prefix byte -- tag 0x9f's unit is one byte longer

Asking what *precedes* the "prefix" bytes settles them. In `resnet18d`
block B, 592 of the 641 one-byte residues follow a unit whose tag is
0x9f (394 after a p-unit on 0x9f, 198 after a bare `9f reg` pair); in
`mnasnet` it is 797 of 992. Per tag, the share of units followed by
exactly one leftover byte is **592 of 624 for 0x9f** and at most 1% for
every other tag (0x81 3/410, 0x84 0/661, 0x95 0/237, 0x8b 0/187);
`mnasnet` 797 of 875 for 0x9f, at most 6% otherwise. A bare 0x9f pair is
followed by one leftover byte in 198 of 208 cases where every other
bare tag is followed by the next unit directly. So 0x9f's unit is
`[p][value][9f][register][extra]` (and bare `9f reg extra`), one byte
longer than any other tag's, and the fourth correction's "one-byte
prefix with model-specific values" is that extra byte read as belonging
to the wrong unit. Its values are the ones the prefix table listed
(`23`, `03`, `16`, `3c`, `0d`, `26` in `resnet18d`; `04`, `05`, `03`,
`1c`, `2d`, `00` in `mnasnet`), below 0x40 in 98.3% and 89.2% of cases
and with no parity constraint (36% and 60% even) -- a small value, not
a register. The 0x9f write shares its register with the *next* unit in
320 of 544 and 315 of 647 cases, so 0x9f looks like a modifier issued
just before a write to the same register; what the extra byte selects
is open.

**Where the blocks stand.** With every tag, `p <= 4`, bare pairs and the
0x9f rule, block B is **96.5%** explained in `resnet18d` (673 of 19,383
bytes left) and **95.2%** in `mnasnet` (2,152 of 44,695); the shuffled
block stays at 44%. Under the same rule the whole mcode is covered
region by region -- `resnet18d`: block A 93.8%, block B 96.5%, the op
programs 98.7%, **95%+ of every tokenized byte** and 94% of the whole
blob including the untokenized FlatBuffers header and trailer;
`mnasnet` 92.6% / 95.2% / 98.5%. What remains in block B is a short
list: `e1 XX` pairs (14, 48), `01 70 04 a1 5e`-style units on tag 0xa1
(12, 37) and bare `a1 reg` pairs (8, 20), a handful of single `05`/`08`/
`06` bytes, and the 6..7-byte runs that sit directly before a verb
(`09 20 02 00 00 01 00`, `08 0b 55 fc 55 01`; 6 + 6 in `resnet18d`, 19 +
17 in `mnasnet`). The tiny model's block B stays at 86% because its
ops region carries proportionally more of those pre-verb runs.

Test: `test_resnet18d_tag_9f_units_carry_one_extra_byte` in
`tests/test_axera_mcode_structure.py` (fresh build, no device).

### The op programs are fully tokenized; the "trailer" is a five-table FlatBuffers tail the header points to

The op region's last residue turned out not to be instructions. Under
the full rule its only leftovers were two runs at the very end whose
bytes read `0c 00 10 00 0f 00 0e 00 08 00 04 00` (a FlatBuffers vtable)
and `7b 7d` (the string `{}`), so the "252-byte trailer" every earlier
section stopped short of was too short by about 230 bytes. Measured on
all three models:

- **The stream ends 7 bytes after its last verb** -- a 7-byte `a2`
  verb -- at 48,583 (`resnet18d`), 78,103 (`mnasnet`) and 3,447 (tiny);
  then 17, 1 and 1 zero bytes; then a FlatBuffers vector of five table
  offsets (`05 00 00 00 | 20 | 2c | 50 | 74 | 98`) **480 bytes before
  the end** (472 in the tiny model). The header's 32-bit word at offset
  272 is a FlatBuffers offset to exactly that vector in all three
  (272 + 48,328 = 48,600; 272 + 77,832 = 78,104; 272 + 3,176 = 3,448),
  so the header and the tail are one FlatBuffers structure wrapped
  around the instruction stream.
- **With the boundary right, the op programs are 100% tokenized** in
  all three models (0 residue bytes in 17,767, 13,431 and 311). The whole
  stream is **98.1%** explained in `resnet18d` (block A 97.75%, block B
  96.5%, ops 100%), 96.0% in `mnasnet` and 88.0% in the tiny model;
  counting the untokenized header and tail against it, 96.5%, 95.1% and
  70.7% of the whole blob.
- **Every configuration block ends the same way the header does.** The
  header's last 17 bytes, `2b a7 00 00 0a 00 00 00 00 a2 00 00 00 13 00
  40 00`, reappear byte for byte as the last 17 bytes of block A in all
  three models (`40` is `30` throughout `mnasnet`), preceded by zero
  bytes. Their last 8 bytes are an 8-byte `a2 00 00 00` verb with
  operand `0x00400013`; the stream's own final bytes are `a2` verbs of
  the same shape with per-model low halves (`0x7b2`; `0x642`, `0x652`;
  `0x12`, `0x22`). So the 297-byte "header" is a FlatBuffers header of
  280 bytes plus the same block terminator the config blocks use.
- **The five tables share one schema.** Table 0 has four fields
  (`59393`, `260`, then two per-model words: 2,228 / 3,812 in
  `resnet18d`); tables 1..4 have six: a type word (4,097 for tables
  1..3, 2,561 for table 4), an id (257, 258, absent, absent), two
  per-model counts, a packed word that is always the second count
  shifted left 8 bits plus 1 (865,281 = 3,380 x 256 + 1, in every
  table of every model), and a byte-count-sized value. Table 4's last
  field is block A's length minus 21 in both large models (11,115 vs
  11,136; 19,658 vs 19,680) -- the 21 bytes being the terminator above
  plus four zeros -- but 606 vs 608 in the tiny model, so that reading
  is a lead, not a result. `resnet18d`'s values, tables 1..4: counts
  (432, 3,380), (612, 2,768), (1,376, 1,392), (1,392, --); sizes 3,455,
  4,881, 11,000, 11,115. None equals a unit, verb or byte count measured
  here; per-engine instruction budgets are the obvious guess.

Test: `test_resnet18d_stream_ends_at_a_five_table_flatbuffers_tail` in
`tests/test_axera_mcode_structure.py` (fresh build, no device).

### The tail is the segment table: the stream is its segments, in reverse, and `a7` is the sixth verb

The five tail tables decoded themselves once a second model shape was
in hand. Verified on seven mcodes -- `resnet18d`, `mnasnet_small`, the
tiny two-conv model, a 1-layer Mistral compiled through the ONNX path
(`build_from_hf_checkpoint()`, see the LLM sections below) and, from a
real `pulsar2 llm_build` of `HuggingFaceTB/SmolLM2-135M`, both `neu
mode` subgraphs of `llama_p512_l0_together.axmodel` and the LM head
`llama_post.axmodel`:

- **Field 2 is a segment length in 8-byte words and field 3 counts
  down.** In every table of every mcode, `f3[k] = f3[k-1] - f2[k]`, and
  table 0's `f3` is the sum of every later table's `f2`. The packed
  field 4 is `f3 << 8 | 1` throughout.
- **The segments tile the blob exactly, in reverse table order.** Take
  the tail vector's offset, subtract 8 x the sum of every table's `f2`,
  and you land on the end of the FlatBuffers header (280 bytes for every
  one-input CNN, 436 for the ONNX-path Mistral with two inputs, 700 and
  744 for the two `llm_build` subgraphs, 296 for the post model); lay
  the segments out from there, last table first, and the last one ends
  on the tail vector to the byte in all seven mcodes. The last table is
  configuration block A exactly (11,136 = 1,392 x 8); the "block B" the
  earlier sections treated as one region is really *three* segments in
  `resnet18d` (11,008 + 4,896 + 3,456 bytes); the op programs are table
  0's segment.
- **Every segment opens with an `a7` verb, and `a7` is a verb.** The
  17-byte "block terminator" was misread: `2b a7 00 00 0a 00 00 00 00`
  is one leading byte (`2b`, `24`, `33`, `3b` -- segment-specific) and
  an 8-byte `a7 00 00 NN <u32>` instruction of exactly the verb shape
  (`NN` = 0x0a in the CNNs and the post model, 0x10 in the ONNX-path
  Mistral, 0x1e in the `llm_build` layer), followed by the segment's
  first real verb. `llm_build` op programs then use `a7 00 00 02 02 00
  00 00` freely -- 205 and 261 times inside the two subgraphs' op
  segments, where the five-verb tokenizer left 8-byte holes.
- **Segment types.** Table field 0 is `X << 8 | 1`: 0x0a for the first
  configuration segment, 0x10 for the others and 0xe8 for the op
  programs in every CNN and the ONNX-path LLM; the `llm_build` layer's
  fifteen segments use 0x0a, 0x10, 0xe8, 0xd0, 0xb8, 0xa0, 0x88, 0x0c,
  0xf0 with its 112-program op segment on 0x88 and two 14/15-program
  segments on 0x0c/0xf0 -- different engines or queues is the obvious
  reading, untested.

**Coverage, by segment, with `a7` admitted.** Op-program segments are
**100% tokenized in all seven mcodes** past the 4..5 marker bytes that
straddle a segment's opening word boundary (resnet18d 17,824 bytes; mnasnet
13,472; the `llm_build` layer's 12,160 + 1,984 + 2,080 and 19,712 + 992
+ 992; the post model's 72,448 with 772 programs). Configuration
segments run 90..99%. Whole streams (zero padding trimmed): `resnet18d`
98.3%, `mnasnet` 96.3%, tiny 91.7%, ONNX-path Mistral 95.4%, `llm_build`
layer 96.1% and 96.5%, post model 99.7%.

**What the LLM builds add.** The ONNX-path Mistral (80 ONNX ops, 75 op
programs, 21 s to compile) keeps the CNN op-program skeleton verb for
verb -- `40.02, 50.01, 50.01, a8 40.03, 50.03, 50.01, a3, 50.01, a9,
[a2], a8 30.02`, the two most common variants covering 57 of 74
programs -- with the trailing slot varying by op: `a1 30.03` in 9
programs, `a1 20.02` (a slot the CNN templates lack) in 8, neither in
the rest; its two graph inputs get two `a2` verbs in the segment opener
where the CNNs' one input gets one, but the `llm_build` layer (5 inputs)
gets one and the post model (1 input) gets four, so that is not an
input count. The `llm_build` subgraphs (decode and prefill, 145 and 167
programs, 92 s for all 30 layers plus the head) are the first mcode
with programs of 104..144 bytes: the skeleton plus `a7` verbs inside.
Their config-segment openers start `1e 00 00 00 00 a2 00 00 00 83 01
c0` -- the `a7` marker's operand straddling the 8-byte segment
boundary, so segment lengths are exact but not marker-aligned.

**What `a7` does inside a program.** In the `llm_build` layer's 0x8801
segment the two most common skeletons (85 of 112 programs in the decode
subgraph, 67 of 151 in prefill) are the CNN core with two `a7` verbs
inserted at fixed slots: `40.02 | 50.01 | 50.01 | a7.1e | a8 40.03 |
50.03 | 50.01 | a3 | 50.01 | a9 | [a2] | a7.02 | a8 30.02`, where `a7.1e`
carries operand 0 and `a7.02` operand 2. The same `a7.1e` with operand 0
opens every segment of these mcodes (`a7.0a` in the CNNs, `a7.10` in the
ONNX-path Mistral), so `a7` reads as a synchronization verb -- a wait
or fence on channel `yy` before the engine-dispatch verbs and a signal
on channel 2 after them -- that the CNN op programs never need and the
BF16 layer programs use on every op. The layer's two small segments (14
and 15 programs on types 0x0c01/0xf001 in decode, 6 and 6 in prefill)
hold the core followed by runs of 2..13 `a2 00.00` verbs, plus one
program per segment that writes every field 0x40..0xe0 of banks 2, 3
and 4 in order -- a full register load, presumably the KV-cache and
I/O binding.

**Confirmed on the AX650N: `a7` is a synchronization verb, and its
operand is what matters.** The layer-0 decode subgraph runs on the real
device only when fed valid inputs (`axcl_run_model`'s random bytes land
in the `indices` gather input and fault it with `0x8030070C`, exactly
as the harness's `run_on_device_with_inputs()` docstring warns); with
zero K/V caches, a zero hidden state and in-range indices it runs in
0.44 ms and its three outputs are bit-identical run to run. Hand-patching
its decode mcode and re-running, every outcome reproduced 3 of 3 times:

| patch (decode subgraph) | edits | result |
|---|---|---|
| in-program `a7.02` operand 2 -> 0 | 102 | runs, **all three outputs change** |
| in-program `a7.02` operand 2 -> 1 | 102 | runs, outputs change differently (K and V outputs become identical) |
| in-program `a7.02` channel 02 -> 1e | 102 | runs, outputs **identical** to baseline |
| in-program `a7.02` channel 02 -> 03 | 102 | runs, outputs change (same as operand 0) |
| in-program `a7.1e` operand 0 -> 1 | 103 | **fault `0x8030070C`** |
| in-program `a7.1e` channel 1e -> 02 | 103 | runs, outputs identical |
| segment-opening `a7.1e` markers, operand 0 -> 1 | 15 | **fault `0x8030070C`** |

So the operand-0 form (`a7.1e`, at every segment start and before each
op's dispatch verbs) is a wait or reset that must read 0, the operand-2
form (`a7.02`, after each op's dispatch) is a post whose value fixes
what the consumers of that op see -- change it and the pipeline still
runs to completion but with a different (racy or stale) data ordering,
producing wrong outputs with no fault -- and the channel byte
distinguishes 0x02/0x1e from 0x03 but not from each other. The
timing differences between variants were within run-to-run noise
(0.43..0.54 ms) and are not claimed.

**0xa1 is also a tag, and bit 6 of a tag flips the register's parity.**
The config-segment residue that remained is mostly three forms, and the
conditional-parity null settles all three (counts are the `llm_build`
layer's decode and prefill subgraphs; a fresh `resnet18d` build has 15
of 16 and 18 of 19 for the first and third): bare `a1 XX` pairs have an
even `XX` in
37 of 37 and 65 of 65 cases against 28:13 and 36:31 in the shuffled
segments; `01 .. a1 XX` p-units have an even register 31 of 31 and 76
of 76 times; and bare `e1 XX` pairs have an **odd** `XX` in 71 of 71 and
116 of 116 cases (`c1 XX` 4 of 4 and 10 of 10) against 4:2 and 8:2
shuffled. So the verb
byte 0xa1 doubles as a tag whenever it is not in verb position (`a1 00
x0`), and the 0x40 bit of a tag (0x81 -> 0xc1, 0xa1 -> 0xe1) selects the
odd register -- the other half of a 2-byte register pair, presumably.
Admitting them takes the config segments' non-zero bytes from 97.4% to
98.1% explained in `resnet18d`, 95.6% to 97.2% in `mnasnet`, and 95.9% /
95.5% to 97.0% / 97.2% in the two `llm_build` subgraphs (shuffled
segments stay at ~46% under the same rule). `_tokenize_mcode` takes
these as `odd_tags`.

**Confirmed on the AX650N: the segment table is a loader manifest --
but only its word counts are load-bearing, and the first version of
this paragraph got that wrong.** The first series of tail-table patches
(type word, word count, remaining count, total, byte size, packed word,
each on its own) reported a `0x8030070C` fault for *every* variant, 3 of
3 each. That table was measured after the first variant -- the type
word `0x8801 -> 0xe801` -- had already stalled the runtime for about two
minutes, and after the series the driver's heartbeat thread stayed
blocked until a reboot; the faults were the degraded device, not
validation. Redone on the healthy device, one variant at a time with an
`axcl-smi` health check and a baseline control run after each
(baseline bit-identical throughout):

| patch (decode subgraph tail tables) | result |
|---|---|
| a 6-field table's byte size `f5 - 64` | runs, outputs **identical** |
| the same table's packed word `f4 + 1` | runs, outputs **identical** |
| op segment word count `f2 + 1` | runs, outputs **identical** |
| op segment remaining count `f3 + 1` | runs, outputs **identical** |
| table 0 word count `f2 + 1` | **`Loading model failed`** -- rejected at load, no fault, device unharmed |
| table 0 total `f3 + 1` | **`Loading model failed`** -- same |
| op segment word count `f2 - 1` | **fault `0x8030070C`** at run, device unharmed |
| table 0 word count `f2 - 1` | **fault `0x8030070C`** at run, device unharmed |
| op segment type word `0x8801 -> 0xe801` | not repeated: the one patch that wedged the runtime |

So the loader reads the word counts (`f2`) and table 0's total (`f3`) to
lay the segments out: a count that overshoots the blob is rejected at
load, a count shorter than its segment lets the run read a truncated
program and fault, and a mid-table count one word too long merely reads
8 bytes of padding. The remaining count, the packed `f3 << 8 | 1` and the
byte size are compiler bookkeeping the runtime does not check. An
emitter must get the word counts and the total exactly right; the rest
it can mirror without consequence. The one caution stands: a bad *type
word* is the patch class that stalls the device rather than failing
fast, and it is the one to avoid.

**`llm_build` emits one instruction stream for all 30 layers.** Every
per-layer file's two mcodes are byte-identical to layer 0's from the
first byte of the FlatBuffers header to the tail vector; the only
differing bytes are in the tail's name string (`llama_p512_l0_together_
decode` / `_prefill_0` -- 1 byte for layers 1..9, 19 or 21 for the
two-digit layers, where the longer string shifts what follows it) and,
of course, in `npu_params` (3,963,652 bytes per layer, all different).
So a transformer layer's mcode is position-independent: the program
addresses its weights through the Wbt offsets the `40 02` writes carry,
and a layer's identity lives entirely in the weight table. An emitter
for this path has one 57,056-byte decode program and one 93,360-byte
prefill program to get right, not thirty.

Tests: `test_resnet18d_tail_segments_tile_the_stream_and_open_with_a7`,
`test_llm_build_layer_mcode_keeps_the_layout_and_uses_a7`, and
`test_onnx_path_llm_mcode_keeps_the_op_program_skeleton` in
`tests/test_axera_mcode_structure.py` (fresh builds, no device; the two
LLM tests skip unless the checkpoints are already in the HuggingFace
cache).

### Where the decoding stands: 98.4% of the instruction stream, across a CNN and two LLM paths

Measured on five real mcodes -- `resnet18d`, a 1-layer `tiny-random-mistral`
compiled through the ONNX path, both `neu mode` subgraphs of an `llm_build`
`SmolLM2-135M` layer, and the LM head -- with every validated form admitted
(six verbs, all tags 0x81..0x9f plus 0xa1, prefixes `p <= 4`, bare
`[tag][register]` pairs, the bit-6 odd-register tags, and tag 0x9f's extra
byte). Zero bytes are segment padding and are excluded on both sides:

| build | stream | non-zero bytes | unexplained | explained |
| --- | --- | --- | --- | --- |
| `resnet18d` | 48,320 | 34,634 | 496 | **98.57%** |
| `mistral` (ONNX path) | 27,104 | 20,252 | 619 | **96.94%** |
| `llama` layer, decode | 55,104 | 40,557 | 975 | **97.60%** |
| `llama` layer, prefill | 91,296 | 70,321 | 1,592 | **97.74%** |
| `llama` LM head | 124,736 | 77,790 | 169 | **99.78%** |
| all five | | 243,554 | 3,851 | **98.42%** |

Admitting the 7-byte companion write described below takes that to **98.69%**
(3,179 unexplained).

One trade-off is worth naming: treating 0xa1 as a tag (the fifth correction)
is a large net win -- it cuts the `llama` prefill subgraph's residue from
2,115 bytes to 1,301, and *improves* most op segments (the decode subgraph's
from 510 to 321) -- but in two builds it costs a handful of bytes inside op
segments that the narrower rule read exactly. Those segments are now asserted
at 99.9% rather than 100%.

The same rule, unchanged, reads a CNN and a transformer -- through either
compiler path -- which is the strongest evidence yet that the grammar is the
hardware's and not an artefact of one model.

**The width rule really does stop at `p = 4`.** The fourth correction
established `p <= 4` with the narrow tag set; with the corrected 31-tag set
the question reopens, since the leftover bytes (0x05..0x10) look exactly like
larger prefixes. They are not. Counting units per prefix across all five
mcodes against a shuffled stream: `p = 1` and `p = 2` run 3.2x over chance,
`p = 0` 1.8x, `p = 3` 1.6x, `p = 4` 1.4x -- and from `p = 5` upward every
prefix falls *below* chance (0.53, 0.58, 0.31, 0.25, 0.97, 0.29, ... down to
0.04 at `p = 16`). Raising `pmax` does raise apparent coverage, but it raises
the shuffled stream's more (46% -> 54% explained), so it is the greedy walk
over-fitting, not a real form. `p <= 4` is a hard limit.

**A 7-byte write that fills the slot below the next one.** Among the
remaining runs, the 7-byte ones have the shape `[X][field][bank][32-bit
operand]`, and their address is always exactly one field slot below the verb
that follows them: same bank one field lower, or the last field of the
previous bank (`09 f0 05 40 fc 03 06` then `a1 00 00 06 ...`: field 0xf0 in
bank 5, then field 0x00 in bank 6). The bank wraps confirm that
`(bank, field)` is one continuous address space, not two independent
selectors.

That adjacency is a strong enough anchor to make the form a first-class
rule: recognise a 7-byte unit only when the next eight bytes are an `a1`
verb writing the adjacent slot. Across the five mcodes it fires **104
times** and on their shuffled counterparts **zero times** -- perfect
discrimination, no threshold to argue about. Admitting it takes the tail
from 3,851 unexplained bytes to 3,179, and the whole-corpus figure from
98.42% to **98.69%**.

Many instances only became visible this way. `0a 90 0e 00 00 80 3f` writes
the float 1.0 to field 0x90 of bank 0x0e and is followed by `a1 00 a0 0e`,
the next slot up -- but the greedy walk used to swallow its first five bytes
as a spurious short unit and leave `80 3f` (the high half of 1.0f) stranded,
which is why `80 3f` was the single most common two-byte leftover. The
leading byte still varies (0x21, 0x05, 0x09, 0x0a, 0x7f, 0x02, 0x20, 0x1d,
0x1c) and what it carries is open; 0x21 is `a1` with bit 7 cleared, which
would fit a compact encoding that drops the 8-byte form's `00`, but the
other values do not.

Tests: `test_full_rule_explains_almost_every_stream_byte` and
`test_companion_writes_fill_the_slot_below_the_next_verb` in
`tests/test_axera_mcode_structure.py` (fresh builds of `resnet18d` and the
ONNX-path Mistral, no device); the `llm_build` layer's coverage is asserted
by its own test.

### A lossless codec: taking a model apart and putting it back together

Reading mcode and *writing* it are different problems, and the coverage
number above answers only the first. This is the first half of the second:
`_decode_mcode()` turns a stream into structured records -- verb, field,
bank and operand; prefix, payload, tag, register; the companion write's
address and value -- and `_encode_mcode()` writes those records back out.
The encoder reads nothing from the original blob, so a byte-exact round trip
proves the decode captures every bit the forms carry, which is the first
thing a generator needs.

It round-trips exactly on a real `resnet18d` and on the ONNX-path
transformer, with **over 95% of the bytes coming from recognised forms** and
the rest riding along as raw escapes. Rebuilding a whole `.axmodel` -- the
FlatBuffers header and tail copied verbatim, the stream re-encoded -- gives
back the original file.

**What that does and does not buy.** It means we can rewrite any field of any
instruction and emit a valid stream, which is what every hand-patch
experiment in this file has done by hand. It does *not* mean we can compile a
new model. Splitting `resnet18d`'s parsed bytes by whether a destination ever
receives more than one value:

| | `resnet18d` | `llama` layer |
| --- | --- | --- |
| single-valued destinations (emit by copying) | 25.3% | 18.4% |
| varying per op (need a rule) | 74.7% | 81.6% |
| of those, at destinations whose meaning is documented here | 24.6% | 16.0% |

So roughly a quarter of the stream is template, another sixth follows rules
this file has verified (the Wbt offset, the four dispatch steps, the tile
arena, the input size, the two synchronisation channels), and the remaining
half goes to 627 destinations in `resnet18d` alone that we can parse but not
compute. The largest single unknown is the `a2` verb: 360 distinct operands
in `resnet18d`, 206 in the `llama` layer, ~6% of the stream.

And the instruction stream is the small half of the problem. In the compiled
artifacts the weight table is 99% of the file (`mistral` 21.4 MB against a
27.5 KB mcode; the `llama` LM head 29.1 MB against 122.8 KB), and its layout
is not decoded -- individual fields inside it have been located and patched,
but not laid out from scratch.

**What the two biggest unknowns look like from here.** Both were probed
while the codec landed, and neither is decoded, but both now have shape:

- **The `a2` operand is a packed record, not an address.** Its low nibble is
  always 2 or 3 (75 and 62 times in the ONNX-path Mistral), the next nibble
  steps through 0..15, byte 1 is a small counter (0..4), byte 2 is always
  16-byte aligned (0x40, 0x10, 0x20, 0x00, 0x30), and byte 3 is zero except
  for a flag-like 0x82/0x86/0x8b/0x92 on a minority. The *first* `a2` of each
  op program rises with the op index in 69 of 71 programs, as does that
  program's Wbt offset, so it carries something ordered per op. Programs
  carry one to three of them (39, 27 and 2 of 77 in Mistral), never with all
  operands equal.
- **The weight table opens with a scale block.** The `llm_build` layer's
  3,963,652-byte table starts with exactly 1,024 float32 words of 0.125
  (4,096 bytes) followed by 768 zero words; only 2.9% of the whole table is
  zero and its byte histogram peaks hard at 0x88/0x78/0x87/0x77, which is
  what packed sub-byte values around mid-scale look like. The ONNX-path
  Mistral's 21 MB table has no such prologue and is 0.4% zeros.

**A refinement to "`40 02` is the Wbt offset."** That holds on the CNN and
ONNX paths -- all 77 of Mistral's offsets land inside its table. It does
*not* hold literally for `llm_build` layers: 22 of the decode subgraph's 147
offsets and 12 of the prefill subgraph's 169 point past the end of the table
they share, clustering around 2.9x its size. So on that path the operand
addresses a device space that holds the weight table *and* other buffers --
the KV caches are graph inputs there -- rather than a file offset. The
hand-patch experiments that established the field remain valid; what changes
is that a generator cannot compute it from the table alone.

Test: `test_decode_encode_round_trip_is_byte_exact` in
`tests/test_axera_mcode_structure.py` (fresh builds, no device).

### A third model family: text-to-speech, and what it did not change

The decoding corpus was two CNNs and two transformers. A vocoder is a third
shape entirely -- transposed-convolution upsampling, dilated 1-D residual
convolutions, LeakyReLU, a bounded output -- and it is what text-to-speech
actually runs on this class of hardware.

**A full TTS graph does not compile, and that is expected.** A Piper voice
(VITS, 63 MB, 2,755 nodes) carries `RandomNormalLike`, `NonZero`, `CumSum`,
`Range` and `Shape`: stochastic sampling and data-dependent shapes, none of
which an NPU compiler takes. Real deployments split the model, keeping the
text encoder, duration prediction and sampling on the CPU and sending only
the vocoder to the device. `_vocoder_model()` in the test file is that part,
built small and static.

**It compiles, and it taught us nothing new about the instruction set --
which is the result.** The vocoder builds cleanly (`ConvTranspose`,
`LeakyRelu`, `Tanh`, dilated `Conv`, `max_cycle` 55,183), its stream is
96.8% explained by the existing rule, and it introduces **zero** verbs, tags
or registers that the CNN and transformer builds had not already used. Its
op programs are the same skeleton: `40.02 | 50.01 | 50.01 | [30.03] | a8
40.03 | 50.03 | 50.01 | a3 | 50.01 | a9 | [a2] | [a1 20.02] | a8 30.02`.

So op type is not encoded by choosing different instructions. Convolution,
transposed convolution, matrix multiplication and elementwise activation all
issue the same program shape, and what distinguishes them lives in the
operand values -- which is exactly where a generator's remaining work is.

**A caution about differencing.** Changing a shape parameter re-lays-out the
whole stream: stride 8 to 4 moved 45 destinations, channel count 128 to 96
moved 238, and input length 32 to 64 moved 286 of roughly 1,400. Attribution
by differencing therefore needs experiments that hold every shape fixed --
changing only `LeakyRelu`'s alpha still moved 2,171 bytes across 1,649 writes
to tag 0x81 alone, so even that is not surgical. Single-register attribution
will need a sharper instrument than model diffing.

Test: `test_tts_vocoder_uses_no_new_instruction_forms` in
`tests/test_axera_mcode_structure.py` (fresh builds, no device).

### Real weights, real speech: the Piper vocoder on the NPU

The synthetic vocoder above answered a question about the instruction set,
but it had random weights, so its output was noise. The next question is
whether this NPU can produce *audible* speech, and that needs a trained
model. It can, and it does.

**Extract the decoder rather than rebuild it.** Hand-building a HiFi-GAN
shape and loading weights into it means guessing dilations, paddings and
upsample rates, and a single mismatch turns speech into noise with no error
message. Lifting the subgraph out of the voice with `onnx.utils.extract_
model` avoids all of it. The boundary is found by weight name -- whichever
tensor `dec.conv_pre.weight`'s convolution reads -- because the intermediate
tensor names are export artefacts and differ between voices.

What comes out of `en_US-lessac-low` is 68 nodes and 47 initializers, and
nothing else: `Conv` (20), `Add` (24), `LeakyRelu` (16), `ConvTranspose`
(3), `Div` (3), `Tanh`, `Unsqueeze`. Every one of those is supported. The
architecture:

| Stage | Shape |
| --- | --- |
| latent in | 192 channels |
| `conv_pre` | 192 to 256, kernel 7 |
| `ups.0` / `ups.1` / `ups.2` | kernel 16 stride 8, kernel 16 stride 8, kernel 8 stride 4 |
| channels after each stage | 128, 64, 32 |
| residual blocks | six, kernels 3/5/7, dilations 1 and 2, 2 and 6, 3 and 12 |
| `conv_post` | 32 to 1, kernel 7, then `Tanh` |

The three upsampling strides multiply to 256, which is exactly the measured
ratio of output samples to latent frames at a 16 kHz sample rate.

**The subgraph is the voice.** Frozen to a fixed 64-frame input and fed the
latent captured from a real utterance, it reproduces the untouched model's
own waveform at a correlation of 0.99999998. The remaining difference comes
only from zero-padding the latent out to the compiled length, which is what
the full graph's own length mask does anyway.

**Quantisation does not destroy it.** Calibrated on real latents drawn from
the model's own encoder, Pulsar2's end-to-end precision report gives a
cosine similarity of 0.9943 at the output for a 64-frame build and 0.9909
for a 256-frame one. On the device:

| Build | Latent frames | Audio | NPU vs CPU correlation |
| --- | --- | --- | --- |
| `out_dec` | 64 | 0.94 s | 0.9956 |
| `out_dec256` | 256 | 2.11 s | 0.9924 |

The 256-frame model compiles to `max_cycle` 12,792,136 and runs in 13.4 ms,
which is about 300 times faster than the 4.1 seconds of audio it emits.

**This settles an open question, and the contrast is the interesting part.**
Plain INT8 quantisation collapsed a 30-layer transformer to near-random
output, so whether a deep convolutional stack would survive was genuinely
open. It does. The plausible reason is structural: the vocoder is 20
convolutions deep with additive residual paths and a `Tanh` bounding every
output sample, where a transformer accumulates attention error across layers
with nothing bounding it. Depth alone does not predict quantisation
survival; what the depth is made of does.

**And the mcode confirms the generalisation.** A trained model in a new
domain, whose weights the decoder had never seen, introduces no new
instruction forms at all. Both builds use exactly the six known verbs
(`a1`, `a2`, `a3`, `a7`, `a8`, `a9`) and no tag outside `0x81`-`0x9f` plus
`0xa1` and the bit-6 odd-register forms, and the codec round-trips both
byte-exactly. Coverage is 96.47% at 64 frames and 97.29% at 256, in line
with the transformer builds rather than the CNNs.

One detail worth recording: the 64-frame build uses the odd-register tags
`0xc1` and `0xe1`, and the 256-frame build uses `0xe1` but not `0xc1`. The
odd-register forms are therefore optional per build, not a fixed part of
every stream -- the same model at a different input length simply does not
need one of them.

Tests: `test_piper_decoder_subgraph_reproduces_the_whole_voice` (CPU only),
`test_real_piper_decoder_uses_no_new_instruction_forms` (Docker) and
`test_real_piper_decoder_makes_speech_on_device` (Docker and device) in
`tests/test_axera_mcode_structure.py`. All three skip unless the voice is in
the local HuggingFace cache, so the suite stays offline.

### The first operand with a known meaning: output channels in `40.02`

Every finding above is about *form* -- which verbs, which tags, how a unit is
framed. None of it says what any operand *means*, and a generator that cannot
compute operand values cannot emit a working program. This is the first
operand field with a confirmed meaning.

**The instrument.** Differencing whole models failed before because changing
a shape re-lays-out the entire stream. The fix is to shrink the experiment
until only one thing can move: a model holding exactly *one* same-padded 1-D
convolution, so the program has one op and its operands are unambiguous, and
then to sweep one parameter at a time across separate builds.

**The result.** In a single-convolution program the leading `a1 40.02`
operand is exactly

    8 * (output channels) - 1

confirmed at 127, 255, 511 and 767 for 16, 32, 64 and 96 channels, and for
`ConvTranspose` as well as `Conv`. At 8 bits per INT8 channel that is an
inclusive bit extent over one output position.

**Two negatives make it an output-channel reading specifically.** Symmetric
convolutions cannot tell input from output, so the asymmetric cases decide
it: 32 to 64 gives 511 and 64 to 32 gives 255, each following the *output*.
And the value does not move when the input length (32 to 128), the kernel
size (1 to 9) or the dilation (1 to 8) changes -- which is what rules out
reading it as a buffer size or a weight-table offset, since both of those
move when the kernel or the length does.

**This corrects an earlier assumption.** `40 02` was being read as a Wbt
offset, and the puzzle was that on `llm_build` layers 22 of 147 and 12 of 169
of those "offsets" pointed past the end of the weight table. They were never
offsets. The small leading values recorded earlier -- `0xff` on resnet18d,
`0x1ff` on a llama layer, `0x3ff` on mistral -- are `8C-1` for 32, 64 and 128
channels.

**It is a convolution-engine register, not a general channel field.** A
lone `Relu`, `LeakyRelu` or `Sigmoid` never writes `40.02` at all, and
`Add`/`Mul` write it with something that is not `8C-1`. Those streams are
otherwise complete and round-trip byte-exactly, so the register is simply not
part of an elementwise program. This is the first concrete instance of the
claim above that op type lives in operand values rather than in different
instructions: the programs share their verbs and tags, and what separates a
convolution from an elementwise op is *which registers the program bothers to
set*.

That also explains an accident in this file's own tooling. `_tail_vector()`
located an mcode's tail by searching for the bytes `a1 00 40 02` -- so it was
silently anchored on the presence of a convolution, and failed outright on a
graph holding only a `Relu`. It now falls back to the end of the fixed
header, which leaves every convolution-bearing blob decoded exactly as
before.

**And the honest limit: it does not generalise to multi-op programs.** On the
real Piper decoder only 1 of 197 program-leading `40.02` operands has the
`8C-1` form; the rest are large values that behave like addresses or strides.
So `40.02` is a field that carries an output-channel extent in the
single-op case and something else once a program holds several fused ops.
The confirmed claim is the narrow one, and the wider question -- what selects
between those uses -- is open.

A note on what the fused programs mean for the method: eight chained
convolutions compile to five programs, not eight, and a two-convolution chain
compiles to one. Op programs are not one-to-one with graph nodes, so any
attribution that assumes they are will be wrong. Shrinking the model until
one op is one program is what makes the operand legible.

Tests: `test_single_conv_program_encodes_its_output_channel_count`,
`test_single_conv_channel_operand_ignores_length_kernel_and_dilation` and
`test_the_channel_operand_belongs_to_the_convolution_engine` in
`tests/test_axera_mcode_structure.py` (fresh builds, no device).

### Generating a weight table: the first piece we can produce ourselves

Counting bytes says where the work is. On the Piper decoder the weight table
is **26x** the size of the instruction stream -- 2.1 MB against 81 KB -- so
generating an `.axmodel` is mostly a weight-table problem, not an instruction
problem. This is the first part of a compiled model we can write ourselves
and have the device execute correctly.

**The instrument, again: make only one thing move.** Compile a convolution
whose weights are all zero, then compile the same convolution with a *single*
non-zero weight at a known `(o, i, k)`, and see which byte changes. Repeated
over nine positions this gives the addressing exactly rather than by fitting.

**The layout.** Weights live in the `npu_params` initializer. A slot index
walks one output channel's weights, and both that index and the output
channel are *tiled*:

    s      = (Cin/2)*k + i//2          slot within an output channel
    chunk  = s // 36,  within = s % 36
    group  = o // 16,  member = o % 16
    groups = ceil(Cout/16),  chunks = ceil((Cin/2)*K / 36)

    low  = 72*groups*chunks*member + 72*(group + groups*chunk) + within
    high = low + 36,   nibble = i % 2

Two input channels share a byte, which is what first looked like 4-bit
weights. The tiling is the part guessing would miss: a channel's weights are
*not* contiguous once either dimension overflows its tile. At 8 channels
neither does, and the formula collapses to `72*o + (Cin/2)*k + i//2`; at 32
channels both do, and weight `(0,8,2)` jumps from a predicted byte 36 to 144
while output channel 16 lands at byte 72 rather than after channel 15.

Across both shapes the formula predicts **35 of 35** measured spike offsets
exactly -- `(3,5,1)` at 222, `(7,7,2)` at 515, `(31,31,2)` at 4547 -- and it
reconstructs a full 32x32x3 weight tensor from a compiled table at a
per-channel correlation of 0.99998.

**They are not 4-bit.** Combining the planes as `high*16 + low` gives INT8
with zero point 128, and the arithmetic closes exactly:

| weight | low, high | byte | 128 + w/scale |
| --- | --- | --- | --- |
| +0.5 | 15, 15 | 255 | 128 + 127 |
| -0.5 | 1, 0 | 1 | 128 - 127 |
| +0.25 | 0, 12 | 192 | 128 + 63.5 |
| -0.125 | 0, 6 | 96 | 128 - 31.75 |
| 0.0 | 0, 8 | 128 | 128 |

So a weight byte is split across two nibble planes: the low nibble at the
offset above, the high nibble 36 bytes later.

**The scale is stored, and should be read rather than derived.** A float32
per output channel sits near the end of the table, exactly proportional to
that channel's peak magnitude. The effective slope is close to `127.5 /
max|w|` but not exactly -- it ranges over 127.27 to 127.76 across channels --
so it is the compiler's choice, not a formula to rederive. Recovering it by
least squares from a compiled reference reproduces pulsar2's own codes for
97.4% of the weights, and the residue is +/-1 rounding.

**Confirmed on the device: we can rewrite a model's weights.** Taking a
compiled convolution and replacing its weights by hand -- no vendor compiler
anywhere in the loop -- the NPU then computes the *new* convolution:

| shape | run | vs CPU reference | correlation |
| --- | --- | --- | --- |
| 8 ch | pulsar2's own build | its own weights | 0.99992 |
| 8 ch | our patched table | the **new** weights | 0.99987 |
| 8 ch | our patched table | the old weights | 0.319 |
| 8 ch | pulsar2's own build | the new weights | 0.320 |
| 32 ch | pulsar2's own build | its own weights | 0.99981 |
| 32 ch | our patched table | the **new** weights | 0.99972 |
| 32 ch | our patched table | the old weights | 0.052 |
| 32 ch | pulsar2's own build | the new weights | 0.052 |

The two cross-controls are the point. The patched model stops matching the
old weights and the untouched model does not match the new ones, so the
function really moved, and it moved to within a hair of what the vendor
compiler achieves on the same problem.

**Two limits, both real.** The new weights here are a permutation of the old
along the input-channel axis, chosen because it leaves every output channel's
peak magnitude untouched. That matters: the stored scale absorbs the
activation scales too, and those are calibrated from the weights, so changing
a channel's dynamic range invalidates the scale the table already holds.
Rewriting weights freely needs the scale rewritten with them.

And the addressing is confirmed at 8, 16 and 32 channels with `K = 3`. The
tiling constants (a 36-byte plane, 72-byte pairs, output groups of 16) are
read off those shapes rather than derived, so a shape that overflows some
*other* limit -- many chunks, a large kernel, 64 channels and up -- may well
expose another level of tiling. A stride search alone never found this
structure; the single-weight builds did, and the same method extends to any
shape worth confirming.

Tests: `test_conv_weights_are_int8_split_across_two_nibble_planes` (Docker)
and `test_conv_weights_can_be_rewritten_without_pulsar2` (Docker and device)
in `tests/test_axera_mcode_structure.py`.

### A second packing, and how far generation reaches on resnet18d

The weight layout above was found on 1-D convolutions. A real CNN is built
from 2-D ones, so the obvious question is whether the same formula reads
resnet18d. It does not, and the reason is worth recording: **the packing is a
property of the convolution's shape, not of the format.**

**2-D convolutions bit-slice differently.** The same INT8 codes are stored as
*four* 2-bit planes 36 bytes apart rather than two 4-bit ones, four input
channels share a byte instead of two, and the kernel is laid out in reverse:

    offset(o, i, kh, kw) = 144*o + 4*(K - 1 - (kw + K_w*kh)) + i//4
    bits  = 2 bits at 2*(i % 4), across planes 3, 2, 1, 0 most significant first

Confirmed at `cin = cout = 8, 3x3`, where it reconstructs a compiled table at
0.99998 and predicts `(7,7,2,2)` at byte 1009 exactly. **Confirmed on the
device too**: rewriting a 2-D convolution's weights by hand gives 0.99982
against the new weights' CPU reference versus 0.99988 for pulsar2's own
build, with both cross-controls at zero.

**At 32 channels the 2-D case adds two more levels.** Input channels group 16
at a time (`+144` per group), and the output channel is addressed through a
*bit-interleave* rather than a stride:

    base(o) = 576*(o % 8) + 72*((o // 8) % 2) + 4608*(o // 16)

which is exact for every measured channel -- `o = 8` lands at 72, `o = 16` at
4608, `o = 31` at 8712 -- and predicts the compiled table's total size. The
plane pairs also separate by `+288` rather than staying 36 apart, and that
placement rule is not yet pinned down.

**This has since been carried through to resnet18d** -- see the next section.
The addressing constants for large channel counts turned out to be two
things: a shape-scaled unit and a cap.

The honest budget for resnet18d today:

| part | size | status |
| --- | --- | --- |
| weight table | 11,855,108 B (242x the mcode) | 100% of conv weights addressable |
| mcode stream | 48,320 B | 98.92% explained, round-trips byte-exactly |
| framing bytes | 28,552 B (59.1% of stream) | emitted from the grammar |
| value bytes | 19,235 B (39.8%) | 3.81% have a confirmed meaning |
| header + tail | 760 B | rules known |

So on resnet18d specifically: we can decode and reproduce its instruction
stream exactly, we can locate and read *all* of its convolution weights, and
we understand under 4% of its operand values.

Tests: `test_conv2d_weights_are_int8_split_across_four_bit_planes` (Docker)
and `test_conv2d_weights_can_be_rewritten_without_pulsar2` (Docker and
device).

### Reading a real network's weights: all of resnet18d

The sweep that finishes this is cheaper than it looks, because **the address
is linear in the bits of the channel indices**. For the 32-channel case
`base(1) + base(2) + base(4) + base(8) + base(16)` sums exactly to
`base(31)`, so one build per *bit* suffices where one per channel would be
hopeless. Fifteen builds characterise a shape.

**Two constants, both shape-dependent.** Probing 8, 32 and 64 channels gives
the same table twice over:

    A = 18 * min(Cin, 128)      output-channel unit
    P =  9 * min(Cin, 128)      separation between the two plane pairs

and the full 2-D address is

    addr(o,i,kh,kw) = A*(o%8) + 72*((o>>3)&1) + 8*A*(o>>4)
                    + 4*(K - 1 - (K_w*kh + kw))
                    + (i%16)//4 + 144*(i>>4)
    planes at 0, 36, P, P+36;  bits 2*(i%4), planes 3,2,1,0 most significant first

Output-channel bit 3 always costs 72 bytes while bits 0-2 cost `A` and bits 4
and up cost `8A`: an interleave, not a stride. With no fitting at all this
reconstructs a compiled 64x64x3x3 table at 0.99997.

**The cap is the part that unlocks a real network.** `min(Cin, 128)` says a
convolution wider than 128 input channels is *split into slices*. Until that
was applied, resnet18d's 256- and 512-channel layers read as noise (0.36);
with it, `(256,256,3,3)` and `(512,256,3,3)` both locate at 0.9999.

**And one measurement trap worth recording.** Scoring a match by pooling
output channels reads about 0.9 even when the addressing is perfectly right,
because every output channel carries its own quantisation scale. Scored *per
channel* the same layers read 0.9999. A 0.9 that should be 0.9999 looks like
a nearly-correct layout and invites fiddling with the formula; it was
actually a wrong metric.

**The result on resnet18d:** 18 of 22 convolutions located and read from an
11.9 MB table with nothing but each layer's shape and a search for its base
offset -- **98.5% of its convolution weights**. The blocks are contiguous and
in graph order, with regular deltas (37,376 bytes between 64-channel layers,
148,480 between 128-channel ones).

**The last four needed two more packings, and they are not variations -- they
are different formats.** Neither the 1x1 convolutions nor the 3-channel stem
bit-slices at all; both store plain INT8 bytes.

A **1x1 convolution** stores one byte per weight, chunking the input channels
36 at a time with the next chunk 144 bytes on:

    addr(o,i) = U*((o>>1)&15) + 144*(i//36) + 36*(o&1) + 72*((o>>5)&1) + (i%36)
    U = 144 * ceil(Cin/36)

A **narrow input** -- the 3-channel stem -- also stores plain bytes, with the
kernel *row* fastest and the output channel on a flat 36-byte stride:

    addr(o,i,kh,kw) = 36*o + 12*kw + 3*i + kh

Both were found the same way, but with *dense* probe models: a nearly-empty
weight tensor gets compressed, so flipping one weight's sign inside an
otherwise dense tensor is what keeps the diff to a single byte. That is worth
remembering -- an earlier 256-channel probe produced a 37 KB table where the
weights alone should have taken 590 KB, and the sparsity was the reason.

**resnet18d is now fully covered: 22 of 22 convolutions, 100% of its
convolution weights.** The stem sits at base 0 and reads at correlation
1.0000; the three 1x1 layers read at 1.0000; the eighteen bit-sliced layers at
0.9998 or better.

So the weight table does not have *a* layout. It has at least three, and the
convolution's shape picks one -- narrow input, 1x1 kernel, or anything wider.
A generator has to dispatch on shape, which is exactly what
`_conv_channel0_addresses()` does.

Test: `test_resnet18d_conv_weights_are_addressable` (Docker, no device),
which locates every one of the 22 layers.

### A second operand, and a method that rejects its own guesses

With the weight table readable, the remaining unknown is operand meaning:
156 register addresses across the models measured so far, of which `40.02`
was the only one understood. This is a systematic attempt at the rest, and
its most useful output is the shape of the method.

**Sweep one parameter at a time, align, then split into bitfields.** Twenty-one
single-convolution builds vary `Cin`, `Cout`, length, kernel and dilation
independently. Aligning their records by *structural signature* (not
position, which shifts) leaves 209 slots present in every build, of which 25
carry an operand that moves at all. Fitting whole operands mostly fails,
because a 32-bit operand is several packed fields; splitting each into the
bits that actually move and fitting those separately produces five candidate
`(register, bitfield)` semantics.

**Then hold out configurations and see which candidates survive. Two of three
did not.** A candidate `(k-1)/2` kernel field fit the sweep exactly and then
read 2 where it should have read 3 at `k = 7`; a candidate `Cin/4 - 1` field
fit and then read 0 on every held-out build. Both were artefacts of fitting
four points with two degrees of freedom. Reporting them would have been easy
and wrong -- the held-out builds are what made the difference.

**What survives is a spatial extent.** Bits 4 to 6 of the first `a1 b0.03`
operand are

    length / 16 - 1

the input's spatial size in 16-element tiles, inclusive -- the same
convention `40.02` uses for channels. It is correct in **all 27**
configurations measured, six of them held out from the fit, and the negatives
are what make it a *spatial* field specifically: changing `Cin`, `Cout`, the
kernel size or the dilation leaves it untouched, while several other operands
that also move with the length fail one or more of those tests.

A near-miss worth recording: bits 20 to 22 of `81.34` carry the same value in
21 of 27 configurations and disagree in the other 6, all of which changed
`Cin`. So it is not a spatial field; it is something that usually coincides
with one. That is exactly the shape of error the held-out set is there to
catch.

So the operand budget moves from one confirmed field to two, and there is now
a repeatable procedure for the rest: sweep, align by signature, split into
moving bitfields, fit, and discard whatever a held-out configuration refutes.

**Scaling the procedure up says the remaining operands are mostly not shape
functions at all.** A larger sweep -- 28 training builds varying `Cin`,
`Cout`, length, kernel, dilation *and* stride, against six held-out
combinations -- finds 28 operand slots that move. Fitting each moving
bitfield against nineteen derived quantities a convolution engine plausibly
holds (raw parameters, padding, dilated kernel extent, output length, and
each of those in 4- or 16-element tiles, plus `Cin*k`, `Cin*L`, `Cout*Lout`),
allowing any one- or two-term exact affine combination, yields exactly **one**
field that also survives the held-out set: the same `length/16 - 1`.

That is a strong negative, and it points somewhere specific. Of the 28 moving
slots, **25 carry values above 4096** and only 3 are small enough to be
counts. Large, shape-sensitive, not affine in any shape quantity -- that is
what an *allocator output* looks like. The compiler is choosing buffer
addresses, and an address depends on the order and size of every prior
allocation, not on the current layer's shape alone.

One more measurement supports the reading: **no slot moves with dilation
unless it also moves with a size parameter.** Dilation changes no tensor
size, so a register that tracked dilation alone would have to be a genuine
configuration field; there is not one. Every dilation-sensitive operand is
sensitive to sizes too, consistent with dilation changing what gets buffered
rather than being programmed directly.

For a generator this reframes the remaining work. It is not "decode 154 more
constants". Most of those operands cannot be computed from a layer's shape at
all -- they require reproducing the compiler's allocator, which is a
different and much larger problem than the weight table turned out to be.

Test: `test_spatial_extent_lives_in_three_bits_of_b0_03` (Docker, no device)
covers the confirmed field and its four invariances. The wider
characterisation above -- the 28 moving slots, the 25 address-like values, the
absence of a dilation-only register -- comes from the 34-build sweep in this
session's notes rather than from a regression test; it is a measurement of
this toolchain version, not an invariant worth 34 Docker builds per CI run.

### The same register, two spatial rules

Running the same procedure over *2-D* convolutions -- 23 training builds
sweeping `Cin`, `Cout`, height, width and kernel, against four held-out
combinations -- first produced two apparent survivors, and both were wrong.
Catching them sharpened the method.

**A held-out set only falsifies what it actually varies.** The two candidates
were quadratic in the kernel size, fitted to the three kernel values in the
sweep with three free parameters -- exact by construction. The held-out
configurations reused those same three kernel values, so they could not
refute a kernel-only model. Extrapolating one to `k = 7` predicts **-9** for
a two-bit field; the measured value is 0. A held-out set has to contain
unseen *values* of whatever a candidate depends on, not merely unseen
combinations.

**What the register actually holds in 2-D.** The same `a1 b0.03` bits 4 to 6
carry

    floor((W + 2*pad - 1) / 32)

the index of the last 32-wide tile of the *padded* width. Confirmed in
**32 of 32** configurations, including four widths never swept. It follows
the innermost dimension only: sweeping the height from 16 to 80 does not move
it at all.

**And it is not the 1-D rule.** For a 1-D convolution the same bits hold
`floor((L - 1) / 16)` -- a 16-wide tile over the *unpadded* length. Two
differences, both real:

| | tile | padding |
| --- | --- | --- |
| 1-D | 16 | unpadded (`k` does not move it) |
| 2-D | 32 | padded (`k` moves it) |

The padding difference is visible only at an exact multiple of the tile,
where a 1x1 and a 3x3 kernel disagree by one -- which is exactly the single
configuration that a `floor(W/32)` reading gets wrong (31 of 32).

So a register's meaning is not fixed by its address: this one is spatial in
both layouts, but *what* it measures and *how* it tiles depend on the
convolution's dimensionality. That is the same lesson the weight table
taught -- the shape picks the encoding -- now showing up in the instruction
stream.

Tests: `test_spatial_extent_lives_in_three_bits_of_b0_03` and
`test_the_spatial_field_follows_the_innermost_dimension_in_2d` (Docker, no
device).

### What the card actually does: 10.13 TOPS against a 10.8 TOPS rating

Everything above is about *what* the hardware runs. This is about how fast.
The graphs are deliberately the most compute-dense thing that will compile --
stacks of identical same-padded convolutions, nothing else attached -- and
accuracy is irrelevant to the measurement. Throughput is
`2 * MACs / latency`, with the MAC count taken from the compiler's own
`build_context.json` rather than recomputed.

**Peak measured: 10.13 TOPS INT8**, which is **93.8%** of Axera's 10.8
TOPS "from NPU alone" figure and 56% of the 18 TOPS INT8 headline. The curve
flattens there:

| graph | GMAC | min ms | TOPS |
| --- | --- | --- | --- |
| 1024ch 16x16 3x3, 8 layers | 19.3 | 4.13 | 9.37 |
| 1024ch 16x16 3x3, 16 layers | 38.7 | 7.82 | 9.89 |
| 1024ch 16x16 3x3, **24 layers** | 58.0 | 11.45 | **10.13** |
| 1024ch 32x32 3x3, 8 layers | 77.3 | 15.35 | 10.07 |
| 512ch 48x48 3x3, 12 layers | 65.2 | 13.04 | 10.01 |

Depth is what buys the last 8%: identical arithmetic at 8, 16 and 24 layers
gives 9.37, 9.89 and 10.13, so a fixed per-inference overhead of roughly
0.3 ms is being amortised.

**The cores scale almost linearly.** The same graph under each `npu_mode`:

| mode | min ms | TOPS | relative |
| --- | --- | --- | --- |
| NPU1 | 11.49 | 3.37 | 1.00 |
| NPU2 | 6.03 | 6.41 | 1.90 |
| NPU3 | 4.09 | 9.45 | 2.80 |

`NPU4` and `NPU5` exist in the toolchain's `NPUMode` enum but are rejected on
this target, so three cores is the ceiling here.

**Shape matters more than size.** All the 3x3 rows above are the *same* 19.3
GMAC of arithmetic, and they differ by a factor of three depending on how it
is shaped:

| shape | TOPS | why |
| --- | --- | --- |
| 1024ch, 16x16 | 9.37 | best of the set |
| 256ch, 64x64 | 8.47 | |
| 128ch, 128x128 | 7.75 | |
| 2048ch, **8x8** | 2.99 | spatial extent too small to fill the array |
| 512ch 32x32, 1x1 kernel | 6.19 | nine times fewer MACs per weight byte |

The 8x8 collapse is the sharpest result: the same arithmetic runs 3.1x slower
purely because the spatial dimension no longer covers the tiles the engine
works in -- which is the same 32-wide tiling the `b0.03` operand counts.

**The 43.2 TOPS INT4 figure is not reachable through *this* path.** The
build config's `weight_data_type` for convolution accepts only `S8` or
`FP32`, and `pulsar2 build --help` offers no 4-bit flag either. There is a
4-bit path, but it is in the other pipeline -- see the next section.

**Against the compiler's own cycle model.** Building with `profile=True`
gives a modelled critical path, and a per-op profile in which convolutions
are **98.9%** of the modelled cycles -- so the benchmark really is measuring
arithmetic and not plumbing. Dividing that cycle count by the measured time
gives an implied clock:

| graph | max_cycle | min ms | implied MHz | MAC/cycle |
| --- | --- | --- | --- | --- |
| 1024ch 16x16 | 3,905,355 | 4.12 | 949 | 4,949 |
| 512ch 32x32 | 3,704,778 | 4.30 | 862 | 5,217 |
| 256ch 64x64 | 3,831,460 | 4.49 | 854 | 5,044 |

**MAC-per-cycle is stable at about 5,000** across the three cores (roughly
1,650 each), which is the more meaningful figure -- it is a property of the
engine, not of the graph. The implied clock is *not* stable: 854 to 949 MHz.
Since the clock cannot actually vary by 11% between two runs a minute apart,
what that spread measures is the cycle model being optimistic by up to ~10%
on some shapes -- stalls it does not account for. The best-matching graph is
also the fastest one, which is consistent with that reading rather than with
a variable clock.

So the honest summary is: the engine sustains ~5,000 MAC/cycle at somewhere
around 0.95 GHz, 9-10 TOPS is what that product comes to, and the compiler's
cycle estimate is a good predictor to within about 10%.

Test: `test_int8_throughput_reaches_a_useful_fraction_of_the_rating` (Docker
and device). Its floor of 5 TOPS sits between a healthy NPU3 run and the
~3 TOPS a single-core or fallback build produces, so it doubles as a health
check for a card that has quietly dropped to one core.

### Where the INT4 path actually is

Searching for 4-bit support the way the LLM pipeline does it finds it
immediately, and in only one of the two pipelines:

| pipeline | weight types offered |
| --- | --- |
| `pulsar2 build` (CNNs, everything above) | `S8`, `FP32` |
| `pulsar2 llm_build` (transformers) | `fp16`, `bf16`, `fp32`, `s8`, **`s4`**, `fp8_e5m2`, `fp8_e4m3` |

So the toolchain does have INT4 weights -- and FP8 in two flavours -- but only
for the LLM path. `pulsar2 build` has no 4-bit option at all, neither in its
config schema nor on its command line.

**It builds, and it is much smaller.** SmolLM2-135M compiled three ways, same
prefill length and KV cache, measured on the AX650N:

| weights | total | per layer | decode/layer | tokens/s |
| --- | --- | --- | --- | --- |
| `s4` | 98.2 MB | 2.30 MB | 0.441 ms | 75.6 |
| `s8` | 152.7 MB | 4.12 MB | 0.485 ms | 68.7 |
| `fp16` | 253.4 MB | 7.47 MB | 0.639 ms | 52.2 |

**But at this size the win is memory, not speed.** Going from `s8` to `s4`
cuts the model 1.56x and buys only 1.10x on decode. The reason is visible in
the three points: fitting `time = overhead + bytes / bandwidth` gives a fixed
**per-layer overhead of about 0.40 ms** against roughly 0.09 ms of weight
streaming at `s8`. Overhead dominates by four to one, so halving the weights
barely moves the total. The same fit predicts the `fp16` measurement to
within 10%, which is about as much as a two-parameter model of this deserves.

That also sets a ceiling worth knowing: 0.40 ms x 30 layers is 12 ms per
token of pure overhead, so this model cannot exceed roughly 80 tokens/s on
this card no matter how the weights are quantised. INT4 pays off on a model
whose per-layer weights are large enough for streaming to dominate that fixed
cost -- which a 135M model's 3.5 MB per layer is not.

**So the next test is a model built to make it pay.** A synthetic
Llama-shaped checkpoint at 4096 hidden and 11008 intermediate -- **177 M
parameters per layer**, fifty times SmolLM2's -- puts weight streaming firmly
in charge:

| weights | per layer | decode/layer | implied GB/s |
| --- | --- | --- | --- |
| `s4` | 100.5 MB | 5.91 ms | 17.0 |
| `s8` | 195.6 MB | 9.20 ms | 21.3 |

**INT4 now buys 1.56x**, against 1.10x on the small model, and the weight
saving reaches 1.95x once layers are large enough that non-weight structure
stops diluting it. The benefit scales with per-layer weight size exactly as
the overhead argument predicts, and 1.56x of a theoretical 2x says roughly a
third of a decode step is still something other than streaming weights.

Effective weight-streaming bandwidth lands around **21 GB/s** at `s8`.

**One caveat on the earlier model, stated plainly:** fitting
`time = overhead + bytes / bandwidth` to the large model gives a fixed
overhead of 2.6 ms per layer, not the 0.40 ms the small model gave. A
genuinely fixed cost cannot do that, so that two-parameter fit describes each
model at its own scale and should not be extrapolated between them. What does
survive across both is the direction and its size: halving the weights buys
almost nothing when layers are small and about 1.56x when they are large.

For deployment arithmetic: at 9.20 ms per layer, a 32-layer model of this
width would take 294 ms per token at `s8` and 189 ms at `s4` -- roughly 3.4
against 5.3 tokens/s.

**And the 43.2 TOPS INT4 rating stays unverified.** It is a compute-throughput
claim, and the decode path cannot demonstrate it: a decode step is about
4.5 MMAC per layer, some 0.02 TOPS, so it is latency-bound by three orders of
magnitude. Showing it would need the prefill subgraph in isolation, and
`axcl_run_model` will not select it -- its `--group` flag indexes shape
groups, of which these layer files have none.

Test: `test_llm_build_offers_an_int4_weight_path_the_cnn_path_lacks`
(Docker, no device).

### Prefill, timed at last -- and INT4 does nothing for it

The section below records four routes that all closed. The fifth works: go
under `axcl_run_model` to the engine API it is built on.

**The CLI was wrong about the shape groups.** `axcl_run_model -g 1` reports
"Selected shape group index {1 vs. 0} is out of range", which reads as "this
model has no groups". Asking the engine directly --
`axclrtEngineGetShapeGroupsCount()` -- says an `llm_build` layer has **two**,
and their sizes say exactly what they are:

| | group 0 | group 1 |
| --- | --- | --- |
| `input` | 8,192 B (1 x 4096 x bf16) | 1,048,576 B (128 x 4096 x bf16) |
| `mask` | 512 B | 32,768 B |

Group 0 is decode, group 1 is prefill. `axclrtEngineExecute()` takes the
group index, so a twenty-line C program can run either. `scripts/axera/tools/`
carries it. Its group-0 timing reproduces `axcl_run_model`'s to within a few
percent (9.117 ms against 9.14 ms), which is the check that the buffers and
the timing loop are honest.

**The result, on both models:**

| model | group | `s8` | `s4` | INT4 gain |
| --- | --- | --- | --- | --- |
| 4096-hidden | decode | 9.117 ms | 5.751 ms | **1.59x** |
| 4096-hidden | prefill, S=128 | 23.493 ms | 22.359 ms | **1.05x** |
| SmolLM2-135M | decode | 0.476 ms | 0.447 ms | 1.06x |
| SmolLM2-135M | prefill, S=512 | 19.952 ms | 19.553 ms | 1.02x |

**INT4 buys 1.59x on decode and essentially nothing on prefill.** That is the
whole answer to the 43.2 TOPS question. Four-bit weights cut weight *traffic*,
which is what decode is made of; they do not make the arithmetic faster, which
is what prefill is made of.

**And prefill's arithmetic is slow in absolute terms.** The 4096-hidden layer
is 22.8 GMAC of prefill, so 23.5 ms is **1.94 TOPS** -- against 5.7-7.5 for an
INT8 matmul stack and 10.13 for convolution. The reason is in the build
options: `llm_build --hidden_state_type` offers `fp16`, `bf16` and `fp32`, and
nothing narrower. **The LLM pipeline is weight-only quantised**: `s4`/`s8`
weights against 16-bit activations. Its arithmetic therefore never touches the
INT8 or INT4 datapath the TOPS ratings describe, whatever the weights are
stored as.

So the 43.2 TOPS INT4 rating is not reachable through either pipeline, and now
for a measured reason rather than a missing measurement: `pulsar2 build` has no
4-bit weights at all, and `llm_build` has 4-bit weights but 16-bit activations.

### Why the earlier routes closed (recorded so nobody repeats them)

Decode measures memory. Prefill is the compute-bound half of an LLM, and it
is what a 43.2 TOPS INT4 rating would have to be claiming. It cannot be timed
on this stack, and the reasons are worth recording so nobody repeats the
attempt.

**The two halves are both in the file.** An `llm_build` layer `.axmodel`
holds *two independent* `neu mode` nodes: `subgraph_npu_0` takes
`K_cache, V_cache, indices, input, mask` (decode) and `subgraph_npu_1` takes
the same names suffixed `_1` (prefill). Neither consumes the other's output.

**`axcl_run_model` always runs the first one.** Its `--group` flag indexes
*shape* groups, of which these files have none -- asking for group 0 or 1
both return "Selected shape group index {n vs. 0} is out of range". Feeding
only the `_1` inputs fails; feeding all ten runs decode and reports decode's
latency. There is no other runner: `/usr/bin/axcl/` ships `axcl_run_model`,
`axcl_demo` and hardware samples, nothing LLM-specific.

**Editing the wrapper does not redirect it.** Deleting the decode node makes
the model fail to load, while an unmodified load-and-re-save runs fine
(9.268 ms against 9.182 ms), so the surgery is the cause, not the round trip.
Renaming so prefill becomes `subgraph_npu_0` gets past loading and then fails
at "Feed stimulus failed" -- the runtime takes its input specification from
the compiled blob, not from the ONNX names, so the wrapper cannot choose
which subgraph runs.

**And a prefill-only build is not on offer.** `llm_build --kv_cache_len 0`
fails in the frontend with degenerate RoPE parameters (`shape (0, 64)`).

**What stands in for it: the arithmetic prefill is made of.** A prefill step
is dominated by large matrix multiplications, and those compile through the
ordinary path:

| matmul stack | GMAC | min ms | TOPS |
| --- | --- | --- | --- |
| S=128, 4096x11008, 4 layers | 46.2 | 16.21 | 5.70 |
| S=512, 2048x5632, 6 layers | 70.9 | 19.01 | 7.46 |

So transformer-shaped INT8 arithmetic sustains **5.7 to 7.5 TOPS**, against
10.13 for convolution. Longer sequences do better, which is the same tiling
story as the CNN measurements: 128 rows do not fill the engine as well as 512.

That puts a bound on the INT4 claim rather than testing it. Prefill
arithmetic on this card runs at roughly 6-7 TOPS in INT8; for INT4 to reach
43.2 it would have to be six times faster than INT8 matmul measured here,
where the decode measurements show INT4 buying 1.56x. The rating is not
verifiable on this toolchain, and nothing measured here suggests it is
reachable.

### Is there a full INT4 path? No, and here is every place it is not

A 43.2 TOPS INT4 rating describes 4-bit *arithmetic* -- 4-bit weights against
4-bit activations. Both pipelines were checked for it, config schema and
behaviour, and it is expressible in neither.

**Activations cannot be 4-bit anywhere.** `pulsar2 build`'s `data_type`
offers `U8, S8, U16, S16, FP32`; asking for the one 4-bit name in the enum
gets an explicit rejection, `sepc_type NVFP4 not in STRING_NUMBER_MAP`.
`llm_build --hidden_state_type` offers `fp16, bf16, fp32`. Nothing narrower
exists in either.

**The CNN path's one 4-bit weight type is silently ignored.** The `DataType`
enum contains `NVFP4` even though `weight_data_type` is documented as
`S8, FP32`, and setting it *builds successfully* -- which looks promising and
is not. Against the `S8` build of the same graph it produces:

| | S8 | NVFP4 |
| --- | --- | --- |
| weight table | 76.02 MB (1.007 B/weight) | 76.02 MB (1.007 B/weight) |
| `max_cycle` | 3,905,355 | 3,905,355 |
| device latency | 4.142 ms | 4.126 ms |
| decoded weight codes | 167 distinct, 26..234 | **the same 167, 26..234** |

The compiled weights are identical code for code -- decoded through the
layout from the "Reading a real network's weights" section, which is what
makes the comparison possible at all. `NVFP4` parses, is recorded in the
build context, and changes nothing about the model.

**`llm_build`'s `s4` is genuinely 4-bit storage, but only storage.** It
measures 0.567 bytes per parameter against `s8`'s 1.10, so the weights really
are four bits. The activations beside them are 16-bit, which is why it buys
1.59x on decode (weight traffic) and 1.05x on prefill (arithmetic).

So the complete picture is **W8A8 in the CNN pipeline, W4A16 or W8A16 in the
LLM pipeline, and W4A4 nowhere.** The 43.2 TOPS figure cannot be reached
through the vendor toolchain, not because the measurements fall short but
because nothing in either pipeline emits that kind of program.

**A note on whether the silicon could.** The weight tables this project
decodes are stored as *bit planes* -- two 4-bit planes for a 1-D convolution,
four 2-bit planes for a 2-D one, eight bits either way. A datapath that reads
weights plane by plane is the kind where fewer weight bits would mean
proportionally fewer passes, which is what a 2.4x INT8-to-INT4 rating would
describe. That is consistent with the hardware supporting it and the
toolchain not exposing it, but it is an inference from a storage format, not
a measurement, and nothing here tests it.

Test: `test_nvfp4_weights_are_accepted_and_then_ignored` (Docker, no device).
It asserts the two builds' weights are identical, so if a future toolchain
implements NVFP4 the test fails and says this section needs revisiting.

### Survey: every 4-bit route the toolchain has, and what each is worth

Re-running the question across the whole toolchain rather than one pipeline at
a time. Six routes, one of which works:

| route | 4-bit? | measured effect |
| --- | --- | --- |
| `pulsar2 build` weights, `NVFP4` | parses, silently ignored | none -- weights identical code for code |
| `pulsar2 build` weights, `U4`/`S4` | not in the enum | rejected at parse |
| `pulsar2 build` activations | `NVFP4 not in STRING_NUMBER_MAP` | rejected |
| `llm_build -w s4` | **yes, real** -- 0.567 B/param | **1.59x decode, 1.05x prefill** |
| `llm_build -w fp8_e4m3` | option exists | build fails, `OpBuildException: op: AxFu` |
| `llm_build --hidden_state_type` | `fp16, bf16, fp32` only | no narrow activations |

**Peak INT4 arithmetic measured: 2.04 TOPS.** That is the `s4` prefill number
-- 22.8 GMAC in 22.36 ms -- against a 43.2 TOPS INT4 rating, so under 5% of
it. The cause is not the 4-bit weights, which are real; it is that nothing
narrower than 16 bits is available for the activations they multiply.

**A tool that looks like the missing path, and is not.**
`/opt/pulsar2/convert_to_4w8f_cli` is a real CLI whose name reads as 4-bit
weights with 8-bit float activations -- exactly the combination that would be
worth measuring. Run on `Conv`, `MatMul` and `Gemm` models it changes nothing
but the IR version: same op counts, same model size to within four bytes.
Whatever it keys on, an ordinary float graph is not it, and it is not a route
to INT4 from here.

**A methodological note, because it nearly produced a false finding.**
Grepping the compiler's per-chip backends for `S4` and `U4` reports them in
~30 of the AX650 backend's 142 files, which reads as "the backend supports
4-bit even though the frontend does not". It does not: those files are
Pyarmor-encrypted blobs, and a two-character string appears in almost any
large binary by chance. The control settles it -- the nonsense pattern `Q7x`
matches exactly as many files as `INT4` and `4w8f` do. Nothing about backend
capability can be read this way, and the earlier draft of this section that
did so was wrong.

So the survey's answer is the same as the narrower one, now checked
everywhere: **4-bit weights yes, 4-bit arithmetic no**, and the 43.2 TOPS
rating is unreachable through any route this toolchain exposes.

### What a weight generator can and cannot do yet

The weight table is fully addressable -- every convolution in resnet18d, three
packings, 100% of its weights. Writing it is more constrained than reading
it, and this pins down exactly how.

**That float array is not the weight scale -- it is the requantisation
multiplier.** One float32 per output channel sits immediately after the
weight block, and it equals

    M[o] = x_scale * (peak[o] / 127.5) / r_scale

to zero relative error: the input activation scale, times the per-channel
weight scale, divided by the output activation scale. That is why its
"constant of proportionality" appeared to absorb the activation scales
earlier -- it is made of them. An earlier draft of this section called it a
weight scale, which was wrong.

**But rewriting weights and scales is still not enough.** Three device
experiments, each stricter than the last:

| what was rewritten | new weights | result |
| --- | --- | --- |
| weights only | a permutation along the input axis | **0.99982** |
| weights only | random, per-channel peak preserved | 0.19, amplitude 3.1x |
| weights and scales | random, peaks 0.4x-2.5x | 0.25, amplitude 2.0x |

The amplitude ratios are the diagnosis: the device output is several times
larger than the reference, which is saturation. Every convolution's *output*
activation scale was calibrated from the original weights and is stored
elsewhere, so any edit that widens the output distribution clips against it.

**So the real constraint is not "preserve the peak" -- it is "preserve the
output distribution".** A permutation does that exactly, which is why it
works and why random weights with the same peak do not: same maximum,
different variance, different output range.

**Where the activation scales live.** Of the eleven activation scales the
compiler's own profile lists for an eight-layer convolution stack, exactly one
appears verbatim as a float32 in the mcode. The rest do not appear in any byte
order -- because they are not float32 at all. See "The activation scales,
decoded" below: they are bfloat16.

Until then the honest capability statement is: **a compiled model's
convolution weights can be replaced by any set that preserves each output
tensor's dynamic range, and verified on hardware.** That is narrower than
"generate a weight table", and wider than it was.

Test: `test_per_channel_weight_scales_sit_just_past_the_weight_block`
(Docker, no device).

### The weight quantiser, exactly

Identifying the multiplier pinned the weight scale inside it, and that turned
out to be the whole quantiser. Pulsar2's convolution weights are

    scale[o] = float32(peak[o]) / float32(127.5)
    code     = clip(round_half_to_even(float32(w) / scale) + 128, 0, 255)

and this reproduces **100.000%** of the codes in a compiled model -- verified
on three independent builds at 8 and 64 channels, with no reference model
involved.

Every detail earns its place, which is why the earlier fitted-slope approach
plateaued at 97.4%:

| variation | codes reproduced |
| --- | --- |
| `float32`, 127.5, round-half-to-even | **100.000%** |
| `float64` instead of `float32` | 99.48% |
| round-half-away instead of half-to-even | 99.83% |
| 127 instead of 127.5 | 86.11% |
| 128 instead of 127.5 | 84.03% |

The 127.5 is what the earlier least-squares recovery was circling: it fitted
slopes of 127.27 to 127.76 per channel and called the spread a property of
the compiler. It was not -- it was noise around an exact constant, and the
residue was float64 arithmetic where the compiler uses float32.

**What this changes for a generator.** Weight codes no longer have to be
recovered from a compiled model and re-quantised against a fitted slope; they
can be computed from the weights alone. Together with the three layouts, a
convolution's weight block is now fully synthesisable. What still requires a
reference is the *multiplier* array, since `x_scale` and `r_scale` come from
calibration -- and the output scale a layer's consumer expects is the same
thing that makes an arbitrary weight rewrite saturate.

**Also located: the output zero point** is an `int32` in the mcode -- byte 176
of the stream in the model measured, holding 121 where the compiler's profile
reports `r_zeropoint = 121`.

Test: `test_conv_weight_quantiser_is_reproduced_exactly` (Docker, no device).

### Where the 1-D layout stops: 32 channels

The TTS vocoder is the easier of the two remaining test beds -- it compiles
through the same `pulsar2 build` path as everything else here, unlike the LLM
pipeline. Pointing the weight locator at it fails almost completely: **1 of 23
layers**, and reading its convolutions as 2-D with a unit dimension (either
`1xk` or `kx1`) locates none at all.

**The cause is a channel-count boundary, not the model.** A controlled sweep
of single 1-D convolutions:

| shape | located |
| --- | --- |
| 32 channels, K=3 | 1.0000 |
| 32 channels, K=7 | 1.0000 |
| 64 channels, K=3 | 0.42 |
| 64 channels, K=7 | 0.19 |
| 128 channels, K=3 | 0.27 |

So the 1-D layout is right up to 32 input channels and wrong from 64. That
explains the vocoder exactly: its convolutions run 32 to 256 channels, and the
single layer that *did* locate is the one 32-channel `3x3`... which is to say
the locator was never failing on the vocoder, it was failing on every shape
past a limit nothing had tested.

Two checks rule out the obvious alternatives. A **three-layer** 1-D model at
32 channels locates all three layers at 1.0000, on contiguous bases -- so
depth is not the problem. And kernel size is not either: 32 channels works at
K=3 and K=7 alike.

**What the 64-channel addressing looks like.** Dense single-weight probes give
the output-channel map directly:

    base(o) = 432*(o % 16) + 72*((o >> 4) & 1) + 6912*(o >> 5)

with the input-channel and kernel rules unchanged (`i//2` bytes with the
nibble chosen by `i % 2`, kernel stride `Cin/2`). The same
bits-0-to-3/bit-4/higher-bits structure as everywhere else, with different
constants.

**What is still missing, and why the probe could not settle it.** The two
nibble planes' separation at 64 channels is unknown. The probe flips a weight
from `+0.1` to `-0.1`, and for this model's scale those quantise to `0xC0` and
`0x40` -- a *low nibble of zero in both*, so only the high plane ever moved
and the low one stayed invisible. Searching gaps from -2000 to +2000 bytes for
a pair that reconstructs the weights found nothing, so the placement is not a
simple offset from the plane the probe did find.

The fix is mechanical and known: probe with a pair of values whose codes
differ in *both* nibbles. That was done -- see the next section.

### Closing the 1-D layout to 128 channels

The blind spot was the probe, not the format. Flipping a weight between `+0.1`
and `-0.1` at a pinned peak of 0.2 quantises to `0xC0` and `0x40` -- both with
a **low nibble of zero** -- so only the high plane ever moved and the byte it
moved was mistaken for the block's base. A pair differing in both nibbles
(`+0.1` and `-0.1176`, codes `0xC0` and `0x35`) shows both planes at once, and
the separation is the same 36 bytes it always was.

**Both widths are now exact -- every code, not a correlation:**

    A    = 144 * ceil((Cin/2) * K / 36)
    m    = min(Cout // 4, 16)
    slot = (Cin/2)*k + i//2
    byte = A*(o % m) + 72*((o // m) & 1) + top*(o // (2*m))
         + 144*(slot // 36) + (slot % 36)
    high plane = byte + 36,  nibble = i % 2

`A` is the piece that unifies the widths: 288, 432 and 864 at 32, 64 and 128
channels, all from `144 * ceil((Cin/2)*K/36)`. Bit `log2(m)` of the output
channel always costs 72, exactly as at every other width. `top` is `m*A` at 64
channels and `m*A + 256` at 128 -- an extra 256-byte region per super-block
that is not yet explained.

The chunk boundary was the other thing the earlier probes missed: they only
touched slots below 36, so they never crossed one. Slot 36 lands at byte 144
and slot 64 at 172, so the chunk stride is 144, unchanged from 32 channels.

**On the vocoder this moves 1 layer to 4 of 23**, and reading the failures
properly then moved it to 10 -- see the next section.

Test: `test_1d_weight_layout_holds_at_64_and_128_channels` (Docker, no
device).

### Dilation reorders the weights

The "every located layer has K=3" reading of the vocoder was the wrong
variable. Its resblocks pair convolutions of the *same* shape that differ only
in dilation -- 1 against 2, 2 against 6, 3 against 12 -- and exactly one of
each pair was locating. Every failure had dilation greater than one.

**Dilation changes the layout, and only the layout.** A 64-channel `K = 3`
convolution built at dilation 1, 2 and 4 produces a weight table of *identical
size* (14,376 bytes) each time, and the undilated rule reads them at 100%,
33.7% and 33.7%. Same data, rearranged -- and the two dilated builds agree
with each other, so the arrangement does not depend on *how much* dilation,
only on whether there is any.

**The dilated rule gives the kernel index its own chunk.** Everything still
chunks at 36 with a stride of 144; dilation only changes what the outer
dimension is:

    undilated:  slot = (Cin/2)*k + i//2
                byte = 144*(slot // 36) + slot % 36

    dilated:    byte = 144*ceil((Cin/2)/36)*k          [a whole tap per chunk]
                     + 144*((i//2) // 36) + (i//2) % 36

Each tap starts a fresh chunk instead of packing taps together, which is what
you would expect when the taps read discontiguous input. The input channels
chunk inside it exactly as before -- invisible at 64 channels, where
`Cin/2 = 32` fits in one chunk, and required at 128, where it does not: the
tap stride is 144 there and 288 here. Everything else, the output-channel
map, the 36-byte plane gap and the nibble by `i % 2`, is unchanged. Verified
at 100.000% of codes at both widths.

**The vocoder goes from 4 to 10 of 23 layers** on the first form of this
rule, and to 14 once the input chunking above is included -- **50.5% of its
weights** together with the transposed convolutions. The remaining failures
are a narrower set again:

| still unread | why it is plausible |
| --- | --- |
| all three `ConvTranspose` | a layout never probed at all |
| `conv_pre` (256, 192, 7) | 192 and 256 channels, past anything measured |
| 128-channel dilated layers | 128 works undilated, so the two rules interact |
| large dilated extents at 64 and 128 channels | see below |

That was 7.1% of the vocoder's weights, because the three transposed
convolutions and `conv_pre` hold most of them. The transposed ones are now
readable too -- see the next section, which takes it to 47.5%.

Test: `test_dilation_changes_the_weight_layout` (Docker, no device), which
checks both rules against their own builds and that the undilated rule does
not read a dilated one.

### Pulsar2 7.0 implements NVFP4, and it is still unusable

All the work above uses Pulsar2 6.0. The vendor publishes releases on
HuggingFace at `AXERA-TECH/Pulsar2`, and **7.0 exists**. Its release notes
carry entries that 6.0's do not:

    support conv nvfp4
    feat/quant-support-conv-nvfp4
    quant: expand nvfp4 full graph config as weight-only
    fix/8860_nvfp4_w_scale_cvt
    fix/8860_mcore_nvfp4

So the "NVFP4 parses and is silently ignored" finding was a statement about
6.0, and it does not carry forward. Under 7.0 the same config **fails to
build**.

**What 7.0 actually does with it.** The failure is specific and says why:

    TileFailException: AxConv, cannot reshape array of size 2304
                       into shape (64, 1, 1, 1)

2304 is 36,864 weights divided by 16 -- NVFP4's **block** scales, one per 16
elements, being reshaped into a per-output-channel array that cannot hold
them. So the quantiser emits block scales and the backend still wants one
scale per channel.

That predicts exactly when it can succeed: when each output channel is
*exactly one block*. It does.

| shape | blocks per output channel | result |
| --- | --- | --- |
| `Cin=16, K=1` | 1.00 | **builds** |
| `Cin=32, K=1` | 2.00 | fails, "size 32" |
| `Cin=16, K=3` | 9.00 | fails, "size 144" |

The failing sizes are the channel count times the block count, which is the
same arithmetic from the other side. So NVFP4 conv support in 7.0 accepts
only `Cin * K * K == 16` -- a shape no real convolution has.

**And it is not a chip gate.** The same build fails identically with
`target_hardware` set to `AX8860`, despite two of the five NVFP4 commits
naming that part. Both targets reach the same reshape.

**Activations are unchanged.** 7.0's `data_type` still offers
`U8, S8, U16, S16, FP32`, and `NVFP4` remains the only 4-bit name in the
enum. So the W4A4 conclusion stands for 7.0 as well: **no 4-bit arithmetic in
either version**, and now for a sharper reason -- 7.0 has the quantiser for
4-bit weights and a backend that cannot yet consume its scales.

Worth noting for anyone extending this: the HuggingFace repo also carries
4.x, 5.x and dated "temp" builds, one of which is `4.2-temp-cosyvoice2` --
a TTS-specific hotfix, which may matter to the vocoder work above.

### The DSPs, and why they cannot close the gap to the rating

**First, what the card is.** Three independent identifiers agree: the PCI
device is `1f4b:0650`, the host firmware is `ax650_card.pac`, and `axcl-smi`
reports `AX650N`. So the measurements above are AX650N measurements. (An
M5Stack LLM-8850 is a different part -- AX8850, rated 24 TOPS INT8 -- and
would present a different device ID and firmware.)

**The DSPs are real and programmable.** The AX650N carries **two Cadence
Tensilica Vision Q7** cores alongside the NPU, and Cadence lists the part on
its platform pages, so a Tensilica toolchain can target them. From this side
the interface is thin: `libaxcl_dsp.so` exports exactly seven functions --
`PowerOn`, `PowerOff`, `LoadBin`, `EnableCore`, `DisableCore`, `PRC`,
`Query`. That is a firmware loader and a mailbox. No kernels ship with AXCL,
and `pulsar2 build` has no DSP path at all: its backends are `ax620l`,
`ax650npu` and `ax8860`, and neither the 6.0 nor the 7.0 release notes
mention DSP.

**And the arithmetic says they would not be enough anyway.** A Vision Q7
provides **512 8-bit MACs per cycle**. Closing the gap from the measured
10.13 TOPS to the rated 18 would need 7.87 TOPS from two DSPs, or 3.9 TOPS
each -- which at 512 MACs/cycle demands roughly **3.8 GHz**. Even a generous
1.5 GHz gives about 1.5 TOPS per core, so the pair can offer perhaps 2 to 3
TOPS: a useful 20-30% on top of the NPU, not a doubling.

**Where the gap actually is: array utilisation, not clock.** The NPU
sustains **4,949 MAC/cycle** across its cores, measured and stable across
shapes.

The clock no longer has to be guessed. A `profile=True` build writes a
`trace.json` whose events carry durations, and comparing their span against
the same build's `max_cycle` gives the toolchain's own conversion:

| graph | max_cycle | trace span | nominal |
| --- | --- | --- | --- |
| 1024ch 16x16 | 3,905,355 | 3906.3 us | 999.8 MHz |
| 256ch 64x64 | 3,831,460 | 3832.7 us | 999.7 MHz |
| 512ch 32x32 | 3,704,778 | 3706.5 us | 999.5 MHz |

The compiler emits exactly one cycle per nanosecond, so its **nominal NPU
clock is 1.0 GHz**. Measured against real device time the same builds give
854 to 949 MHz, so the hardware runs at **85-95% of nominal** -- close to it,
not at half of something larger.

That settles the arithmetic. At 1.0 GHz, 18 TOPS requires

    18e12 / (2 * 1.0e9) = 9,000 MAC/cycle

and the hardware sustains 4,949, or **55%** of it. The kernel log names
sixteen execution units (`EU[0]` through `EU[15]`), and 9,000 over 16 is 562
per unit -- consistent with a 512-MAC unit per EU and a rating that assumes
the whole array busy.

So the distance to 18 TOPS is how much of the array a real convolution keeps
fed, not clock and not the DSPs. An earlier draft of this section proposed a
1.82 GHz rated clock with this card running at half; the trace says
otherwise and that reading was wrong.

**What is still not directly readable** is the hardware's instantaneous
frequency: AXCL exposes no NPU clock (`axcl-smi info --npu` gives usage and
an engine version; `set --freq` sets the *CPU*, offering 1200/1400/1700 MHz),
the device kernel log prints none, and no `AX_SYS_*` clock getter is exported
even though `AX_NPU_CLK_ID` exists in the headers. The 1.0 GHz above is the
compiler's nominal, and the 854-949 MHz is measured effective throughput --
between them there is no room for a large hidden clock deficit.

**So: using the DSPs is possible but would need firmware written from
scratch with a toolchain that is not part of this stack, and by the numbers
it buys a fraction of what the rating implies. The rating's missing factor is
much better explained by clock.**

### A transposed convolution is several convolutions

The three `ConvTranspose` upsamplers hold most of the vocoder's weights and
none of them read. They are not stored in a layout of their own.

**Unstrided, it is the ordinary conv layout with two adjustments.** ONNX
orders a `ConvTranspose` weight `(Cin, Cout, K)` rather than
`(Cout, Cin, K)`, and dimension 0 behaves as the *input* channel exactly as
`i` does for a convolution -- weights `(0,0,0)` and `(1,0,0)` land in the
same byte, different nibbles. The taps are then stored **reversed**: with
`K = 4` at 32 channels, `k = 3` sits at byte 0, `k = 2` at 16, `k = 1` at 32
and `k = 0` at 156, which is slot 48 chunked at 36 into `144 + 12`. Reading
`(Cout, Cin, K)` with `k' = K-1-k` reproduces every code.

**Strided, it is split into `stride` separate convolutions.** At stride 2 the
taps do not stay together: `k = 0` and `k = 2` land 16 bytes apart while
`k = 1` sits in an entirely different region. That is polyphase
decomposition -- a stride-`s` transposed convolution compiled as `s` ordinary
convolutions of `K/s` taps each, which is the standard way to implement one.

**All twenty phases of the vocoder's three upsamplers locate**, at 0.9997 to
0.9999:

| layer | stride | phases | located |
| --- | --- | --- | --- |
| `(256, 128, 16)` | 8 | 8 x 2 taps | 8/8 |
| `(128, 64, 16)` | 8 | 8 x 2 taps | 8/8 |
| `(64, 32, 8)` | 4 | 4 x 2 taps | 4/4 |

**The vocoder goes from 10 to 13 of 23 layers, and from 7.1% to 47.5% of its
weights.** Three layers carried forty points of coverage, which is what
happens when the unread ones are the big ones.

What remains is `conv_pre` at 192 input channels, and the dilated layers at
128 channels and at dilations 6 and 12 -- the same width-and-dilation
interaction noted before, now the only thing between here and a fully read
vocoder.

Test: `test_convtranspose_stores_taps_reversed_in_the_conv_layout` (Docker,
no device), which checks every code and that dropping the reversal breaks it.

### The LLM path's weight encoding

Everything decoded so far is `pulsar2 build` output. `llm_build` is a
different compiler entry point, and its weight tables had never been looked
at -- coverage there was zero. They are not a different format.

**The tell is the zero weight.** In the two-nibble-plane INT8 encoding an
all-zero weight reads `0x00` in the low plane and `0x88` in the high one, so
a real plane gap makes `0x00` at some offset predict `0x88` a fixed distance
later. Scanning gaps 1 to 64 over a layer's `npu_params`:

| model | best gap | P(`0x88` at j+gap given `0x00` at j) | lift |
| --- | --- | --- | --- |
| synthetic 4096-hidden, `s8` | **18** | 0.52 | **12.3x** |
| SmolLM2-135M, `s8` | **18** | 0.21 | **7.9x** |
| synthetic 4096-hidden, `s4` | 64 | 0.005 | 0.06x |

So `llm_build` uses **the same two-nibble-plane INT8 encoding, with the
planes 18 bytes apart** rather than the convolution pipeline's 36 -- half the
plane, and the same idea. Two independent models agree on 18, with 17 and 19
as shoulders, so the fine structure may not be a single fixed stride.

**The `s4` build is what makes this an encoding claim rather than a
coincidence.** Four-bit weights need only one nibble, so there is no second
plane to pair with, and indeed no gap anywhere from 1 to 64 shows any lift at
all -- the best is 0.06x, i.e. *less* than chance. The pairing appears
exactly when the format says it should and vanishes exactly when it should.

Table sizes agree independently: 192,889,348 bytes for a layer of about
177.2M parameters is 1.09 bytes per parameter at `s8`, and the `s4` build of
the same layer is 97,952,260 -- almost exactly half.

That takes the LLM path from nothing to a known weight encoding.

**The addressing is half-scale, and measured but not yet complete.**
Single-weight probes on a small synthetic checkpoint give the index costs
directly. Every one is exactly half its convolution-pipeline counterpart:

| quantity | convolution | `llm_build` |
| --- | --- | --- |
| plane gap | 36 | **18** |
| chunk size | 36 | **18** |
| chunk stride | 144 | **72** |
| the "special bit" | 72 | **36** |
| `A` | `144*ceil((Cin/2)/36)` | `72*ceil((Cin/2)/18)` |

Those are read off the probes, not fitted: a column at index 64 lands 86
bytes in, which is `72*1 + 14` under chunking at 18, and index 128 lands at
226, which is `72*3 + 10`. Row bits cost 576, 1152, 2304, 4608 for bits 0-3,
36 for bit 4, then 9728, 19456, 38912.

**The addressing is solved.** Scored by correlation per output row -- which
tests the index without depending on the quantiser -- the map reads **every
row of a 256x256 matmul at 1.0000**, all 256 of them. The index is linear in
the bits, confirmed directly: row 17 costs 612, which is row 1's 576 plus row
16's 36.

An earlier reading of this section reported 45% and called the addressing
unclosed. That number was an *exactness* score, and what it was measuring was
the quantiser, not the index -- a scale-invariant metric would have separated
the two immediately.

**The quantiser is the part still open.** Its magnitude is `peak/128`, not
the convolution pipeline's `peak/127.5`. About half the rows are additionally
stored **negated**, in a pattern that nearly but not exactly follows the sign
of each row's largest-magnitude element (using that sign directly reproduces
93.3% of codes; using the measured per-row signs reproduces 94.7%). So the
residue is a sign convention plus rounding, on top of an index that is fully
understood. The
convolution work needed single-weight probes for that, and `llm_build` takes
a checkpoint rather than a graph, so the same trick needs a synthetic
checkpoint per probe -- slower, but no different in kind.

Test: `test_llm_build_weights_use_the_same_nibble_planes_at_half_the_gap`
(Docker, no device), which checks the peak at 18, that it beats the
convolution pipeline's 36, and that `s4` shows nothing.

### What is left in the vocoder, stated precisely

At **14 of 23 layers and 50.5% of weights**, the failures are no longer a
grab bag. Sorting them by the dilated kernel extent `d*(K-1)+1` makes the
boundary visible:

| layer | extent | reads |
| --- | --- | --- |
| `(32,32,5)` d=6 | 25 | yes |
| `(32,32,7)` d=3 | 19 | yes |
| `(64,64,5)` d=2 | 9 | yes |
| `(64,64,7)` d=3 | 19 | **no** |
| `(64,64,5)` d=6 | 25 | **no** |
| `(32,32,7)` d=12 | 73 | **no** |
| all `(128,128,*)` with K>3 | 9 to 73 | **no** |

An extent of 25 reads at 32 channels and not at 64; an extent of 19 reads at
32 and not at 64. So the limit is not the extent alone and not the width
alone, but their product against some tile -- the same shape of interaction
the polyphase finding turned out to be.

That suggested a decomposition, and it is one -- see the next section.

### A widely dilated convolution is K convolutions

Flipping one weight and counting how many separate byte regions move settles
it, and the count is the whole argument:

| convolution | dilation | regions that move |
| --- | --- | --- |
| 64ch, `K=3` | 1 | 2 (the weight, and the scales) |
| 64ch, `K=7` | 3 | **8** |
| 64ch, `K=5` | 6 | **6** |

Eight regions at `K = 7` and six at `K = 5`: **one per kernel tap**, plus the
scales. Not one per dilation, which is the decomposition one might guess
first -- `d` is 3 and 6 in those two rows, and neither matches.

So once a dilated kernel's footprint outgrows what one input tile holds, the
convolution is compiled as **K separate single-tap convolutions**, each laid
out as an ordinary `K = 1` block. The same move as the polyphase split of a
strided transposed convolution, along a different axis.

(A single weight perturbs *every* region because it shifts the layer's output
range slightly, and each sub-block carries its own requantisation multiplier.
That is why the region count is visible from a one-weight change at all.)

**Reading each tap as its own block takes the vocoder to 19 of 23 layers and
62.5% of its weights**, from 14 and 50.5% (and to 20 and 67.5% once input
splitting is added, below). Layers that had resisted from the
start -- `(64,64,7)` at dilation 12, `(32,32,7)` at dilation 12,
`(128,128,7)` at dilation 3 -- read at 0.9971 to 0.9999.

**The split is two-dimensional.** Counting regions for the layers that still
failed shows the sub-block count is not always `K`:

| convolution | dilation | sub-blocks |
| --- | --- | --- |
| 128ch, `K=7` | 3 | 7 (one per tap) |
| 128ch, `K=5` | 2 | **2** (input halves, taps together) |
| 128ch, `K=7` | 12 | **21** (7 taps x 3 input groups) |

So a convolution is split along taps *and* along input channels, by whatever
the tiling needs. Adding input-group splits to the reader -- trying 1, 2 and
4 groups -- takes the vocoder to **20 of 23 layers and 67.5% of its
weights**.

**Three remain**: `conv_pre` at `(256,192,7)`, `(128,128,5)` at dilation 6,
and `(128,128,7)` at dilation 12. The last two are close rather than opaque:
15 of 20 and 25 of 28 of their sub-blocks locate individually at 1.0000, and
the misses look like search collisions -- with four input groups a block is
only 32 channels wide, which is a 16-byte pattern, short enough for a
whole-table scan to find a false maximum. Confirming them needs a sharper
search rather than a new rule.

Test: `test_a_widely_dilated_conv_splits_into_single_tap_convolutions`
(Docker, no device), which counts regions rather than searching for them --
the count is the claim, and it is robust where a base-offset search is not.

### Verifying by exact codes, not correlation

Every coverage number above was scored by correlation, which is the right
tool for finding a block but the wrong one for confirming it. With sixteen
input groups a block is twelve channels wide -- a six-byte pattern -- and a
whole-table scan will find a spurious maximum for something that short.

Switching the criterion to **exact code match** fixes both ends. The
probability of twelve bytes matching by chance is about `256^-12`, so a hit is
real; and because it never stops at a false maximum, the search finds *more*:

| criterion | layers | weights |
| --- | --- | --- |
| correlation > 0.99 | 20 of 23 | 67.5% |
| **every code exact** | **20 of 23** | **67.5%** |

Every code of two thirds of a trained vocoder's weights, including the
524K-weight upsampler, is now predicted from the ONNX file alone.

**Three layers remain**, and the pattern in them is not width: `conv_pre`
`(256,192,7)`, and the two 128-channel convolutions with the widest dilations
(`K=5 d=6` at 3 of 5 tap blocks, `K=7 d=12` at 18 of 28). Their 64-channel
twins read exactly, so whatever splits them is something 128 channels and a
wide dilation do together.

**Two bugs, both in the checker rather than the format**, and correlation had
hidden both by being scale- and offset-invariant:

* The first exact-match run lost all three transposed convolutions. It
  computed each polyphase slice's quantisation peak *from that slice*; the
  compiler scales per output channel over the whole kernel, so a phase's
  expected codes have to come from the whole tensor and be sliced afterwards.
  Fixing it is what makes the 256-channel upsampler readable.
* The second lost every dilated convolution. A dilated layer does not pack its
  taps into one slot space -- it rounds **each tap** up to a whole number of
  144-byte chunks, `tap stride = 144*ceil((Cin/2)/36)`, against the undilated
  `slot = (Cin/2)*k + i/2`. At `Cin=128, K=3` the two give the same output
  stride, which is why a probe at that one shape had not separated them.

### The layout does not change at 256 channels

Since 128 channels was where the layout last grew a term, the natural guess
for the unread layers was that 256 grows another. A dense probe says no. At
`Cin = Cout = 256, K = 3`, every constant is the one already recorded:

| probe | address | rule |
| --- | --- | --- |
| `i = 72` | 144 | slot chunked at 36, stride 144 |
| `i = 255` | 451 | `144*3 + 19` |
| `k = 1, 2` | 452, 1012 | `slot = (Cin/2)*k + i/2` |
| `o = 1` | 1584 | `A = 144*ceil((Cin/2)*K/36)` |
| `o = 16` | 72 | bit 4 of the output channel costs 72 |
| `o = 32` | 25600 | `top = 16*A + 256` |

So `top = m*A + 256` holds at 128 channels **and** at 256 -- the extra 256
bytes are not a per-128-channel pad but a fixed region per super-block. A
super-block covers 32 output channels, and the probe also moved four bytes at
`m*A + 4*o` for each channel it touched: the requantisation multipliers, 32 x
4 = 128 bytes. That leaves exactly 128 bytes unaccounted for -- room for one
more per-channel float32 table, which the bias is the obvious candidate for
and which no probe has yet moved.

### Reading the whole vocoder, by discovering the geometry

Exact-code coverage had stalled at 67.5% with three layers unread. The
blocker was not the format -- it was that every search so far *assumed* a
block's geometry and then looked for it. Three assumptions were wrong, and
each one hid the next.

**The first: that a block's input tile starts on a chunk boundary.** Slots are
chunked 36 at a time with a 144-byte stride, and a search that lays a tile out
contiguously from its own start is only right when the tile begins at a
multiple of 36. In a packed layout tap `k` begins at slot `(Cin/2)*k`, which
lands mid-chunk for almost every `k` -- so every tile that straddled a
boundary was silently missed. Addressing each weight at its *absolute*
intra-block slot and searching for the block base instead fixes it, and has
the side benefit that one search now covers a whole block rather than a tile.

**The second: that the member stride, the odd-register offset and the
super-block stride follow the formula.** They usually do, but "usually" is not
a criterion. So they are now read off the table: find output channel 0's block
by exact bytes, then take the stride from where channel 1 sits, the odd offset
from channel 16, and the super-block stride from channel 32, and only then
verify every code of every output channel.

**The third: that a layer splits its input channels one way.** It does not.
The split is per **tap**. A byte-by-byte read of a widely dilated build showed
its tap 0 ending at channel 31 -- with two slots of padding after it -- while
every other tap of the same layer kept all 128 channels in one block. Choosing
the split per tap rather than per layer is what closes the last two layers,
and `conv_pre` wanted an uneven `32 + 160` that no uniform tiling would ever
have proposed.

| | layers at 100% | weights read |
| --- | --- | --- |
| assumed geometry | 20 of 23 | 67.5% |
| discovered geometry | 20 of 23 | 90.6% |
| **+ per-tap splits** | **23 of 23** | **100.0%** |

That is every one of the 1,661,152 weights of a real trained vocoder,
predicted from the ONNX file alone and matched code for code -- transposed
layers, dilated layers and all.

The conventions it discovers are consistent, and worth stating because they
are now measured rather than assumed:

* an **undilated** convolution is one block, the whole kernel packed into a
  single slot space;
* a **dilated** one is one block per tap, each tap padded up to a whole number
  of 144-byte chunks;
* a **transposed** one is one block per polyphase, taps reversed;
* and any tap may split its input channels further, unevenly, independently of
  its neighbours.

That last point is the one to carry forward. Every earlier failure to read a
layer was a search looking for a layer-wide rule that does not exist: the
allocator decides per tap, and the only way to know what it decided is to read
it back out of the table.

### The activation scales, decoded

Rewriting a compiled model's weights only worked for weights that preserved
each output tensor's dynamic range, because every convolution's output
activation scale was calibrated from the original weights and stored somewhere
unread. An earlier search for those scales as float32 found one of eleven and
concluded they must be fixed-point. They are neither: they are **bfloat16**,
which is why a float32 search found almost nothing and a byte-order search
found nothing at all.

**The differential that isolates them.** Build one identical model against
calibration inputs scaled by 1, 2 and 4. Both the input and the output
activation scale then scale with the amplitude, while the weight codes and the
requantisation multiplier

    M[o] = x_scale * (peak[o]/127.5) / r_scale

are invariant -- the amplitude cancels. So the weight table must not move at
all, and every byte that *does* move is an activation scale and nothing else.
The weight table does not move, and 22 bytes of a 2,824-byte mcode do.

**What they are.** Two families of four copies each, both bfloat16 rounded
*toward zero*:

| field | holds | amp 1 | amp 2 | amp 4 |
| --- | --- | --- | --- | --- |
| four 16-bit slots | `1/x_scale` | 35.75 | 17.875 | 8.9375 |
| four 16-bit slots | `y_scale` | 0.0122681 | 0.0245361 | 0.0490723 |

Checked against pulsar2's own `quant_axmodel.json`, all six agree exactly:
`bf16_trunc(1/x_scale)` and `bf16_trunc(y_scale)`, where `bf16_trunc` is the
float32's top 16 bits with the rest discarded. Note the asymmetry -- the input
scale is stored **reciprocated** and the output scale is not, which is what a
requantisation datapath that multiplies by one and divides by the other would
want.

The remaining six moving bytes are not values at all: across the three builds
they hold the same multiset in a different order, so they are an ordering that
depends on the scales rather than a scale.

Test: `test_activation_scales_are_bfloat16_in_the_mcode` (Docker, no device).

### Held out: a second vocoder, and why the splits must be walked

100% on the model a method was developed against is not evidence the method
works. The synthetic HiFi-GAN-shaped vocoder built earlier in this file is a
genuine held-out test: different depths, different channel counts, an 80-band
mel input instead of a 192-wide latent, and never once consulted while the
reading pass was written.

It came back at **91.3%** -- five of six layers exact, and the sixth was its
`conv_pre` equivalent, `(128,80,7)`, at 71.4%. Enumerating candidate input
splits had reached its limit: 80 channels do not split any way the candidate
list contained.

So the splits are now **walked** rather than guessed. Within a block, tap 0
always occupies slots `0..w/2-1` whatever the tap convention is, so its
channels can be walked one pair at a time until the codes stop matching, and
that measures `w`. It is the same move that measured the LLM's column blocks,
and it takes both vocoders to **100.0%**.

What it measured is the argument for it. That first layer's splits, per tap:

| tap | 0 | 1 | 2 |
| --- | --- | --- | --- |
| blocks | `[80]` | `[48, 32]` | `[24, 56]` |

`24 + 56`. No candidate list anyone would write contains that. The allocator
is not applying a rule with parameters -- it is packing, and the only way to
know how it packed is to read it back.

### The LLM layout at 4096 hidden: column blocks, and all of them

The `llm_build` addressing was solved at 256 hidden and scored 0.73 by
correlation at 4096, which was recorded as a likely block split. It is a
column split, and nothing else changes. Walking the table -- following columns
until the codes stop matching, rather than assuming a width -- gives eight
blocks:

| block | columns | width | row stride `a` | `72*ceil(w/2/18)` |
| --- | --- | --- | --- | --- |
| 1 | 0..287 | 288 | 576 | 576 |
| 2..8 | 288..4095 | 544 each | 1152 | 1152 |

Every block's row stride is the same `72*ceil((width/2)/18)` that governs 256
hidden, every block sits `16*a + 512` bytes after the last -- the same `top`
the conv layout uses -- and **all 4096 rows of every block verify exactly**.

That is 16,777,216 codes of a 4096x4096 `q_proj`, **100.0% of them**, from the
checkpoint alone. The row addressing was never the problem at 4096; the guess
that 0.73 meant "the layout breaks at this width" was wrong, and the only
thing that was actually unknown was where one block stops.

### The LLM path quantises differently, and here it is exactly

The convolution pipeline's quantiser was pinned down earlier: `scale =
float32(peak)/float32(127.5)`, round half to even, offset 128. The `llm_build`
path had resisted, sitting at 94.7% with "an unexplained per-row negation".
The negation was the clue, and all three of its differences are real:

```python
scale = -w[argmax|w|] / 128      # the SIGNED weight at the peak, negated
code  = clip(floor(w/scale + 0.5) + 128, 0, 255)     # ties round toward +inf
```

* **`/128`, not `/127.5`.** At 127.5 the fit is 90.4%; at 128, 95.2%.
* **The signed peak, negated.** The scale is not `|peak|/128` -- it is the
  weight *at* the peak index, sign included, with a minus in front. Stated
  without the arithmetic: the row's most extreme weight always maps to code 0
  and zero maps to 128, whichever sign that extreme has. It is the usual code
  mirrored. Using `|peak|` instead scores 54% -- about half the rows, which is
  exactly how many happen to have a positive extreme.
* **Ties round toward `+inf`.** Every last miss sat at exactly `x.5`, and
  every one of them rounded up: `-88.5` became `-88`, not `-89`.

And the precision is **the checkpoint's own**, not the compiler's. The same
weights written as a BF16 safetensors file are quantised from bfloat16 values;
written as F32, from float32. This was worth an experiment rather than an
assumption -- a bfloat16 fit of an F32 checkpoint scores nothing at all.

| checkpoint | matmuls | exact |
| --- | --- | --- |
| BF16 | q, k, v, o, gate, up, down | **100.000%**, 32 of 32 rows each |
| F32 | q_proj | **100.000%**, 32 of 32 rows |

With the addressing already solved, that closes the `llm_build` weight table:
its contents are now computable from the checkpoint alone, the same as the
convolution table. Two quantisers in one toolchain, sharing no constant --
which is worth remembering before assuming any other part of the two paths
matches.

### Checking an mcode without Docker and without a card

Everything above needed a Pulsar2 image to compile with or an AX650N to
confirm on. That makes it untestable in ordinary CI, and a format nobody
re-checks is a format that quietly rots. `mcode.py` fixes that: the codec and
every structural rule confirmed on hardware, in a module that imports numpy
and onnx and nothing else, plus three real compiled streams committed under
`fixtures/`.

`mcode.check(blob)` returns a list of violations -- empty means well-formed as
far as the format is understood. The rules are exactly the confirmed findings,
no more:

1. the segment table is a loader manifest, and its word counts must tile the
   stream from the end of the header to the tail vector;
2. the codec round-trips byte for byte;
3. `a7` sits within four bytes of every segment boundary but the first (the
   verb can begin just before the word boundary its segment starts on);
4. only the six known verbs, and only tags below the verb range plus `a1` and
   the odd-register form of each;
5. at least 94% of non-zero bytes belong to a recognised form;
6. no unexplained non-zero run is longer than twelve bytes.

Rules 3, 5 and 6 are thresholds, so they are measured rather than guessed:
across seventy real streams -- convolutional and `llm_build`, up to 1.8 MB --
coverage never fell below 94.1%, and the longest unexplained run anywhere was
nine bytes. Writing rule 3 down from memory as "every segment opens with a7"
produced a checker that failed on all three fixtures; the real streams
corrected it.

**What it catches, and what it does not.** A deleted byte, bulk noise, a lost
manifest, a stream that is not an mcode at all -- all caught. A single flipped
byte in an operand is *not*, and should not be: operands are addresses and
sizes the allocator chose, and the format permits any of them. This checks
form.

**There is deliberately no evaluator.** What a verb computes is not known --
no verb's datapath semantics have been established, and 25 of the 28 operand
slots that move between builds hold allocator output. An interpreter that
produced numbers would be inventing them. `check` answers the question that
can be answered honestly: could the runtime load and walk this?

## LLMs: a separate pipeline onnxsim has no hook into

**Confirmed real, end to end** (`pulsar2:6.0-lite` + a real `Qwen/Qwen3-0.6B`
checkpoint + the real AX650N): Axera compiles LLMs through a **completely
different** subcommand, `pulsar2 llm_build` (Pulsar2's newer docs call it
`llm_build2` with a slightly different flag set -- v6.0 only has
`llm_build`; see `pulsar2_docker.llm_build()`'s docstring for the exact
confirmed flags). This is *not* a variant of `pulsar2 build` with an LLM
config -- **`--input_path` is a raw HuggingFace checkpoint directory**
(`*.safetensors`/`pytorch_model.bin` + `config.json`), not an ONNX model.
There is no ONNX step anywhere in this pipeline: the public `ax-llm-build`
project (github.com/AXERA-TECH/ax-llm-build) that Pulsar2's own docs point
to for this workflow contains no model-tracing/export code at all, only
per-architecture config JSONs and small pre/post-processing helper scripts
around the actual (closed-source) `pulsar2 llm_build` call.

**So onnxsim has no direct integration point in Axera's LLM ingestion
path** -- there is no ONNX graph for `onnxsim.simplify()` or any of
onnxsim's GPTQ/AWQ/NF4/`auto_quantize_int4`-family quantizers to act on
before Pulsar2 ever sees the model. `pulsar2 llm_build`'s own
`--weight_type` (`s8` by default, `s4` available) is Pulsar2's own built-in
weight quantization -- unrelated to, and not replaceable by,
`pulsar2_quantizer.py`.

What *is* confirmed and now supported by this harness:

- `pulsar2_docker.llm_build()` wraps the real command. Verified against
  `Qwen/Qwen3-0.6B`: ~7-8 minutes end to end on a 32-core host with
  `--parallel 8`, producing one `<name>_p<prefill_len>_l<N>_together.axmodel`
  per transformer layer (28 for this model) plus one `<name>_post.axmodel`
  (the LM head) -- confirming the original handoff notes' guess that LLMs
  compile to "a directory of small, structurally similar single-block
  graphs," not one big graph.
- Each per-layer file has **two** `neu mode` nodes, not one: a decode
  subgraph (batch-1 shapes) and a prefill subgraph (`prefill_len`-batch
  shapes), sharing one `npu_params` initializer, each with explicit
  `K_cache`/`V_cache` graph inputs *and* `*_out` outputs -- the KV cache is
  ordinary graph tensors the host runtime (`ax-llm`/`axllm`) persists
  between calls, not something hidden inside the compiled blob.
- `pulsar2_ops.py`'s corruption detectors (`has_out_of_band_npu_data()`/
  `missing_npu_data()`) already handle multiple NPU nodes per graph
  correctly with no changes needed. Verified: `onnxsim.simplify()` corrupts
  a real per-layer LLM `.axmodel` the exact same way as the CNN case (3
  initializers -> 0). `models.axera_llm_layer_leaf()` reproduces this shape
  in CI without needing hardware or a real LLM download.
- A per-layer file and the post model both ran successfully on the real
  AX650N via `axcl_run_model` (~1.5ms and ~9ms respectively).
- **Confirmed real, directly from a compiled layer's own declared I/O
  dtypes (a real `HuggingFaceTB/SmolLM2-135M` build, `--help`'s
  `hidden_state_type`/`weight_type` defaults of `bf16`/`s8`): this is
  genuine weight-only quantization, not the full weight+activation INT8
  PTQ the generic path below applies.** Every graph input/output on both
  `neu mode` nodes -- `K_cache`, `V_cache`, the hidden state, and the
  attention `mask` -- is declared `BFLOAT16`; activations never get
  quantized at all. Only `npu_params` shrinks: 3,712,328 bytes for a
  576-hidden-size layer whose real weight element count (q/k/v/o/gate/up/
  down projections + 2 RMSNorm weights) is ~3.54M -- ~1 byte/element,
  confirming S8 weights, not the ~2 bytes/element BF16 would need. This is
  the confirmed, direct explanation for the real accuracy gap found
  below ("Confirmed against a real, full-size model"): the generic
  `pulsar2 build --config` path quantizes *both* weights and activations
  uniformly to INT8 with no smoothing, which compounds into near-random
  output by 30 layers deep; `llm_build()` never quantizes the residual
  stream/KV-cache/attention path at all, only the static weights.
- **`model_type` support is narrower here than the generic path's**:
  `llm_build --input_path` on a real `mistral`-architecture checkpoint
  (`distilabel-internal-testing/tiny-random-mistral`, same one used
  elsewhere in this README) fails outright with `AssertionError:
  model_type error mistral` -- confirming its per-architecture allowlist
  (`yasched/llm_builder/{llama,qwen3,gemma,...}_test.py`, all
  Pyarmor-obfuscated, see above) has no `mistral` entry, unlike
  `reconstruct_hf_graph()`, which treats `mistral` as llama-family-
  compatible. `llama` (confirmed via `SmolLM2-135M`) and `qwen3`
  (confirmed via `Qwen3-0.6B`) both work.

## An alternative LLM path that *does* give onnxsim a hook

The section above is about Pulsar2's own, closed-source `pulsar2 llm_build`
ingestion path, which never touches ONNX. Separately, onnxsim has its own
`onnxsim.reconstruct_hf_graph()` (see `onnxsim/hf_reconstruct.py`) --
builds a runnable ONNX graph directly from a HF checkpoint directory
(`config.json` + safetensors; llama/mistral/qwen2/qwen3 today). Feeding
*that* ONNX graph through the ordinary `pulsar2 build` (the same
CNN/vision ingestion `convert_onnxmodelzoo.py` uses, not `llm_build`) is a
second, independent LLM path with a real onnxsim integration point --
`onnxsim.simplify()`/quantizers can act on the graph before Pulsar2 ever
sees it, unlike the `llm_build` path above.

**Confirmed real, end to end**: a synthetic tiny (2-layer) Llama-shaped
checkpoint, run through `reconstruct_hf_graph()` then a real `pulsar2
build --target_hardware AX650`, compiled cleanly to a single-`neu
mode`-node `compiled.axmodel`, which then ran successfully on a real
AX650N via `axcl_run_model`. Notably, `pulsar2_ops.AX650_SUPPORTED_OPS`
(the doc-scraped op list) flags `Neg` (used by RoPE's rotate-half) as
unsupported, but the real build compiled it without complaint regardless
-- a reminder that the scraped table is a fast pre-screen, not a
guaranteed predictor, once fused patterns are involved.

`pulsar2_docker.build_from_hf_checkpoint()` wraps this whole path:
reconstructs the ONNX graph, auto-generates `Numpy`-format calibration
tars for `reconstruct_hf_graph`'s two inputs (`input_ids`, random token
ids in `[0, vocab_size)`; `position_ids`, `arange(seq_len)`), writes the
two-input quant config Pulsar2 needs (`calibration_format: Numpy` per
`InputQuantConfig`, confirmed from the Docker image's own
`build_config.proto`), and calls `build()`. See
`tests/test_pulsar2_hf_to_axmodel.py` for the full working example.

### Confirmed against a real, full-size model: `HuggingFaceTB/SmolLM2-135M`

Everything above was verified against tiny synthetic or near-random-weight
checkpoints. Compiling a real, genuinely-trained 135M-parameter checkpoint
(30 layers, GQA, 49152-token vocabulary, real BF16 weights) through this
same path surfaced a real bug this repo's own BF16 handling had never hit
before, plus a real accuracy caveat:

- **The `Cast`-in-graph BF16 design was never actually exercised against a
  real `pulsar2 build` until now, and it's fundamentally broken there,
  at any size.** `reconstruct_hf_graph()` (confirmed against the real
  ~1.5GB `Qwen/Qwen3-0.6B` checkpoint, see above) always used a
  graph-level `Cast` node for BF16 weights specifically to keep the
  initializer small and avoid protobuf's ~2.1GB serialization limit -- but
  that Cast node had only ever been run through `onnxruntime`, never a
  real Pulsar2 compile. Compiling `SmolLM2-135M` for real hit
  `Exception: op name: model.embed_tokens.weight.f32.1, Cast, pyrun
  failed.` inside Pulsar2's own frontend constant-folding pass. Isolated
  to a standalone, minimal repro: a bare `Cast<to=FLOAT>` on a BFLOAT16
  initializer fails identically at *every* size tested, from a trivial
  4-element tensor up through the real 49152x576 embedding table --
  ruling out "too large" and confirming it's simply unimplemented for
  this dtype pair in Pulsar2's frontend, full stop.
- **Fixed**: `reconstruct_hf_graph()` now upcasts BF16 weights to FLOAT32
  directly in the stored initializer bytes (`_read_tensor()`), the same
  as the *first* approach that Qwen3-0.6B's size had ruled out -- except
  there is no longer a smaller alternative for real hardware, so the size
  cost is accepted. Confirmed safe in practice for a real small/edge-sized
  checkpoint: `SmolLM2-135M`'s ~269MB BF16 checkpoint upcasts to a
  ~251MB *compiled* `.axmodel` (weights end up INT8-quantized on
  Pulsar2's own side, well under the protobuf limit regardless).
  A checkpoint large enough that the FLOAT32 upcast alone would exceed
  ~2.1GB has no working path through `build_from_hf_checkpoint()` today --
  that's what `llm_build()` (above) is for.
- **Compiled successfully**: ~105s wall time (`pulsar2_build` phase; see
  `BuildResult.phase_timings`), `max_cycle=7,334,676` -- and ran
  successfully on the real AX650N with a real tokenized prompt.
- **Real accuracy is bad, and it's a depth-compounding problem, not a
  calibration problem** -- corrected after actually comparing on-device
  output against the real FP32 reference (an earlier pass here just
  eyeballed logit plausibility and wrongly called it fixed). Across 5 real
  prompts, comparing the compiled model's on-device logits against
  `onnxruntime` running the same `reconstruct_hf_graph()` output: **0/5
  top-1 matches, 0/5 top-5 overlap, average cosine similarity ~0.13** --
  the FP32 reference gets every prompt right (" the" for "The capital of
  France is", " dog" for "...the lazy", " oxygen" for "hydrogen and", ...,
  confirming the reconstruction itself is correct), the on-device output
  is close to random. Two follow-ups ruled out calibration as the cause
  rather than confirming it: real, representative English-sentence
  calibration data (32 real sentences, not random token ids) made it
  *worse* (avg cosine ~0.04), and switching `calibration_method` from
  `MinMax` to `MSE` made it worse again (~-0.12). **The real cause,
  isolated by depth**: the identical reconstruction+quantization approach
  gets 0.999 average cosine similarity and 4/5 top-1 matches on a
  synthetic **1-layer** checkpoint (`distilabel-internal-testing/
  tiny-random-mistral`, same pipeline, same code) -- so per-tensor MinMax/
  MSE INT8 post-training quantization, applied uniformly to every weight
  and activation with no smoothing or outlier handling, works fine at
  shallow depth and compounds into essentially-random output somewhere
  between 1 and 30 sequential transformer layers. This is a well-
  documented, expected limitation of naive full-network INT8 PTQ on deep
  transformers in general (it's exactly why techniques like SmoothQuant/
  AWQ/GPTQ exist), not a bug in `reconstruct_hf_graph()`,
  `build_from_hf_checkpoint()`, or its calibration data. Getting real
  accuracy out of a real-depth LLM through this generic ingestion path
  would need a smarter quantization strategy than what a plain `pulsar2
  build --config` currently applies -- `llm_build()` (above), Pulsar2's
  own dedicated LLM path, presumably has one; this generic path doesn't.

### Mitigation attempts: none reproduce `llm_build()`'s accuracy via the generic path

Given the depth-compounding diagnosis above, every quantization knob this
harness has real access to was tried against the real `SmolLM2-135M` build
to see if any of them close the gap to `llm_build()`'s weight-only accuracy.
**None do.** In order tried:

- **`quant.enable_smooth_quant`** (with default and explicit
  threshold/strength): no measurable effect on output.
- **`quant.highest_mix_precision`**: real `TileFailException` -- the
  attention path's promoted-precision matmul tile doesn't fit the NPU's
  memory budget, a hard failure rather than a partial improvement.
- **`quant.layer_configs` with an FP32 override on `ReduceMean`/`Sqrt`**
  (RMSNorm's own ops, the intuitive place to protect precision): confirmed
  **silent no-op**, byte-identical compiled output with and without it.
  Root-caused this session by fetching Pulsar2's own config schema docs
  (`user_guides_advanced/advanced_build_guides.html`): `layer_configs`'
  `data_type: "FP32"` is only valid for a specific, documented op list --
  `LeakyRelu, Sigmoid, Relu, Add, Mul, Div, Sub, Concat, Softmax` -- and
  silently does nothing for anything else, including `ReduceMean`/`Sqrt`.
  Not a bug in this harness; an invalid config value that Pulsar2 doesn't
  validate or warn about.
- **`quant.layer_configs` retried with only doc-valid op types**
  (`Softmax`, `Add`, `Mul`, `Div`, `Sub`, all set to `data_type: "FP32"`,
  stacked incrementally): confirmed **real** this time -- `Add`/`Mul`/
  `Div`/`Sub` overrides each produce a measurably different compiled
  `quant_axmodel.onnx` and different real on-device output (different
  bytes, different per-prompt cosine similarity) than the baseline.
  `Softmax` alone is the one exception: it changes a few bytes of the
  intermediate quantized IR (Pulsar2's own `AxSoftmax` op, not literally
  `Softmax` by then) but produces **bit-identical** on-device output across
  every test prompt -- Softmax's [0, 1]-bounded output is apparently
  already well represented at whatever precision Pulsar2 was already using
  for it. Stacking all four working overrides together does **not**
  recover accuracy -- average cosine similarity across 5 real prompts got
  *worse* (~-0.33 vs baseline's ~-0.20), still 0/5 top-1 matches. Protecting
  a handful of elementwise nonlinearities can't compensate for the
  dominant cost -- MatMul/Conv activations, which `layer_configs` cannot
  set to FP32 at all (`data_type: "FP32"` isn't accepted for `MatMul` or
  `Conv`; `Conv` only accepts a separate `output_data_type: "FP32"`).
- **`quant.enable_adaround`**: never completed -- still running after 30
  minutes on this model, abandoned as impractical for this investigation.
- **Feeding a pre-quantized ONNX graph via `model_type: "QuantONNX"`**,
  to bypass Pulsar2's own PTQ entirely and substitute onnxsim's own
  quantizers instead (`onnxsim.quantize_weight_only()`, matching
  `llm_build()`'s real weight-only S8 scheme exactly, and
  `onnxsim.quantize_static()`, already confirmed elsewhere in this file to
  match AX650's real U8-activation/S8-weight convention). `QuantONNX` is
  confirmed to be a real, distinct ingestion path (`pulsar2 build` prints
  `"... is a QuantONNX model, disable concat align config"` and skips
  requesting calibration ranges for tensors that already carry
  `QuantizeLinear`/`DequantizeLinear`) -- but it hits a real, reproducible
  **Pulsar2-internal bug**: any `MatMul` whose weight input comes through a
  `DequantizeLinear` (the standard ONNX QDQ per-channel weight-quantization
  pattern -- exactly what both onnxsim quantizers above emit) crashes
  Pulsar2's own PPQ-based `ax_quant_graph_optimize` pass with
  `ValueError: Can not feed value to operation <node>, expects exact 2
  inputs, however 1 was given` -- one of `MatMul`'s two inputs is silently
  dropped during Pulsar2's own graph optimization, before either onnxsim
  quantizer's choices could matter. **Isolated to a minimal, 2-node,
  104-byte repro** (`MatMul(x, DequantizeLinear(wq, scale, zp))`, no LLM
  structure involved at all) -- confirms this is a general `QuantONNX` +
  quantized-`MatMul` limitation in Pulsar2 itself, not something specific
  to `reconstruct_hf_graph()`'s output or a fixable onnxsim-side encoding
  choice. Reproduced identically whether the activation side is also
  quantized (`quantize_static()`'s full QDQ output) or left plain float
  (`quantize_weight_only()`'s output) -- ruling out any interaction with
  activation quantization specifically; the crash is purely about `MatMul`
  plus a `DequantizeLinear`'d weight.

**Verdict, after 7 distinct techniques**: the generic `pulsar2 build`
(ONNX) ingestion path cannot currently reproduce `llm_build()`'s real,
usable LLM accuracy on AX650, regardless of which quantization strategy is
applied from the ONNX side -- Pulsar2's own exposed PTQ knobs don't fix the
depth-compounding problem, and substituting a pre-quantized graph runs into
a real toolchain bug for exactly the QDQ pattern that would reproduce
`llm_build()`'s weight-only scheme. `llm_build()`'s separate, closed-source,
non-ONNX ingestion (above) remains the only confirmed-accurate path to a
real LLM `.axmodel` today.

## Real Docker + device conversion driver

`pulsar2_docker.py` and `convert_onnxmodelzoo.py` turn the manual
`docker run ... pulsar2 build` / `axcl_run_model` commands used to produce
every real finding in this README into a reusable pipeline. Unlike
`screen_onnxmodelzoo.py` (static, no Docker/device needed -- run that
first), this does a **real** compile per model, so it needs a loaded Pulsar2
Docker image (see `pulsar2_docker.py`'s docstring for how to get one
matching your device's firmware) and, optionally, a connected AXCL device.

```bash
python scripts/axera/convert_onnxmodelzoo.py \
  --models resnet18d_Opset18 googlenet-6 \
  --profile \
  --output pulsar2-convert.csv
```

For each model: fetches it, `onnxsim.simplify()`s it, `pulsar2 build`s both
the original and simplified ONNX (with `--profile` passing
`--compiler.npu_perf --debug.dump_frontend_graph` through, writing a
`trace.json` per successful build -- see above), and if a device answers,
runs both `.axmodel`s on it with the same input and reports whether the raw
output bytes are bit-identical (this is exactly how the `resnet18d`
bit-identical result in this README was produced). Models are skipped
(`skipped_not_single_image_input`) unless they have exactly one rank-4
input -- NLP/multi-input models need a hand-written config passed to
`pulsar2_docker.build(config_path=...)` directly instead.

One real gotcha worth knowing if you extend this: the Pulsar2 Docker image
must run as root (confirmed: `-u $(id -u):$(id -g)` breaks it -- it needs
root-owned `/root/*.hasplm`/`*.v2c` license files, and a uid absent from the
container's `/etc/passwd` breaks `getpass.getuser()` deep inside a
torchvision import in `pulsar2 version`'s own code path), so everything it
writes under a mounted `work_dir` is root-owned. `pulsar2_docker.
force_rmtree()` handles that (plain `shutil.rmtree` as the host user, falling
back to `docker run --entrypoint /bin/sh <image> -c "rm -rf ..."` on
`PermissionError`) -- use it instead of `shutil.rmtree` for anything under a
Pulsar2 Docker work dir, or root-owned directories accumulate in `/tmp` with
no way for an ordinary user to remove them.

Also note `axcl_run_model -i/-o/-l`'s exact contract, confirmed by trial:
**the input filename must equal the tensor name** (`<in>/0/<tensor_name>.bin`
-- an arbitrary filename fails with "Stimulus file ... is not exist" naming
the tensor). `pulsar2_docker.run_on_device_with_input()` already does this.

## Conv/MatMul variants: what the static heuristic can and can't tell you

Prompted by "how does the axmodel format actually treat different kinds of
Conv/MatMul" -- with no Docker image or AX650N in *this* environment (unlike
the sessions that produced the real-hardware findings above), the honest
thing to check is what the checked-in **static heuristic**
(`pulsar2_ops.AX650_SUPPORTED_OPS` + `pulsar2_simulator.partition()`) can and
can't distinguish, since that heuristic is all a Docker/device-free
environment has to go on. `tests/test_axera_conv_matmul_coverage.py` builds
~16 Conv/MatMul variants via `onnx.parser` and checks them against it:

- **Standard, grouped, depthwise, dilated, strided, `auto_pad`-using, 1-D,
  3-D Conv, and `ConvTranspose`** all read as identical, full NPU coverage.
  So do plain `MatMul`, broadcasting/batched `MatMul`, and `Gemm` under every
  combination of `alpha`/`beta`/`transA`/`transB`. This isn't a bug in the
  test -- it's `partition()`'s own documented design: it classifies purely by
  `node.op_type` membership in `AX650_SUPPORTED_OPS`, the same list
  `inspect_axmodel.py`/`pulsar2_ops.py` scraped from Pulsar2's docs, which
  says nothing about attributes. The docs page itself has per-op
  attribute-level limits (e.g. Conv's `auto_pad` must be `NOTSET`) that
  neither this list nor `partition()` encode -- confirmed absent, not
  confirmed present, since there's no compiler here to check it against.
  Extending `pulsar2_ops.py` with real attribute limits needs a source of
  truth this environment doesn't have (the docs page or a real `pulsar2
  build` failure); making that gap up would be exactly the kind of
  unconfirmed guess this harness otherwise avoids.
- **What *is* checkable without any of that**: none of ONNX's own quantized
  conv/matmul ops (`QLinearConv`, `ConvInteger`, `QLinearMatMul`,
  `MatMulInteger`) are in `AX650_SUPPORTED_OPS` at all, so `partition()`/
  `ax650_build_risks()` flag them as an AX650 build risk regardless of shape.
  This lines up with `pulsar2_quantizer.py`'s separately-confirmed finding
  that Pulsar2's own real PTQ output uses proprietary `AxQuantizedConv`-family
  ops, not standard ONNX quantized operators -- so a graph already quantized
  with ONNX's own vocabulary (e.g. via `onnxsim.quantize_dynamic`, unlike
  `pulsar2_quantizer.quantize_like_pulsar2()`, which stays in QDQ form) is
  something Pulsar2's real frontend has never been confirmed to accept, and
  this heuristic's answer (flag it) is at least consistent with that.
- This also surfaced (and fixed) a real bug in the "no Docker/no-device
  simulator" pitch above: `pulsar2_simulator.py`'s docstring and this
  README both claim `partition()`/`coverage()` "need only `onnx` and work
  regardless" -- but `pulsar2_simulator.py` unconditionally imported
  `pulsar2_quantizer.py`, which unconditionally did `import onnxsim` at
  module scope, so on a checkout where `onnxsim`'s own compiled extension
  isn't built yet (this analysis' own environment, notably -- no real
  `.axmodel`, Docker image, or device either), just importing
  `pulsar2_simulator` for its `onnx`-only `partition()`/`coverage()` raised
  `ModuleNotFoundError` before either function ever ran. Fixed by moving the
  `import onnxsim` inside `pulsar2_quantizer.py`'s existing
  `PULSAR2_QUANTIZER_AVAILABLE` try/except (alongside the `onnxruntime`
  import already there), so a missing/unbuilt `onnxsim` degrades the same
  way a missing `onnxruntime` already did, instead of taking the whole
  import down.

### Confirmed on real hardware: which of these actually build

A later session with real Docker/AX650N access compiled every variant the
static-heuristic analysis above flagged as unverified, through a real
`pulsar2 build`. Results (see
`tests/test_axera_conv_matmul_coverage_hardware.py`):

**Compile successfully, confirming the doc-scraped list under-claims
nothing for these cases**: `Conv` with `auto_pad="SAME_UPPER"` (despite the
docs page reportedly requiring `NOTSET` -- either that limit doesn't hold in
practice for this case, or Pulsar2 silently resolves `auto_pad` to explicit
`pads` before its own limit would apply), 1-D `Conv`, 3-D `Conv`, broadcasting
`MatMul` (rank-3 `A` against a rank-2 `B`), and `Gemm` with `transB=1`.

**Fail outright, with the same "not on `AX650_SUPPORTED_OPS`" mechanism the
static heuristic already predicted**: `ConvInteger`, `QLinearConv`,
`MatMulInteger`, `QLinearMatMul` -- all four real `pulsar2 build` runs threw
the exact `KeyError('dont support <OpType> opr in AXOPS/ONNXOPS/CUSTOM_OPS')`
pattern this repo has seen before (`LRN`, `AxQuantizedGemm` -- see above),
confirming these standard ONNX quantized ops are genuinely unimplemented on
this real toolchain, not merely absent from a possibly-incomplete
docs-scraped list.

**Fails, but with a real, different failure mode neither analysis
predicted**: `ConvTranspose` -- despite being confirmed present in
`AX650_SUPPORTED_OPS` (and passing `partition()`'s coverage check, correctly,
since the op type genuinely is on the list) -- a plain `ConvTranspose`
(kernel 3x3, default strides/padding, upsampling 8x8 -> 10x10) fails during
real quantization with `RuntimeError("Op Execution Error: Y(TargetPlatform.
UNSPECIFIED) - inputs:['X', 'W'], outputs:['Y']")`, not the "dont support"
pattern above. This is exactly the failure mode the static heuristic
structurally cannot see (op-type presence alone says nothing about it) and
is a genuine confirmed gap in `AX650_SUPPORTED_OPS`'s "supported" claim for
at least this shape/parameter combination -- root cause (a missing required
attribute this minimal graph didn't set, a PTQ-engine limitation specific to
transposed conv, or something else) not further diagnosed here.

## Systematic op coverage: from 29% to 99% of `AX650_SUPPORTED_OPS`

The findings above cover Conv/MatMul-family ops specifically. Widening out
to the *entire* 92-op `AX650_SUPPORTED_OPS` list: cross-referencing every
op type actually exercised by a real `pulsar2 build` anywhere in this
investigation (the LLM reconstruction graph, the Conv/MatMul/arith
batteries above, and earlier CNN builds) against the full list started at
27/92 (~29%) confirmed one way or the other. A single-node-per-op battery
(one real `pulsar2 build` per op, small isolated graphs) brought that to
**91/92 (~99%)** -- only `SpatialTransformer` remains genuinely
inconclusive (see below), and every other op in the list now has a real,
confirmed working-or-failing verdict.

**84 confirmed working**, including some genuinely useful discoveries for
future onnxsim work:

- **`RMSNormalization` (native, opset 23) compiles successfully as a single
  op.** `reconstruct_hf_graph()` currently hand-decomposes RMSNorm into
  `ReduceMean`/`Add`/`Sqrt`/`Div`/`Mul` (reused from `gguf_reconstruct.py`,
  written before this op existed in the ONNX opset) -- emitting the native
  op instead would be a smaller, more legible graph, worth a follow-up if
  onnxsim ever bumps its target opset that high.
- **`Silu` compiles even though it isn't a real ONNX operator schema at
  all** -- confirmed by constructing a raw `NodeProto` with `op_type=
  "Silu"` directly (`onnx.checker` has no schema for it and was skipped;
  Pulsar2 doesn't care). It's one of Axera's own extension op names, and a
  real, working one -- though onnxsim should keep emitting the standard
  `Sigmoid`+`Mul` decomposition regardless, for `onnxruntime` compatibility.
- **A real, generalizable gotcha**: `Elu`, `LeakyRelu`, and `TopK` all
  failed on the first attempt with confusing internal errors (`RuntimeError
  ("... convert error: 'alpha'")`, `RuntimeError("... get opr failed")`) --
  not because the ops are unsupported, but because their optional
  attributes (`alpha` for the first two, `largest`/`sorted` for `TopK`)
  were left unset to fall back to the ONNX schema's own documented default.
  Pulsar2's frontend doesn't resolve that default -- it reads the missing
  attribute as `None` and chokes. Setting the exact same default value
  *explicitly* on the node made all three compile without any other
  change. Worth remembering for any ONNX graph -- onnxsim-generated or
  not -- headed for a real `pulsar2 build`: never rely on an attribute's
  schema default being applied for you.

**7 confirmed failing despite being listed in `AX650_SUPPORTED_OPS`**
(beyond `ConvTranspose`, already covered above):

- `Xor`: genuinely unimplemented (`KeyError('dont support Xor opr...')`).
- `Squeeze`: a real internal compiler bug, not a "not supported" error --
  `ZeroDivisionError('division by zero')` inside the NPU backend scheduler
  for the specific shape tested (`(1,4)` squeezed to `(4,)`); other shapes
  may not trigger it.
- `LpNormalization`: fails deep in quantization with an internal exception
  on a U8-quantized intermediate tensor.
- `RotaryEmbedding` (native, opset 23): fails even after applying the
  same "set attributes explicitly" fix confirmed above for `Elu`/
  `LeakyRelu`/`TopK` (`interleaved=0, rotary_embedding_dim=0` explicit) --
  genuinely unimplemented, not an attribute-defaulting issue this time.
  `reconstruct_hf_graph()`'s hand-decomposed RoPE (`Sin`/`Cos`/`Mul`/
  `Concat`/`Neg`/`Slice`) remains the only working path.
- `Swish`: a real ONNX op since opset 24 (distinct from the working
  `HardSwish`/`Silu`), but genuinely unimplemented here (`"Swish, {'alpha':
  1.0} get opr failed"` even with `alpha` explicit).
- `InverseSigmoid`: not a real ONNX operator schema (like `Silu`, likely
  an Axera extension name) -- but unlike `Silu`, fails
  (`"InverseSigmoid, pyrun failed"`).

**`SpatialTransformer` -- inconclusive, not simply untested.** Also not a
real ONNX schema. The first attempt (passing `theta` as a graph input
tensor) failed with a very specific, informative internal error naming
six *scalar attributes* it expected instead: `theta_1_1` through
`theta_2_3` (a 2x3 affine matrix baked into the node as six named floats,
not a runtime tensor -- confirming this op is designed for a compile-time-
constant spatial transform). Rebuilding with those six attributes set gets
past that error, but then fails differently (`IndexError('list index out
of range')`) -- a second real, distinct internal issue, not chased further
here since this is an exotic, rarely-relevant op for this project's
CNN/LLM focus.

## Files

| file | purpose |
| --- | --- |
| `pulsar2_ops.py` | the heuristics and confirmed data: `AX650_SUPPORTED_OPS`/`AX650_MIN_OPSET` (the real, docs-scraped AX650 op list), `CPU_ONLY_OPS` (generic cross-vendor guess), the confirmed `AXERA_NPU_OP_TYPE = "neu mode"` marker, `referenced_const_data_keys()`/`missing_npu_data()`/`has_out_of_band_npu_data()` (the corruption detector), and non-standard-`domain` detection as a fallback for vendor blobs that don't follow Axera's exact convention. |
| `pulsar2_backend.py` | thin wrapper around `pulsar2_ops.py`: `coverage()`, `new_blocking_op_types()`, `stripped_npu_data()`, `unsafe_for_simplify()`, `ax650_build_risks()`. Shaped like the sibling `*_backend.py` modules for interface symmetry (`PULSAR2_AVAILABLE` is always `True` -- there's no external dependency to be missing). |
| `inspect_axmodel.py` | standalone CLI for a **real** `.axmodel` file: loads it with `onnx.load()`, then reports non-standard-domain nodes, op types outside the model's declared opset, and suspiciously large raw attributes -- what originally found the `neu mode` node in the real YOLOv8 file. |
| `models.py` | the shared `scripts/common/synthetic_models.py` suite plus `axera_npu_compiled_leaf` (real CNN `neu mode` node shape) and `axera_llm_layer_leaf` (real per-layer LLM shape: two `neu mode` nodes sharing one initializer) -- no real device needed to exercise the corruption check in CI. |
| `pulsar2_quantizer.py` | `quantize_like_pulsar2()`: a thin wrapper over `onnxsim.quantize_static(method="minmax")`, which already matches Pulsar2's real numeric convention (U8 asymmetric activations, S8 per-channel weights, MinMax calibration). `PULSAR2_QUANTIZER_AVAILABLE` reflects both `onnxruntime`'s availability and `onnxsim` itself actually being importable (a checkout with `onnxsim`'s compiled extension not yet built fails `import onnxsim`, not just the lazy `onnxruntime` import inside it -- both are caught the same way so this degrades gracefully instead of taking `pulsar2_simulator.py`'s `import` down with it). |
| `pulsar2_simulator.py` | `partition()`/`coverage()` (real `AX650_SUPPORTED_OPS` membership, no dependency beyond `onnx`) and `simulate()` (fp32-vs-INT8 estimate via `pulsar2_quantizer.py` + onnxruntime's CPU EP). Validated against real hardware -- see above. |
| `worker.py` | runs the check for one model in an isolated subprocess, printing one `__RESULT__<json>` line. |
| `run_pulsar2_compat.py` | drives the suite, writes a CSV, and exits non-zero on any regression. No `--require-*` flag or `skipped` status -- unlike the EP harnesses, this needs no vendor package or device, so it always runs. Entry point for `axera-integration.yml`'s `pulsar2-compat` job (stock runner, no Docker/device). |
| `screen_onnxmodelzoo.py` | fast, static, Docker/device-free screening of `onnxmodelzoo` models via `pulsar2_simulator`/`pulsar2_backend.ax650_build_risks()` -- run this first. |
| `pulsar2_docker.py` | real `pulsar2 build` (Docker) + `axcl_run_model` (device) wrapper: `build()` (with `profile=` for `trace.json`), `llm_build()` (the separate, ONNX-free `pulsar2 llm_build` LLM path -- see above), `build_from_hf_checkpoint()` (the hf-config+safetensors -> `onnxsim.reconstruct_hf_graph()` -> `build()` path -- see "An alternative LLM path" above), `run_on_device()`, `run_on_device_with_input()`, `force_rmtree()`. Manual/local-only -- needs a loaded Docker image. |
| `convert_onnxmodelzoo.py` | batch driver over `pulsar2_docker.py`: fetch -> onnxsim -> real `pulsar2 build` (orig + simplified, `--profile` optional) -> optional on-device bit-exact diff -> CSV. Entry point for `axera-integration.yml`'s `pulsar2-docker-convert` job -- like `amd-integration.yml`'s MIGraphX check, that job is `workflow_dispatch`-only and targets a `[self-hosted, axcl]` runner this repository doesn't provision, so it's dormant until one exists. |
| `demo_hf_llm.py` | interactive one-shot demo of `build_from_hf_checkpoint()`: compile, print the phase-timing breakdown (and `--profile`'s trace.json/Netron paths), feed one prompt, print top-5 predicted next tokens. |
| `demo_hf_llm_chat.py` | interactive chat REPL on top of the same compiled `.axmodel`, generating one token at a time and reporting real tokens/sec -- see its own docstring for the confirmed ~700ms-per-step `axcl_run_model` process/model-reload overhead this measures alongside the NPU's actual ~0.6-0.8ms compute latency. |

## Running locally

No extra install beyond onnxsim itself:

```bash
pip install .   # or install an onnxsim wheel

python scripts/axera/run_pulsar2_compat.py --output pulsar2-compat.csv
```

To inspect a real compiled model (and check it for the corruption risk
above before considering running it through onnxsim):

```bash
python scripts/axera/inspect_axmodel.py path/to/compiled.axmodel
```

The in-tree smoke test `tests/test_pulsar2_compat.py` reuses this harness and
needs nothing beyond onnxsim's normal test dependencies (it isn't
skip-guarded like the EP compat tests, since there's no external dependency
to be missing). `tests/test_pulsar2_simulator.py` covers the simulator +
quantizer; its `partition()`/`coverage()` tests are likewise unguarded, but
`simulate()`/`quantize_like_pulsar2()` need `onnxruntime` and skip without it.
`tests/test_axera_conv_matmul_coverage.py` is the Conv/MatMul-variant
heuristic analysis above -- also unguarded, needing only `onnx`.

To get a fast partition/coverage read or a quantization-noise estimate for a
model, with no Docker or device:

```python
import onnx
from pulsar2_simulator import coverage, simulate  # scripts/axera/

model = onnx.load("model.onnx")
print(coverage(model))          # "full" / "partial" / "none"
print(simulate(model)["close"]) # fp32 vs. simulated-INT8, roughly sane?
```

## Keeping the AX650N reachable: host driver fixes

Everything above that touches the real device goes through Axera's out-of-tree
AXCL host PCIe driver (`axclhost` 2.25.0, DKMS-built: `ax_pcie_host_dev`,
`ax_pcie_msg`, `ax_pcie_mmb`, `axcl_host`). On this host the card is attached
over **Thunderbolt/USB4** (ASMedia 246x bridge -> PCIe bus 03,
`[1f4b:0650]`), and that link drops on its own from time to time.

Six real kernel bugs in that driver were found and fixed, plus one feature
added; see [`host-driver-patches/NOTES.md`](host-driver-patches/NOTES.md) for
the full symptom -> evidence -> cause -> fix -> verification writeup, the
unified diffs, and the apply scripts.

1. **`ax_mmb` 4MB contiguous `kmalloc`** -- `axcl-smi` sprayed
   `page allocation failure: order:10` WARNs with full backtraces. The
   scatterlist allocator starts at `SZ_4M` (order-10, far above
   `PAGE_ALLOC_COSTLY_ORDER`) and only halves down on failure, so on a
   fragmented host every call dumped a stack trace before succeeding. Fixed
   with `__GFP_NORETRY | __GFP_NOWARN` -- the existing halving loop already
   degrades gracefully.
2. **`axcl_pcie_port_manage` NULL deref / unvalidated ioctl input** -- a real
   kdump-captured Oops (`axcl_pcie_ioctl+0x842`, `CR2=0`, `Comm: axcl-smi`).
   `target` comes straight from a `copy_from_user`'d ioctl argument and indexes
   `port_handle[AXERA_MAX_MAP_DEV][MAX_MSG_PORTS]` with no bounds check, then
   dereferences the slot without a NULL check -- and the slot is NULL until the
   device handshake completes.
3. **Heartbeat thread reads unmapped MMIO after hot-unplug** -- this was the
   one that reset the whole machine, with *no* Oops and *no* vmcore.
   `heartbeat_recv_thread()` caches `axdev->shm_base_virt` (BAR-mapped shared
   memory) once before its loop, and its poll helper reads through that pointer
   once a second for up to 50s. When the link drops, `ax_pcie_dev_remove()`
   `pci_iounmap()`s the BARs and `kfree()`s the `axera_dev`, so the thread keeps
   reading a torn-down ioremap window -- an unrecoverable bus fault, not
   something the kernel can trap and log. Nothing told `axcl_host` about the
   removal at all: its `port_handle[]` and per-target state are populated once
   at `module_init`. Fixed by adding a hotplug notifier
   (`ax_pcie_register_hotplug_notify()`) that fires *before* teardown, dropping
   every cached pointer then; re-resolving the device each loop iteration; and
   an offline flag the poll loop checks so a thread already inside it bails
   within ~1s.
4. **Automatic bring-up on reconnect** -- after (3) a reconnected device is safe
   but unusable until `axcl_host` is reloaded, since per-device bring-up only
   ran at `module_init`. Added `axcl_pcie_device_online()` (firmware load ->
   port creation -> RC/EP handshake -> timestamp sync -> heartbeat thread) on an
   ordered workqueue, because that sequence pushes ~150MB of firmware and can
   block ~120s in the handshake, so it must not run in the PCI `.probe()`
   callback.

5. **`ax_mmb` hands the card raw physical addresses** -- with the IOMMU on
   (a Thunderbolt device is untrusted and always gets a translated domain,
   even under `iommu=pt`), the card's DMA to those addresses faulted
   (`AMD-Vi IO_PAGE_FAULT` storms) and the runtime never came up; this is why
   `amd_iommu=off` had been on the kernel command line. The coherent buffers
   were allocated on the module's own misc device (no IOMMU domain) and the
   scatterlist used `kmalloc` + `virt_to_phys`. Fixed by allocating and
   mapping every card-visible buffer against the card's own `pci_dev`
   through the DMA API, with `mmap` using the CPU-physical page (or
   `dma_mmap_coherent`) instead of the card-visible address. The host now
   runs the card with the IOMMU on and zero faults; with the IOMMU off the
   DMA API is the identity mapping the old code assumed.

6. **Bring-up torn down under itself** -- the "unplugged again during
   bring-up" race fix 4 left open became a real kdump-captured host panic
   (`axcl_firmware_load` writing into a BAR window that `ax_pcie_dev_remove()`
   had just unmapped, one millisecond after the offline notifier ran). The
   bring-up now marks itself busy, checks the offline flag before every
   firmware chunk and inside every completion poll, and the offline path
   waits (bounded) for it to bail before teardown.

7. **The target id must be allowed to differ from the PCI bus number** --
   the driver accepts a heartbeat only when the id the card reports equals
   `pdev->bus->number`. The card reports a fixed 3, which matches this host
   only because the card enumerates on bus 3; in a VM (bus 7) every heartbeat
   and port ack was discarded and a healthy card was declared dead after 50s.
   This was the "device-side handshake timeout" listed below as unexplained
   and assumed to be the card's own agent failing to answer -- it was
   answering all along. A `slot_index_force` module parameter pins the id;
   the default keeps the old bus-number behaviour.

One thing this does **not** fix, still open:

- **The Thunderbolt link itself still drops** (`tbtacl` failure + `boltd` probe
  timeout accompanied one disconnect nobody physically triggered). Fixes 3/4
  make that survivable, not rare.

⚠️ Any `axclhost` package upgrade or `dkms remove` replaces
`/usr/src/axcl-2.25.0` and silently drops all four fixes -- including the one
standing between a Thunderbolt hiccup and a hard reset, and the one that lets
the card work with the IOMMU on. Re-apply from
`host-driver-patches/patches/` (`sudo patch -p1 -d /usr/src/axcl-2.25.0 <
patches/<file>.patch` for each, then `dkms build`/`install`).

## Extending

- If the real device/toolchain becomes available again: automate the manual
  `pulsar2 build` + `axcl_run_model -i/-o/-l` (bit-identical output diff)
  flow used for the `resnet18d`/`googlenet-6` conversions above into a real
  `scripts/axera/pulsar2_docker.py` backend, so `worker.py` can do actual
  compiles instead of only the static `ax650_build_risks()` prediction. The
  input/output folder layout for on-device numeric verification is
  `<dir>/0/<name>.bin` + a `list.txt` containing `0` -- see this README's
  git history / session notes for the exact commands used.
- `AX650_SUPPORTED_OPS` only covers AX650; the same docs site has op lists
  for AX620E/AX615/M57/AX637 (`appendix/op_support_list_<chip>.html`) if
  support for those chips is ever needed.
- The real fix belongs in onnxsim itself (or its vendored onnx-optimizer
  fork): some way to mark an initializer as "referenced, don't touch" beyond
  "is a declared node input" -- e.g. recognizing the custom-op placeholder
  schema `model_prep.cpp` already registers for nodes like `neu mode` and
  treating *all* of a model's initializers as roots whenever any such node is
  present, rather than only the ones it happens to declare as inputs.
- `models.py`'s shared suite is intentionally small and self-contained so the
  CI job needs no downloads; a real `.onnx` (pre-`pulsar2 build`) model can be
  layered on by passing an on-disk path as `worker.py`'s second argument, the
  same way `scripts/qualcomm` and `scripts/regression` do.
