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
