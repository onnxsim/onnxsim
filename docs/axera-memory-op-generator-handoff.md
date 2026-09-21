# AX650 memory-op generator handoff

## Current state

The first narrow memory-op emitter is static float32 `Slice` support for an
input shaped `[1, 8]`, axis 1, with measured steps 1 through 4. Its entry point
is
`emit_slice_axmodel(reference_path, output_path, start=..., end=..., step=...)`
in `scripts/axera/memory_emit.py`. Step 1 accepts length-three and length-four
intervals with starts 0 through 4. Steps 2, 3, and 4 each accept starts 0 or
1 with end 8, using separate compiled fixtures. It checks the reference model
structure and normalized MCode before updating the parameter table and
output-shape metadata. Compiler-built step-one length-three MCode has three
additional shape-specific byte changes, but the retained length-four MCode
was verified on device for every length-three start. Details and hardware
results are in
[`axera-memory-op-generator.md`](axera-memory-op-generator.md).

The model was built with Pulsar2 7.0-lite and run inside the LXD VM `axcl-vm`
on AX8850 V3.6.5 firmware. Five step-one length-three variants, two step-one
length-four variants, and both step-two, step-three, and step-four variants
returned the expected values for input
`[[0,1,2,3,4,5,6,7]]`. Unit coverage is in
`tests/test_axera_memory_emit.py`.

Code and tests are on branch `codex/axera-slice-memory-emitter`. PR
[#1723](https://github.com/onnxsim/onnxsim/pull/1723) is open with auto-merge
enabled; check its CI status before building on this work.
The working tree also has unrelated changes under `third_party/onnx` and two
untracked npm lockfiles under `tools/onnx-finetune/wasm/`; leave those alone.

## Gather emitter: evidence and scope

Scratch builds are under `/home/takecheeze/npu-scratch/t_codegen_gather`
(`even`, `even_r1`, `odd`). They use float32 input `[1,8]`, axis 1, and
indices such as `[0,2,4,6]` and `[1,3,5,7]`. The seven-uint64 `npu_params`
table stores the four indices in its first four uint32 words, then ten zeros.
The AX650A compiler builds also verified irregular `[0,3,5,7]`, descending
`[7,6,5,4]`, and duplicate `[7,7,0,0]` indices; MCode differences stayed in
the 301–325 compiler-noise window. The emitter is `emit_gather_axmodel(...)` in
`scripts/axera/memory_emit.py`; it supports four indices in `[0,7]` and keeps
the MCode unchanged.

All five emitted variants were run on `axcl-vm` with input
`[[-0.8,-0.6,-0.4,-0.2,0.1,0.3,0.5,0.7]]`. They matched expected Gather
values with maximum absolute error at most 0.0032. The earlier apparent
failure came from using `[0,1,2,3,4,5,6,7]`, outside the calibration range
(approximately `[-0.967,0.901]`); the input quantizer saturated large values
at about 0.901. The NPU check is valid only for in-range inputs and this
compiled quantization configuration.

A length-two Gather (`indices=[0,3]`, output `[1,2]`) was compiled separately
in `/home/takecheeze/npu-scratch/t_codegen_gather_len2`. Its parameter table
shrinks to 48 bytes (12 uint32 words), but its 2600-byte MCode differs from
the length-four template across 916 bytes. A length-three build in
`/home/takecheeze/npu-scratch/t_codegen_gather_len3` uses 52 bytes (13 uint32
words) and a 2664-byte MCode, also substantially different from length four.
No NPU runs were made for these output lengths, so the emitter remains
restricted to four outputs; shorter outputs need their own compiled template
and device check.

## Next steps

1. Add Gather output lengths or axes only with separate compiled templates
   where MCode changes, plus device checks. Keep runtime values inside the
   calibration range when comparing against ONNX.
2. Probe another memory operator from a compiler-supported graph with a
   consumer if possible; standalone terminal Reshape/Squeeze can fail in the
   scheduler. Compare exact rebuilds, then run emitted models on the NPU.
3. Keep each emitter restricted to a measured graph family and reject
   unsupported shapes/layouts rather than guessing encodings.

## Reproduction environment

- Pulsar2 7.0-lite Docker image for compiling probe graphs.
- LXD VM `axcl-vm` for actual device runs; wrapper calls can use
  `AXCL_LXD_VM=axcl-vm` with `scripts/axera/pulsar2_docker.py`.
- Slice checks: `uv run --offline --no-project --with onnx --with pytest
  python -m pytest -q tests/test_axera_memory_emit.py`.
