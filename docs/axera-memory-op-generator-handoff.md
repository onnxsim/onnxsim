# AX650 memory-op generator handoff

## Current state

The first narrow memory-op emitter is static float32 `Slice` support for an
input shaped `[1, 8]`, axis 1, unit step. Its entry point is
`emit_slice_axmodel(reference_path, output_path, start=..., end=...)` in
`scripts/axera/memory_emit.py`. It accepts the measured intervals `[0:4]`,
`[1:5]`, `[2:6]`, `[3:7]`, `[4:8]`, and `[2:5]`; it checks the reference
model structure and normalized MCode before updating the parameter table and
output-shape metadata. Details and hardware results are in
[`axera-memory-op-generator.md`](axera-memory-op-generator.md).

The model was built with Pulsar2 7.0-lite and run inside the LXD VM `axcl-vm`
on AX8850 V3.6.5 firmware. Two emitted variants returned the expected values
for input `[[0,1,2,3,4,5,6,7]]`. Unit coverage is in
`tests/test_axera_memory_emit.py` (11 cases passed).

Code and tests are on branch `codex/axera-slice-memory-emitter`, in commit
`7a4d6ff2`. PR [#1723](https://github.com/onnxsim/onnxsim/pull/1723) is open
with auto-merge enabled; check its CI status before building on this work.
The working tree also has unrelated changes under `third_party/onnx` and two
untracked npm lockfiles under `tools/onnx-finetune/wasm/`; leave those alone.

## Gather probe: evidence and limitation

Scratch builds are under `/home/takecheeze/npu-scratch/t_codegen_gather`
(`even`, `even_r1`, `odd`). They use float32 input `[1,8]`, axis 1, and
indices `[0,2,4,6]` or `[1,3,5,7]`. Each MCode stream is 2600 bytes. The
seven-uint64 `npu_params` table stores the four indices in its first four
entries, followed by three zeros. An exact rebuild of `even` had differences
at offsets 311 and 319; `even` versus `odd` differed at 301, 303, 317, and
319. These are all within the same compiler-noise window seen in the Slice
probe, so this comparison has not identified index-dependent MCode fields.

The device check is **not a pass**. Running these compiled models on
`axcl-vm` with `x = [[0,1,2,3,4,5,6,7]]` returned `[0.0, 0.9009736,
0.9009736, 0.9009736]` for `even` and `[0.9009736, 0.9009736, 0.9009736,
0.9009736]` for `odd`, rather than the expected selected values. Do not infer
from the table alone that it is safe to patch Gather models. Establish why the
baseline compiled Gather is already producing these outputs before attempting
an emitter.

## Next steps

1. Inspect the Gather ONNX graph, compiled graph metadata, and model I/O names,
   shapes, and dtypes. Confirm the build config's input preprocessing and
   quantization settings are appropriate for raw float input.
2. Run the original compiled `even` model with calibration/build settings that
   make an unmodified compiler output match the reference, or reduce the probe
   to a verified setup. Compare against ONNX Runtime as well as the expected
   values. Keep the current failing outputs as a known diagnostic.
3. Rebuild `even` and `odd` under controlled settings, compare `npu_params`,
   `outputs_info`, and MCode with repeated identical builds, and identify which
   field carries the Gather indices. Confirm on the NPU before implementing a
   narrowly scoped emitter and its fixture/tests.
4. After Gather is understood, use the same build/compare/device loop to pick
   the next memory operator. Keep each emitter restricted to a measured graph
   family and reject unsupported shapes/layouts rather than guessing encodings.

## Reproduction environment

- Pulsar2 7.0-lite Docker image for compiling probe graphs.
- LXD VM `axcl-vm` for actual device runs; wrapper calls can use
  `AXCL_LXD_VM=axcl-vm` with `scripts/axera/pulsar2_docker.py`.
- Slice checks: `uv run --offline --no-project --with onnx --with pytest
  python -m pytest -q tests/test_axera_memory_emit.py`.
