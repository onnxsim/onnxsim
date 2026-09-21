# AX650 static Slice emitter

`scripts/axera/memory_emit.py` adds a first narrow emitter for a data-movement
operator: a compiled ONNX `Slice` with float input `[1, 8]`, axis 1. For step 1
it supports lengths 3 and 4 with starts 0 through 4. For steps 2 through 4 it
supports only starts 0 or 1 with end 8; the input shape stays fixed.

## What the compiler stores

Pulsar2 7.0-lite length-four builds covered starts 0 through 4, including an
exact-config rebuild to measure nondeterminism. Their `*_neu` MCode streams
stayed byte-identical outside the known compiler-noise region at offsets
301–325. Length-three compiler builds store the expected offsets and output
shape too, but have additional MCode changes at offsets 988, 1172, and 1584.
The emitter instead patches the known length-four template's parameters and
shape metadata for length-three output; all five starts were verified on the
NPU. The operation-specific fields are:

- `npu_params` is five repeated little-endian `uint64` values, each equal to
  `start * 4` for this float32 input.
- The `neu mode` node's `outputs_info` and the ONNX graph output shape contain
  the target slice shape.
- `npu_dyn_params` is empty.

The emitter checks the reference's one-node graph shape, table layout, and
normalized MCode against a real compiled fixture before changing these fields.
It rejects a different input shape or malformed template. The emitted model
keeps the reference MCode unchanged and updates only the offset table and
output-shape metadata.

## NPU check

I emitted all five length-three intervals from the committed `[2:6]` fixture
and ran them inside `axcl-vm` on the AX8850. With input `[[0,1,2,3,4,5,6,7]]`,
they returned the exact expected slices for `[0:3]`, `[1:4]`, `[2:5]`, `[3:6]`,
and `[4:7]`. Length-four variants `[1:5]` and `[4:8]` were also checked on
device. The source was built for AX650 by Pulsar2 7.0-lite; the emitted models
ran with AX8850 V3.6.5 firmware.

This establishes coverage only for this fixed input rank/shape, axis, step,
lengths, and float32 offset encoding. It does not claim general Slice support,
arbitrary memory-op generation, or an MCode interpreter. Regression coverage lives in
`tests/test_axera_memory_emit.py`; the raw compiled reference is
`scripts/axera/fixtures/slice_1x8_axis1_step1_len4.axmodel.gz`.

## Step-two Slice

Pulsar2 builds of `[0:8:2]` and `[1:8:2]` produced the same MCode, with the
same five-word offset table rule (`start * 4`). The step-two program differs
from step one at several instruction fields, so it has a separate compiled
reference: `scripts/axera/fixtures/slice_1x8_axis1_step2_len4.axmodel.gz`.
The emitter uses that template for either start and retains its MCode.

Both emitted variants ran on the AX8850 in `axcl-vm` with input
`[[0,1,2,3,4,5,6,7]]`; `[0:8:2]` returned `[[0,2,4,6]]` and `[1:8:2]`
returned `[[1,3,5,7]]`. The implementation rejects other starts and ends for
step two.

## Step-three Slice

Pulsar2 builds of `[0:8:3]` and `[1:8:3]` also share an MCode template for
the same output shape `[1,3]`. Their `npu_params` values are five copies of
`start * 4`. Step-three MCode is structurally different from the step-one
length-three build, so the emitter uses its own fixture at
`scripts/axera/fixtures/slice_1x8_axis1_step3_len3.axmodel.gz`.

Both emitted variants ran on the AX8850 in `axcl-vm` with input
`[[0,1,2,3,4,5,6,7]]`; `[0:8:3]` returned `[[0,3,6]]` and `[1:8:3]`
returned `[[1,4,7]]`. The implementation rejects other starts and ends for
step three.

## Step-four Slice

Pulsar2 builds of `[0:8:4]` and `[1:8:4]` share another MCode template for
output shape `[1,2]`. The `npu_params` table again stores five copies of
`start * 4`; the fixture is
`scripts/axera/fixtures/slice_1x8_axis1_step4_len2.axmodel.gz`.

Both emitted variants ran on the AX8850 in `axcl-vm` with input
`[[0,1,2,3,4,5,6,7]]`; `[0:8:4]` returned `[[0,4]]` and `[1:8:4]` returned
`[[1,5]]`. The implementation rejects other starts and ends for step four.

## Static Gather index retargeting

The same module now has `emit_gather_axmodel(reference_path, output_path,
indices=...)` for the measured float32 graph `Gather(x[1,8], axis=1)` with
four outputs. It accepts four indices in `[0,7]`, including duplicates.
Pulsar2 stores these indices as the first four little-endian uint32 words in a
56-byte `npu_params` table and zero-fills the remaining ten words. The emitter
checks the compiled graph, table, output metadata, empty dynamic-parameter
table, and normalized MCode template before changing the four index words.

The reference was built for AX650A with Pulsar2 7.0-lite. All five generated
variants ran inside `axcl-vm` on AX8850 V3.6.5 firmware using
input `[-0.8,-0.6,-0.4,-0.2,0.1,0.3,0.5,0.7]`. The even-index output had
maximum absolute error 0.0029 against ONNX values; the odd-index output had
maximum error 0.0032. This is consistent with the model's int8
quantization. Inputs must remain within the calibration range: earlier runs
with `[0,1,2,3,4,5,6,7]` saturated high values near 0.901 and made the Gather
output appear incorrect.

The compiler and NPU checks covered `[0,2,4,6]`, `[1,3,5,7]`, `[0,3,5,7]`,
descending `[7,6,5,4]`, and duplicate `[7,7,0,0]`. Each compiled `npu_params`
table contained those exact first four uint32 words; MCode changes relative to
the even-index build stayed inside offsets 301–325. The largest NPU error
across the five patterns was 0.0032.

This establishes index retargeting only for this graph shape and dtype. It
does not support other axes or output lengths.
