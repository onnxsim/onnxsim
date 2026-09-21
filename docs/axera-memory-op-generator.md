# AX650 static Slice emitter

`scripts/axera/memory_emit.py` adds a first narrow emitter for a data-movement
operator: a compiled ONNX `Slice` with float input `[1, 8]`, axis 1, and step 1.
The measured end-exclusive intervals are `[0:4]`, `[1:5]`, `[2:6]`, `[3:7]`,
`[4:8]`, and `[2:5]`; the input shape stays fixed.

## What the compiler stores

Pulsar2 7.0-lite builds covered starts 0 through 4 and output lengths 3 and
4, including an exact-config rebuild to measure nondeterminism. The `*_neu`
MCode stream stayed byte-identical outside the known
compiler-noise region at offsets 301–325. The operation-specific fields were
elsewhere:

- `npu_params` is five repeated little-endian `uint64` values, each equal to
  `start * 4` for this float32 input.
- The `neu mode` node's `outputs_info` and the ONNX graph output shape contain
  `[1, end - start]`.
- `npu_dyn_params` is empty.

The emitter checks the reference's one-node graph shape, table layout, and
normalized MCode against a real compiled fixture before changing these fields.
It rejects a different input shape or malformed template. The emitted model
keeps the reference MCode unchanged and updates only the offset table and
output-shape metadata.

## NPU check

I emitted two variants from the committed `[2:6]` fixture and ran them inside
the `axcl-vm` on the AX8850. With input `[[0,1,2,3,4,5,6,7]]`, generated
`[1:5]` returned `[[1,2,3,4]]`, and generated `[2:5]` returned `[[2,3,4]]`.
Both matched ONNX Slice semantics. The source was built for AX650 by Pulsar2
7.0-lite; the emitted models ran with AX8850 V3.6.5 firmware.

This establishes coverage only for this fixed input rank/shape, axis, step,
and float32 offset encoding. It does not claim general Slice support, arbitrary
memory-op generation, or an MCode interpreter. Regression coverage lives in
`tests/test_axera_memory_emit.py`; the raw compiled reference is
`scripts/axera/fixtures/slice_1x8_axis1_step1_len4.axmodel.gz`.
