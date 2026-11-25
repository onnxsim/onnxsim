# Intel NPU 5020 DPU descriptors: research summary

What we have established about the DPU task descriptors of an OpenVINO-compiled blob (OpenVINO 2026.4, `compile_tool -d NPU`,
test machine Intel NPU 5020). Every claim is marked as device-confirmed (we ran a blob with an edit and checked the output),
compile-only (we compared compiled blobs), or rejected (a hypothesis we tested and dropped). Word indices are 32-bit words
within the section: `inv` = `task.dpu.invariant.0.0`, `var` = `task.dpu.variant.0.0`, `dmaX.Y` = `task.dma.X.Y`.

## Device-confirmed

| Field | What we established | How |
|---|---|---|
| Output read address | `dma0.1` word 10 is the output read address (bytes) | Shifting it shifts the output by exactly delta/2 elements |
| Input DMA destination | `dma0.0` word 292 is where the input lands | Prediction (input shifted by delta/2) matched at four offsets |
| Input read partner | `inv` word 17 (`base_offset_a`) moves with word 292 | Moving word 17 alone changes the output; moving word 17 and word 292 together leaves it unchanged, at three deltas |
| Weight DMA destination | `dma1.0` word 12 must agree with the device's read addresses | Moving it alone changes the output |
| Per-channel weight record | Each output channel has a 16-byte record at constant-buffer byte `320 + 256*(o//16) + 16*(o%16)` (1x1 and 3x3, one and two groups). Word 0 of the record is that channel's weight address relative to CMX base 0x40200000 | Adding 0x40 (two 1x1 rows) to channel c's word makes channel c compute with row c+2 and the same bias; matched to 0.002-0.007 |
| Weight layout | Slot for (o, i, tap) = `160 + 128*G + (o//16)*16*R + (o%16)*R + tap*cin + i` (stream index, minus addend/2), G = ceil(cout/16), R = roundup(cin*k*k, 16) | Single-element edits for 1x1 and 3x3, cout up to 96; each edit changes only its output channel, by the predicted amount |
| Bias | fp32 per output channel at byte `332 + 256*(o//16) + 16*(o%16)` of the constant buffer | Editing one channel's bias shifts only that channel, by exactly the edit |
| Post-scale | `inv` word 132 (1.0 when unused) | Setting it to 2.0 doubles the output (error 0.0000) |
| Clamp | `inv` word 128 = upper bound, word 129 = lower bound (fp32) | Upper 0.5 and lower -0.5 each give the clamp of the conv output (errors 0.0002 and 0.0005); lower 0 is Relu |
| Negative slope | `inv` word 133 = slope on the negative side (1.0 for Clip, -0.0 for Relu) | Editing to 0.5 and 2.0 scales negative outputs before the clip (four samples per value) |
| Lane mask (partial) | `inv` word 96 bit 20 removes exactly input channel 7 | Clearing bit 20 gives relative error 0.0006 against the removed-channel prediction |
| Weight source | The weight DMA source is the relocation addend (320). Shifting it by S bytes shifts each record by S/16 channels and each weight row by S/32 (1x1) or S/160 (3x3) rows; only whole-row shifts give usable results | 1x1 cout 32 at S = 64, 96; 3x3 cout 32 at S = 160, 320; fractional S = 32 does not match |
| DMA sizes | Halving the input or the output transfer size is rejected by the device | Two runs, both device errors |
| Generated conv | A 3x3 conv (cin 16, cout 32) with new weights, new biases and Relu, written by the layout rules into a compiled template | Maximum error 0.0075 against numpy, with and without Relu |

## Generator coverage (device, ResNet50 conv shapes)

A generator writes a conv from a compiled template of the same shape: new weights in the record layout, new fp32 bias records,
and Relu through the lower clamp. The template's placement, read bases and DMA settings are kept from the compile.

- **Single launch** (`gen_all_shapes.py`): 8 of 23 unique ResNet50 conv shapes pass (relative error about 3e-4). The 15
  failures are all over the single-launch limits we measured (about 147k weights per launch, about 400k input or output
  elements). Examples: the 3x3 with 512 channels (about 2.4M weights), the 1x1 with 256 by 64 (output about 800k elements).
  The failures are inferred from the sizes, not checked one by one.
- **Chunked** (`gen_chunked_device.py`, sub-launches sized by `chunked2.pick`): 23 of 23 pass, relative error 2e-4 to 4e-4.
  The launch count per shape runs from 1 to 64.
- Bias-record note: a template compiled with zero bias has no bias records to patch. The generator uses a nonzero template bias.

Generator debugging (`debug_gen.py`): with the template self-patched, the output matches numpy (error 5e-4); with new weights
and new bias, all four 16-channel groups match (error 1e-3) once the template's bias is nonzero.

## Compile-only (not checked on the device)

- Size formulas, all compiled: input bytes = cin*hin^2*2 (`dma0.0` 286); output bytes = cout*hout^2*2 (`dma0.1` 6); weight bytes = 2*cout*R + 256*ceil(cout/16) (`dma1.0` 6). Matched in 80 of 80 configs.
- Channel and kernel laws: `var` 45 = 256*cin; `inv` 28 and `var` 43 = cin - 1 (low bits); `inv` 116 and `var` 99 = ((cout-1) << 13) | (hout-1); `var` 102 = max(16, cout); `var` 101 = 16*k^2; `inv` 97 low 12 bits = 0x220*k; `var` 58 = pad mask 0x924 << (min(pad, k//2) - 1) for pad >= 1.
- Placement: the weight block sits at output address minus weight bytes in 27 of 32 layouts. The input is at 0x40218000 for these layouts. The grid's 32 configs all execute correctly on the device, including the five the rule does not predict.
- Multi-op: a DPU invariant per layer and a variant per tile; a conv followed by Relu gets the clamp in the same task; Sigmoid after a conv is folded into the conv task (variant grows by one entry).

## Rejected

- Stride field: `var` 58 bits 20-23 do not change the stride (device).
- Pooling address: `var` 48 moved by 0x1000 changes nothing in a conv + MaxPool (device, one configuration).
- Bias-enable: `var`/`inv` 126 set to 2 does not switch the bias off (device).
- `inv` 112 bit 18: no effect on a plain conv (device).
- `inv` 9 as the input read base: moving it alone is rejected by the device; co-edits fail.
- Stream-shift model for the weight destination (word 12): no shift of the weight stream explains the output (device).
- Overlap explanation for word 12: moving it corrupts almost the whole output, including the negative offsets that cannot overlap.
- Node order: listing the side branch first gives identical descriptors (compile-only).
- `inv` 18-20, 92 as the input read base: co-edits with word 292 do not restore the output (device).

## Open

- Mode bits: `inv` 96 bits outside the lane mask (cin 1 and cin 10 and up give different words; the cin 32 modes are not identified); `inv` 97 high bits.
- Lane mask low bits (13 and 16) do not remove a single channel.
- Pad-2 anomaly words (`var` 103, `inv` 135, `var` 45, `inv` 112 at 3x3 pad 2): not a function of the pad excess alone.
- Placement exceptions (five 8x8 configs where the input moves) and the in-place case.
- Multi-op blobs: on-chip allocation, barrier placement, the rule for folding an op into a conv, and the 832-byte SHAVE section that appears for fp32 Relu, LeakyRelu and Clip (identical bytes for Relu and LeakyRelu; not decoded).
- Pooling and stride fields inside a conv descriptor.

## Scripts in this directory

- `reloc.py`: lists the ELF relocations of a compiled blob (what the loader patches).
- `litdiff.py`: diffs the literal descriptor words of two blobs, skipping relocated words.
- `verify_record.py`, `record_predict.py`: per-channel weight record checks.
- `wsrc_model.py`, `wsrc_3x3.py`: weight source shift checks.
- `confirm_base_offset_a.py`: input read partner check.
- `gen_conv_device.py`: generated-conv end-to-end check.
- `verify_dma_sizes.py`, `verify_dma_device.py`: DMA size checks.
- `grid_device.py`: placement grid, device run.

All scripts assume the test machine layout: OpenVINO's `compile_tool` under `~/openvino_git/bin/intel64/Release`, the level-zero loader under `~/ze-loader/lib`, and `reloc.py` in the home directory. The device runs were done one at a time, because the NPU is shared.
