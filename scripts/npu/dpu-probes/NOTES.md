# Intel NPU 5020: DPU descriptor notes

Research notes from compile-only probes (OpenVINO 2026.4, `compile_tool -d NPU`). No NPU run is needed for any result
below. Nothing here is a descriptor generator: the per-layer route in `scripts/npu/resnet50/dpu_layers.py` lets OpenVINO
compile each layer, and that is still the way we run convs on the DPU.

Scope: convs with 1x1, 3x3 and 5x5 kernels, cin and cout 4 to 64, input 16x16 (plus a few 28x28 and 56x56 for the
earlier template work). Each field below is marked with the evidence behind it.

## Relocations

ELF `rela.*` sections name the words that the loader patches. These are the only words whose values are known
without decoding:

- DMA weight transfer: source is `buffer.Constant.0.constant`, with the weight offset as the addend.
- DMA input reads `io.NetworkInput.0`; the output DMA writes `io.NetworkOutput.0`.
- DMA records chain to the next record in their queue (addend = record start).
- DPU invariant and variant sections are referenced by the DMA records.
- `buffer.DDR.0` is referenced twice per queue but has size 0 in every single-conv and chained blob we compiled.

An intermediate activation in a chained graph is not a relocation. Its on-chip address is therefore a literal word.

## Decoded (checked on several configurations)

Invariant = `task.dpu.invariant.0.0`, variant = `task.dpu.variant.0.0`. Word indices are 32-bit words within the section.

| Field | Law | Evidence |
|---|---|---|
| inv 10 / var 1 | cin / 4 in bits 16+; input height in the low byte | cin sweep |
| inv 116 / var 99 | ((cout - 1) << 13) \| (hout - 1) | cout sweep (5 points); pad sweep (6 points) |
| var 45 | 256 * cin | cin sweep (4 points) |
| inv 28 / var 43 | cin - 1 (low bits) | cin sweep (4 points) |
| inv 47, inv 105 to 108 | 0x18000 + 0x200 * cin (1x1, cout 16) | cin sweep (4 points) |
| var 103 | 0x18000 + 0x400 * cin (1x1, cout 16) | cin sweep (4 points); changes at 3x3 pad 2 |
| var 101 | 16 * k^2 | kernel sweep (k = 1, 3, 5) |
| inv 97 (low 12 bits) | 0x220 * k | kernel sweep (k = 1, 3, 5) |
| var 102 | max(16, cout) | cout sweep (5 points); hypothesis for cout below 16 |
| var 58 | pad mask: 0x924 << (min(pad, k // 2) - 1) for pad >= 1; 0x10 for pad 0 | pad sweep (6 points) |

DMA size fields, from a separate check on 4 to 6 configurations: DMA 0.0 word 286 holds the input bytes; DMA 0.1 word
6 the output bytes; DMA 1.0 word 6 the weight bytes, equal to 2 * cout * R + 256 * ceil(cout / 16) with
R = roundup(cin * k^2, 16).

## Post-ops

- Activation clamp, fp32 in the invariant section: word 128 is the upper bound and word 129 the lower bound. Relu,
  Clip(0, 6) and Clip(-1, 2) were checked (3 configurations). Without a clip, word 128 is +FLT_MAX and word 129 is
  -FLT_MAX.
- Invariant word 133 changes with the clip: -0.0 for Relu, 1.0 for both Clip variants. Not decoded.
- A Relu placed on the input changes invariant word 9 by 0x400000 and several post-op constants. Not decoded.

## Multi-op structure

- Single conv: 2 DPU entries (the 704-byte invariant section is two 352-byte entries) and 8 DMA descriptors.
- Conv, Relu, Conv: 3 DPU entries, 9 DMA descriptors. The intermediate stays in CMX; no DMA moves it.
- Conv with a residual Add: 5 DPU entries, 11 DMA descriptors. The block input is read from the network input twice.
- Full ResNet50 (compiler stats): 660 tasks, 219 NCE, 392 NNDMA, 8 spills from CMX to DDR and 6 reads back.

## Address words: not decoded

Rejected hypotheses, all tested with compile-only probes:

- Linear dependence of a literal word on the size of a side output that is a network output: the held-out check failed.
- Linear dependence on the size of an internal side branch that is added into the main output (sweep C in
  {8, 16, 32, 64}, held out at C = 40): most words that fit a line on the sweep miss the held-out point by thousands of
  bytes. The two words that passed were a float constant and a zero.
- Node order: listing the side branch before the main chain gives identical descriptors. The compiler normalizes order.

Literal words in the 0x4020xxxx range do change between graphs, but not in any predictable way we have found.

## Open items

- Address words (on-chip addresses of activations and DMA endpoints).
- Invariant 133 (clip-related mode word) and invariant words 96 and 97 (high bits).
- Anomalies at 3x3 with pad 2: variant 103, invariant 135, variant 45 and invariant 112 change there and nowhere else
  we tested.
- Variant 61 and invariant 112 change only at cout 64.

## Probe scripts

The scripts in this directory are compile-only. They expect the following on the test machine: OpenVINO's `compile_tool`
at `~/openvino_git/bin/intel64/Release/compile_tool`, the level-zero loader at `~/ze-loader/lib`, and the helper modules
`reloc.py` and `litdiff.py` in the home directory. Output goes under `~/npu-dpu/roadmap/`. `reloc.py` lists relocations;
`litdiff.py` diffs literal words between two blobs. The sweeps (`cin_sweep.py`, `cout_sweep.py`, `k_sweep.py`,
`pad_sweep.py`) each print the words that change along one parameter.
