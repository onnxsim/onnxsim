# MinMax calibration without Pulsar2 (AX650)

`docs/axera-graph-stitch.md` builds a fused graph's compiled model from standalone
per-op programs, but still reads the fused scales and zero points from Pulsar2's
`quant_axmodel.json`. This page removes that dependency: the quantization parameters are
computed from the float ONNX graph and the calibration samples.

- Code: `scripts/axera/pulsar_free_calibration.py` (`calibrate`, `stitch_calibration`,
  `calibrate_for_stitch`).
- Tests (no device): `tests/test_axera_pulsar_free_calibration.py`.
- Fixtures: `scripts/axera/fixtures/pulsar_free_calibration/` (129 KB). `index.json`
  holds, per case, the oracle (the resolved `values` / `tensor_configs` entries of the
  native build's `quant_axmodel.json`: scale, zero point, signedness per tensor) and
  names the calibration sample files (`samples_*.npy.gz`, 16 samples per input, shared
  between cases that were built from the same samples).

```python
import pulsar_free_calibration as pfc
import graph_stitch as gs

quants = pfc.calibrate(float_model, {"x": samples})        # every tensor
scales, zero_points, signed = pfc.stitch_calibration(quants, tensor_names)
model = gs.stitch_model(wiring, scales, zero_points, signed)
```

## Rules

All builds are Pulsar2 7.0-lite for AX650 with `calibration_method: MinMax`. The rules
are a fit to their `quant_axmodel.json`, not read from Pulsar2's code.

| rule | applies to | type | scale | zero point |
| --- | --- | --- | --- | --- |
| `UNSIGNED` | graph inputs and node outputs | u8 | `float32((hi - lo) / 255)`, range widened to include 0 | `rint(-lo / s)` |
| `SIGNED_ACT` | both operands of a MatMul of two activations (a Softmax output included) | s8 | `float32(max\|x\|) / 127.5` | 0 |
| `CONST_U8` | an initializer read by Add, Mul or Div | u8 | the `UNSIGNED` formula over the constant's own values | same |
| `WEIGHT_S8_PC` | an initializer that is operand 1 of a MatMul | s8, per output channel | `float32(max\|w[:, c]\|) / 127.5` | 0 |

Two more facts make the numbers come out:

- **Baked forward pass.** The calibration forward pass runs on the fake-quantized
  constants and weights. An activation downstream of a constant gets its range from the
  quantized values. With the float weights the output scale of a 64x64 linear layer is
  hundreds to thousands of float32 ulps off.
- **Weight codes divide in float32.** `code = clip(rint(w / s_c), -128, 127)` with the
  division in float32. This is the quantizer of `emitter.codes_of`, which is byte-exact
  against the compiled weight tables. A float64 division rounds 7 to 18 of the 4096
  codes of a 64x64 layer the other way, and the output scale then misses by up to about
  3,000 ulps. The scratch version of this module divided in float64; the port corrects it.

Neg needs no rule: its output's own range gives `s_y = s_x` and `zp_y = 255 - zp_x`.

## Accuracy against Pulsar2

Zero points and signedness matched on every tensor compared. Scales:

| builds | tensors | bit-exact | 1 ulp | 2 ulps |
| --- | --- | --- | --- | --- |
| 73 builds: standalone ops at `[1,64]` and `[1,576]` (three calibrations), fused SiLU / RMSNorm / attention, the graph_stitch components | 238 | 225 | 13 | 0 |
| 25 single-op builds at other widths (100 to 2048) and the RMSNorm components at 576 | 58 | 52 | 5 | 1 |
| 13 linear layers (six 64x64 and five 576x576 weight sets, the 576 layer at two calibrations): output scale | 13 | 7 | 5 | 1 |

The tensors that are off are outputs of MatMul, Softmax, Sqrt, Div, Mul or ReduceMean.
The forward pass here is numpy float32 and does not round every operation the way
Pulsar2's evaluator does. A MatMul by a constant weight is summed in float64 and rounded
(float32 BLAS depends on the machine and was up to 5 ulps off); no summation order tried
reproduces all 13 linear builds.

The committed oracles cover a unary op (Sigmoid), a two-input op (Mul), Softmax at two
calibrations, SiLU and RMSNorm at two calibrations, the attention block at `pm1` and the
six 64x64 linear layers. The tests assert every zero point and signedness, bit-exact
scales where the comparison was bit-exact and at most one ulp on the nine tensors
recorded as one ulp off (`scale_ulps` in the index).

## Calibrate, then stitch

With `calibrate_for_stitch` in place of the quant json, the stitched model is:

- **the native fused build** (same comparison as `docs/axera-graph-stitch.md`) when
  every scale is bit-exact: SiLU, `sqrt_mul`, `sig_add` and `mul_sig` at both
  calibrations (8 of the 12 committed graphs; SiLU is checked by the tests);
- **the native build except float lanes** when a scale is one ulp off: RMSNorm at both
  calibrations and attention at both. In the three cases the tests cover (RMSNorm at
  both calibrations, attention at `pm1`) exactly 16 lane records (`0x0f50..0x1040`)
  differ, by at most two float32 ulps; attention's MatMul constant in `npu_params`
  (`s_a * s_b / s_y` as float32 lanes) differs by one ulp. No zero point, table,
  address or record count changes.

## Device results

AX8850, AXCL V3.6.5, 2026-10-10. Model built by calibrate + stitch (no quant json)
versus the native fused build:

| graph | result |
| --- | --- |
| `rmsnorm` `pm1` | outputs bit-identical |
| `attention` `pm4` | outputs bit-identical |
| `rmsnorm` `pm4` | identical output codes; output floats differ by at most 2.4e-7 (one ulp of `s_y`) |
| `attention` `pm1` | identical output codes; output floats differ by at most 2.4e-7 (one ulp of `s_y`) |

## Limits

- **A fit.** The rules reproduce these builds. Another calibration method, per-channel
  activations, or ops outside `SUPPORTED_OPS` (Sigmoid, Mul, Add, Div, Sqrt, Neg, MatMul,
  ReduceMean, Softmax) are not covered. Concat is left out on purpose: Pulsar2 leaves a
  standalone Concat float.
- **Refused** (`NotImplementedError`): an unsupported op; a MatMul whose first operand is
  an initializer; an initializer read by anything but Add, Mul, Div or MatMul.
  `ValueError`: missing samples, unequal sample counts, a sample of the wrong size.
- **Not bit-exact everywhere.** About 8% of the scales are one ulp off (one Softmax and
  one linear output two). The float evaluator is numpy's; another numpy build may round
  `exp` or a BLAS sum differently and move a scale by an ulp.
- **Small graphs only.** Shapes are `[1,64]`, `[1,576]`, `[8,64]`; 16 samples per input.
