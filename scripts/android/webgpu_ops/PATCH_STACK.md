# ORT WebGPU patch stack (ORT `125ea21`)

The `ort_*.patch` files in this directory apply, in this order, to ONNX Runtime commit `125ea21` (`git apply`, one after the other; each was checked
to apply on the previous state and the final state was compared with the tested working tree):

| # | patch | what | default |
|---|---|---|---|
| 1 | `ort_transpose_qualcomm_unpadded_tile` | Adreno tiled-Transpose miscompile fix | exact, on |
| 2 | `ort_webgpu_missing_ops` | WebGPU kernels for ops that fell back to the CPU | exact, on |
| 3 | `ort_conv_silu_fusion` | Conv + SiLU/QuickGelu epilogue | exact, on |
| 4 | `ort_conv_gelu_fusion` | Conv + Gelu (erf / tanh) epilogue | exact, on |
| 5 | `ort_webgpu_extra_texture` | Program API: one extra 2-D texture binding | infrastructure |
| 6 | `ort_f16_f32_accumulate` | f32 accumulation for fp16 MatMul/Conv2dMM (`ORT_WEBGPU_F16_ACC32`) | opt-in |
| 7 | `ort_conv_winograd` | Winograd F(2,3) 3x3 convs (+ optional f16 / texture weights) | exact, on |
| 8 | `ort_conv_add_fusion` | Conv + residual Add (+ReLU) -> `NhwcFusedConv` | exact, on |
| 9 | `ort_conv_texdirect` | texture-weight direct conv (`ORT_WEBGPU_CONV_TEXDIRECT`) | opt-in |
| 10 | `ort_concat_split_vec4` | vec4 last-axis Concat / Split | exact, on |
| 11 | `ort_conv_depthwise_vec4` | vec4 depthwise conv | exact, on |
| 12 | `ort_conv_texdirect_f32w` | exact RGBA32F weights for large K, MAXC 1024 | opt-in |
| 13 | `ort_concat_conv` | fused Concat -> 1x1 conv op (`ORT_WEBGPU_CONCAT_CONV`) | opt-in |

Experiments (not part of the stack): `ort_conv2d_regtile_experiment` (register-tile Conv2dMM, slower in the network; replaces nothing in the stack),
`ort_conv2d_splitk_experiment`, `ort_tile_env_override_experiment`.

Results and the measurements behind every patch are in `../WEBGPU_SURVEY.md`.
