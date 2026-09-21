# `graph_grad` op index

`onnxsim/graph_grad.py` (and its C++ mirror, `onnxsim/graph_grad.cpp`) has
grown too large to read top to bottom the way, say, tinygrad's own autodiff
(`tinygrad/function.py`, forward and backward colocated per primitive op in
one small file) can be. This is the compensating index: every op the module
differentiates, its VJP in one line, and the rule's name in each language --
grep for that name rather than trusting a line number here, since line
numbers rot the moment a rule above it grows or shrinks and nothing enforces
them, while a `def`/function name is exactly what changed the least across
the onnxscript-templating migration (see "Templated rules" below).

`tests/test_graph_grad_op_index.py` checks the "ONNX op" column against
`graph_grad.supported_ops()` on every test run, so this table can go stale in
its *content* (a formula going out of sync with the code) but not in its
*coverage* (an op present in one and missing from the other) without a test
failing.

## Templated rules

A rule marked **templated** is authored once in `onnxscript`
(`scripts/codegen/generate_grad_templates.py`), compiled to a checked-in ONNX
`FunctionProto` (`onnxsim/graph_grad_templates_gen.{py,h}`), and instantiated
identically from both languages via a call node + inlining, instead of being
hand-transcribed twice -- see `graph_grad.py`'s own "Templated rules" section
comment for the full rationale (the dvar-derivation bug it exists to
prevent) and for exactly why `Sub` and `Relu` are the two rules that look
templatable but deliberately are not.

## `_RULES` -- single-output, single-`g` rules

| ONNX op | VJP | Templated? | Python rule | C++ rule |
|---|---|:-:|---|---|
| `Add` | `da = db = g`, broadcast-reduced | Y | `_grad_add_templated` (ref: `_grad_add`) | `GradAddTemplated` (ref: `GradAdd`) |
| `AveragePool` | col2im of `g`, scaled by each window's `1/count` | | `_grad_averagepool` | `GradAveragePool` |
| `BatchNormalization` | `xhat`/`inv`-based; the rule whose hand-written `dvar` factor motivated the whole templating effort | Y | `_grad_batch_normalization_templated` (ref: `_grad_batch_normalization`) | `GradBatchNormalizationTemplated` (ref: `GradBatchNormalization`) |
| `Clip` | `g` where `lo < x < hi` (strict), `0` elsewhere | | `_grad_clip` | `GradClip` |
| `Conv` | `dX`/`dW` as im2col/col2im `MatMul`s (never a `Conv`/`ConvTranspose` of its own), `dB = sum(g)` | | `_grad_conv` | `GradConv` |
| `Div` | `da = g/b`, `db = -g·y/b` (reuses the forward quotient `y`) | Y | `_grad_div_templated` (ref: `_grad_div`) | `GradDivTemplated` (ref: `GradDiv`) |
| `Erf` | `dx = g · 2/√π · exp(-x²)` -- exists for GELU | Y | `_grad_erf_templated` (ref: `_grad_erf`) | `GradErfTemplated` (ref: `GradErf`) |
| `Exp` | `dx = g·y` (reuses the forward output `y`) | Y | `_grad_exp_templated` (ref: `_grad_exp`) | `GradExpTemplated` (ref: `GradExp`) |
| `Gather` | scatter-add into `data` via a one-hot `MatMul`; `indices` gets no gradient | | `_grad_gather` | `GradGather` |
| `Gemm` | `dA`/`dB`/`dC` from `MatMul`+`Transpose`, honoring `alpha`/`beta`/`transA`/`transB` | | `_grad_gemm` | `GradGemm` |
| `Identity` | alias -- `g` itself, no node emitted | | `_grad_identity` | `GradIdentity` |
| `InstanceNormalization` | `LayerNormalization`'s shape, reduced over spatial axes only | | `_grad_instance_normalization` | `GradInstanceNormalization` |
| `LayerNormalization` | `xhat`-based `dx`/`dscale`/`dbias` | | `_grad_layer_normalization` | `GradLayerNormalization` |
| `Log` | `dx = g/x` (singular at `x = 0`, genuinely) | Y | `_grad_log_templated` (ref: `_grad_log`) | `GradLogTemplated` (ref: `GradLog`) |
| `MatMul` | `dA = g @ Bᵀ`, `dB = Aᵀ @ g` | | `_grad_matmul` | `GradMatMul` |
| `MaxPool` | col2im, credit split among tied maxima | | `_grad_maxpool` | `GradMaxPool` |
| `Mul` | `da = g·b`, `db = g·a` | Y | `_grad_mul_templated` (ref: `_grad_mul`) | `GradMulTemplated` (ref: `GradMul`) |
| `Neg` | `dx = -g` | Y | `_grad_neg_templated` (ref: `_grad_neg`) | `GradNegTemplated` (ref: `GradNeg`) |
| `ReduceMean` | `g` broadcast back, scaled by `1/N` | | `_grad_reduce` | `GradReduce` |
| `ReduceSum` | `g` broadcast back, unscaled | | `_grad_reduce` | `GradReduce` |
| `Relu` | `dx = g · (x > 0)` -- **not** templated; see below | | `_grad_relu` | `GradRelu` |
| `Reshape` | `g` reshaped back to the input's own shape | | `_grad_reshape` | `GradReshape` |
| `Sigmoid` | `dx = g·y·(1-y)` (reuses `y`) | Y | `_grad_sigmoid_templated` (ref: `_grad_sigmoid`) | `GradSigmoidTemplated` (ref: `GradSigmoid`) |
| `Softmax` | `dx = y·(g - Σ(g·y))` along the softmax axis (reuses `y`) | | `_grad_softmax` | `GradSoftmax` |
| `Sqrt` | `dx = 0.5·g/y` (reuses `y`; singular at `x = 0`) | Y | `_grad_sqrt_templated` (ref: `_grad_sqrt`) | `GradSqrtTemplated` (ref: `GradSqrt`) |
| `Sub` | `da = g`, `db = -g` -- **not** templated; see below | | `_grad_sub` | `GradSub` |
| `Tanh` | `dx = g·(1-y²)` (reuses `y`) | Y | `_grad_tanh_templated` (ref: `_grad_tanh`) | `GradTanhTemplated` (ref: `GradTanh`) |
| `Transpose` | `g` transposed by the inverse permutation | | `_grad_transpose` | `GradTranspose` |

**Why `Sub` isn't templated**: its only arithmetic is a single `Neg`, applied
*after* `reduce_to` (not before) so it runs on the smaller, already-reduced
tensor -- an ordering a template called before the reduction would lose.

**Why `Relu` isn't templated**: its mask `Cast`'s target dtype has to stay a
visible, mutable node in the *raw*, not-yet-inlined backward slice for
`onnxsim.compile_training._cast_backward_to_fp16` to retarget from FLOAT to
FLOAT16 before the surrounding fp16 arithmetic is emitted. A templated
`GradRelu` would hide that `Cast` inside an uninlined `onnxsim.grad`-domain
call until inlining happens much later (`MakeStepGraph`/`make_step_graph`),
by which point that retargeting pass has already run and moved on --
producing an invalid FLOAT16/FLOAT32-mixed graph. This is exactly what broke
CI the first time `Relu` was templated; see the git history around
`generate_grad_templates.py`'s own comment beside where a `GradRelu`
template used to live.

## `_MULTI_OUTPUT_RULES` -- every output may carry its own incoming gradient

| ONNX op | VJP | Python rule | C++ rule |
|---|---|---|---|
| `Dropout` | identity for inference mode (omitted or constant-false `training_mode`); no gradient for optional mask, ratio, or mode inputs | `_grad_inference_dropout` | -- (Python only) |
| `Split` | one `MatMul` per output against a constant 0/1 selection matrix (not a `Concat` of the incoming gradients -- `Concat` isn't in `BACKWARD_OPS`) | `_grad_split` | `GradSplit` |

`Dropout` with a true or runtime `training_mode` remains unsupported. Its VJP
would need the sampled dropout mask and training scale; treating that case as
an identity would be incorrect.

## `_PYTHON_ONLY_RULES` -- no C++/WASM mirror yet

Each of these has exactly one output and could move into `_RULES` outright
once ported to C++; the only reason they live here is the missing mirror
(see that table's own comment in `graph_grad.py`).

| ONNX op | VJP | Python rule |
|---|---|---|
| `Concat` | one `Gather` per input, pulling that input's own contiguous slice of `g` back out along the concat axis (`Split`'s adjoint, reached the other way) | `_grad_concat` |
| `DequantizeLinear` | straight-through: `dx = g` (paired with `QuantizeLinear` below) | `_grad_dequantize_linear` |
| `DepthToSpace` | reshape-transpose-reshape, the exact inverse of the op's own decomposition -- for `nn.PixelShuffle` exports | `_grad_depth_to_space` |
| `IsNaN` | no gradient (boolean output) | `_grad_is_nan` |
| `LeakyRelu` | `g` on `x > 0`, otherwise `alpha * g`; chooses `alpha` at `x = 0` | `_grad_leaky_relu` |
| `Pad` | crop `g` back to the unpadded input with constant-index `Gather`s; constant mode and static nonnegative pads only | `_grad_pad` |
| `PRelu` | `dX = g` for `x >= 0`, `slope*g` for `x < 0`; `dSlope = sum(g*x)` over negative inputs, reduced to the scalar slope shape | `_grad_prelu_scalar` |
| `QuantizeLinear` | straight-through: `dx = g`, no gradient for `scale`/`zero_point` (fake-quantization convention) | `_grad_quantize_linear` |
| `Slice` | embed `g` into the input positions selected by static `starts`/`ends`/`axes`/positive `steps`, using constant selection `MatMul`s | `_grad_slice` |
| `Softplus` | `dx = g / (1 + exp(-x))`, the sigmoid derivative written with existing backward ops | `_grad_softplus` |
| `Squeeze` | `g` reshaped back to `data`'s own (pre-squeeze) shape | `_grad_squeeze_or_unsqueeze` |
| `Unsqueeze` | same as `Squeeze`, the same reshape either direction | `_grad_squeeze_or_unsqueeze` |
| `Where` | `g` masked by `Cast(cond)` and routed to whichever of the two data operands was selected; `cond` gets no gradient | `_grad_where` |

`PRelu` currently accepts only a static slope shape containing one element
(including a rank-0 scalar). Vector and per-channel slopes remain unsupported.

## Genuinely unsupported: control flow

`If`/`Loop`/`Scan` are refused (`_CONTROL_FLOW_OPS` in `graph_grad.py`,
`IsControlFlowOp` in `graph_grad.cpp`) and always will be -- this module
differentiates a static, single-pass ONNX graph with no tape, so there is no
mechanism for routing gradient through whichever branch or iteration
actually ran. In practice this is rarely the genuine runtime branching it
sounds like: an `If` a tracer (PyTorch's, say) inserted for something
statically resolvable (a shape-derived condition, an export-time flag) is
already eliminated by `onnxsim.onnx_simplifier.simplify()`'s own
`eliminate_if_with_const_cond` pass -- part of its default pass set,
specifically paired with constant folding for this -- before `graph_grad`
ever needs to see it. The refusal message says as much when the unsupported
op is one of these three.
