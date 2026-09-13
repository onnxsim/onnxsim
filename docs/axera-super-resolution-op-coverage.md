# Op coverage for training super-resolution models on the AX650N: a survey

Companion to `docs/axera-on-device-training-handoff.md` (resnet18/50) and
`docs/axera-audio-speech-op-coverage.md` (Whisper/wav2vec2/LSTM/GRU), same
question for a new domain: for a real single-image super-resolution (SISR)
architecture, which ops does today's pipeline
(`onnxsim.graph_grad.build_backward` + `scripts/axera/legalize.py`'s
`TRAINING_RULES`) already reach, and which have a real, specific gap?
Cross-referenced against the same two tables every other survey in this
project uses: `onnxsim.graph_grad._RULES`/`_PYTHON_ONLY_RULES`/
`_MULTI_OUTPUT_RULES` (what `build_backward` can differentiate) and
`scripts/axera/pulsar2_ops.py`'s `AX650_SUPPORTED_OPS` (what the NPU can run
at all).

## A real architecture: EDSR

`pip install super-image` (pulls in `torch`, `torchvision`,
`opencv-python`, `h5py`, `huggingface-hub`; export only, no dataset or
pretrained weights needed) ships a real library of published SISR
architectures behind a HuggingFace-`transformers`-style config/model API --
EDSR, CARN, MSRN, PAN, RCAN, DRLN, HAN, and others, each its own
`<Name>Config`/`<Name>Model` pair. **EDSR** ("Enhanced Deep Residual
Networks for Single Image Super-Resolution", Lim et al. 2017) is the
natural first target: one of the field's foundational architectures, and
its own code (`super_image.models.edsr.modeling_edsr`) is about as simple
as a real SISR network gets -- a head `Conv`, `n_resblocks` residual blocks
(each two `Conv`s + `ReLU`, plain `res = body(x); res += x`), and a tail
`Upsampler` (a `Conv` widening channels by `scale**2`, then
`nn.PixelShuffle(scale)` -- the sub-pixel convolution every architecture in
this library's own `Upsampler`/`MeanShift` shared utility uses for
upsampling) before a final color-channel `Conv`.

Exported at a tiny config (`scale=2`, `n_resblocks=2`, `n_feats=8`, a
16x16 input) via `scripts/axera/build_edsr_train_step.py`'s
`export_edsr()`, real (random-initialized, not pretrained) weights:

| op types present | in `_RULES`/`_PYTHON_ONLY_RULES`? |
| --- | --- |
| `Conv`, `Add`, `Relu`, `Mul`, `Constant` | yes (`Constant` folds away entirely at `build_resident_step()`'s own constant-folding step, since `res_scale=1.0`'s `Mul` collapses to a no-op) |
| `DepthToSpace` | **no** (until this survey) |

**Every op except one is already covered on both axes.** `Conv`/`Add`/
`Relu`/`Mul` all have `graph_grad` backward rules and are NPU-executable
(confirmed directly, the same two-table lookup every other survey here
does) -- residual-block training was never in question. `DepthToSpace`
(`nn.PixelShuffle`'s exact ONNX form, `mode="CRD"` -- what
`torch.onnx.export` always emits for it) was the one real gap: **already
NPU-executable** (unlike `GRU`'s own stricter gap), just missing a backward
rule.

## Fixed: `onnxsim.graph_grad._grad_depth_to_space`

`DepthToSpace` is ONNX's own spec-documented `reshape -> transpose ->
reshape`, nothing else -- a pure element permutation, no learned or
run-time-dependent component. Its adjoint is therefore the same chain run
backward: reshape the incoming gradient into the *post-transpose* shape,
transpose by the *inverse* permutation, reshape to the input's own shape --
`Reshape`'s and `Transpose`'s own adjoints, both already in `BACKWARD_OPS`
and already registered rules (`_grad_reshape`/`_grad_transpose`), just
inlined here rather than called since the intermediate tensor never
otherwise exists as a named node in the graph.

**Verified directly against real onnxruntime execution, not derived from
the spec text alone** -- the same discipline this project's LSTM/GRU work
established after an initial `GRU` `linear_before_reset` assumption came
out backwards (off by up to `0.48`) on a first attempt. A dot-product test
(`sum(g * y)`'s gradient via central-difference finite differences, `eps=
1e-3`) against a real, `onnxruntime`-executed native `DepthToSpace` node
agreed to `5.7e-4` -- within that tolerance, and confirmed for both
`mode="CRD"` (what `nn.PixelShuffle` exports) and `mode="DCR"`
(TensorFlow's own convention, covered too since a wrong mode branch would
fail silently, not loudly). Registered in a new `_PYTHON_ONLY_RULES` entry
(no C++/WASM mirror yet, the same "real rule, missing port" situation
`Concat`/`Where`/`IsNaN` are already in that table for) and exercised as
two of `tests/test_graph_grad.py`'s own finite-difference cases
(`depth_to_space_crd`/`depth_to_space_dcr`).

## Host-verified: a real EDSR resident training step

`scripts/axera/build_edsr_train_step.py` builds the full pipeline every
other model in this project uses (`add_mse_loss_nchw` -- the rank-4
`[N,C,H,W]` counterpart of `build_resident_train_step.add_mse_loss`'s
rank-2 and `build_whisper_train_step.add_loss_3d`'s rank-3 versions, same
explicit-`axes` `ReduceMean` guard against the AX650's own bare-`ReduceMean`
vendor bug -- then `build_resident_train_step.build_resident_step()`
unchanged), at three trainable scopes:

- `tail` (4 tensors: the upsampler's widening `Conv` + the final
  color-channel `Conv`) -- the most direct exercise of
  `_grad_depth_to_space`, since a gradient must pass through
  `DepthToSpace` immediately to reach the widening `Conv`'s weight.
- `head` (2 tensors: the very first `Conv`) -- the harder scope, requiring
  a gradient through `DepthToSpace`, *and* every residual block, to reach
  it.
- `all` (16 tensors: every trainable weight in the network).

All three build and pass `onnx.checker.check_model` cleanly (68/91/232
nodes respectively) -- `build_backward` reaches every target tensor in
every scope, `head` included, confirming the fix works through the whole
network's depth, not just locally around `DepthToSpace` itself. Not yet
run on real hardware -- see "What's next" below.

**One real naming gotcha, worth recording generically**: `EdsrModel`'s own
input is naturally named `"lr"` (**l**ow-**r**esolution) in an
export -- colliding silently with this pipeline's own reserved `"lr"`
(**l**earning **r**ate) scalar that `build_resident_step()` always adds.
`onnxsim.simplify()`'s SSA check is what actually catches the collision
(`'lr' has been used as graph input names multiple times`), well
downstream of the export call itself, not at export time. Renamed to
`"lowres"` in `build_edsr_train_step.py`; worth checking for by name in any
future domain whose own natural input name happens to collide with this
pipeline's small reserved vocabulary (`lr`, `grad_seed`, `lowres`'s own
`hr` counterpart is fine since nothing else claims it yet).

## What's next

Not done in this survey (a static/host-only pass, following every other
survey's own first-pass scope in this project): a real `pulsar2 build`
compile and a real AX650N training run, the natural next step matching
`unroll_lstm`/`unroll_gru`'s own real-hardware follow-up
(`docs/axera-audio-speech-op-coverage.md`'s "Both closed" section) --
calibrate carefully (this project's calibration-degeneracy bug class has
bitten every domain surveyed so far at least once), compile, and confirm a
real, gradient-driven, non-frozen loss decrease (the `lr=0`-frozen-loss
control `unroll_lstm`'s own hardware verification established as the
standard proof of genuine training, not a lucky-looking flat line).

Beyond EDSR itself: `super-image`'s other architectures (CARN, MSRN, PAN,
RCAN, DRLN, HAN, ...) share the same `Upsampler`/`MeanShift` utility module
EDSR does, so the same `DepthToSpace` coverage almost certainly transfers
directly -- untried here, but a much smaller lift than EDSR's own first
survey was, since the one real gap this domain has is now closed.
