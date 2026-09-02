"""Post-training weight pruning for MatMul/vanilla-Gemm and Conv layers.

Surveying the pruning literature against what onnxsim can actually act on
(an exported ONNX graph, no training loop, no gradients, usually no labels)
narrows the field a lot. Most well-known pruning *tools* --
``torch.nn.utils.prune``, NNI's pruning API, Neural Magic's SparseML, Intel
Neural Compressor's pruning API -- assume a live framework model mid-training
or at least a fine-tuning loop to recover accuracy after each pruning step
(iterative magnitude pruning / the Lottery Ticket Hypothesis, movement
pruning, "pattern lock" pruning, ...). That is the same reason onnxsim's
existing weight-only quantization stack (:mod:`onnxsim.gptq`,
:mod:`onnxsim.awq`, ...) reimplements each technique's *algorithm* against
raw ONNX MatMul/Gemm weights rather than depending on those libraries
directly: they operate one level up, on a model object onnxsim never has.

*Structured* pruning (removing whole channels/filters, e.g. Torch-Pruning,
NNI's L1/L2 filter pruning, network slimming, or the expert-intermediate-
channel/Mamba-state pruning inside NVIDIA's "Iterative Puzzle" compression
pipeline for hybrid MoE LLMs, https://arxiv.org/abs/2607.04371) is a
fundamentally bigger project than the rest of this module for two separate
reasons, and this module only takes on one of them in general -- with one
narrow, precisely-scoped exception carved out of that paper's own two named
techniques: :func:`apply_moe_expert_channel_pruning` (see its own "MoE
expert-intermediate-channel pruning" section comment far below) reaches the
"expert-intermediate-channel" half, for the one real, bounded ONNX shape
that turns out to make it tractable (``com.microsoft::MoE``'s own
``[num_experts, ...]``-leading-dimension weight tensors, needing no
cross-node dependency-graph walk at all); the paper's other named
technique, Mamba-state pruning, remains fully out of scope -- checked
empirically against this environment's own onnxruntime contrib-op schema
registry, there is no fused Mamba/SSM op with fixed weight tensors this
module could target the way it targets `MoE`'s own, and an arbitrarily
decomposed selective-scan subgraph has no single bounded node pattern to
recognize, the same bar every other declined topology below is held to.
Every *other* case below -- ordinary MatMul/Conv channel pruning and
everything this paragraph goes on to describe -- is the one general problem
this module does take on. It *does* change tensor
shapes, which ripples through every downstream consumer of the pruned
dimension -- real graph surgery, not the self-contained per-layer weight
rewrite every other ``apply_*``/``quantize_*`` pass in onnxsim is. That part
:func:`apply_structured_pruning` takes on, but deliberately only for the
narrowest topology where the surgery is unambiguous: a single MatMul/Gemm
or ordinary (``group=1``) Conv whose output feeds, through a chain of
shape-preserving elementwise ops (activations, and for MatMul/Gemm also a
bias/scale add/mul) with no other consumer anywhere along that chain, into
exactly one downstream layer of the same family whose reduction/input-
channel dimension matches.
Any multi-consumer fan-out or branch (which needs real dependency-graph
analysis -- what Torch-Pruning's DepGraph does in general) is left
untouched rather than guessed at. One narrow, bounded slice of the
residual/skip-connection case *is* handled -- see
:func:`_find_conv_residual_chains`'s own section comment for the full
reasoning -- because it turns out to have a provably-safe special case: a
Conv chain whose forward walk hits a channel-preserving ``Add(a, b)`` with
two non-constant operands (every residual connection's shape) is, rather
than declined outright, treated as a merge point requiring whichever real
Conv producer(s) feed `a` and `b` to be pruned to one shared channel-index
set, found by walking backward from each operand (through the same
unary-activation/depthwise-pass-through hops the forward walk already
allows) to a real ``group=1`` Conv producer -- or, transitively, to
*another* such `Add` merge point, unioning that one's own group in too (the
"many residual blocks share one spine" case). This is still bounded, not
general DepGraph: every hop that walks *toward* a group's own producers
still requires a real Conv/`Add` topology it recognizes, and the two
compositions checked and found unsafe (see :func:`_find_matmul_residual_chains`'s
own section comment for the MatMul/Gemm ones -- a gated combine or a fused
attention op sitting directly on a residual branch, with no projection
MatMul/Conv in between) are still declined outright. What *is* now
reached, and wasn't originally: a real multi-block ResNet or transformer
stage's shared "post-block" tensor, read by *both* the next block's own
first layer *and* directly by that block's own `Add`/`SkipLayerNormalization`
-- once a group's shared channel-index set is established, propagating it
forward to more than one independent, ordinary downstream reader of the
same in-group tensor turns out to need no new tie-break the way resolving
it backward from multiple *producers* would (see
:func:`_find_conv_residual_chains`'s own section comment for the exact
mechanism, :func:`_resolve_conv_fanout_branches`/
:func:`_resolve_matmul_fanout_branches`, and precisely where the remaining
boundary sits: an extra reader that itself forks further, reaches a graph
output, or would need a tie-break between two *conflicting* keep sets on
the same shared weight, still declines the whole group). What this reaches
now: a single residual connection (whether its branches fan out elsewhere
or not), a genuinely linear stack of `Add`-only merges, and a real
*interior* block of a deep residual stage -- essentially the full shape of
a real multi-block ResNet/transformer stage, short of the two compositions
above and a cross-chain conflict on a literally shared weight.

A general grouped Conv (see :func:`_match_conv_producer`/
:func:`_match_conv_consumer`) may also take part in a Conv residual/merge
group, as any producer, the primary consumer, and/or any extra fan-out
branch -- but only when every one of those roles that *is* grouped shares
the exact same `group` count. The reasoning is the same block-partition
argument :func:`_chain_group`'s own docstring works through for an ordinary
(single-producer, single-consumer) chain, just generalized to however many
producers/branches a merge group collects: a grouped Conv's own `group`
output-channel blocks are contiguous ranges of `n_channels / group`
channels, a partition that depends only on `n_channels` and `group` --
never on which particular Conv it is -- so as long as every grouped
participant names the same `group`, one shared per-block top-k (see
:func:`_apply_chains`) simultaneously respects every one of their own
block-uniform-count requirements, exactly the same way today's `keep` set
already respects every ordinary (`group=1`) participant's total lack of one.
Two different non-1 `group` counts anywhere in the same merge group, by
contrast, imply two different block partitions of the same shared
`n_channels` index space with no general way to reconcile them -- so that
case is declined outright (the whole group, not a partial cut), mirroring
:func:`_find_conv_chains`'s own "both sides grouped with a different group
count" decline for the ordinary case.

The exact same construction is repeated for MatMul/Gemm chains -- see
:func:`_find_matmul_residual_chains`'s own section comment for the full
reasoning, which mirrors the Conv case's above closely enough that only the
differences are worth restating here: the backward walk mirrors
:func:`_walk_to_consumer`'s own *wider* MatMul/Gemm hop set (unary
activations plus a per-channel bias/scale ``Add``/``Mul`` against a
constant, not just unary activations) rather than the Conv walk's narrower
one, since MatMul/Gemm has no depthwise-Conv-style transparent pass-through
hop at all. This is exactly the residual-stream shape every current
transformer block takes (``x = x + SelfAttn(LN(x))``, ``x = x + MLP(LN(x))``)
-- previously declined outright, now reached the same way a Conv
projection-shortcut block is. Two compositions were checked; one turned out
to have a provably-safe special case of its own and is now handled, the
other is still not safe to fold in silently and remains declined the same
conservative way as everything else in this pass:

- A gated (SwiGLU/GeGLU) combine feeding a residual branch directly, with no
  output-projection MatMul in between, is now resolved rather than declined:
  the backward walk recognizes both of :func:`_find_gated_chains`'s own
  gated shapes as *another* kind of hop -- a `Mul` of two non-constant
  operands (each operand resolved back to its own real MatMul/Gemm producer
  via :func:`_find_gated_chains`'s own `_trace_gate_producer_backward`) and
  the native fused `SwiGLU` op (opset 28+, each operand required to already
  *be* such a producer's own raw output, reusing that same function's
  `SwiGLU`-branch extraction) alike -- and folds *both* resulting producers
  into the group's shared leaf-producer set, exactly like a gated pair's own
  two producers already are for the non-residual case. Nothing is guessed at
  or dropped: both branches' importance is combined (root-sum-square, the
  same metric a gated pair outside a residual chain already uses) and both
  are pruned to the one shared channel-index set the whole group agrees on
  -- see :func:`_walk_matmul_producer_backward`'s own section comment for
  the composition-safety argument (why the gate/up path's existing
  single-consumer bar and the residual walk's own fan-out/tied-weight
  conflict checks already cover every risk either shape's composition could
  introduce, with no new machinery needed -- `SwiGLU`'s own shape is, if
  anything, a strictly tighter case than `Mul`'s). This is still narrow, not
  general: only exactly `_find_gated_chains`'s own two recognized shapes,
  nothing wider.
- A residual branch that would need to cross a fused self-attention op
  boundary (`Attention`/`GroupQueryAttention`, see the "Attention-head
  pruning" section far below) to reach a real producer, with no
  output-projection MatMul between the attention op and the `Add`, is still
  declined outright -- unlike the gated case, there's no analogous
  "combine every real producer feeding it" fallback available: the op
  itself, not a recognizable elementwise combine of two producers, is what
  sits in the way.

Both are narrow, exporter-dependent shapes rather than the common case --
an FFN's own down-projection or an attention block's own output projection
feeding the residual `Add` (overwhelmingly the normal shape) needs no
special handling at all, since the backward walk stops at that projection's
own MatMul/Gemm node without ever looking further upstream at what feeds
it.

A bare `Add` merge point is, however, the exception rather than the rule
once a transformer has actually been run through onnxruntime's own
transformer-optimizer tool: that pass fuses each residual `Add` (plus an
optional per-channel bias `Add`) together with the *following*
`LayerNorm`/RMSNorm into one ``com.microsoft::SkipLayerNormalization``/
``SkipSimplifiedLayerNormalization`` node, so
:func:`_match_matmul_residual_merge` recognizes that fused node as an
eligible merge point too -- its `input`/`skip` inputs playing `Add`'s own
two-operand role -- while also treating it as a per-channel affine hop
whose `gamma` (required) and, if present, `beta` (``SkipLayerNormalization``
only, dropped by the RMSNorm variant) and `bias` get sliced by the group's
own `keep` set alongside everything else. See
:func:`_find_matmul_residual_chains`'s own section comment for the exact
fused arithmetic (confirmed against onnxruntime's own kernel source and by
direct execution) and how a non-constant/tied `gamma`/`beta`/`bias`, or a
consumed optional `mean`/`inv_std_var` output, is declined the same
conservative way as everything above.

The same optimizer tool typically fuses an FFN block's own bias-add and
activation together too, the same way it fuses a residual `Add` into
`SkipLayerNormalization` above: ``com.microsoft::BiasGelu(A, B) = Gelu(A +
B)`` (erf-based, the common case) and ``com.microsoft::FastGelu(X[, bias])``
(the tanh-approximated Gelu, with `bias` optional) both collapse an ordinary
``MatMul -> Add(bias) -> Gelu`` FFN hop into one node -- confirmed against
onnxruntime's own schema (`contrib_defs.cc`) and CPU kernel (`bias_gelu.cc`)
and by direct execution. Without also recognizing these, an FFN chain this
whole feature exists for (``up = MatMul(x, W1); h = BiasGelu(up, Bias1);
down = MatMul(h, W2)``) would fail at that one hop and the whole chain would
go unpruned, the same gap the `SkipLayerNormalization` fix above closed for
residual connections. :func:`_walk_to_consumer`/
:func:`_walk_matmul_producer_backward` (the forward and backward MatMul/Gemm
hop walkers) both recognize a `BiasGelu`/`FastGelu` node the same way they
already recognize a bias/scale `Add`/`Mul` hop -- see
:func:`_match_fused_bias_gelu` and `_FUSED_BIAS_GELU_OPS`'s own comment --
sliced by the same `keep` set alongside everything else; a non-constant bias
(`BiasGelu`'s own schema requires one; `FastGelu`'s is optional) declines
the node outright, never guessed at. This is a MatMul/Gemm-chain-only hop,
deliberately not extended to Conv chains: a real Conv already carries any
bias in its own third input (see the Conv paragraph below), and neither
fusion targets Conv graphs in practice. `com.microsoft::QuickGelu(X) = X *
Sigmoid(alpha * X)` -- the third Gelu-family fusion the same optimizer tool
emits, used by some model families in place of `BiasGelu`/`FastGelu` -- is a
simpler case still: it takes no bias operand at all (`alpha` is a node
*attribute*, not an input), so it is exactly as unary/shape-preserving as
`Gelu`/`Sigmoid` already in `_UNARY_PASS_THROUGH`, and is matched by simply
being added to that set -- extending every walker that already consults it
(both MatMul/Gemm and Conv, forward and backward alike, plus
:func:`_trace_gate_producer_backward`'s own gated-pair gate-activation
matcher below) for free, with no dedicated hop machinery needed. A gated
(SwiGLU/GeGLU) pair's own gate branch fused into `BiasGelu`/`FastGelu`
specifically (as opposed to plain `Gelu`/`Sigmoid`, or the now-unary
`QuickGelu`) is *not* recognized by :func:`_trace_gate_producer_backward`,
and is left out of scope deliberately rather than extended: that tracer only
ever walks back through single-input unary ops, with nowhere on
:class:`_Producer` to carry a gate-branch-local bias constant the way
`_Chain.chain_ops` already does for the shared post-combine chain, so
supporting it would need new machinery (a `_Producer`-local `chain_ops`
counterpart), not a one-line addition like `QuickGelu`'s. It is also the
narrower case in practice: a gated FFN's gate projection commonly carries no
bias at all (e.g. Llama-family linear layers), and when it does, only a
*separate* `Add`+`Gelu` on that branch is even eligible for onnxruntime's
own fusion pass to collapse into `BiasGelu`/`FastGelu` in the first place --
a plain unfused `Gelu`/`Sigmoid` gate (already handled) is the shape that
survives when there's no bias to fuse to begin with.

A raw, not-yet-optimizer-fused export -- straight out of a training
framework's own ONNX exporter, before onnxruntime's transformer-optimizer
ever runs -- has no `SkipLayerNormalization`/`BiasGelu` fusions at all, just
plain unfused nodes throughout: ``x1 = MatMul(x, W1); x2 =
LayerNormalization(x1, gamma, beta, axis=-1); x3 = MatMul(x2, W2)``. Unlike
the `SkipLayerNormalization` shape above, this `LayerNorm`/RMSNorm sits
*mid-chain* -- between an ordinary producer and consumer, not fused onto a
residual merge point -- so it is recognized directly by
:func:`_walk_to_consumer` as one more hop kind, alongside its existing
bias/scale `Add`/`Mul` and `BiasGelu`/`FastGelu` hops: a plain
`LayerNormalization` (opset 17+)/`RMSNormalization` (opset 23+)/
`SimplifiedLayerNormalization` (onnxruntime's own RMSNorm-equivalent,
confirmed to run under the default ONNX domain -- see
`_NORM_PASS_THROUGH_OPS`'s own comment) node, its own `scale` (required) and,
for `LayerNormalization` only, `bias` (optional; the other two ops have no
such input at all) sliced by the chain's shared `keep` set exactly like a
`SkipLayerNormalization` node's own `gamma`/`beta`/`bias` already are (see
:func:`_norm_pass_through_const_names`, which factors out and reuses
:func:`_flat_channel_const`, the very same per-tensor validity check
:func:`_skip_layer_norm_const_names` uses). Recognized *only* when the node's
own `axis` attribute is confirmed to normalize exactly the one trailing
channel axis being pruned -- `axis == -1` outright, or a positive `axis`
confirmed against a known tensor rank (:func:`_norm_axis_is_last`) -- since
that is also the axis LayerNorm's own mean/variance reduction runs over:
slicing channels *before* that reduction runs (which is exactly what pruning
this producer's own output does) genuinely changes what gets normalized
over, so the reduction on the pruned graph is mathematically a different
(smaller) computation than the reduction on the original graph would have
been -- not a bug, simply what happens whenever channels are removed ahead
of any op that reduces over them, no different in kind from how a channel
pruned ahead of a global-average-pool or another norm changes what that op
reduces over too. The correctness bar this hop is held to is therefore the
same one every other hop in this pass already is: the pruned graph must
match an *independently, already-pruned* reference model (weights, and this
norm's own `scale`/`bias`, sliced to the same kept indices from the start),
not a hypothetical "run the original norm, then slice its output" post-hoc
reference, which is not what pruning ever actually computes once the norm's
own producer has fewer channels. A norm whose secondary, training-only
outputs (`Mean`/`InvStdDev` on `LayerNormalization`, `inv_std_var` on
`SimplifiedLayerNormalization`) are actually consumed by anything is
declined the same conservative way a `SkipLayerNormalization` node's own
consumed `mean`/`inv_std_var` already is -- their *values*, not their shape,
depend on exactly which channels survive, and nothing here has a basis for
whether whatever reads them still expects the original ones. Multi-axis
normalization (`axis` short of the last dimension, e.g. `-2`) is declined
outright, never partially matched -- out of scope by this section's own
"single, full trailing channel axis" bar. This is a MatMul/Gemm-chain-only
hop, the same as the bias/scale `Add`/`Mul` and `BiasGelu`/`FastGelu` hops it
sits alongside: Conv's own channel axis is axis 1 (NCHW), never the *last*
axis a `LayerNormalization`-family node's default `axis=-1` actually
normalizes, so no Conv-side analogue is added. Composes for free with the
residual/gated/`Concat` machinery described below, since all three reuse
this same :func:`_walk_to_consumer` for their own forward continuation, with
no composition-specific code of its own.

A `Concat` merge -- the U-Net-style encoder/decoder skip connection
(`merged = Concat(a, b, axis=1)`, each branch keeping its own disjoint slice
of the merged channel range) -- looks at first glance like it needs the same
general dependency-graph machinery an `Add`/`SkipLayerNormalization` merge
does, and was long declined outright on that assumption. It turns out not
to: unlike `Add`, whose operands are summed position-for-position and so
*must* agree on one shared surviving channel-index set, `Concat`'s branches
are independent -- branch `a` (`Ca` channels) always owns columns `[0, Ca)`
of the merged, pre-pruning tensor and branch `b` always owns `[Ca, Ca+Cb)`,
fixed offsets neither branch's own pruning choice can move -- so each branch
is ranked and pruned entirely on its own, no cross-branch agreement needed
at all, and only the shared downstream consumer's weight needs new slicing
logic (concatenating each branch's own surviving-channel set, shifted by its
own fixed offset) to stay correct. See :func:`_find_matmul_concat_chains`/
:func:`_find_conv_concat_chains`'s own section comment for the bounded,
single-consumer-per-branch shape this reaches (a `Concat` chained
transitively into another `Concat`, or composed with a gated branch, is
declined the same conservative way as everything above) and
:func:`_apply_concat_chains`'s own docstring for why that per-branch,
independent-`keep` shape needed a genuinely new sibling to
`_Chain`/`_apply_chains` rather than fitting into the existing one. A branch
that bottoms out at an `Add`/`SkipLayerNormalization` residual merge instead
of a real producer *is* composed, in one bounded shape: the merge's own
whole transitively-connected group (see the residual sections above) is
resolved exactly as it would be standalone, and -- *only* when that group
has no consumer anywhere else at all (this one `Concat` branch is its
sole reason to exist) -- the group's own combined-importance `keep` set
becomes this one branch's own contribution, its several leaf producers all
sliced together by it. A group with any other fan-out (an interior tensor
also read elsewhere, or its sink feeding some other ordinary consumer too)
is declined outright rather than guessed at -- see
:func:`_find_matmul_concat_chains`/:func:`_find_conv_concat_chains`'s own
section comment for exactly why that line is where it's drawn.

The shared downstream consumer of a Conv `Concat` merge may itself be a
general grouped (`group != 1`) Conv -- but only when every branch's own
fixed offset lands exactly on one of the consumer's own `group` block
boundaries (`block = n_channels / group`), so every block the consumer's
own per-block top-k needs a uniform survivor count from is owned by exactly
one branch, never split across two. This is *not* simply inherited from the
ordinary grouped-producer/grouped-consumer composition above: there, every
producer's own output already sits in one *shared* index space the
consumer's blocks partition, all pruned to one shared `keep` set, so one
global block size settles every block at once. A `Concat` branch, by
contrast, is pruned *independently*, by its own top-k over its own slice,
with no visibility into any sibling branch's ranking -- the entire reason
`Concat` support exists in the first place (see above). When branch
boundaries are block-aligned, that independence is harmless: each block
falls wholly inside one branch, which alone can satisfy that block's own
uniform-count requirement (an internal per-block top-k of its own, mirroring
:func:`_apply_chains`'s mechanism, just scoped to the blocks it contains).
When a block instead straddles two branches, satisfying it needs the counts
each branch independently contributes to that one shared block to sum to
exactly the required `per_group_keep` -- which two rankings computed with no
knowledge of each other have no general way to guarantee, and reconciling it
would need exactly the cross-branch agreement `Concat` support exists to
avoid. So that case is declined outright, the whole chain, the same
conservative way as everywhere else in this module -- see
:func:`_concat_branches_align_to_consumer_group`'s own docstring for the
exact admission condition, the full argument, and a concrete counter-example
of the straddling case. (MatMul/Gemm has no grouping concept at all, so this
paragraph is Conv-only; a MatMul/Gemm `Concat` chain's consumer is always
ordinary.)

General multi-branch dependency-graph pruning remains out of scope --
a non-`Add`/`SkipLayerNormalization`/`Concat` merge op, and fan-out
anywhere *except* forward from an already-established residual/merge
group's own shared channel-index set (see above): an ordinary chain's own
producer output, or any tensor not already inside such a group, is still
declined outright the moment it has more than one consumer, exactly as
before. The
other part of the paper's pipeline -- an architecture *search* over what to prune,
alternated with knowledge-distillation/RL recovery afterwards -- needs a
training loop onnxsim does not have and is not in scope here at all; this
is a single, static, no-retraining structural cut, closer in spirit to Li
et al.'s L2-norm filter pruning (below) than to anything iterative.

What *does* fit that mold, and needs no retraining loop: post-training
*unstructured* (or semi-structured N:M) pruning, à la magnitude pruning
(Han et al., 2015, "Learning both Weights and Connections for Efficient
Neural Networks", https://arxiv.org/abs/1506.02626) and, for the
calibrated variant, Wanda (Sun et al., 2023, "A Simple and Effective
Pruning Approach for Large Language Models",
https://arxiv.org/abs/2306.11695 -- the pruning analogue of this module's
neighbors :mod:`onnxsim.awq`/:mod:`onnxsim.smoothquant`: a single forward
pass over calibration data, no weight update, no backward pass at all).
Both zero out individual weight entries and leave every tensor's shape
exactly as it was, so -- like every ``quantize_weight_only_*`` pass here --
the result is a plain ONNX model, correct by construction (a MatMul/Gemm
with some zeroed entries computes the same op, just with less nonzero
data), that a runtime with sparse-kernel support (or a later, separate
dense-to-sparse repacking step) can exploit for speed.

:func:`apply_magnitude_pruning` uses ``|W|`` as the importance metric and
needs no calibration data at all -- the simple, data-free baseline.
:func:`apply_wanda_pruning` weights that by each input feature's activation
norm over calibration data (``|W_ij| * ||X_j||_2``), which -- per the
Wanda paper -- better protects weights that multiply high-magnitude
activations even when the weight itself is individually small, the same
class of outlier-activation effect that motivates :mod:`onnxsim.smoothquant`.

Both also match 2-D ``Conv`` weights, not just MatMul/vanilla-Gemm: a
Conv's ``[out_channels, in_channels/group, kH, kW]`` weight is reshaped to
``[out_channels, (in_channels/group)*kH*kW]`` -- the same convention
:func:`apply_structured_pruning` already uses for Conv filter importance
below -- and each output filter becomes one comparison group, exactly like
a MatMul/Gemm output channel. Unlike :func:`apply_structured_pruning`'s
producer/consumer chain matching below, this reshape-and-rank-per-filter
operation is completely agnostic to ``group``: it never touches another
layer's channel indices, only ranks each output filter's own row against
itself, so ordinary (``group=1``), depthwise (``group == in_channels ==
out_channels``), and general grouped (``group`` neither 1 nor the channel
count) Conv are all matched identically here by
:func:`_match_conv_weight_only` -- a materially different, and much
easier, bar than the shape-changing coupling problem
:func:`_match_conv_producer`/:func:`_match_conv_consumer` decline part of
below. For magnitude pruning that's the entire story: ``|W|`` on the
reshaped weight, mask computed, reshaped back, working unchanged for every
``group``. For Wanda it needs one more step, since ``X_j`` isn't simply
"input feature ``j``" once a sliding kernel is involved: ``j`` indexes one
``(in_channel, kh, kw)`` offset within the receptive field, so its
activation statistic is the norm of the *im2col-unfolded* input patch
value at that specific offset, over every output spatial position and
calibration sample -- computed via a dedicated zero-padded, strided slice
per ``(kh, kw)`` tap (:func:`_conv_patch_sq_sum`) rather than materializing
an explicit im2col matrix. Every ``pads``/``auto_pad``/``strides``/
``dilations`` combination the ONNX Conv schema defines is handled: explicit
``pads`` are fixed per node, while ``auto_pad`` ``SAME_UPPER``/
``SAME_LOWER``/``VALID`` is resolved fresh from each calibration batch's
own input spatial size (:func:`_resolve_conv_pads`, per the Conv operator's
own ``auto_pad`` formula), and a non-unit ``dilations`` is handled by
offsetting each tap's own slice by ``dilation`` rather than assuming taps
are one apart. Only a genuinely malformed node (a ``kernel_shape``
disagreeing with the weight's own shape, an unrecognized ``auto_pad``
string, or non-positive ``strides``/``dilations``) falls back to plain
magnitude for that layer, the same as any other layer whose activation
norm was never observed -- channel pruning's own producer/consumer
matching above needs no such fallback in the first place, since none of
``auto_pad``/``dilations``/``pads``/``strides`` bear on which weight-tensor
axis a whole output filter or input-channel slice lives on (see
:func:`apply_structured_pruning`'s own docstring).

For a grouped or depthwise Conv, Wanda's per-offset activation norm needs
one more piece of care beyond the reshape above: that norm is always
computed once from the *raw, full-channel* input (:func:`_conv_patch_sq_sum`
never looks at `group` at all, and doesn't need to -- unfolding the whole
input once is cheaper than unfolding it again per group), but a grouped
Conv's output filter ``i`` only ever *reads* its own group's
``in_channels/group``-wide slice of that input (filter ``i`` belongs to
group ``i // (out_channels/group)``, per ONNX's grouped-Conv weight
layout), so "local receptive-field offset ``j``" names a *different*
global input channel depending on which group filter ``i`` falls in.
Sharing one norm row across every filter -- correct, and what this module
did before grouped Conv was matched here at all, when `group` was always 1
-- would silently score every filter outside group 0 against the wrong
channels' statistics for any `group` > 1. :func:`_conv_group_relative_norm`
is the fix: it slices the full-input norm's ``[Cin, kh, kw]`` shape along
its channel axis once per group and repeats each group's own slice across
exactly the filter rows belonging to it, before that expanded,
per-filter-row norm ever reaches the ``|W_ij| * ||X_j||_2`` importance
computation -- collapsing to the previous single-shared-row behavior
exactly when ``group=1``. Verified against a dedicated test engineering one
group's calibration input to have deliberately different activation
statistics from another group's, confirming each filter's resulting mask
reflects its own group's statistics and not another group's or a global
average (``test_wanda_pruning_conv_grouped_uses_own_groups_activation_norm``).

:func:`apply_sparsegpt_pruning` also matches Conv layers -- ordinary
(``group=1``), depthwise, and general grouped alike, exactly the same three
`group` shapes magnitude and Wanda above match; see its own docstring below
for the full ``[K, K]`` im2col cross-covariance Hessian this needed (a real
step up from Wanda's per-offset norm above, not just a reuse of it), how
it's verified, and how a grouped/depthwise Conv gets a genuinely *per-group*
Hessian and its own independent column-processing/error-compensation pass
rather than one shared across every filter: each group's own filters only
ever see their own group's input-channel patches -- the same channel-
slicing subtlety Wanda's grouped support above handles, but now for the
full cross-covariance rather than a per-offset norm, and needing the
sequential column-processing/error-compensation loop partitioned per group
rather than run once across the whole weight. Concretely, filter row ``i``
(belonging to group ``i // (out_channels/group)``, ONNX's own grouped-Conv
weight layout) is pruned only against ``H_g``, group ``g``'s own
``[Cin/group*kh*kw, Cin/group*kh*kw]`` Hessian, built the same way the
``group=1`` case's single ``H`` already is (:func:`_conv_im2col_patches`,
``H_g = patches_g.T @ patches_g``) but fed only that group's own global
input-channel slice ``patches_g`` rather than the full input -- reusing
:func:`_conv_im2col_patches` and :func:`_sparsegpt_prune_columns` completely
unchanged, called once per group, rather than needing any dedicated grouped
Hessian or grouped column-processing machinery of their own (see
:func:`apply_sparsegpt_pruning`'s own docstring for the exact accumulation).
This is comparable in scope to the original SparseGPT+Conv work this module
already did from first principles -- and was verified the same three ways:
a brute-force nested-loop oracle building each group's own Hessian a
completely different way (an explicit outer-product accumulation per
output position, per group, engineered with genuinely different
per-group calibration statistics so a bug sharing one Hessian across
groups, or mixing up which group's slice feeds which filter rows, would be
caught rather than passing on symmetric data), a second, independent
reference transliteration fed each group's own correctly-sliced weight/
Hessian, and the same end-to-end reconstruction-error property (against a
naive same-mask-no-compensation baseline, via onnxruntime) the ordinary
``group=1`` Conv case is already validated against.

All three unstructured/N:M functions also match the two fused self-
attention ops the "Attention-head pruning" section below performs
*structural* (whole-head) pruning on -- ``com.microsoft::Attention``'s
merged QKV weight (``[K, Nq+Nk+Nv]``, matched by
:func:`_match_attention_weight_only`, reusing :func:`_match_attention_producer`'s
own criteria) and ``com.microsoft::GroupQueryAttention``'s separate Q/K/V
projections, which need no special-casing at all: per
:func:`_match_gqa_producer`'s own docstring they are ordinary MatMul/
vanilla-Gemm nodes feeding into that op, not weights the op itself owns, so
:func:`_candidates`' existing MatMul/Gemm matching already reaches them
(ranked no differently from any other MatMul/Gemm layer). This is a
completely different code path from head pruning below: it zeros
individual weight entries within a head's columns rather than removing
whole heads, exactly as useful on its own (e.g. reaching NVIDIA Ampere's
2:4 sparse Tensor Cores, via :func:`onnxsim.convert_matmul_to_gemm`, on an
already-``fuse_attention``'d model's QKV weight) as it is combined with
head pruning first. Since unstructured/N:M pruning only ever zeros values
and never changes shape, an ``Attention`` node's ``num_heads``/
``qkv_hidden_sizes`` attributes -- which describe the merged weight's
column layout, not any zeroed-vs-nonzero distinction within it -- can never
drift out of sync with the (unchanged) weight shape, the same invariant
this holds for every other matched layer type.

Both magnitude and Wanda pruning support two sparsity patterns, chosen per
invocation:

- unstructured: for every output row (comparison group), the lowest-
  importance entries are zeroed until that row reaches the target
  ``sparsity`` fraction.
- semi-structured N:M (e.g. ``n=2, m=4`` -- NVIDIA Ampere's 2:4 structured
  sparsity, the pattern Wanda's own paper evaluates most): within every
  consecutive group of ``m`` input-channel entries in a row, only the
  ``n`` highest-importance survive.

:func:`weight_sparsity` reports the fraction of exact-zero entries across
every matched layer's weight, as a quick way to confirm a pruning call
reached its target (or to measure an already-sparse model).

:func:`apply_structured_pruning` actually removes channels (real shape
reduction, real FLOP/parameter reduction on any runtime, no sparse-kernel
support needed) from every producer -> consumer chain it can prove safe to
cut, per output-channel L2-norm importance by default (Li et al., 2017,
"Pruning Filters for Efficient ConvNets", https://arxiv.org/abs/1608.08710)
-- for a MatMul/Gemm chain, that criterion is a transplant from Conv filters
to output channels (the same one :func:`apply_magnitude_pruning`/
:func:`apply_wanda_pruning` already made for Han et al./Wanda's element-wise
criteria); for a Conv chain it is the paper's own original setting, applied
directly: each output filter's full ``[in_channels, kH, kW]`` kernel is
flattened and ranked by its own L2 norm. An ``importance_norm="l1"`` opt-in
(NNI's own pruning API offers both L1 and L2 filter-pruning criteria as a
user choice, cited above) ranks by L1 (sum of absolute magnitude) instead --
see :func:`_plain_structured_importance`'s own comment for exactly how each
norm's multi-producer combination differs (L2's is root-sum-square across
producers; L1's is a plain sum, with no square/sqrt involved at all) and
:func:`apply_structured_pruning`'s own ``importance_norm`` parameter. Conv
support is deliberately
narrower than the MatMul/Gemm path in one respect that stays true
regardless of grouping: producers/consumers are joined by unary activations
alone -- no per-channel ``Add``/``Mul`` scale-or-bias op, since a real Conv
already carries any bias in its own optional third input, and
``BatchNormalization`` is expected to already be fused into the preceding
Conv's weight by the time this pass runs (onnxsim's own default
optimization does exactly that, see ``fuse_bn_into_conv``), so a raw
per-channel affine between two Convs isn't a shape this pass special-cases.

Within that, three ``group`` shapes are distinguished. Ordinary
(``group=1``) Conv is the base case already described above. The
*depthwise* special case (``group == in_channels == out_channels``, weight
``[C, 1, kH, kW]``) is different: with one filter per channel and no
cross-channel mixing at all, output channel ``i`` depends only on input
channel ``i``, so a depthwise Conv sitting between a chain's real producer
and real consumer needs no independent importance of its own -- the chain
walk (:func:`_walk_to_conv_consumer`) crosses it transparently, like one
more shape-preserving activation hop, carrying whatever channel-index set
survives upstream straight through unchanged, while still slicing that
depthwise layer's own weight/bias by the same indices and shrinking its
``group`` attribute to match. This is exactly the ``Conv(1x1, group=1) ->
DepthwiseConv(3x3, group=C) -> Conv(1x1, group=1)`` "inverted residual"
block MobileNet/EfficientNet-style efficient CNN backbones use throughout,
so it's worth the special case; a depthwise Conv is never itself matched as
a producer or consumer (see
:func:`_match_conv_producer`/:func:`_match_conv_consumer`), only ever a
transparent hop between two real Conv boundaries -- one sitting last before
a graph output or an unhandled branch simply ends the chain unmatched, same
as any other topology this pass declines to guess at.

A *general* grouped Conv (``group`` neither 1 nor equal to its channel
count, weight ``[out_channels, in_channels/group, kH, kW]``) is the
remaining case, and -- unlike the fully-out-of-scope treatment an earlier
version of this pass gave it -- is now matched as a real producer and/or
consumer, because its structure turns out to be tractable in a way general
dependency-graph coupling isn't: since a grouped Conv's ``group`` blocks
never mix (full mixing *within* a block, none across), pruning block ``k``
is completely independent of every other block, as long as the *same
count* is pruned from every block (so ``channels % group == 0`` survives,
exactly as ONNX's Conv schema requires). Concretely:

- **As a producer**, its output-channel axis is flat/global regardless of
  grouping (grouping only ever splits the *input* axis), so each of its
  ``group`` output-filter blocks is ranked and pruned independently by the
  same per-filter L2-norm criterion above, applied within each block's own
  slice rather than across the whole ``out_channels`` axis -- keeping the
  same count from every block.
- **As a consumer**, its input-channel axis *is* per-group-relative (weight
  column ``j`` on a filter belonging to group ``g`` means global input
  channel ``g * (in_channels / group) + j``, not global channel ``j``), so
  slicing it needs dedicated per-group-relative logic
  (:func:`_slice_grouped_consumer_conv_weight`) rather than the flat
  column selection an ordinary consumer's weight uses -- but the *set* of
  surviving channels is still whatever the chain's producer side decided,
  now constrained to keep a uniform count within each of the consumer's own
  ``group`` blocks.
- **Composing the two**: a grouped producer feeding an ordinary
  (``group=1``) consumer is supported -- the consumer imposes no grouping
  constraint of its own, so the producer's own per-block selection is
  already all that's needed. An ordinary producer feeding a grouped
  consumer is likewise supported -- the producer has no grouping constraint
  of its own either, so its selection is simply constrained to the
  consumer's own block boundaries instead of an unconstrained global top-k.
  Both sides grouped is supported *only* when both share the exact same
  ``group`` count (their blocks then partition the shared channel count
  identically, so either side's per-block selection already satisfies the
  other); a mismatched ``group`` count on the two sides is declined
  outright and the whole chain is left untouched, since the two sides'
  block boundaries then wouldn't generally align at all, and reconciling
  that would need real cross-chain bookkeeping this pass does not attempt
  (the same kind of boundary attention-head pruning below draws around
  GroupQueryAttention, and general residual/branch dependency-graph
  coupling is left out of this pass entirely). See
  :func:`_match_conv_producer`/:func:`_match_conv_consumer`/
  :func:`_chain_group` for the exact matching and selection logic.
:func:`apply_structured_wanda_pruning` is the calibrated upgrade of that
same technique -- ``||W_row||_2 * ||X||_2`` per channel instead of weight
magnitude alone -- exactly the same relationship Wanda has to plain
magnitude pruning, transplanted from individual weights (or, for Conv,
whole filters) to whole channels. Because either changes shapes, the
result is unconditionally irreversible and, unlike a retrained pipeline,
has no distillation/RL step to recover whatever accuracy the cut costs --
evaluate the result before shipping it, the same caution any lossy onnxsim
pass deserves.

:func:`apply_sparsegpt_pruning` is a third, more accurate way to reach an
unstructured or N:M pattern (alongside magnitude and Wanda pruning above):
SparseGPT (Frantar & Alistarh, 2023, "SparseGPT: Massive Language Models
Can Be Accurately Pruned in One-Shot", https://arxiv.org/abs/2301.00774) --
the pruning sibling of :mod:`onnxsim.gptq`, from the same authors, reusing
the exact same machinery (:func:`onnxsim.gptq._inverse_hessian_cholesky`'s
Cholesky-factored inverse Hessian, and the same left-to-right,
error-propagating column processing) but pruning each column to a mask
instead of quantizing it to a grid point. Where magnitude/Wanda pick a
mask once from a static (weight- or weight-times-activation-) importance
score and stop, SparseGPT computes each column's OBS-style saliency score
``w_ij^2 / Hinv_jj^2`` from calibration data, then -- after masking a
column -- propagates the resulting reconstruction error into every
not-yet-processed column via the same Hessian-based correction GPTQ uses
for quantization error, so later columns compensate for earlier ones'
removal instead of every column being scored independently against the
original, uncorrected weights. This reliably beats magnitude/Wanda at the
same sparsity, at the cost of needing calibration data (there is no
data-free variant, unlike magnitude vs. Wanda) and being noticeably more
expensive per layer (one Cholesky factorization plus a sequential,
Hessian-propagating pass over every column, rather than one static
element-wise score). Ported directly from the reference implementation's
``fasterprune`` (https://github.com/IST-DASLab/sparsegpt), including one
behavior that's otherwise a departure from every other function in this
module: for *unstructured* sparsity, the reference selects one threshold
per ``proc_block_size``-wide column block, shared across every output row
in that block, rather than :func:`apply_magnitude_pruning`/
:func:`apply_wanda_pruning`'s per-row threshold -- faithfully reproduced
here rather than "corrected" to match, since the point of this function is
to reproduce SparseGPT specifically. N:M pruning is unaffected (it is
already per-row in the reference too, and matches this module's own
``n``/``m`` convention exactly).

Unlike Conv, ``com.microsoft::Attention``'s merged QKV weight is **not**
excluded from :func:`apply_sparsegpt_pruning`'s candidate list (nor, since
they were never excluded to begin with, are ``GroupQueryAttention``'s
separate Q/K/V projections -- see above). SparseGPT's correctness rests on
``H`` accurately capturing which columns of *this weight's own input*
correlate; unlike a Conv's im2col-unfolded receptive field, a merged QKV
weight's input is the same plain ``[*, K]`` activation feeding an ordinary
MatMul, with the same ``H = X^T X`` this function already computes for
every other MatMul/Gemm layer, no new numerical machinery needed. What each
output column of that weight is later used for downstream (split into
Q/K/V, fed into a fused attention kernel) has no bearing on the linear-
algebra correctness of pruning *this* layer's own weight against *this*
layer's own input -- exactly why it would be inconsistent to include GQA's
separate Q/K/V MatMuls (already unconditionally in scope, being ordinary
MatMul/Gemm nodes) while excluding Attention's merged one on some notion of
"this is an attention weight, treat it specially": neither needs it.

Wanda's calibrated metric gets a narrower, Attention-specific version of the
same treatment: the op's own `X` input is rank-3 (``[batch, seq, hidden]``),
not the plain 2-D tensor the metric's shared probe requires, so a *generic*
"is this activation plain 2-D" check spanning every MatMul/Gemm/Attention
candidate would either always miss this weight (the old behavior) or --
generalized to reduce over leading axes -- silently change the documented,
tested fallback behavior of every *existing* MatMul/Gemm layer whose own
activation happens to be rank 3+ too (a batched-sequence input is an
entirely ordinary shape for a plain linear layer, not something unique to
Attention). :func:`apply_wanda_pruning` instead accumulates a *second*,
Attention-only activation statistic alongside the generic one, gated on the
node itself (domain + op_type -- exactly :func:`_match_attention_producer`'s
own check, not activation shape) rather than broadening the generic check,
and reduces that statistic over every leading axis (mirroring
:func:`apply_sparsegpt_pruning`'s own ``x.reshape(-1, x.shape[-1])``) purely
for this one weight. Every other MatMul/Gemm layer's own rank-3+-activation
fallback is untouched -- the generic probe and its ``x.ndim != 2`` check
are exactly as before.

:func:`apply_sparsegpt_pruning` also matches 2-D ``Conv`` layers -- ordinary
(``group=1``), depthwise, and general grouped alike -- using exactly the
same producer matching, Cholesky machinery, and column-processing loop
above -- the only genuinely new piece is how ``H`` itself gets built. For a
MatMul/Gemm layer
``H = X.T @ X`` is already a full cross-covariance, since each column
*is* an independent input feature; for Conv, a weight column instead
indexes one ``(in_channel, kh, kw)`` receptive-field offset (the same
reshape :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning` and
:func:`apply_structured_pruning` already use), so a correct ``H`` needs
the full ``[K, K]`` cross-covariance of every offset against every other
-- not merely each offset's own norm the way Wanda's diagonal-only
``_conv_patch_sq_sum`` needs. :func:`_conv_im2col_patches` builds that:
the same zero-padded, per-tap strided-slice unfolding
:func:`_conv_patch_sq_sum` already does (reusing
:func:`_conv_spatial_attrs`/:func:`_resolve_conv_pads` for the padding/
stride/dilation handling, ``auto_pad`` included -- a node still declined
by :func:`_conv_spatial_attrs` itself, e.g. a malformed `kernel_shape`,
leaves the layer completely untouched, there being no data-free fallback
for SparseGPT, Conv included), but returning the actual ``[n_positions,
K]`` patch matrix rather than reducing straight to a per-offset sum of
squares, so ``H = patches.T @ patches`` can be formed from it. Verified
two independent ways before being trusted here: a brute-force nested-loop
oracle that builds the same
``[K, K]`` Hessian a completely different way (one Python triple-loop per
output position, accumulating an explicit outer product, rather than any
vectorized unfolding), the same bar
``test_conv_patch_sq_sum_matches_naive_nested_loop_oracle`` already set
for Wanda's per-offset norm; and, end to end, the same reconstruction-
error property the MatMul/Gemm path is already validated against -- a
SparseGPT-pruned Conv layer's output should reconstruct the float layer's
output at least as well as naive same-mask zeroing with no compensation,
on well-conditioned calibration data. Because a full patch matrix for a
realistic layer can be large (``n_positions`` grows with output spatial
size, not just channel count), ``H`` accumulates incrementally, one
calibration batch's own unfolded patches at a time (``H += patches.T @
patches``, each batch's patches discarded once folded in), rather than
ever concatenating every batch's patches into one array first the way the
MatMul/Gemm path above still concatenates its (much smaller, already
per-feature) 2-D activations. Unlike the reference implementation's own
``add_batch`` (https://github.com/IST-DASLab/sparsegpt/blob/master/
sparsegpt.py), which never actually unfolds a Conv2d activation at all --
its Conv branch reshapes only the *weight* (``W.flatten(1)``), and its own
driver scripts (``opt.py``, ``llama.py``, ...) never exercise a Conv layer
in the first place, since OPT/BLOOM/Llama have none -- there was no
correct reference to port here, unlike every other technique this module
ports from an upstream implementation; this is original, from-first-
principles machinery, held to the verification bar above precisely
because of that.

For a grouped or depthwise Conv, the same channel-slicing subtlety Wanda's
own grouped support needs (see :func:`_conv_group_relative_norm`'s
paragraph above) applies here too, but for the full cross-covariance
rather than a per-offset norm: filter row ``i`` (belonging to group
``i // (out_channels/group)``, ONNX's own grouped-Conv weight layout) only
ever reads its own group's global input-channel slice
``[g*Cin/group, (g+1)*Cin/group)``, so a shared, whole-input ``H`` would
silently correlate every filter against every other group's channels too
-- wrong, not merely imprecise, since ``H``'s off-diagonal entries would
then encode spurious cross-group covariance no real filter ever sees.
The fix needs both a genuinely *per-group* Hessian **and** the sequential
column-processing/error-compensation loop run independently per group (a
column-masking decision and its downstream error compensation only make
sense within one group's own consistent Hessian/weight coordinate system,
not mixed across groups) -- but, unlike that description's own apparent
scope, turns out to need no new numerical machinery at all: group ``g``'s
own ``H_g = patches_g.T @ patches_g`` is built by feeding
:func:`_conv_im2col_patches` -- completely unchanged -- only that group's
own channel-sliced sub-tensor (``x[:, g*Cin/group:(g+1)*Cin/group, :, :]``)
rather than the full input, exactly the same function called once per
group instead of once total; and :func:`_sparsegpt_prune_columns` --
likewise completely unchanged -- is then simply called once per group on
that group's own ``[Cout/group, Cin/group*kh*kw]`` weight sub-block against
that group's own ``H_g``, rather than once across the whole weight against
one shared ``H``. Total im2col-unfolding work across every group's own
channel-sliced call sums to exactly one full-input unfold (the groups'
channel slices partition the input's channel axis with no overlap), so
this costs no more overall than the ``group=1`` case already did -- see
:func:`apply_sparsegpt_pruning`'s own docstring for exactly how each
group's own ``H_g`` accumulates batch by batch. Verified the same three
ways the ``group=1`` case above was: a brute-force nested-loop oracle
building each group's own ``[K, K]`` Hessian a completely different way
(an explicit outer-product accumulation per output position, per group,
rather than any vectorized unfolding), engineered with genuinely different
per-group calibration statistics (the same technique
:func:`_conv_group_relative_norm`'s own grouped-Wanda test uses) so a bug
sharing one Hessian across groups, or mixing up which group's slice feeds
which filter rows, is caught rather than accidentally passing on symmetric
data; a second, independent reference transliteration
(``_reference_sparsegpt``, already validated against the ``group=1`` case)
fed each group's own correctly-sliced weight/Hessian, confirmed to match
:func:`apply_sparsegpt_pruning`'s actual output exactly, for both
unstructured and N:M sparsity; and the same end-to-end reconstruction-
error property (against a naive same-mask-no-compensation baseline, via
onnxruntime) the ``group=1`` case is validated against, including the
depthwise extreme (``group == Cin == Cout``, ``Cin/group == 1``), where
each group's own Hessian correctly degenerates to a ``[kh*kw, kh*kw]``
per-channel Hessian rather than anything degenerate or wrong at that
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.shape_inference

from onnxsim import backend
from onnxsim.bias_correction import _add_probe_outputs
from onnxsim.calibration import Tensors, generate_random_calibration_data
from onnxsim.gptq import _inverse_hessian_cholesky
from onnxsim.smoothquant import _match_matmul_like

# Weight-magnitude norm choice for the *structured* (channel/filter/head)
# importance rankings below (:func:`apply_structured_pruning`,
# :func:`apply_structured_wanda_pruning`, :func:`apply_attention_head_pruning`,
# :func:`apply_attention_head_wanda_pruning`) -- "l2" (the default, and this
# module's only behavior before this parameter existed) is Li et al.'s own
# criterion; "l1" is NNI's alternative filter-pruning criterion (its pruning
# API offers both as a user choice). Every one of those functions' own
# importance helper takes this as a plain ``str`` (not this ``Literal``
# alias) so a caller-supplied closure/partial can bind it without importing
# the alias too -- this exists purely to give the four public entry points'
# own signatures a checked, self-documenting type.
_ImportanceNorm = Literal["l1", "l2"]


def _validate_importance_norm(importance_norm: str) -> None:
    if importance_norm not in ("l1", "l2"):
        raise ValueError(
            f"importance_norm must be 'l1' or 'l2', got {importance_norm!r}"
        )


def _validate_pattern(sparsity: float, n: Optional[int], m: Optional[int]) -> None:
    if (n is None) != (m is None):
        raise ValueError("n and m must be given together (N:M pruning) or not at all")
    if n is not None and m is not None:
        if not (0 < n <= m):
            raise ValueError(f"require 0 < n <= m, got n={n}, m={m}")
    elif not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")


def _sparsity_mask(importance: np.ndarray, sparsity: float) -> np.ndarray:
    # Per-row (per-output-channel) threshold, matching Wanda's own
    # per-output comparison group rather than a single global threshold --
    # a layer with output-channel-dependent weight/activation scale would
    # otherwise have some rows pruned to nothing and others left untouched.
    rows, cols = importance.shape
    keep = max(1, round(cols * (1.0 - sparsity)))
    if keep >= cols:
        return np.ones((rows, cols), dtype=bool)
    order = np.argsort(importance, axis=1)
    drop = order[:, : cols - keep]
    mask = np.ones((rows, cols), dtype=bool)
    np.put_along_axis(mask, drop, False, axis=1)
    return mask


def _nm_mask(importance: np.ndarray, n: int, m: int) -> np.ndarray:
    """Row-wise N:M mask: within every consecutive group of ``m`` columns,
    keeps only the ``n`` highest-importance entries. A trailing partial
    group (fewer than ``m`` columns) keeps a proportional share (rounded,
    at least 1) instead of raising on a non-multiple-of-``m`` width.
    """
    rows, cols = importance.shape
    mask = np.ones((rows, cols), dtype=bool)
    full_cols = (cols // m) * m
    if full_cols:
        groups = importance[:, :full_cols].reshape(rows, full_cols // m, m)
        order = np.argsort(groups, axis=2)
        drop = order[:, :, : m - n]
        group_mask = np.ones_like(groups, dtype=bool)
        np.put_along_axis(group_mask, drop, False, axis=2)
        mask[:, :full_cols] = group_mask.reshape(rows, full_cols)
    tail = cols - full_cols
    if tail:
        keep = min(tail, max(1, round(n * tail / m)))
        tail_importance = importance[:, full_cols:]
        order = np.argsort(tail_importance, axis=1)
        drop = order[:, : tail - keep]
        tail_mask = np.ones((rows, tail), dtype=bool)
        np.put_along_axis(tail_mask, drop, False, axis=1)
        mask[:, full_cols:] = tail_mask
    return mask


# --- FP16/BFloat16 weight support ---------------------------------------
#
# Every matcher below that used to hard-require ``onnx.TensorProto.FLOAT``
# now accepts float32, float16, and bfloat16 via the two helpers just
# below: read a matched weight/bias out as float64 (:func:`_to_f64`,
# losslessly -- float64 has strictly more mantissa bits than either half-
# precision format, so this upcast never itself rounds), do every existing
# float32-native computation in that widened precision exactly as before,
# then cast the result back down to the tensor's own original dtype
# (:func:`_from_f64`) before writing it into the graph -- so a fp16/bf16
# model's own declared dtypes round-trip unchanged, rather than silently
# widening to float32 on every pruning pass. No existing float32 math,
# importance formula, or numeric behavior changes: float32 in, float32 out
# is exactly ``.astype(np.float64)`` then ``.astype(np.float32)``, the same
# no-op-in-precision round trip this module's float32 code paths already
# performed before this support was added.
#
# BFLOAT16 support leans on ``onnx>=1.22``'s own hard dependency on
# ``ml_dtypes`` (confirmed present transitively -- see this module's own
# test suite) rather than adding a new one: ``onnx.numpy_helper.to_array``
# already decodes a BFLOAT16 tensor to a numpy array of
# ``ml_dtypes.bfloat16`` (a real registered numpy dtype, not a raw
# ``uint16`` view), and ``.astype(np.float64)``/``.astype(bfloat16)`` both
# work on it directly with no manual bit-reinterpretation needed -- verified
# empirically against a real BFLOAT16 tensor, see the test suite's
# ``test_bfloat16_*`` cases.
#
# The structured/attention/MoE pruning sections below need *no*
# :func:`_to_f64`/:func:`_from_f64` calls at all in most of their own
# ``_slice_*``/``_apply_*`` helpers, and that is deliberate, not an
# oversight: those helpers only ever reorder or drop whole
# rows/columns/experts/heads (``w[keep, ...]``, ``np.take(w, keep,
# axis=...)``, ``np.concatenate`` of such slices) -- never recompute a
# surviving value -- and both numpy fancy-indexing/``np.take`` and
# ``onnx.numpy_helper.from_array`` already preserve whatever dtype the
# array they're given carries, FLOAT16/BFLOAT16 included (verified
# empirically). So once a chain's own matcher accepts FLOAT16/BFLOAT16 (the
# actual fix, applied at every matcher above), every downstream slice call
# already round-trips that same dtype correctly with zero source changes.
# The exceptions -- a function that genuinely computes a new value from a
# weight's own contents (an importance/norm score, an averaged/merged
# replacement value) rather than just permuting existing ones -- upcast via
# :func:`_to_f64` for that computation same as every other pass in this
# module, and are called out individually where they occur.
_SUPPORTED_WEIGHT_DTYPES = (
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.FLOAT16,
    onnx.TensorProto.BFLOAT16,
)


def _is_supported_float_dtype(data_type: int) -> bool:
    """True for FLOAT, FLOAT16, and BFLOAT16 -- every element dtype this
    module's matchers accept for a weight/bias initializer. See the
    "FP16/BFloat16 weight support" section comment above for how a matched
    tensor of any of these three dtypes is handled uniformly by every
    downstream ``apply_*`` function.
    """
    return data_type in _SUPPORTED_WEIGHT_DTYPES


def _to_f64(t: onnx.TensorProto) -> np.ndarray:
    """Reads tensor `t` (must be FLOAT, FLOAT16, or BFLOAT16 --
    :func:`_is_supported_float_dtype`) out of the graph as a ``float64``
    numpy array, ready for this module's existing float32/float64 math
    unchanged. Lossless for every input dtype: float64 can represent every
    value either half-precision format can exactly, so this upcast alone
    never rounds -- only whatever pruning math runs afterwards chooses its
    own working/output precision.
    """
    return onnx.numpy_helper.to_array(t).astype(np.float64)


def _from_f64(arr: np.ndarray, data_type: int, name: str) -> onnx.TensorProto:
    """Inverse of :func:`_to_f64`: downcasts a float64 array back to
    `data_type` (the original tensor's own dtype) and wraps it as a new
    ``TensorProto`` named `name`, ready for ``w_init.CopyFrom(...)``. For a
    purely structural rewrite -- masking entries to zero, or slicing
    rows/columns out -- every *surviving* entry's value is untouched all
    the way through (only removed/zeroed), so round-tripping it through
    float64 with no arithmetic in between reproduces the exact original
    fp16/bf16 bit pattern (verified empirically -- see this module's own
    test suite); a technique whose math actually *recomputes* a kept
    value's magnitude (e.g. SparseGPT's Hessian-compensated update) can of
    course still produce a different value there, but that's the
    algorithm's own float64 rounding decision, not an artifact of this
    downcast.
    """
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(data_type)
    return onnx.numpy_helper.from_array(np.asarray(arr).astype(np_dtype), name=name)


def _match_conv_weight_only(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    allow_grouped: bool = True,
) -> Optional[Tuple[str, str]]:
    """If `node` is a 2-D ``Conv`` with a constant 4-D float32/float16/
    bfloat16 ``[out_channels, in_channels/group, kH, kW]`` weight, returns
    ``(x_name, weight_name)``. Mirrors the "Conv2D structured pruning"
    section's own :func:`_match_conv_producer` matching criteria, minus
    that function's bias handling -- magnitude/Wanda/SparseGPT never touch
    bias, only the weight, so there's nothing to validate there.

    Unlike :func:`_match_conv_producer`/:func:`_match_conv_consumer`, a
    grouped or depthwise Conv (``group != 1``) is matched here too when
    `allow_grouped` (the default, used by all three of
    :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning`/
    :func:`apply_sparsegpt_pruning`): those *structured*-pruning functions'
    own ``group=1`` restriction exists because their producer/consumer
    channel-index coupling genuinely doesn't survive grouping (an output or
    input channel's index meaning depends on which of `group`'s blocks it
    falls into on each side of a chain -- see this module's own docstring).
    Nothing here inherits that problem: unstructured/N:M pruning never
    changes any shape or needs any cross-layer index agreement -- it only
    zeros individual weight entries within one output filter's own kernel,
    independently of every other filter. For magnitude/Wanda, that's simply
    :func:`_prune_weight`'s ``w.reshape(n, cin*kh*kw)`` (`cin` here already
    being ``in_channels/group`` by ONNX's own grouped-Conv weight layout)
    ranking and masking each of the `n` output filters' rows the same way
    regardless of `group`; for SparseGPT it needs the materially bigger
    step of a genuinely *per-group* Hessian and column-processing loop (see
    :func:`apply_sparsegpt_pruning`'s own docstring), but the matching
    criterion itself -- whether this Conv is eligible at all -- is
    identical either way, hence one shared matcher for all three. Passing
    `allow_grouped=False` restores the ``group=1``-only match (no current
    caller in this module does; kept as a general-purpose restriction for
    any future caller that needs it). When `allow_grouped` and
    ``group > 1``, ``out_channels % group`` must still be zero -- the
    standard grouped-Conv well-formedness requirement (`group` equal-sized
    output blocks) that :func:`_conv_group_relative_norm` also relies on to
    line up each output filter with its own group's input-channel slice.

    Deliberately still 2-D-only (``len(w_init.dims) != 4``, not the ``>= 3``
    :func:`_match_conv_producer`/:func:`_match_conv_consumer` now accept for
    *structural* channel pruning): this module's calibration machinery for
    Wanda (:func:`_conv_spatial_attrs`/:func:`_conv_patch_sq_sum`) and
    SparseGPT (:func:`_conv_im2col_patches`, plus an unconditional
    ``cout, cin_per_group, kh, kw = w.shape`` unpack in
    :func:`apply_sparsegpt_pruning`'s own body) hard-codes exactly two
    spatial dims throughout its own im2col-style patch unfolding -- padding
    fields (`pad_top`/`pad_left`/`pad_bottom`/`pad_right`),
    stride/dilation pairs, and per-tap loops all assume `n=2`. Generalizing
    that to N spatial dims is a materially bigger, more invasive rewrite
    than the structural side's own "the slicing never touches spatial axes
    at all" generalization, and risks silently changing already-verified
    2-D numeric behavior in the process -- so it is explicitly left out of
    scope here (a Conv1d/Conv3d weight is declined, `None`, not silently
    mismatched or crashed on) rather than attempted alongside the
    structural generalization above. Plain :func:`apply_magnitude_pruning`
    itself needs no im2col calibration at all (its own metric is `|W|`,
    computed the same way regardless of spatial rank) and *could* be
    generalized cheaply on its own, but is kept at this same 2-D-only bar
    for consistency -- one matcher, one scope boundary, shared by all three
    calibration-based callers -- rather than silently diverging per-caller.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or not _is_supported_float_dtype(w_init.data_type)
        or len(w_init.dims) != 4
    ):
        return None
    group = _conv_group(node)
    if group < 1:
        return None
    if not allow_grouped and group != 1:
        return None
    if group > 1 and w_init.dims[0] % group != 0:
        return None
    return node.input[0], w_name


def _match_attention_weight_only(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, str]]:
    """If `node` is a ``com.microsoft::Attention`` node with a constant 2-D
    float32 merged QKV weight (``[K, Nq+Nk+Nv]``), returns
    ``(x_name, weight_name)``. Mirrors the "Attention-head pruning"
    section's own :func:`_match_attention_producer` matching criteria
    (including its ``num_heads``/``qkv_hidden_sizes`` consistency checks --
    reused verbatim rather than re-implemented, even though nothing here
    reads `num_heads` itself, so that a node this module's *structural*
    head-pruning functions would decline as malformed is declined the same
    way here), minus that function's bias handling -- magnitude/Wanda/
    SparseGPT never touch bias, only the weight, so there's nothing to
    validate there. Per :func:`_match_attention_producer`'s own docstring,
    the merged weight has no transpose attribute of its own -- it is
    already ``[K, N]``-shaped by construction -- so it is matched here
    exactly like a non-transposed MatMul weight (``weight_transposed =
    False`` in :func:`_candidates`' returned tuple); :func:`_prune_weight`'s
    existing non-Conv path handles that shape with no Attention-specific
    code at all. Unstructured/N:M pruning only ever zeros entries -- it
    never changes `w_name`'s shape -- so the un-pruned merged weight's
    ``num_heads``/``qkv_hidden_sizes`` attributes (read by every other
    consumer of this weight, e.g. onnx.checker and any runtime) never drift
    out of sync with its actual shape, the same invariant every other
    matched layer type already gets from this module's value-only rewrite.
    """
    info = _match_attention_producer(node, initializer_map)
    if info is None:
        return None
    return node.input[0], node.input[1]


def _candidates(
    graph: onnx.GraphProto, include_conv: bool = True, allow_grouped_conv: bool = True
):
    """Every MatMul/vanilla-Gemm node with a constant 2-D float32 weight
    (this already includes a ``com.microsoft::GroupQueryAttention`` node's
    separate Q/K/V projections: per :func:`_match_gqa_producer`'s own
    docstring they are ordinary MatMul/vanilla-Gemm nodes feeding into that
    op, not weights the op itself owns, so they need no special-casing here
    at all -- they are ranked and pruned no differently from any other
    MatMul/Gemm layer), plus:

    - every ``com.microsoft::Attention`` node's constant 2-D float32 merged
      QKV weight, matched by :func:`_match_attention_weight_only` -- unlike
      GQA's separate projections, this *is* a weight the op itself owns
      (``node.input[1]``), so it needs its own matcher; and
    - when `include_conv` (the default; no caller in this module passes
      ``False``, it exists purely as a general-purpose "MatMul/Gemm/
      Attention only" restriction for any future caller that wants it) --
      every 2-D ``Conv`` node matched by :func:`_match_conv_weight_only`,
      which by default (`allow_grouped_conv`, also default) includes
      depthwise and general grouped Conv, not just ordinary (``group=1``)
      Conv -- every one of :func:`apply_magnitude_pruning`/
      :func:`apply_wanda_pruning`/:func:`apply_sparsegpt_pruning` matches
      all three `group` shapes identically at the `_candidates` level; what
      differs between them is entirely how each one's own importance/
      Hessian machinery downstream handles grouping (a per-filter-row
      reshape for magnitude/Wanda, a genuinely per-group Hessian and
      column-processing loop for SparseGPT -- see each function's own
      docstring), never which Conv nodes get matched in the first place.

    Returns ``(node, x_name, w_name, weight_transposed, is_conv)`` tuples;
    `weight_transposed` is always ``False`` (meaningless) for a Conv entry,
    whose output channel always lives on axis 0 of its fixed
    ``[out_channels, in_channels/group, kH, kW]`` layout, and likewise
    always ``False`` (this time literally correct, not just meaningless --
    see :func:`_match_attention_weight_only`) for an Attention entry.
    Attention matching is unconditional (not gated by `include_conv`): see
    :func:`apply_sparsegpt_pruning`'s own docstring for why a merged QKV
    weight has no Conv-style gap of its own and is included in every
    candidate list regardless.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    out = []
    for node in graph.node:
        match = _match_matmul_like(node)
        if match is not None:
            x_name, w_name, weight_transposed = match
            w_init = initializer_map.get(w_name)
            if (
                w_init is None
                or not _is_supported_float_dtype(w_init.data_type)
                or len(w_init.dims) != 2
            ):
                continue
            out.append((node, x_name, w_name, weight_transposed, False))
            continue
        if include_conv:
            conv_match = _match_conv_weight_only(
                node, initializer_map, allow_grouped=allow_grouped_conv
            )
            if conv_match is not None:
                x_name, w_name = conv_match
                out.append((node, x_name, w_name, False, True))
                continue
        attn_match = _match_attention_weight_only(node, initializer_map)
        if attn_match is not None:
            x_name, w_name = attn_match
            out.append((node, x_name, w_name, False, False))
    return out


def _weight_to_nk(w: np.ndarray, weight_transposed: bool, is_conv: bool) -> np.ndarray:
    """``[out_channels, in_channels, kH, kW]`` -> ``[N, K]`` for a Conv
    weight, or a plain 2-D MatMul/Gemm (or Attention merged-QKV) weight
    transposed to ``[N, K]`` (output channel first) when it isn't already --
    the shared reshape/transpose convention every unstructured-pruning
    importance/masking function in this module works in. Factored out of
    :func:`_prune_weight` so :func:`_apply_global_unstructured_pruning`
    (:func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning`'s own
    ``global_sparsity`` mode) can build the exact same ``[N, K]`` view
    *before* any masking decision -- unlike :func:`_prune_weight`'s own
    per-layer callback, global mode needs every candidate's own ``w_nk``
    gathered up front, to pool their importance before any one of them can
    be masked.
    """
    if is_conv:
        n, cin, kh, kw = w.shape
        return w.reshape(n, cin * kh * kw)
    return w if weight_transposed else w.T


def _nk_to_weight(
    w_nk: np.ndarray,
    orig_shape: Tuple[int, ...],
    weight_transposed: bool,
    is_conv: bool,
) -> np.ndarray:
    """Inverse of :func:`_weight_to_nk`: reshapes/transposes an already-
    masked ``[N, K]`` array back to `orig_shape`, still ``float64`` --
    callers downcast to the tensor's own original dtype (FLOAT/FLOAT16/
    BFLOAT16, via :func:`_from_f64`) right before
    :func:`onnx.numpy_helper.from_array`, not here, so this stays a pure
    reshape/transpose with no dtype opinion of its own.
    """
    if is_conv:
        return w_nk.reshape(orig_shape)
    w_new = w_nk if weight_transposed else w_nk.T
    return w_new.reshape(orig_shape)


def _prune_weight(
    w_init: onnx.TensorProto,
    weight_transposed: bool,
    importance_of_nk,
    is_conv: bool = False,
) -> None:
    w = _to_f64(w_init)
    w_nk = _weight_to_nk(w, weight_transposed, is_conv)
    mask = importance_of_nk(w_nk)
    w_pruned_nk = np.where(mask, w_nk, 0.0)
    w_new = _nk_to_weight(w_pruned_nk, w.shape, weight_transposed, is_conv)
    w_init.CopyFrom(_from_f64(w_new, w_init.data_type, w_init.name))


def _apply_global_unstructured_pruning(
    entries: List[
        Tuple[onnx.TensorProto, bool, bool, Tuple[int, ...], np.ndarray, np.ndarray]
    ],
    sparsity: float,
) -> None:
    """Global-sparsity companion to :func:`_prune_weight`'s own per-layer
    ``importance_of_nk`` masking, shared by :func:`apply_magnitude_pruning`/
    :func:`apply_wanda_pruning`'s own ``global_sparsity=True`` mode (see
    each function's own docstring). Instead of every matched layer
    independently keeping the same *fraction* of its own entries
    (:func:`_sparsity_mask`'s own per-row rule), this pools every entry's
    own already-computed `importance` array (plain ``|W|`` for magnitude
    pruning, Wanda's ``|W_ij| * ||X_j||_2`` for Wanda pruning -- the
    metric itself is entirely the caller's concern, this function only
    ever sees the resulting numbers) into one flat array tagged by which
    layer/position it came from, picks a *single* keep-count from
    `sparsity`'s fraction of the *total* pooled entry count across every
    layer combined, and zeros exactly the lowest-scoring ``total -
    keep_count`` pooled entries wherever they land -- Han et al.'s
    original "global magnitude pruning" (2015), generalized to any of
    this module's own per-entry importance metrics.

    Deliberately enforces no per-row/per-layer floor the way
    :func:`_sparsity_mask` does (``keep = max(1, ...)``): a genuinely
    global threshold can legitimately zero an entire row, or even an
    entire layer's weight, when every one of its entries scores below the
    one shared global cutoff -- exactly the "a layer full of tiny weights
    ends up pruned harder than one full of large weights" redistribution
    this mode exists to provide, not a bug to guard against. This is safe
    here in a way it would not be for :func:`apply_structured_pruning`'s
    own ``global_sparsity`` mode (:func:`_apply_chains_global`): zeroing
    every entry of one row/layer still leaves a well-formed tensor of the
    same shape, unlike collapsing a structural channel count to zero.

    `entries` is ``(w_init, weight_transposed, is_conv, orig_shape, w_nk,
    importance)`` per matched layer -- `w_nk` and `importance` share one
    shape, `w_nk` already in :func:`_weight_to_nk`'s own ``[N, K]``
    convention, ready to reshape back via `orig_shape`
    (:func:`_nk_to_weight`) once masked. Ties at the cutoff value are
    broken by a fixed, deterministic pooled order
    (:func:`numpy.argsort`'s own stable sort over the concatenated array,
    `entries` order then each entry's own row-major flatten order) rather
    than by weight name, so the exact `keep_count` this function computes
    is always met exactly (mirroring every other rounding rule in this
    module, e.g. :func:`_sparsity_mask`'s own ``round``), rather than left
    to however many entries happen to tie exactly at whatever cutoff
    *value* a plain ``>=`` comparison would use instead.
    """
    if not entries:
        return
    pooled = np.concatenate([importance.reshape(-1) for *_, importance in entries])
    total = pooled.size
    if total == 0:
        return
    keep_count = min(max(round(total * (1.0 - sparsity)), 0), total)
    drop_count = total - keep_count

    drop_flat = np.zeros(total, dtype=bool)
    if drop_count > 0:
        order = np.argsort(pooled, kind="stable")
        drop_flat[order[:drop_count]] = True

    offset = 0
    for w_init, weight_transposed, is_conv, orig_shape, w_nk, _importance in entries:
        size = w_nk.size
        drop_here = drop_flat[offset : offset + size].reshape(w_nk.shape)
        offset += size
        w_pruned_nk = np.where(drop_here, 0.0, w_nk)
        w_new = _nk_to_weight(w_pruned_nk, orig_shape, weight_transposed, is_conv)
        w_init.CopyFrom(_from_f64(w_new, w_init.data_type, w_init.name))


# --- Conv im2col-unfolded activation statistics (Wanda only) -----------


@dataclass(frozen=True)
class _ConvSpatialAttrs:
    kh: int
    kw: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    stride_h: int
    stride_w: int
    # Both default to the ONNX Conv schema's own defaults so every existing
    # direct construction of this dataclass (this module's own oracle tests,
    # which only ever exercise the explicit-`pads` case) keeps meaning
    # exactly what it always did: unit dilation, explicit/fixed padding.
    dilation_h: int = 1
    dilation_w: int = 1
    # "NOTSET" (the schema default): `pad_top`/`pad_left`/`pad_bottom`/
    # `pad_right` above are used as-is, fixed per node. "SAME_UPPER"/
    # "SAME_LOWER"/"VALID": those four fields are unused placeholders --
    # the real padding is resolved fresh per calibration batch by
    # :func:`_resolve_conv_pads`, from that batch's own input spatial size.
    auto_pad: str = "NOTSET"


def _conv_spatial_attrs(
    node: onnx.NodeProto, w_init: onnx.TensorProto
) -> Optional[_ConvSpatialAttrs]:
    """Extracts the padding/stride/dilation a Conv node's calibration input
    needs to be correctly im2col-unfolded for Wanda's per-``(in_channel,
    kh, kw)`` activation norm (:func:`_conv_patch_sq_sum`) and SparseGPT's
    full im2col Hessian (:func:`_conv_im2col_patches`) -- see this module's
    own docstring. Handles every ``auto_pad``/``dilations`` combination the
    ONNX Conv schema defines, not just the explicit-``pads``/unit-dilation
    case an earlier version of this function alone confidently handled:

    - ``auto_pad`` ``SAME_UPPER``/``SAME_LOWER``/``VALID`` is resolved to
      concrete padding by :func:`_resolve_conv_pads`, not here -- its own
      padding amount is a function of the input's own spatial size (via
      ``ceil(in / stride)``, per the ONNX Conv operator's own ``auto_pad``
      formula: https://onnx.ai/onnx/operators/onnx__Conv.html), which is
      known once a calibration batch's actual ``x`` is in hand but is
      *not* a fixed, per-node quantity the way an explicit ``pads`` is (it
      can even vary calibration batch to calibration batch, for a
      dynamic-input-shape model) -- so nothing about it is resolved or
      cached here, only the raw ``auto_pad`` string itself is kept, on
      `_ConvSpatialAttrs.auto_pad`;
    - a non-all-ones ``dilations`` no longer needs decline either:
      :func:`_conv_patch_sq_sum`/:func:`_conv_im2col_patches` extract each
      of the ``kh*kw`` receptive-field taps as its own dedicated strided
      slice of the padded input (offset by ``dilation`` per kernel
      position, subsampled by ``stride`` the same as always), rather than
      relying on ``numpy.lib.stride_tricks.sliding_window_view``'s
      unit-offset window, which is what actually assumed unit dilation
      before -- `_ConvSpatialAttrs.dilation_h`/`dilation_w` simply carry
      the real value through.

    Still declines (``None``, meaning "fall back to plain magnitude", per
    this module's own docstring) on a `kernel_shape` that disagrees with
    `w_init`'s own shape (a malformed node -- don't guess), an
    unrecognized `auto_pad` string, non-positive `strides`/`dilations`, or
    a malformed explicit `pads` (wrong length, or negative) -- the same
    "don't guess at a malformed node" bar every other check in this module
    holds to.
    """
    kh, kw = int(w_init.dims[2]), int(w_init.dims[3])
    auto_pad = "NOTSET"
    pads: Optional[List[int]] = None
    strides: Optional[List[int]] = None
    dilations: Optional[List[int]] = None
    for attr in node.attribute:
        if attr.name == "auto_pad":
            auto_pad = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
        elif attr.name == "pads":
            pads = list(attr.ints)
        elif attr.name == "strides":
            strides = list(attr.ints)
        elif attr.name == "dilations":
            dilations = list(attr.ints)
        elif attr.name == "kernel_shape":
            ks = list(attr.ints)
            if len(ks) != 2 or ks[0] != kh or ks[1] != kw:
                return None  # weight/attribute mismatch -- don't guess

    if auto_pad not in ("NOTSET", "", "SAME_UPPER", "SAME_LOWER", "VALID"):
        return None  # unrecognized -- don't guess
    if strides is None:
        strides = [1, 1]  # ONNX Conv schema default
    if len(strides) != 2 or any(s <= 0 for s in strides):
        return None
    if dilations is None:
        dilations = [1, 1]  # ONNX Conv schema default
    if len(dilations) != 2 or any(d <= 0 for d in dilations):
        return None

    if auto_pad in ("NOTSET", ""):
        if pads is None:
            pads = [0, 0, 0, 0]  # ONNX Conv schema default
        if len(pads) != 4 or any(p < 0 for p in pads):
            return None
        pad_top, pad_left, pad_bottom, pad_right = pads
    else:
        # Resolved fresh per calibration batch by _resolve_conv_pads --
        # these four are unused placeholders in this branch.
        pad_top = pad_left = pad_bottom = pad_right = 0

    return _ConvSpatialAttrs(
        kh=kh,
        kw=kw,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        stride_h=strides[0],
        stride_w=strides[1],
        dilation_h=dilations[0],
        dilation_w=dilations[1],
        auto_pad=auto_pad,
    )


def _resolve_conv_pads(
    attrs: _ConvSpatialAttrs, in_h: int, in_w: int
) -> Tuple[int, int, int, int]:
    """Resolves one Conv node's actual ``(pad_top, pad_left, pad_bottom,
    pad_right)`` for one calibration batch's own ``[N, Cin, in_h, in_w]``
    input, per the ONNX Conv operator's own ``auto_pad`` formula
    (https://onnx.ai/onnx/operators/onnx__Conv.html). `NOTSET`/``""``
    (`_ConvSpatialAttrs.auto_pad`) is already a fixed, per-node quantity --
    `attrs.pad_top`/`pad_left`/`pad_bottom`/`pad_right` themselves, returned
    unchanged, no `in_h`/`in_w` dependence at all. ``VALID`` is always zero
    padding, likewise independent of `in_h`/`in_w`. ``SAME_UPPER``/
    ``SAME_LOWER`` are the only genuinely input-size-dependent cases:
    ``pad_total`` is chosen so that ``ceil(in / stride)`` output positions
    exactly cover the (possibly-dilated) kernel, split evenly between the
    two edges with the extra odd unit going to whichever edge the mode
    names (``SAME_UPPER`` -> the trailing edge gets the extra unit,
    ``SAME_LOWER`` -> the leading edge does) -- called fresh for every
    calibration batch (never cached on `_ConvSpatialAttrs` itself) since a
    dynamic-input-shape model's own `in_h`/`in_w` can legitimately differ
    batch to batch, changing `pad_total` (via `ceil`) along with it.
    """
    if attrs.auto_pad in ("NOTSET", ""):
        return attrs.pad_top, attrs.pad_left, attrs.pad_bottom, attrs.pad_right
    if attrs.auto_pad == "VALID":
        return 0, 0, 0, 0
    eff_kh = (attrs.kh - 1) * attrs.dilation_h + 1
    eff_kw = (attrs.kw - 1) * attrs.dilation_w + 1
    out_h = -(-in_h // attrs.stride_h)  # ceil division
    out_w = -(-in_w // attrs.stride_w)
    pad_h = max(0, (out_h - 1) * attrs.stride_h + eff_kh - in_h)
    pad_w = max(0, (out_w - 1) * attrs.stride_w + eff_kw - in_w)
    if attrs.auto_pad == "SAME_UPPER":
        pad_top, pad_left = pad_h // 2, pad_w // 2
    else:  # SAME_LOWER
        pad_top, pad_left = pad_h - pad_h // 2, pad_w - pad_w // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


def _conv_patch_sq_sum(
    x: np.ndarray, attrs: _ConvSpatialAttrs
) -> Tuple[Optional[np.ndarray], int]:
    """Sum of squares of the im2col-unfolded activation patch value at
    every ``(in_channel, kh, kw)`` receptive-field offset -- Wanda's
    ``||X_j||_2`` statistic, generalized from "input feature ``j``" (a
    MatMul/Gemm column) to "receptive-field offset ``j``" (a Conv column
    of the reshaped ``[out_channels, in_channels*kH*kW]`` weight, see this
    module's own docstring) -- reduced over the batch and every output
    spatial position, for one calibration batch's raw ``[N, Cin, H, W]``
    Conv input `x`. Returns ``(sq_sum, count)`` with `sq_sum` shaped
    ``[Cin, kh, kw]`` (flattening it in that order matches
    :func:`_prune_weight`'s own ``w.reshape(n, cin * kh * kw)``), or
    ``(None, 0)`` if `x` isn't a plausible 4-D NCHW activation for this
    Conv's own kernel once padded (too small, or not rank-4 at all --
    the same "no usable calibration signal" case
    :func:`apply_wanda_pruning`'s MatMul/Gemm branch already declines
    with a plain ``x.ndim != 2`` check).

    Padding is resolved fresh from `x`'s own spatial size for every call
    (:func:`_resolve_conv_pads` -- a no-op lookup of `attrs`' own fixed
    ``pads`` for an explicit/``NOTSET`` Conv, the actual ``auto_pad``
    formula otherwise). Each of the ``kh*kw`` receptive-field taps is then
    extracted as its own dedicated strided slice of the zero-padded input
    -- tap ``(i, j)`` is ``xp[:, :, i*dilation_h :: stride_h, j*dilation_w
    :: stride_w]`` (bounded to `h_out`/`w_out` positions) -- rather than
    via ``numpy.lib.stride_tricks.sliding_window_view``'s unit-offset
    window (correct only for unit dilation): a dilated tap's ``kh``/``kw``
    offsets aren't evenly spaced by 1 in the padded input, only by
    `dilation_h`/`dilation_w`, which this per-tap slicing accounts for
    directly. Collapses to the previous ``sliding_window_view``-based
    unfolding's exact numeric result when ``dilation_h == dilation_w ==
    1`` (verified by ``test_conv_patch_sq_sum_matches_naive_nested_loop_oracle``,
    which never sets a non-default dilation) -- ``kh*kw`` dedicated slices
    instead of one vectorized window is no more total work asymptotically
    (`kh*kw` is exactly how many taps a real Conv itself reads too), just
    reorganized to make the per-tap offset explicit.
    """
    if x.ndim != 4:
        return None, 0
    n, cin, in_h, in_w = x.shape
    pad_top, pad_left, pad_bottom, pad_right = _resolve_conv_pads(attrs, in_h, in_w)
    xp = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
    eff_kh = (attrs.kh - 1) * attrs.dilation_h + 1
    eff_kw = (attrs.kw - 1) * attrs.dilation_w + 1
    if xp.shape[2] < eff_kh or xp.shape[3] < eff_kw:
        return None, 0
    h_out = (xp.shape[2] - eff_kh) // attrs.stride_h + 1
    w_out = (xp.shape[3] - eff_kw) // attrs.stride_w + 1
    count = n * h_out * w_out
    if count == 0:
        return None, 0
    sq_sum = np.zeros((cin, attrs.kh, attrs.kw), dtype=x.dtype)
    for i in range(attrs.kh):
        h_start = i * attrs.dilation_h
        h_stop = h_start + attrs.stride_h * (h_out - 1) + 1
        for j in range(attrs.kw):
            w_start = j * attrs.dilation_w
            w_stop = w_start + attrs.stride_w * (w_out - 1) + 1
            tap = xp[
                :,
                :,
                h_start : h_stop : attrs.stride_h,
                w_start : w_stop : attrs.stride_w,
            ]
            sq_sum[:, i, j] = np.sum(np.square(tap), axis=(0, 2, 3))
    return sq_sum, count


def _conv_group_relative_norm(
    norm_flat: np.ndarray, cout: int, cin_per_group: int, kh: int, kw: int, group: int
) -> Optional[np.ndarray]:
    """Expands a Conv layer's flat per-``(in_channel, kh, kw)``-offset
    activation norm (:func:`_conv_patch_sq_sum`'s accumulated statistic,
    ``[Cin, kh, kw]`` flattened to length ``Cin*kh*kw`` -- `Cin` here being
    the raw, *full* input channel count :func:`_conv_patch_sq_sum` unfolds
    from, i.e. ``cin_per_group * group``) into a ``[out_channels,
    in_channels/group * kh * kw]`` array matching the shape of that Conv's
    own reshaped weight (:func:`_prune_weight`'s ``w.reshape(n,
    cin*kh*kw)``) -- one row per output filter, ready to multiply
    elementwise against ``|W|``.

    This is the piece that makes Wanda's Conv support correct for a
    grouped/depthwise Conv rather than silently wrong for every group but
    the first: `norm_flat` is computed once from the *raw, full-channel*
    input (an ordinary Conv's `X`, `_conv_patch_sq_sum` never sees `group`
    at all), but output filter ``i`` of a grouped Conv only ever reads its
    *own* group's global input-channel slice, ``[g * cin_per_group, (g+1)
    * cin_per_group)`` where ``g = i // (out_channels/group)`` (ONNX's
    grouped-Conv weight layout: the first ``out_channels/group`` filters
    belong to group 0, the next block to group 1, and so on) -- so "local
    receptive-field offset ``j``" means a *different* global input channel
    depending on the filter's own group, and reusing one shared
    (group-0-shaped) norm row for every filter would silently score every
    other group's filters against the wrong channels' activation
    statistics. This function slices `norm_flat`'s ``[Cin, kh, kw]`` shape
    along its channel axis once per group and repeats that group's own
    slice across exactly the filter rows that belong to it, so each row
    the caller gets back already carries its own filter's own group's
    statistic -- the "unfold once, select each filter's own group-relative
    channel slice" approach (see this module's own docstring), avoiding a
    second, per-group im2col unfold of the same input.

    For an ordinary (``group=1``) Conv this collapses to the previous
    behavior exactly: one group spanning every channel, broadcast to every
    output filter row identically. Returns ``None`` if `norm_flat`'s length
    doesn't match ``cin_per_group * group * kh * kw`` -- the "no usable
    calibration signal" case, same as this module's other norm-shape
    checks (e.g. a probe whose captured input channel count doesn't match
    the weight this Conv node claims, an already-declined-elsewhere kind
    of malformed model this function simply doesn't guess at either).
    """
    cin_full = cin_per_group * group
    if norm_flat.shape[0] != cin_full * kh * kw:
        return None
    norm_full = norm_flat.reshape(cin_full, kh, kw)
    filters_per_group = cout // group
    rows = np.empty((cout, cin_per_group * kh * kw), dtype=norm_flat.dtype)
    for g in range(group):
        block = norm_full[g * cin_per_group : (g + 1) * cin_per_group].reshape(-1)
        rows[g * filters_per_group : (g + 1) * filters_per_group, :] = block
    return rows


def apply_magnitude_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    global_sparsity: bool = False,
) -> onnx.ModelProto:
    """Zeros the least-magnitude entries of every MatMul/vanilla-Gemm
    layer's constant 2-D float32/float16/bfloat16 weight (this includes
    ``com.microsoft::GroupQueryAttention``'s separate Q/K/V projections,
    ordinary MatMul/Gemm nodes in their own right), every 2-D ``Conv``
    layer's constant 4-D float32/float16/bfloat16 weight -- ordinary
    (``group=1``), depthwise, and general grouped Conv alike, see this
    module's own docstring for why grouping needs no special-casing for
    this technique -- and every ``com.microsoft::Attention`` node's
    constant 2-D float32/float16/bfloat16 merged QKV weight -- the
    data-free pruning baseline (Han et al., 2015). A matched layer's
    weight is read out upcast to float64 for the importance/masking math
    below and the result cast back down to that layer's own original
    dtype before being written back (see the "FP16/BFloat16 weight
    support" section comment above :func:`_match_conv_weight_only`), so a
    fp16/bf16 model's declared dtypes are preserved exactly -- masking
    never changes a surviving entry's own value, only zeros dropped ones,
    so this round trip reproduces every kept entry's exact original bit
    pattern. See this module's own docstring for how importance is grouped
    (including the Conv reshape convention), why the merged QKV weight
    needs no special handling here beyond matching it (:func:`_candidates`,
    via :func:`_match_attention_weight_only`), and why structured
    (shape-changing) pruning isn't offered here.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each row's (or, for Conv, each
            output filter's) entries to zero, ignored when ``n``/``m`` are
            given -- or, when `global_sparsity`, target fraction of every
            matched layer's entries *combined*
    :param n: keep the ``n`` highest-magnitude entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4). Must be given
            together with ``m``; incompatible with `global_sparsity` (see
            below).
    :param m: group size for N:M pruning; see ``n``
    :param global_sparsity: the classic "global magnitude pruning" variant
            (Han et al., 2015) instead of this function's own default
            per-layer-uniform mode: rather than every matched layer
            independently zeroing the same *fraction* of its own entries
            (:func:`_sparsity_mask`'s own per-row rule, applied
            independently per layer), every matched layer's ``|W|``
            entries are first pooled into one ranking across the *whole*
            model, a single keep-count is chosen from `sparsity`'s
            fraction of that pooled total, and exactly that many
            lowest-magnitude entries are zeroed -- wherever in the model
            they land. A layer whose weights are uniformly small relative
            to the rest of the model ends up pruned harder than one whose
            weights are uniformly large, rather than every layer being cut
            by the same fraction regardless of its own weights' scale.
            Unlike the default mode, no per-row (or even per-layer) floor
            is enforced -- an entire row, or an entire layer's weight, can
            legitimately end up all-zero if every one of its entries
            scores below the one shared global cutoff; this is expected
            behavior for a genuinely global threshold, not a bug (a
            structural channel count can't drop to zero the same way, which
            is why :func:`apply_structured_pruning`'s own `global_sparsity`
            mode *does* keep a per-chain floor -- see that function's own
            docstring). Incompatible with ``n``/``m``: N:M's own per-group
            pattern already fixes a uniform local keep-count within every
            group of ``m`` columns, leaving no separate global-fraction
            target for a pooled threshold to redistribute. Default
            ``False`` -- every pre-existing caller's behavior is unchanged.
    :returns: ``model`` with every matched layer's weight zeroed in place
            to the target pattern; layers with a non-constant weight, a
            non-2-D MatMul/Gemm weight, or a non-4-D Conv weight are left
            untouched
    """
    _validate_pattern(sparsity, n, m)
    if global_sparsity and n is not None:
        raise ValueError(
            "global_sparsity is not supported together with N:M pruning "
            "(n/m) -- see apply_magnitude_pruning's own docstring"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    initializer_map = {t.name: t for t in out.graph.initializer}

    candidates = _candidates(out.graph)

    if global_sparsity:
        entries = []
        for _, _, w_name, weight_transposed, is_conv in candidates:
            w_init = initializer_map[w_name]
            w = _to_f64(w_init)
            w_nk = _weight_to_nk(w, weight_transposed, is_conv)
            entries.append(
                (w_init, weight_transposed, is_conv, w.shape, w_nk, np.abs(w_nk))
            )
        _apply_global_unstructured_pruning(entries, sparsity)
        return out

    for _, _, w_name, weight_transposed, is_conv in candidates:
        w_init = initializer_map[w_name]

        def importance_of_nk(w_nk, n=n, m=m, sparsity=sparsity):
            importance = np.abs(w_nk)
            return (
                _nm_mask(importance, n, m)
                if n is not None
                else _sparsity_mask(importance, sparsity)
            )

        _prune_weight(w_init, weight_transposed, importance_of_nk, is_conv=is_conv)

    return out


def _wanda_norm_for_candidate(
    node: onnx.NodeProto,
    x_name: str,
    w_name: str,
    is_conv: bool,
    w_init: onnx.TensorProto,
    act_norm: Dict[str, np.ndarray],
    conv_act_norm: Dict[str, np.ndarray],
    attn_act_norm: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Returns one matched layer's own broadcastable activation-norm array
    -- shape ``(1, K)`` for MatMul/Gemm/Attention (one shared norm row for
    every output channel), or ``(out_channels, K)`` for Conv, already
    expanded per output filter's own group by
    :func:`_conv_group_relative_norm` -- or ``None`` if no calibration
    activation was ever observed for it. Factored out of
    :func:`apply_wanda_pruning`'s own per-layer masking loop so its
    `global_sparsity` mode (see that function's own docstring) can compute
    the exact same per-layer norm before pooling every layer's importance
    into one global threshold, with the actual importance/fallback formula
    (:func:`_wanda_importance`) applied identically by both modes.
    """
    if is_conv:
        flat_norm = conv_act_norm.get(w_name)
        if flat_norm is None:
            return None
        cout, cin_per_group, kh, kw = (int(d) for d in w_init.dims)
        return _conv_group_relative_norm(
            flat_norm, cout, cin_per_group, kh, kw, _conv_group(node)
        )
    if node.domain == _ATTENTION_DOMAIN and node.op_type == "Attention":
        # See `apply_wanda_pruning`'s own `attn_sq_sum` accumulation: this
        # weight's own activation is the rank-3 `X` reduced over leading
        # axes, keyed by `w_name` rather than `x_name`.
        norm_flat = attn_act_norm.get(w_name)
        return norm_flat[np.newaxis, :] if norm_flat is not None else None
    norm_flat = act_norm.get(x_name)
    return norm_flat[np.newaxis, :] if norm_flat is not None else None


def _wanda_importance(
    w_nk: np.ndarray, norm: Optional[np.ndarray], epsilon: float
) -> np.ndarray:
    """Wanda's own ``|W_ij| * ||X_j||_2`` importance for one already-``[N,
    K]``-shaped weight (:func:`_weight_to_nk`), given its own broadcastable
    activation-norm array (:func:`_wanda_norm_for_candidate`) -- falls back
    to plain magnitude (:func:`apply_magnitude_pruning`'s own metric) when
    `norm` is missing or its shape doesn't line up with `w_nk` (no
    calibration activation was ever observed for this layer, or it was
    observed at a mismatched width). Shared between
    :func:`apply_wanda_pruning`'s own per-layer and `global_sparsity`
    masking paths so both compute exactly the same per-entry importance.
    """
    if (
        norm is None
        or norm.shape[-1] != w_nk.shape[1]
        or (norm.shape[0] != 1 and norm.shape[0] != w_nk.shape[0])
    ):
        return np.abs(w_nk)  # fall back to plain magnitude
    return np.abs(w_nk) * np.maximum(norm, epsilon)


def _wanda_unstructured_calibration_stats(
    out: onnx.ModelProto,
    candidates,
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Runs `out` over `calibration_data` and returns the three per-candidate
    activation-norm dicts (`act_norm`, `conv_act_norm`, `attn_act_norm`) that
    :func:`_wanda_norm_for_candidate`/:func:`_wanda_importance` consume --
    the same three keyed exactly as :func:`apply_wanda_pruning`'s own body
    used to compute them inline before this function existed. Factored out,
    read-only (never mutates `out`), so
    :func:`analyze_pruning_sensitivity`'s own dry-run report can compute the
    *exact* same Wanda importance :func:`apply_wanda_pruning` would, from one
    single shared implementation, rather than a second hand-copied version
    that could silently drift from this one.

    `out` is expected to already be the caller's own working copy (as
    :func:`apply_wanda_pruning` always passes its own `out`, never the
    caller's original `model`) -- this function itself never copies it,
    only adds probe outputs to a further internal copy of it via
    :func:`_add_probe_outputs`, whose own docstring establishes that it
    doesn't mutate what's passed to it either.
    """
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    probe_names = sorted({x_name for _, x_name, _, _, _ in candidates})
    probe_model = _add_probe_outputs(out, probe_names)

    # Conv attributes are per-node (two Convs can share an input tensor
    # with different kernels/strides), so the per-node Conv statistic is
    # keyed by its own weight name, not the (possibly shared) input name
    # the plain MatMul/Gemm statistic below is keyed by.
    conv_attrs: Dict[str, Optional[_ConvSpatialAttrs]] = {
        w_name: _conv_spatial_attrs(node, initializer_map[w_name])
        for node, _, w_name, _, is_conv in candidates
        if is_conv
    }

    sq_sum: Dict[str, np.ndarray] = {}
    count: Dict[str, int] = {}
    conv_sq_sum: Dict[str, np.ndarray] = {}
    conv_count: Dict[str, int] = {}
    # `Attention`'s merged QKV weight is the one candidate whose own `X` is
    # *always* rank-3 (`[batch, seq, hidden]`), not the plain 2-D tensor the
    # `sq_sum`/`act_norm` probe above requires -- so on its own it would
    # always fall into that probe's `x.ndim != 2: continue` and fall back to
    # plain magnitude (see this function's own docstring history). Rather
    # than generalizing that check to reduce over leading axes for *every*
    # candidate -- which would also silently change the already-tested
    # fallback behavior of any ordinary MatMul/Gemm layer whose activation
    # happens to be rank 3+ too, a strictly bigger change than this one
    # layer type needs -- this accumulates a second, Attention-only
    # statistic, gated on the node itself (domain + op_type, exactly
    # :func:`_match_attention_producer`'s own check) rather than on activation
    # shape alone, and keyed by `w_name` (mirroring the per-node Conv
    # statistic above, and for the same reason: two Attention nodes could in
    # principle share an `x_name`). Reduces over every leading axis via
    # `x.reshape(-1, x.shape[-1])`, mirroring
    # :func:`apply_sparsegpt_pruning`'s own `H` accumulation for this same
    # weight.
    attn_sq_sum: Dict[str, np.ndarray] = {}
    attn_count: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim != 2:
                continue
            s = np.square(x).sum(axis=0)
            sq_sum[name] = s if name not in sq_sum else sq_sum[name] + s
            count[name] = count.get(name, 0) + x.shape[0]
        for node, x_name, w_name, _, is_conv in candidates:
            if is_conv:
                attrs = conv_attrs[w_name]
                if attrs is None:
                    continue
                x = np.asarray(result[x_name], dtype=np.float64)
                s, cnt = _conv_patch_sq_sum(x, attrs)
                if s is None:
                    continue
                conv_sq_sum[w_name] = (
                    s if w_name not in conv_sq_sum else conv_sq_sum[w_name] + s
                )
                conv_count[w_name] = conv_count.get(w_name, 0) + cnt
                continue
            if node.domain != _ATTENTION_DOMAIN or node.op_type != "Attention":
                continue
            x = np.asarray(result[x_name], dtype=np.float64)
            if x.ndim < 2:
                continue
            x_flat = x.reshape(-1, x.shape[-1])
            s = np.square(x_flat).sum(axis=0)
            attn_sq_sum[w_name] = (
                s if w_name not in attn_sq_sum else attn_sq_sum[w_name] + s
            )
            attn_count[w_name] = attn_count.get(w_name, 0) + x_flat.shape[0]

    act_norm: Dict[str, np.ndarray] = {
        name: np.sqrt(s / max(count[name], 1)) for name, s in sq_sum.items()
    }
    conv_act_norm: Dict[str, np.ndarray] = {
        name: np.sqrt(s / max(conv_count[name], 1)).reshape(-1)
        for name, s in conv_sq_sum.items()
    }
    attn_act_norm: Dict[str, np.ndarray] = {
        name: np.sqrt(s / max(attn_count[name], 1)) for name, s in attn_sq_sum.items()
    }
    return act_norm, conv_act_norm, attn_act_norm


def apply_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
    global_sparsity: bool = False,
) -> onnx.ModelProto:
    """Wanda pruning (Sun et al., 2023): zeros the least-important entries
    of every MatMul/vanilla-Gemm layer's constant 2-D float32/float16/
    bfloat16 weight (this includes ``com.microsoft::GroupQueryAttention``'s
    separate Q/K/V projections, ordinary MatMul/Gemm nodes in their own
    right), every 2-D ``Conv`` layer's constant 4-D float32/float16/
    bfloat16 weight -- ordinary (``group=1``), depthwise, and general
    grouped Conv alike -- and every ``com.microsoft::Attention`` node's
    constant 2-D float32/float16/bfloat16 merged QKV weight (see the
    "FP16/BFloat16 weight support" section comment above
    :func:`_match_conv_weight_only` for the read-upcast/write-downcast
    pattern this applies at every matched weight; calibration activations
    are captured via a real ``onnxruntime`` run and were already cast to
    float64 regardless of the graph's own declared dtype, so they need no
    separate handling here), using ``|W_ij| * ||X_j||_2`` (weight magnitude
    times its
    reduction-dimension entry's activation norm over calibration data) as
    the importance metric instead of plain ``|W|``. See this module's own
    docstring for the technique -- including what ``X_j`` means for a Conv
    column (one ``(in_channel, kh, kw)`` receptive-field offset, not a
    whole input channel), how a grouped/depthwise Conv's activation norm is
    kept group-relative rather than shared across every filter
    (:func:`_conv_group_relative_norm`), which Conv attribute combinations
    this confidently handles, and how ``Attention``'s merged weight -- whose
    own activation input is rank-3 (``[batch, seq, hidden]``), not the plain
    2-D tensor the shared MatMul/Gemm probe requires -- gets its own,
    separately-accumulated activation statistic (reduced over every leading
    axis, mirroring :func:`apply_sparsegpt_pruning`'s own
    ``x.reshape(-1, x.shape[-1])``) so it is calibrated too, without loosening
    the plain-2-D-only check every other MatMul/Gemm layer's activation still
    goes through -- and :func:`apply_magnitude_pruning` for the
    calibration-free baseline this upgrades.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            input channel's (or, for Conv, each receptive-field offset's)
            activation norm on. Each batch is a ``{input_name: np.ndarray}``
            dict matching ``model``'s graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each row's (or, for Conv, each
            output filter's) entries to zero, ignored when ``n``/``m`` are
            given
    :param n: keep the ``n`` highest-importance entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4). Must be given
            together with ``m``; incompatible with `global_sparsity` (see
            below).
    :param m: group size for N:M pruning; see ``n``
    :param epsilon: floor applied to the accumulated per-channel activation
            norm, avoiding every entry of an all-zero channel tying at
            exactly-zero importance
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :param global_sparsity: pools every matched layer's own ``|W_ij| *
            ||X_j||_2`` importance into one ranking across the *whole*
            model and picks a single keep-count from `sparsity`'s fraction
            of that pooled total, mirroring
            :func:`apply_magnitude_pruning`'s own `global_sparsity` mode
            (see that function's own docstring for the full mechanism and
            why no per-row/per-layer floor is enforced). Caveat specific
            to Wanda's own metric, honestly noted rather than hidden:
            ``|W_ij| * ||X_j||_2`` is not scale-normalized across layers
            the way a probability or a fraction would be -- its raw
            numeric range depends on that layer's own weight-init/training
            scale *and* on the raw activation magnitude flowing into it
            (which varies enormously with position in the network: right
            after an embedding lookup vs. right after a LayerNorm vs. right
            after a GELU are all different scales, with no relationship to
            genuine importance). A pooled ranking is still exactly the
            mechanical "one global threshold" this mode promises -- and,
            unlike a metric this module has no way to compute at all, this
            one is at least well-defined and reproducible -- but in
            practice it can end up dominated by whichever layers happen to
            see the largest raw activation norms, pruning them harder (or
            leaving them untouched) for reasons unrelated to how much they
            actually matter. Callers who want cross-layer redistribution
            with a metric that *is* scale-comparable should prefer
            :func:`apply_magnitude_pruning`'s own `global_sparsity` mode
            (plain ``|W|`` has the same cross-layer weight-scale caveat in
            principle, but no *additional* activation-scale one on top of
            it). Incompatible with ``n``/``m``, for the same reason
            :func:`apply_magnitude_pruning` gives. Default ``False`` --
            every pre-existing caller's behavior is unchanged.
    :returns: ``model`` with every matched layer's weight zeroed in place
            to the target pattern; a MatMul/Gemm layer with a non-constant
            or non-2-D weight, a Conv layer with a non-4-D weight, or any
            matched layer whose activation input isn't usable (not a plain
            2-D tensor for MatMul/Gemm; not a rank-2+ tensor for Attention
            (its own ``X`` input is always rank-3 in practice, reduced over
            every leading axis); not a 4-D NCHW tensor, or a malformed Conv
            attribute combination :func:`_conv_spatial_attrs` declines --
            e.g. a `kernel_shape` disagreeing with the weight's own shape --
            for Conv; `auto_pad`/non-unit `dilations` are handled, not
            declined) falls back to plain magnitude pruning (no activation
            norm was ever observed)
    """
    _validate_pattern(sparsity, n, m)
    if global_sparsity and n is not None:
        raise ValueError(
            "global_sparsity is not supported together with N:M pruning "
            "(n/m) -- see apply_wanda_pruning's own docstring"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}

    candidates = _candidates(graph)
    if not candidates:
        return out

    act_norm, conv_act_norm, attn_act_norm = _wanda_unstructured_calibration_stats(
        out, candidates, calibration_data, providers
    )

    if global_sparsity:
        entries = []
        for node, x_name, w_name, weight_transposed, is_conv in candidates:
            w_init = initializer_map[w_name]
            w = _to_f64(w_init)
            w_nk = _weight_to_nk(w, weight_transposed, is_conv)
            norm = _wanda_norm_for_candidate(
                node,
                x_name,
                w_name,
                is_conv,
                w_init,
                act_norm,
                conv_act_norm,
                attn_act_norm,
            )
            importance = _wanda_importance(w_nk, norm, epsilon)
            entries.append(
                (w_init, weight_transposed, is_conv, w.shape, w_nk, importance)
            )
        _apply_global_unstructured_pruning(entries, sparsity)
        return out

    for node, x_name, w_name, weight_transposed, is_conv in candidates:
        w_init = initializer_map[w_name]
        # `norm` is always kept 2-D here, broadcastable elementwise against
        # `w_nk` ([out_channels, K]) inside importance_of_nk below: shape
        # (1, K) for a MatMul/Gemm layer (one shared norm row for every
        # output channel, the same broadcast the plain
        # ``norm[np.newaxis, :]`` used to do directly), or -- for Conv --
        # shape (out_channels, K), already expanded per output filter's own
        # group by :func:`_conv_group_relative_norm` (trivially identical
        # across every row when ``group=1``, genuinely different per group
        # otherwise -- see that function's own docstring for why a single
        # shared row would be wrong for a grouped/depthwise Conv).
        norm = _wanda_norm_for_candidate(
            node,
            x_name,
            w_name,
            is_conv,
            w_init,
            act_norm,
            conv_act_norm,
            attn_act_norm,
        )

        def importance_of_nk(w_nk, norm=norm, n=n, m=m, sparsity=sparsity):
            importance = _wanda_importance(w_nk, norm, epsilon)
            return (
                _nm_mask(importance, n, m)
                if n is not None
                else _sparsity_mask(importance, sparsity)
            )

        _prune_weight(w_init, weight_transposed, importance_of_nk, is_conv=is_conv)

    return out


def weight_sparsity(model: Union[str, onnx.ModelProto]) -> float:
    """Fraction of exact-zero entries across every matched MatMul/vanilla-
    Gemm layer (including ``com.microsoft::GroupQueryAttention``'s separate
    Q/K/V projections), 2-D Conv layer -- ordinary (``group=1``), depthwise,
    and general grouped alike, since :func:`_candidates`'s default
    `allow_grouped_conv` matches all three -- or ``com.microsoft::Attention``
    merged-QKV-weight layer's constant weight -- a quick way to confirm a
    pruning call reached its target, or to measure an already-sparse model.
    Shares :func:`_candidates` with every ``apply_*_pruning`` function
    above, so it automatically reports across whatever layer types they
    match (including FLOAT16/BFLOAT16 weights, needing no dtype-specific
    handling here: an exact-zero comparison and ``.size`` mean the same
    thing regardless of float precision), with no separate list to keep in
    sync.
    Returns ``0.0`` if no matching layer is present.
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    zeros = 0
    total = 0
    initializer_map = {t.name: t for t in model.graph.initializer}
    for _, _, w_name, _, _ in _candidates(model.graph):
        w = onnx.numpy_helper.to_array(initializer_map[w_name])
        zeros += int(np.count_nonzero(w == 0))
        total += w.size

    return zeros / total if total else 0.0


# --- SparseGPT ----------------------------------------------------------


def _sparsegpt_prune_columns(
    w_nk: np.ndarray,
    h: np.ndarray,
    sparsity: float,
    n: Optional[int],
    m: Optional[int],
    percdamp: float,
    proc_block_size: int,
) -> np.ndarray:
    """Returns SparseGPT-pruned values for ``w_nk`` ([N, K], output channel
    first), a direct port of the reference implementation's own
    ``fasterprune`` (https://github.com/IST-DASLab/sparsegpt/blob/master/
    sparsegpt.py). Unlike :func:`_prune_weight`'s ``importance_of_nk``
    callbacks, this returns fully-formed replacement values, not a mask --
    every *kept* entry may also change, having accumulated Hessian-based
    compensation for every *pruned* entry processed before it.
    """
    n_rows, k = w_nk.shape
    diag = np.arange(k)
    dead = h[diag, diag] == 0.0

    w_work = w_nk.copy()
    w_work[:, dead] = 0.0
    w_pruned = np.zeros_like(w_work)

    if n is None and sparsity <= 0.0:
        return w_nk.copy()  # true no-op, rather than the reference's own
        # "always drop the single lowest-scoring entry" edge case at
        # sparsity == 0.0 -- matching every other apply_*_pruning function
        # in this module, all of which treat sparsity=0.0 as a no-op.

    hinv = _inverse_hessian_cholesky(h, percdamp)

    for i1 in range(0, k, proc_block_size):
        i2 = min(i1 + proc_block_size, k)
        count = i2 - i1
        w1 = w_work[:, i1:i2].copy()
        err1 = np.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        hinv1_diag = np.diag(hinv1)

        if n is None:
            score = np.square(w1) / np.square(hinv1_diag)[np.newaxis, :]
            thresh = np.sort(score.reshape(-1))[int(score.size * sparsity)]
            mask1 = score <= thresh
        else:
            mask1 = np.zeros_like(w1, dtype=bool)

        for i in range(count):
            if n is not None and m is not None and i % m == 0:
                group_end = min(i + m, count)
                group_score = (
                    np.square(w1[:, i:group_end])
                    / np.square(hinv1_diag[i:group_end])[np.newaxis, :]
                )
                prune_count = min(group_end - i, m - n)
                mask1[:, i:group_end] = False
                if prune_count > 0:
                    drop_local = np.argsort(group_score, axis=1)[:, :prune_count]
                    np.put_along_axis(mask1[:, i:group_end], drop_local, True, axis=1)

            w_col = w1[:, i]
            d = hinv1_diag[i]
            q_col = np.where(mask1[:, i], 0.0, w_col)
            w_pruned[:, i1 + i] = q_col

            err = (w_col - q_col) / d
            err1[:, i] = err
            if i + 1 < count:
                w1[:, i + 1 :] -= np.outer(err, hinv1[i, i + 1 :])

        if i2 < k:
            w_work[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    return w_pruned


def _conv_im2col_patches(
    x: np.ndarray, attrs: _ConvSpatialAttrs
) -> Optional[np.ndarray]:
    """Returns the ``[n_positions, Cin*kh*kw]`` im2col-unfolded patch
    matrix for one calibration batch's raw ``[N, Cin, H, W]`` Conv input
    `x` -- every output spatial position's full receptive-field patch,
    flattened in the same ``(in_channel, kh, kw)`` row-major order
    :func:`_prune_weight`'s own ``w.reshape(n, cin*kh*kw)`` uses (verified
    against a nested-loop oracle by
    ``test_sparsegpt_conv_hessian_matches_naive_nested_loop_oracle``).
    SparseGPT's Conv Hessian is ``H = patches.T @ patches``, this
    function's own return value being the only new piece: everything else
    (the zero-padded, per-tap strided-slice unfolding that also handles
    ``auto_pad``/non-unit ``dilations`` -- see :func:`_resolve_conv_pads`
    -- and the attribute handling) mirrors :func:`_conv_patch_sq_sum`
    exactly, reusing the same :class:`_ConvSpatialAttrs`/
    :func:`_conv_spatial_attrs` Wanda's own Conv support already built --
    see this module's own docstring for why a *diagonal-only* per-offset
    norm (Wanda's ``_conv_patch_sq_sum``) is not enough here and the
    *full* cross-covariance this returns is needed instead. Returns
    ``None`` on the same "not usable" conditions :func:`_conv_patch_sq_sum`
    declines (not a rank-4 activation, or too small once padded for this
    kernel).

    For a grouped/depthwise Conv, :func:`apply_sparsegpt_pruning` calls
    this once per group on `x` already sliced to that group's own global
    input-channel range (``x[:, g*Cin/group:(g+1)*Cin/group, :, :]``) --
    this function itself carries no notion of `group` at all, and needs
    none: it only ever reads `x`'s own channel count via ``x.shape[1]``, so
    a channel-sliced sub-tensor is unfolded exactly the same way a smaller
    "whole" input would be. See this module's own docstring for why that
    is the correct per-group Hessian rather than an approximation of one.
    """
    if x.ndim != 4:
        return None
    n, cin, in_h, in_w = x.shape
    pad_top, pad_left, pad_bottom, pad_right = _resolve_conv_pads(attrs, in_h, in_w)
    xp = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
    eff_kh = (attrs.kh - 1) * attrs.dilation_h + 1
    eff_kw = (attrs.kw - 1) * attrs.dilation_w + 1
    if xp.shape[2] < eff_kh or xp.shape[3] < eff_kw:
        return None
    h_out = (xp.shape[2] - eff_kh) // attrs.stride_h + 1
    w_out = (xp.shape[3] - eff_kw) // attrs.stride_w + 1
    n_positions = n * h_out * w_out
    if n_positions == 0:
        return None
    # [N, Hout, Wout, Cin, kh, kw]: every output position on the leading
    # axes, (in_channel, kh, kw) on the trailing ones, so the final
    # reshape's row-major flatten of those trailing axes matches
    # w.reshape(n, cin*kh*kw)'s own column order exactly -- filled in one
    # dedicated strided slice per (kh, kw) tap (see _conv_patch_sq_sum's
    # own docstring for why: correct under dilation, unlike a single
    # sliding_window_view call), each tap's own [N, Cin, Hout, Wout] slice
    # transposed to [N, Hout, Wout, Cin] before it's dropped into place.
    patches = np.empty((n, h_out, w_out, cin, attrs.kh, attrs.kw), dtype=x.dtype)
    for i in range(attrs.kh):
        h_start = i * attrs.dilation_h
        h_stop = h_start + attrs.stride_h * (h_out - 1) + 1
        for j in range(attrs.kw):
            w_start = j * attrs.dilation_w
            w_stop = w_start + attrs.stride_w * (w_out - 1) + 1
            tap = xp[
                :,
                :,
                h_start : h_stop : attrs.stride_h,
                w_start : w_stop : attrs.stride_w,
            ]
            patches[:, :, :, :, i, j] = np.transpose(tap, (0, 2, 3, 1))
    return patches.reshape(n_positions, cin * attrs.kh * attrs.kw)


def apply_sparsegpt_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    percdamp: float = 0.01,
    proc_block_size: int = 128,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """SparseGPT (Frantar & Alistarh, 2023): zeros the least-important
    entries of every MatMul/vanilla-Gemm layer's constant 2-D float32/
    float16/bfloat16 weight (this includes
    ``com.microsoft::GroupQueryAttention``'s separate Q/K/V projections,
    ordinary MatMul/Gemm nodes in their own right) and every
    ``com.microsoft::Attention`` node's constant 2-D float32/float16/
    bfloat16 merged QKV weight (read upcast to float64, written back down
    to that weight's own original dtype -- see the "FP16/BFloat16 weight
    support" section comment above :func:`_match_conv_weight_only` --
    though note SparseGPT's Hessian-compensated update, unlike plain
    masking, *recomputes* every kept entry's own value, so a fp16/bf16
    weight's surviving entries do not reproduce their pre-pruning bit
    pattern the way magnitude/Wanda pruning's pure zero-masking does; the
    float64 accumulation this function already used for numerical
    stability is unchanged either way), the same unstructured-or-N:M
    patterns
    :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning` offer, but
    -- unlike either -- using a sequential, Hessian-error-compensating
    algorithm ported from GPTQ (:mod:`onnxsim.gptq`, same authors, same
    Cholesky-factored inverse Hessian) rather than a one-shot static
    importance score. See this module's own docstring for the technique,
    including the one deliberate departure from every other function here:
    for unstructured sparsity, the pruning threshold is shared across every
    output row within each ``proc_block_size``-wide column block (the
    reference implementation's own behavior), not chosen per row.

    Also matches every 2-D ``Conv`` layer :func:`apply_magnitude_pruning`/
    :func:`apply_wanda_pruning` do -- ordinary (``group=1``), depthwise, and
    general grouped Conv alike (``_candidates(graph)``, `allow_grouped_conv`
    at its own default, ``True``) -- see this module's own docstring for the
    full ``[K, K]`` im2col cross-covariance Hessian this needed (materially
    more machinery than Wanda's per-offset norm, and -- unlike everything
    else this function ports from the reference implementation -- with no
    correct reference to work from at all: the official implementation's own
    ``add_batch`` (https://github.com/IST-DASLab/sparsegpt) never actually
    unfolds a ``nn.Conv2d`` activation, only reshapes the *weight*, since its
    own driver scripts never exercise a Conv layer), how it's verified, how
    a grouped/depthwise Conv gets a genuinely *per-group* Hessian and its
    own independent column-processing/error-compensation pass rather than
    one shared across every filter, and how each group's own ``H``
    accumulates batch by batch rather than ever materializing every
    calibration batch's unfolded patches at once. Every ``auto_pad``/
    ``dilations`` combination the ONNX Conv schema defines is handled (see
    :func:`_conv_spatial_attrs`/:func:`_resolve_conv_pads`); only a
    genuinely malformed Conv node (e.g. a `kernel_shape` disagreeing with
    the weight's own shape) is left completely untouched, same as a layer
    with no observed calibration activation at all -- there is still no
    data-free fallback for SparseGPT.

    ``Attention``'s merged QKV weight has no analogous gap and is
    deliberately matched here too (unconditionally -- see
    :func:`_candidates`'s own docstring): its own input is a plain
    ``[*, K]`` activation (the same ``X`` any ordinary MatMul reads), not an
    im2col-unfolded receptive field, so ``H = X^T X`` is exactly as correct
    for it as for every other MatMul/Gemm layer already matched here, with
    no new machinery needed. See this module's own docstring for the fuller
    reasoning, including why it would be inconsistent to exclude it while
    ``GroupQueryAttention``'s separate Q/K/V MatMuls remain (and always
    were) in scope.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to compute each
            layer's Hessian from. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of entries to zero (shared per column
            block, not per row -- see above), ignored when ``n``/``m`` are
            given
    :param n: keep the ``n`` highest-importance entries per group of ``m``
            (semi-structured N:M pruning, e.g. NVIDIA's 2:4, per-row exactly
            as :func:`apply_magnitude_pruning`). Must be given together
            with ``m``.
    :param m: group size for N:M pruning; see ``n``
    :param percdamp: Hessian damping factor (fraction of the mean diagonal
            added to every diagonal entry before inversion), matching
            :func:`onnxsim.apply_gptq`'s own default
    :param proc_block_size: column-processing block size -- both the
            lazy-update granularity (how many columns' errors accumulate
            locally before a full cross-block update, matching
            :func:`onnxsim.apply_gptq`'s ``proc_block_size``) and, for
            unstructured sparsity only, the width each shared per-block
            threshold is computed over
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched layer's weight rewritten in
            place to the target pattern -- every surviving entry may also
            change value, having accumulated compensation for entries
            pruned before it; a MatMul/Gemm layer with no observed 2-D
            calibration activation (dead input, or every batch's
            activation isn't plain 2-D/higher-rank-with-a-trailing-
            feature-axis), or a Conv layer with no observed usable 4-D
            activation (dead input, or a malformed attribute combination
            :func:`_conv_spatial_attrs` declines -- `auto_pad`/non-unit
            `dilations` are handled, not declined) *for any one
            of its groups* (a grouped/depthwise Conv is left completely
            untouched, not partially pruned, if even one group's own
            Hessian was never observed), is left completely untouched --
            unlike Wanda, there is no data-free fallback for a technique
            whose entire mechanism is the Hessian
    """
    _validate_pattern(sparsity, n, m)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}

    # Unlike an earlier version of this function, grouped/depthwise Conv is
    # matched too (_candidates' own allow_grouped_conv default, True) -- see
    # this function's own docstring and this module's own docstring for the
    # per-group Hessian/column-processing-loop partitioning that makes this
    # correct now.
    candidates = _candidates(graph)
    if not candidates:
        return out

    probe_names = sorted({x_name for _, x_name, _, _, _ in candidates})
    probe_model = _add_probe_outputs(out, probe_names)

    # Conv attributes are per-node (two Convs can share an input tensor
    # with different kernels/strides), so the Conv Hessian below is keyed
    # by its own weight name, mirroring apply_wanda_pruning's own
    # conv_attrs/conv_act_norm.
    conv_attrs: Dict[str, Optional[_ConvSpatialAttrs]] = {
        w_name: _conv_spatial_attrs(node, initializer_map[w_name])
        for node, _, w_name, _, is_conv in candidates
        if is_conv
    }
    # `group`/`in_channels_per_group` are fixed per node (read once, not
    # recomputed per batch) -- `cin_per_group` is exactly the weight's own
    # axis-1 extent, ONNX's grouped-Conv convention (see this module's own
    # docstring).
    conv_group_info: Dict[str, Tuple[int, int]] = {
        w_name: (_conv_group(node), int(initializer_map[w_name].dims[1]))
        for node, _, w_name, _, is_conv in candidates
        if is_conv
    }

    activations: Dict[str, List[np.ndarray]] = {
        x_name: [] for _, x_name, _, _, is_conv in candidates if not is_conv
    }
    # Unlike the MatMul/Gemm activations above (each layer's whole
    # calibration set concatenated once, below -- small enough per layer
    # to keep entirely in memory), each Conv layer's H accumulates
    # incrementally, one calibration batch's own im2col-unfolded patch
    # matrix at a time: a full [n_positions, K] patch matrix can be large,
    # so no batch's patches outlive the H += they fold into. See this
    # module's own docstring.
    #
    # conv_h[w_name] is a list of length `group`, one independently
    # accumulated H per group -- filter row i (belonging to group
    # i // (out_channels/group), ONNX's own grouped-Conv weight layout)
    # only ever reads its own group's global input-channel slice
    # [g*cin_per_group, (g+1)*cin_per_group), so group g's own H is built
    # by feeding exactly that channel-sliced sub-tensor through the same
    # per-group-agnostic _conv_im2col_patches used for the group=1 case --
    # no dedicated grouped-Hessian machinery needed, since im2col-unfolding
    # a channel slice is exactly what im2col-unfolding a narrower "whole"
    # input already does. For group=1 this is a length-1 list whose single
    # entry is built from the full (unsliced) input, identical to this
    # function's previous group=1-only behavior. Total unfolding work
    # across every group's own slice equals one full-input unfold (the
    # slices partition the channel axis), so this costs no more overall
    # than the group=1 case did.
    conv_h: Dict[str, List[Optional[np.ndarray]]] = {}

    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in activations:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim < 2:
                continue
            activations[name].append(x.reshape(-1, x.shape[-1]))
        for _, x_name, w_name, _, is_conv in candidates:
            if not is_conv:
                continue
            attrs = conv_attrs[w_name]
            if attrs is None:
                continue
            group, cin_per_group = conv_group_info[w_name]
            x_conv = np.asarray(result[x_name], dtype=np.float64)
            if x_conv.ndim != 4 or x_conv.shape[1] != cin_per_group * group:
                continue  # not this node's own [N, Cin, H, W] input -- skip
            h_accum = conv_h.setdefault(w_name, [None] * group)
            for g in range(group):
                x_group = x_conv[:, g * cin_per_group : (g + 1) * cin_per_group, :, :]
                patches = _conv_im2col_patches(x_group, attrs)
                if patches is None:
                    continue
                h_batch = patches.T @ patches
                h_accum[g] = h_batch if h_accum[g] is None else h_accum[g] + h_batch

    for _, x_name, w_name, weight_transposed, is_conv in candidates:
        w_init = initializer_map[w_name]
        w = _to_f64(w_init)

        if is_conv:
            cout, cin_per_group, kh, kw = w.shape
            group, _ = conv_group_info[w_name]
            h_list = conv_h.get(w_name)
            # Every group's own H must have been observed -- a layer with
            # no usable calibration signal for any one group is left
            # completely untouched, same as the group=1 case's `h is None`
            # check (there is no data-free fallback, and no meaningful way
            # to prune some groups' filters but not others.
            if h_list is None or any(h is None for h in h_list):
                continue
            filters_per_group = cout // group
            w_nk = w.reshape(cout, cin_per_group * kh * kw)
            # Each group's own [filters_per_group, K] weight sub-block is
            # pruned independently against its own group's H -- a column-
            # masking decision and its downstream error compensation only
            # make sense within one group's own consistent Hessian/weight
            # coordinate system (see this module's own docstring), so
            # _sparsegpt_prune_columns (already correct for one full,
            # ungrouped weight/Hessian pair) is simply called once per
            # group rather than needing any grouped-specific version of its
            # own sequential column-processing/error-compensation loop.
            pruned_groups = [
                _sparsegpt_prune_columns(
                    w_nk[g * filters_per_group : (g + 1) * filters_per_group],
                    h_list[g],
                    sparsity,
                    n,
                    m,
                    percdamp,
                    proc_block_size,
                )
                for g in range(group)
            ]
            w_pruned_nk = np.concatenate(pruned_groups, axis=0)
            w_new = w_pruned_nk.reshape(cout, cin_per_group, kh, kw)
        else:
            acts = activations[x_name]
            if not acts:
                continue
            x = np.concatenate(acts, axis=0)
            dim0, dim1 = w.shape
            w_nk = w if weight_transposed else w.T  # [N, K]
            if x.shape[1] != w_nk.shape[1]:
                continue

            h = x.T @ x
            w_pruned_nk = _sparsegpt_prune_columns(
                w_nk, h, sparsity, n, m, percdamp, proc_block_size
            )
            w_new = w_pruned_nk if weight_transposed else w_pruned_nk.T
            w_new = w_new.reshape(dim0, dim1)

        w_init.CopyFrom(_from_f64(w_new, w_init.data_type, w_init.name))

    return out


# --- Structured (channel) pruning -------------------------------------------

# Shape-preserving, channel-order-preserving elementwise ops that may sit
# between a producer and consumer without blocking the chain: unary
# activations (single input, single output, no other operand to worry
# about) and Add/Mul against a constant per-channel bias/scale.
_UNARY_PASS_THROUGH = {
    "Relu",
    "LeakyRelu",
    "Elu",
    "Selu",
    "Sigmoid",
    "Tanh",
    "Softplus",
    "Softsign",
    "Gelu",
    "HardSigmoid",
    "Mish",
    "Identity",
    "Cast",
    # com.microsoft::QuickGelu(X) = X * Sigmoid(alpha * X) (alpha an
    # attribute, default 1.702, not a second *input* -- confirmed against
    # onnxruntime's own schema, contrib_defs.cc, and by direct execution,
    # see this module's own docstring): a single-input, single-output,
    # purely elementwise activation exactly like every other entry in this
    # set, just from a different domain -- membership here is by op_type
    # alone for every entry, never by domain (a same-named op in an
    # unrelated custom domain has always been a theoretical risk this set
    # accepts, not one unique to this entry). Being unary, it needs no
    # dedicated hop machinery at all: adding it here alone already extends
    # every walker that already consults `_UNARY_PASS_THROUGH` --
    # `_walk_to_consumer`/`_walk_to_conv_consumer` (forward), their two
    # backward counterparts, *and* `_trace_gate_producer_backward`'s own
    # gated-pair gate-activation matcher -- for free.
    "QuickGelu",
}
_BINARY_CHANNEL_OPS = {"Add", "Mul"}
_MAX_CHAIN_HOPS = 8

# com.microsoft's fused bias-add + Gelu-family activation nodes -- the FFN
# analogue of the SkipLayerNormalization residual fusion above, done by the
# same onnxruntime transformer-optimizer tool: `BiasGelu(A, B) = Gelu(A + B)`
# (erf-based, exactly plain ONNX Gelu's own default `approximate="none"`)
# and `FastGelu(X[, bias]) = Gelu_tanh(X [+ bias])` (the tanh approximation,
# `bias` optional) both fuse an FFN's bias-add into its following activation
# the same way `BiasGelu`'s own name suggests -- confirmed against
# onnxruntime's own schema (`contrib_defs.cc`) and CPU kernel
# (`bias_gelu.cc`'s shared `BiasGelu<T, use_approximation>::AddBiasGelu`,
# which literally computes `value = input[i] + bias[i]` then erf- or
# tanh-based Gelu on `value`) and by direct execution. Neither fusion
# changes *which* Gelu variant is being computed -- FastGelu's tanh
# approximation is a different formula from plain `Gelu`'s erf-based one
# regardless of fusion, already true before this pass ever mattered to
# either -- so, exactly like a bias/scale `Add`/`Mul` hop's own constant,
# what matters here is only that each is a per-channel-independent,
# shape-preserving elementwise op with one extra constant operand to slice,
# not its particular activation math. `BiasGelu`'s own schema makes `B`
# (bias) a *required* second input; `FastGelu`'s marks its own `bias`
# *optional* -- both handled by :func:`_match_fused_bias_gelu` below. Like
# the `_BINARY_CHANNEL_OPS` bias/scale hop these sit alongside, this is a
# MatMul/Gemm-chain-only hop (:func:`_walk_to_consumer`/
# :func:`_walk_matmul_producer_backward`): Conv chains already decline any
# per-channel `Add`/`Mul` hop at all (a real Conv's bias already lives in
# its own third input, see this module's own docstring), and neither
# optimizer fusion targets Conv graphs in practice, so no Conv-side
# analogue is added.
_FUSED_BIAS_GELU_OPS: Dict[str, bool] = {
    "BiasGelu": True,  # bias (input 1) required by BiasGelu's own schema
    "FastGelu": False,  # bias (input 1) optional for FastGelu
}
_FUSED_BIAS_GELU_DOMAIN = "com.microsoft"

# LayerNormalization (plain ONNX, opset 17+)/RMSNormalization (plain ONNX,
# opset 23+)/SimplifiedLayerNormalization (onnxruntime's own RMSNorm
# equivalent -- confirmed, empirically, to run under the *default* (empty)
# ONNX domain despite living in onnxruntime's own contrib-op source tree, not
# under ``com.microsoft`` the way ``SkipLayerNormalization``/
# ``SkipSimplifiedLayerNormalization`` do -- via ``onnx.defs.get_schema`` for
# the first two and
# ``onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()``
# for the third), recognized as a mid-chain pass-through hop here -- distinct
# from `_ENTRY_LN_OPS`'s own, unrelated use of the very same three op types
# far below, purely as a transformer-block *entry* marker (never pruned
# themselves there): here a norm genuinely sits *between* a producer and a
# consumer, with its own affine `scale`/`bias` co-sliced right along with
# them, exactly like a per-channel `Add`/`Mul` bias/scale hop's own operand
# already is. Only ever a MatMul/Gemm-chain hop (:func:`_walk_to_consumer`),
# the same as the `_BINARY_CHANNEL_OPS`/`_FUSED_BIAS_GELU_OPS` hops it sits
# alongside -- no Conv-side analogue: Conv's own channel axis is axis 1
# (NCHW), never the *last* axis a `_NORM_PASS_THROUGH_OPS` node's default
# ``axis=-1`` actually normalizes, so the shape this hop targets never arises
# on a Conv chain in practice.
_NORM_PASS_THROUGH_OPS = (
    "LayerNormalization",
    "RMSNormalization",
    "SimplifiedLayerNormalization",
)


def _flat_channel_const(
    name: str, initializer_map: Dict[str, onnx.TensorProto]
) -> bool:
    """True if `name` names a constant float initializer shaped like a flat
    per-channel vector (``prod(dims) == dims[-1]``) -- the self-consistency
    bar every per-channel affine/bias/scale hop in this module checks before
    ever accepting a tensor as a slice target (the real ``dims[-1] ==
    n_channels`` check, once the chain's real channel count is known, is
    always left to the caller). Shared by :func:`_skip_layer_norm_const_names`
    (a ``SkipLayerNormalization``-family residual-merge node's own `gamma`/
    `beta`/`bias`) and :func:`_norm_pass_through_const_names` (a plain
    `_NORM_PASS_THROUGH_OPS` node's own `scale`/`bias`) -- same schema fact (a
    per-channel affine operand must broadcast against exactly the channel
    axis, nothing else), same check, regardless of which op reads it.
    """
    init = initializer_map.get(name)
    return (
        init is not None
        and _is_supported_float_dtype(init.data_type)
        and bool(list(init.dims))
        and int(np.prod(init.dims)) == init.dims[-1]
    )


def _norm_axis(node: onnx.NodeProto) -> int:
    for attr in node.attribute:
        if attr.name == "axis":
            return attr.i
    return -1  # schema default for every _NORM_PASS_THROUGH_OPS op


def _norm_axis_is_last(
    node: onnx.NodeProto,
    x_name: str,
    value_info_by_name: Optional[Dict[str, onnx.ValueInfoProto]],
) -> bool:
    """True if `node`'s own ``axis`` attribute is confirmed to normalize
    *only* the last axis of `x_name` -- ``axis == -1`` outright (every one of
    these ops' own schema already normalizes from `axis` through the last
    dimension, so ``-1`` unambiguously means exactly one axis, the last,
    regardless of rank), or a positive `axis` only when `x_name`'s own rank
    is known (:func:`_tensor_rank`) and equals ``axis == rank - 1`` -- the
    same reasoning, and the same "decline rather than guess" bar on an
    unknown rank, as :func:`_concat_axis_is_last`'s own positive-axis case
    (`value_info_by_name` left ``None`` simply narrows this to the `axis ==
    -1` case, never guesses -- every :func:`_walk_to_consumer` caller today
    threads its own graph's `value_info_by_name` through, so this only
    matters for a hypothetical future caller that doesn't). Any other axis
    (e.g. ``-2``, or a positive axis short of ``rank - 1``) spans more than
    this one trailing channel axis --
    multi-axis normalization this hop is deliberately out of scope for (see
    this section's own comment above) -- and is declined here, never guessed
    at.
    """
    axis = _norm_axis(node)
    if axis == -1:
        return True
    if axis < 0 or value_info_by_name is None:
        return False
    rank = _tensor_rank(x_name, value_info_by_name)
    return rank is not None and axis == rank - 1


def _norm_pass_through_const_names(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str]]]:
    """If every constant input a mid-chain `_NORM_PASS_THROUGH_OPS` `node`
    needs sliced -- `scale`/`Scale` (input 1, required by all three ops' own
    schema) and, for ``LayerNormalization`` only (``RMSNormalization``/
    ``SimplifiedLayerNormalization`` have no `bias`/`B` input at all --
    confirmed live via each op's own schema, see this section's own comment
    above), `B` (input 2, optional) -- is present exactly as the node's own
    input list says, and, whenever present, a constant float initializer
    shaped like a flat per-channel vector (:func:`_flat_channel_const`, the
    same bar :func:`_skip_layer_norm_const_names`'s own `gamma`/`beta`/`bias`
    check already uses), returns ``(scale_name, bias_name_or_None)``. The
    real ``dims[-1] == n_channels`` check, and confirming this node's own
    ``axis`` genuinely normalizes only the axis being pruned
    (:func:`_norm_axis_is_last`), are both left to the caller
    (:func:`_walk_to_consumer`) -- exactly like
    :func:`_skip_layer_norm_const_names`'s own deferred channel-count check.
    Declines (``None``) on a missing/non-constant `scale`, a *present* but
    non-constant `B`, or `scale`/`B` naming the same tensor (double-slicing
    it in :func:`_apply_chains`'s own per-hop loop would corrupt it) -- none
    of these is guessed at.
    """
    if node.op_type not in _NORM_PASS_THROUGH_OPS:
        return None
    if (
        len(node.input) < 2
        or not node.input[1]
        or not _flat_channel_const(node.input[1], initializer_map)
    ):
        return None
    scale_name = node.input[1]

    bias_name: Optional[str] = None
    if node.op_type == "LayerNormalization" and len(node.input) > 2 and node.input[2]:
        if not _flat_channel_const(node.input[2], initializer_map):
            return None
        bias_name = node.input[2]

    if scale_name == bias_name:
        return None  # tied scale/bias -- double-slicing would corrupt it

    return scale_name, bias_name


_ConsumerMatch = Tuple[onnx.NodeProto, str, bool]  # (node, weight, weight_transposed)


@dataclass(frozen=True)
class _Producer:
    node: onnx.NodeProto
    weight: str
    weight_transposed: bool
    bias: Optional[str]
    # Activation nodes between this producer's raw output and the point it
    # combines with another producer (a gated pair only -- see
    # :func:`_find_gated_chains`; empty for a plain single-producer chain).
    pre_ops: Tuple[onnx.NodeProto, ...] = ()
    # True for a Conv producer: `weight_transposed` is meaningless then
    # (Conv's ``[out_channels, in_channels, kH, kW]`` weight layout is
    # fixed), and output channels always live on axis 0.
    is_conv: bool = False
    # This Conv's own ``group`` attribute (always 1 for a MatMul/Gemm
    # producer, or an ordinary ``group=1`` Conv). > 1 for a general grouped
    # Conv producer (see :func:`_match_conv_producer`) -- output channel
    # axis 0 stays flat/global either way (grouping only splits the *input*
    # axis), so this only changes how :func:`_apply_chains` picks `keep`,
    # never how the producer's own weight is sliced.
    group: int = 1
    # True for a ``ConvTranspose`` producer (see
    # :func:`_match_conv_transpose_producer`) -- `is_conv` is *also* True
    # then (still Conv-style ellipsis slicing, not a MatMul/Gemm 2-D one),
    # this just further selects *which* axis is the output-channel one.
    # ``ConvTranspose``'s own weight layout is ``[in_channels, out_channels
    # /group, k1, ..., kn]`` -- confirmed live via
    # ``onnx.defs.get_schema("ConvTranspose")`` -- the reverse of ``Conv``'s
    # ``[out_channels, in_channels/group, ...]``: output channels live on
    # axis 1, not axis 0. Every caller that slices/reshapes this producer's
    # own weight by its output-channel axis (:func:`_slice_producer_weight`,
    # :func:`_producer_weight_nk`) consults this flag to use axis 1 instead
    # of axis 0. Only an ordinary (``group == 1``) ``ConvTranspose`` is ever
    # matched as a producer (see :func:`_match_conv_transpose_producer`'s
    # own docstring for why a grouped one is declined) -- `group` above is
    # therefore always 1 whenever this is True.
    is_conv_transpose: bool = False


@dataclass(frozen=True)
class _ConvPassThrough:
    """A depthwise Conv (``group == in_channels == out_channels``) the chain
    walk crossed transparently between a Conv chain's real producer and real
    consumer. A depthwise Conv mixes no channels at all -- output channel
    ``i`` depends only on input channel ``i`` -- so it needs no independent
    importance of its own the way a producer/consumer boundary does; it is
    carried on the matched :class:`_Chain` purely so :func:`_apply_chains`
    can slice its own ``[C, 1, kH, kW]`` weight (and bias, if present) by
    the *same* `keep` index set as the chain's real producer, and update its
    ``group`` attribute to the new channel count. See
    :func:`_walk_to_conv_consumer`.
    """

    node: onnx.NodeProto
    weight: str
    bias: Optional[str]


@dataclass(frozen=True)
class _Chain:
    # One producer for a plain chain; two for a gated (elementwise-product)
    # pair, where both branches must agree on which channels survive.
    producers: Tuple[_Producer, ...]
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    consumer_node: onnx.NodeProto
    consumer_weight: str
    consumer_weight_transposed: bool
    n_channels: int
    # True for a Conv consumer: input channels always live on axis 1 of its
    # ``[out_channels, in_channels, kH, kW]`` weight, regardless of
    # `consumer_weight_transposed` (unused then).
    consumer_is_conv: bool = False
    # Depthwise Conv hops the chain walk crossed transparently between the
    # real producer and the real consumer (Conv chains only -- see
    # :class:`_ConvPassThrough`; always empty for a MatMul/Gemm chain).
    conv_pass_through: Tuple[_ConvPassThrough, ...] = ()
    # The consumer's own ``group`` attribute (always 1 for a MatMul/Gemm
    # consumer, or an ordinary ``group=1`` Conv). > 1 for a general grouped
    # Conv consumer (see :func:`_match_conv_consumer`) -- unlike the
    # producer side, the consumer's input-channel axis *is*
    # per-group-relative, so this drives both `keep` selection (see
    # :func:`_chain_group`) and the dedicated slicing
    # :func:`_slice_grouped_consumer_conv_weight` performs.
    consumer_group: int = 1
    # True for a ``ConvTranspose`` consumer (see
    # :func:`_match_conv_transpose_consumer`, only ever set by
    # :func:`_find_conv_chains` -- every other Conv-chain finder in this
    # module (residual/merge groups, Concat-branch merges) still declines a
    # ``ConvTranspose`` consumer outright, see
    # :func:`_walk_to_conv_consumer`'s own `allow_conv_transpose_consumer`
    # parameter). `consumer_is_conv` is *also* True then; this just further
    # selects which axis is the input-channel one. Unlike ``Conv``'s
    # grouped consumer (axis 1, per-group-relative), a ``ConvTranspose``
    # consumer's own input-channel axis is axis 0, which spans the *full*
    # `in_channels` regardless of `consumer_group` (grouping only ever
    # splits axis 1 -- the *output* side -- for ``ConvTranspose``, the
    # mirror image of ``Conv``): see :func:`_match_conv_transpose_consumer`'s
    # own docstring for why this makes a plain, flat ``w[keep, ...]`` (not
    # :func:`_slice_grouped_consumer_conv_weight`) correct here even when
    # `consumer_group` is > 1.
    consumer_is_conv_transpose: bool = False
    # Extra, independent downstream consumer branches beyond the "primary"
    # one already carried on this chain's own singular `consumer_*` fields
    # above -- populated only by :func:`_find_conv_residual_chains`/
    # :func:`_find_matmul_residual_chains` for a residual/merge group whose
    # own shared spine tensor fans out to more than one safe, ordinary
    # consumer (see those functions' own "fan-out" section comment). Empty
    # for every other chain kind, and for a residual/merge group with no
    # such extra fan-out -- i.e. the exact shape every chain already had
    # before this field existed. Each entry always resolves to an ordinary
    # (`group == 1`) consumer, the same restriction the primary consumer
    # above is already held to for a residual/merge chain.
    extra_consumers: Tuple[_ConsumerBranch, ...] = ()


@dataclass(frozen=True)
class _ConsumerBranch:
    """One independent downstream path fed by an already-established
    residual/merge group's own shared channel-index set, beyond the
    group's primary consumer (see :class:`_Chain.extra_consumers`'s own
    comment and :func:`_find_conv_residual_chains`/
    :func:`_find_matmul_residual_chains`'s "fan-out" section comment).
    Mirrors exactly the subset of :class:`_Chain`'s own consumer-side
    fields a branch needs to be sliced by :func:`_apply_chains` -- its own
    trailing hop constants (`chain_ops`, e.g. an activation or bias/scale
    hop unique to *this* branch's own path to *its* consumer), its own
    real consumer, and, for a Conv branch, its own depthwise pass-through
    hops crossed on the way there. `consumer_group` mirrors
    :class:`_Chain`'s own field of the same name -- > 1 when this branch
    resolves to a general grouped Conv consumer (Conv residual/merge groups
    only; a MatMul/Gemm branch, from :func:`_resolve_matmul_fanout_branches`,
    leaves it at the default, there being no MatMul/Gemm-grouped-consumer
    concept at all). :func:`_find_conv_residual_chains` only ever hands
    back branches whose non-1 `consumer_group` values (if any), together
    with every producer's own `group` field, all agree -- see that
    function's own docstring -- so by the time :func:`_apply_chains` reads
    it here, it's already established as the one shared block boundary
    every producer and every branch alike must honor.
    """

    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    consumer_node: onnx.NodeProto
    consumer_weight: str
    consumer_weight_transposed: bool
    consumer_is_conv: bool = False
    conv_pass_through: Tuple[_ConvPassThrough, ...] = ()
    consumer_group: int = 1


@dataclass
class _TouchedState:
    """Cross-chain touched-role bookkeeping, shared (by reference) between
    :func:`_apply_chains` and :func:`_apply_concat_chains` so a weight one
    of them resizes can never also be resized a second, conflicting time by
    the other -- e.g. a Concat branch's own producer weight happening to
    also be, via a tied/shared initializer, some ordinary chain's producer
    elsewhere in the graph. See :func:`_apply_chains`'s own docstring for
    what each per-role set tracks and why roles are kept separate.
    """

    producer: Set[str] = field(default_factory=set)
    consumer: Set[str] = field(default_factory=set)
    const: Set[str] = field(default_factory=set)
    conv_hop: Set[str] = field(default_factory=set)
    stale_value_info: Set[str] = field(default_factory=set)


def _set_conv_group_attr(node: onnx.NodeProto, group: int) -> None:
    for attr in node.attribute:
        if attr.name == "group":
            attr.i = group
            return
    node.attribute.append(onnx.helper.make_attribute("group", group))


def _consumers_of(graph: onnx.GraphProto) -> Dict[str, List[onnx.NodeProto]]:
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def _match_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, bool, Optional[str], int]]:
    """If `node` is a MatMul/vanilla-Gemm with a constant 2-D float32
    weight (and, for Gemm, either no bias or a constant one), returns
    ``(weight_name, weight_transposed, bias_name_or_None, n_channels)``.
    """
    match = _match_matmul_like(node)
    if match is None:
        return None
    _, w_name, weight_transposed = match
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or not _is_supported_float_dtype(w_init.data_type)
        or len(w_init.dims) != 2
    ):
        return None
    bias_name = None
    if node.op_type == "Gemm" and len(node.input) == 3:
        bias_name = node.input[2]
        if bias_name not in initializer_map:
            return None  # non-constant bias -- can't safely prune it
    n_channels = w_init.dims[0] if weight_transposed else w_init.dims[1]
    return w_name, weight_transposed, bias_name, n_channels


def _match_fused_bias_gelu(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str]]]:
    """If `node` is a ``com.microsoft::BiasGelu``/``FastGelu`` node (see
    `_FUSED_BIAS_GELU_OPS`'s own comment above for the exact fused
    arithmetic and how it was confirmed), returns ``(data_name,
    bias_name_or_None)``: `data_name` is the node's own primary input (`A`/
    `X`, input 0), and `bias_name` is its per-channel bias operand (input 1)
    when present and a constant float initializer shaped like a flat
    per-last-axis vector -- the same self-consistency bar
    :func:`_walk_matmul_producer_backward`'s own `_BINARY_CHANNEL_OPS` hop
    check already uses (the real ``dims[-1] == n_channels`` check is
    deferred to the caller: :func:`_walk_to_consumer` already knows
    `n_channels` and checks immediately; :func:`_walk_matmul_producer_backward`
    doesn't yet, and defers to :func:`_find_matmul_residual_chains` once the
    group's real channel count is known, exactly like that hop's own and
    `_skip_layer_norm_const_names`'s own deferred check).

    Declines (``None``) when `node` isn't one of these ops/domain at all, or
    when its bias is required but missing (`BiasGelu`'s own schema makes its
    `B` input required, unlike `FastGelu`'s optional `bias`) or present but
    non-constant -- never guessed at, the same conservative bar a
    non-constant bias/scale on an ordinary `Add`/`Mul` hop already gets.
    `FastGelu` with its bias genuinely absent (omitted entirely, or present
    as an empty placeholder) returns ``(data_name, None)`` -- no term to
    slice, the same shape a `SkipLayerNormalization` node's own absent
    `beta`/`bias` already gets.
    """
    bias_required = _FUSED_BIAS_GELU_OPS.get(node.op_type)
    if (
        bias_required is None
        or node.domain != _FUSED_BIAS_GELU_DOMAIN
        or not node.input
        or not node.input[0]
        or len(node.output) != 1
    ):
        return None
    data_name = node.input[0]
    has_bias_input = len(node.input) > 1 and bool(node.input[1])
    if not has_bias_input:
        if bias_required:
            return None  # BiasGelu's own schema requires a bias operand
        return data_name, None  # FastGelu with no bias -- plain tanh-Gelu(x)
    bias_name = node.input[1]
    bias_init = initializer_map.get(bias_name)
    if (
        bias_init is None
        or not _is_supported_float_dtype(bias_init.data_type)
        or not list(bias_init.dims)
        or int(np.prod(bias_init.dims)) != bias_init.dims[-1]
    ):
        return None  # non-constant bias -- can't safely slice/prune it
    return data_name, bias_name


def _walk_to_consumer(
    start: str,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    n_channels: int,
    max_hops: int,
    forced_first_hop: Optional[onnx.NodeProto] = None,
    value_info_by_name: Optional[Dict[str, onnx.ValueInfoProto]] = None,
) -> Tuple[Optional[_ConsumerMatch], Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]]:
    """From tensor `start`, walks forward through shape-preserving
    elementwise ops (an activation, an Add/Mul against a constant
    per-channel bias/scale, a fused ``com.microsoft::BiasGelu``/
    ``FastGelu`` node -- see `_FUSED_BIAS_GELU_OPS`'s own comment and
    :func:`_match_fused_bias_gelu` -- or a plain `_NORM_PASS_THROUGH_OPS`
    node whose own ``axis`` normalizes exactly `cur`'s last axis -- see that
    constant's own comment, :func:`_norm_axis_is_last`, and
    :func:`_norm_pass_through_const_names`) with no other consumer anywhere
    along the way, until a MatMul/vanilla-Gemm consumer is found whose
    reduction dimension matches `n_channels`. Returns ``(None, ())`` if the
    walk runs out of hops, hits a branch, or never reaches such a consumer.

    `forced_first_hop`, when given, is used as the walk's very first hop
    instead of deriving it from `consumers_of[start]` -- see
    :func:`_walk_to_conv_consumer`'s own matching parameter for why (used
    only by :func:`_find_matmul_residual_chains`'s "fan-out" post-check);
    every ordinary caller leaves it ``None`` and gets identical behavior to
    before this parameter existed.

    `value_info_by_name`, when given, lets a positive-`axis`
    `_NORM_PASS_THROUGH_OPS` hop confirm that axis against the tensor's own
    known rank (:func:`_norm_axis_is_last`); every finder function that calls
    this (:func:`_find_chains`, :func:`_find_matmul_residual_chains` and
    :func:`_find_matmul_concat_chains` via :func:`_resolve_matmul_fanout_branches`,
    and :func:`_find_gated_chains`) builds and threads its own graph's copy
    through today. Left ``None``, only the unambiguous ``axis == -1`` case of
    that hop is still recognized -- never a correctness gap, just narrower
    coverage where rank isn't threaded through.
    """
    chain_ops: List[Tuple[onnx.NodeProto, Optional[str]]] = []
    consumer = None
    cur = start
    for _hop in range(max_hops):
        if _hop == 0 and forced_first_hop is not None:
            nxt = forced_first_hop
        else:
            candidates = consumers_of.get(cur, [])
            if len(candidates) != 1:
                break
            nxt = candidates[0]

        cm = _match_matmul_like(nxt)
        if cm is not None and cm[0] == cur:
            _, cw_name, c_weight_transposed = cm
            cw_init = initializer_map.get(cw_name)
            if (
                cw_init is not None
                and _is_supported_float_dtype(cw_init.data_type)
                and len(cw_init.dims) == 2
            ):
                k = cw_init.dims[1] if c_weight_transposed else cw_init.dims[0]
                if k == n_channels:
                    consumer = (nxt, cw_name, c_weight_transposed)
            break

        const_name: Optional[str] = None
        extra_const_name: Optional[str] = None
        if (
            nxt.op_type in _UNARY_PASS_THROUGH
            and list(nxt.input) == [cur]
            and len(nxt.output) == 1
        ):
            pass
        elif (
            nxt.op_type in _BINARY_CHANNEL_OPS
            and len(nxt.input) == 2
            and cur in nxt.input
            and len(nxt.output) == 1
        ):
            other = nxt.input[1] if nxt.input[0] == cur else nxt.input[0]
            const_init = initializer_map.get(other)
            if (
                const_init is not None
                and _is_supported_float_dtype(const_init.data_type)
                and list(const_init.dims)
                and const_init.dims[-1] == n_channels
                and int(np.prod(const_init.dims)) == n_channels
            ):
                const_name = other
            else:
                break
        elif nxt.op_type in _FUSED_BIAS_GELU_OPS:
            fused = _match_fused_bias_gelu(nxt, initializer_map)
            if fused is None or fused[0] != cur:
                break
            _, bias_name = fused
            if bias_name is not None and (
                initializer_map[bias_name].dims[-1] != n_channels
            ):
                break
            const_name = bias_name
        elif nxt.op_type in _NORM_PASS_THROUGH_OPS and nxt.domain == "":
            if not nxt.input or nxt.input[0] != cur:
                break
            if not _norm_axis_is_last(nxt, cur, value_info_by_name):
                break
            names = _norm_pass_through_const_names(nxt, initializer_map)
            if names is None:
                break
            scale_name, bias_name = names
            if initializer_map[scale_name].dims[-1] != n_channels:
                break
            if (
                bias_name is not None
                and initializer_map[bias_name].dims[-1] != n_channels
            ):
                break
            # Training-only secondary outputs (LayerNormalization's own
            # Mean/InvStdDev; SimplifiedLayerNormalization's own
            # inv_std_var; RMSNormalization has no secondary output at all)
            # would silently go stale (their *values*, not their shape,
            # depend on exactly which channels survive) if consumed by
            # anything -- declined here the same conservative way
            # :func:`_match_matmul_residual_merge` already declines a
            # consumed `mean`/`inv_std_var` on a `SkipLayerNormalization`
            # -family node.
            if any(
                nxt.output[i]
                and (consumers_of.get(nxt.output[i]) or nxt.output[i] in graph_outputs)
                for i in range(1, len(nxt.output))
            ):
                break
            const_name = scale_name
            extra_const_name = bias_name
        else:
            break

        out2 = nxt.output[0]
        if len(consumers_of.get(out2, [])) != 1 or out2 in graph_outputs:
            break
        chain_ops.append((nxt, const_name))
        if extra_const_name is not None:
            chain_ops.append((nxt, extra_const_name))
        cur = out2

    return consumer, tuple(chain_ops)


def _find_chains(graph: onnx.GraphProto) -> List[_Chain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}
    value_info_by_name = _value_info_by_name(graph)

    def _is_internal(name: str) -> bool:
        # Safe to reshape only if exactly one node reads it and it isn't
        # itself something the caller observes (a graph output).
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains = []
    for node in graph.node:
        info = _match_producer(node, initializer_map)
        if info is None:
            continue
        w_name, weight_transposed, bias_name, n_channels = info

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops = _walk_to_consumer(
            out_name,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_channels,
            _MAX_CHAIN_HOPS,
            value_info_by_name=value_info_by_name,
        )
        if consumer is None:
            continue

        chains.append(
            _Chain(
                producers=(_Producer(node, w_name, weight_transposed, bias_name),),
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
                n_channels=n_channels,
            )
        )
    return chains


def _conv_group(node: onnx.NodeProto) -> int:
    for attr in node.attribute:
        if attr.name == "group":
            return attr.i
    return 1  # ONNX default


def _match_conv_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str], int, int]]:
    """If `node` is an ordinary (``group=1``) *or* a general grouped
    (``1 < group < in_channels``, ``group != out_channels`` -- see this
    module's own docstring) ``Conv`` -- 1-D, 2-D, or 3-D alike, i.e. any
    spatial rank ``n >= 1`` -- with a constant ``[out_channels,
    in_channels/group, k1, ..., kn]`` float32/float16/bfloat16 weight (and,
    if present, a constant bias), returns
    ``(weight_name, bias_name_or_None, out_channels, group)``. Only the
    weight's own *rank* (``>= 3``: one output-channel axis, one
    input-channel axis, at least one spatial axis) is checked here --
    exactly the same schema-documented ``[M, C/group, k1, ..., kn]`` layout
    for any ``n`` (confirmed live via ``onnx.defs.get_schema("Conv")``), not
    a 2-D-specific one -- since every slicing/importance computation this
    producer role ever drives (:func:`_slice_producer_weight`,
    :func:`_producer_weight_nk`) already only ever touches axis 0 and
    leaves every remaining axis (spatial or not) alone via ``...``/``-1``,
    with no assumption about how many of them there are. A depthwise
    Conv (``group == in_channels == out_channels``) never matches: even
    though it *is* given a narrower exception elsewhere in this pass, as a
    transparent pass-through hop the chain walk may cross between two real
    producer/consumer boundaries (see
    :func:`_match_depthwise_conv_pass_through`,
    :func:`_walk_to_conv_consumer`), it is never itself matched as a
    producer -- only a *general* grouped Conv (this function's new case) is.
    A general grouped Conv's `group` output channels never need slicing
    themselves here: axis 0 (`out_channels`) is flat/global regardless of
    grouping (grouping only ever splits the *input* axis), so the caller's
    existing `keep`-index slicing of a producer's own weight/bias needs no
    special-casing for this -- only *which* `keep` indices get chosen (one
    independent top-k per group, see :func:`_apply_chains`) changes.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or not _is_supported_float_dtype(w_init.data_type)
        or len(w_init.dims) < 3
    ):
        return None
    group = _conv_group(node)
    if group < 1:
        return None
    out_channels = w_init.dims[0]
    in_channels = w_init.dims[1] * group
    if group > 1 and (
        group >= in_channels  # depthwise (a transparent hop, not a
        # producer) or an unsupported in-channels-per-group == 1 grouping
        or group == out_channels
        or out_channels % group != 0  # groups must stay equal-sized
    ):
        return None
    bias_name = None
    if len(node.input) == 3 and node.input[2]:
        bias_name = node.input[2]
        if bias_name not in initializer_map:
            return None  # non-constant bias -- can't safely prune it
    return w_name, bias_name, out_channels, group


def _match_conv_transpose_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str], int, int]]:
    """The ``ConvTranspose`` analogue of :func:`_match_conv_producer`. If
    `node` is an ordinary (``group == 1``) ``ConvTranspose`` with a constant
    ``[in_channels, out_channels/group, k1, ..., kn]`` (``n >= 1``)
    float32/float16/bfloat16 weight (and, if present, a constant bias),
    returns ``(weight_name, bias_name_or_None, out_channels, group=1)``.

    ``ConvTranspose``'s own weight layout is the schema-documented
    ``[C, M/group, k1, ..., kn]`` -- confirmed live via
    ``onnx.defs.get_schema("ConvTranspose")`` -- the *reverse* of ``Conv``'s
    ``[M, C/group, ...]``: input channels (``C``) come first, this node's
    own *output* channels (``M``, what "producer" means here -- the count
    downstream layers' input-channel dimension must match) live on axis 1,
    not axis 0. Every caller that slices/reshapes this producer's weight by
    its own output-channel axis (:func:`_slice_producer_weight`,
    :func:`_producer_weight_nk`) is told this via the returned `_Producer`'s
    own ``is_conv_transpose=True`` flag (set by :func:`_find_conv_chains`,
    the only caller of this function), and uses axis 1 instead of axis 0.

    Deliberately restricted to ``group == 1`` -- declined (``None``)
    otherwise -- unlike :func:`_match_conv_producer`'s support for a
    *general* grouped Conv: a grouped ``ConvTranspose`` producer's own
    output-channel axis (1) *is* per-group-relative (weight column ``j`` on
    input-row block ``g`` means global output channel ``g * (out_channels /
    group) + j``, mirroring ``Conv``'s own grouped *consumer* axis, see
    :func:`_match_conv_consumer`'s docstring) -- pruning it needs the
    equivalent of :func:`_slice_grouped_consumer_conv_weight` but for axis 1
    instead of axis 0, which this module does not implement. A scope
    decision, not an oversight: get ordinary ``ConvTranspose`` right first
    (see this module's own docstring for a Conv1d/Conv3d/ConvTranspose
    generalization's full scope), leaving grouped ``ConvTranspose`` as a
    clean, explicit decline for a future pass to pick up.
    """
    if node.op_type != "ConvTranspose" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or not _is_supported_float_dtype(w_init.data_type)
        or len(w_init.dims) < 3
    ):
        return None
    group = _conv_group(node)
    if group != 1:
        return None  # grouped ConvTranspose producer -- declined, see above
    out_channels = w_init.dims[1]
    bias_name = None
    if len(node.input) == 3 and node.input[2]:
        bias_name = node.input[2]
        if bias_name not in initializer_map:
            return None  # non-constant bias -- can't safely prune it
    return w_name, bias_name, out_channels, group


def _match_conv_consumer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, int, int]]:
    """If `node` is an ordinary (``group=1``) *or* a general grouped Conv
    (see :func:`_match_conv_producer`) -- any spatial rank ``n >= 1``, same
    as that function -- with a constant ``[out_channels, in_channels/group,
    k1, ..., kn]`` float32/float16/bfloat16 weight, returns
    ``(weight_name, in_channels, group)``. Like
    :func:`_match_conv_producer`, a depthwise Conv never matches here
    either -- it's only ever a transparent pass-through hop the chain walk
    crosses en route to a *real* consumer, never a consumer itself (see
    :func:`_match_depthwise_conv_pass_through`). Unlike the producer side, a
    grouped consumer's input-channel axis (axis 1 of its weight) *is*
    per-group-relative -- weight column ``j`` on an output filter belonging
    to group ``g`` means global input channel ``g * (in_channels / group) +
    j``, not global channel ``j`` -- so slicing it by the chain's `keep`
    indices needs the dedicated
    :func:`_slice_grouped_consumer_conv_weight`, not the flat
    ``w[:, keep, ...]`` an ordinary consumer's weight uses.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or not _is_supported_float_dtype(w_init.data_type)
        or len(w_init.dims) < 3
    ):
        return None
    group = _conv_group(node)
    if group < 1:
        return None
    out_channels = w_init.dims[0]
    in_channels = w_init.dims[1] * group
    if group > 1 and (
        group >= in_channels or group == out_channels or out_channels % group != 0
    ):
        return None
    return w_name, in_channels, group


def _match_conv_transpose_consumer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, int, int]]:
    """The ``ConvTranspose`` analogue of :func:`_match_conv_consumer`. If
    `node` is a ``ConvTranspose`` (any ``group >= 1`` -- see below) with a
    constant ``[in_channels, out_channels/group, k1, ..., kn]`` weight,
    returns ``(weight_name, in_channels, group)``.

    Unlike :func:`_match_conv_transpose_producer`'s ``group == 1``
    restriction, a grouped ``ConvTranspose`` *consumer* is matched for any
    `group`: this node's own input-channel axis is axis 0 of its
    ``[C, M/group, ...]`` weight, which -- unlike ``Conv``'s grouped
    consumer axis (1, per-group-relative) -- already spans the *full*,
    global ``C`` regardless of `group` (grouping only ever splits axis 1,
    the *output* side, for ``ConvTranspose`` -- the mirror image of
    ``Conv``, whose grouping only ever splits axis 1, the *input* side).
    Pruning it is therefore a plain, flat ``w[keep, ...]``
    (:func:`_slice_consumer_weight` with ``is_conv_transpose=True``) for any
    `group`, identical in form to an *ordinary* Conv producer's own axis-0
    slicing -- no per-group block-relative index translation needed the way
    :func:`_slice_grouped_consumer_conv_weight` provides for a grouped Conv
    consumer. `keep_count` still needs to land as a uniform count per
    `group`-sized block of axis 0 for the pruned node's own `group`
    attribute to stay well-formed (``new_in_channels % group == 0``,
    ``ConvTranspose``'s own well-formedness requirement) -- exactly what
    :func:`_apply_chains`'s existing `group > 1` branch (driven by
    :func:`_chain_group`, keyed on this `group`) already guarantees for any
    grouped consumer, Conv or ``ConvTranspose`` alike, with no
    `ConvTranspose`-specific change needed there.
    """
    if node.op_type != "ConvTranspose" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or not _is_supported_float_dtype(w_init.data_type)
        or len(w_init.dims) < 3
    ):
        return None
    group = _conv_group(node)
    if group < 1:
        return None
    in_channels = w_init.dims[0]
    if group > 1 and in_channels % group != 0:
        return None  # groups must stay equal-sized
    return w_name, in_channels, group


def _match_depthwise_conv_pass_through(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    n_channels: int,
) -> Optional[Tuple[str, Optional[str]]]:
    """If `node` is a depthwise ``Conv`` (``group == in_channels ==
    out_channels == n_channels``, any spatial rank ``n >= 1`` -- see
    :func:`_match_conv_producer`) with a constant ``[n_channels, 1, k1, ...,
    kn]`` float32/float16/bfloat16 weight (and, if present, a constant
    bias), returns ``(weight_name, bias_name_or_None)``. A depthwise Conv
    mixes no channels at all -- output channel ``i`` depends only on input
    channel ``i`` -- unlike a general grouped Conv (``group`` neither 1 nor
    `n_channels`), which is not matched here and stays out of scope for
    this pass entirely (see :func:`_match_conv_producer`/
    :func:`_match_conv_consumer`'s own docstrings): only in the depthwise
    case is every output channel tied 1:1 to the same-index input channel,
    which is what lets the chain walk (:func:`_walk_to_conv_consumer`) treat
    it as a transparent pass-through hop -- carrying whatever channel-index
    set survives upstream straight through, unchanged -- rather than a
    producer or consumer of its own.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or not _is_supported_float_dtype(w_init.data_type)
        or len(w_init.dims) < 3
        or w_init.dims[0] != n_channels
        or w_init.dims[1] != 1
        or _conv_group(node) != n_channels
    ):
        return None
    bias_name = None
    if len(node.input) == 3 and node.input[2]:
        bias_name = node.input[2]
        b_init = initializer_map.get(bias_name)
        if b_init is None or not _is_supported_float_dtype(b_init.data_type):
            return None  # non-constant bias -- can't safely prune it
    return w_name, bias_name


def _walk_to_conv_consumer(
    start: str,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    n_channels: int,
    max_hops: int,
    forced_first_hop: Optional[onnx.NodeProto] = None,
    allow_conv_transpose_consumer: bool = False,
) -> Tuple[
    Optional[Tuple[onnx.NodeProto, str, int, bool]],
    Tuple[Tuple[onnx.NodeProto, None], ...],
    Tuple[_ConvPassThrough, ...],
]:
    """The Conv analogue of :func:`_walk_to_consumer`: from tensor `start`,
    walks forward through unary shape-preserving activations (see
    `_UNARY_PASS_THROUGH`) and depthwise Conv hops (see
    :func:`_match_depthwise_conv_pass_through` -- transparent to the
    channel-index mapping, but each still needs its own weight/bias sliced
    and its ``group`` attribute updated, so they're returned separately as
    `conv_pass_through` rather than folded into `chain_ops`) with no other
    consumer anywhere along the way, until an ordinary (``group=1``) *or*
    general grouped Conv consumer is found whose input channel count
    matches `n_channels` (see :func:`_match_conv_consumer`), or --
    when `allow_conv_transpose_consumer` -- a ``ConvTranspose`` consumer
    (see :func:`_match_conv_transpose_consumer`) is found instead. A
    depthwise Conv is only ever a transparent hop, never a match for the
    consumer role itself -- one sitting last before a graph output or a
    branch simply ends the walk with no consumer found, same as any other
    unmatched topology. Unlike the MatMul/Gemm walk, no per-channel
    ``Add``/``Mul`` op is recognized -- see this module's own docstring for
    why that's out of scope for Conv chains.

    The returned consumer tuple's own 4th element is `is_conv_transpose`
    (see :class:`_Chain`'s own field of the same name) -- always ``False``
    for a Conv consumer, so every existing caller that only ever saw a
    3-tuple here before `is_conv_transpose` existed keeps identical
    behavior once it drops the extra element.

    `allow_conv_transpose_consumer` defaults ``False`` -- a ``ConvTranspose``
    node hit mid-walk then simply ends the walk with no consumer found
    (exactly like any other unmatched topology), *not* an error. Only
    :func:`_find_conv_chains` -- the plain single-producer/single-consumer
    chain finder this generalization was scoped to (see this module's own
    docstring) -- passes ``True``. Every other caller (the residual/merge
    "fan-out" branch resolution below, and the Concat-branch-merge consumer
    walk) leaves it at the default, deliberately excluding ``ConvTranspose``
    from those more elaborate topologies for now -- a scope decision, not
    an oversight (see this module's own docstring).

    `forced_first_hop`, when given, is used as the walk's very first hop
    instead of deriving it from `consumers_of[start]` -- every ordinary
    caller leaves it ``None`` and gets identical behavior to before this
    parameter existed (`start` must still have exactly one consumer, found
    the normal way). It exists only for
    :func:`_find_conv_residual_chains`'s own "fan-out" post-check: `start`
    having *more than one* consumer is expected there (it's an
    already-established residual/merge group's own shared spine tensor),
    and the caller has already picked one specific consumer node to resolve
    this one branch through -- every hop *after* the first still enforces
    the ordinary single-consumer bar unchanged, so a branch that itself
    forks further is still declined exactly as it always was.
    """
    chain_ops: List[Tuple[onnx.NodeProto, None]] = []
    conv_pass_through: List[_ConvPassThrough] = []
    consumer: Optional[Tuple[onnx.NodeProto, str, int, bool]] = None
    cur = start
    for _hop in range(max_hops):
        if _hop == 0 and forced_first_hop is not None:
            nxt = forced_first_hop
        else:
            candidates = consumers_of.get(cur, [])
            if len(candidates) != 1:
                break
            nxt = candidates[0]

        if nxt.op_type == "Conv" and nxt.input[0] == cur:
            depthwise = _match_depthwise_conv_pass_through(
                nxt, initializer_map, n_channels
            )
            if depthwise is not None:
                out2 = nxt.output[0]
                if len(consumers_of.get(out2, [])) != 1 or out2 in graph_outputs:
                    break
                dw_weight, dw_bias = depthwise
                conv_pass_through.append(_ConvPassThrough(nxt, dw_weight, dw_bias))
                cur = out2
                continue

            match = _match_conv_consumer(nxt, initializer_map)
            if match is not None and match[1] == n_channels:
                consumer = (nxt, match[0], match[2], False)
            break

        if (
            allow_conv_transpose_consumer
            and nxt.op_type == "ConvTranspose"
            and nxt.input[0] == cur
        ):
            ct_match = _match_conv_transpose_consumer(nxt, initializer_map)
            if ct_match is not None and ct_match[1] == n_channels:
                consumer = (nxt, ct_match[0], ct_match[2], True)
            break

        if not (
            nxt.op_type in _UNARY_PASS_THROUGH
            and list(nxt.input) == [cur]
            and len(nxt.output) == 1
        ):
            break

        out2 = nxt.output[0]
        if len(consumers_of.get(out2, [])) != 1 or out2 in graph_outputs:
            break
        chain_ops.append((nxt, None))
        cur = out2

    return consumer, tuple(chain_ops), tuple(conv_pass_through)


def _find_conv_chains(graph: onnx.GraphProto) -> List[_Chain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains = []
    for node in graph.node:
        # Try an ordinary/grouped Conv producer first, then (only if that
        # declines -- the two matchers' own op_type checks are mutually
        # exclusive, so this is never ambiguous) a ConvTranspose producer
        # (group == 1 only -- see _match_conv_transpose_producer's own
        # docstring for why grouped ConvTranspose stays out of scope).
        info = _match_conv_producer(node, initializer_map)
        is_producer_conv_transpose = False
        if info is None:
            info = _match_conv_transpose_producer(node, initializer_map)
            is_producer_conv_transpose = True
        if info is None:
            continue
        w_name, bias_name, n_channels, producer_group = info

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops, conv_pass_through = _walk_to_conv_consumer(
            out_name,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_channels,
            _MAX_CHAIN_HOPS,
            allow_conv_transpose_consumer=True,
        )
        if consumer is None:
            continue
        consumer_node, consumer_weight, consumer_group, consumer_is_conv_transpose = (
            consumer
        )

        if (
            producer_group > 1
            and consumer_group > 1
            and producer_group != consumer_group
        ):
            # Both sides grouped, but with a different group count: the two
            # sides' block boundaries wouldn't generally align (a channel
            # surviving as "the k-th of the producer's own group" has no
            # well-defined membership in any of the consumer's
            # differently-sized groups), so this composition needs real
            # cross-chain bookkeeping this pass doesn't attempt -- declined
            # outright, same as any other topology left unmatched rather
            # than guessed at. See this module's own docstring.
            continue

        chains.append(
            _Chain(
                producers=(
                    _Producer(
                        node,
                        w_name,
                        False,
                        bias_name,
                        is_conv=True,
                        group=producer_group,
                        is_conv_transpose=is_producer_conv_transpose,
                    ),
                ),
                chain_ops=chain_ops,
                consumer_node=consumer_node,
                consumer_weight=consumer_weight,
                consumer_weight_transposed=False,
                n_channels=n_channels,
                consumer_is_conv=True,
                conv_pass_through=conv_pass_through,
                consumer_group=consumer_group,
                consumer_is_conv_transpose=consumer_is_conv_transpose,
            )
        )
    return chains


# --- Conv residual (Add-merged) chains -------------------------------------
#
# A bounded slice of the general dependency-graph-grouping problem this
# module's own docstring otherwise disclaims (see its "residual/skip
# connection" paragraph): a channel-preserving `Add(a, b)` where *both*
# operands are non-constant tensors -- `y = Add(x, f(x))`, the shape every
# residual/skip connection takes -- forces whichever real Conv producer(s)
# feed `a` and `b` to be pruned to the exact same channel-index set, since
# they're about to be summed elementwise. `_walk_to_conv_consumer`'s own
# forward walk just breaks at such an `Add` (an ordinary Conv chain has no
# way to represent "two producers must agree"), so this is a *separate*
# finder -- `_find_conv_residual_chains` below -- built entirely on top of
# the existing `_Chain`/`_apply_chains` machinery rather than a change to
# `_walk_to_conv_consumer`/`_find_conv_chains` themselves: every `_Chain` it
# produces still has exactly one (real, `group=1`) consumer and some tuple
# of producers, precisely the shape `_apply_chains` (and both importance
# callbacks, `_plain_structured_importance`'s already-generic root-sum-
# square combination included) already knows how to ride a shared `keep`
# index set through -- only *finding* that tuple of producers needs new
# code.
#
# The union-find grouping :func:`_walk_conv_producer_backward` and
# :func:`_find_conv_residual_chains` build together covers not just a
# single `Add(x, f(x))` but a whole *chain* of such merges transitively
# sharing one spine channel count -- "many residual blocks share one
# spine" -- by walking backward from each `Add` operand and, on hitting
# *another* eligible `Add`'s raw output, unioning that `Add`'s own group in
# rather than stopping.
#
# A real multi-block ResNet stage's post-block tensor is read *twice*
# (once by the next block's own first Conv, once as-is by that block's own
# `Add`) -- exactly the "interior block" shape earlier versions of this
# section declined outright, since every hop here used to require *exactly
# one* consumer, the same bar every other hop in this module still holds
# every intermediate tensor to. That fan-out turns out to have its own
# provably-safe special case, bounded the same way the residual case itself
# is bounded relative to general dependency-graph pruning: once a group's
# shared channel-index set is established (by the *existing* backward
# union-find above -- this doesn't change how a group's own producers are
# found, or relax anything about *that*), it is a fixed, already-decided
# quantity everywhere within the group -- so *propagating* it forward to
# more than one independent downstream reader of the same in-group tensor
# is a different, narrower problem than *resolving* it from multiple
# upstream producers in the first place, and doesn't share that problem's
# ambiguity: there is no tie-break to invent, because every extra reader is
# either (a) an ordinary Conv consumer -- exactly the shape
# `_walk_to_conv_consumer` already knows how to slice, just entered at a
# specific node instead of derived from "the" sole consumer -- or (b)
# another eligible `Add`, which the *existing* union-find machinery above
# already absorbs into the very same group for free (it iterates every
# eligible `Add` in the graph unconditionally, not just ones reached from
# elsewhere), so two merges racing to claim the same spine either land in
# one group with one sink (fine) or in one group with *two* sinks -- caught
# by the pre-existing `len(sinks) != 1` check below, unchanged.
#
# `_resolve_conv_fanout_branches` is the actual new mechanism: once a
# group is otherwise fully resolved (agreeing leaf channel counts, exactly
# one sink, no degenerate producer), every tensor the group's own backward
# walk touched -- from a leaf producer's own output through every
# pass-through/unary hop and every interior `Add`'s own output, to the
# sink's own output -- is checked for *extra* consumers beyond the ones the
# group's own union-find already accounts for, and each extra consumer is
# resolved independently via `_walk_to_conv_consumer` (seeded at that one
# specific node -- see its own `forced_first_hop` parameter). Any extra
# consumer that doesn't resolve this way -- forks further itself, reaches a
# graph output, resolves to a general grouped Conv, or duplicates a weight
# another branch already claims -- declines the *entire* group, never a
# partial cut; what survives is one or more independent forward branches
# (:class:`_Chain.extra_consumers`), every one sliced by the exact same
# shared `keep` array, so there is no "different derivation" for any branch
# to disagree with another one about.
#
# What this still does **not** reach: two chains (a residual group and
# anything else, or two different residual groups) that would each prune
# the *same* weight to a *different* keep set. That can't happen on a
# shared *activation* tensor at all -- ONNX gives every tensor exactly one
# producer, so a tensor can only ever belong to the one group whose own
# backward walk (or extra-branch forward walk) reaches it -- but it can
# still happen on a shared *weight* two otherwise-independent chains both
# want to touch (a tied/reused initializer); `_apply_chains`'s own
# touched-role tracking (`producer_touched`/`consumer_touched`/
# `const_touched`/`conv_hop_touched`) already declines that case for a
# single-consumer chain, and is extended here (see its own comment) to
# check every branch's own consumer weight, not just the primary one, so
# it keeps catching it for a multi-branch chain too. A lone residual
# connection whose branches don't fan out elsewhere (e.g. a
# projection-shortcut block), a genuinely linear stack of `Add`-only
# combinations, and now a real *interior* multi-block stage, are all
# reached with the same oracle-verified numeric guarantee as every other
# chain kind here. See this module's own docstring for the exact boundary
# of what this still declines.


def _is_eligible_add_merge(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> bool:
    """True for an ``Add`` node :func:`_find_conv_residual_chains`/
    :func:`_find_matmul_residual_chains` may treat as a residual merge
    point: exactly two distinct, non-constant operands.
    A per-channel bias/scale ``Add`` (one operand a constant initializer --
    already out of scope for Conv chains generally, see this module's own
    docstring) or a degenerate ``Add(x, x)`` never qualifies -- neither is a
    "two independent producers must agree" merge point at all.
    """
    return (
        node.op_type == "Add"
        and len(node.input) == 2
        and len(node.output) == 1
        and node.input[0] != node.input[1]
        and node.input[0] not in initializer_map
        and node.input[1] not in initializer_map
    )


def _match_conv_pass_through_self(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str]]]:
    """The depthwise-Conv pass-through check :func:`_walk_conv_producer_backward`
    uses: unlike :func:`_match_depthwise_conv_pass_through`, which validates
    a hop against an externally supplied `n_channels`, the backward residual
    walk doesn't know its group's shared channel count yet at the point it
    first crosses a hop (it's still walking toward whichever real producer
    -- or other ``Add`` -- eventually establishes it), so this checks the
    node's own weight is self-consistently depthwise-shaped (``dims[0] ==
    group``, ``dims[1] == 1``) by calling that same matcher with the node's
    own ``dims[0]`` as the "expected" count -- trivially satisfying that one
    check and leaving every other one intact. :func:`_find_conv_residual_chains`
    re-validates every such hop against the group's real, established
    channel count once the whole group is resolved.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    w_init = initializer_map.get(node.input[1])
    if (
        w_init is None
        or not _is_supported_float_dtype(w_init.data_type)
        or len(w_init.dims) < 3
    ):
        return None
    return _match_depthwise_conv_pass_through(node, initializer_map, w_init.dims[0])


def _walk_conv_producer_backward(
    start: str,
    node_by_output: Dict[str, onnx.NodeProto],
    initializer_map: Dict[str, onnx.TensorProto],
    graph_outputs: Set[str],
    max_hops: int,
) -> Tuple[
    str,
    Optional[Union[Tuple[_Producer, int], onnx.NodeProto]],
    Tuple[_ConvPassThrough, ...],
    Tuple[onnx.NodeProto, ...],
    Tuple[Tuple[str, onnx.NodeProto], ...],
]:
    """The backward counterpart of :func:`_walk_to_conv_consumer`, used only
    by :func:`_find_conv_residual_chains` to resolve one operand of an
    ``Add`` merge point back to whatever produces it. Walks backward from
    tensor `start` through unary pass-through activations and
    self-consistently-depthwise Conv hops (see
    :func:`_match_conv_pass_through_self`), declining (only) whenever a
    tensor crossed -- `start` itself included -- is a graph output (a
    caller-observed shape this pass never resizes); *how many* other things
    also read that same tensor is deliberately **not** checked here -- see
    :func:`_find_conv_residual_chains`'s own "fan-out" section comment for
    why, and how every such extra reader still gets its own safety check,
    just later, once the group's real channel count is known.

    Returns one of:

    - ``("producer", (producer, n_channels), pass_through, unary_ops,
      edges)`` -- resolved all the way back to a real Conv producer
      (``group == 1`` or a general grouped Conv -- see
      :func:`_match_conv_producer`; the caller, not this function, checks
      every producer/consumer the group eventually collects agrees on one
      shared `group` count);
    - ``("add", add_node, pass_through, unary_ops, edges)`` -- resolved to
      another eligible ``Add`` merge node's raw output instead (the "many
      residual blocks share one spine" case: the caller unions this group
      with that ``Add``'s own rather than treating it as a separate
      producer);
    - ``("fail", None, (), (), ())`` -- a graph input, a non-Conv/non-``Add``
      producer, a graph output crossed mid-walk, or the hop limit -- the
      caller declines the whole group this operand belongs to, rather than
      guessing.

    `edges` is, for every hop that actually advanced `cur`, the pair
    ``(new_cur, node)`` recording that `new_cur`'s own *in-group* forward
    consumer is `node` -- i.e. the one reader of `new_cur` that this walk
    itself already accounts for, so :func:`_find_conv_residual_chains`
    doesn't re-flag it as a stray extra consumer needing its own separate
    resolution once fan-out is no longer rejected here (`start` itself,
    plus every tensor named as some `edges` entry's own `new_cur`, is
    exactly the full set of tensors this walk checked -- nothing else needs
    tracking separately). `start`'s own in-group forward consumer -- the
    ``Add`` this walk was launched *from* -- isn't a `node_by_output` hop at
    all, so the caller records that one edge itself.
    """
    pass_through: List[_ConvPassThrough] = []
    unary_ops: List[onnx.NodeProto] = []
    edges: List[Tuple[str, onnx.NodeProto]] = []
    cur = start
    for _hop in range(max_hops):
        if cur in graph_outputs:
            return "fail", None, (), (), ()
        node = node_by_output.get(cur)
        if node is None or len(node.output) != 1 or node.output[0] != cur:
            return "fail", None, (), (), ()

        prod_info = _match_conv_producer(node, initializer_map)
        if prod_info is not None:
            w_name, bias_name, n_channels, producer_group = prod_info
            # A general grouped Conv producer is allowed through here
            # unconditionally -- `producer_group` is simply carried on the
            # returned `_Producer` (its output-channel axis 0 stays
            # flat/global regardless of grouping, same as the ordinary
            # `_find_conv_chains` case, see `_Producer.group`'s own
            # docstring). Whether every producer/consumer this group
            # eventually collects actually *agrees* on one shared group
            # count is not decidable per-operand here -- it's a whole-group
            # property -- so the check is deferred to
            # :func:`_find_conv_residual_chains`, which declines the entire
            # group (not just this operand) on a mismatch, mirroring
            # :func:`_find_conv_chains`'s own "both sides grouped with a
            # different group count" decline.
            producer = _Producer(
                node, w_name, False, bias_name, is_conv=True, group=producer_group
            )
            return (
                "producer",
                (producer, n_channels),
                tuple(reversed(pass_through)),
                tuple(reversed(unary_ops)),
                tuple(edges),
            )

        dw = _match_conv_pass_through_self(node, initializer_map)
        if dw is not None:
            dw_weight, dw_bias = dw
            pass_through.append(_ConvPassThrough(node, dw_weight, dw_bias))
            edges.append((node.input[0], node))
            cur = node.input[0]
            continue

        if node.op_type in _UNARY_PASS_THROUGH and len(node.input) == 1:
            unary_ops.append(node)
            edges.append((node.input[0], node))
            cur = node.input[0]
            continue

        if _is_eligible_add_merge(node, initializer_map):
            return (
                "add",
                node,
                tuple(reversed(pass_through)),
                tuple(reversed(unary_ops)),
                tuple(edges),
            )

        return "fail", None, (), (), ()

    return "fail", None, (), (), ()


def _resolve_conv_fanout_branches(
    backbone_tensors: List[str],
    accounted: Dict[str, Set[int]],
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    n_channels: int,
) -> Optional[List[_ConsumerBranch]]:
    """For an already-established Conv residual/merge group -- every tensor
    in `backbone_tensors` is one :func:`_walk_conv_producer_backward`'s own
    backward walk already proved carries that group's shared channel-index
    set, `accounted` marks, per tensor, which specific consumer node(s) are
    already part of the group's own internal wiring (see that function's
    own docstring) -- finds every *extra* consumer (one not in `accounted`)
    of every backbone tensor and resolves each independently via
    :func:`_walk_to_conv_consumer`, seeded at that one specific node (see
    its own `forced_first_hop` parameter). This is the actual "fan-out"
    mechanism: :func:`_find_conv_residual_chains`'s own section comment
    above explains why propagating one already-established `keep` set
    forward to several independent, individually-ordinary consumer
    branches is safe in a way general dependency-graph *merging* isn't.

    Returns ``None`` -- decline the *whole* group, never partially -- if
    any backbone tensor is itself a graph output (this pass never resizes
    a directly-observed shape), any extra consumer fails to resolve to a
    real (ordinary *or* general grouped) Conv consumer within the usual hop
    limit, or two different branches would end up naming the same consumer
    weight (double-slicing it would corrupt it -- the same degenerate case
    :func:`_apply_chains` already guards a single chain's own producers
    against). A resolved branch's own `group` (see :func:`_match_conv_consumer`)
    is carried on its `_ConsumerBranch.consumer_group` unconditionally --
    this function does *not* itself check it agrees with anything else in
    the group (it has no view of the group's other producers/branches);
    :func:`_find_conv_residual_chains`, which does, declines the whole
    group if any two non-1 `group` values collected from every producer and
    every branch alike disagree. Returns an empty list if the group has no
    extra fan-out *and* no branch at all (every backbone tensor's
    consumers, if any, are already accounted for) -- the caller treats that
    exactly like "no consumer found" and declines, same as before this
    function existed. Otherwise returns every resolved branch; the caller
    picks one as this chain's own "primary" consumer (the shape every other
    chain already has) and carries the rest as `_Chain.extra_consumers` --
    an arbitrary choice with no bearing on correctness, since every branch
    is sliced by the exact same shared `keep` array (and, once every
    `group` is confirmed to agree, the exact same block boundaries within
    it).
    """
    branches: List[_ConsumerBranch] = []
    seen_weights: Set[str] = set()
    for tensor in backbone_tensors:
        if tensor in graph_outputs:
            return None
        seen_nodes: Set[int] = set()
        for consumer_node in consumers_of.get(tensor, []):
            if id(consumer_node) in seen_nodes:
                continue
            seen_nodes.add(id(consumer_node))
            if id(consumer_node) in accounted.get(tensor, ()):
                continue  # already part of the group's own established wiring
            resolved, br_chain_ops, br_pass_through = _walk_to_conv_consumer(
                tensor,
                initializer_map,
                consumers_of,
                graph_outputs,
                n_channels,
                _MAX_CHAIN_HOPS,
                forced_first_hop=consumer_node,
            )
            if resolved is None:
                return None
            # `allow_conv_transpose_consumer` was left at its default
            # (False) above, so `_` here is always False -- ConvTranspose
            # is deliberately out of scope for a residual/merge group's own
            # fan-out branches (see _walk_to_conv_consumer's own docstring).
            branch_node, branch_weight, branch_group, _ = resolved
            # A general grouped Conv consumer is allowed through here
            # unconditionally, same as the primary consumer -- its own
            # `group` is simply carried on the returned `_ConsumerBranch`
            # (see its own docstring) and cross-checked against every other
            # producer/branch by the caller (:func:`_find_conv_residual_chains`),
            # which declines the whole group on a mismatch rather than any
            # one branch guessing.
            if branch_weight in seen_weights:
                return None  # two branches naming the same consumer weight
            seen_weights.add(branch_weight)
            branches.append(
                _ConsumerBranch(
                    chain_ops=br_chain_ops,
                    consumer_node=branch_node,
                    consumer_weight=branch_weight,
                    consumer_weight_transposed=False,
                    consumer_is_conv=True,
                    conv_pass_through=br_pass_through,
                    consumer_group=branch_group,
                )
            )
    return branches


def _find_conv_residual_chains(graph: onnx.GraphProto) -> List[_Chain]:
    """Finds Conv residual/skip-connection groups -- see the section comment
    above. For every maximal union-find group of transitively-connected
    eligible ``Add`` merge points (:func:`_is_eligible_add_merge`), resolves
    every member's two operands via :func:`_walk_conv_producer_backward`:
    each must reach either a real Conv producer (``group == 1`` or a
    general grouped Conv -- a "leaf" of the group) or another `Add` already
    in the same group. If *any* operand, anywhere in the group, fails to
    resolve that way, or the leaf producers' channel counts don't all
    agree, the *entire* group is declined -- never partially pruned. Every
    tensor visited along the way (see :func:`_walk_conv_producer_backward`'s
    own `edges`) plus the group's own "sink" (the one member whose own
    output isn't itself consumed by another member) is then handed to
    :func:`_resolve_conv_fanout_branches`, which finds and resolves every
    extra (non-backbone) consumer fan-out reaches, in exactly the bounded
    way this section's own comment above describes -- declining the whole
    group if any such branch can't be resolved. Once every leaf producer
    and every resolved branch (primary and extra alike) is known, their
    `group` values are cross-checked: any two *different* non-1 values
    anywhere in the group decline it entirely (see this module's own
    docstring for why only "everyone agrees on one shared `group` count"
    is a provably-safe slice of the general-grouped-Conv case). What
    survives is one or more independent forward branches, all fed by the
    exact same shared `keep` set (itself computed per-`group`-block when
    that shared count is > 1) once :func:`_apply_chains` computes it.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    node_by_output = {out: node for node in graph.node for out in node.output}
    graph_outputs = {o.name for o in graph.output}

    eligible_adds = [
        node for node in graph.node if _is_eligible_add_merge(node, initializer_map)
    ]
    if not eligible_adds:
        return []
    add_index = {id(node): i for i, node in enumerate(eligible_adds)}

    parent = list(range(len(eligible_adds)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    Edge = Tuple[
        str,
        Optional[Union[Tuple[_Producer, int], onnx.NodeProto]],
        Tuple[_ConvPassThrough, ...],
        Tuple[onnx.NodeProto, ...],
        Tuple[Tuple[str, onnx.NodeProto], ...],
    ]
    edge_results: Dict[int, List[Edge]] = {}
    poisoned: Set[int] = set()
    for idx, add_node in enumerate(eligible_adds):
        results: List[Edge] = []
        for operand in add_node.input:
            edge = _walk_conv_producer_backward(
                operand,
                node_by_output,
                initializer_map,
                graph_outputs,
                _MAX_CHAIN_HOPS,
            )
            results.append(edge)
            kind, payload = edge[0], edge[1]
            if kind == "fail":
                poisoned.add(idx)
            elif kind == "add":
                assert isinstance(payload, onnx.NodeProto)
                j = add_index.get(id(payload))
                if j is None:
                    poisoned.add(idx)  # defensive -- shouldn't happen
                else:
                    union(idx, j)
        edge_results[idx] = results

    groups: Dict[int, List[int]] = {}
    for idx in range(len(eligible_adds)):
        groups.setdefault(find(idx), []).append(idx)

    chains: List[_Chain] = []
    for members in groups.values():
        if any(i in poisoned for i in members):
            continue

        leaf_producers: List[_Producer] = []
        n_channels_set: Set[int] = set()
        pass_through: List[_ConvPassThrough] = []
        unary_ops: List[onnx.NodeProto] = []
        referenced: Set[int] = set()
        # Every tensor either walk of every member proved carries this
        # group's own shared channel-index set (see
        # _walk_conv_producer_backward's own `edges`), and, for
        # each, which specific consumer node is already part of the
        # group's own internal wiring -- fed to _resolve_conv_fanout_branches
        # below so only genuinely *extra* consumers need their own separate
        # resolution. A plain list (not a set) preserves first-seen order,
        # so which resolved branch ends up "primary" is deterministic.
        backbone_tensors: List[str] = []
        accounted: Dict[str, Set[int]] = {}

        def _mark_backbone(tensor: str, node: onnx.NodeProto) -> None:
            if tensor not in accounted:
                backbone_tensors.append(tensor)
            accounted.setdefault(tensor, set()).add(id(node))

        for idx in members:
            add_node = eligible_adds[idx]
            for operand, (kind, payload, pt, uops, edges) in zip(
                add_node.input, edge_results[idx]
            ):
                _mark_backbone(operand, add_node)
                for tensor, node in edges:
                    _mark_backbone(tensor, node)
                pass_through.extend(pt)
                unary_ops.extend(uops)
                if kind == "producer":
                    assert payload is not None and not isinstance(
                        payload, onnx.NodeProto
                    )
                    producer, n_channels = payload
                    leaf_producers.append(producer)
                    n_channels_set.add(n_channels)
                elif kind == "add":
                    assert isinstance(payload, onnx.NodeProto)
                    referenced.add(add_index[id(payload)])

        if len(n_channels_set) != 1:
            continue  # branches disagree on channel count -- decline
        n_channels = next(iter(n_channels_set))

        # Every leaf producer's own `group` (1 for an ordinary Conv, > 1
        # for a general grouped one -- see _match_conv_producer) must agree
        # with every other non-1 value in the group, mirroring
        # _find_conv_chains's own "both sides grouped with a different
        # group count" decline: a group's shared `keep` set can only
        # respect one block partition of `n_channels`, and different
        # `group` counts imply different block boundaries (see
        # _chain_group's own docstring for the single-producer case this
        # generalizes). Checked here, before spending work on fan-out
        # resolution below, since it only depends on already-known
        # producer info; the *consumer* side of this same check (primary
        # and extra branches) can only happen once fan-out is resolved, see
        # below.
        producer_groups = {p.group for p in leaf_producers if p.group > 1}
        if len(producer_groups) > 1:
            continue  # producers disagree on group count -- decline

        # Every depthwise pass-through hop was only self-consistently
        # checked when first crossed (see _match_conv_pass_through_self);
        # now that the group's real channel count is known, re-validate.
        if any(
            initializer_map[hop.weight].dims[0] != n_channels for hop in pass_through
        ):
            continue

        sinks = [idx for idx in members if idx not in referenced]
        if len(sinks) != 1:
            continue  # not a single linear chain of merges -- decline
        sink_add = eligible_adds[sinks[0]]

        if len({p.weight for p in leaf_producers}) != len(leaf_producers):
            continue  # degenerate -- the same producer named twice

        # The sink's own output is never `visited` by any member's own
        # backward walk (nothing in the group walks *through* it -- that's
        # what makes it the sink), so it needs adding explicitly; it starts
        # with no accounted-for consumer of its own at all.
        sink_out = sink_add.output[0]
        if sink_out not in accounted:
            backbone_tensors.append(sink_out)
            accounted[sink_out] = set()

        branches = _resolve_conv_fanout_branches(
            backbone_tensors,
            accounted,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_channels,
        )
        if not branches:
            continue

        # Completes the group-count agreement check started above: every
        # branch's own `consumer_group` (primary and extra alike) must also
        # agree with `producer_groups` -- the *consumer*-side half of
        # `_find_conv_chains`'s own "both sides grouped with a different
        # group count" decline, generalized from one consumer to however
        # many branches this group's fan-out resolved. `producer_groups`
        # was already checked internally consistent above, so folding in
        # every branch's own value here and re-checking once more catches
        # any producer/branch mismatch, in either direction.
        all_groups = producer_groups | {
            b.consumer_group for b in branches if b.consumer_group > 1
        }
        if len(all_groups) > 1:
            continue  # producer(s) and/or branch(es) disagree on group count

        primary, extra_branches = branches[0], tuple(branches[1:])
        chain_ops = (
            tuple((op, None) for op in unary_ops)
            + tuple((eligible_adds[i], None) for i in members)
            + primary.chain_ops
        )

        chains.append(
            _Chain(
                producers=tuple(leaf_producers),
                chain_ops=chain_ops,
                consumer_node=primary.consumer_node,
                consumer_weight=primary.consumer_weight,
                consumer_weight_transposed=False,
                n_channels=n_channels,
                consumer_is_conv=True,
                extra_consumers=extra_branches,
                conv_pass_through=tuple(pass_through) + primary.conv_pass_through,
                consumer_group=primary.consumer_group,
            )
        )
    return chains


# --- MatMul/Gemm residual (Add-merged) chains -------------------------------
#
# The MatMul/Gemm analogue of the Conv residual/Add-merge grouping above --
# same union-find-over-eligible-`Add`-merge-points construction, same
# provably-safe special case (`y = Add(a, b)`, two non-constant operands,
# forces whichever real producer(s) feed `a`/`b` to agree on one shared
# channel-index set), and the same bounded fan-out mechanism
# (`_resolve_matmul_fanout_branches`, the direct analogue of
# `_resolve_conv_fanout_branches`): an *interior* block of a deep residual
# stack -- its own "post-block" tensor read both by the next block and
# directly by that block's own `Add`/`SkipLayerNormalization` -- is reached
# by propagating the group's own already-established `keep` set forward to
# every extra ordinary consumer such a tensor has, exactly as for Conv; see
# `_find_conv_residual_chains`'s own section comment above for the full
# reasoning (why propagation, unlike backward resolution, has no ambiguity
# to guess at, and precisely what still isn't reached: two chains wanting
# different keep sets on the same shared *weight* -- never possible on a
# shared *activation*, since ONNX gives every tensor exactly one producer).
# Only what's different for MatMul/Gemm from the Conv case is covered here.
# `_is_eligible_add_merge` above is reused unchanged -- it was never
# Conv-specific to begin with (it only inspects the `Add` node's own
# operands against `initializer_map`), so no MatMul-specific variant is
# needed.
#
# This is exactly the shape every current transformer block's residual
# stream takes -- `x = x + SelfAttn(LN(x))`, `x = x + MLP(LN(x))` -- the
# single most valuable gap this closes versus the Conv-only residual
# support above, since a MatMul/Gemm residual chain was previously declined
# outright (see this module's own docstring's prior "MatMul/Gemm
# residuals ... remains out of scope" sentence, now narrowed).
#
# One real structural difference from the Conv version, not just a
# find-and-replace of "Conv" with "MatMul/Gemm": `_walk_to_consumer` (the
# MatMul/Gemm forward walk) allows a *wider* hop set than
# `_walk_to_conv_consumer` does -- not just unary activations, but also a
# per-channel bias/scale `Add`/`Mul` against a *constant* initializer (see
# this module's own docstring's "shape-preserving elementwise ops ...
# and for MatMul/Gemm also a bias/scale add/mul" phrase, and
# `_walk_to_consumer`'s own `_BINARY_CHANNEL_OPS` branch). There is no
# MatMul/Gemm analogue of a depthwise-Conv pass-through hop at all -- nothing
# in the MatMul/Gemm producer/consumer vocabulary mixes channels
# transparently the way a depthwise Conv does -- so `_walk_matmul_producer_backward`
# below mirrors *that* wider hop set symmetrically instead: unary
# pass-through activations (as before) plus a per-channel `Add`/`Mul` against
# a constant, walked backward through to whichever tensor it combined with.
# Distinguishing that per-channel bias/scale hop from an eligible residual
# merge is the crux of the whole backward walk, and it falls out for free
# from `_is_eligible_add_merge`'s own definition: an `Add` with exactly one
# constant operand can never be an eligible merge (it requires *both*
# operands non-constant), so it is unambiguously a bias hop instead, and a
# `Mul` is never a merge candidate at all (only `Add` is, per
# `_is_eligible_add_merge`'s own op-type check) -- a per-channel `Mul` hop and
# a residual `Add` merge are never the same node under any input shape.
#
# Because the backward walk doesn't yet know the group's real, shared
# channel count at the point it first crosses a bias/scale hop (the same
# situation `_match_conv_pass_through_self` documents for a depthwise Conv
# hop), it only self-consistently checks the constant is float and
# effectively a flat per-last-axis vector (`prod(dims) == dims[-1]`) when
# first crossed, deferring the real `dims[-1] == n_channels` check to
# `_find_matmul_residual_chains` once the group's producers establish it --
# exactly the same defer-then-revalidate split the depthwise Conv hop above
# uses for its own group-count check.
#
# Two compositions this was checked against. One turns out to have a
# provably-safe special case, the same way a gated pair's own two producers
# already get resolved together for the non-residual case; the other is
# still not safe to handle silently and is declined outright (the group is
# poisoned, left untouched) exactly like everywhere else in this pass a
# composition can't be proven safe:
#
# - **A gated (SwiGLU/GeGLU-style) `Mul` pair feeding directly into a
#   residual branch with no downstream projection in between** -- `Add(x,
#   Mul(gate, up))` rather than the usual `Add(x, MatMul(Mul(gate, up),
#   Wd))`. A `Mul` node reached while walking backward always has *two*
#   non-constant operands (unlike a bias/scale `Mul`, which has exactly one
#   constant operand and is walked through as an ordinary hop above) -- so
#   rather than guessing which one is "the" branch and dropping the other's
#   contribution, both are resolved: `_find_gated_chains`'s own
#   `_trace_gate_producer_backward` walks each operand back to its own real
#   MatMul/Gemm producer (through the same unary-activation pre-ops a gated
#   pair outside a residual chain already tolerates), and *both* resulting
#   producers -- not just one -- are folded into this group's own shared
#   leaf-producer set, ranked by the same combined (root-sum-square)
#   importance a gated pair already uses and pruned to the one channel-index
#   set the whole group shares. Nothing new is dropped or guessed at: every
#   real producer that must agree on the group's `keep` set still gets a say
#   in ranking it.
#
#   Composing this with the rest of the residual machinery -- fan-out,
#   transitive multi-block chains, `SkipLayerNormalization`'s own const-hop
#   bookkeeping -- was checked explicitly and needs no new machinery of its
#   own, because the two mechanisms don't actually overlap in what they each
#   guard:
#
#   - `_trace_gate_producer_backward` already holds *every* tensor it
#     crosses -- the gate/up operand itself, and every pre-op activation
#     output on the way back to the real producer -- to an exact
#     single-consumer bar (see its own docstring), stricter than this walk's
#     own deferred bias/scale-hop tensors (which *are* allowed extra
#     consumers, resolved later via fan-out). So a gate or up branch that
#     fans out anywhere along its own path is never silently resolved -- it
#     fails the trace outright, the same as it would for an ordinary,
#     non-residual gated pair (see
#     `test_structured_pruning_matmul_residual_add_declines_on_gated_branch_with_extra_fanout`).
#     Nothing about being embedded in a residual walk relaxes that bar.
#   - The `Mul` node's own *output* -- the tensor actually read by the `Add`
#     -- is not treated specially: it becomes this operand's own backbone
#     tensor exactly like an ordinary producer's raw output already is (see
#     `_mark_backbone` in `_find_matmul_residual_chains` below), so an extra
#     reader of it (fanning out to, say, a second, unrelated eligible merge)
#     goes through the exact same `_resolve_matmul_fanout_branches` safety
#     net every other backbone tensor's extra fan-out already does --
#     declining the whole group if that extra reader can't be resolved to
#     an ordinary safe consumer, precisely as it already would for a plain
#     (non-gated) shared producer feeding two independent merges (see
#     `test_structured_pruning_matmul_residual_add_declines_on_gated_output_shared_with_second_merge`).
#   - A gate/up producer whose weight happens to be shared (tied) with
#     another leaf producer anywhere in the group -- gated or not -- is
#     caught by `_find_matmul_residual_chains`'s own existing degenerate
#     "same producer weight named twice" check below, unchanged; and a
#     weight shared with some *other*, unrelated chain entirely is caught by
#     `_apply_chains`'s own cross-chain touched-role tracking, also
#     unchanged. Neither needed to learn anything new about a gated pair.
#
#   So the composition adds no new correctness surface: every hazard it
#   could in principle introduce is already an instance of a hazard one of
#   the two mechanisms independently guards against. Both of
#   `_find_gated_chains`'s own two recognized shapes (see its own docstring)
#   are folded in here: a plain `Mul`, as above, *and* the native fused
#   `SwiGLU(a, b[, alpha])` op (opset 28+) -- a second, independent reuse of
#   `_find_gated_chains`'s own `SwiGLU`-branch extraction, which differs
#   from the `Mul` case in one respect the safety argument above has to
#   re-derive rather than inherit for free: `SwiGLU`'s swish lives entirely
#   *inside* the op (there's no separate activation node on the gate branch
#   the way an unfused `Sigmoid`/`Gelu` gate has), so `_find_gated_chains`
#   itself never calls `_trace_gate_producer_backward` for this shape at
#   all -- `a`/`b` must already *be* a real producer's own raw output, with
#   nothing in between, checked with the exact same single-consumer/
#   not-a-graph-output bar (`_find_gated_chains`'s own `_is_internal`) as
#   `_trace_gate_producer_backward`'s own bar, just applied directly instead
#   of after a pre-op walk. That's a strictly *tighter* shape than the `Mul`
#   case (zero permitted pre-ops rather than a bounded walk through unary
#   activations), so every part of the safety argument above -- the
#   single-consumer bar on both operands, the combine node's own output
#   becoming an ordinary backbone tensor for `_resolve_matmul_fanout_branches`
#   to police, and the existing tied-weight check -- carries over unchanged;
#   `alpha`, `SwiGLU`'s only other input, is a node attribute rather than a
#   tensor, so there is nothing about it for this pass to slice or conflict
#   over. What remains deliberately out of scope, a narrower scope choice
#   rather than a safety one: only exactly `_find_gated_chains`'s own two
#   shapes are recognized -- a gate activation exported as more than one
#   node (e.g. SiLU as `x * Sigmoid(x)`) is invisible to
#   `_find_gated_chains` itself already and stays that way here too, no new
#   gap introduced. See
#   `test_structured_pruning_matmul_residual_add_prunes_gated_branch_with_no_projection`,
#   `test_structured_pruning_matmul_residual_add_prunes_swiglu_branch_with_no_projection`,
#   and
#   `test_structured_pruning_matmul_residual_add_declines_on_swiglu_branch_with_extra_fanout`.
# - **A residual branch whose backward walk would need to cross a fused
#   self-attention op boundary** (`com.microsoft::Attention`,
#   `GroupQueryAttention`, or the plain `ai.onnx` `Attention` -- see the
#   "Attention-head pruning" section far below) to reach a real producer --
#   e.g. `Add(x, GroupQueryAttention(q, k, v))` with no output projection
#   MatMul between the attention op and the `Add`. Unlike the gated case
#   above, there's no analogous "resolve every real producer feeding it"
#   fallback available -- none of those ops is a MatMul/Gemm
#   (`_match_producer` never matches them), an `Add` (never an
#   eligible-merge candidate), or one of `_UNARY_PASS_THROUGH`'s shape-preserving
#   activations, and unlike a `Mul` node there's no elementwise-combine
#   structure to resolve two operands through at all -- so a residual branch
#   that bottoms out at one is simply unrecognized by any hop this walk
#   knows and falls through to `"fail"` -- the same outcome as any other
#   unmatched topology, not a special case that needed its own check. The
#   realistic version of this pattern -- an attention block's own
#   output-projection MatMul (`Wo`) feeding the residual `Add`, with
#   `GroupQueryAttention`/`Attention` sitting further upstream of `Wo` --
#   needs no special handling either, for the same reason the gated-FFN
#   case's own down-projection doesn't: the backward walk starts at the
#   `Add` operand and stops at the very first node it finds (`Wo`, an
#   ordinary MatMul/Gemm producer), never looking any further upstream at
#   what feeds `Wo`. See
#   `test_structured_pruning_matmul_residual_add_declines_on_bare_gqa_shortcut`.
#
# A bare `Add` is not, in practice, what the realistic target for this whole
# residual mechanism -- a transformer already run through onnxruntime's own
# transformer-optimizer tool, the same optimization pass that produces the
# `com.microsoft::Attention`/`GroupQueryAttention` fused ops this module
# already targets elsewhere -- actually has at each residual connection: that
# optimizer fuses `Add(input, skip)` (plus an optional per-channel bias
# `Add`) and the *following* `LayerNorm`/`SimplifiedLayerNormalization` into
# one `com.microsoft::SkipLayerNormalization`/`SkipSimplifiedLayerNormalization`
# node (`skip_layer_norm.cc`'s own `ComputeJob`, confirmed against
# onnxruntime's schema (`bert_defs.cc`) and by direct execution --
# `sum = input + skip (+ bias)`; `SkipLayerNormalization` computes ordinary
# LayerNorm on `sum` (population mean/variance, `* gamma + beta` if `beta`
# is given); `SkipSimplifiedLayerNormalization` -- the RMSNorm variant
# LLaMA-style models use -- drops `beta`/mean-centering entirely:
# `sum / sqrt(mean(sum**2) + epsilon) * gamma`). So a fully-optimized
# transformer typically has *no* bare `Add` at its residual connections at
# all, and without also recognizing this fused node the feature above would
# rarely fire on the models it exists for. `_match_matmul_residual_merge`
# closes that gap: such a node is simultaneously (1) the residual merge
# point itself -- its first two inputs, `input`/`skip`, are exactly `Add`'s
# two operands, walked backward the same way -- *and* (2) a per-channel
# affine hop on top of that sum, since `gamma` (and `beta`/`bias`, if
# given) scale/shift each surviving channel independently and so must be
# sliced by the group's own `keep` set precisely like a bias/scale `Add`/
# `Mul` hop's own constant already is. Rather than inventing a new shape for
# that, `_match_matmul_residual_merge` returns those constants as two or
# three synthetic `(node, const_name)` `chain_ops` entries against the same
# node -- `_Chain.chain_ops` already tolerates more than one entry per node
# (nothing about it assumes one entry per distinct node), so
# `_apply_chains`'s existing per-hop constant-slicing loop, its touched-role
# conflict tracking, and its stale-`value_info` cleanup all pick every one
# of them up with no changes of their own. `beta` (`SkipLayerNormalization`
# only) and `bias` are the op's own optional inputs -- simply absent
# becomes no slice needed for that term; *present but non-constant* (like
# `gamma`, required and always checked) declines the node outright, the
# same as a non-constant bias on a MatMul/Gemm producer. The op's optional
# `mean`/`inv_std_var` outputs are training-only bookkeeping onnxruntime's
# own CPU kernel never actually populates (`skip_layer_norm.cc`'s `Compute`
# only ever writes outputs 0 and 3); a real inference-exported graph should
# never wire them anywhere, but if one somehow does, this pass has no basis
# for whether pruning keeps those values meaningful for whatever reads them,
# so it declines outright rather than guessing, same as everywhere else in
# this pass. The optional fourth output, `input_skip_bias_sum` (the raw,
# pre-normalization sum), gets the same "declines if consumed" treatment,
# for a different reason: this pass never reads it itself, and in the
# common post-LN transformer shape a later residual connection's own `skip`
# operand is simply the *previous* `SkipLayerNormalization`'s ordinary
# (normalized) `output` -- not that raw sum -- so the "resolves to another
# eligible merge node's raw output" backward-walk case
# :func:`_walk_matmul_producer_backward` already handles for a chain of
# bare `Add`s covers a chain of `SkipLayerNormalization` nodes for free.
# But if *something else* in the graph reads `input_skip_bias_sum`
# directly, pruning still changes its width -- it's a plain runtime sum of
# `input`/`skip`, so it naturally comes out however wide those two end up
# post-pruning, no static array to reslice -- and this pass has no way to
# confirm that other consumer expects the new width rather than the
# original one (confirmed concretely: a second graph output reading it
# directly ends up with a shape mismatch against its own originally
# declared shape). So it's declined whenever consumed by anything, the
# same conservative bar `mean`/`inv_std_var` get, just for a shape reason
# rather than a values-still-meaningful one.


_SKIP_LAYER_NORM_OPS = ("SkipLayerNormalization", "SkipSimplifiedLayerNormalization")
_SKIP_LAYER_NORM_DOMAIN = "com.microsoft"


def _skip_layer_norm_const_names(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """If every constant input a ``com.microsoft::SkipLayerNormalization``/
    ``SkipSimplifiedLayerNormalization`` `node` needs sliced -- `gamma`
    (input 2, required), plus `beta` (input 3, ``SkipLayerNormalization``
    only) and `bias` (input 4, or input 3 for the simplified/RMSNorm
    variant, which has no `beta`), both optional -- is present exactly as
    the node's own input list says, and, whenever present, a constant float
    initializer shaped like a flat per-channel vector (``prod(dims) ==
    dims[-1]``, the same self-consistency bar
    :func:`_walk_matmul_producer_backward`'s own bias/scale hop check
    already uses -- the real ``dims[-1] == n_channels`` check is deferred to
    :func:`_find_matmul_residual_chains` once the group's real channel
    count is known, exactly like that hop's own), returns
    ``(gamma_name, beta_name_or_None, bias_name_or_None)``. `beta`/`bias`
    simply absent from the node's own input list (as opposed to present but
    non-constant) becomes ``None`` -- the corresponding term the kernel
    itself omits, confirmed against onnxruntime's own ``skip_layer_norm.cc``
    kernel and by direct execution (see this section's own comment).
    Declines (``None``) on a non-constant `gamma`, a *present* but
    non-constant `beta`/`bias`, or the same underlying tensor named for two
    of `gamma`/`beta`/`bias` at once (double-slicing it in
    :func:`_apply_chains`'s own per-hop loop would corrupt it) -- none of
    these is guessed at. The per-tensor validity check itself is
    :func:`_flat_channel_const`, shared verbatim with
    :func:`_norm_pass_through_const_names`'s own plain-`LayerNormalization`
    -family `scale`/`bias` check -- same schema fact, same check, regardless
    of which op reads it.
    """
    simplified = node.op_type == "SkipSimplifiedLayerNormalization"

    if (
        len(node.input) < 3
        or not node.input[2]
        or not _flat_channel_const(node.input[2], initializer_map)
    ):
        return None  # gamma is required
    gamma_name = node.input[2]

    beta_name: Optional[str] = None
    bias_idx = 3
    if not simplified:
        bias_idx = 4
        if len(node.input) > 3 and node.input[3]:
            if not _flat_channel_const(node.input[3], initializer_map):
                return None
            beta_name = node.input[3]

    bias_name: Optional[str] = None
    if len(node.input) > bias_idx and node.input[bias_idx]:
        if not _flat_channel_const(node.input[bias_idx], initializer_map):
            return None
        bias_name = node.input[bias_idx]

    names = [n for n in (gamma_name, beta_name, bias_name) if n is not None]
    if len(set(names)) != len(names):
        return None  # tied gamma/beta/bias -- double-slicing would corrupt it

    return gamma_name, beta_name, bias_name


def _match_matmul_residual_merge(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
) -> Optional[Tuple[Tuple[str, str], Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]]]:
    """The MatMul/Gemm residual finder's own eligible-merge-point check:
    `node` is either a bare ``Add`` (:func:`_is_eligible_add_merge`, reused
    unchanged, with no extra `chain_ops` of its own -- exactly today's
    behavior) *or* a ``com.microsoft::SkipLayerNormalization``/
    ``SkipSimplifiedLayerNormalization`` node (see this section's own
    comment above for the exact fused arithmetic and how it was confirmed).
    Its first two inputs (`input`, `skip`) play exactly the role `Add`'s two
    operands do -- same "two independent branches must agree on one
    channel-index set" merge point, same eligibility bar (distinct, both
    non-constant) -- while its constant `gamma`/`beta`/`bias` inputs (see
    :func:`_skip_layer_norm_const_names`) are a per-channel affine hop
    riding the very same node, so this returns them as extra
    ``(node, const_name)`` entries for the caller to fold into the resolved
    chain's own `chain_ops`, reusing :func:`_apply_chains`'s existing
    per-hop constant slicing verbatim -- the same way a bias/scale
    ``Add``/``Mul`` hop's own single constant already does, just two or
    three entries against the same node instead of one.

    Declines (``None``) the same way :func:`_skip_layer_norm_const_names`
    does for a non-constant/tied `gamma`/`beta`/`bias`, and additionally
    whenever any of the op's optional secondary outputs -- `mean`/
    `inv_std_var` (training-only; onnxruntime's own CPU kernel never
    actually writes them) *or* `input_skip_bias_sum` (the raw pre-norm sum)
    -- are actually consumed by anything else in the graph. `mean`/
    `inv_std_var`: this pass has no basis for whether pruning keeps those
    still meaningful for whatever reads them. `input_skip_bias_sum` is
    different in kind -- this pass never reads it itself, and its *shape*
    (not its meaningfulness) is what's at risk: it naturally comes out
    however wide `input`/`skip` end up post-pruning (a plain runtime sum of
    two already-consistently-pruned tensors, nothing to reslice), but any
    *other* consumer of it outside this chain has no idea that width just
    changed and may expect the original one -- confirmed concretely: a
    second graph output reading it directly ends up with a shape mismatch
    against its own originally-declared shape once pruned. So this output
    is held to the same "not consumed elsewhere" bar as `mean`/
    `inv_std_var`, not because its value would be wrong, but because
    nothing here can confirm whatever reads it still expects the resulting
    shape -- the same "no basis to guess a shape survives" reasoning this
    module already applies to fan-out generally.
    """
    if _is_eligible_add_merge(node, initializer_map):
        return (node.input[0], node.input[1]), ()

    if (
        node.domain != _SKIP_LAYER_NORM_DOMAIN
        or node.op_type not in _SKIP_LAYER_NORM_OPS
    ):
        return None
    if len(node.input) < 3:
        return None
    input_name, skip_name = node.input[0], node.input[1]
    if (
        not input_name
        or not skip_name
        or input_name == skip_name
        or input_name in initializer_map
        or skip_name in initializer_map
    ):
        return None

    const_names = _skip_layer_norm_const_names(node, initializer_map)
    if const_names is None:
        return None
    gamma_name, beta_name, bias_name = const_names

    for out_idx in (1, 2, 3):  # mean, inv_std_var, input_skip_bias_sum
        if len(node.output) > out_idx and node.output[out_idx]:
            out_name = node.output[out_idx]
            if consumers_of.get(out_name) or out_name in graph_outputs:
                return None

    extra_ops = tuple(
        (node, name) for name in (gamma_name, beta_name, bias_name) if name is not None
    )
    return (input_name, skip_name), extra_ops


def _walk_matmul_producer_backward(
    start: str,
    node_by_output: Dict[str, onnx.NodeProto],
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    max_hops: int,
    producer_infos: Optional[
        Dict[str, Tuple[onnx.NodeProto, str, bool, Optional[str], int]]
    ] = None,
) -> Tuple[
    str,
    Optional[
        Union[Tuple[_Producer, int], Tuple[_Producer, _Producer, int], onnx.NodeProto]
    ],
    Tuple[Tuple[onnx.NodeProto, Optional[str]], ...],
    Tuple[Tuple[str, onnx.NodeProto], ...],
]:
    """The backward counterpart of :func:`_walk_to_consumer`, used only by
    :func:`_find_matmul_residual_chains` to resolve one operand of an
    eligible merge node (see :func:`_match_matmul_residual_merge`) back to
    whatever produces it -- the MatMul/Gemm analogue of
    :func:`_walk_conv_producer_backward` (see this function's own section
    comment above for how the two differ: a wider hop set mirroring
    `_walk_to_consumer`'s own per-channel bias/scale ``Add``/``Mul`` hop,
    and no depthwise-pass-through analogue at all). Declines (only) whenever
    a tensor crossed -- `start` itself included -- is a graph output (a
    caller-observed shape this pass never resizes); *how many* other things
    also read that same tensor is deliberately **not** checked here -- see
    :func:`_find_matmul_residual_chains`'s own "fan-out" section comment for
    why, and how every such extra reader still gets its own safety check,
    just later, once the group's real channel count is known. The usual
    exactly-one-output check on every node crossed is relaxed to "its own
    *first* output is `cur`" rather than "it has exactly one output" purely
    to let a multi-output ``SkipLayerNormalization``-family node (`mean`/
    `inv_std_var`/`input_skip_bias_sum`, all beyond its primary `output`)
    through to :func:`_match_matmul_residual_merge`'s own check below --
    every other node type this walk ever matches (`MatMul`/`Gemm`, a unary
    activation, `Add`/`Mul`) already has exactly one output per its own
    ONNX schema, so this is a no-op relaxation for them.

    `producer_infos` is :func:`_find_gated_chains`'s own producer-lookup map
    (raw producer output -> match info), built once by the caller and passed
    through unchanged -- needed to resolve a gated ``Mul`` hop via
    :func:`_trace_gate_producer_backward`, and a native fused ``SwiGLU`` hop
    via a direct lookup of its own two raw operands (see this section's own
    comment above for the composition-safety argument, covering both
    shapes); every other hop ignores it. Left ``None`` (the default),
    neither a `Mul` of two non-constant operands nor a `SwiGLU` node is ever
    resolved as a gated pair, and both simply fall through to `"fail"` the
    same way they always have -- :func:`_find_matmul_concat_chains` relies
    on exactly that unchanged behavior for its own (unrelated) reuse of this
    same walker, since composing a gated combine with a `Concat` merge on
    the same branch is a separate question this module's own docstring
    already declines and this parameter deliberately doesn't touch.

    Returns one of:

    - ``("producer", (producer, n_channels), chain_ops, edges)`` -- resolved
      all the way back to a real MatMul/vanilla-Gemm producer;
    - ``("gated", (producer_a, producer_b, n_channels), chain_ops, edges)``
      -- resolved to a gated (SwiGLU/GeGLU-style) combine of two
      non-constant operands -- either a plain `Mul`, each operand in turn
      walked back to its own real MatMul/vanilla-Gemm producer via
      :func:`_trace_gate_producer_backward`, or the native fused `SwiGLU`
      node, each operand required to already *be* such a producer's own raw
      output (see this section's own comment above for why the two shapes
      differ there) -- both producers, not just one, belong to this group's
      own shared leaf-producer set;
    - ``("add", merge_node, chain_ops, edges)`` -- resolved to another
      eligible merge node's raw output instead -- a bare ``Add`` or a
      ``SkipLayerNormalization``-family node alike (the caller unions this
      group with that node's own rather than treating it as a separate
      producer);
    - ``("fail", None, (), ())`` -- a graph input, an unrecognized producer
      (attention-op boundary, a gated ``Mul``/``SwiGLU`` whose operands
      don't both resolve, ...), a graph output crossed mid-walk, or the hop
      limit -- the caller declines the whole group this operand belongs to.

    `chain_ops` mirrors :class:`_Chain`'s own field exactly (each entry a
    ``(node, const_name_or_None)`` pair, in forward -- producer-to-merge --
    order), so it can be concatenated directly into the resolved chain's
    `chain_ops` the same way `_find_chains` builds them for an ordinary
    single-producer chain. `edges` mirrors
    :func:`_walk_conv_producer_backward`'s own field exactly -- see its
    docstring for what it records and why. A gated ``Mul``/``SwiGLU``'s own
    two operands are deliberately *not* added to `edges`: unlike this walk's
    own deferred bias/scale-hop tensors, a `Mul`'s operands (via
    `_trace_gate_producer_backward`) and a `SwiGLU`'s operands (via the
    same single-consumer/not-a-graph-output bar, checked directly) are both
    already held to an exact single-consumer bar (see this section's own
    comment above), so there is no extra fan-out for a later pass to resolve
    and nothing for `edges`/`backbone_tensors` to track there.
    """
    chain_ops: List[Tuple[onnx.NodeProto, Optional[str]]] = []
    edges: List[Tuple[str, onnx.NodeProto]] = []
    cur = start
    for _hop in range(max_hops):
        if cur in graph_outputs:
            return "fail", None, (), ()
        node = node_by_output.get(cur)
        if node is None or not node.output or node.output[0] != cur:
            return "fail", None, (), ()

        prod_info = _match_producer(node, initializer_map)
        if prod_info is not None:
            w_name, weight_transposed, bias_name, n_channels = prod_info
            producer = _Producer(node, w_name, weight_transposed, bias_name)
            return (
                "producer",
                (producer, n_channels),
                tuple(reversed(chain_ops)),
                tuple(edges),
            )

        if node.op_type in _UNARY_PASS_THROUGH and len(node.input) == 1:
            chain_ops.append((node, None))
            edges.append((node.input[0], node))
            cur = node.input[0]
            continue

        if node.op_type in _BINARY_CHANNEL_OPS and len(node.input) == 2:
            a_name, b_name = node.input
            a_const = a_name in initializer_map
            b_const = b_name in initializer_map
            if a_const != b_const:
                const_name, other = (a_name, b_name) if a_const else (b_name, a_name)
                const_init = initializer_map[const_name]
                if (
                    _is_supported_float_dtype(const_init.data_type)
                    and list(const_init.dims)
                    and int(np.prod(const_init.dims)) == const_init.dims[-1]
                ):
                    chain_ops.append((node, const_name))
                    edges.append((other, node))
                    cur = other
                    continue
                return "fail", None, (), ()
            # Both operands constant (degenerate) or both non-constant: for
            # `Add` the latter is exactly `_is_eligible_add_merge`'s own
            # shape, handled below by the merge check. For `Mul` it's a
            # gated (SwiGLU/GeGLU) combine point -- resolved by walking
            # *both* non-constant operands back to their own real producers
            # (see this section's own comment above for why this is safe to
            # do rather than picking one), reusing `_find_gated_chains`'s
            # own gate-branch tracer unchanged.
            if (
                producer_infos is not None
                and node.op_type == "Mul"
                and not a_const
                and not b_const
                and a_name != b_name
            ):
                trace_a = _trace_gate_producer_backward(
                    a_name,
                    node_by_output,
                    producer_infos,
                    consumers_of,
                    graph_outputs,
                    max_hops,
                )
                trace_b = _trace_gate_producer_backward(
                    b_name,
                    node_by_output,
                    producer_infos,
                    consumers_of,
                    graph_outputs,
                    max_hops,
                )
                if trace_a is not None and trace_b is not None:
                    info_a, pre_a = trace_a
                    info_b, pre_b = trace_b
                    node_a, n_a = info_a[0], info_a[4]
                    node_b, n_b = info_b[0], info_b[4]
                    if node_a is not node_b and n_a == n_b:
                        producer_a = _Producer(
                            info_a[0], info_a[1], info_a[2], info_a[3], pre_a
                        )
                        producer_b = _Producer(
                            info_b[0], info_b[1], info_b[2], info_b[3], pre_b
                        )
                        return (
                            "gated",
                            (producer_a, producer_b, n_a),
                            tuple(reversed(chain_ops)),
                            tuple(edges),
                        )
            # Not a resolvable gated pair either -- falls through to the
            # merge check (which requires `Add` or a
            # ``SkipLayerNormalization``-family node specifically) or
            # `"fail"`. `SwiGLU` is never matched here -- it isn't in
            # `_BINARY_CHANNEL_OPS` at all -- it gets its own check below.

        if (
            producer_infos is not None
            and node.op_type == "SwiGLU"
            and len(node.input) == 2
            and len(node.output) == 1
        ):
            # The native fused SwiGLU(a, b[, alpha]) = swish(a) * b (opset
            # 28+) op, reusing _find_gated_chains's own SwiGLU-branch
            # extraction verbatim (see this section's own comment above for
            # the composition-safety argument, re-derived against this
            # shape specifically): unlike a plain `Mul`, SwiGLU's swish
            # lives entirely *inside* the op, so `a`/`b` must be the two
            # producers' own raw outputs with nothing in between -- no
            # _trace_gate_producer_backward pre-op walk here, just a direct
            # producer_infos lookup, each held to the same single-consumer/
            # not-a-graph-output bar _find_gated_chains's own `_is_internal`
            # applies (consumers_of/graph_outputs are threaded through
            # unchanged). `alpha`, if present, is a node attribute, not a
            # tensor input -- nothing for this pass to slice, so it needs no
            # attention here.
            a_name, b_name = node.input
            if a_name not in initializer_map and b_name not in initializer_map:
                info_a_lookup = producer_infos.get(a_name)
                info_b_lookup = producer_infos.get(b_name)
                if (
                    info_a_lookup is not None
                    and info_b_lookup is not None
                    and len(consumers_of.get(a_name, [])) == 1
                    and a_name not in graph_outputs
                    and len(consumers_of.get(b_name, [])) == 1
                    and b_name not in graph_outputs
                ):
                    node_a, n_a = info_a_lookup[0], info_a_lookup[4]
                    node_b, n_b = info_b_lookup[0], info_b_lookup[4]
                    if node_a is not node_b and n_a == n_b:
                        producer_a = _Producer(
                            info_a_lookup[0],
                            info_a_lookup[1],
                            info_a_lookup[2],
                            info_a_lookup[3],
                            (),
                        )
                        producer_b = _Producer(
                            info_b_lookup[0],
                            info_b_lookup[1],
                            info_b_lookup[2],
                            info_b_lookup[3],
                            (),
                        )
                        return (
                            "gated",
                            (producer_a, producer_b, n_a),
                            tuple(reversed(chain_ops)),
                            tuple(edges),
                        )
            # Not a resolvable gated pair -- SwiGLU is never an eligible
            # merge node either (_match_matmul_residual_merge only matches
            # `Add`/`SkipLayerNormalization`-family nodes), so this falls
            # through to "fail" below, same as any other unmatched shape.

        if node.op_type in _FUSED_BIAS_GELU_OPS:
            fused = _match_fused_bias_gelu(node, initializer_map)
            if fused is not None:
                data_name, bias_name = fused
                chain_ops.append((node, bias_name))
                edges.append((data_name, node))
                cur = data_name
                continue
            return "fail", None, (), ()

        merge = _match_matmul_residual_merge(
            node, initializer_map, consumers_of, graph_outputs
        )
        if merge is not None:
            return "add", node, tuple(reversed(chain_ops)), tuple(edges)

        return "fail", None, (), ()

    return "fail", None, (), ()


def _resolve_matmul_fanout_branches(
    backbone_tensors: List[str],
    accounted: Dict[str, Set[int]],
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    n_channels: int,
    value_info_by_name: Optional[Dict[str, onnx.ValueInfoProto]] = None,
) -> Optional[List[_ConsumerBranch]]:
    """The MatMul/Gemm analogue of :func:`_resolve_conv_fanout_branches` --
    see its own docstring for the shared reasoning this mirrors exactly
    (only the forward walker differs: :func:`_walk_to_consumer` instead of
    :func:`_walk_to_conv_consumer`), and there is no Conv-style grouped-
    consumer or depthwise-pass-through concept to check or carry for a
    MatMul/Gemm branch at all. `value_info_by_name`, when given, is threaded
    straight through to :func:`_walk_to_consumer`'s own same-named parameter
    -- see there for what it enables.
    """
    branches: List[_ConsumerBranch] = []
    seen_weights: Set[str] = set()
    for tensor in backbone_tensors:
        if tensor in graph_outputs:
            return None
        seen_nodes: Set[int] = set()
        for consumer_node in consumers_of.get(tensor, []):
            if id(consumer_node) in seen_nodes:
                continue
            seen_nodes.add(id(consumer_node))
            if id(consumer_node) in accounted.get(tensor, ()):
                continue  # already part of the group's own established wiring
            resolved, br_chain_ops = _walk_to_consumer(
                tensor,
                initializer_map,
                consumers_of,
                graph_outputs,
                n_channels,
                _MAX_CHAIN_HOPS,
                forced_first_hop=consumer_node,
                value_info_by_name=value_info_by_name,
            )
            if resolved is None:
                return None
            branch_node, branch_weight, branch_weight_transposed = resolved
            if branch_weight in seen_weights:
                return None  # two branches naming the same consumer weight
            seen_weights.add(branch_weight)
            branches.append(
                _ConsumerBranch(
                    chain_ops=br_chain_ops,
                    consumer_node=branch_node,
                    consumer_weight=branch_weight,
                    consumer_weight_transposed=branch_weight_transposed,
                    consumer_is_conv=False,
                )
            )
    return branches


def _find_matmul_residual_chains(graph: onnx.GraphProto) -> List[_Chain]:
    """Finds MatMul/Gemm residual/skip-connection groups -- see this
    section's own comment above and :func:`_find_conv_residual_chains`'s
    (this function mirrors that one's union-find structure exactly, over
    :func:`_walk_matmul_producer_backward` instead of
    :func:`_walk_conv_producer_backward`). Every eligible merge point
    (:func:`_match_matmul_residual_merge` -- a bare ``Add`` or a
    ``SkipLayerNormalization``-family node) contributes its own extra
    `chain_ops` (empty for `Add`; `gamma`/`beta`/`bias` for the
    normalization-fused case) up front, before any union-find grouping, so
    every member of a resolved group -- not just its "sink" -- has its own
    per-channel constants, if any, folded into the final chain the same
    way. For every maximal union-find group of transitively-connected
    eligible merge points, resolves every member's two operands: each must
    reach either a real MatMul/vanilla-Gemm producer (a "leaf" of the
    group) or another eligible merge node already in the same group. If
    *any* operand, anywhere in the group, fails to resolve that way, or the
    leaf producers' channel counts don't all agree, the *entire* group is
    declined -- never partially pruned. Every tensor visited along the way
    (see :func:`_walk_matmul_producer_backward`'s own `edges`) plus the
    group's own "sink" (the one member whose own output isn't itself
    consumed by another member) is then handed to
    :func:`_resolve_matmul_fanout_branches`, which finds and resolves every
    extra (non-backbone) consumer fan-out reaches -- declining the whole
    group if any such branch can't be resolved, exactly as
    :func:`_find_conv_residual_chains` does. What survives is one or more
    independent forward branches, all fed by the exact same shared `keep`
    set once :func:`_apply_chains` computes it.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    node_by_output = {out: node for node in graph.node for out in node.output}
    graph_outputs = {o.name for o in graph.output}
    value_info_by_name = _value_info_by_name(graph)

    # _find_gated_chains's own producer-lookup map, built once here and
    # threaded through every _walk_matmul_producer_backward call below --
    # needed only to resolve a gated Mul hop via
    # _trace_gate_producer_backward (see this section's own comment above).
    producer_infos: Dict[str, Tuple[onnx.NodeProto, str, bool, Optional[str], int]] = {}
    for node in graph.node:
        info = _match_producer(node, initializer_map)
        if info is not None:
            w_name, weight_transposed, bias_name, n_channels = info
            producer_infos[node.output[0]] = (
                node,
                w_name,
                weight_transposed,
                bias_name,
                n_channels,
            )

    Merge = Tuple[
        onnx.NodeProto,
        Tuple[str, str],
        Tuple[Tuple[onnx.NodeProto, Optional[str]], ...],
    ]
    merges: List[Merge] = []
    for node in graph.node:
        match = _match_matmul_residual_merge(
            node, initializer_map, consumers_of, graph_outputs
        )
        if match is not None:
            operands, extra_ops = match
            merges.append((node, operands, extra_ops))
    if not merges:
        return []
    merge_index = {id(m[0]): i for i, m in enumerate(merges)}

    parent = list(range(len(merges)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    Edge = Tuple[
        str,
        Optional[
            Union[
                Tuple[_Producer, int], Tuple[_Producer, _Producer, int], onnx.NodeProto
            ]
        ],
        Tuple[Tuple[onnx.NodeProto, Optional[str]], ...],
        Tuple[Tuple[str, onnx.NodeProto], ...],
    ]
    edge_results: Dict[int, List[Edge]] = {}
    poisoned: Set[int] = set()
    for idx, (merge_node, operands, _extra_ops) in enumerate(merges):
        results: List[Edge] = []
        for operand in operands:
            edge = _walk_matmul_producer_backward(
                operand,
                node_by_output,
                initializer_map,
                consumers_of,
                graph_outputs,
                _MAX_CHAIN_HOPS,
                producer_infos,
            )
            results.append(edge)
            kind, payload = edge[0], edge[1]
            if kind == "fail":
                poisoned.add(idx)
            elif kind == "add":
                assert isinstance(payload, onnx.NodeProto)
                j = merge_index.get(id(payload))
                if j is None:
                    poisoned.add(idx)  # defensive -- shouldn't happen
                else:
                    union(idx, j)
        edge_results[idx] = results

    groups: Dict[int, List[int]] = {}
    for idx in range(len(merges)):
        groups.setdefault(find(idx), []).append(idx)

    chains: List[_Chain] = []
    for members in groups.values():
        if any(i in poisoned for i in members):
            continue

        leaf_producers: List[_Producer] = []
        n_channels_set: Set[int] = set()
        pre_chain_ops: List[Tuple[onnx.NodeProto, Optional[str]]] = []
        referenced: Set[int] = set()
        # See _find_conv_residual_chains's own matching comment: every
        # tensor either operand walk of every member proved carries this
        # group's own shared channel-index set, and which specific consumer
        # node is already part of the group's own internal wiring.
        backbone_tensors: List[str] = []
        accounted: Dict[str, Set[int]] = {}

        def _mark_backbone(tensor: str, node: onnx.NodeProto) -> None:
            if tensor not in accounted:
                backbone_tensors.append(tensor)
            accounted.setdefault(tensor, set()).add(id(node))

        for idx in members:
            merge_node = merges[idx][0]
            operands = merges[idx][1]
            pre_chain_ops.extend(merges[idx][2])  # this merge node's own extra_ops
            for operand, (kind, payload, ops, edges) in zip(
                operands, edge_results[idx]
            ):
                _mark_backbone(operand, merge_node)
                for tensor, node in edges:
                    _mark_backbone(tensor, node)
                pre_chain_ops.extend(ops)
                if kind == "producer":
                    assert payload is not None and not isinstance(
                        payload, onnx.NodeProto
                    )
                    producer, n_channels = cast(Tuple[_Producer, int], payload)
                    leaf_producers.append(producer)
                    n_channels_set.add(n_channels)
                elif kind == "gated":
                    # A gated (SwiGLU/GeGLU) combine (a plain Mul, or the
                    # native fused SwiGLU op) of two non-constant operands,
                    # each already walked back to its own real producer --
                    # see _walk_matmul_producer_backward's own
                    # "gated" return-kind docstring and this section's own
                    # comment above. Both producers join this group's shared
                    # leaf-producer set, exactly like an ordinary gated
                    # pair's two producers already do outside a residual
                    # chain (_find_gated_chains); the degenerate "same
                    # producer weight named twice" check below still catches
                    # a tied weight between the two, or against any other
                    # leaf producer already in this group.
                    assert payload is not None and not isinstance(
                        payload, onnx.NodeProto
                    )
                    producer_a, producer_b, n_channels = cast(
                        Tuple[_Producer, _Producer, int], payload
                    )
                    leaf_producers.append(producer_a)
                    leaf_producers.append(producer_b)
                    n_channels_set.add(n_channels)
                elif kind == "add":
                    assert isinstance(payload, onnx.NodeProto)
                    referenced.add(merge_index[id(payload)])

        if len(n_channels_set) != 1:
            continue  # branches disagree on channel count -- decline
        n_channels = next(iter(n_channels_set))

        # Every bias/scale hop's constant (an Add/Mul hop's own, or a fused
        # BiasGelu/FastGelu hop's own bias -- see _match_fused_bias_gelu),
        # and every SkipLayerNorm-family merge's own gamma/beta/bias, was
        # only self-consistently checked when first crossed/matched (see
        # _walk_matmul_producer_backward, _match_matmul_residual_merge); now
        # that the group's real channel count is known, re-validate it
        # actually matches -- mirroring the depthwise-Conv-hop re-validation
        # in _find_conv_residual_chains.
        if any(
            const_name is not None
            and initializer_map[const_name].dims[-1] != n_channels
            for _, const_name in pre_chain_ops
        ):
            continue

        sinks = [idx for idx in members if idx not in referenced]
        if len(sinks) != 1:
            continue  # not a single linear chain of merges -- decline
        sink_node = merges[sinks[0]][0]

        if len({p.weight for p in leaf_producers}) != len(leaf_producers):
            continue  # degenerate -- the same producer named twice

        # The sink's own output is never a backbone tensor via any member's
        # own operand walk (nothing in the group walks *through* it -- see
        # _find_conv_residual_chains's own matching comment), so it needs
        # adding explicitly, with no accounted-for consumer of its own yet.
        sink_out = sink_node.output[0]
        if sink_out not in accounted:
            backbone_tensors.append(sink_out)
            accounted[sink_out] = set()

        branches = _resolve_matmul_fanout_branches(
            backbone_tensors,
            accounted,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_channels,
            value_info_by_name=value_info_by_name,
        )
        if not branches:
            continue

        primary, extra_branches = branches[0], tuple(branches[1:])
        chain_ops = (
            tuple(pre_chain_ops)
            + tuple((merges[i][0], None) for i in members)
            + primary.chain_ops
        )

        chains.append(
            _Chain(
                producers=tuple(leaf_producers),
                chain_ops=chain_ops,
                consumer_node=primary.consumer_node,
                consumer_weight=primary.consumer_weight,
                consumer_weight_transposed=primary.consumer_weight_transposed,
                n_channels=n_channels,
                extra_consumers=extra_branches,
            )
        )
    return chains


def _trace_gate_producer_backward(
    tensor_name: str,
    node_by_output: Dict[str, onnx.NodeProto],
    producer_infos: Dict[str, Tuple[onnx.NodeProto, str, bool, Optional[str], int]],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    max_hops: int,
) -> Optional[
    Tuple[
        Tuple[onnx.NodeProto, str, bool, Optional[str], int], Tuple[onnx.NodeProto, ...]
    ]
]:
    """Walks backward from `tensor_name` through unary activation ops
    (Sigmoid, Gelu, ...) until it resolves to a matmul-like producer's raw
    output -- the mirror image of :func:`_walk_to_consumer`'s forward walk,
    used to recognize a gate branch's own activation (e.g. SwiGLU's
    ``silu(gate)`` when exported as separate Sigmoid/Mul-by-a-second-
    operand rather than a single node -- see :func:`_find_gated_chains`).
    Every tensor walked through, `tensor_name` itself included, must have
    exactly one consumer and not be a graph output: the same safety bar
    the forward walk holds every intermediate tensor to.
    """
    pre_ops: List[onnx.NodeProto] = []
    cur = tensor_name
    for _ in range(max_hops):
        if len(consumers_of.get(cur, [])) != 1 or cur in graph_outputs:
            return None
        if cur in producer_infos:
            return producer_infos[cur], tuple(reversed(pre_ops))
        producer_node = node_by_output.get(cur)
        if producer_node is None:
            return None
        if not (
            producer_node.op_type in _UNARY_PASS_THROUGH
            and len(producer_node.input) == 1
            and len(producer_node.output) == 1
        ):
            return None
        pre_ops.append(producer_node)
        cur = producer_node.input[0]
    return None


def _find_gated_chains(graph: onnx.GraphProto) -> List[_Chain]:
    """Finds gated FFN blocks -- SwiGLU/GeGLU-style ``down(act(gate(x)) *
    up(x))``, the FFN architecture most current LLMs use (Llama, Mistral,
    Qwen, Gemma, ...) -- that :func:`_find_chains` cannot see at all,
    because it only ever follows a *single* producer's output. Two
    matmul-like producers (gate and up) whose outputs, each optionally
    through its own activation, combine via one of:

    - a plain elementwise ``Mul`` of two non-constant operands (covers an
      unactivated GLU, or any activation expressed as ordinary unary ops
      -- e.g. GeGLU's ``Gelu``); or
    - ONNX's native fused ``SwiGLU(a, b[, alpha]) = swish(a) * b`` node
      (opset 28+), whose swish lives entirely inside the op, so ``a``/``b``
      must be the two producers' raw outputs with nothing in between,

    with no other consumer anywhere along either branch or at the combine
    point, into exactly one downstream MatMul/vanilla-Gemm's reduction
    dimension, are pruned together: both branches must drop the *same*
    output-channel indices, since they're about to be multiplied
    elementwise. A gate activation decomposed into more than one node
    (e.g. SiLU exported as the self-referencing ``x * Sigmoid(x)`` rather
    than a single ``Sigmoid``/native ``Swish``) isn't recognized -- that
    block is safely left untouched, not guessed at.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}
    node_by_output = {out: node for node in graph.node for out in node.output}
    value_info_by_name = _value_info_by_name(graph)

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    producer_infos: Dict[str, Tuple[onnx.NodeProto, str, bool, Optional[str], int]] = {}
    for node in graph.node:
        info = _match_producer(node, initializer_map)
        if info is not None:
            w_name, weight_transposed, bias_name, n_channels = info
            producer_infos[node.output[0]] = (
                node,
                w_name,
                weight_transposed,
                bias_name,
                n_channels,
            )

    def _producer(info, pre_ops) -> _Producer:
        node, w_name, weight_transposed, bias_name, _n = info
        return _Producer(node, w_name, weight_transposed, bias_name, pre_ops)

    chains: List[_Chain] = []
    for node in graph.node:
        if node.op_type == "Mul" and len(node.input) == 2 and len(node.output) == 1:
            a_name, b_name = node.input
            if (
                a_name == b_name
                or a_name in initializer_map
                or b_name in initializer_map
            ):
                continue
            trace_a = _trace_gate_producer_backward(
                a_name,
                node_by_output,
                producer_infos,
                consumers_of,
                graph_outputs,
                _MAX_CHAIN_HOPS,
            )
            trace_b = _trace_gate_producer_backward(
                b_name,
                node_by_output,
                producer_infos,
                consumers_of,
                graph_outputs,
                _MAX_CHAIN_HOPS,
            )
            if trace_a is None or trace_b is None:
                continue
            info_a, pre_a = trace_a
            info_b, pre_b = trace_b
        elif (
            node.op_type == "SwiGLU" and len(node.input) == 2 and len(node.output) == 1
        ):
            a_name, b_name = node.input
            if a_name in initializer_map or b_name in initializer_map:
                continue
            if not (_is_internal(a_name) and _is_internal(b_name)):
                continue
            info_a_lookup = producer_infos.get(a_name)
            info_b_lookup = producer_infos.get(b_name)
            if info_a_lookup is None or info_b_lookup is None:
                continue
            info_a, pre_a = info_a_lookup, ()
            info_b, pre_b = info_b_lookup, ()
        else:
            continue

        node_a, n_a = info_a[0], info_a[4]
        node_b, n_b = info_b[0], info_b[4]
        if node_a is node_b or n_a != n_b:
            continue

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops = _walk_to_consumer(
            out_name,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_a,
            _MAX_CHAIN_HOPS,
            value_info_by_name=value_info_by_name,
        )
        if consumer is None:
            continue

        chains.append(
            _Chain(
                producers=(_producer(info_a, pre_a), _producer(info_b, pre_b)),
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
                n_channels=n_a,
            )
        )
    return chains


# --- Concat-merged (skip-connection) chains ---------------------------------
#
# A `Concat` merge -- the U-Net-style encoder/decoder skip connection,
# `merged = Concat(a, b, ..., axis=C)` -- looks, at first glance, like it
# needs the same general dependency-graph machinery an `Add`/
# `SkipLayerNormalization` merge does (see the two residual sections above),
# and this module long declined it outright on that assumption (see its own
# docstring's prior "non-Add merges (`Concat`, ...)" phrase). It turns out
# not to need that: unlike `Add`, whose operands are summed
# position-for-position and therefore *must* agree on one shared surviving
# channel-index set (the entire reason the two residual sections above exist
# at all), `Concat`'s branches are structurally independent. Branch `a`
# (`Ca` channels) always owns columns `[0, Ca)` of the merged, pre-pruning
# tensor; branch `b` always owns `[Ca, Ca+Cb)`; and so on for every further
# operand -- fixed, disjoint offsets into the *original* channel range that
# neither branch's own pruning choice can move, since ONNX's `Concat`
# simply lays its inputs out end to end in operand order. So each branch can
# be ranked and pruned *entirely on its own* -- no cross-branch agreement
# needed at all, unlike a gated pair or a residual group -- and the only new
# work is on the *consumer* side: its weight needs slicing at those same
# fixed block offsets, one independently-chosen `keep` set per block,
# concatenated back together in branch order (:func:`_apply_concat_chains`).
#
# Both node families this module already splits its producer/consumer
# matching by get their own finder here, exactly mirroring the
# `_find_chains`/`_find_conv_chains` split: :func:`_find_matmul_concat_chains`
# (MatMul/vanilla-Gemm) and :func:`_find_conv_concat_chains` (2-D, `group=1`
# Conv). Both resolve every one of a `Concat` node's operands *backward* to a
# real producer, reusing the exact same backward walkers the two residual
# sections above already built and verified --
# :func:`_walk_matmul_producer_backward`/:func:`_walk_conv_producer_backward`
# -- rather than writing new ones: those walkers already hold every
# intermediate tensor to the single-consumer safety bar this pass needs
# (`start` itself included, so a branch that also fans out anywhere else
# fails on its very first hop), and already resolve through the same unary
# activations (plus, for MatMul/Gemm, a per-channel `Add`/`Mul`/
# `BiasGelu`/`FastGelu` hop; for Conv, a self-consistently-depthwise
# pass-through hop) a plain single-producer chain's own forward walk
# recognizes. A `"producer"` outcome is accepted directly, as before. A
# `"add"` outcome -- the branch resolves to an eligible `Add`/
# `SkipLayerNormalization` residual merge instead of a real producer -- is
# now *composed*, but only in one bounded shape (see
# :func:`_resolve_matmul_residual_group_for_concat`/
# :func:`_resolve_conv_residual_group_for_concat`): the merge's own whole
# transitively-connected group is resolved exactly the way the residual
# sections above resolve it standalone (same union-find-over-eligible-merges
# walk, same per-member operand resolution, same "any operand fails, the
# entire group declines" bar) -- except its *sink* is never handed to
# :func:`_walk_to_consumer`/:func:`_walk_to_conv_consumer` the way the
# standalone residual finder needs to (that forward walker doesn't
# recognize `Concat` as a hop at all -- see below -- so a group whose sink
# feeds a `Concat` is *always* declined by the standalone residual finder,
# today, independent of anything this finder does; nothing double-resolves
# it). Instead, the branch's own already-known path from the group's sink
# to this `Concat` operand (`"add"`'s own `pre_ops`/`edges`, exactly what a
# plain producer outcome already carries) is checked for fan-out the same
# way a plain producer branch already is
# (:func:`_branch_walk_has_fanout`), and :func:`_resolve_matmul_fanout_branches`/
# :func:`_resolve_conv_fanout_branches` -- the *exact* existing fan-out
# resolver the standalone residual finder already uses, entirely
# unmodified -- is reused to confirm the group has no *other* consumer
# anywhere: every backbone tensor's only accounted consumer is the group's
# own internal wiring plus this one already-known Concat-ward path, so an
# empty result (no un-accounted consumer found at all) is this composition's
# *success* case, the mirror image of what that function's own existing
# caller treats as "no consumer, decline". Any non-empty result -- real
# fan-out exists, whether resolvable to an ordinary chain or not -- declines
# the whole `Concat` group instead of trying to reconcile a `Concat`
# branch's own fixed-offset slice with an ordinary chain's shared,
# un-offset one; see this module's own docstring for the worked reasoning.
# Once resolved, the group's several leaf producers ride together on this
# one branch (:class:`_ConcatBranch`'s own `producers` is a tuple for
# exactly this reason, not always length one) and are ranked by the same
# combined (root-sum-square) importance :func:`_plain_structured_importance`
# already uses for an ordinary multi-producer chain, not
# :func:`_plain_branch_importance`'s single-producer norm. Likewise, a
# `Concat` chained transitively into *another* `Concat` (a "spine" of
# concatenations) is not walked through: neither backward walker recognizes
# `Concat` as a hop at all, so an operand that bottoms out at one simply
# falls through to `"fail"` the same way an unrecognized producer always
# does -- no dedicated check was needed to draw that boundary, it falls out
# for free from what the walkers already do and don't recognize. A gated
# (SwiGLU/GeGLU) pair feeding a `Concat` operand directly, with no real
# producer's raw output in between and no `Add`/`SkipLayerNormalization`
# merge involved either, *is* resolved for MatMul/Gemm branches -- see
# :func:`_find_matmul_concat_chains`'s own docstring for exactly how (a
# third, `"gated"`, outcome from :func:`_walk_matmul_producer_backward`,
# distinct from both the plain-producer and composed-residual-group shapes
# above). Conv has no such combine point in scope at all -- `_BINARY_CHANNEL_OPS`
# is a MatMul/Gemm-only concept, :func:`_walk_conv_producer_backward` never
# recognizes a `Mul` hop -- so a Conv `Concat` branch bottoming out at one
# still falls through to `"fail"` exactly as before.
#
# `Concat`'s own `axis` attribute must actually be the channel axis this
# pass's importance ranking operates on, and the two node families need
# different answers for what that means:
#
# - **Conv** branches are always rank-4 (`[N, C, H, W]`) -- every Conv this
#   whole module ever matches is 2-D, no exception anywhere in this file --
#   so the channel axis is unambiguously `axis == 1` (or the equivalent
#   negative form, `axis == -3`); no rank lookup is ever needed.
# - **MatMul/Gemm** branches have no fixed rank at all (`[batch, C]`,
#   `[batch, seq, C]`, ...), but the reduction dimension every consumer
#   match in this module already cares about is always the tensor's *last*
#   axis regardless of rank (2-D weight, matrix-multiplied against
#   whatever leading batch dimensions the input happens to carry) -- so
#   `axis == -1` is always recognized outright (ONNX's own negative-axis
#   convention already counts from the end, no rank lookup needed). A model
#   that spells the same last-axis concat with an explicit *positive*
#   `axis` (e.g. `axis=1` on a 2-D `[batch, C]` tensor, numerically
#   identical to `axis=-1` there) is only recognized when the operands'
#   rank can actually be confirmed: unlike every other topology decision in
#   this module (answerable from node attributes and initializer shapes
#   alone), this one genuinely needs a rank, and the *only* place this
#   module ever looks one up is here, from the graph's own
#   `value_info`/`input`/`output` type annotations (:func:`_tensor_rank`) --
#   never from running shape inference itself, which this module (like the
#   rest of onnxsim's `apply_*`/`quantize_*` passes) never does. Those
#   annotations are reliably present for a graph that already went through
#   onnxsim's own (or any) shape-inference pass before reaching this one --
#   the ordinary case, since structured pruning is meant to run as one step
#   in a larger pipeline -- but are just as reliably *absent* for a bare
#   hand-built graph (every model in this module's own test suite, for
#   instance, unless a test opts in with
#   `onnx.shape_inference.infer_shapes()`). So a positive `axis` is accepted
#   only when at least one operand's rank is known and every operand with a
#   known rank agrees this axis is `rank - 1`; if no operand's rank can be
#   confirmed, or two operands disagree, it's declined exactly as before --
#   never guessed at. See
#   `test_structured_pruning_matmul_concat_accepts_positive_last_axis_when_rank_known`,
#   `test_structured_pruning_matmul_concat_declines_on_positive_non_last_axis`,
#   and `test_structured_pruning_matmul_concat_declines_on_positive_axis_unknown_rank`.
#
# Once every operand resolves and the branches' fixed offsets are known, the
# ordinary forward walk (:func:`_walk_to_consumer`/
# :func:`_walk_to_conv_consumer`) continues from the `Concat` node's own
# output exactly as it would from any single producer's raw output, with
# `n_channels` set to the *sum* of every branch's own channel count -- the
# `Concat` node itself never needs its own attributes changed (its output
# shape is simply whatever its inputs' shapes are, so pruning each branch's
# own producer already gives it the right, smaller input on its own). A
# grouped (`group != 1`) Conv consumer (Conv chains only -- MatMul/Gemm has
# no grouping concept) is admitted, but only when every branch's own fixed
# offset lands exactly on one of the consumer's own `group` block
# boundaries (:func:`_concat_branches_align_to_consumer_group`) -- unlike
# the ordinary grouped-producer/grouped-consumer composition
# (:func:`_find_conv_chains`), where every producer already shares one
# combined `keep` set with the consumer's own block partition, a `Concat`
# branch is pruned *independently* with no visibility into any sibling
# branch's ranking, so a block owned by more than one branch has no general
# way to land on that block's own required uniform survivor count without
# reintroducing exactly the cross-branch agreement `Concat` support exists
# to avoid -- that case (and only that case) is still declined outright, the
# same way :func:`_find_conv_residual_chains` declines every grouped
# consumer. See :func:`_concat_branches_align_to_consumer_group`'s own
# docstring for the full safety argument and a concrete counter-example.


def _concat_axis(node: onnx.NodeProto) -> Optional[int]:
    for attr in node.attribute:
        if attr.name == "axis":
            return attr.i
    return None  # required attribute on Concat's own schema -- malformed if absent


def _value_info_by_name(
    graph: onnx.GraphProto,
) -> Dict[str, onnx.ValueInfoProto]:
    """Every ``ValueInfoProto`` the graph carries for its own tensors --
    `input`, `output`, and interior `value_info` -- keyed by tensor name.
    Feeds :func:`_tensor_rank`'s positive-axis rank lookup, used by both a
    `Concat`'s own positive `axis` (this section's own comment) and a
    `_NORM_PASS_THROUGH_OPS` node's own positive `axis`
    (:func:`_norm_axis_is_last`); nothing else in this module ever consults
    a tensor's declared type/shape.
    """
    by_name: Dict[str, onnx.ValueInfoProto] = {}
    for vi in graph.input:
        by_name[vi.name] = vi
    for vi in graph.output:
        by_name[vi.name] = vi
    for vi in graph.value_info:
        by_name[vi.name] = vi
    return by_name


def _tensor_rank(
    name: str, value_info_by_name: Dict[str, onnx.ValueInfoProto]
) -> Optional[int]:
    """The tensor's rank (number of dimensions), if the graph's own
    `value_info`/`input`/`output` annotations state it -- `None` if the
    tensor has no such annotation at all, the annotation isn't a tensor
    type, or it's a tensor type with no `shape` field (ONNX's own "rank not
    statically known" spelling, distinct from a `shape` field present but
    with an unknown/symbolic *dimension value*, which this only needs the
    dimension *count* of and so doesn't care about). Never runs shape
    inference itself -- see this section's own comment for why that's the
    deliberate boundary.
    """
    vi = value_info_by_name.get(name)
    if vi is None or not vi.type.HasField("tensor_type"):
        return None
    tensor_type = vi.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    return len(tensor_type.shape.dim)


def _concat_axis_is_last(
    node: onnx.NodeProto, value_info_by_name: Dict[str, onnx.ValueInfoProto]
) -> bool:
    """True if `node`'s own `axis` attribute is confirmed to select the
    last axis of its operands -- `axis == -1` outright (ONNX's negative-axis
    convention already counts from the end), or a positive `axis` only when
    at least one operand's rank is known (:func:`_tensor_rank`) and every
    operand with a known rank agrees `axis == rank - 1`. See this section's
    own comment for the full reasoning and why a positive axis is otherwise
    declined rather than guessed at.
    """
    axis = _concat_axis(node)
    if axis is None:
        return False
    if axis < 0:
        return axis == -1
    known_rank: Optional[int] = None
    for operand in node.input:
        rank = _tensor_rank(operand, value_info_by_name)
        if rank is None:
            continue
        if known_rank is None:
            known_rank = rank
        elif rank != known_rank:
            return False  # operands disagree -- decline rather than guess
    if known_rank is None:
        return False  # no operand's rank is known -- decline rather than guess
    return axis == known_rank - 1


@dataclass(frozen=True)
class _ConcatBranch:
    """One resolved operand of a matched ``Concat`` merge group -- see this
    section's own comment. Unlike an ``Add``/``SkipLayerNormalization``
    residual merge's operands (:class:`_Chain`'s `producers`, all pruned to
    one *shared* `keep` index set, since they're summed elementwise), every
    `_ConcatBranch` in a :class:`_ConcatChain` is pruned to its *own
    independent* `keep` set -- see :func:`_apply_concat_chains`'s own
    docstring for why that needed a new sibling to :class:`_Producer`/
    :class:`_Chain` rather than folding into them.
    """

    # One producer for a plain branch (`_Producer.pre_ops` always left empty
    # -- see `pre_ops` below); more than one when this branch instead
    # resolves through a composed residual/merge group -- see this section's
    # own comment on the `"add"` outcome -- in which case every producer
    # here shares this one branch's own combined-importance `keep` set,
    # exactly the way :class:`_Chain`'s own multi-producer `producers` does
    # for an ordinary residual chain.
    producers: Tuple[_Producer, ...]
    # Ops between the producer's own raw output (or, for a composed group,
    # between every leaf producer/inter-merge hop *and* the group's own
    # sink merge node, plus the sink's own raw output and this branch's own
    # `Concat` operand -- the whole group collapses onto this one flat list,
    # exactly as :func:`_find_matmul_residual_chains`/
    # :func:`_find_conv_residual_chains` already flatten a standalone
    # group's own internal wiring into one `_Chain.chain_ops`) and this
    # branch's own `Concat` operand: ``(node, const_name_or_None)`` pairs,
    # order-independent -- every entry is sliced by this branch's own single
    # `keep` set regardless of position, exactly :class:`_Chain`'s own
    # `chain_ops` shape (needed here rather than :class:`_Producer`'s own
    # bare-node `pre_ops` tuple, because a MatMul/Gemm branch can carry a
    # per-channel `Add`/`Mul`/`BiasGelu`/`FastGelu` constant on this hop, not
    # just a unary activation).
    pre_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    # Depthwise Conv pass-through hops crossed on this branch (Conv branches
    # only; always empty for a MatMul/Gemm branch -- see
    # :class:`_ConvPassThrough`), same flattening as `pre_ops` above for a
    # composed group's own internal depthwise hops.
    conv_pass_through: Tuple[_ConvPassThrough, ...]
    n_channels: int
    # This branch's fixed offset into the merged (pre-pruning) channel
    # range, in `Concat` operand order -- see this section's own comment for
    # why this is safe to compute once, up front, from operand order alone.
    offset: int
    # The tensor name actually feeding the `Concat` node at this operand
    # position (`== producers[0].node.output[0]` when `pre_ops` is empty and
    # this is a plain, single-producer branch) -- the Wanda activation-probe
    # point for this branch, see :func:`apply_structured_wanda_pruning`. For
    # a composed group branch this is still exactly where the group's own
    # (possibly-multi-producer) output actually feeds the `Concat` node --
    # a perfectly well-defined probe point either way.
    operand_name: str


@dataclass(frozen=True)
class _ConcatChain:
    """A matched ``Concat``-merged skip-connection group -- see this
    section's own comment. `branches` are pruned independently of one
    another (see :class:`_ConcatBranch`); the one shared downstream consumer
    is sliced once, by the concatenation of every branch's own `keep` set,
    each shifted by its own `offset`.
    """

    branches: Tuple[_ConcatBranch, ...]
    concat_node: onnx.NodeProto
    # Ops between the `Concat` node's own output and the real consumer --
    # exactly :class:`_Chain`'s own `chain_ops` shape, built by the same
    # forward walk (:func:`_walk_to_consumer`/:func:`_walk_to_conv_consumer`)
    # an ordinary single-producer chain uses.
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    consumer_node: onnx.NodeProto
    consumer_weight: str
    consumer_weight_transposed: bool
    consumer_is_conv: bool
    n_channels: int  # sum of every branch's own n_channels
    # Depthwise Conv hops crossed between the `Concat` node and the real
    # consumer (Conv chains only; see :class:`_ConvPassThrough`).
    conv_pass_through: Tuple[_ConvPassThrough, ...] = ()
    # 1 for an ordinary consumer (always, for a MatMul/Gemm chain -- MatMul/
    # Gemm has no grouping concept at all) or a Conv consumer this module
    # declines to admit as grouped; > 1 for a general grouped Conv consumer
    # admitted per :func:`_concat_branches_align_to_consumer_group` -- see
    # that function's own docstring and this section's own comment for
    # exactly which grouped consumers are safe to admit and why. Mirrors
    # :class:`_Chain`'s own `consumer_group` field/:func:`_chain_group`, but
    # unlike that field this one is *not* found by inspecting `producers`
    # first: a `Concat` branch's own producer is never itself grouped (see
    # this section's own comment), so the consumer's `group` is always the
    # one that governs here.
    consumer_group: int = 1


def _branch_walk_has_fanout(
    start: str,
    edges: Tuple[Tuple[str, onnx.NodeProto], ...],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    forward_node: onnx.NodeProto,
) -> bool:
    """True if any tensor a `Concat` branch's own backward walk crossed --
    `start` (the branch operand) through the real producer's own output --
    has more than the one in-group forward consumer the walk itself already
    accounts for. The backward walkers (:func:`_walk_conv_producer_backward`/
    :func:`_walk_matmul_producer_backward`) no longer reject a multi-consumer
    tensor mid-walk themselves -- that relaxation exists for the residual/
    fan-out case, which resolves every extra consumer explicitly afterwards
    (see :func:`_resolve_conv_fanout_branches`/
    :func:`_resolve_matmul_fanout_branches`) -- but a `Concat` branch has no
    such resolution: per this section's own comment, a branch that fans out
    to another consumer is declined outright, so this replicates that check
    directly from `edges`, `start`'s own forward consumer being `forward_node`
    (the `Concat` node itself) and each subsequent tensor's being the hop
    node recorded alongside it.
    """
    prev_consumer = forward_node
    cur = start
    for new_cur, node in edges:
        consumers = consumers_of.get(cur, [])
        if len(consumers) != 1 or consumers[0] is not prev_consumer:
            return True
        prev_consumer = node
        cur = new_cur
    consumers = consumers_of.get(cur, [])
    return len(consumers) != 1 or consumers[0] is not prev_consumer


_ResolvedMatmulResidualGroup = Tuple[
    Tuple[_Producer, ...],
    Tuple[Tuple[onnx.NodeProto, Optional[str]], ...],
    int,
    List[str],
    Dict[str, Set[int]],
]


def _resolve_matmul_residual_group_for_concat(
    root: onnx.NodeProto,
    node_by_output: Dict[str, onnx.NodeProto],
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
) -> Optional[_ResolvedMatmulResidualGroup]:
    """Resolves `root` (an ``Add``/``SkipLayerNormalization`` merge a
    ``Concat`` branch's own backward walk bottomed out at -- an `"add"`
    outcome from :func:`_walk_matmul_producer_backward`) and its whole
    transitively-connected residual/merge group, mirroring
    :func:`_find_matmul_residual_chains`'s own per-group union-find loop
    exactly (same per-member operand resolution via
    :func:`_walk_matmul_producer_backward`, same "any operand fails, the
    whole group declines" bar, same post-hoc bias/scale-constant
    re-validation once the group's real channel count is known) but scoped
    to just `root`'s own component -- reached by a plain worklist walk
    outward from `root` rather than a global union-find over every merge
    node in the graph, since `root` is already known to be the group's own
    sink (see this section's own comment on the `"add"` outcome: nothing
    else in the group can consume `root`'s own output, since that output's
    sole consumer was already independently confirmed, by the caller, to be
    the `Concat`-ward hop chain this branch is being resolved for).

    Returns ``None`` the same way :func:`_find_matmul_residual_chains`
    declines a whole group: an operand fails to resolve at all, the leaf
    producers' channel counts disagree, a bias/scale constant doesn't
    actually match that channel count, `root` turns out not to be the
    group's own unique sink after all (defensive -- see above for why this
    shouldn't happen), or the same producer is named twice. On success,
    returns ``(leaf_producers, pre_chain_ops, n_channels, backbone_tensors,
    accounted)`` -- the first three exactly mirror what
    :func:`_find_matmul_residual_chains` would fold into a resolved
    :class:`_Chain`'s own `producers`/`chain_ops`/`n_channels`; the last two
    are the group's own internal wiring (every tensor an operand walk
    crossed, and which specific node already accounts for it), handed to
    :func:`_resolve_matmul_fanout_branches` by the caller to confirm the
    group has no consumer anywhere else -- `root`'s own output is
    deliberately *not* included (unlike that finder's own explicit
    `sink_out` handling), since the caller already knows, and separately
    verifies, its own single accounted consumer.
    """
    visited: List[onnx.NodeProto] = [root]
    visited_ids = {id(root)}
    referenced: Set[int] = set()
    leaf_producers: List[_Producer] = []
    n_channels_set: Set[int] = set()
    pre_chain_ops: List[Tuple[onnx.NodeProto, Optional[str]]] = []
    backbone_tensors: List[str] = []
    accounted: Dict[str, Set[int]] = {}

    def _mark_backbone(tensor: str, node: onnx.NodeProto) -> None:
        if tensor not in accounted:
            backbone_tensors.append(tensor)
        accounted.setdefault(tensor, set()).add(id(node))

    i = 0
    while i < len(visited):
        merge_node = visited[i]
        i += 1
        match = _match_matmul_residual_merge(
            merge_node, initializer_map, consumers_of, graph_outputs
        )
        if match is None:
            return None  # defensive -- every member here was matched once already
        operands, extra_ops = match
        pre_chain_ops.extend(extra_ops)
        for operand in operands:
            _mark_backbone(operand, merge_node)
            kind, payload, ops, edges = _walk_matmul_producer_backward(
                operand,
                node_by_output,
                initializer_map,
                consumers_of,
                graph_outputs,
                _MAX_CHAIN_HOPS,
            )
            for tensor, hop_node in edges:
                _mark_backbone(tensor, hop_node)
            pre_chain_ops.extend(ops)
            if kind == "producer":
                assert payload is not None and not isinstance(payload, onnx.NodeProto)
                # `producer_infos` is never passed to the walk above, so a
                # "producer" outcome here is always the plain 2-tuple -- the
                # 3-tuple "gated" shape (see _walk_matmul_producer_backward's
                # own docstring) is unreachable from this call site.
                producer, n_channels = cast(Tuple[_Producer, int], payload)
                leaf_producers.append(producer)
                n_channels_set.add(n_channels)
            elif kind == "add":
                assert isinstance(payload, onnx.NodeProto)
                referenced.add(id(payload))
                if id(payload) not in visited_ids:
                    visited_ids.add(id(payload))
                    visited.append(payload)
            else:
                return None  # "fail" -- decline the whole group

    if len(n_channels_set) != 1:
        return None  # branches disagree on channel count -- decline
    n_channels = next(iter(n_channels_set))

    if any(
        const_name is not None and initializer_map[const_name].dims[-1] != n_channels
        for _, const_name in pre_chain_ops
    ):
        return None

    sinks = [id(n) for n in visited if id(n) not in referenced]
    if sinks != [id(root)]:
        return None  # not a single linear chain rooted at `root` -- decline

    if len({p.weight for p in leaf_producers}) != len(leaf_producers):
        return None  # degenerate -- the same producer named twice

    return (
        tuple(leaf_producers),
        tuple(pre_chain_ops),
        n_channels,
        backbone_tensors,
        accounted,
    )


_ResolvedConvResidualGroup = Tuple[
    Tuple[_Producer, ...],
    Tuple[_ConvPassThrough, ...],
    Tuple[onnx.NodeProto, ...],
    int,
    List[str],
    Dict[str, Set[int]],
]


def _resolve_conv_residual_group_for_concat(
    root: onnx.NodeProto,
    node_by_output: Dict[str, onnx.NodeProto],
    initializer_map: Dict[str, onnx.TensorProto],
    graph_outputs: Set[str],
) -> Optional[_ResolvedConvResidualGroup]:
    """The Conv analogue of :func:`_resolve_matmul_residual_group_for_concat`
    -- see its own docstring for the shared reasoning this mirrors exactly
    (only the per-member walker differs: :func:`_walk_conv_producer_backward`
    instead of :func:`_walk_matmul_producer_backward`, and there is no
    ``SkipLayerNormalization`` analogue or per-channel bias/scale hop to
    re-validate on the Conv side, only depthwise pass-through hops -- see
    :func:`_find_conv_residual_chains`'s own matching re-validation). Returns
    ``(leaf_producers, pass_through, unary_ops, n_channels, backbone_tensors,
    accounted)`` on success.
    """
    visited: List[onnx.NodeProto] = [root]
    visited_ids = {id(root)}
    referenced: Set[int] = set()
    leaf_producers: List[_Producer] = []
    n_channels_set: Set[int] = set()
    pass_through: List[_ConvPassThrough] = []
    unary_ops: List[onnx.NodeProto] = []
    backbone_tensors: List[str] = []
    accounted: Dict[str, Set[int]] = {}

    def _mark_backbone(tensor: str, node: onnx.NodeProto) -> None:
        if tensor not in accounted:
            backbone_tensors.append(tensor)
        accounted.setdefault(tensor, set()).add(id(node))

    i = 0
    while i < len(visited):
        add_node = visited[i]
        i += 1
        if not _is_eligible_add_merge(add_node, initializer_map):
            return None  # defensive -- every member here was matched once already
        for operand in add_node.input:
            _mark_backbone(operand, add_node)
            kind, payload, pt, uops, edges = _walk_conv_producer_backward(
                operand,
                node_by_output,
                initializer_map,
                graph_outputs,
                _MAX_CHAIN_HOPS,
            )
            for tensor, hop_node in edges:
                _mark_backbone(tensor, hop_node)
            pass_through.extend(pt)
            unary_ops.extend(uops)
            if kind == "producer":
                assert payload is not None and not isinstance(payload, onnx.NodeProto)
                producer, n_channels = payload
                leaf_producers.append(producer)
                n_channels_set.add(n_channels)
            elif kind == "add":
                assert isinstance(payload, onnx.NodeProto)
                referenced.add(id(payload))
                if id(payload) not in visited_ids:
                    visited_ids.add(id(payload))
                    visited.append(payload)
            else:
                return None  # "fail" -- decline the whole group

    if len(n_channels_set) != 1:
        return None  # branches disagree on channel count -- decline
    n_channels = next(iter(n_channels_set))

    if any(initializer_map[hop.weight].dims[0] != n_channels for hop in pass_through):
        return None

    sinks = [id(n) for n in visited if id(n) not in referenced]
    if sinks != [id(root)]:
        return None  # not a single linear chain rooted at `root` -- decline

    if len({p.weight for p in leaf_producers}) != len(leaf_producers):
        return None  # degenerate -- the same producer named twice

    return (
        tuple(leaf_producers),
        tuple(pass_through),
        tuple(unary_ops),
        n_channels,
        backbone_tensors,
        accounted,
    )


def _find_matmul_concat_chains(graph: onnx.GraphProto) -> List[_ConcatChain]:
    """Finds MatMul/Gemm ``Concat``-merged skip connections -- see this
    section's own comment. Every operand of a last-axis `Concat`
    (:func:`_concat_axis_is_last` -- `axis == -1` outright, or a positive
    `axis` only when the operands' rank is confirmed via `value_info` to
    actually be `rank - 1`) is resolved backward, via
    :func:`_walk_matmul_producer_backward` (reused unchanged from the
    MatMul/Gemm residual section above, `producer_infos` passed through the
    same way :func:`_find_matmul_residual_chains` does so a gated ``Mul`` hop
    can resolve too -- see below), to a real MatMul/vanilla-Gemm producer
    (`"producer"`), an eligible residual/`SkipLayerNormalization` merge's
    whole group (`"add"`, composed via
    :func:`_resolve_matmul_residual_group_for_concat` -- see this section's
    own comment for exactly what composing that requires and what it
    declines), or a gated (SwiGLU/GeGLU-style) ``Mul`` of two non-constant
    operands feeding this `Concat` operand directly (`"gated"`, resolved by
    :func:`_walk_matmul_producer_backward` itself via
    :func:`_trace_gate_producer_backward` -- unlike the `"add"` composition
    above, there is no whole transitively-connected group to walk out from
    here: just this one `Mul` node's own two operands, each already resolved
    to its own real producer by the walker, both becoming this one branch's
    own `producers` tuple, ranked together by the same root-sum-square
    importance :func:`_plain_branch_importance` already applies to a
    multi-producer branch. Safe for exactly the reason the residual section's
    own "gated" outcome already is: `_trace_gate_producer_backward` holds
    every tensor on the gate/up branches to an exact single-consumer bar, and
    :func:`_branch_walk_has_fanout` -- reused unchanged, since the walk's own
    `edges` deliberately excludes the gated ``Mul``'s own two operands, see
    :func:`_walk_matmul_producer_backward`'s own docstring -- still confirms
    the ``Mul``'s own raw output has no consumer but this `Concat`, so no
    fan-out anywhere on either shape this composition could miss. Strictly
    disjoint from the `"add"` outcome by construction, not just today's
    matching: :func:`_match_matmul_residual_merge` only ever matches a bare
    ``Add`` or a ``SkipLayerNormalization``-family node, never a ``Mul``, so
    no node can ever resolve as both). If *any* operand fails to resolve at
    all, or two operands (or two leaf/gate/up producers of the same composed
    group or gated pair, or a leaf producer of one branch against another
    branch's own) name the very same producer weight (degenerate), the whole
    `Concat` node is declined -- never partially pruned.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    node_by_output = {out: node for node in graph.node for out in node.output}
    graph_outputs = {o.name for o in graph.output}
    value_info_by_name = _value_info_by_name(graph)

    # _find_gated_chains's own producer-lookup map, built once here and
    # threaded through _walk_matmul_producer_backward below -- needed only to
    # resolve a gated Mul hop via _trace_gate_producer_backward, exactly the
    # way _find_matmul_residual_chains already builds and passes this same
    # map (see this function's own docstring above).
    producer_infos: Dict[str, Tuple[onnx.NodeProto, str, bool, Optional[str], int]] = {}
    for node in graph.node:
        info = _match_producer(node, initializer_map)
        if info is not None:
            w_name, weight_transposed, bias_name, n_channels = info
            producer_infos[node.output[0]] = (
                node,
                w_name,
                weight_transposed,
                bias_name,
                n_channels,
            )

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains: List[_ConcatChain] = []
    for node in graph.node:
        if node.op_type != "Concat" or len(node.input) < 2 or len(node.output) != 1:
            continue
        if not _concat_axis_is_last(node, value_info_by_name):
            continue
        if len(set(node.input)) != len(node.input):
            continue  # degenerate -- the same tensor concatenated with itself

        branches: List[_ConcatBranch] = []
        seen_weights: Set[str] = set()
        offset = 0
        declined = False
        for operand in node.input:
            kind, payload, pre_ops, edges = _walk_matmul_producer_backward(
                operand,
                node_by_output,
                initializer_map,
                consumers_of,
                graph_outputs,
                _MAX_CHAIN_HOPS,
                producer_infos,
            )
            if kind == "fail":
                declined = True
                break
            if _branch_walk_has_fanout(operand, edges, consumers_of, node):
                declined = True
                break
            if kind == "gated":
                producer_a, producer_b, n_channels = cast(
                    Tuple[_Producer, _Producer, int], payload
                )
                if (
                    producer_a.weight == producer_b.weight
                    or producer_a.weight in seen_weights
                    or producer_b.weight in seen_weights
                ):
                    declined = True
                    break
                seen_weights.add(producer_a.weight)
                seen_weights.add(producer_b.weight)
                branches.append(
                    _ConcatBranch(
                        (producer_a, producer_b),
                        pre_ops,
                        (),
                        n_channels,
                        offset,
                        operand,
                    )
                )
                offset += n_channels
                continue
            if kind == "add":
                assert isinstance(payload, onnx.NodeProto)
                resolved = _resolve_matmul_residual_group_for_concat(
                    payload,
                    node_by_output,
                    initializer_map,
                    consumers_of,
                    graph_outputs,
                )
                if resolved is None:
                    declined = True
                    break
                producers, group_chain_ops, n_channels, backbone, accounted = resolved
                extra = _resolve_matmul_fanout_branches(
                    backbone,
                    accounted,
                    initializer_map,
                    consumers_of,
                    graph_outputs,
                    n_channels,
                    value_info_by_name=value_info_by_name,
                )
                # `None` (resolution itself failed) or a non-empty list (real
                # fan-out found, resolvable or not) both decline here -- only
                # an exactly-empty list confirms the group has no consumer
                # anywhere else, safe to compose as this one branch's own
                # contribution (see this section's own comment on the
                # `"add"` outcome for why an empty result is *this*
                # function's success case, the mirror of what its own other
                # caller, `_find_matmul_residual_chains`, treats it as).
                if extra is None or extra:
                    declined = True
                    break
                if any(p.weight in seen_weights for p in producers):
                    declined = True
                    break
                seen_weights.update(p.weight for p in producers)
                branches.append(
                    _ConcatBranch(
                        producers,
                        group_chain_ops + pre_ops,
                        (),
                        n_channels,
                        offset,
                        operand,
                    )
                )
                offset += n_channels
                continue
            assert payload is not None and not isinstance(payload, onnx.NodeProto)
            producer, n_channels = cast(Tuple[_Producer, int], payload)
            if producer.weight in seen_weights:
                declined = True
                break
            seen_weights.add(producer.weight)
            branches.append(
                _ConcatBranch((producer,), pre_ops, (), n_channels, offset, operand)
            )
            offset += n_channels
        if declined:
            continue

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue
        total_n = offset
        consumer, fwd_chain_ops = _walk_to_consumer(
            out_name,
            initializer_map,
            consumers_of,
            graph_outputs,
            total_n,
            _MAX_CHAIN_HOPS,
            value_info_by_name=value_info_by_name,
        )
        if consumer is None:
            continue

        chains.append(
            _ConcatChain(
                branches=tuple(branches),
                concat_node=node,
                chain_ops=fwd_chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
                consumer_is_conv=False,
                n_channels=total_n,
            )
        )
    return chains


def _concat_branches_align_to_consumer_group(
    branches: Sequence[_ConcatBranch], n_channels: int, group: int
) -> bool:
    """True if a grouped (``group > 1``) Conv consumer's own `group`
    equal-sized input-channel blocks (``block = n_channels // group``, the
    same partition :func:`_slice_grouped_consumer_conv_weight` slices
    against) line up exactly with this `Concat` merge's own branch
    boundaries -- every branch's fixed `offset` a multiple of `block` -- so
    that every one of the consumer's `group` blocks is owned by exactly one
    branch, never split across two. Always ``True`` for ``group <= 1`` (an
    ordinary consumer has no block partition to line up with at all).

    This is precisely the boundary this composition is safe at, and no
    wider -- the same block-partition argument :func:`_chain_group`'s own
    docstring works through for a single grouped producer/consumer, but
    *not* simply inherited from it, because a `Concat` branch's own pruning
    is structurally different from that chain's: an ordinary chain's
    producer(s) are all pruned to one *shared* `keep` set (the entire
    reason :func:`_chain_group`'s single global `block` size works there --
    every producer's own output is literally the same index space the
    consumer's blocks partition), whereas a `Concat` branch is pruned
    *independently*, by its own top-k over its own disjoint slice, with no
    visibility into any other branch's ranking or how many channels it
    plans to keep -- the entire reason `Concat` support exists in the first
    place (see this module's own docstring and this section's own comment).
    When every branch boundary is block-aligned, each of the consumer's
    `group` blocks falls entirely within one branch's own slice, so that
    branch alone can satisfy the block's own uniform-survivor-count
    requirement: it simply treats each of its own ``own_width // block``
    contained blocks exactly the way a single grouped producer already
    treats its own `group` blocks in :func:`_apply_chains` (an independent
    per-block top-k, keeping `per_group_keep` channels from each -- the
    *same* count every other block anywhere in this merge keeps, since
    `per_group_keep` is computed once from `block` and `sparsity`, and
    `block` itself depends only on the consumer's own `group` and the
    merge's total `n_channels`, never on which branch a channel happens to
    fall in). No cross-branch agreement is ever needed, because no block
    ever has more than one branch to agree between.

    When a block instead straddles two (or more) branches -- some branch's
    own boundary falls in the interior of a block rather than at its edge --
    satisfying that block's own uniform-count requirement needs the counts
    each of those branches independently contributes *from that one shared
    block* to sum to exactly `per_group_keep`. A branch's own top-k, ranked
    with no knowledge of any sibling branch's importance or how many
    channels that sibling plans to keep, has no general way to land on a
    matching split; reconciling it would need exactly the cross-branch
    importance comparison `Concat` support was built to avoid -- silently
    reintroducing the "must all agree" coupling an `Add`/
    `SkipLayerNormalization` merge has, and a `Concat` merge deliberately
    doesn't. Concrete counter-example: ``block=4``, ``per_group_keep=2``,
    branch `a` owns local columns ``[0, 3)`` of that block (3 channels) and
    branch `b` owns column ``[3, 4)`` (1 channel) -- if `a`'s own top-3
    ranking keeps all 3 of its channels (a legitimate outcome of *its own*
    sparsity target) and `b` keeps its 1, the block ends up with 4
    survivors, not 2; if `a` keeps only 1 and `b` keeps its 1, the block
    ends up with 2 survivors but there was no shared signal that told `a`
    to cut down to 1 rather than the 2 or 3 its own ranking alone would
    justify. Either way, `a` and `b`'s independent decisions can't be made
    to reliably land on the one *any* correct grouped-Conv consumer needs
    without deciding, from *outside* either branch's own ranking, how the
    budget for that one shared block is split between them -- so this case
    is declined outright, the same conservative way as everywhere else in
    this module, whenever any block spans more than one branch (checked by
    :func:`_find_conv_concat_chains` before a chain with `group > 1` is ever
    produced).
    """
    if group <= 1:
        return True
    block = n_channels // group
    return all(b.offset % block == 0 for b in branches)


def _find_conv_concat_chains(graph: onnx.GraphProto) -> List[_ConcatChain]:
    """The Conv analogue of :func:`_find_matmul_concat_chains`: every operand
    of a channel-axis `Concat` (`axis in (1, -3)` -- the channel axis of a
    `[N, C, H, W]` tensor; see this section's own comment for why Conv needs
    no rank ambiguity check the MatMul/Gemm side does) is resolved backward
    via :func:`_walk_conv_producer_backward`, reused unchanged from the Conv
    residual section above, to either a real `group=1` Conv producer
    (`"producer"`, reached through unary activations and/or self-
    consistently-depthwise pass-through hops) or an eligible `Add` merge's
    whole group (`"add"`, composed via
    :func:`_resolve_conv_residual_group_for_concat` -- see
    :func:`_find_matmul_concat_chains`'s own section comment for exactly
    what composing that requires and what it declines, identical reasoning
    on the Conv side) -- `"fail"` is declined outright either way. The
    consumer may be an ordinary (`group=1`) Conv, or a general grouped
    (`group > 1`) Conv whose own block boundaries line up exactly with
    every branch's own fixed offset -- see
    :func:`_concat_branches_align_to_consumer_group`'s own docstring for
    the exact admission condition and the safety argument (and the
    concrete counter-example) for why a *misaligned* grouped consumer is
    still declined, the same conservative way a residual/merge group
    declines one outright regardless of alignment.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    node_by_output = {out: node for node in graph.node for out in node.output}
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains: List[_ConcatChain] = []
    for node in graph.node:
        if node.op_type != "Concat" or len(node.input) < 2 or len(node.output) != 1:
            continue
        if _concat_axis(node) not in (1, -3):
            continue
        if len(set(node.input)) != len(node.input):
            continue

        branches: List[_ConcatBranch] = []
        seen_weights: Set[str] = set()
        offset = 0
        declined = False
        for operand in node.input:
            kind, payload, pass_through, unary_ops, edges = (
                _walk_conv_producer_backward(
                    operand,
                    node_by_output,
                    initializer_map,
                    graph_outputs,
                    _MAX_CHAIN_HOPS,
                )
            )
            if kind == "fail":
                declined = True
                break
            if _branch_walk_has_fanout(operand, edges, consumers_of, node):
                declined = True
                break
            if kind == "add":
                assert isinstance(payload, onnx.NodeProto)
                resolved = _resolve_conv_residual_group_for_concat(
                    payload, node_by_output, initializer_map, graph_outputs
                )
                if resolved is None:
                    declined = True
                    break
                (
                    producers,
                    group_pass_through,
                    group_unary_ops,
                    n_channels,
                    backbone,
                    accounted,
                ) = resolved
                extra = _resolve_conv_fanout_branches(
                    backbone,
                    accounted,
                    initializer_map,
                    consumers_of,
                    graph_outputs,
                    n_channels,
                )
                # See _find_matmul_concat_chains's own matching comment --
                # only an exactly-empty result confirms no fan-out anywhere
                # else in the group.
                if extra is None or extra:
                    declined = True
                    break
                if any(p.weight in seen_weights for p in producers):
                    declined = True
                    break
                seen_weights.update(p.weight for p in producers)
                branches.append(
                    _ConcatBranch(
                        producers,
                        tuple((op, None) for op in group_unary_ops)
                        + tuple((op, None) for op in unary_ops),
                        group_pass_through + pass_through,
                        n_channels,
                        offset,
                        operand,
                    )
                )
                offset += n_channels
                continue
            assert payload is not None and not isinstance(payload, onnx.NodeProto)
            producer, n_channels = payload
            if producer.weight in seen_weights:
                declined = True
                break
            seen_weights.add(producer.weight)
            branches.append(
                _ConcatBranch(
                    (producer,),
                    tuple((op, None) for op in unary_ops),
                    pass_through,
                    n_channels,
                    offset,
                    operand,
                )
            )
            offset += n_channels
        if declined:
            continue

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue
        total_n = offset
        consumer, fwd_chain_ops, fwd_pass_through = _walk_to_conv_consumer(
            out_name,
            initializer_map,
            consumers_of,
            graph_outputs,
            total_n,
            _MAX_CHAIN_HOPS,
        )
        if consumer is None:
            continue
        # `allow_conv_transpose_consumer` was left at its default (False)
        # above, so `_` here is always False -- ConvTranspose is
        # deliberately out of scope for a Concat-branch-merge chain's own
        # consumer (see _walk_to_conv_consumer's own docstring).
        consumer_node, consumer_weight, consumer_group, _ = consumer
        if not _concat_branches_align_to_consumer_group(
            branches, total_n, consumer_group
        ):
            continue  # see _concat_branches_align_to_consumer_group's own docstring

        chains.append(
            _ConcatChain(
                branches=tuple(branches),
                concat_node=node,
                chain_ops=fwd_chain_ops,
                consumer_node=consumer_node,
                consumer_weight=consumer_weight,
                consumer_weight_transposed=False,
                consumer_is_conv=True,
                n_channels=total_n,
                conv_pass_through=fwd_pass_through,
                consumer_group=consumer_group,
            )
        )
    return chains


def _plain_branch_importance(
    w_arrays_nk: List[np.ndarray], importance_norm: str = "l2"
) -> np.ndarray:
    # A plain Concat branch is exactly one producer -- with a single array
    # this is just plain per-row norm, _plain_structured_importance's own
    # single-producer case, standalone rather than routed through a _Chain.
    # A branch composed from a residual/merge group's own multiple leaf
    # producers (see this section's own comment on the `"add"` outcome)
    # combines every producer's own per-row norm the same way
    # _plain_structured_importance already does for an ordinary
    # multi-producer chain (root-sum-square for L2, plain sum for L1 -- see
    # that function's own comment for why the two combining formulas
    # differ), since they're summed elementwise before the group's own merge
    # point ever combines them.
    if importance_norm == "l1":
        importance = np.zeros(w_arrays_nk[0].shape[0], dtype=np.float64)
        for w_nk in w_arrays_nk:
            importance += np.linalg.norm(w_nk, ord=1, axis=1)
        return importance
    squared_norm = np.zeros(w_arrays_nk[0].shape[0], dtype=np.float64)
    for w_nk in w_arrays_nk:
        squared_norm += np.square(np.linalg.norm(w_nk, axis=1))
    return np.sqrt(squared_norm)


def _apply_concat_chains(
    graph: onnx.GraphProto,
    chains: List[_ConcatChain],
    sparsity: float,
    compute_branch_importance,
    touched: _TouchedState,
) -> None:
    """The Concat-merged analogue of :func:`_apply_chains` -- deliberately a
    separate function, not a `_Chain`/`_apply_chains` extension, because the
    two need genuinely different shapes. `_apply_chains` computes *one*
    `keep` index set from *one* combined importance ranking and applies it,
    unchanged, to every producer and the consumer alike -- exactly right for
    a gated pair or a residual merge, where every branch *must* agree on the
    same surviving channels since they're summed/multiplied elementwise
    before the consumer ever sees them. A `Concat` branch never needs that
    agreement (see this section's own comment): each branch owns its own
    disjoint, fixed-offset slice of the merged channel range, so each is
    ranked and pruned to its *own independent* `keep` set by
    ``compute_branch_importance(operand_name, w_arrays_nk) ->
    np.ndarray[branch.n_channels]`` (`w_arrays_nk` one weight matrix per
    producer in the branch -- length one for a plain branch, more than one
    for a branch composed from a residual/merge group's own several leaf
    producers, see this section's own comment on the `"add"` outcome), and
    only the shared downstream consumer is sliced once, by one combined
    index set -- the concatenation of every
    branch's own `keep`, each shifted by its own fixed `offset`. Since
    branch offsets strictly increase in `Concat` operand order and each
    branch's own `keep` is itself ascending, that concatenation is
    automatically ascending overall too, the same `keep` invariant
    :func:`_apply_chains` maintains. `touched` is the same
    :class:`_TouchedState` a sibling :func:`_apply_chains` call shares, so
    the two can never doubly resize the same weight; the caller flushes
    ``value_info`` once, from `touched.stale_value_info`, after every such
    call.

    When `chain.consumer_group` (Conv chains only -- always 1 for a MatMul/
    Gemm chain) is greater than 1, each branch's own independent `keep` is
    still chosen with no cross-branch coordination -- only *how* it's chosen
    within one branch changes: rather than one plain top-k of that branch's
    own `n_channels`, it's chosen as one independent top-k *per
    `block`-sized block the branch contains* (``block = chain.n_channels //
    chain.consumer_group``, the exact same global block size and
    `per_group_keep` target used everywhere else in this merge), exactly
    mirroring :func:`_apply_chains`'s own per-block mechanism for a single
    grouped producer/consumer -- safe here specifically because
    :func:`_concat_branches_align_to_consumer_group` already confirmed every
    such block falls entirely inside one branch before this chain was ever
    produced, so a branch's own `n_channels // block` is always a whole
    number and no block's own uniform-count requirement ever needs
    contributions from more than one branch. See that function's own
    docstring for the full safety argument (and the concrete counter-example
    for why a straddling block is declined instead). The final consumer
    slice is, correspondingly,
    :func:`_slice_grouped_consumer_conv_weight` rather than
    :func:`_slice_consumer_weight` whenever `consumer_group > 1`.
    """
    initializer_map = {t.name: t for t in graph.initializer}

    for chain in chains:
        producer_weights = {p.weight for b in chain.branches for p in b.producers}
        n_producers = sum(len(b.producers) for b in chain.branches)
        if len(producer_weights) != n_producers:
            continue  # degenerate -- two producers (same or different branch) naming the same weight

        conv_hop_weights = {
            h.weight for b in chain.branches for h in b.conv_pass_through
        }
        conv_hop_weights |= {h.weight for h in chain.conv_pass_through}
        n_conv_hops = sum(len(b.conv_pass_through) for b in chain.branches) + len(
            chain.conv_pass_through
        )
        if len(conv_hop_weights) != n_conv_hops:
            continue  # degenerate -- the same depthwise weight named twice

        consts = {
            p.bias for b in chain.branches for p in b.producers if p.bias is not None
        }
        consts.update(
            const_name
            for b in chain.branches
            for _, const_name in b.pre_ops
            if const_name is not None
        )
        consts.update(
            const_name for _, const_name in chain.chain_ops if const_name is not None
        )

        if (
            (producer_weights & touched.producer)
            or chain.consumer_weight in touched.consumer
            or (consts & touched.const)
            or (conv_hop_weights & touched.conv_hop)
        ):
            continue  # a shared/tied initializer another chain already resized

        group = chain.consumer_group
        if group > 1:
            block = chain.n_channels // group
            per_group_keep = max(1, round(block * (1.0 - sparsity)))

        branch_keeps: List[np.ndarray] = []
        any_pruned = False
        for b in chain.branches:
            n = b.n_channels
            if group > 1:
                # Whole number by construction -- see
                # _concat_branches_align_to_consumer_group's own docstring
                # and this function's own docstring above.
                local_blocks = n // block
                keep_count = per_group_keep * local_blocks
            else:
                keep_count = max(1, n - round(n * sparsity))
            if keep_count >= n:
                branch_keeps.append(np.arange(n))
                continue
            any_pruned = True
            w_arrays_nk = []
            for p in b.producers:
                w = _to_f64(initializer_map[p.weight])
                w_arrays_nk.append(
                    w.reshape(w.shape[0], -1)  # [out_channels, in_channels*kH*kW]
                    if p.is_conv
                    else (w if p.weight_transposed else w.T)  # [N, K]
                )
            importance = compute_branch_importance(b.operand_name, w_arrays_nk)
            if group > 1:
                # One independent top-k per block contained in this branch --
                # see this function's own docstring.
                branch_keeps.append(
                    np.concatenate(
                        [
                            np.sort(
                                np.argsort(-importance[gi * block : (gi + 1) * block])[
                                    :per_group_keep
                                ]
                            )
                            + gi * block
                            for gi in range(local_blocks)
                        ]
                    )
                )
            else:
                branch_keeps.append(np.sort(np.argsort(-importance)[:keep_count]))

        if not any_pruned:
            continue  # every branch rounds down to a no-op -- nothing to do

        for b, keep in zip(chain.branches, branch_keeps):
            if len(keep) == b.n_channels:
                continue  # this branch's own sparsity rounded to a no-op
            for p in b.producers:
                _slice_producer_weight(
                    initializer_map[p.weight],
                    p.weight_transposed,
                    keep,
                    is_conv=p.is_conv,
                )
                if p.bias is not None:
                    _slice_last_axis(initializer_map[p.bias], keep)
            for _, const_name in b.pre_ops:
                if const_name is not None:
                    _slice_last_axis(initializer_map[const_name], keep)
            for hop in b.conv_pass_through:
                # Same reasoning as _apply_chains's own depthwise hop
                # handling: channel i is exactly upstream channel i, so the
                # hop's own weight/bias slice by this branch's own `keep`.
                _slice_producer_weight(
                    initializer_map[hop.weight], False, keep, is_conv=True
                )
                if hop.bias is not None:
                    _slice_last_axis(initializer_map[hop.bias], keep)
                _set_conv_group_attr(hop.node, len(keep))

        global_keep = np.concatenate(
            [keep + b.offset for b, keep in zip(chain.branches, branch_keeps)]
        )

        for _, const_name in chain.chain_ops:
            if const_name is not None:
                _slice_last_axis(initializer_map[const_name], global_keep)
        for hop in chain.conv_pass_through:
            _slice_producer_weight(
                initializer_map[hop.weight], False, global_keep, is_conv=True
            )
            if hop.bias is not None:
                _slice_last_axis(initializer_map[hop.bias], global_keep)
            _set_conv_group_attr(hop.node, len(global_keep))

        if chain.consumer_is_conv and group > 1:
            _slice_grouped_consumer_conv_weight(
                initializer_map[chain.consumer_weight],
                global_keep,
                group,
                chain.n_channels,
            )
        else:
            _slice_consumer_weight(
                initializer_map[chain.consumer_weight],
                chain.consumer_weight_transposed,
                global_keep,
                is_conv=chain.consumer_is_conv,
            )

        touched.producer.update(producer_weights)
        touched.consumer.add(chain.consumer_weight)
        touched.const.update(consts)
        touched.conv_hop.update(conv_hop_weights)
        touched.stale_value_info.add(chain.concat_node.output[0])
        for b in chain.branches:
            touched.stale_value_info.update(p.node.output[0] for p in b.producers)
            touched.stale_value_info.update(op.output[0] for op, _ in b.pre_ops)
            touched.stale_value_info.update(
                h.node.output[0] for h in b.conv_pass_through
            )
        touched.stale_value_info.update(op.output[0] for op, _ in chain.chain_ops)
        touched.stale_value_info.update(
            h.node.output[0] for h in chain.conv_pass_through
        )


def _slice_producer_weight(
    w_init: onnx.TensorProto,
    weight_transposed: bool,
    keep: np.ndarray,
    is_conv: bool = False,
    is_conv_transpose: bool = False,
) -> None:
    w = onnx.numpy_helper.to_array(w_init)
    if is_conv:
        if is_conv_transpose:
            # ConvTranspose: [in_channels, out_channels/group, k1, ...] --
            # output channel lives on axis 1, not axis 0 (see
            # _match_conv_transpose_producer's own docstring; only ever
            # called here for a group == 1 ConvTranspose, so this is a
            # plain, ungrouped axis-1 slice).
            w_new = w[:, keep, ...]
        else:
            # Conv: [out_channels, in_channels/group, k1, ...]: output
            # channel is always axis 0, for any spatial rank.
            w_new = w[keep, ...]
    else:
        # [N, K] storage (transB=1): output channel is axis 0. [K, N]
        # storage (the common case): output channel is axis 1.
        w_new = w[keep, :] if weight_transposed else w[:, keep]
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


def _slice_consumer_weight(
    w_init: onnx.TensorProto,
    weight_transposed: bool,
    keep: np.ndarray,
    is_conv: bool = False,
    is_conv_transpose: bool = False,
) -> None:
    w = onnx.numpy_helper.to_array(w_init)
    if is_conv:
        if is_conv_transpose:
            # ConvTranspose: [in_channels, out_channels/group, k1, ...] --
            # input channel lives on axis 0, and (unlike Conv's grouped
            # consumer axis 1) spans the *full* in_channels regardless of
            # group, so a plain, flat slice is correct for any group (see
            # _match_conv_transpose_consumer's own docstring).
            w_new = w[keep, ...]
        else:
            # Conv: [out_channels, in_channels/group, k1, ...]: input
            # channel is always axis 1, for any spatial rank.
            w_new = w[:, keep, ...]
    else:
        # [N, K] storage (transB=1): reduction dim is axis 1. [K, N] storage:
        # reduction dim is axis 0.
        w_new = w[:, keep] if weight_transposed else w[keep, :]
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


def _slice_grouped_consumer_conv_weight(
    w_init: onnx.TensorProto,
    keep: np.ndarray,
    group: int,
    n_channels: int,
) -> None:
    """Slices a *general grouped* Conv consumer's ``[out_channels,
    in_channels/group, kH, kW]`` weight by a global (whole-``in_channels``)
    `keep` index set. Unlike :func:`_slice_consumer_weight`'s flat ``w[:,
    keep, ...]`` (correct only for an ordinary ``group=1`` consumer, whose
    axis 1 truly spans every input channel), a grouped consumer's axis 1 is
    only `in_channels/group` wide and is *per-group-relative*: weight
    column ``j`` on output filter ``o`` means global input channel
    ``(o // out_per_group) * block + j`` -- `block` (`n_channels // group`)
    input channels per group, not `j` itself. So each output-filter group
    needs its own local slice of `keep` -- that group's own retained
    channels, translated from global indices back to local ones by
    subtracting the group's own block offset -- rather than one shared
    index set applied uniformly across the whole axis.

    This is well-defined only because whatever produced `keep` already
    guarantees a *uniform count* of survivors per `group`-sized block (see
    :func:`_chain_group`/:func:`_apply_chains`): both producer-grouped and
    consumer-grouped selection independently keep the same count from every
    block by construction, and the "both sides grouped" composition this
    pass supports requires a matching `group` count on both ends (see
    :func:`_find_conv_chains`), so the producer's own blocks and this
    consumer's own blocks are always the exact same partition of
    `n_channels` -- never a case where one side's block boundaries split a
    count unevenly relative to the other's.

    The other caller, :func:`_apply_concat_chains` (a ``Concat``-merged
    branch group feeding a grouped consumer), establishes the same
    uniform-per-block-count guarantee by a different route: there is no
    single producer whose blocks the consumer's must match, but
    :func:`_concat_branches_align_to_consumer_group` already confirmed every
    one of the consumer's own blocks falls entirely inside one branch, and
    that branch's own per-block top-k (mirroring :func:`_apply_chains`'s
    mechanism exactly, just scoped to the blocks it contains) keeps the same
    `per_group_keep` count from each -- see that function's own docstring
    for the full argument.
    """
    w = onnx.numpy_helper.to_array(w_init)
    out_channels = w.shape[0]
    out_per_group = out_channels // group
    block = n_channels // group
    parts = []
    for gi in range(group):
        lo, hi = gi * block, (gi + 1) * block
        local_keep = keep[(keep >= lo) & (keep < hi)] - lo
        filt_lo, filt_hi = gi * out_per_group, (gi + 1) * out_per_group
        parts.append(w[filt_lo:filt_hi, local_keep, ...])
    w_new = np.concatenate(parts, axis=0)
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_new, name=w_init.name))


def _slice_last_axis(init: onnx.TensorProto, keep: np.ndarray) -> None:
    arr = onnx.numpy_helper.to_array(init)
    new = np.take(arr, keep, axis=-1)
    init.CopyFrom(onnx.numpy_helper.from_array(new, name=init.name))


def _slice_axis1(init: onnx.TensorProto, keep: np.ndarray) -> None:
    """Slices a rank-4 BNSH-format constant (a `past_key`/`past_value`
    initializer -- see :func:`_past_kv_constants_are_sliceable`) along its
    `kv_num_heads` axis (axis 1) by `keep`. Unlike :func:`_slice_last_axis`,
    the axis being sliced here is never the last one -- BNSH's last axis is
    `head_size`/`v_head_size`, untouched by a KV-group drop.
    """
    arr = onnx.numpy_helper.to_array(init)
    new = np.take(arr, keep, axis=1)
    init.CopyFrom(onnx.numpy_helper.from_array(new, name=init.name))


def _slice_axis(init: onnx.TensorProto, keep: np.ndarray, axis: int) -> None:
    """The fully general form of :func:`_slice_axis1`/:func:`_slice_last_axis`:
    slices a constant along an arbitrary `axis` by `keep`. Used for a
    broadcastable per-head mask/bias constant (`com.microsoft::Attention`'s/
    `GroupQueryAttention`'s own `attention_bias` input, or the plain
    ``ai.onnx::Attention`` op's `attn_mask`) whose own head-axis *position*
    depends on its rank -- see :func:`_head_bias_axis`.
    """
    arr = onnx.numpy_helper.to_array(init)
    new = np.take(arr, keep, axis=axis)
    init.CopyFrom(onnx.numpy_helper.from_array(new, name=init.name))


def _head_bias_axis(dims: Sequence[int], num_heads: int) -> Optional[int]:
    """Classifies a broadcastable per-head mask/bias constant's shape
    against the schema-documented rank-4 layout ``(batch_size or 1,
    num_heads or 1, q_sequence_length, kv_sequence_length)`` -- or any rank
    up to 4 broadcastable to it (`com.microsoft::Attention`'s and
    `GroupQueryAttention`'s own `attention_bias` input doc names this shape
    verbatim; the plain ``ai.onnx::Attention`` op's `attn_mask` is added the
    same way against the identically-shaped `(batch, q_num_heads, q_seq,
    kv_seq)` attention-score tensor -- see `onnx.reference.ops.op_attention`,
    where it is added via a plain ``+`` against that tensor, i.e. ordinary
    numpy broadcasting).

    A tensor of rank *r* < 4 broadcasts against that rank-4 target the
    standard numpy/ONNX way: right-aligned, as though `4 - r` size-1 axes
    were implicitly prepended. The target's own `num_heads` axis sits at
    position 1 (of 4); once a rank-*r* tensor is right-aligned, that same
    target position lines up with the tensor's own axis ``r - 3`` -- which
    only exists at all when ``r >= 3``. This is not a corner case nobody
    triggers: a mask given at rank 3 with shape ``(num_heads, q_seq,
    kv_seq)`` (omitting the batch axis entirely, relying on broadcast) is a
    per-head tensor at axis *0*, not axis 1 -- confirmed by actual
    onnxruntime execution, perturbing one index along a rank-3 mask's own
    axis 0 and observing the change land on exactly that one head's own
    output slice and no other (see ``tests/test_pruning.py``'s own
    ``test_onnx_attention_pruning_rank3_attn_mask_head_axis_is_sliced``).

    Returns the position within `dims` of that num_heads-aligned axis when
    its size is exactly `num_heads` (a genuine per-head tensor -- safe to
    slice along that axis by the same kept-head/kept-group index set
    applied everywhere else in the matched chain); ``-1`` when no axis of
    `dims` can ever land on that target position at all (``r < 3`` -- the
    tensor is unconditionally head-count-independent), or it does land
    there but is size 1 (an ordinary broadcast -- no real per-head values
    to slice, and already correct for any head count); or ``None`` when the
    shape doesn't cleanly resolve to either of those -- rank > 4 (off this
    op's own broadcast contract entirely), or the axis lands somewhere but
    is neither 1 nor `num_heads` (which would already make the source model
    invalid against this same broadcasting rule, but this function still
    declines rather than assume what a caller might have meant) -- for the
    caller to decline the whole chain on rather than guess.
    """
    rank = len(dims)
    if rank > 4:
        return None
    axis = rank - 3
    if axis < 0:
        return (
            -1
        )  # no axis of `dims` can ever align with the target's own num_heads slot
    size = dims[axis]
    if size == num_heads:
        return axis
    if size == 1:
        return -1
    return None


def _producer_weight_nk(w: np.ndarray, p: "_Producer") -> np.ndarray:
    """``[N, K]`` view of one chain producer's own (already-float64) weight
    `w`, for structured-pruning importance ranking -- `N` being that
    producer's own output-channel count, `K` everything else flattened.
    Shared by every place that needs this view before slicing (
    :func:`_apply_chains`, :func:`_apply_chains_global`,
    :func:`analyze_pruning_sensitivity`'s own dry-run report) so all three
    stay in exact agreement.

    For an ordinary Conv producer (`p.is_conv`, ``p.is_conv_transpose ==
    False``) output channels are axis 0, for any spatial rank (see
    :func:`_match_conv_producer`'s own docstring) -- a plain
    ``w.reshape(w.shape[0], -1)``. For a ``ConvTranspose`` producer
    (`p.is_conv_transpose`) output channels are axis 1 instead (see
    :func:`_match_conv_transpose_producer`'s own docstring) -- axis 1 is
    moved to the front before the same flatten, so row `i` of the returned
    array is still that producer's own `i`-th output channel's full
    weight, regardless of which axis it started on. For a MatMul/Gemm
    producer, mirrors :func:`_weight_to_nk`'s own transpose convention.
    """
    if p.is_conv:
        if p.is_conv_transpose:
            return np.moveaxis(w, 1, 0).reshape(w.shape[1], -1)
        return w.reshape(w.shape[0], -1)
    return w if p.weight_transposed else w.T


def _plain_structured_importance(
    chain: _Chain, w_arrays_nk: List[np.ndarray], importance_norm: str = "l2"
) -> np.ndarray:
    # Combined importance across every producer in this chain: for a plain
    # chain this is just that producer's own norm; for a gated pair, both
    # branches must agree on which channels survive, so their per-channel
    # norms are combined first -- as though every producer's own channel-c
    # row were concatenated into one long row before that row's own norm
    # were taken (each producer generally owns a different reduction width,
    # so the concatenation itself is never actually materialized, only its
    # norm's own decomposition into per-producer pieces is used). For L2
    # that decomposition is root-sum-square of the per-producer L2 norms
    # (``||concat(a, b)||_2 == sqrt(||a||_2^2 + ||b||_2^2)``, Li et al.'s own
    # criterion, this module's default); for L1 the analogous identity is a
    # plain *sum* of the per-producer L1 norms instead
    # (``||concat(a, b)||_1 == ||a||_1 + ||b||_1``, since L1 is just the sum
    # of every entry's own absolute value, and concatenation doesn't change
    # that sum) -- no square/sqrt anywhere, unlike L2's own combination.
    if importance_norm == "l1":
        importance = np.zeros(chain.n_channels, dtype=np.float64)
        for w_nk in w_arrays_nk:
            importance += np.linalg.norm(w_nk, ord=1, axis=1)
        return importance
    squared_norm = np.zeros(chain.n_channels, dtype=np.float64)
    for w_nk in w_arrays_nk:
        squared_norm += np.square(np.linalg.norm(w_nk, axis=1))
    return np.sqrt(squared_norm)


def _chain_group(chain: _Chain) -> int:
    """The `group` count that governs this chain's `keep`-index selection
    (see :func:`_apply_chains`): 1 for every chain this pass already
    supported before general grouped Conv (a MatMul/Gemm chain, a gated
    pair, or an ordinary ``group=1`` Conv producer/consumer -- all leave
    every `group` field at its default), and > 1 whenever any producer or
    the (primary) consumer of a Conv chain is a general grouped Conv.

    Worked example for why the producer side takes priority when *only* a
    producer is grouped: a grouped producer (`group=g`, `g` output-channel
    blocks of `out_channels/g` filters each) feeding an ordinary `group=1`
    consumer needs every one of the producer's own `g` blocks pruned to a
    uniform count so `out_channels % g == 0` survives -- a requirement the
    consumer itself doesn't share (an ordinary consumer accepts any subset
    of surviving input channels), so the producer's own grouping is what
    the shared `keep` selection must honor. Symmetrically, an ordinary
    `group=1` producer feeding a grouped consumer (`group=g_c`) has no
    grouping constraint of its own -- any subset of its output channels is
    individually a valid producer-side cut -- so it's the *consumer's* `g_c`
    blocks that constrain which subset is safe to choose, making the
    consumer's `group` the one that governs `keep` selection there. When
    both a producer and the consumer are grouped, :func:`_find_conv_chains`
    (an ordinary chain, exactly one producer) already declined the chain
    unless `producer_group == consumer_group`, so either field gives the
    same answer.

    A Conv residual/merge chain (:func:`_find_conv_residual_chains`) can
    have more than one producer, and more than one consumer branch (see
    `extra_consumers`) -- but exactly the same "must all agree" check is
    already enforced there (mirroring `_find_conv_chains`'s own check,
    generalized from one producer/one consumer to however many of each a
    group collects) before a `_Chain` is ever produced, so every non-1
    `group` value anywhere on this chain -- any producer's, the primary
    consumer's, or any extra branch's -- is guaranteed identical by the
    time this runs; checking the first producer found is enough.
    """
    for p in chain.producers:
        if p.group > 1:
            return p.group
    return chain.consumer_group


def _chain_is_global_sparsity_eligible(chain: _Chain) -> bool:
    """Whether `chain` may take part in
    :func:`apply_structured_pruning`/:func:`apply_structured_wanda_pruning`'s
    own `global_sparsity` mode (:func:`_apply_chains_global`) -- an
    ordinary, single-producer chain with no extra fan-out consumer branch
    and no general grouped Conv on either side. See
    :func:`apply_structured_pruning`'s own `global_sparsity` docstring for
    the full reasoning; this is just the predicate it describes.
    """
    return (
        len(chain.producers) == 1
        and not chain.extra_consumers
        and _chain_group(chain) == 1
    )


def _slice_chain_channels(
    initializer_map: Dict[str, onnx.TensorProto],
    chain: _Chain,
    keep: np.ndarray,
    keep_count: int,
    stale_value_info: Set[str],
) -> None:
    """Performs the actual channel-removal slicing for one already-decided
    (ascending) `keep` index set -- every producer's weight/bias, every
    chain-op constant, every depthwise pass-through hop, the (possibly
    grouped) consumer, and every extra fan-out consumer branch -- and
    records every node this leaves with a stale output shape into
    `stale_value_info`. Factored out of :func:`_apply_chains` so
    :func:`_apply_chains_global` (:func:`apply_structured_pruning`'s own
    `global_sparsity` mode) can reuse the identical slicing mechanics: the
    two only ever differ in *how* `keep`/`keep_count` are decided (a local
    per-chain top-k vs. a globally pooled threshold), never in how a
    decided `keep` set is applied to the graph.
    """
    for p in chain.producers:
        _slice_producer_weight(
            initializer_map[p.weight],
            p.weight_transposed,
            keep,
            is_conv=p.is_conv,
            is_conv_transpose=p.is_conv_transpose,
        )
        if p.bias is not None:
            _slice_last_axis(initializer_map[p.bias], keep)
    for _, const_name in chain.chain_ops:
        if const_name is not None:
            _slice_last_axis(initializer_map[const_name], keep)

    def _slice_conv_hop(hop: _ConvPassThrough) -> None:
        # Same `keep` index set as the real producer -- a depthwise
        # Conv's own channel i is exactly upstream channel i, so its
        # weight (output-channel axis 0, like any Conv producer) and
        # bias slice identically, and `group` (== in_channels ==
        # out_channels for a depthwise Conv) drops to the new count
        # right alongside them.
        _slice_producer_weight(initializer_map[hop.weight], False, keep, is_conv=True)
        if hop.bias is not None:
            _slice_last_axis(initializer_map[hop.bias], keep)
        _set_conv_group_attr(hop.node, keep_count)

    for hop in chain.conv_pass_through:
        _slice_conv_hop(hop)
    if (
        chain.consumer_is_conv
        and not chain.consumer_is_conv_transpose
        and chain.consumer_group > 1
    ):
        _slice_grouped_consumer_conv_weight(
            initializer_map[chain.consumer_weight],
            keep,
            chain.consumer_group,
            chain.n_channels,
        )
    else:
        # A grouped ConvTranspose consumer (consumer_is_conv_transpose,
        # consumer_group > 1) also lands here, not in the branch above --
        # its own input-channel axis (0) is flat/global regardless of
        # group, so the plain slice below is already correct; see
        # _match_conv_transpose_consumer's own docstring.
        _slice_consumer_weight(
            initializer_map[chain.consumer_weight],
            chain.consumer_weight_transposed,
            keep,
            is_conv=chain.consumer_is_conv,
            is_conv_transpose=chain.consumer_is_conv_transpose,
        )
    # Extra fan-out branches (see :class:`_Chain.extra_consumers`'s own
    # comment): each is either an ordinary (`group == 1`) consumer, or,
    # for a Conv residual/merge chain, a general grouped Conv consumer
    # whose own `group` was already confirmed (in
    # _find_conv_residual_chains) to agree with `group` above --
    # _resolve_matmul_fanout_branches never resolves a grouped one (no
    # such concept for MatMul/Gemm), so `consumer_group` stays at its
    # default 1 there and this always takes the plain-slice branch for
    # a MatMul/Gemm chain. Either way, fed by the exact same `keep` just
    # computed for the group's shared producers above.
    for branch in chain.extra_consumers:
        for _, const_name in branch.chain_ops:
            if const_name is not None:
                _slice_last_axis(initializer_map[const_name], keep)
        for hop in branch.conv_pass_through:
            _slice_conv_hop(hop)
        if branch.consumer_is_conv and branch.consumer_group > 1:
            _slice_grouped_consumer_conv_weight(
                initializer_map[branch.consumer_weight],
                keep,
                branch.consumer_group,
                chain.n_channels,
            )
        else:
            _slice_consumer_weight(
                initializer_map[branch.consumer_weight],
                branch.consumer_weight_transposed,
                keep,
                is_conv=branch.consumer_is_conv,
            )

    for p in chain.producers:
        stale_value_info.add(p.node.output[0])
        stale_value_info.update(pre_op.output[0] for pre_op in p.pre_ops)
    stale_value_info.update(chain_node.output[0] for chain_node, _ in chain.chain_ops)
    stale_value_info.update(hop.node.output[0] for hop in chain.conv_pass_through)
    stale_value_info.update(
        chain_node.output[0]
        for b in chain.extra_consumers
        for chain_node, _ in b.chain_ops
    )
    stale_value_info.update(
        hop.node.output[0] for b in chain.extra_consumers for hop in b.conv_pass_through
    )


def _apply_chains(
    graph: onnx.GraphProto,
    chains: List[_Chain],
    sparsity: float,
    compute_importance,
    touched: _TouchedState,
) -> None:
    """Shared body for :func:`apply_structured_pruning` and
    :func:`apply_structured_wanda_pruning`: resolves cross-chain touched-role
    conflicts, computes each surviving chain's target channel count, calls
    ``compute_importance(chain, w_arrays_nk) -> np.ndarray[n_channels]`` for
    the ranking, and performs the actual slicing. Mutates ``graph`` in
    place. `touched` accumulates every touched role and stale ``value_info``
    name across this call *and* any sibling :func:`_apply_concat_chains`
    call sharing the same `touched` -- the caller flushes ``value_info``
    once, after every such call, from `touched.stale_value_info`.

    For a chain with :func:`_chain_group` (`group`) > 1 -- a general grouped
    Conv producer or consumer, see this module's own docstring -- `keep` is
    chosen independently *within each of `group` equal-sized blocks* of the
    channel-importance vector, keeping the same count from every block
    (`_chain_group`'s own docstring works through why one side's `group`
    always suffices to pick the block boundaries both roles need to honor).
    This reduces to today's single whole-vector top-k exactly when
    `group == 1` -- the code below keeps that as a literal separate branch,
    not a `group=1` special case of the block formula, so every
    already-supported chain's rounding (and therefore its exact `keep`
    selection) stays byte-identical to before this function learned about
    grouped Conv at all.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    # A weight legitimately plays both roles across two different chains --
    # e.g. the middle layer of a 3-layer MLP is the *consumer* of the first
    # chain (its reduction/input axis gets pruned) and the *producer* of the
    # second (its own output axis gets pruned), two independent axes of the
    # same tensor. Only collapse when the *same role* is claimed twice (a
    # tied/shared weight), tracked separately per role; bias/scale constants
    # only ever play one role, so a single shared set is enough for those.
    producer_touched = touched.producer
    consumer_touched = touched.consumer
    const_touched = touched.const
    conv_hop_touched = touched.conv_hop
    stale_value_info = touched.stale_value_info

    for chain in chains:
        producer_weights = {p.weight for p in chain.producers}
        if len(producer_weights) != len(chain.producers):
            continue  # degenerate (a gated pair naming the same weight twice)

        # Every consumer branch this chain touches -- just the one primary
        # `consumer_*` for every chain kind except a residual/merge group
        # with extra fan-out (see :class:`_Chain.extra_consumers`'s own
        # comment), where there are one or more additional independent
        # branches beyond it. Conflict-checked, touched, and sliced exactly
        # like the single consumer every other chain already has -- each
        # branch is its own axis of its own weight, fed by the exact same
        # shared `keep` this loop computes once, below.
        branches = (
            _ConsumerBranch(
                chain_ops=(),
                consumer_node=chain.consumer_node,
                consumer_weight=chain.consumer_weight,
                consumer_weight_transposed=chain.consumer_weight_transposed,
                consumer_is_conv=chain.consumer_is_conv,
            ),
        ) + chain.extra_consumers

        consumer_weights = {b.consumer_weight for b in branches}
        if len(consumer_weights) != len(branches):
            continue  # degenerate (two branches naming the same weight)

        conv_hop_weights = {h.weight for h in chain.conv_pass_through}
        conv_hop_weights.update(
            h.weight for b in chain.extra_consumers for h in b.conv_pass_through
        )
        n_conv_hops = len(chain.conv_pass_through) + sum(
            len(b.conv_pass_through) for b in chain.extra_consumers
        )
        if len(conv_hop_weights) != n_conv_hops:
            continue  # degenerate (the same depthwise weight named twice)

        consts = {p.bias for p in chain.producers if p.bias is not None}
        consts.update(
            const_name for _, const_name in chain.chain_ops if const_name is not None
        )
        consts.update(
            const_name
            for b in chain.extra_consumers
            for _, const_name in b.chain_ops
            if const_name is not None
        )
        if (
            (producer_weights & producer_touched)
            or (consumer_weights & consumer_touched)
            or (consts & const_touched)
            or (conv_hop_weights & conv_hop_touched)
        ):
            continue  # a shared/tied initializer another chain already resized

        n = chain.n_channels
        group = _chain_group(chain)
        if group > 1:
            block = n // group
            per_group_keep = max(1, round(block * (1.0 - sparsity)))
            keep_count = per_group_keep * group
        else:
            keep_count = max(1, n - round(n * sparsity))
        if keep_count >= n:
            continue  # rounds down to nothing for this layer -- no-op

        w_arrays_nk = []
        for p in chain.producers:
            w = _to_f64(initializer_map[p.weight])
            w_arrays_nk.append(_producer_weight_nk(w, p))
        importance = compute_importance(chain, w_arrays_nk)
        if group > 1:
            # One independent top-k per block -- see _chain_group and this
            # function's own docstring. Blocks are contiguous and already
            # increasing, and each block's own local top-k is sorted
            # ascending before its offset is added back, so the
            # concatenation is sorted ascending overall too, same as the
            # group=1 branch's own `keep` invariant.
            keep = np.concatenate(
                [
                    np.sort(
                        np.argsort(-importance[gi * block : (gi + 1) * block])[
                            :per_group_keep
                        ]
                    )
                    + gi * block
                    for gi in range(group)
                ]
            )
        else:
            keep = np.sort(np.argsort(-importance)[:keep_count])

        _slice_chain_channels(
            initializer_map, chain, keep, keep_count, stale_value_info
        )

        producer_touched.update(producer_weights)
        consumer_touched.update(consumer_weights)
        const_touched.update(consts)
        conv_hop_touched.update(conv_hop_weights)


def _apply_chains_global(
    graph: onnx.GraphProto,
    chains: List[_Chain],
    sparsity: float,
    compute_importance,
    touched: _TouchedState,
) -> None:
    """Global-sparsity companion to :func:`_apply_chains`, used by
    :func:`apply_structured_pruning`/:func:`apply_structured_wanda_pruning`'s
    own `global_sparsity` mode. The caller is required to have already
    filtered `chains` down to ones with exactly one producer and no extra
    fan-out consumer branch and :func:`_chain_group` ``== 1`` -- see
    :func:`apply_structured_pruning`'s own docstring for exactly why every
    other chain kind (a gated pair, a residual/merge group, a
    ``Concat``-merged branch, or any general grouped Conv on either side)
    is left completely untouched by this mode rather than approximated;
    this function itself assumes that filtering already happened and
    doesn't re-check it.

    Two passes rather than :func:`_apply_chains`'s one: this function has
    to know *every* admitted chain's own channel importance before any
    single chain's `keep` can be decided, since the whole point of global
    sparsity is one shared cutoff picked from every admitted chain's
    pooled importance -- unlike :func:`_apply_chains`'s per-chain
    `keep_count`, which only ever needs that one chain's own
    `n_channels`/`sparsity`.

    Admission -- the same tied-weight conflict/degenerate checks
    :func:`_apply_chains` already performs, in the same list order --
    still happens greedily, chain by chain, in one first pass, and an
    admitted chain's own weights are marked touched immediately, *before*
    its own final `keep_count` is known. This is the one deliberate
    behavioral difference from :func:`_apply_chains`: there, a chain whose
    `keep_count` happens to round up to its full `n_channels` (no channels
    actually dropped) is left completely unmarked, so a later, conflicting
    chain sharing the same weight is still free to claim it instead.
    Here, whether a chain's own `keep_count` will round to a no-op can't
    be known until every admitted chain's importance has been pooled and
    the one shared global cutoff is picked -- so admission can't be
    deferred that way without circularity (which chains are even in the
    pool depends on admission; the cutoff depends on the pool). An
    admitted chain that happens to end up keeping every one of its own
    channels once the global cutoff is applied still claims its weights,
    exactly as if it had genuinely been pruned.

    Each admitted chain keeps at least one channel -- the same floor
    :func:`_apply_chains` already enforces per chain (``max(1, n -
    round(n * sparsity))``) -- unlike
    :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning`'s own
    `global_sparsity` mode (:func:`_apply_global_unstructured_pruning`),
    which enforces no such floor: an unstructured *entry* can legitimately
    go to all-zero within an otherwise-unpruned layer and the tensor is
    still perfectly well-formed, but a *structural* channel count can't
    drop to zero without leaving a MatMul/Conv with a zero-sized axis --
    not a valid graph. A chain the global cutoff would otherwise reduce to
    zero channels keeps its own single highest-importance channel instead,
    pulled back out of the global drop set after the fact -- the same
    guarantee :func:`_apply_chains`'s own ``max(1, ...)`` provides per
    chain, just resolved globally rather than locally. One consequence
    worth being explicit about: the *realized* aggregate sparsity across
    every admitted chain can come out slightly below the requested
    `sparsity` whenever one or more small chains hit this floor -- the
    same kind of rounding slack :func:`_apply_chains`'s own per-chain
    ``round`` already introduces at the single-chain level, just visible
    in aggregate here instead of averaged away across independent chains.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    producer_touched = touched.producer
    consumer_touched = touched.consumer
    const_touched = touched.const
    conv_hop_touched = touched.conv_hop
    stale_value_info = touched.stale_value_info

    admitted = []
    for chain in chains:
        producer_weights = {p.weight for p in chain.producers}
        if len(producer_weights) != len(chain.producers):
            continue  # degenerate (shouldn't happen for a single-producer chain)

        consumer_weights = {chain.consumer_weight}
        conv_hop_weights = {h.weight for h in chain.conv_pass_through}
        if len(conv_hop_weights) != len(chain.conv_pass_through):
            continue  # degenerate (the same depthwise weight named twice)

        consts = {p.bias for p in chain.producers if p.bias is not None}
        consts.update(
            const_name for _, const_name in chain.chain_ops if const_name is not None
        )
        if (
            (producer_weights & producer_touched)
            or (consumer_weights & consumer_touched)
            or (consts & const_touched)
            or (conv_hop_weights & conv_hop_touched)
        ):
            continue  # a shared/tied initializer another chain already resized

        w_arrays_nk = []
        for p in chain.producers:
            w = _to_f64(initializer_map[p.weight])
            w_arrays_nk.append(_producer_weight_nk(w, p))
        importance = compute_importance(chain, w_arrays_nk)

        admitted.append(
            (
                chain,
                importance,
                producer_weights,
                consumer_weights,
                consts,
                conv_hop_weights,
            )
        )
        producer_touched.update(producer_weights)
        consumer_touched.update(consumer_weights)
        const_touched.update(consts)
        conv_hop_touched.update(conv_hop_weights)

    if not admitted:
        return

    total_n = sum(chain.n_channels for chain, *_ in admitted)
    keep_count_total = min(max(round(total_n * (1.0 - sparsity)), 0), total_n)
    drop_count_total = total_n - keep_count_total

    pooled = np.concatenate([importance for _, importance, *_ in admitted])
    drop_flat = np.zeros(total_n, dtype=bool)
    if drop_count_total > 0:
        order = np.argsort(pooled, kind="stable")
        drop_flat[order[:drop_count_total]] = True

    offset = 0
    for chain, importance, *_ in admitted:
        n = chain.n_channels
        drop_here = drop_flat[offset : offset + n]
        offset += n
        if drop_here.all():
            # Per-chain floor: never drop every channel of an admitted
            # chain -- keep its own single highest-importance channel
            # instead (see this function's own docstring).
            drop_here = drop_here.copy()
            drop_here[np.argmax(importance)] = False
        keep = np.flatnonzero(~drop_here)  # already ascending
        keep_count = int(keep.size)
        if keep_count >= n:
            continue  # every channel survived -- no-op, nothing to slice

        _slice_chain_channels(
            initializer_map, chain, keep, keep_count, stale_value_info
        )


def apply_structured_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
    global_sparsity: bool = False,
) -> onnx.ModelProto:
    """Removes whole output channels from MatMul/vanilla-Gemm layers --
    real structural pruning (smaller weight tensors, smaller matmuls on any
    runtime, not just one with sparse-kernel support), as opposed to
    :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning`'s value-only
    zeroing. See this module's own docstring for the technique, its L2-norm
    importance metric, and why it's restricted to an unambiguous single
    producer -> consumer topology rather than general dependency-graph
    pruning. :func:`apply_structured_wanda_pruning` is the calibrated
    upgrade of this same technique, exactly as :func:`apply_wanda_pruning`
    is to :func:`apply_magnitude_pruning`.

    For every MatMul/vanilla-Gemm node (the "producer") whose output feeds,
    through zero or more shape-preserving elementwise ops (an activation,
    or an Add/Mul against a constant per-channel bias/scale) with no other
    consumer anywhere along that path, into exactly one downstream
    MatMul/vanilla-Gemm's reduction dimension (the "consumer"): ranks the
    producer's output channels by L2 norm of their own weight row, drops
    the lowest-``sparsity``-fraction of them, and removes the corresponding
    rows/columns from the producer's weight (and bias, if it has a constant
    one) and every intermediate per-channel constant, and the matching
    columns/rows from the consumer's weight -- a shape change that leaves
    the two layers' composition mathematically unaffected for every
    surviving channel.

    The same cut applies to ``Conv`` producer -> consumer pairs -- any
    spatial rank (1-D/2-D/3-D alike, i.e. any ``[in_channels/group, k1, ...,
    kn]`` kernel, not just 2-D's ``[in_channels/group, kH, kW]`` -- see
    :func:`_match_conv_producer`'s own docstring): each output filter's
    whole kernel ranked by its own L2 norm, exactly Li et al.'s original
    filter-pruning criterion -- joined by unary activations and/or
    depthwise Conv hops (``group == in_channels == out_channels``: one
    filter per channel, no cross-channel mixing, so it's crossed
    transparently -- its own weight/bias sliced by the producer's channel
    indices and its ``group`` attribute shrunk to match, but it contributes
    no importance of its own and can't itself be the producer or consumer
    -- see this module's own docstring). No per-channel Add/Mul between two
    Convs (a Conv already carries its own bias, and ``BatchNormalization``
    is expected to already be fused into the preceding Conv by the time
    this pass runs).

    ``ConvTranspose`` is also matched, as a producer and/or an ordinary
    (single, non-residual/Concat) consumer -- its own weight layout is the
    *reverse* of ``Conv``'s (``[in_channels, out_channels/group, k1, ...,
    kn]``, confirmed live via ``onnx.defs.get_schema("ConvTranspose")``), so
    its own output channels are pruned off axis 1 (as a producer) and its
    own input channels off axis 0 (as a consumer, correct for any `group`)
    -- see :func:`_match_conv_transpose_producer`/
    :func:`_match_conv_transpose_consumer`'s own docstrings for the full
    reasoning, including why a *grouped* ``ConvTranspose`` producer is
    deliberately left declined (a scope decision, not an oversight), and
    why ``ConvTranspose`` stays out of scope for a residual/merge group's
    own fan-out branches and a ``Concat``-merged group's own consumer for
    now.

    A *general* grouped Conv (``group`` neither 1 nor its channel count) is
    also matched, as a producer and/or a consumer, ranking/pruning each of
    its ``group`` channel blocks independently (see this module's own
    docstring for exactly why that's safe and how the two roles differ). A
    grouped producer paired with an ordinary consumer, an ordinary producer
    paired with a grouped consumer, and both sides grouped *with the same
    ``group`` count* are all supported; both sides grouped with a
    *different* ``group`` count is declined and the chain is left
    completely untouched, same as any other topology this pass can't prove
    safe to cut.

    Also handles the gated FFN pattern most current LLMs use in place of a
    plain two-layer MLP (SwiGLU/GeGLU: ``down(act(gate(x)) * up(x))``, see
    :func:`_find_gated_chains`) -- two producers (gate and up) combined by
    an elementwise product feed one consumer; both branches are ranked by
    combined (root-sum-square) importance and pruned to the *same*
    surviving channel indices, since they're about to be multiplied. This
    gated form is MatMul/Gemm-only -- Conv chains don't take part in it.

    Also handles a bounded slice of the Conv residual/skip-connection case
    (see :func:`_find_conv_residual_chains` and this module's own
    docstring): a channel-preserving ``Add(a, b)`` with two non-constant
    operands -- every residual connection's shape -- forces whichever real
    Conv producer(s) feed `a` and `b` (found by walking backward through the
    same unary-activation/depthwise-pass-through hops the forward walk
    already allows, transitively through any further such `Add` merges
    sharing the same spine) to be pruned to one shared channel-index set,
    ranked the same combined (root-sum-square) way as a gated pair. Bounded,
    not general DepGraph: every hop that walks *toward* a group's own real
    Conv producers is still held to the same single-consumer bar as
    everywhere else in this pass. Once a group's shared channel-index set is
    established, though, it can also fan out *forward* to more than one
    independent ordinary Conv consumer (see :func:`_resolve_conv_fanout_branches`)
    -- so a real multi-block ResNet stage's shared "post-block" tensor,
    read by both the next block's own first Conv *and*, unchanged, that
    block's own `Add`, is reached rather than declined; what's still
    declined is a branch that itself forks further, reaches a graph output,
    or would need a tie-break between two conflicting keep sets on the same
    shared weight. A general grouped Conv may take part in this merge too --
    as a producer, the primary consumer, and/or an extra fan-out branch --
    as long as every one of those that is grouped shares the exact same
    `group` count (see this module's own docstring for why that's the
    provably-safe slice of it); two different non-1 `group` counts anywhere
    in the same merge group are declined, the same conservative way
    :func:`_find_conv_chains` already declines it for the ordinary,
    single-producer/single-consumer case.

    The MatMul/Gemm analogue of that same residual/skip-connection case is
    also handled (see :func:`_find_matmul_residual_chains` and this
    module's own docstring) -- the transformer-block residual stream shape
    (``x = x + SelfAttn(LN(x))``, ``x = x + MLP(LN(x))``) that was
    previously declined outright. Same union-find grouping over eligible
    merge points, same single-consumer bar on every hop *toward* a group's
    own producers, same forward fan-out to more than one ordinary consumer
    once the group's `keep` set is established (see
    :func:`_resolve_matmul_fanout_branches`), same combined (root-sum-square)
    importance ranking; the one real difference is the backward walk
    mirrors :func:`_walk_to_consumer`'s own
    *wider* MatMul/Gemm hop set (unary activations plus a per-channel
    bias/scale ``Add``/``Mul`` against a constant) rather than the Conv
    walk's narrower one, since there is no depthwise-Conv-style pass-through
    analogue for MatMul/Gemm at all. A bare ``Add`` merge point is only one
    recognized shape -- since onnxruntime's own transformer-optimizer tool
    typically fuses each residual ``Add`` (plus an optional per-channel bias
    ``Add``) together with the *following* LayerNorm/RMSNorm into one
    ``com.microsoft::SkipLayerNormalization``/
    ``SkipSimplifiedLayerNormalization`` node instead, that fused node is
    recognized as an eligible merge point too (:func:`_match_matmul_residual_merge`):
    its ``input``/``skip`` inputs play ``Add``'s own two-operand role, while
    its ``gamma`` (required) and, if present, ``beta``
    (``SkipLayerNormalization`` only) and ``bias`` are sliced by the group's
    own surviving channel indices alongside everything else, confirmed
    against onnxruntime's own kernel source and by direct execution -- see
    :func:`_find_matmul_residual_chains`'s own section comment for the exact
    fused arithmetic. A gated (SwiGLU/GeGLU) combine -- a plain ``Mul`` of
    two non-constant operands, or the native fused ``SwiGLU`` op (opset
    28+) -- feeding a residual branch with no downstream projection in
    between is now resolved the same way a gated pair outside a residual
    chain already is (see :func:`_find_gated_chains`): both the gate and up
    producers it walks back to are folded into the group's own shared
    leaf-producer set, ranked and pruned together with everything else. A
    residual branch that would need to cross a fused self-attention op
    boundary (``com.microsoft::Attention``/``GroupQueryAttention``/
    ``ai.onnx`` ``Attention``) to reach a real producer is still declined
    rather than guessed at -- see the section comment above
    :func:`_walk_matmul_producer_backward` for why it isn't actually
    reachable by any hop this walk recognizes, and why the far more common
    shapes (a gated FFN's own output projection, or an attention block's own
    output-projection MatMul, feeding the residual `Add`/`SkipLayerNormalization`)
    need no special handling at all. A non-constant (or, for ``beta``/``bias``,
    present-but-non-constant) ``gamma``/``beta``/``bias``, or a
    ``SkipLayerNormalization``-family node whose optional ``mean``/
    ``inv_std_var`` outputs are actually consumed elsewhere, is declined the
    same conservative way.

    Also handles a bounded slice of the ``Concat``-merged skip-connection
    case -- the U-Net-style encoder/decoder merge (see
    :func:`_find_matmul_concat_chains`/:func:`_find_conv_concat_chains` and
    this module's own docstring) -- for both MatMul/Gemm (last-axis
    ``Concat`` only -- ``axis == -1`` outright, or a positive `axis`
    confirmed via `value_info` to equal ``rank - 1``, see
    :func:`_concat_axis_is_last`) and Conv (channel-axis ``Concat``,
    ``axis in (1, -3)``) branches. Unlike a gated pair or a residual merge,
    a ``Concat``'s branches need no shared `keep` set at all: each branch
    owns a fixed, disjoint slice of the merged channel range and is ranked
    and pruned entirely on its own, by the same L2-norm criterion as a plain
    single-producer chain; only the shared downstream consumer's weight
    needs new slicing, at each branch's own fixed offset. Every branch is
    held to the same single-consumer safety bar as everywhere else in this
    pass, and must resolve to a real producer of the appropriate family
    (MatMul/vanilla-Gemm, or a ``group=1`` Conv reached through unary
    activations and/or depthwise pass-through hops) -- a branch that fans
    out elsewhere, bottoms out at a graph input, or would need to cross a
    residual (``Add``/``SkipLayerNormalization``) merge or another
    ``Concat`` to reach one, declines the *entire* group, never partially
    pruned. A grouped (``group != 1``) Conv consumer is admitted, but only
    when every branch's own fixed offset lands on one of the consumer's own
    `group` block boundaries, so every block is owned by exactly one branch
    -- see :func:`_concat_branches_align_to_consumer_group`'s own docstring
    for the exact condition and why a block straddling two branches is still
    declined (the same reason a residual group declines any grouped
    consumer): a `Concat` branch's independent, no-cross-branch-visibility
    ranking has no general way to land two branches on a matching split of
    one shared block's required survivor count.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched producer's output
            channels to remove (at least one channel is always kept) -- or,
            when `global_sparsity`, target fraction of every eligible
            chain's channels *combined* (see `global_sparsity` below for
            which chains are eligible)
    :param importance_norm: ``"l2"`` (default, unchanged from before this
            parameter existed) ranks by Li et al.'s own root-sum-square L2
            criterion; ``"l1"`` ranks by NNI's alternative L1 criterion
            (sum of absolute weight magnitude) instead -- see this module's
            own docstring and :func:`_plain_structured_importance`'s own
            comment for exactly how each combines across a multi-producer
            chain's several producers. Applies identically whether or not
            `global_sparsity` is set -- it only changes each channel's own
            importance score, never how the keep-count/cutoff is picked.
    :param global_sparsity: pools every *eligible* matched chain's own
            per-channel importance (see `importance_norm` above) into one
            ranking across the *whole* model and picks a single keep-count
            from `sparsity`'s fraction of that pooled total -- the
            structural analogue of :func:`apply_magnitude_pruning`'s own
            `global_sparsity` mode, so a chain whose weights are uniformly
            small relative to the rest of the model gets pruned harder than
            one whose weights are uniformly large, rather than every chain
            being cut by the same fraction regardless of its own weights'
            scale.

            "Eligible" is deliberately narrower here than this function's
            own default per-chain mode matches: only an ordinary,
            single-producer chain with no extra fan-out consumer branch
            and :func:`_chain_group` ``== 1`` (no general grouped Conv on
            either side) takes part in the global pool. A gated
            (SwiGLU/GeGLU) pair, a Conv or MatMul/Gemm residual/merge
            group, a ``Concat``-merged branch, and any general grouped
            Conv chain are all left *completely untouched* in this mode --
            the same "declined outright rather than approximated"
            treatment this pass already gives any topology it can't prove
            safe to cut -- rather than folded into the pool or pruned
            locally alongside it. Three separate reasons converge on that
            line: (1) a gated pair's or residual group's two-or-more
            producers must all agree on one *shared* `keep` set already,
            before any cross-chain pooling is even considered -- a second,
            *global* layer of index agreement on top of that first one has
            no established meaning; (2) a general grouped Conv's `keep`
            selection is already constrained to a uniform count *per
            group block* (:func:`_apply_chains`'s own per-group top-k) --
            a single global cutoff picked from a pooled ranking has no
            general way to land on a block-uniform count for every
            group of every grouped chain simultaneously; and (3) a
            ``Concat``-merged branch's importance is computed and applied
            through an entirely different code path
            (:func:`_apply_concat_chains`) with its own per-branch
            floor and no shared `keep` set at all -- extending global
            pooling to it would need a third, separately-verified
            mechanism, not a natural extension of the one built for plain
            chains. Every one of those is a genuine correctness
            conflict between "one global cutoff" and that chain kind's own
            structural constraints, not merely an inconvenience -- so
            rather than force an approximation through, this mode simply
            declines them, the same conservative way this pass already
            declines any topology it can't prove safe.

            Every eligible chain still keeps at least one channel (the
            same floor this function's own default mode already
            enforces per chain) -- unlike
            :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning`'s
            own `global_sparsity` mode, which enforces no such floor: an
            unstructured weight *entry* can legitimately go to all-zero
            and the tensor stays well-formed, but a structural channel
            count can't drop to zero without producing an ill-formed
            (zero-sized) tensor axis. This means the *realized* aggregate
            sparsity across every eligible chain can come out slightly
            below the requested `sparsity` whenever one or more small
            chains hit this floor. Default ``False`` -- every pre-existing
            caller's behavior is unchanged.
    :returns: ``model`` with every matched chain's tensors resized in
            place; anything not matching that exact topology (branching,
            a non-constant bias, a consumer whose reduction dimension
            doesn't line up, ...) is left completely untouched

    Every matched producer/consumer weight (and bias, and per-channel
    constant) may be FLOAT, FLOAT16, or BFLOAT16 -- not necessarily the
    same dtype on both sides of a chain. Importance ranking reads each
    producer's own weight upcast to float64 (:func:`_to_f64`); the actual
    channel removal is pure index slicing (``w[keep, ...]``), which
    preserves every tensor's own original dtype with no cast at all -- see
    the "FP16/BFloat16 weight support" section comment above
    :func:`_match_conv_weight_only` for why that needs no separate
    downcast step.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = (
        _find_chains(graph)
        + _find_gated_chains(graph)
        + _find_conv_chains(graph)
        + _find_conv_residual_chains(graph)
        + _find_matmul_residual_chains(graph)
    )
    concat_chains = _find_matmul_concat_chains(graph) + _find_conv_concat_chains(graph)

    touched = _TouchedState()
    if global_sparsity:
        global_chains = [c for c in chains if _chain_is_global_sparsity_eligible(c)]
        if global_chains:
            _apply_chains_global(
                graph,
                global_chains,
                sparsity,
                lambda chain, w_arrays_nk: _plain_structured_importance(
                    chain, w_arrays_nk, importance_norm
                ),
                touched,
            )
    else:
        if chains:
            _apply_chains(
                graph,
                chains,
                sparsity,
                lambda chain, w_arrays_nk: _plain_structured_importance(
                    chain, w_arrays_nk, importance_norm
                ),
                touched,
            )
        if concat_chains:
            _apply_concat_chains(
                graph,
                concat_chains,
                sparsity,
                lambda _operand_name, w_arrays_nk: _plain_branch_importance(
                    w_arrays_nk, importance_norm
                ),
                touched,
            )
    if touched.stale_value_info:
        kept = [
            vi for vi in graph.value_info if vi.name not in touched.stale_value_info
        ]
        del graph.value_info[:]
        graph.value_info.extend(kept)

    return out


def _wanda_structured_calibration_stats(
    out: onnx.ModelProto,
    chains: List[_Chain],
    concat_chains: List[_ConcatChain],
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]],
) -> Dict[str, np.ndarray]:
    """Runs `out` over `calibration_data` and returns the per-probe-point
    `act_norm` dict :func:`apply_structured_wanda_pruning`'s own
    `_wanda_structured_importance`/`_wanda_branch_importance` closures look
    up by `chain.consumer_node.input[0]` (a plain chain) or
    `branch.operand_name` (a ``Concat`` branch) -- the same computation
    :func:`apply_structured_wanda_pruning`'s own body used to perform
    inline before this function existed. Factored out, read-only (never
    mutates `out`), so :func:`analyze_pruning_sensitivity`'s own dry-run
    report can compute the *exact* same activation norms
    :func:`apply_structured_wanda_pruning` would, from one single shared
    implementation.

    `out` is expected to already be the caller's own working copy, exactly
    as :func:`_wanda_unstructured_calibration_stats` expects.
    """
    # The channel axis of the activation feeding each chain's consumer: a
    # MatMul/Gemm's reduction dimension is its input's last axis, while a
    # Conv's input channel dimension is always axis 1 of [N, C, H, W]. Two
    # chains can't disagree on a shared probe name -- a tensor has exactly
    # one producer node, so it feeds one consumer type. A Concat branch's own
    # probe point is instead wherever it feeds into the Concat node itself
    # (see this function's own docstring) -- not the shared downstream
    # consumer every other chain here probes at.
    channel_axis: Dict[str, int] = {
        chain.consumer_node.input[0]: (1 if chain.consumer_is_conv else -1)
        for chain in chains
    }
    for cchain in concat_chains:
        for b in cchain.branches:
            # Every producer of a given branch is always uniformly Conv or
            # uniformly MatMul/Gemm (a residual/merge-group-composed branch
            # is only ever discovered by the one walker family that finder
            # itself uses -- see this section's own comment) -- so any one
            # producer's own `is_conv` speaks for the whole branch.
            channel_axis[b.operand_name] = 1 if b.producers[0].is_conv else -1
    probe_names = sorted(channel_axis)
    probe_model = _add_probe_outputs(out, probe_names)

    sq_sum: Dict[str, np.ndarray] = {}
    count: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            axis = channel_axis[name]
            axis = axis if axis >= 0 else x.ndim + axis
            if axis < 0 or axis >= x.ndim:
                continue
            # Sum of squares over every axis but the channel one -- correct
            # for any activation rank, not just the 2-D case.
            reduce_axes = tuple(i for i in range(x.ndim) if i != axis)
            s = np.square(x).sum(axis=reduce_axes) if reduce_axes else np.square(x)
            cnt = int(np.prod(x.shape, dtype=np.int64)) // x.shape[axis]
            sq_sum[name] = s if name not in sq_sum else sq_sum[name] + s
            count[name] = count.get(name, 0) + cnt

    return {name: np.sqrt(s / max(count[name], 1)) for name, s in sq_sum.items()}


def apply_structured_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
    global_sparsity: bool = False,
) -> onnx.ModelProto:
    """The calibrated upgrade of :func:`apply_structured_pruning`, exactly
    as :func:`apply_wanda_pruning` is to :func:`apply_magnitude_pruning`:
    same real structural channel removal, same topology matching (a single
    producer, a gated pair, or a bounded Conv *or* MatMul/Gemm
    residual/merge group -- an ``Add`` or, for MatMul/Gemm, also a
    ``SkipLayerNormalization``-family node, see :func:`apply_structured_pruning`'s
    own docstring) -> zero or more shape-preserving elementwise ops and, for
    a Conv chain, depthwise Conv hops -> one consumer,
    MatMul/Gemm or Conv, general grouped Conv included on either side, see
    :func:`apply_structured_pruning`'s own docstring) including the same
    depthwise-Conv pass-through sliced by the producer's channel indices
    alone -- it contributes no activation norm of its own to the ranking
    either, being transparent to the chain's channel-index mapping just as
    it is to plain L2-norm importance -- but each chain's
    output channels are ranked by ``||W_row||_2 * ||X||_2`` -- L2 norm of
    that channel's own weight row (or, for Conv, whole filter), times the
    L2 norm of the *activation* actually flowing through that channel over
    calibration data (captured right where the chain feeds into its
    consumer, reduced over every axis but the channel one -- the last axis
    for a MatMul/Gemm consumer, axis 1 of ``[N, C, H, W]`` for a Conv
    consumer) -- instead of weight magnitude alone. This is the same
    protection Wanda's element-wise metric gives unstructured pruning,
    transplanted to whole channels: a channel whose weight is individually
    unremarkable but which gates a consistently high-magnitude activation
    is kept over one with a larger weight norm but a near-dead activation.
    A ``Concat``-merged group (see :func:`apply_structured_pruning`'s own
    docstring) picks this up too: each branch is ranked by that same
    ``||W_row||_2 * ||X||_2`` metric independently, with its own activation
    captured right where it feeds into the ``Concat`` node (reduced the same
    way, over every axis but the channel one), not at the shared downstream
    consumer -- consistent with each branch needing no other branch's
    agreement on anything, unlike a gated pair or residual merge.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            chain's consumer-side activation norm on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched chain's output
            channels to remove (at least one channel is always kept) -- or,
            when `global_sparsity`, target fraction of every eligible
            chain's channels *combined*
    :param importance_norm: ``"l2"`` (default) or ``"l1"`` -- selects the
            *weight*-magnitude term ``||W_row||`` only, exactly as it does
            for :func:`apply_structured_pruning`; the *activation*-norm term
            ``||X||_2`` stays L2 unconditionally either way, per Wanda's own
            ``|W_ij| * ||X_j||_2`` definition (see this module's own
            docstring) -- nothing in that paper's own metric ties the
            activation norm to whichever norm ranks weight magnitude.
            Applies identically whether or not `global_sparsity` is set.
    :param epsilon: floor applied to the accumulated per-channel activation
            norm, avoiding every channel of an all-zero activation tying at
            exactly the weight-only importance
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :param global_sparsity: the structural analogue of
            :func:`apply_magnitude_pruning`'s own `global_sparsity` mode,
            applied to this function's own ``||W_row||_2 * ||X||_2``
            metric -- see :func:`apply_structured_pruning`'s own
            `global_sparsity` docstring for the full mechanism, the
            per-chain floor, and exactly which chains are "eligible" to
            take part in the pool (an ordinary, single-producer,
            non-grouped chain with no extra fan-out branch; a gated pair,
            a residual/merge group, a ``Concat``-merged branch, and any
            general grouped Conv chain are all left completely untouched
            in this mode instead). One additional caveat specific to this
            function's own metric, honestly noted rather than hidden --
            the same one :func:`apply_wanda_pruning`'s own
            `global_sparsity` docstring gives for its unstructured Wanda
            metric: ``||X||_2`` is raw calibration-activation magnitude,
            not scale-normalized across the network, so a pooled ranking
            can end up dominated by whichever eligible chains happen to
            see the largest raw activation norms for reasons unrelated to
            how much they actually matter -- still a well-defined,
            reproducible global threshold, just not a scale-comparable one
            the way plain ``|W|``-based global sparsity
            (:func:`apply_structured_pruning`'s own `global_sparsity`
            mode) more nearly is. Default ``False`` -- every pre-existing
            caller's behavior is unchanged.
    :returns: ``model`` with every matched chain's tensors resized in
            place; anything not matching that exact topology falls back to
            :func:`apply_structured_pruning`'s plain L2-norm ranking if no
            matching activation was ever observed for that chain's consumer

    Weight dtype support is identical to :func:`apply_structured_pruning`
    (FLOAT/FLOAT16/BFLOAT16, see that function's own closing note);
    calibration activations are captured via a real ``onnxruntime`` run and
    cast to float64 on read regardless of the graph's own declared dtype,
    so they need no separate handling here either.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = (
        _find_chains(graph)
        + _find_gated_chains(graph)
        + _find_conv_chains(graph)
        + _find_conv_residual_chains(graph)
        + _find_matmul_residual_chains(graph)
    )
    concat_chains = _find_matmul_concat_chains(graph) + _find_conv_concat_chains(graph)
    if not chains and not concat_chains:
        return out

    act_norm = _wanda_structured_calibration_stats(
        out, chains, concat_chains, calibration_data, providers
    )

    def _wanda_structured_importance(
        chain: _Chain, w_arrays_nk: List[np.ndarray]
    ) -> np.ndarray:
        base = _plain_structured_importance(chain, w_arrays_nk, importance_norm)
        norm = act_norm.get(chain.consumer_node.input[0])
        if norm is None or norm.shape[0] != chain.n_channels:
            return base  # no matching activation observed -- fall back to |W|
        return base * np.maximum(norm, epsilon)

    def _wanda_branch_importance(
        operand_name: str, w_arrays_nk: List[np.ndarray]
    ) -> np.ndarray:
        base = _plain_branch_importance(w_arrays_nk, importance_norm)
        norm = act_norm.get(operand_name)
        if norm is None or norm.shape[0] != base.shape[0]:
            return base  # no matching activation observed -- fall back to |W|
        return base * np.maximum(norm, epsilon)

    touched = _TouchedState()
    if global_sparsity:
        global_chains = [c for c in chains if _chain_is_global_sparsity_eligible(c)]
        if global_chains:
            _apply_chains_global(
                graph, global_chains, sparsity, _wanda_structured_importance, touched
            )
    else:
        if chains:
            _apply_chains(
                graph, chains, sparsity, _wanda_structured_importance, touched
            )
        if concat_chains:
            _apply_concat_chains(
                graph, concat_chains, sparsity, _wanda_branch_importance, touched
            )
    if touched.stale_value_info:
        kept = [
            vi for vi in graph.value_info if vi.name not in touched.stale_value_info
        ]
        del graph.value_info[:]
        graph.value_info.extend(kept)
    return out


# --- QDQ (quantized-weight) structured pruning ------------------------------
#
# Every matcher above this point requires a Conv/MatMul/Gemm weight to be a
# *direct* constant float/fp16/bf16 initializer input to the consuming node
# (``initializer_map.get(w_name)``). A statically-quantized (QDQ) ONNX graph
# doesn't look like that: the weight is an int8/uint8 initializer feeding a
# ``DequantizeLinear`` node (scale, and optionally zero_point, themselves
# constant), and it is *that node's output* -- not an initializer name at all
# -- that reaches the MatMul/Conv/Gemm. Every matcher above already declines
# this shape today, silently and by construction (``initializer_map.get(...)``
# on a ``DequantizeLinear`` output name simply returns ``None``) -- confirmed
# empirically (see ``test_qdq_weight_left_untouched_by_unstructured_pruning``)
# rather than assumed. This section adds a *new*, separate matcher +
# ``apply_structured_pruning_qdq`` specifically for the QDQ case, rather than
# reusing/extending the existing float-only matchers or ``_Chain``/
# ``_apply_chains`` machinery above -- deliberately: every one of those is
# already extensively tested against float32/float16/bfloat16 weights only,
# and retrofitting int8 QDQ handling into their shared slicing/touched-role
# bookkeeping would risk regressing that surface for no compensating benefit
# (a QDQ weight and a plain float weight never alias the same tensor).
#
# What this repo's own QDQ tooling emits, and this section leans on directly
# (:func:`onnxsim.calibration.quantize_static`, via
# ``passes/static_quantize_matmul.h``/``static_quantize_conv.h`` -- read
# directly, not assumed): every quantized weight is INT8, *symmetric*
# (``zero_point`` entirely omitted from the ``DequantizeLinear`` call -- 2
# inputs, not 3) and *per output channel* (``Ws`` shaped ``[out_channels]``,
# ``axis`` set explicitly: 0 for Conv, 0 or 1 for MatMul/Gemm depending on
# ``transB``) -- never per-tensor for a weight. Cross-checked against the
# live ONNX opset-25 ``DequantizeLinear`` schema
# (``onnx.defs.get_schema("DequantizeLinear", domain="")``): ``x_zero_point``
# is genuinely optional (absent means zero -- exactly this repo's own
# symmetric convention), and ``x_scale``/``x_zero_point`` "must have the same
# shape, determining the quantization's granularity: a scalar for
# per-tensor/per-layer quantization, a 1-D tensor for per-axis quantization,
# or have a rank identical to the input for blocked quantization" -- this
# section's own per-tensor/per-channel matcher
# (:func:`_match_dequantize_linear_weight`) accepts exactly the first two
# (scalar and 1-D/``axis``) and declines blocked quantization outright (see
# its own docstring; a sibling matcher,
# :func:`_match_dequantize_linear_weight_blockwise`, handles the third --
# INT4/UINT4 only, see below), and, since nothing in the schema *requires*
# symmetric/per-channel, also accepts an asymmetric (nonzero ``zero_point``)
# and/or per-tensor weight when actually encountered -- broader than what
# this repo's own tooling emits, because nothing about *slicing* (as opposed
# to *unstructured* pruning, see below) cares whether zero_point is zero or
# which axis granularity was chosen, as long as it's consistently one of the
# two shapes above.
#
# Structured (channel) pruning of a QDQ weight turns out to split cleanly
# into two very differently-shaped problems depending on which axis is being
# cut, mirroring this module's own established "slice, don't recompute"
# principle for FP16/BFloat16 weights (see that section's own comment far
# above) -- extended here from a *dtype* metadata pair (never touched by any
# purely structural rewrite) to a *quantization* metadata pair (touched only
# when, and exactly how, the channel axis it's indexed by is touched):
#
#   * Pruning a QDQ weight's OWN output channels (the "producer" role, e.g.
#     the first layer of a two-layer MLP): a per-channel weight's ``Wq``
#     (int8) AND its own ``Ws``/``Wzp`` (each ``[out_channels]``) are all
#     indexed by that exact axis, so all three must be sliced together, by
#     the same ``keep`` index set, in lockstep -- exactly the "slice, don't
#     recompute" pattern the FP16/BFloat16 section already established, just
#     for three co-indexed tensors instead of one. A *per-tensor* weight
#     (scalar ``Ws``/``Wzp``) is simpler still, as the task anticipated:
#     the scale/zero-point don't change at all (a single scalar isn't shaped
#     by channel count), only ``Wq`` itself is sliced.
#   * Pruning a QDQ weight's INPUT channels (the "consumer" role, e.g. that
#     same MLP's second layer): ``Wq`` is sliced along its *reduction* axis
#     -- the axis ``Ws``/``Wzp`` are never indexed by, per-channel or
#     per-tensor alike -- so ``Ws``/``Wzp`` are simply left completely
#     untouched. This is genuinely simpler than the producer side, not just
#     differently shaped: it needs no co-slicing of anything beyond ``Wq``
#     itself, identical in spirit to how an ordinary *float* consumer weight
#     is already sliced along its own input axis with no other tensor
#     involved.
#
# Both is-QDQ combinations are supported for either role (a QDQ producer
# feeding a plain float consumer, or vice versa, or QDQ on both sides) --
# :func:`_resolve_weight_ref` treats a direct float initializer, a
# per-tensor/per-channel QDQ ``DequantizeLinear``-fed one, and (see below) a
# *blockwise* INT4/UINT4 QDQ-fed one as three resolutions of the same
# "weight reference" concept, so a producer/consumer pair is only skipped by
# :func:`apply_structured_pruning_qdq` when *neither* side is QDQ (that
# plain float/float pair is exactly :func:`apply_structured_pruning`'s own
# job, left untouched here rather than pruned twice by two different
# passes) -- any mix of the three (float, per-tensor/per-channel QDQ,
# blockwise QDQ) on either side composes correctly, exactly the way a
# float/QDQ mix already did before blockwise support existed.
#
# Blockwise INT4/UINT4 quantization (opset 21 / IR version 10's own
# ``block_size`` attribute on ``QuantizeLinear``/``DequantizeLinear``, and
# the ``INT4``/``UINT4`` tensor element types added at the same opset) is
# the standard, non-``com.microsoft``-contrib-op path several ONNX export
# toolchains (Olive, various Hugging Face Optimum exporters) use for
# weight-only int4 quantization -- distinct from, and more portable than,
# the already-supported ``MatMulNBits`` contrib op below. It is matched by
# a separate function, :func:`_match_dequantize_linear_weight_blockwise`,
# sitting alongside :func:`_match_dequantize_linear_weight` rather than
# folded into it (the shapes/branches involved are different enough --
# nibble-typed codes, a full-rank scale/zero-point instead of scalar/1-D --
# that keeping them apart is clearer than one function with two unrelated
# code paths), but *is* folded into :class:`_WeightRef`/
# :func:`_resolve_weight_ref` (a third, mutually-exclusive field alongside
# `float_init`/`qdq`), unlike ``MatMulNBits`` below -- because a blockwise
# QDQ weight and a per-tensor/per-channel QDQ weight are still the same
# *op-level* shape (a constant int/uint initializer fed through a
# ``DequantizeLinear`` into an ordinary Conv/MatMul/Gemm), just a different
# quantization granularity, whereas ``MatMulNBits`` is a structurally
# different node (the quantized weight's operands are inputs to a single
# fused compute op, not a separate initializer-then-dequantize pair).
#
# Every fact this support depends on was confirmed empirically against this
# environment's live ``onnx`` (1.22.0) and ``onnxruntime`` (1.29.0)
# installs, cross-checked between ``onnx.reference.ReferenceEvaluator`` and
# a real ``onnxruntime.InferenceSession`` (both agree, to ~7e-7 absolute
# error against an independently-written numpy oracle, for every blockwise
# case below -- see this section's own tests, e.g.
# ``test_qdq_blockwise_int4_producer_matches_oracle``):
#
#   * ``block_size`` blocks along the axis named by the live
#     ``DequantizeLinear`` schema's own ``axis`` attribute (default 1,
#     confirmed via ``onnx.defs.get_schema("DequantizeLinear", domain="")``)
#     -- for a rank-``r`` input and block size ``B``, ``x_scale``/(if
#     present) ``x_zero_point`` have the SAME rank as the weight, full-size
#     (matching the weight's own dim) on every axis except `axis`, where
#     their size is ``ceil(dim_size(axis) / B)``: e.g. a ``(6, 8)`` weight,
#     ``axis=1``, ``block_size=4`` gives a ``(6, 2)`` scale -- confirmed via
#     both evaluators against a hand-computed
#     ``(code - zero_point) * scale`` reference, block-by-block.
#   * ``INT4``/``UINT4`` (``onnx.TensorProto.INT4``/``UINT4``, values 22/21)
#     is a genuinely different packing convention from ``MatMulNBits``'s own
#     nibble-per-``blob`` layout (see that section's own top comment) --
#     confirmed by reading ``onnx.numpy_helper``'s own (private, but stable
#     across the installed 1.22.0 wheel) ``_pack_4bitx2``/``_unpack_4bit``
#     helpers: they flatten the ENTIRE tensor in row-major order first
#     (``array.ravel()``), THEN pack pairs of *consecutive flattened*
#     values 2-per-byte (low nibble = lower flat index, matching
#     ``MatMulNBits``'s own "low nibble first" convention for nibble order,
#     but over a totally different pairing -- adjacent flattened elements,
#     not elements a fixed ``blob_size`` apart within one block/row).
#     Critically, this means this section never needs to hand-roll
#     packing/unpacking at all (unlike the ``MatMulNBits`` section's own
#     ``_pack_nbits_nibbles``/``_unpack_nbits_nibbles``): ``onnx.numpy_helper
#     .to_array()``/``.from_array()`` round-trip an INT4/UINT4
#     ``TensorProto`` through a plain ``ml_dtypes.int4``/``uint4`` numpy
#     array transparently (ordinary numpy indexing/slicing on that array
#     works exactly like any other dtype, confirmed by a direct round-trip
#     test), so "slice, don't recompute" here is *literally* ordinary numpy
#     slicing on the unpacked array, followed by ``from_array`` re-packing
#     the smaller result from scratch -- the packing convention above only
#     matters for understanding what's happening, never for hand-writing it.
#   * ``x_zero_point`` is genuinely optional for INT4/UINT4 blockwise, per
#     the live schema's own doc string ("zero-point is usually not used in
#     the case of ... 4-bit types quantization, but the dequantization
#     formula remains the same for consistency") -- confirmed by running
#     both with and without it and checking the "absent" case matches a
#     hand-computed zero-point-of-0 reference. When present, it is packed
#     via the exact same INT4/UINT4 convention as the weight itself (not a
#     separate encoding the way ``MatMulNBits``'s own packed-vs-unpacked
#     ``zero_points`` dispatch requires) -- also confirmed via the same
#     round-trip.
#
# Structured (channel) pruning of a blockwise weight splits along the same
# producer/consumer axis distinction the per-tensor/per-channel case above
# already established, but with the roles of "simple" and "needs care"
# reversed, because blockwise quantization blocks the *reduction* (input-
# channel) axis, never the output-channel axis (real weight-only blockwise
# quantization -- and this section's own matcher, which requires it,
# declining anything else -- always blocks along the axis a Linear layer
# contracts over, exactly mirroring ``MatMulNBits``'s own ``K``-axis
# blocking):
#
#   * Pruning a blockwise weight's OWN output channels (the producer role):
#     since `block_size` never applies to this axis, `scale`/`zero_point`
#     are already full-size (unreduced) there -- exactly like a per-channel
#     (unblocked) QDQ weight's own scale/zero-point at ITS per-channel
#     axis -- so the int4/uint4 codes and scale/zero-point are co-sliced by
#     the same `keep` in lockstep, with no block-alignment concern
#     whatsoever (:func:`_slice_producer_weight_qdq_block`).
#   * Pruning a blockwise weight's INPUT (reduction) channels (the consumer
#     role): this IS the blocked axis, so an individual input channel can't
#     be dropped without re-quantizing its whole block -- out of scope, this
#     module never invents new quantized values. Mirroring
#     :func:`_matmul_nbits_block_aligned_keep_blocks` exactly,
#     :func:`_qdq_block_aligned_keep_blocks` checks whether the candidate
#     `keep` set (computed the same way as everywhere else in this module --
#     top-``keep_count`` producer output channels by dequantized L1/L2 row
#     norm) happens to already align to the consumer's own `block_size`
#     boundaries (every block wholly kept or wholly dropped); when aligned,
#     the whole aligned blocks are dropped from the codes AND
#     scale/zero-point together (:func:`_slice_consumer_weight_qdq_block`);
#     when NOT aligned, this chain -- both producer and consumer sides -- is
#     left completely untouched (declined), exactly
#     :func:`apply_structured_pruning_matmul_nbits`'s own precedent for the
#     identical dilemma, rather than forcing a partial-block re-quantization
#     or a disagreeing keep-set between the paired producer/consumer.
#
# What's declined, deliberately, rather than guessed at:
#
#   * *Unstructured* (magnitude-style) pruning of a QDQ weight. Zeroing an
#     individual int8 code only zeroes the *dequantized* value when
#     ``zero_point`` happens to be exactly that code (``(q - zp) * scale ==
#     0`` iff ``q == zp``) -- for this repo's own symmetric weights that's
#     ``q == 0``, so it would coincidentally work there, but the general QDQ
#     schema (see above) allows a nonzero, per-channel ``zero_point``, for
#     which "which int8 code represents zero" varies channel by channel and
#     picking the *smallest-magnitude dequantized values* to zero (this
#     module's existing magnitude/Wanda/N:M criteria) would require
#     recomputing and rewriting every kept ``Wq`` entry against the new
#     mask's own statistics, not just dropping some -- a fundamentally
#     different, much more invasive operation than this module's existing
#     unstructured pruning (which only ever *sets entries to 0*, never
#     changes any other entry) and outside this module's own "never
#     recompute a kept value" principle used everywhere else. Rather than
#     force a dubious answer, this is left declined -- and, as noted above,
#     already is: every unstructured matcher's ``initializer_map.get(w_name)``
#     already returns ``None`` for a ``DequantizeLinear``-fed weight name, so
#     no code change was needed to establish this; it is simply documented
#     and empirically confirmed here, per the module's own convention of
#     explicitly discussing every topology boundary.
#   * A *general grouped* or depthwise Conv (``group != 1``) on either side
#     of a QDQ chain, and a gated (SwiGLU/GeGLU) pair, a residual/skip-
#     connection merge, or a Concat-merged branch group anywhere in a QDQ
#     chain. Every one of these is already a materially bigger project for
#     the *plain float* case above (see this module's own top-of-file
#     docstring) even before QDQ enters the picture; composing either with a
#     QDQ weight's extra scale/zero-point bookkeeping is well beyond this
#     section's scope. Only the plain single producer -> [zero or more
#     shape-preserving unary activations, see `_UNARY_PASS_THROUGH`] ->
#     single consumer topology (an ordinary, ``group=1`` Conv or MatMul/
#     vanilla-Gemm on both ends) is matched -- no per-channel Add/Mul
#     bias/scale hop either, unlike the plain float MatMul/Gemm chain walk
#     above (:func:`_walk_to_consumer`), unnecessary complexity for this
#     section's own deliberately narrow first cut.
#   * Blocked quantization (opset 21+'s ``block_size`` attribute) with
#     INT8/UINT8 codes, or blocking along the output-channel axis rather
#     than the reduction axis, is declined by
#     :func:`_match_dequantize_linear_weight`/
#     :func:`_match_dequantize_linear_weight_blockwise` alike (neither
#     matches it) -- only INT4/UINT4 blockwise quantization on the
#     reduction axis is supported, per the empirical investigation above.
#     FLOAT8 quantized codes, a non-default ``output_dtype`` override, a
#     blockwise weight whose reduction-axis dim is not an exact multiple of
#     `block_size` (a padded/partial final block -- mirroring
#     ``MatMulNBits``'s own identical scope decision above, for the same
#     reason: how a partial block's padding region is treated is not
#     verified here), a non-``FLOAT`` blockwise scale, a consumer-side
#     `keep` set that doesn't align to a blockwise weight's own block
#     boundaries (see above -- declined per-chain, not a matcher-level
#     decline), and any DequantizeLinear whose ``x``/``x_scale``/(if
#     present) ``x_zero_point`` initializer is read by more than one node (a
#     shared/tied quantized weight -- slicing it here would silently corrupt
#     whatever else reads it) or whose own output feeds more than one
#     consumer. All declined by :func:`_match_dequantize_linear_weight`/
#     :func:`_match_dequantize_linear_weight_blockwise` themselves (or, for
#     the non-block-aligned `keep` case, by
#     :func:`apply_structured_pruning_qdq`'s own per-chain check), the same
#     conservative "decline anything ambiguous" bar every matcher elsewhere
#     in this module is held to.
#   * INT4/UINT4 quantization of *activations* (as opposed to weights) is
#     entirely out of scope -- this section, like the per-tensor/per-channel
#     one above it, only ever matches a ``DequantizeLinear`` feeding a
#     Conv/MatMul/Gemm's own WEIGHT input (`_match_conv_qdq`/
#     `_match_matmul_qdq` only ever resolve `node.input[1]`), never its
#     activation input.
#
# Finding for the reverse composition -- pruning a still-*float* model with
# :func:`apply_structured_pruning`/:func:`apply_structured_wanda_pruning` and
# only *afterwards* running this repo's own
# :func:`onnxsim.calibration.quantize_static` on the pruned result -- rather
# than something this section needs to build: confirmed empirically (see
# ``test_prune_then_quantize_static_composes_correctly``) to already work
# correctly with no code changes anywhere. Pruning operates entirely on
# float initializers and produces an ordinary (smaller) float graph;
# ``quantize_static`` has no notion of "this graph was pruned" to get wrong.
# The one practical prerequisite -- and it is exactly that, a prerequisite,
# not a pruning-caused defect -- is that ``quantize_static`` (like
# ``list_quantizable_activations``, which it calls internally) reads each
# candidate activation's *declared* element type off the graph's own
# ``value_info`` (``Value::elemType()`` in the C++ IR), which a hand-built
# or ``onnx.parser``-parsed graph never populates for its own intermediate
# tensors, and which :func:`apply_structured_pruning` does not itself run or
# preserve; ``onnx.shape_inference.infer_shapes()`` on the pruned model
# before quantizing (a standard step before handing a graph to *any*
# onnxsim/onnxruntime tooling that inspects intermediate shapes/dtypes, not
# a special QDQ- or pruning-specific one) resolves it, and the two compose
# correctly from there -- verified end to end through onnxruntime, matching
# a hand-computed reconstruction-error expectation (small, from
# quantization, not from pruning).


@dataclass(frozen=True)
class _QDQWeight:
    """A Conv/MatMul/Gemm weight fed through a ``DequantizeLinear`` node from
    a constant int8/uint8 initializer, matched by
    :func:`_match_dequantize_linear_weight` -- see this section's own
    top-of-section comment for the exact pattern accepted and declined.
    `axis` is this weight's own output-channel axis (0 for Conv; 0 or 1 for
    MatMul/Gemm depending on ``transB``) -- meaningful only when
    `per_channel`; a per-tensor (`per_channel=False`) weight's `scale_init`/
    `zero_point_init` are scalars, unaffected by any axis.
    """

    dq_node: onnx.NodeProto
    q_init: onnx.TensorProto
    scale_init: onnx.TensorProto
    zero_point_init: Optional[onnx.TensorProto]
    axis: int
    per_channel: bool


def _match_dequantize_linear_weight(
    weight_name: str,
    rank: int,
    expected_axis: int,
    initializer_map: Dict[str, onnx.TensorProto],
    dq_of: Dict[str, onnx.NodeProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
) -> Optional[_QDQWeight]:
    """If `weight_name` is fed by a ``DequantizeLinear`` node from a constant
    int8/uint8 initializer, with a constant float32 scale that is either a
    scalar (per-tensor) or a 1-D vector of length ``dims[expected_axis]``
    (per-channel, on exactly `expected_axis` -- the weight's own output-
    channel axis, regardless of which role, producer or consumer, it plays
    in the chain this weight sits in) with a matching `axis` attribute,
    returns the match. `rank` is the expected weight rank (4 for Conv, 2 for
    MatMul/Gemm).

    Declines (``None``) whenever anything is ambiguous rather than guessing,
    per this section's own top-of-section comment: a non-constant ``x``/
    ``x_scale``/``x_zero_point``, a dtype other than INT8/UINT8 for the
    weight (or a ``zero_point`` not matching it), a rank mismatch, a scale
    shaped for blocked quantization (rank equal to `rank`) or any shape
    other than scalar/1-D, a per-channel scale on any axis other than
    `expected_axis`, a non-default ``block_size``/``output_dtype``
    attribute, a ``DequantizeLinear`` output read by more than one
    consumer, or any of ``x``/``x_scale``/``x_zero_point`` read by more than
    one node (a shared/tied quantized tensor this weight's own slicing would
    otherwise silently corrupt for that other reader).
    """
    dq = dq_of.get(weight_name)
    if dq is None or dq.op_type != "DequantizeLinear" or len(dq.output) != 1:
        return None
    if len(consumers_of.get(weight_name, [])) != 1:
        return None  # DQ output must feed only this one weight use
    if len(dq.input) not in (2, 3):
        return None
    q_name, scale_name = dq.input[0], dq.input[1]
    zp_name = dq.input[2] if len(dq.input) == 3 and dq.input[2] else None
    if not q_name or not scale_name:
        return None

    q_init = initializer_map.get(q_name)
    scale_init = initializer_map.get(scale_name)
    if q_init is None or scale_init is None:
        return None  # non-constant q/scale -- can't safely slice it
    if q_init.data_type not in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8):
        return None  # the two 8-bit codes this repo's own QDQ tooling (and
        # the ecosystem's standard QDQ pattern) emits; INT4/UINT4 blockwise
        # is `_match_dequantize_linear_weight_blockwise`'s own job (below),
        # and FLOAT8 needs different range handling neither matcher
        # attempts -- see this section's own top comment.
    if scale_init.data_type != onnx.TensorProto.FLOAT:
        return None
    if len(q_init.dims) != rank:
        return None

    zp_init = None
    if zp_name is not None:
        zp_init = initializer_map.get(zp_name)
        if zp_init is None or zp_init.data_type != q_init.data_type:
            return None  # schema: x_zero_point and x must have the same type
        if list(zp_init.dims) != list(scale_init.dims):
            return None  # schema: x_scale and x_zero_point must have the
            # same shape

    for nm in (q_name, scale_name) + ((zp_name,) if zp_name else ()):
        if len(consumers_of.get(nm, [])) != 1:
            return None  # shared/tied quantized tensor -- another node
            # reads it too, so it can't be sliced only for this one

    for attr in dq.attribute:
        if attr.name == "block_size" and attr.i != 0:
            return None  # blocked quantization -- a different scale shape/
            # granularity than either case this matcher handles
        if attr.name == "output_dtype" and attr.i != 0:
            return None  # non-default output dtype -- out of scope

    axis = 1  # DequantizeLinear's own schema default
    for attr in dq.attribute:
        if attr.name == "axis":
            axis = attr.i
            break
    if axis < 0:
        axis += rank

    scale_dims = list(scale_init.dims)
    numel = int(np.prod(scale_dims)) if scale_dims else 1
    if numel == 1:
        per_channel = False  # per-tensor: axis is immaterial (a scalar
        # broadcasts to every channel identically regardless)
    elif len(scale_dims) == 1:
        if axis != expected_axis:
            return None  # per-channel scale on an axis other than this
            # weight's own output-channel axis -- not the shape this pass
            # prunes, decline rather than guess
        if scale_dims[0] != q_init.dims[expected_axis]:
            return None  # scale length must match the channel count
        per_channel = True
    else:
        return None  # anything else (blocked quantization's rank == `rank`
        # scale, or any other shape) -- out of scope, see this section's own
        # top-of-section comment

    return _QDQWeight(
        dq_node=dq,
        q_init=q_init,
        scale_init=scale_init,
        zero_point_init=zp_init,
        axis=expected_axis,
        per_channel=per_channel,
    )


@dataclass(frozen=True)
class _QDQBlockwiseWeight:
    """A Conv/MatMul/Gemm weight fed through a *blockwise*-quantized
    (opset 21+ ``block_size`` attribute) ``DequantizeLinear`` node from a
    constant INT4/UINT4 initializer, matched by
    :func:`_match_dequantize_linear_weight_blockwise` -- see this section's
    own top-of-section comment for the empirical packing/blocking facts
    this depends on. `out_axis` is this weight's own output-channel axis
    (same meaning as :class:`_QDQWeight`'s own `axis`); `block_axis` is the
    axis `block_size` actually blocks along -- always
    ``1 - out_axis`` (the reduction/input-channel axis), enforced by the
    matcher, never `out_axis` itself. `scale_init`/(if present)
    `zero_point_init` have the SAME RANK as `q_init`, full-size on every
    axis except `block_axis`, where their size is `num_blocks` (==
    ``ceil(q_init.dims[block_axis] / block_size)``, and, since the matcher
    requires an exact multiple, also exactly
    ``q_init.dims[block_axis] // block_size``).
    """

    dq_node: onnx.NodeProto
    q_init: onnx.TensorProto
    scale_init: onnx.TensorProto
    zero_point_init: Optional[onnx.TensorProto]
    out_axis: int
    block_axis: int
    block_size: int
    num_blocks: int


def _match_dequantize_linear_weight_blockwise(
    weight_name: str,
    rank: int,
    expected_axis: int,
    initializer_map: Dict[str, onnx.TensorProto],
    dq_of: Dict[str, onnx.NodeProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
) -> Optional[_QDQBlockwiseWeight]:
    """The blockwise INT4/UINT4 analogue of
    :func:`_match_dequantize_linear_weight` -- see this section's own
    top-of-section comment for the empirical schema/packing facts this
    depends on. If `weight_name` is fed by a ``DequantizeLinear`` node from
    a constant INT4/UINT4 initializer with a nonzero ``block_size``
    attribute, a constant FLOAT `x_scale` of the same rank as the weight
    (full-size everywhere except a nonzero `axis` attribute equal to
    ``1 - expected_axis`` -- the weight's own reduction/input-channel axis,
    NEVER `expected_axis`, its output-channel axis, regardless of which
    role, producer or consumer, this weight plays in the chain it sits in),
    and (if present) a matching-shape `x_zero_point` of the same INT4/UINT4
    type as the weight, returns the match. `rank` is the expected weight
    rank (4 for Conv, 2 for MatMul/Gemm).

    Declines (``None``) whenever anything is ambiguous or outside what was
    actually verified, rather than guessing, per this section's own
    top-of-section comment: a non-constant ``x``/``x_scale``/
    ``x_zero_point``, a dtype other than INT4/UINT4 for the weight (or a
    ``zero_point`` not matching it), a non-``FLOAT`` scale, a rank
    mismatch, a zero or absent ``block_size`` (the per-tensor/per-channel
    case :func:`_match_dequantize_linear_weight` handles, not this one), a
    non-default ``output_dtype`` attribute, blocking on any axis other than
    this weight's own reduction axis (``1 - expected_axis``), a
    reduction-axis dim that isn't an exact multiple of `block_size` (a
    padded/partial final block -- mirrors ``MatMulNBits``'s own identical
    scope decision), a scale shaped other than exactly what blockwise
    quantization's own schema specifies for this `block_size`, a
    ``DequantizeLinear`` output read by more than one consumer, or any of
    ``x``/``x_scale``/``x_zero_point`` read by more than one node (a
    shared/tied quantized tensor this weight's own slicing would otherwise
    silently corrupt for that other reader).
    """
    dq = dq_of.get(weight_name)
    if dq is None or dq.op_type != "DequantizeLinear" or len(dq.output) != 1:
        return None
    if len(consumers_of.get(weight_name, [])) != 1:
        return None  # DQ output must feed only this one weight use
    if len(dq.input) not in (2, 3):
        return None
    q_name, scale_name = dq.input[0], dq.input[1]
    zp_name = dq.input[2] if len(dq.input) == 3 and dq.input[2] else None
    if not q_name or not scale_name:
        return None

    q_init = initializer_map.get(q_name)
    scale_init = initializer_map.get(scale_name)
    if q_init is None or scale_init is None:
        return None  # non-constant q/scale -- can't safely slice it
    if q_init.data_type not in (onnx.TensorProto.INT4, onnx.TensorProto.UINT4):
        return None  # INT8/UINT8 (scalar/per-channel) is
        # `_match_dequantize_linear_weight`'s own job; FLOAT8 needs
        # different range handling neither matcher attempts.
    if scale_init.data_type != onnx.TensorProto.FLOAT:
        return None  # only FLOAT32 scale empirically verified -- see this
        # section's own top comment
    if len(q_init.dims) != rank:
        return None

    zp_init = None
    if zp_name is not None:
        zp_init = initializer_map.get(zp_name)
        if zp_init is None or zp_init.data_type != q_init.data_type:
            return None  # schema: x_zero_point and x must have the same type
        if list(zp_init.dims) != list(scale_init.dims):
            return None  # schema: x_scale and x_zero_point must have the
            # same shape

    for nm in (q_name, scale_name) + ((zp_name,) if zp_name else ()):
        if len(consumers_of.get(nm, [])) != 1:
            return None  # shared/tied quantized tensor -- another node
            # reads it too, so it can't be sliced only for this one

    block_size = 0
    output_dtype = 0
    for attr in dq.attribute:
        if attr.name == "block_size":
            block_size = attr.i
        elif attr.name == "output_dtype":
            output_dtype = attr.i
    if block_size <= 0:
        return None  # not blockwise -- `_match_dequantize_linear_weight`'s
        # own job
    if output_dtype != 0:
        return None  # non-default output dtype -- out of scope

    axis = 1  # DequantizeLinear's own schema default
    for attr in dq.attribute:
        if attr.name == "axis":
            axis = attr.i
            break
    if axis < 0:
        axis += rank

    block_axis = 1 - expected_axis
    if axis != block_axis:
        return None  # blocking on any axis other than this weight's own
        # reduction/input-channel axis -- not the shape this pass prunes,
        # decline rather than guess (see this section's own top comment for
        # why this is always the axis this weight is quantized along, in
        # every case actually verified)

    dims = list(q_init.dims)
    block_dim = dims[block_axis]
    if block_size <= 0 or block_dim % block_size != 0:
        return None  # padded/partial final block -- declined, see this
        # section's own top comment
    num_blocks = block_dim // block_size

    expected_scale_dims = list(dims)
    expected_scale_dims[block_axis] = num_blocks
    if list(scale_init.dims) != expected_scale_dims:
        return None

    return _QDQBlockwiseWeight(
        dq_node=dq,
        q_init=q_init,
        scale_init=scale_init,
        zero_point_init=zp_init,
        out_axis=expected_axis,
        block_axis=block_axis,
        block_size=block_size,
        num_blocks=num_blocks,
    )


@dataclass(frozen=True)
class _WeightRef:
    """A Conv/MatMul/Gemm weight resolved from one of the three sources
    :func:`_resolve_weight_ref` distinguishes: a direct float32/float16/
    bfloat16 initializer (`float_init`, exactly what every matcher earlier
    in this module already requires), a QDQ ``DequantizeLinear``-fed
    int8/uint8, per-tensor/per-channel one (`qdq`, matched by
    :func:`_match_dequantize_linear_weight`), or a QDQ
    ``DequantizeLinear``-fed INT4/UINT4, *blockwise* one (`qdq_block`,
    matched by :func:`_match_dequantize_linear_weight_blockwise`). Exactly
    one of the three is set.
    """

    float_init: Optional[onnx.TensorProto] = None
    qdq: Optional[_QDQWeight] = None
    qdq_block: Optional[_QDQBlockwiseWeight] = None

    @property
    def is_qdq(self) -> bool:
        return self.qdq is not None or self.qdq_block is not None


def _resolve_weight_ref(
    weight_name: str,
    rank: int,
    expected_axis: int,
    initializer_map: Dict[str, onnx.TensorProto],
    dq_of: Dict[str, onnx.NodeProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
) -> Optional[_WeightRef]:
    """Resolves `weight_name` to a :class:`_WeightRef`, trying a direct
    float32/float16/bfloat16 initializer first (the common, cheap check),
    then a per-tensor/per-channel QDQ ``DequantizeLinear``-fed one (see
    :func:`_match_dequantize_linear_weight`), then a blockwise INT4/UINT4
    one (see :func:`_match_dequantize_linear_weight_blockwise`). ``None``
    when none of the three matches.
    """
    w_init = initializer_map.get(weight_name)
    if w_init is not None:
        if _is_supported_float_dtype(w_init.data_type) and len(w_init.dims) == rank:
            return _WeightRef(float_init=w_init)
        return None
    qdq = _match_dequantize_linear_weight(
        weight_name, rank, expected_axis, initializer_map, dq_of, consumers_of
    )
    if qdq is not None:
        return _WeightRef(qdq=qdq)
    qdq_block = _match_dequantize_linear_weight_blockwise(
        weight_name, rank, expected_axis, initializer_map, dq_of, consumers_of
    )
    if qdq_block is not None:
        return _WeightRef(qdq_block=qdq_block)
    return None


def _weight_ref_dims(ref: _WeightRef) -> Tuple[int, ...]:
    if ref.float_init is not None:
        return tuple(ref.float_init.dims)
    if ref.qdq is not None:
        return tuple(ref.qdq.q_init.dims)
    assert ref.qdq_block is not None
    return tuple(ref.qdq_block.q_init.dims)


def _weight_ref_key(ref: _WeightRef) -> str:
    """A name uniquely identifying the underlying tensor `ref` resolves to
    -- the int8/int4 ``q_init`` for a QDQ weight, per-tensor/per-channel or
    blockwise alike (which, per :func:`_match_dequantize_linear_weight`'s/
    :func:`_match_dequantize_linear_weight_blockwise`'s own single-consumer
    check, is read by exactly the one ``DequantizeLinear`` feeding exactly
    this one weight use, so it is as safe a per-weight identity key as a
    plain float initializer's own name already is elsewhere in this
    module) or the float initializer's own name otherwise. Used by
    :func:`apply_structured_pruning_qdq` to detect a shared/tied weight
    playing the same role (producer or consumer) in more than one chain.
    """
    if ref.float_init is not None:
        return ref.float_init.name
    if ref.qdq is not None:
        return ref.qdq.q_init.name
    assert ref.qdq_block is not None
    return ref.qdq_block.q_init.name


def _weight_ref_dequantized(ref: _WeightRef) -> np.ndarray:
    """The full float64 array `ref` refers to, for IMPORTANCE RANKING ONLY
    -- never written back to the graph. The actual mutation always slices
    the int8/int4 codes/scale/zero-point directly
    (:func:`_slice_producer_weight_qdq`/:func:`_slice_consumer_weight_qdq`
    for per-tensor/per-channel, :func:`_slice_producer_weight_qdq_block`/
    :func:`_slice_consumer_weight_qdq_block` for blockwise), exactly the
    "slice, don't recompute" principle this section's own top-of-section
    comment describes; this helper exists purely so a QDQ producer's output
    channels can be ranked by the same L1/L2-norm-of-dequantized-row
    criterion a plain float producer's already are
    (:func:`_qdq_channel_importance`), without that ranking caring which
    source it came from.
    """
    if ref.float_init is not None:
        return _to_f64(ref.float_init)
    if ref.qdq is not None:
        qdq = ref.qdq
        q = onnx.numpy_helper.to_array(qdq.q_init).astype(np.float64)
        scale = onnx.numpy_helper.to_array(qdq.scale_init).astype(np.float64)
        if qdq.zero_point_init is not None:
            zp = onnx.numpy_helper.to_array(qdq.zero_point_init).astype(np.float64)
        else:
            zp = np.float64(0.0)
        if qdq.per_channel:
            shape = [1] * q.ndim
            shape[qdq.axis] = -1
            scale = scale.reshape(shape)
            if qdq.zero_point_init is not None:
                zp = zp.reshape(shape)
        return (q - zp) * scale

    qdq_block = ref.qdq_block
    assert qdq_block is not None
    q = onnx.numpy_helper.to_array(qdq_block.q_init).astype(np.float64)
    scale = onnx.numpy_helper.to_array(qdq_block.scale_init).astype(np.float64)
    # `scale`/`zero_point` are full-rank, `num_blocks`-sized (not
    # `block_dim`-sized) along `block_axis` -- broadcast each block's own
    # scalar to every element of its own `block_size`-sized span by
    # repeating along that axis (see this section's own top-of-section
    # comment: `block_size` blocks are always contiguous, `num_blocks *
    # block_size == block_dim` exactly, since the matcher declines a
    # non-exact-multiple reduction-axis dim).
    scale = np.repeat(scale, qdq_block.block_size, axis=qdq_block.block_axis)
    if qdq_block.zero_point_init is not None:
        zp = onnx.numpy_helper.to_array(qdq_block.zero_point_init).astype(np.float64)
        zp = np.repeat(zp, qdq_block.block_size, axis=qdq_block.block_axis)
    else:
        zp = np.float64(0.0)
    return (q - zp) * scale


def _qdq_channel_importance(w_nk: np.ndarray, importance_norm: str) -> np.ndarray:
    """L1 or L2 norm of each output channel's own (dequantized) weight row
    -- the same criterion :func:`_plain_structured_importance` uses for a
    plain float chain, just for the single-producer case this section's own
    narrower QDQ chains are always restricted to (no gated pair to combine).
    """
    if importance_norm == "l1":
        return np.linalg.norm(w_nk, ord=1, axis=1)
    return np.linalg.norm(w_nk, axis=1)


def _slice_producer_weight_qdq(
    ref: _WeightRef, weight_transposed: bool, keep: np.ndarray, is_conv: bool
) -> None:
    """Slices `ref`'s own output channels to `keep` (ascending indices) --
    the producer role. For a per-tensor/per-channel QDQ weight this means
    the int8 ``Wq`` AND (when `per_channel`) its own ``Ws``/``Wzp`` are all
    sliced together by the same `keep`, in lockstep (see this section's own
    top-of-section comment); a per-tensor QDQ weight only slices ``Wq`` --
    its scalar ``Ws``/``Wzp`` apply uniformly to every channel regardless
    of how many survive, so there is nothing else to touch. For a
    *blockwise* QDQ weight (`ref.qdq_block`), delegates to
    :func:`_slice_producer_weight_qdq_block` (co-slicing ``Wq``/``Ws``/
    ``Wzp`` in lockstep too -- `scale`/`zero_point` are always full-size,
    never block-reduced, on this axis, see this section's own top comment).
    Mirrors :func:`_slice_producer_weight` exactly for the float case
    (delegated to it directly).
    """
    if ref.float_init is not None:
        _slice_producer_weight(ref.float_init, weight_transposed, keep, is_conv=is_conv)
        return
    if ref.qdq is not None:
        qdq = ref.qdq
        q = onnx.numpy_helper.to_array(qdq.q_init)
        if is_conv:
            q_new = q[keep, ...]
        else:
            q_new = q[keep, :] if weight_transposed else q[:, keep]
        qdq.q_init.CopyFrom(onnx.numpy_helper.from_array(q_new, name=qdq.q_init.name))
        if qdq.per_channel:
            scale = onnx.numpy_helper.to_array(qdq.scale_init)
            qdq.scale_init.CopyFrom(
                onnx.numpy_helper.from_array(scale[keep], name=qdq.scale_init.name)
            )
            if qdq.zero_point_init is not None:
                zp = onnx.numpy_helper.to_array(qdq.zero_point_init)
                qdq.zero_point_init.CopyFrom(
                    onnx.numpy_helper.from_array(
                        zp[keep], name=qdq.zero_point_init.name
                    )
                )
        return
    qdq_block = ref.qdq_block
    assert qdq_block is not None
    _slice_producer_weight_qdq_block(qdq_block, keep)


def _slice_producer_weight_qdq_block(w: _QDQBlockwiseWeight, keep: np.ndarray) -> None:
    """Slices `w`'s own output-channel axis (`out_axis`) to `keep`
    (ascending indices) -- the producer role for a blockwise-quantized
    weight. `block_size` never applies to `out_axis` (enforced by
    :func:`_match_dequantize_linear_weight_blockwise`, which always
    requires `block_axis == 1 - out_axis`), so `scale_init`/
    `zero_point_init` are full-size (unreduced) there too, exactly like a
    per-channel (unblocked) QDQ weight's own scale/zero-point at ITS own
    per-channel axis -- co-sliced with `q_init` by the same `keep`, in
    lockstep, with no block-alignment concern whatsoever (unlike the
    consumer role below). `q_init`/`scale_init`/`zero_point_init` are all
    read/written via ``onnx.numpy_helper.to_array``/``from_array``, which
    transparently pack/unpack the INT4/UINT4 codes for us through
    ``ml_dtypes`` -- see this section's own top comment for why no manual
    nibble bookkeeping (unlike ``MatMulNBits``'s own byte-packed
    representation) is needed here.
    """
    q = onnx.numpy_helper.to_array(w.q_init)
    q_new = np.take(q, keep, axis=w.out_axis)
    w.q_init.CopyFrom(onnx.numpy_helper.from_array(q_new, name=w.q_init.name))

    scale = onnx.numpy_helper.to_array(w.scale_init)
    scale_new = np.take(scale, keep, axis=w.out_axis)
    w.scale_init.CopyFrom(
        onnx.numpy_helper.from_array(scale_new, name=w.scale_init.name)
    )

    if w.zero_point_init is not None:
        zp = onnx.numpy_helper.to_array(w.zero_point_init)
        zp_new = np.take(zp, keep, axis=w.out_axis)
        w.zero_point_init.CopyFrom(
            onnx.numpy_helper.from_array(zp_new, name=w.zero_point_init.name)
        )


def _slice_consumer_weight_qdq(
    ref: _WeightRef, weight_transposed: bool, keep: np.ndarray, is_conv: bool
) -> None:
    """Slices `ref`'s own input (reduction) channels to `keep` -- the
    consumer role for a plain float or per-tensor/per-channel QDQ weight
    only. For a per-tensor/per-channel QDQ weight this only ever slices the
    int8 ``Wq`` itself: ``Ws``/``Wzp`` are indexed by this weight's own
    OUTPUT channel axis (per-channel) or are a scalar (per-tensor), never
    by the input/reduction axis sliced here, so they are always left
    completely untouched -- genuinely simpler than the producer role, not
    just differently shaped (see this section's own top-of-section
    comment). Mirrors :func:`_slice_consumer_weight` exactly for the float
    case (delegated to it directly).

    Never called for a *blockwise* QDQ weight (`ref.qdq_block`): unlike
    every other case here, that role's own axis (`block_axis`) genuinely IS
    the blocked one, so it needs a pre-validated block-aligned `keep`
    (:func:`_qdq_block_aligned_keep_blocks`) and its own dedicated slicer
    (:func:`_slice_consumer_weight_qdq_block`) that also re-slices
    `scale`/`zero_point` by BLOCK index -- :func:`apply_structured_pruning_qdq`'s
    own main loop dispatches to whichever of the two this chain's consumer
    actually needs, checking alignment (and possibly declining the whole
    chain) before calling either.
    """
    if ref.float_init is not None:
        _slice_consumer_weight(ref.float_init, weight_transposed, keep, is_conv=is_conv)
        return
    qdq = ref.qdq
    assert qdq is not None
    q = onnx.numpy_helper.to_array(qdq.q_init)
    if is_conv:
        q_new = q[:, keep, ...]
    else:
        q_new = q[:, keep] if weight_transposed else q[keep, :]
    qdq.q_init.CopyFrom(onnx.numpy_helper.from_array(q_new, name=qdq.q_init.name))


def _qdq_block_aligned_keep_blocks(
    keep: np.ndarray, block_dim: int, block_size: int
) -> Optional[np.ndarray]:
    """Returns the ascending block indices (into the ``block_dim //
    block_size``-sized block axis) that `keep` (ascending element indices
    into the `block_dim`-length axis being pruned) corresponds to when
    every `block_size`-sized block is either wholly present in `keep` or
    wholly absent -- or ``None`` when some block is only partially kept,
    meaning this consumer cannot be safely pruned to this exact `keep` set
    at all (an individual reduction-axis element can't be dropped without
    re-quantizing its whole block -- out of scope, see this section's own
    top comment). The blockwise INT4/UINT4 analogue of
    :func:`_matmul_nbits_block_aligned_keep_blocks` -- the underlying
    integer-set math is identical, but duplicated here (rather than shared)
    to keep this section fully self-contained, the same reason this
    module's own "MatMulNBits" section gives for not sharing slicing infra
    across differently-shaped ops.
    """
    keep_set = set(int(k) for k in keep)
    num_blocks = block_dim // block_size
    keep_blocks = []
    for b in range(num_blocks):
        lo = b * block_size
        block_positions = set(range(lo, lo + block_size))
        overlap = block_positions & keep_set
        if overlap == block_positions:
            keep_blocks.append(b)
        elif overlap:
            return None  # partial block -- not block-aligned, decline
    return np.array(keep_blocks, dtype=np.int64)


def _slice_consumer_weight_qdq_block(
    w: _QDQBlockwiseWeight, keep: np.ndarray, keep_blocks: np.ndarray
) -> None:
    """Slices `w`'s own input/reduction (`block_axis`) axis -- the consumer
    role for a blockwise-quantized weight. Never called with a
    non-block-aligned `keep`; see :func:`_qdq_block_aligned_keep_blocks`,
    which computes and validates `keep_blocks` from it before this is
    called. `q_init` is sliced element-wise by `keep` directly (each kept
    block's own elements stay contiguous, since `keep` is block-aligned by
    construction); `scale_init`/`zero_point_init` are indexed by BLOCK, not
    element, so they are sliced by `keep_blocks` instead, along the same
    `block_axis`.
    """
    q = onnx.numpy_helper.to_array(w.q_init)
    q_new = np.take(q, keep, axis=w.block_axis)
    w.q_init.CopyFrom(onnx.numpy_helper.from_array(q_new, name=w.q_init.name))

    scale = onnx.numpy_helper.to_array(w.scale_init)
    scale_new = np.take(scale, keep_blocks, axis=w.block_axis)
    w.scale_init.CopyFrom(
        onnx.numpy_helper.from_array(scale_new, name=w.scale_init.name)
    )

    if w.zero_point_init is not None:
        zp = onnx.numpy_helper.to_array(w.zero_point_init)
        zp_new = np.take(zp, keep_blocks, axis=w.block_axis)
        w.zero_point_init.CopyFrom(
            onnx.numpy_helper.from_array(zp_new, name=w.zero_point_init.name)
        )


@dataclass(frozen=True)
class _QDQProducer:
    node: onnx.NodeProto
    ref: _WeightRef
    bias: Optional[str]
    weight_transposed: bool
    is_conv: bool


@dataclass(frozen=True)
class _QDQConsumer:
    node: onnx.NodeProto
    ref: _WeightRef
    weight_transposed: bool
    is_conv: bool


@dataclass(frozen=True)
class _QDQChain:
    producer: _QDQProducer
    chain_ops: Tuple[onnx.NodeProto, ...]
    consumer: _QDQConsumer
    n_channels: int


def _match_conv_qdq(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    dq_of: Dict[str, onnx.NodeProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
) -> Optional[Tuple[_WeightRef, Optional[str], int, int]]:
    """If `node` is an ordinary (``group=1``) 2-D ``Conv`` whose weight
    resolves (:func:`_resolve_weight_ref`) to either a direct float
    initializer or a QDQ one, returns ``(ref, bias_name_or_None,
    out_channels, in_channels)``. A grouped or depthwise Conv is never
    matched here -- see this section's own top-of-section comment for why
    that composition is out of scope.
    """
    if node.op_type != "Conv" or len(node.input) < 2:
        return None
    if _conv_group(node) != 1:
        return None
    w_name = node.input[1]
    ref = _resolve_weight_ref(w_name, 4, 0, initializer_map, dq_of, consumers_of)
    if ref is None:
        return None
    dims = _weight_ref_dims(ref)
    out_channels, in_channels = dims[0], dims[1]
    bias_name = None
    if len(node.input) == 3 and node.input[2]:
        bias_name = node.input[2]
        b_init = initializer_map.get(bias_name)
        if b_init is None or not _is_supported_float_dtype(b_init.data_type):
            return None  # non-constant bias -- can't safely slice it
    return ref, bias_name, out_channels, in_channels


def _match_matmul_qdq(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    dq_of: Dict[str, onnx.NodeProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
) -> Optional[Tuple[str, _WeightRef, bool, Optional[str], int, int]]:
    """If `node` is a MatMul/vanilla-Gemm whose weight resolves
    (:func:`_resolve_weight_ref`) to either a direct float initializer or a
    QDQ one, returns ``(x_name, ref, weight_transposed, bias_name_or_None,
    out_channels, in_channels)``.
    """
    match = _match_matmul_like(node)
    if match is None:
        return None
    x_name, w_name, weight_transposed = match
    axis = 0 if weight_transposed else 1
    ref = _resolve_weight_ref(w_name, 2, axis, initializer_map, dq_of, consumers_of)
    if ref is None:
        return None
    dims = _weight_ref_dims(ref)
    out_channels, in_channels = dims[axis], dims[1 - axis]
    bias_name = None
    if node.op_type == "Gemm" and len(node.input) == 3 and node.input[2]:
        bias_name = node.input[2]
        b_init = initializer_map.get(bias_name)
        if b_init is None or not _is_supported_float_dtype(b_init.data_type):
            return None  # non-constant bias -- can't safely slice it
    return x_name, ref, weight_transposed, bias_name, out_channels, in_channels


def _walk_to_consumer_qdq(
    start: str,
    is_conv: bool,
    initializer_map: Dict[str, onnx.TensorProto],
    dq_of: Dict[str, onnx.NodeProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    n_channels: int,
    max_hops: int,
) -> Optional[Tuple[_QDQConsumer, Tuple[onnx.NodeProto, ...]]]:
    """From tensor `start`, walks forward through shape-preserving unary
    activations (`_UNARY_PASS_THROUGH`) with no other consumer anywhere
    along the way, until a same-family (Conv-only or MatMul/Gemm-only,
    matching `is_conv`) consumer is found whose input-channel count matches
    `n_channels`. No per-channel Add/Mul bias/scale hop, no depthwise Conv
    pass-through, no branch -- narrower than :func:`_walk_to_consumer`/
    :func:`_walk_to_conv_consumer` by design, see this section's own
    top-of-section comment. Returns ``None`` if the walk runs out of hops,
    hits a branch, or never reaches such a consumer.
    """
    chain_ops: List[onnx.NodeProto] = []
    cur = start
    for _hop in range(max_hops):
        candidates = consumers_of.get(cur, [])
        if len(candidates) != 1:
            return None
        nxt = candidates[0]

        if is_conv:
            if nxt.op_type == "Conv" and nxt.input[0] == cur:
                m = _match_conv_qdq(nxt, initializer_map, dq_of, consumers_of)
                if m is None or m[3] != n_channels:
                    return None
                ref, _bias, _out, _in = m
                return _QDQConsumer(nxt, ref, False, True), tuple(chain_ops)
        else:
            mm = _match_matmul_qdq(nxt, initializer_map, dq_of, consumers_of)
            if mm is not None and mm[0] == cur:
                if mm[5] != n_channels:
                    return None
                _x, ref, weight_transposed, _bias, _out, _in = mm
                return _QDQConsumer(nxt, ref, weight_transposed, False), tuple(
                    chain_ops
                )

        if not (
            nxt.op_type in _UNARY_PASS_THROUGH
            and list(nxt.input) == [cur]
            and len(nxt.output) == 1
        ):
            return None
        out2 = nxt.output[0]
        if len(consumers_of.get(out2, [])) != 1 or out2 in graph_outputs:
            return None
        chain_ops.append(nxt)
        cur = out2
    return None


def _find_qdq_chains(graph: onnx.GraphProto) -> List[_QDQChain]:
    """The QDQ analogue of :func:`_find_chains`/:func:`_find_conv_chains`,
    restricted to the single-producer/single-consumer/unary-hops-only
    topology :func:`_walk_to_consumer_qdq` matches, and requiring at least
    one side of the pair to actually be QDQ (a plain float/float pair is
    :func:`apply_structured_pruning`'s own job, not duplicated here).
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    dq_of = {
        n.output[0]: n
        for n in graph.node
        if n.op_type == "DequantizeLinear" and len(n.output) == 1
    }
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains: List[_QDQChain] = []
    for node in graph.node:
        if node.op_type == "Conv":
            m = _match_conv_qdq(node, initializer_map, dq_of, consumers_of)
            if m is None:
                continue
            ref, bias_name, out_channels, _in_channels = m
            weight_transposed = False
            is_conv = True
        else:
            mm = _match_matmul_qdq(node, initializer_map, dq_of, consumers_of)
            if mm is None:
                continue
            _x_name, ref, weight_transposed, bias_name, out_channels, _in_channels = mm
            is_conv = False

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        found = _walk_to_consumer_qdq(
            out_name,
            is_conv,
            initializer_map,
            dq_of,
            consumers_of,
            graph_outputs,
            out_channels,
            _MAX_CHAIN_HOPS,
        )
        if found is None:
            continue
        consumer, chain_ops = found
        if not (ref.is_qdq or consumer.ref.is_qdq):
            continue  # both plain float -- apply_structured_pruning's job

        chains.append(
            _QDQChain(
                producer=_QDQProducer(node, ref, bias_name, weight_transposed, is_conv),
                chain_ops=chain_ops,
                consumer=consumer,
                n_channels=out_channels,
            )
        )
    return chains


def apply_structured_pruning_qdq(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
) -> onnx.ModelProto:
    """Removes whole output channels from a Conv or MatMul/vanilla-Gemm
    layer whose weight -- on either side of the producer/consumer pair, or
    both -- is statically quantized in the QDQ format (a constant int8/
    uint8/int4/uint4 initializer fed through a ``DequantizeLinear`` node),
    per-tensor, per-channel, or opset 21+ blockwise alike, the way
    :func:`onnxsim.calibration.quantize_static` (and the wider ONNX
    ecosystem's own standard static-quantization tooling, e.g. Olive or
    Hugging Face Optimum's own INT4 weight-only exporters for the blockwise
    case) produces one. See this module's "QDQ (quantized-weight)
    structured pruning" section comment for the full investigation this
    scope was reached from: what this repo's own QDQ tooling emits
    (cross-checked against the live ``DequantizeLinear`` schema), why
    structural pruning of a per-channel quantized weight reduces to the
    same "slice, don't recompute" principle the FP16/BFloat16 weight
    support above already established (co-slicing the int8 codes with
    their own per-channel scale/zero-point on the producer side; touching
    only the int8 codes, never the scale/zero-point, on the consumer
    side), why a *blockwise* INT4/UINT4 weight's producer/consumer roles
    are asymmetric in the OPPOSITE way (scale/zero-point untouched on the
    producer side, block-aligned co-slicing -- or an outright decline for a
    non-block-aligned `keep` set -- required on the consumer side, since
    `block_size` always blocks the reduction axis, never the output-channel
    one), why *unstructured* pruning of a QDQ weight is a fundamentally
    harder, declined-rather-than-guessed-at problem (already naturally
    excluded by every existing unstructured matcher, confirmed
    empirically), and why the *opposite* composition -- pruning a float
    model, then quantizing the result -- already works with no code changes
    here at all.

    For every Conv (``group=1``) or MatMul/vanilla-Gemm node (the
    "producer") whose output feeds, through zero or more shape-preserving
    unary activations (`_UNARY_PASS_THROUGH`) with no other consumer
    anywhere along that path, into exactly one downstream same-family node
    (the "consumer") whose input/reduction dimension matches -- and where at
    least one of the two is QDQ-quantized (a plain float/float pair is
    :func:`apply_structured_pruning`'s own job, not duplicated here): ranks
    the producer's output channels by L1/L2 norm of their own dequantized
    weight row (:func:`_qdq_channel_importance` -- dequantized for ranking
    only, never for the actual rewrite), drops the lowest-``sparsity``-
    fraction of them, and slices the producer's weight and bias (if it has
    a constant float one -- bias is never itself quantized by this repo's
    own QDQ tooling) together with the matching input channels from the
    consumer's weight. The producer's own int8/int4 codes are always
    co-sliced with its own scale/zero-point in lockstep, whichever source
    it is (per-tensor: scale/zero-point untouched; per-channel or
    blockwise: co-sliced by the same output-channel-axis `keep` -- see this
    function's own top comment for why blockwise's own `block_size` never
    applies to this axis). On the consumer side, a per-tensor/per-channel
    weight only ever has its int8 codes sliced (its own scale/zero-point
    are indexed by its OUTPUT channel axis, never touched by slicing its
    input axis); a *blockwise* weight's `keep` must instead align to its
    own `block_size` boundaries on this axis (every block wholly kept or
    wholly dropped, since a block's own values are quantized together and
    can't be partially dropped without re-quantizing -- out of scope, this
    module never invents new quantized values) -- when aligned, its codes
    AND scale/zero-point are co-sliced by block; when NOT aligned, this
    whole chain (producer and consumer alike) is left completely untouched
    rather than forcing a partial-block re-quantization or a disagreeing
    keep-set between the two, mirroring
    :func:`apply_structured_pruning_matmul_nbits`'s own identical
    precedent.

    No general grouped/depthwise Conv, gated (SwiGLU/GeGLU) pair, residual/
    skip-connection merge, or Concat-merged branch group is matched for a
    QDQ chain -- every one of those is already a materially bigger project
    for the plain float case (:func:`apply_structured_pruning`'s own
    docstring); composing any of them with QDQ's extra scale/zero-point
    bookkeeping is out of scope here (see this module's "QDQ" section
    comment for the full reasoning). Call :func:`apply_structured_pruning`/
    :func:`apply_structured_wanda_pruning` for those topologies on an
    all-float graph, and this function for the narrower QDQ-aware slice
    above; the two never touch the same tensor (a QDQ chain requires at
    least one QDQ-quantized side, which the float-only passes above can
    never match in the first place, their own weight matchers requiring a
    *direct* float initializer).

    :param model: onnx ModelProto object or file path
    :param sparsity: fraction of each eligible producer's output channels to
            drop (rounded, at least one channel is always kept)
    :param importance_norm: ``"l2"`` (default, Li et al.'s original
            filter-pruning criterion) or ``"l1"``
    :returns: the pruned onnx ModelProto
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = _find_qdq_chains(graph)
    if not chains:
        return out

    initializer_map = {t.name: t for t in graph.initializer}
    producer_touched: Set[str] = set()
    consumer_touched: Set[str] = set()
    stale_value_info: Set[str] = set()

    for chain in chains:
        p, c = chain.producer, chain.consumer
        p_key = _weight_ref_key(p.ref)
        c_key = _weight_ref_key(c.ref)
        if p_key == c_key:
            continue  # degenerate (the same weight in both roles)
        if p_key in producer_touched or c_key in consumer_touched:
            continue  # a shared/tied weight another chain already resized

        n = chain.n_channels
        keep_count = max(1, n - round(n * sparsity))
        if keep_count >= n:
            continue  # rounds down to nothing for this layer -- no-op

        w = _weight_ref_dequantized(p.ref)
        w_nk = _weight_to_nk(w, p.weight_transposed, p.is_conv)
        importance = _qdq_channel_importance(w_nk, importance_norm)
        # `kind="stable"` matters here specifically (unlike most other
        # argsort call sites in this module, which never feed into a
        # block-alignment check): with an exactly-tied `importance` vector,
        # an unstable sort's tie-break order is platform-dependent, and a
        # reordered tie can select a keep-set that no longer aligns to a
        # blockwise consumer's own block boundaries below -- flipping a
        # would-have-been-aligned chain to "declined" nondeterministically
        # (the same hazard confirmed empirically, on a real ARM CI runner,
        # for the MatMulNBits family's own identical block-alignment
        # pattern -- see that function's own comment on this same line). A
        # stable sort preserves input (channel index) order for ties, so an
        # all-tied producer's top-`keep_count` channels are always its
        # first `keep_count` indices, deterministic across platforms, and
        # matching this function's own `_analyze_*` dry-run mirror exactly.
        keep = np.sort(np.argsort(-importance, kind="stable")[:keep_count])

        keep_blocks = None
        if c.ref.qdq_block is not None:
            keep_blocks = _qdq_block_aligned_keep_blocks(
                keep, n, c.ref.qdq_block.block_size
            )
            if keep_blocks is None:
                continue  # non-block-aligned keep set for this blockwise
                # consumer -- decline the whole chain, see this function's
                # own top comment

        _slice_producer_weight_qdq(p.ref, p.weight_transposed, keep, p.is_conv)
        if p.bias is not None:
            _slice_last_axis(initializer_map[p.bias], keep)
        if keep_blocks is not None:
            assert c.ref.qdq_block is not None
            _slice_consumer_weight_qdq_block(c.ref.qdq_block, keep, keep_blocks)
        else:
            _slice_consumer_weight_qdq(c.ref, c.weight_transposed, keep, c.is_conv)

        producer_touched.add(p_key)
        consumer_touched.add(c_key)
        stale_value_info.add(p.node.output[0])
        stale_value_info.update(op.output[0] for op in chain.chain_ops)
        if p.ref.qdq is not None:
            stale_value_info.add(p.ref.qdq.dq_node.output[0])
        elif p.ref.qdq_block is not None:
            stale_value_info.add(p.ref.qdq_block.dq_node.output[0])
        if c.ref.qdq is not None:
            stale_value_info.add(c.ref.qdq.dq_node.output[0])
        elif c.ref.qdq_block is not None:
            stale_value_info.add(c.ref.qdq_block.dq_node.output[0])

    if stale_value_info:
        kept = [vi for vi in graph.value_info if vi.name not in stale_value_info]
        del graph.value_info[:]
        graph.value_info.extend(kept)
    return out


# --- MatMulNBits (block-quantized weight) structured pruning ---------------
#
# ``com.microsoft::MatMulNBits`` is the weight-only block quantization op
# ONNX Runtime GenAI's Model Builder emits for essentially every Linear
# layer in current (2024-2026) ONNX exports of Llama/Phi/Mistral/Qwen/Gemma-
# family models -- distinct from, and today probably more common for LLM
# deployment specifically than, the QDQ (``DequantizeLinear``-fed int8/
# uint8) pattern the section above handles. Unlike QDQ, the quantized
# weight here is never a separate initializer feeding a generic MatMul/Gemm
# through a dequantize node -- ``MatMulNBits`` IS the compute node itself,
# owning its packed weight/scale/zero-point/bias operands directly. This
# section is therefore a new, self-contained matcher + pass (mirroring the
# QDQ section's own *shape* -- producer/consumer chain walk, "slice, don't
# recompute", rank-by-dequantized-row-norm importance -- but not literally
# extending :class:`_WeightRef`/:func:`_resolve_weight_ref`, for the same
# reason the QDQ section itself gave for not retrofitting the plain-float
# machinery: this op's operand shapes/packing are unrelated enough that
# sharing code would risk both, for no compensating benefit).
#
# Every fact this section depends on was confirmed empirically, not
# assumed, against this environment's live onnxruntime (1.29.0) install --
# via ``onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()``
# for the schema itself, and a real ``onnxruntime.InferenceSession`` run
# for the packing layout and dequantization formula (see this section's own
# tests, e.g. ``test_matmul_nbits_pruning_producer_axis_oracle``, for the
# same round-trip re-run as an executable check rather than a comment-only
# claim):
#
#   * Inputs, in order: ``A`` (T1: float32/float16/bfloat16, unquantized
#     activation), ``B`` (T2: uint8 -- the packed/quantized weight),
#     ``scales`` (T1), ``zero_points`` (optional; T3: uint8 OR T1),
#     ``g_idx`` (optional; T4: int32 -- "group_idx. This input is
#     deprecated", per the live schema's own doc string), ``bias``
#     (optional, T1). A naive guess of "4 inputs" would have been wrong --
#     there are up to 6, and ``g_idx`` in particular is real, present in
#     the live 1.29.0 schema, not a hypothetical.
#   * Attributes: ``K``/``N`` (required ints -- input/output feature
#     counts of the logical, unquantized ``[N, K]`` weight matrix),
#     ``block_size`` (required int; "must be a power of two and not
#     smaller than 16"), ``bits`` (optional, default 4 -- confirmed from
#     the schema's own serialized default-value proto, not guessed;
#     "supported values: 2, 4, 8"), ``accuracy_level`` (optional, doesn't
#     affect layout, ignored here), ``weight_prepacked`` (optional,
#     default 0; "If set, input B is already prepacked into an EP-specific
#     layout" -- a nonzero value means ``B``'s bytes are in an opaque,
#     hardware-specific layout this section's slicing logic does not
#     understand, so it is always declined).
#   * ``B``'s packing layout for ``bits=4`` -- confirmed by hand-packing a
#     known float weight matrix, building a real ``MatMulNBits`` node from
#     the packed bytes, running it through a real CPU ``InferenceSession``,
#     and checking the output matches an independently computed
#     dequantize-then-matmul reference to float rounding error (~2e-7), not
#     just "roughly the right ballpark": shape ``(N, k_blocks, blob_size)``
#     with ``k_blocks = ceil(K / block_size)`` and ``blob_size = block_size
#     * bits / 8``, i.e. ``B``'s *leading* dimension is ``N`` (output
#     channels) -- never sub-block-packed -- so producer-side (N-axis)
#     pruning is a uniform row-slice exactly as a plausible a-priori guess
#     would expect. Within each ``(n, k_block)`` blob, K-values are
#     nibble-packed 2-per-byte, **low nibble first** (lower K index in the
#     low nibble, matching the schema doc's own "the first 4 bits are
#     stored in the lower 4 bits of a byte, and the second 4 bits are
#     stored in the higher 4 bits" -- confirmed, not assumed, by the same
#     InferenceSession round-trip).
#   * ``B``'s packing layout for ``bits=8`` -- verified by the SAME
#     methodology, independently, rather than assumed to extrapolate from
#     ``bits=4``: built a real ``bits=8`` ``MatMulNBits`` node from a
#     hand-quantized weight, ran it through a real CPU ``InferenceSession``,
#     and confirmed the output matches an independent dequantize-then-
#     matmul oracle (max abs error ~3.1e-7, relative ~4.3e-8 -- see
#     ``tests/test_pruning.py``'s own
#     ``test_matmul_nbits_pruning_bits8_producer_and_consumer_match_independent_reference_oracle``
#     for the same round-trip as an executable check). The layout is the
#     SAME shape formula (``(N, k_blocks, blob_size)``, ``blob_size =
#     block_size * bits / 8``), which for ``bits=8`` is exactly
#     ``block_size`` -- i.e. **one full uint8 code per ``B`` element, no
#     2-codes-per-byte nibble packing at all**, confirmed rather than
#     guessed from "8 bits fits in one byte so it's probably trivial": the
#     hypothesis was checked against a real kernel run, not asserted from
#     the bit width alone. Because the code-to-byte mapping is already 1:1,
#     the "low nibble first" ordering question that matters for ``bits=4``
#     simply doesn't arise for ``bits=8``.
#   * ``scales``: shape exactly ``(N, k_blocks)``, same float dtype as
#     ``A``. This section requires that exact rank-2 shape and declines
#     anything else (e.g. a flattened ``(N * k_blocks,)`` 1-D tensor) --
#     the live schema's own doc string states the 2-D shape and this
#     section has only empirically verified that one, not any flattened
#     alternative some export tool might in principle emit; broadening
#     this is a follow-up, not a silent guess.
#   * ``zero_points`` (optional): the live schema documents *two* valid
#     encodings, and this section supports both, dispatching on
#     ``zero_points``'s own dtype (never guessed, always read off the
#     tensor itself): **packed** (``uint8``, shape ``(N, ceil(k_blocks *
#     bits / 8))``, "same bit-packing method as Input B" -- confirmed by
#     the same InferenceSession round-trip, substituting a packed
#     ``zero_points`` tensor for the schema's documented default and
#     checking the output changes exactly as the explicit zero-point
#     would predict) or **unpacked** (same float dtype as ``A``, shape
#     ``(N, k_blocks)``, one zero-point value per block, no packing at
#     all -- confirmed by building the *same* logical model with an
#     unpacked float ``zero_points`` tensor instead of a packed uint8 one
#     and checking both real ``InferenceSession`` runs agree to float
#     rounding error). When absent, the schema's own documented default
#     applies: ``2 ** (bits - 1)`` for every block (8, for ``bits=4``; 128,
#     for ``bits=8`` -- both confirmed the same way, by omitting
#     ``zero_points`` entirely and checking the output against that
#     formula). For ``bits=8`` specifically, the **packed** encoding
#     follows the exact same "no nibble packing" fact as ``B`` above: ``(N,
#     ceil(k_blocks * 8 / 8))`` collapses to ``(N, k_blocks)``, one full
#     byte per block, confirmed by substituting a non-default packed
#     ``uint8`` ``zero_points`` tensor and checking the output changes
#     accordingly (~6.9e-7 abs error vs. the same dequantize oracle, and
#     materially different from the default-zero-point output -- not
#     merely "close enough to not notice a bug"). The **unpacked** (same-
#     dtype-as-``A``) encoding is schema-legal for ``bits=8`` too, but this
#     environment's live CPU kernel (onnxruntime 1.29.0) actually REJECTS
#     it at run time -- ``"Only 2b and 4b quantization is supported for
#     unpacked compute for now"`` -- confirmed by attempting exactly that
#     and catching the real ``InferenceSession`` failure, not assumed. This
#     section's matcher still structurally accepts an unpacked
#     ``zero_points`` for ``bits=8`` (matching what the schema's own type
#     constraints allow, the same policy every other matcher in this
#     module uses of not second-guessing a model that's already otherwise
#     well-formed) since a model built that way was never going to run on
#     this kernel regardless of what this section does with it -- but this
#     specific combination is therefore untested against a real
#     ``InferenceSession`` here, unlike every other packing fact this
#     comment documents.
#   * ``bias`` (optional): shape ``[N]``, same float dtype as ``A`` --
#     ordinary per-output-channel bias, added after the matmul.
#
# Scope boundaries this section lands on, deliberately, mirroring the QDQ
# section's own conservative-decline posture (broadening any of these is a
# safe follow-up; silently mishandling one is not):
#
#   * ``bits`` in ``{4, 8}`` are matched, each independently verified
#     against a real ``InferenceSession`` per the packing facts above.
#     ``bits == 2`` remains OUT OF SCOPE: the live schema allows it, but
#     2-bit packing (4 codes/byte? some other ratio? a different
#     zero_points convention again?) has not been empirically verified
#     here with the same rigor as 4- and 8-bit, and this section's own
#     culture (see above) is to verify against a real kernel run rather
#     than extrapolate a plausible-looking formula -- adding it is a safe,
#     independent follow-up once someone does that verification, not a
#     guess to make now.
#   * ``weight_prepacked != 0`` is always declined (see above -- an
#     opaque, EP-specific ``B`` layout).
#   * ``g_idx`` present (non-empty) is always declined: a GPTQ-style
#     column permutation index changes which *logical* K-column each
#     block covers, so "prune this contiguous range of K" and "drop this
#     block" no longer coincide the way the rest of this section assumes;
#     correctly composing pruning with an arbitrary permutation is a
#     materially harder, separate problem, out of scope here.
#   * ``K`` not an exact multiple of ``block_size`` (a padded, partial
#     final block) is declined: this section has not verified how ORT's
#     own kernel treats the padding region of a partial final block (e.g.
#     whether it's read at all), and getting that wrong would silently
#     corrupt exactly the case this module's own test philosophy exists to
#     catch. Every ``block_size`` this section accepts is additionally
#     required to be a power of two, >= 16 -- the live schema's own stated
#     constraint (also the set of sizes this environment's onnxruntime
#     kernel actually accepts at runtime, confirmed empirically: a
#     block_size of 8 raises "Only block sizes 16, 32, 64, 128, and 256
#     are supported").
#   * A chain's two sides need NOT both be ``MatMulNBits`` nodes anymore: a
#     ``MatMulNBits`` producer/consumer may instead pair with a plain-float
#     (directly-constant float32/float16/bfloat16 weight, no QDQ)
#     ``MatMul``/vanilla-``Gemm`` on the OTHER side -- see
#     :func:`_match_plain_matmul_nbits_peer`/:func:`_PlainMatMulNBitsPeer`,
#     the analogue of the QDQ section's own ``_resolve_weight_ref`` mixing
#     (float/QDQ) but narrower (``MatMulNBits``/plain-float only). This
#     mirrors a real, common export shape: ONNX Runtime GenAI's Model
#     Builder int4/int8-quantizes every transformer-block Linear layer via
#     ``MatMulNBits`` but frequently leaves the embedding/``lm_head``
#     layers (or a model exported with a different quantization recipe for
#     just those two layers) as plain float ``MatMul``/``Gemm`` -- so a
#     boundary chain (embedding -> first transformer block, or last block
#     -> ``lm_head``) is genuinely mixed. When the CONSUMER side of a
#     mixed chain is the plain-float one, there is no block structure to
#     respect at all, so any keep-set the producer side computes applies
#     directly, unlike the ``MatMulNBits``-consumer case below. When the
#     PRODUCER side is plain-float, the consumer's own block-alignment
#     requirement (below) still applies exactly as it does for a
#     ``MatMulNBits``-to-``MatMulNBits`` chain -- the constraint comes from
#     the CONSUMER's own quantization, not the producer's. A
#     QDQ-quantized weight (int8/uint8, ``DequantizeLinear``-fed) on the
#     other side remains OUT OF SCOPE: this section does not call
#     :func:`_resolve_weight_ref` at all (see the top-of-section rationale
#     for why this section is self-contained rather than sharing that
#     machinery), and mixing two DIFFERENT quantization schemes'
#     dequantize/requantize paths in one chain is a real but separate
#     extension left to a follow-up, not attempted here.
#   * No grouped/depthwise structure (there is none to speak of --
#     ``MatMulNBits`` has no ``group`` attribute), gated (SwiGLU/GeGLU)
#     pair, residual/skip-connection merge, or Concat-merged branch group
#     is matched -- only the plain single producer -> [zero or more
#     shape-preserving unary activations, `_UNARY_PASS_THROUGH`] -> single
#     consumer topology, identical in spirit to the QDQ section's own
#     ``_walk_to_consumer_qdq``.
#   * A shared/tied ``B``/``scales``/``zero_points``/``bias`` tensor (read
#     by more than one node) is declined by the matcher itself, the same
#     bar every other matcher in this module is held to -- a plain-float
#     peer's own weight/bias are held to the identical bar (see
#     :func:`_match_plain_matmul_nbits_peer`).
#
# The two pruning axes turn out just as asymmetric as the task investigating
# this op predicted, for exactly the reason the packing layout above
# implies:
#
#   * Producer-side (N-axis, output channels): ``B``'s, ``scales``'s, and
#     (if present) ``bias``'s leading dimension IS ``N`` -- an ordinary,
#     un-blocked row-slice for any subset of rows, in any count, the same
#     "slice, don't recompute" pattern used everywhere else in this module.
#     ``zero_points``, when present and **packed** (uint8, nibble-per-
#     block), needs more care: it is nibble-packed along the *block* axis
#     (one nibble per ``(n, k_block)`` pair), not the axis being sliced
#     here, so a row-slice alone is safe (each row's own bytes are
#     self-contained -- slicing whole rows never touches a byte shared
#     with a different row). What genuinely does need care -- and is
#     covered by a dedicated adversarial test
#     (``test_matmul_nbits_pruning_producer_odd_row_count_zero_points``)
#     -- is that a *consumer's* zero_points slicing along the *block* axis
#     (below) drops individual per-block nibbles from the *middle* of each
#     row's own packed bytes, which does shift byte alignment whenever the
#     kept block count is odd; producer-side pruning has no such hazard
#     (whole rows, whole bytes) but its own zero_points must still be
#     unpacked/resliced/repacked correctly to avoid a much simpler but
#     equally silent bug: forgetting this and naively byte-slicing
#     ``zero_points`` along its own row axis together with a *smaller*
#     ``zp_bytes`` width miscomputed from the wrong ``k_blocks`` would
#     silently write back a wrong-width tensor. (For ``bits=8`` this whole
#     packed-``zero_points`` repack is a byte-identity operation -- see the
#     schema-facts comment above -- so there is genuinely no nibble-parity
#     hazard on this axis for ``bits=8`` at all, unlike ``bits=4``; the
#     generalized :func:`_pack_nbits_codes`/:func:`_unpack_nbits_codes`
#     helpers still route through the same code path for both, so this is
#     a fact about the DATA, not a special case the code has to add.)
#   * Consumer-side (K-axis, input channels): each block of ``block_size``
#     K-values shares one scale/zero-point and is packed together within
#     ``B`` (nibble-packed for ``bits=4``, one byte per code for
#     ``bits=8``), so an individual K-column cannot be dropped without
#     re-quantizing its whole block -- out of scope (this module never
#     invents new quantized values, only drops/keeps existing ones
#     intact, exactly the QDQ section's own principle) whenever the
#     consumer is itself a ``MatMulNBits`` node. This section therefore
#     only supports dropping *entire* blocks from a ``MatMulNBits``
#     consumer: given the candidate keep-set computed the same way as
#     everywhere else in this module (top-``keep_count`` producer output
#     channels by dequantized L1/L2 row norm),
#     :func:`_matmul_nbits_block_aligned_keep_blocks` checks whether that
#     keep-set happens to already align to the consumer's own block
#     boundaries (every block either wholly kept or wholly dropped) -- if
#     so, the aligned whole blocks are dropped from ``B``/``scales``/
#     ``zero_points`` along the block axis (again a re-pack, not a raw
#     byte-slice, for packed ``zero_points`` -- the block axis is exactly
#     the packed axis here, unlike the producer-side row-slice above); if
#     not, this chain's consumer-side (and therefore its paired
#     producer-side) pruning is declined entirely, left completely
#     untouched, rather than forcing either a partial-block re-
#     quantization or a different, disagreeing keep-set between the two
#     paired weights. See
#     ``test_matmul_nbits_pruning_consumer_declines_non_block_aligned``
#     for this decline path exercised directly. A plain-float CONSUMER has
#     no block structure at all, so this alignment check simply does not
#     apply when the consumer side of a mixed chain is the plain-float one
#     -- the producer's own keep-set (whatever it is) applies directly, the
#     same as any other plain-float consumer elsewhere in this module.


def _matmul_nbits_int_attr(
    node: onnx.NodeProto, name: str, default: Optional[int] = None
) -> Optional[int]:
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


def _set_matmul_nbits_int_attr(node: onnx.NodeProto, name: str, value: int) -> None:
    for attr in node.attribute:
        if attr.name == name:
            attr.i = value
            return
    node.attribute.append(onnx.helper.make_attribute(name, value))


@dataclass(frozen=True)
class _MatMulNBitsWeight:
    """A ``com.microsoft::MatMulNBits`` node's block-quantized weight
    operands, matched by :func:`_match_matmul_nbits` -- see this section's
    own top comment for the schema facts and packing layout this depends
    on (empirically confirmed, not assumed). ``zero_points_packed``
    distinguishes the two encodings the live schema allows: ``True`` for
    nibble-packed ``uint8`` (same layout as `b_init`, along the block
    axis), ``False`` for one unpacked float value per block (same dtype as
    `scales_init`). Meaningless when `zero_points_init` is ``None``
    (absent -- the schema's own documented default, ``2 ** (bits - 1)``,
    applies).
    """

    node: onnx.NodeProto
    b_init: onnx.TensorProto
    scales_init: onnx.TensorProto
    zero_points_init: Optional[onnx.TensorProto]
    zero_points_packed: bool
    bias_init: Optional[onnx.TensorProto]
    N: int
    K: int
    bits: int
    block_size: int
    k_blocks: int


_MATMUL_NBITS_VALID_BLOCK_SIZES = {16, 32, 64, 128, 256}


def _match_matmul_nbits(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
) -> Optional[_MatMulNBitsWeight]:
    """If `node` is a ``com.microsoft::MatMulNBits`` node matching every
    scope boundary this section's own top comment documents, returns the
    match. ``None`` whenever anything is ambiguous or out of the
    empirically-verified scope, rather than guessing -- see that comment
    for the exhaustive list of what's declined and why.
    """
    if node.op_type != "MatMulNBits" or node.domain != "com.microsoft":
        return None
    if len(node.input) < 3 or len(node.output) != 1:
        return None
    a_name, b_name, scales_name = node.input[0], node.input[1], node.input[2]
    if not a_name or not b_name or not scales_name:
        return None
    zp_name = node.input[3] if len(node.input) > 3 and node.input[3] else None
    g_idx_name = node.input[4] if len(node.input) > 4 and node.input[4] else None
    bias_name = node.input[5] if len(node.input) > 5 and node.input[5] else None
    if g_idx_name is not None:
        return None  # GPTQ-style permutation -- declined, see section comment

    block_size = _matmul_nbits_int_attr(node, "block_size")
    N = _matmul_nbits_int_attr(node, "N")
    K = _matmul_nbits_int_attr(node, "K")
    if block_size is None or N is None or K is None or N <= 0 or K <= 0:
        return None
    if block_size not in _MATMUL_NBITS_VALID_BLOCK_SIZES:
        return None
    if K % block_size != 0:
        return None  # padded/partial final block -- declined, see section comment
    bits = _matmul_nbits_int_attr(node, "bits", 4)  # schema default: 4
    if bits not in (4, 8):
        return (
            None  # only 4-bit/8-bit packing empirically verified -- see section comment
        )
    weight_prepacked = _matmul_nbits_int_attr(node, "weight_prepacked", 0)
    if weight_prepacked != 0:
        return None  # EP-specific opaque prepacked layout -- see section comment

    k_blocks = K // block_size
    blob_size = block_size * bits // 8

    b_init = initializer_map.get(b_name)
    scales_init = initializer_map.get(scales_name)
    if b_init is None or scales_init is None:
        return None  # non-constant B/scales -- can't safely slice them
    if b_init.data_type != onnx.TensorProto.UINT8:
        return None
    if list(b_init.dims) != [N, k_blocks, blob_size]:
        return None
    if scales_init.data_type not in (
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
    ):
        return None
    if list(scales_init.dims) != [N, k_blocks]:
        return None

    zp_init = None
    zp_packed = False
    if zp_name is not None:
        zp_init = initializer_map.get(zp_name)
        if zp_init is None:
            return None  # non-constant zero_points -- can't safely slice it
        zp_bytes = (k_blocks * bits + 7) // 8
        if zp_init.data_type == onnx.TensorProto.UINT8:
            if list(zp_init.dims) != [N, zp_bytes]:
                return None
            zp_packed = True
        elif zp_init.data_type == scales_init.data_type:
            if list(zp_init.dims) != [N, k_blocks]:
                return None
            zp_packed = False
        else:
            return None

    bias_init = None
    if bias_name is not None:
        bias_init = initializer_map.get(bias_name)
        if bias_init is None or bias_init.data_type != scales_init.data_type:
            return None  # non-constant, or dtype-mismatched, bias
        if list(bias_init.dims) != [N]:
            return None

    for nm in (
        (b_name, scales_name)
        + ((zp_name,) if zp_name else ())
        + ((bias_name,) if bias_name else ())
    ):
        if len(consumers_of.get(nm, [])) != 1:
            return None  # shared/tied tensor -- another node reads it too

    return _MatMulNBitsWeight(
        node=node,
        b_init=b_init,
        scales_init=scales_init,
        zero_points_init=zp_init,
        zero_points_packed=zp_packed,
        bias_init=bias_init,
        N=N,
        K=K,
        bits=bits,
        block_size=block_size,
        k_blocks=k_blocks,
    )


def _unpack_nbits_nibbles(packed: np.ndarray, count: int) -> np.ndarray:
    """Unpacks the last axis of `packed` (``uint8``, 2 4-bit values per
    byte, **low nibble first** -- confirmed empirically, see this
    section's own top comment) into `count` ``uint8`` values in [0, 15]
    (dropping the last, padding, half-byte when `count` is odd).
    """
    nbytes = packed.shape[-1]
    out = np.zeros(packed.shape[:-1] + (2 * nbytes,), dtype=np.uint8)
    out[..., 0::2] = packed & 0x0F
    out[..., 1::2] = (packed >> 4) & 0x0F
    return out[..., :count]


def _pack_nbits_nibbles(vals: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_unpack_nbits_nibbles`: packs the last axis
    (``uint8`` values in [0, 15]) 2-per-byte, low nibble first, padding an
    odd trailing count with a zero high nibble. This -- rather than a raw
    byte-slice of the original packed tensor -- is exactly why this
    section re-packs instead of slicing bytes directly: dropping an ODD
    number of rows/blocks from the *middle* of a nibble-packed axis shifts
    every subsequent kept value's nibble parity, silently corrupting the
    result if not accounted for (see this section's own top comment and
    ``test_matmul_nbits_pruning_producer_odd_row_count_zero_points``).
    """
    count = vals.shape[-1]
    padded = vals
    if count % 2:
        pad_shape = vals.shape[:-1] + (1,)
        padded = np.concatenate([vals, np.zeros(pad_shape, dtype=vals.dtype)], axis=-1)
    lo = padded[..., 0::2].astype(np.uint8)
    hi = padded[..., 1::2].astype(np.uint8)
    return (lo & 0x0F) | ((hi & 0x0F) << 4)


def _unpack_nbits_codes(packed: np.ndarray, count: int, bits: int) -> np.ndarray:
    """The `bits`-aware generalization of :func:`_unpack_nbits_nibbles`:
    unpacks the last axis of `packed` (``uint8``) into `count` codes in
    ``[0, 2**bits - 1]``, dispatching on `bits` per this section's own
    empirically-confirmed packing (see this section's own top comment):
    ``bits == 4`` is nibble-packed 2-per-byte
    (:func:`_unpack_nbits_nibbles`, unchanged); ``bits == 8`` is one full
    byte per code -- no packing at all -- so this is a plain truncating
    slice, not a bit-unpack (confirmed, not assumed: see
    ``test_matmul_nbits_pruning_bits8_producer_and_consumer_match_independent_reference_oracle``).
    Every call site in this section routes through this dispatcher (rather
    than branching on `bits` itself) so ``bits=4``/``bits=8`` share one
    code path end to end.
    """
    if bits == 8:
        return packed[..., :count]
    assert bits == 4, bits  # _match_matmul_nbits only ever admits {4, 8}
    return _unpack_nbits_nibbles(packed, count)


def _pack_nbits_codes(vals: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of :func:`_unpack_nbits_codes`: the `bits`-aware
    generalization of :func:`_pack_nbits_nibbles`. ``bits == 8`` is a plain
    dtype cast (no packing to undo); ``bits == 4`` delegates to
    :func:`_pack_nbits_nibbles` unchanged.
    """
    if bits == 8:
        return vals.astype(np.uint8)
    assert bits == 4, bits  # _match_matmul_nbits only ever admits {4, 8}
    return _pack_nbits_nibbles(vals)


def _matmul_nbits_dequantized(w: _MatMulNBitsWeight) -> np.ndarray:
    """The full float64 ``(N, K)`` dequantized weight matrix `w` refers to,
    for IMPORTANCE RANKING ONLY -- never written back to the graph (this
    module's own "slice, don't recompute" principle, exactly as the QDQ
    section's :func:`_weight_ref_dequantized`). Formula and packing per
    this section's own top comment, empirically confirmed against a real
    ``InferenceSession``: ``dequantized[n, k] = (code[n, k] -
    zero_point[n, k // block_size]) * scale[n, k // block_size]``.
    """
    b = onnx.numpy_helper.to_array(w.b_init)  # (N, k_blocks, blob_size) uint8
    codes = _unpack_nbits_codes(b, w.block_size, w.bits).astype(np.float64)
    scales = onnx.numpy_helper.to_array(w.scales_init).astype(np.float64)
    if w.zero_points_init is not None:
        zp_raw = onnx.numpy_helper.to_array(w.zero_points_init)
        if w.zero_points_packed:
            zp = _unpack_nbits_codes(zp_raw, w.k_blocks, w.bits).astype(np.float64)
        else:
            zp = zp_raw.astype(np.float64)
    else:
        zp = np.full((w.N, w.k_blocks), float(1 << (w.bits - 1)), dtype=np.float64)
    dequant = (codes - zp[:, :, None]) * scales[:, :, None]  # (N, k_blocks, block_size)
    return dequant.reshape(w.N, w.k_blocks * w.block_size)


def _slice_matmul_nbits_producer_rows(w: _MatMulNBitsWeight, keep: np.ndarray) -> None:
    """Slices `w`'s own N (output-channel) axis to `keep` (ascending
    indices) -- the producer role. ``B``/``scales``/``bias`` (if present)
    are all row-sliced directly (their leading dim IS ``N``, per this
    section's own top comment); ``zero_points`` (if present) is also
    row-sliced directly when unpacked, or unpacked/row-sliced/re-packed
    when packed (whole-row-safe, no nibble-parity hazard for THIS axis --
    see this section's own top comment -- but still requires re-deriving
    the packed width from the correct row count, not a raw byte-slice).
    Updates the node's own ``N`` attribute to ``len(keep)`` to keep it
    consistent with the now-smaller tensors.
    """
    b = onnx.numpy_helper.to_array(w.b_init)
    w.b_init.CopyFrom(onnx.numpy_helper.from_array(b[keep], name=w.b_init.name))

    scales = onnx.numpy_helper.to_array(w.scales_init)
    w.scales_init.CopyFrom(
        onnx.numpy_helper.from_array(scales[keep], name=w.scales_init.name)
    )

    if w.zero_points_init is not None:
        zp = onnx.numpy_helper.to_array(w.zero_points_init)
        if w.zero_points_packed:
            unpacked = _unpack_nbits_codes(zp, w.k_blocks, w.bits)[keep]
            repacked = _pack_nbits_codes(unpacked, w.bits)
            w.zero_points_init.CopyFrom(
                onnx.numpy_helper.from_array(repacked, name=w.zero_points_init.name)
            )
        else:
            w.zero_points_init.CopyFrom(
                onnx.numpy_helper.from_array(zp[keep], name=w.zero_points_init.name)
            )

    if w.bias_init is not None:
        bias = onnx.numpy_helper.to_array(w.bias_init)
        w.bias_init.CopyFrom(
            onnx.numpy_helper.from_array(bias[keep], name=w.bias_init.name)
        )

    _set_matmul_nbits_int_attr(w.node, "N", len(keep))


def _matmul_nbits_block_aligned_keep_blocks(
    keep: np.ndarray, k_blocks: int, block_size: int
) -> Optional[np.ndarray]:
    """Returns the ascending block indices (into the consumer's own
    ``k_blocks``-sized block axis) that `keep` (ascending element indices
    into its ``K``-length input axis) corresponds to when every
    `block_size`-sized block is either wholly present in `keep` or wholly
    absent -- or ``None`` when some block is only partially kept, meaning
    this consumer cannot be safely pruned to this exact `keep` set at all
    (see this section's own top comment: an individual K-column can't be
    dropped without re-quantizing its whole block, out of scope). Since
    :func:`_match_matmul_nbits` already declines any ``K`` not an exact
    multiple of `block_size`, every block here is full-width -- there is
    no partial *final* block to special-case.
    """
    keep_set = set(int(k) for k in keep)
    keep_blocks = []
    for kb in range(k_blocks):
        k0 = kb * block_size
        block_positions = set(range(k0, k0 + block_size))
        overlap = block_positions & keep_set
        if overlap == block_positions:
            keep_blocks.append(kb)
        elif overlap:
            return None  # partial block -- not block-aligned, decline
    return np.array(keep_blocks, dtype=np.int64)


def _slice_matmul_nbits_consumer_blocks(
    w: _MatMulNBitsWeight, keep_blocks: np.ndarray
) -> None:
    """Drops entire ``k_blocks``-axis blocks NOT in `keep_blocks`
    (ascending block indices) from `w`'s own ``B``/``scales``/
    ``zero_points`` -- the consumer role. Never invoked with a
    non-block-aligned `keep_blocks`; see
    :func:`_matmul_nbits_block_aligned_keep_blocks`, which computes and
    validates it before this is called. Unlike the producer-side row-slice
    above, a packed ``zero_points`` genuinely must be unpacked/re-sliced/
    re-packed here (never a raw byte-slice): the block axis IS the packed
    axis for `zero_points` (unlike `B`, whose own packing is along
    `block_size`, an entirely different, untouched-by-this-slice axis), so
    for ``bits=4`` (nibble-packed) dropping an odd number of blocks from
    the middle shifts every subsequent kept block's nibble parity exactly
    the way the producer-side row-slice's own docstring describes for its
    axis -- :func:`_pack_nbits_codes`/:func:`_unpack_nbits_codes` handle
    this correctly regardless. For ``bits=8`` this repack is a byte
    identity (see this section's own top comment), so there is no such
    hazard for that case, but the same code path is used either way.
    Updates the node's own ``K`` attribute to ``len(keep_blocks) *
    block_size``.
    """
    b = onnx.numpy_helper.to_array(w.b_init)
    w.b_init.CopyFrom(
        onnx.numpy_helper.from_array(b[:, keep_blocks, :], name=w.b_init.name)
    )

    scales = onnx.numpy_helper.to_array(w.scales_init)
    w.scales_init.CopyFrom(
        onnx.numpy_helper.from_array(scales[:, keep_blocks], name=w.scales_init.name)
    )

    if w.zero_points_init is not None:
        zp = onnx.numpy_helper.to_array(w.zero_points_init)
        if w.zero_points_packed:
            unpacked = _unpack_nbits_codes(zp, w.k_blocks, w.bits)[:, keep_blocks]
            repacked = _pack_nbits_codes(unpacked, w.bits)
            w.zero_points_init.CopyFrom(
                onnx.numpy_helper.from_array(repacked, name=w.zero_points_init.name)
            )
        else:
            w.zero_points_init.CopyFrom(
                onnx.numpy_helper.from_array(
                    zp[:, keep_blocks], name=w.zero_points_init.name
                )
            )

    _set_matmul_nbits_int_attr(w.node, "K", len(keep_blocks) * w.block_size)


@dataclass(frozen=True)
class _PlainMatMulNBitsPeer:
    """A plain-float ``MatMul``/vanilla-``Gemm`` node's own weight, matched
    (:func:`_match_plain_matmul_nbits_peer`) as the OTHER side of a MIXED
    ``MatMulNBits``/plain-float chain -- see this section's own top comment
    for the real export shape this covers (an unquantized embedding/
    ``lm_head`` layer next to ``MatMulNBits``-quantized transformer-block
    layers). The narrower, MatMulNBits-section-local analogue of the QDQ
    section's own :class:`_WeightRef`: deliberately supports ONLY a
    directly-constant float weight, never a QDQ one (mixing a QDQ scheme
    with ``MatMulNBits`` in one chain remains out of scope, see that
    comment), so this is not a call into :func:`_resolve_weight_ref` at
    all -- the same "self-contained, not sharing code with the QDQ/plain
    machinery" rationale this section's own top comment gives for
    :class:`_MatMulNBitsWeight` itself.
    """

    node: onnx.NodeProto
    w_init: onnx.TensorProto
    weight_transposed: bool
    bias_init: Optional[onnx.TensorProto]
    out_channels: int
    in_channels: int


_MatMulNBitsChainSide = Union[_MatMulNBitsWeight, _PlainMatMulNBitsPeer]


def _match_plain_matmul_nbits_peer(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
) -> Optional[_PlainMatMulNBitsPeer]:
    """If `node` is a MatMul/vanilla-Gemm (:func:`_match_matmul_like`) whose
    weight is a directly-constant float32/float16/bfloat16 rank-2
    initializer (never a QDQ-fed one -- this section never calls
    :func:`_resolve_weight_ref`, see :class:`_PlainMatMulNBitsPeer`'s own
    docstring), returns the match. Mirrors :func:`_match_matmul_qdq`'s own
    float-only path (and, by extension, :func:`_match_conv_qdq`'s shared/
    tied-tensor bar), restricted to plain float only -- no QDQ dispatch,
    since this section's mixed-chain support pairs ``MatMulNBits`` with
    plain float ONLY.
    """
    match = _match_matmul_like(node)
    if match is None:
        return None
    _x_name, w_name, weight_transposed = match
    w_init = initializer_map.get(w_name)
    if w_init is None or not _is_supported_float_dtype(w_init.data_type):
        return None  # absent, non-constant, or QDQ-fed -- not this section's job
    dims = tuple(w_init.dims)
    if len(dims) != 2:
        return None
    axis = 0 if weight_transposed else 1
    out_channels, in_channels = dims[axis], dims[1 - axis]

    bias_init = None
    bias_name = None
    if node.op_type == "Gemm" and len(node.input) == 3 and node.input[2]:
        bias_name = node.input[2]
        bias_init = initializer_map.get(bias_name)
        if bias_init is None or not _is_supported_float_dtype(bias_init.data_type):
            return None  # non-constant bias -- can't safely slice it
        if list(bias_init.dims) != [out_channels]:
            return None

    for nm in (w_name,) + ((bias_name,) if bias_name else ()):
        if len(consumers_of.get(nm, [])) != 1:
            return None  # shared/tied tensor -- another node reads it too

    return _PlainMatMulNBitsPeer(
        node=node,
        w_init=w_init,
        weight_transposed=weight_transposed,
        bias_init=bias_init,
        out_channels=out_channels,
        in_channels=in_channels,
    )


def _matmul_nbits_chain_side_key(side: _MatMulNBitsChainSide) -> str:
    """A name uniquely identifying the underlying weight tensor `side`
    resolves to -- ``b_init``'s own name for a ``MatMulNBits`` side,
    ``w_init``'s own name for a plain-float peer -- used by
    :func:`apply_structured_pruning_matmul_nbits` to detect a shared/tied
    weight playing the same role (producer or consumer) in more than one
    chain. Mirrors :func:`_weight_ref_key`.
    """
    if isinstance(side, _MatMulNBitsWeight):
        return side.b_init.name
    return side.w_init.name


def _matmul_nbits_chain_producer_weight_nk(side: _MatMulNBitsChainSide) -> np.ndarray:
    """``[N, K]`` float64 view of one chain PRODUCER's own weight, for
    IMPORTANCE RANKING ONLY (:func:`_qdq_channel_importance`) -- never
    written back to the graph. A ``MatMulNBits`` side dequantizes via
    :func:`_matmul_nbits_dequantized`; a plain-float peer is read directly,
    transposed only when NOT already stored ``[N, K]`` (mirrors
    :func:`_producer_weight_nk`'s own MatMul/Gemm convention). Mirrors
    :func:`_weight_ref_dequantized`.
    """
    if isinstance(side, _MatMulNBitsWeight):
        return _matmul_nbits_dequantized(side)
    w = _to_f64(side.w_init)
    return w if side.weight_transposed else w.T


def _slice_matmul_nbits_chain_producer(
    side: _MatMulNBitsChainSide, keep: np.ndarray
) -> None:
    """Slices one chain PRODUCER's own output channels to `keep` (ascending
    indices) -- dispatches to :func:`_slice_matmul_nbits_producer_rows` for
    a ``MatMulNBits`` side, or a direct :func:`_slice_producer_weight` (plus
    its own bias, if present) for a plain-float peer. Mirrors
    :func:`_slice_producer_weight_qdq`.
    """
    if isinstance(side, _MatMulNBitsWeight):
        _slice_matmul_nbits_producer_rows(side, keep)
        return
    _slice_producer_weight(side.w_init, side.weight_transposed, keep, is_conv=False)
    if side.bias_init is not None:
        bias = onnx.numpy_helper.to_array(side.bias_init)
        side.bias_init.CopyFrom(
            onnx.numpy_helper.from_array(bias[keep], name=side.bias_init.name)
        )


def _slice_matmul_nbits_chain_consumer(
    side: _MatMulNBitsChainSide, keep: np.ndarray
) -> None:
    """Slices one chain CONSUMER's own input channels to `keep` -- a
    ``MatMulNBits`` side requires `keep` to already be whole, block-aligned
    BLOCK indices (see :func:`_matmul_nbits_block_aligned_keep_blocks`,
    checked by the caller before this is ever invoked for that side) and
    dispatches to :func:`_slice_matmul_nbits_consumer_blocks`; a plain-float
    peer has no block structure at all, so `keep` there is ordinary element
    indices, dispatched straight to :func:`_slice_consumer_weight`. Mirrors
    :func:`_slice_consumer_weight_qdq`.
    """
    if isinstance(side, _MatMulNBitsWeight):
        _slice_matmul_nbits_consumer_blocks(side, keep)
        return
    _slice_consumer_weight(side.w_init, side.weight_transposed, keep, is_conv=False)


@dataclass(frozen=True)
class _MatMulNBitsChain:
    producer: _MatMulNBitsChainSide
    chain_ops: Tuple[onnx.NodeProto, ...]
    consumer: _MatMulNBitsChainSide
    n_channels: int


def _walk_to_matmul_nbits_consumer(
    start: str,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    n_channels: int,
    max_hops: int,
) -> Optional[Tuple[_MatMulNBitsChainSide, Tuple[onnx.NodeProto, ...]]]:
    """From tensor `start` (a ``MatMulNBits`` OR plain-float MatMul/Gemm
    producer's own output), walks forward through shape-preserving unary
    activations (`_UNARY_PASS_THROUGH`) with no other consumer anywhere
    along the way, until EITHER a ``MatMulNBits`` consumer OR a plain-float
    MatMul/vanilla-Gemm consumer (:func:`_match_plain_matmul_nbits_peer`)
    is found whose input-channel count matches `n_channels` -- the
    ``MatMulNBits``/plain-float union this section's own top comment
    describes, the analogue of :func:`_walk_to_consumer_qdq`'s own float/
    QDQ union but restricted to ``MatMulNBits``/plain-float (never QDQ).
    No gated pair, no branch. Returns ``None`` if the walk runs out of
    hops, hits a branch, or never reaches such a consumer. The caller
    (:func:`_find_matmul_nbits_chains`) is responsible for discarding a
    plain-float-to-plain-float result -- that pairing is
    :func:`apply_structured_pruning`'s own job, not duplicated here.
    """
    chain_ops: List[onnx.NodeProto] = []
    cur = start
    for _hop in range(max_hops):
        candidates = consumers_of.get(cur, [])
        if len(candidates) != 1:
            return None
        nxt = candidates[0]

        if nxt.op_type == "MatMulNBits" and nxt.input[0] == cur:
            w = _match_matmul_nbits(nxt, initializer_map, consumers_of)
            if w is None or w.K != n_channels:
                return None
            return w, tuple(chain_ops)

        mm = _match_matmul_like(nxt)
        if mm is not None and mm[0] == cur:
            peer = _match_plain_matmul_nbits_peer(nxt, initializer_map, consumers_of)
            if peer is None or peer.in_channels != n_channels:
                return None
            return peer, tuple(chain_ops)

        if not (
            nxt.op_type in _UNARY_PASS_THROUGH
            and list(nxt.input) == [cur]
            and len(nxt.output) == 1
        ):
            return None
        out2 = nxt.output[0]
        if len(consumers_of.get(out2, [])) != 1 or out2 in graph_outputs:
            return None
        chain_ops.append(nxt)
        cur = out2
    return None


def _find_matmul_nbits_chains(graph: onnx.GraphProto) -> List[_MatMulNBitsChain]:
    """The ``MatMulNBits`` analogue of :func:`_find_qdq_chains`: every
    producer/consumer pair connected by :func:`_walk_to_matmul_nbits_consumer`
    where AT LEAST ONE side is a ``MatMulNBits`` node (a plain-float-to-
    plain-float pair is :func:`apply_structured_pruning`'s own job, not
    duplicated here -- mirrors :func:`_find_qdq_chains`'s identical
    at-least-one-quantized filter).
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains: List[_MatMulNBitsChain] = []
    for node in graph.node:
        producer: _MatMulNBitsChainSide
        n_channels: int
        if node.op_type == "MatMulNBits":
            w = _match_matmul_nbits(node, initializer_map, consumers_of)
            if w is None:
                continue
            producer, n_channels = w, w.N
        else:
            peer = _match_plain_matmul_nbits_peer(node, initializer_map, consumers_of)
            if peer is None:
                continue
            producer, n_channels = peer, peer.out_channels

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue
        found = _walk_to_matmul_nbits_consumer(
            out_name,
            initializer_map,
            consumers_of,
            graph_outputs,
            n_channels,
            _MAX_CHAIN_HOPS,
        )
        if found is None:
            continue
        consumer, chain_ops = found
        if isinstance(producer, _PlainMatMulNBitsPeer) and isinstance(
            consumer, _PlainMatMulNBitsPeer
        ):
            continue  # both plain float -- apply_structured_pruning's own job
        chains.append(_MatMulNBitsChain(producer, chain_ops, consumer, n_channels))
    return chains


def apply_structured_pruning_matmul_nbits(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
) -> onnx.ModelProto:
    """Removes whole output channels from a ``com.microsoft::MatMulNBits``
    node (ONNX Runtime GenAI Model Builder's block-quantized, weight-only
    int4/int8 Linear-layer op -- see this module's "MatMulNBits (block-
    quantized weight) structured pruning" section comment for the full
    empirical schema/packing investigation this scope was reached from)
    whose output feeds, through zero or more shape-preserving unary
    activations (`_UNARY_PASS_THROUGH`) with no other consumer anywhere
    along that path, into exactly one downstream consumer whose input
    channel count matches -- either another ``MatMulNBits`` node, or a
    plain-float (directly-constant weight) ``MatMul``/vanilla-``Gemm``
    (and symmetrically, a plain-float producer feeding a ``MatMulNBits``
    consumer). At least one side of every matched chain is always a
    ``MatMulNBits`` node; a plain-float-to-plain-float pair is
    :func:`apply_structured_pruning`'s own job, not duplicated here.

    Ranks the producer's output channels by L1/L2 norm of their own
    (dequantized, for a ``MatMulNBits`` producer) weight row
    (:func:`_matmul_nbits_dequantized`/:func:`_qdq_channel_importance` --
    dequantized for ranking only, per this module's "slice, don't
    recompute" principle; the actual rewrite always slices the existing
    int4/int8 codes/scale/zero-point directly, never re-quantizes), drops
    the lowest-``sparsity``-fraction of them, and -- only when the
    CONSUMER side is itself a ``MatMulNBits`` node -- checks whether that
    keep-set happens to align to the consumer's own ``block_size``
    boundaries (every block wholly kept or wholly dropped, since a
    block's own K-values are quantized together and can't be partially
    dropped without re-quantizing -- out of scope, this module never
    invents new quantized values); a plain-float consumer has no such
    block structure, so any keep-set applies directly. When aligned (or
    when the consumer is plain-float), slices the producer's own
    ``N``-axis (any row subset, or the plain-float equivalent) and the
    consumer's own ``K``-axis (whole aligned blocks only for a
    ``MatMulNBits`` consumer, any column subset for a plain-float one)
    together, in lockstep. When NOT block-aligned for a ``MatMulNBits``
    consumer, that chain (both producer and consumer) is left completely
    untouched rather than forcing a partial-block re-quantization or a
    disagreeing keep-set between the two.

    See this module's section comment for the full list of scope
    boundaries: ``bits`` restricted to ``{4, 8}`` (empirically verified;
    ``bits=2`` remains out of scope), ``weight_prepacked`` and ``g_idx``
    always declined, a QDQ-quantized weight on the other side of a mixed
    chain remains out of scope (unlike
    :func:`apply_structured_pruning_qdq`'s own float/QDQ mixing -- this
    section mixes ``MatMulNBits`` with plain float ONLY), no grouped/
    gated/residual/Concat-merge topology, a padded partial final block
    declined, a shared/tied operand declined.

    :param model: onnx ModelProto object or file path
    :param sparsity: fraction of each eligible producer's output channels to
            drop (rounded, at least one channel is always kept)
    :param importance_norm: ``"l2"`` (default, Li et al.'s original
            filter-pruning criterion) or ``"l1"``
    :returns: the pruned onnx ModelProto
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = _find_matmul_nbits_chains(graph)
    if not chains:
        return out

    producer_touched: Set[str] = set()
    consumer_touched: Set[str] = set()
    stale_value_info: Set[str] = set()

    for chain in chains:
        p, c = chain.producer, chain.consumer
        p_key = _matmul_nbits_chain_side_key(p)
        c_key = _matmul_nbits_chain_side_key(c)
        if p_key == c_key:
            continue  # degenerate (the same weight in both roles)
        if p_key in producer_touched or c_key in consumer_touched:
            continue  # a shared/tied weight another chain already resized

        n = chain.n_channels
        keep_count = max(1, n - round(n * sparsity))
        if keep_count >= n:
            continue  # rounds down to nothing for this layer -- no-op

        w_nk = _matmul_nbits_chain_producer_weight_nk(p)
        importance = _qdq_channel_importance(w_nk, importance_norm)
        # `kind="stable"` matters here specifically (unlike most other
        # argsort call sites in this module, which never feed into a
        # block-alignment check): with an exactly-tied `importance` vector,
        # an unstable sort's tie-break order is platform-dependent
        # (confirmed: reordered on an ARM CI runner vs. x86_64), and a
        # reordered tie can select a keep-set that no longer aligns to the
        # consumer's own block boundaries below -- flipping a
        # would-have-been-aligned chain to "declined" nondeterministically,
        # and diverging from this section's own `_analyze_*` dry-run mirror
        # (which must make the identical selection for its predictions to
        # stay trustworthy). A stable sort preserves input (channel index)
        # order for ties, so an all-tied producer's top-`keep_count`
        # channels are always its first `keep_count` indices, deterministic
        # across platforms.
        keep = np.sort(np.argsort(-importance, kind="stable")[:keep_count])

        if isinstance(c, _MatMulNBitsWeight):
            keep_blocks = _matmul_nbits_block_aligned_keep_blocks(
                keep, c.k_blocks, c.block_size
            )
            if keep_blocks is None:
                continue  # non-block-aligned request for this consumer --
                # decline, see this section's own top comment
            consumer_keep = keep_blocks
        else:
            consumer_keep = keep  # plain-float consumer -- no block structure

        _slice_matmul_nbits_chain_producer(p, keep)
        _slice_matmul_nbits_chain_consumer(c, consumer_keep)

        producer_touched.add(p_key)
        consumer_touched.add(c_key)
        stale_value_info.add(p.node.output[0])
        stale_value_info.update(op.output[0] for op in chain.chain_ops)

    if stale_value_info:
        kept = [vi for vi in graph.value_info if vi.name not in stale_value_info]
        del graph.value_info[:]
        graph.value_info.extend(kept)
    return out


# --- Attention-head pruning -----------------------------------------------

# Three fused self-attention ops are matched here -- two from the
# ``com.microsoft`` domain, produced by onnxsim's own fusion passes from a
# decomposed self-attention block, plus the standard ``ai.onnx`` op -- each
# pruned at the granularity its own kernel contract allows:
#
# - `Attention` (onnxsim/passes/fuse_attention.h): a single merged QKV
#   weight/bias ([hidden_size, Nq+Nk+Nv] / [Nq+Nk+Nv]) plus
#   `num_heads`/`qkv_hidden_sizes` attributes, one `num_heads` shared by
#   Q/K/V alike. Every head owns an equally-sized, independent column block
#   of that merged weight, so individual heads can be dropped one at a time
#   -- see :func:`_apply_one_plain_attention_chain`. This op's own
#   contrib-op schema also gives it two unrelated optional mask-shaped
#   inputs: `mask_index`, none of whose several documented shapes ever
#   carry a `num_heads`-sized axis (confirmed via live schema
#   introspection -- see :func:`_match_attention_producer`'s own
#   docstring), so it is always left alone untouched; and `attention_bias`,
#   which *does* (shape `(batch_size or 1, num_heads or 1, sequence_length,
#   total_sequence_length)`, confirmed to have a real, non-ignored numeric
#   effect via actual onnxruntime execution) -- a constant one is sliced
#   along its own head axis by the same kept-head index set whenever that
#   axis is genuinely `num_heads`-sized (:func:`_head_bias_axis`), left
#   alone when it's a broadcast or dynamic, and declines the whole match
#   when its shape resolves to neither. This was a real gap, not merely an
#   overly conservative decline: an earlier version of this matcher never
#   inspected `attention_bias` at all, so a model carrying one would have
#   been pruned with a now-stale, wrong-head-count bias silently left
#   behind -- a genuine correctness bug, not just a missed optimization
#   (see ``tests/test_pruning.py``'s own
#   ``test_attention_head_pruning_attention_bias_is_sliced_and_matches_oracle``,
#   which fails against the pre-fix matcher/slicer).
# - `GroupQueryAttention` (onnxsim/passes/fuse_gqa.h): separate, un-merged
#   Q/K/V projections (ordinary MatMul/vanilla-Gemm nodes feeding directly
#   into the op, not weights the op itself owns) plus independent
#   `num_heads` (query heads)/`kv_num_heads` (key/value heads) attributes,
#   `num_heads` a positive multiple of `kv_num_heads`. A contiguous *group*
#   of `num_heads / kv_num_heads` query heads shares each KV head via the
#   kernel's own internal broadcast -- GQA's real-world purpose is fewer KV
#   heads than query heads, exactly the shape Llama 2/3, Mistral, Qwen, and
#   most current open-weight models export. Because every surviving KV head
#   must keep exactly the same number of query heads mapped to it (the
#   kernel requires `num_heads % kv_num_heads == 0` after pruning just as
#   before), an individual query head cannot be dropped in isolation the
#   way plain `Attention` pruning does -- only a *whole KV group* (that KV
#   head's own K/V column block, together with every query head mapped to
#   it) is ever removed at once, ranked by the combined importance of the
#   group's whole Q+K+V block -- see :func:`_apply_one_gqa_chain` and
#   :func:`_gqa_group_importance`. A connected `past_key`/`past_value` that
#   is a constant of the expected BNSH shape (see
#   :func:`_past_kv_constants_are_sliceable`) is sliced along its own
#   `kv_num_heads` axis by the same `keep_groups` index set used for K's/V's
#   own producer weights -- for a plain FLOAT cache unconditionally, and for
#   a *quantized* one (`float8e4m3fn`/`uint8`/`int8`, which this op's own
#   schema allows specifically to shrink KV-cache memory) together with its
#   own `k_scale`/`v_scale` tensor, sliced the identical way when that scale
#   is itself a constant of the schema's `"PER_CHANNEL"` shape (left alone,
#   needing no slicing at all, when `"PER_TENSOR"` -- a single broadcast
#   scalar with no per-head axis); a cache of any other shape/dtype, a
#   quantized cache with no scale connected or one of an unrecognized shape,
#   or a packed-QKV/missing-required-input node this module cannot prove
#   safe to leave alone, is declined outright -- see
#   :func:`_match_gqa_producer`. That "packed-QKV" decline is
#   `GroupQueryAttention`'s own *schema-level* packed-input convention (the
#   whole packed tensor passed as `query` itself, `key`/`value` left
#   empty -- confirmed via live schema introspection,
#   `onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()`:
#   `query`'s own doc string reads "Query with shape (batch_size,
#   sequence_length, hidden_size), or packed QKV with shape (batch_size,
#   sequence_length, d)"), a different tensor layout from the *graph-level*
#   packed-QKV-then-Split pattern this module does support: one packed
#   MatMul/vanilla-Gemm projection feeding a `Split` whose three outputs
#   feed `GroupQueryAttention`'s three separate, still non-empty,
#   query/key/value inputs directly -- confirmed to be a real export
#   pattern (Microsoft's own onnxruntime-genai model builder's fused
#   Q/K-norm GQA path, e.g. Qwen3-style models on CUDA/WebGPU) rather than
#   assumed -- see :func:`_match_packed_qkv_split`'s own docstring for the
#   exact topology matched and the export code path that produces it. The
#   CPU/DirectML variant of that same export path -- Q/K-norm requested but
#   the fused in-op norm path unsupported there, so a per-head
#   `SimplifiedLayerNorm` (and, unless RoPE is fused into the op itself, a
#   `RotaryEmbedding`) sits between the `Split` and the op's own Q/K
#   inputs -- is *also* matched, via :func:`_walk_back_through_qk_norm_rope`
#   (see that function's own docstring for the exact topology and its own
#   confirmed head-independence: neither the norm's own `scale` nor
#   `RotaryEmbedding`'s `cos_cache`/`sin_cache` ever needs slicing, only a
#   crossed `RotaryEmbedding`'s own `num_heads` attribute and a crossed
#   `Reshape`'s own target width, both handled by
#   :func:`_apply_one_gqa_chain`). Only a shape genuinely outside both of
#   these -- an unrecognized op in the way, RoPE applied *before* norm, or
#   no `Split` at all upstream of Q/K -- is still deliberately left
#   unmatched. Pruning a KV group out of a packed
#   chain removes that group's own Q/K/V column ranges from the *one*
#   shared packed weight (and its packed bias, if any) in a single combined
#   slice, and shrinks the `Split` node's own split-sizes constant to
#   match -- see :class:`_GQAChain`'s own `packed_split_sizes` field and
#   :func:`_apply_one_gqa_chain`'s own packed branch. Two more optional
#   inputs carry a genuine per-*query*-head axis and were, like the plain
#   `Attention` op's own `attention_bias` above, a real unhandled gap this
#   pass now closes: `attention_bias` (same broadcastable shape/treatment
#   as `Attention`'s own, but against `num_heads` meaning *query* heads
#   here, added after GQA's own internal KV-repeat -- sliced by
#   `keep_q_heads`, not `keep_groups`) and `head_sink` (a genuine
#   `(num_heads,)` one-scalar-per-query-head softmax-smoothing constant,
#   sliced directly by `keep_q_heads`) -- both confirmed to have a real,
#   non-ignored numeric effect via actual onnxruntime execution, not
#   assumed from either input's own doc string alone.
# - the plain ``ai.onnx`` `Attention` (opset 24+, domain ``""``, schema
#   confirmed against this environment's installed ``onnx==1.22.0`` via
#   ``onnx.defs.get_schema("Attention", domain="")`` -- it is fully defined
#   there, unlike the still-under-development op ``fuse_attention.h``'s own
#   comment warns about for a different opset/op; see
#   :func:`_match_onnx_attention_producer`'s own docstring for the exact
#   attributes/inputs read off that schema): structurally the same shape as
#   `GroupQueryAttention` -- separate, un-merged Q/K/V projections plus
#   independent `q_num_heads`/`kv_num_heads` attributes, `q_num_heads` a
#   positive multiple of `kv_num_heads` (the op's schema doc names this same
#   MHA/GQA/MQA taxonomy explicitly) -- close enough a cousin that this pass
#   reuses :class:`_GQAChain`, :func:`_apply_one_gqa_chain`, and
#   :func:`_gqa_group_importance` for it outright rather than a parallel
#   implementation (see :func:`_find_separate_qkv_chains`, the two matchers'
#   shared caller). It differs in three ways this pass accounts for: (1) its
#   query-head-count attribute is named `q_num_heads`, not `num_heads` --
#   :class:`_GQAChain` carries which attribute name to write back
#   (`num_heads_attr`); (2) it schema-allows V its own `head_size`
#   independent of Q/K's (confirmed via the op's own backend test suite,
#   e.g. ``test_attention_3d_diff_heads_sizes``), which `GroupQueryAttention`
#   itself can never have (`fuse_gqa.h` requires equal Q/K/V head_size before
#   it will even fuse a node) -- :class:`_GQAChain` carries Q's/K's shared
#   `head_size` and V's own (possibly different) `v_head_size` as two
#   separate fields, and :func:`_apply_one_gqa_chain`'s shared slicing uses
#   whichever of the two actually applies to each tensor it touches (see
#   that function's own docstring and :func:`_find_separate_qkv_chains`'s
#   own `allow_differing_v_head_size` parameter, `False` for
#   `GroupQueryAttention`, `True` here); (3) its optional `attn_mask` input
#   is added against the `(batch, q_num_heads, q_seq, kv_seq)` attention-
#   score tensor via ordinary broadcasting -- a constant one is sliced by
#   `keep_q_heads` when its own head axis (resolved by
#   :func:`_head_bias_axis`, which also accounts for a lower-rank mask's
#   axis genuinely landing on that same broadcast-target position -- see
#   its own docstring) is exactly `q_num_heads`-sized, left untouched when
#   it's a broadcast (absent or size-1), and declines the whole match only
#   when the shape resolves to neither (narrower than an earlier version of
#   this pass, which declined any non-empty constant mask outright
#   regardless of shape); a *dynamic* one is always left alone and never
#   blocks the match, while its `past_key`/`past_value` (a different pair
#   of input indices from
#   `GroupQueryAttention`'s own `past_key`/`past_value`/`seqlens_k`/
#   `total_sequence_length`/`cos_cache`/`sin_cache`) share
#   :func:`_match_gqa_producer`'s own updated treatment via the same
#   :func:`_past_kv_constants_are_sliceable` -- see
#   :func:`_match_onnx_attention_producer`. Verified here via actual
#   execution (``onnx.checker`` plus onnxruntime, both of which handle this
#   op in this environment -- see ``tests/test_pruning.py``'s own "plain
#   ai.onnx Attention" section), the same oracle-vs-onnxruntime bar every
#   other function in this module is held to; no structural-only fallback
#   was needed.
#
# Cross-attention (Q projected from one source tensor, K/V from a genuinely
# *different* one -- the encoder-decoder shape) was investigated explicitly
# for all three matched op types, not left an untested assumption:
#
# - `com.microsoft::Attention`'s own contrib-op schema (`bert_defs.cc`, the
#   same source consulted elsewhere in this module for `SkipLayerNormalization`)
#   takes a single `input` tensor that its one merged weight projects to
#   Q, K, *and* V alike -- there is no second, encoder-side input for K/V
#   to come from at all. Cross-attention isn't a shape this op's schema can
#   express, so it's simply not applicable here -- not a gap in
#   :func:`_match_attention_producer` to close.
# - `GroupQueryAttention` and the plain ``ai.onnx`` `Attention` op both
#   already support it, confirmed by construction and by execution: neither
#   :func:`_match_gqa_producer`/:func:`_match_onnx_attention_producer` nor
#   :func:`_find_separate_qkv_chains` (their shared caller) ever compares
#   Q's own producer against K's or V's own -- each of the three is matched,
#   via :func:`_match_producer`, purely from its own MatMul/vanilla-Gemm
#   node and its own weight, with no check tying it to where the *other two*
#   ultimately trace back to. A model where Q's producer reads from one
#   graph input and K/V's producers read from an entirely different one (a
#   real decoder/encoder pair, potentially different feature dimensions
#   too) matches exactly the same way a self-attention model does. Both
#   ops' own schema docs name this explicitly -- `GroupQueryAttention`'s
#   own doc string opens with "Group Query **Self/Cross** Attention", and
#   the plain op's doc (`onnx.defs.get_schema("Attention", domain="")` on
#   this environment's installed ``onnx==1.22.0``) states outright "this
#   operator covers self and cross variants ... For cross attention, query
#   and key might have different lengths" -- and this is verified here the
#   same oracle-vs-onnxruntime way as everything else (see
#   ``tests/test_pruning.py``'s own "cross-attention" subsections under the
#   GroupQueryAttention/plain-Attention sections), with distinct source
#   tensors of distinct feature dimensions feeding Q vs K/V.
# - One real bug turned up along the way and is fixed here:
#   :func:`_gqa_group_importance`'s combined-importance score used to
#   ``np.concatenate`` each group's Q, K, and V weight *blocks* into one
#   matrix before taking a single Frobenius norm -- silently assuming Q's
#   own producer weight has the same row count (its source tensor's own
#   feature dimension) as K/V's own, true for every self-attention shape
#   but not guaranteed once Q and K/V read from different source tensors of
#   different widths. On such a model it didn't mis-rank -- it raised a
#   bare ``ValueError`` from numpy and crashed the whole pruning call. Fixed
#   to combine each block's own Frobenius norm via
#   ``sqrt(sum of squares)`` instead of concatenating first -- numerically
#   identical to the old formula whenever the concatenation was even legal
#   (``||[A B]||_F^2 == ||A||_F^2 + ||B||_F^2``), well-defined when it isn't
#   -- see that function's own updated comment, and
#   ``test_gqa_pruning_cross_attention_matches_oracle_exactly``/
#   ``test_onnx_attention_pruning_cross_attention_matches_oracle_exactly``
#   in ``tests/test_pruning.py`` (both fail with that bare ``ValueError``
#   without this fix).
# - The Wanda-calibrated variant's own activation probe
#   (:func:`apply_attention_head_wanda_pruning`) sits at
#   `chain.consumer_node.input[0]` -- the *single* tensor downstream of the
#   matched op's own output projection, after Q/K/V have already been
#   reduced to one attention output -- never at Q's or K/V's own activation
#   directly, so it needs no per-source calibration-data key of its own and
#   is unaffected by whether Q and K/V trace back to the same graph input or
#   two different ones; a calibration batch dict simply needs an entry for
#   every graph input the model actually has (both the decoder- and
#   encoder-side ones for cross-attention), exactly as
#   :func:`onnxsim.generate_random_calibration_data` already produces.
# - `GroupQueryAttention`'s own real-world calling convention does impose
#   one genuine restriction beyond this module's control: this environment's
#   onnxruntime (1.29.0) CPU kernel requires Q's and K/V's sequence length
#   to match unless a non-empty `past_key`/`past_value` is also supplied --
#   confirmed empirically, not merely read off the schema doc, which
#   promises cross-attention support without mentioning this. A non-empty
#   constant `past_key`/`past_value` is, since the KV-cache-slicing fix
#   above, no longer declined outright purely for being non-empty (see
#   :func:`_past_kv_constants_are_sliceable`) -- but this module's own
#   cross-attention support was verified only for the ordinary
#   ``past_key``/``past_value``-omitted case (see the tests referenced
#   above); a single-call cross-attention model additionally relying on a
#   non-empty `past_key`/`past_value` specifically to satisfy this
#   onnxruntime-kernel-level sequence-length restriction was not
#   separately re-verified and remains untested here -- an
#   onnxruntime-kernel-level restriction on the op itself either way, not a
#   limitation this module's own matching or pruning logic adds. The plain ``ai.onnx``
#   `Attention` op has no such restriction (also confirmed empirically): its
#   cross-attention test below uses genuinely different Q/K-V sequence
#   lengths (as well as different source tensors and different feature
#   dimensions) throughout, oracle-verified via onnxruntime with no
#   restriction of this pass's own.
_ATTENTION_DOMAIN = "com.microsoft"

# The three quantized-cache dtypes `GroupQueryAttention`'s own `T_CACHE` type
# constraint allows for `past_key`/`past_value` beyond plain FLOAT -- see
# :func:`_past_kv_constants_are_sliceable`'s own docstring for how this was
# confirmed directly off the installed `onnxruntime` package's schema
# registry (`float16`/`bfloat16`, `T_CACHE`'s other two members, are *not*
# quantized dtypes and stay declined by the plain FLOAT-only branch below,
# same as before this constant existed -- out of scope for this quantized-
# cache-specific extension).
_QUANTIZED_KV_CACHE_DTYPES = {
    onnx.TensorProto.UINT8,
    onnx.TensorProto.INT8,
    onnx.TensorProto.FLOAT8E4M3FN,
}


@dataclass(frozen=True)
class _AttentionChain:
    node: onnx.NodeProto
    weight: str
    bias: Optional[str]
    num_heads: int
    nq: int
    nk: int
    nv: int
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    consumer_node: onnx.NodeProto
    consumer_weight: str
    consumer_weight_transposed: bool


@dataclass(frozen=True)
class _GQAChain:
    node: onnx.NodeProto
    q_weight: str
    q_bias: Optional[str]
    q_weight_transposed: bool
    k_weight: str
    k_bias: Optional[str]
    k_weight_transposed: bool
    v_weight: str
    v_bias: Optional[str]
    v_weight_transposed: bool
    num_heads: int
    kv_num_heads: int
    head_size: int
    # V's own head_size -- equal to `head_size` for every `GroupQueryAttention`
    # chain (`fuse_gqa.h` requires `q_head_size == k_head_size == v_head_size`
    # before it will even fuse the op, confirmed by reading that requirement
    # directly off `fuse_gqa.h` itself), but can genuinely differ for a plain
    # ``ai.onnx::Attention`` chain -- that op's own schema gives V an
    # independent `v_head_size` distinct from Q/K's shared `head_size` (see
    # :func:`_match_onnx_attention_producer`'s own docstring and this
    # module's "Attention-head pruning" section comment). Every place that
    # slices/sizes something on Q's or K's own side (`.head_size`) versus
    # V's or the output-projection's own side (`.v_head_size`) needs to pick
    # the right one of these two fields -- see :func:`_apply_one_gqa_chain`.
    v_head_size: int
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]
    consumer_node: onnx.NodeProto
    consumer_weight: str
    consumer_weight_transposed: bool
    # Which attribute on `.node` holds the query head count:
    # ``com.microsoft::GroupQueryAttention`` names it `num_heads`, the plain
    # ``ai.onnx::Attention`` op (see :func:`_match_onnx_attention_producer`)
    # names the same concept `q_num_heads` -- both share `kv_num_heads`
    # verbatim, so only this one name needs to travel with the chain for
    # :func:`_apply_one_gqa_chain`'s shared write-back to target the right
    # attribute on either op.
    num_heads_attr: str = "num_heads"
    # Set (to the name of the constant int64 split-sizes tensor feeding a
    # shared upstream ``Split`` node) when Q/K/V are *not* three independent
    # producer weights but three column ranges of one *packed* MatMul/Gemm
    # weight, split into separate tensors by a single ``Split`` node
    # upstream of this chain's `.node` -- see
    # :func:`_match_packed_qkv_split` for the exact topology matched and the
    # real-world export path (onnxruntime-genai's model builder, fused
    # Q/K-norm GQA path) that produces it. ``None`` for the ordinary
    # three-independent-producer shape every other chain has. When set,
    # `.q_weight`, `.k_weight`, and `.v_weight` all name that *same* single
    # packed initializer (and `.q_bias`/`.k_bias`/`.v_bias`, if not all
    # ``None``, all name the same single packed bias) -- :func:`_apply_one_gqa_chain`
    # branches on this field to slice that one shared tensor exactly once
    # with a combined column-index set, instead of three independent
    # per-producer slices that would each invalidate the others' column
    # offsets into the same underlying storage.
    packed_split_sizes: Optional[str] = None
    # Set only alongside `packed_split_sizes`, and only when a per-head
    # Q/K-norm + RoPE "pass-through" (see :class:`_QKNormRopePassThrough`
    # and :func:`_walk_back_through_qk_norm_rope`) was crossed walking back
    # from `.node`'s own Q (`.q_norm_rope`) or K (`.k_norm_rope`) input to
    # the shared `Split` -- the CPU/DirectML Q/K-norm GQA export shape
    # `_match_packed_qkv_split`'s own docstring names as *not* matched by
    # that function alone. ``None`` (the default) whenever that particular
    # branch feeds `.node` directly from the `Split` with no hop in
    # between -- every ordinary packed-QKV chain, matched before this
    # pass-through existed, gets ``None`` on both exactly as before.
    q_norm_rope: Optional["_QKNormRopePassThrough"] = None
    k_norm_rope: Optional["_QKNormRopePassThrough"] = None


# Either kind of matched attention block, sharing enough of a common shape
# (a `.node`, a `.consumer_node`/`.consumer_weight`, `.chain_ops`) that
# :func:`_apply_attention_chains`'s own bookkeeping (touched-role tracking,
# stale value_info cleanup) and the activation-probing setup in
# :func:`apply_attention_head_wanda_pruning` treat both uniformly, only
# dispatching on which one a given chain is for the actual slicing. A
# matched plain ``ai.onnx::Attention`` node is represented as a
# :class:`_GQAChain` too (see that class's own `num_heads_attr` field) --
# it is a *third* matched node type, not a third dataclass, since its
# separate-Q/K/V-producer shape and whole-KV-group pruning unit are
# identical to `GroupQueryAttention`'s own.
_AttnLikeChain = Union[_AttentionChain, _GQAChain]


def _match_attention_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[str, Optional[str], int, int, int, int]]:
    """If `node` is a ``com.microsoft::Attention`` node with a constant 2-D
    float32 merged QKV weight ``[K, Nq+Nk+Nv]`` (and, if present, a
    constant 1-D float32 merged bias), returns
    ``(weight_name, bias_name_or_None, num_heads, Nq, Nk, Nv)``.

    This op's own contrib-op schema (`onnxruntime.capi
    .onnxruntime_pybind11_state.get_all_operator_schema()`, confirmed live
    against this environment's installed onnxruntime) gives it two,
    unrelated optional mask-shaped inputs: `mask_index` (index 3, type
    `M`/int32) documents five possible shapes -- `(batch_size, 1,
    max_sequence_length, max_sequence_length)`, `(batch_size,
    total_sequence_length)`, `(batch_size, sequence_length,
    total_sequence_length)`, or an index of shape `(batch_size)`, `(2 *
    batch_size)`, or `(3 * batch_size + 2)` -- *none* of which carry a
    `num_heads`-sized axis, so it is unconditionally head-count-independent
    and this function (like this whole matcher) never inspects it at all,
    match or no match. `attention_bias` (index 5, type `T`, same dtype as
    the QKV weight) is different: its own doc gives it shape `(batch_size
    or 1, num_heads or 1, sequence_length, total_sequence_length)` --
    verified to have a real, functional per-head effect via actual
    onnxruntime execution, not assumed from the doc string alone (see
    ``tests/test_pruning.py``'s own "attention_bias" subsection) -- so a
    constant one is validated the same way :func:`_match_gqa_producer` and
    :func:`_match_onnx_attention_producer` validate their own analogous
    inputs, via :func:`_head_bias_axis`: declined outright when its shape
    doesn't cleanly resolve to either "genuinely per-head, safe to slice"
    or "broadcast, no per-head values at all" against this op's own
    (pre-pruning) `num_heads`; left alone (and never blocks the match) when
    dynamic, absent, or a constant that resolves to the latter --
    :func:`_apply_one_plain_attention_chain` performs the actual slice once
    a match succeeds.
    """
    if node.domain != _ATTENTION_DOMAIN or node.op_type != "Attention":
        return None
    if len(node.input) < 2:
        return None
    w_name = node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or not _is_supported_float_dtype(w_init.data_type)
        or len(w_init.dims) != 2
    ):
        return None
    total_n = w_init.dims[1]

    bias_name = None
    if len(node.input) >= 3 and node.input[2]:
        bias_name = node.input[2]
        b_init = initializer_map.get(bias_name)
        if (
            b_init is None
            or not _is_supported_float_dtype(b_init.data_type)
            or list(b_init.dims) != [total_n]
        ):
            return None

    num_heads = None
    qkv_hidden_sizes: Optional[List[int]] = None
    for attr in node.attribute:
        if attr.name == "num_heads":
            num_heads = attr.i
        elif attr.name == "qkv_hidden_sizes":
            qkv_hidden_sizes = list(attr.ints)
    if not num_heads or num_heads <= 0:
        return None

    if qkv_hidden_sizes is not None:
        if len(qkv_hidden_sizes) != 3:
            return None
        nq, nk, nv = qkv_hidden_sizes
    else:
        # Schema default: Q/K/V evenly split the merged width.
        if total_n % 3 != 0:
            return None
        nq = nk = nv = total_n // 3
    if (
        nq <= 0
        or nk <= 0
        or nv <= 0
        or nq + nk + nv != total_n
        or nq % num_heads
        or nk % num_heads
        or nv % num_heads
    ):
        return None

    if len(node.input) > 5 and node.input[5]:  # attention_bias
        bias_init = initializer_map.get(node.input[5])
        if bias_init is not None and int(np.prod(bias_init.dims)) > 0:
            if _head_bias_axis(list(bias_init.dims), num_heads) is None:
                return None  # doesn't statically resolve -- decline rather than guess

    return w_name, bias_name, num_heads, nq, nk, nv


def _reshape_last_dim(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[int]:
    """If `node` is a ``Reshape`` whose target-shape input is a constant
    int64 tensor, returns its last entry (or ``None`` if that entry is a
    wildcard/inferred ``-1`` or ``0``, or the shape can't be read at all).
    """
    if node.op_type != "Reshape" or len(node.input) != 2:
        return None
    shape_init = initializer_map.get(node.input[1])
    if shape_init is None or shape_init.data_type != onnx.TensorProto.INT64:
        return None
    dims = onnx.numpy_helper.to_array(shape_init)
    if dims.size == 0:
        return None
    last = int(dims[-1])
    return last if last > 0 else None


def _walk_to_attention_consumer(
    start: str,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    nv: int,
) -> Tuple[Optional[_ConsumerMatch], Tuple[Tuple[onnx.NodeProto, Optional[str]], ...]]:
    """From `Attention`'s raw (V-hidden-size-wide) output tensor `start`,
    optionally through a single ``Reshape`` hop whose target shape's last
    entry is provably still `nv` (the shape onnxsim's own `fuse_attention`
    pass always appends, reusing the original ``ctx`` reshape's own target
    -- see fuse_attention.h's own doc comment; a hand-authored or
    differently-sourced graph is still handled the same way as long as it
    matches this same shape), to a MatMul/vanilla-Gemm consumer (the output
    projection) whose reduction dimension matches `nv`. Declines (``None``)
    on anything else -- a branch, an activation, a mismatched Reshape --
    rather than guessing. When a Reshape hop is matched, its second (shape)
    input must be single-use too -- the caller overwrites that constant's
    last entry to the post-pruning `nv` in place, which would corrupt any
    other reader of the same tensor.
    """
    candidates = consumers_of.get(start, [])
    if len(candidates) != 1:
        return None, ()
    node = candidates[0]
    chain_ops: Tuple[Tuple[onnx.NodeProto, Optional[str]], ...] = ()
    cur = start

    if node.op_type == "Reshape" and node.input[:1] == [cur]:
        last_dim = _reshape_last_dim(node, initializer_map)
        if last_dim != nv:
            return None, ()
        shape_name = node.input[1]
        if len(consumers_of.get(shape_name, [])) != 1:
            return None, ()  # shared shape constant -- mutating it isn't safe
        out_name = node.output[0]
        if len(consumers_of.get(out_name, [])) != 1 or out_name in graph_outputs:
            return None, ()
        chain_ops = ((node, shape_name),)
        cur = out_name
        node = consumers_of[cur][0]

    cm = _match_matmul_like(node)
    if cm is None or cm[0] != cur:
        return None, chain_ops
    _, cw_name, c_weight_transposed = cm
    cw_init = initializer_map.get(cw_name)
    if (
        cw_init is None
        or not _is_supported_float_dtype(cw_init.data_type)
        or len(cw_init.dims) != 2
    ):
        return None, chain_ops
    k = cw_init.dims[1] if c_weight_transposed else cw_init.dims[0]
    if k != nv:
        return None, chain_ops
    return (node, cw_name, c_weight_transposed), chain_ops


def _find_attention_chains(graph: onnx.GraphProto) -> List[_AttentionChain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains = []
    for node in graph.node:
        info = _match_attention_producer(node, initializer_map)
        if info is None:
            continue
        w_name, bias_name, num_heads, nq, nk, nv = info

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        consumer, chain_ops = _walk_to_attention_consumer(
            out_name, initializer_map, consumers_of, graph_outputs, nv
        )
        if consumer is None:
            continue

        chains.append(
            _AttentionChain(
                node=node,
                weight=w_name,
                bias=bias_name,
                num_heads=num_heads,
                nq=nq,
                nk=nk,
                nv=nv,
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
            )
        )
    return chains


def _past_kv_constants_are_sliceable(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    indices: Tuple[int, int],
    kv_num_heads: int,
    scale_indices: Optional[Tuple[int, int]] = None,
) -> bool:
    """Shared safety gate for a matched node's optional `past_key`/
    `past_value` inputs (at `indices`, a `(past_key_idx, past_value_idx)`
    pair -- ``(3, 4)`` for `GroupQueryAttention`, ``(4, 5)`` for the plain
    ``ai.onnx::Attention`` op), used by both :func:`_match_gqa_producer` and
    :func:`_match_onnx_attention_producer`.

    Both ops' own schemas lay a connected `past_key`/`past_value` out in
    BNSH format -- ``(batch_size, kv_num_heads, past_sequence_length,
    head_size_or_v_head_size)`` (`GroupQueryAttention`'s own contrib-op
    schema doc, `onnxruntime.capi.onnxruntime_pybind11_state
    .get_all_operator_schema()`'s "Cache Format" section: "The past and
    present KV cache tensors are expected in a BNSH format: (batch_size,
    num_heads, cache_sequence_length, head_size)"; the plain ai.onnx op's
    own schema doc, `onnx.defs.get_schema("Attention", domain="")`, names
    the same axis order for its own `past_key`/`past_value` inputs) -- so
    the `kv_num_heads` axis sits at axis 1 of a rank-4 tensor, exactly the
    axis :func:`_apply_one_gqa_chain`'s own `keep_groups` index set already
    selects along K's/V's own producer weight. A *dynamic* (non-constant)
    past_key/past_value -- an ordinary graph input or intermediate
    activation, not a weight -- is always left alone and never blocks a
    match: it is the caller's own runtime data, not something this rewrite
    could corrupt by leaving untouched, the same reasoning that already
    applied before this function existed. A *constant* one is declined
    (this function returns ``False``, and the caller's whole match fails)
    when its shape isn't confidently this exact layout -- not rank 4, or its
    axis-1 length doesn't already match `kv_num_heads` -- rather than
    guessed at.

    A plain-FLOAT constant of that shape is always accepted (the original,
    unquantized case). A constant whose dtype is instead one of
    :data:`_QUANTIZED_KV_CACHE_DTYPES` (`float8e4m3fn`/`uint8`/`int8` --
    confirmed via `GroupQueryAttention`'s own `T_CACHE` type constraint,
    `get_all_operator_schema()` again, whose "Quantization" doc section
    states outright: "When quantization is enabled, `past_key` and
    `past_value` inputs can be of type `float8e4m3fn`, `uint8` or `int8`.
    The corresponding `k_scale` and `v_scale` tensors must be provided.")
    is a *quantized* KV cache, only ever accepted when the caller passes
    `scale_indices` (a `(k_scale_idx, v_scale_idx)` pair the caller's own op
    schema defines) -- `GroupQueryAttention` passes ``(12, 13)``
    (`k_scale`/`v_scale`'s own input positions per that same schema dump);
    the plain ``ai.onnx::Attention`` op has no `k_scale`/`v_scale` inputs at
    all (confirmed directly off `onnx.defs.get_schema("Attention",
    domain="")` on this environment's installed `onnx==1.22.0`: its full
    input list is `Q, K, V, attn_mask?, past_key?, past_value?,
    nonpad_kv_seqlen?` -- no scale inputs -- and its `past_key`/`past_value`
    type constraints `T1`/`T2` are `{float, float16, bfloat16, double}` only,
    with no quantized dtype in either), so :func:`_match_onnx_attention_producer`
    never passes `scale_indices` and a quantized cache there is declined
    outright by the branch below, exactly as it always was. When
    `scale_indices` *is* given and the cache is quantized, the corresponding
    scale input is required to be connected (per the schema quote above) and,
    if it is itself a constant, must be one of the two shapes the schema's
    own "Quantization Modes" doc section names: `"PER_TENSOR"` (a single
    scalar, e.g. shape `[1]` -- broadcasts identically regardless of
    `kv_num_heads`, so it needs no slicing and is left completely alone) or
    `"PER_CHANNEL"` (`[1, kv_num_heads, 1, head_size]` -- the *same* axis-1
    `kv_num_heads` layout as the cache tensor itself, so it is safe to slice
    along axis 1 by the identical `keep_groups` index set, exactly the
    reasoning that already applied to the cache tensor); any other constant
    scale shape (not rank 4 with axis-1 length `kv_num_heads`, and not a
    single-element broadcast) is declined the same conservative way an
    unrecognized cache shape already is, rather than guessed at, and a
    *dynamic* (non-constant) scale is left alone exactly like a dynamic
    cache tensor -- the caller's own runtime data, not a weight this rewrite
    could silently corrupt by leaving untouched. :func:`_apply_one_gqa_chain`
    itself performs the actual axis-1 slice(s) once a match succeeds.
    """
    for idx, scale_idx in zip(indices, scale_indices or (None, None)):
        if len(node.input) <= idx or not node.input[idx]:
            continue
        past_init = initializer_map.get(node.input[idx])
        if past_init is None:
            continue  # dynamic -- the caller's own runtime data, left alone
        if len(past_init.dims) != 4 or past_init.dims[1] != kv_num_heads:
            return False  # not a shape this function can safely slice
        if _is_supported_float_dtype(past_init.data_type):
            continue  # unquantized cache -- nothing else to check
        if scale_idx is None or past_init.data_type not in _QUANTIZED_KV_CACHE_DTYPES:
            return False  # quantized dtype with nowhere to locate its scale
        if len(node.input) <= scale_idx or not node.input[scale_idx]:
            return False  # quantized cache with no k_scale/v_scale connected
        scale_init = initializer_map.get(node.input[scale_idx])
        if scale_init is None:
            continue  # dynamic scale -- the caller's own runtime data
        if scale_init.data_type != onnx.TensorProto.FLOAT:
            return False  # not T_KV_SCALE's own float-only constraint
        if int(np.prod(scale_init.dims)) == 1:
            continue  # PER_TENSOR: a single broadcast scalar, nothing to slice
        if len(scale_init.dims) != 4 or scale_init.dims[1] != kv_num_heads:
            return False  # not the PER_CHANNEL [1, kv_num_heads, 1, head_size] layout
    return True


def _match_gqa_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[int, int]]:
    """If `node` is a ``com.microsoft::GroupQueryAttention`` node this
    module can safely act on, returns ``(num_heads, kv_num_heads)``.

    Requires: separate, non-empty query/key/value inputs (rules out the
    op's packed-QKV calling convention, where key/value are left empty and
    Q/K/V instead live concatenated in `query` -- a different tensor layout
    this function doesn't attempt to slice) and the `seqlens_k`/
    `total_sequence_length` inputs `GroupQueryAttention`'s schema requires
    even for a plain forward pass (both independent of head count, so never
    need touching themselves -- their presence is checked only as a sign
    this is a real, complete GQA node rather than a partially-constructed
    one); `num_heads`/`kv_num_heads` attributes with `num_heads` a positive
    multiple of `kv_num_heads`; and a `past_key`/`past_value` (indices 3/4)
    this module can safely act on, per
    :func:`_past_kv_constants_are_sliceable` (passed `scale_indices=(12,
    13)`, `k_scale`/`v_scale`'s own input positions on this op's schema) --
    a constant float BNSH cache is sliced along its own `kv_num_heads` axis
    by :func:`_apply_one_gqa_chain`, using exactly the same `keep_groups`
    index set K's/V's own producer weights are sliced by; a constant
    quantized (`float8e4m3fn`/`uint8`/`int8`) cache is sliced the same way,
    together with its own `k_scale`/`v_scale` when that scale is itself a
    constant of the schema's `"PER_CHANNEL"` shape (left alone, needing no
    slicing, when `"PER_TENSOR"` -- see :func:`_past_kv_constants_are_sliceable`'s
    own docstring for the full quantized-cache reasoning). cos_cache/sin_cache
    (indices 7/8, for rotary position embedding), if present, are always left
    alone regardless: both are `[max_sequence_length, rotary_dim/2]`,
    broadcast identically across every head, so a head/group count change
    can never invalidate them.

    Two more optional inputs *do* carry a genuine per-(query-)head axis,
    confirmed via actual onnxruntime execution to have a real, non-ignored
    numeric effect (see ``tests/test_pruning.py``'s own "attention_bias"/
    "head_sink" subsections under the GroupQueryAttention section) -- both
    previously unhandled by this matcher entirely, a real gap this function
    now closes: `attention_bias` (index 10), documented shape `(batch_size
    or 1, num_heads or 1, sequence_length, total_sequence_length)` -- note
    "num_heads" here is the *query* head count (this op's own `num_heads`
    attribute, not `kv_num_heads`): `attention_bias` is added to Q*K'
    *after* GQA's own internal K/V-repeat-to-query-head-count broadcast, so
    it is addressed per query head, the same as Q's own producer weight,
    and gets the identical constant-resolves-cleanly-or-is-declined
    treatment via :func:`_head_bias_axis` that :func:`_match_attention_producer`
    gives its own `attention_bias` and :func:`_match_onnx_attention_producer`
    its own `attn_mask` (all three share that one classifier). `head_sink`
    (index 11), documented shape `(num_heads,)` -- again the query head
    count -- is not a broadcastable mask at all but a genuine one-scalar-
    per-query-head parameter (a per-head softmax-denominator smoothing
    factor); a constant one is only ever accepted at exactly that shape
    (declined otherwise, rather than guessed at), a dynamic one left alone
    as always. :func:`_apply_one_gqa_chain` performs the actual slice(s) of
    both once a match succeeds.
    """
    if node.domain != _ATTENTION_DOMAIN or node.op_type != "GroupQueryAttention":
        return None
    if len(node.input) < 7 or not (node.input[0] and node.input[1] and node.input[2]):
        return None

    num_heads = kv_num_heads = None
    for attr in node.attribute:
        if attr.name == "num_heads":
            num_heads = attr.i
        elif attr.name == "kv_num_heads":
            kv_num_heads = attr.i
    if not num_heads or not kv_num_heads or num_heads <= 0 or kv_num_heads <= 0:
        return None
    if num_heads % kv_num_heads != 0:
        return None

    if not _past_kv_constants_are_sliceable(
        node, initializer_map, (3, 4), kv_num_heads, scale_indices=(12, 13)
    ):
        return None

    if len(node.input) > 10 and node.input[10]:  # attention_bias
        bias_init = initializer_map.get(node.input[10])
        if bias_init is not None and int(np.prod(bias_init.dims)) > 0:
            if _head_bias_axis(list(bias_init.dims), num_heads) is None:
                return None  # doesn't statically resolve -- decline rather than guess

    if len(node.input) > 11 and node.input[11]:  # head_sink
        sink_init = initializer_map.get(node.input[11])
        if sink_init is not None and list(sink_init.dims) != [num_heads]:
            return None  # not the schema's own (num_heads,) shape

    return num_heads, kv_num_heads


def _match_onnx_attention_producer(
    node: onnx.NodeProto, initializer_map: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[int, int]]:
    """If `node` is a plain ``ai.onnx`` ``Attention`` node (domain ``""``,
    opset 24+ -- confirmed via ``onnx.defs.get_schema("Attention",
    domain="")`` against this environment's installed ``onnx==1.22.0``, see
    this module's own "Attention-head pruning" section comment) this module
    can safely act on, returns ``(q_num_heads, kv_num_heads)``.

    Structurally the closest cousin of :func:`_match_gqa_producer`: three
    separate, un-merged query/key/value inputs (``Q``, ``K``, ``V`` at
    indices 0/1/2, all required by the schema) rather than one merged
    weight, plus independent `q_num_heads`/`kv_num_heads` attributes with
    `q_num_heads` a positive multiple of `kv_num_heads` -- the same
    MHA/GQA/MQA taxonomy the op's own schema doc names explicitly. Both
    attributes are schema-*optional* (inferable from a rank-4 ``Q``/``K``
    input's own head axis, per ``onnx.reference.ops.op_attention``'s
    reference kernel), but this function requires both given explicitly:
    the topology this pass matches -- ``Q``/``K``/``V`` arriving directly
    from a MatMul/vanilla-Gemm projection's raw (rank-3,
    ``[batch, seq, hidden]``) output, the same shape
    :func:`_match_gqa_producer` already assumes for `GroupQueryAttention`
    -- is exactly the case the reference kernel itself asserts both
    attributes for, so a node relying on rank-4-inferred head counts isn't
    a shape this pass tracks and is declined rather than guessed at.

    The optional `attn_mask` input (index 3) is a boolean-or-float tensor
    added against the `(batch, q_num_heads, q_seq, kv_seq)` attention-score
    tensor via ordinary broadcasting (`onnx.reference.ops.op_attention`
    adds it with a plain ``+``) -- the op's own doc names the full rank-4
    shape or "a shape broadcastable to it" explicitly, and broadcasting a
    lower-rank tensor right-aligns it, so a rank-3 mask's own axis 0 lands
    on the *q_num_heads* slot too, not just a rank-4 mask's axis 1 (see
    :func:`_head_bias_axis`'s own docstring for the full reasoning, verified
    via actual onnxruntime execution). A connected constant is declined
    outright when its shape doesn't cleanly resolve, via
    :func:`_head_bias_axis`, to either "genuinely per-`q_num_heads`-head,
    safe to slice" or "broadcast, no per-head values at all" -- unlike
    `past_key`/`past_value` below, where the one schema-documented shape is
    either matched outright or declined, this mask's shape space is wider
    (any rank up to 4, broadcastable), so a shape landing outside both
    recognized cases is declined rather than guessed at (narrower than an
    earlier version of this matcher, which declined *any* non-empty
    constant here regardless of shape -- overly conservative for, e.g., the
    common ``(seq, seq)``/rank-2 case, which never has a `q_num_heads` axis
    at all and needs no slicing; see
    ``test_onnx_attention_pruning_nonempty_2d_attn_mask_constant_is_pruned``).
    It is left alone -- and does not block the match either way -- if
    dynamic (an ordinary graph input or intermediate activation, the
    caller's own runtime data, the same reasoning `past_key`/`past_value`
    below already gets). :func:`_apply_one_gqa_chain` performs the actual
    slice, by `keep_q_heads`, of a constant resolving to the per-head case.
    The optional `past_key`/`past_value` inputs (indices
    4/5 -- a different pair of indices from `GroupQueryAttention`'s own 3/4,
    and this op has no `seqlens_k`/`total_sequence_length` equivalent to
    require) get the same safety gate :func:`_match_gqa_producer` gives its
    own `past_key`/`past_value`, via the same shared
    :func:`_past_kv_constants_are_sliceable` -- but called here with no
    `scale_indices` (left at that parameter's default, `None`), unlike
    `GroupQueryAttention`'s own call: this op's schema (confirmed via
    `onnx.defs.get_schema("Attention", domain="")`) has no `k_scale`/
    `v_scale` inputs at all, and its `past_key`/`past_value` type
    constraints (`T1`/`T2`) list only `float`/`float16`/`bfloat16`/`double`
    -- no quantized dtype -- so a constant `past_key`/`past_value` here is
    only ever sliced when it is that expected float BNSH shape, along its
    own `kv_num_heads` axis, by :func:`_apply_one_gqa_chain`, using the same
    `keep_groups` index set K's/V's own producer weights are sliced by; a
    quantized cache (off-schema for this particular op) still declines the
    whole match outright, exactly as before, and a dynamic one is left alone
    as always. `nonpad_kv_seqlen` (index 6), like `GroupQueryAttention`'s
    own `seqlens_k`, is `[batch_size]`-shaped and independent of head count,
    so its presence never blocks a match either.
    """
    if node.domain != "" or node.op_type != "Attention":
        return None
    if len(node.input) < 3 or not (node.input[0] and node.input[1] and node.input[2]):
        return None

    q_num_heads = kv_num_heads = None
    for attr in node.attribute:
        if attr.name == "q_num_heads":
            q_num_heads = attr.i
        elif attr.name == "kv_num_heads":
            kv_num_heads = attr.i
    if not q_num_heads or not kv_num_heads or q_num_heads <= 0 or kv_num_heads <= 0:
        return None
    if q_num_heads % kv_num_heads != 0:
        return None

    if len(node.input) > 3 and node.input[3]:  # attn_mask
        mask_init = initializer_map.get(node.input[3])
        if mask_init is not None and int(np.prod(mask_init.dims)) > 0:
            if _head_bias_axis(list(mask_init.dims), q_num_heads) is None:
                return None  # doesn't statically resolve -- decline rather than guess

    if not _past_kv_constants_are_sliceable(
        node, initializer_map, (4, 5), kv_num_heads
    ):
        return None

    return q_num_heads, kv_num_heads


def _match_packed_qkv_split(
    split_node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    node_by_output: Dict[str, onnx.NodeProto],
) -> Optional[Tuple[str, bool, Optional[str], int, int, int, str]]:
    """If `split_node` is a ``Split`` node splitting one *packed*
    MatMul/vanilla-Gemm projection's output into exactly three
    Q-then-K-then-V column ranges -- the graph-level "packed QKV" upstream
    of `GroupQueryAttention`'s (or the plain ai.onnx `Attention` op's) own
    three separate query/key/value *inputs*, as opposed to either op's own
    schema-level packed-input convention (a single tensor passed as
    `query` itself, with `key`/`value` left empty -- already handled by
    :func:`_match_gqa_producer`/:func:`_match_onnx_attention_producer`
    declining it outright, since it's a different tensor layout neither
    matcher attempts to slice) -- returns ``(weight_name,
    weight_transposed, bias_name_or_None, nq, nk, nv, split_sizes_name)``.

    This exact topology -- one packed MatMul/Gemm, optionally biased,
    feeding a two-input ``Split`` (``axis=-1``, a constant int64
    ``[nq, nk, nv]`` second input) whose three outputs are Q/K/V in that
    order -- is confirmed, empirically, to be what Microsoft's own
    onnxruntime-genai model builder emits for `GroupQueryAttention` on
    Qwen3-style models (per-head Q/K RMSNorm fused into the op itself, via
    its own `q_norm_weight`/`k_norm_weight` inputs -- see
    `onnxruntime_genai/models/builders/base.py`'s own
    ``make_attention_input_proj``/``is_fused_qk_norm_gqa_supported``: when
    `use_packed_matmul` and both `q_norm`/`k_norm` are set and the fused
    in-op norm path is supported (CUDA/WebGPU), a single packed
    `qkv_proj` MatMul (plus a single packed `Add` bias, if any bias
    exists) feeds one `Split` whose three raw outputs are wired directly
    into `GroupQueryAttention`'s own three separate query/key/value
    inputs -- exactly the shape matched here). The common case with no
    Q/K norm instead relies on `GroupQueryAttention`'s own native
    packed-`query`-input convention and emits no `Split` at all, and the
    case with Q/K norm but *no* fused in-op norm support (CPU/DirectML --
    `is_fused_qk_norm_gqa_supported` only allows the in-op path on
    CUDA/WebGPU) instead runs per-head `SimplifiedLayerNorm` (via a
    `Reshape` -> `SimplifiedLayerNormalization` -> `Reshape` sandwich --
    `make_qk_norm`'s own subgraph, confirmed by reading it directly) and,
    unless RoPE is itself fused into the op, a `com.microsoft::RotaryEmbedding`
    node between the `Split` and the op's own Q/K inputs (`make_qk_norm`
    then `make_rotary_embedding_op`, in that order, per
    `make_attention_qk_rope_and_norm`'s own comment "Base order: norm
    first, then RoPE"). This function itself still requires `split_node`'s
    own three outputs to feed *something* directly -- it never looks past
    `split_node` itself -- but its caller,
    :func:`_find_separate_qkv_chains`, no longer requires that something to
    be the attention op's own Q/K inputs verbatim: it first walks *back*
    from those inputs, via :func:`_walk_back_through_qk_norm_rope`, through
    exactly this Reshape/Norm/Reshape-then-RotaryEmbedding shape (each hop
    optional, independently, on Q's and K's own branch) to find the real
    `Split` outputs this function is given, so the two together now cover
    this CPU/DirectML shape too -- see that function's own docstring for
    the exact topology and how its own head-independence (no gamma
    slicing, no `RotaryEmbedding` weight/cache slicing -- only its own
    `num_heads` attribute, when nonzero, and a crossed `Reshape`'s own
    target-width entry ever need updating) was confirmed. Only a `Split`
    output feeding neither the attention op directly nor one of these
    exact hops -- an unrecognized intermediate op, a branch, RoPE before
    norm, or a shape this pass hasn't otherwise verified -- still falls
    through to :func:`_find_separate_qkv_chains`'s own per-branch
    :func:`_match_producer` walk, which declines it like any other
    unrecognized producer rather than silently mis-slicing it.

    Declines (``None``) unless every one of the following holds, checked
    with the same conservative bar as every other producer match in this
    module:

    - `split_node` is a plain ``ai.onnx`` (domain ``""``) `Split` with
      exactly two inputs (the confirmed opset-13+ tensor-input form this
      real exporter uses -- the older `split`-as-attribute form, still
      legal on older opsets, is a structurally different rewrite target
      this function doesn't attempt) and exactly three outputs.
    - Its `axis` attribute is present and exactly ``-1`` (the confirmed
      pattern's own value -- other axis values aren't declined as unsafe
      so much as simply not the one shape this function was verified
      against; a differently-axised packed-QKV split isn't guessed at).
    - Its second input is a constant int64 initializer of shape ``[3]``
      (the split sizes ``[nq, nk, nv]``, all strictly positive) with
      exactly one consumer (this `Split` node) -- an initializer shared
      with anything else can't be safely overwritten in place by
      :func:`_apply_one_gqa_chain`'s own write-back, the same "shared
      constant, don't mutate" bar :func:`_walk_to_attention_consumer`
      already holds its own Reshape-shape constant to.
    - Its first (data) input has exactly one consumer (this `Split`
      node) and is produced by a node :func:`_match_producer` accepts (a
      MatMul/vanilla-Gemm with a constant 2-D float32 weight, and, for
      Gemm, either no bias or a constant one) whose own output width
      equals ``nq + nk + nv`` exactly -- anything else (a non-constant
      weight, a shared/branching packed-projection output, an op
      :func:`_match_producer` doesn't recognize) is declined, never
      guessed at.
    """
    if split_node.domain != "" or split_node.op_type != "Split":
        return None
    if len(split_node.output) != 3 or len(split_node.input) != 2:
        return None
    if not split_node.input[0] or not split_node.input[1]:
        return None

    axis = None
    for attr in split_node.attribute:
        if attr.name == "axis":
            axis = attr.i
    if axis != -1:
        return None

    sizes_name = split_node.input[1]
    sizes_init = initializer_map.get(sizes_name)
    if (
        sizes_init is None
        or sizes_init.data_type != onnx.TensorProto.INT64
        or list(sizes_init.dims) != [3]
    ):
        return None
    if len(consumers_of.get(sizes_name, [])) != 1:
        return None  # shared split-sizes constant -- mutating it isn't safe

    nq, nk, nv = (int(x) for x in onnx.numpy_helper.to_array(sizes_init))
    if nq <= 0 or nk <= 0 or nv <= 0:
        return None

    data_name = split_node.input[0]
    if len(consumers_of.get(data_name, [])) != 1:
        return None  # shared packed-projection output -- can't rewrite in isolation
    prod_node = node_by_output.get(data_name)
    if prod_node is None:
        return None
    pinfo = _match_producer(prod_node, initializer_map)
    if pinfo is None:
        return None
    w_name, w_transposed, bias_name, n_channels = pinfo
    if n_channels != nq + nk + nv:
        return None

    return w_name, w_transposed, bias_name, nq, nk, nv, sizes_name


# Scope boundaries this section lands on, deliberately: the CPU/DirectML
# Q/K-norm GQA export shape (see :func:`_match_packed_qkv_split`'s own
# docstring) inserts, between a packed-QKV `Split`'s own Q/K outputs and
# `GroupQueryAttention`'s (or the plain ai.onnx `Attention` op's) own Q/K
# inputs, an optional per-head norm sandwich (a `Reshape` collapsing
# `(batch, seq, hidden)` to `(batch, seq * num_heads, head_size)`, a
# `SimplifiedLayerNormalization` with its default `axis=-1` and a
# `[head_size]` `scale` -- `make_qk_norm`'s own subgraph, read directly off
# `onnxruntime_genai/models/builders/base.py`) and, unless RoPE is fused
# into the attention op itself, an optional `com.microsoft::RotaryEmbedding`
# node applied directly to the flat `(batch, seq, hidden)` tensor
# (`make_rotary_embedding_op`), norm before RoPE when both are present
# (`make_attention_qk_rope_and_norm`'s own comment: "Base order: norm
# first, then RoPE"). :func:`_walk_back_through_qk_norm_rope` below walks
# *backward* from the attention op's own Q or K input through this exact
# optional-hop sequence to find the `Split` output feeding it, the same
# direction :func:`_walk_matmul_producer_backward` already walks in this
# module for an unrelated reason (residual-chain producers), and for the
# same reason: the *forward* direction (`_walk_to_consumer`'s own style)
# would need to start from the `Split`'s own output, guessing which
# consumer among possibly several to follow, before it's known whether the
# far end is even this attention op's own Q/K input at all.
#
# Two head-independence facts make every hop here safe to prune through
# with **no slicing of the hop's own constants at all** -- confirmed
# empirically (real `onnxruntime.InferenceSession` runs, not the schema
# doc alone; see ``tests/test_pruning.py``'s own
# ``test_simplified_layer_norm_is_head_independent_for_pruning``/
# ``test_rotary_embedding_is_head_independent_for_pruning`` for the exact
# models and numbers), not assumed:
#
# - `SimplifiedLayerNormalization`'s own `scale` (`[head_size]`, confirmed
#   live via `onnxruntime.capi.onnxruntime_pybind11_state
#   .get_all_operator_schema()` -- this op isn't in onnx's own schema
#   registry at all despite registering under the default ("") domain, see
#   this module's own `_NORM_PASS_THROUGH_OPS` comment far above -- on this
#   environment's installed `onnx==1.22.0`/`onnxruntime==1.29.0` -- no
#   `bias`/`B` input at all, unlike plain `LayerNormalization`) broadcasts
#   *identically* to every head's own row regardless of how many heads
#   exist: reducing over exactly the trailing `head_size` axis per
#   `(batch, seq, head)` triple, with the *same* `scale` array applied to
#   every head, dropping whole heads before or after this norm gives
#   bit-identical results for the surviving heads either way (confirmed
#   with a hand-built `Reshape` -> `SimplifiedLayerNormalization` ->
#   `Reshape` model, keeping an arbitrary, non-contiguous subset of heads,
#   `max abs diff == 0.0` both before and after). So, unlike every other
#   per-channel affine operand this module tracks (`_NORM_PASS_THROUGH_OPS`'s
#   own `scale`/`bias`, `_skip_layer_norm_const_names`'s own `gamma`/`beta`),
#   this norm's own `scale` is *never* sliced here -- there is nothing
#   head-shaped about it to slice.
# - `com.microsoft::RotaryEmbedding` (confirmed live via
#   `onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()`
#   to be a real op under that domain, inputs `[input, position_ids,
#   cos_cache, sin_cache]`, attributes `num_heads`/`rotary_embedding_dim`/
#   `interleaved`/`is_packed_batching`/`scale`) rotates each head's own
#   `head_size` (or `rotary_embedding_dim`-wide prefix of it) slice using
#   only that head's own data and a position-dependent (never head-
#   dependent) `cos_cache`/`sin_cache` lookup -- confirmed head-independent,
#   again bit-identical, across every `interleaved` (0 and 1) and
#   full-vs-partial `rotary_embedding_dim` combination tried. Its own
#   `num_heads` attribute is the *one* constant here that does need
#   updating post-pruning -- unless it's already the schema's `0`
#   ("infer `num_heads` from the input's own trailing width divided by
#   `cos_cache`'s/`rotary_embedding_dim`'s own implied `head_size`",
#   confirmed to self-adjust correctly to a smaller post-pruning head count
#   with *no* attribute change needed at all) -- to the new post-pruning
#   head count for whichever branch (Q's `num_heads`, K's `kv_num_heads`)
#   it sits on; neither `cos_cache` nor `sin_cache` themselves ever need
#   touching, having no head axis at all.
#
# A `Reshape`'s own two target-shape entries are `head_size` (never
# changes) and this branch's flat width (`num_heads * head_size` or
# `kv_num_heads * head_size`, which *does* shrink) -- only the second
# `Reshape` (the one reconstructing the flat `(batch, seq, hidden)` shape
# right after the norm) ever encodes the latter as a literal, and only
# when it does (a resolvable, single-consumer constant -- see
# :func:`_reshape_last_dim`) is that hop even crossed; otherwise the walk
# simply doesn't cross it, and the match declines further up the call
# chain the same way any other unrecognized shape does.
#
# Neither the `Reshape` collapsing to `(batch, seq * num_heads, head_size)`
# ahead of the norm nor the norm's own training-only `inv_std_var` second
# output (already declined, the same way, by `_walk_to_consumer`'s own
# `_NORM_PASS_THROUGH_OPS` hop, whenever it's actually consumed) needs any
# constant update at all -- neither depends on how many heads survive.
@dataclass(frozen=True)
class _QKNormRopePassThrough:
    """One resolved Q- or K-branch "per-head norm + RoPE" pass-through a
    packed-QKV chain walk crossed between a `_GQAChain`'s own `Split` (see
    `_GQAChain.packed_split_sizes`) and the attention op's own Q or K
    input -- see :func:`_walk_back_through_qk_norm_rope` (which builds
    this) for the exact topology and how its head-independence was
    confirmed. Unlike :class:`_ConvPassThrough`'s own `weight`/`bias`
    fields, nothing here names a tensor :func:`_apply_one_gqa_chain` needs
    to *slice* -- this section's own comment above covers why neither a
    crossed norm's `scale` nor a crossed `RotaryEmbedding`'s `cos_cache`/
    `sin_cache` ever needs touching. Two things still do:

    - `rotary_node`: the crossed `RotaryEmbedding` node, or ``None`` when
      RoPE wasn't crossed (norm-only, or neither). When present, its own
      `num_heads` attribute -- whenever it isn't already the schema's `0`
      -- is rewritten to this branch's new post-pruning head count.
    - `reshape_shape`: the crossed post-norm `Reshape`'s own target-shape
      constant name, or ``None`` when that `Reshape`/norm sandwich wasn't
      crossed (RoPE-only, or neither). When present, its own last entry
      (this branch's flat width) is rewritten to the new post-pruning
      value, mirroring how :func:`_apply_one_gqa_chain`'s existing
      `chain.chain_ops` loop already rewrites a *post*-attention-output
      Reshape's own target width.

    `nodes` lists every node actually crossed (any of `RotaryEmbedding`,
    the post-norm `Reshape`, the `SimplifiedLayerNormalization`, and the
    pre-norm `Reshape`, in that order, whichever were present) purely for
    :func:`_apply_attention_chains`'s own stale-`value_info` bookkeeping --
    every one of their own output tensors is narrower after pruning even
    though, per the two fields above, only two of them ever need editing.
    Never empty when this class is actually constructed --
    :func:`_walk_back_through_qk_norm_rope` returns ``None`` instead of an
    empty-`nodes` instance when nothing was crossed.
    """

    nodes: Tuple[onnx.NodeProto, ...]
    rotary_node: Optional[onnx.NodeProto] = None
    reshape_shape: Optional[str] = None


def _walk_back_through_qk_norm_rope(
    name: str,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
    node_by_output: Dict[str, onnx.NodeProto],
) -> Tuple[str, Optional[_QKNormRopePassThrough]]:
    """Walks *backward* from `name` -- `GroupQueryAttention`'s (or the
    plain ai.onnx `Attention` op's) own Q or K input, already confirmed by
    the caller to have exactly one consumer -- through an optional
    `com.microsoft::RotaryEmbedding` node and, further back, an optional
    `Reshape` -> `SimplifiedLayerNormalization` -> `Reshape` "per-head norm"
    sandwich, stopping at the first tensor not produced by one of these two
    recognized hops (each independently optional; the whole point of this
    walk is that `name` may sit directly on a packed-QKV `Split`'s own
    output with *neither* hop present at all, exactly as before this
    pass-through existed). See this section's own comment above for the
    exact topology (confirmed against onnxruntime-genai's own model
    builder source) and the head-independence facts that make every hop
    here safe with no constant slicing beyond what :class:`_QKNormRopePassThrough`
    already names.

    Every hop crossed is additionally required to have exactly one
    consumer at each internal edge (the next hop inward, or `name`'s own
    eventual attention-op input) and not itself be a graph output -- the
    same "no other consumer along the way" bar every forward chain walk in
    this module already holds (:func:`_walk_to_consumer`) -- so a norm/RoPE
    tensor secretly reused elsewhere (a residual branch, a second attention
    op) safely stops the walk there rather than risking a corrupting slice.
    A crossed norm's own `inv_std_var` second output, when itself consumed
    by anything, likewise stops the walk at that node -- the same
    training-only-secondary-output bar :func:`_walk_to_consumer`'s own
    `_NORM_PASS_THROUGH_OPS` hop already holds. A crossed post-norm
    `Reshape`'s own target-shape input must be a constant whose last entry
    is statically resolvable (:func:`_reshape_last_dim`) and single-consumer
    (safe to overwrite in place) -- the same bar
    :func:`_walk_to_attention_consumer`'s own Reshape hop already holds its
    shape constant to -- or that `Reshape` (and, transitively, the norm
    sandwich behind it) simply isn't crossed.

    The norm's own `scale` is only confirmed here to be a flat
    per-channel-shaped float constant (:func:`_flat_channel_const`) -- the
    real ``dims == [head_size]`` check, only possible once `head_size` is
    itself known (only after the packed producer this walk is feeding into
    is resolved, which needs this walk's own result first), is left to the
    caller (:func:`_qk_norm_rope_hop_is_consistent`), the same
    deferred-check idiom :func:`_norm_pass_through_const_names` already
    uses for its own `_NORM_PASS_THROUGH_OPS` hop.

    Returns ``(root_name, hop)``: `root_name` is `name` itself and `hop` is
    ``None`` when neither hop was crossed (the pre-existing, still-most-common
    "Split feeds the attention op directly" shape); otherwise `root_name` is
    the tensor immediately upstream of every hop crossed (expected, by the
    caller, to be one of the shared `Split`'s own raw outputs) and `hop`
    describes what was crossed.
    """

    def _is_internal(n: str) -> bool:
        return len(consumers_of.get(n, [])) == 1 and n not in graph_outputs

    nodes: List[onnx.NodeProto] = []
    rotary_node: Optional[onnx.NodeProto] = None
    reshape_shape: Optional[str] = None
    cur = name

    rope = node_by_output.get(cur)
    if (
        rope is not None
        and rope.domain == _ATTENTION_DOMAIN
        and rope.op_type == "RotaryEmbedding"
        and len(rope.input) >= 1
        and rope.input[0]
        and len(rope.output) == 1
        and rope.output[0] == cur
        and _is_internal(rope.input[0])
    ):
        rotary_node = rope
        nodes.append(rope)
        cur = rope.input[0]

    reshape2 = node_by_output.get(cur)
    if (
        reshape2 is not None
        and reshape2.domain == ""
        and reshape2.op_type == "Reshape"
        and len(reshape2.input) == 2
        and reshape2.input[0]
        and len(reshape2.output) == 1
        and reshape2.output[0] == cur
        and _is_internal(reshape2.input[0])
        and _reshape_last_dim(reshape2, initializer_map) is not None
        and len(consumers_of.get(reshape2.input[1], [])) == 1
    ):
        norm_out = reshape2.input[0]
        norm_node = node_by_output.get(norm_out)
        if (
            norm_node is not None
            and norm_node.domain == ""
            and norm_node.op_type == "SimplifiedLayerNormalization"
            and len(norm_node.output) >= 1
            and norm_node.output[0] == norm_out
            and len(norm_node.input) == 2
            and norm_node.input[1]
            and _flat_channel_const(norm_node.input[1], initializer_map)
            and _norm_axis_is_last(norm_node, norm_node.input[0], None)
            and not (
                len(norm_node.output) > 1
                and norm_node.output[1]
                and (
                    consumers_of.get(norm_node.output[1])
                    or norm_node.output[1] in graph_outputs
                )
            )
            and norm_node.input[0]
            and _is_internal(norm_node.input[0])
        ):
            reshape1 = node_by_output.get(norm_node.input[0])
            if (
                reshape1 is not None
                and reshape1.domain == ""
                and reshape1.op_type == "Reshape"
                and len(reshape1.input) == 2
                and reshape1.input[0]
                and len(reshape1.output) == 1
                and reshape1.output[0] == norm_node.input[0]
                and _is_internal(reshape1.input[0])
            ):
                nodes.extend([reshape2, norm_node, reshape1])
                reshape_shape = reshape2.input[1]
                cur = reshape1.input[0]

    if not nodes:
        return name, None
    return cur, _QKNormRopePassThrough(tuple(nodes), rotary_node, reshape_shape)


def _qk_norm_rope_hop_is_consistent(
    hop: Optional[_QKNormRopePassThrough],
    initializer_map: Dict[str, onnx.TensorProto],
    branch_width: int,
    head_size: int,
) -> bool:
    """Deferred correctness check for a :func:`_walk_back_through_qk_norm_rope`
    result, run only once `branch_width` (this branch's own pre-pruning
    `nq`/`nk`) and `head_size` are themselves known -- see that function's
    own docstring for why this can't be checked any earlier. `True`
    (nothing to check) when `hop` is ``None`` (no pass-through crossed for
    this branch). Otherwise: any crossed `SimplifiedLayerNormalization`'s
    own `scale` must be exactly `[head_size]`-shaped (not just flat -- the
    real per-head-width fact the walk itself deferred), and, when a post-norm
    `Reshape` was crossed, its own target-shape constant's last entry must
    exactly equal `branch_width` -- both declined (``False``), never
    guessed at, on any mismatch.
    """
    if hop is None:
        return True
    for node in hop.nodes:
        if node.op_type == "SimplifiedLayerNormalization":
            if list(initializer_map[node.input[1]].dims) != [head_size]:
                return False
    if hop.reshape_shape is not None:
        dims = onnx.numpy_helper.to_array(initializer_map[hop.reshape_shape])
        if dims.size == 0 or int(dims[-1]) != branch_width:
            return False
    return True


def _find_separate_qkv_chains(
    graph: onnx.GraphProto,
    match_producer,
    num_heads_attr: str,
    allow_differing_v_head_size: bool = False,
) -> List[_GQAChain]:
    """Shared body for :func:`_find_gqa_chains` and
    :func:`_find_onnx_attention_chains`: both match a fused attention node
    fed by three separate, un-merged Q/K/V MatMul/vanilla-Gemm projections
    (as opposed to :func:`_find_attention_chains`'s single merged-QKV-weight
    ``com.microsoft::Attention``) and prune it at whole-KV-group granularity
    (see :func:`_apply_one_gqa_chain`/:func:`_gqa_group_importance`),
    differing only in which node/attributes `match_producer` recognizes
    (:func:`_match_gqa_producer` or :func:`_match_onnx_attention_producer`),
    which attribute on the matched node holds the query head count
    (`num_heads_attr` -- see :class:`_GQAChain`'s own field of that name),
    and whether V's own head_size is allowed to differ from Q/K's shared one
    (`allow_differing_v_head_size` -- ``False`` for `GroupQueryAttention`,
    which `fuse_gqa.h` never emits with anything but equal Q/K/V head_size,
    ``True`` for the plain ai.onnx op, whose schema genuinely allows it; see
    :class:`_GQAChain`'s own `v_head_size` field).
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}
    node_by_output = {out: node for node in graph.node for out in node.output}

    def _is_internal(name: str) -> bool:
        return len(consumers_of.get(name, [])) == 1 and name not in graph_outputs

    chains = []
    for node in graph.node:
        info = match_producer(node, initializer_map)
        if info is None:
            continue
        num_heads, kv_num_heads = info

        q_name, k_name, v_name = node.input[0], node.input[1], node.input[2]
        if q_name == k_name or q_name == v_name or k_name == v_name:
            continue  # degenerate -- can't independently slice a shared producer

        # A shared upstream `Split` node producing all three -- the
        # packed-QKV-then-Split shape (see :func:`_match_packed_qkv_split`)
        # -- is checked first and handled exclusively: a `Split` node can
        # never itself match `_match_producer` below (it isn't a
        # MatMul/vanilla-Gemm), so falling through to the per-branch loop
        # for it would just decline the same node three times over. Every
        # other shape (three genuinely independent producers, or anything
        # this function doesn't recognize) falls to that per-branch loop
        # unchanged.
        #
        # `root_q`/`root_k` walk *backward* from Q's/K's own attention-op
        # input through an optional per-head norm + RoPE pass-through (see
        # :func:`_walk_back_through_qk_norm_rope` and this section's own
        # comment above) before checking for the shared `Split` -- `q_name`/
        # `k_name` unchanged (and `q_hop`/`k_hop` both `None`) whenever
        # neither hop is present, so this is a strict superset of the
        # pre-existing "`Split` feeds the attention op directly" check, not
        # a behavior change for any graph that already matched. V never
        # gets this treatment -- `make_qk_norm`/`make_rotary_embedding_op`
        # never touch V's own branch (see this section's own comment above)
        # -- so `v_name` is still compared against `prod_q.output[2]`
        # verbatim, exactly as before.
        packed_split_sizes: Optional[str] = None
        q_norm_rope: Optional[_QKNormRopePassThrough] = None
        k_norm_rope: Optional[_QKNormRopePassThrough] = None
        root_q, q_hop = (
            _walk_back_through_qk_norm_rope(
                q_name, initializer_map, consumers_of, graph_outputs, node_by_output
            )
            if _is_internal(q_name)
            else (q_name, None)
        )
        root_k, k_hop = (
            _walk_back_through_qk_norm_rope(
                k_name, initializer_map, consumers_of, graph_outputs, node_by_output
            )
            if _is_internal(k_name)
            else (k_name, None)
        )
        prod_q = node_by_output.get(root_q) if _is_internal(root_q) else None
        prod_k = node_by_output.get(root_k) if _is_internal(root_k) else None
        prod_v = node_by_output.get(v_name) if _is_internal(v_name) else None
        if (
            prod_q is not None
            and prod_q is prod_k
            and prod_q is prod_v
            and prod_q.op_type == "Split"
            and list(prod_q.output) == [root_q, root_k, v_name]
        ):
            packed = _match_packed_qkv_split(
                prod_q, initializer_map, consumers_of, node_by_output
            )
            if packed is None:
                continue
            w_name, w_transposed, bias_name, nq, nk, nv, packed_split_sizes = packed
            producer_infos = [
                (w_name, w_transposed, bias_name, nq),
                (w_name, w_transposed, bias_name, nk),
                (w_name, w_transposed, bias_name, nv),
            ]
            q_norm_rope = q_hop
            k_norm_rope = k_hop
        else:
            producer_infos = []
            matched = True
            for in_name in (q_name, k_name, v_name):
                if not _is_internal(in_name):
                    matched = False
                    break
                prod_node = node_by_output.get(in_name)
                if prod_node is None:
                    matched = False
                    break
                pinfo = _match_producer(prod_node, initializer_map)
                if pinfo is None:
                    matched = False
                    break
                producer_infos.append(pinfo)
            if not matched:
                continue

        (wq, wq_t, bq, nq), (wk, wk_t, bk, nk), (wv, wv_t, bv, nv) = producer_infos
        if packed_split_sizes is None and (wq == wk or wq == wv or wk == wv):
            continue  # degenerate -- can't independently slice a shared producer
        if nq % num_heads or nk % kv_num_heads or nv % kv_num_heads:
            continue
        head_size = nq // num_heads
        v_head_size = nv // kv_num_heads
        if head_size <= 0 or v_head_size <= 0 or nk // kv_num_heads != head_size:
            # Q's and K's own head_size must always agree -- required by the
            # QK^T dot product itself (both ops' schemas name a single
            # shared `head_size` for Q/K, distinct from V's own), not a
            # restriction this pass adds, so a mismatch here is declined
            # regardless of `allow_differing_v_head_size`.
            continue
        if not allow_differing_v_head_size and v_head_size != head_size:
            # `fuse_gqa.h` requires equal Q/K/V head_size before it will
            # even fuse a `GroupQueryAttention` node (confirmed by reading
            # that requirement directly off `fuse_gqa.h` itself: `q_head_size
            # != k_head_size || q_head_size != v_head_size` is one of its own
            # fusion-declining conditions) -- a `GroupQueryAttention` node
            # whose V head size actually differs is declined here rather
            # than mis-sliced, since no real GQA node could ever have one.
            continue

        # Deferred correctness check for any Q/K-norm + RoPE pass-through
        # crossed above (see :func:`_qk_norm_rope_hop_is_consistent`'s own
        # docstring for why this can't run any earlier): a crossed norm's
        # own `scale` must genuinely be `[head_size]`-wide, and a crossed
        # post-norm `Reshape`'s own target width must genuinely be this
        # branch's own pre-pruning flat width -- anything else means this
        # walk crossed a shape it only structurally resembles, not the real
        # Q/K-norm GQA export shape, and the whole match is declined here
        # rather than mis-slicing it.
        if not (
            _qk_norm_rope_hop_is_consistent(q_norm_rope, initializer_map, nq, head_size)
            and _qk_norm_rope_hop_is_consistent(
                k_norm_rope, initializer_map, nk, head_size
            )
        ):
            continue

        out_name = node.output[0]
        if not _is_internal(out_name):
            continue

        # The raw output is always `num_heads * v_head_size` wide -- unlike
        # plain `com.microsoft::Attention` (whose own raw-output-width
        # parameter this same helper takes is named `nv` generically), both
        # matched ops here size their output per *query* head but with
        # *V's* own per-head width (`fuse_gqa.h`'s own "Y =
        # GroupQueryAttention(...)" shape comment; the ai.onnx op's own
        # "hidden_size = q_num_heads * v_head_size" 3D output shape, see its
        # schema doc) -- equal to `nq` exactly when `v_head_size ==
        # head_size` (always true for `GroupQueryAttention`; not required
        # for the plain ai.onnx op once `allow_differing_v_head_size` lets
        # it differ).
        raw_out_width = num_heads * v_head_size
        consumer, chain_ops = _walk_to_attention_consumer(
            out_name, initializer_map, consumers_of, graph_outputs, raw_out_width
        )
        if consumer is None:
            continue

        chains.append(
            _GQAChain(
                node=node,
                q_weight=wq,
                q_bias=bq,
                q_weight_transposed=wq_t,
                k_weight=wk,
                k_bias=bk,
                k_weight_transposed=wk_t,
                v_weight=wv,
                v_bias=bv,
                v_weight_transposed=wv_t,
                num_heads=num_heads,
                kv_num_heads=kv_num_heads,
                head_size=head_size,
                v_head_size=v_head_size,
                chain_ops=chain_ops,
                consumer_node=consumer[0],
                consumer_weight=consumer[1],
                consumer_weight_transposed=consumer[2],
                num_heads_attr=num_heads_attr,
                packed_split_sizes=packed_split_sizes,
                q_norm_rope=q_norm_rope,
                k_norm_rope=k_norm_rope,
            )
        )
    return chains


def _find_gqa_chains(graph: onnx.GraphProto) -> List[_GQAChain]:
    return _find_separate_qkv_chains(graph, _match_gqa_producer, "num_heads")


def _find_onnx_attention_chains(graph: onnx.GraphProto) -> List[_GQAChain]:
    """The plain ``ai.onnx::Attention`` analogue of :func:`_find_gqa_chains`
    -- see :func:`_find_separate_qkv_chains` (the shared body) and
    :func:`_match_onnx_attention_producer` (what's matched and why it's
    declined otherwise). Passes ``allow_differing_v_head_size=True``: unlike
    `GroupQueryAttention`, this op's own schema genuinely allows V its own
    `v_head_size` independent of Q/K's shared `head_size` (see
    :class:`_GQAChain`'s own `v_head_size` field).
    """
    return _find_separate_qkv_chains(
        graph,
        _match_onnx_attention_producer,
        "q_num_heads",
        allow_differing_v_head_size=True,
    )


def _plain_attention_head_importance(
    chain: _AttentionChain,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    dq: int,
    dk: int,
    dv: int,
    importance_norm: str = "l2",
) -> np.ndarray:
    # Combined importance of each head's full Q+K+V weight block -- the Li
    # et al. filter-norm criterion this module uses everywhere else, applied
    # to a whole head's block of columns (across every input row) at once
    # instead of a single output channel/filter. For L2 that's the block's
    # own Frobenius norm (the L2 norm of every entry, vectorized); the L1
    # analogue is the L1 norm of that same vectorized block -- the sum of
    # every entry's own absolute value (``np.abs(block).sum()``) -- rather
    # than any Frobenius-like construction, exactly as
    # :func:`_plain_structured_importance`'s own per-row L1 case has no
    # square/sqrt in it either. wq/wk/wv all share the same row count here
    # (three column-slices of one merged QKV weight), so the concatenation
    # this block is built from is always legal for both norms.
    importance = np.zeros(chain.num_heads, dtype=np.float64)
    for h in range(chain.num_heads):
        block = np.concatenate(
            [
                wq[:, h * dq : (h + 1) * dq],
                wk[:, h * dk : (h + 1) * dk],
                wv[:, h * dv : (h + 1) * dv],
            ],
            axis=1,
        )
        importance[h] = (
            np.abs(block).sum() if importance_norm == "l1" else np.linalg.norm(block)
        )
    return importance


def _head_column_indices(keep_heads: np.ndarray, head_size: int) -> np.ndarray:
    return np.concatenate(
        [np.arange(h * head_size, (h + 1) * head_size) for h in keep_heads]
    )


def _gqa_group_importance(
    chain: _GQAChain,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    importance_norm: str = "l2",
) -> np.ndarray:
    # Combined (Frobenius-norm) importance of each *KV group's* whole
    # block: the group's own K/V head columns plus every query head mapped
    # to it -- the GQA analogue of :func:`_plain_attention_head_importance`,
    # at group instead of individual-head granularity, since a lone query
    # head can't be pruned out from under a shared KV head in isolation
    # (see this module's own "Attention-head pruning" section comment).
    #
    # Combined via sqrt(sum of squared per-block Frobenius norms) rather
    # than norm(concatenate(q_block, k_block, v_block, axis=1)) -- the two
    # are numerically identical whenever the concatenation is even legal
    # (||[A B]||_F^2 == ||A||_F^2 + ||B||_F^2 for any A, B sharing a row
    # count, since Frobenius norm squared is just the sum of every entry
    # squared, and concatenating along columns doesn't change that sum) --
    # but this form stays well-defined when it isn't: Q's own producer
    # weight has as many rows as Q's own source tensor's feature dimension,
    # while K/V's own producer weight has as many rows as K/V's own source
    # tensor's feature dimension, and for cross-attention (Q and K/V drawn
    # from genuinely different source tensors, e.g. a decoder/encoder pair)
    # those two feature dimensions need not match at all -- an ordinary,
    # correctly-matched shape this function must still rank, not one
    # :func:`_match_gqa_producer`/:func:`_match_onnx_attention_producer`
    # decline (nothing about either matcher, or :func:`_find_separate_qkv_chains`
    # that calls them, ties Q's producer weight's row count to K/V's own --
    # see this module's "Attention-head pruning" section comment for the
    # confirmed-supported cross-attention shape this guards). The old
    # concatenate-based form would raise a bare ``ValueError`` from numpy
    # the moment it ran on such a model, crashing the whole pruning call
    # instead of ranking it.
    #
    # `d`/`dv` similarly needn't agree: `d` (`chain.head_size`) is Q's and
    # K's own shared per-head column width in their own producer weights,
    # `dv` (`chain.v_head_size`) is V's own -- equal for every
    # `GroupQueryAttention` chain, but not necessarily for a plain
    # ai.onnx `Attention` chain (see :class:`_GQAChain`'s own `v_head_size`
    # field) -- so `v_block`'s own column stride into `wv` must use `dv`,
    # not `d`.
    #
    # For L1 the same per-block, sum-rather-than-concatenate combination is
    # used, for the same reason: ``||[A B]||_1 == ||A||_1 + ||B||_1``
    # unconditionally, for any A, B (unlike L2's root-sum-square identity
    # above, this one doesn't even *need* a shared row count to hold, since
    # L1 is just a sum over every entry -- but it's the same well-defined
    # answer either way, and keeping both norms on one code path here avoids
    # two subtly-different-looking implementations of the same idea).
    d = chain.head_size
    dv = chain.v_head_size
    group_size = chain.num_heads // chain.kv_num_heads
    importance = np.zeros(chain.kv_num_heads, dtype=np.float64)
    for kv in range(chain.kv_num_heads):
        q_block = np.concatenate(
            [
                wq[:, h * d : (h + 1) * d]
                for h in range(kv * group_size, (kv + 1) * group_size)
            ],
            axis=1,
        )
        k_block = wk[:, kv * d : (kv + 1) * d]
        v_block = wv[:, kv * dv : (kv + 1) * dv]
        if importance_norm == "l1":
            importance[kv] = (
                np.abs(q_block).sum() + np.abs(k_block).sum() + np.abs(v_block).sum()
            )
        else:
            importance[kv] = np.sqrt(
                np.linalg.norm(q_block) ** 2
                + np.linalg.norm(k_block) ** 2
                + np.linalg.norm(v_block) ** 2
            )
    return importance


def _apply_one_plain_attention_chain(
    initializer_map: Dict[str, onnx.TensorProto],
    chain: _AttentionChain,
    sparsity: float,
    compute_importance,
) -> Optional[Tuple[Set[str], str, Set[str]]]:
    """Applies whole-head pruning to one matched ``Attention`` block in
    place: every dropped head removes a *contiguous* ``head_size``-wide
    column block from the single merged QKV weight (and the matching row
    block from the consumer), not an arbitrary top-k column subset. A
    connected constant `attention_bias` (input index 5) already confirmed
    at match time (see :func:`_match_attention_producer`) to resolve, via
    :func:`_head_bias_axis`, to either a genuine per-head tensor or a
    broadcast is sliced along its own head axis by `keep_heads` -- the same
    (not `head_size`-expanded) index set this function's own `keep_heads`
    already is, since `attention_bias`'s axis holds one entry per head, not
    per weight column -- or left untouched when it's a broadcast (or
    absent/dynamic). Returns ``(producer_weight_names, consumer_weight_name,
    stale_output_names)`` on success, or ``None`` if `sparsity` rounds to no
    heads dropped for this block (a no-op, left for the caller to skip).
    """
    h = chain.num_heads
    keep_count = max(1, h - round(h * sparsity))
    if keep_count >= h:
        return None

    dq, dk, dv = chain.nq // h, chain.nk // h, chain.nv // h
    w_init = initializer_map[chain.weight]
    w = _to_f64(w_init)  # [K, Nq+Nk+Nv]
    wq = w[:, : chain.nq]
    wk = w[:, chain.nq : chain.nq + chain.nk]
    wv = w[:, chain.nq + chain.nk :]

    importance = compute_importance(chain, wq, wk, wv, dq, dk, dv)
    keep_heads = np.sort(np.argsort(-importance)[:keep_count])

    q_idx = _head_column_indices(keep_heads, dq)
    k_idx = _head_column_indices(keep_heads, dk) + chain.nq
    v_idx_local = _head_column_indices(keep_heads, dv)
    v_idx = v_idx_local + chain.nq + chain.nk
    all_idx = np.concatenate([q_idx, k_idx, v_idx])

    w_arr = onnx.numpy_helper.to_array(w_init)
    w_init.CopyFrom(onnx.numpy_helper.from_array(w_arr[:, all_idx], name=w_init.name))
    if chain.bias is not None:
        _slice_last_axis(initializer_map[chain.bias], all_idx)

    found_qkv = False
    for attr in chain.node.attribute:
        if attr.name == "num_heads":
            attr.i = keep_count
        elif attr.name == "qkv_hidden_sizes":
            found_qkv = True
            del attr.ints[:]
            attr.ints.extend([keep_count * dq, keep_count * dk, keep_count * dv])
    if not found_qkv:
        chain.node.attribute.append(
            onnx.helper.make_attribute(
                "qkv_hidden_sizes",
                [keep_count * dq, keep_count * dk, keep_count * dv],
            )
        )

    if len(chain.node.input) > 5 and chain.node.input[5]:  # attention_bias
        bias_init = initializer_map.get(chain.node.input[5])
        if bias_init is not None:
            axis = _head_bias_axis(list(bias_init.dims), h)
            if axis is not None and axis >= 0:
                _slice_axis(bias_init, keep_heads, axis)

    _slice_consumer_weight(
        initializer_map[chain.consumer_weight],
        chain.consumer_weight_transposed,
        v_idx_local,
    )

    for _, shape_name in chain.chain_ops:
        if shape_name is not None:
            shape_init = initializer_map[shape_name]
            dims = onnx.numpy_helper.to_array(shape_init).copy()
            dims[-1] = keep_count * dv
            shape_init.CopyFrom(
                onnx.numpy_helper.from_array(dims, name=shape_init.name)
            )

    stale = {chain.node.output[0]}
    stale.update(op.output[0] for op, _ in chain.chain_ops)
    return {chain.weight}, chain.consumer_weight, stale


def _apply_one_gqa_chain(
    initializer_map: Dict[str, onnx.TensorProto],
    chain: _GQAChain,
    sparsity: float,
    compute_group_importance,
) -> Optional[Tuple[Set[str], str, Set[str]]]:
    """Applies whole-KV-group pruning to one matched ``GroupQueryAttention``
    or plain ``ai.onnx::Attention`` block (see :class:`_GQAChain`'s own
    `num_heads_attr` field for how the two are told apart when writing the
    new query head count back) in place: every dropped group removes one
    *contiguous* ``head_size``-wide column block from K's own separate
    weight and one *contiguous* ``v_head_size``-wide column block (equal to
    ``head_size`` for `GroupQueryAttention`, not necessarily for the plain
    ai.onnx op -- see :class:`_GQAChain`'s own `v_head_size` field) from V's
    own, together with the ``num_heads / kv_num_heads`` query-head-sized
    blocks mapped to that group from Q's own separate weight (and the
    matching ``v_head_size``-wide-per-head row block from the consumer,
    since the consumer's own reduction axis is the attention output's own
    hidden dim -- laid out per *query* head at *V's* own per-head width) --
    never an individual query head in isolation. A connected constant
    `past_key`/`past_value` (see :func:`_past_kv_constants_are_sliceable`,
    already confirmed at match time to be a BNSH-format tensor, plain FLOAT
    or a quantized `float8e4m3fn`/`uint8`/`int8` cache with a
    schema-conforming `k_scale`/`v_scale`) is sliced along its own
    `kv_num_heads` axis (axis 1) by the same `keep_groups` index set; for a
    `GroupQueryAttention` block whose cache is quantized, its own
    `k_scale`/`v_scale` (only ever present on this op -- see
    :func:`_match_gqa_producer`'s own docstring) is sliced the identical way
    when it is itself a constant of the schema's `"PER_CHANNEL"` shape
    (`[1, kv_num_heads, 1, head_size]`), left alone when `"PER_TENSOR"` (a
    single broadcast scalar, already confirmed to need no slicing) or
    dynamic.

    A connected constant `attention_bias`/`attn_mask` (`GroupQueryAttention`'s
    own `attention_bias` at index 10, or the plain ai.onnx op's own
    `attn_mask` at index 3 -- see :func:`_match_gqa_producer`'s and
    :func:`_match_onnx_attention_producer`'s own docstrings), already
    confirmed at match time via :func:`_head_bias_axis` to resolve to
    either a genuine per-query-head tensor or a broadcast, is sliced along
    its own head axis by `keep_q_heads` -- *query*-head granularity (this
    input is added against Q*K' scores, laid out per query head, not per KV
    group) -- when it's the former, left untouched when it's the latter (or
    absent/dynamic). `GroupQueryAttention`'s own `head_sink` (index 11, no
    analogue on the plain ai.onnx op), a genuine `(num_heads,)` one-scalar-
    per-query-head constant (already confirmed at match time to be exactly
    that shape, or dynamic), is sliced the same way, directly by
    `keep_q_heads` with no `head_size` expansion.

    When `chain.packed_split_sizes` is set (a packed-QKV-then-Split chain,
    see :func:`_match_packed_qkv_split`), Q's/K's/V's "own separate weight"
    above is the *same* single packed tensor for all three, sliced exactly
    once by a combined column-index set instead of three independent
    per-producer slices, and the upstream `Split` node's own split-sizes
    constant is rewritten to the three new (post-pruning) column widths in
    the same Q-then-K-then-V order -- see the branch below. When
    `chain.q_norm_rope`/`chain.k_norm_rope` are additionally set (the
    CPU/DirectML Q/K-norm GQA export shape -- see
    :func:`_walk_back_through_qk_norm_rope`, :class:`_QKNormRopePassThrough`),
    each crossed hop's own `RotaryEmbedding` `num_heads` attribute (when
    nonzero) and post-norm `Reshape` target-width constant are rewritten to
    that branch's new post-pruning head count/flat width -- neither a
    crossed norm's own `scale` nor a `RotaryEmbedding`'s `cos_cache`/
    `sin_cache` ever needs touching (both confirmed head-independent -- see
    this module's own "Attention-head pruning" section comment).

    Returns ``(producer_weight_names, consumer_weight_name,
    stale_output_names)`` on success, or ``None`` if `sparsity` rounds to no
    groups dropped for this block (a no-op, left for the caller to skip).
    """
    h = chain.kv_num_heads
    keep_count = max(1, h - round(h * sparsity))
    if keep_count >= h:
        return None

    d = chain.head_size
    dv = chain.v_head_size
    group_size = chain.num_heads // chain.kv_num_heads

    # `_gqa_group_importance` (and any caller-supplied Wanda variant of it)
    # indexes columns as the head axis, mirroring
    # `_apply_one_plain_attention_chain`'s own `wq`/`wk`/`wv` convention --
    # so each array is brought to ``[K, N]`` (reduction dim first, head
    # columns last), the *opposite* of `_prune_weight`'s "output channel
    # first" `[N, K]` convention used elsewhere in this module: only
    # transpose when the raw storage is already `[N, K]` (Gemm transB=1).
    wq_init = initializer_map[chain.q_weight]
    wk_init = initializer_map[chain.k_weight]
    wv_init = initializer_map[chain.v_weight]
    if chain.packed_split_sizes is not None:
        # Packed QKV (see :func:`_match_packed_qkv_split`): `.q_weight`,
        # `.k_weight`, and `.v_weight` all name the *same* underlying
        # packed tensor (`wq_init is wk_init is wv_init`), one contiguous
        # `[K, Nq+Nk+Nv]` (or `[Nq+Nk+Nv, K]`, if `.q_weight_transposed`)
        # storage split Q-then-K-then-V by column, matching
        # `_match_packed_qkv_split`'s own confirmed split-sizes order -- so
        # `wq_kn`/`wk_kn`/`wv_kn` are column-range *views* into that one
        # ``[K, N]`` array, computed from the chain's own original (before
        # this call's own pruning) head counts, rather than three
        # independently-stored arrays.
        nq_orig = chain.num_heads * d
        nk_orig = chain.kv_num_heads * d
        w_kn = _to_f64(wq_init)
        if chain.q_weight_transposed:
            w_kn = w_kn.T  # [K, Nq+Nk+Nv]
        wq_kn = w_kn[:, :nq_orig]
        wk_kn = w_kn[:, nq_orig : nq_orig + nk_orig]
        wv_kn = w_kn[:, nq_orig + nk_orig :]
    else:
        wq_kn = _to_f64(wq_init)
        wk_kn = _to_f64(wk_init)
        wv_kn = _to_f64(wv_init)
        if chain.q_weight_transposed:
            wq_kn = wq_kn.T  # [K, Nq]
        if chain.k_weight_transposed:
            wk_kn = wk_kn.T  # [K, Nkv]
        if chain.v_weight_transposed:
            wv_kn = wv_kn.T  # [K, Nkv]

    importance = compute_group_importance(chain, wq_kn, wk_kn, wv_kn)
    keep_groups = np.sort(np.argsort(-importance)[:keep_count])

    keep_q_heads = np.concatenate(
        [np.arange(g * group_size, (g + 1) * group_size) for g in keep_groups]
    )
    # `q_idx`/`k_idx` index Q's/K's own producer weight columns (their
    # shared per-head width `d`); `v_idx` indexes V's own producer weight
    # columns at *its* own per-head width `dv`, which can differ. `y_idx`
    # indexes the *output* side instead -- the consumer's reduction
    # dimension and the raw output's own trailing axis (the reshape hop's
    # target shape, if any) -- both laid out per query head at V's own
    # per-head width `dv`, not Q's/K's `d`: `q_idx` (built with `d`) is only
    # coincidentally the right index set for those two when `dv == d`, which
    # is why the two were never distinguished before V could have its own
    # head_size.
    q_idx = _head_column_indices(keep_q_heads, d)
    k_idx = _head_column_indices(keep_groups, d)
    v_idx = _head_column_indices(keep_groups, dv)
    y_idx = _head_column_indices(keep_q_heads, dv)

    if chain.packed_split_sizes is not None:
        # One shared tensor: a single combined-column slice (Q's own
        # range, then K's shifted by the *original* `nq_orig`, then V's
        # shifted by `nq_orig + nk_orig`) rather than three independent
        # `_slice_producer_weight` calls, which would each invalidate the
        # column offsets the other two still need to read from the same
        # underlying storage.
        full_idx = np.concatenate([q_idx, k_idx + nq_orig, v_idx + nq_orig + nk_orig])
        _slice_producer_weight(wq_init, chain.q_weight_transposed, full_idx)
        if chain.q_bias is not None:
            _slice_last_axis(initializer_map[chain.q_bias], full_idx)
        sizes_init = initializer_map[chain.packed_split_sizes]
        sizes_init.CopyFrom(
            onnx.numpy_helper.from_array(
                np.array([len(q_idx), len(k_idx), len(v_idx)], dtype=np.int64),
                name=sizes_init.name,
            )
        )
    else:
        _slice_producer_weight(wq_init, chain.q_weight_transposed, q_idx)
        _slice_producer_weight(wk_init, chain.k_weight_transposed, k_idx)
        _slice_producer_weight(wv_init, chain.v_weight_transposed, v_idx)
        if chain.q_bias is not None:
            _slice_last_axis(initializer_map[chain.q_bias], q_idx)
        if chain.k_bias is not None:
            _slice_last_axis(initializer_map[chain.k_bias], k_idx)
        if chain.v_bias is not None:
            _slice_last_axis(initializer_map[chain.v_bias], v_idx)

    # `GroupQueryAttention`'s past_key/past_value live at input indices 3/4,
    # the plain ai.onnx op's own at 4/5 (see `_match_gqa_producer`'s and
    # `_match_onnx_attention_producer`'s own docstrings) -- both already
    # confirmed, at match time, to be either absent/dynamic (nothing to do
    # here) or a BNSH-format constant (plain FLOAT, or quantized with a
    # schema-conforming scale) safe to slice along axis 1 by `keep_groups`
    # (see :func:`_past_kv_constants_are_sliceable`).
    is_gqa = (
        chain.node.domain == _ATTENTION_DOMAIN
        and chain.node.op_type == "GroupQueryAttention"
    )
    past_kv_indices = (3, 4) if is_gqa else (4, 5)
    for idx in past_kv_indices:
        if len(chain.node.input) <= idx or not chain.node.input[idx]:
            continue
        past_init = initializer_map.get(chain.node.input[idx])
        if past_init is not None:
            _slice_axis1(past_init, keep_groups)

    # `k_scale`/`v_scale` (indices 12/13, `GroupQueryAttention`-only -- the
    # plain ai.onnx op's schema has no such inputs, see
    # `_match_onnx_attention_producer`'s own docstring) were already
    # confirmed at match time to be either absent/dynamic (nothing to do
    # here), a `"PER_TENSOR"` scalar broadcast (no per-head axis, left as-is
    # below), or a `"PER_CHANNEL"` `[1, kv_num_heads, 1, head_size]` float
    # constant -- the same axis-1 `kv_num_heads` layout as the cache tensor
    # itself -- safe to slice along axis 1 by the identical `keep_groups`
    # (see :func:`_past_kv_constants_are_sliceable`).
    if is_gqa:
        for idx in (12, 13):
            if len(chain.node.input) <= idx or not chain.node.input[idx]:
                continue
            scale_init = initializer_map.get(chain.node.input[idx])
            if scale_init is None:
                continue  # dynamic -- caller's own runtime data, left alone
            if len(scale_init.dims) == 4 and scale_init.dims[1] == h:
                _slice_axis1(scale_init, keep_groups)
            # else: PER_TENSOR broadcast scalar -- no per-head axis to slice

    # `attention_bias`/`attn_mask` (index 10 for `GroupQueryAttention`, 3
    # for the plain ai.onnx op -- see `_match_gqa_producer`'s and
    # `_match_onnx_attention_producer`'s own docstrings) is sliced along its
    # own head axis by `keep_q_heads` -- *query*-head granularity, not
    # `keep_groups`'s kv-group one -- whenever `_head_bias_axis` resolves it
    # to a genuine per-head tensor against the pre-pruning `chain.num_heads`;
    # left untouched when it resolves to a broadcast (or is absent/dynamic,
    # matched the same "nothing to slice" way here as at match time).
    mask_idx = 10 if is_gqa else 3
    if len(chain.node.input) > mask_idx and chain.node.input[mask_idx]:
        mask_init = initializer_map.get(chain.node.input[mask_idx])
        if mask_init is not None:
            axis = _head_bias_axis(list(mask_init.dims), chain.num_heads)
            if axis is not None and axis >= 0:
                _slice_axis(mask_init, keep_q_heads, axis)

    # `head_sink` (index 11, `GroupQueryAttention`-only): a genuine
    # `(num_heads,)` one-scalar-per-query-head constant, already confirmed
    # at match time to be exactly that shape (or dynamic) -- sliced
    # directly by `keep_q_heads`, no `head_size` expansion needed.
    if is_gqa and len(chain.node.input) > 11 and chain.node.input[11]:
        sink_init = initializer_map.get(chain.node.input[11])
        if sink_init is not None:
            _slice_last_axis(sink_init, keep_q_heads)

    new_kv_num_heads = keep_count
    new_num_heads = keep_count * group_size
    for attr in chain.node.attribute:
        if attr.name == chain.num_heads_attr:
            attr.i = new_num_heads
        elif attr.name == "kv_num_heads":
            attr.i = new_kv_num_heads

    _slice_consumer_weight(
        initializer_map[chain.consumer_weight],
        chain.consumer_weight_transposed,
        y_idx,
    )

    for _, shape_name in chain.chain_ops:
        if shape_name is not None:
            shape_init = initializer_map[shape_name]
            dims = onnx.numpy_helper.to_array(shape_init).copy()
            dims[-1] = new_num_heads * dv
            shape_init.CopyFrom(
                onnx.numpy_helper.from_array(dims, name=shape_init.name)
            )

    # Q's/K's own per-head norm + RoPE pass-through, if either branch
    # crossed one (see :class:`_QKNormRopePassThrough`'s own docstring for
    # why neither a crossed norm's own `scale` nor a crossed
    # `RotaryEmbedding`'s `cos_cache`/`sin_cache` ever needs touching --
    # only these two constants do): `new_branch_width`/`new_branch_heads`
    # are Q's own post-pruning `num_heads`/flat width for `.q_norm_rope`,
    # K's own post-pruning `kv_num_heads`/flat width (K's own head_size
    # equals Q's/K's shared `d`, already confirmed at match time) for
    # `.k_norm_rope`.
    for hop, new_branch_heads, new_branch_width in (
        (chain.q_norm_rope, new_num_heads, new_num_heads * d),
        (chain.k_norm_rope, new_kv_num_heads, new_kv_num_heads * d),
    ):
        if hop is None:
            continue
        if hop.rotary_node is not None:
            for attr in hop.rotary_node.attribute:
                if attr.name == "num_heads" and attr.i != 0:
                    attr.i = new_branch_heads
        if hop.reshape_shape is not None:
            shape_init = initializer_map[hop.reshape_shape]
            dims = onnx.numpy_helper.to_array(shape_init).copy()
            dims[-1] = new_branch_width
            shape_init.CopyFrom(
                onnx.numpy_helper.from_array(dims, name=shape_init.name)
            )

    stale = {chain.node.output[0]}
    stale.update(op.output[0] for op, _ in chain.chain_ops)
    for hop in (chain.q_norm_rope, chain.k_norm_rope):
        if hop is not None:
            stale.update(n.output[0] for n in hop.nodes if n.output and n.output[0])
    return (
        {chain.q_weight, chain.k_weight, chain.v_weight},
        chain.consumer_weight,
        stale,
    )


def _apply_attention_chains(
    graph: onnx.GraphProto,
    chains: List[_AttnLikeChain],
    sparsity: float,
    compute_importance,
    compute_group_importance,
) -> None:
    """Shared body for :func:`apply_attention_head_pruning` and
    :func:`apply_attention_head_wanda_pruning`, mirroring
    :func:`_apply_chains`'s own shape (cross-chain touched-role
    bookkeeping, stale ``value_info`` cleanup) but dispatching each chain to
    :func:`_apply_one_plain_attention_chain` (a matched ``Attention``
    block) or :func:`_apply_one_gqa_chain` (a matched
    ``GroupQueryAttention`` block) for the actual per-chain slicing, since
    the two ops' weight layouts (one merged QKV tensor vs. three separate
    Q/K/V producers) and pruning unit (individual head vs. whole KV group)
    are different enough that sharing that part wouldn't simplify either.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    producer_touched: Set[str] = set()
    consumer_touched: Set[str] = set()
    stale_value_info: Set[str] = set()

    for chain in chains:
        if isinstance(chain, _GQAChain):
            producer_names = {chain.q_weight, chain.k_weight, chain.v_weight}
        else:
            producer_names = {chain.weight}
        if (
            producer_names & producer_touched
            or chain.consumer_weight in consumer_touched
        ):
            continue

        applied: Optional[Tuple[Set[str], str, Set[str]]]
        if isinstance(chain, _GQAChain):
            applied = _apply_one_gqa_chain(
                initializer_map, chain, sparsity, compute_group_importance
            )
        else:
            applied = _apply_one_plain_attention_chain(
                initializer_map, chain, sparsity, compute_importance
            )
        if applied is None:
            continue

        touched_producers, touched_consumer, stale = applied
        producer_touched.update(touched_producers)
        consumer_touched.add(touched_consumer)
        stale_value_info.update(stale)

    if stale_value_info:
        kept = [vi for vi in graph.value_info if vi.name not in stale_value_info]
        del graph.value_info[:]
        graph.value_info.extend(kept)


def apply_attention_head_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
) -> onnx.ModelProto:
    """Removes whole attention heads -- or, for grouped-query attention,
    whole KV groups -- from every matched ``com.microsoft::Attention``,
    ``com.microsoft::GroupQueryAttention``, or plain ``ai.onnx::Attention``
    node (the fused self-attention blocks onnxsim's own
    ``fuse_attention``/``fuse_gqa`` optimizer passes produce, plus the
    standard ONNX op those two contrib ops are converging towards -- see
    this module's own "Attention-head pruning" section comment for the real
    schema each was confirmed against and how) whose output feeds,
    optionally through a single shape-preserving ``Reshape``, exactly one
    downstream MatMul/vanilla-Gemm's reduction dimension (the output
    projection) -- the attention analogue of :func:`apply_structured_pruning`,
    at head (or KV-group) instead of single-channel granularity.

    For each matched plain ``com.microsoft::Attention`` block: ranks every
    head by the combined Frobenius norm of its own
    ``[hidden_size, head_size]`` Q, K, and V weight columns, drops the
    lowest-``sparsity``-fraction of heads (at least one head is always
    kept), and removes the corresponding column blocks from the merged QKV
    weight (and bias, if present), decrementing
    ``num_heads``/``qkv_hidden_sizes`` accordingly, and the matching row
    block from the output projection's weight -- mathematically unaffected
    for every surviving head, the same guarantee
    :func:`apply_structured_pruning` gives per channel.

    For each matched ``GroupQueryAttention`` or plain ``ai.onnx::Attention``
    block: ranks every *KV group* (a KV head and the
    ``num_heads / kv_num_heads`` query heads the kernel maps to it) by the
    combined Frobenius norm of that group's own Q+K+V weight block across
    Q's, K's, and V's own separate producer weights, drops the
    lowest-``sparsity``-fraction of groups (at least one group is always
    kept), and removes the corresponding column blocks from all three
    producers (and their biases, if present) together with the matching row
    block from the output projection's weight, decrementing the query head
    count (``num_heads`` for `GroupQueryAttention`, ``q_num_heads`` for the
    plain ``ai.onnx`` op) and ``kv_num_heads`` by the number of groups
    dropped -- so their ratio (query heads per KV head) is unchanged,
    keeping every surviving KV head mapped to exactly the same number of
    query heads the kernel requires. An individual query head is never
    dropped on its own: only a whole group, since neither kernel has a way
    to keep a KV head alive for some, but not all, of the query heads that
    shared it. A connected `past_key`/`past_value` that is a constant of the
    expected float BNSH shape is sliced along its own `kv_num_heads` axis by
    the same index set (see :func:`_past_kv_constants_are_sliceable`,
    :func:`_apply_one_gqa_chain`); a plain ``ai.onnx::Attention`` node's V
    head size may genuinely differ from Q/K's own (a shape that op's schema
    allows but `GroupQueryAttention` never produces) and is sliced correctly
    at its own width -- see :class:`_GQAChain`'s own `v_head_size` field.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched block's heads (or, for
            GroupQueryAttention/plain ai.onnx Attention, KV groups) to
            remove (at least one is always kept)
    :param importance_norm: ``"l2"`` (default, unchanged from before this
            parameter existed) ranks by the combined Frobenius (L2) norm of
            each head's/KV group's own weight block, mirroring
            :func:`apply_structured_pruning`'s own default; ``"l1"`` ranks
            by the sum of absolute weight magnitude across that same block
            instead -- see :func:`_plain_attention_head_importance`'s and
            :func:`_gqa_group_importance`'s own comments for exactly how
            each combines across a KV group's several Q/K/V producers
    :returns: ``model`` with every matched block's tensors resized in
            place -- including, when present and a constant of the
            schema-documented broadcastable shape, `com.microsoft::Attention`'s/
            `GroupQueryAttention`'s own `attention_bias` input, `GroupQueryAttention`'s
            own `head_sink`, and the plain ai.onnx op's own `attn_mask`, each
            sliced along its own genuine per-head axis in lockstep with
            every other pruned tensor whenever that axis is present and
            resolves, statically, to the pre-pruning head count (see
            :func:`_head_bias_axis`); anything not matching that exact
            topology (a non-constant weight, a packed-QKV
            GroupQueryAttention node, a GroupQueryAttention/plain ai.onnx
            Attention node with a non-empty constant past-KV-cache of
            unexpected shape/dtype (e.g. a quantized KV cache) or a
            non-empty constant attention-bias/mask/head_sink input whose
            shape doesn't cleanly resolve to either "genuinely per-head" or
            "broadcast, head-count-independent", an ai.onnx Attention node
            without explicit ``q_num_heads``/``kv_num_heads`` attributes or
            with Q's/K's own head sizes mismatched, a consumer whose
            reduction dimension
            doesn't line up, ...) is left completely untouched

    Every matched weight (merged QKV, or separate Q/K/V), bias, and
    per-head constant (`attention_bias`/`attn_mask`/`head_sink`) may be
    FLOAT, FLOAT16, or BFLOAT16, independently: importance ranking reads a
    matched weight upcast to float64 (:func:`_to_f64`), while every actual
    removal is pure index slicing that preserves each tensor's own original
    dtype untouched -- see the "FP16/BFloat16 weight support" section
    comment above :func:`_match_conv_weight_only`.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains: List[_AttnLikeChain] = [
        *_find_attention_chains(graph),
        *_find_gqa_chains(graph),
        *_find_onnx_attention_chains(graph),
    ]
    if chains:
        _apply_attention_chains(
            graph,
            chains,
            sparsity,
            lambda chain, wq, wk, wv, dq, dk, dv: _plain_attention_head_importance(
                chain, wq, wk, wv, dq, dk, dv, importance_norm
            ),
            lambda chain, wq, wk, wv: _gqa_group_importance(
                chain, wq, wk, wv, importance_norm
            ),
        )

    return out


def _wanda_attention_calibration_stats(
    out: onnx.ModelProto,
    chains: List[_AttnLikeChain],
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]],
) -> Dict[str, np.ndarray]:
    """Runs `out` over `calibration_data` and returns the per-probe-point
    `act_norm` dict :func:`apply_attention_head_wanda_pruning`'s own
    `_wanda_attention_head_importance`/`_wanda_gqa_group_importance`
    closures look up by `chain.consumer_node.input[0]` -- the same
    computation :func:`apply_attention_head_wanda_pruning`'s own body used
    to perform inline before this function existed. Factored out,
    read-only (never mutates `out`), so
    :func:`analyze_pruning_sensitivity`'s own dry-run report can compute
    the *exact* same activation norms :func:`apply_attention_head_wanda_pruning`
    would, from one single shared implementation -- the same
    "one place real duplication would have crept in" extraction this
    module's own "Dry-run pruning sensitivity analysis" section comment
    documents for :func:`_wanda_unstructured_calibration_stats`/
    :func:`_wanda_structured_calibration_stats`.

    `out` is expected to already be the caller's own working copy, exactly
    as :func:`_wanda_unstructured_calibration_stats` expects.
    """
    probe_names = sorted({chain.consumer_node.input[0] for chain in chains})
    probe_model = _add_probe_outputs(out, probe_names)

    sq_sum: Dict[str, np.ndarray] = {}
    count: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            x = np.asarray(result[name], dtype=np.float64)
            if x.ndim < 1:
                continue
            reduce_axes = tuple(range(x.ndim - 1))
            s = np.square(x).sum(axis=reduce_axes) if reduce_axes else np.square(x)
            cnt = int(np.prod(x.shape[:-1], dtype=np.int64)) if x.ndim > 1 else 1
            sq_sum[name] = s if name not in sq_sum else sq_sum[name] + s
            count[name] = count.get(name, 0) + cnt

    return {name: np.sqrt(s / max(count[name], 1)) for name, s in sq_sum.items()}


def apply_attention_head_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """The calibrated upgrade of :func:`apply_attention_head_pruning`,
    exactly as :func:`apply_structured_wanda_pruning` is to
    :func:`apply_structured_pruning`: same real head (or, for
    GroupQueryAttention/plain ai.onnx Attention, whole-KV-group) removal,
    same topology matching, including the identical `attention_bias`/
    `attn_mask`/`head_sink` per-head-axis slicing treatment (see
    :func:`apply_attention_head_pruning`'s own docstring for the three
    matched op types and exactly what each does with those inputs), but
    each unit's importance
    is ``||W||_F * ||X||_2`` -- the plain Frobenius-norm weight score times
    the combined (root-sum-square) activation norm of that unit's own slice
    of the *output projection's* input, captured over calibration data --
    instead of weight magnitude alone. For a plain ``com.microsoft::Attention``
    block this is per head, exactly as before; for a ``GroupQueryAttention``
    or plain ``ai.onnx::Attention`` block the activation norm is combined
    (root-sum-square) over every query head a KV group owns, mirroring how
    :func:`_gqa_group_importance` combines that same group's weight norm
    across Q+K+V -- both matched separate-Q/K/V-producer ops share that one
    importance function (and this one calibrated wrapper around it)
    unmodified.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            block's output-projection-side activation norm on. Each batch
            is a ``{input_name: np.ndarray}`` dict matching ``model``'s
            graph inputs -- see
            :func:`onnxsim.generate_random_calibration_data` (the default
            when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched block's heads (or, for
            GroupQueryAttention/plain ai.onnx Attention, KV groups) to
            remove (at least one is always kept)
    :param importance_norm: ``"l2"`` (default) or ``"l1"`` -- selects the
            *weight*-magnitude term only, exactly as it does for
            :func:`apply_attention_head_pruning`; the *activation*-norm term
            stays L2 unconditionally either way, per Wanda's own
            ``|W_ij| * ||X_j||_2`` definition (see
            :func:`apply_structured_wanda_pruning`'s own ``importance_norm``
            for the same point made there)
    :param epsilon: floor applied to the accumulated per-unit activation
            norm, avoiding every unit of an all-zero activation tying at
            exactly the weight-only importance
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with every matched block's tensors resized in
            place; anything not matching that exact topology falls back to
            :func:`apply_attention_head_pruning`'s plain Frobenius-norm
            ranking if no matching activation was ever observed for that
            block's consumer

    Weight dtype support is identical to
    :func:`apply_attention_head_pruning` (FLOAT/FLOAT16/BFLOAT16); the
    calibration activations captured here are, like every other Wanda-style
    pass in this module, read via a real ``onnxruntime`` run and cast to
    float64 on capture regardless of the graph's own declared dtype.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains: List[_AttnLikeChain] = [
        *_find_attention_chains(graph),
        *_find_gqa_chains(graph),
        *_find_onnx_attention_chains(graph),
    ]
    if not chains:
        return out

    act_norm = _wanda_attention_calibration_stats(
        out, chains, calibration_data, providers
    )

    def _wanda_attention_head_importance(chain, wq, wk, wv, dq, dk, dv):
        base = _plain_attention_head_importance(
            chain, wq, wk, wv, dq, dk, dv, importance_norm
        )
        norm = act_norm.get(chain.consumer_node.input[0])
        if norm is None or norm.shape[0] != chain.nv:
            return base  # no matching activation observed -- fall back to plain
        act_head = np.array(
            [
                np.linalg.norm(norm[h * dv : (h + 1) * dv])
                for h in range(chain.num_heads)
            ]
        )
        return base * np.maximum(act_head, epsilon)

    def _wanda_gqa_group_importance(chain, wq, wk, wv):
        base = _gqa_group_importance(chain, wq, wk, wv, importance_norm)
        norm = act_norm.get(chain.consumer_node.input[0])
        # The probed activation is the consumer's own input -- the attention
        # output, laid out per *query* head at *V's* own per-head width
        # (`chain.v_head_size`), the same `dv` :func:`_apply_one_gqa_chain`
        # itself uses for that tensor's own indexing (equal to
        # `chain.head_size` for `GroupQueryAttention`, not necessarily for
        # the plain ai.onnx op -- see :class:`_GQAChain`'s own `v_head_size`
        # field).
        dv = chain.v_head_size
        width = chain.num_heads * dv
        if norm is None or norm.shape[0] != width:
            return base  # no matching activation observed -- fall back to plain
        group_size = chain.num_heads // chain.kv_num_heads
        act_group = np.array(
            [
                np.linalg.norm(norm[kv * group_size * dv : (kv + 1) * group_size * dv])
                for kv in range(chain.kv_num_heads)
            ]
        )
        return base * np.maximum(act_group, epsilon)

    _apply_attention_chains(
        graph,
        chains,
        sparsity,
        _wanda_attention_head_importance,
        _wanda_gqa_group_importance,
    )
    return out


# --- MoE expert-intermediate-channel pruning --------------------------------
#
# See this module's own docstring for why this is the one narrow slice of
# NVIDIA's "Iterative Puzzle" hybrid-MoE-LLM pipeline (arXiv:2607.04371) that
# turns out to be tractable here, and why Mamba-state pruning (the paper's
# other named technique) is not: empirically, against this environment's own
# onnxruntime 1.29.0 contrib-op schema registry
# (``onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()``),
# ``MoE``/``QMoE`` are the only two ``*MoE*``-named schemas, and nothing
# matches ``*amba*``/``*SSM*`` at all (only the unrelated, standard-ONNX
# ``Scan`` op) -- there is no fused Mamba/SSM op whose weight tensors this
# pass could target the way it targets `MoE`'s own, and an arbitrarily
# decomposed selective-scan subgraph has no single bounded node pattern to
# recognize in the first place, the same bar this module's own docstring
# already holds every other declined topology to.
#
# `com.microsoft::MoE`'s own schema (confirmed both by reading onnxsim's own
# `contrib_schemas.cpp`, which registers this exact schema from ONNX
# Runtime's `docs/ContribOperators.md`/`contrib_defs.cc`, and by querying
# onnxruntime's live schema registry directly in this environment) gives
# `fc1_experts_weights`/`fc2_experts_weights` (and the optional
# `fc3_experts_weights`) a clean ``[num_experts, ...]`` leading dimension,
# with `inter_size` -- the expert FFN's own intermediate width, exactly what
# the paper's "expert-intermediate-channel" pruning targets -- living on
# `fc1`'s axis 1 (``[num_experts, inter_size, hidden_size]``) and `fc2`'s
# axis 2 (``[num_experts, hidden_size, inter_size]``). Unlike *removing
# whole experts* (which would also need `num_experts`/`k`-consistency
# bookkeeping and a safe way to resize `router_probs`' own upstream
# producer -- a second, independent MatMul/Gemm this pass would need to walk
# to and prove has no other consumer, real but strictly more machinery), or
# a *plain per-2-D-slice* channel cut (which `_candidates`'s existing
# MatMul/Gemm matching already doesn't reach, since these weights are 3-D
# and rank-3-input-shaped, not the op's own reduction dimension), pruning
# `inter_size` uniformly across every expert at once needs nothing outside
# the node's own five tensors: no upstream producer, no downstream consumer,
# no attribute to update (`k`/`num_experts`/`activation_type` are all
# unaffected -- the node's own *output* shape doesn't change either, since
# it always equals `input`'s), and no cross-node dependency-graph walk at
# all. That is what makes this the "narrowest safe slice" of the paper's
# own technique this module can precisely justify, rather than a fragile
# stand-in for it: every one of `fc1`/`fc2`(`/fc3`)'s three axes plays
# exactly one of {expert, in, out} unambiguously, per the schema's own
# documented shapes, and dropping the same `inter_size` index from all of
# them together is provably shape-consistent (`fc1`'s and `fc3`'s own output
# rows, `fc2`'s own input columns) -- the same "remove a row from one
# producer's weight and the matching column from its one consumer's weight"
# argument :func:`apply_structured_pruning` already makes for an ordinary
# 2-D MatMul/Gemm chain, just carried out on three tensors sharing one
# extra, untouched leading `num_experts` axis instead of two 2-D tensors,
# with no elementwise hop in between to walk through (the node's own fused
# semantics already are that hop).
#
# `fc3_experts_weights` (the Mixtral-style separate gate/up/down projection
# `com.microsoft::MoE`'s own docstring, and `contrib_schemas.cpp`'s
# `BuildMoEFunctionBody`, both already document -- see this module's own
# docstring for the exact composition) shares `fc1`'s own row-output-channel
# role along `inter_size`, and would in principle prune identically -- but
# this pass declines any node with `fc3_experts_weights` present rather than
# support it: empirically (`onnxruntime.InferenceSession` construction
# against this environment's own CPU execution provider, opset 18, MoE
# opset 1), ONNX Runtime's *CPU* MoE kernel raises "FC3 is not implemented
# for CPU MoE" unconditionally, for every `activation_type` -- exactly the
# same "disclosed validation gap" `contrib_schemas.cpp`'s own comment names
# for its `fc3` decomposition. Every other case this pass *does* support was
# checked, in this same environment, to actually execute end to end on the
# CPU provider first (`relu`/`identity`/`silu`/`gelu`, no `fc3`) -- so
# rather than build a code path this environment has no real runtime to
# validate against, `fc3_experts_weights` stays out of scope, the same
# conservative call every other declined case in this module makes. The
# `swiglu` activation is declined for the same reason from the opposite
# direction: it doubles `fc1_experts_weights`' own row count (`fusion_size`
# 2, "fused and interleaved" -- and CPU MoE construction fails outright
# without `swiglu_fusion=1`, confirmed the same empirical way), so its own
# ``inter_size``-indexed rows are no longer one contiguous, ``inter_size``
# long per-expert block this pass's own channel-index slicing could safely
# reach without unpacking the interleaving first; :func:`_match_moe_producer`
# doesn't even need to special-case this activation explicitly -- the
# ``fc1_experts_weights.dims[1] == fc2_experts_weights.dims[2]`` shape check
# it already runs for every candidate fails to hold the moment `fusion_size`
# is 2, so the doubled-width shape declines itself, the same
# shape-consistency-not-attribute-value style :func:`_conv_spatial_attrs`'s
# own ``kernel_shape`` cross-check already uses elsewhere in this module.
#
# `router_probs` (the node's own second input) is never touched by this
# pass at all -- `num_experts` doesn't change, so neither its own shape nor
# its upstream producer's weight needs to.


@dataclass(frozen=True)
class _MoEChain:
    node: onnx.NodeProto
    fc1_w: str
    fc1_b: Optional[str]
    fc2_w: str
    fc2_b: Optional[str]
    num_experts: int
    inter_size: int
    hidden_size: int


_MOE_DOMAIN = "com.microsoft"
_MOE_ACTIVATIONS = {"relu", "identity", "silu", "gelu"}


def _match_moe_producer(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
) -> Optional[_MoEChain]:
    """If `node` is a ``com.microsoft::MoE`` node this pass can safely prune
    the `inter_size` axis of, returns the matched :class:`_MoEChain`. See
    this section's own comment above for the exact safety argument and why
    each decline below is drawn where it is; every check here mirrors one
    piece of that argument, in the same order:

    - `activation_type` (default ``"relu"`` per the op's own schema) must be
      one of `relu`/`identity`/`silu`/`gelu` -- `swiglu` is declined (see
      above; caught structurally by the shape check below, not this
      attribute check, but ruled out here too for a node whose declared
      activation is unrecognized entirely, e.g. a future value this pass has
      never seen).
    - `fc3_experts_weights` (input 6) must be absent -- no CPU execution
      provider in this environment implements it (see above); a present-but-
      empty-string placeholder is treated the same as absent.
    - `fc1_experts_weights`/`fc2_experts_weights` must both be constant
      float32 initializers, rank 3, with `fc1`'s ``(num_experts, inter_size,
      hidden_size)`` and `fc2`'s ``(num_experts, hidden_size, inter_size)``
      agreeing on both `num_experts` and `inter_size` -- the one shape check
      that also rules out a fused/interleaved `swiglu` `fc1` (whose own row
      count is ``2 * inter_size``, never equal to `fc2`'s own column count).
    - each is required to have exactly one consumer (this node) -- a
      tied/shared weight reused elsewhere would be corrupted by an in-place
      resize, the same tied-weight guard every other chain-matcher in this
      module already applies via `consumers_of`.
    - `fc1_experts_bias`/`fc2_experts_bias` (inputs 3/5), if present, get the
      same float32/initializer/single-consumer checks, plus an exact
      ``(num_experts, inter_size)``/``(num_experts, hidden_size)`` shape
      match -- `fc2`'s own bias indexes `hidden_size` (its *output* axis),
      not `inter_size`, so it is matched here (to confirm it really is a
      well-formed MoE node) but is never itself sliced by this pass.
    """
    if node.domain != _MOE_DOMAIN or node.op_type != "MoE":
        return None

    activation = "relu"
    swiglu_fusion = 0
    for attr in node.attribute:
        if attr.name == "activation_type":
            activation = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
        elif attr.name == "swiglu_fusion":
            swiglu_fusion = attr.i
    if activation not in _MOE_ACTIVATIONS or swiglu_fusion != 0:
        return None

    if len(node.input) > 6 and node.input[6]:
        return None  # fc3_experts_weights present -- no CPU oracle, see above

    if len(node.input) < 5 or not node.input[2] or not node.input[4]:
        return None
    fc1_w_name, fc2_w_name = node.input[2], node.input[4]
    fc1_w = initializer_map.get(fc1_w_name)
    fc2_w = initializer_map.get(fc2_w_name)
    if (
        fc1_w is None
        or fc2_w is None
        or not _is_supported_float_dtype(fc1_w.data_type)
        or not _is_supported_float_dtype(fc2_w.data_type)
        or len(fc1_w.dims) != 3
        or len(fc2_w.dims) != 3
        or len(consumers_of.get(fc1_w_name, [])) != 1
        or len(consumers_of.get(fc2_w_name, [])) != 1
    ):
        return None
    num_experts, inter_size, hidden_size = list(fc1_w.dims)
    if list(fc2_w.dims) != [num_experts, hidden_size, inter_size]:
        return None  # also rules out a fused-swiglu fc1 (doubled row count)

    def _optional_bias(
        index: int, expected_dims: List[int]
    ) -> Tuple[bool, Optional[str]]:
        if index >= len(node.input) or not node.input[index]:
            return True, None
        name = node.input[index]
        init = initializer_map.get(name)
        if (
            init is None
            or not _is_supported_float_dtype(init.data_type)
            or list(init.dims) != expected_dims
            or len(consumers_of.get(name, [])) != 1
        ):
            return False, None
        return True, name

    ok, fc1_b_name = _optional_bias(3, [num_experts, inter_size])
    if not ok:
        return None
    ok, fc2_b_name = _optional_bias(5, [num_experts, hidden_size])
    if not ok:
        return None

    return _MoEChain(
        node=node,
        fc1_w=fc1_w_name,
        fc1_b=fc1_b_name,
        fc2_w=fc2_w_name,
        fc2_b=fc2_b_name,
        num_experts=int(num_experts),
        inter_size=int(inter_size),
        hidden_size=int(hidden_size),
    )


def _find_moe_chains(graph: onnx.GraphProto) -> List[_MoEChain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    chains = []
    for node in graph.node:
        chain = _match_moe_producer(node, initializer_map, consumers_of)
        if chain is not None:
            chains.append(chain)
    return chains


def _moe_importance(
    chain: _MoEChain, initializer_map: Dict[str, onnx.TensorProto]
) -> np.ndarray:
    """Combined (root-sum-square) L2-norm importance per `inter_size`
    channel index, accumulating squared norms -- in float64, the same
    accumulate-then-sqrt-once precision convention
    :func:`_plain_structured_importance` already uses -- over every expert
    and the full `hidden_size` axis at once: index `j` is one shared row of
    `fc1` (and, if present, `fc1_experts_bias`) and one shared column of
    `fc2`, across *every* expert simultaneously (the node's own
    ``[num_experts, inter_size, ...]``/``[num_experts, ..., inter_size]``
    layout gives every expert the identical `inter_size` axis, with no
    independent per-expert choice possible -- exactly why `keep` is computed
    once here and applied to every expert's own slice alike, mirroring how
    :func:`apply_structured_pruning`'s own gated-pair combination ranks both
    branches by one shared score before picking one shared `keep`).
    """
    fc1_w = _to_f64(initializer_map[chain.fc1_w])
    fc2_w = _to_f64(initializer_map[chain.fc2_w])
    squared = np.sum(np.square(fc1_w), axis=(0, 2)) + np.sum(
        np.square(fc2_w), axis=(0, 1)
    )
    if chain.fc1_b is not None:
        fc1_b = _to_f64(initializer_map[chain.fc1_b])
        squared = squared + np.sum(np.square(fc1_b), axis=0)
    return np.sqrt(squared)


def _apply_moe_chains(
    graph: onnx.GraphProto,
    chains: List[_MoEChain],
    sparsity: float,
    compute_importance,
) -> None:
    initializer_map = {t.name: t for t in graph.initializer}
    touched: Set[str] = set()

    for chain in chains:
        weight_names = {chain.fc1_w, chain.fc2_w}
        if chain.fc1_b is not None:
            weight_names.add(chain.fc1_b)
        if weight_names & touched:
            continue  # a shared/tied initializer another MoE node already resized
        touched |= weight_names

        n = chain.inter_size
        keep_count = max(1, n - round(n * sparsity))
        if keep_count >= n:
            continue  # rounds down to nothing for this layer -- no-op

        importance = compute_importance(chain, initializer_map)
        keep = np.sort(np.argsort(-importance)[:keep_count])

        fc1_init = initializer_map[chain.fc1_w]
        fc1_w_new = onnx.numpy_helper.to_array(fc1_init)[:, keep, :]
        fc1_init.CopyFrom(
            onnx.numpy_helper.from_array(
                np.ascontiguousarray(fc1_w_new), name=chain.fc1_w
            )
        )

        fc2_init = initializer_map[chain.fc2_w]
        fc2_w_new = onnx.numpy_helper.to_array(fc2_init)[:, :, keep]
        fc2_init.CopyFrom(
            onnx.numpy_helper.from_array(
                np.ascontiguousarray(fc2_w_new), name=chain.fc2_w
            )
        )

        if chain.fc1_b is not None:
            fc1_b_init = initializer_map[chain.fc1_b]
            fc1_b_new = onnx.numpy_helper.to_array(fc1_b_init)[:, keep]
            fc1_b_init.CopyFrom(
                onnx.numpy_helper.from_array(
                    np.ascontiguousarray(fc1_b_new), name=chain.fc1_b
                )
            )
        # fc2_experts_bias indexes hidden_size, fc2's own *output* axis --
        # unaffected by an inter_size cut, so it is never sliced here.


def apply_moe_expert_channel_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
) -> onnx.ModelProto:
    """Removes intermediate (`inter_size`) channels from every expert of a
    matched ``com.microsoft::MoE`` node at once -- real structural pruning
    (smaller `fc1`/`fc2` weight tensors, smaller per-expert matmuls on any
    runtime) of the one slice of NVIDIA's "Iterative Puzzle" hybrid-MoE-LLM
    pipeline's (arXiv:2607.04371) "expert-intermediate-channel" pruning this
    module can precisely prove safe from the graph alone -- see this
    section's own comment above for the full safety argument, exactly which
    nodes are matched (:func:`_match_moe_producer`), and why `fc3`/`swiglu`
    and whole-expert removal are out of scope.

    Ranks every `inter_size` index by combined (root-sum-square) L2 norm of
    `fc1_experts_weights`' own row (across every expert and `hidden_size`
    at once) and `fc2_experts_weights`' own column (same reduction), plus
    `fc1_experts_bias`'s own entry when present -- the same L2-norm
    importance :func:`apply_structured_pruning` already uses, transplanted
    from a single 2-D producer/consumer pair to `MoE`'s own batched 3-D
    ``[num_experts, ...]`` weights -- drops the lowest-``sparsity``-fraction
    of indices (at least one is always kept), and removes the matching row
    from `fc1_experts_weights`/`fc1_experts_bias` and column from
    `fc2_experts_weights`, identically across every expert. `num_experts`,
    `k`, and every node attribute are untouched -- pruning `inter_size`
    changes no other tensor's shape anywhere in the graph, including the
    node's own output (always equal to `input`'s shape).

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched node's `inter_size`
            channels to remove (at least one channel is always kept)
    :returns: ``model`` with every matched ``MoE`` node's `fc1`/`fc2`
            tensors resized in place; a node with `fc3_experts_weights`, a
            `swiglu`/unrecognized `activation_type`, a non-constant or
            tied/shared weight, or any other shape this pass doesn't
            recognize is left completely untouched

    `fc1_experts_weights`/`fc2_experts_weights`/`fc1_experts_bias` may be
    FLOAT, FLOAT16, or BFLOAT16 (:func:`_moe_importance` reads them upcast
    to float64 for the norm computation above; the actual channel removal
    in :func:`_apply_moe_chains` is pure index slicing, which preserves
    each tensor's own original dtype with no separate downcast needed --
    see the "FP16/BFloat16 weight support" section comment above
    :func:`_match_conv_weight_only`).
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = _find_moe_chains(graph)
    if chains:
        _apply_moe_chains(graph, chains, sparsity, _moe_importance)
    return out


# --- MoE whole-expert pruning ------------------------------------------------
#
# The complementary half of :func:`apply_moe_expert_channel_pruning`: instead
# of narrowing every expert's own `inter_size`, this shrinks the `num_experts`
# leading axis itself -- dropping entire experts. This module's own docstring
# scopes :func:`apply_moe_expert_channel_pruning` to the "expert-intermediate-
# channel" half of the "Iterative Puzzle" paper (arXiv:2607.04371) precisely
# *because* whole-expert removal needs strictly more machinery: `num_experts`/
# `k`-consistency bookkeeping and a safe way to resize `router_probs`' own
# upstream producer (see that function's own section comment). This section
# builds that machinery -- following the general post-training MoE-pruning
# literature's standard technique (calibration-based expert-usage ranking,
# e.g. dropping the experts with the lowest average router gate weight over a
# representative batch -- the "prune-by-usage" family the task's own
# background cites), not any single paper's exact recipe -- and confirms,
# empirically, that the result is exactly safe once four separate questions
# are answered:
#
# 1. *What does `router_probs` actually hold, and how does `num_experts`
#    shrinking affect it?* Confirmed by reading `contrib_schemas.cpp`'s own
#    `MakeMoESchema()` (which itself transcribes ONNX Runtime's
#    `docs/ContribOperators.md`/`contrib_defs.cc`): despite the name,
#    `router_probs` (`MoE`'s own input 1) holds raw per-token, per-expert
#    routing *logits* -- a Softmax is applied *internally*, over the
#    `num_experts` axis, before top-k selection. That Softmax is exactly why
#    whole-expert removal is safe to express as *shrinking* the axis rather
#    than needing some separate "disable this expert" signal: shrinking
#    `router_probs`' own width to `num_experts_kept` changes the Softmax's own
#    denominator to sum over only the surviving experts, identically to what
#    forcing the dropped experts' logits to `-inf` in the *original*-width
#    model would do (`exp(-inf) == 0`, dropping out of both the Softmax
#    numerator and denominator). Verified directly against a real CPU
#    `onnxruntime.InferenceSession`: a same-shape "masking" oracle (dropped
#    experts' `fc1`/`fc2` zeroed, their router weight column's bias forced to
#    `-1e9`) matches an actually-shrunk-`num_experts` model's output to
#    *exactly* 0.0 max-abs-diff, both for the schema's default
#    `normalize_routing_weights=0` and for `normalize_routing_weights=1`
#    (which renormalizes the selected top-k weights to sum to 1 -- dropping
#    an expert that was in some token's top-k changes who else is, which
#    changes that renormalization too, and the two models still agree
#    exactly, since ONNX Runtime's own top-k+renormalize logic never sees the
#    dropped expert as a candidate in the pruned model either). So this pass
#    needs no separate bookkeeping for `router_probs`' own *values* -- slicing
#    its upstream producer's weight (see point 3) to the same kept-expert
#    index set is the entire correctness argument, the same
#    "shared keep set across multiple weights" pattern the rest of this
#    module already uses for gated FFN pairs and residual merge groups.
#
# 2. *Does `k` need adjusting when `num_experts` shrinks?* Empirically
#    confirmed against a real CPU `onnxruntime.InferenceSession`: `k` is a
#    *required* attribute (`contrib_defs.cc`'s own `.Attr("k", ..., INT,
#    /*required=*/true)`, cross-checked live via
#    ``onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()``
#    too) with no schema default, so it always has some concrete value on any
#    valid node. Session *construction* with `k > num_experts` succeeds
#    (shape inference never looks at `k` against `num_experts`) but
#    *execution* fails hard with an explicit, unambiguous ONNX Runtime error
#    -- "MoE attribute 'k' must be <= num_experts; got k=<k>, num_experts=<n>"
#    (`moe_cpu.cc`) -- confirmed by actually triggering it. So `num_experts`
#    can never be pruned below `k`. Rather than *clamp `k` down* to fit (which
#    would silently change how many experts every surviving token gets
#    combined across, a real behavior change on top of the pruning itself,
#    and would touch a node attribute this pass has no principled way to
#    "undo" the effect of), this pass instead treats `k` as a hard floor on
#    how many experts survive -- `num_experts_to_keep` is silently raised to
#    `k` (never lowered below it) if the requested `sparsity` would otherwise
#    ask for fewer, the same "at least one channel is always kept" floor
#    :func:`apply_moe_expert_channel_pruning` already uses, just floored at
#    `k` instead of 1 (below `k`, the node cannot execute *at all*, not
#    merely "prunes to nothing usable"). `k` itself is never written to.
#
# 3. *Can the router's own weight be identified/isolated as a pruneable
#    producer in the general case?* Only when it is provably safe to, the
#    same bar every other producer this module touches is held to: exactly
#    one node produces `router_probs`, it is a plain MatMul or (`transA=0`,
#    `alpha=1`, `beta=1` if biased) Gemm (:func:`_match_matmul_like`, the
#    same matcher every ordinary MatMul/Gemm producer in this module is
#    matched with), its weight is a constant 2-D float32 initializer whose
#    output-channel width equals `num_experts`, `router_probs` itself has no
#    consumer besides this one `MoE` node (an auxiliary head or logging
#    output reading the same tensor would otherwise silently see a
#    now-differently-shaped tensor), and neither the weight nor its optional
#    bias is a tied/shared initializer read anywhere else (the same
#    tied-weight guard :func:`_match_moe_producer` already applies to
#    `fc1`/`fc2`). Any node whose `router_probs` producer doesn't match this
#    exactly -- a router expressed as more than one node (e.g. a bias `Add`
#    kept separate from the Gemm, a `Reshape`/`Cast` in between, jitter noise
#    added during training-mode export, ...), fed from more than one
#    consumer, or backed by a non-constant/tied weight -- is left completely
#    untouched rather than guessed at.
#
# 4. *Does this hold for every `MoE` configuration, or only some?* This pass
#    reuses :func:`_match_moe_producer`'s own `fc1`/`fc2`/`fc3`/`activation_type`
#    checks outright (so it declines `fc3_experts_weights` and `swiglu` for
#    the exact same confirmed-empirically reasons documented in this module's
#    "MoE expert-intermediate-channel pruning" section above -- `num_experts`
#    lives on `fc1`'s/`fc2`'s shared axis 0 regardless of `fusion_size`, so
#    neither restriction is *structurally* required for whole-expert pruning
#    the way it is for `inter_size` pruning, but this pass keeps the same
#    narrow, already-verified surface rather than opening a new one that
#    would need its own independent oracle check) -- plus one restriction of
#    its own: `use_sparse_mixer=1` (a different, jitter-named top-2-only
#    routing path -- confirmed empirically to hard-require `k == 2`
#    specifically, `moe_base_cpu.h`'s own "Sparse mixer only supports k=2"
#    check) is declined outright. It was confirmed deterministic run-to-run
#    on this environment's CPU provider (no actual training-time jitter
#    applied at inference), but its own comparison-based expert tie-break
#    logic was not independently re-derived and checked against the same
#    `-inf`-masking oracle point 1 relies on, so -- the same conservative
#    call every other under-verified case in this module makes -- it is left
#    untouched rather than assumed safe.


@dataclass(frozen=True)
class _MoEExpertChain:
    node: onnx.NodeProto
    fc1_w: str
    fc1_b: Optional[str]
    fc2_w: str
    fc2_b: Optional[str]
    num_experts: int
    k: int
    router_probs: str
    router_w: str
    router_w_transposed: bool
    router_b: Optional[str]


def _match_moe_whole_expert_producer(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    node_by_output: Dict[str, onnx.NodeProto],
    graph_outputs: Set[str],
) -> Optional[_MoEExpertChain]:
    """If `node` is a ``com.microsoft::MoE`` node this pass can safely prune
    whole experts (the `num_experts` axis) from, returns the matched
    :class:`_MoEExpertChain`. See this section's own comment above for the
    full safety argument; this mirrors :func:`_match_moe_producer`'s own
    `fc1`/`fc2`/`fc3`/`activation_type` checks exactly (reused outright, via
    a direct call), then adds the `k`/`use_sparse_mixer` checks and the
    `router_probs` producer match described there.
    """
    base = _match_moe_producer(node, initializer_map, consumers_of)
    if base is None:
        return None

    k = None
    use_sparse_mixer = 0
    for attr in node.attribute:
        if attr.name == "k":
            k = attr.i
        elif attr.name == "use_sparse_mixer":
            use_sparse_mixer = attr.i
    if k is None or k < 1 or k > base.num_experts or use_sparse_mixer != 0:
        return None

    if len(node.input) < 2 or not node.input[1]:
        return None
    router_probs = node.input[1]
    if router_probs in graph_outputs or len(consumers_of.get(router_probs, [])) != 1:
        return None
    router_node = node_by_output.get(router_probs)
    if router_node is None:
        return None
    router_info = _match_producer(router_node, initializer_map)
    if router_info is None:
        return None
    router_w_name, router_w_transposed, router_b_name, n_channels = router_info
    if n_channels != base.num_experts or len(consumers_of.get(router_w_name, [])) != 1:
        return None
    if router_b_name is not None and len(consumers_of.get(router_b_name, [])) != 1:
        return None

    return _MoEExpertChain(
        node=node,
        fc1_w=base.fc1_w,
        fc1_b=base.fc1_b,
        fc2_w=base.fc2_w,
        fc2_b=base.fc2_b,
        num_experts=base.num_experts,
        k=int(k),
        router_probs=router_probs,
        router_w=router_w_name,
        router_w_transposed=router_w_transposed,
        router_b=router_b_name,
    )


def _find_moe_whole_expert_chains(graph: onnx.GraphProto) -> List[_MoEExpertChain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    node_by_output = {out: node for node in graph.node for out in node.output}
    graph_outputs = {o.name for o in graph.output}
    chains = []
    for node in graph.node:
        chain = _match_moe_whole_expert_producer(
            node, initializer_map, consumers_of, node_by_output, graph_outputs
        )
        if chain is not None:
            chains.append(chain)
    return chains


def _moe_expert_weight_importance(
    chain: _MoEExpertChain, initializer_map: Dict[str, onnx.TensorProto]
) -> np.ndarray:
    """Combined (root-sum-square) L2-norm importance per *expert* -- the
    weight-magnitude-only fallback used when no calibration data was
    observed for a chain's `router_probs` (mirrors
    :func:`apply_structured_wanda_pruning`'s own "no matching activation
    observed -> fall back to |W|" behavior). Each expert `e` owns one whole
    `fc1_experts_weights[e]`/`fc2_experts_weights[e]` (and, if present,
    `fc1_experts_bias[e]`) slice; unlike :func:`_moe_importance` (which
    reduces *across* the expert axis to rank `inter_size`), this reduces
    *within* each expert's own slice to rank experts themselves.
    """
    fc1_w = _to_f64(initializer_map[chain.fc1_w])
    fc2_w = _to_f64(initializer_map[chain.fc2_w])
    squared = np.sum(np.square(fc1_w), axis=(1, 2)) + np.sum(
        np.square(fc2_w), axis=(1, 2)
    )
    if chain.fc1_b is not None:
        fc1_b = _to_f64(initializer_map[chain.fc1_b])
        squared = squared + np.sum(np.square(fc1_b), axis=1)
    return np.sqrt(squared)


def _apply_moe_whole_expert_chains(
    graph: onnx.GraphProto,
    chains: List[_MoEExpertChain],
    sparsity: float,
    compute_importance,
) -> Set[str]:
    initializer_map = {t.name: t for t in graph.initializer}
    touched: Set[str] = set()
    stale_value_info: Set[str] = set()

    for chain in chains:
        weight_names = {chain.fc1_w, chain.fc2_w, chain.router_w}
        if chain.fc1_b is not None:
            weight_names.add(chain.fc1_b)
        if chain.router_b is not None:
            weight_names.add(chain.router_b)
        if weight_names & touched:
            continue  # a shared/tied initializer another MoE node already resized
        touched |= weight_names

        n = chain.num_experts
        floor = max(1, min(chain.k, n))
        keep_count = max(floor, n - round(n * sparsity))
        if keep_count >= n:
            continue  # rounds down to nothing (or below k) for this layer -- no-op

        importance = compute_importance(chain, initializer_map)
        keep = np.sort(np.argsort(-importance)[:keep_count])

        fc1_init = initializer_map[chain.fc1_w]
        fc1_w_new = np.take(onnx.numpy_helper.to_array(fc1_init), keep, axis=0)
        fc1_init.CopyFrom(
            onnx.numpy_helper.from_array(
                np.ascontiguousarray(fc1_w_new), name=chain.fc1_w
            )
        )

        fc2_init = initializer_map[chain.fc2_w]
        fc2_w_new = np.take(onnx.numpy_helper.to_array(fc2_init), keep, axis=0)
        fc2_init.CopyFrom(
            onnx.numpy_helper.from_array(
                np.ascontiguousarray(fc2_w_new), name=chain.fc2_w
            )
        )

        if chain.fc1_b is not None:
            fc1_b_init = initializer_map[chain.fc1_b]
            fc1_b_new = np.take(onnx.numpy_helper.to_array(fc1_b_init), keep, axis=0)
            fc1_b_init.CopyFrom(
                onnx.numpy_helper.from_array(
                    np.ascontiguousarray(fc1_b_new), name=chain.fc1_b
                )
            )
        # fc2_experts_bias indexes hidden_size, unaffected by an expert-count
        # cut -- never sliced here, same reasoning as expert-channel pruning.

        router_w_init = initializer_map[chain.router_w]
        _slice_producer_weight(router_w_init, chain.router_w_transposed, keep)
        if chain.router_b is not None:
            _slice_last_axis(initializer_map[chain.router_b], keep)

        stale_value_info.add(chain.router_probs)

    return stale_value_info


class _HasRouterProbs(Protocol):
    """Structural type for `_moe_router_gate_calibration_stats`'s own
    `chains` parameter: any per-node chain record with a `router_probs`
    tensor name -- :class:`_MoEExpertChain` (plain ``MoE``) and
    :class:`_QMoEExpertChain` (``QMoE``, see this module's own "QMoE
    (quantized-weight MoE) pruning" section below) both satisfy this
    structurally, with no shared base class needed, so this one
    calibration implementation serves both whole-expert-pruning families
    -- router-gate calibration only ever reads `router_probs` (the
    router's own *output*, upstream of either node's own MoE/QMoE-specific
    machinery), so it is genuinely oblivious to which family produced it.
    Declared as a read-only property, not a plain attribute -- mypy treats
    a Protocol's own mutable-attribute members invariantly (a plain
    ``router_probs: str`` would then reject ``List[_MoEExpertChain]``/
    ``List[_QMoEExpertChain]`` here, since neither is *literally*
    ``_HasRouterProbs``), while a read-only property is covariant, which is
    all this function ever needs (it only ever reads `router_probs`).
    """

    @property
    def router_probs(self) -> str: ...


def _moe_router_gate_calibration_stats(
    out: onnx.ModelProto,
    chains: Sequence[_HasRouterProbs],
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]],
) -> Dict[str, np.ndarray]:
    """Runs `out` over `calibration_data` and returns the per-`router_probs`
    mean gate weight (post-Softmax over the expert axis, averaged over every
    calibration token) that :func:`apply_moe_whole_expert_pruning`'s own
    body used to compute inline before this function existed -- keyed by
    each chain's own `router_probs` tensor name. Factored out, read-only
    (never mutates `out`), so :func:`analyze_pruning_sensitivity`'s own
    dry-run report can compute the *exact* same per-expert usage ranking
    :func:`apply_moe_whole_expert_pruning` would, from one single shared
    implementation -- the same "one place real duplication would have
    crept in" extraction this module's own "Dry-run pruning sensitivity
    analysis" section comment documents for
    :func:`_wanda_unstructured_calibration_stats`/
    :func:`_wanda_structured_calibration_stats`. Also reused, unchanged, by
    :func:`apply_qmoe_whole_expert_pruning` (see :class:`_HasRouterProbs`
    above).

    `out` is expected to already be the caller's own working copy, exactly
    as :func:`_wanda_unstructured_calibration_stats` expects.
    """
    probe_names = sorted({chain.router_probs for chain in chains})
    probe_model = _add_probe_outputs(out, probe_names)

    sum_prob: Dict[str, np.ndarray] = {}
    count: Dict[str, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for name in probe_names:
            logits = np.asarray(result[name], dtype=np.float64)
            if logits.ndim != 2:
                continue  # router_probs is always documented 2-D; skip if not
            # Numerically-stable Softmax over the expert axis (the same
            # normalization MoE's own kernel applies internally, per this
            # section's own comment), averaged over every token.
            shifted = logits - logits.max(axis=-1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp / exp.sum(axis=-1, keepdims=True)
            s = probs.sum(axis=0)
            sum_prob[name] = s if name not in sum_prob else sum_prob[name] + s
            count[name] = count.get(name, 0) + logits.shape[0]
    return {name: s / max(count[name], 1) for name, s in sum_prob.items()}


def apply_moe_whole_expert_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Removes whole experts (shrinks the `num_experts` leading axis) from a
    matched ``com.microsoft::MoE`` node and its upstream router projection at
    once -- the complementary technique to
    :func:`apply_moe_expert_channel_pruning`'s own `inter_size` pruning (see
    that function's own docstring and this module's docstring for how the two
    relate), and the general, calibration-based "prune the least-used
    experts" technique from the post-training MoE-pruning literature: experts
    are ranked by their mean router *gate weight* -- ``softmax(router_probs)``
    averaged over every calibration token -- not raw logit magnitude (logits
    have no shared scale across experts to compare on their own; Softmax is
    what makes them comparable) and not exact top-k selection
    frequency/combine weight (which would require re-deriving ONNX Runtime's
    own top-k + optional `normalize_routing_weights` renormalization exactly,
    an unnecessary duplication of runtime semantics this pass doesn't need
    just to get a solid usage signal). See this section's own comment above
    for the full safety argument -- in particular why shrinking
    `router_probs`' own width is *exactly* equivalent to forcing the dropped
    experts' routing logits to `-inf` (confirmed to 0.0 max-abs-diff against
    a real onnxruntime CPU session), why `k` is never touched but instead
    floors how many experts can ever be pruned away, and exactly which nodes
    are matched (:func:`_match_moe_whole_expert_producer`) -- `fc3`/`swiglu`/
    `use_sparse_mixer` and any router not expressed as one plain, untied
    MatMul/Gemm feeding `router_probs` and nothing else are all out of scope.

    Every matched expert's `fc1_experts_weights`/`fc2_experts_weights` (and
    `fc1_experts_bias`, if present) row, and the router projection weight's
    (and bias's, if present) matching output column, are dropped together for
    the lowest-``sparsity``-fraction of experts by that ranking -- with
    `num_experts_to_keep` silently floored at the node's own `k` (pruning
    below `k` experts remaining is a hard onnxruntime execution failure, not
    merely suboptimal -- confirmed empirically, see above) rather than ever
    adjusting `k` itself. A chain whose `router_probs` was never observed
    during calibration (e.g. ``calibration_data=[]``) falls back to each
    expert's own combined `fc1`/`fc2` (+`fc1_experts_bias`) L2 weight norm,
    the same "no matching activation observed" fallback
    :func:`apply_structured_wanda_pruning` already uses.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to rank experts by
            mean router gate weight on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched node's `num_experts` to
            remove (floored at the node's own `k`, so fewer may actually be
            removed -- never more)
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration router activations
    :returns: ``model`` with every matched ``MoE`` node's `fc1`/`fc2`(/`fc1`
            bias) and its router projection's weight(/bias) resized in
            place; a node with `fc3_experts_weights`, a `swiglu`/unrecognized
            `activation_type`, `use_sparse_mixer`, a non-constant or
            tied/shared weight anywhere in the chain (including the router
            projection), a `router_probs` with more than one consumer, or any
            other shape this pass doesn't recognize is left completely
            untouched

    Weight dtype support mirrors :func:`apply_moe_expert_channel_pruning`
    (FLOAT/FLOAT16/BFLOAT16 for `fc1`/`fc2`/biases, and the router
    projection weight); the router-gate calibration activations here are
    likewise captured via a real ``onnxruntime`` run and cast to float64
    on capture regardless of the graph's own declared dtype.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = _find_moe_whole_expert_chains(graph)
    if not chains:
        return out

    mean_gate_weight = _moe_router_gate_calibration_stats(
        out, chains, calibration_data, providers
    )

    def _importance(
        chain: _MoEExpertChain, initializer_map: Dict[str, onnx.TensorProto]
    ) -> np.ndarray:
        gate = mean_gate_weight.get(chain.router_probs)
        if gate is None or gate.shape[0] != chain.num_experts:
            return _moe_expert_weight_importance(chain, initializer_map)
        return gate

    stale_value_info = _apply_moe_whole_expert_chains(
        graph, chains, sparsity, _importance
    )
    if stale_value_info:
        kept = [vi for vi in graph.value_info if vi.name not in stale_value_info]
        del graph.value_info[:]
        graph.value_info.extend(kept)
    return out


# --- QMoE (quantized-weight MoE) pruning -------------------------------------
#
# This section's own comment above (the plain-``MoE`` "expert-intermediate-
# channel pruning" section) already names ``QMoE`` as the *other* ``*MoE*``-
# named ``com.microsoft`` schema this environment's own live schema registry
# returns -- confirmed there, empirically, that until this section existed
# nothing in this module's matching code ever checked for ``op_type ==
# "QMoE"`` at all. This section closes that gap: both of :func:`_match_moe_producer`'s
# own two techniques (per-expert intermediate-channel pruning and
# whole-expert removal), retargeted at ``QMoE``'s own packed/quantized
# ``fc1``/``fc2`` weights instead of plain ``MoE``'s float ones.
#
# Every fact below was re-derived from ``QMoE``'s own *live* schema
# (``onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()``,
# onnxruntime 1.29.0 in this environment) and, for anything the schema's own
# prose doesn't pin down byte-for-byte, from real ``onnxruntime.InferenceSession``
# runs on the CPU execution provider -- never assumed from ``QMoE``'s name
# alone or from how a superficially similar op (``MatMulNBits``, or this
# module's own int8 QDQ pattern -- see the "QDQ (quantized-weight) structured
# pruning" section above) happens to work:
#
# 1. *Schema shape, ``MoE`` vs ``QMoE``.* ``QMoE`` shares `MoE`'s own
#    `input`/`router_probs`/`fc1_experts_bias`/`fc2_experts_bias` inputs and
#    `activation_type`/`swiglu_fusion`/`k`/`use_sparse_mixer`/
#    `normalize_routing_weights` attributes verbatim, but balloons to 21
#    inputs total: `fc1_experts_weights`/`fc2_experts_weights`/(optional)
#    `fc3_experts_weights` become packed ``uint8`` (or, for the FP4/FP8
#    modes below, ``float8e4m3fn``) tensors instead of float ones, each
#    gains an optional per-fc `scales` and `zero_points` operand, and eight
#    more FP4/FP8-quantization-specific inputs (`router_weights`,
#    `fc{1,2}_global_scale`, `fc{1,2}_act_scale`, `fc{1,2}_act_block_scale`)
#    appear that plain `MoE` has no equivalent of at all. `QMoE` also adds
#    three of its own attributes: `quant_type` (``'int'`` (the schema's own
#    default), ``'fp4'``, ``'nvfp4'``, ``'fp8'``, or ``'wfp4afp8'``),
#    `expert_weight_bits` (2, 4, or 8 -- `quant_type='int'` only), and
#    `block_size`/`weights_prepacked` (both -- see points 3 and 5 below).
#    This pass targets `quant_type == 'int'` only, both with no `block_size`
#    (whole-row per-channel scale, point 2) and with `block_size` set to a
#    groupwise value (point 3) -- and declines everything else outright
#    (point 6); the FP4/FP8 modes are a genuinely different tensor format
#    (MXFP block-scaled, `float8e4m3fn`-packed weights, separate activation
#    scale operands) this pass was never built or verified against, and
#    remains explicitly out of scope.
#
# 2. *Packing/dequantization ("column-wise", `quant_type='int'`, no
#    `block_size`).* The op's own doc text says weights are "stored in
#    column major order per expert" and gives the dequantization formula
#    ``dequantized_weight = (quantized_weight - zero_point) * scale``, with
#    `zero_point` defaulting to ``2**(bits - 1)`` when absent -- but neither
#    of those alone pins down the *byte-level* packing order, so this was
#    verified against a real, independent reference: `onnxruntime`'s own
#    ``onnxruntime.quantization.cuda_quantizer.CudaQuantizer`` ships a pure-
#    Python re-derivation of this exact "raw QMoE storage" contract
#    (`qmoe_symmetric_per_channel_quantize`, used to build real QMoE test
#    models, not this module's own code) -- confirming: `fc1_experts_weights`/
#    `fc2_experts_weights` keep the same ``[num_experts, N, K]`` axis
#    *meaning* as plain `MoE`'s own (`N` = `fc1`'s `inter_size`/`fc2`'s
#    `hidden_size`, `K` = `fc1`'s `hidden_size`/`fc2`'s `inter_size`) --
#    "column major" describes the *scale* granularity (one scale per output
#    row/channel `N`, over the *whole* `K` row at once -- what the op's own
#    doc calls "column wise quantization" when `block_size` is absent, i.e.
#    exactly `MatMulNBits`' own per-channel case with one block spanning all
#    of `K`), not a transposed tensor -- while the *last* axis (`K`) is
#    packed `8 // bits` logical values per byte, low-index-in-low-bits first
#    (byte ``= v[0] | v[1] << bits | ...``). This was independently
#    confirmed to be the *exact* convention this environment's own CPU
#    `QMoE` kernel expects too (not merely what the CUDA-side Python
#    reference happens to produce): a hand-built `QMoE` node, quantized this
#    exact way and unpacked/dequantized back with :func:`_unpack_subbyte`/
#    the ``(q - zero_point) * scale`` formula, matches a real
#    ``onnxruntime.InferenceSession`` run of that same node's own output to
#    *exactly* 0.0 max-abs-diff -- for `bits` in ``{2, 4, 8}`` each, for a
#    single-expert/single-selected-expert node (isolating the dequantization
#    formula from any routing/gating effects), and again with a deliberately
#    *non-default*, per-channel `zero_point` operand (confirming
#    `zero_points`, when present, packs with the identical
#    low-index-in-low-bits convention along its own channel axis, not merely
#    that the *default* zero point is honored). `fc1_zero_points`'/
#    `fc2_zero_points`'s own channel axis is `N` (`inter_size`/`hidden_size`
#    respectively, the *same* axis as `fc1_scales`/`fc2_scales`, per the
#    schema's own shape doc) -- **not** `MatMulNBits`' own block-relative
#    layout, so this pass's own packed-slicing helpers were written fresh
#    from this confirmed-independently layout rather than reused from (or
#    force-fit to) any `MatMulNBits` pruning code, even where one exists
#    elsewhere in this session -- see point 4.
#
# 3. *`block_size` (blocked/groupwise quantization) -- SUPPORTED, re-derived
#    independently rather than assumed from the schema's own prose alone.*
#    The live schema's own per-input shape doc (`get_all_operator_schema()`,
#    not merely its free-text doc string) states the exact shapes: with
#    `block_size` set, `fc1_scales`/`fc2_scales` go from 2D
#    ``(num_experts, N)`` to 3D ``(num_experts, N, K / block_size)`` -- one
#    scale per `block_size`-sized group along `K`, `N`'s own axis position
#    unchanged -- and `fc1_zero_points`/`fc2_zero_points` go from 2D
#    ``(num_experts, N / pack_size)`` to 3D ``(num_experts, N, K /
#    block_size / pack_size)``, where `pack_size = 8 // expert_weight_bits`.
#    Crucially, this *moves* `zero_points`' own sub-byte-packed axis: point
#    2 above (no `block_size`) confirmed `zero_points` packs along `N`
#    (matching `scales`' own single-scalar-per-channel axis); with
#    `block_size` set, `zero_points` instead packs along the *new*,
#    trailing `K`-block axis -- packing follows whichever axis is
#    `zero_points`' own last one, not a fixed axis. Both of these were
#    verified against a real CPU `onnxruntime.InferenceSession`, not just
#    read off the schema: a hand-built two-block (`block_size=16`,
#    `K=32`), two-expert `QMoE` node -- non-default per-block `zero_points`
#    included -- dequantized independently in numpy (blockwise formula:
#    ``dequantized[e, n, k] = (code[e, n, k] - zero_point[e, n, k //
#    block_size]) * scale[e, n, k // block_size]``, weights unpacked with
#    the *same* :func:`_unpack_subbyte` used for the no-`block_size` case)
#    matches the real kernel's own output to ~1e-6 max-abs-diff (float32
#    quantization/matmul noise, not a formula mismatch). `K` (`fc1`'s
#    `hidden_size`, `fc2`'s `inter_size`) is packed **contiguously**
#    across the whole row, the same low-index-in-low-bits convention as
#    point 2, *not* restarted at each block boundary -- confirmed by using
#    a plain flat `_unpack_subbyte`/`_pack_subbyte` round trip (oblivious
#    to block boundaries entirely) and matching the real kernel exactly.
#    This pass additionally requires `block_size` itself to be a multiple
#    of `pack_size` (true for every `block_size` this op's own kernel
#    already accepts -- 16/32/64/128, all multiples of `pack_size` in
#    ``{1, 2, 4}`` -- so this excludes no real configuration) so that a
#    dropped whole block's own boundary is always also a packed-byte
#    boundary; this was never independently exercised against a
#    deliberately *non*-byte-aligned `block_size` (the CPU kernel's own
#    ``block_size >= 16`` floor, re-confirmed live below, makes one
#    genuinely hard to construct for `bits` in ``{2, 4, 8}`` anyway), so
#    such a value is declined defensively rather than assumed to also work.
#    The CPU kernel's own ``block_size >= 16`` floor
#    (``"block_size must be >= 16 when provided"``) was re-confirmed live
#    here too (unchanged from the previous, no-longer-accurate version of
#    this comment, which declined `block_size` outright specifically
#    *because* this shape was never independently re-derived -- now done,
#    see above). `hidden_size`/`inter_size` not dividing evenly by
#    `block_size` is declined the same way every other shape mismatch in
#    this section is: the matcher's own expected-shape check on
#    `fc1_scales`/`fc2_scales`/`fc1_zero_points`/`fc2_zero_points` simply
#    fails to match (integer floor division alone would silently expect a
#    too-small tensor otherwise), which also transitively enforces the
#    schema doc's own extra "`hidden_size`/`inter_size` divisible by
#    `block_size * pack_size`" rule for whichever of `fc1`/`fc2` has
#    `zero_points` present, without this pass needing to check it
#    separately.
#
#    A separate, genuinely surprising finding, confirmed live and entirely
#    independent of anything this pass's own matcher checks: when
#    `hidden_size == block_size` (`fc1`'s own block count, `hidden_blocks`,
#    degenerates to exactly 1), this environment's real CPU kernel demands
#    `fc2_scales`' own trailing block-axis size ALSO be 1 -- regardless of
#    `inter_size`'s own, independently-computed block count -- rejecting a
#    schema-doc-shaped `fc2_scales` tensor with a real, on-topic error
#    (``"Input 'fc2_experts_scales' is expected to have shape {...,1}, got
#    {...,N}"``) whenever `inter_size`'s own block count differs from 1.
#    Reproduced directly against a hand-built node with no pruning involved
#    at all, confirming this is a pre-existing kernel quirk in this
#    degenerate shape, not something this pass's own slicing could
#    introduce or needs to guard against: a model already built this way
#    was already unrunnable on this kernel before any pruning touched it,
#    and this pass never changes `hidden_size` (`fc1`'s own `K`) at all, so
#    it can neither create nor fix this quirk for any chain it prunes.
#
#    Channel pruning's own two weights sit on *opposite* sides of the
#    `inter_size` axis relative to where blocking lives: `fc1`'s blocks
#    group along `hidden_size` (`fc1`'s `K`), an axis this pass never
#    touches when pruning `inter_size` (`fc1`'s `N`) -- so `fc1_scales`'/
#    `fc1_zero_points`' pruned axis (`N`, the *first* of their two/three
#    dims) is untouched by blocking either way, a plain row-slice with no
#    unpack/repack needed for `zero_points` here even though one *was*
#    needed in the no-`block_size` case (blocking moved `zero_points`'
#    packed axis off of `N` entirely, per above). `fc2`'s blocks, by
#    contrast, group along `inter_size` itself (`fc2`'s `K`, this pass's
#    own pruned axis) -- so a channel-level pruned `inter_size` keep-set
#    can split a block's scale/zero-point in two, which no quantized value
#    already in the graph can represent (this module's own "never invent
#    new quantized values" principle, exactly :func:`_matmul_nbits_block_aligned_keep_blocks`'s
#    own reasoning for `MatMulNBits`' analogous K-axis case) -- so this
#    pass resolves the keep-set to whole `block_size`-sized groups before
#    ever slicing, by ranking `block_size`-sized *groups* of the same
#    per-channel importance score (root-sum-square, the same reduction
#    :func:`_qmoe_channel_importance` already uses across weights) rather
#    than individual channels, then keeping/dropping each block as a whole
#    -- see :func:`_apply_qmoe_channel_chains` for the exact mechanics.
#
# 4. *`weights_prepacked`.* A tri-state attribute (default ``-1``) that
#    selects between this pass's own assumed raw ``[N, K/pack]`` storage (0,
#    or -1 -- confirmed live to produce byte-identical CPU output to an
#    explicit 0, i.e. -1 means "raw" on this environment's CPU execution
#    provider, not "some other layout") and a CUTLASS mixed-GEMM prepacked
#    byte layout (1) built for CUDA kernels -- an entirely different,
#    GPU-specific byte shuffle (row permutation, sub-byte transpose, column
#    interleaving; see `onnxruntime.quantization.cuda_quantizer`'s own
#    `_preprocess_weights_for_mixed_gemm_torch`) this pass makes no attempt
#    to slice correctly. `weights_prepacked` values other than ``-1``/``0``
#    are declined outright.
#
# 5. *Activation/`fc3` support -- verified independently for `QMoE`, not
#    assumed from `MoE`'s own already-confirmed set.* Every one of
#    `relu`/`gelu`/`silu`/`identity` was independently run end-to-end
#    against this environment's real CPU `onnxruntime.InferenceSession`,
#    for `expert_weight_bits` in ``{2, 4, 8}`` each -- all execute cleanly,
#    the same activation set plain `MoE` already supports (`swiglu` requires
#    `swiglu_fusion=1` on CPU here too, "CPU QMoE only supports interleaved
#    SwiGLU format", the identical restriction/decline plain `MoE` already
#    documents, for the identical reason -- the doubled/interleaved
#    `fc1_experts_weights` row count this pass's own shape check already
#    rules out structurally). `fc3_experts_weights` presence was also
#    independently re-confirmed to fail on this environment's real CPU
#    kernel -- ``"FC3 gating is not yet implemented on CPU for QMoE"`` --
#    not merely assumed to inherit plain `MoE`'s own already-documented
#    ``"FC3 is not implemented for CPU MoE"`` limitation; both are declined
#    for the same "no CPU oracle to validate against" reasoning.
#
# 6. *A genuinely surprising finding, independent of every point above:
#    `QMoE`'s `normalize_routing_weights` attribute is a no-op on this
#    environment's CPU kernel.* Plain `MoE` genuinely switches gating
#    behavior on this attribute (confirmed live: with it unset/0, `MoE`'s
#    own gate weight is the *raw* Softmax-over-*every*-expert probability of
#    each selected expert -- summing to well under 1 whenever ``k <
#    num_experts``; with it set to 1, `MoE` *renormalizes* the selected
#    top-`k` probabilities to sum to exactly 1). `QMoE`, on this same
#    environment's real CPU kernel, gives *byte-identical* output whether
#    `normalize_routing_weights` is left unset, explicitly 0, or explicitly
#    1 -- confirmed by fitting each selected expert's own implied gate
#    weight (least-squares against each expert's own real, independently
#    computed output slice) back out of a real multi-expert `QMoE` run:
#    `QMoE` *always* renormalizes to top-`k`-sums-to-1, unconditionally,
#    regardless of what the attribute says. This does not weaken point 7's
#    own whole-expert-pruning safety argument below (a real masking-oracle
#    run against `QMoE` itself, not inherited from `MoE`'s own already-
#    verified equivalence, still matches a truly-shrunk-`num_experts` `QMoE`
#    node to *exactly* 0.0 max-abs-diff) -- but it does mean neither this
#    pass's matcher nor its `apply_*` body reads or conditions on this
#    attribute's value for `QMoE` at all (any value has the identical real
#    effect), unlike an a-priori assumption that `QMoE` would simply inherit
#    `MoE`'s own two-mode behavior verbatim.
#
# 7. *Whole-expert removal's own safety argument, re-verified for `QMoE`
#    specifically.* The plain-`MoE` whole-expert-pruning section above
#    already lays out the full "shrinking `router_probs`' own width is
#    exactly equivalent to `-inf`-masking the dropped experts' logits"
#    argument (points 1-4 of that section's own comment) and why `k` floors
#    rather than adjusts. Every one of those points -- `router_probs` still
#    holds raw per-token/per-expert logits with an *internal* Softmax
#    (unaffected by `QMoE`'s own weight quantization, which only touches
#    `fc1`/`fc2`/`fc3`), `k` is still a required attribute with an identical
#    ``"k must be <= num_experts"`` execution-time failure (confirmed live
#    against `QMoE` directly), `use_sparse_mixer=1` still hard-requires
#    ``k == 2`` (confirmed live against `QMoE` directly, same "Sparse mixer
#    only supports k=2" message plain `MoE` gives) -- transfers unchanged,
#    since none of it depends on how `fc1`/`fc2`'s own *weights* are
#    represented. The one piece worth re-deriving independently rather than
#    assuming transfers is the masking-equivalence oracle itself (point 6
#    above is exactly why it can't be assumed to transfer unmodified from
#    `MoE`'s own already-verified version) -- confirmed: a same-shape
#    ``QMoE`` "masking" oracle (dropped experts' `fc1`/`fc2` (quantized, any
#    values -- never selected, so never read) and router weight column/bias
#    forced so those experts are never in any token's top-`k`) matches a
#    genuinely-`num_experts`-shrunk `QMoE` node's own real output to exactly
#    0.0 max-abs-diff.
#
# 8. *Whole-expert removal needs no `block_size`-specific handling at all.*
#    Every per-expert tensor `_apply_qmoe_whole_expert_chains` touches
#    (`fc1`/`fc2` weights/scales/biases/zero_points) keeps `num_experts` as
#    its own leading axis regardless of `block_size` (point 3 above: a
#    `block_size`-shaped tensor only ever gains a *trailing* `K`-block axis,
#    never touching axis 0) -- so the exact same generic, rank-agnostic
#    :func:`_slice_axis`-based leading-axis index-select this pass already
#    uses for the no-`block_size` case slices a blockwise tensor correctly
#    unmodified, with no unpack/repack anywhere (dropping whole experts
#    never touches a byte shared between two surviving experts, blockwise
#    or not). Confirmed live end-to-end (not merely inferred from shape):
#    a blockwise-int `QMoE` node pruned by
#    :func:`apply_qmoe_whole_expert_pruning` matches a real, independently
#    hand-built "already `num_experts`-shrunk" blockwise-int reference node
#    to float32 noise, the same bar every other pruned-vs-reference test in
#    this module is held to -- see
#    ``test_qmoe_whole_expert_pruning_blockwise_int_matches_hand_built_reference``.
#
# `router_weights` (input 14, a *separate* DeepSeek-style
# select/aggregate-with-different-tensors combine-weight operand) and every
# one of the eight FP4/FP8/global-scale/activation-scale inputs (15-20) are
# required absent by this pass's own matcher -- naturally so for any real
# `quant_type='int'` export, but checked explicitly rather than assumed,
# since none of their effects on the masking-equivalence argument above (or
# on `fc1`/`fc2`'s own packed-channel semantics) were independently
# verified.


def _unpack_subbyte(packed: np.ndarray, bits: int, logical_len: int) -> np.ndarray:
    """Unpacks `packed`'s last axis -- ``uint8`` bytes, each holding
    ``8 // bits`` sub-byte values of `bits` bits apiece -- into ``uint8``
    values in ``[0, 2**bits)``, one per logical index, trimmed to
    `logical_len` (a byte's own top slot goes unused, and is dropped here,
    whenever the *logical* axis length isn't itself a multiple of
    ``8 // bits``). Low index in a byte's own low bits, increasing index in
    increasingly high bits -- ``com.microsoft::QMoE``'s own raw
    (``weights_prepacked`` in ``{-1, 0}``), ``quant_type='int'`` storage
    convention for both `fc1_experts_weights`/`fc2_experts_weights` and
    `fc1_zero_points`/`fc2_zero_points` alike -- confirmed empirically
    against a real CPU ``onnxruntime.InferenceSession``, not assumed from
    the schema's own prose; see this section's own top comment (point 2)
    for the full verification.
    """
    pack = 8 // bits
    mask = (1 << bits) - 1
    parts = [(packed >> (bits * i)) & mask for i in range(pack)]
    unpacked = np.stack(parts, axis=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * pack
    )
    return unpacked[..., :logical_len].astype(np.uint8)


def _pack_subbyte(unpacked: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of :func:`_unpack_subbyte`: packs `unpacked`'s last axis
    (``uint8`` values in ``[0, 2**bits)``) back into bytes, ``8 // bits``
    logical values per byte, the same low-index/low-bits convention. Pads
    the logical axis up to a whole number of ``8 // bits`` first (with
    zeros) when it doesn't already divide evenly -- the exact same "one
    wasted slot" case :func:`_unpack_subbyte` trims away, so a
    slice-then-repack round trip through both is exact for any survivor
    count, not just ones divisible by the pack width (the "genuinely
    bit-packed odd-count" edge case this module's own int4 packing
    elsewhere already has to handle).
    """
    pack = 8 // bits
    n = unpacked.shape[-1]
    pad = (-n) % pack
    if pad:
        pad_width = [(0, 0)] * (unpacked.ndim - 1) + [(0, pad)]
        unpacked = np.pad(unpacked, pad_width)
    reshaped = unpacked.reshape(*unpacked.shape[:-1], -1, pack).astype(np.uint8)
    out = np.zeros(reshaped.shape[:-1], dtype=np.uint8)
    for i in range(pack):
        out = out | (
            (reshaped[..., i] & np.uint8((1 << bits) - 1)) << np.uint8(bits * i)
        )
    return out


def _qmoe_default_zero_point(bits: int) -> int:
    """``2 ** (bits - 1)`` -- the schema's own documented default
    `zero_point` for whichever of `fc1_zero_points`/`fc2_zero_points`/
    `fc3_zero_points` is absent from a given `QMoE` node.
    """
    return 1 << (bits - 1)


def _qmoe_dequantize(
    packed: onnx.TensorProto,
    scale: onnx.TensorProto,
    zero_points: Optional[onnx.TensorProto],
    bits: int,
    k: int,
    block_size: int = 0,
) -> np.ndarray:
    """Dequantizes one `QMoE` `fc1_experts_weights`/`fc2_experts_weights`
    tensor (raw ``[num_experts, N, K/pack]`` storage, `N` output channels by
    `K` input features) to a ``float64`` ``[num_experts, N, K]`` array,
    ``(quantized - zero_point) * scale`` per output channel `N` -- `K`
    packing is always the flat, block-boundary-oblivious
    :func:`_unpack_subbyte` convention (confirmed unchanged by `block_size`,
    see this section's own top comment, point 3).

    With `block_size` absent/0 (the default), `scale`/`zero_points` are one
    entry per `N` channel, over the whole `K` row at once (point 2).  With
    `block_size` set, `scale` is one entry per `(N, K // block_size)` pair
    (3D, ``[E, N, K // block_size]``) and, when present, `zero_points` packs
    along that *same trailing* `K`-block axis instead of `N` (point 3) --
    ``dequantized[e, n, k] = (code[e, n, k] - zero_point[e, n, k //
    block_size]) * scale[e, n, k // block_size]``. `zero_points`, when
    absent, defaults to :func:`_qmoe_default_zero_point` for every
    channel/block, exactly as the op's own schema documents.
    """
    packed_arr = onnx.numpy_helper.to_array(packed)  # [E, N, K/pack]
    q = _unpack_subbyte(packed_arr, bits, k).astype(np.float64)  # [E, N, K]
    if block_size:
        num_experts, n = packed_arr.shape[0], packed_arr.shape[1]
        k_blocks = k // block_size
        scale_arr = _to_f64(scale)  # [E, N, k_blocks]
        if zero_points is not None:
            zp_packed = onnx.numpy_helper.to_array(zero_points)  # [E, N, k_blocks/pack]
            zp = _unpack_subbyte(zp_packed, bits, k_blocks).astype(
                np.float64
            )  # [E, N, k_blocks]
        else:
            zp = np.full(
                (num_experts, n, k_blocks), float(_qmoe_default_zero_point(bits))
            )
        q_blocks = q.reshape(num_experts, n, k_blocks, block_size)
        dequant = (q_blocks - zp[:, :, :, None]) * scale_arr[:, :, :, None]
        return dequant.reshape(num_experts, n, k)

    scale_arr = _to_f64(scale)  # [E, N]
    if zero_points is not None:
        zp_packed = onnx.numpy_helper.to_array(zero_points)  # [E, N/pack]
        zp = _unpack_subbyte(zp_packed, bits, packed_arr.shape[1]).astype(
            np.float64
        )  # [E, N]
    else:
        zp = np.full(scale_arr.shape, float(_qmoe_default_zero_point(bits)))
    return (q - zp[:, :, None]) * scale_arr[:, :, None]


@dataclass(frozen=True)
class _QMoEChannelChain:
    node: onnx.NodeProto
    fc1_w: str
    fc1_scale: str
    fc1_bias: Optional[str]
    fc1_zp: Optional[str]
    fc2_w: str
    fc2_scale: str
    fc2_bias: Optional[str]
    fc2_zp: Optional[str]
    num_experts: int
    inter_size: int
    hidden_size: int
    bits: int
    block_size: int


def _match_qmoe_producer(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
) -> Optional[_QMoEChannelChain]:
    """If `node` is a ``com.microsoft::QMoE`` node this pass can safely
    prune the `inter_size` axis of, returns the matched
    :class:`_QMoEChannelChain`. Mirrors :func:`_match_moe_producer`'s own
    `activation_type`/`swiglu_fusion`/`fc3` checks exactly, then adds the
    quantization-specific checks this section's own top comment derives
    (points 1-5): `quant_type` must be (or default to) ``'int'``,
    `expert_weight_bits` one of ``{2, 4, 8}``, `block_size` either absent/0
    (whole-row scale) or a groupwise value the CPU kernel itself accepts
    (``>= 16``, a multiple of ``8 // expert_weight_bits``, and dividing both
    `hidden_size` and `inter_size` evenly), `weights_prepacked` absent/
    ``-1``/``0``, `router_weights` and every FP4/FP8-only input (15-20)
    absent, `fc1_experts_weights`/`fc2_experts_weights` constant ``uint8``
    rank-3 initializers with a single consumer, `fc1_scales`/`fc2_scales`
    constant float initializers (single consumer) matching shape -- rank-2
    when `block_size` is absent/0, rank-3 (gaining a trailing `K`-block
    axis) when set -- and `fc1_experts_bias`/`fc2_experts_bias`/
    `fc1_zero_points`/`fc2_zero_points`, when present, the same
    single-consumer/shape checks (`zero_points`' own packed axis moves from
    `N` to the trailing `K`-block axis when `block_size` is set -- see this
    section's own top comment, point 3).
    """
    if node.domain != _MOE_DOMAIN or node.op_type != "QMoE":
        return None

    activation = "relu"
    swiglu_fusion = 0
    quant_type = "int"
    bits = 4
    block_size = 0
    weights_prepacked = -1
    for attr in node.attribute:
        if attr.name == "activation_type":
            activation = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
        elif attr.name == "swiglu_fusion":
            swiglu_fusion = attr.i
        elif attr.name == "quant_type":
            quant_type = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
        elif attr.name == "expert_weight_bits":
            bits = attr.i
        elif attr.name == "block_size":
            block_size = attr.i
        elif attr.name == "weights_prepacked":
            weights_prepacked = attr.i

    if activation not in _MOE_ACTIVATIONS or swiglu_fusion != 0:
        return None
    if quant_type != "int":
        return None  # fp4/nvfp4/fp8/wfp4afp8 -- a different tensor format
        # entirely (MXFP block-scaled or float8e4m3fn-packed weights,
        # separate activation-scale operands); out of scope, see this
        # section's own top comment (point 1).
    if bits not in (2, 4, 8):
        return None
    pack = 8 // bits
    if block_size != 0:
        if block_size < 16:
            return None  # the CPU kernel's own floor -- "block_size must be
            # >= 16 when provided"; see point 3.
        if block_size % pack != 0:
            return None  # not byte-aligned to the sub-byte packing width --
            # every real block_size this op's kernel accepts already is, so
            # this excludes no genuine configuration; see point 3.
    if weights_prepacked not in (-1, 0):
        return None  # a CUTLASS mixed-GEMM prepacked byte layout, not the
        # raw [N, K/pack] storage this pass slices; see point 4.

    if len(node.input) > 8 and node.input[8]:
        return None  # fc3_experts_weights present -- CPU QMoE errors "FC3
        # gating is not yet implemented on CPU for QMoE" (confirmed
        # empirically); see point 5.
    if len(node.input) > 14 and node.input[14]:
        return None  # router_weights (DeepSeek-style select/aggregate with
        # a separate combine-weight tensor) -- a topology this pass's own
        # masking-equivalence argument (see the whole-expert section below)
        # was never verified against.
    for idx in range(15, 21):
        # fc1/fc2_global_scale, fc1/fc2_act_scale, fc1/fc2_act_block_scale
        # -- FP4/FP8-activation-only inputs, naturally absent for
        # quant_type='int' in every real export but checked explicitly.
        if len(node.input) > idx and node.input[idx]:
            return None

    if len(node.input) < 7 or not node.input[2] or not node.input[5]:
        return None
    fc1_w_name, fc2_w_name = node.input[2], node.input[5]
    fc1_w = initializer_map.get(fc1_w_name)
    fc2_w = initializer_map.get(fc2_w_name)
    if (
        fc1_w is None
        or fc2_w is None
        or fc1_w.data_type != onnx.TensorProto.UINT8
        or fc2_w.data_type != onnx.TensorProto.UINT8
        or len(fc1_w.dims) != 3
        or len(fc2_w.dims) != 3
        or len(consumers_of.get(fc1_w_name, [])) != 1
        or len(consumers_of.get(fc2_w_name, [])) != 1
    ):
        return None

    num_experts, inter_size, fc1_k_packed = (int(d) for d in fc1_w.dims)
    fc2_num_experts, hidden_size, fc2_k_packed = (int(d) for d in fc2_w.dims)
    if (
        fc2_num_experts != num_experts
        or fc1_k_packed * pack != hidden_size
        or fc2_k_packed * pack != inter_size
    ):
        return None  # also rules out a fused-swiglu fc1 (doubled row count)
    if block_size != 0 and (
        hidden_size % block_size != 0 or inter_size % block_size != 0
    ):
        return None  # partial/padded final block -- declined, see point 3;
        # also transitively enforces the schema doc's own extra
        # "hidden_size/inter_size divisible by block_size * pack_size" rule
        # for whichever of fc1/fc2 has zero_points present, via the
        # fc1_zp/fc2_zp expected-shape checks below.

    if not node.input[3] or not node.input[6]:
        return None  # fc1_scales/fc2_scales required by this pass, even
        # though the schema itself marks them optional -- without a scale
        # this pass has nothing to dequantize with for importance ranking.
    fc1_scale_name, fc2_scale_name = node.input[3], node.input[6]
    fc1_scale = initializer_map.get(fc1_scale_name)
    fc2_scale = initializer_map.get(fc2_scale_name)
    if block_size:
        hidden_blocks = hidden_size // block_size
        inter_blocks = inter_size // block_size
        fc1_scale_dims = [num_experts, inter_size, hidden_blocks]
        fc2_scale_dims = [num_experts, hidden_size, inter_blocks]
    else:
        hidden_blocks = inter_blocks = 0  # unused (no block axis)
        fc1_scale_dims = [num_experts, inter_size]
        fc2_scale_dims = [num_experts, hidden_size]
    if (
        fc1_scale is None
        or fc2_scale is None
        or not _is_supported_float_dtype(fc1_scale.data_type)
        or not _is_supported_float_dtype(fc2_scale.data_type)
        or list(fc1_scale.dims) != fc1_scale_dims
        or list(fc2_scale.dims) != fc2_scale_dims
        or len(consumers_of.get(fc1_scale_name, [])) != 1
        or len(consumers_of.get(fc2_scale_name, [])) != 1
    ):
        return None

    def _optional_float(
        index: int, expected_dims: List[int]
    ) -> Tuple[bool, Optional[str]]:
        if index >= len(node.input) or not node.input[index]:
            return True, None
        name = node.input[index]
        init = initializer_map.get(name)
        if (
            init is None
            or not _is_supported_float_dtype(init.data_type)
            or list(init.dims) != expected_dims
            or len(consumers_of.get(name, [])) != 1
        ):
            return False, None
        return True, name

    def _optional_uint8(
        index: int, expected_dims: List[int]
    ) -> Tuple[bool, Optional[str]]:
        if index >= len(node.input) or not node.input[index]:
            return True, None
        name = node.input[index]
        init = initializer_map.get(name)
        if (
            init is None
            or init.data_type != onnx.TensorProto.UINT8
            or list(init.dims) != expected_dims
            or len(consumers_of.get(name, [])) != 1
        ):
            return False, None
        return True, name

    ok, fc1_bias_name = _optional_float(4, [num_experts, inter_size])
    if not ok:
        return None
    ok, fc2_bias_name = _optional_float(7, [num_experts, hidden_size])
    if not ok:
        return None
    # zero_points' own packed axis moves from N (whole-row case) to the
    # trailing K-block axis (blockwise case) -- see this section's own top
    # comment, point 3.
    if block_size:
        fc1_zp_dims = [num_experts, inter_size, hidden_blocks // pack]
        fc2_zp_dims = [num_experts, hidden_size, inter_blocks // pack]
    else:
        fc1_zp_dims = [num_experts, inter_size // pack]
        fc2_zp_dims = [num_experts, hidden_size // pack]
    ok, fc1_zp_name = _optional_uint8(11, fc1_zp_dims)
    if not ok:
        return None
    ok, fc2_zp_name = _optional_uint8(12, fc2_zp_dims)
    if not ok:
        return None

    return _QMoEChannelChain(
        node=node,
        fc1_w=fc1_w_name,
        fc1_scale=fc1_scale_name,
        fc1_bias=fc1_bias_name,
        fc1_zp=fc1_zp_name,
        fc2_w=fc2_w_name,
        fc2_scale=fc2_scale_name,
        fc2_bias=fc2_bias_name,
        fc2_zp=fc2_zp_name,
        num_experts=num_experts,
        inter_size=inter_size,
        hidden_size=hidden_size,
        block_size=block_size,
        bits=bits,
    )


def _find_qmoe_chains(graph: onnx.GraphProto) -> List[_QMoEChannelChain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    chains = []
    for node in graph.node:
        chain = _match_qmoe_producer(node, initializer_map, consumers_of)
        if chain is not None:
            chains.append(chain)
    return chains


def _qmoe_channel_importance(
    chain: _QMoEChannelChain, initializer_map: Dict[str, onnx.TensorProto]
) -> np.ndarray:
    """Combined (root-sum-square) L2-norm importance per `inter_size`
    channel index, mirroring :func:`_moe_importance` exactly but computed
    over each weight's own *dequantized* value (:func:`_qmoe_dequantize`)
    rather than a plain float initializer -- the packed ``uint8`` bytes
    themselves carry no usable magnitude information on their own.
    """
    fc1_dq = _qmoe_dequantize(
        initializer_map[chain.fc1_w],
        initializer_map[chain.fc1_scale],
        initializer_map[chain.fc1_zp] if chain.fc1_zp is not None else None,
        chain.bits,
        chain.hidden_size,
        chain.block_size,
    )  # [E, inter_size, hidden_size]
    fc2_dq = _qmoe_dequantize(
        initializer_map[chain.fc2_w],
        initializer_map[chain.fc2_scale],
        initializer_map[chain.fc2_zp] if chain.fc2_zp is not None else None,
        chain.bits,
        chain.inter_size,
        chain.block_size,
    )  # [E, hidden_size, inter_size]
    squared = np.sum(np.square(fc1_dq), axis=(0, 2)) + np.sum(
        np.square(fc2_dq), axis=(0, 1)
    )
    if chain.fc1_bias is not None:
        fc1_b = _to_f64(initializer_map[chain.fc1_bias])
        squared = squared + np.sum(np.square(fc1_b), axis=0)
    return np.sqrt(squared)


def _qmoe_block_aligned_keep(
    importance: np.ndarray, n: int, block_size: int, sparsity: float
) -> Optional[np.ndarray]:
    """Resolves a target `inter_size` keep-set to whole `block_size`-sized
    groups: aggregates `importance` (one entry per channel, `n` long) into
    one combined (root-sum-square) score per `block_size`-sized block --
    the same reduction :func:`_qmoe_channel_importance` already uses across
    weights, just one level up -- ranks *blocks* by that score, and keeps
    the top ``max(1, n // block_size - round(n // block_size * sparsity))``
    of them, exactly mirroring how the whole-row case ranks and keeps
    individual channels (see this section's own top comment, point 3, and
    :func:`_matmul_nbits_block_aligned_keep_blocks` for the analogous
    ``MatMulNBits`` precedent). Ranking/keeping at block granularity from
    the start -- rather than computing a channel-level keep-set and
    checking whether it happens to land on block boundaries -- means the
    result is *always* block-aligned by construction, never partial. Returns
    ``None`` when every block would be kept (rounds down to nothing to
    prune for this layer).
    """
    num_blocks = n // block_size
    block_importance = np.sqrt(
        np.sum(importance.reshape(num_blocks, block_size) ** 2, axis=1)
    )
    keep_blocks_count = max(1, num_blocks - round(num_blocks * sparsity))
    if keep_blocks_count >= num_blocks:
        return None
    keep_block_idx = np.sort(np.argsort(-block_importance)[:keep_blocks_count])
    return np.arange(n).reshape(num_blocks, block_size)[keep_block_idx].reshape(-1)


def _apply_qmoe_channel_chains(
    graph: onnx.GraphProto,
    chains: List[_QMoEChannelChain],
    sparsity: float,
    compute_importance,
) -> None:
    initializer_map = {t.name: t for t in graph.initializer}
    touched: Set[str] = set()

    for chain in chains:
        weight_names = {chain.fc1_w, chain.fc2_w, chain.fc1_scale}
        if chain.fc1_bias is not None:
            weight_names.add(chain.fc1_bias)
        if chain.fc1_zp is not None:
            weight_names.add(chain.fc1_zp)
        if chain.block_size:
            # Blockwise fc2_scales -- and fc2_zero_points, if present -- are
            # now also mutated below (their own trailing axis is
            # inter_size-block-indexed, see this section's own top comment,
            # point 3), unlike the whole-row case where neither depends on
            # inter_size at all.
            weight_names.add(chain.fc2_scale)
            if chain.fc2_zp is not None:
                weight_names.add(chain.fc2_zp)
        if weight_names & touched:
            continue  # a shared/tied initializer another MoE/QMoE node
            # already resized
        touched |= weight_names

        n = chain.inter_size
        pack = 8 // chain.bits
        block_size = chain.block_size

        if block_size:
            # Resolved at block granularity from the start (never a
            # channel-level keep-set checked for alignment after the fact)
            # -- see :func:`_qmoe_block_aligned_keep` and this section's own
            # top comment, point 3. A block is always a multiple of `pack`
            # (the matcher itself requires `block_size % pack == 0`), so
            # this also transitively satisfies the same "survivor count is
            # a multiple of `pack`" constraint the whole-row case floors to
            # separately below.
            importance = compute_importance(chain, initializer_map)
            keep = _qmoe_block_aligned_keep(importance, n, block_size, sparsity)
            if keep is None:
                continue  # rounds down to nothing for this layer -- no-op
            keep_block_idx = np.unique(keep // block_size)
        else:
            keep_count = max(1, n - round(n * sparsity))
            # The survivor count must itself stay an exact multiple of
            # `pack`: confirmed empirically (a real CPU
            # onnxruntime.InferenceSession rejects a mismatched-shape node)
            # that this environment's own QMoE kernel derives `inter_size`
            # purely from `fc2_experts_weights`' own packed last axis
            # length times `pack` -- it has no way to represent "the last
            # packed byte's own high nibble is unused padding, not a real
            # channel" the way :func:`_pack_subbyte` alone (a general,
            # correct-for-any-length helper) can round-trip internally. So
            # a survivor count that isn't itself a multiple of `pack` is
            # rounded *down* to the nearest one here (floored at `pack`,
            # never below it) rather than ever handed to
            # :func:`_pack_subbyte` as the pruned channel count -- the
            # "genuinely bit-packed odd-count" hazard this section's own
            # top comment flags is resolved by never producing such a
            # shape in the first place, not by attempting to represent
            # one.
            keep_count = max(pack, (keep_count // pack) * pack)
            if keep_count >= n:
                continue  # rounds down to nothing for this layer -- no-op
            importance = compute_importance(chain, initializer_map)
            keep = np.sort(np.argsort(-importance)[:keep_count])

        # fc1_experts_weights: [E, inter_size, hidden_size/pack] -- the
        # pruned axis (1) is the *unpacked* one (packing lives on axis 2,
        # untouched), so this is a plain index-select, no unpack/repack
        # needed at all -- unaffected by block_size (fc1's own blocks, if
        # any, group along hidden_size, an axis this pass never touches).
        fc1_w_init = initializer_map[chain.fc1_w]
        fc1_new = onnx.numpy_helper.to_array(fc1_w_init)[:, keep, :]
        fc1_w_init.CopyFrom(
            onnx.numpy_helper.from_array(
                np.ascontiguousarray(fc1_new), name=chain.fc1_w
            )
        )

        # fc1_scales: [E, inter_size] (whole-row) or [E, inter_size,
        # hidden_blocks] (blockwise) -- `keep` always indexes axis 1 either
        # way, a plain row-slice; numpy leaves a trailing hidden_blocks axis
        # (if any) untouched automatically.
        fc1_scale_init = initializer_map[chain.fc1_scale]
        fc1_scale_new = onnx.numpy_helper.to_array(fc1_scale_init)[:, keep]
        fc1_scale_init.CopyFrom(
            onnx.numpy_helper.from_array(
                np.ascontiguousarray(fc1_scale_new), name=chain.fc1_scale
            )
        )

        if chain.fc1_bias is not None:
            fc1_bias_init = initializer_map[chain.fc1_bias]
            fc1_bias_new = onnx.numpy_helper.to_array(fc1_bias_init)[:, keep]
            fc1_bias_init.CopyFrom(
                onnx.numpy_helper.from_array(
                    np.ascontiguousarray(fc1_bias_new), name=chain.fc1_bias
                )
            )

        if chain.fc1_zp is not None:
            fc1_zp_init = initializer_map[chain.fc1_zp]
            if block_size:
                # fc1_zero_points: [E, inter_size, hidden_blocks/pack] --
                # blockwise moves the packed axis off of N entirely (see
                # this section's own top comment, point 3), so slicing N
                # (axis 1) is now a plain index-select, no unpack/repack
                # needed at all.
                zp_new = onnx.numpy_helper.to_array(fc1_zp_init)[:, keep, :]
            else:
                # fc1_zero_points: [E, inter_size/pack] -- packed along the
                # same axis being pruned, so this one genuinely needs
                # unpack/select/repack (see this section's own top comment,
                # point 2).
                zp_unpacked = _unpack_subbyte(
                    onnx.numpy_helper.to_array(fc1_zp_init), chain.bits, n
                )
                zp_new = _pack_subbyte(zp_unpacked[:, keep], chain.bits)
            fc1_zp_init.CopyFrom(
                onnx.numpy_helper.from_array(
                    np.ascontiguousarray(zp_new), name=chain.fc1_zp
                )
            )

        # fc2_experts_weights: [E, hidden_size, inter_size/pack] -- the
        # pruned axis (inter_size) *is* the packed one here, so this needs
        # a real unpack/select/repack round trip. Packing is flat/
        # block-boundary-oblivious (confirmed empirically, see this
        # section's own top comment, point 3), so this is unaffected by
        # block_size beyond `keep` itself already being block-aligned.
        fc2_w_init = initializer_map[chain.fc2_w]
        fc2_unpacked = _unpack_subbyte(
            onnx.numpy_helper.to_array(fc2_w_init), chain.bits, n
        )  # [E, hidden_size, inter_size]
        fc2_new = _pack_subbyte(fc2_unpacked[:, :, keep], chain.bits)
        fc2_w_init.CopyFrom(
            onnx.numpy_helper.from_array(
                np.ascontiguousarray(fc2_new), name=chain.fc2_w
            )
        )

        if block_size:
            # fc2_scales/fc2_zero_points: [E, hidden_size, inter_blocks(/
            # pack)] -- unlike the whole-row case (below), these now DO
            # depend on inter_size, via the block axis, so they must be cut
            # too -- by *block* index (`keep_block_idx`), not channel index
            # (`keep`).
            fc2_scale_init = initializer_map[chain.fc2_scale]
            fc2_scale_new = onnx.numpy_helper.to_array(fc2_scale_init)[
                :, :, keep_block_idx
            ]
            fc2_scale_init.CopyFrom(
                onnx.numpy_helper.from_array(
                    np.ascontiguousarray(fc2_scale_new), name=chain.fc2_scale
                )
            )
            if chain.fc2_zp is not None:
                fc2_zp_init = initializer_map[chain.fc2_zp]
                inter_blocks = n // block_size
                fc2_zp_unpacked = _unpack_subbyte(
                    onnx.numpy_helper.to_array(fc2_zp_init), chain.bits, inter_blocks
                )  # [E, hidden_size, inter_blocks]
                fc2_zp_new = _pack_subbyte(
                    fc2_zp_unpacked[:, :, keep_block_idx], chain.bits
                )
                fc2_zp_init.CopyFrom(
                    onnx.numpy_helper.from_array(
                        np.ascontiguousarray(fc2_zp_new), name=chain.fc2_zp
                    )
                )
        # fc2_experts_bias always indexes hidden_size, fc2's own *output*
        # axis -- unaffected by an inter_size cut regardless of block_size,
        # the same reasoning plain MoE's own fc2 bias already gets -- never
        # sliced here. In the whole-row case (block_size == 0),
        # fc2_scales/fc2_zero_points are the same "indexes hidden_size
        # only" shape and are likewise never sliced.


def apply_qmoe_expert_channel_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
) -> onnx.ModelProto:
    """Removes intermediate (`inter_size`) channels from every expert of a
    matched ``com.microsoft::QMoE`` node at once -- the quantized-weight
    counterpart of :func:`apply_moe_expert_channel_pruning`, targeting
    `QMoE`'s own packed ``uint8`` `fc1`/`fc2` weights (plus their
    `scales`/`zero_points`, co-sliced in lockstep) instead of plain `MoE`'s
    float ones. Supports both `quant_type='int'` with no `block_size`
    (whole-row per-channel scale) and with a groupwise `block_size` set
    (see this section's own comment above, :func:`_match_qmoe_producer`,
    for the exact matched shape either way and every empirically-confirmed
    decline: `quant_type` other than ``'int'``, a prepacked layout, `fc3`,
    `router_weights`, FP4/FP8-only inputs, an unrecognized/`swiglu`
    activation, or a `block_size` this pass's own scope boundary excludes
    -- below 16, not a multiple of ``8 // expert_weight_bits``, or not
    dividing `hidden_size`/`inter_size` evenly).

    Ranks every `inter_size` index by combined (root-sum-square) L2 norm of
    `fc1_experts_weights`'/`fc2_experts_weights`' own *dequantized* row/
    column (:func:`_qmoe_dequantize`, across every expert and `hidden_size`
    at once) plus `fc1_experts_bias`'s own entry when present -- the same
    criterion :func:`apply_moe_expert_channel_pruning` uses, just computed
    over dequantized rather than already-float values.

    With no `block_size`, drops the lowest-``sparsity``-fraction of indices
    directly and removes the matching row from `fc1_experts_weights`/
    `fc1_scales`/`fc1_experts_bias`/`fc1_zero_points` and column from
    `fc2_experts_weights`, identically across every expert.
    `fc1_experts_weights`' pruned axis is never the packed one (a plain
    index-select); `fc2_experts_weights`'/`fc1_zero_points`' own pruned axis
    *is* the packed one, so both go through a real unpack/select/repack
    round trip (:func:`_unpack_subbyte`/:func:`_pack_subbyte`). The
    *survivor* count is always rounded down to a multiple of
    ``8 // expert_weight_bits`` (never below it) before that repack --
    confirmed empirically that this environment's own CPU `QMoE` kernel
    derives `inter_size` from `fc2_experts_weights`' own packed byte count
    alone, with no way to represent a partial/padded trailing byte, so this
    pass never asks :func:`_pack_subbyte` to produce one.

    With `block_size` set, the keep-set is instead resolved to whole
    `block_size`-sized groups before ever slicing -- ranking groups (not
    individual channels) by the same combined-L2-norm criterion aggregated
    across each group (:func:`_qmoe_block_aligned_keep`), since a group's
    shared scale/zero-point can't be split -- because `fc2`'s own blocks
    group along `inter_size` (this pass's own pruned axis) while `fc1`'s
    group along the untouched `hidden_size`, `fc2_scales`/
    `fc2_zero_points` also need slicing (by *block* index) in this case,
    unlike the no-`block_size` case where neither depends on `inter_size` at
    all; `fc1_zero_points`' own packed axis moves off of `inter_size`
    entirely when blocked, so it becomes a plain index-select instead of an
    unpack/repack round trip. See :func:`_apply_qmoe_channel_chains`'s own
    comments for the exact per-tensor mechanics either way. `num_experts`,
    `k`, and every node attribute are untouched.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of each matched node's `inter_size`
            channels (or, with `block_size` set, `block_size`-sized groups)
            to remove (rounded down to a multiple of
            ``8 // expert_weight_bits`` -- or, with `block_size` set, of
            `block_size` itself, which is always also a multiple of
            ``8 // expert_weight_bits`` -- floored at that same value --
            never zero -- so the result stays a shape the real `QMoE` CPU
            kernel can execute)
    :returns: ``model`` with every matched ``QMoE`` node's `fc1`/`fc2`
            tensors (and `fc1`'s own `scales`/`bias`/`zero_points`, plus
            `fc2`'s own `scales`/`zero_points` too when `block_size` is
            set) resized in place; any node this pass doesn't recognize is
            left completely untouched
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = _find_qmoe_chains(graph)
    if chains:
        _apply_qmoe_channel_chains(graph, chains, sparsity, _qmoe_channel_importance)
    return out


@dataclass(frozen=True)
class _QMoEExpertChain:
    node: onnx.NodeProto
    fc1_w: str
    fc1_scale: str
    fc1_bias: Optional[str]
    fc1_zp: Optional[str]
    fc2_w: str
    fc2_scale: str
    fc2_bias: Optional[str]
    fc2_zp: Optional[str]
    num_experts: int
    inter_size: int
    hidden_size: int
    bits: int
    block_size: int
    k: int
    router_probs: str
    router_w: str
    router_w_transposed: bool
    router_b: Optional[str]


def _match_qmoe_whole_expert_producer(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    node_by_output: Dict[str, onnx.NodeProto],
    graph_outputs: Set[str],
) -> Optional[_QMoEExpertChain]:
    """If `node` is a ``com.microsoft::QMoE`` node this pass can safely
    prune whole experts (the `num_experts` axis) from, returns the matched
    :class:`_QMoEExpertChain`. Mirrors :func:`_match_moe_whole_expert_producer`
    exactly: reuses :func:`_match_qmoe_producer`'s own checks outright
    (block_size and all -- see that function's own docstring), then adds
    the identical `k`/`use_sparse_mixer` checks and `router_probs` producer
    match -- see this section's own top comment (point 7) for why every one
    of those transfers unchanged from plain `MoE` to `QMoE`, and point 8 for
    why a `block_size`-matched chain needs no extra handling at all here
    (every per-expert tensor keeps `num_experts` as its own leading axis
    regardless of `block_size`).
    """
    base = _match_qmoe_producer(node, initializer_map, consumers_of)
    if base is None:
        return None

    k = None
    use_sparse_mixer = 0
    for attr in node.attribute:
        if attr.name == "k":
            k = attr.i
        elif attr.name == "use_sparse_mixer":
            use_sparse_mixer = attr.i
    if k is None or k < 1 or k > base.num_experts or use_sparse_mixer != 0:
        return None

    if len(node.input) < 2 or not node.input[1]:
        return None
    router_probs = node.input[1]
    if router_probs in graph_outputs or len(consumers_of.get(router_probs, [])) != 1:
        return None
    router_node = node_by_output.get(router_probs)
    if router_node is None:
        return None
    router_info = _match_producer(router_node, initializer_map)
    if router_info is None:
        return None
    router_w_name, router_w_transposed, router_b_name, n_channels = router_info
    if n_channels != base.num_experts or len(consumers_of.get(router_w_name, [])) != 1:
        return None
    if router_b_name is not None and len(consumers_of.get(router_b_name, [])) != 1:
        return None

    return _QMoEExpertChain(
        node=node,
        fc1_w=base.fc1_w,
        fc1_scale=base.fc1_scale,
        fc1_bias=base.fc1_bias,
        fc1_zp=base.fc1_zp,
        fc2_w=base.fc2_w,
        fc2_scale=base.fc2_scale,
        fc2_bias=base.fc2_bias,
        fc2_zp=base.fc2_zp,
        num_experts=base.num_experts,
        inter_size=base.inter_size,
        hidden_size=base.hidden_size,
        bits=base.bits,
        block_size=base.block_size,
        k=int(k),
        router_probs=router_probs,
        router_w=router_w_name,
        router_w_transposed=router_w_transposed,
        router_b=router_b_name,
    )


def _find_qmoe_whole_expert_chains(graph: onnx.GraphProto) -> List[_QMoEExpertChain]:
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    node_by_output = {out: node for node in graph.node for out in node.output}
    graph_outputs = {o.name for o in graph.output}
    chains = []
    for node in graph.node:
        chain = _match_qmoe_whole_expert_producer(
            node, initializer_map, consumers_of, node_by_output, graph_outputs
        )
        if chain is not None:
            chains.append(chain)
    return chains


def _qmoe_expert_weight_importance(
    chain: _QMoEExpertChain, initializer_map: Dict[str, onnx.TensorProto]
) -> np.ndarray:
    """Combined (root-sum-square) L2-norm importance per *expert*, mirroring
    :func:`_moe_expert_weight_importance` -- the weight-magnitude-only
    fallback used when no calibration data was observed for a chain's
    `router_probs` -- computed over each weight's own dequantized value
    (:func:`_qmoe_dequantize`) the same way :func:`_qmoe_channel_importance`
    is, just reduced *within* each expert's own slice instead of *across*
    experts.
    """
    fc1_dq = _qmoe_dequantize(
        initializer_map[chain.fc1_w],
        initializer_map[chain.fc1_scale],
        initializer_map[chain.fc1_zp] if chain.fc1_zp is not None else None,
        chain.bits,
        chain.hidden_size,
        chain.block_size,
    )  # [E, inter_size, hidden_size]
    fc2_dq = _qmoe_dequantize(
        initializer_map[chain.fc2_w],
        initializer_map[chain.fc2_scale],
        initializer_map[chain.fc2_zp] if chain.fc2_zp is not None else None,
        chain.bits,
        chain.inter_size,
        chain.block_size,
    )  # [E, hidden_size, inter_size]
    squared = np.sum(np.square(fc1_dq), axis=(1, 2)) + np.sum(
        np.square(fc2_dq), axis=(1, 2)
    )
    if chain.fc1_bias is not None:
        fc1_b = _to_f64(initializer_map[chain.fc1_bias])
        squared = squared + np.sum(np.square(fc1_b), axis=1)
    return np.sqrt(squared)


def _apply_qmoe_whole_expert_chains(
    graph: onnx.GraphProto,
    chains: List[_QMoEExpertChain],
    sparsity: float,
    compute_importance,
) -> Set[str]:
    """Drops whole experts from every matched `QMoE` chain -- unlike
    :func:`_apply_qmoe_channel_chains`, `num_experts` is *every* per-expert
    tensor's own leading axis (confirmed from the schema: packing always
    lives on a *later* axis, never axis 0), so every one of
    `fc1`/`fc2`'s weights/scales/biases/zero_points, plus the router
    projection's own weight/bias, is a plain leading-axis index-select --
    no unpack/repack needed anywhere, the "comparatively simpler" half of
    this section's own two techniques.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    touched: Set[str] = set()
    stale_value_info: Set[str] = set()

    for chain in chains:
        weight_names = {
            chain.fc1_w,
            chain.fc2_w,
            chain.fc1_scale,
            chain.fc2_scale,
            chain.router_w,
        }
        if chain.fc1_bias is not None:
            weight_names.add(chain.fc1_bias)
        if chain.fc2_bias is not None:
            weight_names.add(chain.fc2_bias)
        if chain.fc1_zp is not None:
            weight_names.add(chain.fc1_zp)
        if chain.fc2_zp is not None:
            weight_names.add(chain.fc2_zp)
        if chain.router_b is not None:
            weight_names.add(chain.router_b)
        if weight_names & touched:
            continue  # a shared/tied initializer another MoE/QMoE node
            # already resized
        touched |= weight_names

        n = chain.num_experts
        floor = max(1, min(chain.k, n))
        keep_count = max(floor, n - round(n * sparsity))
        if keep_count >= n:
            continue  # rounds down to nothing (or below k) for this layer

        importance = compute_importance(chain, initializer_map)
        keep = np.sort(np.argsort(-importance)[:keep_count])

        for name in (chain.fc1_w, chain.fc2_w, chain.fc1_scale, chain.fc2_scale):
            _slice_axis(initializer_map[name], keep, axis=0)
        for opt_name in (chain.fc1_bias, chain.fc2_bias, chain.fc1_zp, chain.fc2_zp):
            if opt_name is not None:
                _slice_axis(initializer_map[opt_name], keep, axis=0)

        router_w_init = initializer_map[chain.router_w]
        _slice_producer_weight(router_w_init, chain.router_w_transposed, keep)
        if chain.router_b is not None:
            _slice_last_axis(initializer_map[chain.router_b], keep)

        stale_value_info.add(chain.router_probs)

    return stale_value_info


def apply_qmoe_whole_expert_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Removes whole experts (shrinks the `num_experts` leading axis) from a
    matched ``com.microsoft::QMoE`` node and its upstream router projection
    at once -- the quantized-weight counterpart of
    :func:`apply_moe_whole_expert_pruning`, ranking experts by the exact
    same calibration-based mean router gate weight
    (:func:`_moe_router_gate_calibration_stats`, reused unchanged --
    `router_probs` is `QMoE`'s own second input, upstream of and oblivious
    to its quantized `fc1`/`fc2`, so this needs no `QMoE`-specific version
    at all). See this section's own top comment (point 7) for the full
    masking-equivalence safety argument, re-derived and re-verified
    specifically against a real `QMoE` node -- in particular why point 6's
    own finding (`QMoE` always renormalizes top-`k` gate weights regardless
    of `normalize_routing_weights`) does not change that argument's own
    conclusion, only how it had to be re-checked. Works unchanged for a
    matched chain with `block_size` set (groupwise `quant_type='int'`
    quantization, see :func:`_match_qmoe_producer`) -- see this section's
    own top comment, point 8: every per-expert tensor keeps `num_experts`
    as its own leading axis regardless of `block_size`, so no
    `block_size`-specific handling is needed anywhere in this function.

    Every matched expert's `fc1_experts_weights`/`fc2_experts_weights`/
    `fc1_scales`/`fc2_scales` (and `fc1_experts_bias`/`fc2_experts_bias`/
    `fc1_zero_points`/`fc2_zero_points`, if present) row, and the router
    projection weight's (and bias's, if present) matching output column,
    are dropped together for the lowest-``sparsity``-fraction of experts by
    that ranking -- with `num_experts_to_keep` silently floored at the
    node's own `k`, never adjusting `k` itself, the same floor
    :func:`apply_moe_whole_expert_pruning` uses. A chain whose
    `router_probs` was never observed during calibration falls back to each
    expert's own dequantized `fc1`/`fc2` (+`fc1_experts_bias`) combined L2
    weight norm (:func:`_qmoe_expert_weight_importance`), the same
    "no matching activation observed" fallback
    :func:`apply_structured_wanda_pruning` already uses.

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to rank experts by
            mean router gate weight on. Each batch is a
            ``{input_name: np.ndarray}`` dict matching ``model``'s graph
            inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: target fraction of each matched node's `num_experts` to
            remove (floored at the node's own `k`, so fewer may actually be
            removed -- never more)
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration router activations
    :returns: ``model`` with every matched ``QMoE`` node's per-expert
            tensors and its router projection's weight(/bias) resized in
            place; any node this pass doesn't recognize is left completely
            untouched
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    chains = _find_qmoe_whole_expert_chains(graph)
    if not chains:
        return out

    mean_gate_weight = _moe_router_gate_calibration_stats(
        out, chains, calibration_data, providers
    )

    def _importance(
        chain: _QMoEExpertChain, initializer_map: Dict[str, onnx.TensorProto]
    ) -> np.ndarray:
        gate = mean_gate_weight.get(chain.router_probs)
        if gate is None or gate.shape[0] != chain.num_experts:
            return _qmoe_expert_weight_importance(chain, initializer_map)
        return gate

    stale_value_info = _apply_qmoe_whole_expert_chains(
        graph, chains, sparsity, _importance
    )
    if stale_value_info:
        kept = [vi for vi in graph.value_info if vi.name not in stale_value_info]
        del graph.value_info[:]
        graph.value_info.extend(kept)
    return out


# --- Embedding / lm_head vocabulary pruning --------------------------------
#
# Every other pass in this module changes a graph's *internal* channel/
# head/entry counts while leaving what counts as a valid *input* untouched.
# This one is different in kind: it targets the ``vocab_size`` axis of a
# token-embedding table (and, where identifiable, the tied/untied output
# projection back to vocab logits, "lm_head") -- for an LLM this is very
# often the single largest weight tensor in the whole model, so a
# deployment restricted to a known, narrower vocabulary (one language, one
# task's token set, ...) can genuinely drop a large fraction of it. But
# dropping vocabulary row `i` doesn't just shrink a tensor the way dropping
# a MatMul/Conv channel does -- it means token id `i` can never again be
# fed as an `input_ids` value without producing an out-of-bounds `Gather`
# index at runtime. That is a real correctness hazard, not a cosmetic one,
# and it is fundamentally different from every other technique here: no
# other `apply_*` function in this module changes what a valid model input
# looks like. See :func:`apply_embedding_vocab_pruning`'s own docstring for
# how that contract change is surfaced (its own :class:`EmbeddingPruningResult`
# return type, not a bare `ModelProto` the way every other pass returns, and
# an `id_map` the caller must apply to every `input_ids` value going
# forward).
#
# Graph pattern (confirmed via live `onnx.defs.get_schema("Gather", domain="")`
# introspection, not assumed): an exported `torch.nn.Embedding` is
# universally a plain ``Gather(data=embedding_table, indices=input_ids,
# axis=0)`` -- `Gather`'s own schema has exactly two inputs (`data`,
# `indices`) and one optional `axis` attribute (default ``0``, not
# required), matching that pattern exactly with nothing exporter-specific
# about it. `lm_head` (the output projection producing per-token vocab
# logits) is a plain MatMul/vanilla-Gemm whose *weight tensor* is either:
#
# - **tied** to the embedding table -- literally the same initializer,
#   consumed a second time either directly (a `Gemm` with `transB=1`,
#   whose ``[N, K]`` convention already matches the embedding table's own
#   ``[vocab_size, hidden_size]`` layout with no reshape needed -- the
#   pattern e.g. GPT-2's ONNX export uses) or through one interposed
#   `Transpose` (`perm=[1, 0]`) feeding a plain `MatMul`/`Gemm` with
#   `transB=0` -- the pattern a 3-D-hidden-state model (`Gemm` has no
#   batch-dim support, so a real `[batch, seq, hidden]` lm_head has to be a
#   `MatMul`, which needs the weight pre-transposed to ``[hidden,
#   vocab]``); or
# - **untied**: a fully independent ``[hidden_size, vocab_size]`` (or
#   ``[vocab_size, hidden_size]``, Gemm ``transB=1``) weight with no
#   relation to the embedding table at all.
#
# Both are handled, and handled *distinctly* -- see
# :func:`_match_tied_lm_head`/:func:`_match_untied_lm_head`'s own
# docstrings for exactly how each is recognized and, in particular, how
# the tied case's shared initializer is sliced exactly once (never twice,
# never two different ways) regardless of which of the two tied sub-shapes
# matched it.
#
# Matching/safety bar, held to the same conservative standard as every
# other pass in this module -- decline outright, model left completely
# untouched, rather than guess, for anything not confidently recognized:
#
# - the `Gather`'s own `axis` must be exactly `0`;
# - its `indices` operand must itself be one of the graph's own declared
#   inputs, or (the one bounded hop allowed, since it is extremely common
#   in real exports -- an integer dtype cast between the tokenizer's own
#   output dtype and whatever `Gather` needs) the output of a `Cast` node
#   whose own input is a declared graph input -- never a computed/constant
#   indices tensor this pass would need to also rewrite;
# - the embedding weight must be a constant float (FLOAT/FLOAT16/BFLOAT16
#   -- :func:`_is_supported_float_dtype`, this module's own established
#   FP16/BFloat16 support) 2-D initializer;
# - the embedding weight must have *exactly* one consumer (the `Gather`
#   alone -- the ordinary untied case) or *exactly* two, where the second
#   is a structurally-confirmed tied `lm_head` consumer
#   (:func:`_match_tied_lm_head`). A second consumer that doesn't resolve
#   to one of the two recognized tied shapes declines the *whole* match --
#   not "prune the embedding and ignore the unexplained second reader",
#   which would silently corrupt whatever that reader actually does;
# - a candidate untied `lm_head` is only ever auto-identified
#   (:func:`_match_untied_lm_head`) when its own weight has exactly one
#   consumer *and* its output is itself a genuine graph output -- the one
#   structural signal (short of the caller naming it explicitly) that
#   reliably distinguishes "the" vocab-logits projection from some other,
#   unrelated MatMul/Gemm that happens to share the same output width by
#   coincidence. More than one such candidate is ambiguity, not evidence --
#   declined the same as zero;
# - a `lm_head` bias is handled in exactly two recognized shapes -- a
#   `Gemm` node's own built-in (constant, ``(vocab_size,)``/``(1,
#   vocab_size)``-shaped) `C` input, or a plain `MatMul` (no built-in bias)
#   whose sole consumer is one `Add` against a constant of that same shape
#   (:func:`_match_lm_head_tail`) -- the common real-export shape for a
#   biased linear layer sitting on a 3-D hidden state, where `Gemm`'s lack
#   of batch-dim support already forces `MatMul` in the first place. Any
#   other bias-looking shape (more than one consumer of the projection's
#   own output, an `Add` operand of the wrong width, ...) declines that
#   `lm_head` match outright, rather than silently leaving an unrecognized
#   bias at the old, now-mismatched width;
# - when more than one qualifying `Gather` pattern exists in the graph and
#   the caller hasn't named which one via `input_name`, the whole call
#   declines -- there is no reliable structural way to tell "the" token
#   embedding apart from e.g. a positional embedding that also happens to
#   read a genuine graph input (`position_ids`) through the identical
#   `Gather(data, indices, axis=0)` shape.
#
# Which vocabulary rows are safe to keep is fundamentally a question this
# pass cannot answer from the graph alone -- calibration data can show a
# token id *was* used in some sample, never that it is safe to drop
# (absence of evidence isn't evidence of absence). So the primary, default-
# recommended entry point, :func:`apply_embedding_vocab_pruning`, requires
# the caller to supply the keep-set explicitly (`keep_token_ids`/
# `drop_token_ids`) -- mirroring how a real restricted-vocabulary
# deployment actually works: the caller (who owns the tokenizer/deployment
# surface) already knows the exact keep-set; this pass's only job is to
# slice every consumer to match it, correctly. A second, explicitly weaker
# entry point, :func:`apply_embedding_vocab_magnitude_pruning`, is also
# offered -- the same "rank by weight magnitude" spirit as
# :func:`apply_magnitude_pruning`'s own unstructured technique -- but its
# own docstring is explicit that a row's L2 norm is a much weaker safety
# signal than a caller-supplied keep-set (a rarely-used but still
# legitimate token can have a small norm) and must not be used without
# validating the result against the deployment's actual expected input
# distribution.
#
# Whichever indices survive, they are renumbered contiguously (kept token
# id `i`'s new id is its rank among the sorted kept ids) -- so both public
# entry points here return an :class:`EmbeddingPruningResult`, not a bare
# `ModelProto` the way every other `apply_*` function does: `result.model`
# is the pruned graph, and `result.id_map` is the old-token-id ->
# new-token-id mapping every caller of the pruned model must now apply to
# its own `input_ids` before feeding them in. See that class's own
# docstring for the full contract.


@dataclass(frozen=True)
class _LMHeadMatch:
    """A recognized vocab-logits projection paired with a matched embedding
    Gather. `node` is the *final* node whose output is the (pre-rename)
    ``[..., vocab_size]`` logits tensor -- the matched MatMul/Gemm itself,
    or, when :func:`_match_lm_head_tail` recognized a `MatMul`-then-
    `Add(bias)` tail, the `Add` (its output, not the raw MatMul's, is the
    tensor whose shape/graph-output status actually matters downstream).
    `tied` is True when the underlying MatMul/Gemm's *weight* is the
    embedding table itself (already sliced once, as part of the embedding
    weight itself -- `weight_name`/`weight_transposed` are then
    meaningless and left `None`); False means `weight_name` is a fully
    independent initializer this match's own caller must slice separately,
    using `weight_transposed` (the same MatMul/Gemm ``[N, K]``-vs-``[K,
    N]`` convention :func:`_match_producer`/:func:`_slice_producer_weight`
    already use). `bias` is the projection's own constant vocab-width bias
    (`Gemm`'s built-in `C`, or a following `Add`'s operand), when present
    and safely matched -- see :func:`_match_lm_head_tail`'s own docstring
    for exactly which shapes qualify. `via_transpose` is the interposed
    `Transpose` node for the tied "`Transpose` then `MatMul`" sub-shape,
    else `None` -- it owns no weight of its own to slice (its output shape
    simply follows the already-resized embedding table it transposes), but
    its own now-stale output shape still needs invalidating after pruning.
    """

    node: onnx.NodeProto
    tied: bool
    weight_name: Optional[str]
    weight_transposed: Optional[bool]
    bias: Optional[str]
    via_transpose: Optional[onnx.NodeProto]


@dataclass(frozen=True)
class _EmbeddingChain:
    gather: onnx.NodeProto
    weight_name: str
    indices_name: str
    vocab_size: int
    hidden_size: int
    lm_head: Optional[_LMHeadMatch]


# Sentinel distinguishing "no bias" (fine, proceed) from "a bias exists but
# this pass doesn't confidently recognize how to keep it in sync" (decline
# the whole lm_head match) -- `None` alone can't carry that distinction.
_LM_HEAD_BIAS_INVALID = object()


def _match_embedding_gather(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_input_names: Set[str],
    node_by_output: Dict[str, onnx.NodeProto],
) -> Optional[Tuple[str, str, str]]:
    """If `node` is a well-formed embedding-table `Gather` (this section's
    own comment above has the full matching criteria), returns
    ``(weight_name, indices_name, underlying_input_name)`` --
    `indices_name` is `node`'s own literal `indices` operand,
    `underlying_input_name` is that same tensor with one `Cast` hop
    unwrapped when present (identical to `indices_name` otherwise); both
    are returned only for :func:`_match_embedding_chain`'s own
    `input_name` disambiguation, never used to mutate anything -- this
    pass never touches the indices tensor itself.
    """
    if node.op_type != "Gather" or len(node.input) != 2:
        return None
    axis = 0
    for attr in node.attribute:
        if attr.name == "axis":
            axis = attr.i
    if axis != 0:
        return None
    w_name, indices_name = node.input[0], node.input[1]
    w_init = initializer_map.get(w_name)
    if (
        w_init is None
        or not _is_supported_float_dtype(w_init.data_type)
        or len(w_init.dims) != 2
    ):
        return None

    underlying = indices_name
    if underlying not in graph_input_names:
        cast_node = node_by_output.get(underlying)
        if (
            cast_node is not None
            and cast_node.op_type == "Cast"
            and len(cast_node.input) == 1
            and cast_node.input[0] in graph_input_names
        ):
            underlying = cast_node.input[0]
        else:
            return None  # not a graph input, nor a Cast of one

    consumers = consumers_of.get(w_name, [])
    if node not in consumers or not (1 <= len(consumers) <= 2):
        return None  # unexpected consumer count -- decline, don't guess

    return w_name, indices_name, underlying


def _match_lm_head_tail(
    node: onnx.NodeProto,
    initializer_map: Dict[str, onnx.TensorProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    vocab_size: int,
) -> Any:
    """Resolves `node` (an already-matched MatMul/Gemm producing
    `vocab_size`-wide output) to its own bias, if any, and the node whose
    *output* is the real, final logits tensor -- returns
    ``(bias_name_or_None, output_node)``, or the `_LM_HEAD_BIAS_INVALID`
    sentinel when a bias exists but not in a shape this pass safely
    handles. Two shapes are recognized: a `Gemm` node's own built-in
    (constant, vocab-width) `C` input needs no extra hop at all
    (`output_node` is `node` itself); a plain `MatMul` (which has no
    built-in bias) followed by exactly one `Add` against a constant
    vocab-width operand, with no other consumer of `node`'s own output, is
    also recognized -- an extremely common real export shape (`nn.Linear`
    with `bias=True` exported as `MatMul` + `Add` rather than `Gemm`, e.g.
    whenever the surrounding hidden state is 3-D and `Gemm`'s lack of
    batch-dim support already forced `MatMul` in the first place) --
    `output_node` is then the `Add`, since *its* output is the tensor a
    graph-output/shape-staleness check must actually look at, not the raw
    `MatMul`'s own. Anything else that still looks bias-shaped (more than
    one consumer of `node`'s output, an `Add` whose other operand isn't a
    constant vocab-width tensor, ...) returns the sentinel -- declined
    rather than silently left stale, since this pass has no broader
    mechanism to keep an unrecognized bias hop in sync with the new
    (smaller) vocab width.
    """

    def _valid_bias(name: str) -> bool:
        b_init = initializer_map.get(name)
        return (
            b_init is not None
            and _is_supported_float_dtype(b_init.data_type)
            and list(b_init.dims) in ([vocab_size], [1, vocab_size])
        )

    if node.op_type == "Gemm" and len(node.input) == 3 and node.input[2]:
        bias_name = node.input[2]
        if not _valid_bias(bias_name):
            return _LM_HEAD_BIAS_INVALID
        return bias_name, node

    out_name = node.output[0]
    out_consumers = consumers_of.get(out_name, [])
    if not out_consumers:
        return None, node  # no bias, nothing downstream -- node is final
    if any(c.op_type == "Add" for c in out_consumers):
        if len(out_consumers) != 1 or len(out_consumers[0].input) != 2:
            return _LM_HEAD_BIAS_INVALID  # ambiguous fan-out -- decline
        add_node = out_consumers[0]
        other = (
            add_node.input[0] if add_node.input[1] == out_name else add_node.input[1]
        )
        if not _valid_bias(other):
            return _LM_HEAD_BIAS_INVALID
        return other, add_node
    return None, node  # output feeds something else entirely -- no bias here


def _match_tied_lm_head(
    weight_name: str,
    vocab_size: int,
    gather_node: onnx.NodeProto,
    consumers_of: Dict[str, List[onnx.NodeProto]],
    initializer_map: Dict[str, onnx.TensorProto],
) -> Optional[_LMHeadMatch]:
    """If the embedding table `weight_name` has a second consumer besides
    `gather_node`, and that second consumer resolves to one of the two
    tied `lm_head` sub-shapes this section's own comment describes
    (direct `Gemm(transB=1)`, or one `Transpose` then a plain
    `MatMul`/`Gemm(transB=0)`), returns the matched :class:`_LMHeadMatch`
    (`tied=True`, `weight_name`/`weight_transposed` left `None` -- the
    shared initializer is already fully accounted for by the embedding
    weight's own slice, nothing more to do for it here). Returns `None`
    both when there is no second consumer at all (an ordinary untied
    embedding -- the caller falls back to :func:`_match_untied_lm_head`)
    and when there *is* one but it doesn't resolve to either recognized
    shape (an unexplained second reader -- the caller must then decline
    the whole chain, not just skip lm_head detection, since slicing the
    shared weight would silently corrupt that unrecognized consumer too).
    """
    others = [c for c in consumers_of.get(weight_name, []) if c is not gather_node]
    if len(others) != 1:
        return None
    other = others[0]

    match = _match_matmul_like(other)
    if match is not None:
        _, w2, weight_transposed = match
        if w2 == weight_name and weight_transposed:
            tail = _match_lm_head_tail(other, initializer_map, consumers_of, vocab_size)
            if tail is _LM_HEAD_BIAS_INVALID:
                return None
            bias, output_node = tail
            return _LMHeadMatch(
                node=output_node,
                tied=True,
                weight_name=None,
                weight_transposed=None,
                bias=bias,
                via_transpose=None,
            )
        return None

    if other.op_type == "Transpose" and list(other.input) == [weight_name]:
        perm = None
        for attr in other.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
        if perm is not None and perm != [1, 0]:
            return None
        t_out = other.output[0]
        t_consumers = consumers_of.get(t_out, [])
        if len(t_consumers) != 1:
            return None
        node2 = t_consumers[0]
        match2 = _match_matmul_like(node2)
        if match2 is None:
            return None
        _, w3, weight_transposed2 = match2
        if w3 != t_out or weight_transposed2:
            return (
                None  # must consume the transposed [hidden, vocab] tensor untransposed
            )
        tail = _match_lm_head_tail(node2, initializer_map, consumers_of, vocab_size)
        if tail is _LM_HEAD_BIAS_INVALID:
            return None
        bias, output_node = tail
        return _LMHeadMatch(
            node=output_node,
            tied=True,
            weight_name=None,
            weight_transposed=None,
            bias=bias,
            via_transpose=other,
        )

    return None


def _match_untied_lm_head(
    graph: onnx.GraphProto,
    embedding_weight_name: str,
    vocab_size: int,
    consumers_of: Dict[str, List[onnx.NodeProto]],
    initializer_map: Dict[str, onnx.TensorProto],
    graph_outputs: Set[str],
) -> Optional[_LMHeadMatch]:
    """Auto-detects a fully independent `lm_head` weight: exactly one
    MatMul/vanilla-Gemm node in the whole graph whose constant 2-D weight
    is distinct from `embedding_weight_name`, has exactly one consumer,
    produces `vocab_size`-wide output, and -- the one structural signal
    that reliably distinguishes "the" vocab-logits projection from some
    unrelated layer of the same output width -- whose *final* output
    (:func:`_match_lm_head_tail`'s own `output_node`, which is the node
    itself or, for a recognized `MatMul`-then-`Add(bias)` tail, the `Add`)
    is itself a genuine graph output. Zero or more than one such candidate
    is declined (`None`) rather than guessed at; only ever called when
    :func:`_match_tied_lm_head` already found no tied lm_head to prefer.
    """
    candidates = []
    for node in graph.node:
        match = _match_matmul_like(node)
        if match is None:
            continue
        _, w_name, weight_transposed = match
        if w_name == embedding_weight_name:
            continue  # handled by the tied path, not here
        w_init = initializer_map.get(w_name)
        if (
            w_init is None
            or not _is_supported_float_dtype(w_init.data_type)
            or len(w_init.dims) != 2
        ):
            continue
        n_channels = w_init.dims[0] if weight_transposed else w_init.dims[1]
        if n_channels != vocab_size or len(consumers_of.get(w_name, [])) != 1:
            continue
        tail = _match_lm_head_tail(node, initializer_map, consumers_of, vocab_size)
        if tail is _LM_HEAD_BIAS_INVALID:
            continue  # not a confident candidate -- skip, don't guess
        bias, output_node = tail
        if output_node.output[0] not in graph_outputs:
            continue
        candidates.append((w_name, weight_transposed, bias, output_node))

    if len(candidates) != 1:
        return None
    w_name, weight_transposed, bias, output_node = candidates[0]
    return _LMHeadMatch(
        node=output_node,
        tied=False,
        weight_name=w_name,
        weight_transposed=weight_transposed,
        bias=bias,
        via_transpose=None,
    )


def _match_embedding_chain(
    graph: onnx.GraphProto, input_name: Optional[str]
) -> Optional[_EmbeddingChain]:
    """Finds the one token-embedding `Gather` this pass should act on
    (plus its tied/untied `lm_head`, if any) -- see this section's own
    comment above for the full matching/safety bar. When `input_name` is
    given, only a `Gather` whose indices resolve to that exact graph
    input is considered, and it is an error (`ValueError` -- a caller
    mistake, not an ambiguous topology) if none does. When `input_name`
    is omitted, exactly one qualifying `Gather` must exist in the whole
    graph -- zero or more than one (with no way to tell which is "the"
    token embedding, e.g. a `position_ids`-driven positional embedding
    matches the identical structural shape) declines the whole call
    (`None`), the model left completely untouched.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    node_by_output = {out: n for n in graph.node for out in n.output}
    graph_outputs = {o.name for o in graph.output}
    graph_input_names = {i.name for i in graph.input}

    matches = []
    for node in graph.node:
        m = _match_embedding_gather(
            node, initializer_map, consumers_of, graph_input_names, node_by_output
        )
        if m is None:
            continue
        w_name, indices_name, underlying = m
        if input_name is not None and input_name not in (indices_name, underlying):
            continue
        matches.append((node, w_name))

    if input_name is not None and not matches:
        raise ValueError(
            f"no embedding Gather found reading graph input {input_name!r}"
        )
    if len(matches) != 1:
        return None  # zero, or ambiguous with no input_name to disambiguate

    gather_node, w_name = matches[0]
    w_init = initializer_map[w_name]
    vocab_size, hidden_size = int(w_init.dims[0]), int(w_init.dims[1])
    indices_name = gather_node.input[1]

    consumers = consumers_of.get(w_name, [])
    if len(consumers) == 2:
        lm_head = _match_tied_lm_head(
            w_name, vocab_size, gather_node, consumers_of, initializer_map
        )
        if lm_head is None:
            return None  # unexplained second consumer -- decline, don't guess
    else:
        lm_head = _match_untied_lm_head(
            graph, w_name, vocab_size, consumers_of, initializer_map, graph_outputs
        )

    return _EmbeddingChain(
        gather=gather_node,
        weight_name=w_name,
        indices_name=indices_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        lm_head=lm_head,
    )


def _drop_value_info(graph: onnx.GraphProto, name: str) -> None:
    kept = [vi for vi in graph.value_info if vi.name != name]
    if len(kept) != len(graph.value_info):
        del graph.value_info[:]
        graph.value_info.extend(kept)


def _update_vocab_output_shape(
    graph: onnx.GraphProto, name: str, old_v: int, new_v: int
) -> bool:
    """If `name` is a declared graph output with a fixed (`dim_value`) last
    shape dimension equal to `old_v`, updates it to `new_v` in place and
    returns True. Every other pass in this module forbids the dimension it
    resizes from ever reaching a graph output at all (see this module's own
    docstring); this pass is the first that can't make that same
    restriction (an `lm_head`'s vocab-logits output routinely *is* the
    model's own output), so unlike every other pass's `stale_value_info`
    handling (which only ever needs to drop an internal `value_info`
    entry -- see :func:`_drop_value_info`), a stale *graph output* shape
    needs to be actively corrected, not just dropped -- an output entry
    can't be removed the way an internal one can. A symbolic (dim_param)
    or altogether absent last dimension is left alone -- nothing there
    could have gone stale.
    """
    for o in graph.output:
        if o.name != name:
            continue
        dims = o.type.tensor_type.shape.dim
        if (
            len(dims) >= 1
            and dims[-1].HasField("dim_value")
            and dims[-1].dim_value == old_v
        ):
            dims[-1].dim_value = new_v
        return True
    return False


def _finalize_embedding_shapes(
    graph: onnx.GraphProto, chain: _EmbeddingChain, new_v: int
) -> None:
    """Invalidates/corrects every downstream shape this pass's slicing
    makes stale. The `Gather` node's own output shape never needs
    touching -- gathering along `axis=0` doesn't change `hidden_size`, the
    only dimension its own output shape carries from the embedding table.
    An `lm_head`'s output (and, for the tied "Transpose then MatMul"
    sub-shape, the `Transpose`'s own output) *does* change width
    (`vocab_size` -> `new_v`) and is handled here.
    """
    if chain.lm_head is None:
        return
    out_name = chain.lm_head.node.output[0]
    if not _update_vocab_output_shape(graph, out_name, chain.vocab_size, new_v):
        _drop_value_info(graph, out_name)
    if chain.lm_head.via_transpose is not None:
        t_out = chain.lm_head.via_transpose.output[0]
        if not _update_vocab_output_shape(graph, t_out, chain.vocab_size, new_v):
            _drop_value_info(graph, t_out)


def _apply_embedding_vocab_prune(
    model: onnx.ModelProto, chain: _EmbeddingChain, keep_ids: List[int]
) -> onnx.ModelProto:
    """Performs the actual slicing for an already-decided (ascending)
    `keep_ids` set: the embedding table's own `vocab_size` axis (axis 0,
    always -- the Gather's own `axis=0` requirement), and, for an untied
    `lm_head`, its own independent weight/bias. A tied `lm_head`'s weight
    needs no separate slicing call at all: it *is* the embedding table
    (the exact same initializer object), already fully accounted for by
    the one slice below.
    """
    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph
    initializer_map = {t.name: t for t in graph.initializer}
    keep = np.asarray(keep_ids, dtype=np.int64)
    new_v = len(keep_ids)

    _slice_axis(initializer_map[chain.weight_name], keep, axis=0)

    if chain.lm_head is not None and not chain.lm_head.tied:
        assert chain.lm_head.weight_name is not None
        assert chain.lm_head.weight_transposed is not None
        _slice_producer_weight(
            initializer_map[chain.lm_head.weight_name],
            chain.lm_head.weight_transposed,
            keep,
            is_conv=False,
        )
    if chain.lm_head is not None and chain.lm_head.bias is not None:
        _slice_last_axis(initializer_map[chain.lm_head.bias], keep)

    _finalize_embedding_shapes(graph, chain, new_v)
    return out


@dataclass(frozen=True)
class EmbeddingPruningResult:
    """Return type of :func:`apply_embedding_vocab_pruning`/
    :func:`apply_embedding_vocab_magnitude_pruning` -- deliberately *not*
    a bare `onnx.ModelProto` the way every other `apply_*` function in
    this module returns, precisely because this pass changes what counts
    as a valid model *input* (see this section's own comment above), and a
    caller silently treating `result` as if it were a plain pruned model
    (e.g. chaining it straight into another `apply_*` call, or feeding
    original-vocabulary `input_ids` straight to `result.model`) is exactly
    the mistake this shape exists to make hard to make by accident.

    :ivar model: the pruned `onnx.ModelProto` -- unchanged from the input
            model when `matched` is False.
    :ivar matched: False when no eligible embedding `Gather` was found (see
            :func:`_match_embedding_chain`'s own docstring for every reason
            that can happen) -- `model` is then a plain, untouched copy of
            the original, and `kept_token_ids`/`id_map` are both `None`.
    :ivar kept_token_ids: the original (pre-pruning) token ids that survive,
            **sorted ascending** -- row `i` of the pruned embedding table
            (and, if `lm_head_pruned`, column `i` of the pruned logits
            output) corresponds to ``kept_token_ids[i]`` in the *original*
            vocabulary. `None` iff `matched` is False.
    :ivar id_map: ``{old_token_id: new_token_id}`` for every kept id --
            exactly ``{tok: i for i, tok in enumerate(kept_token_ids)}``,
            provided directly so a caller doesn't have to reconstruct it.
            **Every caller of `model` must remap its own `input_ids`
            through this mapping before running the model** -- a dropped
            token id (any id not present as a key) can no longer be fed to
            `model` at all; `id_map` has no entry for it. `None` iff
            `matched` is False.
    :ivar lm_head_pruned: whether a tied or untied `lm_head` was also
            resized (its output logits then have `len(kept_token_ids)`
            columns, ordered the same way `kept_token_ids` orders the
            embedding table's rows). When False and `matched` is True, any
            `lm_head` in the model -- untied and not confidently
            auto-identified, or none present -- was left completely
            untouched: still safe to run (it simply still produces
            full-original-vocabulary-width logits, unaffected by the
            embedding-side renumbering, so no output-side remapping is
            needed in that case).
    """

    model: onnx.ModelProto
    matched: bool
    kept_token_ids: Optional[List[int]] = None
    id_map: Optional[Dict[int, int]] = None
    lm_head_pruned: bool = False


def apply_embedding_vocab_pruning(
    model: Union[str, onnx.ModelProto],
    keep_token_ids: Optional[Sequence[int]] = None,
    drop_token_ids: Optional[Sequence[int]] = None,
    input_name: Optional[str] = None,
) -> EmbeddingPruningResult:
    """Shrinks a matched token-embedding `Gather`'s vocabulary axis (and,
    where a tied or confidently-auto-identified untied `lm_head` exists,
    its own vocab-logits projection too) down to a caller-supplied,
    explicit keep-set. This is the primary, default-recommended entry
    point in this section -- see this module's own "Embedding / lm_head
    vocabulary pruning" section comment for why: calibration data can show
    a token id *was* used, never that it is safe to drop in general, so
    (unlike every other technique in this module) there is no defensible
    way to *infer* a safe keep-set from data alone. The caller -- who
    controls the tokenizer/deployment's own restricted vocabulary -- is
    expected to already know it; this function's only job is to slice
    every matched consumer to agree with it, correctly.

    **Contract change, unmissable on purpose**: unlike every other
    `apply_*` function in this module, the pruned model this returns does
    **not** accept the same `input_ids` values the original model did.
    Token id `i` is only ever a valid input going forward if
    ``i in result.id_map`` -- feed `result.id_map[i]` in its place,
    for every input id, before running `result.model`. A dropped id (any
    id not in `keep_token_ids`, or in `drop_token_ids`) can never be fed
    to the pruned model again at all. See :class:`EmbeddingPruningResult`'s
    own docstring for the full return-value contract this enforces.

    :param model: the original onnx ModelProto or file path
    :param keep_token_ids: the exact set of original token ids to keep,
            in any order/with any duplicates (deduplicated internally,
            then sorted ascending to decide the new row/column order --
            see :class:`EmbeddingPruningResult`). Give exactly one of
            `keep_token_ids`/`drop_token_ids`.
    :param drop_token_ids: the exact set of original token ids to drop;
            every other id in ``range(vocab_size)`` is kept. Give exactly
            one of `keep_token_ids`/`drop_token_ids`.
    :param input_name: when the graph has more than one structurally-
            eligible embedding `Gather` (e.g. a positional embedding also
            reading a genuine graph input), names which one to target by
            its `indices` operand's graph input name (e.g. ``"input_ids"``).
            Required in that case -- omitted, an ambiguous graph declines
            the whole call rather than guessing. A name that matches no
            eligible `Gather` raises `ValueError` (a caller mistake, not an
            ambiguous topology).
    :returns: an :class:`EmbeddingPruningResult` -- see its own docstring.
            `matched` is False (model left completely untouched) for any
            topology this pass doesn't confidently recognize: a non-zero
            `Gather` axis, indices that aren't a graph input (or a `Cast`
            of one), a non-constant/wrong-dtype/non-2-D embedding weight,
            an embedding weight with more than the expected one-or-two
            consumers (or a second consumer that isn't one of the two
            recognized tied `lm_head` shapes), or an ambiguous/absent
            match for `input_name` (or no `input_name` given, more than
            one eligible `Gather`).
    """
    if (keep_token_ids is None) == (drop_token_ids is None):
        raise ValueError(
            "give exactly one of keep_token_ids or drop_token_ids, not both/neither"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    chain = _match_embedding_chain(model.graph, input_name)
    if chain is None:
        out = onnx.ModelProto()
        out.CopyFrom(model)
        return EmbeddingPruningResult(model=out, matched=False)

    vocab_size = chain.vocab_size
    if keep_token_ids is not None:
        keep_set = {int(i) for i in keep_token_ids}
    else:
        assert drop_token_ids is not None
        drop_set = {int(i) for i in drop_token_ids}
        bad_drop = sorted(i for i in drop_set if not (0 <= i < vocab_size))
        if bad_drop:
            raise ValueError(
                f"drop_token_ids out of range [0, {vocab_size}): {bad_drop[:5]}"
            )
        keep_set = set(range(vocab_size)) - drop_set
    bad_keep = sorted(i for i in keep_set if not (0 <= i < vocab_size))
    if bad_keep:
        raise ValueError(
            f"keep_token_ids out of range [0, {vocab_size}): {bad_keep[:5]}"
        )
    if not keep_set:
        raise ValueError("keep_token_ids resolves to an empty vocabulary")
    keep_ids = sorted(keep_set)

    out = _apply_embedding_vocab_prune(model, chain, keep_ids)
    id_map = {old: new for new, old in enumerate(keep_ids)}
    return EmbeddingPruningResult(
        model=out,
        matched=True,
        kept_token_ids=keep_ids,
        id_map=id_map,
        lm_head_pruned=chain.lm_head is not None,
    )


def apply_embedding_vocab_magnitude_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    protect_token_ids: Optional[Sequence[int]] = None,
    input_name: Optional[str] = None,
) -> EmbeddingPruningResult:
    """The importance-ranked variant of :func:`apply_embedding_vocab_pruning`:
    drops the lowest-L2-norm ``sparsity`` fraction of vocabulary rows
    (combined, root-sum-square, with the matched untied `lm_head`'s own
    per-row weight norm when one is identified -- the tied case needs no
    such combination, the embedding table's own row norm already *is*
    the tied `lm_head`'s row norm) -- the same magnitude-based spirit as
    this module's own :func:`apply_magnitude_pruning`, and, like that
    function, needs no calibration data at all.

    **This mode's safety bar is meaningfully weaker than
    :func:`apply_embedding_vocab_pruning`'s own explicit keep-set, and
    that must stay in view for anyone using it**: a small embedding-row
    norm means a token was *initialized/trained with small weights*, not
    that it is safe to drop from a real deployment's input space -- a
    rare-but-still-legitimate token (a domain-specific term, a rare
    Unicode codepoint, a special/control token used only in some request
    paths) can easily have a small norm despite being load-bearing for
    those requests. Using this function without validating the *actual*
    resulting model against the deployment's real expected input
    distribution risks silently breaking real inputs the same way any
    other unvalidated pruning would -- except here "breaking" specifically
    means a token id that used to work now raises an out-of-bounds
    `Gather` index at runtime (not present in `result.id_map` at all), not
    merely a small accuracy regression. `protect_token_ids` (below) covers
    the common, cheap case of guaranteeing a known-important set (special
    tokens, an explicit small always-needed set) never gets ranked away by
    norm alone, but does not by itself make norm-based ranking a safe
    substitute for the explicit-keep-set entry point in general.

    Same contract change as :func:`apply_embedding_vocab_pruning` -- see
    its own docstring and :class:`EmbeddingPruningResult`'s.

    :param model: the original onnx ModelProto or file path
    :param sparsity: target fraction of `vocab_size` to drop (floored so at
            least one row, and every `protect_token_ids` row, always
            survives)
    :param protect_token_ids: token ids to always keep regardless of their
            own norm ranking -- folded into the keep-set before ranking
            fills any remaining budget from the rest
    :param input_name: identical to :func:`apply_embedding_vocab_pruning`'s
            own `input_name` -- disambiguates which `Gather` to target when
            more than one structurally-eligible one exists
    :returns: an :class:`EmbeddingPruningResult` -- see
            :func:`apply_embedding_vocab_pruning`'s own docstring for
            exactly which topologies decline (`matched=False`)
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)

    chain = _match_embedding_chain(model.graph, input_name)
    if chain is None:
        out = onnx.ModelProto()
        out.CopyFrom(model)
        return EmbeddingPruningResult(model=out, matched=False)

    vocab_size = chain.vocab_size
    initializer_map = {t.name: t for t in model.graph.initializer}
    emb = _to_f64(initializer_map[chain.weight_name])
    importance = np.sum(np.square(emb), axis=1)
    if (
        chain.lm_head is not None
        and not chain.lm_head.tied
        and chain.lm_head.weight_name is not None
    ):
        lw = _to_f64(initializer_map[chain.lm_head.weight_name])
        lw_nk = (
            lw if chain.lm_head.weight_transposed else lw.T
        )  # -> [vocab_size, hidden]
        importance = importance + np.sum(np.square(lw_nk), axis=1)
    importance = np.sqrt(importance)

    protect = {int(i) for i in (protect_token_ids or ())}
    bad_protect = sorted(i for i in protect if not (0 <= i < vocab_size))
    if bad_protect:
        raise ValueError(
            f"protect_token_ids out of range [0, {vocab_size}): {bad_protect[:5]}"
        )

    keep_count = max(1, min(vocab_size, round(vocab_size * (1.0 - sparsity))))
    keep_count = max(keep_count, len(protect))
    order = np.argsort(-importance)
    keep_set = set(protect)
    for idx in order:
        if len(keep_set) >= keep_count:
            break
        keep_set.add(int(idx))
    keep_ids = sorted(keep_set)

    out = _apply_embedding_vocab_prune(model, chain, keep_ids)
    id_map = {old: new for new, old in enumerate(keep_ids)}
    return EmbeddingPruningResult(
        model=out,
        matched=True,
        kept_token_ids=keep_ids,
        id_map=id_map,
        lm_head_pruned=chain.lm_head is not None,
    )


# --- Transformer block (depth) pruning --------------------------------------
#
# Every technique above prunes *within* a layer -- channels, heads, experts,
# individual weight entries -- never whether a whole residual sub-block
# (an entire ``x = x + SelfAttn(LN(x))`` or ``x = x + MLP(LN(x))``) is worth
# keeping at all. That is a different axis of the literature entirely --
# "depth pruning"/LayerDrop (Fan et al., 2019)/ShortGPT (Men et al., 2024)/
# Sheared-LLaMA (Xia et al., 2023)/ShortenedLLaMA (Kim et al., 2024): drop
# whole transformer sub-blocks wholesale, rather than shrinking every block
# a little. Unlike everything above, this changes the graph's own
# *topology* -- nodes are deleted and edges rewired, not just tensors
# resized in place -- so it is held to an extra bar of caution the rest of
# this module doesn't need: a candidate is identified, safety-checked, and
# *independently* shape-verified before a single node is ever touched, and
# an entire candidate is declined outright the moment any one of its own
# checks doesn't hold -- never partially applied.
#
# The canonical pattern (pre-norm, the standard shape for every current
# transformer family): ``x_out = Add(x_in, F_out)``, where `F_out` is
# `F`'s own final node's output (`F` -- self-attention or an MLP/FFN --
# fed by ``LN(x_in)``, `LN` one of `LayerNormalization` (plain ONNX,
# opset 17+), `RMSNormalization` (plain ONNX, opset 23+), or
# `SimplifiedLayerNormalization` (onnxruntime's own RMSNorm-equivalent --
# confirmed, empirically, against a real onnxruntime CPU session, to only
# actually register under the *default* (empty/``""``) domain despite
# living in onnxruntime's own contrib-op source tree, not under
# ``com.microsoft`` the way `SkipLayerNormalization`/
# `SkipSimplifiedLayerNormalization` do -- see :func:`_is_entry_ln_node`).
# Dropping the block means: every consumer of `x_out` (including a graph
# output) instead reads `x_in` directly, and every node strictly between
# `x_in` and `x_out` -- `LN`, every one of `F`'s own internal nodes, and the
# merge `Add` itself -- is deleted, provided none of them has any consumer
# outside the block. See :func:`_try_resolve_droppable_block` for the exact
# backward graph walk and every one of its own decline conditions.
#
# Two scope decisions, each investigated against this module's own existing
# residual machinery and this environment's real onnxruntime schemas rather
# than assumed, and each worth stating plainly:
#
# 1. The merge point matched here is the unfused shape only: it is *only*
#    a bare `Add` (:func:`_is_eligible_add_merge`, reused unchanged from
#    the residual *channel*-pruning machinery above -- exactly the same
#    "two distinct, non-constant operands" eligibility bar), never a fused
#    `SkipLayerNormalization`/`SkipSimplifiedLayerNormalization` node the
#    way :func:`_match_matmul_residual_merge` accepts it as the merge for
#    channel pruning. That is a deliberate, real narrowing, not an
#    oversight:
#
#    - As the merge, a `SkipLayerNormalization`-family node's own
#      *primary* output is always the *normalized* sum (`LN(x_in +
#      F_out)`), never the raw sum -- so it can never stand in for `x_out`
#      the way a bare `Add`'s output can (there is no tensor this pass
#      could safely repoint every consumer to that is both already
#      present in the graph *and* means "`x_in`, unnormalized" the way a
#      bare `Add`'s own `x_in` operand already, directly, does).
#
#    The *entry* norm, `LN(x_in)`, is recognized in both shapes: a plain,
#    single-input `LayerNormalization`/`RMSNormalization`/
#    `SimplifiedLayerNormalization` node (:func:`_is_entry_ln_node`,
#    `node.input[0] == x_in`), *or* a fused `SkipLayerNormalization`/
#    `SkipSimplifiedLayerNormalization` node (:func:`_is_fused_entry_ln_node`)
#    reached via its own *primary* (normalized) output, whose optional
#    fourth output -- `input_skip_bias_sum`, confirmed (against
#    onnxruntime's real `com.microsoft` schema via
#    `get_all_operator_schema()`, and by direct execution -- see this
#    section's own comment near :func:`_is_fused_entry_ln_node`) to be the
#    raw, pre-normalization `input + skip (+ bias)` sum, bit-exact -- when
#    *present* stands in for `x_in`: it names, as an ordinary graph
#    tensor, exactly the same "previous block's own raw, unnormalized
#    residual value" a bare `Add`'s own `x_in` operand already, directly,
#    is. onnxruntime's transformer optimizer only ever emits this fourth
#    output when something downstream still needs it (confirmed
#    empirically: omitted from the node's own `output` list, it simply
#    isn't computed or exposed at all -- no error, no silent
#    materialization), so the match declines cleanly whenever it's absent,
#    exactly the "decline outright, never guess" bar every other check in
#    this section holds to. Unlike the merge case, this fused node is
#    never added to the block's own `block_nodes` (never deleted) when
#    matched this way: unlike a plain `LN` node (whose own *input* is
#    `x_in`, produced entirely outside the block), this node's own
#    *output* (`input_skip_bias_sum`) *is* `x_in` -- deleting the node that
#    produces the very tensor every rewired `x_out` consumer needs to keep
#    reading would corrupt the graph. It is left in place, computing
#    exactly what it always did; its own primary (normalized) output
#    simply loses its only consumer (`F`, deleted along with the rest of
#    the block) and goes unread, which is harmless -- a later
#    simplification pass is free to notice and remove it, this pass has no
#    need to. The merge itself is untouched by any of this -- still
#    matched, `x_out`-side, only as a bare `Add`; see point above.
#
#    A raw, not-yet-optimizer-fused export (the common case for a model
#    straight out of a training framework's own ONNX exporter) still
#    matches fully -- it has no `SkipLayerNormalization` nodes to begin
#    with, only plain `Add`/`LayerNormalization`-family nodes throughout.
#    A fully optimizer-fused export -- where the residual `Add` immediately
#    preceding block `N`'s own entry `LN` got fused into block `N-1`'s own
#    merge, producing exactly this `SkipLayerNormalization`-family node --
#    is now matched too, provided block `N`'s own *merge* (its `x_out`)
#    remains a bare `Add` (e.g. the last block before a final, non-fused
#    `LayerNormalization`) -- the realistic shape this extension targets.
# 2. Attention and MLP/FFN blocks are matched -- and dropped -- fully
#    independently of each other, never only as a paired "whole
#    transformer layer". Nothing in the graph structure requires pairing:
#    each block is its own self-contained ``x_out = x_in + F(LN(x_in))``
#    unit with its own merge point, and the backward walk below never
#    needs to know or care whether `F` is self-attention, an MLP, or
#    anything else -- it only ever asks "does everything between this
#    `Add` and its own `LN(x_in)` stay wholly inside the block". Forcing
#    pairing would need extra machinery (recognizing which attention
#    block's own `x_out` feeds which MLP block's own `x_in`, and declining
#    otherwise) to enforce a constraint the safety argument itself never
#    needed -- narrower for no safety benefit, so it is not done.
#
# KV-cache-bearing attention blocks need no special-case handling, and are
# never separately detected by name/shape -- they decline themselves for
# free, via the same generic "no block-internal node's own output may be
# read outside the block, or be a graph output" check every candidate is
# already held to (see :func:`_try_resolve_droppable_block`): a real
# `past_key`/`past_value`-consuming attention op invariably exposes its own
# `present_key`/`present_value` as a *second* graph output on the very same
# node that reads `LN(x_in)` -- a block-internal node's own extra output
# read nowhere else in the block, tripping that check outright. No
# `layer_idx`-attribute renumbering or cache-input/output splicing is
# needed because no such block is ever accepted as droppable in the first
# place, not because it is handled -- a case honestly declined, not one
# quietly made to work.
#
# Selection uses this module's own established calibration-probe
# infrastructure (:func:`_add_probe_outputs`, reused unchanged, the same
# way :func:`apply_wanda_pruning`/:func:`apply_moe_whole_expert_pruning`
# already do) to capture each candidate's own `x_in`/`x_out` over real
# calibration data, then ranks by mean cosine similarity between the two --
# the literature-standard depth-pruning signal (ShortGPT's own "Block
# Influence" metric is exactly ``1 - cosine_similarity(x_in, x_out)``
# averaged per token; this uses the same quantity, just not inverted, so
# higher means more redundant): a block whose own output is nearly
# *identical* to its input, over real data the model is meant to run on,
# contributes almost nothing beyond the identity function and is the
# safest to drop first. This is deliberately not a weight-magnitude
# metric -- unlike a single MatMul/Conv layer's own weight row, a whole
# nonlinear sub-network's importance has no meaningful summary in its own
# weights alone, only in what it actually *does* to real activations. See
# :func:`_transformer_block_similarity` for the exact per-token reduction.


_ENTRY_LN_OPS = (
    "LayerNormalization",
    "RMSNormalization",
    "SimplifiedLayerNormalization",
)


def _is_entry_ln_node(node: onnx.NodeProto, x_in: str) -> bool:
    """True if `node` is one of :data:`_ENTRY_LN_OPS` (all three confirmed,
    empirically, to run under the default/empty ONNX domain, not
    ``com.microsoft`` -- see this section's own comment above) applied
    directly to `x_in` -- exactly the `LN(x_in)` boundary
    :func:`_try_resolve_droppable_block`'s own backward walk stops at.
    """
    return (
        node.domain == ""
        and node.op_type in _ENTRY_LN_OPS
        and len(node.input) >= 1
        and node.input[0] == x_in
    )


def _is_fused_entry_ln_node(node: onnx.NodeProto, x_in: str, t: str) -> bool:
    """True if `node` is a ``com.microsoft::SkipLayerNormalization``/
    ``SkipSimplifiedLayerNormalization`` node (:data:`_SKIP_LAYER_NORM_OPS`/
    :data:`_SKIP_LAYER_NORM_DOMAIN`, the same constants
    :func:`_skip_layer_norm_const_names` uses for the unrelated channel-
    pruning residual merge above) reached, during the backward walk, via
    its own *primary* output `t` -- i.e. `t == node.output[0]`, the
    normalized `LN(input + skip (+ bias))` value `F` actually reads -- and
    whose own optional fourth output, `input_skip_bias_sum`, is both
    present (``len(node.output) > 3`` and non-empty -- onnxruntime only
    ever materializes it when something downstream still needs it;
    confirmed empirically, see this section's own comment above) and
    textually equal to `x_in`.

    This is the fused analogue of :func:`_is_entry_ln_node`'s own
    `node.input[0] == x_in` check -- confirmed, via
    `onnxruntime.capi.onnxruntime_pybind11_state.get_all_operator_schema()`
    against both ops' real ``com.microsoft`` schemas, and independently by
    running a real node through a real `onnxruntime.InferenceSession` and
    comparing bit-exact (not merely close) against an independently
    hand-computed `input + skip (+ bias)`, that `input_skip_bias_sum` is
    exactly the raw, pre-normalization sum -- i.e. exactly what a bare
    `Add`'s own `x_in` operand would already, directly, be, standing in
    for it here since the fused node's own first *input* (unlike a plain
    `LN` node's) is not `x_in` itself but a sum of two separate tensors.
    Requiring `t == node.output[0]` specifically (rather than accepting a
    match reached via `mean`/`inv_std_var`, the op's other two optional
    outputs) mirrors :func:`_is_entry_ln_node`'s own single-input
    specificity, and keeps this from ever misfiring on a node reached only
    through those training-only, in-practice-never-wired outputs.
    """
    return (
        node.domain == _SKIP_LAYER_NORM_DOMAIN
        and node.op_type in _SKIP_LAYER_NORM_OPS
        and len(node.output) >= 1
        and node.output[0] == t
        and len(node.output) > 3
        and node.output[3] != ""
        and node.output[3] == x_in
    )


@dataclass(frozen=True)
class _DroppableBlock:
    merge_node: onnx.NodeProto
    x_in: str
    x_out: str
    # Every node strictly between `x_in` and `x_out` -- `LN`, every one of
    # `F`'s own internal nodes, and `merge_node` itself -- already confirmed
    # to have no consumer outside this set (see
    # :func:`_try_resolve_droppable_block`). Deleting exactly this set and
    # rewiring every `x_out` consumer to `x_in` is the whole graph-surgery
    # operation :func:`_apply_transformer_block_pruning` performs.
    block_nodes: Tuple[onnx.NodeProto, ...]


def _try_resolve_droppable_block(
    merge_node: onnx.NodeProto,
    x_in: str,
    other: str,
    node_by_output: Dict[str, onnx.NodeProto],
    consumers_of: Dict[str, List[onnx.NodeProto]],
    graph_outputs: Set[str],
) -> Optional[_DroppableBlock]:
    """Attempts to resolve `merge_node` (an eligible `Add`,
    :func:`_is_eligible_add_merge`) as a droppable block boundary with
    `x_in` as its own identity/entry operand and `other` (`merge_node`'s
    *other* operand) as `F`'s own final output -- `None` if any of the
    following isn't confirmed, exactly the module's own "decline outright,
    never guess" bar:

    - Walking backward from `other` (through every node's own inputs, in
      the ordinary graph-dependency sense -- no special-cased op set,
      unlike every chain walk elsewhere in this module, since nothing here
      needs to recognize any *particular* shape of `F`, only that
      everything inside it stays inside it) must reach `x_in` *only* via
      one or more entry-norm boundaries, not recursed past: either a plain
      `LN(x_in)` node (:func:`_is_entry_ln_node`), or a fused
      `SkipLayerNormalization`/`SkipSimplifiedLayerNormalization` node
      reached via its own primary output whose optional fourth output
      equals `x_in` (:func:`_is_fused_entry_ln_node` -- see this section's
      own comment above for why the fourth output is the correct `x_in`
      stand-in). Reaching `x_in` directly, bypassing every such boundary,
      means `F` reads `x_in` raw rather than only its own norm, not the
      pattern this pass targets, and declines the whole candidate; finding
      *no* such boundary at all (`F`'s own backward walk never touches
      `x_in`) declines it the same way -- some other tensor entirely feeds
      the merge's `other` operand, not `F(LN(x_in))`.
    - Every node the walk collects (the block's own interior, plus
      `merge_node` itself) must have every one of its own output tensors
      read by nothing outside that same set, and never be a graph output
      -- except `merge_node`'s own primary output (`x_out`), which is
      *expected* to have outside consumers: that is exactly what gets
      rewired to `x_in`. A KV-cache-bearing attention op's own
      `present_key`/`present_value` output, or any other tensor the block
      computes and something else in the graph still needs, trips this
      and declines the block -- see this section's own comment above for
      why this one general check is also exactly what makes a
      KV-cache-bearing block decline itself, with no dedicated detection.
      A fused entry-norm node matched via :func:`_is_fused_entry_ln_node`
      is deliberately *not* added to this set -- see that function's own
      docstring and this section's own comment for why it is preserved
      (never deleted) rather than collected as ordinary block interior.

    A graph input reached mid-walk (an attention mask, position ids, a
    KV-cache input, or any other tensor `F` reads that isn't produced by
    any node) is not itself a problem -- it simply terminates that branch
    of the walk, contributing nothing to the block's own interior node set
    (there is no producer node to collect). Only an actual node whose own
    output leaks outside the block is a decline condition.
    """
    block_node_ids: Set[int] = {id(merge_node)}
    block_nodes: List[onnx.NodeProto] = [merge_node]
    # Fused entry-norm nodes matched via `_is_fused_entry_ln_node`: kept
    # out of `block_node_ids`/`block_nodes` (never deleted -- their own
    # fourth output *is* `x_in`; see this function's own docstring and
    # this section's own comment). Tracked separately so a *second* visit
    # to the same node, via one of its other outputs (`mean`/
    # `inv_std_var`, in practice never wired to anything -- see this
    # section's own comment), can neither re-add it to `block_nodes` nor
    # be silently accepted as some other, contradictory role.
    fused_boundary_ids: Set[int] = set()
    found_entry_ln = False
    visited: Set[str] = set()
    frontier: List[str] = [other]
    while frontier:
        t = frontier.pop()
        if t in visited:
            continue
        visited.add(t)
        if t == x_in:
            return None  # F reads x_in raw, bypassing every LN boundary
        node = node_by_output.get(t)
        if node is None:
            continue  # graph input or initializer -- a leaf, not a problem
        if id(node) in fused_boundary_ids:
            continue  # already resolved as this block's own fused boundary
        if _is_entry_ln_node(node, x_in):
            found_entry_ln = True
            if id(node) not in block_node_ids:
                block_node_ids.add(id(node))
                block_nodes.append(node)
            continue  # boundary -- LN's own input (x_in) is not recursed into
        if _is_fused_entry_ln_node(node, x_in, t):
            if id(node) in block_node_ids:
                # Already collected as ordinary block interior via one of
                # this same node's *other* outputs, before this visit
                # discovered it's actually this block's own fused
                # boundary -- irreconcilable (its own `x_in`-bearing
                # output can't be both deleted and preserved), so decline
                # rather than guess which role is right.
                return None
            found_entry_ln = True
            fused_boundary_ids.add(id(node))
            continue  # boundary -- preserved, not deleted, not recursed into
        if id(node) not in block_node_ids:
            block_node_ids.add(id(node))
            block_nodes.append(node)
        for inp in node.input:
            if inp:
                frontier.append(inp)
    if not found_entry_ln:
        return None

    x_out = merge_node.output[0]
    for node in block_nodes:
        for out_name in node.output:
            if not out_name:
                continue
            if node is merge_node and out_name == x_out:
                continue  # x_out itself -- every consumer gets rewired below
            if out_name in graph_outputs:
                return None
            if any(id(c) not in block_node_ids for c in consumers_of.get(out_name, ())):
                return None

    return _DroppableBlock(
        merge_node=merge_node, x_in=x_in, x_out=x_out, block_nodes=tuple(block_nodes)
    )


def _tensor_shape_dims(
    name: str, value_info_by_name: Dict[str, onnx.ValueInfoProto]
) -> Optional[Tuple[Union[int, str], ...]]:
    """The tensor's own fully-known shape -- one entry per dimension, each
    either a concrete `dim_value` or a named `dim_param` -- or `None` if
    the graph's own annotations don't state every dimension (rank not
    statically known, or any one dimension neither a fixed value nor a
    named symbolic one). Unlike :func:`_tensor_rank`'s own deliberately
    inference-free design elsewhere in this module (a low-stakes heuristic
    that only ever *narrows* which `Concat` axis is accepted, never *what*
    gets pruned), :func:`_shapes_match`'s own correctness -- whether
    replacing `x_out` with `x_in` is shape-safe at all -- depends on this
    being right, so its caller runs real `onnx.shape_inference` first (see
    :func:`apply_transformer_block_pruning`) to populate as much of
    `value_info_by_name` as it safely can before this ever gets called,
    rather than relying only on whatever the graph already happened to
    declare.
    """
    vi = value_info_by_name.get(name)
    if vi is None or not vi.type.HasField("tensor_type"):
        return None
    tensor_type = vi.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims: List[Union[int, str]] = []
    for d in tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.HasField("dim_param") and d.dim_param:
            dims.append(d.dim_param)
        else:
            return None
    return tuple(dims)


def _shapes_match(
    a: str, b: str, value_info_by_name: Dict[str, onnx.ValueInfoProto]
) -> bool:
    """True only when both `a` and `b` have a fully-known shape
    (:func:`_tensor_shape_dims`) and the two are identical, dimension for
    dimension. This is what actually guards against `merge_node`'s own
    `Add` having silently broadcast `x_in` up to a wider `x_out` -- the one
    way replacing every `x_out` consumer with `x_in` directly could change
    a downstream shape rather than simply removing a redundant
    computation. An unknown shape on either side declines rather than
    assumes equality, the same bar every other safety check in this
    section holds to.
    """
    dims_a = _tensor_shape_dims(a, value_info_by_name)
    dims_b = _tensor_shape_dims(b, value_info_by_name)
    return dims_a is not None and dims_a == dims_b


def _find_transformer_block_candidates(
    graph: onnx.GraphProto,
    value_info_by_name: Dict[str, onnx.ValueInfoProto],
) -> List[_DroppableBlock]:
    """Every droppable-block candidate in `graph`: one per eligible `Add`
    merge (:func:`_is_eligible_add_merge`) that :func:`_try_resolve_droppable_block`
    confirms for exactly one operand ordering (a DAG can never confirm
    both -- confirming operand `p` as `x_in` requires `q` to be backward-
    reachable from it, and vice versa, which is impossible for both
    directions on an acyclic graph at once, so this never needs a tie-break
    between the two), and whose `x_in`/`x_out` shapes
    :func:`_shapes_match` independently confirms identical.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    node_by_output: Dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out_name in node.output:
            if out_name:
                node_by_output[out_name] = node
    consumers_of = _consumers_of(graph)
    graph_outputs = {o.name for o in graph.output}

    candidates: List[_DroppableBlock] = []
    for node in graph.node:
        if not _is_eligible_add_merge(node, initializer_map):
            continue
        p, q = node.input[0], node.input[1]
        for x_in, other in ((p, q), (q, p)):
            block = _try_resolve_droppable_block(
                node, x_in, other, node_by_output, consumers_of, graph_outputs
            )
            if block is not None:
                if _shapes_match(block.x_in, block.x_out, value_info_by_name):
                    candidates.append(block)
                break
    return candidates


def _transformer_block_similarity(
    out: onnx.ModelProto,
    candidates: Sequence[_DroppableBlock],
    calibration_data: Sequence[Tensors],
    providers: Optional[Sequence[str]],
) -> Dict[int, float]:
    """Mean cosine similarity between each candidate's own `x_in` and
    `x_out`, over every token of every calibration batch -- the ranking
    signal :func:`apply_transformer_block_pruning` drops the *highest*-
    scoring (most redundant -- `x_out` nearly identical to `x_in` already,
    over real data) candidates by. See this section's own comment above
    for why this, not a weight-magnitude metric, is this pass's own
    importance signal.

    Both tensors are probed in one shared `onnxruntime` pass (reusing
    :func:`_add_probe_outputs`), reshaped to `[-1, hidden]` (mirroring
    every other calibration reduction in this module -- e.g.
    :func:`apply_wanda_pruning`'s own `Attention` statistic, or
    `apply_sparsegpt_pruning`'s own `x.reshape(-1, x.shape[-1])`) so this
    works regardless of the model's own batch/sequence-axis layout, cast
    to float64 the same way every calibration statistic in this module
    already is (real activations, not the model's own possibly-narrower
    declared dtype). A token whose own `x_in` or `x_out` norm is (numerically)
    zero is skipped for that token rather than dividing by (near-)zero -- a
    candidate with no valid token across all of `calibration_data` returns
    ``float("-inf")`` (never the most-redundant pick; there is no evidence
    either way, so it is conservatively kept rather than guessed to be
    droppable).
    """
    keyed = list(enumerate(candidates))
    probe_names = sorted({name for _, c in keyed for name in (c.x_in, c.x_out)})
    if not probe_names:
        return {}
    probe_model = _add_probe_outputs(out, probe_names)

    sim_sum: Dict[int, float] = {}
    sim_count: Dict[int, int] = {}
    for batch in calibration_data:
        result = backend.run_model(probe_model, batch, providers=providers)
        for idx, c in keyed:
            x_in = np.asarray(result[c.x_in], dtype=np.float64)
            x_out = np.asarray(result[c.x_out], dtype=np.float64)
            if x_in.shape != x_out.shape or x_in.ndim == 0 or x_in.shape[-1] == 0:
                continue
            a = x_in.reshape(-1, x_in.shape[-1])
            b = x_out.reshape(-1, x_out.shape[-1])
            norm_a = np.linalg.norm(a, axis=-1)
            norm_b = np.linalg.norm(b, axis=-1)
            denom = norm_a * norm_b
            valid = denom > 1e-12
            if not np.any(valid):
                continue
            cos = np.sum(a[valid] * b[valid], axis=-1) / denom[valid]
            sim_sum[idx] = sim_sum.get(idx, 0.0) + float(np.sum(cos))
            sim_count[idx] = sim_count.get(idx, 0) + int(np.sum(valid))

    return {
        idx: (sim_sum[idx] / sim_count[idx]) if sim_count.get(idx) else float("-inf")
        for idx in range(len(candidates))
    }


def _select_droppable_blocks(
    ranked: Sequence[_DroppableBlock], target: int
) -> List[_DroppableBlock]:
    """Greedily selects from `ranked` (already sorted most- to least-
    redundant) in order, skipping any candidate whose own `block_nodes`
    overlaps an already-selected one's (a candidate whose own backward
    walk happened to reach into another candidate's own interior --
    unusual, but not impossible in principle -- can only ever have *one*
    of the two safely dropped), until `target` blocks are selected or
    `ranked` is exhausted. Returns the selected blocks, in `ranked` order.

    Extracted, behavior-preserving, from
    :func:`_apply_transformer_block_pruning`'s own former inline commit
    loop -- :func:`_analyze_transformer_block_pruning` needs the exact
    same selection (which candidates a real call would actually drop,
    respecting the same overlap-skip rule) to report an honest
    `would_drop` count without duplicating this logic, the same factor-
    out-first approach this module's own Wanda-attention/MoE calibration
    helpers already established (see this module's own "Dry-run pruning
    sensitivity analysis" section comment).
    """
    committed: List[_DroppableBlock] = []
    committed_ids: Set[int] = set()
    for block in ranked:
        if len(committed) >= target:
            break
        ids = {id(n) for n in block.block_nodes}
        if ids & committed_ids:
            continue
        committed.append(block)
        committed_ids |= ids
    return committed


def _apply_transformer_block_pruning(
    graph: onnx.GraphProto, ranked: Sequence[_DroppableBlock], target: int
) -> int:
    """Commits :func:`_select_droppable_blocks`'s own selection from
    `ranked` (already sorted most- to least-redundant) -- up to `target`
    blocks, skipping any candidate whose own `block_nodes` overlaps an
    already-committed one's -- then applies every commit at once. Returns
    the number of blocks actually dropped.

    Each commit rewrites every current reference to its own `x_out` --
    every node's own input, in place, plus (via a small inserted
    `Identity` node, so the model's own declared output name never
    changes) any graph output -- to its own `x_in`, *resolved* through
    every earlier commit's own alias first (`resolve`, below): a chain of
    two directly-adjacent committed blocks (the second's own `x_in` being
    the first's own `x_out`) needs this regardless of commit order --
    committing the *upstream* block first leaves the downstream block's
    own recorded `x_in` referring to a name nothing produces anymore (the
    upstream block's own now-deleted merge node's output), so its own
    rewrite target has to be resolved through the upstream commit's own
    alias rather than used as recorded; committing the *downstream* block
    first instead leaves a stale reference (its own inserted `Identity`,
    or its own entry `LN`, if the block itself doesn't survive to
    deletion first) for the upstream commit's own plain graph-wide
    node-input rewrite to catch and correct directly, no resolution
    needed for that direction. `resolve` handles both by construction,
    with no dependency on which order `committed` happens to process
    them in. Every one of a committed block's own `block_nodes` is then
    deleted in one final pass, preserving every surviving node's own
    relative order -- already topologically valid before deletion, and
    deleting nodes (never reordering or inserting anything ahead of what
    it depends on) can't break that.
    """
    committed = _select_droppable_blocks(ranked, target)
    committed_ids: Set[int] = {id(n) for block in committed for n in block.block_nodes}

    alias: Dict[str, str] = {}

    def resolve(name: str) -> str:
        seen: Set[str] = set()
        while name in alias and name not in seen:
            seen.add(name)
            name = alias[name]
        return name

    for block in committed:
        target_name = resolve(block.x_in)
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == block.x_out:
                    node.input[i] = target_name
        for output in graph.output:
            if output.name == block.x_out:
                graph.node.append(
                    onnx.helper.make_node(
                        "Identity",
                        [target_name],
                        [block.x_out],
                        name=f"{block.x_out}/transformer_block_pruning_identity",
                    )
                )
        alias[block.x_out] = target_name

    if committed_ids:
        kept_nodes = [n for n in graph.node if id(n) not in committed_ids]
        del graph.node[:]
        graph.node.extend(kept_nodes)
        removed_names = {
            out_name
            for block in committed
            for n in block.block_nodes
            for out_name in n.output
            if out_name
        }
        kept_value_info = [
            vi for vi in graph.value_info if vi.name not in removed_names
        ]
        del graph.value_info[:]
        graph.value_info.extend(kept_value_info)

    return len(committed)


def apply_transformer_block_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.25,
    num_blocks_to_drop: Optional[int] = None,
    providers: Optional[Sequence[str]] = None,
) -> onnx.ModelProto:
    """Depth/block-level pruning: drops whole redundant pre-norm
    transformer residual sub-blocks (``x = x + SelfAttn(LN(x))`` or
    ``x = x + MLP(LN(x))``) wholesale, rather than shrinking every block a
    little the way every other `apply_*` function in this module does --
    see this section's own comment above for the exact pattern matched,
    the two scope decisions this narrows to (the merge is a bare `Add`
    only -- a `SkipLayerNormalization`-family node is never recognized in
    that role; the entry norm accepts either a plain, unfused `LN(x_in)`
    node *or* a fused `SkipLayerNormalization`/
    `SkipSimplifiedLayerNormalization` node, via its own optional fourth
    output, standing in for `x_in` -- so a model already run through
    onnxruntime's own transformer optimizer is matched too, provided each
    block's own merge itself is still a bare `Add`; attention and MLP/FFN
    blocks matched and dropped fully independently, never only as a
    "whole layer" pair) and why a KV-cache-bearing attention block needs
    no dedicated handling to always decline safely on its own.

    Every candidate is found by :func:`_find_transformer_block_candidates`
    and confirmed shape-safe (:func:`_shapes_match`, using real
    `onnx.shape_inference` output, not just whatever `value_info` `model`
    already happened to carry) before ranking ever runs. Candidates are
    ranked by mean cosine similarity between their own `x_in` and `x_out`
    over `calibration_data` (:func:`_transformer_block_similarity`) --
    the literature-standard ("Block Influence"/ShortGPT-style)
    redundancy signal: a block whose output is already nearly identical to
    its input, over data the model is meant to actually run on, changes
    almost nothing and is safest to drop -- and the highest-similarity
    ones are dropped first, up to the target count, skipping (not
    failing) any candidate whose own interior overlaps an
    already-committed one's (see :func:`_apply_transformer_block_pruning`'s
    own docstring for why this can happen and why skipping, not declining
    the whole call, is the right response).

    :param model: the original onnx ModelProto or file path
    :param calibration_data: representative input batches to measure each
            candidate block's own input/output similarity on. Each batch
            is a ``{input_name: np.ndarray}`` dict matching ``model``'s
            graph inputs -- see :func:`onnxsim.generate_random_calibration_data`
            (the default when omitted)
    :param num_samples: random batches to generate when
            ``calibration_data`` is omitted
    :param seed: seed for the random calibration data (ignored if
            ``calibration_data`` is supplied)
    :param sparsity: fraction of *matched candidate* blocks to drop
            (rounded to the nearest whole block), ignored when
            ``num_blocks_to_drop`` is given. Note this is a fraction of
            however many blocks this pass actually matched, not of the
            model's total layer count -- the same "fraction of what was
            actually found eligible" meaning
            :func:`apply_moe_whole_expert_pruning`'s own `sparsity` already
            has for experts.
    :param num_blocks_to_drop: an explicit number of blocks to drop
            instead of a fraction, silently capped at however many
            candidates were actually matched
    :param providers: onnxruntime execution providers to run ``model`` on
            when capturing calibration activations
    :returns: ``model`` with the target number of matched candidate blocks
            -- whichever ones ranked most redundant -- deleted and their
            own consumers rewired to read straight through to their own
            block's own input; unchanged (a byte-for-byte copy) if no
            candidate was matched, ``sparsity``/``num_blocks_to_drop``
            rounds to zero blocks, or ``calibration_data`` never gives any
            candidate a valid (non-degenerate) token to rank on
    """
    if num_blocks_to_drop is not None:
        if num_blocks_to_drop < 0:
            raise ValueError(
                f"num_blocks_to_drop must be >= 0, got {num_blocks_to_drop}"
            )
    elif not (0.0 <= sparsity <= 1.0):
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")

    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    out = onnx.ModelProto()
    out.CopyFrom(model)
    graph = out.graph

    try:
        inferred = onnx.shape_inference.infer_shapes(out, strict_mode=False)
        value_info_by_name = _value_info_by_name(inferred.graph)
    except Exception:
        value_info_by_name = _value_info_by_name(graph)

    candidates = _find_transformer_block_candidates(graph, value_info_by_name)
    if not candidates:
        return out

    if num_blocks_to_drop is not None:
        target = min(num_blocks_to_drop, len(candidates))
    else:
        target = int(round(sparsity * len(candidates)))
    if target <= 0:
        return out

    similarity = _transformer_block_similarity(
        out, candidates, calibration_data, providers
    )
    ranked = [
        c
        for _, c in sorted(
            enumerate(candidates), key=lambda item: similarity[item[0]], reverse=True
        )
    ]
    _apply_transformer_block_pruning(graph, ranked, target)
    return out


# --- Dry-run pruning sensitivity analysis ---------------------------------
#
# Every `apply_*_pruning` function above has to actually commit to one
# `sparsity` (or `n`/`m`) before a caller can see what it would do -- the
# only existing introspection, `weight_sparsity`, measures the *result* of
# a mutation that already happened. `analyze_pruning_sensitivity` answers
# the question those functions can't: given the same arguments a real call
# would take, which layers/chains/heads would actually be touched, how many
# of their channels/heads/entries would be dropped, and how "safe" a cut is
# that -- all without ever mutating `model`.
#
# Design: one public entry point (`analyze_pruning_sensitivity`) dispatched
# by the identity of the `apply_fn` a caller passes in, rather than either a
# single fully-generic function that introspects an arbitrary `apply_fn`
# via some hooking protocol, or a family of separately-named
# `analyze_*_sensitivity` functions. The dispatch itself is the thinnest
# possible layer -- an `is` chain to one dedicated `_analyze_*` per family
# below -- so callers get one name to learn and export, while each family's
# own dedicated implementation reuses that family's own real matching/
# importance helpers directly (`_candidates`, `_wanda_norm_for_candidate`/
# `_wanda_importance`, `_plain_structured_importance`, `_chain_group`,
# `_plain_attention_head_importance`/`_gqa_group_importance`,
# `_moe_importance`/`_moe_expert_weight_importance`, ...) rather than a
# duplicated or reimplemented copy of any of them. The one place real
# duplication would otherwise have crept in -- calibration-activation-
# probing loops needed by both a mutating `apply_*` function and this
# module's own dry-run analyzers -- was factored out into shared helpers
# instead, which the real functions now call too (a pure,
# behavior-preserving extraction each time, not a new code path): one
# implementation of "what does calibration measure" per family, shared by
# both the mutating and the dry-run caller, rather than two. This covers
# every Wanda-calibrated family here: `_wanda_unstructured_calibration_stats`
# (`apply_wanda_pruning`), `_wanda_structured_calibration_stats`
# (`apply_structured_wanda_pruning`), `_wanda_attention_calibration_stats`
# (`apply_attention_head_wanda_pruning`), and
# `_moe_router_gate_calibration_stats` (`apply_moe_whole_expert_pruning`'s
# own router-usage ranking, not Wanda-style but the exact same
# probe-then-reduce shape). `apply_transformer_block_pruning`'s own
# calibration probe (`_transformer_block_similarity`) was already its own
# standalone helper before this change and needed no extraction of its
# own -- but its *selection* logic (which ranked candidates a real call
# actually commits, respecting its own overlap-skip rule) was still
# inline in `_apply_transformer_block_pruning` and duplication-prone the
# same way an un-extracted calibration loop would have been, so it was
# factored out the identical way: `_select_droppable_blocks`, now called
# by both the real mutating function and `_analyze_transformer_block_pruning`.
#
# Supported families -- covering every shape this module's own
# `_apply_chains`/`_apply_attention_chains`/`_apply_moe_chains`/
# `_apply_moe_whole_expert_chains`/`_apply_transformer_block_pruning` cover,
# plus the QDQ-quantized structured family and the magnitude-ranked half of
# the embedding/lm_head family, *except* the deliberately out-of-scope
# cases noted below:
#
# - `apply_magnitude_pruning`/`apply_wanda_pruning` (unstructured, per-
#   weight-entry): every mode both support -- per-layer sparsity, N:M, and
#   `global_sparsity` -- is covered.
# - `apply_structured_pruning`/`apply_structured_wanda_pruning` (channel):
#   every non-`Concat`-merged chain kind `_apply_chains` covers -- a plain
#   MatMul/Gemm or Conv producer/consumer pair, a gated (SwiGLU/GeGLU) pair,
#   and a Conv or MatMul/Gemm residual/merge group, general grouped Conv
#   included on either side -- is covered. Two things this module's own
#   `apply_structured_pruning`/`apply_structured_wanda_pruning` *do* handle
#   are deliberately declined here (a clear `NotImplementedError`, not a
#   silent wrong answer), as a scope decision rather than an oversight:
#     * a model containing any `Concat`-merged skip-connection chain
#       (`_find_matmul_concat_chains`/`_find_conv_concat_chains`) -- that
#       mechanism is a *third*, structurally distinct application (its own
#       per-branch floor, no shared `keep` set at all, see
#       `_apply_concat_chains`'s own docstring) that would need its own
#       separately-verified dry-run mirror, not a natural extension of the
#       `_apply_chains`-mirroring loop below;
#     * `global_sparsity=True` -- its own pooled-threshold mechanism
#       (`_apply_chains_global`) is a second, differently-shaped loop atop
#       the one already mirrored below, and -- unlike the unstructured
#       family's own `global_sparsity`, which this module *does* support,
#       being a much more local, per-entry pooling with no cross-chain
#       topology bookkeeping to replicate -- was judged not to earn its own
#       separately-verified mirror within this change's own scope.
# - `apply_attention_head_pruning`/`apply_attention_head_wanda_pruning`
#   (heads/KV-groups): every chain kind either matches -- plain `Attention`
#   (per-head), `GroupQueryAttention`/plain `ai.onnx::Attention`
#   (per-KV-group, packed-QKV included) -- is covered, both plain and
#   Wanda-calibrated, sharing one `_analyze_attention_chains` loop
#   (mirroring `_apply_attention_chains`'s own shared loop) parametrized by
#   which importance callback to rank with -- exactly the same
#   shared-loop-plus-importance-callback shape `_analyze_chains` already
#   gives the structured family.
# - `apply_moe_expert_channel_pruning`/`apply_moe_whole_expert_pruning`
#   (MoE): both are covered, sharing the same "precomputed importance
#   vector, then top-k" shape every other family here already has --
#   `_moe_importance` for channel pruning (pure weight magnitude, no
#   calibration), and, for whole-expert pruning, the exact same
#   calibration-ranked-by-mean-router-gate-weight-with-weight-norm-fallback
#   importance `apply_moe_whole_expert_pruning` itself uses (via the shared
#   `_moe_router_gate_calibration_stats`/`_moe_expert_weight_importance`
#   pair), not just the fallback alone -- so the dry run's own `margin`
#   reflects the real router-usage ranking whenever calibration data
#   produces one, the same fidelity every other calibrated family here
#   already has.
# - `apply_qmoe_expert_channel_pruning`/`apply_qmoe_whole_expert_pruning`
#   (QMoE, the quantized-weight counterpart of the plain-`MoE` bullet just
#   above): both covered by the exact same shape, retargeted at `QMoE`'s
#   own packed ``uint8`` `fc1`/`fc2` weights -- `_qmoe_channel_importance`/
#   `_qmoe_expert_weight_importance` rank each weight's own *dequantized*
#   row/column/slice (`_qmoe_dequantize`, never the packed bytes
#   themselves), and whole-expert pruning reuses
#   `_moe_router_gate_calibration_stats` unchanged (via `_HasRouterProbs`
#   -- `router_probs` is upstream of and oblivious to either node's own
#   quantization). Expert-channel pruning's own keep-count is additionally
#   floored down to a multiple of ``8 // expert_weight_bits`` before
#   ranking (`_apply_qmoe_channel_chains`'s own pack-alignment requirement
#   -- the real CPU `QMoE` kernel has no way to represent a partial
#   trailing packed byte), mirrored here exactly so `would_drop` never
#   reports a count the real call wouldn't actually produce.
# - `apply_structured_pruning_qdq` (QDQ-quantized channel pruning): the
#   same single-producer/single-consumer/unary-hops-only topology
#   `_find_qdq_chains` matches (requiring at least one side to actually be
#   QDQ-quantized), ranked by `_qdq_channel_importance` on the producer's
#   own *dequantized* weight row -- the exact same helper the real
#   function uses to rank, never for the actual (still-quantized) rewrite.
#   No grouped/depthwise Conv, gated pair, residual merge, or Concat
#   branch group is matched for a QDQ chain by the real function either,
#   so none is reported as a matched unit here (they fall out via
#   `not_eligible` exactly like any other unmatched Conv/MatMul/Gemm node,
#   same as the plain float structured family above).
# - `apply_structured_pruning_matmul_nbits` (``MatMulNBits``-quantized
#   channel pruning): the ``MatMulNBits``-to-``MatMulNBits``-only topology
#   `_find_matmul_nbits_chains` matches, ranked by `_qdq_channel_importance`
#   on the producer's own *dequantized* weight row
#   (`_matmul_nbits_dequantized`, the same helper the real function uses to
#   rank, never for the actual int4-code rewrite). A chain whose keep-set
#   doesn't happen to land on the consumer's own `block_size` boundaries
#   (`_matmul_nbits_block_aligned_keep_blocks` returning `None`) is reported
#   `would_drop=0`/`margin=None` -- a genuinely matched unit the real call
#   still declines to touch, mirroring that outcome exactly rather than
#   folding it into `not_eligible`. No grouped, gated, residual, or
#   Concat-merged topology is matched here either, for the same reason the
#   QDQ family above has none.
# - `apply_embedding_vocab_magnitude_pruning` (embedding/lm_head vocabulary,
#   magnitude-ranked): `_match_embedding_chain`'s own single matched-or-not
#   embedding table, ranked by combined embedding-row/untied-lm_head-row L2
#   norm -- the exact computation the real function performs. Reports at
#   most one :class:`PruningLayerSensitivity` per model (there is only ever
#   one token-embedding `Gather` this pass can act on at all -- see
#   `_analyze_embedding_vocab_magnitude_pruning`'s own docstring).
#   `apply_embedding_vocab_pruning` -- the sibling, explicit-`keep_token_ids`
#   entry point -- deliberately has **no** `_analyze_*` counterpart at all;
#   see this module's own "Embedding / lm_head vocabulary family" dry-run
#   section comment for the full reasoning (in short: it has no data-driven
#   importance ranking of its own to report a `margin` for at all -- the
#   caller already supplies the exact keep-set).
# - `apply_transformer_block_pruning` (whole-transformer-block depth
#   pruning): every candidate `_find_transformer_block_candidates` matches,
#   ranked by the exact same mean-cosine-similarity "Block Influence"
#   signal (`_transformer_block_similarity`) and committed via the exact
#   same greedy overlap-skip selection (`_select_droppable_blocks`, shared
#   with the real mutating function -- see this section's own comment
#   above) the real function uses. Unlike every other family, there is no
#   node that "owns" the set of candidate blocks the way a MoE node owns
#   its own experts, so this reports at most one aggregate
#   :class:`PruningLayerSensitivity` per model (`total` the number of
#   matched candidates) rather than one per matched node -- see
#   `_analyze_transformer_block_pruning`'s own docstring for the full
#   shape and why its own `importance`/`margin` are expressed as *negated*
#   similarity (restoring the "higher importance is kept" convention every
#   other family's own `margin` already assumes).
#
# Not supported at all: `apply_sparsegpt_pruning`. Its own per-column
# sequential/compensated Hessian-update loop (see that function's own
# docstring, and its own `global_sparsity`-decline reasoning) has no single
# upfront "here is the importance vector, here is the cut" moment the way
# every other family here does -- an entry's own effective importance
# depends on every *other* entry processed before it in the same
# column-elimination order, via the running Hessian-inverse update, not on
# a fixed score computed once before pruning starts. A dry run could still
# *run* the full column-elimination loop and report which entries ended up
# zeroed, but the "margin" this report exists to surface would have no
# honest meaning: there is no single per-entry importance score to report a
# gap between, only a sequential process whose own intermediate state has
# no per-entry-comparable analogue -- reporting a margin anyway would be
# noise dressed up as a number, so it's left out rather than faked.
# `apply_embedding_vocab_pruning` is not supported either, for a different
# reason -- see the `apply_embedding_vocab_magnitude_pruning` bullet above
# and this module's own "Embedding / lm_head vocabulary family" dry-run
# section comment.


@dataclass(frozen=True)
class PruningLayerSensitivity:
    """One entry of :class:`PruningSensitivityReport`: what a pruning call
    would do to one matched, independently-ranked unit -- a whole weight
    tensor for unstructured (magnitude/Wanda) pruning, a matched producer/
    consumer chain for structured (channel) pruning, or one attention block
    for head/KV-group pruning -- without the model itself ever being
    touched (see :func:`analyze_pruning_sensitivity`).

    :param label: identifies the unit -- the matched node's own `name`
            (falling back to its first output name, or ``"<unnamed>"`` if
            it has neither), or, for a multi-producer chain (a gated FFN
            pair, a residual/merge group), every producer's own label
            joined by ``" + "``
    :param family: which matcher/topology this unit was matched by:
            ``"matmul"``/``"conv"``/``"attention_qkv"`` for unstructured
            pruning; ``"matmul_plain"``/``"matmul_gated"``/
            ``"matmul_residual"``/``"conv_plain"``/``"conv_residual"`` for
            structured pruning; ``"qdq_conv"``/``"qdq_matmul"`` for
            :func:`apply_structured_pruning_qdq` (split by whether the
            matched producer is a Conv or a MatMul/vanilla-Gemm --
            `_find_qdq_chains` matches both through one unified walk,
            unlike the plain float structured family's five separate
            finders, so this is the one family-string split available for
            it); ``"attention_head"``/
            ``"attention_gqa_group"`` for attention pruning (shared by
            :func:`apply_attention_head_pruning` and its Wanda-calibrated
            counterpart :func:`apply_attention_head_wanda_pruning` alike --
            `family` names the topology a unit was matched by, not the
            calibration method used to rank it, exactly as unstructured/
            structured pruning's own plain-vs-Wanda variants already share
            their family strings above); ``"moe_expert_channel"`` for
            :func:`apply_moe_expert_channel_pruning`; ``"moe_whole_expert"``
            for :func:`apply_moe_whole_expert_pruning`;
            ``"qmoe_expert_channel"``/``"qmoe_whole_expert"`` for their
            quantized-weight ``QMoE`` counterparts,
            :func:`apply_qmoe_expert_channel_pruning`/
            :func:`apply_qmoe_whole_expert_pruning`; ``"matmul_nbits"`` for
            :func:`apply_structured_pruning_matmul_nbits` (a single string,
            unlike the QDQ family's Conv/MatMul split --
            `_find_matmul_nbits_chains` only ever matches
            ``MatMulNBits``-to-``MatMulNBits`` chains, so there is no second
            topology to distinguish);
            ``"embedding_vocab_magnitude"`` for
            :func:`apply_embedding_vocab_magnitude_pruning`;
            ``"transformer_block"`` for
            :func:`apply_transformer_block_pruning` -- see
            :func:`analyze_pruning_sensitivity`'s own docstring for exactly
            which topologies each maps to
    :param total: how many independently-ranked elements this unit owns --
            individual weight entries for unstructured pruning, output
            channels for a structured chain, heads or KV groups for
            attention pruning
    :param would_drop: how many of `total` the real, mutating call (given
            the same arguments) would actually zero/remove. Can be ``0``
            even for a genuinely matched unit: `sparsity` rounding to no
            change for a small `total`, or this unit losing a touched-role
            conflict against another matched unit that claims the same
            underlying weight first -- exactly mirroring the real call's
            own "left completely untouched" outcome for that same case,
            rather than being folded into `not_eligible` (which is reserved
            for topology the matcher never recognized at all)
    :param margin: a scale-free proxy for how "safe" this cut is -- the gap
            between the lowest-kept and highest-dropped importance score,
            expressed as a fraction of this unit's own importance range
            (``importance_max - importance_min``) rather than a raw,
            cross-layer-incomparable number. Closer to ``1`` means a wide,
            unambiguous gap between what's kept and what's dropped; closer
            to ``0`` means the cut lands in a near-tie. Can be *negative*:
            several of this module's own masks (a magnitude/Wanda layer's
            own per-row threshold, or a grouped-Conv chain's per-block
            top-k) pick an independent boundary per row/block rather than
            one whole-unit cutoff, so a globally "kept" entry can
            legitimately score below a globally "dropped" one -- a negative
            margin faithfully reports that heterogeneity rather than
            hiding it behind a single, less accurate, whole-unit boundary.
            ``None`` when there is no boundary to measure at all
            (`would_drop` is ``0``, or -- degenerately -- every element
            would be dropped)
    :param importance_min: this unit's own lowest raw importance score --
            context only; not comparable across units on its own (see
            `margin`, which *is* the cross-unit-comparable number)
    :param importance_max: this unit's own highest raw importance score
    """

    label: str
    family: str
    total: int
    would_drop: int
    margin: Optional[float]
    importance_min: float
    importance_max: float

    @property
    def drop_fraction(self) -> float:
        """`would_drop` / `total`, or ``0.0`` for a (never occurring in
        practice) zero-`total` unit.
        """
        return self.would_drop / self.total if self.total else 0.0


@dataclass(frozen=True)
class PruningSensitivityReport:
    """Dry-run "what would happen" report from
    :func:`analyze_pruning_sensitivity`: every unit the target pruning
    function *would* match and rank (`layers`), plus every node whose
    topology it would decline outright -- not even considered a candidate
    to rank -- because it doesn't match that function's own matching
    criteria (`not_eligible`), so the report's own coverage is honest about
    what wasn't even looked at, not just what would be pruned. Building
    this never mutates the input model (see
    :func:`analyze_pruning_sensitivity`'s own docstring).
    """

    layers: List[PruningLayerSensitivity]
    not_eligible: List[str]


def _node_label(node: onnx.NodeProto) -> str:
    if node.name:
        return node.name
    if node.output:
        return node.output[0]
    return "<unnamed>"


def _normalized_margin(
    importance: np.ndarray, keep_mask: np.ndarray
) -> Optional[float]:
    """The `margin` :class:`PruningLayerSensitivity` documents: the gap
    between the lowest-kept and highest-dropped entry of `importance`
    (flattened, alongside `keep_mask`, so this works identically whether
    `importance`/`keep_mask` are a 1-D per-channel/per-head vector or a 2-D
    per-row unstructured-pruning array), divided by `importance`'s own
    ``max - min`` range so the result is comparable across units of very
    different raw importance scale. ``None`` when `keep_mask` is all-True
    or all-False -- no boundary exists to measure. ``0.0`` (not ``None``)
    for the degenerate case where every entry shares the exact same
    importance (`value_range == 0`): there's no *meaningful* boundary
    there either, but unlike the "nothing to measure" case above, this one
    has a definite, correctly-zero answer -- an exactly-tied cut is exactly
    as risky as it can be.
    """
    importance = importance.reshape(-1)
    keep_mask = keep_mask.reshape(-1)
    kept = importance[keep_mask]
    dropped = importance[~keep_mask]
    if kept.size == 0 or dropped.size == 0:
        return None
    value_range = float(importance.max() - importance.min())
    if value_range <= 0:
        return 0.0
    return float(kept.min() - dropped.max()) / value_range


# --- Unstructured (magnitude/Wanda) family --------------------------------


def _unstructured_not_eligible(graph: onnx.GraphProto, candidates: List) -> List[str]:
    matched_ids = {id(node) for node, *_ in candidates}
    not_eligible = []
    for node in graph.node:
        is_attn = node.domain == _ATTENTION_DOMAIN and node.op_type == "Attention"
        if node.op_type not in ("Conv", "MatMul", "Gemm") and not is_attn:
            continue
        if id(node) in matched_ids:
            continue
        not_eligible.append(f"{node.op_type} '{_node_label(node)}'")
    return not_eligible


def _unstructured_family(node: onnx.NodeProto, is_conv: bool) -> str:
    if is_conv:
        return "conv"
    if node.domain == _ATTENTION_DOMAIN and node.op_type == "Attention":
        return "attention_qkv"
    return "matmul"


def _analyze_unstructured(
    graph: onnx.GraphProto,
    candidates: List,
    compute_importance: Callable[
        [onnx.NodeProto, str, str, bool, bool, onnx.TensorProto, np.ndarray],
        np.ndarray,
    ],
    sparsity: float,
    n: Optional[int],
    m: Optional[int],
    global_sparsity: bool,
) -> PruningSensitivityReport:
    """Shared dry-run body for :func:`_analyze_magnitude_pruning`/
    :func:`_analyze_wanda_pruning`, mirroring
    :func:`apply_magnitude_pruning`/:func:`apply_wanda_pruning`'s own
    per-layer (:func:`_sparsity_mask`/:func:`_nm_mask`) and
    `global_sparsity` (:func:`_apply_global_unstructured_pruning`) masking
    exactly, but reporting each layer's own would-be would_drop/margin
    instead of actually zeroing anything. `compute_importance` takes
    ``(node, x_name, w_name, weight_transposed, is_conv, w_init, w_nk)`` and
    returns an importance array shaped like `w_nk` -- plain ``|w_nk|`` for
    magnitude, :func:`_wanda_importance`'s metric for Wanda.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    layers: List[PruningLayerSensitivity] = []

    if global_sparsity:
        entries = []
        for node, x_name, w_name, weight_transposed, is_conv in candidates:
            w_init = initializer_map[w_name]
            w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
            w_nk = _weight_to_nk(w, weight_transposed, is_conv)
            importance = compute_importance(
                node, x_name, w_name, weight_transposed, is_conv, w_init, w_nk
            )
            entries.append((node, is_conv, w_nk, importance))

        pooled = (
            np.concatenate([importance.reshape(-1) for *_, importance in entries])
            if entries
            else np.zeros(0)
        )
        total_pooled = pooled.size
        keep_count = (
            min(max(round(total_pooled * (1.0 - sparsity)), 0), total_pooled)
            if total_pooled
            else 0
        )
        drop_count = total_pooled - keep_count
        drop_flat = np.zeros(total_pooled, dtype=bool)
        if drop_count > 0:
            order = np.argsort(pooled, kind="stable")
            drop_flat[order[:drop_count]] = True

        offset = 0
        for node, is_conv, w_nk, importance in entries:
            size = w_nk.size
            drop_here = drop_flat[offset : offset + size].reshape(w_nk.shape)
            offset += size
            keep_mask = ~drop_here
            would_drop = int(drop_here.sum())
            margin = _normalized_margin(importance, keep_mask) if would_drop else None
            layers.append(
                PruningLayerSensitivity(
                    label=_node_label(node),
                    family=_unstructured_family(node, is_conv),
                    total=size,
                    would_drop=would_drop,
                    margin=margin,
                    importance_min=float(importance.min()) if size else 0.0,
                    importance_max=float(importance.max()) if size else 0.0,
                )
            )
    else:
        for node, x_name, w_name, weight_transposed, is_conv in candidates:
            w_init = initializer_map[w_name]
            w = onnx.numpy_helper.to_array(w_init).astype(np.float64)
            w_nk = _weight_to_nk(w, weight_transposed, is_conv)
            importance = compute_importance(
                node, x_name, w_name, weight_transposed, is_conv, w_init, w_nk
            )
            mask = (
                _nm_mask(importance, n, m)
                if n is not None and m is not None
                else _sparsity_mask(importance, sparsity)
            )
            would_drop = int((~mask).sum())
            margin = _normalized_margin(importance, mask) if would_drop else None
            layers.append(
                PruningLayerSensitivity(
                    label=_node_label(node),
                    family=_unstructured_family(node, is_conv),
                    total=int(mask.size),
                    would_drop=would_drop,
                    margin=margin,
                    importance_min=float(importance.min()),
                    importance_max=float(importance.max()),
                )
            )

    return PruningSensitivityReport(
        layers=layers, not_eligible=_unstructured_not_eligible(graph, candidates)
    )


def _analyze_magnitude_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    global_sparsity: bool = False,
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_magnitude_pruning` -- same arguments,
    same matching (:func:`_candidates`) and masking
    (:func:`_sparsity_mask`/:func:`_nm_mask`/
    :func:`_apply_global_unstructured_pruning`) logic, reused directly, but
    `model` is never mutated: see :func:`analyze_pruning_sensitivity`.
    """
    _validate_pattern(sparsity, n, m)
    if global_sparsity and n is not None:
        raise ValueError(
            "global_sparsity is not supported together with N:M pruning "
            "(n/m) -- see apply_magnitude_pruning's own docstring"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    graph = model.graph
    candidates = _candidates(graph)

    def _importance(
        node: onnx.NodeProto,
        x_name: str,
        w_name: str,
        weight_transposed: bool,
        is_conv: bool,
        w_init: onnx.TensorProto,
        w_nk: np.ndarray,
    ) -> np.ndarray:
        return np.abs(w_nk)

    return _analyze_unstructured(
        graph, candidates, _importance, sparsity, n, m, global_sparsity
    )


def _analyze_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    n: Optional[int] = None,
    m: Optional[int] = None,
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
    global_sparsity: bool = False,
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_wanda_pruning` -- same arguments, same
    matching (:func:`_candidates`), calibration
    (:func:`_wanda_unstructured_calibration_stats`, the exact same helper
    :func:`apply_wanda_pruning` itself calls), and importance
    (:func:`_wanda_norm_for_candidate`/:func:`_wanda_importance`) logic,
    reused directly, but `model` is never mutated: see
    :func:`analyze_pruning_sensitivity`. `model` is only ever read from
    (:func:`_add_probe_outputs`, called by the calibration helper, returns
    a fresh copy to run calibration on rather than modifying what it's
    given).
    """
    _validate_pattern(sparsity, n, m)
    if global_sparsity and n is not None:
        raise ValueError(
            "global_sparsity is not supported together with N:M pruning "
            "(n/m) -- see apply_wanda_pruning's own docstring"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    graph = model.graph
    candidates = _candidates(graph)
    if not candidates:
        return PruningSensitivityReport(
            layers=[], not_eligible=_unstructured_not_eligible(graph, candidates)
        )

    act_norm, conv_act_norm, attn_act_norm = _wanda_unstructured_calibration_stats(
        model, candidates, calibration_data, providers
    )

    def _importance(
        node: onnx.NodeProto,
        x_name: str,
        w_name: str,
        weight_transposed: bool,
        is_conv: bool,
        w_init: onnx.TensorProto,
        w_nk: np.ndarray,
    ) -> np.ndarray:
        norm = _wanda_norm_for_candidate(
            node,
            x_name,
            w_name,
            is_conv,
            w_init,
            act_norm,
            conv_act_norm,
            attn_act_norm,
        )
        return _wanda_importance(w_nk, norm, epsilon)

    return _analyze_unstructured(
        graph, candidates, _importance, sparsity, n, m, global_sparsity
    )


# --- Structured (channel) family -------------------------------------------


def _producer_label(p: _Producer) -> str:
    return _node_label(p.node)


def _chain_label(chain: _Chain) -> str:
    return " + ".join(_producer_label(p) for p in chain.producers)


def _structured_chain_groups(
    graph: onnx.GraphProto,
) -> List[Tuple[str, List[_Chain]]]:
    """The same five finders, in the same order, that
    :func:`apply_structured_pruning`/:func:`apply_structured_wanda_pruning`
    concatenate into their own single `chains` list -- kept as separate
    `(family_label, chains)` groups here purely so
    :class:`PruningLayerSensitivity`'s own `family` field can say which
    matcher found each chain; :func:`_analyze_chains` still processes them
    in this exact flattened order, so cross-chain touched-role conflicts
    resolve identically to the real call's own first-claim-wins order.
    """
    return [
        ("matmul_plain", _find_chains(graph)),
        ("matmul_gated", _find_gated_chains(graph)),
        ("conv_plain", _find_conv_chains(graph)),
        ("conv_residual", _find_conv_residual_chains(graph)),
        ("matmul_residual", _find_matmul_residual_chains(graph)),
    ]


def _structured_not_eligible(
    graph: onnx.GraphProto, chain_groups: List[Tuple[str, List[_Chain]]]
) -> List[str]:
    matched_ids: Set[int] = set()
    for _, chains in chain_groups:
        for chain in chains:
            matched_ids.update(id(p.node) for p in chain.producers)
            matched_ids.add(id(chain.consumer_node))
            matched_ids.update(id(h.node) for h in chain.conv_pass_through)
            for b in chain.extra_consumers:
                matched_ids.add(id(b.consumer_node))
                matched_ids.update(id(h.node) for h in b.conv_pass_through)
    not_eligible = []
    for node in graph.node:
        if node.op_type not in ("Conv", "MatMul", "Gemm"):
            continue
        if id(node) in matched_ids:
            continue
        not_eligible.append(f"{node.op_type} '{_node_label(node)}'")
    return not_eligible


def _analyze_chains(
    graph: onnx.GraphProto,
    chain_groups: List[Tuple[str, List[_Chain]]],
    sparsity: float,
    compute_importance: Callable[[_Chain, List[np.ndarray]], np.ndarray],
) -> List[PruningLayerSensitivity]:
    """Dry-run mirror of :func:`_apply_chains`: replicates its exact
    per-chain control flow -- degenerate-naming skips (no report row: an
    internally-malformed chain object, not a meaningful "would touch
    nothing" outcome), cross-chain touched-role-conflict skips (reported,
    `would_drop=0` -- the real call leaves these completely untouched too,
    losing to an earlier chain that already claimed the same weight),
    per-:func:`_chain_group` block-wise `keep` selection, everything --
    except the final :func:`_slice_chain_channels` call, which it never
    makes, so `graph` is never mutated. `chain_groups` must be in the exact
    same order :func:`apply_structured_pruning`/
    :func:`apply_structured_wanda_pruning` concatenate their own finders'
    output in (see :func:`_structured_chain_groups`) -- touched-role
    conflicts are resolved by that same first-claim-wins order, so
    preserving it is required for the reported would-drop counts to match
    the real call's own chain-by-chain outcome exactly.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    producer_touched: Set[str] = set()
    consumer_touched: Set[str] = set()
    const_touched: Set[str] = set()
    conv_hop_touched: Set[str] = set()
    layers: List[PruningLayerSensitivity] = []

    for family, chains in chain_groups:
        for chain in chains:
            producer_weights = {p.weight for p in chain.producers}
            if len(producer_weights) != len(chain.producers):
                continue  # degenerate (a gated pair naming the same weight twice)

            branches = (
                _ConsumerBranch(
                    chain_ops=(),
                    consumer_node=chain.consumer_node,
                    consumer_weight=chain.consumer_weight,
                    consumer_weight_transposed=chain.consumer_weight_transposed,
                    consumer_is_conv=chain.consumer_is_conv,
                ),
            ) + chain.extra_consumers
            consumer_weights = {b.consumer_weight for b in branches}
            if len(consumer_weights) != len(branches):
                continue  # degenerate (two branches naming the same weight)

            conv_hop_weights = {h.weight for h in chain.conv_pass_through}
            conv_hop_weights.update(
                h.weight for b in chain.extra_consumers for h in b.conv_pass_through
            )
            n_conv_hops = len(chain.conv_pass_through) + sum(
                len(b.conv_pass_through) for b in chain.extra_consumers
            )
            if len(conv_hop_weights) != n_conv_hops:
                continue  # degenerate (the same depthwise weight named twice)

            consts = {p.bias for p in chain.producers if p.bias is not None}
            consts.update(
                const_name
                for _, const_name in chain.chain_ops
                if const_name is not None
            )
            consts.update(
                const_name
                for b in chain.extra_consumers
                for _, const_name in b.chain_ops
                if const_name is not None
            )

            label = _chain_label(chain)
            n = chain.n_channels

            if (
                (producer_weights & producer_touched)
                or (consumer_weights & consumer_touched)
                or (consts & const_touched)
                or (conv_hop_weights & conv_hop_touched)
            ):
                layers.append(
                    PruningLayerSensitivity(
                        label=label,
                        family=family,
                        total=n,
                        would_drop=0,
                        margin=None,
                        importance_min=0.0,
                        importance_max=0.0,
                    )
                )
                continue  # a shared/tied initializer another chain already claimed

            group = _chain_group(chain)
            if group > 1:
                block = n // group
                per_group_keep = max(1, round(block * (1.0 - sparsity)))
                keep_count = per_group_keep * group
            else:
                keep_count = max(1, n - round(n * sparsity))

            if keep_count >= n:
                layers.append(
                    PruningLayerSensitivity(
                        label=label,
                        family=family,
                        total=n,
                        would_drop=0,
                        margin=None,
                        importance_min=0.0,
                        importance_max=0.0,
                    )
                )
                continue  # rounds down to nothing for this chain -- no-op

            w_arrays_nk = []
            for p in chain.producers:
                w = onnx.numpy_helper.to_array(initializer_map[p.weight]).astype(
                    np.float64
                )
                w_arrays_nk.append(_producer_weight_nk(w, p))
            importance = compute_importance(chain, w_arrays_nk)

            if group > 1:
                keep = np.concatenate(
                    [
                        np.sort(
                            np.argsort(-importance[gi * block : (gi + 1) * block])[
                                :per_group_keep
                            ]
                        )
                        + gi * block
                        for gi in range(group)
                    ]
                )
            else:
                keep = np.sort(np.argsort(-importance)[:keep_count])

            keep_mask = np.zeros(n, dtype=bool)
            keep_mask[keep] = True
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family=family,
                    total=n,
                    would_drop=int(n - keep_count),
                    margin=_normalized_margin(importance, keep_mask),
                    importance_min=float(importance.min()),
                    importance_max=float(importance.max()),
                )
            )

            producer_touched.update(producer_weights)
            consumer_touched.update(consumer_weights)
            const_touched.update(consts)
            conv_hop_touched.update(conv_hop_weights)

    return layers


def _analyze_structured_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
    global_sparsity: bool = False,
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_structured_pruning` -- see
    :func:`analyze_pruning_sensitivity`'s own docstring for exactly which
    of that function's own topologies/modes are (and, for `Concat`-merged
    chains and `global_sparsity`, deliberately are not) covered.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if global_sparsity:
        raise NotImplementedError(
            "analyze_pruning_sensitivity does not support "
            "apply_structured_pruning's own global_sparsity=True mode -- "
            "see analyze_pruning_sensitivity's own docstring for why"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    graph = model.graph

    concat_chains = _find_matmul_concat_chains(graph) + _find_conv_concat_chains(graph)
    if concat_chains:
        raise NotImplementedError(
            "analyze_pruning_sensitivity does not support models containing "
            "Concat-merged skip-connection chains for the structured family "
            "yet -- see analyze_pruning_sensitivity's own docstring for why"
        )

    chain_groups = _structured_chain_groups(graph)
    layers = _analyze_chains(
        graph,
        chain_groups,
        sparsity,
        lambda chain, w_arrays_nk: _plain_structured_importance(
            chain, w_arrays_nk, importance_norm
        ),
    )
    return PruningSensitivityReport(
        layers=layers, not_eligible=_structured_not_eligible(graph, chain_groups)
    )


def _analyze_structured_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
    global_sparsity: bool = False,
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_structured_wanda_pruning` -- same
    scope caveats (`Concat`-merged chains, `global_sparsity`) as
    :func:`_analyze_structured_pruning`, plus the same calibration
    (:func:`_wanda_structured_calibration_stats`, the exact same helper
    :func:`apply_structured_wanda_pruning` itself calls) and importance
    logic, reused directly.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if global_sparsity:
        raise NotImplementedError(
            "analyze_pruning_sensitivity does not support "
            "apply_structured_wanda_pruning's own global_sparsity=True mode "
            "-- see analyze_pruning_sensitivity's own docstring for why"
        )
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    graph = model.graph

    concat_chains = _find_matmul_concat_chains(graph) + _find_conv_concat_chains(graph)
    if concat_chains:
        raise NotImplementedError(
            "analyze_pruning_sensitivity does not support models containing "
            "Concat-merged skip-connection chains for the structured family "
            "yet -- see analyze_pruning_sensitivity's own docstring for why"
        )

    chain_groups = _structured_chain_groups(graph)
    all_chains = [c for _, cs in chain_groups for c in cs]
    if not all_chains:
        return PruningSensitivityReport(
            layers=[], not_eligible=_structured_not_eligible(graph, chain_groups)
        )

    act_norm = _wanda_structured_calibration_stats(
        model, all_chains, [], calibration_data, providers
    )

    def _importance(chain: _Chain, w_arrays_nk: List[np.ndarray]) -> np.ndarray:
        base = _plain_structured_importance(chain, w_arrays_nk, importance_norm)
        norm = act_norm.get(chain.consumer_node.input[0])
        if norm is None or norm.shape[0] != chain.n_channels:
            return base
        return base * np.maximum(norm, epsilon)

    layers = _analyze_chains(graph, chain_groups, sparsity, _importance)
    return PruningSensitivityReport(
        layers=layers, not_eligible=_structured_not_eligible(graph, chain_groups)
    )


# --- QDQ (quantized-weight) structured family -------------------------------


def _qdq_not_eligible(graph: onnx.GraphProto, chains: List[_QDQChain]) -> List[str]:
    matched_ids: Set[int] = set()
    for chain in chains:
        matched_ids.add(id(chain.producer.node))
        matched_ids.add(id(chain.consumer.node))
    not_eligible = []
    for node in graph.node:
        if node.op_type not in ("Conv", "MatMul", "Gemm"):
            continue
        if id(node) in matched_ids:
            continue
        not_eligible.append(f"{node.op_type} '{_node_label(node)}'")
    return not_eligible


def _analyze_structured_pruning_qdq(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_structured_pruning_qdq` -- same
    matching (:func:`_find_qdq_chains`, the single-producer/single-
    consumer/unary-hops-only topology this family's own docstring
    describes, requiring at least one side to actually be QDQ-quantized),
    touched-role bookkeeping (a chain sharing a weight -- on either side
    -- with an earlier one in `_find_qdq_chains`'s own return order is
    reported `would_drop=0`/`margin=None`, exactly the "left completely
    untouched" outcome the real call gives it too, not folded into
    `not_eligible`), keep-count, and importance
    (:func:`_qdq_channel_importance`, ranking the producer's own
    *dequantized* weight row -- :func:`_weight_ref_dequantized`, the exact
    same helper :func:`apply_structured_pruning_qdq` itself calls, never
    for the actual rewrite) logic, reused directly, but `model` is never
    mutated. `family` is ``"qdq_conv"``/``"qdq_matmul"`` depending on
    whether the matched producer is a Conv or a MatMul/vanilla-Gemm --
    :func:`_find_qdq_chains` matches both shapes through one unified
    walk, unlike the plain float structured family's five separate
    finders, so this is the one family-string split available to tell
    the two apart.

    A degenerate chain naming the exact same weight in both the producer
    and consumer role gets no report row at all (mirroring
    :func:`_analyze_chains`'s own identical treatment of a gated pair
    naming the same weight twice) -- not a meaningful "would touch
    nothing" outcome, an internally malformed match.

    A chain whose consumer is a *blockwise* INT4/UINT4 QDQ weight
    additionally mirrors the real function's own block-alignment check
    (:func:`_qdq_block_aligned_keep_blocks`): when the importance-ranked
    `keep` set doesn't align to that weight's own `block_size` boundaries,
    this chain is reported `would_drop=0`/`margin=None` -- the same "left
    completely untouched" outcome a touched-role conflict gets above, since
    the real call declines the whole chain identically in both cases,
    rather than a `not_eligible` entry (the chain WAS matched; it just
    can't be safely cut at this exact `sparsity`).

    No general grouped/depthwise Conv, gated pair, residual/skip-
    connection merge, or Concat-merged branch group is ever matched here
    at all (see :func:`apply_structured_pruning_qdq`'s own docstring) --
    such a node simply never enters `_find_qdq_chains`'s own return value
    in the first place, so it is reported via `not_eligible` exactly like
    any other unmatched Conv/MatMul/Gemm node, with no QDQ-specific label
    of its own distinguishing *why* it was declined (the real function
    itself gives none either -- these topologies are out of scope, not
    individually diagnosed).
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    graph = model.graph

    chains = _find_qdq_chains(graph)
    not_eligible = _qdq_not_eligible(graph, chains)
    if not chains:
        return PruningSensitivityReport(layers=[], not_eligible=not_eligible)

    producer_touched: Set[str] = set()
    consumer_touched: Set[str] = set()
    layers: List[PruningLayerSensitivity] = []

    for chain in chains:
        p, c = chain.producer, chain.consumer
        p_key = _weight_ref_key(p.ref)
        c_key = _weight_ref_key(c.ref)
        if p_key == c_key:
            continue  # degenerate (the same weight in both roles) -- no report row

        label = _node_label(p.node)
        family = "qdq_conv" if p.is_conv else "qdq_matmul"
        n = chain.n_channels

        if p_key in producer_touched or c_key in consumer_touched:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family=family,
                    total=n,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue  # a shared/tied weight another chain already claimed

        keep_count = max(1, n - round(n * sparsity))
        if keep_count >= n:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family=family,
                    total=n,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue  # rounds down to nothing for this chain -- no-op

        w = _weight_ref_dequantized(p.ref)
        w_nk = _weight_to_nk(w, p.weight_transposed, p.is_conv)
        importance = _qdq_channel_importance(w_nk, importance_norm)
        # `kind="stable"` to match `apply_structured_pruning_qdq`'s own
        # tie-break exactly (see that function's own comment on this same
        # line) -- otherwise an exactly-tied `importance` vector could make
        # this dry-run mirror's block-alignment decision diverge from the
        # real call's, platform-dependently.
        keep = np.sort(np.argsort(-importance, kind="stable")[:keep_count])

        if c.ref.qdq_block is not None:
            keep_blocks = _qdq_block_aligned_keep_blocks(
                keep, n, c.ref.qdq_block.block_size
            )
            if keep_blocks is None:
                layers.append(
                    PruningLayerSensitivity(
                        label=label,
                        family=family,
                        total=n,
                        would_drop=0,
                        margin=None,
                        importance_min=0.0,
                        importance_max=0.0,
                    )
                )
                continue  # non-block-aligned keep set -- real call declines
                # this whole chain too, see this function's own docstring

        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[keep] = True

        layers.append(
            PruningLayerSensitivity(
                label=label,
                family=family,
                total=n,
                would_drop=int(n - keep_count),
                margin=_normalized_margin(importance, keep_mask),
                importance_min=float(importance.min()),
                importance_max=float(importance.max()),
            )
        )

        producer_touched.add(p_key)
        consumer_touched.add(c_key)

    return PruningSensitivityReport(layers=layers, not_eligible=not_eligible)


# --- MatMulNBits (block-quantized weight) structured family -----------------


def _matmul_nbits_not_eligible(
    graph: onnx.GraphProto, chains: List[_MatMulNBitsChain]
) -> List[str]:
    matched_ids: Set[int] = set()
    for chain in chains:
        matched_ids.add(id(chain.producer.node))
        matched_ids.add(id(chain.consumer.node))
    not_eligible = []
    for node in graph.node:
        if node.op_type != "MatMulNBits" or node.domain != "com.microsoft":
            continue
        if id(node) in matched_ids:
            continue
        not_eligible.append(f"{node.op_type} '{_node_label(node)}'")
    return not_eligible


def _analyze_structured_pruning_matmul_nbits(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_structured_pruning_matmul_nbits` --
    same matching (:func:`_find_matmul_nbits_chains`, the
    ``MatMulNBits``-to-``MatMulNBits``-only topology this family's own
    section comment describes), touched-role bookkeeping (a chain sharing a
    `B` weight -- on either side -- with an earlier one in
    :func:`_find_matmul_nbits_chains`'s own return order is reported
    `would_drop=0`/`margin=None`, exactly the "left completely untouched"
    outcome the real call gives it too, not folded into `not_eligible`),
    keep-count, and importance (:func:`_qdq_channel_importance`, ranking the
    producer's own *dequantized* weight row -- :func:`_matmul_nbits_dequantized`,
    the exact same helper :func:`apply_structured_pruning_matmul_nbits`
    itself calls, never for the actual int4-code rewrite) logic, reused
    directly, but `model` is never mutated. `family` is always
    ``"matmul_nbits"`` -- unlike the QDQ family's own Conv/MatMul split,
    :func:`_find_matmul_nbits_chains` only ever matches
    ``MatMulNBits``-to-``MatMulNBits`` chains, so there is no second
    topology to distinguish.

    A chain whose keep-set doesn't happen to align to the consumer's own
    `block_size` boundaries (:func:`_matmul_nbits_block_aligned_keep_blocks`
    returning ``None`` -- an individual K-column can't be dropped without
    re-quantizing its whole block, out of scope) is also reported
    `would_drop=0`/`margin=None`, mirroring the real function's own decline
    of that chain entirely (both producer and consumer left completely
    untouched, rather than a partial-block re-quantization or a
    disagreeing keep-set between the two) -- this is a genuinely matched
    unit the real call still declines to touch, so it gets a report row
    here too, unlike `not_eligible` (reserved for topology
    :func:`_find_matmul_nbits_chains` never recognized at all).

    A degenerate chain naming the exact same weight in both the producer
    and consumer role gets no report row at all (mirroring
    :func:`_analyze_structured_pruning_qdq`'s own identical treatment).

    No grouped/depthwise structure, gated pair, residual/skip-connection
    merge, or Concat-merged branch group is ever matched here at all (see
    :func:`apply_structured_pruning_matmul_nbits`'s own docstring) -- such a
    node simply never enters :func:`_find_matmul_nbits_chains`'s own return
    value in the first place, so it is reported via `not_eligible` exactly
    like any other unmatched ``MatMulNBits`` node.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    graph = model.graph

    chains = _find_matmul_nbits_chains(graph)
    not_eligible = _matmul_nbits_not_eligible(graph, chains)
    if not chains:
        return PruningSensitivityReport(layers=[], not_eligible=not_eligible)

    producer_touched: Set[str] = set()
    consumer_touched: Set[str] = set()
    layers: List[PruningLayerSensitivity] = []

    for chain in chains:
        p, c = chain.producer, chain.consumer
        p_key = _matmul_nbits_chain_side_key(p)
        c_key = _matmul_nbits_chain_side_key(c)
        if p_key == c_key:
            continue  # degenerate (the same weight in both roles) -- no report row

        label = _node_label(p.node)
        family = "matmul_nbits"
        n = chain.n_channels

        if p_key in producer_touched or c_key in consumer_touched:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family=family,
                    total=n,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue  # a shared/tied weight another chain already claimed

        keep_count = max(1, n - round(n * sparsity))
        if keep_count >= n:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family=family,
                    total=n,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue  # rounds down to nothing for this chain -- no-op

        w_nk = _matmul_nbits_chain_producer_weight_nk(p)
        importance = _qdq_channel_importance(w_nk, importance_norm)
        # `kind="stable"` to match `apply_structured_pruning_matmul_nbits`'s
        # own tie-break exactly (see that function's own comment on this
        # same line) -- otherwise an exactly-tied `importance` vector could
        # make this dry-run mirror's block-alignment decision diverge from
        # the real call's, platform-dependently.
        keep = np.sort(np.argsort(-importance, kind="stable")[:keep_count])

        # Block-alignment only applies when the CONSUMER is itself a
        # `MatMulNBits` node (see `apply_structured_pruning_matmul_nbits`'s
        # own identical branch): a plain-float consumer (mixed-chain support,
        # onnxsim/onnxsim#969) has no block structure at all, so any keep-set
        # applies directly there -- mirrored here so this dry-run's
        # would_drop/margin never diverges from what the real call decides.
        if isinstance(c, _MatMulNBitsWeight):
            keep_blocks = _matmul_nbits_block_aligned_keep_blocks(
                keep, c.k_blocks, c.block_size
            )
            if keep_blocks is None:
                layers.append(
                    PruningLayerSensitivity(
                        label=label,
                        family=family,
                        total=n,
                        would_drop=0,
                        margin=None,
                        importance_min=0.0,
                        importance_max=0.0,
                    )
                )
                continue  # non-block-aligned keep-set for this consumer --
                # the real call declines this chain entirely too, see this
                # function's own docstring

        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[keep] = True

        layers.append(
            PruningLayerSensitivity(
                label=label,
                family=family,
                total=n,
                would_drop=int(n - keep_count),
                margin=_normalized_margin(importance, keep_mask),
                importance_min=float(importance.min()),
                importance_max=float(importance.max()),
            )
        )

        producer_touched.add(p_key)
        consumer_touched.add(c_key)

    return PruningSensitivityReport(layers=layers, not_eligible=not_eligible)


# --- Attention-head family ---------------------------------------------


def _attention_not_eligible(
    graph: onnx.GraphProto, chains: List[_AttnLikeChain]
) -> List[str]:
    matched_ids = {id(c.node) for c in chains}
    not_eligible = []
    for node in graph.node:
        is_attn_like = (
            node.op_type == "Attention" and node.domain in (_ATTENTION_DOMAIN, "")
        ) or (
            node.op_type == "GroupQueryAttention" and node.domain == _ATTENTION_DOMAIN
        )
        if not is_attn_like or id(node) in matched_ids:
            continue
        not_eligible.append(f"{node.op_type} '{_node_label(node)}'")
    return not_eligible


def _analyze_attention_chains(
    graph: onnx.GraphProto,
    chains: List[_AttnLikeChain],
    sparsity: float,
    compute_importance: Callable[..., np.ndarray],
    compute_group_importance: Callable[..., np.ndarray],
) -> List[PruningLayerSensitivity]:
    """Dry-run mirror of :func:`_apply_attention_chains`'s own per-chain
    loop -- shared body for :func:`_analyze_attention_head_pruning`/
    :func:`_analyze_attention_head_wanda_pruning`, mirroring its touched-role
    bookkeeping and keep-count (``max(1, h - round(h * sparsity))``,
    :func:`_apply_one_plain_attention_chain`'s/:func:`_apply_one_gqa_chain`'s
    own formula) exactly, but never actually slicing anything.
    `compute_importance`/`compute_group_importance` take the exact same
    arguments :func:`_apply_one_plain_attention_chain`'s/
    :func:`_apply_one_gqa_chain`'s own callback parameter does --
    ``(chain, wq, wk, wv, dq, dk, dv)``/``(chain, wq, wk, wv)`` -- so a
    caller can pass :func:`_plain_attention_head_importance`/
    :func:`_gqa_group_importance` (optionally wrapped, as
    :func:`_analyze_attention_head_wanda_pruning` does, to fold in a
    calibrated activation norm) unmodified.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    producer_touched: Set[str] = set()
    consumer_touched: Set[str] = set()
    layers: List[PruningLayerSensitivity] = []

    for chain in chains:
        label = _node_label(chain.node)
        if isinstance(chain, _GQAChain):
            producer_names = {chain.q_weight, chain.k_weight, chain.v_weight}
            h = chain.kv_num_heads
            family = "attention_gqa_group"
        else:
            producer_names = {chain.weight}
            h = chain.num_heads
            family = "attention_head"

        if (
            producer_names & producer_touched
            or chain.consumer_weight in consumer_touched
        ):
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family=family,
                    total=h,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue

        keep_count = max(1, h - round(h * sparsity))
        if keep_count >= h:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family=family,
                    total=h,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue

        if isinstance(chain, _GQAChain):
            d = chain.head_size
            wq_init = initializer_map[chain.q_weight]
            wk_init = initializer_map[chain.k_weight]
            wv_init = initializer_map[chain.v_weight]
            if chain.packed_split_sizes is not None:
                nq_orig = chain.num_heads * d
                nk_orig = chain.kv_num_heads * d
                w_kn = onnx.numpy_helper.to_array(wq_init).astype(np.float64)
                if chain.q_weight_transposed:
                    w_kn = w_kn.T
                wq_kn = w_kn[:, :nq_orig]
                wk_kn = w_kn[:, nq_orig : nq_orig + nk_orig]
                wv_kn = w_kn[:, nq_orig + nk_orig :]
            else:
                wq_kn = onnx.numpy_helper.to_array(wq_init).astype(np.float64)
                wk_kn = onnx.numpy_helper.to_array(wk_init).astype(np.float64)
                wv_kn = onnx.numpy_helper.to_array(wv_init).astype(np.float64)
                if chain.q_weight_transposed:
                    wq_kn = wq_kn.T
                if chain.k_weight_transposed:
                    wk_kn = wk_kn.T
                if chain.v_weight_transposed:
                    wv_kn = wv_kn.T
            importance = compute_group_importance(chain, wq_kn, wk_kn, wv_kn)
        else:
            dq = chain.nq // h
            dk = chain.nk // h
            dv2 = chain.nv // h
            w = onnx.numpy_helper.to_array(initializer_map[chain.weight]).astype(
                np.float64
            )
            wq = w[:, : chain.nq]
            wk = w[:, chain.nq : chain.nq + chain.nk]
            wv = w[:, chain.nq + chain.nk :]
            importance = compute_importance(chain, wq, wk, wv, dq, dk, dv2)

        keep_heads = np.sort(np.argsort(-importance)[:keep_count])
        keep_mask = np.zeros(h, dtype=bool)
        keep_mask[keep_heads] = True
        layers.append(
            PruningLayerSensitivity(
                label=label,
                family=family,
                total=h,
                would_drop=int(h - keep_count),
                margin=_normalized_margin(importance, keep_mask),
                importance_min=float(importance.min()),
                importance_max=float(importance.max()),
            )
        )

        producer_touched.update(producer_names)
        consumer_touched.add(chain.consumer_weight)

    return layers


def _analyze_attention_head_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_attention_head_pruning`: same
    matching (:func:`_find_attention_chains`/:func:`_find_gqa_chains`/
    :func:`_find_onnx_attention_chains`) and per-chain loop
    (:func:`_analyze_attention_chains`, the same shared body
    :func:`_analyze_attention_head_wanda_pruning` uses), with plain
    (:func:`_plain_attention_head_importance`/:func:`_gqa_group_importance`)
    importance, reused directly, but `model` is never mutated.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    graph = model.graph

    chains: List[_AttnLikeChain] = [
        *_find_attention_chains(graph),
        *_find_gqa_chains(graph),
        *_find_onnx_attention_chains(graph),
    ]
    layers = _analyze_attention_chains(
        graph,
        chains,
        sparsity,
        lambda chain, wq, wk, wv, dq, dk, dv: _plain_attention_head_importance(
            chain, wq, wk, wv, dq, dk, dv, importance_norm
        ),
        lambda chain, wq, wk, wv: _gqa_group_importance(
            chain, wq, wk, wv, importance_norm
        ),
    )
    return PruningSensitivityReport(
        layers=layers, not_eligible=_attention_not_eligible(graph, chains)
    )


def _analyze_attention_head_wanda_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    importance_norm: _ImportanceNorm = "l2",
    epsilon: float = 1e-8,
    providers: Optional[Sequence[str]] = None,
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_attention_head_wanda_pruning` -- same
    matching (:func:`_find_attention_chains`/:func:`_find_gqa_chains`/
    :func:`_find_onnx_attention_chains`), calibration
    (:func:`_wanda_attention_calibration_stats`, the exact same helper
    :func:`apply_attention_head_wanda_pruning` itself calls), per-chain loop
    (:func:`_analyze_attention_chains`), and importance
    (:func:`_plain_attention_head_importance`/:func:`_gqa_group_importance`,
    scaled by the calibrated activation norm exactly as
    :func:`apply_attention_head_wanda_pruning`'s own
    `_wanda_attention_head_importance`/`_wanda_gqa_group_importance`
    closures do) logic, reused directly, but `model` is never mutated.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    _validate_importance_norm(importance_norm)
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    graph = model.graph

    chains: List[_AttnLikeChain] = [
        *_find_attention_chains(graph),
        *_find_gqa_chains(graph),
        *_find_onnx_attention_chains(graph),
    ]
    if not chains:
        return PruningSensitivityReport(
            layers=[], not_eligible=_attention_not_eligible(graph, chains)
        )

    act_norm = _wanda_attention_calibration_stats(
        model, chains, calibration_data, providers
    )

    def _wanda_attention_head_importance(chain, wq, wk, wv, dq, dk, dv):
        base = _plain_attention_head_importance(
            chain, wq, wk, wv, dq, dk, dv, importance_norm
        )
        norm = act_norm.get(chain.consumer_node.input[0])
        if norm is None or norm.shape[0] != chain.nv:
            return base  # no matching activation observed -- fall back to plain
        act_head = np.array(
            [
                np.linalg.norm(norm[h * dv : (h + 1) * dv])
                for h in range(chain.num_heads)
            ]
        )
        return base * np.maximum(act_head, epsilon)

    def _wanda_gqa_group_importance(chain, wq, wk, wv):
        base = _gqa_group_importance(chain, wq, wk, wv, importance_norm)
        norm = act_norm.get(chain.consumer_node.input[0])
        dv = chain.v_head_size
        width = chain.num_heads * dv
        if norm is None or norm.shape[0] != width:
            return base  # no matching activation observed -- fall back to plain
        group_size = chain.num_heads // chain.kv_num_heads
        act_group = np.array(
            [
                np.linalg.norm(norm[kv * group_size * dv : (kv + 1) * group_size * dv])
                for kv in range(chain.kv_num_heads)
            ]
        )
        return base * np.maximum(act_group, epsilon)

    layers = _analyze_attention_chains(
        graph,
        chains,
        sparsity,
        _wanda_attention_head_importance,
        _wanda_gqa_group_importance,
    )
    return PruningSensitivityReport(
        layers=layers, not_eligible=_attention_not_eligible(graph, chains)
    )


# --- MoE expert-intermediate-channel family --------------------------------


def _moe_not_eligible(
    graph: onnx.GraphProto, chains: Union[List[_MoEChain], List[_MoEExpertChain]]
) -> List[str]:
    """Shared `not_eligible` for both MoE families
    (:func:`_analyze_moe_expert_channel_pruning`/
    :func:`_analyze_moe_whole_expert_pruning`) -- every unmatched
    ``com.microsoft::MoE`` node, regardless of which of the two matchers
    (:func:`_find_moe_chains`/:func:`_find_moe_whole_expert_chains`)
    declined it; both `_MoEChain` and `_MoEExpertChain` name the same node
    via their own `node` field, so one shared implementation suffices.
    """
    matched_ids = {id(c.node) for c in chains}
    not_eligible = []
    for node in graph.node:
        if node.domain != _MOE_DOMAIN or node.op_type != "MoE":
            continue
        if id(node) in matched_ids:
            continue
        not_eligible.append(f"{node.op_type} '{_node_label(node)}'")
    return not_eligible


def _analyze_moe_chains(
    graph: onnx.GraphProto,
    chains: List[_MoEChain],
    sparsity: float,
    compute_importance: Callable[[_MoEChain, Dict[str, onnx.TensorProto]], np.ndarray],
) -> List[PruningLayerSensitivity]:
    """Dry-run mirror of :func:`_apply_moe_chains`'s own per-node loop --
    the shared body :func:`_analyze_moe_expert_channel_pruning` uses,
    mirroring its touched-role bookkeeping and keep-count
    (``max(1, n - round(n * sparsity))``, :func:`_apply_moe_chains`'s own
    formula) exactly, but never actually slicing `fc1`/`fc2`(`/fc1_experts_bias`).
    """
    initializer_map = {t.name: t for t in graph.initializer}
    touched: Set[str] = set()
    layers: List[PruningLayerSensitivity] = []

    for chain in chains:
        weight_names = {chain.fc1_w, chain.fc2_w}
        if chain.fc1_b is not None:
            weight_names.add(chain.fc1_b)
        label = _node_label(chain.node)

        if weight_names & touched:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family="moe_expert_channel",
                    total=chain.inter_size,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue
        touched |= weight_names

        n = chain.inter_size
        keep_count = max(1, n - round(n * sparsity))
        if keep_count >= n:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family="moe_expert_channel",
                    total=n,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue

        importance = compute_importance(chain, initializer_map)
        keep = np.sort(np.argsort(-importance)[:keep_count])
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[keep] = True
        layers.append(
            PruningLayerSensitivity(
                label=label,
                family="moe_expert_channel",
                total=n,
                would_drop=int(n - keep_count),
                margin=_normalized_margin(importance, keep_mask),
                importance_min=float(importance.min()),
                importance_max=float(importance.max()),
            )
        )

    return layers


def _analyze_moe_expert_channel_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_moe_expert_channel_pruning` -- same
    matching (:func:`_find_moe_chains`), keep-count, and importance
    (:func:`_moe_importance`) logic, reused directly, but `model` is never
    mutated.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    graph = model.graph

    chains = _find_moe_chains(graph)
    layers = _analyze_moe_chains(graph, chains, sparsity, _moe_importance)
    return PruningSensitivityReport(
        layers=layers, not_eligible=_moe_not_eligible(graph, chains)
    )


# --- MoE whole-expert family -------------------------------------------


def _analyze_moe_whole_expert_chains(
    graph: onnx.GraphProto,
    chains: List[_MoEExpertChain],
    sparsity: float,
    compute_importance: Callable[
        [_MoEExpertChain, Dict[str, onnx.TensorProto]], np.ndarray
    ],
) -> List[PruningLayerSensitivity]:
    """Dry-run mirror of :func:`_apply_moe_whole_expert_chains`'s own
    per-node loop -- the shared body
    :func:`_analyze_moe_whole_expert_pruning` uses, mirroring its
    touched-role bookkeeping and `k`-floored keep-count
    (``max(max(1, min(chain.k, n)), n - round(n * sparsity))``,
    :func:`_apply_moe_whole_expert_chains`'s own formula) exactly, but never
    actually slicing `fc1`/`fc2`/the router projection.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    touched: Set[str] = set()
    layers: List[PruningLayerSensitivity] = []

    for chain in chains:
        weight_names = {chain.fc1_w, chain.fc2_w, chain.router_w}
        if chain.fc1_b is not None:
            weight_names.add(chain.fc1_b)
        if chain.router_b is not None:
            weight_names.add(chain.router_b)
        label = _node_label(chain.node)

        if weight_names & touched:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family="moe_whole_expert",
                    total=chain.num_experts,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue
        touched |= weight_names

        n = chain.num_experts
        floor = max(1, min(chain.k, n))
        keep_count = max(floor, n - round(n * sparsity))
        if keep_count >= n:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family="moe_whole_expert",
                    total=n,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue

        importance = compute_importance(chain, initializer_map)
        keep = np.sort(np.argsort(-importance)[:keep_count])
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[keep] = True
        layers.append(
            PruningLayerSensitivity(
                label=label,
                family="moe_whole_expert",
                total=n,
                would_drop=int(n - keep_count),
                margin=_normalized_margin(importance, keep_mask),
                importance_min=float(importance.min()),
                importance_max=float(importance.max()),
            )
        )

    return layers


def _analyze_moe_whole_expert_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    providers: Optional[Sequence[str]] = None,
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_moe_whole_expert_pruning` -- same
    matching (:func:`_find_moe_whole_expert_chains`), calibration
    (:func:`_moe_router_gate_calibration_stats`, the exact same helper
    :func:`apply_moe_whole_expert_pruning` itself calls), `k`-floored
    keep-count, and importance (mean router gate weight, falling back to
    :func:`_moe_expert_weight_importance` when no matching calibration
    activation was observed for a chain's `router_probs` -- exactly
    :func:`apply_moe_whole_expert_pruning`'s own `_importance` closure)
    logic, reused directly, but `model` is never mutated.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    graph = model.graph

    chains = _find_moe_whole_expert_chains(graph)
    if not chains:
        return PruningSensitivityReport(
            layers=[], not_eligible=_moe_not_eligible(graph, chains)
        )

    mean_gate_weight = _moe_router_gate_calibration_stats(
        model, chains, calibration_data, providers
    )

    def _importance(
        chain: _MoEExpertChain, initializer_map: Dict[str, onnx.TensorProto]
    ) -> np.ndarray:
        gate = mean_gate_weight.get(chain.router_probs)
        if gate is None or gate.shape[0] != chain.num_experts:
            return _moe_expert_weight_importance(chain, initializer_map)
        return gate

    layers = _analyze_moe_whole_expert_chains(graph, chains, sparsity, _importance)
    return PruningSensitivityReport(
        layers=layers, not_eligible=_moe_not_eligible(graph, chains)
    )


# --- QMoE (quantized-weight MoE) family --------------------------------


def _qmoe_not_eligible(
    graph: onnx.GraphProto,
    chains: Union[List[_QMoEChannelChain], List[_QMoEExpertChain]],
) -> List[str]:
    """Shared `not_eligible` for both QMoE families
    (:func:`_analyze_qmoe_expert_channel_pruning`/
    :func:`_analyze_qmoe_whole_expert_pruning`) -- mirrors
    :func:`_moe_not_eligible`'s own shape, just for ``QMoE`` nodes instead
    of plain ``MoE`` (both `_QMoEChannelChain` and `_QMoEExpertChain` name
    the matched node via their own `node` field, the same way
    `_MoEChain`/`_MoEExpertChain` do).
    """
    matched_ids = {id(c.node) for c in chains}
    not_eligible = []
    for node in graph.node:
        if node.domain != _MOE_DOMAIN or node.op_type != "QMoE":
            continue
        if id(node) in matched_ids:
            continue
        not_eligible.append(f"{node.op_type} '{_node_label(node)}'")
    return not_eligible


def _analyze_qmoe_channel_chains(
    graph: onnx.GraphProto,
    chains: List[_QMoEChannelChain],
    sparsity: float,
    compute_importance: Callable[
        [_QMoEChannelChain, Dict[str, onnx.TensorProto]], np.ndarray
    ],
) -> List[PruningLayerSensitivity]:
    """Dry-run mirror of :func:`_apply_qmoe_channel_chains`'s own per-node
    loop -- the shared body :func:`_analyze_qmoe_expert_channel_pruning`
    uses, mirroring its touched-role bookkeeping (`fc1_w`/`fc2_w`/
    `fc1_scale`(/`fc1_bias`/`fc1_zp`), :func:`_apply_qmoe_channel_chains`'s
    own set) and pack-rounded keep-count (``max(1, n - round(n *
    sparsity))``, then floored down to the nearest multiple of ``8 //
    bits`` -- :func:`_apply_qmoe_channel_chains`'s own formula, since the
    real CPU `QMoE` kernel derives `inter_size` purely from
    `fc2_experts_weights`' own packed byte count, with no way to represent
    a partial trailing packed byte) exactly, but never actually slicing
    `fc1`/`fc2`(/`scales`/`bias`/`zero_points`).
    """
    initializer_map = {t.name: t for t in graph.initializer}
    touched: Set[str] = set()
    layers: List[PruningLayerSensitivity] = []

    for chain in chains:
        weight_names = {chain.fc1_w, chain.fc2_w, chain.fc1_scale}
        if chain.fc1_bias is not None:
            weight_names.add(chain.fc1_bias)
        if chain.fc1_zp is not None:
            weight_names.add(chain.fc1_zp)
        label = _node_label(chain.node)

        if weight_names & touched:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family="qmoe_expert_channel",
                    total=chain.inter_size,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue
        touched |= weight_names

        n = chain.inter_size
        pack = 8 // chain.bits
        keep_count = max(1, n - round(n * sparsity))
        keep_count = max(pack, (keep_count // pack) * pack)
        if keep_count >= n:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family="qmoe_expert_channel",
                    total=n,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue

        importance = compute_importance(chain, initializer_map)
        keep = np.sort(np.argsort(-importance)[:keep_count])
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[keep] = True
        layers.append(
            PruningLayerSensitivity(
                label=label,
                family="qmoe_expert_channel",
                total=n,
                would_drop=int(n - keep_count),
                margin=_normalized_margin(importance, keep_mask),
                importance_min=float(importance.min()),
                importance_max=float(importance.max()),
            )
        )

    return layers


def _analyze_qmoe_expert_channel_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_qmoe_expert_channel_pruning` -- same
    matching (:func:`_find_qmoe_chains`), pack-rounded keep-count, and
    importance (:func:`_qmoe_channel_importance`, ranking each weight's own
    *dequantized* row/column -- :func:`_qmoe_dequantize`, the exact same
    helper the real function uses to rank, never for the actual packed-byte
    rewrite) logic, reused directly, but `model` is never mutated.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    graph = model.graph

    chains = _find_qmoe_chains(graph)
    layers = _analyze_qmoe_channel_chains(
        graph, chains, sparsity, _qmoe_channel_importance
    )
    return PruningSensitivityReport(
        layers=layers, not_eligible=_qmoe_not_eligible(graph, chains)
    )


def _analyze_qmoe_whole_expert_chains(
    graph: onnx.GraphProto,
    chains: List[_QMoEExpertChain],
    sparsity: float,
    compute_importance: Callable[
        [_QMoEExpertChain, Dict[str, onnx.TensorProto]], np.ndarray
    ],
) -> List[PruningLayerSensitivity]:
    """Dry-run mirror of :func:`_apply_qmoe_whole_expert_chains`'s own
    per-node loop -- the shared body
    :func:`_analyze_qmoe_whole_expert_pruning` uses, mirroring its
    touched-role bookkeeping (every one of `fc1_w`/`fc2_w`/`fc1_scale`/
    `fc2_scale`/`router_w`(/`fc1_bias`/`fc2_bias`/`fc1_zp`/`fc2_zp`/
    `router_b`), :func:`_apply_qmoe_whole_expert_chains`'s own set) and
    `k`-floored keep-count (``max(max(1, min(chain.k, n)), n - round(n *
    sparsity))``, :func:`_apply_qmoe_whole_expert_chains`'s own formula)
    exactly, but never actually slicing `fc1`/`fc2`/the router projection.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    touched: Set[str] = set()
    layers: List[PruningLayerSensitivity] = []

    for chain in chains:
        weight_names = {
            chain.fc1_w,
            chain.fc2_w,
            chain.fc1_scale,
            chain.fc2_scale,
            chain.router_w,
        }
        if chain.fc1_bias is not None:
            weight_names.add(chain.fc1_bias)
        if chain.fc2_bias is not None:
            weight_names.add(chain.fc2_bias)
        if chain.fc1_zp is not None:
            weight_names.add(chain.fc1_zp)
        if chain.fc2_zp is not None:
            weight_names.add(chain.fc2_zp)
        if chain.router_b is not None:
            weight_names.add(chain.router_b)
        label = _node_label(chain.node)

        if weight_names & touched:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family="qmoe_whole_expert",
                    total=chain.num_experts,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue
        touched |= weight_names

        n = chain.num_experts
        floor = max(1, min(chain.k, n))
        keep_count = max(floor, n - round(n * sparsity))
        if keep_count >= n:
            layers.append(
                PruningLayerSensitivity(
                    label=label,
                    family="qmoe_whole_expert",
                    total=n,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            )
            continue

        importance = compute_importance(chain, initializer_map)
        keep = np.sort(np.argsort(-importance)[:keep_count])
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[keep] = True
        layers.append(
            PruningLayerSensitivity(
                label=label,
                family="qmoe_whole_expert",
                total=n,
                would_drop=int(n - keep_count),
                margin=_normalized_margin(importance, keep_mask),
                importance_min=float(importance.min()),
                importance_max=float(importance.max()),
            )
        )

    return layers


def _analyze_qmoe_whole_expert_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.5,
    providers: Optional[Sequence[str]] = None,
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_qmoe_whole_expert_pruning` -- same
    matching (:func:`_find_qmoe_whole_expert_chains`), calibration
    (:func:`_moe_router_gate_calibration_stats`, the exact same helper
    :func:`apply_qmoe_whole_expert_pruning` itself calls -- see
    :class:`_HasRouterProbs`, which both `_QMoEExpertChain` and
    `_MoEExpertChain` satisfy structurally, needing no `QMoE`-specific
    calibration implementation at all), `k`-floored keep-count, and
    importance (mean router gate weight, falling back to
    :func:`_qmoe_expert_weight_importance` when no matching calibration
    activation was observed for a chain's `router_probs` -- exactly
    :func:`apply_qmoe_whole_expert_pruning`'s own `_importance` closure)
    logic, reused directly, but `model` is never mutated.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )
    graph = model.graph

    chains = _find_qmoe_whole_expert_chains(graph)
    if not chains:
        return PruningSensitivityReport(
            layers=[], not_eligible=_qmoe_not_eligible(graph, chains)
        )

    mean_gate_weight = _moe_router_gate_calibration_stats(
        model, chains, calibration_data, providers
    )

    def _importance(
        chain: _QMoEExpertChain, initializer_map: Dict[str, onnx.TensorProto]
    ) -> np.ndarray:
        gate = mean_gate_weight.get(chain.router_probs)
        if gate is None or gate.shape[0] != chain.num_experts:
            return _qmoe_expert_weight_importance(chain, initializer_map)
        return gate

    layers = _analyze_qmoe_whole_expert_chains(graph, chains, sparsity, _importance)
    return PruningSensitivityReport(
        layers=layers, not_eligible=_qmoe_not_eligible(graph, chains)
    )


# --- Embedding / lm_head vocabulary family ----------------------------------
#
# Only :func:`apply_embedding_vocab_magnitude_pruning` gets a `_analyze_*`
# counterpart here -- :func:`apply_embedding_vocab_pruning` deliberately
# does not, and that is a considered scope decision, not an oversight:
# every other family this module's dry-run analysis covers has its own
# *data-driven* importance ranking (weight magnitude, a Wanda-style
# calibrated norm, router-gate usage, ...) that a `sparsity` argument
# turns into a keep/drop boundary -- exactly the boundary a `margin`
# means anything for. `apply_embedding_vocab_pruning` has no such ranking
# at all: its `keep_token_ids`/`drop_token_ids` argument *is* the answer,
# supplied directly by the caller, not derived from anything this pass
# could report a "would keep/would drop, how safe" verdict on. Given an
# explicit keep-set, dry-running it would only ever be able to echo the
# caller's own input straight back (`total=vocab_size`,
# `would_drop=vocab_size - len(keep_token_ids)`, verbatim) with no
# `importance`/`margin` signal behind it at all -- not a "what would this
# call do that I don't already know" report, just a restatement of the
# call's own arguments. That isn't the shape
# :class:`PruningLayerSensitivity` exists to carry (see its own `margin`
# field docstring), so it is left unregistered rather than forced into a
# shape that would carry no real information.


def _embedding_not_eligible(
    graph: onnx.GraphProto,
    chain: Optional[_EmbeddingChain],
    input_name: Optional[str],
) -> List[str]:
    """Every `Gather` node :func:`_match_embedding_gather` (the exact same
    per-node matcher :func:`_match_embedding_chain` itself calls) confirms
    structurally embedding-shaped, but that the whole call still declines
    to touch -- every one of them when `chain` is `None` (whether because
    none exist, more than one does and the call can't disambiguate
    without `input_name`, or the sole match's own second-consumer/
    `lm_head` shape wasn't confidently recognized --
    :func:`_match_embedding_chain` returns `None` for all three), or none
    at all when `chain` is not `None` (:func:`_match_embedding_chain`'s
    own ``len(matches) != 1`` check guarantees the chosen `chain.gather`
    is the *only* structurally-matching node reachable under this same
    `input_name` in that case). Mirrors :func:`_match_embedding_chain`'s
    own `input_name` filter exactly, so a structurally-matching `Gather`
    reading a *different* graph input than a given `input_name` is still
    reported here -- this call would never touch it either.
    """
    initializer_map = {t.name: t for t in graph.initializer}
    consumers_of = _consumers_of(graph)
    node_by_output = {out: n for n in graph.node for out in n.output}
    graph_input_names = {i.name for i in graph.input}

    not_eligible = []
    for node in graph.node:
        m = _match_embedding_gather(
            node, initializer_map, consumers_of, graph_input_names, node_by_output
        )
        if m is None:
            continue
        _w_name, indices_name, underlying = m
        if input_name is not None and input_name not in (indices_name, underlying):
            continue
        if chain is not None and node is chain.gather:
            continue
        not_eligible.append(f"{node.op_type} '{_node_label(node)}'")
    return not_eligible


def _analyze_embedding_vocab_magnitude_pruning(
    model: Union[str, onnx.ModelProto],
    sparsity: float = 0.5,
    protect_token_ids: Optional[Sequence[int]] = None,
    input_name: Optional[str] = None,
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_embedding_vocab_magnitude_pruning` --
    same matching (:func:`_match_embedding_chain`), `protect_token_ids`
    validation, keep-count, and importance (combined embedding-row/
    untied-`lm_head`-row L2 norm -- the exact same computation
    :func:`apply_embedding_vocab_magnitude_pruning` itself performs) logic,
    reused directly, but `model` is never mutated.

    Reports at most a *single* :class:`PruningLayerSensitivity`
    (`family` ``"embedding_vocab_magnitude"``, `label` the matched
    `Gather` node's own label, `total=vocab_size`) -- unlike every
    channel/head/expert family above, :func:`_match_embedding_chain`
    matches at most one embedding table per graph by construction (an
    ambiguous multi-`Gather` graph declines the whole call outright
    rather than guessing which one is "the" token embedding, see its own
    docstring), so there is only ever one independently-ranked unit for
    this family to report at all -- never zero-or-more the way a chain/
    head/expert-matching family's own `_find_*` can return.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    graph = model.graph

    chain = _match_embedding_chain(graph, input_name)
    not_eligible = _embedding_not_eligible(graph, chain, input_name)
    if chain is None:
        return PruningSensitivityReport(layers=[], not_eligible=not_eligible)

    vocab_size = chain.vocab_size
    initializer_map = {t.name: t for t in graph.initializer}
    emb = _to_f64(initializer_map[chain.weight_name])
    importance = np.sum(np.square(emb), axis=1)
    if (
        chain.lm_head is not None
        and not chain.lm_head.tied
        and chain.lm_head.weight_name is not None
    ):
        lw = _to_f64(initializer_map[chain.lm_head.weight_name])
        lw_nk = (
            lw if chain.lm_head.weight_transposed else lw.T
        )  # -> [vocab_size, hidden]
        importance = importance + np.sum(np.square(lw_nk), axis=1)
    importance = np.sqrt(importance)

    protect = {int(i) for i in (protect_token_ids or ())}
    bad_protect = sorted(i for i in protect if not (0 <= i < vocab_size))
    if bad_protect:
        raise ValueError(
            f"protect_token_ids out of range [0, {vocab_size}): {bad_protect[:5]}"
        )

    keep_count = max(1, min(vocab_size, round(vocab_size * (1.0 - sparsity))))
    keep_count = max(keep_count, len(protect))
    order = np.argsort(-importance)
    keep_set = set(protect)
    for idx in order:
        if len(keep_set) >= keep_count:
            break
        keep_set.add(int(idx))

    keep_mask = np.zeros(vocab_size, dtype=bool)
    keep_mask[sorted(keep_set)] = True

    return PruningSensitivityReport(
        layers=[
            PruningLayerSensitivity(
                label=_node_label(chain.gather),
                family="embedding_vocab_magnitude",
                total=vocab_size,
                would_drop=int(vocab_size - len(keep_set)),
                margin=_normalized_margin(importance, keep_mask),
                importance_min=float(importance.min()),
                importance_max=float(importance.max()),
            )
        ],
        not_eligible=not_eligible,
    )


# --- Transformer block (depth) family ---------------------------------------


def _transformer_block_not_eligible(
    graph: onnx.GraphProto, candidates: List[_DroppableBlock]
) -> List[str]:
    initializer_map = {t.name: t for t in graph.initializer}
    matched_ids = {id(c.merge_node) for c in candidates}
    not_eligible = []
    for node in graph.node:
        if not _is_eligible_add_merge(node, initializer_map):
            continue
        if id(node) in matched_ids:
            continue
        not_eligible.append(f"{node.op_type} '{_node_label(node)}'")
    return not_eligible


def _analyze_transformer_block_pruning(
    model: Union[str, onnx.ModelProto],
    calibration_data: Optional[Sequence[Tensors]] = None,
    num_samples: int = 8,
    seed: int = 0,
    sparsity: float = 0.25,
    num_blocks_to_drop: Optional[int] = None,
    providers: Optional[Sequence[str]] = None,
) -> PruningSensitivityReport:
    """Dry-run mirror of :func:`apply_transformer_block_pruning` -- same
    matching (:func:`_find_transformer_block_candidates`, using real
    `onnx.shape_inference` output exactly as the real call does),
    calibration (:func:`_transformer_block_similarity`, the exact same
    helper :func:`apply_transformer_block_pruning` itself calls), ranking,
    and greedy overlap-skip commit selection
    (:func:`_select_droppable_blocks`, likewise the exact same helper the
    real call uses -- extracted from its own former inline loop
    specifically so this analyzer could reuse it verbatim, see that
    function's own docstring) logic, reused directly, but no node is ever
    deleted or rewired.

    Unlike every other family here, a matched candidate is a whole
    residual sub-block with no further per-block internal channel/head/
    entry count to subdivide -- a block is dropped whole or kept whole,
    never partially -- and, unlike the embedding family just above, there
    can be many independent candidates spread anywhere across the graph
    with no common owning node to group them under (a MoE node owns its
    own experts; no node "owns" the set of transformer blocks). The
    natural unit this reports is therefore the *whole matched-candidate
    set*, exactly the same "one precomputed importance vector, then
    top-k/greedy-selected" shape :func:`apply_moe_whole_expert_pruning`'s
    own dry-run mirror already reports as a single row per MoE node, just
    with the whole model standing in for that one node: a single
    :class:`PruningLayerSensitivity` (`family` ``"transformer_block"``,
    `label` every candidate's own merge-`Add` label joined by ``" + "``,
    mirroring :func:`_chain_label`'s own multi-producer convention,
    `total` the number of matched candidates) whenever at least one
    candidate is matched, none at all otherwise.

    The ranking signal is mean cosine similarity between each candidate's
    own `x_in`/`x_out` -- *higher* means more redundant (safer to drop),
    the opposite sense every other family's own importance has (there,
    higher means safer to *keep*). So the `importance` this reports
    (and `margin` is computed from) is the *negated* similarity score,
    restoring the same "higher importance is kept, lower is dropped"
    convention :func:`_normalized_margin` itself assumes -- a monotonic
    negation changes neither which candidates rank above which others nor
    the greedy selection itself (only :func:`_transformer_block_similarity`'s
    own raw, un-negated score, ranked descending, drives that -- see
    :func:`apply_transformer_block_pruning`'s own docstring), only the
    sign convention this report's own `importance_min`/`importance_max`/
    `margin` fields are expressed in.
    """
    if num_blocks_to_drop is not None:
        if num_blocks_to_drop < 0:
            raise ValueError(
                f"num_blocks_to_drop must be >= 0, got {num_blocks_to_drop}"
            )
    elif not (0.0 <= sparsity <= 1.0):
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")

    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    graph = model.graph

    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        value_info_by_name = _value_info_by_name(inferred.graph)
    except Exception:
        value_info_by_name = _value_info_by_name(graph)

    candidates = _find_transformer_block_candidates(graph, value_info_by_name)
    not_eligible = _transformer_block_not_eligible(graph, candidates)
    if not candidates:
        return PruningSensitivityReport(layers=[], not_eligible=not_eligible)

    label = " + ".join(_node_label(c.merge_node) for c in candidates)
    total = len(candidates)

    if num_blocks_to_drop is not None:
        target = min(num_blocks_to_drop, total)
    else:
        target = int(round(sparsity * total))

    if target <= 0:
        return PruningSensitivityReport(
            layers=[
                PruningLayerSensitivity(
                    label=label,
                    family="transformer_block",
                    total=total,
                    would_drop=0,
                    margin=None,
                    importance_min=0.0,
                    importance_max=0.0,
                )
            ],
            not_eligible=not_eligible,
        )

    if calibration_data is None:
        calibration_data = generate_random_calibration_data(
            model, num_samples=num_samples, seed=seed
        )

    similarity = _transformer_block_similarity(
        model, candidates, calibration_data, providers
    )
    ranked_idx = sorted(range(total), key=lambda i: similarity[i], reverse=True)
    ranked = [candidates[i] for i in ranked_idx]
    committed_ids = {id(c.merge_node) for c in _select_droppable_blocks(ranked, target)}

    importance = np.array([-similarity[i] for i in range(total)])
    keep_mask = np.array([id(c.merge_node) not in committed_ids for c in candidates])

    return PruningSensitivityReport(
        layers=[
            PruningLayerSensitivity(
                label=label,
                family="transformer_block",
                total=total,
                would_drop=int((~keep_mask).sum()),
                margin=_normalized_margin(importance, keep_mask),
                importance_min=float(importance.min()),
                importance_max=float(importance.max()),
            )
        ],
        not_eligible=not_eligible,
    )


# --- Public entry point -----------------------------------------------------

# The exact set of `apply_*` functions `analyze_pruning_sensitivity` can
# dispatch to, and each one's own dedicated dry-run implementation -- see
# this module's own "Dry-run pruning sensitivity analysis" section comment
# for why dispatch-by-identity to a dedicated `_analyze_*` per family, not a
# single generic function that introspects an arbitrary `apply_fn`, or a
# family of separately-named public functions.
_SENSITIVITY_ANALYZERS: Dict[
    Callable[..., Union[onnx.ModelProto, EmbeddingPruningResult]], Callable[..., Any]
] = {
    apply_magnitude_pruning: _analyze_magnitude_pruning,
    apply_wanda_pruning: _analyze_wanda_pruning,
    apply_structured_pruning: _analyze_structured_pruning,
    apply_structured_wanda_pruning: _analyze_structured_wanda_pruning,
    apply_structured_pruning_qdq: _analyze_structured_pruning_qdq,
    apply_structured_pruning_matmul_nbits: _analyze_structured_pruning_matmul_nbits,
    apply_attention_head_pruning: _analyze_attention_head_pruning,
    apply_attention_head_wanda_pruning: _analyze_attention_head_wanda_pruning,
    apply_moe_expert_channel_pruning: _analyze_moe_expert_channel_pruning,
    apply_moe_whole_expert_pruning: _analyze_moe_whole_expert_pruning,
    apply_qmoe_expert_channel_pruning: _analyze_qmoe_expert_channel_pruning,
    apply_qmoe_whole_expert_pruning: _analyze_qmoe_whole_expert_pruning,
    apply_embedding_vocab_magnitude_pruning: _analyze_embedding_vocab_magnitude_pruning,
    apply_transformer_block_pruning: _analyze_transformer_block_pruning,
}


def analyze_pruning_sensitivity(
    model: Union[str, onnx.ModelProto],
    apply_fn: Callable[..., Union[onnx.ModelProto, EmbeddingPruningResult]],
    **kwargs: Any,
) -> PruningSensitivityReport:
    """Dry-run sensitivity/"what would happen" report for one of this
    module's own mutating `apply_*_pruning` functions: given `model` and
    the exact same arguments a real call to `apply_fn` would take
    (`**kwargs` -- `sparsity`, `calibration_data`, `importance_norm`,
    `global_sparsity`, whichever subset `apply_fn` itself accepts), reports
    which layers/chains/heads/blocks/vocabulary rows that call would
    actually touch, how many of each one's channels/heads/weight-entries/
    blocks/rows it would drop, and a normalized "margin" proxy for how
    safe or risky that cut is -- all without ever mutating `model`. See
    :func:`weight_sparsity` for the complementary *after-the-fact*
    measurement this module already had (actual zero-fraction of an
    already-pruned model); this is the *before* half that was missing.

    `apply_fn` must be one of the fourteen functions this module itself
    exports: :func:`apply_magnitude_pruning`, :func:`apply_wanda_pruning`,
    :func:`apply_structured_pruning`, :func:`apply_structured_wanda_pruning`,
    :func:`apply_structured_pruning_qdq`,
    :func:`apply_structured_pruning_matmul_nbits`,
    :func:`apply_attention_head_pruning`,
    :func:`apply_attention_head_wanda_pruning`,
    :func:`apply_moe_expert_channel_pruning`,
    :func:`apply_moe_whole_expert_pruning`,
    :func:`apply_qmoe_expert_channel_pruning`,
    :func:`apply_qmoe_whole_expert_pruning`,
    :func:`apply_embedding_vocab_magnitude_pruning`, or
    :func:`apply_transformer_block_pruning` -- passed by reference (e.g.
    ``analyze_pruning_sensitivity(model, apply_wanda_pruning, sparsity=0.6)``),
    not by name. Each dispatches to its own dedicated `_analyze_*`
    implementation, which directly reuses that same family's own real
    matching (`_candidates`/`_find_*_chains`/`_match_embedding_chain`) and
    importance-computation helpers -- never a duplicated or reimplemented
    copy of either -- so the report's own numbers are computed the exact
    same way the mutating call itself would compute them, up to (but never
    actually calling) the final slice/zero/delete step. :func:`apply_sparsegpt_pruning`
    is not supported, and neither is :func:`apply_embedding_vocab_pruning`
    (a `ValueError` naming the fourteen functions that are supported); nor is
    `apply_structured_pruning`/`apply_structured_wanda_pruning`'s own
    `global_sparsity=True` mode or a model containing any `Concat`-merged
    skip-connection chain (both a `NotImplementedError`, from within the
    structured family's own dispatch) -- see this module's own "Dry-run
    pruning sensitivity analysis" section comment for the full reasoning
    behind every one of these scope decisions, `apply_embedding_vocab_pruning`
    included (see this module's own "Embedding / lm_head vocabulary family"
    dry-run section comment specifically for that one).

    :param model: the onnx ModelProto or file path `apply_fn` would take
    :param apply_fn: one of this module's own fourteen supported `apply_*`
            functions, passed by reference
    :param kwargs: forwarded to `apply_fn`'s own dedicated `_analyze_*`
            counterpart, which accepts the same parameters `apply_fn`
            itself does (minus `model`, passed positionally above)
    :returns: a :class:`PruningSensitivityReport` -- `layers` (one
            :class:`PruningLayerSensitivity` per matched unit) and
            `not_eligible` (labels of nodes `apply_fn` would decline
            outright, topology it doesn't match at all)
    """
    analyzer = _SENSITIVITY_ANALYZERS.get(apply_fn)
    if analyzer is None:
        supported = ", ".join(fn.__name__ for fn in _SENSITIVITY_ANALYZERS)
        raise ValueError(
            f"analyze_pruning_sensitivity does not support {apply_fn!r} -- "
            f"supported functions are: {supported}"
        )
    return analyzer(model, **kwargs)
