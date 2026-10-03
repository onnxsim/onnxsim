"""A Quark-ONNX-API-shaped quantization shim backed by onnxsim's own
quantizers -- no ``amd-quark`` install required.

It reproduces the public calling convention of ``quark.onnx`` (names and
signatures read from the ``amd-quark`` 0.13 wheel's ``quark/onnx/__init__.py``
and ``quark/onnx/quantization/config``) so a Quark script can switch to
onnxsim by changing its import:

    # before
    from quark.onnx import ModelQuantizer, QConfig
    # after
    from onnxsim.quark_compat import ModelQuantizer, QConfig

    config = QConfig.get_default_config("A8W8")
    ModelQuantizer(config).quantize_model(
        "model.onnx", "model.quant.onnx", calibration_data_reader
    )

The implementation is independent: Quark's source was read for its public
names and preset *meanings*, not copied.

**Scope, deliberately narrower than Quark:**

- Presets with a real backend here: ``A8W8``, ``A16W8`` (and the
  ``S8S8_AAWS`` / ``U8S8_AAWS`` / ``U8U8_AAWA`` / ``U16S8_AAWS`` /
  ``S16S8_ASWS`` / ``XINT8`` spellings, see ``_PRESETS``), ``FP16``, ``BF16``.
  Integer presets use :func:`onnxsim.full_qdq.quantize_full_qdq`: signed /
  unsigned 8- and 16-bit activations, symmetric (``A8W8``, ``A16W8``,
  ``XINT8``: centred zero point) or asymmetric with percentile calibration
  (``*_AAWS``), power-of-2 scales for ``XINT8``, and symmetric int8 weights
  **per tensor** like Quark (``extra_options={"PerChannel": True}`` for per
  channel; the weight-rounding algorithms need it and switch it on). Scales and
  zero points match Quark's for the probed models
  (``tests/test_quark_parity.py``). Calibration follows the preset: MinMax
  (``A8W8``, ``A16W8``), Percentile (99.999; ``S8S8_AAWS`` 99.9999; the
  ``Int8Spec`` family's default, as in Quark) and, for ``XINT8``, Quark's
  power-of-two MinMSE (``method="minmse_pof2"`` in
  :func:`onnxsim.calibration.calibrate`): the same 2048-bin histogram and five
  candidate scales, so activation scales are identical to Quark's, and
  weights and biases get the same MinMSE search -- biases are **int8** with a
  per-tensor power-of-two scale like Quark's
  (``extra_options={"Int32Bias": True}`` keeps int32). Like Quark, every
  non-weight constant of a quantized node (LayerNorm scale, Mul operand, ...)
  is quantized as an int8 weight (activation dtype for Add / Sub / Mul / Div /
  Min / Max constants under ``A16W8``'s ``AlignEltwiseQuantType``), and
  Softmax outputs are calibrated to the fixed range (0, 1) except under
  ``XINT8``.
  Every other :class:`CalibMethod` is Quark's calibrator, scale for scale
  (:mod:`onnxsim.quark_calibration`: Quark's growing-histogram layout and
  its search, so ranges agree to float32 rounding -- exact, not to histogram
  binning): ``Percentile`` (symmetric absolute-value histogram, or the
  two-sided one with ``CalibTensorRangeSymmetric=False``), ``Entropy``
  (128 bins / 128 quantized bins by default, ``NumBins`` /
  ``NumQuantizedBins``), ``Distribution`` (the histogram extent) and
  ``LayerwisePercentile`` (``LWPMetric``, ``PercentileCandidates``). The
  ``calibration_method`` strings ``"entropy"``, ``"percentile[:p]"``,
  ``"distribution"`` and ``"layerwise_percentile"`` mean the same;
  ``"onnxsim:entropy"`` etc. select onnxsim's own variants. Calibration
  ``extra_options`` read: ``Percentile``, ``CalibTensorRangeSymmetric``,
  ``CalibMovingAverage`` (mean of the per-batch ranges), ``CalibDataSize``,
  ``NumBins``, ``NumQuantizedBins``, ``LWPMetric``, ``PercentileCandidates``
  (``Scenario`` only changes Distribution's float-8 statistics; ``CalibWorkerNum``
  / ``CalibOptimizeMem`` / ``LWPUseHistogram`` have no effect). Distribution
  reports ``(-T, T)`` for a post-Relu tensor; for uint8 activations Quark then
  folds the Relu node onto that centred grid, which no longer clamps -- onnxsim
  reproduces the graph and warns. Not matched: Quark's non-power-of-two
  ``MinMSE`` (Quark's ``CalibMethod.MinMSE`` is the power-of-two search, which
  is what :class:`CalibMethod` maps it to).
- ``XINT8`` (power-of-two activations and weights, ``EnableNPUCnn``): Quark's
  NPU CNN quantizer also simulates the DPU and moves the power-of-two
  positions to the compiler's limits; :mod:`onnxsim.quark_npu` reproduces it
  graph for graph (``tests/test_quark_xint8_parity.py``): ``AveragePool`` /
  ``GlobalAveragePool`` / ``ReduceMean`` -> followed by a ``Mul``,
  ``Sigmoid`` -> ``HardSigmoid`` + ``Mul``, ``LeakyRelu`` alpha in 1/256,
  ``AlignConcat`` / ``AlignPool`` / ``AlignPad`` / ``AlignSlice`` to the smaller
  position and the ``AdjustShiftCut`` / ``Bias`` / ``Read`` / ``Write`` /
  ``HardSigmoid`` / ``Swish`` passes (options ``SimulateDPU``,
  ``NPULimitationCheck``, ``MaxLoopNum``, ``Convert*ToDPUVersion`` ...). The
  float graph is converted first like Quark's pre-processing
  (:mod:`onnxsim.quark_convert`): BatchNorm folded into a Conv or turned into a
  depthwise Conv (``ConvertBNToConv``), ``ReduceMean`` -> ``GlobalAveragePool``,
  a large ``GlobalAveragePool`` split (``SplitLargeKernelPool``), ``Split`` ->
  ``Slice`` (``ConvertSplitToSlice``), Pad fusion, HardSwish inlining and
  Identity removal (``OptimizeModel`` / ``SimplifyModel``); these conversions are
  also on for the extended (``A8W8`` ...) and transformer flows, and follow
  their options in ``VINT8`` (whose ``ConvertClipToRelu`` is implemented too).
  Also reproduced (``tests/test_quark_xint8_parity.py``): ``ConvertSoftmaxToDPUVersion``
  (the bfloat16 exponential / sum / division chain) and
  ``ConvertInstanceNormToDPUVersion`` (``ExtendedInstanceNormalization``); the
  op types Quark quantizes (its registries only: a ``Flatten`` or ``Neg`` marks
  nothing); the order-dependent rule that leaves a ``Relu`` / ``Clip`` whose input
  no earlier node marked as a plain float node (so one fed by a graph input keeps
  that input float); the node order of Quark's own topological sort -- of the float
  graph and again of the Q/DQ graph, with the DPU nodes appended at its end, which
  the position passes visit (:mod:`onnxsim.quark_marking`); ``Pad``'s
  ``constant_value`` quantized like a weight; a shared bias quantized once (or
  copied per node under ``CopyBiasInit`` for the non-power-of-two calibrations);
  the zero point and scale of an all-zero activation. Quark's float-graph
  optimizers are run for real when installed (onnxslim's ``slim``, then ONNX
  Runtime's basic graph optimizations with ``ConstantSharing`` off, then Quark's
  own BatchNorm folding after a ConvTranspose / Gemm / Concat), so constant
  folding, Conv + Add / Mul, Relu + Clip, MatMul + Add -> Gemm, no-op and
  duplicated node removal are Quark's; without them (or with
  ``UseRuntimeOptimizers=False``) onnxsim's own reproductions of the BatchNorm,
  Pad, Identity and HardSwish passes are used. The calibration session runs with
  ONNX Runtime's optimizations off, as Quark's does (a fused or re-laid-out graph
  moves values across histogram bin edges). Quark's own operator fusions run
  after them (:mod:`onnxsim.quark_fusions`, ``tests/test_quark_fusions_parity.py``;
  on by default, in every preset, also with ``OptimizeModel`` off): the
  TensorFlow-style InstanceNorm -> ``InstanceNormalization`` (``FuseInstanceNorm``),
  the ``ReduceSum`` / ``Max`` / ``Reciprocal`` L2 normalization ->
  ``LpNormalization`` (``FuseL2Norm``), the decomposed LayerNorm ->
  ``LayerNormalization`` (``FuseLayerNorm``, opset >= 17) and the erf Gelu -> the
  ``com.microsoft`` ``Gelu`` (``FuseGelu``, opset >= 20); ``SkipPreprocess`` skips them
  and the other pre-processing, ``ConvertOpsetVersion`` runs before them. ONNX Runtime's
  own optimizer fuses a torch LayerNorm / Gelu first (``UseRuntimeOptimizers=False``
  stands in for that too).
- The block-format (BFP16, MX4 / 6 / 9, MXFP4 / 6 / 8, MXINT8), bfloat16 / float16 and
  ``MATMUL_NBITS`` flows run the same pre-processing as Quark does for them
  (``tests/test_quark_block_preproc_parity.py``: node by node, with the same ONNX
  Runtime outputs): onnxslim, ONNX Runtime's basic optimizer, BatchNorm folding, the
  operator fusions, the extended ``QDQ`` format's conversions (``ReduceMean`` ->
  ``GlobalAveragePool``, a leftover ``BatchNormalization`` -> ``Conv``, ``Split`` ->
  ``Slice``; not for ``MATMUL_NBITS``, whose plain ``QDQ`` format leaves them off) and
  the topological sort; ``SkipPreprocess`` skips all but the sort, and a shared bias is
  copied (``CopyBiasInit``) for ``MATMUL_NBITS``'s integer types only. As in Quark,
  nothing runs -- and the model comes back as given, opset included -- when no node
  of the registries' op types (``QuantizeAllOpTypes``: every op type; BF16's preset)
  touches a float tensor; a ``BatchNormalization`` that op list holds is left to the
  quantizer (Quark's own BN folding skips it). Which tensors get a block node or a
  Q/DQ pair is Quark's marking (:func:`onnxsim.quark_marking.skipped_nodes`): the
  FP16 / BF16 presets set ``ForceQuantizeNoInputCheck`` (BF16 also
  ``QuantizeAllOpTypes``), a bare ``Clip`` / ``Relu`` output is quantized behind a
  quantized producer, a Conv / Gemm weight or bias that is not a constant is not, the
  float16 / bfloat16 pair of a ``Reshape`` / ``Transpose`` / ``Squeeze`` /
  ``Unsqueeze`` / ``Resize`` / ``MaxPool`` / ``Split`` / ``Gather`` output reads its
  input's scale and zero point, a Conv / MatMul / Gemm / ... output before a ReLU-like
  op and a ``Pad`` before a pool go without a pair (``RemoveQDQConv*``), unused
  constants are dropped, and the ``opset_import`` list is Quark's (the model's, then
  ``com.microsoft``, then ``com.amd.quark``). Not covered: float16 *input* models
  (Quark's ``QuantizeFP16`` mode), the ``AutoMixprecision`` presets
  (``BF16_MIXED_*``, ``tests/test_quark_amp_parity.py``) run no pre-processing, and the
  output of a ``Gather`` on a constant table in the mixed presets (``BF16_BFP16``,
  ``MX9_INT8``) is quantized with the baseline format instead of the table's.
- Models with a default-domain opset below 13 are quantized in place like Quark
  does (``tests/test_quark_low_opset_parity.py``): the opset is never converted,
  per-tensor Q/DQ carry no ``axis`` (the bias DequantizeLinear gets Quark's
  one-element scale and scalar zero point), a per-channel configuration
  (``extra_options["PerChannel"]``) raises Quark's ``Per-Channel support with QDQ
  format requires onnx opset version 13 or above.``, ``MaxPool`` is left alone
  below opset 12 and ``Resize`` below opset 11 (ONNX Runtime's QDQ operator
  quantizers, reused by Quark's registries, return without marking anything),
  and the ``com.microsoft`` / ``com.amd.quark`` operator sets are registered
  on every quantized model. Quark's extended quantizer (``A8W8`` and the 16-bit
  presets) emits ``com.microsoft`` Q/DQ nodes. GPTQ re-grids the weights per
  tensor there, and ``UINT8_DYNAMIC_QUANT`` below opset 11 computes each
  activation's scale and zero point with ordinary operators (no
  ``DynamicQuantizeLinear``), as Quark does. ``extra_options["ConvertOpsetVersion"]``
  converts the opset first, like Quark's pre-processing (which is how a per-channel
  configuration is reached from such a model). Not covered below opset 13: the
  legacy AdaQuant engine's per-channel mode.
- Q/DQ placement and quantizer options follow Quark's rules
  (:func:`onnxsim.full_qdq.quantize_full_qdq`, ``tests/test_quark_parity.py``):
  the Q/DQ pair between a Conv / Add / MaxPool / AveragePool /
  GlobalAveragePool / MatMul / Gemm / ConvTranspose (and, with
  ``RemoveQDQInstanceNorm``, InstanceNormalization) and its single consumer is
  dropped when the consumer is a Relu (``RemoveQDQConvRelu``), Clip with
  bounds (0, 6) or (0, 1) (``RemoveQDQConvClip``), LeakyRelu
  (``RemoveQDQConvLeakyRelu``), PRelu (``RemoveQDQConvPRelu``) or, opt-in, Gelu
  (``RemoveQDQConvGelu``); a Relu / Clip node whose input range Quark inherits
  from its output keeps its own Q/DQ with that range when the pair stays. For
  asymmetric activations the Relu / Clip node itself folds into its producer
  (always under the plain QDQ quantizer; under the extended one -- Quark's
  ``A8W8`` and 16-bit presets, ``QConfig.quant_format`` -- only with
  ``FoldRelu``). ``ActivationSymmetric`` / ``WeightSymmetric`` override the
  specs' symmetry (asymmetric or uint8 weights included; Quark clips weight
  codes to the symmetric code range), ``QuantizeBias=False`` keeps biases
  float, and the extended quantizer's ``AlignConcat`` / ``AlignPool`` /
  ``AlignPad`` / ``AlignSlice`` / ``AlignTranspose`` / ``AlignReshape`` copy
  quantization parameters (Concat / Pad / Transpose / Reshape inputs from their
  output, Pool / Slice outputs from their input). Quark's ``Slice`` (and, under
  the extended quantizer, ``Split``) outputs are calibrated on their own, its
  plain quantizer's ``AveragePool`` shares its input's parameters, and an
  InstanceNormalization bias is an int32 bias. ``extra_options["ReduceRange"]``
  is Quark's legacy ``QuantizationConfig.reduce_range``: weights (and the
  constants quantized like weights) keep to the reduced code range, ``[-64,
  64]`` for int8, ``[0, 127]`` for uint8; activations are untouched. It is
  refused with ``XINT8`` (Quark refuses it too) and with power-of-two weights
  otherwise. Not implemented: ``AlignEltwise`` beyond ``AlignEltwiseQuantType``
  and the 16-bit ``AlignPool`` etc. for ``XINT8``.
- The details the other integer presets (``A8W8``, ``A16W8``, ``VINT8``, the
  ``*_AAWS`` ones, ``INT8_CNN_DEFAULT``, the transformer ones) depend on, all
  checked against Quark 0.13 on randomized graphs (``tests/test_quark_int_presets_parity.py``):
  ONNX Runtime's ``adjust_tensor_ranges`` runs twice (a Relu / Clip chain passes
  its range two steps); the extended quantizer's alignment rewrites the shared
  scale / zero-point initializers in place (a MaxPool output that shares its
  input's moves with it) and its ``adjust_bias_scale`` re-quantizes the int32
  biases with truncation after every alignment round; ``AlignEltwiseQuantType``
  (extended quantizer only) gives every eltwise input its own parameters instead of
  sharing; an all-zero activation range is scale 1 / zero point 0 for every
  integer type; int32 biases saturate at the int32 limits; a Softmax gets the
  unit range only when the quantizer quantizes it (not in the transformer
  scheme); ``CalibMovingAverage`` averages the batch extremes in float32.
  ``VINT8`` does not force data-movement ops on an unmarked input to quantize,
  quantizes the op types of the model it is handed (so not the Slices
  ``ConvertSplitToSlice`` makes), quantizes a PRelu slope like a weight, and
  with ``DedicatedQDQPair`` gives each *quantized* reader (each input slot of
  it) its own Q/DQ pair -- other readers see the float tensor, and a graph output
  read by several of them stays float. Also: an excluded node is only left
  unmarked (its quantized neighbours still wrap it in Q/DQ pairs); a ``Gemm``
  ``beta`` moves into the int32 bias scale (ONNX Runtime's ``QDQGemm``);
  ``Int32Bias=False`` biases follow ``PerChannel`` (one scale per element) and
  ``WeightSymmetric=False``; an asymmetric power-of-two weight takes the zero
  point of the min / max scale and the best MinMSE scale around it; a PRelu slope
  is per row under ``PerChannel`` with the extended quantizer. Not reproduced:
  Quark's failures on graphs its own pre-processing breaks (an ``x - mean(y)``
  pattern its InstanceNormalization fusion chokes on).
- Per-layer overrides: ``layer_type_config`` then ``specific_layer_config``
  (which wins) retarget the *activation* dtype / symmetry of a layer's inputs
  (``input_tensors``, or the deprecated ``activation``) and outputs
  (``output_tensors``) among int8/uint8/int16/uint16; a ``None`` key in
  ``layer_type_config`` and ``exclude`` keep nodes float. Node names may be
  Quark's ``^...*`` regular expressions (subgraph tuples raise). Weight dtypes
  other than int8 raise; ``bias`` specs are ignored (biases stay int32).
  Scales / zero points match Quark's (``tests/test_quark_parity.py``).
- ``INT8_TRANSFORMER_DEFAULT`` / ``INT16_TRANSFORMER_DEFAULT`` /
  ``INT8_TRANSFORMER_ACCURATE`` / ``INT16_TRANSFORMER_ACCURATE`` (Quark's
  ``enable_npu_transformer``): asymmetric uint8 / uint16 activations, per-tensor
  symmetric int8 / int16 weights, int32 biases, and -- the whole difference to
  the CNN presets -- *only* ``Gemm`` and ``MatMul`` nodes whose second operand
  is a constant are quantized (``MatMulConstBOnly=False`` adds the
  activation x activation MatMuls). Softmax, LayerNormalization, Gelu, Add,
  Mul, Transpose, ... stay float, with Q/DQ only on the inputs and outputs of
  the quantized nodes (a Gemm / MatMul feeding a sole Relu-like consumer keeps
  a float output, as Quark's Q/DQ removal does). A model with no such node is
  returned as it is, nodes in Quark's order (Quark: "No quantizable ops"). DEFAULT calibrates with the
  *mean over batches of each batch's min / max* (Quark's ``CalibMovingAverage``,
  ``method="minmax_mean"``); ACCURATE is percentile 99.9999 + AdaRound like
  ``INT*_CNN_ACCURATE`` (also on int16 weights, which Quark's FastFinetune
  trains too). Placement, scales,
  zero points, weights and outputs equal Quark's for the probed models
  (MLP, attention + MLP block, Conv, residual Gemm chain), the ACCURATE ones to
  histogram binning. Not matched: Quark's pre-processing (``MatMul`` + ``Add``
  -> ``Gemm`` fusion, Gelu fusion at opset >= 20 -- a model that relies on it
  quantizes differently), its implicit CLE, and the AdaRound weight codes
  themselves (a different optimizer run; the same objective).
- ``UINT8_DYNAMIC_QUANT``: :mod:`onnxsim.quark_dynamic` emits Quark's /
  ONNX Runtime's dynamic pattern (``DynamicQuantizeLinear`` +
  ``MatMulInteger`` / ``ConvInteger``); no calibration data is needed.
- Block formats (``BFP16``, ``MX4/6/9``, ``MXFP4/6/8``, ``MXINT8``):
  :mod:`onnxsim.quark_fakequant_graph` inserts the same ``com.amd.quark``
  ``BFPQuantizeDequantize`` / ``MXQuantizeDequantize`` nodes Quark's quantizer
  does -- same tensors, names, attributes and block axes for the ops it lists
  (activations, outputs, weights *and* biases) -- so the model runs wherever
  Quark's ONNX custom-op library is registered
  (``quark.onnx.operators.custom_ops.get_library_path()``); **onnxsim cannot
  execute those nodes itself**. ``tests/test_quark_parity.py`` checks this
  against the installed ``amd-quark`` in CI: identical placement on the probed
  ops and bit-identical outputs under ONNX Runtime. Not replicated: Quark's
  model pre-processing (BatchNormalization folding, ``ReduceMean`` ->
  ``GlobalAveragePool``, the implicit CLE below). Options (``QConfig(..., extra_options=...)``):
  ``BlockFormatActivations=False`` quantizes only the constants, offline, so the
  model runs anywhere; ``BlockFormatFoldWeights=True`` folds the constants
  offline (via :mod:`onnxsim.quark_block_formats`) instead of leaving a node on
  them. ``algo_config`` is not applied to block formats.
  Dynamic quantization raises ``NotImplementedError``.
- ``algo_config``: the float -> float passes run in Quark's order (stem
  equalization + CLE, SmoothQuant, Quarot) before quantization, and are
  *bit-identical* to Quark's (``tests/test_quark_algo_parity.py``):
  CLE (:mod:`onnxsim.quark_equalization`) with ``CLESteps`` /
  ``CLEWeightThreshold`` / ``CLEScaleAppendBias`` / ``CLEScaleUseThreshold`` /
  ``CLETotalLayerDiffThreshold`` (``CLEBalanceMethod`` only has ``"max"``, as in
  Quark; ``ReplaceClip6Relu``; ``CLEConfig`` fields or the same-named
  ``extra_options``, which win), Conv / Gemm pairs and Conv - depthwise Conv -
  pointwise Conv triples. A Conv without an explicit ``group`` attribute is
  skipped, as in Quark. SmoothQuant (:mod:`onnxsim.quark_smoothquant`, ``alpha``
  or ``extra_options["SmoothAlpha"]``) smooths constant-weight ``MatMul``
  nodes only -- no ``Gemm``, no LayerNorm folding (it inserts a ``Mul`` per
  MatMul) -- over activations of any rank. Quark enables CLE implicitly in
  *every* preset (``include_cle=True``) while onnxsim only runs it when
  ``CLEConfig`` is listed, so a preset's weights differ from Quark's wherever a
  CLE pattern exists. BiasCorrection
  (:mod:`onnxsim.quark_bias_correction`) rewrites the quantized *bias* of
  every Conv / Gemm that has one, from the layer-local float - quantized
  output mean, exactly as Quark does (integer biases equal Quark's for
  ``MinMax`` / ``Percentile`` calibration; the other histogram calibrators
  leave the biases alone, like Quark). With power-of-two calibration (``XINT8``)
  Quark re-derives the bias scale through its power-of-two quantizer without
  storing it, so the integer codes and the stored scale disagree wherever the
  fresh scale differs -- a Quark quirk that is reproduced for parity (identical
  codes, with a warning); ``extra_options["BiasCorrectionStoredScale"]=True``
  writes codes for the stored scale instead, i.e. what the float intent says.
  AutoMixprecision replaces the plain quantization step with
  :func:`onnxsim.quark_auto_mixprecision.auto_mixprecision` and mixes what
  Quark's ``MixingStrategy`` mixes, by the same in-place edit of the quantized
  baseline (:class:`onnxsim.quark_mixing.QuarkMixer`): a target
  ``QLayerConfig``'s ``activation`` moves a layer's activation inputs *and*
  outputs, ``input_tensors`` / ``output_tensors`` one side each, ``weight`` /
  ``bias`` the constants (a weight is re-quantized per tensor from its
  *already quantized* values, the int32 bias scale is refreshed to
  ``input_scale * weight_scale`` and its codes truncated, a layer that does not
  move keeps its baseline bias -- as a one-element scale vector, as in Quark).
  Every slot can go to any precision: integer (int8 / uint8 / int16 / uint16,
  power-of-two scales -- a ``PowerOf2`` scale type or ``MinMSE`` calibration --
  rounded as Quark's ``PowerOfTwoMethod`` does), ``float16`` / ``bfloat16`` (an
  ``ExtendedQuantizeLinear`` / ``ExtendedDequantizeLinear`` pair, scale 1) or a
  BFP / MX block format (a ``com.amd.quark`` node), over an integer baseline or
  a ``float16`` / ``bfloat16`` / BFP / MX one (its Quark baseline, with Quark's
  fake ``[0, 1]`` ranges and its default ``BFPAttributes`` / ``MXAttributes``);
  ``tests/test_quark_amp_parity.py`` compares the whole quantizer structure and
  the outputs bit for bit with Quark for the mixes it lists. A
  ``QuantizeBias=False`` bfloat16 baseline with a block target (the
  ``BF16_MIXED_*`` presets) keeps its dedicated flow. A half / block baseline
  takes any constant format: activations in ``float16`` / ``bfloat16`` / BFP /
  MX with weights in any half, block, ``int8`` or ``uint8`` format (Quark's
  quantizer picks the node kind of every tensor from its own dtype; the BFP / MX
  attributes are its plain defaults unless the activation and weight formats are
  the same ``MX*`` format or ``BFPAttributes`` / ``MXAttributes`` say so);
  integer activations over half / block weights are not implemented.
  ``target_layer_config`` as one ``QLayerConfig``, a list (each candidate takes
  its best-scoring config) or ``{QLayerConfig: [node names]}``; ``subgraph_json``
  partitions (a missing file is ignored, as in Quark), a
  ``sensitivity_cache_file`` (Quark's JSON schema *and* key: Quark reads a
  ranking onnxsim wrote and onnxsim one Quark wrote, see
  :mod:`onnxsim.quark_amp_cache`; ``"enabled": false`` pins a layer),
  ``worker_num`` threads, ``no_input_qdq_shared`` and ``shared_param_mode``
  (Quark's ``"propagate"`` / ``"unshare"`` for the scale / zero point
  initializers a promoted quantizer shares with the ones around a pass-through
  op such as ``Transpose`` or ``MaxPool``; the baseline shares them as ONNX
  Runtime's quantizer does) and ``dual_quant_nodes`` -- on the int16 -> int8
  promotion (``S16S16_MIXED_S8S8``) and the block-format presets
  (``BF16_MIXED_BFP16`` / ``_MXINT8``) too. ``dual_quant_nodes`` is Quark's
  post-processing of the *final* mixed model (candidates are scored without
  it, see :mod:`onnxsim.quark_boundary_qdq`) and works for every mix -- integer,
  half, block and power-of-two, over an integer or a float / block baseline --
  graph for graph and bit for bit. Candidates
  are scored like Quark's analysis does: ONNX Runtime with every graph
  optimization off, over ``data_size + 1`` calibration batches (so the default
  ``0`` scores one batch, whatever "0 = all" says), which is what makes the
  ranking, the cache's scores and the moved layers equal to Quark's, bit for
  bit. A model with ``com.amd.quark`` nodes runs on
  :func:`onnxsim.quark_fakequant_eval.run_fake_quantized` (ONNX Runtime for the
  ordinary ops, bit-exact numpy for the ``com.amd.quark`` ones).
  AdaRound and AdaQuant are Quark's ``FastFinetune``
  (:mod:`onnxsim.quark_finetune`, a numpy port of ``quark.onnx.algorithm.
  finetuning``): per Conv / ConvTranspose / Gemm / MatMul / InstanceNorm /
  LayerNorm *block* (input Q/DQ, op, bias, a following Relu / PRelu / LeakyRelu /
  Clip / Sigmoid / Tanh / Gelu / Softmax, optionally the output Q/DQ), in graph order,
  the quantized model's layer input is re-captured after every update
  (``parallel=True``: once up front), mini-batches of ``batch_size`` samples are
  drawn each iteration, and Quark's loss, ``early_stop`` rule, cosine ``beta``
  schedule, ``lr_adjust``, ``drop_ratio`` (default 1.0, as in Quark),
  ``selective_update`` (end-to-end L2 over ``output_index``), ``output_qdq``,
  ``num_batches``, ``data_size``, ``target_op_type``, ``select_max_mem_layer``,
  ``fixed_seed`` and ``QuantizationPreference="accuracy"`` are implemented;
  AdaQuant trains the float weight (and, with ``update_bias``, the quantized
  bias) straight-through at ``learning_rate=1e-5`` and re-quantizes it. The
  presets carry the ``FastFinetune`` dict of Quark's (``batch_size=2``,
  ``early_stop=True``, ``data_size=1000``, ...). Given Quark's own random
  stream the integer codes are identical to Quark's for AdaRound (and for short
  AdaQuant runs), see ``tests/test_quark_finetune_parity.py``; with numpy's
  stream the result agrees statistically (the optimization is stochastic in
  Quark too). Exactly like Quark 0.13, the config fields ``reg_param`` /
  ``beta_range`` / ``warm_start`` / ``parallel`` / ``output_index`` /
  ``ref_model_path`` / ``dynamic_batch`` are *not* forwarded (only
  ``extra_options["FastFinetune"]`` reaches them) and ``update_bias`` only
  matters to AdaQuant; ``extra_options["FastFinetune"]`` keys override the
  config. ``num_workers`` and ``dynamic_batch`` are modelled (``MemOptLevel=2``
  is Quark's separate ``DataLoader`` loop, mirrored; ``DynamicBatch`` only
  works there for readers that yield one sample per batch, otherwise Quark
  trains nothing). No effect on the numbers, so accepted and ignored:
  ``optim_device`` / ``infer_device`` / ``pin_memory`` / ``use_gds`` /
  ``log_period`` / ``cache_dir`` and ``mem_opt_level`` 0 vs 1 (probed against
  Quark: identical codes). ``extra_options["SaveAndRestore"]`` (a JSON file) is
  Quark's checkpoint: the layer reached is written before every layer
  (``model_to_finetune`` next to it, ``layers_to_finetune``) and an existing file
  restricts training to the layers it lists, from the *original* quantized model
  (Quark drops the model it loads); the calibration ranges Quark also keeps in that
  file are not written.
  ``ref_model_path`` must be a float model. Layers Quark's torch modules cannot
  handle are skipped exactly where Quark skips them (``auto_pad``,
  ``ConvTranspose`` with ``output_padding`` / ``output_shape`` or asymmetric
  pads, a ``Gemm`` with ``transA`` unless its shapes line up -- see
  :mod:`onnxsim.quark_finetune`; ``select_max_mem_layer`` raises there, as
  Quark does) and what Quark does train is trained: 1-D / 2-D / 3-D ``Conv`` /
  ``ConvTranspose`` (also grouped), ``PRelu`` blocks (with torch's fixed 0.25
  slope, as Quark), bias-less ``Gemm`` (with Quark's random Linear bias) and
  int16 / uint8 / asymmetric weights; weights stay *per tensor* here too
  (Quark's default), unlike the
  GPTQ and legacy AdaQuant paths. onnxsim additions: ``guard`` (default on:
  keep a layer's new codes only if its block error did not rise; ``False`` is
  Quark's behaviour) and ``AdaQuantConfig(legacy_engine=True)`` for the older
  :func:`onnxsim.apply_adaquant` (a different algorithm: rounding relaxation
  plus a learnable activation range). GPTQ is in
  :mod:`onnxsim.quark_weight_rounding` (Conv / Gemm / MatMul, guarded so a
  layer's reconstruction error never gets worse). Like Quark's GPTQ (which never
  raises for them) it runs for int16 / uint8 weight presets too, re-gridding the
  float weights to 8 bits whatever the preset's weight dtype.
  GPTQ with
  ``bits`` / ``group_size`` / ``per_channel`` / ``mse`` / ``weight_symmetric``
  set re-grids the weights the way Quark's GPTQ does (``bits``-bit codes,
  per-tensor / per-channel / per-group scales, scales and zero points written
  back into the QDQ weights; ``group_size`` needs a blocked
  ``DequantizeLinear``, opset >= 21, which ONNX Runtime only runs next to
  activation Q/DQ with ``session.disable_quant_qdq=1``); with none of them set
  the model's own scales are kept. ``act_order`` together with ``group_size``
  raises. Quark 0.13's GPTQ error-propagation step is a no-op (it indexes a
  triangular factor by column), so Quark's result is round-to-nearest on its
  grid; onnxsim really propagates the error (and is never worse). Quarot folds the R1 residual-stream rotation into the float
  weights before quantization (:mod:`onnxsim.quark_quarot`; needs
  ``r_config_path``; R2-R4 do not exist, as in Quark's ONNX flow -- its
  ``transform`` has them as TODO). The R1 weights are bit-identical to
  Quark's for the same matrix, and the power-of-two Hadamard matrix is the
  same, and so are the tabulated Hadamard matrices Quark uses for sizes 12, 20,
  28, 36, 40, 52, 60, 108, 140, 156, 172 times a power of two
  (:mod:`onnxsim.quark_hadamard`; any other size raises the same error as
  Quark). ``UseRandomHad`` draws its row signs from numpy, not torch's RNG. An
  ``algo_config`` that cannot run for a preset (block formats, FP16 / BF16)
  raises ``NotImplementedError`` unless ``ignore_unsupported_algos=True``.
- ``MATMUL_NBITS`` (``UseMatMulNBits``): weight-only 4-bit quantization to
  ``com.microsoft::MatMulNBits`` (:mod:`onnxsim.quark_matmul_nbits`, which
  documents the layout, numerics and limits); no activation calibration.
  ``MatMulNBitsParams`` (``GroupSize`` 128, ``Symmetric`` True, ``Bits`` 4,
  ``AccuracyLevel`` 1, ``Algorithm`` DEFAULT / HQQ / GPTQ) and ``GPTQParams``
  are read (a ``GPTQConfig`` in ``algo_config`` is ignored, as in Quark);
  packed weights, scales, zero points and attributes are identical to Quark's
  for all three algorithms
  (``tests/test_quark_parity.py``). Differences: ``Bits != 4`` raises (Quark
  emits an unrunnable model), MatMul + Add pairs that Quark's ONNX Runtime
  pre-processing would fuse into an unquantized Gemm are converted
  (``SkipPreprocess=True`` in Quark equals this), ``exclude`` is honoured, GPTQ
  error propagation is Quark's no-op unless ``GPTQParams["Compensate"]``, and
  other ``algo_config`` entries raise.
- ``Constant`` nodes and the conversion scope
  (``tests/test_quark_constants_misc_parity.py``): with neither ``OptimizeModel``
  nor ``SimplifyModel`` (or with ``SkipPreprocess``) the ``Constant`` nodes of the
  float graph stay in the quantized graph, as in Quark, whose quantizers then treat
  their outputs as activations -- a Conv weight / Mul operand / Pad value gets a
  calibrated ``Q -> DQ`` pair, a Conv / Gemm / InstanceNormalization bias stays
  raw, integer parameters are not quantized, ``VINT8`` (``QuantizeAllOpTypes``)
  quantizes every float one as an op, the NPU transformer scheme does not take a
  ``MatMul`` with a Constant ``B`` for a weight MatMul -- and the nodes come in
  Quark's order. The ``Constant`` nodes Quark's own conversions add (``Split`` ->
  ``Slice``) stay whatever the optimizers did. Quark's pre-optimizer converts only
  nodes whose op type the quantizer takes: the transformer scheme (Gemm / MatMul)
  leaves a ``ReduceMean`` / ``Split`` / ``Clip`` / pool alone (a ``BatchNormalization``
  is still converted, the conversion adds its type to the list first -- so a
  left-over one is quantized there too), a plain
  quantizer does not convert a ``ReduceMean`` (not in its registry). The extended
  quantizer (``A8W8``, the 16-bit presets) runs the DPU simulation too -- every
  ``Convert*ToDPUVersion`` / ``ConvertSigmoidToHardSigmoid`` option opt-in, with
  no ``value_info``, so a ``GlobalAveragePool`` / ``ReduceMean`` window is never
  found (an ``AveragePool`` with a square kernel is rescaled); a ``Split`` that stays
  a ``Split`` shares its input's quantization parameters there too.
- ``extra_options`` are stored, not interpreted -- except ``PerChannel``,
  ``Int32Bias`` (``False``: the bias is quantized like a weight, int8 / int16
  per the weight dtype), ``DedicateDQNode`` (block-format AutoMixprecision
  presets: a shared dequantizer is copied per reader), ``AlignEltwiseQuantType``, ``MatMulConstBOnly`` (the
  transformer presets), the block-format options above, the ``MATMUL_NBITS`` ones, and, for the integer
  presets, the calibration options (``Percentile``, ``CalibTensorRangeSymmetric``,
  ``CalibMovingAverage``, ``CalibDataSize``, ``NumBins``, ``NumQuantizedBins``,
  ``LWPMetric``, ``PercentileCandidates``), ``RemoveQDQConv{Relu,Clip,LeakyRelu,
  PRelu,Gelu}``, ``RemoveQDQInstanceNorm``, ``FoldRelu``, ``Align{Concat,Pool,
  Pad,Slice,Transpose,Reshape}``, ``ActivationSymmetric``, ``WeightSymmetric``,
  ``QuantizeBias``, and the pre-processing options of the integer presets
  (``OptimizeModel``, ``SimplifyModel``, ``Fuse{InstanceNorm,L2Norm,LayerNorm,Gelu}``,
  ``FoldBatchNorm``, ``ConvertOpsetVersion``, ``SkipPreprocess``,
  ``QuantizeAllOpTypes``; see above).
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)

import numpy as np
import onnx

if TYPE_CHECKING:
    from onnxsim.quark_auto_mixprecision import TargetSpec

# -- data-type specs ---------------------------------------------------------


@dataclass(eq=True, unsafe_hash=True)  # hashable: usable as a dict key, as in Quark
class QSpec:
    """Base of the per-tensor spec classes (Quark's ``Int8Spec`` etc.).

    ``dtype`` is the onnx-style name; ``symmetric`` / ``pof2`` mirror the
    fields Quark's specs expose that change which backend path is valid.
    """

    dtype: str = "int8"
    symmetric: bool = True
    pof2: bool = False
    #: ``"minmax"``, ``"percentile[:p]"``, ``"entropy"``, ``"distribution"``,
    #: ``"layerwise_percentile"`` (Quark's calibrators of those names, see
    #: :mod:`onnxsim.quark_calibration`), ``"minmse_pof2"`` (Quark's MinMSE),
    #: ``"mse"`` / ``"onnxsim:<method>"`` (onnxsim's own methods, e.g.
    #: ``"onnxsim:entropy"``) or a :class:`CalibMethod`
    calibration_method: Any = "minmax"
    is_dynamic: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.calibration_method, CalibMethod):
            self.calibration_method = _CALIB_NAMES[self.calibration_method]


class CalibMethod(Enum):
    """Quark's ``CalibMethod`` (``quark.onnx.CalibMethod``). ``MinMSE`` is its
    power-of-two MinMSE search (:mod:`onnxsim.calibration` ``"minmse_pof2"``);
    ``Percentile`` / ``Entropy`` / ``Distribution`` / ``LayerwisePercentile``
    are Quark's histogram calibrators (:mod:`onnxsim.quark_calibration`)."""

    MinMax = 0
    MinMSE = 1
    Percentile = 2
    Entropy = 3
    LayerwisePercentile = 4
    Distribution = 5


_CALIB_NAMES = {
    CalibMethod.MinMax: "minmax",
    CalibMethod.MinMSE: "minmse_pof2",
    CalibMethod.Percentile: "percentile:99.999",
    CalibMethod.Entropy: "entropy",
    CalibMethod.LayerwisePercentile: "layerwise_percentile",
    CalibMethod.Distribution: "distribution",
}


# Quark's Align* options -> the op types whose parameters they align
_ALIGN_OPTIONS = (
    ("AlignConcat", ("Concat",)),
    ("AlignPool", ("MaxPool", "AveragePool", "GlobalAveragePool")),
    ("AlignPad", ("Pad",)),
    ("AlignSlice", ("Slice",)),
    ("AlignTranspose", ("Transpose",)),
    ("AlignReshape", ("Reshape",)),
)


def _without_batch_norm(
    model: onnx.ModelProto, op_types: "Optional[set[str]]"
) -> "Optional[set[str]]":
    """``op_types`` without ``BatchNormalization`` (every op type of the model
    when ``None``) if the model still has one: Quark quantizes BatchNorms only
    when it converts them (``ConvertBNToConv``), so one left over otherwise stays
    float between its quantized neighbours."""
    if not any(n.op_type == "BatchNormalization" for n in model.graph.node):
        return op_types
    types = (
        set(op_types) if op_types is not None else {n.op_type for n in model.graph.node}
    )
    return types - {"BatchNormalization"}


#: ops onnxsim's own flows share quantization parameters across but Quark's default
#: operator quantizer (``QDQOperatorBase``) calibrates input and output separately
_OWN_RANGE_OPS = (
    "Flatten",
    "Expand",
    "Tile",
    "Identity",
    "GridSample",
    "DepthToSpace",
    "SpaceToDepth",
)


def _activation_rules(
    opts: Dict[str, Any], symmetric: bool, extended: bool, npu_cnn: bool
) -> Dict[str, Any]:
    """:func:`onnxsim.full_qdq.quantize_full_qdq` keywords for Quark's
    Q/DQ-removal options: ``RemoveQDQConvRelu`` / ``ConvClip`` (default on),
    ``ConvLeakyRelu`` / ``ConvPRelu`` (on), ``ConvGelu`` (off) choose the
    consumers whose producer output stays float, ``RemoveQDQInstanceNorm``
    (off) adds InstanceNormalization to the producers. A Relu / Clip node is
    itself folded into its producer for asymmetric activations -- under the
    extended quantizer (see ``QConfig.quant_format``) only with ``FoldRelu``.
    The ``Align*`` options are only run by the extended quantizer."""
    from onnxsim.full_qdq import QUARK_QDQ_PRODUCERS

    after = [
        op
        for op, key, default in (
            ("Relu", "RemoveQDQConvRelu", True),
            ("Clip", "RemoveQDQConvClip", True),
            ("LeakyRelu", "RemoveQDQConvLeakyRelu", True),
            ("PRelu", "RemoveQDQConvPRelu", True),
            ("Gelu", "RemoveQDQConvGelu", False),
        )
        if opts.get(key, default)
    ]
    producers: "tuple[str, ...]" = QUARK_QDQ_PRODUCERS
    if opts.get("RemoveQDQInstanceNorm", False):
        producers = producers + ("InstanceNormalization",)
    return {
        "remove_qdq_after": after,
        "remove_qdq_producers": producers,
        "fold_activation": (not symmetric)
        and (bool(opts.get("FoldRelu", False)) if extended else True),
        "adjust_activation_ranges": True,
        # (a plain quantizer reaches a PRelu only when the op type was asked for --
        # QuantizeAllOpTypes / ExtraOpTypesToQuantize -- and then quantizes every
        # input of it, the slope among them)
        "quantize_prelu_slope": extended
        or npu_cnn
        or bool(opts.get("QuantizeAllOpTypes"))
        or "PRelu" in (opts.get("ExtraOpTypesToQuantize") or ()),
        "align_ops": [
            op
            for key, ops in _ALIGN_OPTIONS
            if extended and opts.get(key, False)
            for op in ops
        ],
        # outputs calibrated on their own: Slice (what ``ConvertSplitToSlice`` makes
        # of a Split with sizes, on by default for the extended quantizer too). A
        # Split that stays a Split -- sizes that are not an initializer, none given,
        # or ``ConvertSplitToSlice`` off -- shares its input's parameters (ONNX
        # Runtime's ``QDQSplit``), whichever quantizer
        # (and so are the ops its plain operator quantizer handles -- the default one,
        # which gives input and output their own ranges -- that onnxsim's own flows
        # would share: Flatten, Expand, Tile, Identity, ...)
        "unshared_ops": ("Slice",) + _OWN_RANGE_OPS,
        # ... and ONNX Runtime's plain quantizer gives AveragePool its input's
        "shared_ops": () if extended or npu_cnn else ("AveragePool",),
    }


def _calibration_args(
    method: str, opts: Dict[str, Any]
) -> "tuple[str, Dict[str, Any]]":
    """``(method, calibrate_options)`` for :func:`onnxsim.calibration.calibrate`
    from a spec's ``calibration_method`` string and Quark's calibration
    ``extra_options`` (``Percentile``, ``CalibTensorRangeSymmetric``,
    ``CalibMovingAverage``, ``NumBins``, ``NumQuantizedBins``, ``LWPMetric``,
    ``PercentileCandidates``). ``"entropy"`` / ``"percentile"`` /
    ``"distribution"`` / ``"layerwise_percentile"`` are Quark's algorithms;
    ``"onnxsim:<method>"`` selects onnxsim's own variant of the same name."""
    kw: Dict[str, Any] = {}
    if method.startswith("onnxsim:"):
        return method[len("onnxsim:") :], kw
    base, _, arg = method.partition(":")
    if base in ("entropy", "percentile", "distribution", "layerwise_percentile"):
        method = "quark_" + method
    elif base not in (
        "minmax",
        "minmax_mean",
        "mse",
        "minmse_pof2",
        "auto",
    ) and not base.startswith("quark_"):
        raise ValueError(f"unknown calibration method: {method!r}")
    if method.startswith("quark_percentile") and "Percentile" in opts:
        method = f"quark_percentile:{float(opts['Percentile'])}"
    if "CalibTensorRangeSymmetric" in opts:
        kw["range_symmetric"] = bool(opts["CalibTensorRangeSymmetric"])
    if "NumBins" in opts:
        kw["quark_num_bins"] = int(opts["NumBins"])
    if "NumQuantizedBins" in opts:
        kw["num_quantized_bins"] = int(opts["NumQuantizedBins"])
    if "LWPMetric" in opts:
        kw["lwp_metric"] = str(opts["LWPMetric"])
    if "PercentileCandidates" in opts:
        kw["percentile_candidates"] = tuple(opts["PercentileCandidates"])
    return method, kw


def _exact(options: Dict[str, Any]) -> Dict[str, Any]:
    """``options`` with the calibration session run unoptimized, as Quark's
    calibrators run it (ONNX Runtime's fusions change the order of float
    operations, enough to move a value across a histogram bin edge)."""
    return {"exact_session": True, **options}


def _spec(
    name: str,
    dtype: str,
    symmetric: bool,
    pof2: bool = False,
    calibration_method: str = "minmax",
):
    def __init__(self, **kwargs: Any) -> None:
        fields: Dict[str, Any] = dict(
            dtype=dtype,
            symmetric=symmetric,
            pof2=pof2,
            calibration_method=calibration_method,
        )
        fields.update(kwargs)  # a caller may override e.g. ``symmetric``
        QSpec.__init__(self, **fields)

    return type(name, (QSpec,), {"__init__": __init__, "__doc__": f"{name}."})


# Quark's defaults: the integer specs calibrate with a 99.999 percentile, the
# power-of-2 ones with MinMSE, everything else with MinMax.
_PCT_DEFAULT = "percentile:99.999"
Int8Spec = _spec("Int8Spec", "int8", True, calibration_method=_PCT_DEFAULT)
UInt8Spec = _spec("UInt8Spec", "uint8", False, calibration_method=_PCT_DEFAULT)
Int16Spec = _spec("Int16Spec", "int16", True, calibration_method=_PCT_DEFAULT)
UInt16Spec = _spec("UInt16Spec", "uint16", False, calibration_method=_PCT_DEFAULT)
XInt8Spec = _spec("XInt8Spec", "int8", True, True, "minmse_pof2")
XUInt8Spec = _spec("XUInt8Spec", "uint8", True, True, "minmse_pof2")
Float16Spec = _spec("Float16Spec", "float16", True)
BFloat16Spec = _spec("BFloat16Spec", "bfloat16", True)
BFP16Spec = _spec("BFP16Spec", "bfp16", True)
MX4Spec = _spec("MX4Spec", "mx4", True)
MX6Spec = _spec("MX6Spec", "mx6", True)
MX9Spec = _spec("MX9Spec", "mx9", True)

MXFP4E2M1Spec = _spec("MXFP4E2M1Spec", "mxfp4_e2m1", True)
MXFP6E3M2Spec = _spec("MXFP6E3M2Spec", "mxfp6_e3m2", True)
MXFP6E2M3Spec = _spec("MXFP6E2M3Spec", "mxfp6_e2m3", True)
MXFP8E5M2Spec = _spec("MXFP8E5M2Spec", "mxfp8_e5m2", True)
MXFP8E4M3Spec = _spec("MXFP8E4M3Spec", "mxfp8_e4m3", True)
MXInt8Spec = _spec("MXInt8Spec", "mxint8", True)


def _block_fn(dtype: str) -> Optional[Callable[[np.ndarray, int], np.ndarray]]:
    """Weight fake-quantizer ``f(array, axis)`` for a block-format / half dtype."""
    from onnxsim import quark_block_formats as bf

    if dtype == "bfp16":
        return lambda a, ax: bf.bfp16(a, axis=ax)
    if dtype in ("mx4", "mx6", "mx9"):
        bw = {"mx4": 11, "mx6": 13, "mx9": 16}[dtype]
        return lambda a, ax: bf.bfp_prime(a, bit_width=bw, axis=ax)
    if dtype == "mxint8":
        return lambda a, ax: bf.mx(a, element_dtype="int8", axis=ax)
    if dtype.startswith("mxfp"):
        elem = dtype.replace("mxfp", "fp", 1)  # mxfp8_e4m3 -> fp8_e4m3
        return lambda a, ax: bf.mx(a, element_dtype=elem, axis=ax)
    if dtype == "float16":
        return lambda a, ax: bf.fp16_round(a)
    if dtype == "bfloat16":
        return lambda a, ax: bf.bf16_round(a)
    return None


PROMOTABLE_OPS = ("Conv", "ConvTranspose", "Gemm", "MatMul")

# Quark's ``BFP_OP_DEFAULT_ATTRS`` / ``MX_OP_DEFAULT_ATTRS`` and the attributes its
# config mapping gives a same-format ``MX*`` pair (``DEFAULT_MICROEXPONENTS_PARAMS``
# / ``DEFAULT_MICROSCALING_PARAMS``)
_BFP_DEFAULTS: Dict[str, Any] = dict(
    bfp_method="to_bfp",
    axis=1,
    bit_width=16,
    block_size=8,
    rounding_mode=0,
    sub_block_size=2,
    sub_block_shift_bits=1,
    convert_to_bfloat_before_bfp=0,
)
_MX_DEFAULTS: Dict[str, Any] = dict(
    element_dtype="int8", axis=1, block_size=32, rounding_mode=0
)
_BFP_PRIME_PARAMS: Dict[str, Any] = dict(
    bfp_method="to_bfp_prime",
    axis=1,
    bit_width=13,
    block_size=16,
    sub_block_size=2,
    sub_block_shift_bits=1,
    rounding_mode=2,
)
_MX_PARAMS: Dict[str, Any] = dict(
    element_dtype="int8", axis=1, block_size=32, rounding_mode=2
)
_MX_BFP_BITS = {"mx4": 11, "mx6": 13, "mx9": 16}
_MX_ELEMENT = {
    "mxfp4_e2m1": "fp4_e2m1",
    "mxfp6_e3m2": "fp6_e3m2",
    "mxfp6_e2m3": "fp6_e2m3",
    "mxfp8_e5m2": "fp8_e5m2",
    "mxfp8_e4m3": "fp8_e4m3",
    "mxint8": "int8",
}
_BLOCK_DTYPES = {
    "bfp16",
    "mx4",
    "mx6",
    "mx9",
    "mxint8",
    "mxfp4_e2m1",
    "mxfp6_e3m2",
    "mxfp6_e2m3",
    "mxfp8_e5m2",
    "mxfp8_e4m3",
}
_FAKEQUANT_DTYPES = _BLOCK_DTYPES | {"float16", "bfloat16"}


@dataclass(eq=True, unsafe_hash=True)
class QLayerConfig:
    """Spec of one layer group (Quark's ``QLayerConfig``).

    ``activation`` is Quark's deprecated spelling of ``input_tensors`` (giving
    both raises, as in Quark). A spec left ``None`` means "inherit": the global
    config's, for a global config that is Int8. Positional order is
    ``(activation, weight)`` for backward compatibility.
    """

    activation: Optional[QSpec] = None
    weight: Optional[QSpec] = None
    input_tensors: Optional[QSpec] = None
    bias: Optional[QSpec] = None
    output_tensors: Optional[QSpec] = None

    def __post_init__(self) -> None:
        if self.input_tensors is not None and self.activation is not None:
            raise ValueError(
                "Both `activation` and `input_tensors` are provided. Please just "
                "use `input_tensors`."
            )
        #: AutoMixprecision applies ``activation`` to the inputs *and* outputs of
        #: a layer, but ``input_tensors`` to the inputs only (``output_tensors``
        #: to the outputs); the two spellings are merged below
        self._activation_spelled = self.activation is not None
        if self.activation is None:
            self.activation = self.input_tensors
        self.input_tensors = self.activation

    def resolved(self) -> "QLayerConfig":
        """The global form: a missing activation / weight spec is Int8."""
        return QLayerConfig(
            activation=self.activation or Int8Spec(),
            weight=self.weight or Int8Spec(),
            bias=self.bias,
            output_tensors=self.output_tensors,
        )


def _activation_inputs(node: onnx.NodeProto, inits: "set[str]") -> List[str]:
    """Inputs before the first constant, which Quark treats as activations."""
    out: List[str] = []
    for x in node.input:
        if not x or x in inits:
            break
        out.append(x)
    return out


def _matmul_add_consumers(model: onnx.ModelProto, converted: List[str]) -> List[str]:
    """Converted MatMul nodes (by their new ``*_Q4`` names) whose only consumer
    is an ``Add`` -- the pattern ONNX Runtime fuses into a Gemm."""
    adds = [n for n in model.graph.node if n.op_type == "Add"]
    out: List[str] = []
    for n in model.graph.node:
        if n.op_type != "MatMul" or (n.name + "_Q4" if n.name else "") not in converted:
            continue
        users = [m for m in model.graph.node if n.output[0] in m.input]
        if len(users) == 1 and users[0] in adds:
            out.append(n.name or n.output[0])
    return out


def _quantizable(
    model: onnx.ModelProto, op_types: "Optional[Iterable[str]]", exclude: Iterable[str]
) -> bool:
    """Quark's ``check_model_quantizable``: whether a node of an op type on the
    list (not excluded) reads or writes a float tensor that is not an initializer;
    when none does, Quark quantizes nothing and returns the model as given, before
    any pre-processing."""
    types = set(op_types) if op_types else None
    excluded = set(exclude)
    inits = {t.name for t in model.graph.initializer}
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    floats = {
        vi.name
        for vi in list(inferred.graph.value_info)
        + list(inferred.graph.output)
        + list(inferred.graph.input)
        if vi.type.HasField("tensor_type")
        and vi.type.tensor_type.elem_type
        in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)
    }
    for n in model.graph.node:
        if (
            (types is None or n.op_type in types)
            and n.name not in excluded
            and (not n.output or n.output[0] not in excluded)
        ):
            if any(
                x in floats and x not in inits for x in list(n.input) + list(n.output)
            ):
                return True
    return False


def _quantized_batch_norms(
    model: onnx.ModelProto,
    op_types: "Optional[Iterable[str]]",
    exclude: Iterable[str],
) -> "Set[str]":
    """Names of the ``BatchNormalization`` nodes whose op type is on Quark's list
    (``QuantizeAllOpTypes`` or an extra op type) and that are not excluded: its own
    BatchNorm folding passes skip a node it is asked to quantize."""
    if op_types is None or "BatchNormalization" not in set(op_types):
        return set()
    excluded = set(exclude)
    return {
        n.name
        for n in model.graph.node
        if n.op_type == "BatchNormalization"
        and n.name not in excluded
        and (not n.output or n.output[0] not in excluded)
    }


def _register_domains(model: onnx.ModelProto) -> onnx.ModelProto:
    """Quark registers the ``com.microsoft`` and ``com.amd.quark`` operator sets on
    every model it quantizes (``set_parameters_and_domain``, in that order, on the
    float model before the quantizer runs; a domain the model has keeps its place and
    its version, at least 1); the ``MATMUL_NBITS`` flow does not."""
    for domain in ("com.microsoft", "com.amd.quark"):
        for o in model.opset_import:
            if o.domain == domain:
                o.version = max(o.version, 1)
                break
        else:
            model.opset_import.append(onnx.helper.make_opsetid(domain, 1))
    return model


def _match_nodes(model: onnx.ModelProto, patterns: List[Any]) -> List[str]:
    """Node names selected by ``patterns``: plain names, and Quark's
    ``^...*``-style regular expressions (must contain ``.*``)."""
    names = [n.name for n in model.graph.node]
    out: List[str] = []
    for p in patterns:
        if isinstance(p, tuple):
            raise NotImplementedError("subgraph patterns are not supported")
        if not isinstance(p, str):
            raise TypeError(f"expected a node name or pattern, got {type(p).__name__}")
        if p.startswith("^"):
            if ".*" not in p:
                raise ValueError(
                    f"invalid pattern {p!r}: patterns start with ^ and contain .*"
                )
            hit = [n for n in names if n and re.search(p, n)]
            if not hit:
                raise ValueError(f"pattern {p!r} matches no node")
            out += hit
        else:
            out.append(p)
    return out


# -- algorithm configs (stored only, see module docstring) --------------------


@dataclass(eq=True)
class AlgoConfig:
    """Base class of the algorithm configs; ``name`` is the Quark algo name."""

    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


def _algo(name: str):
    def __init__(self, **params: Any) -> None:
        AlgoConfig.__init__(self, name=name, params=params)

    return type(
        name.title().replace("_", "") + "Config", (AlgoConfig,), {"__init__": __init__}
    )


SmoothQuantConfig = _algo("smooth_quant")
CLEConfig = _algo("cle")
BiasCorrectionConfig = _algo("bias_correction")
GPTQConfig = _algo("gptq")
AdaRoundConfig = _algo("adaround")
AdaQuantConfig = _algo("adaquant")
QuarotConfig = _algo("quarot")
AutoMixprecisionConfig = _algo("auto_mixprecision")


# -- QConfig and presets -------------------------------------------------------


class QConfig:
    """Mirror of ``quark.onnx.QConfig`` (global spec, per-layer overrides,
    excluded nodes, algorithms, extra options)."""

    def __init__(
        self,
        global_config: QLayerConfig,
        specific_layer_config: Optional[Dict[Any, List[str]]] = None,
        layer_type_config: Optional[Dict[Any, List[str]]] = None,
        exclude: Optional[List[Any]] = None,
        algo_config: Optional[List[AlgoConfig]] = None,
        use_external_data_format: bool = False,
        quant_format: Optional[str] = None,
        **extra_options: Any,
    ) -> None:
        #: ``"extended"`` for Quark's ExtendedQuantFormat quantizer (its A8W8 /
        #: 16-bit presets), ``"qdq"`` for the plain one; ``None`` derives it from
        #: the dtypes like Quark's own QConfig mapping (8-bit both -> plain)
        self.quant_format = quant_format
        self.global_config = global_config
        self.specific_layer_config = specific_layer_config or {}
        self.layer_type_config = layer_type_config or {}
        self.exclude = exclude or []
        self.algo_config = algo_config or []
        self.use_external_data_format = use_external_data_format
        # Quark passes these as ``extra_options={...}``; accept both spellings.
        self.extra_options = dict(
            extra_options.pop("extra_options", {}), **extra_options
        )

    @staticmethod
    def get_default_config(config_name: str) -> "QConfig":
        """Preset by Quark name (``"A8W8"``, ``"XINT8"``, ``"BF16"``, ...)."""
        try:
            return _PRESETS[config_name]()
        except KeyError:
            raise ValueError(
                f"unknown preset {config_name!r}; known: {sorted(_PRESETS)}"
            ) from None


def _layer(act: type, wt: type, **act_kwargs: Any) -> QLayerConfig:
    return QLayerConfig(activation=act(**act_kwargs), weight=wt())


# Quark calibrates the asymmetric ("AA") presets with percentiles (99.999;
# S8S8_AAWS 99.9999), S16S8_ASWS with a symmetric 99.999 percentile.
_PCT = dict(symmetric=False, calibration_method="percentile:99.999")
_PCT4 = dict(symmetric=False, calibration_method="percentile:99.9999")


# extra_options Quark's A8W8 / A16W8 presets carry (the ones onnxsim reads)
_A8_EXTRAS: Dict[str, Any] = dict(
    ActivationSymmetric=True, FoldRelu=True, AlignConcat=True, AlignSlice=False
)

_PRESETS: Dict[str, Callable[[], QConfig]] = {
    "XINT8": lambda: QConfig(_layer(XUInt8Spec, XInt8Spec)),
    "UINT8_DYNAMIC_QUANT": lambda: QConfig(
        _layer(Int8Spec, UInt8Spec, is_dynamic=True)
    ),
    "A8W8": lambda: QConfig(
        _layer(Int8Spec, Int8Spec, calibration_method="minmax"),
        quant_format="extended",
        **_A8_EXTRAS,
    ),
    "S8S8_AAWS": lambda: QConfig(_layer(Int8Spec, Int8Spec, **_PCT4)),
    "U8S8_AAWS": lambda: QConfig(_layer(UInt8Spec, Int8Spec, **_PCT)),
    "U8U8_AAWA": lambda: QConfig(_layer(UInt8Spec, UInt8Spec, **_PCT)),
    "A16W8": lambda: QConfig(
        _layer(Int16Spec, Int8Spec, calibration_method="minmax"),
        AlignEltwiseQuantType=True,
        **_A8_EXTRAS,
    ),
    "S16S8_ASWS": lambda: QConfig(
        _layer(Int16Spec, Int8Spec), ActivationSymmetric=True
    ),
    "MATMUL_NBITS": lambda: QConfig(
        _layer(Int8Spec, Int8Spec, calibration_method="minmax"),
        UseMatMulNBits=True,
        MatMulNBitsParams={
            "GroupSize": 128,
            "Symmetric": True,
            "Bits": 4,
            "AccuracyLevel": 1,
        },
    ),
    "U16S8_AAWS": lambda: QConfig(_layer(UInt16Spec, Int8Spec, **_PCT)),
    "FP16": lambda: QConfig(_layer(Float16Spec, Float16Spec)),
    "BF16": lambda: QConfig(_layer(BFloat16Spec, BFloat16Spec)),
    "BFP16": lambda: QConfig(
        _layer(BFP16Spec, BFP16Spec),
        BFPAttributes=dict(
            bfp_method="to_bfp", axis=1, bit_width=16, block_size=8, rounding_mode=2
        ),
    ),
    "MX4": lambda: QConfig(_layer(MX4Spec, MX4Spec)),
    "MX6": lambda: QConfig(_layer(MX6Spec, MX6Spec)),
    "MX9": lambda: QConfig(_layer(MX9Spec, MX9Spec)),
    "MXFP4E2M1": lambda: QConfig(_layer(MXFP4E2M1Spec, MXFP4E2M1Spec)),
    "MXFP6E3M2": lambda: QConfig(_layer(MXFP6E3M2Spec, MXFP6E3M2Spec)),
    "MXFP6E2M3": lambda: QConfig(_layer(MXFP6E2M3Spec, MXFP6E2M3Spec)),
    "MXFP8E5M2": lambda: QConfig(_layer(MXFP8E5M2Spec, MXFP8E5M2Spec)),
    "MXFP8E4M3": lambda: QConfig(_layer(MXFP8E4M3Spec, MXFP8E4M3Spec)),
    "MXINT8": lambda: QConfig(_layer(MXInt8Spec, MXInt8Spec)),
}


def _preset_finetune_params(cls: type) -> Dict[str, Any]:
    """The ``FastFinetune`` options Quark's ``*_ADAROUND`` / ``*_ADAQUANT``
    presets carry. Their dict has no ``UpdateBias`` key, so Quark's training
    default (on) applies to AdaQuant, unlike ``AdaQuantConfig()`` (off)."""
    if cls is AdaQuantConfig:
        return dict(_FASTFT_PRESET, learning_rate=1e-5, update_bias=True)
    if cls is AdaRoundConfig:
        return dict(_FASTFT_PRESET, learning_rate=0.1)
    return {}


def _algo_variant(base: str, cls: type) -> Callable[[], QConfig]:
    def make() -> QConfig:
        return _with_algo(_PRESETS[base](), cls(**_preset_finetune_params(cls)))

    return make


for _n in list(_PRESETS):
    _block = _n.startswith(("BFP", "MX"))
    if _n in ("BF16", "UINT8_DYNAMIC_QUANT", "MATMUL_NBITS") or _n.startswith("FP16"):
        continue
    if not _block:
        _PRESETS[f"{_n}_ADAROUND"] = _algo_variant(_n, AdaRoundConfig)
    _PRESETS[f"{_n}_ADAQUANT"] = _algo_variant(_n, AdaQuantConfig)


# Quark's mixed-format presets (no ADAROUND variants -- and, apart from the
# BF16_MIXED_* ones registered below, no ADAQUANT ones -- exist in Quark):
# bfloat16 activations over block-format constants, block-format activations
# over int8 constants.
def _cnn_accurate(act: type, wt: type) -> QConfig:
    return QConfig(
        _layer(act, wt, symmetric=False, calibration_method="percentile:99.9999"),
        algo_config=[AdaRoundConfig(**_preset_finetune_params(AdaRoundConfig))],
    )


def _transformer(act: type, wt: type, accurate: bool = False) -> QConfig:
    """Quark's ``INT{8,16}_TRANSFORMER_{DEFAULT,ACCURATE}``: asymmetric
    uint8 / uint16 activations, per-tensor symmetric int8 / int16 weights,
    quantization restricted to Gemm / weight MatMul (``NPUTransformer``)."""
    if accurate:
        cfg = _cnn_accurate(act, wt)
    else:
        cfg = QConfig(
            _layer(act, wt, symmetric=False, calibration_method="minmax_mean")
        )
    cfg.extra_options["NPUTransformer"] = True
    return cfg


def _s16s16_mixed_s8s8() -> QConfig:
    """int16 activations / weights (asymmetric, percentile 99.9999) with
    biases quantized like weights (``Int32Bias=False``: int16), every Conv /
    Gemm / MatMul promoted to int8 inputs, weights and biases (Quark:
    AutoMixprecision, threshold disabled -- its ``input_tensors`` target leaves
    the outputs int16)."""
    return QConfig(
        _layer(
            Int16Spec,
            Int16Spec,
            symmetric=False,
            calibration_method="percentile:99.9999",
        ),
        algo_config=[
            AutoMixprecisionConfig(
                # (Quark's preset spells Int8Spec() throughout: the activation's
                # asymmetry comes from the global spec, via ActivationSymmetric)
                target_layer_config=QLayerConfig(
                    input_tensors=Int8Spec(),
                    weight=Int8Spec(),
                    bias=Int8Spec(),
                ),
                metric_threshold=0,
                data_size=1000,  # the preset's own (extra_options) default
            )
        ],
        Int32Bias=False,
    )


def _mixed_block(block: Callable[[], QSpec], *algos: AlgoConfig) -> QConfig:
    """bfloat16 everywhere, every Conv / Gemm / MatMul promoted to ``block``
    (Quark: AutoMixprecision with the metric threshold disabled, dual nodes at
    the boundaries, biases left unquantized)."""
    target = QLayerConfig(input_tensors=block(), weight=block(), bias=block())
    return QConfig(
        _layer(BFloat16Spec, BFloat16Spec),
        algo_config=[
            AutoMixprecisionConfig(
                target_layer_config=target,
                dual_quant_nodes=True,
                metric_threshold=0,
                data_size=1000,  # the preset's own (extra_options) default
            ),
            *algos,
        ],
        QuantizeBias=False,
        DedicateDQNode=True,
    )


_PRESETS.update(
    {
        "BF16_MIXED_BFP16": lambda: _mixed_block(BFP16Spec),
        "BF16_MIXED_MXINT8": lambda: _mixed_block(MXInt8Spec),
        "BF16_MIXED_BFP16_ADAQUANT": lambda: _mixed_block(BFP16Spec, AdaQuantConfig()),
        "BF16_MIXED_MXINT8_ADAQUANT": lambda: _mixed_block(
            MXInt8Spec, AdaQuantConfig()
        ),
        # (their block attributes are the presets' own: Quark's plain default
        # nodes use ``rounding_mode`` 0, see ``_block_attr_overrides``)
        "BF16_BFP16": lambda: QConfig(
            _layer(BFloat16Spec, BFP16Spec),
            BFPAttributes=dict(
                bfp_method="to_bfp", axis=1, bit_width=16, block_size=8, rounding_mode=2
            ),
        ),
        "BF16_MXINT8": lambda: QConfig(
            _layer(BFloat16Spec, MXInt8Spec),
            MXAttributes=dict(
                element_dtype="int8", axis=1, block_size=32, rounding_mode=2
            ),
        ),
        "MX9_INT8": lambda: QConfig(
            _layer(MX9Spec, Int8Spec),
            BFPAttributes=dict(_BFP_PRIME_PARAMS, bit_width=16),
        ),
        # The "amateur" CNN presets: asymmetric uint8 / uint16 activations,
        # per-tensor symmetric int8 / int16 weights; ACCURATE adds percentile
        # 99.9999 calibration and AdaRound (Quark's FastFinetune defaults).
        # VINT8: signed power-of-2 int8 everywhere, every op type quantized, no
        # Relu folding, int8 biases, one Q/DQ pair per consumer (the VAIML
        # deployment flavour of XINT8; Quark has no ADAROUND/ADAQUANT variant).
        "VINT8": lambda: QConfig(
            _layer(XInt8Spec, XInt8Spec),
            OptimizeModel=False,
            EnableNPUCnn=False,
            ConvertBNToConv=True,
            ConvertClipToRelu=True,
            ConvertSplitToSlice=True,
            SplitLargeKernelPool=False,
            ReplaceClip6Relu=True,
            ConvertReduceMeanToGlobalAvgPool=False,
            RemoveQDQConvClip=False,
            RemoveQDQConvPRelu=False,
            RemoveQDQConvRelu=False,
            RemoveQDQConvLeakyRelu=False,
            Int32Bias=False,
            DedicatedQDQPair=True,
            QuantizeAllOpTypes=True,
            # (VINT8 is the one Quark preset that leaves it off: a LayerNormalization
            # or data-movement op fed by a graph input stays float)
            ForceQuantizeNoInputCheck=False,
        ),
        "S16S16_MIXED_S8S8": lambda: _s16s16_mixed_s8s8(),
        "INT8_CNN_DEFAULT": lambda: QConfig(
            _layer(UInt8Spec, Int8Spec, symmetric=False, calibration_method="minmax")
        ),
        "INT16_CNN_DEFAULT": lambda: QConfig(
            _layer(UInt16Spec, Int16Spec, symmetric=False, calibration_method="minmax")
        ),
        "INT8_CNN_ACCURATE": lambda: _cnn_accurate(UInt8Spec, Int8Spec),
        "INT16_CNN_ACCURATE": lambda: _cnn_accurate(UInt16Spec, Int16Spec),
        # Quark's NPU transformer presets (``enable_npu_transformer``): only
        # Gemm and weight-carrying MatMul nodes are quantized (see
        # ``_transformer_scope``); DEFAULT calibrates with the mean of the
        # per-batch min / max (``CalibMovingAverage``), ACCURATE with the
        # CNN-ACCURATE percentile + AdaRound recipe.
        "INT8_TRANSFORMER_DEFAULT": lambda: _transformer(UInt8Spec, Int8Spec),
        "INT16_TRANSFORMER_DEFAULT": lambda: _transformer(UInt16Spec, Int16Spec),
        "INT8_TRANSFORMER_ACCURATE": lambda: _transformer(
            UInt8Spec, Int8Spec, accurate=True
        ),
        "INT16_TRANSFORMER_ACCURATE": lambda: _transformer(
            UInt16Spec, Int16Spec, accurate=True
        ),
    }
)


# FP16 / BF16 AdaQuant: Quark emits the FP16 / BF16 graph with tuned
# weights / biases. The graph half is reproduced; the tuning is not (like
# BFP16_ADAQUANT, quantize_model raises unless ignore_unsupported_algos=True).
_PRESETS["FP16_ADAQUANT"] = _algo_variant("FP16", AdaQuantConfig)
_PRESETS["BF16_ADAQUANT"] = _algo_variant("BF16", AdaQuantConfig)


def _with_algo(cfg: QConfig, algo: AlgoConfig) -> QConfig:
    cfg.algo_config = [algo]
    return cfg


class Config:
    """Mirror of ``quark.onnx.Config`` (wraps a global config)."""

    def __init__(self, global_quant_config: QConfig) -> None:
        self.global_quant_config = global_quant_config


# -- calibration-reader adapter ------------------------------------------------


def _drain_reader(
    reader: Any, limit: Optional[int] = None
) -> List[Dict[str, np.ndarray]]:
    """Materialize an onnxruntime-style ``CalibrationDataReader`` (anything
    with ``get_next() -> dict | None``), or pass through a list of dicts."""
    if reader is None:
        return []
    if isinstance(reader, (list, tuple)):
        return [dict(b) for b in reader]
    batches: List[Dict[str, np.ndarray]] = []
    while limit is None or len(batches) < limit:
        batch = reader.get_next()
        if batch is None:
            break
        batches.append({k: np.asarray(v) for k, v in batch.items()})
    return batches


_RUNNABLE_ALGOS = {
    "quarot",
    "smooth_quant",
    "cle",
    "adaquant",
    "adaround",
    "gptq",
    "bias_correction",
    "auto_mixprecision",
}
# Quark AdaQuant param -> onnxsim.apply_adaquant kwarg (``legacy_engine=True``).
_ADAQUANT_PARAMS = {
    "num_iterations": "num_iterations",
    "learning_rate": "weight_learning_rate",
    "reg_param": "reg_param",
}
# ``extra_options["FastFinetune"]`` keys (they win over the algo config's
# values, as in Quark) -> AdaRoundConfig / AdaQuantConfig params.
_FASTFT_KEYS = {
    "DataSize": "data_size",
    "FixedSeed": "fixed_seed",
    "BatchSize": "batch_size",
    "NumBatches": "num_batches",
    "NumIterations": "num_iterations",
    "LearningRate": "learning_rate",
    "EarlyStop": "early_stop",
    "OutputIndex": "output_index",
    "LRAdjust": "lr_adjust",
    "SelectiveUpdate": "selective_update",
    "UpdateBias": "update_bias",
    "OutputQDQ": "output_qdq",
    "DropRatio": "drop_ratio",
    "MemOptLevel": "mem_opt_level",
    "NumWorkers": "num_workers",
    "DynamicBatch": "dynamic_batch",
    "Parallel": "parallel",
    "RegParam": "reg_param",
    "BetaRange": "beta_range",
    "WarmStart": "warm_start",
    "SelectMaxMemLayer": "select_max_mem_layer",
    "TargetOpType": "target_op_type",
    "RefModelPath": "ref_model_path",
}
# AdaRoundConfig / AdaQuantConfig fields that Quark 0.13's ``_get_config`` never
# copies into ``extra_options["FastFinetune"]`` (it stores them on the config and
# stops there), so there they only take effect through extra_options.
_FASTFT_NOT_FORWARDED = (
    "output_index",
    "reg_param",
    "beta_range",
    "warm_start",
    "parallel",
    "dynamic_batch",
    "ref_model_path",
)
# What Quark's ``*_ADAROUND`` / ``*_ADAQUANT`` presets put in
# ``extra_options["FastFinetune"]`` (read from the amd-quark 0.13 wheel).
_FASTFT_PRESET = {
    "data_size": 1000,
    "fixed_seed": 1705472343,
    "batch_size": 2,
    "num_iterations": 1000,
    "early_stop": True,
}


# -- quantizer -----------------------------------------------------------------


def _default_opset(model: onnx.ModelProto) -> int:
    """The model's default-domain opset (0 when it has none)."""
    return next(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0
    )


def _qdq_to_ms_domain(model: onnx.ModelProto) -> onnx.ModelProto:
    """Quark's extended quantizer (its ``A8W8`` and 16-bit presets) emits custom
    Q/DQ nodes and ends by converting them to the ``com.microsoft`` ones
    (``_convert_qdq_nodes``): every ``QuantizeLinear`` / ``DequantizeLinear`` whose
    zero point is an int8 / uint8 / int16 / uint16 / int32 initializer, except an
    int32 ``QuantizeLinear`` and an int32 ``DequantizeLinear`` with a non-zero
    zero point (which the ``com.microsoft`` ops do not support). Works in place;
    the ``com.microsoft`` operator set is registered by the caller."""
    inits = {t.name: t for t in model.graph.initializer}
    types = (
        onnx.TensorProto.INT8,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.INT16,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.INT32,
    )
    for n in model.graph.node:
        if n.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if n.domain not in ("", "ai.onnx"):
            continue
        zp = inits.get(n.input[2]) if len(n.input) > 2 else None
        if zp is None or zp.data_type not in types:
            continue
        if zp.data_type == onnx.TensorProto.INT32:
            if n.op_type == "QuantizeLinear":
                continue
            if np.count_nonzero(onnx.numpy_helper.to_array(zp)) != 0:
                continue
        n.domain = "com.microsoft"
    return model


class ModelQuantizer:
    """Mirror of ``quark.onnx.ModelQuantizer``.

    ``last_approximations`` lists every place the requested spec was mapped
    to the nearest thing onnxsim can actually do (empty when exact).
    """

    def __init__(self, config: Union[QConfig, Config]) -> None:
        if isinstance(config, Config):
            config = config.global_quant_config
        if not isinstance(config, QConfig):
            raise TypeError(f"expected QConfig or Config, got {type(config).__name__}")
        self.config = config
        self.last_approximations: List[str] = []
        self._overrides_applied = False
        #: the :class:`~onnxsim.quark_auto_mixprecision.AutoMixprecisionResult`
        #: (sensitivity ranking, moved layers, scores) of the last run, if any
        self.last_auto_mixprecision: Any = None
        #: ``{"adaround" | "gptq": [LayerReport, ...]}`` of the last run
        self.last_weight_rounding: Dict[str, Any] = {}
        #: :class:`~onnxsim.quark_matmul_nbits.MatMulNBitsReport` of the last
        #: ``MATMUL_NBITS`` run, if any
        self.last_matmul_nbits: Any = None

    def quantize_model(
        self,
        model_input: Union[str, onnx.ModelProto],
        model_output: Optional[str] = None,
        calibration_data_reader: Any = None,
        ignore_unsupported_algos: bool = False,
    ) -> onnx.ModelProto:
        """Quantize and (if ``model_output`` is given) save. Returns the model."""
        cfg = self.config
        self.last_approximations = []
        self._overrides_applied = False
        self.last_auto_mixprecision = None
        self.last_weight_rounding = {}
        cfg.global_config = cfg.global_config.resolved()
        act, wt = cfg.global_config.activation, cfg.global_config.weight
        assert act is not None and wt is not None  # resolved() fills both

        if cfg.extra_options.get("UseMatMulNBits"):  # the MATMUL_NBITS preset
            result = self._quantize_matmul_nbits(
                model_input, calibration_data_reader, ignore_unsupported_algos
            )
            for msg in self.last_approximations:
                warnings.warn(f"onnxsim.quark_compat: {msg}", UserWarning, stacklevel=2)
            if model_output:
                onnx.save(result, model_output)
            return result

        if wt.is_dynamic:
            raise NotImplementedError("dynamic weight quantization is not supported")
        # Quark's GPTQ re-grids the float weights to 8 bits whatever the preset's
        # weight dtype (probed: it never raises for int16 / uint8 weights), so
        # it runs for every integer weight dtype; AdaRound / AdaQuant (its
        # FastFinetune) take any integer weight grid
        can_run = _RUNNABLE_ALGOS
        unsupported = [a.name for a in cfg.algo_config if a.name not in can_run]
        if unsupported and not ignore_unsupported_algos:
            raise NotImplementedError(
                f"algo_config [{', '.join(unsupported)}] is not executed by "
                "onnxsim.quark_compat; pass ignore_unsupported_algos=True to "
                "quantize without them"
            )
        runnable = [a for a in cfg.algo_config if a.name in can_run]

        if isinstance(model_input, str):
            model_input = onnx.load(model_input)

        # Quark's pre-processing converts the opset first when asked to
        # (``ConvertOpsetVersion``; it warns and skips when the converter fails).
        # Without it a model is quantized in place, whatever its opset.
        # (The block-format / bfloat16 / float16 flow converts inside its own
        # pre-processing, after Quark's check that anything is quantizable: a model
        # it returns as given keeps its opset.)
        fake = wt.dtype in _FAKEQUANT_DTYPES or act.dtype in _FAKEQUANT_DTYPES
        block_flow = (
            fake
            and not (
                act.dtype in ("float16", "bfloat16")
                and cfg.extra_options.get("ConvertToHalf")
            )
            and self._mixed_block_target(act) is None
            and not any(a.name == "auto_mixprecision" for a in cfg.algo_config)
            and not act.is_dynamic
        )
        target = cfg.extra_options.get("ConvertOpsetVersion")
        if (
            isinstance(target, int)
            and not block_flow
            and not cfg.extra_options.get("SkipPreprocess", False)
        ):
            from onnxsim.quark_tools import convert_opset_version

            try:
                model_input = convert_opset_version(model_input, target)
            except ValueError as e:
                warnings.warn(
                    f"onnxsim.quark_compat: failed to convert the opset version "
                    f"({e}), skipping the conversion",
                    UserWarning,
                    stacklevel=2,
                )

        half = act.dtype in ("float16", "bfloat16")
        if half and cfg.extra_options.get("ConvertToHalf"):
            if runnable and not ignore_unsupported_algos:
                raise NotImplementedError(
                    "algo_config is not applied to float presets "
                    f"({act.dtype}); pass ignore_unsupported_algos=True to "
                    "convert without it"
                )
            from onnxsim.onnx_simplifier import quantize_bf16, quantize_fp16

            fn = quantize_fp16 if act.dtype == "float16" else quantize_bf16
            result = fn(model_input)
        elif self._mixed_block_target(act) is not None:
            result = self._quantize_mixed_block(
                model_input, act, ignore_unsupported_algos, calibration_data_reader
            )
        elif (wt.dtype in _FAKEQUANT_DTYPES or act.dtype in _FAKEQUANT_DTYPES) and any(
            a.name == "auto_mixprecision" for a in cfg.algo_config
        ):
            result = self._amp_on_fakequant_base(
                model_input, act, wt, ignore_unsupported_algos, calibration_data_reader
            )
        elif act.is_dynamic:
            result = self._quantize_dynamic(model_input, act, wt)
        elif wt.dtype in _FAKEQUANT_DTYPES or act.dtype in _FAKEQUANT_DTYPES:
            if cfg.algo_config and not ignore_unsupported_algos:
                what = "float presets" if half else "block formats"
                raise NotImplementedError(
                    f"algo_config is not applied to {what} "
                    f"({act.dtype}/{wt.dtype}); pass ignore_unsupported_algos=True "
                    "to quantize without it"
                )
            result = self._quantize_block(model_input, act, wt)
        else:
            result = self._quantize_int(
                model_input, act, wt, calibration_data_reader, runnable
            )

        if (
            cfg.specific_layer_config or cfg.layer_type_config
        ) and not self._overrides_applied:
            self._approx(
                "per-layer / per-type overrides ignored for "
                f"{act.dtype}/{wt.dtype} (global spec used)"
            )
        for msg in self.last_approximations:
            warnings.warn(f"onnxsim.quark_compat: {msg}", UserWarning, stacklevel=2)
        if model_output:
            onnx.save(result, model_output)
        return result

    def _quantize_matmul_nbits(
        self,
        model: Union[str, onnx.ModelProto],
        reader: Any,
        ignore_unsupported: bool,
    ) -> onnx.ModelProto:
        """``MATMUL_NBITS``: constant-weight MatMuls -> ``MatMulNBits`` (see
        :mod:`onnxsim.quark_matmul_nbits`). Options as Quark reads them:
        ``MatMulNBitsParams`` (``GroupSize``, ``Symmetric``, ``Bits``,
        ``AccuracyLevel``, ``Algorithm`` = DEFAULT / HQQ / GPTQ) and, for GPTQ,
        ``GPTQParams``; a ``GPTQConfig`` in ``algo_config`` is ignored, as in
        Quark."""
        from onnxsim.quark_matmul_nbits import quantize_matmul_nbits

        cfg = self.config
        opts = cfg.extra_options
        if isinstance(model, str):
            model = onnx.load(model)
        mm = dict(opts.get("MatMulNBitsParams", {}))
        gptq = dict(opts.get("GPTQParams", {}))
        others = [a.name for a in cfg.algo_config if a.name != "gptq"]
        if others and not ignore_unsupported:
            raise NotImplementedError(
                f"algo_config [{', '.join(others)}] is not applied to MatMulNBits; "
                "pass ignore_unsupported_algos=True to quantize without it"
            )
        if any(a.name == "gptq" for a in cfg.algo_config):
            self._approx(
                "GPTQConfig is not read by MATMUL_NBITS (as in Quark): set "
                "extra_options['GPTQParams'] and MatMulNBitsParams['Algorithm']"
            )
        algorithm = str(mm.get("Algorithm", "DEFAULT"))
        calibration = None
        if algorithm.upper() == "GPTQ":
            calibration = _drain_reader(reader, limit=1)  # Quark: first batch only
            if not calibration:
                raise ValueError("calibration_data_reader is required for GPTQ")
            if gptq.get("Compensate"):
                self._approx(
                    "GPTQ propagates the rounding error (Quark 0.13's propagation "
                    "is a no-op, so its codes are round-to-nearest)"
                )
        exclude = _match_nodes(
            model, [e for e in cfg.exclude if isinstance(e, (str, tuple))]
        )
        # Quark's pre-processing runs before the 4-bit conversion (BatchNorm folding,
        # Identity removal, MatMul + Add -> Gemm, the operator fusions, ...)
        op_types = self._static_op_types(
            model, extended=False, quantize_all=bool(opts.get("QuantizeAllOpTypes"))
        )
        if not _quantizable(model, op_types, exclude):
            # Quark: "No quantizable ops in this model" -- returned as given
            return model
        model = self._preprocess_block_flow(
            model, exclude, False, op_types, copy_bias=True
        )
        result, report = quantize_matmul_nbits(
            model,
            group_size=int(mm.get("GroupSize", 128)),
            symmetric=bool(mm.get("Symmetric", True)),
            bits=int(mm.get("Bits", 4)),
            accuracy_level=mm.get("AccuracyLevel", 0),
            algorithm=algorithm,
            exclude_nodes=exclude,
            gptq_params=gptq,
            calibration=calibration,
        )
        # (Quark's quantizers end with ``clean_initializers``)
        from onnxsim.quark_fakequant_graph import drop_unused_initializers

        drop_unused_initializers(result)
        self.last_matmul_nbits = report
        if cfg.specific_layer_config or cfg.layer_type_config:
            self._approx("per-layer / per-type overrides ignored for MatMulNBits")
        if not opts.get("SkipPreprocess", False):
            # (ONNX Runtime's optimizer is what fuses MatMul + Add into a Gemm, which
            # Quark does not quantize; without it here the MatMul stays and is converted)
            fused = _matmul_add_consumers(model, report.converted)
            if fused:
                self._approx(
                    "Quark's pre-processing fuses MatMul + Add into a Gemm, which it "
                    f"does not quantize; converted anyway: {', '.join(fused)}"
                )
        return result

    def _mixed_block_target(self, act: QSpec) -> Optional[AlgoConfig]:
        """The ``AutoMixprecisionConfig`` of a bfloat16 model whose target
        layers use a block format (``BF16_MIXED_BFP16`` / ``_MXINT8``), if any."""
        if act.dtype != "bfloat16":
            return None
        if self.config.extra_options.get("QuantizeBias", True) is not False:
            return None  # the presets leave biases alone; see _amp_on_fakequant_base
        for a in self.config.algo_config:
            t = a.params.get("target_layer_config")
            if (
                a.name == "auto_mixprecision"
                and isinstance(t, QLayerConfig)
                and t.weight is not None
                and t.weight.dtype in _BLOCK_DTYPES
            ):
                return a
        return None

    def _quantize_mixed_block(
        self,
        model: onnx.ModelProto,
        act: QSpec,
        ignore_unsupported: bool,
        reader: Any = None,
    ) -> onnx.ModelProto:
        result = self._mix_block_formats(model, act, ignore_unsupported, reader)
        if self.config.extra_options.get("DedicateDQNode"):
            from onnxsim.quark_preset_graphs import dedicate_dq_nodes

            result = dedicate_dq_nodes(result)
        return result

    def _mix_block_formats(
        self,
        model: onnx.ModelProto,
        act: QSpec,
        ignore_unsupported: bool,
        reader: Any = None,
    ) -> onnx.ModelProto:
        from onnxsim.quark_preset_graphs import apply_mixed_block_format

        algo = self._mixed_block_target(act)
        assert algo is not None
        p = algo.params
        target = p["target_layer_config"]
        others = [a.name for a in self.config.algo_config if a is not algo]
        if others and not ignore_unsupported:
            raise NotImplementedError(
                f"algo_config [{', '.join(others)}] is not applied to block formats; "
                "pass ignore_unsupported_algos=True to quantize without it"
            )
        exclude = [e for e in self.config.exclude if isinstance(e, str)]
        ops = tuple(p.get("target_op_type") or PROMOTABLE_OPS)
        if p.get("shared_param_mode", "propagate") not in ("propagate", "unshare"):
            raise ValueError("shared_param_mode must be 'propagate' or 'unshare'")
        sg_path = p.get("subgraph_json")
        subgraphs = None
        if sg_path is not None:
            if Path(sg_path).exists():
                from onnxsim.quark_auto_mixprecision import parse_subgraph_json

                specs = parse_subgraph_json(sg_path, model, model)
                subgraphs = [(sg.name, sg.resolved_nodes) for sg in specs]
            else:  # Quark checks ``Path(subgraph_json).exists()`` and moves on
                self._approx(
                    f"subgraph_json {sg_path!r} does not exist: ignored, as Quark does"
                )
        threshold = p.get("metric_threshold", 0)
        cache = p.get("sensitivity_cache_file")
        # The candidates only need scoring when something depends on the
        # scores: a cache to write / read, a threshold, or an analysis-only run.
        # (subgraph_json alone only regroups candidates that all move anyway.)
        if cache is None and threshold == 0:
            self._approx(
                f"{target.weight.dtype} layers: every candidate is promoted (Quark's "
                "metric_threshold=0), no sensitivity ranking; the model uses "
                "com.amd.quark custom ops that onnxsim cannot execute"
            )
            skipped: List[str] = []
            if p.get("no_input_qdq_shared"):
                from onnxsim.quark_auto_mixprecision import _shared_inputs
                from onnxsim.quark_preset_graphs import _with_node_names

                model = _with_node_names(model)
                skipped = sorted(_shared_inputs(model))
            return apply_mixed_block_format(
                model,
                target.weight.dtype,
                exclude=exclude,
                target_ops=ops,
                include_layers=p.get("include_layers") or (),
                exclude_layers=[*(p.get("exclude_layers") or ()), *skipped],
                dual_nodes=bool(p.get("dual_quant_nodes", False)),
            )
        from onnxsim.quark_auto_mixprecision import auto_mixprecision_blocks

        calibration = _drain_reader(reader)
        if not calibration:
            raise ValueError(
                "calibration_data_reader is required to score the AutoMixprecision "
                "candidates (sensitivity_cache_file / metric_threshold)"
            )
        self._approx(
            f"{target.weight.dtype} layers are scored by running the fake-quantized "
            "models on the ONNX reference evaluator (the com.amd.quark custom ops "
            "need Quark's operator library for ONNX Runtime): scores can differ "
            "from Quark's by float32 accumulation noise"
        )
        res = auto_mixprecision_blocks(
            model,
            calibration,
            target.weight.dtype,
            exclude_nodes=exclude,
            target_op_types=ops,
            include_layers=p.get("include_layers") or (),
            exclude_layers=p.get("exclude_layers") or (),
            metric=p.get("metric_default", "l2"),
            metric_distance_fn=p.get("metric_distance_fn"),
            metric_evaluate_fn=p.get("metric_evaluate_fn"),
            metric_threshold=threshold,
            optimize=p.get("metric_optimize_object", "speed"),
            metric_output_index=p.get("metric_output_index", 0),
            data_size=int(p.get("data_size", 0)) + 1,
            subgraphs=subgraphs,
            cache_file=cache,
            worker_num=p.get("worker_num", 1),
            no_input_qdq_shared=bool(p.get("no_input_qdq_shared", False)),
            dual_quant_nodes=bool(p.get("dual_quant_nodes", False)),
            cache_key_fn=self._amp_cache_key_fn(p),
        )
        self.last_auto_mixprecision = res
        return res.model

    def _amp_cache_key_fn(
        self, p: Dict[str, Any]
    ) -> "Callable[[onnx.ModelProto], str]":
        """``f(baseline) -> str``: Quark's sensitivity-cache key for this
        AutoMixprecision request (see :mod:`onnxsim.quark_amp_cache`)."""
        from onnxsim.quark_amp_cache import quark_cache_key

        target = p.get("target_layer_config")
        ops = tuple(p.get("target_op_type") or PROMOTABLE_OPS)
        include = list(p.get("include_layers") or ())
        exclude = list(p.get("exclude_layers") or ())
        return lambda baseline: quark_cache_key(baseline, target, ops, include, exclude)

    def _block_attr_overrides(
        self, act: QSpec, wt: QSpec
    ) -> "Dict[str, Dict[str, Any]]":
        """The attributes of the BFP / MX nodes Quark's quantizer emits for
        this ``(activation, weight)`` pair: its ``BFP_OP_DEFAULT_ATTRS`` /
        ``MX_OP_DEFAULT_ATTRS`` (``rounding_mode`` 0), updated by the
        ``BFPAttributes`` / ``MXAttributes`` extra options -- which its config
        mapping fills in (the ``MX4`` / ``MX6`` / ``MX9`` / ``MXFP*`` /
        ``MXInt8`` attributes) only when activations and weights use the *same*
        such format, and its presets set explicitly. So a ``bfloat16`` model
        with ``MX4`` constants gets plain default ``BFPQuantizeDequantize``
        nodes, whatever the weight format."""
        opts = self.config.extra_options
        bfp: Dict[str, Any] = dict(_BFP_DEFAULTS)
        mx: Dict[str, Any] = dict(_MX_DEFAULTS)
        same = act.dtype == wt.dtype
        if opts.get("BFPAttributes") is not None:
            bfp.update(opts["BFPAttributes"])
        elif same and act.dtype in _MX_BFP_BITS:
            bfp.update(_BFP_PRIME_PARAMS, bit_width=_MX_BFP_BITS[act.dtype])
        if opts.get("MXAttributes") is not None:
            mx.update(opts["MXAttributes"])
        elif same and act.dtype in _MX_ELEMENT:
            mx.update(_MX_PARAMS, element_dtype=_MX_ELEMENT[act.dtype])
        return {"BFPQuantizeDequantize": bfp, "MXQuantizeDequantize": mx}

    def _amp_on_fakequant_base(
        self,
        model: onnx.ModelProto,
        act: QSpec,
        wt: QSpec,
        ignore_unsupported: bool,
        reader: Any = None,
    ) -> onnx.ModelProto:
        """AutoMixprecision over a ``float16`` / ``bfloat16`` / BFP / MX baseline
        (built as the plain preset would, see :meth:`_quantize_block`), with
        targets of any precision: Quark's in-place surgery on the baseline, the
        calibration being its fake ``[0, 1]`` ranges (no calibration is run for
        these formats)."""
        from onnxsim.quark_auto_mixprecision import (
            auto_mixprecision_from_baseline,
            parse_subgraph_json,
        )
        from onnxsim.quark_preset_graphs import _with_node_names

        cfg = self.config
        opts = cfg.extra_options
        algo = next(a for a in cfg.algo_config if a.name == "auto_mixprecision")
        others = [a.name for a in cfg.algo_config if a is not algo]
        if others and not ignore_unsupported:
            raise NotImplementedError(
                f"algo_config [{', '.join(others)}] is not applied with "
                f"{act.dtype}/{wt.dtype} baselines; pass "
                "ignore_unsupported_algos=True to quantize without it"
            )
        if not opts.get("BlockFormatActivations", True) or opts.get(
            "BlockFormatFoldWeights", False
        ):
            raise NotImplementedError(
                "AutoMixprecision needs the baseline's quantizer nodes: "
                "BlockFormatActivations=False / BlockFormatFoldWeights=True are "
                "not supported"
            )
        p = algo.params
        shared_mode = p.get("shared_param_mode", "propagate")
        if shared_mode not in ("propagate", "unshare"):
            raise ValueError("shared_param_mode must be 'propagate' or 'unshare'")
        calibration = _drain_reader(reader)
        if not calibration:
            raise ValueError(
                "calibration_data_reader is required to score the AutoMixprecision "
                "candidates"
            )
        model = _with_node_names(model)
        baseline = self._quantize_block(
            model, act, wt, self._block_attr_overrides(act, wt)
        )
        targets, pinned = self._amp_targets(p.get("target_layer_config"))
        subgraphs = None
        sg_path = p.get("subgraph_json")
        if sg_path is not None:
            if Path(sg_path).exists():
                specs = parse_subgraph_json(sg_path, model, model)
                subgraphs = [(sg.name, sg.resolved_nodes) for sg in specs]
            else:  # Quark checks ``Path(subgraph_json).exists()`` and moves on
                self._approx(
                    f"subgraph_json {sg_path!r} does not exist: ignored, as Quark does"
                )
        self._approx(
            f"AutoMixprecision over a {act.dtype}/{wt.dtype} baseline: candidates "
            "are scored on the ONNX reference evaluator for Quark's custom ops "
            "(ONNX Runtime for the rest); tensor ranges are Quark's fake [0, 1]"
        )
        res = auto_mixprecision_from_baseline(
            model,
            calibration,
            baseline,
            targets,
            candidate_targets=pinned,
            base_label=f"{act.dtype}/{wt.dtype}",
            target_op_types=tuple(p.get("target_op_type") or PROMOTABLE_OPS),
            include_layers=p.get("include_layers") or (),
            exclude_layers=p.get("exclude_layers") or (),
            metric=p.get("metric_default", "l2"),
            metric_distance_fn=p.get("metric_distance_fn"),
            metric_evaluate_fn=p.get("metric_evaluate_fn"),
            metric_threshold=p.get("metric_threshold", 0),
            optimize=p.get("metric_optimize_object", "speed"),
            metric_output_index=p.get("metric_output_index", 0),
            data_size=int(p.get("data_size", 0)) + 1,
            subgraphs=subgraphs,
            cache_file=p.get("sensitivity_cache_file"),
            worker_num=p.get("worker_num", 1),
            no_input_qdq_shared=bool(p.get("no_input_qdq_shared", False)),
            activation_symmetric=bool(opts.get("ActivationSymmetric", act.symmetric)),
            weight_symmetric=bool(opts.get("WeightSymmetric", wt.symmetric)),
            shared_param_mode=shared_mode,
            cache_key_fn=self._amp_cache_key_fn(p),
            dual_quant_nodes=bool(p.get("dual_quant_nodes", False)),
        )
        self.last_auto_mixprecision = res
        return res.model

    def _quantize_dynamic(
        self, model: onnx.ModelProto, act: QSpec, wt: QSpec
    ) -> onnx.ModelProto:
        from onnxsim.quark_dynamic import quantize_dynamic_integer

        cfg = self.config
        if cfg.algo_config:
            raise NotImplementedError(
                "algo_config is not applied to dynamic quantization"
            )
        if wt.dtype not in ("int8", "uint8"):
            raise NotImplementedError(f"weight dtype {wt.dtype} unsupported")
        if act.dtype not in ("int8", "uint8"):
            raise NotImplementedError(
                f"dynamic activation dtype {act.dtype} unsupported"
            )
        if act.dtype != "uint8":
            self._approx(
                "dynamic activations are quantized uint8 asymmetric "
                "(DynamicQuantizeLinear), whatever the activation spec's dtype"
            )
        exclude = _match_nodes(
            model, [e for e in cfg.exclude if isinstance(e, (str, tuple))]
        )
        _, _, type_excluded = self._layer_overrides(model, allow_dtypes=False)
        self._overrides_applied = True
        return quantize_dynamic_integer(
            model, weight_dtype=wt.dtype, exclude_nodes=exclude + type_excluded
        )

    def _quantize_block(
        self,
        model: onnx.ModelProto,
        act: QSpec,
        wt: QSpec,
        attr_overrides: "Optional[Dict[str, Dict[str, Any]]]" = None,
    ) -> onnx.ModelProto:
        """A model whose activations use a half / block format and whose
        constants use any half, block or ``int8`` / ``uint8`` format (Quark's
        quantizer picks the node kind per tensor from its own dtype)."""
        from onnxsim.quark_fakequant_graph import apply_fake_quant_format

        opts = self.config.extra_options
        exclude = [e for e in self.config.exclude if isinstance(e, str)]
        if act.dtype not in _FAKEQUANT_DTYPES:
            raise NotImplementedError(
                f"{act.dtype} activations over {wt.dtype} weights: integer "
                "activations need a calibrated integer quantizer next to the "
                "half / block weights"
            )
        block_consts = act.dtype == "bfloat16" and wt.dtype in _BLOCK_DTYPES
        # Quark's FP16 / BF16 presets set ``ForceQuantizeNoInputCheck`` (BF16 also
        # ``QuantizeAllOpTypes``, the op types of the model as given); the block
        # formats set neither. The extended quantizer's registry is NPU CNN's.
        half_preset = act.dtype in ("float16", "bfloat16") and not block_consts
        op_types = self._static_op_types(
            model,
            quantize_all=bool(
                opts.get("QuantizeAllOpTypes", act.dtype == "bfloat16" and half_preset)
            ),
        )
        if not _quantizable(model, op_types, exclude):
            # Quark: "No quantizable ops in this model" -- returned as given, before
            # any pre-processing
            return model
        model = _register_domains(
            self._preprocess_block_flow(model, exclude, True, op_types)
        )
        if not opts.get("SkipPreprocess", False) and opts.get(
            "ConvertBNToConv",
            not (opts.get("BF16QDQToCast") or opts.get("EnableVaimlBF16")),
        ):
            # (the BatchNormalizations its conversion to Conv leaves are quantized as
            # the Convs they became; with ``SkipPreprocess`` none is converted)
            op_types = op_types | {"BatchNormalization"}
        marking = {
            "op_types": op_types,
            "force_no_input_check": bool(
                opts.get("ForceQuantizeNoInputCheck", half_preset)
            ),
        }
        if attr_overrides is None:
            attr_overrides = self._block_attr_overrides(act, wt)
        if wt.dtype in ("int8", "uint8"):
            from onnxsim.quark_preset_graphs import apply_fake_quant_int_constants

            self._approx(
                f"{act.dtype} uses com.amd.quark custom ops: the model needs Quark's "
                "ONNX custom-op library to run (onnxsim cannot execute it)"
                if act.dtype in _BLOCK_DTYPES
                else f"{act.dtype} activations over {wt.dtype} constants use "
                "com.amd.quark extended Q/DQ ops: the model needs Quark's ONNX "
                "custom-op library to run (onnxsim cannot execute it)"
            )
            return apply_fake_quant_int_constants(
                model,
                act.dtype,
                wt.dtype,
                exclude,
                attr_overrides,
                weight_symmetric=bool(opts.get("WeightSymmetric", wt.symmetric)),
                int32_bias=opts.get("Int32Bias", True) is not False,
                quantize_bias=opts.get("QuantizeBias", True) is not False,
                marking=marking,
            )
        fn = _block_fn(wt.dtype)
        # constants in a format of their own over the activations' (BF16_BFP16 /
        # BF16_MXINT8, and every other pair of half / block formats)
        mixed = act.dtype != wt.dtype
        if fn is None:
            raise NotImplementedError(f"weight dtype {wt.dtype} unsupported")
        quantize_acts = act.dtype in _FAKEQUANT_DTYPES and opts.get(
            "BlockFormatActivations", True
        )
        fold = bool(opts.get("BlockFormatFoldWeights", False))
        if mixed and (not quantize_acts or fold):
            raise NotImplementedError(
                f"{act.dtype} activations over {wt.dtype} constants need the "
                "quantizer nodes: BlockFormatActivations=False / "
                "BlockFormatFoldWeights=True are not supported"
            )
        if act.dtype in _FAKEQUANT_DTYPES and not quantize_acts:
            self._approx(
                f"{act.dtype} activations are not quantized "
                "(extra_options BlockFormatActivations=False) -- weights only"
            )
        elif quantize_acts:
            self._approx(
                f"{act.dtype} uses com.amd.quark custom ops: the model needs Quark's "
                "ONNX custom-op library to run (onnxsim cannot execute it)"
            )
        return apply_fake_quant_format(
            model,
            act.dtype if mixed else wt.dtype,
            activations=bool(quantize_acts),
            fold_weights=fold,
            fold_fn=fn,
            exclude=exclude,
            const_dtype=wt.dtype if mixed and quantize_acts and not fold else None,
            attr_overrides=attr_overrides,
            remove_after=[
                op
                for op, option in (
                    ("Relu", "RemoveQDQConvRelu"),
                    ("Clip", "RemoveQDQConvClip"),
                    ("LeakyRelu", "RemoveQDQConvLeakyRelu"),
                    ("PRelu", "RemoveQDQConvPRelu"),
                )
                if opts.get(option, True)
            ],
            marking=marking,
        )

    def _extended(self, act: QSpec, wt: QSpec) -> bool:
        """Whether Quark would use its extended QDQ quantizer (see
        ``QConfig.quant_format``)."""
        fmt = self.config.quant_format
        if fmt is not None:
            return fmt == "extended"
        return not (act.dtype in ("int8", "uint8") and wt.dtype in ("int8", "uint8"))

    def _approx(self, msg: str) -> None:
        self.last_approximations.append(msg)

    def _quarot(self, model: onnx.ModelProto, algo: AlgoConfig) -> onnx.ModelProto:
        from onnxsim.quark_quarot import rotate_model

        p = algo.params
        if not p.get("r_config_path"):
            raise ValueError(
                "QuarotConfig.r_config_path is required (a JSON file with "
                '"R1_pairs": [{"prev_nodes", "next_nodes", "norm_node"}, ...])'
            )
        dim = p.get("r_matrix_dim", 4096)
        use_random_had = bool(p.get("use_random_had", False))
        from onnxsim.quark_hadamard import hadamard_factor

        try:
            hadamard_factor(dim)
        except ValueError as e:
            # Quark's `apply_QuaRot` wraps the failure like this.
            raise AssertionError(
                f"Error! The dim of the target R1 matrix is not support due to {e}."
            ) from e
        if use_random_had:
            self._approx(
                "Quarot: only the R1 (residual-stream) rotation is applied; the "
                "random-Hadamard row signs come from a numpy seed, not torch's RNG"
            )
        else:
            self._approx("Quarot: only the R1 (residual-stream) rotation is applied")
        return rotate_model(
            model,
            p["r_config_path"],
            r_matrix_dim=dim,
            use_random_had=use_random_had,
        )

    def _finetune(
        self,
        name: str,
        float_model: onnx.ModelProto,
        quantized: onnx.ModelProto,
        calibration: List[Dict[str, np.ndarray]],
        algo: AlgoConfig,
    ) -> onnx.ModelProto:
        """Quark's FastFinetune (``AdaRoundConfig`` / ``AdaQuantConfig``) via
        :mod:`onnxsim.quark_finetune`; ``extra_options["FastFinetune"]`` keys
        override the config's params, and ``QuantizationPreference="accuracy"``
        applies Quark's own overrides (``EarlyStop`` off, ``UpdateBias`` and
        ``OutputQDQ`` on). ``guard`` (an onnxsim addition, default on) keeps a
        layer's new codes only if its block reconstruction error did not get
        worse; ``guard=False`` is Quark's behaviour."""
        from onnxsim.quark_finetune import (
            TARGET_OPS,
            FinetuneOptions,
            finetune,
            load_saved_layers,
            save_checkpoint,
        )

        p = dict(algo.params)
        dropped = [k for k in _FASTFT_NOT_FORWARDED if k in p]
        for k in dropped:
            del p[k]
        if dropped:
            self._approx(
                f"{name}: Quark's config does not forward {', '.join(dropped)} "
                "(only extra_options['FastFinetune'] does), so they are ignored here too"
            )
        ff = self.config.extra_options.get("FastFinetune")
        if isinstance(ff, dict):
            p.update({_FASTFT_KEYS[k]: v for k, v in ff.items() if k in _FASTFT_KEYS})
        if self.config.extra_options.get("QuantizationPreference") == "accuracy":
            p.update(early_stop=False, update_bias=True, output_qdq=True)
        if "data_size" in p:
            calibration = calibration[: int(p["data_size"])]
        ref = p.get("ref_model_path")
        if isinstance(ref, str) and os.path.exists(ref):
            float_model = onnx.load(ref)
        adaquant = name == "adaquant"
        targets = tuple(p.get("target_op_type") or TARGET_OPS)
        opt = FinetuneOptions(
            algorithm=name,
            num_iterations=int(p.get("num_iterations", 3000 if adaquant else 1000)),
            learning_rate=p.get("learning_rate"),
            batch_size=int(p.get("batch_size", 1)),
            num_batches=int(p.get("num_batches", 1)),
            early_stop=bool(p.get("early_stop", False)),
            reg_param=float(p.get("reg_param", 0.01)),
            beta_range=tuple(p.get("beta_range", (20.0, 2.0))),  # type: ignore[arg-type]
            warm_start=float(p.get("warm_start", 0.2)),
            drop_ratio=float(p.get("drop_ratio", 1.0)),
            lr_adjust=tuple(p["lr_adjust"]) if p.get("lr_adjust") else None,  # type: ignore[arg-type]
            selective_update=bool(p.get("selective_update", False)),
            update_bias=bool(p.get("update_bias", False)) and adaquant,
            output_qdq=bool(p.get("output_qdq", False)),
            parallel=bool(p.get("parallel", False)),
            mem_opt_level=int(p.get("mem_opt_level", 1)),
            num_workers=int(p.get("num_workers", 1)),
            dynamic_batch=bool(p.get("dynamic_batch", False)),
            output_index=p.get("output_index"),
            select_max_mem_layer=bool(p.get("select_max_mem_layer", False)),
            target_ops=targets,
            seed=int(p.get("fixed_seed", 1705472343)),
            guard=bool(p.get("guard", True)),
        )
        self._approx(
            f"{name} is a numpy port of Quark's FastFinetune loop: mini-batches "
            "come from numpy's generator instead of torch.randperm"
            + (
                " and the arithmetic is float32 in torch's order (MatMul / "
                "Gemm layers are bit-identical, convolutions and norms differ "
                "in the last bit)"
                if name == "adaquant"
                else " and the arithmetic is float32 for 16-bit weight grids, "
                "float64 otherwise"
            )
            + ", so results match Quark statistically "
            "(bit for bit given the same mini-batch indices)"
        )
        # update_bias: only AdaQuant reads it, as in Quark.
        # ``SaveAndRestore`` (a JSON checkpoint file): Quark writes the layer it
        # reached before every layer and, when the file already exists, trains
        # only the layers it lists
        saver = self.config.extra_options.get("SaveAndRestore")
        layers = load_saved_layers(saver)
        checkpoint = (
            (lambda i, n, m: save_checkpoint(saver, i, n, m)) if saver else None
        )
        out, self.last_weight_rounding[name] = finetune(
            float_model,
            quantized,
            calibration,
            opt,
            layers=layers,
            checkpoint=checkpoint,
        )
        return out

    def _gptq(
        self,
        float_model: onnx.ModelProto,
        quantized: onnx.ModelProto,
        calibration: List[Dict[str, np.ndarray]],
        algo: AlgoConfig,
    ) -> onnx.ModelProto:
        from onnxsim.quark_weight_rounding import gptq_int8

        p = algo.params
        bits = int(p.get("bits", 8))
        group_size = int(p.get("group_size", -1))
        sym = bool(p.get("weight_symmetric", True))
        mse = bool(p.get("mse", False))
        low_opset = _default_opset(quantized) < 13
        requantize = (
            bits != 8
            or group_size != -1
            or not sym
            or mse
            or "per_channel" in p
            # (below opset 13 the Q/DQ weights are per tensor: there are no
            # per-channel scales to keep)
            or low_opset
        )
        if requantize:
            self._approx(
                "GPTQ re-grids the weights like Quark's GPTQ (GPTQConfig.bits / "
                "group_size / per_channel / mse / weight_symmetric) from all "
                "calibration batches, and does propagate rounding error (Quark "
                "0.13's update is a no-op)"
            )
        else:
            self._approx(
                "GPTQ keeps quantize_full_qdq's per-channel scales "
                "(GPTQConfig.per_channel / mse are not used)"
            )
        kwargs: Dict[str, Any] = {
            "bits": bits,
            "group_size": group_size,
            "weight_symmetric": sym,
            "mse": mse,
            "per_channel": bool(p.get("per_channel", False)),
            "requantize": requantize,
        }
        if "perc_damp" in p:
            kwargs["perc_damp"] = p["perc_damp"]
        if "block_size" in p:
            kwargs["block_size"] = p["block_size"]
        if "act_order" in p:
            kwargs["act_order"] = bool(p["act_order"])
        out, self.last_weight_rounding["gptq"] = gptq_int8(
            float_model, quantized, calibration, **kwargs
        )
        return out

    def _layer_overrides(
        self, model: onnx.ModelProto, allow_dtypes: bool = True
    ) -> "tuple[Dict[str, str], Dict[str, bool], List[str]]":
        """``(tensor_dtypes, tensor_symmetric, excluded nodes)`` from
        ``layer_type_config`` then ``specific_layer_config`` (the latter wins,
        as in Quark). A layer's ``input_tensors`` spec applies to its
        activation inputs (those before the first constant), ``output_tensors``
        to its outputs; weight / bias overrides are not supported."""
        cfg = self.config
        inits = {i.name for i in model.graph.initializer}
        dtypes: Dict[str, str] = {}
        symmetric: Dict[str, bool] = {}
        excluded: List[str] = []
        by_name = {n.name: n for n in model.graph.node if n.name}

        def apply(node: onnx.NodeProto, layer: QLayerConfig) -> None:
            if layer.weight is not None and layer.weight.dtype not in (
                "int8",
                "uint8",
            ):
                raise NotImplementedError(
                    f"per-layer weight dtype {layer.weight.dtype} is not supported "
                    "(weights are int8)"
                )
            if layer.bias is not None:
                self._approx("per-layer bias specs ignored (biases stay int32)")
            for spec, tensors in (
                (layer.activation, _activation_inputs(node, inits)),
                (layer.output_tensors, list(node.output)),
            ):
                if spec is None:
                    continue
                dt = self._int_act_dtype(spec)
                for t in tensors:
                    dtypes[t] = dt
                    symmetric[t] = spec.symmetric

        for layer, op_types in cfg.layer_type_config.items():
            if layer is None:
                excluded += [
                    n.name for n in model.graph.node if n.op_type in op_types and n.name
                ]
                continue
            for n in model.graph.node:
                if n.op_type in op_types:
                    apply(n, layer)
        for layer, names in cfg.specific_layer_config.items():
            for name in _match_nodes(model, names):
                if name not in by_name:
                    raise ValueError(f"specific_layer_config: no node named {name!r}")
                apply(by_name[name], layer)
        if dtypes and not allow_dtypes:
            self._approx("per-layer activation dtypes ignored (dynamic quantization)")
            dtypes, symmetric = {}, {}
        return dtypes, symmetric, excluded

    def _int_act_dtype(self, spec: QSpec) -> str:
        """The ``quantize_full_qdq`` activation dtype for an int spec."""
        if spec.dtype in ("int8", "uint8", "int16", "uint16"):
            return spec.dtype
        raise NotImplementedError(f"activation dtype {spec.dtype} unsupported")

    def _amp_targets(
        self, target: Any
    ) -> "tuple[List[TargetSpec], Dict[str, TargetSpec]]":
        """``(targets, pinned)`` of an ``AutoMixprecisionConfig.target_layer_config``:
        a single :class:`QLayerConfig`, a list of them (the best-scoring one
        is used per candidate) or ``{QLayerConfig: [node names]}`` (the named
        nodes use their entry; the others the entry with ``[]``, or the first
        entry when there is none).

        Each entry becomes a :class:`TargetSpec` the way Quark's
        ``MixingStrategy`` reads a ``QLayerConfig``: ``activation`` moves the
        node's activation inputs *and* outputs, ``input_tensors`` /
        ``output_tensors`` one side each, ``weight`` / ``bias`` the constant
        operands; a missing spec leaves that slot as quantized. A target equal
        to the base precision is not an error (Quark re-quantizes the weights
        per tensor and refreshes the bias scales regardless)."""
        from onnxsim.quark_auto_mixprecision import TargetSpec

        opts = self.config.extra_options
        g_act = self.config.global_config.activation
        g_wt = self.config.global_config.weight
        assert g_act is not None and g_wt is not None

        def prec(spec: Optional[QSpec], role: str) -> "Optional[tuple[Any, ...]]":
            if spec is None:
                return None
            from onnxsim.quark_mixing import kind_of

            try:
                kind_of(spec.dtype)
            except ValueError:
                raise NotImplementedError(
                    f"target_layer_config {role} dtype {spec.dtype} is not supported"
                ) from None
            # Quark's ``extra_options`` carry the *global* spec's symmetry as
            # ``ActivationSymmetric`` / ``WeightSymmetric``, and those win over
            # the target spec's own (a bias spec always keeps its own)
            glob = {"activation": g_act.symmetric, "weight": g_wt.symmetric}
            key = {"activation": "ActivationSymmetric", "weight": "WeightSymmetric"}
            sym = spec.symmetric
            if role in key:
                sym = bool(opts.get(key[role], glob[role]))
            # a power-of-two scale (Quark's ``PowerOfTwoMethod``: a PowerOf2
            # scale type or the MinMSE calibration) rounds the new scale
            if spec.pof2 or spec.calibration_method == "minmse_pof2":
                return spec.dtype, sym, True
            return spec.dtype, sym

        def spec_of(cfg: Any) -> TargetSpec:
            if not isinstance(cfg, QLayerConfig):
                raise TypeError(
                    "target_layer_config must be a QLayerConfig, a list of "
                    f"them or a dict {{QLayerConfig: [names]}}, got {cfg!r}"
                )
            out_spec = (
                cfg.activation
                if getattr(cfg, "_activation_spelled", False)
                else cfg.output_tensors
            )
            return TargetSpec(
                inputs=prec(cfg.activation, "activation"),
                outputs=prec(out_spec, "activation"),
                weight=prec(cfg.weight, "weight"),
                bias=prec(cfg.bias, "bias"),
            )

        if isinstance(target, dict):
            if not target:
                raise ValueError("target_layer_config dict must not be empty")
            fallback = [c for c, names in target.items() if not names]
            default = fallback[0] if fallback else next(iter(target))
            pinned = {n: spec_of(c) for c, names in target.items() for n in names}
            return [spec_of(default)], pinned
        if isinstance(target, (list, tuple)):
            if not target:
                raise ValueError("target_layer_config list must not be empty")
            return [spec_of(c) for c in target], {}
        return [spec_of(target)], {}

    def _base_bias_post(
        self, work: onnx.ModelProto, act: QSpec, wt: QSpec
    ) -> "Optional[Callable[[onnx.ModelProto], onnx.ModelProto]]":
        """The re-quantization of the baseline's int32 biases Quark's
        ``Int32Bias=False`` asks for (a bias quantized like a weight, in the
        weight's dtype); ``None`` when biases stay int32."""
        opts = self.config.extra_options
        if opts.get("Int32Bias", True) is not False or not opts.get(
            "QuantizeBias", True
        ):
            return None
        from onnxsim.quark_preset_graphs import requantize_biases_int8

        pof2 = act.pof2 or wt.pof2
        bits = "int16" if wt.dtype == "int16" else "int8"
        self._approx(
            f"{bits} bias (Int32Bias=False): symmetric per tensor"
            + (", power-of-2 scale" if pof2 else "")
        )
        return lambda q: requantize_biases_int8(
            q,
            work,
            ("Conv", "ConvTranspose", "Gemm"),
            power_of_two=pof2,
            dtype=bits,
            per_channel=bool(opts.get("PerChannel", False)),
            symmetric=bool(opts.get("WeightSymmetric", wt.symmetric)),
        )

    def _auto_mixprecision(
        self,
        model: onnx.ModelProto,
        calibration: List[Dict[str, np.ndarray]],
        base_dtype: str,
        act: QSpec,
        exclude: List[str],
        algo: AlgoConfig,
        quantize_kwargs: Optional[Dict[str, Any]] = None,
        post_quantize: "Optional[Callable[[onnx.ModelProto], onnx.ModelProto]]" = None,
    ) -> onnx.ModelProto:
        from onnxsim.quark_auto_mixprecision import auto_mixprecision

        p = algo.params
        targets, pinned = self._amp_targets(p.get("target_layer_config"))
        subgraphs = None
        sg_path = p.get("subgraph_json")
        if sg_path is not None:
            if Path(sg_path).exists():
                from onnxsim.quark_auto_mixprecision import parse_subgraph_json

                specs = parse_subgraph_json(sg_path, model, model)
                subgraphs = [(s.name, s.resolved_nodes) for s in specs]
            else:  # Quark checks ``Path(subgraph_json).exists()`` and moves on
                self._approx(
                    f"subgraph_json {sg_path!r} does not exist: ignored, as Quark does"
                )
        shared_mode = p.get("shared_param_mode", "propagate")
        if shared_mode not in ("propagate", "unshare"):
            raise ValueError("shared_param_mode must be 'propagate' or 'unshare'")
        self._approx(
            "AutoMixprecision: activation, weight and bias precisions are mixed "
            "as Quark's MixingStrategy does (weights re-quantized per tensor from "
            "their quantized values)"
        )
        optimize = p.get("metric_optimize_object", "speed")
        cal_method, cal_options = _calibration_args(
            self._calib_method(act), self.config.extra_options
        )
        # Quark's ``inference_model`` stops once ``len(results) > data_size``:
        # ``data_size=N`` scores N + 1 batches and the default 0 only the first
        data_size = int(p.get("data_size", 0)) + 1
        res = auto_mixprecision(
            model,
            calibration,
            base_dtype=base_dtype,
            targets=targets,
            candidate_targets=pinned,
            subgraphs=subgraphs,
            cache_file=p.get("sensitivity_cache_file"),
            worker_num=p.get("worker_num", 1),
            no_input_qdq_shared=bool(p.get("no_input_qdq_shared", False)),
            dual_quant_nodes=bool(p.get("dual_quant_nodes", False)),
            quantize_kwargs=quantize_kwargs,
            post_quantize=post_quantize,
            target_op_types=tuple(p.get("target_op_type") or PROMOTABLE_OPS),
            include_layers=p.get("include_layers") or (),
            exclude_layers=p.get("exclude_layers") or (),
            exclude_nodes=exclude,
            metric=p.get("metric_default", "l2"),
            metric_distance_fn=p.get("metric_distance_fn"),
            metric_evaluate_fn=p.get("metric_evaluate_fn"),
            metric_threshold=p.get("metric_threshold", 0),
            optimize=optimize,
            metric_output_index=p.get("metric_output_index", 0),
            data_size=data_size,
            method=cal_method,
            calibrate_options=_exact(cal_options),
            shared_param_mode=shared_mode,
            cache_key_fn=self._amp_cache_key_fn(p),
        )
        self.last_auto_mixprecision = res
        return res.model

    def _calib_method(self, act: QSpec) -> str:
        """The activation calibration method, with Quark's
        ``CalibMovingAverage`` extra option applied to min / max calibration
        (the mean of the per-batch ranges instead of the global range)."""
        method = act.calibration_method
        if isinstance(method, CalibMethod):  # assigned after construction
            method = _CALIB_NAMES[method]
        moving = self.config.extra_options.get("CalibMovingAverage")
        if method == "minmax" and moving:
            return "minmax_mean"
        if method == "minmax_mean" and moving is False:
            return "minmax"
        return str(method)

    def _keeps_constants(self) -> bool:
        """Quark folds the ``Constant`` nodes into initializers in ONNX Runtime's graph
        loader / onnxslim only: with neither pass (``OptimizeModel`` and
        ``SimplifyModel`` off), or with ``SkipPreprocess``, they stay in the graph and
        its quantizers treat their outputs as activations (not as weights)."""
        opts = self.config.extra_options
        return bool(opts.get("SkipPreprocess", False)) or not (
            opts.get("OptimizeModel", True) or opts.get("SimplifyModel", True)
        )

    def _transformer_scope(
        self, model: onnx.ModelProto
    ) -> "tuple[Optional[set[str]], List[str]]":
        """``(op types, nodes to keep float)`` of Quark's NPU transformer
        scheme: only ``Gemm`` and ``MatMul`` are quantized, and (unless
        ``MatMulConstBOnly=False``) only MatMuls whose second operand is a
        constant. ``(None, [])`` for every other preset."""
        opts = self.config.extra_options
        if not opts.get("NPUTransformer"):
            return None, []
        consts = {i.name for i in model.graph.initializer}
        if not self._keeps_constants():
            consts |= {n.output[0] for n in model.graph.node if n.op_type == "Constant"}
        skip = []
        if opts.get("MatMulConstBOnly", True):
            skip = [
                n.name or n.output[0]
                for n in model.graph.node
                if n.op_type == "MatMul" and n.input[1] not in consts
            ]
        return {"Gemm", "MatMul"}, skip

    def _float_preprocess(
        self,
        model: onnx.ModelProto,
        copy_bias: bool = False,
        keep_bn: "Optional[Set[str]]" = None,
    ) -> onnx.ModelProto:
        """Quark's float-graph pre-processing before the algorithms
        (``apply_pre_optimization_before_algo``), in Quark's order: the opset
        conversion, onnxslim, ONNX Runtime's basic optimizer and BatchNorm folding,
        Quark's operator fusions, then its topological sort (which
        ``SkipPreprocess`` does not skip). ``copy_bias`` copies a shared bias per node
        (``CopyBiasInit``; Quark does it for the integer min / max-style
        calibrations only)."""
        from onnxsim.quark_marking import quark_sorted

        opts = self.config.extra_options
        if opts.get("SkipPreprocess", False):
            return quark_sorted(model)
        work = model
        # Quark's operator fusions (``optimize_model``, after onnxslim and ONNX
        # Runtime's optimizer, whether or not those ran): on by default, opset >= 17
        # (LayerNorm) / >= 20 (Gelu), see :mod:`onnxsim.quark_fusions`
        fuse: Dict[str, Any] = {
            "instance_norm": bool(opts.get("FuseInstanceNorm", True)),
            "l2_norm": bool(opts.get("FuseL2Norm", True)),
            "layer_norm": bool(opts.get("FuseLayerNorm", True)),
            "gelu": bool(opts.get("FuseGelu", True)),
        }
        target_opset = opts.get("ConvertOpsetVersion")
        if isinstance(target_opset, int):
            # Quark's first pre-processing step; a failed conversion is a warning
            # and the model goes on as it is
            from onnxsim.quark_tools import convert_opset_version

            try:
                work = convert_opset_version(work, target_opset)
            except ValueError as e:
                self._approx(f"opset conversion skipped: {e}")
        if opts.get("OptimizeModel", True) or opts.get("SimplifyModel", True):
            # Quark first runs onnxslim and ONNX Runtime's graph optimizer on the
            # float model (BN folds, Pad fusion, ...)
            from onnxsim.quark_convert import expand_hardswish, graph_cleanup

            # (``UseRuntimeOptimizers`` False: onnxsim's own reproductions of
            # the passes, for environments where Quark's would not be found)
            runtime = bool(opts.get("UseRuntimeOptimizers", True))
            if opts.get("OptimizeModel", True) and not runtime:
                work = expand_hardswish(work)
            work = graph_cleanup(
                work,
                bool(opts.get("OptimizeModel", True)),
                bool(opts.get("SimplifyModel", True)),
                runtime=runtime,
                slim_config=opts.get("SimplifyModelOptions"),
                fold_bn=bool(
                    opts.get("FoldBatchNorm", opts.get("OptimizeModel", True))
                ),
                copy_bias_ops=(
                    opts.get("CopyBiasInit", ("Conv", "ConvTranspose", "Gemm"))
                    if copy_bias
                    else None
                ),
                fuse=fuse,
                keep_bn=keep_bn,
            )
        else:
            from onnxsim.quark_fusions import apply_fusions

            work = apply_fusions(work, **fuse)
        # (Quark's fusion / folding passes end with its own topological sort)
        return quark_sorted(work)

    def _static_op_types(
        self,
        model: onnx.ModelProto,
        extended: bool = True,
        quantize_all: bool = False,
    ) -> "frozenset[str]":
        """Quark's ``get_static_op_types`` for these flows: the registry's op types
        (the extended quantizer's include the NPU CNN ones), the extra ones, and with
        ``QuantizeAllOpTypes`` every op type of the model as given."""
        from onnxsim.quark_marking import quark_op_types

        opts = self.config.extra_options
        types = set(quark_op_types(extended, opts.get("ExtraOpTypesToQuantize") or ()))
        if quantize_all:
            types |= {n.op_type for n in model.graph.node}
        return frozenset(types)

    def _preprocess_block_flow(
        self,
        model: onnx.ModelProto,
        exclude: Sequence[str] = (),
        extended: bool = True,
        op_types: "Optional[Iterable[str]]" = None,
        copy_bias: bool = False,
    ) -> onnx.ModelProto:
        """Quark's whole float pre-processing for the block-format, bfloat16 /
        float16 and ``MATMUL_NBITS`` flows (they go through the same
        ``apply_pre_process`` as the integer presets): :meth:`_float_preprocess`
        (onnxslim, ONNX Runtime's optimizer, BatchNorm folding, the operator
        fusions, the sort), then the conversions that follow the algorithms
        (``ReduceMean`` -> ``GlobalAveragePool``, ``BatchNormalization`` -> ``Conv``,
        ``Split`` -> ``Slice``, large pooling kernels), which Quark switches on for its
        extended ``QDQ`` format (``extended``: not ``MATMUL_NBITS``, whose plain ``QDQ``
        format leaves them off) unless the bfloat16-as-Cast options are set. A shared
        bias is copied per node (``CopyBiasInit``) only for ``copy_bias``: Quark does it
        when both types are integer (``MATMUL_NBITS``'s, not the block / half ones's).
        ``SkipPreprocess``
        skips everything but the sort."""
        opts = self.config.extra_options
        work = self._float_preprocess(
            model,
            copy_bias=copy_bias,
            keep_bn=_quantized_batch_norms(model, op_types, exclude),
        )
        if opts.get("SkipPreprocess", False):
            return work
        from onnxsim.quark_convert import convert_for_npu

        keep = set(exclude)
        work = convert_for_npu(
            work,
            opts,
            lambda n: n.name not in keep,
            default=extended
            and not (opts.get("BF16QDQToCast") or opts.get("EnableVaimlBF16")),
        )
        return work

    def _quantize_int(
        self,
        model: onnx.ModelProto,
        act: QSpec,
        wt: QSpec,
        reader: Any,
        algos: List[AlgoConfig],
    ) -> onnx.ModelProto:
        from onnxsim.full_qdq import quantize_full_qdq

        if wt.dtype not in ("int8", "uint8", "int16"):
            raise NotImplementedError(f"weight dtype {wt.dtype} unsupported")
        # GPTQ and the legacy AdaQuant engine work on int8 weight codes (Quark's
        # GPTQ ignores the preset's weight dtype and emits an 8-bit grid; its
        # FastFinetune -- AdaRound, AdaQuant -- trains any integer weight grid)
        int8_only = any(
            a.name == "gptq" or (a.name == "adaquant" and a.params.get("legacy_engine"))
            for a in algos
        )
        uint8_weights = wt.dtype == "uint8" and not int8_only
        int16_weights = wt.dtype == "int16" and not int8_only
        if wt.dtype in ("uint8", "int16") and int8_only:
            self._approx(
                f"weights quantized int8-symmetric instead of {wt.dtype} "
                "(GPTQ / the legacy AdaQuant engine work on int8 codes)"
            )
        act_dtype = self._int_act_dtype(act)

        calibration = _drain_reader(reader)
        if not calibration:
            raise ValueError("calibration_data_reader is required for integer presets")
        exclude = _match_nodes(
            model, [e for e in self.config.exclude if isinstance(e, (str, tuple))]
        )
        t_dtypes, t_sym, type_excluded = self._layer_overrides(model)
        self._overrides_applied = True
        exclude += type_excluded
        opts = self.config.extra_options
        op_types, scope_excluded = self._transformer_scope(model)
        exclude += scope_excluded
        if op_types is not None and not any(
            n.op_type in op_types and (n.name or n.output[0]) not in set(exclude)
            for n in model.graph.node
        ):
            # Quark: "No quantizable ops in this model" -- the float graph comes
            # back as it is (its Constant nodes unfolded), in Quark's node order
            from onnxsim.quark_marking import quark_sorted

            return quark_sorted(model)
        by_name = {a.name: a for a in algos}

        # Quark's presets quantize weights per tensor; the weight-rounding
        # algorithms below work per output channel.
        per_channel = bool(self.config.extra_options.get("PerChannel", False))
        legacy_adaquant = "adaquant" in by_name and by_name["adaquant"].params.get(
            "legacy_engine"
        )
        needs_axis = not per_channel and ("gptq" in by_name or legacy_adaquant)
        if needs_axis and _default_opset(model) >= 13:
            per_channel = True
            self._approx("weights quantized per channel (needed by the algorithm)")
        elif needs_axis and legacy_adaquant:
            # A DequantizeLinear has no ``axis`` below opset 13, so per-channel
            # weights cannot be written there (Quark raises for a per-channel
            # *request* -- an algorithm's own need is no request). GPTQ re-grids
            # the weights itself; the legacy engine only knows the per-channel
            # layout, finds no layer and leaves the model as quantized.
            self._approx(
                "the legacy AdaQuant engine needs per-channel weights, which "
                "DequantizeLinear cannot express below opset 13: weights stay "
                "per tensor"
            )

        # Float -> float pre-quantization passes (quantize_full_qdq is fed
        # the transformed model; the untouched one stays the reference).
        float_model = model
        work = model
        npu_cnn = bool(act.pof2 and wt.pof2 and opts.get("EnableNPUCnn", True))
        if npu_cnn and bool(opts.get("PerChannel", False)):
            raise ValueError(
                "Only per-tensor quantization is supported when enable_npu_cnn=True, "
                "`per_channel` must be set to False."
            )
        if opts.get("ReduceRange") and npu_cnn:
            raise ValueError(
                "ReduceRange is not supported with the NPU CNN scheme (power-of-two "
                "scales); Quark refuses it too"
            )
        skip_pre = bool(opts.get("SkipPreprocess", False))
        static_types = (
            op_types
            if op_types is not None
            else self._static_op_types(
                model,
                extended=npu_cnn or self._extended(act, wt),
                quantize_all=bool(opts.get("QuantizeAllOpTypes")),
            )
        )
        work = self._float_preprocess(
            work,
            copy_bias=self._calib_method(act)
            not in ("minmse_pof2", "nonoverflow", "layerwise_percentile"),
            keep_bn=_quantized_batch_norms(model, static_types, exclude),
        )
        # Quark's order: CLE (stem equalization first), SmoothQuant, Quarot
        if "cle" in by_name:
            from onnxsim.quark_equalization import apply_cle_config

            work = apply_cle_config(
                work, by_name["cle"].params, opts, exclude=list(exclude)
            )
        if "smooth_quant" in by_name:
            from onnxsim.quark_smoothquant import apply_smooth_quant_config

            work = apply_smooth_quant_config(
                work, by_name["smooth_quant"].params, opts, calibration
            )
        if "quarot" in by_name:
            work = self._quarot(work, by_name["quarot"])
        # the compiler-oriented conversions are on by default for these flows (and
        # follow their own options elsewhere, as in VINT8)
        from onnxsim.quark_convert import convert_for_npu
        from onnxsim.quark_marking import (
            opset_unquantized_ops,
            quark_op_types,
            quark_sorted,
            skipped_nodes,
        )

        keep = set(exclude)
        conv_default = bool(
            npu_cnn or self._extended(act, wt) or opts.get("NPUTransformer")
        )
        ext = self._extended(act, wt)
        # the op types Quark's quantizer takes (``get_static_op_types``): its
        # pre-optimizer converts only nodes of these types (so the NPU transformer
        # flow, Gemm / MatMul only, leaves a ReduceMean or a Split alone; a plain
        # quantizer, whose registry has no ReduceMean, does too)
        cnn_types: "Optional[set[str]]" = None
        if op_types is None:
            cnn_types = set(
                quark_op_types(npu_cnn or ext, opts.get("ExtraOpTypesToQuantize") or ())
            )
            if opts.get("QuantizeAllOpTypes"):
                # (Quark lists the op types of the model as given, before its
                # pre-processing: an op type that only a conversion or fusion
                # introduces and no registry has -- Slice, LpNormalization -- is
                # not on the list)
                cnn_types |= {n.op_type for n in model.graph.node}
        convert_types = op_types if op_types is not None else cnn_types
        constants_before = {
            n.output[0] for n in work.graph.node if n.op_type == "Constant"
        }
        if not skip_pre:
            work = convert_for_npu(
                work,
                opts,
                # (Quark's BatchNorm conversion adds BatchNormalization to the list
                # itself, first: it is never held back by it)
                lambda n: (
                    n.name not in keep
                    and (
                        convert_types is None
                        or n.op_type in convert_types
                        or n.op_type == "BatchNormalization"
                    )
                ),
                default=conv_default,
            )
        if opts.get("ConvertBNToConv", conv_default):
            # (Quark's conversion adds BatchNormalization to the shared list of op
            # types the quantizer takes later, converted or not: a BatchNorm that
            # is left over -- its parameters Constants, or 2-D -- is quantized, in
            # the NPU transformer scheme too)
            if cnn_types is not None:
                cnn_types.add("BatchNormalization")
            elif op_types is not None:
                op_types = set(op_types) | {"BatchNormalization"}
        # the Constant nodes Quark's own conversions add stay in its graph whatever
        # ONNX Runtime and onnxslim did to the float model before
        converted_constants = {
            n.output[0]
            for n in work.graph.node
            if n.op_type == "Constant" and n.output[0] not in constants_before
        }
        # Quark topologically sorts the float graph (with its own sort) before the
        # quantizer visits it, and quantizes the op types of its registries only
        work = quark_sorted(work)
        scope_types = op_types if op_types is not None else cnn_types
        skip_nodes = skipped_nodes(
            work,
            scope_types,
            exclude,
            # (every Quark preset sets it; a bare QConfig does not)
            force_no_input_check=bool(opts.get("ForceQuantizeNoInputCheck", True)),
            direct_pool=not (ext or npu_cnn),
            npu_registry=bool(ext or npu_cnn),
            unquantized_ops=opset_unquantized_ops(work),
        )
        # Quark adds BatchNormalization to the op types it quantizes when it
        # converts BNs; without ConvertBNToConv a leftover one stays float
        bn_quantized = bool(opts.get("ConvertBNToConv", conv_default))
        if work is not model:
            float_model = work

        cal_method, cal_options = _calibration_args(self._calib_method(act), opts)
        cal_size = int(opts.get("CalibDataSize") or 0)
        act_sym = bool(opts.get("ActivationSymmetric", act.symmetric))
        qkw: Dict[str, Any] = dict(
            keep_constants=True if self._keeps_constants() else converted_constants,
            calibration_data=calibration[:cal_size] if cal_size else calibration,
            activation_dtype=act_dtype,
            op_types=(
                scope_types
                if bn_quantized or scope_types is not None
                else _without_batch_norm(work, op_types)
            ),
            float_clamp_input=op_types is not None,
            # (the contrib Gelu Quark's FuseGelu writes is quantized by op type)
            contrib_ops=("Gelu",),
            exclude_nodes=exclude,
            skip_nodes=skip_nodes,
            method=cal_method,
            calibrate_options=_exact(cal_options),
            symmetric_activations=act_sym,
            power_of_two=act.pof2 or wt.pof2,
            per_channel=per_channel,
            weight_dtype=(
                "int16" if int16_weights else "uint8" if uint8_weights else "int8"
            ),
            weight_symmetric=bool(
                opts.get("WeightSymmetric", wt.symmetric or int8_only)
            ),
            quantize_bias=bool(opts.get("QuantizeBias", True)),
            **_activation_rules(opts, act_sym, self._extended(act, wt), npu_cnn),
            # Quark's XINT8 (power-of-2 weights): MinMSE scale search on
            # weights and int8 biases (``Int32Bias=True`` keeps int32)
            pof2_mode="minmse" if wt.pof2 else "ceil",
            # (the NPU quantizer keeps int8 biases unless Int32Bias; the others int32
            # unless Int32Bias=False)
            int8_bias=bool(
                wt.pof2 and (not opts["Int32Bias"] if "Int32Bias" in opts else npu_cnn)
            ),
            int8_constants=True,
            reduce_range=bool(opts.get("ReduceRange", False)),
            # (Quark applies it to the extended quantizer only: elsewhere it warns
            # and does nothing)
            ort_gemm_beta=True,
            asymmetric_minmse_pof2=True,
            prelu_slope_per_row=self._extended(act, wt),
            excluded_nodes_stay_float=False,
            # (the extended quantizer's refinement -- alignment, then the bias
            # scale adjustment -- always runs)
            adjust_bias_scale=(
                bool(opts.get("AdjustBiasScale", True))
                if self._extended(act, wt) and not npu_cnn
                else None
            ),
            align_eltwise_dtype=bool(
                self.config.extra_options.get("AlignEltwiseQuantType")
                and self._extended(act, wt)
                and not npu_cnn
            ),
            softmax_unit_range=True,
            tensor_dtypes=t_dtypes or None,
            tensor_symmetric=t_sym or None,
        )
        mixed_algo = by_name.get("auto_mixprecision")
        skip_names = set(exclude)

        def finish(q: onnx.ModelProto) -> onnx.ModelProto:
            """Quark's quantizer-side rewrites that follow the Q/DQ insertion
            (and so come *before* AutoMixprecision, which edits their result)."""
            if npu_cnn:
                from onnxsim.quark_marking import quark_qdq_sorted
                from onnxsim.quark_npu import apply_npu_cnn_rewrites

                # Quark sorts the Q/DQ graph (``topological_sort`` again, after it
                # has pruned the Q/DQ pairs) before the rewrites visit it
                q = quark_qdq_sorted(q)
                q = apply_npu_cnn_rewrites(
                    q,
                    opts,
                    _activation_rules(opts, act_sym, False, True)["remove_qdq_after"],
                    lambda n: n.name not in skip_names,
                )
                if not opts.get("OnnxsimKeepQuarkNodeOrder", False):
                    # Quark's DPU nodes sit at the end of the node list; give the
                    # graph the topological order ONNX requires
                    from onnxsim.full_qdq import _toposort

                    _toposort(q.graph)
            elif ext:
                from onnxsim.quark_npu import (
                    apply_extended_rewrites,
                    extended_simulates,
                )

                if extended_simulates(opts):
                    # (Quark's extended quantizer sorts its pruned Q/DQ graph, then
                    # converts the nodes the options ask for)
                    from onnxsim.full_qdq import _toposort
                    from onnxsim.quark_marking import quark_qdq_sorted

                    q = apply_extended_rewrites(
                        quark_qdq_sorted(q),
                        opts,
                        lambda n: n.name not in skip_names,
                    )
                    if not opts.get("OnnxsimKeepQuarkNodeOrder", False):
                        _toposort(q.graph)
            if opts.get("ConvertClipToRelu", False):
                from onnxsim.quark_convert import convert_clip_to_relu

                q = convert_clip_to_relu(q, lambda n: n.name not in skip_names)
            if opts.get("DedicatedQDQPair", False):
                from onnxsim.quark_preset_graphs import dedicate_qdq_pairs

                keep = set(skip_names)
                q = dedicate_qdq_pairs(
                    q,
                    {
                        n.name
                        for n in work.graph.node
                        if n.name
                        and n.name not in keep
                        and (scope_types is None or n.op_type in scope_types)
                    },
                )
            if self._keeps_constants() and not npu_cnn:
                # (the Constant nodes lead the list, as in Quark's graph, whose own
                # ``topological_sort`` ends every flow)
                from onnxsim.quark_marking import quark_qdq_sorted

                q = quark_qdq_sorted(q)
            if ext and mixed_algo is not None:
                # (the extended quantizer converts its Q/DQ nodes before
                # AutoMixprecision re-quantizes layers with standard ones)
                q = _qdq_to_ms_domain(q)
            return q

        bias_post = self._base_bias_post(work, act, wt)
        if mixed_algo is not None:
            quantized = self._auto_mixprecision(
                work,
                calibration,
                act_dtype,
                act,
                exclude,
                mixed_algo,
                qkw,
                (lambda q: finish(bias_post(q) if bias_post is not None else q)),
            )
        else:
            quantized = quantize_full_qdq(work, **qkw)
            if bias_post is not None:
                quantized = bias_post(quantized)
            quantized = finish(quantized)

        # Post-quantization passes, which compare against the float model.
        if "adaquant" in by_name and by_name["adaquant"].params.get("legacy_engine"):
            from onnxsim.adaquant import apply_adaquant

            params = by_name["adaquant"].params
            before = quantized.SerializeToString()
            quantized = apply_adaquant(
                float_model,
                quantized,
                calibration_data=calibration,
                **{
                    kwarg: params[key]
                    for key, kwarg in _ADAQUANT_PARAMS.items()
                    if key in params
                },
            )
            if quantized.SerializeToString() == before:
                self._approx(
                    "the legacy AdaQuant engine found no layer it can optimize "
                    "(it needs onnxsim's int8-weight / uint8-activation layout): "
                    "the model is unchanged"
                )
        elif "adaquant" in by_name:
            quantized = self._finetune(
                "adaquant", float_model, quantized, calibration, by_name["adaquant"]
            )
        if "adaround" in by_name:
            quantized = self._finetune(
                "adaround", float_model, quantized, calibration, by_name["adaround"]
            )
        if "gptq" in by_name:
            quantized = self._gptq(float_model, quantized, calibration, by_name["gptq"])
        if "bias_correction" in by_name:
            from onnxsim.quark_bias_correction import correct_bias_quark

            cm = self._calib_method(act)
            quantized = correct_bias_quark(
                float_model,
                quantized,
                calibration,
                activation_symmetric=bool(opts.get("ActivationSymmetric", act_sym)),
                method=(
                    "pof2"
                    if act.pof2 or cm in ("minmse_pof2", "nonoverflow")
                    else "minmax"
                    if cm.startswith(("minmax", "percentile", "onnxsim:percentile"))
                    else "none"
                ),
                quark_scale=not opts.get("BiasCorrectionStoredScale", False),
            )
        if ext and mixed_algo is None:
            quantized = _qdq_to_ms_domain(quantized)
        return _register_domains(quantized)


__all__ = [
    "AdaQuantConfig",
    "AdaRoundConfig",
    "AlgoConfig",
    "AutoMixprecisionConfig",
    "BFP16Spec",
    "BFloat16Spec",
    "BiasCorrectionConfig",
    "CalibMethod",
    "CLEConfig",
    "Config",
    "Float16Spec",
    "GPTQConfig",
    "Int16Spec",
    "Int8Spec",
    "MX4Spec",
    "MX6Spec",
    "MX9Spec",
    "MXFP4E2M1Spec",
    "MXFP6E2M3Spec",
    "MXFP6E3M2Spec",
    "MXFP8E4M3Spec",
    "MXFP8E5M2Spec",
    "MXInt8Spec",
    "ModelQuantizer",
    "QConfig",
    "QLayerConfig",
    "QSpec",
    "QuarotConfig",
    "SmoothQuantConfig",
    "UInt16Spec",
    "UInt8Spec",
    "XInt8Spec",
    "XUInt8Spec",
]
