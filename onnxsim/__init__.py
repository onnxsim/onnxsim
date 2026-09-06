from onnxsim.accuracy import (
    DEFAULT_QUANTIZATION_CANDIDATES,
    AccuracyDropReport,
    OutputAccuracyStats,
    QuantizationConfig,
    QuantizationRecommendation,
    measure_accuracy_drop,
    quantize,
    quantize_auto,
    recommend_quantization,
)
from onnxsim.adaquant import apply_adaquant
from onnxsim.adaround import apply_adaround
from onnxsim.adpq import quantize_weight_only_adpq
from onnxsim.affinequant import apply_affinequant
from onnxsim.any_precision_llm import apply_any_precision_llm
from onnxsim.aqlm import quantize_weight_only_aqlm
from onnxsim.attention_quantization import apply_attention_quantization
from onnxsim.autoquant import AutoQuantResult, auto_quantize_int4
from onnxsim.autoround import apply_autoround
from onnxsim.awq import apply_awq
from onnxsim.bias_correction import correct_bias
from onnxsim.billm import quantize_weight_only_billm
from onnxsim.brecq import apply_brecq
from onnxsim.bwa_ptq import apply_bwa_ptq
from onnxsim.calibration import (
    calibrate,
    generate_random_calibration_data,
    load_huggingface_calibration_data,
    quantize_qoperator,
    quantize_qoperator_activation,
    quantize_qoperator_concat,
    quantize_qoperator_elementwise,
    quantize_qoperator_gemm,
    quantize_qoperator_pool,
    quantize_qoperator_softmax,
    quantize_qoperator_where,
    quantize_static,
    quantize_static_int16,
)
from onnxsim.coreml_export import export_coreml
from onnxsim.d2quant import apply_dac, apply_dsq
from onnxsim.diffusion_export import export_diffusion_model
from onnxsim.double_quantization import apply_double_quantization
from onnxsim.drop_by_drop import (
    quantize_weight_only_drop_by_drop,
    select_drop_by_drop_prefix,
)
from onnxsim.duquant import apply_duquant
from onnxsim.easyquant import apply_easyquant
from onnxsim.embedding_quantization import (
    quantize_embedding_binary,
    quantize_embedding_int8,
)
from onnxsim.finetune import apply_pruning_finetune
from onnxsim.flexround import apply_flexround
from onnxsim.foem import apply_foem
from onnxsim.fptq import apply_fptq
from onnxsim.gear import apply_gear
from onnxsim.gguf_reconstruct import (
    UnsupportedArchitectureError,
    reconstruct_gguf_graph,
)
from onnxsim.gptq import apply_gptq
from onnxsim.gptvq import quantize_weight_only_gptvq
from onnxsim.hf_reconstruct import read_hf_config, reconstruct_hf_graph
from onnxsim.hqq import quantize_weight_only_int4_hqq
from onnxsim.ibert_gelu import apply_ibert_gelu
from onnxsim.ibert_softmax import apply_ibert_softmax
from onnxsim.icquant import icquant_metadata_bits, quantize_weight_only_icquant
from onnxsim.if4_quantization import quantize_weight_only_if4
from onnxsim.intactkv import apply_intactkv
from onnxsim.kbvq_moe import apply_kbvq_moe
from onnxsim.kmeans_quantization import quantize_weight_only_kmeans
from onnxsim.kv_cache_quantization import quantize_kv_cache
from onnxsim.llm_fp4 import FP4_FORMATS, quantize_weight_only_llm_fp4
from onnxsim.llm_int8 import apply_llm_int8
from onnxsim.lo_bcq import quantize_weight_only_lo_bcq
from onnxsim.low_rank_compensation import apply_low_rank_compensation
from onnxsim.memory_planning import (
    MemoryPlan,
    annotate_memory_plan,
    plan_activation_memory,
    print_memory_plan,
)
from onnxsim.mixed_precision import apply_mixed_precision_quantization
from onnxsim.mlir_export import export_mlir
from onnxsim.moequant import apply_moequant
from onnxsim.mx_quantization import MXFP4_CODEBOOK, quantize_weight_only_mxfp4
from onnxsim.nf4 import NF4_CODEBOOK, quantize_weight_only_nf4
from onnxsim.norm_tweaking import apply_norm_tweaking
from onnxsim.olive import quantize_weight_only_olive
from onnxsim.omniquant import apply_omniquant
from onnxsim.onnx_simplifier import (
    apply_any_precision_llm_cpp,
    apply_attention_head_pruning_cpp,
    apply_attention_head_wanda_pruning_cpp,
    apply_double_quantization_cpp,
    apply_embedding_vocab_magnitude_pruning_cpp,
    apply_embedding_vocab_pruning_cpp,
    apply_moe_expert_channel_pruning_cpp,
    apply_moe_whole_expert_pruning_cpp,
    apply_qmoe_expert_channel_pruning_cpp,
    apply_qmoe_whole_expert_pruning_cpp,
    apply_quarot_cpp,
    apply_sparsegpt_pruning_cpp,
    apply_structured_pruning_cpp,
    apply_structured_wanda_pruning_cpp,
    apply_transformer_block_pruning_cpp,
    apply_wanda_pruning_cpp,
    cross_layer_equalize,
    export_gguf,
    export_safetensors,
    import_gguf,
    import_gguf_weights,
    import_onnx_schemas,
    import_safetensors,
    load_model,
    main,
    prune_magnitude_cpp,
    quantize_attention_dynamic,
    quantize_bf16,
    quantize_dynamic,
    quantize_dynamic_matmul_integer_to_float,
    quantize_fp8,
    quantize_fp16,
    quantize_ternary,
    quantize_weight_only,
    quantize_weight_only_int4,
    quantize_weight_only_int8_block,
    quantize_weight_only_int16,
    quantize_weight_only_matmul_nbits,
    quantize_weight_only_mxfp4_cpp,
    read_gguf_metadata,
    simplify,
)
from onnxsim.optimize_pipeline import (
    OptimizationPipelineResult,
    apply_optimization_pipeline,
)
from onnxsim.ort_matmul_nbits_workaround import workaround_ort_matmul_nbits_axis0_bug
from onnxsim.outlier_suppression import apply_outlier_suppression
from onnxsim.outlier_suppression_plus import apply_outlier_suppression_plus
from onnxsim.owq import apply_owq
from onnxsim.paroquant import apply_paroquant
from onnxsim.pb_llm import quantize_weight_only_pb_llm
from onnxsim.ppq_integration import quantize_with_ppq
from onnxsim.precision_estimator import (
    ModelQuantizationEstimate,
    estimate_model_quantization_drop,
    estimate_quantization_precision,
)
from onnxsim.pruning import (
    EmbeddingPruningResult,
    PruningLayerSensitivity,
    PruningSensitivityReport,
    analyze_pruning_sensitivity,
    apply_attention_head_pruning,
    apply_attention_head_wanda_pruning,
    apply_embedding_vocab_magnitude_pruning,
    apply_embedding_vocab_pruning,
    apply_magnitude_pruning,
    apply_moe_expert_channel_pruning,
    apply_moe_whole_expert_pruning,
    apply_qmoe_expert_channel_pruning,
    apply_qmoe_whole_expert_pruning,
    apply_sparsegpt_pruning,
    apply_structured_pruning,
    apply_structured_pruning_dynamic_quantize_conv,
    apply_structured_pruning_dynamic_quantize_matmul,
    apply_structured_pruning_matmul_block_quantized_fp4,
    apply_structured_pruning_matmul_block_quantized_fp8,
    apply_structured_pruning_matmul_bnb4,
    apply_structured_pruning_matmul_nbits,
    apply_structured_pruning_qdq,
    apply_structured_pruning_qoperator,
    apply_structured_wanda_pruning,
    apply_transformer_block_pruning,
    apply_wanda_pruning,
    weight_sparsity,
)
from onnxsim.qoq import apply_smooth_attention, quantize_weight_only_qoq
from onnxsim.qronos import apply_qronos
from onnxsim.quantease import apply_quantease
from onnxsim.quarot import apply_quarot
from onnxsim.quip_sharp import apply_quip_sharp
from onnxsim.rotatekv import apply_rotatekv
from onnxsim.rptq import apply_rptq_reorder
from onnxsim.slim_llm import apply_slim_llm
from onnxsim.smoothquant import apply_smoothquant
from onnxsim.spinquant import apply_spinquant
from onnxsim.spqr import quantize_weight_only_spqr
from onnxsim.squeezellm import quantize_weight_only_squeezellm
from onnxsim.svdquant import apply_svdquant
from onnxsim.tensorrt_sparsity import convert_matmul_to_gemm
from onnxsim.tesseraq import apply_tesseraq
from onnxsim.tflite_export import export_tflite
from onnxsim.transformers_export import export_transformers_model
from onnxsim.xnnpack_codegen import export_xnnpack_c, generate_xnnpack_c
from onnxsim.zeroquant import apply_zeroquant

from .version import version as __version__

__all__ = [
    "simplify",
    "quantize",
    "QuantizationConfig",
    "recommend_quantization",
    "quantize_auto",
    "QuantizationRecommendation",
    "DEFAULT_QUANTIZATION_CANDIDATES",
    "cross_layer_equalize",
    "correct_bias",
    "apply_adaround",
    "apply_adaquant",
    "apply_tesseraq",
    "apply_pruning_finetune",
    "apply_autoround",
    "auto_quantize_int4",
    "AutoQuantResult",
    "apply_optimization_pipeline",
    "OptimizationPipelineResult",
    "apply_awq",
    "apply_attention_quantization",
    "apply_gptq",
    "apply_qronos",
    "quantize_weight_only_qoq",
    "apply_smooth_attention",
    "apply_quantease",
    "apply_flexround",
    "apply_foem",
    "apply_brecq",
    "apply_fptq",
    "apply_owq",
    "apply_bwa_ptq",
    "quantize_with_ppq",
    "apply_smoothquant",
    "apply_rptq_reorder",
    "apply_outlier_suppression_plus",
    "apply_outlier_suppression",
    "apply_dsq",
    "apply_dac",
    "apply_llm_int8",
    "apply_low_rank_compensation",
    "apply_svdquant",
    "apply_norm_tweaking",
    "MemoryPlan",
    "plan_activation_memory",
    "print_memory_plan",
    "annotate_memory_plan",
    "apply_mixed_precision_quantization",
    "apply_slim_llm",
    "apply_quip_sharp",
    "apply_quarot",
    "apply_quarot_cpp",
    "apply_any_precision_llm_cpp",
    "apply_duquant",
    "apply_easyquant",
    "apply_zeroquant",
    "apply_spinquant",
    "apply_paroquant",
    "apply_rotatekv",
    "apply_omniquant",
    "apply_affinequant",
    "apply_any_precision_llm",
    "apply_magnitude_pruning",
    "prune_magnitude_cpp",
    "apply_wanda_pruning",
    "apply_wanda_pruning_cpp",
    "apply_sparsegpt_pruning",
    "apply_sparsegpt_pruning_cpp",
    "apply_structured_pruning",
    "apply_structured_pruning_cpp",
    "apply_structured_pruning_qdq",
    "apply_structured_pruning_qoperator",
    "apply_structured_pruning_matmul_nbits",
    "apply_structured_pruning_matmul_bnb4",
    "apply_structured_pruning_dynamic_quantize_matmul",
    "apply_structured_pruning_dynamic_quantize_conv",
    "apply_structured_pruning_matmul_block_quantized_fp8",
    "apply_structured_pruning_matmul_block_quantized_fp4",
    "apply_structured_wanda_pruning",
    "apply_structured_wanda_pruning_cpp",
    "apply_attention_head_pruning",
    "apply_attention_head_pruning_cpp",
    "apply_attention_head_wanda_pruning",
    "apply_attention_head_wanda_pruning_cpp",
    "apply_moe_expert_channel_pruning",
    "apply_moe_expert_channel_pruning_cpp",
    "apply_moe_whole_expert_pruning",
    "apply_moe_whole_expert_pruning_cpp",
    "apply_qmoe_expert_channel_pruning",
    "apply_qmoe_expert_channel_pruning_cpp",
    "apply_qmoe_whole_expert_pruning",
    "apply_qmoe_whole_expert_pruning_cpp",
    "apply_moequant",
    "apply_kbvq_moe",
    "apply_embedding_vocab_pruning",
    "apply_embedding_vocab_pruning_cpp",
    "apply_embedding_vocab_magnitude_pruning",
    "apply_embedding_vocab_magnitude_pruning_cpp",
    "EmbeddingPruningResult",
    "apply_transformer_block_pruning",
    "apply_transformer_block_pruning_cpp",
    "weight_sparsity",
    "analyze_pruning_sensitivity",
    "PruningSensitivityReport",
    "PruningLayerSensitivity",
    "convert_matmul_to_gemm",
    "workaround_ort_matmul_nbits_axis0_bug",
    "quantize_attention_dynamic",
    "quantize_dynamic",
    "quantize_dynamic_matmul_integer_to_float",
    "quantize_ternary",
    "quantize_weight_only",
    "quantize_weight_only_int4",
    "quantize_weight_only_billm",
    "quantize_weight_only_pb_llm",
    "quantize_weight_only_int4_hqq",
    "apply_ibert_gelu",
    "apply_ibert_softmax",
    "quantize_weight_only_icquant",
    "icquant_metadata_bits",
    "quantize_weight_only_kmeans",
    "quantize_weight_only_lo_bcq",
    "quantize_weight_only_nf4",
    "NF4_CODEBOOK",
    "quantize_weight_only_mxfp4",
    "quantize_weight_only_if4",
    "MXFP4_CODEBOOK",
    "quantize_weight_only_llm_fp4",
    "FP4_FORMATS",
    "quantize_weight_only_squeezellm",
    "quantize_weight_only_aqlm",
    "quantize_weight_only_drop_by_drop",
    "select_drop_by_drop_prefix",
    "quantize_weight_only_gptvq",
    "quantize_weight_only_spqr",
    "quantize_weight_only_adpq",
    "quantize_weight_only_olive",
    "apply_double_quantization",
    "apply_double_quantization_cpp",
    "quantize_kv_cache",
    "apply_intactkv",
    "apply_gear",
    "quantize_embedding_binary",
    "quantize_embedding_int8",
    "quantize_weight_only_matmul_nbits",
    "quantize_weight_only_int8_block",
    "quantize_weight_only_int16",
    "quantize_weight_only_mxfp4_cpp",
    "quantize_static",
    "quantize_static_int16",
    "quantize_qoperator",
    "quantize_qoperator_elementwise",
    "quantize_qoperator_activation",
    "quantize_qoperator_concat",
    "quantize_qoperator_softmax",
    "quantize_qoperator_pool",
    "quantize_qoperator_where",
    "quantize_qoperator_gemm",
    "quantize_fp16",
    "quantize_bf16",
    "quantize_fp8",
    "calibrate",
    "generate_random_calibration_data",
    "load_huggingface_calibration_data",
    "estimate_quantization_precision",
    "ModelQuantizationEstimate",
    "estimate_model_quantization_drop",
    "measure_accuracy_drop",
    "AccuracyDropReport",
    "OutputAccuracyStats",
    "main",
    "import_onnx_schemas",
    "export_safetensors",
    "import_safetensors",
    "export_gguf",
    "import_gguf",
    "import_gguf_weights",
    "load_model",
    "read_gguf_metadata",
    "reconstruct_gguf_graph",
    "reconstruct_hf_graph",
    "read_hf_config",
    "UnsupportedArchitectureError",
    "export_transformers_model",
    "export_diffusion_model",
    "export_mlir",
    "export_coreml",
    "export_tflite",
    "generate_xnnpack_c",
    "export_xnnpack_c",
    "__version__",
]
