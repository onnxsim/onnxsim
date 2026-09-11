/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * A minimal, stable C ABI over the onnxsim C++ core. It exists so that other
 * languages (e.g. Rust, Go, C) can drive the simplifier without dealing with
 * C++ name mangling, exceptions, or the onnx::ModelProto type across the FFI
 * boundary. Models are exchanged as serialized ONNX ModelProto bytes, exactly
 * like the Python binding does.
 */
#ifndef ONNXSIM_C_API_H_
#define ONNXSIM_C_API_H_

#include <stddef.h>

#include "dlpack/dlpack.h"

#if defined(_WIN32)
#if defined(ONNXSIM_C_API_BUILD)
#define ONNXSIM_C_API __declspec(dllexport)
#else
#define ONNXSIM_C_API __declspec(dllimport)
#endif
#else
#define ONNXSIM_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes for every fallible entry point. */
typedef enum OnnxsimStatus {
  ONNXSIM_OK = 0,
  ONNXSIM_ERROR = 1,
} OnnxsimStatus;

/*
 * Optional custom graph-rewriter callback.
 *
 * When supplied to onnxsim_simplify / onnxsim_simplify_path it is run inside
 * onnxsim's simplification fixed point, interleaved with the built-in
 * optimizer, shape inference and constant folding, so a rewrite can unlock
 * further simplification and vice versa. It mirrors the C++ `GraphRewriter` and
 * the Python `custom_rewriter` parameter, exchanging models as serialized ONNX
 * ModelProto bytes across the C boundary.
 *
 * On each round it is called with the current model in `in_model_data`
 * (`in_model_size` bytes) and must return:
 *   - a value > 0 : the model was rewritten.
 * `*out_model_data`/`*out_model_size` must be set to a newly allocated buffer
 * holding the rewritten serialized ModelProto. onnxsim parses it and then, if a
 * free callback was supplied, calls it with exactly that pointer and size to
 * release the buffer.
 *   - 0           : nothing was rewritten this round. The out-parameters are
 *     ignored and onnxsim keeps the model it already has (skipping a copy).
 *   - a value < 0 : the callback failed; simplification aborts with
 *     ONNXSIM_ERROR.
 *
 * `user_data` is passed through untouched on every call (both the rewrite and
 * the free callback). Passing a NULL rewrite callback disables the feature and
 * reproduces the previous behaviour exactly.
 */
typedef int (*OnnxsimRewriteFn)(void* user_data, const void* in_model_data,
                                size_t in_model_size, void** out_model_data,
                                size_t* out_model_size);

/*
 * Releases a buffer produced by an OnnxsimRewriteFn. Called with the same
 * `user_data` and with the `out_model_data`/`out_model_size` the rewrite
 * callback returned, once onnxsim has finished parsing it. May be NULL, in
 * which case onnxsim never frees the rewriter's output (the callback owns it).
 */
typedef void (*OnnxsimRewriteFreeFn)(void* user_data, void* model_data,
                                     size_t model_size);

/*
 * Optional custom constant-folding executor callback (the embeddability seam).
 *
 * When supplied to onnxsim_simplify_with_executor it REPLACES the built-in
 * ONNX Runtime constant folder: onnxsim hands each fold group's throwaway
 * sub-model plus its input tensors to this callback, which evaluates it in the
 * host's own ONNX runtime and returns the resulting tensors. This lets onnxsim
 * be embedded in another compiler/runtime stack (a different ORT build, IREE,
 * TVM, a hardware vendor's runtime) without depending on the vendored ORT, and
 * without serializing tensors to TensorProto: tensors cross as DLPack
 * DLManagedTensors (see third_party/dlpack/dlpack.h, docs/dlpack-executor.md).
 *
 * Call contract, per fold group:
 *   - `model_data`/`model_size`: the sub-model as a serialized ONNX ModelProto.
 *     Its graph.input() are fed positionally by `inputs`; produce one output
 *     per graph.output(), in that order.
 *   - `inputs`/`num_inputs`: input tensors, BORROWED for the duration of the
 *     call. The callback must NOT free them or retain them past return.
 *   - On success return 0 and set `*out_outputs` to an array of `num` owned
 *     DLManagedTensor* (set `*out_num_outputs = num`). onnxsim takes ownership
 *     of each tensor and releases it via that tensor's own DLPack `deleter`; it
 *     then calls the paired OnnxsimExecuteFreeFn (if any) to release the array
 *     container itself. All tensors must be CPU (kDLCPU), contiguous, and of a
 *     dtype onnxsim supports (float16/float/double/bfloat16, 8/16/32/64-bit
 *     ints, bool).
 *   - Return non-zero to signal failure; simplification aborts with
 *     ONNXSIM_ERROR.
 *
 * `user_data` is passed through untouched on every call.
 */
typedef int (*OnnxsimExecuteFn)(void* user_data, const void* model_data,
                                size_t model_size,
                                const DLManagedTensor* const* inputs,
                                size_t num_inputs,
                                DLManagedTensor*** out_outputs,
                                size_t* out_num_outputs);

/*
 * Releases the output array a OnnxsimExecuteFn returned (the container, not the
 * tensors -- each DLManagedTensor is released through its own deleter). Called
 * with the same `user_data` and the exact `outputs`/`num_outputs` the execute
 * callback produced. May be NULL, in which case onnxsim does not free the array
 * (e.g. when the callback returns a reused/static buffer).
 */
typedef void (*OnnxsimExecuteFreeFn)(void* user_data, DLManagedTensor** outputs,
                                     size_t num_outputs);

/*
 * Simplify a model given as a serialized ONNX ModelProto.
 *
 * skip_optimizers semantics mirror the C++/Python API:
 *   - skip_optimizers_is_null != 0  => skip ALL optimizer passes (no graph
 *     optimization is performed); `skip_optimizers`/`num_skip_optimizers` are
 *     ignored.
 *   - skip_optimizers_is_null == 0  => run every fuse/elimination pass EXCEPT
 *     the `num_skip_optimizers` passes named in `skip_optimizers` (pass 0 to
 *     run all of them).
 *
 * constant_folding / shape_inference are treated as booleans (0 = false).
 * tensor_size_threshold bounds the byte size of tensors produced by constant
 * folding that are kept as initializers.
 *
 * target_opset_version, when > 0, converts the model to that opset version of
 * the default ONNX domain (using onnx's version converter) before simplifying;
 * a value <= 0 leaves the opset version unchanged.
 *
 * extra_optimizers/num_extra_optimizers name onnx-optimizer passes to run in
 * ADDITION to the default fuse/elimination set (plus whatever
 * skip_optimizers left in) -- the counterpart to skip_optimizers, and how a
 * pass registered outside the default set (typically PassType::Other, e.g. a
 * defusion that trades a backend-specific op for a more portable one) gets
 * opted into. num_extra_optimizers == 0 behaves exactly like the C++ core's
 * ``std::nullopt`` (no extra passes; also a no-op when skip_optimizers_is_null
 * != 0, since that already disables optimization entirely). An unknown pass
 * name is not validated here -- it throws from onnx-optimizer's own pass
 * registry lookup, surfaced as ONNXSIM_ERROR with a message in *out_error,
 * rather than being silently ignored. See onnxsim_list_other_optimizers for
 * the set of names this can name.
 *
 * rewrite_fn, when non-NULL, is a custom graph rewriter run inside the
 * simplification fixed point (see OnnxsimRewriteFn). rewrite_free_fn releases
 * the buffers it produces (may be NULL). rewrite_user_data is passed through to
 * both callbacks untouched. Pass a NULL rewrite_fn to disable the feature.
 *
 * On ONNXSIM_OK, *out_data / *out_size receive a newly allocated buffer holding
 * the serialized simplified ModelProto; release it with onnxsim_free_buffer.
 * On ONNXSIM_ERROR, *out_error receives a newly allocated, NUL-terminated
 * message; release it with onnxsim_free_string. Either out_* pointer may be
 * NULL if the caller does not want that value.
 */
ONNXSIM_C_API OnnxsimStatus
onnxsim_simplify(const void* model_data, size_t model_size,
                 const char* const* skip_optimizers, size_t num_skip_optimizers,
                 int skip_optimizers_is_null, int constant_folding,
                 int shape_inference, size_t tensor_size_threshold,
                 int target_opset_version, const char* const* extra_optimizers,
                 size_t num_extra_optimizers, OnnxsimRewriteFn rewrite_fn,
                 OnnxsimRewriteFreeFn rewrite_free_fn, void* rewrite_user_data,
                 void** out_data, size_t* out_size, char** out_error);

/*
 * Same as onnxsim_simplify, but drives constant folding through a host-provided
 * executor callback instead of the built-in ONNX Runtime (see
 * OnnxsimExecuteFn). This is the seam for embedding onnxsim in another
 * ONNX-based stack. Passing a NULL execute_fn falls back to the built-in
 * executor, making this a drop-in superset of onnxsim_simplify. execute_free_fn
 * (may be NULL) releases each output array; execute_user_data is passed through
 * to both callbacks. All other parameters and the out_* contract match
 * onnxsim_simplify, including extra_optimizers/num_extra_optimizers.
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_simplify_with_executor(
    const void* model_data, size_t model_size,
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null, int constant_folding, int shape_inference,
    size_t tensor_size_threshold, int target_opset_version,
    const char* const* extra_optimizers, size_t num_extra_optimizers,
    OnnxsimRewriteFn rewrite_fn, OnnxsimRewriteFreeFn rewrite_free_fn,
    void* rewrite_user_data, OnnxsimExecuteFn execute_fn,
    OnnxsimExecuteFreeFn execute_free_fn, void* execute_user_data,
    void** out_data, size_t* out_size, char** out_error);

/*
 * Same as onnxsim_simplify, but also applies a set of data-driven rewrite rules
 * inside the simplification fixed point. Each rule is a (pattern, replacement)
 * pair of serialized ONNX FunctionProto: the pattern's inputs are wildcards
 * binding to graph values, its body is the subgraph to match, and its outputs
 * are rewired to the replacement's outputs. This is the same
 * FunctionProto-based rewriter the Python and Rust bindings expose, so a rule
 * set authored once (e.g. via onnx.parser.parse_function) works from any
 * binding without depending on onnxscript.
 *
 * `pattern_data[i]`/`pattern_sizes[i]` and `replacement_data[i]`/
 * `replacement_sizes[i]` describe rule `i`, for `i` in [0, num_rules). Passing
 * num_rules == 0 behaves exactly like onnxsim_simplify. All other parameters
 * match onnxsim_simplify, including extra_optimizers/num_extra_optimizers.
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_simplify_with_rules(
    const void* model_data, size_t model_size,
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null, int constant_folding, int shape_inference,
    size_t tensor_size_threshold, int target_opset_version,
    const void* const* pattern_data, const size_t* pattern_sizes,
    const void* const* replacement_data, const size_t* replacement_sizes,
    size_t num_rules, const char* const* extra_optimizers,
    size_t num_extra_optimizers, void** out_data, size_t* out_size,
    char** out_error);

/*
 * Same as onnxsim_simplify, but reads the input model from `in_path` and writes
 * the simplified model to `out_path`. On failure, *out_error receives a message
 * (free with onnxsim_free_string).
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_simplify_path(
    const char* in_path, const char* out_path,
    const char* const* skip_optimizers, size_t num_skip_optimizers,
    int skip_optimizers_is_null, int constant_folding, int shape_inference,
    size_t tensor_size_threshold, int target_opset_version,
    const char* const* extra_optimizers, size_t num_extra_optimizers,
    OnnxsimRewriteFn rewrite_fn, OnnxsimRewriteFreeFn rewrite_free_fn,
    void* rewrite_user_data, char** out_error);

/*
 * Parses `text` -- ONNX's textual IR syntax, the same format
 * onnx.parser.parse_model reads in Python (see
 * https://onnx.ai/onnx/repo-docs/Syntax.html and onnx/defs/parser.h's
 * OnnxParser) -- into a serialized ONNX ModelProto. Lets a binding with no
 * protobuf library of its own (e.g. this repo's Rust/Ruby samples) build a
 * test or example model far more readably than hand-encoding protobuf bytes,
 * the same way this codebase's own Python tests prefer onnx.parser over
 * onnx.helper.make_node/make_graph/make_model chains.
 *
 * On ONNXSIM_OK, *out_data/*out_size receive a newly allocated buffer holding
 * the serialized ModelProto; release it with onnxsim_free_buffer. On
 * ONNXSIM_ERROR (a syntax error, most commonly), *out_error receives a
 * newly allocated message describing where parsing failed; release it with
 * onnxsim_free_string. Either out_* pointer may be NULL if the caller does
 * not want that value.
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_parse_model_text(const char* text,
                                                     void** out_data,
                                                     size_t* out_size,
                                                     char** out_error);

/*
 * Return the names of all available fuse/elimination optimizer passes as a
 * single NUL-terminated string with one pass name per line ('\n' separated).
 * Returns NULL on allocation failure. Release with onnxsim_free_string.
 */
ONNXSIM_C_API char* onnxsim_list_optimizers(void);

/*
 * Return the names of the optimizer passes registered but NOT in the default
 * fuse/elimination set (typically PassType::Other, e.g.
 * defuse_matmul_integer_to_float)
 * -- the names valid for extra_optimizers/num_extra_optimizers on the
 * onnxsim_simplify* functions above. Same format/ownership contract as
 * onnxsim_list_optimizers: one name per line, NUL-terminated, NULL on
 * allocation failure, release with onnxsim_free_string.
 */
ONNXSIM_C_API char* onnxsim_list_other_optimizers(void);

/*
 * Render a human-readable diff between an original and a simplified model, both
 * given as serialized ONNX ModelProto bytes. The report is an ASCII table
 * comparing op counts and model size, the same information the Python CLI
 * prints as "the difference" after simplifying. It lets the non-Python bindings
 * show a before/after summary without re-implementing the analysis.
 *
 * On ONNXSIM_OK, *out_text receives a newly allocated, NUL-terminated string
 * holding the table; release it with onnxsim_free_string. On ONNXSIM_ERROR,
 * *out_error receives a newly allocated message; release it the same way.
 * Either out_* pointer may be NULL if the caller does not want that value.
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_model_info_diff(const void* original_data,
                                                    size_t original_size,
                                                    const void* simplified_data,
                                                    size_t simplified_size,
                                                    char** out_text,
                                                    char** out_error);

/*
 * Render a node- and value-level diff between an original and a simplified
 * model, both given as serialized ONNX ModelProto bytes: which nodes/values
 * were removed, added, or changed (matched by output tensor name), e.g. a
 * Conv whose bias input got folded away. Complementary to
 * onnxsim_model_info_diff, which reports op-count/size/MACs aggregates rather
 * than the specific nodes and values involved.
 *
 * On ONNXSIM_OK, *out_text receives a newly allocated, NUL-terminated string
 * holding the report; release it with onnxsim_free_string. On ONNXSIM_ERROR,
 * *out_error receives a newly allocated message; release it the same way.
 * Either out_* pointer may be NULL if the caller does not want that value.
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_graph_diff(const void* original_data,
                                               size_t original_size,
                                               const void* simplified_data,
                                               size_t simplified_size,
                                               char** out_text,
                                               char** out_error);

/*
 * Export a model (given as a serialized ONNX ModelProto) to a standalone
 * safetensors archive at `out_path`: every initializer's bytes move into the
 * archive with real, byte-accurate offsets -- openable by the `safetensors`
 * Python package / HF tooling with no onnxsim involved -- and the graph
 * itself is embedded alongside them, so `out_path` alone is both the
 * model's weights and its graph (see onnxsim/tensor_pool_bridge.h's
 * SaveModelAsSafetensorsStandalone). Reload it with
 * onnxsim_import_safetensors.
 *
 * On ONNXSIM_ERROR, *out_error receives a newly allocated message; release it
 * with onnxsim_free_string. out_error may be NULL.
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_export_safetensors(const void* model_data,
                                                       size_t model_size,
                                                       const char* out_path,
                                                       char** out_error);

/*
 * Import a standalone safetensors archive at `in_path` (as produced by
 * onnxsim_export_safetensors, or any other tool following the same
 * self-describing-archive convention -- an embedded "model.onnx" entry
 * alongside the tensors) back into an ordinary, fully in-memory ONNX
 * ModelProto.
 *
 * On ONNXSIM_OK, *out_data/*out_size receive a newly allocated buffer holding
 * the serialized ModelProto; release it with onnxsim_free_buffer. Fails with
 * ONNXSIM_ERROR (and a message in *out_error, release with
 * onnxsim_free_string) if the archive has no embedded onnxsim model -- e.g. a
 * plain weights-only safetensors file with no graph to import -- or on any
 * other read/parse failure. Either out_* pointer may be NULL if the caller
 * does not want that value.
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_import_safetensors(const char* in_path,
                                                       void** out_data,
                                                       size_t* out_size,
                                                       char** out_error);

/*
 * GGUF counterpart of onnxsim_export_safetensors; see onnxsim/
 * tensor_pool_gguf_bridge.h's SaveModelAsGGUFStandalone. Same contract.
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_export_gguf(const void* model_data,
                                                size_t model_size,
                                                const char* out_path,
                                                char** out_error);

/*
 * GGUF counterpart of onnxsim_import_safetensors; see onnxsim/
 * tensor_pool_gguf_bridge.h's LoadModelFromGGUF. Same contract -- including
 * failing with ONNXSIM_ERROR when the archive has no embedded model. Any
 * OTHER pooled tensor LoadModelFromGGUF could not hydrate (e.g. a quantized
 * weight) is silently left un-hydrated in the returned model rather than
 * surfaced through this narrow C ABI; use the C++/Python/Rust bindings
 * directly for that detail if it matters to the caller.
 */
ONNXSIM_C_API OnnxsimStatus onnxsim_import_gguf(const char* in_path,
                                                void** out_data,
                                                size_t* out_size,
                                                char** out_error);

/* Free a buffer returned via an out_data parameter. NULL is ignored. */
ONNXSIM_C_API void onnxsim_free_buffer(void* data);

/* Free a string returned via an out_error parameter or onnxsim_list_optimizers.
 * NULL is ignored. */
ONNXSIM_C_API void onnxsim_free_string(char* data);

#ifdef __cplusplus
}
#endif

#endif /* ONNXSIM_C_API_H_ */
