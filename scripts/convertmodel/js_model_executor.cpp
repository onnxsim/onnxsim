#include "js_model_executor.h"

#if defined(__EMSCRIPTEN__) && defined(ONNXSIM_WASM_HOOKABLE_EXECUTOR)

#include <emscripten/val.h>

#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "dlpack_bridge.h"

// wasm is always little-endian; the (de)serialization below memcpy's typed data
// to/from raw little-endian bytes, so bail loudly if that ever stops holding.
static_assert(std::endian::native == std::endian::little,
              "the onnxruntime-web executor assumes a little-endian target");

namespace {

using emscripten::val;

// The host registers a runner on the Emscripten Module under this name. The
// legacy onnxruntime-web name is accepted as a compatibility alias. The
// crossing is batched: all of
// a fold group's tensors travel in one concatenated byte blob plus one flat
// [dtype, ndim, dims...] metadata array, in both directions. Signature, in JS:
//   async (modelBytes: Uint8Array,
//          inputsData: Uint8Array,      // all feed bytes concatenated
//          inputsMeta: Float64Array)    // [dtype, ndim, dims...] per feed
//     => Promise<{ data: Uint8Array, meta: Float64Array }>  // same layout
// `dtype` is the ONNX TensorProto.DataType enum value; tensor bytes are raw
// little-endian element data. Tensors are positional (no names cross).
constexpr const char *kRunnerProp = "onnxsimModelExecutorRun";
constexpr const char *kLegacyRunnerProp = "onnxsimOrtWebRun";

// Copy `len` bytes at `data` into a fresh JS-owned Uint8Array. Constructing
// `new Uint8Array(view)` from a view over the wasm heap copies the bytes into a
// new ArrayBuffer, so the result stays valid across the Asyncify suspend in
// Run() -- while suspended the wasm heap can grow (ALLOW_MEMORY_GROWTH=1) and
// any view still pointing at the old heap would be detached.
val RawToJsU8Copy(const uint8_t *data, size_t len) {
  val view = val(emscripten::typed_memory_view(len, data));
  return val::global("Uint8Array").new_(view);
}

val StringToJsU8Copy(const std::string &s) {
  return RawToJsU8Copy(reinterpret_cast<const uint8_t *>(s.data()), s.size());
}

// Copy a JS Uint8Array back into a std::string. Called after the await, so the
// destination view is created against the current (post-suspend) heap.
std::string JsU8ToString(const val &u8) {
  const size_t len = u8["length"].as<size_t>();
  std::string out;
  out.resize(len);
  val dest = val(emscripten::typed_memory_view(
      len, reinterpret_cast<uint8_t *>(out.data())));
  dest.call<void>("set", u8);
  return out;
}

// Copy a std::vector<double> into a fresh JS-owned Float64Array (the batched
// input metadata). Like the byte copies, this survives the Asyncify suspend.
val DoublesToJsF64Copy(const std::vector<double> &v) {
  val view = val(emscripten::typed_memory_view(v.size(), v.data()));
  return val::global("Float64Array").new_(view);
}

// Copy a JS Float64Array (the batched output metadata) into a vector. Called
// after the await, against the current heap.
std::vector<double> JsF64ToVector(const val &arr) {
  const size_t len = arr["length"].as<size_t>();
  std::vector<double> out(len);
  if (len != 0) {
    val dest = val(emscripten::typed_memory_view(len, out.data()));
    dest.call<void>("set", arr);
  }
  return out;
}

// A ModelExecutor that evaluates each constant-folding sub-model with
// onnxruntime-web. Tensors cross the executor boundary as DLPack
// DLManagedTensors (see onnxsim.h / dlpack_bridge.h): inputs are borrowed for
// the call, and outputs are returned as freshly owned managed tensors. Across
// the wasm<->JS boundary the payload still has to be copied into JS-owned
// buffers (onnxsim's wasm heap and onnxruntime-web's are separate memories),
// but nothing is serialized to TensorProto -- the contract is ONNX dtype enums
// plus raw little-endian bytes, matching the built-in CppModelExecutor's dtype
// set.
struct JsModelExecutor : public ModelExecutor {
  std::vector<DLManagedTensorPtr>
  Run(const onnx::ModelProto &model,
      const std::vector<const DLManagedTensor *> &inputs) const override {
    // Reach the runner the page registered on the Module.
    val runner = val::module_property(kRunnerProp);
    if (runner.isUndefined() || runner.isNull()) {
      runner = val::module_property(kLegacyRunnerProp);
    }
    if (runner.isUndefined() || runner.isNull()) {
#ifdef ONNXSIM_HAS_ORT
      // The hook is optional in the default WASM build. This preserves the
      // in-module ORT behavior while allowing the page to replace folding with
      // WebGPU, a worker, or a remote runner by installing the callback.
      return GetBuiltinModelExecutor()->Run(model, inputs);
#else
      throw std::runtime_error(
          "hookable WASM executor: Module." + std::string(kRunnerProp) +
          " is not set and this build has no built-in ORT fallback");
#endif
    }

    // Marshal the sub-model and its feeds into JS-owned buffers *before* the
    // await, so nothing points into the wasm heap while it is suspended.
    const std::string model_str = model.SerializeAsString();
    val js_model = StringToJsU8Copy(model_str);

    // Batch the feeds into two buffers that cross the boundary once each: all
    // input bytes concatenated, plus a flat [dtype, ndim, dims...] metadata
    // array (mirrored by ort_executor.mjs). This avoids the per-tensor,
    // per-field embind round trips of building a JS object array. Inputs are
    // positional w.r.t. model.graph().input(); the JS side rebinds them to the
    // session's input names in the same order (DLPack tensors carry no name).
    std::string in_blob;
    std::vector<double> in_meta;
    for (const DLManagedTensor *in : inputs) {
      const DLTensor &t = in->dl_tensor;
      int32_t onnx_dtype;
      if (!onnxsim::dlpack::TryDLToOnnx(t.dtype, &onnx_dtype)) {
        throw std::invalid_argument(
            "onnxruntime-web executor: unsupported input dtype (DLPack code=" +
            std::to_string(t.dtype.code) +
            ", bits=" + std::to_string(t.dtype.bits) + ")");
      }
      in_meta.push_back(onnx_dtype);
      in_meta.push_back(t.ndim);
      for (int32_t d = 0; d < t.ndim; d++) {
        in_meta.push_back(static_cast<double>(t.shape[d]));
      }
      const size_t nbytes =
          static_cast<size_t>(onnxsim::dlpack::NumElements(t.shape, t.ndim)) *
          onnxsim::dlpack::SizeOf(t.dtype);
      const char *base = static_cast<const char *>(t.data) + t.byte_offset;
      in_blob.append(base, nbytes);
    }
    val js_data = StringToJsU8Copy(in_blob);
    val js_meta = DoublesToJsF64Copy(in_meta);

    // Run onnxruntime-web and block on its Promise. val::await() unwinds the
    // wasm stack via Asyncify and resumes here once the Promise settles; a
    // rejected Promise surfaces as a C++ exception.
    val result = runner(js_model, js_data, js_meta).await();

    // The runner returns { data, meta } in the same batched layout, with
    // outputs in graph-output order (which is how RunOps names them
    // positionally). Walk the metadata, slicing each output's bytes out of the
    // blob and rebuilding it as a TensorProto handed to the DLPack boundary as
    // an owning tensor.
    const std::string out_blob = JsU8ToString(result["data"]);
    const std::vector<double> out_meta = JsF64ToVector(result["meta"]);

    std::vector<DLManagedTensorPtr> outputs;
    outputs.reserve(model.graph().output_size());
    size_t m = 0;   // index into out_meta
    size_t off = 0; // byte offset into out_blob
    while (m < out_meta.size()) {
      const int32_t onnx_dtype = static_cast<int32_t>(out_meta[m++]);
      const int32_t ndim = static_cast<int32_t>(out_meta[m++]);
      onnx::TensorProto tp;
      tp.set_data_type(static_cast<onnx::TensorProto::DataType>(onnx_dtype));
      int64_t numel = 1;
      for (int32_t d = 0; d < ndim; d++) {
        const int64_t dim = static_cast<int64_t>(out_meta[m++]);
        tp.add_dims(dim);
        numel *= dim;
      }
      DLDataType dl;
      if (!onnxsim::dlpack::TryOnnxToDL(onnx_dtype, &dl)) {
        throw std::invalid_argument(
            "onnxruntime-web executor: unsupported output dtype " +
            std::to_string(onnx_dtype));
      }
      const size_t nbytes =
          static_cast<size_t>(numel) * onnxsim::dlpack::SizeOf(dl);
      if (off + nbytes > out_blob.size()) {
        throw std::runtime_error(
            "onnxruntime-web executor: output blob shorter than its metadata "
            "implies");
      }
      tp.set_raw_data(out_blob.data() + off, nbytes);
      off += nbytes;
      outputs.emplace_back(
          onnxsim::dlpack::FromTensorProtoOwning(std::move(tp)));
    }
    return outputs;
  }
};

} // namespace

std::shared_ptr<const ModelExecutor> GetJsModelExecutor() {
  static std::shared_ptr<const ModelExecutor> executor =
      std::make_shared<JsModelExecutor>();
  return executor;
}

#endif // __EMSCRIPTEN__ && ONNXSIM_WASM_HOOKABLE_EXECUTOR
