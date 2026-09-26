#include <fstream>
#include <iostream>

#include "model_info.h"
#include "onnx/common/file_utils.h"
#include "onnxsim.h"
#include "onnxsim_option.h"

// In a hookable WASM build (ONNXSIM_WASM_HOOKABLE_EXECUTOR), folding may be
// delegated through GetJsModelExecutor(). In the ORT-web variant this avoids
// linking ONNX Runtime; in the default WASM build the hook falls back to the
// built-in executor when no JavaScript callback is installed.
// main() is not auto-run under Emscripten (INVOKE_RUN=0) but still has to
// compile.
#ifdef ONNXSIM_WASM_HOOKABLE_EXECUTOR
#include "js_model_executor.h"
#endif

int main(int argc, char** argv) {
  // force env initialization to register opset
  InitEnv();
  OnnxsimOption option(argc, argv);
  bool no_opt = option.Get<bool>("no-opt");
  bool no_sim = option.Get<bool>("no-sim");
  bool no_shape_inference = option.Get<bool>("no-shape-inference");
  int target_opset = option.Get<int>("target-opset");
  bool graph_diff = option.Get<bool>("graph-diff");
  auto input_model_filename = option.Get<std::string>("input-model");
  auto output_model_filename = option.Get<std::string>("output-model");

  onnx::ModelProto model;
  onnx::LoadProtoFromPath(input_model_filename, model);

  onnx::ModelProto simplified = Simplify(
#ifdef ONNXSIM_WASM_HOOKABLE_EXECUTOR
      *GetJsModelExecutor(), model,
#else
      *GetBuiltinModelExecutor(), model,
#endif
      no_opt ? std::nullopt : std::make_optional<std::vector<std::string>>({}),
      !no_sim, !no_shape_inference, SIZE_MAX,
      // A target opset of <= 0 means "leave the opset version unchanged".
      target_opset > 0 ? std::make_optional(target_opset) : std::nullopt);

  std::ofstream ofs(output_model_filename,
                    std::ios::out | std::ios::trunc | std::ios::binary);
  if (!simplified.SerializeToOstream(&ofs)) {
    throw std::invalid_argument("save model error");
  }

  // Report the difference between the original and simplified models, matching
  // the Python CLI's "Finish! Here is the difference:" output.
  std::cout << "Finish! Here is the difference:\n";
  std::cout << FormatSimplifyingInfo(model, simplified);
  if (graph_diff) {
    std::cout << FormatGraphDiff(model, simplified);
  }
  return 0;
}
