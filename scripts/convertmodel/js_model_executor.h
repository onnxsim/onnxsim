// A ModelExecutor that runs constant-folding sub-models with onnxruntime-web
// (the JavaScript/WASM ONNX Runtime) instead of a statically linked ONNX
// Runtime C++ library.
//
// This is the WebAssembly analogue of the Python "trampoline" executor
// (PyModelExecutor in onnxsim/cpp2py_export.cc): the C++ constant folder keeps
// calling ModelExecutor::Run, but the actual op execution is delegated across
// the language boundary -- to the pip `onnxruntime` package in Python, and to
// the page's `onnxruntime-web` module here. Tensors cross the executor boundary
// as DLPack DLManagedTensors (see onnxsim.h); this executor converts them
// to/from onnxruntime-web's dtype+dims+raw-bytes contract.
//
// Why: the default WASM build compiles ONNX Runtime from source and links it
// into onnxsim's own .wasm (ONNXSIM_BUILTIN_ORT=ON). That from-source ORT
// compile is by far the slowest part of the WASM build and bloats the module
// with a second copy of ORT. When the hosting page already loads
// onnxruntime-web (the converter page does, for its inference panel), routing
// constant folding through it lets the onnxsim WASM build drop ORT entirely
// (built without ONNXSIM_HAS_ORT), shrinking the module and removing the ORT
// compile from the build.
//
// This path is compiled for Emscripten by default. The host may install a
// custom ModelExecutor callback (remote, WebGPU, worker, or onnxruntime-web);
// when no callback is installed, the implementation falls back to built-in
// ORT if that backend is present. The synchronous-C++/asynchronous-JS bridge
// requires the module to be linked with Asyncify.
#pragma once

#if defined(__EMSCRIPTEN__) && defined(ONNXSIM_WASM_HOOKABLE_EXECUTOR)

#include <memory>

#include "onnxsim.h" // ModelExecutor

// Returns the singleton onnxruntime-web-backed executor. Its Run reaches into
// JavaScript for the actual session run, so it must be called from a context
// linked with Asyncify (as onnxsimplify_export is in the ORT-web build).
std::shared_ptr<const ModelExecutor> GetJsModelExecutor();

#endif // __EMSCRIPTEN__ && ONNXSIM_WASM_HOOKABLE_EXECUTOR
