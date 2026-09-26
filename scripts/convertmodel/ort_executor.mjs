// The JavaScript half of the onnxruntime-web constant-folding executor.
//
// When onnxsim's WASM module is built with ONNXSIM_WASM_ORT_WEB, its C++
// constant folder does not link ONNX Runtime; instead JsModelExecutor::Run
// (js_model_executor.cpp) calls a runner registered on the Emscripten Module as
// `Module.onnxsimModelExecutorRun` (with the legacy
// `Module.onnxsimOrtWebRun` alias). This file builds that runner on top of an
// already-loaded `onnxruntime-web` module.
//
// Contract (must match js_model_executor.cpp) -- a *batched* crossing: rather
// than one JS object + Uint8Array per tensor (which costs an embind round trip
// per field, per tensor), all of a fold group's tensors cross in a single
// concatenated byte blob plus one flat metadata array, in both directions:
//
//   runner(modelBytes: Uint8Array,
//          inputsData: Uint8Array,     // all input tensors' bytes, concatenated
//          inputsMeta: Float64Array)   // [dtype, ndim, dims...] per input
//     => Promise<{ data: Uint8Array,   // all output tensors' bytes, concatenated
//                  meta: Float64Array }>// [dtype, ndim, dims...] per output
//
// `dtype` is the ONNX TensorProto.DataType enum value; tensor bytes are raw
// little-endian element data. Tensors are positional: input i binds to
// session.inputNames[i] and outputs are emitted in session.outputNames order
// (both equal the sub-model's graph input/output order, which is how the C++
// side maps them). No tensor names cross the boundary.

// ONNX TensorProto.DataType enum value -> [onnxruntime-web tensor type string,
// TypedArray constructor for that type]. Kept in lock-step with the dtype set
// js_model_executor.cpp emits/accepts (onnxsim::dlpack's supported types).
const ONNX_DTYPE_TO_ORT = {
  1: ["float32", Float32Array], // FLOAT
  2: ["uint8", Uint8Array], // UINT8
  3: ["int8", Int8Array], // INT8
  4: ["uint16", Uint16Array], // UINT16
  5: ["int16", Int16Array], // INT16
  6: ["int32", Int32Array], // INT32
  7: ["int64", BigInt64Array], // INT64
  9: ["bool", Uint8Array], // BOOL (onnxruntime-web bool data is a Uint8Array)
  11: ["float64", Float64Array], // DOUBLE
  13: ["uint64", BigUint64Array], // UINT64
};

// onnxruntime-web tensor type string -> ONNX TensorProto.DataType enum value.
const ORT_TYPE_TO_ONNX = {
  float32: 1,
  uint8: 2,
  int8: 3,
  uint16: 4,
  int16: 5,
  int32: 6,
  int64: 7,
  bool: 9,
  float64: 11,
  uint64: 13,
};

// Reinterpret a Uint8Array's bytes as `Ctor` elements. The bytes come from a
// slice of the concatenated input blob, so the byte offset may not be aligned
// to the element size; fall back to a copy (onto a fresh 0-offset buffer) when
// it isn't.
function typedFromBytes(Ctor, u8) {
  const bpe = Ctor.BYTES_PER_ELEMENT;
  if (u8.byteOffset % bpe === 0 && u8.byteLength % bpe === 0) {
    return new Ctor(u8.buffer, u8.byteOffset, u8.byteLength / bpe);
  }
  const copy = u8.slice(); // fresh, 0-offset ArrayBuffer
  return new Ctor(copy.buffer, 0, copy.byteLength / bpe);
}

// Raw little-endian bytes backing a TypedArray, without copying.
function bytesFromTyped(ta) {
  return new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
}

// Elements in a tensor of shape `dims` (1 for a scalar / empty dims).
function elementCount(dims) {
  let n = 1;
  for (const d of dims) n *= d;
  return n;
}

// Build the runner that JsModelExecutor calls. `ort` is a loaded onnxruntime-web
// module. Sessions are created per fold group with graph optimization disabled,
// mirroring the built-in CppModelExecutor (ORT_DISABLE_ALL): we want ORT to
// execute the sub-model as-is, not rewrite it.
//
// `providers` lets a browser prefer e.g. WebGPU and fall back to wasm; folding
// correctness does not depend on the provider.
export function makeOrtRunner(ort, { providers = ["wasm"] } = {}) {
  return async function onnxsimOrtWebRun(modelBytes, inputsData, inputsMeta) {
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: providers,
      graphOptimizationLevel: "disabled",
    });

    // Unpack the input blob: walk inputsMeta, slicing the next tensor's bytes
    // out of inputsData, and bind each positionally to session.inputNames.
    const inputNames = session.inputNames;
    const feeds = {};
    let m = 0; // index into inputsMeta
    let off = 0; // byte offset into inputsData
    let i = 0; // input index
    while (m < inputsMeta.length) {
      const dataType = inputsMeta[m++];
      const ndim = inputsMeta[m++];
      const dims = [];
      for (let d = 0; d < ndim; d++) dims.push(inputsMeta[m++]);
      const entry = ONNX_DTYPE_TO_ORT[dataType];
      if (!entry) {
        throw new Error(
          `onnxruntime-web executor: unsupported input dtype ${dataType} for '${inputNames[i]}'`,
        );
      }
      const [type, Ctor] = entry;
      const byteLen = elementCount(dims) * Ctor.BYTES_PER_ELEMENT;
      const sub = inputsData.subarray(off, off + byteLen);
      off += byteLen;
      feeds[inputNames[i]] = new ort.Tensor(type, typedFromBytes(Ctor, sub), dims);
      i++;
    }

    const results = await session.run(feeds);

    // Pack outputs in session.outputNames order (== graph.output() order, which
    // is what the C++ side reads positionally) into one blob + one meta array.
    const parts = [];
    const meta = [];
    let totalBytes = 0;
    for (const name of session.outputNames) {
      const t = results[name];
      const onnxType = ORT_TYPE_TO_ONNX[t.type];
      if (onnxType === undefined) {
        throw new Error(
          `onnxruntime-web executor: unsupported output dtype '${t.type}' for '${name}'`,
        );
      }
      const bytes = bytesFromTyped(t.data);
      parts.push(bytes);
      totalBytes += bytes.byteLength;
      meta.push(onnxType, t.dims.length, ...t.dims);
    }
    const data = new Uint8Array(totalBytes);
    let o = 0;
    for (const p of parts) {
      data.set(p, o);
      o += p.byteLength;
    }

    // Free the session's native resources; folding creates one per group.
    if (typeof session.release === "function") {
      await session.release();
    }
    return { data, meta: new Float64Array(meta) };
  };
}
