# onnxsim

[onnxsim](https://github.com/onnxsim/onnxsim) compiled to WebAssembly, packaged for
Node.js. Constant folding is delegated to
[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) rather than
being compiled into the module, so installing this package does **not** pull
in a second copy of ONNX Runtime (see
[`docs/wasm_ort_web.md`](https://github.com/onnxsim/onnxsim/blob/master/docs/wasm_ort_web.md)
in the main repository for how the bridge works).

> **Status:** experimental. The underlying ORT-web WASM variant is new; please
> [file an issue](https://github.com/onnxsim/onnxsim/issues) if a model
> simplifies differently here than with the
> [`onnxsim` Python package](https://pypi.org/project/onnxsim/) or the CLI.

## Install

```sh
npm install onnxsim
```

## Usage

```js
import { readFile, writeFile } from "node:fs/promises";
import { simplify } from "onnxsim";

const input = await readFile("model.onnx");
const { model } = await simplify(input);
await writeFile("model.simplified.onnx", model);
```

`simplify` accepts an options object mirroring the CLI's flags:

```js
const { model, trace } = await simplify(input, {
  skipOptimizers: [],        // onnx-optimizer pass names to skip
  constantFolding: true,
  shapeInference: true,
  tensorSizeThreshold: 1.5 * 1024 ** 3, // bytes; skip folding larger constants
  targetOpsetVersion: -1,    // <= 0 keeps the model's opset
  profile: false,            // also return a Chrome trace JSON
  annotateModelInfo: false,  // bake MAC/FLOP counts into metadata_props
  graphDiff: false,          // console.log a node/value-level before/after diff
});
```

Constant-folding subgraphs can be sent to a custom runner (for example a
remote accelerator) with `setModelExecutorRunner`:

```js
import { setModelExecutorRunner } from "onnxsim";

await setModelExecutorRunner((modelBytes, inputsData, inputsMeta) =>
  remoteRunner(modelBytes, inputsData, inputsMeta));
```

## Why not the native Python/C++ build?

The Python package (`pip install onnxsim`) and the Rust crate use a native
binary and are faster for local/CI use. This package exists for JavaScript
tooling — bundlers, browser demos, CLIs written in Node — that want ONNX
simplification without a native build step or a Python runtime.

## License

Apache-2.0, same as the [onnxsim](https://github.com/onnxsim/onnxsim) project.
