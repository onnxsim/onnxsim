# Notes for Claude

## The Python wheel build does NOT build ONNX Runtime

Don't get confused by `ONNXSIM_BUILTIN_ORT` when working on the wheel build.

- `CMakeLists.txt` defaults `ONNXSIM_BUILTIN_ORT` to **ON**, which builds ONNX Runtime
  (from `third_party/onnxruntime-1.29.0`, via `cmake/build_ort.cmake`, fully out-of-tree
  -- see that file) and makes it available as a constant-folding backend
  (`GetBuiltinModelExecutor()`). That default is for the standalone C++/WASM builds,
  **not** the Python wheel.
- `setup.py` explicitly passes **`-DONNXSIM_BUILTIN_ORT=OFF`** when building the wheel, so
  ONNX Runtime is **never compiled** as part of `pip install` / wheel builds. The
  extension is compiled without `ONNXSIM_HAS_ORT`, which `#ifdef`s out all the `Ort::`
  C++ code (`GetBuiltinModelExecutor()`, `dlpack_bridge.h`'s `Ort::Value` glue).
- onnxsim's own optimizer/shape-inference pipeline (its `onnx`/`onnx-optimizer` fork
  under `third_party/onnx`) is **always** built, regardless of `ONNXSIM_BUILTIN_ORT` --
  it is not part of what that flag controls. When `ONNXSIM_BUILTIN_ORT=ON`, ONNX Runtime
  is built/linked as a fully separate, out-of-tree artifact specifically so its own
  (differently-versioned, vendored) onnx copy never enters onnxsim's own CMake target
  graph -- onnx's own `CMakeLists.txt` hardcodes its target names, so two onnx copies
  cannot coexist in one CMake project.
- At runtime, `onnxruntime` is only an **optional** Python dependency
  (`[project.optional-dependencies]` in `pyproject.toml`). onnxsim uses the pip-installed
  `onnxruntime` package for constant folding / correctness checking when present, and
  falls back to onnx's reference evaluator when it isn't.

So: **building ONNX Runtime is not required to build, test, or ship the Python wheel.**
If you see long ONNX Runtime C++ compilation, that's the `ONNXSIM_BUILTIN_ORT=ON` path
(standalone C++/WASM), not the wheel path.

## Prefer `onnx.parser`-based model construction in tests

When writing new tests or touching an existing test file's model-building code, build
`ModelProto`/`GraphProto`/`FunctionProto` via `onnx.parser.parse_model()` /
`parse_function()` (the ONNX text format) instead of chains of
`onnx.helper.make_node`/`make_graph`/`make_model`. The text form reads like the graph it
describes, so a test's structure is visible at a glance instead of buried in positional
`make_node(...)` argument lists.

- A common pattern across the migrated test files: a small
  `_model(body, initializer=(), opset=..., ir_version=...)` helper that wraps
  `parser.parse_model(f"<ir_version: ..., opset_import: [...]> {body}")` and then does
  `model.graph.initializer.extend(initializer)`.
- Keep random/large weight arrays as numpy-built `onnx.numpy_helper.from_array`
  initializers attached programmatically after parsing -- don't try to spell out large
  tensors as text literals. Small, fixed/deterministic constants are fine as text
  literals (`<float C = {2.0}>`).
- Helpers that build a reusable node sequence should return a text fragment (a string)
  rather than a list of node protos, so callers can concatenate/interpolate them the same
  way as the rest of the body.
- It's fine to fall back to `onnx.helper` for the rare case where the text format can't
  express the exact semantics under test -- e.g. a value info with no shape field at all
  ("rank not statically known", `ClearField("shape")` after a placeholder parse), or a
  tensor that must be byte-equal to a `numpy_helper.from_array`-produced one (the parser
  encodes tensor literals as `float_data`, not `raw_data`). When you do this, leave a
  short comment saying why the text form doesn't fit.

See `tests/test_python_api.py`, `tests/test_fusion_patterns.py`, `tests/test_pruning.py`,
and the other already-migrated `tests/test_*.py` files for the established pattern.
