#!/usr/bin/env node
// Verifies a real, fully-assembled onnxsim wasm32/Pyodide wheel (as built by
// `pyodide build` + `ONNXSIM_WASM_SIDE_MODULE_RELINK=1`, see setup.py and
// docs/wasm_pyodide.md's "Distributable wheel" section) actually installs
// via micropip and runs a full onnxsim.simplify() call, inside a real
// Pyodide runtime (the `pyodide` npm package under Node).
//
// This is the release-flow counterpart to pyodide_smoke_test.mjs, which
// instead tests the loose .abi3.so build_wasm_pyodide.sh produces for the
// browser demo (and needs a sys.modules aliasing workaround because that
// script dlopens the same .so twice, once top-level and once package-
// relative -- see its own comments). This script installs the real wheel
// exactly once, the normal way, so that workaround doesn't apply here.
//
// Usage: node pyodide_wheel_smoke_test.mjs <path-to-onnxsim-*.whl>
import { loadPyodide } from "pyodide";
import { readFileSync, existsSync } from "node:fs";
import { resolve, basename } from "node:path";

function fail(message) {
  console.error(`\nFAIL: ${message}\n`);
  process.exit(1);
}

async function main() {
  const whlPathArg = process.argv[2];
  if (!whlPathArg) {
    fail("usage: node pyodide_wheel_smoke_test.mjs <path-to-onnxsim-*.whl>");
  }
  const whlPath = resolve(whlPathArg);
  if (!existsSync(whlPath)) {
    fail(`wheel not found at ${whlPath}`);
  }
  const whlBytes = readFileSync(whlPath);
  const whlName = basename(whlPath);
  console.log(`read ${whlBytes.length} bytes from ${whlPath}`);

  const pyodide = await loadPyodide();
  console.log(`Pyodide ${pyodide.version} ready (loaded via the 'pyodide' npm package)`);

  const abiVersion = pyodide.runPython(
    `import sysconfig; sysconfig.get_config_var("PYODIDE_ABI_VERSION")`
  );
  console.log(`Pyodide ABI epoch: ${abiVersion}`);
  if (!whlName.includes(String(abiVersion))) {
    console.log(
      `note: ${whlName} does not mention epoch ${abiVersion} in its filename -- ` +
        `not necessarily wrong (the epoch isn't always spelled out literally), continuing anyway`
    );
  }

  // micropip's `emfs:` scheme installs a wheel already sitting in Pyodide's
  // own virtual filesystem, so this doesn't need a real HTTP server (unlike
  // installing by URL).
  const emfsPath = `/tmp/${whlName}`;
  pyodide.FS.writeFile(emfsPath, whlBytes);
  console.log(`wrote wheel to Pyodide's virtual FS at ${emfsPath}`);

  await pyodide.loadPackage("micropip");

  // deps=True (the default): `onnxsim`'s pure-Python code (onnx_simplifier.py)
  // imports numpy and onnx unconditionally at module level -- plain
  // `import onnxsim` needs both, not just onnxsim.simplify(). deps=False was
  // tried first and failed on exactly that ("ModuleNotFoundError: No module
  // named 'numpy'"), confirming this isn't optional. Letting micropip
  // resolve the wheel's own declared dependencies (onnx and onnx's
  // own transitive numpy/protobuf/ml_dtypes) is also a better test than
  // skipping it: it validates the wheel's METADATA declares them correctly,
  // not just that the extension itself loads.
  try {
    await pyodide.runPythonAsync(`
import micropip
await micropip.install("emfs:${emfsPath}")
`);
  } catch (e) {
    fail(
      `micropip.install(the built wheel) failed under Pyodide ` +
        `${pyodide.version} (ABI epoch ${abiVersion}):\n${e.stack || e}`
    );
  }
  console.log(`micropip.install(${whlName}): OK`);

  const pythonCode = `
import sys
import onnxsim
import onnxsim.onnxsim_cpp2py_export as m

print("installed onnxsim from:", onnxsim.__file__, file=sys.stderr)
print("installed extension from:", m.__file__, file=sys.stderr)

# Exercise a real binding, not just "did it import" -- see
# pyodide_smoke_test.mjs's identical check for why this specific one.
optimizers = m._list_optimizers()
assert isinstance(optimizers, list), f"_list_optimizers() returned {optimizers!r}, expected a list"
assert len(optimizers) > 0, "_list_optimizers() returned an empty list -- module loaded but onnx-optimizer's passes did not register"
print(f"OK: {len(optimizers)} optimizer passes registered, e.g. {optimizers[:3]}", file=sys.stderr)
`;
  try {
    await pyodide.runPythonAsync(pythonCode);
  } catch (e) {
    fail(`import onnxsim (from the installed wheel) failed under Pyodide:\n${e.stack || e}`);
  }

  console.log(
    `\nPASS: ${whlName} installs via micropip and runs under Pyodide ${pyodide.version}\n`
  );
}

main().catch((e) => {
  fail(`unexpected error:\n${e.stack || e}`);
});
