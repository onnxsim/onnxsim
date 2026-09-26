// Smoke-test a standalone Node/NODERAWFS Emscripten module.
//
// Usage:
//   node scripts/wasm_node_smoke.mjs build-wasm-node-ON/onnxsim.js
//
// This deliberately checks only module loading and the exported version
// binding. Constant folding through ORT is covered by the ORT-web package
// tests; keeping this test dependency-free makes it useful on small hosts.

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const modulePath = process.argv[2];
if (!modulePath) {
  console.error("usage: node scripts/wasm_node_smoke.mjs MODULE.js");
  process.exit(2);
}

const imported = await import(pathToFileURL(modulePath));
const factory = imported.default ?? imported;
if (typeof factory !== "function") {
  throw new TypeError("Emscripten module does not export a factory function");
}
const runtime = await factory({ noInitialRun: true });
if (typeof runtime.onnxsim_versions !== "function") {
  throw new Error("onnxsim_versions binding is missing");
}
const versions = runtime.onnxsim_versions();
if (!versions || typeof versions.onnxsim !== "string") {
  throw new Error("onnxsim_versions returned an invalid result");
}
console.log(`WASM Node smoke passed (${versions.onnxsim})`);
