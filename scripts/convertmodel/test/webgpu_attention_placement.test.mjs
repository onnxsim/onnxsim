// Does ONNX Runtime Web's WebGPU execution provider actually reject an
// Attention node that uses mask_index, the way onnxsim's own
// check_webgpu_attention_support (onnxsim/webgpu_target.py) claims?
//
// This was originally written to check EP *placement* (expecting the flagged
// node to gracefully fall back to a different execution provider, the way
// ORT's docs describe most unsupported-op gaps). Running it against a real
// onnxruntime-web build found that is not what happens here: ONNX Runtime's
// partitioner assigns Attention to WebGPU based on op type alone --
// GetCapability has no way to inspect *which inputs are wired up* -- so a
// mask_index-bearing Attention node is still committed to WebGPU at
// partition time, and only fails once its WebGPU kernel actually runs:
//
//   Error: [WebGPU] Kernel "[Attention] " failed. Error: Mask not supported
//
// This happens even when "wasm" is listed as a fallback provider (verified
// below) -- a fallback provider only catches nodes GetCapability declined
// outright, not ones that were accepted and then failed at
// Compute()-time. So the real, measured consequence of this gap is
// `session.run()` throwing and the WHOLE session failing, not "just this
// node loses acceleration". onnxsim/webgpu_target.py's own docstring and
// message text were corrected to say so once this was found -- if a future
// onnxruntime-web makes this pass without throwing, that is worth noticing:
// update the docstring/message there and the assertions here together, not
// just one side.
//
// The control (`clean`, no mask) proves Attention runs fine on WebGPU here
// at all; without it, "the mask case failed" wouldn't distinguish "this
// specific gap" from "Attention doesn't work here full stop".
//
// Requires the "playwright" package and a Chromium binary -- see
// package.json's test:webgpu-attention-placement comment and
// .github/workflows/convertmodel-ep-placement.yml, the only place this runs.
//
// Usage:
//   npx playwright install chromium   # once
//   python3 make_ep_placement_fixtures.py   # regenerate fixtures if needed
//   node test/webgpu_attention_placement.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const MANIFEST = JSON.parse(readFileSync(join(HERE, "ep_placement_fixtures.json"), "utf8"));
const { flaggedOp, cases } = MANIFEST.webgpuAttention;

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// Same minimal static file server as the other browser tests here.
function serveConvertmodelDir() {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    if (reqPath === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>webgpu attention placement</title>");
      return;
    }
    const filePath = join(ROOT, reqPath);
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end("not found: " + reqPath);
        return;
      }
      const type = {
        ".html": "text/html", ".mjs": "text/javascript", ".js": "text/javascript",
        ".json": "application/json", ".onnx": "application/octet-stream",
        ".wasm": "application/wasm",
      }[path.extname(filePath)] || "application/octet-stream";
      res.writeHead(200, { "Content-Type": type });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, () => resolve(server));
  });
}

// Runs inside the real browser page via page.evaluate -- no access to this
// file's own scope, only what's passed in and what it can import()/fetch().
// Returns { ok: true } or { ok: false, error } instead of throwing, so a
// case that's *expected* to fail (the mask fixture) doesn't need Node-side
// exception plumbing to observe.
async function runCaseInPage({ port, file, inputs }) {
  const base = `http://localhost:${port}`;
  try {
    const ortMod = await import(`${base}/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs`);
    const ort = ortMod.default ?? ortMod;
    ort.env.wasm.wasmPaths = `${base}/node_modules/onnxruntime-web/dist/`;

    const modelBytes = await fetch(`${base}/test/${file}`).then((r) => r.arrayBuffer());
    const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
      // "wasm" listed as a fallback provider deliberately, matching how a
      // real caller would configure this -- see this file's own top comment
      // on why that does NOT save the mask case.
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "disabled",
    });

    const feeds = {};
    for (const [name, spec] of Object.entries(inputs)) {
      const Ctor = spec.dtype === "int32" ? Int32Array : Float32Array;
      const size = spec.dims.reduce((a, b) => a * b, 1);
      feeds[name] = new ort.Tensor(spec.dtype, new Ctor(size), spec.dims);
    }
    await session.run(feeds);
    await session.release?.();
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e && e.message ? e.message : String(e) };
  }
}

async function runCase(browser, port, caseSpec) {
  const page = await browser.newPage();
  try {
    // A real page, not about:blank -- WebGPU's adapter needs one.
    await page.goto(`http://localhost:${port}/`);
    return await page.evaluate(runCaseInPage, { port, file: caseSpec.file, inputs: caseSpec.inputs });
  } finally {
    await page.close();
  }
}

async function main() {
  console.log(`WebGPU Attention EP-placement check -- flagged op: ${flaggedOp}\n`);

  const server = await serveConvertmodelDir();
  const port = server.address().port;
  const browser = await chromium.launch({
    headless: true,
    // Verified sufficient on its own in this repo's own dev sandbox (via
    // SwiftShader's software Vulkan path); see webgpu_hf_demo.test.mjs's own
    // comment. A real GPU, as the macOS CI runner this is meant for has,
    // needs nothing more.
    args: ["--enable-unsafe-webgpu"],
  });

  let cleanResult, maskResult;
  try {
    cleanResult = await runCase(browser, port, cases.clean);
    maskResult = await runCase(browser, port, cases.mask);
  } finally {
    await browser.close();
    server.close();
  }

  await check(`control: ${flaggedOp} (no mask) runs on WebGPU here`, () => {
    assert.ok(
      cleanResult.ok,
      `${flaggedOp} failed even without a mask -- the control itself failed, so no conclusion can ` +
        `be drawn about the mask case. Error: ${cleanResult.error}`,
    );
  });

  await check(`${flaggedOp} with mask_index fails on WebGPU (even with wasm listed as fallback)`, () => {
    assert.ok(
      !maskResult.ok,
      `${flaggedOp} with mask_index ran successfully on WebGPU -- either onnxruntime-web has gained ` +
        `mask support (great news -- update onnxsim/webgpu_target.py's docstring/message and this ` +
        `test's comment) or this measurement is wrong.`,
    );
    console.log(`    (failed as expected: ${maskResult.error})`);
  });

  console.log(`\nwebgpu attention placement: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
