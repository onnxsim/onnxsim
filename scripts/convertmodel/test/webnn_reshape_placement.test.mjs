// Does ONNX Runtime Web's WebNN execution provider actually reject a
// Reshape node with a non-constant shape input, the way onnxsim's own
// check_webnn_support (onnxsim/webnn_target.py) claims -- citing
// onnxruntime-web's own docs (js/web/docs/webnn-operators.md: "Input 'shape'
// should be a constant"), not a measurement?
//
// Modeled after webgpu_attention_placement.test.mjs's own finding for the
// Attention/mask gap: rather than assume a graceful EP fallback (this file
// originally checked EP *placement* logs before that assumption was found
// wrong for WebGPU), each case's session.run() success/failure is checked
// directly -- if WebNN behaves the same way WebGPU did (GetCapability
// accepting the node by op type alone, then the kernel failing once it
// actually runs and discovers the shape isn't constant), that shows up as
// the dynamic-shape case's session.run() rejecting, not as a placement
// difference.
//
// Unlike webgpu_attention_placement.test.mjs, this is deliberately ATTEMPTED
// AND REPORTED, never required to pass CI -- docs/webnn.md is explicit that
// WebNN's browser support is still experimental and "broadest on Windows;
// GPU and NPU paths on other platforms are still maturing", and nothing else
// in this repo has ever exercised a real WebNN backend in a browser
// (webnn.test.mjs only unit-tests the pure helpers against a mocked
// navigator.ml under plain Node). A macOS GitHub-hosted runner's Chromium
// may simply not have a working WebNN backend at all, or may hit an
// unrelated backend bug on even the control fixture (both observed while
// developing this against this repo's own dev sandbox, on different
// hardware) -- either is a fact about the runner/backend, not a regression
// in onnxsim, so this only fails outright when the *dynamic* case fails
// where the *clean* control succeeded. Set ORT_REQUIRE_WEBNN=1 to make every
// skip below a hard failure once/if this is confirmed reliable in CI.
//
// Requires the "playwright" package and a Chromium binary -- see
// package.json's test:webnn-reshape-placement comment and
// .github/workflows/convertmodel-ep-placement.yml, the only place this runs.
//
// Usage:
//   npx playwright install chromium   # once
//   python3 make_ep_placement_fixtures.py   # regenerate fixtures if needed
//   node test/webnn_reshape_placement.test.mjs
//   ORT_REQUIRE_WEBNN=1 node test/webnn_reshape_placement.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const MANIFEST = JSON.parse(readFileSync(join(HERE, "ep_placement_fixtures.json"), "utf8"));
const { flaggedOp, cases } = MANIFEST.webnnReshape;
const REQUIRE_WEBNN = !!process.env.ORT_REQUIRE_WEBNN;

let passed = 0;
let skipped = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}
function skip(name, reason) {
  if (REQUIRE_WEBNN) {
    throw new Error(`${name}: ${reason} (failing because ORT_REQUIRE_WEBNN=1)`);
  }
  skipped += 1;
  console.log("  skip -", name, "-", reason);
}

// Same minimal static file server as the other browser tests here.
function serveConvertmodelDir() {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    if (reqPath === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>webnn reshape placement</title>");
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

// Runs inside the real browser page via page.evaluate. Probes every WebNN
// device type (gpu, npu, cpu -- same order webnn.mjs's WEBNN_DEVICE_TYPES
// uses) and returns the first that creates an MLContext, or null.
async function detectWebnnDeviceInPage(port) {
  const base = `http://localhost:${port}`;
  const { detectWebnn, WEBNN_DEVICE_TYPES } = await import(`${base}/webnn.mjs`);
  const report = await detectWebnn();
  for (const deviceType of WEBNN_DEVICE_TYPES) {
    if (report.devices[deviceType]) return { deviceType, report };
  }
  return { deviceType: null, report };
}

// Runs inside the real browser page via page.evaluate. Returns
// { ok: true } or { ok: false, error } instead of throwing -- see
// webgpu_attention_placement.test.mjs's own comment on why.
async function runCaseInPage({ port, file, inputs, deviceType }) {
  const base = `http://localhost:${port}`;
  try {
    const ortMod = await import(`${base}/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs`);
    const ort = ortMod.default ?? ortMod;
    ort.env.wasm.wasmPaths = `${base}/node_modules/onnxruntime-web/dist/`;
    const { webnnProvider } = await import(`${base}/webnn.mjs`);

    const modelBytes = await fetch(`${base}/test/${file}`).then((r) => r.arrayBuffer());
    const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
      executionProviders: [webnnProvider(deviceType), "wasm"],
      graphOptimizationLevel: "disabled",
    });

    const feeds = {};
    for (const [name, spec] of Object.entries(inputs)) {
      const size = spec.dims.reduce((a, b) => a * b, 1);
      feeds[name] = new ort.Tensor(spec.dtype, new Float32Array(size), spec.dims);
    }
    await session.run(feeds);
    await session.release?.();
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e && e.message ? e.message : String(e) };
  }
}

async function runCase(browser, port, caseSpec, deviceType) {
  const page = await browser.newPage();
  try {
    await page.goto(`http://localhost:${port}/`);
    return await page.evaluate(runCaseInPage, {
      port,
      file: caseSpec.file,
      inputs: caseSpec.inputs,
      deviceType,
    });
  } finally {
    await page.close();
  }
}

async function main() {
  console.log(`WebNN Reshape EP-placement check (best-effort) -- flagged op: ${flaggedOp}\n`);

  const server = await serveConvertmodelDir();
  const port = server.address().port;
  const browser = await chromium.launch({
    headless: true,
    args: ["--enable-features=WebMachineLearningNeuralNetwork"],
  });

  try {
    const probePage = await browser.newPage();
    // A real page, not about:blank -- navigator.ml (like WebGPU's adapter)
    // needs one, and this page's own dynamic import() of webnn.mjs resolves
    // relative to it.
    await probePage.goto(`http://localhost:${port}/`);
    const { deviceType, report } = await probePage.evaluate(detectWebnnDeviceInPage, port);
    await probePage.close();

    if (!deviceType) {
      skip(
        "WebNN device availability",
        `no WebNN device could be created in this browser (${report.message})`,
      );
      console.log(`\nwebnn reshape placement: ${passed} checks passed, ${skipped} skipped`);
      return;
    }
    console.log(`  using WebNN device type: ${deviceType}`);

    const cleanResult = await runCase(browser, port, cases.clean, deviceType);
    const dynamicResult = await runCase(browser, port, cases.dynamic, deviceType);

    if (!cleanResult.ok) {
      skip(
        `control: ${flaggedOp} (constant shape) runs on WebNN here`,
        `the control itself failed on this WebNN backend (${cleanResult.error}) -- unrelated to the ` +
          "gap under test, so no conclusion can be drawn about the dynamic-shape case",
      );
    } else {
      passed += 1;
      console.log(`  ok - control: ${flaggedOp} (constant shape) runs on WebNN here`);
      await check(`${flaggedOp} with a non-constant shape fails on WebNN`, () => {
        assert.ok(
          !dynamicResult.ok,
          `${flaggedOp} with a non-constant shape ran successfully on WebNN -- either onnxruntime-web ` +
            "has gained this support (update onnxsim/webnn_target.py's docstring/message and this " +
            "test's comment) or this measurement is wrong.",
        );
        console.log(`    (failed as expected: ${dynamicResult.error})`);
      });
    }
  } finally {
    await browser.close();
    server.close();
  }

  console.log(`\nwebnn reshape placement: ${passed} checks passed, ${skipped} skipped`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
