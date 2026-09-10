// onnxsim.compile_training_loop's compiled step graph, trained end to end on
// a real browser's WebGPU backend -- the "torch.compile-styled training
// loop" onnxsim/compile_training.py adds, verified the same way
// webgpu_hf_demo.test.mjs verifies onnxsim.qat_graph's own step graphs: not
// "attempted and reported" the way step_graph_ep.test.mjs treats webgpu
// under plain Node (which has no navigator.gpu at all), but REQUIRED, via a
// real Chromium page Playwright drives specifically so WebGPU is reachable.
//
// What this proves, and what it does not:
//   * step_train_loop_demo.onnx (test/make_step_graph_fixtures.py's
//     build_train_loop_demo) is exactly what a caller of
//     onnxsim.compile_training_loop gets -- the fixture is built by calling
//     that function directly and reading back its own .step_graph, never a
//     hand-assembled approximation of it. So this is compile_training.py
//     itself running on WebGPU, not a lookalike graph.
//   * Unlike webgpu_hf_demo.test.mjs this fixture's data is baked (a small
//     fixed linear-regression batch -- see build_train_loop_demo's own
//     docstring for why compile_training needs no live network to
//     demonstrate), so this test also replays it against the reference loss
//     trajectory recorded in Python on onnxruntime's CPU provider
//     (step_graph_ep.test.mjs's own comparison, reused here): agreement
//     means "onnxruntime-web on WebGPU agrees with onnxruntime on CPU" is a
//     numeric check, not a claim.
//
// Requires the "playwright" package and a Chromium binary (not installed by
// default here -- see package.json's test:webgpu-train-loop-demo comment and
// .github/workflows/convertmodel-webgpu-demo.yml, the only place this is
// run).
//
// Usage:
//   npx playwright install chromium   # once
//   node test/webgpu_train_loop_demo.test.mjs

import assert from "node:assert/strict";
import http from "node:http";
import fs, { readFileSync } from "node:fs";
import path, { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const MANIFEST = JSON.parse(readFileSync(join(HERE, "step_train_loop_demo.json"), "utf8"));

const NUM_STEPS = MANIFEST.scalars.length;

// Every op onnxsim.compile_training_loop's own build_backward/adam_update
// composition emitted for this fixture (make_step_graph_fixtures.py's
// _op_histogram) -- the set this test requires to have actually run on
// WebGPU (JsExecutionProvider), not just "somewhere". Read from the
// manifest rather than hand-enumerated, so it cannot drift from what the
// fixture actually contains the way a copied literal list could.
const TRAINABLE_OPS = new Set(Object.keys(MANIFEST.ops));

// Same tolerance step_graph_ep.test.mjs uses for the same comparison
// (onnxruntime-web's CPU kernels and onnxruntime's are the same C++ code;
// WebGPU's own numerics reassociate reductions differently, so this leaves
// more room than that file's CPU-vs-CPU 1e-3 while staying far below what a
// wrong kernel would cost).
const RTOL = 1e-2;
const ATOL = 1e-6;

function closeEnough(got, want) {
  return Math.abs(got - want) <= ATOL + RTOL * Math.abs(want);
}

let passed = 0;
async function check(name, fn) {
  await fn();
  passed += 1;
  console.log("  ok -", name);
}

// A minimal static file server over scripts/convertmodel/ -- see
// webgpu_hf_demo.test.mjs's own copy of this function for why: just enough
// for the page's dynamic import()/fetch() calls to resolve relative URLs.
function serveConvertmodelDir() {
  const server = http.createServer((req, res) => {
    const reqPath = decodeURIComponent(req.url.split("?")[0]);
    if (reqPath === "/") {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end("<!doctype html><title>webgpu train loop demo</title>");
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

// Everything below runs inside the real browser page, via page.evaluate --
// it has no access to this file's own scope, only what it's passed and what
// it can import()/fetch() from the server above.
async function runInPage(port) {
  const base = `http://localhost:${port}`;
  const ortMod = await import(`${base}/node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs`);
  const ort = ortMod.default ?? ortMod;
  ort.env.wasm.wasmPaths = `${base}/node_modules/onnxruntime-web/dist/`;

  const manifest = await fetch(`${base}/test/step_train_loop_demo.json`).then((r) => r.json());
  const modelBytes = await fetch(`${base}/test/${manifest.file}`).then((r) => r.arrayBuffer());

  const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
    executionProviders: ["webgpu"],
    // Optimization off, so what gets partitioned is the operator set
    // compile_training_loop actually emitted, not whatever ORT fused it
    // into -- the same reason step_graph_ep.test.mjs/webgpu_hf_demo.mjs
    // disable it.
    graphOptimizationLevel: "disabled",
    logSeverityLevel: 0,
    logVerbosityLevel: 4,
  });

  const tensor = (spec) => new ort.Tensor("float32", Float32Array.from(spec.data), spec.dims);
  const constants = {};
  for (const [name, spec] of Object.entries(manifest.constants)) constants[name] = tensor(spec);
  let state = {};
  for (const [name, spec] of Object.entries(manifest.state)) state[name] = tensor(spec);

  const t0 = performance.now();
  const losses = [];
  for (let t = 0; t < manifest.scalars.length; t++) {
    const feeds = { ...constants, ...state };
    for (const [name, value] of Object.entries(manifest.scalars[t])) {
      feeds[name] = new ort.Tensor("float32", Float32Array.from([value]), []);
    }
    const out = await session.run(feeds);
    const next = {};
    for (const [name, spec] of Object.entries(manifest.state)) next[name] = out[spec.output];
    state = next;
    losses.push(Number(out[manifest.loss].data[0]));
  }
  const trainMs = performance.now() - t0;
  await session.release?.();

  return { losses, trainMs };
}

// Same parser step_graph_ep.test.mjs/webgpu_hf_demo.test.mjs both use,
// duplicated rather than imported (see webgpu_hf_demo.test.mjs's own
// comment on why: no shared module needs it on the Node side of a real
// browser run).
function parsePlacements(lines) {
  const placements = {};
  let current = null;
  for (const raw of lines) {
    const line = raw.replace(/^.*VerifyEachNodeIsAssignedToAnEp\]\s?/, "");
    const header = line.match(/(All nodes|Node\(s\)) placed on \[(\w+)\]\. Number of nodes: (\d+)/);
    if (header) {
      const all = header[1] === "All nodes";
      current = placements[header[2]] || { count: 0, ops: all ? null : new Set() };
      current.count += Number(header[3]);
      if (all) current.ops = null;
      placements[header[2]] = current;
      continue;
    }
    const node = current && current.ops && line.match(/^\s*([A-Za-z][\w.]*)\s*\(/);
    if (node) current.ops.add(node[1]);
  }
  return placements;
}

async function main() {
  console.log(
    `onnxsim.compile_training_loop WebGPU training demo -- ${NUM_STEPS} steps, ` +
      `${Object.keys(MANIFEST.ops).length} distinct ops\n`,
  );

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
  const logLines = [];
  let result;
  try {
    const page = await browser.newPage();
    page.on("console", (msg) => {
      const text = msg.text();
      if (/onnxruntime/i.test(text)) logLines.push(text);
    });
    // A real page, not about:blank -- see webgpu_hf_demo.test.mjs's own
    // comment on why WebGPU's adapter needs one.
    await page.goto(`http://localhost:${port}/`);
    result = await page.evaluate(runInPage, port);
  } finally {
    await browser.close();
    server.close();
  }

  const trace = `loss ${result.losses[0].toExponential(3)} -> ` +
    `${result.losses[result.losses.length - 1].toExponential(3)}`;
  console.log(`  ${NUM_STEPS} steps in ${result.trainMs.toFixed(0)}ms, ${trace}`);

  await check("every loss is finite", () => {
    for (const loss of result.losses) {
      assert.ok(Number.isFinite(loss), `loss ${loss} is not finite`);
    }
  });
  await check("loss decreased meaningfully over the run", () => {
    const first = result.losses[0];
    const last = result.losses[result.losses.length - 1];
    assert.ok(last < 0.5 * first, `loss did not meaningfully decrease: ${first} -> ${last}`);
  });
  await check("WebGPU's loss trajectory matches the Python/CPU reference", () => {
    for (let t = 0; t < NUM_STEPS; t++) {
      const got = result.losses[t];
      const want = MANIFEST.referenceLosses[t];
      assert.ok(
        closeEnough(got, want),
        `step ${t}: WebGPU loss ${got} vs. CPU reference ${want} ` +
          `(rtol ${RTOL}, atol ${ATOL})`,
      );
    }
  });

  const placements = parsePlacements(logLines);
  await check("every op ran on WebGPU (JsExecutionProvider)", () => {
    const gpu = placements.JsExecutionProvider;
    assert.ok(gpu, `no JsExecutionProvider placement reported at all -- captured ${logLines.length} log lines`);
    const gpuOps = gpu.ops === null ? new Set(TRAINABLE_OPS) : gpu.ops;
    const missing = [...TRAINABLE_OPS].filter((op) => !gpuOps.has(op));
    assert.deepEqual(
      missing, [],
      `these ops never showed up under JsExecutionProvider: ${missing.join(", ")} -- ` +
        `full placement report: ${JSON.stringify(placements, (k, v) => (v instanceof Set ? [...v] : v))}`,
    );
    for (const [provider, info] of Object.entries(placements)) {
      if (provider === "JsExecutionProvider") continue;
      const strayed = info.ops && [...info.ops].filter((op) => TRAINABLE_OPS.has(op));
      assert.ok(
        !strayed || strayed.length === 0,
        `${strayed && strayed.join(", ")} ran on ${provider} instead of WebGPU`,
      );
    }
  });

  console.log(`\nwebgpu train loop demo: ${passed} checks passed`);
}

main().catch((e) => {
  console.error("FAIL:", e.stack ?? String(e));
  process.exit(1);
});
